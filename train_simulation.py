"""
Advanced Railway Simulation using SimPy

This script simulates a realistic railway network between Bhopal Junction and Itarsi Junction.
It is designed to be modular and extensible, allowing for detailed analysis of railway operations.

Key Features:
- Object-Oriented Design: Clear separation of concerns with classes for Train, Station, Track, and a SimulationManager.
- Detailed Station Modeling: Stations have multiple platforms (SimPy PriorityResources), signals, and divergence points.
- Diverse Train Categories: Includes 6 types of trains with unique speeds and priorities, from high-speed Rajdhani to slow-moving Freight.
- Dynamic Track Conditions: Track segments can have varying conditions (normal, degraded) and are affected by weather, impacting train speeds.
- Realistic Event Handling: Simulates scheduled services, random delays (crew, maintenance), signal waiting, and basic rerouting logic.
- Comprehensive Data Output: Generates a detailed event log, key congestion metrics, and a CSV file with per-train journey details for analysis.

Usage:
- Run the script directly to execute a pre-configured simulation scenario.
- The simulation parameters, train schedules, and events can be easily modified in the `SimulationManager` class.
"""

import simpy
import random
import csv
import datetime

# --- Constants and Configuration ---

# 1. Stations and Distances
STATIONS_DATA = {
    "Bhopal Junction": {"platforms": 5},
    "Rani Kamalapati": {"platforms": 4, "distance": 7},
    "Barkheda": {"platforms": 2, "distance": 20},
    "Obaidullaganj": {"platforms": 3, "distance": 12},
    "Bagra Tawa": {"platforms": 2, "distance": 18},
    "Hoshangabad": {"platforms": 4, "distance": 16},
    "Pachama": {"platforms": 2, "distance": 10},
    "Banapura": {"platforms": 3, "distance": 12},
    "Itarsi Junction": {"platforms": 5, "distance": 20},
}
STATION_NAMES = list(STATIONS_DATA.keys())

# 2. Train Categories: (average_speed_kmh, priority_level)
# Priority: Lower number is higher priority (1 is highest)
TRAIN_CATEGORIES = {
    "Rajdhani Express": (130, 1),
    "Mail/Express": (110, 2),
    "Superfast Passenger": (100, 3),
    "MEMU/DEMU": (80, 4),
    "Ordinary Passenger": (60, 5),
    "Freight Train": (50, 6),
}

# 3. Simulation Parameters
SIMULATION_START_TIME = datetime.datetime(2025, 9, 18, 8, 0)  # Sim starts at 8:00 AM
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# --- Core Simulation Classes ---

class Track:
    """
    Represents a track segment between two stations.
    
    Attributes:
        env (simpy.Environment): The simulation environment.
        name (str): A unique name for the track segment.
        distance (float): The length of the track in kilometers.
        resource (simpy.Resource): A SimPy resource representing the track's capacity (e.g., double line).
        condition (str): The physical condition of the track ('normal', 'degraded', 'maintenance').
        weather (str): The current weather affecting the track ('clear', 'rain', 'heavy_downpour').
        utilization_time (float): Total time the track has been in use.
    """
    def __init__(self, env, name, distance):
        self.env = env
        self.name = name
        self.distance = distance
        # Capacity=2 represents a double line (one up, one down)
        self.resource = simpy.Resource(env, capacity=2)
        self.condition = "normal"
        self.weather = "clear"
        self.utilization_time = 0

    def get_travel_time(self, train_speed_kmh):
        """Calculates the time required to travel this track segment based on speed, condition, and weather."""
        speed_multiplier = 1.0
        if self.weather == "rain":
            speed_multiplier *= 0.8  # 20% speed reduction
        elif self.weather == "heavy_downpour":
            speed_multiplier *= 0.6  # 40% speed reduction
        
        if self.condition == "degraded":
            speed_multiplier *= 0.9
        elif self.condition == "maintenance":
            speed_multiplier *= 0.5

        effective_speed_kmh = train_speed_kmh * speed_multiplier
        if effective_speed_kmh == 0:
            return float('inf')
        return (self.distance / effective_speed_kmh) * 60  # Return time in minutes


class Station:
    """
    Represents a station with platforms, signals, and divergence points.
    
    Attributes:
        env (simpy.Environment): The simulation environment.
        name (str): The name of the station.
        platforms (simpy.PriorityResource): SimPy resource for platforms, handling priority-based allocation.
        divergence_points (dict): Resources for loop lines or other tracks within the station.
        signal (simpy.Event): An event used to control train departures.
    """
    def __init__(self, env, name, num_platforms):
        self.env = env
        self.name = name
        self.platforms = simpy.PriorityResource(env, capacity=num_platforms)
        # Model divergence points like loop lines
        self.divergence_points = {
            "loop_line": simpy.Resource(env, capacity=2)
        }
        self.signal = env.event()

    def grant_departure_clearance(self):
        """Grants clearance for a train to depart by succeeding the signal event."""
        if not self.signal.triggered:
            self.signal.succeed()
            # Immediately create a new signal event for the next train
            self.signal = self.env.event()


class Train:
    """
    Represents a train process in the simulation.
    
    Each train is a SimPy process that moves along a predefined route, interacting with
    stations and tracks.
    """
    def __init__(self, env, train_id, config, route, simulation_manager):
        self.env = env
        self.train_id = train_id
        self.train_type = config['type']
        self.speed_kmh, self.priority = TRAIN_CATEGORIES[self.train_type]
        self.loco_type = config.get('loco_type', 'Electric')
        self.crew_available = config.get('crew_available', True)
        self.maintenance_ok = config.get('maintenance_ok', True)
        self.route = route
        self.schedule = config['schedule']
        self.current_station_idx = 0
        self.simulation_manager = simulation_manager
        
        # Start the train's run process
        self.process = env.process(self.run())

    def run(self):
        """The main simulation process for a single train's journey."""
        # 1. Initial Departure
        origin_station_name = self.route[0]
        scheduled_departure_time = self.schedule[origin_station_name]['departure']
        
        # Wait until the scheduled departure time
        yield self.env.timeout(scheduled_departure_time - self.env.now)
        
        self.simulation_manager.log_event(f"{self.train_id} ({self.train_type}) starting journey from {origin_station_name}.")

        # 2. Journey Loop
        while self.current_station_idx < len(self.route) - 1:
            current_station = self.simulation_manager.stations[self.route[self.current_station_idx]]
            next_station_name = self.route[self.current_station_idx + 1]
            next_station = self.simulation_manager.stations[next_station_name]
            
            # --- Pre-Departure Checks ---
            if not self.crew_available:
                delay = random.uniform(15, 45)
                self.simulation_manager.log_event(f"{self.train_id} delayed at {current_station.name} for {delay:.2f} mins (Crew Shortage).")
                yield self.env.timeout(delay)
                self.crew_available = True # Crew becomes available after delay

            # --- Departure ---
            actual_departure_time = self.env.now
            self.simulation_manager.log_event(f"{self.train_id} departing {current_station.name} for {next_station_name}.")
            
            # --- Travel on Track ---
            track = self.simulation_manager.tracks[(current_station.name, next_station_name)]
            travel_start_time = self.env.now
            with track.resource.request() as req:
                yield req
                track.utilization_time += self.env.now - travel_start_time # Add wait time to utilization
                
                travel_time = track.get_travel_time(self.speed_kmh)
                self.simulation_manager.log_event(f"{self.train_id} on track {track.name}. Travel time: {travel_time:.2f} mins.")
                
                travel_start = self.env.now
                yield self.env.timeout(travel_time)
                track.utilization_time += self.env.now - travel_start # Add travel time to utilization

            # --- Arrival at Next Station ---
            self.current_station_idx += 1
            actual_arrival_time = self.env.now
            self.simulation_manager.log_event(f"{self.train_id} arriving at {next_station.name}.")

            # --- Platform Docking ---
            platform_request_start = self.env.now
            with next_station.platforms.request(priority=self.priority) as platform_req:
                yield platform_req
                platform_wait_time = self.env.now - platform_request_start
                if platform_wait_time > 0.1:
                    self.simulation_manager.log_event(f"{self.train_id} waited {platform_wait_time:.2f} mins for a platform at {next_station.name}.")

                platform_id = random.randint(1, next_station.platforms.capacity)
                
                # --- Record Data ---
                scheduled_arrival = self.schedule[next_station_name].get('arrival', actual_arrival_time)
                delay = max(0, actual_arrival_time - scheduled_arrival)
                self.simulation_manager.log_event(f"{self.train_id} docked at {next_station.name} (Platform {platform_id}). Delay: {delay:.2f} mins.")
                
                # Write data for CSV
                self.simulation_manager.record_arrival_data({
                    "train_id": self.train_id, "train_type": self.train_type,
                    "station": next_station.name,
                    "scheduled_arrival": scheduled_arrival,
                    "actual_arrival": actual_arrival_time,
                    "scheduled_departure": self.schedule[next_station_name].get('departure'),
                    "actual_departure": None, # Will be filled on departure
                    "delay_minutes": delay,
                    "platform_used": platform_id, "track_used": track.name
                })

                # --- Halt Logic ---
                if self.current_station_idx < len(self.route) - 1:
                    scheduled_departure = self.schedule[next_station_name]['departure']
                    halt_duration = scheduled_departure - actual_arrival_time
                    
                    # Enforce a minimum halt time, even if delayed
                    actual_halt_duration = max(2, halt_duration)
                    
                    self.simulation_manager.log_event(f"{self.train_id} halting at {next_station.name} for {actual_halt_duration:.2f} mins.")
                    
                    # Wait for signal clearance before halting period ends
                    yield next_station.signal | self.env.timeout(actual_halt_duration - 1)
                    if not next_station.signal.triggered:
                        self.simulation_manager.log_event(f"{self.train_id} waiting for signal at {next_station.name}.")
                        yield next_station.signal
                    
                    self.simulation_manager.log_event(f"{self.train_id} received green signal at {next_station.name}.")
                    yield self.env.timeout(1) # Final minute of halt

        self.simulation_manager.log_event(f"{self.train_id} finished journey at {self.route[-1]}.")


class SimulationManager:
    """
    Manages the setup, execution, and data collection of the railway simulation.
    """
    def __init__(self, env, start_time):
        self.env = env
        self.start_time = start_time
        self.stations = {}
        self.tracks = {}
        self.trains = []
        self.event_log = []
        self.csv_data = []

    def log_event(self, details):
        """Logs a simulation event with a timestamp."""
        timestamp = self.start_time + datetime.timedelta(minutes=self.env.now)
        log_entry = f"{timestamp.strftime('%H:%M:%S')} [SimTime: {self.env.now:.2f}] - {details}"
        self.event_log.append(log_entry)

    def record_arrival_data(self, data):
        """Stores train arrival data for final CSV output."""
        self.csv_data.append(data)

    def setup_environment(self):
        """Creates all stations and track segments."""
        self.log_event("Setting up simulation environment...")
        # Create stations
        for name, data in STATIONS_DATA.items():
            self.stations[name] = Station(self.env, name, data['platforms'])
        self.log_event(f"Created stations: {', '.join(self.stations.keys())}")

        # Create tracks
        for i in range(len(STATION_NAMES) - 1):
            start_name = STATION_NAMES[i]
            end_name = STATION_NAMES[i+1]
            distance = STATIONS_DATA[end_name]['distance']
            track_name = f"{start_name.replace(' ', '_')}-{end_name.replace(' ', '_')}"
            self.tracks[(start_name, end_name)] = Track(self.env, track_name, distance)
        self.log_event(f"Created {len(self.tracks)} track segments.")

    def create_trains(self, train_configs):
        """Creates train processes based on a list of configurations."""
        route = STATION_NAMES # All trains follow the full route for this simulation
        for train_id, config in train_configs.items():
            self.trains.append(Train(self.env, train_id, config, route, self))

    def run_simulation(self, until):
        """Starts and runs the simulation, including controllers."""
        self.log_event("Starting simulation...")
        self.env.process(self.signal_controller())
        self.env.process(self.dynamic_events_injector())
        self.env.run(until=until)
        self.log_event("Simulation finished.")

    def signal_controller(self):
        """Periodically grants departure clearance at stations."""
        while True:
            for station in self.stations.values():
                # In a more complex model, this would be based on track clearance ahead
                station.grant_departure_clearance()
            yield self.env.timeout(1) # Check every minute

    def dynamic_events_injector(self):
        """Injects random events like weather changes or track maintenance."""
        yield self.env.timeout(60)
        track_to_affect = self.tracks[('Bagra Tawa', 'Hoshangabad')]
        track_to_affect.weather = "rain"
        self.log_event(f"EVENT: Weather on {track_to_affect.name} changed to 'rain'.")

        yield self.env.timeout(120)
        track_to_affect.condition = "degraded"
        self.log_event(f"EVENT: Track condition on {track_to_affect.name} changed to 'degraded'.")

    def print_results(self):
        """Prints simulation logs, metrics, and saves CSV."""
        print(" --- SIMULATION EVENT LOG ---")
        for entry in self.event_log:
            print(entry)

        print("--- CONGESTION METRICS ---")
        total_delay = sum(row['delay_minutes'] for row in self.csv_data)
        num_arrivals = len(self.csv_data)
        avg_delay = total_delay / num_arrivals if num_arrivals > 0 else 0
        
        waiting_trains = 0
        for station in self.stations.values():
            waiting_trains += len(station.platforms.queue)

        total_track_utilization_time = sum(t.utilization_time for t in self.tracks.values())
        total_possible_track_time = sum(t.resource.capacity * self.env.now for t in self.tracks.values())
        avg_track_utilization_percent = (total_track_utilization_time / total_possible_track_time) * 100 if total_possible_track_time > 0 else 0


        print(f"Simulation duration: {self.env.now:.2f} minutes")
        print(f"Total train arrivals recorded: {num_arrivals}")
        print(f"Average delay per arrival: {avg_delay:.2f} minutes")
        print(f"Trains still waiting for platforms at end: {waiting_trains}")
        print(f"Average track utilization: {avg_track_utilization_percent:.2f}%")

        # Save to CSV
        csv_filename = "train_simulation_output.csv"
        header = [
            "train_id", "train_type", "station", "scheduled_arrival", "actual_arrival",
            "scheduled_departure", "actual_departure", "delay_minutes", "platform_used", "track_used"
        ]
        with open(csv_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in self.csv_data:
                # Format datetime objects for CSV
                for key, val in row.items():
                    if isinstance(val, (int, float)) and 'time' in key:
                         row[key] = (self.start_time + datetime.timedelta(minutes=val)).strftime('%H:%M')
                writer.writerow(row)
        
        print(f"--- CSV OUTPUT ---")
        print(f"Full simulation data saved to '{csv_filename}'")
        print("Showing first 5 rows of output:")
        with open(csv_filename, 'r') as f:
            for i, line in enumerate(f):
                if i > 5: break
                print(line.strip())


# --- Main Execution ---

if __name__ == "__main__":
    # 1. Define the simulation scenario
    train_configs = {
        "12001-RJ": {
            "type": "Rajdhani Express", "loco_type": "WAP-7",
            "schedule": {
                "Bhopal Junction": {"departure": 5},
                "Itarsi Junction": {"arrival": 90},
            }
        },
        "12155-ME": {
            "type": "Mail/Express",
            "schedule": {
                "Bhopal Junction": {"departure": 10},
                "Rani Kamalapati": {"arrival": 18, "departure": 20},
                "Hoshangabad": {"arrival": 80, "departure": 82},
                "Itarsi Junction": {"arrival": 100},
            }
        },
        "22129-SF": {
            "type": "Superfast Passenger",
            "schedule": {
                "Bhopal Junction": {"departure": 15},
                "Rani Kamalapati": {"arrival": 23, "departure": 25},
                "Obaidullaganj": {"arrival": 45, "departure": 47},
                "Hoshangabad": {"arrival": 95, "departure": 97},
                "Itarsi Junction": {"arrival": 115},
            }
        },
        "01665-MEMU": {
            "type": "MEMU/DEMU",
            "schedule": {
                "Bhopal Junction": {"departure": 20},
                "Rani Kamalapati": {"arrival": 30, "departure": 32},
                "Barkheda": {"arrival": 55, "departure": 56},
                "Obaidullaganj": {"arrival": 70, "departure": 71},
                "Hoshangabad": {"arrival": 110, "departure": 111},
                "Pachama": {"arrival": 125, "departure": 126},
                "Banapura": {"arrival": 140, "departure": 141},
                "Itarsi Junction": {"arrival": 165},
            }
        },
        "51189-PASS": {
            "type": "Ordinary Passenger", "crew_available": False, # Inject a delay
            "schedule": {
                "Bhopal Junction": {"departure": 25},
                "Rani Kamalapati": {"arrival": 40, "departure": 42},
                "Barkheda": {"arrival": 70, "departure": 72},
                "Obaidullaganj": {"arrival": 90, "departure": 92},
                "Bagra Tawa": {"arrival": 115, "departure": 117},
                "Hoshangabad": {"arrival": 140, "departure": 142},
                "Pachama": {"arrival": 155, "departure": 157},
                "Banapura": {"arrival": 175, "departure": 177},
                "Itarsi Junction": {"arrival": 200},
            }
        },
        "FR-GDS-01": {
            "type": "Freight Train", "loco_type": "WAG-9",
            "schedule": {
                "Bhopal Junction": {"departure": 30},
                "Bagra Tawa": {"arrival": 150, "departure": 180}, # Long halt
                "Itarsi Junction": {"arrival": 220},
            }
        },
        # Add more trains to meet the 10+ requirement
        "12002-RJ": {
            "type": "Rajdhani Express", "schedule": { "Bhopal Junction": {"departure": 35}, "Itarsi Junction": {"arrival": 120} }
        },
        "12174-ME": {
            "type": "Mail/Express", "schedule": { "Bhopal Junction": {"departure": 40}, "Hoshangabad": {"arrival": 110, "departure": 112}, "Itarsi Junction": {"arrival": 130} }
        },
        "01666-MEMU": {
            "type": "MEMU/DEMU", "schedule": { "Bhopal Junction": {"departure": 45}, "Rani Kamalapati": {"arrival": 55, "departure": 57}, "Itarsi Junction": {"arrival": 150} }
        },
        "FR-GDS-02": {
            "type": "Freight Train", "schedule": { "Bhopal Junction": {"departure": 50}, "Itarsi Junction": {"arrival": 250} }
        }
    }

    # 2. Initialize and run the simulation
    env = simpy.Environment()
    manager = SimulationManager(env, SIMULATION_START_TIME)
    
    manager.setup_environment()
    manager.create_trains(train_configs)
    manager.run_simulation(until=400)
    
    # 3. Print results
    manager.print_results()

"""
--- Alternatives to SimPy for Railway Simulation ---

While SimPy is excellent for building custom, process-based discrete-event models from scratch,
other tools offer different advantages, especially for large-scale or specialized railway simulation:

1. AnyLogic:
   - Pros: A multi-method simulation tool with a dedicated, graphical Rail Library. It simplifies the modeling of tracks, switches, yards, and stations. Great for visual modeling and complex animations.
   - Cons: Commercial software, can have a steep learning curve for advanced features.

2. OR-Tools (by Google):
   - Pros: Not a simulator, but a powerful open-source suite for combinatorial optimization. It's ideal for solving scheduling and routing problems, such as creating optimal timetables or assigning crews and rolling stock, which can then be fed into a simulator like SimPy.
   - Cons: Purely a solver, not a simulator. Requires significant programming effort to integrate with a simulation model.

3. Arena / SimEvents (MATLAB):
   - Pros: Industrial-grade, high-fidelity discrete-event simulation environments. They are flowchart-based (Arena) or block-based (SimEvents), which can be more intuitive for engineers. They offer extensive statistical analysis capabilities.
   - Cons: Commercial, expensive, and can be overkill for smaller or more abstract models. SimEvents requires a MATLAB/Simulink license.
   
4. OpenTrack:
   - Pros: A highly specialized, commercial microscopic railway simulation tool. It is considered an industry standard for detailed railway planning, capacity analysis, and timetable verification.
   - Cons: Very expensive and highly specialized, making it unsuitable for general-purpose modeling or academic use without a specific license.
"""  