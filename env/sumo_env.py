import os
import sys
import numpy as np
import gym
from gym import spaces

# We need to import traci and sumolib from SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib


class SumoSingleIntersectionEnv(gym.Env):
    """
    SUMO Environment for single intersection traffic light control
    
    State: tuple (X1, X2, L) where:
           X1: Number of vehicles in the west-east direction
           X2: Number of vehicles in the north-south direction
           L: Traffic light configuration (0: green for X1, 1: yellow for X1, 
                                          2: green for X2, 3: yellow for X2)
    
    Action: binary value:
           0: Continue current traffic light phase
           1: Switch to the next phase
    
    Reward: Negative of the sum of squared queue lengths, i.e., -(X1^2 + X2^2)
    """
    
    def __init__(self, 
                 sumocfg_file='env/sumo_files/single_intersection.sumocfg',
                 use_gui=False,
                 max_steps=1000,
                 arrival_rate=0.25):
        super(SumoSingleIntersectionEnv, self).__init__()
        
        # SUMO configuration
        self.sumocfg = sumocfg_file
        self.use_gui = use_gui
        if os.name == 'nt':  # Check if we're on Windows
            sumo_bin_dir = os.path.join(os.environ['SUMO_HOME'], 'bin')
            self.sumo_binary = os.path.join(sumo_bin_dir, 'sumo-gui.exe' if self.use_gui else 'sumo.exe')
        else:
            self.sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        self.max_steps = max_steps
        self.arrival_rate = arrival_rate  # Bernoulli parameter for vehicle arrivals
        
        # Traffic light ID and lanes
        self.tls_id = 'TL'  # Traffic light ID
        self.we_lanes = ['EW_0']  # West-East lanes
        self.ns_lanes = ['NS_0']  # North-South lanes
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0: Continue, 1: Switch
        
        # Observation: (X1, X2, L)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(100),  # X1: Queue length limit
            spaces.Discrete(100),  # X2: Queue length limit
            spaces.Discrete(4),    # L: Traffic light configuration
        ))
        
        self.reset()
    
    def _get_state(self):
        """Get the current state of the environment."""
        # Get number of vehicles in each direction
        x1 = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.we_lanes)
        x2 = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.ns_lanes)
        
        # Get current traffic light phase
        tls_phase = traci.trafficlight.getPhase(self.tls_id)
        
        return (x1, x2, tls_phase)
    
    def _compute_reward(self, state):
        """Compute reward based on the current state."""
        x1, x2, _ = state
        # Negative of sum of squared queue lengths
        reward = -(x1**2 + x2**2)
        return reward
    
    def step(self, action):
        """
        Take an action and move the simulation forward.
        
        Args:
            action: 0 (continue) or 1 (switch)
            
        Returns:
            next_state: The new environment state
            reward: The reward for the action taken
            done: Whether the episode is finished
            info: Additional information
        """
        # Current state
        current_state = self._get_state()
        x1, x2, tls_phase = current_state
        
        # Take action
        if action == 1:  # Switch
            next_phase = (tls_phase + 1) % 4
            traci.trafficlight.setPhase(self.tls_id, next_phase)
        
        # Simulate one step
        traci.simulationStep()
        self.step_count += 1
        
        # Generate new vehicles based on arrival probability
        self._generate_vehicles()
        
        # Get new state
        next_state = self._get_state()
        
        # Compute reward
        reward = self._compute_reward(next_state)
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        # Additional info
        info = {
            'step_count': self.step_count,
            'queue_length_we': next_state[0],
            'queue_length_ns': next_state[1],
            'tls_phase': next_state[2]
        }
        
        return next_state, reward, done, info
    
    def reset(self):
        """Reset the environment to initial state."""
        # Close any existing SUMO simulation
        if 'traci' in sys.modules and traci.isLoaded():
            traci.close()
        
        # Start SUMO with TraCI
        sumo_cmd = [self.sumo_binary, "-c", self.sumocfg]
        
        try:
            print(f"Starting SUMO with command: {sumo_cmd}")
            traci.start(sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            print(f"SUMO_HOME is set to: {os.environ.get('SUMO_HOME', 'Not set')}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Configuration file path: {os.path.abspath(self.sumocfg)}")
            raise
        
        # Reset step counter
        self.step_count = 0
        
        # Reset vehicle IDs counter
        self.vehicle_id_counter = 0
        
        # Generate initial vehicles
        self._generate_vehicles()
        
        # Get initial state
        state = self._get_state()
        
        return state
    def _generate_vehicles(self):
        """Generate vehicles with Bernoulli probability."""
        # West-East direction
        if np.random.random() < self.arrival_rate:
            self._add_vehicle(self.we_lanes[0], 'WE')
        
        # North-South direction
        if np.random.random() < self.arrival_rate:
            self._add_vehicle(self.ns_lanes[0], 'NS')
    
    def _add_vehicle(self, lane, direction):
        """Add a vehicle to the specified lane."""
        vehicle_id = f"{direction}_{self.vehicle_id_counter}"
        self.vehicle_id_counter += 1
        
        try:
            traci.vehicle.add(
                vehicle_id,
                routeID=f"{direction}_route",
                departLane="best",
                departSpeed="0",
                departPos="0",
                typeID="standard_car"
            )
        except traci.exceptions.TraCIException:
            # Vehicle could not be inserted (e.g., lane is full)
            pass
    
    def close(self):
        """Close the environment."""
        if 'traci' in sys.modules and traci.isLoaded():
            traci.close()


def create_sumo_files(output_dir='env/sumo_files'):
    """Create necessary SUMO network and route files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create network file (.net.xml)
    net_file = os.path.join(output_dir, 'single_intersection.net.xml')
    with open(net_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="100.00,100.00" convBoundary="0.00,0.00,200.00,200.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    
    <!-- Traffic Light Logic Definition -->
    <tlLogic id="TL" type="static" programID="0" offset="0">
        <phase duration="31" state="Gr"/>  <!-- Green for WE, red for NS -->
        <phase duration="4" state="yr"/>   <!-- Yellow for WE, red for NS -->
        <phase duration="31" state="rG"/>  <!-- Red for WE, green for NS -->
        <phase duration="4" state="ry"/>   <!-- Red for WE, yellow for NS -->
    </tlLogic>
    
    <edge id=":TL_0" function="internal">
        <lane id=":TL_0_0" index="0" speed="13.89" length="9.03" shape="95.20,107.40 95.20,98.37"/>
    </edge>
    <edge id=":TL_1" function="internal">
        <lane id=":TL_1_0" index="0" speed="13.89" length="9.03" shape="107.40,104.80 98.37,104.80"/>
    </edge>
    <edge id="EW" from="E" to="TL" priority="1">
        <lane id="EW_0" index="0" speed="13.89" length="92.60" shape="200.00,104.80 107.40,104.80"/>
    </edge>
    <edge id="NS" from="N" to="TL" priority="1">
        <lane id="NS_0" index="0" speed="13.89" length="92.60" shape="95.20,200.00 95.20,107.40"/>
    </edge>
    <edge id="TLE" from="TL" to="E" priority="1">
        <lane id="TLE_0" index="0" speed="13.89" length="92.60" shape="98.37,104.80 200.00,104.80"/>
    </edge>
    <edge id="TLS" from="TL" to="S" priority="1">
        <lane id="TLS_0" index="0" speed="13.89" length="92.60" shape="95.20,98.37 95.20,0.00"/>
    </edge>
    <junction id="E" type="dead_end" x="200.00" y="100.00" incLanes="TLE_0" intLanes="" shape="200.00,100.00 200.00,103.20 200.00,100.00"/>
    <junction id="N" type="dead_end" x="100.00" y="200.00" incLanes="" intLanes="" shape="100.00,200.00 96.80,200.00 100.00,200.00"/>
    <junction id="S" type="dead_end" x="100.00" y="0.00" incLanes="TLS_0" intLanes="" shape="100.00,0.00 93.60,0.00 100.00,0.00"/>
    <junction id="TL" type="traffic_light" x="100.00" y="100.00" incLanes="NS_0 EW_0" intLanes=":TL_0_0 :TL_1_0" shape="93.60,107.40 96.80,107.40 107.40,106.40 107.40,103.20 98.80,98.37 93.60,98.37">
        <request index="0" response="00" foes="10" cont="0"/>
        <request index="1" response="01" foes="01" cont="0"/>
    </junction>
    <connection from="EW" to="TLE" fromLane="0" toLane="0" via=":TL_1_0" tl="TL" linkIndex="1" dir="s" state="o"/>
    <connection from="NS" to="TLS" fromLane="0" toLane="0" via=":TL_0_0" tl="TL" linkIndex="0" dir="s" state="o"/>
    <connection from=":TL_0" to="TLS" fromLane="0" toLane="0" dir="s" state="M"/>
    <connection from=":TL_1" to="TLE" fromLane="0" toLane="0" dir="s" state="M"/>
    </net>
    """)
    # Create route file (.rou.xml)
    route_file = os.path.join(output_dir, 'single_intersection.rou.xml')
    with open(route_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
    <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="standard_car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15" color="1,1,0"/>
    
    <route id="WE_route" edges="EW TLE"/>
    <route id="NS_route" edges="NS TLS"/>
    </routes>
""")