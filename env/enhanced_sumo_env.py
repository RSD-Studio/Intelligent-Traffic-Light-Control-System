"""
Enhanced SUMO Environment with Expanded State Space and Multi-Objective Rewards
"""

import os
import sys
import numpy as np
import gym
from gym import spaces
from collections import deque

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib


class EnhancedSumoEnv(gym.Env):
    """
    Enhanced SUMO Environment with rich state representation and multi-objective rewards

    Enhanced State Space (15+ dimensions):
    - Queue lengths (WE, NS)
    - Average waiting times
    - Average speeds
    - Traffic light phase and duration
    - Queue trends (temporal)
    - Throughput metrics
    - Fairness metrics
    """

    def __init__(self,
                 sumocfg_file='env/sumo_files/single_intersection.sumocfg',
                 use_gui=False,
                 max_steps=1000,
                 arrival_rate=0.25,
                 time_varying_traffic=True,
                 reward_weights=None):
        """
        Args:
            sumocfg_file: Path to SUMO configuration
            use_gui: Whether to use SUMO GUI
            max_steps: Maximum simulation steps
            arrival_rate: Base arrival rate for vehicles
            time_varying_traffic: Use realistic time-varying traffic patterns
            reward_weights: Dict of reward component weights
        """
        super(EnhancedSumoEnv, self).__init__()

        # SUMO configuration
        self.sumocfg = sumocfg_file
        self.use_gui = use_gui
        if os.name == 'nt':
            sumo_bin_dir = os.path.join(os.environ['SUMO_HOME'], 'bin')
            self.sumo_binary = os.path.join(sumo_bin_dir, 'sumo-gui.exe' if self.use_gui else 'sumo.exe')
        else:
            self.sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'

        self.max_steps = max_steps
        self.base_arrival_rate = arrival_rate
        self.time_varying_traffic = time_varying_traffic

        # Traffic light and lanes
        self.tls_id = 'TL'
        self.we_lanes = ['EW_0']
        self.ns_lanes = ['NS_0']

        # State tracking
        self.queue_history = deque(maxlen=5)  # Last 5 time steps
        self.waiting_time_history = deque(maxlen=5)
        self.phase_duration = 0
        self.last_phase = 0
        self.vehicles_passed = 0
        self.total_switches = 0

        # Reward weights (multi-objective)
        self.reward_weights = reward_weights or {
            'queue': 0.40,
            'waiting_time': 0.30,
            'throughput': 0.15,
            'fairness': 0.10,
            'switch_penalty': 0.05
        }

        # Action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0: Continue, 1: Switch

        # Enhanced observation: 15-dimensional state
        self.observation_space = spaces.Box(
            low=np.array([0] * 15, dtype=np.float32),
            high=np.array([100] * 15, dtype=np.float32),
            dtype=np.float32
        )

        # Metrics tracking
        self.episode_metrics = {
            'total_waiting_time': 0,
            'total_vehicles': 0,
            'total_throughput': 0
        }

        self.reset()

    def _get_time_varying_arrival_rate(self):
        """Get arrival rate based on time of day (realistic traffic patterns)"""
        if not self.time_varying_traffic:
            return self.base_arrival_rate

        # Simulate 24-hour cycle (1 step = 1 second)
        hour = (self.step_count % 86400) / 3600  # Convert to hours

        # Morning rush: 7-9 AM
        if 7 <= hour <= 9:
            return min(0.8, self.base_arrival_rate * 3.2)
        # Evening rush: 5-7 PM
        elif 17 <= hour <= 19:
            return min(0.85, self.base_arrival_rate * 3.4)
        # Night: 11 PM - 5 AM
        elif hour >= 23 or hour <= 5:
            return max(0.05, self.base_arrival_rate * 0.2)
        # Midday: 11 AM - 2 PM
        elif 11 <= hour <= 14:
            return self.base_arrival_rate * 1.5
        # Regular daytime
        else:
            return self.base_arrival_rate * 1.2

    def _get_enhanced_state(self):
        """Get enhanced 15-dimensional state representation"""
        # Basic queue information
        x1 = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.we_lanes)
        x2 = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.ns_lanes)

        # Waiting times
        we_waiting_times = []
        ns_waiting_times = []

        for lane in self.we_lanes:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                we_waiting_times.append(traci.vehicle.getWaitingTime(veh_id))

        for lane in self.ns_lanes:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                ns_waiting_times.append(traci.vehicle.getWaitingTime(veh_id))

        avg_wait_we = np.mean(we_waiting_times) if we_waiting_times else 0
        avg_wait_ns = np.mean(ns_waiting_times) if ns_waiting_times else 0
        max_wait = max([avg_wait_we, avg_wait_ns])

        # Speeds
        we_speeds = []
        ns_speeds = []

        for lane in self.we_lanes:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                we_speeds.append(traci.vehicle.getSpeed(veh_id))

        for lane in self.ns_lanes:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                ns_speeds.append(traci.vehicle.getSpeed(veh_id))

        avg_speed_we = np.mean(we_speeds) if we_speeds else 0
        avg_speed_ns = np.mean(ns_speeds) if ns_speeds else 0

        # Traffic light state
        tls_phase = traci.trafficlight.getPhase(self.tls_id)

        # Update phase duration
        if tls_phase != self.last_phase:
            self.phase_duration = 0
            self.last_phase = tls_phase
        else:
            self.phase_duration += 1

        # Queue trends (change from previous step)
        if len(self.queue_history) > 0:
            prev_we, prev_ns = self.queue_history[-1]
            queue_trend_we = x1 - prev_we
            queue_trend_ns = x2 - prev_ns
        else:
            queue_trend_we = 0
            queue_trend_ns = 0

        # Update history
        self.queue_history.append((x1, x2))
        self.waiting_time_history.append((avg_wait_we, avg_wait_ns))

        # Construct 15-dimensional state vector
        state = np.array([
            x1,                             # 0: WE queue length
            x2,                             # 1: NS queue length
            x1 + x2,                        # 2: Total queue
            abs(x1 - x2),                   # 3: Queue difference
            avg_wait_we,                    # 4: WE avg waiting time
            avg_wait_ns,                    # 5: NS avg waiting time
            max_wait,                       # 6: Max waiting time
            avg_speed_we,                   # 7: WE avg speed
            avg_speed_ns,                   # 8: NS avg speed
            tls_phase,                      # 9: Current phase
            self.phase_duration / 100.0,    # 10: Phase duration (normalized)
            queue_trend_we,                 # 11: WE queue trend
            queue_trend_ns,                 # 12: NS queue trend
            self.vehicles_passed / 100.0,   # 13: Throughput (normalized)
            self.total_switches / 10.0      # 14: Number of switches (normalized)
        ], dtype=np.float32)

        return state

    def _compute_multi_objective_reward(self, state, action, next_state):
        """
        Compute multi-objective reward function

        Components:
        1. Queue length penalty (minimize congestion)
        2. Waiting time penalty (minimize delays)
        3. Throughput reward (maximize flow)
        4. Fairness reward (balance between directions)
        5. Switch penalty (discourage excessive switching)
        """
        # Extract relevant features
        x1 = next_state[0]  # WE queue
        x2 = next_state[1]  # NS queue
        avg_wait_we = next_state[4]
        avg_wait_ns = next_state[5]
        throughput = next_state[13] * 100.0  # Denormalize

        # 1. Queue length penalty (squared to emphasize large queues)
        queue_penalty = -(x1**2 + x2**2)

        # 2. Waiting time penalty (linear)
        waiting_penalty = -(avg_wait_we + avg_wait_ns)

        # 3. Throughput reward (number of vehicles that completed their journey)
        throughput_reward = self._get_vehicles_completed() * 10

        # 4. Fairness penalty (penalize imbalance)
        fairness_penalty = -abs(avg_wait_we - avg_wait_ns)

        # 5. Switch penalty (discourage frequent switching)
        switch_penalty = -10 if action == 1 else 0

        # Combine with weights
        total_reward = (
            self.reward_weights['queue'] * queue_penalty +
            self.reward_weights['waiting_time'] * waiting_penalty +
            self.reward_weights['throughput'] * throughput_reward +
            self.reward_weights['fairness'] * fairness_penalty +
            self.reward_weights['switch_penalty'] * switch_penalty
        )

        return total_reward, {
            'queue_penalty': queue_penalty,
            'waiting_penalty': waiting_penalty,
            'throughput_reward': throughput_reward,
            'fairness_penalty': fairness_penalty,
            'switch_penalty': switch_penalty
        }

    def _get_vehicles_completed(self):
        """Count vehicles that completed their journey this step"""
        # This is an approximation - count vehicles that left the network
        arrived_vehicles = traci.simulation.getArrivedNumber()
        return arrived_vehicles

    def step(self, action):
        """Execute action and advance simulation"""
        # Get current state
        current_state = self._get_enhanced_state()

        # Take action
        if action == 1:  # Switch
            tls_phase = traci.trafficlight.getPhase(self.tls_id)
            next_phase = (tls_phase + 1) % 4
            traci.trafficlight.setPhase(self.tls_id, next_phase)
            self.total_switches += 1

        # Simulate one step
        traci.simulationStep()
        self.step_count += 1

        # Generate new vehicles with time-varying arrival rate
        arrival_rate = self._get_time_varying_arrival_rate()
        self._generate_vehicles(arrival_rate)

        # Track throughput
        self.vehicles_passed += self._get_vehicles_completed()

        # Get next state
        next_state = self._get_enhanced_state()

        # Compute multi-objective reward
        reward, reward_components = self._compute_multi_objective_reward(
            current_state, action, next_state
        )

        # Update metrics
        self.episode_metrics['total_waiting_time'] += next_state[6]
        self.episode_metrics['total_vehicles'] = next_state[0] + next_state[1]
        self.episode_metrics['total_throughput'] = self.vehicles_passed

        # Check if done
        done = self.step_count >= self.max_steps

        # Additional info
        info = {
            'step_count': self.step_count,
            'queue_length_we': next_state[0],
            'queue_length_ns': next_state[1],
            'total_queue': next_state[2],
            'avg_waiting_time': (next_state[4] + next_state[5]) / 2,
            'throughput': self.vehicles_passed,
            'reward_components': reward_components
        }

        return next_state, reward, done, info

    def reset(self):
        """Reset environment to initial state"""
        # Close any existing SUMO simulation
        if 'traci' in sys.modules and traci.isLoaded():
            traci.close()

        # Start SUMO
        sumo_cmd = [self.sumo_binary, "-c", self.sumocfg, "--no-warnings", "true"]

        try:
            traci.start(sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            raise

        # Reset counters
        self.step_count = 0
        self.vehicle_id_counter = 0
        self.vehicles_passed = 0
        self.total_switches = 0
        self.phase_duration = 0
        self.last_phase = 0

        # Clear history
        self.queue_history.clear()
        self.waiting_time_history.clear()

        # Reset metrics
        self.episode_metrics = {
            'total_waiting_time': 0,
            'total_vehicles': 0,
            'total_throughput': 0
        }

        # Generate initial vehicles
        self._generate_vehicles(self.base_arrival_rate)

        # Get initial state
        state = self._get_enhanced_state()

        return state

    def _generate_vehicles(self, arrival_rate):
        """Generate vehicles with given arrival probability"""
        # West-East direction
        if np.random.random() < arrival_rate:
            self._add_vehicle(self.we_lanes[0], 'WE')

        # North-South direction
        if np.random.random() < arrival_rate:
            self._add_vehicle(self.ns_lanes[0], 'NS')

    def _add_vehicle(self, lane, direction):
        """Add a vehicle to the specified lane"""
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
        """Close the environment"""
        if 'traci' in sys.modules and traci.isLoaded():
            traci.close()

    def get_episode_metrics(self):
        """Return episode performance metrics"""
        return self.episode_metrics.copy()
