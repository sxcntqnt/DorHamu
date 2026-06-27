import random
import requests
import numpy as np
from collections import OrderedDict
from environments.environment import Environment

# Match original constants scale where possible
PER_AGENT_STATE_SIZE = 6
GLOBAL_STATE_SIZE = 1
ACTION_SIZE = 2

class ABStreetIntersectionsEnv(Environment):
    def __init__(self, constants, device, agent_ID, eval_agent, map_path, vis=False):
        # map_path here will point to the A/B Street map binary or scenario binary path
        super(ABStreetIntersectionsEnv, self).__init__(constants, device, agent_ID, eval_agent, map_path, vis)
        
        # In multi-worker environments, dynamically assign ports to avoid cross-talk
        # e.g., Worker 0 uses port 1234, Worker 1 uses port 1235
        self.port = 1234 + self.agent_ID
        self.host = f"http://localhost:{self.port}"
        
        self.env_name = f"{constants['environment']['shape'][0]}_{constants['environment']['shape'][1]}_intersections"
        
        # Simulation step tracking in seconds (A/B street uses absolute timestamps)
        self.sim_time_seconds = 0
        self.step_interval_seconds = 30  # Adjust based on how long each step lasts
        
        # Initialize map metadata
        self.intersections = self._discover_intersections()
        self.intersections_index = {intersection: i for i, intersection in enumerate(self.intersections)}
        
        # Mock/Calculate coordinates for reward interpolation if map metadata allows
        self.distances = self._get_intersection_distances()

    def _discover_intersections(self):
        """Discovers intersections running on this server instance."""
        try:
            response = requests.get(f"{self.host}/traffic-signals/get-all-current-state").json()
            return list(response.keys())
        except Exception:
            # Fallback placeholder if server isn't up at instantiation
            return [f"intersection_{i}" for i in range(self.constants['environment']['shape'][0] * self.constants['environment']['shape'][1])]

    def _get_intersection_distances(self):
        """Mock distances for grid layout fallback if geometry queries aren't used."""
        distances = {}
        for i in self.intersections:
            distances[i] = {}
            for j in self.intersections:
                # Fallback to structural placeholder distances if needed
                distances[i][j] = 1.0 if i != j else 0.0
        return distances

    def _open_connection(self):
        """Initializes/Loads the target scenario over the headless API."""
        payload = {
            # Use specific map scenario from your constants configuration
            "scenario": self.net_path, 
            "modifiers": [],
            "edits": None
        }
        requests.post(f"{self.host}/sim/load", json=payload)
        self.sim_time_seconds = 0

    def _get_sim_step(self, normalize):
        """Gets current time in seconds after midnight."""
        sim_step = self.sim_time_seconds
        if normalize: 
            sim_step /= (self.constants['episode']['max_ep_steps'] / 10.)
        return sim_step

    def _seconds_to_hhmmss(self, total_seconds):
        """Formats raw seconds into the 'HH:MM:SS' string format required by A/B Street."""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_state(self):
        state = self._make_state()
        normalize = True if self.agent_type != 'rule' else False
        sim_step = self._get_sim_step(normalize)
        
        # Query global signal states directly from the API
        all_signals = requests.get(f"{f'{self.host}/traffic-signals/get-all-current-state'}").json()
        
        for intersection in self.intersections:
            signal_data = all_signals.get(intersection, {"waiting": [], "stage": 0})
            
            # Jam length -> directly maps to waiting agents at the signal intersection
            waiting_count = len(signal_data.get("waiting", []))
            # Padding to mimic old detector array dimensions if shape demands it
            jam_length = [waiting_count] * 4  
            self._add_to_state(state, jam_length, key='jam_length', intersection=intersection)
            
            # Current active stage/phase index
            curr_phase = signal_data.get("stage", 0)
            self._add_to_state(state, curr_phase, key='curr_phase', intersection=intersection)
            
            # Elapsed time placeholder (or track internally if needed)
            elapsed_phase_time = 0.0
            self._add_to_state(state, elapsed_phase_time, key='elapsed_phase_time', intersection=intersection)
            
            if not self.single_agent and self.agent_type != 'rule':
                self._add_to_state(state, sim_step, key='sim_step', intersection=intersection)
                
        if self.single_agent or self.agent_type == 'rule':
            self._add_to_state(state, sim_step, key='sim_step', intersection=None)

        # Retain original Multi-Agent neighborhood pooling/interpolation rules
        if self.single_agent or self.agent_type == 'rule' or self.constants['multiagent']['state_interpolation'] == 0:
            return self._process_state(state)

        state_size = PER_AGENT_STATE_SIZE + GLOBAL_STATE_SIZE
        final_state = []
        for intersection in self.intersections:
            neighborhood = self.neighborhoods.get(intersection, [])
            intersection_state = state[intersection]
            final_state.append(np.zeros(shape=(state_size * self.max_num_neighbors,)))
            final_state[-1][:state_size] = np.array(intersection_state)[:state_size]
            
            for n, neighbor in enumerate(neighborhood):
                if neighbor in state:
                    extension = self.constants['multiagent']['state_interpolation'] * np.array(state[neighbor])[:state_size]
                    range_start = (n + 1) * state_size
                    range_end = range_start + state_size
                    final_state[-1][range_start:range_end] = extension
                    
        return self._process_state(final_state)

    def get_reward(self, get_global):
        reward_interpolation = self.constants['multiagent']['reward_interpolation']
        local_rewards = {}
        
        # Pull global block metric updates
        all_signals = requests.get(f"{self.host}/traffic-signals/get-all-current-state").json()
        
        for intersection in self.intersections:
            signal_data = all_signals.get(intersection, {"waiting": []})
            waiting_cars = len(signal_data.get("waiting", []))
            
            # Negative optimization target: fewer waiting cars yields higher rewards
            local_rewards[intersection] = -float(waiting_cars) / 10.0  

        if get_global:
            return sum(local_rewards.values())
            
        if len(self.intersections) == 1 or self.single_agent:
            return np.array([sum(local_rewards.values())])
            
        if reward_interpolation == 0.:
            return np.array([r for r in list(local_rewards.values())])
            
        if reward_interpolation == 1.:
            gr = sum(local_rewards.values())
            return np.array([gr] * len(self.intersections))
            
        # Reapply spatial context distance decay matrix
        arr = []
        for intersection in self.intersections:
            dists = self.distances[intersection]
            max_dist = max([d for d in list(dists.values())]) if dists else 1.0
            local_rew = 0.
            for inner_int in self.intersections:
                d = dists.get(inner_int, 1.0)
                r = local_rewards[inner_int]
                local_rew += pow(reward_interpolation, (d / (max_dist or 1.0))) * r
            arr.append(local_rew)
        return np.array(arr)

    def _execute_action(self, action):
        """Translates action dictionary states to direct A/B Street configuration setups."""
        for intersection, value in action.items():
            if value == 0: 
                continue  # Hold phase configuration
                
            # Fetch current structure pattern
            try:
                signal_config = requests.get(f"{self.host}/traffic-signals/get?id={intersection}").json()
                
                # Cycle configuration indices
                current_stage = signal_config.get("current_stage_index", 0)
                total_stages = len(signal_config.get("stages", []))
                
                if total_stages > 0:
                    new_stage = (current_stage + 1) % total_stages
                    signal_config["current_stage_index"] = new_stage
                    
                    # Submit payload changes back upstream
                    requests.post(f"{self.host}/traffic-signals/set", json=signal_config)
            except Exception:
                pass

    def step(self, a, ep_step, get_global_reward, def_agent=False):
        action = self._process_action(a)
        if not def_agent:
            self._execute_action(action)
            
        # Advance simulation clock ahead internally and sync server
        self.sim_time_seconds += self.step_interval_seconds
        target_time_str = self._seconds_to_hhmmss(self.sim_time_seconds)
        
        requests.get(f"{self.host}/sim/goto-time?t={target_time_str}")
        
        s_ = self.get_state()
        r = self.get_reward(get_global_reward)
        
        done = False
        if ep_step >= self.constants['episode']['max_ep_steps']:
            if self.eval_agent:
                # Clean exit routines
                pass
            else:
                s_ = self.reset()
            done = True
            
        return s_, r, done
