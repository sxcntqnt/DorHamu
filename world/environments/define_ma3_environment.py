import gym
from gym import spaces
import numpy as np
import json
import subprocess
import pandas as pd
import pickle
import traci
import os
from pathlib import Path
from google.cloud import logging
from google.cloud import storage
from google.cloud import bigquery
import bigframes.pandas as bpd
from zenml import step
import networkx as nx

class MatatuEnv(gym.Env):
    def __init__(self, grg_path, abstreet_map_file, abstreet_scenario_file, sumo_config_file, curriculum_phase=0, env_id=0):
        super(MatatuEnv, self).__init__()

        # Initialize logging
        self.logger = logging.Client().logger("matatu_pipeline")
        self.curriculum_phase = curriculum_phase
        self.env_id = env_id  # Unique ID for SubprocVecEnv
        self.visitation_counts = {}  # For intrinsic exploration rewards
        self.place_cell_sigma = 2.0  # Spread for place cell activations
        self.intrinsic_weight = 0.1  # Weight for intrinsic curiosity rewards
        self.max_nodes = 100  # Fixed size for place_cells

        # Load GRG and embeddings
        try:
            gcs_client = storage.Client(project="my_project")
            bucket = gcs_client.bucket("my_bucket")
            blob = bucket.blob(grg_path)
            local_path = "/tmp/grg.pkl"
            blob.download_to_filename(local_path)
            with open(local_path, "rb") as f:
                grg_data = pickle.load(f)
            self.grg = grg_data["grg"]
            self.hex_to_node = grg_data["hex_to_node"]
            self.region_to_node = grg_data["region_to_node"]
            self.logger.log_text(f"Env {self.env_id}: Loaded Cloverleaf GRG from GCS")
        except Exception as e:
            self.logger.log_text(f"Env {self.env_id}: GRG load error: {str(e)}")
            raise

        try:
            grg_embeddings_df = bpd.read_gbq("SELECT hex_code, embedding, type FROM my_project.my_dataset.grg_embeddings").to_pandas()
            self.h3_embeddings = {
                row["hex_code"]: np.array(row["embedding"])
                for _, row in grg_embeddings_df.iterrows() if row["type"] == "hexagon"
            }
            self.region_embeddings = {
                row["hex_code"]: np.array(row["embedding"])
                for _, row in grg_embeddings_df.iterrows() if row["type"] == "region"
            }
            self.logger.log_text(f"Env {self.env_id}: Loaded GRG embeddings from BigQuery")
        except Exception as e:
            self.logger.log_text(f"Env {self.env_id}: Embeddings query error: {str(e)}")
            raise

        # Load H3 mappings and unified features
        try:
            h3_mappings_df = bpd.read_gbq("SELECT hex_code, road_id, edge_id, passenger_demand FROM my_project.my_dataset.h3_mappings").to_pandas()
            unified_df = pd.read_parquet("/tmp/unified_features.parquet")
            blob = bucket.blob("processed_data/unified_features.parquet")
            blob.download_to_filename("/tmp/unified_features.parquet")
            self.h3_mappings = h3_mappings_df
            self.unified_df = unified_df
            self.logger.log_text(f"Env {self.env_id}: Loaded H3 mappings and unified features")
        except Exception as e:
            self.logger.log_text(f"Env {self.env_id}: Data load error: {str(e)}")
            raise

        # Curriculum: Limit nodes based on phase
        self.node_list = list(self.hex_to_node.keys()) + list(self.region_to_node.keys())
        if curriculum_phase == 0:
            high_demand_hexes = self.unified_df.sort_values("passenger_demand", ascending=False).head(int(0.1 * len(self.unified_df)))["hex_code"].tolist()
            self.node_list = [n for n in self.node_list if n in high_demand_hexes][:self.max_nodes]
            self.max_steps = 50
        elif curriculum_phase == 1:
            high_demand_hexes = self.unified_df.sort_values("passenger_demand", ascending=False).head(int(0.5 * len(self.unified_df)))["hex_code"].tolist()
            self.node_list = [n for n in self.node_list if n in high_demand_hexes][:self.max_nodes]
            self.max_steps = 75
        else:
            self.node_list = self.node_list[:self.max_nodes]
            self.max_steps = 100

        # Initialize simulation paths
        self.abstreet_map = Path(abstreet_map_file)
        self.abstreet_scenario = Path(abstreet_scenario_file)
        self.sumo_config = Path(sumo_config_file)
        self.abstreet_sim_dir = Path(f"/tmp/abstreet_sim_{self.env_id}")
        self.abstreet_sim_dir.mkdir(exist_ok=True)
        self.abstreet_running = False
        self.sumo_running = False
        self.vehicle_id = f"matatu_{self.env_id}"

        # Start simulations
        self._start_abstreet()
        self._start_sumo()

        # Define spaces
        self.observation_space = spaces.Dict({
            "current_embedding": spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32),
            "traffic_level": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "passenger_count": spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32),
            "edge_congestion": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "hourly_demand": spaces.Box(low=0, high=100, shape=(24,), dtype=np.float32),
            "target_embedding": spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32),
            "place_cells": spaces.Box(low=0, high=1, shape=(self.max_nodes,), dtype=np.float32),
            "goal_node": spaces.Discrete(self.max_nodes)  # For HER
        })
        self.action_space = spaces.Dict({
            "high_level": spaces.Discrete(self.max_nodes),
            "low_level": spaces.Discrete(2)  # 0: Move, 1: Stop
        })

        self.current_node = None
        self.target_node = None
        self.passenger_count = 0
        self.traffic_level = 0.0
        self.edge_congestion = 0.0
        self.step_count = 0
        self.bq_client = bigquery.Client(project="my_project")
        self.max_reward = 1.0  # For reward normalization
        self.place_cell_memory = {}  # Store historical demand per node
        self.forward_error = 0.0  # For curiosity-driven exploration

    def _start_abstreet(self):
        try:
            if not self.abstreet_map.exists():
                raise FileNotFoundError(f"Env {self.env_id}: AStreet map {self.abstreet_map} not found")
            if not self.abstreet_scenario.exists():
                raise FileNotFoundError(f"Env {self.env_id}: AStreet scenario {self.abstreet_scenario} not found")
            self.abstreet_process = subprocess.Popen(
                [
                    "./abstreet",
                    "--simulate",
                    f"--map={self.abstreet_map}",
                    f"--scenario={self.abstreet_scenario}",
                    f"--output={self.abstreet_sim_dir}/state.json"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.abstreet_running = True
            self.logger.log_text(f"Env {self.env_id}: AStreet started successfully")
        except Exception as e:
            self.logger.log_text(f"Env {self.env_id}: AStreet start error: {str(e)}")
            self.abstreet_running = False

    def _start_sumo(self):
        try:
            if not self.sumo_config.exists():
                raise FileNotFoundError(f"Env {self.env_id}: SUMO config {self.sumo_config} not found")
            if "SUMO_HOME" not in os.environ:
                raise EnvironmentError(f"Env {self.env_id}: SUMO_HOME not set")
            self.sumo_cmd = ["sumo", "-c", str(self.sumo_config)]
            traci.start(self.sumo_cmd, label=f"conn_{self.env_id}")
            traci.vehicle.add(self.vehicle_id, routeID="route_0", typeID="bus")
            self.sumo_running = True
            self.logger.log_text(f"Env {self.env_id}: SUMO started successfully")
        except Exception as e:
            self.logger.log_text(f"Env {self.env_id}: SUMO start error: {str(e)}")
            self.sumo_running = False

    def _get_traffic_level(self):
        abstreet_traffic = 0.5
        sumo_traffic = 0.5
        weight_abstreet = 0.3 if self.step_count < 50 else 0.1

        if self.abstreet_running:
            try:
                with open(self.abstreet_sim_dir / "state.json", "r") as f:
                    state = json.load(f)
                abstreet_traffic = min(state.get("average_delay", 30.0) / 60.0, 1.0)
            except Exception as e:
                self.logger.log_text(f"Env {self.env_id}: AStreet state error: {str(e)}")

        if self.sumo_running:
            try:
                edge_id = self.h3_mappings[self.h3_mappings["hex_code"] == self.current_node]["edge_id"].iloc[0]
                if edge_id != "none":
                    travel_time = traci.edge.getTraveltime(edge_id)
                    sumo_traffic = min(travel_time / 60.0, 1.0)
                    self.edge_congestion = min(traci.edge.getLastStepVehicleNumber(edge_id) / 10.0, 1.0)
                else:
                    self.edge_congestion = 0.0
            except Exception as e:
                self.logger.log_text(f"Env {self.env_id}: SUMO traffic error: {str(e)}")

        return (weight_abstreet * abstreet_traffic + (1 - weight_abstreet) * sumo_traffic) if self.abstreet_running or self.sumo_running else np.random.random()

    def _get_passenger_demand(self):
        try:
            passenger_demand = self.unified_df[self.unified_df["hex_code"] == self.current_node]["passenger_demand"].iloc[0]
            hour = pd.Timestamp.now().hour
            hourly_demand = self.unified_df[self.unified_df["hex_code"] == self.current_node][f"hour_{hour}"].iloc[0]
            reservation_query = f"""
            SELECT COUNT(*) as demand
            FROM my_project.my_dataset.reservations
            WHERE hex_code = '{self.current_node}'
            AND booking_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
            """
            reservation_demand = self.bq_client.query(reservation_query).to_dataframe()["demand"].iloc[0]
            demand = max(hourly_demand, reservation_demand, 3.0) + 2 * passenger_demand
            return int(np.clip(np.random.normal(demand, 1.5), 0, 20))
        except Exception as e:
            self.logger.log_text(f"Env {self.env_id}: Passenger demand error: {str(e)}")
            return np.random.randint(0, 5)

    def _get_place_cells(self):
        place_cells = np.zeros(self.max_nodes, dtype=np.float32)
        if self.current_node in self.node_list:
            current_idx = self.node_list.index(self.current_node)
            place_cells[current_idx] = 1.0
            current_tuple = self.hex_to_node.get(self.current_node, self.region_to_node.get(self.current_node))
            for i, node in enumerate(self.node_list):
                if node != self.current_node:
                    node_tuple = self.hex_to_node.get(node, self.region_to_node.get(node))
                    try:
                        distance = nx.shortest_path_length(self.grg, current_tuple, node_tuple, weight="edge_weight")
                        place_cells[i] = np.exp(-distance / self.place_cell_sigma)
                    except nx.NetworkXNoPath:
                        place_cells[i] = 0.0
        return place_cells / (np.sum(place_cells) + 1e-6)  # Normalize

    def _get_intrinsic_reward(self):
        memory_values = np.array(list(self.place_cell_memory.values()))
        memory_variance = np.var(memory_values) if len(memory_values) > 1 else 0.0
        intrinsic_reward = self.intrinsic_weight * memory_variance
        intrinsic_reward += self.intrinsic_weight * self.forward_error
        return intrinsic_reward

    def _get_observation(self):
        embedding = self.h3_embeddings.get(self.current_node, self.region_embeddings.get(self.current_node, np.zeros(64)))
        target_embedding = self.h3_embeddings.get(self.target_node, self.region_embeddings.get(self.target_node, np.zeros(64)))
        hourly_demand = np.array([self.unified_df[self.unified_df["hex_code"] == self.current_node][f"hour_{h}"].iloc[0] if self.current_node in self.unified_df["hex_code"].values else 0.0 for h in range(24)])
        place_cells = self._get_place_cells()
        goal_idx = self.node_list.index(self.target_node) if self.target_node in self.node_list else 0
        return {
            "current_embedding": embedding.astype(np.float32),
            "traffic_level": np.array([self.traffic_level], dtype=np.float32),
            "passenger_count": np.array([self.passenger_count], dtype=np.float32),
            "edge_congestion": np.array([self.edge_congestion], dtype=np.float32),
            "hourly_demand": hourly_demand.astype(np.float32),
            "target_embedding": target_embedding.astype(np.float32),
            "place_cells": place_cells.astype(np.float32),
            "goal_node": goal_idx
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        return 1.0 if achieved_goal == desired_goal else -0.1

    def reset(self):
        if self.abstreet_running:
            self.abstreet_process.terminate()
            self._start_abstreet()
        if self.sumo_running:
            if traci.isLoaded():
                traci.close()
            self._start_sumo()

        hexagon_nodes = [n for n in self.node_list if n in self.hex_to_node]
        demand_weights = [self.unified_df[self.unified_df["hex_code"] == n]["passenger_demand"].iloc[0] if n in self.unified_df["hex_code"].values else 1.0 for n in hexagon_nodes]
        demand_weights = np.array(demand_weights) / np.sum(demand_weights)
        self.current_node = np.random.choice(hexagon_nodes, p=demand_weights)
        self.target_node = self.current_node
        self.passenger_count = self._get_passenger_demand()
        self.traffic_level = self._get_traffic_level()
        self.edge_congestion = 0.0
        self.step_count = 0
        self.visitation_counts = {node: 1 for node in self.node_list}
        self.place_cell_memory = {node: 0.0 for node in self.node_list}
        self.forward_error = 0.0
        obs = self._get_observation()
        return {k: np.expand_dims(v, axis=0) for k, v in obs.items()}  # Batch dimension for SubprocVecEnv

    def step(self, action):
        high_level_action = action["high_level"]
        low_level_action = action["low_level"]
        self.target_node = self.node_list[high_level_action] if high_level_action < len(self.node_list) else self.node_list[0]
        high_level_reward = 0.0
        low_level_reward = 0.0
        done = False
        prev_node = self.current_node

        current_node_tuple = self.hex_to_node.get(self.current_node, self.region_to_node.get(self.current_node))
        target_node_tuple = self.hex_to_node.get(self.target_node, self.region_to_node.get(self.target_node))

        if not self.grg.has_edge(current_node_tuple, target_node_tuple):
            high_level_reward = -2.0
        else:
            hourly_demand = self.unified_df[self.unified_df["hex_code"] == self.target_node][f"hour_{pd.Timestamp.now().hour}"].iloc[0] if self.target_node in self.unified_df["hex_code"].values else 0.0
            high_level_reward += 0.5 * hourly_demand
            prev_demand = self.unified_df[self.unified_df["hex_code"] == self.current_node][f"hour_{pd.Timestamp.now().hour}"].iloc[0] if self.current_node in self.unified_df["hex_code"].values else 0.0
            high_level_reward += 0.2 * (hourly_demand - prev_demand)
            place_cells = self._get_place_cells()
            high_level_reward += 0.1 * place_cells[min(self.node_list.index(self.target_node), self.max_nodes - 1)]
            self.visitation_counts[self.target_node] = self.visitation_counts.get(self.target_node, 1) + 1
            high_level_reward += 0.1 / np.sqrt(self.visitation_counts[self.target_node])

        if self.grg.has_edge(current_node_tuple, target_node_tuple):
            if low_level_action == 1:  # Stop
                passenger_delta = self._get_passenger_demand() - self.passenger_count
                self.passenger_count = max(0, min(20, self.passenger_count + passenger_delta))
                low_level_reward += 2.0 if self.passenger_count > 0 else -1.0
                self.place_cell_memory[self.current_node] = 0.9 * self.place_cell_memory.get(self.current_node, 0.0) + 0.1 * passenger_delta
            else:  # Move
                self.current_node = self.target_node
                travel_time = self.traffic_level + self.edge_congestion
                low_level_reward += 1.0 - 0.3 * travel_time
                place_cells = self._get_place_cells()
                low_level_reward += 0.1 * place_cells[min(self.node_list.index(self.current_node), self.max_nodes - 1)]
            self.traffic_level = self._get_traffic_level()
            self.edge_congestion = 0.0

        intrinsic_reward = self._get_intrinsic_reward()
        reward = 0.6 * high_level_reward + 0.4 * low_level_reward + intrinsic_reward
        self.max_reward = max(self.max_reward, abs(reward))
        reward /= self.max_reward

        self.step_count += 1
        done = self.step_count >= self.max_steps

        achieved_goal = self.node_list.index(self.current_node) if self.current_node in self.node_list else 0
        desired_goal = self.node_list.index(self.target_node) if self.target_node in self.node_list else 0
        her_reward = self.compute_reward(achieved_goal, desired_goal, {})

        info = {
            "passengers": self.passenger_count,
            "traffic": self.traffic_level,
            "congestion": self.edge_congestion,
            "step": self.step_count,
            "high_level_reward": high_level_reward,
            "low_level_reward": low_level_reward,
            "intrinsic_reward": intrinsic_reward,
            "her_reward": her_reward,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
            "prev_node": prev_node,
            "forward_error": self.forward_error,
            "current_node": self.current_node
        }

        obs = self._get_observation()
        return (
            {k: np.expand_dims(v, axis=0) for k, v in obs.items()},
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.bool_),
            [info]
        )

    def close(self):
        if self.abstreet_running:
            self.abstreet_process.terminate()
            self.abstreet_running = False
        if self.sumo_running and traci.isLoaded():
            traci.close()
            self.sumo_running = False

    def set_forward_error(self, error: float):
        self.forward_error = error
        self.logger.log_text(f"Env {self.env_id}: Set forward error to {error}")

    def get_attr(self, attr, indices=None):
        if attr == "current_node":
            return [self.current_node]
        elif attr == "sumo_running":
            return [self.sumo_running]
        elif attr == "vehicle_id":
            return [self.vehicle_id]
        elif attr == "abstreet_running":
            return [self.abstreet_running]
        elif attr == "abstreet_sim_dir":
            return [str(self.abstreet_sim_dir)]
        return getattr(self, attr)

    @staticmethod
    def generate_scenario(unified_df, output_file):
        try:
            scenario = {
                "scenario_name": "nairobi_rush_hour",
                "people": [
                    {
                        "id": f"matatu_{i}",
                        "mode": "Transit",
                        "trip": {
                            "origin": {"lat": row["hex_centroid_y"], "lon": row["hex_centroid_x"]},
                            "destination": {"lat": row["hex_centroid_y"] + 0.01, "lon": row["hex_centroid_x"] + 0.01},
                            "departure_time": "08:00:00",
                            "vehicle_type": "Bus"
                        }
                    } for i, row in unified_df.sample(100).iterrows()
                ]
            }
            with open(output_file, "w") as f:
                json.dump(scenario, f)
            logging.Client().logger("matatu_pipeline").log_text(f"Generated AStreet scenario: {output_file}")
        except Exception as e:
            logging.Client().logger("matatu_pipeline").log_text(f"Scenario generation error: {str(e)}")
            raise

@step
def define_ma3_environment(
    unified_df: pd.DataFrame,
    grg_path: str = "processed_data/grg.pkl",
    abstreet_map_file: str = "gs://my_bucket/input_data/nairobi.bin",
    abstreet_scenario_file: str = "gs://my_bucket/input_data/nairobi_scenario.json",
    sumo_config_file: str = "gs://my_bucket/input_data/nairobi.sumocfg",
    curriculum_phase: int = 0
) -> MatatuEnv:
    """Define the Matatu environment for RL training."""
    logger = logging.Client().logger("matatu_pipeline")

    try:
        scenario_path = "/tmp/nairobi_scenario.json"
        MatatuEnv.generate_scenario(unified_df, scenario_path)

        if abstreet_scenario_file.startswith("gs://"):
            gcs_client = storage.Client(project="my_project")
            bucket = gcs_client.bucket("my_bucket")
            bucket.blob("input_data/nairobi_scenario.json").upload_from_filename(scenario_path)
            abstreet_scenario_file = scenario_path

        env = MatatuEnv(
            grg_path=grg_path,
            abstreet_map_file=abstreet_map_file,
            abstreet_scenario_file=abstreet_scenario_file,
            sumo_config_file=sumo_config_file,
            curriculum_phase=curriculum_phase
        )
        logger.log_text("Defined Matatu environment")
        return env
    except Exception as e:
        logger.log_text(f"Environment definition error: {str(e)}")
        raise
