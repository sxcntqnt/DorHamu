import folium
import pandas as pd
import numpy as np
import json
import pickle
import time
import requests
from google.cloud import pubsub_v1
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import logging
from google.cloud import monitoring_v3
from redis import Redis
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from zenml import step
from typing import Dict
from .define_ma3_environment import MatatuEnv

@step
def evaluate_and_deploy_api(
    unified_df: pd.DataFrame,
    high_level_policy: PPO,
    low_level_policy: PPO,
    n_envs: int = 4
) -> Dict[str, pd.DataFrame]:
    """Evaluate high-level and low-level PPO policies, visualize routes, and prepare data for API deployment."""
    # Initialize clients
    logger = logging.Client().logger("matatu_pipeline")
    monitoring_client = monitoring_v3.MetricServiceClient()
    bq_client = bigquery.Client(project="my_project")
    gcs_client = storage.Client(project="my_project")
    bucket = gcs_client.bucket("my_bucket")

    # Load data
    try:
        grg_embeddings_df = pd.read_gbq("SELECT hex_code, embedding, type FROM my_project.my_dataset.grg_embeddings")
        h3_mappings = pd.read_gbq("SELECT hex_code, road_id, edge_id, passenger_demand FROM my_project.my_dataset.h3_mappings")
        logger.log_text("Loaded GRG embeddings and H3 mappings")
    except Exception as e:
        logger.log_text(f"Data load error: {str(e)}")
        raise

    # Create environment
    def make_env():
        try:
            env = MatatuEnv(
                grg_path="processed_data/grg.pkl",
                abstreet_map_file="gs://my_bucket/input_data/nairobi.bin",
                abstreet_scenario_file="gs://my_bucket/input_data/nairobi_scenario.json",
                sumo_config_file="gs://my_bucket/input_data/nairobi.sumocfg"
            )
            logger.log_text("Created MatatuEnv")
            return env
        except Exception as e:
            logger.log_text(f"Environment creation error: {str(e)}")
            raise

    # Evaluation function
    def evaluate_policy(high_level_model, low_level_model, env, n_envs=4, n_episodes=10):
        results = {
            "rewards": [[] for _ in range(n_envs)],
            "passengers_served": [[] for _ in range(n_envs)],
            "travel_time": [[] for _ in range(n_envs)],
            "wait_time": [[] for _ in range(n_envs)],
            "fuel_efficiency": [[] for _ in range(n_envs)],
            "modal_split": [[] for _ in range(n_envs)],
            "intrinsic_reward": [[] for _ in range(n_envs)],
            "her_reward": [[] for _ in range(n_envs)],
            "forward_error": [[] for _ in range(n_envs)],
            "high_level_reward": [[] for _ in range(n_envs)],
            "low_level_reward": [[] for _ in range(n_envs)],
            "routes": [[] for _ in range(n_envs)]
        }

        for episode in range(n_episodes):
            obs = env.reset()
            dones = [False] * n_envs
            episode_rewards = [0.0] * n_envs
            passengers_served = [0.0] * n_envs
            total_travel_times = [0.0] * n_envs
            wait_times = [0.0] * n_envs
            intrinsic_rewards = [0.0] * n_envs
            her_rewards = [0.0] * n_envs
            forward_errors = [0.0] * n_envs
            high_level_rewards = [0.0] * n_envs
            low_level_rewards = [0.0] * n_envs
            steps = [0] * n_envs
            routes = [[env.get_attr("current_node")[i]] for i in range(n_envs)]

            # Track reservation demand for wait time
            reservation_query = f"""
            SELECT hex_code, COUNT(*) as demand
            FROM my_project.my_dataset.reservations
            WHERE booking_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
            GROUP BY hex_code
            """
            try:
                reservation_demand = bq_client.query(reservation_query).to_dataframe()
            except Exception as e:
                logger.log_text(f"Reservation query error: {str(e)}")
                reservation_demand = pd.DataFrame({"hex_code": [], "demand": []})

            while not all(dones):
                high_level_action, _ = high_level_model.predict(obs)
                low_level_action, _ = low_level_model.predict(obs)
                action = {
                    "high_level": high_level_action,
                    "low_level": low_level_action
                }

                next_obs, rewards, dones, infos = env.step(action)

                for i in range(n_envs):
                    if not dones[i]:
                        episode_rewards[i] += rewards[i]
                        passengers_served[i] += infos[i]["passengers"] if infos[i]["passengers"] > 0 else 0
                        steps[i] += 1
                        routes[i].append(env.get_attr("current_node")[i])

                        # Travel time
                        travel_time = infos[i]["traffic"] + infos[i]["congestion"]
                        total_travel_times[i] += travel_time

                        # Wait time
                        current_node = env.get_attr("current_node")[i]
                        node_demand = reservation_demand[reservation_demand["hex_code"] == current_node]["demand"].iloc[0] if current_node in reservation_demand["hex_code"].values else 0
                        wait_times[i] += 1.0 / (node_demand + 1e-6)

                        # Advanced RL metrics
                        intrinsic_rewards[i] += infos[i]["intrinsic_reward"]
                        her_rewards[i] += infos[i]["her_reward"]
                        forward_errors[i] += infos[i]["forward_error"]
                        high_level_rewards[i] += infos[i]["high_level_reward"]
                        low_level_rewards[i] += infos[i]["low_level_reward"]

                obs = next_obs
                if any(dones):
                    for i in range(n_envs):
                        if dones[i]:
                            routes[i].append(env.get_attr("current_node")[i])

                # Fuel efficiency from SUMO
                fuel_efficiencies = [0.0] * n_envs
                for i in range(n_envs):
                    if env.get_attr("sumo_running")[i]:
                        try:
                            fuel = traci.vehicle.getFuelConsumption(env.get_attr("vehicle_id")[i])
                            fuel_efficiencies[i] = 1000 / (fuel + 1e-6)  # km/L
                        except Exception as e:
                            logger.log_text(f"SUMO fuel efficiency error for env {i}: {str(e)}")

                # Modal split from AStreet
                modal_splits = [0.0] * n_envs
                for i in range(n_envs):
                    if env.get_attr("abstreet_running")[i]:
                        try:
                            with open(env.get_attr("abstreet_sim_dir")[i] / "state.json", "r") as f:
                                state = json.load(f)
                            modal_splits[i] = state.get("bus_trips", 0) / max(state.get("total_trips", 1), 1)
                        except Exception as e:
                            logger.log_text(f"ABStreet modal split error for env {i}: {str(e)}")

                # Store episode results
                if all(dones):
                    for i in range(n_envs):
                        results["rewards"][i].append(episode_rewards[i])
                        results["passengers_served"][i].append(passengers_served[i])
                        results["travel_time"][i].append(total_travel_times[i])
                        results["wait_time"][i].append(wait_times[i])
                        results["fuel_efficiency"][i].append(fuel_efficiencies[i])
                        results["modal_split"][i].append(modal_splits[i])
                        results["intrinsic_reward"][i].append(intrinsic_rewards[i])
                        results["her_reward"][i].append(her_rewards[i])
                        results["forward_error"][i].append(forward_errors[i])
                        results["high_level_reward"][i].append(high_level_rewards[i])
                        results["low_level_reward"][i].append(low_level_rewards[i])
                        results["routes"][i].append(routes[i])

        # Aggregate results
        aggregated_results = {
            key: [item for sublist in values for item in sublist] if key != "routes" else [route for env_routes in values for route in env_routes]
            for key, values in results.items()
        }
        return aggregated_results

    # Evaluate policies
    try:
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
        evaluation_results = evaluate_policy(high_level_policy, low_level_policy, env, n_envs=n_envs)
        logger.log_text("Evaluated high-level and low-level policies across multiple environments")
    except Exception as e:
        logger.log_text(f"Evaluation error: {str(e)}")
        raise
    finally:
        env.close()

    # Log metrics to Google Cloud Monitoring
    def log_metrics(results):
        project_name = f"projects/my_project"
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/matatu/evaluation"
        series.resource.type = "global"
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        interval = monitoring_v3.TimeInterval({"end_time": {"seconds": seconds, "nanos": nanos}})

        for metric_name, values in results.items():
            if metric_name != "routes":
                point = series.points.add()
                point.interval.CopyFrom(interval)
                point.value.double_value = np.mean(values)
                point.metric.labels["metric"] = metric_name
                point.metric.labels["mean"] = str(np.mean(values))
                point.metric.labels["std"] = str(np.std(values))
                monitoring_client.create_time_series(name=project_name, time_series=[series])

        logger.log_text("Logged evaluation metrics to Google Cloud Monitoring")

    try:
        log_metrics(evaluation_results)
    except Exception as e:
        logger.log_text(f"Monitoring error: {str(e)}")

    # Visualize routes
    def visualize_routes(routes, unified_df, n_envs=4):
        try:
            m = None
            colors = ["red", "blue", "green", "purple"]
            for env_idx in range(n_envs):
                route = routes[env_idx]
                route_points = []
                for node in route:
                    if node in unified_df["hex_code"].values:
                        hex_row = unified_df[unified_df["hex_code"] == node].iloc[0]
                        route_points.append((hex_row["hex_centroid_y"], hex_row["hex_centroid_x"]))

                if not route_points:
                    logger.log_text(f"No valid route points for environment {env_idx}")
                    continue

                if m is None:
                    m = folium.Map(location=route_points[0], zoom_start=12)

                folium.PolyLine(
                    route_points,
                    color=colors[env_idx % len(colors)],
                    weight=2.5,
                    opacity=0.8,
                    popup=f"Environment {env_idx}"
                ).add_to(m)
                folium.Marker(
                    route_points[0],
                    icon=folium.Icon(color="green"),
                    popup=f"Start Env {env_idx}"
                ).add_to(m)
                folium.Marker(
                    route_points[-1],
                    icon=folium.Icon(color="red"),
                    popup=f"End Env {env_idx}"
                ).add_to(m)

            if m is None:
                raise ValueError("No valid routes to visualize")

            m.save("/tmp/route_visualization.html")
            bucket.blob("visualizations/route_visualization.html").upload_from_filename("/tmp/route_visualization.html")
            logger.log_text("Saved multi-environment route visualization to GCS")
        except Exception as e:
            logger.log_text(f"Route visualization error: {str(e)}")
            raise

    try:
        visualize_routes(evaluation_results["routes"][:n_envs], unified_df, n_envs)
    except Exception as e:
        logger.log_text(f"Route visualization error: {str(e)}")
        raise

    # Save evaluation results
    try:
        eval_df = pd.DataFrame({
            "episode": range(len(evaluation_results["rewards"])),
            "reward": evaluation_results["rewards"],
            "passengers_served": evaluation_results["passengers_served"],
            "travel_time": evaluation_results["travel_time"],
            "wait_time": evaluation_results["wait_time"],
            "fuel_efficiency": evaluation_results["fuel_efficiency"],
            "modal_split": evaluation_results["modal_split"],
            "intrinsic_reward": evaluation_results["intrinsic_reward"],
            "her_reward": evaluation_results["her_reward"],
            "forward_error": evaluation_results["forward_error"],
            "high_level_reward": evaluation_results["high_level_reward"],
            "low_level_reward": evaluation_results["low_level_reward"]
        })
        eval_df.to_gbq(
            "my_project.my_dataset.evaluation_results",
            if_exists="replace",
            table_schema=[
                {"name": "episode", "type": "INTEGER"},
                {"name": "reward", "type": "FLOAT64"},
                {"name": "passengers_served", "type": "FLOAT64"},
                {"name": "travel_time", "type": "FLOAT64"},
                {"name": "wait_time", "type": "FLOAT64"},
                {"name": "fuel_efficiency", "type": "FLOAT64"},
                {"name": "modal_split", "type": "FLOAT64"},
                {"name": "intrinsic_reward", "type": "FLOAT64"},
                {"name": "her_reward", "type": "FLOAT64"},
                {"name": "forward_error", "type": "FLOAT64"},
                {"name": "high_level_reward", "type": "FLOAT64"},
                {"name": "low_level_reward", "type": "FLOAT64"}
            ]
        )
        logger.log_text("Stored evaluation results in BigQuery")
    except Exception as e:
        logger.log_text(f"Evaluation storage error: {str(e)}")
        raise

    # Package offline data
    def package_offline_data():
        try:
            blob = bucket.blob("processed_data/grg.pkl")
            blob.download_to_filename("/tmp/grg.pkl")
            with open("/tmp/grg.pkl", "rb") as f:
                grg_data = pickle.load(f)
            grg = grg_data["grg"]

            offline_data = {
                "grg_embeddings": grg_embeddings_df.to_dict(),
                "grg_nodes": list(grg.nodes(data=True)),
                "grg_edges": [(edge[0], edge[1], edge[2].get("edge_weight", 1.0)) for edge in grg.edges(data=True)],
                "h3_mappings": h3_mappings.to_dict(),
                "unified_features": unified_df[["hex_code", "hex_centroid_x", "hex_centroid_y", "passenger_demand"] + [f"hour_{h}" for h in range(24)]].to_dict()
            }
            with open("/tmp/offline_data.pkl", "wb") as f:
                pickle.dump(offline_data, f)
            bucket.blob("processed_data/offline_data.pkl").upload_from_filename("/tmp/offline_data.pkl")
            logger.log_text("Packaged offline data to GCS")
        except Exception as e:
            logger.log_text(f"Offline data packaging error: {str(e)}")
            raise

    try:
        package_offline_data()
    except Exception as e:
        logger.log_text(f"Offline data packaging error: {str(e)}")
        raise

    return {"evaluation_results": eval_df}
