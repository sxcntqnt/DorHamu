import pickle
import subprocess
from typing import Dict, Tuple
import pandas as pd
import geopandas as gpd
import networkx as nx
from google.cloud import storage
from google.cloud import logging
from zenml import step
import numpy as np
import os
import traci

@step
def build_goals_relational_graph(
    unified_df: pd.DataFrame,
    h3_embeddings_df: pd.DataFrame,
    osm_embeddings_df: pd.DataFrame,
    osm_gdf: gpd.GeoDataFrame
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, int]]:
    """Build the Goals Relational Graph (GRG) using NetworkX for graph construction and Cloverleaf binary for node embeddings."""
    logger = logging.Client().logger("matatu_pipeline")

    try:
        # Initialize GCS client
        gcs_client = storage.Client(project="my_project")
        bucket = gcs_client.bucket("my_bucket")
        logger.log_text("Initialized GCS client for GRG")
    except Exception as e:
        logger.log_text(f"GCS client error: {str(e)}")
        raise

    # Initialize SUMO for edge weights (optional, if available)
    sumo_running = False
    try:
        sumo_cmd = ["sumo", "-c", "gs://my_bucket/input_data/nairobi.sumocfg"]
        traci.start(sumo_cmd)
        sumo_running = True
        logger.log_text("Started SUMO for edge weights")
    except Exception as e:
        logger.log_text(f"SUMO start error: {str(e)}")

    # Initialize NetworkX graph
    try:
        G = nx.DiGraph()
        logger.log_text("Initialized NetworkX DiGraph")
    except Exception as e:
        logger.log_text(f"Graph initialization error: {str(e)}")
        raise

    # Add nodes and edges
    try:
        hex_to_node = {}
        region_to_node = {}
        node_counter = 0

        # Add hex nodes with demand attributes
        for _, row in unified_df.iterrows():
            hex_code = row["hex_code"]
            G.add_node(
                node_counter,
                hex_code=hex_code,
                type="hex",
                urban_overlap=row["urban_overlap"],
                hourly_demand=[row[f"hour_{h}"] for h in range(24)]
            )
            hex_to_node[hex_code] = node_counter
            node_counter += 1

        # Add region nodes
        for _, row in osm_gdf.iterrows():
            region_id = row["region_id"]
            G.add_node(node_counter, region_id=region_id, type="region")
            region_to_node[region_id] = node_counter
            node_counter += 1

        # Add edges based on spatial proximity
        edge_list = []
        for _, row in unified_df.iterrows():
            hex_code = row["hex_code"]
            hex_node = hex_to_node[hex_code]
            for _, osm_row in osm_gdf.iterrows():
                region_id = osm_row["region_id"]
                region_node = region_to_node[region_id]
                distance = ((row["hex_centroid_x"] - osm_row["geometry"].x)**2 +
                           (row["hex_centroid_y"] - osm_row["geometry"].y)**2)**0.5
                if distance < 0.2:  # Relaxed threshold for denser graph
                    edge_weight = 1.0
                    if sumo_running:
                        edge_id = unified_df[unified_df["hex_code"] == hex_code]["edge_id"].iloc[0] if hex_code in unified_df["hex_code"].values else "none"
                        if edge_id != "none":
                            edge_weight = traci.edge.getTraveltime(edge_id) / 60.0  # Normalize to minutes
                    G.add_edge(hex_node, region_node, weight=edge_weight)
                    edge_list.append(f"{hex_node} {region_node} {edge_weight}")

        logger.log_text("Added nodes and edges to graph")
    except Exception as e:
        logger.log_text(f"Node/edge creation error: {str(e)}")
        raise
    finally:
        if sumo_running:
            traci.close()

    # Write edge list for Cloverleaf
    try:
        edge_list_path = "/tmp/grg_edges.txt"
        with open(edge_list_path, "w") as f:
            f.write("\n".join(edge_list))
        logger.log_text("Wrote edge list for Cloverleaf")
    except Exception as e:
        logger.log_text(f"Edge list write error: {str(e)}")
        raise

    # Run Cloverleaf binary for embeddings
    try:
        output_path = "/tmp/grg_embeddings.txt"
        cmd = [
            "cloverleaf",
            "--input", edge_list_path,
            "--output", output_path,
            "--dims", "130",  # Increased to include centroid coordinates
            "--num-walks", "10000",
            "--hashes", "3",
            "--steps", "0.3",
            "--beta", "0.8"
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.log_text(f"Cloverleaf output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.log_text(f"Cloverleaf execution error: {e.stderr}")
        raise
    except Exception as e:
        logger.log_text(f"Cloverleaf processing error: {str(e)}")
        raise

    # Read Cloverleaf embeddings
    try:
        clover_embeddings = {}
        with open(output_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                node_id = int(parts[0])
                embedding = [float(x) for x in parts[1:]]
                clover_embeddings[node_id] = np.array(embedding)
        logger.log_text("Read Cloverleaf embeddings")
    except Exception as e:
        logger.log_text(f"Embedding read error: {str(e)}")
        raise

    # Create embeddings DataFrame with centroid coordinates
    try:
        grg_embeddings = []
        for node_id, node_data in G.nodes(data=True):
            emb = clover_embeddings.get(node_id, np.zeros(130))
            centroid_x = unified_df[unified_df["hex_code"] == node_data.get("hex_code")]["hex_centroid_x"].iloc[0] if node_data.get("hex_code") in unified_df["hex_code"].values else 0.0
            centroid_y = unified_df[unified_df["hex_code"] == node_data.get("hex_code")]["hex_centroid_y"].iloc[0] if node_data.get("hex_code") in unified_df["hex_code"].values else 0.0
            emb[-2:] = [centroid_x, centroid_y]
            grg_embeddings.append({"node_id": node_id, "embedding": emb.tolist()})
        grg_embeddings_df = pd.DataFrame(grg_embeddings)
        logger.log_text("Generated GRG embeddings with centroids")
    except Exception as e:
        logger.log_text(f"Embedding processing error: {str(e)}")
        raise

    # Merge with input embeddings
    try:
        if not h3_embeddings_df.empty and not osm_embeddings_df.empty:
            grg_embeddings_df = grg_embeddings_df.merge(
                h3_embeddings_df[["hex_code", "embedding"]].rename(columns={"embedding": "h3_embedding"}),
                left_on="node_id",
                right_on=[hex_to_node.get(hc, -1) for hc in h3_embeddings_df["hex_code"]],
                how="left"
            )
            grg_embeddings_df = grg_embeddings_df.merge(
                osm_embeddings_df[["region_id", "embedding"]].rename(columns={"embedding": "osm_embedding"}),
                left_on="node_id",
                right_on=[region_to_node.get(rid, -1) for rid in osm_embeddings_df["region_id"]],
                how="left"
            )
            grg_embeddings_df["embedding"] = grg_embeddings_df.apply(
                lambda row: (row["h3_embedding"] if pd.notnull(row["h3_embedding"]) else
                            row["osm_embedding"] if pd.notnull(row["osm_embedding"]) else
                            row["embedding"]),
                axis=1
            )
        logger.log_text("Merged embeddings")
    except Exception as e:
        logger.log_text(f"Embedding merge error: {str(e)}")
        raise

    # Store artifacts
    try:
        grg_embeddings_df.to_parquet("/tmp/grg_embeddings.parquet")
        bucket.blob("processed_data/grg_embeddings.parquet").upload_from_filename("/tmp/grg_embeddings.parquet")
        with open("/tmp/grg_mappings.pkl", "wb") as f:
            pickle.dump({"hex_to_node": hex_to_node, "region_to_node": region_to_node}, f)
        bucket.blob("processed_data/grg_mappings.pkl").upload_from_filename("/tmp/grg_mappings.pkl")
        with open("/tmp/grg.pkl", "wb") as f:
            pickle.dump({"grg": G, "hex_to_node": hex_to_node, "region_to_node": region_to_node}, f)
        bucket.blob("processed_data/grg.pkl").upload_from_filename("/tmp/grg.pkl")
        logger.log_text("Stored GRG artifacts in GCS")
    except Exception as e:
        logger.log_text(f"Artifact storage error: {str(e)}")
        raise

    logger.log_text("Built Goals Relational Graph")
    return grg_embeddings_df, hex_to_node, region_to_node
