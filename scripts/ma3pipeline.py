import os
import pickle
import pandas as pd
from google.cloud import storage
from google.cloud import logging
from zenml import pipeline
from zenml.steps.entrypoint_function_utils import StepArtifact
from zenml.artifacts.utils import load_artifact

from step.load_datasets import load_datasets
from step.spatial_joins import perform_spatial_joins
from step.enhance_features import enhance_feature_set
from step.train_regiondcl import train_regiondcl_model
from step.build_grg import build_goals_relational_graph
from step.define_ma3_environment import define_ma3_environment
from step.train_rl_policy import train_hierarchical_rl_policy
from step.evaluate_ai import evaluate_and_deploy_api

@pipeline
def matatu_route_finding_pipeline(n_envs: int = 4):
    """Matatu Route-Finding AI with RegionDCL Embeddings pipeline."""
    # Initialize clients
    logger = logging.Client().logger("matatu_pipeline")
    gcs_client = storage.Client(project="my_project")
    bucket = gcs_client.bucket("my_bucket")
    logger.log_text("Initialized matatu_route_finding_pipeline")

    # Step 1: Load datasets
    african_gdf, osm_gdf, h3_gdf, unified_df = load_datasets()
    logger.log_text("Completed load_datasets step")

    # Step 2: Perform spatial joins
    unified_df = perform_spatial_joins(african_gdf, osm_gdf, h3_gdf)
    logger.log_text("Completed perform_spatial_joins step")

    # Step 3: Enhance feature set
    unified_df = enhance_feature_set(unified_df)
    logger.log_text("Completed enhance_feature_set step")

    # Step 4: Train RegionDCL model
    h3_embeddings_df, osm_embeddings_df = train_regiondcl_model(
        unified_df=unified_df,
        raster_data_path="input_data/nairobi_raster.tif",
        pairs_query="""
        SELECT anchor_hex, positive_hex, negative_hex
        FROM my_project.my_dataset.contrastive_pairs
        WHERE anchor_hex IS NOT NULL
        """
    )
    logger.log_text("Completed train_regiondcl_model step")

    # Step 5: Build Goals Relational Graph (GRG)
    grg_embeddings_df, hex_to_node, region_to_node = build_goals_relational_graph(
        unified_df=unified_df,
        h3_embeddings_df=h3_embeddings_df,
        osm_embeddings_df=osm_embeddings_df,
        osm_gdf=osm_gdf
    )
    logger.log_text("Completed build_goals_relational_graph step")

    # Store GRG outputs
    try:
        # Log artifact details for debugging
        logger.log_text(f"grg_embeddings_df type: {type(grg_embeddings_df)}")
        logger.log_text(f"grg_embeddings_df attributes: {dir(grg_embeddings_df)}")

        # Handle StepArtifact
        if isinstance(grg_embeddings_df, pd.DataFrame):
            logger.log_text("grg_embeddings_df is already a pandas.DataFrame")
        elif isinstance(grg_embeddings_df, StepArtifact):
            logger.log_text("grg_embeddings_df is a ZenML StepArtifact, resolving it")
            try:
                grg_embeddings_df = load_artifact(grg_embeddings_df)
                logger.log_text(f"Loaded grg_embeddings_df type: {type(grg_embeddings_df)}")
                if not isinstance(grg_embeddings_df, pd.DataFrame):
                    raise TypeError(f"Loaded artifact is not a pandas.DataFrame, got {type(grg_embeddings_df)}")
            except Exception as e:
                logger.log_text(f"Error loading grg_embeddings_df: {str(e)}")
                raise
        else:
            raise TypeError(f"Expected pandas.DataFrame or ZenML StepArtifact, got {type(grg_embeddings_df)}")

        # Store DataFrame
        grg_embeddings_df.to_parquet("/tmp/grg_embeddings.parquet", engine="pyarrow")
        bucket.blob("processed_data/grg_embeddings.parquet").upload_from_filename("/tmp/grg_embeddings.parquet")
        with open("/tmp/grg_mappings.pkl", "wb") as f:
            pickle.dump({"hex_to_node": hex_to_node, "region_to_node": region_to_node}, f)
        bucket.blob("processed_data/grg_mappings.pkl").upload_from_filename("/tmp/grg_mappings.pkl")
        logger.log_text("Stored GRG embeddings and mappings to GCS")
    except Exception as e:
        logger.log_text(f"GRG storage error: {str(e)}")
        raise

    # Step 6: Define Matatu environment
    env = define_ma3_environment(unified_df=unified_df, curriculum_phase=0)
    logger.log_text("Completed define_ma3_environment step")

    # Store environment attributes
    try:
        with open("/tmp/matatu_env.pkl", "wb") as f:
            pickle.dump({
                "node_list": env.node_list,
                "observation_space": str(env.observation_space),
                "action_space": str(env.action_space),
                "curriculum_phase": env.curriculum_phase,
                "forward_error": env.forward_error
            }, f)
        bucket.blob("processed_data/matatu_env.pkl").upload_from_filename("/tmp/matatu_env.pkl")
        logger.log_text("Stored Matatu environment attributes to GCS")
    except Exception as e:
        logger.log_text(f"Environment storage error: {str(e)}")
        raise

    # Step 7: Train hierarchical RL policy
    high_level_policy, low_level_policy = train_hierarchical_rl_policy(
        unified_df=unified_df,
        n_envs=n_envs
    )
    logger.log_text("Completed train_hierarchical_rl_policy step")

    # Step 8: Evaluate and deploy API
    evaluation_results = evaluate_and_deploy_api(
        unified_df=unified_df,
        high_level_policy=high_level_policy,
        low_level_policy=low_level_policy,
        n_envs=n_envs
    )
    logger.log_text("Completed evaluate_and_deploy_api step")

    # Store evaluation results and advanced metrics
    try:
        eval_df = evaluation_results["evaluation_results"]
        logger.log_text(f"eval_df type: {type(eval_df)}")
        logger.log_text(f"eval_df attributes: {dir(eval_df)}")

        # Handle StepArtifact
        if isinstance(eval_df, pd.DataFrame):
            logger.log_text("eval_df is already a pandas.DataFrame")
        elif isinstance(eval_df, StepArtifact):
            logger.log_text("eval_df is a ZenML StepArtifact, resolving it")
            try:
                eval_df = load_artifact(eval_df)
                logger.log_text(f"Loaded eval_df type: {type(eval_df)}")
                if not isinstance(eval_df, pd.DataFrame):
                    raise TypeError(f"Loaded artifact is not a pandas.DataFrame, got {type(eval_df)}")
            except Exception as e:
                logger.log_text(f"Error loading eval_df: {str(e)}")
                raise
        else:
            raise TypeError(f"Expected pandas.DataFrame or ZenML StepArtifact, got {type(eval_df)}")

        eval_df.to_parquet("/tmp/evaluation_results.parquet", engine="pyarrow")
        bucket.blob("processed_data/election_results.parquet").upload_from_filename("/tmp/evaluation_results.parquet")
        advanced_metrics_df = eval_df[[
            "intrinsic_reward", "her_reward", "forward_error",
            "high_level_reward", "low_level_reward"
        ]]
        advanced_metrics_df.to_parquet("/tmp/advanced_metrics.parquet", engine="pyarrow")
        bucket.blob("processed_data/advanced_metrics.parquet").upload_from_filename("/tmp/advanced_metrics.parquet")
        logger.log_text("Stored evaluation results and advanced metrics to GCS")
    except Exception as e:
        logger.log_text(f"Evaluation storage error: {str(e)}")
        raise

if __name__ == "__main__":
    # Initialize ZenML repository (if not already initialized)
    if not os.path.exists(".zen"):
        os.system("zenml init")

    # Run the pipeline
    matatu_route_finding_pipeline()
