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

import time
from utils.utils import *
from utils.random_search import grid_choices_random
from utils.grid_search import grid_choices, get_num_grid_choices
from run_agent_parallel import train_PPO, test_rule_based, test_PPO
import sys
import os
from data_collector import DataCollector, POSSIBLE_DATA


def run_grid_search(verbose, num_repeat_experiment, df_path=None, overwrite=True, data_to_collect=POSSIBLE_DATA,
                    MVP_key='waitingTime', save_model=True):
    grid = load_constants('constants/constants-grid.json')

    # Make the grid choice generator
    bases, num_choices = get_num_grid_choices(grid)
    grid_choice_gen = grid_choices(grid, bases)
    for diff_experiment, constants in enumerate(grid_choice_gen):
        data_collector_obj = DataCollector(data_to_collect, MVP_key, constants,
                                           'test' if constants['agent']['agent_type'] == 'rule' else 'eval',
                                           df_path, overwrite if diff_experiment == 0 else False, verbose)

        for same_experiment in range(num_repeat_experiment):
            print(' --- Running experiment {}.{} / {}.{} --- '.format(diff_experiment+1, same_experiment+1,
                                                                      num_choices, num_repeat_experiment))
            if save_model: data_collector_obj.set_save_model_path(
                'models/saved_models/grid_{}-{}.pt'.format(diff_experiment + 1, same_experiment + 1))
            run_experiment(diff_experiment+1, same_experiment+1, constants, data_collector_obj)


def run_random_search(verbose, num_diff_experiments, num_repeat_experiment, allow_duplicates=False, df_path=None,
                      overwrite=True, data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime', save_model=True):
    grid = load_constants('constants/constants-grid.json')

    if not allow_duplicates:
        _, num_choices = get_num_grid_choices(grid)
        num_diff_experiments = min(num_choices, num_diff_experiments)
    # Make grid choice generator
    grid_choice_gen = grid_choices_random(grid, num_diff_experiments)
    for diff_experiment, constants in enumerate(grid_choice_gen):
        data_collector_obj = DataCollector(data_to_collect, MVP_key, constants,
                                           'test' if constants['agent']['agent_type'] == 'rule' else 'eval',
                                           df_path, overwrite if diff_experiment == 0 else False, verbose)

        for same_experiment in range(num_repeat_experiment):
            print(' --- Running experiment {}.{} / {}.{} --- '.format(diff_experiment+1, same_experiment+1,
                                                                      num_diff_experiments, num_repeat_experiment))
            if save_model: data_collector_obj.set_save_model_path('models/saved_models/random_{}-{}.pt'.
                                                                  format(diff_experiment+1, same_experiment+1))
            run_experiment(diff_experiment+1, same_experiment+1, constants, data_collector_obj)


def run_normal(verbose, num_experiments=1, df_path=None, overwrite=True, data_to_collect=POSSIBLE_DATA,
               MVP_key='waitingTime', save_model=True, load_model_file=None):
    # if loading, then dont save
    if load_model_file:
        save_model = False

    if not df_path:
        df_path = 'run-data.xlsx'  # def. path

    # Load constants
    constants = load_constants('constants/constants.json')
    data_collector_obj = DataCollector(data_to_collect, MVP_key, constants,
                                       'test' if constants['agent']['agent_type'] == 'rule' or load_model_file else 'eval',
                                       df_path, overwrite, verbose)

    loaded_model = None
    if load_model_file:
        loaded_model = torch.load('models/saved_models/' + load_model_file)

    for exp in range(num_experiments):
        print(' --- Running experiment {} / {} --- '.format(exp + 1, num_experiments))
        if save_model: data_collector_obj.set_save_model_path('models/saved_models/normal_{}.pt'.format(exp+1))
        run_experiment(exp+1, None, constants, data_collector_obj, loaded_model=loaded_model)


def run_experiment(exp1, exp2, constants, data_collector_obj, loaded_model=None):
    data_collector_obj.start_timer()

    if loaded_model:
        test_PPO(constants, device, data_collector_obj, loaded_model)
    elif constants['agent']['agent_type'] == 'ppo':
        train_PPO(constants, device, data_collector_obj)
    else:
        assert constants['agent']['agent_type'] == 'rule'
        test_rule_based(constants, device, data_collector_obj)

    # Save and Refresh the data_collector
    data_collector_obj.end_timer(printIt=True)
    data_collector_obj.process_data()
    data_collector_obj.print_summary(exp1, exp2)
    data_collector_obj.done_with_experiment()


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

    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    device = torch.device('cpu')

    # df_path = 'run-data.xlsx'

    # print('Num cores: {}'.format(mp.cpu_count()))

    run_normal(verbose=False, num_experiments=3, df_path='run-data.xlsx', overwrite=True,
               data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime', save_model=True, load_model_file=None)

    # run_random_search(verbose=False, num_diff_experiments=800, num_repeat_experiment=3, allow_duplicates=False,
    #                   df_path='run-data.xlsx', overwrite=True, data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime',
    #                   save_model=True)

    # run_grid_search(verbose=False, num_repeat_experiment=3, df_path='run-data.xlsx', overwrite=True,
    #                 data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime', save_model=True)


