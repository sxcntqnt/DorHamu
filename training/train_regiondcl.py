import tensorflow as tf
import numpy as np
import pandas as pd
import rasterio
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import logging
from zenml import step
from typing import Tuple

@step
def train_regiondcl_model(
    unified_df: pd.DataFrame,
    raster_data_path: str,
    pairs_query: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train RegionDCL model and generate embeddings for H3 and OSM regions."""
    # Initialize logging
    logger = logging.Client().logger("matatu_pipeline")

    # Set up BigQuery and GCS clients
    try:
        bq_client = bigquery.Client(project="my_project")
        gcs_client = storage.Client(project="my_project")
        bucket = gcs_client.bucket("my_bucket")
        logger.log_text("Initialized BigQuery and GCS clients")
    except Exception as e:
        logger.log_text(f"Client initialization error: {str(e)}")
        raise

    # Load contrastive pairs
    try:
        pairs_df = bq_client.query(pairs_query).to_dataframe()
        logger.log_text("Loaded contrastive pairs from BigQuery")
    except Exception as e:
        logger.log_text(f"Contrastive pairs query error: {str(e)}")
        raise

    # Load raster data
    try:
        if raster_data_path.startswith("gs://"):
            bucket_name, blob_name = raster_data_path.replace("gs://", "").split("/", 1)
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            local_path = "/tmp/raster_data.tif"
            blob.download_to_filename(local_path)
        else:
            local_path = raster_data_path
        with rasterio.open(local_path) as src:
            raster_data = src.read()
        logger.log_text("Loaded raster data")
    except Exception as e:
        logger.log_text(f"Raster data load error: {str(e)}")
        raise

    # Define RegionDCL model
    def build_regiondcl_model(input_shape=(64, 64, 3), embedding_dim=64):
        try:
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            embeddings = tf.keras.layers.Dense(embedding_dim, activation=None)(x)
            model = tf.keras.Model(inputs, embeddings)
            logger.log_text("Built RegionDCL model")
            return model
        except Exception as e:
            logger.log_text(f"Model building error: {str(e)}")
            raise

    # Contrastive loss function
    def contrastive_loss(anchor, positive, negative, margin=1.0):
        try:
            pos_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
            neg_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
            loss = tf.maximum(pos_distance - neg_distance + margin, 0.0)
            return tf.reduce_mean(loss)
        except Exception as e:
            logger.log_text(f"Contrastive loss error: {str(e)}")
            raise

    # Prepare training data
    def prepare_training_data(pairs_df, unified_df, raster_data):
        try:
            anchor_data = []
            positive_data = []
            negative_data = []
            for _, row in pairs_df.iterrows():
                anchor_hex = row["anchor_hex"]
                positive_hex = row["positive_hex"]
                negative_hex = row["negative_hex"]
                for hex_code in [anchor_hex, positive_hex, negative_hex]:
                    if hex_code in unified_df["hex_code"].values:
                        hex_row = unified_df[unified_df["hex_code"] == hex_code].iloc[0]
                        x, y = int(hex_row["hex_centroid_x"]), int(hex_row["hex_centroid_y"])
                        patch = raster_data[:, x-32:x+32, y-32:y+32]
                        if patch.shape == (3, 64, 64):
                            if hex_code == anchor_hex:
                                anchor_data.append(patch)
                            elif hex_code == positive_hex:
                                positive_data.append(patch)
                            elif hex_code == negative_hex:
                                negative_data.append(patch)
            return (np.array(anchor_data), np.array(positive_data), np.array(negative_data))
        except Exception as e:
            logger.log_text(f"Training data preparation error: {str(e)}")
            raise

    # Train model
    try:
        model = build_regiondcl_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        anchor_data, positive_data, negative_data = prepare_training_data(pairs_df, unified_df, raster_data)
        
        @tf.function
        def train_step(anchor, positive, negative):
            with tf.GradientTape() as tape:
                anchor_emb = model(anchor, training=True)
                positive_emb = model(positive, training=True)
                negative_emb = model(negative, training=True)
                loss = contrastive_loss(anchor_emb, positive_emb, negative_emb)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        for epoch in range(10):
            loss = train_step(anchor_data, positive_data, negative_data)
            logger.log_text(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
        
        model.save("/tmp/regiondcl_model")
        bucket.blob("models/regiondcl_model").upload_from_filename("/tmp/regiondcl_model")
        logger.log_text("Trained and saved RegionDCL model")
    except Exception as e:
        logger.log_text(f"Model training error: {str(e)}")
        raise

    # Generate embeddings
    def generate_embeddings(model, unified_df, raster_data):
        try:
            h3_embeddings = []
            osm_embeddings = []
            for _, row in unified_df.iterrows():
                hex_code = row["hex_code"]
                x, y = int(row["hex_centroid_x"]), int(row["hex_centroid_y"])
                patch = raster_data[:, x-32:x+32, y-32:y+32]
                if patch.shape == (3, 64, 64):
                    embedding = model.predict(np.expand_dims(patch, axis=0))
                    h3_embeddings.append({"hex_code": hex_code, "embedding": embedding[0].tolist()})
                    if "region_id" in row and row["region_id"] != -1:
                        osm_embeddings.append({"region_id": row["region_id"], "embedding": embedding[0].tolist()})
            h3_embeddings_df = pd.DataFrame(h3_embeddings)
            osm_embeddings_df = pd.DataFrame(osm_embeddings)
            logger.log_text("Generated RegionDCL embeddings")
            return h3_embeddings_df, osm_embeddings_df
        except Exception as e:
            logger.log_text(f"Embedding generation error: {str(e)}")
            raise

    try:
        h3_embeddings_df, osm_embeddings_df = generate_embeddings(model, unified_df, raster_data)
        h3_embeddings_df.to_parquet("/tmp/h3_embeddings.parquet")
        osm_embeddings_df.to_parquet("/tmp/osm_embeddings.parquet")
        bucket.blob("processed_data/h3_embeddings.parquet").upload_from_filename("/tmp/h3_embeddings.parquet")
        bucket.blob("processed_data/osm_embeddings.parquet").upload_from_filename("/tmp/osm_embeddings.parquet")
        logger.log_text("Stored RegionDCL embeddings in GCS")
    except Exception as e:
        logger.log_text(f"Embedding storage error: {str(e)}")
        raise

    return h3_embeddings_df, osm_embeddings_df
