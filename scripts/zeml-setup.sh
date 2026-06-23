#!/bin/bash

# Script to register a comprehensive ZenML stack in Google Cloud with Vertex AI and set it as default

# Configuration variables
STACK_NAME="gcp_vertex_stack"
PROJECT_ID="<YOUR_PROJECT_ID>"  # Replace with your Google Cloud project ID
BUCKET_NAME="<your bucket name>"
REGION="africa-south1"  # Matches your bucket's region (Johannesburg)
SERVICE_ACCOUNT="zenml-vertex-account"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
CONTAINER_REGISTRY="gcr.io/${PROJECT_ID}"
METADATA_STORE="zenml-metadata"
SECRETS_MANAGER="zenml-secrets"

# Exit on error
set -e

# Check if ZenML is installed
if ! command -v zenml &> /dev/null; then
    echo "ZenML not found. Please install ZenML with GCP and Vertex AI integrations using 'pip install zenml[gcp,vertex,secrets]'."
    exit 1
fi

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "Google Cloud SDK not found. Please install it from https://cloud.google.com/sdk/docs/install."
    exit 1
fi

# Verify gcloud authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "No active gcloud authentication found. Run 'gcloud auth login' to authenticate."
    exit 1
fi

# Set the Google Cloud project
echo "Setting Google Cloud project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

# Enable required Google Cloud APIs
echo "Enabling required Google Cloud APIs..."
gcloud services enable \
    storage.googleapis.com \
    containerregistry.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com

# Create a service account for ZenML (if it doesn't exist)
echo "Creating/checking service account ${SERVICE_ACCOUNT}..."
if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" &> /dev/null; then
    gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
        --display-name="ZenML Vertex AI Service Account"
else
    echo "Service account ${SERVICE_ACCOUNT_EMAIL} already exists."
fi

# Grant necessary roles to the service account
echo "Granting roles to the service account..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.admin"  # For GCS artifact store
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/containerregistry.admin"  # For Container Registry
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.admin"  # For Secrets Manager
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/aiplatform.user"  # For Vertex AI Custom Code Service
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/aiplatform.serviceAgent"  # For Vertex AI Service Agent

# Generate and download a service account key
echo "Generating service account key..."
KEY_FILE="zenml-gcp-key.json"
gcloud iam service-accounts keys create "${KEY_FILE}" \
    --iam-account="${SERVICE_ACCOUNT_EMAIL}"

# Initialize ZenML (if not already initialized)
echo "Initializing ZenML..."
zenml init

# Register the GCS artifact store
echo "Registering GCS artifact store..."
zenml artifact-store register gcs_store \
    --flavor=gcp \
    --path="gs://${BUCKET_NAME}/zenml-artifacts" \
    --authentication_secret=gcp_service_account

# Register the metadata store (using GCS for simplicity)
echo "Registering GCS metadata store..."
zenml metadata-store register gcs_metadata \
    --flavor=gcp \
    --path="gs://${BUCKET_NAME}/zenml-metadata" \
    --authentication_secret=gcp_service_account

# Register the container registry
echo "Registering Google Container Registry..."
zenml container-registry register gcr_registry \
    --flavor=gcp \
    --uri="${CONTAINER_REGISTRY}"

# Register the secrets manager
echo "Registering GCP Secrets Manager..."
zenml secrets-manager register gcp_secrets \
    --flavor=gcp \
    --project_id="${PROJECT_ID}" \
    --authentication_secret=gcp_service_account

# Register the Vertex AI orchestrator
echo "Registering Vertex AI orchestrator..."
zenml orchestrator register vertex_orchestrator \
    --flavor=vertex \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --service_account="${SERVICE_ACCOUNT_EMAIL}"

# Create a secret for the GCP service account
echo "Creating ZenML secret for GCP service account..."
zenml secret create gcp_service_account \
    --gcp_service_account_credentials=@"${KEY_FILE}"

# Register the ZenML stack
echo "Registering ZenML stack '${STACK_NAME}'..."
zenml stack register "${STACK_NAME}" \
    -a gcs_store \
    -m gcs_metadata \
    -c gcr_registry \
    -s gcp_secrets \
    -o vertex_orchestrator

# Set the stack as default
echo "Setting '${STACK_NAME}' as the default stack..."
zenml stack set "${STACK_NAME}"

# Verify the active stack
echo "Verifying active stack..."
zenml stack describe

# Clean up the service account key file
echo "Cleaning up service account key file..."
rm -f "${KEY_FILE}"

echo "ZenML stack '${STACK_NAME}' successfully registered and set as default!"
