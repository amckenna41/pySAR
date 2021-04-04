#!/bin/bash

### check current version of pip and update, if neccessry ###
# python3 -m pip install --user --upgrade pip

#update Python Path
# export PATH="${PATH}:/root/.local/bin"
# export PYTHONPATH="${PYTHONPATH}:/root/.local/bin"

MODULE="ProAct.main_gcp"
MODEL="PlsReg"
PACKAGE_PATH="ProAct"
STAGING_BUCKET="keras-python-models-2"
REGION="us-central1"
BUCKET_NAME="gs://keras-python-models-2"
RUNTIME_VERSION="2.1"
PYTHON_VERSION="3.7"

JOB_NAME="ProSAR"_"$(date +"%Y%m%d_%H%M")"

JOB_DIR="$BUCKET_NAME/ProtSAR$(date +"%Y%m%d_%H%M")"
SCALE_TIER="CUSTOM"
MASTER_TYPE="n2-standard-4"

MODULE="ProAct.main_gcp"
# MASTER_TYPE="e2-standard-4"
 #submitting Tensorflow training job to Google Cloud
 gcloud ai-platform jobs submit training $JOB_NAME \
     --package-path "ProAct" \
     --module-name $MODULE \
     --staging-bucket $BUCKET_NAME \
     --runtime-version $RUNTIME_VERSION \
     --python-version $PYTHON_VERSION  \
     --job-dir $JOB_DIR \
     --region $REGION \
     --scale-tier $SCALE_TIER \
     --master-machine-type $MASTER_TYPE \
     -- \
     --job_name $JOB_NAME
     --model $MODEL


   # gcloud ai-platform local train \
   #   --module-name $MODULE \
   #   --package-path "ProAct" \
   #   -- \
   #   --job-dir$JOB_NAME
