docker rm -f tfserving_app
docker run --gpus all -p 8500:8500 -p 8501:8501 --name tfserving_app \
--mount type=bind,source=/home/chonlatid/Python/Project/docsegmentation/rest_grpc_services/tfmodels/corner,target=/models/corner \
-e MODEL_NAME=corner -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -t tensorflow/serving:latest-gpu &