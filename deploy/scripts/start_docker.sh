#!/bin/bash

# Log everything
exec > /home/ubuntu/start_docker.log 2>&1

echo "Setting DAGSHUB token..."

export DAGSHUB_TOKEN="YOUR_DAGSHUB_TOKEN"

echo "Logging in to ECR..."

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 125840290869.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling Docker image..."

docker pull 125840290869.dkr.ecr.us-east-1.amazonaws.com/yt-chrome-plugin:latest

echo "Checking existing container..."

if [ "$(docker ps -q -f name=yt-chrome-plugin-api)" ]; then
    docker stop yt-chrome-plugin-api
fi

if [ "$(docker ps -aq -f name=yt-chrome-plugin-api)" ]; then
    docker rm yt-chrome-plugin-api
fi

echo "Starting container..."

docker run -d \
  -p 80:5000 \
  -e DAGSHUB_TOKEN=$DAGSHUB_TOKEN \
  --name yt-chrome-plugin-api \
  125840290869.dkr.ecr.us-east-1.amazonaws.com/yt-chrome-plugin:latest

echo "Container started successfully."