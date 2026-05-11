#!/bin/bash

# Log everything
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 125840290869.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling Docker image..."

docker pull 125840290869.dkr.ecr.us-east-1.amazonaws.com/yt-chrome-plugin:latest

echo "Stopping existing container if running..."

docker stop yt-chrome-plugin-api || true
docker rm yt-chrome-plugin-api || true

echo "Starting container..."

docker run -d \
  --env-file /home/ubuntu/app/.env \
  -p 80:5000 \
  --name yt-chrome-plugin-api \
  125840290869.dkr.ecr.us-east-1.amazonaws.com/yt-chrome-plugin:latest

echo "Container started successfully."