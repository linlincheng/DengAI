#!/usr/bin/env bash

# build base image
docker build -t dengai_project .

docker run -v "$(pwd)/.":/app/. -it dengai_project  