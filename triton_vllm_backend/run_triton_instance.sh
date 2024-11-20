docker run --gpus all -it --net=host --rm -P --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:24.05-vllm-python-py3 tritonserver --model-store ./model_repository
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:24.05-py3-sdk bash