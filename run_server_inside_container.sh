bash run_cluster.sh \
                  vllm/vllm-openai \
                  176.99.131.117 \
                  --head \
                  /bin \
                  --privileged

bash run_cluster.sh \
                  vllm/vllm-openai \
                  192.168.12.210 \
                  --worker \
                  /bin \
                  --privileged

sudo docker exec -it node /bin/bash

vllm serve "Qwen/Qwen2.5-0.5B-Instruct" --pipeline-parallel-size 2 --dtype "float16"

GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1 test.py
GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=1 --rdzv_backend=c10d --rdzv_endpoint=192.168.12.51 test.py