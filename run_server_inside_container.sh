bash run_cluster.sh \
                  vllm/vllm-openai \
                  192.168.13.220 \
                  --head \
                  /bin \
                  --privileged

bash run_cluster.sh \
                  vllm/vllm-openai \
                  192.168.13.220 \
                  --worker \
                  /bin \
                  --privileged

sudo docker exec -it node /bin/bash

vllm serve "Qwen/Qwen2.5-0.5B-Instruct" --pipeline-parallel-size 2 --dtype "float16" # check
 
vllm serve "Qwen/Qwen2.5-0.5B-Instruct" --worker-use-ray --dtype "float16" # error nccl

vllm serve "Qwen/Qwen2.5-0.5B-Instruct" --distributed-executor-backend "mp"  --dtype "float16" # check

vllm serve "Qwen/Qwen2.5-Coder-7B-Instruct" --distributed-executor-backend "mp"  --dtype "float16" #error cuda memory

vllm serve "Qwen/Qwen2.5-Coder-7B-Instruct" --pipeline-parallel-size 2  --dtype "float16" #error cuda memory

GLOO_SOCKET_IFNAME=enp8s0 NCCL_SOCKET_IFNAME=enp8s0 NCCL_DEBUG=INFO vllm serve "Qwen/Qwen2.5-Coder-7B-Instruct" --pipeline-parallel-size 2  --dtype "float16" # stable

GLOO_SOCKET_IFNAME=ens4 NCCL_SOCKET_IFNAME=ens4 NCCL_DEBUG=INFO vllm serve "Qwen/Qwen2.5-Coder-3B-Instruct" --pipeline-parallel-size 2  --dtype "float16" # for rent servers

# debug
GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=1 --rdzv_backend=c10d --rdzv_endpoint=172.19.0.177 test.py
GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=1 --rdzv_backend=c10d --rdzv_endpoint=172.19.0.177 test.py