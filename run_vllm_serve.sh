model="../models/judger"

# vllm serve $model \
#     --dtype auto \
#     --port 33618 \
#     --tensor-parallel-size 4 \
#     --api-key custom-key \
#     --trust-remote-code \
#     --gpu-memory-utilization 0.2

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
    --model-path $model \
    --tp-size 4 \
    --api-key custom-key \
    --port 33618 \
    --mem-fraction-static 0.2
