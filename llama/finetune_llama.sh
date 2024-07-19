# This is a script for finetuning Megatron-LM llama2 and llama3 model (default platform: 16 * H20 GPU, 2 nodes)
# repository: https://github.com/NVIDIA/Megatron-LM.git
# branch：git checkout 86850db
# Setting the environment variables
# If using docker swarm for distributed training, please set NCCL_SOCKET_IFNAME, otherwise unset it
# Important arguments need to be set manually and checked:
# TP: megatron tensor model parallel size
# PP: megatron pipeline model parallel size
# MODEL_NAME: llama model name
# SRC_PATH: pretrain code path
# MODEL_BASE_PATH: huggingface format model weights dir
# DATA_PATH: preprocess dataset path for finetuning
# TOKENIZER_PATH: llama model tokenizer path
# CKPT_LOAD_PATH: llama model megatron checkpoint load path

export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export TORCH_CUDA_ARCH_LIST=Hooper

# Distributed training variables
NNODES=2
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6543
MASTER_ADDR="10.0.1.6"

# Parallelism variables
TP=8
PP=2
DP=$((${GPU_NUM}/${TP}/${PP}))

# Network name variables
MODEL_NAME=llama3-8B

if   [[ ${MODEL_NAME} == llama2-7B ]]; then 
       HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5; ROTARY_BASE=10000; MAKE_VOCAB_SIZE_DIVISBLE_BY=1; TOKENIZER_TYPE=Llama2Tokenizer;
elif [[ ${MODEL_NAME} == llama2-13B ]]; then
       HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5; ROTARY_BASE=10000; MAKE_VOCAB_SIZE_DIVISBLE_BY=1; TOKENIZER_TYPE=Llama2Tokenizer;
elif [[ ${MODEL_NAME} == llama2-70B ]]; then
       HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5; ROTARY_BASE=10000; MAKE_VOCAB_SIZE_DIVISBLE_BY=1; TOKENIZER_TYPE=Llama2Tokenizer;
elif [[ ${MODEL_NAME} == llama3-8B ]]; then 
       HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=8;  NUM_LAYERS=32; FFN_HIDDEN_SIZE=14336; NORM_EPS=1e-5; ROTARY_BASE=500000; MAKE_VOCAB_SIZE_DIVISBLE_BY=16128; TOKENIZER_TYPE=Llama3Tokenizer;
elif [[ ${MODEL_NAME} == llama3-70B ]]; then
       HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5; ROTARY_BASE=500000; MAKE_VOCAB_SIZE_DIVISBLE_BY=16128; TOKENIZER_TYPE=Llama3Tokenizer;
elif [[ ${MODEL_NAME} == llama2-tiny ]]; then
       HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5; ROTARY_BASE=10000; MAKE_VOCAB_SIZE_DIVISBLE_BY=1; TOKENIZER_TYPE=Llama2Tokenizer;
else echo "invalid MODEL_NAME: ${MODEL_NAME}"; exit 1
fi

# base path
SRC_PATH=/workspace/megatron/pretrain_gpt.py
MODEL_BASE_PATH=/workspace/model_weights/llama3-8b
DATA_PATH=/workspace/dataset/finetune_dataset/llama3-8b/alpaca_text_document
TOKENIZER_PATH=${MODEL_BASE_PATH}/original/tokenizer.model

# log dir & log save paths
RESULT_SAVE_PATH=/workspace/megatron_train_result
LOG_NAME=${MODEL_NAME}_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${RESULT_SAVE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${RESULT_SAVE_PATH}/log/${LOG_NAME}

# ckpt load path & save path
CKPT_LOAD_PATH=${MODEL_BASE_PATH}-tp${TP}-pp${PP}
CKPT_SAVE_PATH=${RESULT_SAVE_PATH}/ckpt/${LOG_NAME}
mkdir -p ${RESULT_SAVE_PATH}/ckpt/

# training args
TRAIN_ITERS=200
SAVE_INTERVAL=200
EVAL_INTERVAL=100
EVAL_ITERS=10
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=256
DROP_OUT=0.0
MAX_LR=1.25e-6
MIN_LR=1.25e-7
MAX_SEQ_LEN=8192
MAX_POSITION_EMBEDDINGS=8192
INITIAL_LOSS_SCALE=4096
MIXED_PRECISION_ARGS=--bf16

# Set training command
LAUNCHER=" \
       torchrun \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "
       
DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       --sequence-parallel \
       --overlap-grad-reduce \
       "  

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --group-query-attention \
       --num-query-groups ${NUM_QUERY_GROUP} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --position-embedding-type rope \
       --use-rotary-position-embeddings \
       --rotary-base ${ROTARY_BASE} \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --make-vocab-size-divisible-by ${MAKE_VOCAB_SIZE_DIVISBLE_BY} \
       --norm-epsilon ${NORM_EPS} \
       --normalization RMSNorm \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --use-flash-attn \
       --attention-softmax-in-fp32 \
       "

LOGGING_ARGS=" \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --log-memory-to-tensorboard \
       --log-interval 1 \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       --no-gradient-accumulation-fusion \
       "
 
TRAINING_ARGS=" \
       --micro-batch-size ${MICRO_BATCH_SIZE} \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --train-iters ${TRAIN_ITERS} \
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --optimizer adam \
       --recompute-activations \
       --recompute-granularity selective \
       "

INITIALIZATION_ARGS=" \
       --seed 2024 \
       --init-method-std 0.01 \
       --initial-loss-scale ${INITIAL_LOSS_SCALE} \
       "

LEARNING_RATE_ARGS=" \
       --lr ${MAX_LR} \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.01 \
       --min-lr ${MIN_LR} \
       --weight-decay 1e-1 \
       "

CHECKPOINTING_ARGS=" \
       --load ${CKPT_LOAD_PATH} \
       --finetune \
       --no-load-optim \
       --no-load-rng \
       --save ${CKPT_SAVE_PATH} \
       --save-interval ${SAVE_INTERVAL} \
       "
 
MIXED_PRECISION_ARGS=" \
       ${MIXED_PRECISION_ARGS} \
       "
 
VALIDATION_ARGS=" \
       --eval-interval ${EVAL_INTERVAL} \
       --eval-iters ${EVAL_ITERS} \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 949,50,1 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 0 \
       --tokenizer-type ${TOKENIZER_TYPE} \
       --tokenizer-model ${TOKENIZER_PATH} \
       "
 
CMD="${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       ${MOE_ARGS} \
       "
echo ${CMD}
${CMD} 2>&1 | tee ${LOG_PATH}
