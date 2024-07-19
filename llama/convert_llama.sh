# This is a script for finetuning Megatron-LM llama2 and llama3 model (default platform: 16 * H20 GPU, 2 nodes)
# repository: https://github.com/NVIDIA/Megatron-LM.git
# branch：git checkout 86850db
# This is a script to convert llama2 and llama3 model downloading from huggingface into mcore(megatron) type checkpoint
# TP=8, PP=2 were used in 2 nodes 16 GPUs, change these two arguments if you use different amounts of GPUs
# params dtype: torch.float32(default), torch.float16(--fp16), torch.bfloat16(--bf16)

TP=8

PP=2

MODEL_SIZE=llama3-8B

HF_FORMAT_DIR=/workspace/model_weights/llama3-8b

MEGATRON_FORMAT_DIR=${HF_FORMAT_DIR}-tp${TP}-pp${PP}

TOKENIZER_MODEL=${HF_FORMAT_DIR}/original/tokenizer.model


python tools/checkpoint/convert.py \
--model-type GPT \
--loader llama_mistral \
--saver mcore \
--checkpoint-type hf \
--model-size ${MODEL_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL} \
--target-tensor-parallel-size ${TP} \
--target-pipeline-parallel-size ${PP} \
--bf16
