# This is a script for finetuning Megatron-LM llama2 and llama3 model (default platform: 16 * H20 GPU, 2 nodes)
# repository: https://github.com/NVIDIA/Megatron-LM.git
# branch：git checkout 86850db
# This is a script to preprocess finetune dataset
# The dataset we use is downloaded from: 
# https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
# And after downloading the dataset, we convert the data type into json type


INPUT_FILE=/workspace/dataset/finetune_dataset/train-00000-of-00001-a09b74b3ef9c3b56.json

MODEL_PATH=/workspace/model_weights/llama3-8b
TOKENIZER_MODEL=${MODEL_PATH}/original/tokenizer.model
OUTPUT_DIR=/workspace/dataset/finetune_dataset/llama3-8b
OUTPUT_PREFIX=${OUTPUT_DIR}/alpaca
TOKENIZER_TYPE=Llama3Tokenizer

mkdir -p ${OUTPUT_DIR}

python ./tools/preprocess_data.py \
--input ${INPUT_FILE} \
--output-prefix ${OUTPUT_PREFIX} \
--tokenizer-model ${TOKENIZER_MODEL} \
--workers 4 \
--log-interval 1000 \
--tokenizer-type ${TOKENIZER_TYPE} \
--append-eod
