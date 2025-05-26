# Meta-Tool: Unleash Open-World Function Calling Capabilities of General-Purpose Large Language Models

This repository contains the code and resources for the **Meta-Tool** project, which aims to enhance the open-world function calling capabilities of general-purpose large language models (LLMs). Below, you will find instructions on how to set up and use the software, as well as how to evaluate the model's performance on the Meta-Bench benchmark. Please download our lora weight from https://huggingface.co/Shengqian12138/meta-tool-lora and our benchmark from https://huggingface.co/Shengqian12138/Meta-Bench.

## Setup Instructions

### 1. Download Model Weights

You need to download the following models from Hugging Face:

1. **Base Model**: `meta-llama/Llama-3.1-8B-Instruct`
2. **Retriever Model**: `intfloat/multilingual-e5-large`

Run the following commands to download the models:

```bash
huggingface-cli download --resume-download meta-llama/Llama-3.1-8B-Instruct --local-dir {MODEL_DIR_ROOT}/meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download --resume-download intfloat/multilingual-e5-large --local-dir {MODEL_DIR_ROOT}/intfloat/multilingual-e5-large
```
Replace {MODEL_DIR_ROOT} with the path where you typically store Hugging Face model weights on your computer.

### 2. Install Dependencies
Create new conda environments and install the required dependencies:

```bash
conda create --name meta-tool python=3.10
conda activate meta-tool
pip install -r requirements.txt
```

### 3. Start vLLM Inference
To start the vLLM inference server, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve PATH_of_meta-llama/Llama-3.1-8B-Instruct --port 1053 --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 4096 --api-key token-abc123 --enable-lora --lora-modules lora1=PATH_of_Meta_Tool_Lora_Module --trust-remote-code
```

Replace PATH_of_meta-llama/Llama-3.1-8B-Instruct and PATH_of_Meta_Tool_Lora_Module with the appropriate paths on your system.

### 4. Evaluate Meta-Tool on Meta-Bench
To evaluate the performance of Meta-Tool on the Meta-Bench benchmark, run the following commands:

```bash
sh test_meta_tool_llama.sh
python evaluate.py --model_name meta-llama/Llama-3.1-8B-Instruct-meta-tool
```

Detailed inference results will be stored in ./result. Scores will be saved in ./score.

To evaluate other models, modify the provided scripts accordingly.
