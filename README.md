<h1 align="center">How to Run LSAQ</h1>
<p align="center">LSAQ: Layer-Specific Adaptive Quantization for Large Language Model Deployment
</p>


## Install from source

Create a conda environment.

```bash
conda create -n lsaq python==3.10
conda activate lsaq
```

Pull the LSAQ code from the repository..
```bash
git clone https://github.com/Brett-z/LSAQ-main.git && cd LSAQ-main
```

Then, install locally from source:
```bash
pip install -r requirements.txt
```

## How to run

Please download the model to your local machine. For the list of supported models and their corresponding links, refer to the **Supported Models** section. 

```bash
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
git clone https://huggingface.co/Qwen/Qwen3-8B
```

Run the `main.py` function and specify the specific parameters at the same time.

```bash
python main.py --model xx --limit yy
```

The specific parameters and their descriptions are as follows:

| parameter | description                                                  |
| --------- | ------------------------------------------------------------ |
| `--model` | Path to the model to be used (e.g.: /data/LLMs/Llama-2-7b-hf); **Required parameter** |
| `--limit` | Restricts the available GPU memory for model deployment (e.g.: 10000; default unit is MiB); **Optional parameter**, all available GPU memory will be used for deployment if not set |

### Example Usage

Tests can be run with:

```
python main.py --model /data/LLMs/Llama-2-7b-hf --limit 10000
```

### Supported Models

| model type | model version | model parameter | quantization | inference | link                                            |
| ---------- | ------------- | --------------- | ------------ | --------- | ----------------------------------------------- |
| llama      | 2             | 7B              | ✅            | ✅         | https://huggingface.co/meta-llama/Llama-2-7b-hf |
| qwen       | 3             | 8B              | ✅            | ✅         | https://huggingface.co/Qwen/Qwen3-8B            |

## How to test

For different models (qwen & llama), the `--limit` parameter can be set to restrict available resources, thereby testing the quantization level of the model under different memory limits and the inference capability of the quantized model.

>Notice:  Ensure that the device has at least **24GB** of GPU memory before conducting tests.

Our code will ultimately output the **quantization precision of each layer**, as well as the **perplexity (ppl) of the quantized model** and the **maximum GPU memory allocated**. Detailed test cases are provided as follows:

| model type | --limit | quantization strategy | average bit | command                                                      |
| ---------- | ------- | --------------------- | ----------- | ------------------------------------------------------------ |
| **llama**  | 16000   | no quantize           | 16          | python main.py --model /data/LLMs/Llama-2-7b-hf --limit 16000 |
|            | 13000   | fp16+int8             | 12          | python main.py --model /data/LLMs/Llama-2-7b-hf --limit 13000 |
|            | 9000    | int8                  | 8           | python main.py --model /data/LLMs/Llama-2-7b-hf --limit 9000 |
|            | 8000    | int8+int4             | 6           | python main.py --model /data/LLMs/Llama-2-7b-hf --limit 8000 |
|            | 7000    | int4                  | 4           | python main.py --model /data/LLMs/Llama-2-7b-hf --limit 7000 |
| **Qwen**   | 20000   | no quantize           | 16          | python main.py --model /data/LLMs/Qwen-3-8b --limit 20000    |
|            | 17000   | fp16+int8             | 12          | python main.py --model /data/LLMs/Qwen-3-8b --limit 17000    |
|            | 13000   | int8                  | 8           | python main.py --model /data/LLMs/Qwen-3-8b --limit 13000    |
|            | 11000   | int8+int4             | 6           | python main.py --model /data/LLMs/Qwen-3-8b --limit 11000    |
|            | 10000   | int4                  | 4           | python main.py --model /data/LLMs/Qwen-3-8b --limit 10000    |

