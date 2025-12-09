import os
import sys
import tqdm
import time 
import json
import torch
import torch.nn as nn
import GPUtil
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from functools import partial
import lsaq.config as cfg
from lsaq.gpu_select import gpu_select
from lsaq.lsaq_quant import quantize_model
from lsaq.jaccard import cal_importance
from lsaq.strategy import quant_strategy
from benchmark.ppl import ppl_test
import argparse


def find_model_type(model_path):

    config_path = os.path.join(model_path, "config.json")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        model_type = config_data.get("model_type", "Could not find the model_type field.")
        print(f"The model_type of the current weights is：{model_type}")

    except FileNotFoundError:
        print(f"Error: Could not find the config.json file in path {model_path}")
    except json.JSONDecodeError:
        print(f"Error: {config_path} is not a valid JSON file. Please check the file format.")
    except Exception as e:
        print(f"Unexpected error occurred while reading config.json: {str(e)}")
    
    return model_type

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="提供使用的模型路径（如：/data/zbr/LLMs/Llama-2-7b-hf）"
    )
    parser.add_argument(
        "--limit",
        dest="gpu_limit",
        type=int,
        default=-1,
        help="限制 GPU 的可用显存用于模型部署（如：10000；不需要输入单位，默认为MiB）"
    )
    # parser.add_argument(
    #     "--gpuid",
    #     dest="gpu_id",
    #     type=int,
    #     default=-1,
    #     help="使用哪个GPU"
    # )

    args = parser.parse_args()
    gpu_id, gpu_memory = gpu_select()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model_path = args.model_path
    # model_path = "/data/zbr/LLMs/Llama-2-7b-hf"
    # model_path = "/data/llms/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

    # ## How to calculate layer importance
    # model_jaccard = model
    # model_jaccard.cuda()
    # importance = cal_importance(model_jaccard, tokenizer)
    # print(importance)

    model_type = find_model_type(model_path)

    if model_type == "llama":
        Model_LI = cfg.Llama2_7B_LI
    elif model_type == "qwen3":
        Model_LI = cfg.Qwen3_8B_LI
    else:
        sys.exit("[ERROR] LSAQ does not currently support the current model. Please select a different model for deployment.")

    print("[INFO] The order of LLM layer importance from lowest to highest is as follows:")
    print(Model_LI)

    if args.gpu_limit != -1:
        gpu_memory = args.gpu_limit
        print(f"[INFO] The available GPU resources are limited to {gpu_memory} MiB.")

    num_to_quant, low_bit = quant_strategy(gpu_memory, model_type)

    if num_to_quant == 0:
        model_lsaq = model
        print("[INFO] No quantization is needed; deploy the original model directly (in fp16).")
    elif num_to_quant == -1:
        sys.exit("[ERROR] The current GPU device resource status does not support the deployment of the quantized model.")
    else:
        layer_to_quant = Model_LI[:num_to_quant]
        mlp_quant = [f'layers.{item}.mlp' for item in layer_to_quant]
        self_attn_quant = [f'layers.{item}.self_attn' for item in layer_to_quant]

        model_lsaq = quantize_model(model_type, model, mlp_quant, self_attn_quant, low_bit)

        for key in cfg.scales_dict.keys():
            cfg.scales_dict[key] = cfg.scales_dict[key].to('cuda')

    model_lsaq.cuda()

    ### evaluate the ppl of quantized model
    ppl = ppl_test(model_lsaq, tokenizer)

    print(f"[RESULT] Perplexity of this model is: {ppl}")
    print(f"[RESULT] Max memory:{torch.cuda.max_memory_allocated(model_lsaq.device)/ 1024**2:.2f}M")
    
    # while True:
    #     # prompt = "Where is the capital city of America"
    #     prompt = input("user:")
    #     inputs = tokenizer(prompt, return_tensors="pt").to(model_lsaq.device)

    #     # Generate
    #     start_time = time.time()
    #     generate_ids = model_lsaq.generate(inputs.input_ids, max_length=20)
    #     end_time = time.time()
    #     speed = len(generate_ids[0])/(end_time-start_time)

    #     print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    #     print(f"speed:{speed:.2f}token/s max memory:{torch.cuda.max_memory_allocated(model_lsaq.device)/ 1024**2:.2f}M")

if __name__ == "__main__":
    main()