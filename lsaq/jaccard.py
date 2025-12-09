import os
import math
import tqdm
import torch
import numpy as np
from datasets import load_dataset, DatasetDict, load_from_disk

topk=30
batch_size = 1
MAX_SEQ_LEN = 1024

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "..", "dataset")
dataset_path = os.path.abspath(dataset_path)

dataset = load_from_disk(dataset_path)
# dataset = load_dataset("/data/zbr/LLM-Quant/LSAQ-main/dataset", split="test")
dataset_size = 200
# dataset_size = len(dataset)

def encode(tok, text, padding=True, truncation=True, max_length=None):
    # 将文本转换为输入 IDs
    input_ids = [tok.bos_id] + tok.encode(text)

    # 生成注意力掩码
    attention_mask = [1] * len(input_ids)

    # 如果进行了填充，则调整注意力掩码
    if padding:
        padding_length = max_length - len(input_ids)
        attention_mask = [0] * padding_length + attention_mask
        input_ids = [tok.eos_id] * padding_length + input_ids

    encoded_input = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    return encoded_input

def batch_encode_plus(tok, texts, max_length=None, return_tensors=None):
    encoded_inputs = []

    # 循环处理每个文本
    if max_length is None:
        max_length = -1
        for text in texts:
            # if isinstance(text, list):
            #     text = text[0]
            # print(text)
            len_ = len([tok.bos_id] + tok.encode(text))
            if len_ > max_length:
                max_length = len_
    for text in texts:
        # if isinstance(text, list):
        #     text = text[0]
        encoded_input = encode(tok, text, max_length = max_length)
        encoded_inputs.append(encoded_input)

    # 合并结果
    batch_encoded = {
        'input_ids': [encoded_input['input_ids'] for encoded_input in encoded_inputs],
        'attention_mask': [encoded_input['attention_mask'] for encoded_input in encoded_inputs]
    }

    batch_encoded = {key: torch.tensor(val) for key, val in batch_encoded.items()}

    return batch_encoded

def jaccard_set(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def cal_jaccard(model, tokenizer):

    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.bos_id = tokenizer.bos_token_id
    tokenizer.eos_id = tokenizer.eos_token_id
    importances = [0 for i in range(len(model.model.layers))]

    for i in tqdm.tqdm(range(0, dataset_size, batch_size), total = dataset_size / batch_size):
        
        prompts = dataset['text'][i:i + batch_size]
        max_seq_len = MAX_SEQ_LEN
        stride = 256
        max_gen_len = 0

        prompt_tokens = batch_encode_plus(
            tokenizer,
            prompts,
            return_tensors='pt'
        )

        input_ids = prompt_tokens['input_ids']
        attn_mask = prompt_tokens['attention_mask']
        max_prompt_len = max(len(t) for t in input_ids)
        all_jac_sim = [0 for i in range(len(model.model.layers))] 
        E = model.get_input_embeddings().weight.detach()
        
        # authors use a sliding window of size 1024 with a shift of 256
        for start in range(0, max_prompt_len, stride):
            seq_ids = (attn_mask.sum(dim=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.dim() == 0 else seq_ids  # ensure 2d
            inputs = input_ids[seq_ids, start:start+max_seq_len]
            attn = attn_mask[seq_ids, start:start+max_seq_len]

            if max_gen_len == 0:
                outputs = model(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    output_hidden_states=True,
                )
            else:
                outputs = model.generate(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    max_new_tokens=max_gen_len, 
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            hiddens = outputs.hidden_states

            for i in range(len(hiddens) - 1):
                in_hidden = hiddens[i][:,-1,:]
                out_hidden = hiddens[i+1][:,-1,:]

                # _, _, d = in_hidden.shape
                # in_hidden = in_hidden.reshape(-1, d)
                # out_hidden = out_hidden.reshape(-1, d)

                in_projs = in_hidden @ E.T
                out_projs = out_hidden @ E.T

                in_projs = in_projs.detach().cpu().numpy()
                ot_projs = out_projs.detach().cpu().numpy()

                in_ind = np.argsort(-in_projs)
                ot_ind = np.argsort(-ot_projs)

                in_topks = [tokenizer.decode(i) for i in in_ind[0][:topk]]
                ot_topks = [tokenizer.decode(i) for i in ot_ind[0][:topk]]

                all_jac_sim[i] += jaccard_set(in_topks, ot_topks)

        
        importances = [x + y for x, y in zip(importances, all_jac_sim)]
    
    return importances

def normalize(lst, range_min=0, range_max=1):
    min_val = min(lst)
    max_val = max(lst)
    normalized = [(range_max - range_min) * (x - min_val) / (max_val - min_val) + range_min for x in lst]
    return normalized

def cal_importance(model, tokenizer):
    jac_list = cal_jaccard(model, tokenizer)
    filtered_values = [0 if math.isinf(value) else value for value in jac_list]
    normalized_lst = normalize(filtered_values)
    sorted_indices = sorted(range(len(normalized_lst)), key=lambda i: normalized_lst[i])
    reversed_list = list(reversed(sorted_indices))
    return reversed_list