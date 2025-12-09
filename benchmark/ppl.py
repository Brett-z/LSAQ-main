import torch
import torch.nn as nn
import tqdm
import os
import sys
from datasets import load_dataset, DatasetDict, load_from_disk

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

dataset_path = os.path.join(root_path, "dataset")
dataset_path = os.path.abspath(dataset_path)
# dataset_path = "/data/zbr/LLM-Quant/LSAQ-main/dataset"
dataset = load_from_disk(dataset_path)

def ppl_test(model, tokenizer):
    testenc = dataset
    testenc = tokenizer("\n\n".join(testenc["text"][:100]), return_tensors="pt")
    # model.seqlen = 2048
    model.seqlen = 1024
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    # for i in range(nsamples):
    for i in tqdm.tqdm(range(nsamples), desc="Evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen    
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    # print(ppl.item())

    return ppl.item()
