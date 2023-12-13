
import re
import tqdm
import random
import time
import os
import transformers
import numpy as np
import torch
from datasets import load_dataset

from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from peft import prepare_model_for_int8_training, prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from utils import train,preprocessing,chunk_examples,CastOutputToFloat,print_trainable_parameters


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def collate_batch(batch):
    x_list, attention_list, y_list = [], [], []

    for seq in batch:
          encoding_dict = preprocessing(seq, tokenizer,seq_length = 40)
          token_ids = encoding_dict['input_ids'].squeeze(0)
          attention_masks = encoding_dict['attention_mask'].squeeze(0)

          x_list.append(token_ids[:-1])
          y_list.append(token_ids[1:])
          attention_list.append(attention_masks[:-1])


    x_list = torch.stack(x_list)
    y_list = torch.stack(y_list)
    attention_list = torch.stack(attention_list)

    return x_list.to(device), attention_list.to(device), y_list.to(device)

tiny_shakespeare_ = load_dataset("tiny_shakespeare")
text = tiny_shakespeare_["train"][0]['text'] + tiny_shakespeare_["test"][0]['text'] + tiny_shakespeare_["validation"][0]['text']

chunked_text = np.array(chunk_examples(text,300)) #Character level chunks
train_dataset = to_map_style_dataset(chunked_text)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)

model_name = "vilsonrodrigues/falcon-7b-instruct-sharded"

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_use_double_quant=True,
bnb_4bit_compute_dtype=torch.float16,
)

falcon = AutoModelForCausalLM.from_pretrained(
model_name,
quantization_config=bnb_config,
device_map="auto",
trust_remote_code=True
)
falcon.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.unk_token = tokenizer.eos_token



falcon = prepare_model_for_kbit_training(falcon)

for param in falcon.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

    falcon.gradient_checkpointing_enable()  # reduce number of stored activations
    falcon.enable_input_require_grads()

    falcon.lm_head = CastOutputToFloat(falcon.lm_head)

lora_alpha = 16 #16
lora_dropout = 0.1 #0.1
lora_rank = 8 #64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_rank,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        #"dense",
        #"dense_h_to_4h",
        #"dense_4h_to_h",
    ]
)

falcon_lora = get_peft_model(falcon, peft_config)

LR = 0.001
NUM_EPOCHS = 1
optimizer = torch.optim.AdamW(falcon_lora.parameters(), lr=LR,weight_decay = 0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-5)
loss_fn = nn.CrossEntropyLoss()


if __name__ == "__main__":
     

    print_trainable_parameters(falcon_lora)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader, falcon_lora, optimizer,scheduler, loss_fn, epoch)
        #scheduler.step()
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | ".format(epoch, time.time() - epoch_start_time)
        )
        print("-" * 59)

    falcon_lora.push_to_hub("AymenKallala/falcon-7b-instruct-Shakespearian-attention",
                  use_auth_token=True,
                  commit_message="Lora finetuning on Tiny Shakespeare - rank of Lora: 8 - 1 epoch - Only attention layers.",)







