import torch
device = 'cuda' if torch.cuda_is_available() else 'cpu'


def generate(model,input,tokenizer,sampler, max_new_tokens,context_length = None):
        """
        idx is (B, T) array of indices in the current context
        This will generate B total paths in parrallel
        We will just geenrate 1 batch below
        """


        if context_length == None:
            context_length = tokenizer.model_max_length

        model.eval()
        encoded = tokenizer(input,return_tensors="pt").to(device)
        EOS = tokenizer.eos_token_id

        idx = encoded['input_ids']
        generated = torch.tensor([],dtype= torch.int32,device = device)
        last_token = None
        step=0

        while last_token != EOS and step < max_new_tokens:
            # crop idx to the last block_size tokens
            # The model only has kowledge of the context of maximum size block_size
            # Get the newest (B, T) data; T = block_size
            B,T = idx.shape
            idx_cond = idx[:,-max(T,context_length):]

            # Get the predictions
            # (B, T, vocab_size)
            logits = model(idx_cond).logits

            # Focus only on the last time step, get the logits
            # (B, vocab_size)
            logits = logits[:, -1, :]
            idx_next = sampler(logits).to(device)
            last_token = idx_next.view(-1,1)

            # Append sampled index to the running sequence
            # (B, T+1)
            generated = torch.cat([generated,last_token],dim=-1)
            idx = torch.cat([idx[:,1:],last_token],dim=-1)
            step+=1
        return tokenizer.decode(generated.squeeze())

def train(dataloader, model, optimizer,scheduler, criterion, epoch):
    model.train()
    total_loss,total_count = 0,0
    log_interval = 5

    for idx, (text,attention,target) in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        logits = model(input_ids = text,attention_mask=attention,output_attentions = False).logits


        B, T, V = logits.shape
        logits = logits.view(B*T, V)
        targets = target.view(B*T)
        loss = criterion(logits, targets)
        loss.backward()

        # Clip the gradients at 0.1
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # Do an optimization step
        optimizer.step()
        total_count += len(target)
        total_loss+= loss.item()
        if idx % log_interval == 0 and idx > 0:
            scheduler.step()
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| train loss {:8.3f}".format(epoch, idx, len(dataloader), total_loss / total_count)
            )
            total_loss,total_count = 0,0

def preprocessing(input_text, tokenizer,seq_length = 100):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
    # Use the tokenizer to preprocess text
    # add_special_tokens = True, let the max_length = 32, pad_to_max_length = True, return_tensors = 'pt'
    # Look up tokenizer.encode_plus
  return(tokenizer.encode_plus(input_text,add_special_tokens=True,max_length=seq_length,pad_to_max_length = True,return_tensors='pt'))

def chunk_examples(examples,seq_length):
    chunks = [examples[i:i + seq_length] for i in range(0, len(examples), seq_length)]
    return chunks


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )