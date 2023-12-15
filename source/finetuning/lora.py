import torch
from peft import prepare_model_for_int8_training, prepare_model_for_kbit_training, LoraConfig, get_peft_model


class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
  
def prepare_model_for_lora(model):
    model = prepare_model_for_kbit_training(model)

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()  # reduce number of stored activations
        model.enable_input_require_grads()

        model.lm_head = CastOutputToFloat(model.lm_head)
    return model
  
def LoRA(model,lora_alpha,lora_dropout,lora_rank,target_modules):

    model = prepare_model_for_lora(model)
    
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
    model_lora = get_peft_model(model, peft_config)
    return model_lora
