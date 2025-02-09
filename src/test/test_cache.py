import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import numpy as np
import argparse

def calculate_loss_direct(model, input_ids, attention_mask):
    """直接计算整句话的loss"""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # 因为是自回归任务，labels就是输入
        )
    return outputs.loss.item()

def calculate_loss_with_kvcache(model, input_ids, attention_mask):
    """使用KV cache逐token计算loss"""
    total_loss = 0
    past_key_values = None
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        for i in range(0, seq_len-1):  # 从第二个token开始计算
            # 当前位置之前的输入
            current_input_ids = input_ids[:, i:i+1]  # 只取当前token
            current_attention_mask = attention_mask[:, :i+1]  # 更新attention mask
            
            # 使用past_key_values进行计算
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask[:, :i+1],  # 保持attention mask的长度一致
                past_key_values=past_key_values,
                labels=None,  # labels只包含当前token
                use_cache=True
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            labels = input_ids[:,i+1:i+2]
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fn(logits, labels.view(-1))
            # 更新past_key_values
            past_key_values = outputs.past_key_values
            # 累加loss
            total_loss += loss.item()
    
    # 计算平均loss
    return total_loss / (seq_len - 1)  # 除以预测的token数量


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The model name or path")
    parser.add_argument("--is_mla", action="store_true", help="Whether the model is a MLA model")
    args = parser.parse_args()
    # Monkey Patching
    if args.is_mla:
        import json,os
        from ..mla.mla_patch_hf import mla_patch_hf
        with open(os.path.join(args.model_name,"config.json"),"r") as f:
            model_config = json.load(f)
        mla_patch_hf(model_config["RoPE"])

    # 加载模型和分词器
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()
    
    # 将模型移至GPU（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 准备输入文本
    text = "Hello, this is a test sentence to verify the loss calculation."
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 计算两种方式的loss
    direct_loss = calculate_loss_direct(model, inputs["input_ids"], inputs["attention_mask"])
    kvcache_loss = calculate_loss_with_kvcache(model, inputs["input_ids"], inputs["attention_mask"])
    
    print(f"Direct Loss: {direct_loss:.6f}")
    print(f"KV Cache Loss: {kvcache_loss:.6f}")
    print(f"Difference: {abs(direct_loss - kvcache_loss):.6f}")

if __name__ == "__main__":
    main()
