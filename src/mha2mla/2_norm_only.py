from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import torch
from transformers.models.llama.modeling_llama import LlamaAttention

path = "/data/shenth/models/SmolLM/135m"

config = AutoConfig.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
model = LlamaForCausalLM.from_pretrained(path, config=config)
model.eval().to("cuda")

head_dim = config.hidden_size // config.num_attention_heads
num_layers = config.num_hidden_layers
query_states = [[] for _ in range(num_layers)]
key_states = [[] for _ in range(num_layers)]

def cal_2_norm(states):
    states = torch.norm(
        # 
        states.reshape(states.shape[0],states.shape[1],states.shape[2],2,-1).transpose(-1,-2),
        p=2,
        dim=4,
    )
    return states

hidden_states_dict = {}

def create_hook_fn(name):
    def hook(module, args, kwargs, output):
        hidden_states_dict[name] = kwargs["hidden_states"]

    return hook

for name, module in model.named_modules():
    if not isinstance(module, LlamaAttention):
        continue
    hook_fn = create_hook_fn(name)
    module.register_forward_hook(hook_fn,with_kwargs=True)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
for name,module in model.named_modules():
    if not isinstance(module, LlamaAttention):
        continue
    idx = int(name.split(".")[2])
    bsz,q_len,_ = hidden_states_dict[name].shape
    q = module.q_proj(hidden_states_dict[name]).reshape(bsz,q_len,config.num_attention_heads,head_dim) # [bsz,q_len,num_heads,head_dim]
    k = module.k_proj(hidden_states_dict[name]).reshape(bsz,q_len,config.num_key_value_heads,head_dim)
    query_states[idx].append(cal_2_norm(q).mean(dim=1,keepdim=False).cpu()) # [bsz,num_heads,head_dim//2]
    key_states[idx].append(cal_2_norm(k).mean(dim=1,keepdim=False).cpu())

from IPython import embed
embed()