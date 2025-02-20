import torch

# config
hidden_size= 576
num_attention_heads=9
num_key_value_heads=3
low_rank=8
head_dim = hidden_size // num_attention_heads

# Traing Linear
q_proj = torch.nn.Linear(hidden_size, hidden_size,bias=False)
nope_mask = torch.ones(head_dim*num_key_value_heads, hidden_size)
false_indices = torch.randperm(head_dim*num_key_value_heads)[:low_rank*num_key_value_heads]
nope_mask[false_indices] = False
nope_mask_for_q = nope_mask.reshape(num_key_value_heads,head_dim).repeat_interleave(3, dim=0).reshape(-1)
W_k_r = torch.nn.Linear(hidden_size, (nope_mask==False).sum().item(),bias=False)
W_down_k = torch.nn.Linear(hidden_size, low_rank*num_key_value_heads,bias=False)
W_down_v = torch.nn.Linear(hidden_size, low_rank*num_key_value_heads,bias=False)
W_up_k = torch.nn.Linear(low_rank*num_key_value_heads, hidden_size-(nope_mask==False).sum().item(),bias=False)
W_up_v = torch.nn.Linear(low_rank*num_key_value_heads, hidden_size,bias=False)
o_proj = torch.nn.Linear(hidden_size, hidden_size,bias=False)
hidden_states = torch.randn((2,3,hidden_size))

# function
def apply_rope(q,k):
    return q, k

# Converting training to inference
# q
new_q_weight = []
for i in range(num_attention_heads//num_key_value_heads):
    group_nope_mask=nope_mask_for_q[...,i*head_dim*num_key_value_heads:(i+1)*head_dim*num_key_value_heads]
    q_weight = q_proj.weight.T()[...,i*head_dim:(i+1)*head_dim]
    new_q_weight.append(torch.cat([q_weight[...,~nope_mask_for_q],q_weight[...,nope_mask_for_q]@(W_down_k.weight.T())], dim=0).T())
new_q_weight = torch.cat(new_q_weight, dim=0)
new_q_proj = torch.nn.Linear(hidden_size, (nope_mask==False).sum().item()+low_rank*num_attention_heads,bias=False)
new_q_proj.weight = torch.nn.Parameter(new_q_weight.T())
# v and o_oroj
new_o_proj = torch.nn.Linear(low_rank*num_key_value_heads, hidden_size,bias=False)
new_o_proj.weight = torch.nn.Parameter(W_up_v.weight.T()@o_proj.weight.T())

# inference
bsz,q_len,hidden_size = hidden_states.size()
# q
query_states = new_q_proj(hidden_states)
rope_q = torch.zeros((bsz,q_len,hidden_size))
rope_q[...,~nope_mask_for_q] = query_states[...,:(nope_mask==False).sum().item()].view(bsz,q_len,num_attention_heads,head_dim).transpose(1,2)
nope_q = query_states[...,(nope_mask==False).sum().item():].view(bsz,q_len,num_attention_heads,low_rank).transpose(1,2)
# k
rope_k = torch.zeros((bsz,q_len,num_key_value_heads*head_dim))
rope_k[...,~nope_mask] = W_k_r(hidden_states)
rope_k = rope_k.view(bsz,q_len,num_key_value_heads,head_dim).transpose(1,2)
nope_k = W_down_k(hidden_states).view(bsz,q_len,num_key_value_heads,low_rank).transpose(1,2)
# attention
rope_q,rope_k = apply_rope(rope_q,rope_k)
query_states = torch.cat([rope_q, nope_q], dim=-1)
key_states = torch.cat([rope_k, nope_k], dim=-1)
value_states = W_down_v(hidden_states).view(bsz,q_len,num_key_value_heads,low_rank).transpose(1,2)