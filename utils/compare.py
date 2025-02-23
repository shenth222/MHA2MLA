import torch
import pickle

# with open("qk_tensor_135M.pkl", "rb") as f:
#     a = pickle.load(f)
# a=a.view(30,3,3,32).sum(dim=2)
# with open("../utils/qk_tensor_135M.pth", "rb") as fin:
#     b = torch.load(fin)
# print(torch.allclose(a,b))

with open("../utils/qk_tensor_360M.pth", "rb") as fin:
    b = torch.load(fin)
print(b.shape)