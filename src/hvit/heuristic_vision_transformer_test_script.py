from hvit.heuristic_vision_transformer_block import HViT
from torchsummary import summary

print(summary(HViT(), (3, 224, 224), device='cpu'))
