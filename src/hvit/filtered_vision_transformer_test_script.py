from hvit.filtered_vision_transformer_block import FViT
from torchsummary import summary

print(summary(HViT(), (3, 224, 224), device='cpu'))
