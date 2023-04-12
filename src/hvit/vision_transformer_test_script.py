from hvit.vision_transformer_block import ViT
from torchsummary import summary

print(summary(ViT(), (3, 224, 224), device='cpu'))
