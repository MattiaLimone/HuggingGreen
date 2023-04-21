from torch import nn
from hvit.classification_block.classification_head import ClassificationHead
from hvit.decision_block.patch_embedding import PatchEmbedding
from hvit.transformer_block.gmm_transformer_encoder_block import TransformerEncoder
from hvit.decision_block.filter_block import ImagePatchFilter


class HViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 12,
                 num_heads: int = 12,
                 n_classes: int = 1000,
                 top_k: int = 8,
                 heuristic: str = 'contrast',
                 probabilistic: bool = True,
                 prob: float = 0.5,
                 decay_rate: float = 0.0,
                 batch_size: int = 1,
                 verbose: bool = False,
                 **kwargs):
        super().__init__(
            ImagePatchFilter(patch_size, top_k, heuristic, probabilistic, prob, decay_rate, batch_size, verbose),
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
