
# HuggingGreen
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU General Public License v2.0][license-shield]][license-url]

<div style="text-align: justify"> 
Multi-head attention is a crucial component in Transformers that has led to state-of-the-art performance in various natural language processing. Dividing an image in patch and interpreting each patch as a sequence as lead to the Vision Transformer. Recent research has shown that many of the attention heads in these models learn similar embeddings, which implies that removing some of them may not affect the model's performance.
With this insight, we introduce a new Vision Transformer architecture called GMM-HViT.
The architecture only processes certain patches of each image. During training, a heuristic function is used to determine which parts of the image should be utilized.
Furthermore, instead of redundant attention heads, GMM-ViT uses a mixture of keys at each head that follows a Gaussian mixture model. This enables each attention head to efficiently focus on different segments of the input sequence. 
In addition only some patches of each image are actually processed by the architecture. An heuristic function will select which parts must be used in the training.
</div>
[contributors-shield]: https://img.shields.io/github/contributors/MattiaLimone/HuggingGreen.svg?style=for-the-badge
[contributors-url]: https://github.com/MattiaLimone/HuggingGreen/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MattiaLimone/HuggingGreen.svg?style=for-the-badge
[forks-url]: https://github.com/MattiaLimone/HuggingGreen/network/members
[stars-shield]: https://img.shields.io/github/stars/MattiaLimone/HuggingGreen.svg?style=for-the-badge
[stars-url]: https://github.com/MattiaLimone/HuggingGreen/stargazers
[issues-shield]: https://img.shields.io/github/issues/MattiaLimone/HuggingGreen.svg?style=for-the-badge
[issues-url]: https://github.com/MattiaLimone/HuggingGreen/issues
[license-shield]: https://img.shields.io/github/license/MattiaLimone/HuggingGreen.svg?style=for-the-badge
[license-url]: https://github.com/MattiaLimone/HuggingGreen/blob/main/LICENSE
