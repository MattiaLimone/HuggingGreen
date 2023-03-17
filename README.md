<div id="readme-top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU General Public License v2.0][license-shield]][license-url]

<h1 align="center">HuggingGreen</h1>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a>
        <ul>
          <li><a href="#pytorch-with-cuda">PyTorch with CUDA</a></li>
        </ul>
        </li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project
<div style="text-align: justify"> 
Multi-head attention is a crucial component in Transformers that has led to state-of-the-art performance in various natural language processing. Dividing an image in patch and interpreting each patch as a sequence as lead to the Vision Transformer. Recent research has shown that many of the attention heads in these models learn similar embeddings, which implies that removing some of them may not affect the model's performance.
With this insight, we introduce a new Vision Transformer architecture called GMM-HViT.
The architecture only processes certain patches of each image. During training, a heuristic function is used to determine which parts of the image should be utilized.
Furthermore, instead of redundant attention heads, GMM-ViT uses a mixture of keys at each head that follows a Gaussian mixture model. This enables each attention head to efficiently focus on different segments of the input sequence. 
In addition only some patches of each image are actually processed by the architecture. An heuristic function will select which parts must be used in the training.
</div>

### Built With

* ![NumPy]
* ![PyTorch]
* ![Matplotlib]
## Getting started

### Prerequisites

What you will need:
- Python 3.8 or later (for Linux, Python 3.8.1+ is needed)
- A C++17 compatible compiler, such as clang

#### PyTorch with CUDA

If you want to compile the PyTorch with CUDA support, install the following (note that CUDA is not supported on macOS)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.0 or above
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 or above
- [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) for cuDNN versions with the various supported CUDA, CUDA driver and NVIDIA hardware

If you want to disable CUDA support, export the environment variable `USE_CUDA=0`.
Other potentially useful environment variables may be found in `setup.py`.

If you are building for NVIDIA's Jetson platforms (Jetson Nano, TX1, TX2, AGX Xavier), Instructions to install PyTorch for Jetson Nano are [available here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

## Installation

1) Clone the repo
   ```sh
   git clone https://github.com/MattiaLimone/HuggingGreen.git
   ```
2) Install the requirements
   ```sh
   pip install -r requirements.txt
   ```
3) Use the train script to train the model (Hyperparameters must be set inside this file)
   ```sh
   python train_script.py
   ```
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Mattia Limone [[Linkedin profile](https://www.linkedin.com/in/mattia-limone/)]

Carmine Iannotti [[Linkedin Profile](https://www.linkedin.com/in/carmine-iannotti-aa031b232/)]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
[TensorFlow]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[Keras]: https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white
[NumPy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Matplotlib]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[PyTorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[SciPy]: https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white

