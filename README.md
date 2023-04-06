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
Vision Transformers (ViTs) have achieved state-of-the-art performance in many computer vision tasks. However, their high computational cost and energy consumption during training remain a major obstacle for their widespread adoption in resource-constrained scenarios. In this work, we propose a modified ViT architecture with probabilistic attention mechanisms that reduce the energy consumption during training without sacrificing the model's performance. Our approach utilize a probabilistic interpretation of the attention score in order to reduce the computational complexity required for self-attention, as well as a sparsity-inducing regularization term that encourages the attention weights to be more sparse. Additionally, we introduce a novel technique that selectively activates the attention mechanism only for important feature maps, further reducing energy consumption. We evaluate our proposed method on image classification datasets, and show that our method achieves comparable or better accuracy than the state-of-the-art ViT models while reducing energy consumption. The results aims to demonstrate the effectiveness of our modified ViT architecture in achieving energy-efficient training, which can benefit a wide range of applications, especially in resource-constrained environments. 
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

To install PyTorch with CUDA in your local env:
   ```sh
   pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio===2.0.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```

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

