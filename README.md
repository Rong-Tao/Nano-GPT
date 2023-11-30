# Nano-GPT

This repository contains code and resources for my personal learning project, inspired by and following the tutorial titled "Let's build GPT: from scratch, in code, spelled out." The tutorial is conducted by Andrej Karpathy and is available on YouTube.

[![Tutorial: Let's build GPT](https://img.youtube.com/vi/kCc8FmEb1nY/0.jpg)](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy)

## About the Project

Nano-GPT is a learning project where I explore the intricacies of building a GPT (Generative Pretrained Transformer) model from scratch. The primary objective is to gain a deeper understanding of the underlying concepts and mechanisms of GPT models.

## Tutorial Reference

The tutorial I am following can be found here: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy).

## Learning Outcomes

Through this project, I aim to achieve the following learning outcomes:

- Understanding the architecture of GPT models.
- Implementing the model in Python.
- Grasping the training and fine-tuning processes of language models.

## How to Use This Repository

This repository contains code, notes, and other relevant resources related to the Nano-GPT project. To get started:

1. Clone the repository.
2. Explore the code and accompanying documentation.
3. Refer to the original tutorial for a comprehensive understanding.
I am using GPUs with computation capability equivalent to Nvidia V100 with 16GB VRAM.

## Naive Shakspeare

I made a slight modification, instead of guess the next letter I am using the gpt2 token. The training process can be viewed in the [Tensorboard](Naive_Shakespeare/tensorboard). A single GPU slurm files is also provided for uploading to HPC.

## Distributed Shakspeare

This is simply naive Shakespeare with distributed code added. The code in .slurm file now uses 2 nodes each with 4 GPU, you can adjust the number if you want

## Acknowledgements

Special thanks to Andrej Karpathy for the insightful and detailed tutorial on GPT models.

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.
