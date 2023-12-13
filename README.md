# Talk2Shakespeare
Making Large Language Models generate old-fashioned English. Project in process - I am currently editing this repo.

![alt-text-1](imgs/shakespeare.png "Image generated on the Real-Time Latent Consistency Model space.")

## Table of Contents

- [Talk2Shakespeare](#project-name)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Demo](#demo)
  - [Dataset](#features)
  - [Installation](#installation)

## Description

This project is an attempt to build a language model generating on "old-fashioned" English by fine-tuning state-of-the-art (SOTA) models on ancient texts. The project utilizes open-source datasets, primarily consisting of Shakespearean texts, to train models such as BLOOM 7B and Falcon 7B. Fine-tuning is performed using techniques like LoRA with the PEFT (Parametet Efficient Fine Tuning) library from Hugging Face.

## Demo

| Prompt    | Falcon7B generation | Shakespearian-Falcon7B (mine) |
| -------- | ------- | --------|
| Tell me something| The capital is the largest city in the United States.|   Those hours, that with gentle work did frame The lovely gaze where every eye doth dwell, Will play the tyrants to the very same And that unfair which fairly doth excel: For never-resting time leads summer on |

## Dataset

List of the data used so far

1. [Open-source Shakespeare books](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt)
2. [Shakespeare dataset on HuggingFace](https://huggingface.co/datasets/tiny_shakespeare)

## Installation

Provide instructions on how to install your project. Include any dependencies that need to be installed and step-by-step instructions.

```bash
# Example installation steps
$ git clone https://github.com/your-username/your-project.git
$ cd your-project
$ npm install
