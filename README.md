# Talk2Shakespeare
Making Large Language Models generate old-fashioned English.


*Project in process - I am currently editing this repo.*

![alt-text-1](imgs/shakespeare.png "Image generated on the Real-Time Latent Consistency Model space.")

## Demo

| Prompt    | Falcon7B generation | Shakespearian-Falcon7B (mine) |
| -------- | ------- | --------|
| Tell me something| The capital is the largest city in the United States.|   Those hours, that with gentle work did frame The lovely gaze where every eye doth dwell, Will play the tyrants to the very same And that unfair which fairly doth excel: For never-resting time leads summer on |

## Table of Contents

- [Talk2Shakespeare](#project-name)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Demo](#demo)
  - [Dataset](#features)
  - [Installation](#installation)


## Description

This project is an attempt to build a language model generating on "old-fashioned" English by fine-tuning state-of-the-art (SOTA) models on ancient texts. The project utilizes open-source datasets, primarily consisting of Shakespearean texts, to train models such as BLOOM 7B and Falcon 7B. Fine-tuning is performed using techniques like LoRA with the PEFT (Parametet Efficient Fine Tuning) library from Hugging Face.

## LoRA

To make fine-tuning more efficient, LoRA’s approach is to represent the weight updates with two smaller matrices (called update matrices) through low-rank decomposition. These new matrices can be trained to adapt to the new data while keeping the overall number of changes low. The original weight matrix remains frozen and doesn’t receive any further adjustments. To produce the final results, both the original and the adapted weights are combined.

This approach has a number of advantages:

1. LoRA makes fine-tuning more efficient by drastically reducing the number of trainable parameters.
2. The original pre-trained weights are kept frozen, which means you can have multiple lightweight and portable LoRA models for various downstream tasks built on top of them.
3. LoRA is orthogonal to many other parameter-efficient methods and can be combined with many of them.
4. Performance of models fine-tuned using LoRA is comparable to the performance of fully fine-tuned models.
5. LoRA does not add any inference latency because adapter weights can be merged with the base model.

_Paragraph from Huggingface blog_

![alt-text-1](imgs/LoRA_diagram.png "Image generated on the Real-Time Latent Consistency Model space.")

## Dataset

List of the data used so far

1. [Open-source Shakespeare books](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt)
2. [Shakespeare dataset on HuggingFace](https://huggingface.co/datasets/tiny_shakespeare)

## How to get the model

My model is findable on HuggingFace Hub.

