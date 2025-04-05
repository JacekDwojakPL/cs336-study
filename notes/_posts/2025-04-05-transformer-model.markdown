---
layout: post
title: "Assignment 1 - Transformer model"
date: 2025-04-05 20:00:00 +0100
categories: assignment1
author: jacek
---

### Transformer Model

After completing the Tokenizer part of the assignment and saving the tokenized training datasets as .npy arrays, it’s now time to move on to the Transformer component, where the task is to build each part of the Transformer model from scratch.

This section is somewhat rudimentary, as there is a plethora of resources that explain how to develop Multi-Head Attention layers, normalization layers, transformer blocks, and more. A nice aspect of this assignment is that each module includes its own tests. So if I found myself stuck, I could refer to the test files and the expected outputs to help debug and refine my implementation.

The assignment begins with implementing the RMSNormalization layer, which normalizes the inputs fed into the Transformer. Normalization layers are crucial for rescaling features, which helps stabilize training and prevents issues like exploding or vanishing gradients. The Root Mean Square (RMS) normalization layer also includes learnable weights (denoted as `g` in the assignment handout).

Next, the assignment moves on to implementing the Feed Forward component of a Transformer block. Here, the feed-forward network is a simple two-layer neural network with a **GeLU** activation function. This component follows the output from the Multi-Head Attention module. In a full Transformer model, the number of such blocks (or layers) is a **hyperparameter**, which can be tuned to find the optimal configuration.
