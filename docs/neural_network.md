# AlphaChess Neural Network Architecture

This document details the design of the neural network used in AlphaChess, which is inspired by the architecture of AlphaZero.

## 1. Framework and Core Principles

- **Framework:** The neural network will be implemented using PyTorch, leveraging its capabilities for building deep learning models and utilizing GPU acceleration.
- **Dual-Head Architecture:** Similar to AlphaZero, the network will have a shared "trunk" that processes the input board state, and then branches into two distinct "heads": a policy head and a value head. This allows the network to learn both move probabilities and game outcome predictions simultaneously.

## 2. Network Structure

### 2.1. Input Layer

- The input to the network is a tensor representing the encoded chess board state. The shape is `(batch_size, 29, 8, 8)`, where `29` is the number of input planes as defined in `docs/architecture.md` and `src/chess_env.py`.

### 2.2. Convolutional Trunk

- **Initial Convolutional Block:** The input will first pass through a convolutional layer with a small kernel size (e.g., 3x3) and a sufficient number of filters, followed by batch normalization and a ReLU activation. This block extracts initial features from the board state.
- **Residual Blocks:** The core of the trunk will consist of a configurable number of residual blocks (e.g., 10 to 20 blocks). Each residual block typically includes:
  - Two convolutional layers.
  - Batch normalization layers after each convolution.
  - ReLU activation functions.
  - A skip connection that adds the input of the block to its output, helping to mitigate vanishing gradients and enabling deeper networks.
- **Output of Trunk:** The trunk will output a feature map that encapsulates high-level representations of the board state, which will then be fed into both the policy and value heads.

### 2.3. Policy Head

- **Purpose:** To predict a probability distribution over all legal moves from the current board state.
- **Structure:**
  - A small convolutional layer (e.g., 1x1 kernel) to reduce the number of channels in the feature map from the trunk.
  - Batch normalization and ReLU activation.
  - A flattening operation to convert the feature map into a 1D vector.
  - A fully connected (dense) layer that maps the flattened features to a vector representing the probabilities of all possible moves. The size of this output vector will correspond to the total number of possible moves in chess (e.g., 4672 for a common move encoding scheme).
  - A `log_softmax` activation function applied to the output, as implemented, to ensure the probabilities sum to 1 (in log space).

### 2.4. Value Head

- **Purpose:** To predict the scalar outcome of the game from the current player's perspective (e.g., +1 for a win, -1 for a loss, 0 for a draw).
- **Structure:**
  - A small convolutional layer (e.g., 1x1 kernel) to reduce the number of channels in the feature map from the trunk.
  - Batch normalization and ReLU activation.
  - A flattening operation.
  - A fully connected layer.
  - Another fully connected layer that outputs a single scalar value.
  - A `tanh` activation function to scale the output to the range `[-1, 1]`.

## 3. Hardware Considerations and Optimization

- **VRAM Management:** The number of residual blocks and filters will be chosen to ensure the model fits within the 8GB VRAM of the Nvidia 3070ti.
- **Mixed-Precision Training (FP16):** The training process will utilize `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` to enable mixed-precision training. This reduces memory footprint and can accelerate training on compatible GPUs.
- **`torch.compile`:** For further performance optimization, `torch.compile` (available in PyTorch 2.0+) will be considered to compile the model into optimized kernels.
