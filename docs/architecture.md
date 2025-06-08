# AlphaChess Architecture and Design Choices

This document outlines the key architectural and design decisions made for the AlphaChess project, a chess bot inspired by AlphaGo.

## 1. Project Setup & Environment

- **Language:** Python has been chosen as the primary programming language due to its extensive ecosystem for machine learning and data science.
- **Dependency Management:** `uv` will be used for creating isolated virtual environments and managing project dependencies. This ensures reproducibility and avoids conflicts with system-wide Python packages.
- **Key Dependencies:**
  - `torch`: The core PyTorch library is selected for building and training neural networks, leveraging its flexibility and strong GPU acceleration capabilities.
  - `torchvision`: Included for potential utility functions and ensuring proper GPU support with PyTorch.
  - `torchrl`: This library will be explored to leverage existing reinforcement learning components and utilities, particularly for implementing the PPO algorithm.
  - `python-chess`: A robust and widely-used library for handling chess board representation, move generation, and rule validation, simplifying the game logic implementation.
  - `numpy`: Essential for efficient numerical operations, especially when processing board states and neural network inputs/outputs.
  - `tqdm`: For providing clear progress bars during computationally intensive processes like self-play game generation and neural network training.
  - `tensorboard`: Chosen for logging training metrics, visualizing network graphs, and monitoring the learning process effectively.
  - `PyYAML` or `json`: For parsing configuration files, allowing for flexible and human-readable parameter management.
- **Version Control:** Git will be used for version control, enabling collaborative development, tracking changes, and managing different versions of the codebase.
- **`.gitignore`:** A comprehensive `.gitignore` file will be maintained to exclude temporary files, virtual environment directories, model checkpoints, and log files from the Git repository, keeping the codebase clean and focused.

## 2. Chess Environment Representation

- **Game Logic Library:** The `python-chess` library is central to managing the chess game state. It provides functionalities for:
  - Representing the chess board.
  - Generating legal moves for any given position.
  - Applying moves to update the board state.
  - Detecting game termination conditions (checkmate, stalemate, draw by repetition, fifty-move rule).
- **Neural Network Input Encoding:** The chess board state will be transformed into a numerical format suitable for input to the convolutional neural network. This encoding will consist of a stack of binary planes, where each plane represents a specific feature of the board. This approach is inspired by successful chess AI implementations like AlphaZero. The planes will include:
  - **Piece Positions:** Separate planes for each type of piece (Pawn, Knight, Bishop, Rook, Queen, King) for both White and Black (12 planes total).
  - **Castling Rights:** Planes indicating which castling rights are available for both White and Black.
  - **En Passant Target Square:** A plane indicating the square where an en passant capture is possible.
  - **Halfmove Clock:** A plane representing the number of halfmoves since the last capture or pawn advance, normalized to a value between 0 and 1.
  - **Fullmove Number:** A plane indicating the current fullmove number, normalized to a value between 0 and 1.
  - **Move History (Optional but Recommended):** To capture temporal information and handle situations like threefold repetition, a few previous board states (e.g., the last 8 half-moves) could be stacked as additional input planes. _Note: This feature is currently a design consideration and not yet implemented in the `ChessEnv`'s `get_state_planes` method._

The total number of planes will be 21 (12 for pieces + 2 for player to move + 4 for castling rights + 1 for en passant + 1 for halfmove clock + 1 for fullmove number).
