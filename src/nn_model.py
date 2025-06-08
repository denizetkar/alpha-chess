import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants for neural network dimensions
INPUT_PLANES = 29  # 21 board planes + 8 history planes
POLICY_PLANES = 73  # AlphaZero's 73 action planes
FLATTENED_POLICY_SIZE = POLICY_PLANES * 8 * 8  # 73 * 8 * 8 = 4672


class AlphaChessNet(nn.Module):
    """
    Neural network for AlphaChess, inspired by AlphaZero.

    This network takes a batch of board state planes as input and outputs
    a policy (move probabilities) and a value (win probability).
    The architecture consists of an initial convolutional block,
    a series of residual blocks, and two heads: a policy head and a value head.
    """

    def __init__(self, num_residual_blocks: int = 10, num_filters: int = 256) -> None:
        """
        Initializes the AlphaChessNet.

        Args:
            num_residual_blocks (int): Number of residual blocks in the network.
            num_filters (int): Number of filters used in convolutional layers
                                and intermediate linear layers.
        """
        super().__init__()
        self.num_residual_blocks: int = num_residual_blocks
        self.num_filters: int = num_filters

        # Initial convolutional block
        # INPUT_PLANES from ChessEnv (21 + 8 history)
        self.conv_block: nn.Sequential = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        self.residual_blocks: nn.ModuleList = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # Policy head
        # AlphaZero uses 73 action planes (8x8 board) for move representation.
        # This includes:
        # - 8 directions * 7 steps for Queen/Rook/Bishop moves (56 planes)
        # - 8 Knight moves (8 planes)
        # - 9 King moves (8 directions + 1 for castling, often simplified to 1 plane for all king moves,
        #   and pawn moves including 2 pushes, 2 captures, 3 underpromotions for each)
        # Total POLICY_PLANES: 56 + 8 + 9 (for king moves, pawn moves, and underpromotions)
        # This is a common interpretation.
        self.policy_head: nn.Sequential = nn.Sequential(
            nn.Conv2d(num_filters, POLICY_PLANES, kernel_size=1),  # POLICY_PLANES action planes
            nn.BatchNorm2d(POLICY_PLANES),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(FLATTENED_POLICY_SIZE, FLATTENED_POLICY_SIZE),  # Flattened to FLATTENED_POLICY_SIZE
        )

        # Value head
        self.value_head: nn.Sequential = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),  # 1 filter for value
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, num_filters),  # Intermediate linear layer
            nn.ReLU(),
            nn.Linear(num_filters, 1),  # Output single scalar value
            nn.Tanh(),  # Scale to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing board state planes
                              (batch_size, INPUT_PLANES, 8, 8).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - policy_logits (torch.Tensor): Log probabilities of moves
                                                (batch_size, FLATTENED_POLICY_SIZE).
                - value (torch.Tensor): Predicted win probability scaled to [-1, 1]
                                        (batch_size, 1).
        """
        x = self.conv_block(x)
        for block in self.residual_blocks:
            x = block(x)

        policy_logits: torch.Tensor = self.policy_head(x)
        value: torch.Tensor = self.value_head(x)

        return F.log_softmax(policy_logits, dim=1), value


class ResidualBlock(nn.Module):
    """
    A standard residual block used in AlphaZero-like architectures.

    Consists of two convolutional layers with batch normalization and ReLU activations,
    with a skip connection adding the input to the output of the second convolutional layer.
    """

    def __init__(self, num_filters: int) -> None:
        """
        Initializes a ResidualBlock.

        Args:
            num_filters (int): Number of filters for the convolutional layers.
        """
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(num_filters)
        self.conv2: nn.Conv2d = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor to the block.

        Returns:
            torch.Tensor: Output tensor after applying residual connections and activations.
        """
        residual: torch.Tensor = x
        out: torch.Tensor = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add skip connection
        out = F.relu(out)
        return out
