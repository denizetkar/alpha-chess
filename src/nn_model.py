import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaChessNet(nn.Module):
    """
    Neural network for AlphaChess, inspired by AlphaZero.
    It takes board state planes as input and outputs a policy (move probabilities)
    and a value (win probability).
    """

    def __init__(self, num_residual_blocks: int = 10, num_filters: int = 256):
        super().__init__()
        self.num_residual_blocks = num_residual_blocks
        self.num_filters = num_filters

        # Initial convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(35, num_filters, kernel_size=3, padding=1),  # 35 input planes from ChessEnv
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_residual_blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),  # 2 filters for policy (e.g., one for move, one for pass)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4672),  # 4672 is a common number of possible moves in chess
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),  # 1 filter for value
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, num_filters),  # Intermediate linear layer
            nn.ReLU(),
            nn.Linear(num_filters, 1),  # Output single scalar value
            nn.Tanh(),  # Scale to [-1, 1]
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.
        x: Input tensor representing board state planes (batch_size, 35, 8, 8)
        """
        x = self.conv_block(x)
        for block in self.residual_blocks:
            x = block(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return F.log_softmax(policy_logits, dim=1), value


class ResidualBlock(nn.Module):
    """
    A standard residual block used in AlphaZero-like architectures.
    """

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add skip connection
        out = F.relu(out)
        return out


# Example Usage (for testing)
if __name__ == "__main__":
    # Test with default parameters
    model = AlphaChessNet()
    dummy_input = torch.randn(1, 35, 8, 8)  # Batch size 1, 35 planes, 8x8 board
    policy_logits, value = model(dummy_input)

    print(f"Model: {model}")
    print(f"Policy logits shape: {policy_logits.shape}")  # Expected: (1, 4672)
    print(f"Value shape: {value.shape}")  # Expected: (1, 1)

    # Test with custom parameters
    model_large = AlphaChessNet(num_residual_blocks=20, num_filters=512)
    dummy_input_large = torch.randn(2, 35, 8, 8)  # Batch size 2
    policy_logits_large, value_large = model_large(dummy_input_large)

    print(f"\nModel (Large): {model_large}")
    print(f"Policy logits (Large) shape: {policy_logits_large.shape}")  # Expected: (2, 4672)
    print(f"Value (Large) shape: {value_large.shape}")  # Expected: (2, 1)

    # Test GPU availability
    if torch.cuda.is_available():
        print("\nCUDA is available. Testing model on GPU.")
        model_gpu = AlphaChessNet().cuda()
        dummy_input_gpu = torch.randn(1, 35, 8, 8).cuda()
        policy_logits_gpu, value_gpu = model_gpu(dummy_input_gpu)
        print(f"Policy logits (GPU) device: {policy_logits_gpu.device}")
        print(f"Value (GPU) device: {value_gpu.device}")
    else:
        print("\nCUDA is not available. Model will run on CPU.")
