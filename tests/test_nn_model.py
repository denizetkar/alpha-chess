import pytest
import torch
import torch.nn as nn
from src.nn_model import AlphaChessNet, ResidualBlock, INPUT_PLANES, FLATTENED_POLICY_SIZE


class TestAlphaChessNet:
    def test_model_init(self) -> None:
        """
        Tests the initialization of AlphaChessNet, including parameter assignment,
        output feature dimensions, and device placement.
        """
        model: AlphaChessNet = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        assert isinstance(model, nn.Module)
        assert model.conv_block[0].in_channels == INPUT_PLANES
        assert model.policy_head[-1].out_features == FLATTENED_POLICY_SIZE
        assert model.value_head[-2].out_features == 1

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        assert next(model.parameters()).device.type == device.type

    def test_forward_pass(self) -> None:
        """
        Tests the forward pass of AlphaChessNet, verifying output shapes and
        that policy logits are correctly log-softmaxed.
        """
        model: AlphaChessNet = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        dummy_input: torch.Tensor = torch.randn(1, INPUT_PLANES, 8, 8)

        policy_logits: torch.Tensor
        value: torch.Tensor
        policy_logits, value = model(dummy_input)

        assert policy_logits.shape == (1, FLATTENED_POLICY_SIZE)
        assert value.shape == (1, 1)

        assert torch.isclose(torch.exp(policy_logits).sum(), torch.tensor(1.0), atol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_forward_pass_multiple_batch_sizes(self, batch_size: int) -> None:
        """
        Tests the forward pass with multiple batch sizes to ensure consistent output shapes.
        """
        model: AlphaChessNet = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        dummy_input: torch.Tensor = torch.randn(batch_size, INPUT_PLANES, 8, 8)

        policy_logits: torch.Tensor
        value: torch.Tensor
        policy_logits, value = model(dummy_input)

        assert policy_logits.shape == (batch_size, FLATTENED_POLICY_SIZE)
        assert value.shape == (batch_size, 1)

    def test_value_output_range(self) -> None:
        """
        Tests that the value output of AlphaChessNet is within the expected range [-1, 1].
        """
        model: AlphaChessNet = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        dummy_input: torch.Tensor = torch.randn(1, INPUT_PLANES, 8, 8)
        _: torch.Tensor
        value: torch.Tensor
        _, value = model(dummy_input)

        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_cuda_availability(self) -> None:
        """
        Tests that the model correctly utilizes CUDA if available,
        verifying device placement of outputs.
        """
        if torch.cuda.is_available():
            model: AlphaChessNet = AlphaChessNet().cuda()
            dummy_input: torch.Tensor = torch.randn(1, INPUT_PLANES, 8, 8).cuda()
            policy_logits: torch.Tensor
            value: torch.Tensor
            policy_logits, value = model(dummy_input)
            assert policy_logits.device.type == "cuda"
            assert value.device.type == "cuda"
        else:
            pytest.skip("CUDA not available for testing.")


class TestResidualBlock:
    def test_residual_block_forward(self) -> None:
        """
        Tests the forward pass of a ResidualBlock, verifying output shape
        and that the output is transformed from the input.
        """
        num_filters: int = 64
        block: ResidualBlock = ResidualBlock(num_filters)
        dummy_input: torch.Tensor = torch.randn(1, num_filters, 8, 8)
        output: torch.Tensor = block(dummy_input)

        assert output.shape == dummy_input.shape
        assert not torch.equal(output, dummy_input)


@pytest.mark.parametrize("num_residual_blocks, num_filters", [(1, 32), (5, 128), (10, 256)])
class TestAlphaChessNetCustomParams:
    def test_model_with_custom_params(self, num_residual_blocks: int, num_filters: int) -> None:
        """
        Tests AlphaChessNet with various custom parameters for residual blocks and filters,
        verifying output shapes.
        """
        model: AlphaChessNet = AlphaChessNet(num_residual_blocks=num_residual_blocks, num_filters=num_filters)
        dummy_input: torch.Tensor = torch.randn(1, INPUT_PLANES, 8, 8)

        policy_logits: torch.Tensor
        value: torch.Tensor
        policy_logits, value = model(dummy_input)

        assert policy_logits.shape == (1, FLATTENED_POLICY_SIZE)
        assert value.shape == (1, 1)
