import pytest
import torch
import torch.nn as nn
from src.nn_model import AlphaChessNet, ResidualBlock


class TestAlphaChessNet:
    def test_model_init(self):
        model = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        assert isinstance(model, nn.Module)
        assert model.conv_block[0].in_channels == 29
        assert model.policy_head[-1].out_features == 4672
        assert model.value_head[-2].out_features == 1  # Check the Linear layer before Tanh

        # Test device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        assert next(model.parameters()).device.type == device.type  # Compare only device type

    def test_forward_pass(self):
        model = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        dummy_input = torch.randn(1, 29, 8, 8)  # Batch size 1, 29 planes, 8x8 board

        policy_logits, value = model(dummy_input)

        assert policy_logits.shape == (1, 4672)
        assert value.shape == (1, 1)

        # Check if policy_logits are log-softmaxed
        assert torch.isclose(torch.exp(policy_logits).sum(), torch.tensor(1.0))

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_forward_pass_multiple_batch_sizes(self, batch_size):
        model = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        dummy_input = torch.randn(batch_size, 29, 8, 8)

        policy_logits, value = model(dummy_input)

        assert policy_logits.shape == (batch_size, 4672)
        assert value.shape == (batch_size, 1)

    def test_value_output_range(self):
        model = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        dummy_input = torch.randn(1, 29, 8, 8)
        _, value = model(dummy_input)

        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_cuda_availability(self):
        if torch.cuda.is_available():
            model = AlphaChessNet().cuda()
            dummy_input = torch.randn(1, 29, 8, 8).cuda()
            policy_logits, value = model(dummy_input)
            assert policy_logits.device.type == "cuda"
            assert value.device.type == "cuda"
        else:
            pytest.skip("CUDA not available for testing.")


class TestResidualBlock:
    def test_residual_block_forward(self):
        num_filters = 64
        block = ResidualBlock(num_filters)
        dummy_input = torch.randn(1, num_filters, 8, 8)
        output = block(dummy_input)

        assert output.shape == dummy_input.shape
        # Ensure some transformation happened (not identical due to conv/relu)
        assert not torch.equal(output, dummy_input)


@pytest.mark.parametrize("num_residual_blocks, num_filters", [(1, 32), (5, 128), (10, 256)])
class TestAlphaChessNetCustomParams:
    def test_model_with_custom_params(self, num_residual_blocks, num_filters):
        model = AlphaChessNet(num_residual_blocks=num_residual_blocks, num_filters=num_filters)
        dummy_input = torch.randn(1, 29, 8, 8)

        policy_logits, value = model(dummy_input)

        assert policy_logits.shape == (1, 4672)
        assert value.shape == (1, 1)
