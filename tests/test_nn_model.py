import pytest
import torch
import torch.nn as nn
from src.nn_model import AlphaChessNet


class TestAlphaChessNet:
    def test_model_init(self):
        model = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        assert isinstance(model, nn.Module)
        assert model.conv_block[0].in_channels == 21
        assert model.policy_head[-1].out_features == 4672
        assert model.value_head[-2].out_features == 1  # Check the Linear layer before Tanh

        # Test device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        assert next(model.parameters()).device.type == device.type  # Compare only device type

    def test_forward_pass(self):
        model = AlphaChessNet(num_residual_blocks=2, num_filters=64)
        dummy_input = torch.randn(1, 21, 8, 8)  # Batch size 1, 21 planes, 8x8 board

        policy_logits, value = model(dummy_input)

        assert policy_logits.shape == (1, 4672)
        assert value.shape == (1, 1)

        # Check if policy_logits are log-softmaxed
        assert torch.isclose(torch.exp(policy_logits).sum(), torch.tensor(1.0))

    def test_cuda_availability(self):
        if torch.cuda.is_available():
            model = AlphaChessNet().cuda()
            dummy_input = torch.randn(1, 21, 8, 8).cuda()
            policy_logits, value = model(dummy_input)
            assert policy_logits.device.type == "cuda"
            assert value.device.type == "cuda"
        else:
            pytest.skip("CUDA not available for testing.")
