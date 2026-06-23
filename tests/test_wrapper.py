"""Tests for StageGuardWrapper."""

import numpy as np
import torch

from stageguard.wrapper import StageGuardWrapper


class TestStageGuardWrapper:
    def test_forward_shape(self, tiny_backbone, dummy_config):
        wrapper = StageGuardWrapper(tiny_backbone, dummy_config)
        x = torch.randn(2, 10, 8)
        logits = wrapper(x)
        assert logits.shape == (2, 10, 3)

    def test_training_step(self, tiny_backbone, dummy_config):
        wrapper = StageGuardWrapper(tiny_backbone, dummy_config)
        x = torch.randn(2, 10, 8)
        targets = torch.randint(0, 3, (2, 10))
        loss, details = wrapper.training_step(x, targets)
        assert loss.dim() == 0
        assert "ce_loss" in details
        assert "trans_loss" in details

    def test_training_step_gradient(self, tiny_backbone, dummy_config):
        wrapper = StageGuardWrapper(tiny_backbone, dummy_config)
        x = torch.randn(2, 10, 8)
        targets = torch.randint(0, 3, (2, 10))
        loss, _ = wrapper.training_step(x, targets)
        loss.backward()
        for p in wrapper.parameters():
            assert p.grad is not None

    def test_predict_shape(self, tiny_backbone, dummy_config):
        wrapper = StageGuardWrapper(tiny_backbone, dummy_config)
        x = torch.randn(2, 10, 8)
        preds = wrapper.predict(x)
        assert preds.shape == (2, 10)
        assert np.issubdtype(preds.dtype, np.integer)

    def test_predict_with_sqi(self, tiny_backbone, dummy_config):
        wrapper = StageGuardWrapper(tiny_backbone, dummy_config)
        torch.manual_seed(0)
        x = torch.randn(2, 10, 8)

        # SQI = 1 is a no-op: identical to decoding without SQI.
        base = wrapper.predict(x)
        ones = wrapper.predict(x, sqi_scores=np.ones((2, 10)))
        assert ones.shape == (2, 10)
        assert np.array_equal(base, ones)

        # SQI = 0 damps emissions to uniform, so the backbone output is ignored:
        # two different inputs decode to the same path.
        x2 = torch.randn(2, 10, 8)
        z1 = wrapper.predict(x, sqi_scores=np.zeros((2, 10)))
        z2 = wrapper.predict(x2, sqi_scores=np.zeros((2, 10)))
        assert np.array_equal(z1, z2)

    def test_arbitrary_backbone(self, dummy_config):
        """Any nn.Module with correct output shape works."""
        import torch.nn as nn

        class CustomBackbone(nn.Module):
            def forward(self, x):
                B, T, _ = x.shape
                return torch.randn(B, T, 3)

        wrapper = StageGuardWrapper(CustomBackbone(), dummy_config)
        x = torch.randn(1, 15, 4)
        logits = wrapper(x)
        assert logits.shape == (1, 15, 3)
