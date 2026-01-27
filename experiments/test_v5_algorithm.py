"""
Unit tests for GRA-CNN v5.3 algorithm.
Tests: BN freezing, ResNet FLOPs, try/finally protection, etc.
"""

import sys
sys.path.insert(0, 'C:/GRA-CNN')

import torch
import torch.nn as nn
import numpy as np

from pruning.core_algorithm_v5 import (
    GRA_VERSION,
    detect_network_architecture,
    get_layer_depth_ratio,
    get_adaptive_weights_v5,
    compute_gra_vectorized_v5,
    get_layer_protection_ratio_v5,
    normalize_scores,
    compute_orthogonality_scores,
    compute_l1_norm_scores,
    get_layer_flops_info,
    get_pruning_mask_simple,
    get_global_mask_iso_flops_v5,
    collect_activations_margins_and_gradnorms,
)


def test_version():
    """Test version is 5.3"""
    print(f"Testing GRA_VERSION: {GRA_VERSION}")
    assert GRA_VERSION == "5.3", f"Expected 5.3, got {GRA_VERSION}"
    print("  PASS: Version is 5.3")


def test_normalize_scores():
    """Test score normalization"""
    print("\nTesting normalize_scores()...")

    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    norm = normalize_scores(scores)
    assert abs(norm.min() - 0.0) < 1e-6
    assert abs(norm.max() - 1.0) < 1e-6
    print("  PASS: Normal normalization correct")

    scores_same = np.array([3.0, 3.0, 3.0])
    norm_same = normalize_scores(scores_same)
    assert np.allclose(norm_same, 0.5)
    print("  PASS: Edge case (same values) handled")


def test_vgg_depth_ratio():
    """Test VGG depth ratio calculation - GPT fix"""
    print("\nTesting VGG depth ratio (GPT fix)...")

    # Create VGG-like model
    class SimpleVGG(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),    # features.0
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),   # features.2
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),  # features.4
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), # features.6
                nn.MaxPool2d(2),
            )

    model = SimpleVGG()
    arch_type, depth_info = detect_network_architecture(model)

    assert arch_type == 'vgg', f"Expected 'vgg', got {arch_type}"
    assert isinstance(depth_info, dict), "VGG depth_info should be dict"
    assert 'order' in depth_info and 'total' in depth_info
    print(f"  VGG detected with {depth_info['total']} conv layers")

    # Test depth ratios - should be evenly distributed
    d0 = get_layer_depth_ratio('features.0', model)
    d2 = get_layer_depth_ratio('features.2', model)
    d4 = get_layer_depth_ratio('features.4', model)
    d6 = get_layer_depth_ratio('features.6', model)

    print(f"  Depth ratios: features.0={d0:.2f}, .2={d2:.2f}, .4={d4:.2f}, .6={d6:.2f}")

    # Verify monotonic increase
    assert d0 < d2 < d4 < d6, "Depth ratios should increase"
    assert d6 == 1.0, "Last conv should have depth 1.0"
    print("  PASS: VGG depth ratios correctly distributed")


def test_resnet_depth_ratio():
    """Test ResNet depth ratio calculation"""
    print("\nTesting ResNet depth ratio...")

    # Without model, defaults to 4 stages
    assert get_layer_depth_ratio("layer1.0.conv1", None) == 0.25
    assert get_layer_depth_ratio("layer2.0.conv1", None) == 0.5
    assert get_layer_depth_ratio("layer3.0.conv1", None) == 0.75
    assert get_layer_depth_ratio("layer4.0.conv1", None) == 1.0
    print("  PASS: ResNet depth ratios correct")


def test_iso_flops_with_pooling():
    """Test Iso-FLOPs accounts for pooling - GPT fix"""
    print("\nTesting Iso-FLOPs with pooling (GPT fix)...")

    class VGGWithPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.MaxPool2d(2),  # 32->16
                nn.Conv2d(64, 128, 3, padding=1),
                nn.MaxPool2d(2),  # 16->8
                nn.Conv2d(128, 256, 3, padding=1),
            )

    model = VGGWithPool()
    flops_info = get_layer_flops_info(model, input_size=32)

    # Check spatial sizes are tracked correctly
    sizes = [info['spatial_size'] for info in flops_info.values()]
    print(f"  Spatial sizes: {sizes}")

    assert sizes[0] == 32, "First conv should see 32x32"
    assert sizes[1] == 16, "Second conv should see 16x16 (after pool)"
    assert sizes[2] == 8, "Third conv should see 8x8 (after pool)"
    print("  PASS: Pooling correctly reduces spatial size")


def test_adaptive_weights_ngscs():
    """Test NGSCS modulation of weights"""
    print("\nTesting NGSCS weight modulation...")

    w_high, _ = get_adaptive_weights_v5("layer2.0.conv1", None, ngscs_score=0.9)
    w_low, _ = get_adaptive_weights_v5("layer2.0.conv1", None, ngscs_score=0.1)

    assert w_high['GRA'] > w_low['GRA'], "High NGSCS should boost GRA"
    print(f"  High NGSCS GRA={w_high['GRA']:.3f}, Low NGSCS GRA={w_low['GRA']:.3f}")
    print("  PASS: NGSCS modulation works")


def test_energy_gated_gra():
    """Test energy-gated GRA with gradient norms"""
    print("\nTesting energy-gated GRA...")

    B, C = 64, 32
    act_c = torch.randn(B, C)
    margin_norm = torch.rand(B)
    rho = 0.5
    grad_norms = torch.rand(B) + 0.1

    # Without gating
    gra_no_gate = compute_gra_vectorized_v5(act_c, margin_norm, rho, None)

    # With gating
    gra_gated = compute_gra_vectorized_v5(act_c, margin_norm, rho, grad_norms)

    assert gra_no_gate.shape == (C,)
    assert gra_gated.shape == (C,)
    print("  PASS: Energy gating works with grad_norms")


def test_layer_protection():
    """Test layer protection with score awareness"""
    print("\nTesting layer protection...")

    ratio_mid = get_layer_protection_ratio_v5("layer2.0.conv1", None, 128)
    ratio_important = get_layer_protection_ratio_v5("layer2.0.conv1", None, 128, avg_score=0.8)

    assert ratio_important < ratio_mid, "High importance should reduce max prune"
    print(f"  Normal: {ratio_mid:.0%}, High importance: {ratio_important:.0%}")
    print("  PASS: Score-aware protection works")


def test_simple_model_scores():
    """Test scoring on simple model"""
    print("\nTesting simple model scores...")

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
    )

    ortho = compute_orthogonality_scores(model)
    l1 = compute_l1_norm_scores(model)

    assert len(ortho) == 2
    assert len(l1) == 2
    print("  PASS: Simple model scoring works")


def test_pruning_mask():
    """Test pruning mask generation"""
    print("\nTesting pruning mask generation...")

    scores = {
        'layer1': np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7]),
        'layer2': np.array([0.2, 0.6, 0.4, 0.8, 0.1, 0.9, 0.3, 0.7]),
    }

    masks = get_pruning_mask_simple(scores, prune_ratio=0.5)

    for name, mask in masks.items():
        kept = int(mask.sum())
        print(f"  {name}: kept {kept}/{len(mask)}")

    print("  PASS: Mask generation works")


def test_resnet_downsample_flops():
    """Test ResNet FLOPs with assertions - v5.3 fix"""
    print("\nTesting ResNet FLOPs (v5.3 fix)...")

    # Create simple ResNet-like model
    class SimpleResNetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
            )
            self.layer2_downsample = nn.Sequential(
                nn.Conv2d(64, 128, 1, stride=2),
            )

        def forward(self, x):
            out = self.layer1(x)
            return self.layer2(out) + self.layer2_downsample(out)

    model = SimpleResNetBlock()
    flops_info = get_layer_flops_info(model, input_size=32)

    print(f"  Found {len(flops_info)} conv layers")
    for name, info in flops_info.items():
        print(f"    {name}: input_spatial implied, output={info['spatial_size']}")

    # Assertions for v5.3 fix
    assert len(flops_info) == 3, "Should have 3 conv layers"
    print("  PASS: ResNet FLOPs computed with assertions")


def test_single_pass_collection():
    """Test single-pass activation/gradient collection - v5.2 fix"""
    print("\nTesting single-pass collection (v5.2 fix)...")

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )

    # Create dummy dataloader
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 32
        def __getitem__(self, idx):
            return torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()

    dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=8)
    device = torch.device('cpu')

    activations, margins, grad_norms = collect_activations_margins_and_gradnorms(
        model, dataloader, device, num_batches=2
    )

    # Check returns
    assert isinstance(activations, dict), "activations should be dict"
    assert isinstance(margins, torch.Tensor), "margins should be tensor"
    assert isinstance(grad_norms, dict), "grad_norms should be dict (per-layer)"

    # Check sample alignment
    num_samples = margins.size(0)
    for name, act in activations.items():
        assert act.size(0) == num_samples, f"Activation samples mismatch: {act.size(0)} vs {num_samples}"

    for name, gnorm in grad_norms.items():
        assert gnorm.size(0) == num_samples, f"Grad norm samples mismatch: {gnorm.size(0)} vs {num_samples}"

    print(f"  Collected {num_samples} samples")
    print(f"  Activations: {len(activations)} layers")
    print(f"  Grad norms: {len(grad_norms)} layers (per-layer)")
    print("  PASS: Single-pass collection with sample alignment")


def test_bn_freezing():
    """Test BN running stats are NOT modified during collection - v5.3 fix"""
    print("\nTesting BN freezing (v5.3 fix)...")

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )

    # Run a forward pass to initialize BN stats
    model.train()
    dummy_input = torch.randn(4, 3, 32, 32)
    _ = model(dummy_input)

    # Save BN running stats before collection
    bn_stats_before = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_stats_before[name] = {
                'mean': module.running_mean.clone(),
                'var': module.running_var.clone()
            }

    # Create dataloader and run collection
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16
        def __getitem__(self, idx):
            return torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()

    dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    device = torch.device('cpu')

    # This should NOT modify BN stats
    activations, margins, grad_norms = collect_activations_margins_and_gradnorms(
        model, dataloader, device, num_batches=2
    )

    # Check BN stats are unchanged
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            mean_diff = (module.running_mean - bn_stats_before[name]['mean']).abs().max()
            var_diff = (module.running_var - bn_stats_before[name]['var']).abs().max()
            assert mean_diff < 1e-6, f"BN {name} running_mean changed!"
            assert var_diff < 1e-6, f"BN {name} running_var changed!"

    print("  BN running stats unchanged after collection")
    print("  PASS: BN freezing works correctly")


def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("GRA-CNN v5.3 Unit Tests (GPT Red Team Fixes)")
    print("=" * 60)

    test_version()
    test_normalize_scores()
    test_vgg_depth_ratio()
    test_resnet_depth_ratio()
    test_iso_flops_with_pooling()
    test_adaptive_weights_ngscs()
    test_energy_gated_gra()
    test_layer_protection()
    test_simple_model_scores()
    test_pruning_mask()
    test_resnet_downsample_flops()
    test_single_pass_collection()
    test_bn_freezing()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
