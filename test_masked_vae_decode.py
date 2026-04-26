"""
Test script for InpaintVAEDecode node.

Run from ComfyUI root directory:
    python custom_nodes/ComfyUI-AdvancedTiling/test_masked_vae_decode.py

Verifies:
  1. Stage module detection matches decoder structure
  2. Reference features are saved at all stages
  3. Masked decode produces valid output
  4. Bleed reduction vs regular VAE decode
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def find_vae():
    """Find an available VAE file and return (path, label)."""
    import folder_paths

    # Check standalone VAEs
    for vae_dir in folder_paths.get_folder_paths("vae"):
        if os.path.exists(vae_dir):
            for f in sorted(os.listdir(vae_dir)):
                if f.endswith((".safetensors", ".pt", ".bin")):
                    return os.path.join(vae_dir, f), f"vae/{f}"

    # Fall back to checkpoint's embedded VAE
    for ckpt_dir in folder_paths.get_folder_paths("checkpoints"):
        if os.path.exists(ckpt_dir):
            for f in sorted(os.listdir(ckpt_dir)):
                if f.endswith((".safetensors", ".pt", ".bin")):
                    return os.path.join(ckpt_dir, f), f"checkpoint/{f}"

    return None, None


def load_vae_model(path):
    """Load VAE model using ComfyUI's infrastructure."""
    import comfy.utils
    import comfy.sd

    sd, _ = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd)
    vae.throw_exception_if_invalid()
    return vae


def create_test_data(latent_channels=4, H_lat=64, W_lat=64):
    """
    Create synthetic test data.

    Source latent: noise pattern A everywhere
    Inpainted latent: noise A in preserved area, noise B in masked area
    Mask: circle in the center
    """
    torch.manual_seed(42)

    z_source = torch.randn(1, latent_channels, H_lat, W_lat)

    # Create circular mask at latent resolution
    ys = torch.arange(H_lat).float() - H_lat / 2
    xs = torch.arange(W_lat).float() - W_lat / 2
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    radius = min(H_lat, W_lat) * 0.35
    dist = torch.sqrt(xx ** 2 + yy ** 2)

    # Feathered mask: 1 inside circle, 0 outside, smooth transition
    mask_lat = torch.clamp(1.0 - (dist - radius) / 3.0, 0.0, 1.0)
    mask_lat = mask_lat.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Inpainted latent: different content in masked area
    z_different = torch.randn(1, latent_channels, H_lat, W_lat) * 2.0
    z_inpaint = z_source * (1 - mask_lat) + z_different * mask_lat

    # Image-resolution mask for the node input (H, W)
    mask_img = F.interpolate(mask_lat, size=(H_lat * 8, W_lat * 8), mode='bilinear', align_corners=False)
    mask_img = mask_img.squeeze(0).squeeze(0)  # (H, W)

    return z_source, z_inpaint, mask_lat, mask_img


def save_image(tensor, path):
    """Save image tensor (B, H, W, C) to file."""
    img = tensor[0].cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def test_stage_detection(decoder):
    """Test that _get_stage_modules finds the correct modules."""
    from masked_vae_decode import _get_stage_modules

    stages = _get_stage_modules(decoder)

    print(f"\n{'='*60}")
    print("TEST 1: Stage Detection")
    print(f"{'='*60}")
    print(f"Decoder: num_resolutions={decoder.num_resolutions}, "
          f"num_res_blocks={decoder.num_res_blocks}")

    assert len(stages) >= 3, f"Expected >= 3 stages, got {len(stages)}"

    # First stage should be conv_in
    assert stages[0][0] == "conv_in", f"First stage should be conv_in, got {stages[0][0]}"
    assert stages[0][1] is decoder.conv_in

    # Second stage should be mid
    assert stages[1][0] == "mid", f"Second stage should be mid, got {stages[1][0]}"
    assert stages[1][1] is decoder.mid.block_2

    # Verify up level ordering matches decoder forward (reversed)
    up_stages = [(n, m) for n, m in stages if n.startswith("up_")]
    num_res = decoder.num_resolutions
    assert len(up_stages) == num_res, f"Expected {num_res} up stages, got {len(up_stages)}"

    # Verify upsample modules exist where expected
    for name, module in up_stages:
        i_level = int(name.split("_")[1])
        up = decoder.up[i_level]
        if i_level != 0:
            assert hasattr(up, "upsample"), f"Level {i_level} should have upsample"
            assert module is up.upsample, f"Stage {name} should use upsample"
        else:
            assert module is up.block[-1], f"Stage {name} should use last block"

    print(f"  Stages ({len(stages)}): {[n for n, _ in stages]}")
    for i, (name, mod) in enumerate(stages):
        print(f"    [{i}] {name}: {type(mod).__name__}")

    print(f"\n  Up level structure:")
    for i in range(num_res):
        up = decoder.up[i]
        has_up = hasattr(up, "upsample")
        print(f"    up[{i}]: {len(up.block)} blocks, "
              f"{len(up.attn)} attn, upsample={has_up}")

    print("  PASSED")
    return stages


def test_reference_features(vae, z_source, expected_stages):
    """Test that reference decode saves features at all stages."""
    from masked_vae_decode import _save_hook

    print(f"\n{'='*60}")
    print("TEST 2: Reference Feature Saving")
    print(f"{'='*60}")

    decoder = vae.first_stage_model.decoder
    ref_features = {}
    hooks = []
    for name, module in expected_stages:
        hooks.append(module.register_forward_hook(_save_hook(ref_features, name)))

    with torch.no_grad():
        vae.decode(z_source.to(device=vae.device, dtype=vae.vae_dtype))

    for h in hooks:
        h.remove()

    assert len(ref_features) == len(expected_stages), \
        f"Expected {len(expected_stages)} features, got {len(ref_features)}"

    for name, _ in expected_stages:
        assert name in ref_features, f"Missing feature: {name}"

    print(f"  Saved {len(ref_features)} feature maps:")
    for name, feat in ref_features.items():
        print(f"    {name}: shape={tuple(feat.shape)}, "
              f"dtype={feat.dtype}, device={feat.device}")

    print("  PASSED")
    return ref_features


def test_masked_decode(vae, z_source, z_inpaint, mask_lat, mask_img):
    """Test full masked decode and compare with regular decode."""
    from masked_vae_decode import InpaintVAEDecode

    print(f"\n{'='*60}")
    print("TEST 3: Masked Decode vs Regular Decode")
    print(f"{'='*60}")

    device = vae.device
    dtype = vae.vae_dtype

    # Regular decode of inpainted latent (has bleed)
    with torch.no_grad():
        img_regular = vae.decode(z_inpaint.to(device=device, dtype=dtype))

    # Masked decode (reduced bleed)
    node = InpaintVAEDecode()
    samples = {"samples": z_inpaint}
    source_samples = {"samples": z_source}

    (img_masked,) = node.decode(samples, source_samples, vae, mask_img)

    assert img_regular.shape == img_masked.shape, \
        f"Shape mismatch: regular={img_regular.shape} vs masked={img_masked.shape}"

    # Measure difference in preserved area (where mask=0)
    H_img, W_img = img_regular.shape[1], img_regular.shape[2]
    mask_full = F.interpolate(mask_lat.to(device=device), size=(H_img, W_img),
                              mode='bilinear', align_corners=False)
    # mask_full is (1,1,H,W), image is (B,H,W,C) — reshape to (1,H,W,1)
    mask_full = mask_full.squeeze(1).unsqueeze(-1)  # (1, H, W, 1)

    # Also decode source directly for ground truth
    with torch.no_grad():
        img_source = vae.decode(z_source.to(device=device, dtype=dtype))

    # Difference in preserved area between regular decode and source decode
    preserved_mask = (1 - mask_full).expand_as(img_regular)
    n_preserved = preserved_mask.sum()

    diff_regular = ((img_regular - img_source.to(device=device, dtype=torch.float32)).abs() * preserved_mask).sum() / n_preserved
    diff_masked = ((img_masked.to(device=device) - img_source.to(device=device, dtype=torch.float32)).abs() * preserved_mask).sum() / n_preserved

    print(f"  Preserved area avg pixel difference from source:")
    print(f"    Regular decode: {diff_regular.item():.6f}")
    print(f"    Masked decode:  {diff_masked.item():.6f}")
    print(f"    Reduction:      {(1 - diff_masked / diff_regular).item() * 100:.1f}%")

    assert diff_masked < diff_regular, \
        f"Masked decode should have less bleed: {diff_masked} vs {diff_regular}"

    # Save comparison images
    out_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(out_dir, exist_ok=True)

    save_image(img_source.cpu().float(), os.path.join(out_dir, "source_decode.png"))
    save_image(img_regular.cpu().float(), os.path.join(out_dir, "regular_decode.png"))
    save_image(img_masked.cpu().float(), os.path.join(out_dir, "masked_decode.png"))

    # Save difference maps (amplified)
    diff_reg_map = (img_regular - img_source.to(device=device, dtype=torch.float32)).abs()
    diff_mask_map = (img_masked.to(device=device) - img_source.to(device=device, dtype=torch.float32)).abs()
    amp = 10.0  # amplify for visibility
    save_image((diff_reg_map.cpu().float() * amp).clamp(0, 1),
               os.path.join(out_dir, "diff_regular_x10.png"))
    save_image((diff_mask_map.cpu().float() * amp).clamp(0, 1),
               os.path.join(out_dir, "diff_masked_x10.png"))

    print(f"\n  Saved comparison images to: {out_dir}")
    print(f"    source_decode.png    - VAE decode of source latent")
    print(f"    regular_decode.png   - VAE decode of inpainted latent (with bleed)")
    print(f"    masked_decode.png    - Masked decode (reduced bleed)")
    print(f"    diff_regular_x10.png - |regular - source| x10")
    print(f"    diff_masked_x10.png  - |masked - source| x10")
    print("  PASSED")


def main():
    print("=" * 60)
    print("InpaintVAEDecode Test Suite")
    print("=" * 60)

    # Find and load VAE
    vae_path, vae_label = find_vae()
    if not vae_path:
        print("ERROR: No VAE or checkpoint file found!")
        sys.exit(1)

    print(f"Loading VAE: {vae_label}")
    vae = load_vae_model(vae_path)
    print(f"VAE device: {vae.device}, dtype: {vae.vae_dtype}")

    decoder = vae.first_stage_model.decoder

    # Create test data matching VAE's latent channels
    z_source, z_inpaint, mask_lat, mask_img = create_test_data(
        latent_channels=vae.latent_channels
    )
    print(f"Test latent: {z_source.shape}, mask: {mask_img.shape}")

    # Run tests
    stages = test_stage_detection(decoder)
    test_reference_features(vae, z_source, stages)
    test_masked_decode(vae, z_source, z_inpaint, mask_lat, mask_img)

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
