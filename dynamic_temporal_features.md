# Dynamic Temporal Segmentation and Masking in SpaMo

## Overview

This document describes the implementation of **Dynamic Temporal Segmentation and Masking** features in the SpaMo sign language translation model, inspired by the AVRET paper (Liu et al., IEEE TCSVT 2024). These features enhance the model's robustness to temporal variations in signing speed by adaptively processing video sequences.

## Features Implemented

### 1. Dynamic Segmentation (`DynamicSegmenter`)

**Purpose**: Computes frame-level importance scores based on motion detection to identify critical temporal regions.

**How it Works**:
- Uses a lightweight 1D convolutional network to analyze feature sequences.
- Computes motion scores for each frame, normalized to [0,1].
- Higher scores indicate frames with significant motion (likely gesture boundaries or key signs).
- Acts as a proxy for dynamic clip partitioning from AVRET.

**Technical Details**:
- Input: Feature tensor (B, T, C) and sequence lengths.
- Output: Importance scores (B, T) for guiding masking.
- Trained end-to-end with the model.

### 2. Adaptive Masking (`AdaptiveMasker`)

**Purpose**: Applies variable-length masking to low-importance regions during training, forcing the model to infer missing information from context.

**How it Works**:
- Identifies low-importance frames (scores below median).
- Randomly selects segments in these regions for masking.
- Masks with zeros for variable lengths (min_mask_len to max_mask_len).
- Only applied during training with probability `mask_prob`.

**Benefits**:
- Improves robustness to temporal distortions (speed variations).
- Prevents overfitting to fixed patterns.
- Encourages better contextual understanding.

## Pipeline Changes

The visual input processing pipeline in `FlanT5SLT.prepare_visual_inputs()` has been updated:

### Original Pipeline:
1. Extract features (spatial, spatiotemporal, pose)
2. Apply data augmentation (if enabled)
3. Fuse features (joint/adaptive/single mode)
4. Temporal encoding (TemporalConv)
5. Fusion projection to T5 hidden size

### Updated Pipeline:
1. Extract features (spatial, spatiotemporal, pose)
2. Apply data augmentation (if enabled)
3. Fuse features (joint/adaptive/single mode)
4. **NEW**: Dynamic Segmentation (compute importance scores)
5. **NEW**: Adaptive Segmentation (aggregate frames into variable-length temporal segments)
6. **NEW**: Adaptive Masking (apply variable masking if training)
7. Temporal encoding (TemporalConv)
8. Fusion projection to T5 hidden size

**Key Points**:
- Segmentation and masking are applied after fusion for joint/adaptive modes, or after individual projections for single modes.
- When `use_dynamic_segmentation=true`, the model now pools frame groups into adaptive segments before TemporalConv.
- When `use_adaptive_masking=true`, masking is applied on the segmented stream, not on raw frames.
- Masking is training-only to avoid degrading inference performance.
- Importance scores are computed on-the-fly and not stored.

### What “segmentation only” now means
- `use_dynamic_segmentation=true` without masking now transforms the input into adaptive temporal segments.
- Low-motion boundaries become segment boundaries, so TemporalConv processes a shorter, gesture-aligned sequence.
- This gives the same benefit as AVRET-style segmentation: variable-length temporal units instead of rigid frame windows.

## Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_dynamic_segmentation` | bool | False | Enable motion-based importance scoring |
| `motion_threshold` | float | 0.5 | Sensitivity for motion detection (0.3-0.7 recommended) |
| `use_adaptive_masking` | bool | False | Enable variable-length masking during training |
| `mask_prob` | float | 0.15 | Probability of masking a sequence (0.1-0.3 typical) |
| `min_mask_len` | int | 5 | Minimum number of consecutive frames to mask |
| `max_mask_len` | int | 20 | Maximum number of consecutive frames to mask |

## Tuning Guide

### Step-by-Step Tuning Process:

1. **Baseline Training**: Train without new features to establish baseline BLEU/loss.

2. **Enable Segmentation Only**:
   - Set `use_dynamic_segmentation: true`
   - Keep `use_adaptive_masking: false`
   - Monitor validation metrics for stability.

3. **Tune Motion Threshold**:
   - Lower values (0.3) = more sensitive to motion, more frames marked important.
   - Higher values (0.7) = less sensitive, fewer frames marked important.
   - Adjust based on dataset (faster signing may need lower threshold).

4. **Enable Masking**:
   - Set `use_adaptive_masking: true`
   - Start with default `mask_prob: 0.15`

5. **Tune Masking Parameters**:
   - **mask_prob**: Increase (0.2-0.3) if overfitting, decrease (0.1) if underfitting.
   - **min_mask_len/max_mask_len**: Increase for longer sequences (e.g., 10-30 for 512-frame videos).
   - Monitor training stability and validation BLEU.

### Expected Effects:
- **Positive**: Better generalization to varying signing speeds, improved BLEU on diverse test sets.
- **Potential Issues**: Slight increase in training time, possible temporary drop in validation metrics during initial tuning.
- **Validation**: Compare BLEU on held-out data with/without features enabled.

### Best Practices:
- Enable both features together for full benefit.
- Use with `cross_modal_align: true` for combined contrastive losses.
- Tune on validation set, not training set.
- Start conservative and gradually increase masking intensity.

## Example Configuration

```yaml
model:
  # Enable features
  use_dynamic_segmentation: true
  motion_threshold: 0.4
  use_adaptive_masking: true
  mask_prob: 0.2
  min_mask_len: 8
  max_mask_len: 25
  
  # Recommended complementary settings
  cross_modal_align: true
  combined_loss: true
  alpha: 1.0
```

## Troubleshooting

- **Training instability**: Reduce `mask_prob` or disable masking temporarily.
- **No improvement**: Check `motion_threshold` - too high/low may not identify important frames.
- **Slow training**: Features add minimal compute; monitor GPU usage.
- **Inference issues**: Ensure masking is disabled during eval (handled automatically).

For questions or issues, refer to the code in `spamo/dynamic_segmentation.py` and `spamo/t5_slt.py`.