# LIFE: Localized Inference-Free Enhancement for Efficient Edge Classification

A post-training compression study for transformer-based vision models on resource-constrained edge devices, using MobileViT-S as the deployment backbone. The work distinguishes model footprint reduction from inference-latency reduction — a distinction that is frequently conflated in the compression literature — and introduces a validation-driven, mixed-precision quantization policy that shields sensitive operators from precision loss.

**Backbone:** MobileViT-S
**Benchmarks:** PatchCamelyon (PCam) for architecture selection; APTOS 2019 Blindness Detection for quantization and deployment evaluation
**Reproducibility:** https://github.com/harvi374/LIFE.git

---

## Table of Contents

- [Overview](#overview)
- [Contributions](#contributions)
- [Method](#method)
  - [Backbone Selection Protocol](#backbone-selection-protocol)
  - [Compression and Quantization Pipeline](#compression-and-quantization-pipeline)
- [Results](#results)
  - [Architecture Comparison](#architecture-comparison)
  - [Quantization Safety Ablation](#quantization-safety-ablation)
  - [Compression Footprint and Precision Variants](#compression-footprint-and-precision-variants)
  - [Hardware-Dependent Latency](#hardware-dependent-latency)
  - [Clinical Diagnostic Metric Retention](#clinical-diagnostic-metric-retention)
- [Discussion](#discussion)
- [Limitations and Future Work](#limitations-and-future-work)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [References](#references)

---

## Overview

Transformer and hybrid CNN-transformer architectures achieve strong performance in medical and general-purpose vision tasks, but their storage and memory requirements complicate deployment on edge hardware. Compression is frequently evaluated in terms of inference-latency improvement, an outcome that is contingent on the availability of dedicated low-precision execution units (for example, Intel DL Boost / VNNI, or Tensor Cores) on the target hardware. Absent such support, quantized inference may not run faster, and in some cases runs slower, than full-precision inference, even though the serialized model is substantially smaller.

This work separates these two concerns explicitly. It first identifies an effective compact backbone through a controlled, multi-architecture comparison, then studies post-training quantization (PTQ) of that backbone with a focus on **footprint reduction achieved independently of hardware-specific latency acceleration**, while validating that diagnostic performance is preserved under compression using clinically relevant metrics (sensitivity, specificity, calibrated and fixed-threshold F1, AUC with 95% confidence intervals).

## Contributions

1. A controlled comparison of eight compact CNN and CNN-transformer architectures under a matched training protocol, establishing MobileViT-S as an effective edge-oriented backbone.
2. A multi-stage post-training quantization pipeline achieving up to 42.6% model-size reduction (19.01 MB to 10.90 MB) with near-zero loss in diagnostic accuracy.
3. A validation-driven mixed-precision policy — **SAFE_CONV** and **Conv+MatMul QDQ** — that protects sensitive stem and attention operators from quantization, avoiding the catastrophic accuracy collapse (AUC 0.8164) observed under unguided stem quantization.
4. A clinical metric assessment with 95% confidence intervals demonstrating preservation of sensitivity (0.9624 vs. 0.9543) and specificity (0.9363 vs. 0.9474) under the selected compression configuration.

---

## Method

### Backbone Selection Protocol

Eight lightweight architectures are evaluated under an identical training protocol to select a deployment backbone: **ResNet-18**, **MobileNetV3-Small**, **EfficientNet-B0**, **MobileViT-S**, **EdgeNeXt-X-Small**, **MobileViTv2-050**, **FastViT-T8**, and **RepViT-M0.9**.

- **Dataset:** PatchCamelyon (PCam), 96×96 histopathology lymph-node patches (train / validation / test: 50,000 / 5,000 / 5,000).
- **Protocol:** eight training epochs, Adam optimizer, learning rate 3×10⁻⁴, weight decay 1×10⁻⁴, evaluated across three random seeds.
- **Selection criterion:** highest test AUC. MobileViT-S is selected and carried forward to the quantization study.

### Compression and Quantization Pipeline

The selected backbone is subsequently fine-tuned and evaluated on **APTOS 2019 Blindness Detection**, a clinical benchmark of high-resolution retinal fundus images for diabetic retinopathy screening. The golden model is exported to ONNX in FP32 format (19.01 MB) and compressed along two paths:

**1. FP16 conversion** — float16 casting applied to initializers and supported graph operators.

**2. Mixed-precision static INT8 PTQ (QDQ)** — performed with ONNX Runtime's `quantize_static`, calibrated on 733 class-balanced representative images. Candidate operator sets for quantization are selected according to validation AUC, yielding three configurations:

| Configuration | Definition | Rationale |
|---|---|---|
| `STEM_CONV` | Quantizes the 13 early stem convolution nodes | Included as a negative control to probe operator sensitivity |
| `SAFE_CONV` (Conv-only) | Restricts quantization to 10 convolution nodes that do not directly feed sensitive operations; MatMul, LayerNormalization, and Softmax remain in FP32 | Establishes a conservative, accuracy-preserving floor |
| `Conv+MatMul QDQ` (Gen-4) | Quantizes 10 Conv nodes and 36 non-attention MatMul nodes (46 nodes total); matrix multiplications directly involved in attention scoring (QKᵀ and softmax(QKᵀ)V) remain in FP32 | Extends compression beyond convolutions while explicitly shielding attention computation |

---

## Results

### Architecture Comparison

Test-set performance across the eight compact architectures (mean ± SD over three seeds), evaluated on PCam:

| Model | Test AUC | Test F1 | Size (MB) |
|---|---|---|---|
| MobileNetV3-Small | 0.8974 ± 0.0041 | 0.7649 ± 0.0270 | 5.93 |
| ResNet-18 | 0.9266 ± 0.0046 | 0.8374 ± 0.0069 | 42.72 |
| FastViT-T8 | 0.9279 ± 0.0070 | 0.8063 ± 0.0174 | 12.72 |
| RepViT-M0.9 | 0.9321 ± 0.0046 | 0.7434 ± 0.0788 | 18.46 |
| EfficientNet-B0 | 0.9380 ± 0.0040 | 0.8112 ± 0.0050 | 15.58 |
| MobileViTv2-050 | 0.9427 ± 0.0076 | 0.8454 ± 0.0080 | 4.38 |
| EdgeNeXt-X-Small | 0.9438 ± 0.0076 | 0.8529 ± 0.0164 | 8.26 |
| **MobileViT-S (selected)** | **0.9520 ± 0.0038** | **0.8577 ± 0.0033** | 19.02 |

MobileViT-S attains the highest test AUC and F1 among the evaluated architectures, and its seed-to-seed stability is confirmed across validation and test splits.

### Quantization Safety Ablation

Unguided quantization of the early stem convolutions produces a severe collapse in diagnostic performance, motivating the SAFE_CONV shielding policy:

| Policy | Test AUC | Test F1 |
|---|---|---|
| SAFE_CONV (selected) | 0.9829 (95% CI: 0.9744–0.9902) | 0.9481 |
| STEM_CONV (unshielded) | 0.8164 (95% CI: 0.7864–0.8497) | 0.7204 |

The comparison confirms that operator sensitivity is highly non-uniform: naively quantizing early feature-extraction layers is catastrophic, while a validation-guided candidate selection policy preserves accuracy essentially intact.

### Compression Footprint and Precision Variants

Full performance, footprint, and execution comparison across precision variants on the golden model:

| Variant | Precision | Device | Size (MB) | Compression | Test AUC (95% CI) | F1 (Fixed) | F1 (Calibrated) | Sensitivity | Specificity | Median Latency (ms) | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileViT FP32 | FP32 | T4 GPU | 19.01 | 1.00× | 0.9841 (0.9759–0.9908) | 0.9517 | 0.9588 | 0.9543 | 0.9474 | 2.77 | 360.9 |
| MobileViT FP16 | FP16 | T4 GPU | 10.51 | 1.81× | 0.9841 (0.9758–0.9907) | 0.9531 | 0.9588 | 0.9570 | 0.9474 | 2.93 | 341.8 |
| MobileViT FP32 | FP32 | CPU | 19.01 | 1.00× | 0.9841 (0.9759–0.9908) | 0.9517 | 0.9588 | 0.9543 | 0.9474 | 6.99 | 143.2 |
| INT8 Conv-only | INT8 | CPU | 19.14 | 0.99× | 0.9829 (0.9744–0.9902) | 0.9481 | 0.9516 | 0.9570 | 0.9363 | 9.98 | 100.2 |
| **INT8 Conv+MatMul** | INT8 | CPU | **10.90** | **1.74×** | **0.9832 (0.9748–0.9903)** | 0.9509 | 0.9516 | 0.9624 | 0.9363 | 9.80 | 102.0 |

*The two F1 columns correspond to fixed-threshold and calibrated-threshold evaluation, respectively.*

The Conv+MatMul (Gen-4) configuration is the paper's headline result: a **42.6% reduction in serialized model footprint** (19.01 MB → 10.90 MB, 1.74× compression) with test AUC essentially unchanged from the FP32 baseline (0.9832 vs. 0.9841). Note that the INT8 Conv-only variant achieves *no* footprint reduction (19.14 MB, 0.99×) despite quantization, since without also compressing the MatMul operators the serialized size is not meaningfully reduced.

### Hardware-Dependent Latency

Measured on the evaluation hardware (NVIDIA T4 GPU; CPU without dedicated INT8 vector instructions):

| Configuration | Device | Median Latency | Throughput |
|---|---|---|---|
| FP32 | T4 GPU | 2.77 ms | 360.9 FPS |
| FP32 | CPU | 6.99 ms | 143.2 FPS |
| INT8 Conv+MatMul | CPU | 9.80 ms | 102.0 FPS |

On this CPU platform, INT8 inference is *slower* than FP32 inference, because the host lacks hardware-accelerated INT8 vector instructions (VNNI). This result is presented as evidence that quantization-driven memory savings and quantization-driven latency savings are separable engineering outcomes: the former is achieved independently of target hardware, while the latter is contingent on it.

### Clinical Diagnostic Metric Retention

Comparing the FP32 baseline against the selected INT8 Conv+MatMul configuration:

| Metric | FP32 | INT8 (Conv+MatMul) |
|---|---|---|
| AUC | 0.9841 | 0.9832 |
| Sensitivity | 0.9543 | 0.9624 |
| Specificity | 0.9474 | 0.9363 |
| F1 (fixed threshold) | 0.9517 | 0.9509 |

Sensitivity rises slightly under quantization while specificity decreases marginally; the fixed-threshold F1 score is essentially unchanged. The substantial overlap between the FP32 and INT8 AUC confidence intervals indicates that the compressed model retains clinical operating characteristics comparable to the full-precision baseline.

---

## Discussion

Two findings anchor the evaluation:

1. **MobileViT-S is an effective edge-oriented backbone**, achieving superior AUC relative to the other seven compact CNN and hybrid CNN-transformer architectures evaluated under the same protocol.
2. **Mixed-precision static INT8 PTQ (Conv+MatMul QDQ) reduces model footprint by 42.6%** while preserving diagnostic accuracy (AUC 0.9832 vs. 0.9841), provided that stem convolutions and attention-scoring matrix multiplications are shielded from quantization.

Footprint compression and runtime acceleration are shown to be distinct engineering objectives: INT8 speedups require hardware-level VNNI (or equivalent) support, while serialized model-size reduction is achieved independently of target-specific INT8 kernel acceleration.

## Limitations and Future Work

- Latency evaluation was conducted on a CPU platform without VNNI acceleration; the reported INT8 latency figures do not generalize to hardware with dedicated low-precision execution units.
- Static QDQ quantization relies on representative calibration data (733 class-balanced images); calibration-set composition may influence quantization scale selection.
- Future work will investigate hardware-aware quantization on dedicated edge microcontrollers and NPUs, where INT8 acceleration is natively supported.

---

## Reproducibility

Source code, preprocessing scripts, model checkpoints, hyperparameter configurations, and evaluation scripts are available at:

**https://github.com/harvi374/LIFE.git**

Datasets used in this work:

- PatchCamelyon (PCam): https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon
- APTOS 2019 Blindness Detection: https://www.kaggle.com/c/aptos2019-blindness-detection

---

## Citation

```bibtex
@inproceedings{life2026,
  title     = {LIFE: Localized Inference-Free Enhancement for Efficient Edge Classification},
  year      = {2026}
}
```

---

## References

1. Sachin Mehta and Mohammad Rastegari. 2022. MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. In *ICLR*.
2. Bastiaan S. Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, and Max Welling. 2018. Rotation Equivariant CNNs for Digital Pathology. In *MICCAI*, 210–218.
3. Andrew Howard et al. 2019. Searching for MobileNetV3. In *ICCV*.
4. Mingxing Tan and Quoc V. Le. 2019. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In *ICML*.
5. Kaiming He et al. 2016. Deep Residual Learning for Image Recognition. In *CVPR*.
6. Sachin Mehta and Mohammad Rastegari. 2022. Separable Self-attention for Mobile Vision Transformers. *TMLR*.
7. Muhammad Maaz et al. 2022. EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications. In *ECCV*.
8. Pavan Kumar Anasosalu Vasu et al. 2023. FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization. In *ICCV*.
9. Ao Wang et al. 2024. RepViT: Revisiting Mobile CNN From ViT Perspective. In *CVPR*.
10. Markus Nagel et al. 2021. A White Paper on Neural Network Quantization. arXiv:2106.08295.
11. Zhen Dong et al. 2019. HAWQ: Hessian Aware Quantization of Neural Networks with Mixed-Precision. In *ICCV*.
12. Intel Corporation. Intel Deep Learning Boost (Intel DL Boost) — Vector Neural Network Instructions (VNNI).
13. Kaggle. 2019. APTOS 2019 Blindness Detection. Available at: https://www.kaggle.com/c/aptos2019-blindness-detection.
