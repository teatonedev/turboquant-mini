# TurboQuant Mini

Welcome to our mini demo and small packaging of this awesome method.
I strongly encourage any developer / researcher to read the paper before proceding with the implementation.

Have fun, and build more for the future!


This package provides drop in PyTorch layers for both Mean squared Error (MSE) optimized quantization and unbiased inner product estimation. It features a Straight through Estimator (STE) for quantization aware training and a bit packing utility to achieve physical memory compression during inference [arXiv:2504.19874 [cs.LG]]

# Results

Baseline FP32 KV Cache Memory : 292.97 MB
TurboQuant 4-bit KV Cache Memory: 36.62 MB

========================================
MEMORY SAVED: 256.35 MB
COMPRESSION RATIO: 8.0x
========================================

