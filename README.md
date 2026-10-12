# apex_winograd

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SymPy](https://img.shields.io/badge/SymPy-Exact-green.svg)](https://www.sympy.org/)

> **Neuro-Symbolic search and exact symbolic verification of Winograd-type convolution algorithms for arbitrary geometries.**

`apex_winograd` is an automated engine that discovers exact, minimal-FLOPs bilinear Winograd-type convolution algorithms for arbitrary spatial geometries, including non-continuous dilated (`dilation > 1`) and strided (`stride > 1`) kernels where classical Chinese Remainder Theorem (CRT) polynomial algebra fails.

---

## Key Features

- **Neuro-Symbolic AI Engine:** Combines high-precision continuous search (PyTorch Autograd) with exact analytical symbolic proofs (SymPy matrix expansion).
- **Dynamic Scale Annealing:** Implements an S-curve cosine scale warmup mechanism (W_C_eff = W_C * scale(t)) derived from minimal denominator LCM bounds to prevent L1-regularization from stifling matrix sparsity.
- **Hard Example Mining Buffer:** Accelerates optimization convergence by dynamically caching high-residual training samples for focused mini-batch gradient updates.
- **Arbitrary N-D Layouts:** Supports 1D, 2D, 3D, non-square, anisotropic, dilated, and strided convolutions out of the box.
- **Zero-Residual Verification:** Mathematically proves exact equality with naive convolution in SymPy, generating exact rational matrices over Q.

---

## Examples

Unlike classical Winograd formulations restricted to stride=1 and dilation=1, `apex_winograd` finds exact rational transforms for non-continuous layouts:

| Geometry | Parameters | Naive FLOPs | Winograd FLOPs | Savings (%) | W_D Sparsity | SymPy Verification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Anisotropic Dilated 2D** | in=(4,6), k=(3,3), d=(1,2) | 36 | **24** | **-33.3%** | **91.0% zeros** | **Proved (Residual = 0)** |
| **Strided Downsampling 2D** | in=(5,5), k=(3,3), s=(2,2) | 36 | **25** | **-30.5%** | **90.9% zeros** | **Proved (Residual = 0)** |

---

## How It Works

The engine factors N-D convolutions into bilinear tensor decompositions:

    Y = W_C * ((W_D * D) * (W_F * F))

    Input Data (D)   ---> [ W_D ] ---\
                                      +---> (*) Element-wise ---> [ W_C ] ---> Output (Y)
    Kernel Weights (F) -> [ W_F ] ---/

1. **Geometry Calculation:** Computes exact tensor layouts and effective spatial input dimensions for N-D strided/dilated kernels (`calculate_winograd_geometry`).
2. **Constrained Gradient Optimization:** Jointly trains W_D, W_F, and W_C with Adam + Cosine Annealing while harvesting hard examples into a memory buffer (`HardExampleBuffer`).
3. **Rational Snapping:** Normalizes matrix weights and snaps floating-point parameters to canonical low-denominator fractions (`snap_tensor_to_fractions`).
4. **SymPy Symbolic Proof:** Solves exact rational linear systems via SymPy and proves zero polynomial residual: Delta_sym = expand(Y_pred - Y_target) == 0.

---

## Installation

    git clone https://github.com/your-username/apex_winograd.git
    cd apex_winograd
    pip install -r requirements.txt

---

## Quick Start

To quickly test the engine on a standard 1D Winograd F(2, 3) configuration:

    from apex_winograd import run_stable_experiment

    # Fast 1D Winograd F(2, 3) search: out_shape=2, kernel_shape=3
    run_stable_experiment(
        out_shape=2,
        kernel_shape=3,
        stride=1,
        dilation=1,
        steps=30000,
        batch_size=1024,
        lr=0.002,
        l1_lambda=0.00015
    )

---

## Call for Contributors

We welcome contributions from the community:

* **Method Improvements:** Enhancements to search convergence, scale annealing logic, dynamic pruning, or loss formulation.
* **Algorithm Vault:** Submitting newly discovered and verified exact Winograd matrices for unusual or non-standard spatial geometries (dilated, strided, 2D/3D).

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
