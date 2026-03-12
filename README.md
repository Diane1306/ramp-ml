# ramp-ml
## Overview

This repository develops a **PyTorch neural-network detector** for identifying **temperature ramp events** in canopy-flow time series. The goal is to improve ramp detection reliability and event timing compared with commonly used signal-processing approaches, especially when event amplitudes and background variability change over short periods.

### Install (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

ramp-ml -h
```

## Motivation

Temperature ramps in canopy flows are often detected using methods such as:

- **Wavelet transforms**
- **Variable-Interval Time-Averaging (VITA)**
- **Absolute temperature-drop thresholds**

While these approaches are widely used, each has practical limitations for robust, visually consistent ramp identification:

### Wavelet-based detection
Wavelet methods are well suited for intermittent turbulent signals, but detection performance can be highly sensitive to:
- the **wavelet family** (time–frequency localization properties),
- the **scale selection**, and
- the resulting trade-off between **temporal precision** and **frequency resolution**.

In practice, wavelet choices can strongly affect the **exact timing** of ramp rises and resets (drops) in scalar time series.

### VITA and threshold-based drop detection
VITA- and drop-threshold methods depend on user-defined thresholds. However:
- thresholds can vary substantially even within a **single 30-minute window**, and
- a single fixed threshold often fails across changing stability, turbulence intensity, and ramp magnitudes.

As a result, these methods can struggle to deliver a single set of parameters that consistently matches **visual inspection** across diverse conditions.

## Approach

To address these challenges, this repo uses a **neural-network (1D CNN/TCN-style) event detector** implemented in **PyTorch**. The model is trained using ramp/reset times identified by established methods (e.g., VITA) and refined via event-selection logic, with the goal of learning **ramp morphology** rather than relying on fixed thresholds or wavelet parameter choices.

In short: **learn the ramp pattern**, rather than hand-tuning detection parameters for each time period.