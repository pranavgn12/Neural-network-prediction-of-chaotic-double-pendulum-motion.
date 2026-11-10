# Neural-network-prediction-of-chaotic-double-pendulum-motion.

A PyTorch project for learning the governing dynamics of a chaotic double pendulum from generated simulation data and comparing the learned model against the exact physical solution in a real-time visualization.

## Overview

The double pendulum is a classic nonlinear dynamical system with extreme sensitivity to initial conditions. This repository trains a multilayer perceptron (MLP) to predict the angular accelerations `alpha1` and `alpha2` from the current pendulum state, then compares the neural-network trajectory against the numerical physics simulation side by side.

The project includes:

- a generator for synthetic double-pendulum data
- a PyTorch training pipeline for the MLP
- a saved trained checkpoint and metrics
- interactive visual comparison of physics vs model predictions
- training diagnostics and plots for validation and residual analysis

## Project structure

- `double_pendulum.csv` — generated dataset containing trajectories with state and acceleration columns
- `train_mlp.py` — training script for the MLP model and evaluation metrics
- `double_pendulum_app.py` — Pygame-based visual comparison between physics and neural network
- `double_pendulum_mlp.pt` — trained model checkpoint
- `double_pendulum_mlp.history.csv` — per-epoch training history in CSV format
- `double_pendulum_mlp.history.json` — per-epoch training history in JSON format
- `double_pendulum_mlp.metrics.json` — final training/evaluation metrics
- `double_pendulum_mlp_training_plots/` — loss, RMSE, residual, and prediction plots
- `generate_dataset` and `generate_dataset.c` — C-based dataset generation utilities
- `simulation.c` — reference numerical dynamics implementation

## Scientific goal

The model takes the pendulum state vector as input:

- `sin(theta1)`
- `cos(theta1)`
- `sin(theta2)`
- `cos(theta2)`
- `omega1`
- `omega2`

and predicts:

- `alpha1` (angular acceleration of the first link)
- `alpha2` (angular acceleration of the second link)

This mirrors the physical equations behind the pendulum dynamics while exploiting a learned mapping from state to acceleration.

## Dependencies

Install the required Python packages:

```bash
python -m pip install numpy torch matplotlib pygame
```

If you want to regenerate the dataset using the C toolchain, you will also need a C compiler such as `gcc`.

## Training a model

From the repository root:

```bash
python train_mlp.py double_pendulum.csv --epochs 60 --batch-size 512
```

Optional arguments include:

- `--lr` for learning rate
- `--patience` for early stopping
- `--output` to choose a checkpoint name
- `--plot-every` to control snapshot generation

This script will:

1. load and normalize the dataset
2. split it into train/validation/test partitions
3. train the MLP using AdamW
4. save the best model checkpoint
5. generate training diagnostics and scatter/residual plots

## Running the interactive comparison

Start the visualization with:

```bash
python double_pendulum_app.py --model double_pendulum_mlp.pt --device cpu
```

Controls:

- `Space` — pause/resume
- `R` — reset
- `1`, `2`, `3` — switch initial conditions
- `+` / `-` — increase or decrease simulation speed
- `Q` or `Esc` — quit

The simulation shows the exact physics trajectory on the left and the neural-network-predicted trajectory on the right, along with the current angular-state error and energy metrics.

## Rebuilding the dataset

To regenerate the data with the C-based simulator:

```bash
gcc generate_dataset.c -O3 -o generate_dataset
./generate_dataset
```

The generated data is stored in `double_pendulum.csv` and matches the format expected by the training pipeline.

## Notes

This project is designed as a learning and visualization exercise in dynamics modeling. Because the double pendulum is chaotic, even small numeric differences or model inaccuracies can grow rapidly over time, which makes this an interesting benchmark for neural surrogate models in physical systems.

## License

This project does not appear to include a formal license file in the repository. If you plan to reuse or redistribute the code, confirm whether a license applies before publishing or commercial use.

## Example output

The repository already includes example artifacts such as:

- a trained checkpoint (`double_pendulum_mlp.pt`)
- training history (`double_pendulum_mlp.history.csv`)
- metrics (`double_pendulum_mlp.metrics.json`)
- diagnostic plots in `double_pendulum_mlp_training_plots/`

These artifacts allow you to inspect the model behavior without retraining from scratch.
