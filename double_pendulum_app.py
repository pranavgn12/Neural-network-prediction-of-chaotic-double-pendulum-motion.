#!/usr/bin/env python3
"""Side-by-side double pendulum: numerical physics vs learned MLP dynamics."""

from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    import pygame
except ImportError:
    raise SystemExit("pygame is missing. Install dependencies with: python -m pip install -r requirements.txt")


# Match the training dataset and original C simulation parameters.
M1, M2 = 10.0, 2.0
L1, L2 = 150.0, 150.0
G = 1.0
DEFAULT_DT = 0.002
SCREEN_W, SCREEN_H = 1320, 760
PIVOT_Y = 145
PIXELS_PER_LENGTH = 1.0


class AccelerationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class LearnedDynamics:
    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = AccelerationMLP().to(self.device)
        state_dict = ckpt["model_state_dict"]
        # Accept both common names for the Sequential module.
        if any(key.startswith("network.") for key in state_dict):
            state_dict = {
                ("net." + key[len("network."):]) if key.startswith("network.") else key: value
                for key, value in state_dict.items()
            }
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        required = ("x_mean", "x_std", "y_mean", "y_std")
        missing = [key for key in required if key not in ckpt]
        if missing:
            raise RuntimeError(f"Checkpoint is missing normalization data: {missing}")
        self.x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32, device=self.device)
        self.x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32, device=self.device)
        self.y_mean = torch.tensor(ckpt["y_mean"], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(ckpt["y_std"], dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def acceleration(self, state):
        t1, t2, w1, w2 = state
        features = torch.tensor(
            [[math.sin(t1), math.cos(t1), math.sin(t2), math.cos(t2), w1, w2]],
            dtype=torch.float32, device=self.device,
        )
        normalized = (features - self.x_mean) / self.x_std
        prediction = self.model(normalized)
        acceleration = prediction * self.y_std + self.y_mean
        return float(acceleration[0, 0]), float(acceleration[0, 1])


def physics_acceleration(state):
    t1, t2, w1, w2 = state
    delta = t2 - t1
    sd, cd = math.sin(delta), math.cos(delta)
    A = (M1 + M2) * L1
    B = M2 * L2 * cd
    C = M2 * L2 * w2 * w2 * sd - (M1 + M2) * G * math.sin(t1)
    D = L1 * cd
    E = L2
    F = -L1 * w1 * w1 * sd - G * math.sin(t2)
    determinant = A * E - D * B
    if abs(determinant) < 1e-14:
        return 0.0, 0.0
    alpha1 = (C * E - F * B) / determinant
    alpha2 = (A * F - D * C) / determinant
    return alpha1, alpha2


def derivative(state, acceleration_fn):
    a1, a2 = acceleration_fn(state)
    return np.array([state[2], state[3], a1, a2], dtype=np.float64)


def rk4_step(state, dt, acceleration_fn):
    k1 = derivative(state, acceleration_fn)
    k2 = derivative(state + 0.5 * dt * k1, acceleration_fn)
    k3 = derivative(state + 0.5 * dt * k2, acceleration_fn)
    k4 = derivative(state + dt * k3, acceleration_fn)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def total_energy(state):
    t1, t2, w1, w2 = state
    # Angles are measured from the downward vertical, matching simulation.c.
    y1 = -L1 * math.cos(t1)
    y2 = y1 - L2 * math.cos(t2)
    vx1 = L1 * w1 * math.cos(t1)
    vy1 = L1 * w1 * math.sin(t1)
    vx2 = vx1 + L2 * w2 * math.cos(t2)
    vy2 = vy1 + L2 * w2 * math.sin(t2)
    kinetic = 0.5*M1*(vx1*vx1+vy1*vy1) + 0.5*M2*(vx2*vx2+vy2*vy2)
    potential = M1*G*y1 + M2*G*y2
    return kinetic + potential


def wrap_angle(a):
    return (a + math.pi) % (2*math.pi) - math.pi


def pendulum_points(state, origin):
    ox, oy = origin
    t1, t2 = state[0], state[1]
    p1 = (ox + L1*PIXELS_PER_LENGTH*math.sin(t1), oy + L1*PIXELS_PER_LENGTH*math.cos(t1))
    p2 = (p1[0] + L2*PIXELS_PER_LENGTH*math.sin(t2), p1[1] + L2*PIXELS_PER_LENGTH*math.cos(t2))
    return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))


def draw_pendulum(surface, state, origin, color, trail, label, font, accent):
    p1, p2 = pendulum_points(state, origin)
    pygame.draw.line(surface, (165, 174, 190), origin, p1, 5)
    pygame.draw.line(surface, (165, 174, 190), p1, p2, 5)
    pygame.draw.circle(surface, (235, 82, 82), p1, 11)
    pygame.draw.circle(surface, (70, 150, 245), p2, 9)
    if len(trail) > 1:
        pygame.draw.lines(surface, color, False, list(trail), 2)
    pygame.draw.circle(surface, (235, 240, 248), origin, 6)
    title = font.render(label, True, accent)
    surface.blit(title, (origin[0]-100, 28))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("double_pendulum_mlp.pt"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Simulation time step in seconds")
    args = parser.parse_args()
    if args.dt <= 0:
        raise SystemExit("--dt must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot see a CUDA device.")
    if not args.model.exists():
        raise SystemExit(f"Model checkpoint not found: {args.model}")

    learned = LearnedDynamics(args.model, args.device)
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Double Pendulum: Physics vs Neural Network")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("dejavusans", 20)
    small = pygame.font.SysFont("dejavusans", 15)
    big = pygame.font.SysFont("dejavusans", 27, bold=True)

    left_origin, right_origin = (330, PIVOT_Y), (990, PIVOT_Y)
    initial = np.array([math.pi/2, math.pi/2, 0.0, 0.0], dtype=np.float64)
    actual = initial.copy()
    predicted = initial.copy()
    actual_trail, predicted_trail = [], []
    paused = False
    speed = 1.0
    sim_time = 0.0
    frame_counter = 0
    fps_smooth = 0.0
    start_energy = total_energy(initial)
    last_error = 0.0
    max_error = 0.0
    last_inference_ms = 0.0
    running = True

    def reset():
        nonlocal actual, predicted, actual_trail, predicted_trail, sim_time, start_energy, last_error, max_error
        actual = initial.copy()
        predicted = initial.copy()
        actual_trail, predicted_trail = [], []
        sim_time = 0.0
        start_energy = total_energy(initial)
        last_error = max_error = 0.0

    while running:
        frame_ms = clock.tick(60)
        fps_now = clock.get_fps()
        fps_smooth = fps_now if fps_smooth == 0 else 0.9*fps_smooth + 0.1*fps_now
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = min(20.0, speed*2.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(0.125, speed/2.0)
                elif event.key == pygame.K_1:
                    initial = np.array([math.pi/2, math.pi/2, 0.0, 0.0])
                    reset()
                elif event.key == pygame.K_2:
                    initial = np.array([2.1, 1.2, 0.0, 0.0])
                    reset()
                elif event.key == pygame.K_3:
                    initial = np.array([2.4, -0.8, 0.15, -0.1])
                    reset()

        if not paused:
            substeps = max(1, math.ceil((1/60) * speed / args.dt))
            step_dt = (1/60) * speed / substeps
            for _ in range(substeps):
                actual = rk4_step(actual, step_dt, physics_acceleration)
                t0 = time.perf_counter()
                predicted = rk4_step(predicted, step_dt, learned.acceleration)
                last_inference_ms = (time.perf_counter()-t0)*1000.0
                sim_time += step_dt
                frame_counter += 1
            actual = np.nan_to_num(actual, nan=0.0, posinf=0.0, neginf=0.0)
            predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
            pa = np.array([wrap_angle(actual[0]), wrap_angle(actual[1])])
            pp = np.array([wrap_angle(predicted[0]), wrap_angle(predicted[1])])
            angle_diff = np.array([wrap_angle(actual[0]-predicted[0]), wrap_angle(actual[1]-predicted[1])])
            last_error = float(np.sqrt(np.mean(angle_diff**2)))
            max_error = max(max_error, last_error)
            p1a, p2a = pendulum_points(actual, left_origin)
            p1p, p2p = pendulum_points(predicted, right_origin)
            actual_trail.append(p2a)
            predicted_trail.append(p2p)
            actual_trail = actual_trail[-900:]
            predicted_trail = predicted_trail[-900:]

        screen.fill((15, 19, 29))
        pygame.draw.rect(screen, (23, 29, 43), (25, 18, 625, 500), border_radius=15)
        pygame.draw.rect(screen, (23, 29, 43), (670, 18, 625, 500), border_radius=15)
        draw_pendulum(screen, actual, left_origin, (247, 166, 80), actual_trail, "ACTUAL PHYSICS", font, (247, 184, 105))
        draw_pendulum(screen, predicted, right_origin, (110, 205, 255), predicted_trail, "NEURAL NETWORK", font, (110, 205, 255))

        # Bottom information panel
        pygame.draw.rect(screen, (23, 29, 43), (25, 535, 1270, 200), border_radius=15)
        status = "PAUSED" if paused else "RUNNING"
        screen.blit(big.render(f"{status}   |   Simulated time: {sim_time:7.2f} s   |   Speed: {speed:g}x", True, (240, 243, 250)), (50, 550))
        e_actual = total_energy(actual)
        e_pred = total_energy(predicted)
        energy_drift = abs(e_actual-start_energy) / max(abs(start_energy), 1e-9) * 100
        metrics = [
            f"Angular state RMSE: {last_error:.5f} rad",
            f"Maximum observed angular RMSE: {max_error:.5f} rad",
            f"Actual energy drift: {energy_drift:.3f}%",
            f"Actual energy: {e_actual:.3f}   |   NN energy: {e_pred:.3f}",
            f"NN inference (last RK4 step): {last_inference_ms:.3f} ms   |   Display FPS: {fps_smooth:.1f}",
        ]
        for i, line in enumerate(metrics):
            screen.blit(font.render(line, True, (205, 214, 229)), (55, 595+i*27))
        controls = "SPACE pause/resume   R reset   1/2/3 initial conditions   +/- speed   Q/Esc quit"
        screen.blit(small.render(controls, True, (145, 158, 180)), (55, 710))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
