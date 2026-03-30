# RL-Guided PINNs for Adaptive Melt Front Control in Laser Melting

This repository contains the implementation for the Master's thesis:

**“Reinforcement-Learning-Guided Physics-Informed Neural Networks for Adaptive Control of the Melt Front in Fast Laser Melting”**

**Author:** Madiyar Bolatov
**Supervisor:** Dr. PhD Samat Kassabek 
**Program:** Applied Artificial Intelligence (MSc), Astana IT University  
**Years:** 2025–2027

---

## Overview

Laser-based manufacturing processes require precise control of the **melt pool and phase transition boundary**. The evolution of the melt front is governed by a **two-phase Stefan problem**, which involves:

- Heat diffusion in solid and liquid phases  
- A moving boundary (melting front)  
- Latent heat effects  

This project combines:

- **Physics-Informed Neural Networks (PINNs)** — to solve PDEs without labeled datasets  
- **Reinforcement Learning (RL)** — to adaptively control laser parameters  

The goal is to create a **data-efficient and physically consistent framework** for real-time melt front control.

---

## Key Contributions

- Implementation of a **two-phase 1D Stefan PINN**
- Incorporation of **laser heat flux boundary conditions**
- Use of **analytical supervision** to stabilize PINN training
- Development of an **RL-based control strategy**
- Validation against **analytical melt depth models**

---

## Mathematical Model

We consider a **1D transient heat conduction problem** with a moving interface \( X(t) \).

### Governing equations

Liquid phase:

$$
\frac{\partial T_l}{\partial t} = \alpha_l \frac{\partial^2 T_l}{\partial z^2}
$$

Solid phase:

$$
\frac{\partial T_s}{\partial t} = \alpha_s \frac{\partial^2 T_s}{\partial z^2}
$$

---

### Boundary conditions

Laser heat flux at the surface:

$$
-k_l \frac{\partial T}{\partial z}\bigg|_{z=0} = A \cdot I
$$

Far-field condition:

$$
T(z_{\max}, t) = T_0
$$

---

### Interface conditions (Stefan problem)

Temperature continuity:

$$
T_l(X(t), t) = T_s(X(t), t) = T_m

---

$$

Stefan condition:

$$
\rho L \frac{dX}{dt} =
k_s \frac{\partial T_s}{\partial z}
-
k_l \frac{\partial T_l}{\partial z}
$$

---

## PINN Formulation

The neural network approximates:

- \(T_l(z,t)\) — liquid temperature  
- \(T_s(z,t)\) — solid temperature  
- \(X(t)\) — melt front position  

### Loss function

Loss =
- PDE_liquid +
- PDE_solid + 
- Initial_condition + 
- Boundary_conditions + 
- Interface_temperature + 
- Stefan_condition + 

Sampling is performed separately in:

- liquid region: `0 ≤ z ≤ X(t)`
- solid region: `X(t) ≤ z ≤ z_max`

---

## PINN + Analytical Supervision

Pure PINNs often suffer from instability in moving boundary problems.

To address this, the model is weakly guided using an analytical solution:

- melt depth \(X(t)\)
- surface temperature \(T_s(t)\)

This hybrid approach improves convergence and physical consistency.

---

## Reinforcement Learning

IN PROCESS...

---

## Installation

### Recommended Python version

Python 3.10

### Setup


git clone https://github.com/Mad03633/rl-pinn-stefan-laser-melting.git

cd rl-pinn-stefan-laser-melting

python -m venv env
env\Scripts\activate # Windows

pip install --upgrade pip
pip install -r requirements.txt

## Example Results

### Ti-6AI-4V

Analytical solution by (Ngwenya and Kahlen (2012)).

![](https://github.com/Mad03633/rl-pinn-stefan-laser-melting/blob/main/Stefan_1D_2P_laser/Ti-6AI-4V/figures/analytical_solution_Ngwenya.png)

PINN (I = 5kW, 50kW, 500kW, 5MW)

<p align="center">
  <img src="https://github.com/Mad03633/rl-pinn-stefan-laser-melting/blob/main/Stefan_1D_2P_laser/Ti-6AI-4V/figures/PINN%2Bsupervision/analytical_vs_PINN_I_5kW.png" width="45%" />
  <img src="https://github.com/Mad03633/rl-pinn-stefan-laser-melting/blob/main/Stefan_1D_2P_laser/Ti-6AI-4V/figures/PINN%2Bsupervision/analytical_vs_PINN_I_50kW.png" width="45%" />
</p>

<p align="center">
  <img src="https://github.com/Mad03633/rl-pinn-stefan-laser-melting/blob/main/Stefan_1D_2P_laser/Ti-6AI-4V/figures/PINN%2Bsupervision/temp_field-pinn_pred_I_5kW.png" width="45%" />
  <img src="https://github.com/Mad03633/rl-pinn-stefan-laser-melting/blob/main/Stefan_1D_2P_laser/Ti-6AI-4V/figures/PINN%2Bsupervision/temp_field-pinn_pred_I_50kW.png" width="45%" />
</p>