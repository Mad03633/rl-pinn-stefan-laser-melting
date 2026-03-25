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
