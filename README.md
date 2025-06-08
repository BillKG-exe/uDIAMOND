# uDIAMOND Agent Training Notebook

This notebook sets up and runs the training loop for both DIAMOND or uDIAMOND
model as part of our project. The goal is to evaluate how noise conditioning
affects policy learning, visual fidelity, and computational efficiency in
imagined rollouts.

---

## What This Notebook Does

1. **Clones the uDIAMOND repository**:
   - Repository: [https://github.com/BillKG-exe/uDIAMOND](https://github.com/BillKG-exe/uDIAMOND)

2. **Installs required dependencies**, including:
   - `torch`, `gym`, `gymnasium[atari]`, `wandb`, `hydra-core`, `torcheval`

3. **Log in to Weights & Biases (wandb)** for experiment tracking. The training
   information such as losses of the diffrent models involved as well as
   console logs are stored on wandb.

4. **Runs DIAMOND training**:
   - Environment: `BreakoutNoFrameskip-v4`
   - Noise conditioning parameter: `noise_conditioning=true` to setup DIAMOND
     architecture with noise conditioning or false to activate uDIAMOND
     architecture which removes noise conditioning. 

---

## How to Run

To run the uDIAMOND version (without noise conditioning), run the command as shown below:

```bash
!python3 src/main.py \
  "env.train.id=BreakoutNoFrameskip-v4" \
  "agent.denoiser.noise_conditioning=false"
```

## Computation

Training both DIAMOND and uDIAMOND models requires substantial compute
resources. Due to the use of diffusion models and the iterative denoising
process involved in generating frame predictions, training is computationally
intensive, therefore **heavily relies on GPUs**. 

Running this notebook on a CPU-only environment is not recommended and will
result in extremely slow training. We used Kaggle to run our notebook using
their GPU T4 x 2. The training resulted in about 1 hour for 140. Training it on
the orginal number of epochs as DIAMOND(1000 epochs) will take couple days.

## Note
uDIAMOND is a variation of the DIAMOND model from [Diffusion for World Modeling:  
Visual Details Matter in Atari](https://github.com/eloialonso/diamond). The 
uDiamond model implements DIAMOND without noise conditioning.
