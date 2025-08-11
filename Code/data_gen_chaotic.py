#import matplotlib.pyplot as plt
import os
os.system("ml tqdm/4.64.1-GCCcore-12.2.0")
from tqdm import tqdm
os.system("ml PyTorch/2.2.1-foss-2023b-CUDA-12.4.0")
import torch
import numpy as np

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

num_trajectories_train = 65536
num_trajectories_val = 8192
num_trajectories_test = 4096
T = 5
dt = 1e-3
noise_level_1 = 0.005
noise_level_2 = 0.01

def henon_heiles(y, args, kargs):
    q1, q2, p1, p2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    
    dq1 = p1
    dq2 = p2
    dp1 = -q1 - 2 * q1 * q2
    dp2 = -q2 - q1**2 + q2**2
    
    return torch.stack((dq1, dq2, dp1, dp2), dim=-1)


def rk2_step(dyn, y, dt, dynamics, args, kargs):
    h = dt
    i = kargs[0]
    q1, q2, p1, p2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    dy1 = dynamics(y, args, kargs)
    q1_1 = q1 + 0.5 * dy1[:, 0] * h
    q2_1 = q2 + 0.5 * dy1[:, 1] * h
    p1_1 = p1 + 0.5 * dy1[:, 2] * h
    p2_1 = p2 + 0.5 * dy1[:, 3] * h

    y1 = torch.stack((q1_1, q2_1, p1_1, p2_1), dim=-1)
    dy2 = dynamics(y1, args, kargs)

    q1_new = q1 + dy2[:, 0] * h
    q2_new = q2 + dy2[:, 1] * h
    p1_new = p1 + dy2[:, 2] * h
    p2_new = p2 + dy2[:, 3] * h
    
    return torch.stack((q1_new, q2_new, p1_new, p2_new), dim=-1)


def sv_step(dyn, y, dt, dynamics, iterations, y_init, args, kargs):
    h = dt
    q1, q2, p1, p2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    i = kargs[0]

    p1_half = p1 + 0.5 * h * dynamics(torch.stack((q1, q2, y_init[:, 2], y_init[:, 3]), dim=-1), args, kargs)[:, 2]
    p2_half = p2 + 0.5 * h * dynamics(torch.stack((q1, q2, y_init[:, 2], y_init[:, 3]), dim=-1), args, kargs)[:, 3]
    
    for _ in range(iterations):
        p1_half = p1 + 0.5 * h * dynamics(torch.stack((q1, q2, p1_half, p2_half), dim=-1), args, kargs)[:, 2]
        p2_half = p2 + 0.5 * h * dynamics(torch.stack((q1, q2, p1_half, p2_half), dim=-1), args, kargs)[:, 3]

    q1_half = q1 + 0.5 * h * dynamics(torch.stack((y_init[:, 0], y_init[:, 1], p1_half, p2_half), dim=-1), args, kargs)[:, 0]
    q2_half = q2 + 0.5 * h * dynamics(torch.stack((y_init[:, 0], y_init[:, 1], p1_half, p2_half), dim=-1), args, kargs)[:, 1]
    
    for _ in range(iterations):
        q1_half = q1 + 0.5 * h * dynamics(torch.stack((q1_half, q2_half, p1, p2), dim=-1), args, kargs)[:, 0]
        q2_half = q2 + 0.5 * h * dynamics(torch.stack((q1_half, q2_half, p1, p2), dim=-1), args, kargs)[:, 1]

    q1_new = q1 + h * dynamics(torch.stack((q1_half, q2_half, p1_half, p2_half), dim=-1), args, kargs)[:, 0]
    q2_new = q2 + h * dynamics(torch.stack((q1_half, q2_half, p1_half, p2_half), dim=-1), args, kargs)[:, 1]
    p1_new = p1_half + 0.5 * h * dynamics(torch.stack((q1_new, q2_new, p1_half, p2_half), dim=-1), args, kargs)[:, 2]
    p2_new = p2_half + 0.5 * h * dynamics(torch.stack((q1_new, q2_new, p1_half, p2_half), dim=-1), args, kargs)[:, 3]

    return torch.stack((q1_new, q2_new, p1_new, p2_new), dim=-1)

def solve_ivp_custom(type, dynamics, dyn, y0_batch, t_span, dt, args, iters):
    batch_size = y0_batch.shape[0]
    t0, t1 = t_span
    if t0 > t1:
        dt = -dt
    num_steps = int((t1 - t0) / dt) + 1

    ys_batch = [y0_batch]

    for i in range(1, num_steps):
        y = ys_batch[-1]
        y_ = rk2_step(dyn, y, dt, dynamics, args, kargs=(i,))
        y_next = sv_step(dyn, y, dt, dynamics, iters, y_, args, kargs=(i,))
        ys_batch.append(y_next)

    ys_batch = torch.stack(ys_batch, dim=1)

    print("generated trajectories for: ", type)
    return ys_batch


# Function to add noise
def add_noise(trajectories, noise_level):
    # Clone the trajectories to avoid modifying the original data
    noisy_trajectories = trajectories.clone()
    
    noise = noise_level * torch.randn_like(trajectories[:, 1:, :], device=device)
    # Add the generated noise to the trajectories (excluding the first time step)
    noisy_trajectories[:, 1:, :] += noise
    
    return noisy_trajectories

# 4D initial conditions: [q1, q2, p1, p2]
def generate_initial_conditions(num_trajectories, device):
    q_init = -0.3 + 0.6 * torch.rand(num_trajectories, 2, device=device)  # Initial positions
    p_init = -0.3 + 0.6 * torch.rand(num_trajectories, 2, device=device)  # Initial momenta
    inits = torch.cat([q_init, p_init], dim=1)
    return inits


dynamics = henon_heiles
dynamics_name = "henon_heiles"

# Generate initial conditions
inits_train = generate_initial_conditions(num_trajectories_train, device)
inits_val = generate_initial_conditions(num_trajectories_val, device)
inits_test = generate_initial_conditions(num_trajectories_test, device)

# Generate trajectories
trajectories_train = solve_ivp_custom("train", dynamics, dynamics_name, inits_train, [0, T], dt, None, 5)
trajectories_val = solve_ivp_custom("val", dynamics, dynamics_name, inits_val, [0, T], dt, None, 5)
trajectories_test = solve_ivp_custom("test", dynamics, dynamics_name, inits_test, [0, T], dt, None, 5)

# Add noise to the trajectories
noisy_trajectories_train_1 = add_noise(trajectories_train, noise_level_1)
noisy_trajectories_val_1 = add_noise(trajectories_val, noise_level_1)
noisy_trajectories_test_1 = add_noise(trajectories_test, noise_level_1)

# Add noise to the trajectories
noisy_trajectories_train_2 = add_noise(trajectories_train, noise_level_2)
noisy_trajectories_val_2 = add_noise(trajectories_val, noise_level_2)
noisy_trajectories_test_2 = add_noise(trajectories_test, noise_level_2)

# Save the generated data
savepath = f'/home/choudhar/HNNs/data_new/{dynamics_name}_{T}'

if not os.path.exists(savepath):
    os.makedirs(savepath)

torch.save(noisy_trajectories_train_1.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_train_1.pt'))
torch.save(noisy_trajectories_val_1.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_val_1.pt'))
torch.save(noisy_trajectories_test_1.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_test_1.pt'))

torch.save(noisy_trajectories_train_2.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_train_2.pt'))
torch.save(noisy_trajectories_val_2.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_val_2.pt'))
torch.save(noisy_trajectories_test_2.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_test_2.pt'))

torch.save(trajectories_train.cpu(), os.path.join(savepath, f'{dynamics_name}_train.pt'))
torch.save(trajectories_val.cpu(), os.path.join(savepath, f'{dynamics_name}_val.pt'))
torch.save(trajectories_test.cpu(), os.path.join(savepath, f'{dynamics_name}_test.pt'))

