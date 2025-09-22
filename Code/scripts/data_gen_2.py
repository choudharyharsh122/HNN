#import matplotlib.pyplot as plt
import os
#os.system("ml tqdm/4.64.1-GCCcore-12.2.0")
from tqdm import tqdm
#os.system("ml PyTorch/2.2.1-foss-2023b-CUDA-12.4.0")
import torch
import numpy as np
import argparse

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

num_trajectories_train = 16384
num_trajectories_val = 8192
num_trajectories_test = 1024
T = 10
dt = 1e-2
noise_level = 0.008

"""
Dynamics of spring-mass system

Parameters:
-----------
y : torch.Tensor, shape (batch_size, 2d)  
    state of the system, where:  
    - The first `d` elements represent the position \( q \).  
    - The last `d` elements represent the momentum \( p \).
args : tuple  
    Contains additional arguments required for computation.  
kargs : tuple  
    Contains additional keyword arguments.  

Returns:
--------
torch.Tensor, shape (batch_size, 2d)  
    The computed time derivative \([dq/dt, dp/dt]\), where:
    - \( dq/dt = \partial H / \partial p \)
    - \( dp/dt = -\partial H / \partial q \)
"""
def mass_spring(y, args, kargs):
    q = y[:, 0]
    p = y[:, 1]
    dq_dt = p
    dp_dt = -q
    return torch.stack((dq_dt, dp_dt), dim=-1)

"""
Dynamics of double-well system

Parameters:
-----------
y : torch.Tensor, shape (batch_size, 2d)  
    state of the system, where:  
    - The first `d` elements represent the position \( q \).  
    - The last `d` elements represent the momentum \( p \).
args : tuple  
    Contains additional arguments required for computation.  
kargs : tuple  
    Contains additional keyword arguments.  

Returns:
--------
torch.Tensor, shape (batch_size, 2d)  
    The computed time derivative \([dq/dt, dp/dt]\), where:
    - \( dq/dt = \partial H / \partial p \)
    - \( dp/dt = -\partial H / \partial q \)
"""
def double_well(y, args, kargs):
    q = y[:, 0]
    p = y[:, 1]
    dq_dt = p
    dp_dt = q - q**3
    return torch.stack((dq_dt, dp_dt), dim=-1)

"""
Dynamics of coupled-oscillator system

Parameters:
-----------
y : torch.Tensor, shape (batch_size, 2d)  
    state of the system, where:  
    - The first `d` elements represent the position \( q \).  
    - The last `d` elements represent the momentum \( p \).
args : tuple  
    Contains additional arguments required for computation.  
kargs : tuple  
    Contains additional keyword arguments.  

Returns:
--------
torch.Tensor, shape (batch_size, 2d)  
    The computed time derivative \([dq/dt, dp/dt]\), where:
    - \( dq/dt = \partial H / \partial p \)
    - \( dp/dt = -\partial H / \partial q \)
"""
def coupled_ho(y, args, kargs):
    alpha = args["alpha"]
    q, p = y[:, 0], y[:, 1]
    dqdt = p + alpha * q
    dpdt = -q - alpha * p
    return torch.stack((dqdt, dpdt), dim=-1)

"""
Dynamics of henon-heiles system

Parameters:
-----------
y : torch.Tensor, shape (batch_size, 2d)  
    state of the system, where:  
    - The first `d` elements represent the position \( q \).  
    - The last `d` elements represent the momentum \( p \).
args : tuple  
    Contains additional arguments required for computation.  
kargs : tuple  
    Contains additional keyword arguments.  

Returns:
--------
torch.Tensor, shape (batch_size, 2d)  
    The computed time derivative \([dq/dt, dp/dt]\), where:
    - \( dq/dt = \partial H / \partial p \)
    - \( dp/dt = -\partial H / \partial q \)
"""
def henon_heiles(y, args, kargs):
    q1, q2, p1, p2 = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    
    dq1 = p1
    dq2 = p2
    dp1 = -q1 - 2 * q1 * q2
    dp2 = -q2 - q1**2 + q2**2
    
    return torch.stack((dq1, dq2, dp1, dp2), dim=-1)


"""
RK2 solver for initial guess of solution

Parameters:
-----------
dyn      : dynamics function
y        : torch.Tensor, shape (batch_size, 2d)  
           state of the system, where:  
           - The first `d` elements represent the position \( q \).  
           - The last `d` elements represent the momentum \( p \).
dt       : timestep
dynamics : dynamics name (a string)
args     : tuple  
           Contains additional arguments required for computation.  
kargs    : tuple  
           Contains additional keyword arguments.  

Returns:
--------
torch.Tensor, shape (batch_size, 2d)  
    The computed future state \([q, p]\)
"""
def rk2_step(dyn, y, dt, dynamics, args, kargs):
    h = dt
    i = kargs[0]
    br_i = y.shape[1]//2
    q, p = y[:, 0:br_i], y[:, br_i:2*br_i]

    y = torch.cat((q, p), dim=-1)  # Shape: (batch_size, 2)

    #print(q.shape)

    dy1 = dynamics(y, args, kargs)
    q1 = q + 0.5 * dy1[:, 0:br_i] * h
    p1 = p + 0.5 * dy1[:, br_i:2*br_i] * h

    y1 = torch.cat((q1, p1), dim=-1)  # Shape: (batch_size, 2)
    dy2 = dynamics(y1, args, kargs)

    q_new = q + dy2[:, 0:br_i] * h
    p_new = p + dy2[:, br_i:2*br_i] * h
    return q_new, p_new

"""
Implicit Midpoint solver for trajectory

Parameters:
-----------
dyn      : dynamics function
y        : torch.Tensor, shape (batch_size, 2d)  
           state of the system, where:  
           - The first `d` elements represent the position \( q \).  
           - The last `d` elements represent the momentum \( p \).
dt       : timestep
dynamics : dynamics name (a string)
iters    : number of fixed point iterations
args     : tuple  
           Contains additional arguments required for computation.  
kargs    : tuple  
           Contains additional keyword arguments.  

Returns:
--------
torch.Tensor, shape (batch_size, 2d)  
    The computed future state \([q, p]\)
"""
def im_step(dyn, y, dt, dynamics, iters, y_init, args, kargs):
    h = dt
    br_i = y.shape[1]//2
    q, p = y[:, 0:br_i], y[:, br_i:2*br_i]
    y_init_concat = torch.cat((y_init[:, 0:br_i], y_init[:, br_i:2*br_i]), dim=-1)
    f_init = dynamics(y_init_concat, args, kargs)
    q_new = q + 0.5 * h * f_init[:, 0:br_i]
    p_new = p + 0.5 * h * f_init[:, br_i:2*br_i]

    for _ in range(iters):
        mid_q = 0.5 * (q + q_new)
        mid_p = 0.5 * (p + p_new)
        f_mid = dynamics(torch.cat((mid_q, mid_p), dim=-1), args, kargs)
        q_new = q + h * f_mid[:, 0:br_i]
        p_new = p + h * f_mid[:, br_i:2*br_i]

    return torch.cat((q_new, p_new), dim=-1)

"""
Utility function for forward solve

Parameters:
-----------
type     : data/trajectory category (train/val/test)
dt       : timestep
dynamics : dynamics function
dyn      : dynamics name (string)
y_init   : torch.Tensor, shape (batch_size, 2d)  
           initial state of the system, where:  
           - The first `d` elements represent the position \( q \).  
           - The last `d` elements represent the momentum \( p \).
t_span   : simulation range [t_init, t_final]
dt       : timestep
args     : tuple  
           Contains additional arguments required for computation.   
iters    : number of fixed point iterations 

Returns:
--------
torch.Tensor, shape (batch_size, len, 2d)  
    The computed trajectory 
"""
def solve_ivp_custom(type, dynamics, dyn, y_init, t_span, dt, args, iters):
    t0, t1 = t_span
    if t0 > t1:
        dt = -dt
    #t_vals = np.arange(t0, t1, dt)
    t_vals = np.linspace(t0, t1, int(t1/dt)+1)
    batch_size = y_init.shape[0]

    _y = torch.zeros((batch_size, t_vals.shape[0], y_init.shape[1]), dtype=torch.float32, requires_grad=False, device=device)
    y = y_init.clone()

    _y[:, 0, :] = y

    for i, t in enumerate(t_vals[1:], start=1):
        q_, p_ = rk2_step(dyn, y.clone(), dt, dynamics, args, kargs=(i,))
        y = im_step(dyn, y.clone(), dt, dynamics, iters, torch.cat((q_, p_), dim=-1), args, kargs=(i,))
        _y[:, i, :] = y
    
    print("generated trajectories for: ", type)
    return _y

"""
Utility function for forward solve

Parameters:
-----------
trajectories     : trajectory (tensor with shape (batch_size, len, dims))
noise_level      : noise variance
Returns:
--------
torch.Tensor, shape (batch_size, len, 2d)  
    The computed trajectory 
"""
def add_noise(trajectories, noise_level):
    # Clone the trajectories to avoid modifying the original data
    noisy_trajectories = trajectories.clone()
    
    noise = noise_level * torch.randn_like(trajectories[:, 1:, :], device=device)
    # Add the generated noise to the trajectories (excluding the first time step)
    noisy_trajectories[:, 1:, :] += noise

    noisy_trajectories[:, 1:, :] += noise
    
    return noisy_trajectories


# Parse command line arguments
parser = argparse.ArgumentParser(description="Enter the simulation parameters")
parser.add_argument("--dynamics_name", type=str, required=True, choices=["mass_spring", "double_well", "coupled_ho", "henon_heiles"], help="The name of the dynamics function.")
parser.add_argument("--q_range", type=float, nargs=2, required=True, help="range for q0 (pair of space seperated floating point numbers)")
parser.add_argument("--p_range", type=float, nargs=2, required=True, help="range for p0 (pair of space seperated floating point numbers)")
parser.add_argument("--sim_len", type=int, required=True, help="The total simulation time length (an int)")
parser.add_argument("--time_step", type=float, required=True, help="The time step length (< sim_len)")
parser.add_argument("--noise_level", type=float, required=False, default=0.005, help="The noise variance")


args = parser.parse_args()

# Unpack the ranges directly
q_min, q_max = args.q_range
p_min, p_max = args.p_range

dynamics_name = args.dynamics_name
q_range = args.q_range
p_range = args.p_range
sim_len = args.sim_len
time_step = args.time_step
noise_level = args.noise_level
#q_min, q_max = args.q_range
#p_min, p_max = args.p_range

#print(type(q_min))

if args.dynamics_name == "mass_spring":
    dynamics = mass_spring
elif args.dynamics_name == "double_well":
    dynamics = double_well
elif args.dynamics_name == "coupled_ho":
    dynamics = coupled_ho


# Ensure that q_min, q_max, p_min, and p_max are tensors
q_min = torch.tensor(q_min, device=device)
q_max = torch.tensor(q_max, device=device)
p_min = torch.tensor(p_min, device=device)
p_max = torch.tensor(p_max, device=device)


# initial conditions: [q1, q2, p1, p2]
def generate_initial_conditions(num_trajectories, q_max, q_min, p_max, p_min, device, d):

    inits = torch.cat(((torch.rand(num_trajectories, d, device=device) - 0.5)*(q_max - q_min) + ((q_max + q_min)/2),  # q0 range q_min to q_max
                         (torch.rand(num_trajectories, d, device=device) - 0.5)*(p_max - p_min) + ((p_max + p_min)/2)) , # p0 range p_min to p_max
                        dim=-1)
    return inits



arguments = {"alpha": 0.5}

### We define a custom range for henon-heiles as it is a chaotic system and there are small regions with bounded trajectories ###
### The trajectories in any arbitrary range are not necessarily bounded and diverge quickly ###
if args.dynamics_name=="henon_heiles":
    dynamics = henon_heiles
    # Initialize the tensors and move them to the GPU if available
    # Generate initial conditions
    inits_train = generate_initial_conditions(num_trajectories_train, 0.3, -0.3, 0.3, -0.3, device, 2)
    inits_val = generate_initial_conditions(num_trajectories_val, 0.3, -0.3, 0.3, -0.3, device, 2)
    inits_test = generate_initial_conditions(num_trajectories_test, 0.3, -0.3, 0.3, -0.3, device, 2)

### For stable systems we use the provided range from the command line ###
else:
    # Generate initial conditions
    inits_train = generate_initial_conditions(num_trajectories_train, q_max, q_min, p_max, p_min, device, 1)
    inits_val = generate_initial_conditions(num_trajectories_val, q_max, q_min, p_max, p_min, device, 1)
    inits_test = generate_initial_conditions(num_trajectories_test, q_max, q_min, p_max, p_min, device, 1)


trajectories_train = solve_ivp_custom("train", dynamics, dynamics_name, inits_train, [0, sim_len], time_step, arguments, 5)
trajectories_val = solve_ivp_custom("val", dynamics, dynamics_name, inits_val, [0, sim_len], time_step, arguments, 5)
trajectories_test = solve_ivp_custom("test", dynamics, dynamics_name, inits_test, [0, sim_len], time_step, arguments, 5)



noisy_trajectories_train = add_noise(trajectories_train, noise_level)
noisy_trajectories_val = add_noise(trajectories_val, noise_level)
noisy_trajectories_test = add_noise(trajectories_test, noise_level)

# Define the save path
savepath = os.path.join('data', f"{dynamics_name}_{sim_len}")

os.makedirs(savepath, exist_ok=True)


torch.save(noisy_trajectories_train.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_train.pt'))
torch.save(noisy_trajectories_val.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_val.pt'))
torch.save(noisy_trajectories_test.cpu(), os.path.join(savepath, f'noisy_{dynamics_name}_test.pt'))

torch.save(trajectories_train.cpu(), os.path.join(savepath, f'{dynamics_name}_train.pt'))
torch.save(trajectories_val.cpu(), os.path.join(savepath, f'{dynamics_name}_val.pt'))
torch.save(trajectories_test.cpu(), os.path.join(savepath, f'{dynamics_name}_test.pt'))

print(f"Data successfully saved in: {savepath}")
