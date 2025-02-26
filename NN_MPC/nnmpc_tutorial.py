# nnmpc_tutorial.py
#
# This module demonstrates the process of defining, solving, and validating a Neural Network Model Predictive Controller (NNMPC) for a given dynamic system.
# This leverages the impact software to model dynamical systems and implement MPC controllers. 
#
# Author: Andrea Giusti
# Date:   February 2025

# OUTLINE
# 1) Define model and problem
#    - Define model
#    - Define control problem
#    - Define grid of initial/final conditions
# 2) Generate MPC solutions
#    - Solve MPC problems
#    - Store data
#    - Visualize data
# 3) Build Neural Network MPC
#    - Define NN structure
#    - Train NN
#    - Visualize training
#    - Save NNMPC
# 4) Validate NNMPC 
#    - Open loop validation
#    - Closed loop validation
# 5) Export NNMPC 
#    - Export NNMPC as casadi function
#    - Export NNMPC as simulink block

# %% -----------------------------------------------
# 0) Bootstrap

# Imports
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import contextlib
import importlib
import os
from sklearn.model_selection import train_test_split

import casadi
import rockit
import impact
rockit.GlobalOptions.set_cmake_flags(['-G','Ninja','-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang'])

import Controllers
import Utility
importlib.reload(Controllers)
importlib.reload(Utility)

# Options
load_mpc_data     = False    # Load saved simulations data
generate_mpc_data = True     # Run MPC simulations
load_nn           = False    # Load saved NNMPC
train_nn          = True     # Train NNMPC using simulations data
save_outputs      = False    # Save new data (simulations and NNMPC) to results folder


# %% -----------------------------------------------
# 1) Define impact MPC

# parameters for furuta pendulum
model_name = 'furuta'  # name of the model (corresponding yaml file must be in the models folder)
x_names = ["theta1","theta2","omega1","omega2"]
u_names = ["tau1","tau2"]
# Bounds
u_max = 90                                 # Maximum input: shoulder torque [Nm]
x_range = np.array([[-2,2],[-0.5,1.5],[-10,10]])*np.pi # State bounds: angles [rad] and speeds [rad/s]
# MPC parameters
n_points  = 1                              # Number of points per non singular range of initial/target conditions
n_times   = 4                               # Number of simulations per trajectory
MPC_noise = 1                               # Actuation noise
xf_weights = np.array([1000,1000, 10,10, 1,1])*1e6 # Weights for final state mismatch
x0_range = [[0],[0]]                        # Range of initial conditions: angles [rad]
# x0_range = [[-np.pi, np.pi],[0]]            # Range of initial conditions: angles [rad]
xf_range = [[0],[np.pi]]                    # Range of target conditions: angles [rad]
# NNMPC parameters
NN_inputs = [True]*4 + [False]*2            # Select model states used as NN inputs
NN_use_err = True                           # Use error wrt target condition instead of state
nn_epochs = 10 * 1000                       # Number of NN training epochs
nn_depth  = 3                               # Number of activation layers in the NN
nn_width  = 200                             # Number of neurons per layer
test_size = 0.05                            # Fraction of data used for validation
test_split = "last"                         # Select validation data ("last" or "rand")
lr        = 0.001                           # NN learning rate
# Simulation parameters
sim = Utility.Simulator()                   # Instantiate simulator
sim.Tf = 1.5                                # Total simulation time [s]
sim.dt = 0.03                               # Sampling time [s]
sim.verbose = 2                             # Simulator verbosity

# Init
sims_path = "results/"+model_name+"_MPC_sims.npz"
NN_path = "results/"+model_name+"_NNMPC"
os.makedirs("results",exist_ok=True)
nnmpc = None

# Instantiate MPC controller
mpc = Controllers.MPC()
mpc.xf_weights = xf_weights
model = mpc.add_model(model_name, "models/"+model_name+".yaml")
print("Loaded",model_name,"model with",model.nx,"states and",model.nu, "inputs")

# Define control problem
mpc.add_objective(mpc.integral(model.u[0]**2))
# Add state and input constraints
mpc.subject_to(x_range[:,0] <= (model.x[:len(x_range)] <= x_range[:,1]))
mpc.subject_to(-u_max <= (model.u <= u_max))

# %% -----------------------------------------------
# 2) Generate MPC data

assert (load_mpc_data or generate_mpc_data), "At least one between load_mpc_data and generate_mpc_data must be True"

# Allocate empty ndarrays for storing simulation data
X_series_l = np.empty((0, int(sim.Tf/sim.dt + 1), model.nx))
U_series_l = np.empty((0, int(sim.Tf/sim.dt + 1), model.nu))
R_series_l = np.empty((0, int(sim.Tf/sim.dt + 1), model.nx))
T_series_l = np.empty((0, int(sim.Tf/sim.dt + 1)))
X_series_s = np.empty((0, int(sim.Tf/sim.dt + 1), model.nx))
U_series_s = np.empty((0, int(sim.Tf/sim.dt + 1), model.nu))
R_series_s = np.empty((0, int(sim.Tf/sim.dt + 1), model.nx))
T_series_s = np.empty((0, int(sim.Tf/sim.dt + 1)))

if load_mpc_data:
    # Load saved MPC simulation data
    X_series_l, U_series_l, R_series_l, T_series_l = Utility.load_sim_data(sims_path)
    print("Loaded "+ str(X_series_l.shape[0]) +" simulations data from "+sims_path)
 
if generate_mpc_data:
    # Define grid of initial/final conditions
    x0_combinations = Utility.get_states_grid(model, n_points, x0_range)
    xf_combinations = Utility.get_states_grid(model, n_points, xf_range)

    # Simulate MPC solutions
    sim.noise_u = MPC_noise
    X_series_s, U_series_s, R_series_s, T_series_s = sim.simulate_multi(model, mpc, x0_combinations, xf_combinations, closed_loop=True, n_times=n_times)
    sim.noise_u = 0.0
    
# Concatenate data
X_series = np.concatenate((X_series_l, X_series_s), axis=0)
U_series = np.concatenate((U_series_l, U_series_s), axis=0)
R_series = np.concatenate((R_series_l, R_series_s), axis=0)
T_series = np.concatenate((T_series_l, T_series_s), axis=0)

# Save MPC simulation data
if save_outputs and generate_mpc_data:
    Utility.save_sims(sims_path, X_series, U_series, R_series, T_series) 
    
# Visualize MPC data
fig = Utility.plot_time_evolution(X_series[:,:,0:4],U_series, R_series[:,:,0:4],T_series, colors=["blue","red","black"],x_names=x_names,u_names=u_names)
fig.update_layout(title='MPC trajectory')    
fig.show()
xu = np.concatenate([X_series[:,:,0:4],U_series], axis=2)
xu_range = np.concatenate([x_range[0:4],[[-np.nan, np.nan]],[[-u_max, u_max]]], axis=0)
fig = Utility.plot_io_samples(xu, xu, x_names=x_names+u_names, y_names=x_names+u_names, bounds=[xu_range,xu_range])
fig.update_layout(title_text="MPC States", height=800, width=900)
fig.show()

# %% -----------------------------------------------
# 3) Generate Neural Network MPC
assert (load_nn or train_nn), "At least one between load_nn and train_nn must be True"

if load_nn:
    # Load existing NN model
    nnmpc = Controllers.NNMPC.load_nnmpc(path=NN_path)
    
    # print NN structure
    struct = nnmpc.get_structure()
    print("Loaded NNMPC with:")
    [print(f"> {key}: {value}") for key, value in struct.items()]
    
if not nnmpc:
    # Define new NN-MPC controller for the given impact model
    nnmpc = Controllers.NNMPC(model, selected_inputs=NN_inputs, use_error=NN_use_err)

    # Define NN structure
    nnmpc.define_nn_structure(depth=nn_depth, width=nn_width)

    # print NN structure
    struct = nnmpc.get_structure()
    print("\nBuilt NNMPC with:")
    [print(f"> {key}: {value}") for key, value in struct.items()]

# Arrange and clean simulation data
print("\nTraining NNMPC from simulation data...")
inputs, outputs = nnmpc.arrange_sim_data(X_series, U_series, R_series, T_series)
inputs, outputs, n_nans, n_outliers = nnmpc.clean_dataset(inputs,outputs, out_thresh=5)
print(f"{inputs.shape[0]} valid datapoints. Removed {n_nans} NaNs and {n_outliers} outliers.")

# Split dataset into training and test sets
if test_size == 0:
    inputs_train, outputs_train = inputs, outputs
    inputs_test, outputs_test = None, None
elif test_split == "last":
    split_index = int(inputs.shape[0] * (1 - test_size))
    inputs_train, inputs_test = inputs[:split_index], inputs[split_index:]
    outputs_train, outputs_test = outputs[:split_index], outputs[split_index:]
elif test_split == "rand":
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=test_size, random_state=0)

# Train NN from simulation data
if train_nn:
    # fig = Utility.plot_io_samples(inputs, outputs, mode='markers')
    # fig.show()
    
    # Train NN
    loss = nnmpc.train_nn(inputs_train, outputs_train, n_epochs=nn_epochs, lr=lr)

    
# Visualize training
fig_train = nnmpc.plot_training()
fig_train.show()

# visualize NN approximation
fig, NMSE = nnmpc.plot_approximation(inputs_train, outputs_train)
fig.update_layout(title=f'NNMPC approximation: training set {inputs_train.shape[0]} datapoints')
fig.show()
if inputs_test is not None:
    fig, NMSE = nnmpc.plot_approximation(inputs_test, outputs_test)
    fig.update_layout(title=f'NNMPC approximation: test set {inputs_test.shape[0]} datapoints')
    fig.show()


if save_outputs: 
    # Save NN model
    nnmpc.save_nnmpc(NN_path)
    

# %% -----------------------------------------------
# 4) Validate NNMPC 

# indx = np.random.randint(0, X_series.shape[0])
# x0 = X_series[indx, 0, :]
x0 = np.zeros(model.nx)
# x0[0] = np.pi
# x0 = Utility.get_random_state(model, [[-np.pi, np.pi]])

xf = np.zeros_like(x0)
xf[1] = np.pi

print('\nRunning validation:', np.round(x0,2),' -> ',np.round(xf,2))

X_mpc, U_mpc, t_mpc = sim.simulate(model, mpc, x0, xf, closed_loop=False)

# Open loop validation
U_nn_op = U_mpc * np.nan
t = t_mpc
for i in range(X_mpc.shape[0]):
    x = X_mpc[i]
    time2go = sim.Tf - i*sim.dt
    param = {"time2go": time2go}
    U_nn_op[i] = nnmpc.control_action(x, xf, param)

err = np.abs(U_nn_op - U_mpc)[:,0]
fig = make_subplots(rows=2, cols=1) 
fig = Utility.plot_time_evolution(np.array([]), U_mpc, xf,t_mpc, fig, colors=["black","black","black"], trace_name="MPC", u_names=u_names)
fig = Utility.plot_time_evolution(np.array([]), U_nn_op, xf,t, fig, colors=["blue","red","black"], trace_name="NNMPC", u_names=u_names)    
fig.add_trace(go.Scatter(x=t, y=err, mode='lines', line=dict(color="red"), name="err", legendgroup=2), row=2, col=1)
fig.update_layout(title='Open loop validation: NNMPC vs OCP', showlegend=True)    
fig.show()

# Closed loop validation
X_nn, U_nn, t_nn    = sim.simulate(model, nnmpc, x0, xf)

fig = Utility.plot_time_evolution(X_series[:,:,0:4],U_series, R_series[:,:,0:4],T_series, colors=["blue","blue","blue"], trace_name="MPC dataset")
fig = Utility.plot_time_evolution(X_mpc[:,0:4],U_mpc, xf,t_mpc, fig, colors=["black","black","black"], trace_name="MPC")
fig = Utility.plot_time_evolution(X_nn[:,0:4],U_nn, xf,t_nn, fig, colors=["red","red","black"], trace_name="NNMPC",x_names=x_names,u_names=u_names)
fig.update_layout(title='Closed loop validation: NNMPC vs MPC', showlegend=True)    
fig.show()

fig = Utility.plot_io_samples(X_series[:,:,0:4], X_series[:,:,0:4], colors = ["blue","blue","black"], name="MPC dataset")
fig = Utility.plot_io_samples(X_mpc[:,0:4], X_mpc[:,0:4], colors=["black","black","black"], name="MPC trajectory", fig=fig)
fig = Utility.plot_io_samples(X_nn[:,0:4], X_nn[:,0:4], x_names=x_names, y_names=x_names, bounds=[x_range[0:4],x_range[0:4]], colors=["red","red","black"], name="NNMPC trajectory", fig=fig)
fig.update_layout(title_text="Closed loop validation: NNMPC vs MPC", height=800, width=900)
fig.show()

# %% -----------------------------------------------
# 5) Export NNMPC 

if save_outputs:
    # Export NN model as casadi function
    nnmpc_casadi = nnmpc.export2casadi(NN_path)

    # Export MPC and NNMPC to Simulink
    try:
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            mpc.set_value(mpc.x_current, np.zeros(model.nx))
            mpc.set_value(mpc.x_final, np.zeros(model.nx))
            mpc.add_function(nnmpc_casadi)
            name = NN_path.split('/')[-1]
            mpc.export(name, src_dir=NN_path+"_build_dir")
        print(f"NNMPC exported to {name}_build_dir for Simulink. ",
            "To use it:\n",
            f"1) Open Matlab and navigate to {name}_build_dir.\n",
            "2) Add casadi to Matlab path with: addpath(MY_CASADI_DIR).\n",
            "3) Run build.m (can take few minutes).\n",
            f"4) Copy the {name} block from library_{name}.xls to your Simulink model.\n",
        )
    except:
        print("Error exporting NNMPC to Simulink. Try setting generate_mpc_data = False.")
