# NN-MPC tutorial 
# By: Taranjit Singh and Andrea Giusti
# Date : 2021-06-01

# %%
# 0) Bootstrap

import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
import contextlib
import importlib
import os
from sklearn.model_selection import train_test_split

import casadi
import rockit
import impact

import rockit

rockit.GlobalOptions.set_cmake_flags(['-G','Ninja','-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang'])

import Controllers
import Utility
importlib.reload(Controllers)
importlib.reload(Utility)

# OUTLINE
# 1) Define plant model and parameters
#    - Define dynamical model
#    - Define control problem
# 2) Generate MPC solutions
#    - Define grid of initial/final conditions
#    - Solve MPC problems
#    - Store data
#    - Visualize data
# 3) Build Neural Network MPC
#    - Define NN structure
#    - Train NN
#    - Visualize training
#    - Save NNMPC
#    - Export NNMPC as casadi function and Simulink block
# 4) Validate NNMPC in closed loop
#    - Open loop validation
#    - Closed loop validation


# %% -----------------------------------------------
# 1) Define plant model and parameters

# Options
generate_mpc_data = True            # generate new MPC data, if False load saved ones
train_nn          = True            # train new NNMPC, if False load saved one
save_outputs      = False           # save outputs simulations and NN model

# parameters for mass-spring-damper
model_name = 'spring_mass_damper'       # name of the model (corresponding yaml file must be in the models folder)
# Bounds
u_max = 20                              # max control input [N]
x_range =  np.array([[-1, 1], [-4, 4]]) # state bounds
# MPC parameters
n_points = 5                        # number of grid points for each state
x0_range = [[-0.5, 0.5], [-2, 2]]   # initial state range
xf_range = x0_range                 # final state range
# NN parameters
# NN_inputs = None
# NN_use_err = False
nn_epochs = 10000                   # number of training epochs
nn_depth  = 3                       # number of activation function layers
nn_width  = 150                     # number of neurons per layer
test_size = 0.4                     # fraction of data used for validation
lr        = 0.01                    # learning rate
# Simulation parameters
sim = Utility.Simulator()
sim.Tf = 1.0                        # total simulation time [s]
sim.dt = 0.025                      # sampling time [s]

# Init
sims_path = "results/"+model_name+"_MPC_sims.npz"
NN_path = "results/"+model_name+"_NNMPC"
os.makedirs("results",exist_ok=True)

mpc = Controllers.MPC()
model = mpc.add_model(model_name, "models/"+model_name+".yaml")
print("Loaded",model_name,"model with",model.nx,"states and",model.nu, "inputs")

# Define control problem
# impact_mpc.add_objective(impact_mpc.integral(casadi.sumsqr(model.u)))
mpc.add_objective(mpc.integral(model.u[0]**2))

# %% -----------------------------------------------
# 2) Generate MPC solutions

# Define MPC controller for the given impact model
mpc.bound_states[0:len(x0_range)] =  x_range
mpc.bound_inputs[0] = np.array([-u_max, u_max])

if generate_mpc_data:
    # Define grid of initial/final conditions
    x0_combinations = Utility.get_states_grid(model, n_points, x0_range)
    xf_combinations = Utility.get_states_grid(model, n_points, xf_range)

    # Simulate MPC solutions
    X_series, U_series, R_series, T_series = sim.simulate_multi(model, mpc, x0_combinations, xf_combinations, closed_loop=False)
    
    # Save MPC simulation data
    if save_outputs:
        Utility.save_sims(sims_path, X_series, U_series, R_series, T_series)

else:
    X_series, U_series, R_series, T_series = Utility.load_sim_data(sims_path)
    print("Loaded "+ str(X_series.shape[0]) +" simulations data from "+sims_path)
    
    
# Visualize MPC data
indx = 0
fig = Utility.plot_time_evolution(X_series[indx],U_series[indx],R_series[indx],T_series[indx])
fig.update_layout(title='Example MPC trajectory')    
fig.show()
if model.nx <= 2:
    fig = Utility.plot_io_samples(X_series[:,:,0:1], X_series[:,:,1:2], names=["x1","x2"], bounds=[x_range[0],x_range[1]])
else:
    fig = Utility.plot_io_samples(X_series[:,:,0:4], X_series[:,:,0:4], names=["x","x"], bounds=[x_range[0:4],x_range[0:4]])
fig.update_layout(title_text="MPC States", height=600, width=600)
fig.show()
fig = Utility.plot_io_samples(X_series, U_series, names=["x","u"], bounds=[x_range, np.array([-u_max, u_max])])
fig.update_layout(title_text="MPC States vs Inputs", )
fig.show()

# %% -----------------------------------------------
# 3) Generate Neural Network MPC

if train_nn:
    # Define NN-MPC controller for the given impact model
    nnmpc = Controllers.NNMPC(model)

    # Define NN structure
    nnmpc.define_nn_structure(depth=nn_depth, width=nn_width)
    struct = nnmpc.get_structure()
    print("\nBuilt NNMPC with:")
    [print(f"> {key}: {value}") for key, value in struct.items()]

    # Train NN from simulation data
    print("\nTraining NNMPC from simulation data...")
    inputs, outputs = nnmpc.arrange_sim_data(X_series, U_series, R_series, T_series)
    inputs, outputs, n_nans, n_outliers = nnmpc.clean_dataset(inputs,outputs, out_thresh=5)
    print(f"{inputs.shape[0]} valid datapoints. Removed {n_nans} NaNs and {n_outliers} outliers.")

    # Split dataset into training and test sets
    if test_size == 0:
        inputs_train, outputs_train = inputs, outputs
        inputs_test, outputs_test = inputs, outputs
    else:
        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=test_size, random_state=0)

    # fig = Utility.plot_io_samples(inputs, outputs)
    # fig.show()
    
    # Train NN
    loss = nnmpc.train_nn(inputs_train, outputs_train, n_epochs=nn_epochs, lr=lr)


else:
    # Load existing NN model and data
    nnmpc = Controllers.NNMPC.load_nnmpc(path=NN_path)
    struct = nnmpc.get_structure()
    print("NNMPC has:")
    [print(f"> {key}: {value}") for key, value in struct.items()]
    inputs, outputs = nnmpc.arrange_sim_data(X_series, U_series, R_series, T_series)
    inputs, outputs, n_nans, n_outliers = nnmpc.clean_dataset(inputs,outputs, out_thresh=5)
    print(f"{inputs.shape[0]} valid datapoints. Removed {n_nans} NaNs and {n_outliers} outliers.")
    # Split dataset into training and test sets
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=test_size, random_state=0)
     
# Visualize training
fig_train = go.Figure() # training figure
fig_train.add_trace(go.Scatter(y=nnmpc.loss_history, mode='lines', name='loss'))
fig_train.update_yaxes(type="log")
fig_train.update_layout(title='NNMPC training', xaxis_title='Epoch', yaxis_title='Loss [MSE]')
fig_train.show()

# visualize NN approximation
fig, NMSE = nnmpc.plot_approximation(inputs_train, outputs_train)
fig.update_layout(title='NNMPC approximation: training set')
fig.show()
fig, NMSE = nnmpc.plot_approximation(inputs_test, outputs_test)
fig.update_layout(title='NNMPC approximation: test set')
fig.show()


if save_outputs: 
    # Save NN model
    nnmpc.save_nnmpc(NN_path)
    
    # Export NN model as casadi function
    nnmpc_casadi = nnmpc.export2casadi(NN_path)

    # Export NN model to Simulink
    try:
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            mpc.clear_constraints()
            x_current = mpc.parameter('x_current',model.nx)
            mpc.set_value(x_current, np.zeros(model.nx))
            mpc.add_function(nnmpc_casadi)
            name = NN_path.split('/')[-1]
            mpc.export(name, src_dir=NN_path+"_build_dir")
        print(f"NNMPC exported to {name}_build_dir for Simulink. ",
            "To use it:\n",
            f"1) Open Matlab and navigate to {name}_build_dir.\n",
            "2) Add casadi to Matlab path.\n",
            "3) Run build.m (can take few minutes).\n",
            f"4) Copy the {name} block from library_{name}.xls to your Simulink model.\n",
            "5) The block gets [state, target_state, time2go] and returns [control_output]."
        )
    except:
        print("Error exporting NNMPC to Simulink. Try setting generate_mpc_data = False.")

# %% -----------------------------------------------
# 4) Validate NN 

# Select initial and final condition for validation
# indx=X_series.shape[0]//2 
indx=0
x0 = X_series[indx,0,:]
# x0 = np.zeros(model.nx)
# x0 = Utility.get_random_state(model, x0_range)

xf = X_series[indx,-1,:]
# xf = np.zeros_like(x0)
# xf = Utility.get_random_state(model, x0_range)

print('\nRunning validation...')

# MPC reference trajectory
X_mpc, U_mpc, t_mpc = sim.simulate(model, mpc, x0, xf, closed_loop=False)


# open loop validation
u_nn = []
t = t_mpc
for i in range(X_series.shape[1]):
    x = X_mpc[i]
    time2go = sim.Tf - i*sim.dt
    param = {"time2go": time2go}
    u_nn.append(nnmpc.control_action(x, xf, param))

u_nn = np.array(u_nn)
fig = Utility.plot_time_evolution(np.array([]), U_mpc, xf,t_mpc, colors=["black","black","black"], name="MPC")
fig = Utility.plot_time_evolution(np.array([]), u_nn[:,:,0], xf,t, fig, colors=["blue","red","black"], name="NNMPC")    
fig.update_layout(title='Open loop validation: NNMPC vs MPC', showlegend=True)    
fig.show()

# closed loop validation
X_nn, U_nn, t_nn    = sim.simulate(model, nnmpc, x0, xf)

fig = Utility.plot_time_evolution(X_mpc[:,0:4],U_mpc, xf,t_mpc, colors=["black","black","black"], name="MPC")
fig = Utility.plot_time_evolution(X_nn[:,0:4],U_nn, xf,t_nn, fig, colors=["blue","red","black"], name="NNMPC")
fig.update_layout(title='Closed loop validation: NNMPC vs MPC', showlegend=True)    
fig.show()

# %% -----------------------------------------------
