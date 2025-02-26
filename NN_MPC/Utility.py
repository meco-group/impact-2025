"""
Utility.py

This module implements the Simulator class and various utility functions used in nnmpc_tutorial.

Author: Andrea Giusti
Date:   February 2025
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import product
import time

import impact

import Controllers

class Simulator():
    Tf : np.number
    dt : np.number
    verbose : int = 2
    noise_dyn : np.number = 0.0
    noise_u   : np.number = 0.0
    
    def simulate(self, impact_model : impact.Model, controller : Controllers.Controller, x0:np.ndarray, xf:np.ndarray, closed_loop:bool=True):
        N_steps = int(np.ceil(self.Tf/self.dt))
        t = np.linspace(0, self.Tf, N_steps+1)
        param = {"time2go":self.Tf, "dt":self.dt}

        if closed_loop:
            X = np.zeros([len(t), impact_model.nx]) * np.nan
            U = np.zeros([len(t), impact_model.nu]) * np.nan
            X[0] = x0
            
            for now in range(len(t)-1):
                tic = time.time()
                time2go = self.Tf - t[now]
                param["time2go"] = time2go
                param["current_index"] = now
                U[now] = controller.control_action(X[now], xf, param)
                U[now] = U[now] + np.random.normal(0, self.noise_u, impact_model.nu)
                temp = impact_model.sim_step(x0 = X[now], u = U[now], T = self.dt)
                X[now+1] = temp['xf'].full().flatten()
                
                if time2go>0.5:
                    X[now+1] = X[now+1] + np.random.normal(np.zeros(impact_model.nx), np.sqrt(self.dt)*self.noise_dyn, impact_model.nx)

                elapsed = time.time() - tic
                if self.verbose > 2:
                    print('  Elapsed time ', np.round(elapsed,2),"s")
                
            U[-1] = U[-2]
            
        else:
            U, X = controller.control_plan(x0, xf, param)
            X = X.reshape((N_steps+1), impact_model.nx)
            U = U.reshape((N_steps+1), impact_model.nu)
        
        controller.sim_termination()
        
        return X, U, t
        
    def simulate_multi(self, impact_model : impact.Model, controller : Controllers.Controller, x0_combinations:np.ndarray, xf_combinations:np.ndarray, closed_loop:bool=True, n_times=1):
        # Compute all combinations of initial and final conditions
        N_trajectories = x0_combinations.shape[0]*xf_combinations.shape[0]*n_times
        N_points = int(np.ceil(self.Tf/self.dt))
        cnt = 0

        X_series = np.ndarray([N_trajectories, (N_points+1), impact_model.nx]) * np.nan
        R_series = np.ndarray([N_trajectories, (N_points+1), impact_model.nx]) * np.nan
        U_series = np.ndarray([N_trajectories, (N_points+1), impact_model.nu]) * np.nan
        T_series = np.ndarray([N_trajectories, (N_points+1)]) * np.nan
        
        for i in range(x0_combinations.shape[0]):
            for j in range(xf_combinations.shape[0]):
                for k in range(n_times):
                    if Simulator.verbose > 1:
                        print('> Solving trajectory ',cnt+1,' of ', N_trajectories,': ', np.round(x0_combinations[i],2),' -> ',np.round(xf_combinations[j],2))
                    
                    x0 = x0_combinations[i]
                    xf = xf_combinations[j]
                    
                    tic = time.time()
                    X, U, t = self.simulate(impact_model, controller, x0, xf, closed_loop)
                    elapsed = time.time() - tic
                    if Simulator.verbose > 1:
                        print('  Elapsed time ', np.round(elapsed,2),"s")

                    X_series[cnt] = X
                    U_series[cnt] = U
                    R_series[cnt] = np.tile(xf, (N_points+1, 1))
                    T_series[cnt] = t
                    
                    cnt = cnt + 1
        
        # Remove NANs
        mask = ~(np.isnan(X_series).any(axis=(1,2)))
        X_series = X_series[mask]
        U_series = U_series[mask]
        R_series = R_series[mask]
        T_series = T_series[mask]
        
        n_nans = np.sum(~mask)
        
        if Simulator.verbose > 0:
            print('Simulations completed: ', n_nans, 'failed, ', X_series.shape[0], 'succeeded')
        
        return X_series, U_series, R_series, T_series
    


## Utility functions

def save_sims(path, X_series:np.ndarray, U_series:np.ndarray, R_series:np.ndarray, T_series:np.ndarray):
    assert X_series.shape[0] == U_series.shape[0] == R_series.shape[0] == T_series.shape[0]
    assert X_series.shape[1] == U_series.shape[1] == R_series.shape[1] == T_series.shape[1]
    
    np.savez(path,X_series = X_series,U_series = U_series,R_series = R_series,
        T_series=T_series)

    print( str(X_series.shape[0]) + " simulations saved in: " + path)
    return None

def load_sim_data(path):
    data = np.load(path)
    X_series = data['X_series']
    U_series = data['U_series']
    R_series = data['R_series']
    T_series = data['T_series']
    return X_series, U_series, R_series, T_series

def get_states_grid(model:impact.MPC, n_points:int, x_range:list):
    x_values = []
    for i in range(len(x_range)):
        if len(x_range[i]) > 1:
            x_values.append(np.linspace(x_range[i][0], x_range[i][1], n_points))
    
    x_values = np.array(x_values)
    x_combinations = np.array(list(product(*x_values)))
    
    if x_combinations.shape[1] < model.nx:
        x_combinations = np.concatenate(([x_combinations, np.zeros([x_combinations.shape[0], model.nx-x_combinations.shape[1]])]), axis=1)
    
    for i in range(len(x_range)):
        if len(x_range[i]) == 1:
            x_combinations[:,i] = x_range[i]
            
    return x_combinations

def get_random_state(model:impact.MPC, x_range:np.ndarray):
    x_range = np.array(x_range)
    x = np.random.uniform(x_range[:, 0], x_range[:, 1])    
    
    if x.shape[0] < model.nx:
        x = np.concatenate(([x, np.zeros([model.nx-x.shape[0]])]), axis=0)
    
    return x

def plot_io_samples(inputs:np.ndarray, outputs:np.ndarray, fig=None, colors = ["blue","red","black"], x_names=["in"], y_names=["out"], bounds=None, max_points=1e5, name="Data", mode="lines"):

    if inputs.ndim <= 2:
        inputs_flat = inputs
        inputs = np.expand_dims(inputs, axis=0)
        outputs_flat = outputs
        outputs = np.expand_dims(outputs, axis=0)
    else:
        inputs_nan = np.concatenate([inputs, np.full((inputs.shape[0], 1, inputs.shape[2]), np.nan)], axis=1)
        outputs_nan = np.concatenate([outputs, np.full((outputs.shape[0], 1, outputs.shape[2]), np.nan)], axis=1)
        inputs_flat = inputs_nan.reshape(-1, inputs_nan.shape[-1])
        outputs_flat = outputs_nan.reshape(-1, outputs_nan.shape[-1])

    if inputs_flat.size > max_points:
        factor = inputs_flat.size // max_points
        inputs_flat  = inputs_flat[::factor]
        outputs_flat = outputs_flat[::factor]
    
    n_in = inputs.shape[-1]
    n_out = outputs.shape[-1]
    
    if len(x_names) < n_in:
        x_names = [f"{x_names[0]}{i+1}" for i in range(n_in)]
    if len(y_names) < n_out:
        y_names = [f"{y_names[0]}{i+1}" for i in range(n_out)]
    
    if not fig:
        fig = make_subplots(rows=n_out, cols=n_in)
    
    for i in range(n_in):
        for j in range(n_out):
            fig.add_trace(
                go.Scatter(x=inputs_flat[:, i], y=outputs_flat[:, j], mode=mode, line=dict(color=colors[0]), name=name, showlegend=(i==0 and j==0) ),
                row=j+1, col=i+1
            )
            fig.add_trace(
                go.Scatter(x=inputs[:, 0, i], y=outputs[:, 0, j], mode='markers', marker=dict(symbol='circle-open'), line=dict(color=colors[1]), name="Initial", showlegend=(i==0 and j==0)),
                row=j+1, col=i+1
            )
            fig.add_trace(
                go.Scatter(x=inputs[:, -1, i], y=outputs[:, -1, j], mode='markers', line=dict(color=colors[1]), name="Final", showlegend=(i==0 and j==0)),
                row=j+1, col=i+1
            )
            
            # plot bounds
            if bounds is not None:
                if bounds[0].ndim < 2:
                    bounds[0] = np.expand_dims(bounds[0], axis=0)
                
                if bounds[1].ndim < 2:
                    bounds[1] = np.expand_dims(bounds[1], axis=0)
                
                if i < bounds[0].shape[0] and np.isfinite(bounds[0][i,0]):
                    fig.add_vline(x=bounds[0][i,0], line=dict(color=colors[2], dash='dash'), row=j+1, col=i+1)
                    fig.add_vline(x=bounds[0][i,1], line=dict(color=colors[2], dash='dash'), row=j+1, col=i+1)
                    fig.update_xaxes(range=[bounds[0][i,0], bounds[0][i,1]], row=j+1, col=i+1)
                    
                if j < bounds[1].shape[0] and np.isfinite(bounds[1][j,0]):
                    fig.add_hline(y=bounds[1][j,0], line=dict(color=colors[2], dash='dash'), row=j+1, col=i+1)
                    fig.add_hline(y=bounds[1][j,1], line=dict(color=colors[2], dash='dash'), row=j+1, col=i+1)
                    fig.update_yaxes(range=[bounds[1][j,0], bounds[1][j,1]], row=j+1, col=i+1)
            
            # set axis labels
            fig.update_xaxes(title_text=x_names[i], row=j+1, col=i+1)
            fig.update_yaxes(title_text=y_names[j], row=j+1, col=i+1)

    fig.update_layout(height=300*n_out, width=300*n_in, title_text="Inputs vs Outputs Scatter Plots")
    return fig

def plot_time_evolution(X, U, R, t, fig=None, colors = ["blue","red","black"], trace_name="data", x_names=["x"], u_names=["u"]):
    
    if X.ndim > 2:
        X_nan = np.concatenate([X, np.full((X.shape[0], 1, X.shape[2]), np.nan)], axis=1)
        U_nan = np.concatenate([U, np.full((U.shape[0], 1, U.shape[2]), np.nan)], axis=1)
        R_nan = np.concatenate([R, np.full((R.shape[0], 1, R.shape[2]), np.nan)], axis=1)
        t_nan = np.concatenate([t, np.full((t.shape[0], 1), np.nan)], axis=1)
        X = X_nan.reshape(-1, X_nan.shape[-1])
        U = U_nan.reshape(-1, U_nan.shape[-1])
        R = R_nan.reshape(-1, R_nan.shape[-1])
        t = t_nan.reshape(-1)
    
    nx = X.shape[-1]
    nu = U.shape[-1]
    
    if R.ndim > 1:
        Rf = R[-1]
    else:
        Rf = R
        
    if len(x_names) < nx:
        x_names = [f"x{i+1}" for i in range(nx)]
    if len(u_names) < nu:
        u_names = [f"u{i+1}" for i in range(nu)]
        
    if not fig:
        fig = make_subplots(rows=nx + nu, cols=1) 
    
    for i in range(nx):
        fig.add_trace(go.Scatter(x=t, y=X[:,i], mode='lines', line=dict(color=colors[0]), name=trace_name, legendgroup=i+1), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=[t[-1]], y=np.array(Rf[i]), line=dict(color=colors[2]), showlegend=False), row=i+1, col=1)
        fig.update_yaxes(title_text=x_names[i], row=i+1, col=1)
        fig.update_xaxes(range=[0, t[-1]], row=i+1, col=1)
    for i in range(nu):
        fig.add_trace(go.Scatter(x=t, y=U[:,i], mode='lines', line=dict(color=colors[1]), name=trace_name, legendgroup=nx+i+1), row=nx+i+1, col=1)
        fig.update_yaxes(title_text=u_names[i], row=nx+i+1, col=1)
    fig.update_xaxes(title_text=f'time [s]', row=nx+i+1, col=1)
    fig.update_layout(showlegend=False, height=100*(nx+nu+2), width=700, legend_tracegroupgap=40)    
    
    return fig

def remove_sims(condition, X_series, U_series, R_series, T_series):
    # condition = X_series[:,:,1]>6
    # condition = U_series[:,:,0]< -80
    sims2del = np.where(np.any(condition, axis=1))
    X_series = np.delete(X_series, sims2del, axis=0)
    U_series = np.delete(U_series, sims2del, axis=0)
    R_series = np.delete(R_series, sims2del, axis=0)
    T_series = np.delete(T_series, sims2del, axis=0)
    
    return X_series, U_series, R_series, T_series