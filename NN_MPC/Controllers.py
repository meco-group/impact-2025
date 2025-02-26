# Controllers.py
# 
# This module implements the controllers used in nnmpc_tutorial.py.
# 
# Classes:
#     Controller(ABC): Abstract base class for controllers.
#     MPC(impact.MPC, Controller): Model Predictive Controller using the impact library.
#     NNMPC(Controller): Neural Network Model Predictive Controller.
#     NullController(Controller): A controller that outputs zero control actions.
# 
# Author: Andrea Giusti
# Date:   February 2025

from abc import ABC, abstractmethod
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings 
import contextlib
import os

import impact
import casadi


class Controller(ABC):
    """
    Abstract base class for controllers.
    This class defines the interface for controllers, which must implement the following methods:
    Methods
    -------
    control_action(x0: np.array, xf: np.array, param: dict)
        Abstract method to compute the control action.
    control_plan(x0: np.array, xf: np.array, param: dict)
        Abstract method to compute the control plan.
    sim_termination()
        Method called at the end of each simulation, can be used to reset the controller.
    """
    
    @abstractmethod
    def control_action(self, x0 : np.array, xf : np.array, param:dict ):
        pass
    
    @abstractmethod
    def control_plan(self, x0 : np.array, xf : np.array, param:dict):
        pass
    
    def sim_termination(self):
        return None
    
class MPC(impact.MPC, Controller):
    """
    A Model Predictive Controller (MPC) class that integrates the impact MPC.
    Attributes:
        impact_model (impact.Model): The impact model used by the MPC.
        OCP_mode (bool): If True the control plan is computed only once.
        MPC_failed (bool): Flag to indicate if the MPC has failed during the current simulation.
        default_u (np.number): Default control action to use if MPC fails.
        soft_final (bool): Flag to indicate if the final condition constraint is soft or hard.
        xf_weights (np.ndarray): Weights for the final state error (only if soft_final=True).
        solver_options (dict): Options for the solver.
    Methods:
        __init__(): Initializes the MPC instance and sets the solver options.
        add_model(model_name, path): Adds a model from a yamal file to the MPC.
        control_action(x0, xf, param): Computes the control action for current time step.
        control_plan(x0, xf, param): Computes the control plan over a time horizon.
        sim_termination(): Called at the end of each simulation to reset the controller.
    """
    
    impact_model : impact.Model = None
    OCP_mode     : bool         = False
    MPC_failed   : bool         = False
    default_u    : np.number    = np.nan
    soft_final   : bool         = True
    xf_weights   : np.ndarray   = 1e6
    solver_options = { "ipopt.print_level":0, 
                    "print_time":0, 
                    "expand": True,
                    "verbose": False,
                    "print_time": False,
                    "error_on_fail": False,
                    "ipopt": {"sb":"yes", "tol": 1e-8, "max_iter":1e3}}
    
    def __init__(self):
        
        impact.MPC.__init__(self)
        self.solver('ipopt', self.solver_options)
        
        return None
  
    def add_model(self, model_name, path):
        model = super().add_model(model_name, path)
        self.impact_model = model
        self.method(impact.MultipleShooting(N=20, M = 1, intg='rk'))
        model.sim_step = self._method.discrete_system(self)        
        
        # set parametrics initial and final constraints
        self.x_current = self.parameter('x_current',model.nx)
        self.x_final = self.parameter('x_final', model.nx)
        self.subject_to(self.at_t0(model.x) == self.x_current)
        
        # set final condition constraint, soft or hard
        if self.soft_final:
            if np.array(self.xf_weights).size == 1:
                self.xf_weights = np.full(model.nx, self.xf_weights)
            
            e = self.variable('e', model.nx)
            for i in range(model.nx):
                self.subject_to( (self.at_tf(model.x[i]) - self.x_final[i])**2 <= e[i] ) 
        
            for i in range(model.nx):
                self.add_objective(self.xf_weights[i]*e[i]**2)  
        else:
            self.subject_to(self.at_tf(model.x) == self.x_final)
             
        return model
    
    def control_action(self, x0: np.array, xf: np.array, param:dict):

        if self.OCP_mode and 'init_guess_U' in param.keys():
            U = param["init_guess_U"]
            param["init_guess_U"] = U[1:]
            u = U[0]
        else:
            U, X = self.control_plan(x0, xf, param)
            u = U[0]
        
        return u
        
    def control_plan(self, x0: np.array, xf: np.array, param:dict):
        model = self.impact_model
        dt = param["dt"]
        time2go = param["time2go"]
        
        N_steps = int(np.round(time2go/dt))
        self.set_T(N_steps*dt)
        
        X = np.full((N_steps + 1, model.nx), np.nan)
        U = np.full((N_steps + 1, model.nu), np.nan)  
     
        # Solve                                
        self.method(impact.MultipleShooting(N=N_steps, M = 1, intg='rk'))

        # set initial conditions
        self.set_value(self.x_current, x0)
        self.set_value(self.x_final, xf)

        
        if 'init_guess_X' in param.keys():
            for i in range(model.nx):
                self.set_initial(model.x[i], param["init_guess_X"][:,i])
            for i in range(model.nu):
                self.set_initial(model.u[i], param["init_guess_U"][:,i])
        else:
            self.set_initial(model.x, x0+self.t/self.T*(xf-x0))
        
        try:
            # solve mpc (while preventing annoing prints)
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                # sol = self.solve()
                sol = self.solve_limited() # no error on max iterations
            
            # sample optimal solution
            ts, X = sol.sample(model.x, grid='integrator')
            ts, U = sol.sample(model.u, grid='integrator')
            
            if U.ndim<2: U = np.expand_dims(U,axis=1)
            if X.ndim<2: X = np.expand_dims(X,axis=1)
            
            # save optimal solution for future iterations
            param["init_guess_X"] = X[1:]
            param["init_guess_U"] = U[1:]
            
            # check returned solution is valid
            assert np.allclose(X[0], x0, rtol=1.e-3, atol=1.e1), f"Initial state mismatch: {np.round(X[0], 2).tolist()} != {np.round(x0, 2).tolist()}"
            if not np.allclose(X[-1], xf, rtol=1.e-3, atol=1.e1):
                print(f"Final state mismatch: {np.round(X[-1], 2).tolist()} != {np.round(xf, 2).tolist()}")
        
        # if mpc solving fails return previously saved control action or default value
        except:
            if not self.MPC_failed:
                print(f"MPC failed! Continuing...")
                self.MPC_failed = True

            if 'init_guess_X' in param.keys():
                X = param["init_guess_X"]
                U = param["init_guess_U"]
                param["init_guess_X"] = X[1:]
                param["init_guess_U"] = U[1:]
            else:
                U = np.full((N_steps+1, model.nu), self.default_u)
        
        return U, X
    
    def sim_termination(self):
        self.MPC_failed = False
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            self.set_initial(self.impact_model.x, 0)
            self.set_initial(self.impact_model.u, 0)
        return None

class NNMPC(Controller):
    """
    NNMPC class: Implements a Neural Network Model Predictive Controller that can be trained from MPC simulation data.
    Attributes:
        n_inputs (int): Number of inputs to the neural network.
        n_outputs (int): Number of outputs from the neural network.
        selected_inputs (list of bool): List indicating which states are used as inputs.
        use_error (bool): If True the error is used as input, else state and target are used.
        nn_model (nn.Sequential): The neural network model.
        scalerIn (MinMaxScaler): Scaler for input normalization.
        scalerOut (MinMaxScaler): Scaler for output normalization.
        loss_history (list): List to store the loss values during training.
    Methods:
        __init__(self, impact_model, selected_inputs, use_error): Initializes the NNMPC object.
        load_nnmpc(path): Loads a pre-trained NNMPC model from a .pt file.
        save_nnmpc(self, path): Saves the current NNMPC model to a .pt file.
        control_action(self, x0, xf, param): Computes the control action based on the current state, target state, and parameters.
        control_plan(self, x0, xf, param): Placeholder for control plan method.
        define_nn_structure(self, depth, width, activation): Defines the structure of the neural network.
        arrange_sim_data(self, X_series, U_series, R_series, T_series): Arranges simulation data for training.
        clean_dataset(self, inputs, outputs, out_thresh, scaler_range): Cleans the dataset by removing NaNs and outliers.
        train_nn(self, inputs_train, outputs_train, n_epochs, lr): Trains the neural network.
        export2casadi(self, path, save): Exports the neural network model to a CasADi function.
        get_structure(self): Returns the structure of the neural network.
        plot_training(self, fig_train): Plots the training loss history.
        plot_approximation(self, inputs, outputs, fig): Plots the approximation of the neural network model against the true outputs.
    """
    n_inputs        : int   = None
    n_outputs       : int   = None
    selected_inputs : bool  = None
    use_error       : bool  = False
    nn_model        : nn.Sequential = None
    scalerIn        : MinMaxScaler  = None
    scalerOut       : MinMaxScaler  = None
    loss_history    : list  = []
    
    def __init__(self, impact_model : impact.MPC = None, selected_inputs = None, use_error = False):
        if impact_model is not None:
            if selected_inputs is None:
                selected_inputs = [True] * impact_model.nx
            
            if use_error:
                self.n_inputs  = sum(selected_inputs)+1
            else:
                self.n_inputs  = sum(selected_inputs)*2+1
            self.n_outputs = impact_model.nu
        
        self.use_error = use_error
        self.selected_inputs = selected_inputs
        
        return None

    def load_nnmpc(path):
        nnmpc = NNMPC()
        path = path + '.pt'
        nnmpc.nn_model = nn.Sequential()
        
        load_temp = torch.load(path, weights_only=False)
        if type(load_temp) == dict and "nn_model" in load_temp.keys():
            nnmpc.nn_model = load_temp["nn_model"]
            if "loss_history" in load_temp.keys():
                nnmpc.loss_history = load_temp["loss_history"]
            if "scalerIn" in load_temp.keys():
                nnmpc.scalerIn = load_temp["scalerIn"]
            if "scalerOut" in load_temp.keys():
                nnmpc.scalerOut = load_temp["scalerOut"] 
            if "use_error" in load_temp.keys():
                nnmpc.use_error = load_temp["use_error"]   
            if "selected_inputs" in load_temp.keys():
                nnmpc.selected_inputs = load_temp["selected_inputs"]                  
        else:    
            nnmpc.nn_model = load_temp
        
        nnmpc.n_inputs = nnmpc.nn_model[0].in_features
        nnmpc.n_outputs = nnmpc.nn_model[-1].out_features
        
        if nnmpc.selected_inputs is None:
            nnmpc.selected_inputs = [True] * ((nnmpc.n_inputs-1)//2)
        
        print('NNMPC loaded from:', path)
        return nnmpc

    def save_nnmpc(self, path):
        path = path + '.pt'
        torch.save({'nn_model': self.nn_model, 
                    'loss_history': self.loss_history, 
                    'scalerIn':self.scalerIn, 
                    'scalerOut':self.scalerOut,
                    'use_error':self.use_error,
                    'selected_inputs':self.selected_inputs,
                    }, path)
        return print('NNMPC saved as:', path)
           
    def control_action(self, x0: np.array, xf: np.array, param:dict):
        time2go = param["time2go"]
        x0 = x0[self.selected_inputs]
        xf = xf[self.selected_inputs]
        if self.use_error:
            inputs = np.concatenate((xf-x0, np.array([time2go])))
        else:
            inputs = np.concatenate((x0, xf, np.array([time2go])))
        
        scaled_inputs = self.scalerIn.transform(inputs.reshape(1, -1))
        scaled_u = self.nn_model(torch.tensor(scaled_inputs, dtype=torch.float32)).detach().numpy()        
        u = self.scalerOut.inverse_transform(scaled_u)
        return u
    
    def control_plan(self, x0: np.array, xf: np.array, param:dict):
        return None

    def define_nn_structure(self, depth:int, width:int, activation = nn.Tanh()):
        model = nn.Sequential()
        model.add_module('input', nn.Linear(self.n_inputs, width))
        model.add_module('act', activation)
        for i in range(depth-1):
            model.add_module(f'hidden_{i}', nn.Linear(width, width))
            model.add_module(f'act_{i}', activation)
        model.add_module('output', nn.Linear(width, self.n_outputs))
        
        self.nn_model = model
        return model

    def arrange_sim_data(self, X_series, U_series, R_series, T_series):
        X_series = X_series[:, 0:-1]
        R_series = R_series[:, 0:-1]
        U_series = U_series[:, 0:-1]
        T2go     = np.max(T_series) - T_series
        T2go = T2go[:, 0:-1]

        X_flat = X_series.reshape(-1, X_series.shape[-1])
        R_flat = R_series.reshape(-1, R_series.shape[-1])
        T2go_flat = T2go.reshape(-1, 1)
        U_flat = U_series.reshape(-1, U_series.shape[-1])
        
        X_flat = X_flat[:,self.selected_inputs]
        R_flat = R_flat[:,self.selected_inputs] 
        
        assert U_flat.shape[1] == self.n_outputs
        assert X_flat.shape[0] == R_flat.shape[0] == T2go_flat.shape[0] == U_flat.shape[0]
        
        if self.use_error:
            E_flat = R_flat - X_flat
            assert E_flat.shape[1]+1 == self.n_inputs
            inputs = np.concatenate((E_flat, T2go_flat), axis=1)
        else:
            assert X_flat.shape[1]+R_flat.shape[1]+1 == self.n_inputs
            inputs = np.concatenate((X_flat, R_flat, T2go_flat), axis=1)
        
        outputs = U_flat
        
        return inputs, outputs
    
    def clean_dataset(self, inputs:np.ndarray, outputs:np.ndarray, out_thresh:np.number=1.5, scaler_range=(-1, 1)):
        io = np.concatenate([inputs,outputs], axis=1)
        
        # Remove NANs
        mask = ~np.isnan(io).any(axis=1)
        io = io[mask]
        n_nans = np.sum(~mask)
        
        # Remove outliers using IQR
        Q1 = np.percentile(io, 25, axis=0)
        Q3 = np.percentile(io, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - out_thresh * IQR
        upper_bound = Q3 + out_thresh * IQR

        mask = np.all((io >= lower_bound) & (io <= upper_bound), axis=1)
        io = io[mask]
        n_outliers = np.sum(~mask)
        
        inputs  = io[:,0:self.n_inputs]
        outputs = io[:,self.n_inputs:]
        
        # If necessary instantiate and tune the input and output scalers
        if not self.scalerIn:
            self.scalerIn = MinMaxScaler( feature_range=scaler_range)
            self.scalerIn.fit(inputs)
            self.scalerOut = MinMaxScaler(feature_range=scaler_range)
            self.scalerOut.fit(outputs)

        return inputs, outputs, n_nans, n_outliers
        
    def train_nn(self, inputs_train, outputs_train, n_epochs=1000, lr=0.001):        
        # Normalize data
        inputs_train_scaled = self.scalerIn.transform(inputs_train)
        outputs_train_scaled = self.scalerOut.transform(outputs_train)
        
        # Convert to torch tensors
        inputs_train_scaled = torch.tensor((inputs_train_scaled), dtype=torch.float32)
        outputs_train_scaled = torch.tensor((outputs_train_scaled), dtype=torch.float32)
        
        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.nn_model.parameters(), lr=lr)
        self.nn_model.train(True)
        best_mse = np.min(self.loss_history) if len(self.loss_history) > 0 else np.inf
        best_weights = copy.deepcopy(self.nn_model.state_dict())
        print(f"Training: {inputs_train_scaled.shape[0]} data points, learning rate={lr}")
        with tqdm.tqdm(range(n_epochs), unit = "epoch", mininterval = 0, disable = False) as bar:
            for epoch in bar:
                bar.set_description(f"Epoch {epoch}")
                #forward pass
                y_pred = self.nn_model(inputs_train_scaled)
                loss = loss_fn(y_pred, outputs_train_scaled)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # track progress
                mse = float(loss)
                bar.set_postfix(mse=mse)
                self.loss_history.append(mse)
                # save best model
                if mse < best_mse:
                    best_mse = mse
                    best_weights = copy.deepcopy(self.nn_model.state_dict())
        
        # restore best model and return loss history
        self.nn_model.load_state_dict(best_weights)
        print(f"Best MSE : {np.round(best_mse,6)}")
        print(f"Best RMSE: {np.round(np.sqrt(best_mse),6)}")
        
        return self.loss_history

    def export2casadi(self, path, save=True):
        # save as casadi function
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            name = path.split('/')[-1]
            features = casadi.SX.sym('features',self.n_inputs)
            scaled_inputs = features*self.scalerIn.scale_ + self.scalerIn.min_
            
            z=[scaled_inputs]
            for i in range((len(self.nn_model)+1)//2):
                W=np.float64( self.nn_model[i*2].weight.detach().numpy())
                b=np.float64( self.nn_model[i*2].bias.detach().numpy())
                if i < (len(self.nn_model) - 1) // 2:
                    z.append(casadi.tanh((W @ z[i]) + b))
                else:
                    z.append((W @ z[i]) + b)

            scaled_output = z[-1]
            output = (scaled_output - self.scalerOut.min_)/self.scalerOut.scale_ 
            
            # generate casadi function 
            if self.use_error:
                nx = self.n_inputs-1
                func = casadi.Function(name+'_block',[features[:nx],features[-1]],[output],["error","time2go"],["control_output"]) 
                casadi_out = np.array(func(np.ones(nx),1), dtype=np.float32).T[0]
            else:
                nx = (self.n_inputs-1)//2
                func = casadi.Function(name+'_block',[features[:nx],features[nx:2*nx],features[-1]],[output],["state","target_state","time2go"],["control_output"]) 
                casadi_out = np.array(func(np.ones(nx),np.ones(nx),1), dtype=np.float32).T[0]

        # validate casadi function 
        nominal_in = np.ones(self.n_inputs)
        scaled_in = self.scalerIn.transform(nominal_in.reshape(1, -1))
        scaled_out = self.nn_model(torch.tensor(scaled_in, dtype=torch.float32)).detach().numpy()        
        nominal_out = self.scalerOut.inverse_transform(scaled_out)
        assert np.allclose(casadi_out, nominal_out, rtol=1e-5, atol=1e-5), f"casadi output={casadi_out}, torch output={nominal_out}"
        
        # save casadi function
        if save:
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                func.save(path+'.casadi')
            
            print("NNMPC exported as: "+ path+ ".casadi")
        
        return func
    
    # def export2simulink(self, name):
    #     # Export NN model to Simulink
    #     mpc = impact.MPC()
    #     mpc.solver('ipopt')
    #     mpc.method(impact.MultipleShooting(N=1, M = 1, intg='rk'))
        
    #     mpc.state(1)
    #     mpc.control(1)
    #     mpc.set_T(1)

    #     with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
    #         nnmpc_casadi = self.export2casadi(name, save=False)

    #         mpc.clear_constraints()
    #         x_current = mpc.parameter('x_current',1)
    #         mpc.set_value(x_current, np.zeros(1))
    #         mpc.add_function(nnmpc_casadi)
    #         # name = NN_path.split('/')[-1]
    #         mpc.export(name)
            
    #     return None
        
    def get_structure(self):
        struct = {'n_inputs': self.n_inputs, 
                'n_outputs': self.n_outputs,
                'depth': (len(self.nn_model)-1)//2,
                'width': self.nn_model[0].out_features}
        return struct
    
    def plot_training(self, fig_train=None):
        if not fig_train:
            fig_train = go.Figure() # training figure
        
        if len(fig_train.data) == 0:
            fig_train.add_trace(go.Scatter(y=self.loss_history, mode='lines', name='loss', showlegend=False))
            fig_train.update_yaxes(type="log")
            fig_train.update_layout(title='NNMPC training', xaxis_title='Epoch', yaxis_title='Loss [MSE]')
        else:
            fig_train.data[0].y = self.loss_history
        
        return fig_train
    
    def plot_approximation(self, inputs:np.ndarray, outputs:np.ndarray, fig=None):
        
        # Remove NANs
        io = np.concatenate([inputs,outputs], axis=1)
        mask = ~np.isnan(io).any(axis=1)
        inputs = inputs[mask]
        outputs = outputs[mask]
        
        if outputs.size > 100000:
            factor = outputs.size // 100000
            inputs  = inputs[::factor]
            outputs = outputs[::factor]
        
        # Compute predictions
        scaled_inputs = self.scalerIn.transform(inputs)
        scaled_inputs = torch.tensor(scaled_inputs, dtype=torch.float32)
        scaled_y_pred = self.nn_model(scaled_inputs).detach().numpy()
        y_pred = self.scalerOut.inverse_transform(scaled_y_pred)
        
        # Fitting error
        nmse = np.linalg.norm(outputs - y_pred)**2 / np.linalg.norm(outputs - np.mean(outputs))**2 
        
        # Plot
        if not fig:
            fig = make_subplots(rows=1, cols=self.n_outputs)
        
        for i in range(self.n_outputs):
            u_range  = [np.min(outputs[:,i]), np.max(outputs[:,i])]
            fig.add_trace(
                go.Scatter(x=outputs[:,i], y=y_pred[:,i], mode='markers', name=f'Output {i}'),
                row=1, col=i+1)
            fig.add_trace(go.Scatter(x=u_range,y=u_range,line=dict(color='black')),row=1, col=i+1)
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.1, y=0.9,
                showarrow=False,
                text=f"NMSE = {nmse*100:.1f}%",
                font=dict(size=14)
            )
            fig.update_xaxes(scaleanchor="y", scaleratio=1,title_text=f'u {i+1}', row=i+1, col=1)
            fig.update_yaxes(scaleanchor="x", scaleratio=1,title_text=f'pred u {i+1}', row=i+1, col=1)
        fig.update_layout(height=500, width=500*self.n_outputs,title='NNMPC approximation', showlegend=False)
        
        return fig, nmse

class NullController(Controller):
    """
    NullController class: A controller that outputs zero control actions. 
    Attributes:
        nu (int): Number of control inputs.
    Methods:
        __init__(nu=1): Initializes the NullController with the specified number of control inputs.
        control_action(x0, xf, param): Returns a zero control action.
        control_plan(x0, xf, param): Placeholder method for control planning.
    """
    nu = 1
    
    def __init__(self, nu = 1):
        self.nu = nu
        return
    
    def control_action(self, x0 : np.array, xf : np.array, param:dict ):
        return np.zeros(self.nu)
    
    def control_plan(self, x0 : np.array, xf : np.array, param:dict):
        pass