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
    
    @abstractmethod
    def control_action(self, x0 : np.array, xf : np.array, param:dict ):
        pass
    
    @abstractmethod
    def control_plan(self, x0 : np.array, xf : np.array, param:dict):
        pass
    
    
class MPC(impact.MPC, Controller):
    
    impact_model : impact.Model = None
    bound_states : np.ndarray   = None
    bound_inputs : np.ndarray   = None
    OCP_mode     : bool         = False
    planned_control : np.ndarray= None
    solver_options = { "ipopt.print_level":5, 
                    "print_time":0, 
                    "expand": True,
                    "verbose": False,
                    "print_time": True,
                    "error_on_fail": False,
                    "ipopt": {"sb":"yes", "tol": 1e-8, "max_iter":1e3}}
    
    def __init__(self):
        # # setting up boundary constraints
        # self.x_current = impact_mpc.parameter('x_current',impact_model.nx)
        # self.x_final = impact_mpc.parameter('x_final',impact_model.nx)
        # impact_mpc.subject_to(impact_mpc.at_t0(impact_model.x) == self.x_current)
        # impact_mpc.subject_to(impact_mpc.at_tf(impact_model.x) == self.x_final)
        
        impact.MPC.__init__(self)
        self.solver('ipopt', self.solver_options)

        return None
  
    def add_model(self, model_name, path):
        model = super().add_model(model_name, path)
        self.impact_model = model
        self.method(impact.MultipleShooting(N=20, M = 1, intg='rk'))
        model.sim_step = self._method.discrete_system(self)        
        
        self.bound_states = np.array([[-np.inf, np.inf]] * model.nx)
        self.bound_inputs = np.array([[-np.inf, np.inf]] * model.nu)
        return model
    
    def control_action(self, x0: np.array, xf: np.array, param:dict):
        # U, X = self.control_plan(x0, xf, param)
        # u = U[0]
        
        if self.OCP_mode and self.planned_control is not None:
            U = self.planned_control
            k = param["current_index"]
            u = U[k]
        else:
            U, X = self.control_plan(x0, xf, param)
            u = U[0]
        
        return u
        
    def control_plan(self, x0: np.array, xf: np.array, param:dict):
        # set control duration
        # mpc = copy.deepcopy(self.impact_model)
        # mpc = self.impact_mpc
        model = self.impact_model
        dt = param["dt"]
        time2go = param["time2go"]
        
        self.clear_constraints()
        
        N_steps = int(np.ceil(time2go/dt))
        self.set_T(N_steps*dt)
        
        X = np.full((N_steps + 1, model.nx), np.nan)
        U = np.full((N_steps + 1, model.nu), np.nan)
        
        # set initial conditions
        self.subject_to(self.at_t0(model.x) == x0)
        # self.set_value(self.x_current, x0)
        
        # set final conditions using soft constraint if necessary
        if N_steps > model.nx:
            self.subject_to(self.at_tf(model.x) == xf)
            # self.set_value(self.x_final, xf)
        else:
            for i in range(model.nx):
                self.add_objective( 1e9 * (self.at_tf(model.x[i]) - xf[i])**2 )
     
        # set bounds
        self.subject_to(self.bound_states[:,0]<= ( model.x <= self.bound_states[:,1]) )    
        self.subject_to(self.bound_inputs[:,0]<= ( model.u <= self.bound_inputs[:,1]) )    
     
        # Solve                                
        self.method(impact.MultipleShooting(N=N_steps, M = 1, intg='rk'))

        # if 'init_guess_X' in param.keys():
        #     print('using init_guess')
        #     mpc.set_initial(model.x, param["init_guess_X"])
        #     mpc.set_initial(model.u, param["init_guess_U"])
        # else:
            # print('using linear interp')
        
        self.set_initial(model.x, x0+self.t/self.T*(xf-x0))

        try:
            # solve mpc (while preventing annoing prints)
            sol = self.solve()
            
            ts, X = sol.sample(model.x, grid='integrator')
            ts, U = sol.sample(model.u, grid='integrator')
            
            assert np.allclose(X[0], x0, rtol=1.e-3, atol=1.e1), f"Initial state mismatch: {np.round(X[0], 2).tolist()} != {np.round(x0, 2).tolist()}"
            #assert np.allclose(X[-1], xf, rtol=1.e-3, atol=1.e-3), f"Final state mismatch: {X[-1]} != {xf}"
            if not np.allclose(X[-1], xf, rtol=1.e-3, atol=1.e1):
                print(f"Final state mismatch: {np.round(X[-1], 2).tolist()} != {np.round(xf, 2).tolist()}")
        except:
            print(f"MPC failed! Continuing...")
        
        # param["init_guess_X"] = X[1:]
        # param["init_guess_U"] = U[1:]
        
        self.planned_control = U
        
        return U, X

class NNMPC(Controller):
    n_inputs  = None
    n_outputs = None
    selected_inputs = None
    use_error = False
    nn_model  = None
    scalerIn  = None
    scalerOut = None
    loss_history = []
    
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
        
        load_temp = torch.load(path, weights_only = False)
        if type(load_temp) == dict and "nn_model" in load_temp.keys():
            nnmpc.nn_model = load_temp["nn_model"]
            if "loss_history" in load_temp.keys():
                nnmpc.loss_history = load_temp["loss_history"]
        else:    
            nnmpc.nn_model = load_temp
        
        nnmpc.n_inputs = nnmpc.nn_model[0].in_features
        nnmpc.n_outputs = nnmpc.nn_model[-1].out_features
        
        if nnmpc.selected_inputs is None:
            if nnmpc.use_error:
                nnmpc.selected_inputs = [True] * (nnmpc.n_inputs-1)
            else:
                nnmpc.selected_inputs = [True] * ((nnmpc.n_inputs-1)//2)
        
        print('NNMPC loaded from:', path)
        return nnmpc

    def save_nnmpc(self, path):
        path = path + '.pt'
        torch.save({'nn_model': self.nn_model, 'loss_history': self.loss_history}, path)
        # torch.save(self.nn_model, path)
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
        X_flat = X_series.reshape(-1, X_series.shape[-1])
        R_flat = R_series.reshape(-1, R_series.shape[-1])
        T_flat = T_series.reshape(-1, 1)
        T2go_flat = np.max(T_flat)-T_flat
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
    
    def clean_dataset(self, inputs:np.ndarray, outputs:np.ndarray, out_thresh:np.number=1.5):
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
        
        self.scalerIn = MinMaxScaler()
        self.scalerIn.fit(inputs)
        self.scalerOut = MinMaxScaler()
        self.scalerOut.fit(outputs)

        return inputs, outputs, n_nans, n_outliers
        
    def train_nn(self, inputs_train, outputs_train, n_epochs=1000, lr=0.001):        
        # Normalize data
        inputs_train = self.scalerIn.transform(inputs_train)
        outputs_train = self.scalerOut.transform(outputs_train)
        
        # Convert to torch tensors
        inputs_train = torch.tensor((inputs_train), dtype=torch.float32)
        outputs_train = torch.tensor((outputs_train), dtype=torch.float32)
        # inputs_test = torch.tensor((inputs_test), dtype=torch.float32)
        # outputs_test = torch.tensor((outputs_test), dtype=torch.float32)
        
        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.nn_model.parameters(), lr=lr)
        self.nn_model.train(True)
        best_mse = np.min(self.loss_history) if len(self.loss_history) > 0 else np.inf
        best_weights = copy.deepcopy(self.nn_model.state_dict())
        # fig = go.FigureWidget()
        # fig.display()
        print(f"Training: {inputs_train.shape[0]} data points, learning rate={lr}")
        with tqdm.tqdm(range(n_epochs), unit = "epoch", mininterval = 0, disable = False) as bar:
            for epoch in bar:
                bar.set_description(f"Epoch {epoch}")
                #forward pass
                y_pred = self.nn_model(inputs_train)
                loss = loss_fn(y_pred, outputs_train)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # track progress
                mse = float(loss)
                bar.set_postfix(mse=mse)
                self.loss_history.append(mse)
                
                if mse <best_mse:
                    best_mse = mse
                    best_weights = copy.deepcopy(self.nn_model.state_dict())

                # if epoch%500 == 0 or epoch == n_epochs-1:
                #     fig = self.plot_training(fig)
                #     # fig.show()
        
        # restore best model and return best accuracy
        self.nn_model.load_state_dict(best_weights)
        print(f"Best MSE: {np.round(best_mse,4)}")
        print(f"BestRMSE: {np.round(np.sqrt(best_mse),4)}")
        return self.loss_history

    def export2casadi(self, path, save=True):
        # save as casadi function
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            name = path.split('/')[-1]
            features = casadi.SX.sym('features',self.n_inputs)

            z=[features]
            for i in range((len(self.nn_model)+1)//2):
                W=np.float64( self.nn_model[i*2].weight.detach().numpy())
                b=np.float64( self.nn_model[i*2].bias.detach().numpy())
                if i < (len(self.nn_model) - 1) // 2:
                    z.append(casadi.tanh((W @ z[i]) + b))
                else:
                    z.append((W @ z[i]) + b)

            output = z[-1]

            func = casadi.Function(name,[features],[output])
        
            # validate casadi function 
            casadi_out = np.array(func(np.ones(self.n_inputs)), dtype=np.float32).T[0]
            torch_out = self.nn_model(torch.tensor(np.ones(self.n_inputs), dtype=torch.float32)).detach().numpy()        
        assert np.allclose(casadi_out, torch_out, rtol=1e-5, atol=1e-5), f"casadi output={casadi_out}, torch output={torch_out}"
        
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
            fig_train = go.FigureWidget() # training figure
        
        if len(fig_train.data) > 0:
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
        
        if outputs.size > 10000:
            factor = outputs.size // 10000
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
    nu = None
    
    def __init__(self, nu = None):
        self.nu = nu
        return
    
    def control_action(self, x0 : np.array, xf : np.array, param:dict ):
        return np.zeros(self.nu)
    
    def control_plan(self, x0 : np.array, xf : np.array, param:dict):
        pass