% Open Matlab from the conda shell
% Make sure to add casadi-matlab to the path
import casadi.*

addpath(char(py.rockit.matlab_path))
addpath(char(py.impact.matlab_path))

rockit.GlobalOptions.set_cmake_flags({'-G','Ninja','-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang'})
rockit.GlobalOptions.set_cmake_build_type('Debug')

mpc = impact.MPC('T',0.5);

% Add furuta model
furuta = mpc.add_model('fu_pendulum','furuta.yaml');

% Parameters
x_current = mpc.parameter('x_current',furuta.nx);
x_final = mpc.parameter('x_final',furuta.nx);

% Objectives
mpc.add_objective(mpc.sum(furuta.Torque1^2 ));

% Initial and final state constraints
mpc.subject_to(mpc.at_t0(furuta.x)==x_current);
mpc.subject_to(mpc.at_tf(furuta.x)==x_final);

% Path constraints
mpc.subject_to(-40 <= furuta.dtheta1 <= 40 , 'include_first',false, 'include_last', false)
mpc.subject_to(-40 <= furuta.dtheta2 <= 40 , 'include_first',false, 'include_last', false)

mpc.subject_to(-pi <= furuta.theta1 <= pi, 'include_first',false)

ee = [furuta.ee_x; furuta.ee_y; furuta.ee_z];
pivot = [furuta.pivot_x; furuta.pivot_y; furuta.pivot_z];

options = struct();
options.expand = true;
options.structure_detection = 'auto';
options.fatrop.tol = 1e-4;
options.fatrop.print_level = 0;
options.debug = true;
options.print_time = false;
options.common_options.final_options.cse = true;

mpc.solver('fatrop', options);

mpc.set_value(x_current, [-pi/3,0,0,0]);
mpc.set_value(x_final, [pi/3,0,0,0]);

% Transcription choice
mpc.method(rockit.MultipleShooting('N',25,'intg','heun'));

% Solve
sol = mpc.solve_limited();



[ts, theta2sol] = sol.sample(furuta.theta2, 'grid','control');
[ts_fine, theta2sol_fine] = sol.sample(furuta.theta2, 'grid','integrator','refine',10);

theta2sol


