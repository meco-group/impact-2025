import casadi.*

addpath(char(py.rockit.matlab_path))
addpath(char(py.impact.matlab_path))

rockit.GlobalOptions.set_cmake_flags({'-G','Ninja','-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang'})

mpc = impact.MPC('T',3.0);


furuta_pendulum = mpc.add_model('fu_pendulum','furuta.yaml');

furuta_pendulum.ee_x
% Parameters
x_current = mpc.parameter('x_current',furuta_pendulum.nx);
x_final = mpc.parameter('x_final',furuta_pendulum.nx);


% Objectives
mpc.add_objective(mpc.sum(furuta_pendulum.Torque1^2 ));

% Initial and final state constraints
mpc.subject_to(mpc.at_t0(furuta_pendulum.x)==x_current);
mpc.subject_to(mpc.at_tf(furuta_pendulum.x)==x_final);

% Torque limits
mpc.subject_to(-40 <= furuta_pendulum.Torque1 <= 40 );
% Constraint to only one turn 
mpc.subject_to(-pi<= furuta_pendulum.theta1 <= pi, 'include_first','false');


ee = [furuta_pendulum.ee_x; furuta_pendulum.ee_y; furuta_pendulum.ee_z];
pivot = [furuta_pendulum.pivot_x; furuta_pendulum.pivot_y; furuta_pendulum.pivot_z];

ee_nominal = evalf(substitute(ee,furuta_pendulum.x,[0,0,0,0]));
pivot_nominal = evalf(substitute(pivot,furuta_pendulum.x,[0,0,0,0]));

mpc.set_value(x_current, [-pi/6,0,0,0]); % Start point
mpc.set_value(x_final, [pi/6,0,0,0]); % End point

% Transcription
method = rockit.external_method('acados', 'N',50,'qp_solver','PARTIAL_CONDENSING_HPIPM','nlp_solver_max_iter',200,'hessian_approx','EXACT','regularize_method','CONVEXIFY','integrator_type','ERK','nlp_solver_type','SQP','qp_solver_cond_N',10);

mpc.method(method);

% Solve
sol = mpc.solve();

mpc.export('torq_obs_aca','short_output',true);

% Sample a state/control trajectory
[tsa, theta1sol] = sol.sample(furuta_pendulum.theta1, 'grid','control');
[tsa, theta2sol] = sol.sample(furuta_pendulum.theta2', 'grid','control');
[tsa, dtheta1sol] = sol.sample(furuta_pendulum.dtheta1, 'grid','control');
[tsa, dtheta2sol] = sol.sample(furuta_pendulum.dtheta2, 'grid','control');

[tsb, Torque1sol] = sol.sample(furuta_pendulum.Torque1, 'grid','control');
% tsb, Torque2sol = sol.sample(furuta_pendulum.Torque2, grid='control')

theta1sol

assert(abs(theta1sol(1)-(-0.52359878))<1e-5)



fatrop_options = struct();
fatrop_options.expand = true;
fatrop_options.structure_detection = 'auto';
fatrop_options.fatrop.tol = 1e-4;
fatrop_options.debug = true;
fatrop_options.common_options.final_options.cse = true;

solver_config = {{'fatrop',fatrop_options},{'ipopt',struct},{'sqpmethod',struct('qpsol', 'osqp')},{'sleqp',struct}};

for i=1:length(solver_config)
    solver = solver_config{i}{1};
    solver_options = solver_config{i}{2};

    
    mpc = impact.MPC('T',3.0);
    
    
    furuta_pendulum = mpc.add_model('fu_pendulum','furuta.yaml');
    
    furuta_pendulum.ee_x
    % Parameters
    x_current = mpc.parameter('x_current',furuta_pendulum.nx);
    x_final = mpc.parameter('x_final',furuta_pendulum.nx);
    
    
    % Objectives
    mpc.add_objective(mpc.sum(furuta_pendulum.Torque1^2 ));
    
    % Initial and final state constraints
    mpc.subject_to(mpc.at_t0(furuta_pendulum.x)==x_current);
    mpc.subject_to(mpc.at_tf(furuta_pendulum.x)==x_final);
    
    % Torque limits
    mpc.subject_to(-40 <= furuta_pendulum.Torque1 <= 40 );
    % Constraint to only one turn 
    mpc.subject_to(-pi<= furuta_pendulum.theta1 <= pi, 'include_first','false');
    
    
    ee = [furuta_pendulum.ee_x; furuta_pendulum.ee_y; furuta_pendulum.ee_z];
    pivot = [furuta_pendulum.pivot_x; furuta_pendulum.pivot_y; furuta_pendulum.pivot_z];
    
    ee_nominal = evalf(substitute(ee,furuta_pendulum.x,[0,0,0,0]));
    pivot_nominal = evalf(substitute(pivot,furuta_pendulum.x,[0,0,0,0]));
    
    mpc.set_value(x_current, [-pi/6,0,0,0]); % Start point
    mpc.set_value(x_final, [pi/6,0,0,0]); % End point
    
    % Transcription

    mpc.method(rockit.MultipleShooting('N',50,'M',1,'intg','rk'));

    mpc.solver(solver, solver_options);
    
    % Solve
    sol = mpc.solve();
    
    mpc.export('torq_obs_aca','short_output',true);
    
    % Sample a state/control trajectory
    [tsa, theta1sol] = sol.sample(furuta_pendulum.theta1, 'grid','control');
    [tsa, theta2sol] = sol.sample(furuta_pendulum.theta2', 'grid','control');
    [tsa, dtheta1sol] = sol.sample(furuta_pendulum.dtheta1, 'grid','control');
    [tsa, dtheta2sol] = sol.sample(furuta_pendulum.dtheta2, 'grid','control');
    
    [tsb, Torque1sol] = sol.sample(furuta_pendulum.Torque1, 'grid','control');
    % tsb, Torque2sol = sol.sample(furuta_pendulum.Torque2, grid='control')
    
    theta1sol
    
    assert(abs(theta1sol(1)-(-0.52359878))<1e-5)


end


