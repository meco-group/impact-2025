"""
Furuta-pendulum
=============
"""
from impact import *
import casadi as ca
import numpy as np
import subprocess

import rockit

rockit.GlobalOptions.set_cmake_flags(['-G','Ninja','-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang'])

mpc = MPC(T=3.0)


furuta_pendulum = mpc.add_model('fu_pendulum','furuta.yaml')

print(furuta_pendulum.ee_x)
## Parameters
x_current = mpc.parameter('x_current',furuta_pendulum.nx)
x_final = mpc.parameter('x_final',furuta_pendulum.nx)


## Objectives
mpc.add_objective(mpc.sum(furuta_pendulum.Torque1**2 ))

# Initial and final state constraints
mpc.subject_to(mpc.at_t0(furuta_pendulum.x)==x_current)
mpc.subject_to(mpc.at_tf(furuta_pendulum.x)==x_final)

# Torque limits
mpc.subject_to(-40 <= (furuta_pendulum.Torque1 <= 40 ))
# Constraint to only one turn 
mpc.subject_to(-ca.pi<= (furuta_pendulum.theta1 <= ca.pi), include_first=False)


ee = ca.vertcat(furuta_pendulum.ee_x, furuta_pendulum.ee_y, furuta_pendulum.ee_z)
pivot = ca.vertcat(furuta_pendulum.pivot_x, furuta_pendulum.pivot_y, furuta_pendulum.pivot_z)

ee_nominal = ca.evalf(ca.substitute(ee,furuta_pendulum.x,[0,0,0,0]))
pivot_nominal = ca.evalf(ca.substitute(pivot,furuta_pendulum.x,[0,0,0,0]))

mpc.set_value(x_current, [-np.pi/6,0,0,0]) # Start point
mpc.set_value(x_final, [np.pi/6,0,0,0]) # End point

# Transcription
method = external_method('acados', N=50,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=10)

mpc.method(method)

# Solve
sol = mpc.solve()

mpc.export('torq_obs_aca',short_output=True)
env = dict(os.environ)
env["MPLBACKEND"] = "Agg"
assert subprocess.run(["python","hello_world_torq_obs_aca.py"],env=env,cwd="torq_obs_aca_build_dir").returncode==0

# Sample a state/control trajectory
tsa, theta1sol = sol.sample(furuta_pendulum.theta1, grid='control')
tsa, theta2sol = sol.sample(furuta_pendulum.theta2, grid='control')
tsa, dtheta1sol = sol.sample(furuta_pendulum.dtheta1, grid='control')
tsa, dtheta2sol = sol.sample(furuta_pendulum.dtheta2, grid='control')

tsb, Torque1sol = sol.sample(furuta_pendulum.Torque1, grid='control')
# tsb, Torque2sol = sol.sample(furuta_pendulum.Torque2, grid='control')

print(theta1sol)

assert abs(theta1sol[0]-(-0.52359878))<1e-5


for solver, solver_options in [("fatrop",{
        "expand": True,
        "structure_detection": "auto",
        "fatrop.tol": 1e-4,
        "debug": True,
        "common_options":{"final_options":{"cse":True}},
        "jit": False,
        "jit_options": {"flags":["-O3","-ffast-math"]}
    }),("ipopt",{}),("sqpmethod",{"qpsol": "osqp"}),("sleqp",{})]:

    mpc = MPC(T=3.0)

    furuta_pendulum = mpc.add_model('fu_pendulum','furuta.yaml')


    print(furuta_pendulum.ee_x)
    ## Parameters
    x_current = mpc.parameter('x_current',furuta_pendulum.nx)
    x_final = mpc.parameter('x_final',furuta_pendulum.nx)


    ## Objectives
    # mpc.add_objective(mpc.sum(furuta_pendulum.Torque1**2 + furuta_pendulum.Torque2**2))
    mpc.add_objective(mpc.sum(furuta_pendulum.Torque1**2 ))

    # Initial and final state constraints
    mpc.subject_to(mpc.at_t0(furuta_pendulum.x)==x_current)
    mpc.subject_to(mpc.at_tf(furuta_pendulum.x)==x_final)

    # Torque limits
    mpc.subject_to(-40 <= (furuta_pendulum.Torque1 <= 40 ))
    # Constraint to only one turn 
    mpc.subject_to(-ca.pi<= (furuta_pendulum.theta1 <= ca.pi), include_first=False)

    mpc.solver(solver, solver_options)

    ee = ca.vertcat(furuta_pendulum.ee_x, furuta_pendulum.ee_y, furuta_pendulum.ee_z)
    pivot = ca.vertcat(furuta_pendulum.pivot_x, furuta_pendulum.pivot_y, furuta_pendulum.pivot_z)

    ee_nominal = ca.evalf(ca.substitute(ee,furuta_pendulum.x,[0,0,0,0]))
    print("ee_nominal",ee_nominal)
    pivot_nominal = ca.evalf(ca.substitute(pivot,furuta_pendulum.x,[0,0,0,0]))


    mpc.set_value(x_current, [-np.pi/6,0,0,0]) # Start point
    mpc.set_value(x_final, [np.pi/6,0,0,0]) # End point

    # Transcription
    mpc.method(MultipleShooting(N=50,M=1,intg='rk'))


    # Solve
    sol = mpc.solve()


    mpc.export(f'torq_obs_{solver}',short_output=True)
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    assert subprocess.run(["python",f"hello_world_torq_obs_{solver}.py"],env=env,cwd=f"torq_obs_{solver}_build_dir").returncode==0


    # Sample a state/control trajectory
    tsa, theta1sol = sol.sample(furuta_pendulum.theta1, grid='control')
    tsa, theta2sol = sol.sample(furuta_pendulum.theta2, grid='control')
    tsa, dtheta1sol = sol.sample(furuta_pendulum.dtheta1, grid='control')
    tsa, dtheta2sol = sol.sample(furuta_pendulum.dtheta2, grid='control')

    tsb, Torque1sol = sol.sample(furuta_pendulum.Torque1, grid='control')
    # tsb, Torque2sol = sol.sample(furuta_pendulum.Torque2, grid='control')

    print(theta1sol)

    assert abs(theta1sol[0]-(-0.52359878))<1e-5



