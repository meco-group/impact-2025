"""
Furuta-pendulum
=============
"""
from impact import *
import casadi as ca
import numpy as np


import rockit

rockit.GlobalOptions.set_cmake_flags(['-G','Ninja','-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang'])

mpc = MPC(T=3.0)


# forward_dynamics = ca.Function.load("furuta_pendulum/casadi_functions/forward_dynamics.casadi")
# furuta_pendulum2 = mpc.add_model('fu_pendulum','furuta2.yaml')
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

## Solver
# options = {"ipopt": {"print_level": 3}}
# options["expand"] = True
# options["print_time"] = False
# mpc.solver('ipopt',options)

ee = ca.vertcat(furuta_pendulum.ee_x, furuta_pendulum.ee_y, furuta_pendulum.ee_z)
pivot = ca.vertcat(furuta_pendulum.pivot_x, furuta_pendulum.pivot_y, furuta_pendulum.pivot_z)

ee_nominal = ca.evalf(ca.substitute(ee,furuta_pendulum.x,[0,0,0,0]))
print("ee_nominal",ee_nominal)
pivot_nominal = ca.evalf(ca.substitute(pivot,furuta_pendulum.x,[0,0,0,0]))


if False: # Obstacle avoidance?    
    obstacle = [ee_nominal+ca.vertcat(-0.1,0,0.05),ee_nominal+ca.vertcat(0.1,0,0.05),ee_nominal+ca.vertcat(-0.1,0,-0.3),ee_nominal+ca.vertcat(0.1,0,-0.3)]

    print("obstacle", obstacle)


    # Plane spanned by a unit vector ...
    a = mpc.control(3)
    #mpc.set_initial(a,ca.vertcat(0,0,1))
    mpc.subject_to(-1<=(a<=1), include_first=False, include_last=False)

    # ... and an offset
    b = mpc.control()

    # ee and pivot are on one side ...
    mpc.subject_to(ca.dot(a,ee)<= b, include_first=False, include_last=False)
    mpc.subject_to(ca.dot(a,pivot)<= b, include_first=False, include_last=False)
    # ... obstacle on the other
    for vertex in obstacle:
        mpc.subject_to(ca.dot(a,vertex)>= b, include_first=True, include_last=False)

mpc.set_value(x_current, [-np.pi/6,0,0,0]) # Start point
mpc.set_value(x_final, [np.pi/6,0,0,0]) # End point

# Transcription
method = external_method('acados', N=50,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=10)

mpc.method(method)

# Solve
sol = mpc.solve()

mpc.export('torq_obs_aca',short_output=True)



# Sample a state/control trajectory
tsa, theta1sol = sol.sample(furuta_pendulum.theta1, grid='control')
tsa, theta2sol = sol.sample(furuta_pendulum.theta2, grid='control')
tsa, dtheta1sol = sol.sample(furuta_pendulum.dtheta1, grid='control')
tsa, dtheta2sol = sol.sample(furuta_pendulum.dtheta2, grid='control')

tsb, Torque1sol = sol.sample(furuta_pendulum.Torque1, grid='control')
# tsb, Torque2sol = sol.sample(furuta_pendulum.Torque2, grid='control')


print(theta1sol)

# Plots
from pylab import *


figure(figsize=(10, 4))
subplot(2, 1, 1)
plot(tsa, theta1sol, '.-', label='Rotation axe angle')
grid(True)
title('Rotation axe')
ylabel("Angle [Rad]", fontsize=14)
subplot(2, 1, 2)
plot(tsa, dtheta1sol, '.-', color='orange', label='Rotation axe AngVel')
grid(True)
xlabel("Times [s]", fontsize=14)
ylabel("Angle Vel [Rad/s]", fontsize=14)

figure(figsize=(10, 4))
subplot(2, 1, 1)
plot(tsa, theta2sol, '.-', label='Rotation axe angle')
grid(True)
title('Pendulum')
ylabel("Angle [Rad]", fontsize=14)
subplot(2, 1, 2)
plot(tsa, dtheta2sol, '.-', color='orange', label='Rotation axe AngVel')
grid(True)
xlabel("Times [s]", fontsize=14)
ylabel("Angle Vel [Rad/s]", fontsize=14)


figure(figsize=(10, 4))
step(tsb,Torque1sol, where='post',label='Torque1')
# step(tsb,Torque2sol, label='Torque2')
title("Control signal")
xlabel("Times [s]")
ylabel("Torque [Nm]", fontsize=14)
grid(True)


show(block=True)

