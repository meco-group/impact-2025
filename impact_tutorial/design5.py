from impact import *
import casadi as ca
import numpy as np
import rockit
import impact

rockit.GlobalOptions.set_cmake_flags(['-G','Ninja','-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang'])
rockit.GlobalOptions.set_cmake_build_type('Release')

mpc = impact.MPC(T=0.5)

# Add furuta model
furuta = mpc.add_model('fu_pendulum','furuta_velocity_mode.yaml')

# Parameters
x_current = mpc.parameter('x_current',furuta.nx)
x_final = mpc.parameter('x_final',furuta.nx)

# Objectives
mpc.add_objective(mpc.sum(furuta.dtheta2**2 ))

# Initial and final state constraints
mpc.subject_to(mpc.at_t0(furuta.x)==x_current)
mpc.subject_to(mpc.at_tf(furuta.x)==x_final)

# Path constraints
mpc.subject_to(-40 <= (furuta.dtheta1 <= 40 ), include_first=False, include_last=False)
mpc.subject_to(-40 <= (furuta.dtheta2 <= 40 ), include_first=False, include_last=False)
mpc.subject_to(-30 <= (mpc.der(furuta.dtheta2) <= 30 ), include_first=False, include_last=False)
mpc.subject_to(-ca.pi<= (furuta.theta1 <= ca.pi), include_first=False)

mpc.set_value(x_current, [-np.pi/3,0,0,0])
mpc.set_value(x_final, [np.pi/3,0,0,0])

ee = ca.vertcat(furuta.ee_x, furuta.ee_y, furuta.ee_z)
pivot = ca.vertcat(furuta.pivot_x, furuta.pivot_y, furuta.pivot_z)

# Transcription choice
method = external_method('acados', N=25,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=5)
mpc.method(method)

# Solve
sol = mpc.solve()


from pylab import *


[ts, theta2sol] = sol.sample(furuta.theta2, grid='control')
[ts_fine, theta2sol_fine] = sol.sample(furuta.theta2, grid='control')

[ts, dtheta2sol] = sol.sample(furuta.dtheta2, grid='control')
[ts_fine, dtheta2sol_fine] = sol.sample(furuta.dtheta2, grid='control')

print("theta2sol",theta2sol)

figure()
plot(ts, theta2sol,'b.')
plot(ts_fine, theta2sol_fine,'b')
plot(ts, dtheta2sol,'g.')
plot(ts_fine, dtheta2sol_fine,'g')
xlabel('Time [s]')
ylabel('theta2')

[ts, Torque1sol] = sol.sample(furuta.Torque1, grid='control')
[ts_fine, Torque1sol_fine] = sol.sample(furuta.Torque1, grid='control')

figure()
plot(ts, Torque1sol,'b.')
plot(ts_fine, Torque1sol_fine,'b')
xlabel('Time [s]')
ylabel('Torque [N]')

figure()


[_,ee_sol_fine] = sol.sample(ee,grid='control')
[_,ee_sol] = sol.sample(ee,grid='control')
[_,pivot_sol] = sol.sample(pivot,grid='control')


[_,theta1_sol] = sol.sample(furuta.theta1,grid='control')
[_,theta2_sol] = sol.sample(furuta.theta2,grid='control')

xlabel("theta1")
ylabel("theta2")
plot(theta1_sol,theta2_sol)

axis('square')

figure()

plot(ee_sol_fine[:,1],ee_sol_fine[:,2])
plot(ee_sol[:,1],ee_sol[:,2],'k.')

for k in range(ee_sol.shape[0]):
    plot([ee_sol[k,1],pivot_sol[k,1]],[ee_sol[k,2],pivot_sol[k,2]],'k-')

axis('square')
show()

# Export MPC controller

mpc.export('tutorial',short_output=True)
