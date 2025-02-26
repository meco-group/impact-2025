# Impact Tutorial

## Step 0: Hello world with the furuta pendulum

 * From the `workshop_impact` conda prompt, run `design0.py`
 * If you prefer _designing_ your MPC controller in Matlab, you may do with `design0.m`, to be run from a Matlab session that was launched using teh conda prompt.
 * You can run a simulation of the MPC controller running over 1000 samples using `simulate.py`
 
 
## Step 1: Torque control versus velocity control

 * In the `add_model` command, swap `furuta.yaml` by `furuta_velocity_mode.yaml`. Explain why the Torque plot is no longer piecewise-constant.
 
## Step 2: Playing with the objective function

 * Try changing the objective function, first to `dtheta1**2` and `dtheta**2`? Can you explain why the `theta2` plot looks differently? Hint: try plotting `dtheta2`, too.
 
 Trick question: explain the plots if you use `theta1**2` in the objective.
 
## Step 3: On the runtime statistics
 * Run `simulate.py` a couple of times, looking at the bottom plot.
   You'll notice the runtime having a noisy behaviour.
   That's because teh controller is running on a non-realtime operating system.
   
 * Switch `set_cmake_build_type('Debug')` to `set_cmake_build_type('Release')`.
   Verify that you find a speedup.
   
## Step 4: Extra constraints

 * In the `dtheta2` plot you made, you'll notice that max velocity is reached in one sampling time. Let's try to calm down this motion.
   Constrain `mpc.der(furuta.dtheta2)` to lie between -30 and 30 rad/s^2.
 * What happens if you change this constraints's `include_first` keyword agument to `True`? How does this affect the plot?
 
## Step 5: Switch to ACADOS

 * Instead of fatrop + MultipleShooting, switch to ACADOS solver:
```
method = external_method('acados', N=25,qp_solver='PARTIAL_CONDENSING_HPIPM',nlp_solver_max_iter=200,hessian_approx='EXACT',regularize_method = 'CONVEXIFY',integrator_type='ERK',nlp_solver_type='SQP',qp_solver_cond_N=5)
mpc.method(method)
```
 * You'll encounter two errors to fix
  1. `lh <= h(x,u) <= uh only supported for qualifier include_first=False`
      Solution: set `include_first` to `False` again.
  2. `AttributeError: 'AcadosMethod' object has no attribute 'poly_coeff'`
     It was nice being able to make high-resolution plots with (settings `grid='integrator', refine=10`)
     Unforntunately, such introspection is not availble with acados. Replace all occurances with `grid='control'`
 * Verify that you get simular results for `design5.py` and `simulate.py` for FATROP versus ACADOS.
 
## Step 6: Simulate in Simulink

 * In Simulink, load `library_tutorial.slx` that you'll find generated in `tutorial_build_dir`
 * Follow instructions by the tutor
 


