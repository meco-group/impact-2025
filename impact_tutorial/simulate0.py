import pylab as plt
import numpy as np

import sys
sys.path.insert(0,"tutorial_build_dir")
from impact import Impact

impact = Impact("tutorial",src_dir=".")

impact.solve()

# Get solution trajectory
x_opt = impact.get("x_opt", impact.ALL, impact.EVERYWHERE, impact.FULL)

# Plotting
_, ax = plt.subplots(3,1,sharex=True)
ax[0].plot(x_opt.T)
ax[0].set_title('Single OCP')
ax[0].set_xlabel('Sample')

print("Running MPC simulation loop")

history = []
runtime = []


for i in range(1000):

  mark = ((i//300) % 2 == 0)
  sign = (mark-0.5)*2

  impact.set("p", "x_final", impact.EVERYWHERE, impact.FULL, [sign*np.pi/3,0,0,0])

  impact.solve()
  
  runtime.append(impact.get_stats().runtime*1000)

  # Optimal input at k=0
  u = impact.get("u_opt", impact.ALL, 0, impact.FULL)

  # Simulate 1 step forward in time (ask MPC prediction model)
  x_sim = impact.get("x_opt", impact.ALL, 1, impact.FULL)
  
  # Add some artificial noise
  x_sim+= np.random.normal(0, 0.001, size=(x_sim.shape[0],1))

  # Update current state
  impact.set("x_current", impact.ALL, 0, impact.FULL, x_sim)
  history.append(x_sim)

# More plotting
ax[1].plot(np.hstack(history).T)
ax[1].set_title('Simulated MPC')
ax[1].set_xlabel('Sample')

ax[2].plot(np.hstack(runtime).T)
ax[2].set_title('Runtime [ms]')
ax[2].set_xlabel('Sample')
plt.show()
plt.show()



      
