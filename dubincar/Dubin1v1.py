import time
import os
import gc
import psutil
import heterocl as hcl
from odp.Grid import Grid
import numpy as np
from odp.Plots import PlotOptions
from odp.Plots.plotting_utilities import plot_isosurface, plot_valuefunction
from odp.Shapes import *
from odp.solver import HJSolver
from DubinCarDynamics1v1 import DubinCar1v1

start_time = time.time()

grid_size = 6
grid = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
             6, np.array([grid_size, grid_size, grid_size, grid_size, grid_size, grid_size]))
process = psutil.Process(os.getpid())
print("1. Gigabytes consumed of the grid init {}".format(process.memory_info().rss / 1e9))

# Init the Dynamics
agents_1v1 = DubinCar1v1(uMode = "min", dMode = "max")

# Write the avoid set
capture_a = agents_1v1.capture_set(grid, 0.1, "capture")
capture_a = np.array(capture_a, dtype='float32')

avoid_set = np.array(capture_a, dtype='float32')

del capture_a
gc.collect()

process = psutil.Process(os.getpid())
print("2. Gigabytes consumed of the capture set {}".format(process.memory_info().rss / 1e9))

# Write the target set
target_dest = ShapeRectangle(grid, [0.6, 0.1, -1000, -1000, -1000, -1000], [0.8, 0.3, 1000, 1000, 1000, 1000])

target_a = agents_1v1.capture_set(grid, 0.1, "escape")
a_win = np.maximum(target_dest, target_a)
a_win = np.array(a_win, dtype='float32')
del target_a
del target_dest
gc.collect()

obs1_d = ShapeRectangle(grid, [-1000, -1000, -1000, -1000, -0.1, -1.0],
                                    [1000, 1000, 1000, 1000, 0.1, -0.3])  # defender stuck in obs1
obs2_d = ShapeRectangle(grid, [-1000, -1000, -1000, -1000, -0.1, 0.30],
                                    [1000, 1000, 1000, 1000, 0.1, 0.60])  # defender stuck in obs2
d_lose = np.minimum(obs1_d, obs2_d)
d_lose = np.array(d_lose, dtype='float32')
del obs2_d
del obs1_d
gc.collect()

reach_set = np.minimum(a_win, d_lose) # original
reach_set = np.array(reach_set, dtype='float32')
del a_win
del d_lose
gc.collect()
process = psutil.Process(os.getpid())
print("3. Gigabytes consumed of the reach_set {}".format(process.memory_info().rss/1e9))  # in bytes

# Timing
loopback_length = 2.5
t_step = 0.025

small_number = 1e-5
tau = np.arange(0, loopback_length + small_number, t_step)

po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0, 1, 2])


compMethods = {"TargetSetMode": "minVWithVTarget", "ObstacleSetMode": "maxVWithObstacle"} # original one
# compMethods = {"TargetSetMode": "minVWithVTarget"}
solve_start_time = time.time()

result = HJSolver(agents_1v1, grid, [reach_set, avoid_set], tau, compMethods, po, saveAllTimeSteps=None)
# While file needs to be saved locally, set save_fig=True and filename, recommend to set interactive_html=True for better visualization
po2 = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1,2],
                  slicesCut=[1], colorscale="Bluered", save_fig=True, filename="plots/3D_0_sublevel_set", interactive_html=True)

# STEP 6: Call Plotting function
plot_isosurface(grid, result, po2)
process = psutil.Process(os.getpid())
print(f"The CPU memory used during the calculation of the value function is {process.memory_info().rss/1e9: .2f} GB.")  # in bytes

solve_end_time = time.time()
print(f'The shape of the value function is {result.shape} \n')
print(f"The size of the value function is {result.nbytes / 1e9: .2f} GB or {result.nbytes/(1e6)} MB.")
print(f"The time of solving HJ is {solve_end_time - solve_start_time} seconds.")
print(f'The shape of the value function is {result.shape} \n')

# 6. Save the value function
np.save(f'DubinCar1v1_grid{grid_size}.npy', result)

print(f"The value function has been saved successfully.")

# Record the time of whole process
end_time = time.time()
print(f"The time of whole process is {end_time - start_time} seconds.")







