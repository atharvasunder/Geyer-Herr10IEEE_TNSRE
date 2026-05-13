"""
Profiling script for the integration loop.
Runs cProfile on the simulation and prints the top hotspots.
"""
import cProfile
import pstats
import io
import time
import numpy as np

from config import human_model 

dt = 3e-5
tStop = 5.0

rbs = human_model(dt)

# --- Profile the integration loop ---
def run_integration():
    t = [0.0]
    while t[-1] <= tStop:
        rbs.update_contacts(dt, ground_height=0)
        torque_array = np.zeros(6)
        rbs.update_joints(dt, torque_array)
        rbs.integrate_bodies(dt)
        t.append(t[-1] + dt)
        rbs.log_data()
    return t

pr = cProfile.Profile()
pr.enable()
t = run_integration()
pr.disable()    

with open('profile_output.txt', 'w') as f:
    # Print top 40 results sorted by cumulative time
    ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
    ps.print_stats(40)

    f.write("\n" + "=" * 80 + "\n")
    f.write("SORTED BY SELF TIME (tottime):\n")
    f.write("=" * 80 + "\n\n")
    
    ps2 = pstats.Stats(pr, stream=f).sort_stats('tottime')
    ps2.print_stats(40)

print("Profile saved to profile_output.txt")
