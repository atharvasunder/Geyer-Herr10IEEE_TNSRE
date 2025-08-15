import time
import numpy as np

from config import human_model
# from output import RbsAnimation
from output_after_integration import PyqtgraphAnimation
from trackers import ContactTracker
import matplotlib.pyplot as plt

# clear screen (equivalent to Matlab 'clc'), close figures
print('\033[H\033[J')
plt.close('all')

########### define integration params ##############

dt = 3e-5               # [s] integration time step
sim_dt = 1e-3           # [s] simulation timestep
tStop = 1.28            # [s] simulation stop time

############ instantiate rigid body system ############

# instantiate rigid body system object
rbs = human_model(dt)

# store bodies comprising each leg
trunk = rbs.body_list[0]

thigh_1 = rbs.body_list[1]
shank_1 = rbs.body_list[2]
foot_1 = rbs.body_list[3]

thigh_2 = rbs.body_list[4]
shank_2 = rbs.body_list[5]
foot_2 = rbs.body_list[6]

# store joints of each leg
hip_1 = rbs.joint_list[0]
knee_1 = rbs.joint_list[1]
ankle_1 = rbs.joint_list[2]

hip_2 = rbs.joint_list[3]
knee_2 = rbs.joint_list[4]
ankle_2 = rbs.joint_list[5]

# store contact points and instantiate trackers for each leg
heel_1 = rbs.contact_list[0]
heel_1_tracker = ContactTracker(heel_1)
toe_1 = rbs.contact_list[1]
toe_1_tracker = ContactTracker(toe_1)

heel_2 = rbs.contact_list[2]
heel_2_tracker = ContactTracker(heel_2)
toe_2 = rbs.contact_list[3]
toe_2_tracker = ContactTracker(toe_2)

# store muscles (instances of mtc.py's class) for each leg (VAS, SOL, GAS, TA, GLU, HAM, HFL)
VAS_1 = rbs.muscle_list[0]
SOL_1 = rbs.muscle_list[1]
GAS_1 = rbs.muscle_list[2]
TA_1 = rbs.muscle_list[3]
GLU_1 = rbs.muscle_list[4]
HAM_1 = rbs.muscle_list[5]
HFL_1 = rbs.muscle_list[6]

VAS_2 = rbs.muscle_list[7]
SOL_2 = rbs.muscle_list[8]
GAS_2 = rbs.muscle_list[9]
TA_2 = rbs.muscle_list[10]
GLU_2 = rbs.muscle_list[11]
HAM_2 = rbs.muscle_list[12]
HFL_2 = rbs.muscle_list[13]

# store lever arms between muscles and bodies for all joints the muscle actuates (instances of musculo_skeleton.py class)
VAS_1_musculo_skeletal = rbs.musculo_skeletal_list[0]
SOL_1_musculo_skeletal = rbs.musculo_skeletal_list[1]
GAS_1_musculo_skeletal = rbs.musculo_skeletal_list[2]
TA_1_musculo_skeletal = rbs.musculo_skeletal_list[3]
GLU_1_musculo_skeletal = rbs.musculo_skeletal_list[4]
HAM_1_musculo_skeletal = rbs.musculo_skeletal_list[5]
HFL_1_musculo_skeletal = rbs.musculo_skeletal_list[6]

VAS_2_musculo_skeletal = rbs.musculo_skeletal_list[7]
SOL_2_musculo_skeletal = rbs.musculo_skeletal_list[8]
GAS_2_musculo_skeletal = rbs.musculo_skeletal_list[9]
TA_2_musculo_skeletal = rbs.musculo_skeletal_list[10]
GLU_2_musculo_skeletal = rbs.musculo_skeletal_list[11]
HAM_2_musculo_skeletal = rbs.musculo_skeletal_list[12]
HFL_2_musculo_skeletal = rbs.musculo_skeletal_list[13]

# store feedback pathways for each muscle (same order as muscle list)
VAS_1_feedback_loop = rbs.feedback_list[0]
SOL_1_feedback_loop = rbs.feedback_list[1]
GAS_1_feedback_loop = rbs.feedback_list[2]
TA_1_feedback_loop = rbs.feedback_list[3]
GLU_1_feedback_loop = rbs.feedback_list[4]
HAM_1_feedback_loop = rbs.feedback_list[5]
HFL_1_feedback_loop = rbs.feedback_list[6]

VAS_2_feedback_loop = rbs.feedback_list[7]
SOL_2_feedback_loop = rbs.feedback_list[8]
GAS_2_feedback_loop = rbs.feedback_list[9]
TA_2_feedback_loop = rbs.feedback_list[10]
GLU_2_feedback_loop = rbs.feedback_list[11]
HAM_2_feedback_loop = rbs.feedback_list[12]
HFL_2_feedback_loop = rbs.feedback_list[13]

############# initialize stance and swing muscle lists and start integration loop ##############

t = [0.0]
t_wall_start = time.perf_counter() # start wall time

leg_1_muscles = rbs.muscle_list[:7]
leg_2_muscles = rbs.muscle_list[7:]

leg_1_feedbacks = rbs.feedback_list[:7]
leg_2_feedbacks = rbs.feedback_list[7:]

leg_1_musculo_skeletal = rbs.musculo_skeletal_list[:7]
leg_2_musculo_skeletal = rbs.musculo_skeletal_list[7:]

leg_1_contacts = rbs.contact_list[:2]
leg_2_contacts = rbs.contact_list[2:]

leg_1_trackers = [heel_1_tracker, toe_1_tracker]
leg_2_trackers = [heel_2_tracker, toe_2_tracker]

phase = 'single support'

trailing_leg = {'muscles':leg_2_muscles,'feedbacks':leg_2_feedbacks,'musculo_skeletal':leg_2_musculo_skeletal,'contacts':leg_2_contacts,'trackers':leg_2_trackers}
swing_leg = {'muscles':leg_1_muscles,'feedbacks':leg_1_feedbacks,'musculo_skeletal':leg_1_musculo_skeletal,'contacts':leg_1_contacts,'trackers':leg_1_trackers}
heading_leg = {'muscles':leg_1_muscles,'feedbacks':leg_1_feedbacks,'musculo_skeletal':leg_1_musculo_skeletal,'contacts':leg_1_contacts,'trackers':leg_1_trackers}     # initialized randomly here

# integration while loop
while t[-1] <= tStop:

    # compute contact forces based on position and velocity of contact points/sites
    heel_1_tracker.append() # logs data of the contact (eg: grf)
    heel_1.update(dt, ground_height=0)

    heel_2_tracker.append()
    heel_2.update(dt, ground_height=0)
    
    toe_1_tracker.append()
    toe_1.update(dt, ground_height=0)

    toe_2_tracker.append()
    toe_2.update(dt, ground_height=0)

    torque_array = np.zeros(6)  # array to store muscle torques acting on each joint in the current timestep (refreshed every timestep)

    if phase == 'single support':

        # step 1: update muscles

        grf = np.zeros(2)
        for tracker in trailing_leg['trackers']:
            grf += tracker.get_data()

        grf_norm_trailing = np.linalg.norm(grf)   # magnitude of grf
        grf_norm_swing = 0           # grf norm of the flight leg here (for use by vasti)

        for i, muscle in enumerate(trailing_leg['muscles']):
            l_mtc = trailing_leg['musculo_skeletal'][i].update_l_mtc()
            S = trailing_leg['feedbacks'][i].update('stance', 0, grf_norm_trailing, grf_norm_swing)   # 0 is for Dsup
            F_mtc = muscle.update(S, l_mtc)
            r_array = trailing_leg['musculo_skeletal'][i].update_lever_arm()
            torque_contribution = trailing_leg['musculo_skeletal'][i].get_torques(F_mtc) # gives contribution of this stance leg muscle to all joints  
            torque_array += torque_contribution
        
        for i, muscle in enumerate(swing_leg['muscles']):
            l_mtc = swing_leg['musculo_skeletal'][i].update_l_mtc()
            S = swing_leg['feedbacks'][i].update('swing', 0, grf_norm_swing, grf_norm_trailing)   # 0 is for Dsup
            F_mtc = muscle.update(S, l_mtc)
            r_array = swing_leg['musculo_skeletal'][i].update_lever_arm()
            torque_contribution = swing_leg['musculo_skeletal'][i].get_torques(F_mtc) # gives contribution of this stance leg muscle to all joints  

            torque_array += torque_contribution
            
        # step 2: check for switch condition
        if swing_leg['contacts'][0].contact is True:
            phase = 'double support'
            heading_leg = swing_leg

    elif phase == 'double support':

        # step 1: update muscles

        grf_trailing = np.zeros(2)
        for tracker in trailing_leg['trackers']:
            grf_trailing += tracker.get_data()

        grf_norm_trailing = np.linalg.norm(grf_trailing)   # magnitude of grf

        grf_heading = np.zeros(2)
        for tracker in heading_leg['trackers']:
            grf_heading += tracker.get_data()

        grf_norm_heading = np.linalg.norm(grf_heading)   # magnitude of grf of heading leg

        for i,muscle in enumerate(trailing_leg['muscles']):
            l_mtc = trailing_leg['musculo_skeletal'][i].update_l_mtc()
            S = trailing_leg['feedbacks'][i].update('stance', 1, grf_norm_trailing, grf_norm_heading)   # 1 is for Dsup
            F_mtc = muscle.update(S, l_mtc)
            r_array = trailing_leg['musculo_skeletal'][i].update_lever_arm()
            torque_contribution = trailing_leg['musculo_skeletal'][i].get_torques(F_mtc) # gives contribution of this stance leg muscle to all joints  
            torque_array += torque_contribution
        
        for i, muscle in enumerate(heading_leg['muscles']):
            l_mtc = heading_leg['musculo_skeletal'][i].update_l_mtc()
            S = heading_leg['feedbacks'][i].update('stance', 0, grf_norm_heading, grf_norm_trailing)   # 0 is for Dsup
            F_mtc = muscle.update(S, l_mtc)
            r_array = heading_leg['musculo_skeletal'][i].update_lever_arm()
            torque_contribution = heading_leg['musculo_skeletal'][i].get_torques(F_mtc) # gives contribution of this stance leg muscle to all joints  
            torque_array += torque_contribution

        # step 2: check for switch condition
        if trailing_leg['contacts'][1].contact is False:
            phase = 'single support'
            swing_leg = trailing_leg
            trailing_leg = heading_leg

    # "external torques on joints"
    tau_h_1, tau_k_1, tau_a_1 = torque_array[:3]
    tau_h_2, tau_k_2, tau_a_2 = torque_array[3:]
    
    # update mate and base forces and torques using joint interactions
    hip_1.update(dt,tau_h_1)
    knee_1.update(dt,tau_k_1)
    ankle_1.update(dt,tau_a_1)

    hip_2.update(dt,tau_h_2)
    knee_2.update(dt,tau_k_2)
    ankle_2.update(dt,tau_a_2)

    # integrate for a single timestep
    for body in rbs.body_list:
        body.integrate(dt)

    t.append(t[-1] + dt)
    
    # log data
    for body in rbs.body_list:
        body.log_data()
    
    for joint in rbs.joint_list:
        joint.log_data()

    for muscle in rbs.muscle_list:
        muscle.log_data()

print(f"[DONE] Simulated time: {tStop:.2f}s | Wall time for integration: {time.perf_counter() - t_wall_start:.2f}s")

################### plotting output ####################

plot_start = time.perf_counter()
walking_anim = PyqtgraphAnimation(time_step=sim_dt, rigid_body_system=rbs)
N = len(t)  # no of timesteps
ratio=1     # animation speed multiplier
step = int(ratio*sim_dt / dt)

indices = np.arange(0, N-1, step, dtype=int)

print(t[indices[-1]])

idx = 0
k = 0
while k < len(indices) - 1:
    walking_anim.update_animation(idx)
    idx = indices[k]

    k+=1

print(f"[DONE] Simulated time: {tStop:.2f}s | Wall time for plotting: {time.perf_counter() - plot_start:.2f}s")

# walking_anim.muscle_torque_contributions(t)