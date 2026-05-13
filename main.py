import time
import numpy as np

import config
from rbs_init import human_model
# from output import RbsAnimation
from visualizer import WalkVisualizer
from gait_control import control_mode_manager
from gait_control.single_support import SingleSupportControl
from gait_control.double_support import DoubleSupportControl
from gait_control.NMcontrol.nm_controller import NMController
import matplotlib.pyplot as plt

# clear screen (equivalent to Matlab 'clc'), close figures
print('\033[H\033[J')   
plt.close('all')

########### define integration params ##############

dt = 3e-5               # [s] integration time step
sim_dt = 1e-3            # [s] data logging / visualization timestep
tStop = 0.5            # [s] simulation stop time

############ instantiate rigid body system ############

# instantiate rigid body system object
rbs = human_model(dt)

############ instantiate control state machine ############

# instantiate state machine class
ctrl_mgr = control_mode_manager.ctrl_manager()

# load discrete control rate
ctrl_dt = config.NERVOUS_SYSTEM_DICTIONARY["GENERAL"]["TIME_STEP"]

# instantiate gait controller
nm_controller = NMController(ctrl_dt, tStop)

# instantiate gait phase control modes
ss_ctrl_mode = SingleSupportControl(ctrl_mgr, 'SingleSupport', ['DoubleSupport'], nm_controller)
ds_ctrl_mode = DoubleSupportControl(ctrl_mgr, 'DoubleSupport', ['SingleSupport'], nm_controller)

# register control modes in the state machine
ctrl_mgr.add_ctrl_mode(ss_ctrl_mode.name, ss_ctrl_mode)
ctrl_mgr.add_ctrl_mode(ds_ctrl_mode.name, ds_ctrl_mode)

# start in single support (one foot on the ground at initialization)
ctrl_mgr.set_ctrl_mode(ss_ctrl_mode.name, rbs.states, 'leg_2')

# initialize the neuromuscular controller buffers to starting state
nm_controller.reset(rbs.states, 'leg_1')
nm_controller.reset(rbs.states, 'leg_2')

############# start integration loop ##############

# Pre-compute time array
N = int(tStop / dt) + 1
t = np.arange(0, N * dt, dt)

# Pre-allocate torque array
torque_array = np.zeros(6)

# Phase arrays for logging at sim_dt
swing_phase_log_1 = []
swing_phase_log_2 = []
grf_log_1 = []
grf_log_2 = []

# Log data only every log_interval steps
log_interval = max(1, int(sim_dt / dt))

t_wall_start = time.perf_counter() # start wall time

t_ctrl_next = 0.0

# integration while loop
for step_i in range(N):
    sim_time = step_i * dt

    # compute contact forces based on position and velocity of contact points/sites
    rbs.update_contacts(dt, ground_height=0)

    # update mate and base sites forces and torques using joint interactions
    rbs.update_joints(dt, torque_array)

    # integrate for a single timestep to obtain new positions and velocities of all bodies
    rbs_states = rbs.integrate_bodies(dt)

    # compute control commands at 1000 Hz or desired neural loop frequency
    if sim_time >= t_ctrl_next - 1e-8:
        ctrl_mode, ctrl_cmd = ctrl_mgr.get_ctrl(rbs_states)
        torque_array[:] = ctrl_cmd  # apply the hold
        nm_controller.log_data()
        t_ctrl_next += ctrl_dt

    # log data only at sim_dt intervals
    if step_i % log_interval == 0:
        rbs.log_data()
        swing_phase_log_1.append(1 if rbs_states['leg_1'].get('phase') == 'swing' else 0)
        swing_phase_log_2.append(1 if rbs_states['leg_2'].get('phase') == 'swing' else 0)
        grf_log_1.append(rbs_states['leg_1']['grf'])
        grf_log_2.append(rbs_states['leg_2']['grf'])

    # print progress every 0.5 simulated seconds
    if step_i % int(0.5 / dt) == 0 and step_i > 0:
        print(f"Simulated {step_i * dt:.2f} seconds...")

print(f"[DONE] Simulated time: {tStop:.2f}s | Wall time for integration: {time.perf_counter() - t_wall_start:.2f}s")

################### joint angle plots ####################

# Build time array for logged data (logged at sim_dt intervals)
N_logged_joints = len(rbs.joint_list[0].q_list)
t_logged = np.arange(N_logged_joints) * sim_dt

# Helper function to shade swing phase dynamically
def shade_swing(ax, t_logged, swing_log, color):
    if len(swing_log) > 0:
        log_len = min(len(t_logged), len(swing_log))
        ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_log[:log_len])==1, 
                        transform=ax.get_xaxis_transform(), alpha=0.2, color=color, label='Swing')

t_ctrl = np.arange(nm_controller.num_ctrl_steps) * ctrl_dt

# ==========================================
# NEW PLOTS: HIP, KNEE, ANKLE
# ==========================================

# --- 1. HIP PLOT --- (4 rows, 2 columns)
fig_hip, axes_hip = plt.subplots(4, 2, sharex=True, figsize=(12, 10))
fig_hip.suptitle('Hip Kinematics & Muscle Activations', fontsize=16)

# Leg 1 (Red)
axes_hip[0, 0].plot(t_logged, np.degrees(rbs.joint_list[0].q_list), color='red')
axes_hip[0, 0].set_ylabel('Angle [deg]')
axes_hip[0, 0].set_title('Leg 1 (Hip)')

axes_hip[1, 0].plot(t_logged, rbs.joint_list[0].tau_list, color='red')
axes_hip[1, 0].set_ylabel('Torque [Nm]')

axes_hip[2, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['HFL']['A'][:len(t_ctrl)], color='red')
axes_hip[2, 0].set_ylabel('HFL Act')

axes_hip[3, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['GLU']['A'][:len(t_ctrl)], color='red')
axes_hip[3, 0].set_ylabel('GLU Act')
axes_hip[3, 0].set_xlabel('Time [s]')

for i in range(4):
    axes_hip[i, 0].grid(True, alpha=0.3)
    shade_swing(axes_hip[i, 0], t_logged, swing_phase_log_1, 'red')

# Leg 2 (Blue)
axes_hip[0, 1].plot(t_logged, np.degrees(rbs.joint_list[3].q_list), color='blue')
axes_hip[0, 1].set_title('Leg 2 (Hip)')

axes_hip[1, 1].plot(t_logged, rbs.joint_list[3].tau_list, color='blue')

axes_hip[2, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['HFL']['A'][:len(t_ctrl)], color='blue')

axes_hip[3, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['GLU']['A'][:len(t_ctrl)], color='blue')
axes_hip[3, 1].set_xlabel('Time [s]')

for i in range(4):
    axes_hip[i, 1].grid(True, alpha=0.3)
    shade_swing(axes_hip[i, 1], t_logged, swing_phase_log_2, 'blue')

fig_hip.tight_layout()

# --- 2. KNEE PLOT --- (4 rows, 2 columns)
fig_knee, axes_knee = plt.subplots(4, 2, sharex=True, figsize=(12, 10))
fig_knee.suptitle('Knee Kinematics & Muscle Activations', fontsize=16)

# Leg 1 (Red)
axes_knee[0, 0].plot(t_logged, np.degrees(rbs.joint_list[1].q_list), color='red')
axes_knee[0, 0].set_ylabel('Angle [deg]')
axes_knee[0, 0].set_title('Leg 1 (Knee)')

axes_knee[1, 0].plot(t_logged, rbs.joint_list[1].tau_list, color='red')
axes_knee[1, 0].set_ylabel('Torque [Nm]')

axes_knee[2, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['VAS']['A'][:len(t_ctrl)], color='red')
axes_knee[2, 0].set_ylabel('VAS Act')

axes_knee[3, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['HAM']['A'][:len(t_ctrl)], color='red')
axes_knee[3, 0].set_ylabel('HAM Act')
axes_knee[3, 0].set_xlabel('Time [s]')

for i in range(4):
    axes_knee[i, 0].grid(True, alpha=0.3)
    shade_swing(axes_knee[i, 0], t_logged, swing_phase_log_1, 'red')

# Leg 2 (Blue)
axes_knee[0, 1].plot(t_logged, np.degrees(rbs.joint_list[4].q_list), color='blue')
axes_knee[0, 1].set_title('Leg 2 (Knee)')

axes_knee[1, 1].plot(t_logged, rbs.joint_list[4].tau_list, color='blue')

axes_knee[2, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['VAS']['A'][:len(t_ctrl)], color='blue')

axes_knee[3, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['HAM']['A'][:len(t_ctrl)], color='blue')
axes_knee[3, 1].set_xlabel('Time [s]')

for i in range(4):
    axes_knee[i, 1].grid(True, alpha=0.3)
    shade_swing(axes_knee[i, 1], t_logged, swing_phase_log_2, 'blue')

fig_knee.tight_layout()

# --- 3. ANKLE PLOT --- (5 rows, 2 columns)
fig_ankle, axes_ankle = plt.subplots(5, 2, sharex=True, figsize=(12, 12))
fig_ankle.suptitle('Ankle Kinematics & Muscle Activations', fontsize=16)

# Leg 1 (Red)
axes_ankle[0, 0].plot(t_logged, np.degrees(rbs.joint_list[2].q_list), color='red')
axes_ankle[0, 0].set_ylabel('Angle [deg]')
axes_ankle[0, 0].set_title('Leg 1 (Ankle)')

axes_ankle[1, 0].plot(t_logged, rbs.joint_list[2].tau_list, color='red')
axes_ankle[1, 0].set_ylabel('Torque [Nm]')

axes_ankle[2, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['GAS']['A'][:len(t_ctrl)], color='red')
axes_ankle[2, 0].set_ylabel('GAS Act')

axes_ankle[3, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['TA']['A'][:len(t_ctrl)], color='red')
axes_ankle[3, 0].set_ylabel('TA Act')

axes_ankle[4, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['SOL']['A'][:len(t_ctrl)], color='red')
axes_ankle[4, 0].set_ylabel('SOL Act')
axes_ankle[4, 0].set_xlabel('Time [s]')

for i in range(5):
    axes_ankle[i, 0].grid(True, alpha=0.3)
    shade_swing(axes_ankle[i, 0], t_logged, swing_phase_log_1, 'red')

# Leg 2 (Blue)
axes_ankle[0, 1].plot(t_logged, np.degrees(rbs.joint_list[5].q_list), color='blue')
axes_ankle[0, 1].set_title('Leg 2 (Ankle)')

axes_ankle[1, 1].plot(t_logged, rbs.joint_list[5].tau_list, color='blue')

axes_ankle[2, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['GAS']['A'][:len(t_ctrl)], color='blue')

axes_ankle[3, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['TA']['A'][:len(t_ctrl)], color='blue')

axes_ankle[4, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['SOL']['A'][:len(t_ctrl)], color='blue')
axes_ankle[4, 1].set_xlabel('Time [s]')

for i in range(5):
    axes_ankle[i, 1].grid(True, alpha=0.3)
    shade_swing(axes_ankle[i, 1], t_logged, swing_phase_log_2, 'blue')

fig_ankle.tight_layout()

# --- GRF Plot ---
fig_grf, ax_grf = plt.subplots(figsize=(10, 4))
fig_grf.suptitle('Ground Reaction Forces', fontsize=14)

log_len = min(len(t_logged), len(grf_log_1))
ax_grf.plot(t_logged[:log_len], grf_log_1[:log_len], label='Leg 1 GRF', color='red', linewidth=1.5)
ax_grf.plot(t_logged[:log_len], grf_log_2[:log_len], label='Leg 2 GRF', color='blue', linewidth=1.5)

ax_grf.set_ylabel('Force [N]')
ax_grf.set_xlabel('Time [s]')
ax_grf.grid(True, alpha=0.3)
ax_grf.legend(loc='upper right')
fig_grf.tight_layout()

# --- Torso Angle Plot ---
fig_torso, ax_torso = plt.subplots(figsize=(8, 4))
fig_torso.suptitle('Torso (Trunk) Angle', fontsize=14)

log_len_torso = min(len(t_logged), len(rbs.body_list[0].p_list))
ax_torso.plot(t_logged[:log_len_torso], np.degrees(rbs.body_list[0].p_list[:log_len_torso]), color='purple', linewidth=1.5, label='Torso Angle')

ax_torso.set_ylabel('Angle [deg]')
ax_torso.set_xlabel('Time [s]')
ax_torso.grid(True, alpha=0.3)
ax_torso.legend(loc='upper right')
fig_torso.tight_layout()

# plt.show(block=False)

# ============================================================
# AESTHETIC SUMMARY PLOTS: JOINT ANGLES AND JOINT TORQUES
# Stance phase is highlighted using the phase logic logged above:
#   swing_phase_log == 0  -> stance
#   swing_phase_log == 1  -> swing
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 24,
    "lines.linewidth": 2.5,
})

joint_names = ["Hip", "Knee", "Ankle"]
leg_1_joints = rbs.joint_list[0:3]
leg_2_joints = rbs.joint_list[3:6]


def get_stance_mask(t_logged, swing_phase_log):
    """
    Derive stance indices from the logged phase logic above.

    Earlier in the code, swing_phase_log is stored as:
        1 if phase == 'swing' else 0

    Therefore, stance corresponds to swing_phase_log == 0.
    """
    log_len = min(len(t_logged), len(swing_phase_log))
    stance_mask = np.array(swing_phase_log[:log_len]) == 0
    return log_len, stance_mask


def shade_stance(ax, t_logged, swing_phase_log, label="Stance phase"):
    """Highlight stance phase on an axis using the existing phase logs."""
    if len(swing_phase_log) == 0:
        return

    log_len, stance_mask = get_stance_mask(t_logged, swing_phase_log)
    ax.fill_between(
        t_logged[:log_len],
        0,
        1,
        where=stance_mask,
        transform=ax.get_xaxis_transform(),
        alpha=0.18,
        color="gray",
        label=label,
        zorder=0,
    )


def plot_leg_summary(leg_name, leg_joints, swing_phase_log, line_color):
    """Create separate aesthetic angle and torque windows for one leg."""

    # -----------------------------
    # JOINT ANGLES FIGURE
    # -----------------------------
    fig_angles, axes_angles = plt.subplots(3, 1, sharex=True, figsize=(13, 9))
    # fig_angles.suptitle(f"{leg_name}: Hip, Knee, and Ankle Joint Angles vs Time", fontweight="bold")

    for i, ax in enumerate(axes_angles):
        q = np.degrees(leg_joints[i].q_list)
        n = min(len(t_logged), len(q))

        shade_stance(ax, t_logged, swing_phase_log)
        ax.plot(t_logged[:n], q[:n], color=line_color, label=f"{joint_names[i]} angle", zorder=2)

        ax.set_ylabel(f"{joint_names[i]}\nAngle [deg]")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", frameon=True)

    axes_angles[-1].set_xlabel("Time [s]")
    fig_angles.tight_layout(rect=[0, 0, 1, 0.95])

    # -----------------------------
    # JOINT TORQUES FIGURE
    # -----------------------------
    fig_torques, axes_torques = plt.subplots(3, 1, sharex=True, figsize=(13, 9))
    # fig_torques.suptitle(f"{leg_name}: Hip, Knee, and Ankle Joint Torques vs Time", fontweight="bold")

    for i, ax in enumerate(axes_torques):
        tau = leg_joints[i].tau_list
        n = min(len(t_logged), len(tau))

        shade_stance(ax, t_logged, swing_phase_log)
        ax.plot(t_logged[:n], tau[:n], color=line_color, label=f"{joint_names[i]} torque", zorder=2)

        ax.set_ylabel(f"{joint_names[i]}\nTorque [Nm]")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", frameon=True)

    axes_torques[-1].set_xlabel("Time [s]")
    fig_torques.tight_layout(rect=[0, 0, 1, 0.95])


# Separate windows for each leg
plot_leg_summary("Leg 1", leg_1_joints, swing_phase_log_1, "red")
plot_leg_summary("Leg 2", leg_2_joints, swing_phase_log_2, "blue")

################### plotting output ####################

plot_start = time.perf_counter()

# Visualization settings: achieve ~60 FPS visualization rate for smooth playback
render_fps = 60
vis_dt = 1.0 / render_fps

# Logged data is at sim_dt resolution, not dt resolution
N_logged = len(rbs.body_list[0].x_list)
ratio = 8   # animation speed multiplier
# Determine step to subsample logged frames to achieve render_fps playback
step = max(1, int(vis_dt / (ratio * sim_dt)))

indices = np.arange(0, N_logged, step, dtype=int)

viz = WalkVisualizer(rigid_body_system=rbs)

# Build timestamped save path in 'videos' folder
import os
from datetime import datetime
vidros_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_path = os.path.join(vidros_dir, f'walk_{timestamp}.mp4')

# interval is in ms per frame (e.g. 16ms for ~60 FPS)
viz.animate(indices, interval=int(1000/render_fps), save_path=save_path)

print(f"[DONE] Simulated time: {tStop:.2f}s | Wall time for plotting: {time.perf_counter() - plot_start:.2f}s")


# ==========================================
# OLD PLOTS (COMMENTED OUT)
# ==========================================

# # --- Leg 1: hip_1, knee_1, ankle_1 (joint indices 0, 1, 2) ---
# fig1, axes1 = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
# fig1.suptitle('Leg 1 – Joint Angles', fontsize=14)
# 
# for ax, joint in zip(axes1, rbs.joint_list[0:3]):
#     ax.plot(t_logged, np.degrees(joint.q_list), linewidth=0.8)
#     ax.set_ylabel(f'{joint.name} [deg]')
#     ax.grid(True, alpha=0.3)
#     # Add swing phase highlights
#     if len(swing_phase_log_1) > 0:
#         log_len = min(len(t_logged), len(swing_phase_log_1))
#         ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_phase_log_1[:log_len])==1, 
#                         transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# 
# axes1[0].legend(loc='upper right')
# axes1[-1].set_xlabel('Time [s]')
# fig1.tight_layout()
# 
# # --- Leg 2: hip_2, knee_2, ankle_2 (joint indices 3, 4, 5) ---
# fig2, axes2 = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
# fig2.suptitle('Leg 2 – Joint Angles', fontsize=14)
# 
# for ax, joint in zip(axes2, rbs.joint_list[3:6]):
#     ax.plot(t_logged, np.degrees(joint.q_list), linewidth=0.8)
#     ax.set_ylabel(f'{joint.name} [deg]')
#     ax.grid(True, alpha=0.3)
#     # Add swing phase highlights
#     if len(swing_phase_log_2) > 0:
#         log_len = min(len(t_logged), len(swing_phase_log_2))
#         ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_phase_log_2[:log_len])==1, 
#                         transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# 
# axes2[0].legend(loc='upper right')
# axes2[-1].set_xlabel('Time [s]')
# fig2.tight_layout()
# 
# # --- Leg 1: hip_1, knee_1, ankle_1 (joint indices 0, 1, 2) Joint Torques ---
# fig3, axes3 = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
# fig3.suptitle('Leg 1 – Joint Torques', fontsize=14)
# 
# for ax, joint in zip(axes3, rbs.joint_list[0:3]):
#     ax.plot(t_logged, joint.tau_list, linewidth=0.8)
#     ax.set_ylabel(f'{joint.name} [Nm]')
#     ax.grid(True, alpha=0.3)
#     if len(swing_phase_log_1) > 0:
#         log_len = min(len(t_logged), len(swing_phase_log_1))
#         ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_phase_log_1[:log_len])==1, 
#                         transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# 
# axes3[0].legend(loc='upper right')
# axes3[-1].set_xlabel('Time [s]')
# fig3.tight_layout()
# 
# # --- Leg 2: hip_2, knee_2, ankle_2 (joint indices 3, 4, 5) Joint Torques ---
# fig4, axes4 = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
# fig4.suptitle('Leg 2 – Joint Torques', fontsize=14)
# 
# for ax, joint in zip(axes4, rbs.joint_list[3:6]):
#     ax.plot(t_logged, joint.tau_list, linewidth=0.8)
#     ax.set_ylabel(f'{joint.name} [Nm]')
#     ax.grid(True, alpha=0.3)
#     if len(swing_phase_log_2) > 0:
#         log_len = min(len(t_logged), len(swing_phase_log_2))
#         ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_phase_log_2[:log_len])==1, 
#                         transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# 
# axes4[0].legend(loc='upper right')
# axes4[-1].set_xlabel('Time [s]')
# fig4.tight_layout()
# 
# # --- MTC States and Lever Arms Plots ---
# # t_ctrl = np.arange(nm_controller.num_ctrl_steps) * ctrl_dt
# 
# # mtc_names = ["VAS", "SOL", "GAS", "TA", "HAM", "GLU", "HFL"]
# # mtc_state_names = ["A", "F_mtc", "l_mtc", "l_ce"]
# # ms_states = ["hip_lever", "knee_lever", "ankle_lever"]
# 
# # for leg in ['leg_1', 'leg_2']:
# #     swing_log = swing_phase_log_1 if leg == 'leg_1' else swing_phase_log_2
# #     for m in mtc_names:
# #         # MTC states figure
# #         fig_mtc, axes_mtc = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
# #         fig_mtc.suptitle(f'{m} {leg} - MTC States', fontsize=14)
# #         for ax, state in zip(axes_mtc, mtc_state_names):
# #             data = nm_controller.log_data_dict[leg][m][state]
# #             ax.plot(t_ctrl[:len(data)], data, linewidth=1.0)
# #             ax.set_ylabel(state)
# #             ax.grid(True, alpha=0.3)
# #             if len(swing_log) > 0:
# #                 log_len = min(len(t_logged), len(swing_log))
# #                 ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_log[:log_len])==1, 
# #                                 transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# #         axes_mtc[0].legend(loc='upper right')
# #         axes_mtc[-1].set_xlabel('Time [s]')
# #         fig_mtc.tight_layout()
#     
# #         # MS states figure
# #         fig_ms, axes_ms = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
# #         fig_ms.suptitle(f'{m} {leg} - Lever Arms', fontsize=14)
# #         for ax, state in zip(axes_ms, ms_states):
# #             data = nm_controller.log_data_dict[leg][m][state]
# #             ax.plot(t_ctrl[:len(data)], data, linewidth=1.0)
# #             ax.set_ylabel(state)
# #             ax.grid(True, alpha=0.3)
# #             if len(swing_log) > 0:
# #                 log_len = min(len(t_logged), len(swing_log))
# #                 ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_log[:log_len])==1, 
# #                                 transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# #         axes_ms[-1].set_xlabel('Time [s]')
# #         fig_ms.tight_layout()


# import time
# import numpy as np

# import config
# from rbs_init import human_model
# # from output import RbsAnimation
# from visualizer import WalkVisualizer
# from gait_control import control_mode_manager
# from gait_control.single_support import SingleSupportControl
# from gait_control.double_support import DoubleSupportControl
# from gait_control.NMcontrol.nm_controller import NMController
# import matplotlib.pyplot as plt

# # clear screen (equivalent to Matlab 'clc'), close figures
# print('\033[H\033[J')   
# plt.close('all')

# ########### define integration params ##############

# dt = 3e-5               # [s] integration time step
# sim_dt = 1e-3            # [s] data logging / visualization timestep
# tStop = 0.5            # [s] simulation stop time

# ############ instantiate rigid body system ############

# # instantiate rigid body system object
# rbs = human_model(dt)

# ############ instantiate control state machine ############

# # instantiate state machine class
# ctrl_mgr = control_mode_manager.ctrl_manager()

# # load discrete control rate
# ctrl_dt = config.NERVOUS_SYSTEM_DICTIONARY["GENERAL"]["TIME_STEP"]

# # instantiate gait controller
# nm_controller = NMController(ctrl_dt, tStop)

# # instantiate gait phase control modes
# ss_ctrl_mode = SingleSupportControl(ctrl_mgr, 'SingleSupport', ['DoubleSupport'], nm_controller)
# ds_ctrl_mode = DoubleSupportControl(ctrl_mgr, 'DoubleSupport', ['SingleSupport'], nm_controller)

# # register control modes in the state machine
# ctrl_mgr.add_ctrl_mode(ss_ctrl_mode.name, ss_ctrl_mode)
# ctrl_mgr.add_ctrl_mode(ds_ctrl_mode.name, ds_ctrl_mode)

# # start in single support (one foot on the ground at initialization)
# ctrl_mgr.set_ctrl_mode(ss_ctrl_mode.name, rbs.states, 'leg_2')

# # initialize the neuromuscular controller buffers to starting state
# nm_controller.reset(rbs.states, 'leg_1')
# nm_controller.reset(rbs.states, 'leg_2')

# ############# start integration loop ##############

# # Pre-compute time array
# N = int(tStop / dt) + 1
# t = np.arange(0, N * dt, dt)

# # Pre-allocate torque array
# torque_array = np.zeros(6)

# # Phase arrays for logging at sim_dt
# swing_phase_log_1 = []
# swing_phase_log_2 = []
# grf_log_1 = []
# grf_log_2 = []

# # Log data only every log_interval steps
# log_interval = max(1, int(sim_dt / dt))

# t_wall_start = time.perf_counter() # start wall time

# t_ctrl_next = 0.0

# # integration while loop
# for step_i in range(N):
#     sim_time = step_i * dt

#     # compute contact forces based on position and velocity of contact points/sites
#     rbs.update_contacts(dt, ground_height=0)

#     # update mate and base sites forces and torques using joint interactions
#     rbs.update_joints(dt, torque_array)

#     # integrate for a single timestep to obtain new positions and velocities of all bodies
#     rbs_states = rbs.integrate_bodies(dt)

#     # compute control commands at 1000 Hz or desired neural loop frequency
#     if sim_time >= t_ctrl_next - 1e-8:
#         ctrl_mode, ctrl_cmd = ctrl_mgr.get_ctrl(rbs_states)
#         torque_array[:] = ctrl_cmd  # apply the hold
#         nm_controller.log_data()
#         t_ctrl_next += ctrl_dt

#     # log data only at sim_dt intervals
#     if step_i % log_interval == 0:
#         rbs.log_data()
#         swing_phase_log_1.append(1 if rbs_states['leg_1'].get('phase') == 'swing' else 0)
#         swing_phase_log_2.append(1 if rbs_states['leg_2'].get('phase') == 'swing' else 0)
#         grf_log_1.append(rbs_states['leg_1']['grf'])
#         grf_log_2.append(rbs_states['leg_2']['grf'])

#     # print progress every 0.5 simulated seconds
#     if step_i % int(0.5 / dt) == 0 and step_i > 0:
#         print(f"Simulated {step_i * dt:.2f} seconds...")

# print(f"[DONE] Simulated time: {tStop:.2f}s | Wall time for integration: {time.perf_counter() - t_wall_start:.2f}s")

# ################### joint angle plots ####################

# # Build time array for logged data (logged at sim_dt intervals)
# N_logged_joints = len(rbs.joint_list[0].q_list)
# t_logged = np.arange(N_logged_joints) * sim_dt

# # Helper function to shade swing phase dynamically
# def shade_swing(ax, t_logged, swing_log, color):
#     if len(swing_log) > 0:
#         log_len = min(len(t_logged), len(swing_log))
#         ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_log[:log_len])==1, 
#                         transform=ax.get_xaxis_transform(), alpha=0.2, color=color, label='Swing')

# t_ctrl = np.arange(nm_controller.num_ctrl_steps) * ctrl_dt

# # ==========================================
# # NEW PLOTS: HIP, KNEE, ANKLE
# # ==========================================

# # --- 1. HIP PLOT --- (4 rows, 2 columns)
# fig_hip, axes_hip = plt.subplots(4, 2, sharex=True, figsize=(12, 10))
# fig_hip.suptitle('Hip Kinematics & Muscle Activations', fontsize=16)

# # Leg 1 (Red)
# axes_hip[0, 0].plot(t_logged, np.degrees(rbs.joint_list[0].q_list), color='red')
# axes_hip[0, 0].set_ylabel('Angle [deg]')
# axes_hip[0, 0].set_title('Leg 1 (Hip)')

# axes_hip[1, 0].plot(t_logged, rbs.joint_list[0].tau_list, color='red')
# axes_hip[1, 0].set_ylabel('Torque [Nm]')

# axes_hip[2, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['HFL']['A'][:len(t_ctrl)], color='red')
# axes_hip[2, 0].set_ylabel('HFL Act')

# axes_hip[3, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['GLU']['A'][:len(t_ctrl)], color='red')
# axes_hip[3, 0].set_ylabel('GLU Act')
# axes_hip[3, 0].set_xlabel('Time [s]')

# for i in range(4):
#     axes_hip[i, 0].grid(True, alpha=0.3)
#     shade_swing(axes_hip[i, 0], t_logged, swing_phase_log_1, 'red')

# # Leg 2 (Blue)
# axes_hip[0, 1].plot(t_logged, np.degrees(rbs.joint_list[3].q_list), color='blue')
# axes_hip[0, 1].set_title('Leg 2 (Hip)')

# axes_hip[1, 1].plot(t_logged, rbs.joint_list[3].tau_list, color='blue')

# axes_hip[2, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['HFL']['A'][:len(t_ctrl)], color='blue')

# axes_hip[3, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['GLU']['A'][:len(t_ctrl)], color='blue')
# axes_hip[3, 1].set_xlabel('Time [s]')

# for i in range(4):
#     axes_hip[i, 1].grid(True, alpha=0.3)
#     shade_swing(axes_hip[i, 1], t_logged, swing_phase_log_2, 'blue')

# fig_hip.tight_layout()

# # --- 2. KNEE PLOT --- (4 rows, 2 columns)
# fig_knee, axes_knee = plt.subplots(4, 2, sharex=True, figsize=(12, 10))
# fig_knee.suptitle('Knee Kinematics & Muscle Activations', fontsize=16)

# # Leg 1 (Red)
# axes_knee[0, 0].plot(t_logged, np.degrees(rbs.joint_list[1].q_list), color='red')
# axes_knee[0, 0].set_ylabel('Angle [deg]')
# axes_knee[0, 0].set_title('Leg 1 (Knee)')

# axes_knee[1, 0].plot(t_logged, rbs.joint_list[1].tau_list, color='red')
# axes_knee[1, 0].set_ylabel('Torque [Nm]')

# axes_knee[2, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['VAS']['A'][:len(t_ctrl)], color='red')
# axes_knee[2, 0].set_ylabel('VAS Act')

# axes_knee[3, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['HAM']['A'][:len(t_ctrl)], color='red')
# axes_knee[3, 0].set_ylabel('HAM Act')
# axes_knee[3, 0].set_xlabel('Time [s]')

# for i in range(4):
#     axes_knee[i, 0].grid(True, alpha=0.3)
#     shade_swing(axes_knee[i, 0], t_logged, swing_phase_log_1, 'red')

# # Leg 2 (Blue)
# axes_knee[0, 1].plot(t_logged, np.degrees(rbs.joint_list[4].q_list), color='blue')
# axes_knee[0, 1].set_title('Leg 2 (Knee)')

# axes_knee[1, 1].plot(t_logged, rbs.joint_list[4].tau_list, color='blue')

# axes_knee[2, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['VAS']['A'][:len(t_ctrl)], color='blue')

# axes_knee[3, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['HAM']['A'][:len(t_ctrl)], color='blue')
# axes_knee[3, 1].set_xlabel('Time [s]')

# for i in range(4):
#     axes_knee[i, 1].grid(True, alpha=0.3)
#     shade_swing(axes_knee[i, 1], t_logged, swing_phase_log_2, 'blue')

# fig_knee.tight_layout()

# # --- 3. ANKLE PLOT --- (5 rows, 2 columns)
# fig_ankle, axes_ankle = plt.subplots(5, 2, sharex=True, figsize=(12, 12))
# fig_ankle.suptitle('Ankle Kinematics & Muscle Activations', fontsize=16)

# # Leg 1 (Red)
# axes_ankle[0, 0].plot(t_logged, np.degrees(rbs.joint_list[2].q_list), color='red')
# axes_ankle[0, 0].set_ylabel('Angle [deg]')
# axes_ankle[0, 0].set_title('Leg 1 (Ankle)')

# axes_ankle[1, 0].plot(t_logged, rbs.joint_list[2].tau_list, color='red')
# axes_ankle[1, 0].set_ylabel('Torque [Nm]')

# axes_ankle[2, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['GAS']['A'][:len(t_ctrl)], color='red')
# axes_ankle[2, 0].set_ylabel('GAS Act')

# axes_ankle[3, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['TA']['A'][:len(t_ctrl)], color='red')
# axes_ankle[3, 0].set_ylabel('TA Act')

# axes_ankle[4, 0].plot(t_ctrl, nm_controller.log_data_dict['leg_1']['SOL']['A'][:len(t_ctrl)], color='red')
# axes_ankle[4, 0].set_ylabel('SOL Act')
# axes_ankle[4, 0].set_xlabel('Time [s]')

# for i in range(5):
#     axes_ankle[i, 0].grid(True, alpha=0.3)
#     shade_swing(axes_ankle[i, 0], t_logged, swing_phase_log_1, 'red')

# # Leg 2 (Blue)
# axes_ankle[0, 1].plot(t_logged, np.degrees(rbs.joint_list[5].q_list), color='blue')
# axes_ankle[0, 1].set_title('Leg 2 (Ankle)')

# axes_ankle[1, 1].plot(t_logged, rbs.joint_list[5].tau_list, color='blue')

# axes_ankle[2, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['GAS']['A'][:len(t_ctrl)], color='blue')

# axes_ankle[3, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['TA']['A'][:len(t_ctrl)], color='blue')

# axes_ankle[4, 1].plot(t_ctrl, nm_controller.log_data_dict['leg_2']['SOL']['A'][:len(t_ctrl)], color='blue')
# axes_ankle[4, 1].set_xlabel('Time [s]')

# for i in range(5):
#     axes_ankle[i, 1].grid(True, alpha=0.3)
#     shade_swing(axes_ankle[i, 1], t_logged, swing_phase_log_2, 'blue')

# fig_ankle.tight_layout()

# # --- GRF Plot ---
# fig_grf, ax_grf = plt.subplots(figsize=(10, 4))
# fig_grf.suptitle('Ground Reaction Forces', fontsize=14)

# log_len = min(len(t_logged), len(grf_log_1))
# ax_grf.plot(t_logged[:log_len], grf_log_1[:log_len], label='Leg 1 GRF', color='red', linewidth=1.5)
# ax_grf.plot(t_logged[:log_len], grf_log_2[:log_len], label='Leg 2 GRF', color='blue', linewidth=1.5)

# ax_grf.set_ylabel('Force [N]')
# ax_grf.set_xlabel('Time [s]')
# ax_grf.grid(True, alpha=0.3)
# ax_grf.legend(loc='upper right')
# fig_grf.tight_layout()

# # --- Torso Angle Plot ---
# fig_torso, ax_torso = plt.subplots(figsize=(8, 4))
# fig_torso.suptitle('Torso (Trunk) Angle', fontsize=14)

# log_len_torso = min(len(t_logged), len(rbs.body_list[0].p_list))
# ax_torso.plot(t_logged[:log_len_torso], np.degrees(rbs.body_list[0].p_list[:log_len_torso]), color='purple', linewidth=1.5, label='Torso Angle')

# ax_torso.set_ylabel('Angle [deg]')
# ax_torso.set_xlabel('Time [s]')
# ax_torso.grid(True, alpha=0.3)
# ax_torso.legend(loc='upper right')
# fig_torso.tight_layout()

# # plt.show(block=False)

# # ============================================================
# # AESTHETIC SUMMARY PLOTS: JOINT ANGLES AND JOINT TORQUES
# # ============================================================

# plt.rcParams.update({
#     "font.size": 16,
#     "axes.titlesize": 20,
#     "axes.labelsize": 18,
#     "xtick.labelsize": 15,
#     "ytick.labelsize": 15,
#     "legend.fontsize": 15,
#     "figure.titlesize": 24,
#     "lines.linewidth": 2.5,
# })

# joint_names = ["Hip", "Knee", "Ankle"]

# leg_1_joints = rbs.joint_list[0:3]
# leg_2_joints = rbs.joint_list[3:6]

# # -----------------------------
# # JOINT ANGLES FIGURE
# # -----------------------------
# fig_angles, axes_angles = plt.subplots(3, 1, sharex=True, figsize=(13, 9))
# fig_angles.suptitle("Hip, Knee, and Ankle Joint Angles vs Time", fontweight="bold")

# for i, ax in enumerate(axes_angles):
#     q1 = np.degrees(leg_1_joints[i].q_list)
#     q2 = np.degrees(leg_2_joints[i].q_list)

#     n1 = min(len(t_logged), len(q1))
#     n2 = min(len(t_logged), len(q2))

#     ax.plot(t_logged[:n1], q1[:n1], label=f"Leg 1 {joint_names[i]}")
#     ax.plot(t_logged[:n2], q2[:n2], linestyle="--", label=f"Leg 2 {joint_names[i]}")

#     ax.set_ylabel(f"{joint_names[i]}\nAngle [deg]")
#     ax.grid(True, alpha=0.35)
#     ax.legend(loc="best", frameon=True)

# axes_angles[-1].set_xlabel("Time [s]")
# fig_angles.tight_layout(rect=[0, 0, 1, 0.95])


# # -----------------------------
# # JOINT TORQUES FIGURE
# # -----------------------------
# fig_torques, axes_torques = plt.subplots(3, 1, sharex=True, figsize=(13, 9))
# fig_torques.suptitle("Hip, Knee, and Ankle Joint Torques vs Time", fontweight="bold")

# for i, ax in enumerate(axes_torques):
#     tau1 = leg_1_joints[i].tau_list
#     tau2 = leg_2_joints[i].tau_list

#     n1 = min(len(t_logged), len(tau1))
#     n2 = min(len(t_logged), len(tau2))

#     ax.plot(t_logged[:n1], tau1[:n1], label=f"Leg 1 {joint_names[i]}")
#     ax.plot(t_logged[:n2], tau2[:n2], linestyle="--", label=f"Leg 2 {joint_names[i]}")

#     ax.set_ylabel(f"{joint_names[i]}\nTorque [Nm]")
#     ax.grid(True, alpha=0.35)
#     ax.legend(loc="best", frameon=True)

# axes_torques[-1].set_xlabel("Time [s]")
# fig_torques.tight_layout(rect=[0, 0, 1, 0.95])

# ################### plotting output ####################

# plot_start = time.perf_counter()

# # Visualization settings: achieve ~60 FPS visualization rate for smooth playback
# render_fps = 60
# vis_dt = 1.0 / render_fps

# # Logged data is at sim_dt resolution, not dt resolution
# N_logged = len(rbs.body_list[0].x_list)
# ratio = 8   # animation speed multiplier
# # Determine step to subsample logged frames to achieve render_fps playback
# step = max(1, int(vis_dt / (ratio * sim_dt)))

# indices = np.arange(0, N_logged, step, dtype=int)

# viz = WalkVisualizer(rigid_body_system=rbs)

# # Build timestamped save path in 'videos' folder
# import os
# from datetime import datetime
# vidros_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')
# timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# save_path = os.path.join(vidros_dir, f'walk_{timestamp}.mp4')

# # interval is in ms per frame (e.g. 16ms for ~60 FPS)
# viz.animate(indices, interval=int(1000/render_fps), save_path=save_path)

# print(f"[DONE] Simulated time: {tStop:.2f}s | Wall time for plotting: {time.perf_counter() - plot_start:.2f}s")


# ==========================================
# OLD PLOTS (COMMENTED OUT)
# ==========================================

# # --- Leg 1: hip_1, knee_1, ankle_1 (joint indices 0, 1, 2) ---
# fig1, axes1 = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
# fig1.suptitle('Leg 1 – Joint Angles', fontsize=14)
# 
# for ax, joint in zip(axes1, rbs.joint_list[0:3]):
#     ax.plot(t_logged, np.degrees(joint.q_list), linewidth=0.8)
#     ax.set_ylabel(f'{joint.name} [deg]')
#     ax.grid(True, alpha=0.3)
#     # Add swing phase highlights
#     if len(swing_phase_log_1) > 0:
#         log_len = min(len(t_logged), len(swing_phase_log_1))
#         ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_phase_log_1[:log_len])==1, 
#                         transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# 
# axes1[0].legend(loc='upper right')
# axes1[-1].set_xlabel('Time [s]')
# fig1.tight_layout()
# 
# # --- Leg 2: hip_2, knee_2, ankle_2 (joint indices 3, 4, 5) ---
# fig2, axes2 = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
# fig2.suptitle('Leg 2 – Joint Angles', fontsize=14)
# 
# for ax, joint in zip(axes2, rbs.joint_list[3:6]):
#     ax.plot(t_logged, np.degrees(joint.q_list), linewidth=0.8)
#     ax.set_ylabel(f'{joint.name} [deg]')
#     ax.grid(True, alpha=0.3)
#     # Add swing phase highlights
#     if len(swing_phase_log_2) > 0:
#         log_len = min(len(t_logged), len(swing_phase_log_2))
#         ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_phase_log_2[:log_len])==1, 
#                         transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# 
# axes2[0].legend(loc='upper right')
# axes2[-1].set_xlabel('Time [s]')
# fig2.tight_layout()
# 
# # --- Leg 1: hip_1, knee_1, ankle_1 (joint indices 0, 1, 2) Joint Torques ---
# fig3, axes3 = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
# fig3.suptitle('Leg 1 – Joint Torques', fontsize=14)
# 
# for ax, joint in zip(axes3, rbs.joint_list[0:3]):
#     ax.plot(t_logged, joint.tau_list, linewidth=0.8)
#     ax.set_ylabel(f'{joint.name} [Nm]')
#     ax.grid(True, alpha=0.3)
#     if len(swing_phase_log_1) > 0:
#         log_len = min(len(t_logged), len(swing_phase_log_1))
#         ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_phase_log_1[:log_len])==1, 
#                         transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# 
# axes3[0].legend(loc='upper right')
# axes3[-1].set_xlabel('Time [s]')
# fig3.tight_layout()
# 
# # --- Leg 2: hip_2, knee_2, ankle_2 (joint indices 3, 4, 5) Joint Torques ---
# fig4, axes4 = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
# fig4.suptitle('Leg 2 – Joint Torques', fontsize=14)
# 
# for ax, joint in zip(axes4, rbs.joint_list[3:6]):
#     ax.plot(t_logged, joint.tau_list, linewidth=0.8)
#     ax.set_ylabel(f'{joint.name} [Nm]')
#     ax.grid(True, alpha=0.3)
#     if len(swing_phase_log_2) > 0:
#         log_len = min(len(t_logged), len(swing_phase_log_2))
#         ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_phase_log_2[:log_len])==1, 
#                         transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# 
# axes4[0].legend(loc='upper right')
# axes4[-1].set_xlabel('Time [s]')
# fig4.tight_layout()
# 
# # --- MTC States and Lever Arms Plots ---
# # t_ctrl = np.arange(nm_controller.num_ctrl_steps) * ctrl_dt
# 
# # mtc_names = ["VAS", "SOL", "GAS", "TA", "HAM", "GLU", "HFL"]
# # mtc_state_names = ["A", "F_mtc", "l_mtc", "l_ce"]
# # ms_states = ["hip_lever", "knee_lever", "ankle_lever"]
# 
# # for leg in ['leg_1', 'leg_2']:
# #     swing_log = swing_phase_log_1 if leg == 'leg_1' else swing_phase_log_2
# #     for m in mtc_names:
# #         # MTC states figure
# #         fig_mtc, axes_mtc = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
# #         fig_mtc.suptitle(f'{m} {leg} - MTC States', fontsize=14)
# #         for ax, state in zip(axes_mtc, mtc_state_names):
# #             data = nm_controller.log_data_dict[leg][m][state]
# #             ax.plot(t_ctrl[:len(data)], data, linewidth=1.0)
# #             ax.set_ylabel(state)
# #             ax.grid(True, alpha=0.3)
# #             if len(swing_log) > 0:
# #                 log_len = min(len(t_logged), len(swing_log))
# #                 ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_log[:log_len])==1, 
# #                                 transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# #         axes_mtc[0].legend(loc='upper right')
# #         axes_mtc[-1].set_xlabel('Time [s]')
# #         fig_mtc.tight_layout()
#     
# #         # MS states figure
# #         fig_ms, axes_ms = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
# #         fig_ms.suptitle(f'{m} {leg} - Lever Arms', fontsize=14)
# #         for ax, state in zip(axes_ms, ms_states):
# #             data = nm_controller.log_data_dict[leg][m][state]
# #             ax.plot(t_ctrl[:len(data)], data, linewidth=1.0)
# #             ax.set_ylabel(state)
# #             ax.grid(True, alpha=0.3)
# #             if len(swing_log) > 0:
# #                 log_len = min(len(t_logged), len(swing_log))
# #                 ax.fill_between(t_logged[:log_len], 0, 1, where=np.array(swing_log[:log_len])==1, 
# #                                 transform=ax.get_xaxis_transform(), alpha=0.2, color='red', label='Swing')
# #         axes_ms[-1].set_xlabel('Time [s]')
# #         fig_ms.tight_layout()