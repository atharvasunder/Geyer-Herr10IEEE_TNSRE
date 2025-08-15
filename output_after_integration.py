import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication


class PyqtgraphAnimation:
    def __init__(self, time_step, rigid_body_system, ratio=1):
        self.time_step = time_step
        self.ratio = ratio
        self.rbs = rigid_body_system

        # Initialize Qt app
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        # Create window and plot
        self.win = pg.GraphicsLayoutWidget(title="Rigid Body Animation")
        self.plot = self.win.addPlot(title="X-Z View")
        self.plot.setAspectLocked(True)
        self.plot.setXRange(-1, 1)
        self.plot.setYRange(-2, 2)
        self.plot.showGrid(x=True, y=True)

        # Add ground line
        self.plot.plot([-1000, 1000], [0, 0], pen=pg.mkPen((100, 100, 100), width=1))

        # Create lines for each body
        self.body_lines = []
        for _ in self.rbs.body_list:
            line = self.plot.plot([], [], pen=pg.mkPen('r', width=3))
            self.body_lines.append(line)

        self.win.show()
        print("pyqtgraph window initialized")

    def update_animation(self, idx):
        for j, body in enumerate(self.rbs.body_list):
            x_w, z_w = body.update_body_geometry(
                body.x_list[idx], body.z_list[idx], body.p_list[idx]
            )
            self.body_lines[j].setData(x_w, z_w)

        # Process Qt events to update plot
        self.app.processEvents()
        time.sleep(0.0001)  # very small sleep to mimic pause

    def muscle_torque_contributions(self, t):

        fig, axs = plt.subplots(3, 2)
        
        fig.suptitle("Torque contributions of system of muscles on each joint")

        muscle_list = self.rbs.muscle_list
        torque_contributions_list = []

        for i in range(len(muscle_list)):
            torque_contributions_list.append(muscle_list[i].torque_contributions)

        print(torque_contributions_list[1][10000])
            
        # Sum element-wise across the lists
        summed_list = [sum(arrays) for arrays in zip(*torque_contributions_list)]

        # Stack into a 2D array of shape (T, 6)
        torque_matrix = np.vstack(summed_list)  # shape: (timesteps, 6)

        # Now split into 6 arrays (one per joint) containing the torque contributions of muscles on all joints for all times
        joint_torque_muscle = [torque_matrix[:, i] for i in range(6)]

        # hip torque
        axs[0,0].plot(t, joint_torque_muscle[0], 'k', lw=2)
        axs[0,0].set(ylabel='hip 1 torque (Nm)')

        axs[0,1].plot(t, joint_torque_muscle[1], 'k', lw=2)
        axs[0,1].set(ylabel='hip 2 torque (Nm)')

        # knee torque
        axs[1,0].plot(t, joint_torque_muscle[2], 'k', lw=2)
        axs[1,0].set(ylabel='knee 1 torque (Nm)')

        axs[1,1].plot(t, joint_torque_muscle[3], 'k', lw=2)
        axs[1,1].set(ylabel='knee 2 torque (Nm)')

        axs[2,0].plot(t, joint_torque_muscle[4], 'k', lw=2)
        axs[2,0].set(ylabel='ankle 1 torque (Nm)')

        axs[2,1].plot(t, joint_torque_muscle[5], 'k', lw=2)
        axs[2,1].set(ylabel='ankle 2 torque (Nm)')

        plt.show()








            

# ######################################################################

# def plot_xzp(body, t, fig_num, title):
#     fig, axs = plt.subplots(3, 1, num=fig_num)  # Use fig_num for figure identification

#     x = body.x_list
#     z = body.z_list
#     p = body.p_list

#     # Plot trajectory of point mass in space
#     axs[0].plot(t, x, linewidth=2)
#     axs[0].set(xlabel='time (s)', ylabel='x (m)')

#     axs[1].plot(t, z, linewidth=2)
#     axs[1].set(xlabel='time (s)', ylabel='z (m)')

#     # Plot orientation in degrees
#     p_deg = [angle * 180 / math.pi for angle in p]
#     axs[2].plot(t, p_deg, linewidth=2)
#     axs[2].set(xlabel='time (s)', ylabel='pitch (deg)')

#     # Set the overall title for the figure
#     fig.suptitle(title, fontsize=14, fontweight='bold')

#     fig.show()

# ######################################################################
# def plot_grf(t, grf_x, grf_z, slide_flag):
#     fig = plt.figure(3)
#     fig.clear()
#     # fig.set_size_inches(6, 4)

#     ax = plt.axes(xlabel='t (s)', ylabel='GRF (N)')
#     ax.plot(t, grf_x, 'k', lw=2)
#     ax.plot(t, grf_z, 'r', lw=2)
#     slide_flag_100 = [flag * 100 for flag in slide_flag]
#     ax.plot(t, slide_flag_100, 'b', lw=1)

#     ax.set(ylim=(-200, 2000))

#     # fig.tight_layout()
#     plt.show()


# def plot_joint_angles(t, qa, qk, qh):
#     fig = plt.figure(4)

#     fig.clear()
#     # fig.set_size_inches(6, 6)
#     _, axs = plt.subplots(3, 1, num=4)

#     # hip angle
#     qh_deg = [angle * 180 / math.pi for angle in qh]
#     axs[0].plot(t, qh_deg, 'k', lw=2)
#     axs[0].set(ylabel='hip angle (deg)')

#     qk_deg = [angle * 180 / math.pi for angle in qk]
#     axs[1].plot(t, qk_deg, 'k', lw=2)
#     axs[1].set(ylabel='knee angle (deg)')

#     qa_deg = [angle * 180 / math.pi for angle in qa]
#     axs[2].plot(t, qa_deg, 'k', lw=2)
#     axs[2].set(xlabel='time (s)', ylabel='ankle angle (deg)')

#     plt.show()

# def plot_activations(t, A_list):

#     # Create the plot
#     fig = plt.figure(6)
    
#     plt.plot(t, A_list,  linewidth=2)

#     # Labels and title
#     plt.xlabel('time')
#     plt.ylabel('Activation')
#     plt.title('A vs t')
#     plt.grid()

#     # Show the plot
#     plt.show()

# def plot_F_mtc(t, F_mtc_list):

#     # Create the plot
#     fig = plt.figure(7)
    
#     plt.plot(t, F_mtc_list,  linewidth=2)

#     # Labels and title
#     plt.xlabel('time')
#     plt.ylabel('F_mtc')
#     plt.title('F_mtc vs t')
#     plt.grid()

#     # Show the plot
#     plt.show()

# def plot_f_l(t, f_l_list):

#     # Create the plot
#     fig = plt.figure(8)
    
#     plt.plot(t, f_l_list,  linewidth=2)

#     # Labels and title
#     plt.xlabel('time')
#     plt.ylabel('f_l')
#     plt.title('f_l vs t')
#     plt.grid()

#     # Show the plot
#     plt.show()

# def plot_f_v(t, f_v_list):

#     # Create the plot
#     fig = plt.figure(9)
    
#     plt.plot(t, f_v_list,  linewidth=2)

#     # Labels and title
#     plt.xlabel('time')
#     plt.ylabel('f_v')
#     plt.title('f_v vs t')
#     plt.grid()

#     # Show the plot
#     plt.show()