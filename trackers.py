import matplotlib.pyplot as plt
import numpy as np

class ContactTracker():
    """
    Contact point tracking class. Instances record key information
    about a contact point such as ground reaction forces and contact
    status!!!!!!

    The tracker class provides methods for initiating the tracking,
    collect data, and plot the results.
    """

    def __init__(self, contact_point):
        self.contact_point = contact_point

        self.fx = [contact_point.base.fx]
        self.fz = [contact_point.base.fz]

        self.contact_flag = [contact_point.contact]
        self.slide_flag = [contact_point.sliding_mode]

        self.dist_x = [contact_point.base_x - contact_point.x_stick]
        self.base_z = [contact_point.base_z]
        self.ground_height = [contact_point.ground_height]

    def get_data(self): # to retrieve grf's
        return np.array([self.fx[-1], self.fz[-1]])

    # append data
    def append(self):
        self.fx.append(self.contact_point.base.fx)
        self.fz.append(self.contact_point.base.fz)
        self.contact_flag.append(self.contact_point.contact)
        self.slide_flag.append(self.contact_point.sliding_mode)
        self.dist_x.append(self.contact_point.base_x - self.contact_point.x_stick)
        self.base_z.append(self.contact_point.base_z)
        self.ground_height.append(self.contact_point.ground_height)

    def plot(self, t, fig_num, title):
        fig = plt.figure(fig_num)
        fig.clear()
        # fig.set_size_inches(6, 12)

        _, axs = plt.subplots(7, 1, sharex=True, height_ratios=[3, 1, 2, 2, 1, 2, 2], num=fig_num) # originally 7,1
        # _, axs = plt.subplots(1, 1, num=fig_num) # originally 7,1
        # grf plot x and y
        # axs.set(ylabel='GRF (N)', xlabel='time (s)', xlim=(t[0], t[-1]))
        # axs.plot(t, self.fx, 'k', lw=2)
        # axs.plot(t, self.fz, 'r', lw=2)
        
        # base_z = [self.base_z[i] for i in range(len(t))]
        # axs.set(ylabel='z contact_site', xlabel='time (s)')
        # axs.plot(t, base_z, lw=2)
        # axs.plot(t, self.ground_height, 'k', lw=1)
        
        axs[0].set(ylabel='GRF (N)', xlim=(t[0], t[-1]), ylim=(-1500, 5000))
        axs[0].plot(t, self.fx, 'k', lw=2)
        axs[0].plot(t, self.fz, 'r', lw=2)

        t_cntct = [t[i] for i in range(len(t)) if self.contact_flag[i] == 1]
        
        slide_flag = [self.slide_flag[i] for i in range(len(t)) if self.contact_flag[i] == 1]
        axs[1].set(ylabel='stiction / sliding', ylim=(-0.05, 1.05))
        axs[1].plot(t_cntct, slide_flag, 'b', lw=2)

        dist_x = [self.dist_x[i] for i in range(len(t)) if self.contact_flag[i] == 1]
        axs[2].set(ylabel='x position')
        axs[2].plot(t_cntct, dist_x, 'k', lw=2)

        deriv_x = [(dist_x[i] - dist_x[i - 1]) / (t_cntct[i] - t_cntct[i - 1]) for i in range(1, len(t_cntct))]
        axs[3].set(ylabel='x velocity')
        axs[3].plot(t_cntct[1:], deriv_x, 'k', lw=2)

        axs[4].set(ylabel='contact flag', ylim=(-0.05, 1.05))
        axs[4].plot(t, self.contact_flag, 'k', lw=2)

        base_z = [self.base_z[i] for i in range(len(t)) if self.contact_flag[i] == 1]
        axs[5].set(ylabel='z position', xlabel='time (s)')
        axs[5].plot(t_cntct, base_z, 'k', lw=2)
        axs[5].plot(t, self.ground_height, 'k', lw=1)

        deriv_z = [(base_z[i] - base_z[i - 1]) / (t_cntct[i] - t_cntct[i - 1]) for i in range(1, len(t_cntct))]
        axs[6].set(ylabel='z velocity')
        axs[6].plot(t_cntct[1:], deriv_z, 'k', lw=2)
        axs[6].plot((t[0], t[-1]), (self.contact_point.max_vz, self.contact_point.max_vz), 'r', lw=1)
        
        # Set the overall title for the figure
        fig.suptitle(title, fontsize=14, fontweight='bold')

        fig.tight_layout()
        plt.show()
