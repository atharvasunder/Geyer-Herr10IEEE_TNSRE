"""
Class for muscle tendon complex. Contains functions to compute the
force exerted by the muscle tendon complex (MTC), with inputs as stimulation,
and MTC length. Also updates length of internal degree of freedom of the
muscle (the contractile element) based on the hill type muscle model mechanics.

Written by Atharva Sunder, Russell Xing, Bill Ma
Advised by Prof. Hartmut Geyer
"""

import math
import numpy as np
import config

class muscle_tendon_unit():
    """ 

    Inputs:

    muscle_params:
    - MTC properties
    - Force-length relationship params
    - Force-velocity relationship params
    - Tendon properties
    - timestep of integration for internal dof resolution
    
    class contains the function:
    update- takes brain stimulation and l_mtc as input, outputs F_mtc

    """

    def __init__(self, muscle_name, state_names, state_units, dt_mtc):
        """
        All inputs should be in string format
        """

        self.muscle_name = muscle_name  # muscle name

        # mtc states, useful for logging data
        self.states = np.nan * np.zeros(len(state_names), dtype=np.float32)
        self.state_names = state_names
        self.state_units = state_units
        
        # constants parameters
        self.muscle_params = config.MUSCLE_DICTIONARY  # dictionary for all the muscle constants

        # common constants to all muscle instances
        self.dt_mtc = dt_mtc
        self.w = self.muscle_params["GENERAL"]["w"]
        self.c = self.muscle_params["GENERAL"]["c"]  # remaining forces at +/- w
        self.log_c = math.log(self.c)
        self.K = self.muscle_params["GENERAL"]["K"]
        self.N = self.muscle_params["GENERAL"]["N"]
        self.tau = self.muscle_params["GENERAL"]["tau"]
        self.ns_params = config.NERVOUS_SYSTEM_DICTIONARY  # nervous system dictionary for prestim values

        # constants unique to the muscle instance
        self.F_max_iso = None   # [N] maximum force can generate when it is contracting isometrically (constant length)
        self.v_max = None  # [1/s] maximum contraction velocity, will be multiplied with l_opt later
        self.l_slack = None  # [m] slack length
        self.l_opt = None  # [m] optimal fiber length
        self.l_ce_norm = None
        self.l_se_norm = None
        self.e_ref_see = None
        self.pre_stim = None

        # inner mtc updated values
        self.l_ce = None  # [m] length of the contractile element
        self.l_se = None  # [m] length of the series elastic element
        self.v_ce = None # [m/s] velocity of the contractile element
        
        # output values, external inputs
        self.F_mtc = None
        self.stim = None  # stimulation amplitude range [0.01 - 1]
        self.l_mtc = None  # [m]

        # initial conditions
        self.A = 0.01  # temporary, will be overwritten below by find_muscle

        self.find_muscle()  # deliver all the constants

        self.A = self.pre_stim  # Proper activation using stim0

    def find_muscle(self):
        """
        Based on input muscle, store muscle-unique constants
        """
        self.F_max_iso = self.muscle_params[self.muscle_name]['F_max']
        self.pre_stim = self.ns_params[self.muscle_name]['PRESTIM']
        self.stim = self.pre_stim
        self.v_max = self.muscle_params[self.muscle_name]['v_max']
        self.l_slack = self.muscle_params[self.muscle_name]['l_slack']
        self.l_opt = self.muscle_params[self.muscle_name]['l_opt']
        self.e_ref_see = self.muscle_params[self.muscle_name]['e_ref_see']

    def update(self, stim, l_mtc):

        self.l_mtc = l_mtc
        
        sub_steps = 20  # control process running at 100 Hz, so 10 sub steps makes effective dt_mtc be at 1000 Hz
        dt_sub = self.dt_mtc / sub_steps  # dt is the time step for the sub steps

        for i in range(sub_steps):
            """
            take 20 small steps instead of 1 big step to avoid numerical instability due to euler integration
            """

            # Forward Euler method to compute contractile element length
            self.l_ce += dt_sub * self.v_ce

            # Get force length relations
            self.l_ce_norm = self.l_ce / self.l_opt
            f_be = self.be_force_length()
            f_pe = self.pe_force_length()
            f_ce = self.ce_force_length()

            # Compute F_mtc (normalized to F_max)
            self.l_se = self.l_mtc - self.l_ce
            self.l_se_norm = self.l_se / self.l_slack
            f_se = self.se_force_length()

            # compute muscle activation
            self.A = np.clip(self.A + dt_sub * self.activation_dynamics(self.A, stim, self.tau), 0, 1) 

            # Compute force velocity relationship for inner muscle updates
            f_v = (f_se + f_be) / (f_pe + f_ce * self.A + 1e-8)  # f_v = f_se / (A*f_ce)
            
            v_ce_norm = self.f_v_inverse(f_v)

            self.v_ce = v_ce_norm * self.v_max * self.l_opt

        self.stim = stim
        self.F_mtc = self.F_max_iso * f_se

        # update mtc states array for data logging
        self.set_states()

        return self.F_mtc, self.l_ce

    def reset(self, l_mtc):
        """
        Reset all the muscle velocity and length
        """
        self.A = self.pre_stim  # activation level restored to prestim
        self.stim = self.pre_stim
        self.v_ce = 0.0
        self.l_mtc = l_mtc  # defined based on the latest state of the mtc (makes sense if this is called right before stance starts to get an accurate l_ce) [m]
        self.l_ce = l_mtc - self.l_slack

        return self.l_ce

    def set_states(self):
        self.states[:] = (self.A, self.F_mtc, self.l_mtc, self.l_ce)
    
    def se_force_length(self):
        """
        SE force-length relation
        """

        relative_strain = (self.l_se_norm-1) / self.e_ref_see
        
        if self.l_se_norm > 1:
            f_se = relative_strain**2
        else:
            f_se = 0.0

        return f_se

    def be_force_length(self):
        """
        Buffer Elasticity BE
        """

        be_input = self.l_ce_norm-1+self.w
        relative_strain = (be_input)/(self.w/2)
        
        if be_input < 0: 
            f_be = relative_strain**2
        else:
            f_be = 0.0

        return f_be

    def pe_force_length(self):
        """
        Parallel Elasticity PE
        """
        relative_strain = (self.l_ce_norm - 1)/self.w

        if self.l_ce_norm > 1:
            f_pe = relative_strain**2
        else:
            f_pe = 0.0
        
        return f_pe

    def ce_force_length(self):
        """
        CE Force-Length Relationship
        """
        
        pos = abs(self.l_ce_norm-1)/self.w
        f_ce = math.exp((math.pow(pos, 3) * self.log_c))
        
        return f_ce
    
    def activation_dynamics(self, A, stim, tau):

        return (1/tau)*(stim - A)  # equals dA/dt

    def f_v_inverse(self, f_v):
        """
        Inverse of Force-Velocity Relationship
        """
        # concentric branch
        if f_v <= 1:
            con_branch = (f_v-1)/(f_v*self.K+1)
        else:
            con_branch = 0.0

        # eccentric branch
        if (f_v > 1) and (f_v <= self.N):
            step_1 = (f_v-self.N)/(self.N-1)
            exc_branch = (1 + step_1)/(1 - step_1 * 7.56 * self.K)
        else:
            exc_branch = 0.0

        # eccentric overshoot
        if f_v > self.N:
            exc_overshoot = ((f_v - self.N) * 0.01 + 1) * self.N
        else:
            exc_overshoot = 0.0

        # if f_v > 1:
        #     exc = exc_branch + exc_overshoot
        # else:
        #     exc = 0.0

        exc = exc_branch + exc_overshoot + con_branch

        v_ce_norm = exc 
        
        return v_ce_norm

    # @ property
    # def l_slack(self):
    #     """get l_slack"""
    #     return self.l_slack

    # @ property
    # def l_opt(self):
    #     """get l_opt"""
    #     return self.l_opt

    # @ property
    # def l_ce(self):
    #     """get l_opt"""
    #     return self.l_ce

    # @ property
    # def fmax_iso(self):
    #     """get F MAX"""
    #     return self.F_max_iso

    # @ property
    # def pre_stim(self):
    #     """get pre situmlation"""
    #     return self.pre_stim



# if __name__ == "__main__":
#     hill_mtc = muscle_tendon_unit()
