import math
import numpy as np

class muscle_tendon_unit():
    """
    Class describing a muscle connected between two joints. 

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

    # constructor
    def __init__(self, name, muscle_params,dt):
        self.name = name

        self.muscle_params = muscle_params
        self.dt = dt    # dt for muscle internal dof integration and activation dynamics integration

        ################ initialize muscle params ##################

        # Tendon properties
        self.l_rest = muscle_params.l_rest
        self.e_ref_see = muscle_params.e_ref_see

        # Force-length relationship
        self.F_max = muscle_params.F_max
        self.l_opt = muscle_params.l_opt
        self.c = muscle_params.c
        self.w = muscle_params.w

        # Force-velocity relationship
        self.K = muscle_params.K
        self.N = muscle_params.N
        self.v_max = muscle_params.v_max
        
        # initialize tendon length to rest length (always done!)
        self.l_se = self.l_rest
        self.l_ref_mtc = muscle_params.l_ref_mtc # MTC reference length, when the joint angles are q_ref (equals l_rest + l_opt always)
        self.l_mtc = self.l_ref_mtc # initialize to reference length (updated in update function)
        self.l_ce = 0
        
        # initialize muscle activation params
        self.A = 0.001
        self.tau = muscle_params.tau       # activation dynamics constant

        # muscle force and v_ce = 0 in flight
        self.F_mtc = 0
        self.f_l = None    
        self.f_v = None  
        self.v_ce = None

        ############### for data logging ####################
        self.F_mtc_list = [self.F_mtc]
        self.f_l_list = [self.f_l]
        self.f_v_list = [self.f_v]
        self.A_list = [self.A]
        self.l_mtc_list = [self.l_mtc]

    def update(self, S, l_mtc):

        self.l_mtc = l_mtc

        if self.l_ce == 0:  # initial condition (we always assume l_se equals l_rest at the start)
            self.l_ce = self.l_mtc - self.l_se
        else:
            self.l_se = self.l_mtc - self.l_ce # instantaneous length of series elastic element

        # series elastic element force (= mtc force)
        self.F_mtc = self.F_max*self.f_se(self.l_se/self.l_rest , self.e_ref_see)
        
        # normalized (just to keep it similar to the simulink model)
        l_ce_norm = self.l_ce/self.l_opt

        # Force length relationship of contractile element
        self.f_l = self.f_l_relationship(l_ce_norm, self.c, self.w)

        # Force velocity relationship of contractile element
        self.f_v = self.F_mtc / (self.F_max * self.f_l * self.A + 1e-5)

        # inverse of force velocity relationship
        self.v_ce = self.f_v_inverse(self.f_v, self.K, self.N)

        # integrate using forward euler to get l_ce for next timestep
        self.l_ce = self.l_ce + self.v_ce*self.dt

        # compute muscle activation for next timestep from activation dynamics
        self.A = np.clip(self.A + self.activation_dynamics(self.A, S, self.tau) * self.dt, 0, 1)

        return self.F_mtc

    def reset(self):

        self.A = 0.001
        self.l_se = self.l_rest
        self.l_ce = 0
        
        self.F_mtc = None # not relevant in flight rn
        self.f_l = None    # zero force at flight phase
        self.f_v = None # zero velocity at flight phase

    def f_se(self, l_se_norm, e_ref_see):

        e = l_se_norm - 1
        # print("e from f_se function: ", e)

        if e > 0:
            return (e / e_ref_see) ** 2
        else:
            return 0
        

    def f_l_relationship(self, l_ce_norm, c, w):
        return math.exp(c * (abs((l_ce_norm - 1) / w) ** 3))

    
    def f_v_inverse(self, f_v, K, N):

        if f_v < 1:
            v_ce = (f_v - 1) / (1 + K * f_v)
        elif 1 <= f_v < N:
            v_ce = (1 - f_v) / ((1 - N) + 7.56 * K * (f_v - N))
        elif f_v >= N:
            v_ce = 1
        else:
            v_ce = 0

        return v_ce
    
    def activation_dynamics(self, A, S, tau):

        return (1/tau)*(S - A)  # equals dA/dt
    
    def log_data(self):
        self.A_list.append(self.A)
        self.F_mtc_list.append(self.F_mtc)
        self.f_l_list.append(self.f_l)
        self.f_v_list.append(self.f_v)
        self.l_mtc_list.append(self.l_mtc)