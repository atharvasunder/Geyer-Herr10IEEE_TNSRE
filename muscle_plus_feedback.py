import math
import numpy as np
from fifo_buffer import FIFOBuffer

class mtu_and_feedback():
    """
    Class describing a muscle connected between two joints. 

    Inputs:
    joint_list: list of all joints
    actuated_joint_list: the two joints which the
                          muscle actuates
    location: front or back, used to decide torque contribution on joint's sign

    by definition here, base joint is the connection that is vertically higher
    (output of muscle update on the base joint is positive and on mate joint is negative)    

    muscle_params:
    - MTC properties
    - Force-length relationship params
    - Force-velocity relationship params
    - Tendon properties
    - moment arm constants (eg: r_max)
    
    nervous params:
    - feedback gain
    - feedback constant delay (stim0)
    - etc
    
    Out:
    joint torques

    """

    # constructor
    def __init__(self, name, joint_list, actuated_joint_list, location, muscle_type, muscle_params, nervous_params, dt):
        self.name = name

        self.muscle_params = muscle_params
        self.nervous_params = nervous_params
        self.actuated_joint_list = actuated_joint_list

        ################ initialize muscle params ##################

        # muscle type (useful to decide feedback pathway)
        # VAS, SOL, GAS, TA, GLU, HAM, HFL
        self.muscle_type = muscle_type

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
        
        # initialize muscle-joint properties
        self.r0_array = muscle_params.r0_array # size = no of joints (contains zeros for unactuated joints)
        self.q_ref_array = muscle_params.q_ref_array # joint angle at which MTU length = l_opt + l_slack
        self.q_max_array = muscle_params.q_max_array # joint angle at which r = r0
        self.rho_array = muscle_params.rho_array # rho ensures that the MTU fiber length stays within physiological limits
        self.q_array = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.r_array = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        
        # torque contributions of muscle on each joint
        self.torque_array = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        
        self.joint_list = joint_list
        self.location = location # front/back

        # muscle activation
        self.A = 0.001
        
        # initialize tendon length to rest length (always done!)
        self.l_se = self.l_rest
        self.l_mtc = self.l_rest + self.l_opt # initialize to reference length (updated in update function)
        self.l_ref_mtc = self.l_rest + self.l_opt # MTC reference length, when the joint angles are q_ref
        self.l_ce = 0
        
        # for feedback
        self.F_mtc_feedback = None
        self.l_ce_feedback = None
        
        # muscle force and v_ce = 0 in flight
        self.F_mtc = None
        self.f_l = None    
        self.f_v = None  
        self.v_ce = None 

        ############## store moment arm values for muscle at each joint ###################
        
        for j in range(len(joint_list)):
            joint = joint_list[j]
            r0 = self.r0_array[j]
            q = joint.q
            q_ref = self.q_ref_array[j] # reference values of q
            q_max = self.q_max_array[j]
            self.q_array[j] = q
            
            if joint.name in self.actuated_joint_list:
                if joint.name == 'hip_1' or joint.name == 'hip_2':
                    self.r_array[j] = r0
                else:
                    self.r_array[j] = r0*np.cos(q - q_max)

        print('r_array: ', self.r_array)
        self.r_array_initial = self.r_array     # store initial lever arm for use in reset function

        ################ initialize nervous system ################

        # nervous system properties
        self.stim0 = nervous_params.stim0   # Constant stimulation from the brain to the muscle
        self.tau = nervous_params.tau       # activation dynamics constant
        self.delP_list = nervous_params.delP_list

        self.delP_short = self.delP_list[0]
        self.delP_medium = self.delP_list[1]
        self.delP_long = self.delP_list[2]

        # Gain for feedback
        self.G = nervous_params.G
        self.G_l = nervous_params.G_l
        self.l_off = nervous_params.l_off

        self.k_p = nervous_params.k_p
        self.k_d = nervous_params.k_d

        self.k_bw = nervous_params.k_bw
        self.delta_S = nervous_params.delta_S

        self.p_trunk_ref = nervous_params.p_trunk_ref
        
        self.buffer_size_short = int(self.delP_short/dt) # equals number of timesteps comprising the transport delay
        self.buffer_size_medium = int(self.delP_medium/dt)
        self.buffer_size_long = int(self.delP_long/dt)
        
        # delay durations for different muscles
        self.F_plus_sol = FIFOBuffer(self.buffer_size_long)

        self.l_plus_ta = FIFOBuffer(self.buffer_size_long)
        self.F_minus_solta = FIFOBuffer(self.buffer_size_long)

        self.F_plus_gas = FIFOBuffer(self.buffer_size_long)

        self.F_plus_vas = FIFOBuffer(self.buffer_size_medium)
        self.knee_angle_vas = FIFOBuffer(self.buffer_size_medium)
        self.knee_angle_speed_vas = FIFOBuffer(self.buffer_size_medium)
        self.grf_opposite_vas = FIFOBuffer(self.buffer_size_short)

        self.trunk_lean_ham = FIFOBuffer(self.buffer_size_short)
        self.trunk_lean_speed_ham = FIFOBuffer(self.buffer_size_short)
        self.grf_ham = FIFOBuffer(self.buffer_size_short)

        self.trunk_lean_glu = FIFOBuffer(self.buffer_size_short)
        self.trunk_lean_speed_glu = FIFOBuffer(self.buffer_size_short)
        self.grf_glu = FIFOBuffer(self.buffer_size_short)

        self.trunk_lean_hfl = FIFOBuffer(self.buffer_size_short)
        self.trunk_lean_speed_hfl = FIFOBuffer(self.buffer_size_short)
        self.grf_hfl = FIFOBuffer(self.buffer_size_short)

        self.F_minus_solta_feedback = 0     # as TA uses F_SOL as feedback

        ############### for data logging ####################

        self.A_list = [self.A]
        self.F_mtc_list = [self.F_mtc]
        self.f_l_list = [self.f_l]
        self.f_v_list = [self.f_v]
        # self.moment_arms = [self.r_ref_array] # will be a list of arrays
        self.torque_contributions = [self.torque_array] # List of arrays of torque contribution of muscle on each joint
    
    ########### update torque contribution ##################

    def update(self, p_trunk, p_dot_trunk, grf_norm, grf_norm_opposite, Dsup, phase, dt):
        
        # l_mtc instantaneous length
        for j in range(len(self.joint_list)):
            joint = self.joint_list[j]
            r0 = self.r0_array[j]
            rho = self.rho_array[j]
            q = joint.q
            q_ref = self.q_ref_array[j]
            q_max = self.q_max_array[j]

            # keep updating l_mtc with each joints contribution
            if joint.name in self.actuated_joint_list:
                if joint.name == 'hip_1' or joint.name == 'hip_2':
                    self.r_array[j] = r0
                    self.l_mtc = self.l_ref_mtc + rho*self.r_array[j]*(q - q_ref)
                else:
                    self.r_array[j] = r0*np.cos(q - q_max)
                    self.l_mtc = self.l_ref_mtc + rho*r0*(np.sin(q - q_max) - np.sin(q_ref - q_max))
                     
            self.q_array[j] = q

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
        # self.f_v = self.F_mtc/(self.F_max*self.f_l*self.A + 1e-15)
        self.f_v = min(self.F_mtc / (self.F_max * self.f_l * self.A + 1e-15), 1.0)  # clip to be <= 1

        # inverse of force velocity relationship
        self.v_ce = self.f_v_inverse(self.f_v, self.K, self.N)
        
        # calculate brain stimulation
        S = self.feedback(phase, Dsup, grf_norm, grf_norm_opposite, p_trunk, p_dot_trunk)

        # integrate using forward euler to get l_ce for next timestep
        self.l_ce = self.l_ce + self.v_ce*dt

        # compute muscle activation for next timestep from activation dynamics
        self.A = np.clip(self.A + self.activation_dynamics(self.A, S, self.tau) * dt, 0, 1)

        for j in range(len(self.joint_list)):
            r = self.r_array[j]
            if self.location == 'front':
                self.torque_array[j] = -r*self.F_mtc
            elif self.location == 'back':
                self.torque_array[j] = r*self.F_mtc
        
        # returns array of torques that this muscle produces at each joint
        return self.torque_array
    
    ################## feedback mechanisms ######################
    
    def feedback(self, phase, Dsup, grf_norm, grf_norm_opposite, p_trunk, p_dot_trunk):

        # VAS, SOL, GAS, TA, GLU, HAM, HFL

        if phase == 'stance':

            if self.muscle_type == 'SOL':
                self.F_plus_sol_feedback = self.F_plus_sol.push(self.F_mtc) # returns none unless the buffer is full, then returns first in element
                self.F_minus_solta_feedback = self.F_minus_solta.push(self.F_mtc)   # used by TA
                return self.F_plus_sol_feedback*self.G + self.stim0
            
            elif self.muscle_type == 'GAS':
                self.F_plus_gas_feedback = self.F_plus_gas.push(self.F_mtc)
                return self.F_plus_gas_feedback*self.G + self.stim0
            
            elif self.muscle_type == 'VAS':

                self.F_plus_vas_feedback = self.F_plus_vas.push(self.F_mtc)
                self.grf_opposite_vas_feedback = self.grf_opposite_vas.push(grf_norm_opposite)
                
                for joint in self.joint_list:
                    if joint.name == self.actuated_joint_list[0]:
                        joint_angle = joint.q
                        joint_vel = joint.q_dot

                self.knee_angle_vas_feedback = self.knee_angle_vas.push(joint_angle)
                self.knee_angle_speed_vas_feedback = self.knee_angle_speed_vas.push(joint_vel)

                if self.knee_angle_vas_feedback > 170*np.pi/180 and self.knee_angle_speed_vas_feedback > 0:
                    return self.G*self.F_plus_vas_feedback - self.k_p*(self.knee_angle_vas_feedback - 170*np.pi/180) + self.stim0 - self.k_bw*self.grf_opposite_vas_feedback*Dsup
                else:
                    return self.G*self.F_plus_vas_feedback + self.stim0 - self.k_bw*self.grf_opposite_vas_feedback*Dsup

            elif self.muscle_type == 'TA':
                self.l_plus_ta_feedback = self.l_plus_ta.push(self.l_ce)
                return  self.G_l*(self.l_plus_ta_feedback - self.l_off) - self.G*self.F_minus_solta_feedback + self.stim0

            elif self.muscle_type == 'GLU':

                self.trunk_lean_glu_feedback = self.trunk_lean_glu.push(p_trunk)
                self.trunk_lean_speed_glu_feedback = self.trunk_lean_speed_glu.push(p_dot_trunk)
                self.grf_glu_feedback = self.grf_glu.push(grf_norm)

                return self.stim0 + 0.68*self.k_p*(self.trunk_lean_glu_feedback - self.p_trunk_ref) + self.k_d*self.trunk_lean_speed_glu_feedback + self.k_bw*self.grf_glu_feedback - Dsup*self.delta_S
        
            elif self.muscle_type == 'HFL':

                self.trunk_lean_hfl_feedback = self.trunk_lean_hfl.push(p_trunk)
                self.trunk_lean_speed_hfl_feedback = self.trunk_lean_speed_hfl.push(p_dot_trunk)
                self.grf_hfl_feedback = self.grf_hfl.push(grf_norm)

                return self.stim0 + self.k_p*(self.trunk_lean_hfl_feedback - self.p_trunk_ref) + self.k_d*self.trunk_lean_speed_hfl_feedback - self.k_bw*self.grf_hfl_feedback + Dsup*self.delta_S
            
            elif self.muscle_type == 'HAM':

                self.trunk_lean_ham_feedback = self.trunk_lean_ham.push(p_trunk)
                self.trunk_lean_speed_ham_feedback = self.trunk_lean_speed_ham.push(p_dot_trunk)
                self.grf_ham_feedback = self.grf_ham.push(grf_norm)

                return self.stim0 + self.k_p*(self.trunk_lean_ham_feedback - self.p_trunk_ref) + self.k_d*self.trunk_lean_speed_ham_feedback + self.k_bw*self.grf_ham_feedback
            
            else:
                raise ValueError(f"Invalid muscle type: {self.muscle_type}")
        
        elif phase == 'swing':

            return self.stim0

        else:
            raise ValueError(f"Invalid phase type: {self.muscle_type}")
    
    ################# reset function for muscle during flight phases ####################

    def reset(self):

        self.A = 0.001
        self.l_se = self.l_rest
        self.l_ce = 0
        
        self.F_mtc = None # not relevant in flight rn
        self.f_l = None    # zero force at flight phase
        self.f_v = None # zero velocity at flight phase
        
        self.torque_array = np.array([0,0,0,0,0,0])
        self.r_array = self.r_array_initial

    ##################### series elastic element force #########################

    def f_se(self, l_se_norm, e_ref_see):

        e = l_se_norm - 1
        # print("e from f_se function: ", e)

        if e > 0:
            return (e / e_ref_see) ** 2
        else:
            return 0
        
    ####################### force length rleationship ############################

    def f_l_relationship(self, l_ce_norm, c, w):

        return math.exp(c * (abs((l_ce_norm - 1) / w) ** 3))
    
    ####################### inverse of force velocity relationship ########################
    
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

        return (1/tau)*(S - A)  # dt/tau????
    
    def log_data(self):
        self.A_list.append(self.A)
        self.F_mtc_list.append(self.F_mtc)
        self.f_l_list.append(self.f_l)
        self.f_v_list.append(self.f_v)
        # self.moment_arms.append(self.r_array)
        self.torque_contributions.append(self.torque_array)