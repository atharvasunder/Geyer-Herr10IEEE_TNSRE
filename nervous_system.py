import math
import numpy as np
from fifo_buffer import FIFOBuffer

class feedback_loop():
    """
    Class describing the feedback pathway for a muscle.
    It requires inputs from all lists of the rigid body system as
    different feedback pathways use different sets of information. This class
    contains functions that compute stimulation of the muscle under consideration.

    Inputs: joint list, body_list, muscle_list

    nervous_params

    phase (used in update function)

    output: muscle stimulations
    
    """

    # constructor
    def __init__(self, muscle_type, joint_list, body_list, muscle_list, nervous_params, dt):
        
        self.muscle_type = muscle_type       # name of muscle

        self.body_list = body_list
        self.joint_list = joint_list
        self.muscle_list = muscle_list

        self.nervous_params = nervous_params

        ################ initialize nervous system ################

        # nervous system properties
        self.stim0 = nervous_params.stim0   # Constant stimulation from the brain to the muscle
        self.S = self.stim0
        self.delP_list = nervous_params.delP_list

        self.delP_short = self.delP_list[0]
        self.delP_medium = self.delP_list[1]
        self.delP_long = self.delP_list[2]

        self.buffer_size_short = int(self.delP_short/dt) # equals number of timesteps comprising the transport delay
        self.buffer_size_medium = int(self.delP_medium/dt)
        self.buffer_size_long = int(self.delP_long/dt)

        # Gain for feedback
        self.G = nervous_params.G
        self.G_l = nervous_params.G_l
        self.G_l_2 = nervous_params.G_l_2
        self.l_off_2 = nervous_params.l_off_2
        self.l_off = nervous_params.l_off

        self.k_p = nervous_params.k_p
        self.k_d = nervous_params.k_d
        self.k_lean = nervous_params.k_lean

        self.k_bw = nervous_params.k_bw
        self.delta_S = nervous_params.delta_S

        self.p_trunk_ref = nervous_params.p_trunk_ref
        
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

        self.F_plus_ham = FIFOBuffer(self.buffer_size_short)
        self.F_plus_glu = FIFOBuffer(self.buffer_size_short)

        self.l_ce_hfl = FIFOBuffer(self.buffer_size_short)
        self.l_ce_ham = FIFOBuffer(self.buffer_size_short)

    def update(self, phase, Dsup, grf_norm, grf_norm_opposite):
        
        # VAS, SOL, GAS, TA, GLU, HAM, HFL

        if phase == 'stance':

            if self.muscle_type == 'SOL_1':
                F_mtc_sol_1 = self.muscle_list[1].F_mtc
                self.F_plus_sol_feedback = self.F_plus_sol.push(F_mtc_sol_1) # returns 0 unless the buffer is full, then returns first in element
                return self.F_plus_sol_feedback*self.G + self.stim0
            
            elif self.muscle_type == 'SOL_2':
                F_mtc_sol_2 = self.muscle_list[8].F_mtc
                self.F_plus_sol_feedback = self.F_plus_sol.push(F_mtc_sol_2)
                # print("l_ce/l_opt: ", self.muscle_list[8].l_ce/self.muscle_list[8].l_opt)
                # print('F_mtc_sol_2: ', self.F_plus_sol_feedback)
                # print("G: ", self.G)
                # print("feedback: ", self.F_plus_sol_feedback*self.G + self.stim0)
                return self.F_plus_sol_feedback*self.G + self.stim0
            
            elif self.muscle_type == 'GAS_1':
                F_mtc_gas_1 = self.muscle_list[2].F_mtc
                self.F_plus_gas_feedback = self.F_plus_gas.push(F_mtc_gas_1)
                return self.F_plus_gas_feedback*self.G + self.stim0
            
            elif self.muscle_type == 'GAS_2':
                F_mtc_gas_2 = self.muscle_list[9].F_mtc
                self.F_plus_gas_feedback = self.F_plus_gas.push(F_mtc_gas_2)
                return self.F_plus_gas_feedback*self.G + self.stim0
            
            elif self.muscle_type == 'VAS_1':
                
                F_mtc_vas_1 = self.muscle_list[0].F_mtc
                self.F_plus_vas_feedback = self.F_plus_vas.push(F_mtc_vas_1)

                self.grf_opposite_vas_feedback = self.grf_opposite_vas.push(grf_norm_opposite)
                
                knee_angle = self.joint_list[1].q
                knee_vel = self.joint_list[1].q_dot

                self.knee_angle_vas_feedback = self.knee_angle_vas.push(knee_angle)
                self.knee_angle_speed_vas_feedback = self.knee_angle_speed_vas.push(knee_vel)

                if self.knee_angle_vas_feedback > 170*np.pi/180 and self.knee_angle_speed_vas_feedback > 0:
                    return self.G*self.F_plus_vas_feedback - self.k_p*(self.knee_angle_vas_feedback - 170*np.pi/180) \
                        + self.stim0 - self.k_bw*self.grf_opposite_vas_feedback*Dsup
                else:
                    return self.G*self.F_plus_vas_feedback + self.stim0 - self.k_bw*self.grf_opposite_vas_feedback*Dsup
                
            elif self.muscle_type == 'VAS_2':
                
                F_mtc_vas_2 = self.muscle_list[7].F_mtc
                self.F_plus_vas_feedback = self.F_plus_vas.push(F_mtc_vas_2)

                self.grf_opposite_vas_feedback = self.grf_opposite_vas.push(grf_norm_opposite)
                
                knee_angle = self.joint_list[4].q
                knee_vel = self.joint_list[4].q_dot
                # print('angle: ', knee_angle)
                # print('vel: ', knee_vel)

                self.knee_angle_vas_feedback = self.knee_angle_vas.push(knee_angle)
                self.knee_angle_speed_vas_feedback = self.knee_angle_speed_vas.push(knee_vel)

                if self.knee_angle_vas_feedback > 170*np.pi/180 and self.knee_angle_speed_vas_feedback > 0:
                    # print('true')
                    return self.G*self.F_plus_vas_feedback - self.k_p*(self.knee_angle_vas_feedback - 170*np.pi/180) \
                        + self.stim0 - self.k_bw*self.grf_opposite_vas_feedback*Dsup
                else:
                    # print("l_ce/l_opt: ", self.muscle_list[7].l_ce/self.muscle_list[7].l_opt)
                    # print('F_mtc_vas_2: ', self.F_plus_vas_feedback)
                    # print("Dsup: ", Dsup)
                    # print("G: ", self.G)
                    # print("grf opposite: ", self.grf_opposite_vas_feedback)
                    # print("feedback: ", self.G*self.F_plus_vas_feedback + self.stim0 - self.k_bw*self.grf_opposite_vas_feedback*Dsup)
                    return self.G*self.F_plus_vas_feedback + self.stim0 - self.k_bw*self.grf_opposite_vas_feedback*Dsup

            elif self.muscle_type == 'TA_1':
                l_ce_ta_1 = self.muscle_list[3].l_ce
                self.l_plus_ta_feedback = self.l_plus_ta.push(l_ce_ta_1)

                F_mtc_sol_1 = self.muscle_list[1].F_mtc
                self.F_minus_solta_feedback = self.F_minus_solta.push(F_mtc_sol_1)

                return  self.G_l*(self.l_plus_ta_feedback - self.l_off) - self.G*self.F_minus_solta_feedback + self.stim0
            
            elif self.muscle_type == 'TA_2':
                l_ce_ta_2 = self.muscle_list[10].l_ce
                self.l_plus_ta_feedback = self.l_plus_ta.push(l_ce_ta_2)

                F_mtc_sol_2 = self.muscle_list[8].F_mtc
                self.F_minus_solta_feedback = self.F_minus_solta.push(F_mtc_sol_2)
                
                return  self.G_l*(self.l_plus_ta_feedback - self.l_off) - self.G*self.F_minus_solta_feedback + self.stim0

            elif self.muscle_type == 'GLU_1':

                p_trunk = self.body_list[0].p
                # print(p_trunk)
                p_dot_trunk = self.body_list[0].vp
                # print("vel", p_dot_trunk)

                self.trunk_lean_glu_feedback = self.trunk_lean_glu.push(p_trunk)
                self.trunk_lean_speed_glu_feedback = self.trunk_lean_speed_glu.push(p_dot_trunk)
                self.grf_glu_feedback = self.grf_glu.push(grf_norm)
                # print('grf', grf_norm)

                # print("k_bw", self.k_bw)

                lean_check =  -0.68*self.k_p*(self.trunk_lean_glu_feedback - self.p_trunk_ref) + self.k_d*self.trunk_lean_speed_glu_feedback

                if lean_check > 0:
                    # print(self.stim0 - lean_check*self.k_bw*self.grf_glu_feedback - Dsup*self.delta_S)
                    return self.stim0 + lean_check*self.k_bw*self.grf_glu_feedback - Dsup*self.delta_S
                else:
                    return self.stim0 - Dsup*self.delta_S
                
            elif self.muscle_type == 'GLU_2':

                p_trunk = self.body_list[0].p
                p_dot_trunk = self.body_list[0].vp

                self.trunk_lean_glu_feedback = self.trunk_lean_glu.push(p_trunk)
                self.trunk_lean_speed_glu_feedback = self.trunk_lean_speed_glu.push(p_dot_trunk)
                self.grf_glu_feedback = self.grf_glu.push(grf_norm)

                lean_check = -0.68*self.k_p*(self.trunk_lean_glu_feedback - self.p_trunk_ref) + self.k_d*self.trunk_lean_speed_glu_feedback

                if lean_check > 0:
                    return self.stim0 + lean_check*self.k_bw*self.grf_glu_feedback - Dsup*self.delta_S
                else:
                    return self.stim0 - Dsup*self.delta_S
        
            elif self.muscle_type == 'HFL_1':

                p_trunk = self.body_list[0].p
                p_dot_trunk = self.body_list[0].vp

                self.trunk_lean_hfl_feedback = self.trunk_lean_hfl.push(p_trunk)
                self.trunk_lean_speed_hfl_feedback = self.trunk_lean_speed_hfl.push(p_dot_trunk)
                self.grf_hfl_feedback = self.grf_hfl.push(grf_norm)

                lean_check = -self.k_p*(self.trunk_lean_hfl_feedback - self.p_trunk_ref) + self.k_d*self.trunk_lean_speed_hfl_feedback

                if lean_check < 0:
                    return self.stim0 + lean_check*self.k_bw*self.grf_hfl_feedback + Dsup*self.delta_S
                
                else:
                    return self.stim0 + Dsup*self.delta_S
                
            elif self.muscle_type == 'HFL_2':

                p_trunk = self.body_list[0].p
                p_dot_trunk = self.body_list[0].vp

                self.trunk_lean_hfl_feedback = self.trunk_lean_hfl.push(p_trunk)
                self.trunk_lean_speed_hfl_feedback = self.trunk_lean_speed_hfl.push(p_dot_trunk)
                self.grf_hfl_feedback = self.grf_hfl.push(grf_norm)

                lean_check = -self.k_p*(self.trunk_lean_hfl_feedback - self.p_trunk_ref) + self.k_d*self.trunk_lean_speed_hfl_feedback

                if lean_check < 0:
                    # print(Dsup)
                    # print(lean_check*self.k_bw*self.grf_hfl_feedback)
                    # print(self.stim0 - lean_check*self.k_bw*self.grf_hfl_feedback + Dsup*self.delta_S)
                    return self.stim0 + lean_check*self.k_bw*self.grf_hfl_feedback + Dsup*self.delta_S
                
                else:
                    return self.stim0 + Dsup*self.delta_S
            
            elif self.muscle_type == 'HAM_1':

                p_trunk = self.body_list[0].p
                p_dot_trunk = self.body_list[0].vp

                self.trunk_lean_ham_feedback = self.trunk_lean_ham.push(p_trunk)
                self.trunk_lean_speed_ham_feedback = self.trunk_lean_speed_ham.push(p_dot_trunk)
                self.grf_ham_feedback = self.grf_ham.push(grf_norm)

                # HAM is only activated when its compression is useful in aligning the trunk with the reference trunk lean angle
                lean_check = -self.k_p*(self.trunk_lean_ham_feedback - self.p_trunk_ref) + self.k_d*self.trunk_lean_speed_ham_feedback

                if lean_check > 0:
                    return self.stim0 + lean_check*self.k_bw*self.grf_ham_feedback
                else:
                    return self.stim0
                
            elif self.muscle_type == 'HAM_2':

                p_trunk = self.body_list[0].p
                p_dot_trunk = self.body_list[0].vp

                self.trunk_lean_ham_feedback = self.trunk_lean_ham.push(p_trunk)
                self.trunk_lean_speed_ham_feedback = self.trunk_lean_speed_ham.push(p_dot_trunk)
                self.grf_ham_feedback = self.grf_ham.push(grf_norm)

                # HAM is only activated when its compression is useful in aligning the trunk with the reference trunk lean angle
                lean_check = -self.k_p*(self.trunk_lean_ham_feedback - self.p_trunk_ref) + self.k_d*self.trunk_lean_speed_ham_feedback

                if lean_check > 0:
                    return self.stim0 + lean_check*self.k_bw*self.grf_ham_feedback
                else:
                    return self.stim0
            
            else:
                raise ValueError(f"Invalid muscle type: {self.muscle_type}")
        
        elif phase == 'swing':

            if self.muscle_type == 'SOL_1':
                return self.stim0
            
            elif self.muscle_type == 'SOL_2':
                return self.stim0
            
            elif self.muscle_type == 'GAS_1':
                return self.stim0
            
            elif self.muscle_type == 'GAS_2':
                return self.stim0
            
            elif self.muscle_type == 'VAS_1':
               return self.stim0
                
            elif self.muscle_type == 'VAS_2':
                return self.stim0

            elif self.muscle_type == 'TA_1':
                l_ce_ta_1 = self.muscle_list[3].l_ce
                self.l_plus_ta_feedback = self.l_plus_ta.push(l_ce_ta_1)

                return  self.G_l*(self.l_plus_ta_feedback - self.l_off) + self.stim0
            
            elif self.muscle_type == 'TA_2':
                l_ce_ta_2 = self.muscle_list[10].l_ce
                self.l_plus_ta_feedback = self.l_plus_ta.push(l_ce_ta_2)
                
                return  self.G_l*(self.l_plus_ta_feedback - self.l_off) + self.stim0

            elif self.muscle_type == 'GLU_1':
                F_mtc_glu_1 = self.muscle_list[4].F_mtc
                self.F_plus_glu_feedback = self.F_plus_glu.push(F_mtc_glu_1)

                return self.stim0 + self.G*self.F_plus_glu_feedback
                
            elif self.muscle_type == 'GLU_2':
                F_mtc_glu_2 = self.muscle_list[11].F_mtc
                self.F_plus_glu_feedback = self.F_plus_glu.push(F_mtc_glu_2)

                return self.stim0 + self.G*self.F_plus_glu_feedback
        
            elif self.muscle_type == 'HFL_1':

                peek_val = self.trunk_lean_hfl.peek()
        
                bias = self.k_lean * (peek_val - self.p_trunk_ref)

                l_ce_hfl_1 = self.muscle_list[6].l_ce
                self.l_ce_hfl_feedback = self.l_ce_hfl.push(l_ce_hfl_1)

                l_ce_ham_1 = self.muscle_list[5].l_ce
                self.l_ce_ham_feedback = self.l_ce_ham.push(l_ce_ham_1)

                return self.stim0 + bias + self.G_l*(self.l_ce_hfl_feedback - self.l_off)\
                - self.G_l_2*(self.l_ce_ham_feedback - self.l_off_2)
                
            elif self.muscle_type == 'HFL_2':
                
                peek_val = self.trunk_lean_hfl.peek()
        
                bias = -self.k_lean * (peek_val - self.p_trunk_ref)

                l_ce_hfl_2 = self.muscle_list[13].l_ce
                # print(l_ce_hfl_2)
                self.l_ce_hfl_feedback = self.l_ce_hfl.push(l_ce_hfl_2)

                l_ce_ham_2 = self.muscle_list[12].l_ce
                self.l_ce_ham_feedback = self.l_ce_ham.push(l_ce_ham_2)
                
                # print(bias)
                # print(self.l_ce_hfl_feedback)
                # print(self.l_off)
                # print(self.stim0 + bias + self.G_l*(self.l_ce_hfl_feedback - self.l_off))
                # print(self.G_l_2*(self.l_ce_ham_feedback - self.l_off_2))

                # print(self.stim0 + bias + self.G_l*(self.l_ce_hfl_feedback - self.l_off)\
                # - self.G_l_2*(self.l_ce_ham_feedback - self.l_off_2))

                return self.stim0 + bias + self.G_l*(self.l_ce_hfl_feedback - self.l_off)\
                - self.G_l_2*(self.l_ce_ham_feedback - self.l_off_2)
            
            elif self.muscle_type == 'HAM_1':

                F_mtc_ham_1 = self.muscle_list[5].F_mtc
                self.F_plus_ham_feedback = self.F_plus_ham.push(F_mtc_ham_1)

                return self.stim0 + self.G*self.F_plus_ham_feedback
                
            elif self.muscle_type == 'HAM_2':

                F_mtc_ham_2 = self.muscle_list[12].F_mtc
                self.F_plus_ham_feedback = self.F_plus_ham.push(F_mtc_ham_2)

                return self.stim0 + self.G*self.F_plus_ham_feedback

        else:
            raise ValueError(f"Invalid phase type: {self.muscle_type}")