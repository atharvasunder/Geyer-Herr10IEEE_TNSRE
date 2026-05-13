import numpy as np
import config

class stim_feedback_mixing:
    """
    Handles inter-muscle feedback mixing for both stance and swing controllers.
    """

    def __init__(self, feedback_list, leg_name):
        self.name = "feedback_mixing"
        self.leg_name = leg_name
        self.contra_leg_name = 'leg_2' if leg_name == 'leg_1' else 'leg_1'
        
        self.vas_fb = feedback_list[0]
        self.sol_fb = feedback_list[1]
        self.gas_fb = feedback_list[2]
        self.ta_fb  = feedback_list[3]
        self.ham_fb = feedback_list[4]
        self.glu_fb = feedback_list[5]
        self.hfl_fb = feedback_list[6]

        _NSD = config.NERVOUS_SYSTEM_DICTIONARY
        self.SWING_STIM = _NSD["GENERAL"]["SWING_STIM"]
        self.TIME_STEP  = _NSD["GENERAL"]["TIME_STEP"]

        # Delays
        self.MID_DELAY_SAMPLES = max(1, int(_NSD["DELAY"]["MID"] / self.TIME_STEP))
        self.SHORT_DELAY_SAMPLES = max(1, int(_NSD["DELAY"]["SHORT"] / self.TIME_STEP))

        # Couplings and Balance Gains
        self.G_SOLTA = _NSD["COUPLINGS"]["G_SOLTA"]
        self.G_HAMHFL = _NSD["COUPLINGS"]["G_HAMHFL"]
        
        bal = _NSD["BALANCE"]
        self.KP = bal["KP"]
        self.KD = bal["KD"]
        self.KBW = bal["KBW"]
        self.DELTA_S = bal["DELTA_S"]
        self.THETA_REF = bal["THETA_REF"]
        self.K_PHI = bal["K_PHI"]
        self.K_LEAN = bal["K_LEAN"]
        self.PHI_K_OFF = bal["PHI_K_OFF"]

        self.body_weight = config.MODEL_PARAMS['body_weight']

        # Short/Mid delay buffers for kinematic/force states used in cross-muscle logic
        self.knee_angle_buf = None
        self.knee_vel_buf = None
        
        self.trunk_angle_buf = None
        self.trunk_vel_buf = None
        
        self.F_ipsi_buf = None
        self.F_contra_buf = None
        
        self.F_ipsi_smoothed = 0.0
        self.F_contra_smoothed = 0.0

        self.last_gait_phase = 'stance'
        self.S_lean_PTO = 0.0

    def reset(self, rbs_states):
        """
        Resets and initializes buffers properly.
        """
        knee_angle = rbs_states[self.leg_name]['q']['knee']
        knee_vel = rbs_states[self.leg_name]['q_dot']['knee']
        
        trunk_angle = rbs_states['trunk']['theta']
        trunk_vel = rbs_states['trunk']['theta_dot']
        
        F_ipsi = rbs_states[self.leg_name]['grf']
        F_contra = rbs_states[self.contra_leg_name]['grf']

        self.knee_angle_buf = [knee_angle] * self.MID_DELAY_SAMPLES
        self.knee_vel_buf   = [knee_vel] * self.MID_DELAY_SAMPLES

        self.trunk_angle_buf = [trunk_angle] * self.SHORT_DELAY_SAMPLES
        self.trunk_vel_buf   = [trunk_vel] * self.SHORT_DELAY_SAMPLES

        self.F_ipsi_buf   = [F_ipsi] * self.SHORT_DELAY_SAMPLES
        self.F_contra_buf = [F_contra] * self.SHORT_DELAY_SAMPLES
        
        self.F_ipsi_smoothed = F_ipsi / self.body_weight
        self.F_contra_smoothed = F_contra / self.body_weight

        self.last_gait_phase = 'stance'
        self.S_lean_PTO = 0.0

    def update_buffers(self, rbs_states):
        """
        Pushes new kinematic/force states into the delay buffers.
        """
        self.knee_angle_buf.append(rbs_states[self.leg_name]['q']['knee'])
        self.knee_vel_buf.append(rbs_states[self.leg_name]['q_dot']['knee'])
        
        self.trunk_angle_buf.append(rbs_states['trunk']['theta'])
        self.trunk_vel_buf.append(rbs_states['trunk']['theta_dot'])

        self.F_ipsi_buf.append(rbs_states[self.leg_name]['grf'])
        self.F_contra_buf.append(rbs_states[self.contra_leg_name]['grf'])

    def compute_stimulations(self, rbs_states, gait_phase):
        """
        Computes final stimulations applying inter-muscle linking per Stance and Swing Reflex paths.
        """
        # Call individual muscle `compute_stimulation` first
        vas_self = self.vas_fb.compute_stimulation(gait_phase)
        sol_self = self.sol_fb.compute_stimulation(gait_phase)
        gas_self = self.gas_fb.compute_stimulation(gait_phase)
        ta_self  = self.ta_fb.compute_stimulation(gait_phase)
        ham_self = self.ham_fb.compute_stimulation(gait_phase)
        glu_self = self.glu_fb.compute_stimulation(gait_phase)
        hfl_self = self.hfl_fb.compute_stimulation(gait_phase)

        # Extract delayed kinematic/force states from buffers (must be done ALWAYS to avoid infinite buffers)
        # Mid delay (t_m)
        phi_k = self.knee_angle_buf.pop(0)
        phi_k_dot = self.knee_vel_buf.pop(0)
        
        # Short delay (t_s)
        theta = self.trunk_angle_buf.pop(0)
        theta_dot = self.trunk_vel_buf.pop(0)

        # Exponential smoothing on GRF to mitigate rigid-body impact spikes
        alpha = 0.99
        raw_F_ipsi = self.F_ipsi_buf.pop(0) / self.body_weight
        raw_F_contra = self.F_contra_buf.pop(0) / self.body_weight
        
        self.F_ipsi_smoothed = alpha * self.F_ipsi_smoothed + (1.0 - alpha) * raw_F_ipsi
        self.F_contra_smoothed = alpha * self.F_contra_smoothed + (1.0 - alpha) * raw_F_contra
        
        F_ipsi = self.F_ipsi_smoothed
        F_contra = self.F_contra_smoothed

        # Detect Take-Off transition to calculate S_lean_PTO
        if self.last_gait_phase == 'stance' and gait_phase == 'swing':
            self.S_lean_PTO = self.K_LEAN * (-self.THETA_REF + theta)    # for HFL, I want the muscle to activate less when the trunk leans too much
            
        self.last_gait_phase = gait_phase

        if gait_phase == 'swing':
            S_sol = sol_self
            S_ta  = ta_self
            S_gas = gas_self
            S_vas = vas_self
            S_ham = ham_self
            S_glu = glu_self
            
            # HFL crossing
            l_ce_ham_term = self.ham_fb.last_length_feedback
            S_hfl = hfl_self - self.G_HAMHFL * l_ce_ham_term + self.S_lean_PTO
            
            res = [S_vas, S_sol, S_gas, S_ta, S_ham, S_glu, S_hfl]
            return [np.clip(s, self.SWING_STIM, 1.0) for s in res]

        else:
            # Stance reflexes
            # Dsup flag -> provided by leg state directly
            DSup = rbs_states[self.leg_name]['Dsup']
            
            S_sol = sol_self
            S_gas = gas_self

            # TA cross feedback
            F_sol_tl = self.sol_fb.last_force_feedback
            S_ta = ta_self - self.G_SOLTA * (F_sol_tl / self.sol_fb.F_max_iso)

            # VAS
            knee_hyper = 0.0
            if phi_k > self.PHI_K_OFF and phi_k_dot > 0:
                knee_hyper = self.K_PHI * (phi_k - self.PHI_K_OFF)
                
            S_vas = vas_self - knee_hyper - self.KBW * abs(F_contra) * DSup

            # Torso feedback
            pd_torso = self.KP * (self.THETA_REF - theta) - self.KD * theta_dot
            pd_torso_pos = max(pd_torso, 0.0)
            pd_torso_neg = min(pd_torso, 0.0)  
            
            S_ham = ham_self + pd_torso_pos * self.KBW * abs(F_ipsi)
            
            pd_torso_glu = 0.68 * self.KP * (self.THETA_REF - theta) - self.KD * theta_dot  # for extensor HAM, GLU, i want to activate them more when the trunk leans more
            pd_torso_glu_pos = max(pd_torso_glu, 0.0)
            S_glu = glu_self + pd_torso_glu_pos * self.KBW * abs(F_ipsi) - self.DELTA_S * DSup
            
            S_hfl = hfl_self - pd_torso_neg * self.KBW * abs(F_ipsi) + self.DELTA_S * DSup

            res = [S_vas, S_sol, S_gas, S_ta, S_ham, S_glu, S_hfl]
            return [np.clip(s, self.SWING_STIM, 1.0) for s in res]