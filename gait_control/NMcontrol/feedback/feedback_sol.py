import numpy as np
from gait_control.NMcontrol.feedback.base_feedback import BaseFeedback
import config

class SolFeedback(BaseFeedback):
    """
    Feedback pathway for the Soleus (SOL) muscle.
    """

    def __init__(self):
        super().__init__()
        
        _NSD = config.NERVOUS_SYSTEM_DICTIONARY
        self.LONG_DELAY = _NSD["DELAY"]["LONG"]
        self.TIME_STEP = _NSD["GENERAL"]["TIME_STEP"]
        self.ST_FGAIN_SOL = _NSD["SOL"]["FGAIN"]
        self.ST_PRESTIM_SOL = _NSD["SOL"]["PRESTIM"]

        MUSCLE_PARAMS = config.MUSCLE_DICTIONARY
        self.F_max_iso = MUSCLE_PARAMS['SOL']['F_max']

        self.sol_fmtc_buffer = None
        self.sol_length_buffer = None

        self.sol_stim_total = None
        self.last_force_feedback = 0.0

    def reset(self, l_ce):
        size = max(1, int(self.LONG_DELAY / self.TIME_STEP))
        self.sol_fmtc_buffer   = [0.0] * size
        self.sol_length_buffer = [l_ce] * size

        self.sol_stim_total = None
        self.last_force_feedback = 0.0

    def update_buffer(self, sol_fmtc, sol_length):
        self.sol_fmtc_buffer.append(sol_fmtc)
        self.sol_length_buffer.append(sol_length)

    def compute_stimulation(self, gait_phase):
        self.last_force_feedback = self.sol_fmtc_buffer.pop(0)
        self.sol_length_buffer.pop(0) # not used by SOL self feedback
            
        if gait_phase == 'stance':
            sol_stim = self.last_force_feedback / self.F_max_iso * self.ST_FGAIN_SOL
            self.sol_stim_total = sol_stim + self.ST_PRESTIM_SOL
        else:
            self.sol_stim_total = self.ST_PRESTIM_SOL

        return self.sol_stim_total
