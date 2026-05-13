import numpy as np
from gait_control.NMcontrol.feedback.base_feedback import BaseFeedback
import config

class GasFeedback(BaseFeedback):
    """
    Feedback pathway for the Gastrocnemius (GAS) muscle.
    """

    def __init__(self):
        super().__init__()
        
        _NSD = config.NERVOUS_SYSTEM_DICTIONARY
        self.LONG_DELAY     = _NSD["DELAY"]["LONG"]
        self.TIME_STEP      = _NSD["GENERAL"]["TIME_STEP"]
        self.ST_FGAIN_GAS   = _NSD["GAS"]["FGAIN"]
        self.PRESTIM        = _NSD["GAS"]["PRESTIM"]

        MUSCLE_PARAMS = config.MUSCLE_DICTIONARY
        self.F_max_iso = MUSCLE_PARAMS['GAS']['F_max']

        self.gas_fmtc_buffer = None
        self.gas_length_buffer = None

    def reset(self, l_ce):
        size = max(1, int(self.LONG_DELAY / self.TIME_STEP))
        self.gas_fmtc_buffer   = [0.0] * size
        self.gas_length_buffer = [l_ce] * size

        self.gas_stim_total = None

    def update_buffer(self, gas_fmtc, gas_length):
        self.gas_fmtc_buffer.append(gas_fmtc)
        self.gas_length_buffer.append(gas_length)

    def compute_stimulation(self, gait_phase):
        force_feedback = self.gas_fmtc_buffer.pop(0)
        self.gas_length_buffer.pop(0)
            
        if gait_phase == 'stance':
            gas_stim = force_feedback / self.F_max_iso * self.ST_FGAIN_GAS
            self.gas_stim_total = gas_stim + self.PRESTIM
        else:
            self.gas_stim_total = self.PRESTIM

        return self.gas_stim_total
