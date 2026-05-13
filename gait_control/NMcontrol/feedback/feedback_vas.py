import numpy as np
from gait_control.NMcontrol.feedback.base_feedback import BaseFeedback
import config

class VasFeedback(BaseFeedback):
    """
    Feedback pathway for the Vastus (VAS) muscle.
    """

    def __init__(self):
        super().__init__()
        
        _NSD = config.NERVOUS_SYSTEM_DICTIONARY
        self.MID_DELAY   = _NSD["DELAY"]["MID"]
        self.TIME_STEP = _NSD["GENERAL"]["TIME_STEP"]
        self.ST_FGAIN_VAS = _NSD["VAS"]["FGAIN"]
        self.PRESTIM = _NSD["VAS"]["PRESTIM"]

        MUSCLE_PARAMS = config.MUSCLE_DICTIONARY
        self.F_max_iso = MUSCLE_PARAMS['VAS']['F_max']

        self.vas_fmtc_buffer = None
        self.vas_length_buffer = None

        self.vas_stim_total = None

    def reset(self, l_ce):
        size = max(1, int(self.MID_DELAY / self.TIME_STEP))
        self.vas_fmtc_buffer   = [0.0] * size
        self.vas_length_buffer = [l_ce] * size

        self.vas_stim_total = None

    def update_buffer(self, vas_fmtc, vas_lce):
        self.vas_fmtc_buffer.append(vas_fmtc)
        self.vas_length_buffer.append(vas_lce)

    def compute_stimulation(self, gait_phase):
        force_feedback = self.vas_fmtc_buffer.pop(0)
        self.vas_length_buffer.pop(0)
        
        if gait_phase == 'stance':
            vas_stim = force_feedback / self.F_max_iso * self.ST_FGAIN_VAS
            self.vas_stim_total = vas_stim + self.PRESTIM
        else:
            self.vas_stim_total = self.PRESTIM

        return self.vas_stim_total