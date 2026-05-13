import numpy as np
from gait_control.NMcontrol.feedback.base_feedback import BaseFeedback
import config

class HamFeedback(BaseFeedback):
    """
    Feedback pathway for the Hamstring (HAM) muscle.
    """

    def __init__(self):
        super().__init__()
        _NSD = config.NERVOUS_SYSTEM_DICTIONARY
        self.SHORT_DELAY = _NSD["DELAY"]["SHORT"]
        self.TIME_STEP = _NSD["GENERAL"]["TIME_STEP"]

        self.FGAIN  = _NSD["HAM"]["FGAIN"]
        self.LCEOFF = _NSD["HAM"]["LCEOFF"]
        self.PRESTIM = _NSD["HAM"]["PRESTIM"]

        MUSCLE_PARAMS = config.MUSCLE_DICTIONARY
        self.F_max_iso = MUSCLE_PARAMS['HAM']['F_max']
        self.l_opt = MUSCLE_PARAMS['HAM']['l_opt']

        self.force_buffer = None
        self.length_buffer = None

        # stored to expose to mixing class
        self.last_length_feedback = 0.0

    def reset(self, l_ce):
        size = max(1, int(self.SHORT_DELAY / self.TIME_STEP))
        self.force_buffer = [0.0] * size
        self.length_buffer = [l_ce] * size
        self.last_length_feedback = l_ce / self.l_opt - self.LCEOFF

    def update_buffer(self, fmtc, l_ce):
        self.force_buffer.append(fmtc)
        self.length_buffer.append(l_ce)

    def compute_stimulation(self, gait_phase):
        force_feedback = self.force_buffer.pop(0)
        l_ce = self.length_buffer.pop(0)

        self.last_length_feedback = l_ce / self.l_opt - self.LCEOFF

        if gait_phase == 'swing':
            ham_stim = self.PRESTIM + self.FGAIN * (force_feedback / self.F_max_iso)
            return ham_stim
        else:
            return self.PRESTIM
