import numpy as np
from gait_control.NMcontrol.feedback.base_feedback import BaseFeedback
import config

class HflFeedback(BaseFeedback):
    """
    Feedback pathway for the Hip Flexor (HFL) muscle.
    """

    def __init__(self):
        super().__init__()
        _NSD = config.NERVOUS_SYSTEM_DICTIONARY
        self.SHORT_DELAY = _NSD["DELAY"]["SHORT"]
        self.TIME_STEP = _NSD["GENERAL"]["TIME_STEP"]

        self.FGAIN  = _NSD["HFL"]["FGAIN"]
        self.LCEOFF = _NSD["HFL"]["LCEOFF"]
        self.PRESTIM = _NSD["HFL"]["PRESTIM"]

        MUSCLE_PARAMS = config.MUSCLE_DICTIONARY
        self.l_opt = MUSCLE_PARAMS['HFL']['l_opt']

        self.force_buffer = None
        self.length_buffer = None

    def reset(self, l_ce):
        size = max(1, int(self.SHORT_DELAY / self.TIME_STEP))
        self.force_buffer = [0.0] * size
        self.length_buffer = [l_ce] * size

    def update_buffer(self, fmtc, l_ce):
        self.force_buffer.append(fmtc)
        self.length_buffer.append(l_ce)

    def compute_stimulation(self, gait_phase):
        self.force_buffer.pop(0)
        l_ce = self.length_buffer.pop(0)

        if gait_phase == 'swing':
            hfl_stim = self.PRESTIM + self.FGAIN * (l_ce / self.l_opt - self.LCEOFF)
            return hfl_stim
        else:
            return self.PRESTIM
