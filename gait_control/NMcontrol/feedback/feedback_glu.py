import numpy as np
from gait_control.NMcontrol.feedback.base_feedback import BaseFeedback
import config

class GluFeedback(BaseFeedback):
    """
    Feedback pathway for the Gluteus (GLU) muscle.
    """

    def __init__(self):
        super().__init__()
        _NSD = config.NERVOUS_SYSTEM_DICTIONARY
        self.SHORT_DELAY = _NSD["DELAY"]["SHORT"]
        self.TIME_STEP = _NSD["GENERAL"]["TIME_STEP"]

        self.FGAIN = _NSD["GLU"]["FGAIN"]
        self.PRESTIM = _NSD["GLU"]["PRESTIM"]

        MUSCLE_PARAMS = config.MUSCLE_DICTIONARY
        self.F_max_iso = MUSCLE_PARAMS['GLU']['F_max']

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
        force_feedback = self.force_buffer.pop(0)
        self.length_buffer.pop(0)

        if gait_phase == 'swing':
            glu_stim = self.PRESTIM + self.FGAIN * (force_feedback / self.F_max_iso)
            return glu_stim
        else:
            return self.PRESTIM
