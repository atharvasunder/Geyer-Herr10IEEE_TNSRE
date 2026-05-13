import numpy as np
from gait_control.NMcontrol.feedback.base_feedback import BaseFeedback
import config

class TaFeedback(BaseFeedback):
    """
    Feedback pathway for the Tibialis Anterior (TA) muscle.
    """

    def __init__(self):
        super().__init__()
        
        _NSD = config.NERVOUS_SYSTEM_DICTIONARY
        self.LONG_DELAY = _NSD["DELAY"]["LONG"]
        self.TIME_STEP = _NSD["GENERAL"]["TIME_STEP"]
        
        self.FGAIN = _NSD["TA"]["FGAIN"]
        self.LCEOFF = _NSD["TA"]["LCEOFF"]
        self.PRESTIM = _NSD["TA"]["PRESTIM"]
        
        MUSCLE_PARAMS = config.MUSCLE_DICTIONARY
        self.l_opt = MUSCLE_PARAMS['TA']['l_opt']

        self.length_buffer = None
        self.force_buffer = None

    def reset(self, l_ce):
        size = max(1, int(self.LONG_DELAY / self.TIME_STEP))
        self.length_buffer = [l_ce] * size
        self.force_buffer  = [0.0] * size

    def update_buffer(self, fmtc, l_ce):
        self.force_buffer.append(fmtc)
        self.length_buffer.append(l_ce)

    def compute_stimulation(self, gait_phase):
        self.force_buffer.pop(0)
        l_ce_t = self.length_buffer.pop(0)
        
        if gait_phase == 'stance':
            # l_CE feedback is S0 + G_TA * (l_CE - l_off)
            ta_stim = self.PRESTIM + self.FGAIN * (l_ce_t / self.l_opt - self.LCEOFF)
            return ta_stim
        else:
            # l_CE feedback is S0 + G_TA * (l_CE - l_off)
            ta_stim = self.PRESTIM + self.FGAIN * (l_ce_t / self.l_opt - self.LCEOFF)
            return ta_stim
