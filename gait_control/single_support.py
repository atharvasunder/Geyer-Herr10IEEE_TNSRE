"""
Written by Atharva Sunder
Advised by Prof. Hartmut Geyer
October 2025
"""

from time import perf_counter
from gait_control.base_ctrl_class import ControlMode
import numpy as np

class SingleSupportControl(ControlMode):
    """
    Control mode for the single support phase of the gait cycle.

    During single support, one leg is planted on the ground (stance leg) while
    the other swings forward (swing leg). Neuromuscular stance control is applied
    to the stance leg, and neuromuscular swing control to the swing leg. The phase
    ends when the swing leg makes ground contact (landing).

    attributes (inherited):
    mode_name: 'SingleSupport'
    transition_targets: ['DoubleSupport']
    fsm: The control mode managing state machine class.
    """

    def __init__(self, fsm, mode_name, transition_targets, nm_controller):
        self.time_enter = None
        self.duration = None
        self.fsm = fsm
        self.nm_controller = nm_controller

        self.name = mode_name
        self.transition_targets = transition_targets

        # joint torques for both legs (hip, knee, ankle per leg)
        self.torques = {
            'hip_1': 0.0, 'knee_1': 0.0, 'ankle_1': 0.0,
            'hip_2': 0.0, 'knee_2': 0.0, 'ankle_2': 0.0,
        }

        # leg role assignments (updated on phase entry)
        self.stance_leg = None  # name/identifier of the leg in ground contact
        self.swing_leg = None   # name/identifier of the leg swinging forward

    def enter(self, rbs_states, transition_leg):
        """Called when entering single support; record entry time and determine which leg is stance and which is swing."""
        # print(f"Entering Single Support: transition leg = {transition_leg}")
        self.time_enter = perf_counter()
        if transition_leg is None:
            self.swing_leg = 'leg_1'
            self.stance_leg = 'leg_2'
        else:
            self.swing_leg = transition_leg
            self.stance_leg = 'leg_1' if transition_leg == 'leg_2' else 'leg_2'
        
        # Set Dsup and phase flags
        rbs_states[self.stance_leg]['Dsup'] = 0
        rbs_states[self.swing_leg]['Dsup'] = 0
        rbs_states[self.stance_leg]['phase'] = 'stance'
        rbs_states[self.swing_leg]['phase'] = 'swing'

    def check_transition(self, rbs_states):
        """Check if swing leg GRFs indicate landing to transition to DoubleSupport.
        Returns (next_mode_name, transition_leg) or (None, transition_leg) if no transition."""
        swing_grf = rbs_states[self.swing_leg]['grf']
        swing_hip_angle = rbs_states[self.swing_leg]['q']['hip']
        
        if swing_grf > 500 and swing_hip_angle < np.pi and (-self.time_enter + perf_counter()) > 0.1:
            print("transitioning to double support at time", perf_counter(), "with swing leg", self.swing_leg, "with grf", swing_grf, "with hip angle", swing_hip_angle)
            return 'DoubleSupport', self.swing_leg
        else:
            return None, self.swing_leg

    def compute_control(self, rbs_states):
        """Compute NM stance torques for the stance leg and NM swing torques for the swing leg."""
        
        # Call NMController for stance leg (gait_phase='stance') and swing leg (gait_phase='swing'), populate self.torques
        st_idx, sw_idx = self.stance_leg[-1], self.swing_leg[-1]    # st_idx = 1/2, sw_idx = 2/1
        
        self.torques[f'hip_{st_idx}'], self.torques[f'knee_{st_idx}'], self.torques[f'ankle_{st_idx}'] = self.nm_controller.compute_control(rbs_states, self.stance_leg, 'stance')
        self.torques[f'hip_{sw_idx}'], self.torques[f'knee_{sw_idx}'], self.torques[f'ankle_{sw_idx}'] = self.nm_controller.compute_control(rbs_states, self.swing_leg, 'swing')

        return np.array([self.torques['hip_1'], self.torques['knee_1'], self.torques['ankle_1'],
                         self.torques['hip_2'], self.torques['knee_2'], self.torques['ankle_2']])

    def exit(self, rbs_states):
        """Called when leaving single support; record phase duration."""
        self.duration = perf_counter() - self.time_enter
        return self.duration
