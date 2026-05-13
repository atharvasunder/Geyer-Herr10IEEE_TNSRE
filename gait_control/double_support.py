"""
Written by Atharva Sunder
Advised by Prof. Hartmut Geyer
October 2025
"""

from time import perf_counter
from gait_control.base_ctrl_class import ControlMode
import numpy as np

class DoubleSupportControl(ControlMode):
    """
    Control mode for the double support phase of the gait cycle.

    During double support, both feet are in ground contact. Neuromuscular stance
    control is applied to both the heading and trailing legs. The phase ends
    when the trailing leg's ground reaction forces drop below threshold (toe-off).

    attributes (inherited):
    mode_name: 'DoubleSupport'
    transition_targets: ['SingleSupport']
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

        # leg role assignments (updated on phase entry based on which leg is leading)
        self.heading_leg = None  # name/identifier of the leg heading the stride
        self.trailing_leg = None  # name/identifier of the leg trailing (about to toe-off)

    def enter(self, rbs_states, transition_leg):
        """Called when entering double support; record entry time and reasons for which leg is heading and trailing"""
        # print(f"Entering Double Support: transition leg = {transition_leg}")
        self.time_enter = perf_counter()
        self.heading_leg = transition_leg   # leg that just landed
        self.trailing_leg = 'leg_1' if transition_leg == 'leg_2' else 'leg_2'

        # Set Dsup and phase flags
        rbs_states[self.heading_leg]['Dsup'] = 0
        rbs_states[self.trailing_leg]['Dsup'] = 1
        rbs_states[self.heading_leg]['phase'] = 'stance'
        rbs_states[self.trailing_leg]['phase'] = 'stance'

    def check_transition(self, rbs_states):
        """Check if trailing leg GRFs indicate toe-off to transition to SingleSupport.
        Returns (next_mode_name, transition_leg) or (None, transition_leg) if no transition."""
        trailing_grf = rbs_states[self.trailing_leg]['grf']
        
        # Get the ankle torque of the trailing leg from the previous timestep
        t_idx = self.trailing_leg[-1]
        trailing_ankle_torque = self.torques[f'ankle_{t_idx}']

        if trailing_grf < 100 and trailing_ankle_torque < 2.5 and (-self.time_enter + perf_counter()) > 0.1:
            print("transitioning to single support at time", perf_counter(), "with trailing leg", self.trailing_leg, "with grf", trailing_grf, "with ankle torque", trailing_ankle_torque)
            return 'SingleSupport', self.trailing_leg
        else:
            return None, self.trailing_leg

    def compute_control(self, rbs_states):
        """Compute neuromuscular stance torques for both heading and trailing legs."""
        
        # Call NMController for both legs (gait_phase='stance') and populate self.torques
        h_idx, t_idx = self.heading_leg[-1], self.trailing_leg[-1]
        
        self.torques[f'hip_{h_idx}'], self.torques[f'knee_{h_idx}'], self.torques[f'ankle_{h_idx}'] = self.nm_controller.compute_control(rbs_states, self.heading_leg, 'stance')
        self.torques[f'hip_{t_idx}'], self.torques[f'knee_{t_idx}'], self.torques[f'ankle_{t_idx}'] = self.nm_controller.compute_control(rbs_states, self.trailing_leg, 'stance')

        return np.array([self.torques['hip_1'], self.torques['knee_1'], self.torques['ankle_1'],
                         self.torques['hip_2'], self.torques['knee_2'], self.torques['ankle_2']])

    def exit(self, rbs_states):
        """Called when leaving double support; record phase duration."""
        self.duration = perf_counter() - self.time_enter
        return self.duration
