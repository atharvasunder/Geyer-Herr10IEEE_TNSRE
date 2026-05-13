"""
Written by Atharva Sunder
Advised by Prof. Hartmut Geyer
October 2025
"""
from time import perf_counter
from abc import ABC, abstractmethod

class ControlMode(ABC):
    """
    Parent class for all gait control modes in the rigid body walking model.

    Each mode defines a phase of the gait cycle (e.g. single support, double support)
    and encapsulates the control logic and transition conditions for that phase.

    Every mode's controller is initialized with parameters in the control process,
    and added to the control_mode_manager class.

    attributes:
    mode_name: Name of mode ('SingleSupport', 'DoubleSupport')
    transition_targets: List of possible transition target modes from the current mode
    fsm: The control mode managing state machine class.

    """

    def __init__(self, fsm, mode_name, transition_targets):
        self.time_enter = None
        self.time_exit = None
        self.fsm = fsm

    @abstractmethod
    def enter(self, rbs_states):
        """Called when entering this gait phase; record entry time and initialize phase state."""
        pass

    @abstractmethod
    def check_transition(self):
        """Check if current rigid body states trigger a transition to another gait phase."""
        pass

    @abstractmethod
    def compute_control(self):
        """Compute joint torques for the current gait phase."""
        pass

    @abstractmethod
    def exit(self, rbs_states):
        """Called when leaving this gait phase; record exit time and finalize phase state."""
        pass
