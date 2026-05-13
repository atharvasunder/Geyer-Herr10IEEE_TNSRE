"""
Written by Atharva Sunder
Advised by Prof. Hartmut Geyer
October 2025
"""

class ctrl_manager():
    """
    Finite State Machine for rigid body neuromuscular walking model control.
    Manages transitions between gait phases:
        - SingleSupport: one foot is in ground contact; the other leg swings forward.
        - DoubleSupport: both feet are in ground contact during the weight transfer phase.
    """

    def __init__(self):
        self.ctrl_modes = {}  # all state classes
        self.stop_flag = False # flag to stop the control
        self.curr_ctrl_mode = None  # [class object]
        self.curr_ctrl_mode_name = None  # [string]
        self.prev_ctrl_mode_name = None  # [string]
        self.next_ctrl_mode_name = None  # the target mode for a transition, stores that mode`s name.

    def add_ctrl_mode(self, ctrl_mode_name, ctrl_mode):
        """
        Register a gait phase controller under the given name.
        """
        self.ctrl_modes[ctrl_mode_name] = ctrl_mode

    def set_ctrl_mode(self, new_mode_name, rbs_states, transition_leg):
        """
        Transition to a new gait phase: update bookkeeping and call the phase's enter() method.
        """
        self.prev_ctrl_mode_name = self.curr_ctrl_mode_name
        self.curr_ctrl_mode_name = new_mode_name
        self.curr_ctrl_mode = self.ctrl_modes[new_mode_name]
        self.curr_ctrl_mode.enter(rbs_states, transition_leg)

    def get_ctrl(self, rbs_states):
        """
        Check if a gait phase transition is needed based on current rigid body states.
        If yes, execute the transition. In any case, compute and return the control
        command for the current timestep.
        """

        # transition leg is the leg that triggers the phase change: the swing leg for DS to SS, 
        # and the stance leg for SS to DS
        self.next_ctrl_mode_name, transition_leg = self.curr_ctrl_mode.check_transition(rbs_states)

        if self.next_ctrl_mode_name is not None:    # if a transition is triggered to another phase
            self.curr_ctrl_mode.exit(rbs_states)
            self.set_ctrl_mode(self.next_ctrl_mode_name, rbs_states, transition_leg)
            self.next_ctrl_mode_name = None     # reset once the mode transition is complete

        ctrl_cmd = self.curr_ctrl_mode.compute_control(rbs_states)
        ctrl_mode = self.curr_ctrl_mode_name
        
        return ctrl_mode, ctrl_cmd
        
    def get_current_state(self):
        """
        Return the name of the current gait phase.
        """
        return self.curr_ctrl_mode_name
