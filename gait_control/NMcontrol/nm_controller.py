"""
Written by Atharva Sunder
Advised by Prof. Hartmut Geyer
October 2025

Neuromuscular Controller
Computes joint torques for a given leg during stance or swing phase using Hill-type
muscle-tendon complexes, musculoskeletal lever arm models, and neural feedback loops.
"""

import numpy as np

import config
from gait_control.NMcontrol.mtc import muscle_tendon_unit

from gait_control.NMcontrol.musculo_skeletal.musculo_skeletal_vas import musculo_skeletal_vas
from gait_control.NMcontrol.musculo_skeletal.musculo_skeletal_sol import musculo_skeletal_sol
from gait_control.NMcontrol.musculo_skeletal.musculo_skeletal_gas import musculo_skeletal_gas
from gait_control.NMcontrol.musculo_skeletal.musculo_skeletal_ta import musculo_skeletal_ta
from gait_control.NMcontrol.musculo_skeletal.musculo_skeletal_ham import musculo_skeletal_ham
from gait_control.NMcontrol.musculo_skeletal.musculo_skeletal_glu import musculo_skeletal_glu
from gait_control.NMcontrol.musculo_skeletal.musculo_skeletal_hfl import musculo_skeletal_hfl

from gait_control.NMcontrol.feedback.feedback_vas import VasFeedback
from gait_control.NMcontrol.feedback.feedback_sol import SolFeedback
from gait_control.NMcontrol.feedback.feedback_gas import GasFeedback
from gait_control.NMcontrol.feedback.feedback_ta  import TaFeedback
from gait_control.NMcontrol.feedback.feedback_ham import HamFeedback
from gait_control.NMcontrol.feedback.feedback_glu import GluFeedback
from gait_control.NMcontrol.feedback.feedback_hfl import HflFeedback
from gait_control.NMcontrol.feedback.feedback_mixing import stim_feedback_mixing

class NMController:
    """
    Neuromuscular controller for the rigid body walking model.

    Manages muscle-tendon complexes (MTCs), musculoskeletal lever arm models,
    and neural feedback loops for both legs independently. Computes hip, knee,
    and ankle joint torques for a requested leg based on the current rigid body states.
    """

    def __init__(self, dt, tStop):

        # ---- Logging setup ----
        self.num_ctrl_steps = int(tStop / dt) + 1
        self.ctrl_step_idx = 0
        self.log_data_dict = {'leg_1': {}, 'leg_2': {}}

        # ---- Muscle-Tendon Complex (Hill-type muscle model) instances ----
        mtc_names = ["VAS", "SOL", "GAS", "TA", "HAM", "GLU", "HFL"]
        mtc_state_names = ["A", "F_mtc", "l_mtc", "l_ce"]
        mtc_state_units = ["-", "N", "m", "m"]

        self.mtc_dict = {
            'leg_1': [muscle_tendon_unit(name, mtc_state_names, mtc_state_units, dt) for name in mtc_names],
            'leg_2': [muscle_tendon_unit(name, mtc_state_names, mtc_state_units, dt) for name in mtc_names],
        }

        # ---- Musculoskeletal (lever arm / geometry) instances ----
        ms_states = ["hip_lever", "knee_lever", "ankle_lever"]
        ms_units = ["m", "m", "m"]

        self.musculoskeletal_dict = {
            'leg_1': [
                musculo_skeletal_vas("vas_1", ms_states, ms_units, leg_name='leg_1'),
                musculo_skeletal_sol("sol_1", ms_states, ms_units, leg_name='leg_1'),
                musculo_skeletal_gas("gas_1", ms_states, ms_units, leg_name='leg_1'),
                musculo_skeletal_ta("ta_1", ms_states, ms_units, leg_name='leg_1'),
                musculo_skeletal_ham("ham_1", ms_states, ms_units, leg_name='leg_1'),
                musculo_skeletal_glu("glu_1", ms_states, ms_units, leg_name='leg_1'),
                musculo_skeletal_hfl("hfl_1", ms_states, ms_units, leg_name='leg_1')
            ],
            'leg_2': [
                musculo_skeletal_vas("vas_2", ms_states, ms_units, leg_name='leg_2'),
                musculo_skeletal_sol("sol_2", ms_states, ms_units, leg_name='leg_2'),
                musculo_skeletal_gas("gas_2", ms_states, ms_units, leg_name='leg_2'),
                musculo_skeletal_ta("ta_2", ms_states, ms_units, leg_name='leg_2'),
                musculo_skeletal_ham("ham_2", ms_states, ms_units, leg_name='leg_2'),
                musculo_skeletal_glu("glu_2", ms_states, ms_units, leg_name='leg_2'),
                musculo_skeletal_hfl("hfl_2", ms_states, ms_units, leg_name='leg_2')
            ]
        }

        # ---- Neural feedback instances ----
        self.feedback_dict = {
            'leg_1': [VasFeedback(), SolFeedback(), GasFeedback(), TaFeedback(), HamFeedback(), GluFeedback(), HflFeedback()],
            'leg_2': [VasFeedback(), SolFeedback(), GasFeedback(), TaFeedback(), HamFeedback(), GluFeedback(), HflFeedback()]
        }

        # ---- Feedback mixing ----
        self.feedback_mixing_dict = {
            'leg_1': stim_feedback_mixing(self.feedback_dict['leg_1'], 'leg_1'),
            'leg_2': stim_feedback_mixing(self.feedback_dict['leg_2'], 'leg_2')
        }

        # ---- Joint torques [Nm] ----
        self.torques = {
            'leg_1': {'hip': 0.0, 'knee': 0.0, 'ankle': 0.0},
            'leg_2': {'hip': 0.0, 'knee': 0.0, 'ankle': 0.0},
        }

        # ---- Initialize Logging Dictionary ----
        for leg in ['leg_1', 'leg_2']:
            for m_name in mtc_names:
                self.log_data_dict[leg][m_name] = {}
                for state in mtc_state_names + ms_states:
                    self.log_data_dict[leg][m_name][state] = np.zeros(self.num_ctrl_steps)

    def compute_control(self, rbs_states, leg_name, gait_phase):
        """Compute neuromuscular joint torques for the specified leg and phase.

        Args:
            rbs_states: rigid body system states dictionary
            leg_name: 'leg_1' or 'leg_2'
            gait_phase: 'stance' or 'swing'

        Returns:
            tuple: (hip_torque, knee_torque, ankle_torque) for the given leg
        """

        # reset torques for this leg
        self.torques[leg_name]['hip'] = 0.0
        self.torques[leg_name]['knee'] = 0.0
        self.torques[leg_name]['ankle'] = 0.0

        # get this leg's objects
        mtc_list = self.mtc_dict[leg_name]
        ms_list = self.musculoskeletal_dict[leg_name]
        fb_list = self.feedback_dict[leg_name]
        fb_mixing = self.feedback_mixing_dict[leg_name]

        # extract states and push to mixing buffer
        fb_mixing.update_buffers(rbs_states)
        
        # compute muscle stimulations from feedback mixing for this leg
        stimulations = fb_mixing.compute_stimulations(rbs_states, gait_phase)

        for i in range(len(mtc_list)):
            l_mtc = ms_list[i].update_lmtc(rbs_states)
            stimulation = stimulations[i]
            F_mtc, l_ce = mtc_list[i].update(stimulation, l_mtc)
            fb_list[i].update_buffer(F_mtc, l_ce)
            ms_list[i].get_lever_arms(rbs_states)
            torques = ms_list[i].get_torque_contributions(F_mtc)
            self.torques[leg_name]['hip'] += torques[0]
            self.torques[leg_name]['knee'] += torques[1]
            self.torques[leg_name]['ankle'] += torques[2]

        # Scale joint torques by subject body weight
        body_weight = config.MODEL_PARAMS["body_weight"]
        self.torques[leg_name]['hip'] *= body_weight
        self.torques[leg_name]['knee'] *= body_weight
        self.torques[leg_name]['ankle'] *= body_weight

        return (self.torques[leg_name]['hip'],
                self.torques[leg_name]['knee'],
                self.torques[leg_name]['ankle'])

    def reset(self, rbs_states, leg_name):
        """Reset the neuromuscular controller state for the specified leg."""
        self.torques[leg_name]['hip'] = 0.0
        self.torques[leg_name]['knee'] = 0.0
        self.torques[leg_name]['ankle'] = 0.0

        mtc_list = self.mtc_dict[leg_name]
        ms_list = self.musculoskeletal_dict[leg_name]
        fb_list = self.feedback_dict[leg_name]
        fb_mixing = self.feedback_mixing_dict[leg_name]

        for i in range(len(mtc_list)):
            l_mtc = ms_list[i].update_lmtc(rbs_states)
            l_ce = mtc_list[i].reset(l_mtc)
            fb_list[i].reset(l_ce)

        fb_mixing.reset(rbs_states)

    def log_data(self):
        """Update log dictionary with current states for both legs."""
        if self.ctrl_step_idx >= self.num_ctrl_steps:
            return

        mtc_state_names = ["A", "F_mtc", "l_mtc", "l_ce"]
        ms_states = ["hip_lever", "knee_lever", "ankle_lever"]
        mtc_names = ["VAS", "SOL", "GAS", "TA", "HAM", "GLU", "HFL"]

        for leg in ['leg_1', 'leg_2']:
            mtc_list = self.mtc_dict[leg]
            ms_list = self.musculoskeletal_dict[leg]

            for i, m_name in enumerate(mtc_names):
                for j, state in enumerate(mtc_state_names):
                    self.log_data_dict[leg][m_name][state][self.ctrl_step_idx] = mtc_list[i].states[j]
                for j, state in enumerate(ms_states):
                    self.log_data_dict[leg][m_name][state][self.ctrl_step_idx] = ms_list[i].states[j]

        self.ctrl_step_idx += 1