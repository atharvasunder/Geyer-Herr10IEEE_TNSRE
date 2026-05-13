import numpy as np
import config
from gait_control.NMcontrol.musculo_skeletal.base_musculo_skeletal import BaseMusculoSkeletal

class musculo_skeletal_vas(BaseMusculoSkeletal):
    def __init__(self, interface_name, state_names, state_units, leg_name='leg_1'):
        super().__init__(interface_name, state_names, state_units, leg_name)
        # Ensure states is sized for 3 joints
        self.states = np.nan * np.zeros(3, dtype=np.float32)

        self.p = config.MUSCULO_SKELETAL_DICTIONARY["VAS"]
        self.l_slack = config.MUSCLE_DICTIONARY['VAS']['l_slack']
        self.l_opt = config.MUSCLE_DICTIONARY['VAS']['l_opt']

        self.vas_knee_lever = None

    def update_lmtc(self, rbs_states):
        """
        Returns updated l_mtc for vas. 
        Please note that for muscles that reduce l_mtc when the joint angle increases, 
        the radius is taken as negative.
        """
        knee_angle = rbs_states[self.leg_name]['q']['knee']  # [rad]

        dl = self._joint_lmtc_contribution(
            joint_angle_rad=knee_angle,
            r0=-self.p["RADIUS"],
            phi_ref=self.p["REF_ANGLE"],
            rho=self.p["PENNATION"],
            joint_type="knee",
            phi_max=self.p["PHI_MAX"]
        )

        return self.l_slack + self.l_opt + dl

    def get_lever_arms(self, rbs_states):
        """
        Returns updated lever arms for vas (hip, knee, ankle).
        """
        knee_angle = rbs_states[self.leg_name]['q']['knee']  # [rad]
        # print('knee_angle', knee_angle*180/np.pi)

        self.vas_knee_lever = self._joint_lever_arm(
            joint_angle_rad=knee_angle,
            r0=self.p["RADIUS"],
            joint_type="knee",
            phi_max=self.p["PHI_MAX"]
        )
        # print('self.vas_knee_lever', self.vas_knee_lever)

        self.set_states()
        return [0.0, self.vas_knee_lever, 0.0]

    def set_states(self):
        self.states[:] = (0.0, self.vas_knee_lever, 0.0)

    def get_torque_contributions(self, F_mtc):
        """
        Calculates the torque contribution from the VAS muscle.
        - VAS produces POSITIVE knee torques.
        """
        knee_torque = F_mtc * self.vas_knee_lever
        return np.array([0.0, knee_torque, 0.0])
