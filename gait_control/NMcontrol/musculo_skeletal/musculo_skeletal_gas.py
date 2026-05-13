import numpy as np
import config
from gait_control.NMcontrol.musculo_skeletal.base_musculo_skeletal import BaseMusculoSkeletal

class musculo_skeletal_gas(BaseMusculoSkeletal):
    def __init__(self, interface_name, state_names, state_units, leg_name='leg_1'):
        super().__init__(interface_name, state_names, state_units, leg_name)
        # Ensure states is sized for 3 joints
        self.states = np.nan * np.zeros(3, dtype=np.float32)

        self.p_knee = config.MUSCULO_SKELETAL_DICTIONARY["GAS_KNEE"]
        self.p_ankle = config.MUSCULO_SKELETAL_DICTIONARY["GAS_ANKLE"]
        self.l_slack = config.MUSCLE_DICTIONARY['GAS']['l_slack']
        self.l_opt = config.MUSCLE_DICTIONARY['GAS']['l_opt']

        self.gas_knee_lever = None
        self.gas_ankle_lever = None

    def update_lmtc(self, rbs_states):
        """
        Returns updated l_mtc for gas (bi-articular: knee + ankle). 
        Please note that for muscles that reduce l_mtc when the joint angle increases, 
        the radius is taken as negative.
        """
        knee_angle  = rbs_states[self.leg_name]['q']['knee']   # [rad]
        ankle_angle = rbs_states[self.leg_name]['q']['ankle']  # [rad]

        dl_knee = self._joint_lmtc_contribution(
            joint_angle_rad=knee_angle,
            r0=self.p_knee["RADIUS"],
            phi_ref=self.p_knee["REF_ANGLE"],
            rho=self.p_knee["PENNATION"],
            joint_type="knee",
            phi_max=self.p_knee["PHI_MAX"]
        )

        dl_ankle = self._joint_lmtc_contribution(
            joint_angle_rad=ankle_angle,
            r0=-self.p_ankle["RADIUS"],
            phi_ref=self.p_ankle["REF_ANGLE"],
            rho=self.p_ankle["PENNATION"],
            joint_type="ankle",
            phi_max=self.p_ankle["PHI_MAX"]
        )

        return self.l_slack + self.l_opt + dl_knee + dl_ankle

    def get_lever_arms(self, rbs_states):
        """
        Returns updated lever arms for gas (hip, knee, ankle).
        """
        knee_angle  = rbs_states[self.leg_name]['q']['knee']   # [rad]
        ankle_angle = rbs_states[self.leg_name]['q']['ankle']  # [rad]

        self.gas_knee_lever = self._joint_lever_arm(
            joint_angle_rad=knee_angle,
            r0=self.p_knee["RADIUS"],
            joint_type="knee",
            phi_max=self.p_knee["PHI_MAX"]
        )

        self.gas_ankle_lever = self._joint_lever_arm(
            joint_angle_rad=ankle_angle,
            r0=self.p_ankle["RADIUS"],
            joint_type="ankle",
            phi_max=self.p_ankle["PHI_MAX"]
        )

        self.set_states()
        return [0.0, self.gas_knee_lever, self.gas_ankle_lever]

    def set_states(self):
        self.states[:] = (0.0, self.gas_knee_lever, self.gas_ankle_lever)

    def get_torque_contributions(self, F_mtc):
        """
        Calculates the torque contribution from the GAS muscle.
        - GAS produces POSITIVE ankle and NEGATIVE knee torques.
        """
        knee_torque = -F_mtc * self.gas_knee_lever
        ankle_torque = F_mtc * self.gas_ankle_lever
        return np.array([0.0, knee_torque, ankle_torque])
