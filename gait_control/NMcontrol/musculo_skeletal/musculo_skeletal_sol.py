import numpy as np
import config
from gait_control.NMcontrol.musculo_skeletal.base_musculo_skeletal import BaseMusculoSkeletal

class musculo_skeletal_sol(BaseMusculoSkeletal):
    def __init__(self, interface_name, state_names, state_units, leg_name='leg_1'):
        super().__init__(interface_name, state_names, state_units, leg_name)
        # Ensure states is sized for 3 joints
        self.states = np.nan * np.zeros(3, dtype=np.float32)

        self.p = config.MUSCULO_SKELETAL_DICTIONARY["SOL"]
        self.l_slack = config.MUSCLE_DICTIONARY['SOL']['l_slack']
        self.l_opt = config.MUSCLE_DICTIONARY['SOL']['l_opt']

        self.sol_ankle_lever = None

    def update_lmtc(self, rbs_states):
        """
        Returns updated l_mtc for sol. 
        Please note that for muscles that reduce l_mtc when the joint angle increases, 
        the radius is taken as negative.
        """
        ankle_angle = rbs_states[self.leg_name]['q']['ankle']  # [rad]

        dl = self._joint_lmtc_contribution(
            joint_angle_rad=ankle_angle,
            r0=-self.p["RADIUS"],
            phi_ref=self.p["REF_ANGLE"],
            rho=self.p["PENNATION"],
            joint_type="ankle",
            phi_max=self.p["PHI_MAX"]
        )

        return self.l_slack + self.l_opt + dl

    def get_lever_arms(self, rbs_states):
        """
        Returns updated lever arms for sol (hip, knee, ankle).
        """
        ankle_angle = rbs_states[self.leg_name]['q']['ankle']  # [rad]

        self.sol_ankle_lever = self._joint_lever_arm(
            joint_angle_rad=ankle_angle,
            r0=self.p["RADIUS"],
            joint_type="ankle",
            phi_max=self.p["PHI_MAX"]
        )

        self.set_states()
        return [0.0, 0.0, self.sol_ankle_lever]

    def set_states(self):
        self.states[:] = (0.0, 0.0, self.sol_ankle_lever)

    def get_torque_contributions(self, F_mtc):
        """
        Calculates the torque contribution from the SOL muscle.
        - SOL produces POSITIVE ankle torques.
        """
        ankle_torque = F_mtc * self.sol_ankle_lever
        return np.array([0.0, 0.0, ankle_torque])