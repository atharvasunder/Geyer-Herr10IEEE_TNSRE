import numpy as np
import config
from gait_control.NMcontrol.musculo_skeletal.base_musculo_skeletal import BaseMusculoSkeletal

class musculo_skeletal_ham(BaseMusculoSkeletal):
    def __init__(self, interface_name, state_names, state_units, leg_name='leg_1'):
        super().__init__(interface_name, state_names, state_units, leg_name)
        # Ensure states is sized for 3 joints
        self.states = np.nan * np.zeros(3, dtype=np.float32)

        self.p_knee = config.MUSCULO_SKELETAL_DICTIONARY["HAM_KNEE"]
        self.p_hip = config.MUSCULO_SKELETAL_DICTIONARY["HAM_HIP"]
        self.l_slack = config.MUSCLE_DICTIONARY['HAM']['l_slack']
        self.l_opt = config.MUSCLE_DICTIONARY['HAM']['l_opt']

        self.ham_knee_lever = None
        self.ham_hip_lever = None

    def update_lmtc(self, rbs_states):
        """
        Returns updated l_mtc for hamstrings (bi-articular: knee + hip).
        Please note that for muscles that reduce l_mtc when the joint angle increases, 
        the radius is taken as negative.
        """
        knee_angle = rbs_states[self.leg_name]['q']['knee']  # [rad]
        hip_angle = rbs_states[self.leg_name]['q']['hip']    # [rad]

        dl_knee = self._joint_lmtc_contribution(
            joint_angle_rad=knee_angle,
            r0=self.p_knee["RADIUS"],
            phi_ref=self.p_knee["REF_ANGLE"],
            rho=self.p_knee["PENNATION"],
            joint_type="knee",
            phi_max=self.p_knee["PHI_MAX"]
        )

        dl_hip = self._joint_lmtc_contribution(
            joint_angle_rad=hip_angle,
            r0=-self.p_hip["RADIUS"],
            phi_ref=self.p_hip["REF_ANGLE"],
            rho=self.p_hip["PENNATION"],
            joint_type="hip"
        )

        return self.l_slack + self.l_opt + dl_knee + dl_hip

    def get_lever_arms(self, rbs_states):
        """
        Returns updated lever arms for hamstrings (hip, knee, ankle).
        """
        knee_angle = rbs_states[self.leg_name]['q']['knee']  # [rad]
        hip_angle = rbs_states[self.leg_name]['q']['hip']    # [rad]

        self.ham_knee_lever = self._joint_lever_arm(
            joint_angle_rad=knee_angle,
            r0=self.p_knee["RADIUS"],
            joint_type="knee",
            phi_max=self.p_knee["PHI_MAX"]
        )

        self.ham_hip_lever = self._joint_lever_arm(
            joint_angle_rad=hip_angle,
            r0=self.p_hip["RADIUS"],
            joint_type="hip"
        )

        self.set_states()
        return [self.ham_hip_lever, self.ham_knee_lever, 0.0]

    def set_states(self):
        self.states[:] = (self.ham_hip_lever, self.ham_knee_lever, 0.0)

    def get_torque_contributions(self, F_mtc):
        """
        HAM contributes:
        - NEGATIVE knee torque
        - POSITIVE hip torque
        """
        hip_torque = F_mtc * self.ham_hip_lever
        knee_torque = -F_mtc * self.ham_knee_lever
        return np.array([hip_torque, knee_torque, 0.0])