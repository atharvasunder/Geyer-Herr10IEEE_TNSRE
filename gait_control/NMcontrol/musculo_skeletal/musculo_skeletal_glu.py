import numpy as np
import config
from gait_control.NMcontrol.musculo_skeletal.base_musculo_skeletal import BaseMusculoSkeletal

class musculo_skeletal_glu(BaseMusculoSkeletal):
    def __init__(self, interface_name, state_names, state_units, leg_name='leg_1'):
        super().__init__(interface_name, state_names, state_units, leg_name)
        # Ensure states is sized for 3 joints
        self.states = np.nan * np.zeros(3, dtype=np.float32)

        self.p = config.MUSCULO_SKELETAL_DICTIONARY["GLU"]
        self.l_slack = config.MUSCLE_DICTIONARY['GLU']['l_slack']
        self.l_opt = config.MUSCLE_DICTIONARY['GLU']['l_opt']

        self.glu_hip_lever = None

    def update_lmtc(self, rbs_states):
        """
        Returns updated l_mtc for gluteus (hip only). 
        Please note that for muscles that reduce l_mtc when the joint angle increases, 
        the radius is taken as negative.
        """
        hip_angle = rbs_states[self.leg_name]['q']['hip']  # [rad]

        dl = self._joint_lmtc_contribution(
            joint_angle_rad=hip_angle,
            r0=-self.p["RADIUS"],
            phi_ref=self.p["REF_ANGLE"],
            rho=self.p["PENNATION"],
            joint_type="hip"
        )

        return self.l_slack + self.l_opt + dl

    def get_lever_arms(self, rbs_states):
        """
        Returns updated lever arm for GLU (hip, knee, ankle).
        """
        hip_angle = rbs_states[self.leg_name]['q']['hip']  # [rad]

        self.glu_hip_lever = self._joint_lever_arm(
            joint_angle_rad=hip_angle,
            r0=self.p["RADIUS"],
            joint_type="hip"
        )

        self.set_states()
        return [self.glu_hip_lever, 0.0, 0.0]

    def set_states(self):
        self.states[:] = (self.glu_hip_lever, 0.0, 0.0)

    def get_torque_contributions(self, F_mtc):
        """
        GLU contribute POSITIVE hip torques.
        """
        hip_torque = F_mtc * self.glu_hip_lever
        return np.array([hip_torque, 0.0, 0.0])