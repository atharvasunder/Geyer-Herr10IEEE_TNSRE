import math
import numpy as np
from abc import ABC, abstractmethod

class BaseMusculoSkeletal(ABC):
    """
    Abstract base class for musculo-skeletal geometry.

    Uses the Appendix III equations:
      - hip:
            r_m(phi) = r0
            Δl_mtu = rho * r0 * (phi - phi_ref)

      - ankle/knee:
            r_m(phi) = r0 * cos(phi - phi_max)
            Δl_mtu = rho * r0 * [sin(phi - phi_max) - sin(phi_ref - phi_max)]
    """

    def __init__(self, interface_name, state_names, state_units, leg_name='leg_1'):
        self.states = np.nan * np.zeros(len(state_names), dtype=np.float32)
        self.state_names = state_names
        self.state_units = state_units
        self.interface_name = interface_name
        self.leg_name = leg_name

    # ------------------------------------------------------------------ #
    # Shared helpers for per-joint musculoskeletal geometry
    # ------------------------------------------------------------------ #

    @staticmethod
    def _joint_lmtc_contribution(joint_angle_rad,
                                 r0,
                                 phi_ref,
                                 rho,
                                 joint_type,
                                 phi_max=None):
        """
        Compute the MTU length-change contribution from one joint.

        Args:
            joint_angle_rad (float): Current joint angle [rad]
            r0 (float): Moment-arm scale from table [m]
            phi_ref (float): Reference angle where l_mtu = l_opt + l_slack [rad]
            rho (float): Pennation scaling factor
            joint_type (str): 'hip', 'knee', or 'ankle'
            phi_max (float or None): Angle of maximal lever arm [rad].
                                     Required for ankle/knee, ignored for hip.

        Returns:
            float: Δl_mtu contribution [m]
        """
        if joint_type == "hip":
            return rho * r0 * (joint_angle_rad - phi_ref)

        elif joint_type in ("ankle", "knee"):
            if phi_max is None:
                raise ValueError(f"phi_max is required for joint_type='{joint_type}'")
            return rho * r0 * (
                math.sin(joint_angle_rad - phi_max)
                - math.sin(phi_ref - phi_max)
            )

        else:
            raise ValueError(f"Unsupported joint_type: {joint_type}")

    @staticmethod
    def _joint_lever_arm(joint_angle_rad,
                         r0,
                         joint_type,
                         phi_max=None):
        """
        Compute the instantaneous lever arm for one joint.

        Args:
            joint_angle_rad (float): Current joint angle [rad]
            r0 (float): Moment-arm scale from table [m] (positive for all joints)
            joint_type (str): 'hip', 'knee', or 'ankle'
            phi_max (float or None): Angle of maximal lever arm [rad].
                                     Required for ankle/knee, ignored for hip.

        Returns:
            float: Lever arm [m]
        """
        if joint_type == "hip":
            return r0

        elif joint_type in ("ankle", "knee"):
            if phi_max is None:
                raise ValueError(f"phi_max is required for joint_type='{joint_type}'")
        
            return r0 * math.cos(joint_angle_rad - phi_max)

        else:
            raise ValueError(f"Unsupported joint_type: {joint_type}")

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    def update_lmtc(self, rbs_states):
        pass

    @abstractmethod
    def get_lever_arms(self, rbs_states):
        pass

    @abstractmethod
    def get_torque_contributions(self, F_mtc):
        pass