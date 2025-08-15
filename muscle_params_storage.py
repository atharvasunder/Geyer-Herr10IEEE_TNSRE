# File for storing params classes for bodies, muscles, joints, nervous system 
import math
import numpy as np

from dataclasses import dataclass

# @dataclass
# class general_params:
#     # integration params
#     dt: float  # (s) integration time step
#     tStop: float  # (s) simulation stop time

@dataclass
class muscle_params:
    # Force-length relationship
    F_max: float # (N) max force MTC can produce
    l_opt: float # (m) peak of force-length relationship
    w: float # param for the gaussian shaped F-l curve
    c : float # param for the gaussian shaped F-l curve

    # Force-velocity relationship
    v_max: float # (m/s) max shortening velocity of muscle
    N: float # eccentric force enhancement
    K: float # curvature constant

    # Tendon properties
    l_rest: float # (m) rest length of tendon
    e_ref_see: float # reference strain for tendon force calculation

    # activation parameters
    tau: float  # (s) excitation contraction coupling constant

    # mtc properties
    l_ref_mtc: float # MTC reference length, when the joint angles are q_ref (equals l_rest + l_opt)

@dataclass
class nervous_params:
    # feedback params
    delP_list: list  # (s) feedback time delay (list of delP's: long, medium, short)

    # Parameters for feedback-based stimulation
    stim0: float # Constant stimulation received by muscle
    G: float # gain for force feedback
    G_l: float # gain for length feedback
    G_l_2: float # gain for length feedback if there are 2 length feedbacks required 
    k_p: float # gain for Proportional feedback
    k_d: float # gain for derivative feedback
    k_lean: float # for hfl swing feedback
    l_off: float # offset for length feedback
    l_off_2: float # offset if there are 2 length feedbacks required
    p_trunk_ref: float # reference angle for hip
    k_bw: float # gain for grf feeback on the hip actuating muscles
    delta_S: float # stimulation for the hip activating muscles to help start swing

@dataclass
class musculo_skeletal_params:
    # musculoskeletal properties
    r0_array: np.ndarray # (m) moment arm max values
    q_ref_array: np.ndarray # (rad) reference angle values (joint angle at which MTU length = l_opt + l_slack)
    q_max_array: np.ndarray # (rad) max moment arm angle values (joint angle at which moment arm is maximum)
    rho_array: np.ndarray # ensures that the MTU fiber length stays within physiological limits

    # mtc properties
    l_ref_mtc:float # MTC reference length, when the joint angles are q_ref (equals l_rest + l_opt)