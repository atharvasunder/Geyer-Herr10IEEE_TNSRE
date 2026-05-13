"""
Rigid Body Walker Configuration

This configuration file defines a rigid body walker and assigns all
relevant kinematic and dynamic properties. The properties need to be
configured by the user in this file.

"""

import math

pi = math.pi

CONTACT_PARAMS = {
    'stiffness_x': 8200.0,
    'max_vx': 0.03,
    'stiffness_z': 81500.0,
    'max_vz': 0.03, # 0.03
    'mu_slide': 0.8,
    'v_transition': 0.01,
    'mu_stick': 0.9,
}

############### parameters ##############

MODEL_PARAMS = {
    'NAME': 'Human Neuromuscular Walking Model',
    
    # joint spring-damper parameters
    'DAMPING': 4000,
    'STIFFNESS': 1 * 4000**2 / 100, # critical damping?

    # joint angle limit torque parameters
    'k_lim': 0.3 * 180 / pi,            # [Nm/rad] mechanical limits stiffness
    'q_dot_max': 1 * pi / 180,          # [rad/s] mechanical limits damping

    # joint angle limits
    'q_max_hip': 230 * pi / 180,
    'q_min_hip': 20 * pi / 180, 

    'q_max_knee': 175 * pi / 180,
    'q_min_knee': 10 * pi / 180,

    'q_max_ankle': 130 * pi / 180,
    'q_min_ankle': 70 * pi / 180,

    # physical dimensions of bodies [m]
    'trunk_length': 0.8,
    'trunk_width': 0.1,
    
    'thigh_length': 0.5,
    'thigh_width': 0.06,

    'shank_length': 0.5,
    'shank_width': 0.06,

    'foot_length': 0.2, 
    'foot_width': 0.02,

    # masses [Kg] and inertia about -y axis [Kgm^2] of bodies 
    'm_trunk': 53.5,
    'I_trunk': 3,
    'm_thigh': 8.5,
    'I_thigh': 0.15,
    'm_shank': 3.5,
    'I_shank': 0.05,
    'm_foot': 1.25,
    'I_foot': 0.005,
    'g': 9.81,

    # COM locations from proximal joint of body
    # (assumes there is a frame parallel to the body frame at these joint locations)
    'trunk_com_prox': (0, -0.45),
    'thigh_com_prox': (0, 0.30),
    'shank_com_prox': (0, -0.30),
    'foot_com_prox': (0, 0.06),

    # site locations from COM of body
    'trunk_thigh_site': (0, -0.35),
    'thigh_trunk_site': (0, -0.30),
    'thigh_shank_site': (0, 0.20),
    'shank_thigh_site': (0, 0.30),
    'shank_foot_site': (0, -0.20),
    'foot_shank_site': (0, -0.06),

    # geometric centers of bodies from com (written in com frame)
    'gc_trunk': (0, 0.05),
    'gc_thigh': (0, -0.05),
    'gc_shank': (0, 0.05),
    'gc_foot': (0, 0.0),
}

# Calculated parameters based on primitive values
MODEL_PARAMS['body_weight'] = MODEL_PARAMS['g'] * (MODEL_PARAMS['m_trunk'] + 2 * (MODEL_PARAMS['m_thigh'] + MODEL_PARAMS['m_shank'] + MODEL_PARAMS['m_foot']))

############### initial conditions ##############

INITIAL_CONDITIONS = {
    'x0': 0.0,          # [m] inital forward (along horizontal axis) position of trunk CoM
    'h0': 1.33,         # [m] initial height (along vertical axis) of trunk CoM
    'p0': 0.0,          # [rad] initial orientation of trunk CoM wrt global z-axis (counterclockwise positive)

    'vx0': 1.3,         # [m/s] initial forward (along horizontal axis) velocity of trunk CoM
    'vz0': 0.0,         # [m/s] initial vertical velocity of trunk
    'vp0': 0.0,         # [rad/s] initial ang velocity of trunk 

    # initial joint angles defined between bodies from base to mate body of joint [rad]
    'initial_hip_1_angle': 175 * pi / 180,      
    'initial_knee_1_angle': 175 * pi / 180,     
    'initial_ankle_1_angle': 85 * pi / 180,    

    'initial_hip_2_angle': 156 * pi / 180,  
    'initial_knee_2_angle': 165 * pi / 180, 
    'initial_ankle_2_angle': 90 * pi / 180,
}

############### musculoskeletal parameters ##############

MUSCULO_SKELETAL_DICTIONARY = {

    "subject_weight": 80,   # unchanged

    # ---------------- ANKLE ----------------
    "SOL": {
        "RADIUS": 0.05,  # m
        "PHI_MAX": math.radians(110),
        "REF_ANGLE": math.radians(80),
        "PENNATION": 0.5
    },

    "TA": {
        "RADIUS": 0.04,
        "PHI_MAX": math.radians(80),
        "REF_ANGLE": math.radians(110),
        "PENNATION": 0.7
    },

    "GAS_ANKLE": {
        "RADIUS": 0.05,
        "PHI_MAX": math.radians(110),
        "REF_ANGLE": math.radians(80),
        "PENNATION": 0.7
    },

    # ---------------- KNEE ----------------
    "GAS_KNEE": {
        "RADIUS": 0.05,
        "PHI_MAX": math.radians(140),
        "REF_ANGLE": math.radians(165),
        "PENNATION": 0.7
    },

    "VAS": {
        "RADIUS": 0.06,
        "PHI_MAX": math.radians(165),       # 165
        "REF_ANGLE": math.radians(125),
        "PENNATION": 0.7
    },

    "HAM_KNEE": {
        "RADIUS": 0.05,
        "PHI_MAX": math.radians(180),
        "REF_ANGLE": math.radians(180),
        "PENNATION": 0.7
    },

    # ---------------- HIP ----------------
    "HAM_HIP": {
        "RADIUS": 0.08,
        "REF_ANGLE": math.radians(155),
        "PENNATION": 0.7
    },

    "GLU": {
        "RADIUS": 0.10,
        "REF_ANGLE": math.radians(150),
        "PENNATION": 0.5
    },

    "HFL": {
        "RADIUS": 0.10,
        "REF_ANGLE": math.radians(180),
        "PENNATION": 0.5
    }
}

# divided by body weight just for the feedback gains
# later torque computed is multiplied by the body weight
F_max_sol = 4000 / MODEL_PARAMS['body_weight']
F_max_ta  =  800 / MODEL_PARAMS['body_weight']
F_max_gas = 1500 / MODEL_PARAMS['body_weight']
F_max_vas = 6000 / MODEL_PARAMS['body_weight']
F_max_ham = 3000 / MODEL_PARAMS['body_weight']
F_max_glu = 1500 / MODEL_PARAMS['body_weight']
F_max_hfl = 2000 / MODEL_PARAMS['body_weight']

MUSCLE_DICTIONARY = {

    "GENERAL": {
        "w": 0.56,         # CE force-length width parameter
        "c": 0.05,         # remaining force at +/- w
        "K": 5.0,          # force-velocity concentric shape factor
        "N": 1.5,          # eccentric force enhancement factor
        "tau": 0.01,       # excitation-contraction coupling constant [s]
    },

    "VAS": {
        "F_max": F_max_vas,   # [N/kg]
        "l_slack": 0.23,      # [m]
        "l_opt": 0.08,        # [m]
        "v_max": 12.0,        # [l_opt/s]
        "e_ref_see": 0.04
    },

    "SOL": {
        "F_max": F_max_sol,
        "l_slack": 0.26,
        "l_opt": 0.04,
        "v_max": 6.0,         # [l_opt/s]
        "e_ref_see": 0.04
    },

    "GAS": {
        "F_max": F_max_gas,
        "l_slack": 0.40,
        "l_opt": 0.05,
        "v_max": 12.0,        # [l_opt/s]
        "e_ref_see": 0.04
    },

    "TA": {
        "F_max": F_max_ta,
        "l_slack": 0.24,
        "l_opt": 0.06,
        "v_max": 12.0,        # [l_opt/s]
        "e_ref_see": 0.04
    },

    "HAM": {
        "F_max": F_max_ham,
        "l_slack": 0.31,
        "l_opt": 0.10,
        "v_max": 12.0,        # [l_opt/s]
        "e_ref_see": 0.04
    },

    "GLU": {
        "F_max": F_max_glu,
        "l_slack": 0.13,
        "l_opt": 0.11,
        "v_max": 12.0,        # [l_opt/s]
        "e_ref_see": 0.04
    },

    "HFL": {
        "F_max": F_max_hfl,
        "l_slack": 0.10,
        "l_opt": 0.11,
        "v_max": 12.0,        # [l_opt/s]
        "e_ref_see": 0.04
    }
}

NERVOUS_SYSTEM_DICTIONARY = {

    "DELAY": {
        "LONG":  0.020,
        "MID":   0.010,
        "SHORT": 0.005,
    },

    "GENERAL": {
        "SWING_STIM": 0.01, # just used as the lower limit for clipping stimulations 
        "TIME_STEP": 0.001
    },

    "COUPLINGS": {
        "G_SOLTA": 0.3,
        "G_HAMHFL": 4.0,
    },

    "BALANCE": {
        "KP": 1.91,
        "KD": 0.25,
        "KBW": 1.2, # 1.2
        "DELTA_S": 0.25, # 0.25
        "THETA_REF": -0.105,
        "K_LEAN": 1.15,
        "K_PHI": 2.0,
        "PHI_K_OFF": 2.97,
    },

    "SOL": {"FGAIN": 1.1, "PRESTIM": 0.01}, # 1.1
    "TA":  {"FGAIN": 1.1, "PRESTIM": 0.01, "LCEOFF": 0.71},
    "GAS": {"FGAIN": 1.3, "PRESTIM": 0.05},     # 1.1, 0.01
    "VAS": {"FGAIN": 1.35, "PRESTIM": 0.09},    # 1.15, 0.09
    "HAM": {"FGAIN": 0.65, "PRESTIM": 0.05, "LCEOFF": 0.85},
    "GLU": {"FGAIN": 0.4, "PRESTIM": 0.05},
    "HFL": {"FGAIN": 0.35, "PRESTIM": 0.05, "LCEOFF": 0.6},
}