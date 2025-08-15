"""
Rigid Body Walker Configuration

This configuration file defines a rigid body walker and assigns all
relevant kinematic and dynamic properties. The properties need to be
configured by the user in this file.

"""

import math
import numpy as np
from body_system import RigidBodySystem
from joint import RevoluteJoint
from contact import GroundContact
from muscle_params_storage import muscle_params, nervous_params, musculo_skeletal_params
from mtc import muscle_tendon_unit
from musculo_skeletal import musculo_skeletal
from nervous_system import feedback_loop

def human_model(dt):

    pi = math.pi

    ############### parameters ##############

    NAME = 'Human Neuromuscular Walking Model'

    # joint spring-damper parameters
    DAMPING = 4000
    STIFFNESS = 1 * DAMPING**2 / (100) # critical damping?

    # joint angle limit torque parameters
    k_lim = 0.3 * 180/pi            # [Nm/rad] mechanical limits stiffness
    q_dot_max = 1 * pi/180      # [rad/s] mechanical limits damping

    # Muscle max force parameter [N]
    F_max_SOL = 4000
    F_max_VAS = 6000
    F_max_TA = 800
    F_max_GAS = 1500
    F_max_HAM = 3000
    F_max_GLU = 1500
    F_max_HFL = 2000

    # muscle ce optimal length [m]
    l_opt_VAS = 0.08
    l_opt_SOL = 0.04
    l_opt_TA = 0.06
    l_opt_GAS = 0.05
    l_opt_HAM = 0.10
    l_opt_GLU = 0.11
    l_opt_HFL = 0.11

    ############### initial conditions ##############

    x0 = 0          # [m] inital forward (along horizontal axis) position of trunk CoM
    h0 = 1.315       # [m] initial height (along vertical axis) of trunk CoM
    p0 = 0   # [rad] initial orientation of trunk CoM wrt global z-axis (counterclockwise positive)

    vx0 = 1.3         # [m/s] initial forward (along horizontal axis) velocity of trunk CoM
    vz0 = 0         # [m/s] initial vertical velocity of trunk
    vp0 = 0         # [rad/s] initial ang velocity of trunk 

    # initial joint angles defined between bodies from base to mate body of joint [rad]
    initial_hip_1_angle =  150 * pi / 180
    initial_knee_1_angle = 155 * pi / 180
    initial_ankle_1_angle = 90 * pi / 180

    initial_hip_2_angle =  171 * pi / 180  
    initial_knee_2_angle = 155 * pi / 180 
    initial_ankle_2_angle = 75 * pi / 180

    # joint angle limits
    q_max_hip = 230*pi/180
    q_min_hip = 20*pi/180 

    q_max_knee = 175*pi/180
    q_min_knee = 10*pi/180

    q_max_ankle = 130*pi/180
    q_min_ankle = 70*pi/180

    # physical dimensions of bodies [m]
    trunk_length = 0.8
    trunk_width = 0.1
    
    thigh_length =  0.5
    thigh_width =  0.06

    shank_length = 0.5
    shank_width = 0.06

    foot_length = 0.2 
    foot_width = 0.02

    # masses [Kg] and inertia about -y axis [Kgm^2] of bodies 
    
    m_trunk = 53.5
    I_trunk = 3
    m_thigh = 8.5
    I_thigh = 0.15
    m_shank = 3.5
    I_shank = 0.05
    m_foot = 1.25
    I_foot = 0.005
    g = 9.81
    body_weight = g*(m_trunk + 2*(m_thigh + m_shank + m_foot))

    # COM locations from proximal joint of body
    # (assumes there is a frame parallel to the body frame at these joint locations)
    trunk_com_prox = (0,-0.45)
    thigh_com_prox = (0,0.2)
    shank_com_prox = (0,-0.2)
    foot_com_prox = (0,0.06)

    # site locations from COM of body
    trunk_thigh_site = (0,-0.35)

    thigh_trunk_site = (0,-0.20)
    thigh_shank_site = (0,0.30)

    shank_thigh_site = (0,0.2)
    shank_foot_site = (0,-0.30)

    foot_shank_site = (0,-0.02)

    # geometric centers of bodies from com (written in com frame)
    gc_trunk = (0,0.05)
    gc_thigh = (0,0.05)
    gc_shank = (0,-0.05)
    gc_foot = (0,0.04)

    ################ time delays ###############
    delP_short = 0.005
    delP_medium = 0.010
    delP_long = 0.020

    delP_list = [delP_short,delP_medium,delP_long]

    ############### initialize rigid body system #################

    # create empty rigid body system
    rbs = RigidBodySystem(NAME)

    # add segments and sites to rigid body system
    trunk = rbs.add_body('trunk', mass=m_trunk, moment_of_inertia=I_trunk,
                         x=x0, z=h0, p=p0,
                         vx=vx0, vz=vz0, vp=vp0, x_gc=gc_trunk[0], z_gc=gc_trunk[1])
    
    trunk.add_site('hip', x_b=trunk_thigh_site[0], z_b=trunk_thigh_site[1]) # location in trunk frame

    thigh_1 = rbs.add_body('thigh_1',
                         mass=m_thigh, moment_of_inertia=I_thigh,
                         x=x0, z=h0, p=p0,
                         vx=0.0, vz=0.0, vp=0.0, x_gc=gc_thigh[0], z_gc=gc_thigh[1])
    thigh_1.add_site('hip', x_b=thigh_trunk_site[0], z_b=thigh_trunk_site[1])
    thigh_1.add_site('knee', x_b=thigh_shank_site[0], z_b=thigh_shank_site[1])

    shank_1 = rbs.add_body('shank_1', mass=m_shank, moment_of_inertia=I_shank,
                         x=x0, z=h0, p=p0,
                         vx=0.0, vz=0.0, vp=0.0, x_gc=gc_shank[0], z_gc=gc_shank[1])
    shank_1.add_site('knee', x_b=shank_thigh_site[0], z_b=shank_thigh_site[1])
    shank_1.add_site('ankle', x_b=shank_foot_site[0], z_b=shank_foot_site[1])

    foot_1 = rbs.add_body('foot_1', mass=m_foot, moment_of_inertia=I_foot,
                        x=x0, z=h0, p=p0,
                        vx=0.0, vz=0.0, vp=0.0, x_gc=gc_foot[0], z_gc=gc_foot[1])
    foot_1.add_site('ankle', x_b=foot_shank_site[0], z_b=foot_shank_site[1])
    foot_1.add_site('heel', x_b=0.0, z_b=-foot_length/2)
    foot_1.add_site('toe', x_b=0.0, z_b=foot_length/2)

    thigh_2 = rbs.add_body('thigh_2',
                         mass=m_thigh, moment_of_inertia=I_thigh,
                         x=x0, z=h0, p=p0,
                         vx=0.0, vz=0.0, vp=0.0, x_gc=gc_thigh[0], z_gc=gc_thigh[1])
    thigh_2.add_site('hip', x_b=thigh_trunk_site[0], z_b=thigh_trunk_site[1])
    thigh_2.add_site('knee', x_b=thigh_shank_site[0], z_b=thigh_shank_site[1])

    shank_2 = rbs.add_body('shank_2', mass=m_shank, moment_of_inertia=I_shank,
                         x=x0, z=h0, p=p0,
                         vx=0.0, vz=0.0, vp=0.0, x_gc=gc_shank[0], z_gc=gc_shank[1])
    shank_2.add_site('knee', x_b=shank_thigh_site[0], z_b=shank_thigh_site[1])
    shank_2.add_site('ankle', x_b=shank_foot_site[0], z_b=shank_foot_site[1])

    foot_2 = rbs.add_body('foot_2', mass=m_foot, moment_of_inertia=I_foot,
                        x=x0, z=h0, p=p0,
                        vx=0.0, vz=0.0, vp=0.0, x_gc=gc_foot[0], z_gc=gc_foot[1])
    foot_2.add_site('ankle', x_b=foot_shank_site[0], z_b=foot_shank_site[1])
    foot_2.add_site('heel', x_b=0.0, z_b=-foot_length/2)
    foot_2.add_site('toe', x_b=0.0, z_b=foot_length/2)

    ############ initialize animation #################

    # The first tuple eg:(-0.04, 0.04, 0.04, -0.04, -0.04) represents the x-coordinates.
    # The second tuple eg:(-0.3, -0.3, 0.3, 0.3, -0.3) represents the z-coordinates.
    # these form a rectangular shape, Each shape is a closed loop, where the last point repeats the first point

    # add body visual geometries for animation
    trunk.geometry = ((-trunk_width/2, trunk_width/2, trunk_width/2, -trunk_width/2, -trunk_width/2), 
                      (-trunk_length/2, -trunk_length/2, trunk_length/2, trunk_length/2, -trunk_length/2))
         
    thigh_1.geometry = ((-thigh_width/2, thigh_width/2, thigh_width/2, -thigh_width/2, -thigh_width/2),  
                      (-thigh_length/2, -thigh_length/2, thigh_length/2, thigh_length/2, -thigh_length/2))
    shank_1.geometry = ((-shank_width/2, shank_width/2, shank_width/2, -shank_width/2, -shank_width/2),
                      (-shank_length/2, -shank_length/2, shank_length/2, shank_length/2, -shank_length/2))
    foot_1.geometry = ((-foot_width/2,  foot_width/2, foot_width/2, -foot_width/2, -foot_width/2),
                      (-foot_length/2, -foot_length/2, foot_length/2, foot_length/2, -foot_length/2))
    
    thigh_2.geometry = ((-thigh_width/2, thigh_width/2, thigh_width/2, -thigh_width/2, -thigh_width/2),  
                      (-thigh_length/2, -thigh_length/2, thigh_length/2, thigh_length/2, -thigh_length/2))
    shank_2.geometry = ((-shank_width/2, shank_width/2, shank_width/2, -shank_width/2, -shank_width/2),
                      (-shank_length/2, -shank_length/2, shank_length/2, shank_length/2, -shank_length/2))
    foot_2.geometry = ((-foot_width/2,  foot_width/2, foot_width/2, -foot_width/2, -foot_width/2),
                      (-foot_length/2, -foot_length/2, foot_length/2, foot_length/2, -foot_length/2))
    
    ############# apply translations and rotations to set bodies' initial conditions in world frame #################

    trunk.p = p0 # initial trunk lean
    trunk.update_site_coords()  # converts initial site coordinates to world frame

    thigh_1.p = trunk.p - initial_hip_1_angle # angle made by thigh with world frame (joint angles defined differently for different frames!!!)
    pos_ab = (trunk.x + trunk.sites[0].x_b2w, # [x_b2w,z_b2w]: vector from the center of the trunk to the trunk-thigh site expresed in the world frame
              trunk.z + trunk.sites[0].z_b2w) # position of of trunk-thigh site wrt world frame origin
    thigh_1.x, thigh_1.z = transform_point(pos_ab, thigh_1.p, thigh_com_prox) # gives the cordinates of the com of thigh in world frame
    thigh_1.update_site_coords()

    thigh_2.p = trunk.p - initial_hip_2_angle
    pos_ab = (trunk.x + trunk.sites[0].x_b2w, 
              trunk.z + trunk.sites[0].z_b2w) 
    thigh_2.x, thigh_2.z = transform_point(pos_ab, thigh_2.p, thigh_com_prox)
    thigh_2.update_site_coords()

    shank_1.p = thigh_1.p + initial_knee_1_angle
    pos_ab = (thigh_1.x + thigh_1.sites[1].x_b2w,
              thigh_1.z + thigh_1.sites[1].z_b2w)
    shank_1.x, shank_1.z = transform_point(pos_ab, shank_1.p, shank_com_prox)
    shank_1.update_site_coords()

    shank_2.p = thigh_2.p + initial_knee_2_angle
    pos_ab = (thigh_2.x + thigh_2.sites[1].x_b2w,
              thigh_2.z + thigh_2.sites[1].z_b2w)
    shank_2.x, shank_2.z = transform_point(pos_ab, shank_2.p, shank_com_prox)
    shank_2.update_site_coords()

    foot_1.p = shank_1.p - initial_ankle_1_angle
    pos_ab = (shank_1.x + shank_1.sites[1].x_b2w,
              shank_1.z + shank_1.sites[1].z_b2w)
    foot_1.x, foot_1.z = transform_point(pos_ab, foot_1.p, foot_com_prox)
    foot_1.update_site_coords()

    foot_2.p = shank_2.p - initial_ankle_2_angle
    pos_ab = (shank_2.x + shank_2.sites[1].x_b2w,
              shank_2.z + shank_2.sites[1].z_b2w)
    foot_2.x, foot_2.z = transform_point(pos_ab, foot_2.p, foot_com_prox)
    foot_2.update_site_coords()

    ############ initialize joints ################

    hip_1 = RevoluteJoint('hip_1', thigh_1, thigh_1.sites[0], 'distal', trunk, trunk.sites[0],
                        q_min_hip, q_max_hip, q_dot_max,
                        STIFFNESS, DAMPING, k_lim)
    hip_2 = RevoluteJoint('hip_2', thigh_2, thigh_2.sites[0], 'distal', trunk, trunk.sites[0],
                        q_min_hip, q_max_hip, q_dot_max,
                        STIFFNESS, DAMPING, k_lim)
    
    knee_1 = RevoluteJoint('knee_1', thigh_1, thigh_1.sites[1], 'proximal', shank_1, shank_1.sites[0], 
                         q_min_knee, q_max_knee, q_dot_max,
                         STIFFNESS, DAMPING, k_lim)
    knee_2 = RevoluteJoint('knee_2', thigh_2, thigh_2.sites[1], 'proximal', shank_2, shank_2.sites[0], 
                         q_min_knee, q_max_knee, q_dot_max,
                         STIFFNESS, DAMPING, k_lim)
    
    ankle_1 = RevoluteJoint('ankle_1', foot_1, foot_1.sites[0], 'distal', shank_1, shank_1.sites[1], 
                          q_min_ankle, q_max_ankle, q_dot_max,
                          STIFFNESS, DAMPING, k_lim)
    ankle_2 = RevoluteJoint('ankle_2', foot_2, foot_2.sites[0], 'distal', shank_2, shank_2.sites[1], 
                          q_min_ankle, q_max_ankle, q_dot_max,
                          STIFFNESS, DAMPING, k_lim)
    
    rbs.joint_list = [hip_1,knee_1,ankle_1,hip_2,knee_2,ankle_2]

    ############ initialize contact points #################
    
    heel_1 = GroundContact('heel_1 contact', foot_1, foot_1.sites[1],
                          stiffness_x=8200.0, max_vx=0.03,
                          stiffness_z=81500.0, max_vz=0.03,
                          mu_slide=0.8, v_transition=0.01,
                          mu_stick=0.9)
    heel_2 = GroundContact('heel_2 contact', foot_2, foot_2.sites[1],
                          stiffness_x=8200.0, max_vx=0.03,
                          stiffness_z=81500.0, max_vz=0.03,
                          mu_slide=0.8, v_transition=0.01,
                          mu_stick=0.9)
    
    toe_1 = GroundContact('toe_1 contact', foot_1, foot_1.sites[2],
                          stiffness_x=8200.0, max_vx=0.03,
                          stiffness_z=81500.0, max_vz=0.03,
                          mu_slide=0.8, v_transition=0.01,
                          mu_stick=0.9)
    toe_2 = GroundContact('toe_2 contact', foot_2, foot_2.sites[2],
                          stiffness_x=8200.0, max_vx=0.03,
                          stiffness_z=81500.0, max_vz=0.03,
                          mu_slide=0.8, v_transition=0.01,
                          mu_stick=0.9)
    
    rbs.contact_list = [heel_1,toe_1,heel_2,toe_2]

    ################### initialize muscles and their feedback pathways ################

    VAS_params = muscle_params(F_max=F_max_VAS,l_opt=l_opt_VAS, w=0.56, c=math.log(0.05), v_max=12*l_opt_VAS, N=1.5, K=5, l_rest=0.23, 
                               e_ref_see=0.04,tau=0.01,l_ref_mtc=l_opt_VAS+0.23)
    VAS_musculo_skeletal_params = musculo_skeletal_params(r0_array = np.array([0,0.06,0,0,0.06,0]), q_ref_array = np.array([0,125*pi/180,0,0,125*pi/180,0]), 
                               q_max_array = np.array([0,165*pi/180,0,0,165*pi/180,0]), rho_array = np.array([0,0.7,0,0,0.7,0]),l_ref_mtc=l_opt_VAS+0.23)
    VAS_neural_params = nervous_params(delP_list=delP_list, stim0=0.09, G=4.3/(F_max_VAS), 
                                       G_l=None, G_l_2=None, k_p=20, k_d=None,k_lean=None, l_off=None, l_off_2=None, p_trunk_ref=None, k_bw=1.3/(body_weight), delta_S=None)

    SOL_params = muscle_params(F_max=F_max_SOL, l_opt=l_opt_SOL, w=0.56, c=math.log(0.05), v_max=6*l_opt_SOL, N=1.5, K=5, l_rest=0.26, 
                               e_ref_see=0.04,tau=0.01,l_ref_mtc=l_opt_SOL+0.26)
    SOL_musculo_skeletal_params = musculo_skeletal_params(r0_array = np.array([0,0,0.05,0,0,0.05]), q_ref_array = np.array([0,0,80*pi/180,0,0,80*pi/180]), 
                               q_max_array = np.array([0,0,110*pi/180,0,0,110*pi/180]), rho_array = np.array([0,0,0.5,0,0,0.5]),l_ref_mtc=l_opt_SOL+0.26)
    SOL_neural_params = nervous_params(delP_list=delP_list, stim0=0.01, G=4.5/(F_max_SOL), 
                                       G_l=None, G_l_2=None, k_p=None, k_d=None,k_lean=None, l_off=None, l_off_2=None, p_trunk_ref=None, k_bw=None, delta_S=None)

    GAS_params = muscle_params(F_max=F_max_GAS, l_opt=l_opt_GAS, w=0.56, c=math.log(0.05),v_max=12*l_opt_GAS, N=1.5, K=5, l_rest=0.4, 
                               e_ref_see=0.04, tau = 0.01,l_ref_mtc=l_opt_GAS+0.4)
    GAS_musculo_skeletal_params = musculo_skeletal_params(r0_array = np.array([0,0.05,0.05,0,0.05,0.05]), q_ref_array = np.array([0,165*pi/180,80*pi/180,0,165*pi/180,80*pi/180]), 
                               q_max_array = np.array([0,140*pi/180,110*pi/180,0,140*pi/180,110*pi/180]), rho_array = np.array([0,0.7,0.7,0,0.7,0.7]), l_ref_mtc=l_opt_GAS+0.4)
    GAS_neural_params = nervous_params(delP_list=delP_list, stim0=0.01, G=100/(F_max_GAS), G_l=None, G_l_2=None, k_p=None, k_d=None,k_lean=None, l_off=None, l_off_2=None, p_trunk_ref=None, k_bw=None, delta_S=None)

    TA_params = muscle_params(F_max=F_max_TA, l_opt=l_opt_TA, w=0.56, c=math.log(0.05), v_max=12*l_opt_TA, N=1.5, K=5, l_rest=0.24, 
                              e_ref_see=0.04,tau = 0.01,l_ref_mtc=l_opt_TA+0.24)
    TA_musculo_skeletal_params = musculo_skeletal_params(r0_array = np.array([0,0,0.04,0,0,0.04]), q_ref_array = np.array([0,0,110*pi/180,0,0,110*pi/180]), 
                              q_max_array = np.array([0,0,80*pi/180,0,0,80*pi/180]), rho_array = np.array([0,0,0.7,0,0,0.7]),l_ref_mtc=l_opt_TA+0.24)
    TA_neural_params = nervous_params(delP_list=delP_list, stim0=0.01, G=1.1, G_l=3.2/(F_max_TA), G_l_2= None,
                                      k_p=None, k_d=None,k_lean=None, l_off=0.71*l_opt_TA, l_off_2=None, p_trunk_ref=None, k_bw=None, delta_S=None)

    GLU_params = muscle_params(F_max=F_max_GLU, l_opt=l_opt_GLU, w=0.56, c=math.log(0.05),v_max=1.2, N=1.5, K=5, l_rest=0.13, 
                               e_ref_see=0.04,tau=0.01,l_ref_mtc=l_opt_GLU+0.13)
    GLU_musculo_skeletal_params = musculo_skeletal_params(r0_array = np.array([0.1,0,0,0.1,0,0]), q_ref_array = np.array([150*pi/180,0,0,150*pi/180,0,0]), 
                               q_max_array = np.array([0,0,0,0,0,0]), rho_array = np.array([0.5,0,0,0.5,0,0]),l_ref_mtc=l_opt_GLU+0.13)
    GLU_neural_params = nervous_params(delP_list=delP_list, stim0=0.05, G=0.4/F_max_GLU, G_l=None, G_l_2=None, k_p=14.1, k_d=0.25,k_lean=None, l_off=None, l_off_2=None, p_trunk_ref=-0.105, k_bw=2/(body_weight), delta_S=0.35)

    HAM_params = muscle_params(F_max=F_max_HAM, l_opt=0.1, w=0.56, c=math.log(0.05),v_max=12*l_opt_GLU, N=1.5, K=5, l_rest=0.31, 
                               e_ref_see=0.04, tau=0.01,l_ref_mtc=l_opt_HAM+0.31)
    HAM_musculo_skeletal_params = musculo_skeletal_params(r0_array = np.array([0.08,0.05,0,0.08,0.05,0]), q_ref_array = np.array([155*pi/180,pi,0,155*pi/180,pi,0]), 
                               q_max_array = np.array([0,pi,0,0,pi,0]), rho_array = np.array([0.7,0.7,0,0.7,0.7,0]),l_ref_mtc=l_opt_HAM+0.31)
    HAM_neural_params = nervous_params(delP_list=delP_list, stim0=0.05, G=0.65/F_max_HAM, G_l=None, G_l_2=None, k_p=14.1, k_d=0.5,k_lean=None, l_off=None, l_off_2=None, p_trunk_ref=-0.105, k_bw=2/(body_weight), delta_S=None)

    HFL_params = muscle_params(F_max=F_max_HFL, l_opt=l_opt_HFL, w=0.56, c=math.log(0.05), v_max=12*l_opt_HFL, N=1.5, K=5, l_rest=0.1, 
                               e_ref_see=0.04,tau=0.01,l_ref_mtc=l_opt_HFL+0.1)
    HFL_musculo_skeletal_params = musculo_skeletal_params(r0_array = np.array([0.1,0,0,0.1,0,0]), q_ref_array = np.array([pi,0,0,pi,0,0]), 
                               q_max_array = np.array([0,0,0,0,0,0]), rho_array = np.array([0.5,0,0,0.5,0,0]),l_ref_mtc=l_opt_HFL+0.1)
    HFL_neural_params = nervous_params(delP_list=delP_list, stim0=0.05, G=None, G_l=1, G_l_2=1, k_p=14.1, k_d=0.25, k_lean=1.15,
                                       l_off=0.55*l_opt_HFL, l_off_2=0.85*l_opt_HAM, p_trunk_ref=-0.105, k_bw=2/(body_weight), delta_S=0.35)

    VAS_1 = muscle_tendon_unit('VAS_1', VAS_params, dt)
    SOL_1 = muscle_tendon_unit('SOL_1', SOL_params, dt)
    GAS_1 = muscle_tendon_unit('GAS_1', GAS_params, dt)
    TA_1 =  muscle_tendon_unit('TA_1', TA_params, dt)
    GLU_1 = muscle_tendon_unit('GLU_1', GLU_params, dt)
    HAM_1 = muscle_tendon_unit('HAM_1', HAM_params, dt)
    HFL_1 = muscle_tendon_unit('HFL_1', HFL_params, dt)

    VAS_2 = muscle_tendon_unit('VAS_2', VAS_params, dt)
    SOL_2 = muscle_tendon_unit('SOL_2', SOL_params, dt)
    GAS_2 = muscle_tendon_unit('GAS_2', GAS_params, dt)
    TA_2 =  muscle_tendon_unit('TA_2', TA_params, dt)
    GLU_2 = muscle_tendon_unit('GLU_2', GLU_params, dt)
    HAM_2 = muscle_tendon_unit('HAM_2', HAM_params, dt)
    HFL_2 = muscle_tendon_unit('HFL_2', HFL_params, dt)

    rbs.muscle_list = [VAS_1,SOL_1,GAS_1,TA_1,GLU_1,HAM_1,HFL_1,VAS_2,SOL_2,GAS_2,TA_2,GLU_2,HAM_2,HFL_2]  # VAS, SOL, GAS, TA, GLU, HAM, HFL

    VAS_1_feedback = feedback_loop('VAS_1', rbs.joint_list,rbs.body_list,rbs.muscle_list, VAS_neural_params, dt)
    SOL_1_feedback = feedback_loop('SOL_1', rbs.joint_list,rbs.body_list,rbs.muscle_list, SOL_neural_params, dt)
    GAS_1_feedback = feedback_loop('GAS_1', rbs.joint_list,rbs.body_list,rbs.muscle_list, GAS_neural_params, dt)
    TA_1_feedback =  feedback_loop('TA_1', rbs.joint_list,rbs.body_list,rbs.muscle_list,TA_neural_params, dt)
    GLU_1_feedback = feedback_loop('GLU_1', rbs.joint_list,rbs.body_list,rbs.muscle_list,GLU_neural_params, dt)
    HAM_1_feedback = feedback_loop('HAM_1', rbs.joint_list,rbs.body_list,rbs.muscle_list, HAM_neural_params,dt)
    HFL_1_feedback = feedback_loop('HFL_1', rbs.joint_list,rbs.body_list,rbs.muscle_list,HFL_neural_params, dt)

    VAS_2_feedback = feedback_loop('VAS_2', rbs.joint_list,rbs.body_list,rbs.muscle_list, VAS_neural_params, dt)
    SOL_2_feedback = feedback_loop('SOL_2', rbs.joint_list,rbs.body_list,rbs.muscle_list, SOL_neural_params, dt)
    GAS_2_feedback = feedback_loop('GAS_2', rbs.joint_list,rbs.body_list,rbs.muscle_list, GAS_neural_params, dt)
    TA_2_feedback =  feedback_loop('TA_2', rbs.joint_list,rbs.body_list,rbs.muscle_list, TA_neural_params, dt)
    GLU_2_feedback = feedback_loop('GLU_2', rbs.joint_list,rbs.body_list,rbs.muscle_list, GLU_neural_params, dt)
    HAM_2_feedback = feedback_loop('HAM_2', rbs.joint_list,rbs.body_list,rbs.muscle_list, HAM_neural_params, dt)
    HFL_2_feedback = feedback_loop('HFL_2', rbs.joint_list,rbs.body_list,rbs.muscle_list, HFL_neural_params, dt)

    rbs.feedback_list = [VAS_1_feedback,SOL_1_feedback,GAS_1_feedback,TA_1_feedback,GLU_1_feedback,HAM_1_feedback,HFL_1_feedback,
                         VAS_2_feedback,SOL_2_feedback,GAS_2_feedback,TA_2_feedback,GLU_2_feedback,HAM_2_feedback,HFL_2_feedback]  # VAS, SOL, GAS, TA, GLU, HAM, HFL

    VAS_1_musculo_skeletal = musculo_skeletal('VAS_1', rbs.joint_list, [rbs.joint_list[1].name], 'front', VAS_musculo_skeletal_params)
    SOL_1_musculo_skeletal = musculo_skeletal('SOL_1', rbs.joint_list, [rbs.joint_list[2].name], 'back', SOL_musculo_skeletal_params)
    GAS_1_musculo_skeletal = musculo_skeletal('GAS_1', rbs.joint_list, [rbs.joint_list[1].name, rbs.joint_list[2].name], 'back', GAS_musculo_skeletal_params)
    TA_1_musculo_skeletal =  musculo_skeletal('TA_1', rbs.joint_list, [rbs.joint_list[2].name], 'front', TA_musculo_skeletal_params)
    GLU_1_musculo_skeletal = musculo_skeletal('GLU_1', rbs.joint_list, [rbs.joint_list[0].name], 'back', GLU_musculo_skeletal_params)
    HAM_1_musculo_skeletal = musculo_skeletal('HAM_1', rbs.joint_list, [rbs.joint_list[0].name, rbs.joint_list[1].name], 'back', HAM_musculo_skeletal_params)
    HFL_1_musculo_skeletal = musculo_skeletal('HFL_1', rbs.joint_list, [rbs.joint_list[0].name], 'front', HFL_musculo_skeletal_params)

    VAS_2_musculo_skeletal = musculo_skeletal('VAS_2', rbs.joint_list, [rbs.joint_list[4].name], 'front', VAS_musculo_skeletal_params)
    SOL_2_musculo_skeletal = musculo_skeletal('SOL_2', rbs.joint_list, [rbs.joint_list[5].name], 'back', SOL_musculo_skeletal_params)
    GAS_2_musculo_skeletal = musculo_skeletal('GAS_2', rbs.joint_list, [rbs.joint_list[4].name, rbs.joint_list[5].name], 'back', GAS_musculo_skeletal_params)
    TA_2_musculo_skeletal =  musculo_skeletal('TA_2', rbs.joint_list, [rbs.joint_list[5].name], 'front', TA_musculo_skeletal_params)
    GLU_2_musculo_skeletal = musculo_skeletal('GLU_2', rbs.joint_list, [rbs.joint_list[3].name], 'back', GLU_musculo_skeletal_params)
    HAM_2_musculo_skeletal = musculo_skeletal('HAM_2', rbs.joint_list, [rbs.joint_list[3].name, rbs.joint_list[4].name], 'back', HAM_musculo_skeletal_params)
    HFL_2_musculo_skeletal = musculo_skeletal('HFL_2', rbs.joint_list, [rbs.joint_list[3].name], 'front', HFL_musculo_skeletal_params)
    
    rbs.musculo_skeletal_list = [VAS_1_musculo_skeletal,SOL_1_musculo_skeletal,GAS_1_musculo_skeletal,TA_1_musculo_skeletal,GLU_1_musculo_skeletal,HAM_1_musculo_skeletal,HFL_1_musculo_skeletal,
                         VAS_2_musculo_skeletal,SOL_2_musculo_skeletal,GAS_2_musculo_skeletal,TA_2_musculo_skeletal,GLU_2_musculo_skeletal,HAM_2_musculo_skeletal,HFL_2_musculo_skeletal]
    
    return rbs

# transforms coordinates of point in frame B to frame A (rA = rAB + Cab*rB)
def transform_point(pos_ab, angle_ab, pos_b):
    cp, sp = math.cos(angle_ab), math.sin(angle_ab)
    x_a = pos_ab[0] + cp * pos_b[0] - sp * pos_b[1]
    z_a = pos_ab[1] + sp * pos_b[0] + cp * pos_b[1]

    return x_a, z_a