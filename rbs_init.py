"""
Rigid Body System Initialization

Builds the human walking model (bodies, joints, contacts, geometry)
from the parameters defined in config.py.
"""

from config import MODEL_PARAMS
from physics.body_system import RigidBodySystem


def human_model(dt):

    params = MODEL_PARAMS

    ############### initialize rigid body system #################

    # create empty rigid body system
    rbs = RigidBodySystem(params['NAME'])

    # add segments and sites to rigid body system
    trunk = rbs.add_body('trunk', mass=params['m_trunk'], moment_of_inertia=params['I_trunk'],
                         x_gc=params['gc_trunk'][0], z_gc=params['gc_trunk'][1])
    
    trunk.add_site('hip', x_b=params['trunk_thigh_site'][0], z_b=params['trunk_thigh_site'][1]) # location in trunk frame

    thigh_1 = rbs.add_body('thigh_1', mass=params['m_thigh'], moment_of_inertia=params['I_thigh'],
                           x_gc=params['gc_thigh'][0], z_gc=params['gc_thigh'][1])
    thigh_1.add_site('hip', x_b=params['thigh_trunk_site'][0], z_b=params['thigh_trunk_site'][1])
    thigh_1.add_site('knee', x_b=params['thigh_shank_site'][0], z_b=params['thigh_shank_site'][1])

    shank_1 = rbs.add_body('shank_1', mass=params['m_shank'], moment_of_inertia=params['I_shank'],
                           x_gc=params['gc_shank'][0], z_gc=params['gc_shank'][1])
    shank_1.add_site('knee', x_b=params['shank_thigh_site'][0], z_b=params['shank_thigh_site'][1])
    shank_1.add_site('ankle', x_b=params['shank_foot_site'][0], z_b=params['shank_foot_site'][1])

    foot_1 = rbs.add_body('foot_1', mass=params['m_foot'], moment_of_inertia=params['I_foot'],
                          x_gc=params['gc_foot'][0], z_gc=params['gc_foot'][1])
    foot_1.add_site('ankle', x_b=params['foot_shank_site'][0], z_b=params['foot_shank_site'][1])
    foot_1.add_site('heel', x_b=0.0, z_b=-params['foot_length']/2)
    foot_1.add_site('toe', x_b=0.0, z_b=params['foot_length']/2)

    thigh_2 = rbs.add_body('thigh_2', mass=params['m_thigh'], moment_of_inertia=params['I_thigh'],
                           x_gc=params['gc_thigh'][0], z_gc=params['gc_thigh'][1])
    thigh_2.add_site('hip', x_b=params['thigh_trunk_site'][0], z_b=params['thigh_trunk_site'][1])
    thigh_2.add_site('knee', x_b=params['thigh_shank_site'][0], z_b=params['thigh_shank_site'][1])

    shank_2 = rbs.add_body('shank_2', mass=params['m_shank'], moment_of_inertia=params['I_shank'],
                           x_gc=params['gc_shank'][0], z_gc=params['gc_shank'][1])
    shank_2.add_site('knee', x_b=params['shank_thigh_site'][0], z_b=params['shank_thigh_site'][1])
    shank_2.add_site('ankle', x_b=params['shank_foot_site'][0], z_b=params['shank_foot_site'][1])

    foot_2 = rbs.add_body('foot_2', mass=params['m_foot'], moment_of_inertia=params['I_foot'],
                          x_gc=params['gc_foot'][0], z_gc=params['gc_foot'][1])
    foot_2.add_site('ankle', x_b=params['foot_shank_site'][0], z_b=params['foot_shank_site'][1])
    foot_2.add_site('heel', x_b=0.0, z_b=-params['foot_length']/2)
    foot_2.add_site('toe', x_b=0.0, z_b=params['foot_length']/2)

    ############ initialize animation #################

    # add body visual geometries for animation
    trunk.geometry = ((-params['trunk_width']/2, params['trunk_width']/2, params['trunk_width']/2, -params['trunk_width']/2, -params['trunk_width']/2), 
                      (-params['trunk_length']/2, -params['trunk_length']/2, params['trunk_length']/2, params['trunk_length']/2, -params['trunk_length']/2))
         
    thigh_1.geometry = ((-params['thigh_width']/2, params['thigh_width']/2, params['thigh_width']/2, -params['thigh_width']/2, -params['thigh_width']/2),  
                        (-params['thigh_length']/2, -params['thigh_length']/2, params['thigh_length']/2, params['thigh_length']/2, -params['thigh_length']/2))
    shank_1.geometry = ((-params['shank_width']/2, params['shank_width']/2, params['shank_width']/2, -params['shank_width']/2, -params['shank_width']/2),
                        (-params['shank_length']/2, -params['shank_length']/2, params['shank_length']/2, params['shank_length']/2, -params['shank_length']/2))
    foot_1.geometry = ((-params['foot_width']/2,  params['foot_width']/2, params['foot_width']/2, -params['foot_width']/2, -params['foot_width']/2),
                       (-params['foot_length']/2, -params['foot_length']/2, params['foot_length']/2, params['foot_length']/2, -params['foot_length']/2))
    
    thigh_2.geometry = ((-params['thigh_width']/2, params['thigh_width']/2, params['thigh_width']/2, -params['thigh_width']/2, -params['thigh_width']/2),  
                        (-params['thigh_length']/2, -params['thigh_length']/2, params['thigh_length']/2, params['thigh_length']/2, -params['thigh_length']/2))
    shank_2.geometry = ((-params['shank_width']/2, params['shank_width']/2, params['shank_width']/2, -params['shank_width']/2, -params['shank_width']/2),
                        (-params['shank_length']/2, -params['shank_length']/2, params['shank_length']/2, params['shank_length']/2, -params['shank_length']/2))
    foot_2.geometry = ((-params['foot_width']/2,  params['foot_width']/2, params['foot_width']/2, -params['foot_width']/2, -params['foot_width']/2),
                       (-params['foot_length']/2, -params['foot_length']/2, params['foot_length']/2, params['foot_length']/2, -params['foot_length']/2))
    
    ############# apply translations and rotations to set bodies' initial conditions in world frame #################

    rbs.set_initial_conditions()

    ############ initialize joints ################

    rbs.add_joint('hip_1', thigh_1, thigh_1.sites[0], trunk, trunk.sites[0],
                  params['q_min_hip'], params['q_max_hip'])
    rbs.add_joint('knee_1', thigh_1, thigh_1.sites[1], shank_1, shank_1.sites[0], 
                  params['q_min_knee'], params['q_max_knee'])    
    rbs.add_joint('ankle_1', foot_1, foot_1.sites[0], shank_1, shank_1.sites[1], 
                  params['q_min_ankle'], params['q_max_ankle'])

    rbs.add_joint('hip_2', thigh_2, thigh_2.sites[0], trunk, trunk.sites[0],
                  params['q_min_hip'], params['q_max_hip'])
    rbs.add_joint('knee_2', thigh_2, thigh_2.sites[1], shank_2, shank_2.sites[0], 
                  params['q_min_knee'], params['q_max_knee'])
    rbs.add_joint('ankle_2', foot_2, foot_2.sites[0], shank_2, shank_2.sites[1], 
                  params['q_min_ankle'], params['q_max_ankle'])

    ############ initialize contact points #################
    
    rbs.add_contact('heel_1 contact', foot_1, foot_1.sites[1])
    rbs.add_contact('toe_1 contact', foot_1, foot_1.sites[2])

    rbs.add_contact('heel_2 contact', foot_2, foot_2.sites[1])
    rbs.add_contact('toe_2 contact', foot_2, foot_2.sites[2])

    # initialize states dictionary now that joints and contacts are set up
    rbs.init_states()

    return rbs
