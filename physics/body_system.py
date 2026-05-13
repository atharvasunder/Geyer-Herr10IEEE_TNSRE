from physics.body import Body
from physics.joint import RevoluteJoint
from physics.contact import GroundContact
import config
import math


class RigidBodySystem():
    """
    Class for a rigid body system
    """

    # Constructor of class
    def __init__(self, number_of_bodies=None, name=''):

        self.name = name
        self.anchor = None       # special body that does not move
        self.body_list = []     # rigid bodies that move
        self.joint_list = []    # joints that connect between bodies
        self.contact_list = []  # contact points for ground interaction
        self.initial_conditions = config.INITIAL_CONDITIONS
        self.model_params = config.MODEL_PARAMS

        # current simulation states: joint kinematics and ground reaction forces
        # populated after joints and contacts are added via init_states()
        self.states = {}

    # # add anchor to rigid body system
    # def add_anchor(self, name, x, z, p):
    #     self.anchor = Anchor(name, x, z, p)
    #     return self.anchor

    def add_body(self, name, mass, moment_of_inertia, x_gc, z_gc):
        self.body_list.append(Body(name, mass, moment_of_inertia, x_gc, z_gc))

        return self.body_list[-1]  # return body

    def add_joint(self, name, base_body, base_site, mate_body, mate_site,
                  q_min, q_max):
        self.joint_list.append(RevoluteJoint(name, base_body, base_site, mate_body, mate_site,
                                             q_min, q_max))
        return self.joint_list[-1]

    def add_contact(self, name, base_body, base_site):
        self.contact_list.append(GroundContact(name, base_body, base_site))
        return self.contact_list[-1]

    def retrieve_body(self, name):
        return next((b for b in self.body_list if b.name == name), None)

    def transform_point(self, pos_ab, angle_ab, pos_b):
        """transforms coordinates of point in frame B to frame A (rA = rAB + Cab*rB)"""
        cp, sp = math.cos(angle_ab), math.sin(angle_ab)
        x_a = pos_ab[0] + cp * pos_b[0] - sp * pos_b[1]
        z_a = pos_ab[1] + sp * pos_b[0] + cp * pos_b[1]
        return x_a, z_a

    def set_initial_conditions(self):
        ic = self.initial_conditions
        p = self.model_params

        trunk = self.retrieve_body('trunk')
        thigh_1 = self.retrieve_body('thigh_1')
        shank_1 = self.retrieve_body('shank_1')
        foot_1 = self.retrieve_body('foot_1')
        thigh_2 = self.retrieve_body('thigh_2')
        shank_2 = self.retrieve_body('shank_2')
        foot_2 = self.retrieve_body('foot_2')

        trunk.p = ic['p0'] # initial trunk lean
        
        # Set explicit kinematic conditions since they are no longer handled in Body constructor
        trunk.x = ic['x0']
        trunk.z = ic['h0']
        trunk.vx = ic['vx0']
        trunk.vz = ic['vz0']
        trunk.vp = ic['vp0']

        thigh_1.vx, thigh_1.vz, thigh_1.vp = 0.0, 0.0, 0.0
        thigh_2.vx, thigh_2.vz, thigh_2.vp = 0.0, 0.0, 0.0
        shank_1.vx, shank_1.vz, shank_1.vp = 0.0, 0.0, 0.0
        shank_2.vx, shank_2.vz, shank_2.vp = 0.0, 0.0, 0.0
        foot_1.vx, foot_1.vz, foot_1.vp = 0.0, 0.0, 0.0
        foot_2.vx, foot_2.vz, foot_2.vp = 0.0, 0.0, 0.0
        
        trunk.update_site_coords()  # converts initial site coordinates to world frame

        thigh_1.p = trunk.p - ic['initial_hip_1_angle'] # angle made by thigh with world frame
        pos_ab = (trunk.x + trunk.sites[0].x_b2w, 
                  trunk.z + trunk.sites[0].z_b2w) 
        thigh_1.x, thigh_1.z = self.transform_point(pos_ab, thigh_1.p, p['thigh_com_prox']) 
        thigh_1.update_site_coords()

        thigh_2.p = trunk.p - ic['initial_hip_2_angle']
        pos_ab = (trunk.x + trunk.sites[0].x_b2w, 
                  trunk.z + trunk.sites[0].z_b2w) 
        thigh_2.x, thigh_2.z = self.transform_point(pos_ab, thigh_2.p, p['thigh_com_prox'])
        thigh_2.update_site_coords()

        shank_1.p = thigh_1.p + ic['initial_knee_1_angle']
        pos_ab = (thigh_1.x + thigh_1.sites[1].x_b2w,
                  thigh_1.z + thigh_1.sites[1].z_b2w)
        shank_1.x, shank_1.z = self.transform_point(pos_ab, shank_1.p, p['shank_com_prox'])
        shank_1.update_site_coords()

        shank_2.p = thigh_2.p + ic['initial_knee_2_angle']
        pos_ab = (thigh_2.x + thigh_2.sites[1].x_b2w,
                  thigh_2.z + thigh_2.sites[1].z_b2w)
        shank_2.x, shank_2.z = self.transform_point(pos_ab, shank_2.p, p['shank_com_prox'])
        shank_2.update_site_coords()

        foot_1.p = shank_1.p - ic['initial_ankle_1_angle']
        pos_ab = (shank_1.x + shank_1.sites[1].x_b2w,
                  shank_1.z + shank_1.sites[1].z_b2w)
        foot_1.x, foot_1.z = self.transform_point(pos_ab, foot_1.p, p['foot_com_prox'])
        foot_1.update_site_coords()

        foot_2.p = shank_2.p - ic['initial_ankle_2_angle']
        pos_ab = (shank_2.x + shank_2.sites[1].x_b2w,
                  shank_2.z + shank_2.sites[1].z_b2w)
        foot_2.x, foot_2.z = self.transform_point(pos_ab, foot_2.p, p['foot_com_prox'])
        foot_2.update_site_coords()

    def update_contacts(self, dt, ground_height=0):
        for contact in self.contact_list:
            contact.update(dt, ground_height=ground_height)

    def update_joints(self, dt, torque_array):
        for j_idx, joint in enumerate(self.joint_list):
            joint.update(dt, torque_array[j_idx])

    def integrate_bodies(self, dt):
        for body in self.body_list:
            body.integrate(dt)

        self.update_states()
        return self.states

    def log_data(self):
        for body in self.body_list:
            body.log_data()
        for joint in self.joint_list:
            joint.log_data()

    def _leg_grf_magnitude(self, heel_idx, toe_idx):
        """Compute total GRF magnitude for a leg from its heel and toe contact forces."""
        fx = self.contact_list[heel_idx].base_contact_site.fx + self.contact_list[toe_idx].base_contact_site.fx
        fz = self.contact_list[heel_idx].base_contact_site.fz + self.contact_list[toe_idx].base_contact_site.fz
        return math.sqrt(fx * fx + fz * fz)

    def init_states(self):
        """Populate self.states dict from current joint and contact data.
        Call once after all joints and contacts have been added.
        Structure: states['leg_1'] and states['leg_2'], each with 'q', 'q_dot', 'grf'."""
        self.states = {
            'trunk': {
                'theta': self.retrieve_body('trunk').p,
                'theta_dot': self.retrieve_body('trunk').vp
            },
            'leg_1': {
                # joint angles [rad]
                'q':     {'hip': self.joint_list[0].q, 'knee': self.joint_list[1].q, 'ankle': self.joint_list[2].q},
                # joint angular velocities [rad/s]
                'q_dot': {'hip': self.joint_list[0].q_dot, 'knee': self.joint_list[1].q_dot, 'ankle': self.joint_list[2].q_dot},
                # total ground reaction force magnitude [N] (heel + toe combined)
                'grf': self._leg_grf_magnitude(0, 1),
                # double support flag (set by gait controller)
                'Dsup': 0,
            },
            'leg_2': {
                'q':     {'hip': self.joint_list[3].q, 'knee': self.joint_list[4].q, 'ankle': self.joint_list[5].q},
                'q_dot': {'hip': self.joint_list[3].q_dot, 'knee': self.joint_list[4].q_dot, 'ankle': self.joint_list[5].q_dot},
                'grf': self._leg_grf_magnitude(2, 3),
                'Dsup': 0,
            },
        }

    def update_states(self):
        """Refresh self.states with latest joint and contact values."""
        self.states['trunk']['theta'] = self.retrieve_body('trunk').p
        self.states['trunk']['theta_dot'] = self.retrieve_body('trunk').vp

        # leg 1: joints [0,1,2], contacts [0,1]
        leg1 = self.states['leg_1']
        leg1['q']['hip']     = self.joint_list[0].q
        leg1['q']['knee']    = self.joint_list[1].q
        leg1['q']['ankle']   = self.joint_list[2].q
        leg1['q_dot']['hip']   = self.joint_list[0].q_dot
        leg1['q_dot']['knee']  = self.joint_list[1].q_dot
        leg1['q_dot']['ankle'] = self.joint_list[2].q_dot
        leg1['grf'] = self._leg_grf_magnitude(0, 1)

        # leg 2: joints [3,4,5], contacts [2,3]
        leg2 = self.states['leg_2']
        leg2['q']['hip']     = self.joint_list[3].q
        leg2['q']['knee']    = self.joint_list[4].q
        leg2['q']['ankle']   = self.joint_list[5].q
        leg2['q_dot']['hip']   = self.joint_list[3].q_dot
        leg2['q_dot']['knee']  = self.joint_list[4].q_dot
        leg2['q_dot']['ankle'] = self.joint_list[5].q_dot
        leg2['grf'] = self._leg_grf_magnitude(2, 3)