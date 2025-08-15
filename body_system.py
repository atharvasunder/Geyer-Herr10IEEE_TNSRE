from bodies import Body
# from bodies import Anchor


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
        self.muscle_list = []  # muscles to actuate bodies and control joints
        self.feedback_list = [] # feedback pathways for stimulating muscles
        self.musculo_skeletal_list = [] # connections between the muscles and bones (lever arms, mtc length computation)

    # # add anchor to rigid body system
    # def add_anchor(self, name, x, z, p):
    #     self.anchor = Anchor(name, x, z, p)
    #     return self.anchor

    # add body to rigid body system list
    def add_body(self, name, mass, moment_of_inertia,
                 x, z, p, vx, vz, vp, x_gc, z_gc):
        self.body_list.append(Body(name, mass, moment_of_inertia,
                                   x, z, p, vx, vz, vp, x_gc, z_gc))

        return self.body_list[-1]  # return body