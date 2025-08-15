import math
pi = math.pi

class RevoluteJoint():
    """
    Class describing a revolute joint between two bodies. 

    In: 
    base_body, mate_body: the two bodies between which the
                          joint is located
    base, mate: the specific connection sites at these bodies
    stiffness: the constraint force stiffness and limit torque stiffness
    damping: damping parameters

    Out: n/a

    The joint is defined such that a positive joint angle describes 
    a counter clock-wise rotation of the mate body relative to the 
    base body.

    """

    # constructor
    def __init__(self, name, base_body, base, base_type, mate_body, mate, q_min, q_max, q_dot_max, stiffness, damping, k_lim):
        self.name = name
        self.base = base            # base site of joint
        self.base_type = base_type  # proximal or distal base body
        self.mate = mate            # mate site of joint
        self.base_body = base_body  # body to which base site belongs
        self.mate_body = mate_body  # body to which mate site belongs

        # print(self.mate_body)

        self.k = stiffness            # constraint stiffness and
        self.b = damping              # damping coefficient
        self.k_lim = k_lim            # for limit torques
        self.dist_x, self.dist_z = \
            self.site_distance()      # initial site distance

        self.tau = 0.0                                # joint torque, initialized as 0
        self.q = self.mate_body.p - self.base_body.p  # joint angle (by definition for all joints)
        self.q_dot = 0 # just for debugging
        self.q_min = q_min # joint angle limits
        self.q_max = q_max
        self.q_dot_max = q_dot_max
        
        self.sep_x_dot = 0
        self.sep_z_dot = 0
        
        # for data logging
        
        self.q_list = [self.q]
        self.q_dot_list = [self.q_dot]
        self.tau_list = [self.tau]

    # compute joint forces and torques based on relative position, 
    # velocity and angle between base and mate bodies
    def update(self, dt, tau):

        dist_x, dist_z = self.site_distance()
        self.sep_x_dot = (dist_x - self.dist_x) / dt
        self.sep_z_dot = (dist_z - self.dist_z) / dt
        # joint forces
        self.base.fx = self.k * dist_x \
            + self.b * self.sep_x_dot
        self.base.fz = self.k * dist_z \
            + self.b * self.sep_z_dot

        self.mate.fx = -self.base.fx
        self.mate.fz = -self.base.fz

        # update site distances, 
        # done to calculate the velocity of the body. This uses prev timesteps positions as input
        # to compute instantaneous velocity (which helps compute damper force)
        self.dist_x, self.dist_z = dist_x, dist_z 

        # assign joint angle and torque
        q = self.mate_body.p - self.base_body.p

        q_dot = (q - self.q) / dt
        self.q = q
        self.q_dot = q_dot
        
        if self.base_type == 'proximal':
            self.base.tau = tau
            self.mate.tau = -tau # here the base and mate sites are updated with the newly computed joint torque and forces!
        elif self.base_type == 'distal':
            self.base.tau = -tau
            self.mate.tau = tau

        # computing joint limit torques
        if q < self.q_min and q_dot > self.q_dot_max: # joint angle is lower than min but increasing fast to return to safe zone

            tau += 0
        elif q < self.q_min and q_dot < self.q_dot_max: # joint angle is lower than min 
                                                        # but increasing v-slowly/decreasing: apply restoring torque
            self.mate.tau += self.k_lim*(self.q_min - q)*(1 - q_dot/self.q_dot_max)
            self.base.tau -= self.k_lim*(self.q_min - q)*(1 - q_dot/self.q_dot_max)

        
        elif q > self.q_max and q_dot < -self.q_dot_max: # joint angle is > max but decreasing fast to return to safe zone
            tau += 0

        elif q > self.q_max and q_dot > -self.q_dot_max: # joint angle is > max but decreasing slow/increasing: appluy restoring torque
            self.mate.tau += self.k_lim*(self.q_max - q)*(1 + q_dot/self.q_dot_max)
            self.base.tau -= self.k_lim*(self.q_max - q)*(1 + q_dot/self.q_dot_max)

        self.tau = tau
 
    # get position of the joint sites in the world frame coordinates
    def site_distance(self):
        base_x = self.base_body.x + self.base.x_b2w
        base_z = self.base_body.z + self.base.z_b2w
        mate_x = self.mate_body.x + self.mate.x_b2w
        mate_z = self.mate_body.z + self.mate.z_b2w

        return mate_x - base_x, mate_z - base_z  # deltaX , deltaY
    
    def log_data(self):
        self.tau_list.append(self.tau)
        self.q_list.append(self.q)
        self.q_dot_list.append(self.q_dot)