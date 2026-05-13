import config

class GroundContact():
    """
    Class describing ground contacts. It updates forces 
    produced at contact sites based on the position and
    velocity of the contact sites relative to the ground.
    """

    # constructor
    def __init__(self, name, base_body, base):
        self.name = name
        self.base_body = base_body  # home body of contact site
        self.base_contact_site = base # contact site object

        self.base_x = self.base_body.x + self.base_contact_site.x_b2w  # contact site location in world frame
        self.base_z = self.base_body.z + self.base_contact_site.z_b2w
        
        self.contact_x = self.base_x    # initial x-coordinate of contact point in sim
        self.ground_height = 0.0        # ground height
        self.contact = False            # true if in ground contact

        self.sliding_mode = True     # horizontal friction state, initialized true implying sliding friction

        self.x_stick = 0.0        # horizontal position at start of stiction
        
        self.kz = config.CONTACT_PARAMS['stiffness_z']
        self.max_vz = config.CONTACT_PARAMS['max_vz']
        self.v_trans = config.CONTACT_PARAMS['v_transition']
        self.mu_slide = config.CONTACT_PARAMS['mu_slide']
        self.kx = config.CONTACT_PARAMS['stiffness_x']
        self.max_vx = config.CONTACT_PARAMS['max_vx']
        self.mu_stick = config.CONTACT_PARAMS['mu_stick']

    # update ground contact
    def update(self, dt, ground_height):

        self.ground_height = ground_height  # for reporting only

        # compute current contact site location in world frame
        base_x = self.base_body.x + self.base_contact_site.x_b2w
        base_z = self.base_body.z + self.base_contact_site.z_b2w

        # check for contact
        if base_z < ground_height:
            self.contact = True

            # compute vertical ground reaction force
            dist_z = ground_height - base_z
            base_vz = (base_z - self.base_z) / dt
            base_vx = (base_x - self.base_x) / dt

            damping_multiplier = max(0.0, 1 - base_vz / self.max_vz)
            damping_multiplier = min(damping_multiplier, 2.0) # Clamp to prevent explosion
            self.base_contact_site.fz = self.kz * dist_z * damping_multiplier

            # compute horizontal ground reaction force
            if self.sliding_mode is True: # initialized as true,
                # compute as sliding mode friction, oppose direction of motion
                if base_vx > 0:
                    self.base_contact_site.fx = - self.mu_slide * self.base_contact_site.fz
                elif base_vx < 0:
                    self.base_contact_site.fx = self.mu_slide * self.base_contact_site.fz
                else:
                    self.base_contact_site.fx = 0

                # check for friction mode transition
                if abs(base_vx) < self.v_trans:
                    self.sliding_mode = False
                    self.x_stick = base_x  # store x position when stiction engages
            else:
                # compute as stiction
                dist_x = base_x - self.x_stick
                self.base_contact_site.fx = self.nonlin_stiction_model(dist_x, base_vx, self.kx, self.max_vx)

                # check for friction mode transition (occues if force greater than mu*normal)
                if abs(self.base_contact_site.fx) > self.mu_stick * self.base_contact_site.fz:
                    self.base_contact_site.fx = (self.base_contact_site.fx/abs(self.base_contact_site.fx))*self.mu_stick * self.base_contact_site.fz # captures sign
                    self.sliding_mode = True
        else:
            # check for transition out of contact
            if self.contact is True:
                self.contact = False
                self.base_contact_site.fx, self.base_contact_site.fz = 0.0, 0.0  # reset GRFs
                self.sliding_mode = True  # preset sliding model
            
        # store location to compute base velocity again next time
        self.base_x, self.base_z = base_x, base_z

    # compute stiction force as linear spring damper
    def lin_stiction_model(self, dist_x, vx, k, b):
        return - k * dist_x - b * vx

    # compute stiction force as nonlinear spring damper
    def nonlin_stiction_model(self, dist, v, k, max_v):
        if dist >= 0: # ensures that if mass moves away from stiction point, spring and damper force act in same direction.
            fx = - k * dist * max(0.0, 1 + v / max_v) # prevent changing sign
        else:
            fx = - k * dist * max(0.0, 1 - v / max_v) # prevent changing sign
        
        return fx