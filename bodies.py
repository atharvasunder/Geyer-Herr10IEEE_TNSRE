import math

class Site():
    """
    Class for connection sites at rigid bodies. Interaction classes
    such as joints, ground contacts, and external actuators use 
    connection sites to define forces and torques acting on a body.
    """

    # constructor
    def __init__(self, name, x_b, z_b):
        self.name = name
        self.x_b = x_b  # coords in body frame
        self.z_b = z_b
        self.x_b2w = 0.0  # coords mapped to world frame
        self.z_b2w = 0.0

        self.fx = 0.0     # forces and moments acting at site
        self.fz = 0.0
        self.tau = 0.0

class Body():
    """
    Class for rigid bodies in the plane. The rigid body is defined by 
    its total mass m at the coordinate origin, a moment of inertia In 
    about it, and connections sites at which other elements apply 
    forces and torques.

    In addition, a body instance stores auxiliary information such
    as the body's geometry used in animations.
    """

    g = 9.8067  # gravitational acceleration

    # construct 2-D rigid body
    def __init__(self, name, mass, moment_of_inertia,
                 x, z, p, vx, vz, vp, x_gc, z_gc):
        self.name = name

        # (x, z, p): CoM pos and body pitch
        self.x = x
        self.z = z
        self.p = p
        
        # for inhibiting motion in x-direction, only useful in this sim
        self.x_start = self.x

        # (vx, vz, vp): CoM vel and body angvel
        self.vx = vx 
        self.vz = vz
        self.vp = vp

        self.x_gc = x_gc
        self.z_gc = z_gc

        cp, sp = math.cos(self.p), math.sin(self.p)
        self.x_geom = self.x + cp * self.x_gc - sp * self.z_gc
        self.z_geom = self.z + sp * self.x_gc + cp * self.z_gc

        self.m = mass
        self.In = moment_of_inertia  # about CoM along perpendicular-axis

        self.sites = []
        self.geometry = []  # convex hull points for animation
        
        # for plotting
        self.x_list = [self.x]
        self.z_list = [self.z] 
        self.p_list = [self.p]
        
        self.vx_list = [self.vx]
        self.vz_list = [self.vz]
        self.vp_list = [self.vp]

    # create and add connection site to the body
    def add_site(self, name, x_b, z_b):
        self.sites.append(Site(name, x_b, z_b))

    # map body coords of all sites in a specific body into world frame coords
    def update_site_coords(self):

        # The transformed coordinates are stored at the sites and later
        # used in the integration method of the RgidBody2D class when
        # computing the net torque acting on a body:
        # tau_net = tau + x_wrld * fz - z_wrld * fx
        
        cp, sp = math.cos(self.p), math.sin(self.p)

        for site in self.sites:
            site.x_b2w = cp * site.x_b - sp * site.z_b
            site.z_b2w = sp * site.x_b + cp * site.z_b

    # integrate equations of motion, done for one body in this function
    def integrate(self, dt):
    
        net_fx, net_fz, net_tau = 0.0, 0.0, 0.0

        # update site coordinates ()
        self.update_site_coords()

        for site in self.sites:
            net_fx += site.fx
            net_fz += site.fz
            
            # if self.name == 'trunk':
            #     dist = self.x - self.x_start
            #     net_fx += self.nonlin_stiction_model(dist, self.vx, 8000, 0.05)  # dont let the trunk move sideways
            
            net_tau += site.tau \
                + site.x_b2w * site.fz - site.z_b2w * site.fx

        # perform one step of integration on dynamics using forward euler integration
        self.x = self.x + self.vx * dt
        self.vx = self.vx + (net_fx / self.m) * dt
        
        self.z = self.z + self.vz * dt
        self.vz = self.vz + (net_fz / self.m - self.g) * dt  # accounts for gravity here
        
        self.p = self.p + self.vp * dt
        self.vp = self.vp + (net_tau / self.In) * dt # angular velocity update

    # get body geometry in the world frame
    def update_body_geometry(self, x, z, p):
        cp, sp = math.cos(p), math.sin(p)

        # update geometric center of the body in world frame
        # using the vector from com to geometric center in body frame
        # and orientation of body
        self.x_geom = x + cp * self.x_gc - sp * self.z_gc
        self.z_geom = z + sp * self.x_gc + cp * self.z_gc

        # compute coordinates of every corner of the body in world frame
        x_w = [self.x_geom + cp * x - sp * z for x, z
               in zip(self.geometry[0], self.geometry[1])]
        z_w = [self.z_geom + sp * x + cp * z for x, z
               in zip(self.geometry[0], self.geometry[1])]

        return x_w, z_w
    
    # def nonlin_stiction_model(self, dist, v, k, max_v):
    #     if dist >= 0: # imp to do this, as it ensures that if mass moves away
    #     # from stiction point, spring and damper force act in same direction.
        
    #     # here is v > v_max in magnitude, should we transition????
    #         fx = - k * dist * (1 + v / max_v)  # * (v > - max_v)
    #     else:
    #         fx = - k * dist * (1 - v / max_v)  # * (v < max_v)
        
    #     return fx
    
    def log_data(self):
        
        self.x_list.append(self.x)
        self.z_list.append(self.z)
        self.p_list.append(self.p)
        
        self.vx_list.append(self.vx)
        self.vz_list.append(self.vz)
        self.vp_list.append(self.vp)

# class Anchor():
#     """
#     Class for defining imovable objects in the plane. Thess objects can
#     have sites attached with which other bodies can interact through
#     joints and contact points.
#     """

#     # constructor
#     def __init__(self, name, x, z, p):
#         self.name = name

#         self.x = x    # reference position and orientation
#         self.z = z    # in inertial frame
#         self.p = p

#         self.sites = []  # list of connection sites attached to anchor

#         self.geometry = []  # convex hull points for animation

#     # create and add connection site to the anchor
#     def add_site(self, name, x_b, z_b):
#         self.sites.append(Site(name, x_b, z_b))
#         self.update_site_coords()

#     # map body coords of all sites into world frame coords
#     def update_site_coords(self):
#         cp, sp = math.cos(self.p), math.sin(self.p)

#         for site in self.sites:  # <1>
#             site.x_b2w = cp * site.x_b - sp * site.z_b
#             site.z_b2w = sp * site.x_b + cp * site.z_b

#     # body geometry position in the world frame
#     def update_body_geometry(self):
#         cp, sp = math.cos(self.p), math.sin(self.p)
#         x_w = [self.x + cp * x - sp * z for x, z
#                 in zip(self.geometry[0], self.geometry[1])]
#         z_w = [self.z + sp * x + cp * z for x, z
#                 in zip(self.geometry[0], self.geometry[1])]

#         return x_w, z_w