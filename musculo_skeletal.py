import numpy as np

class musculo_skeletal():
    """
    Class describing the connection between muscle and joints.
    contains 2 functions:
    1. update_lmtc - calculates the new l_mtc length from the joint angles of actuated joints and l_mtc reference length

    Inputs:
    joint_list: list of all joints
    actuated_joint_list: the two joints which the
                          muscle actuates
    location: front or back, used to decide torque contribution on joint's sign

    by definition here, base joint is the connection that is vertically higher
    (output of muscle update on the base joint is positive and on mate joint is negative)    

    lever_params:
    - moment arm constants (eg: r_max)
    - reference joint angles for muscle slack lengths
    - joint angles at which moment arm length is max
    - etc
    
    Out:
    mtc_length
    lever_arms
    joint torques
    """

    # constructor
    def __init__(self, name, joint_list, actuated_joint_list, location, musculoskeletal_params):
        self.name = name    # name of the lever

        self.lever_params = musculoskeletal_params
        self.actuated_joint_list = actuated_joint_list
        self.joint_list = joint_list
        self.location = location # front/back
        
        # initialize muscle-joint properties
        self.r0_array = musculoskeletal_params.r0_array # size = no of joints (contains zeros for unactuated joints)
        self.q_ref_array = musculoskeletal_params.q_ref_array # joint angle at which MTU length = l_opt + l_slack
        self.q_max_array = musculoskeletal_params.q_max_array # joint angle at which r = r0
        self.rho_array = musculoskeletal_params.rho_array # rho ensures that the MTU fiber length stays within physiological limits
        self.q_array = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.r_array = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

        self.l_ref_mtc = musculoskeletal_params.l_ref_mtc
        
        # torque contributions of muscle on each joint
        self.torque_array = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

        ############## store moment arm values for muscle at each joint ###################
        
        for j in range(len(joint_list)):
            joint = joint_list[j]
            r0 = self.r0_array[j]
            q = joint.q
            q_ref = self.q_ref_array[j] # reference values of q
            q_max = self.q_max_array[j]
            self.q_array[j] = q
            
            if joint.name in self.actuated_joint_list:
                if joint.name == 'hip_1' or joint.name == 'hip_2':
                    self.r_array[j] = r0
                else:
                    self.r_array[j] = r0*np.cos(q - q_max)

        print('r_array: ', self.r_array)
        self.r_array_initial = self.r_array     # store initial lever arm for use in reset function``

        ############### for data logging ####################

        # self.moment_arms = [self.r_ref_array] # will be a list of arrays
        self.torque_contributions = [self.torque_array] # List of arrays of torque contribution of muscle on each joint

    def update_lever_arm(self):
        # compute r_array for 1 muscle
        for j in range(len(self.joint_list)):
            joint = self.joint_list[j]
            r0 = self.r0_array[j]
            q = joint.q
            q_max = self.q_max_array[j]

            # keep updating lever arm for each joint
            if joint.name in self.actuated_joint_list:
                if joint.name == 'hip_1' or joint.name == 'hip_2':
                    self.r_array[j] = r0
                else:
                    self.r_array[j] = r0*np.cos(q - q_max)
        
        return self.r_array

    def update_l_mtc(self):

        l_mtc = self.l_ref_mtc  # initialize l_mtc then update with each joints contribution
        # compute l_mtc instantaneous length of 1 muscle!
        for j in range(len(self.joint_list)):
            joint = self.joint_list[j]
            r0 = self.r0_array[j]
            rho = self.rho_array[j]
            q = joint.q
            q_ref = self.q_ref_array[j]
            q_max = self.q_max_array[j]
            self.q_array[j] = q

            # keep updating l_mtc with each joints contribution
            if joint.name in self.actuated_joint_list:
                if joint.name == 'hip_1' or joint.name == 'hip_2':
                    if self.location == 'front':    # increasing joint angle may increase or decrease l_mtc based on muscle location
                        l_mtc += rho*r0*(q - q_ref)
                    elif self.location == 'back':
                        l_mtc -= rho*r0*(q - q_ref)
                elif joint.name == 'knee_1' or joint.name == 'knee_2':
                    if self.location == 'front':
                        l_mtc -= rho*r0*(np.sin(q - q_max) - np.sin(q_ref - q_max))
                    elif self.location == 'back':
                        l_mtc += rho*r0*(np.sin(q - q_max) - np.sin(q_ref - q_max))
                elif joint.name == 'ankle_1' or joint.name == 'ankle_2':
                    if self.location == 'front':
                        l_mtc += rho*r0*(np.sin(q - q_max) - np.sin(q_ref - q_max))
                    elif self.location == 'back':
                        l_mtc -= rho*r0*(np.sin(q - q_max) - np.sin(q_ref - q_max))
            
        return l_mtc
    
    def get_torques(self, F_mtc):

        for j in range(len(self.joint_list)):
            r = self.r_array[j]
            if self.location == 'front':
                self.torque_array[j] = -r*F_mtc
            elif self.location == 'back':
                self.torque_array[j] = r*F_mtc

        return self.torque_array
    
    def reset(self):
        self.torque_array = np.array([0,0,0,0,0,0])
        self.r_array = self.r_array_initial
    
    def log_data(self):
        # self.moment_arms.append(self.r_array)
        self.torque_contributions.append(self.torque_array)