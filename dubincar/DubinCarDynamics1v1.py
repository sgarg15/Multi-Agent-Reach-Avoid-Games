import heterocl as hcl
import numpy as np

"""
6D 1v1 Dubin Car Dynamics Implementation

xA = speedA * cos(thetaA)
yA = speedA * sin(thetaA)
thetaA = uA

xD = speedD * cos(thetaD)
yD = speedD * sin(thetaD)
thetaD = uD

Parameters:
    speed: float
        The speed of the car
    uA: float
        The control input for the attacker.
    uD: float
        The control input for the defender.
"""

class DubinCar1v1:
    def __init__(self, x = [0, 0, 0, 0, 0, 0], uMin = -1, uMax = 1, dMin = -1, dMax = 1, uMode = "min", dMode = "max", speedA = 1, speedD = 1):

        print("Initializing the 1v1 Dubin Car")
        # x = [xA, yA, thetaA, xD, yD, thetaD]
        self.x = x

        # Control bounds
        self.uMin = uMin
        self.uMax = uMax

        self.dMin = dMin
        self.dMax = dMax
        
        # Control modes, which represents whether the control is trying to reach a target set or avoid it. 
        # Minimizing the value of the value function means that the control is trying to reach the target set.
        # Maximizing the value of the value function means that the control is trying to avoid the target set. 
        self.uMode = uMode
        self.dMode = dMode
        
        # Speed of the attacker and defender
        self.speedA = speedA
        self.speedD = speedD


    def dynamics(self,t, state, optU, optD):
        """ Dynamics of the 1v1 Dubin Car
        Parameters:
            t: float
                time
            state: list
                state of the system
            optU: list
                control input for the attacker
            optD: list
                control input for the defender
        Returns:
            list
                the dynamics of the 1v1 Dubin Car, which includes the position and orientation of the attacker and defender
        """
        print("Computing the dynamics")
        
        xA = hcl.scalar(0, "xA")
        yA = hcl.scalar(0, "yA") 
        thetaA = hcl.scalar(0, "thetaA")
        xD = hcl.scalar(0, "xD")
        yD = hcl.scalar(0, "yD")
        thetaD = hcl.scalar(0, "thetaD")

        currentThetaA = state[2]

        # Attacker dynamics, describing the motion of the attacker
        xA[0] = self.speedA * hcl.cos(currentThetaA)
        yA[0] = self.speedA * hcl.sin(currentThetaA)
        thetaA[0] = optU[0]

        currentThetaD = state[5]

        # Defender dynamics, describing the motion of the defender
        xD[0] = self.speedD * hcl.cos(currentThetaD)
        yD[0] = self.speedD * hcl.sin(currentThetaD)
        thetaD[0] = optD[0]

        return (xA[0], yA[0], thetaA[0], xD[0], yD[0], thetaD[0])

    def opt_ctrl(self, t, state, spat_deriv):
        """ Optimal control of the 1v1 Dubin Car
        Parameters:
            t: float
                time
            state: list
                the current state of the system
            spat_deriv: list
                spatial derivative of the system, meaning the derivative of the value function with respect to the state
        Returns:
            list
                the optimal control of the 1v1 Dubin Car
        """
        print("Computing the optimal control (hcl)")

        optU = hcl.scalar(self.uMax, "optU")

        #Temporary control variable
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        # The following calculations are used to determine the optimal control of the attacker 
        with hcl.if_(spat_deriv[2] > 0):
            # If the uMode is "min", then the control is trying to reach the target set and the optimal control is the minimum control input value
            # Therefore set the optimal control to the minimum control input value
            with hcl.if_(self.uMode == "min"):
                optU[0] = -optU

        with hcl.elif_(spat_deriv[2] < 0):
            # If the uMode is "max", then the control is trying to avoid the target set and the optimal control is the maximum control input value
            # Therefore set the optimal control to the maximum control input value
            with hcl.if_(self.uMode == "max"):
                optU[0] = -optU

        return (optU[0], in2[0], in3[0], in4[0])
    
    def opt_dstb(self, t, state, spat_deriv):
        """ Optimal disturbance of the 1v1 Dubin Car
        Parameters:
            t: float
                time
            state: list
                the current state of the system
            spat_deriv: list
                spatial derivative of the system, meaning the derivative of the value function with respect to the state
        Returns:
            list
                the optimal disturbance of the 1v1 Dubin Car
        """
        print("Computing the optimal disturbance (hcl)")

        optD = hcl.scalar(self.dMax, "optD")

        #Temporary control variable
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        # The following calculations are used to determine the optimal disturbance of the defender 
        with hcl.if_(spat_deriv[5] > 0):
            # If the dMode is "min", then the control is trying to reach the target set and the optimal disturbance is the minimum disturbance input value
            # Therefore set the optimal disturbance to the minimum disturbance input value
            with hcl.if_(self.dMode == "min"):
                optD[0] = -optD

        with hcl.elif_(spat_deriv[5] < 0):
            # If the dMode is "max", then the control is trying to avoid the target set and the optimal disturbance is the maximum disturbance input value
            # Therefore set the optimal disturbance to the maximum disturbance input value
            with hcl.if_(self.dMode == "max"):
                optD[0] = -optD

        return (optD[0], in2[0], in3[0], in4[0])
    
    def optCntrl_inPython(self, spat_deriv):
        """ Optimal control of the 1v1 Dubin Car in Python
        Parameters:
            spat_deriv: list
                spatial derivative of the system, meaning the derivative of the value function with respect to the state
        Returns:
            list
                the optimal control of the 1v1 Dubin Car
        """
        print("Computing the optimal control")

        optU = self.uMax

        # The following calculations are used to determine the optimal control of the attacker 
        if spat_deriv[2] > 0:
            # If the uMode is "min", then the control is trying to reach the target set and the optimal control is the minimum control input value
            # Therefore set the optimal control to the minimum control input value
            if self.uMode == "min":
                optU = -optU

        elif spat_deriv[2] < 0:
            # If the uMode is "max", then the control is trying to avoid the target set and the optimal control is the maximum control input value
            # Therefore set the optimal control to the maximum control input value
            if self.uMode == "max":
                optU = -optU

        return optU 
    
    def optDstb_inPython(self, spat_deriv):
        """ Optimal disturbance of the 1v1 Dubin Car in Python
        Parameters:
            spat_deriv: list
                spatial derivative of the system, meaning the derivative of the value function with respect to the state
        Returns:
            list
                the optimal disturbance of the 1v1 Dubin Car
        """
        print("Computing the optimal disturbance")

        optD = self.dMax

        # The following calculations are used to determine the optimal disturbance of the defender 
        if spat_deriv[5] > 0:
            # If the dMode is "min", then the control is trying to reach the target set and the optimal disturbance is the minimum disturbance input value
            # Therefore set the optimal disturbance to the minimum disturbance input value
            if self.dMode == "min":
                optD = -optD

        elif spat_deriv[5] < 0:
            # If the dMode is "max", then the control is trying to avoid the target set and the optimal disturbance is the maximum disturbance input value
            # Therefore set the optimal disturbance to the maximum disturbance input value
            if self.dMode == "max":
                optD = -optD

        return optD
    
    def capture_set(self, grid, capture_radius, mode):
        """ Capture set of the 1v1 Dubin Car

        Parameters:
            grid: Grid
                grid of the system
            capture_radius: float
                the radius of the capture set
            mode: string
                the mode of the capture set, which can be "capture" or "escape"
        
        Returns:
            list
                the capture set of the 1v1 Dubin Car
        """

        print("Computing the capture set")

        # The following variables are used to represent the state of the attacker and defender 
        xA, yA, xD, yD = np.meshgrid(grid.grid_points[0], grid.grid_points[1], grid.grid_points[2], grid.grid_points[3], indexing='ij')

        # The following calculations are used to determine the distance between the attacker and defender
        dist = np.sqrt(np.power(xA - xD, 2) + np.power(yA - yD, 2))

        if mode == "capture":
            # If the mode is "capture", then the capture set is the set of states where the distance between the attacker and defender 
            # is less than the capture radius
            capture_set = dist - capture_radius
            return capture_set
        elif mode == "escape":
            # If the mode is "escape", then the capture set is the set of states where the distance between the attacker and defender 
            # is greater than the capture radius
            capture_set = capture_radius - dist
            return capture_set
    
