# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
import math

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Add additional member variables according to your need here.
        
        self.error_ki_delta = 0
        self.previous_error_delta = 0
        self.error_ki_F = 0
        self.previous_error_F = 0
        
        
        self.kp_delta = 1
        self.ki_delta = 0
        self.kd_delta = 0.1
        
        
        self.kp_F = 5
        self.ki_F = 0
        self.kd_F = 0.1
        
        
  

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        kp_delta = self.kp_delta
        ki_delta = self.ki_delta
        kd_delta = self.kd_delta
        kp_F = self.kp_F
        ki_F = self.ki_F
        kd_F = self.kd_F 

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        
 
        dis, ind = closestNode(X,Y,trajectory)

       
        X_desired = trajectory[ind + 20,0]
        Y_desired = trajectory[ind + 20,1]
        
        
        psi_desired = np.arctan2(Y_desired - Y, X_desired - X)

               
        error_delta = wrapToPi(psi_desired - psi)          
        error_F = math.sqrt(np.power(X_desired - X, 2) + np.power(Y_desired - Y, 2))/delT
       
 
        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
      
        """
        Please design your lateral controller below
 
        """
        self.error_ki_delta += error_delta * delT
        error_kd_delta = (error_delta - self.previous_error_delta) / delT
        self.previous_error_delta = error_delta       
        delta = kp_delta * error_delta + ki_delta * self.error_ki_delta + kd_delta * error_kd_delta
        

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.

        """
        self.error_ki_F += error_F * delT
        error_kd_F = (error_F - self.previous_error_F) / delT
        self.previous_error_F = error_F
        F = kp_F * error_F + ki_F * self.error_ki_F + kd_F * error_kd_F

        return X, Y, xdot, ydot, psi, psidot, F, delta
