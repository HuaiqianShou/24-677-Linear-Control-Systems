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
        
        self.previous_error_kp = 0
        self.error_ki = 0
        self.previous_delta = 0
        
        self.kp = 650
        self.ki = 180
        self.kd = 3

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        
        kp = self.kp
        kd = self.ki
        ki = self.kd 
        

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        
        
        
        
        dis, ind = closestNode(X,Y,trajectory)

       
        X_desired = trajectory[ind + 20,0]
        Y_desired = trajectory[ind + 20,1]
        
        
        psi_desired = np.arctan2(Y_desired - Y, X_desired - X)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        A = np.array([[0, 1, 0, 0], [0, -4*Ca / (m * xdot), 4*Ca/m, (-2*Ca*(lf - lr))/(m*xdot)], [0, 0, 0, 1], [0, (-2*Ca*(lf - lr)) / (Iz * xdot), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(np.power(lf, 2) + np.power(lr, 2))) / (Iz * xdot)]])
        B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]])
        
       
       
        
        P = np.array([-6, -0.2, -2, -0.25])  
        
        Vx = 6
        
        
        
        pole = signal.place_poles(A, B, P)
        
        K = pole.gain_matrix
        
        e1 = math.sqrt(np.power(X_desired - X, 2) + np.power(Y_desired - Y, 2))
        
        e2 = wrapToPi(psi - psi_desired)
        
        e1d = xdot * e2 + ydot
        
        e2d = psidot

        e = np.hstack((e1, e1d, e2, e2d))
        
        
        delta = wrapToPi(-np.asscalar(np.matmul(K, e)))
        

     
        

        # ---------------|Longitudinal Controller|-------------------------
 
 
        error_kp = Vx - xdot
        
        
        self.error_ki += error_kp * delT
        
        
        error_kd = (error_kp - self.previous_error_kp)/delT
        self.previous_error_kp = error_kp
        
        
        F = kp * error_kp + ki * self.error_ki + kd * error_kd


        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
