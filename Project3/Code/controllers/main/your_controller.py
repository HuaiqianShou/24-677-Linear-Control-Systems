# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
import math
from scipy.ndimage import gaussian_filter1d

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
        
        self.kp = 400
        self.ki = 8
        self.kd = -0.01
        self.error_dis = 0
        
        
        self.curve = self.computeCurvature()
    def computeCurvature(self):
        # Function to compute and return the curvature of a trajectory.
        sigmaGauss = 5 # We can change this value to increase filter strength
        trajectory = self.trajectory
        xp = gaussian_filter1d(input=trajectory[:,0],sigma=sigmaGauss,order=1)
        xpp = gaussian_filter1d(input=trajectory[:,0],sigma=sigmaGauss,order=2)
        yp = gaussian_filter1d(input=trajectory[:,1],sigma=sigmaGauss,order=1)
        ypp = gaussian_filter1d(input=trajectory[:,1],sigma=sigmaGauss,order=2)
        curve = np.zeros(len(trajectory))
        for i in range(len(xp)):
            curve[i] = (xp[i]*ypp[i] - yp[i]*xpp[i])/(xp[i]**2 + yp[i]**2)**1.5
            
        return curve

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
        jia = 0
        if ind < len(trajectory) - 100:
            jia = 100
        else:
            jia = len(trajectory) - ind - 1
            
        if ind < len(trajectory) - 150:
            lakua = 150
        else:
            lakua = len(trajectory) - ind - 1
       
        X_desired = trajectory[ind + jia,0]
        Y_desired = trajectory[ind + jia,1]
        
        X_curv = trajectory[ind + lakua,0]
        Y_curv = trajectory[ind + lakua,1]
        
        X_pre = trajectory[ind,0]
        Y_pre = trajectory[ind,1]
        psi_desired = np.arctan2(Y_desired - Y_pre, X_desired - X_pre)
        psi_curv = np.arctan2(Y_curv - Y_pre, X_curv - X_pre)
        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        # ---------------|Lateral Controller|-------------------------
        A = np.array([[0, 1, 0, 0], [0, -4*Ca / (m * xdot), 4*Ca/m, (-2*Ca*(lf - lr))/(m*xdot)], [0, 0, 0, 1], [0, (-2*Ca*(lf - lr)) / (Iz * xdot), (2*Ca*(lf - lr)) / Iz, (-2*Ca*(np.power(lf, 2) + np.power(lr, 2))) / (Iz * xdot)]])
        B = np.array([[0], [2*Ca / m], [0], [(2 * Ca* lf) / Iz]])
        C = np.identity(4)
        D = np.array([[0],[0],[0],[0]])
        CT = signal.StateSpace(A,B,C,D)
        DT = CT.to_discrete(delT)
        A = DT.A
        B = DT.B
       
        
        haha = (Y-Y_curv)*np.cos(psi_curv)-(X-X_curv)*np.sin(psi_curv)
        #print(self.curve[ind + jia])
        e1 = (Y-Y_desired)*np.cos(psi_desired)-(X-X_desired)*np.sin(psi_desired)
        e2 = wrapToPi(psi - psi_desired)
        
        e1d = xdot * e2 + ydot
        psidot_desired = xdot*self.curve[ind + jia]
        e2d = psidot - psidot_desired
        if(abs(e2)>0.11):
            Vx = 5
        else:
            Vx = 18
        
        
        
       
        Q = np.eye(4)
        R = 1
        

        S = linalg.solve_discrete_are(A, B, Q, R)
        K= -np.linalg.inv(B.T@S@B+R)@B.T@S@A
        
        
        
        self.error_dis = e1
       

        e = np.array([e1,e1d,e2,e2d])
        delta = float(K @ e)
        
        
        

        # ---------------|Longitudinal Controller|-------------------------
        error_kp = Vx - xdot
        
        
        self.error_ki += error_kp * delT 
        
        
        error_kd = (error_kp - self.previous_error_kp)/delT
        self.previous_error_kp = error_kp
        
        
        F = kp * error_kp + ki * self.error_ki + kd * error_kd

        # Return all states and calculated control inputs (F, delta)
        
       
        return X, Y, xdot, ydot, psi, psidot, F, delta
