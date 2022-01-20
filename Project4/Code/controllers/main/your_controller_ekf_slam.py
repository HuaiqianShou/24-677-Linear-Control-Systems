# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM
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
        
        self.counter = 0
        np.random.seed(99)

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


    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -350., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y

    def update(self, timestep, driver):

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

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)
        
        
        
        
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

        # You are free to reuse or refine your code from P3 in the spaces below.

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
        
        # Setting brake intensity is enabled by passing
        # the driver object, which is used to provide inputs
        # to the car, to our update function
        # Using this function is purely optional.
        # An input of 0 is no brakes applied, while
        # an input of 1 is max brakes applied

        # driver.setBrakeIntensity(clamp(someValue, 0, 1))

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta