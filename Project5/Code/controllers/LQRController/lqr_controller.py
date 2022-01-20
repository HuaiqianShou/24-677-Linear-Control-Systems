# LQR optimal controller

# Import libraries
import numpy as np
from base_controller import BaseController
from lqr_solver import dlqr, lqr
from scipy.linalg import solve_continuous_lyapunov, solve_lyapunov, solve_discrete_lyapunov
from math import cos, sin
import numpy as np
from scipy import signal

class LQRController(BaseController):
    """ The LQR controller class.

    """

    def __init__(self, robot, lossOfThurst=0):
        """ LQR controller __init__ method.

        Initialize parameters here.

        Args:
            robot (webots controller object): Controller for the drone.
            lossOfThrust (float): percent lost of thrust.

        """
        super().__init__(robot, lossOfThurst)

        # define integral error
        self.int_e1 = 0
        self.int_e2 = 0
        self.int_e3 = 0
        self.int_e4 = 0
        self.delT = 0.01
        # define K matrix
        self.K = None

    def initializeGainMatrix(self):
        """ Calculate the gain matrix.

        """

        # ---------------|LQR Controller|-------------------------
        # Use the results of linearization to create a state-space model

        n_p = 12 # number of states
        m = 4 # number of integral error terms

        # ----------------- Your Code Here ----------------- #
        # Compute the discretized A_d, B_d, C_d, D_d, for the computation of LQR gain 
        
        Ap = np.zeros((12,12))
        id6 = np.eye(6)
        Ap[0:6,6:] = id6
        Ap[6,4]= self.g
        Ap[7,3] = -self.g
        Bp = np.zeros((12,4))
        Bp[8,0] = 1/self.m
        Bp[9,1] = 1/self.Ix
        Bp[10,2] = 1/self.Iy
        Bp[11,3] = 1/self.Iz
        Cp = np.zeros((4,12))
        Cp[:3,0:3] = np.eye(3)
        Cp[3,5] = 1
        zeros_12_4 = np.zeros((12,4))
        zeros_4_4 = np.zeros((4,4))
        A1 = np.hstack((Ap,zeros_12_4))
        A2 = np.hstack((Cp,zeros_4_4))
        At = np.vstack((A1,A2))
        Bt = np.vstack((Bp,zeros_4_4))
        Bc = np.vstack((zeros_12_4,-np.eye(4)))
        BtBc = np.hstack((Bt,Bc))
        Ct = np.hstack((Cp,zeros_4_4))
        #Dt = np.array([[0],[0],[0],[0]])
        Dt = np.zeros((4,8))
        print(At)
        print(BtBc)
        print(Ct)
        print(Dt)
        delT = 0.01
        CT_sys = signal.StateSpace(At,BtBc,Ct,Dt)
        DT_sys = CT_sys.to_discrete(self.delT)
        A_DT = DT_sys.A
        B_DT = DT_sys.B
        A_d = A_DT
        B_td = B_DT[:,:4]
        B_cd = B_DT[:,4:]
        B_d = B_td
        print(A_d)
        print(B_d)
        # self.m = 0.4
        # self.d1x = 0.1122
        # self.d1y = 0.1515
        # self.d2x = 0.11709
        # self.d2y = 0.128
        # self.Ix = 0.000913855
        # self.Iy = 0.00236242
        # self.Iz = 0.00279965

        #define constants
        # self.g = 9.81
        # self.ct = 0.00026
        # self.ctau = 5.2e-06
        # self.U1_max = 10
        # self.pi = 3.1415926535
        #print(haha)
        #print(self.m)

        # ----------------- Your Code Ends Here ----------------- #

        
        
        
        

        # -----------------    Example code     ----------------- #
        max_pos = 15.0
        max_ang = 0.2 * self.pi
        max_vel = 6.0
        max_rate = 0.015 * self.pi
        max_eyI = 3

        max_states = np.array([0.1 * max_pos, 0.1 * max_pos, max_pos,max_ang, max_ang, max_ang,0.5 * max_vel, 0.5 * max_vel, max_vel,max_rate, max_rate, max_rate, 0.1 * max_eyI, 0.1 * max_eyI, 1 * max_eyI, 0.1 * max_eyI])

        max_inputs = np.array([0.2 * self.U1_max, self.U1_max, self.U1_max, self.U1_max])

        Q = np.diag(1/max_states**2)
        R = np.diag(1/max_inputs**2)
        print("Q:",Q)
        print("R:",R)
        # Q = np.eye(16)
        # R = np.eye(4)
        # -----------------  Example code Ends ----------------- #
        # ----------------- Your Code Here ----------------- #
        # Come up with reasonable values for Q and R (state and control weights)
        # The example code above is a good starting point, feel free to use them or write you own.
        # Tune them to get the better performance
        
        
        
        # ----------------- Your Code Ends Here ----------------- #

        # solve for LQR gains   
        [K, _, _] = dlqr(A_d, B_d, Q, R)

        self.K = -K

    def update(self, r):
        """ Get current states and calculate desired control input.

        Args:
            r (np.array): reference trajectory.

        Returns:
            np.array: states. information of the 16 states.
            np.array: U. desired control input.

        """

        # Fetch the states from the BaseController method
        x_t = super().getStates()

        # update integral term
        self.int_e1 += float((x_t[0]-r[0])*(self.timestep*1e-3))
        self.int_e2 += float((x_t[1]-r[1])*(self.timestep*1e-3))
        self.int_e3 += float((x_t[2]-r[2])*(self.timestep*1e-3))
        self.int_e4 += float((x_t[5]-r[3])*(self.timestep*1e-3))

        # Assemble error-based states into array
        error_state = np.array([self.int_e1, self.int_e2, self.int_e3, self.int_e4]).reshape((-1,1))
        states = np.concatenate((x_t, error_state))

        # calculate control input
        U = np.matmul(self.K, states)
        U[0] += self.g * self.m

        # Return all states and calculated control inputs U
        return states, U