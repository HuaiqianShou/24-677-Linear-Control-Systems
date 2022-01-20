import numpy as np
import math

class EKF_SLAM():
    def __init__(self, init_mu, init_P, dt, W, V, n):
        """Initialize EKF SLAM

        Create and initialize an EKF SLAM to estimate the robot's pose and
        the location of map features

        Args:
            init_mu: A numpy array of size (3+2*n, ). Initial guess of the mean 
            of state. 
            init_P: A numpy array of size (3+2*n, 3+2*n). Initial guess of 
            the covariance of state.
            dt: A double. The time step.
            W: A numpy array of size (3+2*n, 3+2*n). Process noise
            V: A numpy array of size (2*n, 2*n). Observation noise
            n: A int. Number of map features
            

        Returns:
            An EKF SLAM object.
        """
        self.mu = init_mu  # initial guess of state mean
        self.P = init_P  # initial guess of state covariance
        self.dt = dt  # time step
        self.W = W  # process noise 
        self.V = V  # observation noise
        self.n = n  # number of map features


    def _f(self, x, u):
        """Non-linear dynamic function.

        Compute the state at next time step according to the nonlinear dynamics f.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            x_next: A numpy array of size (3+2*n, ). The state at next time step
        """
        
        
        # update the first three items in the vector and the rest are just same constant
        
        xdot = u[0]
        ydot = u[1]
        psidot = u[2]
        x_next = np.zeros(3+2*self.n)
        xt = x[0]
        yt = x[1]
        psit = x[2]
        x_next[0] = xt+self.dt*(xdot*np.cos(psit)-ydot*np.sin(psit))
        x_next[1] = yt+self.dt*(xdot*np.sin(psit)+ydot*np.cos(psit))
        x_next[2] = psit+self.dt*psidot
        x_next[3:] = x[3:]
        
        

        return x_next


    def _h(self, x):
        """Non-linear measurement function.

        Compute the sensor measurement according to the nonlinear function h.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.

        Returns:
            y: A numpy array of size (2*n, ). The sensor measurement.
        """
        
        
        #
        n = self.n
        y = np.zeros(2*n)
        #get all the constant from x size(2n,)
        x_normlize = x[3:]
        
        
        xt = x[0]
        yt = x[1]
        psit = x[2] 
        #get all the mx and my from x both size (n,)
        mx = x_normlize[::2]
        my = x_normlize[1::2]
        y[0:n] = np.sqrt(np.power((mx-xt),2)+np.power((my-yt),2))
        #y[n:2*n] = math.atan2((my-yt),(mx-xt)) - psit
        y[n:2*n] = np.arctan2((my-yt),(mx-xt)) - psit
        
        return y


    def _compute_F(self, u):
        """Compute Jacobian of f
        
        You will use self.mu in this function.

        Args:
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            F: A numpy array of size (3+2*n, 3+2*n). The jacobian of f evaluated at x_k.
            
            
        """
        
        n = self.n
        
        xdot = u[0]
        ydot = u[1]
        psidot = u[2]
        
        psit = self.mu[2]
        
        At = np.zeros((3,3))
        At[0][0] = 1
        At[1][1] = 1
        At[2][2] = 1
        At[0][2] = self.dt*(-xdot*np.sin(psit)-ydot*np.cos(psit))
        At[1][2] = self.dt*(xdot*np.cos(psit)-ydot*np.sin(psit))
        O1 = np.zeros((3,2*n))
        O2 = np.zeros((2*n,3))
        I1 = np.eye(2*n)
        con1 = np.concatenate((At,O1),axis = 1)
        con2 = np.concatenate((O2,I1),axis = 1)
        F = np.concatenate((con1,con2),axis = 0)
        
        return F


    def _compute_H(self):
        """Compute Jacobian of h
        
        You will use self.mu in this function.

        Args:

        Returns:
            H: A numpy array of size (2*n, 3+2*n). The jacobian of h evaluated at x_k.
        """

        # distance sensor

        # bearing sensor
        n = self.n
        
        mu_normlize = self.mu[3:]
        
        
        xt = self.mu[0]
        yt = self.mu[1]
        psit = self.mu[2] 
        #get all the mx and my from x both size (n,)
        mx = mu_normlize[::2]
        my = mu_normlize[1::2]
        d1 = np.sqrt(np.power((mx-xt),2)+np.power((my-yt),2))
        d2 = np.power((mx-xt),2)+np.power((my-yt),2)
        h1x = (-mx+xt)/d1
        h1y = (-my+yt)/d1
        h2x = (my-yt)/d2
        h2y = (-mx+xt)/d2
        
 
        H = np.zeros((2*n,3+2*n))
        
        H[0:n,0] = h1x
        H[0:n,1] = h1y
        H[n:2*n,0] = h2x
        H[n:2*n,1] = h1y
        H[n:2*n,2] = -1
        
        
        
        for i in range(n):

            H[i,2*i+3] = -h1x[i]
            H[i,2*i+4] = -h1y[i]
            
            H[i+n,2*i+3] = -h2x[i]
            H[i+n,2*i+4] = -h2y[i]

        return H


    def predict_and_correct(self, y, u):
        """Predice and correct step of EKF
        
        You will use self.mu in this function. You must update self.mu in this function.

        Args:
            y: A numpy array of size (2*n, ). The measurements according to the project description.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            self.mu: A numpy array of size (3+2*n, ). The corrected state estimation
            self.P: A numpy array of size (3+2*n, 3+2*n). The corrected state covariance
        """

        # compute F and H matrix

        last_mu = self.mu
        
        n = self.n
        F =self._compute_F(u)
        H =self._compute_H()
        
        last_p = self.P
        #***************** Predict step *****************#
        # predict the state
        
        xhat = self._f(last_mu, u)
        # predict the error covariance
        P = F@last_p@(F.T)+self.W 
        #***************** Correct step *****************#
        # compute the Kalman gain
        
        L = P@(H.T)@np.linalg.inv((H@P@H.T+self.V))
        # update estimation with new measurement
        yhat = self._h(xhat)
        self.mu = xhat+L@self._wrap_to_pi((y-yhat))
        # update the error covariance
        I_out = np.eye(3+2*n)
        self.P = (I_out-L@H)@P
        return self.mu, self.P


    def _wrap_to_pi(self, angle):
        angle = angle - 2*np.pi*np.floor((angle+np.pi )/(2*np.pi))
        return angle


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    m = np.array([[0.,  0.],
                  [0.,  20.],
                  [20., 0.],
                  [20., 20.],
                  [0,  -20],
                  [-20, 0],
                  [-20, -20],
                  [-50, -50]]).reshape(-1)

    dt = 0.01
    T = np.arange(0, 20, dt)
    n = int(len(m)/2)
    W = np.zeros((3+2*n, 3+2*n))
    W[0:3, 0:3] = dt**2 * 1 * np.eye(3)
    V = 0.1*np.eye(2*n)
    V[n:,n:] = 0.01*np.eye(n)

    # EKF estimation
    mu_ekf = np.zeros((3+2*n, len(T)))
    mu_ekf[0:3,0] = np.array([2.2, 1.8, 0.])
    # mu_ekf[3:,0] = m + 0.1
    mu_ekf[3:,0] = m + np.random.multivariate_normal(np.zeros(2*n), 0.5*np.eye(2*n))
    init_P = 1*np.eye(3+2*n)

    # initialize EKF SLAM
    slam = EKF_SLAM(mu_ekf[:,0], init_P, dt, W, V, n)
    
    # real state
    mu = np.zeros((3+2*n, len(T)))
    mu[0:3,0] = np.array([2, 2, 0.])
    mu[3:,0] = m

    y_hist = np.zeros((2*n, len(T)))
    for i, t in enumerate(T):
        if i > 0:
            # real dynamics
            u = [-5, 2*np.sin(t*0.5), 1*np.sin(t*3)]
            # u = [0.5, 0.5*np.sin(t*0.5), 0]
            # u = [0.5, 0.5, 0]
            mu[:,i] = slam._f(mu[:,i-1], u) + \
                np.random.multivariate_normal(np.zeros(3+2*n), W)

            # measurements
            y = slam._h(mu[:,i]) + np.random.multivariate_normal(np.zeros(2*n), V)
            y_hist[:,i] = (y-slam._h(slam.mu))
            # apply EKF SLAM
            mu_est, _ = slam.predict_and_correct(y, u)
            mu_ekf[:,i] = mu_est


    plt.figure(1, figsize=(10,6))
    ax1 = plt.subplot(121, aspect='equal')
    ax1.plot(mu[0,:], mu[1,:], 'b')
    ax1.plot(mu_ekf[0,:], mu_ekf[1,:], 'r--')
    mf = m.reshape((-1,2))
    ax1.scatter(mf[:,0], mf[:,1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2 = plt.subplot(322)
    ax2.plot(T, mu[0,:], 'b')
    ax2.plot(T, mu_ekf[0,:], 'r--')
    ax2.set_xlabel('t')
    ax2.set_ylabel('X')

    ax3 = plt.subplot(324)
    ax3.plot(T, mu[1,:], 'b')
    ax3.plot(T, mu_ekf[1,:], 'r--')
    ax3.set_xlabel('t')
    ax3.set_ylabel('Y')

    ax4 = plt.subplot(326)
    ax4.plot(T, mu[2,:], 'b')
    ax4.plot(T, mu_ekf[2,:], 'r--')
    ax4.set_xlabel('t')
    ax4.set_ylabel('psi')

    plt.figure(2)
    ax1 = plt.subplot(211)
    ax1.plot(T, y_hist[0:n, :].T)
    ax2 = plt.subplot(212)
    ax2.plot(T, y_hist[n:, :].T)

    plt.show()
