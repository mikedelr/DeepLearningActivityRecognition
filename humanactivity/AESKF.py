import numpy as np

from math import floor
from math import sqrt
from math import pi


class AESKF:

    # definition of class members

    ctr  = 0 # idx corresponding to the number of samples processed
    ctr2 = 0
    ctrM = 0 # idx corresponding to the number of samples processed
    sigma_gyr = float('nan')
    delta_b_xy = float('nan')

    ACC_VAR_THRESH = 10 # WIN = 1 * fs
    WIN_SECS = 0.25

    A_VAR_WIN       = floor(100 * 0.25) # NUM_SAMPLES
    A_VAR_WIN_N_INV = (-1 / floor(100 * float(0.25) ))
    VAR_A_MAG_SQ    = 1 / (floor(100 * float(0.25)) - 1)
    a_mag_var       = 0
    BufAmagVar      = np.empty(0)
    rSum_AmagSq     = 0
    rSumSq_AmagSq   = 0
    vCtr            = 0

    BufGamM       = np.empty(0)
    N_VAR_GAM_M   = float('nan')
    rSumGamM      = 0
    rSumSqGamM    = 0
    rSumN2GamM    = 0
    AvgConstGamM  = float('nan')
    nAvgConstGamM = float('nan')
    VarConstGamM  = float('nan')

    BufGamA    = np.empty(0)
    rSumGamA   = 0
    rSumSqGamA = 0

    N_VAR_GAM_A   = float('nan')
    AvgConstGamA  = float('nan')
    nAvgConstGamA = float('nan')
    VarConstGamA  = float('nan')

    BufGamA2    = np.empty(0)
    rSumGamA2   = 0
    rSumSqGamA2 = 0

    N_VAR_GAM_A2   = float('nan')
    AvgConstGamA2  = float('nan')
    nAvgConstGamA2 = float('nan')
    VarConstGamA2  = float('nan')

    gRef    = 9.796720 # magnitude of acceleration due to gravity(Sydney) 
    cA      = 0.1 # normalised cutoff frequency
    ext_acc = np.zeros(3)

    mRef       = 57.0732 # magnitude of the local geomagnetic field(Sydney)
    cM         = 0.99 # normalised cutoff - frequency
    mag_dis    = np.zeros(3)
    mag_now    = np.zeros(3)
    mag_pre    = np.zeros(3)
    mag_now_xy = np.zeros(2)
    mag_pre_xy = np.zeros(2)
    gGyr       = np.zeros(3)
    gacc       = np.zeros(3)
    gAcc_now   = np.zeros(3)
    gAcc_pre   = np.zeros(3)
    gmag       = np.zeros(3)

    deltaMagBxy = float('nan')

    fs = 100 # additional constants defined for speed
    dt = 0.01
    qAngVelDev  = np.array([1, 0, 0 ,0],dtype=np.float64)
    qGlobal     = np.array([1, 0, 0 ,0],dtype=np.float64)
    qGlobalPrev = np.array([1, 0, 0 ,0],dtype=np.float64)
    muAcc       = 0.005
    muMag       = 0.005

    # these are for storing the angles gamma_a and gamma_m
    gammaA = 0
    gammaM = 0
    sin_mu_a_gamma_a = 0
    cos_mu_a_gamma_a = 0
    ## New variables for Adaptive gain via indirect kalman filter
    # --- Process noise covariance matrix, in this case 1D - -- #
    # Q = 2 * dt ^ {2} * \sigma_{gyr} ^ {2} i.e., the variance in the 
    # gyro when stationary, user should specify the noise level
    # of the gyro for example, sigma = 0.015
    Q       = float('nan')
    R_GamA  = float('nan') # variance in the measurement noise
    AvgGamA = float('nan')
    P_pri   = float('nan') # A Priori error state covariance matrix
    P_pos   = float('nan') # A posteriori error state covariance matrix, can be initialised to Q
    muA     = 0.005 # adaptive kalman gain

    Qmag     = float('nan')
    R_GamM   = float('nan') # variance in the measurement noise
    AvgGamM  = float('nan')
    Pmag_pri = float('nan') # initialise a posteriori as Q
    Pmag_pos = float('nan') # initialise a priori as Q
    muM      = 0.005

# definition of class functions
    def __init__(self, q0):
        # print('constructor')
        self.qGlobal[0] = q0[0]
        self.qGlobal[1] = q0[1]
        self.qGlobal[2] = q0[2]
        self.qGlobal[3] = q0[3]

    def setMuAcc(self, muAcc):
        self.muA = muAcc

    def setMuMag(self, muMag):
        self.muM = muMag

    def updateAttitude(self, acc, gyr, mag):
        muA2 = self.muAcc / 2
        muM2 = self.muMag / 2
        qG1 = self.qGlobal[0]
        qG2 = self.qGlobal[1]
        qG3 = self.qGlobal[2]
        qG4 = self.qGlobal[3]

        # --- Correct for gyros - get quaternion from gyro in device frame and rotate
        if not(np.isnan(np.any(gyr))):
            dt2 = 0.5 * self.dt
            qW1 = 1
            qW2 = gyr[0] * dt2
            qW3 = gyr[1] * dt2
            qW4 = gyr[2] * dt2
            qWnorm = 1 / sqrt(qW1 * qW1 + qW2 * qW2 + qW3 * qW3 + qW4 * qW4) # can be replaced by the fast inverse square root
            qW1 = qW1 * qWnorm
            qW2 = qW2 * qWnorm
            qW3 = qW3 * qWnorm
            qW4 = qW4 * qWnorm
            self.qAngVelDev = np.array([qW1, qW2, qW3, qW4], dtype=np.float64)
            # Convert to back to global frame
            qGi1 = qG1 * qW1 - qG2 * qW2 - qG3 * qW3 - qG4 * qW4
            qGi2 = qG1 * qW2 + qG2 * qW1 + qG3 * qW4 - qG4 * qW3
            qGi3 = qG1 * qW3 - qG2 * qW4 + qG3 * qW1 + qG4 * qW2
            qGi4 = qG1 * qW4 + qG2 * qW3 - qG3 * qW2 + qG4 * qW1
        else:
            qGi1 = qG1
            qGi2 = qG2
            qGi3 = qG3
            qGi4 = qG4

        # --- Correct for accelerometer - get next rotation required to align accelerometer / gravity with 'up'.
        if not(np.isnan(np.any(acc))):
            # intermediate calculation in calculating v' = qvq*
            qa1 = -qGi2 * acc[0] - qGi3 * acc[1] - qGi4 * acc[2]
            qa2 = qGi1 * acc[0] + qGi3 * acc[2] - qGi4 * acc[1]
            qa3 = qGi1 * acc[1] - qGi2 * acc[2] + qGi4 * acc[0]
            qa4 = qGi1 * acc[2] + qGi2 * acc[1] - qGi3 * acc[0]
            # Convert acceleration to global frame
            self.gacc[0] = -qa1 * qGi2 + qa2 * qGi1 - qa3 * qGi4 + qa4 * qGi3
            self.gacc[1] = -qa1 * qGi3 + qa2 * qGi4 + qa3 * qGi1 - qa4 * qGi2
            self.gacc[2] = -qa1 * qGi4 - qa2 * qGi3 + qa3 * qGi2 + qa4 * qGi1
            # Get fraction of rotation from acc in global to up
            if muA2 < 0:
                muA2 = 0  #warning('Capping mu at 0')
            elif muA2 > 0.5:
                muA2 = 0.5 #warning('Capping mu at 0.5')

            sq_g_acc_y = self.gacc[1] * self.gacc[1]
            sq_g_acc_x = self.gacc[0] * self.gacc[0]
            sq_g_acc_x_g_acc_y = sq_g_acc_x + sq_g_acc_y
            axisVecNorm = 1 / sqrt(sq_g_acc_x_g_acc_y) # can be replaced by the fast inverse square root

            if axisVecNorm == 0:
                qUp1 = 1
                qUp2 = 0
                qUp3 = 0
                qUp4 = 0
            else:
                gamma_a = AESKF.arctan2_approx(axisVecNorm * sq_g_acc_x_g_acc_y, self.gacc[2])
                self.gammaA = gamma_a
                mu_a_gamma_a = muA2 * gamma_a

                self.cos_mu_a_gamma_a = 1 # small angle approximation
                self.sin_mu_a_gamma_a = mu_a_gamma_a # small angle approximation

                qUp1 = self.cos_mu_a_gamma_a
                qUp2 = self.gacc[1] * axisVecNorm * self.sin_mu_a_gamma_a
                qUp3 = -self.gacc[0] * axisVecNorm * self.sin_mu_a_gamma_a
                qUpNorm = 1 / sqrt((qUp1 * qUp1 + qUp2 * qUp2 + qUp3 * qUp3)) # can be replaced by the fast inverse square root
                # normalise to unit quaternions
                qUp1 = qUp1 * qUpNorm
                qUp2 = qUp2 * qUpNorm
                qUp3 = qUp3 * qUpNorm
                qUp4 = 0

            # Rotate global frame towards 'up'
                qGii1 = qUp1 * qGi1 - qUp2 * qGi2 - qUp3 * qGi3 - qUp4 * qGi4
                qGii2 = qUp1 * qGi2 + qUp2 * qGi1 + qUp3 * qGi4 - qUp4 * qGi3
                qGii3 = qUp1 * qGi3 - qUp2 * qGi4 + qUp3 * qGi1 + qUp4 * qGi2
                qGii4 = qUp1 * qGi4 + qUp2 * qGi3 - qUp3 * qGi2 + qUp4 * qGi1
        else:
            qGii1 = qGi1
            qGii2 = qGi2
            qGii3 = qGi3
            qGii4 = qGi4
        # --- Correct for magnetometer - get next rotation around vertical to align measured global xy
        # Transform magnetometer reading into global frame
        if not(np.isnan(np.any(mag))):
            qm1 = -qGii2 * mag[0] - qGii3 * mag[1] - qGii4 * mag[2]
            qm2 =  qGii1 * mag[0] + qGii3 * mag[2] - qGii4 * mag[1]
            qm3 =  qGii1 * mag[1] - qGii2 * mag[2] + qGii4 * mag[0]
            qm4 =  qGii1 * mag[2] + qGii2 * mag[1] - qGii3 * mag[0]
            self.gmag[0] = -qm1 * qGii2 + qm2 * qGii1 - qm3 * qGii4 + qm4 * qGii3
            self.gmag[1] = -qm1 * qGii3 + qm2 * qGii4 + qm3 * qGii1 - qm4 * qGii2
            qn2 = 0
            qn3 = 0
            if self.gmag[1] == 0: # y - component is 0 therefore pointing north
                qn1 = 1
                qn4 = 0
            else:
                gamma_m = AESKF.arctan2_approx(abs(self.gmag[1]), self.gmag[0])  # absolute value unsure if always correct
                self.gammaM = gamma_m
                mu_m_gamma_m = muM2 * gamma_m

                cos_mu_m_gamma_m = 1
                sin_mu_m_gamma_m = mu_m_gamma_m

                self.cos_muM_gammaM = cos_mu_m_gamma_m
                self.sin_muM_gammaM = sin_mu_m_gamma_m

                qn1 = cos_mu_m_gamma_m
                qn4 = np.sign(-self.gmag[1]) * sin_mu_m_gamma_m
                qNorm = 1 / sqrt(qn1 * qn1 + qn4 * qn4) # can be replaced by the fast inverse square root
                qn1 = qn1 * qNorm
                qn4 = qn4 * qNorm # normalise to unit quaternions end
                # Rotate global frame towards 'north'
                qGiii1 = qn1 * qGii1 - qn2 * qGii2 - qn3 * qGii3 - qn4 * qGii4
                qGiii2 = qn1 * qGii2 + qn2 * qGii1 + qn3 * qGii4 - qn4 * qGii3
                qGiii3 = qn1 * qGii3 - qn2 * qGii4 + qn3 * qGii1 + qn4 * qGii2
                qGiii4 = qn1 * qGii4 + qn2 * qGii3 - qn3 * qGii2 + qn4 * qGii1
        else:
            qGiii1 = qGii1
            qGiii2 = qGii2
            qGiii3 = qGii3
            qGiii4 = qGii4

        self.qGlobal[0] = qGiii1
        self.qGlobal[1] = qGiii2
        self.qGlobal[2] = qGiii3
        self.qGlobal[3] = qGiii4

    def getAttitude(self):
        return self.qGlobal

    def toString(self):
        return '[' + str(self.qGlobal[0]) + ', ' + str(self.qGlobal[1]) + ', ' + str(self.qGlobal[2]) + ', ' + str(self.qGlobal[3]) + ']'

    @staticmethod
    # multiply(x4), addition(x1), subtraction(x2)
    def arctan_approx(x):
        qtr_pi = pi / 4
        return (qtr_pi*x) - x * (abs(x) - 1) * (0.2447 + 0.0663 * abs(x))

    @staticmethod
    def arctan2_approx(y,x):
        # only works when y is positive i.e.the numerator in atan2(y, x)
        if x >= 0:
            if x >= y:
                invsqrtxsq = np.sign(x) * (1 / sqrt(x * x))  # can be replaced by the fast inverse square root
                # atan2x = arctan_approx(y / x);
                atan2x = AESKF.arctan_approx(y * invsqrtxsq)
            else: # x < y
                invsqrtysq = np.sign(y) * (1 / sqrt(y * y))  # can be replaced by the fast inverse square root
                # atan2x = pi / 2 - arctan_approx(x / y)
                atan2x = pi / 2 - AESKF.arctan_approx(x * invsqrtysq)

        else: # x <= 0
            if y > abs(x):
                # atan2x = pi / 2 + arctan_approx(abs(x) / y)
                invsqrtysq = np.sign(y) * (1 / sqrt(y * y))  # can be replaced by the fast inverse square root
                atan2x = pi / 2 + AESKF.arctan_approx(abs(x) * invsqrtysq)
            else:
                # atan2x = pi - arctan_approx(y / abs(x))
                invsqrtxsq = 1 / sqrt(x * x)  # can be replaced by the fast inverse square root
                atan2x = pi - AESKF.arctan_approx(y * invsqrtxsq)

        return atan2x
