# functions for manipulation of quaternions
# convention for a quaternion q = [scalar, i , j ,k]
import numpy as np


def to_nautical_angles(q):
    """
        euler angles from quaternion

        calculate

        Parameters
        ----------
        q : np.array()
            quaternion or array of quaternions, q.shape = [M x 4]

        Returns
        -------
        nautical_angles : np.array()
            euler angles or array of euler angles following the nautical angle convention, nautical_angles.shape = [M x 3]
    """
    # calculate the euler angles in radians from a 2D array of quaternions whose shape is M x 4
    nautical_angles = np.zeros([len(q), 3], dtype=np.float64)

    r23 = 2 * (q[:,2] * q[:, 3] + q[:, 0] * q[:, 1])
    r33 = q[:, 3] * q[:, 3] - q[:, 2] * q[:, 2] - q[:, 1] * q[:, 1] + q[:, 0] * q[:, 0]
    phi_roll = np.arctan2(r23, r33)
    nautical_angles[:, 0] = phi_roll

    r13 = 2 * (q[:, 1] * q[:, 3] - q[:, 0]*q[:, 2])
    theta_pitch = -np.arcsin(r13)
    nautical_angles[:, 1] = theta_pitch

    r12 = 2 * (q[:, 1] * q[:, 2] + q[:, 0]*q[:, 3])
    r11 = q[:, 1] * q[:, 1] + q[:, 0] * q[:, 0] - q[:, 3] * q[:, 3] - q[:, 2] * q[:, 2]
    psi_yaw = np.arctan2(r12, r11)
    nautical_angles[:, 2] = psi_yaw

    return nautical_angles


# define quaternion multiplication
def quatmultiply(q_a, q_b):
    """
        quaternion multiplication

        multiplies two series of quaternions

        Parameters
        ----------
        q_a : np.array, shape = [M x 4]
            quaternion or array of quaternions
        q_b : np.array, shape = [M x 4]
            quaternion or array of quaternions

        Returns
        -------
        q_ab : np.array, shape = [M x 4]
            quaternion or array of quaternions
    """
    dim_q_a = q_a.shape
    dim_q_b = q_b.shape
    q_ab = np.zeros(dim_q_a, dtype=np.float64)
    if q_a.ndim == 1 & q_b.ndim == 1:
        q_ab[0] = q_a[0] * q_b[0] - q_a[1] * q_b[1] - q_a[2] * q_b[2] - q_a[3] * q_b[3]
        q_ab[1] = q_a[0] * q_b[1] + q_a[1] * q_b[0] + q_a[2] * q_b[3] - q_a[3] * q_b[2]
        q_ab[2] = q_a[0] * q_b[2] - q_a[1] * q_b[3] + q_a[2] * q_b[0] + q_a[3] * q_b[1]
        q_ab[3] = q_a[0] * q_b[3] + q_a[1] * q_b[2] - q_a[2] * q_b[1] + q_a[3] * q_b[0]
    # multiplication of two matrices of quaternions
    elif q_a.ndim == 2 & q_b.ndim == 2:
        if (dim_q_a[1] != 4) | (dim_q_b[1] != 4):
            raise ValueError('quatmultiply expects q_a and q_b to be 2D arrays of size M x 4')
        elif dim_q_a[0] == dim_q_b[0]:
            q_ab[:, 0] = q_a[:, 0] * q_b[:, 0] - q_a[:, 1] * q_b[:, 1] - q_a[:, 2] * q_b[:, 2] - q_a[:, 3] * q_b[:, 3]
            q_ab[:, 1] = q_a[:, 0] * q_b[:, 1] + q_a[:, 1] * q_b[:, 0] + q_a[:, 2] * q_b[:, 3] - q_a[:, 3] * q_b[:, 2]
            q_ab[:, 2] = q_a[:, 0] * q_b[:, 2] - q_a[:, 1] * q_b[:, 3] + q_a[:, 2] * q_b[:, 0] + q_a[:, 3] * q_b[:, 1]
            q_ab[:, 3] = q_a[:, 0] * q_b[:, 3] + q_a[:, 1] * q_b[:, 2] - q_a[:, 2] * q_b[:, 1] + q_a[:, 3] * q_b[:, 0]
        elif dim_q_a[0] == 1 & dim_q_b[0] != 1:
            print('quatmultiply: q_a is [1 x 4]')
        elif dim_q_a[0] != 1 & dim_q_b[0] == 1:
            print('quatmultiply: q_b is [1 x 4]')
        else:
            print('quatmultiply: one of q_a and q_b must be [1 x 4]')
    else:
        ValueError('quatmultiply only supports multiplication of two quaternions, or two arrays of quaternions')
    return q_ab


# normalize a single quaternion
def normalize(q):
    """
        quaternion normalization

        normalizes a quaternion or series of quaternions

        Parameters
        ----------
        q : np.array, shape = [M x 4]
            quaternion or array of quaternions

        Returns
        -------
        q_unit : np.array, shape = [M x 4]
            q = [a b c d]
            q = [a b c d]/sqrt(a^2+b^2+c^2+d^2)
    """
    q_unit = np.copy(q)
    if q_unit.ndim == 1:
        q_sq = q_unit*q_unit
        q_norm = np.sqrt(np.sum(q_sq))
        q_unit[0] /= q_norm
        q_unit[1] /= q_norm
        q_unit[2] /= q_norm
        q_unit[3] /= q_norm
    else:
        # normalize a 2D array of quaternions
        dim_q = q_unit.shape
        if dim_q[0] == 4:
            qdim = 0
        elif dim_q[1] == 4:
            qdim = 1
        else:
            raise ValueError('normalize expects q to be a 4 column or 4 row 2D array')
        q_unit = np.apply_along_axis(normalize, qdim, q_unit)

    return q_unit


# conjugate of a single quaternion
def conjugate(q):
    """
        quaternion conjugation

        negates the vector component of a quaternion or array of quaternions i.e., if q = [a b c d]
        conjugate(q) = [a -b -c -d]

        Parameters
        ----------
        q : np.array()
            quaternion or array of quaternions, q.shape = [M x 4]

        Returns
        -------
        q_conj : np.array()
            quaternion or array of quaternions, q.shape = [M x 4]
    """
    q_conj = np.copy(q)
    if q_conj.ndim == 1:
        q_conj[1] *= -1
        q_conj[2] *= -1
        q_conj[3] *= -1
    else:
        # normalize a 2D array of quaternions
        shape_q = q_conj.shape
        if shape_q[0] == 4:
            q_dim = 0
        elif shape_q[1] == 4:
            q_dim = 1
        else:
            raise ValueError('conjugate expects q to be a 4 column or 4 row 2D array')

        q_conj = np.apply_along_axis(conjugate, q_dim, q_conj)

    return q_conj


# inverse of a quaternion
def inverse(q):
    """
        quaternion inverse

        augments the quaternion by normalizing the quaternion and calculating its conjugate

        Parameters
        ----------
        q : np.array()
            quaternion or array of quaternions, q.shape = [M x 4]

        Returns
        -------
        q_inv : np.array()
            quaternion or array of quaternions, q.shape = [M x 4]
    """
    q_inv = np.copy(q)
    if q_inv.ndim == 1:
        q_inv = conjugate(q_inv)
        q_inv = normalize(q_inv)
    # inverse of an array of quaternions
    elif q_inv.ndim == 2:
        shape_q = q_inv.shape
        if shape_q[0] == 4:
            q_dim = 0
        elif shape_q[1] == 4:
            q_dim = 1
        else:
            raise ValueError('inverse expects q to be a 4 column or 4 row 2D array')
        q_inv = np.apply_along_axis(inverse, q_dim, q_inv)
    else:
        raise ValueError('inverse only supports a 1D or 2D array')

    return q_inv


#
def to_rotation_matrix(q):
    """
        rotation matrix from quaternion

        calculates the rotation matrix or rotation matrices from a quaternion or array
        of quaternions

        Parameters
        ----------
        q : np.array()
            quaternion or array of quaternions [M x 4]

        Returns
        -------
        rot_mat : np.array()
            3 x 3 rotation matrix or array of 3 x 3 rotation matrices i.e.,
            q.shape = [3 x 3 x M]
    """
    normalize(q)
    if q.ndim == 1:
        rot_mat = np.zeros([3, 3], dtype=np.float64)
        # row 1
        rot_mat[0, 0] = q[0] * 2 + q[1] * 2 - q[2] * 2 - q[3] * 2
        rot_mat[0, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
        rot_mat[0, 2] = 2 * (q[1] * q[3] - q[0] * q[2])
        # row 2
        rot_mat[1, 0] = 2 * (q[1] * q[2] - q[0] * q[3])
        rot_mat[1, 1] = q[0] * 2 - q[1] * 2 + q[2] * 2 - q[3] * 2
        rot_mat[1, 2] = 2 * (q[2] * q[3] + q[0] * q[1])
        # row 3
        rot_mat[2, 0] = 2 * (q[1] * q[3] + q[0] * q[2])
        rot_mat[2, 1] = 2 * (q[2] * q[3] - q[0] * q[1])
        rot_mat[2, 2] = q[0] * 2 - q[1] * 2 - q[2] * 2 + q[3] * 2
    elif q.ndim == 2:
        shape_q = q.shape
        if shape_q[0] == 4:
            # ensure the quaternions are in the rows of the 2D array
            q.transpose()
            r_dim = shape_q[0]
        elif shape_q[1] == 4:
            r_dim = shape_q[0]
        else:
            r_dim = 0
            ValueError('toRotationMatrix')
        rot_mat = np.zeros([3, 3, r_dim], dtype=np.float64)
        # row 1
        rot_mat[0, 0, :] = q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3]
        rot_mat[0, 1, :] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        rot_mat[0, 2, :] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        # row 2
        rot_mat[1, 0, :] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        rot_mat[1, 1, :] = q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] - q[:, 3] * q[:, 3]
        rot_mat[1, 2, :] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
        # row 3
        rot_mat[2, 0, :] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
        rot_mat[2, 1, :] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
        rot_mat[2, 2, :] = q[:, 0] * q[:, 0] - q[:, 1] * q[:, 1] - q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    else:
        raise ValueError('toRotationMatrix only supports a 1D or 2D array')

    return rot_mat


def rotate(q, v):
    """
        rotates a vector or array of vectors by a quaternion or array of quaternions

        Extended description of function.

        Parameters
        ----------
        q : np.array()
            quaternion or array of quaternions, q.shape = [M x 4]
        v : np.array()
            vector or array of vectors, q.shape = [M x 3]

        Returns
        -------
        v_rot : np.array()
            vector or array of vectors expressed in the co-ordinate frame defined by
            the quaternion or array of quaternions, v_rot.shape = [M x 3]
    """
    # normalize quaternions first
    q_unit = normalize(q)
    inv_q_unit = inverse(q)
    v_shape = v.shape
    # find dimension whose length is 3
    v_idx = v_shape.index(3)
    vshape = np.asarray(v_shape)
    # increment dimension by 1 to convert to quaternion domain
    vshape[v_idx] += 1
    q_v = np.zeros(vshape, dtype=np.float64)

    q_dim = q_unit.ndim
    v_dim = v.ndim
    # append zeros to vector to generate a quaternion with zero scalar component
    if q_dim == 1 & v_dim == 1:
        # single rotation
        q_v[1:] = v
        q_v_rotated = quatmultiply(quatmultiply(inv_q_unit, q_v), q_unit)
        v_rotated = q_v_rotated[1:]
    elif q_dim == 2 & v_dim == 2:
        q_v[:, 1:] = v
        q_v_rotated = quatmultiply(quatmultiply(inv_q_unit, q_v), q_unit)
        v_rotated = q_v_rotated[:, 1:]
    else:
        ValueError('rotate only supports 2D arrays of quaternions and vectors')
        v_rotated = np.zeros(v.shape, dtype=np.float64)

    return v_rotated


def quat_to_angular_velocity(q, dt):
    """
        calculates the angular velocity for a matrix of quaternions assuming a
        fixed sampling rate or inter-arrival time

        Extended description of function.

        Parameters
        ----------
        q : np.array()
            quaternion or array of quaternions, q.shape = [M x 4]
        dt : scalar
            vector or array of vectors, q.shape = [M x 3]

        Returns
        -------
        dq : np.array() [M x 3]
            array of vectors expressed in the co-ordinate frame defined
            by the quaternion
    """
    # log of quaternion
    q_shape = q.shape
    q1 = q.values[1::]
    q2 = q.values[0:-1:]
    q1iq2 = quatmultiply(q1, inverse(q2))
    neg = q1iq2[:, 0] < 0
    q1iq2[neg, :] *= -1
    n_dim_diff = q_shape[0]-1
    derivq = np.zeros([n_dim_diff, 4], dtype=np.float64)
    norm_q1iq2 = np.sqrt(np.sum(np.square(q1iq2), 1))
    derivq[:, 0] = np.log(norm_q1iq2)
    norm_vec = np.sqrt(np.sum(np.square(q1iq2[:, 1::]), 1))

    # calculate the normalized rotation vector
    omega_v = np.zeros([n_dim_diff, 3], dtype=np.float64)
    omega_v[:, 0] = np.divide(q1iq2[:, 1], norm_vec)
    omega_v[:, 1] = np.divide(q1iq2[:, 2], norm_vec)
    omega_v[:, 2] = np.divide(q1iq2[:, 3], norm_vec)

    theta = np.arccos(np.divide(q1iq2[:, 0], norm_q1iq2))
    derivq[:, 1] = np.divide(np.multiply(omega_v[:, 0], theta), dt)
    derivq[:, 2] = np.divide(np.multiply(omega_v[:, 1], theta), dt)
    derivq[:, 3] = np.divide(np.multiply(omega_v[:, 2], theta), dt)

    dq = np.zeros(q_shape, dtype=np.float64)
    dq[0, :] = np.multiply(2, derivq[0, :])
    dq[1:(q_shape[0]-1), :] = derivq[0: (n_dim_diff-1), :] + derivq[1: n_dim_diff, :]
    dq[n_dim_diff, :] = np.multiply(2, derivq[(n_dim_diff-1), :])

    return dq[:, 1: 4]

