import os

import numpy as np
import pandas as pd

from unittest import TestCase
from humanactivity import quaternion


class TestQuaternion(TestCase):

    def test_normalize(self):
        q = np.array([1, 2, 3, 4], dtype=np.float64)
        q1 = quaternion.normalize(q)

        q_norm = np.sqrt(np.sum(np.square(q)))

        self.assertEqual(q1[0], q[0]/q_norm)
        self.assertEqual(q1[1], q[1]/q_norm)
        self.assertEqual(q1[2], q[2]/q_norm)
        self.assertEqual(q1[3], q[3]/q_norm)

    def test_conjugate(self):
        q = np.array([1, 2, 3, 4], dtype=np.float64)
        q1 = quaternion.conjugate(q)

        self.assertEqual(q1[0], q[0])
        self.assertEqual(q1[1], -q[1])
        self.assertEqual(q1[2], -q[2])
        self.assertEqual(q1[3], -q[3])

    def test_quat_to_angular_velocity(self):
        # from humanactivity.quaternion import normalize, inverse, conjugate, quatmultiply, quat_to_angular_velocity
        dirpath = os.getcwd()
        quat_filename = '\\quaternions.csv'
        quat = pd.read_csv(dirpath + quat_filename, header=None, sep=',')
        omega_v = quaternion.quat_to_angular_velocity(quat, 0.01)


