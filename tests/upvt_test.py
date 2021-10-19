import unittest
import neftpy.upvt as pvt
import numpy as np

#from ..neftpy.upvt import *

class PVTTestCase(unittest.TestCase):
    def test_unf_pb_Standing_MPaa(self):
        rsb_m3m3 = 100
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvt.unf_pb_Standing_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), 20.170210695316566,
                               delta=0.0001)

    def test_array_unf_pb_Standing_MPaa(self):
        """ check with np.array as input """
        rsb_m3m3 = np.array([1.0, 5.0, 10.0, 100.0, 200.0]) 
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        res = np.array([ 0.44433583,  1.67824568,  2.98339278, 20.1702107 , 35.85628831])
        self.assertTrue(np.allclose(pvt.unf_pb_Standing_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), res))
        

# Executing the tests in the above test case class
if __name__ == "__main__":
  unittest.main()