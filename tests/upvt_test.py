import unittest
import neftpy.upvt as pvt
import numpy as np

#from ..neftpy.upvt import *

class PVTTestCase(unittest.TestCase):

    """
    Расчет давления насыщения по Стендингу
    """
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
    
    def test_array_2_unf_pb_Standing_MPaa(self):
        """ compares array input for vectorised realisations """
        rsb_m3m3 = np.array([1.0, 5.0, 10.0, 100.0, 200.0]) 
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        res = pvt.unf_pb_Standing_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K)  # naive vectorisation
        self.assertTrue(np.allclose(pvt.unf_pb_Standing_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), res))

    """
    Расчет давления насыщения по Валко МакКейну
    """
    def test_unf_pb_Valko_MPaa(self):
        rsb_m3m3 = 100
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvt.unf_pb_Valko_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), 23.29991481380937,
                               delta=0.0001)

    def test_array_unf_pb_Valko_MPaa(self):
        """ compares array input for vectorised realisations """
        rsb_m3m3 = np.array([1.0, 5.0, 10.0, 100.0, 200.0]) 
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        res = pvt._unf_pb_Valko_MPaa_(rsb_m3m3, gamma_oil, gamma_gas, t_K)  # naive vectorisation
        self.assertTrue(np.allclose(pvt.unf_pb_Valko_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), res))

    def test_unf_rs_Standing_m3m3(self):
        p_MPaa = 10
        pb_MPaa = 15
        rsb = 200
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvt.unf_rs_Standing_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K),
                               122.74847910146916, delta=0.0001)

    def test_array_unf_rs_Standing_m3m3(self):
        p_MPaa = np.array([1.0, 5.0, 10.0, 100.0, 200.0]) 
        pb_MPaa = 15
        rsb = np.array([1.0, 5.0, 10.0, 100.0, 200.0]) 
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertTrue(np.allclose(pvt.unf_rs_Standing_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K),
                                    pvt._unf_rs_Standing_m3m3_(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K)
                                    ) 
                        )

    def test_unf_rs_Velarde_m3m3(self):
        p_MPaa = 10
        pb_MPaa = 15
        rsb = 250
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvt.unf_rs_Velarde_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K),
                               170.25302712587356, delta=0.0001)

    def test_array_unf_rs_Velarde_m3m3(self):
        p_MPaa = np.array([1.0, 5.0, 10.0, 100.0, 200.0]) 
        pb_MPaa = 15
        rsb = np.array([1.0, 5.0, 10.0, 100.0, 200.0]) 
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertTrue(np.allclose(pvt.unf_rs_Velarde_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K),
                                    pvt._unf_rs_Velarde_m3m3_(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K)
                                    ) 
                        )
# Executing the tests in the above test case class
if __name__ == "__main__":
  unittest.main()