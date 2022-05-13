import unittest
import neftpy.upvt as pvt
import neftpy.upvt_old as pvt_old
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
        res = pvt_old.unf_pb_Standing_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K)  # naive vectorisation
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
                               delta=0.00001)

    def test_array_unf_pb_Valko_MPaa(self):
        """ compares array input for vectorised realisations """
        rsb_m3m3 = np.array([1.0, 5.0, 10.0, 100.0, 200.0]) 
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        res = pvt_old.unf_pb_Valko_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K)  # naive vectorisation
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
                                    pvt_old.unf_rs_Standing_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K)
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
                                    pvt_old.unf_rs_Velarde_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K)
                                    ) 
                        )
                        
    def test_unf_rsb_Mccain_m3m3(self):
        rsp_m3m3 = 150
        gamma_oil = 0.86
        psp_MPaa = 5
        tsp_K = 320
        self.assertAlmostEqual(pvt.unf_rsb_Mccain_m3m3(rsp_m3m3, gamma_oil, psp_MPaa, tsp_K),
                               161.03286985548442, delta=0.0001)

                        
    def test_array_unf_rsb_Mccain_m3m3(self):
        rsp_m3m3 = np.linspace(1,100,20)
        gamma_oil = 0.86
        psp_MPaa = 5
        tsp_K = 320
        self.assertTrue(np.allclose(pvt_old.unf_rsb_Mccain_m3m3(rsp_m3m3, gamma_oil, psp_MPaa, tsp_K),
                                    pvt.unf_rsb_Mccain_m3m3(rsp_m3m3, gamma_oil, psp_MPaa, tsp_K)
                                    ) 
                        )
        
    def test_unf_bo_above_pb_m3m3(self):
        bob = 1.3
        cofb_1MPa = 3e-3
        pb_MPaa = 12
        p_MPaa = 15
        self.assertAlmostEqual(pvt_old.unf_bo_above_pb_m3m3(bob, cofb_1MPa, pb_MPaa, p_MPaa), 1.2883524924047487, delta=0.0001)

    def test_array_unf_bo_above_pb_m3m3(self):
        bob = 1.3
        cofb_1MPa = 3e-3
        pb_MPaa = 12
        p_MPaa = np.linspace(0.1,10,20)
        self.assertTrue(np.allclose(pvt_old.unf_bo_above_pb_m3m3(bob, cofb_1MPa, pb_MPaa, p_MPaa),
                                    pvt.unf_bo_above_pb_m3m3(bob, cofb_1MPa, pb_MPaa, p_MPaa)
                                    ) 
                        )

    def test_unf_bo_below_m3m3(self):
        density_oilsto_kgm3 = 800
        rs_m3m3 = 200
        density_oil_kgm3 = 820
        gamma_gas = 0.6
        self.assertAlmostEqual(pvt.unf_bo_below_pb_m3m3(density_oilsto_kgm3, rs_m3m3, density_oil_kgm3, gamma_gas),
                               1.1542114715227887, delta=0.0001
                               )

    def test_array_unf_bo_below_m3m3(self):
        density_oilsto_kgm3 = 800
        rs_m3m3 = np.linspace(1,100,20)
        density_oil_kgm3 = 820
        gamma_gas = 0.6
        self.assertTrue(np.allclose(pvt_old.unf_bo_below_pb_m3m3(density_oilsto_kgm3, rs_m3m3, density_oil_kgm3, gamma_gas),
                                    pvt.unf_bo_below_pb_m3m3(density_oilsto_kgm3, rs_m3m3, density_oil_kgm3, gamma_gas)
                                    ) 
                        )


    def test_unf_density_oil_Mccain(self):
        p_MPaa = 10
        pb_MPaa = 12
        co_1MPa = 3 * 10**(-3)
        rs_m3m3 = 250
        gamma_gas = 0.6
        t_K = 350
        gamma_oil = 0.86
        gamma_gassp = 0
        self.assertAlmostEqual(pvt.unf_density_oil_Mccain(p_MPaa, pb_MPaa, co_1MPa, rs_m3m3, gamma_gas, t_K, gamma_oil,
                               gamma_gassp), 630.0536681794456, delta=0.0001)


    def test_array_unf_density_oil_Mccain(self):
        p_MPaa = np.array([0.1, 1,5,10, 20, 30])
        pb_MPaa = 10
        co_1MPa = 3 * 10**(-3)
        rs_m3m3 = 150
        gamma_gas = 0.6
        t_K = 350
        gamma_oil = 0.86
        gamma_gassp = 0
        self.assertTrue(np.allclose(pvt.unf_density_oil_Mccain(p_MPaa, pb_MPaa, co_1MPa, rs_m3m3, gamma_gas, t_K, gamma_oil,
                               gamma_gassp),
                               [684.20874678, 684.97844157, 688.3460601,  692.43351253, 713.52125245, 735.25120966]) )
  
    def test_unf_density_oil_Standing(self):
        p_MPaa = 10
        pb_MPaa = 12
        co_1MPa = 3 * 10**(-3)
        rs_m3m3 = 250
        bo_m3m3 = 1.1
        gamma_gas = 0.6
        gamma_oil = 0.86
        self.assertAlmostEqual(pvt.unf_density_oil_Standing(p_MPaa, pb_MPaa, co_1MPa, rs_m3m3, bo_m3m3, gamma_gas, gamma_oil
                                                        ), 948.7272727272725, delta=0.0001)

    #def test_unf_compressibility_saturated_oil_VB_1Mpa(self):
    #    rsb_m3m3 = 200
    #    t_k = 350
    #    gamma_oil = 0.86
    #    p_MPaa = 10
    #    pb_mpa = 15
    #    self.assertAlmostEqual(pvt.unf_compressibility_saturated_oil_McCain_1Mpa(p_MPaa, pb_mpa, t_k, gamma_oil, rsb_m3m3),
    #                           0.00078861, delta=0.0001)
    #  тест пока работает неправильно, надо отдельно разобраться с расчетом сжимаемостей

    def test_unf_compressibility_oil_VB_1Mpa(self):
        rs_m3m3 = 200
        t_K = 350
        gamma_oil = 0.86
        p_MPaa = 15
        gamma_gas = 0.6
        self.assertAlmostEqual(pvt.unf_compressibility_oil_VB_1Mpa(rs_m3m3, t_K, gamma_oil, p_MPaa, gamma_gas),
                               0.004546552811369566, delta=0.0001)


    def test_unf_gamma_gas_Mccain(self):
        rsp_m3m3 = 30
        rst_m3m3 = 20
        gamma_gassp = 0.65
        gamma_oil = 0.86
        psp_MPaa = 5
        tsp_K = 350
        self.assertAlmostEqual(pvt.unf_gamma_gas_Mccain(rsp_m3m3, rst_m3m3, gamma_gassp, gamma_oil, psp_MPaa, tsp_K),
                               0.7932830162938984, delta=0.0001)




# Executing the tests in the above test case class
if __name__ == "__main__":
  unittest.main()