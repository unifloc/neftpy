import unittest
import neftpy.upvt_oil as pvto
import neftpy.upvt_np_vect as pvtovect

import numpy as np

class PVTTestCase(unittest.TestCase):

    """
    Расчет давления насыщения по Стендингу
    """
    def test_unf_pb_Standing_MPaa(self):
        rsb_m3m3 = 100
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvto.unf_pb_Standing_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), 
                                20.170210695316566,
                                delta=0.0001)

    def test_array_unf_pb_Standing_MPaa(self):
        """ check with np.array as input """

        # векторные значения импортированы из расчета unifloc vba
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 353
        unf_vba_res = [0.448907580141515,13.4987764643021,23.8326700405825,33.2913002316227,42.2211469979671,50.7767705913968]
        unf_vba_rsb = [1,60.8,120.6,180.4,240.2,300]

        rsb_m3m3 = np.array(unf_vba_rsb)        
        res = np.array(unf_vba_res)

        self.assertTrue(np.allclose(pvtovect.unf_pb_Standing_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), 
                                    res, 
                                    rtol=0.0001))

    """
    Расчет давления насыщения по Валко МакКейну
    """
    def test_unf_pb_Valko_MPaa(self):
        rsb_m3m3 = 100
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvto.unf_pb_Valko_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), 
                                23.29991481380937,
                                delta=0.00001)

    def test_array_unf_pb_Valko_MPaa(self):
        """ compares array input for vectorised realisations """

        # векторные значения импортированы из расчета unifloc vba
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 353
        unf_vba_res = [0.574600669143255,15.9048454457591,27.0883630604543,35.8231121745442,42.8445997446697,48.5938545235162]
        unf_vba_rsb = [1,60.8,120.6,180.4,240.2,300]

        rsb_m3m3 = np.array(unf_vba_rsb)        
        res = np.array(unf_vba_res)
        
        self.assertTrue(np.allclose(pvtovect.unf_pb_Valko_MPaa(rsb_m3m3, gamma_oil, gamma_gas, t_K), 
                                    res, 
                                    rtol=0.001))


    def test_unf_rs_Standing_m3m3(self):
        p_MPaa = 10
        pb_MPaa = 15
        rsb = 200
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvto.unf_rs_Standing_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K),
                               122.74847910146916, 
                               delta=0.0001)

    def test_array_unf_rs_Standing_m3m3(self):

        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 353
        unf_vba_res = [2.63345652886037,15.6762859290458,31.2325939746016,48.2819334435268,66.436537812051,85.4819163871848]
        unf_vba_p_MPaa = [1,4.4,7.8,11.2,14.6,18]
        pb_MPaa = 0
        rsb = 120


        p_MPaa = np.array(unf_vba_p_MPaa)        
        res = np.array(unf_vba_res)
        
        self.assertTrue(np.allclose(pvtovect.unf_rs_Standing_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K),
                                    res, 
                                    rtol=0.0001))
        

    def test_unf_rs_Velarde_m3m3(self):
        p_MPaa = 10
        pb_MPaa = 15
        rsb = 250
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvto.unf_rs_Velarde_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K),
                               170.25302712587356, delta=0.0001)


    def test_array_unf_rs_Velarde_m3m3(self):

        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 353
        pb_MPaa = 12.8300024673082
        rsb = 120
        unf_vba_res = [13.769156001658,47.1724892015713,76.4710407632712,105.644331604199,120,120]
        unf_vba_p_MPaa = [1,4.4,7.8,11.2,14.6,18]


        p_MPaa = np.array(unf_vba_p_MPaa)        
        res = np.array(unf_vba_res)
        
        # получается сходимость на уровне 1 знак после запятой с unifloc vba, что для газосодержания в принципе допустимо
        # причины расхождени видимо в количестве учитываемых знаков после запятой в константах или в определении стандартных величин
        # принты далее могут показать расхождения по отдельным значениям
        #print(unf_vba_res)
        #print(pvtovect.unf_rs_Velarde_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K))

        self.assertTrue(np.allclose(pvtovect.unf_rs_Velarde_m3m3(p_MPaa, pb_MPaa, rsb, gamma_oil, gamma_gas, t_K),
                                    res, 
                                    rtol=0.01))

                  
    def test_unf_rsb_Mccain_m3m3(self):
        rsp_m3m3 = 150
        gamma_oil = 0.86
        psp_MPaa = 5
        tsp_K = 320
        self.assertAlmostEqual(pvto.unf_rsb_Mccain_m3m3(rsp_m3m3, gamma_oil, psp_MPaa, tsp_K),
                               161.03286985548442, delta=0.0001)

# Executing the tests in the above test case class
if __name__ == "__main__":
  unittest.main()