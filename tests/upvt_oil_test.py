import unittest
import neftpy.upvt_oil as pvto
import neftpy.upvt_np_vect as pvt_vect

import numpy as np

class PVToTestCase(unittest.TestCase):

    """
    Расчет давления насыщения по Стендингу
    """
    def test_unf_pb_Standing_MPaa(self):
        rsb_m3m3 = 100
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvto.unf_pb_Standing_MPaa(t_K, rsb_m3m3, gamma_oil, gamma_gas), 
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

        self.assertTrue(np.allclose(pvt_vect.unf_pb_Standing_MPaa(t_K, rsb_m3m3, gamma_oil, gamma_gas), 
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
        self.assertAlmostEqual(pvto.unf_pb_Valko_MPaa(t_K, rsb_m3m3, gamma_oil, gamma_gas), 
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
        
        self.assertTrue(np.allclose(pvt_vect.unf_pb_Valko_MPaa(t_K, rsb_m3m3, gamma_oil, gamma_gas), 
                                    res, 
                                    rtol=0.001))


    def test_unf_rs_Standing_m3m3(self):
        p_MPaa = 10
        pb_MPaa = 15
        rsb = 200
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvto.unf_rs_Standing_m3m3(p_MPaa, t_K, pb_MPaa, rsb, gamma_oil, gamma_gas),
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
        
        self.assertTrue(np.allclose(pvt_vect.unf_rs_Standing_m3m3(p_MPaa, t_K, pb_MPaa, rsb, gamma_oil, gamma_gas),
                                    res, 
                                    rtol=0.0001))
        

    def test_unf_rs_Velarde_m3m3(self):
        p_MPaa = 10
        pb_MPaa = 15
        rsb = 250
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvto.unf_rs_Velarde_m3m3(p_MPaa, t_K, pb_MPaa, rsb, gamma_oil, gamma_gas),
                               170.25302712587356, 
                               delta=0.0001)


    def test_unf_rs_Velarde_2_m3m3(self):
        p_MPaa = 10
        pb_MPaa = 15
        rsb = 250
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 350
        self.assertAlmostEqual(pvto._unf_rs_Velarde_m3m3_(p_MPaa, t_K, pb_MPaa, rsb, gamma_oil, gamma_gas),
                               170.25302712587356, 
                               delta=0.0001)

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

        self.assertTrue(np.allclose(pvt_vect.unf_rs_Velarde_m3m3(p_MPaa, t_K, pb_MPaa, rsb, gamma_oil, gamma_gas),
                                    res, 
                                    rtol=0.01))

    def test_array_unf_rs_Velarde_2_m3m3(self):

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

        self.assertTrue(np.allclose(pvt_vect.unf_rs_Velarde_2_m3m3(p_MPaa, t_K, pb_MPaa, rsb, gamma_oil, gamma_gas),
                                    res, 
                                    rtol=0.01))

                  
    def test_unf_rsb_Mccain_m3m3(self):
        rsp_m3m3 = 150
        gamma_oil = 0.86
        psp_MPaa = 5
        tsp_K = 320
        self.assertAlmostEqual(pvto.unf_rsb_Mccain_m3m3(rsp_m3m3, psp_MPaa, tsp_K, gamma_oil),
                               161.03286985548442, 
                               delta=0.0001)
        
    
    def test_unf_bo_above_pb_m3m3(self):
        bob = 1.3
        cofb_1MPa = 3e-3
        pb_MPaa = 12
        p_MPaa = 15
        self.assertAlmostEqual(pvto.unf_bo_above_pb_m3m3(p_MPaa, pb_MPaa, bob, cofb_1MPa), 
                               1.2883524924047487, 
                               delta=0.0001)


    def test_unf_bo_below_m3m3(self):
        density_oilsto_kgm3 = 800
        rs_m3m3 = 200
        density_oil_kgm3 = 820
        gamma_gas = 0.6
        self.assertAlmostEqual(pvto.unf_bo_below_pb_m3m3(density_oilsto_kgm3, density_oil_kgm3,  rs_m3m3,  gamma_gas),
                               1.1542114715227887, 
                               delta=0.001
                               )
        
    def test_unf_fvf_Standing_m3m3_saturated(self):
        rs_m3m3 = 200
        gamma_gas = 0.6
        gamma_oil = 0.86
        t_K = 350
        self.assertAlmostEqual(pvto.unf_bo_saturated_Standing_m3m3(t_K, rs_m3m3, gamma_oil, gamma_gas),
                               1.5527836202040448, delta=0.0001)
        
    def test_array_unf_bo_saturated_Standing_m3m3(self):
        """ compares array input for vectorised realisations """

        # векторные значения импортированы из расчета unifloc vba
        gamma_oil = 0.86
        gamma_gas = 0.6
        t_K = 353
        unf_vba_res = [1.05686373329632,1.19233438531874,1.34242780220036,1.50241084963365,1.67002047706012,1.8438982299009]
        unf_vba_rsb = [1,60.8,120.6,180.4,240.2,300]


        rsb_m3m3 = np.array(unf_vba_rsb)        
        res = np.array(unf_vba_res)
        
        self.assertTrue(np.allclose(pvto.unf_bo_saturated_Standing_m3m3(t_K, rsb_m3m3, gamma_oil,  gamma_gas), 
                                    res, 
                                    rtol=0.001))
        
        
    def test_unf_density_oil_Mccain(self):
        p_MPaa = 10
        pb_MPaa = 12
        co_1MPa = 3e-3
        rs_m3m3 = 250
        gamma_gas = 0.6
        t_K = 350
        gamma_oil = 0.86
        gamma_gassp = 0
        self.assertAlmostEqual(pvto.unf_density_oil_Mccain(p_MPaa, t_K, pb_MPaa, rs_m3m3, co_1MPa, gamma_oil, gamma_gas, gamma_gassp), 
                               630.0536681794456, 
                               delta=0.0001)

    def test_array_unf_density_oil_Mccain(self):
        p_MPaa = np.array([0.1, 1,5,10, 20, 30])
        pb_MPaa = 10
        co_1MPa = 3e-3
        rs_m3m3 = 150
        gamma_gas = 0.6
        t_K = 350
        gamma_oil = 0.86
        gamma_gassp = 0
        self.assertTrue(np.allclose(pvto.unf_density_oil_Mccain(p_MPaa, t_K,pb_MPaa, rs_m3m3, co_1MPa, gamma_oil, gamma_gas, gamma_gassp),
                               [684.20874678, 684.97844157, 688.3460601,  692.43351253, 713.52125245, 735.25120966]) )
  
    def test_unf_density_oil_Standing(self):
        p_MPaa = 10
        pb_MPaa = 12
        co_1MPa = 3e-3
        rs_m3m3 = 250
        bo_m3m3 = 1.1
        gamma_gas = 0.6
        gamma_oil = 0.86
        self.assertAlmostEqual(pvto.unf_density_oil_Standing(p_MPaa, rs_m3m3, pb_MPaa, bo_m3m3, co_1MPa, gamma_oil, gamma_gas), 
                               948.863636, 
                               delta=0.0001)
        # значение изменено по сравнению с выводом унифлок vba на 0.13 из за корректировки плотности воздуха
    
    def test_unf_compressibility_saturated_oil_VB_1Mpa(self):
        rsb_m3m3 = 200
        t_k = 350
        gamma_oil = 0.86
        p_MPaa = 10
        pb_mpa = 15
        self.assertAlmostEqual(pvto.unf_compressibility_saturated_oil_McCain_1Mpa(p_MPaa, t_k, pb_mpa, rsb_m3m3, gamma_oil),
                               0.004934802450463976, 
                               delta=0.003)
 
    def test_unf_compressibility_oil_VB_1Mpa(self):
        rs_m3m3 = 200
        t_K = 350
        gamma_oil = 0.86
        p_MPaa = 15
        gamma_gas = 0.6
        self.assertAlmostEqual(pvto.unf_compressibility_oil_VB_1Mpa(p_MPaa, t_K, rs_m3m3, gamma_oil,  gamma_gas),
                               0.004546552811369566, 
                               delta=0.0001)


    def test_unf_gamma_gas_Mccain(self):
        rsp_m3m3 = 30
        rst_m3m3 = 20
        gamma_gassp = 0.65
        gamma_oil = 0.86
        psp_MPaa = 5
        tsp_K = 350
        self.assertAlmostEqual(pvto.unf_gamma_gas_Mccain(psp_MPaa, tsp_K, rsp_m3m3, rst_m3m3, gamma_oil, gamma_gassp, ),
                               0.7932830162938984, 
                               delta=0.0001)


    def test_unf_deadoilviscosity_Beggs_cP(self):
        gamma_oil = 0.86
        t_K = 350
        self.assertAlmostEqual(pvto.unf_viscosity_deadoil_Beggs_cP(t_K, gamma_oil), 
                               2.86938394460968, 
                               delta=0.0001)

    def test_unf_saturatedoilviscosity_Beggs_cP(self):
        deadoilviscosity_cP = 2.87
        rs_m3m3 = 150
        self.assertAlmostEqual(pvto.unf_viscosity_saturatedoil_Beggs_cP(deadoilviscosity_cP, rs_m3m3), 
                               0.5497153091178292,
                               delta=0.0001)

    def test_unf_undersaturatedoilviscosity_VB_cP(self):
        p_MPaa = 10
        pb_MPaa = 12
        bubblepointviscosity_cP = 1
        self.assertAlmostEqual(pvto.unf_viscosity_undersaturatedoil_VB_cP(p_MPaa, pb_MPaa, bubblepointviscosity_cP),
                               0.9767303348551418, 
                               delta=0.0001)


    def test_unf_oil_viscosity_Beggs_VB_cP(self):
        deadoilviscosity_cP = 2.87
        rs_m3m3 = 150
        p_MPaa = 10
        pb_MPaa = 12
        self.assertAlmostEqual(pvto.unf_viscosity_oil_Beggs_VB_cP(p_MPaa, rs_m3m3,  pb_MPaa, deadoilviscosity_cP),
                               0.5497153091178292, 
                               delta=0.0001)
        
    def test_unf_heat_capacity_oil_Gambill_JkgC(self):
        gamma_oil = 0.8
        t_c = 60
        self.assertAlmostEqual(pvto.unf_heat_capacity_oil_Gambill_JkgC(gamma_oil, t_c),
                               2110.7207148850844,
                               delta=0.0001)

    def test_unf_thermal_conductivity_oil_Cragoe_WmK(self):
        gamma_oil = 0.8
        t_c = 60
        self.assertAlmostEqual(pvto.unf_thermal_conductivity_oil_Cragoe_WmK(gamma_oil, t_c),
                               0.1427090525,
                               delta=0.0001)

    def test_unf_deadoilviscosity_Standing(self):
        t_K = 353
        gamma_oil = 0.86
        val = 2.19722776476976
        self.assertAlmostEqual(pvto.unf_viscosity_deadoil_Standing(t_K, gamma_oil),
                               val,
                               delta=0.01)


    def test_unf_deadoilviscosity_BeggsRobinson_VBA_cP(self):
        gamma_oil = 0.8
        t_K = 300
        self.assertAlmostEqual(pvto.unf_viscosity_deadoil_BeggsRobinson_cP(t_K, gamma_oil),
                               5.264455765058494,
                               delta=0.1)
        
    def test_unf_deadoilviscosity_BeggsRobinson_cP(self):
        t_K = 353
        gamma_oil = 0.86
        val = 2.68955979075439

        self.assertAlmostEqual(pvto.unf_viscosity_deadoil_BeggsRobinson_cP(t_K, gamma_oil),
                               val,
                               delta=0.0001)

    def test_unf_viscosity_oil_Standing_cP(self):
        rs = 100
        nu = 2
        p = 10
        pb = 12
        val = 0.716575048793071


        self.assertAlmostEqual(pvto.unf_viscosity_oil_Standing_cP(p,rs,  pb, nu),
                               val,
                               delta=0.0001)
        

    def test_unf_undersaturatedoilviscosity_Petrovsky_cP(self):
        p_MPaa = 10
        pb_MPaa = 12
        bubblepointviscosity_cP = 1
        self.assertAlmostEqual(pvto.unf_viscosity_undersaturatedoil_Petrosky_cP(p_MPaa, pb_MPaa, bubblepointviscosity_cP),
                               0.9622774530985722, 
                               delta=0.0001)

    def test_unf_pb_Glaso_MPaa(self):
        rs_m3m3 = 100
        t_K = 350
        gamma_oil = 0.86
        gamma_gas = 0.6
        self.assertAlmostEqual(pvto.unf_pb_Glaso_MPaa(t_K, rs_m3m3,  gamma_oil, gamma_gas), 
                               23.365669948236604, 
                               delta=0.0001)

    def test_unf_fvf_Glaso_m3m3_saturated(self):
        rs_m3m3 = 100
        t_K = 350
        gamma_oil = 0.86
        gamma_gas = 0.6
        self.assertAlmostEqual(pvto.unf_bo_saturated_Glaso_m3m3( t_K, rs_m3m3, gamma_oil, gamma_gas), 1.2514004319480372,
                               delta=0.0001)

    def test_unf_fvf_Glaso_m3m3_below(self):
        rs_m3m3 = 100
        t_K = 350
        gamma_oil = 0.86
        gamma_gas = 0.6
        p_MPaa = 10
        self.assertAlmostEqual(pvto.unf_bo_below_Glaso_m3m3(p_MPaa, t_K, rs_m3m3, gamma_oil, gamma_gas ), 1.7091714311161692,
                               delta=0.0001)

    def test_unf_McCain_specificgravity(self):
        p_MPaa = 10
        rsb_m3m3 = 100
        t_K = 350
        gamma_oil = 0.8
        gamma_gassp = 0.6
        self.assertAlmostEqual(pvto.unf_McCain_specificgravity(p_MPaa, t_K, rsb_m3m3, gamma_oil, gamma_gassp), 0.6004849666507259,
                               delta=0.0001)
        
    def test_unf_heat_capacity_oil_Wes_Wright_JkgC(self):
        gamma_oil = 0.8
        t_c = 60
        self.assertAlmostEqual(pvto.unf_heat_capacity_oil_Wes_Wright_JkgC(gamma_oil, t_c),
                               2162.0, delta=0.0001)

    def test_unf_thermal_conductivity_oil_Abdul_Seoud_Moharam_WmK(self):
        gamma_oil = 0.8
        t_c = 60
        self.assertAlmostEqual(pvto.unf_thermal_conductivity_oil_Abdul_Seoud_Moharam_WmK(gamma_oil, t_c),
                               0.10999860144810972, delta=0.0001)

    def test_unf_thermal_conductivity_oil_Smith_WmK(self):
        gamma_oil = 0.8
        t_c = 60
        self.assertAlmostEqual(pvto.unf_thermal_conductivity_oil_Smith_WmK(gamma_oil, t_c),
                               0.16568762875, delta=0.0001)
# Executing the tests in the above test case class
if __name__ == "__main__":
  unittest.main()