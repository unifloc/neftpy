import unittest
import neftpy.upvt_oil as pvto
import neftpy.upvt_gas as pvtg

import neftpy.upvt_np_vect as pvt_vect

import numpy as np

class PVTgTestCase(unittest.TestCase):

        
    def test_unf_pseudocritical_pressure_MPa_temperature_K(self):
        gamma_gas = 0.6
        y_h2s = 0.01
        y_co2 = 0.03
        y_n2 = 0.02
        self.assertTrue(np.allclose(pvtg.unf_pseudocritical_McCain_p_MPa_t_K(gamma_gas, y_h2s, y_co2, y_n2), 
                               (5.09893164741181, 198.0708725589674),
                               rtol=0.0001))


    def test_unf_pseudocritical_Standing_p_MPa_t_K(self):
        gamma_gas = 0.6
        self.assertTrue(np.allclose(pvtg.unf_pseudocritical_Standing_p_MPa_t_K(gamma_gas), 
                               (4.567119999999999, 198.8016),
                               rtol=0.0001))
        

        
    def test_unf_zfactor(self):
        p_MPaa = 10
        t_K = 350
        gamma_gas = 0.6
        self.assertAlmostEqual(pvtg.unf_zfactor(p_MPaa, t_K, gamma_gas), 
                               0.91052266, 
                               delta=0.0001)

    def test_unf_zfactor_DAK(self):
        ppr = 4
        tpr = 2

        self.assertAlmostEqual(pvtg.unf_zfactor_DAK(ppr, tpr), 
                               0.9426402059431057,
                               delta=0.0001)

    def test_unf_z_factor_Kareem(self):
        tpr = 1.2
        ppr = 1.2
        self.assertAlmostEqual(pvtg.unf_zfactor_Kareem(ppr, tpr),
                               0.71245963496651,
                               delta=0.0001)
        
    def test_unf_gasviscosity_Lee_cP(self):
        t_K = 350
        p_MPaa = 10
        z = 0.84
        gamma_gas = 0.6
        self.assertAlmostEqual(pvtg.unf_mu_gas_Lee_z_cP(p_MPaa, t_K,  z, gamma_gas), 0.017563524546592567, delta=0.0001)

    def test_unf_gas_fvf_m3m3(self):
        t_K = 350
        p_MPaa = 10
        z = 0.84
        self.assertAlmostEqual(pvtg.unf_bg_gas_z_m3m3(p_MPaa, t_K,  z), 0.010162381033600544, delta=0.0001)


    def test_unf_rho_gas_bg_kgm3(self):
        gamma_gas = 0.6
        b_gas_m3m3 = 0.005
        self.assertAlmostEqual(pvtg.unf_rho_gas_bg_kgm3(gamma_gas, b_gas_m3m3),
                               144.49815477244613,
                               delta=0.0001)

    def test_unf_bg_gas_z_m3m3(self):
        T_K = 300
        z = 1.1
        P_MPa = 0.3
        self.assertAlmostEqual(pvtg.unf_bg_gas_z_m3m3(P_MPa, T_K, z ),
                               0.3802063789868669,
                               delta=0.0001)
        
        

    def test_unf_heat_capacity_gas_Mahmood_Moshfeghian_JkgC(self):
        p_MPaa = 3
        gamma_gas = 0.8
        t_K = 300
        self.assertAlmostEqual(pvtg.unf_heat_capacity_gas_Mahmood_Moshfeghian_JkgC(p_MPaa, t_K, gamma_gas),
                               2471.4603282835255,
                               delta=0.0001)

    def test_unf_thermal_conductivity_gas_methane_WmK(self):
        t_c = 20
        self.assertAlmostEqual(pvtg.unf_thermal_conductivity_gas_methane_WmK(t_c),
                               0.033390322580645164,
                               delta=0.0001)
        
    def test_unf_zfactor_BrillBeggs(self):
        ppr = 2
        tpr = 2
        self.assertAlmostEqual(pvtg.unf_zfactor_BrillBeggs(ppr, tpr), 
                               0.9540692750239955, 
                               delta=0.0001)

    def test_unf_unf_zfactor_SK(self):
        ppr = 2
        tpr = 1.2
        self.assertAlmostEqual(pvtg.unf_zfactor_SK(ppr, tpr), 
                               0.554, 
                               delta=0.0001)
        
    def test_unf_gas_density_kgm3(self):
        t_K = 350
        p_MPaa = 0.1
        gamma_gas = 0.6
        z = 1
        self.assertAlmostEqual(pvtg.unf_rho_gas_z_kgm3( p_MPaa, t_K, gamma_gas, z), 
                               0.5972244062111675, 
                               delta=0.0001)


    #def test_unf_compressibility_gas_Mattar_1MPa(self):
    #    p_MPaa = 10
    #    t_K = 350
    #    ppc_MPa = 7.477307083789863
    #    tpc_K = 239.186917147216
    #    self.assertAlmostEqual(pvtg.unf_gas_isotermal_compressibility_1MPa(p_MPaa, t_K, ppc_MPa, tpc_K), 
    #                           0.4814932416304309,
    #                           delta=0.0001)