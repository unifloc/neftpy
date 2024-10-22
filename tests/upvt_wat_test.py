import unittest
import neftpy.upvt_wat as pvtw
import neftpy.upvt_np_vect as pvt_vect

import numpy as np

class PVTwTestCase(unittest.TestCase):
    

    def test_unf_density_brine_Spivey_kgm3(self):
        t_K = 350
        p_MPaa = 20
        s_ppm = 10000
        par = 1
        self.assertAlmostEqual(pvtw.unf_density_brine_Spivey_kgm3(t_K, p_MPaa, s_ppm, par), 987.685677686006, delta=0.0001)

    def test_unf_compressibility_brine_Spivey_1MPa(self):
        t_K = 350
        p_MPaa = 20
        s_ppm = 10000
        z = 1
        par = 0
        self.assertAlmostEqual(pvtw.unf_compressibility_brine_Spivey_1MPa(t_K, p_MPaa, s_ppm, z, par), 0.0004241522548512511,
                               delta=0.0001)

    def test_unf_density_brine_uniflocvba_kgm3(self):
        gamma_w = 1
        bw_m3m3 = 1.1
        self.assertAlmostEqual(pvtw.unf_rho_water_bw_kgm3(gamma_w, bw_m3m3), 909.090909090909, delta=0.0001)


    def test_unf_fvf_brine_McCain_m3m3(self):
        t_K = 300
        p_MPaa =20
        self.assertAlmostEqual(pvtw.unf_fvf_brine_McCain_m3m3(t_K, p_MPaa), 1.0007434853666817,
                               delta=0.0001)

    def test_unf_fvf_brine_Spivey_m3m3(self):
        t_K = 350
        p_MPaa = 20
        s_ppm = 10000
        self.assertAlmostEqual(pvtw.unf_fvf_brine_Spivey_m3m3(t_K, p_MPaa, s_ppm), 1.0279011434122953, delta=0.0001)

    def test_unf_viscosity_brine_McCain_cp(self):
        t_K = 350
        p_MPaa = 20
        s_ppm = 10000
        self.assertAlmostEqual(pvtw.unf_mu_water_McCain_cP(t_K, p_MPaa, s_ppm), 0.4165673950441691, delta=0.0001)

    def test_unf_viscosity_brine_MaoDuan_cP(self):
        t_K = 350
        p_MPaa = 20
        s_ppm = 10000
        self.assertAlmostEqual(pvtw.unf_viscosity_brine_MaoDuan_cP(t_K, p_MPaa, s_ppm), 0.3745199364964906, delta=0.0001)



    def test_unf_gwr_brine_Spivey_m3m3(self):
        s_ppm = 10000
        z = 1
        self.assertAlmostEqual(pvtw.unf_gwr_brine_Spivey_m3m3(s_ppm, z), 0.0013095456419714546, delta=0.0001)

    def test_unf_surface_tension_go_Abdul_Majeed_Nm(self):
        t_K = 350
        rs_m3m3 = 50
        gamma_oil = 0.6
        z = 1
        self.assertAlmostEqual(pvtw.unf_surface_tension_go_Abdul_Majeed_Nm(t_K, gamma_oil, rs_m3m3),
                               0.003673109943227455, delta=0.0001)

    def test_unf_surface_tension_go_Baker_Swerdloff_Nm(self):
        t_K = 350
        p_MPaa = 0.1
        gamma_oil = 0.6
        z = 1
        self.assertAlmostEqual(pvtw.unf_surface_tension_go_Baker_Swerdloff_Nm(t_K, gamma_oil, p_MPaa),
                               0.01054309596387229, delta=0.0001)





    def test_unf_heat_capacity_water_IAPWS_JkgC(self):
        t_c = 20
        self.assertAlmostEqual(pvtw.unf_heat_capacity_water_IAPWS_JkgC(t_c),
                               4184.92592,
                               delta=0.0001)

    def test_unf_thermal_conductivity_water_IAPWS_WmC(self):
        t_c = 20
        self.assertAlmostEqual(pvtw.unf_thermal_conductivity_water_IAPWS_WmC(t_c),
                               0.5992595999999999,
                               delta=0.0001)

    def test_unf_thermal_expansion_coefficient_water_IAPWS_1C(self):
        t_c = 20
        self.assertAlmostEqual(pvtw.unf_thermal_expansion_coefficient_water_IAPWS_1C(t_c),
                               0.00022587,
                               delta=0.0001)

    def test_unf_surface_tension_gw_Sutton_Nm(self):
        rho_water_kgm3 = 1000
        rho_gas_kgm3 = 50
        t_c = 60
        self.assertAlmostEqual(pvtw.unf_surface_tension_gw_Sutton_Nm(rho_water_kgm3, rho_gas_kgm3, t_c),
                               0.06256845320633196,
                               delta=0.0001)




    def test_unf_surface_tension_Baker_Sverdloff_vba_nm(self):
        p_atma = 10
        t_C = 20
        gamma_o_ = 40
        self.assertAlmostEqual(sum(pvtw.unf_surface_tension_Baker_Sverdloff_vba_nm(p_atma, t_C, gamma_o_)),
                               0.12299551951661537,
                               delta=0.0001)
