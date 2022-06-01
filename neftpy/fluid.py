"""
реализация модели нелетучей нефти
"""

import numpy as np
import neftpy.uconvert as uc
import neftpy.upvt as upvt
import scipy.optimize as opt

class BlackOilStanding:

    def __init__(self, 
                gamma_oil=0.86, 
                gamma_gas=0.6, 
                gamma_wat=1.0, 
                rsb_m3m3=100.0,
                tb_C=80, 
                y_h2s=0, 
                y_co2=0,
                y_n2=0  ):
        """
        Создает флюид с заданными базовыми свойствами

        калибровочные параметры при необходимости надо задавать отдельно

        :param gamma_oil: specific gravity of oil
        :param gamma_gas: specific gravity of gas (by air), dimensionless
        :param gamma_wat: specific gravity of water
        :param rsb_m3m3: solution gas ratio at bubble point
        :param tb_C: reservoir temperature 
        :param y_h2s: mole fraction of the hydrogen sulfide
        :param y_co2: mole fraction of the carbon dioxide
        :param y_n2: mole fraction of the nitrogen
        """
        self.gamma_gas = gamma_gas
        self.gamma_oil = gamma_oil
        self.gamma_wat = gamma_wat
        self.rsb_m3m3 = rsb_m3m3
        self.tb_C = tb_C  
        self.y_h2s = y_h2s
        self.y_co2 = y_co2
        self.y_n2 = y_n2
        # термобарические условия для которых рассчитаны свойства
        self.p_bar = uc.pressure_sc_bar                 # thermobaric conditions for all parameters
        self.t_c = uc.temperature_sc_C                  # can be set up by calc method

        self.__p_calibrated_MPaa =  uc.bar_2_MPa(self.p_bar)

        # расчетные параметры доступны только для чтения через свойства

        self.pb_bar = 0.0
        self.rs_m3m3 = 0.0
        self.rho_oil_kgm3 = 0.0
        self.b_oilb_m3m3 = 0.0
        self.b_oil_m3m3 = 1.0
        self.compr_oil_1bar = 0.0

        self.mu_dead_oil_cP = 0.0
        self.mu_oilb_cP = 0.0
        self.mu_oil_cP = 0.0  
    
        self.mu_gas_cP = 0.0
        self.mu_wat_cP = 0.0
        self.rho_gas_kgm3 = 0.0
        self.rho_wat_kgm3 = 0.0
        self.rsw_m3m3 = 0.0   # TODO проверить - равен GWR?
        self.b_gas_m3m3 = 0.0
        self.b_wat_m3m3 = 0.0
        self.z = 0.0
        self.compr_gas_1bar = 0.0
        self.compr_wat_1bar = 0.0
        self.heatcap_oil_jkgc = 0.0
        self.heatcap_gas_jkgc = 0.0
        self.heatcap_wat_jkgc = 0.0
        self.sigma_oil_gas_Nm = 0.0
        self.sigma_wat_gas_Nm = 0.0
        self.thermal_conduct_oil_wmk = 0.0
        self.thermal_conduct_gas_wmk = 0.0
        self.thermal_conduct_wat_wmk = 0.0
        self.thermal_expansion_wat_1c = 0.0
        
        self.activate_rus_cor = 0

        # калибровочные параметры для внутренних расчетов
        self.p_cal_mult = 1
        self.b_oil_cal_mult = 1
        self.mu_cal_mult = 1

        self.pb_calibr_bar = 0.0
        self.tb_calibr_C = 0.0
        self.b_oilb_calibr_m3m3 = 0.0
        self.mu_oilb_calibr_cP = 0.0

    def set_calibration_param(
                self,
                pb_calibr_bar=-1., 
                tb_calibr_C=80, 
                b_oilb_calibr_m3m3=0, 
                mu_oilb_calibr_cP=0.5
                ):
        # установка калибровочных параметров  
        # как отдельный метод, чтобы в явном виде обратить внимание пользователя на 
        # механизм калибровки
        self.pb_calibr_bar = pb_calibr_bar
        self.tb_calibr_C = tb_calibr_C
        self.b_oilb_calibr_m3m3 = b_oilb_calibr_m3m3
        self.mu_oilb_calibr_cP = mu_oilb_calibr_cP
        pass


    def __calc_pb_MPaa(self):
        self.pb_MPaa = upvt.unf_pb_Standing_MPaa(self.rsb_m3m3, self.gamma_oil, self.gamma_gas, self.t_K)
        self.pb_bar = uc.MPa_2_bar(self.pb_MPaa)

        # найдем калибровочный коэффициент для давления насыщения
        self.p_cal_mult = np.where(self.pb_calibr_bar > 0, 
                                   self.pb_bar / self.pb_calibr_bar, 
                                   1.)  
        self.__p_calibrated_MPaa = uc.bar_2_MPa(self.p_bar * self.p_cal_mult)

    def __calc_rs_m3m3(self):
        self.rs_m3m3 = upvt.unf_rs_Standing_m3m3(self.__p_calibrated_MPaa, self.pb_MPaa, self.rsb_m3m3, self.gamma_oil, self.gamma_gas, self.t_K)

    #def _calc_compressibility_oil_1Mpa(self):
    #    return upvt.unf_compressibility_oil_VB_1Mpa(self._rs_m3m3, self.t_K, self.gamma_oil, self.pcal_MPaa, self.gamma_gas)

    #def _calc_b_oilb_m3m3(self):
    #    return upvt.unf_bo_saturated_Standing_m3m3(self.rsb_m3m3, self.gamma_gas, self.gamma_oil, self.t_K)

    def __calc_b_rho_compressibility_oil(self):
        # поскольку объемный коэффициент, плотность и сжимаемость при давлениях выше pb 
        # тесно связаны между собой - считаем их вместе
        co_1MPa = upvt.unf_compressibility_oil_VB_1Mpa(self.rs_m3m3, self.t_K, self.gamma_oil, self.__p_calibrated_MPaa, self.gamma_gas)
        self.compr_oil_1bar = uc.compr_1mpa_2_1bar(co_1MPa)

        # оценим значение объемного коэффициента
        # оценим значение объемного коэффициента при давлении насыщения
        self.b_oilb_m3m3 = upvt.unf_bo_saturated_Standing_m3m3(self.rsb_m3m3, self.gamma_gas, self.gamma_oil, self.t_K)
        self.__b_oil_m3m3 = np.where(self.__p_calibrated_MPaa > self.pb_MPaa, 
                                 upvt.unf_bo_above_pb_m3m3(self.b_oilb_m3m3, co_1MPa, self.pb_MPaa, self.__p_calibrated_MPaa),
                                 upvt.unf_bo_saturated_Standing_m3m3(self.rs_m3m3, self.gamma_gas, self.gamma_oil, self.t_K)
                                )
        
        # проверим необходимость калибровки значения объемного коэффициента
        self.b_oil_cal_mult = np.where(self.b_oilb_calibr_m3m3 > 0, (self.b_oilb_calibr_m3m3 - 1) / (self.b_oilb_m3m3 - 1), 1.)
        self.b_oil_m3m3 = 1 + self.b_oil_cal_mult * (self.__b_oil_m3m3 - 1)
        # плотность нефти    
        self.rho_oil_kgm3 = upvt.unf_density_oil_Standing(self.__p_calibrated_MPaa , self.pb_MPaa, co_1MPa, self.rs_m3m3, self.b_oil_m3m3,
                                                          self.gamma_gas, self.gamma_oil) 
        

    def __calc_mu_oil_cP(self):
        # оценим значение вязкости
        self.mu_dead_oil_cP = upvt.unf_deadoilviscosity_Beggs_cP(self.gamma_oil, self.t_K)
        self.mu_oilb_cP = upvt.unf_saturatedoilviscosity_Beggs_cP(self.mu_dead_oil_cP, self.rsb_m3m3)
        self.mu_oil_cP = upvt.unf_oil_viscosity_Beggs_VB_cP(self.mu_dead_oil_cP, self.rs_m3m3, self.__p_calibrated_MPaa , self.pb_MPaa)
        if self.mu_oilb_calibr_cP > 0:
            self.mu_cal_mult = self.mu_oilb_calibr_cP / self.mu_oilb_cP
            self.mu_oil_cP = self.mu_cal_mult * self.mu_oil_cP
        

    def __calc_termodynamic_oil_props(self):
        # определим термодинамические свойства нефти
        self._heatcap_oil_jkgc = upvt.unf_heat_capacity_oil_Gambill_JkgC(self.gamma_oil, self.t_c)
        self._thermal_conduct_oil_wmk = upvt.unf_thermal_conductivity_oil_Cragoe_WmK(self.gamma_oil, self.t_c)

    def calc(self, p_bar, t_C):
        self.p_bar = p_bar
        self.t_C = t_C 
        # в расчете часто используются MPa и K - сконвертируем и сохраним соответствующие значения
        self.p_MPaa = uc.bar_2_MPa(p_bar)
        self.t_K = uc.C_2_K(self.t_C)

        # оценим свойства нефти 
        self.__calc_pb_MPaa()       # давление насыщения

        self.__calc_rs_m3m3()       # газосодержание 
        
        self.__calc_b_rho_compressibility_oil()   # объемный коэффициент нефти, плотность нефти, сжимаемость нефти
        
        self.__calc_mu_oil_cP()     # вязкость нефти

        self.__calc_termodynamic_oil_props()

        # свойства газа 

        # свойства воды

"""
        # определим термодинамические свойства нефти
        self._heatcap_oil_jkgc = self._calc_heat_capacity_oil_JkgC()
        self._thermal_conduct_oil_wmk = self._calc_thermal_conductivity_oil_WmK()


        # gas
        tpc_K = PVT.unf_pseudocritical_temperature_K(self.gamma_gas, self.y_h2s, self.y_co2, self.y_n2)
        ppc_MPa = PVT.unf_pseudocritical_pressure_MPa(self.gamma_gas, self.y_h2s, self.y_co2, self.y_n2)
        self._z = PVT.unf_zfactor_DAK(p_MPaa, t_K, ppc_MPa, tpc_K)
        self._mu_gas_cP = PVT.unf_gasviscosity_Lee_cP(t_K, p_MPaa, self._z, self.gamma_gas)
        self._bg_m3m3 = PVT.unf_gas_fvf_m3m3(t_K, p_MPaa, self._z)
        self._compr_gas_1bar = uc.compr_1mpa_2_1bar(PVT.unf_compressibility_gas_Mattar_1MPa(p_MPaa, t_K,
                                                                                            ppc_MPa, tpc_K))
        self._rho_gas_kgm3 = PVT.unf_gas_density_kgm3(t_K, p_MPaa, self.gamma_gas, self._z)

        # определим термодинамические свойства газа
        self._heatcap_gas_jkgc = PVT.unf_heat_capacity_gas_Mahmood_Moshfeghian_JkgC(p_MPaa, t_K, self.gamma_gas)
        self._thermal_conduct_gas_wmk = PVT.unf_thermal_conductivity_gas_methane_WmK(self.t_c)

        # water
        # TODO НУЖНО проверить GWR
        self._rho_wat_kgm3 = PVT.unf_density_brine_Spivey_kgm3(t_K, p_MPaa, self.s_ppm, self.par_wat)
        self._compr_wat_1bar = uc.compr_1mpa_2_1bar(PVT.unf_compressibility_brine_Spivey_1MPa(t_K, p_MPaa, self.s_ppm,
                                                                                              self._z, self.par_wat))
        self._bw_m3m3 = PVT.unf_fvf_brine_Spivey_m3m3(t_K, p_MPaa, self.s_ppm)
        self._mu_wat_cP = PVT.unf_viscosity_brine_MaoDuan_cP(t_K, p_MPaa, self.s_ppm)
        self._rsw_m3m3 = PVT.unf_gwr_brine_Spivey_m3m3(self.s_ppm, self._z)
        # определим термодинамические свойства воды
        self._heatcap_wat_jkgc = PVT.unf_heat_capacity_water_IAPWS_JkgC(self.t_c)
        self._thermal_conduct_wat_wmk = PVT.unf_thermal_conductivity_water_IAPWS_WmC(self.t_c)
        self._thermal_expansion_wat_1c = PVT.unf_thermal_expansion_coefficient_water_IAPWS_1C(self.t_c)

        # определим свойства системы
        self._sigma_oil_gas_Nm = PVT.unf_surface_tension_go_Baker_Swerdloff_Nm(t_K, self.gamma_oil, p_MPaa)
        self._sigma_wat_gas_Nm = PVT.unf_surface_tension_gw_Sutton_Nm(self.rho_wat_kgm3, self.rho_gas_kgm3, self.t_c)
        return 1
"""
