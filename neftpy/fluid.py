"""
реализация модели нелетучей нефти
"""

import numpy as np
import neftpy.uconvert as uc
import neftpy.uconst as uconst
import neftpy.upvt_np_vect as upvt
import scipy.optimize as opt


class Feed:
    """
    свойства потока флюидов
    поток флюидов определяет PVT свойства всех флюидов - нефти, газа и воды
    а также их расходные характеристики - дебиты, доли флюида в потоке и прочее
    """
    def __init__(self):
        self.fluid = BlackOilStanding()
        
        self.q_liq_sm3day = 10
        self.fw_perc = 0
        self.rp_m3m3 = 100
        self.q_gas_free_sm3day = 0

    def calc(self, p_atma, t_C):
        self.fluid.calc(p_bar=p_atma, t_C=t_C)

    def fw_fr(self):
        """
        обводненность в долях единиц
        """
        return self.fw_perc / 100

    def q_oil_sm3day(self):
        """
        дебит нефти приведенный к стандартным условиях
        """
        return self.q_liq_sm3day * (1 - self.fw_fr)

    def q_gas_sm3day(self):
        """
        дебит газа приведенный к стандартным условиях
        учитывает весь газ потока - как свободный, так и растворенный в нефти
        """
        return self.q_oil_sm3day * self.rp_m3m3 + self.q_gas_free_sm3day  
    
    def q_gas_insitu_sm3day(self):
        """ 
        расход свободного газа в заданных термобарических условиях приведенный к стандартным условиям
        без учета растворенного в нефти газа
        """
        _q_gas_insitu_sm3day = self.q_gas_sm3day - self.fluid.rs_m3m3 * self.q_oil_sm3day
        return np.where(_q_gas_insitu_sm3day < 0,
                        0,
                        _q_gas_insitu_sm3day )

    def q_gas_rc_m3day(self):
        """
        расход свободного газа приведенный к расчетным термобарическим условиям
        """
        return self.q_gas_insitu_sm3day * self.fluid.b_gas_m3m3
    
    def q_water_sm3day(self):
        """
        расход воды, приведенные к стандартным условиям
        """
        return self.q_liq_sm3day * self.fw_fr


    def gas_fraction(self):
        pass



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
        self.p_atma = uconst.P_SC_atma                 # thermobaric conditions for all parameters
        self.t_C = uconst.T_SC_C                  # can be set up by calc method

        self.__p_calibrated_MPaa =  uc.atm_2_MPa(self.p_atma)

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
        self.pb_MPaa = upvt.unf_pb_Standing_MPaa(rsb_m3m3 = self.rsb_m3m3, 
                                                 gamma_oil = self.gamma_oil, 
                                                 gamma_gas = self.gamma_gas, 
                                                 t_K = self.t_K)
        self.pb_bar = uc.MPa_2_bar(self.pb_MPaa)

        # найдем калибровочный коэффициент для давления насыщения
        self.p_cal_mult = np.where(self.pb_calibr_bar > 0, 
                                   self.pb_bar / self.pb_calibr_bar, 
                                   1.)  
        self.__p_calibrated_MPaa = uc.bar_2_MPa(self.p_atma * self.p_cal_mult)

    def __calc_rs_m3m3(self):
        self.rs_m3m3 = upvt.unf_rs_Standing_m3m3(p_MPaa = self.__p_calibrated_MPaa, 
                                                 pb_MPaa = self.pb_MPaa, 
                                                 rsb_m3m3 = self.rsb_m3m3, 
                                                 gamma_oil = self.gamma_oil, 
                                                 gamma_gas = self.gamma_gas, 
                                                 t_K = self.t_K)

    def __calc_b_rho_compressibility_oil(self):
        # поскольку объемный коэффициент, плотность и сжимаемость при давлениях выше pb 
        # тесно связаны между собой - считаем их вместе
        co_1MPa = upvt.unf_compressibility_oil_VB_1Mpa(rs_m3m3 = self.rs_m3m3, 
                                                       t_K = self.t_K, 
                                                       gamma_oil = self.gamma_oil, 
                                                       p_MPaa = self.__p_calibrated_MPaa, 
                                                       gamma_gas = self.gamma_gas)
        self.compr_oil_1bar = uc.compr_1mpa_2_1bar(co_1MPa)

        # оценим значение объемного коэффициента
        # оценим значение объемного коэффициента при давлении насыщения
        self.b_oilb_m3m3 = upvt.unf_bo_saturated_Standing_m3m3(rs_m3m3 = self.rsb_m3m3, 
                                                               gamma_gas = self.gamma_gas, 
                                                               gamma_oil = self.gamma_oil, 
                                                               t_K = self.t_K)
        self.__b_oil_m3m3 = np.where(self.__p_calibrated_MPaa > self.pb_MPaa, 
                                 upvt.unf_bo_above_pb_m3m3(bob_m3m3 = self.b_oilb_m3m3, 
                                                           compr_o_1MPa = co_1MPa, 
                                                           pb_MPaa = self.pb_MPaa, 
                                                           p_MPaa = self.__p_calibrated_MPaa),
                                 upvt.unf_bo_saturated_Standing_m3m3(rs_m3m3 = self.rs_m3m3, 
                                                                     gamma_gas = self.gamma_gas, 
                                                                     gamma_oil = self.gamma_oil, 
                                                                     t_K = self.t_K)
                                )
        
        # проверим необходимость калибровки значения объемного коэффициента
        self.b_oil_cal_mult = np.where(self.b_oilb_calibr_m3m3 > 0, 
                                       (self.b_oilb_calibr_m3m3 - 1) / (self.b_oilb_m3m3 - 1), 
                                       1.)
        self.b_oil_m3m3 = 1 + self.b_oil_cal_mult * (self.__b_oil_m3m3 - 1)
        # плотность нефти    
        self.rho_oil_kgm3 = upvt.unf_rho_oil_Standing_kgm3(p_MPaa = self.__p_calibrated_MPaa , 
                                                          pb_MPaa = self.pb_MPaa, 
                                                          co_1MPa = co_1MPa, 
                                                          rs_m3m3 = self.rs_m3m3, 
                                                          bo_m3m3 = self.b_oil_m3m3,
                                                          gamma_gas = self.gamma_gas, 
                                                          gamma_oil = self.gamma_oil) 
        

    def __calc_mu_oil_cP(self):
        # оценим значение вязкости
        self.mu_dead_oil_cP = upvt.unf_viscosity_deadoil_Beggs_cP(gamma_oil = self.gamma_oil, 
                                                                 t_K = self.t_K)
        self.mu_oilb_cP = upvt.unf_viscosity_saturatedoil_Beggs_cP(mu_oil_dead_cP = self.mu_dead_oil_cP, 
                                                                   rs_m3m3 = self.rsb_m3m3)
        self.mu_oil_cP = upvt.unf_viscosity_oil_Beggs_VB_cP(mu_oil_dead_cP = self.mu_dead_oil_cP, 
                                                            rs_m3m3 = self.rs_m3m3, 
                                                            p_MPaa = self.__p_calibrated_MPaa, 
                                                            pb_MPaa = self.pb_MPaa)
        if self.mu_oilb_calibr_cP > 0:
            self.mu_cal_mult = self.mu_oilb_calibr_cP / self.mu_oilb_cP
            self.mu_oil_cP = self.mu_cal_mult * self.mu_oil_cP
        

    def __calc_termodynamic_oil_props(self):
        # определим термодинамические свойства нефти
        self._heatcap_oil_jkgc = upvt.unf_heat_capacity_oil_Gambill_JkgC(gamma_oil = self.gamma_oil, 
                                                                         t_C = self.t_C)
        self._thermal_conduct_oil_wmk = upvt.unf_thermal_conductivity_oil_Cragoe_WmK(gamma_oil = self.gamma_oil, 
                                                                                     t_C = self.t_C)

        # определим термодинамические свойства нефти
        #self._heatcap_oil_jkgc = self._calc_heat_capacity_oil_JkgC()
        #self._thermal_conduct_oil_wmk = self._calc_thermal_conductivity_oil_WmK()

    def __calc_gas_props(self, p_bar:float, t_C:float)->float:
        # расчет свойств газа
        # gas
        self.p_atma = p_bar
        self.t_C = t_C 
        ppc_MPa, tpc_K = upvt.unf_pseudocritical_McCain_p_MPa_t_K(gamma_gas = self.gamma_gas, 
                                                                  y_h2s = self.y_h2s, 
                                                                  y_co2 = self.y_co2, 
                                                                  y_n2 = self.y_n2)
        p_MPaa = uc.bar_2_MPa(p_bar)
        ppr = p_MPaa / ppc_MPa
        t_K = uc.C_2_K(t_C)
        tpr = t_K / tpc_K
        self.p_gas_pc_MPa = ppc_MPa
        self.t_gas_pc_K = tpc_K
        self.p_gas_pr = ppr
        self.t_gas_pr = tpr

        self.z = upvt.unf_zfactor_SK(ppr=ppr, tpr=tpr, safe=True)
        self.rho_gas_kgm3 = upvt.unf_rho_gas_z_kgm3(t_K = t_K, 
                                                    p_MPaa = p_MPaa, 
                                                    gamma_gas = self.gamma_gas, 
                                                    z =self.z)

        self.mu_gas_cP = upvt.unf_mu_gas_Lee_rho_cP(t_K = t_K, 
                                                    p_MPaa = p_MPaa, 
                                                    rho_gas_kgm3 = self.rho_gas_kgm3)
        self.bg_m3m3 = upvt.unf_bg_gas_z_m3m3(t_K = t_K, 
                                              p_MPaa = p_MPaa,
                                              z =self.z)
        #
        #
        #  TODO сжимаемость надо будет переделать чтобы тут расчитывалась через производные z 
        # 
        #self._compr_gas_1bar = uc.compr_1mpa_2_1bar(PVT.unf_compressibility_gas_Mattar_1MPa(p_MPaa, t_K,
        #                                                                                    ppc_MPa, tpc_K))

        # определим термодинамические свойства газа
        self._heatcap_gas_jkgc = upvt.unf_heat_capacity_gas_Mahmood_Moshfeghian_JkgC(p_MPaa = p_MPaa, 
                                                                                     t_K = t_K, 
                                                                                     gamma_gas = self.gamma_gas)
        self._thermal_conduct_gas_wmk = upvt.unf_thermal_conductivity_gas_methane_WmK(t_C = t_C)

    def __calc_water_props(self, p_bar:float, t_C:float)->float:
    
        p_MPaa = uc.bar_2_MPa(p_bar)
        t_K = uc.C_2_K(t_C)
        # water
        self.salinity_ppm = upvt.unf_salinity_from_gamma_water_ppm(gamma_water = self.gamma_wat)

        self.bw_m3m3 = upvt.unf_bw_McCain_m3m3(t_K = t_K, 
                                               p_MPaa = p_MPaa)
        
        self.rho_wat_kgm3 = upvt.unf_rho_water_bw_kgm3(gamma_w = self.gamma_wat,
                                                       bw_m3m3=self.bw_m3m3)

        self.mu_wat_cP = upvt.unf_mu_water_McCain_cP(t_K = t_K, 
                                                     p_MPaa = p_MPaa, 
                                                     s_ppm = self.salinity_ppm)

        #self._compr_wat_1bar = uc.compr_1mpa_2_1bar(upvt.unf_compressibility_brine_Spivey_1MPa(t_K, p_MPaa, self.s_ppm,
        #                                                                                      self._z, self.par_wat))
        
        #self._rsw_m3m3 = upvt.unf_gwr_brine_Spivey_m3m3(self.s_ppm, self._z)
        # определим термодинамические свойства воды
        #self._heatcap_wat_jkgc = upvt.unf_heat_capacity_water_IAPWS_JkgC(self.t_c)
        #self._thermal_conduct_wat_wmk = upvt.unf_thermal_conductivity_water_IAPWS_WmC(self.t_c)
        #self._thermal_expansion_wat_1c = upvt.unf_thermal_expansion_coefficient_water_IAPWS_1C(self.t_c)
    


    def calc(self, p_bar:float, t_C:float)->float:
        """
        расчет всех свойств флюида
        """
        self.p_atma = p_bar
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

        self.__calc_gas_props(p_bar, t_C)

        # свойства воды

        self.__calc_water_props(p_bar, t_C)

    #def _calc_compressibility_oil_1Mpa(self):
    #    return upvt.unf_compressibility_oil_VB_1Mpa(self._rs_m3m3, self.t_K, self.gamma_oil, self.pcal_MPaa, self.gamma_gas)

    #def _calc_b_oilb_m3m3(self):
    #    return upvt.unf_bo_saturated_Standing_m3m3(self.rsb_m3m3, self.gamma_gas, self.gamma_oil, self.t_K)

    """


        # определим свойства системы
        self._sigma_oil_gas_Nm = PVT.unf_surface_tension_go_Baker_Swerdloff_Nm(t_K, self.gamma_oil, p_MPaa)
        self._sigma_wat_gas_Nm = PVT.unf_surface_tension_gw_Sutton_Nm(self.rho_wat_kgm3, self.rho_gas_kgm3, self.t_c)
        return 1
    """
