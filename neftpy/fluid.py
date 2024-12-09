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
        self.fluid = Fluid_BlackOilStanding()
        
        self.q_liq_sm3day = 10
        self.fw_perc = 0
        self.rp_m3m3 = 100
        self.q_gas_free_sm3day = 0

    def calc(self, p_atma, t_C):
        self.fluid.calc(p_atma=p_atma, t_C=t_C)

    # ------------- обводненность ------------------
    @property 
    def fw_fr(self):
        """
        обводненность в долях единиц
        """
        return self.fw_perc / 100

    @fw_fr.setter
    def fw_fr(self, value:float):
        """
        установка значения обводненности
        """
        self.fw_perc = np.clip(value, 0, 1) * 100


    # ------------- дебит нефти --------------------
    @property
    def q_oil_sm3day(self):
        """
        дебит нефти приведенный к стандартным условиях
        """
        return self.q_liq_sm3day * (1 - self.fw_fr)

    @property
    def q_oil_m3day(self):
        """
        дебит нефти в рабочих условиях
        """
        return self.q_oil_sm3day * self.fluid._b_oil_m3m3

    @property
    def q_oil_kgday(self):
        """
        массовый расход нефти
        """
        return self.q_oil_m3day * self.fluid._rho_oil_kgm3

    # ------------ дебит газв ------------------------
    @property
    def q_gas_sm3day(self):
        """
        дебит газа приведенный к стандартным условиях
        учитывает весь газ потока - как свободный, так и растворенный в нефти
        """
        return self.q_oil_sm3day * self.rp_m3m3 + self.q_gas_free_sm3day  

    @property   
    def q_gas_insitu_sm3day(self):
        """ 
        расход свободного газа в заданных термобарических условиях приведенный к стандартным условиям
        без учета растворенного в нефти газа
        """
        _q_gas_insitu_sm3day = self.q_gas_sm3day - self.fluid._rs_m3m3 * self.q_oil_sm3day
        return np.where(_q_gas_insitu_sm3day < 0,
                        0,
                        _q_gas_insitu_sm3day )
 
    @property
    def q_gas_m3day(self):
        """
        расход свободного газа приведенный к расчетным термобарическим условиям
        """
        return self.q_gas_insitu_sm3day * self.fluid.b_gas_m3m3
    
    # ------------ дебит воды ------------------------
    @property
    def q_water_sm3day(self):
        """
        расход воды, приведенные к стандартным условиям
        """
        return self.q_liq_sm3day * self.fw_fr

    @property
    def q_water_m3day(self):
        """
        расход воды, приведенные к стандартным условиям
        """
        return self.q_water_sm3day * self.fluid._rho_water_kgm3

    # ------------ свойства жидкости (вода+нефть) ----------------------
    @property
    def q_liq_m3day(self):
        return self.q_oil_m3day + self.q_water_m3day
    
    @property
    def rho_liq_kgm3(self):
        """
        плотность жидкости (вода+нефть) в рабочих условиях
        """
        fw = self.fw_fr
        return self.fluid._rho_oil_kgm3 * (1-fw) + self.fluid.rho_water_kgm3 * fw
    
    @property
    def mu_liq_cP(self):
        """
        вязкость жидкости в рабочих условиях
        """
        fw = self.fw_fr
        return self.fluid.mu_oil_cP* (1-fw) + self.fluid.mu_water_cP * fw
  

    # ------------ дебит смести ------------------------
    @property
    def q_mix_m3day(self):
        return self.q_oil_m3day + self.q_water_m3day + self.q_gas_m3day
    
    @property
    def rho_mix_kgm3(self):
        """
        плотность смеси в рабочих условиях
        """
        return  1

    # ------------ доля газа в потоке ------------------------
    @property
    def gas_fraction(self):
        """
        объемная расходная доля газа в потоке
        """
        return self.q_gas_m3day / (self.q_gas_m3day + self.q_oil_m3day + self.q_water_m3day)



class Fluid_BlackOilStanding:
    """
    базовый класс задания флюидов (нефти, воды и газа) на основе модели нелетучей нефти (black oil)
    позволяет рассчитать все статические свойства флюидов необходимые для расчета
    статические свойства - которые могут быть определены в PVT лаборатории
    реализует фиксированный набор корреляционных зависимостей на основе
    корреляции Стенданга для давления насыщения и газосодержания
    поддерживает калибровку модели на измеренные значения давления насыщения,
    объемного коэффициента нефти при давлении насыщения и вязкости нефти при давлении насыщения. 
    """
    def __init__(self, 
                gamma_oil=0.86, 
                gamma_gas=0.6, 
                gamma_water=1.0, 
                rsb_m3m3=100.0,
                tb_C=80):
        """
        Создает флюид с заданными базовыми свойствами

        калибровочные параметры при необходимости надо задавать отдельно

        :param gamma_oil: specific gravity of oil
        :param gamma_gas: specific gravity of gas (by air), dimensionless
        :param gamma_wat: specific gravity of water
        :param rsb_m3m3: solution gas ratio at bubble point
        :param tb_C: reservoir temperature 
        """

        self._gamma_gas = gamma_gas         # удельная плотность газа, по воздуху
        self._gamma_oil = gamma_oil         # удельная плотность нефти по воде
        self._gamma_water = gamma_water     # удельная плотность воды по воде
        self._rsb_m3m3 = rsb_m3m3           # газосодержание при давлении насыщения
        self._tb_C = tb_C                   # температура давления насыщения (пластовая)

        # неуглеводородные примеси в газе
        self._y_h2s = 0.0
        self._y_co2 = 0.0
        self._y_n2 = 0.0

        # термобарические условия для которых рассчитаны свойства
        self._p_atma = uconst.P_SC_atma                 # thermobaric conditions for all parameters
        self._t_C = uconst.T_SC_C                  # can be set up by calc method

        # внутренние калибровочные значения
        # если меньше или равны 0 - калибровка не применяется
        self._pb_calibr_atma = 0.0
        self._b_oilb_calibr_m3m3 = 0.0
        self._mu_oilb_calibr_cP = 0.0

        # настройка схемы расчета 
        # авторасчет - любое изменение исходного параметра привет к пересчету всех параметров
        self.auto_calc = True
        # флаг состояния - свойства рассчитаны или нет
        self._calculated = False

    #
    # описание исходных параметров флюида
    #
    @property
    def gamma_gas(self):
        """
        удельная плотность газа, по воздуху
        """
        return self._gamma_gas
    
    @gamma_gas.setter
    def gamma_gas(self, value:float):
        self._gamma_gas = value
        self._calculated = False
        if self.auto_calc:
            self.calc(p_atma=self.p_atma, t_C=self.t_C)

    @property
    def gamma_oil(self):
        """
        удельная плотность нефти, по воде
        """
        return self._gamma_oil
    
    @gamma_oil.setter
    def gamma_oil(self, value:float):
        self._gamma_oil = value
        self._calculated = False
        if self.auto_calc:
            self.calc(p_atma=self.p_atma, t_C=self.t_C)

    @property
    def gamma_water(self):
        """
        удельная плотность воды, по воде
        """
        return self._gamma_water
    
    @gamma_water.setter
    def gamma_water(self, value:float):
        self._gamma_water = value
        self._calculated = False
        if self.auto_calc:
            self.calc(p_atma=self.p_atma, t_C=self.t_C)

    @property
    def rsb_m3m3(self):
        """
        газосодержание при давлении насыщения, м3/м3, исходный параметр
        """
        return self._rsb_m3m3
    
    @rsb_m3m3.setter
    def rsb_m3m3(self, value):
        self._rsb_m3m3 = value
        self._calculated = False
        if self.auto_calc:
            self.calc(p_atma=self.p_atma, t_C=self.t_C)

    @property
    def tb_C(self):
        return self._tb_C
    
    @tb_C.setter
    def tb_C(self, value):
        self._tb_C = value
        self._calculated = False
        if self.auto_calc:
            self.calc(p_atma=self.p_atma, t_C=self.t_C)

    #
    # калибровочные параметры 
    #

    @property
    def pb_atma(self):
        """
        давление насыщения, атма, при заданной температуре пласта tb_C
        """
        return self._pb_atma
    
    @pb_atma.setter
    def pb_atma(self, value):
        """
        попытка задать давление насыщения эквивалентна
        заданию калибровки по давлению насыщения при текущей температуре пласта
        """
        self._pb_calibr_atma = value
        self._calculated = False
        if self.auto_calc:
            self.calc(p_atma=self.p_atma, t_C=self.t_C)

    @property
    def mu_oilb_cP(self):
        """
        вязкость нефти при давлении насыщения и температуре давления насыщения, сП, только для чтения
        """
        return self._mu_oilb_cP
    
    
    def set_calibration(self,
                        pb_calibr_atma=-1., 
                        tb_calibr_C=80, 
                        b_oilb_calibr_m3m3=0, 
                        mu_oilb_calibr_cP=0
                        ):
        # установка калибровочных параметров  
        # как отдельный метод, чтобы в явном виде обратить внимание пользователя на 
        # механизм калибровки
        self._pb_calibr_atma = pb_calibr_atma
        self._tb_C = tb_calibr_C
        self._b_oilb_calibr_m3m3 = b_oilb_calibr_m3m3
        self._mu_oilb_calibr_cP = mu_oilb_calibr_cP
        
        self._calculated = False
        if self.auto_calc:
            self.calc(p_atma=self.p_atma, t_C=self.t_C)
    #
    # расчетные свойства флюида - только для чтения
    #

    @property
    def calculated(self):
        """
        флаг состояния экземпляра класса
        """
        return self._calculated

    @property
    def p_atma(self):
        """
        давление для которого расчитаны все свойства
        может быть задано только через методы calc
        """
        return self._p_atma
    
    @property
    def p_bara(self):
        """
        давление для которого расчитаны все свойства
        может быть задано только через методы calc
        """
        return uc.atm_2_bar(self._p_atma)
    
    @property
    def p_MPaa(self):
        """
        давление для которого расчитаны все свойства
        может быть задано только через методы calc
        """
        return uc.atm_2_MPa(self._p_atma)
    
    @property
    def t_C(self):
        """
        температура для которой были расчитаны все свойства
        может быть задана только через методы calc
        """
        return self._t_C
    
    @property
    def t_K(self):
        """
        температура для которой были расчитаны все свойства
        может быть задана только через методы calc
        """
        return uc.C_2_K(self._t_C)
    
    @property
    def pt_atma_C(self):
        """
        давление и температура как кортеж, только для чтения
        """
        return (self._p_atma, self._t_C)

    @property
    def b_oil_m3m3(self):
        """
        объемный коэффициент нефти при расчетных условиях, м3/м3, только для чтения
        """
        return self._b_oil_m3m3

    @property
    def b_gas_m3m3(self):
        """
        объемный коэффициент газа при расчетных условиях, м3/м3, только для чтения
        """
        return self._b_gas_m3m3
    
    @property
    def b_water_m3m3(self):
        """
        объемный коэффициент воды при расчетных условиях, м3/м3, только для чтения
        """
        return self._b_water_m3m3
    
    @property
    def rs_m3m3(self):
        """
        газосодержение в нефти при расчетных условиях, м3/м3, только для чтения
        """
        return self._rs_m3m3
    
    @property
    def pb_at_tcalc_atma(self):
        """
        давление насыщения, атма, при расчетной температуре t_C
        """
        return self._pb_at_tcalc_atma
    
    @property
    def rho_gas_kgm3(self):
        """
        плотность газа, кг/м3, при расчетных условиях, только для чтения
        """
        return self._rho_gas_kgm3

    @property
    def rho_oil_kgm3(self):
        """
        плотность нефти, кг/м3, при расчетных условиях, только для чтения
        """
        return self._rho_oil_kgm3
    
    @property
    def rho_water_kgm3(self):
        """
        плотность воды, кг/м3, при расчетных условиях, только для чтения
        """
        return self._rho_water_kgm3

    @property
    def mu_oil_cP(self):
        """
        вязкость нефти, сП, при расчетных условиях, только для чтения
        """
        return self._mu_oil_cP

    @property
    def mu_dead_oil_cP(self):
        """
        вязкость дегаризированной нефти, сП, при стандартных условиях, только для чтения
        """
        return self._mu_dead_oil_cP

    @property
    def mu_gas_cP(self):
        """
        вязкость газа, сП, только для чтения
        """
        return self._mu_gas_cP

    @property
    def mu_water_cP(self):
        """
        вязкость воды, сП, только для чтения
        """
        return self._mu_water_cP


    @property
    def p_gas_pc_MPa (self):
        """
        псевдо критическое давление газа, МПа, только для чтения
        """
        return self._p_gas_pc_MPa 

    @property
    def t_gas_pc_K (self):
        """
        псевдо критическая температура газа, К, только для чтения
        """
        return self._t_gas_pc_K 
    
    @property
    def p_gas_pr(self):
        """
        приведенное давление газа - отношение расчетного к псевдо-критическому, безразмерное, только для чтения
        """
        return self._p_gas_pr 
    
    @property
    def t_gas_pr (self):
        """
        приведенная температура газа - отношение расчетного к псевдо-критическому, безразмерное, только для чтения
        """
        return self._t_gas_pr 

    @property
    def salinity_water_ppm (self):
        """
        соленость воды, ppm
        """
        return self._salinity_ppm 

    #
    # расчетные функции
    #

    def set_gas_impurities(self,
                            y_h2s=0, 
                            y_co2=0,
                            y_n2=0  ):
        """
        :param y_h2s: mole fraction of the hydrogen sulfide
        :param y_co2: mole fraction of the carbon dioxide
        :param y_n2: mole fraction of the nitrogen
        """
        self._y_h2s = y_h2s
        self._y_co2 = y_co2
        self._y_n2 = y_n2

    def __calc_oil_props(self):
        """
        расчет свойств нефти по набору корреляций
        с калибровками параметров
        """
        t_res_K = uc.C_2_K(self._tb_C)
        # рассчитаем pb из rsb при температура пласта _tb_C
        pb_calc_MPaa = upvt.unf_pb_Standing_MPaa(rsb_m3m3 = self._rsb_m3m3, 
                                                       gamma_oil = self._gamma_oil, 
                                                       gamma_gas = self._gamma_gas, 
                                                       t_K = t_res_K)
        pb_calc_atma = uc.MPa_2_bar(pb_calc_MPaa)

        # найдем калибровочный коэффициент для давления насыщения
        p_cal_mult = np.divide(pb_calc_atma, self._pb_calibr_atma,
                               out=np.full_like(pb_calc_atma, 1.0),
                               where=self._pb_calibr_atma > 0)
                
        # оценим значение объемного коэффициента при давлении насыщения
        b_oilb_calc_m3m3 = upvt.unf_bo_saturated_Standing_m3m3(rs_m3m3 = self._rsb_m3m3, 
                                                               gamma_gas = self._gamma_gas, 
                                                               gamma_oil = self._gamma_oil, 
                                                               t_K = t_res_K)

        # проверим необходимость калибровки значения объемного коэффициента
        b_oil_cal_mult = np.where(self._b_oilb_calibr_m3m3 > 0, 
                                 (self._b_oilb_calibr_m3m3 - 1) / (b_oilb_calc_m3m3 - 1), 
                                 1.)
        
        # найдем эффективные параметры, с которыми надо дальше все считать 
        # эффективное давление может быть искажено при калибровке относительно реального
        # нужно только на время проведения расчета 
        p_effective_MPaa = uc.bar_2_MPa(self._p_atma * p_cal_mult)
        # эффективное давление насыщения - рассчитывается при фактической температуре
        pb_effective_MPaa = upvt.unf_pb_Standing_MPaa(rsb_m3m3 = self._rsb_m3m3, 
                                                      gamma_oil = self._gamma_oil, 
                                                      gamma_gas = self._gamma_gas, 
                                                      t_K = self.t_K)
        # сохраним давление насыщения
        self._pb_atma = pb_calc_atma / p_cal_mult
        self._pb_at_tcalc_atma = uc.MPa_2_atm(pb_effective_MPaa / p_cal_mult)
        # сохраним объемный коэффициент нефти при давлении насыщения
        self._b_oilb_m3m3 = b_oilb_calc_m3m3 / b_oil_cal_mult

        # найдем газосдержание используя эффективные значения давления и давления насыщения
        # при фактической температуре
        self._rs_m3m3 = upvt.unf_rs_Standing_m3m3(p_MPaa = p_effective_MPaa, 
                                                  pb_MPaa = pb_effective_MPaa, 
                                                  rsb_m3m3 = self._rsb_m3m3, 
                                                  gamma_oil = self._gamma_oil, 
                                                  gamma_gas = self._gamma_gas, 
                                                  t_K = self.t_K)
        
        # поскольку объемный коэффициент и сжимаемость при давлениях выше pb 
        # тесно связаны между собой - считаем их вместе
        co_1MPa = upvt.unf_compressibility_oil_VB_1Mpa(rs_m3m3 = self._rs_m3m3, 
                                                       t_K = self.t_K, 
                                                       gamma_oil = self._gamma_oil, 
                                                       p_MPaa = p_effective_MPaa, 
                                                       gamma_gas = self._gamma_gas)
        self._compr_oil_1bar = uc.compr_1mpa_2_1bar(co_1MPa)
        b_oil_m3m3 = np.where(p_effective_MPaa > pb_effective_MPaa, 
                              upvt.unf_bo_above_pb_m3m3(bob_m3m3 = b_oilb_calc_m3m3, 
                                                        compr_o_1MPa = co_1MPa, 
                                                        pb_MPaa = pb_effective_MPaa, 
                                                        p_MPaa = p_effective_MPaa),
                              upvt.unf_bo_saturated_Standing_m3m3(rs_m3m3 = self._rs_m3m3, 
                                                                  gamma_gas = self._gamma_gas, 
                                                                  gamma_oil = self._gamma_oil, 
                                                                  t_K = self.t_K)
                            )
        self._b_oil_m3m3 = 1 + b_oil_cal_mult * (b_oil_m3m3 - 1)
        
        self._rho_oil_kgm3 = upvt.unf_rho_oil_Standing_kgm3(p_MPaa = p_effective_MPaa , 
                                                            pb_MPaa = pb_effective_MPaa, 
                                                            co_1MPa = co_1MPa, 
                                                            rs_m3m3 = self._rs_m3m3, 
                                                            bo_m3m3 = self._b_oil_m3m3,
                                                            gamma_gas = self._gamma_gas, 
                                                            gamma_oil = self._gamma_oil) 
        

        # оценим значение вязкости
        self._mu_dead_oil_cP = upvt.unf_viscosity_deadoil_Beggs_cP(gamma_oil = self._gamma_oil, 
                                                                   t_K = self.t_K)
        self._mu_oilb_cP = upvt.unf_viscosity_saturatedoil_Beggs_cP(mu_oil_dead_cP = self._mu_dead_oil_cP, 
                                                                    rs_m3m3 = self._rsb_m3m3)
        self._mu_oil_cP = upvt.unf_viscosity_oil_Beggs_VB_cP(mu_oil_dead_cP = self._mu_dead_oil_cP, 
                                                             rs_m3m3 = self._rs_m3m3, 
                                                             p_MPaa = p_effective_MPaa, 
                                                             pb_MPaa = pb_effective_MPaa)
        if self._mu_oilb_calibr_cP > 0:
            mu_cal_mult = self._mu_oilb_calibr_cP / self._mu_oilb_cP
            self._mu_oil_cP = mu_cal_mult * self._mu_oil_cP
        

    def __calc_termodynamic_oil_props(self):
        # определим термодинамические свойства нефти
        self._heatcap_oil_jkgc = upvt.unf_heat_capacity_oil_Gambill_JkgC(gamma_oil = self._gamma_oil, 
                                                                         t_C = self._t_C)
        self._thermal_conduct_oil_wmk = upvt.unf_thermal_conductivity_oil_Cragoe_WmK(gamma_oil = self._gamma_oil, 
                                                                                     t_C = self._t_C)

        # определим термодинамические свойства нефти
        #self._heatcap_oil_jkgc = self._calc_heat_capacity_oil_JkgC()
        #self._thermal_conduct_oil_wmk = self._calc_thermal_conductivity_oil_WmK()

    def __calc_gas_props(self)->None:
        # расчет свойств газа
        # gas
        ppc_MPa, tpc_K = upvt.unf_pseudocritical_McCain_p_MPa_t_K(gamma_gas = self._gamma_gas, 
                                                                  y_h2s = self._y_h2s, 
                                                                  y_co2 = self._y_co2, 
                                                                  y_n2 = self._y_n2)
  
        ppr = self.p_MPaa / ppc_MPa
        tpr = self.t_K / tpc_K
        self._p_gas_pc_MPa = ppc_MPa
        self._t_gas_pc_K = tpc_K
        self._p_gas_pr = ppr
        self._t_gas_pr = tpr

        self._z = upvt.unf_zfactor_SK(ppr=ppr, tpr=tpr, safe=True)
        self._rho_gas_kgm3 = upvt.unf_rho_gas_z_kgm3(t_K = self.t_K, 
                                                     p_MPaa = self.p_MPaa, 
                                                     gamma_gas = self._gamma_gas, 
                                                     z =self._z)

        self._mu_gas_cP = upvt.unf_mu_gas_Lee_rho_cP(t_K = self.t_K, 
                                                     gamma_gas = self._gamma_gas, 
                                                     rho_gas_kgm3 = self._rho_gas_kgm3)
        self._b_gas_m3m3 = upvt.unf_b_gas_z_m3m3(t_K = self.t_K, 
                                               p_MPaa = self.p_MPaa,
                                               z =self._z)
        #
        #
        #  TODO сжимаемость надо будет переделать чтобы тут расчитывалась через производные z 
        # 
        #self._compr_gas_1bar = uc.compr_1mpa_2_1bar(PVT.unf_compressibility_gas_Mattar_1MPa(p_MPaa, t_K,
        #                                                                                    ppc_MPa, tpc_K))

        # определим термодинамические свойства газа
        self._heatcap_gas_jkgc = upvt.unf_heat_capacity_gas_Mahmood_Moshfeghian_JkgC(p_MPaa = self.p_MPaa, 
                                                                                     t_K = self.t_K, 
                                                                                     gamma_gas = self._gamma_gas)
        self._thermal_conduct_gas_wmk = upvt.unf_thermal_conductivity_gas_methane_WmK(t_C = self.t_C)

    def __calc_water_props(self)->None:
    
        # water
        self._salinity_ppm = upvt.unf_salinity_from_gamma_water_ppm(gamma_water = self._gamma_water)

        self._b_water_m3m3 = upvt.unf_b_water_McCain_m3m3(t_K = self.t_K, 
                                               p_MPaa = self.p_MPaa)
        
        self._rho_water_kgm3 = upvt.unf_rho_water_bw_kgm3(gamma_w = self._gamma_water,
                                                       bw_m3m3=self._b_water_m3m3)

        self._mu_water_cP = upvt.unf_mu_water_McCain_cP(t_K = self.t_K, 
                                                     p_MPaa = self.p_MPaa, 
                                                     s_ppm = self._salinity_ppm)

        #self._compr_wat_1bar = uc.compr_1mpa_2_1bar(upvt.unf_compressibility_brine_Spivey_1MPa(t_K, p_MPaa, self.s_ppm,
        #                                                                                      self._z, self.par_wat))
        
        #self._rsw_m3m3 = upvt.unf_gwr_brine_Spivey_m3m3(self.s_ppm, self._z)
        # определим термодинамические свойства воды
        #self._heatcap_wat_jkgc = upvt.unf_heat_capacity_water_IAPWS_JkgC(self.t_c)
        #self._thermal_conduct_wat_wmk = upvt.unf_thermal_conductivity_water_IAPWS_WmC(self.t_c)
        #self._thermal_expansion_wat_1c = upvt.unf_thermal_expansion_coefficient_water_IAPWS_1C(self.t_c)
    


    def calc(self, p_atma:float, t_C:float)->None:
        """
        расчет всех свойств флюида
        """
        self._p_atma = p_atma
        self._t_C = t_C 

        # оценим свойства нефти 
        self.__calc_oil_props()       

        self.__calc_termodynamic_oil_props()

        # свойства газа 

        self.__calc_gas_props()

        # свойства воды

        self.__calc_water_props()

        self._calculated = True

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
