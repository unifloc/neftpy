
import numpy as np 

 # модули для проверки типов
import numpy.typing as npt
from numpy.typing import ArrayLike
from typing import Union, Callable

# базовый тип для расчетов с массивами
FloatArray = Union[float, np.ndarray]

import neftpy.fluid as fl
import neftpy.curves as crv

import scipy.constants as const
import scipy


# ===========================================================
# Базовый класс для расчета трубы - давление и температура
# ===========================================================

class CAmbientFormation:
    """ 
    Класс для описания теплофизических характеристик породы в которой
    проходят трубы
    Задает распределение температуры по глубине
    и базовые свойства
    """
    def __init__(self):
        self.therm_cond_form_WmC = 2.4252    # теплопроводность породы Дж/сек/м/С
        self.sp_heat_capacity_form_JkgC = 200          # теплоемкость породы
        self.density_formation_kgm3 = 4000                  # плотность породы вокруг скважины
        
        self.therm_cond_cement_WmC = 6.965       # теплопроводность цемента вокруг скважины
        self.therm_cond_tubing_WmC = 32          # теплопроводность металла НКТ
        self.therm_cond_casing_WmC = 32          # теплопроводность металла эксплуатационной колонны

        # convective heat transfer coeficients
        self.heat_transfer_casing_liquid_Wm2C = 200        # конвективная теплопередача через затруб с жидкостью  Дж/м2/сек/С
        self.heat_transfer_casing_gas_Wm2C = 10            # теплопередача через затруб с газом (радиационная)
        self.heat_transfer_fluid_convection_Wm2C = 200     # теплопередача конвективная в потоке жидкости

        # радиусы для проведения расчета
        self.rti_m = 0.06           # НКТ внутренний
        self.rto_m = self.rti_m + 0.01 # НКТ наружный
        self.rci_m = 0.124          # Эксп колонна внутренний
        self.rco_m = self.rci_m + 0.01 # Эксп колонна наружный
        self.rcem_m = 0.3           # Радиус цементного кольца вокруг скважины
        self.rwb_m = 0.3
        
        # исходные данные по умолчанию, чтобы все считало
        self.h_vert_data_m = 2500
        self.reservoir_temp_data_C = 95
        self.surf_temp_data_C = 25

        self.t_calc_hr = 10 * 24   # задаем по умолчанию время расчета распределения температуры через 10 дней

        self.h_dyn_m = -1
        self.h_pump_m = -1
        
        self.amb_temp_curve = crv.Curve()

        self.amb_temp_curve.add_point(0, self.surf_temp_data_C)
        self.amb_temp_curve.add_point(self.h_vert_data_m, self.reservoir_temp_data_C)
        
        self.TGeoGrad_C100m_ = self.h_vert_data_m / 100 / (self.reservoir_temp_data_C - self.surf_temp_data_C)

    def amb_temp_grad_Cm(self, h_vert_m:float)->float:
        """
        выдает геотермальный градиент на заданной вертикальной глубине
        """
        return self.amb_temp_curve.get_grad(h_vert_m)

    def amb_temp_C(self, h_vert_m:float)->float:
        """
        выдает температуру на заданной вертикальной глубине
        """
        return self.amb_temp_curve.get_point(h_vert_m)
    
    @property
    def td_d(self):
        return self.therm_cond_form_WmC * self.t_calc_hr * 3600 / self.density_formation_kgm3 / self.sp_heat_capacity_form_JkgC / (self.rwb_m ** 2)

    @property
    def td(self):
        return np.log(np.exp(-0.2 * self.td_d) + (1.5 - 0.3719 * np.exp(-self.td_d)) * (self.td_d ** 0.5))    
    
    def Lr_1m(self, wt_kgsec:float, Uto_Jm2secC:float, Cp_JkgC:float):
        if wt_kgsec != 0:
            return 2 * np.pi / (Cp_JkgC * wt_kgsec) * (Uto_Jm2secC * self.therm_cond_form_WmC / (self.therm_cond_form_WmC + Uto_Jm2secC * self.td))
        else:
            return 10000

    @property
    def Uto_cas_Jm2secC(self):
        Uto_cas_Jm2secC = 1 / (np.log(self.rwb_m / self.rco_m) / self.therm_cond_cement_WmC + 
                               np.log(self.rco_m / self.rci_m) / self.therm_cond_casing_WmC + 
                                  1 / self.rci_m / self.heat_transfer_fluid_convection_Wm2C)
    
    @property
    def Uto_tub_liqcas_Jm2secC(self):
        Uto_tub_liqcas_Jm2secC = 1 / ( 
                                    1 * np.log(self.rwb_m / self.rco_m) / self.therm_cond_cement_WmC + 
                                    1 * np.log(self.rco_m / self.rci_m) / self.therm_cond_casing_WmC + 
                                    1 / self.rto_m / (self.heat_transfer_casing_gas_Wm2C + self.heat_transfer_casing_liquid_Wm2C) + 
                                    1 * np.log(self.rto_m / self.rti_m) / self.therm_cond_tubing_WmC + 
                                    1 / self.rti_m / self.heat_transfer_fluid_convection_Wm2C 
                                )
    @property
    def Uto_tub_gascas_Jm2secC(self):
        Uto_tub_gascas_Jm2secC = 1 / ( 
                                    1 * np.log(self.rwb_m / self.rco_m) / self.therm_cond_cement_WmC + 
                                    1 * np.log(self.rco_m / self.rci_m) / self.therm_cond_casing_WmC + 
                                    1 / self.rto_m / (self.heat_transfer_casing_gas_Wm2C) + 
                                    1 * np.log(self.rto_m / self.rti_m) / self.therm_cond_tubing_WmC + 
                                    1 / self.rti_m / self.heat_transfer_fluid_convection_Wm2C 
                                )

    def calc_dtdl_Cm(self, 
                     h_vert_m:float, 
                     sinTheta_deg:float, 
                     T1_C:float, 
                     w_kgsec:float, 
                     Cp_JkgC:float,
                     dPdL_atmm:float = 0,
                     v_ms:float = 0,
                     dvdL_msm:float = 0,
                     Cj_Catm:float = 0,
                     flow_along_coord:bool=True):
        """ 
        ' h_vert_m     -  vertical depth where calculation take place
        ' sinTheta_deg - angle sin
        ' T1_C         - fluid temp at depth gien
        ' W_kgsec      - mass rate of fluid
        ' Cp_JkgC      - heat capasity
        ' dPdL_atmm    - pressure gradient at depth given (needed to account Joule Tompson effect)
        ' v_ms         - velocity of fluid mixture
        ' dvdL_msm     - acceleration of fluid mixture. acount inetria force influence (should be small but ..)
        ' Cj_Catm      - коэффициент Джоуля Томсона Joule Thomson coeficient
        ' flowUp       - flow direction
        """

        if w_kgsec == 0:
            return self.amb_temp_grad_Cm(h_vert_m)
                
        #' set Uto - temperature emission depents on well condition
        if h_vert_m > self.h_pump_m:
            Uto = self.Uto_cas_Jm2secC
        elif h_vert_m > self.h_dyn_m:
            Uto = self.Uto_tub_liqcas_Jm2secC
        else:
            Uto = self.Uto_tub_gascas_Jm2secC
            
        if flow_along_coord:
            sign = -1
        else:
            sign = 1

        Lr = self.Lr_1m(w_kgsec, Uto, Cp_JkgC)
        calc_dtdl_Cm = sign * (T1_C - self.amb_temp_C(h_vert_m)) * Lr
        calc_dtdl_Cm = calc_dtdl_Cm - (const.g * sinTheta_deg / Cp_JkgC + v_ms / Cp_JkgC * dvdL_msm - Cj_Catm * dPdL_atmm)


class Pipe:
    """ 
    Труба - базовый класс для гидравлических расчетов
    Позволяет найти распределение давления и температуры в трубе
    В трубе могут меняться траектория, диаметры
    В трубе постоянный поток флюидов (feed)
    """
    def __init__(self):
        """ 
        Простая инициализация трубы
        """
        # поток флюидов
        self.feed = fl.Feed()
        self.ambient_formation_ = CAmbientFormation()
                
        self.p_wh_atma = 20         # давление на устье, бар
        self.t_wh_C = 20          # температура на устье скважины, С
        self.dT_dz = 0.03      # температурный градиент град С на 100 м
        self.h_well = 2000          # измеренная глубина забоя скважины
        self.d_tub_m = 0.089      # диаметр НКТ по которой ведется закачка
        self.roughness_m = 0.01   # шероховатость
        
        # траектория скважины, задается как массив измеренных глубин и значений отклонения от вертикали
        self.trajectory_h_mes_m = np.array([0,50,100,200,800,1300,1800,2200,2500])
        self.trajectory_h_vert_m = np.array([0,50,100,200,780,1160,1450,1500,1500])

        self.diam_h_mes_m = np.array([0,1000])
        self.diam_d_internal_m = np.array([0.062, 0.062])
        self.diam_by_h_mes_m = scipy.interpolate.interp1d(self.diam_h_mes_m, self.diam_d_internal_m, kind='previous')

    def d_int_m(self, h_mes_m):
        return self.diam_by_h_mes_m(h_mes_m)
    
    def theta_rad(self, h_mes_m):
        #return 90
        # построим массив углов отклонения от вертикали
        ang =np.arccos(np.diff(self.trajectory_h_vert_m)/np.diff(self.trajectory_h_mes_m))
        theta = np.interp(h_mes_m, self.trajectory_h_mes_m, ang)
        return theta
    
    def calc_grad(self, h_mes_m:float, p_atma:float, t_C:float)->float:
        """ 
        расчет градиента давления на заданной глубине
        """

        self.feed.calc(p_atma, t_C)

        diam_internal_m = self.d_int_m(h_mes_m)
        theta_rad = self.theta_rad(h_mes_m)
        roughness_m = self.roughness_m
        q_liq_rc_m3day = self.feed.q_liq_m3day
        q_gas_rc_m3day = self.feed.q_gas_m3day
        mu_liq_rc_cP = self.feed.mu_liq_cP
        mu_gas_rc_cP = self.feed.fluid.mu_gas_cP
        # sigma_l_Nm
        return p_atma

    
        #, 
        #                theta_deg,
        #                roughness_m , 
        #                q_liq_rc_m3day,
        #                q_gas_rc_m3day,
        #                mu_liq_rc_cP,
        #               mu_gas_rc_cP,
        #                sigma_l_Nm,
        #                rho_liq_rc_kgm3,
        #                rho_gas_rc_kgm3,
        #                Payne_et_all_holdup = 0,
        #                Payne_et_all_friction = 1, 
        #                calibr_grav = 1,
        #                calibr_fric = 1
    

# ===========================================================
# Расчет коэффиуциента трения в трубе
# ===========================================================

def _npy_friction_factor_Zigrang_Sylvester(n_Re: FloatArray, 
                                           ed:float) -> np.ndarray:
    """ 
    calculate friction factor for rough pipes 
    Zigrang and Sylvester  1982  https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae
    
    n_Re - Reinolds number
    ed - pipe relative roughness
    """
    
    f_n = (2 * np.log10(2 / 3.7 * ed - 5.02 / n_Re * np.log10(2 / 3.7 * ed + 13 / n_Re)))**-2
    return f_n

def _npy_friction_factor_Zigrang_Sylvester_refined(n_Re:FloatArray, 
                                                   ed:float)-> np.ndarray:
    
    f_n = _npy_friction_factor_Zigrang_Sylvester(n_Re, ed)
    i = 0
    while True: 
        # iterate until error in friction factor is sufficiently small
        # https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae
        # expanded form  of the Colebrook equation
        f_n_new = (1.7384 - 2 * np.log10(2 * ed + 18.574 / (n_Re * f_n ** 0.5))) ** -2
        i += 1
        f_int = f_n
        f_n = f_n_new
        #stop when error is sufficiently small or max number of iterations exceedied
        if (np.abs(f_n_new - f_int).max() <= 0.001 or i > 19):
            break
    return f_n

def _npy_friction_factor_Brkic(n_Re: FloatArray, 
                               ed:float) -> np.ndarray:
    """  
    Brkic shows one approximation of the Colebrook equation based on the Lambert W-function
    Brkic, Dejan (2011). "An Explicit Approximation of Colebrook#s equation for fluid flow friction factor" (PDF). 
    Petroleum Science and Technology. 29 (15): 1596–1602. doi:10.1080/10916461003620453
    http://hal.archives-ouvertes.fr/hal-01586167/file/article.pdf
    https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae
    http://www.math.bas.bg/infres/MathBalk/MB-26/MB-26-285-292.pdf
    """
    Svar = np.log(n_Re / (1.816 * np.log(1.1 * n_Re / (np.log(1 + 1.1 * n_Re)))))
    f_1 = -2 * np.log10(ed / 3.71 + 2 * Svar / n_Re)
    f_n = 1 / (f_1 ** 2)
    return f_n

def _npy_friction_factor_Drew(n_Re: FloatArray, 
                               ed:float) -> np.ndarray:
    #Calculate friction factor for smooth pipes using Drew correlation - original Begs&Brill with no modification
    return  0.0056 + 0.5 * n_Re ** -0.32


def _npy_friction_factor_Haaland(n_Re: FloatArray, 
                                 ed:float) -> np.ndarray:

    # from unified TUFFP model
    # Haaland equation   Haaland, SE (1983). "Simple and Explicit Formulas for the Friction Factor in Turbulent Flow". 
    # Journal of Fluids Engineering. 105 (1): 89–90. doi:10.1115/1.3240948

    f_n = 4 / (3.6 * np.log10(6.9 / n_Re + (ed / 3.7) ** 1.11)) ** 2
    return f_n


def npy_friction_factor(n_Re:FloatArray, 
                        ed:float,
                        friction_corr:str="Brkic",
                        smooth_transition:bool=True) -> np.ndarray:

    # Define the correlation functions
    friction_func_options = {
        "Zigrang_Sylvester": _npy_friction_factor_Zigrang_Sylvester,
        "Zigrang_Sylvester_refined": _npy_friction_factor_Zigrang_Sylvester_refined,
        "Haaland": _npy_friction_factor_Haaland,
        "Brkic": _npy_friction_factor_Brkic,
        "Drew": _npy_friction_factor_Drew
    }    
    _friction_func = friction_func_options[friction_corr]
    # Define the transition limits
    lower_Re_lim = 2000.0
    upper_Re_lim = 4000.0
    # Ensure n_Re are numpy arrays
    n_Re = np.asarray(n_Re)

    laminar_mask = (n_Re <= lower_Re_lim) & (n_Re > 0)
    turbulent_mask = n_Re > lower_Re_lim
    transition_mask =  (n_Re <= upper_Re_lim) & (n_Re > lower_Re_lim)

    # Initialize the friction factor array
    f_n = np.zeros_like(n_Re, dtype=float)
    f_n[laminar_mask] = 64 / n_Re[laminar_mask]

    # Calculate friction factor using the selected correlation
    f_n[turbulent_mask] = _friction_func(n_Re[turbulent_mask], ed)

    if smooth_transition:
        # apply smooth transition
        mult_n_Re = np.array([0, lower_Re_lim, upper_Re_lim, upper_Re_lim * 10])
        mult_val = np.array([0, 0, 1, 1])
        f_n[transition_mask] = (_friction_func(n_Re[transition_mask], ed) * 
                               np.interp(n_Re[transition_mask], mult_n_Re, mult_val ) + 
                               64 / lower_Re_lim * 
                               (1 -  np.interp(n_Re[transition_mask], mult_n_Re, mult_val )))
    return f_n


# ====================================================================================
# корреляция Беггса Брилла
# ====================================================================================


def hl_arr_theta_deg(fl_pat:int, 
                     lambda_l:float, 
                     n_fr:float, 
                     n_lv:float, 
                     arr_theta_rad:float, 
                     Payne_et_all):
    """ 
    function calculating liquid holdup for Beggs Brill gradient function
    fl_pat - flow pattern (0 -Segregated, 1 - Intermittent, 2 - Distributed)
    lambda_l - volume fraction of liquid at no-slip conditions
    n_fr - Froude number
    n_lv - liquid velocity number
    arr_theta_deg - pipe inclination angle, (Degrees)
    payne_et_all - flag indicationg weather to applied Payne et all correction for holdup (0 - not applied, 1 - applied)
    Constants to determine liquid holdup

    """
   
    #constants to determine liquid holdup correction
    a = (0.98,   0.845,  1.065)    # flow pattern - segregated
    b = (0.4846, 0.5351, 0.5824)    # flow pattern - intermittent
    c = (0.0868, 0.0173, 0.0609)    # flow pattern - distributed
    e = (0.011,  2.96,   1.,   4.700)    # flow pattern - segregated uphill
    f = (-3.768, 0.305,  0,   -0.3692)    # flow pattern - segregated
    g = (3.539, -0.4473, 0,    0.1244)    # flow pattern - segregated
    h = (-1.614, 0.0978, 0,   -0.5056)    # flow pattern - segregated
           
    h_l_0 = a[fl_pat] * lambda_l**b[fl_pat] / n_fr**c[fl_pat] #calculate liquid holdup at no slip conditions
    #arr_theta_rad = np.pi / 180 * arr_theta_rad #convert angle to radians
    if np.sin(arr_theta_rad) < 0:
        fl_pat = 3
        
    CC = np.maximum(0, 
                    (1 - lambda_l)**np.log(e[fl_pat] * lambda_l**f[fl_pat] * n_lv**g[fl_pat] * n_fr**h[fl_pat])) #calculate correction for inclination angle
    
    
    psi = 1 + CC * (np.sin(1.8 * arr_theta_rad) - 0.333 * (np.sin(1.8 * arr_theta_rad)) ** 3)  

    #calculate liquid holdup with payne et al. correction factor
    Payne_corr = 1.
    if Payne_et_all > 0:
        if arr_theta_rad > 0: #uphill flow
            Payne_corr = 0.924
        else:                 #downhill flow
            Payne_corr = 0.685
    return np.maximum(np.minimum(1, Payne_corr * h_l_0 * psi), lambda_l)


def npy_Begs_Brill_gradient(diam_internal_m, 
                          theta_rad,
                          roughness_m , 
                          q_liq_rc_m3day,
                          q_gas_rc_m3day,
                          mu_liq_rc_cP,
                          mu_gas_rc_cP,
                          sigma_l_Nm,
                          rho_liq_rc_kgm3,
                          rho_gas_rc_kgm3,
                          Payne_et_all_holdup = 0,
                          Payne_et_all_friction = 1, 
                          calibr_grav = 1,
                          calibr_fric = 1):
    """ 
    function for calculation of pressure gradient in pipe according to Begs and Brill method
    Return (atma/m)

    Arguments
    diam_internal_m - pipe internal diameter (m)
    theta_rad - pipe inclination angel (radians)
    roughness_m - pipe wall roughness (m)
    q_liq_rc_m3day - liquid rate in situ conditions (m3/day)
    q_gas_rc_m3day - gas rate in situ conditions (m3/day)
    mu_liq_rc_cP - oil viscosity at reference pressure (cP)
    mu_gas_rc_cP - gas viscosity at reference pressure (cP)
    sigma_l_Nm - oil-gAs surface tension coefficient (Newton/m)
    rho_liq_rc_kgm3 - oil density in situ conditions (kg/m3)
    rho_gas_rc_kgm3 - gas density in situ conditions (kg/m3)

    Payne_et_all_holdup - flag indicationg weather to applied Payne et all correction and holdup (0 - not applied, 1 - applied)
    Payne_et_all_friction - flag indicationg weather to apply Payne et all correction for friction (0 - not applied, 1 - applied)  obsolete

    calibr_grav
    calibr_fric
    """    
    c_p = 0.000009871668   # переводной коэффициент
    const_convert_sec_day = 1 / 86400
    const_g = 9.81

    ap_m2 = np.pi * diam_internal_m ** 2 / 4
    
    lambda_l = q_liq_rc_m3day / (q_liq_rc_m3day + q_gas_rc_m3day)
    roughness_d = roughness_m / diam_internal_m
    
    vsl_msec = const_convert_sec_day * q_liq_rc_m3day / ap_m2
    vsg_msec = const_convert_sec_day * q_gas_rc_m3day / ap_m2
    vsm_msec = vsl_msec + vsg_msec
    
    rho_n_kgm3 = rho_liq_rc_kgm3 * lambda_l + rho_gas_rc_kgm3 * (1 - lambda_l) # No-slip mixture density
    mu_n_cP = mu_liq_rc_cP * lambda_l + mu_gas_rc_cP * (1 - lambda_l) # No slip mixture viscosity
    
    n_re = 1000 * rho_n_kgm3 * vsm_msec * diam_internal_m / mu_n_cP
    
    n_fr = vsm_msec ** 2 / (const_g * diam_internal_m)
    n_lv = vsl_msec * (rho_liq_rc_kgm3 / (const_g * sigma_l_Nm)) ** 0.25

    #-----------------------------------------------------------------------
    #determine flow pattern
    if (n_fr >= 316 * lambda_l ** 0.302 or n_fr >= 0.5 * lambda_l ** -6.738): 
        flow_pattern = 2
    else:
        if (n_fr <= 0.000925 * lambda_l ** -2.468): 
            flow_pattern = 0
        else:
            if (n_fr <= 0.1 * lambda_l ** -1.452):
                flow_pattern = 3
            else:
                flow_pattern = 1
    #-----------------------------------------------------------------------
    #determine liquid holdup
    if (flow_pattern == 0 or flow_pattern == 1 or flow_pattern == 2):
        hl_out_fr = hl_arr_theta_deg(flow_pattern, lambda_l, n_fr, n_lv, theta_rad, Payne_et_all_holdup)
    else:
        l_2 = 0.000925 * lambda_l ** -2.468
        l_3 = 0.1 * lambda_l ** -1.452
        aa = (l_3 - n_fr) / (l_3 - l_2)
        hl_out_fr = (aa * hl_arr_theta_deg(0, lambda_l, n_fr, n_lv, theta_rad, Payne_et_all_holdup) + 
                  (1 - aa) * hl_arr_theta_deg(1, lambda_l, n_fr, n_lv, theta_rad, Payne_et_all_holdup))

    # Calculate normalized friction factor
    f_n = npy_friction_factor(n_re, roughness_d)
    # calculate friction factor correction for multiphase flow
    y = np.maximum(lambda_l / hl_out_fr ** 2, 0.000001)
    if (y > 1 and y < 1.2):
        s = np.log(2.2 * y - 1.2)
    else:
        FY = np.maximum(np.log(y), 0.000001)
        s = FY / (-0.0523 + 3.182 * FY - 0.8725 * FY ** 2 + 0.01853 * FY ** 4)
    #calculate friction factor
    f = f_n * np.exp(s)
    
    
    rho_s = rho_liq_rc_kgm3 * hl_out_fr + rho_gas_rc_kgm3 * (1 - hl_out_fr) #calculate mixture density
    dPdLg_out_atmm = c_p * rho_s * const_g * np.sin(theta_rad) #calculate pressure gradient due to gravity
    dPdLf_out_atmm = c_p * f * rho_n_kgm3 * vsm_msec ** 2 / (2 * diam_internal_m)  #calculate pressure gradient due to friction
    dPdLa_out_atmm = 0  #calculate pressure gradient # not acounted in BeggsBrill
    fpat_out_num = flow_pattern
    
    #return  dPdLg_out_atmm * calibr_grav + dPdLf_out_atmm * calibr_fric
    return  (dPdLg_out_atmm * calibr_grav + dPdLf_out_atmm * calibr_fric, 
            dPdLg_out_atmm * calibr_grav, 
            dPdLf_out_atmm * calibr_fric, 
            dPdLa_out_atmm, 
            vsl_msec, 
            vsg_msec, 
            hl_out_fr, 
            fpat_out_num)
    