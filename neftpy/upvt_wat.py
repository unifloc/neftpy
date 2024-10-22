from neftpy.uconvert import *
from neftpy.uconst import *

import numpy as np
import scipy.optimize as opt
#import scipy.optimize as sp  # модуль для решения уравения



# uPVT свойства для воды

def unf_bw_McCain_m3m3(t_K:float, p_MPaa:float)->float:
    """
        FVF of brine by McCain

        https://petrowiki.org/Produced_water_formation_volume_factor

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :return: formation volume factor, m3/m3
    """
    t_f = K_2_F(t_K)
    p_psi = bar_2_psi(p_MPaa * 10)
    dvwp = -1.95301e-9 * p_psi * t_f - 1.72834e-13 * p_psi ** 2 * t_f - 3.58922e-7 * p_psi - 2.25341e-10 * p_psi ** 2
    dvwt = -1.0001e-2 + 1.33391e-4 * t_f + 5.50654e-7 * t_f ** 2
    return  (1 + dvwp) * (1 + dvwt)

def unf_rho_water_bw_kgm3(gamma_w:float, 
                          bw_m3m3:float
                          )->float:
    """
        Equation from UniflocVBA

    :param gamma_w:
    :param bw_m3m3:
    :return:
    """
    rho_wat_rc_kgm3 = 1000 * gamma_w / bw_m3m3
    return rho_wat_rc_kgm3

def unf_mu_water_McCain_cP(t_K:float, p_MPaa:float, s_ppm:float)->float:  
    """
        McCain correlation for brine(water) viscosity

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :param s_ppm: salinity, ppm
    :return: viscosity, cP

    ref 1 McCain, W.D. Jr.: McCain, W.D. Jr. 1990. The Properties of Petroleum Fluids, second edition. Tulsa,
    Oklahoma: PennWell Books.

    ref 2 https://petrowiki.org/Produced_water_properties#cite_note-r1-1
    """

    wpTDS = s_ppm / 10000

    a = 109.574 - 8.40564 * wpTDS + 0.313314 * wpTDS ** 2 + 0.00872213 * wpTDS ** 3
    b = -1.12166 + 0.0263951 * wpTDS - 0.000679461 * wpTDS ** 2 - 5.47119e-5 * wpTDS ** 3 + 1.55586e-6 * wpTDS ** 4

    visc = a * K_2_F(t_K) ** b
    p_psi = MPa_2_psi(p_MPaa)
    return visc * (0.9994 + 4.0295e-5 * p_psi + 3.1062e-9 * p_psi ** 2)

def unf_gamma_water_from_salinity_m3m3(salinity_ppm: float) -> float:
    """
    Water density at standard conditions
    """
    wpTDS = salinity_ppm / 10000
    return 0.0160185 * (62.368 + 0.438603 * wpTDS + 0.00160074 * wpTDS ** 2)


def unf_salinity_from_gamma_water_ppm(gamma_water: float) -> float:
    """
    Salinity in ppm from water density at standard conditions (inverse of unf_gamma_water_from_salinity_m3m3)
    """
    result = (624.711071129603 * gamma_water / 0.0160185 - 20192.9595437054) ** 0.5 - 137.000074965329
    return result * 10000


# =========================================================
# следующие функции требует проверки и разбирательств
# =========================================================



def unf_density_brine_Spivey_kgm3(t_K:float, 
                                  p_MPaa:float, 
                                  s_ppm:float, 
                                  par:int=1
                                  )->float:
    """
        Modified Spivey et al. correlation for brine(water) density (2009)

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :param s_ppm: salinity, ppm
    :param par: parameter, 0 - methane-free brine, 1 - brine containing dissolved methane
    :return: density, kg/m3

        корреляция позволяет найти плотность соленой воды с растворенным в ней метаном

        ref 1 Spivey, J.P., McCain, W.D., Jr., and North, R. “Estimating Density, Formation
    Volume Factor, Compressibility, Methane Solubility, and Viscosity for Oilfield
    Brines at Temperatures From 0 to 275°C, Pressures to 200 MPa, and Salinities to
    5.7 mole/kg.” Journal of Canadian Petroleum Technology. Vol. 43, No. 7 (July 2004)
    52–61.

    ref 2 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011

    """

    t_C = K_2_C(t_K)
    s = s_ppm / 1000000
    m = 1000 * s / (58.4428 * (1 - s))
    # Первым шагом вычисляется плотность чистой воды при давлении 70 MPa и температуре
    ro_w70 = (-0.127213 * (t_C/100)**2 + 0.645486 * (t_C/100) + 1.03265)/(-0.070291 * (t_C/100)**2 +
                                                                          0.639589 * (t_C/100) + 1)
    # Температурные коэффициенты сжимаемости чистой воды
    ew = (4.221 * (t_C / 100)**2 - 3.478 * (t_C/100) + 6.221)/(0.5182 * (t_C/100)**2 - 0.4405 * (t_C/100) + 1)
    fw = (-11.403 * (t_C / 100) ** 2 + 29.932 * (t_C / 100) + 27.952) / (0.20684 * (t_C / 100) ** 2 +
                                                                         0.3768 * (t_C / 100) + 1)
    iw70 = np.log(abs(ew + fw))/ew
    iw = np.log(abs(ew*(p_MPaa/70) + fw))/ew
    # Плотность чистой воды при T, P
    ro_w = ro_w70 * np.exp(iw - iw70)
    # Температурные коэффициенты плотности раствора
    d_m1 = -1.1149e-4 * (t_C/100)**2 + 1.7105e-4 * (t_C/100) - 4.3766e-4
    d_m2 = (-8.878e-4 * (t_C/100)**2 - 1.388e-4 - 2.96318e-3)/(0.51103 * (t_C / 100) + 1)
    d_m3 = (2.1466e-3 * (t_C / 100) ** 2 + 1.2427e-2 * (t_C / 100) + 4.2648e-2) / \
           (-8.1009e-2 * (t_C / 100) ** 2 + 0.525417 * (t_C / 100) + 1)
    d_m4 = 2.356e-4 * (t_C/100)**2 - 3.636e-4 * (t_C/100) - 2.278e-4
    # Плотность раствора воды с хлоридом натрия при давлении 70 MPa и температуре
    ro_b70 = ro_w70 + d_m1 * m ** 2 + d_m2 * m ** 1.5 + d_m3 * m + d_m4 * m ** 0.5
    # Температурные коэффициенты сжимаемости раствора
    eb = ew + 0.1249
    f_m1 = (-0.617 * (t_C / 100) ** 2 - 0.747 * (t_C / 100) - 0.4339) / (10.26 * (t_C / 100) + 1)
    f_m2 = (9.917 * (t_C / 100) + 5.1128) / (3.892 * (t_C / 100) + 1)
    f_m3 = 0.0365 * (t_C / 100) ** 2 - 0.0369 * (t_C / 100)
    fb = fw + f_m1 * m ** 1.5 + f_m2 * m + f_m3 * m ** 0.5
    ib70 = np.log(abs(eb + fb))/eb
    ib = np.log(abs(eb * p_MPaa / 70 + fb)) / eb
    # Плотность раствора при T, P
    ro_b = ro_b70 * np.exp(ib - ib70)
    if s == 0:
        ro = ro_w
    elif par == 0:
        ro = ro_b
    elif par == 1:
        # Найдем растворимость метана в растворе
        # Сперва определим давление насыщенных паров для чистой воды
        eps = 1 - t_K/647.096
        p_sigma = 22.064 * np.exp(647.096/t_K * (-7.85951783 * eps + 1.84408259 * eps ** 1.5 - 11.7866497 *
                                                 eps ** 3 + 22.6807411 * eps ** 3.5 - 15.9619719 * eps ** 4 +
                                                 1.80122502 * eps ** 7.5))
        # Определим коэффициенты растворимости метана
        a = -0.004462 * (t_C / 100) - 0.06763
        b = -0.03602 * (t_C / 100) ** 2 + 0.18917 * (t_C / 100) + 0.97242
        c = (0.6855 * (t_C / 100) ** 2 - 3.1992 * (t_C / 100) - 3.7968) / (0.07711 * (t_C / 100) ** 2 + 0.2229 *
                                                                           (t_C / 100) + 1)
        # Растворимость метана в чистой воде
        m_ch4_w = np.exp(a * (np.log(p_MPaa - p_sigma))**2 + b * np.log(p_MPaa - p_sigma) + c)
        # Далее найдем коэффициенты взаимодействия
        lyambda = -0.80898 + 1.0827e-3 * t_C + 183.85 / t_C + 3.924e-4 * p_MPaa - 1.97 *\
            10 ** (-6) * p_MPaa ** 2
        dzeta = -3.89e-3
        # Растворимость метана в растворе
        # нужно отметить, что в обозначениях было сложно разобраться, для понимания этой формулы лучше читать статью
        m_ch4_b = m_ch4_w * np.exp(-2 * lyambda * m - dzeta*m**2)
        # Производные необходимые для расчета формулы
        derivative1 = 7.6985890e-2 - 5.0253331e-5 * t_K - 30.092013 / t_K +\
            4.8468502e3/ t_K ** 2
        derivative2 = 3.924e-4 - 2 * 1.97e-6 * p_MPaa
        # Парциальный объем метана в растворе
        v_ch4_b = 8.314467 * t_K * (derivative1 + 2 * m * derivative2)
        # Удельный объем раствора без метана
        v_b0 = 1 / ro_b
        # Плотность раствора с растворенным в нем метаном
        ro_b_ch4 = (1000 + m * 58.4428 + m_ch4_b * 16.043)/((1000 + m * 58.4428) * v_b0 + m_ch4_b * v_ch4_b)
        ro = ro_b_ch4
    else:
        ro = 0
    ro = ro * 1000
    return ro


def unf_compressibility_brine_Spivey_1MPa(t_K, p_MPaa, s_ppm, z=1.0, par=1):
    """
        Modified Spivey et al. correlation for brine(water) compressibility (2009)

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :param s_ppm: salinity, ppm
    :param z: z-factor
    :param par: parameter, 0 - methane-free brine, 1 - brine containing dissolved methane,
                            2 - brine containing partially dissolved methane
    :return: compressibility, 1/MPa

    корреляция позволяет найти сжимаемость соленой воды с частично или полностью растворенным в ней метаном

    ref 1 Spivey, J.P., McCain, W.D., Jr., and North, R. “Estimating Density, Formation
    Volume Factor, Compressibility, Methane Solubility, and Viscosity for Oilfield
    Brines at Temperatures From 0 to 275°C, Pressures to 200 MPa, and Salinities to
    5.7 mole/kg.” Journal of Canadian Petroleum Technology. Vol. 43, No. 7 (July 2004)
    52–61.

    ref 2 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011

    """

    t_C = K_2_C(t_K)
    s = s_ppm / 1000000
    m = 1000 * s / (58.4428 * (1 - s))
    # Первым шагом вычисляется плотность чистой воды при давлении 70 MPa и температуре
    ro_w70 = (-0.127213 * (t_C / 100) ** 2 + 0.645486 * (t_C / 100) + 1.03265) / (-0.070291 * (t_C / 100) ** 2 +
                                                                                  0.639589 * (t_C / 100) + 1)
    # Температурные коэффициенты сжимаемости чистой воды
    ew = (4.221 * (t_C / 100)**2 - 3.478 * (t_C/100) + 6.221)/(0.5182 * (t_C/100)**2 - 0.4405 * (t_C/100) + 1)
    fw = (-11.403 * (t_C / 100) ** 2 + 29.932 * (t_C / 100) + 27.952) / (0.20684 * (t_C / 100) ** 2 +
                                                                         0.3768 * (t_C / 100) + 1)
    # Сжимаемость чистой воды при T, P
    c_w = (1/70) * 1 / (ew * (p_MPaa/70) + fw)
    # Температурные коэффициенты плотности раствора
    d_m1 = -1.1149e-4 * (t_C/100)**2 + 1.7105e-4 * (t_C/100) - 4.3766e-4
    d_m2 = (-8.878e-4 * (t_C/100)**2 - 1.388e-4 - 2.96318e-3)/(0.51103 * (t_C / 100) + 1)
    d_m3 = (2.1466e-3 * (t_C / 100) ** 2 + 1.2427e-2 * (t_C / 100) + 4.2648e-2) / \
           (-8.1009e-2 * (t_C / 100) ** 2 + 0.525417 * (t_C / 100) + 1)
    d_m4 = 2.356e-4 * (t_C/100)**2 - 3.636e-4 * (t_C/100) - 2.278e-4
    # Плотность раствора воды с хлоридом натрия при давлении 70 MPa и температуре
    ro_b70 = ro_w70 + d_m1 * m ** 2 + d_m2 * m ** 1.5 + d_m3 * m + d_m4 * m ** 0.5
    # Температурные коэффициенты сжимаемости раствора
    eb = ew + 0.1249
    f_m1 = (-0.617 * (t_C / 100) ** 2 - 0.747 * (t_C / 100) - 0.4339) / (10.26 * (t_C / 100) + 1)
    f_m2 = (9.917 * (t_C / 100) + 5.1128) / (3.892 * (t_C / 100) + 1)
    f_m3 = 0.0365 * (t_C / 100) ** 2 - 0.0369 * (t_C / 100)
    fb = fw + f_m1 * m ** 1.5 + f_m2 * m + f_m3 * m ** 0.5
    ib70 = np.log(abs(eb + fb)) / eb
    ib = np.log(abs(eb * p_MPaa / 70 + fb)) / eb
    # Плотность раствора при T, P
    ro_b = ro_b70 * np.exp(ib - ib70)
    # Сжимаемость соленого раствора при T, P
    c_b = (1/70) * 1 / (eb * (p_MPaa/70) + fb)
    # Найдем растворимость метана в растворе
    # Сперва определим давление насыщенных паров для чистой воды
    eps = 1 - t_K/647.096
    p_sigma = 22.064 * np.exp(647.096/t_K * (-7.85951783 * eps + 1.84408259 * eps ** 1.5 - 11.7866497 * eps ** 3 +
                                             22.6807411 * eps ** 3.5 - 15.9619719 * eps ** 4 + 1.80122502 * eps ** 7.5))
    # Определим коэффициенты растворимости метана
    a = -0.004462 * (t_C / 100) - 0.06763
    b = -0.03602 * (t_C / 100) ** 2 + 0.18917 * (t_C / 100) + 0.97242
    c = (0.6855 * (t_C / 100) ** 2 - 3.1992 * (t_C / 100) - 3.7968) / (0.07711 * (t_C / 100) ** 2 + 0.2229 * (t_C / 100)
                                                                       + 1)
    # Растворимость метана в чистой воде
    m_ch4_w = np.exp(a * (np.log(p_MPaa - p_sigma))**2 + b * np.log(p_MPaa - p_sigma) + c)
    # Далее найдем коэффициенты взаимодействия
    lyambda = -0.80898 + 1.0827e-3 * t_C + 183.85 / t_C + 3.924e-4 * p_MPaa - 1.97 * \
        10 ** (-6) * p_MPaa ** 2
    dzeta = -3.89e-3
    # Растворимость метана в растворе
    # нужно отметить, что в обозначениях было сложно разобраться, для понимания этой формулы лучше читать статью
    m_ch4_b = m_ch4_w * np.exp(-2 * lyambda * m - dzeta*m**2)
    # Производные необходимые для расчета формулы
    derivative1 = 7.6985890e-2 - 5.0253331e-5 * t_K - 30.092013 / t_K +\
        4.8468502e3/ t_K ** 2
    derivative2 = 3.924e-4 - 2 * 1.97e-6 * p_MPaa
    # Парциальный объем метана в растворе
    v_ch4_b = 8.314467 * t_K * (derivative1 + 2 * m * derivative2)
    # Удельный объем раствора без метана
    v_b0 = 1 / ro_b
    # Необходимые вторые производные
    dderivative1 = - v_b0 * c_b
    dderivative2 = 2 * (-1.97e-6)
    # Производная парциального объема метана в растворе
    derivative_v_ch4_b = 8.314467 * t_K * (2 * m * dderivative2)
    # Сжимаемость раствора в котором метан растворен полностью (однофазная система)
    c_b_ch4_u = -((1000 + m * 58.4428) * dderivative1 + m_ch4_b * derivative_v_ch4_b) / ((1000 + m * 58.4428) * v_b0
                                                                                        + m_ch4_b * v_ch4_b)
    # Найдем сжимаемость раствора в случае двуфазной системы
    # Необходимая производная от растворимости метана по давлению
    derivative_m_ch4_b = m_ch4_w * ((2 * a * np.log(p_MPaa - p_sigma) + b)/(p_MPaa - p_sigma) - 2 * m * derivative2)
    # Молярный объем метана в газовой фазе
    vm_ch4_g = z * 8.314467 * t_K / p_MPaa
    # Cжимаемость раствора в случае двуфазной системы
    c_b_ch4_s = -((1000 + m * 58.4428) * dderivative1 + m_ch4_b * derivative_v_ch4_b + derivative_m_ch4_b *
                 (v_ch4_b - vm_ch4_g)) / ((1000 + m * 58.4428) * v_b0 + m_ch4_b * v_ch4_b)
    if s == 0:
        c_1MPa = c_w
    elif par == 0:
        c_1MPa = c_b
    elif par == 1:
        c_1MPa = c_b_ch4_u
    elif par == 2:
        c_1MPa = c_b_ch4_s
    else:
        c_1MPa = 0
    return c_1MPa


def unf_fvf_brine_Spivey_m3m3(t_K, p_MPaa, s_ppm):  # TODO check
    """
        Modified Spivey et al. correlation for brine(water) formation volume factor (2009)

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :param s_ppm: salinity, ppm
    :return: formation volume factor, m3/m3

    корреляция позволяет найти объемный коэффициент для соленой воды с учетом растворенного метана

    ref 1 Spivey, J.P., McCain, W.D., Jr., and North, R. “Estimating Density, Formation
    Volume Factor, Compressibility, Methane Solubility, and Viscosity for Oilfield
    Brines at Temperatures From 0 to 275°C, Pressures to 200 MPa, and Salinities to
    5.7 mole/kg.” Journal of Canadian Petroleum Technology. Vol. 43, No. 7 (July 2004)
    52–61.

    ref 2 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011
    """

    t_C = K_2_C(t_K)
    s = s_ppm / 1000000
    m = 1000 * s / (58.4428 * (1 - s))
    # Первым шагом вычисляется плотность чистой воды при давлении 70 MPa и температуре
    ro_w70 = (-0.127213 * (t_C / 100) ** 2 + 0.645486 * (t_C / 100) + 1.03265) / (-0.070291 * (t_C / 100) ** 2 +
                                                                                  0.639589 * (t_C / 100) + 1)
    ro_w70_sc = (-0.127213 * (20 / 100) ** 2 + 0.645486 * (20 / 100) + 1.03265) / (-0.070291 * (20 / 100) ** 2 +
                                                                                   0.639589 * (20 / 100) + 1)
    # Температурные коэффициенты сжимаемости чистой воды
    ew = (4.221 * (t_C / 100) ** 2 - 3.478 * (t_C / 100) + 6.221) / (
                0.5182 * (t_C / 100) ** 2 - 0.4405 * (t_C / 100) + 1)
    fw = (-11.403 * (t_C / 100) ** 2 + 29.932 * (t_C / 100) + 27.952) / (0.20684 * (t_C / 100) ** 2 +
                                                                         0.3768 * (t_C / 100) + 1)
    ew_sc = (4.221 * (20 / 100) ** 2 - 3.478 * (20 / 100) + 6.221) / (
                0.5182 * (20 / 100) ** 2 - 0.4405 * (20 / 100) + 1)
    fw_sc = (-11.403 * (20 / 100) ** 2 + 29.932 * (20 / 100) + 27.952) / (0.20684 * (20 / 100) ** 2 +
                                                                          0.3768 * (20 / 100) + 1)
    # Температурные коэффициенты плотности раствора
    d_m1 = -1.1149e-4 * (t_C / 100) ** 2 + 1.7105e-4 * (t_C / 100) - 4.3766e-4
    d_m2 = (-8.878e-4 * (t_C / 100) ** 2 - 1.388e-4 - 2.96318e-3) / (
                0.51103 * (t_C / 100) + 1)
    d_m3 = (2.1466e-3 * (t_C / 100) ** 2 + 1.2427e-2 * (t_C / 100) + 4.2648e-2) / \
           (-8.1009e-2 * (t_C / 100) ** 2 + 0.525417 * (t_C / 100) + 1)
    d_m4 = 2.356e-4 * (t_C / 100) ** 2 - 3.636e-4 * (t_C / 100) - 2.278e-4
    d_m1_sc = -1.1149e-4 * (20 / 100) ** 2 + 1.7105e-4 * (20 / 100) - 4.3766e-4
    d_m2_sc = (-8.878e-4 * (20 / 100) ** 2 - 1.388e-4 - 2.96318e-3) / (
            0.51103 * (20 / 100) + 1)
    d_m3_sc = (2.1466e-3 * (20 / 100) ** 2 + 1.2427e-2 * (20 / 100) + 4.2648e-2) / \
        (-8.1009e-2 * (20 / 100) ** 2 + 0.525417 * (20 / 100) + 1)
    d_m4_sc = 2.356e-4 * (20 / 100) ** 2 - 3.636e-4 * (20 / 100) - 2.278e-4
    # Плотность раствора воды с хлоридом натрия при давлении 70 MPa и температуре
    ro_b70 = ro_w70 + d_m1 * m ** 2 + d_m2 * m ** 1.5 + d_m3 * m + d_m4 * m ** 0.5
    ro_b70_sc = ro_w70_sc + d_m1_sc * m ** 2 + d_m2_sc * m ** 1.5 + d_m3_sc * m + d_m4_sc * m ** 0.5
    # Температурные коэффициенты сжимаемости раствора
    eb = ew + 0.1249
    eb_sc = ew_sc + 0.1249
    f_m1 = (-0.617 * (t_C / 100) ** 2 - 0.747 * (t_C / 100) - 0.4339) / (10.26 * (t_C / 100) + 1)
    f_m2 = (9.917 * (t_C / 100) + 5.1128) / (3.892 * (t_C / 100) + 1)
    f_m3 = 0.0365 * (t_C / 100) ** 2 - 0.0369 * (t_C / 100)
    f_m1_sc = (-0.617 * (20 / 100) ** 2 - 0.747 * (20 / 100) - 0.4339) / (10.26 * (20 / 100) + 1)
    f_m2_sc = (9.917 * (20 / 100) + 5.1128) / (3.892 * 20 / 100 + 1)
    f_m3_sc = 0.0365 * (20 / 100) ** 2 - 0.0369 * (20 / 100)
    fb = fw + f_m1 * m ** 1.5 + f_m2 * m + f_m3 * m ** 0.5
    fb_sc = fw_sc + f_m1_sc * m ** 1.5 + f_m2_sc * m + f_m3_sc * m ** 0.5
    ib70 = np.log(abs(eb + fb)) / eb
    ib70_sc = np.log(abs(eb_sc + fb_sc)) / eb_sc
    ib = np.log(abs(eb * p_MPaa / 70 + fb)) / eb
    ib_sc = np.log(abs(eb_sc * p_MPaa / 70 + fb_sc)) / eb_sc
    # Плотность раствора при T, P
    ro_b = ro_b70 * np.exp(ib - ib70)
    ro_b_sc = ro_b70_sc * np.exp(ib_sc - ib70_sc)
    # Найдем растворимость метана в растворе
    # Сперва определим давление насыщенных паров для чистой воды
    eps = 1 - t_K/647.096
    p_sigma = 22.064 * np.exp(647.096/t_K * (-7.85951783 * eps + 1.84408259 * eps ** 1.5 - 11.7866497 * eps ** 3 +
                                             22.6807411 * eps ** 3.5 - 15.9619719 * eps ** 4 + 1.80122502 * eps ** 7.5))
    # Определим коэффициенты растворимости метана
    a = -0.004462 * (t_C / 100) - 0.06763
    b = -0.03602 * (t_C / 100) ** 2 + 0.18917 * (t_C / 100) + 0.97242
    c = (0.6855 * (t_C / 100) ** 2 - 3.1992 * (t_C / 100) - 3.7968) / (0.07711 * (t_C / 100) ** 2 + 0.2229 * (t_C / 100)
                                                                       + 1)
    # Растворимость метана в чистой воде
    m_ch4_w = np.exp(a * (np.log(p_MPaa - p_sigma))**2 + b * np.log(p_MPaa - p_sigma) + c)
    # Далее найдем коэффициенты взаимодействия
    lyambda = -0.80898 + 1.0827e-3 * t_C + 183.85 / t_C + 3.924e-4 * p_MPaa - 1.97e-6 *\
        p_MPaa ** 2
    dzeta = -3.89e-3
    # Растворимость метана в растворе
    # нужно отметить, что в обозначениях было сложно разобраться, для понимания этой формулы лучше читать статью
    m_ch4_b = m_ch4_w * np.exp(-2 * lyambda * m - dzeta*m**2)
    # Производные необходимые для расчета формулы
    derivative1 = 7.6985890e-2 - 5.0253331e-5 * t_K - 30.092013 / t_K + 4.8468502e3/\
        t_K ** 2
    derivative2 = 3.924e-4 - 2 * 1.97e-6 * p_MPaa
    # Парциальный объем метана в растворе
    v_ch4_b = 8.314467 * t_K * (derivative1 + 2 * m * derivative2)
    # Удельный объем раствора без метана
    v_b0 = 1 / ro_b
    v_b0_sc = 1 / ro_b_sc
    bw = ((1000 + m * 58.4428) * v_b0 + m_ch4_b * v_ch4_b)/((1000 + m * 58.4428) * v_b0_sc)
    return bw


def unf_gwr_brine_Spivey_m3m3(s_ppm, z):  # TODO check
    """
        Modified Spivey et al. correlation for solution gas-water ratio of methane in brine(2009)

    :param s_ppm: salinity, ppm
    :param z: z-factor
    :return: GWR, m3/m3

    корреляция позволяет найти газосодержание метана в соленой воде

    ref 1 Spivey, J.P., McCain, W.D., Jr., and North, R. “Estimating Density, Formation
    Volume Factor, Compressibility, Methane Solubility, and Viscosity for Oilfield
    Brines at Temperatures From 0 to 275°C, Pressures to 200 MPa, and Salinities to
    5.7 mole/kg.” Journal of Canadian Petroleum Technology. Vol. 43, No. 7 (July 2004)
    52–61.

    ref 2 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011

    """

    s = s_ppm / 1000000
    m = 1000 * s / (58.4428 * (1 - s))
    t_C_sc = 20
    t_K_sc = 293.15
    p_MPaa_sc = 0.1013
    # Первым шагом вычисляется плотность чистой воды при давлении 70 MPa и температуре
    ro_w70_sc = (-0.127213 * (t_C_sc/100)**2 + 0.645486 * (t_C_sc/100) + 1.03265)/(-0.070291 * (t_C_sc/100)**2 +
                                                                                   0.639589 * (t_C_sc/100) + 1)
    # Температурные коэффициенты сжимаемости чистой воды
    ew_sc = (4.221 * (t_C_sc / 100) ** 2 - 3.478 * (t_C_sc / 100) + 6.221) / (0.5182 * (t_C_sc / 100) ** 2 - 0.4405 *
                                                                              (t_C_sc / 100) + 1)
    fw_sc = (-11.403 * (t_C_sc / 100) ** 2 + 29.932 * (t_C_sc / 100) + 27.952) / (0.20684 * (t_C_sc / 100) ** 2 +
                                                                                  0.3768 * (t_C_sc / 100) + 1)
    # Температурные коэффициенты плотности раствора
    d_m1 = -1.1149e-4 * (t_C_sc/100)**2 + 1.7105e-4 * (t_C_sc/100) - 4.3766e-4
    d_m2 = (-8.878e-4 * (t_C_sc/100)**2 - 1.388e-4 - 2.96318e-3)/(0.51103 * (t_C_sc /
                                                                                                           100) + 1)
    d_m3 = (2.1466e-3 * (t_C_sc / 100) ** 2 + 1.2427e-2 * (t_C_sc / 100) + 4.2648e-2) / \
           (-8.1009e-2 * (t_C_sc / 100) ** 2 + 0.525417 * (t_C_sc / 100) + 1)
    d_m4 = 2.356e-4 * (t_C_sc/100)**2 - 3.636e-4 * (t_C_sc/100) - 2.278e-4
    # Плотность раствора воды с хлоридом натрия при давлении 70 MPa и температуре
    ro_b70_sc = ro_w70_sc + d_m1 * m ** 2 + d_m2 * m ** 1.5 + d_m3 * m + d_m4 * m ** 0.5
    # Температурные коэффициенты сжимаемости раствора
    eb_sc = ew_sc + 0.1249
    f_m1 = (-0.617 * (t_C_sc / 100) ** 2 - 0.747 * (t_C_sc / 100) - 0.4339) / (10.26 * (t_C_sc / 100) + 1)
    f_m2 = (9.917 * (t_C_sc / 100) + 5.1128) / (3.892 * (t_C_sc / 100) + 1)
    f_m3 = 0.0365 * (t_C_sc / 100) ** 2 - 0.0369 * (t_C_sc / 100)
    fb_sc = fw_sc + f_m1 * m ** 1.5 + f_m2 * m + f_m3 * m ** 0.5
    ib70_sc = np.log(abs(eb_sc + fb_sc))/eb_sc
    ib_sc = np.log(abs(eb_sc * p_MPaa_sc / 70 + fb_sc)) / eb_sc
    # Плотность раствора при T, P
    ro_b_sc = ro_b70_sc * np.exp(ib_sc - ib70_sc)
    # Найдем растворимость метана в растворе
    # Сперва определим давление насыщенных паров для чистой воды
    eps = 1 - t_K_sc/647.096
    p_sigma = 22.064 * np.exp(647.096/t_K_sc * (-7.85951783 * eps + 1.84408259 * eps ** 1.5 - 11.7866497 *
                                                eps ** 3 + 22.6807411 * eps ** 3.5 - 15.9619719 * eps ** 4 +
                                                1.80122502 * eps ** 7.5))
    # Определим коэффициенты растворимости метана
    a = -0.004462 * (t_C_sc / 100) - 0.06763
    b = -0.03602 * (t_C_sc / 100) ** 2 + 0.18917 * (t_C_sc / 100) + 0.97242
    c = (0.6855 * (t_C_sc / 100) ** 2 - 3.1992 * (t_C_sc / 100) - 3.7968) / (0.07711 * (t_C_sc / 100) ** 2 + 0.2229 *
                                                                             (t_C_sc / 100) + 1)
    # Растворимость метана в чистой воде
    m_ch4_w = np.exp(a * (np.log(p_MPaa_sc - p_sigma))**2 + b * np.log(p_MPaa_sc - p_sigma) + c)
    # Далее найдем коэффициенты взаимодействия
    lyambda = -0.80898 + 1.0827e-3 * t_C_sc + 183.85 / t_C_sc + 3.924e-4 * p_MPaa_sc - 1.97 * 10 **\
        (-6) * p_MPaa_sc ** 2
    dzeta = -3.89e-3
    # Растворимость метана в растворе
    # нужно отметить, что в обозначениях было сложно разобраться, для понимания этой формулы лучше читать статью
    m_ch4_b = m_ch4_w * np.exp(-2 * lyambda * m - dzeta*m**2)
    # Удельный объем раствора без метана
    v_b0_sc = 1 / ro_b_sc
    # Молярный объем метана в газовой фазе
    vm_ch4_g_sc = z * 8.314467 * t_K_sc / p_MPaa_sc
    # Найдем GWR
    gwr = m_ch4_b * vm_ch4_g_sc / ((1000 + m * 58.4428) * v_b0_sc)
    return gwr



def unf_viscosity_brine_MaoDuan_cP(t_K, p_MPaa, s_ppm):  #TODO check
    """
        Mao-Duan correlation for brine(water) viscosity (2009)

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :param s_ppm: salinity, ppm
    :return: viscosity, cP

    корреляция позволяет найти вязкость соленой воды

    ref 1 Mao, S., and Duan, Z. “The Viscosity of Aqueous Alkali-Chloride Solutions up to
    623 K, 1,000 bar, and High Ionic Strength.” International Journal of Thermophysics.
    Vol. 30 (2009) 1,510–1,523.

    ref 2 book Mccain_w_d_spivey_j_p_lenn_c_p_petroleum_reservoir_fluid,third edition, 2011

    """
    t_C = K_2_C(t_K)
    s = s_ppm / 1000000
    m = 1000 * s / (58.4428 * (1 - s))
    # Первым шагом вычисляется плотность чистой воды при давлении 70 MPa и температуре
    ro_w70 = (-0.127213 * (t_C/100)**2 + 0.645486 * (t_C/100) + 1.03265)/(-0.070291 * (t_C/100)**2 +
                                                                          0.639589 * (t_C/100) + 1)
    # Температурные коэффициенты сжимаемости чистой воды
    ew = (4.221 * (t_C / 100)**2 - 3.478 * (t_C/100) + 6.221)/(0.5182 * (t_C/100)**2 - 0.4405 * (t_C/100) + 1)
    fw = (-11.403 * (t_C / 100) ** 2 + 29.932 * (t_C / 100) + 27.952) / (0.20684 * (t_C / 100) ** 2 +
                                                                         0.3768 * (t_C / 100) + 1)
    iw70 = np.log(abs(ew + fw))/ew
    iw = np.log(abs(ew*(p_MPaa/70) + fw))/ew
    # Плотность чистой воды при T, P
    ro_w = ro_w70 * np.exp(iw - iw70)
    # Вязкость чистой воды
    viscosity = np.exp(0.28853170 * 10 ** 7 * t_K ** (-2) - 0.11072577 * 10 ** 5 * t_K ** (-1) - 0.90834095 * 10 +
                       0.30925651 * 10 ** (-1) * t_K - 0.27407100e-4 * t_K ** 2 + ro_w *
                       (-0.19298951 * 10 ** 7 * t_K ** (-2) + 0.56216046 * 10 ** 4 * t_K ** (-1) + 0.13827250 *
                        10 ** 2 - 0.47609523 * 10 ** (-1) * t_K + 0.35545041e-4 * t_K ** 2))
    # Коэффициенты, зависящие от температуры
    a = -0.213119213 + 0.13651589e-2 * t_K - 0.12191756e-5 * t_K ** 2
    b = -0.69161945 * 10 ** (-1) - 0.27292263e-3 * t_K + 0.20852448e-6 * t_K ** 2
    c = -0.25988855e-2 + 0.77989227e-5 * t_K
    # Относительная вязкость
    viscosity_rel = np.exp(a * m + b * m ** 2 + c * m ** 3)
    # Вязкость соленой воды
    viscosity_cP = 1000 * viscosity * viscosity_rel
    return viscosity_cP


# TODO добавить для термодинамических свойств воды учет давления, температуры, солености
"""
Следующие свойства взяты при давлении 1 бар для дистилированной воды
за границей применимости 5 < T < 95 C произведена линейная экстраполяция
"""


def unf_heat_capacity_water_IAPWS_JkgC(t_c):  # TODO заменить
    """
        Теплоемкость дистилированной воды в диапазоне 5 < T < 95 C при 1 бар
    выше диапазона - линейная экстраполяция

    :param t_c: температура в С
    :return: теплоемкость в Дж / кг С

    ref1 https://syeilendrapramuditya.wordpress.com/2011/08/20/water-thermodynamic-properties/
    """
    def cor_in_range_5_95(t_c):
        return (4.214 - 2.286*10**(-3) * t_c +
               4.991 * 10**(-5) * t_c ** 2 -
               4.519 * 10**(-7) * t_c ** 3 +
               1.857 * 10**(-9) * t_c ** 4)
    if t_c < 95:
        return cor_in_range_5_95(t_c)* 1000
    else:
        return (cor_in_range_5_95(95) +
                (cor_in_range_5_95(95) - cor_in_range_5_95(85)) / 10 *
                (t_c - 95)) * 1000


def unf_thermal_conductivity_water_IAPWS_WmC(t_c):  # TODO заменить
    """
        Теплопроводность дистилированной воды в диапазоне 5 < T < 95 C при 1 бар
    выше диапазона - линейная экстраполяция

    :param t_c: температура в С
    :return: теплопроводность в Вт / м С

    ref1 https://syeilendrapramuditya.wordpress.com/2011/08/20/water-thermodynamic-properties/
    """
    def cor_in_range_5_95(t_c):
        return (0.5636 + 1.946 * 10**(-3) * t_c -
                8.151 * 10** (-6) *t_c **2)
    if t_c < 95:
        return cor_in_range_5_95(t_c)
    else:
        return (cor_in_range_5_95(95) +
                (cor_in_range_5_95(95) - cor_in_range_5_95(85)) / 10 *
                (t_c - 95))


def unf_thermal_expansion_coefficient_water_IAPWS_1C(t_c):  # TODO заменить
    """
        Коэффициент термического расширения дистилированной воды в диапазоне 5 < T < 95 C при 1 бар
    выше диапазона - линейная экстраполяция

    :param t_c: температура в С
    :return: Коэффициент термического расширения в 1 / с

    ref1 https://syeilendrapramuditya.wordpress.com/2011/08/20/water-thermodynamic-properties/
    """
    return 7.957 * 10**(-5) + 7.315 * 10**(-6) * t_c


# Корреляцияные зависимости для нефтяных систем


def unf_surface_tension_go_Abdul_Majeed_Nm(t_K, gamma_oil, rs_m3m3):
    """
        Корреляция Абдул-Маджида (2000 г.) для поверхностного натяжения нефти, насыщенной газом

    :param t_K: температура, градусы Кельвина
    :param gamma_oil: относительная плотность нефти
    :param rs_m3m3: газосодержание, м3 / м3
    :return: поверхностное натяжение на границе нефть-газ, Н / м

        Источник: Справочник инженера-нефтяника. Том 1. Введение в нефтяной инжиниринг. Газпром Нефть
    """
    t_C = K_2_C(t_K)
    surface_tension_dead_oil_dynes_cm = (1.17013 - 1.694e-3 * (1.8 * t_C + 32)) * (
                38.085 - 0.259 * (141.5 / gamma_oil - 131.5))
    relative_surface_tension_go_od = (0.056379 + 0.94362 * np.exp(-21.6128e-3 * rs_m3m3))
    surface_tension_dynes_cm = surface_tension_dead_oil_dynes_cm * relative_surface_tension_go_od
    return dyncm_2_Nm(surface_tension_dynes_cm)


def unf_surface_tension_go_Baker_Swerdloff_Nm(t_K, gamma_oil, p_MPa):
    """
        Корреляция Бэйкера и Свердлоффа (1955 г.) для поверхностного натяжения нефти, насыщенной газом

    :param t_K: температура, градусы Кельвина
    :param gamma_oil: относительная плотность нефти
    :param p_MPa: давление, МПа
    :return: поверхностное натяжения на границе нефть-газ в Н /м

    Источник: Справочник инженера-нефтяника. Том 1. Введение в нефтяной инжиниринг. Газпром Нефть
    """
    p_bar = MPa_2_bar(p_MPa)
    t_C = K_2_C(t_K)
    surface_tension_dead_oil_20_c_dynes_cm = 39 - 0.2571 * (141.5 / gamma_oil - 131.5 )
    surface_tension_dead_oil_38_c_dynes_cm = 37.5 - 0.2571 * (141.5 / gamma_oil - 131.5 )
    if t_C <= 20:
        surface_tension_dead_oil_dynes_cm = surface_tension_dead_oil_20_c_dynes_cm
    elif t_C >= 38:
        surface_tension_dead_oil_dynes_cm = surface_tension_dead_oil_38_c_dynes_cm
    else:
        surface_tension_dead_oil_dynes_cm = (surface_tension_dead_oil_20_c_dynes_cm -
                                             (t_C - 20) * (surface_tension_dead_oil_20_c_dynes_cm -
                                                           surface_tension_dead_oil_38_c_dynes_cm) / 18)
    surface_tension_go_Baker_Swerdloff_dynes_cm = (surface_tension_dead_oil_dynes_cm *
                                                   np.exp(-125.1763e-4 * p_bar))

    return dyncm_2_Nm(surface_tension_go_Baker_Swerdloff_dynes_cm)


def unf_surface_tension_gw_Sutton_Nm(rho_water_kgm3, rho_gas_kgm3, t_c):  # TODO поправка на соленость добавить
    """
        Корреляция Саттона для поверхностного натяжения на границе вода-газ

    :param rho_water_kgm3: плотность воды кг / м3
    :param rho_gas_kgm3:  плотность газа кг / м3
    :param t_c: температура в С
    :return: поверхностное натяжение на границе вода-газ, Н / м

    ref 1 Pereira L. et al. Interfacial tension of reservoir fluids: an integrated experimental
    and modelling investigation : дис. – Heriot-Watt University, 2016. page 41

    ref2 Ling K. et al. A new correlation to calculate oil-water interfacial tension
    //SPE Kuwait International Petroleum Conference and Exhibition. – Society of Petroleum Engineers, 2012.

    """
    rho_water_gcm3 = rho_water_kgm3 / 1000
    rho_gas_gcm3 = rho_gas_kgm3 / 1000
    t_r = C_2_R(t_c)
    surface_tension_dyncm = ((1.53988 * (rho_water_gcm3 - rho_gas_gcm3) + 2.08339) /
            ((t_r /302.881) ** (0.821976 - 0.00183785 * t_r +
            0.00000134016 * t_r ** 2))) ** 3.6667
    return dyncm_2_Nm(surface_tension_dyncm)


def unf_surface_tension_Baker_Sverdloff_vba_nm(p_atma, t_C, gamma_o_):
    t_F = t_C * 1.8 + 32
    P_psia = p_atma / 0.068046
    P_MPa = p_atma / 10
    ST68 = 39 - 0.2571 * (141.5 / gamma_o_ - 131.5)
    ST100 = 37.5 - 0.2571 * (141.5 / gamma_o_ - 131.5)
    if t_F < 68:
        STo = ST68
    else:
        Tst = t_F
        if t_F > 100:
            Tst = 100
        STo = (68 - (((Tst - 68) * (ST68 - ST100)) / 32)) * np.exp(-0.00086306 * P_psia)

    STw74 = (75 - (1.108 * (P_psia) ** 0.349))
    STw280 = (53 - (0.1048 * (P_psia) ** 0.637))

    if t_F < 74:
        STw = STw74
    else:
        Tstw = t_F
        if t_F > 280:
            Tstw = 280
        STw = STw74 - (((Tstw - 74) * (STw74 - STw280)) / 206)
    STw = 10 ** (-(1.19 + 0.01 * P_MPa)) * 1000
    ST_oilgas_dyncm_ = STo
    ST_watgas_dyncm_ = STw
    return [dyncm_2_Nm(ST_oilgas_dyncm_), dyncm_2_Nm(ST_watgas_dyncm_)]

