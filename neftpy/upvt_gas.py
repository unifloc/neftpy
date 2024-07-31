from neftpy.uconvert import *
from neftpy.uconst import *

import numpy as np
import scipy.optimize as opt


# uPVT свойства для газа

def unf_pseudocritical_temperature_K(gamma_gas:float, 
                                     y_h2s:float=0.0, 
                                     y_co2:float=0.0, 
                                     y_n2:float=0.0,
                                     )->float:
    """
        Correlation for pseudocritical temperature taking into account the presense of non-hydrocarbon gases

    :param gamma_gas: specific gas density (by air)
    :param y_h2s: mole fraction of the hydrogen sulfide
    :param y_co2: mole fraction of the carbon dioxide
    :param y_n2: mole fraction of the nitrogen
    :return: pseudocritical temperature, K

    ref 1 Piper, L.D., McCain, W.D., Jr., and Corredor, J.H. “Compressibility Factors for
    Naturally Occurring Petroleum Gases.” Gas Reservoir Engineering. Reprint Series. Richardson,
    TX: SPE. Vol. 52 (1999) 186–200
    """
    """
    tc_h2s_K,            critical temperature for hydrogen sulfide, K
    tc_co2_K,            critical temperature for carbon dioxide, K
    tc_n2_K,             critical temperature for nitrogen, K
    pc_h2s_MPaa,         critical pressure for hydrogen sulfide, MPaa
    pc_co2_MPaa,         critical pressure for carbon dioxide, MPaa
    pc_n2_MPaa,          critical pressure for nitrogen, MPaa
    """
    tc_h2s_R = K_2_R(373.6)
    tc_co2_R = K_2_R(304.13)
    tc_n2_R = K_2_R(126.25)
    pc_h2s_psia = MPa_2_psi(9.007)
    pc_co2_psia = MPa_2_psi(7.375)
    pc_n2_psia = MPa_2_psi(3.4)
    J = 1.1582e-1 - \
        4.5820e-1 * y_h2s * (tc_h2s_R / pc_h2s_psia) - \
        9.0348e-1 * y_co2 * (tc_co2_R / pc_co2_psia) - \
        6.6026e-1 * y_n2 * (tc_n2_R / pc_n2_psia) + \
        7.0729e-1 * gamma_gas - \
        9.9397e-2 * gamma_gas ** 2
    K = 3.8216 - \
        6.5340e-2 * y_h2s * (tc_h2s_R / pc_h2s_psia) - \
        4.2113e-1 * y_co2 * (tc_co2_R / pc_co2_psia) - \
        9.1249e-1 * y_n2 * (tc_n2_R / pc_n2_psia) + \
        17.438  * gamma_gas - \
        3.2191 * gamma_gas ** 2
    tpc_R = K ** 2 / J
    tpc_K = R_2_K(tpc_R)
    return tpc_K


def unf_pseudocritical_pressure_MPa(gamma_gas:float, 
                                    y_h2s:float=0.0, 
                                    y_co2:float=0.0, 
                                    y_n2:float=0.0
                                    )->float:
    """
        Correlation for pseudocritical pressure taking into account the presense of non-hydrocarbon gases

    :param gamma_gas: specific gas density (by air)
    :param y_h2s: mole fraction of the hydrogen sulfide
    :param y_co2: mole fraction of the carbon dioxide
    :param y_n2: mole fraction of the nitrogen
    :return: pseudocritical pressure, MPa

    ref 1 Piper, L.D., McCain, W.D., Jr., and Corredor, J.H. “Compressibility Factors for
    Naturally Occurring Petroleum Gases.” Gas Reservoir Engineering. Reprint Series. Richardson,
    TX: SPE. Vol. 52 (1999) 186–200
    """
    """          
    tc_h2s_K,            critical temperature for hydrogen sulfide, K
    tc_co2_K,            critical temperature for carbon dioxide, K
    tc_n2_K,             critical temperature for nitrogen, K
    pc_h2s_MPaa,         critical pressure for hydrogen sulfide, MPaa
    pc_co2_MPaa,         critical pressure for carbon dioxide, MPaa
    pc_n2_MPaa,          critical pressure for nitrogen, MPaa               
    """
    tc_h2s_R = K_2_R(373.6)
    tc_co2_R = K_2_R(304.13)
    tc_n2_R = K_2_R(126.25)
    pc_h2s_psia = MPa_2_psi(9.007)
    pc_co2_psia = MPa_2_psi(7.375)
    pc_n2_psia = MPa_2_psi(3.4)

    J = 1.1582e-1 - \
        4.5820e-1 * y_h2s * (tc_h2s_R / pc_h2s_psia) - \
        9.0348e-1 * y_co2 * (tc_co2_R / pc_co2_psia) - \
        6.6026e-1 * y_n2 * (tc_n2_R / pc_n2_psia) + \
        7.0729e-1 * gamma_gas - \
        9.9397e-2 * gamma_gas ** 2
    K = 3.8216 - \
        6.5340e-2 * y_h2s * (tc_h2s_R / pc_h2s_psia) - \
        4.2113e-1 * y_co2 * (tc_co2_R / pc_co2_psia) - \
        9.1249e-1 * y_n2 * (tc_n2_R / pc_n2_psia) + \
        1.7438 * 10 * gamma_gas - \
        3.2191 * gamma_gas ** 2
    tpc_R = K ** 2 / J
    ppc_psia = tpc_R / J
    ppc_MPa = psi_2_MPa(ppc_psia)
    return ppc_MPa


def unf_pseudocritical_temperature_Standing_K(gamma_gas:float)->float:  # VBA
    return 93.3 + 180 * gamma_gas - 6.94 * gamma_gas ** 2


def unf_pseudocritical_pressure_Standing_MPa(gamma_gas:float)->float:  # VBA
    return 4.6 + 0.1 * gamma_gas - 0.258 * gamma_gas ** 2


def unf_zfactor_BrillBeggs(ppr:float, 
                           tpr:float
                           )->float:
    """
        Correlation for z-factor according Beggs & Brill correlation (1977)

    используется для приближения функции дранчука

    :param ppr: preudoreduced pressure
    :param tpr: pseudoreduced temperature
    :return: z-factor

    Можно использовать при tpr<=2 и ppr<=4
    при tpr <== 1.5 ppr<=10
    """

    a = 1.39 * (tpr - 0.92) ** 0.5 - 0.36 * tpr - 0.101
    b = (0.62 - 0.23 * tpr) * ppr
    c = (0.066/(tpr - 0.86) - 0.037) * ppr ** 2
    d = (0.32/(10 ** (9 * (tpr - 1)))) * ppr ** 6
    e = b + c + d
    f = (0.132 - 0.32 * np.log10(tpr))
    g = 10 ** (0.3106 - 0.49 * tpr + 0.1824 * tpr ** 2)
    z = a + (1 - a) * np.exp(-e) + f * ppr ** g
    return z


def unf_zfactor_DAK(p_MPaa:float, 
                    t_K:float, 
                    ppc_MPa:float, 
                    tpc_K:float)->float:
    """
        Correlation for z-factor

    :param p_MPaa: pressure, MPaa
    :param t_K: temperature, K
    :param ppc_MPa: pseudocritical pressure, MPa
    :param tpc_K: pseudocritical temperature, K
    :return: z-factor

    range of applicability is (0.2<=ppr<30 and 1.0<tpr<=3.0) and also ppr < 1.0 for 0.7 < tpr < 1.0

    ref 1 Dranchuk, P.M. and Abou-Kassem, J.H. “Calculation of Z Factors for Natural
    Gases Using Equations of State.” Journal of Canadian Petroleum Technology. (July–September 1975) 34–36.

    """

    ppr = p_MPaa / ppc_MPa
    tpr = t_K / tpc_K

    return unf_zfactor_DAK_ppr(ppr, tpr)


def unf_zfactor_DAK_ppr(ppr:float, tpr:float)->float:
    """
        Correlation for z-factor

    :param ppr: pseudoreduced pressure
    :param tpr: pseudoreduced temperature
    :return: z-factor

    range of applicability is (0.2<=ppr<30 and 1.0<tpr<=3.0) and also ppr < 1.0 for 0.7 < tpr < 1.0

    ref 1 Dranchuk, P.M. and Abou-Kassem, J.H. “Calculation of Z Factors for Natural
    Gases Using Equations of State.” Journal of Canadian Petroleum Technology. (July–September 1975) 34–36.

    """

    z0 = 1
    ropr0 = 0.27 * (ppr / (z0 * tpr))

    def f(variables):
        z, ropr = variables
        func = np.zeros(2)
        func[0] = 0.27 * (ppr / (z * tpr)) - ropr
        func[1] = -z + 1 + \
                (0.3265 - 1.0700 / tpr - 0.5339 / tpr**3 + 0.01569 / tpr ** 4 - 0.05165 / tpr ** 5) * ropr +\
                (0.5475 - 0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 2 - \
                0.1056 * (-0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 5 + \
                0.6134 * (1 + 0.7210 * ropr ** 2) * (ropr ** 2 / tpr ** 3) * np.exp(-0.7210 * ropr ** 2)
        return func
    solution = opt.fsolve(f, np.array([z0, ropr0]))
    """
    solution = opt.newton(f, z0, maxiter=150, tol=1e-4)
    """
    return solution[0]


def unf_z_factor_Kareem(Tpr:float, 
                        Ppr:float
                        )->float:
    """
    based on  https://link.springer.com/article/10.1007/s13202-015-0209-3
    Kareem, L.A., Iwalewa, T.M. & Al-Marhoun, M.
    New explicit correlation for the compressibility factor of natural gas: linearized z-factor isotherms.
    J Petrol Explor Prod Technol 6, 481–492 (2016).
    https://doi.org/10.1007/s13202-015-0209-3
    :param Tpr:
    :param Ppr:
    :return:
    """


    a = (0,
         0.317842,
        0.382216,
        -7.768354,
        14.290531,
        0.000002,
        -0.004693,
        0.096254,
        0.16672,
        0.96691,
        0.063069,
        -1.966847,
        21.0581,
        -27.0246,
        16.23,
        207.783,
        -488.161,
        176.29,
        1.88453,
        3.05921	
        )

    t = 1 / Tpr
    AA = a[1]* t * np.exp(a[2] * (1 - t) ** 2) * Ppr
    BB = a[3]* t + a[4] * t ** 2 + a[5] * t ** 6 * Ppr ** 6
    CC = a[9]+ a[8] * t * Ppr + a[7] * t ** 2 * Ppr ** 2 + a[6] * t ** 3 * Ppr ** 3
    DD = a[10] * t * np.exp(a[11] * (1 - t) ** 2)
    EE = a[12] * t + a[13] * t ** 2 + a[14] * t ** 3
    FF = a[15] * t + a[16] * t ** 2 + a[17] * t ** 3
    GG = a[18] + a[19] * t

    DPpr = DD * Ppr
    y = DPpr / ((1 + AA ** 2) / CC - AA ** 2 * BB / (CC ** 3))

    z = DPpr * (1 + y + y ** 2 - y ** 3) / (DPpr + EE * y ** 2 - FF * y ** GG) / ((1 - y) ** 3)

    return z


def unf_compressibility_gas_Mattar_1MPa(p_MPaa:float, 
                                        t_K:float, 
                                        ppc_MPa:float, 
                                        tpc_K:float
                                        )->float:
    """
        Correlation for gas compressibility

    :param p_MPaa: pressure, MPaa
    :param t_K: temperature, K
    :param ppc_MPa: pseudocritical pressure, MPa
    :param tpc_K: pseudocritical temperature, K
    :return: gas compressibility, 1/MPa

    ref 1 Mattar, L., Brar, G.S., and Aziz, K. 1975. Compressibility of Natural Gases.
    J Can Pet Technol 14 (4): 77. PETSOC-75-04-08

    """
    #TODO надо разобраться с реализацией
    ppr = p_MPaa / ppc_MPa
    tpr = t_K / tpc_K
    z0 = 1
    ropr0 = 0.27 * (ppr / (z0 * tpr))

    def f(variables):
        z = variables[0]
        ropr = variables[1]
        func = np.zeros(2)
        func[0] = 0.27 * (ppr / (z * tpr)) - ropr
        func[1] = -z + 1 + \
            (0.3265 - 1.0700 / tpr - 0.5339 / tpr ** 3 + 0.01569 / tpr ** 4 - 0.05165 / tpr ** 5) * ropr +\
            (0.5475 - 0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 2 - \
            0.1056 * (-0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 5 + \
            0.6134 * (1 + 0.7210 * ropr ** 2) * (ropr ** 2 / tpr ** 3) * np.exp(-0.7210 * ropr ** 2)
        return func
    solution = np.array(opt.fsolve(f, np.array([z0, ropr0])))

    z_derivative = 0.3265 - 1.0700 / tpr - 0.5339 / tpr ** 3 + 0.01569 / tpr ** 4 - 0.05165 / tpr ** 5 + \
        2 * solution[1] * (0.5475 - 0.7361 / tpr + 0.1844 / tpr ** 2) - \
        5 * 0.1056 * (-0.7361 / tpr + 0.1844 / tpr ** 2) * solution[1] ** 4 + \
        2 * 0.6134 * solution[1] / tpr ** 3 * (1 + \
                                               0.7210 * solution[1] ** 2 - \
                                               0.7210 ** 2 * solution[1] ** 4) * np.exp(-0.7210 * solution[1] ** 2)
    
    cpr = 1 / solution[1] - 0.27 / (solution[0] ** 2 * tpr) * (z_derivative / (1 + solution[1] * z_derivative /
                                                                               solution[0]))
    cg = cpr / ppc_MPa
    return cg


def unf_gasviscosity_Lee_cP(t_K:float, p_MPaa:float, z:float, gamma_gas:float)->float:
    """
        Lee correlation for gas viscosity

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :param z: z-factor
    :param gamma_gas: specific gas density (by air)
    :return: gas viscosity,cP

    ref 1 Lee, A.L., Gonzalez, M.H., and Eakin, B.E. “The Viscosity of Natural Gases.” Journal
    of Petroleum Technology. Vol. 18 (August 1966) 997–1,000.
    """

    t_R = K_2_R(t_K)
    m = M_AIR_GMOL * gamma_gas  # Molar mass
    a = ((9.379 + 0.01607 * m) * t_R ** 1.5)/(209.2 + 19.26 * m + t_R)
    b = 3.448 + 986.4/t_R + 0.01009 * m
    c = 2.447 - 0.2224 * b
    ro_gas = p_MPaa * m/(z * t_K * 8.31)
    gasviscosity_cP = 10**(-4) * a * np.exp(b * ro_gas**c)
    return gasviscosity_cP


def unf_gas_fvf_m3m3(t_K:float, p_MPaa:float, z:float)->float:
    """
        Equation for gas FVF

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :param z: z-factor
    :return: formation volume factor for gas bg, m3/m3
    """

    bg = 101.33e-3 * t_K * z / (293.15 * p_MPaa) # тут от нормальный условий по температуре
    return bg


def unf_fvf_gas_vba_m3m3(T_K:float, z:float, P_MPa:float)->float:
    return 0.00034722 * T_K * z / P_MPa  # от какой температуры?


def unf_gas_density_kgm3(t_K:float, p_MPaa:float, gamma_gas:float, z:float)->float:
    """
        Equation for gas density from state equation

    :param t_K: temperature
    :param p_MPaa: pressure
    :param gamma_gas: specific gas density by air
    :param z: z-factor
    :return: gas density
    """
    m = gamma_gas * 0.029
    p_Pa = 10 ** 6 * p_MPaa
    rho_gas = p_Pa * m / (z * 8.31 * t_K)
    return rho_gas


def unf_gas_density_VBA_kgm3(gamma_gas:float, bg_m3m3:float)->float:
    return gamma_gas * RHO_AIR_kgm3 / bg_m3m3


def unf_heat_capacity_gas_Mahmood_Moshfeghian_JkgC(p_MPaa:float, t_K:float, gamma_gas:float)->float:
    """
        Gas heat capacity by Mahmood Moshfeghian for 0.1 to 20 MPa
    should be noted that the concept of heat capacity is valid only for the single phase region.

    :param p_MPaa: pressure, MPaa
    :param t_K: temperature, K
    :param gamma_gas: specific gas density by air
    :return: gas heat capacity in JkgC = JkgK

    ref1 https://www.jmcampbell.com/tip-of-the-month/2009/07/
    variation-of-natural-gas-heat-capacity-with-temperature-pressure-and-relative-density/
    """
    t_c = K_2_C(t_K)
    a = 0.9
    b = 1.014
    c = -0.7
    d = 2.170
    e = 1.015
    f = 0.0214
    return ((a * (b**t_c) * (t_c**c) + d * (e**p_MPaa) * (p_MPaa**f)) * ((gamma_gas / 0.60) ** 0.025)) * 1000


def unf_thermal_conductivity_gas_methane_WmK(t_c:float)->float: # TODO заменить
    """
        Теплопроводность метана

    :param t_c: температура в С
    :return: теплопроводность в Вт / м К

    Данная функкия является линейным приближением табличных значений при 1 бар
    требует корректировки, является временной затычкой, взята от безысходности
    """
    return (42.1 + (42.1 - 33.1)/(80-18)*(t_c - 80))/1000
# uPVT свойства для сжимаемости нефти(требует немного свойств газа)

"""
def unf_weightedcompressibility_oil_Mccain_1MPa_greater(gamma_oil, gamma_gas, pb_MPa, p_MPa, rsb_m3m3, tres_K, gamma_gassp = 0):

    pass


def unf_compressibility_oil_Mccain_1MPa_greater(gamma_oil, gamma_gas, pb_MPa, p_MPa, rsb_m3m3, tres_K, gamma_gassp = 0):

    pass


def unf_compressibility_oil_Mccain_1MPa_lower():
    pass
"""
