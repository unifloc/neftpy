from neftpy.uconvert import *
from neftpy.uconst import *

import numpy as np
import pandas as pd

import scipy.optimize as opt
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import RegularGridInterpolator
import json


# uPVT свойства для газа

""" 
====================================================================================================
Критические свойства газа
====================================================================================================
"""

def unf_pseudocritical_McCain_p_MPa_t_K(gamma_gas:float, 
                                 y_h2s:float=0.0, 
                                 y_co2:float=0.0, 
                                 y_n2:float=0.0,
                                 )->float:
    """
    Correlation for pseudocritical pressure and temperature 
    taking into account the presense of non-hydrocarbon gases
    
    Расчет псевдо критического давления и температуры 
    с учетом наличия не углеводородных примесей в газе

    :param gamma_gas: specific gas density (by air)
    :param y_h2s: mole fraction of the hydrogen sulfide
    :param y_co2: mole fraction of the carbon dioxide
    :param y_n2: mole fraction of the nitrogen
    :return: tuple with pseudocritical pressure and temperature, K

    ref 1 Piper, L.D., McCain, W.D., Jr., and Corredor, J.H. “Compressibility Factors for
    Naturally Occurring Petroleum Gases.” Gas Reservoir Engineering. Reprint Series. Richardson,
    TX: SPE. Vol. 52 (1999) 186–200
    """

    tc_h2s_R = K_2_R(373.6)         # critical temperature for hydrogen sulfide, K->R
    tc_co2_R = K_2_R(304.13)        # critical temperature for carbon dioxide, K->R
    tc_n2_R = K_2_R(126.25)         # critical temperature for nitrogen, K->R
    pc_h2s_psia = MPa_2_psi(9.007)  # critical pressure for hydrogen sulfide, MPaa->psi
    pc_co2_psia = MPa_2_psi(7.375)  # critical pressure for carbon dioxide, MPaa->psi
    pc_n2_psia = MPa_2_psi(3.4)     # critical pressure for nitrogen, MPaa->psi

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
    
    ppc_psia = tpc_R / J
    ppc_MPa = psi_2_MPa(ppc_psia)
    return (ppc_MPa, tpc_K)


def unf_pseudocritical_Standing_p_MPa_t_K(gamma_gas:float)->float:  # VBA
    tpc = 93.3 + 180 * gamma_gas - 6.94 * gamma_gas ** 2
    ppc = 4.6 + 0.1 * gamma_gas - 0.258 * gamma_gas ** 2
    return (ppc, tpc)


def unf_pseudocritical_Sutton_p_MPa_t_K(gamma_gas:float)->float:  # VBA
    """
    https://petrowiki.spe.org/Real_gases
    """
    tpc = 169.2 + 349.5 * gamma_gas - 74.0 * gamma_gas ** 2
    tpc = R_2_K(tpc)
    ppc = 756.8 - 131.07 * gamma_gas - 3.6 * gamma_gas ** 2
    ppc = psi_2_MPa(ppc)
    return (ppc, tpc)


""" 
====================================================================================================
Расчет z фактора
====================================================================================================
"""

def unf_zfactor_BrillBeggs(ppr:float, 
                           tpr:float, 
                           safe:bool=True
                           )->float:
    """
    Correlation for z-factor according Beggs & Brill correlation (1977)

    Расчет z фактора по корреляции Беггса Брилла (1977) (корреляция Standing из Pipesim)
    используется для приближения функции Дранчука (Стендинга Катцв)
    Быстрый явный расчет, но плох по краям
    Можно использовать при tpr<=2 и ppr<=4
    при tpr <== 1.5 ppr<=10

    :param ppr: preudoreduced pressure
    :param tpr: pseudoreduced temperature
    :param safe: флаг ограничений на температуру для повышения устойчивости расчета
    :return: z-factor
    """

    # для корреляции BrillBeggs крайне не рекомендуется включать safe=False
    if safe and tpr <= 1.05:
        tpr=1.05
        #TODO надо бы тут выдать в лог какое то предупреждение - может корежить результаты
    a = 1.39 * (tpr - 0.92) ** 0.5 - 0.36 * tpr - 0.101
    b = (0.62 - 0.23 * tpr) * ppr
    c = (0.066/(tpr - 0.86) - 0.037) * ppr ** 2
    d = (0.32/(10 ** (9 * (tpr - 1)))) * ppr ** 6
    e = b + c + d
    f = (0.132 - 0.32 * np.log10(tpr))
    g = 10 ** (0.3106 - 0.49 * tpr + 0.1824 * tpr ** 2)
    z = a + (1 - a) * np.exp(-e) + f * ppr ** g
    return z

def unf_zfactor_DAK(ppr:float, 
                    tpr:float, 
                    safe:bool=True
                    )->float:
    """
    Correlation for z-factor
    Расчет z фактора по методу Дранчука Абу Кассема
    Использует приближение к графикам Стандинга Катца на основе решения уравнения состояния
    Медленный расчет из за необходимости решения уравнения состояния (неявно)
    Относительно точен.
    С осторожностью можно использовать для температур ниже критической (safe=False)

    :param ppr: pseudoreduced pressure
    :param tpr: pseudoreduced temperature
    :param safe: если True то расчет ограничивается только tpr>1.05, что гарантирует корректность
    :return: z-factor

    range of applicability is (0.2<=ppr<30 and 1.0<tpr<=3.0) and also ppr < 1.0 for 0.7 < tpr < 1.0

    ref 1 Dranchuk, P.M. and Abou-Kassem, J.H. “Calculation of Z Factors for Natural
    Gases Using Equations of State.” Journal of Canadian Petroleum Technology. (July–September 1975) 34–36.
    """
    if safe and tpr <= 1.05:
        tpr=1.05
        #TODO надо бы тут выдать в лог какое то предупреждение - может корежить результаты
    def f(z, ppr, tpr):
        ropr = 0.27 * (ppr / (z * tpr)) 
        ropr2 = ropr * ropr
        func = -z + 1 + \
                (0.3265 - 1.0700 / tpr - 0.5339 / tpr**3 + 0.01569 / tpr**4 - 0.05165 / tpr**5) * ropr +\
                (0.5475 - 0.7361 / tpr + 0.1844 / tpr**2) * ropr2 - \
                0.1056 * (-0.7361 / tpr + 0.1844 / tpr**2) * ropr**5 + \
                0.6134 * (1 + 0.7210 * ropr2) * (ropr2 / tpr ** 3) * np.exp(-0.7210 * ropr2)
        return func
    if safe or tpr >= 1.05:
        solution = opt.root_scalar(f, (ppr, tpr),  method='bisect', bracket=(0.1, 5))
        solution = solution.root
    else:
        if tpr < 1.05:
            # может быть несколько решений - применяем поиск решения по сплайн аппроксимации
            z = np.logspace(-2, 0.5, 30) # 30 точек с небольшим запасом для интерполяции
            spl = PchipInterpolator(z, f(z, ppr, tpr))
            solutions = spl.roots(extrapolate=False)
            solution = max(solutions)
    # по идее расчет в безопасном диапазоне по температуре должен считать быстрее, так как требует меньше вызовов f
    # но по факту опасный расчет с определение корней по интерполированному сплайну тоже быстр, хотя требует 25 вызовов f
    # скорее всего из за того, что все вызовы делаются в векторной форме.
    # возможно далее для унификации можно отказаться от расчета
    return solution

def unf_zfactor_Kareem(ppr:float, 
                       tpr:float,
                       safe:bool=True
                       )->float:
    """
    Correlation for z-factor
    Расчет z фактора по корреляции Карима
    Использует приближение к графикам Стандинга Катца с использованием явных функций
    относительно быстрый

    С осторожностью можно использовать для температур ниже критической (safe=False)
    based on  https://link.springer.com/article/10.1007/s13202-015-0209-3
    Kareem, L.A., Iwalewa, T.M. & Al-Marhoun, M.
    New explicit correlation for the compressibility factor of natural gas: linearized z-factor isotherms.
    J Petrol Explor Prod Technol 6, 481–492 (2016).
    https://doi.org/10.1007/s13202-015-0209-3

    :param ppr: pseudoreduced pressure
    :param tpr: pseudoreduced temperature
    :param safe: если True то расчет ограничивается только tpr>1.05, что гарантирует корректность
    :return:
    """

    # для корреляции Карима крайне не рекомендуется включать safe=False
    if safe and tpr <= 1.05:
        tpr=1.05
        #TODO надо бы тут выдать в лог какое то предупреждение - может корежить результаты

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

    t = 1 / tpr
    t2 = t * t
    t3 = t2 * t
    AA = a[1]* t * np.exp(a[2] * (1 - t) ** 2) * ppr
    BB = a[3]* t + a[4] * t2 + a[5] * t3*t3 * ppr ** 6
    CC = a[9]+ a[8] * t * ppr + a[7] * t2 * ppr ** 2 + a[6] * t3 * ppr ** 3
    DD = a[10] * t * np.exp(a[11] * (1 - t) ** 2)
    EE = a[12] * t + a[13] * t2 + a[14] * t3
    FF = a[15] * t + a[16] * t2 + a[17] * t3
    GG = a[18] + a[19] * t

    DPpr = DD * ppr
    y = DPpr / ((1 + AA ** 2) / CC - AA ** 2 * BB / (CC ** 3))

    z = DPpr * (1 + y + y ** 2 - y ** 3) / (DPpr + EE * y ** 2 - FF * y ** GG) / ((1 - y) ** 3)

    return z

def _load_StandingKatz_curve_():
    """
    загружает табулированные значения кривых с графика Standing Katz из набора файлов в папке data
    возвращает DataFrame загруженных данных и функцию линейной интерполяции по загруженным данным
    """
    Standing_Katz_Chart_Data = '''
    {
"1.05":{
      "ppr": [0.0, 0.204, 0.3, 0.405, 0.504, 0.602, 0.7, 1.0, 1.203, 1.301, 1.332, 1.351, 1.357, 1.363, 1.37, 1.372, 1.373, 
              1.378, 1.386, 1.397, 1.441, 1.5, 1.601, 1.7, 1.753, 1.801, 1.849, 1.9, 1.951, 2.0, 2.102, 2.503, 3.005, 3.509, 
              4.007, 4.507, 5.007, 5.207, 5.307, 5.505, 5.703, 6.004, 6.507, 7.002, 7.104, 7.201, 7.303, 7.402, 7.506, 7.604, 
              7.802, 7.941, 8.0, 8.1, 8.202, 8.4, 8.537, 8.7, 9.001, 9.505, 10.003, 12.404, 15.0], 
      "z": [1.0, 0.937, 0.905, 0.866, 0.829, 0.79, 0.748, 0.589, 0.44, 0.35, 0.319, 0.3, 0.29, 0.285, 0.279, 0.274, 0.27, 
            0.267, 0.264, 0.262, 0.257, 0.253, 0.251, 0.252, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.291, 0.343, 0.407, 
            0.471, 0.534, 0.598, 0.663, 0.688, 0.701, 0.727, 0.75, 0.786, 0.846, 0.904, 0.916, 0.928, 0.939, 0.95, 0.962, 
            0.973, 0.995, 1.011, 1.018, 1.028, 1.039, 1.06, 1.074, 1.092, 1.125, 1.178, 1.231, 1.483, 1.751], 
      "tpr": 1.05
      }, 
"1.1": {
      "ppr": [0.0, 0.225, 0.253, 0.279, 0.321, 0.349, 0.378, 0.415, 0.444, 0.456, 0.504, 0.539, 0.567, 0.628, 0.651, 0.676, 
              0.735, 0.757, 0.833, 0.843, 0.863, 0.939, 0.968, 1.004, 1.044, 1.067, 1.081, 1.129, 1.151, 1.169, 1.231, 1.251, 
              1.273, 1.3, 1.327, 1.357, 1.37, 1.403, 1.428, 1.447, 1.467, 1.5, 1.527, 1.55, 1.583, 1.601, 1.63, 1.655, 1.7, 1.744, 
              1.8, 1.85, 1.902, 1.953, 2.004, 2.102, 2.204, 2.301, 2.403, 2.5, 2.605, 2.656, 2.753, 2.855, 2.952, 3.001, 3.054, 
              3.141, 3.206, 3.305, 3.404, 3.504, 3.607, 4.005, 4.505, 5.006, 5.506, 6.004, 6.507, 6.904, 7.002, 7.206, 7.303, 7.403, 
              7.507, 7.801, 8.0, 8.301, 10.001, 11.008, 13.301, 14.802, 15.0], 
      "z": [1.0, 0.942, 0.933, 0.926, 0.912, 0.903, 0.895, 0.882, 0.874, 0.869, 0.854, 0.842, 0.832, 0.811, 0.802, 0.793, 0.772, 
            0.764, 0.736, 0.733, 0.724, 0.694, 0.684, 0.669, 0.654, 0.643, 0.637, 0.615, 0.604, 0.594, 0.565, 0.554, 0.544, 0.53, 
            0.516, 0.503, 0.494, 0.476, 0.463, 0.454, 0.444, 0.426, 0.412, 0.405, 0.396, 0.393, 0.386, 0.382, 0.377, 0.373, 0.37, 
            0.369, 0.368, 0.368, 0.369, 0.372, 0.376, 0.38, 0.387, 0.393, 0.401, 0.405, 0.414, 0.424, 0.434, 0.44, 0.445, 0.456, 
            0.464, 0.476, 0.487, 0.5, 0.511, 0.557, 0.615, 0.673, 0.729, 0.784, 0.841, 0.886, 0.897, 0.92, 0.931, 0.942, 0.952,   
            0.985, 1.006, 1.037, 1.209, 1.31, 1.543, 1.697, 1.717], 
      "tpr": 1.1
      }, 
"1.2": {
      "ppr": [0.0, 0.2, 0.3, 0.4, 0.504, 0.547, 0.634, 0.755, 0.847, 1.0, 1.146, 1.267, 1.351, 1.435, 1.503, 1.549, 1.643, 1.734, 
              1.837, 1.943, 2.0, 2.052, 2.103, 2.201, 2.25, 2.302, 2.501, 2.593, 2.657, 2.75, 2.833, 3.005, 3.203, 3.507, 3.705, 
              3.802, 3.905, 4.005, 4.106, 4.203, 4.342, 4.505, 4.658, 4.872, 5.007, 5.128, 5.239, 5.357, 5.509, 5.659, 5.758, 5.854, 
              6.001, 6.507, 7.004, 7.1, 7.289, 7.355, 7.5, 7.901, 8.001, 8.602, 10.305, 10.605, 11.91, 13.502, 14.603, 14.903, 15.0], 
      "z": [1.0, 0.958, 0.938, 0.916, 0.893, 0.883, 0.864, 0.836, 0.815, 0.779, 0.744, 0.715, 0.695, 0.674, 0.657, 0.646, 0.625, 
            0.605, 0.584, 0.565, 0.554, 0.545, 0.538, 0.528, 0.524, 0.522, 0.519, 0.519, 0.52, 0.523, 0.526, 0.534, 0.545, 0.565, 
            0.581, 0.589, 0.599, 0.607, 0.616, 0.624, 0.636, 0.65, 0.663, 0.683, 0.695, 0.706, 0.716, 0.726, 0.741, 0.755, 0.765, 
            0.774, 0.789, 0.841, 0.891, 0.901, 0.922, 0.926, 0.941, 0.98, 0.99, 1.045, 1.203, 1.231, 1.352, 1.501, 1.605, 1.633, 1.642], 
      "tpr": 1.2
       }, 
"1.3": {
      "ppr": [0.0, 0.201, 0.302, 0.401, 0.504, 0.6, 0.804, 1.002, 1.202, 1.303, 1.503, 1.802, 2.009, 2.203, 2.504, 2.704, 3.008, 
              3.204, 3.509, 3.808, 4.001, 4.308, 4.509, 4.809, 5.009, 5.308, 5.506, 5.806, 6.004, 6.207, 6.307, 6.503, 7.005, 7.29, 
              7.309, 7.504, 7.906, 8.001, 10.361, 10.707, 14.803, 15.0], 
      "z": [1.0, 0.967, 0.95, 0.935, 0.916, 0.901, 0.868, 0.835, 0.804, 0.788, 0.756, 0.713, 0.684, 0.663, 0.638, 0.629, 0.624, 
            0.626, 0.633, 0.645, 0.653, 0.671, 0.684, 0.705, 0.719, 0.743, 0.759, 0.786, 0.802, 0.82, 0.829, 0.844, 0.892, 0.917, 
            0.916, 0.934, 0.969, 0.979, 1.182, 1.212, 1.567, 1.585], 
      "tpr": 1.3
       }, 
"1.4": {
      "ppr": [0.0, 0.267, 0.344, 0.503, 0.744, 1.001, 1.402, 1.502, 1.905, 2.003, 2.132, 2.243, 2.358, 2.503, 2.607, 2.746, 2.84, 
              2.957, 3.005, 3.062, 3.103, 3.203, 3.304, 3.404, 3.506, 3.611, 3.707, 3.807, 3.905, 4.006, 4.258, 4.344, 4.507, 4.754, 
              5.007, 5.262, 5.408, 5.507, 6.005, 6.507, 6.748, 6.804, 7.002, 7.302, 7.356, 7.503, 7.836, 7.903, 8.002, 12.907, 13.805, 
              14.304, 14.504, 15.0], 
      "z": [1.0, 0.965, 0.955, 0.936, 0.906, 0.874, 0.827, 0.816, 0.774, 0.764, 0.753, 0.744, 0.736, 0.727, 0.722, 0.715, 0.712, 
            0.708, 0.707, 0.705, 0.705, 0.704, 0.704, 0.704, 0.705, 0.706, 0.708, 0.71, 0.713, 0.716, 0.724, 0.727, 0.734, 0.746, 
            0.76, 0.775, 0.785, 0.792, 0.828, 0.865, 0.884, 0.888, 0.903, 0.928, 0.931, 0.942, 0.968, 0.974, 0.981, 1.364, 1.433, 
            1.472, 1.487, 1.526], 
      "tpr": 1.4
       }, 
"1.5": {
      "ppr": [0.0, 0.202, 0.303, 0.404, 0.506, 0.701, 0.801, 1.0, 1.304, 1.505, 1.803, 2.006, 2.103, 2.504, 2.705, 3.006, 3.104, 
            3.509, 3.806, 4.006, 4.209, 4.505, 4.604, 5.008, 5.307, 5.508, 5.806, 6.005, 6.307, 6.502, 6.703, 7.003, 7.305, 7.506, 
            7.707, 7.804, 7.904, 8.001, 12.808, 13.207, 13.907, 14.304, 14.903, 15.0], 
      "z": [1.0, 0.978, 0.967, 0.958, 0.948, 0.929, 0.919, 0.9, 0.875, 0.859, 0.836, 0.822, 0.816, 0.794, 0.786, 0.776, 0.774, 0.77, 
            0.773, 0.777, 0.781, 0.79, 0.794, 0.81, 0.825, 0.836, 0.851, 0.863, 0.88, 0.892, 0.904, 0.924, 0.943, 0.956, 0.97, 0.977, 
            0.984, 0.99, 1.334, 1.362, 1.412, 1.441, 1.485, 1.492], 
      "tpr": 1.5
       }, 
"1.6": {
      "ppr": [0.0, 0.226, 0.502, 1.0, 1.505, 1.704, 2.0, 2.303, 2.502, 2.606, 2.707, 3.006, 3.204, 3.407, 3.506, 3.708, 3.807, 
            4.006, 4.207, 4.508, 4.706, 4.809, 5.006, 5.211, 5.309, 5.503, 5.707, 6.005, 6.506, 7.0, 7.504, 7.6, 8.0, 15.0], 
      "z": [1.0, 0.981, 0.959, 0.923, 0.888, 0.876, 0.86, 0.847, 0.839, 0.835, 0.832, 0.824, 0.82, 0.817, 0.816, 0.815, 0.815,
             0.818, 0.821, 0.829, 0.835, 0.838, 0.846, 0.854, 0.858, 0.868, 0.877, 0.892, 0.918, 0.946, 0.974, 0.979, 1.002, 1.46],
      "tpr": 1.6
       }, 
"1.7": {
      "ppr": [0.0, 0.198, 0.297, 0.396, 0.497, 0.694, 0.795, 0.996, 1.096, 1.303, 1.497, 1.696, 1.899, 2.0, 2.196, 2.501, 2.6, 
            2.902, 3.004, 3.196, 3.502, 3.604, 3.804, 4.0, 4.101, 4.302, 4.504, 4.596, 5.002, 5.203, 5.502, 5.701, 6.005, 6.201, 
            6.5, 6.799, 7.003, 7.201, 7.4, 7.598, 8.0, 14.801, 15.0], 
      "z": [1.0, 0.986, 0.979, 0.974, 0.968, 0.957, 0.951, 0.941, 0.935, 0.924, 0.914, 0.905, 0.896, 0.892, 0.885, 0.876, 0.873, 
            0.866, 0.863, 0.861, 0.857, 0.856, 0.855, 0.856, 0.857, 0.86, 0.864, 0.868, 0.879, 0.886, 0.897, 0.905, 0.918, 0.928, 
            0.942, 0.956, 0.966, 0.975, 0.985, 0.996, 1.018, 1.427, 1.439], 
      "tpr": 1.7}, 
"1.8": {
      "ppr": [0.0, 0.504, 1.003, 1.501, 2.0, 2.505, 3.003, 3.507, 4.005, 4.503, 5.006, 5.504, 6.002, 6.505, 7.003, 7.5, 8.003, 15.0], 
      "z": [1.0, 0.974, 0.952, 0.933, 0.917, 0.905, 0.896, 0.891, 0.893, 0.901, 0.913, 0.929, 0.947, 0.967, 0.988, 1.01, 1.034, 1.42], 
      "tpr": 1.8
       }, 
"1.9": {
      "ppr": [0.0, 0.503, 1.003, 1.507, 2.005, 2.504, 3.008, 3.506, 4.004, 4.502, 5.006, 5.504, 6.007, 6.505, 7.002, 7.505, 8.002, 15.0], 
      "z": [1.0, 0.978, 0.96, 0.945, 0.933, 0.924, 0.917, 0.916, 0.917, 0.924, 0.935, 0.949, 0.966, 0.985, 1.005, 1.026, 1.047, 1.406], 
      "tpr": 1.9
       }, 
"2": {
      "ppr": [0.0, 0.3, 0.403, 0.505, 0.702, 0.801, 1.002, 1.2, 1.501, 1.602, 1.805, 2.001, 2.104, 2.205, 2.504, 2.903, 3.006, 3.108, 
            3.506, 3.605, 3.806, 4.01, 4.405, 4.506, 4.605, 5.009, 5.106, 5.408, 5.505, 5.708, 6.008, 6.305, 6.504, 6.802, 7.004, 
            7.205, 7.505, 7.701, 8.005, 8.203, 13.006, 14.309, 15.0], 
      "z": [1.0, 0.989, 0.985, 0.982, 0.977, 0.974, 0.969, 0.963, 0.956, 0.954, 0.95, 0.947, 0.945, 0.944, 0.941, 0.937, 0.937, 0.937, 
            0.937, 0.937, 0.938, 0.939, 0.944, 0.945, 0.947, 0.955, 0.957, 0.966, 0.969, 0.976, 0.986, 0.996, 1.003, 1.014, 1.021, 
            1.028, 1.039, 1.046, 1.059, 1.066, 1.291, 1.354, 1.388], 
      "tpr": 2
       }, 
"2.2": {
      "ppr": [0.0, 0.502, 1.0, 1.503, 2.003, 2.501, 3.002, 3.104, 3.2, 3.3, 3.503, 3.803, 4.006, 4.505, 5.006, 5.504, 6.003, 6.505, 
            7.002, 7.506, 7.602, 8.001, 8.2, 8.755, 13.006, 14.006, 14.907, 15.0], 
      "z": [1.0, 0.989, 0.98, 0.973, 0.967, 0.963, 0.961, 0.961, 0.961, 0.962, 0.963, 0.965, 0.968, 0.976, 0.987, 1.0, 1.014, 1.029, 
            1.044, 1.06, 1.064, 1.077, 1.083, 1.103, 1.28, 1.32, 1.356, 1.36], 
      "tpr": 2.2
       }, 
"2.4": {
      "ppr": [0.0, 0.402, 0.504, 0.602, 1.0, 1.201, 1.401, 1.502, 1.704, 1.9, 2.004, 2.504, 3.008, 3.102, 3.506, 3.601, 3.705, 3.808, 
            4.006, 4.306, 4.504, 4.708, 5.004, 5.207, 5.503, 5.606, 6.001, 6.506, 6.703, 7.004, 7.102, 7.5, 7.801, 7.901, 8.0, 8.502, 
            8.801, 9.102, 9.402, 9.601, 14.208, 15.0], 
      "z": [1.0, 0.994, 0.993, 0.992, 0.988, 0.986, 0.985, 0.984, 0.982, 0.981, 0.981, 0.98, 0.98, 0.981, 0.983, 0.984, 0.986, 0.987, 
            0.99, 0.995, 0.999, 1.004, 1.011, 1.016, 1.023, 1.026, 1.036, 1.049, 1.054, 1.062, 1.065, 1.076, 1.085, 1.087, 1.091, 
            1.105, 1.115, 1.126, 1.137, 1.145, 1.317, 1.346], 
      "tpr": 2.4
       }, 
"2.6": {
      "ppr": [0.0, 0.502, 1.001, 1.5, 2.002, 2.505, 3.004, 3.505, 4.003, 4.504, 5.004, 5.506, 6.003, 6.501, 7.001, 7.4, 7.502, 7.901, 
            14.106, 14.606, 14.905, 15.0], 
      "z": [1.0, 0.997, 0.995, 0.994, 0.994, 0.994, 0.996, 1.0, 1.007, 1.016, 1.026, 1.038, 1.05, 1.062, 1.074, 1.084, 1.086, 1.097, 
            1.306, 1.324, 1.334, 1.338], 
      "tpr": 2.6
       }, 
"2.8": {
      "ppr": [0.0, 0.501, 1.002, 1.253, 1.503, 2.003, 2.501, 3.005, 3.504, 4.005, 4.506, 5.006, 5.506, 6.004, 6.504, 7.004, 7.5, 15.0], 
      "z": [1.0, 0.999, 1.0, 1.001, 1.002, 1.005, 1.008, 1.011, 1.016, 1.022, 1.03, 1.039, 1.049, 1.059, 1.069, 1.081, 1.093, 1.333], 
      "tpr": 2.8
       }, 
"3": {
      "ppr": [0.0, 0.503, 1.0, 1.501, 2.002, 2.502, 3.005, 3.102, 3.504, 4.004, 4.507, 5.005, 5.501, 6.003, 6.504, 7.001, 7.601, 
            7.801, 8.2, 14.102, 14.805, 15.0], 
      "z": [1.0, 1.002, 1.006, 1.009, 1.013, 1.018, 1.023, 1.024, 1.029, 1.034, 1.041, 1.048, 1.056, 1.065, 1.075, 1.086, 1.1, 
            1.104, 1.115, 1.302, 1.325, 1.332], 
      "tpr": 3
       }
}
    '''
    
    #with open(r'data\Standing_Katz_Chart_Data.json', 'r') as fp:
    #    dict_SK = json.load( fp)
    dict_SK = json.loads(Standing_Katz_Chart_Data)
    tpr = list(dict_SK.keys())
    
    df = pd.DataFrame()
    try:
        for k in dict_SK.keys():
            
            ppr = np.array(dict_SK[k]['ppr'])
            z = np.array(dict_SK[k]['z'])
            t = dict_SK[k]['tpr']
            df_SK_t = pd.DataFrame({f"tpr = {t}":z}, index = ppr)
            if df.shape[0]==0:
                df = df_SK_t.copy()
            else:
                df =df.join(df_SK_t, how='outer')
            df = df.sort_index().interpolate(method='index')
        # пробуем сделать интерполятор на регулярной сетке
        ppr = np.array(df.index)
        tpr = np.array(tpr)
        data = np.array(df)
        interp = RegularGridInterpolator((ppr, tpr), data,
                                         bounds_error=False, 
                                         fill_value=None)
        return {'df':df, 'interp':interp}
    except Exception as e:
        print(e)

def unf_zfactor_SK(ppr:float, 
                   tpr:float,
                   safe:bool=True
                   )->float:
    """
    Correlation for z-factor
    Расчет z фактора по графикам Стандинга Катца на основе табулированных данных
    Очень быстрый.
    Относительно точен в зоне определения графиков.
    при safe=False может экстраполировать за границы и сильно врать для низких температур

    :param ppr: pseudoreduced pressure
    :param tpr: pseudoreduced temperature
    :param safe: если True то расчет ограничивается только tpr>1.05, что гарантирует корректность
    :return:
    """
    zSk = StandingKatz_curve['interp']
    if safe and tpr < 1.05:
        tpr = 1.05
    return zSk((ppr, tpr), method='linear')

def unf_zfactor(p_MPaa:float, 
                t_K:float, 
                gamma_gas:float,
                method_z:str='Standing_Katz',
                method_crit:str='Sutton',
                y_h2s:float=0.0, 
                y_co2:float=0.0, 
                y_n2:float=0.0,
                safe_z:bool=True
                )->float:
    """
    General function for z-factor estimation by correlation
    Обобщающая функция расчетр z фактора с возможность выбора метода расчета z фактора
    и метода расчета критических параметров газа

    :param p_MPaa: Давление газа, МПа
    :param t_K: Температура газа, К
    :param gamma_gas: удельная плотность газа по воздуху
    :param method_z: метод расчета z фактора (Standing_Katz, DAK, Kareem, BrillBeggs)
    :param method_crit: метод расчета критического давления и температуры (Standing, McCain)
    :param y_h2s: мольная доля H2S
    :param y_co2: мольная доля CO2
    :param y_n2: мольная доля N2
    :param safe: если True то расчет ограничивается только tpr>1.05, что гарантирует корректность
    :return:
    """
    # оценим псевдо критичесие давление и температуру
    if method_crit=='Standing':
        (ppc_MPa, tpc_K) = unf_pseudocritical_Standing_p_MPa_t_K(gamma_gas=gamma_gas)
    elif method_crit=='Sutton':
        (ppc_MPa, tpc_K) = unf_pseudocritical_Sutton_p_MPa_t_K(gamma_gas=gamma_gas)
    else:
        (ppc_MPa,tpc_K) = unf_pseudocritical_McCain_p_MPa_t_K(gamma_gas=gamma_gas,y_h2s=y_h2s,y_co2=y_co2,y_n2=y_n2)

    # оценим приведенные давление и температуру
    ppr = p_MPaa / ppc_MPa
    tpr = t_K / tpc_K

    #оценим величину z фактора
    if method_z=='Standing_Katz':
        z = unf_zfactor_SK(ppr=ppr, tpr=tpr, safe=safe_z)
    elif method_z=='DAK':
        z = unf_zfactor_DAK(ppr=ppr, tpr=tpr, safe=safe_z)
    elif method_z=='Kareem':
        z = unf_zfactor_Kareem(ppr=ppr, tpr=tpr, safe=safe_z)
    elif method_z=='BrillBeggs':
        z = unf_zfactor_BrillBeggs(ppr=ppr, tpr=tpr, safe=safe_z)
    
    return z

def unf_dzdp(p_MPaa:float, 
                t_K:float, 
                gamma_gas:float,
                method_z:str='Standing_Katz',
                method_crit:str='Sutton',
                y_h2s:float=0.0, 
                y_co2:float=0.0, 
                y_n2:float=0.0,
                safe_z:bool=True
                )->float:
    """
    расчет производной от z фактора по давлению
    """
    
    dp = 0.1 
    
    z1 = unf_zfactor(p_MPaa, t_K, gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    z2 = unf_zfactor(p_MPaa+dp, t_K, gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    return (z2 - z1) / dp
   

def unf_dzdt(p_MPaa:float, 
                t_K:float, 
                gamma_gas:float,
                method_z:str='Standing_Katz',
                method_crit:str='Sutton',
                y_h2s:float=0.0, 
                y_co2:float=0.0, 
                y_n2:float=0.0,
                safe_z:bool=True
                )->float:
    """
    расчет производной от z фактора по температуре
    """
    
    dt = 0.01 
    
    z1 = unf_zfactor(p_MPaa, t_K, gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    z2 = unf_zfactor(p_MPaa, t_K + dt, gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    return (z2 - z1) / dt
   

""" 
====================================================================================================
Расчет вязкости газа
====================================================================================================
"""

def unf_mu_gas_cP(p_atma:float, 
                  t_C:float, 
                  gamma_gas:float,
                  method_z:str='Standing_Katz',
                  safe_z:bool=True
                )->float:
    """
     Lee correlation for gas viscosity
     Метод расчета вязкости газа по корреляции Lee
     включает расчет z фактора

    :param p_atma: pressure, MPaa
    :param t_C: temperature, K
    :param gamma_gas: specific gas density (by air)
    :param method_z: method z-factor
    :param safe_z: safe_z 
    :return: gas viscosity,cP

    ref 1 Lee, A.L., Gonzalez, M.H., and Eakin, B.E. “The Viscosity of Natural Gases.” Journal
    of Petroleum Technology. Vol. 18 (August 1966) 997–1,000.
    """
    
    p_MPaa=atm_2_MPa(p_atma)
    t_K=C_2_K(t_C)
    z = unf_zfactor(p_MPaa=p_MPaa, 
                    t_K=t_K, 
                    gamma_gas=gamma_gas,
                    method_z=method_z, 
                    safe_z=safe_z)
    return unf_mu_gas_Lee_z_cP(p_MPaa=p_MPaa,
                               t_K=t_K,
                               gamma_gas=gamma_gas,
                               z=z)

def unf_mu_gas_Lee_z_cP(p_MPaa:float, 
                        t_K:float,
                        gamma_gas:float,
                        z:float,
                        )->float:
    """
     Lee correlation for gas viscosity
     Метод расчета вязкости газа по корреляции Lee

    :param p_MPaa: pressure, MPaa
    :param t_K: temperature, K
    :param gamma_gas: specific gas density (by air)
    :param z: z-factor
    :return: gas viscosity,cP

    ref 1 Lee, A.L., Gonzalez, M.H., and Eakin, B.E. “The Viscosity of Natural Gases.” Journal
    of Petroleum Technology. Vol. 18 (August 1966) 997–1,000.
    """

    t_R = K_2_R(t_K)

    m = M_AIR_GMOL * gamma_gas  # молярная масса газа
    a = ((9.4 + 0.02 * m) * t_R ** 1.5)/(209 + 19 * m + t_R)
    b = 3.5 + 986/t_R + 0.01 * m
    c = 2.4 - 0.2 * b

    ro_gas = unf_rho_gas_z_kgm3(p_MPaa, t_K, gamma_gas, z) * 1e-3
    return 1e-4 * a * np.exp(b * ro_gas**c)

def unf_mu_gas_Lee_z_cP_(p_MPaa:float, 
                         t_K:float,
                         gamma_gas:float,
                         z:float,
                         )->float:
    """
     Lee correlation for gas viscosity
     Метод расчета вязкости газа по корреляции Lee
     коэффициенты как в VBA - точнее чем в статье, но не понятно откуда (из справочника ЮКОСа?)

    :param p_MPaa: pressure, MPaa
    :param t_K: temperature, K
    :param gamma_gas: specific gas density (by air)
    :param z: z-factor
    :return: gas viscosity,cP

    ref 1 Lee, A.L., Gonzalez, M.H., and Eakin, B.E. “The Viscosity of Natural Gases.” Journal
    of Petroleum Technology. Vol. 18 (August 1966) 997–1,000.
    """

    t_R = K_2_R(t_K)

    m = M_AIR_GMOL * gamma_gas  # молярная масса газа
    a = ((9.379 + 0.01607 * m) * t_R ** 1.5)/(209.2 + 19.26 * m + t_R)
    b = 3.448 + 986.4/t_R + 0.01009 * m
    c = 2.447 - 0.2224 * b

    ro_gas = unf_rho_gas_z_kgm3(p_MPaa, t_K, gamma_gas, z) * 1e-3
    return 1e-4 * a * np.exp(b * ro_gas**c)

def unf_mu_gas_Lee_rho_cP(t_K: float, 
                        gamma_gas: float, 
                        rho_gas_kgm3: float
                        ) -> float:
    """
    Метод расчета вязкости газа по корреляции Lee
    корреляция Lee как в Pipesim
    раняя версия корреляции - описана в статье Ли

    ref 1 Lee, A.L., Gonzalez, M.H., and Eakin, B.E. “The Viscosity of Natural Gases.” Journal
    of Petroleum Technology. Vol. 18 (August 1966) 997–1,000.

    Parameters
    ----------
    :param t: температура, К
    :param gamma_gas: относительная плотность газа, доли,
    (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
    :param rho_gas: плотность газа при данном давлении температуре, кг/м3
    :return: вязкость газа, сПз
    -------
    """

    t_R = K_2_R(t_K)

    # Новая корреляция Lee
    # m = 28.9612403 * gamma_gas  # молярная масса газа
    # a = ((9.379 + 0.01607 * m) * t_r ** 1.5) / (209.2 + 19.26 * m + t_r)
    # b = 3.448 + 986.4 / t_r + 0.01009 * gamma_gas * 28.966
    # c = 2.447 - 0.2224 * b
    # Старая корреляция Lee как в Pipesim
    m = 28.9612403 * gamma_gas  # молярная масса газа
    a = (7.77 + 0.183 * gamma_gas) * t_R**1.5 / (122.4 + 373.6 * gamma_gas + t_R)
    b = 2.57 + 1914.5 / t_R + 0.275 * gamma_gas
    c = 1.11 + 0.04 * b
    gas_viscosity = 1e-4 * a * np.exp(b * (rho_gas_kgm3 / 1000) ** c)
    return gas_viscosity

""" 
====================================================================================================
Расчет объемного коэффициента газа
====================================================================================================
"""

def unf_bg_gas_z_m3m3(p_MPaa:float, 
                       t_K:float,
                       z:float)->float:
    """
    Equation for gas FVF based on z

    Расчет объемного коэффициента через z фактор
    расчет z может быть затратным, чтобы пересчитывать каждый раз, а он него много чего зависит
    тут можно использовать уже посчитанный ранее z

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :param z: z-factor
    :return: formation volume factor for gas bg, m3/m3
    """
      # z фактор при стандартных условиях берем из констант равный 1
      # хотя он немного отличается - можно было бы и учесть может быть?

    return P_SC_MPa * t_K * z / (Z_SC * T_SC_K * p_MPaa) # тут от нормальный условий по температуре

def unf_bg_gas_m3m3(p_atma:float, 
                    t_C:float,
                    gamma_gas:float,
                    method_z:str='Standing', 
                    safe_z:bool=True
                    )->float:
    """
    Equation for gas FVF based on gamma_gas

    Расчет объемного коэффициента через удельную плотность газа
    включает расчет z фактора

    :param t_K: temperature, K
    :param p_MPaa: pressure, MPaa
    :param z: z-factor
    :return: formation volume factor for gas bg, m3/m3
    """
    p_MPaa=atm_2_MPa(p_atma)
    t_K=C_2_K(t_C)
    z = unf_zfactor(p_MPaa=p_MPaa, 
                    t_K=t_K, 
                    gamma_gas=gamma_gas,
                    method_z=method_z, 
                    safe_z=safe_z)
    return unf_bg_gas_z_m3m3(p_MPaa=p_MPaa, t_K=t_K, z=z)

""" 
====================================================================================================
Расчет плотности газа
====================================================================================================
"""

def unf_rho_gas_z_kgm3(p_MPaa:float, 
                       t_K:float, 
                       gamma_gas:float, 
                       z:float
                       )->float:
    """
    выражение для плотности воздуха на основе уравнения состояния
    """

    mmol_kgmol = gamma_gas * M_AIR_KGMOL
    return p_MPaa * const.mega * mmol_kgmol / (z * const.R * t_K)

def unf_rho_gas_kgm3(p_atma:float, 
                     t_C:float,
                     gamma_gas:float,
                     method_z:str='Standing')->float:
    """
    выражение для плотности воздуха на основе уравнения состояния
    """

    p_MPaa=atm_2_MPa(p_atma)
    t_K=C_2_K(t_C)
    z = unf_zfactor(p_MPaa=p_MPaa, 
                    t_K=t_K, 
                    gamma_gas=gamma_gas,
                    method_z=method_z)
    
    return unf_rho_gas_z_kgm3(p_MPaa, t_K, gamma_gas, z)

def unf_rho_gas_bg_kgm3(gamma_gas:float, 
                        bg_m3m3:float
                       )->float:
    return gamma_gas * RHO_AIR_kgm3 / bg_m3m3

""" 
====================================================================================================
Теплофизические свойств газа
====================================================================================================
"""

def unf_heat_capacity_gas_Mahmood_Moshfeghian_JkgC(p_MPaa:float, 
                                                   t_K:float, 
                                                   gamma_gas:float
                                                   )->float:
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
    t_C = K_2_C(t_K)
    a = 0.9
    b = 1.014
    c = -0.7
    d = 2.170
    e = 1.015
    f = 0.0214
    return ((a * (b**t_C) * (t_C**c) + d * (e**p_MPaa) * (p_MPaa**f)) * ((gamma_gas / 0.60) ** 0.025)) * 1000



def unf_thermal_conductivity_gas_methane_WmK(t_C:float)->float: # TODO заменить
    """
        Теплопроводность метана

    :param t_c: температура в С
    :return: теплопроводность в Вт / м К

    Данная функкия является линейным приближением табличных значений при 1 бар
    требует корректировки, является временной затычкой, взята от безысходности
    """
    return (42.1 + (42.1 - 33.1)/(80-18)*(t_C - 80))/1000

def unf_gas_ideal_heat_capacity_ratio(gamma_gas:float, 
                                t_K:float
                                )->float:
    """
    http://www.jmcampbell.com/tip-of-the-month/2013/05/variation-of-ideal-gas-heat-capacity-ratio-with-temperature-and-relative-density/
     eq 6
     temp range - 25C to 150 C
     gg range 0.55 to 2
    """
    
    return (1.6 - 0.44 * gamma_gas + 0.097 * gamma_gas * gamma_gas) * (1 + 0.0385 * gamma_gas -  0.000286 * t_K)
    





# =================================================================================================

def unf_gas_isotermal_compressibility_1MPa(p_MPaa:float, 
                                            t_K:float, 
                                            gamma_gas:float,
                                            method_z:str='Standing_Katz',
                                            method_crit:str='Sutton',
                                            y_h2s:float=0.0, 
                                            y_co2:float=0.0, 
                                            y_n2:float=0.0,
                                            safe_z:bool=True
                                            )->float:
    """
    расчет изотермической сжимаемости газа на основе z фактора
    """
    z = unf_zfactor(p_MPaa, t_K,  gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    dzdp = unf_dzdp(p_MPaa, t_K,  gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    betta_t = 1/p_MPaa - 1/z *dzdp
    return betta_t 


def unf_gas_thermal_expansion_1K(p_MPaa:float, 
                                            t_K:float, 
                                            gamma_gas:float,
                                            method_z:str='Standing_Katz',
                                            method_crit:str='Sutton',
                                            y_h2s:float=0.0, 
                                            y_co2:float=0.0, 
                                            y_n2:float=0.0,
                                            safe_z:bool=True
                                            )->float:
    """
    расчет изотермической сжимаемости газа на основе z фактора
    """
    z = unf_zfactor(p_MPaa, t_K,  gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    dzdt =unf_dzdt(p_MPaa, t_K,  gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    alpha = 1/t_K + 1/z *dzdt
    return alpha



def unf_cp_gas_JkgC(p_MPaa: float, 
                    t_K: float, 
                    gamma_gas: float
                    ) -> float:
    """
    Расчет удельной теплоемкости газа при постоянном давлении
    по упрощенной корреляции на основе расчетов уравнения состояния



    Parameters
    ----------
    :param p: давление, МПа
    :param t: температура, К
    :param gamma_gas: относительная плотность газа, доли,
    (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

    Returns
    -------
    Удельная теплоемкость газа при постоянном давлении, Дж/(кг*К)

    Ref:
    Mahmood Moshfeghian Petroskills
    https://www.jmcampbell.com/tip-of-the-month/2009/07/variation-of-natural-gas-heat-capacity-with-temperature-pressure-and-relative-density/
    """
    t_C = K_2_C(t_K)
    a = 0.9
    b = 1.014
    c = -0.7
    d = 2.170
    e = 1.015
    f = 0.0214

    return (a * b**t_C * t_C**c + d * e**p_MPaa *  p_MPaa**f) * (gamma_gas / 0.6) ** 0.025 * 1000


def  unf_cv_gas_JkgC(p_MPaa:float, 
                        t_K:float, 
                        gamma_gas:float,
                        method_z:str='Standing_Katz',
                        method_crit:str='Sutton',
                        y_h2s:float=0.0, 
                        y_co2:float=0.0, 
                        y_n2:float=0.0,
                        safe_z:bool=True
                        )->float:

    dp = 0.1 
    
    z = unf_zfactor(p_MPaa, t_K, gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    z2 = unf_zfactor(p_MPaa+dp, t_K, gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    betta_t_1MPa = 1/p_MPaa - 1/z *  (z2 - z) / dp
    betta_t_1Pa = betta_t_1MPa / 1e6

    dt = 0.01 
    
    z2 = unf_zfactor(p_MPaa, t_K + dt, gamma_gas, method_z, method_crit, y_h2s, y_co2, y_n2, safe_z)
    alpha_1K = 1/t_K + 1/z *(z2 - z) / dt


    rho_kgm3 = unf_rho_gas_z_kgm3(p_MPaa, t_K, gamma_gas, z)

    cp_JkgC = unf_cp_gas_JkgC(p_MPaa, t_K, gamma_gas)

    cv_gas_JkgC = cp_JkgC - t_K * alpha_1K**2 / betta_t_1Pa /rho_kgm3

    return cv_gas_JkgC





# при загрузке модуля считываем из json таблицу Стендинга Катца для оценки z фактора
# и готовим ее для расчета
StandingKatz_curve = _load_StandingKatz_curve_()

