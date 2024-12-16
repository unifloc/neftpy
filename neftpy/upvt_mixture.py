import neftpy.uconvert as uc
import neftpy.uconst as uconst

import numpy as np

from typing import Union

# базовый тип для расчетов с массивами
FloatArray = Union[float, np.ndarray]


""" 
====================================================================================================
поверхностное натяжение на границе нефть-газ, вода - газ
====================================================================================================
"""

def unf_surface_tension_Baker_Sverdloff_Nm(p_atma:FloatArray, 
                                           t_C:float, 
                                           gamma_oil:FloatArray):
    """
    calculate surface tension according Baker Sverdloff correlation
    """
   
    t_F = uc.C_2_F(t_C) 
    p_psia = uc.atm_2_psi(p_atma) 
    p_MPa = uc.atm_2_MPa(p_atma) 
    oil_API = uc.gamma_oil_2_api(gamma_oil)

    st_68_F = 39 - 0.2571 * oil_API
    st_100_F = 37.5 - 0.2571 * oil_API

    if t_F < 68:
        st_oil_gas_dyncm = st_68_F
    else:
        t_st = t_F
        if t_F > 100:
            t_st = 100
        st_oil_gas_dyncm = (68 - (((t_st - 68) * (st_68_F - st_100_F)) / 32)) * np.exp(-0.00086306 * p_psia)

    st_w_74F = (75 - (1.108 * (p_psia) ** 0.349))
    st_w_280_F = (53 - (0.1048 * (p_psia) ** 0.637))

    if t_F < 74:
        st_water_gas_dyncm = st_w_74F
    else:
        t_st_w = t_F
        if t_F > 280:
            t_st_w = 280
        st_water_gas_dyncm = st_w_74F - (((t_st_w - 74) * (st_w_74F - st_w_280_F)) / 206)
    st_water_gas_dyncm = 10 ** (-(1.19 + 0.01 * p_MPa)) * 1000
    return (uc.dyncm_2_Nm(st_oil_gas_dyncm), uc.dyncm_2_Nm(st_water_gas_dyncm))



def unf_surface_tension_gw_Sutton_Nm(rho_water_kgm3:FloatArray, 
                                     rho_gas_kgm3:FloatArray, 
                                     t_C:FloatArray):  # TODO поправка на соленость добавить
    """
        Корреляция Саттона для поверхностного натяжения на границе вода-газ

    :param rho_water_kgm3: плотность воды кг / м3
    :param rho_gas_kgm3:  плотность газа кг / м3
    :param t_C: температура в С
    :return: поверхностное натяжение на границе вода-газ, Н / м

    ref 1 Pereira L. et al. Interfacial tension of reservoir fluids: an integrated experimental
    and modelling investigation : дис. – Heriot-Watt University, 2016. page 41

    ref2 Ling K. et al. A new correlation to calculate oil-water interfacial tension
    //SPE Kuwait International Petroleum Conference and Exhibition. – Society of Petroleum Engineers, 2012.

    """
    rho_water_gcm3 = rho_water_kgm3 / 1000
    rho_gas_gcm3 = rho_gas_kgm3 / 1000
    t_R = uc.C_2_R(t_C)
    st_dyncm = ((1.53988 * (rho_water_gcm3 - rho_gas_gcm3) + 2.08339) /
            ((t_R /302.881) ** (0.821976 - 0.00183785 * t_R +
            0.00000134016 * t_R ** 2))) ** 3.6667
    return uc.dyncm_2_Nm(st_dyncm)


def unf_surface_tension_go_Abdul_Majeed_Nm(t_K:FloatArray, 
                                           gamma_oil:FloatArray, 
                                           rs_m3m3:FloatArray):
    """
        Корреляция Абдул-Маджида (2000 г.) для поверхностного натяжения нефти, насыщенной газом

    :param t_K: температура, градусы Кельвина
    :param gamma_oil: относительная плотность нефти
    :param rs_m3m3: газосодержание, м3 / м3
    :return: поверхностное натяжение на границе нефть-газ, Н / м

        Источник: Справочник инженера-нефтяника. Том 1. Введение в нефтяной инжиниринг. Газпром Нефть
    """

    t_F = uc.K_2_F(t_K)
    oil_API = uc.gamma_oil_2_api(gamma_oil)
    st_dead_oil_dynes_cm = (1.17013 - 1.694e-3 * t_F) * (38.085 - 0.259 * oil_API)
    relative_st_go_od = (0.056379 + 0.94362 * np.exp(-21.6128e-3 * rs_m3m3))
    return uc.dyncm_2_Nm(st_dead_oil_dynes_cm * relative_st_go_od)
