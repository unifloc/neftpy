"""
Constant and units conversion functions for for petroleum engineering calculations
"""
import numpy as np
import scipy.constants as const


g = const.g   # gravity
pi = const.pi
pressure_sc_bar = 1   # pressure standard condition
temperature_sc_C = 15    # temperature standard condition
const_at = 98066.5  # technical atmosphere in Pa,  техническая атмосфера в Па
air_density_sc_kgm3 = 1.225  # definition from https://en.wikipedia.org/wiki/Density_of_air
rho_water_sc_kgm3 = 1000

# some default values 
default_z = 0.9
default_gamma_gas = 0.8
default_rsb_m3m3 = 100


# simple unit conversion functions

# pressure

def psi_2_Pa(value):
    """
    converts pressure in psi (pound-force per square inch) to Pa (Pascal)
    :param value: pressure value in psi
    :return: pressure value in Pa
    """
    return value * const.psi


def Pa_2_psi(value):
    """
    converts pressure in Pa (Pascal) to psi (pound-force per square inch)
    :param value: pressure value in Pa
    :return: pressure value in psi
    """
    return value / const.psi

def Pa_2_psig(value):
    """
    converts pressure in abs Pa (Pascal) to gauged psig (pound-force per square inch)
    :param value: pressure value in Pa
    :return: pressure value in psi
    """
    return value / const.psi - 14.7

def MPa_2_psi(value):
    """
    converts pressure in MPa (Mega Pascal, 1e6 Pa) to psi (pound-force per square inch)
    :param value: pressure value in MPa
    :return: pressure value in Pa
    """
    return value * const.mega / const.psi

def MPa_2_psig(value):
    """
    converts pressure 
    in absolute MPa (Mega Pascal, 1e6 Pa) to measured psig (pound-force per square inch)
    :param value: pressure value in MPa
    :return: pressure value in Pa
    """
    return value * const.mega / const.psi - 14.7

def bar_2_Pa(value):
    """
    converts pressure in bar (1e5 Pa) to Pa
    :param value: pressure value in bar
    :return: pressure value in Pa
    """
    return value * const.bar

def bar_2_MPa(value):
    """
    converts pressure in bar to Pa
    :param value: pressure value in bar
    :return: pressure value in Pa
    """
    return value * const.bar / const.mega


def Pa_2_bar(value):
    """
    converts pressure in Pa to bar
    :param value: pressure value in Pa
    :return: pressure value in bar
    """
    return value / const.bar

def MPa_2_bar(value):
    """
    converts pressure in MPa to bar
    :param value: pressure value in MPa
    :return: pressure value in bar
    """
    return value * const.mega / const.bar

def atm_2_Pa(value):
    """
    converts pressure in atm (standard atmosphere) to Pa
    :param value: pressure value in atm
    :return: pressure value in Pa
    """
    return value * const.atm


def Pa_2_atm(value):
    """
    converts pressure in Pa to atm (standard atmosphere)
    :param value: pressure value in Pa
    :return: pressure value in atm (standard atmosphere)
    """
    return value / const.atm


def at_2_Pa(value):
    """
    converts pressure in at (technical atmosphere) to Pa
    :param value: pressure value in at (technical atmosphere)
    :return: pressure value in Pa
    """
    return value * const_at


def Pa_2_at(value):
    """
    converts pressure in Pa (Pascal) to at (technical atmosphere)
    :param value: pressure value in Pa
    :return: pressure value in at (technical atmosphere)
    """
    return value * const_at


def bar_2_psi(value):
    """
    converts pressure in bar to psi
    :param value: pressure value in bar
    :return: pressure value in psi
    """
    return value * const.bar / const.psi


def psi_2_bar(value):
    """
     converts pressure in psi to bar
     :param value: pressure value in psi
     :return: pressure value in bar
     """
    return value / const.bar * const.psi


def bar_2_atm(value=1):
    """
     converts pressure in bar to atm(standard atmosphere)
     :param value: pressure value in bar
     :return: pressure value in atm(standard atmosphere)
     """
    return value * const.bar / const.atm


def atm_2_bar(value=1):
    """
     converts pressure in atm(standard atmosphere) to bar
     :param value: pressure value in atm(standard atmosphere)
     :return: pressure value in bar
     """
    return value * const.atm / const.bar

# temperature


def C_2_F(value):
    """
     converts temperature in C(degrees Celsius) to F(degrees Fahrenheit)
     :param value: temperature in C(degrees Celsius)
     :return: temperature in F(degrees Fahrenheit)
     """
    return const.convert_temperature(value, 'C', 'F')


def F_2_C(value):
    """
     converts temperature in F(degrees Fahrenheit) to C(degrees Celsius)
     :param value: temperature in F(degrees Fahrenheit)
     :return: temperature in C(degrees Celsius)
     """
    return const.convert_temperature(value, 'F', 'K')


def C_2_K(value):
    """
     converts temperature in C(degrees Celsius) to K(Kelvins)
     :param value: temperature in C(degrees Celsius)
     :return: temperature in K(Kelvins)
     """
    return const.convert_temperature(value, 'C', 'K')


def K_2_C(value):
    """
     converts temperature in K(Kelvins) to C(degrees Celsius)
     :param value: temperature in K(Kelvins)
     :return: temperature in C(degrees Celsius)
     """
    return const.convert_temperature(value, 'K', 'C')


def K_2_F(value):
    """
     converts temperature in F(degrees Fahrenheit) to K(Kelvins)
     :param value: temperature in F(degrees Fahrenheit)
     :return: temperature in K(Kelvins)
     """
    return const.convert_temperature(value, 'K', 'F')


def F_2_K(value):
    """
     converts temperature in F(degrees Fahrenheit) to K(Kelvins)
     :param value: temperature in F(degrees Fahrenheit)
     :return: temperature in K(Kelvins)
     """
    return const.convert_temperature(value, 'F', 'K')


def F_2_R(value):
    """
     converts temperature in F(degrees Fahrenheit) to R(degrees Rankine)
     :param value: temperature in F(degrees Fahrenheit)
     :return: temperature in R(degrees Rankine)
     """
    return const.convert_temperature(value, 'F', 'R')


def R_2_F(value):
    """
     converts temperature in R(degrees Rankine) to F(degrees Fahrenheit)
     :param value: temperature in R(degrees Rankine)
     :return: temperature in F(degrees Fahrenheit)
     """
    return const.convert_temperature(value, 'R', 'F')


def C_2_R(value):
    """
     converts temperature in C(degrees Celsius) to R(degrees Rankine)
     :param value: temperature in C(degrees Celsius)
     :return: temperature in R(degrees Rankine)
     """
    return const.convert_temperature(value, 'C', 'R')


def R_2_C(value):
    """
     converts temperature in R(degrees Rankine) to C(degrees Celsius)
     :param value: temperature in C(degrees Celsius)
     :return: temperature in R(degrees Rankine)
     """
    return const.convert_temperature(value, 'R', 'C')


def K_2_R(value):
    """
     converts temperature in K(Kelvins) to R(degrees Rankine)
     :param value: temperature in K(Kelvins)
     :return: temperature in R(degrees Rankine)
     """
    return const.convert_temperature(value, 'K', 'R')


def R_2_K(value):
    """
     converts temperature in R(degrees Rankine) to K(Kelvins)
     :param value: temperature in R(degrees Rankine)
     :return: temperature in K(Kelvins)
     """
    return const.convert_temperature(value, 'R', 'K')

# length


def m_2_in(value):
    """
    converts length in m(meters) to in(inches)
    :param value: length in m(meters)
    :return: length in in(inches)
    """
    return value / const.inch


def in_2_m(value):
    """
    converts length in in(inches) to m(meters)
    :param value: length in in(inches)
    :return: length in m(meters)
    """
    return value * const.inch


def m_2_ft(value):
    """
    converts length in m(meters) to ft(feet)
    :param value: length in m(meters)
    :return: length in ft(feet)
    """
    return value / const.foot


def ft_2_m(value):
    """
    converts length in ft(feet) to m(meters)
    :param value: length in ft(feet)
    :return: length in m(meters)
    """
    return value * const.foot

# volume


def m3_2_bbl(value):
    """
     converts volume in m3(cubic metres) to bbl(barrels)
     :param value: volume in m3(cubic metres)
     :return: volume in bbl(barrels)
     """
    return value / const.barrel


def bbl_2_m3(value):
    """
     converts volume in m3(cubic metres) to bbl(barrels)
     :param value: volume in m3(cubic metres)
     :return: volume in bbl(barrels)
     """
    return value * const.barrel

# GOR


def m3m3_2_m3t(value, gamma=1):
    """
     converts Gas-Oil Ratio in m3/m3(cubic metres/cubic meter) to m3/t(cubic metres/ton)
     :param value: Gas-Oil Ratio in m3/m3(cubic metres/cubic meter);
            gamma=1: oil density(by air)
     :return: Gas-Oil Ratio in m3/t(cubic metres/ton)
     """
    return value * gamma


def m3t_2_m3m3(value, gamma=1):
    """
     converts Gas-Oil Ratio in m3/t(cubic metres/ton) to m3/m3(cubic metres/cubic meter)
     :param value: Gas-Oil Ratio in m3/t(cubic metres/ton);
            gamma=1: oil density(by air)
     :return: Gas-Oil Ratio in m3/m3(cubic metres/cubic meter)
     """
    return value / gamma


def scfstb_2_m3m3(value):
    """
     converts Gas-Oil Ratio in scf/stb(standard cubic feet/standard barrel) to m3/m3(cubic metres/cubic meter)
     :param value: Gas-Oil Ratio in scf/stb(standard cubic feet/standard barrel)
     :return: Gas-Oil Ratio in m3/m3(cubic metres/cubic meter)
     """
    return value * (const.foot ** 3 / const.bbl)


def m3m3_2_scfstb(value):
    """
     converts Gas-Oil Ratio in m3/m3(cubic metres/cubic meter) to scf/stb(standard cubic feet/standard barrel)
     :param value: Gas-Oil Ratio in m3/m3(cubic metres/cubic meter)
     :return: Gas-Oil Ratio in scf/stb(standard cubic feet/standard barrel)
     """
    return value / (const.foot ** 3 / const.bbl)


# density


def api_2_gamma_oil(value):
    """
     converts density in API(American Petroleum Institute gravity) to gamma_oil (oil relative density by water)
     :param value: density in API(American Petroleum Institute gravity)
     :return: oil relative density by water
     """
    return (value + 131.5) / 141.5


def gamma_oil_2_api(value):
    """
     converts density in API(American Petroleum Institute gravity) to gamma_oil (oil relative density by water)
     :param value: oil relative density by water
     :return: density in API(American Petroleum Institute gravity)
     """
    return 141.5 / value - 131.5


def kgm3_2_lbft3(value):
    """
     converts density in kg/m3(kilogrammes/cubic meter) to lb/ft3(pounds/cubic feet)
     :param value: density in kgm3(kilogrammes/cubic meter)
     :return: density in lb/ft3(pounds/cubic feet)
     """
    return value * (const.foot ** 3 / const.lb)


def lbft3_2_kgm3(value):
    """
     converts density in lb/ft3(pounds/cubic feet) to kgm3(kilogrammes/cubic meter)
     :param value: density in lb/ft3(pounds/cubic feet)
     :return: density in kgm3(kilogrammes/cubic meter)
     """
    return value / (const.foot ** 3 / const.lb)


# compressibility


def compr_1pa_2_1psi(value):
    """
     converts compressibility in 1/pa to 1/psi
     :param value: compressibility in 1/pa
     :return: compressibility in 1/psi
     """
    return value * const.psi


def compr_1psi_2_1Pa(value):
    """
     converts compressibility in 1/psi to 1/pa
     :param value: compressibility in 1/psi
     :return: compressibility in 1/pa
     """
    return value / const.psi


def compr_1psi_2_1MPa(value):
    """
     converts compressibility in 1/psi to 1/pa
     :param value: compressibility in 1/psi
     :return: compressibility in 1/pa
     """
    return value / const.psi * const.mega

def compr_1pa_2_1bar(value):
    """
     converts compressibility in 1/pa to 1/bar
     :param value: compressibility in 1/pa
     :return: compressibility in 1/bar
     """
    return value * const.bar


def compr_1bar_2_1pa(value):
    """
     converts compressibility in 1/bar to 1/pa
     :param value: compressibility in 1/bar
     :return: compressibility in 1/pa
     """
    return value / const.bar


def compr_1mpa_2_1bar(value):
    """
     converts compressibility in 1/mpa to 1/bar
     :param value: compressibility in 1/mpa
     :return: compressibility in 1/bar
     """
    return value * const.bar / const.mega


def compr_1bar_2_1mpa(value):
    """
     converts compressibility in 1/bar to 1/mpa
     :param value: compressibility in 1/bar
     :return: compressibility in 1/mpa
     """
    return value * const.mega / const.bar


# heat capacity


def btulbmF_2_kJkgK(value):
    """
    converts heat capacity from FIELD to SI
    :param value: heat capacity in Btu/lbm F
    :return: heat capacity in kJ/kgK equal to kJ/kgC
    """
    return value * 4.186


# surface tension


def dyncm_2_Nm(value):
    """
    converts surface tension
    :param value: surface tension in dyn / cm
    :return: surface tension in N / m
    """
    return value / 1000


def Nm_2_dyncm(value):
    """
    converts surface tension
    :param value: surface tension in dyn / cm
    :return: surface tension in N / m
    """
    return value * 1000

# volumetric rates

def m3day_2_m3sec(value):
    """
    comverts volumetric flow rates from m3 per day to m3 per second
    :param value: flow volume in m3  / day
    :return: flow volume in m3  / sec
    """
    return value / 86400


def m3sec_2_m3day(value):
    """
    comverts volumetric flow rates from m3 per second to m3 per day
    :param value: flow volume in m3  / day
    :return: flow volume in m3  / sec
    """
    return value * 86400


def m3day_2_bblday(value):
    """
    comverts volumetric flow rates from m3 per day to barrel per day
    :param value: flow volume in m3  / day
    :return: flow volume in barrel  / day
    """
    return value / const.barrel


def bblday_2_m3day(value):
    """
    comverts volumetric flow rates from  barrel per day to m3 per day
    :param value: flow volume in m3  / day
    :return: flow volume in barrel  / day
    """
    return value * const.barrel

    
# viscosity 


def cP_2_Pasec(value):
    """
    converts viscosity
    :param value: viscosity in cP
    :return: viscosity in pa * sec
    """
    return value / 1000

# angles

def grad_2_rad(value):
    """
    converts angle from degrees to radians
    :param value: angle in grad
    :return: angle in rad
    """
    return value * pi / 180

