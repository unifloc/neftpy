"""
nt for petroleum engineering calculations
"""



# ходовые константы от которых все должно зависеть
# названия большими буквами
# названия размерностей стандартные - маленькие буквы, кроме фамилий и множителей


G_ms2 = 9.81               # gravity
PI = 3.14159265359

P_SC_atma = 1           # pressure standard condition
T_SC_atma = 20          # temperature standard condition
P_SC_MPa = 0.101325 
P_SC_Pa = 101325
P_SC_PSI = 14.7

RS_MAX_Velarde = 800                # внутреннее ограничение для корреляции Веларде МакКейна для расчета газосодержания
T_McCain_K_MIN = 289   

T_K_MIN = 273         
T_K_MAX = 573                   
T_K_ZERO_C = 273

T_C_MIN = T_K_MIN - T_K_ZERO_C
T_C_MAX = T_K_MAX - T_K_ZERO_C
T_SC_K = T_SC_atma + T_K_ZERO_C     # температура в стандартных условиях, К

R  = 8.31                           # Universal gas constant
RHO_AIR_kgm3 = 1.225                # плотность воздуха при стандартных условиях definition from https://en.wikipedia.org/wiki/Density_of_air
GAMMA_WAT = 1                       # удельная плотность воды при стандартных условиях   
RHO_WATER_SC_kgm3 = 1000            # плотность воды при стандартных условиях (для расчета удельной)

ZNLF_RATE = 0.1                     # константа перехода к расчету барботажа (предельный/минимальный дебит жидкости)

M_AIR_KGMOL = 0.029                 # Air molar mass

SIGMA_WAT_GAS_NM = 0.01             # поверхностное натяжение на границе с воздухом (газом) - типовые значения для дефолтных параметров  Н/м
SIGMA_OIL_NM = 0.025

MU_WAT_cP = 0.36
MU_GAS_cP = 0.0122
MU_OIL_cP = 0.7

# параметры для задания свойств по умолчанию 
GAMMA_GAS = 0.6
GAMMA_WAT = 1
GAMMA_OIL = 0.86
RSB_m3m3 = 100
BOB_m3m3 = 1.2
Z = 0.9

T_RES_C = 90
ROUGHNESS = 0.0001

AT = 98066.5  # technical atmosphere in Pa,  техническая атмосфера в Па
