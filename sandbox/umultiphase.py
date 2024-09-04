import numpy as np


def npy_friction_factor(n_re, 
                        roughness_d, 
                        friction_corr_type = 3, 
                        smoth_transition = False):
    """
    Calculates friction factor given pipe relative roughness and Reinolds number
    Parameters
    n_re - Reinolds number
    roughness_d - pipe relative roughness
    friction_corr_type - flag indicating correlation type selection
     0 - Colebrook equation solution
     1 - Drew correlation for smooth pipes
    """
    
    lower_Re_lim = 2000
    upper_Re_lim = 4000
    
    ed = roughness_d
    
    if n_re == 0: 
        f_n = 0
    elif n_re <= lower_Re_lim:  #laminar flow
        f_n = 64 / n_re
    else:                      #turbulent flow
        Re_save = -1
        if smoth_transition and (n_re > lower_Re_lim and n_re < upper_Re_lim):
        # be ready to interpolate for smooth transition
            Re_save = n_re
            n_re = upper_Re_lim
        
        if friction_corr_type == 0:
                #calculate friction factor for rough pipes according to Moody method - Payne et all modification for Beggs&Brill correlation
                # Zigrang and Sylvester  1982  https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae
                f_n = (2 * np.log10(2 / 3.7 * ed - 5.02 / n_re * np.log10(2 / 3.7 * ed + 13 / n_re)))**-2
                  
                i = 0
                while True: 
                    # iterate until error in friction factor is sufficiently small
                    # https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae
                    # expanded form  of the Colebrook equation
                    f_n_new = (1.7384 - 2 * np.log10(2 * ed + 18.574 / (n_re * f_n ** 0.5))) ** -2
                    i += 1
                    f_int = f_n
                    f_n = f_n_new
                    #stop when error is sufficiently small or max number of iterations exceedied
                    if (np.abs(f_n_new - f_int) <= 0.001 or i > 19):
                        break
        elif friction_corr_type == 1:
                #Calculate friction factor for smooth pipes using Drew correlation - original Begs&Brill with no modification
                f_n = 0.0056 + 0.5 * n_re ** -0.32
            
        elif friction_corr_type == 2:
                # Zigrang and Sylvester  1982  https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae
                f_n = (2 * np.log10(1 / 3.7 * ed - 5.02 / n_re * np.log10(1 / 3.7 * ed + 13 / n_re))) ** -2
        elif friction_corr_type == 3:
                # Brkic shows one approximation of the Colebrook equation based on the Lambert W-function
                #  Brkic, Dejan (2011). "An Explicit Approximation of Colebrook's equation for fluid flow friction factor" (PDF). Petroleum Science and Technology. 29 (15): 1596–1602. doi:10.1080/10916461003620453
                # http://hal.archives-ouvertes.fr/hal-01586167/file/article.pdf
                # https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae
                # http://www.math.bas.bg/infres/MathBalk/MB-26/MB-26-285-292.pdf
                Svar = np.log(n_re / (1.816 * np.log(1.1 * n_re / (np.log(1 + 1.1 * n_re)))))
                f_1 = -2 * np.log10(ed / 3.71 + 2 * Svar / n_re)
                f_n = 1 / (f_1 ** 2)
        elif friction_corr_type == 4:
                # from unified TUFFP model
                # Haaland equation   Haaland, SE (1983). "Simple and Explicit Formulas for the Friction Factor in Turbulent Flow". Journal of Fluids Engineering. 105 (1): 89–90. doi:10.1115/1.3240948
                # with smooth transition zone
                
                fr2 = 16 / 2000
                fr3 = 1 / (3.6 * np.log10(6.9 / 3000 + (ed / 3.7) ** 1.11)) ** 2
                if n_re == 0:
                    f_n = 0
                elif (n_re < 2000):
                    f_n = 16 / n_re
                elif (n_re > 3000):
                    f_n = 1 / (3.6 * np.log10(6.9 / n_re + (ed / 3.7) ** 1.11)) ** 2
                elif (n_re >= 2000 and n_re <= 3000):
                    f_n = fr2 + (fr3 - fr2) * (n_re - 2000) / 1000
                
                f_n = 4 * f_n
        elif friction_corr_type == 5:
                # from unified TUFFP model
                # Haaland equation   Haaland, SE (1983). "Simple and Explicit Formulas for the Friction Factor in Turbulent Flow". Journal of Fluids Engineering. 105 (1): 89–90. doi:10.1115/1.3240948

                f_n = 4 / (3.6 * np.log10(6.9 / n_re + (ed / 3.7) ** 1.11)) ** 2

        if smoth_transition and Re_save > 0 :
            x1 = lower_Re_lim
            y1 = 64 / lower_Re_lim
            x2 = n_re
            y2 = f_n
            f_n = ((y2 - y1) * Re_save + (y1 * x2 - y2 * x1)) / (x2 - x1)
    
    return  f_n
