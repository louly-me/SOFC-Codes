# Install necessary packages (run these in your terminal or script setup)
# pip install numpy
# pip install thermo
# pip install scipy

import numpy as np
from thermo import Chemical
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt

def calculate_voltage_and_power(J):
    """
    Calculate the cell voltage and power for a given current density.
    Parameters:
        J (float): Current density in A/m²
    Returns:
        E (float): Cell voltage in V
        P (float): Power output in W
    """
    
    # Constants and Assumptions
    F = 96485  # Faraday constant [C/mol]
    T_stack_in = 700 + 273.15  # Stack temperature [K]
    P_0 = 101.3  # Ambient pressure [kPa]
    T_0 = 298.15  # Ambient temperature [K]
    R = 8.314  # Universal gas constant [J/(mol*K)]
    pi = math.pi
    delta_c = 350e-6 # Anode thickness [m]
    delta_a = 40e-6 # Cathode thickness [m]
    N_cell=1 # Number of cells


    # 2-Anode Inlet
    # 3-Anode Outlet
    # 4-Cathode Inlet
    # 5-Catode Outlet
    # Inputs
    P = [101.3,101.3,112.3, 110.3, 112.3, 110.3]  # Pressure at different points [kPa]
    y_H2 = [0.8] * 6  # Mole fraction of H2 at different points
    y_O2 = [0.21] * 6  # Mole fraction of O2 at different points
    y_H2O = [0.1] * 6  # Mole fraction of H2O at different points
    T = [700 + 273.15, 700 + 273.15, 800 + 273.15, 700 + 273.15, 800 + 273.15, 700 + 273.15]  # Temperatures [K]
    T_react = T_stack_in  # Reaction temperature [K]

    # Thermodynamic properties using thermo
    def get_properties(species, T, P):
        """Get enthalpy and entropy for a given species."""
        chem = Chemical(species, T=T, P=P * 1000)  # P in Pa for thermo
        return chem.Hc, chem.S0

    # Calculate Gibbs free energy changes
    species = ["H2O", "H2", "O2"]
    enthalpy_dict = {}
    entropy_dict = {}

    for sp in species:
        enthalpy, entropy = get_properties(sp, T_stack_in, P_0)
        enthalpy_dict[sp] = enthalpy  # Enthalpy in J/mol
        entropy_dict[sp] = entropy  # Entropy in J/mol-K

    def gibbs_free_energy(enthalpy, entropy, T):
        return enthalpy - T * entropy

    g_0 = {sp: gibbs_free_energy(enthalpy_dict[sp], entropy_dict[sp], T_stack_in) for sp in species}

    # Gibbs free energy of reaction
    dG_0 = -g_0["H2O"] + g_0["H2"] + 0.5 * g_0["O2"]
    # Voltage calculations
    E_r_in = dG_0 / (2 * F) + (R * T_stack_in / (2 * F)) * np.log(((y_H2[2] * P[2] / P_0) * (y_O2[4] * P[4] / P_0) ** 0.5) / (y_H2O[2] * P[2] / P_0))
    E_r_out = dG_0 / (2 * F) + (R * T[3] / (2 * F)) * np.log(((y_H2[3] * P[3] / P_0) * (y_O2[5] * P[5] / P_0) ** 0.5) / (y_H2O[3] * P[3] / P_0))
    E_r = -0.5 * (E_r_in + E_r_out)/1000


    # Constants for activation calculations
    alpha_a = 0.83035  # Anode transfer coefficient
    alpha_c = 0.77251  # Cathode transfer coefficient
    m = 0.18247  # Reaction order for oxygen in anode exchange current density
    a1 = -0.058348  # Reaction order for hydrogen in cathode exchange current density
    b = 0.48814  # Reaction order for water in cathode exchange current density
    E_act_a = 148141.9979  # Activation energy for anode reaction [J/mol]
    E_act_c = 120551.2717  # Activation energy for cathode reaction [J/mol]

    # Pre-exponential factors for exchange current density [A/m²]
    gamma_a_in = 243410164.8322 * T[4]  # Anode factor at inlet temperature
    gamma_a_out = 243410164.8322 * T[5]  # Anode factor at outlet temperature
    gamma_c_in = 8997693.4385 * T_react  # Cathode factor at reaction temperature
    gamma_c_out = 8997693.4385 * T[3]  # Cathode factor at outlet temperature
    
    # Exchange current density calculations
    J_0_a_in = (
        gamma_a_in 
        * ((y_O2[4] * P[4] / P_0)**m) 
        * math.exp(-E_act_a / (R * T[4]))
    )

    J_0_a_out = (
        gamma_a_out 
        * ((y_O2[5] * P[5] / P_0)**m) 
        * math.exp(-E_act_a / (R * T[5]))
    )

    J_0_c_in = (
        gamma_c_in 
        * ((y_H2[2] * P[2] / P_0)**a1) 
        * ((y_H2O[2] * P[2] / P_0)**b) 
        * math.exp(-E_act_c / (R * T_react))
    )

    J_0_c_out = (
        gamma_c_out 
        * ((y_H2[3] * P[3] / P_0)**a1) 
        * ((y_H2O[3] * P[3] / P_0)**b) 
        * math.exp(-E_act_c / (R * T[3]))
    )

    def solve_eta_act(J, J_0, alpha, T):
        def func(eta_act):
            term1 = np.exp(2 * F * alpha * eta_act / (R * T))
            term2 = np.exp(2 * F * (alpha - 1) * eta_act / (R * T))
            return J_0 * (term1 - term2) - J

        # Initial guess for eta_act
        eta_guess = 0.1
        eta_act_solution = fsolve(func, eta_guess)
        return eta_act_solution[0]


    # Solving for eta_act for each case
    eta_act_c_in = solve_eta_act(J, J_0_c_in, alpha_c, T_react)
    eta_act_c_out = solve_eta_act(J, J_0_c_out, alpha_c, T[3])
    eta_act_a_in = solve_eta_act(J, J_0_a_in, alpha_a, T[4])
    eta_act_a_out = solve_eta_act(J, J_0_a_out, alpha_a, T[5])

    # Ohmic loss calculation
    B_ohm = 7.8247e11  # Effective conductivity constant [S·m²/K]
    E_act_ohm = 8.0022e4  # Activation energy for ionic conductivity [J/mol]
    eta_ohm_in = (T_stack_in / B_ohm) * np.exp(E_act_ohm / (R * T_stack_in)) * J  # Ohmic loss at inlet
    eta_ohm_out = (T[3] / B_ohm) * np.exp(E_act_ohm / (R * T[3])) * J  # Ohmic loss at outlet
    eta_ohm = 0.5 * (eta_ohm_in + eta_ohm_out)  # Average ohmic loss

    # Activation loss (combined for anode and cathode)
    eta_act_a = 0.5 * (eta_act_a_in + eta_act_a_out)  # Anode activation loss
    eta_act_c = 0.5 * (eta_act_c_in + eta_act_c_out)  # Cathode activation loss
    eta_act = eta_act_a + eta_act_c  # Total activation loss

    # Partial pressure calculations [Pa]
    P_H2_zero_in = y_H2[2] * P[2] * 1000  # Hydrogen partial pressure at inlet
    P_H2O_zero_in = y_H2O[2] * P[2] * 1000  # Water partial pressure at inlet
    P_O2_zero_in = y_O2[4] * P[4] * 1000  # Oxygen partial pressure at inlet

    P_H2_zero_out = y_H2[3] * P[3] * 1000  # Hydrogen partial pressure at outlet
    P_H2O_zero_out = y_H2O[3] * P[3] * 1000  # Water partial pressure at outlet
    P_O2_zero_out = y_O2[5] * P[5] * 1000  # Oxygen partial pressure at outlet

    # Diffusion volumes [m³/mol]
    V_H2 = 6.12e-6  # Diffusion volume of hydrogen
    V_H2O = 13.10e-6  # Diffusion volume of water

    # Tortuosity and porosity of the electrodes
    t_f = 2.8  # Tortuosity factor for anode
    t_o = 2.8  # Tortuosity factor for cathode
    p_f = 0.3  # Porosity of anode
    p_o = 0.3  # Porosity of cathode

    # Pore diameters [m]
    r_f = 0.5e-6  # Pore diameter of anode
    r_o = r_f  # Assuming cathode has the same pore diameter

    # Molecular weights [kg/mol]
    M_O2 = 0.032  # Molecular weight of oxygen
    M_H2 = 0.002  # Molecular weight of hydrogen
    M_H2O = 0.018  # Molecular weight of water

    # Molar masses for mixtures
    M_H2_H2O = 2 / ((1 / M_H2) + (1 / M_H2O)) * 1000

    # Binary diffusion coefficients [m^2/s]
    D_H2_H2O = (1.43e-2 * T_react**1.75) / (1000 * P[1] * M_H2_H2O**0.5 * (V_H2**(1/3) + V_H2O**(1/3))**2)

    # Knudsen diffusion coefficients [m^2/s]
    D_O2_k = (2 / 3) * r_o * (8 * R * T_react / (pi * M_O2))**0.5
    D_H2_k = (2 / 3) * r_f * (8 * R * T_react / (pi * M_H2))**0.5
    D_H2O_k = (2 / 3) * r_f * (8 * R * T_react / (pi * M_H2O))**0.5

    # Effective diffusion coefficients [m^2/s]
    D_O2_eff = (p_o / t_o**2) * D_O2_k
    D_H2_eff = (p_f / t_f**2) * ((D_H2_k * D_H2_H2O) / (D_H2_k + D_H2_H2O))
    D_H2O_eff = (p_f / t_f**2) * ((1 / D_H2O_k + 1 / D_H2_H2O)**-1)

    # Concentration overpotential calculations
    eta_conc_c_in = (
        (R * T_react) / (2 * F) 
        * math.log(
            (1 + (J * R * T_react * delta_c) / (2 * F * D_H2O_eff * P_H2O_zero_in)) / 
            (1 - (J * R * T_react * delta_c) / (2 * F * D_H2O_eff * P_H2_zero_in))
        )
    )

    eta_conc_c_out = (
        (R * T[3]) / (2 * F) 
        * math.log(
            (1 + (J * R * T[3] * delta_c) / (2 * F * D_H2O_eff * P_H2O_zero_out)) / 
            (1 - (J * R * T[3] * delta_c) / (2 * F * D_H2O_eff * P_H2_zero_out))
        )
    )

    eta_conc_a_in = (
        (R * T[4]) / (4 * F) 
        * math.log(1 / (1 - (R * T[4] * delta_a / (4 * F * D_O2_eff * P_O2_zero_in)) * J))
    )

    eta_conc_a_out = (
        (R * T[5]) / (4 * F) 
        * math.log(1 / (1 - (R * T[5] * delta_a / (4 * F * D_O2_eff * P_O2_zero_out)) * J))
    )

    #  Concentration Loss
    eta_conc_c = 0.5 * (eta_conc_c_in + eta_conc_c_out)
    eta_conc_a = 0.5 * (eta_conc_a_in + eta_conc_a_out)
    eta_conc = eta_conc_c + eta_conc_a


    # Cell Voltage and Power
    E = E_r - (eta_ohm + eta_act + eta_conc) 
    P = 0.001 * E* J * N_cell # Adjust for current and cell area
    return E, P

# Generate data for plot
J_values = np.linspace(0.01, 1, 100)  # Current density range
E_values = []
P_values = []

for J in J_values:
    E, P = calculate_voltage_and_power(J*1000)
    E_values.append(E)
    P_values.append(P)

# Create a single figure with two subplots
plt.figure(figsize=(8, 6))

# Subplot 1: Cell Voltage vs Current Density
plt.subplot(2, 1, 1)  # (rows, columns, subplot index)
plt.plot(J_values, E_values, label="Cell Voltage (V)", color='blue')
plt.xlabel("Current Density (A/cm²)")
plt.ylabel("Voltage (V)")
plt.title("Cell Voltage vs Current Density")
plt.legend()
plt.grid(True)

# Subplot 2: Power vs Current Density
plt.subplot(2, 1, 2)  # (rows, columns, subplot index)
plt.plot(J_values, P_values, label="Power (W)", color='green')
plt.xlabel("Current Density (A/cm²)")
plt.ylabel("Power (W)")
plt.title("Power vs Current Density")
plt.legend()
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the figure
plt.show()