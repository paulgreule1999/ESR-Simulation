# ESR-STM Simulation of Spins on Surfaces
import numpy as np

class SpinSys:

    # Constants
    mu0 = 1.25663706127e-6  # [N/A^2]
    muB = 0.05788 # [meV/T]
    muBHz = 9.274010078328e-24 # [J/T]
    h = 6.62607015e-34 #[J⋅s]
    hbar = 1.054571817e-34 #[J⋅s]
    hbar_meV = 6.582119569e-16*1e3 #[meV⋅s]
    h_meV = 4.13566769692e-15*1e3 #[meV⋅s]
    g = 2.00; 
    kB = 0.0861733 # [meV/K]
    kBHz = 1.380649e-23 # [J/K]
    e = 1.602176e-19 
    meVtoGHzConversion = 1e-12*e/h # approx 241.8 GHz/meV
    GHztomeVConversion =1/meVtoGHzConversion # approx 4.1357 uV/GHz
    meVtoJConversion=1.602e-22; # approx 1.6e-22 J/meV
    JtomeVConversion=1/meVtoJConversion; # approx 6.24e21 meV/J

    ## Initialize the Spin System ##
    def __init__(self,Spins):

        self.Spins=Spins
        self.NSpins=len(self.Spins)
        self.dimensionOfMatrix=1 # A system with N spins [S1,S2,S3,...,SN] is represented through a matrix of dimension (2*S1+1)*(2*S2+1)*...*(2*SN+1)
        for i in range(self.NSpins):  
            self.dimensionOfMatrix *= (2 * self.Spins[i] + 1)
        self.ReadoutSpin = 1 
        self.AtomPositions = np.zeros((self.NSpins,3))
        self.latticeLengthXY = [0.2877, 0.2877, 0] # [nm]

        # External Parameters
        self.T = 1 # [K]
        self.B = [0.0,0.0,0.0] # [T]
        self.BTip = [0.0,0.0,0.0] # [T]
        self.BTipGradient = [0,0,0] # [T]

        # Spin Hamiltoninan
        self.gfactorvector = [self.g]*self.NSpins # Each spin gets his own g-factor, by default 2
        self.Dvector = [0]*self.NSpins # Out of plane Anistropy
        self.Evector = [0]*self.NSpins # In plane Anisotropy
        self.D=np.zeros((self.NSpins,self.NSpins)) # Creates the Dipole Interaction Matrix
        self.DipoleBool = np.zeros((1,(self.NSpins * (self.NSpins - 1)) // 2)) # Dipole-Bool array to activate and deactivate the dipole interaction between specific spins
        self.Jvector = np.zeros(((self.NSpins * (self.NSpins - 1)) // 2, 3)) # A vector that contains all possible spin combinations

        # Tunneling Parameters
        self.VDC = 0 # [V]
        self.TipPolarization = [0,0,0]
        self.SamplePolarization = [0,0,0]
        self.U = [0]*self.NSpins 
        self.b0 = 0 # Ratio how many electrons that tunnel do not care about the spin!!
        self.G = 1e-6; # [A/V], Experimental conductance at 0V;
        self.G_ss = [2e-6]*self.NSpins # Sample-Sample conductance
        self.G_tt = 1 # 1 enables tip-tip scattering contribution, 0 disables it!

        self.T02 = 2e-5 
        self.Jrs = 0.1 

        # ESR Parameters


    ## Spin-Operator Calculations ##    
    def calcSpinOperators(self):

        self.Sx = np.zeros((0, 0, self.NSpins))
        self.Sy = np.zeros((0, 0, self.NSpins))
        self.Sz = np.zeros((0, 0, self.NSpins))

        # Loop through each spin
        for i_spin in range(self.NSpins):
            # Calculate angular momentum matrices for the given spin type
            sigma_x, sigma_y, sigma_z = self.calcAngMomMatrices(self.Spins[i_spin])

            # Initialize temporary variables for spin operators
            SxTemp = 1
            SyTemp = 1
            SzTemp = 1

            # Apply Kronecker product for previous spins
            for j in range(i_spin):
                unit = np.eye(2 * int(self.Spins[j]) + 1)
                SxTemp = np.kron(unit, SxTemp)
                SyTemp = np.kron(unit, SyTemp)
                SzTemp = np.kron(unit, SzTemp)

            # Multiply with current spin angular momentum matrices
            SxTemp = np.kron(SxTemp, sigma_x)
            SyTemp = np.kron(SyTemp, sigma_y)
            SzTemp = np.kron(SzTemp, sigma_z)

            # Apply Kronecker product for following spins
            for j in range(i_spin + 1, self.NSpins):
                unit = np.eye(2 * int(self.Spins[j]) + 1)
                SxTemp = np.kron(SxTemp, unit)
                SyTemp = np.kron(SyTemp, unit)
                SzTemp = np.kron(SzTemp, unit)

            # Store the calculated spin operators in the class attributes
            self.Sx = np.append(self.Sx, SxTemp)
            self.Sy = np.append(self.Sy, SyTemp)
            self.Sz = np.append(self.Sz, SzTemp)

    def calcAngMomMatrices(self, s):
        m_values = np.arange(-s, s + 1, 1)  # m = -s to +s

        # Create raising and lowering operators S_p (creation) and S_m (annihilation)
        S_p = np.zeros((len(m_values), len(m_values)), dtype=complex)
        S_m = np.zeros((len(m_values), len(m_values)), dtype=complex)

        # Fill in the elements of S_p and S_m
        for m1 in m_values:
            for m2 in m_values:
                if m1 == m2 + 1:
                    S_p[int(m2 + s), int(m1 + s)] = np.sqrt((s - m2) * (s + m2 + 1))
                elif m1 == m2 - 1:
                    S_m[int(m2 + s), int(m1 + s)] = np.sqrt((s + m2) * (s - m2 + 1))

        # Calculate S_x, S_y, and S_z
        S_x = (S_p + S_m) / 2
        S_y = (S_p - S_m) / (2j)
        S_z = np.diag(-m_values)

        return S_x, S_y, S_z