# ESR-STM Simulation of Spins on Surfaces
import numpy as np
import matplotlib.pyplot as plt

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
        self.dimensionOfMatrix = int(self.dimensionOfMatrix)
        self.ReadoutSpin = 0 
        self.AtomPositions = np.zeros((self.NSpins,3))
        self.latticeLengthXY = np.array([0.2877, 0.2877, 0]) # [nm]

        # External Parameters
        self.T = 1 # [K]
        self.B = [0.0,0.0,0.0] # [T]
        self.BTip = [0.0,0.0,0.0] # [T]
        self.BTipGradient = [0,0,0] # [T]
        self.tip = 'f' # 'a' for antiferromagentic and 'f' for ferromagnetic

        # Spin Hamiltoninan
        self.H = np.zeros((self.dimensionOfMatrix,self.dimensionOfMatrix),dtype=complex)
        self.gfactorvector = [self.g]*self.NSpins # Each spin gets his own g-factor, by default 2
        self.Dvector = np.zeros(self.NSpins) # Out of plane Anistropy
        self.Evector = np.zeros(self.NSpins) # In plane Anisotropy
        self.D0=np.zeros((self.NSpins,self.NSpins)) # Creates the Dipole Interaction Matrix
        self.DipoleBool = np.zeros((1,(self.NSpins * (self.NSpins - 1)) // 2),dtype=bool) # Dipole-Bool array to activate and deactivate the dipole interaction between specific spins
        self.Jvector = np.zeros(((self.NSpins * (self.NSpins - 1)) // 2, 3)) # A vector that contains all possible spin combinations

        # Tunneling Parameters
        self.VDC = 0 # [V]
        self.TipPolarization = [0,0,0]
        self.SamplePolarization = [0,0,0]
        self.U = np.zeros(self.NSpins) 
        self.b0 = 0 # Ratio how many electrons that tunnel do not care about the spin!!
        self.G = 1e-6; # [A/V], Experimental conductance at 0V;
        self.G_ss = [2e-6]*self.NSpins # Sample-Sample conductance
        self.G_tt = 1 # 1 enables tip-tip scattering contribution, 0 disables it!

        self.T02 = 2e-5 
        self.Jrs = 0.1 

        # ESR Parameters


    ## Spin-Operator Calculations ##    
    def calcSpinOperators(self):

        self.Sx = np.zeros((self.NSpins, self.dimensionOfMatrix, self.dimensionOfMatrix),dtype=complex)
        self.Sy = np.zeros((self.NSpins, self.dimensionOfMatrix, self.dimensionOfMatrix),dtype=complex)
        self.Sz = np.zeros((self.NSpins, self.dimensionOfMatrix, self.dimensionOfMatrix),dtype=complex)

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
                unit = np.eye(len(np.arange(-self.Spins[j],self.Spins[j]+1)))
                SxTemp = np.kron(unit, SxTemp)
                SyTemp = np.kron(unit, SyTemp)
                SzTemp = np.kron(unit, SzTemp)

            # Multiply with current spin angular momentum matrices
            SxTemp = np.kron(SxTemp, sigma_x)
            SyTemp = np.kron(SyTemp, sigma_y)
            SzTemp = np.kron(SzTemp, sigma_z)

            # Apply Kronecker product for following spins
            for j in range(i_spin + 1, self.NSpins):
                unit = np.eye(len(np.arange(-self.Spins[j],self.Spins[j]+1)))
                SxTemp = np.kron(SxTemp, unit)
                SyTemp = np.kron(SyTemp, unit)
                SzTemp = np.kron(SzTemp, unit)

            # Store the calculated spin operators in the class attributes
            self.Sx[i_spin,:,:]=SxTemp
            self.Sy[i_spin,:,:]=SyTemp
            self.Sz[i_spin,:,:]=SzTemp
            

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
    
    ## Spin Hamiltonian Functions ##
    def calcEigEnergies(self):
        # This function builds the Spin Hamiltonian and then solves it 
        self.H_addZeeman()
        self.H_addTipField()
        self.H_addZeroField()
        self.H_addExchangeInteraction()
        self.calcDipolarCoupling()
        self.H_addDipolarCoupling_new()

        # Solving the Hamiltonian and Sorting its Result 
        E, Evec = np.linalg.eigh(self.H)
        E_sort = np.sort(E)
        Eindex = np.argsort(E)
        self.E_All = E_sort- E_sort[0]
        self.eigVectors = Evec[Eindex]

        self.calcEigStates()

    def H_addZeeman(self):
        # Adding the Zeeman Term 
        for i_spin in range(self.NSpins):  # Python is 0-indexed, unlike MATLAB's 1-indexed loops

            # Apply Zeeman interaction to Hamiltonian H
            self.H += - self.gfactorvector[i_spin] * self.muB * self.B[0] * self.Sx[i_spin,:, :]
            self.H += - self.gfactorvector[i_spin] * self.muB * self.B[1] * self.Sy[i_spin,:, :]
            self.H += - self.gfactorvector[i_spin] * self.muB * self.B[2] * self.Sz[i_spin,:, :]
    
    def H_addTipField(self):
        # Adding the Tipfield to the Readout Spin only
        if self.tip == 'a':
            # Add the tip field contribution to the Hamiltonian H
            self.H += self.gfactorvector[self.ReadoutSpin] * self.muB * self.BTip[0] * self.Sx[self.ReadoutSpin,:, :]
            self.H += self.gfactorvector[self.ReadoutSpin] * self.muB * self.BTip[1] * self.Sy[self.ReadoutSpin,:, :]
            self.H += self.gfactorvector[self.ReadoutSpin] * self.muB * self.BTip[2] * self.Sz[self.ReadoutSpin,:, :]
        else:
            # Subtract the tip field contribution to the Hamiltonian H
            self.H += - self.gfactorvector[self.ReadoutSpin] * self.muB * self.BTip[0] * self.Sx[self.ReadoutSpin,:, :]
            self.H += - self.gfactorvector[self.ReadoutSpin] * self.muB * self.BTip[1] * self.Sy[self.ReadoutSpin,:, :]
            self.H += - self.gfactorvector[self.ReadoutSpin] * self.muB * self.BTip[2] * self.Sz[self.ReadoutSpin,:, :]
    
    def H_addZeroField(self):
        # Adding out of plane Anistropy
        for i_spin in range(self.NSpins):
            self.H += self.Dvector[i_spin]*(self.Sz[i_spin,:,:] @ self.Sz[i_spin,:,:])

    def H_addExchangeInteraction(self):
        # Adds the Exchange interaction to the Hamiltonian
        index_counter = 0  # Initialize the index counter (0-based indexing in Python)

        for i_spin in range(self.NSpins):
            for j_spin in range(self.NSpins):
                if i_spin < j_spin:  # We only look at the case where i_spin < j_spin
                    # Add the spin-spin interaction terms to the Hamiltonian H
                    self.H += self.Jvector[index_counter, 0] * self.Sx[i_spin, :, :] @ self.Sx[j_spin, :, :]
                    self.H += self.Jvector[index_counter, 1] * self.Sy[i_spin, :, :] @ self.Sy[j_spin, :, :]
                    self.H += self.Jvector[index_counter, 2] * self.Sz[i_spin, :, :] @ self.Sz[j_spin, :, :]
                    # Increment the index counter
                    index_counter += 1
    
    def calcDipolarCoupling(self):
        # Calculates the Dipolar Coupling Constants of each Spin-Pair based on their relative distance
        index_counter = 0
        for i_spin in range(self.NSpins):
            for j_spin in range(self.NSpins):
                if i_spin > j_spin:
                    if self.DipoleBool[index_counter]:
                        print("Test")
                     # Calculate the distance between atoms i_spin and j_spin
                        distance_temp = np.linalg.norm(
                            (self.AtomPositions[i_spin, :] - self.AtomPositions[j_spin, :]) * self.latticeLengthXY * 1e-9
                        )
                    # Calculate the dipolar coupling strength D0
                        self.D0[j_spin, i_spin] = (
                            self.gfactorvector[i_spin] *
                            self.gfactorvector[j_spin] *
                            self.muBHz ** 2 *
                            self.mu0 /
                            (4 * np.pi * distance_temp ** 3) *
                            self.JtomeVConversion
                        )
                        index_counter += 1

    def H_addDipolarCoupling_new(self):
        # Adds Dipolar coupling to the Spin-Hamiltonian."""
        # Loop over spins
        for i_spin in range(self.NSpins):
            for j_spin in range(i_spin + 1):  # Only consider j_spin <= i_spin to avoid double counting
                if i_spin != j_spin:
                    # Calculate the normalized connection vector (rhat)
                    diff_vec = self.AtomPositions[i_spin, :] - self.AtomPositions[j_spin, :]
                    rhat = diff_vec / np.linalg.norm(diff_vec)

                    # Add the dipolar interaction to the Hamiltonian
                    self.H += (
                        self.D0[j_spin, i_spin]
                        * (self.Sx[j_spin, :, :] @ self.Sx[i_spin, :, :]
                           + self.Sy[j_spin, :, :] @ self.Sy[i_spin, :, :]
                           + self.Sz[j_spin, :, :] @ self.Sz[i_spin, :, :])
                    )

                    # Subtract the 3*(D0) term with the projection of spin operators along rhat
                    self.H -= (
                        3 * self.D0[j_spin, i_spin]
                        * ((self.Sx[j_spin, :, :] * rhat[0]
                           + self.Sy[j_spin, :, :] * rhat[1]
                          + self.Sz[j_spin, :, :] * rhat[2])
                        @ (self.Sx[i_spin, :, :] * rhat[0]
                           + self.Sy[i_spin, :, :] * rhat[1]
                           + self.Sz[i_spin, :, :] * rhat[2]))
                    )
    
    def calcEigStates(self):
        # Create the basis states in Bra-Ket notation
        basisStates = np.empty(self.dimensionOfMatrix, dtype=object)
        for i in range(self.dimensionOfMatrix):
            basisStates[i] = "|"
            for j in range(self.NSpins):
                if j != self.NSpins - 1:
                    basisStates[i] += str(np.real(self.Sz[j, i, i])) + ","
                else:
                    basisStates[i] += str(np.real(self.Sz[j, i, i]))
            basisStates[i] += ">"

        # Initialize the states and other strings
        states = np.empty(self.dimensionOfMatrix, dtype=object)
        statesOnlyLHS = np.empty(self.dimensionOfMatrix, dtype=object)
        statesWithoutE = np.empty(self.dimensionOfMatrix, dtype=object)
        statesOnlyRHS = np.empty(self.dimensionOfMatrix, dtype=object)

        # Sort the absolute values of the eigenvectors
        SortedIndices = np.argsort(-np.abs(self.eigVectors), axis=0)  # Sort by absolute value, descending
        SortedEigenvectors = np.empty_like(self.eigVectors, dtype=complex)

        for j in range(self.dimensionOfMatrix):
            SortedEigenvectors[:, j] = self.eigVectors[SortedIndices[:, j], j]

        # Construct states in bracket notation
        for i in range(self.dimensionOfMatrix):
            states[i] = f"|{i}> (E = {round(self.E_All[i],4)}) = "
            statesOnlyLHS[i] = f"|{i}> (E = {round(self.E_All[i],4)})"
            statesWithoutE[i] = f"|{i}>"
            statesOnlyRHS[i] = ""

            for j in range(self.dimensionOfMatrix):
                if np.abs(SortedEigenvectors[j, i]) > 1e-4:  # Threshold for relevance
                    sign = "+" if np.real(SortedEigenvectors[j, i]) >= 0 else ""
                    states[i] += f"{sign}{SortedEigenvectors[j, i]}{basisStates[SortedIndices[j, i]]}"

                    if j < 2:  # Only include the top two contributions in statesOnlyRHS
                        statesOnlyRHS[i] += f"{sign}{SortedEigenvectors[j, i]}{basisStates[SortedIndices[j, i]]}"

        # Store the results in the Sys object
        self.states = states
        self.statesOnlyLHS = statesOnlyLHS
        self.statesWithoutE = statesWithoutE
        self.statesOnlyRHS = statesOnlyRHS
        self.basisStates = basisStates
    
    def showEigenMatrix(self):
        m = np.zeros((self.dimensionOfMatrix, self.dimensionOfMatrix))

        # Fill the matrix with probabilities
        for i in range(self.dimensionOfMatrix):
            for j in range(self.dimensionOfMatrix):
                m[i, j] = np.abs(self.eigVectors[j, self.dimensionOfMatrix - 1 - i])**2

        # Generate the figure
        plt.figure(np.random.randint(30000, 40000))
        name0 = 'Probability'

        # Display the matrix as an image (colormap)
        plt.imshow(m, cmap='hot', aspect='auto')
        c = plt.colorbar()

        # Axis labels
        plt.xlabel('Basis States')
        plt.ylabel('Eigenstates')

         # Title with external magnetic field values
        plt.title(f"System Eigenstates, B_Ext = [{self.B[0]}, {self.B[1]}, {self.B[2]}]")

        # Set colorbar label
        c.set_label(name0)

        # Set x-axis ticks and labels
        plt.xticks(np.arange(self.dimensionOfMatrix), self.basisStates)

        # Set y-axis ticks and labels, flipping them
        plt.yticks(np.arange(self.dimensionOfMatrix), self.statesOnlyLHS[::-1])

        # Set color limits
        plt.clim(0, 1) 