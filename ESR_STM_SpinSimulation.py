# ESR-STM Simulation of Spins on Surfaces
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import null_space

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
        self.V_DC = 0 # [V]
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
        self.H = np.zeros((self.dimensionOfMatrix,self.dimensionOfMatrix),dtype=complex) #Important to reset the spin Hamiltonian
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
        self.E_All_inGHz = self.E_All*self.meVtoGHzConversion
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
    
    def calcRecordedTransitions(self,MinE,MaxE):
    
     # This function calculates the recorded transitions based on the occupation probability
     # and the energy differences between states. It records transitions where the state
     # has sufficiently high occupation and the energy difference is within the specified range.

     # Initialize RecordedTransitions with a dummy value to avoid issues when appending later
        RecordedTransitions = [(0, 0)]

     # Loop over all possible state pairs
        for i in range(self.dimensionOfMatrix):
        # Check if the state has sufficiently high occupation based on the Boltzmann distribution
            if self.p0[i] > 10**(-5):
             for j in range(i + 1, self.dimensionOfMatrix):
                # Calculate the energy difference
                deltaE = self.E_All_inGHz[j] - self.E_All_inGHz[i]
                if MinE < deltaE < MaxE:
                    RecordedTransitions.append((i, j))
                
                    

        # Remove the initial dummy (0, 0) element
        RecordedTransitions.pop(0)

        return RecordedTransitions 
    
    def calcESR_Benjamin(self,**kwargs):
        # Default options (like ops struct in MATLAB)
        options = {
         'FreqRange': [10, 20],
         'AllowPumping': 0,
         'N': 5000,
         'lw': 0.02,
         'plot': 1,
         'norm': 1
        }

         #  Update with user-provided options
        options.update(kwargs)

        # Initialize
        self.ESRshift = 0
        freq = np.linspace(options['FreqRange'][0], options['FreqRange'][1], options['N'])  # Frequency array
        norm = options['norm']  # Normalization option

        # Lorentzian function equivalent
        fitlorentz = lambda para, x1: (1 / (para[3]**2 + 1) * abs(para[1]) * 
                                        (1 + (x1 - para[0]) / (0.5 * para[2]) * para[3])**2 / 
                                        (1 + ((x1 - para[0]) / (0.5 * para[2]))**2))

        ESRsignal = np.zeros(len(freq))

        # Get the populations

        

        # Look at transitions
        p0 = np.exp((-self.E_All)/(self.kB*self.T)) # Both E and kb in meV bzw. meV/T
        pTot = sum(p0)
        self.p0 = p0/pTot
        self.RecordedTransitions = self.calcRecordedTransitions(freq[0],freq[-1])
        self.ResonanceFrequencies = np.zeros(len(self.RecordedTransitions))

        for i in range(len(self.RecordedTransitions)):
            self.ResonanceFrequencies[i] = abs(self.E_All_inGHz[self.RecordedTransitions[i][1]] - 
                                            self.E_All_inGHz[self.RecordedTransitions[i][0]])

        # Optionally print the transitions
        if options['plot']:
            if len(self.RecordedTransitions) < 300:
                print(f"Found {len(self.RecordedTransitions)} potential transitions:")
                for i in range(len(self.RecordedTransitions)):
                    print(f"|{self.RecordedTransitions[i][0]}> -> |{self.RecordedTransitions[i][1]}>; "
                        f"f = {np.round(self.ResonanceFrequencies[i],4)} GHz")
            else:
                print(f"There were {len(self.RecordedTransitions)} potential transitions in the specified frequency range")

        # Amplitude calculation
        for k in range(len(self.ResonanceFrequencies)):
            i = self.RecordedTransitions[k][0]  # Initial state index
            j = self.RecordedTransitions[k][1]  # Final state index

            self.dp = abs(self.p0[i] - self.p0[j])

            # Rabi rate factor
            rabiRateFactor = 1

            self.M_x = rabiRateFactor * (self.eigVectors[:, j].T.conj() @ self.Sx[self.ReadoutSpin,:, :] @ self.eigVectors[:, i])
            self.M_y = rabiRateFactor * (self.eigVectors[:, j].T.conj() @ self.Sy[self.ReadoutSpin,:, :] @ self.eigVectors[:, i])
            self.M_z = rabiRateFactor * (self.eigVectors[:, j].T.conj() @ self.Sz[self.ReadoutSpin,:, :] @ self.eigVectors[:, i])

            self.Amp = self.dp * (self.BTip[0] * self.M_x + self.BTip[1] * self.M_y + self.BTip[2] * self.M_z)**2

            fitlorentz = lambda para, x1: (1 / (para[3]**2 + 1) * abs(para[1]) *
                               (1 + (x1 - para[0]) / (0.5 * para[2]) * para[3])**2 /
                               (1 + ((x1 - para[0]) / (0.5 * para[2]))**2))

            ESRsignal += fitlorentz([self.ResonanceFrequencies[k], self.Amp, options['lw'], 0], freq)

        # Normalization
        if norm == 1:
            ESRsignal = (ESRsignal - ESRsignal.min()) / (ESRsignal.max() - ESRsignal.min())

        # Plot the ESR signal if requested
        if options['plot']:
            color_list = plt.cm.hsv(np.linspace(0, 1, len(self.ResonanceFrequencies)))

            plt.figure(1225)
            plt.plot(freq, ESRsignal, '-k', linewidth=2, label='ESR Signal')

            # Mark resonances
            for i, freq_res in enumerate(self.ResonanceFrequencies):
                index = np.argmin(np.abs(freq - freq_res))
                amp = ESRsignal[index]
                transition_str = (f"|{self.RecordedTransitions[i][0]}> -> |{self.RecordedTransitions[i][1]}>; "
                                  f"f = {np.round(freq_res,4)} GHz")
                plt.plot(freq_res, amp, color=color_list[i], marker='.', markersize=12, 
                        label=transition_str, linewidth=2)

            plt.xlabel('Freq (GHz)')
            plt.ylabel('ΔI (a.u.)')
            plt.title('ESR-Simulation')
            plt.legend()
            plt.xlim([freq[0],freq[-1]])
            plt.show()

        return freq, ESRsignal, self.p0
    
    def plotESRTipfieldsweep(self,**kwargs):
        # Function that plots various ESR Spectra for different Tipfields

        # Default options (like ops struct in MATLAB)
        options = {
         'FreqRange': [10, 20],
         'BTipLimits': [-0.05,0.05],
         'AllowPumping': 0,
         'N_B': 20,
         'N_Freq': 200,
         'lw': 0.02,
         'plot': 1,
         'norm': 1
        }

         #  Update with user-provided options
        options.update(kwargs)

         # Predefine Arrays
        BTipRange=np.linspace(options['BTipLimits'][0],options['BTipLimits'][1],options['N_B'])
        Freq = np.zeros((options['N_Freq']), dtype=float)
        ESRsignal = np.zeros((options['N_B'],options['N_Freq']), dtype=float)
        
        # calculating the ESR signal for each Field 
        for i in range(options['N_B']):
            self.BTip[2]=float(BTipRange[i])
            self.calcEigEnergies()
            Freq, ESRsignal[i][:],_ = self.calcESR_Benjamin(plot=0,norm=options['norm'],N=options['N_Freq'],lw=options['lw'],FreqRange=options['FreqRange'],AllowPumping=options['AllowPumping'])
            
        # Calculating the Detuning from the Tipfield
        Detuning=self.g*self.muB*BTipRange*self.meVtoGHzConversion

        # Plot the ESR Signal as a color plot
        fig, ax1 = plt.subplots()

        # Plot the matrix using imshow
        im = ax1.imshow(ESRsignal, cmap='Blues', interpolation='none', 
                        extent=[Freq[0], Freq[-1], BTipRange[0], BTipRange[-1]], origin='lower', aspect='auto')

        # Add labels and title for primary Y-axis
        ax1.set_xlabel('Freq (GHz)')
        ax1.set_ylabel('B_Tip (T)')
        ax1.set_title('Tipfield dependent ESR')

        # Create a secondary Y-axis
        ax2 = ax1.twinx()
        ax2.set_ylim(BTipRange[0], BTipRange[-1])
        y_ticks = ax1.get_yticks()
        detuning_labels = np.interp(y_ticks, BTipRange, Detuning)
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(np.round(detuning_labels, 2))
        ax2.set_ylabel('Detuning (GHz)')

        cbar = plt.colorbar(im, ax=ax1)
        cbar.ax.set_position([0.9, 0.1, 0.03, 0.8])

        plt.show()

    def calcTunnelingMatrixElements(self,scattertype):

     

     # Get the electron tunneling matrix elements
     # scattertype = "st", "tt", "ss"
     x_el, y_el, z_el, u_el = self.calcTunnelingElectronMatrixElements(scattertype)

     # Initialize RateMatrix
     if scattertype == "ss":
        RateMatrix = np.zeros((self.NSpins,self.dimensionOfMatrix, self.dimensionOfMatrix))
        N=self.NSpins
        Index=np.arange(N)
     else:
        RateMatrix = np.zeros((1,self.dimensionOfMatrix, self.dimensionOfMatrix)) 
        N=1
        Index=np.array([self.ReadoutSpin])

     # x_el, y_el, z_el, and u_el are assumed to be 2x2 matrices (flattened to 4 elements)
     # Loop over 4 elements, combining matrix operations and calculating the rate matrix
     for k in range(N):
    
        # Calculate matrix elements for Sx, Sy, Sz, and U using the eigenvectors and spin operators
        MatrixX = np.conjugate(np.transpose(self.eigVectors)) @ self.Sx[Index[k],:, :] @ self.eigVectors
        MatrixY = np.conjugate(np.transpose(self.eigVectors)) @ self.Sy[Index[k],:, :] @ self.eigVectors
        MatrixZ = np.conjugate(np.transpose(self.eigVectors)) @ self.Sz[Index[k],:, :] @ self.eigVectors
        MatrixU = self.U[Index[k]] * np.conjugate(np.transpose(self.eigVectors)) @ self.eigVectors

        for i in range(2):
            for j in range(2):
                RateMatrix[k][:][:] += np.abs(MatrixX * x_el[i][j] + MatrixY * y_el[i][j] + MatrixZ * z_el[i][j] + MatrixU * u_el[i][j]) ** 2

     return RateMatrix
    
    def calcTunnelingElectronMatrixElements(self,scattertype):
   
    
     # Normalize the Tip and Sample Polarizations if their norms are greater than 1
     if np.linalg.norm(self.TipPolarization) > 1:
         self.TipPolarization = self.TipPolarization / np.linalg.norm(self.TipPolarization)
     if np.linalg.norm(self.SamplePolarization) > 1:
         self.SamplePolarization = self.SamplePolarization / np.linalg.norm(self.SamplePolarization)
    
     # Get Spin-1/2 matrices (Pauli matrices) for electrons
     S_x, S_y, S_z = self.calcAngMomMatrices(0.5)

     # Calculate the density matrices for tip and sample
     densityTip = 0.5 * np.eye(2) + self.TipPolarization[0] * S_x + self.TipPolarization[1] * S_y + self.TipPolarization[2] * S_z
     densitySample = 0.5 * np.eye(2) + self.SamplePolarization[0] * S_x + self.SamplePolarization[1] * S_y + self.SamplePolarization[2] * S_z
     
     # Calculate the eigenvalues and eigenvectors of the density matrices
     eigValTip_temp, eigVecTip = np.linalg.eig(densityTip)
     eigValSample_temp, eigVecSample = np.linalg.eig(densitySample)
     # Convert eigenvalue arrays to diagonal matrices (like MATLAB output)
     eigValTip = np.diag(eigValTip_temp)
     eigValSample = np.diag(eigValSample_temp)
     
     # Swap the eigen value order
     eigValTip[[0, 1], [0, 1]] = eigValTip[[1, 0], [1, 0]]
     eigVecTip[:, [0, 1]] = eigVecTip[:, [1, 0]]

     # Calculate the weights matrix as sqrt(diag(eigValSample) * diag(eigValTip))
     Weights_st = np.sqrt(np.outer(np.diag(eigValSample), np.diag(eigValTip)))
     Weights_tt = np.sqrt(np.outer(np.diag(eigValTip), np.diag(eigValTip)))
     Weights_ss = np.sqrt(np.outer(np.diag(eigValSample), np.diag(eigValSample)))

     # Correct alignment of eigenvectors if needed (manual step)
     # Make sure that x_el, y_el, z_el are calculated correctly
     x_st = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_x) @ eigVecTip), Weights_st)
     y_st = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_y) @ eigVecTip), Weights_st)
     z_st = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_z) @ eigVecTip), Weights_st)
     u_st = np.multiply((2 * np.conjugate(eigVecSample.T) @ eigVecTip), Weights_st)

     x_tt = np.multiply((np.conjugate(eigVecTip.T) @ (2 * S_x) @ eigVecTip), Weights_tt)
     y_tt = np.multiply((np.conjugate(eigVecTip.T) @ (2 * S_y) @ eigVecTip), Weights_tt)
     z_tt = np.multiply((np.conjugate(eigVecTip.T) @ (2 * S_z) @ eigVecTip), Weights_tt)
     u_tt = np.multiply((2 * np.conjugate(eigVecTip.T) @ eigVecTip), Weights_tt)

     x_ss = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_x) @ eigVecSample), Weights_ss)
     y_ss = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_y) @ eigVecSample), Weights_ss)
     z_ss = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_z) @ eigVecSample), Weights_ss)
     u_ss = np.multiply((2 * np.conjugate(eigVecSample.T) @ eigVecSample), Weights_ss)
     
     if scattertype == "st":
         return x_st, y_st, z_st, u_st
     elif scattertype == "tt":
         return x_tt, y_tt, z_tt, u_tt
     elif scattertype == "ss":
         return x_ss, y_ss, z_ss, u_ss
     else:
         return x_st, y_st, z_st, u_st
     
    def calcRateIntegrals(self):
     
     # Create the three nxn matrices [I_ts, I_st, I_ss]
     RateIntegrals = np.zeros((3,self.dimensionOfMatrix, self.dimensionOfMatrix))

     # Define the Fermi-Dirac Distribution function
     FermiDirac = lambda energy: 1 / (1 + np.exp(energy / (self.kB * self.T)))

     # Calculating the integral
     # If DC voltage is zero, use the integral
     if self.V_DC == 0:
        for i in range(self.dimensionOfMatrix):
            for j in range(self.dimensionOfMatrix):
                # Define the integrand functions
                fun_ts = lambda x: FermiDirac(x - self.V_DC) * (1 - FermiDirac(x - (self.E_All[j] - self.E_All[i])))
                fun_st = lambda x: FermiDirac(x + self.V_DC) * (1 - FermiDirac(x - (self.E_All[j] - self.E_All[i])))
                fun_ss = lambda x: FermiDirac(x) * (1 - FermiDirac(x - (self.E_All[j] - self.E_All[i])))

                # Integrate over the range (-inf, inf)
                RateIntegrals[0, i, j] = quad(fun_ts, -np.inf, np.inf)[0]
                RateIntegrals[1, i, j] = quad(fun_st, -np.inf, np.inf)[0]
                RateIntegrals[2, i, j] = quad(fun_ss, -np.inf, np.inf)[0]
     else:
        # Use the analytical solution if V_DC is non-zero
        for i in range(self.dimensionOfMatrix):
            for j in range(self.dimensionOfMatrix):
                delta_E = self.E_All[j] - self.E_All[i]

                RateIntegrals[0, i, j] = abs((delta_E - self.V_DC) / (np.exp((delta_E - self.V_DC) / (self.kB * self.T)) - 1))
                RateIntegrals[1, i, j] = abs((delta_E + self.V_DC) / (np.exp((delta_E + self.V_DC) / (self.kB * self.T)) - 1))

                # Exception for the sample-sample rate where delta_E = 0
                if delta_E == 0.0:
                    fun_ss = lambda x: FermiDirac(x) * (1 - FermiDirac(x - delta_E))
                    RateIntegrals[2, i, j] = quad(fun_ss, -np.inf, np.inf)[0]
                else:
                    RateIntegrals[2, i, j] = abs(delta_E / (np.exp(delta_E / (self.kB * self.T)) - 1))

     return RateIntegrals
        
    def calcRates(self, **kwargs):

        # Default options (like ops struct in MATLAB)
        options = {
         'AllowPumping': 1,
         'Approach': "Loth"
        }

         #  Update with user-provided options
        options.update(kwargs)

        # First calculate the Eigenenergies and States
        self.calcEigEnergies()

        # Initialize the Rates matrix
        self.Rates = np.zeros((5,self.dimensionOfMatrix, self.dimensionOfMatrix))

        # Getting the relevant matrix elements
        self.Matrix = self.calcTunnelingMatrixElements("st")[0]
        self.MatrixSS = self.calcTunnelingMatrixElements("ss")
        self.MatrixTT = self.calcTunnelingMatrixElements("tt")[0]
        
        MatrixSS_summed = np.zeros((self.dimensionOfMatrix, self.dimensionOfMatrix))

        for i in range(self.NSpins):
            MatrixSS_summed += self.MatrixSS[i, :, :] * self.G_ss[i]

        # Calculating the 3 different integrals
        RateIntegrals = self.calcRateIntegrals()

        Rate_0 = self.Matrix[0][0]  # Transition matrix element of the ground state

        ### Rate-Factors using Loth 2010
        if options['Approach'] == "Loth":
            self.G_st = (1 - self.b0) * self.G

            RateFactor_ts = self.G_st / (self.e * Rate_0) / 1e3
            RateFactor_st = self.G_st / (self.e * Rate_0) / 1e3
            RateFactor_ss = 1 / (self.e * Rate_0) / 1e3
            RateFactor_tt = 0

        ### Rate-Factors using Ternes 2010 (not default)
        if options['Approach'] == "Ternes":
            RateFactor_ts = 2 * np.pi / self.hbar_meV * self.T02
            RateFactor_st = 2 * np.pi / self.hbar_meV * self.T02
            RateFactor_ss = 2 * np.pi / self.hbar_meV * self.Jrs**2
            RateFactor_tt = 2 * np.pi / self.hbar_meV * self.T02**2 / (self.Jrs**2) * self.G_tt

        # Calculate rates based on rate factors and integrals
        self.Rates[0, :, :] = RateFactor_ts * self.Matrix.T * RateIntegrals[0, :, :]
        self.Rates[1, :, :] = RateFactor_st * self.Matrix * RateIntegrals[1, :, :]
        self.Rates[2, :, :] = RateFactor_ss * MatrixSS_summed * RateIntegrals[2, :, :]
        self.Rates[3, :, :] = RateFactor_tt * self.MatrixTT * RateIntegrals[2, :, :]

        # Sum the rates
        self.Rates_Summed = (self.Rates[0,:, :] + self.Rates[1, :, :] + 
                            self.Rates[2, :, :] + self.Rates[3, :, :] + 
                            self.Rates[4, :, :])

        # Build the Rate-Equation matrix
        self.K = np.zeros((self.dimensionOfMatrix, self.dimensionOfMatrix))
        R1 = np.zeros(self.dimensionOfMatrix)
        RSS = np.zeros(self.dimensionOfMatrix)

        for i in range(self.dimensionOfMatrix):
            for j in range(self.dimensionOfMatrix):
                if i != j:
                    self.K[i, i] -= self.Rates_Summed[i, j]  # Rates leaving state i
                    self.K[i, j] = self.Rates_Summed[j, i]  # Rates leading to state i
                    R1[i] += self.Rates[2, i, j]  # Summing up all the rates

        # Calculate the total lifetime (Tau_total)
        self.Tau_total = 1 / R1

        # Calculate the thermal population (Boltzmann distribution)
        p0 = np.exp(-self.E_All / (self.kB * self.T))
        pTot = np.sum(p0)
        self.p0 = p0 / pTot  # Normalized thermal population

        # Solve the Rate-Equation if non-thermal population is desired
        if options['AllowPumping']:
            p = null_space(self.K)
            if p.size == 0:  # If no valid solution, use thermal distribution
                self.Populations = self.p0
            else:
                self.Populations = p[:, 0] / np.sum(p[:, 0])
        else:
            self.Populations = self.p0

        # Deriving the tunneling current using Equation 41 from Ternes paper
        self.I = 0
        for i in range(self.dimensionOfMatrix):
            for j in range(self.dimensionOfMatrix):
                I_temp = self.e * self.Populations[i] * (self.Rates[0, i, j] - self.Rates[1, i, j])
                self.I += I_temp

        # Adjust current calculation based on approach
        if options['Approach'] == "Ternes":
            self.I = self.I / (self.e**2 * self.T02 / self.h_meV)
        if options['Approach'] == "Loth":
            self.I += self.b0 * self.G * self.V_DC
                
    import numpy as np

    def calculate_derivative(self, I, V):
        
        n = len(I)  # Length of the array
        dIdV = np.zeros(n)  # Initialize the output array for derivative
        
        # Forward difference for the first point
        dIdV[0] = (I[1] - I[0]) / (V[1] - V[0])
        
        # Central difference for the middle points
        for i in range(1, n-1):
            dIdV[i] = (I[i+1] - I[i-1]) / (V[i+1] - V[i-1])
        
        # Backward difference for the last point
        dIdV[n-1] = (I[n-1] - I[n-2]) / (V[n-1] - V[n-2])
        
        return dIdV

    def calcIETS(self, **kwargs):
        
        # Default options (like ops struct in MATLAB)
        options = {
         'Vrange': 30,
         'N': 500,
         'AllowPumping': True,
         'norm': False,
         'plot': True
        }

         #  Update with user-provided options
        options.update(kwargs)
        
        # Initialize the arrays
        V_array = np.linspace(-options['Vrange'], options['Vrange'], options['N'])
        I = np.zeros((options['N']),dtype=float)
        P = np.zeros((self.dimensionOfMatrix, options['N']))
        
        # Solve the rate equation for each voltage value
        for i in range(options['N']):
            self.V_DC = V_array[i]
            self.calcRates(AllowPumping=options['AllowPumping'])
            I[i] = self.I
            P[:, i] = self.Populations

        # Calculate the derivative (dI/dV)
        dIdV = self.calculate_derivative(I, (V_array*1e-3))  # Convert V_array to volts
        
        # Normalize dI/dV if required
        if options['norm']:
            dIdV = (dIdV - np.min(dIdV)) / (np.max(dIdV) - np.min(dIdV))
        
        # Optional Plot
        if options['plot']:
            # Plotting the dI/dV spectrum
            plt.figure(188)
            plt.plot(V_array, dIdV, '-k', linewidth=2)
            plt.ylabel('dI/dV (a.u.)')
            plt.xlabel('Bias Voltage (mV)')
            plt.xlim([-options['Vrange'], options['Vrange']])
            plt.title('IETS-Spectrum')
            plt.grid(True)
            plt.tick_params(axis='both', which='major', labelsize=16)  # Set tick label size
            plt.show()

            # Plotting the populations of each state
            plt.figure(189)
            colors = plt.cm.hsv(np.arange(self.dimensionOfMatrix))  # Get the colormap
            
            for i in range(len(P[:, 0])):
                plt.plot(V_array, P[i, :], color=colors[i], linewidth=2, label=f'p{i+1}')
            
            plt.yscale('log')
            plt.xlabel('Bias Voltage (mV)')
            plt.ylim([1e-3, 1])
            plt.xlim([-options['Vrange'], options['Vrange']])
            plt.legend()
            plt.title('Populations')
            plt.grid(True)
            plt.tick_params(axis='both', which='major', labelsize=16)  # Set tick label size
            plt.show()

        return V_array, dIdV, P


            
