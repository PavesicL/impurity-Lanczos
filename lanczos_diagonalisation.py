#!/usr/bin/env python3

"""
Finds the spectrum and the eigenstates of the hamiltonian of a small superconductor, coupled to an impurity. In this specific implementation, one has control over the number of
particles in the system and the total spin of the system. The implemented impurity coupling is SIAM.

IMPLEMENTATION
The calculation is performed in the basis of occupation numbers: |n_impUP, n_impDOWN, n_0UP, n_0DOWN, n_1UP, n_1DOWN, ... >.
First we find all basis vectors in the Hilbert space of the system with N levels, which has a specified amount of particles with spin up (nUP) and spin down (nDOWN). 
This is done by the makeBase() function. A state is represented by a vector of probability amplitudes, accompanied by basisList, which hold information of what basis state 
each index in the state vector represents. The fermionic operators are implemented using bit-wise operations, specifically functions flipBit() and countSetBits(). 

INDEXING OF STATES IN THE STATE VECTOR: 
The occupancy number of a single particle state (i, s) is given as the 2i+s'th bit of the basis state, written in binary (and counted from the left), where (spin) s=0 for UP 
and s=1 for down. 
The offset of the given bit (counted from right to left) in the binary representation is (2N-1) - (2i+s) = 2(N-i)-1-s. The impurity spin state is at offset 2N and 2N+1.

DIAGONALISATION
Diagonalisation is implemented using the Lanczos algorithm. The linear operator H is implemented using the numpy class LinearOperator. It allows us to define a linear operator
and diagonalise it, using scipy.sparse.linalg.eigsh. 
The complete diagonalisation, in the functions LanczosDiag_states() and LanczosDiag_states() (if one wants to obtain the eigenvectors too) is done subspace by subspace. The
smallest subspace of the Hamiltonian is one with defined number of particles (n) and total spin z of the system. Alternatively, one defines n and the number of particles with
spin UP/DOWN in the system. Here, we have to count the particles at the impurity levels too.
We have:
	n = nUP + nDOWN,
	1/2 (nUP + nDOWN) = Sz.
All quantities in these two equations are constant in a given subspace. 
It is possible to only obtain the spectrum of the system in the specified state (n, nUP, nDOWN), using LanczosDiag_nUPnDOWN or LanczosDiagStates_nUPnDOWN.

NUMBA
@jit dislikes the class (p) as a function argument. This throws a warning, but the code gives results. 
I tried compiling the class with njitclass(), but it fails when infering the types of the getInput() function. 
"""
################################################################################
import re

import numpy as np 
import scipy
import matplotlib.pyplot as plt

from scipy.special import comb
from numpy.linalg import norm

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

import numba
from numba import jit, njit
from numba import types
from numba import int32, float64, float32, int64, boolean, char

from joblib import Parallel, delayed

# PARAMS CLASS ####################################################################################

class params():
	"""
	The class of parameters necessary to define the Hamiltonian.
	"""

	def __init__(self, inputFile):
		self.D=1	#energy unit

		self.band_level_shift = getInput("band_level_shift", inputFile, 1)

		self.model = ""

		self.parallel = getInput("parallel", inputFile, 0)	#whether to run the calculations for different sectors in parallel

		self.N = getInput("N", inputFile, 9, integer=True)	#number of levels

		self.U = getInput("U", inputFile, 10)			
		self.epsimp = getInput("epsimp", inputFile, -self.U/2)
		
		self.EZ_imp = getInput("EZ_imp", inputFile, 0)

		#single channel parameters:
		self.gamma = getInput("gamma", inputFile, 0) 	
		self.alpha = getInput("alpha", inputFile, 0)
		self.Ec = getInput("Ec", inputFile, 0)
		self.n0 = getInput("n0", inputFile, self.N)
		self.t = getInput("t", inputFile, 0)

		self.EZ_bulk = getInput("EZ_bulk", inputFile, 0)
		
		#two channel parameters:
		self.gamma1 = getInput("gamma1", inputFile, 0) 	
		self.alpha1 = getInput("alpha1", inputFile, 0)
		self.gamma2 = getInput("gamma2", inputFile, 0) 	
		self.alpha2 = getInput("alpha2", inputFile, 0)
		
		self.Ec1 = getInput("Ec1", inputFile, 0)
		self.n01 = getInput("n01", inputFile, self.N/2)
		self.Ec2 = getInput("Ec2", inputFile, 0)
		self.n02 = getInput("n02", inputFile, self.N/2)		

		self.EZ_bulk1 = getInput("EZ_bulk1", inputFile, 0)
		self.EZ_bulk2 = getInput("EZ_bulk2", inputFile, 0)

		self.d = 2*self.D/(self.N-1)
		self.rho=1/(2*self.D)
		self.V = np.sqrt(self.gamma/(np.pi*self.rho))
		self.V1 = np.sqrt(self.gamma1/(np.pi*self.rho))
		self.V2 = np.sqrt(self.gamma2/(np.pi*self.rho))

		self.nref = getInput("nref", inputFile, int(self.n01+self.n02+(0.5-self.epsimp/self.U)) if self.n01+self.n02+(0.5-self.epsimp/self.U)>0 else self.n01+self.n02+1, integer=True)		#the central n of the calculation
		self.nrange = getInput("nrange", inputFile, 1, integer=True)		#the range of calculation

		self.Eshift = { 
						"SIAM"			:	self.U/2 + self.Ec*self.n0**2,
						"SIAM_2channel"	:	self.U/2 + self.Ec1*self.n01**2 + self.Ec2*self.n02**2	#MODEL DEPENDENT!!!
						}

		self.refisn0 = getInput("refisn0", inputFile, 0)

		self.flat_band = getInput("flat_band", inputFile, 0)

		self.get_vectors = getInput("get_vectors", inputFile, 0)
		self.vector_precision = getInput("vector_precision", inputFile, 0.1)	#how prominent does the vector element have to be to be printed

		self.excited_states = getInput("excited_states", inputFile, 2)	#used in sweep scripts, how many excited states are saved. 2 means the ground state and the first excited state.

		self.print_parity = getInput("print_parity", inputFile, 0)	

		self.get_charge_susceptibility = getInput("get_charge_susceptibility", inputFile, 0)
		self.how_many_charge_susceptibilities = getInput("how_many_charge_susceptibilities", inputFile, 0)
		self.charge_susceptibility_cutoff = getInput("charge_susceptibility_cutoff", inputFile, 0)

		self.get_doublet_overlaps = getInput("get_doublet_overlaps", inputFile, 0)

		self.verbosity = getInput("verbosity", inputFile, 0)
		self.include_all_sectors = getInput("include_all_sectors", inputFile, 0)

		self.all_states = getInput("all_states", inputFile, 0)
		self.states_per_sector = getInput("states_per_sector", inputFile, 3)
		self.print_all_energies = getInput("print_all_energies", inputFile, 0)

# READ PARAMS FROM INPUT FILE #####################################################################

def getInput(paramName, inputFile, default, integer=False):
	"""
	#Given paramName (string) and inputFile (string), return the value of the param, saved in the inputFile. If not found, uses default as value.
	"""
	with open(inputFile, "r") as inputF:
		for line in inputF:
			
			a=re.fullmatch(paramName+"\s*=\s*([+-]?[0-9]+(?:\.?[0-9]*(?:[eE][+-]?[0-9]+)?)?)", line.strip())
			if a:
				val = float(a.group(1))		

				if integer:
					return int(val)
				else:
					return val

	if integer:
		return int(default)
	else:
		return default

# UTILITY #########################################################################################

def printVector(p, vector, basisList):
	"""
	Prints the most prominent elements of a vector and their amplitudes.
	"""

	for i in range(len(vector)):
		if abs(vector[i])>p.vector_precision:
			#print(basisList[i], bin(basisList[i]), vector[i])
			sketch = sketchBasisVector(p.N, basisList[i])

			print(basisList[i], format(basisList[i], "0{}b".format(2*p.N)), sketch, vector[i])

def binaryString(m, N):
	return format(m, "0{}b".format(2*N))

def checkParams(N, n, nwimpUP, nwimpDOWN):
	"""
	Checks if the parameter values make sense.
	"""
	allOK = 1

	if n>2*N:
		print("WARNING: {0} particles is too much for {2} levels!".format(n, N))
		allOK=0

	if nwimpUP + nwimpDOWN != n:
		print("WARNING: some mismatch in the numbers of particles!")
		allOK=0

	if allOK:
		print("ns check OK.")

def setAlpha(N, d, dDelta):
	"""
	Returns alpha for a given dDelta. 
	"""
	omegaD = 0.5*N*d	
	Delta = d/dDelta
	return 1/(np.arcsinh(omegaD/(Delta)))

def make_nrange_list(p):
	#set the n sectors
	if p.refisn0:
		nrange=[int(p.n0 + 0.5-(p.epsimp/p.U))]
	else: 
		nrange=[p.nref]

	i=1
	while i <= p.nrange:
		nrange.append(nrange[0]+i)
		nrange.append(nrange[0]-i)	
		i+=1

	return nrange	

def saveToFile(savelist, fname):

	with open(fname, "w") as ff:
		for ll in savelist:
			ff.write("	".join([str(i) for i in ll]) + "\n") # works with any number of elements in a line
		
def sketchBasisVector(N, basisVector):
	"""
	Given a basis vector, sketch its level occupanies.
	"""
	Vstring = ""
	for i in range(N):
		Vstring += checkLevel(basisVector, i, N) + "|"

	return Vstring	

def checkLevel(basisVector, i, N):
	"""
	Checks the occupancy of the i-th energy level.
	"""
	off = 2*(N-i)-1	#offset without spin

	up, down = False, False
	if bit(basisVector, off-0):
		up=True
	if bit(basisVector, off-1):
		down=True

	if up and down:
		return "2"
	elif up:
		return "UP"
	elif down:
		return "DO"
	else:
		return "0"	

# BASIS ###########################################################################################

@jit
def makeBase(N, nUP, nDOWN):
	"""
	Creates a basis of a system with N levels, nwimpUP fermions with spin UP and nwimpDOWN fermions with spin DOWN, including the impurity. 
	The impurity level is restricted to exactly one fermion, and is not included in the N levels fo the system. 
	The resulting basis defines the smallest ls subset of the Hamiltonian, where the number of particles and the total spin z of the system 
	are good quantum numbers.
	"""
	resList = []

	for m in range(2**(2*N)):
		if countSetBits(m) == nUP + nDOWN:					#correct number of spins UP and DOWN
			if spinUpBits(m, N, allBits=True) == nUP and spinDownBits(m, N, allBits=True) == nDOWN:
				resList.append(m)

	lengthOfBasis = len(resList)
	resList = np.array(resList)

	return lengthOfBasis, resList

# BIT-WISE ########################################################################################

@jit
def flipBit(n, offset):
	"""Flips the bit at position offset in the integer n."""
	mask = 1 << offset
	return(n ^ mask)

@jit
def countSetBits(m): 
	"""Counts the number of bits that are set to 1 in a given integer."""
	count = 0
	while (m): 
		count += m & 1
		m >>= 1
	return count 

@jit
def bit(m, off):
	"""
	Returns the value of a bit at offset off in integer m.
	"""

	if m & (1 << off):
		return 1
	else:
		return 0

@jit
def spinUpBits(m, N, allBits=False):
	"""
	Counts the number of spin up electrons in the state. If allBits, the impurity level is also counted.
	"""

	count=0
	
	if allBits:
		for i in range(1, 2*N, 2):
			if bit(m, i)==1:
				count+=1
	else:
		for i in range(1, 2*N-2, 2):
			if bit(m, i)==1:
				count+=1

	return count		

@jit
def spinDownBits(m, N, allBits=False):
	"""
	Counts the number of spin down electrons in the state. If allBits, the impurity level is also counted.
	"""

	count=0

	if allBits:
		for i in range(0, 2*N, 2):
			if bit(m, i)==1:
				count+=1	

	else:
		for i in range(0, 2*N-2, 2):
			if bit(m, i)==1:
				count+=1

	return count	

@jit
def clearBitsAfter(m, off, length):
	"""Clears all bits of a number m with length length with offset smaller OR EQUAL off. Used to determine the fermionic +/- prefactor."""
	clearNUm = 0
	for i in range(off+1, length):
		clearNUm += 2**(i)

	return m & clearNUm

@jit
def prefactor_offset(m, off, N):
	"""
	Calculates the fermionic prefactor for a fermionic operator, acting on site given with the offset off. Sets all succeeding bits to zero and count the rest. 
	"""

	count=0
	for i in range(off+1, 2*N-2):	#count bits from offset to (not including) the impurity
		count+=bit(m, i)
	
	"""
	#THIS WORKS MUCH SLOWER BUT IS CLEARER TO UNDERSTAND
	#turns off the impurity bit, as it does not contribute
	turnOffImp = (2**(2*N))-1	#this is the number 100000..., where 1 is at the position of the impurity.
	m = m & turnOffImp
	#set bits to zero
	
	#m = clearBitsAfter(m, off, 2*N)

	#count the 1s of the cleared bit
	count = countSetBits(clearBitsAfter(m, off, 2*N))
	"""

	return -(2 * (count%2)-1)

@jit
def prefactor_offset_imp(m, s, N):
	"""
	Calculates the fermionic prefactor for a fermionic operator acting on the impurity site. 
	"""

	if s==1 and bit(m, 2*N-1):
		return -1
	return 1

# ELECTRON OPERATORS ##############################################################################

@jit
def crcranan(i, j, m, N):
	"""
	Calculates the action of c_iUP^dag c_iDOWN^dag c_jDOWN c_jUP on a basis vector m, given as an integer.
	"""

	offi = 2*(N-1-i)-1	#offset of the bits representing the i-th and j-th energy levels, not accounting for spin
	offj = 2*(N-1-j)-1

	#at each step, the if statement gets TRUE if the operation is valid (does not destroy the state)
	m1 = flipBit(m, offj-0)
	if m>m1:
		m2 = flipBit(m1, offj-1)
		if m1>m2:
			m3 = flipBit(m2, offi-1)
			if m2<m3:
				m4 = flipBit(m3, offi)
				if m3<m4:
					return m4

	return 0  

@jit
def crcrananOnState(i, j, state, N, basisList, lengthOfBasis):
	"""
	Calculates the action of c_iUP^dag c_iDOWN^dag c_jDOWN c_jUP on a state.
	"""

	new_state = np.zeros(lengthOfBasis)
	
	for k in range(lengthOfBasis):

		coef = state[k]
		if coef!=0:

			m = crcranan(i, j, basisList[k], N)

			if m!=0:
				"""
				THIS IS ONE OF THE BOTTLENECKS - given a resulting state (m), find which index in basisList it corresponds to. 
				The solution with the dictionary (of type { state : index_in_basisList }) turns out to be slow. So is the usage 
				of the numpy function np.where(), which finds all occurences of a given value in a list. The current solution is 
				using searchsorted, a function which returns (roughly) the first position of the given value, but needs the list 
				to be sorted. basisList is sorted by construction, so this works. 
				"""
				res = np.searchsorted(basisList, m)

				new_state[res] += coef
				
	return new_state
	
@jit					
def countingOp(i, s, m, N):
	"""
	Calculates the application of the counting operator to a basis state m. Returns 0 or 1, according to the occupation of the energy level.
	"""
	return bit(m, 2*(N-1-i)-1-s)

@jit
def CountingOpOnState(i, s, state, N, basisList, lengthOfBasis):
	"""	
	Calculates the application of the counting operator on a given state vector.
	"""
	new_state = np.zeros(lengthOfBasis)
	
	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			new_state[k] += countingOp(i, s, basisList[k], N)*coef
	
	return new_state		

@jit
def CountingOpOnStateTotal(i, state, N, basisList, lengthOfBasis):
	"""	
	Calculates the application of the counting operator on a given state vector without spin.
	"""
	new_state = np.zeros(lengthOfBasis)
	
	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			new_state[k] += countingOp(i, 0, basisList[k], N)*coef
			new_state[k] += countingOp(i, 1, basisList[k], N)*coef
	
	return new_state

@jit
def CountSCOnState(state, basisList, lengthOfBasis, N, ilast, ifirst=0):
	"""
	Calculates the application of the SUM of counting operators of levels from ilast to ifirst to a given state vector.
	"""	
	new_state = np.zeros(lengthOfBasis)

	for i in range(ifirst, ilast):
		new_state += CountingOpOnStateTotal(i, state, N, basisList, lengthOfBasis)

	return new_state	

@jit
def CountSC2OnState(state, basisList, lengthOfBasis, N, ilast, ifirst=0):
	"""
	Calculates the application of the SUM of counting operators of all levels to a given state vector.
	"""	
	new_state = np.zeros(lengthOfBasis)

	for i in range(ifirst, ilast):
		for j in range(ifirst, ilast):
			new_state += CountingOpOnStateTotal(i, CountingOpOnStateTotal(j, state, N, basisList, lengthOfBasis), N, basisList, lengthOfBasis)

	return new_state	

# HOPPING OPERATOR ##################################################################################

@jit
def hoppingOp(i, j, s, m, N):
	"""
	This is c^dag_i,s c_j,s
	"""

	offi = 2*(N-1-i)-1-s	#offset of the bits representing the i-th and j-th energy levels, not accounting for spin
	offj = 2*(N-1-j)-1-s
	
	if offi<0 or offj<0:
		print("OFFSET SMALLER THAN ZERO!")

	if bit(m, offi)==0:
		if bit(m, offj)==1:
			m = flipBit(m, offj)
			prefactor = prefactor_offset(m, offj, N)
			m = flipBit(m, offi)
			prefactor *= prefactor_offset(m, offi, N)

			return prefactor, m
	return 0, 0

@jit
def hoppingOnState(i, state, N, basisList, lengthOfBasis):
	"""
	Calculates the application of c^dag_i c_i+1 + c^dag_i+1 c_i on a given state vector.
	"""
	new_state = np.zeros(lengthOfBasis)

	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			for s in [0, 1]:
				prefac1, m1 = hoppingOp(i, i+1, s, basisList[k], N)
				prefac2, m2 = hoppingOp(i+1, i, s, basisList[k], N)

				if m1!=0:
					new_state[np.searchsorted(basisList, m1)] += coef * prefac1 
				if m2!=0:
					new_state[np.searchsorted(basisList, m2)] += coef * prefac2

	return new_state		

# ELECTRON-IMPURITY OPERATORS #####################################################################

@jit
def cranimp(i, s, m, N):
	"""
	Calculates the result of c_i,s^dag a_s acting on an integer m. Returns the new basis state and the fermionic prefactor.
	Spin: UP - s=0, DOWN - s=1.
	"""

	offi = 2*(N-1-i)-1-s
	offimp = 2*N-1-s

	m1 = flipBit(m, offimp)
	if m1<m:
		m2=flipBit(m1, offi)
		if m2>m1:
			prefactor = prefactor_offset(m1, offi, N)
			prefactor *= prefactor_offset_imp(m, s, N)

			return prefactor, m2
	return 0, 0

@jit
def crimpan(i, s, m, N):	
	"""
	Calculates the result of a_s^dag c_i,s acting on an integer m. Returns the new basis state and the fermionic prefactor.
	Spin: UP - s=0, DOWN - s=1.
	"""

	offi = 2*(N-1-i)-1-s
	offimp = 2*N-1-s	

	m1 = flipBit(m, offi)
	if m1<m:
		m2=flipBit(m1, offimp)
		if m2>m1:
			prefactor = prefactor_offset(m, offi, N)
			prefactor *= prefactor_offset_imp(m1, s, N)

			return prefactor, m2
	return 0, 0		

# SIAM OPERATORS ##################################################################################

@jit
def impurityEnergyOnState(state, N, epsimp, U, EZ_imp, basisList, lengthOfBasis):
	"""
	Calculates the contribution of the impurity to energy (kinetic and potential energy).
	"""

	#impurity is at position i=-1
	nimpUP = CountingOpOnState(-1, 0, state, N, basisList, lengthOfBasis)
	nimpDOWN = CountingOpOnState(-1, 1, state, N, basisList, lengthOfBasis)
	nimpUPnimpDOWN = CountingOpOnState(-1, 0, CountingOpOnState(-1, 1, state, N, basisList, lengthOfBasis), N, basisList, lengthOfBasis)
	
	new_state = epsimp*(nimpUP + nimpDOWN) + U*nimpUPnimpDOWN + 0.5*EZ_imp*(nimpUP - nimpDOWN)

	return new_state

@jit
def impurityInteractionOnState(i, state, N, basisList, lengthOfBasis):
	"""
	Calculates the contribution of the interaction term between the impurity and the system.
	"""
	new_state = np.zeros(lengthOfBasis)

	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	

			for s in [0, 1]:
				
				prefactor1, m1 = cranimp(i, s, basisList[k], N)
				prefactor2, m2 = crimpan(i, s, basisList[k], N)

				if m1!=0:
					new_state[np.searchsorted(basisList, m1)] += coef * prefactor1 

				if m2!=0:
					new_state[np.searchsorted(basisList, m2)] += coef * prefactor2

	return new_state

# LIN OP SETUP ####################################################################################

class HLinOP(LinearOperator):
	"""
	This is a class, built-in to scipy, which allows for a representation of a given function as a linear operator. The method _matvec() defines how it acts on a vector.
	The operator can be diagonalised using the function scipy.sparse.linalg.eigsh().
	"""	
	def __init__(self, p, basisList, lengthOfBasis, dtype='float64'):
		self.shape = (lengthOfBasis, lengthOfBasis)
		self.dtype = np.dtype(dtype)

		self.basisList = basisList
		self.lengthOfBasis = lengthOfBasis
		self.p = p

	def _matvec(self, state):
		return globals()[self.p.model](state, self.p, self.basisList, self.lengthOfBasis)
		
# PHYSICS #########################################################################################

@jit
def eps(i, D, d, alpha, flat_band, band_level_shift):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""
	if band_level_shift:
		shift = alpha * d/2
	else:
		shift = 0	

	if flat_band:
		N = 2*D/d
		if i >= N/2:
			return -0.5 + shift
		else:
			return +0.5 + shift	
	else:	
		return -D + (i-0.5)*d + shift

# DIAGONALISATION #################################################################################

def LanczosDiagStates_nUPnDOWN(p, n, nUP, nDOWN, NofValues):
	"""
	Returns the spectrum of the Hamiltonian in the subspace, defined by
	D - half-width of the banc
	N - number of levels
	n - number of particles in the system
	nUP, nDOWN - number of particles with spin UP/DOWN, INCLUDING ON THE IMPURITY
	d, alpha, Eimp, U, V - physical constants.
	"""

	lengthOfBasis, basisList = makeBase(p.N, nUP, nDOWN)
	
	if p.all_states:	#compute the entire spectrum
		NofValues = lengthOfBasis

	if p.verbosity:
		checkParams(p.N, n, nUP, nDOWN)

	if lengthOfBasis==1:
		Hs = globals()[p.model](np.array([1]), p, basisList, lengthOfBasis)	#p.model has to be the same as the name of the function where the hamiltonian is defined!

		values = [np.dot(np.array([1]), Hs)]
		vectors = [np.array([1])]

	else:	
		LinOp = HLinOP(p, basisList, lengthOfBasis) 
		values, vectors = eigsh(LinOp, k=max(1, min(lengthOfBasis-1, NofValues)), which="SA")

	return values, vectors, basisList

def LanczosDiag_states(p, n):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies and eigenstates of the Hamiltonian.
	Eigenvectors are already transposed, so SortedVectors[i] corresponds to the eigenstate with energy SortedValues[i], with the
	basis given with SortedBasisLists[i].
	"""

	values, vectors, basisLists = [], [], []

	for nUP in range(max(0, n-min(p.N, n)), min(p.N, n) + 1):	#+1 in range to take into account also the last case	

		nDOWN = n - nUP

		#if not p.include_all_sectors only consider the sectors with Sz = 0 or 1/2
		if nUP - nDOWN == 0 or nUP - nDOWN == 1 or p.include_all_sectors:
			if p.verbosity:
				print(f"In the sector with nUP={nUP}, nDOWN={nDOWN}")

			val, vec, basisList = LanczosDiagStates_nUPnDOWN(p, n, nUP, nDOWN, NofValues=int(p.states_per_sector))

			values.extend(val)
			vectors.extend(np.transpose(vec))
			basisLists.extend([basisList for i in range(len(val))])

		else:
			if p.verbosity:
				print(f"Skipping sector with nUP={nUP}, nDOWN={nDOWN}")
			continue			

	values, vectors, basisLists = np.asarray(values), np.asarray(vectors), np.asarray(basisLists)

	#SORT THE VALUES AND VECTORS BY THE EIGENVALUES
	sortedZippedList = sorted(list(zip(values, vectors, basisLists)), key=lambda x: x[0])

	SortedValues = [i[0] for i in sortedZippedList]
	SortedVectors = [i[1] for i in sortedZippedList]
	SortedBasisLists = [i[2] for i in sortedZippedList]

	return SortedValues, SortedVectors, SortedBasisLists

# CALCULATION #####################################################################################

def findEnergies(p, n):

	a=LanczosDiag(p, n, NofValues=4)
	a=np.array(a) + p.Eshift[p.model]	

	return a

def findEnergiesStates(p, n):

	val, vec, bas = LanczosDiag_states(p, n)
	val=[i+p.Eshift[p.model] for i in val]	

	return val, vec, bas

###################################################################################################

def printGS(p):

	#Print all class attributes:	
	attrs = vars(p)
	print('\n'.join("%s = %s" % item for item in attrs.items()))
	print()

	nrange = make_nrange_list(p)

	GSEs={}
	Es={}

	#parallel stuff:
	num_cores=len(nrange)
	
	res = Parallel(n_jobs=num_cores)(delayed(findEnergiesStates)(p, n)for n in nrange)
	#res = [findEnergiesStates(p, n) for n in nrange]

	for i in range(len(nrange)):
		print("######################################################################################")

		print("Results in the sector with {} particles:".format(nrange[i]))

		val, vec, bas = res[i]

		#fill a dictionary of all energies
		for j in range(len(val)):
			Es[(nrange[i], j)] = val[j]

		#fill a dictionary of only GS energies	
		GSEs[nrange[i]] = val[0]

		print("ENERGIES:")
		print(' '.join("{}".format(E) for E in val))
	
		#print("GROUND STATE:")
		#printV(p, vec[0], bas[0], prec=0.1)
		
		print("IMPURITY OCCUPATION:")
		print(" ".join("{}".format(impOccupation(p, vec[i], bas[i])) for i in range(len(val))))
		print()	

		if p.get_vectors:
			print("EIGENVECTORS:")
			for i in range(min(int(p.get_vectors), len(val))):
				print(f"{i}, E = {val[i]}")
				printVector(p, vec[i], bas[i])
				print()

		if p.get_charge_susceptibility:
			print("CHARGE SUSCEPTIBILITY")
			how_many = int( min( p.how_many_charge_susceptibilities, len(val) ) )

			for ii in range(how_many):
				for jj in range(ii, how_many):
					gij = chargeSusceptibility(vec[ii], vec[jj], bas[ii], bas[jj], p)

					print(f"i={ii}	j={jj}	<i|nimp|j> = {gij}")
		
	print("######################################################################################")

	if p.print_all_energies:
		print("Energies:")
		for n, i in Es.keys():
			print("n = {} i = {} E = {}	DeltaE = {}".format(n, i, Es[(n, i)], Es[(n, i)]-min(GSEs.values())))	

	print("######################################################################################")

	print("Ground state energies:")	
	for n in sorted(nrange):
		print("n = {} E = {}	DeltaE = {}".format(n, GSEs[n], GSEs[n]-min(GSEs.values())))	

def getEs(p):

	nrange = make_nrange_list(p)

	#parallel stuff:
	#num_cores=len(nrange)	
	#res = Parallel(n_jobs=num_cores)(delayed(findEnergiesStates)(p, n)for n in nrange)
	
	#non-parallel:
	res = [findEnergiesStates(p, n) for n in nrange]

	Es, ns, iis = [], [], []
	for i in range(len(nrange)):

		val, vec, bas = res[i]

		for j in range(int(p.excited_states)):

			Es.append(val[j])
			ns.append(nrange[i])
			iis.append(j)

	return Es, ns, iis

def getgij(p):
		
	#ATTENTION!!! nrange=0, n is given, n=p.nref!

	#non-parallel:
	val, vec, bas = findEnergiesStates(p, p.nref)

	CSs, iis, jjs = [], [], []

	for ii in range(int(p.how_many_charge_susceptibilities)):
		for jj in range(ii, int(p.how_many_charge_susceptibilities)):
			
			gij = chargeSusceptibility(vec[ii], vec[jj], bas[ii], bas[jj], p)

			if gij >= p.charge_susceptibility_cutoff:
				CSs.append(gij)
				iis.append(ii)
				jjs.append(jj)

	return CSs, iis, jjs, val	

# HAMILTONIAN #####################################################################################

def SIAM(state, p, basisList, lengthOfBasis):
	"""
	Calculates the action of the Hamiltonian to a given state.
	INPUT:
	d, alpha - physical constants (float)
	state - the state vector (vector)
	N - number of levels (int). There is 2*N available single-particle states (2 for spin)
	basisList - a list of all basis states (list)
	basisDict - a dictionary of positions of basis states in basisList (dictionary)
	lengthOfBasis - the length of the state vector (int)
	OUTPUT:
	the resulting vector, equal to H|state> (np.array)
	"""

	kinetic, magnetic, interaction, impurity, charging, hopping = np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis)
	
	#impurity energy
	impurity += impurityEnergyOnState(state, p.N, p.epsimp, p.U, p.EZ_imp, basisList, lengthOfBasis)

	for i in range(p.N-1):
		niUP = CountingOpOnState(i, 0, state, p.N, basisList, lengthOfBasis)
		niDOWN = CountingOpOnState(i, 1, state, p.N, basisList, lengthOfBasis)

		#kinetic and magnetic terms depend only on nUP and nDOWN
		kinetic += eps(i+1, p.D, p.d, p.alpha, p.flat_band, p.band_level_shift) * (niUP + niDOWN)
		magnetic += 0.5*p.EZ_bulk * niUP
		magnetic +=  - 0.5*p.EZ_bulk * niDOWN


		#impurity interaction
		impurity += (p.V/np.sqrt(p.N-1))*impurityInteractionOnState(i, state, p.N, basisList, lengthOfBasis)
		
		for j in range(p.N-1):
			if p.d*p.alpha!=0:
				interaction += crcrananOnState(i, j, state, p.N, basisList, lengthOfBasis)
		
		#SC chain hopping
		if i!=p.N-2:
			hopping += hoppingOnState(i, state, p.N, basisList, lengthOfBasis)

	#charging energy	
	if p.Ec!=0:
		nSC = CountSCOnState(state, basisList, lengthOfBasis, p.N, p.N)
		nSC2 = CountSC2OnState(state, basisList, lengthOfBasis, p.N, p.N)

		charging += nSC2 - 2*p.n0*nSC

	else:
		charging += 0

	return kinetic + magnetic - p.d*p.alpha*interaction + impurity + p.Ec*charging + p.t*hopping

def SIAM_2channel(state, p, basisList, lengthOfBasis):
	"""
	Calculates the action of the Hamiltonian to a given state.
	INPUT:
	d, alpha - physical constants (float)
	state - the state vector (vector)
	N - number of levels (int). There is 2*N available single-particle states (2 for spin)
	basisList - a list of all basis states (list)
	basisDict - a dictionary of positions of basis states in basisList (dictionary)
	lengthOfBasis - the length of the state vector (int)
	OUTPUT:
	the resulting vector, equal to H|state> (np.array)
	"""

	kinetic, magnetic, interaction1, interaction2, impurity, charging1, charging2 = np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis)
	
	#impurity energy
	impurity += impurityEnergyOnState(state, p.N, p.epsimp, p.U, p.EZ_imp, basisList, lengthOfBasis)

	#FIRST SC
	for i in range((p.N-1)//2):
		niUP = CountingOpOnState(i, 0, state, p.N, basisList, lengthOfBasis)
		niDOWN = CountingOpOnState(i, 1, state, p.N, basisList, lengthOfBasis)

		#kinetic and magnetic terms depend only on nUP and nDOWN
		kinetic += eps(i+1, p.D, 2*p.d, p.alpha1, p.flat_band, p.band_level_shift) * (niUP + niDOWN)	#2*d because the bath is actually two times smaller (there is two of them)
		magnetic += p.EZ_bulk1 * niUP
		magnetic += - p.EZ_bulk1 * niDOWN

		#impurity interaction
		impurity += (p.V1/np.sqrt((p.N-1)/2))*impurityInteractionOnState(i, state, p.N, basisList, lengthOfBasis)
		
		if p.alpha1!=0:
			for j in range((p.N-1)//2):
				interaction1 += crcrananOnState(i, j, state, p.N, basisList, lengthOfBasis)
	
	#SECOND SC	
	kk=0	#this is the actual count		
	for i in range((p.N-1)//2, p.N-1):	
		niUP = CountingOpOnState(i, 0, state, p.N, basisList, lengthOfBasis)
		niDOWN = CountingOpOnState(i, 1, state, p.N, basisList, lengthOfBasis)

		#kinetic and magnetic terms depend only on nUP and nDOWN
		kinetic += eps(kk+1, p.D, 2*p.d, p.alpha2, p.flat_band, p.band_level_shift) * (niUP + niDOWN)	#2*d because the bath is actually two times smaller (there is two of them)
		magnetic += p.EZ_bulk2 * niUP
		magnetic += - p.EZ_bulk2 * niDOWN

		#impurity interaction
		impurity += (p.V2/np.sqrt((p.N-1)/2))*impurityInteractionOnState(i, state, p.N, basisList, lengthOfBasis)
		
		if p.alpha2!=0:
			for j in range((p.N-1)//2, p.N-1):
				interaction2 += crcrananOnState(i, j, state, p.N, basisList, lengthOfBasis)

		kk+=1	

	#charging energy	
	if p.Ec1!=0:
		nSC = CountSCOnState(state, basisList, lengthOfBasis, p.N, p.N//2)
		nSC2 = CountSC2OnState(state, basisList, lengthOfBasis, p.N, p.N//2)

		charging1 += nSC2 - 2*p.n01*nSC

	if p.Ec2!=0:
		nSC = CountSCOnState(state, basisList, lengthOfBasis, p.N, p.N-1, p.N//2)
		nSC2 = CountSC2OnState(state, basisList, lengthOfBasis, p.N, p.N-1, p.N//2)

		charging2 += nSC2 - 2*p.n02*nSC

	return kinetic + magnetic - 2*p.d*p.alpha1*interaction1 - 2*p.d*p.alpha2*interaction2 + impurity + p.Ec1*charging1 + p.Ec2*charging2

# MEASUREMENTS ####################################################################################

def multiply_states(state0, state1, basisList0, basisList1):
	"""
	Computes the inner product <state0|state1>.
	"""
	res = 0
	#compare all basis vectors of state0 and state1. When they match, this adds the product of the corresponding state coefficients to the resulting inner product. 
	i = -1
	for bs0 in basisList0:
		i += 1
		j = -1
		for bs1 in basisList1:
			j += 1		

			if bs0 == bs1:
				#print(bs0, state0[i], state1[j], state0[i] * state1[j])
				res += state0[i] * state1[j] 

	return res			

def impOccupation(p, state, basisList):

	lengthOfBasis = len(basisList)

	npsi = CountingOpOnState(-1, 0, state, p.N, basisList, lengthOfBasis) + CountingOpOnState(-1, 1, state, p.N, basisList, lengthOfBasis)	

	impOcc = np.dot(state, npsi)

	return impOcc

def siteOccupations(p, state, basisList):

	occupations = []

	lengthOfBasis = len(basisList)

	for i in range(-1, p.N-1):
		npsi = CountingOpOnState(i, 0, state, p.N, basisList, lengthOfBasis) + CountingOpOnState(i, 1, state, p.N, basisList, lengthOfBasis)	
		occ = np.dot(state, npsi)

		occupations.append(occ)

	return np.array(occupations) 

def chargeSusceptibility(statei, statej, basisListi, basisListj, p):
	"""
	Computes g_ij = <i|n_imp|j>. Used for calculations described eg. in https://sci-hub.do/https://doi.org/10.1103/PhysRevLett.125.077701
	"""

	lengthOfBasisi = len(basisListi)

	n_statei = CountingOpOnState(-1, 0, statei, p.N, basisListi, lengthOfBasisi) + CountingOpOnState(-1, 1, statei, p.N, basisListi, lengthOfBasisi)

	gij = multiply_states(statej, n_statei, basisListj, basisListi)

	return gij

def makeCSDict(vec, bas, p, how_many):

	gijDict = {}
	for ii in range(how_many):
		for jj in range(ii, how_many):
			gij = chargeSusceptibility(vec[ii], vec[jj], bas[ii], bas[jj], p)
			gijDict[(ii, jj)] = gij

	return gijDict
			
def zero_bandwidth_doublet_overlaps(state, basisList):
	"""
	In the zero bandwidth limit of the two channel problem, in the doublet sector with n=3, Sz=1/2, there are three interesting states with the charge configurations in the (imp, L, R): 
	BCSL: (up, 2, 0)
	BCSR: (up, 0, 2)
	OVERSCREENED: 1/2 * (up, do, up) + 1/2 * (up, up, do) - 1/sqrt(2) * (do, up, up)
	Here we calculate the overlaps of these states with a given state.  
	"""

	BCSL, BCSLbasisList = [1], [44]
	BCSR, BCSRbasisList = [1], [35]
	OS, OSbasisList = [-1/np.sqrt(2), 0.5, 0.5], [26, 38, 41]

	BCSLOverlap = multiply_states(state, BCSL, basisList, BCSLbasisList)
	BCSROverlap = multiply_states(state, BCSR, basisList, BCSRbasisList)
	OSOverlap = multiply_states(state, OS, basisList, OSbasisList)

	return BCSLOverlap, BCSROverlap, OSOverlap

def computeParity(vector, basisList, p):
	"""	
	Computes the expected value of the parity operator for a given vector. 
	This is the overlap with the vector after space inversion - exchanging the left and right channel.
	"""

	newVector, newBasis = parity_transform_vector(vector, basisList, p)

	P = multiply_states(vector, newVector, basisList, newBasis)

	return P 


def parity_transform_basis(basisList, p):
	"""
	The space parity operator in the 2 channel problem is defined as the reflection over the quantum dot level.
	This is equivalent to transforming each basis vector so that its occupied levels in the left channel are now in the right and vice versa.
	The parity operation (or space inversion) is defined as: 
	The state is a string of operators acting on vacuum:
		c_imp_up c_imp_dn c_L1_up c_L1_dn ... c_LN/2_up c_LN/2_dn c_R1_up c_R1_dn ... |vac>
	The partiy transformation changes all c_Li to c_Ri and vice versa. Then the operator string has to be rewritten back in normal order. 
	This is achieved by moving all of the new c_L operators (which are now to the right of the c_Rs) back to the left. While doing this the c_Rs have to be counted in order to 
	account for the fermionic sign.	 
	"""

	newBasisList, prefactorList = [], []
	for m in basisList:
		newm = 0
		totalPrefactorPower = 0

		newBasisList.append(transformBasisVector(m, p))
		prefactorList.append(getTransformationPrefactor(m, p))
	
	return newBasisList, prefactorList

def computeParityOffset(offset, p):
	"""
	Compute where to move a given bit during a space inversion transformation.
	"""

	#there is 2 sites for each level, so N total bits for each channel
	#if the offset is larger than N, we are in the left channel and the bit as to be moved into the right one, and vice versa
	if offset >= p.N - 1 and offset < 2 * (p.N - 1): 	#if in the left channel and not at the impurity level move to the right channel	
		newoffset = offset - (p.N - 1)
	elif offset < p.N - 1:						#if in the right channel move to the left channel
		newoffset = offset + (p.N - 1)			
	elif offset >= 2 * (p.N - 1):				#if on the impurity level stay the same
		newoffset = offset

	return newoffset

def transformBasisVector(m, p):
	"""
	Transforms a basis vector with the partity (space inversion) operation.
	"""

	newm = 0

	#iterate over all bits in the integer m
	for i in range(2 * p.N):

		b = bit(m, i)

		newoffset = computeParityOffset(i, p)

		#this is an iteration from the end of the number (increasing the offset)
		newm += b * 2**(newoffset)

	return int(newm)


def getTransformationPrefactor(m, p):
	"""
	Compute the fermionic prefactor by counting the number of necessary swaps of the creation operators in the parity transformation.
	For a string of (without the impurity levels)
	l1 l2 ... ln r1 r2 ... rn 
	the parity transformation will give
	r1 r2 ... rn l1 l2 ... ln
	To recreate the normal ordering one has to permute all li past all of the ri. This gives sum(li) * sum(ri) operations. 
	The prefactor is then (-1)**( sum(li) * sum(ri) )
	"""

	R = range( 0, p.N-1 )
	L = range( p.N-1, 2*(p.N-1) )

	sumR = 0
	#count the number of bits in the right channel
	for i in R:
		sumR += bit(m, i)

	#count the number of bits in the left channel
	sumL = 0
	for i in L:
		sumL += bit(m, i)

	#mutliply them
	prefactor = (-1) ** (sumL * sumR)

	return prefactor


def parity_transform_vector(vector, basis, p):

	newBasis, prefactors = parity_transform_basis(basis, p)

	#multiply the correct prefactors
	newVector = [prefactors[i] * vector[i] for i in range(len(vector))]

	newVector, newBasis = zip(*sorted(zip(newVector, newBasis), key= lambda x : x[1]))

	return newVector, newBasis

# SAVE OUTPUT #####################################################################################

def Szstring(n):
	if n%2==0:
		return "0"
	elif n%2==1:
		return "0.5"

def nSzi_string(n, i):
	#this is to make it the same as the DMRG output	
	Sz = Szstring(n)
	return f"{n}/{Sz}/{i}/"

def ij_string(n, i, j):
	
	Sz = Szstring(n)
	return f"{n}/{Sz}/{i}/{j}/"	

def generic_print_and_save(n, values, resultString, h5, printMode="single"):
	#print
	if printMode=="list":
		for i in range(len(values)):
			print('	'.join("{}".format(x) for x in values[i]))
		
	else:
		print('	'.join("{}".format(x) for x in values))

	#save
	for i in range(len(values)):
		h5.create_dataset(nSzi_string(n, i) + resultString + "/", data=values[i])

def print_and_save_energies(n, val, h5):
	generic_print_and_save(n, val, "E", h5)	

def print_and_save_impurity_occupancies(n, impOccs, h5):
	generic_print_and_save(n, impOccs, "impurity_occupancies", h5)

def print_and_save_occupancies(n, site_occupancies, h5):
	generic_print_and_save(n, site_occupancies, "site_occupancies", h5, printMode="list")

def print_and_save_parities(n, parities, h5):
	generic_print_and_save(n, parities, "parity", h5)

def print_and_save_charge_susc(n, gijDict, h5):

	for ij in gijDict.keys():
		gij = gijDict[ij]
		i, j = ij

		print(f"n = {n}	Sz = {Szstring(n)}	i = {i}	j = {j}	|<i|nimp|j>| = {gij}")
		h5.create_dataset("charge_susceptibilty/" + ij_string(n, i, j), gij)		

def print_and_save_doublet_overlaps(n, BCSL_overlaps, BCSR_overlaps, OS_overlaps, h5):
	print("BCSL")
	generic_print_and_save(n, BCSL_overlaps, "doublet_overlaps_BCSL", h5)
	print("BCSR")
	generic_print_and_save(n, BCSR_overlaps, "doublet_overlaps_BCSR", h5)
	print("OVERSCREENED")
	generic_print_and_save(n, OS_overlaps, "doublet_overlaps_OS", h5)

import h5py
def calculate_print_and_save(p):
	"""
	The function runs the calculation, prints the results and saves them to a hdf5 file. 
	"""

	#create the output hdf5 file
	h5file = h5py.File("solution.h5", "w")

	#Print out all class attributes:	
	attrs = vars(p)
	print('\n'.join("%s = %s" % item for item in attrs.items()))
	print()

	#setup nrange
	nrange = make_nrange_list(p)

	GSEs={}
	Es={}
	
	if p.parallel:
		#parallel stuff:
		num_cores=len(nrange)
		res = Parallel(n_jobs=num_cores)(delayed(findEnergiesStates)(p, n)for n in nrange)
	else:
		res = [findEnergiesStates(p, n) for n in nrange]

	#Write out and save the data
	for i in range(len(nrange)):
		n = nrange[i]

		print("######################################################################################")

		print("Results in the sector with {} particles:".format(n))

		val, vec, bas = res[i]

		#fill a dictionary of all energies
		for j in range(len(val)):
			Es[(n, j)] = val[j]

		#fill a dictionary of only GS energies	
		GSEs[nrange[i]] = val[0]
		
		print("\nENERGIES:")
		print_and_save_energies(n, val, h5file)	

		print("\nIMPURITY OCCUPATION:")
		impOccs = [impOccupation(p, vec[i], bas[i]) for i in range(len(val))]
		print_and_save_impurity_occupancies(n, impOccs, h5file)


		print("\nSITE OCCUPATIONS:")
		siteOccs = [siteOccupations(p, vec[i], bas[i]) for i in range(len(val))]
		print_and_save_occupancies(n, siteOccs, h5file)
	
		if p.print_parity:
			print("\nPARITY:")

			parities = []
			for j in range(len(val)):

				parity = computeParity(vec[j], bas[j], p)

				parities.append( parity )	

			print_and_save_parities(n, parities, h5file)


		if p.get_vectors:
			#This is special, I am not saving the vectors, just printing them if needed.
			print("\nEIGENVECTORS:")
			for j in range(min(int(p.get_vectors), len(val))):
				print(f"{j}, E = {val[j]}")
				printVector(p, vec[j], bas[j])
				print()

		if p.get_charge_susceptibility:
			print("\nCHARGE SUSCEPTIBILITY")
			how_many = int( min( p.how_many_charge_susceptibilities, len(val) ) )
			gijDict = makeCSDict(vec, bas, p, how_many)
	
			print_and_save_charge_susc(n, gijDict, h5)

		if p.get_doublet_overlaps:	
			print("\nDOUBLET OVERLAPS")

			BCSL_overlaps, BCSR_overlaps, OS_overlaps = [], [], []
			for i in range(len(val)):
				L, R, OS = zero_bandwidth_doublet_overlaps(vec[i], bas[i]) 
			
				BCSL_overlaps.append(L)
				BCSR_overlaps.append(R) 
				OS_overlaps.append(OS)
				
			print_and_save_doublet_overlaps(n, BCSL_overlaps, BCSR_overlaps, OS_overlaps, h5file)

	print("######################################################################################")

	if p.print_all_energies:
		print("Energies:")
		for n, i in Es.keys():
			print("n = {} i = {} E = {}	DeltaE = {}".format(n, i, Es[(n, i)], Es[(n, i)]-min(GSEs.values())))	

	print("######################################################################################")

	print("Ground state energies:")	
	for n in sorted(nrange):
		print("n = {} E = {}	DeltaE = {}".format(n, GSEs[n], GSEs[n]-min(GSEs.values())))	

	h5file.close()
