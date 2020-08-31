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
from numba import jit, njit, jitclass
from numba import types
from numba import int32, float64, float32, int64, boolean, char

from joblib import Parallel, delayed

# READ PARAMS FROM INPUT FILE #####################################################################

def getInput(paramName, inputFile, default, integer=False):
	"""
	#Given paramName (string) and inputFile (string), return the value of the param, saved in the inputFile. If not found, uses default as value.
	"""
	with open(inputFile, "r") as inputF:
		for line in inputF:
			
			a=re.fullmatch(paramName+"\s*=\s*(-?\d*\.?\d*)", line.strip())
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

# CLASS OF PARAMETRERS ############################################################################

spec = [
	("D", int32),
	("N", int32),
	("nref", int32),
	("nrange", int32), 
	("U", float64),
	("epsimp", float64),
	("gamma", float64),
	("alpha", float64),
	("Ec", float64),
	("n0", float64),
	("EZ_imp", float64),
	("EZ_bulk", float64),
	("d", float64),
	("rho", float64),
	("V", float64)
	]

#@jitclass(spec)
class params():
	"""
	The class of parameters necessary to define the Hamiltonian.
	"""

	def __init__(self, inputFile):
		self.D=1	#energy unit
	
		#THIS CONFUSION IS BECAUSE IN THE DMRG CODE N=NUM OF LEVELS, INCLUDING THE IMPURITY
		#THIS IS WRITTEN SO THAT THE inputFile CAN BE THE SAME FOR BOTH CODES

		self.N = getInput("N", inputFile, 10, integer=True)			#number of levels

		self.U = getInput("U", inputFile, 10)			
		self.epsimp = getInput("epsimp", inputFile, -self.U/2)
		
		self.gamma = getInput("gamma", inputFile, 0) 	
		self.alpha = getInput("alpha", inputFile, 0)
		
		self.Ec = getInput("Ec", inputFile, 0)
		self.n0 = getInput("n0", inputFile, self.N)

		self.EZ_imp = getInput("EZ_imp", inputFile, 0)
		self.EZ_bulk = getInput("EZ_bulk", inputFile, 0)

		self.d = 2*self.D/(self.N-1)
		self.rho=1/(2*self.D)
		self.V = np.sqrt(self.gamma/(np.pi*self.rho))

		self.nref = getInput("nref", inputFile, int(self.n0+(0.5-self.epsimp/self.U)) if self.n0+(0.5-self.epsimp/self.U)>0 else self.n0+1, integer=True)		#the central n of the calculation
		self.nrange = getInput("nrange", inputFile, 1, integer=True)		#the range of calculation

# UTILITY #########################################################################################

def printV(p, vector, basisList, prec=0.1):
	"""
	Prints the most prominent elements of a vector and their amplitudes.
	"""
	for i in range(len(vector)):
		if abs(vector[i])>prec:
			#print(basisList[i], bin(basisList[i]), vector[i])
			print(basisList[i], format(basisList[i], "0{}b".format(2*p.N+2)), vector[i])

def checkParams(N, n, nwimpUP, nwimpDOWN):
	"""
	Checks if the parameter values make sense.
	"""
	allOK = 1

	if n>2*N:
		print("WARNING: {0} particles is too much for {2} levels!".format(n, N))
		allOK=0

	if nwimpUP + nwimpDOWN != n+1:
		print("WARNING: some mismatch in the numbers of particles!")
		allOK=0

	if allOK:
		print("Param check OK.")

def setAlpha(N, d, dDelta):
	"""
	Returns alpha for a given dDelta. 
	"""
	omegaD = 0.5*N*d	
	Delta = d/dDelta
	return 1/(np.arcsinh(omegaD/(Delta)))

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
				using searchsorted, a function which returns (rouglhy) the first position of the given value, but needs the list 
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
def CountSCOnState(state, N, basisList, lengthOfBasis):
	"""
	Calculates the application of the SUM of counting operators of all levels to a given state vector.
	"""	
	new_state = np.zeros(lengthOfBasis)

	for i in range(N):
		new_state += CountingOpOnStateTotal(i, state, N, basisList, lengthOfBasis)

	return new_state	

@jit
def CountSC2OnState(state, N, basisList, lengthOfBasis):
	"""
	Calculates the application of the SUM of counting operators of all levels to a given state vector.
	"""	
	new_state = np.zeros(lengthOfBasis)

	for i in range(N):
		for j in range(N):
			new_state += CountingOpOnStateTotal(i, CountingOpOnStateTotal(j, state, N, basisList, lengthOfBasis), N, basisList, lengthOfBasis)

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
	
	new_state = epsimp*(nimpUP + nimpDOWN) + U*nimpUPnimpDOWN + EZ_imp*(nimpUP - nimpDOWN)

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

# HAMILTONIAN #####################################################################################

#@jit
#@profile	
def HonState(state, p, basisList, lengthOfBasis):
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

	kinetic, magnetic, interaction, impurity, charging = np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis)
	
	#impurity energy
	impurity += impurityEnergyOnState(state, p.N, p.epsimp, p.U, p.EZ_imp, basisList, lengthOfBasis)

	for i in range(p.N-1):
		niUP = CountingOpOnState(i, 0, state, p.N, basisList, lengthOfBasis)
		niDOWN = CountingOpOnState(i, 1, state, p.N, basisList, lengthOfBasis)

		#kinetic and magnetic terms depend only on nUP and nDOWN
		kinetic += eps(i+1, p.D, p.d, p.alpha) * (niUP + niDOWN)
		magnetic += p.EZ_bulk * niUP
		magnetic +=  - p.EZ_bulk * niDOWN


		#impurity interaction
		impurity += (p.V/np.sqrt(p.N-1))*impurityInteractionOnState(i, state, p.N, basisList, lengthOfBasis)
		
		for j in range(p.N-1):
			if p.d*p.alpha!=0:
				interaction += crcrananOnState(i, j, state, p.N, basisList, lengthOfBasis)
		

	#charging energy	
	if p.Ec!=0:
		nSC = CountSCOnState(state, p.N, basisList, lengthOfBasis)
		nSC2 = CountSC2OnState(state, p.N, basisList, lengthOfBasis)

		charging += nSC2 - 2*p.n0*nSC #+ (np.ones(lengthOfBasis)*(p.n0**2))
		#print(totalN2-nSC2)
		#print(np.ones(lengthOfBasis)*(p.n0**2))

	else:
		charging += 0

	return kinetic + magnetic - p.d*p.alpha*interaction + impurity + p.Ec*charging

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
		return HonState(state, self.p, self.basisList, self.lengthOfBasis)

# PHYSICS #########################################################################################

@jit
def eps(i, D, d, alpha):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""
	return -D + (i-0.5)*d + alpha*d/2

# DIAGONALISATION #################################################################################

def LanczosDiag_nUPnDOWN(p, n, nUP, nDOWN, NofValues=2, verbosity=False):
	"""
	Returns the spectrum of the Hamiltonian in the subspace, defined by
	D - half-width of the banc
	N - number of levels
	n - number of particles in the system
	nwimpUP, nwimpDOWN - number of particles with spin UP/DOWN, INCLUDING ON THE IMPURITY
	d, alpha, Eimp, U, V - physical constants.
	"""
	
	lengthOfBasis, basisList = makeBase(p.N, nUP, nDOWN)

	if verbosity:
		checkParams(p.N, n, nUP, nDOWN)	#checks if the parameter values make sense

	if lengthOfBasis==1:
		Hs = HonState(np.array([1]), p, basisList, lengthOfBasis)
		values = [np.dot(np.array([1]), Hs)]
	
	else:	
		LinOp = HLinOP(p, basisList, lengthOfBasis) 
		values = eigsh(LinOp, k=max(1, min(lengthOfBasis-1, NofValues)), which="SA", return_eigenvectors=False)[::-1]

	return values

def LanczosDiagStates_nUPnDOWN(p, n, nUP, nDOWN, NofValues=2, verbosity=False):
	"""
	Returns the spectrum of the Hamiltonian in the subspace, defined by
	D - half-width of the banc
	N - number of levels
	n - number of particles in the system
	nUP, nDOWN - number of particles with spin UP/DOWN, INCLUDING ON THE IMPURITY
	d, alpha, Eimp, U, V - physical constants.
	"""
	
	lengthOfBasis, basisList = makeBase(p.N, nUP, nDOWN)
	
	if verbosity:
		checkParams(p.N, n, nUP, nDOWN)

	if lengthOfBasis==1:
		Hs = HonState(np.array([1]), p, basisList, lengthOfBasis)
		values = [np.dot(np.array([1]), Hs)]
		vectors = [np.array([1])]

	else:	
		LinOp = HLinOP(p, basisList, lengthOfBasis) 
		values, vectors = eigsh(LinOp, k=max(1, min(lengthOfBasis-1, NofValues)), which="SA")

	return values, vectors, basisList

def LanczosDiag(p, n, NofValues=4, verbosity=False):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies of the Hamiltonian.
	Computed as a combination of eigenenergies from all smallest subspace with set number of particles (n) and 
	total system spin (Sz = 1/2 (nUP - nDOWN)).
	"""

	val=[]

	for nUP in range(max(0, n-min(p.N, n)), min(p.N, n) + 1):	#+1 in range to take into account also the last case	

		nDOWN = n - nUP

		if verbosity:
			print(nUP, nDOWN)

		val.extend(LanczosDiag_nUPnDOWN(p, n, nUP, nDOWN, NofValues=NofValues, verbosity=verbosity))

	return np.sort(val)	

def LanczosDiag_states(p, n, NofValues=4, verbosity=False):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies and eigenstates of the Hamiltonian.
	Eigenvectors are already transposed, so SortedVectors[i] corresponds to the eigenstate with energy SortedValues[i], with the
	basis given with SortedBasisLists[i].
	"""

	values, vectors, basisLists = [], [], []

	for nUP in range(max(0, n-min(p.N, n)), min(p.N, n) + 1):	#+1 in range to take into account also the last case	

		nDOWN = n - nUP

		if verbosity:
			print(nUP, nDOWN)

		val, vec, basisList = LanczosDiagStates_nUPnDOWN(p, n, nUP, nDOWN, NofValues=5, verbosity=verbosity)

		values.extend(val)
		vectors.extend(np.transpose(vec))
		basisLists.extend([basisList for i in range(len(val))])

	values, vectors, basisLists = np.asarray(values), np.asarray(vectors), np.asarray(basisLists)

	#SORT THE VALUES AND VECTORS BY THE EIGENVALUES
	sortedZippedList = sorted(list(zip(values, vectors, basisLists)), key=lambda x: x[0])

	SortedValues = [i[0] for i in sortedZippedList]
	SortedVectors = [i[1] for i in sortedZippedList]
	SortedBasisLists = [i[2] for i in sortedZippedList]

	return SortedValues, SortedVectors, SortedBasisLists

# DATA ANALYSIS ###################################################################################

def findGap(values, precision=1e-16):
	"""
	From a given set of eigenvalues, finds the first spectroscopic gap, bigger than PRECISION.
	"""

	for i in range(1, len(values)):
		difference = values[i] - values[i-1]
		if difference > precision:
			return difference

def findFirstExcited(values, precision=1e-16):
	"""
	From a given set of eigenvalues, finds the first excited state (state with smallest energy
	different than ground state). The resolution is set by precision.
	"""
	for i in range(1, len(values)):
		if abs(values[i] - values[0]) > precision:
			return values[i]

	print("NOTHING FOUND; PROBLEM IN SPECTRUM?")

def findSecondExcited(values, precision=1e-16):
	"""
	From a given set of eigenvalues, finds the first excited state (state with smallest energy,
	different than ground state). The resolution is set by precision.
	"""
	E1 = findFirstExcited(values, precision)

	for i in range(1, len(values)):
		if values[i] - E1 > precision:
			return values[i]

	print("NOTHING FOUND; PROBLEM IN SPECTRUM?")

def transitionE(N, d, alpha, J, initn, finaln):
	"""
	Calculates the difference between ground state energies of a system with initn and finaln particles. If normalized=True, 
	it returns the energy, normalized with the value without the impurity. 
	"""
	
	initValue = LanczosDiag(N, initn, d, alpha, J, NofValues=2)[0]#/initn
	finalValue = LanczosDiag(N, finaln, d, alpha, J, NofValues=2)[0]#/finaln

	if initn>finaln:	#n->n-1 process
		return initValue - finalValue
	elif initn<finaln:	#n->n+1 process
		return finalValue - initValue

def findSector(basisList, N):
	"""
	For a state vector, written in the basis basisList, find which subspace of the hamiltonian it is from.
	Returns the number of particles n and the total spin of the system Sz.
	The assumption is that all states in the basis are from the same subspace.
	"""

	m = basisList[0]

	n = countSetBits(m) - 1	#impurity is not counted as a particle

	nwimpUP, nwimpDOWN = 0, 0
	for off in range(2*N):
		if off%2==0 and bit(m, off):		#even offset are spin down states 
			nwimpDOWN += 1

		elif off%2!=0 and bit(m, off):	#odd offset correspond to spin up states
			nwimpUP += 1

	return n, nwimpUP, nwimpDOWN		

# MEASURE #########################################################################################

def impOccupation(p, state, basisList):

	lengthOfBasis = len(basisList)

	npsi = CountingOpOnState(-1, 0, state, p.N, basisList, lengthOfBasis) + CountingOpOnState(-1, 1, state, p.N, basisList, lengthOfBasis)	

	impOcc = np.dot(state, npsi)

	return impOcc

# CALCULATION #####################################################################################

def findEnergies(p, n):
	
	Eshift = p.U/2
	Eshift += p.Ec * p.n0**2

	a=LanczosDiag(p, n, NofValues=4)
	a=np.array(a) + Eshift	

	return a

def findEnergiesStates(p, n):
	
	Eshift = p.U/2
	Eshift += p.Ec * p.n0**2

	val, vec, bas = LanczosDiag_states(p, n)
	val=[i+Eshift for i in val]	

	return val, vec, bas

###################################################################################################

def printEnergies(p):
	
	#Print all class attributes:	
	attrs = vars(p)
	print('\n'.join("%s = %s" % item for item in attrs.items()))
	print()

	#set the n sectors
	nrange=[p.nref]
	i=1
	while i <= p.nrange:
		nrange.append(p.nref+i)
		nrange.append(p.nref-i)	
		i+=1

	num_cores=len(nrange)
	energies = Parallel(n_jobs=num_cores)(delayed(findEnergies)(p, n)for n in nrange)


	for i in range(len(nrange)):
		print("Results in the sector with {} particles:".format(nrange[i]))
		print(energies[i])

def printGS(p):
	#Print all class attributes:	
	attrs = vars(p)
	print('\n'.join("%s = %s" % item for item in attrs.items()))
	print()

	#set the n sectors
	nrange=[p.nref+int(0.5-p.epsimp/p.U)]
	i=1
	while i <= p.nrange:
		nrange.append(nrange[0]+i)
		nrange.append(nrange[0]-i)	
		i+=1

	GSEs={}

	#parallel stuff:
	num_cores=len(nrange)
	res = Parallel(n_jobs=num_cores)(delayed(findEnergiesStates)(p, n)for n in nrange)
	for i in range(len(nrange)):
		print("Results in the sector with {} particles:".format(nrange[i]))

		val, vec, bas = res[i]

		GSEs[nrange[i]] = val[0]

		print("ENERGIES:")
		print(' '.join("{}".format(E) for E in val))

		#print("GROUND STATE:")
		#printV(p, vec[0], bas[0], prec=0.1)
		
		print("IMPURITY OCCUPATION:")
		print(" ".join("{}".format(impOccupation(p, vec[i], bas[i])) for i in range(len(val))))
		print()	

	for n in nrange:
		print("n = {} E = {}".format(n, GSEs[n]))	

###################################################################################################
"""
import sys
if len(sys.argv)!=2:
	print("Usage: {0} inputFile".format(sys.argv[0]))
	exit()

inputFile = sys.argv[1]
p=params(inputFile)
printGS(p)
"""
