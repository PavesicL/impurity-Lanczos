#!/usr/bin/env python3

from lanczos_diagonalisation import *
import sys

# CLASS OF PARAMETRERS ############################################################################

#@jitclass(spec)
class params():
	"""
	The class of parameters necessary to define the Hamiltonian.
	"""

	def __init__(self, inputFile):
		self.D=1	#energy unit

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

		self.model = "SIAM"
		self.Eshift = self.U/2 + self.Ec*self.n0**2	#MODEL DEPENDENT!!!

		self.refisn0 = getInput("refisn0", inputFile, 0)

###################################################################################################

if len(sys.argv)!=2:
	print("Usage: {0} inputFile".format(sys.argv[0]))
	exit()

inputFile = sys.argv[1]
p=params(inputFile)
printGS(p)
