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

		self.N = getInput("N", inputFile, 9, integer=True)			#number of levels

		self.U = getInput("U", inputFile, 10)			
		self.epsimp = getInput("epsimp", inputFile, -self.U/2)
		
		self.gamma1 = getInput("gamma1", inputFile, 0) 	
		self.alpha1 = getInput("alpha1", inputFile, 0)
		self.gamma2 = getInput("gamma2", inputFile, 0) 	
		self.alpha2 = getInput("alpha2", inputFile, 0)
		
		self.Ec1 = getInput("Ec1", inputFile, 0)
		self.n01 = getInput("n01", inputFile, self.N/2)
		self.Ec2 = getInput("Ec2", inputFile, 0)
		self.n02 = getInput("n02", inputFile, self.N/2)		

		self.EZ_imp = getInput("EZ_imp", inputFile, 0)
		self.EZ_bulk1 = getInput("EZ_bulk1", inputFile, 0)
		self.EZ_bulk2 = getInput("EZ_bulk2", inputFile, 0)

		self.d = 2*self.D/(self.N-1)
		self.rho=1/(2*self.D)
		self.V1 = np.sqrt(self.gamma1/(np.pi*self.rho))
		self.V2 = np.sqrt(self.gamma2/(np.pi*self.rho))

		self.nref = getInput("nref", inputFile, int(self.n01+self.n02+(0.5-self.epsimp/self.U)) if self.n01+self.n02+(0.5-self.epsimp/self.U)>0 else self.n01+self.n02+1, integer=True)		#the central n of the calculation
		self.nrange = getInput("nrange", inputFile, 1, integer=True)		#the range of calculation

		self.model = "SIAM_2channel"
		self.Eshift = self.U/2 + self.Ec1*self.n01**2 + self.Ec2*self.n02**2	#MODEL DEPENDENT!!!

		self.refisn0 = getInput("refisn0", inputFile, 0)

###################################################################################################

if len(sys.argv)!=2:
	print("Usage: {0} inputFile".format(sys.argv[0]))
	exit()

inputFile = sys.argv[1]
p=params(inputFile)


printGS(p)
