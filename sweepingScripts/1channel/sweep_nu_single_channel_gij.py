#!/usr/bin/env python3

import sys
import numpy as np
sys.path.insert(1, '../../')
from lanczos_diagonalisation import getgij, params, saveToFile

#PARSE INPUT:

if len(sys.argv)!=6:
	print("Usage: {0} inputFile nu_min nu_max nu_step n".format(sys.argv[0]))
	exit()

inputFile = sys.argv[1]
start = float(sys.argv[2])
stop = float(sys.argv[3])
step = float(sys.argv[4])
n = int(sys.argv[5])

#MAKE LIST OF nus
nus = [start]
numsteps = int(round((stop-start)/step, 1))

for i in range(numsteps-1):
	nus.append(round(start+(i+1)*step, 5))


#RUN CALCULATION

print(nus)
okay = input(f"Run {len(nus)} calculations? y/n\n")
if okay == "y":
	
	p=params(inputFile)

	p.nref = n
	p.nrange = 0

	p.model = "SIAM"

	CSs, iis, jjs, Es = [], [], [], []
	for nu in nus:
		print(nu)

		p.epsimp = (0.5 - nu) * p.U

		res = getgij(p)

		CSs.append([nu] + res[0])
		iis.append([nu] + res[1])
		jjs.append([nu] + res[2])
		Es.append([nu] + res[3])

	saveName = f"N{int(p.N)}_n{int(n)}_alpha{p.alpha}_U{p.U}_gamma{p.gamma}_Ec{p.Ec}_n0{p.n0}_nstates{(p.how_many_charge_susceptibilities)}"
	
	saveToFile(CSs, "ZBM_charge_susceptibilities_"+saveName)
	saveToFile(iis, "ZBM_CS_iis_"+saveName)
	saveToFile(jjs, "ZBM_CS_jjs_"+saveName)
	saveToFile(Es, "ZBM_energies_"+saveName)
