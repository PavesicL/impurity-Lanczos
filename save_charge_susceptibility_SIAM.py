#!/usr/bin/env python3

from lanczos_diagonalisation import *
import sys
import time

if len(sys.argv)!=2:
	print("Usage: {0} inputFile".format(sys.argv[0]))
	exit()
	
#start timer:
start = time.time()

inputFile = sys.argv[1]
p=params(inputFile)

p.model = "SIAM"

res = getgij(p)

CSs = [res[0]]
iis = [res[1]]
jjs = [res[2]]
Es = [res[3]]


saveName = f"N{p.N}_n{p.nref}_alpha{p.alpha}_U{p.U}_epsimp{p.epsimp}_gamma{p.gamma}_Ec{p.Ec}_n0{p.n0}_nstates{int(p.how_many_charge_susceptibilities)}"
	
saveToFile(CSs, "charge_susceptibilities_"+saveName)
saveToFile(iis, "CS_iis_"+saveName)
saveToFile(jjs, "CS_jjs_"+saveName)
saveToFile(Es, "energies_"+saveName)


end = time.time()
print(f"Wall time: {round(end - start, 1)} s")
