#
# A Params() object which describes Orbital numbering
#
# and Provides a global dictionary of tensors
# reads them off disk. Things like that. 
#

import numpy, scipy
#from scipy import special
from numpy import array
from numpy import linalg
# standard crap
import os, sys, time, math, re, random, cmath
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, log, pi, sqrt, isnan
import copy
from LooseAdditions import *
try:
	import mkl
except ImportError:
	print "No MKL"

#
# Provides a dictionary of tensors used in the computation. 
# and a set of calculation parameters. 
#

class CalculationParameters: 
	def __init__(self): 

		import os, sys
		MoleculeName = ''
		if len(sys.argv) < 2:
			print "Ideally you give me a molecule name, Defaulting to /Integrals"
		else: 
			MoleculeName = sys.argv[1]

		self.MoleculeName = MoleculeName # Defined Globally in Main.py 
		self.nocc = 4
		self.nvirt = 4
		self.nmo = 8    # These will be determined on-the-fly reading from disk anyways. 
		self.occ = []
		self.virt = []
		self.all = []
		self.alpha = []
		self.beta = []

		# Try to get a faster timestep by freezing the CoreOrbitals. 
		# if it's not None, then freeze that many orbitals (evenly alpha and beta.) 
		self.FreezeCore = 8 

		self.AvailablePropagators = ["phTDA2TCL","Whole2TCL", "AllPH"]
		self.Propagator = "phTDA2TCL"
		self.Correlated = True
		self.SecularApproximation = 0 # 0 => Nonsecular 1=> Secular 
		self.ImaginaryTime = True
		
		self.Temperature = 303.15*2
		self.TMax =  250.0 
		self.TStep = 0.01
		self.tol = 1e-9
		self.Safety = .9 #Ensures that if Error = Emax, the step size decreases slightly
		self.RK45 = True
		self.LoadPrev = False
#		self.MarkovEta = 0.05
		self.MarkovEta = 0.001
		self.Tc = 3000.0
		self.Compiler = "gcc"
		self.Flags = ["-mtune=native", "-O3"]
		self.latex = True
		self.fluorescence = True # If true, this performs the fluorescence calculations in SpectralAnalysis
		# above a certain energy 
		# no relaxation is included and the denominator is the bare electronic 
		# denominator to avoid integration issues.
		self.UVCutoff = 300.0/27.2113 # No Relaxation above 1eV		

		try:
			mkl.set_num_threads(8)
		except NameError:
			print "No MKL so I can't set the number of threads"
		
		# This is a hack switch to shut off the Boson Correlation Function 
		# ------------------------------------------------------------------------
		# Adiabatic = 0 # Implies numerical integration of expdeltaS and bosons. 
		# Adiabatic = 1 # Implies numerical integration of expdeltaS no bosons. 
		# Adiabatic = 2 # Implies analytical integration of expdeltaS, no bosons.  
		# Adiabatic = 3 # Implies analytical integration of expdeltaS, no bosons, and the perturbative terms are forced to be anti-hermitian. 		
		# Adiabatic = 4 # Implies Markov Approximation. 
		self.Adiabatic = 0

		self.Inhomogeneous = False
		self.InhomogeneousTerms = self.Inhomogeneous
		self.Inhomogenous = False
		self.Undressed = True
		self.ContBath = True # Whether the bath is continuous/There is a continuous bath. 
		self.ReOrg = True 
		self.FiniteDifferenceBosons = False

		self.DipoleGuess = True # if False, a superposition of bright states will be used. 
		self.AllDirections = True # Will initalize three concurrent propagations. 
		self.DirectionSpecific = True		
		
		self.InitialDirection = -1 # 0 = x etc. Only excites in the x direction -1=isotropic
		self.BeginWithStatesOfEnergy = None 
#		self.BeginWithStatesOfEnergy = 18.7/27.2113
#		self.BeginWithStatesOfEnergy = 18.2288355687/27.2113
		self.PulseWidth = 1.7/27.2113
		
		self.Plotting = True
		self.DoCisDecomposition = True
		self.DoBCT = True # plot and fourier transform the bath correlation tensor. 
		self.DoEntropies = True
		if (self.Undressed): 
			self.DoEntropies = False
		self.FieldThreshold = pow(10.0,-7.0)
		self.ExponentialStep = False  #This will be set automatically if a matrix is made. 

		self.LegendFontSize = 14 
		self.LabelFontSize = 16
		
		print "--------------------------------------------"		
		print "Running With Overall Parameters: "
		print "--------------------------------------------"
		print "self.MoleculeName", self.MoleculeName 
		print "self.AllDirections", self.AllDirections 
		print "self.Propagator", self.Propagator 
		print "self.Temperature", self.Temperature 
		print "self.TMax", self.TMax 
		print "self.TStep", self.TStep 
		print "self.MarkovEta", self.MarkovEta
		print "self.Adiabatic", self.Adiabatic
		print "self.Inhomogeneous", self.Inhomogeneous
		print "self.Undressed", self.Undressed
		print "self.Correlated", self.Correlated 
		print "self.SecularApproximation", self.SecularApproximation
		print "self.DipoleGuess", self.DipoleGuess
		print "self.BeginWithStatesOfEnergy ", self.BeginWithStatesOfEnergy
		print "self.DoCisDecomposition", self.DoCisDecomposition 
		print "self.DoBCT", self.DoBCT 
		print "self.DoEntropies", self.DoEntropies 
		print "self.FieldThreshold", self.FieldThreshold 
		return 
		
	def TypeOfIndex(self,dex): 
		typ = lambda XX: (0 if XX in self.occ else 1)
		return map(typ,dex)
	def OVShape(self,OsAndVs): 
		tore = []
		for T in OsAndVs: 
			if (T == 0) :# occ. 
				tore.append(self.nocc)
			elif (T == 1): 
				tore.append(self.nvirt)
		return tuple(tore)
	def ShiftVirtuals(self): 
		return max(self.occ)+1

	# The numbering is Occ(Alpha), Occ(beta), Virt(alpha), Virt(beta)
	# index is clear, types are o=0 or v=1
	def SpinFlipOfIndex(self,ind,types): 
		tore = [0 for I in range(len(ind))]
		nspato = len(self.occ)/2
		nspatv = len(self.virt)/2
		for I in range(len(ind)): 
			if types[I] == 0: 
				if (ind[I] >= nspato): 
					tore[I] = ind[I] - nspato 
				else: 
					tore[I] = ind[I] + nspato 
			elif types[I] == 1: 
				if (ind[I] >= nspatv): 
					tore[I] = ind[I] - nspatv 
				else: 
					tore[I] = ind[I] + nspatv 			
			else : 
				print "SpinFlip of index... Error... "
				raise Exception("Fouuuuuuuu")
		return tuple(tore)
	def Homo(self,spin = "alpha"):
		if (spin == "alpha"): 
			return self.occ[len(self.occ)/2]
		else: 
			return self.occ[-1]
	def Lumo(self,spin = "alpha", plus=0):
		if (spin == "alpha"): 
			return self.virt[0+plus]
		else: 
			return self.virt[len(self.virt)/2 + plus]
	def GeneralTensorShape(self,rank): 
		tore = []
		for rr in range(rank): 
			tore.append(self.nocc+self.nvirt)
		return tuple(tore)
	# This is the shape of an excitation operator. 
	def UpDownOVShape(rank):
		tore = []
		for rr in range(rank): 
			tore.append(self.nocc+self.nvirt)
		return tuple(tore)
	
Params = CalculationParameters()

# Holds raw numerical data for tensors. 
class TensorDictionary(dict): 
	def __init__(self): 
		dict.__init__(self)
		self.Types = dict()  # OV types for each key
		return
	def AllElementKeys(self): 
		tore = []
		for T in self.iterkeys(): 
			iter = numpy.ndindex(self[T].shape)
			for I in iter: 
				tore.append((T,I))
		return tore
	def Print(self): 
		print "Tensor Dictionary Contents: "
		for O in self.iterkeys(): 
			print "T: ",O," : ", self[O].shape," Sum: " ,numpy.sum(self[O])," L2norm: ", cmath.sqrt(numpy.sum(self[O]*self[O]))
		return 
#	def Zero(self,Name): 
#		shp = self[Name].shape 
#		self[Name] = numpy.zeros(shp,dtype=complex)
	def Create(self,name,rank,OsAndVs = None):
		if (name in self): 
			print "Creating what already exists!"
			raise Exception("Too many tensors")
			return
		shape=Params.GeneralTensorShape(rank)
		if (OsAndVs == None): 
			shape = Params.GeneralTensorShape(rank)
		else: 
			self.Types[name] = OsAndVs
			shape = shape=Params.OVShape(OsAndVs)
# check the size. 
		sz = 1.0
		for i in shape: 
			sz *= i
		typ = numpy.dtype(numpy.complex)
		sz *= typ.itemsize
		if (sz > 100*pow(1024,2)): 
			print name," Alloc of > 100mb: ", sz/(100*pow(1024,2)), " Shape ", shape
# Finally actually make it. 
		self[name] = numpy.zeros(shape,dtype = complex)
		return 
	def PrintSample(self, akey,Num = 20,PrintNearZero = False):
			print "Sample of: ", akey
			I = 0
			TDEX = numpy.ndindex(AllTensors[akey].shape)
			for dex in TDEX:
				if (not PrintNearZero): 
					if (abs(AllTensors[akey][dex]) < pow(10.0,-6.0)): 
						continue
				if (Params.TypeOfIndex(dex) == [0,0,1,1]): 
					if (dex[0] != dex[1] and dex[2] != dex[3]):
						print dex, " : ", AllTensors[akey][dex]
						I += 1
				else: 
					print dex, " : ", AllTensors[akey][dex]
				if I > Num : 
					break

# Unlike Integrals which are stored globally in the AllTensors Object.
# State Vectors (which are lists of tensors) are held in one of these.  
class StateVector(TensorDictionary): 
	def __init__(self): 
		TensorDictionary.__init__(self)
		return 
	def Print(self): 
		for k in self.iterkeys(): 
			print "Tensor:", k
			print self[k]
		return 
	def Save(self,Outputfile):
		import pickle
		of = open(Outputfile,"w")						
		pickle.Pickler(of,0).dump(self)
		of.close()
		return 
	def Load(self,Outputfile): 
		import pickle
		of = open(Outputfile,"rb")						
		self = pickle.Unpickler(of).load()
		of.close()
		return 	
	def clone(self): 
		return copy.deepcopy(self)
	def Fill(self,Value = complex(0.0,0.0)): 
		for T in self.iterkeys(): 
			self[T].fill(Value)
	def Add(self,Other,fac = 1.0): 
		for T in self.iterkeys(): 
			self[T] += fac*Other[T]
		return 		
	def MultiplyScalar(self,fac): 		
		for T in self.iterkeys(): 
			self[T] *= fac
		return 
	def LinearCombination(self,SelfFac,OtherFac,Other): 		
		tore = self.clone()
		tore.MultiplyScalar(SelfFac)
		for T in tore.iterkeys(): 
			tore[T] += OtherFac*Other[T]
		return tore

	def TwoCombo(self,SelfFac,Other1, OtherFac1, Other2, OtherFac2): 		
		tore = self.clone()
		tore.MultiplyScalar(SelfFac)
		for T in tore.iterkeys(): 
			tore[T] += OtherFac1*Other1[T]
			tore[T] += OtherFac2*Other2[T]
		return tore

	def ThreeCombo(self,SelfFac,Other1, OtherFac1, Other2, OtherFac2, Other3, OtherFac3): 		
		tore = self.clone()
		tore.MultiplyScalar(SelfFac)
		for T in tore.iterkeys(): 
			tore[T] += OtherFac1*Other1[T]
			tore[T] += OtherFac2*Other2[T]
			tore[T] += OtherFac3*Other3[T]
		return tore

	def FourCombo(self,SelfFac,Other1, OtherFac1, Other2, OtherFac2, Other3, OtherFac3, Other4, OtherFac4): 		
		tore = self.clone()
		tore.MultiplyScalar(SelfFac)
		for T in tore.iterkeys(): 
			tore[T] += OtherFac1*Other1[T]
			tore[T] += OtherFac2*Other2[T]
			tore[T] += OtherFac3*Other3[T]
			tore[T] += OtherFac4*Other4[T]
		return tore

	def FiveCombo(self,SelfFac,Other1, OtherFac1, Other2, OtherFac2, Other3, OtherFac3, Other4, OtherFac4, Other5, OtherFac5): 		
		tore = self.clone()
		tore.MultiplyScalar(SelfFac)
		for T in tore.iterkeys(): 
			tore[T] += OtherFac1*Other1[T]
			tore[T] += OtherFac2*Other2[T]
			tore[T] += OtherFac3*Other3[T]
			tore[T] += OtherFac4*Other4[T]
			tore[T] += OtherFac5*Other5[T]
		return tore

	def CheckHermitianSym(self): 
		print "Not yet implemented"
	def CheckSpinSymmetry(self): 
		AllKeys = self.AllElementKeys()
		for k in AllKeys: 
			Flipped = Params.SpinFlipOfIndex(k[1],self.Types[k[0]])
			if (((self[k[0]])[k[1]] - (self[k[0]])[Flipped]) > pow(10.0,6.0)): 
				print "Spin Symmetry Broken... "
				print "types: ", self.Types[k[0]], " Ten: ",k[0]," ele:", k[1], " vs: ", Flipped, " at index: ", 
				print (self[k[0]])[k[1]] , (self[k[0]])[Flipped]
	def InnerProduct(self,Other): 		
		tore = complex(0.0,0.0)
		for T in self.iterkeys(): 
			tore += numpy.sum(self[T].conj()*Other[T])
		return tore	

# Holds all the raw numerical data for all tensors. 
class IntegralDictionary(TensorDictionary): 
	def __init__(self): 
		TensorDictionary.__init__(self)
		self.ReadFromDiskAndAllocate()
		#self.MakeHFEnergy()
		return
	
	def Delta(self,indices,types,signs): 
		tore = 0.0
		for i in range(len(indices)): 
			if (types[i] == 0):
				tore += signs[i]*self["h_hh"][indices[i]][indices[i]]
			elif (types[i] == 1): 
				tore += signs[i]*self["h_pp"][indices[i]][indices[i]]
		return tore 		

	# Types are 0 or 1 = V
	# signs are 1 or -1
	# returns e^{i \Delta time}
	def TimeExponential(self,types,signs,time): 
		ShapeOutput = Params.OVShape(types)
		DeltaIndex = lambda X: self.Delta(X,types,signs)
		ToExp = numpy.fromfunction(DeltaIndex, ShapeOutput)
		return numpy.exp(ToExp*complex(0.0,time))
		
	def CheckHermitianSymmetry(self): 
		for Tensor1 in Integrals.iterkeys(): 
			Ten1 = Integrals[Tensor1]
			if (Ten1.max() > 100.0): 
				print "Too Big.", Ten1.max()
			if (Ten1.min() < -100.0): 				
				print "Too Small.", Ten1.min()
			for Tensor2 in Integrals.iterkeys():
				Ten2 = Integrals[Tensor2]
				if (not (len(Ten1.shape) == len(Ten2.shape) == 4)): 
					continue 
				typs1 = Integrals.Types[Tensor1]
				typs2 = Integrals.Types[Tensor2]
				if (typs1[0] == typs2[2] and typs1[1] == typs2[3] and typs1[2] == typs2[0] and typs1[3] == typs2[1]): 
					print "Hermitian Partners, ", Tensor1, Tensor2
					it = numpy.ndindex(Ten1.shape)					
					iter2 = [0,0,0,0]
					for iter in it: 
						iter2[0], iter2[1], iter2[2], iter2[3] = iter[2], iter[3], iter[0], iter[1]
						if ( abs(Ten1[iter] - Ten2[tuple(iter2)]) > pow(10.0,-10.0)):
							print "Hermitian asymmetry:", Tensor, " Types: ", typs, " : ", iter, Integrals[Tensor][iter], " Vs: ",Params.SpinFlipOfIndex(iter,typs) ," at ", Integrals[Tensor][Params.SpinFlipOfIndex(iter,typs)]
							raise Exception("God is dead.")
		return True


	def CheckSpinSymmetry(self): 
		for Tensor in Integrals.iterkeys(): 
			if (not Tensor in Integrals.Types): 
				continue
			typs = Integrals.Types[Tensor]
			Ten = Integrals[Tensor]
			it = numpy.ndindex(Ten.shape)
			for iter in it: 
				if ( abs(Integrals[Tensor][iter] - Integrals[Tensor][Params.SpinFlipOfIndex(iter,typs)]) > pow(10.0,-10.0)):
					print "Spin asymmetry:", Tensor, " Types: ", typs, " : ", iter, Integrals[Tensor][iter], " Vs: ",Params.SpinFlipOfIndex(iter,typs) ," at ", Integrals[Tensor][Params.SpinFlipOfIndex(iter,typs)]
					return False
		return True

	# doesn't include the nuclear repulsion energy. 
	def MakeHFEnergy(self): 
		PHF = numpy.eye(Params.nocc)
		EHF = 0.0
		for mu in range(Params.nocc): 
			for nu in range(Params.nocc): 
				EHF += self["h_hh"][mu][nu] * PHF[mu][nu]
				for lam in range(Params.nocc): 	
					for sig in range(Params.nocc): 
						EHF += 0.5*PHF[mu][nu]*PHF[sig][lam]*(self["v_hhhh"][mu][nu][lam][sig] - self["v_hhhh"][mu][lam][nu][sig])


		print "Hartree-Fock energy (no nuclear repulsion) : ", EHF
		return 
		EHF = 0.0						
		FockMatrix = numpy.ndarray(shape=(Params.nocc,Params.nocc),dtype=float)
		for mu in range(Params.nocc): 
			for nu in range(Params.nocc): 
				FockMatrix[mu][nu] += self["h_hh"][mu][nu]
				for lam in range(Params.nocc): 	
					for sig in range(Params.nocc): 
						FockMatrix[mu][nu] += PHF[lam][sig]*(self["v_hhhh"][mu][nu][lam][sig] - self["v_hhhh"][mu][lam][nu][sig])
		EHF = 0.5 * numpy.sum( (self["h_hh"] + FockMatrix)*PHF )
		print "Hartree-Fock energy (no nuclear repulsion) : ", EHF		
		# Make the large V matrix and P and make the energy again... 
		ntot = Params.nocc+Params.nvirt 
		no = Params.nocc
		self["p"] = numpy.zeros(shape=(ntot,ntot),dtype = float)
		self["h"] = numpy.zeros(shape=(ntot,ntot),dtype = float)
		self["v"] = numpy.zeros(shape=(ntot,ntot,ntot,ntot),dtype = float)
		for i in range(no): 
			self["p"][i][i] = 1.0				
		tmp = self["h_hh"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["h"][ind] += tmp[ind]
		tmp = self["h_hp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["h"][ind[0]][ind[1]+no] += tmp[ind]
		tmp = self["h_pp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["h"][ind[0]+no][ind[1]+no] += tmp[ind]
		tmp = self["h_ph"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["h"][ind[0]+no][ind[1]] += tmp[ind]
		tmp = self["v_hhhh"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]][ind[1]][ind[2]][ind[3]] += tmp[ind]
		tmp = self["v_hhhp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]][ind[1]][ind[2]][ind[3]+no] += tmp[ind]
		tmp = self["v_hhpp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]][ind[1]][ind[2]+no][ind[3]+no] += tmp[ind]
		tmp = self["v_hphp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]][ind[1]+no][ind[2]][ind[3]+no] += tmp[ind]
		tmp = self["v_hppp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]][ind[1]+no][ind[2]+no][ind[3]+no] += tmp[ind]
		tmp = self["v_pppp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]+no][ind[1]+no][ind[2]+no][ind[3]+no] += tmp[ind]
		tmp = self["v_phhp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]+no][ind[1]][ind[2]][ind[3]+no] += tmp[ind]
		tmp = self["v_ppph"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]+no][ind[1]+no][ind[2]+no][ind[3]] += tmp[ind]
		tmp = self["v_phph"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]+no][ind[1]][ind[2]+no][ind[3]] += tmp[ind]
		tmp = self["v_hhph"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]][ind[1]][ind[2]+no][ind[3]] += tmp[ind]
		tmp = self["v_pphp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]+no][ind[1]+no][ind[2]][ind[3]+no] += tmp[ind]
		tmp = self["v_phpp"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]+no][ind[1]][ind[2]+no][ind[3]+no] += tmp[ind]
		tmp = self["v_hphh"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]][ind[1]+no][ind[2]][ind[3]] += tmp[ind]
		tmp = self["v_phhh"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]+no][ind[1]][ind[2]][ind[3]] += tmp[ind]
		tmp = self["v_hpph"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]][ind[1]+no][ind[2]+no][ind[3]] += tmp[ind]
		tmp = self["v_pphh"] 
		for ind in numpy.ndindex(tmp.shape): 
			self["v"][ind[0]+no][ind[1]+no][ind[2]][ind[3]] += tmp[ind]
		EHF = 0.0						
		FockMatrix = numpy.ndarray(shape=(ntot,ntot),dtype=float)
		for mu in range(ntot): 
			for nu in range(ntot): 
				FockMatrix[mu][nu] += self["h"][mu][nu]
				for lam in range(ntot): 	
					for sig in range(ntot): 
						FockMatrix[mu][nu] += self["p"][lam][sig]*(self["v"][mu][nu][lam][sig] - self["v"][mu][lam][nu][sig])
		EHF = 0.5 * numpy.sum( (self["h"] + FockMatrix)* self["p"] )
		print "Hartree-Fock energy (no nuclear repulsion) : ", EHF	
		return 

	def MakeDuplicate(self,oldt,newt): 
		self[newt] = self[oldt].copy()
		return
	def CanDivide(self,t1,t2): 
		if (self[t1].shape != self[t2].shape): 
			return False
		iter1 = numpy.ndindex(self[t1].shape)
		for d in iter1: 
			if (((self[t1])[d] != 0.0) and ((self[t2])[d] == 0.0)): 
				print "Cant Divide at pos: ",d
				print (self[t1])[d], " :: ", (self[t2])[d]
		return 
	def ReadFromDiskAndAllocate(self): 
		print "ReadFromDiskAndAllocate"
		ocs=[]
		vir=[]
		Hootmp = []
		Hvvtmp = []
		DirectoryPrefix = "./Integrals"+Params.MoleculeName+"/"
		hoo = open(DirectoryPrefix+"HOO","r")
		for s in hoo: 
			i = int(s.split()[0])
			j = int(s.split()[1])
			ocs.append(i)
			ocs.append(j)						
			v = float(s.split()[2])
			Hootmp.append([i,j,v])
		hoo.close()
		del hoo
		hvv = open(DirectoryPrefix+"HVV","r")
		for s in hvv: 
			i = int(s.split()[0])
			j = int(s.split()[1])
			vir.append(i)
			vir.append(j)
			v = float(s.split()[2])
			Hvvtmp.append([i,j,v])
		hvv.close()
		del hvv
			
		
		Params.occ=list(set(tuple(ocs)))
		Params.nocc=len(Params.occ)
		Params.virt=list(set(tuple(vir)))
		Params.nvirt=len(Params.virt)
		Params.occ.sort()
		Params.virt.sort()
		
		if (Params.nocc <= Params.FreezeCore): 
			print "Invalid Number of Frzn orbitals."
			Params.FreezeCore = None
		#Freezing map takes a qchem-numbered orbital and maps it to a frozen-numbered orbital
		FreezingMap = None
		if (Params.FreezeCore != None): 
			FreezingMap = dict()
			print "Freezing ", Params.FreezeCore, " Core Orbitals. "
			alphas = Params.occ[:len(Params.occ)/2]
	 		betas = Params.occ[len(Params.occ)/2:]
			for tmp in range(Params.FreezeCore/2): 
				FreezingMap[alphas.pop(0)] = 10000000
				FreezingMap[betas.pop(0)] = 10000000				
			alphas.extend(betas)
			for a in alphas: 
				FreezingMap[a] = alphas.index(a)
			for K in FreezingMap.iterkeys():
				print K, "=>",FreezingMap[K]
			Params.occ = rlen(alphas)
			Params.nocc=len(Params.occ)
		else: 
			FreezingMap = dict()
			for O in Params.occ: 
				FreezingMap[O] = O 
						
		# The numbering is Occ(Alpha), Occ(beta), Virt(alpha), Virt(beta)
		print "Occ:", Params.occ
		print "Virt:", Params.virt
		print "nOcc:", Params.nocc
		print "nVirt:", Params.nvirt
		
		Params.nmo = Params.nocc+Params.nvirt
		Params.all.extend(Params.occ)
		Params.all.extend(Params.virt)
		Params.alpha = sorted(Params.occ)[:len(Params.occ)/2]		
		Params.beta = sorted(Params.occ)[len(Params.occ)/2:]
		Params.alpha.extend(sorted(Params.virt)[:len(Params.virt)/2])
		Params.beta.extend(sorted(Params.virt)[len(Params.virt)/2:])				
		print "Allocating tensors..." 
		if (False): 
			self["h"] = numpy.ndarray(shape=Params.GeneralTensorShape(2),dtype = complex)
			self["v"] = numpy.ndarray(shape=Params.GeneralTensorShape(4),dtype = complex)
			self["r"] = numpy.ndarray(shape=Params.GeneralTensorShape(4),dtype = complex)
			self["dr"] = numpy.ndarray(shape=Params.GeneralTensorShape(4),dtype = complex)
		# The above Won't be used. 		
		self["h_hh"] = numpy.zeros(shape=Params.OVShape([0,0]),dtype = float)
		self["h_hp"] = numpy.zeros(shape=Params.OVShape([0,1]),dtype = float)	
		self["h_pp"] = numpy.zeros(shape=Params.OVShape([1,1]),dtype = float)
		self.Types["h_hh"] = [0,0]
		self.Types["h_hp"] = [0,1]
		self.Types["h_pp"] = [1,1]				

		self["v_hphp"] = numpy.zeros(shape=Params.OVShape([0,1,0,1]),dtype = float)
		self.Types["v_hphp"] = [0,1,0,1]
		if (Params.Correlated): 
			self["v_hhpp"] = numpy.zeros(shape=Params.OVShape([0,0,1,1]),dtype = float)
			self.Types["v_hhpp"] = [0,0,1,1]
			self["v_hhhh"] = numpy.zeros(shape=Params.OVShape([0,0,0,0]),dtype = float)
			self["v_hhhp"] = numpy.zeros(shape=Params.OVShape([0,0,0,1]),dtype = float)	
			self["v_hhpp"] = numpy.zeros(shape=Params.OVShape([0,0,1,1]),dtype = float)
			self["v_hppp"] = numpy.zeros(shape=Params.OVShape([0,1,1,1]),dtype = float)
			self["v_pppp"] = numpy.zeros(shape=Params.OVShape([1,1,1,1]),dtype = float)
			self.Types["v_hhhh"] = [0,0,0,0]
			self.Types["v_hhhp"] = [0,0,0,1]
			self.Types["v_hhpp"] = [0,0,1,1]
			self.Types["v_hppp"] = [0,1,1,1]
			self.Types["v_pppp"] = [1,1,1,1]
		
		for O in self.iterkeys(): 
			self[O] *= 0.0				
			
		for HH in Hootmp: 
			if (FreezingMap[HH[0]] > 10000 or FreezingMap[HH[1]] > 10000): 
				continue
			(self["h_hh"])[FreezingMap[HH[0]]][FreezingMap[HH[1]]] = HH[2]
		for HH in Hvvtmp: 
			(self["h_pp"])[HH[0]][HH[1]] = HH[2]
		hov = open(DirectoryPrefix+"HOV","r")
		for s in hov: 
			i = int(s.split()[0])
			j = int(s.split()[1])
			if (FreezingMap[i] > 10000): 
				continue
			v = float(s.split()[2])
			(self["h_hp"])[FreezingMap[i]][j] = v 
		hov.close()
		del hov
		vovov = open(DirectoryPrefix+"VOVOV","r")
		for s in vovov: 
			i = int(s.split()[0])
			j = int(s.split()[1])
			a = int(s.split()[2])
			b = int(s.split()[3])			
			if (FreezingMap[i] > 10000): 
				continue
			if (FreezingMap[a] > 10000): 
				continue
			v = float(s.split()[4])
			(self["v_hphp"])[FreezingMap[i]][j][FreezingMap[a]][b] = v		
		vovov.close()	
		del vovov
		if (Params.Correlated): 
			voovv = open(DirectoryPrefix+"VOOVV","r")
			for s in voovv: 
				i = int(s.split()[0])
				j = int(s.split()[1])
				a = int(s.split()[2])
				b = int(s.split()[3])
				if (FreezingMap[i] > 10000): 
					continue
				if (FreezingMap[j] > 10000): 
					continue								
				v = float(s.split()[4])
				(self["v_hhpp"])[FreezingMap[i]][FreezingMap[j]][a][b] = v
			voovv.close()		
			del voovv	
			voooo = open(DirectoryPrefix+"VOOOO","r")
			for s in voooo: 
				i = int(s.split()[0])
				j = int(s.split()[1])
				a = int(s.split()[2])
				b = int(s.split()[3])
				if (FreezingMap[i] > 10000): 
					continue
				if (FreezingMap[j] > 10000): 
					continue	
				if (FreezingMap[a] > 10000): 
					continue
				if (FreezingMap[b] > 10000): 
					continue										
				v = float(s.split()[4])
				(self["v_hhhh"])[FreezingMap[i]][FreezingMap[j]][FreezingMap[a]][FreezingMap[b]] = v
			voooo.close()	
			del voooo		
			vooov = open(DirectoryPrefix+"VOOOV","r")
			for s in vooov: 
				i = int(s.split()[0])
				j = int(s.split()[1])
				a = int(s.split()[2])
				b = int(s.split()[3])
				if (FreezingMap[i] > 10000): 
					continue
				if (FreezingMap[j] > 10000): 
					continue	
				if (FreezingMap[a] > 10000): 
					continue									
				v = float(s.split()[4])
				(self["v_hhhp"])[FreezingMap[i]][FreezingMap[j]][FreezingMap[a]][b] = v
			vooov.close()		
			del vooov					
			vovvv = open(DirectoryPrefix+"VOVVV","r")
			for s in vovvv: 
				i = int(s.split()[0])
				j = int(s.split()[1])
				a = int(s.split()[2])
				b = int(s.split()[3])
				if (FreezingMap[i] > 10000): 
					continue
				v = float(s.split()[4])
				(self["v_hppp"])[FreezingMap[i]][j][a][b] = v				
			vovvv.close()
			del vovvv
			vvvvv = open(DirectoryPrefix+"VVVVV","r")		
			for s in vvvvv: 
				i = int(s.split()[0])
				j = int(s.split()[1])
				a = int(s.split()[2])
				b = int(s.split()[3])
				v = float(s.split()[4])
				(self["v_pppp"])[i][j][a][b] = v		
			vvvvv.close()
			del vvvvv
		
		# I'll keep these around, For shits. 
		# self["h"] += self["h_hh"]
		# self["h"] += self["h_hp"]
		# self["h"] += self["h_pp"]
		
		# note the hphp here corresponds to the types on the tensor indices, 
		# not the order of the second quantized string associated with the term. 
		
		self["h_ph"] = 0.0*numpy.zeros(shape=Params.OVShape([1,0]),dtype = float)
		self.Types["h_ph"] = [1,0]
		self["h_ph"] += self["h_hp"].transpose()		
		
		self.Types["v_phhp"] = [1,0,0,1]
		self.Types["v_phph"] = [1,0,1,0]	
		self.Types["v_hpph"] = [0,1,1,0]
		self["v_phhp"] = 0.0*numpy.zeros(shape=Params.OVShape([1,0,0,1]),dtype = float)
		self["v_phph"] = 0.0*numpy.zeros(shape=Params.OVShape([1,0,1,0]),dtype = float)
		self["v_hpph"] = 0.0*numpy.zeros(shape=Params.OVShape([0,1,1,0]),dtype = float)
		self["v_phhp"] += (-1.0)*self["v_hphp"].transpose(1,0,2,3)
		self["v_hpph"] += (-1.0)*self["v_hphp"].transpose(0,1,3,2) 
		self["v_phph"] += self["v_hphp"].transpose(1,0,3,2)
		if Params.Correlated: 
			self["v_pphh"] = 0.0*numpy.zeros(shape=Params.OVShape([1,1,0,0]),dtype = float)
			self.Types["v_pphh"] = [1,1,0,0]
			self["v_pphh"] += self["v_hhpp"].transpose(2,3,0,1) 					
			self["v_ppph"] = 0.0*numpy.zeros(shape=Params.OVShape([1,1,1,0]),dtype = float)
			self["v_hhph"] = 0.0*numpy.zeros(shape=Params.OVShape([0,0,1,0]),dtype = float)
			self["v_pphp"] = 0.0*numpy.zeros(shape=Params.OVShape([1,1,0,1]),dtype = float)
			self["v_phpp"] = 0.0*numpy.zeros(shape=Params.OVShape([1,0,1,1]),dtype = float)
			self["v_hphh"] = 0.0*numpy.zeros(shape=Params.OVShape([0,1,0,0]),dtype = float)
			self["v_phhh"] = 0.0*numpy.zeros(shape=Params.OVShape([1,0,0,0]),dtype = float)
			self.Types["v_ppph"] = [1,1,1,0]
			self.Types["v_hhph"] = [0,0,1,0]
			self.Types["v_pphp"] = [1,1,0,1]
			self.Types["v_phpp"] = [1,0,1,1]
			self.Types["v_hphh"] = [0,1,0,0]
			self.Types["v_phhh"] = [1,0,0,0]
			self["v_hhph"] += (-1.0)*self["v_hhhp"].transpose(0,1,3,2) 
			self["v_hphh"] += self["v_hhhp"].transpose(2,3,0,1) 
			self["v_phhh"] += (-1.0)*self["v_hhhp"].transpose(3,2,0,1) 
			self["v_ppph"] += (-1.0)*self["v_hppp"].transpose(2,3,1,0)
			self["v_pphp"] += self["v_hppp"].transpose(2,3,0,1) 
			self["v_phpp"] += (-1.0)*self["v_hppp"].transpose(1,0,2,3) 

		# this is useful for making interaction picture rotations. 
		self["e_h"] = self["h_hh"].diagonal().copy()
		self["e_p"] = self["h_pp"].diagonal().copy()
		
		print "Hole Energies (au): ", self["e_h"].real
		print "Particle Energies (au): ", self["e_p"].real	
		tmp = self["e_h"].tolist()
		tmp.extend(self["e_p"].tolist())

		if (not Params.Undressed): 
			UniqueTransitions = []
			for i in range(Params.nmo): 
				for j in range(Params.nmo):
					for k in range(Params.nmo): 
						for l in range(Params.nmo):	
							UniqueTransitions.append(round(abs(tmp[i]+tmp[j]-tmp[k]-tmp[l]),8))

			tmp2 = list(set(tuple(UniqueTransitions)))
			tmp2.sort()
			tmp3 = (numpy.array(tmp2)/(4.5563*math.pow(10.0,-6.0))).tolist()	
			print "Zeroth Transitions In Wavenumbers: ", filter(lambda X: X<10000,tmp3)
		
		print "Hole Energies (eV): ", self["e_h"].real * 27.2113
		print "Particle Energies (eV): ", self["e_p"].real * 27.2113
		
		if (False): 
			#Those read off disk. 
			self["v"] += self["v_hhhh"]
			self["v"] += self["v_hhhp"]
			self["v"] += self["v_hhpp"]
			self["v"] += self["v_hphp"]
			self["v"] += self["v_hppp"]
			self["v"] += self["v_pppp"]	
			#Those Transposed. 
			self["h"] += self["h_ph"]
			self["v"] += self["v_pphh"]		
			self["v"] += self["v_phhp"]
			self["v"] += self["v_hpph"]
			self["v"] += self["v_phph"]		
			self["v"] += self["v_hhph"]
			self["v"] += self["v_hphh"]
			self["v"] += self["v_phhh"]		
			self["v"] += self["v_ppph"]
			self["v"] += self["v_pphp"]
			self["v"] += self["v_phpp"]

		# Dipole matrices...
		self["mu_hh"] = numpy.zeros(shape=(Params.nocc, Params.nocc,3),dtype = float)
		self["mu_hp"] = numpy.zeros(shape=(Params.nocc, Params.nvirt,3),dtype = float)
		self["mu_ph"] = numpy.zeros(shape=(Params.nvirt, Params.nocc,3),dtype = float)
		self["mu_pp"] = numpy.zeros(shape=(Params.nvirt, Params.nvirt,3),dtype = float)		

		# These are given only for AlphaSpin (ntot x ntot) 
		# So they need to know the length of the orig. qchem length arrays. 
		noccover2 = (Params.nocc)/2			
		if (Params.FreezeCore != None): 
			noccover2 = (Params.nocc+Params.FreezeCore)/2 
		nvirtover2 = Params.nvirt/2
		for k in range(3): 
			if (k==0): 
				Mu = open(DirectoryPrefix+"Mux","r")
			elif (k==1): 
				Mu = open(DirectoryPrefix+"Muy","r")
			elif (k==2): 
				Mu = open(DirectoryPrefix+"Muz","r")
			for s in Mu: 
				i = int(s.split()[0])
				j = int(s.split()[1])
				v = float(s.split()[2])
				if (i<noccover2):
					if (j<noccover2): 
						if (FreezingMap[i] < 10000 and FreezingMap[j] < 10000): 
							self["mu_hh"][FreezingMap[i]][FreezingMap[j]][k] = v  #alpha alpha
						if (FreezingMap[i+noccover2] < 10000 and FreezingMap[j+noccover2] < 10000): 							
							self["mu_hh"][FreezingMap[i+noccover2]][FreezingMap[j+noccover2]][k] = v # beta beta
					else: 
						#i is occ, j is virt
						if (FreezingMap[i] < 10000):
							self["mu_hp"][FreezingMap[i]][j-noccover2][k] = v # alpha alpha
						if (FreezingMap[i+noccover2] < 10000):
							self["mu_hp"][FreezingMap[i+noccover2]][j-noccover2+nvirtover2][k] = v # beta beta
				else: # i is virt 
					if (j<noccover2): 
						if (FreezingMap[j] < 10000):							
							self["mu_ph"][i-noccover2][FreezingMap[j]][k] = v  #alpha alpha
						if (FreezingMap[j+noccover2] < 10000):							
							self["mu_ph"][i-noccover2+nvirtover2][FreezingMap[j+noccover2]][k] = v # beta beta
					else: 
						#j is virt
						self["mu_pp"][i-noccover2][j-noccover2][k] = v
						self["mu_pp"][i-noccover2+nvirtover2][j-noccover2+nvirtover2][k] = v
			Mu.close()			

		self["mux_hh"] = (self["mu_hh"])[:,:,0]
		self["mux_hp"] = (self["mu_hp"])[:,:,0]
		self["mux_ph"] = (self["mu_ph"])[:,:,0]
		self["mux_pp"] = (self["mu_pp"])[:,:,0]
		self["muy_hh"] = (self["mu_hh"])[:,:,1]
		self["muy_hp"] = (self["mu_hp"])[:,:,1]
		self["muy_ph"] = (self["mu_ph"])[:,:,1]
		self["muy_pp"] = (self["mu_pp"])[:,:,1]
		self["muz_hh"] = (self["mu_hh"])[:,:,2]
		self["muz_hp"] = (self["mu_hp"])[:,:,2]
		self["muz_ph"] = (self["mu_ph"])[:,:,2]
		self["muz_pp"] = (self["mu_pp"])[:,:,2]
		   
#		All Permutational symmetries of the hamiltonian. 
#		ident1 = range(2)
#		perms = [X for X in AllPerms(ident1)]
#		perms1234 = map(lambda X:([X[0][0],X[0][1],2,3],X[1]),copy.copy(perms))
#		perms34 = map(lambda X:([0,1,X[0][0]+2,X[0][1]+2],X[1]),copy.copy(perms))
#		HermitianSymmetry = [([0,1,2,3],1),([2,3,0,1],1)]
#		A = MuliplyPermutationLists(perms1234,perms34)
#		TwoParticlePermutationGroup = MuliplyPermutationLists(A,HermitianSymmetry)

		overallsize = 0.0
		for T in self.iterkeys(): 
			Tdim = 1
			sh = self[T].shape 
			for d in sh : 
				Tdim *= d
			overallsize += Tdim*self[T].itemsize
		print "Overall Integral Storage: ", overallsize/pow(1024,2), " MB"

		# Check to make sure there are Absolutely no NANs. 
		for T in self.iterkeys(): 
			for E in numpy.ndindex((self[T]).shape): 
				if ( isnan(self[T][E])): 
					print "Warning. Read nan in: ",E, " of ", T 
					self[T][E] = 0.0

		import gc
		gc.collect()
		del tmp
		
		return

Integrals = IntegralDictionary()	


