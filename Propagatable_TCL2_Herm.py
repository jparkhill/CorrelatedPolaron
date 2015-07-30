import numpy, scipy
#from scipy import special
from numpy import array
from numpy import linalg
# standard crap
import __builtin__ 
import os, sys, time, math, re, random, cmath
import pickle
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, log, pi, sqrt, isnan
# local modules written by Moi.
from TensorNumerics import * 
from Wick import * 
from LooseAdditions import * 
from NumGrad import * 
from BosonCorrelationFunction import * 

#
# A TDA-like approximation to the response matrix (studying only ph->ph transformations) 
# Also splitting up the terms and building on LCISD 
#

class TCL2_Herm: 
	def __init__(self):
		print "Getting ready for LIOUVILLE SPACE Hermitian Propagation... "
		
		#Options for how the equations are derived. 
		self.SubtractRightClosure = False
		self.QSpaceProjector = True 
		self.HermitianTheory = True
		self.HHandPPTerms = False # Includes hole->hole and particle particle terms. 
		self.TammDancoff = True  # This kills all ph->hp and hp->ph couplings. 
		self.OnlyPH = True # This kills everything except ph->ph transitions 
		self.OnlyHP = False
		self.ForceAntiHermitian = False 
		self.SecondOrderDipole = False
		self.RenormalizeIntegrals = False 
		
		# these things must be defined by the algebra. 
		self.VectorShape = None 
		self.CISTerms = None  
		self.PTerms = None	
		self.DebugTerms1 = None	
		self.DebugTerms2 = None			

		self.InitAlgebra()
		print "Algebra Initalization Complete... "
		self.V0 = StateVector()
		self.Mu0 = None
		self.InitalizeNumerics() # Initalizes BCF. 
		print "TDA-2TCL Initalization Complete ----  "
		
		print "self.RenormalizeIntegrals: ", self.RenormalizeIntegrals 
		print "self.SecondOrderDipole: ", self.SecondOrderDipole 
		print "self.QSpaceProjector: ", self.QSpaceProjector 	
		print "self.ForceAntiHermitian: ", self.ForceAntiHermitian 		
		return 
	
	def Energy(self,AStateVector): 
		return AStateVector["r1_ph"][0][1]
	
	def InitAlgebra(self): 
		print "Algebra... "		
		HRho = H_Fermi.clone()
		MaxRho = 1
		Rho = RhosUpTo(MaxRho,2) #second argument is normalization of rho. 
		FermiRho = Rho.ToFermi()
		FermiRhoC = FermiRho.MyConjugate()
		print "State vector represented by sectors: "
		for Term in FermiRho: 
			Term.Print()

		print "State vector conjugate represented by sectors: "
		for Term in FermiRhoC: 
			Term.Print()

		# if the files for the algebra exist, just read them in. 
		if (os.path.isfile("./Terms/HermCIS") and os.path.isfile("./Terms/HermPterm")):
			print "Found Existing Terms, Unpickling them."
			cf = open("./Terms/HermCIS","rb")
			pf = open("./Terms/HermPterm","rb")	
			UnpickleCIS = pickle.Unpickler(cf)
			UnpicklePT = pickle.Unpickler(pf)			
			self.CISTerms = UnpickleCIS.load()
			cf.close()
			self.PTerms = UnpicklePT.load()
			pf.close()
			print "------Overall Read in: ------- ",len(self.CISTerms),"------CIS-like Terms----------"
			for Term in self.CISTerms: 
				Term.Print()			
			print "------Overall Read in: ------- ",len(self.PTerms),"------Pert Terms----------"
			for Term in self.PTerms: 
				Term.Print()			
		else: 
			#------------------------------		
			#PLP#
			#------------------------------		
			
			rchr = FermiRho.clone()
			rchr.LeftTimes(H_Fermi)
			rchr.LeftTimes(FermiRhoC)
			rchr.NormalOrderDiag([[0,0,0,0,0,0,0]])
			rchr.NoSelfContractPart()
			rchr.ConnectedPart()			
			
			print "r_c hr"
			for term in rchr: 
				term.Print()		
			
			rcrh = H_Fermi.clone()
			rcrh.LeftTimes(FermiRho)
			rcrh.LeftTimes(FermiRhoC)
			rcrh.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
			rcrh.NoSelfContractPart()
			rcrh.ConnectedPart()	
			
			print "rcrh"
			for term in rcrh: 
				term.Print()							
			
			hrrc = FermiRhoC.clone()
			hrrc.LeftTimes(FermiRho)
			hrrc.LeftTimes(H_Fermi)
			hrrc.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
			hrrc.NoSelfContractPart()
			hrrc.ConnectedPart()	
			
			print "hrrc"
			for term in hrrc: 
				term.Print()									

			rhrc = FermiRhoC.clone()
			rhrc.LeftTimes(H_Fermi)
			rhrc.LeftTimes(FermiRho)
			rhrc.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
			rhrc.NoSelfContractPart()
			rhrc.ConnectedPart()	
			
			print "rhrc"
			for term in rhrc: 
				term.Print()				
		
			if (self.SubtractRightClosure):
				self.CISTerms = rchr.clone()
				self.CISTerms.Subtract(hrrc)
				self.CISTerms.Subtract(rcrh)
				self.CISTerms.Add(rhrc)
			else: 
				self.CISTerms = rchr.clone()
				self.CISTerms.Add(hrrc)
				self.CISTerms.Subtract(rcrh)
				self.CISTerms.Subtract(rhrc)

			
			if (False): 
				self.CISTerms = l0.clone()
				if self.SubtractRightClosure: 
					self.CISTerms.Subtract(r0)
				else : 
					self.CISTerms.Add(r0)
				
			OutFile = open("./Terms/HermCIS","w")
			pickle.Pickler(OutFile,0).dump(self.CISTerms)
			OutFile.close()	

			print "-----------------------------"
			print "CIS Terms: ", len(self.CISTerms)
			for Term in self.CISTerms: 
				Term.Print()			
			print "-----------------------------"

			self.PTerms = ManyOpStrings()
			if (Params.Correlated): 
				#------------------------------		
				#[V(t)Q[V(s),\Rho]]#
				#------------------------------
				
				VTt = h2_Fermi.clone()
				VTs = h2_Fermi.clone()
				VTt.GiveTime("t")
				VTs.GiveTime("s")
				
				QVsRho = VTs.clone()
				QVsRho.Times(FermiRho)
				QVsRho.NormalOrderDiagBigMemory()
				QVsRho.NoSelfContractPart()

				if False: 
					print "QVsRho: "
					for Term in QVsRho: 
						Term.Print()
				
				QRhoVs = FermiRho.clone()
				QRhoVs.Times(VTs)
				QRhoVs.NormalOrderDiagBigMemory()
				QRhoVs.NoSelfContractPart()
				
				if False: 
					print "QRhoVs: "
					for Term in QRhoVs: 
						Term.Print()
				
				l1=l2=l3=l4=r1=r2=r3=r4=None 
			
				l1 = FermiRhoC.clone()
				l1.Times(VTt)
				l1.NormalOrderDiagBigMemory(QVsRho.ClassesIContain())
				l1.NoSelfContractPart()
				l1.Times(QVsRho)
				l1.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				l1.NoSelfContractPart()			
				l1.ConnectedPart()
				if (False): 
					print "-----------------------------"
					print "l1 Terms (Vt Vs r): ", len(l1)
					for Term in l1: 
						Term.Print()			
					print "-----------------------------"
				
				r1 = VTt.clone()
				r1.Times(QVsRho)
				r1.NormalOrderDiagBigMemory(FermiRhoC.ClassesIContain())
				r1.NoSelfContractPart()
				r1.Times(FermiRhoC)
				r1.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				r1.NoSelfContractPart()		
				r1.ConnectedPart()
				if (False): 		
					print "-----------------------------"			
					print "r1 Terms (Vt Vs r): ", len(r1)
					for Term in r1: 
						Term.Print()
					print "-----------------------------"			

				l2 = FermiRhoC.clone()
				l2.Times(VTt)
				l2.NormalOrderDiagBigMemory(QRhoVs.ClassesIContain())
				l2.NoSelfContractPart()
				l2.Times(QRhoVs)
				l2.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				l2.NoSelfContractPart()
				l2.ConnectedPart()
				if (False): 
					print "-----------------------------"
					print "l2 Terms (Vt r Vs): ", len(l2)
					for Term in l2: 
						Term.Print()			
					print "-----------------------------"			
					
				r2 = VTt.clone()
				r2.Times(QRhoVs)
				r2.NormalOrderDiagBigMemory(FermiRhoC.ClassesIContain())
				r2.NoSelfContractPart()
				r2.Times(FermiRhoC)
				r2.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				r2.NoSelfContractPart()			
				r2.ConnectedPart()
				if (False): 
					print "-----------------------------"
					print "r2 Terms (Vt r Vs): ", len(r2)
					for Term in r2: 
						Term.Print()			
					print "-----------------------------"
				
				l3 = FermiRhoC.clone()
				l3.Times(QVsRho)
				l3.NormalOrderDiagBigMemory(VTt.ClassesIContain())
				l3.NoSelfContractPart()
				l3.Times(VTt)
				l3.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				l3.NoSelfContractPart()
				l3.ConnectedPart()
				
				if (False): 
					print "-----------------------------"
					print "l3 Terms (Vs r Vt): ", len(l3)
					for Term in l3: 
						Term.Print()			
					print "-----------------------------"			

				r3 = QVsRho.clone()
				r3.Times(VTt)
				r3.NormalOrderDiagBigMemory(FermiRhoC.ClassesIContain())
				r3.NoSelfContractPart()
				r3.Times(FermiRhoC)
				r3.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				r3.NoSelfContractPart()			
				r3.ConnectedPart()
				if (False): 
					print "-----------------------------"
					print "r3 Terms (Vs r Vt): ", len(r3)
					for Term in r3: 
						Term.Print()			
					print "-----------------------------"

				l4 = FermiRhoC.clone()
				l4.Times(QRhoVs)
				l4.NormalOrderDiagBigMemory(VTt.ClassesIContain())
				l4.NoSelfContractPart()
				l4.Times(VTt)
				l4.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				l4.NoSelfContractPart()			
				l4.ConnectedPart()
				if (False): 
					print "-----------------------------"			
					print "l4 Terms (r Vs Vt): ", len(l4)
					for Term in l4: 
						Term.Print()
					print "-----------------------------"
				
				r4 = QRhoVs.clone()
				r4.Times(VTt) 		
				r4.NormalOrderDiagBigMemory(FermiRhoC.ClassesIContain())
				r4.NoSelfContractPart()
				r4.Times(FermiRhoC)
				r4.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				r4.NoSelfContractPart()			
				r4.ConnectedPart()
				if (False): 
					print "-----------------------------"
					print "r4 Terms (r Vs Vt): ", len(r4)
					for Term in r4: 
						Term.Print()			
					print "-----------------------------"		
				
				if self.SubtractRightClosure:
					l1.Subtract(r1)
					l2.Subtract(r2)
					l3.Subtract(r3)
					l4.Subtract(r4)
				else :
					l1.Add(r1)
					l2.Add(r2)
					l3.Add(r3)
					l4.Add(r4)
				del self.PTerms[:]
				self.PTerms.Subtract(l1)
				self.PTerms.Add(l2)
				self.PTerms.Add(l3)
				self.PTerms.Subtract(l4)
				# The Above factors result from:
				# - The Commutators [Vt,[Vs,rho]]
				# - closing on the left*=(1) or right *=(-1)
				# - The (i/hbar)^2 prefactor
				# There is the usual Fermion sign factors which also occur resulting from wick.py. 
				# These are included in the terms as well. 
				print "Saving Derived Terms..."
				OutFile = open("./Terms/HermPterm","w")
				pickle.Pickler(OutFile,0).dump(self.PTerms)
				OutFile.close()
				print "Derived Terms saved in ./Terms/HermPterm"

		self.VectorShape = FermiRho.clone()
		# ------------------------------------------------------------------
		# Okay so in the above if... else either the terms were rederived or 
		# read in from disk. Here they are approximated or chopped down. 	
		if (self.TammDancoff): 
			print "Applying The Tamm-Dancoff Approximation. "
			print "All couplings between difference sectors of the electronic problem will be set to zero."
			self.CISTerms.ApplyTDA()
			self.PTerms.ApplyTDA()				
		if (not self.HHandPPTerms): 
			print "Removing the hh and pp terms."
			self.VectorShape.TermsNotContainingTensor("r1_hh")
			self.VectorShape.TermsNotContainingTensor("r1_pp")				
			self.VectorShape.TermsNotContainingTensor("r1_Vac_hh")				
			self.VectorShape.TermsNotContainingTensor("r1_Vac_pp")
			self.CISTerms.TermsNotContainingTensor("r1_hh")
			self.CISTerms.TermsNotContainingTensor("r1_pp")				
			self.CISTerms.TermsNotContainingTensor("r1_Vac_hh")				
			self.CISTerms.TermsNotContainingTensor("r1_Vac_pp")
			self.PTerms.TermsNotContainingTensor("r1_Vac_hh")				
			self.PTerms.TermsNotContainingTensor("r1_Vac_pp")
			self.PTerms.TermsNotContainingTensor("r1_hh")
			self.PTerms.TermsNotContainingTensor("r1_pp")
			print "State vector represented by sectors: "
			for Term in self.VectorShape: 
				Term.Print()
			print "**State vector represented by sectors: "
		if (self.OnlyPH): 
			print "Only ph terms will be retained."
			self.VectorShape.TermsContainingTensor("r1_ph")
			self.VectorShape.TermsContainingTensor("r1_Vac_ph")
			self.CISTerms.TermsContainingTensor("r1_ph")
			self.PTerms.TermsContainingTensor("r1_ph")
			self.CISTerms.TermsContainingTensor("r1_Vac_ph")
			self.PTerms.TermsContainingTensor("r1_Vac_ph")
			print "State vector represented by sectors: "
			for Term in self.VectorShape: 
				Term.Print()
			print "**State vector represented by sectors: "

		elif (self.OnlyHP): 
			print "Only hp terms will be retained"
			self.VectorShape.TermsContainingTensor("r1_hp")
			self.VectorShape.TermsContainingTensor("r1_Vac_hp")
			self.CISTerms.TermsContainingTensor("r1_hp")
			self.PTerms.TermsContainingTensor("r1_hp")
			self.CISTerms.TermsContainingTensor("r1_Vac_hp")
			self.PTerms.TermsContainingTensor("r1_Vac_hp")			
			print "State vector represented by sectors: "
			for Term in self.VectorShape: 
				Term.Print()
			print "**State vector represented by sectors: "

		print "State vector represented by sectors: "
		for Term in self.VectorShape: 
			Term.Print()
		print "**State vector represented by sectors: "
		
		self.PTerms.AssignBCFandTE()
		if (self.QSpaceProjector): 
			print "Applying Qspace projector to the result of the inner-commutator."
			self.PTerms.QPart()
		if (self.ForceAntiHermitian): 
			print "Adding time-reversed terms to force antihermiticity."
			Tmp = self.PTerms.AddAntiHermitianCounterpart()

		print "Using Non-Perturbative terms: "
		for Term in self.CISTerms:
			Term.Print()			
		
		print "Using Perturbative terms: "
		phph = self.PTerms.clone()
		phph.TermsContainingTensor("r1_ph")
		phph.TermsContainingTensor("r1_Vac_ph")
		print "------------- phph",len(phph),"----------------"
		for Term in phph:
			Term.Print()			
		print "-----------------------------"			
		#
		hphp = self.PTerms.clone()
		hphp.TermsContainingTensor("r1_hp")
		hphp.TermsContainingTensor("r1_Vac_hp")		
		print "------------- hphp",len(hphp),"----------------"
		for Term in hphp: 
			Term.Print()			
		print "-----------------------------"

		phhp = self.PTerms.clone()
		phhp.TermsContainingTensor("r1_hp")
		phhp.TermsContainingTensor("r1_Vac_ph")
		print "------------- phhp",len(phhp),"----------------"
		for Term in phhp: 
			Term.Print()			
		print "-----------------------------"
		#
		hpph = self.PTerms.clone()
		hpph.TermsContainingTensor("r1_ph")
		hpph.TermsContainingTensor("r1_Vac_hp")
		print "------------- hpph",len(hpph),"----------------"
		for Term in hpph: 
			Term.Print()			
		print "-----------------------------"

		#
		hhhh = self.PTerms.clone()
		hhhh.TermsContainingTensor("r1_hh")
		hhhh.TermsContainingTensor("r1_Vac_hh")
		print "------------- hh->hh: ",len(hhhh),"----------------"
		for Term in hhhh: 
			Term.Print()			
		print "-----------------------------"
		#
		pppp = self.PTerms.clone()
		pppp.TermsContainingTensor("r1_pp")
		pppp.TermsContainingTensor("r1_Vac_pp")
		print "------------- pp->pp: ",len(pppp),"----------------"
		for Term in pppp: 
			Term.Print()			
		print "-----------------------------"			

		# These are the critical things assumed by the numerical part. 
		return 

	def DipoleMoment(self,AVector, Time=0.0): 
		Mu = numpy.tensordot(AVector["r1_ph"],Integrals["mu_ph"],axes=([0,1],[0,1]))
#		Mu += numpy.tensordot(AVector["r1_hp"],Integrals["mu_hp"],axes=([0,1],[0,1]))
#		if "r1_hh" in AVector: 
#			Mu += numpy.tensordot(AVector["r1_hh"],Integrals["mu_hh"],axes=([0,1],[0,1]))		
#		if "r1_pp" in AVector: 
#			Mu += numpy.tensordot(AVector["r1_pp"],Integrals["mu_pp"],axes=([0,1],[0,1]))
		if (self.Mu0 == None): 
			return (1.0/3.0)*numpy.sum(Mu*Mu)
		else:
			return (1.0/3.0)*numpy.sum(Mu*Mu) - self.Mu0 

	def AlphaMinusBeta(self,rho1): 
		al = 0.0
		be = 0.0
		for I in Params.occ: 
			if (I in Params.alpha): 
				al += rho1["r1_hh"][I][I]
			else : 
				be += rho1["r1_hh"][I][I]
		for I in Params.virt: 
			if (I in Params.beta): 
				al += rho1["r1_pp"][I][I]
			else : 
				be += rho1["r1_pp"][I][I]			
		return al-be


# Explictly Diagonalize the Liouvillian. 
	def ExactStates(self): 
		print "explicitly diagonalizing the singles liouvillian ... "
		# Explicitly construct and diagonalize matrix. 
		InVector = StateVector()
		OutVector = StateVector()
		self.VectorShape.MakeTensorsWithin(InVector)
		self.VectorShape.MakeTensorsWithin(OutVector)		
		IndexVector = InVector.AllElementKeys()
		dim = len(IndexVector)
		ToDiag = numpy.zeros((dim,dim),dtype = complex)
		for I in range(dim): 
			InVector.Fill(0.0)
			OutVector.Fill(0.0)			
			(InVector[((IndexVector[I])[0])])[(IndexVector[I])[1]] = 1.0
			self.VectorShape.EvaluateContributions(self.CISTerms,OutVector,InVector)
			for J in range(dim): 
				ToDiag[I][J] = (OutVector[((IndexVector[J])[0])])[(IndexVector[J])[1]]
#		print "Indices: ", IndexVector
#		print "Matrix: ", ToDiag
		ToDiag = ToDiag + ToDiag.transpose()
		ToDiag *= 1.0/2.0 
		w,v = numpy.linalg.eig(ToDiag) 
		print "Energies Au: ", w.real
		print "Energies eV: ", sorted(w.real*EvPerAu)
		print "Significant Energies eV: ", sorted(list(set(tuple([round(E,4) for E in sorted(w.real*EvPerAu) if E > 0.5]))))
		states = [StateVector() for X in range(dim)]
		for I in range(dim): 
			self.VectorShape.MakeTensorsWithin(states[I])
			states[I].Fill(0.0)
			for J in range(dim):
				((states[I])[((IndexVector[J])[0])])[(IndexVector[J])[1]] = v[J,I]				
		return (w.real, states)

	def InitalizeNumerics(self): 
		# ------------------------ 
		# Boson section. 
		# ------------------------ 
		print "Initializing Boson Information... "		
		Bs = BosonInformation(Params)
		Bs.DefaultBosons()
		if (os.path.isfile("./Terms/Bosons")):
			print "Found Existing Bosons, Unpickling them."
			bf = open("./Terms/Bosons","rb")
			UnpickleBos = pickle.Unpickler(bf)
			Bs = UnpickleBos.load()
			bf.close()
			return  
		print "---------------------------------------------"		
		print "Using Boson Correlation Parameters: "
		Bs.Print()
#		Here we should modify the integrals, but we aren't doing that yet. 
#		Bs.PlotCorrelationFunction()
		print "Initializing Boson Correlation Functions/Tensors... "
		self.PTerms.MakeBCFs(Bs)

		# BATH RENORMALIZATION	
		if (self.RenormalizeIntegrals): 
			print "about to... RENORMALIZE INTEGRALS!!!!... "
			print "Mean field energy before... "
			Integrals.MakeHFEnergy()
						
			for Tensor in Integrals.iterkeys(): 
				if (not Tensor in Integrals.Types): 
					continue
				if ((len(Integrals[Tensor].shape))==2): 
					typs = Integrals.Types[Tensor]
					if (typs[0] == typs[1]): 
						Ten = Integrals[Tensor]
						Shifts = [0 if X==0 else Params.nocc for X in typs]
						it = numpy.ndindex(Ten.shape)
						for iter in it: 
							if (iter[0]==iter[1]):
								for b in rlen(Bs.Freqs): 
									Ten[iter] -= Bs.Freqs[b]*pow(Bs.MTilde[b][iter[0]+Shifts[0]],2.0)
				elif ((len(Integrals[Tensor].shape))==4):
					typs = Integrals.Types[Tensor]
					if (typs[0] == typs[2] and typs[1] == typs[3]): 
						Ten = Integrals[Tensor]
						Shifts = [0 if X==0 else Params.nocc for X in typs]
						it = numpy.ndindex(Ten.shape)
						for iter in it: 
							if (iter[0]==iter[2] and iter[1]==iter[3] and iter[0] != iter[1]):
								for b in rlen(Bs.Freqs): 
									Ten[iter] -= 2*(Bs.Freqs[b]*Bs.MTilde[b][iter[0]+Shifts[0]])*(Bs.MTilde[b][iter[1]+Shifts[1]])
			print "Mean field energy after... "
			Integrals.MakeHFEnergy()
			print "Checking Spin Symmetry... "
			if (not Integrals.CheckSpinSymmetry()): 
				raise Exception("BrokeSpin...")
			
	# prepare some single excitation as the initial guess. 
	# homo->lumo (remembering alpha and beta.)
	
		if (not Params.DipoleGuess): 
			print "WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
			print "Only Dipole Guess is implemented in this case..."
			print "WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

		# This is the Dipole guess. 
		print "Warning... Missing second-order dipole correction."
		self.VectorShape.MakeTensorsWithin(self.V0)
		self.V0.Fill(0.0)
		r1_ph = self.V0["r1_ph"] 
		r1_ph += Integrals["mu_ph"][:,:,0]
		r1_ph += Integrals["mu_ph"][:,:,1]
		r1_ph += Integrals["mu_ph"][:,:,2]
		if "r1_hp" in self.V0:
			self.V0["r1_hp"] += r1_ph.transpose()
		if "r1_hh" in self.V0: 
			self.V0["r1_hh"] += Integrals["mu_hh"][:,:,0]
			self.V0["r1_hh"] += Integrals["mu_hh"][:,:,1]
			self.V0["r1_hh"] += Integrals["mu_hh"][:,:,2]
		if "r1_pp" in self.V0: 
			self.V0["r1_pp"] += Integrals["mu_pp"][:,:,0]
			self.V0["r1_pp"] += Integrals["mu_pp"][:,:,1]
			self.V0["r1_pp"] += Integrals["mu_pp"][:,:,2]
		self.V0.MultiplyScalar(1.0/numpy.sqrt(self.V0.InnerProduct(self.V0)))		
	
		if (False): 
			self.VectorShape.MakeTensorsWithin(self.V0)
			Ens,Sts = self.ExactStates()
			# simply make an even mix bright states. 
			self.V0.Fill(0.0)
			for s in range(len(Ens)): 
				if (self.DipoleMoment(Sts[s]) > 0.001):
					self.V0.Add(Sts[s])
			if (not self.OnlyPH and not self.OnlyHP): 
				self.V0["r1_hp"] += numpy.transpose(self.V0["r1_ph"].conj())
				self.V0["r1_ph"] = numpy.transpose(self.V0["r1_hp"].conj())			
			self.V0.MultiplyScalar(1.0/numpy.sqrt(self.V0.InnerProduct(self.V0)))		
			SumSq = 0.0
			for s in range(len(Ens)): 
				Wght = Sts[s].InnerProduct(self.V0)
				print "En: ", Ens[s]," (ev) ", Ens[s]*EvPerAu ,  " Weight ", Wght, " dipole: ", self.DipoleMoment(Sts[s])
				SumSq += Wght*numpy.conjugate(Wght)
			# The exact states are not orthogonal. 
			print "SumSq: ", SumSq
			# Debug just two of the homogeneous terms together.  
		if (False): 
			print "Debugging phph terms.. "
			Input = self.V0.clone()
			print "input: "
			Input.Print()
			Output = self.V0.clone()
			Output.Fill()
			phph = self.PTerms.clone()
			phph.TermsContainingTensor("r1_ph")
			phph.TermsContainingTensor("r1_Vac_ph")
			Matrix1 = phph[0].EvaluateMatrixForm(Input,0.025)
			Matrix1 *= 0.0
			for Term in phph: 
			
				Tmp = Term.EvaluateMatrixForm(Input,0.025)
				print "Matrix Form "
				K = numpy.ndindex(Tmp.shape)
				for D in K: 
					if (abs(Matrix1[D]) > pow(10.0,-8.0)): 
						print D," : ", Tmp[D], " , ", Tmp[(D[2],D[3],D[0],D[1])].conj() ," : ", Tmp[D]+Tmp[(D[2],D[3],D[0],D[1])].conj()

				Tmp2 = numpy.tensordot(Tmp,Input["r1_ph"],axes=((2,3),(0,1)))
				Tmp3 = Term.EvaluateMyselfAdiabaticallyTest(Input,0.025)	
				Tmp4 = Term.EvaluateMyselfAdiabaticallyAntiHerm(Input,0.025)
				Tmp5 = Term.EvaluateMyselfAdiabatically(Input,0.025)				
				print "MatrixForm ", Tmp2.mean() , " Test: ", Tmp3.mean(), " AntiHerm: ", Tmp4.mean()
				print "Adiabatically: ", Tmp5.mean()
				print "TEST:",Tmp3
				print "ADIA:",Tmp5
				print "AntiHerm:",Tmp4						
			if (False): 
				Matrix1 += Term.EvaluateMatrixForm(Input,0.025)
				print "Matrix 1 "
				K = numpy.ndindex(Matrix1.shape)
				for D in K: 
					if (abs(Matrix1[D]) > pow(10.0,-8.0)): 
						print D," : ", Matrix1[D], " , ", Matrix1[(D[2],D[3],D[0],D[1])].conj() ," : ", Matrix1[D]+Matrix1[(D[2],D[3],D[0],D[1])].conj()
		if False: 
			self.DebugTerms1[0].Print()
			self.VectorShape.EvaluateContributions(self.DebugTerms1,Output,Input,TimeInformation=0.05)
			Matrix1 = self.DebugTerms1[0].EvaluateMatrixForm(Input,0.025)
			print "Matrix 1 "
			K = numpy.ndindex(Matrix1.shape)
			for D in K: 
				if (abs(Matrix1[D]) > pow(10.0,-8.0)): 
					print D," : ", Matrix1[D], " , ", Matrix1[(D[2],D[3],D[0],D[1])]

			self.DebugTerms2[0].Print()	
			Matrix2 = self.DebugTerms2[0].EvaluateMatrixForm(Input,0.025)
			print "Matrix 2 "
			K = numpy.ndindex(Matrix2.shape)
			for D in K: 
				if (abs(Matrix2[D]) > pow(10.0,-8.0)): 
					print D," : ", Matrix2[D], " , ", Matrix2[(D[2],D[3],D[0],D[1])]

			self.VectorShape.EvaluateContributions(self.DebugTerms2,Output,Input,TimeInformation=0.05)
			print "Sum of Matrix Forms: "
			Matrix1 += numpy.transpose(Matrix2,axes=(1,0,3,2))
			print Matrix1.mean()
			K = numpy.ndindex(Matrix1.shape)
			for D in K: 
				if (abs(Matrix1[D]) > pow(10.0,-8.0)): 
					print D," : ", Matrix1[D], " , ", Matrix1[(D[2],D[3],D[0],D[1])]
			
			Output["r1_ph"] += numpy.transpose(Output["r1_hp"],axes=(1,0))
			Output["r1_hp"] *= 0.0
			print "Result of adding both terms: ", Output["r1_hp"].mean(), Output["r1_ph"].mean()
			Output.Print()
		self.Mu0 = self.DipoleMoment(self.V0)						
		return 

	def DebugTermWise(self,OldState,Time): 
		Tmp = OldState.clone()
		Tmp2 = OldState.clone()		
		Tmp3 = OldState.clone()		
		Tmp.Fill()
		Tmp2.Fill()
		Tmp3.Fill()		
		print " -=-=-=- Start =-=-=-=- "
		for Term in self.PTerms:
			print "Evaluating...." 
			print "Input hp and ph ", OldState["r1_hp"].mean() , " , " , OldState["r1_ph"].mean()
			Term.Print()
			Tmp2.Fill()
			Tmp3.Fill()			
			Tmp2.Add(Tmp)
			Tmp4 = ManyOpStrings()
			Tmp4.Add([Term])
			self.VectorShape.EvaluateContributions(Tmp4,Tmp3,OldState,TimeInformation=Time)
			print "Output hp and ph ", Tmp3["r1_hp"].mean() , " , " , Tmp3["r1_ph"].mean()
			print "Output hp and ph ", (Tmp3["r1_hp"]*Tmp3["r1_hp"]).sum() , " , " , (Tmp3["r1_ph"]*Tmp3["r1_ph"]).sum()

		print " -=-=-=-=-=-=-=- "
		for Term in self.DebugTerms:
			print "Evaluating...." 
			print "Input hp and ph ", OldState["r1_hp"].mean() , " , " , OldState["r1_ph"].mean()
			Term.Print()
			Tmp2.Fill()
			Tmp3.Fill()			
			Tmp2.Add(Tmp)
			Tmp4 = ManyOpStrings()
			Tmp4.Add([Term])
			self.VectorShape.EvaluateContributions(Tmp4,Tmp3,OldState,TimeInformation=Time)
			print "Output hp and ph ", Tmp3["r1_hp"].mean() , " , " , Tmp3["r1_ph"].mean()
			print "Output hp and ph ", (Tmp3["r1_hp"]*Tmp3["r1_hp"]).sum() , " , " , (Tmp3["r1_ph"]*Tmp3["r1_ph"]).sum()
		print " -=-=-=- End =-=-=-=- "
		return 

				
	def Step(self,OldState,Time,Field = None, AdiabaticMultiplier = 1.0): 
		NewState = OldState.clone()
		NewState.Fill()
		NewStateCIS = OldState.clone()
		NewStateCIS.Fill()
		NewStatePT = OldState.clone()
		NewStatePT.Fill()				

		self.VectorShape.EvaluateContributions(self.CISTerms,NewStateCIS,OldState)
		NewStateCIS.MultiplyScalar(complex(0,-1.0))
		
		if (Params.Correlated): 
			self.VectorShape.EvaluateContributions(self.PTerms,NewStatePT,OldState,TimeInformation=Time, MultiplyBy=AdiabaticMultiplier)
		
		NewState.Add(NewStateCIS)
		NewState.Add(NewStatePT)
		return NewState

