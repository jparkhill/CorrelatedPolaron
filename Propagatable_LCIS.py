import numpy, scipy
#from scipy import special
from numpy import array
from numpy import linalg
# standard crap
import __builtin__ 
import os, sys, time, math, re, random, cmath
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
	   
# This one is superoperator, or Liouville space CIS
#
# This class provides an interface to a quantum state which can be propagated. 
# self.V0 is a StateVector object (Python dict of numpy.ndarrays) which is created and 
# named according to the algebra of this model. 
#
# And an Step(OldState) method which basically applies H*v
#

class Liouville_CIS: 
	def __init__(self):
		print "Getting ready for LIOUVILLE SPACE CIS Propagation... "
		# these things must be defined by the algebra. 
		self.VectorShape = None 
		self.ResidualShape = None
#		self.DipoleShape = None
		self.InitAlgebra()
		print "Algebra Initalization Complete... "
		self.V0 = StateVector()
		self.Mu0 = None
		self.InitalizeNumerics()
		print "CIS Initalization Complete... "
		return 
	
	def Energy(self,AStateVector): 
		return AStateVector["r1_ph"][0][1]
	
	def InitAlgebra(self): 
		print "Algebra... "		
		HRho = H_Fermi.clone()
		MaxRho = 1
		Rho = RhosUpTo(MaxRho,2) #second argument is normalization of rho. 
		FermiRho = Rho.ToFermi()
		# Temporarily only do ph->ph 
		FermiRho.QPCreatePart()
		RhoH = FermiRho.clone()
		RhoH.NormalOrder(0)
		RhoH.NormalPart(0)
		RhoH.UnContractedPart()			
		HRho.NormalOrder(0)
		HRho.NormalPart(0)
		HRho.UnContractedPart()			
		LeftVac = RhoH.MyConjugate()
		# Since this is a DM expression we also have to project on the Right 
		# and add the two resulting expressions. 
		# (add or subtract.... !!!!!! - JAP 2011) 
		RightVac = LeftVac.clone()
		RhoTemp = RhoH.clone()
		RhoH.Times(HRho)
		HRho.Times(RhoTemp)
		HRho.Subtract(RhoH)
		print " NormalOrdering HRho - RhoH ... "
		HRho.NormalOrder(0,LeftVac.ClassesIContain()) # only those which close against leftvac are kept during this process. 
		LeftVac.Times(HRho)
		HRho.Times(RightVac) 
		print "NormalOrdering LeftVac*(HRho-RhoH) ... "
		LeftVac.NormalOrder(0,[[0,0,0,0,0,0,0]], False)		
		HRho.NormalOrder(0,[[0,0,0,0,0,0,0]], False)		
		LeftVac.Add(HRho) # Note: Adding instead of subtracting the right vaccum part fucked everything up. 
		LeftVac.FullyContractedPart()
		print "*_*_*_*_*_*_*_*__*_*_*_*_*_*_*_*_*"
		print "Using this expression for LCIS: "
		for Term in LeftVac: 
			Term.Print()
		print "*_*_*_*_*_*_*_*__*_*_*_*_*_*_*_*_*"

		
		# These are the critical things assumed by the numerical part. 
		self.VectorShape = FermiRho.clone()
		self.ResidualShape = LeftVac.clone()
		return 

	def DipoleMoment(self,AVector, Time=0.0): 
		Mu = numpy.tensordot(AVector["r1_ph"],Integrals["mu_ph"],axes=([0,1],[0,1]))
		return (1.0/3.0)*numpy.sum(Mu*Mu)

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
			self.VectorShape.EvaluateContributions(self.ResidualShape,OutVector,InVector)
			for J in range(dim): 
				ToDiag[I][J] = (OutVector[((IndexVector[J])[0])])[(IndexVector[J])[1]]
		print "Indices: ", IndexVector
		print "Matrix: ", ToDiag
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

# Explictly Diagonalize the Liouvillian. 
	def ExplicitlyDiagonalize(self): 
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
			self.VectorShape.EvaluateContributions(self.ResidualShape,OutVector,InVector)
			for J in range(dim): 
				ToDiag[I][J] = (OutVector[((IndexVector[J])[0])])[(IndexVector[J])[1]]
				
		# Create a summarized version of the matrix. 
		IndexTypes = list(set(tuple(map(lambda X: X[0],IndexVector))))
		print "index types: ", IndexTypes
		TypedMatrix = numpy.zeros((len(IndexTypes),len(IndexTypes)),dtype = complex)
		# make a typed summary of the matrix. 
		for I in range(len(IndexVector)): 
			for J in range(len(IndexVector)): 		
				SI = IndexTypes.index(IndexVector[I][0])
				SJ = IndexTypes.index(IndexVector[J][0])
				TypedMatrix[SI][SJ] += ToDiag[I][J]
		print "Typed Matrix: ", TypedMatrix
		
		print "Indices: ", IndexVector
		print "Matrix: ", ToDiag
		ToDiag = ToDiag + ToDiag.transpose()
		ToDiag *= 1.0/2.0 
		w,v = numpy.linalg.eig(ToDiag) 
		print "Energies Au: ", w.real
		print "Energies eV: ", sorted(w.real*EvPerAu)
#		print "Zero Eigenvects:" 
#		for i in range(len(w)):
#			if abs((w.real)[i]) < pow (10.0,-5.0): 
#				print " ----- ", w[i] , v[:i]
#				print " ***** "
		print "Significant Energies eV: ", sorted(list(set(tuple([round(E,4) for E in sorted(w.real*EvPerAu) if E > 0.5]))))
		
		return 

	def InitalizeNumerics(self): 
	# prepare some single excitation as the initial guess. 
	# homo->lumo (remembering alpha and beta.)
		self.VectorShape.MakeTensorsWithin(self.V0)
		r1_ph = self.V0["r1_ph"]
		# r1_hp = self.V0["r1_hp"]
		# Try just propagating ph->ph 
		r1_hp = numpy.zeros(shape=(Params.nocc,Params.nvirt) , dtype=complex)
		if (False): 
			print "Checking Spin Symmetry ... "
			self.V0.CheckSpinSymmetry() # I don't think this was working (at all) Jan 20-2012.)
			print "Projecting onto exact eigenstates... "
			Ens,Sts = self.ExactStates()
			# UGH! The degenerate states are not orthogonal. 
			#SymmetricOrthogonalize(Sts)
			print [ [(Sts[k]).InnerProduct(Sts[j]).real.round(5) for j in range(len(Sts)) ] for k in range(len(Sts))]
			SumSq = 0.0
			for s in range(len(Ens)): 
				Wght = Sts[s].InnerProduct(self.V0)
				print "En: ", Ens[s]," (ev) ", Ens[s]*EvPerAu ,  " Weight ", Wght, " dipole: ", self.DipoleMoment(Sts[s])
				SumSq += Wght*numpy.conjugate(Wght)
			print "SumSq: ", SumSq


		Ens,Sts = self.ExactStates()
		# simply make an even mix bright states. 

		self.V0.Fill(0.0)
		for s in range(len(Ens)): 
			if (self.DipoleMoment(Sts[s]) > 0.001):
				self.V0.Add(Sts[s])
		self.V0.MultiplyScalar(0.4)
		
		SumSq = 0.0
		for s in range(len(Ens)): 
			Wght = Sts[s].InnerProduct(self.V0)
			print "En: ", Ens[s]," (ev) ", Ens[s]*EvPerAu ,  " Weight ", Wght, " dipole: ", self.DipoleMoment(Sts[s])
			SumSq += Wght*numpy.conjugate(Wght)
		# The exact states are not orthogonal. 
		print "SumSq: ", SumSq
		
		print "Trace Rho0: ", r1_ph.sum(), r1_hp.sum(), r1_ph.sum()+r1_hp.sum()		
		self.Mu0 = self.DipoleMoment(self.V0)
		return 
		
	def First(self): 
		return self.V0
		
	# now operate -i[H, ] on rho:
	def Step(self,OldState): 
		NewState = OldState.clone()
		NewState.Fill()
		self.VectorShape.EvaluateContributions(self.ResidualShape,NewState,OldState)
		NewState.MultiplyScalar(complex(0,-1.0))
		return NewState
