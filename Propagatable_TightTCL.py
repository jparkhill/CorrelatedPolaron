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

# A 4x4 tight binding model. 

class TightBindingTCL: 
	def __init__(self):
		# these things must be defined by the algebra. 
		self.VectorShape = None 
		self.ResidualShape = None
	
		self.H = numpy.zeros(shape=(4,4),dtype = complex)
		self.H[0][0] = 0.0 
		self.H[1][1] = 0.3
		self.H[2][2] = 0.6
		self.H[3][3] = 0.92				
		self.S = numpy.random.rand(4,4)*0.1
		#Make S hermitian#
		self.S += self.S.transpose()
		
		self.Mu = numpy.random.rand(4,4)*0.1
		self.Mu += self.Mu.transpose()

		self.InitAlgebra()
		print "Algebra Initalization Complete... "
		self.V0 = StateVector()
		self.Mu0 = None
		print "CIS Initalization Complete... "
		return 
	
	def Energy(self,AStateVector): 
		return AStateVector["r1_ph"][0][1]
	
	def InitAlgebra(self): 
		print "Algebra... "		
		self.VectorShape = StateVector()
		self.VectorShape["r"] = numpy.zeros(shape = (4,4), dtype = complex)
		return 

	def DipoleMoment(self,AVector, Time=0.0): 
		Mu = numpy.tensordot(AVector["r"],self.Mu,axes=([0,1],[0,1]))
		return (1.0/3.0)*numpy.sum(Mu*Mu)

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
#			if (abs(Ens[s]-11.54/EvPerAu) < .1): 
				self.V0.Add(Sts[s])

		self.V0.MultiplyScalar(1.0/numpy.sqrt(self.V0.InnerProduct(self.V0)))		
				
		SumSq = 0.0
		for s in range(len(Ens)): 
			Wght = Sts[s].InnerProduct(self.V0)
			print "En: ", Ens[s]," (ev) ", Ens[s]*EvPerAu ,  " Weight ", Wght, " dipole: ", self.DipoleMoment(Sts[s])
			print "Contributions: "
			Sts[s].Print()
			SumSq += Wght*numpy.conjugate(Wght)
		# The exact states are not orthogonal. 
		print "SumSq: ", SumSq
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
		print "Trace Rho0: ", r1_ph.sum(), r1_hp.sum(), r1_ph.sum()+r1_hp.sum()		
		self.Mu0 = self.DipoleMoment(self.V0)
		return 
		
	def First(self): 
		return self.V0
		
	# now operate -i[H, ] on rho:
	def Step(self,OldState,Time,Field = None, AdiabaticMultiplier = 1.0): 
		NewState = OldState.clone()
		NewState.Fill()
		self.VectorShape.EvaluateContributions(self.CISTerms,NewState,OldState)
		if (Field != None):
			if (sqrt(Field[0]*Field[0]+Field[1]*Field[1]+Field[2]*Field[2]) > Params.FieldThreshold):
				NewState["r1_ph"] += Integrals["mux_ph"]*Field[0]
				NewState["r1_ph"] += Integrals["muy_ph"]*Field[1]
				NewState["r1_ph"] += Integrals["muz_ph"]*Field[2]			
				self.VectorShape.EvaluateContributions(self.MuTerms,NewState,OldState, FieldInformation = Field)		
		NewState.MultiplyScalar(complex(0,-1.0))
#   NOTE: the overall negative one factor for the pertubative term is included in self.Pterms. 
		self.VectorShape.EvaluateContributions(self.PTerms,NewState,OldState,TimeInformation=Time, MultiplyBy=AdiabaticMultiplier)
		return NewState
