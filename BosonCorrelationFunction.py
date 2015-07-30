#
# Evaluates Bosonic correlation functions for a polaron transformed CIS model
#

# Make plot styles visible. 
import numpy, scipy
#from scipy import special
from numpy import array
from numpy import linalg
import scipy.weave as weave # To the cloud!
# standard crap
import os, sys, time, math, re, random
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, log, pi, sqrt, isnan
import copy
from SpectralAnalysis import * 
from LooseAdditions import *  # Basic list operations and whatnot. 
from NumGrad import *    # Has useful Physical constants. 

numpy.seterr(all = 'raise')
imi = complex(0.0,1.0)
def Coth(X): 
	if (abs(X) < pow(10,-22.0)):
		raise Exception("Coth Overflow... ")
	elif (X>10.0): 
		return 1.0
	elif (X < -10.0): 
		return -1.0
	else: 
		return (numpy.exp(-1.0*X)+numpy.exp(X))/(numpy.exp(X)-numpy.exp(-1.0*X))	

# for the purposes of this routine 
# Alpha = Beta and the orbitals are numbered (alpha occ)...(alpha virt)
class BosonInformation: 
	def __init__(self, AParams): 
		self.NBos = []
		self.NSts = AParams.nocc+AParams.nvirt # Number of electronic states 
		self.nocc = AParams.nocc
		self.Freqs = [] # Boson Frequencies. 
		self.M = [] # Bath-modes X system-modes
		self.MTilde = [] # Bath-modes X system-modes
		self.Temp = AParams.Temperature   
		self.beta = (1.0/(Kb*self.Temp))
		self.TMax = AParams.TMax
		self.TStep = AParams.TStep
		self.ohmicParam = 1
		self.wc = 20 #frequency cutoff
		self.GeneralCouplings = dict()
		self.a = 0.0 # I guess this is the alpha of J(w) ohmic. 

		self.ContBath = AParams.ContBath
		# Three indicating J(w) ~ w3/wc2 exp(-w/wc)
		self.OmegaCThree = None # Number of Bath modes (1)
		self.OhmicEqCF = None 

		# This just speeds stuff up. 
		self.Nss = None
		self.Coths = None # Like Ns, but instead. Coth(\beta \omega_s/2)
		self.Css = None 
		self.BroadCss = None # like Css, except integrated from a lorentzian spectral density. 
		self.DefaultBosons()
		self.SuperOhmics = []
		return

	def Ns(self,s): 
		if (self.Freqs[s]*self.beta < 10.0): 
			return 1.0/(numpy.exp(self.Freqs[s]*self.beta)-1.0)		
		else :
			return (1.0/(self.Freqs[s]*self.beta))

	def SetTime(self,ta): 
		self.Css = self.CAt(ta)
		return 
		
	def Cs(self,s,T):
		return (self.Coths[s]*(numpy.cos(self.Freqs[s]*T)) - imi*numpy.sin(self.Freqs[s]*T))
		
	def CAt(self,T):
		if (self.ContBath): 
			tore = numpy.zeros(shape=self.NSts, dtype=complex)
			alpha = 1.0#self.EtaCThree[S]
			wc = self.OmegaCThree
			X=self.beta*wc
			Y=1.0/(wc*wc)
			tt = T*T
			# These power series are from Suggi's lineshape paper. 
			realpart = (1./6.)*Y*alpha*(((-tt+Y)/(pow(tt+Y,2.)))+((2.*(-tt+Y*pow(1.+X,2.)))/pow(tt+Y*pow(1.+X,2.),2.))+((2.*(-tt+Y*pow(1.+2.*X,2.)))/pow(tt+Y*pow(1.+2.*X,2.),2.))+((4.*(2.+5.*X)*wc)/(self.beta*(4.+wc*(5.*(4.+5.*X)*self.beta+4.*tt*wc)))))
			tore += complex(realpart,-T*alpha*wc/(3.*pow(1.+tt*wc*wc,2.0)))
			return tore
		else: 
			return numpy.array([self.Cs(s,T) for s in rlen(self.Freqs)])		

	def CorrectReorganization(self):
		for alpha in range(self.NBos): 
			for i in range(Params.nocc):
				Integrals["h_hh"] -= self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][alpha]**2*self.Freqs[alpha]
				Integrals["h_pp"] -= self.GeneralCouplings["B_pp"][Params.Homo()][Params.Homo()][alpha]**2*self.Freqs[alpha]
			for i in range(Params.nvirt):
				Integrals["h_hh"] -= self.GeneralCouplings["B_hh"][Params.Lumo()][Params.Lumo()][0]**2*self.Freqs[alpha]
				Integrals["h_pp"] -= self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][0]**2*self.Freqs[alpha]
		return
		
	# provides a default set of vibronic information. 
	def DefaultBosons(self): 	
		if (Params.Undressed):
			print "Default Bosons called" 
			# Bath coupling matrices...
			if(self.ContBath):
				self.NBos = 1 
				self.OmegaCThree = 5580*AuPerWavenumber
				tmp = self.beta*self.OmegaCThree
				self.a = 0.001
				self.OhmicEqCF = (1./6.)*(1.0+2.0/pow(1.+tmp,2.0)+2.0/pow(1.+2.*tmp,2.0)+4./(2.*tmp+5.*tmp*tmp))

			if(Params.FiniteDifferenceBosons):
				try: 
					self.Freqs = numpy.loadtxt('./Integrals'+Params.MoleculeName+'/VibFreqs',delimiter = ',')				
				except Exception as Ex: 
					print "Exception: ", Ex
					print "No Boson information found at: ", './Integrals'+Params.MoleculeName+'/VibFreqs'
					Params.FiniteDifferenceBosons = False
			if(Params.FiniteDifferenceBosons):
				print "Reading in numerically calculated orbital gradient for a boson model... "
				self.Freqs = numpy.loadtxt('./Integrals'+Params.MoleculeName+'/VibFreqs',delimiter = ',')				
				DimDisps = numpy.loadtxt('./Integrals'+Params.MoleculeName+'/DimDisps',delimiter = ' ')
				print "Shape of Dimensionless Displacements: ", DimDisps.shape
				print "Nocc: ", Params.nocc, " NVirt: ", Params.nvirt, "Params.FreezeCore: ", Params.FreezeCore
				self.NBos = len(self.Freqs)
				self.GeneralCouplings["B_hh"] = numpy.zeros(shape=(Params.nocc, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_hp"] = numpy.zeros(shape=(Params.nocc, Params.nvirt,self.NBos),dtype = float)
				self.GeneralCouplings["B_ph"] = numpy.zeros(shape=(Params.nvirt, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_pp"] = numpy.zeros(shape=(Params.nvirt, Params.nvirt,self.NBos),dtype = float)					
				frznover2 = Params.FreezeCore/2
				nalpha = (Params.nocc + Params.nvirt)/2 + frznover2
				
				if (DimDisps.shape[1] != nalpha): 
					print "Error.. DimDisp dimension mismatch: ", nalpha, DimDisps.shape
					return 
				
				nqalphao = (Params.nocc+Params.FreezeCore)/2
				nalphao = Params.nocc/2
				nalphav = Params.nvirt/2			
				scaledown = 100.0	
				for i in rlen(self.Freqs): 
					for j in range(nalpha):
						if (j < nqalphao): # Orbital is occupied. 
							if (j+1 > frznover2): # Frozen orbitals Aren't included. 
								if (abs(DimDisps[i][j]) > 0.001): 
									self.GeneralCouplings["B_hh"][j-frznover2][j-frznover2][i] = abs(DimDisps[i][j])/scaledown
									self.GeneralCouplings["B_hh"][j-frznover2+nalphao][j-frznover2+nalphao][i] = abs(DimDisps[i][j])/scaledown
						else : # Orbital is virtual 
							if (abs(DimDisps[i][j]) > 0.001): 
								self.GeneralCouplings["B_pp"][j-nqalphao][j-nqalphao][i] = abs(DimDisps[i][j])/scaledown
								self.GeneralCouplings["B_pp"][j-nqalphao+nalphav][j-nqalphao+nalphav][i] = abs(DimDisps[i][j])/scaledown
				print "Using the following boson frequencies: ", self.Freqs

				#Kill bosons of low frequencies. 
				for i in rlen(self.Freqs): 
					if (self.Freqs[i]<0.008): 
						self.GeneralCouplings["B_hh"][:,:,i] = 0
						self.GeneralCouplings["B_pp"][:,:,i] = 0
						self.Freqs[i] = 1.0
												
				for i in rlen(self.Freqs): 
					print "Coupling (Hole-Hole)", self.GeneralCouplings["B_hh"][:,:,i].diagonal()
					print "Coupling (Particle-Particle)", self.GeneralCouplings["B_pp"][:,:,i].diagonal()
				
			elif(Params.MoleculeName == 'H4'):
				print "H4 BOSONS --- !!!!!!!!!!!!!!!!!!!!! "
				self.NBos = 3
				self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)					
				self.GeneralCouplings["B_hh"] = numpy.zeros(shape=(Params.nocc, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_hp"] = numpy.zeros(shape=(Params.nocc, Params.nvirt,self.NBos),dtype = float)
				self.GeneralCouplings["B_ph"] = numpy.zeros(shape=(Params.nvirt, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_pp"] = numpy.zeros(shape=(Params.nvirt, Params.nvirt,self.NBos),dtype = float)		
				self.Freqs[0] = 0.214/EvPerAu
				self.Freqs[1] = 0.162/EvPerAu
				self.Freqs[2] = 0.214/EvPerAu
				self.a = 0.01
				for alpha in range(self.NBos): 
					self.GeneralCouplings["B_hh"][1][1][alpha] = 0.005
					self.GeneralCouplings["B_hh"][3][3][alpha] = 0.005
					self.GeneralCouplings["B_pp"][0][0][alpha] = 0.005
					self.GeneralCouplings["B_pp"][2][2][alpha] = 0.005
#				self.GeneralCouplings["B_hh"][1][1][alpha] = 0.001
#				self.GeneralCouplings["B_hh"][3][3][alpha] = 0.001					
			elif (Params.MoleculeName == 'C2F2'):
				print "C2F2 BOSONS --- !!!!!!!!!!!!!!!!!!!!! "
				# The first coupling is the one to the bath of continuous oscillators. 
				self.NBos = 3
				self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)					
				self.GeneralCouplings["B_hh"] = numpy.zeros(shape=(Params.nocc, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_hp"] = numpy.zeros(shape=(Params.nocc, Params.nvirt,self.NBos),dtype = float)
				self.GeneralCouplings["B_ph"] = numpy.zeros(shape=(Params.nvirt, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_pp"] = numpy.zeros(shape=(Params.nvirt, Params.nvirt,self.NBos),dtype = float)		
				nu2 = 0.214/EvPerAu
				nu8 = 0.162/EvPerAu
				nu4 = 0.114/EvPerAu
				self.Freqs[0] = nu2
				self.Freqs[1] = nu4
				self.Freqs[2] = nu8

				self.a = 0.01
				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][0] = 			 0.010
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][0] = 0.010 # 1 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][0] = 			 0.010
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][0] = 0.010

				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][1] = 			 0.002
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][1] = 0.002 # 2 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][1] = 			 0.002
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][1] = 0.002

				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][2] = 			 0.002*5/4
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][2] = 0.002*5/4 # 3 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][2] = 			 0.002*5/4
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][2] = 0.002*5/4
				
				self.a = 0.100*0.1
				
				if (False): 
					self.Freqs[3] = nu8+nu4
					self.Freqs[4] = 2*nu2
					self.Freqs[5] = 3*nu2
					self.Freqs[6] = 4*nu2
					self.Freqs[7] = 5*nu2
					self.Freqs[8] = 6*nu2
					self.Freqs[9] = 7*nu2
					self.Freqs[10] = 8*nu2
					self.Freqs[11] = 1*nu2 + nu4 + nu8
					self.Freqs[12] = 2*nu2 + nu4 + nu8
					self.Freqs[13] = 3*nu2 + nu4 + nu8
					self.Freqs[14] = 4*nu2 + nu4 + nu8
					self.Freqs[15] = 5*nu2 + nu4 + nu8
					self.Freqs[16] = 6*nu2 + nu4 + nu8
					self.Freqs[17] = 7*nu2 + nu4 + nu8
				
					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][3] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][3] = 0.001# 4 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][3] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][3] = 0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][4] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][4] = 0.001# 5 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][4] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][4] = 0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][5] =			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][5] = 0.001#6 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][5] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][5] = 0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][6] =			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][6] = 0.001#7 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][6] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][6] = 0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][7] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][7] = 0.001#8 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][7] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][7] = 0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][8] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][8] = 0.001#9 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][8] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][8] = 0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][9] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][9] = 0.001#10 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][9] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][9] = 0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][10] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][10] =0.001 #11 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][10] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][10] =0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][11] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][11] =0.001 #12 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][11] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][11] =0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][12] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][12] =0.001 #13 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][12] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][12] =0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][13] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][13] =0.001 #14 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][13] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][13] =0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][14] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][14] =0.001 #15 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][14] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][14] =0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][15] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][15] =0.001 #16 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][15] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][15] =0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][16] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][16] =0.001 #17 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][16] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][16] =0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][17] = 			 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][17] =0.001 #18 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][17] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][17] =0.001
			elif (Params.MoleculeName == 'C2F21'):
				print "C2F21 BOSONS --- !!!!!!!!!!!!!!!!!!!!! "
				# The first coupling is the one to the bath of continuous oscillators. 
				self.NBos = 3
				self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)					
				self.GeneralCouplings["B_hh"] = numpy.zeros(shape=(Params.nocc, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_hp"] = numpy.zeros(shape=(Params.nocc, Params.nvirt,self.NBos),dtype = float)
				self.GeneralCouplings["B_ph"] = numpy.zeros(shape=(Params.nvirt, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_pp"] = numpy.zeros(shape=(Params.nvirt, Params.nvirt,self.NBos),dtype = float)		
				nu2 = 0.214/EvPerAu
				nu8 = 0.162/EvPerAu
				nu4 = 0.114/EvPerAu
				self.Freqs[0] = nu2
				self.Freqs[1] = nu4
				self.Freqs[2] = nu8

				self.a = 0.01
				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][0] = 			 0.010
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][0] = 0.010 # 1 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][0] = 			 0.010
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][0] = 0.010

				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][1] = 			 0.002
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][1] = 0.002 # 2 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][1] = 			 0.002
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][1] = 0.002

				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][2] = 			 0.002*2
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][2] = 0.002*2 # 3 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][2] = 			 0.002*2
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][2] = 0.002*2
				self.OmegaCThree = 900*AuPerWavenumber
				self.a = 0.100/2
			elif (Params.MoleculeName == 'C2F22'):
				print "C2F22 BOSONS --- !!!!!!!!!!!!!!!!!!!!! "
				# The first coupling is the one to the bath of continuous oscillators. 
				self.NBos = 3
				self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)					
				self.GeneralCouplings["B_hh"] = numpy.zeros(shape=(Params.nocc, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_hp"] = numpy.zeros(shape=(Params.nocc, Params.nvirt,self.NBos),dtype = float)
				self.GeneralCouplings["B_ph"] = numpy.zeros(shape=(Params.nvirt, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_pp"] = numpy.zeros(shape=(Params.nvirt, Params.nvirt,self.NBos),dtype = float)		
				nu2 = 0.214/EvPerAu
				nu8 = 0.162/EvPerAu
				nu4 = 0.114/EvPerAu
				self.Freqs[0] = nu2
				self.Freqs[1] = nu4
				self.Freqs[2] = nu8

				self.a = 0.01
				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][0] = 			 0.010
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][0] = 0.010 # 1 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][0] = 			 0.010
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][0] = 0.010

				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][1] = 			 0.002
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][1] = 0.002 # 2 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][1] = 			 0.002
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][1] = 0.002

				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][2] = 			 0.002*3/2
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][2] = 0.002*3/2 # 3 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][2] = 			 0.002*3/2
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][2] = 0.002*3/2

				self.OmegaCThree = 1100*AuPerWavenumber
				self.a = 0.100
			elif (Params.MoleculeName == 'C2F2LB'):
				print "C2F2LB BOSONS --- !!!!!!!!!!!!!!!!!!!!! "

				self.NBos = 3
				self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)					
				self.GeneralCouplings["B_hh"] = numpy.zeros(shape=(Params.nocc, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_hp"] = numpy.zeros(shape=(Params.nocc, Params.nvirt,self.NBos),dtype = float)
				self.GeneralCouplings["B_ph"] = numpy.zeros(shape=(Params.nvirt, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_pp"] = numpy.zeros(shape=(Params.nvirt, Params.nvirt,self.NBos),dtype = float)		
				nu2 = 0.214/EvPerAu
				nu8 = 0.162/EvPerAu
				nu4 = 0.114/EvPerAu
				self.Freqs[0] = nu2
				self.Freqs[1] = nu4
				self.Freqs[2] = nu8

				self.a = 0.01
				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][0] = 			 0.010
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][0] = 0.010 # 1 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][0] = 			 0.010
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][0] = 0.010

				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][1] = 			 0.002
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][1] = 0.002 # 2 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][1] = 			 0.002
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][1] = 0.002

				self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][2] = 			 0.002*5/4
				self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][2] = 0.002*5/4 # 3 boson
				self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo()][2] = 			 0.002*5/4
				self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo("beta")][2] = 0.002*5/4
				
				self.a = 0.100*0.1

				if(False):
					self.NBos = 18
					self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)					
					self.GeneralCouplings["B_hh"] = numpy.zeros(shape=(Params.nocc, Params.nocc,self.NBos),dtype = float)
					self.GeneralCouplings["B_hp"] = numpy.zeros(shape=(Params.nocc, Params.nvirt,self.NBos),dtype = float)
					self.GeneralCouplings["B_ph"] = numpy.zeros(shape=(Params.nvirt, Params.nocc,self.NBos),dtype = float)
					self.GeneralCouplings["B_pp"] = numpy.zeros(shape=(Params.nvirt, Params.nvirt,self.NBos),dtype = float)		
					nu2 = 0.214/EvPerAu
					nu8 = 0.162/EvPerAu
					nu4 = 0.114/EvPerAu
					self.Freqs[0] = nu2
					self.Freqs[1] = nu4
					self.Freqs[2] = nu8
					self.Freqs[3] = nu8+nu4
					self.Freqs[4] = 2*nu2
					self.Freqs[5] = 3*nu2
					self.Freqs[6] = 4*nu2
					self.Freqs[7] = 5*nu2
					self.Freqs[8] = 6*nu2
					self.Freqs[9] = 7*nu2
					self.Freqs[10] = 8*nu2
					self.Freqs[11] = 1*nu2 + nu4 + nu8
					self.Freqs[12] = 2*nu2 + nu4 + nu8
					self.Freqs[13] = 3*nu2 + nu4 + nu8
					self.Freqs[14] = 4*nu2 + nu4 + nu8
					self.Freqs[15] = 5*nu2 + nu4 + nu8
					self.Freqs[16] = 6*nu2 + nu4 + nu8
					self.Freqs[17] = 7*nu2 + nu4 + nu8

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][0] = 						 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][0] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][0] = 					 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][0] = 0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][1] = 						 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][1] = 			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][1] = 					 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][1] = 0.001

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][2] = 						 0.001
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][2] =			 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][2] =					 0.001
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][2] = 0.001
					
					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][3] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][3] =			 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][0] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][3] = 0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][0] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][0] =			 0.001/2# 1 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][0] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][0] = 0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][1] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][1] =			 0.001/2# 2 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][1] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][1] = 0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][2] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][2] =			 0.001/2# 3 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][2] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][2] = 0.001/2
					
					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][3] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][3] =			 0.001/2# 4 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][3] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][3] = 0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][4] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][4] =			 0.001/2# 5 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][4] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][4] = 0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][5] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][5] =			 0.001/2# 6 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][5] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][5] = 0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][6] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][6] =			 0.001/2# 7 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][6] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][6] = 0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][7] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][7] =			 0.001/2# 8 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][7] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][7] = 0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][8] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][8] =			 0.001/2# 9 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][9] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][9] = 0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][10] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][10] =			 0.001/2 # 10 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][10] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][10] =0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][11] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][11] = 			 0.001/2 # 11 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][11] = 				 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][11] =0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][12] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][12] =			 0.001/2 # 12 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][12] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][12] =0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][13] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][13] = 			 0.001/2 # 13 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][13] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][13] =0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][14] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][14] =			 0.001/2 # 14 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][14] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][14] =0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][15] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][15] =			 0.001/2 # 15 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][15] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][15] =0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][16] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][16] =			 0.001/2 # 16 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][16] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][16] =0.001/2

					self.GeneralCouplings["B_hh"][Params.Homo()][Params.Homo()][17] =						 0.001/2
					self.GeneralCouplings["B_hh"][Params.Homo("beta")][Params.Homo("beta")][17] =			 0.001/2 # 17 boson
					self.GeneralCouplings["B_pp"][Params.Lumo()][Params.Lumo(plus=1)][17] =					 0.001/2
					self.GeneralCouplings["B_pp"][Params.Lumo("beta")][Params.Lumo(plus=1,spin="beta")][17] =0.001/2
			else:
				self.NBos = 10
				self.GeneralCouplings["B_hh"] = numpy.zeros(shape=(Params.nocc, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_hp"] = numpy.zeros(shape=(Params.nocc, Params.nvirt,self.NBos),dtype = float)
				self.GeneralCouplings["B_ph"] = numpy.zeros(shape=(Params.nvirt, Params.nocc,self.NBos),dtype = float)
				self.GeneralCouplings["B_pp"] = numpy.zeros(shape=(Params.nvirt, Params.nvirt,self.NBos),dtype = float)		
				self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)
				self.Freqs[0] = 0.34817095-0.34652797
				self.Freqs[1] = 0.0578021531956
				self.Freqs[2] = 2.*0.0578021531956
				self.Freqs[3] = 1600.0*AuPerWavenumber
				self.Freqs[4] = 2831.537137940873*AuPerWavenumber
				self.Freqs[5] = 2.*2831.537137940873*AuPerWavenumber
				self.Freqs[6] = 3.*2831.537137940873*AuPerWavenumber
				self.Freqs[7] = 4.*2831.537137940873*AuPerWavenumber
				self.Freqs[8] = 4.2*2831.537137940873*AuPerWavenumber
				self.Freqs[9] = 4.5*2831.537137940873*AuPerWavenumber

				for alpha in range(self.NBos): 
					for i in range(Params.nocc):
						self.GeneralCouplings["B_hh"][i][i][alpha] = 0.002*pow(self.Freqs[alpha],2.0)/((alpha+1.0))
					for i in range(Params.nvirt):
						self.GeneralCouplings["B_pp"][i][i][alpha] = 0.002*pow(self.Freqs[alpha],2.0)/(5.0)


		elif (self.ContBath): 
			self.NBos = 1 
			self.OmegaCThree = 900*AuPerWavenumber
			tmp = self.beta*self.OmegaCThree
			self.OhmicEqCF = (1./6.)*(1.0+2.0/pow(1.+tmp,2.0)+2.0/pow(1.+2.*tmp,2.0)+4./(2.*tmp+5.*tmp*tmp)) # This is the equilibrium cf of the super ohmic bath
		elif (True):  # These are the settings that I'm using to decay H4 in the discrete case. 
			self.NBos = 1
			self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)
			self.Freqs[0] = 1600.0*AuPerWavenumber
		elif (False):  # These are the settings that I'm using to decay H4 in the discrete case. 
			self.NBos = 2
			self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)
			self.Freqs[0] = 2831.537137940873*AuPerWavenumber
			self.Freqs[1] = 2.*2831.537137940873*AuPerWavenumber
		elif (False): 
			self.NBos = 3		
			self.Freqs = numpy.ndarray(shape = (self.NBos), dtype = float)			
			self.Freqs[0] = 0.0578021531956
			self.Freqs[1] = 2.*0.0578021531956
			self.Freqs[2] = 0.34817095-0.34652797

		# make the typical range of vibronic coupling between 0 and 1.5 
		self.MTilde = numpy.zeros(shape=(self.NBos,self.NSts)) #numpy.random.rand(self.NBos,self.NSts)*0.15
		nao = self.nocc/2
		na = self.NSts/2
		no = self.nocc
		nav = (self.NSts - self.nocc)/2
		# Increasing Tilde M is the physical case
		
		if (True): 
			for b in range(self.NBos): 
				MainS = (float(self.NBos-b)/float(self.NBos)+1.0)*0.15
				for i in range(na): 
					if (i < nao): 
						tmp = (i)*(MainS/(na)) #+ random.uniform(-0.03,0.03)
						self.MTilde[b][i] = tmp
						self.MTilde[b][i+nao] = tmp
					else: 
						tmp = (i)*(MainS/(na)) #+ random.uniform(-0.03,0.03)
						self.MTilde[b][i+nao] = tmp
						self.MTilde[b][i+nao+nav] = tmp
		
		
		# This decays H4 nicely. 
		if (False): 
			for b in range(self.NBos): 
				i = nao - 2  # Displace the homo-1. 
				self.MTilde[b][i] = 0.025
				self.MTilde[b][i+nao] = 0.025
				i = nao - 1  # Displace the homo. 
				self.MTilde[b][i] = 0.025
				self.MTilde[b][i+nao] = 0.025
				i = no # Displace the lumo
				self.MTilde[b][i] = 0.165
				self.MTilde[b][i+nav] = 0.165
				i = no+1 # Displace the lumo+1
				self.MTilde[b][i] = 0.165
				self.MTilde[b][i+nav] = 0.165
				for i in range(no+2, nao+na): 
					self.MTilde[b][i] = 0.0003
					self.MTilde[b][i+nav] = 0.0003

		if (True): 
			for b in range(self.NBos): 
				i = nao - 2  # Displace the homo-1. 
				self.MTilde[b][i] = 0.025
				self.MTilde[b][i+nao] = 0.025
				i = nao - 1  # Displace the homo. 
				self.MTilde[b][i] = 0.05
				self.MTilde[b][i+nao] = 0.05
				i = no # Displace the lumo
				self.MTilde[b][i] = 0.165
				self.MTilde[b][i+nav] = 0.165
				i = no+1 # Displace the lumo+1
				self.MTilde[b][i] = 0.055
				self.MTilde[b][i+nav] = 0.055
				for i in range(no+2, nao+na): 
					self.MTilde[b][i] = 0.0003
					self.MTilde[b][i+nav] = 0.0003

		self.MTilde *= 2.0
		
		# Try diplacing all virtuals by the same amount. 
		MainS = 1.0		
		if (False):
			for b in range(self.NBos): 
				for i in range(na): 
					if (i < nao): 
						tmp = 0.0
						self.MTilde[b][i] = tmp
						self.MTilde[b][i+nao] = tmp
					else: 
						tmp = .20
						self.MTilde[b][i+nao] = tmp
						self.MTilde[b][i+nao+nav] = tmp
										
		if (False): 
			for b in range(self.NBos): 
				MainS = (float(self.NBos-b)/float(self.NBos)+1.0)*0.2
				for i in range(na): 
					if (i < nao): 
						tmp = (i)*(MainS/(na)) + random.uniform(-0.03,0.03)
						self.MTilde[b][i] = tmp
						self.MTilde[b][i+nao] = tmp
					else: 
						tmp = (i)*(MainS/(na)) + random.uniform(-0.03,0.03)
						self.MTilde[b][i+nao] = tmp
						self.MTilde[b][i+nao+nav] = tmp
				
#		self.MTilde *= 0.0
		if (Params.ContBath): 
			self.Coths = numpy.arange(1,dtype=float)
			self.Coths[0] = self.OhmicEqCF
			return 
		else: 
			#Calculate Ns's once-and-for all and Fs once per time. 	
			Tmp = numpy.vectorize(self.Ns) 
			self.Nss = Tmp(numpy.arange(len(self.Freqs)))		
			Tmp2 = lambda s: Coth(self.Freqs[s]*self.beta/2)
			Tmp3 = numpy.vectorize(Tmp2)
			try: 
				self.Coths = Tmp3(numpy.arange(len(self.Freqs)))
			except Exception as Ex:
				print "Nan exception."
		return 
	
	def Print(self): 
		print " ----- Vibronic information ----- "
		print " T: ", self.Temp
		print " Beta: ", self.beta
		print " Freqs: ", self.Freqs
		print " Ns: ", self.Nss
		print " Coths: ", self.Coths
	#		print " M: ", self.M
		if (Params.Undressed): 
			print "hh",self.GeneralCouplings["B_hh"]
			print "pp",self.GeneralCouplings["B_pp"]
		else: 
			print " tilde{M}: ", self.MTilde		
		if (self.ContBath): 
			print "CONTINUOUS BATH IS BEING USED.... "
			print "self.OmegaCThree", self.OmegaCThree 
				