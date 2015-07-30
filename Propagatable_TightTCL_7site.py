import numpy as np
import scipy
#from scipy import special
from numpy import array
from numpy import linalg
from scipy import interpolate
from scipy import linalg
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

# A 7x7 tight binding model. 

class TightBindingTCL: 
	def __init__(self, Jin):
		# these things must be defined by the algebra. 
		self.VectorShape = None 
		self.ResidualShape = None
	
		self.Vnow = StateVector()
		self.Vnow["myrho"] = np.zeros(shape=(7,7)) 
		self.rho = self.Vnow["myrho"]
		self.rho[0,0] = 1 # and to start we set one of the values to one and leave the rest equal to zero.

		AuPerWavenumber = 4.5563*math.pow(10.0,-6.0)
		self.Hsys = np.zeros(shape=(7,7)) # This will be our system Hamiltonian, filled with the values below from renger. 
		couplings = [-87.7,5.5,-5.9,6.7,-13.7,-9.9,30.8,8.2,0.7,11.8,4.3,-53.5,-2.2,-9.6,6.0,-70.7,-17.0,-63.3,81.1,-1.3,39.7]
		siteE = [12445,12520,12205,12335,12490,12640,12450]
		index = 0
		for ii in range(7): # Construct Hsys in the right units...
			self.Hsys[ii,ii] = siteE[ii]*AuPerWavenumber
			for jj in range(ii+1,7):
				self.Hsys[ii,jj] = couplings[index]*AuPerWavenumber
				self.Hsys[jj,ii] = couplings[index]*AuPerWavenumber
				index += 1

		self.J = [[],[],[],[],[],[],[]] # This will hold our seven spectral densities
		self.Jindex = [[],[],[],[],[],[],[]] # This will hold the indices of our seven spectral densities
		self.Jvalue = [[],[],[],[],[],[],[]] # This will hold the values of our seven spectral densities
		# We need all three objects for the interpolation done later in our constructor. 

		import csv
		for ii in range(7): # Now we'll read in the spectral densities from seven .csv files
			reader = csv.reader(open('jdat/j' + str(ii+1) + '.csv', 'rb'))
			for row in reader:
				self.J[ii].append(row)
			for jj in range(len(self.J[ii])): # and construct our three objects
				self.J[ii][jj][0] = float(self.J[ii][jj][0])
				self.J[ii][jj][1] = float(self.J[ii][jj][1])
				self.Jindex[ii].append(float(self.J[ii][jj][0]))
				self.Jvalue[ii].append(float(self.J[ii][jj][1]))

		self.Nw = np.zeros(shape=(7,1)) # This will be the number of w values each of our spectral densities has
		for ii in range(7):
			self.Nw[ii] = len(self.J[ii])-1

		self.deltaw = np.zeros(shape=(7,1)) # This will be the change in w between any two entries in each of our spectral densities
		for ii in range(7):
			self.deltaw[ii] = self.J[ii][1][0]

		# Clenshaw-curtis points and weights:
		self.rawpts = [0.0000000001, 0.00084592086436589598, 0.00338082112902850573, 0.00759612349389597032, 0.01347756471008808058, 0.02100524384225556278, 0.03015368960704580797, 0.04089194655986299262, 0.05318367983829387590, 0.06698729810778067662, 0.08225609429353179017, 0.09893840362247810746, 0.11697777844051098240, 0.13631317921347565201, 0.15687918106563320714, 0.17860619515673033684, 0.20142070414860691757, 0.22524551096459698237, 0.25000000000000000000, 0.27560040989976891361, 0.30196011698042158815, 0.32898992833716563348, 0.35659838364445487345, 0.38469206462877991077, 0.41317591116653482557, 0.44195354293738488516, 0.47092758554476208573, 0.50000000000000000000, 0.52907241445523791427, 0.55804645706261511484, 0.58682408883346517443, 0.61530793537122008923, 0.64340161635554512655, 0.67101007166283436652, 0.69803988301957841185, 0.72439959010023108639, 0.75000000000000000000, 0.77475448903540301763, 0.79857929585139308243, 0.82139380484326966316, 0.84312081893436679286, 0.86368682078652434799, 0.88302222155948901760, 0.90106159637752189254, 0.91774390570646820983, 0.93301270189221932338, 0.94681632016170612410, 0.95910805344013700738, 0.96984631039295419203, 0.97899475615774443722, 0.98652243528991191942, 0.99240387650610402968, 0.99661917887097149427, 0.99915407913563410402, 1.00000000000000000000]
		self.weights = [0.000171526586620926244, 0.00165122470044098011, 0.00339111987353215106, 0.00504422805578450891, 0.00671248669542539131, 0.00834002819726836588, 0.00995091426546717072, 0.01152002633337574528, 0.01305618772778094983, 0.01454348142332781198, 0.01598533945562664699, 0.01736998493114156977, 0.01869849926418002619, 0.01996146321998176669, 0.02115888422875303256, 0.02228293018082077865, 0.02333318236729113377, 0.02430301914729495101, 0.02519197081023012843, 0.02599442576319749809, 0.02671010242300557311, 0.02733428427242412077, 0.02786704133835237994, 0.02830447974030215349, 0.028647138641665108276, 0.028891894419204383378, 0.029039843028758918469, 0.029088585817491658204, 0.029039843028758918469, 0.028891894419204383378, 0.028647138641665108276, 0.028304479740302153493, 0.027867041338352379937, 0.027334284272424120772, 0.026710102423005573108, 0.025994425763197498095, 0.025191970810230128428, 0.024303019147294951013, 0.023333182367291133766, 0.022282930180820778649, 0.021158884228753032561, 0.019961463219981766691, 0.018698499264180026191, 0.017369984931141569773, 0.015985339455626646993, 0.014543481423327811980, 0.013056187727780949827, 0.011520026333375745281, 0.009950914265467170718, 0.008340028197268365876, 0.006712486695425391306, 0.005044228055784508906, 0.003391119873532151056, 0.001651224700440980113, 0.000171526586620926244]
		
		# Rescale the pts over the range of each Ji(w)
		self.Wpts = np.zeros(shape=(7,len(self.rawpts)))
		for ii in range(7):
			self.Wpts[ii] = self.Nw[ii]*self.deltaw[ii]*self.rawpts

		# now we'll interpolate Ji(w):
		self.Jinterp = [[],[],[],[],[],[],[]]
		for ii in range(7):
			self.Jinterp[ii] = scipy.interpolate.interp1d(self.Jindex[ii], self.Jvalue[ii] , kind='linear')

		self.S = np.random.rand(7,7)*0.1

		#Make S hermitian#
		self.S += self.S.transpose()
	
		self.Mu = np.random.rand(7,7)*0.1

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

		self.VectorShape["r"] = np.zeros(shape = (7,7), dtype = complex)

		return 

	def DipoleMoment(self,AVector, Time=0.0): 
		Mu = np.tensordot(AVector["r"],self.Mu,axes=([0,1],[0,1]))
		return (1.0/3.0)*np.sum(Mu*Mu)

	def InitalizeNumerics(self): 
	# prepare some single excitation as the initial guess. 
	# homo->lumo (remembering alpha and beta.)
		self.VectorShape.MakeTensorsWithin(self.V0)
		r1_ph = self.V0["r1_ph"]
		# r1_hp = self.V0["r1_hp"]
		# Try just propagating ph->ph 
		r1_hp = np.zeros(shape=(Params.nocc,Params.nvirt) , dtype=complex)
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
				SumSq += Wght*np.conjugate(Wght)
			print "SumSq: ", SumSq


		Ens,Sts = self.ExactStates()
		# simply make an even mix bright states. 

		self.V0.Fill(0.0)
		for s in range(len(Ens)): 
			if (self.DipoleMoment(Sts[s]) > 0.001):
#			if (abs(Ens[s]-11.54/EvPerAu) < .1): 
				self.V0.Add(Sts[s])

		self.V0.MultiplyScalar(1.0/np.sqrt(self.V0.InnerProduct(self.V0)))		
				
		SumSq = 0.0
		for s in range(len(Ens)): 
			Wght = Sts[s].InnerProduct(self.V0)
			print "En: ", Ens[s]," (ev) ", Ens[s]*EvPerAu ,  " Weight ", Wght, " dipole: ", self.DipoleMoment(Sts[s])
			print "Contributions: "
			Sts[s].Print()
			SumSq += Wght*np.conjugate(Wght)
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

	def myterm(self, w, beta, t0): # As per Valleau et al eq 33.
		result = (1.0 + 2.0*np.exp(-w*beta) + 2.0*np.exp(-2.0*w*beta) + (2.0*np.exp((-5.0/2.0)*w*beta))/(w*beta)) * math.sin(w*t0)/w - 1j*(1.0/w + math.cos(w*t0)/w)
		return result # Note that we've performed a symbolic integration over time in here. 

	def numintJ(self, index, t0): # Here we will numerically integrate our J. Index denotes which site we are on.
		EvPerAu = 27.2113
		Kb = 8.61734315*pow(10.0,-5.0)/EvPerAu
		Temp = 303.15*2
		beta = (1.0/(Kb*Temp))
		vals = []

		vecMyTerm = np.vectorize(self.myterm) # Vectorize the function myterm
		vals = vecMyTerm(w = self.Wpts[index], beta = beta, t0 = t0) * self.Jinterp[index](self.Wpts[index]) # multiply each entry of myterm by the corresponding J(w)
		result = np.dot(self.weights, vals) # Then dot our value vector and our weight vector
		return result/np.pi # and divide by pi, as per Valleau eq 33.

	def buildHI(self, t0):
		V_SB = np.zeros(shape=(7,7),dtype=complex) # This will be our system - bath interaction potential at a time t0.
		for ii in range(7):
			V_SB[ii,ii] = self.numintJ(ii,t0) # which is just a diagonal matrix of the correlation functions we obtained from numerical integration above.
		# Then, we obtain the hamiltonian by left multiplying by exp(i Hsys t) and right multiplying by exp(-i Hsys t):
		HI = np.dot(np.dot(scipy.linalg.expm(t0*1j*self.Hsys),V_SB),scipy.linalg.expm(t0*(-1j)*self.Hsys)) 
		return HI

	def doublecomm(self, oldstate, t0, s0): # This will construct the four parts of the TCL2 term that come from the double commutator
		Ht = self.buildHI(t0) # We define our system-bath hamiltonian at times t0
		Hs = self.buildHI(s0) # and s0
		result = np.dot(np.dot(Ht,Hs),oldstate) # and add on each of the four parts one at a time
		result -= np.dot(np.dot(Ht,oldstate),Hs) 
		result -= np.dot(np.dot(Hs,oldstate),Ht) 
		result += np.dot(np.dot(oldstate,Hs),Ht) 
		return result

	def TCL(self, oldstate, t0): # Now we need to numerically integrate our double commutator from 0 to t ds. We will once again use the clenshaw curtis points and weights
		mypts = np.zeros(shape = (len(self.rawpts),1))
		for ii in range(len(self.rawpts)):
			mypts[ii] = self.rawpts[ii] * t0 # First we scale our clenshaw curtis points over our interval
		result = np.zeros(shape=(7,7),dtype=complex)
		for ii in range(len(mypts)): # Then we numerically integrate by computing our double commutator at points s0 = mypts, multiplying by the corresponding weight, and summing
			result += self.doublecomm(oldstate, t0, mypts[ii]) * self.weights[ii]
		return result # and return our result, which should be a matrix. 
	


	# now operate -i[H, ] on rho:
	def Step(self,OldState,Time,Field = None, AdiabaticMultiplier = 1.0): 
		NewState = OldState.clone()
		NewState.Fill()
		#self.VectorShape.EvaluateContributions(self.CISTerms,NewState,OldState)
		NewState["myrho"] += np.dot(self.Hsys,OldState["myrho"])
		NewState["myrho"] -= np.dot(OldState["myrho"].self.Hsys)

		#if (Field != None):
		#	if (sqrt(Field[0]*Field[0]+Field[1]*Field[1]+Field[2]*Field[2]) > Params.FieldThreshold):
		#		NewState["r1_ph"] += Integrals["mux_ph"]*Field[0]
		#		NewState["r1_ph"] += Integrals["muy_ph"]*Field[1]
		#		NewState["r1_ph"] += Integrals["muz_ph"]*Field[2]			
		#		self.VectorShape.EvaluateContributions(self.MuTerms,NewState,OldState, FieldInformation = Field)		
		NewState.MultiplyScalar(complex(0,-1.0))
#   NOTE: the overall negative one factor for the pertubative term is included in self.Pterms. 
		# second term goes here
		#self.VectorShape.EvaluateContributions(self.PTerms,NewState,OldState,TimeInformation=Time, MultiplyBy=AdiabaticMultiplier)
		return NewState
