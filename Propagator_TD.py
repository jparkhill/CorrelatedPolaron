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
from LooseAdditions import * 
from TensorNumerics import * 
from Wick import * 
#from MatrixExponential import * 
from SpectralAnalysis import * 
from NumGrad import * 
from Propagator import *
	   
#
# This does the exterior work of propagation for a time dependent Hamiltonian matrix. 
# d(Rho)/dt = H(t)Rho 
#

class Propagator_TD(Propagator): 
	def __init__(self,a_propagatable, Field = None):
		self.ToExp = None
		self.Propm = None
		self.TimesEvaluated = []
		
		Propagator.__init__(self,a_propagatable)
				
		if (Params.ExponentialStep): 
			self.Propm = self.ShortTimePropagator.NonPertMtrx + self.ShortTimePropagator.MarkovMatrix
		if (False and Params.ExponentialStep):
			Tmp = self.ShortTimePropagator.NonPertMtrx + self.ShortTimePropagator.MarkovMatrix
			self.ToExp = Tmp.reshape((Params.nocc*Params.nvirt,Params.nocc*Params.nvirt))
			print "ToExp:",self.ToExp
			import scipy.linalg.matfuncs
			self.Propm = scipy.linalg.matfuncs.expm(self.ToExp*Params.TStep)
		
		self.Field = Field
		if (not hasattr(a_propagatable,"CISWeights") or Params.Undressed):
			Params.DoCisDecomposition = False	
		if (not hasattr(a_propagatable,"MuTerms") or Params.Undressed):
			self.Field = None 
		if (not hasattr(a_propagatable,"EntropyAndONSpectrum") or Params.Undressed): 
			Params.DoBCT = Params.DoEntropies = False

		if (Params.AllDirections): 
			Params.DoBCT = False
			Params.DoEntropies = False

		if (Params.Adiabatic >= 2): 
			Params.DoBCT = False
		if (Params.DoBCT): 
			self.BathCF = numpy.zeros(self.NSteps,dtype = complex)
			self.BathCorrelationIntegral = numpy.zeros(self.NSteps,dtype = complex)
			self.BathCorrelationIntegrand = numpy.zeros(self.NSteps,dtype = complex)		
		if (Params.DoEntropies): 
			self.TimeDependentEntropy = numpy.zeros(self.NSteps,dtype = complex)
			self.ONSpectrum = numpy.zeros(shape=(self.NSteps,len(self.ShortTimePropagator.EntropyAndONSpectrum(self.ShortTimePropagator.V0)[1])),dtype = complex)
		print "Ready For propagation... "
		return 
		
	def Propagate(self, aPipe=None): 
		Step=0

#		print "------------------------ r1_ph --------------------------------"
#		print self.VNow["r1_ph"]

		if self.Field != None : 
			self.VNow.MultiplyScalar(0.000001)
		
		if (Params.DoCisDecomposition): 
			CisDecomposition = numpy.zeros(shape = (len(self.Dipoles),len(self.ShortTimePropagator.CISWeights(self.VNow))),dtype = complex)
		
		# Checks to make sure every process starts. 
		if (Params.AllDirections): 
			aPipe.send(None)
			
		while (self.TNow < Params.TMax): 
			# returns an estimate of the energy. 
			TimingStart = time.time()

			# Because of the variable timestep you can exhaust self.Dipoles. 
			# So resize if you come close to filling it. 
			if (len(self.Dipoles)-Step < 2000):
				if (len(self.Dipoles.shape)>1): 
					self.Dipoles.resize((len(self.Dipoles)+2000,3))
					self.Norms.resize(len(self.Norms)+2000)					
				else: 
					self.Dipoles.resize(len(self.Dipoles)+2000)
					self.Norms.resize(len(self.Norms)+2000)					
			
			if(Params.RK45): # should be default. 
				self.RK45Step(self.VNow,self.TNow)
			elif (Params.ExponentialStep): 
				self.ExponentialStep(self.VNow,self.TNow)				
			else:
				self.RungeKuttaStep(self.VNow,self.TNow)

#			Whether to force Hermiticity on the time-dependent state. 
			if ("r1_hh" in self.VNow):
				self.VNow["r1_hh"] += self.VNow["r1_hh"].conj().transpose()
				self.VNow["r1_hh"] *= 0.5
			if ("r1_pp" in self.VNow):			 
				self.VNow["r1_pp"] += self.VNow["r1_pp"].conj().transpose()
				self.VNow["r1_pp"] *= 0.5
			if ("r1_hp" in self.VNow and "r1_ph" in self.VNow): 
				self.VNow["r1_ph"] += self.VNow["r1_hp"].conj().transpose()
				self.VNow["r1_ph"] *= 0.5
				self.VNow["r1_hp"] = self.VNow["r1_ph"].conj().transpose()
			
			self.TimesEvaluated.append(self.TNow)
			if (self.Field == None): 
				# Only collect dipoles for fourier transform if the field is off. 
				self.Dipoles[Step] = self.ShortTimePropagator.DipoleMoment(self.VNow,self.TNow)			
			elif (self.TNow > self.Field.TOff()):
				self.Dipoles[Step] = self.ShortTimePropagator.DipoleMoment(self.VNow,self.TNow)			
			self.Norms[Step] = sqrt(self.VNow.InnerProduct(self.VNow).real)
			TimingEnd = time.time()
			
#			if (self.Field != None): 
#				print "T: ", self.TNow, " E: ", round(self.ShortTimePropagator.Energy(self.VNow),4), " Mu : ", '{0:s}'.format(self.Dipoles[Step]), " |St| ", round(self.Norms[Step],4), " Applied Mu: ", numpy.sum(self.Field(self.TNow)) , " WallMinToGo: ", round(((Params.TMax-self.TNow)/Params.TStep)*(TimingEnd-TimingStart)/60.0)

			if (Params.AllDirections): 
				print "T"+self.ShortTimePropagator.Polarization+": ", self.TNow, " Mu : ", '{0:s}'.format(self.Dipoles[Step]), " |St| ", round(self.Norms[Step],4), " WallMinToGo: ", round(((Params.TMax-self.TNow)/Params.TStep)*(TimingEnd-TimingStart)/60.0)				
			elif (Params.DirectionSpecific): 
				print "T: ", self.TNow, " Mu : ", '{0:s}'.format(self.Dipoles[Step]), " |St| ", round(self.Norms[Step],4), " WallMinToGo: ", round(((Params.TMax-self.TNow)/Params.TStep)*(TimingEnd-TimingStart)/60.0)
			else:
				print "T: ", self.TNow, " Mu : ", round(self.Dipoles[Step],4), " |St| ", round(self.Norms[Step],4), " WallMinToGo: ", round(((Params.TMax-self.TNow)/Params.TStep)*(TimingEnd-TimingStart)/60.0)

			if (Params.DoCisDecomposition): 
				CisDecomposition[Step] = self.ShortTimePropagator.CISWeights(self.VNow)
				
			# optional things to plot. 
			if (Params.DoBCT): 
				self.BathCorrelationIntegral[Step], self.BathCorrelationIntegrand[Step], self.BathCF[Step] = self.ShortTimePropagator.BathCorrelationMeasures()
			if (Params.DoEntropies): 
				self.TimeDependentEntropy[Step], self.ONSpectrum[Step] = self.ShortTimePropagator.EntropyAndONSpectrum(self.VNow)
			Step += 1

		print "Propagation Complete... Collecting Data. "

		# Return dipole to generate isotropic dipole spectrum... 
		if Params.AllDirections: 
			ToInterpolate = None
			if (self.ShortTimePropagator.Polarization == "x"):
				ToInterpolate = self.Dipoles[:len(self.TimesEvaluated),0]
			elif (self.ShortTimePropagator.Polarization == "y"):
				ToInterpolate = self.Dipoles[:len(self.TimesEvaluated),1]
			elif (self.ShortTimePropagator.Polarization == "z"):
				ToInterpolate = self.Dipoles[:len(self.TimesEvaluated),2]						
			print "Interpolating... "
			from scipy import interpolate
			from scipy.interpolate import InterpolatedUnivariateSpline 
			interpf = InterpolatedUnivariateSpline(self.TimesEvaluated, ToInterpolate, k=1)
			print "Resampling at intervals of: ",Params.TMax/3000.," Atomic units"			
			Tstep = Params.TMax/3000.
			print "Resolution up to: ", (1/Tstep)*EvPerAu
			Times = numpy.arange(0.0,Params.TMax,Tstep)
			Mu = interpf(Times)
			print "Saving Resampled Dipoles... "
			numpy.savetxt('./Output'+Params.MoleculeName+'/Dipole'+self.ShortTimePropagator.Polarization,Mu,fmt='%.18e')	
			print "Saving Last State Vector"
			self.VNow.Save('./Output'+Params.MoleculeName+'/LastState'+self.ShortTimePropagator.Polarization)
			print "Sending Array of shape: ", Mu.shape
			aPipe.send((Tstep,Mu))
			print "sent... (Killing This Child Process)"
			return 

		self.Norms[-1] = self.Norms[-2]
		if (Params.DoEntropies): 
			self.TimeDependentEntropy[-1] = self.TimeDependentEntropy[-2]		
		
		numpy.savetxt('./Output'+Params.MoleculeName+'/TimesEvaluated',self.TimesEvaluated,fmt='%.18e')	
		numpy.savetxt('./Output'+Params.MoleculeName+'/Dipoles',self.Dipoles,fmt='%.18e')	
		numpy.savetxt('./Output'+Params.MoleculeName+'/Norms',self.Norms,fmt='%.18e')	

		if (Params.DoEntropies): 
			numpy.savetxt('./Output'+Params.MoleculeName+'/ONSpectrum',self.ONSpectrum,fmt='%.18e')		
	
		if (Params.Plotting): 
			import matplotlib.pyplot as plt
			import matplotlib.font_manager as fnt
			# Make plot styles visible. 
			PlotFont = {'fontname':'Helvetica','fontsize':18,'weight':'bold'}
			LegendFont = fnt.FontProperties(family='Helvetica',size='17',weight='bold')	
			l1 = plt.plot(self.TimesEvaluated,self.Norms,'k--')
			plt.setp(l1,linewidth=2, color='r')
			plt.xlabel('Time(au)',fontsize = Params.LabelFontSize)
			plt.ylabel('|State|',fontsize = Params.LabelFontSize)
			plt.savefig("./Figures"+Params.MoleculeName+"/"+'NormOfState')
			plt.clf()

			if (Params.DirectionSpecific):
				lx,ly,lz = plt.plot(self.TimesEvaluated,self.Dipoles[:,0],'k',self.TimesEvaluated,self.Dipoles[:,1],'k--',self.TimesEvaluated,self.Dipoles[:,2],'k.')
				plt.setp(lx,linewidth=2, color='r')
				plt.setp(ly,linewidth=2, color='g')
				plt.setp(lz,linewidth=2, color='b')		
				plt.legend(['x','y','z'],loc=2)
				plt.xlabel('Time (au)',fontsize = Params.LabelFontSize)
				plt.ylabel('Mu (au)',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+'Dipole')
				plt.clf()
				Nrm = lambda X: (X[0]*X[0]+X[1]*X[1]+X[2]*X[2])
				DipoleStrength = map(Nrm,self.Dipoles)				
				SpectralAnalysis(numpy.array(DipoleStrength), Params.TStep, DesiredMaximum = 26.0/EvPerAu,Smoothing = True)
				if(self.fluorescence): #blau, put the flourescence code here.
					SpectralAnalysis( stuff goes here, Flouresce = True)
			else : 
				l1 = plt.plot(self.TimesEvaluated,self.Dipoles,'k--')
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Time(au)',fontsize = Params.LabelFontSize)
				plt.ylabel('|Mu|',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+'Dipole')
				plt.clf()
				SpectralAnalysis(self.Dipoles, Params.TStep, DesiredMaximum = 26.0/EvPerAu,Smoothing = True)

				
			if(Params.Correlated):
				if (hasattr(self.ShortTimePropagator.PTerms[0],"Bs") ): 		
					Ns = numpy.vectorize(lambda X: 1.0/(numpy.exp(X*self.ShortTimePropagator.Bs.beta)-1.0))	
					Freqs = numpy.arange(min(self.ShortTimePropagator.PTerms[0].Bs.Freqs),max(self.ShortTimePropagator.PTerms[0].Bs.Freqs),max(self.ShortTimePropagator.PTerms[0].Bs.Freqs)/100.0)
					Nss = Ns(Freqs)
					l1 = plt.plot(Freqs,Nss,'k--')
					plt.setp(l1,linewidth=2, color='r')
					for Freq in self.ShortTimePropagator.Bs.Freqs: 
						plt.axvline(x=Freq, ymin=0, ymax=1)
					plt.xlabel('Freq')
					plt.ylabel('Ns')
					plt.savefig("./Figures"+Params.MoleculeName+"/"+'BosonOccupations')
					plt.clf()

			if Params.DoCisDecomposition: 
				Fsself.TimesEvaluated = self.TimesEvaluated*FsPerAu
				Energies = numpy.array(self.ShortTimePropagator.CISEnergies)
				CisDecomposition[-1] = CisDecomposition[-2]
				for k in range(len(CisDecomposition[0])): 
					if (len(CisDecomposition[0]) > 2): 
						plt.plot(Fsself.TimesEvaluated,CisDecomposition[:,k]*CisDecomposition[:,k],label = '{0:g}'.format(Energies[k]*27.2113))
					else: 
						# this is an unrepentant hack. 
						plt.plot(Fsself.TimesEvaluated,CisDecomposition[:,k]*CisDecomposition[:,k],label = '{0:g}'.format(int(k)))					
				l3 = plt.plot(Fsself.TimesEvaluated,self.Norms,label='Total Norm')
				LegendFont = fnt.FontProperties(family='Helvetica',size='8')	
				plt.xlabel('Time (fs)',fontsize = Params.LabelFontSize)
				plt.ylabel('Probability of CIS State',fontsize = Params.LabelFontSize)
				plt.legend(loc=2)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+'CISDecomposition')
				plt.clf()
				LegendFont = fnt.FontProperties(family='Helvetica',size='17',weight='bold')					

			if (self.Field != None): 
				Fields = numpy.vectorize(self.Field.NormNow)
				l1 = plt.plot(self.TimesEvaluated,Fields(self.TimesEvaluated),'k--')
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Time(au)',fontsize = Params.LabelFontSize)
				plt.ylabel('|Mu|',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+'AppliedField')
				plt.clf()
				SpectralAnalysis(Fields(self.TimesEvaluated), Params.TStep, DesiredMaximum = 23.0/EvPerAu, Title = "AppliedFields",Smoothing = False)
			
			if (Params.DoBCT): 			
				l1,l2 = plt.plot(self.TimesEvaluated,self.BathCorrelationIntegrand.real,'k--',self.TimesEvaluated,self.BathCorrelationIntegrand.imag,'k.')
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Time(au)',fontsize = Params.LabelFontSize)
				plt.ylabel('Average BC',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+'BathCorrelationIntegrand')
				plt.clf()
				l1,l2 = plt.plot(self.TimesEvaluated,self.BathCorrelationIntegral.real,'k--',self.TimesEvaluated,self.BathCorrelationIntegral.imag,'k.')
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Time(au)',fontsize = Params.LabelFontSize)
				plt.ylabel('Average BC',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+'BathCorrelationIntegral')
				plt.clf()
							
				l1,l2 = plt.plot(self.TimesEvaluated,self.BathCF.real,'k--',self.TimesEvaluated,self.BathCF.imag,'k.')
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Time(au)',fontsize = Params.LabelFontSize)
				plt.ylabel('Average BC',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+'BathCF')
				plt.clf()
				# The imaginary part for some reason lacks the right oscillation. 
				SpectralAnalysis(self.BathCF, Params.TStep, DesiredMaximum = 23.0/EvPerAu, Title = "TransformedBathCF",Smoothing = True)						
				SpectralAnalysis(self.BathCF.real, Params.TStep, DesiredMaximum = 23.0/EvPerAu, Title = "ImTransformedBathCF",Smoothing = True)						
				SpectralAnalysis(self.BathCF.real, Params.TStep, DesiredMaximum = 23.0/EvPerAu, Title = "RealTransformedBathCF",Smoothing = True)						
				SpectralAnalysis(self.BathCorrelationIntegral, Params.TStep, DesiredMaximum = 23.0/EvPerAu, Title = "TransformedBathIntegrals",Smoothing = False)
			
			if (Params.DoEntropies):
				l1 = plt.plot(self.TimesEvaluated,self.TimeDependentEntropy,'k--')
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Time(au)',fontsize = Params.LabelFontSize)
				plt.ylabel('|Mu|',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+'Entropy')
				plt.clf()
				LegendLabels = []
				for k in range(self.ONSpectrum.shape[1]):
					plt.plot(self.TimesEvaluated,self.ONSpectrum[:,k],label=str(k))
				plt.xlabel('Time(au)',fontsize = Params.LabelFontSize)
				plt.ylabel('Transition Density Spectrum',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+'Number Spectrum')
				plt.clf()
		return 

	def AdiabaticRamp(self,Time): 
		if Time < 25: 
			return 0.0
		elif Time >= 25 and Time < 125: 
			return 1.0*((Time - 25.)/100.0)
		elif Time >=125 and Time < 225: 
			return 1.0*(100.0-(Time - 125.))/100.0
		else: 
			return 0.0

	def ExponentialStepDebug(self,VNow,TNow): 	
		self.TNow += Params.TStep	
		Tmp = self.ShortTimePropagator.NonPertMtrx + self.ShortTimePropagator.MarkovMatrix
		self.ToExp = Tmp.reshape((Params.nocc*Params.nvirt,Params.nocc*Params.nvirt))
	#	print "ToExp:",self.ToExp
		import scipy.linalg.matfuncs
		self.Propm = scipy.linalg.matfuncs.expm(self.ToExp*self.TNow)
		Tmp2 = self.Propm.reshape(Params.nocc,Params.nvirt,Params.nocc,Params.nvirt)
		self.VNow["r1_ph"] = numpy.tensordot(Tmp2,self.ShortTimePropagator.V0["r1_ph"],axes=([2,3],[0,1]))
		#tmp = self.Propm.reshape((Params.nocc,Params.nvirt,Params.nocc,Params.nvirt))
		#self.VNow["r1_ph"] = numpy.tensordot(tmp,self.VNow["r1_ph"],axes=([2,3],[0,1]))
		#self.TNow += Params.TStep	
		return 

	def ExponentialStep(self,VNow,TNow): 	
	#	import scipy.linalg.matfuncs
	#	self.Propm = scipy.linalg.matfuncs.expm(self.ToExp*self.TNow)
		V1 = numpy.tensordot(self.Propm,self.VNow["r1_ph"],axes=([2,3],[0,1]))
		V2 = numpy.tensordot(self.Propm,self.VNow["r1_ph"]+0.5*Params.TStep*V1,axes=([2,3],[0,1]))		
		V3 = numpy.tensordot(self.Propm,self.VNow["r1_ph"]+0.5*Params.TStep*V2,axes=([2,3],[0,1]))		
		V4 = numpy.tensordot(self.Propm,self.VNow["r1_ph"]+Params.TStep*V3,axes=([2,3],[0,1]))
		VNow["r1_ph"] += (1.0/6.0)*Params.TStep*V1
		VNow["r1_ph"] += (2.0/6.0)*Params.TStep*V2
		VNow["r1_ph"] += (2.0/6.0)*Params.TStep*V3
		VNow["r1_ph"] += (1.0/6.0)*Params.TStep*V4
		self.TNow += Params.TStep	
		return 

	def RungeKuttaStep(self,VNow,TNow): 	
		if (self.Field != None) : 
			V1 = self.ShortTimePropagator.Step(VNow,TNow,Field=self.Field(self.TNow))
			V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V1),TNow+0.5*Params.TStep,Field=self.Field(self.TNow+0.5*Params.TStep))
			V3 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V2),TNow+0.5*Params.TStep,Field=self.Field(self.TNow+0.5*Params.TStep))
			V4 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,Params.TStep,V3),TNow+Params.TStep,Field=self.Field(self.TNow+Params.TStep))
		else: 
			V1 = self.ShortTimePropagator.Step(VNow,TNow)
			V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V1),TNow+0.5*Params.TStep)
			V3 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V2),TNow+0.5*Params.TStep)
			V4 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,Params.TStep,V3),TNow+Params.TStep)
		VNow.Add(V1,(1.0/6.0)*Params.TStep)
		VNow.Add(V2,(2.0/6.0)*Params.TStep)
		VNow.Add(V3,(2.0/6.0)*Params.TStep)
		VNow.Add(V4,(1.0/6.0)*Params.TStep)	
		self.TNow += Params.TStep		
		return 

	def RK45Step(self,VNow,TNow):
		print "TimeStep", Params.TStep
		a = [ 0.0, 0.2, 0.3, 0.6, 1.0, 0.875 ]
 		b = [[],
      		  [0.2],
      		  [3.0/40.0, 9.0/40.0],
       		  [0.3, -0.9, 1.2],
      		  [-11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0],
       		  [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]]
 		c  = [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0]
  		dc = [c[0]-2825.0/27648.0, c[1]-0.0, c[2]-18575.0/48384.0, c[3]-13525.0/55296.0, c[4]-277.00/14336.0, c[5]-0.25]

  		if(self.TNow + Params.TStep > Params.TMax):
  			Params.TStep = Params.TMax - self.TNow 
		if (self.Field != None) : 
			print "To Be Done"
		else: 
			V1 = self.ShortTimePropagator.Step(VNow,TNow) #This is fine
			V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,b[1][0]*Params.TStep,V1),TNow+a[1]*Params.TStep) #This is fine
			V3 = self.ShortTimePropagator.Step(VNow.TwoCombo(1.0, V1, Params.TStep*b[2][0], V2, Params.TStep*b[2][1]),TNow+a[2]*Params.TStep)
			V4 = self.ShortTimePropagator.Step(VNow.ThreeCombo(1.0, V1, Params.TStep*b[3][0], V2, Params.TStep*b[3][1], V3, Params.TStep*b[3][2]),TNow+a[3]*Params.TStep)
			V5 = self.ShortTimePropagator.Step(VNow.FourCombo(1.0, V1, Params.TStep*b[4][0], V2, Params.TStep*b[4][1], V3, Params.TStep*b[4][2], V4, Params.TStep*b[4][3]),TNow+a[4]*Params.TStep)
			V6 = self.ShortTimePropagator.Step(VNow.FiveCombo(1.0, V1, Params.TStep*b[5][0], V2, Params.TStep*b[5][1], V3, Params.TStep*b[5][2], V4, Params.TStep*b[5][3], V5, Params.TStep*b[5][4]),TNow+a[4]*Params.TStep)

		E = V1.FiveCombo(dc[0], V2, dc[1], V3, dc[2], V4, dc[3], V5, dc[4], V6, dc[5])
		Error = E.InnerProduct(E).real
		Emax = Params.tol*VNow.InnerProduct(VNow).real
		#print V1.InnerProduct(V1), V2.InnerProduct(V2), V3.InnerProduct(V3), V4.InnerProduct(V4), V5.InnerProduct(V5), V6.InnerProduct(V6) 
		#print VNow.InnerProduct(VNow)

		#print "E ", Error, "   Emax ", Emax
		
		if Error < Emax or Emax == 0.0:
			self.TNow += Params.TStep
			VNow.Add(V1,c[0]*Params.TStep)
			VNow.Add(V2,c[1]*Params.TStep)
			VNow.Add(V3,c[2]*Params.TStep)
			VNow.Add(V4,c[3]*Params.TStep)
			VNow.Add(V5,c[4]*Params.TStep)
			VNow.Add(V6,c[5]*Params.TStep)
		else:
			Params.TStep = Params.Safety*Params.TStep*(Emax/Error)**0.2
			V1 = self.ShortTimePropagator.Step(VNow,TNow) #This is fine
			V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,b[1][0]*Params.TStep,V1),TNow+a[1]*Params.TStep) #This is fine
			V3 = self.ShortTimePropagator.Step(VNow.TwoCombo(1.0, V1, Params.TStep*b[2][0], V2, Params.TStep*b[2][1]),TNow+a[2]*Params.TStep)
			V4 = self.ShortTimePropagator.Step(VNow.ThreeCombo(1.0, V1, Params.TStep*b[3][0], V2, Params.TStep*b[3][1], V3, Params.TStep*b[3][2]),TNow+a[3]*Params.TStep)
			V5 = self.ShortTimePropagator.Step(VNow.FourCombo(1.0, V1, Params.TStep*b[4][0], V2, Params.TStep*b[4][1], V3, Params.TStep*b[4][2], V4, Params.TStep*b[4][3]),TNow+a[4]*Params.TStep)
			V6 = self.ShortTimePropagator.Step(VNow.FiveCombo(1.0, V1, Params.TStep*b[5][0], V2, Params.TStep*b[5][1], V3, Params.TStep*b[5][2], V4, Params.TStep*b[5][3], V5, Params.TStep*b[5][4]),TNow+a[4]*Params.TStep)
			self.TNow += Params.TStep
			VNow.Add(V1,c[0]*Params.TStep)
			VNow.Add(V2,c[1]*Params.TStep)
			VNow.Add(V3,c[2]*Params.TStep)
			VNow.Add(V4,c[3]*Params.TStep)
			VNow.Add(V5,c[4]*Params.TStep)
			VNow.Add(V6,c[5]*Params.TStep)
			return

		if (Emax == 0):
			Params.TStep = Params.TStep*2.0
			return
		Params.TStep = Params.Safety*Params.TStep*(Emax/Error)**0.2
		return 
		
	def ImRungeKuttaStep(self,VNow,TNow): 	
		V1 = self.ShortTimePropagator.Step(VNow,TNow)
		V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V1),TNow+0.5*Params.TStep)
		V3 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V2),TNow+0.5*Params.TStep)
		V4 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,Params.TStep,V3),TNow+Params.TStep)
		VNow.Add(V1,(1.0/6.0)*(1.0j)*Params.TStep)
		VNow.Add(V2,(2.0/6.0)*(1.0j)*Params.TStep)
		VNow.Add(V3,(2.0/6.0)*(1.0j)*Params.TStep)
		VNow.Add(V4,(1.0/6.0)*(1.0j)*Params.TStep)	
		self.TNow += Params.TStep		
		return 
				