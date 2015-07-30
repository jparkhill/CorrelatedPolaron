import numpy, scipy
#from scipy import special
from numpy import array
from numpy import linalg
# standard crap
import __builtin__ 
# something is redefining sum. 
import os, sys, time, math, re, random, cmath
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, log, pi, sqrt, isnan
from LooseAdditions import * 
from NumGrad import * 
from TensorNumerics import *

EvPerAu = 27.2113
	   
#
#  This is a total Hack, giving the positions and heights for Exact and CIS 
#

ExactStates = [0.0]
ExactMoments = [0.0]
CISStates = [0.0]
CISMoments = [0.0]
CISDStates = [0.0]
CISDMoments = [0.0]
ADCStates = [0.0]
ADCMoments = [0.0]

# h4 in sto-3g
if (Params.MoleculeName == 'H4'):
	ExactStates = [12.1303,13.0332, 20.5407,20.6194,22.9708,29.8450]
	ExactMoments = [0.19500,1.34323,0.83273,0.00000,1.04753, 0.04113]
	CISStates = [3.8113, 9.5723, 11.5422, 17.2466, 18.3783, 22.6084, 22.6981,27.1071]
	CISMoments = [0.0000, 0.0000, 0.8545, 0.2415, 0.0000, 1.3440, 0.0000, 0.0080]
	CISDStates = [12.2810, 16.3122, 22.1965, 27.2765, 4.5116, 10.1989, 19.4070]
	CISDMoments = [0.8545, 0.2415, 1.3440, 0.0080, 0.0000, 0.0, 0.0, 0.0]
	ADCStates = [7.84714, 12.23696, 14.96185, 19.16222, 27.21851, 29.87175]
	ADCMoments = [0.0,0.0,0.0,0.0,0.0,0.0]

# These are for H4, in the DZ. 
if False: 
	ExactStates = [8.2977,8.8821,14.9034,15.1880,15.4442,18.6930,19.0694,21.9516]
	ExactMoments = [1.41250,0.02249,1.12815,1.20760, 0.00000, 0.20173, 0.13964, 0.01182]
	ADCStates = [7.84714, 12.23696, 14.96185, 19.16222, 27.21851]
	ADCMoments = [0.444139, 0.001734, 2.179479, 0.013379, 0.001712]
	CISDStates = [7.8599, 12.1977, 14.9599, 19.1429]
	CISDMoments = [0.5480, 0.0032, 1.5073, 0.0175]
	CISStates = [2.6996, 7.2580, 7.4339, 12.5143, 12.9570, 14.9162, 16.7363, 19.1885, 27.1364]
	CISMoments = [0.0000, 0.0000, 0.5480, 0.0000, 0.0032, 1.5073, 0.0000, 0.0175, 0.0000]

# This is for BH3 in Sto-3g
if Params.MoleculeName == 'BH3' : 
	ExactStates = [6.9670, 7.7157 ,14.9602 ,17.3394 ,17.4891 ,20.0649 ,21.3179 ,21.6709 ,22.0354 , 22.8722]
	ExactMoments = [ 0.0130, 0.03758, 1.05912 , 0.00685 , 0.03845 , 0.10665 , 0.47261 , 0.00000 , 0.01626 , 0.64112 ]
	CISStates = [ 6.6744 , 7.5164, 7.6835,  8.5451, 11.9717, 15.7432, 16.4790,17.0950, 17.2743,  21.5812, 22.1433, 22.7809,  22.7986,  23.4035, 23.7109, 24.9729]
	CISMoments = [ 0.0000,  0.0000, 0.0001,  0.0005, 0.0000, 0.5030,  0.0000, 0.0000, 0.0000,  0.0000, 0.2462, 0.0146,  0.0000, 1.3304, 0.5781, 1.4186]
	CISDStates = [0.0, 0.0]
	CISDMoments = [0.0000, 0.0000]
	ADCStates = [7.85368, 9.06496, 16.86377, 17.62648, 18.83761, 24.15848]
	ADCMoments = [0.001146, 0.002703, 0.474132, 0.000000, 0.000000, 0.000000]

# This is for H4 transfer model in Dz. 
if (Params.MoleculeName == '2H2'):
	ExactStates = [17.4871, 18.3532, 21.9012, 22.2010]
	ExactMoments = [0.57212, 1.38506, 0.25474, 0.10678]
	CISStates = [17.8400,18.8050, 22.0737, 22.2843]
	CISMoments =[ 0.1860, 0.9823, 0.0361, 0.0097]
	CISDStates = [0.0]
	CISDMoments = [0.0]
	ADCStates = [0.0]
	ADCMoments = [0.0]
	
# The basis is mixed, DZ for everything but F. 
if (Params.MoleculeName == 'C2F2' or Params.MoleculeName == 'C2F2LB'):
	ExactStates = [0.0,1.0]
	ExactMoments = [0.0,1.0]
	CISStates = [9.3747, 11.5186, 11.8079] # 9.37 is a homo->lumo transition. 
	CISMoments = [0.7055, 0.0028, 0.0141]
	CISDStates = [9.1343, 11.1658, 11.5269]
	CISDMoments = [0.7055, 0.0028, 0.0141]
	ADCStates = [0.0,1.0]
	ADCMoments = [0.0,1.0]

if (Params.MoleculeName == 'H2O'):
	ExactStates = [0.0000, 9.0170, 11.1330, 11.2523, 13.7250, 16.2885, 19.4117, 21.0287]
	ExactMoments = [0.0000, 0.29394, 0.00004, 0.66569, 0.61685, 1.09874, 0.77995, 0.00000]
	CISStates = [0.0000, 8.8697, 9.9023, 10.5985, 11.0911, 11.7275, 12.0852, 12.3298, 14.2635, 14.4559, 16.0627, 16.3727, 19.5305, 20.9039, 23.1716, 24.4252, 25.0257]
	CISMoments = [0.0000, 0.0000, 0.0194, 0.0000, 0.0000, 0.0000, 0.1273, 0.0000, 0.1219, 0.0000, 0.0000, 0.5403, 0.3244, 0.0000, 0.0000, 0.0115, 0.0369]
	CISDStates = [8.8455, 11.0727, 11.0239, 13.6619, 19.3999, 25.0253, 25.7748]
	CISDMoments = [0.0000, 0.0000, 0.0000, 0.1219, 0.0000, 0.0115, 0.0369]
	ADCStates = [0.0]
	ADCMoments = [0.0]

def ExactCurve(Energy): 
	return numpy.sum( [pow(ExactMoments[i],1.0)*Lorentzian(Energy,0.0005,ExactStates[i]/EvPerAu) for i in rlen(ExactStates)] )
def CISCurve(Energy): 
	return numpy.sum( [pow(CISMoments[i],1.0)*Lorentzian(Energy,0.0005,CISStates[i]/EvPerAu) for i in rlen(CISStates)] )
def CISDCurve(Energy): 
	return numpy.sum( [pow(CISDMoments[i],1.0)*Lorentzian(Energy,0.0005,CISDStates[i]/EvPerAu) for i in rlen(CISDStates)] )
def ADCCurve(Energy): 
	return numpy.sum( [pow(ADCMoments[i],1.0)*Lorentzian(Energy,0.0005,ADCStates[i]/EvPerAu) for i in rlen(ADCStates)] )

def	GeneralizedFFT(Input,b=1.0,a=0.0): 
	dim = len(Input)
	t1 = numpy.fromfunction(lambda X,Y: 2.0*pi*complex(0.0,1.0)*b*(X-1.0)*(Y-1.0)/dim,shape=[dim,dim])
	t1 = numpy.exp(t1)
	return (pow(dim,-(1.0-a)/2.0)*numpy.tensordot(Input,t1,axes=([0],[0])))

# Pass a numpy ndarray 
# get a list of positions which "Stand out."
def LocalMaxima(AList, Tol = 0.2 ,NMax=10): 
	tore = []
	Z = abs(AList)
	tmean = numpy.mean(abs(Z))
	tstdev = numpy.std(abs(Z))
	a = numpy.diff(Z)
	FDiffDer = numpy.insert(a,0,a[0])
	from scipy import interpolate, optimize
	dZ = scipy.interpolate.interp1d(numpy.arange(len(FDiffDer)), FDiffDer, kind='linear')
	print "Finding LocalMaxima of List ... "
	print "Mean: ", tmean, " stdev: ", tstdev

	# find the 10 most important maxima
	N0 = 0
	for N in range(NMax): 
		for Nz in range(N0,len(Z)): 
			OuterBreak = False
			if (abs(Z[Nz]) > tmean+Tol*tstdev):
				# find interval where the derivative changes sign
				for Nz2 in range(Nz,len(Z)-1): 
					if ( numpy.sign(dZ(Nz))*numpy.sign(dZ(Nz2)) < 0 ): 
						Xm = scipy.optimize.brentq(dZ,Nz,Nz2)
						if (Z[int(Xm)] > tmean+Tol*tstdev): 
							tore.append(Xm)
						# N0 begins when it goes back under tolerance. 
						for Nz3 in range(int(Xm),len(Z)): 
							if (abs(Z[Nz3]) < tmean+Tol*tstdev): 
								N0 = int(Xm)
								break 
						N0 = min(N0,len(Z))
						OuterBreak = True
				if (not OuterBreak): 
					OuterBreak = True
			if (OuterBreak): 
				break 
	return list(set(tuple(map(lambda X: round(X,3),tore))))

def SpectralAnalysis(Arg_Data,Arg_SampleInterval,DesiredZoom = 1.0/30.0,Title = "Spec",DesiredMaximum=None,Smoothing = False, Flouresce = False): #add logic here to take into account the density of states from imaginary time propagation
	Data = Arg_Data.copy()	
	# By default, remove all the zero frequency information. 
	Arg_Data -= Arg_Data.mean()
	DataPts = len(Data)
	SampleInterval = Arg_SampleInterval
	if (pow(DataPts,2.0) > 30*1024*1024): 
		print "Spectral analysis of very large dataset, downsampling"
		KeepEvery = int(DataPts/5500.0)
		print " by factor of ", KeepEvery
		SampleInterval = Arg_SampleInterval*KeepEvery
		Data = Arg_Data[0::KeepEvery].copy()
		DataPts = len(Data) 

	Zoom = DesiredZoom
	if (DesiredMaximum != None): 
		Zoom = SampleInterval*DesiredMaximum*(1.0/pi)
		print "Assigning Zoom", Zoom		

	Freqs = pi*(2.0/SampleInterval)*Zoom*(numpy.arange(DataPts/2.0))/(DataPts)		
#	Freqs = pi*(1.0/SampleInterval)*Zoom*(numpy.arange(DataPts/2.0))/(DataPts)
	CplxStrengths = GeneralizedFFT(Data,-1.0*Zoom)
	print "Generalized FFT result: ", numpy.sum(CplxStrengths*CplxStrengths.conj())
	
	MakeSimplePlot(CplxStrengths.real,tit=Title+"RealStrengths")
	MakeSimplePlot(CplxStrengths.imag,tit=Title+"ImStrengths")	
	CplxStrengths = CplxStrengths[:len(Freqs)] # I bet the problem is the negative frequencies... 
	import scipy.special
	# Damp out the low frequency information. 
	DampLow= numpy.vectorize(lambda X: scipy.special.erf(X/int(0.07*len(CplxStrengths))))
	Damping = DampLow(numpy.arange(len(CplxStrengths)))
	CplxStrengths = CplxStrengths*Damping
	
	numpy.savetxt('./Output'+Params.MoleculeName+'/FFTStrengths',CplxStrengths,fmt='%.18e')	
	Strengths = CplxStrengths.real
	
	# Annotate the positions of Maxima. 
	SpecialPts = LocalMaxima(Strengths)
	SpecialCoords = [[N,Strengths[N]] for N in SpecialPts]
	
	import matplotlib
	import matplotlib.pyplot as plt
	import matplotlib.font_manager as fnt
	matplotlib.rcParams['legend.fancybox'] = True	
	PlotFont = {'fontname':'Helvetica','fontsize':18,'weight':'bold'}
	LegendFont = fnt.FontProperties(family='Helvetica',size='17',weight='bold')

	fig = plt.figure()
	ax = fig.add_subplot(111)		
	l1 = plt.plot( Freqs , Strengths ,'k')
	for C in SpecialCoords: 
		xya = (Freqs[C[0]],C[1])
		if (C[1] > 0.2*max(Strengths)):
			ax.annotate("{0:.3f}".format(EvPerAu*Freqs[C[0]]), xy=xya, arrowprops=dict(facecolor='black', shrink=0.2))
	plt.setp(l1,linewidth=2, color='r')
	plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
	plt.ylabel('Strength',fontsize = Params.LabelFontSize)
	plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"TransformedSignal")
	plt.clf()	

	if True:  #(Title == "ZoomedSpectrum"): 
		# Add CIS, CIS(D) and Exact stick spectra
		# Normalized to the exact spectrum . 
		StrCIS = (numpy.vectorize(CISCurve))(Freqs)
		StrCISD = (numpy.vectorize(CISDCurve))(Freqs)
		StrADC = (numpy.vectorize(ADCCurve))(Freqs)
		StrExact = (numpy.vectorize(ExactCurve))(Freqs)
		MaxExact = StrExact.max()
		if (StrCIS.max() != 0.0): 
			StrCIS *= MaxExact/StrCIS.max()
		if (StrCISD.max() != 0.0): 
			StrCISD *= MaxExact/StrCISD.max()
		Strengths *= MaxExact/abs(Strengths).max()
		if (StrADC.max() != 0.0): 
			StrADC *= MaxExact/StrADC.max()

		fig = plt.figure()
		ax = fig.add_subplot(111)		
		l1,l2 = plt.plot( Freqs , StrCIS ,'k',Freqs , StrExact ,'k-')
		plt.setp(l1,linewidth=2, color='g')
		plt.setp(l2,linewidth=2, color='b')
		plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
		plt.legend(["CIS","Exact"],loc=2,prop={'size':Params.LegendFontSize})
		plt.ylabel('Strength',fontsize = Params.LabelFontSize)
		plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"CisExact")
		plt.clf()		
		
		fig = plt.figure()
		ax = fig.add_subplot(111)		
		l2,l4,l5 = plt.plot( Freqs , StrCIS ,'k', Freqs , StrCISD ,'k:', Freqs , StrExact ,'k-')
		plt.setp(l5,linewidth=2, color='b')
		plt.setp(l4,linewidth=1)
		plt.setp(l2,linewidth=2, color='g')
		plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
		plt.legend(["CIS","CIS(D)","Exact"],loc=2,prop={'size':Params.LegendFontSize})
		plt.ylabel('Strength',fontsize = Params.LabelFontSize)
		plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"CisExactAndCisD")
		plt.clf()			

		fig = plt.figure()
		ax = fig.add_subplot(111)		
		l1,l2,l3,l4 = plt.plot( Freqs , Strengths ,'k', Freqs , StrCIS ,'k', Freqs , StrCISD ,'k:', Freqs , StrExact ,'k-')
		plt.setp(l5,linewidth=2, color='b')
		plt.setp(l4,linewidth=1)
		plt.setp(l2,linewidth=2, color='g')
		plt.setp(l1,linewidth=2, color='r')
		plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
		plt.legend(["2-TCL","CIS","CIS(D)","Exact"],loc=2,prop={'size':Params.LegendFontSize})
		plt.ylabel('Strength',fontsize = Params.LabelFontSize)
		plt.savefig("./Figures"+Params.MoleculeName+"/"+Title)
		plt.clf()		

		if False: 
				# Get a close-up of the region between .6 and .8 Eh
				# Get a close-up of the region between .6 and .8 Eh
				fig = plt.figure()
				ax = fig.add_subplot(111)		
				l1 = plt.plot( Freqs , Strengths ,'k')
				minfreq = 0.82
				maxfreq = 0.85
				plt.xlim(minfreq,maxfreq)
				#pi*(1.0/SampleInterval)*Zoom*(numpy.arange(DataPts/2.0))/(DataPts)
				xminpos = int(minfreq*SampleInterval*DataPts/(pi*Zoom))
				xmaxpos = int(maxfreq*SampleInterval*DataPts/(pi*Zoom))
				plt.ylim(0.0,Strengths[xminpos:xmaxpos] .max()*1.1)
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
				plt.legend(["2-TCL"],loc=2,prop={'size':Params.LegendFontSize})
				plt.ylabel('Strength',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"Zoomedp82top85")
				plt.clf()		
						
				fig = plt.figure()
				ax = fig.add_subplot(111)		
				l1 = plt.plot( Freqs , Strengths ,'k')
				minfreq = 0.73
				maxfreq = 0.77
				plt.xlim(minfreq,maxfreq)
				#pi*(1.0/SampleInterval)*Zoom*(numpy.arange(DataPts/2.0))/(DataPts)
				xminpos = int(minfreq*SampleInterval*DataPts/(pi*Zoom))
				xmaxpos = int(maxfreq*SampleInterval*DataPts/(pi*Zoom))
				plt.ylim(0.0,Strengths[xminpos:xmaxpos].max()*1.1)
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
				plt.legend(["2-TCL"],loc=2,prop={'size':Params.LegendFontSize})
				plt.ylabel('Strength',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"Zoomedp73top77")
				plt.clf()		

				fig = plt.figure()
				ax = fig.add_subplot(111)		
				print len(Strengths)
				l1 = plt.plot(Freqs ,Smooth(Strengths,2.0/1000.0) ,'k')
				minfreq = 0.70
				maxfreq = 0.80
				plt.xlim(minfreq,maxfreq)
				#pi*(1.0/SampleInterval)*Zoom*(numpy.arange(DataPts/2.0))/(DataPts)
				xminpos = int(minfreq*SampleInterval*DataPts/(pi*Zoom))
				xmaxpos = int(maxfreq*SampleInterval*DataPts/(pi*Zoom))
				plt.ylim(0.0,Strengths[xminpos:xmaxpos] .max()*1.1)
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
				plt.legend(["2-TCL"],loc=2,prop={'size':Params.LegendFontSize})
				plt.ylabel('Strength',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"ZSoomedp70top80")
				plt.clf()	

				fig = plt.figure()
				ax = fig.add_subplot(111)	
				l1 = plt.plot( Freqs , Smooth(Strengths,2.0/1000.0) ,'k')
				minfreq = 0.82
				maxfreq = 0.85
				plt.xlim(minfreq,maxfreq)
				#pi*(1.0/SampleInterval)*Zoom*(numpy.arange(DataPts/2.0))/(DataPts)
				xminpos = int(minfreq*SampleInterval*DataPts/(pi*Zoom))
				xmaxpos = int(maxfreq*SampleInterval*DataPts/(pi*Zoom))
				plt.ylim(0.0,Strengths[xminpos:xmaxpos] .max()*1.1)
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
				plt.legend(["2-TCL"],loc=2,prop={'size':Params.LegendFontSize})
				plt.ylabel('Strength',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"ZSoomedp82top85")
				plt.clf()		
						
				fig = plt.figure()
				ax = fig.add_subplot(111)		
				l1 = plt.plot( Freqs , Smooth(Strengths,2.0/1000.0) ,'k')
				minfreq = 0.71
				maxfreq = 0.76
				plt.xlim(minfreq,maxfreq)
				#pi*(1.0/SampleInterval)*Zoom*(numpy.arange(DataPts/2.0))/(DataPts)
				xminpos = int(minfreq*SampleInterval*DataPts/(pi*Zoom))
				xmaxpos = int(maxfreq*SampleInterval*DataPts/(pi*Zoom))
				plt.ylim(0.0,Strengths[xminpos:xmaxpos].max()*1.1)
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
				plt.legend(["2-TCL"],loc=2,prop={'size':Params.LegendFontSize})
				plt.ylabel('Strength',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"ZSoomedp71top76")
				plt.clf()		
		
		# ------------------------------------------
		# Hope that smoothing will clean up the ringing. 
		# ------------------------------------------

		if (Smoothing): 
			for DR in [5,6,7,8,9,10,11,12,13,14,15,20,30,40,50,100]: 
	
		# Don't Convolve the Stick Spectra with Gaussians 
				drStrExact = StrExact #Smooth(StrExact,(float(DR)/1000.0))
				MaxExact = drStrExact.max()
			
				DeRung = Smooth(numpy.abs(CplxStrengths),(float(DR)/1000.0))
				DeRung *= MaxExact/abs(DeRung).max()

				numpy.savetxt('./Output'+Params.MoleculeName+'/Freqs',Freqs,fmt='%.18e')	
				numpy.savetxt('./Output'+Params.MoleculeName+'/Derung'+str(DR),DeRung,fmt='%.18e')	
				
				drStrCIS = StrCIS #Smooth(StrCIS,(float(DR)/1000.0))
				if ( abs(drStrCIS).max() != 0.0): 
					drStrCIS *= MaxExact/abs(drStrCIS).max()
				drStrCISD = StrCISD #Smooth(StrCISD,(float(DR)/1000.0))
				if (abs(drStrCISD).max() != 0.0): 
					drStrCISD *= MaxExact/abs(drStrCISD).max()

				SpecialPts = LocalMaxima(DeRung.real)
				SpecialCoords = [[N,(DeRung.real)[N]] for N in SpecialPts]
				fig = plt.figure()
				ax = fig.add_subplot(111)		
				l1,l2,l4,l5 = plt.plot( Freqs , abs(DeRung) ,'k', Freqs , drStrCIS ,'k', Freqs , drStrCISD ,'k:', Freqs , drStrExact ,'k-')
				for C in SpecialCoords: 
					if (C[1] > 0.2*max(Strengths)):
						xya = (Freqs[C[0]],C[1])
						ax.annotate("{0:.3f}".format(EvPerAu*Freqs[C[0]]), xy=xya, arrowprops=dict(facecolor='black', shrink=0.2))
				plt.setp(l5,linewidth=2, color='b')
				plt.setp(l4,linewidth=2)
				plt.setp(l2,linewidth=2, color='g')
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
				plt.legend(["2-TCL","CIS","CIS(D)","Exact"],loc=2,prop={'size':Params.LegendFontSize})
				plt.ylabel('Strength',fontsize = Params.LabelFontSize)
				plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"_Derung"+str(DR)+"cent")
				plt.clf()
						
				fig = plt.figure()
				ax = fig.add_subplot(111)		
				l1 = plt.plot( Freqs , abs(DeRung) ,'k')
				for C in SpecialCoords: 
					if (C[1] > 0.2*max(Strengths)):
						xya = (Freqs[C[0]],C[1])
						ax.annotate("{0:.3f}".format(EvPerAu*Freqs[C[0]]), xy=xya, arrowprops=dict(facecolor='black', shrink=0.2))
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
				plt.ylabel('Strength',fontsize = Params.LabelFontSize)
				plt.legend(["2-TCL"],loc=2,prop={'size':Params.LegendFontSize})
				plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"JustTcl"+"_Derung"+str(DR)+"cent")
				plt.clf()

				# This is a hack
				if (Params.MoleculeName == 'C2F2' or Params.MoleculeName == 'C2F2LB' or Params.MoleculeName == 'C2F21' or Params.MoleculeName == 'C2F22'):
					from scipy import interpolate
					Experimental = numpy.loadtxt('./IntegralsC2F2/c2f2.csv',delimiter = ',')
					#Experimental2 = numpy.loadtxt('./IntegralsC2F2/c2f26to10.csv',delimiter = ',')

					#ExpInt = interpolate.interp1d(Experimental[:,0], Experimental[:,1], kind = 'linear')
					#ExpInt2 = ExpInt = interpolate.interp1d(Experimental2[:,0], Experimental2[:,1], kind = 'linear')
					#Exp1Max = Experimental[:,0].max() 
					#Exp1Min = Experimental[:,0].min()
					#Exp2Max = Experimental2[:,0].max()
					#dx = 0.00001
					#xInt1 = numpy.arange(Exp1Min, Exp1Max, dx)
					#xInt2 = numpy.arange(Exp1Max, Exp2Max, dx)
					#x = numpy.concatenate((xInt1, xInt2))
					#tmp = numpy.concatenate((ExpInt(xInt1),ExpInt2(xInt2)) )
					#Experimental = numpy.vstack((x,tmp)).T
					ExpMax = Experimental[:,1].max()

					print "Experimental Range: ", Experimental[:,0].min(), " ",Experimental[:,0].max() 
				#	ExpFreqs = (Experimental[:,0]+1.89)/EvPerAu
					ExpFreqs = (Experimental[:,0])/EvPerAu + 0.035
					Exp = Experimental[:,1]*(abs(DeRung).max()/Experimental[:,1].max())	
					import matplotlib
					import matplotlib.pyplot as plt
					import matplotlib.font_manager as fnt
					matplotlib.rcParams['legend.fancybox'] = True	
					PlotFont = {'fontname':'Helvetica','fontsize':18,'weight':'bold'}
					LegendFont = fnt.FontProperties(family='Helvetica',size='17',weight='bold')
					if(DR==12):
						numpy.savetxt('./Output'+Params.MoleculeName+'/Exp',Exp,fmt='%.18e')
						numpy.savetxt('./Output'+Params.MoleculeName+'/ExpFreqs',ExpFreqs,fmt='%.18e')

					fig = plt.figure()
					ax = fig.add_subplot(111)		
					l1 = plt.plot( Freqs , abs(DeRung), 'k')
					plt.setp(l1,linewidth=2, color='b')
					plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
					plt.ylabel('Strength',fontsize = Params.LabelFontSize)
					plt.legend(["Calculated"])
					plt.xlim(8.7/EvPerAu,10.0/EvPerAu)		
					plt.savefig("./Figures"+Params.MoleculeName+"/C2F2_Derung"+str(DR)+"cent")
					plt.clf()

					fig = plt.figure()
					ax = fig.add_subplot(111)		
					l1,l2 = plt.plot( Freqs , abs(DeRung), 'k', ExpFreqs, Exp,'k:')
					plt.setp(l1,linewidth=2, color='b')
					plt.setp(l2,linewidth=2, color='r')		
					plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
					plt.ylabel('Strength',fontsize = Params.LabelFontSize)
					plt.legend(["Calculated", "Experimental"])
					plt.savefig("./Figures"+Params.MoleculeName+"/C2F2VsExperiment"+"_Derung"+str(DR)+"cent")
					plt.clf()

					fig = plt.figure()
					ax = fig.add_subplot(111)		
					l1,l2 = plt.plot( Freqs , abs(DeRung), 'k', ExpFreqs, Exp,'k:')
					plt.setp(l1,linewidth=2, color='b')
					plt.setp(l2,linewidth=2, color='r')		
					plt.xlim(0.2,0.4)
					plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
					plt.ylabel('Strength',fontsize = Params.LabelFontSize)
					plt.legend(["Calculated", "Experimental"],loc=2)
					plt.savefig("./Figures"+Params.MoleculeName+"/C2F2VsExperiment04"+"_Derung"+str(DR)+"cent")
					plt.clf()

					# Log Version in the region of interest  
					print numpy.log(numpy.abs(DeRung)+2.0)
					logged = numpy.log(numpy.abs(DeRung)+2.0) - numpy.log(numpy.abs(DeRung)+2.0).min()
					Exp = Experimental[:,1]*(abs(logged).max()/Experimental[:,1].max())	
					fig = plt.figure()
					ax = fig.add_subplot(111)		
					l1,l2 = plt.plot( Freqs , logged, 'k', ExpFreqs, Exp,'k:')
					plt.setp(l1,linewidth=2, color='b')
					plt.setp(l2,linewidth=2, color='r')
					plt.xlim(0.0,0.4)
				#	plt.xlim(8.7/EvPerAu,10.0/EvPerAu)		
					plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
					plt.ylabel('Strength',fontsize = Params.LabelFontSize)
					plt.legend(["Calculated", "Experimental"])
					plt.savefig("./Figures"+Params.MoleculeName+"/LoggedC2F2VsExperiment"+"_Derung"+str(DR)+"cent")
					plt.clf()


				fig = plt.figure()
				ax = fig.add_subplot(111)		
				l1,l2 = plt.plot( Freqs , abs(DeRung) ,'k', Freqs , abs(drStrExact) ,'k')
				for C in SpecialCoords: 
					if (C[1] > 0.2*max(Strengths)):
						xya = (Freqs[C[0]],C[1])
						ax.annotate("{0:.3f}".format(EvPerAu*Freqs[C[0]]), xy=xya, arrowprops=dict(facecolor='black', shrink=0.2))
				plt.ylim(0.0,abs(drStrExact).max()*1.15)
				plt.setp(l1,linewidth=2, color='r')
				plt.setp(l2,linewidth=2, color='b')
				plt.xlabel('Frequency(au)',fontsize = Params.LabelFontSize)
				plt.ylabel('Strength',fontsize = Params.LabelFontSize)
				plt.legend(["2-TCL","Exact"],loc=2,prop={'size':Params.LegendFontSize})
				plt.savefig("./Figures"+Params.MoleculeName+"/"+Title+"JustTCLExact"+"_Derung"+str(DR)+"cent")
				plt.clf()	
	return 

def Smooth(ToSmooth,GaussianWidth=0.05): 
	BlurWidth = int(GaussianWidth*len(ToSmooth)) 
	print BlurWidth
	if(BlurWidth*numpy.size(ToSmooth)/2 < 1.4 or BlurWidth == 2):
		return ToSmooth
	else:
		Cnv = numpy.blackman(BlurWidth)/(numpy.hanning(BlurWidth).sum())
		return numpy.convolve(ToSmooth,Cnv,mode='same')

