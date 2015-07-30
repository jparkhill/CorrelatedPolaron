import os, numpy
from math import pow 

def GoodMkdir(newdir):
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        #print "_mkdir %s" % repr(newdir)
        if tail:
            os.mkdir(newdir)

def ListComplement(list1, list2): 
	return [I for I in list1 if (not I in list2)]

def rlen(Alista): 
	return range(len(Alista))

#Why would we not use fill? 
#def AssignTensorValue(aTensor,valu=0.0): 
#	iter = numpy.ndindex(aTensor.shape)
#	for i in iter: 
#		aTensor[i] = valu

def SummarizeNDarray(arg): 
	print "size: ", arg.size
	print "flags: ", arg.flags
	print "Size in memory: ", arg.nbytes

def StupidDot(a,b,dices): 
	i1 = numpy.ndindex(a.shape)
	i2 = numpy.ndindex(b.shape)
	tore = 0.0
	torea = 0.0
	toreb = 0.0
	for i in i1: 
		torea += abs(a[i])
		for j in i2 : 
			toreb += abs(b[j])
			if a[i] == 0.0 or b[j] == 0.0: 
				continue 
			matches = True
			for k in range(len(dices[0])): 
				if i[dices[0][k]] != j[dices[1][k]]:
					matches = False 
					break 
			if (matches) : 
				print i, j, a[i]*b[j]
				tore += abs(a[i]*b[j])
	return (tore,torea,toreb)

class Laser: 
	def __init__(self, Amp ,Freq = 0.01,TOn = 1.0,Width = 10.0,Pol = [1.0,0.0,0.0]):
		self.Amp = Amp
		self.Width = Width
		self.Freq = Freq
		self.TOn = TOn
		self.Pol = Pol
		if (TOn < Width*2): 
			print "Warning. Field is coming on abruptly." 
		return 
	def NormNow(self,Time):
		from math import exp,cos,pow,pi,sqrt
		return exp(-pow(Time-self.TOn,2.0)/(2*pow(self.Width,2.0)))*cos(2*self.Freq*Time)*self.Amp*sqrt(self.Pol[0]*self.Pol[0]+self.Pol[1]*self.Pol[1]+self.Pol[2]*self.Pol[2])
	def __call__(self,Time):
		from math import exp,cos,pow,pi,sqrt
		return [exp(-pow(Time-self.TOn,2.0)/(2*pow(self.Width,2.0)))*cos(2*self.Freq*Time)*X*self.Amp for X in self.Pol]
	def Polarization(self):
		return self.Pol
	def TOff(self): 
		return self.TOn+0.5*self.Width
	def IsNonZero(self,ATime): 
		if (abs(ATime - self.TOn)/self.Width > 300) :
			return False
		else :
			return True
	def AsString(self):
		return "Stren: "+str(self.Amp)+" Wid: "+str(self.Width)+" Ton: "+str(self.TOn) #+str(self.Pol)

