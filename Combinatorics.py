# Delivers compositions of integers. 
# and signed permutation groups up to 8. 

# standard crap
import os, sys, time, math, re, random
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, log, pi, sqrt
import copy

# Gives an iterator over all Signed permutations on a list. 
def AllPerms(str):
    if len(str) <=1:
        yield (str,1)
    else:
        for perm in AllPerms(str[1:]):
            for i in range(len(perm[0])+1):
                yield (perm[0][:i] + str[0:1] + perm[0][i:],pow(-1,i)*perm[1])
				
def GiveAllPerms(str):
	return [X for X in AllPerms(str)]

#Composes permutations. 
def ComposePermutations(List1,List2): 
	if (len(List1) != len(List2)): 
		print "Compose requires the same lengths"
	return ([List2[0][I] for I in List1[0]],List1[1]*List2[1])

def InterleaveMissing(l1,l2): 
	for l in l2: 
		if (not l in l1): 
			l1.insert(l,l)
	return l1

def MuliplyPermutationLists(List1,List2): 
	tore = []
	for I in List1: 
		for J in List2: 
			tore.append(ComposePermutations(I,J))
	return tore

def binomial(n, k):
"""binomial(n, k): return the binomial coefficient (n k)."""
	assert n>0 and isinstance(n, (int, long)) and isinstance(k, (int,long))
	if k < 0 or k > n:
		return 0
	if k == 0 or k == n:
		return 1
	return (factorial(n)/(factorial(k) * factorial(n-k)))


class AntisymmetricTensorSymmetry: 
	def __init__(self,rank,hs = False,firsthalf = True, secondhalf = True): 
		self.rank = rank
		self.AllIndices = range(rank)
		self.AntiSymmetricSets = []
		if (firsthalf and secondhalf): 
			self.AntiSymmetricSets = [range(rank/2),range(rank/2,rank)]
		elif firsthalf: 
			self.AntiSymmetricSets = [range(rank/2)]
		elif secondhalf: 
			self.AntiSymmetricSets = [range(rank/2,rank)] 
		self.HermitianSymmetry = hs 
		self.SelfDict = dict()
		return 
	def ExpandSelf(self):
		# these should be mutually exclusive. 
		GroupsList0 = map(GiveAllPerms, self.AntiSymmetricSets)
		GroupsList1 = [ map(lambda Z: InterleaveMissing(Z,self.AllIndices),G) for G in GroupsList0]
		# Now multiply them. 
		TotalGroup = GroupsList1.pop()
		while (len(GroupsList1) > 0): 
			TotalGroup = MuliplyPermutationLists(TotalGroup,GroupsList1.pop())
		if (self.HermitianSymmetry): 
			tmp = range(rank/2,rank)
			tmp.extend(range(rank/2))
			TotalGroup = MuliplyPermutationLists(TotalGroup,[(tmp,1)])
		print TotalGroup
		# Populate selfdict. 
		for XX in TotalGroup:
			self.SelfDict[XX[0]] = self.SelfDict[XX[1]]
		return 
# Returns 1 or -1 if there is a permutation which can achieve order
# otherwise False
	def SignBetween(self,order): 
		if (order in self.SelfDict): 
			return self.SelfDict[order]
		else : 
			return False

#class Combinatorica: 
#	def __init__(self): 
#		self.FourComp = self.FourCompofN()
#		self.NPerm = self.PermsOfN()
#		return
#	def FourCompofN(self):
#		tore = []
#		return tore
#	def PermsOfN(self) 
#		tore = []
#		return tore
#
#Psym = Combinatorica()