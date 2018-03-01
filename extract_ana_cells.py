# Valentin Heinitz, vheinitz@googlemail.com, 2018.01.01
# L E A R N I N G   K E R A S   WITH
# https://www.youtube.com/playlist?list=PLtPJ9lKvJ4oiz9aaL_xcZd-x0qd8G0VN_
# Using ANA-HEp2, ANCA, dDNA data sets 
#
# Extract cells using external tool objex 

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
import os
from subprocess import call
import sys




classes=['cent' ,  'cyt' , 'env', 'hom', 'nuc', 'nucdot', 'spe' ]
dirs=['test','train']
basedir='C:/tmp/ana/'
outdir='C:/tmp/ana/rects/'
for cl in classes:
	for d in dirs:
		cnt=0
		for imgfn in os.listdir(basedir+cl+'/'+d):
			objex = ["C:/Development/build/objex.exe", "--infile="+basedir+cl+'/'+d+'/'+imgfn, "--outdir="+outdir+d+'/'+cl, '--rects=256x256']
			call(objex)
			print(objex) 
			sys.stdout.flush()
			#cnt = cnt +1
			#if cnt > 3:
			#	break

