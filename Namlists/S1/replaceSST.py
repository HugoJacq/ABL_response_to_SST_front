"""
	replace the SST of a MNH file with prescribed SST, canals case
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from function_replaceSST import *
import os

## INPUT ===================================
# MODEL ---------------------------------- 
nhalo=1
# TYPE OF SST ----------------------------
DIR = 'X' # direction of change of SST
THT0=296.55 # K 295.5 296.55
deltaTHT=1.5
x1=5000
x2=25000
L1=1000
L2=1000
# Name of files --------------------------
path = './'
name_in = 'INIT_CANAL'
name_out = 'INIT_CANAL_SST'
SAVING = True
#===========================================
print('Currently replacing SST')

dsini = xr.open_dataset('INIT_CANAL.nc')
X = dsini.ni.values
Y = dsini.nj.values
res = X[1] - X[0]

# First let us have a look at the SST prescribed
print('ni is :')
print(X)
print('nj is :')
print(Y)
if DIR=='X':
	dim=X
else:
	dim=Y
SST1D = tanhSST(dim,THT0,x1,x2,deltaTHT,L1,L2)
		
# 1D SST plot
fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100) 
ax.plot(dim/1000,SST1D,color='k')
ax.set_xlabel(DIR+' (km)')
ax.set_ylabel('SST (°C)')
plt.show()

# Then if its ok, we can build the SST field 
SST = buildSST(SST1D,len(X),len(Y))
fig, ax = plt.subplots(1,1,figsize = (10,2),constrained_layout=True,dpi=100) 
s = ax.pcolormesh(X/1000,Y/1000,SST-273,cmap='rainbow')
plt.colorbar(s,ax=ax,location='bottom')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_title('SST (°C)')
ax.set_aspect('equal')
plt.show()

# Replacing the SST from the initial file 
ds = xr.open_dataset(path+name_in+'.nc')
print('SST old shape :',ds.SST.shape)
print('SST new shape :',SST.shape)
SSTold = ds.SST[:,:]
attrs = ds.SST.attrs 
ds['SST'] = (['nj', 'ni'], SST)
ds.SST.attrs = attrs
fig, ax = plt.subplots(2,1,figsize = (10,4),constrained_layout=True,dpi=100) 
vmax=np.amax(SST)
vmin=np.amin(SST)
ax[0].pcolormesh(X/1000,Y/1000,SSTold,cmap='rainbow',vmin=vmin,vmax=vmax)
ax[1].pcolormesh(X/1000,Y/1000,SST,cmap='rainbow',vmin=vmin,vmax=vmax)
ax[1].set_xlabel('X (km)')
ax[1].set_ylabel('Y (km)')
ax[0].set_title(r'$\theta_0$='+str(THT0)+r'K $\Delta\theta$='+str(deltaTHT)+r' x1='+str(x1/1000)+'km x2='+str(x2/1000)+'km L1='+str(L1/1000)+'km L2='+str(L2/1000)+'km')
plt.colorbar(s,ax=ax,location='bottom')
fig.savefig(path+'check_SST.png')
plt.show()
if SAVING:
	print('saving ...')
	ds.to_netcdf(path+name_out+'.nc')
	os.system("cp "+path+name_in+".des "+path+name_out+".des")
	print('done !')



