# To be used with module_CS.py 
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt

from module_cst import *
from module_tools import *

def Integ_min(Z,Y,X,var,gamma,z1,zi=10000):
	"""This function is computing the mean of 'var' of all the previous level
		at each level. If z1=0 and if zi is present then above zi the values are the 
		same at zi. 
		
			z1=0 (RVT or SVT1)	
				var_min[z] = gamma * mean(var[z1:z]) at every z<zi
					   = var_min[zi] above zi
			z1 not 0 (SVT2 or SVT3)
				var_min[z] = gamma * mean(var[z:z1]) at every z
				
		- Z are the levels (1D)
		- var is a mean field
		- gamma is scalar
		- z1 : Altitude to start the vertical average at
		- zi is same shape as var
		- OUTPUT : var_min of same dimensions as 'var'
		
		CAN BE SIMPLIFIED
	"""	
	if var.ndim==4:	# here the mean is time and Y, TO BE MODIFIED WHEN SVT AVAILABLE IN CAS06W
		mean = var.mean(axis=(0,2))
		ABLHm = zi.mean(axis=(0,2))
		var_min = xr.zeros_like(mean)
		Z = Z.expand_dims(dim={'ni':var.coords['ni']},axis=1)
		if type(z1)==int:
			if z1==0:
				for z in range(len(Z)-1):
					var_min[z+1] = xr.where( Z[z+1,:] < ABLHm[z+1,:], gamma * mean[:z+1,:].mean(axis=0), var_min[z,:])
		else:
			if len(z1)==len(X):
				for x in range(len(X)):
					indz1 = nearest(Z.values,z1[x].values) + 2
					for z in range(indz1,0,-1):
						var_min[z,x] = gamma * mean[z:indz1,x].mean(axis=0)	
				
	elif var.ndim==3: # here the mean is X and Y, and z1 is scalar
		mean = var.mean(axis=(1,2))
		if not type(zi)==type(1): #len(zi.shape)>0:
			ABLHm = zi.mean(axis=(1,2)) # zi is array
		else:
			ABLHm = np.ones(Z.shape)*zi 	# zi is scalar
		var_min = xr.zeros_like(mean)
		if z1==0:
			for z in range(len(Z)-1):
				var_min[z+1] = xr.where( Z[z+1] < ABLHm[z+1], gamma * mean[:z+1].mean(axis=0), var_min[z])
		else:
			indz1 = nearest(Z.values,z1) + 2
			for z in range(indz1-2,0,-1):
				var_min[z] = gamma * mean[z:indz1].mean(axis=0)	
			var_min = xr.where(Z<Z[indz1-1],var_min,var_min[indz1-2])
			var_min[0] = var_min[1]
	var_min = Complete_dim_like([var_min],var)
	return var_min	
	
def Integ_min3(Z,var,dim,gamma):	
	"""

	var is 3D field of a variable
	NEEDS TO BE DOCUMENTED
	
	xarray version of Integ_min, using dimension to locate what axis to do the mean

	if z1 = 0
		std_min = gamma * 1 (z-z1) integrale(from z1 to z)( agg(var(h))dh )
	
	if z1 != 0
		std_min = gamma * 1 (z1-z) integrale(from z to z1)( agg(var(h))dh )

		with :
			agg an aggregate operation.
			default behavior is to use agg=mean(var), if a tracer is to be used then agg=std(var)
	
	How it is meant to be used :
		if var == RVT or a tracer emitted at surface,
			agg=std and z1=0
		if var == tracer emitted above the ABL height,
			agg=std and z1=Z[ argmax( mean(var)(z)) + 2]
		if var == turbulent intensity (ITURB) or turbulent energy (E)
			agg=std and z1=0
			
	INPUTS:
		TBD	
		XARRAY version of integ_min	
	"""	
	dimh = dim[:]
	for dimx in ['ni','nj']:
		if dimx not in dim:
			dimh.append(dimx)
	mean = var.mean(dim=dimh) # over horizontal dim also (tracer is emitted the same horizontally)
	std = var.std(dim=dim)
	
	integ = xr.ones_like(std) * 999 
	if var.name=='SVT002' or var.name=='SVT003' or var.name=='SVCS001' or var.name=='SVCS002': # top ABL tracer
		indz0,indzmax = 0, mean.argmax(dim='level').values
		for indz in range(indz0,indzmax,1):
			integ[dict(level=indz)] = std.isel(level=slice(indz,indzmax)).integrate('level') / (Z[indzmax].values-Z[indz].values)
	elif (var.name=='RVT' 
			or var.name=='SVT001' 
			or var.name=='SVT004' 
			or var.name=='SVCS000' 
			or var.name=='SVCS003'): # surface tracer or integration from surface
		indz0,indzmax = 0,len(Z)-1
		for indz in range(indz0,indzmax,1):
			integ[dict(level=indz)] = std.isel(level=slice(0,indz+1)).integrate('level') / (Z[indz+1].values-Z[indz0].values)
		integ[dict(level=0)] = integ[dict(level=1)]
	else: 
		indz0,indzmax = 0,len(Z)-1
		for indz in range(indz0,indzmax-1,1):
			integ[dict(level=indz)] = var.mean(dim=dim).isel(level=slice(0,indz+1)).integrate('level') / (Z[indz+1].values-Z[indz0].values)
			#integ[dict(level=indz)] = mean.isel(level=slice(0,indz+1)).mean(dim='level') #/ (Z[indz+1].values-Z[indz0].values)
		integ[dict(level=0)] = integ[dict(level=1)]
	std_min = gamma * integ.where( integ!=999, other = integ.isel(level=indzmax-1) )

#	print('std_min.dims,std.dims')
#	print(std_min.dims,std.dims)
#	print('std_min.shape,std.shape')
#	print(std_min.shape,std.shape)
					    
	return std_min	
		
def test_integmin3(X,Y,Z,Z_w,nhalo,dpi):
	"""This procedure is testing the good behavior of integmin3 by plotting:
		fluctuations, std and std_min.
	"""
	gamma = 0.75 # =0.75 for ITURB2, =0.05 for C10
	dsCS1 = xr.open_dataset('DATA_turb/06W_CS1_warm_C10_SVTMEAN.nc')
	dsO = xr.open_mfdataset('CAS10_SVT/FICHIERS_OUT/*')
	SV3 = dsO.SVT003[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	SV1 = dsO.SVT001[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	U,V,W = dsCS1.UT,dsCS1.VT,dsCS1.WT
	Um,Vm,Wm = dsCS1.UTm,dsCS1.VTm,dsCS1.WTm
	E,Em = dsCS1.E,dsCS1.Em
	u_turb = np.sqrt(2/3*E)
	M = np.sqrt(U**2+V**2+W**2)
	u_turb_mean = np.sqrt(2/3*Em)
	M_mean = np.sqrt(Um**2+Vm**2+Wm**2)
	I_turb = u_turb/M
	I_turb_mean = u_turb_mean/M_mean
	VAR = SV1
	THTV = Compute_THTV(dsO)[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	gTHTV = THTV.differentiate('level')
#			Ones = xr.ones_like(VAR[0,:,:,:])
#			csum = Ones.cumsum(dim=['level'])
#			print(csum[0,:,:].values)
#			print(csum[5,:,:].values)
#			
#			raise Exception('stop')

	# temps et Y
#			VAR = SV3
#			indx = 269
#			std = VAR[:,:,:,:].std(dim=['time','nj'])
#			mean = VAR[:,:,:,:].mean(dim=['time','nj'])
#			fluc = VAR[:,:,:,:] - Complete_dim_like([mean],VAR)
#			
#			std_min = Integ_min3(Z,VAR,dim=['time','nj'],gamma=gamma)	
#			#z1 = VAR[:,:,:,:].mean(dim=['time','nj']).idxmax(dim='ni').values[0]	
#			#gTHTVm = gTHTV.mean(dim=['time','nj'])
#			#ABLH = gTHTVm.idxmax(dim='ni')
#			#std_min_ref = Integ_min(Z,Y,X,Complete_dim_like([std],VAR),gamma,z1=0,zi=Complete_dim_like([ABLH],VAR))
#			
#			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
#			ax.plot(std_min[:,indx],Z/680,c='k',label='std_min')
#			ax.plot(std[:,indx],Z/680,c='b',label='std')
#			#ax.plot(std_min_ref[:,indx],Z/680,c='k',label='std_min_ref',ls='--')
#			ax.plot(fluc[-1,:,20,500],Z/680,c='g',label='fluc [:,20,500]')
#			ax.plot(fluc[-1,:,20,269],Z/680,c='chartreuse',label='fluc [:,20,269]')
#			ax.legend()
#			ax.set_ylabel('z/zi')
#			ax.set_xlabel('SV1')
	
	# X et Y, this is the same nice
	VAR=I_turb[:,:,:]
	std = VAR.std(dim=['ni','nj'])
	mean = VAR.mean(dim=['ni','nj'])
	fluc = VAR - mean
	std_min = Integ_min3(Z,VAR,dim=['ni','nj'],gamma=gamma)
	z1 = VAR.mean(dim=['ni','nj']).idxmax().values	
	std_min_ref = Integ_min(Z,Y,X,Complete_dim_like([std],VAR[0]),gamma,z1=z1,zi=10000)[:,0,0]
	
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot(std_min[:],Z/680,c='k',label='v3')
	ax.plot(std_min_ref[:],Z/680,c='r',label='original',ls='--')
	ax.plot(std[:],Z/680,c='b',label='std')
	ax.plot(VAR[:,20,500],Z/680,c='g',label='VAR [:,20,500]')
	ax.plot(VAR[:,20,269],Z/680,c='chartreuse',label='VAR [:,20,269]')
	ax.plot(mean[:],Z/680,c='orange',label='mean')
	ax.legend()
	ax.set_ylabel('z/zi')
	ax.set_xlabel('std_min')
	
#			mask = xr.ones_like(VAR).where( VAR > 
#			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	plt.show()
	
def get_mixed_layer_indexes(Z,gthtv,SEUIL):
	""" This function is computing indexes of the mixed layer.
		
		it starts from surface and check at each altitude if :
			        |gthtv| > SEUIL
			 
		- Z : Altitude array 1D
		- gthtv : d(thetav)/dz 1D
		- SEUIL : in Kelvin/km	
		
		TBD : better doc
	"""
	indz1 = 0
	k = 0
	while k<len(Z)-1 and np.abs(gthtv[k])-SEUIL/1000>0:
		k=k+1
		indz1 = k
	indz2 = indz1	
	while k<len(Z)-1 and np.abs(gthtv[k])-SEUIL/1000<0: # restart from previous index
		k=k+1
		indz2 = k

	return indz1,indz2
	
def Compute_bool_turb(dim,COND,param_COND,X,Y,Z,ABLH,SV,SVm,U,V,W,Um,Vm,Wm,E,Em):
	"""
	Given COND, detects where there is a turbulent coherent structure
	
		- COND	: 'C10','ITURB','ITURB2','EC'
		- param_COND : dic with parameteres corresponding to COND
		- X,Y,Z : spatial dimensions of the instantaneous file
		- ABLH : height of boundary layer (same shape as SV)
				see Integ_min
		- SV	: passive scalar for C10
		- SVm	: mean of passive scalar for C10 (same shape as SV)
		- U,V,W,E : instantaneous wind and total TKE
		- Um,Vm,Wm,Em : mean wind and total TKE
	"""	
	if SV.name=='RVT' or SV.name=='SVCS000': # bottom up tracers
		z1 = 0
	elif SV.name=='SVCS001' or SV.name=='SVCS002': # top down tracers
		z1 = DoMean(SVm).idxmax(dim='level')
		
	if COND=='C10':
		mCS,gammaSV = param_COND['C10']
		SV_std = SV.std(dim)
#		SV_std_min = Integ_min(Z,Y,X,SV_std_mean,gammaSV,z1,zi=ABLH)	
		SV_std_min = Integ_min3(Z,SV,dim,gammaSV)
		if SV.name=='RVT':	
			abs_SV_fluc = np.abs(SV - SVm)
		else:
			abs_SV_fluc = SV - SVm
		max_cond = mCS * xr.where(SV_std > SV_std_min,SV_std,SV_std_min)
		max_cond = Complete_dim_like([max_cond],SV)
		BOOL_turb = abs_SV_fluc >= max_cond
	elif COND=='ITURB':
		gammaTurb1 = param_COND['ITURB']
		u_turb = np.sqrt(2/3*E)
		M = np.sqrt(U**2+V**2+W**2)
		I_turb = u_turb/M
		BOOL_turb = I_turb > gammaTurb1	
	elif COND=='ITURB2':
		gammaTurb2 = param_COND['ITURB2']
		u_turb = np.sqrt(2/3*E)
		M = np.sqrt(Um**2+Vm**2+Wm**2)
		#u_turb_mean = np.sqrt(2/3*Em)
		#M_mean = np.sqrt(Um**2+Vm**2+Wm**2)
		I_turb = u_turb/M
		#I_turb_mean = u_turb_mean/M_mean								
		#I_turb_min = Integ_min(Z,Y,X,I_turb_mean,gammaTurb2,z1=0,zi=ABLH)
		I_turb_min = Integ_min3(Z,I_turb,dim,gammaTurb2)
		BOOL_turb = I_turb > I_turb_min
	elif COND=='ITURB3':
		# i tested : ITURB3 is I > m.max(Iturb_min,std Iturb)	
		# conclusion of the tests -> std(Iturb) is very small so it is mainly Iturb_min that does the condition
		# 	so for ITURB3 i keep only this : I > m.Iturb_min
		m,gamma = param_COND['ITURB3']
		u_turb = np.sqrt(2/3*E)
		M = np.sqrt(Um**2+Vm**2+Wm**2)
		I_turb = u_turb/M
		I_turb_min = Integ_min3(Z,I_turb,dim,gamma)
		BOOL_turb = I_turb > m*I_turb_min
	elif COND=='EC':	
		gammaEc = param_COND['EC']		
#		E_min = Integ_min(Z,Y,X,Em,gammaEc,z1=0,zi=ABLH)
		E_min = Integ_min3(Z,Em,dim,gammaEc)
		BOOL_turb = E > E_min
		
	else:
		raise Exception('COND='+COND+' is not coded... check Compute_bool_turb')
	return BOOL_turb
		
def mean_vertical_contrib(flx_i,flx_mean,indzi):
	"""This function is computing the mean contribution of flx_i to the total
		flx_mean from surface to Z[indzi]
		
		eq 8 of Brient 2018 "Object-Oriented Identification of Coherent Structures in
					Large Eddy Simulations: Importance of Downdrafts
					in Stratocumulus"
					
		- flx_i    : field of the contribution from structure i to flux flx (xr.DataArray), Z is first dim
		- flx_mean : same size as flx_i, domain average flx (xr.DataArray)
		- indzi    : Z index to stop the average (usually 1.1zi)
	"""
	#return np.abs(flx_i.isel(level=slice(2,indzi))).mean(dim='level') / np.abs(flx_mean.isel(level=slice(2,indzi))).mean(dim='level')
	return np.abs(flx_i.isel(level=slice(2,indzi))).integrate('level') / np.abs(flx_mean.isel(level=slice(2,indzi))).integrate('level')
	
	
def compute_alpha(mask,keepdim={'Z'}):
	"""
	This function compute the area coverage of the 'mask'
	
	- mask is a 3D array of form [Z,Y,X] with 1 and 0 only
	- keepdim is a dict with the dimensions to not average
	"""
	# this needs to be modified to be more understandable
	
	if 'X' in keepdim.keys():
		F = mask.mean(axis=1)
	elif 'Y' in keepdim.keys():
		F = mask.mean(axis=1)
	elif keepdim.keys()==['Z']:
		F = mask.mean(axis=(1,2))
	return F
	
def compute_flx_contrib(flx,L_mask,meanDim=['ni','nj']):
	"""This function is computing the contribution of the flux a'b' (='flx')
		relative to 'mask'. This is the sum of the "top-hat" and "inter/intra variability" 
		
		F_i = alpha_i * flx
		
			where alpha_i is the area cover of structure i
				alpha_i = 1/N * sum_over_ipoints(mask)
		
		- flx : flux to be partionned
		- L_mask : list of mask for the current structure, 1=belong to structure
		- meanDim : dimensions to be averaged
	"""	
	L_out = []
	for i,mask in enumerate(L_mask):
		flx_i_m = flx.where(mask,other=0).mean(dim=meanDim)
		L_out.append( flx_i_m )
	if len(L_mask)==1:
		return L_out[0]
	else:
		return tuple(flx_i for flx_i in L_out)

def compute_std_flx(flx,L_mask,stdDim=['ni','nj']):
	"""Compute the standard deviation of a flx
	
		- flx : flux to be partionned
		- L_mask : list of mask for the current structure
		- stdDim : dimensions to be std
	"""	
	L_out = []
	for i,mask in enumerate(L_mask):
		flx_i_std = flx.where(mask).std(dim=stdDim,skipna=True,ddof=1)
		L_out.append( flx_i_std )
	return tuple(flx_i_std for flx_i_std in L_out)
	
def compute_beta(alpha,L_ab,mask):
	"""This function compute the top-hat contribution to total variance
		of the flux <a'b'> from coherent structure s
		
		<a'b'> = sum_on_s[beta_s] + sum_on_s[gamma_s]
		
			beta_s = alpha_s * (a_s - <a>) * (b_s - <b>)
				
				with ab_s the ab flux where object s is present
		
		see eq14 of Chinita2018 'A JPDF-based decomposition of turbulence in the atmospheric boundary layer
		
		- alpha : area fraction of the object
		- L_ab : list of the fluxes ab for object considered
		- mask of the object considered
	"""	
	# this has to be done
	
	
def compute_beta(alpha,ab):
	"""This function compute the top-hat contribution to total variance
		of the flux <a'b'> from coherent structure s
		
		<a'b'> = sum_on_s[beta_s] + sum_on_s[gamma_s]
		
			gammas = alpha_s * a'b'_s
		
		see eq14 of Chinita2018 'A JPDF-based decomposition of turbulence in the atmospheric boundary layer
		
		- alpha_s : area fraction of object s
		- a'b' : total flux (resolved)
	"""		
	# this has to be done
	

