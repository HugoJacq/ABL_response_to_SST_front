# To be used with analyse.py 
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from module_cst import *
from scipy.ndimage import uniform_filter1d
from math import factorial
from numpy.fft import fft, ifft

def DISCRETIZED_2CMAP(cmap1,cmap2,IDX,data,crit_value,coords):
	"""This function purpose is to discretize in N values a cmap
		depending on if data < crit_value or >.
	
		- cmap1 : string matplotlib cmap
		- cmap2 : string matplotlib cmap
		- IDX : step (in coordinate indices)
		- data : 1D
		- crit_value : 0D,to be compared to data
		- coords : coordinate of data
	"""
	cmap = mpl.colormaps.get_cmap(cmap1) # Colors from cmap1
	N = len(coords)//IDX
	colorsX1=[]
	for k in range(N+1):
		colorsX1.append(cmap(k/N))
	cmap = mpl.colormaps.get_cmap(cmap2) # Colors from cmap2
	colorsX2=[]
	for k in range(N+1):
		colorsX2.append(cmap(k/N))
	colorsX = []
	for x in range(0,len(coords),IDX):
		if data[x] < crit_value:
			colorsX.append(colorsX1[x//IDX])
		else:
			colorsX.append(colorsX2[x//IDX])
	return colorsX
	
def DISCRETIZED_2CMAP_2(cmap1,cmap2,list_x,data,crit_value,coords):
	"""This function purpose is to discretize in N values a cmap
		depending on if data < crit_value or >.
	
		- cmap1 : string matplotlib cmap
		- cmap2 : string matplotlib cmap
		- list_x : list of value to sample the mixed cmap at
		- data : 1D
		- crit_value : 0D,to be compared to data
		- coords : coordinate of data
		
		this function is the same as DISCRETIZED_2CMAP but with a list as input instead 
			of a dx.
	"""
	N = len(list_x)
	indx = []
	for x in range(len(list_x)):
		indx.append(np.argmin(np.abs(coords-list_x[x])))
	cmap = mpl.colormaps.get_cmap(cmap1) # Colors from cmap1
	colorsX1=[]
	for k in range(N+1):
		colorsX1.append(cmap(k/N))
	cmap = mpl.colormaps.get_cmap(cmap2) # Colors from cmap2
	colorsX2=[]
	for k in range(N+1):
		colorsX2.append(cmap(k/N))
	colorsX = []
	for kx,indice in enumerate(indx):
		if data[indice] < crit_value:
			colorsX.append(colorsX1[kx])
		else:
			colorsX.append(colorsX2[kx])
	return colorsX
	
def EXNER_FROM_TOP(EXNERTOP,Z,THTV,nhalo):
	g = 9.81
	Cpd = 1004
	IKU = len(Z) - 1
	Pi_ref = np.zeros(Z.shape)
	Pi_refmass = np.zeros(Z.shape)
	IKE =  IKU - nhalo
	#print('IKU,IKE',IKU,IKE)
	Pi_ref[IKE] = EXNERTOP #+ g/(dsB.THVREFZ[0,IKE])*(dsB.level_w[IKE+1]-dsB.level_w[IKE])
	for JK in range(IKE-1,-1,-1):
		Pi_ref[JK] = Pi_ref[JK+1] + g/(Cpd* THTV[JK])*(Z[JK+1]-Z[JK])
		#print(Pi_ref[JK])
	for JK in range(IKE+1,IKU+1,1):
		Pi_ref[JK] = Pi_ref[JK-1] - g/(Cpd* THTV[JK-1])*(Z[JK]-Z[JK-1])
		#print(Pi_ref[JK])
	#print(Pi_ref)
	Pi_refmass[:IKU-1] =  (Pi_ref[:IKU-1] - Pi_ref[1:IKU]) / (np.log(Pi_ref[:IKU-1]) - np.log(Pi_ref[1:IKU]))
	Pi_refmass[IKU] = Pi_ref[IKU-1] + (Pi_refmass[IKU-1]-Pi_refmass[IKU-2]) / (Z[IKU-1]-Z[IKU-2]) * (Z[IKU]-Z[IKU-1])	
	return Pi_refmass
	
def EXNER_FROM_GROUND(EXNERSURFACE,Z,THTV,nhalo):
	g = 9.81
	Cpd = 1004
	# THTV 1D f(z)
	# EXNERSURFACE réel
	EXNER = np.zeros(len(Z))
	EXNERM = np.zeros(len(Z))
	IKU = len(Z)
	IKB = nhalo
	EXNER[IKB] = EXNERSURFACE
	#print('IKB,IKU',IKB,IKU)
	# At flux points
	for kz in range(IKB+1,IKU,1):
		EXNER[kz] = EXNER[kz-1] - g/(Cpd*THTV[kz-1])*(Z[kz]-Z[kz-1])
		#print(kz,EXNER[kz],Z[kz]-Z[kz-1])
	for kz in range(IKB-1,-1,-1):
		EXNER[kz] = EXNER[kz+1] + g/(Cpd*THTV[kz])*(Z[kz+1]-Z[kz])
		#print(kz,EXNER[kz],Z[kz+1]-Z[kz])
	# At mass points
	EXNERM[:-1] = (EXNER[:-1] -  EXNER[1:]) / (np.log(EXNER[:-1]) - np.log(EXNER[1:]))
	EXNERM[IKU-1] = EXNER[IKU-2] + (EXNERM[IKU-2] - EXNERM[IKU-3])/(Z[IKU-2] - Z[IKU-3]) * (Z[IKU-1] - Z[IKU-2])
	#print(EXNERM)
	return EXNERM
	
def Add_SST_ticks(ax,length):
	color = 'chartreuse' # orange
	ax.vlines(5,0,length,colors=color,clip_on=False)
	ax.vlines(25,0,length,colors=color,clip_on=False)
	
def Add_SST_bar(X,Y,Ny,SST,ax):
	"""This procedure adds the change in SST as a colored line at the bottom of the plot
		from on 'ax'.
		
		X is the dimension along which the SST is changing
		Ny is the width of the line in number of points (to be adapted for each plots)
		SST is a 1d array with the values of the SST
		ax is the ax concerned.
	"""
	Nx = X.shape[0]
	cmap1 = 'bwr'
	cmap = mpl.colormaps.get_cmap(cmap1)
	Nc = 100 # how many colors for the whole range of temperature
	SST2D = np.repeat(SST[:, np.newaxis], Ny, axis=1)
	ax.pcolormesh(X,Y[:Ny],SST2D.transpose(),cmap=cmap1,vmin=np.amin(SST),vmax=np.amax(SST), shading='nearest')
	
def Newton(f,dfdx,eps,guess,Yarray):
	# f is the function
	# eps is the tolerance
		out=np.zeros(len(Yarray))
		for indz in range(len(Yarray)):
			k=0
			x = guess[indz]
			y = Yarray[indz]
			err = f(x)-y
			while np.abs(err) > eps or k>1000:
				k=k+1
				dydx = dfdksi(x)
				x = x - (f(x)-y)/dfdx(x)
				err = f(x)-y
			out[indz] = x
		return out	

#def OpenALL_OUTFILES(CHOICE):
#	"""This procedure open the files for the period considered and return the dataset and the period
#	"""
#	abs_path = '/home/jacqhugo/scripts/simu_canal_across/'
#	if CHOICE==1:
#		dsOUT = xr.open_mfdataset([abs_path+'CAS06/FICHIERS_OUT_2h_to_6h/CAS06.1.003.OUT.0'+f"{k:02}"+'.nc' for k in range(2,50)],combine='nested',concat_dim='time')
#		what='2h-6h'
#	elif CHOICE==2:
#		dsOUT = xr.open_mfdataset([abs_path+'CAS06/FICHIERS_OUT_2h_to_6h/CAS06.1.003.OUT.0'+f"{k:02}"+'.nc' for k in range(2,50)] + 
#					[abs_path+'CAS06/FICHIERS_OUT_6h_to_10h/CAS06.1.004.OUT.0'+f"{k:02}"+'.nc' for k in range(2,50)],combine='nested',concat_dim='time')
#		what='2h-10h'
#	elif CHOICE==3:
#		dsOUT = xr.open_mfdataset([abs_path+'CAS06/FICHIERS_OUT_2h_to_6h/CAS06.1.003.OUT.0'+f"{k:02}"+'.nc' for k in range(2,50)] + 
#					[abs_path+'CAS06/FICHIERS_OUT_6h_to_10h/CAS06.1.004.OUT.0'+f"{k:02}"+'.nc' for k in range(2,50)] +
#					[abs_path+'CAS06/FICHIERS_OUT_10h_to_12h/CAS06.1.005.OUT.0'+f"{k:02}"+'.nc' for k in range(2,50)],combine='nested',concat_dim='time')
#		what='2h-14h'
#	else:
#		raise Exception('Wrong choice')
#	return dsOUT,what

	
def MovieMaker(dsO,X,Y,Z,path_save,path_save_frames,atZ,dpi,delay):
	"""This procedure is ploting the wind component and then makes a gif of the frames
	"""	
	Umin,Umax = 3,6
	Vmin,Vmax = -1,1
	Wmin,Wmax = -1,1
	cmap = {'UT':'Greys_r','VT':'bwr','WT':'bwr'}
	TIME = dsO.time
	Ntime = TIME.shape[0]	
	indz=np.argmin(np.abs(Z.values-atZ))
	U = dsO['UT'][:,nhalo+0,nhalo:-nhalo,nhalo:-nhalo]
	V = dsO['VT'][:,nhalo+0,nhalo:-nhalo,nhalo:-nhalo]
	W = dsO['WT'][:,nhalo+indz,nhalo:-nhalo,nhalo:-nhalo]
	location = 'left' # 'left', 'right', 'top', 'bottom'
	for t in range(1,Ntime): #Ntime
		fig, ax = plt.subplots(3,1,figsize = (30,7),constrained_layout=True,dpi=dpi)
		s = ax[0].pcolormesh(X/1000,Y/1000,U[t],cmap=cmap['UT'],vmin=Umin,vmax=Umax)
		plt.colorbar(s,ax=ax[0],location=location,pad=0.01,shrink=0.5)
		s = ax[1].pcolormesh(X/1000,Y/1000,V[t],cmap=cmap['VT'],vmin=Vmin,vmax=Vmax)
		plt.colorbar(s,ax=ax[1],location=location,pad=0.01,shrink=0.5)
		s = ax[2].pcolormesh(X/1000,Y/1000,W[t],cmap=cmap['WT'],vmin=Wmin,vmax=Wmax)
		plt.colorbar(s,ax=ax[2],location=location,pad=0.01,shrink=0.5)
		ax[0].set_title('U (m/s), z=2m',loc='left')
		ax[1].set_title('V (m/s), z=2m',loc='left')
		ax[2].set_title('W (m/s), z='+str(atZ)+'m',loc='left')
		ax[0].tick_params(axis='x',labelbottom=False)
		ax[1].tick_params(axis='x',labelbottom=False)
		ax[2].set_xlabel('X (km)')
		ax[0].set_ylabel('Y (km)')
		ax[1].set_ylabel('Y (km)')
		ax[2].set_ylabel('Y (km)')
		for axe in ax:
			axe.set_aspect('equal')
		fig.suptitle('frame '+str(t)+'/48')
		fig.savefig(path_save_frames+f"{t:03}"+'.png')
	#os.system('convert -delay '+str(delay)+' -loop 0 '+path_save_frames+'*.png '+path_save+'Wind.gif')
	os.system('ffmpeg -framerate '+str(delay)+' -i %03d.png output.mp4')
	
def MovieMaker1VAR(VAR,time,X,Y,Z,view,atpos,cmap,vmin,vmax,coeff,dpi,fps,path_save,path_save_frames=None):
	"""Build a movie of 'VAR' with 'view' at a fixed position
	
		- VAR 4D field [t,z,y,x] (xr.DataArray)
		- time,X,Y,Z : coords of VAR
		- view : directions of the slice
		- path_save = where to save the mp4
		- path_save_frames = where to save the frames (default=path_save)
		- atpos : position (meters) of the slice on the omitted dimension in 'view'
		- cmap : for background
		- vmin : minimum of VAR
		- vmax : maximum of VAR
		- coeff : coefficient to multiply VAR
		- dpi
		- fps : number of input images displayed per second 
	"""
	nameVar = VAR.name
	units = VAR.attrs['units']
	Ntime = time.shape[0]	
	if path_save_frames==None:
		path_save_frames = path_save
	if view=='XY':
		x,y = X/1000,Y/1000
		xlabel,ylabel,orth_dim = 'X (km)','Y (km)','Z'
		aspect = 'equal'
		indpos = nearest(Z,atpos)
		var = VAR[:,indpos,:,:]
		cbar_loc = 'bottom'
		figsize = (20,3)
	elif view=='XZ':
		x,y = X/1000,Z
		xlabel,ylabel,orth_dim = 'X (km)','Z (m)','Y'
		aspect = 'auto'
		indpos = nearest(Y,atpos)
		var = VAR[:,:,indpos,:]
		cbar_loc = 'right'
		figsize = (20,5)
	elif view=='YZ':
		x,y = Y/1000,Z
		xlabel,ylabel,orth_dim = 'Y (km)','Z (m)','X'
		aspect = 'auto'
		indpos = nearest(X,atpos)
		var = VAR[:,:,:,indpos]
		cbar_loc = 'right'
		figsize = (20,5)
	for t in range(1,Ntime): #Ntime
		fig, ax = plt.subplots(1,1,figsize = figsize,constrained_layout=True,dpi=dpi)
		s = ax.pcolormesh(x,y,var[t],cmap=cmap,vmin=vmin,vmax=vmax)
		plt.colorbar(s,ax=ax,location=cbar_loc,pad=0.01,shrink=0.5)
		ax.set_title(nameVar+' ('+units+') at '+orth_dim+'='+str(atpos)+'m',loc='left')
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_aspect(aspect)
		fig.suptitle('frame '+str(t)+'/'+str(Ntime))
		fig.savefig(path_save_frames+nameVar+f"{t:03}"+'.png')
	os.system('ffmpeg -framerate '+str(fps)+' -i '+path_save_frames+nameVar+'%03d.png '+path_save+nameVar+'_at'+orth_dim+str(atpos)+'.mp4')

def MovieMaker_Vector(VAR,Vx,Vy,vect_param,time,X,Y,Z,view,atpos,cmap,coeff,bornes,dpi,fps,path_save,path_save_frames=None):
	"""Build a movie with 'VAR' in background and vector field Vx,Vy, with 'view'
	
		- VAR 4D field [t,z,y,x] (xr.DataArray)
		- Vx : x component of vector field
		- Vy : y component of vector field
		- vect_param : dict that contains
			stepx,stepy : how many points are skipped for vectors 
			headwidth : headwidth of arrows
			headaxislength : headaxislength of arrows
			scale : if scale is small then arrow is big
		- time,X,Y,Z : coords of VAR
		- view : directions of the slice
		- path_save = where to save the mp4
		- path_save_frames = where to save the frames (default=path_save)
		- atpos : position (meters) of the slice on the omitted dimension in 'view'
		- coeff : coefficient to multiply VAR
		- bornes : x and y boundaries, and var vmin/vmax
		- dpi
		- fps : number of input images displayed per second 
	"""
	nameVar = VAR.name
	units = VAR.attrs['units']
	Ntime = time.shape[0]	
	stepx = vect_param['stepx']
	stepy = vect_param['stepy']
	headwidth = vect_param['headwidth']
	headaxislength = vect_param['headaxislength']
	scale = vect_param['scale']
	vect_ref = vect_param['vect_ref']
	vmin,vmax = bornes['var'][0],bornes['var'][1]
	if path_save_frames==None:
		path_save_frames = path_save
	if view=='XY':
		x,y = X/1000,Y/1000
		xlabel,ylabel,orth_dim = 'X (km)','Y (km)','Z'
		aspect = 'equal'
		indpos = nearest(Z,atpos)
		var = VAR[:,indpos,:,:]
		U,V = Vx[:,indpos,::stepy,::stepx],Vy[:,indpos,::stepy,::stepx]
		cbar_loc = 'bottom'
		figsize = (20,3)
	elif view=='XZ':
		x,y = X/1000,Z
		xlabel,ylabel,orth_dim = 'X (km)','Z (m)','Y'
		aspect = 'auto'
		indpos = nearest(Y,atpos)
		var = VAR[:,:,indpos,:]
		U,V = Vx[:,::stepy,indpos,::stepx],Vy[:,::stepy,indpos,::stepx]
		cbar_loc = 'right'
		figsize = (10,5)
	elif view=='YZ':
		x,y = Y/1000,Z
		xlabel,ylabel,orth_dim = 'Y (km)','Z (m)','X'
		aspect = 'auto'
		indpos = nearest(X,atpos)
		var = VAR[:,:,:,indpos]
		U,V = Vx[:,::stepy,::stepy,indpos],Vy[:,::stepy,::stepy,indpos]
		cbar_loc = 'right'
		figsize = (20,5)
	for t in range(1,Ntime): #Ntime
		fig, ax = plt.subplots(1,1,figsize = figsize,constrained_layout=True,dpi=dpi)
		s = ax.pcolormesh(x,y,var[t]*coeff,cmap=cmap,vmin=vmin,vmax=vmax)
		plt.colorbar(s,ax=ax,location=cbar_loc,pad=0.01,shrink=0.5)
		Q = ax.quiver(x[::stepx],y[::stepy],U[t].values,V[t].values,angles='uv',pivot='middle',headwidth=headwidth,headaxislength=headaxislength,scale=scale)
		ax.quiverkey(Q, 0.9, 0.05, vect_ref, str(vect_ref)+' m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
		ax.set_title(nameVar+' ('+units+') at '+orth_dim+'='+str(atpos)+'m',loc='left')
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_aspect(aspect)
		ax.set_ylim(bornes['y'])
		ax.set_xlim(bornes['x'])
		fig.suptitle('frame '+str(t)+'/'+str(Ntime))
		fig.savefig(path_save_frames+'VEC'+nameVar+f"{t:03}"+'.png')
		plt.close()
	os.system('ffmpeg -framerate '+str(fps)+' -i '+path_save_frames+'VEC'+nameVar+'%03d.png '+path_save+nameVar+'_at'+orth_dim+str(atpos)+'.mp4')
	# ffmpeg -framerate 5 -i PNG_CAS06W/TURB_STRUCTURE/ITURB2_g75.0_MIXED/frames/VECRVT%03d.png PNG_CAS06W/TURB_STRUCTURE/ITURB2_g75.0_MIXED/RVT_atY1000.mp4
	
def Compute_Mass(ds):
	"""This function computes the time evolution of mass of a dataset, both dry air and water mass
	"""
	Ntime=ds.time.shape[0]
	PI_3D = ( ds.PABST[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]/P00 )**(Rd/Cpd)
	RVT_3D = ds.RVT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	THT_3D = ds.THT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	THTV_3D = Compute_THTV(THT_3D,RVT_3D)
	RHO_3D = PI_3D**(Cvd/Rd)*P00 / (Rd*THTV_3D) / (1+RVT_3D) # equation of state for moist air, to get dry density
	MD = RHO_3D.integrate(['ni','nj','level'])
	MW = (RHO_3D*RVT_3D).integrate(['ni','nj','level'])
	if Ntime>1:
		return MD,MW # if files from OUTPUT
	else:
		return MD[0].values,MW[0].values # if initial files
	
def nearest(array,value):
	"""
	Array is 1D
	value is 0D
	"""
	return np.argmin(np.abs(array-value))	
	
	
def Complete_dim_like(L_VAR,VAR_like):
	""" This function is expanding the dimensions of VAR to be the same as VAR_like.
	
		this is useful in the case where VAR = VAR_like.mean(axis=1) for eg.
		It allow then a fast computation of operations involving the 2.
		
		L_VAR is a list of the variable to expand to be like VAR_like
	"""
	L_OUT = []
	#VAR_like = VAR_like.reset_coords(drop=True) # remove unused coordinates if input is not clean
	for VAR in L_VAR:
		dic_dim = {}
		dic_coords = {}
		index = []
		VAR = VAR.reset_coords(drop=True) # remove unused coordinates
		for k,dim in enumerate(VAR_like.dims):
			if dim not in VAR.dims:
				dic_dim[dim] = VAR_like.coords[dim]
				dic_dim[dim] = dic_dim[dim].reset_coords(drop=True) 
				index.append(k)
		VAR = VAR.expand_dims(dim=dic_dim,axis=index)
		for coords in VAR.coords:
			if coords not in VAR.coords and coords in VAR.dims:
					dic_coords[coords] = coords,VAR_like.coords[coords]
		VAR = VAR.assign_coords(dic_coords)
		if VAR_like.chunksizes != None: # chunking if necessary, for // computing. Should be avoided
			VAR = VAR.chunk(chunks=VAR_like.chunksizes)
		L_OUT.append(VAR)
	if len(L_OUT)>1:
		return tuple(L_OUT)
	else:
		return L_OUT[0]
	
	
def DoMean(VAR):
	"""Average depending on size of VAR
	
		if size = 4 (T,Z,Y,X)
			average is done on T,Y
		else average is done on all but Z
		
		- Var is a dataarray
	
	"""
	shape = VAR.shape
	if len(shape)==4:
		L = ['time','nj']
	else:
		L = []
		for dim in VAR.dims:
			if dim != 'level':
				L.append(dim)
	return VAR.mean(dim=L)
	
	
	
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
	The Savitzky-Golay filter removes high frequency noise from data.
	It has the advantage of preserving the original shape and
	features of the signal better than other types of filtering
	approaches, such as moving averages techniques.
	 Parameters
	----------
	y : array_like, shape (N,)
		the values of the time history of the signal.
	window_size : int
		the length of the window. Must be an odd integer number.
	order : int
		the order of the polynomial used in the filtering.
		Must be less then `window_size` - 1.
	deriv: int
		the order of the derivative to compute (default = 0 means only smoothing)
	Returns
	-------
	ys : ndarray, shape (N)
		the smoothed signal (or it's n-th derivative).
	Notes
	-----
	The Savitzky-Golay is a type of low-pass filter, particularly
	suited for smoothing noisy data. The main idea behind this
	approach is to make for each point a least-square fit with a
	polynomial of high order over a odd-sized window centered at
	the point.
	Examples
	--------
	t = np.linspace(-4, 4, 500)
	y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
	ysg = savitzky_golay(y, window_size=31, order=4)
	import matplotlib.pyplot as plt
	plt.plot(t, y, label='Noisy signal')
	plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
	plt.plot(t, ysg, 'r', label='Filtered signal')
	plt.legend()
	plt.show()
	References
	----------
	.. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
	Data by Simplified Least Squares Procedures. Analytical
	Chemistry, 1964, 36 (8), pp 1627-1639.
	[2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
	W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
	Cambridge University Press ISBN-13: 9780521880688
	"""
	try:
	    window_size = np.abs(int(window_size))
	    order = np.abs(int(order))
	except ValueError:
	    raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
	    raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
	    raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] ) # not cyclic condition
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

def profil_ini_tht(Z,tht0,slope,zi):
	"""
	Define a profile of theta that is constant=tht0 up until zi and then changes
		with the slope.
	IN
		Z altitude (m)
		tht0 potential temperature in K
		slope (K/m)
		zi height of initial boundary layer
	OUT
		array of theta at each altitude
	"""
	tht = np.zeros(Z.shape)
	for k,z in enumerate(Z):
		if z<zi:
			tht[k] = tht0
		else:
			tht[k] = tht0+slope*(z-zi)
	return tht

# OLD
#def f_MeanTurb(field,Tstart=0,Tstop=-1,window=20):
#	"""
#	Mean operator for turbulent statistics, time mean and Y mean
#	IN
#		MNH field of the form DATA(t,z,y,x)
#		Tstart indice of time to start the mean
#		Tstop indice of time to stop the mean
#		window is the size of the window for running average (in points)
#	OUT
#		MNH mean field DATA(z,x)
#	"""
#	meaned = field[Tstart:Tstop,:,:,:].mean(dim=['time','nj'])
#	meaned = uniform_filter1d(meaned, size=window,axis=1,mode='wrap')
#	return meaned
#def MeanTurb(field,Tstart,Tstop,window):
#	return xr.apply_ufunc(
#		f_MeanTurb,
#			field,
#			dask="allowed",
#			input_core_dims=[['time','nj']],
#			vectorize=True,
#			kwargs={'Tstart':Tstart,'Tstop':Tstop,'window':window},
#			keep_attrs=True
#		)


# EXAMPLE OF UFUNC
#RV = dsO.RVT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
##def wrapper(array, **kwargs):
##    #print(f"received {type(array)} shape: {array.shape}, kwargs: {kwargs}")
##    result = uniform_filter1d(array, **kwargs)
##   # print(f"result.shape: {result.shape}")
##    return result
##RV_f = ufunc_1DuniformFilter(wrapper,RV,size=window,axis=3,mode='wrap')
##print('done //')
##expected = uniform_filter1d(RV.values,size=window,axis=3,mode='wrap')
##print('done classic')
##print(RV_f)
##print(expected)
#start = time.time()
#RV_f = MeanTurb2(RV,Tstart,Tstop,window).compute()
#step1 = time.time()
#expected = RV[Tstart:Tstop,:,:,:].mean(dim=['time','nj'])
#expected = uniform_filter1d(expected.values,size=window,axis=1,mode='wrap')
#end = time.time()
#print('ufunc:',step1-start)
#print('numpy:',end-step1)
#print(RV_f)
#print(expected)
#print(RV_f.values == expected)
##xr.testing.assert_allclose(
##    expected, RV_f
##) 
#raise Exception('zone de test')

def ufunc_1DuniformFilter(array,size=20,axis=0,mode='wrap'):
	return xr.apply_ufunc(
		uniform_filter1d,	# func to use
		array,				# input of func, usually numpy array
		dask="parallelized", # to allow // computing if array is already chuncked
		input_core_dims=[['ni']],	# axis of work of func
		output_core_dims=[['ni']],
		kwargs={'size':size,'axis':1,'mode':'wrap'}, # kwargs of func
		output_dtypes=[array.dtype],				# this is passed to Dask
		dask_gufunc_kwargs={'allow_rechunk':True}	# this is passed to Dask, if core dim is chuncked
	)	

def MeanTurb(array,Tstart,Tstop,window):
	meaned = array[Tstart:Tstop,:,:,:].mean(dim=['time','nj'])
	#meaned = moving_average(meaned.values,axis=1,n=window) # reduce the dimension of 'array' by 'window' points
	meaned = ufunc_1DuniformFilter(meaned, size=window,axis=1,mode='wrap')
	return meaned

def moving_average(a,axis=0, n=3):
    ret = np.cumsum(a,axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def T_to_Theta(T,P): 
	#Renvoie la température potentielle en K
	#IN : Température (K)
	#     Pression (Pa)
	#OUT : Theta (K)	
	theta=T*(P00/P)**(Rd/Cpd) #K
	return theta
	
def Theta_to_T(Theta,P): 
	#Renvoie la température en K
	#IN : Température potentielle (K)
	#     Pression (Pa)
	#OUT : Température (K)	
	temp=Theta*(P/P00)**(Rd/Cpd) #K
	return temp

def sat_vap_pressure(T):
	# compute saturation vapor pressure
	return e0*np.exp( Lv/Rv * (1/273.15 - 1/T) )

def Td(rv,P):
	# compute dew point temperature = 'point de rosé'
	# A saturation mixing ratio can be deduced from this on emagram
	#	if you follow iso rvsat and crosses the dry adiabatic, you get LCL
	# P in Pa
	return 1/( 1/273.15 - Rv/Lv * np.ln(rv*P/(e0*(rv+Rd/Rv))) )
	
def compute_rv_s(T,P):
	# Compute saturation mixing ratio at T and P
	# P in Pa
	es = sat_vap_pressure(T)
	return Rd/Rv * es / (P-es)
	
def qtoRH(RVT,THT,P):
	# Calcul l'humidité relative
	# Tiré du code de MNH
	# Constantes (déclarée : modd_cst.f90 , initialisée : ini_cst.f90)	
	XBOLTZ      = 1.380658E-23		# Boltzman constant
	XAVOGADRO   = 6.0221367E+23		# Avogadro number
	XMD    = 28.9644E-3			# Molar mass of dry air
	XMV    = 18.0153E-3			# Molar mass of vapor
	#XRD    = XAVOGADRO * XBOLTZ / XMD	# Gaz constant for dry air
	XRV    = XAVOGADRO * XBOLTZ / XMV	# Gaz constant for vapor
	#XCPD   = 7.* XRD /2.			# Cpd (dry air)
	XCPV   = 4.* XRV			# Cpv (vapor)
	#XRHOLW = 1000.				# Volumic mass of liquid water
	#XRHOLI = 900.				# Volumic mass of ice
	#XCONDI = 2.22				# Thermal conductivity of ice (W m-1 K-1)
	XCL    = 4.218E+3			# Cl (liquid)
	#XCI    = 2.106E+3			# Ci (ice)
	XTT    = 273.16				# Triple point temperature
	XLVTT  = 2.5008E+6			# Vaporization heat constant
	#XLSTT  = 2.8345E+6			# Sublimation heat constant
	#XLMTT  = XLSTT - XLVTT			# Melting heat constant
	XESTT  = 611.14				# Saturation vapor pressure  at triple point
	# Constants for saturation vapor
	XGAMW  = (XCL - XCPV) / XRV
	XBETAW = (XLVTT/XRV) + (XGAMW * XTT)
	XALPW  = np.log(XESTT) + (XBETAW /XTT) + (XGAMW *np.log(XTT))
	# Conversion Theta en T
	T = Theta_to_T(THT,P)
	# Calcul Pression de saturation (compute_function_thermo.f90)
	PSAT = np.exp( XALPW - XBETAW/T - XGAMW*np.log(T))
	SATmr = (XMV/XMD)*PSAT/(P-PSAT)
	# Humidité relative
	RHU = RVT/SATmr
	return RHU

def Exner(P):
	P0 = 100000	# Pression de référence
	Cpd = 1004.71	# Capacité thermique à pression cste
	Rd = 287.05	# Constante des gaz parfaits air sec
	return (P/P0)**(Rd/Cpd)
	
def inv_Exner(Pi):
	P0 = 100000	# Pression de référence
	Cpd = 1004.71	# Capacité thermique à pression cste
	Rd = 287.05	# Constante des gaz parfaits air sec
	return P0*Pi**(Cpd/Rd)
	
def L_Obukhov(tht0,u_star,surf_wtht_flx):
	g = 9.81
	K = 0.41 # von karman
	return - tht0*u_star**3/( K*g*surf_wtht_flx)
	
def Compute_w_star(flx,zi):
	g=9.81 # gravity
	thtv=300 # K ref temperature
	return (g/thtv * flx*zi)**(1/3)
	
	
def Compute_THTV(THT,RVT):
	"""input can be DataArray
	"""
	return THT*(1+Rv/Rd*RVT)/(1+RVT)
	
	
	
def spectrum(x,fs):
	"""
	Computes the 1D power spectrum of the random variable 'x'
	INPUTS:
		- x : random variable 
		- fs : sampling frequency
	"""
	N = len(x)
	tm = N/fs
	df = 1./tm   
	#print(df)
	f = np.arange(0,fs,df) 
	#calcul fft et PSD
	xx = np.fft.fft(x)
	pxx = np.real(xx*np.conj(xx)/(N**2))
	psdx = pxx
	#size of psdw
	di = len(psdx)
	di2 = int(np.floor(di/2))
	#fold over spectrum
	psdx = 2*psdx[1:di2]
	f = f[1:di2]
	# outputs:
	# f and psdx : frequencies and associated power spectrum
	return [f,psdx]
	
	
def cospectrum(a,b,fs):
	"""
	Computes the 1D power cospectrum of the correlation ab
	INPUTS:
		- a : random variable 1
		- b : random variable 2 of same length as 'a'
		- fs : sampling frequency
	"""
	N = len(a)
	tm = N/fs
	df = 1./tm   
	#print(df)
	f = np.arange(0,fs,df) 
	#calcul fft et PSD
	ffta = np.fft.fft(a)
	fftb = np.fft.fft(b)
	pxx = np.real(ffta*np.conj(fftb)/(N**2))
	psdx = pxx
	#size of psdw
	di = len(psdx)
	di2 = int(np.floor(di/2))
	#fold over spectrum
	psdx = 2*psdx[1:di2]
	f = f[1:di2]
	# outputs:
	# f and psdx : frequencies and associated power spectrum
	return [f,psdx]
	
	
def Minor_ticks_symLog(major_loc,linthresh):
	"""
	Compute the minor ticks position (inbetween the powers of 10) to be used with a symetric log colorbar
	
	INPUTS:
		- major_loc : is obtained with cb.get_ticks(). This might have to be rounded to give exact values (and not a form like 0.999999 for 1.0)
		- linthresh : same as used in the SymLogNorm
	"""
	minor_loc = []
	for i in range(1,len(major_loc)):
		majorstep = major_loc[i] - major_loc[i-1]
		neg_list = np.arange(-9,0,1)
		pos_list = np.arange(2,10,1)
		if abs(major_loc[i-1] + majorstep/2) > linthresh:
			if major_loc[i] < 0:
				val = neg_list
				mult = major_loc[i]
			else:
				val = pos_list
				mult = major_loc[i-1]
			locs = ( val*np.abs(mult) )[1:]
			minor_loc.extend(locs)	
	return minor_loc
	
	
def phim_H88(z,L):
	"""
	Stability function for momentum, following Hogstrom 1988 (Stull book)
	"""
	if z/L < 0:
		return (1-19.3*z/L)**(-1/4)
	else:
		return 1+4.8*z/L
		
def phih_H88(z,L):
	"""
	Stability function for heat, following Hogstrom 1988 (Stull book)
	"""
	if z/L < 0:
		return (1-12*z/L)**(-1/2)
	else:
		return 1+7.8*z/L
	
def RCorr_cyclic(X,Y):
	"""
	Compute the pearson coefficient of X and Y and the lag associated.
	The lag is from 0 to len(X)
	
	R(tau) = corr(X(t),Y(t+tau)) / stdX*stdY
	
	
	return Corr_coef of size len(X)
	
	Note : 	- it is assumed that X and Y are cyclic : Y[-1] = Y[0]
	 	- X and Y have same length
	 	- X is lagging behind Y
	 	
	 	
	Example :
	 
	P = 4
	N = 2000
	T = np.linspace(0,P*np.pi,N)
	X1 = np.cos(T) + 2
	Y1 = np.cos(T+np.pi/2) + 4
	fig, ax = plt.subplots(1,1,figsize = (3,3),constrained_layout=True,dpi=dpi)
	ax.plot(T/np.pi,X1,label='X1')
	ax.plot(T/np.pi,Y1,label='Y1')
	ax.legend()
	Corr_coef = RCorr_cyclic(X1,Y1)
	fig, ax = plt.subplots(1,1,figsize = (3,3),constrained_layout=True,dpi=dpi)
	ax.plot(np.arange(0,len(X1))/N*P,Corr_coef,c='k')
	"""	
	sig_X = np.std(X)
	sig_Y = np.std(Y)
	mean_X = X.mean()
	mean_Y = Y.mean()
	Corr = np.zeros(X.shape)
	Corr_coef = np.zeros(X.shape)
	Y2 = np.zeros(X.shape)
	for idx in range(1,len(X)): 
		Y2[:len(X)-idx] = Y[idx:]
		Y2[len(X)-idx:] = Y[0:idx]
		Corr[idx] = ((X-mean_X)*(Y2-mean_Y)).mean()
		#Corr[idx] = ((X1[:-idx]-mean_X1)*(Y1[idx:]-mean_Y1)).mean()
		Corr_coef[idx] = Corr[idx]/(sig_Y*sig_X)
	Corr_coef = periodic_corr(X-mean_X,Y-mean_Y)/(len(X)*sig_Y*sig_X)
	return Corr_coef

def RCorr_nocyclic(X,Y):
	"""
	X and Y have the same length (=N)
	
	We roll Y to match a pattern in X, N times 
	"""
	
	X1 = X - np.mean(X)
	Y1 = Y - np.mean(Y)
	stdX = np.std(X)
	stdY = np.std(Y)
	
	N = len(X)
	R = np.zeros(X.shape)
	for ilag in range(0,len(X)):
		R[ilag] = np.mean( X1*np.roll(Y1,ilag) )
	return R/(stdX*stdY)
	
	
def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
