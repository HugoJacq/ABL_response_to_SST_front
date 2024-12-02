# To be used with analyse.py 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from module_cst import *
from module_tools import *
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress
import cmocean
import numpy.polynomial.polynomial as poly
from module_building_files import *
import matplotlib.ticker as ticker


# In paper ---------------------
def FIRST_LOOK(dataSST,dsINI,dsB,dsmean,dsref,dsflx,X,Y,Z,Z_w,time,nhalo,height,crit_value,path_save,dpi):
	"""The purpose of this procedure is to give an overview of the S1 simulation and 
		compare it to the homogeneous reference simulations.
	 
	 INPUTS:
	 	- dataSST	: 1D SST profile
	 	- dsINI 	: dataset that contains initial conditions
	 	- dsB 		: dataset that contains full BACKUP files from MNH sim
	 	- dsmean 	: dataset with mean fields (built with Build_mean_file)
	 	- dsref 	: dataset opened with Open_LES_MEAN to get the data of a .000.nc file (diachronic MNH file) 
	 	- dsflx 	: dataset with mean flux (built with Build_flx_file)
	 	- X 		: X dimension
	 	- Y 		: Y dimension
	 	- Z 		: Z dimension
	 	- Z_w 		: Top cell location Z dimension
	 	- time 		: time dimension
	 	- nhalo 	: MNH halo
	 	- height 	: altitude where to plot the instantaneous vertical velocity
	 	- crit_value 	: threshold to consider the SST on 'cold' or 'warm' side
	 	- path_save 	: where to save the figures
	 	- dpi 		: for figures
	 OUPUTS:
	 	- A snapshot of last instant of S1 : U,V at surface and W at 'height'
	 	- <U> and <W> fields (X and Z dependant)
	 	- Surface fluxes (sensible and latent heat, friction velocity) and surface temperatures (SST and T(z=1m)) 
	 		for both S1 and reference simulations
	 	- initial conditions profiles (theta, theta_v, rv)
	 	- a set of anomalies of U,THT,THTV,RV compared to the profile at X=4km
	 
	"""
	normAtX = 4 	# km
	normAtX2 = 23 	# km
	indnormAtX = np.argmin(np.abs(X.values - normAtX*1000))
	indnormAtX2 = np.argmin(np.abs(X.values - normAtX2*1000))
	gTHT = np.gradient(dsmean.THTm,Z,axis=0)
	gTHT_contour = 0.0005 # K/m
	ABLH = 600 	# m, unified ABLH for S1
	indt_c = -1
	indt_w = -1
	
	print('Time for last time of S1 =',dsB.time[-1].values)
	print('Time for warm ref =',dsref['warm']['000'].time_les[indt_w].values)
	print('Time for cold ref =',dsref['cold']['000'].time_les[indt_c].values)
	E0, E0_c, E0_w 			= dsmean.E0, dsref['cold']['nomean']['Surface']['E0'][indt_c].values, dsref['warm']['nomean']['Surface']['E0'][indt_w].values
	Q0, Q0_c, Q0_w 			= dsmean.Q_star, dsref['cold']['nomean']['Surface']['Q0'][indt_c].values, dsref['warm']['nomean']['Surface']['Q0'][indt_w].values
	THT_z0,THT_z0_c,THT_z0_w 	= dsmean.THTm[0,:], dsref['cold']['nomean']['Mean']['MEAN_TH'][indt_c,0].values, dsref['warm']['nomean']['Mean']['MEAN_TH'][indt_w,0].values
	RV_z0,RV_z0_c,RV_z0_w 		= dsmean.RVTm[0,:], dsref['cold']['nomean']['Mean']['MEAN_RV'][indt_c,0].values, dsref['warm']['nomean']['Mean']['MEAN_RV'][indt_w,0].values
	THTv_z0,THTv_z0_c,THTv_z0_w 	= Compute_THTV(THT_z0,RV_z0), Compute_THTV(THT_z0_c,RV_z0_c), Compute_THTV(THT_z0_w,RV_z0_w)
	WTHTVs, WTHTVs_c, WTHTVs_w 	= dsflx.FLX_THvW[0,:], THTv_z0_c/THT_z0_c*Q0_c+0.61*THT_z0_c*E0_c, THTv_z0_w/THT_z0_w*Q0_w+0.61*THT_z0_w*E0_w
	u_star,u_star_c,u_star_w 	= dsmean.u_star, dsref['cold']['nomean']['Surface']['Ustar'][indt_c].values, dsref['warm']['nomean']['Surface']['Ustar'][indt_w].values
	THT1M_c,THT1M_w = dsref['cold']['nomean']['Mean']['MEAN_TH'][indt_c,0].values,dsref['warm']['nomean']['Mean']['MEAN_TH'][indt_w,0].values
	P_c,P_w = dsref['cold']['nomean']['Mean']['MEAN_PRE'][indt_c,0].values,dsref['warm']['nomean']['Mean']['MEAN_PRE'][indt_w,0].values
	T1M_c,T1M_w = Theta_to_T(THT1M_c,P_c),Theta_to_T(THT1M_w,P_w)

	# normalization
	tht_c = dsref['cold']['nomean']['Mean']['MEAN_TH'][indt_c,:]
	z_les_c = dsref['cold']['000'].level_les
	gTHTV_w = Compute_THTV(dsref['warm']['nomean']['Mean']['MEAN_TH'],dsref['warm']['nomean']['Mean']['MEAN_RV'])[indt_w].differentiate('level_les')
	gTHTV_c = Compute_THTV(dsref['cold']['nomean']['Mean']['MEAN_TH'],dsref['cold']['nomean']['Mean']['MEAN_RV'])[indt_c].differentiate('level_les')	
		
	zi,	zi_c,      zi_w     = ( ABLH,
					Z[gTHTV_c.argmax('level_les').values].values, 
					Z[gTHTV_w.argmax('level_les').values].values )
	tht0 = 295.5 # K
	tht_star,tht_star_c,tht_star_w 	= WTHTVs/u_star, WTHTVs_c/u_star_c, WTHTVs_w/u_star_w
	Q_star = np.abs(Q0[indnormAtX].values)
	E_star = np.abs(E0[indnormAtX].values)
	u_norm = u_star[indnormAtX].values
	YLIM = [0,1.2]
	NAMEY = r'z/z$_i$'
	print('Zi: S1,cold,warm:',zi,zi_c,zi_w)
		
	print('	- Last instant XY')
	Xsplit = 19 # km
	indz = np.argmin(np.abs(Z_w.values-height))
	U = dsB.UT[time,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	V = dsB.VT[time,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	W = dsB.WT[time,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	Umin,Umax = 4,6
	Vmin,Vmax = -1,1
	Wmin,Wmax = -1,1
	figsize= (4,10)	 	

	# fig, ax = plt.subplots(6,1,figsize = figsize,constrained_layout=True,dpi=dpi)
	# # U
	# s = ax[0].pcolormesh(X/1000,Y/1000,U[0,:,:],cmap='Greys_r',vmin=Umin,vmax=Umax)
	# ax[0].set_xlim([0,Xsplit])
	# s = ax[1].pcolormesh(X/1000,Y/1000,U[0,:,:],cmap='Greys_r',vmin=Umin,vmax=Umax)
	# ax[1].set_xlim([Xsplit,X[-1]/1000])
	# fig.colorbar(s, ax=ax[0:2], shrink=0.6,pad = 0.01)
	# # V
	# s = ax[2].pcolormesh(X/1000,Y/1000,V[0,:,:],cmap=cmocean.cm.curl,vmin=Vmin,vmax=Vmax)
	# ax[2].set_xlim([0,Xsplit])
	# s = ax[3].pcolormesh(X/1000,Y/1000,V[0,:,:],cmap=cmocean.cm.curl,vmin=Vmin,vmax=Vmax)
	# ax[3].set_xlim([Xsplit,X[-1]/1000])
	# fig.colorbar(s, ax=ax[2:4], shrink=0.6,pad = 0.01)
	# # W
	# s = ax[4].pcolormesh(X/1000,Y/1000,W[indz,:,:],cmap='seismic',vmin=Wmin,vmax=Wmax)
	# ax[4].set_xlim([0,Xsplit])
	# s = ax[5].pcolormesh(X/1000,Y/1000,W[indz,:,:],cmap='seismic',vmin=Wmin,vmax=Wmax)
	# ax[5].set_xlim([Xsplit,X[-1]/1000])
	# fig.colorbar(s, ax=ax[4:6], shrink=0.6,pad = 0.01)
	# axis names
	# ax[0].set_title(r'U (m.s$^{-1}$), z=1m',loc='right')
	# ax[2].set_title(r'V (m.s$^{-1}$), z=1m',loc='right')
	# ax[4].set_title(r'W (m.s$^{-1}$), z='+str(height)+'m',loc='right')
	# ax[5].set_xlabel('X (km)')
	# for axe in ax.flatten():
	# 	axe.set_aspect('equal')
	# 	axe.set_ylabel('Y (km)')
	# 	axe.xaxis.set_major_locator(ticker.MultipleLocator(2))
	# 	axe.yaxis.set_major_locator(ticker.MultipleLocator(2))
	# 	Add_SST_bar(X.values/1000,Y.values/1000,2,dataSST,axe) # add SST front representation
	# fig.savefig(path_save+'LastTime_UVW_XY.png')

	figsize= (4,4)
	# U
	fig, ax = plt.subplots(2,1,figsize = (4,4),constrained_layout=True,dpi=dpi)
	s = ax[0].pcolormesh(X/1000,Y/1000,U[0,:,:],cmap='Greys_r',vmin=Umin,vmax=Umax)
	ax[0].set_xlim([0,Xsplit])
	s = ax[1].pcolormesh(X/1000,Y/1000,U[0,:,:],cmap='Greys_r',vmin=Umin,vmax=Umax)
	ax[1].set_xlim([Xsplit,X[-1]/1000])
	fig.colorbar(s, ax=ax[0:2], shrink=0.6,pad = 0.01)
	ax[0].set_title(r'U (m.s$^{-1}$), z=1m',loc='right')
	ax[1].set_xlabel('X (km)')
	for axe in ax.flatten():
		axe.set_aspect('equal')
		axe.set_ylabel('Y (km)')
		axe.xaxis.set_major_locator(ticker.MultipleLocator(2))
		axe.yaxis.set_major_locator(ticker.MultipleLocator(2))
		Add_SST_bar(X.values/1000,Y.values/1000,3,dataSST,axe) # add SST front representation
	fig.savefig(path_save+'LastTime_UVW_XY_U.png')
	# V
	fig, ax = plt.subplots(2,1,figsize = (4,4),constrained_layout=True,dpi=dpi)
	s = ax[0].pcolormesh(X/1000,Y/1000,V[0,:,:],cmap=cmocean.cm.curl,vmin=Vmin,vmax=Vmax)
	ax[0].set_xlim([0,Xsplit])
	s = ax[1].pcolormesh(X/1000,Y/1000,V[0,:,:],cmap=cmocean.cm.curl,vmin=Vmin,vmax=Vmax)
	ax[1].set_xlim([Xsplit,X[-1]/1000])
	fig.colorbar(s, ax=ax[0:2], shrink=0.6,pad = 0.01)
	ax[0].set_title(r'V (m.s$^{-1}$), z=1m',loc='right')
	ax[1].set_xlabel('X (km)')
	for axe in ax.flatten():
		axe.set_aspect('equal')
		axe.set_ylabel('Y (km)')
		axe.xaxis.set_major_locator(ticker.MultipleLocator(2))
		axe.yaxis.set_major_locator(ticker.MultipleLocator(2))
		Add_SST_bar(X.values/1000,Y.values/1000,3,dataSST,axe) # add SST front representation
	fig.savefig(path_save+'LastTime_UVW_XY_V.png')
	# W
	fig, ax = plt.subplots(2,1,figsize = (4,4),constrained_layout=True,dpi=dpi)
	s = ax[0].pcolormesh(X/1000,Y/1000,W[indz,:,:],cmap='seismic',vmin=Wmin,vmax=Wmax)
	ax[0].set_xlim([0,Xsplit])
	s = ax[1].pcolormesh(X/1000,Y/1000,W[indz,:,:],cmap='seismic',vmin=Wmin,vmax=Wmax)
	ax[1].set_xlim([Xsplit,X[-1]/1000])
	fig.colorbar(s, ax=ax[0:2], shrink=0.6,pad = 0.01)
	ax[0].set_title(r'W (m.s$^{-1}$), z='+str(height)+'m',loc='right')
	ax[1].set_xlabel('X (km)')
	for axe in ax.flatten():
		axe.set_aspect('equal')
		axe.set_ylabel('Y (km)')
		axe.xaxis.set_major_locator(ticker.MultipleLocator(2))
		axe.yaxis.set_major_locator(ticker.MultipleLocator(2))
		Add_SST_bar(X.values/1000,Y.values/1000,3,dataSST,axe) # add SST front representation
	fig.savefig(path_save+'LastTime_UVW_XY_W.png')

	print('	- Mean fields XZ') # no V bc no use
	U0 = dsmean.Um[:,0]
	U0.expand_dims(dim={'ni':X},axis=1)
	Um = dsmean.Um #- U0
	levelsU=np.arange(6.0,7.5,0.1) # 4.5,7.5,0.25
	Vm = dsmean.Vm
	levelsV=np.arange(-0.5,0.05,0.05)
	Wm = dsmean.Wm
	THTm = dsmean.THTm
	Pm = dsmean.Pm
	PIm = Exner(Pm)
	Tm = THTm*PIm
	Umin,Umax = -1,1
	Vmin,Vmax = -0.5,0.
	Wmin,Wmax = -1,1
	figsize= (10,4)
	fig, ax = plt.subplots(1,2,figsize = figsize,constrained_layout=True,dpi=dpi)	
	s = ax[0].contourf(X/1000,Z/zi,Um,cmap='Greys_r',levels=levelsU,extend='both')
	plt.colorbar(s,ax=ax[0],orientation='vertical')
	s = ax[1].pcolormesh(X/1000,Z/zi,Wm*100,cmap="bwr",vmin=Wmin,vmax=Wmax)
	plt.colorbar(s,ax=ax[1],orientation='vertical') #,format='%.0e'
	#ax[0].contour(X/1000,Z/zi,gTHT,levels=[gTHT_contour],colors='grey',linestyles='--') # this is for IBLs
	#ax[1].contour(X/1000,Z/zi,gTHT,levels=[gTHT_contour],colors='grey',linestyles='--') # this is for IBLs
	#ax[2].contour(X/1000,Z/zi,gTHT,levels=[gTHT_contour],colors='grey',linestyles='--') # this is for IBLs
	Add_SST_bar(X.values/1000,Z.values/zi,5,dataSST,ax[0])
	Add_SST_bar(X.values/1000,Z.values/zi,5,dataSST,ax[1])
	ax[0].set_ylabel(NAMEY)
	ax[1].set_ylabel(NAMEY)
	ax[0].set_title(r'<U> (m.s$^{-1}$)',loc='right')
	ax[1].set_title(r'<W> (cm.s$^{-1}$)',loc='right')
	ax[1].set_xlabel('X (km)')
	ax[0].set_xlabel('X (km)')
	ax[0].set_ylim(YLIM)
	ax[1].set_ylim(YLIM)
	for axe in ax.flatten():	
		axe.xaxis.label.set_fontsize(13)
		axe.yaxis.label.set_fontsize(13)
	fig.savefig(path_save+'FirstLook_UVW.png')
	
	# Divergence
	# fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
	# s = ax.pcolormesh(X/1000,Z/zi,Wm.differentiate('level')*1e5,cmap='bwr',vmin=-8,vmax=8)
	# plt.colorbar(s,ax=ax)
	# ax.set_title('divergence d<W>/dz')
	# ax.set_ylim([0,1.2])
	
	
	
	print('	- Surface fluxes and SST')
	fig, ax = plt.subplots(2,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax[1,1].plot(X/1000,np.ones(X.shape)*E0_w/E_star,c='r',ls='-')
	ax[1,1].plot(X/1000,np.ones(X.shape)*E0_c/E_star,c='b',ls='-')
	ax[1,1].plot(X/1000,dsmean.E0/E_star,c='k',ls='-')
	ax[1,1].set_ylabel(r"$<wr_v>$/$E^*$") # Normalized \nlatent heat flux E0 (g/kg m/s)
	#ax[1,1].tick_params(axis='both',labelbottom=False)
	ax[1,1].set_xlim([X[0]/1000,X[-1]/1000])
	ax[1,1].set_xlabel('X (km)')
	ax[0,1].plot(X/1000,np.ones(X.shape)*Q0_w/Q_star,c='r',ls='-')
	ax[0,1].plot(X/1000,np.ones(X.shape)*Q0_c/Q_star,c='b',ls='-')
	ax[0,1].plot(X/1000,dsmean.Q_star/Q_star,c='k',ls='-')
	ax[0,1].set_ylabel(r"$<w\theta>$/$Q^*$") # Normalized \nsensible heat flux Q0 (K m/s)
	#ax[0,1].tick_params(axis='both',labelbottom=False)
	ax[0,1].set_xlim([X[0]/1000,X[-1]/1000])
	ax[0,1].set_xlabel('X (km)')
	ax[1,0].plot(X/1000,np.ones(X.shape)*u_star_w/u_norm,c='r',ls='-')
	ax[1,0].plot(X/1000,np.ones(X.shape)*u_star_c/u_norm,c='b',ls='-')
	ax[1,0].plot(X/1000,u_star/u_norm,c='k',ls='-')
	ax[1,0].set_ylabel(r"$<uw>$/$u^{*2}$") # Normalized \nfriction velocity
	ax[1,0].set_xlim([X[0]/1000,X[-1]/1000])
	ax[1,0].set_xlabel('X (km)')
	ax[0,0].plot(X/1000,dataSST-273,c='k',label='SST',ls='--')
	ax[0,0].plot(X/1000,Tm[0,:]-273,c='k',label=r'$T_{z=1m}$',ls='-')
	ax[0,0].plot(X/1000,np.ones(X.shape)*298.05-273,c='r',ls='--') #,label='ref: warm'
	ax[0,0].plot(X/1000,np.ones(X.shape)*T1M_w-273,c='r',ls='-') #,label='ref: warm'
	ax[0,0].plot(X/1000,np.ones(X.shape)*296.55-273,c='b',ls='--') #,label='ref: cold'
	ax[0,0].plot(X/1000,np.ones(X.shape)*T1M_c-273,c='b',ls='-') #,label='ref: warm'
	ax[0,0].set_xlim([X[0]/1000,X[-1]/1000])
	ax[0,0].legend(loc='upper right')
	#ax[0,0].tick_params(axis='both',labelbottom=False)
	ax[0,0].set_ylabel('T (°C)')
	ax[0,0].set_xlabel('X (km)')
	
	for axe in ax.flatten():	
		axe.xaxis.label.set_fontsize(13)
		axe.yaxis.label.set_fontsize(13)
		
	fig.savefig(path_save+'surf_flx_X.png')
	print('	     surf values :')
	print('		S1 (cold) Qv0 =',np.round(dsmean.Qv_star[indnormAtX].values,5),' K m/s')
	print('		S1 (cold) E0 =',np.round(E_star*1000,5),' g/kg m/s')
	print('		S1 (cold) Q0 =',np.round(Q0[indnormAtX].values,5),' K m/s')
	print('		S1 (cold) u* =',np.round(u_norm,5),' m/s')
	print('		S1 (warm) Qv0 =',np.round(dsmean.Qv_star[indnormAtX2].values,5),' K m/s')
	print('		S1 (warm) E0 =',np.round(dsmean.E0[indnormAtX2].values*1000,5),' g/kg m/s')
	print('		S1 (warm) Q0 =',np.round(dsmean.Q_star[indnormAtX2].values,5),' K m/s')
	print('		S1 (warm) u* =',np.round(dsmean.u_star[indnormAtX2].values,5),' m/s')
	print('		cold SST= 296.55K')
	print('		cold E0 =',np.round(E0_c*1000,5),' g/kg m/s')
	print('		cold Q0 =',np.round(Q0_c,5),' K m/s')
	print('		cold Qv0 =',np.round(WTHTVs_c,5),' K m/s')
	print('		cold u* =',np.round(u_star_c,5),' m/s')
	print('		warm SST= 298.05K')
	print('		warm E0 =',np.round(E0_w*1000,5),' g/kg m/s')
	print('		warm Q0 =',np.round(Q0_w,5),' K m/s')
	print('		warm Qv0 =',np.round(WTHTVs_w,5),' K m/s')
	print('		warm u* =',np.round(u_star_w,5),' m/s')
	
	print('	- Initial conditions')
	# THT = dsINI.THT[0,nhalo:-nhalo,1,1]
	# RV = dsINI.RVT[0,nhalo:-nhalo,1,1]
	# THTV = Compute_THTV(THT,RV)
	# U = dsINI.UT[0,nhalo:-nhalo,1,1]
	# V = dsINI.VT[0,nhalo:-nhalo,1,1]
	# fig, ax1 = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
	# ax1_b = ax1.twiny()
	# ax1.plot(THT[1:2],Z[1:2],c='aqua',label=r'$r_v$ (g/kg)') # dummy field for nice legend
	# ax1.plot(THT[1:],Z[1:],c='r',label=r'$\theta$ (K)')
	# ax1_b.plot(RV[1:]*1000,Z[1:],c='aqua',label=r'$\r_v$ (g/kg)')
	# ax1.plot(THTV[1:],Z[1:],c='orange',label=r'$\theta_v$ (K)')
	# ax1.set_ylim([0,2000])
	# ax1.set_xlabel(r'$\theta$ and $r_v$')
	# ax1.set_ylabel('Altitude (m)')
	# ax1.legend(loc='upper right')
	# fig.savefig(path_save+'CI.png')
	
	# plot 2D var - var(x=0) with profiles next to it
	liste_x = np.array([4,7.5,15,20,26,38]) #0,5,10,15,20,30,35
	refAtX = 4 #km
	indxREF = np.argmin(np.abs(X.values-refAtX*1000))
	res = 50
	indx = []
	for x in range(len(liste_x)):
		indx.append(np.argmin(np.abs(X.values-liste_x[x]*1000)))
	U0 = dsmean.Um[:,indxREF]
	U0.expand_dims(dim={'ni':X},axis=1)
	dU = dsmean.Um - U0
	THV0 = dsmean.THTvm[:,indxREF]
	THV0 = THV0.expand_dims(dim={'ni':X},axis=1)
	dTHTV = dsmean.THTvm - THV0
	TH0 = dsmean.THTm[:,indxREF]
	TH0 = TH0.expand_dims(dim={'ni':X},axis=1)
	dTHT = dsmean.THTm - TH0
	RV0 = dsmean.RVTm[:,indxREF]
	RV0.expand_dims(dim={'ni':X},axis=1)
	dRV = dsmean.RVTm - RV0
	Pref = dsINI.PABST[0,nhalo:-nhalo,1,nhalo:-nhalo]
	P_prime = dsmean.Pm - Pref
	P_prime0 = P_prime[:,indxREF]
	P_prime0.expand_dims(dim={'ni':X},axis=1)
	dP_prime = P_prime - P_prime0 
	
	#crit_value = 296.25 # K handled globally
	map1 = 'winter'
	map2 = 'Reds'
	
	colorsX = DISCRETIZED_2CMAP_2(map1,map2,liste_x*1000,dataSST,crit_value,dsB.ni[nhalo:-nhalo].values) # if more than 2 x positions
	#colorsX = ['lightblue','tomato'] # if only at 2 X positions
	
	# dU
	if False:
		print('	- U-U(x='+str(refAtX)+'km) and profiles of U')
		fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
		ax[0].set_title('U-U(x=0) (m/s)')
		s = ax[0].pcolormesh(X/1000,Z/zi,dU,cmap='bwr',vmin=-1,vmax=1)
		plt.colorbar(s,ax=ax[0])
		#ax[0].contour(X/1000,Z/zi,gTHT,levels=[gTHT_contour],colors='grey',linestyles='--')
		ax[0].set_ylabel(NAMEY)
		ax[0].set_xlabel('X (km)')
		ax[0].set_ylim(YLIM)
		for kx,indice in enumerate(indx):
			ax[1].plot(dsmean.Um[:,indice],Z/zi,c=colorsX[kx],label='x='+str(liste_x[kx])+'km')
		ax[1].plot(dsref['warm']['nomean']['Mean']['MEAN_U'][indt_w,:],Z/zi_w,c='red',ls='--',label='ref:warm')
		ax[1].plot(dsref['cold']['nomean']['Mean']['MEAN_U'][indt_c,:],Z/zi_c,c='blue',ls='--',label='ref:cold')
		ax[1].set_ylim(YLIM)
		ax[1].set_xlim([5,7.6])
		ax[1].legend()
		ax[1].tick_params(axis='both',labelleft=False)
		ax[1].set_title('U profiles (m/s)')
		Add_SST_ticks(ax[0],0.02)
		fig.savefig(path_save+'dU_Uprofiles_FILE.png')

	# dTHTv
	if False:
		print('	- THv-THv(x='+str(refAtX)+'km) and profiles of THv')
		fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	#	ax[0].set_title(r'$\theta_v$-$\theta_{v0}$ (K)')
	#	s = ax[0].pcolormesh(X/1000,Z/zi,dTHTV,cmap='bwr',vmin=-0.2,vmax=0.2)
	#	plt.colorbar(s,ax=ax[0])
		ax[0].set_title(r'<$\theta_v$> (K)')
		CS = ax[0].contour(X/1000,Z/zi,dsmean.THTvm,levels=np.arange(297.6,297.8,0.01),colors='k')
		ax[0].clabel(CS,inline=True, fontsize=8,inline_spacing=0,manual=False) # , levels = [297.68,297.69,297.70,297.71,297.72]
		#CS = ax[0].contourf(X/1000,Z/zi,dsmean.THTvm,levels=np.arange(297.6,297.8,0.01),cmap='rainbow')
		#plt.colorbar(CS,ax=ax[0])
		ax[0].set_ylabel(NAMEY)
		ax[0].set_xlabel('X (km)')
		ax[0].set_ylim(YLIM)
		for kx,indice in enumerate(indx):
			ax[1].plot(dsmean.THTvm[:,indice],Z/zi,c=colorsX[kx],label='x='+str(liste_x[kx])+'km')			
		print(Compute_THTV(dsref['warm']['nomean']['Mean']['MEAN_TH'],dsref['warm']['nomean']['Mean']['MEAN_RV'])[indt_w,:].shape)
		print(Z.shape)
		ax[1].plot(Compute_THTV(dsref['warm']['nomean']['Mean']['MEAN_TH'],dsref['warm']['nomean']['Mean']['MEAN_RV'])[indt_w,:],
				Z/zi_w,c='red',ls='--',label='ref:warm')
		ax[1].plot(Compute_THTV(dsref['cold']['nomean']['Mean']['MEAN_TH'],dsref['warm']['nomean']['Mean']['MEAN_RV'])[indt_c,:],
				Z/zi_c,c='blue',ls='--',label='ref:cold')
		ax[1].set_ylim(YLIM)
		ax[1].set_xlim([297.2,298.5])
		ax[1].legend()
		ax[1].tick_params(axis='both',labelleft=False)
		ax[1].set_title(r'$\theta_v$ profiles (K)')
		Add_SST_ticks(ax[0],0.02)
		fig.savefig(path_save+'dTHv_THvprofiless_FILE.png')

	# dTHT
	if False:
		print('	- TH-TH(x='+str(refAtX)+'km) and profiles of TH')
		fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
		ax[0].set_title(r'$\theta$-$\theta_{0}$ (K)')
		s = ax[0].pcolormesh(X/1000,Z/zi,dTHT,cmap='bwr',vmin=-0.4,vmax=0.4)
		plt.colorbar(s,ax=ax[0])
		#ax[0].contour(X/1000,Z/zi,gTHT,levels=[gTHT_contour],colors='grey',linestyles='--')
		ax[0].set_ylabel(NAMEY)
		ax[0].set_xlabel('X (km)')
		ax[0].set_ylim(YLIM)
		for kx,indice in enumerate(indx):
			ax[1].plot(dsmean.THTm[:,indice],Z/zi,c=colorsX[kx],label='x='+str(liste_x[kx])+'km')
		ax[1].plot(dsref['warm']['nomean']['Mean']['MEAN_TH'][indt_w,:],Z/zi_w,c='red',ls='--',label='ref:warm')
		ax[1].plot(dsref['cold']['nomean']['Mean']['MEAN_TH'][indt_c,:],Z/zi_c,c='blue',ls='--',label='ref:cold')
		ax[1].set_ylim(YLIM)
		ax[1].set_xlim([295.4,296.5])
		ax[1].legend()
		ax[1].tick_params(axis='both',labelleft=False)
		ax[1].set_title(r'$\theta$ profiles (K)')
		Add_SST_ticks(ax[0],0.02)
		fig.savefig(path_save+'dTH_THprofiles_FILE.png')
	# dRv
	if False:
		print('	- Rv-Rv(x='+str(refAtX)+'km) and profiles of Rv')
		fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
		ax[0].set_title(r'$r_v$-$r_{v0}$ (g/kg)')
		s = ax[0].pcolormesh(X/1000,Z/zi,dRV*1000,cmap='PuOr',vmin=-0.5,vmax=0.5)
		plt.colorbar(s,ax=ax[0])
		#ax[0].contour(X/1000,Z/zi,gTHT,levels=[gTHT_contour],colors='grey',linestyles='--')
		ax[0].set_ylabel(NAMEY)
		ax[0].set_xlabel('X (km)')
		ax[0].set_ylim(YLIM)
		for kx,indice in enumerate(indx):
			ax[1].plot(dsmean.RVTm[:,indice]*1000,Z/zi,c=colorsX[kx],label='x='+str(liste_x[kx])+'km')
		ax[1].plot(dsref['warm']['nomean']['Mean']['MEAN_RV'][indt_w,:]*1000,Z/zi_w,c='red',ls='--',label='ref:warm')
		ax[1].plot(dsref['cold']['nomean']['Mean']['MEAN_RV'][indt_c,:]*1000,Z/zi_c,c='blue',ls='--',label='ref:cold')
		ax[1].set_ylim(YLIM)
		ax[1].set_xlim([10,13])
		ax[1].legend()
		ax[1].tick_params(axis='both',labelleft=False)
		ax[1].set_title(r'$r_v$ profiles (g/kg)')
		Add_SST_ticks(ax[0],0.02)
		fig.savefig(path_save+'dRV_RVprofiles_FILE.png')

def PROFILES_AT_X_THTV_THVWFLX(dataSST,dsflx,dsmean,dsref,X,Z,X_liste,Q0_atX,NORM,crit_value,path_save,dpi):
	"""This procedure plot the profiles of thtv and the total turbulent vertical flux of thtv.
		1 color for cold part, 1 color for warm part. Base on the 'crit_value' the separate domains 
		where the SST is "cold" to where it is "warm"
		
		INPUTS: 
			- dataSST: 1D SST data
			- dsflx: custom built file with mean flux
			- dsmean: custom built file with mean prognostic variables
			- dsref: reference profiles for homogeneous simulations
			- X: X dimension
			- Z: Z dimension
			- X_liste: liste of X positions (in meters)
			- Q0_atX: X position for normalisation of the flux
			- NORM: wether to normalize or not
			- crit_value: SST threshold that differentiate 'warm' SST from 'cold' SST
			- path_save: where to save the figures
			- dpi: dot per inches
		OUTPUTS:
			- A figure with profiles at some X position of: THTV and wthtv
	"""
	
	
	ind_Q0_atX = np.argmin(np.abs(X.values-Q0_atX))
	indt_c = -1 # app. the same instant as S1
	indt_w = -1 # app. the same instant as S1
	ABLH=600 # m
	
	THT_c = dsref['cold']['nomean']['Mean']['MEAN_TH'][indt_w,:]
	THT_w = dsref['warm']['nomean']['Mean']['MEAN_TH'][indt_w,:]
	RV_c = dsref['cold']['nomean']['Mean']['MEAN_RV'][indt_w,:]
	RV_w = dsref['warm']['nomean']['Mean']['MEAN_RV'][indt_w,:]
	THTV_c = Compute_THTV(THT_c,RV_c) 
	THTV_w = Compute_THTV(THT_w,RV_w)

	zi,zi_c,zi_w = (  ABLH, 
			Z[int(np.argmax(THT_c.differentiate('level_les').values))].values,
			Z[int(np.argmax(THT_w.differentiate('level_les').values))].values   ) # max of Dtheta/dz

	RES_WTHV_c = dsref['cold']['nomean']['Resolved']['RES_WTHV'][indt_c,:]
	RES_WTHV_w = dsref['warm']['nomean']['Resolved']['RES_WTHV'][indt_w,:]
	SBG_WTHV_c = THTV_c/THT_c*dsref['cold']['nomean']['Subgrid']['SBG_WTHL'][indt_c,:] + 0.61*THT_c*dsref['cold']['nomean']['Subgrid']['SBG_WRT'][indt_c,:]
	SBG_WTHV_w = THTV_w/THT_w*dsref['warm']['nomean']['Subgrid']['SBG_WTHL'][indt_w,:] + 0.61*THT_w*dsref['warm']['nomean']['Subgrid']['SBG_WRT'][indt_w,:]
	WTHV_c = RES_WTHV_c + SBG_WTHV_c
	WTHV_w = RES_WTHV_w + SBG_WTHV_w
	flux_wthtv_hand = dsflx.FLX_THvW
	thtv_hand = dsmean.THTvm
	
	
	
	mixed_thtv = thtv_hand[np.argmin(np.abs(Z.values-0.1*zi)):np.argmin(np.abs(Z.values-0.7*zi)),:].mean()
	mixed_thtv_c = THTV_c[np.argmin(np.abs(Z.values-0.1*zi_c)):np.argmin(np.abs(Z.values-0.7*zi_c))].mean()
	mixed_thtv_w = THTV_w[np.argmin(np.abs(Z.values-0.1*zi_w)):np.argmin(np.abs(Z.values-0.7*zi_w))].mean()
	
	print('	mixed thtv [0.1,0.7]zi :',mixed_thtv.values,mixed_thtv_c.values,mixed_thtv_w.values)
	
	indx = []
	for kx in range(len(X_liste)):
		indx.append(np.argmin(np.abs(X.values-X_liste[kx])))
	QV0 = np.abs(dsmean.Qv_star[ind_Q0_atX])
	map1 = 'winter'
	map2 = 'Reds'
	if NORM:
		normFLX=QV0
		QV_c = WTHV_c[0]
		QV_w = WTHV_w[0]
		borne_flx = [-1,3.5]
		bornesZ = [0,1.2]
		title_flx = r"< $w \theta_v$>/$Q_v^*$"
		title_alti = r'z/z$_i$'
		print('	QV_c,QV_w:',QV_c.values,QV_w.values)
		print('	zi_c,zi_w:',zi_c,zi_w)
	else:
		normFLX=1
		QV_c,QV_w = 1,1
		zi,zi_w,zi_c = 1,1,1
		bornesZ = [0,700]
		borne_flx = [-0.005,0.025]
		title_flx = r"< $w \theta_v$>"
		title_alti = 'Altitude (m)'
	
	
	colorsX = DISCRETIZED_2CMAP_2(map1,map2,X_liste,dataSST,crit_value,X.values)	
	fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax[0].plot(THTV_c-mixed_thtv,Z/zi_c,c='b',ls='--',label='refC')	
	ax[0].plot(THTV_w-mixed_thtv,Z/zi_w,c='r',ls='--',label='refW')	
	ax[1].plot(WTHV_c/QV_c,Z/zi_c,c='b',ls='--',label='refC')	
	ax[1].plot(WTHV_w/QV_w,Z/zi_w,c='r',ls='--',label='refW')
	print('max of dthtv/dz :')
	for kx in range(0,len(X_liste)):
		ax[0].plot( thtv_hand[:,indx[kx]]-mixed_thtv,Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' ) # [:-1]
		print( 'X=',X_liste[kx],'km, dthtv/dz_max=',thtv_hand[:,indx[kx]].differentiate('level').sel(level=slice(0,1000)).max().values)
		ax[1].plot( flux_wthtv_hand[:,indx[kx]]/normFLX,Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' )
	ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax[1].xaxis.major.formatter._useMathText = True	
	ax[0].set_ylabel(title_alti)
	#ax[1].set_ylabel(title_alti)
	ax[0].set_xlabel(r'< $\theta_v$ > - $\theta_{v,mixed}$ (K)')
	ax[1].set_xlabel(title_flx)
	ax[0].set_ylim(bornesZ)
	ax[1].set_ylim(bornesZ)
	ax[0].set_xlim([-0.2,0.4]) # 297.4,298  297.6,298
	ax[1].vlines(0,0,2000,colors='grey',ls='--')
	ax[1].set_xlim(borne_flx)
	ax[1].legend(fontsize=12,loc='upper right')
	#fig.suptitle(r"Profils (x) of <$\theta_v$> and <$\overline{w'\theta_v'}$>")
	ax[0].grid()
	ax[1].grid()
	for axe in ax.flatten():	
		axe.xaxis.label.set_fontsize(13)
		axe.yaxis.label.set_fontsize(13)
	fig.savefig(path_save+'ProfilesX_THTv_wthtv.png')
	
	# normalized with local surface flux
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot(WTHV_c/QV_c,Z/zi_c,c='b',ls='--',label='refC')	
	ax.plot(WTHV_w/QV_w,Z/zi_w,c='r',ls='--',label='refW')
	for kx in range(0,len(X_liste)):
		ax.plot( dsflx.FLX_THvW[:,indx[kx]]/dsmean.Qv_star[indx[kx]],Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' )
	ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax.xaxis.major.formatter._useMathText = True	
	ax.set_ylabel(title_alti)
	ax.set_xlabel(r"< w $\theta_v$ >(x)/$Q_v^*$(x)")
	ax.vlines(0,0,2000,colors='grey',ls='--')
	ax.legend(fontsize=8,loc='upper right')
	ax.set_ylim(bornesZ)
	ax.set_xlim([-0.5,1.6])
	ax.grid()
	fig.savefig(path_save+'ProfilesX_THTv_wthtv_atX.png')
	
	# entrainment evolution along X with Q* fixed at X=4km
	E_c = ( WTHV_c/QV_c ).sel(level_les=slice(0,1.2*zi_c)).min(dim='level_les')
	E_w = ( WTHV_w/QV_w ).sel(level_les=slice(0,1.2*zi_w)).min(dim='level_les')
	E = ( flux_wthtv_hand/QV0 ).sel(level=slice(0,1.2*zi)).min(dim='level')
	fig, ax = plt.subplots(1,1,figsize = (5,3),constrained_layout=True,dpi=dpi)
	ax.hlines( E_c,0,X[-1]/1000,colors='b',label='refC',ls='--')
	ax.hlines( E_w,0,X[-1]/1000,colors='r',label='refW',ls='--')
	ax.plot( X/1000,E,c='k',label='S1')
	ax.set_xlim(X[0]/1000,X[-1]/1000)
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax.yaxis.major.formatter._useMathText = True	
	ax.set_ylabel(r"E = < w $\theta_v$ >(x)/$Q_v^*$")
	ax.set_xlabel('X (km)')
	fig.savefig(path_save+'entrainment_Xevolution.png')

	# entrainment evolution along X with Q* that changes with X.
	#	this is not valid where surface and top ABL are less related (just after the up and down fronts)
	#	but it can give info where ABL is in quasi equilibrium with surface (end of cold and warm patches)
	# 	the hatched areas are determined with the wthtv total flux plot XZ
	E = ( flux_wthtv_hand/dsmean.Qv_star ).sel(level=slice(0,1.2*zi)).min(dim='level')
	fig, ax = plt.subplots(1,1,figsize = (5,3),constrained_layout=True,dpi=dpi)
	ax.hlines( E_c,0,X[-1]/1000,colors='b',label='refC',ls='--')
	ax.hlines( E_w,0,X[-1]/1000,colors='r',label='refW',ls='--')
	ax.plot( X/1000,E,c='k',label='S1')
	ax.add_patch(mpl.patches.Rectangle((5, -0.6), 12, 0.7, hatch='/',fill=False)) # up front region, non linear
	ax.add_patch(mpl.patches.Rectangle((25, -0.6), 7, 0.7, hatch='/',fill=False)) # down front region, non linear
	ax.set_xlim(X[0]/1000,X[-1]/1000)
	ax.set_ylim([-0.6,0.1])
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax.yaxis.major.formatter._useMathText = True	
	ax.set_ylabel(r"E = < w $\theta_v$ >(x)/$Q_v^*(x)$")
	ax.set_xlabel('X (km)')
	fig.savefig(path_save+'entrainment_Xevolution_localQ.png')

def PROFILES_AT_X_U_UWFLX(dataSST,dsflx,dsmean,dsref,X,Z,X_liste,ustar_atX,NORM,crit_value,path_save,dpi):
	"""This procedure plot the profiles of U and the total turbulent vertical flux of uw.
		1 color for cold part, 1 color for warm part. Base on the 'crit_value' the separate domains 
		where the SST is "cold" to where it is "warm"
		
		INPUTS: 
			- dataSST: 1D SST data
			- dsflx: custom built file with mean flux
			- dsmean: custom built file with mean prognostic variables
			- dsref: reference profiles for homogeneous simulations
			- X: X dimension
			- Z: Z dimension
			- X_liste: liste of X positions (in meters)
			- Q0_atX: X position for normalisation of the flux
			- NORM: wether to normalize or not
			- crit_value: SST threshold that differentiate 'warm' SST from 'cold' SST
			- path_save: where to save the figures
			- dpi: dot per inches
		OUTPUTS:
			- A figure with profiles at some X position of: U and uw
	"""
	ind_ustar_atX = np.argmin(np.abs(X.values-ustar_atX))
	indt_c = -1 # app. the same instant as S1
	indt_w = -1 # app. the same instant as S1
	ABLH=600 # m
	THT_c = dsref['cold']['nomean']['Mean']['MEAN_TH'][indt_w,:]
	THT_w = dsref['warm']['nomean']['Mean']['MEAN_TH'][indt_w,:]
	RV_c = dsref['cold']['nomean']['Mean']['MEAN_RV'][indt_w,:]
	RV_w = dsref['warm']['nomean']['Mean']['MEAN_RV'][indt_w,:]
	THTV_c = Compute_THTV(THT_c,RV_c) 
	THTV_w = Compute_THTV(THT_w,RV_w)

	zi,zi_c,zi_w = (  ABLH, 
			Z[int(np.argmax(THT_c.differentiate('level_les').values))].values,
			Z[int(np.argmax(THT_w.differentiate('level_les').values))].values   ) # max of Dtheta/dz

	U_c = dsref['cold']['nomean']['Mean']['MEAN_U'][indt_c,:]
	U_w = dsref['warm']['nomean']['Mean']['MEAN_U'][indt_w,:]

	UW_c = ( dsref['cold']['nomean']['Resolved']['RES_WU'] + dsref['cold']['nomean']['Subgrid']['SBG_WU'] )[indt_c,:]
	UW_w = ( dsref['warm']['nomean']['Resolved']['RES_WU'] + dsref['warm']['nomean']['Subgrid']['SBG_WU'] )[indt_w,:]
	flux_uw_hand = dsflx.FLX_UW
	U_hand = dsmean.Um
	
	
	indx = []
	for kx in range(len(X_liste)):
		indx.append(np.argmin(np.abs(X.values-X_liste[kx])))
	u_star = np.abs(dsmean.u_star[ind_ustar_atX])
	map1 = 'winter'
	map2 = 'Reds'
	#dataSST = dsB.SST[0,1,nhalo:-nhalo].values
	if NORM:
		normFLX=u_star**2
		NormC = np.abs(UW_c[0])
		NormW = np.abs(UW_w[0])
		borne_flx = [-1.5,0.3]
		bornesZ = [0,1.2]
		title_flx = r"< uw >/$u^{*2}$"
		title_alti = r'z/z$_i$'
		print('	u_starC,u_starW:',np.sqrt(NormC).values,np.sqrt(NormW).values)
		print('	zi_c,zi_w:',zi_c,zi_w)
	else:
		normFLX=1
		QV_c,QV_w = 1,1
		zi,zi_w,zi_c = 1,1,1
		bornesZ = [0,700]
		borne_flx = [-0.005,0.025] # to be modified
		title_flx = r"< uw >"
		title_alti = 'Altitude (m)'
	
	
	colorsX = DISCRETIZED_2CMAP_2(map1,map2,X_liste,dataSST,crit_value,X.values)	
	fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax[0].plot(U_c,Z/zi_c,c='b',ls='--',label='refC')	
	ax[0].plot(U_w,Z/zi_w,c='r',ls='--',label='refW')	
	ax[1].plot(UW_c/NormC,Z/zi_c,c='b',ls='--',label='refC')	
	ax[1].plot(UW_w/NormW,Z/zi_w,c='r',ls='--',label='refW')
	for kx in range(0,len(X_liste)):
		ax[0].plot( U_hand[:,indx[kx]],Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' ) # [:-1]
		ax[1].plot( flux_uw_hand[:,indx[kx]]/normFLX,Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' )
	ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax[1].xaxis.major.formatter._useMathText = True	
	ax[0].set_ylabel(title_alti)
	#ax[1].set_ylabel(title_alti)
	ax[0].set_xlabel(r'< U > (m.s$^{-1}$)')
	ax[1].set_xlabel(title_flx)
	ax[0].set_ylim(bornesZ)
	ax[1].set_ylim(bornesZ)
	ax[0].set_xlim([5,7.6]) # 
	ax[1].vlines(0,0,2000,colors='grey',ls='--')
	ax[1].set_xlim(borne_flx)
	ax[0].legend(fontsize=12,loc='upper left')
	ax[0].grid()
	ax[1].xaxis.set_major_locator(MultipleLocator(0.5))
	ax[1].grid(True,'major')
	for axe in ax.flatten():	
		axe.xaxis.label.set_fontsize(13)
		axe.yaxis.label.set_fontsize(13)
	#fig.suptitle(r"Profils (x) of <$\theta_v$> and <$\overline{w'\theta_v'}$>")
	fig.savefig(path_save+'ProfilesX_U_uw.png')
	
	# normalized with local surface flux
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot(UW_c/NormC,Z/zi_c,c='b',ls='--',label='refC')	
	ax.plot(UW_w/NormW,Z/zi_w,c='r',ls='--',label='refW')
	for kx in range(0,len(X_liste)):
		ax.plot( flux_uw_hand[:,indx[kx]]/dsmean.u_star[indx[kx]]**2,Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' )
	ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax.xaxis.major.formatter._useMathText = True	
	ax.set_ylabel(title_alti)
	ax.set_xlabel(r"< uw >(x)/$u^{*2}$(x)")
	ax.vlines(0,0,2000,colors='grey',ls='--')
	ax.legend(fontsize=8,loc='upper right')
	ax.set_ylim(bornesZ)
	ax.set_xlim([-1.5,0.3])
	fig.savefig(path_save+'ProfilesX_U_uw_atX.png')

def Corr_atZ_AllInOne(X,Z,dsmean,SST,CHOICE,D_VAR,D_SST,S_SST,atZ,V_INTEG,UPFRONT_ONLY,PRESET,res,path_save,dpi):
	"""
	Compute the normalized correlation between 'CHOICE' and derivatives of SST
	
	INPUT:
		- X 		: X dimension of S1
		- Z 		: Z dimension of S1
		- dsmean	: dataset with mean fields
		- SST		: SST 1D field
		- CHOICE	: what variable to compute the correlation
		- D_VAR		: order of derivative for VAR
		- D_SST		: order of derivative for SST
		- S_SST		: sign of SST, 1 is positive, -1 is negative
		- atZ		: height for VAR
		- V_INTEG	: weither to compute vertically integrated value or not
		- UPFRONT_ONLY : compute correlation with only the 1st front
		- PRESET	: premade setup (PA,DMM,LA13)
		- res		: horizontal resolution
		- path_save	: where to save figures
		- dpi		: for figures
	
	OUTPUTS:
		- a plot of Corr(VAR,SST) vs lag, with index of max of lag
		- a plot with the variable used and the SST shifted
	"""

	if PRESET=='DMM':
		CHOICE = 'Um'
		D_VAR = 1
		D_SST = 1
		S_SST = 1
		atZ = 10
		V_INTEG = False
	elif PRESET=='PA':
		CHOICE = 'Um'
		D_VAR = 1
		D_SST = 2 
		S_SST = -1
		atZ = 10
		V_INTEG = True
	elif PRESET=='LA13': # like Lambaerts 2013
		CHOICE = 'Wm'
		D_VAR = 0
		D_SST = 2
		S_SST = -1
		atZ = 300
		V_INTEG = False
	elif PRESET=='DMMtau':
		CHOICE = 'Tau'
		D_VAR = 1
		D_SST = 1
		S_SST = 1
		atZ = 10
		V_INTEG = False
	indz = nearest(Z.values,atZ)	
	#
	# > Variable selector	 (TBD: include momentum budget term for ex.)	
	if CHOICE in ['Um','Vm','Wm']:
		if V_INTEG:
			VAR = 1/ABLH_S1 * dsmean[CHOICE][:,:].integrate('level')
			string_integ = 'integ'
		else:
			VAR =  dsmean[CHOICE][indz,:]
			string_integ = str(atZ)
	elif CHOICE in ['Tau','u_star']:
		VAR = dsmean[CHOICE]
		string_integ = ''	
	#
	# > getting VAR
	if D_VAR==0:
		#VAR = dsmean[CHOICE][indz,:]
		name,nicename = CHOICE, CHOICE
	elif D_VAR==1:
		#VAR = dsmean[CHOICE][indz,:].differentiate('ni')
		VAR = VAR.differentiate('ni')
		name,nicename = 'g'+CHOICE, r'$\bigtriangledown$'+CHOICE
	elif D_VAR==2:
		#VAR = dsmean[CHOICE][indz,:].differentiate('ni').differentiate('ni')
		VAR = VAR.differentiate('ni').differentiate('ni')
		name,nicename = 'gg'+VAR, r'$\bigtriangleup$'+CHOICE
	#	
	# >getting SST variable	
	if D_SST==0:
		X1 = SST
		nameSST,nicenameSST = 'SST', 'SST'
	if D_SST==1:
		X1 = SST.differentiate('ni')
		nameSST,nicenameSST = 'gSST', r'$\bigtriangledown$SST'
	elif D_SST==2:
		X1 = SST.differentiate('ni').differentiate('ni')
		nameSST,nicenameSST = 'ggSST', r'$\bigtriangleup$SST'
		
	if S_SST==-1:
		X1 = -X1	
		nameSST,nicenameSST = '-'+nameSST,'-'+nicenameSST
	
	if UPFRONT_ONLY:
		name_front = 'CtoW'
	else:
		name_front = 'CtoW_WtoC'
	
	
	name_bundle = name+string_integ+'_'+nameSST
	Savename_SST = 		path_save + name_front + '_SSTonly_'+ name_bundle
	Savename_corr = 	path_save + name_front + '_Corr_'	+ name_bundle
	Savename_Shifted = 	path_save + name_front + '_Shifted_'+ name_bundle
	Savename_Var_vs_X1 = path_save + name_front + '_VS_'	+ name_bundle
	Savename_RegLin = 	path_save + name_front + '_RegLin_'	+ name_bundle
	
	
	if PRESET=='paper':
	
		SaveName = path_save + name_front + '_gU10_gTau_vs_gSST.png'
		atZ = 10 # m
		XMAX = 23 # km
		
		indz = nearest(Z.values,atZ)	
		VAR1 = dsmean['Um'][indz,:].differentiate('ni')
		VAR2 = dsmean['Tau'].differentiate('ni')
		name1,nicename1 = 'gU10', r'$\partial_x$U$_{10}$'
		name2,nicename2 = 'gTau', r'$\partial_x \tau$'
		X1 = SST.differentiate('ni')
		nameSST,nicenameSST = 'gSST', r'$dSST$/$dx$' # r'$\frac{dSST}{dx}$'
		
		indx_borne = nearest(X.values,23*1000)
		X1[200:] = X1[200] 		# only the first front
		X1 = X1[:indx_borne] 	# size = indx_borne
		VAR1 = VAR1[:indx_borne]
		VAR2 = VAR2[:indx_borne]
		
		rho1 = RCorr_nocyclic(VAR1.values,X1.values)
		indlag1 = np.argmax(rho1)
		X1_rolled1 = np.roll(X1,indlag1)
		print('corr max for gU10 is ',np.max(rho1))
		print('lagmax for gU10 is ',indlag1*res/1000,'km')
		rho2 = RCorr_nocyclic(VAR2.values,X1.values)
		indlag2 = np.argmax(rho2)
		X1_rolled2 = np.roll(X1,indlag2)
		print('corr max for gTau is ',np.max(rho2))
		print('lagmax for gTau is ',indlag2*res/1000,'km')
		
		fig, ax = plt.subplots(1,1,figsize = (5.5,3),constrained_layout=True,dpi=dpi)
		ax3 = ax.twinx()
		ax2 = ax.twinx()
		ax2.spines.right.set_position(("axes", 1.23))
		
		p3 = ax.plot(X[:indx_borne]/1000,VAR1.values*1e5,c='b')
		p2 = ax2.plot(X[:indx_borne]/1000,VAR2.values*1e6,c='g')
		#p1 = ax.plot(X[:indx_borne]/1000,np.roll(X1,indlag1)*1e4,c='k',label='lag='+str(indlag1*res/1000)+'km')
		#ax3.plot(X[:indx_borne]/1000,np.roll(X1,indlag2),c='grey',label='lag='+str(indlag2*res/1000)+'km, R='+str(np.max(rho2)) )
		p1 = ax.plot(X[:indx_borne]/1000,X1*1e4,c='k',ls='-',label='no lag')
		
		ax.set_xlabel('X (km)')
		ax3.set_ylabel(nicename1) #+ r' (10$^{-5}$ s$^{-1}$)')
		ax2.set_ylabel(nicename2) #+ r' (10$^{-6}$ N.m$^{-3}$)')
		ax.set_ylabel(nicenameSST)# + r' (10$^{-4}$ K.m$^{-1}$)')
		
		ax.yaxis.label.set_color(p1[0].get_color())
		ax2.yaxis.label.set_color(p2[0].get_color())
		ax3.yaxis.label.set_color(p3[0].get_color())

		ax.tick_params(axis='y', colors=p1[0].get_color())
		ax2.tick_params(axis='y', colors=p2[0].get_color())
		ax3.tick_params(axis='y', colors=p3[0].get_color())
		
		ax.set_xlim([0,23])
		
		for axe in [ax,ax2,ax3]:
			yabs_max = abs(max(axe.get_ylim(), key=abs))
			axe.set_ylim(ymin=-yabs_max/2, ymax=yabs_max)
		
		#ax.legend()		
		fig.savefig(SaveName)
		
	else:	
		if UPFRONT_ONLY:
			# here only the cold-to-warm is taken into account
			# > cutting SST
			XMAX = 23 # km
			indx_borne = nearest(X.values,23*1000)
			X1[200:] = X1[200] 		# only the first front
			X1 = X1[:indx_borne] 	# size = indx_borne
			VAR = VAR[:indx_borne]
			# > Correlation
			rho = RCorr_nocyclic(VAR.values,X1.values)
			indlag = np.argmax(rho)
			X1_rolled = np.roll(X1,indlag)
			print('corr max is ',np.max(rho))
			print('lagmax is ',indlag*res/1000,'km')
			# > linear regression
			
			
			# > Plot
			colorsX = DISCRETIZED_2CMAP_2('winter','plasma',X[:indx_borne].values,SST[:indx_borne].values,297.3,X[:indx_borne].values)
			# SST
			fig, ax = plt.subplots(1,1,figsize = (3,3),constrained_layout=True,dpi=dpi)
			ax.scatter(X[:indx_borne]/1000,SST[:indx_borne],marker='x',c=colorsX)
			ax.set_xlabel('X (km)')
			ax.set_ylabel('SST')
			fig.savefig(Savename_SST+'.png')
			
			# Correlation coefficient
			fig, ax = plt.subplots(1,1,figsize = (3,3),constrained_layout=True,dpi=dpi)
			ax.plot(X[:indx_borne]/1000,rho)
			ax.set_ylabel('rho')
			ax.set_xlabel('lag')
			ax.set_title('imax='+str(indlag)+' CorrMax='+str(np.round(rho[indlag],2)))
			fig.savefig(Savename_corr+'.png')
			
			# VAR and SST Shifted vs X
			fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
			ax.scatter(X[:indx_borne]/1000,VAR.values,marker='x',c=colorsX)
			ax2 = ax.twinx()
			ax2.plot(X[:indx_borne]/1000,np.roll(X1,indlag),c='k',label='lag='+str(indlag*res/1000)+'km')
			if PRESET=='DMMtau':
				ax2.plot(X[:indx_borne]/1000,X1,c='k',ls='--',label='no lag')
			ax.set_xlabel('X (km)')
			ax.set_ylabel(nicename)
			ax2.set_ylabel(nicenameSST)
			yabs_max = abs(max(ax.get_ylim(), key=abs))
			ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
			yabs_max = abs(max(ax2.get_ylim(), key=abs))
			ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
			ax2.legend()
			fig.savefig(Savename_Shifted+'.png')
			
			# VAR vs X1
			fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
			ax.scatter(np.roll(X1,indlag),VAR.values,marker='x',c=colorsX)
			ax.set_xlabel(nicenameSST+' shifted')
			ax.set_ylabel(nicename)
			fig.savefig(Savename_Var_vs_X1+'.png')
			fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
			ax.scatter(X1,VAR.values,marker='x',c=colorsX)
			ax.set_xlabel(nicenameSST+' no shift')
			ax.set_ylabel(nicename)
			
			# linear regression
			if PRESET=='DMM' or PRESET=='DMMtau':
				seuilGradSST = 1e-4
				VAR_reg = np.ma.masked_where( np.abs(X1_rolled) < seuilGradSST, VAR)
				X1_reg = np.ma.masked_where( np.abs(X1_rolled) < seuilGradSST, X1_rolled)
				a, b, r, p_value, std_err = linregress(X1_reg.compressed(),VAR_reg.compressed() )
				print('linear regression:')
				print("a   ={:8.5f},b   ={:8.3f},r^2 ={:8.5f}".format(a, b, r**2))				
				fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
				ax.scatter(X1_reg.compressed(),VAR_reg.compressed(),marker='x',c='k')
				ax.plot(X1_reg.compressed(), a*X1_reg.compressed() + b,c='r',label="y={:8.5f}x+b,r^2={:8.5f}".format(a, r**2))
				ax.set_xlabel(nicenameSST+' (shifted) gradSST>'+str(seuilGradSST))
				ax.set_ylabel(nicename)
				ax.legend()
				fig.savefig(Savename_RegLin+'.png')
				
				if PRESET=='DMMtau':
					seuilGradSST = 2e-4
					VAR_reg = np.ma.masked_where( np.abs(X1) < seuilGradSST, VAR)
					X1_reg = np.ma.masked_where( np.abs(X1) < seuilGradSST, X1)
					a, b, r, p_value, std_err = linregress(X1_reg.compressed(),VAR_reg.compressed() )
					print('linear regression at no lag:')
					print("a   ={:8.5f},b   ={:8.3f},r^2 ={:8.5f}".format(a, b, r**2))
					fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
					ax.scatter(X1_reg.compressed(),VAR_reg.compressed(),marker='x',c='k')
					ax.plot(X1_reg.compressed(), a*X1_reg.compressed() + b,c='r',label="y={:8.5f}x+b,r^2={:8.5f}".format(a, r**2))
					ax.set_xlabel(nicenameSST+' gradSST>'+str(seuilGradSST))
					ax.set_ylabel(nicename)
					ax.legend()
		else:
			# here the cold-to-warm and warm-to-cold are taken into account
			# Note : linear regression has not meaning in this case (see plot VAR vs XA)
			
			# > computing normalized correlation
			Corr_coef = RCorr_cyclic(VAR.values,X1.values)
			imax = np.argmax(Corr_coef)
			print('	lag at max correlation is:',imax*res/1000,'km')
			# > Shifting SST
			Shifted_SST = np.zeros(len(SST))
			Shifted_SST[imax: ] = X1[0:len(SST)-imax]
			Shifted_SST[:imax] = X1[len(SST)-imax:]
			
			# > Plots
			# Correlation
			fig, ax = plt.subplots(1,1,figsize = (3,3),constrained_layout=True,dpi=dpi)
			ax.plot(X/1000,Corr_coef,c='k')
			ax.set_xlabel('X lag (km)')
			ax.set_xlim([0,38.4])
			ax.set_ylim(-1,1)
			ax.hlines(0.5,0,40,colors='r')
			ax.hlines(-0.5,0,40,colors='r')
			ax.set_ylabel('Corr('+nicename+','+nicenameSST+')')
			ax.set_title('imax='+str(imax)+' CorrMax='+str(np.round(Corr_coef[imax],2)))
			fig.savefig(Savename_corr+'.png')
				
			# VAR and SST Shifted vs X
			fig, ax = plt.subplots(1,1,figsize = (7,5),constrained_layout=True,dpi=dpi)
			ax.plot(X[0]/1000,VAR[0],c='b',label=nicenameSST) # dummy 
			ax.plot(X[0]/1000,VAR[0],c='r',label=nicenameSST+' shifted') # dummy 
			ax.plot(X/1000,VAR,c='k',label=nicename)
			ax2 = ax.twinx()
			ax2.plot(X/1000,X1,c='b',label=nicenameSST)
			ax2.plot(X/1000,Shifted_SST,c='r',label=nicenameSST+' shifted')
			ax.set_xlabel('X (km)')
			ax.set_ylabel(nicename)
			ax2.set_ylabel(nicenameSST)
			ax.legend()
			fig.savefig(Savename_Shifted+'.png')	
			
			# Note: this plot has no meaning, to be replaced with VAR vs X1
			# Relation coefficient at max correlation lag
	#		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	#		ax.plot(X/1000,alpha,c='k')
	#		ax.set_xlim([0,38.4])
	#		ax.set_ylim([0,0.5])
	#		ax.set_xlabel('X (km)')
	#		ax.set_ylabel(r'$\alpha$')
	#		ax.set_title(nicename + r'= $\alpha$' + nicenameSST)
	#		fig.savefig(path_save+'regCoeff_'+string_integ+name+'_'+nameSST+'_atz'+str(atZ)+'m.png')	
	
# after rev1
def RollFactor(VAR,atZi,dsO_i,dico_dsrefO,path_save,dpi):
	"""
	Computes the Roll Factor from Salesky et al. 2017. And plot the variable + the polar autocorrelation function


	INPUT:

	OUTPUT:

	"""
	dsWref = dico_dsrefO['warm'] # mass_interpolator(xr.open_dataset(path_warm_ref))
	dsCref = dico_dsrefO['cold'] # mass_interpolator(xr.open_dataset(path_cold_ref))
	res = 50 # m
	
	zi_c = 600 #m
	zi_w = 750 #m
	
	atZw = atZi*zi_w # m, altitude where to compute the roll factor
	atZc = atZi*zi_c # m, altitude where to compute the roll factor
	coldslice = slice(30000,38000)
	warmslice = slice(15000,23000)

	if atZi < 0:
		atZw,atZc = -atZi,-atZi
		nameZ = 'Z_'+str(-atZi)+'m'
	else:
		nameZ = 'zZi_'+str(atZi)
	if VAR=='UT':
		VARmin,VARmax = -1,1
	elif VAR=='WT':
		VARmin,VARmax = -1,1

	print('		-> variable:',VAR,'altitude:',nameZ)

	dico = {'cold SST':{'var':dsO_i[VAR][0],
								'altitude':atZc,
								'ni_slice':coldslice,
								'nice_name':'cold'},
			'warm SST':{'var':dsO_i[VAR][0],
								'altitude':atZw,
								'ni_slice':warmslice,
								'nice_name':'warm'},
			'ref cold SST':{'var': dsCref[VAR][0],
								'altitude':atZi*531,
								'ni_slice':slice(0,8000),
								'nice_name':'Refcold'},
			'ref warm SST':{'var': dsWref[VAR][0],
								'altitude':atZi*710,
								'ni_slice':slice(0,8000),
								'nice_name':'Refwarm'},
			}

	for case in dico.keys():
		ni_slice = dico[case]['ni_slice']
		altitude = dico[case]['altitude']
		nicename = dico[case]['nice_name']
		W = dico[case]['var'].sel(ni=ni_slice).sel(level=altitude,method='nearest')
		units = W.attrs['units']
		W = W - W.mean()
		R_ww,lag_tht,lag_r = Corr_in_polar(W,res)
		R = Roll_factor(R_ww,lag_r,zi=600) 
		print('Roll factor is ('+case+')=',R)
		fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
		s = ax[0].pcolormesh(W.ni/1000,W.nj/1000,W,cmap='seismic',vmin=VARmin,vmax=VARmax)
		plt.colorbar(s,ax=ax[0])
		ax[0].set_xlabel('X (km)')
		ax[0].set_ylabel('Y (km)')
		ax[0].set_title(VAR+"' ("+units+') at '+nameZ)
		s = ax[1].contourf(lag_r/1000,lag_tht*180/np.pi,R_ww,cmap='seismic',levels=np.linspace(-0.4,0.4,20),extend='both')
		plt.colorbar(s,ax=ax[1])
		ax[1].set_ylim([-180,180])
		ax[1].yaxis.set_major_locator(ticker.MultipleLocator(45))
		ax[1].set_ylabel(r'$r_{\theta}$')
		ax[1].set_xlabel(r'$r_{\rho}$ (km)')
		ax[1].set_title(case+', R='+str(np.round(R,2)))
		fig.savefig(path_save+'RollFactor_'+nameZ+'_'+VAR+'_'+nicename+'.png')

def meanV_and_Vbudget(X,Z,atX,atZ,listeX,L_c,dsflx,dsmean,SST,dsref,path_save,dpi):
	"""
	"""
	indx = nearest(X.values,atX*1000)
	indz = nearest(Z.values,atZ)
	term_turb_V = - dsflx.FLX_VW.differentiate('level')
	term_corio_V = f*(dsmean.Um-7.5)
	term_adv_V = - dsmean.Um * dsmean.Vm.differentiate('ni')
	# terme de pression en Y qui est nul (conséquence de la moyenne)
	Pm = dsmean.Pm
	dPdX = Pm.differentiate('ni') # this is the pressure without geostrophic gradient
	Rhom = Exner(Pm)**(Cvd/Rd) * P00 / (Rd*dsmean.THTvm)
	Tm = Theta_to_T(dsmean.THTm,Pm)
	dPdX = (Rhom*Tm*Rd).differentiate('ni')
	V_equilibre_thermique = 1/(f*Rhom) * dPdX
	rho_10m = Rhom.mean('ni').isel(level=indz).values
	V_equilibre_thermique2 = 1/f * Rd*np.gradient(SST)

	# <V> flow
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	for k,Xni in enumerate(listeX):
		indni = nearest(X.values,Xni*1000)
		ax.plot(dsmean.Vm.isel(ni=indni),Z/ABLH_S1,c=L_c[k],label=str(Xni)+'km')
	ax.plot(dsref['cold']['nomean']['Mean']['MEAN_V'][-1],Z/ABLH_S1,c='b',label='cold ref',ls='--')
	ax.plot(dsref['warm']['nomean']['Mean']['MEAN_V'][-1],Z/ABLH_S1,c='r',label='warm ref',ls='--')
	ax.set_ylabel('z/zi')
	ax.vlines(0,0,2,colors='gray',linestyles='--')
	ax.set_ylim([0,1.2])
	fig.savefig(path_save+'V_flow_atX.png')

	# <V> flow along X
	fig, ax = plt.subplots(1,1,figsize = (5,3),constrained_layout=True,dpi=dpi)
	indzi = nearest(Z.values,0.1*ABLH_S1)
	ax.plot(X/1000,dsmean.Vm.isel(level=indzi),c='k')
	ax.set_ylabel('<V> at 0.1 zi')
	ax.legend()
	ax.set_xlabel('X (km)')
	fig.savefig(path_save+'V_flow_at0.1zi.png')

	# here this is the V that results from the equilibrium between
	# coriolis (fV) and pressure (1/rho * dP/dx)
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	for k,Xni in enumerate(listeX):
		indni = nearest(X.values,Xni*1000)
		ax.plot(V_equilibre_thermique.isel(ni=indni),Z/ABLH_S1,c=L_c[k],label=str(Xni)+'km')
	ax.set_ylabel('z/zi')
	ax.vlines(0,0,2,colors='gray',linestyles='--')
	ax.set_ylim([0,1.2])
	ax.legend()
	#fig.savefig(path_save_png+'V_flow_atX.png')

	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot(X/1000,V_equilibre_thermique.isel(level=indz),c='k')
	ax.set_ylabel(r'V$_{\theta}$ = 1/f * rho * dP/dx')
	ax.set_xlabel('X (km)')

	# at z = 10m, dP/dx ~ rho.Rd.dSST/dx
	# fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	# ax.plot(X/1000,V_equilibre_thermique2,c='k')
	# ax.set_ylabel(r'V$_{\theta}$ = 1/f * rho * dSST/dx')
	# ax.set_xlabel('X (km)')


	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	for k,Xni in enumerate(listeX):
		indni = nearest(X.values,Xni*1000)
		ax.plot(Rhom.isel(ni=indni),Z/ABLH_S1,c=L_c[k],label=str(Xni)+'km')
	ax.set_ylabel('z/zi')
	ax.vlines(0,0,2,colors='gray',linestyles='--')
	ax.set_ylim([0,1.2])
	ax.legend()
	ax.set_xlabel('<rho>')

	# # Z plot
	# fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	# ax.plot(term_turb_V.isel(ni=indx)*10000,Z/ABLH_S1,c='g',label='-dvw/dz')
	# ax.plot(term_corio_V.isel(ni=indx)*10000,Z/ABLH_S1,c='orange',label='f.<U>')
	# ax.plot(term_adv_V.isel(ni=indx)*10000,Z/ABLH_S1,c='k',label='-<U>.d<V>/dx')
	# ax.set_ylabel('z/zi')
	# ax.set_xlabel('V budget terms (x10e4)')
	# ax.legend()
	# ax.vlines(0,0,2,colors='gray',linestyles='--')
	# ax.set_ylim([0,1.2])
	# ax.set_title('at X = '+str(atX)+'km')	
	# fig.savefig(path_save_png+'V_budget_Z_X13km.png')	
	
	# # X plot
	# fig, ax = plt.subplots(2,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	# indz = nearest(Z.values,540)
	# ax[0].plot(X.values/1000,term_turb_V.isel(level=indz)*10000,c='g',label='-dvw/dz')
	# ax[0].plot(X.values/1000,term_corio_V.isel(level=indz)*10000,c='orange',label=r'f.(<U>-$U_g$)>')
	# ax[0].plot(X.values/1000,term_adv_V.isel(level=indz)*10000,c='k',label='-<U>.d<V>/dx')
	# ax[0].set_title('at Z = '+str(540)+'m')
	# indz = nearest(Z.values,10)
	# ax[1].plot(X.values/1000,term_turb_V.isel(level=indz)*10000,c='g',label='-dvw/dz')
	# ax[1].plot(X.values/1000,term_corio_V.isel(level=indz)*10000,c='orange',label=r'f.(<U>-$U_g$)')
	# ax[1].plot(X.values/1000,term_adv_V.isel(level=indz)*10000,c='k',label='-<U>.d<V>/dx')
	# ax[1].set_xlabel('X (km)')
	# ax[1].set_title('at Z = '+str(10)+'m')
	# for axe in ax:
	# 	axe.set_ylabel('V budget terms (x10e4)')
	# 	axe.legend()
	# 	axe.set_ylim([-7,7])
	# 	axe.hlines(0,0,40,colors='gray',linestyles='--')
	# 	axe.set_xlim([X[0]/1000,X[-1]/1000])
	# fig.savefig(path_save_png+'V_budget_X_z10_540m.png')	

def PROFILES_AT_X_RV_WRVFLX(dataSST,dsflx,dsmean,dsref,X,Z,X_liste,E0_atX,NORM,crit_value,path_save,dpi):
	"""This procedure plot the profiles of RV and the total turbulent vertical flux of rvw.
		1 color for cold part, 1 color for warm part. Base on the 'crit_value' the separate domains 
		where the SST is "cold" to where it is "warm"
		
		INPUTS: 
			- dataSST: 1D SST data
			- dsflx: custom built file with mean flux
			- dsmean: custom built file with mean prognostic variables
			- dsref: reference profiles for homogeneous simulations
			- X: X dimension
			- Z: Z dimension
			- X_liste: liste of X positions (in meters)
			- Q0_atX: X position for normalisation of the flux
			- NORM: wether to normalize or not
			- crit_value: SST threshold that differentiate 'warm' SST from 'cold' SST
			- path_save: where to save the figures
			- dpi: dot per inches
		OUTPUTS:
			- A figure with profiles at some X position of: rv and wrv
	"""
	ind_E0_atX = np.argmin(np.abs(X.values-E0_atX))
	indt_c = -1 # app. the same instant as S1
	indt_w = -1 # app. the same instant as S1
	ABLH=600 # m
	
	THT_c = dsref['cold']['nomean']['Mean']['MEAN_TH'][indt_w,:]
	THT_w = dsref['warm']['nomean']['Mean']['MEAN_TH'][indt_w,:]
	RV_c = dsref['cold']['nomean']['Mean']['MEAN_RV'][indt_w,:]
	RV_w = dsref['warm']['nomean']['Mean']['MEAN_RV'][indt_w,:]
	THTV_c = Compute_THTV(THT_c,RV_c) 
	THTV_w = Compute_THTV(THT_w,RV_w)

	zi,zi_c,zi_w = (  ABLH, 
			Z[int(np.argmax(THT_c.differentiate('level_les').values))].values,
			Z[int(np.argmax(THT_w.differentiate('level_les').values))].values   ) # max of Dtheta/dz

	RV_c = dsref['cold']['nomean']['Mean']['MEAN_RV'][indt_c,:]
	RV_w = dsref['warm']['nomean']['Mean']['MEAN_RV'][indt_w,:]

	RVW_c = ( dsref['cold']['nomean']['Resolved']['RES_WRV'] + dsref['cold']['nomean']['Subgrid']['SBG_WRT'] )[indt_c,:] # here RT = RV, no clouds 
	RVW_w = ( dsref['warm']['nomean']['Resolved']['RES_WRV'] + dsref['warm']['nomean']['Subgrid']['SBG_WRT'] )[indt_w,:]
	flux_RVW_hand = dsflx.FLX_RvW
	RV_hand = dsmean.RVTm
	
	mixed_rv = RV_hand[np.argmin(np.abs(Z.values-0.1*zi)):np.argmin(np.abs(Z.values-0.7*zi)),:].mean()
	mixed_rv_c = RV_c[np.argmin(np.abs(Z.values-0.1*zi_c)):np.argmin(np.abs(Z.values-0.7*zi_c))].mean()
	mixed_rv_w = RV_w[np.argmin(np.abs(Z.values-0.1*zi_w)):np.argmin(np.abs(Z.values-0.7*zi_w))].mean()
	print('mixed_rv,mixed_rv_c,mixed_rv_w (g/kg):',mixed_rv.values*1000,mixed_rv_c.values*1000,mixed_rv_w.values*1000)
	
	indx = []
	for kx in range(len(X_liste)):
		indx.append(np.argmin(np.abs(X.values-X_liste[kx])))
	E0 = np.abs(dsmean.E0[ind_E0_atX])
	map1 = 'winter'
	map2 = 'Reds'
	#dataSST = dsB.SST[0,1,nhalo:-nhalo].values
	if NORM:
		normFLX=E0
		NormC = np.abs(RVW_c[0])
		NormW = np.abs(RVW_w[0])
		borne_flx = [-0.1,2]
		bornesZ = [0,1.1]
		title_flx = r"< r$_v$w >/$E^{*}$"
		title_alti = 'z/zi'
		print('	E0_C,E0_W:',np.sqrt(NormC).values,np.sqrt(NormW).values)
		print('	zi_c,zi_w:',zi_c,zi_w)
	else:
		normFLX=1
		QV_c,QV_w = 1,1
		zi,zi_w,zi_c = 1,1,1
		bornesZ = [0,700]
		borne_flx = [-0.005,0.025] # to be modified
		title_flx = r"< r$_v$w > (kg.kg$^{-1}$.m.s$^{-1}$)"
		title_alti = 'Altitude (m)'
	
	
	colorsX = DISCRETIZED_2CMAP_2(map1,map2,X_liste,dataSST,crit_value,X.values)	
	fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax[0].plot((RV_c-mixed_rv_c)*1000,Z/zi_c,c='b',ls='--',label='refC')	
	ax[0].plot((RV_w-mixed_rv_w)*1000,Z/zi_w,c='r',ls='--',label='refW')	
	ax[1].plot(RVW_c/NormC,Z/zi_c,c='b',ls='--',label='refC')	
	ax[1].plot(RVW_w/NormW,Z/zi_w,c='r',ls='--',label='refW')
	for kx in range(0,len(X_liste)):
		ax[0].plot( (RV_hand[:,indx[kx]]-mixed_rv)*1000,Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' ) # [:-1]
		ax[1].plot( flux_RVW_hand[:,indx[kx]]/normFLX,Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' )
	ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax[1].xaxis.major.formatter._useMathText = True	
	ax[0].set_ylabel(title_alti)
	ax[0].set_xlabel(r'< $r_v$ > - $r_{v,mixed}$ (g.kg$^{-1}$)')
	ax[1].set_xlabel(title_flx)
	ax[0].set_ylim(bornesZ)
	ax[1].set_ylim(bornesZ)
	ax[0].set_xlim([-0.4,0.4]) # 
	ax[1].vlines(0,0,2000,colors='grey',ls='--')
	ax[1].set_xlim(borne_flx)
	ax[0].legend(fontsize=8,loc='upper right')
	ax[0].grid()
	ax[1].grid()
	fig.savefig(path_save+'ProfilesX_RV_RVW.png')
	
	# normalized with local surface flux
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot(RVW_c/NormC,Z/zi_c,c='b',ls='--',label='refC')	
	ax.plot(RVW_w/NormW,Z/zi_w,c='r',ls='--',label='refW')
	for kx in range(0,len(X_liste)):
		ax.plot( flux_RVW_hand[:,indx[kx]]/dsmean.E0[indx[kx]],Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' )
	ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax.xaxis.major.formatter._useMathText = True	
	ax.set_ylabel(title_alti)
	ax.set_xlabel(r"< r$_v$w >(x)/$E^{*}$(x)")
	ax.vlines(0,0,2000,colors='grey',ls='--')
	ax.legend(fontsize=8,loc='upper right')
	ax.set_ylim(bornesZ)
	ax.set_xlim([-0.1,1.4])
	fig.savefig(path_save+'ProfilesX_RV_RVW_atX.png')	

def PROFILES_AT_X_THT_THWFLX(dataSST,dsflx,dsmean,dsref,X,Z,X_liste,Q0_atX,NORM,crit_value,path_save,dpi):
	"""This procedure plot the profiles of THT and the total turbulent vertical flux of wtht.
		1 color for cold part, 1 color for warm part. Base on the 'crit_value' the separate domains 
		where the SST is "cold" to where it is "warm"
		
		INPUTS: 
			- dataSST: 1D SST data
			- dsflx: custom built file with mean flux
			- dsmean: custom built file with mean prognostic variables
			- dsref: reference profiles for homogeneous simulations
			- X: X dimension
			- Z: Z dimension
			- X_liste: liste of X positions (in meters)
			- Q0_atX: X position for normalisation of the flux
			- NORM: wether to normalize or not
			- crit_value: SST threshold that differentiate 'warm' SST from 'cold' SST
			- path_save: where to save the figures
			- dpi: dot per inches
		OUTPUTS:
			- A figure with profiles at some X position of: rv and wrv
	"""
	ind_Q0_atX = np.argmin(np.abs(X.values-Q0_atX))
	indt_c = -1 # app. the same instant as S1
	indt_w = -1 # app. the same instant as S1
	ABLH=600 # m
	
	THT_c = dsref['cold']['nomean']['Mean']['MEAN_TH'][indt_w,:]
	THT_w = dsref['warm']['nomean']['Mean']['MEAN_TH'][indt_w,:]
	RV_c = dsref['cold']['nomean']['Mean']['MEAN_RV'][indt_w,:]
	RV_w = dsref['warm']['nomean']['Mean']['MEAN_RV'][indt_w,:]
	THTV_c = Compute_THTV(THT_c,RV_c) 
	THTV_w = Compute_THTV(THT_w,RV_w)

	zi,zi_c,zi_w = (  ABLH, 
			Z[int(np.argmax(THT_c.differentiate('level_les').values))].values,
			Z[int(np.argmax(THT_w.differentiate('level_les').values))].values   ) # max of Dtheta/dz

	TH_c = dsref['cold']['nomean']['Mean']['MEAN_TH'][indt_c,:]
	TH_w = dsref['warm']['nomean']['Mean']['MEAN_TH'][indt_w,:]

	WTH_c = ( dsref['cold']['nomean']['Resolved']['RES_WTH'] + dsref['cold']['nomean']['Subgrid']['SBG_WTHL'] )[indt_c,:] # here RT = RV, no clouds 
	WTH_w = ( dsref['warm']['nomean']['Resolved']['RES_WTH'] + dsref['warm']['nomean']['Subgrid']['SBG_WTHL'] )[indt_w,:]
	flux_WTH_hand = dsflx.FLX_THW
	TH_hand = dsmean.THTm
	
	mixed_th = TH_hand[np.argmin(np.abs(Z.values-0.1*zi)):np.argmin(np.abs(Z.values-0.7*zi)),:].mean()
	mixed_th_c = TH_c[np.argmin(np.abs(Z.values-0.1*zi_c)):np.argmin(np.abs(Z.values-0.7*zi_c))].mean()
	mixed_th_w = TH_w[np.argmin(np.abs(Z.values-0.1*zi_w)):np.argmin(np.abs(Z.values-0.7*zi_w))].mean()
	print('mixed_th,mixed_th_c,mixed_th_w (K):',mixed_th.values,mixed_th_c.values,mixed_th_w.values)
	
	indx = []
	for kx in range(len(X_liste)):
		indx.append(np.argmin(np.abs(X.values-X_liste[kx])))
	Q0 = np.abs(dsmean.Q_star[ind_Q0_atX])
	map1 = 'winter'
	map2 = 'Reds'
	#dataSST = dsB.SST[0,1,nhalo:-nhalo].values
	if NORM:
		normFLX=Q0
		NormC = np.abs(WTH_c[0])
		NormW = np.abs(WTH_w[0])
		borne_flx = [-12.5,7.5]
		bornesZ = [0,1.1]
		title_flx = r"< w $\theta$ >/$Q^{*}$"
		title_alti = 'z/zi'
		print('	Q0_C,Q0_W:',np.sqrt(NormC).values,np.sqrt(NormW).values)
		print('	zi_c,zi_w:',zi_c,zi_w)
	else:
		normFLX=1
		QV_c,QV_w = 1,1
		zi,zi_w,zi_c = 1,1,1
		bornesZ = [0,700]
		borne_flx = [-0.005,0.025] # to be modified
		title_flx = r"< w \theta > (K.m.s$^{-1}$)"
		title_alti = 'Altitude (m)'
	
	
	colorsX = DISCRETIZED_2CMAP_2(map1,map2,X_liste,dataSST,crit_value,X.values)	
	fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax[0].plot(TH_c-mixed_th_c,Z/zi_c,c='b',ls='--',label='refC')	
	ax[0].plot(TH_w-mixed_th_w,Z/zi_w,c='r',ls='--',label='refW')	
	ax[1].plot(WTH_c/NormC,Z/zi_c,c='b',ls='--',label='refC')	
	ax[1].plot(WTH_w/NormW,Z/zi_w,c='r',ls='--',label='refW')
	for kx in range(0,len(X_liste)):
		ax[0].plot( TH_hand[:,indx[kx]]-mixed_th,Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' ) # [:-1]
		ax[1].plot( flux_WTH_hand[:,indx[kx]]/normFLX,Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' )
	#ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	#ax[1].xaxis.major.formatter._useMathText = True	
	ax[0].set_ylabel(title_alti)
	ax[0].set_xlabel(r'< $\theta$ > - $\theta_{mixed}$ (K)')
	ax[1].set_xlabel(title_flx)
	ax[0].set_ylim(bornesZ)
	ax[1].set_ylim(bornesZ)
	ax[0].set_xlim([-0.2,0.2]) # 
	ax[1].vlines(0,0,2000,colors='grey',ls='--')
	ax[1].set_xlim(borne_flx)
	ax[0].legend(fontsize=8,loc='upper right')
	ax[0].grid()
	ax[1].grid()
	fig.savefig(path_save+'ProfilesX_TH_WTH.png')
	
	# normalized with local surface flux
	# fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	# ax.plot(WTH_c/NormC,Z/zi_c,c='b',ls='--',label='refC')	
	# ax.plot(WTH_w/NormW,Z/zi_w,c='r',ls='--',label='refW')
	# for kx in range(0,len(X_liste)):
	# 	ax.plot( flux_WTH_hand[:,indx[kx]]/dsmean.Q_star[indx[kx]],Z/zi,c=colorsX[kx],label='x='+str(np.round(X[indx[kx]].values/1000,1))+'km' )
	# ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	# ax.xaxis.major.formatter._useMathText = True	
	# ax.set_ylabel(title_alti)
	# ax.set_xlabel(r"< w $\theta$ >(x)/$Q^{*}$(x)")
	# ax.vlines(0,0,2000,colors='grey',ls='--')
	# ax.legend(fontsize=8,loc='upper right')
	# ax.set_ylim(bornesZ)
	# #ax.set_xlim([-0.1,1.4])
	# fig.savefig(path_save+'ProfilesX_TH_WTH_atX.png')

def entrainment_velocity(X,Z,CENTER_DTHTV,dsflx,dsmean,dsref,path_save,dpi):
	"""
	This procedure is plotting:
		- mean ABL height (along X)
		- entrainment velocity (dzi/dt, no large scale sub)
		- entrainment velocity ( min(wthtv)/delta theta)
		
	INPUTS:
		- X: X dimension
		- Z: Z dimension
		- CENTER_DTHTV: 'ABLH' or 'MIN_FLX', middle of the jump of thtv
		- dsflx: custom built file with mean flux
		- dsmean: custom built file with mean prog. variables
		- dsref: mean profiles for reference homogeneous simulations
		- path_save: where to save the figure
		- dpi: dot per inches
	OUTPUTS:
		- a plot with entrainment velocity for 
			-> S1 (via we=dzi/dt=U.dzi/dx)
			-> ref (via dzi/dt)
			-> estimate with we = min(wthtv)*delta thetav, with different jumps
	"""
	# entrainment velocity = dzi/dt
	#	-> for ref sims, easy to compute with ABLH(t) from MNH outputs
	#	-> for S1, we consider taylor's hypothesis : dzi/dt ~ Udzi/dx at every x
	#		and with U = <Um(zi=600)>x
	#	-> compared with min(wthtv)/dTHTV with different widths for dTHTV
	
	# for smoothing
	WSIZE = 251 # odd, number of X points
	ORD = 2 # order: because dx=cst, 2=3 for smooth and 3=4 for d/dx
	
	thtv_hand = dsmean.THTvm
	wthtv = dsflx.FLX_THvW.min(dim=['level'])
	wthtv_p = dsflx.FLX_THvW
	L_indz_minflx = wthtv_p.sel(level=slice(0,900)).argmin('level').values
	smoothed_flx = savitzky_golay(Z[L_indz_minflx].values, window_size=WSIZE, order=ORD, deriv=0, rate=1/50)
	# here we test look at time evolution of E = wthtv_min/wthtv(z=0) for ref sims
	THT_c = dsref['cold']['nomean']['Mean']['MEAN_TH'][:,:]
	THT_w = dsref['warm']['nomean']['Mean']['MEAN_TH'][:,:]
	RV_c = dsref['cold']['nomean']['Mean']['MEAN_RV'][:,:]
	RV_w = dsref['warm']['nomean']['Mean']['MEAN_RV'][:,:]
	THTV_c = Compute_THTV(THT_c,RV_c)
	THTV_w = Compute_THTV(THT_w,RV_w)
	RES_WTHV_c = dsref['cold']['nomean']['Resolved']['RES_WTHV'][:,:]
	RES_WTHV_w = dsref['warm']['nomean']['Resolved']['RES_WTHV'][:,:]
	SBG_WTHV_c = THTV_c/THT_c*dsref['cold']['nomean']['Subgrid']['SBG_WTHL'][:,:] + 0.61*THT_c*dsref['cold']['nomean']['Subgrid']['SBG_WRT'][:,:]
	SBG_WTHV_w = THTV_w/THT_w*dsref['warm']['nomean']['Subgrid']['SBG_WTHL'][:,:] + 0.61*THT_w*dsref['warm']['nomean']['Subgrid']['SBG_WRT'][:,:]
	WTHV_c = RES_WTHV_c + SBG_WTHV_c
	WTHV_w = RES_WTHV_w + SBG_WTHV_w
	time = np.arange(0,121)
	#zi_c,zi_w = dsref['cold']['nomean']['Misc']['BL_H'].values[:],dsref['warm']['nomean']['Misc']['BL_H'].values[:]
	Ec,Ew = np.zeros(len(time)),np.zeros(len(time))
	zi_c,zi_w = np.zeros(len(time)),np.zeros(len(time))
	for t in range(len(time)):
		zi_c[t] = Z[THTV_c[-len(time)+t].differentiate('level_les').argmax('level_les')]
		zi_w[t] = Z[THTV_w[-len(time)+t].differentiate('level_les').argmax('level_les')]
		Ec[t] = ( WTHV_c[-len(time)+t,:]/SBG_WTHV_c[-len(time)+t,0] ).sel(level_les=slice(0,1.2*zi_c[t])).min(dim='level_les')
		Ew[t] = ( WTHV_w[-len(time)+t,:]/SBG_WTHV_w[-len(time)+t,0] ).sel(level_les=slice(0,1.2*zi_w[t])).min(dim='level_les')
		
	# we_c =  (dsref['cold']['mean']['Misc']['BL_H'][-1] - dsref['cold']['mean']['Misc']['BL_H'][0])/3600 # 1,251 cm/s
	# we_w =  (dsref['warm']['mean']['Misc']['BL_H'][-1] - dsref['warm']['mean']['Misc']['BL_H'][0])/3600 # 1,943 cm/s
	we_c = (zi_c[-1] - zi_c[0]) / 3600
	we_w = (zi_w[-1] - zi_w[0]) / 3600
	print('refC we, refW we')
	print(we_c,we_w)


	# Height of ABL along X
	L_indz = thtv_hand.differentiate('level').sel(level=slice(0,900)).argmax('level').values # altitude of max gradient thtv
	ABLH_x = Z[L_indz].values
	Nx = len(ABLH_x)
	ABLH_x_3 = np.zeros(3*Nx)
	ABLH_x_3[:Nx] = ABLH_x
	ABLH_x_3[Nx:2*Nx] = ABLH_x
	ABLH_x_3[2*Nx:3*Nx] = ABLH_x
	dHdx = savitzky_golay(ABLH_x_3, window_size=WSIZE, order=ORD, deriv=1, rate=1/50)[Nx:2*Nx] # fake cyclic conditions
	Uadv = dsmean.Um.sel(level=600,method='nearest').mean(dim='ni').values # for Taylor's hypothesis
	print('Uadv',Uadv)
	smoothed = savitzky_golay(ABLH_x, window_size=WSIZE, order=ORD, deriv=0, rate=1/50)
	
	# Computing dthtv
	L_n_inversion = [2,3,4,5] # number of cell around inversion height (ABLH_x) 
	color = ['cyan','green','orange','purple']
	deltaTHTV = np.zeros((len(L_n_inversion),Nx))					
	if CENTER_DTHTV == 'MIN_FLX': # chosing where to compute the dthtv
		L_mid = L_indz_minflx
	elif CENTER_DTHTV == 'ABLH':
		L_mid = L_indz
	for k,dz in enumerate(L_n_inversion):
		borne_up = L_mid+dz
		borne_down = L_mid-dz
		thtv_z2 = np.diag(thtv_hand[borne_up,:].values)
		thtv_z1 = np.diag(thtv_hand[borne_down,:].values)
		deltaTHTV[k,:] =  thtv_z2 -  thtv_z1
	
	
	##### PLOTS
	
	# min(wthtv) of reference along time axis
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax.plot(time,Ec,c='b',label='refC')
	ax.plot(time,Ew,c='r',label='refW')
	ax.set_xlabel('time')
	ax.set_ylabel('E')
	ax.set_title(r"min($w'\theta_v'$) des references")
	
	# ABLH and Z[min(wthtv)]
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax.plot(X/1000,ABLH_x,c='grey') # not smoothed
	ax.plot(X/1000,smoothed,c='k',label='ABLH(x)') # smoothed
	ax.set_xlabel('X (km)')
	ax.set_ylabel('altitude')
	ax.plot(X/1000,Z[L_indz_minflx].values, c='orange')
	ax.plot(X/1000,smoothed_flx , c='r',label=r'Z[min($w\theta_v$)]') # smoothed 
	#ax.set_title('savgol(w,o,d)=('+str(WSIZE)+','+str(ORD)+',0)')
	ax.legend()
	fig.savefig(path_save+'ABL_height_and_smoothed.png')
	
	# U.Dzi/DX, we of ref, min(wthtv)/delta_THTV with different width for delta_THTV
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.hlines( 0,X[0]/1000,X[-1]/1000,colors='grey',linestyles='--')
	ax.hlines( we_c*100,X[0]/1000,X[-1]/1000,colors='b',label='refC')
	ax.hlines( we_w*100,X[0]/1000,X[-1]/1000,colors='r',label='refW')
	ax.plot( X/1000, Uadv*dHdx*100,c='k',label='U.dzi/dx')
	print('zimax for S1 (cm/s):',np.max(Uadv*dHdx*100))
	smoothed_minflx = savitzky_golay(wthtv, window_size=WSIZE, order=ORD, deriv=0, rate=1/50)
	for k,dz in enumerate(L_n_inversion):
		ax.plot( X/1000, -wthtv/deltaTHTV[k,:]*100,c=color[k],label=r'$\pm$ '+str(dz)+' cell') # not smoothed
		#ax.plot( X/1000, -smoothed_minflx/deltaTHTV[k,:]*100,c=color[k],label=r'$\pm$ '+str(dz)+' cell') # smoothed
	ax.set_xlabel('X (km)')
	ax.set_ylabel('we (cm/s)')
	ax.legend()
	#ax.set_title('savgol(w,o,d)=('+str(WSIZE)+','+str(ORD)+',1)')
	fig.savefig(path_save+'entrainment_rate_'+CENTER_DTHTV+'.png')

	# checking that delta_THTV is well around the altitude of min(wthtv)
	#	X = 13km
	indx = nearest(X.values,13*1000)
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.vlines(0,0,50,colors='grey')
	ax.plot(thtv_hand[:,indx]-297.7,Z/600,c='k')
	for k,dz in enumerate(L_n_inversion):
		borne_up = L_mid[indx]+dz
		borne_down = L_mid[indx]-dz
		ax.hlines(Z[borne_up].values/600,-10,10,colors=color[k],ls='-',label=r'\pm '+str(dz)+' cell')
		ax.hlines(Z[borne_down].values/600,-10,10,colors=color[k],ls='--')
	ax2 = ax.twiny()
	ax2.plot(wthtv_p[:,indx],Z/600,c='r')
	ax.set_xlim([-0.2,0.8])
	ax2.set_xlim([-0.005,0.02])
	ax.set_ylim([0,1.5])
	ax.set_xlabel(r'$\theta_v$-$\theta_{v,mixed}$ and $\Delta \theta_v$')
	ax2.set_xlabel(r'$<w\theta_v>$ at X=13km')
	fig.savefig(path_save+'checking_at_X13km'+CENTER_DTHTV+'.png')
	#	X = 24km
	indx = nearest(X.values,24*1000)
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.vlines(0,0,50,colors='grey')
	ax.plot(thtv_hand[:,indx]-297.7,Z/600,c='k')
	for k,dz in enumerate(L_n_inversion):
		borne_up = L_mid[indx]+dz
		borne_down = L_mid[indx]-dz
		ax.hlines(Z[borne_up].values/600,-10,10,colors=color[k],ls='-',label=r'\pm '+str(dz)+' cell')
		ax.hlines(Z[borne_down].values/600,-10,10,colors=color[k],ls='--')
	ax2 = ax.twiny()
	ax2.plot(wthtv_p[:,indx],Z/600,c='r')
	ax.set_xlim([-0.2,0.8])
	ax2.set_xlim([-0.005,0.02])
	ax.set_ylim([0,1.5])
	ax.set_xlabel(r'$\theta_v$ and $\Delta \theta_v$')
	ax2.set_xlabel(r'$<w\theta_v>$ at X=24km')
	fig.savefig(path_save+'checking_at_X24km'+CENTER_DTHTV+'.png')
# End in paper -----------------

	
def TURNOVER_EDDY_TIME(nhalo,path,savepath):
	"""This procedure compute a turn over eddy time
		over the X dimension of a single file, located
		at path. The result plot is save at savepath
	"""
	# This time will decide the time needed to have a converged average operator for NAM_BUDGET
	# On cold side, it should be the longest
	#
	# This part is file specific and so is independant of the other parts of this script
	#	as a consequence, a specific file needs to be set
	#
	# Conclusion for X768_SST_tanh5_moist: turn over eddy time is 2500s on cold part ==> statistics on 5x ~3,5h and runtime is 8x ~5,5h-> 6h + spinup
	ds=xr.open_dataset(path)
	THT = ds.THT[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].values
	RVT = ds.RVT[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].values
	thtv = Compute_THTV(THT,RVT)
	mean = thtv.mean(axis=0)
	Z = ds.level[nhalo:-nhalo]
	X = ds.ni[nhalo:-nhalo]
	x = 0
	print('thtv at lvl 0 and x=0',mean[x])
	UW_FLX = ds.UW_HFLX[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=0) + ds.UW_VFLX[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=0)
	VW_FLX = ds.VW_HFLX[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=0) + ds.VW_VFLX[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=0)
	u_star = (( UW_FLX**2 + VW_FLX**2 ))**(1/4)
	Q = ds.THW_FLX[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=0).values
	Qv = ( ds.THW_FLX[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].values*(1+0.61*RVT) + 0.61*THT*ds.RCONSW_FLX[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].values ).mean(axis=0)
	beta = 9.81/THT.mean(axis=0)
	betav = 9.81/mean
	
	print('At x=',x)
	print('Q* =',Q[x])
	print('turn over eddy time THT =',1/(beta[x]*Q[x]))
	T1 = 1/(beta*Q)
	print('Qv* =',Qv[x])
	print('turn over eddy time THTv =',1/(betav[x]*Qv[x]))
	T2 = 1/(betav*Qv)
	fig, ax = plt.subplots(3,1,figsize = (5,5),constrained_layout=True,dpi=100)
	ax[0].plot(X,mean,c='k')
	ax[0].set_title('mean thtv at surface (K)')
	ax[1].plot(X,Q,c='k',label='Q* (m K s-1)')
	ax[1].plot(X,Qv,c='grey',label='Qv* (m K s-1)')
	ax[1].legend()
	ax[2].plot(X,T2,c='k')
	ax[2].set_title('Turnover eddy time at differents x locations (s)')
	fig.savefig(savepath)
	
def GAMMA(SOURCE,dsmean,dsBU,dsB,dsflx,Ug,X,Z,res,path_save,dpi):
	"""This procedure is meant to compute terms that are used
		in the computation of gamma as in 
		Samelson 2006 'On the Coupling of Wind Stress and Sea Surface Temperature'
		
		- Tau : friction (m-1 s-2) 1D (x)
		- THTvm : mean virtual potential temperature (K) 2D (x,z)
		- Um : mean zonal wind (m s-1) 2D (x,z)
		- Ug : geostrophic wind (m s-1) float
	"""
	
	# Description :
	# Based on : Samelson 2006 'On the Coupling of Wind Stress and Sea Surface Temperature'
	#
	# What i compute : gamma = Time scale of friction / time scale of advection
	#
	#			gamma = h*deltaV*V / tau*L
	#	with :
	#		h : caracteristic ABL height 
	#		deltaV : change of wind speed at surface across the front of SST
	#		V : caracteristic wind speed of the ABL (mean between before and after)
	#		tau : caracteristic friction norme (mean between before and after) (in m-1 s-2)
	#		L : lenght scale of the front
	#
	#	--> gamma > 1 : advection dominates, Downward mixing momentum can be used to describe the U increase
	#	--> gamma < 1 : friction dominates, local equilibrium is achieved (non rotating BL like physic)
	#	
	# Comment : I dont have surface (subgrid) fluxes in OUT files so my mean here is spatial in Y and in a bow for X and on 3 files in t.
	
	gTHT = np.gradient(dsmean.THTm,Z,axis=0)
	gTHT_contour = 0.0005 # K/m
	if SOURCE=='HAND':
		TAU	= dsmean.Tau
		THTvm	= dsmean.THTvm
		Um	= dsmean.Um
	elif SOURCE=='NAMLIST':
		TAU	= dsmean.Tau # No computation by NAM_BUDGET of Tau
		THTvm 	= Compute_THTV(dsBU['TH'].AVEF,dsBU['RV'].AVEF)
		Um	= dsBU['UU'].AVEF
	SST = dsB.SST[0,1,nhalo:-nhalo].values
	wthtv_flx_surface = dsflx.FLX_THvW[0,:]
	GTHTV = np.gradient(THTvm,Z,axis=0)
	ABLH1 = np.zeros(X.shape[0]) # 1D
	ABLHW = np.zeros(X.shape[0]) # 1D
	ABLHC = np.zeros(X.shape[0]) # 1D	
	u_star = np.sqrt(TAU)
	L = L_Obukhov(297.2,u_star,wthtv_flx_surface)
	F = Um/Ug * (1- Um/Ug)
	G = (1- Um/Ug)
	H_disp = sp.integrate.trapezoid(G,Z,axis=0) # displacement height : difference of debit (m3/s) from the case of U=Ug at every Z
	H_qdm = sp.integrate.trapezoid(F[:,:],Z,axis=0) # momentum height : height to add to displacement height to conserve momentum debit compare to case U=Ug at every Z
	H_95 = np.zeros(X.shape) # 0.95*Ug height
	for x in range(len(X)):
		H_95[x] = Z[np.argmin(np.abs(Um[:,x].values-0.95*Ug))].values
	
	for x in range(len(X)): #
		#ABLH1[x] = Z[10+np.argmax(GTHTV[10:,x])].values
		ABLH1[x] = Z[np.argmax(GTHTV[:,x])].values
		k0 = 10
		zi,k = Z[k0],k0
		while k<len(Z) and GTHTV[k,x] > GTHTV[k+1,x]:
			k=k+1
			zi=Z[k]
		#ABLH[x] = zi
		#ABLH[x] = Z[np.argmin(dsflx.FLX_THvW.values[:,x])]
	# custom cmap
	levels = [-1.71e-3,-1.03e-3,-6.85e-4,-3.42e-4,-1.71e-4,-6.85e-5,-5e-5,-3.42e-5,-1.71e-5,
			0,
			1.71e-5,3.42e-5,5e-5,6.85e-5,1.71e-4,3.42e-4,6.85e-4,1.03e-3,1.71e-3]	
	cmap = mpl.colormaps.get_cmap('Spectral') #Greys
	colors=[]
	for k in range(len(levels)-1):
		colors.append(cmap(k/len(levels)))
	colors[7] = (1,1,1,1)
	colors[8] = (1,1,1,1)
	cmap = mpl.colors.ListedColormap(colors)
	norm = 	mpl.colors.BoundaryNorm(boundaries=levels, ncolors=256, extend='both')	
		
	V_U = dsmean.Um
	V_U0 = V_U[:,0]
	V_U0 = V_U0.expand_dims(dim={'ni':X},axis=1)
	V_dU = V_U - V_U0
	V_W = dsmean.Wm
	jumpx,jumpz = 15,2
	
	# Slice of GTHTV and ABLH(x)
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	s = ax.contourf(X/1000,Z,GTHTV,cmap='RdBu_r',levels=levels,norm=norm,extend='both')
	ax.plot(X/1000,savitzky_golay(ABLH1[:], 31, order=1),color='k',ls='--')
	plt.colorbar(s,ax=ax)
	Q = ax.quiver(X[::jumpx]/1000,Z[::jumpz],V_dU[::jumpz,::jumpx],V_W[::jumpz,::jumpx]*100,angles='xy',pivot='mid',headlength=5,scale=50)
	ax.quiverkey(Q, 0.85, 0.07, 1, r'$\Delta Wind X$=$1 m/s$', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
	ax.quiverkey(Q, 0.85, 0.03, 1, r'$\Delta Wind Z$=$1 cm/s$', labelpos='E',coordinates='figure',angle=0) # Reference arrow vertical
	Add_SST_ticks(ax,0.02)
	ax.set_ylim([0,600])
	ax.set_xlabel('X (km)')
	ax.set_ylabel('Altitude (m)')
	ax.set_title(r'gradient $\theta_v$ K/m')
	fig.savefig(path_save+'gTHTV_ABLH.png')
	
	# tau_surface(x) and surface wind(x)	
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax.plot(X/1000,TAU,c='blue')
	ax.set_ylabel(r'$\tau$ (blue) (m2/s2)')
	ax2 = ax.twinx()
	ax2.plot(X/1000,Um[0,:],c='k')
	ax2.set_ylabel('U wind (black) m/s')
	ax.set_xlim([0,X[-1]/1000])
	#fig.savefig(path_save+'Tau_wind_surface.png')
	
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax.plot(X/1000,(Um[0,:]-Um[0,0])/TAU,c='k')
	ax.set_xlim([0,X[-1]/1000])
	ax.set_title(r'U-U(x=0) surface over Tau')
	ax.set_xlabel('X (km)')
	ax.set_ylabel(r'$\Delta$U (m/s)')
	#fig.savefig(path_save+'DeltaU_over_Tau_surface.png')
	
#	DX = 50 #m
#	IDX = DX//res
#	crit_value = 296.25 # K
#	map1 = 'Blues'
#	map2 = 'Reds'
#	data = SST
#	colorsX = DISCRETIZED_2CMAP(map1,map2,IDX,data,crit_value,X)		
#	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
#	#ax.plot(TAU,Um[0,:],c='k')
#	ax.plot(TAU,TAU*44.51-1.77,c='blue',label='V=44.51*Tau-1.77',alpha=0.5)
#	ax.plot(TAU,TAU*43.61-1.79,c='r',label='V=43.61*Tau-1.79',alpha=0.5)
#	ax.scatter(TAU[0],Um[0,0]-Um[0,0],c='grey',label='Simulation',s=1.0)
#	ax.scatter(TAU,Um[0,:]-Um[0,0],c=colorsX,s=1.0)
#	ax.set_xlabel('Tau (m s-2)')
#	ax.set_ylabel('U-U(x=0) (m s-1)')
#	ax.legend()
#	#fig.savefig(path_save+'DeltaU_Tau_surface.png')
	
#	# Boundary layer height : non rotating physics
#	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
#	ax.plot(X/1000,H_qdm,color='aqua',label='H_qdm')
#	ax.plot(X/1000,H_disp,color='orange',label='H_displacement')
#	ax.set_title('Boundary layer height, non rotating physic')
#	ax.legend()
#	ax.set_xlabel('X (km)')
#	ax.set_ylabel('altitude (m)')
#	#fig.savefig(path_save+'BLH_nonrotating.png')
	
	# slice of U wind
	fig, ax = plt.subplots(1,1,figsize = (7,5),constrained_layout=True,dpi=dpi)
	levels=np.arange(4.5,7.5,0.25)
	s = ax.contourf(X/1000,Z,Um[:,:],cmap='rainbow',levels=levels,extend='both')
	ax.plot(X/1000,savitzky_golay(ABLH1[:], 31, order=1),color='k',ls='--') # filtered
	ax.contour(X/1000,Z,gTHT,levels=[gTHT_contour],colors='grey',linestyles='--')
	#ax.plot(X/1000,ABLH[:],color='k',ls='--') # not filtered
	plt.colorbar(s,ax=ax)
	ax.set_xlabel('X (km)')
	ax.set_ylabel('Altitude (m)')
	ax.set_title('U wind (m/s)')
	ax.set_ylim([0,600])
	fig.savefig(path_save+'Uwind_ABLH.png')
	
	# Bulk aero coefficient Cd defined with u*^2 = Cd*U²

def CHECK_MASS_CONSERVATION(X,Z,TYPE,X0,AT_X0,dsO,dsINI,ds000,dsBU,path_save,METHOD,dpi):
	"""
	This procedure plots the 3 gradients terms of the continuity equation
		- it AT_X0 is True then profiles are plotted
		- else 2D (X,Z) are plotted
	
	rho is the one defined by 
		Durran 1989, rho = rhoref*thtvref*(1+rvref)
		AE, rho = rhoref(z)
		incompressible, rho = cst
	"""
	
	indx = np.argmin(np.abs(X.values-X0*1000))
	
	P_ref = dsINI.PABST[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	tht_ref = dsINI.THT[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	rvref = dsINI.RVT[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	Pi_ref = Exner(P_ref)
	thtvref = Compute_THTV(tht_ref,rvref)
	rhoref = Pi_ref**(Cvd/Rd)*P00/(Rd*thtvref)
	
	if METHOD=='HAND':
		U = dsO.UT.mean(axis=0).interp({'ni_u':ds000.ni})[nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
		V = dsO.VT.mean(axis=0).interp({'nj_v':ds000.nj}) [nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
		W = dsO.WT.mean(axis=0).interp({'level_w':ds000.level})[nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 
		P = dsO.PABST[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		Pi = Exner(P)
		THT = dsO.THT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		RVT = dsO.RVT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		THTV = Compute_THTV(THT,RVT)
		RHO = Pi**(Cvd/Rd)*P00/(Rd*THTV)
		
		gradTRHO = np.gradient(RHO,5*60,axis=0).mean(axis=2)
		gradTRHO_m = gradTRHO.mean(axis=0)
		# Durran formulation, with H=0
		rhodeff = rhoref*thtvref*(1+rvref)/300
		gradX = np.gradient(rhodeff*U,ds000.ni[nhalo:-nhalo],axis=2).mean(axis=1)
		gradY = np.gradient(rhodeff*V,ds000.nj[nhalo:-nhalo],axis=1).mean(axis=1)
		gradZ = np.gradient(rhodeff*W,ds000.level[nhalo:-nhalo],axis=0).mean(axis=1)
		gradX = uniform_filter1d(gradX, size=40,axis=1,mode='wrap')
		gradY = uniform_filter1d(gradY, size=40,axis=1,mode='wrap')
		gradZ = uniform_filter1d(gradZ, size=40,axis=1,mode='wrap')
		# Original anelastic formulation
		gradXAE = np.gradient(rhoref*U,ds000.ni[nhalo:-nhalo],axis=2).mean(axis=1)
		gradYAE = np.gradient(rhoref*V,ds000.nj[nhalo:-nhalo],axis=1).mean(axis=1)
		gradZAE = np.gradient(rhoref*W,ds000.level[nhalo:-nhalo],axis=0).mean(axis=1)
	#	gradXAE = uniform_filter1d(gradX, size=40,axis=1,mode='wrap')
	#	gradYAE = uniform_filter1d(gradY, size=40,axis=1,mode='wrap')
	#	gradZAE = uniform_filter1d(gradZ, size=40,axis=1,mode='wrap')
		# incompressible formulation : rho=cst
		gradXIC = np.gradient(U,ds000.ni[nhalo:-nhalo],axis=2).mean(axis=1)
		gradYIC = np.gradient(V,ds000.nj[nhalo:-nhalo],axis=1).mean(axis=1)
		gradZIC = np.gradient(W,ds000.level[nhalo:-nhalo],axis=0).mean(axis=1)
	#	gradXIC = uniform_filter1d(gradX, size=40,axis=1,mode='wrap')
	#	gradYIC = uniform_filter1d(gradY, size=40,axis=1,mode='wrap')
	#	gradZIC = uniform_filter1d(gradZ, size=40,axis=1,mode='wrap')
	elif METHOD=='NAMELIST':
		U = dsBU['UU'].AVEF[0,:,:].interp({'cart_ni_u':ds000.ni[nhalo:-nhalo]})
		U = U.rename(new_name_or_name_dict={'cart_level':'level'})
		#U = U.rename(new_name_or_name_dict={'cart_ni':'ni'})
		W = dsBU['WW'].AVEF[0,:,:].interp({'cart_level_w':ds000.level[nhalo:-nhalo]})
		#W = W.rename(new_name_or_name_dict={'cart_level':'level'})
		W = W.rename(new_name_or_name_dict={'cart_ni':'ni'})
		
		RHOREF = rhoref.mean(axis=1)[:-1,:-1]
		RVREF = rvref.mean(axis=1)[:-1,:-1]
		THTVREF = thtvref.mean(axis=1)[:-1,:-1]
		RHODEFF = RHOREF*THTVREF*(1+RVREF)/300
		gradX = (RHODEFF*U).differentiate('ni')
		gradZ = (RHODEFF*W).differentiate('level')
		gradY = np.zeros(gradX.shape)
		Z = ds000.cart_level.values
		X = ds000.cart_ni.values
	if AT_X0:
		if TYPE=='DUR':
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
#			ax.plot(gradX[:,:].mean(axis=1),Z,label=r'd($\rho_{dref}$U)/dx',c='r')
#			ax.plot(gradY[:,:].mean(axis=1),Z,label=r'd($\rho_{dref}$V)/dy',c='g')
#			ax.plot(gradZ[:,:].mean(axis=1),Z,label=r'd($\rho_{dref}$W)/dz',c='b')
#			ax.plot(gradTRHO_m[:,:].mean(axis=1),Z,label=r'd$\rho$/dt',c='purple')
#			ax.plot(gradX[:,:].mean(axis=1)+gradY[:,:].mean(axis=1)+gradZ[:,:].mean(axis=1)+gradTRHO_m[:,:].mean(axis=1),Z,label='Sum',c='k')
			ax.plot(gradX[:,indx],Z,label=r'd($\rho_{dref}$U)/dx',c='r')
			ax.plot(gradY[:,indx],Z,label=r'd($\rho_{dref}$V)/dy',c='g')
			ax.plot(gradZ[:,indx],Z,label=r'd($\rho_{dref}$W)/dz',c='b')
			#ax.plot(gradTRHO_m[:,indx],Z,label=r'd$\rho$/dt',c='purple')
			ax.plot(gradX[:,indx]+gradY[:,indx]+gradZ[:,indx],Z,label='Sum',c='k') #+gradTRHO_m[:,indx]
			ax.set_ylabel('Altitude')
			ax.legend()
			ax.vlines(0,0,600,colors='grey',ls='--')
			ax.set_ylim([0,600])
			ax.set_xlim([-0.0003,0.0003])
			ax.set_title('Pseudo-compressible Durran (kg m-3 s-1)')
			name_save = 'MASS_DUR_profiles_X'+str(X0)+'km.png'
		elif TYPE=='AE':
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			ax.plot(gradXAE[:,indx],Z,label=r'd($\rho$U)/dx',c='r')
			ax.plot(gradYAE[:,indx],Z,label=r'd($\rho$V)/dy',c='g')
			ax.plot(gradZAE[:,indx],Z,label=r'd($\rho$W)/dz',c='b')
			ax.plot(gradXAE[:,indx]+gradYAE[:,indx]+gradZAE[:,indx],Z,label='Sum',c='k')
			ax.set_ylabel('Altitude')
			ax.legend()
			ax.vlines(0,0,600,colors='grey',ls='--')
			ax.set_ylim([0,600])
			ax.set_xlim([-0.025,0.025])
			ax.set_title('Original anelastic (kg m-3 s-1)')
			name_save = 'MASS_AE_profiles_X'+str(X0)+'km.png'
		elif TYPE=='INCOMP':
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			ax.plot(gradXIC[:,indx],Z,label=r'dU/dx',c='r')
			ax.plot(gradYIC[:,indx],Z,label=r'dV/dy',c='g')
			ax.plot(gradZIC[:,indx],Z,label=r'dW/dz',c='b')
			ax.plot(gradXIC[:,indx]+gradYIC[:,indx]+gradZIC[:,indx],Z,label='Sum',c='k')
			ax.set_ylabel('Altitude')
			ax.legend()
			ax.vlines(0,0,600,colors='grey',ls='--')
			ax.set_ylim([0,600])
			ax.set_xlim([-0.025,0.025])
			ax.set_title('rho=cst (s-1)')
			name_save = 'MASS_INCOMP_profiles_X'+str(X0)+'km.png'
		ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
		ax.xaxis.major.formatter._useMathText = True	
	else:
		vmin,vmax = -0.0001,0.0001 # vmin,vmax = -0.005,0.005
		if METHOD=='HAND':
			vmin,vmax = -0.0001,0.0001
			fig, axe = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			s = axe.pcolormesh(X/1000,Z,gradTRHO[-1,:,:],cmap='gnuplot2',vmin=-4e-6,vmax=0)
			cb = plt.colorbar(s,ax=axe)
			cb.formatter.set_powerlimits((0, 0))
			axe.set_ylabel('Altitude')
			axe.set_xlabel('X (km)')
			#axe.set_ylim([0,600])
			axe.set_title(r'd$\rho$/dt at t+6h')
			fig.savefig(path_save+'DrhoDT_XZ.png')
		else:
			vmin,vmax = -0.0001,0.0001
		fig, axe = plt.subplots(2,2,figsize = (15,10),constrained_layout=True,dpi=dpi)
		ax = axe.flatten()
		s = ax[0].pcolormesh(X/1000,Z,gradX[:,:],cmap='bwr',vmin=vmin,vmax=vmax)
		cb = plt.colorbar(s,ax=ax[0])
		cb.formatter.set_powerlimits((0, 0))
		ax[0].set_title(r'd($\rho$U)/dx')
		s = ax[1].pcolormesh(X/1000,Z,gradY[:,:],cmap='bwr',vmin=vmin,vmax=vmax)
		cb = plt.colorbar(s,ax=ax[1])
		cb.formatter.set_powerlimits((0, 0))
		ax[1].set_title(r'd($\rho$V)/dy')
		s = ax[2].pcolormesh(X/1000,Z,gradZ[:,:],cmap='bwr',vmin=vmin,vmax=vmax)
		cb = plt.colorbar(s,ax=ax[2])
		cb.formatter.set_powerlimits((0, 0))
		ax[2].set_title(r'd($\rho$W)/dz')
		s = ax[3].pcolormesh(X/1000,Z,gradX[:,:]+gradY[:,:]+gradZ[:,:],cmap='bwr',vmin=vmin,vmax=vmax) # +gradTRHO_m[:,:]
		cb = plt.colorbar(s,ax=ax[3])
		cb.formatter.set_powerlimits((0, 0))
		ax[3].set_title('Sum') # d$\rho$/dt +
		for k in range(len(ax)):
			ax[k].set_ylabel('Altitude')
			ax[k].set_xlabel('X (km)')
			#ax[k].set_ylim([0,600])
		name_save = 'MASS_XZ.png'
		fig.suptitle('Continuity equation terms (Durran) (kg m-3 s-1)')	
		#INT_DIV = (gradX[:,:]+gradY[:,:]+gradZ[:,:]).fillna(0).integrate(['ni','level'])
		#print(INT_DIV.values)
	fig.savefig(path_save+'_'+METHOD+'_'+name_save)
	
def RH_SLICE(dsB,time,nhalo,path_save,dpi):
	"""This procedure is plotting the relative humidity of the mean field
	"""	
	X = dsB.ni[nhalo:-nhalo]
	Z = dsB.level[nhalo:-nhalo]
	THT = dsB.THT[time,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=1).values
	RVT = dsB.RVT[time,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=1).values
	P = dsB.PABST[time,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=1).values
	RH = qtoRH(RVT,THT,P)
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	levels=np.arange(0,102,2)
	#s = ax.contourf(X/1000,Z,RH*100,cmap='rainbow',levels=levels)
	s = ax.pcolormesh(X/1000,Z,RH*100,cmap='rainbow',vmin=60,vmax=80)
	plt.colorbar(s,ax=ax)
	ax.set_xlabel('X (km)')
	ax.set_ylabel('Altitude (m)')
	ax.set_title(' RH (%)')
	fig.savefig(path_save)
	
def TIME_EVOLUTION_PROGVAR_PROFILES(X,Z,dsO,step,X1,X2,SOURCE,CMAP,path_save,dpi,VAR):
	"""This procedure plots the evolution of vertical velocity W in time,
		every 'step'x5min if SOURCE=='OUTPUT' and every 30min if SOURCE=='BUDGETNAM'.
	"""
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	cmap = mpl.colormaps.get_cmap(CMAP)
	indx1,indx2 = np.argmin(np.abs(X.values-X1*1000)),np.argmin(np.abs(X.values-X2*1000))
	N_timeO = dsO.time.shape[0]
	dic_bornes = {'W':[-0.017,0.004],'U':[5,7.5],'V':[-1.5,0],'THT':[295.2,296.2],'THTV':[297.0,298],'RV':[8,13]}
	dic_units = {'W':'m/s','U':'m/s','V':'m/s','THT':'K','THTV':'K','RV':'g/kg'}
	if SOURCE=='HAND':
		N = N_timeO
		dt = 5
		if VAR in ['U','V','W']:
			PROGVAR = dsO[VAR+'T'][:,nhalo:-nhalo-1,nhalo:-nhalo,indx1:indx2].mean(axis=(2,3))	
		elif VAR=='RV':
			PROGVAR = dsO['RVT'][:,nhalo:-nhalo-1,nhalo:-nhalo,indx1:indx2].mean(axis=(2,3))*1000
		elif VAR=='THTV':
			THT = dsO['THT'][:,nhalo:-nhalo-1,nhalo:-nhalo,indx1:indx2].mean(axis=(2,3))
			RVT = dsO['RVT'][:,nhalo:-nhalo-1,nhalo:-nhalo,indx1:indx2].mean(axis=(2,3))
			PROGVAR = Compute_THTV(THT,RVT)
		else:
			PROGVAR = dsO[VAR][:,nhalo:-nhalo-1,nhalo:-nhalo,indx1:indx2].mean(axis=(2,3))		
		for t in range(0,N-step,step):
			ax.plot(PROGVAR[t:t+step,:].mean(axis=0),Z[:-1],c=cmap(t/N),label=str((t+step)*dt)+'min')	
	ax.set_xlabel(VAR+' ('+dic_units[VAR]+')')
	ax.set_ylabel('Altitude (m)')
	ax.set_title('')
	ax.set_ylim([0,600])
	ax.set_xlim(dic_bornes[VAR])
	ax.legend(loc='upper left')
	fig.savefig(path_save)	
	
def TIME_EVOLUTION_PROGVAR_PROFILES_2(X,Z,dsO,dsref,dsmean,step,dt,path_save,dpi,VAR,ZONES):
	"""This procedure plots the time evolution of VAR, on differents ZONES, with the reference profiles.
	
	INPUTS:
	 - X : X dimension DataArray
	 - Z : Z dimension DataArray
	 - dsO : DataSet from MNH output files
	 - dsref : DataSet from diachronic MNH file
	 - dsmean : Dataset built with 'Build_mean_file'
	 - step : width of the time window, in number of instantaneous files
	 - dt : MNH output frequency, in minutes 
	 - path_save : where to save figures
	 - dpi : for figures
	 - VAR : what variable to plot, available : U V W THTv RVT gTHTv
	 - ZONES : X positions where to plot the mean profiles.
	
	
	Note :  - to use NORM=True, work need to be done to define ref values
		- argmax is used here and the vertical grid is discretized so :
			Z[ argmax(d(<THTV>)/dz) ] =/= <Z[ argmax(dTHTV/dz) ]>
		     and this explains that the mean profiles seem to be outliers and not the mean of the smaller time windows profiles averaged.
 	"""
	NORME = False
	cmap ='rainbow'
	
	print('	plotting '+VAR+' ...')
	dic_bornes = {'W':[-0.017,0.004],'U':[5.5,7.6],'V':[-1.5,0],'TH':[294.8,296.2],'THTv':[297.2,298.2],'RVT':[8,13],'gTHTv':[-0.005,0.005]}
	dic_units = {'W':'m/s','U':'m/s','V':'m/s','TH':'K','THTv':'K','RVT':'g/kg','gTHTv':'K/m'}
	cmap = mpl.colormaps.get_cmap(cmap)
	indt_w = -1
	indt_c = -1 # time index of is the last instant
	N = dsO.time.shape[0]
	
	# computing ABLH averaged over 1h and all Y
	THTV = dsmean.THTvm
	gTHTV = THTV.differentiate('level')
	mean_TY_zi = Z[ gTHTV.argmax('level').values ]
	Nx = len(mean_TY_zi)
	ABLH_x_3 = np.zeros(3*Nx)
	ABLH_x_3[:Nx] = mean_TY_zi
	ABLH_x_3[Nx:2*Nx] = mean_TY_zi
	ABLH_x_3[2*Nx:3*Nx] = mean_TY_zi
	smoothed = savitzky_golay(ABLH_x_3, window_size=251, order=2, deriv=0, rate=1/50)[Nx:2*Nx] # fake cyclic conditions
	
	# Computing ABLH averaged over a window of dt*step
	# 	first : moving average in time, then : argmax
	THW_FLX = dsO['THW_FLX'].interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	RCONSW_FLX = dsO['RCONSW_FLX'].interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	THTV = Compute_THTV(dsO)[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(dim=['nj'])
	THT = dsO.THT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	L_zi = np.zeros((N//step,len(X))) 	# size= size(X)
	L_wstar = np.ones((N//step,len(X))) 	# size= size(X)
	L_Q0 = np.zeros((N//step,len(X))) 	# size= size(X)
	L_E0 = np.zeros((N//step,len(X))) 	# size= size(X)
	for t in range(0,N-step,step):
		k = t//step
		# ABLH entre t et t+step
		THTVm = THTV.isel(time=slice(t,t+step)).mean(dim=['time'])
		gTHTV = THTVm.differentiate('level')
		zi = Z[ gTHTV.argmax('level').values ].values
		L_zi[k,:] = zi
		if NORME:
			# w* entre t et t+step
			THTm = THT.isel(time=slice(t,t+step)).mean(dim=['time','nj'])
			THW_FLXm = THW_FLX.isel(time=slice(t,t+step)).mean(dim=['time','nj'])
			RCONSW_FLXm = RCONSW_FLX.isel(time=slice(t,t+step)).mean(dim=['time','nj'])
			THvW_FLXm = THW_FLXm*THTVm/THTm + 0.61*THTm*RCONSW_FLXm
			wstar = ( g*zi*THvW_FLXm.isel(level=0)/THTVm.isel(level=0) ) **(1/3)
			L_wstar[k,:] = wstar
			L_Q0[k,:] = THvW_FLXm.isel(level=0)
			L_E0[k,:] = RCONSW_FLXm.isel(level=0)
		
	# Normalization for S1 and ref
	print('	Normalizing with values (cold,warm):')
	gTHTV_w = Compute_THTV(dsref['warm']['nomean']['Mean']['MEAN_TH'],dsref['warm']['nomean']['Mean']['MEAN_RV'])[indt_w].differentiate('level_les')
	gTHTV_c = Compute_THTV(dsref['cold']['nomean']['Mean']['MEAN_TH'],dsref['cold']['nomean']['Mean']['MEAN_RV'])[indt_c].differentiate('level_les')
	zi_c,      zi_w     = ( Z[gTHTV_c.argmax('level_les').values].values, 
				Z[gTHTV_w.argmax('level_les').values].values )
	print('	   zi (m)', np.round(zi_c,1), np.round(zi_w,1) )
	if VAR in ['U','V','W']:
		L_norm , norm_c, norm_w = L_wstar, 1,1
		#print('	   u* (m/s)', np.round(norm_c,1), np.round(norm_w,1) )
	elif VAR == 'RVT':
		L_norm,norm_c, norm_w = L_E0/L_wstar,1,1
	elif VAR == 'THTv':
		L_norm,norm_c, norm_w = L_Q0/L_wstar,1,1
		#print('	   tht* (m/s)', np.round(norm_c,1), np.round(norm_w,1) )
	elif VAR == 'gTHTv':
		L_norm,norm_c, norm_w = zi*L_Q0/L_wstar,1,1
	
	# Plotting
	fig, ax = plt.subplots(1,len(ZONES),figsize = (5*len(ZONES),5),constrained_layout=True,dpi=dpi)
	for i,zone in enumerate(ZONES):
		X1,X2 = ZONES[zone][0],ZONES[zone][1]
		print('ZONE:',X1,X2)
		indx1,indx2 = np.argmin(np.abs(X.values-X1*1000)),np.argmin(np.abs(X.values-X2*1000))
		zi_m = smoothed[indx1:indx2].mean()  # averaging ABL heigth of 1h averaged zi over the zone
		
		print('zi_m',zi_m)
		
		# setting the variable to plot
		REFVAR = {}
		PROGVAR = {}		
		if VAR in ['U','V','W']:
			PROGVAR = dsO[VAR+'T'][:,nhalo:-nhalo,nhalo:-nhalo,indx1:indx2].mean(axis=(2,3))	
			REFVAR['warm'] = dsref['warm']['nomean']['Mean']['MEAN_'+VAR][indt_w]
			REFVAR['cold'] = dsref['cold']['nomean']['Mean']['MEAN_'+VAR][indt_c]
			MEANVAR = dsmean[VAR+'m'][:,indx1:indx2].mean(dim='ni')
		elif VAR=='RVT':
			PROGVAR = dsO['RVT'][:,nhalo:-nhalo-1,nhalo:-nhalo,indx1:indx2].mean(dim=['ni','nj'])*1000
			REFVAR['warm'] = dsref['warm']['nomean']['Mean']['MEAN_'+VAR][indt_w]*1000
			REFVAR['cold'] = dsref['cold']['nomean']['Mean']['MEAN_'+VAR][indt_c]*1000
			MEANVAR = dsmean[VAR+'m'][:,indx1:indx2].mean(dim='ni')*1000
		elif VAR=='THTv':
			PROGVAR = THTV.isel(ni=slice(indx1,indx2)).mean(dim=['ni'])
			REFVAR['warm'] = Compute_THTV(dsref['warm']['nomean']['Mean']['MEAN_TH'],dsref['warm']['nomean']['Mean']['MEAN_RV'])[indt_w]
			REFVAR['cold'] = Compute_THTV(dsref['cold']['nomean']['Mean']['MEAN_TH'],dsref['cold']['nomean']['Mean']['MEAN_RV'])[indt_c]
			MEANVAR = dsmean[VAR+'m'][:,indx1:indx2].mean(dim='ni')
		elif VAR=='gTHTv':
			PROGVAR = THTV.differentiate('level').isel(ni=slice(indx1,indx2)).mean(dim=['ni'])
			REFVAR['warm'] = Compute_THTV(dsref['warm']['nomean']['Mean']['MEAN_TH'],dsref['warm']['nomean']['Mean']['MEAN_RV'])[indt_w].differentiate('level_les')
			REFVAR['cold'] = Compute_THTV(dsref['cold']['nomean']['Mean']['MEAN_TH'],dsref['cold']['nomean']['Mean']['MEAN_RV'])[indt_c].differentiate('level_les')
			MEANVAR = dsmean['THTvm'].differentiate('level')[:,indx1:indx2].mean(dim='ni')
		
		for t in range(0,N-step,step):
			zi = L_zi[t//step,indx1:indx2].mean() # averaging ABL heigth of dt*step averaged zi over the zone
			norm = L_norm[t//step,indx1:indx2].mean()
			print(zi)
			ax[i].set_title('X='+str(ZONES[zone])+'km',loc='right')
			if not NORME: norm=1
			ax[i].plot(PROGVAR[t:t+step,:].mean(axis=0)/norm,Z/zi,c=cmap(t/N),label='['+str(t*(dt))+'-'+str((t+step)*dt)+']min')
		if NORME:
			norm_m = L_norm[:,indx1:indx2].mean()
		else:
			norm_m = 1
		ax[i].plot(MEANVAR/norm_m,Z/zi_m,c='k',label='[0,60]min')
		ax[i].plot(REFVAR['warm']/norm_w,dsref['warm']['000'].level_les/zi_w,c='r',ls='--',label='ref:warm')
		ax[i].plot(REFVAR['cold']/norm_c,dsref['cold']['000'].level_les/zi_c,c='b',ls='--',label='ref:cold')		
		ax[i].set_title('')
		ax[i].set_ylim([0,1.3])
		ax[i].set_ylabel('z/zi(x)')	
		ax[i].set_xlim(dic_bornes[VAR])
		ax[i].grid()
	if VAR=='gTHTv':
		nameVar = r'$<\partial \theta_v / \partial z>$'
	elif VAR=='THTv':
		nameVar = r'$\theta_v$'
	elif VAR=='RVT':
		nameVar = r'$r_v$'
	else:
		nameVar = VAR
	fig.suptitle(nameVar+' ('+dic_units[VAR]+')')
	ax[0].legend(loc='upper left')
	fig.savefig(path_save+VAR+'_Zprofiles_at_t_NORM.png')

def TIME_EVOLUTION_ET_PROFILES(X,Z,step,dt,dsO,dsmean,dsref,CMAP,path_save,dpi,ZONES):
	"""This procedure plots the evolution of total turbulent kinetic energy in time,
		every 'step'x5min as SOURCE=='OUTPUT'
	"""
	#zi = 600
	indt_c = -1
	indt_w = indt_c
	N = dsO.time.shape[0]
	
	# computing ABLH averaged over 1h and all Y
	THTV = dsmean.THTvm
	gTHTV = THTV.differentiate('level')
	mean_TY_zi = Z[ gTHTV.argmax('level').values ]
	Nx = len(mean_TY_zi)
	ABLH_x_3 = np.zeros(3*Nx)
	ABLH_x_3[:Nx] = mean_TY_zi
	ABLH_x_3[Nx:2*Nx] = mean_TY_zi
	ABLH_x_3[2*Nx:3*Nx] = mean_TY_zi
	smoothed = savitzky_golay(ABLH_x_3, window_size=251, order=2, deriv=0, rate=1/50)[Nx:2*Nx] # fake cyclic conditions
	
	# Computing ABLH averaged over a window of dt*step
	# 	first : moving average in time, then : argmax
	THTV = Compute_THTV(dsO)[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(dim=['nj'])
	L_zi = np.zeros((N//step,len(X))) 	# size= size(X)
	for t in range(0,N-step,step):
		k = t//step
		# ABLH entre t et t+step
		THTVm = THTV.isel(time=slice(t,t+step)).mean(dim=['time'])
		gTHTV = THTVm.differentiate('level')
		zi = Z[ gTHTV.argmax('level').values ].values
		L_zi[k,:] = zi
	
	# ref values
	ET_c = 0.5*( dsref['cold']['nomean']['Resolved']['RES_U2'][indt_c] + dsref['cold']['nomean']['Subgrid']['SBG_U2'][indt_c] +
	 		dsref['cold']['nomean']['Resolved']['RES_V2'][indt_c] + dsref['cold']['nomean']['Subgrid']['SBG_V2'][indt_c] +
	 		dsref['cold']['nomean']['Resolved']['RES_W2'][indt_c] + dsref['cold']['nomean']['Subgrid']['SBG_W2'][indt_c] )
	ET_w = 0.5*( dsref['warm']['nomean']['Resolved']['RES_U2'][indt_w] + dsref['warm']['nomean']['Subgrid']['SBG_U2'][indt_w] +
	 		dsref['warm']['nomean']['Resolved']['RES_V2'][indt_w] + dsref['warm']['nomean']['Subgrid']['SBG_V2'][indt_w] +
	 		dsref['warm']['nomean']['Resolved']['RES_W2'][indt_w] + dsref['warm']['nomean']['Subgrid']['SBG_W2'][indt_w] )
	 		
	
	gTHTV_w = Compute_THTV(dsref['warm']['nomean']['Mean']['MEAN_TH'],dsref['warm']['nomean']['Mean']['MEAN_RV'])[indt_w].differentiate('level_les')
	gTHTV_c = Compute_THTV(dsref['cold']['nomean']['Mean']['MEAN_TH'],dsref['cold']['nomean']['Mean']['MEAN_RV'])[indt_c].differentiate('level_les')
	zi_c,      zi_w     = ( Z[gTHTV_c.argmax('level_les').values].values, 
				Z[gTHTV_w.argmax('level_les').values].values )
	
	
	fig, ax = plt.subplots(1,len(ZONES),figsize = (5*len(ZONES),5),constrained_layout=True,dpi=dpi)
	cmap = mpl.colormaps.get_cmap(CMAP)
	for i,zone in enumerate(ZONES):
		X1,X2 = ZONES[zone][0],ZONES[zone][1]
		indx1,indx2 = np.argmin(np.abs(X.values-X1*1000)),np.argmin(np.abs(X.values-X2*1000))
		zi_m = smoothed[indx1:indx2].mean()  # averaging ABL heigth of 1h averaged zi over the zone
		# computing local ET from indx1 to indx2
		U = dsO.UT.interp({'ni_u':dsO.ni})[:,nhalo:-nhalo,nhalo:-nhalo,indx1:indx2] # grid : 2
		V = dsO.VT.interp({'nj_v':dsO.nj})[:,nhalo:-nhalo,nhalo:-nhalo,indx1:indx2] # grid : 3
		W = dsO.WT.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,indx1:indx2] # grid : 4
		U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
		V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
		TKET = dsO.TKET[:,nhalo:-nhalo,nhalo:-nhalo,indx1:indx2].mean(dim=['nj']) # sbgrid tke
		Um = dsmean.Um[:,indx1:indx2]
		Vm = dsmean.Vm[:,indx1:indx2]
		Wm = dsmean.Wm[:,indx1:indx2]
		
		for t in range(0,N-step,step):
			zi = L_zi[t//step,indx1:indx2].mean() # averaging ABL heigth of dt*step averaged zi over the zone
			sgs_tke = TKET[t:t+step,:,:].mean(dim=['time'])
			u_fluc = (U - Um)
			v_fluc = (V - Vm)
			w_fluc = (W - Wm)
			ET = ( 0.5*( u_fluc**2 + v_fluc**2 + w_fluc**2 ) + sgs_tke )[t:t+step,:,:,:].mean(dim=['time','nj','ni'])
			ax[i].plot(ET ,Z/zi,c=cmap(t/N),label='['+str(t*(dt))+'-'+str((t+step)*dt)+']min')	
		ax[i].set_ylim([0,1.2])
		ax[i].set_title('X='+str(ZONES[zone])+'km',loc='right')
		ax[i].plot(dsmean.ETm[:,indx1:indx2].mean('ni'),Z/zi_m,c='k',label='<ET>')
		ax[i].plot(ET_c,Z/zi_c,c='b',ls='--',label='refC')
		ax[i].plot(ET_w,Z/zi_w,c='r',ls='--',label='refW')
		ax[i].grid()
	ax[0].set_ylabel('z/zi')
	ax[0].legend(loc='upper right')
	fig.suptitle('ET (m2 s-2)')
	fig.savefig(path_save+'ET_Zprofiles_at_t_NORM.png')
	
def TIME_EVOLUTION_ABLH(dsmean,DATA,PERIOD,DT,LISTE,X,Z,CMAP,path_save,dpi):
	"""Time evolution of the boundary layer height, for the simulations
		listed in LISTE and with details in DATA and PERIOD
	
	
	TBD : 	- inputs
		- move step and dt to function call
			
	NOTE : argmax is used here and the vertical grid is discretized so
	
		Z[ argmax(d(<THTV>)/dz) ] =/= <Z[ argmax(dTHTV/dz) ]
	"""	
	step = 20 # = moving average over 10min
	dt = 0.5 # in minutes
	WSIZE = 251
	ORD = 2
	
	# ABLH with dsmean
	THTV = dsmean.THTvm
	gTHTV = THTV.differentiate('level')
	mean_TY_zi = Z[ gTHTV.argmax('level').values ]
	Nx = len(mean_TY_zi)
	ABLH_x_3 = np.zeros(3*Nx)
	ABLH_x_3[:Nx] = mean_TY_zi
	ABLH_x_3[Nx:2*Nx] = mean_TY_zi
	ABLH_x_3[2*Nx:3*Nx] = mean_TY_zi	
	smoothed_mean_TY= savitzky_golay(ABLH_x_3, window_size=WSIZE, order=ORD, deriv=0, rate=1/50)[Nx:2*Nx]
	
	cmap = mpl.colormaps.get_cmap(CMAP)
#	fig2, ax2 = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	for k,case in enumerate(LISTE):
		print("	",case)
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
		ds,what = xr.open_mfdataset(DATA[case]),PERIOD[case][-1]	# ,combine='nested',concat_dim='time'
		dt = DT[case] # min
		start,end = PERIOD[case][0],PERIOD[case][1]
		Ntime = ds.time.shape[0]
		X_time = np.linspace(start,end,Ntime)
		#ABLH = np.zeros(X_time.shape)
		THT,RVT = ds.THT,ds.RVT
		THTV = Compute_THTV(THT,RVT)[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		
		# V0
		THTVm = THTV[:,:,:,:].mean(dim=['time','nj'])
		indzi = THTVm.differentiate('level').argmax('level')
		ABLHmean = Z[indzi.values]
		# smoothing the moving time averaged ABLH
		temp = ABLHmean
		Nx = len(temp)
		ABLH_x_3 = np.zeros(3*Nx)
		ABLH_x_3[:Nx] = temp
		ABLH_x_3[Nx:2*Nx] = temp
		ABLH_x_3[2*Nx:3*Nx] = temp
		smoothed_mean = savitzky_golay(ABLH_x_3, window_size=WSIZE, order=ORD, deriv=0, rate=1/50)[Nx:2*Nx]

		THTV = Compute_THTV(THT,RVT)[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		THTV = THTV[:,:,:,:].mean(dim=['nj'])
		SUM = np.zeros(len(X))
		for t in range(0,Ntime-step,step): 
			print(ds.time[t:t+step].values)
			print(THTV[t:t+step,:,:].shape)
			THTVm = THTV[t:t+step,:,:].mean(dim=['time'])
			indzi = THTVm.differentiate('level').argmax('level')
			zi = Z[indzi.values]
			# smoothing the moving time averaged ABLH
			temp = zi
			Nx = len(temp)
			ABLH_x_3 = np.zeros(3*Nx)
			ABLH_x_3[:Nx] = temp
			ABLH_x_3[Nx:2*Nx] = temp
			ABLH_x_3[2*Nx:3*Nx] = temp
			smoothed = savitzky_golay(ABLH_x_3, window_size=WSIZE, order=ORD, deriv=0, rate=1/50)[Nx:2*Nx]
			# plotting the moving averaged ABLH
			ax.plot(X/1000, smoothed,c=cmap(t/Ntime),label='['+str(t*dt)+'-'+str((t+step)*dt)+']min')	
			#ax.plot(X/1000, zi,c=cmap(t/Ntime),label='['+str(t*dt)+'-'+str((t+step)*dt)+']min')	
		
		#ax.plot( X/1000, mean_TY_zi,c='k',label='dsmean',ls='--')
		ax.plot( X/1000, smoothed_mean_TY,c='k',label='[0,60]min',ls='--')
		ax.set_title('Time evolution of ABLH of '+case)
		ax.set_ylabel('Altitude (m)')
		ax.set_xlabel('X (km)')
		ax.set_ylim([400,800])
		ax.legend()
		fig.savefig(path_save+'TimeEvolution_ABLH_'+case+'_X.png')
	
	
	# KEEP THE FOLLOWING, to be cleaned	
	
def SKEWNESS_W_FLUC(X,Y,Z,Time,dsO,dsmean,dsflx,nhalo,Tstart,Tstop,window,manual_label,path_save,dpi):
	"""This procedure plots the assimetry parameter (skewness) of the distribution
		of the vertical velocity fluctuations :
		
			Sk = <www>/<ww>**(3/2)
			
	Note : only the resolved part is taken into account as the triple subgrid correlations are not parametrized in the model. 
	"""
	W = dsO.WT[:,:,nhalo:-nhalo,nhalo:-nhalo].interp({'level_w':dsO.level})[nhalo:-nhalo]
	Wm = dsmean.Wm[:,:]
	Wm = Wm.expand_dims(dim={"nj":Y,"time":Time},axis=(2,0))
	w_prime = W - Wm
	FLX_WW = dsflx.FLX_WW
	Sk = MeanTurb(w_prime*w_prime*w_prime,Tstart,Tstop,window)/FLX_WW**(3/2)
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	s = ax.pcolormesh(X/1000,Z,Sk,cmap='rainbow')
	plt.colorbar(s,ax=ax)
	CS = ax.contour(X/1000,Z,dsmean.Wm[:,:]*100,levels=[-1.5,-1,-0.5],colors='grey',linestyles='solid',linewidths=1)
	ax.clabel(CS, inline=True, fontsize=8,inline_spacing=2.5,manual=manual_label)
	ax.set_xlabel('X (km)')
	ax.set_ylabel('Altitude (m)')
	ax.set_title("Skewness of w', with mean W contours (cm/s)")
	ax.set_ylim([0,600])
	fig.savefig(path_save)
	
def ANOMALY_WIND_AND_W_PRESS(X,Z,dsmean,ds_hbudget,w_press_levels,path_save,cmapU,dpi):
	"""This procedure plots :
		- mean U - meanU(x=0) in color
		- pressure term in W budget w_pres in contours
	"""

	Ylabel='Altitude (m)'
	loccb='vertical'
	
	Ylim = [0,600]
	
	U0 = dsmean.Um[:,0]
	U0.expand_dims(dim={'ni':X},axis=1)
	U = dsmean.Um - U0
	w_pres = ds_hbudget.w_pres
	
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
	s = ax.pcolormesh(X/1000,Z,U,cmap=cmapU,vmin=-1,vmax=1)
	plt.colorbar(s,ax=ax,orientation=loccb)
	CS = ax.contour(X[:-1]/1000,Z[:-1],w_pres,levels=w_press_levels,colors='grey',linestyles='solid',linewidths=1.0)
	ax.clabel(CS, inline=True, fontsize=5,inline_spacing=2.5)
	ax.set_ylim(Ylim)
	ax.set_xlabel('X (km)')
	ax.set_ylabel('Altitude (m)')
	ax.set_title('U-U(x=0) in color (m/s), w_press in contour')
	fig.savefig(path_save)
	
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
	s = ax.pcolormesh(X[:-1]/1000,Z[:-1],w_pres,cmap='rainbow',vmin=-0.02,vmax=0.0)
	CS = ax.contour(X[:-1]/1000,Z[:-1],w_pres,levels=w_press_levels,colors='grey',linestyles='solid',linewidths=1.0)
	ax.clabel(CS, inline=True, fontsize=5,inline_spacing=2.5)
	plt.colorbar(s,ax=ax)
	ax.set_ylim(Ylim)
	ax.set_xlabel('X (km)')
	ax.set_ylabel('Altitude (m)')
	ax.set_title('w_press')
	
def ANOMALY_WIND_AND_THTV(X,Z,dsmean,dsINI,theta_levels,path_save,cmapU,dpi):
	Ylabel='Altitude (m)'
	loccb='vertical'
	Ylim = [0,600]
	
	U0 = dsmean.Um[:,0]
	U0.expand_dims(dim={'ni':X},axis=1)
	U = dsmean.Um - U0
	thetav = dsmean.THTvm
	THT_ini = dsINI.THT[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=1)
	RVT_ini = dsINI.RVT[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo].mean(axis=1)
	thetavref = Compute_THTV(THT_ini,RVT_ini)
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
	s = ax.pcolormesh(X/1000,Z,U,cmap=cmapU,vmin=-1,vmax=1)
	plt.colorbar(s,ax=ax,orientation=loccb)
	CS = ax.contour(X/1000,Z,thetav-thetavref,levels=theta_levels,colors='grey',linestyles='solid',linewidths=1.0)
	ax.clabel(CS, inline=True, fontsize=5,inline_spacing=2.5)
	ax.set_ylim(Ylim)
	ax.set_xlabel('X (km)')
	ax.set_ylabel('Altitude (m)')
	ax.set_title('U-U(x=0) in color (m/s), mean thetav - thtvref in contour')
	fig.savefig(path_save)
	
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
	s = ax.pcolormesh(X/1000,Z,thetav-thetavref,cmap='rainbow',vmin=0.,vmax=0.5)
	CS = ax.contour(X/1000,Z,thetav-thetavref,levels=theta_levels,colors='grey',linestyles='solid',linewidths=1.0)
	ax.clabel(CS, inline=True, fontsize=5,inline_spacing=2.5)
	plt.colorbar(s,ax=ax)
	ax.set_ylim(Ylim)
	ax.set_xlabel('X (km)')
	ax.set_ylabel('Altitude (m)')
	ax.set_title('mean thetav - thtvref in colors')
	
def T_OVERSHOOT_FLX_WTHTV(X,Z,dataSST,dsflx,dsmean,liste_x,Q0_atX,path_save,dpi):
	"""This procedure is plotting a slice of the vertical flux of virtual
		potential temperature and total turbulent kinetic energy
		with profiles at differents X positions
	"""
	indx = []
	indxQ0 = np.argmin(np.abs(X.values-Q0_atX))
	for x in range(len(liste_x)):
		indx.append(np.argmin(np.abs(X.values-liste_x[x]*1000)))

	Qnorme = dsflx.FLX_THvW[0,indxQ0]
	bornes = [-1,3]
	nameflx = r"$< w  \theta_v >/Q_v^*$"
	ETnorme = dsmean.u_star[indxQ0]**2
	UWnorme = dsmean.u_star[indxQ0]**2
	bornesET = [1,10]
	nameET = 'TKE/u*²'
	zi=600
	nameZ = r'z/z$_i$'
	nameX = 'X (km)'
	bornesZ = [0,1.2]
		
	FLX = dsflx.FLX_THvW/Qnorme
	ET = dsmean.ETm/ETnorme
	UW = dsflx.FLX_UW/UWnorme
	# FLX plot
	axes = []
	fig, ax = plt.subplots(1,1,figsize = (10,4),constrained_layout=True,dpi=dpi)
	s = ax.pcolormesh(X/1000,Z/zi,FLX,cmap=cmocean.cm.thermal,vmin=bornes[0],vmax=bornes[1])
	Add_SST_bar(X/1000,Z/zi,4,dataSST,ax)
	plt.colorbar(s,ax=ax,pad=0.005)
	ax.set_ylim(bornesZ)
	ax.set_ylabel(nameZ)
	ax.set_xlabel(nameX)
	ax.set_title(nameflx)
	fig.savefig(path_save+'flx_wthtv.png')
	# FLX plot
	axes = []
	fig, ax = plt.subplots(1,1,figsize = (10,4),constrained_layout=True,dpi=dpi)
	s = ax.pcolormesh(X/1000,Z/zi,UW,cmap=cmocean.cm.matter_r,vmin=-1.5,vmax=0.2)
	Add_SST_bar(X/1000,Z/zi,4,dataSST,ax)
	plt.colorbar(s,ax=ax,pad=0.005)
	ax.set_ylim(bornesZ)
	ax.set_ylabel(nameZ)
	ax.set_xlabel(nameX)
	ax.set_title(r"$< uw >/u^{*2}$")
	fig.savefig(path_save+'flx_uw.png')

	# ET plot

def PLOT_FLUXES(X,Z,liste_x,liste_z,dsflx,dsref,B_FLXTOTAL,B_HEIGHT_OF_SWITCH,path_save,dpi):
	"""This procedure is giving a first look of the fluxes,
		profiles and f(x).
		
		TBD : liste of inputs.
				modify "fonction of X" to have both total and SGS
	"""
	indx,indz = [],[]
	indt = -1 # for ref sim
	
	for x in range(len(liste_x)):
		indx.append(np.argmin(np.abs(X.values-liste_x[x]*1000)))
	for z in range(len(liste_z)):
		indz.append(np.argmin(np.abs(Z.values-liste_z[z])))
	print('	plotting total fluxes : '+str(B_FLXTOTAL))
	if B_FLXTOTAL:
		toadd = '_tot'
	else:
		toadd = '_split'
	ABLH = 600 # m, is mean (t,x,y) height of ABL of C
	###
	### fonction of X
	###	
	if len(liste_z)!=0:
		# Variances 
		vmin,vmax = 0.,0.41
		fig, ax = plt.subplots(len(liste_z),1,figsize = (10,20),constrained_layout=True,dpi=dpi)
		fig.suptitle(r"Variances $\overline{u'u'}$, $\overline{v'v'}$, $\overline{w'w'}$ (m2 s-2)")
		for kz,z in enumerate(liste_z):
			ax[kz].plot(X/1000,dsflx.FLX_UU[indz[kz],:],c='k',label=r"$\overline{u'u'}$")
			ax[kz].set_title('z='+str(z)+'m',loc='right')
			ax[kz].plot(X/1000,dsflx.FLX_VV[indz[kz],:],c='g',label=r"$\overline{v'v'}$")
			ax[kz].plot(X/1000,dsflx.FLX_WW[indz[kz],:],c='b',label=r"$\overline{w'w'}$")
			ax[kz].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			ax[kz].yaxis.major.formatter._useMathText = True	
		for k,axe in enumerate(ax):
			if k<len(ax)-1:
				axe.tick_params(axis='both',labelbottom=False)
			axe.set_ylim([vmin,vmax])
			axe.set_xlim([X[0]/1000,X[-1]/1000])
		ax[0].legend()
		fig.savefig(path_save+'variance_X'+toadd+'.png')
		# Covariances
		vmin,vmax = -0.15,0.22
		fig, ax = plt.subplots(len(liste_z),1,figsize = (10,20),constrained_layout=True,dpi=dpi)
		fig.suptitle(r"Covariances $\overline{u'w'}$, $\overline{v'w'}$, $\overline{w'w'}$ (m2 s-2)")
		for kz,z in enumerate(liste_z):
			ax[kz].plot(X/1000,dsflx.FLX_UW[indz[kz],:],c='k',label=r"$\overline{u'w'}$")
			ax[kz].set_title('z='+str(z)+'m',loc='right')
			ax[kz].plot(X/1000,dsflx.FLX_VW[indz[kz],:],c='g',label=r"$\overline{v'w'}$")
			ax[kz].plot(X/1000,dsflx.FLX_WW[indz[kz],:],c='b',label=r"$\overline{w'w'}$")
			ax[kz].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			ax[kz].yaxis.major.formatter._useMathText = True	
		for k,axe in enumerate(ax):
			if k<len(ax)-1:
				axe.tick_params(axis='both',labelbottom=False)
			axe.set_ylim([vmin,vmax])
			axe.set_xlim([X[0]/1000,X[-1]/1000])
		ax[0].legend()	
		fig.savefig(path_save+'covariance_X'+toadd+'.png')
		#  w'thtv', w'tht' and w'rv'
		fig, ax = plt.subplots(len(liste_z),1,figsize = (10,20),constrained_layout=True,dpi=dpi)
		fig.suptitle(r"Covariances"+'\n'+r"$\overline{rv'w'}$ (left; g kg-1 m s-1), $\overline{w'\theta_v'}$, $\overline{w'\theta'}$ (right; K m s-1)")
		for kz,z in enumerate(liste_z):
			ax[kz].plot(X[0]/1000,dsflx.FLX_RvW[indz[kz],0],c='r',label=r"$\overline{w'\theta_v'}$") # dummy for legend
			ax[kz].plot(X[0]/1000,dsflx.FLX_RvW[indz[kz],0],c='b',label=r"$\overline{w'\theta'}$") # dummy for legend
			ax[kz].plot(X/1000,dsflx.FLX_RvW[indz[kz],:]*1000,c='k',label=r"$\overline{w'r_v'}$")
			ax[kz].set_title('z='+str(z)+'m',loc='right')
			ax[kz].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			ax[kz].yaxis.major.formatter._useMathText = True
			ax2 = ax[kz].twinx()
			ax2.plot(X/1000,dsflx.FLX_THvW[indz[kz],:],c='r',label='thvw')
			ax2.plot(X/1000,dsflx.FLX_THW[indz[kz],:],c='b',label='thw')	
			ax[kz].set_ylim([-0.1,0.1])
			ax2.set_ylim([-0.015,0.015])
			ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			ax2.yaxis.major.formatter._useMathText = True
		for k,axe in enumerate(ax):
			if k<len(ax)-1:
				axe.tick_params(axis='both',labelbottom=False)
			axe.set_xlim([X[0]/1000,X[-1]/1000])
		ax[0].legend()	
		ax[-1].set_xlabel('X (km)')
		fig.savefig(path_save+'thermo_covariance_X'+toadd+'.png')
	###
	### fonction of Z
	###
	if len(liste_x)!=0:
		# Variances
		vmin,vmax = 0.,0.41
		# -> S1
		fig, ax = plt.subplots(1,len(liste_x),figsize = (20,5),constrained_layout=True,dpi=dpi)
		fig.suptitle(r"Variances $\overline{u'u'}$, $\overline{v'v'}$, $\overline{w'w'}$ (m2 s-2)")
		for kx,x in enumerate(liste_x):
			if B_FLXTOTAL:
				ax[kx].plot(dsflx.FLX_UU[:,indx[kx]],Z/ABLH,c='k',label=r"$\overline{u'u'}$")
				ax[kx].plot(dsflx.FLX_VV[:,indx[kx]],Z/ABLH,c='g',label=r"$\overline{v'v'}$")
				ax[kx].plot(dsflx.FLX_WW[:,indx[kx]],Z/ABLH,c='b',label=r"$\overline{w'w'}$")
			else:
				ax[kx].plot((dsflx.FLX_UU-dsflx.FLX_UU_s)[:,indx[kx]],Z/ABLH,c='k',label=r"$\overline{u'u'}$")
				ax[kx].plot(dsflx.FLX_UU_s[:,indx[kx]],Z/ABLH,c='k',ls='--')
				ax[kx].plot((dsflx.FLX_VV-dsflx.FLX_VV_s)[:,indx[kx]],Z/ABLH,c='g',label=r"$\overline{v'v'}$")
				ax[kx].plot(dsflx.FLX_VV_s[:,indx[kx]],Z/ABLH,c='g',ls='--')
				ax[kx].plot((dsflx.FLX_WW-dsflx.FLX_WW_s)[:,indx[kx]],Z/ABLH,c='b',label=r"$\overline{w'w'}$")
				ax[kx].plot(dsflx.FLX_WW_s[:,indx[kx]],Z/ABLH,c='b',ls='--')
			ax[kx].set_title('x='+str(x)+'km',loc='right')
			ax[kx].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax[kx].xaxis.major.formatter._useMathText = True
		for k,axe in enumerate(ax):
			if k>0:
				axe.tick_params(axis='both',labelleft=False)
			axe.set_xlim([vmin,vmax])
			axe.set_ylim([0,1.2])
			axe.grid()
		ax[0].legend()
		ax[0].set_ylabel('z/zi')
		fig.savefig(path_save+'variance_Z'+toadd+'.png')
		# -> REF
		fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
		fig.suptitle(r"Variances $\overline{u'u'}$, $\overline{v'v'}$, $\overline{w'w'}$ (m2 s-2)")
		for k,case in enumerate(['warm','cold']):
			zi = dsref[case]['nomean']['Misc']['BL_H'][indt].values
			if B_FLXTOTAL:
				ax[k].plot( (dsref[case]['nomean']['Resolved']['RES_U2'] + 
							dsref[case]['nomean']['Subgrid']['SBG_U2'])[indt,:],	Z/zi,c='k',label=r"$\overline{u'u'}$")
				ax[k].plot((dsref[case]['nomean']['Resolved']['RES_V2'] + 
							dsref[case]['nomean']['Subgrid']['SBG_V2'])[indt,:],	Z/zi,c='g',label=r"$\overline{v'v'}$")
				ax[k].plot((dsref[case]['nomean']['Resolved']['RES_W2'] + 
							dsref[case]['nomean']['Subgrid']['SBG_W2'])[indt,:],	Z/zi,c='b',label=r"$\overline{w'w'}$")
			else:
				ax[k].plot(dsref[case]['nomean']['Resolved']['RES_U2'][indt,:],		Z/zi,c='k',label=r"$\overline{u'u'}$")
				ax[k].plot(dsref[case]['nomean']['Subgrid']['SBG_U2'][indt,:],		Z/zi,c='k',ls='--')
				ax[k].plot(dsref[case]['nomean']['Resolved']['RES_V2'][indt,:],		Z/zi,c='g',label=r"$\overline{v'v'}$")
				ax[k].plot(dsref[case]['nomean']['Subgrid']['SBG_V2'][indt,:],		Z/zi,c='g',ls='--')
				ax[k].plot(dsref[case]['nomean']['Resolved']['RES_W2'][indt,:],		Z/zi,c='b',label=r"$\overline{w'w'}$")
				ax[k].plot(dsref[case]['nomean']['Subgrid']['SBG_W2'][indt,:],		Z/zi,c='b',ls='--')
			ax[k].set_title(case,loc='right')
			ax[k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax[k].xaxis.major.formatter._useMathText = True
			ax[k].set_xlim([vmin,vmax])
			ax[k].set_ylim([0,1.2])
			ax[k].grid()
		ax[1].tick_params(axis='both',labelleft=False)
		ax[0].legend()
		ax[0].set_ylabel('z/zi')
		fig.savefig(path_save+'ref_variance_Z'+toadd+'.png')
		
		# Covariances
		vmin,vmax = -0.1,0.1
		# -> S1
		fig, ax = plt.subplots(1,len(liste_x),figsize = (20,5),constrained_layout=True,dpi=dpi)
		fig.suptitle(r"Covariances $\overline{u'w'}$, $\overline{v'w'}$, $\overline{w'w'}$ (m2 s-2)")
		for kx,x in enumerate(liste_x):
			if B_FLXTOTAL:
				ax[kx].plot(dsflx.FLX_UW[:,indx[kx]],Z/ABLH,c='k',label=r"$\overline{u'w'}$")
				ax[kx].plot(dsflx.FLX_VW[:,indx[kx]],Z/ABLH,c='g',label=r"$\overline{v'w'}$")
				ax[kx].plot(dsflx.FLX_UV[:,indx[kx]],Z/ABLH,c='b',label=r"$\overline{u'v'}$")
			else:
				ax[kx].plot((dsflx.FLX_UW-dsflx.FLX_UW_s)[:,indx[kx]],Z/ABLH,c='k',label=r"$\overline{u'w'}$")
				ax[kx].plot(dsflx.FLX_UW_s[:,indx[kx]],Z/ABLH,c='k',ls='--')
				ax[kx].plot((dsflx.FLX_VW-dsflx.FLX_VW_s)[:,indx[kx]],Z/ABLH,c='g',label=r"$\overline{v'w'}$")
				ax[kx].plot(dsflx.FLX_VW_s[:,indx[kx]],Z/ABLH,c='g',ls='--')
				ax[kx].plot((dsflx.FLX_UV-dsflx.FLX_UV_s)[:,indx[kx]],Z/ABLH,c='b',label=r"$\overline{u'v'}$")
				ax[kx].plot(dsflx.FLX_UV_s[:,indx[kx]],Z/ABLH,c='b',ls='--')
			ax[kx].set_title('x='+str(x)+'km',loc='right')
			ax[kx].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax[kx].xaxis.major.formatter._useMathText = True
		for k,axe in enumerate(ax):
			if k>0:
				axe.tick_params(axis='both',labelleft=False)
			axe.set_xlim([vmin,vmax])
			axe.set_ylim([0,1.2])
			axe.vlines(0,0,600,color='grey',ls='--')
			axe.grid()
		ax[0].legend()
		ax[0].set_ylabel('z/zi')	
		fig.savefig(path_save+'covariance_Z'+toadd+'.png')
		# -> REF
		fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
		fig.suptitle(r"Covariances $\overline{u'w'}$, $\overline{v'w'}$, $\overline{w'w'}$ (m2 s-2)")
		for k,case in enumerate(['warm','cold']):
			zi = dsref[case]['nomean']['Misc']['BL_H'][indt].values
			if B_FLXTOTAL:
				ax[k].plot( (dsref[case]['nomean']['Resolved']['RES_WU'] + 
							dsref[case]['nomean']['Subgrid']['SBG_WU'])[indt,:],	Z/zi,c='k',label=r"$\overline{u'u'}$")
				ax[k].plot((dsref[case]['nomean']['Resolved']['RES_WV'] + 
							dsref[case]['nomean']['Subgrid']['SBG_WV'])[indt,:],	Z/zi,c='g',label=r"$\overline{v'v'}$")
				ax[k].plot((dsref[case]['nomean']['Resolved']['RES_UV'] + 
							dsref[case]['nomean']['Subgrid']['SBG_UV'])[indt,:],	Z/zi,c='b',label=r"$\overline{w'w'}$")
			else:
				ax[k].plot(dsref[case]['nomean']['Resolved']['RES_WU'][indt,:],		Z/zi,c='k',label=r"$\overline{u'u'}$")
				ax[k].plot(dsref[case]['nomean']['Subgrid']['SBG_WU'][indt,:],		Z/zi,c='k',ls='--')
				ax[k].plot(dsref[case]['nomean']['Resolved']['RES_WV'][indt,:],		Z/zi,c='g',label=r"$\overline{v'v'}$")
				ax[k].plot(dsref[case]['nomean']['Subgrid']['SBG_WV'][indt,:],		Z/zi,c='g',ls='--')
				ax[k].plot(dsref[case]['nomean']['Resolved']['RES_UV'][indt,:],		Z/zi,c='b',label=r"$\overline{w'w'}$")
				ax[k].plot(dsref[case]['nomean']['Subgrid']['SBG_UV'][indt,:],		Z/zi,c='b',ls='--')
			ax[k].set_title(case,loc='right')
			ax[k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax[k].xaxis.major.formatter._useMathText = True
			ax[k].set_xlim([vmin,vmax])
			ax[k].set_ylim([0,1.2])
			ax[k].grid()
		ax[1].tick_params(axis='both',labelleft=False)
		ax[0].legend()
		ax[0].set_ylabel('z/zi')
		fig.savefig(path_save+'ref_covariance_Z'+toadd+'.png')
		
		# w'tht' and w'rv'
		# -> S1
		fig, ax = plt.subplots(1,len(liste_x),figsize = (20,5),constrained_layout=True,dpi=dpi)
		fig.suptitle(r"Covariances"+'\n'+r"$\overline{w'r_v'}$ (bottom; g kg-1 m s-1), $\overline{w'\theta_v'}$, $\overline{w'\theta'}$ (top; K m s-1)")
		for kx,x in enumerate(liste_x):
			ax[kx].plot(dsflx.FLX_RvW[0,indx[kx]],Z[0]/ABLH,c='r',label=r"$\overline{w'\theta_v'}$") # dummy for legend
			ax[kx].plot(dsflx.FLX_RvW[0,indx[kx]],Z[0]/ABLH,c='b',label=r"$\overline{w'\theta'}$") # dummy for legend
			ax2 = ax[kx].twiny()
			if B_FLXTOTAL:
				ax[kx].plot(dsflx.FLX_RvW[:,indx[kx]]*1000,Z/ABLH,c='k',label=r"$\overline{w'r_v'}$")
				ax2.plot(dsflx.FLX_THvW[:,indx[kx]],Z/ABLH,c='r',label=r"$\overline{w'\theta_v'}$")
				ax2.plot(dsflx.FLX_THW[:,indx[kx]],Z/ABLH,c='b',label=r"$\overline{w'\theta'}$")
			else:
				ax[kx].plot((dsflx.FLX_RvW-dsflx.FLX_RvW_s)[:,indx[kx]]*1000,Z/ABLH,c='k',label=r"$\overline{w'r_v'}$")
				ax[kx].plot(dsflx.FLX_RvW_s[:,indx[kx]]*1000,Z/ABLH,c='k',ls='--')
				ax2.plot((dsflx.FLX_THvW-dsflx.FLX_THvW_s)[:,indx[kx]],Z/ABLH,c='r',label=r"$\overline{w'\theta_v'}$")
				ax2.plot(dsflx.FLX_THvW_s[:,indx[kx]],Z/ABLH,c='r',ls='--')
				ax2.plot((dsflx.FLX_THW-dsflx.FLX_THW_s)[:,indx[kx]],Z/ABLH,c='b',label=r"$\overline{w'\theta'}$")
				ax2.plot(dsflx.FLX_THW_s[:,indx[kx]],Z/ABLH,c='b',ls='--')
			ax[kx].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax[kx].xaxis.major.formatter._useMathText = True
			ax[kx].set_title('x='+str(x)+'km',loc='right')
			ax[kx].set_xlim([-0.1,0.1])
			ax2.set_xlim([-0.025,0.025])
			ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax2.xaxis.major.formatter._useMathText = True
		for k,axe in enumerate(ax):
			if k>0:
				axe.tick_params(axis='both',labelleft=False)
			axe.set_ylim([0,1.2])
			axe.vlines(0,0,600,color='grey',ls='--')
			axe.grid()
		ax[0].legend()
		ax[0].set_ylabel('z/zi')	
		fig.savefig(path_save+'thermo_covariance_Z'+toadd+'.png')
		# -> REF
		fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
		fig.suptitle(r"Covariances"+'\n'+r"$\overline{w'r_v'}$ (bottom; g kg-1 m s-1), $\overline{w'\theta_v'}$, $\overline{w'\theta'}$ (top; K m s-1)")
		for k,case in enumerate(['warm','cold']):
			zi = dsref[case]['nomean']['Misc']['BL_H'][indt].values
			ax[k].plot(dsflx.FLX_RvW[0,indx[kx]],Z[0]/ABLH,c='r',label=r"$\overline{w'\theta_v'}$") # dummy for legend
			ax[k].plot(dsflx.FLX_RvW[0,indx[kx]],Z[0]/ABLH,c='b',label=r"$\overline{w'\theta'}$") # dummy for legend
			ax2 = ax[k].twiny()
			SBG_WTHTV = ( dsref[case]['nomean']['Mean']['MEAN_THV'] / dsref[case]['nomean']['Mean']['MEAN_TH']*dsref[case]['nomean']['Subgrid']['SBG_WTHL']
						+ 0.61*dsref[case]['nomean']['Mean']['MEAN_TH']*dsref[case]['nomean']['Subgrid']['SBG_WRT'] ) # not available in 000 from MNH ...
			if B_FLXTOTAL:
				ax[k].plot( (dsref[case]['nomean']['Resolved']['RES_WRT'] + 
							dsref[case]['nomean']['Subgrid']['SBG_WRT'])[indt,:]*1000,	Z/zi,c='k',label=r"$\overline{w'r_v'}$")
				ax2.plot((dsref[case]['nomean']['Resolved']['RES_WTHV'] + 
							SBG_WTHTV)[indt,:],											Z/zi,c='r',label=r"$\overline{w'\theta_v'}$")
				ax2.plot((dsref[case]['nomean']['Resolved']['RES_WTHL'] + 
							dsref[case]['nomean']['Subgrid']['SBG_WTHL'])[indt,:],	Z/zi,c='b',label=r"$\overline{w'\theta'}$")
			else:
				ax[k].plot(dsref[case]['nomean']['Resolved']['RES_WRT'][indt,:]*1000,Z/zi,	c='k',label=r"$\overline{w'r_v'}$")
				ax[k].plot(dsref[case]['nomean']['Subgrid']['SBG_WRT'][indt,:]*1000,Z/zi,	c='k',ls='--')
				ax2.plot(dsref[case]['nomean']['Resolved']['RES_WTHV'][indt,:],		Z/zi,	c='r',label=r"$\overline{w'\theta_v'}$")
				ax2.plot(SBG_WTHTV[indt,:],													Z/zi,	c='r',ls='--')
				ax2.plot(dsref[case]['nomean']['Resolved']['RES_WTHL'][indt,:],		Z/zi,	c='b',label=r"$\overline{w'\theta'}$")
				ax2.plot(dsref[case]['nomean']['Subgrid']['SBG_WTHL'][indt,:],		Z/zi,	c='b',ls='--')
			ax[k].set_title(case,loc='right')
			ax[k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax[k].xaxis.major.formatter._useMathText = True
			ax[k].set_xlim([-0.1,0.1])
			ax[k].set_ylim([0,1.2])
			ax[k].grid()
			ax2.set_xlim([-0.025,0.025])
			ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax2.xaxis.major.formatter._useMathText = True
		ax[1].tick_params(axis='both',labelleft=False)
		ax[0].legend()
		ax[0].set_ylabel('z/zi')
		fig.savefig(path_save+'ref_thermo_covariance_Z'+toadd+'.png')
		
		
	if B_HEIGHT_OF_SWITCH:
		colors = ['blue','dodgerblue','cyan','orange','red']
		ABLH = 600
		cmap = 'YlGnBu'
		# uu,vv,ww
		r1 = ( dsflx.FLX_UU_s/dsflx.FLX_UU )
		r2 = ( dsflx.FLX_VV_s/dsflx.FLX_VV )
		r3 = ( dsflx.FLX_WW_s/dsflx.FLX_WW )
		fig, ax = plt.subplots(3,1,figsize = (8,8),constrained_layout=True,dpi=dpi)
		s = ax[0].pcolormesh(X/1000,Z/ABLH,r1[:,:]*100,cmap=cmap,vmin=0,vmax=50)
		plt.colorbar(s,ax=ax[0])
		s = ax[1].pcolormesh(X/1000,Z/ABLH,r2[:,:]*100,cmap=cmap,vmin=0,vmax=50)
		plt.colorbar(s,ax=ax[1])
		s = ax[2].pcolormesh(X/1000,Z/ABLH,r3[:,:]*100,cmap=cmap,vmin=0,vmax=50)
		plt.colorbar(s,ax=ax[2])
		ax[0].set_title("u'u'")
		ax[1].set_title("v'v'")
		ax[2].set_title("w'w'")
		for axe in ax:
			axe.set_xlim([X[0]/1000,X[-1]/1000])
			axe.set_ylim([0,1.2])
		fig.suptitle('SGS/total (%)')
		fig.savefig(path_save+'sgs_contrib_variance.png')
		# uv,uw,vw
		r1 = ( dsflx.FLX_UV_s/dsflx.FLX_UV )
		r2 = ( dsflx.FLX_UW_s/dsflx.FLX_UW )
		r3 = ( dsflx.FLX_VW_s/dsflx.FLX_VW )
		fig, ax = plt.subplots(3,1,figsize = (8,8),constrained_layout=True,dpi=dpi)
		s = ax[0].pcolormesh(X/1000,Z/ABLH,r1[:,:]*100,cmap=cmap,vmin=0,vmax=50)
		plt.colorbar(s,ax=ax[0])
		s = ax[1].pcolormesh(X/1000,Z/ABLH,r2[:,:]*100,cmap=cmap,vmin=0,vmax=50)
		plt.colorbar(s,ax=ax[1])
		s = ax[2].pcolormesh(X/1000,Z/ABLH,r3[:,:]*100,cmap=cmap,vmin=0,vmax=50)
		plt.colorbar(s,ax=ax[2])
		ax[0].set_title("u'v'")
		ax[1].set_title("u'w'")
		ax[2].set_title("v'w'")
		for axe in ax:
			axe.set_xlim([X[0]/1000,X[-1]/1000])
			axe.set_ylim([0,1.2])
		fig.suptitle('SGS/total (%)')
		fig.savefig(path_save+'sgs_contrib_covariance.png')
		# wtht,wthtv,wrv
		r1 = dsflx.FLX_THW_s/dsflx.FLX_THW
		r2 = dsflx.FLX_THvW_s/dsflx.FLX_THvW
		r3 = dsflx.FLX_RvW_s/dsflx.FLX_RvW
		ID = xr.ones_like(r1)
		r1 = xr.where( np.logical_or(r1 <= 0.1,r1>= 0),r1,0 ) # removing non physical values <0 or >1
		r2 = xr.where( np.logical_or(r2 <= 0.1,r2>= 0),r2,0 )
		r3 = xr.where( np.logical_or(r3 <= 0.1,r3>= 0),r3,0 )
		fig, ax = plt.subplots(3,1,figsize = (8,8),constrained_layout=True,dpi=dpi)
		s = ax[0].pcolormesh(X/1000,Z/ABLH,r1[:,:]*100,cmap=cmap,vmin=0,vmax=50)
		plt.colorbar(s,ax=ax[0])
		s = ax[1].pcolormesh(X/1000,Z/ABLH,r2[:,:]*100,cmap=cmap,vmin=0,vmax=50)
		plt.colorbar(s,ax=ax[1])
		s = ax[2].pcolormesh(X/1000,Z/ABLH,r3[:,:]*100,cmap=cmap,vmin=0,vmax=50)
		plt.colorbar(s,ax=ax[2])
		ax[0].set_title(r"$\theta$'w'")
		ax[1].set_title(r"$\theta_v$'w'")
		ax[2].set_title(r"$r_v$'w'")
		for axe in ax:
			axe.set_xlim([X[0]/1000,X[-1]/1000])
			axe.set_ylim([0,1.2])
		fig.suptitle('SGS/total (%)')
		fig.savefig(path_save+'sgs_contrib_thermo_covariance.png')
		
def LS_RELATION_DMM(atZ,X,Z,VAR,dsmean,ds_hbudget,dsB,res,dpi,path_save):		
	"""This procedure plots correlation between a variable and the gradient of SST.
		It is used to see if large scale relation still holds at the current scale (DMM)
		
		TBD : description of inputs
	"""
	
	indz = np.argmin(np.abs(Z.values-atZ))
	print('	z=',np.round(Z[indz].values,2),'m')
	if VAR in ['Um','Vm','THm','THvm']:
		DVARDX = np.gradient(dsmean[VAR][indz,:],X,axis=0)
		name = 'gradX'+VAR
		nicename = r'$\bigtriangledown$'+VAR
	if VAR=='Wm':
		DVARDX = dsmean['Wm'][indz,:]
		name = VAR
		nicename = VAR
#	elif VAR=='Tau':
#		DVARDX = np.gradient(dsmean[VAR][:],X,axis=0)
#	elif VAR in ['u_cor','u_hadv','u_hturb','u_pres','u_vadv','u_vturb',
#			'v_cor','v_hadv','v_hturb','v_vadv','v_vturb',
#			'w_hadv','w_hturb','w_vadv','w_vturb','w_boytotale','w_cor',
#			'ET_DIFF','ET_DISS','ET_HADV','ET_HDP','ET_TP','ET_VADV','ET_VDP']:
#		DVARDX = np.gradient(ds_hbudget[VAR][indz,:],X,axis=0)
	DSSTDX = np.gradient(dsB.SST[0,1,nhalo:-nhalo],X,axis=0)
	sig_DVARDX = np.std(DVARDX)
	sig_DSSTDX = np.std(DSSTDX)
	Corr = np.zeros(DSSTDX.shape)
	Corr_coef = np.zeros(DSSTDX.shape)
	X1 = np.zeros(DSSTDX.shape)
	Y1 = np.zeros(DSSTDX.shape)
	X1 = DSSTDX
	
	for idx in range(len(DVARDX)): #
		Y1[:len(DVARDX)-idx] = DVARDX[idx:]
		Y1[len(DVARDX)-idx:] = 0.
		Corr[idx] = (X1*Y1).mean()
		Corr_coef[idx] = Corr[idx]/(sig_DVARDX*sig_DSSTDX)
	idxMAXC = np.argmax(np.abs(Corr_coef))
	
	fig, ax = plt.subplots(1,1,figsize = (3,3),constrained_layout=True,dpi=dpi)
	ax.plot(np.arange(0,len(DVARDX))*50/1000,Corr_coef,c='k')
	ax.set_xlim([0,X[-1]/1000])
	ax.hlines(0.,X[0]/1000,X[-1]/1000,colors='grey',linestyles='--')
	ax.set_xlabel('lag (km)')
	ax.set_ylabel(r'Corr('+nicename+r',$\bigtriangledown$SST)')
	ax.set_title('Cmax='+str(np.round(Corr_coef[idxMAXC],2))+', lag='+str(idxMAXC*50/1000)+'km')
	print('	Coeff correlation : '+name+' and gradSST, dx(max(Coeff))='+str(idxMAXC*50/1000)+'km where Coeff='+str(np.round(Corr_coef[idxMAXC],2)))
	fig.savefig(path_save + 'Corr_'+name+'_gSST_z'+str(np.round(Z[indz].values,1))+'.png')
	
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	DVARDX_masked = np.ma.masked_where(X < idxMAXC*res,DVARDX)
	ax.plot(DSSTDX,DVARDX_masked)
	ax.set_xlabel(r'$\bigtriangledown$SST')
	ax.set_ylabel(nicename)
	
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax.plot(X/1000 - idxMAXC*res/1000,DVARDX_masked,label='D'+VAR+'DX(dx-dxMAX)',c='k')
	ax.plot(X/1000,DSSTDX,label='DSSTDX(x)',c='b')
	ax.set_xlabel('X (km)')
	ax.set_ylabel(name)
	ax.set_title(name+' (shifted by dxmax='+str(idxMAXC*50/1000)+'km) and SST')
	ax.legend()	
	fig.savefig(path_save + 'Shifted_'+name+'_gSST_z'+str(np.round(Z[indz].values,1))+'.png')
	
def LS_RELATION_PA(atZ,X,Z,VAR,dsmean,ds_hbudget,dsB,res,dpi,path_save):		
	"""This procedure plots correlation between a variable and the laplacian of SST.
		It is used to see if large scale relation still holds at the current scale (PA)
	"""
	indz = np.argmin(np.abs(Z.values-atZ))
	print('	z=',np.round(Z[indz].values,2),'m')
	if VAR in ['Um','Vm','THm','THvm']:
		DVARDX = np.gradient(dsmean[VAR][indz,:],X,axis=0)
	if VAR=='Wm':
		DVARDX = dsmean['Wm'][indz,:]
	elif VAR=='Tau':
		DVARDX = np.gradient(dsmean[VAR][:],X,axis=0)
	elif VAR in ['u_cor','u_hadv','u_hturb','u_pres','u_vadv','u_vturb',
			'v_cor','v_hadv','v_hturb','v_vadv','v_vturb',
			'w_hadv','w_hturb','w_vadv','w_vturb','w_boytotale','w_cor',
			'ET_DIFF','ET_DISS','ET_HADV','ET_HDP','ET_TP','ET_VADV','ET_VDP']:
		DVARDX = np.gradient(ds_hbudget[VAR][indz,:],X,axis=0)
	DSSTDX = np.gradient(np.gradient(dsB.SST[0,1,nhalo:-nhalo],X,axis=0),X,axis=0)
	sig_DVARDX = np.std(DVARDX)
	sig_DSSTDX = np.std(DSSTDX)
	Corr = np.zeros(DSSTDX.shape)
	Corr_coef = np.zeros(DSSTDX.shape)
	X1 = np.zeros(DSSTDX.shape)
	Y1 = np.zeros(DSSTDX.shape)
	X1 = DSSTDX
	
	for idx in range(len(DVARDX)): #
		Y1[:len(DVARDX)-idx] = DVARDX[idx:]
		Y1[len(DVARDX)-idx:] = 0.
		Corr[idx] = (X1*Y1).mean()
		Corr_coef[idx] = Corr[idx]/(sig_DVARDX*sig_DSSTDX)
	idxMAXC = np.argmax(Corr_coef)
	fig, ax = plt.subplots(1,1,figsize = (3,3),constrained_layout=True,dpi=dpi)
	ax.plot(np.arange(0,len(DVARDX))*res/1000,Corr_coef,c='k')
	ax.set_xlim([0,X[-1]/1000])
	ax.hlines(0.,X[0]/1000,X[-1]/1000,colors='grey',linestyles='--')
	ax.set_xlabel('lag (km)')
	ax.set_ylabel(r'Corr($\bigtriangledown$U,$\bigtriangleup$SST)')
	ax.set_title('Cmax='+str(np.round(Corr_coef[idxMAXC],2))+', lag='+str(idxMAXC*res/1000)+'km')
	print('	Coeff correlation : gradX'+VAR+' and laplacianSST, dx(max(Coeff))='+str(idxMAXC*res/1000)+'km where Coeff='+str(np.round(Corr_coef[idxMAXC],2)))
	fig.savefig(path_save + 'Corr_g'+VAR+'_ggSST_z'+str(np.round(Z[indz].values,1))+'.png')		
			
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	ax.plot(X/1000,DVARDX,label='D'+VAR+'DX',c='k')
	ax2 = ax.twinx()
	ax2.plot(X/1000,DSSTDX,label=r'$\bigtriangleup$SST',c='b')
	ax.set_xlabel('X (km)')
	ax.set_ylabel('D'+VAR+'DX')
	ax2.set_ylabel(r'$\bigtriangleup$SST')
	#ax.set_title('X gradient of '+VAR+' (shifted by dxmax='+str(idxMAXC*50/1000)+'km) and SST')
	ax.legend()	
	#fig.savefig(path_save + 'Shifted_g'+VAR+'_gSST_z'+str(np.round(Z[indz].values,1))+'.png')	
			
def Explain_lag_correlation_DMM(atZ,lag,X,Z,VAR,dsmean,dsB,dpi,path_save):
	"""This procedure plots the components of the sum (the mean from the correlation coefficient)
		for the DMM large scale relation. This can leverage doubts about where are localized the 
		signal correlated. 
		The result is positive as it is normalized by the correlation with the specified lag
		
				Corr(tau) = < X(t-tau)*Y(t) >_t		
					
					  = 1/Nt * Sum(from t0 to t1) of X(t-tau)*Y(t)
		
		so the components plotteds are:
		
			X(t-tau)*Y(t) / <X(t-tau)*Y(t)>_t at the specified tau
			
		Note : this is only for DMM relation (grad of SST)	
	"""			
	idlag = np.argmin(np.abs(X.values-lag*1000))
	indz = np.argmin(np.abs(Z.values-atZ))
	DVARDX = np.gradient(dsmean[VAR][indz,:],X,axis=0)
	DSSTDX = np.gradient(dsB.SST[0,1,nhalo:-nhalo],X,axis=0)
	sig_DVARDX = np.std(DVARDX)
	sig_DSSTDX = np.std(DSSTDX)
	Corr = np.zeros(DSSTDX.shape)
	Corr_coef = np.zeros(DSSTDX.shape)
	Nx = len(DVARDX)
	X1 = DSSTDX
	Y1 = np.zeros(DSSTDX.shape)
	for idx in range(len(DVARDX)): #
		Y1[:len(DVARDX)-idx] = DVARDX[idx:]
		Y1[len(DVARDX)-idx:] = 0.
		Corr[idx] = (X1*Y1).mean()
		Corr_coef[idx] = Corr[idx]/(sig_DVARDX*sig_DSSTDX)
	C_at_lag = Corr_coef[idlag]
	Y1[:len(DVARDX)-idlag] = DVARDX[idlag:]
	Y1[len(DVARDX)-idlag:] = 0. 
	prod = X1*Y1/(Nx*sig_DVARDX*sig_DSSTDX)
	Norm_prod = prod/C_at_lag
	print('	sum of terms = '+str(np.sum(Norm_prod))) # this should be 1
	fig, ax = plt.subplots(1,1,figsize = (3,3),constrained_layout=True,dpi=dpi)
	ax.plot(X/1000,Norm_prod*100,c='k')
	ax.set_title('lag='+str(lag)+'km, Corr(lag)='+str(np.round(C_at_lag,2)))
	ax.set_ylabel(r'$\frac{ \bigtriangledown U(x-lag).\bigtriangledown SST(x)}{<\bigtriangledown U(x-lag).\bigtriangledown SST(x)>_x}$ (%)')
	ax.set_xlabel('X (km)')
	fig.savefig(path_save+'Corr_at'+str(lag)+'.png')	
						
def HODOGRAPH_t(dsO_t,cmap,atX,X,atZ,Z,step,dpi,path_save):
	"""This procedure is plotting a graph with U as X and V as Y, at differents instants.
	"""
	U = dsO_t.UT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo].mean('nj_u')
	V = dsO_t.VT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo].mean('nj_v')
	indz =  []
	Ntime = dsO_t.time.shape[0]
	N = Ntime//step
	for z in atZ:
		indz.append(np.argmin(np.abs(Z.values-z)))
	indx = np.argmin(np.abs(X.values-atX))
	Time = np.arange(0,Ntime,step)*5*60
	
#	 inertial oscillation solution is V = V0*exp(-ift)
#	 	U0,V0 are computed from a few points to smooth
	
#	U0 = U[:3,indz,indx].mean().values
#	V0 = V[:3,indz,indx].mean().values
#	print(Time,U0,V0)
#	IN_X = (U0*np.cos(f*Time) + V0*np.sin(f*Time))
#	IN_Y = -(V0*np.cos(f*Time) - U0*np.sin(f*Time))
	
	for kz,indz in enumerate(indz):
		fig, ax = plt.subplots(1,1,figsize = (5,3),constrained_layout=True,dpi=dpi)	
		ax.grid()
		s = ax.scatter(U[::step,indz,indx],V[::step,indz,indx],c=np.arange(0,Ntime,step)*5/60,cmap=cmap) #,label=labels
		plt.colorbar(s,ax=ax,label='h')
		#ax.scatter(IN_X,IN_Y,c=Time,cmap='bwr')
		ax.set_xlabel('U(t) (m/s)')
		ax.set_ylabel('V(t) (m/s)')
		ax.set_aspect('equal')
		ax.set_title('z='+str(np.round(Z[indz].values,1))+'m',loc='right')
		fig.savefig(path_save+'Ut_and_Vt_at_z'+str(atZ[kz])+'m.png')		
		
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	Xarray = np.arange(0,Ntime,1)*5/60
	ax.plot(0,U[0,indz,indx],c='b',label='V') # dummy
	ax.plot(Xarray,U[:,indz,indx],c='k',label='U')
	ax2 = ax.twinx()
	ax2.plot(Xarray,V[:,indz,indx],c='b',label='V')	
	ax.set_title('Time evolution of U and V (z='+str(np.round(Z[indz].values,1))+'m)')
	ax.set_xlabel('Time (h)')
	ax.set_xlim([0,Xarray[-1]])
	ax.set_ylabel('U m/s')
	ax2.set_ylabel('V m/s')
	ax.legend()
	
def Surface_TA_wtht_FLX(X,dsmean,dsB,dsflx,path_save,dpi):
	"""This procedure is plotting the surface atm temperature and flux of the mean field (2h-6h)"""

	Ta = dsmean.THTm[0,:] * ( dsmean.Pm[0,:]/P00)**(Rd/Cpd) # temperature at first level
	SST = dsB.SST[0,1,1:-1]
	FLX = dsflx.FLX_THW[0,:]
	fig, ax = plt.subplots(2,1,figsize = (5,10),constrained_layout=True,dpi=100)
	ax[0].plot(X/1000,Ta,c='k',label='Ta')
	ax[0].plot(X/1000,SST,c='b',label='SST')
	ax[0].set_ylabel('Temperature (K)')
	ax[0].legend()	
	ax[1].plot(X[0]/1000,(Ta-SST)[0],c='k',label=r"$\overline{w'\theta'}$",ls='--')
	ax[1].plot(X/1000,Ta-SST,c='k',label='Ta-SST (K)')
	ax[1].set_xlabel('X (km)')
	ax[1].set_ylabel(r'$\Delta$Temperature')
	ax2 = ax[1].twinx()
	ax[1].set_ylim([-1,1])
	ax2.set_ylim([-0.008,0.008])
	ax2.plot(X/1000,FLX,c='k',ls='--')
	ax2.set_ylabel(r"$\overline{w'\theta'}$")
	ax[1].legend()
	#fig.savefig(path_save+)
		
def TIME_EVOLUTION_TA_wtht(dsO,dsB,X,nameds,nhalo,path_save,dpi):
	"""Plot the time evolution of surface air absolute temperature and surface subgrid flux wtht"""
	cmap = mpl.colormaps.get_cmap('rainbow')
	step = 12 # 1 step is 5min
	fig, ax = plt.subplots(2,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	N = dsO.time.shape[0]
	SST = dsB.SST[0,1,1:-1]
	Ta = dsO.THT[:,1,nhalo:-nhalo,nhalo:-nhalo].mean(axis=1) * ( dsO.PABST[:,1,nhalo:-nhalo,nhalo:-nhalo].mean(axis=1)/P00)**(Rd/Cpd)
	FLX = dsO.THW_FLX[:,1,nhalo:-nhalo,nhalo:-nhalo].mean(axis=1)
	for t in range(0,N-step,step):
		ax[0].plot(X/1000,Ta[t:t+step,:].mean(axis=0),c=cmap(t/N))
		ax[1].plot(X/1000,FLX[t:t+step,:].mean(axis=0),c=cmap(t/N))
	ax[0].plot(X/1000,SST,c='k',ls='--')
	ax[0].set_xlim([0,38.4])
	ax[1].set_xlim([0,38.4])
	ax[1].set_xlabel('X (km)')
	ax[0].set_ylabel('Temperature')
	ax[0].set_title('Atm temperature on period ['+nameds+'], every '+str(step//12)+'h')
	ax[1].set_ylabel(r"$\overline{w'\theta'}$")
	ax[1].set_title(r"$\overline{w'\theta'}$ on period ["+nameds+'], every '+str(step//12)+'h')
	ax[1].set_ylim([-0.015,0.005])
	fig.savefig(path_save+'TimeEvolution_Ta_WTHflx_'+nameds+'.png')
	
def check_pressure_consistency(X,Z,dsCS1,dsO,dsB,dsmean,dsINI,atX,atX2,dpi):
	"""
		GOAL : Compute the pressure terms in the W budgets (pressure fluctuations from reference state and buoyancy) with differents reference states
		
			-1/rho*dP/dz - g --> ... --> 1/rho_ref*dP'/dz + g(thtv/thtv_ref - 1)
	
		INPUTS:
			
	
		Une différence d'environnement peut-elle changer l'interprétation du terme de flottabilité de l'équation de W ?		
		
		Conclusion : 
			* Il faudrait comparer la somme des termes de pression et de flottabilité dans le cas où la référence est l'état initial (mnh) et le cas où la référence est l'état moyen.
				Sauf qu'après avoir récupéré l'état hydrostatique, il faudrait appliquer une correction anélastique pour adapter le champ de vent (et donc le champ de pression) à
				cet état de référence, ce que je ne peut faire ici sans un gros investissement en temps..
			* I have tried to integrated myself the reference pressure but this is not simple.
			
	"""
	indx = nearest(X.values/1000,atX)
	indx2 = nearest(X.values/1000,atX2)	
	THTV = dsCS1.THTV 									# =f(dsO.THT,dsO.RVT)
	global_objects = dsCS1.global_objects				# all objects detected by
	is_up = global_objects.where(global_objects==1) 	# mask for updrafts
	THTVm = dsCS1.THTVm # 4D
	THTVref = Compute_THTV(dsINI)[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] # ref=ini state
	THTV_up = THTV.where(is_up==1) 						# thtv only in updrafts
	
	Flott_up2 = g*(THTV_up/THTVref - 1)					# buoyancy term for ref=ini and ref=refZ
	Flott_up2m = Flott_up2.mean(dim=['time','nj'])
	# instantaneous fields
	P = dsO.PABST[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	Pi = Exner(P)
	Tv = Pi*THTV
	RHO = P/(Rd*Tv) # = (Pi**(Cvd/Rd)*P00/(Rd*THTV))
	# ref state is initial state (only z):
	#	form is : classic with ref state
	# -> not ok, idkw
	Pref = dsINI.PABST[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	Pi_ref = Exner(Pref)
	THTVref = Compute_THTV(dsINI)[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	RVref = dsINI.RVT[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	RHOREF = (Pi_ref**(Cvd/Rd)*P00/(Rd*THTVref))
	dpdz = (P-Pref).differentiate('level')
	dpdzm = (dpdz/RHOREF).where(is_up==1).mean(dim=['time','nj'])
	# MNH form of pressure term
	#	form is : MNH (durran)
	dPidz = (Pi-Pi_ref).differentiate('level')
	RHODREF = RHOREF/(1+RVref)
	rhoeff = RHODREF*THTVref*(1+RVref)
	dPidzm = ( Cpd*THTV*dPidz ).where(is_up==1).mean(dim=['time','nj'])
	# ref state is mean state (only z):
	#	form is : classic with ref state
	RHOREF0 = RHO.mean(dim=['time','nj','ni'])
	dpdz_ref = - RHOREF0*g
	dpdz0 = P.differentiate('level')
	press = (-(dpdz0 - dpdz_ref)/RHOREF0).where(is_up==1)
	pressm = press.mean(dim=['nj','time'])
	THTV0 = dsmean.THTvm.mean(dim=['ni'])
	buoy = g*(THTV_up/THTV0 -1) 
	buoym = buoy.mean(dim=['nj','time'])	
	# ref state is 'ref' from mnh
	#  thtvref is the same as THTVref above
	# -> not ok, need anelastic constraint
	RHOREF1 = dsB.RHOREFZ.interp({'level_w':Z})
	RHOEFF1 = RHOREF1*THTVref
	pressMNH = - Cpd*( Pi.differentiate('level') + g/(Cpd*THTVref) )
	pressMNHm = pressMNH.where(is_up==1).mean(dim=['time','nj'])
	buoyMNHm = 	Flott_up2m
	# ref state is mean state (but x,y)
	RHOREF2 = RHO.mean(	dim=['time','nj'])
	dpdz_ref2 = - RHOREF2*g
	press2 = (-(dpdz0 - dpdz_ref2)/RHOREF2).where(is_up==1)
	print('RHOREF2.shape',RHOREF2.shape)
	print('press2.shape',press2.shape)
	press2m = press2.mean(dim=['nj','time'])
	THTV2 = dsmean.THTvm
	buoy2 = g*(THTV_up/THTV2 -1) 
	buoy2m = buoy2.mean(dim=['nj','time'])
	# PLOT	
	# sum
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot( ( press2m + buoy2m).isel(ni=indx),Z/600,c='pink',label="-dp'/dz + buoy, ref=mean(x,z) (classic+ref)")
	ax.plot( ( pressMNHm + buoyMNHm).isel(ni=indx),Z/600,c='b',label=" -dPi'/dz + buoy, ref=refZ (MNH)") # form is not ok
	ax.plot( ( pressm + buoym).isel(ni=indx),Z/600,c='r',label="-dp'/dz + buoy, ref=mean(z) (classic+ref)",ls='--')
	ax.plot( (-dPidzm+Flott_up2m).isel(ni=indx),Z/600,c='chartreuse',label="-dPi'/dz + buoy, ref=ini (MNH)",ls='--')
	ax.plot( -( (P.differentiate('level')/RHO).where(is_up==1).mean(dim=['time','nj']).isel(ni=indx) + g ),Z/600,c='k',label='-(dP/dz + g) (classic)',ls='--')
	ax.plot( (-dpdzm+Flott_up2m).isel(ni=indx),Z/600,c='g',label="-dp'/dz + buoy, ref=ini (classic+ref)") # not ok idkw
	ax.set_ylim([0,1.2])
	ax.set_ylabel('z/zi')
	ax.legend()
	ax.set_xlabel('press + buoya')
	ax.set_title('Contribution to W budget inside updraft')
	# buoyancy only
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot( buoym.isel(ni=indx),Z/600,c='r',label="buoy, ref=mean(z) (classic+ref)")
	ax.plot( buoy2m.isel(ni=indx),Z/600,c='pink',label="buoy, ref=mean(x,z) (classic+ref)")
	ax.plot( Flott_up2m.isel(ni=indx),Z/600,c='g',label="buoy, ref=ini (classic+ref)")
	ax.set_ylim([0,1.2])
	ax.set_ylabel('z/zi')
	ax.legend()
	ax.set_xlabel('buoya in updrafts')
	# thtv ref state
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot( THTV0, Z/600, c='r',label='ref=mean(z)')
	ax.plot( THTV2[:,indx], Z/600, c='pink',label='ref=mean(x,z)')
	ax.plot( THTVref[:,0,0],Z/600,c='g',label='ref=ini')
	ax.plot( THTV_up.mean(dim=['nj','time']).isel(ni=indx), Z/600,c='orange',label='thtv in updrafts object')
	ax.set_ylabel('z/zi')
	ax.set_xlabel('thtv reference')
	ax.set_ylim([0,1.2])
	ax.set_xlim([297.2,298])
	ax.legend()

def ABLH_convergence(Z,ds1,ds2,ndt,nhalo,path_outpng,dpi):
	"""
	This procedure plots the evolution of the domain mean height of the boundary layer.	
	
	INPUTS:
		- Z	: Z dimension
		- ds1 	: has files output every 5 min (from t+0h o t+2h)
		- ds2 	: has files output every 30s (from t+2h to t+3h)
		- ndt 	: for the moving average in time for ds2
		- nhalo : mnh halo
		- path_outpng : where to save figures
		- dpi 	: for figures
	"""
	data = {'0':ds1,'1':ds2}
	N1 = len(ds1.time)
	N2 = len(ds2.time)		
	zi = {}
	minzi = {}
	maxzi = {}
	time = {}
	for case in data.keys():
		ds = data[case]
		time[case] = ds.time.dt.hour.values + ds.time.dt.minute.values/60 + ds.time.dt.second.values/3600 # in sec -> hours
		THT,RVT = ds.THT,ds.RVT
		THTV = Compute_THTV(THT,RVT)
		GTHTV = THTV.differentiate('level')[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		GTHTV_tmean = xr.zeros_like(GTHTV)
		for t in range(len(time[case])):
			if t<ndt:
				tmin = t
			else:
				tmin = ndt
			if t>len(time[case])-1-ndt:
				tmax = -1
			else:
				tmax = ndt
			GTHTV_tmean[t,:,:,:] = GTHTV[t-tmin:t+tmax,:,:,:].mean(dim='time')
					
		
		zi[case] = Z[GTHTV.mean(dim=['ni','nj']).sel(level=slice(0,1000)).argmax('level').values] # only look for under 1km to avoid catching something above
		indzi2D = GTHTV_tmean.mean(dim=['nj']).sel(level=slice(0,1000)).argmax('level') # only look for under 1km to avoid catching something above
		maxzi[case] = Z[indzi2D.max(dim=['ni'])]
		minzi[case] = Z[indzi2D.min(dim=['ni'])]
		
	
	TIME,ABLH = np.zeros( N1 + N2 ),np.zeros( N1 + N2 )
	MIN,MAX = np.zeros( N1 + N2 ),np.zeros( N1 + N2 )
	TIME[:N1] = time['0']
	TIME[N1:N1+N2] = time['1']
	ABLH[:N1] = zi['0']
	ABLH[N1:N1+N2] = zi['1']
	MAX[:N1] = maxzi['0']
	MAX[N1:N1+N2] = maxzi['1']
	MIN[:N1] = minzi['0']
	MIN[N1:N1+N2] = minzi['1']
	
	# linear fit
	indt = nearest(TIME,2)
	coefs = poly.polyfit(TIME[indt:], ABLH[indt:], 1)
	ffit = poly.polyval(TIME[indt:], coefs)
	
	SS_res = np.sum( (ABLH[indt:] - ffit)**2 )
	SS_tot = np.sum( (ABLH[indt:]-ABLH[indt:].mean())**2 )
	R2 = 1 - SS_res/SS_tot # in this case, = Pearson coeff squared because i use least square fit !
	print('R squared = ',R2) 
	
	fig, ax = plt.subplots(1,1,figsize = (8,5),constrained_layout=True,dpi=dpi)
	ax.plot(TIME,ABLH,c='grey',label='S1')
	ax.plot(TIME[N1:N1+N2],MAX[N1:N1+N2],c='r',label='max')
	ax.plot(TIME[N1:N1+N2],MIN[N1:N1+N2],c='b',label='min')
	ax.set_xlabel('time (h)')
	ax.set_ylabel('<ABLH>xy(t)')
	ax.plot(TIME[indt:],ffit,c='k',ls='-',label=r'fit R$^2$='+str(np.round(R2,3)))
	ax.grid()
	ax.legend()
	fig.savefig(path_outpng+'Time_evolution_ABLH_S1.png')
	
def surface_fluxes(dsB,nhalo,dsmean,dataSST,dpi):
	"""This procedure plots the surface fluxes from S1.
	it also decomposes terms according to the general formulation
	for a given variable c
		flux(W/m2) = Constant * U * delta(c_air - c_surface)
	"""
	X = dsB.ni[nhalo:-nhalo]
	# stability coeff	
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
	ax.plot(X/1000,dsB.CD[:,nhalo:-nhalo,nhalo:-nhalo].mean(dim=['time','nj']),label='CD')
	ax.plot(X/1000,dsB.CE[:,nhalo:-nhalo,nhalo:-nhalo].mean(dim=['time','nj']),label='CE')
	ax.plot(X/1000,dsB.CH[:,nhalo:-nhalo,nhalo:-nhalo].mean(dim=['time','nj']),label='CH')
	ax.set_xlabel('X (km)')
	ax.set_ylabel('CD,CE,CH')
	ax.legend()
	ax.grid()
	ax.set_xlim([0,38.4])
	# Computing Delta rv,q,tht
	P = dsmean.Pm[0,:] # in Pa
	rv = dsmean.RVTm[0,:]
	q = 1/(1+1/rv)
	tht = dsmean.THTm[0,:]
	T = Theta_to_T(tht,P)
	thts = T_to_Theta(dataSST,dsmean.Pm[0,:])
	rv_s = compute_rv_s(T,P)
	if False:
		# delta tht
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
		ax.plot(X/1000,-(tht-thts))
		ax.set_xlabel('X (km)')
		ax.set_ylabel(r'-($\theta-\theta_s$) (K)')
		ax.grid()
		ax.set_xlim([0,38.4])
	if False:
		# delta rv
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
		ax.plot(X/1000,-(rv-rv_s)*1000)
		ax.set_xlabel('X (km)')
		ax.set_ylabel(r'-($r_v-r_{v,sat}$) (g/kg)')
		ax.grid()
		ax.set_xlim([0,38.4])
	if False:
		# rv
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
		ax.plot(X/1000,rv*1000,label='rv(z=1m)')
		ax.plot(X/1000,rv_s*1000-6,label='rv_sat-6g/kg')
		ax.set_xlabel('X (km)')
		ax.set_ylabel(r'mixing ratio(g/kg)')
		ax.grid()
		ax.legend()
		ax.set_xlim([0,38.4])
	if True:
		# flux en W/m2
		CHs = savitzky_golay(dsB.CH[:,nhalo:-nhalo,nhalo:-nhalo].mean(dim=['time','nj']).values,21,2)
		CEs = savitzky_golay(dsB.CE[:,nhalo:-nhalo,nhalo:-nhalo].mean(dim=['time','nj']).values,21,2)
		CDs = savitzky_golay(dsB.CD[:,nhalo:-nhalo,nhalo:-nhalo].mean(dim=['time','nj']).values,21,2)
		LE = -1.2*Lv*CEs*(rv-rv_s)
		H = -1.2*Cpd*CHs*(tht-thts)
		Tau = np.abs(-1.2*CDs*dsmean.Um[0,:]**2)
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
		ax2 = ax.twinx()
		ax.plot(X/1000,LE,label='LE',c='b')
		ax.plot(X[0]/1000,H[0],label='Tau',c='g')
		ax.plot(X/1000,H,label='H',c='r')
		ax2.plot(X/1000,Tau,label='Tau',c='g')
		ax.set_xlabel('X (km)')
		ax.set_ylabel('W/m2')
		ax.grid()
		ax.legend()
		ax.set_xlim([0,38.4])
	if False:
		# U wind speed at 1 level model
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
		ax.plot(X/1000,dsmean.Um[0,:])
		ax.set_xlabel('X (km)')
		ax.set_ylabel('U (m/s)')
		ax.grid()
		ax.set_xlim([0,38.4])
	if False:
		# U² 
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
		ax.plot(X/1000,dsmean.Um[0,:]**2)
		ax.set_xlabel('X (km)')
		ax.set_ylabel(r'U$^2$ (m/s)')
		ax.grid()
		ax.set_xlim([0,38.4])

def K_profiles(X,Z,dsmean,dsflx,dsref,dataSST,crit_value,L_x,path_save,dpi):
	"""
	This procedure plot the K closure, computed with actual flux.
	
	K = - uw/ (dU/dz)
	
	INPUTS:
		- TBD
	"""
	indx = [nearest(X.values,atX) for atX in L_x]
			
	# S1
	Um = dsmean.Um
	DZU = Um.differentiate('level')
	uw = dsflx.FLX_UW
	K = -uw/DZU
	# ref
	indt = -1
	THT_c = dsref['cold']['nomean']['Mean']['MEAN_TH'][indt,:]
	THT_w = dsref['warm']['nomean']['Mean']['MEAN_TH'][indt,:]
	RV_c = dsref['cold']['nomean']['Mean']['MEAN_RV'][indt,:]
	RV_w = dsref['warm']['nomean']['Mean']['MEAN_RV'][indt,:]
	uw_warm = dsref['warm']['nomean']['Resolved']['RES_WU'][indt,:] + dsref['warm']['nomean']['Subgrid']['SBG_WU'][indt,:]
	uw_cold = dsref['cold']['nomean']['Resolved']['RES_WU'][indt,:] + dsref['cold']['nomean']['Subgrid']['SBG_WU'][indt,:]
	DZU_w = dsref['warm']['nomean']['Mean']['MEAN_U'][indt,:].differentiate('level_les')
	DZU_c = dsref['cold']['nomean']['Mean']['MEAN_U'][indt,:].differentiate('level_les')
	Kwarm = -uw_warm/DZU_w
	Kcold = -uw_cold/DZU_c
	zi_w = Z[Compute_THTV(THT_w,RV_w).differentiate('level_les').argmax('level_les').values].values
	zi_c = Z[Compute_THTV(THT_c,RV_c).differentiate('level_les').argmax('level_les').values].values
	
	cmap_warm ='Reds'
	cmap_cold ='winter' 
	colorsX = DISCRETIZED_2CMAP_2(cmap_cold,cmap_warm,L_x,dataSST,crit_value,X.values)
			
				
	# K plot		
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	for k in range(len(L_x)):
		ax.plot(K[:,indx[k]],Z/ABLH_S1,c=colorsX[k],label='X='+str(L_x[k])+'km')
	#ax.plot(Kwarm,Z/zi_w,c='r',ls='--')
	#ax.plot(Kcold,Z/zi_c,c='b',ls='--')
	ax.grid()
	ax.set_ylim([0,1.2])
	ax.legend()	
	ax.set_xlim([-10000,10000])
	ax.set_ylabel(r'Z/z$_i$')
	ax.set_xlabel(r'K (m$^2$.s$^{-1}$)')
	#ax.set_title('imax='+str(indlag)+' CorrMax='+str(np.round(rho[indlag],2)))
	#fig.savefig(Savename_corr+'.png')
	
	# DZU plot
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	for k in range(len(L_x)):
		ax.plot(DZU[:,indx[k]],Z/ABLH_S1,c=colorsX[k],label='X='+str(L_x[k])+'km')
	ax.plot(DZU_w,Z/zi_w,c='r',ls='--')
	ax.plot(DZU_c,Z/zi_c,c='b',ls='--')
	ax.grid()
	ax.set_ylim([0,1.2])
	ax.legend()	
	ax.set_ylabel(r'Z/z$_i$')
	ax.set_xlabel(r'dU/dz (s$^{-1}$)')
	
	# uw plot
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	for k in range(len(L_x)):
		ax.plot(uw[:,indx[k]],Z/ABLH_S1,c=colorsX[k],label='X='+str(L_x[k])+'km')
	ax.plot(uw_warm,Z/zi_w,c='r',ls='--')
	ax.plot(uw_cold,Z/zi_c,c='b',ls='--')
	ax.grid()
	ax.set_ylim([0,1.2])
	ax.legend()	
	ax.set_ylabel(r'Z/z$_i$')
	ax.set_xlabel(r'<uw> (m$^2$.s$^{-2}$)')








	
