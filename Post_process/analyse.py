"""
##########################################
 * Goal : Exploitation of canal simulation across front
 	
 * Execution : python3 analyse.py
 
 Hugo Jacquet August 2024
##########################################
"""

# ---------------------
# importing packages
# ---------------------
import warnings
warnings.filterwarnings('ignore')
import os
import pathlib
import numpy as np
import scipy as sp
import xarray as xr
from scipy import signal
from scipy.ndimage import uniform_filter1d,gaussian_filter
import pandas as pd
import time

# plot related
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors as c
from matplotlib.ticker import Locator
# for 3D plots
from skimage.measure import marching_cubes
from pyevtk.hl import *
# custom functions
from module_building_files 	import *
from module_budget 			import *	
from module_phy 			import *			
from module_CS 				import *
from module_CS_func 		import *
from module_cst 			import *
from module_tools 			import *

from dask.distributed import Client,LocalCluster


# ===================================================
# Figure selector
# ---------
BUILD = True
NUM_FIG = -1 # number from 1 to 11, or 'A1' or 'S1' or -1 for all figures
#pathWORKDIR = '/mnt/60df13e6-86a2-4cac-b207-817433cddfe5/WORKDIR2/56MNH/simu_paper_clean_HighRes/'
pathWORKDIR = '/home/jacqhugo/files_for_zenodo_CanalSim/DATA/'
abs_path = '/home/jacqhugo/scripts/simu_canal_across_paper/'
# ---------
# ===================================================

if NUM_FIG==-1:
	print('I will print every figures from paper')
else:
	print('I will print figure number '+str(NUM_FIG))
os.system('ln -sf '+pathWORKDIR+'S1/ .')
os.system('ln -sf '+pathWORKDIR+'RefC/ .')
os.system('ln -sf '+pathWORKDIR+'RefW/ .')

if __name__ == "__main__":  # This avoids infinite subprocess creation
	
	if False:
		# sometimes dask cluster can cause problems "memoryview is too large"
		# (writing a big netcdf file for eg, hbudget_file)
		cluster = LocalCluster(n_workers=8)
		client = Client(cluster)
		print("Dashboard at :",client.dashboard_link)
	else:
		client = None
	
	mpl.rcParams['xtick.major.size'] = 5
	mpl.rcParams['xtick.major.width'] = 1
	mpl.rcParams['xtick.minor.size'] = 5
	mpl.rcParams['xtick.minor.width'] = 1
	mpl.rcParams['ytick.major.size'] = 5
	mpl.rcParams['ytick.major.width'] = 1
	mpl.rcParams['ytick.minor.size'] = 5
	mpl.rcParams['ytick.minor.width'] = 1
	
	
	
	
	# ===================================================
	# data opening, file structure
	# ===================================================
	# INPUT----
	CHOIX = 'S1'
	dpi = 200
	res = 50 # dx=dy, in meters
	nhalo = 1
	# ---------
	print('Case is : '+CHOIX)
	print(' * Reading files...')
	
	
	# dask related
	ChunksT = 10
	ChunksZ = 18
	ChunksX = 770
	ChunksY = 21
	chunksOUT = {'time':-1,
				'level':81,
				'level_w':81,
				'nj':21,
				'nj_u':21,
				'nj_v':21,
				'ni':770,
				'ni_u':770,
				'ni_v':770}		
	chunksNOHALO = {'time':10,
				'level':16,
				'level_w':16,
				'nj':20,
				'nj_u':20,
				'nj_v':20,
				'ni':768,
				'ni_u':768,
				'ni_v':768}
	
	path_BACKUP = {'S1':pathWORKDIR+'S1/'}
	path_OUT = {'S1':pathWORKDIR+'S1/FICHIERS_OUT/'}
	path_BUDGET = {'S1':''}
	L_BUDGET = {'S1':''}
	L_OUT = {'S1':'CAS06.1.002.OUT*'}
	L_BACKUP ={'S1':[path_BACKUP[CHOIX]+'CAS06.1.002.002.nc'],
				'cold':abs_path+'RefC/CAS09.1.001.003.nc',
				'warm':abs_path+'RefW/CAS10.1.001.003.nc'}
	L_var_budget = ['UU','VV','WW','RC','RhodJ','RR','RV','TH','TK']
	group = 'Budgets/'
	path_INI = {'S1':abs_path+'S1/INIT_CANAL_SST.nc'}

	# This is for the reference sims	
	path_ref = {'cold':abs_path+'RefC/CAS09.1.001.000.nc',	# this has SST=296.55K=Ta_ini
		'warm':abs_path+'RefW/CAS10.1.001.000.nc'}			# this has SST=298.05K=Ta_ini+1.5K
	nameGroups = { 'nameKE' : 'LES_budgets/BU_KE/Cartesian/',
			'nameMean' : 'LES_budgets/Mean/Cartesian/',
			'nameResolved' : 'LES_budgets/Resolved/Cartesian/',
			'nameSGS' : 'LES_budgets/Subgrid/Cartesian/',
			'nameSurface' : 'LES_budgets/Surface/Cartesian/',
			'nameMisc' : 'LES_budgets/Miscellaneous/Cartesian/',
			'meaned' : 'Time_averaged/Not_normalized/cart/',
			'notmeaned' : 'Not_time_averaged/Not_normalized/cart/'}
	# Opening files
	dsB,dsO,dsBU,ds000,dsINI,dsref = Opening_files(chunksOUT,abs_path,path_BACKUP,path_OUT,path_BUDGET,L_BUDGET,L_OUT,L_BACKUP,L_var_budget,group,path_INI,CHOIX,
					path_ref,nameGroups)
	print('	done!')
	
	# naming save path
	Name_out = CHOIX + '_'
	path_outpng = './PNG_'+CHOIX+'/' # './'
	path_budget_hand = path_outpng+'BUDGET_HAND/'
	path_temp = './TEMP/'
	path_TURB = path_outpng+'TURB_STRUCTURE/'
	os.system('mkdir -p '+path_outpng)
	os.system('mkdir -p DATA_turb')
	os.system('mkdir -p '+path_budget_hand)
	os.system('mkdir -p '+path_TURB)
	os.system('mkdir -p '+path_outpng+'/object_videos')

	# global variables (+ see module_cst.py)
	crit_value = 297.3 # K, = (296.55+296.55+1.5)/2
	dataSST = dsB.SST[1,nhalo:-nhalo].values
	N_timeB = dsB.time.shape[0]
	N_timeO = dsO.time.shape[0]
	X,X_u,X_v = dsB.ni[nhalo:-nhalo],dsB.ni_u[nhalo:-nhalo],dsB.ni_v[nhalo:-nhalo]
	Y,Y_u,Y_v = dsB.nj[nhalo:-nhalo],dsB.nj_u[nhalo:-nhalo],dsB.nj_v[nhalo:-nhalo]
	Z,Z_w = dsB.level[nhalo:-nhalo],dsB.level_w[nhalo:-nhalo]
	Time = dsO.time

	# ===================================================
	# description of how the statistics are computed
	#	and files building
	# ===================================================
	"""
	The operator "average" used in the computation of turbulent statistics can be decomposed in 2 :
	-> a spatial mean in Y direction (cyclic condition)
	-> a spatial running average in X direction, size is 'window'
	-> a temporal mean during between the file Tstart and the file Tstop
	"""
	Tstart = 1 # number of the first file 
	Tstop = -1 # number of the last file
	window = 20 # for X running average, number of points
	 
	if BUILD:
		EXCLUDE = '06WdecayHalf' 
		# building the mean files
		name_mean = 'DATA_turb/'+CHOIX+'_mean_fields.nc'
		if not pathlib.Path(name_mean).is_file(): # Checking if the work has already be done
			print(' * Building mean file')
			Build_mean_file(path_OUT[CHOIX],dsO,dsB,Tstart,Tstop,name_mean,nhalo,window)
		dsmean = xr.open_dataset(name_mean,chunks=chunksNOHALO)
		# building the flux files
		name_flx = 'DATA_turb/'+CHOIX+'_flx_fields.nc'
		if not pathlib.Path(name_flx).is_file() and not CHOIX==EXCLUDE:
			print(' * Building flux file')
			Build_flx_file(path_OUT[CHOIX],dsO,Tstart,Tstop,dsmean,name_flx,nhalo,window)
		dsflx = xr.open_dataset(name_flx,chunks=chunksNOHALO)
		# building the budgets (U,V,THT,TKE) components files
		name_hbudget = 'DATA_turb/'+CHOIX+'_budget_fields.nc'
		if not pathlib.Path(name_hbudget).is_file() and not CHOIX==EXCLUDE: 
			print(' * Building budget file')
			Build_budget_file(path_OUT[CHOIX],path_INI[CHOIX],dsB,dsO,dsmean,Tstart,Tstop,name_hbudget,nhalo,window)
		ds_hbudget = xr.open_dataset(name_hbudget,chunks=chunksNOHALO)
		# building quadrant analysis file for sim with front
		# 	following "On the Nature of the Transition Between Roll and
		# 	Cellular Organization in the Convective Boundary Layer" Salesky 2016
		name_quadrant = 'DATA_turb/'+CHOIX+'_quadrants_fields.nc'
		if not pathlib.Path(name_quadrant).is_file() and not CHOIX==EXCLUDE: 
			print(' * Building quadrant analysis file')
			Build_Quadrants_terms(path_OUT[CHOIX],dsO,dsmean,name_quadrant,nhalo)
		ds_quadrant = xr.open_dataset(name_quadrant,chunks=chunksNOHALO)
		#	building quadrant analysis file for ref sim 
		name_quadrant_cold = 'DATA_turb/'+CHOIX+'_cold_quadrants_fields.nc'
		if not pathlib.Path(name_quadrant_cold).is_file() and not CHOIX==EXCLUDE:
			print(' * Building quadrant analysis file (cold)')
			chemin_cold = L_BACKUP['cold'] # last instant, t=+3h
			file_for_quad_cold = xr.open_dataset(chemin_cold)
			Build_Quadrants_terms(chemin_cold,file_for_quad_cold,dsref['cold'],name_quadrant_cold,nhalo)
		ds_quadrant_cold = xr.open_dataset(name_quadrant_cold,chunks=chunksNOHALO)
		name_quadrant_warm = 'DATA_turb/'+CHOIX+'_warm_quadrants_fields.nc'
		if not pathlib.Path(name_quadrant_warm).is_file() and not CHOIX==EXCLUDE:
			print(' * Building quadrant analysis file (warm)')
			chemin_warm = L_BACKUP['warm'] # last instant, t=+3h
			file_for_quad_warm = xr.open_dataset(chemin_warm)
			Build_Quadrants_terms(chemin_warm,file_for_quad_warm,dsref['warm'],name_quadrant_warm,nhalo)
		ds_quadrant_warm = xr.open_dataset(name_quadrant_warm,chunks=chunksNOHALO)
		
	# ===================================================
	# Fig.1,2,3 First look
	# ===================================================

	if NUM_FIG==1 or NUM_FIG==2 or NUM_FIG==3 or NUM_FIG==-1:
		print(' * First look')
		# this is in paper
		time = -1 
		height = 300 #m
		path_save = path_outpng+Name_out
		FIRST_LOOK(dataSST,dsINI,dsB,dsmean,dsref,dsflx,X,Y,Z,Z_w,time,nhalo,height,crit_value,path_save,dpi)

	# ===================================================
	# Fig.4 THTV and flux wthtv
	# 	    U and flux uw
	# ===================================================
	if NUM_FIG==4 or NUM_FIG==-1:
		# to do : w'thtv' (résolus + sgs) à coté des profils moyens (enlever les profils du dernier instant)
		print(' * Theta_v and fluxes at differents positions')
		Q0_atX = 4000 	# (m) X location to chose to normalize
		NORM = True 	# (bool) norm w'tht' with Q* at 'Q0_atX' m
		X_liste = [4000,10000,13000,23000,26000,38000] # (m)
		#X_liste = [4000,7500,15000,20000,26000,38000]
		path_save = path_outpng+Name_out
		PROFILES_AT_X_THTV_THVWFLX(dataSST,dsflx,dsmean,dsref,res,X,Z,X_liste,Q0_atX,NORM,crit_value,path_save,dpi)
		#
		#-------------------------------------
		print(' * U and fluxes at differents positions')
		ustar_atX = 4000 	# (m) X location to chose to normalize
		NORM = True 		# (bool) norm u'w' with u*² at 'ustar_atX' m
		X_liste = [4000,10000,13000,23000,26000,38000] # (m)
		path_save = path_outpng+Name_out
		PROFILES_AT_X_U_UWFLX(dataSST,dsflx,dsmean,dsref,X,Z,X_liste,ustar_atX,NORM,crit_value,path_save,dpi)
		#
		#-------------------------------------
	# ===================================================
	# Fig.5 Use of budget from NAM_BUDGET and by Hand
	# ===================================================	
	if NUM_FIG==5 or NUM_FIG==-1:
		print(' * Budgets')
		print('	- Budgets terms function of position x')
		#
		height = [10,120,300,540] # in meters, good for paper : [540,300,120,10], must be 4 items
		factor = 10000
		VAR_BU = {'u':['u_cor','u_hadv','u_hturb','u_pres','u_vadv','u_vturb'],
			'v':['v_cor','v_hadv','v_hturb','v_vadv','v_vturb'],
			'w':['w_hadv','w_hturb','w_vadv','w_vturb','w_boytotale','w_cor'],
			'ET':['ET_DIFF','ET_DISS','ET_HADV','ET_VDP','ET_TP'],
			'tht':['THT_HTURB','THT_HADV','THT_VADV','THT_VTURB']}
		BORNES = {'u':[-5,4],'v':[-0.0005,0.0005],'w':[-0.005,0.005],'ET':[-0.002,0.002],'tht':[-0.0004,0.0004]}
		COLORS = {'TEND':'k','hadv':'sienna','vadv':'tan','pres':'blue','vturb':'green','hturb':'chartreuse','cor':'orange','boy':'red','boytotale':'red',
				'DIFF':'blue','DISS':'magenta','HADV':'sienna','VDP':'green','TP':'red','HTURB':'chartreuse','VTURB':'green','VADV':'tan','HADV':'sienna'}
		X_BUDGET_HAND(dataSST,dsB,ds_hbudget,X,Z,height,VAR_BU,COLORS,BORNES,factor,path_budget_hand,dpi)
		#
		#-------------------------------------			
	# ===================================================
	# Fig.6 How the large scale relations behave here ??
	# ===================================================
	if NUM_FIG==6 or NUM_FIG==-1:
		path_save = path_outpng+Name_out			
		print(' * Correlations to check linear relations')
		# unified function, this figure is used in paper
		CHOICE = 'Um' 			# 'Um','Wm','Vm','THm','THvm'
		D_VAR = 1 				# order of derivative of CHOICE
		D_SST = 1 				# order of derivative of SST
		S_SST = -1 				# sign of SST
		atZ = 10 				# m, height of CHOICE
		PRESET = 'paper'		# None or DMM or PA or LA13 or DMMtau or paper
		res = 50 				# m
		V_INTEG = False 			# vertical integration or not
		UPFRONT_ONLY = True 	# compute correlation with only the 1st front
		#
		Corr_atZ_AllInOne(X,Z,dsmean,dsB.SST[1,nhalo:-nhalo],CHOICE,D_VAR,D_SST,S_SST,atZ,V_INTEG,UPFRONT_ONLY,PRESET,res,path_save,dpi)
		#
		#-------------------------------------
	# ====================================================================
	# Fig.7 Coherent structures analysis : quadrants
	# ====================================================================
	if NUM_FIG==7 or NUM_FIG==-1:
			print(' * Quadrant analysis')
			# Paper for reference : 
			# 	- neutral rough ABL ref : Lin 1996
			# 	- convective ABL ref : Salesky 2016
			#
			L_ds = {'cold':xr.open_dataset(L_BACKUP['cold']),	# 3D field with U,W,THT,RV,THTV
					'warm':xr.open_dataset(L_BACKUP['warm']),
					'S1':dsO}
			L_dsmean = { 'cold':dsref['cold'],											# mean fields
						'warm':dsref['warm'],
						'S1':dsmean}
			L_dsquadrant = {'cold':ds_quadrant_cold,									# quadrants of fluxes
							'warm':ds_quadrant_warm,
							'S1':ds_quadrant}		

			print('		- momentum efficiency at some X')
			#
			# this is in paper
			L_atX = np.array([4.0,10.0,13.0,23.0,26.0,38.0]) # in km
			zimax = 0.3 # paper is with 0.3
			path_save = path_TURB
			#
			plot_uw_efficiencies_atX(X,Z,Tstart,Tstop,window,crit_value,dsflx,L_dsquadrant,dsref,L_atX,dataSST,zimax,path_save,dpi)
			#
			#--------------------------------------------------			
	
	# ====================================================================
	# Fig.9,10,A1 Coherent structures analysis : conditionnal sampling
	# ====================================================================
	if NUM_FIG==9 or NUM_FIG==10 or NUM_FIG=='A1' or NUM_FIG==-1:		
		print(' * Conditionnal sampling C10')
		# + INPUT FOR BUILDING FILES |
		print('	- Building conditionnal sampling files')
		path_CS1 = 'DATA_turb/'+Name_out
		path_save = path_outpng+'TURB_STRUCTURE/'
		data = {'cold':xr.open_dataset(L_BACKUP['cold']), 
				'warm':xr.open_dataset(L_BACKUP['warm']),
				'S1':dsO} 						# 'S1decayHalf':xr.open_mfdataset('./CAS06_SST_is_Tini_SVT_DecayHalf/FICHIERS_OUT/*nc',chunks=chunksOUT)
											# 'warm_wide':xr.open_dataset('CAS10_SVT_wide/CAS10.1.002.002.nc')
		data_mean = {'cold':dsref['cold'],'warm':dsref['warm'],'S1':dsmean} 
													# ,'S1decayHalf':xr.open_dataset('DATA_turb/06WdecayHalf_mean_fields.nc',chunks=chunksNOHALO)
													# ,'warm_wide':dsref['warm_wide']
		param_COND = {'C10':(1,	# if C10 : strength of the conditionnal sampling (default = 1)
					0.005),		# if C10 : set the fraction of <rv'²> under each z to detect objects (default = 0.05)
								# if C10 and RV : m=0.3 g=1 ?
				'ITURB':0.05,	# if ITURB : set the minimum turbulence intensity to detect objects (defaut for ref = 0.05)
				'ITURB2':0.75,	# if ITURB2 : set the fraction of mean of Iturb under each z
				'ITURB3':(1,0.75),# if ITURB3 : set strength of CS and minimum of turbulence, if m=1 then is ITURB2
				'EC':0.5}		# if EC : set the fraction of mean of Ec under each z	
		L_TURB_COND = ['C10']# 'C10','ITURB','ITURB2','EC' C10 is the threshold of Couvreux 2010
		SVT_type = 'SVT'		# 'SVT' or 'RV' : choose which tracer to use
		RV_DIFF = 'MEAN' 		# if RV : how rv' are computed : MEAN, MIXED, MIDDLE_MIXED. best is MIXED
		SEUIL_ML = 0.5 			# K/km, for mixed layer detection on thtv
		indt = -1				# time index for ref simus, -1 <=> t=+3h
		build_CS1(nhalo,data,data_mean,param_COND,L_TURB_COND,SVT_type,RV_DIFF,SEUIL_ML,indt,path_CS1)
		print('		done!')
		# - END INPUT FOR BUILDING FILES |
		
		# + INPUT FOR PLOTS |
		K = 0.05 							 # surface coverage threshold for profiles plots (needs to be <1)
		L_var = ['U','W','RVT','THT','THTV'] # for profiles and XY plots : 'U','W','RVT','THT','THTV'			
		L_plot = ['flx_profile_nice'] # 	L_plot is the selector of plots to show 
		#					['ML_bounds','turb_cond','profils_var','std_var_profiles','uw_profile','wtht_profile','wthtv_profile','wrv_profile','XY','XZ','YZ','obj_frac','obj_movie']
		TURB_COND = 'C10' 	# 'C10' or 'ITURB2'
		SVT_type = 'SVT' 	# 'RV' or 'SVT'
		RV_DIFF = 'MIXED' 	# if SVT_type='RVT'
		if SVT_type=='SVT':
			RV_DIFF='MEAN'
		# - END INPUT FOR PLOTS |
		
		if NUM_FIG==9 or NUM_FIG==-1:
			print('	- Looking at coherent structures for ref simus')
			#
			dsCS1 = {'warm' : xr.open_dataset('DATA_turb/S1_CS1_warm_C10_SVTMEAN_m1.nc',chunks=chunksNOHALO),
					'cold' : xr.open_dataset('DATA_turb/S1_CS1_cold_C10_SVTMEAN_m1.nc',chunks=chunksNOHALO)}
			atzi = 0.9 	# k*zi, for plot XY
			aty = 20 	# j index, for plot XZ
			atx = 269	# i index, for plot YZ
			indt = -1 	# time index, -1 <=> t=+3h
			L_case = ['warm','cold'] # 'cold' and/or 'warm'
			if ('flx_profile_nice' in L_plot) and (TURB_COND=='C10'):
				path_save_NICE = path_save + 'NICE_FLX_REF'+'C10_m1_g0.5_SVT' # to be changed if CS changes
				plots_nice_flx_ref_CS1(Z,dsCS1,dsref,path_save,dpi)
			else:
				plots_ref_CS1(X,Y,Z,dsCS1,dsref,K,atzi,aty,atx,SEUIL_ML,SVT_type,indt,L_case,L_var,L_plot,path_save_NICE,dpi)
			#
			#--------------------------------------------------	
			print('	- uw decomposition at 10 ,13 23 km for S1')
			# = plot_CS1_S1 in a nice way
			# this figure is in paper
			L_atX = np.array([10.0,13.0,23.0])
			path_saving = path_outpng+'TURB_STRUCTURE/'
			dsCS1 = xr.open_dataset('DATA_turb/S1_CS1_S1_C10_SVTMEAN_m1.nc',chunks=chunksNOHALO)
			PLOT_REF = False	# plot total flux of ref simulations (paper = False)
			PLOT_CONTRIB = True	# decompose mean flux into structures' contribution (paper = True)
			#
			uw_decomposition_10_13_23_km(X,Z,dsCS1,dsref,dsflx,L_atX,PLOT_REF,PLOT_CONTRIB,path_saving,dpi)
			#
			#--------------------------------------------------	
		if NUM_FIG=='A1' or NUM_FIG==-1: #or NUM_FIG==-1:
			print('	- Mean flow advected coherent strutures: a movie')
			#
			# Advection of structures : building the history of a structure based on a advection velocity
			#
			# U can be :
			#	- the mean profile (x,z)
			#	- a domain mean at each instant (t,z)
			#	- a domain mean of the mean profile (z)
			#	- a integrated velocity inside the ABL (t)
			#	
			# First try : advection by a constant velocity, integrated from the x averaged mean profile ()
			# so the fetch to get the previous cell at t - dt(=30s) is approximately U*30s/dx = 4
			#
			# Note : t*=15min => tmax-tmin should be around 30 to capture ascent and descent of a convective structure
			
			L_TURB_COND = ['C10'] # 'C10','ITURB2'
			ini_t = 120 		# in index of time, instant of interested to start from, = ncview -1
			ini_x = 395 		# in index of ni, location of the structure to start from
			tmin = 60		# how far back in time to look 
			tmax = 120  		# how far forward in time to look
			fps = 5			# movie frame per seconde
			stepy,stepz = 1,1 	# vector field : skipping cells 
			Awidth = 0.002		# vector field : arrow width
			scale = 70			# vector field : size of arrows
			path_save = path_outpng + 'object_videos/'
			case = 'clean' 		# 'clean': less things on the plot
			#
			if case=='clean': # up/down obj, rv and SST
				movie_coherent_structures_cleaner(X,Y,Z,chunksNOHALO,L_TURB_COND,dataSST,SEUIL_ML,ini_t,ini_x,tmin,tmax,fps,stepy,stepz,Awidth,scale,path_save)
			elif case=='both':
				movie_coherent_structures_cleaner(X,Y,Z,chunksNOHALO,L_TURB_COND,dataSST,SEUIL_ML,ini_t,ini_x,tmin,tmax,fps,stepy,stepz,Awidth,scale,path_save)
				movie_coherent_structures(X,Y,Z,chunksNOHALO,L_TURB_COND,dataSST,SEUIL_ML,ini_t,ini_x,tmin,tmax,fps,stepy,stepz,Awidth,scale,path_save)
			elif case=='full': # all obj, sv1,sv4,sv3,rv and SST
				movie_coherent_structures(X,Y,Z,chunksNOHALO,L_TURB_COND,dataSST,SEUIL_ML,ini_t,ini_x,tmin,tmax,fps,stepy,stepz,Awidth,scale,path_save)
			#
			#--------------------------------------------------					
		if NUM_FIG==10 or NUM_FIG==-1:
			TURB_COND = 'C10' # C10 or ITURB3
			print('	- '+TURB_COND+': m sensitivity study') 
			# this is used in paper
			#
			# ITURB3 or C10 : m sensitivity study
			#
			# 0, TBD?) Masked of what is turbulent (i=329,t=120) with objects + physical fields (iturb, tracers, rv, wind field)
			# 1) profiles of uw, 
			# 2) and further decomposed in tophat+intra var contributions. This is in paper
			# 3) profiles of cover for each object
			#
			path_save = path_outpng
			i = 259 	# ni index for YZ plot
			t = 120 	# time index for YZ plot
			atX = 13 	# km, for fluxes
			L_choice = ['2'] # 1, 2, 3
			#
			CS_m_sensitivity(X,Y,Z,data,data_mean,dsflx,TURB_COND,L_choice,chunksNOHALO,i,t,atX,path_save,path_CS1,dpi)
			#
			#--------------------------------------------------
			
	# ============================================================
	# Fig.8 Characteristics of updrafts
	# ============================================================
	if NUM_FIG==8 or NUM_FIG==-1:
		print(" * Characteristics of updrafts")
		# Buoy_in_structures + u_and_w_fluc_in_updrafts in a nice way
		# this figure is in paper
		BLIM = [-6,6] # x10e-3
		L_atX = np.array([4,10.0,13.0,23.0,26.0,38.0])
		path_saving = path_TURB
		K = 5/100 # minimum surface covered of a structure to plot profiles
		dsCS1 = xr.open_dataset('DATA_turb/S1_CS1_S1_C10_SVTMEAN_m1.nc',chunks=chunksNOHALO) # to get 'updrafts' detection
		dsCS1cold = xr.open_dataset('DATA_turb/S1_CS1_cold_C10_SVTMEAN_m1.nc',chunks=chunksNOHALO)
		dsCS1warm = xr.open_dataset('DATA_turb/S1_CS1_warm_C10_SVTMEAN_m1.nc',chunks=chunksNOHALO)
		#
		updraft_charateristics(X,Z,dsCS1,dsCS1warm,dsCS1cold,dataSST,crit_value,Tstart,Tstop,window,BLIM,L_atX,K,path_saving,dpi)
		#
		#-------------------------------------
	# ===================================================
	# Fig.11 Exploring the 'thermal overshoot'
	# ===================================================
	if NUM_FIG==11 or NUM_FIG==-1:
		print(' * Exploring the thermal overshoot')	
		if True:
			# this is in paper (background of recap)
			Q0_atX = 4000	
			liste_x = [4,10,15,20,27.5,38] 
			path_save = path_outpng+Name_out
			#
			T_OVERSHOOT_FLX_WTHTV(X,Z,dataSST,dsflx,dsmean,liste_x,Q0_atX,path_save,dpi)
			#
			#-------------------------------------	
		if True:
			# This is in paper		
			# plotting a few velocity profil for the recap
			Lx = [4.0,13.0,23.0] # km
			Lindx = [nearest(X.values,Xpos*1000) for Xpos in Lx]
			linewidth = 3
			alpha = 1
			U = dsmean.Um
			for kx in range(len(Lindx)):
				fig, ax = plt.subplots(1,1,figsize = (3,3),constrained_layout=True,dpi=200)
				ax.plot(U[:,Lindx[0]],Z/ABLH_S1,c='grey',lw=linewidth,alpha=alpha,ls='--')
				ax.plot(U[:,Lindx[1]],Z/ABLH_S1,c='grey',lw=linewidth,alpha=alpha,ls='--')
				ax.plot(U[:,Lindx[2]],Z/ABLH_S1,c='grey',lw=linewidth,alpha=alpha,ls='--')
				if kx==0:
					color='b'
				elif kx==1:
					color='k'
				elif kx==2:
					color='r'
				color = 'k'
				ax.plot(U[:,Lindx[kx]],Z/ABLH_S1,c=color,lw=linewidth+2)
				ax.set_ylim([0,1.2])
				ax.set_xlim([6,7.6])
				ax.vlines(7.5,0,2,colors='grey')
				ax.spines[['right', 'top']].set_visible(False)
				ax.set_yticks([1.0])
				ax.set_yticklabels([1], fontsize=20)
				ax.set_xlabel(r'U', fontsize=20)
				ax.set_ylabel('Z/$z_i$', fontsize=20)
				ax.tick_params(axis='both', bottom=False, labelbottom=False)
				fig.savefig(path_outpng+'forRecap_U_X'+str(Lx[kx])+'km.png')
			
	# ==================================================
	# Fig.S1 Optimal radioactive decay estimation
	# ==================================================
	#
	# Here we try to have a rough estimation of the tau in the radio-decay equation for the 
	#	passive tracer injected by NAM_CONSAMP.
	# 9 tau have been tested : 1min,4min,7min,10min,12min,15min,20min,30min,40min

	if NUM_FIG=='S1' or NUM_FIG==-1:
		print(' * Investigating radio decay')
		path_CS1 = 'radio_decay_sensitivity/'
		os.system('mkdir -p radio_decay_sensitivity')
		data = {'1min':xr.open_dataset(pathWORKDIR + path_CS1+'/1min/FICHIERS_OUT/NU1mi.1.001.OUT.001.nc',chunks=chunksOUT),
				'4min':xr.open_dataset(pathWORKDIR + path_CS1+'/4min/FICHIERS_OUT/NU4mi.1.001.OUT.001.nc',chunks=chunksOUT),
				'7min':xr.open_dataset(pathWORKDIR + path_CS1+'/7min/FICHIERS_OUT/NU7mi.1.001.OUT.001.nc',chunks=chunksOUT),
				'10min':xr.open_dataset(pathWORKDIR + path_CS1+'/10min/FICHIERS_OUT/NU10m.1.001.OUT.001.nc',chunks=chunksOUT),
				'12min':xr.open_dataset(pathWORKDIR + path_CS1+'/12min/FICHIERS_OUT/NU12m.1.001.OUT.001.nc',chunks=chunksOUT),
				'15min':xr.open_dataset(pathWORKDIR + '/RefC/FICHIERS_OUT/CAS09.1.001.OUT.001.nc',chunks=chunksOUT),
				'20min':xr.open_dataset(pathWORKDIR + path_CS1+'/20min/FICHIERS_OUT/NU20m.1.001.OUT.001.nc',chunks=chunksOUT),
				'30min':xr.open_dataset(pathWORKDIR + path_CS1+'/30min/FICHIERS_OUT/NU30m.1.001.OUT.001.nc',chunks=chunksOUT),
				'40min':xr.open_dataset(pathWORKDIR + path_CS1+'/40min/FICHIERS_OUT/NU40m.1.001.OUT.001.nc',chunks=chunksOUT)}		
		data_mean = {'1min':Open_LES_MEAN(pathWORKDIR + path_CS1+'/1min/NU1mi.1.001.000.nc'),
				'4min':Open_LES_MEAN(pathWORKDIR + path_CS1+'/4min/NU4mi.1.001.000.nc'),
				'7min':Open_LES_MEAN(pathWORKDIR + path_CS1+'/7min/NU7mi.1.001.000.nc'),
				'10min':Open_LES_MEAN(pathWORKDIR + path_CS1+'/10min/NU10m.1.001.000.nc'),
				'12min':Open_LES_MEAN(pathWORKDIR + path_CS1+'/12min/NU12m.1.001.000.nc'),
				'15min':Open_LES_MEAN(pathWORKDIR + '/RefC/CAS09.1.001.000.nc'),
				'20min':Open_LES_MEAN(pathWORKDIR + path_CS1+'/20min/NU20m.1.001.000.nc'),
				'30min':Open_LES_MEAN(pathWORKDIR + path_CS1+'/30min/NU30m.1.001.000.nc'),
				'40min':Open_LES_MEAN(pathWORKDIR + path_CS1+'/40min/NU40m.1.001.000.nc')}
		
		param_COND = {'C10':(1,	# if C10 : strength of the conditionnal sampling (default = 1)
					0.005),		# if C10 : set the fraction of <rv'²> under each z to detect objects (default = 0.05)
								# if C10 and RV : m=0.3 g=1 ?
				'ITURB':0.05,	# if ITURB : set the minimum turbulence intensity to detect objects (defaut for ref = 0.05)
				'ITURB2':0.75,	# if ITURB2 : set the fraction of mean of Iturb under each z
				'ITURB3':(1,0.75),# if ITURB3 : set strength of CS and minimum of turbulence, if m=1 then is ITURB2
				'EC':0.5}		# if EC : set the fraction of mean of Ec under each z	
		L_TURB_COND = ['C10']# 'C10','ITURB','ITURB2','EC' C10 is the threshold of Couvreux 2010
		SVT_type = 'SVT'		# 'SVT' or 'RV' : choose which tracer to use
		RV_DIFF = 'MEAN' 		# if RV : how rv' are computed : MEAN, MIXED, MIDDLE_MIXED. best is MIXED
		#
		Contrib_flux_Tau(chunksNOHALO,data,data_mean,param_COND,L_TURB_COND,SVT_type,RV_DIFF,path_CS1,dpi)
		#
		#-------------------------------------
	# ===================================================
	# Some small specific plots
	# ===================================================
	if False:
		print(' * Ratio res/sgs')
		# this figure is used in paper
		ET = dsmean.ETm
		E = dsmean.Em
		TKE = dsmean.TKEm
		R = E/ET
		zi=600
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
		s = ax.pcolormesh(X/1000,Z/zi,R,cmap='Spectral_r',vmin=0,vmax=1)
		ax.contour(X/1000,Z/zi,R,levels=[0.8],colors='k')
		plt.colorbar(s,ax=ax)
		ax.set_ylim([0,1.2])
		ax.set_ylabel('z/zi')
		ax.set_xlabel('X (km)')
		ax.set_title('E(res) / E(total), contour = 0.8')
		fig.savefig(path_outpng+'ratio_res_over_sgs.png')
	if False:
		print(' * Slice of relative humidity')
		#
		time = -1
		path_save = path_outpng+Name_out+'RH_lasttime.png'
		RH_SLICE(dsB,time,nhalo,path_save,dpi)
		#
		#-------------------------------------
	if False:
		print('ABLH of mean field')
		fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
		gTHTV = dsmean.THTvm.differentiate('level')
		ABLH = np.zeros(X.shape)
		for x in range(len(X)):
			ABLH[x] = Z[np.argmax(gTHTV[:,x].values)]
		ax.plot(X/1000,ABLH,c='k')
		ax.set_ylabel('ABLH (m)')
		ax.set_xlabel('X (km)')
		#
		#-------------------------------------
	if client != None:			
		client.shutdown()
	

	
	
	
	
	
	
	

