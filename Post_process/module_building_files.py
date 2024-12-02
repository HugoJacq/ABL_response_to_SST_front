# To be used with analyse.py 
import xarray as xr
import numpy as np
import os
import pathlib
from module_cst import *
from module_tools import *
from module_CS_func import get_mixed_layer_indexes,Compute_bool_turb
import time



def Opening_files(chunksOUT,path_OUT,path_BUDGET,L_BUDGET,L_OUT,L_BACKUP,L_var_budget,group,path_INI,CHOIX,
				path_ref):
	"""
	Master procedure that open all files.
		-> complete file (BACKUP) in dsB
		-> OUTPUT files (OUT) in dsO
		-> diachronic file (000) in ds000
		-> initial file (from PRE_IDEAL) in dsINI
		-> reference file (000 from reference sim) in dsref
	"""
	# Bugfix if dsB = xr.open_mfdataset
#	for filename in L_BACKUP[CHOIX]:
#		bugfix_func(filename)
	
	
	dsB = xr.open_dataset(L_BACKUP[CHOIX][0],chunks=chunksOUT) # all fields 
	dir = os.listdir(path_OUT[CHOIX])
	if len(dir)!=0:
		#dsO = xr.open_mfdataset(path_OUT[CHOIX],chunks=chunksOUT, parallel=True) # U,V,THT,RVT,TKET and subgrid quantities  
		dsO = xr.open_mfdataset(path_OUT[CHOIX]+L_OUT[CHOIX],chunks=chunksOUT) # sometimes parallel opening is causing problems, and is not loading every files
		print('	number of OUT:',dsO.time.shape[0])
	else:
		raise Exception('No OUTPUT files found for case '+CHOIX)
	#dir = os.listdir(path_BUDGET[CHOIX])
	dsBU = {}
	if L_BUDGET[CHOIX]!='':#if len(dir)!=0:
		for var in L_var_budget:
			dsBU[var] = Open_BU(path_BUDGET[CHOIX]+L_BUDGET[CHOIX],group+var+'/')
	else:
		print('	No BUDGET file found for case '+CHOIX)
		print('	I will use the OUPUT files as dsBU and ds000')
		dsBU = dsO
		ds000 = dsO
	dsINI = xr.open_dataset(path_INI[CHOIX]) # ,chunks=chunksOUT
	dsref = {}
	for SIM in path_ref:
		dsref[SIM] = Open_LES_MEAN(path_ref[SIM])
	return dsB,dsO,dsBU,ds000,dsINI,dsref

def Open_LES_MEAN(path_000):
	"""
	This procedure opens a .000.nc file from MNH simulation.
	It returns a dictionnary of dataset with keys corresponding to the successive categories of the NAM_LES output.
	
	INPUTS:
		path_000	: string to locate 000.nc file 
	
	OUT:
		dataset[time averaged or not][type of variable][variable]
		
			1rst key : mean, nomean, 000 ('000' contains everything that is outside the LES_budgets group)
			2nd key : Mean,Resolved,Subgrid,Surface,Misc
			3rd key : variable something like MEAN_UU or BL_H
			
	Example to get the horizontaly averaged U field (but not time averaged):
		path_000 = 'SIM1.1.001.000.nc'
		dataSimu = Open_SimuRef(path_000)
		MEAN_UU = dataSimu['nomean']['Mean']['MEAN_UU']
		
		this reproduces the acces to the following group:
		'LES_budgets/Mean/Cartesian/Not_time_averaged/Not_normalized/cart/MEAN_UU'
	"""
	nameGroups = { 'nameKE' : 'LES_budgets/BU_KE/Cartesian/',
			'nameMean' : 'LES_budgets/Mean/Cartesian/',
			'nameResolved' : 'LES_budgets/Resolved/Cartesian/',
			'nameSGS' : 'LES_budgets/Subgrid/Cartesian/',
			'nameSurface' : 'LES_budgets/Surface/Cartesian/',
			'nameMisc' : 'LES_budgets/Miscellaneous/Cartesian/',
			'meaned' : 'Time_averaged/Not_normalized/cart/',
			'notmeaned' : 'Not_time_averaged/Not_normalized/cart/'}
	ds = {}
	ds['000'] = xr.open_dataset(path_000)
	if 'time_les_avg' in ds['000'].keys(): 
		ds['mean'] = {'KE':xr.open_dataset(path_000,group= nameGroups['nameKE']+nameGroups['meaned']),
				'Mean': xr.open_dataset(path_000,group= nameGroups['nameMean']+nameGroups['meaned']),
				'Resolved': xr.open_dataset(path_000,group= nameGroups['nameResolved']+nameGroups['meaned']),
				'Subgrid': xr.open_dataset(path_000,group= nameGroups['nameSGS']+nameGroups['meaned']),
				'Surface': xr.open_dataset(path_000,group= nameGroups['nameSurface']+nameGroups['meaned']),
				'Misc': xr.open_dataset(path_000,group= nameGroups['nameMisc']+nameGroups['meaned'])
				}
	else:
		ds['mean'] = 'NOTE : no time average in this file'
	ds['nomean'] = {'KE':xr.open_dataset(path_000,group= nameGroups['nameKE']+nameGroups['notmeaned']),
			'Mean': xr.open_dataset(path_000,group= nameGroups['nameMean']+nameGroups['notmeaned']),
			'Resolved': xr.open_dataset(path_000,group= nameGroups['nameResolved']+nameGroups['notmeaned']),
			'Subgrid': xr.open_dataset(path_000,group= nameGroups['nameSGS']+nameGroups['notmeaned']),
			'Surface': xr.open_dataset(path_000,group= nameGroups['nameSurface']+nameGroups['notmeaned']),
			'Misc': xr.open_dataset(path_000,group= nameGroups['nameMisc']+nameGroups['notmeaned'])
			}
	for Name in ['KE','Mean','Resolved','Subgrid','Surface','Misc']:
		ds['nomean'][Name]['time_les'] = ds['000']['time_les'].values
		ds['nomean'][Name]['level_les'] = ds['000']['level_les'].values
		if 'time_les_avg' in ds['000'].keys(): 
			ds['mean'][Name]['time_les_avg'] = ds['000']['time_les_avg'].values
			ds['mean'][Name]['level_les'] = ds['000']['level_les'].values
	return ds
	
def Open_BU(path_000,group):
	"""
	This procedure opens a .000.nc file from MNH simulation.
	It returns a dataset for the specified group, where user can choose a budget from a chosen variable.
	
	INPUTS:
		path_000	: string to locate 000.nc file 
		group		: string to specify the variable to get budget from
	
	Example:
		path_000 = 'SIM1.1.001.000.nc'
		group = 'Budgets/UU/'
		dataU_budget = Open_BU(path_000,group)
	"""
	dsBU = xr.open_mfdataset(path_000,group=group)
	ds000 = xr.open_mfdataset(path_000)
	# when opening a file with the group option, variable (or here coordinates)
	#	are not in the opened dataset anymore so we need to add them back
	#	This is usefull later for methods like .differentiate('level')
	dsBU['cart_ni_u'] = ds000.cart_ni_u
	dsBU['cart_ni_v'] = ds000.cart_ni_v
	dsBU['cart_ni'] = ds000.cart_ni
	dsBU['cart_level'] = ds000.cart_level
	dsBU['cart_level_w'] = ds000.cart_level_w
	dsBU.set_coords('cart_ni_u')
	dsBU.set_coords('cart_ni_v')
	dsBU.set_coords('cart_level')
	dsBU.set_coords('cart_level_w')
	return dsBU
		
def Build_mean_file(path_in,dsO,dsB,Tstart,Tstop,name_mean,nhalo,window):
	"""
	This procedure is building a file with mean fields of pronostics variables 
		from OUTPUT files of MNH simulation
	
	INPUTS :
		- path_in 	: Path of the OUTPUT files (to indicate origin of files in .nc)
		- dsO 		: Dataset with OUTPUT files
		- dsB 		: Dataset with BACKUP files
		- Tstart 	: Number of the file at which to start temporal average
		- Tstop 	: Number of the file at which to stop temporal average
		- name_mean : Name of the written file
		- nhalo 	: HALO from MNH
		- window 	: Size of the window for MeanTurb (in points)
		
	OUTPUTS :
		a netcdf file written at 'name_mean', with informations about the mean
			and the origin of the files used in the averaging as attributes.
	"""
	X = dsO.ni[nhalo:-nhalo]
	Y = dsO.nj[nhalo:-nhalo]
	Z = dsO.level[nhalo:-nhalo]
	Time = dsO.time
	if len(dsB.time.values) > 1:
		rho = dsB.RHOREFZ[-1,nhalo:-nhalo].values
	else:
		rho = dsB.RHOREFZ[nhalo:-nhalo].values
	# interpolation at mass point
	U = dsO.UT.interp({'ni_u':dsO.ni})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]					# Grid : 2
	V = dsO.VT.interp({'nj_v':dsO.nj})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]					# Grid : 3
	W = dsO.WT.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 				# Grid : 4
	TKET = dsO.TKET[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 										# grid 1
	UW_HFLX = dsO.UW_HFLX.interp({'level_w':dsO.level,'ni_u':dsO.ni})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# Grid : 6
	UW_VFLX = dsO.UW_VFLX.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]					# Grid : 4
	VW_HFLX = dsO.VW_HFLX.interp({'level_w':dsO.level,'nj_v':dsO.nj})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# Grid : 7
	VW_VFLX = dsO.VW_VFLX.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]					# Grid : 4
	THW_FLX = dsO.THW_FLX.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]					# Grid : 4
	RCONSW_FLX = dsO.RCONSW_FLX.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]				# Grid : 4
	
	U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
	V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
	UW_HFLX = UW_HFLX.rename(new_name_or_name_dict={'nj_u':'nj'})
	VW_HFLX = VW_HFLX.rename(new_name_or_name_dict={'ni_v':'ni'})
	UW_FLX = UW_HFLX + UW_VFLX
	VW_FLX = VW_HFLX + VW_VFLX

	# averaging
	Um = MeanTurb(U,Tstart,Tstop,window)
	Vm = MeanTurb(V,Tstart,Tstop,window)
	Wm = MeanTurb(W,Tstart,Tstop,window)
	THTm = MeanTurb(dsO.THT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo],Tstart,Tstop,window)
	RVTm = MeanTurb(dsO.RVT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo],Tstart,Tstop,window)
	Pm = np.zeros(RVTm.shape)
	SV1m,SV2m,SV3m,SV4m = np.zeros(RVTm.shape),np.zeros(RVTm.shape),np.zeros(RVTm.shape),np.zeros(RVTm.shape)
	if 'PABST' in dsO.keys():
		Pm = MeanTurb( dsO.PABST[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] ,Tstart,Tstop,window)
	if 'SVT001' in dsO.keys():
		SV1m = MeanTurb( dsO.SVT001[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] ,Tstart,Tstop,window)
		#SV2m = MeanTurb( dsO.SVT002[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] ,Tstart,Tstop,window)
		SV3m = MeanTurb( dsO.SVT003[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] ,Tstart,Tstop,window)
	if 'SVT004' in dsO.keys():
		SV4m = MeanTurb( dsO.SVT004[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] ,Tstart,Tstop,window)
	UW_FLXm = MeanTurb( UW_FLX ,Tstart,Tstop,window) 		# subgrid
	VW_FLXm = MeanTurb( VW_FLX ,Tstart,Tstop,window) 		# subgrid
	THW_FLXm = MeanTurb( THW_FLX ,Tstart,Tstop,window) 		# subgrid
	RCONSW_FLXm = MeanTurb( RCONSW_FLX ,Tstart,Tstop,window)# subgrid
	THTvm = THTm*(1+Rv/Rd*RVTm)/(1+RVTm)
	THvW_FLXm = THW_FLXm*THTvm/THTm + 0.61*THTm*RCONSW_FLXm # subgrid
	# Turbulent kinetic nrj
	Um3D,Vm3D,Wm3D = Complete_dim_like([Um,Vm,Wm],U)
	u_fluc = (U - Um3D)
	v_fluc = (V - Vm3D)
	w_fluc = (W - Wm3D)
	Em = MeanTurb( 0.5*( u_fluc**2 + v_fluc**2 + w_fluc**2 ) ,Tstart,Tstop,window)
	TKEm = MeanTurb( TKET ,Tstart,Tstop,window)
	ETm = Em + TKEm
	
	Q = THW_FLXm[0,:] 		# at ground level
	Qv = THvW_FLXm[0,:] 	# at ground level
	E0 = RCONSW_FLXm[0,:]	# at ground level
	TAU = rho[0]*np.sqrt( UW_FLXm[0,:]**2 + VW_FLXm[0,:]**2 )
	u_star = np.sqrt( TAU / rho[0] )
	
	data_vars = {'Um':(['level','ni'],Um.data,{'long_name':'Mean zonal wind',
						'units':'m s-1',
						'grid location':'mass_center'}),
			'Vm':(['level','ni'],Vm.data,{'long_name':'Mean meridional wind',
						'units':'m s-1',
						'grid location':'mass_center'}),
			'Wm':(['level','ni'],Wm.data,{'long_name':'Mean vertical wind',
						'units':'m s-1',
						'grid location':'mass_center'}),
			'THTm':(['level','ni'],THTm.data,{'long_name':'Mean potential temperature',
						'units':'K',
						'grid location':'mass_center'}),
			'RVTm':(['level','ni'],RVTm.data,{'long_name':'Mean mixing ratio',
						'units':'kg/kg',
						'grid location':'mass_center'}),
			'THTvm':(['level','ni'],THTvm.data,{'long_name':'Mean virtual potential temperature',
						'units':'K',
						'grid location':'mass_center'}),
			'Pm':(['level','ni'],Pm.data,{'long_name':'Mean pressure',
						'units':'Pa',
						'grid location':'mass_center'}),			
			'Q_star':(['ni'],Q.data,{'long_name':'Sensible heat flux',
						'units':'m K s-1',
						'grid location':'mass_center'}),
			'Qv_star':(['ni'],Qv.data,{'long_name':'Buoyancy flux',
						'units':'m K s-1',
						'grid location':'mass_center'}),
			'E0':(['ni'],E0.data,{'long_name':'Latent heat flux',
						'units':'kg m kg-1 s-1',
						'grid location':'mass_center'}),
			'Tau':(['ni'],TAU.data,{'long_name':'Tau',
						'units':'m-1 s-2',
						'grid location':'mass_center'}),
			'u_star':(['ni'],u_star.data,{'long_name':'Friction velocity',
						'units':'m s-1',
						'grid location':'mass_center'}),
			'ETm':(['level','ni'],ETm.data,{'long_name':'Total mean turbulent kinetic energy',
						'units':'m2 s-1',
						'grid location':'mass_center'}),
			'Em':(['level','ni'],Em.data,{'long_name':'Resolved mean turbulent kinetic energy',
						'units':'m2 s-1',
						'grid location':'mass_center'}),
			'TKEm':(['level','ni'],TKEm.data,{'long_name':'Subgrid mean turbulent kinetic energy',
						'units':'m2 s-1',
						'grid location':'mass_center'}),
			'SV1m':(['level','ni'],SV1m.data,{'long_name':'mean tracer1 concentration',
						'units':'kg kg-1',
						'grid location':'mass_center'}),
			'SV2m':(['level','ni'],SV2m.data,{'long_name':'mean tracer2 concentration',
						'units':'kg kg-1',
						'grid location':'mass_center'}),
			'SV3m':(['level','ni'],SV3m.data,{'long_name':'mean tracer3 concentration',
						'units':'kg kg-1',
						'grid location':'mass_center'}),
			'SV4m':(['level','ni'],SV4m.data,{'long_name':'mean tracer4 concentration',
						'units':'kg kg-1',
						'grid location':'mass_center'}),
			}

	coords={'level': Z,'nj':Y,'ni':X}
	ds_mean = xr.Dataset(data_vars=data_vars,coords=coords,
				attrs={'Tstart':Tstart,'Tstop':Tstop,'window':window,'files':path_in})
	ds_mean.to_netcdf(path=name_mean,mode='w')  

def Build_flx_file(path_in,dsO,Tstart,Tstop,dsmean,name_fluc,nhalo,window):
	"""
	This procedure is building a file with mean turbulent fluxes.
	
	INPUTS :
		- path_in	: Path of the OUTPUT files (to indicate origin of files in .nc)
		- dsO		: Dataset of the OUTPUT files
		- Tstart 	: Number of the file at which to start temporal average
		- Tstop 	: Number of the file at which to stop temporal average
		- dsmean	: Dataset of the mean fields (built with 'Build_mean_file')
		- name_fluc	: Output name of the netcdf produced
		- nhalo 	: HALO from MNH
		- window 	: Size of the window for MeanTurb (in points)
	OUTPUT :  
		A netcdf file with averaged turbulent fluxes and informations in the netcdf attributes about
			the mean used as well as the origin of the files used.
	"""
	X = dsO.ni[nhalo:-nhalo]
	Y = dsO.nj[nhalo:-nhalo]
	Z = dsO.level[nhalo:-nhalo]
	
	THT = dsO.THT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	RVT = dsO.RVT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	THTv = THT*(1+1.61*RVT)/(1+RVT)
	
	# Interpolation at mass point
	print('	starting interpolation')
	U = dsO.UT.interp({'ni_u':dsO.ni})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] # grid : 2
	V = dsO.VT.interp({'nj_v':dsO.nj})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] # grid : 3
	W = dsO.WT.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] # grid : 4
	UW_HFLX = dsO.UW_HFLX.interp({'level_w':dsO.level,'ni_u':dsO.ni})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# Grid : 6
	UW_VFLX = dsO.UW_VFLX.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]			# Grid : 4
	VW_HFLX = dsO.VW_HFLX.interp({'level_w':dsO.level,'nj_v':dsO.nj})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# Grid : 7
	VW_VFLX = dsO.VW_VFLX.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]			# Grid : 4
	THW_FLX = dsO.THW_FLX.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]		# Grid : 4
	RCONSW_FLX = dsO.RCONSW_FLX.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]	# Grid : 4
	UV_FLX = dsO.UV_FLX.interp({'ni_u':dsO.ni,'nj_v':dsO.nj})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]		# Grid : 5
	U_VAR = dsO.U_VAR[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]		# Grid : 1
	V_VAR = dsO.V_VAR[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]		# Grid : 1
	W_VAR = dsO.W_VAR[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]		# Grid : 1
	TKET = dsO.TKET[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 		# grid 1
	# We need to rename dimensions at mass point before doing any calculation
	U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
	V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
	UW_HFLX = UW_HFLX.rename(new_name_or_name_dict={'nj_u':'nj'})
	VW_HFLX = VW_HFLX.rename(new_name_or_name_dict={'ni_v':'ni'})
	UW_FLX = UW_HFLX + UW_VFLX
	VW_FLX = VW_HFLX + VW_VFLX
	THvW_FLX = THW_FLX*THTv/THT + 0.61*THT*RCONSW_FLX 
	
	print('	reshaping')
	Um,Vm,Wm,THm,RVm,THvm = Complete_dim_like([dsmean.Um,dsmean.Vm,dsmean.Wm,dsmean.THTm,dsmean.RVTm,dsmean.THTvm],U)

	print('	computing resolved fluctuations')
	u_fluc = (U - Um)
	v_fluc = (V - Vm)
	w_fluc = (W - Wm)
	tht_fluc = (THT - THm)
	rv_fluc = (RVT - RVm)
	thtv_fluc = (THTv - THvm)

	print('	mean of fluxes')
	uv_r,uv_s = MeanTurb( u_fluc*v_fluc ,Tstart,Tstop,window),MeanTurb( UV_FLX ,Tstart,Tstop,window)
	uw_r,uw_s = MeanTurb( u_fluc*w_fluc ,Tstart,Tstop,window),MeanTurb( UW_FLX ,Tstart,Tstop,window)
	vw_r,vw_s = MeanTurb( v_fluc*w_fluc ,Tstart,Tstop,window),MeanTurb( VW_FLX ,Tstart,Tstop,window)
	uu_r,uu_s = MeanTurb( u_fluc*u_fluc ,Tstart,Tstop,window),MeanTurb( U_VAR ,Tstart,Tstop,window)
	vv_r,vv_s = MeanTurb( v_fluc*v_fluc ,Tstart,Tstop,window),MeanTurb( V_VAR ,Tstart,Tstop,window)
	ww_r,ww_s = MeanTurb( w_fluc*w_fluc ,Tstart,Tstop,window),MeanTurb( W_VAR ,Tstart,Tstop,window)
	thw_r,thw_s = MeanTurb( tht_fluc*w_fluc ,Tstart,Tstop,window),MeanTurb( THW_FLX ,Tstart,Tstop,window)
	thvw_r,thvw_s = MeanTurb( thtv_fluc*w_fluc ,Tstart,Tstop,window),MeanTurb( THvW_FLX ,Tstart,Tstop,window)
	wrv_r,wrv_s = MeanTurb( w_fluc*rv_fluc ,Tstart,Tstop,window),MeanTurb( RCONSW_FLX,Tstart,Tstop,window)
	ET = MeanTurb( 0.5*( u_fluc**2 + v_fluc**2 + w_fluc**2 ) + TKET,Tstart,Tstop,window)
	E = MeanTurb( 0.5*( u_fluc**2 + v_fluc**2 + w_fluc**2 ) ,Tstart,Tstop,window)
	TKE = MeanTurb( TKET,Tstart,Tstop,window)
	data_vars = {'FLX_UV':(['level','ni'],uv_r.data + uv_s.data,{'long_name':'Horizontal wind turbulent flux (total)',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_UW':(['level','ni'],uw_r.data + uw_s.data,{'long_name':'Turbulent vertical flux in x direction (total)',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_VW':(['level','ni'],vw_r.data + vw_s.data,{'long_name':'Turbulent vertical flux in y direction (total)',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_UU':(['level','ni'],uu_r.data + uu_s.data,{'long_name':"u'2 covariance (total)",
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_VV':(['level','ni'],vv_r.data + vv_s.data,{'long_name':"v'2 covariance (total)",
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_WW':(['level','ni'],ww_r.data + ww_s.data,{'long_name':"w'2 covariance (total)",
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_THW':(['level','ni'],thw_r.data + thw_s.data,{'long_name':'Turbulent vertical flux of heat (dry) (total)',
						'units':'K m s-1',
						'grid location':'mass_center'}),
			'FLX_THvW':(['level','ni'],thvw_r.data + thvw_s.data,{'long_name':'Turbulent vertical flux of heat (moist) (total)',
						'units':'K m s-1',
						'grid location':'mass_center'}),
			'FLX_RvW':(['level','ni'],wrv_r.data + wrv_s.data,{'long_name':'Turbulent vertical flux of vapor mixing ratio (total)',
						'units':'kg kg-1 m s-1',
						'grid location':'mass_center'}),
			'FLX_UV_s':(['level','ni'],uv_s.data,{'long_name':'Horizontal wind turbulent flux (sgs)',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_UW_s':(['level','ni'],uw_s.data,{'long_name':'Turbulent vertical flux in x direction (sgs)',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_VW_s':(['level','ni'],vw_s.data,{'long_name':'Turbulent vertical flux in y direction (sgs)',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_UU_s':(['level','ni'],uu_s.data,{'long_name':"u'2 covariance (sgs)",
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_VV_s':(['level','ni'],vv_s.data,{'long_name':"v'2 covariance (sgs)",
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_WW_s':(['level','ni'],ww_s.data,{'long_name':"w'2 covariance (sgs)",
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'FLX_THW_s':(['level','ni'],thw_s.data,{'long_name':'Turbulent vertical flux of heat (dry) (sgs)',
						'units':'K m s-1',
						'grid location':'mass_center'}),
			'FLX_THvW_s':(['level','ni'],thvw_s.data,{'long_name':'Turbulent vertical flux of heat (moist) (sgs)',
						'units':'K m s-1',
						'grid location':'mass_center'}),
			'FLX_RvW_s':(['level','ni'],wrv_s.data,{'long_name':'Turbulent vertical flux of vapor mixing ratio (sgs)',
						'units':'kg kg-1 m s-1',
						'grid location':'mass_center'}),
			'TKE':(['level','ni'],TKE.data,{'long_name':'subgrid turbulent kinetic energy',
						'units':'m2 s-2',
						'grid location':'mass_center'}),	
			'E':(['level','ni'],E.data,{'long_name':'resolved turbulent kinetic energy',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'ET':(['level','ni'],ET.data,{'long_name':'Total turbulent kinetic energy',
						'units':'m2 s-2',
						'grid location':'mass_center'}),		
			}

	coords={'level': Z,'nj':Y,'ni':X}
	print('	building dataset')
	ds_fluc = xr.Dataset(data_vars=data_vars,coords=coords,
				attrs={'Tstart':Tstart,'Tstop':Tstop,'window':window,'files':path_in})
	print('	saving')
	ds_fluc.to_netcdf(path=name_fluc,mode='w')  
	
def Build_budget_file(path_in,path_ini,dsB,dsO,dsmean,Tstart,Tstop,name_budget,nhalo,window):
	"""
	This procedure is building a file with U,V,W,THT,TKE budget components.
	
	INPUTS :
		- path_in	: Path of the OUTPUT files (to indicate origin of files in .nc)
		- path_ini	: Path of initial file
		- dsB		: Dataset of at least 1 BACKUP file (for reference state RHOREFZ)
		- dsO		: Dataset of the OUTPUT files
		- dsmean	: Dataset of the mean fields (built with 'Build_mean_file')
		- Tstart 	: Number of the file at which to start temporal average
		- Tstop 	: Number of the file at which to stop temporal average
		- name_budget: Output name of the netcdf produced
		- nhalo 	: HALO from MNH
		- window 	: Size of the window for MeanTurb (in points)
	OUTPUT :  
		A netcdf file with terms of the budget for the wind, the temperature and turbulent kinetic energy, 
			as well as informations in the netcdf attributes about the mean used and the origin 
			of the files used.
			
	NOTE : 
		- Sometimes I expand a 2D field into a 3D or 4D field. This is for xarray to work properly.
			The average operator MeanTurb applied on thoses fields thus only reduces dimensions back
			to what they were originally. 	
		- Please see 'Comments' at the end of the procedure for more details on how terms are computed.
	"""
	if 'PABST' not in dsO.keys():
		raise Exception('No pressure in OUTPUT files') 
	starttime = time.time()
	dsINI = xr.open_dataset(path_ini)
	
	# Interpolation au pt de masse
	print('	starting interpolation')
	chunksNOHALO_interp = {'time':-1,
				'level':16,
				'nj':21,
				'ni':-1}
	dsO_i = mass_interpolator(dsO,chunksNOHALO_interp)

	# Attention à bien interpoler au pt de masse
	X = dsO_i.ni.values
	Y = dsO_i.nj.values
	Z = dsO_i.level.values
	#Z_w = dsO_i.level_w.values
	#Time = dsO_i.time
	U_frc = 7.5	# m s-1
	
	THT = dsO_i.THT 		# grid 1
	RVT = dsO_i.RVT 		# grid 1
	P = dsO_i.PABST 		# grid 1
	Pi = Exner(P)
	#Pi_ref = Exner(dsINI.PABST[:,:,nhalo:-nhalo,nhalo:-nhalo].mean(dim=['time','ni','nj']))
	#Pi_ref = xr.DataArray(data=Pi_ref,coords={'level':dsO.level},name='Pi_ref')
	#Pi_ref = Pi_ref.expand_dims(dim={'nj':Y,'time':Time,'ni':X},axis=(2,0,3))
	#Pi_prime = Pi - Pi_ref
	THTv = THT*(1+Rv/Rd*RVT)/(1+RVT)
	# ref is average over xyt
	THTv_ref = THTv.mean(dim=['time','nj','ni'])
	P_ref = P.mean(dim=['time','nj','ni'])
	#RHOREF = P_ref/(Rd*Exner(P_ref)*THTv_ref) 
	
	TKE = dsO_i.TKET 		# grid 1
	U_VAR = dsO_i.U_VAR		# Grid : 1
	#V_VAR = dsO_i.V_VAR	# Grid : 1
	#W_VAR = dsO_i.W_VAR		# Grid : 1
	#thtvref3D =  ( dsINI.THT*(1+1.61*dsINI.RVT)/(1+dsINI.RVT))[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] # Grid : 1
	
	U,V,W = dsO_i.UT,dsO_i.VT,dsO_i.WT
	UW_HFLX = dsO_i.UW_HFLX
	UW_VFLX = dsO_i.UW_VFLX
	VW_HFLX = dsO_i.VW_HFLX
	VW_VFLX = dsO_i.VW_VFLX
	THW_FLX = dsO_i.THW_FLX
	#RCONSW_FLX = dsO_i.RCONSW_FLX
	UV_FLX = dsO_i.UV_FLX														
	
	UW_FLX = UW_HFLX + UW_VFLX
	VW_FLX = VW_HFLX + VW_VFLX
	#THvW_FLX = THW_FLX*THTv/THT + 0.61*THT*RCONSW_FLX 
	
	Um,Vm,Wm,THm,THvm = dsmean.Um,dsmean.Vm,dsmean.Wm,dsmean.THTm,dsmean.THTvm

	print('	computing resolved fluctuations ...')
	u_fluc = (U - Um)
	v_fluc = (V - Vm)
	w_fluc = (W - Wm)
	tht_fluc = (THT - THm)
	#rv_fluc = (RVT - RVm)
	thtv_fluc = (THTv - THvm)
	#p_fluc = (P - Pm)
	E = 0.5*( u_fluc**2 + v_fluc**2 + w_fluc**2 )
	#  total = resolved fluxes + subgrid fluxes
	print('	computing fluxes ...')
	uv = (u_fluc*v_fluc) + UV_FLX
	uw = (u_fluc*w_fluc) + UW_FLX
	vw = (v_fluc*w_fluc) + VW_FLX
	uu = (u_fluc*u_fluc) + U_VAR
	thw =(tht_fluc*w_fluc) + THW_FLX
	Em = MeanTurb(E,Tstart,Tstop,window)
	Em = xr.DataArray(data=Em,coords={'level':Z,'ni':X},name='Em')
	TKEm = MeanTurb(TKE,Tstart,Tstop,window)
	TKEm = xr.DataArray(data=TKEm,coords={'level':Z,'ni':X},name='TKEm')
	TKEm,Em =  Complete_dim_like([TKEm,Em],U) # adding missing dim and unify chunks between variables
	
	print('	computing budgets terms ...')
	print('		- U')
	u_hadv = - Um*Um.differentiate('ni')
	u_vadv = - Wm*Um.differentiate('level')
	u_hturb = - ( MeanTurb(uu,Tstart,Tstop,window) ).differentiate('ni')
	u_vturb = - ( MeanTurb(uw,Tstart,Tstop,window) ).differentiate('level')
	u_pres = - MeanTurb( Cpd*THTv*Pi.differentiate('ni') ,Tstart,Tstop,window)  # here Pi or Pi' is the same as ref is only f(z)
	u_cor =  f*Vm - f_star*Wm
	
	
	print('		- V')
	v_hadv = - Um*Vm.differentiate('ni')	
	v_vadv = - Wm*Vm.differentiate('level')
	v_hturb = - ( MeanTurb(uv,Tstart,Tstop,window) ).differentiate('ni')
	v_vturb = - ( MeanTurb(vw,Tstart,Tstop,window) ).differentiate('level')
	v_cor = -f*(Um-U_frc)
	
	# print('		- W')
	# w_hadv = - MeanTurb( Um*Wm.differentiate('ni') ,Tstart,Tstop,window) 		
	# w_vadv = - MeanTurb( Wm*Wm.differentiate('level') ,Tstart,Tstop,window)
	# w_hturb = - ( MeanTurb(uw,Tstart,Tstop,window) ).differentiate('ni')
	# w_vturb = - ( MeanTurb(ww,Tstart,Tstop,window) ).differentiate('level')
	# w_cor = + MeanTurb( f_star*Um ,Tstart,Tstop,window)	
	# w_pres = -( (P.differentiate('level') + RHOREF*g)/RHOREF ).mean(dim=['time','nj'])
	# w_boy = g*(THTv/THTv_ref - 1).mean(dim=['time','nj'])
	
	# print('		- TKE resolved')  # from MNH userguide 'LES Budgets'
	# TKE_HADV_r = - MeanTurb( Um*Em.differentiate('ni'),Tstart,Tstop,window) 	
	# TKE_VADV_r = - MeanTurb( Wm*Em.differentiate('level'),Tstart,Tstop,window)
	# TKE_PRES_r = - MeanTurb( (u_fluc*p_fluc.differentiate('ni') + 
	# 				v_fluc*p_fluc.differentiate('nj') +
	# 				w_fluc*p_fluc.differentiate('level'))/rhoref ,Tstart,Tstop,window)
	# TKE_HDP_r = - ( MeanTurb(u_fluc*u_fluc,Tstart,Tstop,window)*dsmean.Um[:,:].differentiate('ni') +
	# 		MeanTurb(u_fluc*v_fluc,Tstart,Tstop,window)*dsmean.Vm[:,:].differentiate('ni') +
	# 		MeanTurb(u_fluc*w_fluc,Tstart,Tstop,window)*dsmean.Wm[:,:].differentiate('ni') )
	# TKE_VDP_r = - ( MeanTurb(w_fluc*u_fluc,Tstart,Tstop,window)*dsmean.Um[:,:].differentiate('level') +
	# 		MeanTurb(w_fluc*v_fluc,Tstart,Tstop,window)*dsmean.Vm[:,:].differentiate('level') +
	# 		MeanTurb(w_fluc*w_fluc,Tstart,Tstop,window)*dsmean.Wm[:,:].differentiate('level') )
	# TKE_TP_r = g * MeanTurb( w_fluc*thtv_fluc/THvm ,Tstart,Tstop,window) 
	# TKE_HTR_r = - ( MeanTurb(u_fluc*E,Tstart,Tstop,window) ).differentiate('ni')
	# TKE_VTR_r = - ( MeanTurb(w_fluc*E,Tstart,Tstop,window) ).differentiate('level')
	# TKE_SBG = -( MeanTurb( u_fluc*U_VAR.differentiate('ni'),Tstart,Tstop,window) +
	# 		MeanTurb( u_fluc*UV_FLX.differentiate('nj'),Tstart,Tstop,window) +
	# 		MeanTurb( u_fluc*UW_FLX.differentiate('level'),Tstart,Tstop,window) +
	# 		MeanTurb( v_fluc*UV_FLX.differentiate('ni'),Tstart,Tstop,window) +
	# 		MeanTurb( v_fluc*V_VAR.differentiate('nj'),Tstart,Tstop,window) +
	# 		MeanTurb( v_fluc*VW_FLX.differentiate('level'),Tstart,Tstop,window) +
	# 		MeanTurb( w_fluc*UW_FLX.differentiate('ni'),Tstart,Tstop,window) +
	# 		MeanTurb( w_fluc*VW_FLX.differentiate('nj'),Tstart,Tstop,window) +
	# 		MeanTurb( w_fluc*W_VAR.differentiate('level'),Tstart,Tstop,window) )
			
	# print('		- TKE subgrid') # from MNH userguide 'LES Budgets'	
	# e_HADVM_s = - MeanTurb( Um*TKEm.differentiate('ni'),Tstart,Tstop,window) 
	# e_VADVM_s = - MeanTurb( Wm*TKEm.differentiate('level'),Tstart,Tstop,window)
	# e_ADVR_s = - MeanTurb( u_fluc*TKE.differentiate('ni') +  
	# 			v_fluc*TKE.differentiate('nj') + 
	# 			w_fluc*TKE.differentiate('level') ,Tstart,Tstop,window)
	# e_DIFF_s = MeanTurb( 1/rhoref * ( ( C2M*rhoref*LM*np.sqrt(TKE)*TKE.differentiate('ni')).differentiate('ni') +
	# 			( C2M*rhoref*LM*np.sqrt(TKE)*TKE.differentiate('level')).differentiate('level') ),Tstart,Tstop,window) # this is pressure corr + transport (triple terms). Param of mnh
				
	# e_HDPM_s = - ( MeanTurb(UV_FLX,Tstart,Tstop,window)*dsmean.Vm[:,:].differentiate('ni') +
	# 		MeanTurb(UW_FLX,Tstart,Tstop,window)*dsmean.Wm[:,:].differentiate('ni') +
	# 		MeanTurb(U_VAR,Tstart,Tstop,window)*dsmean.Um[:,:].differentiate('ni') )
			
	# e_VDPM_s = - (	MeanTurb(VW_FLX,Tstart,Tstop,window)*dsmean.Vm[:,:].differentiate('level') +
	# 		MeanTurb(W_VAR,Tstart,Tstop,window)*dsmean.Wm[:,:].differentiate('level') +
	# 		MeanTurb(UW_FLX,Tstart,Tstop,window)*dsmean.Um[:,:].differentiate('level') )
					
	# e_DPR_s = - MeanTurb(   U_VAR*u_fluc.differentiate('ni') +
	# 			UV_FLX*v_fluc.differentiate('ni') +
	# 			UW_FLX*w_fluc.differentiate('ni') +
				
	# 			UV_FLX*u_fluc.differentiate('nj') +
	# 			V_VAR*v_fluc.differentiate('nj') +
	# 			VW_FLX*w_fluc.differentiate('nj') +
				
	# 			UW_FLX*u_fluc.differentiate('level') +
	# 			VW_FLX*v_fluc.differentiate('level') +
	# 			W_VAR*w_fluc.differentiate('level') ,Tstart,Tstop,window)
	# e_TP_s = MeanTurb( g/thtvref3D*THvW_FLX ,Tstart,Tstop,window)
	# e_DISS_s = - MeanTurb( Ceps*TKE**(3/2)/LM ,Tstart,Tstop,window)
	
	print('		- theta')
	tht_hadv = - Um*THm.differentiate('ni')
	tht_vadv = - Wm*THm.differentiate('level')
	tht_vturb = - ( MeanTurb(thw,Tstart,Tstop,window) ).differentiate('level')
	tht_hturb = - ( MeanTurb(thw,Tstart,Tstop,window) ).differentiate('ni')
	
	# RV
	# TBD
	"""Comments 
	
	
	* On the validity of the assumption : total flux = resolved flux + subgrid flux --------------------------------------------------------
	
		Let us define some notations:
			<> is a ensemble mean
			A = <a> + a is the total U wind, decomposed in its mean part and fluctuation part
			a = a_r + a_s is the fluctuation of a, decomposed in its filtered part (resolved) and unresolved part
		
		for a flux of the form <ab>, we can decompose it like so :
			<ab> = <a_r*b_r> + <a_r*b_s> + <a_s*b_r> + <a_s*b_s>
			
			<a_r*b_r> 		is the mean resolved flux (mean of u_fluc*v_fluc for eg)
			<a_s*b_s> 		is the mean subgrid flux (mean of the modeled fluxes like UV_FLX in MNH)
			<a_r*b_s> + <a_s*b_r> 	is a transfert term between resolved and unresolved scales. In case the use of a cutoff filter,
							this sum of 2 terms is equal to zero.
							Here we neglect it.
							
		So in computations done in this script, <ab> =   <a_r*b_r>     +     <a_s*b_s>
								computed by 	   use of subgrid
								   hand		        fluxes from
								   		              MNH code
			And as a consequence, TKE total = E + e, annd we can add resolved and subgrid terms by terms. 
			
	* On the TKE budget ----------------------------------------------------------------------------------------------------------------------
	
		e_DIFF_s (diffusion term) subgrid is pressure + transport as in 
				Deardorff 1980 'Stratocumulus-capped Mixed Layers Derived From A Three-Dimensional Model', eq 6,7e,9a
				
	* On the transfert of TKE from resolved to subgrid scales --------------------------------------------------------------------------------
	
		what is lost by resolved scales is transfered to subgrid scales :
							     e_ADVR_s + e_DPR_s = - E_SBG
		or
				sources of subgrid TKE from resolved quantities = sink of resolved TKE from subgrid scales	
	
	* On the pressure terms (especially in the W budget) -------------------------------------------------------------------------------------
		
		How the gradient are discretised is important and can induce large divergence with the code computed budget, 
			also the change of point (from level_w to mass point for e.g.) is important.
		The reference state is the initial state, not the "REFZ" state saved in the variables in the out files (like THVREFZ,RHOREFZ). 
			This is because there is a anelastic correction done after the saving of thoses variables, and it modifies a lot the reference state.
		
		At the end of the day, the reference state is CHOSEN and as long as the pressure term + buoyancy term is consistent then everything is ok.
		note : here the formalism follows MNH terms. Could be modified.	
	 
		
		what has been tested (n°15 for pressure and n°8 for gravity term are the correct ones to reproduce MNH terms):  
			- pressure term
				
				1 : 1/rho * dP/dz + g avec rho = rhod+rhov")
					pas la bonne forme, surement à cause de l'utilisation de rho totale et pas de référence

				2 : Cpd*THTv*dPi_prime/dz Pi_ref calculé à partir de l'intégration de EXNER_FROM_TOP")
					intégration comme dans le code, c'est strictement ce qui est calculé par NAM_BUDGET
					voir terme de pression eq 2.21 scidoc1 (forme Durran)
					Bonne forme mais petit offset <0

				3 : Cpd*THTvref*dPi_prime/dz (EXNER_FROM_TOP)")
					Comme 2 mais avec thtvref au lieu de thtv
					voir terme de pression eq 2.21 scidoc1 (forme MAE)	
					bonne forme mais petit offset <0

				4 : d( Cpd*THTv*Pi_prime)/dz (EXNER_FROM_TOP)")
					tout mettre dans la dérivée ne va pas, proche surface on a pas la bonne forme

				5 : 1/rhoref * dp_fluc/dz")
					Pm devrait contenir l'état hydrostatique
					Mais Pm contient aussi d'autres termes qui font que w_pres ~ 0 ici

				6 : 1/rhoreftotal * d(P-p_ref)/dz avec p_ref = inv_exner(Pi_ref) (EXNER_FROM_TOP)")		
				7 : 1/rhoref * d(P-p_ref)/dz avec p_ref = inv_exner(Pi_ref) (EXNER_FROM_TOP)")
					Bonne forme mais petit offset >0

				8 : 1/rhodref * dP/dz + g avec rhodref = RHOREFZ")
					bonne forme mais gros offset >0

				9 : 1/rhodeff * dP/dz + g avec rhodeff = rhodref*thtvref*(1+rvref)")
					>0 et pas la bonne forme 

				10 : 1/rhoref * d(P-p_ref)/dz avec p_ref du programme PRE_IDEAL (OUTPUT_LISTING1_preideal)")
					discontinuité à z=100m ??, peut etre ce qu'est qu'un pression intermédiaire. Mais alors pourquoi est elle affiché ...

				11 : Cpd*THTv*dPi_prime/dz avec Pi_ref calculé à partir de rhoref et thtvref (Cvd)")
					equation 2.10 scidoc 1
					Non, ce n'est pas Cvd mais plutot Cpd dans les équations thermodynamiques : voir eq 8.19,8.20,8.21 de Wyngaard book p179

				12 : Cpd*THTv*dPi_prime/dz avec Pi_ref calculé à partir de rhoref et thtvref (Cpd)")
					equation 2.10 scidoc 1
					
				13 : Cpd*THTv*dPi_prime/dz avec Pi_ref calculé à partir de EXNER_FROM_BOTTOM")
				14 : Cpd*THTv* (dPI/dz + g/(Cpd*thtvref)), avec Dpi_ref/dz = -g(Cpd*thtvref)
				15 : Cpd*THTv*dPi_prime/dz avec Pi_ref calculé avec PABST du fichier initial
					Correction anelastique ok !!
				
			- the gravity term
			
				 1 thtvref is dsB.THVREFZ, at level_w points interp for thtvm (no change in time so [0] = [1])
				 2 thtvref is dsB.THVREFZ, at level_w by the same operator as in code MZM
				 3 thtvref is dsB.THVREFZ, at level points (with interp thtvref to Z levels)
				 4 thtvref is thtv_ini from THT and RVT of INIT file		
				 5 thtvref is thtv_ini from hydrostatic equation from THTv ini file		
				 6 thtvref is dsO.THVREFZ, at level_w points
				 7 thtvref is dsINI.THVREFZ, at level_w points		
				 8 thtvref is thtv_ini from THT and RVT of INIT file but considered at level_w
				 
				 
	OLD CODE : reference state is MNH initial state, pressure term uses Cpd*THTV*Pi.
			Discretisation is the same as in MNH code.
			Why this is not used ? -> the reference state is chosen and can be different from initial state. 
	
		# Calcul du terme de pression avec la discrétisation de MNH, et par rapport à l'état initial
		Dpi_primeDZ = np.zeros(Pi_prime.shape)
		DZ = np.zeros(dsB.level_w.shape)
		DZ[1:] = (dsB.level_w.values[1:] - dsB.level_w.values[:-1])
		DZ[0] = -999
		DZ = xr.DataArray(data=DZ,coords={'level_w':dsB.level_w.values},name='DZ')
		DZ = DZ.expand_dims(dim={'nj':Y,'time':Time,'ni':X},axis=(2,0,3))
		Dpi_primeDZ[:,1:,:,:] = ( Pi_prime[:,1:,:,:].values - Pi_prime[:,:-1,:,:].values) / DZ[:,1:,:,:]
		Dpi_primeDZ[:,0,:,:] = Dpi_primeDZ[:,1,:,:]
		Dpi_primeDZ = Dpi_primeDZ[:,nhalo:-nhalo,:,:]
		w_pres = - MeanTurb( Cpd*THTv*Dpi_primeDZ,Tstart,Tstop,window)
		w_pres = xr.DataArray(data=w_pres.values,coords={'level_w':Z_w,'ni':X},name='w_pres')
		w_pres = w_pres.interp({'level_w':Z})
		# Calcul du terme de gravité avec la discrétisation de MNH, et par rapport à l'état initial
		thtvref2D = ( thtvref3D ).mean(axis=(0,2,3))  # 1D grid 4, level_w
		thtvref2D = xr.DataArray(data=thtvref2D,coords={'level':Z},name='thtvref2D')
		thtvref2D = thtvref2D.expand_dims(dim={'ni':X},axis=(1))
		PA = dsmean.THTvm[:,:].values/thtvref2D.values -1
		MZM_PA = np.zeros(PA.shape)
		MZM_PA[1:,:] = 0.5*(PA[1:,:] + PA[:-1,:])
		MZM_PA[0,:] = MZM_PA[1,:]
		w_boy = g*MZM_PA
		w_boy = xr.DataArray(data=w_boy,coords={'level_w':Z_w,'ni':X},name='w_boy')
		w_boy = w_boy.interp({'level_w':Z})
	
				 
	* On why we do not have a Y advection term

	 	i) if you refer to Wyngaard p187 eq 8.65, the advection term are 
			in X : Um*dUm/dx + Vm*dUm/dY + Wm*dUm/dZ
	 		But here the average operator reduces U(t,z,y,x) to <U>(z,x)
			so dUm/dY = 0
	
		ii) if you start from the base equation, the base advection term is 
			d(rho*UiUj)/dxj. With the use of the continuity equation with variable density,
			we can write 
					d(rho*UiUj)/dxj = rho*Uj*dUi/dxj		(1)
			and so
					< d(rho*UiUj)/dxj > = < rho*Uj*dUi/dxj >
			which could let us think that the advection in Y plays a role as the term 
				VdU/dx is computed before the average operator
	
			But to get a momentum equation for the mean field with the advection term
			of the form Um_j*dUm/dxj like for the instantaneous field (with the rule (1)), we add this term 
	 		to the LHS and RHS : 
						Um_j*dUm/dxj
			and by construction we then get (on RHS) :
						Tau_ij = rho(<Ui>.<Uj> - <UiUj>)
	
			This is well done because then when writing Ui = Umi + ui we have
						Tau_ij = rho(<uiuj>) (see 2.9 p32 Wyngaard)
			
			Conclusion : the form of the equations of momentum is CHOSEN to be more pratical to use and
					 to give physical meaning to the terms (both advection and Tau_ij)
		
	 		And after that we can decompose in resolved and subgrid scales as done in this script
	"""
	
	print('	building dataset')
	data_vars = {'u_hadv':(['level','ni'],u_hadv.data,{'long_name':'Horizontal advection',
						'units':'m s-2',
						'grid location':'mass_center'}),
			'u_vadv':(['level','ni'],u_vadv.data,{'long_name':'Vertical advection',
						'units':'m s-2',
						'grid location':'mass_center'}),
			'u_hturb':(['level','ni'],u_hturb.data,{'long_name':'Horizontal turbulent stress',
						'units':'m s-2',
						'grid location':'mass_center'}),
			'u_vturb':(['level','ni'],u_vturb.data,{'long_name':"Vertical turbulent stress",
						'units':'m s-2',
						'grid location':'mass_center'}),
			'u_pres':(['level','ni'],u_pres.data,{'long_name':"Pressure gradient",
						'units':'m s-2',
						'grid location':'mass_center'}),
			'u_cor':(['level','ni'],u_cor.data,{'long_name':"Coriolis forces (fU-f_starW)",
						'units':'m s-2',
						'grid location':'mass_center'}),
			'v_hadv':(['level','ni'],v_hadv.data,{'long_name':'Horizontal advection',
						'units':'m s-2',
						'grid location':'mass_center'}),
			'v_vadv':(['level','ni'],v_vadv.data,{'long_name':'Vertical advection',
						'units':'m s-2',
						'grid location':'mass_center'}),
			'v_hturb':(['level','ni'],v_hturb.data,{'long_name':'Horizontal turbulent stress',
						'units':'m s-2',
						'grid location':'mass_center'}),
			'v_vturb':(['level','ni'],v_vturb.data,{'long_name':"Vertical turbulent stress",
						'units':'m s-2',
						'grid location':'mass_center'}),
			'v_cor':(['level','ni'],v_cor.data,{'long_name':"Coriolis forces (-fU)",
						'units':'m s-2',
						'grid location':'mass_center'}),
			# 'w_hadv':(['level','ni'],w_hadv.data,{'long_name':'Horizontal advection',
			# 			'units':'m s-2',
			# 			'grid location':'mass_center'}),
			# 'w_vadv':(['level','ni'],w_vadv.data,{'long_name':'Vertical advection',
			# 			'units':'m s-2',
			# 			'grid location':'mass_center'}),
			# 'w_hturb':(['level','ni'],w_hturb.data,{'long_name':'Horizontal turbulent stress',
			# 			'units':'m s-2',
			# 			'grid location':'mass_center'}),
			# 'w_vturb':(['level','ni'],w_vturb.data,{'long_name':"Vertical turbulent stress",
			# 			'units':'m s-2',
			# 			'grid location':'mass_center'}),
			# 'w_pres':(['level','ni'],w_pres.data,{'long_name':"Pressure gradient",
			# 			'units':'m s-2',
			# 			'grid location':'mass_center'}),
			# 'w_grav':(['level','ni'],w_boy.data,{'long_name':"Gravity",
			# 			'units':'m s-2',
			# 			'grid location':'mass_center'}),
			# 'w_cor':(['level','ni'],w_cor.data,{'long_name':"Coriolis (f_star*U)",
						# 'units':'m s-2',
						# 'grid location':'mass_center'}),
			# 'E_HADV_r':(['level','ni'],TKE_HADV_r.data,{'long_name':"Horizontal advection of resolved TKE by mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'E_VADV_r':(['level','ni'],TKE_VADV_r.data,{'long_name':"Vertical advection of resolved TKE by mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'E_PRES_r':(['level','ni'],TKE_PRES_r.data,{'long_name':"Pressure term (resolved)",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'E_HDP_r':(['level','ni'],TKE_HDP_r.data,{'long_name':"Horizontal dynamical production of resolved TKE by shear of mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'E_VDP_r':(['level','ni'],TKE_VDP_r.data,{'long_name':"Vertical dynamical production of resolved TKE by shear of mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'E_HTR_r':(['level','ni'],TKE_HTR_r.data,{'long_name':"Horizontal transport of resolved TKE by resolved fluctuations",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'E_VTR_r':(['level','ni'],TKE_VTR_r.data,{'long_name':"Vertical transport of resolved TKE by resolved fluctuations",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'E_TP_r':(['level','ni'],TKE_TP_r.data,{'long_name':"Thermal production/sink of resolved TKE",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'E_SBG':(['level','ni'],TKE_SBG.data,{'long_name':"Transport from resolved to subgrid scale",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'e_HADVM_s':(['level','ni'],e_HADVM_s.data,{'long_name':"Horizontal advection of subgrid TKE by resolved mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'e_VADVM_s':(['level','ni'],e_VADVM_s.data,{'long_name':"Vertical advection of subgrid TKE by resolved mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'e_ADVR_s':(['level','ni'],e_ADVR_s.data,{'long_name':"Advection of subgrid TKE by resolved fluctuations",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'e_DIFF_s':(['level','ni'],e_DIFF_s.data,{'long_name':"Transport term (pressure and subgrid fluctuations) of subgrid TKE",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'e_TP_s':(['level','ni'],e_TP_s.data,{'long_name':"Thermal production/sink of subgrid TKE",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'e_DISS_s':(['level','ni'],e_DISS_s.data,{'long_name':"Subgrid dissipation",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'e_HDPM_s':(['level','ni'],e_HDPM_s.data,{'long_name':"Horizontal dynamical production by mean resolved flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'e_VDPM_s':(['level','ni'],e_VDPM_s.data,{'long_name':"Vertical dynamical production by mean resolved flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'e_DPR_s':(['level','ni'],e_DPR_s.data,{'long_name':"Dynamical production by resolved fluctuations",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'ET_HADV':(['level','ni'],TKE_HADV_r.data + e_HADVM_s.data,{'long_name':"Horizontal advection of total TKE by mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'ET_VADV':(['level','ni'],TKE_VADV_r.data + e_VADVM_s.data,{'long_name':"Vertical advection of total TKE by mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'ET_HDP':(['level','ni'], TKE_HDP_r.data + e_HDPM_s.data,{'long_name':"Horizontal dynamical production of total TKE by shear of mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'ET_VDP':(['level','ni'], TKE_VDP_r.data + e_VDPM_s.data,{'long_name':"Vertical dynamical production of total TKE by shear of mean flow",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'ET_DIFF':(['level','ni'],TKE_HTR_r.data+TKE_VTR_r.data+TKE_PRES_r.data+e_DIFF_s.data,{'long_name':"Diffusion of total TKE (pressure + transport by fluctuations)",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'ET_TP':(['level','ni'],TKE_TP_r.data + e_TP_s.data,{'long_name':"Thermal production/sink of total TKE",
			# 			'units':'m2 s-3',
			# 			'grid location':'mass_center'}),
			# 'ET_DISS':(['level','ni'],e_DISS_s.data,{'long_name':"Dissipation of total TKE",
						# 'units':'m2 s-3',
						# 'grid location':'mass_center'}),
			'THT_HADV':(['level','ni'],tht_hadv.data,{'long_name':"Horizontal advection of THT by mean flow",
						'units':'K s-1',
						'grid location':'mass_center'}),
			'THT_VADV':(['level','ni'],tht_vadv.data,{'long_name':"Vertical advection of THT by mean flow",
					'units':'K s-1',
					'grid location':'mass_center'}),
			'THT_HTURB':(['level','ni'],tht_hturb.data,{'long_name':"Horizontal gradient of sensible heat flux",
					'units':'K s-1',
					'grid location':'mass_center'}),
			'THT_VTURB':(['level','ni'],tht_vturb.data,{'long_name':"Vertical gradient of sensible heat flux",
					'units':'K s-1',
					'grid location':'mass_center'}) }	 
	coords={'level': Z,'nj':Y,'ni':X}
	ds_fluc = xr.Dataset(data_vars=data_vars,coords=coords,
				attrs={'Tstart':Tstart,'Tstop':Tstop,'window':window,'files':path_in,'Note':'e_ADVR_s+e_DPR_s=-TKE_SBG'})
				
	print('	saving ...')
	ds_fluc.load().to_netcdf(path=name_budget,mode='w')  # here all // computations are triggered so this is the longest operation
	print('	done !')
	endtime = time.time()
	print('total time for budget building (s):',endtime-starttime)
	
def Build_flxbudget_file(path_in,dsB,dsO,dsmean,dsflx,Tstart,Tstop,name_budget,nhalo,window):
	"""This procedure is building a file with the terms of each budgets for differents fluxes.
	u'w',u'u',w'w',w'tht',tht'tht'
	
	INPUTS :
		- path_in	: Path of the OUTPUT files (to indicate origin of files in .nc)
		- dsB		: Dataset of at least 1 BACKUP file (for reference state RHOREFZ)
		- dsO		: Dataset of the OUTPUT files
		- dsmean	: Dataset of the mean fields (built with 'Build_mean_file')
		- dsflx		: Dataset of the averaged fluxes (built with 'Build_flx_file')
		- Tstart 	: Number of the file at which to start temporal average
		- Tstop 	: Number of the file at which to stop temporal average
		- name_budget: Output name of the netcdf produced
		- nhalo 	: HALO from MNH
		- window 	: Size of the window for MeanTurb (in points)
	OUTPUT :  
		A netcdf file with terms of the budget for turbulent fluxes as well as informations in the 
			netcdf attributes about the mean used as well as the origin of the files used.
	
	Notes : 
		- Only the resolved part is taken into account (because of the transport term and the pressure term 
			subgrid contribution not available in the code output).
		- No interscale is represented : in a resolved scale only budget, one should have a 
			term that transfer resolved TKE to subgrid TKE.
		- Equations from STULL 1988 chapter 4
	"""
	KB = 0 # this can be adjusted to 1 to match NAM_BUDGET outputs fields
	
	X = dsO.ni[nhalo:-nhalo-KB].values
	Y = dsO.nj[nhalo:-nhalo-KB].values
	Z = dsO.level[nhalo:-nhalo-KB].values
	
	print('	setting reference state ...')
	dsINI = xr.open_dataset('CAS06/INIT_CANAL_6.nc')
	P_ref = dsINI.PABST[0,nhalo:-nhalo-KB,nhalo:-nhalo-KB,nhalo:-nhalo-KB] 	# at mass point
	THT_ref = dsINI.THT[0,nhalo:-nhalo-KB,nhalo:-nhalo-KB,nhalo:-nhalo-KB] 	# at mass point
	RV_ref = dsINI.RVT[0,nhalo:-nhalo-KB,nhalo:-nhalo-KB,nhalo:-nhalo-KB] 	# at mass point
	THTv_ref = THT_ref*(1+Rv/Rd*RV_ref)/(1+RV_ref)
	rhoref = dsB.RHOREFZ.interp({'level_w':dsO.level})[0,nhalo:-nhalo-KB] # at w point interpolated at mass point
	
	print('	setting mean state ...')
	Um = dsmean.Um[:,:] # here the halo has already been removed (in procedure build_mean)
	Vm = dsmean.Vm[:,:]
	Wm = dsmean.Wm[:,:]
	THm = dsmean.THTm[:,:]
	RVm = dsmean.RVTm[:,:]
	THvm = dsmean.THTvm[:,:]
	Pm = dsmean.Pm[:,:]
	
	print('	setting mean fluxes...')
	flx_uw = dsflx.FLX_UW[:,:]
	flx_uv = dsflx.FLX_UV[:,:]
	flx_uu = dsflx.FLX_UU[:,:]
	flx_wtht = dsflx.FLX_THW[:,:]
	flx_ww = dsflx.FLX_WW[:,:]
	flx_rvw = dsflx.FLX_RvW[:,:]
	
	print('	allocating temporal series ...')
	THT = dsO.THT[:,nhalo:-nhalo-KB,nhalo:-nhalo-KB,nhalo:-nhalo-KB] 	# at mass point
	RVT = dsO.RVT[:,nhalo:-nhalo-KB,nhalo:-nhalo-KB,nhalo:-nhalo-KB] 	# at mass point
	P = dsO.PABST[:,nhalo:-nhalo-KB,nhalo:-nhalo-KB,nhalo:-nhalo-KB] 	# at mass point
	THTv = THT*(1+Rv/Rd*RVT)/(1+RVT)
	U = dsO.UT.interp({'ni_u':dsO.ni})[:,nhalo:-nhalo-KB,nhalo:-nhalo-KB,nhalo:-nhalo-KB] 		# grid : 2
	V = dsO.VT.interp({'nj_v':dsO.nj})[:,nhalo:-nhalo-KB,nhalo:-nhalo-KB,nhalo:-nhalo-KB] 		# grid : 3
	W = dsO.WT.interp({'level_w':dsO.level})[:,nhalo:-nhalo-KB,nhalo:-nhalo-KB,nhalo:-nhalo-KB] 	# grid : 4
	U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
	V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
	Um3D,Vm3D,Wm3D,THm3D,RVm3D,THvm3D,Pm3D,rhoref3D = Complete_dim_like([Um,Vm,Wm,THm,RVm,THvm,Pm,rhoref],U)
	
	print('	computing resolved fluctuations ...')
	u_fluc = (U - Um3D)
	v_fluc = (V - Vm3D)
	w_fluc = (W - Wm3D)
	tht_fluc = (THT - THm3D)
	#tht_fluc = (THT - THT_ref)
	rv_fluc = (RVT - RVm3D)
	#thtv_fluc = (THTv - THvm3D)
	thtv_fluc = (THTv - THTv_ref)
	#p_fluc = (P - Pm3D)
	p_fluc = (P - P_ref)
	
	print('	computing resolved triple correlations ...')
	uwu = u_fluc*w_fluc*u_fluc
	uwv = u_fluc*w_fluc*v_fluc
	uww = u_fluc*w_fluc*w_fluc
	
	print('	computing budgets terms :')
	print("		u'w'")
	uw_hadv = - Um*flx_uw.differentiate('ni')
	uw_vadv = - Wm*flx_uw.differentiate('level')
	uw_hDP = - ( flx_uu*Wm.differentiate('ni') + flx_uw*Um.differentiate('ni'))
	uw_vDP = - ( flx_uw*Wm.differentiate('level') + flx_ww*Um.differentiate('level') )
	uw_TR = - ( MeanTurb(uwu,Tstart,Tstop,window).differentiate('ni') + MeanTurb(uww,Tstart,Tstop,window).differentiate('level') )
	uw_BOY = g * MeanTurb(u_fluc*thtv_fluc/THTv_ref,Tstart,Tstop,window)
	uw_PRES = - MeanTurb( 1/rhoref3D*( u_fluc*p_fluc.differentiate('level') + w_fluc*p_fluc.differentiate('ni') ),Tstart,Tstop,window)
	print("		w'tht'")	
	wtht_hadv = - Um*flx_wtht.differentiate('ni')
	wtht_vadv = - Wm*flx_wtht.differentiate('level')
	wtht_hDP = -( MeanTurb(u_fluc*tht_fluc,Tstart,Tstop,window)*Wm.differentiate('ni') + flx_uw*THm.differentiate('ni') )
	wtht_vDP = -( flx_wtht*Wm.differentiate('level') + flx_ww*THm.differentiate('level') )
	wtht_TR = -(  MeanTurb(tht_fluc*w_fluc*u_fluc,Tstart,Tstop,window).differentiate('ni') +  MeanTurb(tht_fluc*w_fluc*w_fluc,Tstart,Tstop,window).differentiate('level'))
	wtht_TP = + g*MeanTurb(tht_fluc*thtv_fluc/THTv_ref,Tstart,Tstop,window)
	wtht_PRES = - MeanTurb( 1/rhoref3D*tht_fluc*p_fluc.differentiate('level'),Tstart,Tstop,window)
	print("		w'rv'")
	wrv_hadv = - Um*flx_rvw.differentiate('ni')
	wrv_vadv = - Wm*flx_rvw.differentiate('level')
	wrv_hDP = -( MeanTurb(u_fluc*rv_fluc,Tstart,Tstop,window)*Wm.differentiate('ni') + flx_uw*RVm.differentiate('ni') )
	wrv_vDP = -( flx_rvw*Wm.differentiate('level') + flx_ww*RVm.differentiate('level') )
	wrv_TR = -(  MeanTurb(rv_fluc*w_fluc*u_fluc,Tstart,Tstop,window).differentiate('ni') +  MeanTurb(rv_fluc*w_fluc*w_fluc,Tstart,Tstop,window).differentiate('level'))
	wrv_TP = + g*MeanTurb(rv_fluc*thtv_fluc/THTv_ref,Tstart,Tstop,window)
	wrv_PRES = - MeanTurb( 1/rhoref3D*rv_fluc*p_fluc.differentiate('level'),Tstart,Tstop,window)
	
	print('	building dataset')
	data_vars = {'uw_hadv':(['level','ni'],uw_hadv.data,{'long_name':'Horizontal advection of uw',
						'units':'m2 s-3',
						'grid location':'mass_center'}),
			'uw_vadv':(['level','ni'],uw_vadv.data,{'long_name':'Vertical advection of uw',
						'units':'m2 s-3',
						'grid location':'mass_center'}),
			'uw_vDP':(['level','ni'],uw_vDP.data,{'long_name':'Horizontal production of uw by vertical gradient',
						'units':'m2 s-3',
						'grid location':'mass_center'}),
			'uw_hDP':(['level','ni'],uw_hDP.data,{'long_name':"Vertical production of uw by horizontal gradient",
						'units':'m2 s-3',
						'grid location':'mass_center'}),
			'uw_TR':(['level','ni'],uw_TR.data,{'long_name':"Transport by turbulence of uw",
						'units':'m2 s-3',
						'grid location':'mass_center'}),
			'uw_TP':(['level','ni'],uw_BOY.data,{'long_name':"Buoyancy production/sink of uw",
						'units':'m2 s-3',
						'grid location':'mass_center'}),
			'uw_PRES':(['level','ni'],uw_PRES.data,{'long_name':'Pressure return-to-isotropy term of uw',
						'units':'m2 s-3',
						'grid location':'mass_center'}),
			'wtht_hadv':(['level','ni'],wtht_hadv.data,{'long_name':'Horizontal advection of wtht',
						'units':'K m s-2',
						'grid location':'mass_center'}),
			'wtht_vadv':(['level','ni'],wtht_vadv.data,{'long_name':'Vertical advection of wtht',
						'units':'K m s-2',
						'grid location':'mass_center'}),
			'wtht_hDP':(['level','ni'],wtht_hDP.data,{'long_name':'Horizontal production of wtht by vertical gradient',
						'units':'K m s-2',
						'grid location':'mass_center'}),
			'wtht_vDP':(['level','ni'],wtht_vDP.data,{'long_name':"Vertical production of wtht by horizontal gradient",
						'units':'K m s-2',
						'grid location':'mass_center'}),
			'wtht_TR':(['level','ni'],wtht_TR.data,{'long_name':"Transport by turbulence of wtht",
						'units':'K m s-2',
						'grid location':'mass_center'}),
			'wtht_TP':(['level','ni'],wtht_TP.data,{'long_name':"Buoyancy production/sink of wtht",
						'units':'K m s-2',
						'grid location':'mass_center'}),
			'wtht_PRES':(['level','ni'],wtht_PRES.data,{'long_name':'Pressure return-to-isotropy term of wtht',
						'units':'K m s-2',
						'grid location':'mass_center'}),
			'wrv_hadv':(['level','ni'],wrv_hadv.data,{'long_name':'Horizontal advection of wrv',
						'units':'kg kg-1 m s-2',
						'grid location':'mass_center'}),
			'wrv_vadv':(['level','ni'],wrv_vadv.data,{'long_name':'Vertical advection of wrv',
						'units':'kg kg-1 m s-2',
						'grid location':'mass_center'}),
			'wrv_hDP':(['level','ni'],wrv_hDP.data,{'long_name':'Horizontal production of wrv by vertical gradient',
						'units':'kg kg-1 m s-2',
						'grid location':'mass_center'}),
			'wrv_vDP':(['level','ni'],wrv_vDP.data,{'long_name':"Vertical production of wrv by horizontal gradient",
						'units':'kg kg-1 m s-2',
						'grid location':'mass_center'}),
			'wrv_TR':(['level','ni'],wrv_TR.data,{'long_name':"Transport by turbulence of wrv",
						'units':'kg kg-1 m s-2',
						'grid location':'mass_center'}),
			'wrv_TP':(['level','ni'],wrv_TP.data,{'long_name':"Buoyancy production/sink of wrv",
						'units':'kg kg-1 m s-2',
						'grid location':'mass_center'}),
			'wrv_PRES':(['level','ni'],wrv_PRES.data,{'long_name':'Pressure return-to-isotropy term of wrv',
						'units':'kg kg-1 m s-2',
						'grid location':'mass_center'})
			 }	 			 
	coords={'level': Z,'nj':Y,'ni':X}
	ds_flxbudget = xr.Dataset(data_vars=data_vars,coords=coords,
				attrs={'Tstart':Tstart,'Tstop':Tstop,'window':window,'files':path_in,'Note':'resolved budget only'})
	print('	saving ...')
	ds_flxbudget.to_netcdf(path=name_budget,mode='w')  # this is where most computations are done, in //
	print('	done !')
	
def Build_Quadrants_terms(path_in,dsO,dsmean,name_quadrant,nhalo):
	"""	This procedure decomposes some turbulent fluxes following a quadrant analysis:
		
		for a flux ab:
			- QI : a>0 b>0
			- QII: a<0 b>0
			- QIII: a<0 b<0
 			- QIV: a>0 b<0
 			
	 		a = A - <A>, b = B - <B>	
	 		-> in case of momentum flux uw, QII is ejection (updrafts) motion and QIV sweeps (downdrafts)
	 		-> in case of temperature, QI is ejection and QIII is sweeps
 		
 		INPUTS : 
	 		- path_in		: string of the path of dsO (for information)
	 		- dsO			: Dataset with 3D or 4D fields
	 		- dsmean		: DataSet with mean fields (see note)
	 		- name_quadrant	: name of the outpute file
	 		- nhalo			: MNH halo
 		OUTPUT :
 			A netcdf file with fluxes uw,wtht,wthtv,wrv decomposed into 4 quadrants
 			
 		Note : 
 			* dsO can be one or more BACKUP file(s) from MNH simulation
				OR one or more OUTPUT file(s) from MNH simulation
			* dsmean is a file containing mean field, either from hand building (with 'Build_mean_file')
				 or from a MNH diachronic file (opened with 'Opening_files', dsref variable)
			* This quadrant analysis is only considering resolved fluxes

	"""
	indt = -1 # last time step for diachronic mean file, could be better coded ...
	X = dsO.ni[nhalo:-nhalo].values
	Y = dsO.nj[nhalo:-nhalo].values
	Z = dsO.level[nhalo:-nhalo].values
	Time = dsO.time
	dicoAttrs = {'files':path_in,'Note':'resolved fluxes only'}
	# Instantaneous fields
	U = dsO.UT.interp({'ni_u':dsO.ni})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 		# grid : 2
	V = dsO.VT.interp({'nj_v':dsO.nj})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 		# grid : 3
	W = dsO.WT.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# grid : 4
	THT = dsO.THT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	RVT = dsO.RVT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	THTv = THT*(1+Rv/Rd*RVT)/(1+RVT)
	U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
	V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
	# Mean fields
	if '000' in dsmean: # MNH diachronic file
		dicoAttrs['Meanfile'] = 'By diachronic file'
		Um = dsmean['nomean']['Mean']['MEAN_U'][indt,:]
		Vm = dsmean['nomean']['Mean']['MEAN_V'][indt,:]
		Wm = dsmean['nomean']['Mean']['MEAN_W'][indt,:]
		RVTm = dsmean['nomean']['Mean']['MEAN_RV'][indt,:]
		THTm = dsmean['nomean']['Mean']['MEAN_TH'][indt,:]
		THTvm = dsmean['nomean']['Mean']['MEAN_THV'][indt,:]
		Um = Um.rename(new_name_or_name_dict={'level_les':'level'})
		Vm = Vm.rename(new_name_or_name_dict={'level_les':'level'})
		Wm = Wm.rename(new_name_or_name_dict={'level_les':'level'})
		RVTm = RVTm.rename(new_name_or_name_dict={'level_les':'level'})
		THTm = THTm.rename(new_name_or_name_dict={'level_les':'level'})
		THTvm = THTvm.rename(new_name_or_name_dict={'level_les':'level'})
	else: 				# hand build mean file
		dicoAttrs['Tstart'] = dsmean.attrs['Tstart']
		dicoAttrs['Tstop'] = dsmean.attrs['Tstop']
		dicoAttrs['window'] = dsmean.attrs['window']
		dicoAttrs['Meanfile'] = 'By Build_mean_file()'
		Um = dsmean.Um[:,:]
		Vm = dsmean.Vm[:,:]
		Wm = dsmean.Wm[:,:]
		THTm = dsmean.THTm[:,:]
		RVTm = dsmean.RVTm[:,:]
		THTvm = dsmean.THTvm[:,:]
	Um3D,Vm3D,Wm3D,THT3D,THTv3D,RVT3D = Complete_dim_like([Um,Vm,Wm,THTm,THTvm,RVTm],U) # to be added : THTvm
	# fluctuations
	u_fluc = (U - Um3D)
	w_fluc = (W - Wm3D)
	tht_fluc = (THT - THT3D)
	rv_fluc = (RVT - RVT3D)
	thtv_fluc = (THTv - THTv3D)
	u_pos = xr.where(u_fluc>0,u_fluc,0)
	w_pos = xr.where(w_fluc>0,w_fluc,0)
	u_neg = xr.where(u_fluc<0,u_fluc,0)
	w_neg = xr.where(w_fluc<0,w_fluc,0)
	tht_pos = xr.where(tht_fluc>0,tht_fluc,0)
	tht_neg = xr.where(tht_fluc<0,tht_fluc,0)
	thtv_pos = xr.where(thtv_fluc>0,thtv_fluc,0)
	thtv_neg = xr.where(thtv_fluc<0,thtv_fluc,0)
	rv_pos = xr.where(rv_fluc>0,rv_fluc,0)
	rv_neg = xr.where(rv_fluc<0,rv_fluc,0)
	
	Ones = np.ones(u_fluc.shape)
	Zeros = np.zeros(u_fluc.shape)
	
	uw_1 = u_pos*w_pos
	uw_2 = u_neg*w_pos
	uw_3 = u_neg*w_neg
	uw_4 = u_pos*w_neg
	
	# wtht_1 = w_pos*tht_pos
	# wtht_2 = w_pos*tht_neg
	# wtht_3 = w_neg*tht_neg
	# wtht_4 = w_neg*tht_pos
	
	# wthtv_1 = w_pos*thtv_pos
	# wthtv_2 = w_pos*thtv_neg
	# wthtv_3 = w_neg*thtv_neg
	# wthtv_4 = w_neg*thtv_pos
	
	# wrv_1 = w_pos*rv_pos
	# wrv_2 = w_pos*rv_neg
	# wrv_3 = w_neg*rv_neg
	# wrv_4 = w_neg*rv_pos
	
	data_vars = {'uw_1':(['time','level','nj','ni'],uw_1.data,{'long_name':'outward interaction of uw',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'uw_2':(['time','level','nj','ni'],uw_2.data,{'long_name':'ejection of uw',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'uw_3':(['time','level','nj','ni'],uw_3.data,{'long_name':'inward intercation of uw',
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			'uw_4':(['time','level','nj','ni'],uw_4.data,{'long_name':"sweeps of uw",
						'units':'m2 s-2',
						'grid location':'mass_center'}),
			# 'wtht_1':(['time','level','nj','ni'],wtht_1.data,{'long_name':'ejection of wtht',
			# 			'units':'K m s-1',
			# 			'grid location':'mass_center'}),
			# 'wtht_2':(['time','level','nj','ni'],wtht_2.data,{'long_name':'outward interaction of wtht',
			# 			'units':'K m s-1',
			# 			'grid location':'mass_center'}),
			# 'wtht_3':(['time','level','nj','ni'],wtht_3.data,{'long_name':'sweeps of wtht',
			# 			'units':'K m s-1',
			# 			'grid location':'mass_center'}),
			# 'wtht_4':(['time','level','nj','ni'],wtht_4.data,{'long_name':"inward interaction of wtht",
			# 			'units':'K m s-1',
			# 			'grid location':'mass_center'}),
			# 'wthtv_1':(['time','level','nj','ni'],wthtv_1.data,{'long_name':'ejection of wthtv',
			# 			'units':'K m s-1',
			# 			'grid location':'mass_center'}),
			# 'wthtv_2':(['time','level','nj','ni'],wthtv_2.data,{'long_name':'outward interaction of wthtv',
			# 			'units':'K m s-1',
			# 			'grid location':'mass_center'}),
			# 'wthtv_3':(['time','level','nj','ni'],wthtv_3.data,{'long_name':'sweeps of wthtv',
			# 			'units':'K m s-1',
			# 			'grid location':'mass_center'}),
			# 'wthtv_4':(['time','level','nj','ni'],wthtv_4.data,{'long_name':"inward interaction of wthtv",
			# 			'units':'K m s-1',
			# 			'grid location':'mass_center'}),
			# 'wrv_1':(['time','level','nj','ni'],wrv_1.data,{'long_name':'ejection of wrv',
			# 			'units':'kg/kg m s-1',
			# 			'grid location':'mass_center'}),
			# 'wrv_2':(['time','level','nj','ni'],wrv_2.data,{'long_name':'outward interaction of wrv',
			# 			'units':'kg/kg m s-1',
			# 			'grid location':'mass_center'}),
			# 'wrv_3':(['time','level','nj','ni'],wrv_3.data,{'long_name':'sweeps of wrv',
			# 			'units':'kg/kg m s-1',
			# 			'grid location':'mass_center'}),
			# 'wrv_4':(['time','level','nj','ni'],wrv_4.data,{'long_name':"inward interaction of wrv",
			# 			'units':'kg/kg m s-1',
			# 			'grid location':'mass_center'})
			 }	 
	# Note : no need for scalar flux but you can add them by un-commenting the few lines above
			 
	coords={'time':Time,'level': Z,'nj':Y,'ni':X}
	ds_flxbudget = xr.Dataset(data_vars=data_vars,coords=coords,
				attrs=dicoAttrs)
	print('	saving ...')
	ds_flxbudget.to_netcdf(path=name_quadrant,mode='w')  
	print('	done !')
	
def build_CS1(nhalo,data,data_mean,param_COND,L_TURB_COND,SVT_type,RV_DIFF,SEUIL_ML,indt,path_CS1):
	"""This procedure is building a file where coherent structures are masked.
		this is following the study of Brient et al 2023, Couvreux et al 2010.
		
		The tracer used here is the vapor mixing ratio or a passive tracer
		every files in the dictionnary 'data' is scanned.
		
		
		INTPUTS : 
			- nhalo		: MNH halo
			- data		: instantaneous files to apply CS on
			- data_mean	: mean fields (from diachronic file or manually averaged)
						note : diachronic file is opened with 'Opening_files' function
			- param_COND	:
				mCS			: if TURB_COND=C10, CS strength
				gammaRV		: if TURB_COND=C10, percentage of minimum integrated std rv'²
				gammaTurb1	: if TURB_COND=ITURB : minimum of turbulent intensity
				gammaTurb2	: if TURB_COND=ITURB2, percentage of minimum integrated turbulent intensity
				gammaEc		: if TURB_COND=EC, percentage of minimum integrated total TKE
			- L_TURB_COND	: liste of turbulent detection condition
			- SVT_type	: 'SVT' or 'RV' to choose tracer or RV
			- RV_DIFF	: how the condition on rv' is computed
			- SEUIL_ML	: threshold to detect mixed layer on tht
			- indt		: time index for reference sim
			- path_CS1	: path to save the files
			
		OUTPUT :
			A netcdf file with :
				* fields used in the averaging process (all member of ensemble mean)
					U,V,W,THT,THTV,RVT,ET,SV1,SV2,SV3,(SV4)
				* Mean fields associated expanded to match their dimensions 
				* mask(s) of the turbulent condition(s)
				* indicating functions of the differents objects
				* global indicating function gathering all objects
				
		Notes :
			- Main assumptions : 
				If the data file is only 1 time step, the mean is only spatial XY
				if the data file is more than 1 time step, the mean is time and spatial TY but not X.
	"""
	mCS,gammaRV = param_COND['C10']
	gammaTurb1 = param_COND['ITURB']
	gammaTurb2 = param_COND['ITURB2']
	mITURB3,gammaITURB3 = param_COND['ITURB3']
	gammaEc = param_COND['EC']
	
	for case in data.keys():
		if case in ['cold','warm','warm_wide'] or case[-3:]=='min':
			dim=['ni','nj']
		elif case[:2]=='S1':
			dim=['time','nj']
		else:
			dim=['time']
		
		for TURB_COND in L_TURB_COND:
			ds = data[case] # already opened
			# getting time info
			DimTime = ds.time
			dt = ds.time.values - np.datetime64('2000-01-01')
			Time0 = int(dt[0].item()/1e9)
			Time1 = int(dt[-1].item()/1e9)
			X = ds.ni[nhalo:-nhalo]
			Y = ds.nj[nhalo:-nhalo]
			Z = ds.level[nhalo:-nhalo]
			Z_w = ds.level_w[nhalo:-nhalo]
			
			if SVT_type=='RV':
				nSVT = SVT_type + RV_DIFF
			else:
				nSVT = SVT_type + 'MEAN'
			path_write = path_CS1+'CS1_'+case+'_'+TURB_COND+'_'+nSVT
			if TURB_COND=='C10':
				path_write = path_write + '_m'+str(mCS)+'.nc'
			else:
				path_write = path_write + '.nc'
			
			BOOL_HERE = pathlib.Path(path_write).is_file()
			BOOL_SAME = False
			print('		checking if '+case+' is here')
			print('			mean is ',dim)
			if BOOL_HERE:
				dstemp = xr.open_dataset(path_write)
				NmCS,NgammaRv = dstemp.attrs['mCS'],dstemp.attrs['gammaRv']
				NmITURB3,NgammaITURB3 = dstemp.attrs['mCS'],dstemp.attrs['gammaRv']
				NgammaTurb1,NgammaTurb2 = dstemp.attrs['gammaTurb1'],dstemp.attrs['gammaTurb2']
				NgammaEc = dstemp.attrs['gammaEc']
				if TURB_COND=='C10' and (NmCS,NgammaRv) == param_COND['C10']: 
					BOOL_SAME = True
				if TURB_COND=='ITURB' and NgammaTurb1 == param_COND['ITURB']:
					BOOL_SAME = True
				elif TURB_COND=='ITURB2' and NgammaTurb2 == param_COND['ITURB2']:
					BOOL_SAME = True
				elif TURB_COND=='ITURB3' and (NmITURB3,NgammaITURB3) == param_COND['ITURB3']:
					BOOL_SAME = True
				elif TURB_COND=='EC' and NgammaEc == param_COND['EC']:
					BOOL_SAME = True
				if not BOOL_SAME:
					os.remove(path_write) # overwrite if parameters are differents
			if (case=='cold' or case=='warm') and (mCS!=1):
				BOOL_HERE,BOOL_SAME=True,True # we dont need ref with other mCS
			if not BOOL_HERE or (BOOL_HERE and not BOOL_SAME):
				print('		--> case = '+case,'from t0=',Time0,'s to t1=',Time1,'s')
				# info about the parameters used
				print('		input: number of time stamps =',len(DimTime))
				print('		input: min turb detection =',TURB_COND)
				print("		input: how sv' is computed:",RV_DIFF)
				print('		input: mCS =',mCS)
				print('		input: gammaSV =',gammaRV*100,'%')
				# print('		input: minimum turb intensity 1 =',gammaTurb1*100,'%')
				# print('		input: minimum turb intensity 2 =',gammaTurb2*100,'%')
				# print('		input: minimum turb intensity 3 =',gammaITURB3*100,'%')
				# print('		input: coeff turb intensity 3 =',mITURB3)
				# print('		input: minimum Ec=',gammaEc*100,'%')
				print('		output file:',path_write)
				print('loading inst. fields')
				# Instantaneous fields
				U = ds.UT.interp({'ni_u':ds.ni})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# grid : 2
				V = ds.VT.interp({'nj_v':ds.nj})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# grid : 3
				W = ds.WT.interp({'level_w':ds.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] # grid : 4
				U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
				V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
				TKET = ds.TKET[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
				THT = ds.THT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
				RV = ds.RVT[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
				THTV = THT*(1+1.61*RV)/(1+RV)
				SBG_RV2 = xr.zeros_like(U)
				if 'R_HVAR' not in ds.keys() and SVT_type=='RV':
					print("WARNING for "+case+" : No rv'² in OUT files so standard variation for C10 is only resolved part")
				elif 'R_HVAR' in ds.keys():
					SBG_RV2 = (ds.R_HVAR + ds.RTOT_VVAR)[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
				SV1m,SV4m,SV3m = xr.zeros_like(U[0]),xr.zeros_like(U[0]),xr.zeros_like(U[0])
				SV1,SV4,SV3 = xr.zeros_like(U[0]),xr.zeros_like(U[0]),xr.zeros_like(U[0])
				if 'SVCS000' in ds.keys():
					SV1 = ds.SVCS000[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
					SV3 = ds.SVCS002[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
					if 'SVCS003' in ds.keys():
						SV4 = ds.SVCS003[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
				if 'SVT001' in ds.keys():
					SV1 = ds.SVT001[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
					SV3 = ds.SVT003[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
					if 'SVT004' in ds.keys():
						SV4 = ds.SVT004[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
				print('loading mean fields')
				# mean fields
				if '000' in data_mean[case]: # mnh diachronic file	
					Um = data_mean[case]['nomean']['Mean']['MEAN_U'][indt,:]
					Vm = data_mean[case]['nomean']['Mean']['MEAN_V'][indt,:]
					Wm = data_mean[case]['nomean']['Mean']['MEAN_W'][indt,:]
					RVm = data_mean[case]['nomean']['Mean']['MEAN_RV'][indt,:]
					THTm = data_mean[case]['nomean']['Mean']['MEAN_TH'][indt,:]
					THTVm = data_mean[case]['nomean']['Mean']['MEAN_THV'][indt,:]
					Em =  1/2*( (data_mean[case]['nomean']['Resolved']['RES_U2']+
							data_mean[case]['nomean']['Resolved']['RES_V2']+
							data_mean[case]['nomean']['Resolved']['RES_V2']+
							data_mean[case]['nomean']['Subgrid']['SBG_U2']+
							data_mean[case]['nomean']['Subgrid']['SBG_V2']+
							data_mean[case]['nomean']['Subgrid']['SBG_V2']) )[indt,:]
					RV2m = ( data_mean[case]['nomean']['Resolved']['RES_RT2'] + 
						data_mean[case]['nomean']['Subgrid']['SBG_RT2'])[indt,:]
					ABLH = data_mean[case]['nomean']['Misc']['BL_H'][indt]
					Um = Um.rename(new_name_or_name_dict={'level_les':'level'})
					Vm = Vm.rename(new_name_or_name_dict={'level_les':'level'})
					Wm = Wm.rename(new_name_or_name_dict={'level_les':'level'})
					RVm = RVm.rename(new_name_or_name_dict={'level_les':'level'})
					THTm = THTm.rename(new_name_or_name_dict={'level_les':'level'})
					THTVm = THTVm.rename(new_name_or_name_dict={'level_les':'level'})
					Em = Em.rename(new_name_or_name_dict={'level_les':'level'})
					RV2m = RV2m.rename(new_name_or_name_dict={'level_les':'level'})
					
					Um,Vm,Wm,RVm,THTm,THTVm,Em,RV2m = Complete_dim_like([Um,Vm,Wm,RVm,THTm,THTVm,Em,RV2m],U[0,:,:,:]) # Expanding profil to 3D
					ABLH = ABLH.expand_dims(dim={'level':Z,'nj':Y,'ni':X},axis=(0,1,2))
					SV1m = data_mean[case]['nomean']['Mean']['MEAN_SV'][0,indt,:]
					SV3m = data_mean[case]['nomean']['Mean']['MEAN_SV'][2,indt,:]
					if 'SVCS003' in ds.keys() or 'SVT004' in ds.keys():
						#SV4m = data_mean[case]['nomean']['Mean']['MEAN_SV'][3,indt,:]
						SV4 = SV4[0]
						SV4m = SV4.mean(dim=['ni','nj'])
					SV1m = SV1m.rename(new_name_or_name_dict={'level_les':'level'})
					SV3m = SV3m.rename(new_name_or_name_dict={'level_les':'level'})
					z1 = SV3m.idxmax(dim='level') 
					
					SV1m,SV4m,SV3m = Complete_dim_like([SV1m,SV4m,SV3m],U[0,:,:,:]) # Expanding profil to 3D
					E0 	= ds['RCONSW_FLX'].interp({'level_w':ds.level})[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
					Q0 	= ds['THW_FLX'][0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
					THT_z0 	= ds['THT'][0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
					RV_z0 	= ds['RVT'][0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
					THTv_z0	= THT_z0*(1+1.61*RV_z0)/(1+RV_z0)
					Qv0	= THTv_z0/THT_z0*Q0+0.61*THT_z0*E0
					UW_FLX = (ds['UW_HFLX'].interp({'level_w':ds.level,'ni_u':ds.ni}).values + 
							ds['UW_VFLX'].interp({'level_w':ds.level}).values)[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean()
					VW_FLX = (ds['VW_HFLX'].interp({'level_w':ds.level,'nj_v':ds.nj}).values + 
							ds['VW_VFLX'].interp({'level_w':ds.level}).values)[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean()
					u_star	= (UW_FLX**2 + VW_FLX**2)**(1/4)
					U,V,W,TKET,THT,THTV,RV,SBG_RV2 = U[0],V[0],W[0],TKET[0],THT[0],THTV[0],RV[0],SBG_RV2[0] # 3D fields, not 4D
					SV1,SV3 = SV1[0],SV3[0]
					dicoSurfDim = []
					dicoDim = ['level','nj','ni']
				else: # mean file is a manually averaged file
					Um = data_mean[case].Um
					Vm = data_mean[case].Vm
					Wm = data_mean[case].Wm
					RVm = data_mean[case].RVTm
					THTm = data_mean[case].THTm
					THTVm = data_mean[case].THTvm
					Em = data_mean[case].ETm
					if 'SV1m' in data_mean[case].keys():
						SV1m = data_mean[case].SV1m
						SV4m = data_mean[case].SV4m
						SV3m = data_mean[case].SV3m
					RV2m = ((RV - RVm)**2 + SBG_RV2).mean(dim=['time','nj']) # t and Y average
					gTHT = THTVm[:,:].differentiate('level')					
					ABLH = Z.isel(level=gTHT.argmax(dim='level')).rename('ABLH')
					Um,Vm,Wm,RVm,THTm,THTVm,Em,RV2m,ABLH,SV1m,SV3m,SV4m = Complete_dim_like([Um,Vm,Wm,RVm,THTm,THTVm,Em,RV2m,ABLH,SV1m,SV3m,SV4m],U)
					E0 	= data_mean[case].E0.values
					Q0 	= data_mean[case].Q_star.values
					Qv0 	= data_mean[case].Qv_star.values
					u_star	= data_mean[case].u_star.values
					dicoSurfDim = ['ni']
					dicoDim = ['time','level','nj','ni']
					
				if SVT_type=='RV' and (RV_DIFF=='MIDDLE_MIXED' or RV_DIFF=='MIXED'):
					print('mixed layer indexes')
					if '000' in data_mean[case]:
						gTHT = THTVm[:,0,0].differentiate('level')
						indz1,indz2 = get_mixed_layer_indexes(Z,gTHT,SEUIL_ML)
						RVmixed = RVm.isel(level=slice(indz1,indz2+1)).mean(dim='level')
						indcrit = int(( indz1 + indz2 ) / 2)
						rv_crit = RVm.isel(level=indcrit).mean() # rv middle of mixed layer
					else:
						gTHT = THTVm[0,:,0,:].differentiate('level')
#						RVmixed = xr.zeros_like(X)
#						rv_crit = xr.zeros_like(X)
#						for i in range(len(X)): # this cannot be // as the mixed layer changes with X
#							print(i)
#							indz1,indz2 = get_mixed_layer_indexes(Z,gTHT[:,i],SEUIL_ML)
#							RVmixed[i] = RVm.isel(time=0,level=slice(indz1,indz2),nj=0,ni=i).mean(dim='level')
#							indcrit = int(( indz1 + indz2 ) / 2)
#							rv_crit[i] = RVm.isel(time=0,level=indcrit,nj=0,ni=i) # rv middle of mixed layer
						indz1,indz2 = get_mixed_layer_indexes(Z,gTHT[:,:].mean(dim='ni'),SEUIL_ML)
						indcrit = int(( indz1 + indz2 ) / 2)
						#RVmixed = RVm.mean(dim='ni').isel(time=0,level=slice(indz1,indz2+1),nj=0).mean(dim='level')
						RVmixed = ( RVm.mean(dim='ni').isel(level=slice(indz1,indz2+1),time=0,nj=0).integrate('level')/ (Z[indz2]-Z[indz1]) ) # here only 1 value for every X. 
						rv_crit = RVm.mean(dim='ni').isel(time=0,level=indcrit,nj=0)
						
						# Note: 
						#	if RVmixed is computed with RVm.mean(dim='ni'), then the mixed layer value of rv is unique for the whole domain and is not evolving with X
						#		this is used to remove the loop on ni to compute a rvmixed at every X.
						#	In fact, using a rvmixed at every X might not even have any sense at all bc we want to detect coherent structures that are advected and thus could change categories
						#		during the advection time ...
						#
						#	-> 1 rvmixed for the whole X domain is an error, but at least the category error is the same for every altitude for each structure
						
						
					RVmixed,rv_crit = Complete_dim_like([RVmixed,rv_crit],U)
				
				print('fluctuations')				
				u_fluc, v_fluc, w_fluc 		= U - Um, V - Vm, W - Wm
				tht_fluc, thtv_fluc 		= THT - THTm, THTV - THTVm
				E 	= 1/2*(u_fluc**2+v_fluc**2+w_fluc**2) + TKET
				RV2 = (RV - RVm)**2 + SBG_RV2				
				sv1_fluc,sv3_fluc,sv4_fluc 	= SV1 - SV1m, SV3 - SV3m, SV4 - SV4m
					
				if RV_DIFF=='MIXED':
					rv_fluc = RV - RVmixed
				elif RV_DIFF=='MIDDLE_MIXED':
					rv_fluc = RV - rv_crit
				elif RV_DIFF=='MEAN':
					rv_fluc = RV - RVm
					
				print('CS1')
				# Conditional sampling 1
				if SVT_type=='RV':
					BOOL_turb = Compute_bool_turb(dim,TURB_COND,param_COND,X,Y,Z,ABLH,
							RV,RVm,U,V,W,Um,Vm,Wm,E,Em)
					BOOL_turb1,BOOL_turb3 = BOOL_turb,BOOL_turb
					# 	updrafts 			: |rv_fluc| > max(RV2,rv_min) AND rv_fluc > 0 AND w_fluc > 0 (moist and upward motion)
					# 	subsiding shells 	: |rv_fluc| > max(RV2,rv_min) AND rv_fluc > 0 AND  w_fluc < 0 (moist and downward motion)
					# 	downdrafts 			: |rv_fluc| > max(RV2,rv_min) AND rv_fluc < 0 AND w_fluc < 0 (dry and downward motion)
					#	environnement 		: |rv_fluc| > max(RV2,rv_min) AND rv_fluc < 0 AND w_fluc > 0 (dry and upward motion)
					# 	others : NOT (updrafts AND sub.shell AND downdrafts)
					BOOL_up = np.logical_and(np.logical_and(BOOL_turb,rv_fluc > 0),w_fluc > 0)
					BOOL_sub = np.logical_and(np.logical_and(BOOL_turb,rv_fluc > 0),w_fluc < 0)
					BOOL_down = np.logical_and(np.logical_and(BOOL_turb,rv_fluc < 0),w_fluc < 0)
					BOOL_env = np.logical_and(np.logical_and(BOOL_turb	,rv_fluc < 0),w_fluc > 0)
				elif SVT_type=='SVT':	
					RV_DIFF = 'MEAN'					
					BOOL_turb1 = Compute_bool_turb(dim,TURB_COND,param_COND,X,Y,Z,ABLH,
							SV1,SV1m,U,V,W,Um,Vm,Wm,E,Em)
					BOOL_turb3 = Compute_bool_turb(dim,TURB_COND,param_COND,X,Y,Z,ABLH,
							SV3,SV3m,U,V,W,Um,Vm,Wm,E,Em)
					BOOL_turb = np.logical_or(BOOL_turb1,BOOL_turb3)
					if case[:2]=='S1':
						BOOL_turb4 = Compute_bool_turb(dim,TURB_COND,param_COND,X,Y,Z,ABLH,
							SV4,SV4m,U,V,W,Um,Vm,Wm,E,Em)
						BOOL_turb = np.logical_or(BOOL_turb,BOOL_turb4)
						BOOL_up2 = np.logical_and(np.logical_and(BOOL_turb4,sv4_fluc>0),w_fluc > 0)
						BOOL_sub2 = np.logical_and(np.logical_and(BOOL_turb4,sv4_fluc>0),w_fluc < 0)
					# 	updrafts 			: BOOL_turb1  AND s1'>0 AND w_fluc > 0 (surface tracer and upward motion)
					# 	subsiding shells 	: BOOL_turb1  AND s1'>0 AND w_fluc < 0 (surface tracer and downward motion)
					# 	downdrafts 			: BOOL_turb3  AND s3'>0 AND w_fluc < 0 (topABL tracer and downward motion)
					#	environnement 		:  BOOL_turb3 AND s3'>0 AND w_fluc > 0 (topABL tracer and upward motion)
					# 	others : (NOT BOOL_turb1) AND (NOT BOOL_turb3) AND (NOT (BOOL_turb1 U BOOL_turb3))				
					BOOL_up = np.logical_and(np.logical_and(BOOL_turb1,sv1_fluc>0),w_fluc > 0)
					BOOL_sub = np.logical_and(np.logical_and(BOOL_turb1,sv1_fluc>0),w_fluc < 0)
					BOOL_down = np.logical_and(np.logical_and(BOOL_turb3,sv3_fluc>0),w_fluc < 0)
					BOOL_env = np.logical_and(np.logical_and(BOOL_turb3,sv3_fluc>0),w_fluc > 0)			
					
					
					
				# mix of all structures for file save
				#	if SVT_type=='RV' : there is no overlap thanks to mutually exclusive CS
				# 	if SV4 and SV1 are both present, overlap can happen. So most concentrated one
				#		if used in this case.
				if case[:2]=='S1' and SVT_type=='SVT':				
					global_mask = xr.where( np.logical_and(BOOL_up,SV1>SV4) ,1,0)			# up=1
					global_mask = xr.where( np.logical_and(BOOL_sub,SV1>SV4),2,global_mask) # sub=2
					global_mask = xr.where( np.logical_and(BOOL_up2,SV1<SV4) ,5,global_mask)# up2=5
					global_mask = xr.where( np.logical_and(BOOL_sub2,SV1<SV4),6,global_mask)# sub2=6
					global_mask = xr.where(BOOL_down,3,global_mask) # down=3
					global_mask = xr.where(BOOL_env,4,global_mask)  # env=4
					
				else:
					global_mask = xr.where(BOOL_up,1,0) 		# up=1
					global_mask = xr.where(BOOL_sub,2,global_mask)  # sub=2
					global_mask = xr.where(BOOL_down,3,global_mask) # down=3
					global_mask = xr.where(BOOL_env,4,global_mask)  # env=4
				print('		shape of U',U.shape)

				# data_vars = {'UT':(dicoDim,U.data,{'long_name':'UT',
				# 					'units':'m s-1',
				# 					'grid location':'mass_center'}),
				# 		'VT':(dicoDim,V.data,{'long_name':'VT',
				# 					'units':'m s-1',
				# 					'grid location':'mass_center'}),
				# 		'WT':(dicoDim,W.data,{'long_name':'WT',
				# 					'units':'m s-1',
				# 					'grid location':'mass_center'}),
				# 		'THT':(dicoDim,THT.data,{'long_name':'THT',
				# 					'units':'K',
				# 					'grid location':'mass_center'}),
				# 		'THTV':(dicoDim,THTV.data,{'long_name':'THTV',
				# 					'units':'K',
				# 					'grid location':'mass_center'}),
				# 		'RVT':(dicoDim,RV.data,{'long_name':'RVT',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'UTm':(dicoDim,Um.data,{'long_name':'Horizontal mean U in 3D',
				# 					'units':'m s-1',
				# 					'grid location':'mass_center'}),
				# 		'VTm':(dicoDim,Vm.data,{'long_name':'Horizontal mean V in 3D',
				# 					'units':'m s-1',
				# 					'grid location':'mass_center'}),
				# 		'WTm':(dicoDim,Wm.data,{'long_name':'Horizontal mean W in 3D',
				# 					'units':'m s-1',
				# 					'grid location':'mass_center'}),
				# 		'THTm':(dicoDim,THTm.data,{'long_name':'Horizontal mean THT in 3D',
				# 					'units':'K',
				# 					'grid location':'mass_center'}),
				# 		'THTVm':(dicoDim,THTVm.data,{'long_name':'Horizontal mean THTV in 3D',
				# 					'units':'K',
				# 					'grid location':'mass_center'}),
				# 		'RVTm':(dicoDim,RVm.data,{'long_name':'Horizontal mean RVT in 3D',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'E':(dicoDim,E.data,{'long_name':'Total turbulent kinetic energ',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'Em':(dicoDim,Em.data,{'long_name':'Total horizontal mean turbulent kinetic energy in 3D',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'SV1':(dicoDim,SV1.data,{'long_name':'Passive scalar 1 (surface)',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'SV1m':(dicoDim,SV1m.data,{'long_name':'Mean Passive scalar 1 in 3D',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'SV4':(dicoDim,SV4.data,{'long_name':'Passive scalar 2 (surface)',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'SV4m':(dicoDim,SV4m.data,{'long_name':'Mean Passive scalar 2 in 3D',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'SV3':(dicoDim,SV3.data,{'long_name':'Passive scalar 3 (top ABL)',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'SV3m':(dicoDim,SV3m.data,{'long_name':'Mean Passive scalar 3 in 3D',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'is_turb':(dicoDim,xr.where(BOOL_turb,1,0).data,{'long_name':'mask for turbulent motion only',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'is_turb1':(dicoDim,xr.where(BOOL_turb1,1,0).data,{'long_name':'mask for turbulent motion only (SVT1)',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'is_turb3':(dicoDim,xr.where(BOOL_turb3,1,0).data,{'long_name':'mask for turbulent motion only (SVT3)',
				# 					'units':'kg/kg',
				# 					'grid location':'mass_center'}),
				# 		'is_up':(dicoDim,xr.where(BOOL_up,1,0).data,{'long_name':'mask for updrafts',
				# 					'units':'',
				# 					'grid location':'mass_center'}),
				# 		'is_sub':(dicoDim,xr.where(BOOL_sub,1,0).data,{'long_name':'mask for subsiding shells',
				# 					'units':'',
				# 					'grid location':'mass_center'}),
				# 		'is_down':(dicoDim,xr.where(BOOL_down,1,0).data,{'long_name':'mask for downdrafts',
				# 					'units':'',
				# 					'grid location':'mass_center'}),
				# 		'is_env':(dicoDim,xr.where(BOOL_env,1,0).data,{'long_name':'mask for environment',
				# 					'units':'',
				# 					'grid location':'mass_center'}),
				# 		'global_objects':(dicoDim,global_mask.data,{'long_name':'All identified structures',
				# 					'units':'',
				# 					'grid location':'mass_center'}),
				# 		'E0':(dicoSurfDim,E0,{'long_name':'Surface latent heat flux',
				# 					'units':'kg/kg m.s-1',
				# 					'grid location':'mass_center'}),
				# 		'Q0':(dicoSurfDim,Q0,{'long_name':'Surface sensible heat flux',
				# 					'units':'K.m.s-1',
				# 					'grid location':'mass_center'}),
				# 		'Qv0':(dicoSurfDim,Qv0,{'long_name':'Surface buoyancy flux',
				# 					'units':'K.m.s-1',
				# 					'grid location':'mass_center'}),
				# 		'u_star':(dicoSurfDim,u_star,{'long_name':'Surface friction velocity',
				# 					'units':'m.s-1',
				# 					'grid location':'mass_center'}),
				# 		 }	
				# if (case=='S1' or case=='S1decayHalf') and SVT_type=='SVT':
				# 	data_vars['is_turb4'] = (dicoDim,xr.where(BOOL_turb4,1,0).data,{'long_name':'mask for turbulent motion only (SVT4)',
				# 					'units':'',
				# 					'grid location':'mass_center'})
				# 	data_vars['is_up2'] = (dicoDim,xr.where(BOOL_up2,1,0).data,{'long_name':'mask for updrafts sv4',
				# 				'units':'',
				# 				'grid location':'mass_center'})
				# 	data_vars['is_sub2'] = (dicoDim,xr.where(BOOL_sub2,1,0).data,{'long_name':'mask for subsiding shells sv4',
				# 				'units':'',
				# 				'grid location':'mass_center'})

				data_vars = {'global_objects':(dicoDim,global_mask.data,{'long_name':'All identified structures',
									'units':'',
									'grid location':'mass_center'}),
							'UT':(dicoDim,U.data,{'long_name':'UT',
									'units':'m s-1',
				 					'grid location':'mass_center'}),
							'WT':(dicoDim,W.data,{'long_name':'WT',
				 					'units':'m s-1',
				 					'grid location':'mass_center'}),
							'THT':(dicoDim,THT.data,{'long_name':'THT',
				 					'units':'K',
									'grid location':'mass_center'}),
							'RVT':(dicoDim,RV.data,{'long_name':'RVT',
									'units':'kg/kg',
				 					'grid location':'mass_center'}),
						 }	
				if TURB_COND=='C10':
					NmCS,NgammaRv,NgammaTurb1,NgammaTurb2,NgammaEc,NmITURB3,NgammaITURB3 = mCS,gammaRV,-99,-99,-99,-99,-99
				elif TURB_COND=='ITURB':
					NmCS,NgammaRv,NgammaTurb1,NgammaTurb2,NgammaEc,NmITURB3,NgammaITURB3 = -99,-99,gammaTurb1,-99,-99,-99,-99
				elif TURB_COND=='ITURB2':
					NmCS,NgammaRv,NgammaTurb1,NgammaTurb2,NgammaEc,NmITURB3,NgammaITURB3 = -99,-99,-99,gammaTurb2,-99,-99,-99
				elif TURB_COND=='EC':			
					NmCS,NgammaRv,NgammaTurb1,NgammaTurb2,NgammaEc,NmITURB3,NgammaITURB3 = -99,-99,-99,-99,gammaEc,-99,-99
				elif TURB_COND=='ITURB3':
					NmCS,NgammaRv,NgammaTurb1,NgammaTurb2,NgammaEc,NmITURB3,NgammaITURB3 = -99,-99,-99,-99,-99,mITURB3,gammaITURB3
				coords={'time':DimTime,'level': Z,'nj':Y,'ni':X}
				ds_CS1 = xr.Dataset(data_vars=data_vars,coords=coords,
							attrs={'Input file':case,'Time (s)':str([Time0,Time1]),'Turb_cond':TURB_COND,'Note':'resolved fluctuations only',
								'mCS':NmCS,'gammaRv':NgammaRv,'gammaTurb1':NgammaTurb1,'gammaTurb2':NgammaTurb2,'gammaEc':NgammaEc,
								'mITURB3':NmITURB3,'gammaITURB3':NgammaITURB3,
								'RV_DIFF':RV_DIFF})
				print('	saving ...')
				ds_CS1.to_netcdf(path=path_write,mode='w')  
				ds_CS1.close()
				print('	done !')
	

