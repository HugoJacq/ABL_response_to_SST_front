# To be used with analyse.py 
import xarray as xr
import pandas as pd
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy import signal
import matplotlib as mpl
from matplotlib import colors as c
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import Locator,MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import uniform_filter1d,gaussian_filter
# my modules
from module_cst import *
from module_tools import *
from module_CS_func import *
from module_building_files import *
			
def plot_quadrants(X,Y,Z,X1,L_plots,L_dsmean,L_ds,L_dsquadrant,L_string_var,L_filter,L_paramSlice,path_save,dpi):
	"""Quadrant analysis of turbulent flux a'b'.
		Every files from L_dsquadrant is scanned and plots are files specifics (no comparison)
		Input simulations are : 
			* CAS06W_SVT (canal sim with front, S1)
			* CAS09_SVT (canal ref sim cold)
			* CAS10_SVT (canal ref sim warm)
		 	
		 	INPUTS : 
		 	- X 			: DataArray containing X dimension
		 	- Y  			: DataArray containing Y dimension
		 	- Z  			: DataArray containing Z dimension
		 	- X1 			: X location (in km) to plot profiles of S1 simulation
		 	- L_plots 		: list of chosen plots ('flx_split','cover','intensity','efficiency')
		 	- L_dsmean 		: dic of mean files with name of simu as key
		 	- L_ds 			: dic of instantaneous file with name of simu as key
		 	- L_dsquadrant 	: dic of quadrant splitting file with name of simu as key
		 	- L_string_var 	: list of chosen flux to visualize
		 	- L_filter 		: dic float to specify the coeff to multiply 
		 					the normalisation value to filter quadrants
		 	- L_paramSlice : contains coordinates for slices
		 			if 'XY', height (m) where to slice, X1 and X2 the start/end for plot in km, 
		 						stepx and stepy skipped cell for arrow plot
		 			if 'XZ', TBD
		 			if 'YZ', TBD
		 	- path_save		: where to save the plots
		 	- dpi
		 	
		 Based on "On the Nature of the Transition Between Roll and
		 	Cellular Organization in the Convective Boundary Layer" Salesky 2016
	"""
	indX1 = np.argmin(np.abs(X.values-X1*1000)) # a passer en arg
	indt = -1 # for diachronic mean file, latest instant
	indX_norm = np.argmin(np.abs(X.values-4000)) # for S1, normalization is from values at X=4km
	n_ab = {}		# this is for efficiency plot
	dic_ABLH = {} 	# this is for efficiency plot
	dic_norm = {}	# this is for normalization 
	dz = np.zeros(Z.shape) # dz is for contour plot
	dz[:-1] = Z[1:].values - Z[:-1].values
	dz[-1] = dz[-2]
	for string_var in L_string_var:
		for case in L_ds.keys():
			print(case)
			dic_norm[case] = {}
			# -> Choosing the right variable
			if string_var=='uw':
				nicename = 'uw'
				varname = 'u'
				Coeff = 1
				title_profile_quadrants = r"$\overline{u'w'}$/$u^{*2}$"
				title_efficiency = r" u'w' / $(u'w'_{II} + u'w'_{IV}$)"
				ylim = [0,1.2]
				xlim = [-1.5,0.4]
			elif string_var=='wrv':
				nicename = r'wr_v'
				varname = 'rv'
				Coeff = 1000 # kg/kg -> g/kg
				title_profile_quadrants = r"$\overline{w'r_v'}$/$\overline{w'r_v'}_s$"
				title_efficiency = r" w'r_v' / $(w'r_v'_{I} + w'r_v'_{III}$)"
				ylim = [0,1.2]
			elif string_var=='wtht':
				nicename = r'w$\theta'
				varname = 'tht'
				Coeff = 1
				title_profile_quadrants = r"$\overline{w'\theta'}$/$\overline{w'\theta'}_s$"
				title_efficiency = r" w'$\theta$' / $(w'\theta'_{I} + w'\theta'_{III}$)"
				ylim = [0,1.2]
			elif string_var=='wthtv':
				varname = 'thtv'
				nicename = r'w$\theta_v$'
				Coeff = 1
				title_profile_quadrants = r"$\overline{w'\theta_v'}$/$\overline{w'\theta_v'}_s$"
				title_efficiency = r" w'$\theta_v$' / $(w'\theta_v'_{I} + w'\theta_v'_{III}$)"
				ylim = [0,1.2]
				if case=='cold' or case=='warm':
					xlim = [-0.4,1]
				elif case=='S1':
					xlim = [-1,3.1]
			else:
				raise Exception('string_var='+string_var+' is not available')
				
			# instantatenous fields for slice plots
			U = L_ds[case].UT.interp({'ni_u':L_ds[case].ni})[indt,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# grid : 2
			V = L_ds[case].VT.interp({'nj_v':L_ds[case].nj})[indt,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# grid : 3
			U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
			V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
			W = L_ds[case].WT.interp({'level_w':L_ds[case].level})[indt,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
			THT = L_ds[case].THT[indt,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
			RVT = L_ds[case].RVT[indt,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
			THTV = THT*(1+1.61*RVT)/(1+RVT)
			# -> Quadrants decomposition
			ab_1 = L_dsquadrant[case][string_var+'_1']*Coeff
			ab_2 = L_dsquadrant[case][string_var+'_2']*Coeff
			ab_3 = L_dsquadrant[case][string_var+'_3']*Coeff
			ab_4 = L_dsquadrant[case][string_var+'_4']*Coeff
			ab = ab_1 + ab_2 + ab_3 + ab_4
			
			# -> Normalization values
			if '000' in L_dsmean[case]: # MNH diachronic file
				ABLHm = L_dsmean[case]['nomean']['Misc']['BL_H'][indt].values
				Um = L_dsmean[case]['nomean']['Mean']['MEAN_U'][indt]
				Vm = L_dsmean[case]['nomean']['Mean']['MEAN_V'][indt]
				THTVm = L_dsmean[case]['nomean']['Mean']['MEAN_THV'][indt,:]
				u_star = L_dsmean[case]['nomean']['Surface']['Ustar'][indt]
				Um = Um.rename(new_name_or_name_dict={'level_les':'level'})
				Vm = Vm.rename(new_name_or_name_dict={'level_les':'level'})
				THTVm = THTVm.rename(new_name_or_name_dict={'level_les':'level'})
				E0 	= L_ds[case]['RCONSW_FLX'].interp({'level_w':L_ds[case].level})[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
				Q0	= L_ds[case]['THW_FLX'][0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
				THT_z0 	= L_ds[case]['THT'][0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
				RV_z0 	= L_ds[case]['RVT'][0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
				THTv_z0	= THT_z0*(1+1.61*RV_z0)/(1+RV_z0)
				Qv0	= THTv_z0/THT_z0*Q0+0.61*THT_z0*E0
				title=case
			else: # hand built mean file
				THTVm = L_dsmean[case].THTvm[:,:]
				Um = L_dsmean[case].Um[:,:]
				Vm = L_dsmean[case].Vm[:,:]
				gTHT = THTVm[:,:].differentiate('level')
				ABLH = Z[gTHT.argmax(dim='level')]
				ABLHm = ABLH.mean().values
				u_star = L_dsmean[case].u_star[indX_norm]
				E0 	= L_dsmean[case].E0[indX_norm]
				Q0 	= L_dsmean[case].Q_star[indX_norm]
				Qv0	= L_dsmean[case].Qv_star[indX_norm]
				title=case+' at X='+str(X1)+'km'
			indz = np.argmin(np.abs(Z.values-ABLHm/2))		# W at ABLH/2
			indzi = np.argmin(np.abs(Z.values-1.1*ABLHm))	# for vertical integ up to 1.1zi
			Um,Vm,THTVm = Complete_dim_like([Um,Vm,THTVm],U)
			dic_ABLH[case] = ABLHm
			dic_norm[case]['uw'] = u_star**2
			dic_norm[case]['uw'].name = 'u*2'
			dic_norm[case]['wrv'] = E0*1000
			dic_norm[case]['wtht'] = Q0
			dic_norm[case]['wthtv'] = Qv0
			
			# if the quadrants are filtered,
			#	values outside the 'hole' are rejected
			if np.abs(L_filter[string_var])>0.01: 
				seuil = L_filter[string_var]*dic_norm[case][string_var]
				CFILTER = '_FILTERED'
				CFILTER_title = ', filtered with '+str(L_filter[string_var])+dic_norm[case][string_var].name
				if string_var=='uw':
					ab_1 = ab_1.where(ab_1 > - seuil,0)
					ab_2 = ab_2.where(ab_2 < seuil,0)
					ab_3 = ab_3.where(ab_3 > - seuil,0)
					ab_4 = ab_4.where(ab_4 < seuil,0)
				elif string_var in ['wtht','wrv','wthtv']:
					ab_1 = ab_1.where(ab_1 > seuil,0)
					ab_2 = ab_2.where(ab_2 < - seuil,0)
					ab_3 = ab_3.where(ab_3 > seuil,0)
					ab_4 = ab_4.where(ab_4 < - seuil,0)
			else:
				CFILTER = ''	
				CFILTER_title = ''	

			# -> Flux decomposition,
			# 	and fraction of area covered by every quadrants
			if '000' in L_dsmean[case]: # MNH diachronic file
				# mean is X and Y, only 1 instant 
				ab_1_p = ab_1[0,:,:,:].mean(dim={'ni','nj'})
				ab_2_p = ab_2[0,:,:,:].mean(dim={'ni','nj'})
				ab_3_p = ab_3[0,:,:,:].mean(dim={'ni','nj'})
				ab_4_p = ab_4[0,:,:,:].mean(dim={'ni','nj'})
				ab_p   = ab[0,:,:,:].mean(dim={'ni','nj'})
				F1 = xr.where(np.abs(ab_1[0,:,:,:])>0,1,0).mean(dim={'ni','nj'})
				F2 = xr.where(np.abs(ab_2[0,:,:,:])>0,1,0).mean(dim={'ni','nj'})
				F3 = xr.where(np.abs(ab_3[0,:,:,:])>0,1,0).mean(dim={'ni','nj'})
				F4 = xr.where(np.abs(ab_4[0,:,:,:])>0,1,0).mean(dim={'ni','nj'})
			else: # hand built mean file
				# mean is Time and Y
				ab_1_p = ab_1[:,:,:,indX1].mean(dim={'time','nj'})
				ab_2_p = ab_2[:,:,:,indX1].mean(dim={'time','nj'})
				ab_3_p = ab_3[:,:,:,indX1].mean(dim={'time','nj'})
				ab_4_p = ab_4[:,:,:,indX1].mean(dim={'time','nj'})
				ab_p   = ab[:,:,:,indX1].mean(dim={'time','nj'})
				F1 = xr.where(np.abs(ab_1[:,:,:,indX1])>0,1,0).mean(dim={'time','nj'})
				F2 = xr.where(np.abs(ab_2[:,:,:,indX1])>0,1,0).mean(dim={'time','nj'})
				F3 = xr.where(np.abs(ab_3[:,:,:,indX1])>0,1,0).mean(dim={'time','nj'})
				F4 = xr.where(np.abs(ab_4[:,:,:,indX1])>0,1,0).mean(dim={'time','nj'})
					
			# Intensity of each quadrant
			S1m = np.fabs(ab_1_p) / ( np.fabs(ab_1_p) + np.fabs(ab_2_p) + np.fabs(ab_3_p) + np.fabs(ab_4_p) )
			S2m = np.fabs(ab_2_p) / ( np.fabs(ab_1_p) + np.fabs(ab_2_p) + np.fabs(ab_3_p) + np.fabs(ab_4_p) )
			S3m = np.fabs(ab_3_p) / ( np.fabs(ab_1_p) + np.fabs(ab_2_p) + np.fabs(ab_3_p) + np.fabs(ab_4_p) )
			S4m = np.fabs(ab_4_p) / ( np.fabs(ab_1_p) + np.fabs(ab_2_p) + np.fabs(ab_3_p) + np.fabs(ab_4_p) )
			
			# efficiency is different is momentum or scalar
			if string_var=='uw':
				n_ab[case] = ab_p / (ab_2_p+ab_4_p)
				c1,c2,c3,c4 = "blue",'red','black','green'
				updrafts,downdrafts = ab_2[indt,:,:,:],ab_4[indt,:,:,:] # for XY plots
			elif string_var in ['wtht','wrv','wthtv']:
				n_ab[case] = ab_p / (ab_1_p+ab_3_p)
				c1,c2,c3,c4 = 'red','black','green','blue'
				updrafts,downdrafts = ab_1[indt,:,:,:],ab_3[indt,:,:,:] # for XY plots
		
			# -> PLOTS
			# profiles of flux decomposition (intensity weighted by area coverage)
			if 'flx_split' in L_plots:
				part_1 = mean_vertical_contrib(ab_1_p,ab_p,indzi).values
				part_2 = mean_vertical_contrib(ab_2_p,ab_p,indzi).values
				part_3 = mean_vertical_contrib(ab_3_p,ab_p,indzi).values
				part_4 = mean_vertical_contrib(ab_4_p,ab_p,indzi).values
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
				ax.vlines(0,0,1.5,colors='grey')
				if string_var=='wthtv':
					beta = (np.amax(ab_p) - np.amin(ab_p)).values
					txt_beta = r', $\beta$='+str(np.round(beta,5))
				else:
					txt_beta =''
				
				ax.plot(ab_1_p/dic_norm[case][string_var],Z/ABLHm,color=c1,label="QI: "+varname+"'>0 w'>0 ("+str(np.round(part_1*100,2))+"%)")
				ax.plot(ab_2_p/dic_norm[case][string_var],Z/ABLHm,color=c2,label="QII: "+varname+"'<0 w'>0 ("+str(np.round(part_2*100,2))+"%)")
				ax.plot(ab_3_p/dic_norm[case][string_var],Z/ABLHm,color=c3,label="QIII: "+varname+"'<0 w'<0 ("+str(np.round(part_3*100,2))+"%)")
				ax.plot(ab_4_p/dic_norm[case][string_var],Z/ABLHm,color=c4,label="QIV: "+varname+"'>0 w'<0 ("+str(np.round(part_4*100,2))+"%)")
				ax.plot(ab_p/dic_norm[case][string_var],Z/ABLHm,color='black',label="mean",ls='--')
				#ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
				ax.set_title(title+CFILTER_title+txt_beta,loc='right')
				ax.legend()
				ax.set_ylim(ylim)
				if string_var =='uw' or string_var=='wthtv':
					ax.set_xlim(xlim)
				ax.set_ylabel('z/zi')
				ax.set_xlabel(title_profile_quadrants)
				fig.savefig(path_save+title+'_profile_quadrants_'+string_var+CFILTER+'.png')
			# area coverage
			if 'cover' in L_plots:
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
				ax.plot(F1,Z/ABLHm,color=c1,label="QI: "+varname+"'>0 w'>0")
				ax.plot(F2,Z/ABLHm,color=c2,label="QII: "+varname+"'<0 w'>0")
				ax.plot(F3,Z/ABLHm,color=c3,label="QIII: "+varname+"'<0 w'<0")
				ax.plot(F4,Z/ABLHm,color=c4,label="QIV: "+varname+"'>0 w'<0")
				ax.set_title(title+CFILTER_title,loc='right')
				ax.legend()	
				ax.set_ylim(ylim)
				ax.set_ylabel('z/zi')
				ax.set_xlabel(r"Area of each quadrants")
				fig.savefig(path_save+case+'_profile_quadrants_'+string_var+'_coverage'+CFILTER+'.png')
			# profiles of quadrants terms intensity
			#   cf fig 9 Salesky et al 2017 
			# 	this is the mean value of the ab flux on points where (x,y) belongs to Qi.
			#	this operation is quite long for a 4D field 
			if 'intensity' in L_plots:
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
				ax.plot(S1m,Z/ABLHm,color=c1,label="QI: "+varname+"'>0 w'>0")
				ax.plot(S2m,Z/ABLHm,color=c2,label="QII: "+varname+"'<0 w'>0")
				ax.plot(S3m,Z/ABLHm,color=c3,label="QIII: "+varname+"'<0 w'<0")
				ax.plot(S4m,Z/ABLHm,color=c4,label="QIV: "+varname+"'>0 w'<0")
				ax.set_ylabel('z/zi')
				ax.set_title(title+CFILTER_title,loc='right')
				ax.legend()
				ax.set_xlim([0,0.7])
				ax.set_ylim(ylim)
				ax.set_xlabel(r" $\vert \overline{"+string_var+r"}_i \vert$ / $\sum \vert \overline{"+string_var+r"} \vert$")
				fig.savefig(path_save+case+'_profile_quadrants_'+string_var+'_intensity'+CFILTER+'.png')
			# XY Slice	
			if 'XY' in L_plots:
				stepx,stepy = L_paramSlice['XY']['stepx'],L_paramSlice['XY']['stepy']
				X1,X2 = L_paramSlice['XY']['X1'],L_paramSlice['XY']['X2']
				indx1,indx2 = np.argmin(np.abs(X.values-X1*1000)),np.argmin(np.abs(X.values-X2*1000))
				atzi = L_paramSlice['XY']['AtCoord']
				indz = np.argmin(np.abs(Z.values-atzi*ABLHm)) 	
				dic_seuil = {'uw':L_filter['uw']*dic_norm[case]['uw'],'wthtv':L_filter['wthtv']*dic_norm[case]['wthtv']}
				
				fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)				
				ax.set_title('W, objects ('+string_var+') and (wind - mean wind) at '+str(atzi)+'zi'+CFILTER_title)
				s = ax.pcolormesh( X[indx1:indx2]/1000,Y/1000,W[indz,:,indx1:indx2], cmap= 'Greys_r',vmin=-1,vmax=1)
				plt.colorbar(s,ax=ax,orientation='horizontal')
				ax.contour( X[indx1:indx2]/1000+0.05/2,Y/1000+0.05/2,updrafts[indz,:,indx1:indx2],levels=[dic_seuil[string_var]],colors=['r'],linewidths=1.0,linestyles=['-'])
				ax.contour( X[indx1:indx2]/1000+0.05/2,Y/1000+0.05/2,downdrafts[indz,:,indx1:indx2],levels=[dic_seuil[string_var]],colors=['g'],linewidths=1.0,linestyles=['-'])
				Q = ax.quiver(X[indx1:indx2:stepx]/1000,Y[::stepy]/1000,(U-Um)[indz,::stepy,indx1:indx2:stepx],(V-Vm)[indz,::stepy,indx1:indx2:stepx],
						(THTV-THTVm)[indz,::stepy,indx1:indx2:stepx],cmap='coolwarm',clim=(-0.1,0.1),
						angles='xy',pivot='middle',scale=30) #,headwidth=2
				ax.set_ylabel('Y (km)')	
				ax.set_xlabel('X (km)')
				ax.set_aspect('equal')
				ax.quiverkey(Q, 0.9, 0.05, 1, '1 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
				fig.savefig(path_save+case+'_quadrants_obj_'+string_var+'_'+str(atzi)+'zi'+CFILTER+'.png')	
			# YZ Slice	
			if 'YZ' in L_plots:
				stepy,stepz = L_paramSlice['YZ']['stepy'],L_paramSlice['YZ']['stepz']
				#Y1,Y2 = L_paramSlice['YZ']['X1'],L_paramSlice['YZ']['X2']
				#indx1,indx2 = np.argmin(np.abs(X.values-X1*1000)),np.argmin(np.abs(X.values-X2*1000))
				atX = L_paramSlice['YZ']['AtCoord']
				indx = np.argmin(np.abs(X.values-atX*1000)) 	
				dic_seuil = {'uw':L_filter['uw']*dic_norm[case]['uw'],'wthtv':L_filter['wthtv']*dic_norm[case]['wthtv']}
				
				fig, ax = plt.subplots(1,1,figsize = (6,6),constrained_layout=True,dpi=dpi)				
				ax.set_title('W, objects ('+string_var+') and (wind - mean wind) \nat X='+str(atX)+'km'+CFILTER_title)
				s = ax.pcolormesh( Y/1000,Z/ABLHm,W[:,:,indx], cmap= 'coolwarm',vmin=-1,vmax=1,alpha=0.5) # Greys_r
				plt.colorbar(s,ax=ax,orientation='vertical')
				ax.contour( Y/1000+0.05/2,(Z-dz/2)/ABLHm,updrafts[:,:,indx],levels=[dic_seuil[string_var]],colors=['r'],linewidths=1.0,linestyles=['-'])
				ax.contour( Y/1000+0.05/2,(Z-dz/2)/ABLHm,downdrafts[:,:,indx],levels=[dic_seuil[string_var]],colors=['g'],linewidths=1.0,linestyles=['-'])
				Q = ax.quiver(Y[::stepy]/1000,Z[::stepz]/ABLHm,(V-Vm)[::stepz,::stepy,indx],W[::stepz,::stepy,indx],
						(THTV-THTVm)[::stepz,::stepy,indx],cmap='coolwarm',clim=(-0.1,0.1),
						angles='xy',pivot='middle',scale=20) #,headwidth=2
				ax.set_ylabel('Y (km)')	
				ax.set_xlabel('X (km)')
				#ax.set_aspect('equal')
				ax.set_aspect(2.0)
				ax.set_ylim([0,1.2])
				ax.quiverkey(Q, 0.9, 0.05, 1, '1 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
				fig.savefig(path_save+case+'_quadrants_obj_'+string_var+'_'+str(atX)+'X'+CFILTER+'.png')	
				
		# profile of efficiency
		#		note : for 'cold' ref, as Q0<0 at surface this figure doesnt say much
		""" -> this could be done at several X, plus the ref profiles"""
		if 'efficiency' in L_plots:
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			ax.plot(n_ab['S1'],Z/dic_ABLH['S1'],color='k',label='S1 at X='+str(X1)+'km')
			ax.plot(n_ab['cold'],Z/dic_ABLH['cold'],color='blue',label='cold')
			ax.plot(n_ab['warm'],Z/dic_ABLH['warm'],color='red',label='warm')
			ax.set_ylabel('z/zi')
			ax.set_xlim([0,1])
			ax.set_ylim([0,1])
			ax.legend()
			ax.set_xlabel(title_efficiency)
			ax.set_title(CFILTER_title[1:])
			fig.savefig(path_save+'profile_quadrants_efficiency_'+string_var+'_'+str(X1)+'km.png')


def plot_uw_efficiencies_atX(X,Z,Tstart,Tstop,window,crit_value,dsflx,L_dsquadrant,dsref,L_atX,dataSST,zimax,path_save,dpi):
	"""
	This procedure plots the momentum efficiency as defined in Salesky et al. "On the Nature of the Transition Between Roll and Cellular Organization in the Convective Boundary Layer".
	The plots shows the profiles of efficiency at some X for S1 and the reference profiles.
	
	INPUTS:
		- X 		: X direction dataArray
		- Z 		: Z direction dataArray
		- Tsart 	: for MeanTurb, index for 1rst instant
		- Tstop, 	: for MeanTurb, index for last instant
		- window 	: for MeanTurb, index window for 1D filter
		- crit_value	: threshold of SST to change colormap in 'DISCRETIZED_2CMAP_2'
		- dsflx 	: dataset with the total and resolved fluxes for S1
		- L_dsquadrant 	: dictionnary with quadrant dataset for S1 and ref sim
		- dsref 	: dataset with reference sim values
		- L_atX 	: liste of X position to plot profiles of S1 at
		- dataSST 	: 1D SST jump
		- zimax	: altitude max to plot profiles (in z/zi)
		- path_save 	: where to save figures
		- dpi 		: for figures
	
	"""
	indt = -1 # for last instant for ref sim
	n_uw = {}
	dic_ABLH = {}
	# computing efficiencies
	for case in L_dsquadrant.keys():
		uw_1 = L_dsquadrant[case]['uw_1']
		uw_2 = L_dsquadrant[case]['uw_2']
		uw_3 = L_dsquadrant[case]['uw_3']
		uw_4 = L_dsquadrant[case]['uw_4']
		uw_r = uw_1 + uw_2 + uw_3 + uw_4
		if case in ['warm','cold']:	
			gTHTV = ( dsref[case]['nomean']['Mean']['MEAN_TH']
				* (1+1.61*dsref[case]['nomean']['Mean']['MEAN_RV'])
				/ (1+dsref[case]['nomean']['Mean']['MEAN_RV']) )[indt].differentiate('level_les')
			dic_ABLH[case] = Z[gTHTV.argmax('level_les').values].values
			uw_sgs = dsref[case]['nomean']['Subgrid']['SBG_WU'][indt,:]
			uw_2 = uw_2[0,:,:,:].mean(dim={'ni','nj'})
			uw_4 = uw_4[0,:,:,:].mean(dim={'ni','nj'})
			uw_r = uw_r[0,:,:,:].mean(dim={'ni','nj'})
			uw   = (uw_sgs + uw_r).values	
		elif case=='S1':	
			uw_2 = MeanTurb(uw_2,Tstart,Tstop,window)
			uw_4 = MeanTurb(uw_4,Tstart,Tstop,window)
			uw_r = MeanTurb(uw_r,Tstart,Tstop,window)
			uw   = dsflx.FLX_UW
			dic_ABLH['S1'] = ABLH_S1	
		n_uw[case] = uw_r / (uw_2+uw_4)	
	# building a nice colormap
	cmap_warm ='Reds'
	cmap_cold ='winter' 
	colorsX = DISCRETIZED_2CMAP_2(cmap_cold,cmap_warm,L_atX*1000,dataSST,crit_value,X.values)	
	# plot
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot(n_uw['cold'],Z/dic_ABLH['cold'],color='blue',label='cold',ls='--')
	ax.plot(n_uw['warm'],Z/dic_ABLH['warm'],color='red',label='warm',ls='--')
	for k,X1 in enumerate(L_atX):
		indx = nearest(X.values,X1*1000)
		ax.plot( n_uw['S1'].isel(ni=indx), Z/dic_ABLH['S1'],color=colorsX[k],label='S1 at X='+str(X1)+'km')
	ax.set_ylabel('z/zi')
	ax.set_xlim([0.0,1])
	ax.set_ylim([0,zimax])
	ax.xaxis.label.set_fontsize(13)
	ax.yaxis.label.set_fontsize(13)
	ax.set_xlabel(r" <$\~u \~w$> / <$\~u\~w_{II} + \~u\~w_{IV}$>")
	ax.legend() 
	ax.grid()
	fig.savefig(path_save+'efficiency_uw.png')


def plot_wthtv_efficiencies_atX(X,Z,Tstart,Tstop,window,crit_value,dsflx,L_dsquadrant,dsref,L_atX,dataSST,zimax,path_save,dpi):
	"""
	This procedure plots the buoyancy efficiency as defined in Salesky et al. "On the Nature of the Transition Between Roll and Cellular Organization in the Convective Boundary Layer".
	The plots shows the profiles of efficiency at some X for S1 and the reference profiles.
	
	INPUTS:
		- X 		: X direction dataArray
		- Z 		: Z direction dataArray
		- Tsart 	: for MeanTurb, index for 1rst instant
		- Tstop, 	: for MeanTurb, index for last instant
		- window 	: for MeanTurb, index window for 1D filter
		- crit_value	: threshold of SST to change colormap in 'DISCRETIZED_2CMAP_2'
		- dsflx 	: dataset with the total and resolved fluxes for S1
		- L_dsquadrant 	: dictionnary with quadrant dataset for S1 and ref sim
		- dsref 	: dataset with reference sim values
		- L_atX 	: liste of X position to plot profiles of S1 at
		- dataSST 	: 1D SST jump
		- zimax	: altitude max to plot profiles (in z/zi)
		- path_save 	: where to save figures
		- dpi 		: for figures
	
	"""
	indt = -1 # for last instant for ref sim
	n_wthtv = {}
	dic_ABLH = {}
	tracer = "wthtv"
	# computing efficiencies
	for case in L_dsquadrant.keys():
		wthtv_1 = L_dsquadrant[case][tracer+'_1']
		wthtv_2 = L_dsquadrant[case][tracer+'_2']
		wthtv_3 = L_dsquadrant[case][tracer+'_3']
		wthtv_4 = L_dsquadrant[case][tracer+'_4']
		wthtv_r = wthtv_1 + wthtv_2 + wthtv_3 + wthtv_4
		if case in ['warm','cold']:	
			gTHTV = ( dsref[case]['nomean']['Mean']['MEAN_TH']
				* (1+1.61*dsref[case]['nomean']['Mean']['MEAN_RV'])
				/ (1+dsref[case]['nomean']['Mean']['MEAN_RV']) )[indt].differentiate('level_les')
			dic_ABLH[case] = Z[gTHTV.argmax('level_les').values].values
			#wthtv_sgs = dsref[case]['nomean']['Subgrid']['SBG_WU'][indt,:]
			wthtv_1 = wthtv_1[0,:,:,:].mean(dim={'ni','nj'})
			wthtv_3 = wthtv_3[0,:,:,:].mean(dim={'ni','nj'})
			wthtv_r = wthtv_r[0,:,:,:].mean(dim={'ni','nj'})
			#wthtv   = (wthtv_sgs + wthtv_r).values #ab[0,:,:,:].mean(dim={'ni','nj'})	
		elif case=='S1':	
			wthtv_1 = MeanTurb(wthtv_1,Tstart,Tstop,window)
			wthtv_3 = MeanTurb(wthtv_3,Tstart,Tstop,window)
			wthtv_r = MeanTurb(wthtv_r,Tstart,Tstop,window)
			wthtv   = dsflx.FLX_THvW # ab[:,:,:,:].mean(dim={'ni','nj'})	
			dic_ABLH['S1']= ABLH_S1	
		n_wthtv[case] = wthtv_r / (wthtv_1+wthtv_3)	
	# building a nice colormap
	cmap_warm ='Reds'
	cmap_cold ='winter' 
	colorsX = DISCRETIZED_2CMAP_2(cmap_cold,cmap_warm,L_atX*1000,dataSST,crit_value,X.values)	
	# plot
	fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
	ax.plot(n_wthtv['cold'],Z/dic_ABLH['cold'],color='blue',label='cold',ls='--')
	ax.plot(n_wthtv['warm'],Z/dic_ABLH['warm'],color='red',label='warm',ls='--')
	for k,X1 in enumerate(L_atX):
		indx = nearest(X.values,X1*1000)
		ax.plot( n_wthtv['S1'].isel(ni=indx), Z/dic_ABLH['S1'],color=colorsX[k],label='S1 at X='+str(X1)+'km')
	ax.set_ylabel('z/zi')
	ax.set_xlim([0.0,1])
	ax.set_ylim([0,1.2])
	#ax.set_xlabel(r" <$\~w \~\theta_v$> / <$\~w\~\theta_{v,I} + \~w\~\theta_{v,III}$>")
	ax.set_xlabel('efficiency of '+tracer)
	ax.legend() 
	ax.grid()
	fig.savefig(path_save+'efficiency_'+tracer+'.png')

def plot_JPDF(X,Y,Z,X1,L_dsmean,L_ds,L_dsquadrant,case,string_flx,string_obj,indt,L_atzi,dpi):
	"""
	This procedure is plotting the JPDF of two variable
	
	
	INPUT:
		- X 			: DataArray containing X dimension
	 	- Y  			: DataArray containing Y dimension
	 	- Z  			: DataArray containing Z dimension
	 	- X1 			: X location (in km) to plot profiles of S1 simulation
	 	- L_dsmean 		: dic of mean files with name of simu as key
	 	- L_ds 			: dic of instantaneous file with name of simu as key
	 	- L_dsquadrant 	: dic of quadrant splitting file with name of simu as key
		- case 			: 'warm' or 'cold' or 'S1'
		- string_flx 	: what to plot : u' and w' or thtv' and w'
		- string_obj 	: flux on which the decomposition is done, 'wthtv' or 'uw'.
							can be '' and so in this case no decomposition is done.
		- indt			: time index
		- L_atzi 		: list of float, fraction of zi at which to plot the JPDF
		- dpi 			: for plot
	"""
	
	

	# -> Mean values and ABLH
	if '000' in L_dsmean[case]: # MNH diachronic file
		ABLH = L_dsmean[case]['nomean']['Misc']['BL_H'][indt].values
		Utm = L_dsmean[case]['nomean']['Mean']['MEAN_U'][indt,:]
		THTVtm = L_dsmean[case]['nomean']['Mean']['MEAN_THV'][indt,:]
		u_star = L_dsmean[case]['nomean']['Surface']['Ustar'][indt].values
		Utm = Utm.rename(new_name_or_name_dict={'level_les':'level'})
		THTVtm = THTVtm.rename(new_name_or_name_dict={'level_les':'level'})
#			E0 	= L_ds[case]['RCONSW_FLX'].interp({'level_w':L_ds[case].level})[0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
#			Q0	= L_ds[case]['THW_FLX'][0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
#			THT_z0 	= L_ds[case]['THT'][0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
#			RV_z0 	= L_ds[case]['RVT'][0,nhalo,nhalo:-nhalo,nhalo:-nhalo].mean().values
#			THTv_z0	= THT_z0*(1+1.61*RV_z0)/(1+RV_z0)
#			Qv0	= THTv_z0/THT_z0*Q0+0.61*THT_z0*E0
		#title=case
	else: # hand built mean file
		THTVtm = L_dsmean[case].THTvm[:,:]
		Utm = L_dsmean[case].Um[:,:]
		#Vm = L_dsmean[case].Vm[:,:]
		gTHT = THTVtm[:,:].differentiate('level')
		ABLH = Z[gTHT.argmax(dim='level')]
		ABLH = ABLH.mean().values
		u_star = L_dsmean[case].u_star[indX_norm].values
#			E0 	= L_dsmean[case].E0[indX_norm]
#			Q0 	= L_dsmean[case].Q_star[indX_norm]
#			Qv0	= L_dsmean[case].Qv_star[indX_norm]
		title=case+' at X='+str(X1)+'km'

	fig, ax = plt.subplots(1,len(L_atzi),figsize = (15,5), constrained_layout=True	,dpi=dpi)
	for k,atzi in enumerate(L_atzi):
		print('z/zi='+str(atzi))
		indz = nearest(Z.values,atzi*ABLH)
		Um,THTVm = Utm[indz],THTVtm[indz]
		
		# Instantaneous field
		U = L_ds[case].UT.interp({'ni_u':L_ds[case].ni})[indt,indz,nhalo:-nhalo,nhalo:-nhalo] 	# grid : 2
		U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
		W = L_ds[case].WT.interp({'level_w':L_ds[case].level})[indt,indz,nhalo:-nhalo,nhalo:-nhalo]
		THT = L_ds[case].THT[indt,indz,nhalo:-nhalo,nhalo:-nhalo]
		RVT = L_ds[case].RVT[indt,indz,nhalo:-nhalo,nhalo:-nhalo]
		THTV = THT*(1+Rv/Rd*RVT)/(1+RVT)
		Um,THTVm = Complete_dim_like([Um,THTVm],U)
		# what flux is plotted
		if string_flx=='uw':
			a,b = W,U-Um
			nameA,nameB = "w'","u'"
			xlim,ylim = [-1,1],[-1,1]
		elif string_flx=='wthtv':
			a,b = W, THTV-THTVm
			nameA,nameB = "w'","thtv'"
			xlim,ylim = [-1,1],[-0.2,0.2]
		a_f = np.ravel(a.values)
		b_f = np.ravel(b.values)
		# how quadrants are shown depends on physical meaning of flux
		if string_obj=='uw':
			colors = ['blue','red','black','green']
		elif string_obj=='wthtv':
			colors = ['red','black','green','blue']
		if string_obj!='':
			quad_info = (xr.where( L_dsquadrant[case][string_obj+'_1'][indt,indz] > 0, 1,0) +
				xr.where( L_dsquadrant[case][string_obj+'_2'][indt,indz] < 0, 2,0) +
				xr.where( L_dsquadrant[case][string_obj+'_3'][indt,indz] > 0, 3,0) +
				xr.where( L_dsquadrant[case][string_obj+'_4'][indt,indz] < 0, 4,0) )
			quad_info_f = np.ravel(quad_info.values)
		else:	
			quad_info_f = np.zeros(a_f.shape)

		ab_f = a_f*b_f
		d = {string_flx:ab_f,nameA:a_f,nameB:b_f,"quadrant":quad_info_f}
		data = pd.DataFrame(data=d)
		ax[k].vlines(0,ylim[0],ylim[-1],colors='grey')
		ax[k].hlines(0,xlim[0],xlim[-1],colors='grey')
		levels = [0.1,0.5,0.9]
		if string_obj=='':
			sns.kdeplot(ax=ax[k],data=data, x=nameA, y=nameB,color='k',levels=levels)
			#ax.get_legend().set_title("quadrant")
		else:
			sns.kdeplot(ax=ax[k],data=data, x=nameA, y=nameB,color='grey',levels=levels,linestyles='--')
			sns.kdeplot(ax=ax[k],data=data, x=nameA, y=nameB,hue='quadrant',levels=levels,palette=colors)
			ax[k].get_legend().set_title("quadrant("+string_obj+')')
		
		ax[k].set_ylim(ylim)
		ax[k].set_xlim(xlim)
		ax[k].set_title("JPDF of "+nameA+" and "+nameB+" at z/zi="+str(atzi))
	


def plots_ref_CS1(X,Y,Z,dsCS1,dsref,K,atzi,aty,atx,SEUIL_ML,SVT_type,indt,L_case,L_var,L_plot,path_save,dpi):
	"""	
	This procedure is plotting slices of ref simulations with coherent structures
		- X,Y,Z are dimensions of MNH domain
		- dsCS1 is the instantaneous file with the CS1 masks
		- dsref is the 000.nc file of the reference runs
		- K is the minimum area threshold to plot profiles
		- atzi is the fraction of zi at which XY plots are shown
		- aty is the Y location of the XZ slice
		- atx is the X location of the YZ slice
		- SEUIL_ML is the threshold to detect the mixed layer on tht
		- SVT_type : type of tracer used
		- indt is the time index selected (for zi)
		- L_case is the list of the name of the reference runs
		- L_var is the list of variable plotted (for XY plots and profiles)
			available : 'U','W','RVT','THT','THTV'
		- L_plot is the list of the desired plots :
			 * 'ML_bounds' : show mixed layer bounds with gradTHT profile
			 * 'turb_cond' : show how 1 profile at x,y fixed is filtered
			 * 'profils_var' : show the profile of VAR with objects decomposition
			 * 'XY' : 1 fig, 2 plots. VAR seule, then VAR in background and object forground
			 * 'XZ' : 2 fig, 2 and 3 plots each.
				1rst fig: W and (objects + vector wind)
				2nd fig: anomaly of thtv-<thtv>, contours of objects
					then tht - tht(mixed layer)
					then rv - rv(mixed layer)
			 * 'YZ' : same as 'XZ' but for 'YZ' plane
			 * 'obj_frac' : show area coverage fraction as profiles for each objects
		- path_save is the abs path to store figures
		- dpi is dot per inches for figures
		
		Note : the mean used here is in X and in Y (homogeneous cases)
	"""
	for case in L_case:	
		is_turb = dsCS1[case].is_turb
		is_up = dsCS1[case].is_up
		is_sub = dsCS1[case].is_sub
		is_down = dsCS1[case].is_down
		is_env = dsCS1[case].is_env
		global_objects = dsCS1[case].global_objects
		ABLH = dsref[case]['nomean']['Misc']['BL_H'][indt].values
		Z_w = dsref[case]['000'].level_w[nhalo:-nhalo].values
		indzi = np.argmin(np.abs(Z.values-1.1*ABLH)) # looking at 1.1*zi to account for overshoots
		TURB_COND = dsCS1[case].attrs['Turb_cond']
		RV_DIFF = dsCS1[case].attrs['RV_DIFF']
		U,Um = dsCS1[case].UT,dsCS1[case].UTm
		V,Vm = dsCS1[case].VT,dsCS1[case].VTm
		W,Wm = dsCS1[case].WT,dsCS1[case].WTm
		THT,THTm = dsCS1[case].THT,dsCS1[case].THTm
		THTV,THTVm = dsCS1[case].THTV,dsCS1[case].THTVm
		RV,RVm = dsCS1[case].RVT,dsCS1[case].RVTm
		E,Em = dsCS1[case].E,dsCS1[case].Em
		
		if SVT_type=='SVT':
			SV1,SV1m = dsCS1[case].SV1,dsCS1[case].SV1m
			SV3,SV3m = dsCS1[case].SV3,dsCS1[case].SV3m
		rv_fluc = RV - RVm
		tht_fluc = THT - THTm
		thtv_fluc = THTV - THTVm
		u_fluc = U - Um
		v_fluc = V - Vm
		w_fluc = W - Wm
		uw = u_fluc*w_fluc
		wtht = w_fluc*tht_fluc
		wthtv = w_fluc*thtv_fluc
		wrv = w_fluc*rv_fluc
		uw_mean= uw.mean(dim=['ni','nj'])
		uw_up_p		,uw_sub_p	,uw_down_p	,uw_env_p 	= compute_flx_contrib(uw,[is_up,is_sub,is_down,is_env],meanDim=['ni','nj'])
		#std_uw_up	,std_uw_sub	,std_uw_down,std_uw_env = compute_std_flx(uw,[is_up,is_sub,is_down,is_env],stdDim=['ni','nj'])
		wtht_mean= wtht.mean(dim=['ni','nj'])
		wtht_up_p,wtht_sub_p,wtht_down_p,wtht_env_p = compute_flx_contrib(wtht,[is_up,is_sub,is_down,is_env],meanDim=['ni','nj'])
		wthtv_mean= wthtv.mean(dim=['ni','nj'])
		wthtv_up_p,wthtv_sub_p,wthtv_down_p,wthtv_env_p = compute_flx_contrib(wthtv,[is_up,is_sub,is_down,is_env],meanDim=['ni','nj'])
		wrv_mean= wrv.mean(dim=['ni','nj'])
		wrv_up_p,wrv_sub_p,wrv_down_p,wrv_env_p = compute_flx_contrib(wrv,[is_up,is_sub,is_down,is_env],meanDim=['ni','nj'])
		# surface values		
		E0 = dsCS1[case]['E0'].values
		Q0 = np.abs(dsCS1[case]['Q0'].values)
		Qv0 = dsCS1[case]['Qv0'].values
		u_star = dsCS1[case]['u_star'].values
		# area cover
		F_up = dsCS1[case].is_up.mean(dim=['ni','nj']) 
		F_sub = dsCS1[case].is_sub.mean(dim=['ni','nj']) 
		F_down = dsCS1[case].is_down.mean(dim=['ni','nj']) 
		F_env = dsCS1[case].is_env.mean(dim=['ni','nj']) 
		# mixed layer indexes
		gTHT = THTVm[:,0,0].differentiate('level')
		indz1,indz2 = get_mixed_layer_indexes(Z,gTHT[:],SEUIL_ML)	
		indcrit = int(( indz1 + indz2 ) / 2)
		THTmixed = THTm[indz1:indz2,:,:].mean().values
		THTvmixed = THTVm[indz1:indz2,:,:].mean().values
		Umixed = Um[indz1:indz2,:,:].mean().values
		Vmixed = Vm[indz1:indz2,:,:].mean().values
		RVmixed = RVm[indz1:indz2,:,:].mean().values
		rv_crit = RVm[indcrit,:,:].mean().values # rv middle of mixed layer
		print('		--> case is '+case)
		print('		TURB_COND='+TURB_COND)
		print('		SVT_type='+SVT_type)
		print("		RV_DIFF=",RV_DIFF)
		print('		z1 =',np.round(Z[indz1].values,2),'m, z2 =',np.round(Z[indz2].values,2),'m, zi =',np.round(ABLH,2),'m')
		print('		thtv mixed =',np.round(THTvmixed,2),'K')
		print('		tht mixed =',np.round(THTmixed,2),'K')
		print('		rv mixed =',np.round(RVmixed*1000,2),'g/kg')
		print('		U mixed =',np.round(Umixed,2),'m/s')
		print('		V mixed =',np.round(Vmixed,2),'m/s')
		print('		rv_crit',np.round(rv_crit*1000,2),'g/kg at z =',np.round(Z[indcrit].values,2),'m')
		
		if 'ML_bounds' in L_plot: # plot the mixed layer boundaries
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=100)
			ax.plot(gTHT,Z,c='k')
			ax.hlines(Z[indz1],-0.01,0.01,colors='b',ls='--')
			ax.hlines(Z[indz2],-0.01,0.01,colors='r',ls='--')
			ax.set_ylabel('z')
			ax.set_xlabel('gTHT')			

		dz = np.zeros(Z_w.shape)
		dz[:-1] = Z_w[1:] - Z_w[:-1]
		dz[-1] = dz[-2]	
		
		path_save2 = path_save+TURB_COND
		if TURB_COND=='C10':
			mCS = dsCS1[case].attrs['mCS']
			gammaRV = dsCS1[case].attrs['gammaRv']
			path_save2 = path_save2+'_m'+str(mCS)+'_g'+str(gammaRV*100)
			if 'turb_cond' in L_plot:
				# This figure shows that the fluctuation of rv in the less turbulent air above the 
				#	ABL are very close to the fluctuations of rv in the ABL so we cannot 
				#	clearly sperate them.
				print('			-C10 turb condition with mCS=',mCS,'gammaRV=',gammaRV*100,'%')
#				rv_var = np.std(rv_fluc,axis=(1,2))
#				rv_var_min = Integ_min(Z,Y,X,np.sqrt(RV2m),gammaRV,z1,zi=ABLH)
#				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
#				ax.plot(rv_fluc[:,20,20],Z/ABLH,c='r',ls='--',label="rv' pre filtre")
#				ax.plot(xr.where(is_turb,rv_fluc,0)[:,20,20],Z/ABLH,c='r',ls='-',label="rv' post filtre")
#				ax.plot((mCS*rv_var_min),Z/ABLH,c='k',ls='--',label='mCS*RV²')
#				ax.plot((mCS*np.sqrt(RV2m))[:,20,20],Z/ABLH,c='grey',ls='--',label='mCS*RV²m')
#				ax.set_xlabel('rv fluc')
#				ax.set_ylabel('z/zi')
#				ax.set_ylim([0,2.0])
#				ax.legend()
#				ax.set_title(case)
		elif TURB_COND=='ITURB':
			gammaTurb1 = dsCS1[case].attrs['gammaTurb1']
			path_save2 = path_save2+ '_g'+str(gammaTurb1*100)
			if 'turb_cond' in L_plot:
				print('			-Iturb turb condition with gammaTurb1=',gammaTurb1*100,'%')
				u_turb = np.sqrt(2/3*E)
				M = np.sqrt(U**2+V**2+W**2)
				I_turb = u_turb/M	
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				ax.plot(I_turb[:,20,20],Z/ABLH,c='r',ls='-',label='I_turb at (i,j)=(20,20)')
				ax.vlines(gammaTurb1,0,3,color='grey')
				ax.set_xlabel('')
				ax.set_ylabel('z/zi')
				ax.set_ylim([0,2.0])
				ax.legend()
				ax.set_title(case)
		elif TURB_COND=='ITURB2':
			gammaTurb2 = dsCS1[case].attrs['gammaTurb2']
			path_save2 = path_save2+'_g'+str(gammaTurb2*100)
			if 'turb_cond' in L_plot:
				print('			-Iturb2 turb condition with gammaTurb2=',gammaTurb2*100,'%')
				u_turb = np.sqrt(2/3*E)
				M = np.sqrt(U**2+V**2+W**2)
				u_turb_mean = np.sqrt(2/3*Em)
				M_mean = np.sqrt(Um**2+Vm**2+Wm**2)
				I_turb = u_turb/M
				I_turb_mean = u_turb_mean/M_mean								
				I_turb_min = Integ_min(Z,Y,X,I_turb_mean,gammaTurb2,z1=0,zi=ABLH)
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				ax.plot(I_turb_mean[:,20,20],Z/ABLH,c='g',ls='-',label='mean h turb intensity')
				ax.plot(I_turb_min[:,20,20],Z/ABLH,c='b',ls='-',label='I turb min')
				ax.plot(I_turb[:,20,20],Z/ABLH,c='r',ls='-',label='I_turb at (i,j)=(20,20)')
				ax.set_xlabel('')
				ax.set_ylabel('z/zi')
				ax.set_ylim([0,2.0])
				ax.legend()
				ax.set_title(case)
		elif TURB_COND=='EC':	
			gammaEc = dsCS1[case].attrs['gammaEc']
			path_save2 = path_save2 +'_g'+str(gammaEc*100)
			if 'turb_cond' in L_plot:
				print('			-EC turb condition with gammaEc=',gammaEc*100,'%')
				# show E intensity
				E_min = Integ_min(Z,Y,X,Em,gammaEc,z1=0,zi=ABLH)
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				ax.plot(Em[:,20,20],Z/ABLH,c='g',ls='-',label='mean h E')
				ax.plot(E_min[:,20,20],Z/ABLH,c='b',ls='-',label='E min')
				ax.plot(E[:,20,20],Z/ABLH,c='r',ls='-',label='E at (i,j)=(20,20)')
				ax.set_xlabel('')
				ax.set_ylabel('z/zi')
				ax.set_ylim([0,2.0])
				ax.legend()
				ax.set_title(case)
		if SVT_type=='RV':
			path_save2 = path_save2+'_'+RV_DIFF+'_'+SVT_type+'/'
		else:
			path_save2 = path_save2+'_'+SVT_type+'/'
		path_save2 = path_save2 + case + '/'
		if not os.path.isdir(path_save2): # creat folder if it doesnt exist
			os.makedirs(path_save2) 
		
		if 'turb_cond' in L_plot: # plot the turbulent filter
			print('			-turb filter')
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=100)
			ax.plot(is_turb[:,20,20],Z/ABLH,c='k')
			ax.set_ylabel('z/zi')
			ax.set_xlabel('1= keep, 0= filtered out')
			ax.set_title('Filter at i,j = 20,20') 
			fig.savefig(path_save2+'CS1_'+case+'_turbfilter.png')	
			
		
		# Plots of slices	
		indz = np.argmin(np.abs(Z.values-atzi*ABLH)) 	
		for nVAR in L_var:
			if nVAR=='W':
				VAR,units,bornesXY,bornesP,bornesStd,cmap = W,'(m/s)',[-1,1],[-0.8,1.1],[0,1],'coolwarm'
				unitstd = units
			elif nVAR=='U':
				VAR,units,bornesXY,bornesP,bornesStd,cmap = U,'(m/s)',[5,8],[5.0,7.5],[0,1],'Greys_r'
				unitstd = units
			elif nVAR=='RVT':
				VAR,units,bornesXY,bornesP,bornesStd,cmap = RV*1000,'(g/kg)',[10,13],[10,13],[0,1.2],'Blues'
				unitstd = units
			elif nVAR=='THT':
				VAR,units,cmap,bornesStd = THT-295,'- 295K (K)','Reds',[0,1]
				unitstd = '(K)'
				if case=='warm':
					bornesXY,bornesP = [0.9,1.25],[0.9,1.25]
				if case=='cold':
					bornesXY,bornesP = [0.5,0.7],[0.5,0.75]
			elif nVAR=='THTV':
				VAR,units,cmap,bornesStd = THTV-297,'- 297K (K)','Reds',[0,1]
				unitstd = '(K)'
				if case=='warm':
					bornesXY,bornesP = [0.75,1.5],[0.75,1.5]
				elif case=='cold':
					bornesXY,bornesP = [0.5,1],[0.5,1]
			else:
				raise Exception('Your choice of '+nVAR+' is not available yet')			
			#	masking 
			VAR_up_ma = np.ma.masked_where(np.logical_not(is_up),VAR)
			VAR_up_ma = VAR.where(is_up)
			VAR_sub_ma = VAR.where(is_sub)
			VAR_down_ma = VAR.where(is_down)
			VAR_env_ma = VAR.where(is_env)
			VAR_all_ma = VAR.where(is_turb)
			VAR_up_p = np.ma.masked_where(F_up<=K, VAR_up_ma.mean(axis=(1,2)) ) # select profiles where coverage > K
			VAR_sub_p = np.ma.masked_where(F_sub<=K, VAR_sub_ma.mean(axis=(1,2)) )
			VAR_down_p = np.ma.masked_where(F_down<=K, VAR_down_ma.mean(axis=(1,2)) )
			VAR_env_p = np.ma.masked_where(F_env<=K, VAR_env_ma.mean(axis=(1,2)) )
			VAR_mean= VAR.mean(axis=(1,2))
			VAR_std_up,VAR_std_sub,VAR_std_down,VAR_std_env = compute_std_flx(VAR,[is_up,is_sub,is_down,is_env],stdDim=['ni','nj'])
			# profils of VAR with CS
			if 'profils_var' in L_plot:
				print('			-profiles of '+nVAR)
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				ax.plot( VAR_up_p,Z/ABLH,c='red',label='updrafts')
				ax.fill_betweenx(Z/ABLH, (VAR_up_p-VAR_std_up),(VAR_up_p+VAR_std_up), color='red',alpha=0.5)
				ax.plot( VAR_sub_p,Z/ABLH,c='purple',label='sub. shells')
				ax.fill_betweenx(Z/ABLH, (VAR_sub_p-VAR_std_sub),(VAR_sub_p+VAR_std_sub), color='purple',alpha=0.5)
				ax.plot( VAR_down_p,Z/ABLH,c='green',label='downdrafts')
				ax.fill_betweenx(Z/ABLH, (VAR_down_p-VAR_std_down),(VAR_down_p+VAR_std_down), color='green',alpha=0.5)
				ax.plot( VAR_env_p,Z/ABLH,c='grey',label='env',ls='-')
				ax.fill_betweenx(Z/ABLH, (VAR_env_p-VAR_std_env),(VAR_env_p+VAR_std_env), color='grey',alpha=0.5)
				ax.plot( VAR_all_ma.mean(axis=(1,2)),Z/ABLH,c='k',label='all',ls='-')
				ax.plot( VAR_mean,Z/ABLH,c='k',label='mean',ls='--')
				ax.set_ylim([0,1.2])
				ax.grid()
				ax.set_title(case)
				ax.set_xlim(bornesP)
				ax.legend()
				ax.set_xlabel(nVAR+' '+units+''  )
				ax.set_ylabel('z/zi')
				fig.savefig(path_save2+'CS1_'+case+'_'+nVAR+'_profiles.png')
			if 'std_var_profiles' in L_plot:
				print('			-profiles of std of '+nVAR)
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				ax.plot( VAR_std_up,Z/ABLH,c='red',label='updrafts')
				ax.plot( VAR_std_sub,Z/ABLH,c='purple',label='sub. shells')
				ax.plot( VAR_std_down,Z/ABLH,c='green',label='downdrafts')
				ax.plot( VAR_std_env,Z/ABLH,c='grey',label='env',ls='-')
				ax.grid()
				ax.set_ylim([0,1.2])
				ax.set_title(case)
				ax.set_xlim(bornesStd)
				ax.legend()
				ax.set_xlabel(r'$\sigma$('+nVAR+') '+unitstd)
				ax.set_ylabel('z/zi')
				fig.savefig(path_save2+'CS1_'+case+'_'+nVAR+'_std_profiles.png')
		# spatial view XY, RV with CS
		if 'XY' in L_plot:
			print('			-XY with objects + vectors')
			stepx,stepy = 2,2 # number of skiped cell to draw arrows
			X1,X2 = 15,20	
			indx1,indx2 = np.argmin(np.abs(X.values-X1*1000)),np.argmin(np.abs(X.values-X2*1000))
			
			fig, ax = plt.subplots(1,1,figsize = (10,4),constrained_layout=True,dpi=dpi)				
#			ax.set_title('Rv, objects and wind - mean wind at '+str(atzi)+'zi')
#			s = ax.pcolormesh( X[indx1:indx2]/1000,Y/1000,RV[indz,:,indx1:indx2]*1000, cmap= 'Greys_r',vmin=RVmixed*1000-1,vmax=RVmixed*1000+1)
			ax.set_title('W, objects and (wind - mean wind) at '+str(atzi)+'zi')
			s = ax.pcolormesh( X[indx1:indx2]/1000,Y/1000,W[indz,:,indx1:indx2], cmap= 'Greys_r',vmin=-1,vmax=1)
			plt.colorbar(s,ax=ax,orientation='vertical')
			ax.contour( X[indx1:indx2]/1000+0.05/2,Y/1000+0.05/2,xr.where(is_up,1,0)[indz,:,indx1:indx2],levels=[0.55],colors=['r'],linewidths=1.0)
			ax.contour( X[indx1:indx2]/1000+0.05/2,Y/1000+0.05/2,xr.where(is_sub,1,0)[indz,:,indx1:indx2],levels=[0.55],colors=['purple'],linewidths=1.0)
			ax.contour( X[indx1:indx2]/1000+0.05/2,Y/1000+0.05/2,xr.where(is_down,1,0)[indz,:,indx1:indx2],levels=[0.55],colors=['g'],linewidths=1.0)
			ax.contour( X[indx1:indx2]/1000+0.05/2,Y/1000+0.05/2,xr.where(is_env,1,0)[indz,:,indx1:indx2],levels=[0.55],colors=['grey'],linewidths=1.0)
			Q = ax.quiver(X[indx1:indx2:stepx]/1000,Y[::stepy]/1000,(U-Um)[indz,::stepy,indx1:indx2:stepx],(V-Vm)[indz,::stepy,indx1:indx2:stepx],
					(THTV-THTVm)[indz,::stepy,indx1:indx2:stepx],cmap='coolwarm',clim=(-0.1,0.1),
					angles='xy',pivot='middle',scale=40,headwidth=2,headaxislength=4,headlength=4) #
			# 0.05 zi : ,scale=40,headwidth=2,headaxislength=4,headlength=4  
			# 0.5zi : ,scale=30,headwidth=2,headaxislength=4,headlength=4  
			ax.set_ylabel('Y (km)')	
			ax.set_xlabel('X (km)')
			ax.set_aspect('equal')
			ax.quiverkey(Q, 0.9, 0.05, 1, '1 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
			fig.savefig(path_save2+'CS1_'+case+'_XY_'+str(atzi)+'zi.png')
		# Spatial view XZ, CS and wind arrows
		if 'XZ' in L_plot:
			print('			-XZ with objects')
#			indy = np.argmin(np.abs(Y.values-aty*1000))
			indy = aty
			stepx,stepz = 3,4 # number of skiped cell to draw arrows
			X1,X2 = 15,20	
			indx1,indx2 = np.argmin(np.abs(X.values-X1*1000)),np.argmin(np.abs(X.values-X2*1000))
			fig, ax = plt.subplots(1,1,figsize = (8,4),constrained_layout=True,dpi=dpi)	
			ax.set_title('Objects and wind - mean wind',loc='right')
			ax.pcolormesh( X/1000,Z/ABLH,global_objects[:,indy,:],cmap=c.ListedColormap(['white','r','purple','g','grey']))
			Q = ax.quiver(X[indx1:indx2:stepx]/1000,Z[::stepz]/ABLH,(U-Um)[::stepz,indy,indx1:indx2:stepx],W[::stepz,indy,indx1:indx2:stepx],
					angles='xy',pivot='middle',headwidth=2,scale=30)
			ax.quiverkey(Q, 0.85, 0.05, 0.5, '0.5 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
			ax.set_ylim([0,1.2])
			ax.set_xlim([X1,X2])
			ax.set_aspect(2.0)
			ax.set_xlabel('X (km)')
			fig.savefig(path_save2+'CS1_'+case+'_XZ_j'+str(indy)+'.png')
			# thtv,tht and rv
			fig, ax = plt.subplots(3,1,figsize = (6,7),constrained_layout=True,dpi=dpi)
			s = ax[0].pcolormesh( X/1000,Z/ABLH,(THTV-THTVm)[:,indy,:],cmap='coolwarm',vmin=-0.15,vmax=0.15) 
			plt.colorbar(s,ax=ax[0])
			ax[0].tick_params(axis='both',labelbottom=False)
			ax[0].set_title(r'$\theta_v-<\theta_v>$ (K)',loc='right')
			ax[0].contour( X/1000+0.05/2,(Z+dz/2)/ABLH,xr.where(is_up,1,0)[:,indy,:],levels=[0.55],colors=['r'],linewidths=1.0)
			ax[0].contour( X/1000+0.05/2,(Z+dz/2)/ABLH,xr.where(is_sub,1,0)[:,indy,:],levels=[0.55],colors=['purple'],linewidths=1.0)
			ax[0].contour( X/1000+0.05/2,(Z+dz/2)/ABLH,xr.where(is_down,1,0)[:,indy,:],levels=[0.55],colors=['g'],linewidths=1.0)
			ax[0].contour( X/1000+0.05/2,(Z+dz/2)/ABLH,xr.where(is_env,1,0)[:,indy,:],levels=[0.55],colors=['grey'],linewidths=1.0)
			s = ax[1].pcolormesh( X/1000,Z/ABLH,THT[:,indy,:]-THTmixed,cmap='coolwarm',vmin=-0.15,vmax=0.15) 
			plt.colorbar(s,ax=ax[1])
			ax[1].tick_params(axis='both',labelbottom=False)
			ax[1].set_title(r'$\theta-\theta_{mixed}$ (K)',loc='right')
			s = ax[2].pcolormesh( X/1000,Z/ABLH,(RV[:,indy,:]-RVmixed)*1000,cmap='BrBG',vmin=-1,vmax=1) 
			plt.colorbar(s,ax=ax[2])
			ax[2].set_title(r'$r_v-r_{v,mixed}$ (g/kg) and objects',loc='right')
			ax[2].set_xlabel('X (km)')
			for axe in ax:
				axe.set_ylim([0,1.2])
				axe.set_xlim([X1,X2])
				axe.set_aspect(2.0)
				axe.set_ylabel('z/zi')
			fig.savefig(path_save2+'CS1_'+case+'_XZ_tht_rv_j'+str(indy)+'.png')
			# W and scalar fluctuations
			fig, ax = plt.subplots(3,1,figsize = (6,7),constrained_layout=True,dpi=dpi)
			s = ax[0].pcolormesh( X/1000,Z/ABLH,W[:,indy,:],cmap='coolwarm',vmin=-1,vmax=1) 
			plt.colorbar(s,ax=ax[0],orientation='vertical')
			ax[0].tick_params(axis='both',labelbottom=False)
			ax[0].set_title('W (m/s)',loc='right')
			if SVT_type=='SVT':
				s = ax[1].pcolormesh( X/1000,Z/ABLH,(SV1-SV1m)[:,indy,:],cmap='Reds',vmin=0.01,vmax=10,norm="log") 
				plt.colorbar(s,ax=ax[1])
			ax[1].tick_params(axis='both',labelbottom=False)
			ax[1].set_title(r"$sv_1'$",loc='right')
			s = ax[2].pcolormesh( X/1000,Z/ABLH,(SV3-SV3m)[:,indy,:],cmap='Blues',vmin=0.1,vmax=500,norm="log") 
			plt.colorbar(s,ax=ax[2])
			ax[2].set_xlabel('X (km)')
			ax[2].set_title(r"$sv_3'$",loc='right')
			for axe in ax:
				axe.set_ylim([0,1.2])
				axe.set_xlim([X1,X2])
				axe.set_aspect(2.0)
				axe.set_ylabel('z/zi')
			fig.savefig(path_save2+'CS1_'+case+'_XZ_sv_j'+str(indy)+'.png')
		# Spatial view YZ
		if 'YZ' in L_plot:
			print('			-YZ with objects')
			#indx = np.argmin(np.abs(X.values-atx*1000))
			indx = atx
			stepy,stepz = 2,2 # number of skiped cell to draw arrows
			fig, ax = plt.subplots(1,1,figsize = (5,6),constrained_layout=True,dpi=dpi)	
			ax.set_title('Objects and wind - mean wind',loc='right')
			ax.pcolormesh( Y/1000,Z/ABLH,global_objects[:,:,indx],cmap=c.ListedColormap(['white','r','purple','g','grey']))
			Q = ax.quiver(Y[::stepy]/1000,Z[::stepz]/ABLH,(V-Vm)[::stepz,::stepy,indx],W[::stepz,::stepy,indx],
					angles='uv',pivot='middle',headwidth=2,headaxislength=4,scale=30)
			ax.quiverkey(Q, 0.9, 0.05, 0.5, '0.5 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
			ax.set_ylim([0,1.2])
			ax.set_xlim([Y[0]/1000,Y[-1]/1000])
			ax.set_aspect(2.0)
			ax.set_xlabel('Y (km)')
			fig.savefig(path_save2+'CS1_'+case+'_YZ_i'+str(indx)+'.png')
			# thtv,tht and rv
			fig, ax = plt.subplots(3,1,figsize = (5,12),constrained_layout=True,dpi=dpi)
			s = ax[0].pcolormesh( Y/1000,Z/ABLH,(THTV-THTVm)[:,:,indx],cmap='coolwarm',vmin=-0.15,vmax=0.15) 
			plt.colorbar(s,ax=ax[0])
			ax[0].tick_params(axis='both',labelbottom=False)
			ax[0].set_title(r'$\theta_v-<\theta_v>$ (K)',loc='right')
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLH,xr.where(is_up,1,0)[:,:,indx],levels=[0.55],colors=['r'],linewidths=1.0)
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLH,xr.where(is_sub,1,0)[:,:,indx],levels=[0.55],colors=['purple'],linewidths=1.0)
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLH,xr.where(is_down,1,0)[:,:,indx],levels=[0.55],colors=['g'],linewidths=1.0)
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLH,xr.where(is_env,1,0)[:,:,indx],levels=[0.55],colors=['grey'],linewidths=1.0)
			s = ax[1].pcolormesh( Y/1000,Z/ABLH,THT[:,:,indx]-THTmixed,cmap='coolwarm',vmin=-0.15,vmax=0.15) 
			plt.colorbar(s,ax=ax[1])
			ax[1].tick_params(axis='both',labelbottom=False)
			ax[1].set_title(r'$\theta-\theta_{mixed}$ (K)',loc='right')
			s = ax[2].pcolormesh( Y/1000,Z/ABLH,(RV[:,:,indx]-RVmixed)*1000,cmap='BrBG',vmin=-1,vmax=1) 
			plt.colorbar(s,ax=ax[2])
			ax[2].set_title(r'$r_v-r_{v,mixed}$ (g/kg) and objects',loc='right')
			ax[2].set_xlabel('Y (km)')
			for axe in ax:
				axe.set_ylim([0,1.2])
				axe.set_xlim([Y[0]/1000,Y[-1]/1000])
				axe.set_aspect(2.0)
				axe.set_ylabel('z/zi')
			fig.savefig(path_save2+'CS1_'+case+'_YZ_tht_rv_i'+str(indx)+'.png')
			# W and scalar fluctuations
			fig, ax = plt.subplots(3,1,figsize = (5,12),constrained_layout=True,dpi=dpi)
			s = ax[0].pcolormesh( Y/1000,Z/ABLH,W[:,:,indx],cmap='coolwarm',vmin=-1,vmax=1) 
			plt.colorbar(s,ax=ax[0],orientation='vertical')
			ax[0].tick_params(axis='both',labelbottom=False)
			ax[0].set_title('W (m/s)',loc='right')
			if SVT_type=='SVT':
				s = ax[1].pcolormesh( Y/1000,Z/ABLH,(SV1-SV1m)[:,:,indx],cmap='Reds',vmin=0.01,vmax=10,norm="log") 
				plt.colorbar(s,ax=ax[1])
			ax[1].tick_params(axis='both',labelbottom=False)
			ax[1].set_title(r"$sv_1'$",loc='right')
			s = ax[2].pcolormesh( Y/1000,Z/ABLH,(SV3-SV3m)[:,:,indx],cmap='Blues',vmin=0.1,vmax=500,norm="log") 
			plt.colorbar(s,ax=ax[2])
			ax[2].set_xlabel('Y (km)')
			ax[2].set_title(r"$sv_3'$",loc='right')
			for axe in ax:
				axe.set_ylim([0,1.2])
				axe.set_xlim([Y[0]/1000,Y[-1]/1000])
				axe.set_aspect(2.0)
				axe.set_ylabel('z/zi')
			fig.savefig(path_save2+'CS1_'+case+'_YZ_sv_i'+str(indx)+'.png')
		# Object fraction
		if 'obj_frac' in L_plot:
			print('			-object fraction')
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			ax.plot( F_up,Z/ABLH,c='red',label='updrafts')
			ax.plot( F_sub,Z/ABLH,c='purple',label='sub. shells')
			ax.plot( F_down,Z/ABLH,c='green',label='downdrafts')
			ax.plot( F_env,Z/ABLH,c='grey',label='env')
			ax.plot( F_up + F_sub + F_down + F_env,Z/ABLH,c='k',label='total')
			ax.set_title('Object area cover over entire domain ('+case+')')
			ax.set_ylabel('z/zi')
			ax.legend(loc='upper right')
			ax.set_ylim([0,1.2])
			ax.set_xlim([0,0.4])
			ax.grid()
			fig.savefig(path_save2+'CS1_'+case+'_cover_fraction.png')
		
		if 'test' in L_plot:	
			# tests
			# je garde pour avoir une vision du comportement de Integ_minNEW quand S1_SVT a tourné
			"""13/02/24 A voir si toujours utile 
			"""
			gamma = 0.005
			m = 0.5
			SV3,SV3m = dsCS1[case].SV3,dsCS1[case].SV3m
			SV3_std = SV3.std(dim=['ni','nj'])
			sv3_prime = SV3-SV3m
		
			sig_min = Integ_min3(X,Z_w,SV3,gamma)
			sig_max = xr.where(sig_min > SV3_std,sig_min,SV3_std)
			
			# looking at the turbulent condition for C10
			if False:
				sig_max = m*sig_max
				turb3 = xr.where( sv3_prime > sig_max, 1,0)
				# filter at a specific position
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				ax.plot( sig_min,Z/ABLH,c='k',label='sigmin')
				ax.plot( sv3_prime[:,20,245],Z/ABLH,c='b',label='sv3_prime at i,j=245,20')
				ax.plot( SV3_std,Z/ABLH,c='r',label='SV3_std')
				ax.plot( sig_max,Z/ABLH,c='g',label='sigmax',ls='--')
				ax.set_ylabel('z/zi')
				ax.legend(loc='upper right')
				ax.set_ylim([0,1.2])
				ax.grid()
				# global coverage
				Lsv3_pos = xr.where( sv3_prime > 0,1,0)
				Lsv3_neg = xr.where( sv3_prime < 0,1,0)
				Lw_pos = xr.where( W > 0,1,0)
				Lw_neg = xr.where( W < 0,1,0)
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				ax.plot( turb3.mean(dim=['ni','nj']),Z/ABLH,c='k',label='comes from ABLH top')
				ax.plot( (turb3*Lsv3_pos*Lw_pos).mean(dim=['ni','nj']),Z/ABLH,c='grey',label='new env')
				ax.plot( (turb3*Lsv3_pos*Lw_neg).mean(dim=['ni','nj']),Z/ABLH,c='g',label='new down')
				ax.plot( dsCS1[case].is_turb3.mean(dim=['ni','nj']),Z/ABLH,c='b',label='is_turb3 actuel',ls='--')
				ax.plot( dsCS1[case].is_env.mean(dim=['ni','nj']),Z/ABLH,c='grey',label='is_env actuel',ls='--')
				ax.plot( dsCS1[case].is_down.mean(dim=['ni','nj']),Z/ABLH,c='g',label='is_down actuel',ls='--')
				ax.set_ylabel('z/zi')
				ax.legend(loc='upper right')
				ax.set_ylim([0,1.2])
				ax.grid()
		
			# tests of sensibility to m
			if True:
				atx = 269
				# 	m=1
				sig_max = 1*sig_max
				sig_max = Complete_dim_like([sig_max],SV3)
				turb3 = xr.where( sv3_prime > sig_max, 1,0)
				fig,ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				ax.pcolormesh(Y/1000,Z/ABLH,xr.where(turb3,1,0)[:,:,atx],cmap='bwr',vmin=0,vmax=1)
				ax.set_title('with m=1')
				ax.set_xlabel('Y')
				ax.set_ylabel('z/zi')
				ax.set_ylim([0,1.2])
				# 	m=2
				sig_max = 0.5*sig_max
				sig_max = Complete_dim_like([sig_max],SV3)
				turb2 = xr.where( sv3_prime > sig_max, 1,0)
				fig,ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				ax.pcolormesh(Y/1000,Z/ABLH,xr.where(turb2,1,0)[:,:,atx],cmap='bwr',vmin=0,vmax=1)
				ax.set_title('with m=0.5')
				ax.set_xlabel('Y')
				ax.set_ylabel('z/zi')
				ax.set_ylim([0,1.2])
			
			
		if 'uw_profile' in L_plot:
			print('			-uw profiles') # resolved only
			somme = uw_up_p+uw_sub_p+uw_down_p+uw_env_p
			part_up = mean_vertical_contrib(uw_up_p,uw_mean,indzi).values
			part_sub = mean_vertical_contrib(uw_sub_p,uw_mean,indzi).values
			part_down = mean_vertical_contrib(uw_down_p,uw_mean,indzi).values
			part_env = mean_vertical_contrib(uw_env_p,uw_mean,indzi).values
			obj_over_all = mean_vertical_contrib(somme,uw_mean,indzi).values
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			ax.plot( uw_up_p/u_star**2,Z/ABLH,c='red',label='updrafts ('+str(np.round(part_up*100,1))+'%)')
			#ax.fill_betweenx(Z/ABLH, (uw_up_p-std_uw_up)/u_star**2,(uw_up_p+std_uw_up)/u_star**2, color='red',alpha=0.5)
			ax.plot( uw_sub_p/u_star**2,Z/ABLH,c='purple',label='sub. shells ('+str(np.round(part_sub*100,1))+'%)')
			#ax.fill_betweenx(Z/ABLH, (uw_sub_p-std_uw_sub)/u_star**2,(uw_sub_p+std_uw_sub)/u_star**2, color='purple',alpha=0.5)
			ax.plot( uw_down_p/u_star**2,Z/ABLH,c='green',label='downdrafts ('+str(np.round(part_down*100,1))+'%)')
			ax.plot( uw_env_p/u_star**2,Z/ABLH,c='grey',label='env ('+str(np.round(part_env*100,1))+'%)')
			ax.plot( somme/u_star**2,Z/ABLH,c='black',label='all ('+str(np.round(obj_over_all*100,1))+'%)')
			ax.plot( uw_mean/u_star**2,Z/ABLH,c='k',label='mean',ls='--')
			ax.set_ylim([0,1.2])
			ax.grid()
			ax.set_title(case)
			ax.set_xlim([-1.5,0.3])
			ax.legend()
			ax.set_xlabel(r"u'w'/$u^{*2}$") #  (m2.s-2)
			ax.set_ylabel('z/zi')
			print('			 coherent/all =',obj_over_all*100,'%')
			fig.savefig(path_save2+'CS1_'+case+'_uw_profiles.png')
		if 'wtht_profile' in L_plot:
			print('			-wtht profiles') # resolved only
			somme = wtht_up_p+wtht_sub_p+wtht_down_p+wtht_env_p
			part_up = mean_vertical_contrib(wtht_up_p,wtht_mean,indzi)
			part_sub = mean_vertical_contrib(wtht_sub_p,wtht_mean,indzi)
			part_down = mean_vertical_contrib(wtht_down_p,wtht_mean,indzi)
			part_env = mean_vertical_contrib(wtht_env_p,wtht_mean,indzi)
			obj_over_all = mean_vertical_contrib(somme,wtht_mean,indzi)
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			ax.plot( wtht_up_p/Q0,Z/ABLH,c='red',label='updrafts ('+str(np.round(part_up*100,1))+'%)')
			ax.plot( wtht_sub_p/Q0,Z/ABLH,c='purple',label='sub. shells ('+str(np.round(part_sub*100,1))+'%)')
			ax.plot( wtht_down_p/Q0,Z/ABLH,c='green',label='downdrafts ('+str(np.round(part_down*100,1))+'%)')
			ax.plot( wtht_env_p/Q0,Z/ABLH,c='grey',label='env ('+str(np.round(part_env*100,1))+'%)')
			ax.plot( somme/Q0,Z/ABLH,c='black',label='all ('+str(np.round(obj_over_all*100,1))+'%)')
			ax.plot( wtht_mean/Q0,Z/ABLH,c='k',label='mean',ls='--')
			ax.set_ylim([0,1.2])
			ax.grid()
			ax.set_title(case)
			if case=='cold':
				ax.set_xlim([-4,1])
			else:
				ax.set_xlim([-1.5,1])
			ax.legend()
			ax.set_xlabel(r"w'$\theta$'/Q0") #  (K.m.s-1)
			ax.set_ylabel('z/zi')
			print('			 coherent/all =',obj_over_all*100,'%')
			fig.savefig(path_save2+'CS1_'+case+'_wtht_profiles.png')
		if 'wthtv_profile' in L_plot:
			print('			-wthtv profiles') # resolved only
			somme = wthtv_up_p+wthtv_sub_p+wthtv_down_p+wthtv_env_p
			beta = (np.amax(wthtv_mean) - np.amin(wthtv_mean))  #/Qv0
			part_up = mean_vertical_contrib(wthtv_up_p,wthtv_mean,indzi)
			part_sub = mean_vertical_contrib(wthtv_sub_p,wthtv_mean,indzi)
			part_down = mean_vertical_contrib(wthtv_down_p,wthtv_mean,indzi)
			part_env = mean_vertical_contrib(wthtv_env_p,wthtv_mean,indzi)
			obj_over_all = mean_vertical_contrib(somme,wthtv_mean,indzi)
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			ax.plot( wthtv_up_p/Qv0,Z/ABLH,c='red',label='updrafts ('+str(np.round(part_up*100,1))+'%)')
			ax.plot( wthtv_sub_p/Qv0,Z/ABLH,c='purple',label='sub. shells ('+str(np.round(part_sub*100,1))+'%)')
			ax.plot( wthtv_down_p/Qv0,Z/ABLH,c='green',label='downdrafts ('+str(np.round(part_down*100,1))+'%)')
			ax.plot( wthtv_env_p/Qv0,Z/ABLH,c='grey',label='env ('+str(np.round(part_env*100,1))+'%)')
			ax.plot( somme/Qv0,Z/ABLH,c='black',label='all ('+str(np.round(obj_over_all*100,1))+'%)')
			ax.plot( wthtv_mean/Qv0,Z/ABLH,c='k',label='mean',ls='--')
			ax.set_ylim([0,1.2])
			ax.grid()
			ax.set_title(case+r', $\beta$='+str(np.round(beta.values,6)))
			ax.set_xlim([-0.3,1])
			ax.legend()
			ax.set_xlabel(r"w'$\theta_v$'/Qv0") #  (K.m.s-1)
			ax.set_ylabel('z/zi')
			print('			 coherent/all =',obj_over_all*100,'%')	
			#print('			 coherent/all = CROSSING 0')	
			fig.savefig(path_save2+'CS1_'+case+'_wthtv_profiles.png')
		if 'wrv_profile' in L_plot:
			print('			-wrv profiles') # resolved only
			somme = wrv_up_p+wrv_sub_p+wrv_down_p+wrv_env_p
			part_up = mean_vertical_contrib(wrv_up_p,wrv_mean,indzi)
			part_sub = mean_vertical_contrib(wrv_sub_p,wrv_mean,indzi)
			part_down = mean_vertical_contrib(wrv_down_p,wrv_mean,indzi)
			part_env = mean_vertical_contrib(wrv_env_p,wrv_mean,indzi)
			obj_over_all = mean_vertical_contrib(somme,wrv_mean,indzi)
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			ax.plot( wrv_up_p/E0,Z/ABLH,c='red',label='updrafts ('+str(np.round(part_up*100,1))+'%)')
			ax.plot( wrv_sub_p/E0,Z/ABLH,c='purple',label='sub. shells ('+str(np.round(part_sub*100,1))+'%)')
			ax.plot( wrv_down_p/E0,Z/ABLH,c='green',label='downdrafts ('+str(np.round(part_down*100,1))+'%)')
			ax.plot( wrv_env_p/E0,Z/ABLH,c='grey',label='env ('+str(np.round(part_env*100,1))+'%)')
			ax.plot( somme/E0,Z/ABLH,c='black',label='all ('+str(np.round(obj_over_all*100,1))+'%)')
			ax.plot( wrv_mean/E0,Z/ABLH,c='k',label='mean',ls='--')
			ax.set_ylim([0,1.2])
			ax.grid()
			ax.set_title(case)
			ax.set_xlim([-0.3,1.1])
			ax.legend()
			ax.set_xlabel(r"w'$r_v$'/E0") #  (K.m.s-1)
			ax.set_ylabel('z/zi')
			print('			 coherent/all =',obj_over_all*100,'%')
			fig.savefig(path_save2+'CS1_'+case+'_wrv_profiles.png')
	
def plots_nice_flx_ref_CS1(Z,dsCS1,dsref,path_save,dpi):
	""" Nicer plots than 'plots_ref_CS1' for uw and wthtv fluxes
		2 figure with 2 plots each
		
		INPUTS : 
			- Z : vertical coordinate (without halo)
			- dsCS1 : dataset with conditional sampling filter, ouput of 'build_CS1' from module_building_files
			- dsref : 000 MNH file opened with 'Open_LES_MEAN' from module_building_files
			- path_save : where to save figures
			- dpi : for the saved figures
		
	"""
	fig, ax = plt.subplots(1,2,figsize = (8,5),constrained_layout=True,dpi=dpi) # uw
	fig2, ax2 = plt.subplots(1,2,figsize = (8,5),constrained_layout=True,dpi=dpi) # wthtv
	
	indt = -1 # t=+3h
	
	for i,case in enumerate(['cold','warm']):
	
		# Conditional sampling related
		is_turb = dsCS1[case].is_turb
		is_up = dsCS1[case].is_up
		is_sub = dsCS1[case].is_sub
		is_down = dsCS1[case].is_down
		is_env = dsCS1[case].is_env
		global_objects = dsCS1[case].global_objects
		TURB_COND = dsCS1[case].attrs['Turb_cond']
		RV_DIFF = dsCS1[case].attrs['RV_DIFF']
	
		# Instantaneous fields
		U,Um = dsCS1[case].UT,dsCS1[case].UTm
		W,Wm = dsCS1[case].WT,dsCS1[case].WTm
		THTV,THTVm = dsCS1[case].THTV,dsCS1[case].THTVm
		thtv_fluc = THTV - THTVm
		u_fluc = U - Um
		w_fluc = W - Wm
		uw = u_fluc*w_fluc
		wthtv = w_fluc*thtv_fluc
		uw_mean= uw.mean(dim=['ni','nj']) # mean resolved uw
		uw_up_p		,uw_sub_p	,uw_down_p	,uw_env_p 	= compute_flx_contrib(uw,[is_up,is_sub,is_down,is_env],meanDim=['ni','nj'])
		wthtv_mean= wthtv.mean(dim=['ni','nj']) # mean resolved wthtv
		wthtv_up_p,wthtv_sub_p,wthtv_down_p,wthtv_env_p = compute_flx_contrib(wthtv,[is_up,is_sub,is_down,is_env],meanDim=['ni','nj'])
		
		# mean fields
		THT_case = dsref[case]['nomean']['Mean']['MEAN_TH'][indt,:]
		THTV_case = dsref[case]['nomean']['Mean']['MEAN_THV'][indt,:]
		ABLH = Z[THTV_case.differentiate('level_les').argmax().values].values
		indzi = nearest(Z.values,1.1*ABLH) # looking at 1.1*zi to account for overshoots
		RES_WTHV_case = dsref[case]['nomean']['Resolved']['RES_WTHV'][indt,:]
		SBG_WTH_case = dsref[case]['nomean']['Subgrid']['SBG_WTHL'][indt,:]
		SBG_WRT_case = dsref[case]['nomean']['Subgrid']['SBG_WRT'][indt,:]
		SBG_WTHV_case = ( THTV_case/THT_case*SBG_WTH_case + 0.61*THT_case*SBG_WRT_case )
		WTHV_total = RES_WTHV_case + SBG_WTHV_case
		RES_UW_case = dsref[case]['nomean']['Resolved']['RES_WU'][indt,:]
		SBG_UW_case = dsref[case]['nomean']['Subgrid']['SBG_WU'][indt,:]
		UW_total = RES_UW_case + SBG_UW_case
		# surface values		
		Qv0 = dsCS1[case]['Qv0'].values
		u_star = dsCS1[case]['u_star'].values
		
		# uw 
		somme = uw_up_p+uw_sub_p+uw_down_p #  +uw_env_p
		part_up = mean_vertical_contrib(uw_up_p,uw_mean,indzi).values
		part_sub = mean_vertical_contrib(uw_sub_p,uw_mean,indzi).values
		part_down = mean_vertical_contrib(uw_down_p,uw_mean,indzi).values
		part_env = mean_vertical_contrib(uw_env_p,uw_mean,indzi).values
		obj_over_all = mean_vertical_contrib(somme,uw_mean,indzi).values
		ax[i].plot( uw_up_p/u_star**2,Z/ABLH,c='red',label='up ('+str(np.round(part_up*100,1))+'%)')
		ax[i].plot( uw_sub_p/u_star**2,Z/ABLH,c='purple',label='ss ('+str(np.round(part_sub*100,1))+'%)')
		ax[i].plot( uw_down_p/u_star**2,Z/ABLH,c='green',label='down ('+str(np.round(part_down*100,1))+'%)')
		#ax[i].plot( uw_env_p/u_star**2,Z/ABLH,c='grey',label='env ('+str(np.round(part_env*100,1))+'%)')
		ax[i].plot( somme/u_star**2,Z/ABLH,c='black',label='all ('+str(np.round(obj_over_all*100,1))+'%)')
		ax[i].plot( UW_total/u_star**2,Z/ABLH,c='k',label='mean',ls='--')
		ax[i].set_ylim([0,1.2])
		ax[i].grid()
		ax[i].set_xlim([-1.5,0.5])
		ax[i].legend(loc='upper left')
		# wthtv
		somme = wthtv_up_p+wthtv_sub_p+wthtv_down_p # +wthtv_env_p
		print(case,'QV0=',Qv0,'min(wthtv)=',np.amin(WTHV_total.values))
		part_up = mean_vertical_contrib(wthtv_up_p,wthtv_mean,indzi).values
		part_sub = mean_vertical_contrib(wthtv_sub_p,wthtv_mean,indzi).values
		part_down = mean_vertical_contrib(wthtv_down_p,wthtv_mean,indzi).values
		part_env = mean_vertical_contrib(wthtv_env_p,wthtv_mean,indzi).values
		obj_over_all = mean_vertical_contrib(somme,wthtv_mean,indzi).values
		ax2[i].plot( wthtv_up_p/Qv0,Z/ABLH,c='red',label='up ('+str(np.round(part_up*100,1))+'%)')
		ax2[i].plot( wthtv_sub_p/Qv0,Z/ABLH,c='purple',label='ss ('+str(np.round(part_sub*100,1))+'%)')
		ax2[i].plot( wthtv_down_p/Qv0,Z/ABLH,c='green',label='down ('+str(np.round(part_down*100,1))+'%)')
		#ax2[i].plot( wthtv_env_p/Qv0,Z/ABLH,c='grey',label='env ('+str(np.round(part_env*100,1))+'%)')
		ax2[i].plot( somme/Qv0,Z/ABLH,c='black',label='all ('+str(np.round(obj_over_all*100,1))+'%)')
		ax2[i].plot( WTHV_total/Qv0,Z/ABLH,c='k',label='mean',ls='--')
		ax2[i].set_ylim([0,1.2])
		ax2[i].grid()
		ax2[i].set_xlim([-0.3,1.1])
		ax2[i].legend(loc='upper right')
		# format uw plot
		ax[0].set_title(r'refC',loc='right')
		ax[0].set_ylabel(r'z/$z_i$')
		ax[0].set_xlabel(r"$<\~ u \~ w >/u^{*2}$")
		ax[1].set_title(r'refW',loc='right')
		ax[1].set_xlabel(r"$<\~ u \~ w >/u^{*2}$")
		ax[0].xaxis.label.set_fontsize(13)
		ax[0].yaxis.label.set_fontsize(13)
		ax[1].xaxis.label.set_fontsize(13)
		ax[1].yaxis.label.set_fontsize(13)
		ax[0].xaxis.set_major_locator(MultipleLocator(0.5))
		ax[0].grid(True,'major')
		ax[1].xaxis.set_major_locator(MultipleLocator(0.5))
		ax[1].grid(True,'major')
		#ax[1].tick_params(axis='both',labelleft=False)
		# format wthtv plot
		ax2[0].set_title(r'refC',loc='right')
		ax2[0].set_ylabel(r'z/$z_i$')
		ax2[0].set_xlabel(r"$<\~ w \~ \theta_v >/ Q_v^*$")
		ax2[1].set_title(r'refW',loc='right')
		ax2[1].set_xlabel(r"$<\~ w \~ \theta_v >/ Q_v^*$")
		#ax2[1].tick_params(axis='both',labelleft=False)		
	# saving		
	fig.savefig(path_save+'ref_uw_nice_flx.png')	
	fig2.savefig(path_save+'ref_wthtv_nice_flx.png')	

def plot_CS1_S1(X,Y,Z,Z_w,dsref,dsflx,dsCS1,SVT_type,SEUIL_ML,L_atziXY,atyXZ,L_var,L_plot,L_atX,Xloc_forYZ,Tloc_forYZ,K,PLOT_CONTRIB,PLOT_REF,path_save,name,dpi):
	"""description à adapter pour le cas S1

	- bar de SST sous les figures avec X en coordonnées
	
	This procedure is plotting slices of ref simulations with coherent structures
	
	INPUTS
		- X,Y,Z,Z_w	: are dimensions of MNH domain
		- dsflx 	: contains fluxes (total and sgs)
		- dsCS1 	: is the instantaneous file with the CS1 masks (from 'build_CS1')
		- SVT_type 	: type of tracer to use, RV or SVT
		- SEUIL_ML 	: is the threshold to detect the mixed layer on thtv
		- atziXY	: is the fraction of zi at which XY plots are shown
		- atyXZ 	: is the Y location of the XZ slice
		- L_var 	: is the list of variable plotted (for XY plots and profiles)
			available : 'U','W','RVT','THT','THTV'
		- L_plot 	: is the list of the desired plots
			* 'YZ' : same as 'XZ' but for 'YZ' plane
			* 'obj_frac' 	: show area coverage fraction as profiles for each objects
			* 'obj_movie'	: build a movie of the coherent structures
			* fluxes		: plot profiles of fluxes decomposed with the conditional sampling at selected X locations
		- L_atX 		: list of X location where to plot area coverage and profiles of VAR
		- Xloc_forYZ	: list of X location where to plot YZ slices
		- PLOT_CONTRIB 	: boolean, plot contribution of each structures
		- PLOT_REF 	: boolean, plot reference profiles
		- path_save 	: is the abs path to store figures
		- dpi 			: is dot per inches for figures
		
		
	OUTPUTS
		- plots from L_plot input
		
		Note : 
			* the mean used here is in time (over the all OUT files) and in Y
			* 'updrafts' objects can overlap. If so, the most concentrated one is taken 
				(for SV1 and SV4 only, SV3 is no injected in the same manner)
				
	THIS PROCEDURE NEEDS TO BE MODIFIED TO GET BETTER REPRODUCTIBILITY
	> le changement de type de plot des profils de uw
				
	"""	
	print('			- pre-computations ...')
	
	indt = -1 # we look at last instant
	indt_c = -1
	indt_w = -1
	
	Ntime = dsCS1.time.shape[0]
	Time = np.arange(7200,10800,30)
	#indzi = np.argmin(np.abs(Z.values-1.1*ABLH)) # looking at 1.1*zi to account for overshoots
	TURB_COND = dsCS1.attrs['Turb_cond']
	RV_DIFF = dsCS1.attrs['RV_DIFF']
	dz = np.zeros(Z_w.shape)
	dz[:-1] = Z_w[1:].values - Z_w[:-1].values
	dz[-1] = dz[-2]
	U,Um = dsCS1.UT,dsCS1.UTm
	V,Vm = dsCS1.VT,dsCS1.VTm
	W,Wm = dsCS1.WT,dsCS1.WTm
	THT,THTm = dsCS1.THT,dsCS1.THTm
	THTV,THTVm = dsCS1.THTV,dsCS1.THTVm
	RV,RVm = dsCS1.RVT,dsCS1.RVTm
	E,Em = dsCS1.E,dsCS1.Em
	SV1,SV1m = dsCS1.SV1,dsCS1.SV1m
	SV3,SV3m = dsCS1.SV3,dsCS1.SV3m
	if 'SV4' in dsCS1.keys():
		SV4,SV4m = dsCS1.SV4,dsCS1.SV4m
		sv4_fluc = SV4 - SV4m
	gTHT = THTVm[0,:,0,:].differentiate('level')
	ABLH = Z[gTHT.argmax(dim='level')]
	ABLHm = 600 # =ABLH.mean().values
	#indzi = np.array([np.argmin(np.abs(Z.values-1.1*ABLH[x].values)) for x in range(len(X))] ) # to be //
	#l_indx = [nearest(X.values,atx*1000) for atx in L_atX]
	l_indx = L_atX
	#l_indxYZ = [nearest(X.values,atx*1000) for atx in Xloc_forYZ]
	l_indxYZ = Xloc_forYZ
	indzi = nearest(Z.values,1.1*ABLHm)
	
	rv_fluc = RV - RVm
	tht_fluc = THT - THTm
	thtv_fluc = THTV - THTVm
	u_fluc = U - Um
	v_fluc = V - Vm
	w_fluc = W - Wm
	sv1_fluc = SV1 - SV1m
	sv3_fluc = SV3 - SV3m
	uw = u_fluc*w_fluc
	wtht = w_fluc*tht_fluc
	wthtv = w_fluc*thtv_fluc
	wrv = w_fluc*rv_fluc
	
	Ones = xr.ones_like(U)
	Zeros = xr.zeros_like(U)
	is_turb = dsCS1.is_turb
	is_up = xr.where( dsCS1.global_objects==1,Ones,Zeros)
	is_sub = xr.where( dsCS1.global_objects==2,Ones,Zeros )
	is_down = xr.where( dsCS1.global_objects==3,Ones,Zeros )
	is_env = xr.where( dsCS1.global_objects==4,Ones,Zeros )
	is_up2 = xr.where( dsCS1.global_objects==5,Ones,Zeros )
	is_sub2 = xr.where( dsCS1.global_objects==6,Ones,Zeros )
	global_mask = dsCS1.global_objects
	
	normX = 4 # km
	indnormX = nearest(X.values,normX*1000)
	u_star = 0.211 #  =dsCS1.u_star[indnormX].values
	Qv0 = dsCS1.Qv0[indnormX].values
		
	path_save2 = path_save+TURB_COND
	if TURB_COND=='C10':
		mCS = dsCS1.attrs['mCS']
		gammaRV = dsCS1.attrs['gammaRv']
		path_save2 = path_save2+'_m'+str(mCS)+'_g'+str(gammaRV*100)
	elif TURB_COND=='ITURB2':
		gammaTurb2 = dsCS1.attrs['gammaTurb2']
		path_save2 = path_save2+'_g'+str(gammaTurb2*100)
	elif TURB_COND=='ITURB':
		gammaTurb1 = dsCS1.attrs['gammaTurb1']
		path_save2 = path_save2+ '_g'+str(gammaTurb1*100)
	elif TURB_COND=='EC':
		gammaEc = dsCS1.attrs['gammaEc']
		path_save2 = path_save2 +'_g'+str(gammaEc*100)
	if SVT_type=='RV':
		path_save2 = path_save2+'_'+RV_DIFF+'_'+SVT_type+'/'
	else:
		path_save2 = path_save2+'_'+SVT_type+'/'
	path_save2 = path_save2 + name+ '/'
	if not os.path.isdir(path_save2): # creat folder if it doesnt exist
		os.makedirs(path_save2)
			
	# Flux profiles		
	uw_mean= uw.mean(dim=['time','nj']).compute()
	uw_mean_tot = dsflx.FLX_UW
	uw_up_p,uw_sub_p,uw_up_p2,uw_sub_p2,uw_down_p,uw_env_p = compute_flx_contrib(uw,[is_up,is_sub,is_up2,is_sub2,is_down,is_env],meanDim=['time','nj'])
	wthtv_mean= wthtv.mean(dim=['time','nj']).compute()
	wthtv_up_p,wthtv_sub_p,wthtv_up_p2,wthtv_sub_p2,wthtv_down_p,wthtv_env_p = compute_flx_contrib(wthtv,[is_up,is_sub,is_up2,is_sub2,is_down,is_env],meanDim=['time','nj'])	

	# area cover
	F_up = is_up.mean(dim=['time','nj']) 
	F_sub = is_sub.mean(dim=['time','nj']) 
	F_down = is_down.mean(dim=['time','nj'])  
	F_env = is_env.mean(dim=['time','nj']) 
	F_up2 = is_up2.mean(dim=['time','nj']) 
	F_sub2 = is_sub2.mean(dim=['time','nj']) 
	unity = xr.ones_like(F_up)
	
	# This is for Z integrated plots of object cover
	part_up,part_sub,part_down,part_env,part_up2,part_sub2 = xr.zeros_like(X),xr.zeros_like(X),xr.zeros_like(X),xr.zeros_like(X),xr.zeros_like(X),xr.zeros_like(X)
	part_up = mean_vertical_contrib(F_up,unity,indzi)
	part_sub = mean_vertical_contrib(F_sub,unity,indzi)
	part_up2 = mean_vertical_contrib(F_up2,unity,indzi)
	part_sub2 = mean_vertical_contrib(F_sub2,unity,indzi)
	part_down = mean_vertical_contrib(F_down,unity,indzi)
	part_env = mean_vertical_contrib(F_env,unity,indzi)

	# profiles of uw at selected X points
	if 'uw_profile' in L_plot:
		print('			- profiles uw at i='+str(l_indx))
		flx_summ = (uw_up_p + uw_sub_p + uw_up_p2 + uw_sub_p2 + uw_down_p + uw_env_p ).isel(ni=l_indx)
		flx_part_up 	= mean_vertical_contrib((uw_up_p).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
		flx_part_sub 	= mean_vertical_contrib((uw_sub_p).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
		flx_part_up2 	= mean_vertical_contrib((uw_up_p2).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
		flx_part_sub2 	= mean_vertical_contrib((uw_sub_p2).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
		flx_part_down 	= mean_vertical_contrib((uw_down_p).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
		flx_part_env 	= mean_vertical_contrib((uw_env_p).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
		flx_obj_over_all = mean_vertical_contrib(flx_summ,(uw_mean).isel(ni=l_indx),indzi).values
		for i,indx in enumerate(l_indx):
			norm = u_star**2
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			if PLOT_REF: # if True, plots the uw profils from references
				uw_c = dsref['cold']['nomean']['Resolved']['RES_WU'][indt_c,:] + dsref['cold']['nomean']['Subgrid']['SBG_WU'][indt_c,:] 
				uw_w = dsref['warm']['nomean']['Resolved']['RES_WU'][indt_c,:] + dsref['warm']['nomean']['Subgrid']['SBG_WU'][indt_c,:]
				gTHTV_w = ( dsref['warm']['nomean']['Mean']['MEAN_TH']
					* (1+1.61*dsref['warm']['nomean']['Mean']['MEAN_RV'])
					/ (1+dsref['warm']['nomean']['Mean']['MEAN_RV']) )[indt_w].differentiate('level_les')
				gTHTV_c = ( dsref['cold']['nomean']['Mean']['MEAN_TH']
					* (1+1.61*dsref['cold']['nomean']['Mean']['MEAN_RV'])
					/ (1+dsref['cold']['nomean']['Mean']['MEAN_RV']) )[indt_c].differentiate('level_les')
				zi_c,      zi_w     = ( Z[gTHTV_c.argmax('level_les').values].values, 
							Z[gTHTV_w.argmax('level_les').values].values )
				ax.plot( -uw_c/uw_c[0],Z/zi_c, c='b', label='mean refC')
				ax.plot( -uw_w/uw_w[0],Z/zi_w, c='r', label='mean refW')
			if PLOT_CONTRIB: # if False, plot only the total contribution of coherent structures
				ax.plot( uw_up_p[:,indx]/norm,Z/ABLHm	,c='red'	,label='updrafts ('		+str(np.round(flx_part_up[i]*100,1))	+'%)')
				ax.plot( uw_sub_p[:,indx]/norm,Z/ABLHm	,c='purple'	,label='sub. shells ('	+str(np.round(flx_part_sub[i]*100,1))	+'%)')
				if 'SV4' in dsCS1.keys():
					ax.plot( uw_up_p2[:,indx]/norm,Z/ABLHm	,c='orange'	,label='updrafts 2 ('	+str(np.round(flx_part_up2[i]*100,1))	+'%)')
					ax.plot( uw_sub_p2[:,indx]/norm,Z/ABLHm	,c='pink'	,label='sub. shells 2 ('+str(np.round(flx_part_sub2[i]*100,1))	+'%)')
				ax.plot( uw_down_p[:,indx]/norm,Z/ABLHm	,c='green'	,label='downdrafts ('	+str(np.round(flx_part_down[i]*100,1))	+'%)')
				#ax.plot( uw_env_p[:,indx]/norm,Z/ABLHm	,c='grey'	,label='env ('			+str(np.round(flx_part_env[i]*100,1))	+'%)')
				ax.plot( flx_summ[:,i]/norm,Z/ABLHm		,c='k'		,label='all ('			+str(np.round(flx_obj_over_all[i]*100,1))+'%)')
			#ax.plot( uw_mean[:,indx]/norm,Z/ABLHm	,c='k'		,label='mean',ls='--') # only resolved flux
			ax.plot( uw_mean_tot[:,indx]/norm,Z/ABLHm	,c='k'		,label='mean S1',ls='--')
			ax.set_title("X="+str(np.round(X[L_atX[i]].values/1000,0))+"km",loc='right')
			ax.set_ylabel('z/zi')
			ax.set_xlabel("$<\~ u \~ w>$/u*²")
			ax.legend(loc='upper left')
			ax.set_ylim([0,1.2])
			ax.set_xlim([-1.5,0.3])
			ax.grid()
			fig.savefig(path_save2+'CS1_'+name+'_uw_profile_i'+str(indx)+'.png')
			
	if 'wthtv_profile' in L_plot:
		print('			- profiles wthtv at i='+str(l_indx))
		flx_summ = (wthtv_up_p + wthtv_sub_p + wthtv_up_p2 + wthtv_sub_p2 + wthtv_down_p + wthtv_env_p ).isel(ni=l_indx)
		flx_part_up 	= mean_vertical_contrib((wthtv_up_p).isel(ni=l_indx),	(wthtv_mean).isel(ni=l_indx),indzi).values
		flx_part_sub 	= mean_vertical_contrib((wthtv_sub_p).isel(ni=l_indx),	(wthtv_mean).isel(ni=l_indx),indzi).values
		flx_part_up2 	= mean_vertical_contrib((wthtv_up_p2).isel(ni=l_indx),	(wthtv_mean).isel(ni=l_indx),indzi).values
		flx_part_sub2 	= mean_vertical_contrib((wthtv_sub_p2).isel(ni=l_indx),	(wthtv_mean).isel(ni=l_indx),indzi).values
		flx_part_down 	= mean_vertical_contrib((wthtv_down_p).isel(ni=l_indx),	(wthtv_mean).isel(ni=l_indx),indzi).values
		flx_part_env 	= mean_vertical_contrib((wthtv_env_p).isel(ni=l_indx),	(wthtv_mean).isel(ni=l_indx),indzi).values
		flx_obj_over_all = mean_vertical_contrib(flx_summ,(wthtv_mean).isel(ni=l_indx),indzi).values
		for i,indx in enumerate(l_indx):
			norm = Qv0
			#print('Qv0(4km)',Qv0)
			#print('X=',L_atX[i],'km, QV0=',dsCS1.Qv0[indx].values,',min(wthtv)=',np.amin(dsflx.FLX_THvW[:indzi,indx]).values)
			beta = (dsCS1.Qv0[indx].values - np.amin(dsflx.FLX_THvW[:indzi,indx])) # beta uses total fluxes
			# beta = (Qv0 - np.amin(wthtv_mean[:,indx]))
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			ax.plot( wthtv_up_p[:,indx]/norm,Z/ABLHm	,c='red'	,label='updrafts ('		+str(np.round(flx_part_up[i]*100,1))	+'%)')
			ax.plot( wthtv_sub_p[:,indx]/norm,Z/ABLHm	,c='purple'	,label='sub. shells ('	+str(np.round(flx_part_sub[i]*100,1))	+'%)')
			if 'SV4' in dsCS1.keys():
				ax.plot( wthtv_up_p2[:,indx]/norm,Z/ABLHm	,c='orange'	,label='updrafts 2 ('	+str(np.round(flx_part_up2[i]*100,1))	+'%)')
				ax.plot( wthtv_sub_p2[:,indx]/norm,Z/ABLHm	,c='pink'	,label='sub. shells 2 ('+str(np.round(flx_part_sub2[i]*100,1))	+'%)')
			ax.plot( wthtv_down_p[:,indx]/norm,Z/ABLHm	,c='green'	,label='downdrafts ('	+str(np.round(flx_part_down[i]*100,1))	+'%)')
			#ax.plot( wthtv_env_p[:,indx]/norm,Z/ABLHm	,c='grey'	,label='env ('			+str(np.round(flx_part_env[i]*100,1))	+'%)')
			ax.plot( flx_summ[:,i]/norm,Z/ABLHm			,c='k'		,label='all ('			+str(np.round(flx_obj_over_all[i]*100,1))+'%)')
			ax.plot( wthtv_mean[:,indx]/norm,Z/ABLHm	,c='k'		,label='mean',ls='--')
			#ax.set_title("S1 at X="+str(np.round(X[L_atX[i]].values/1000,0))+r"km, $\beta$="+str(np.round(beta.values,6)),loc='right')
			ax.set_title("X="+str(np.round(X[L_atX[i]].values/1000,0))+"km",loc='right')
			ax.set_ylabel('z/zi')
			ax.set_xlabel(r"$<\~ w \~ \theta_v>$/$Q_v^*$")
			ax.legend(loc='upper right')
			ax.set_ylim([0,1.2])
			ax.set_xlim([-1.1,3.5])
			ax.grid()
			fig.savefig(path_save2+'CS1_'+name+'_wthtv_profile_i'+str(indx)+'.png')
			
	# profiles of VAR at selected X points
	if 'profils_var' in L_plot:
		print('			- profiles VAR at X='+str(L_atX)+' km')
		# TBD
		
	if 'YZ' in L_plot:
		print('			- YZ plots at i='+str(Xloc_forYZ))
		for t,indx in enumerate(l_indxYZ):
			time = Tloc_forYZ[t]
			ind1,ind2 = get_mixed_layer_indexes(Z,gTHT[:,indx],SEUIL_ML)		
			THTmixed = ( THTm.isel(level=slice(ind1,ind2),ni=indx,time=0,nj=0).integrate('level')/ (Z[ind2]-Z[ind1]) ).values # here the mean field has been extend artificially into 4D
			RVmixed = ( RVm.isel(level=slice(ind1,ind2),ni=indx,time=0,nj=0).integrate('level')/ (Z[ind2]-Z[ind1]) ).values
			
			stepy,stepz = 2,2 # number of skiped cell to draw arrows
			fig, ax = plt.subplots(1,1,figsize = (5,6),constrained_layout=True,dpi=dpi)	
			ax.set_title('Objects and wind - mean wind',loc='right')
			ax.pcolormesh( Y/1000,Z/ABLHm,global_mask[time,:,:,indx],cmap=c.ListedColormap(['white','r','purple','g','grey','orange','pink']))
			Q = ax.quiver(Y[::stepy]/1000,Z[::stepz]/ABLHm,(V-Vm)[time,::stepz,::stepy,indx],W[time,::stepz,::stepy,indx],
					angles='uv',pivot='middle',headwidth=2,headaxislength=4,scale=30)
			ax.quiverkey(Q, 0.9, 0.05, 0.5, '0.5 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
			ax.set_ylim([0,1.2])
			ax.set_xlim([Y[0]/1000,Y[-1]/1000])
			ax.set_aspect(2.0)
			ax.set_xlabel('Y (km)')
			fig.savefig(path_save2+'CS1_'+name+'_YZ_i'+str(indx)+'.png')
			# thtv,W and rv
			fig, ax = plt.subplots(3,1,figsize = (5,12),constrained_layout=True,dpi=dpi)
			s = ax[0].pcolormesh( Y/1000,Z/ABLHm,(THTV-THTVm)[time,:,:,indx],cmap='coolwarm',vmin=-0.15,vmax=0.15) 
			plt.colorbar(s,ax=ax[0])
			ax[0].tick_params(axis='both',labelbottom=False)
			ax[0].set_title(r'$\theta_v-<\theta_v>$ (K)',loc='right')
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLHm,is_up[time,:,:,indx],levels=[0.55],colors=['r'],linewidths=1.0)
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLHm,is_sub[time,:,:,indx],levels=[0.55],colors=['purple'],linewidths=1.0)
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLHm,is_up2[time,:,:,indx],levels=[0.55],colors=['orange'],linewidths=1.0)
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLHm,is_sub2[time,:,:,indx],levels=[0.55],colors=['pink'],linewidths=1.0)
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLHm,is_down[time,:,:,indx],levels=[0.55],colors=['g'],linewidths=1.0)
			ax[0].contour( Y/1000+0.05/2,(Z-dz/2)/ABLHm,is_env[time,:,:,indx],levels=[0.55],colors=['grey'],linewidths=1.0)		
			s = ax[1].pcolormesh( Y/1000,Z/ABLHm,W[time,:,:,indx],cmap='coolwarm',vmin=-1,vmax=1) 
			plt.colorbar(s,ax=ax[1],orientation='vertical')
			ax[1].tick_params(axis='both',labelbottom=False)
			ax[1].set_title('W (m/s)',loc='right')
			
			s = ax[2].pcolormesh( Y/1000,Z/ABLHm,(RV[time,:,:,indx]-RVmixed)*1000,cmap='BrBG',vmin=-1,vmax=1) 
			plt.colorbar(s,ax=ax[2])
			ax[2].set_title(r'$r_v-r_{v,mixed}$ (g/kg) and objects',loc='right')
			ax[2].set_xlabel('Y (km)')
			for axe in ax:
				axe.set_ylim([0,1.2])
				axe.set_xlim([Y[0]/1000,Y[-1]/1000])
				axe.set_aspect(2.0)
				axe.set_ylabel('z/zi')
			fig.savefig(path_save2+'CS1_'+name+'_YZ_tht_rv_i'+str(indx)+'.png')
			# Scalar fluctuations
			fig, ax = plt.subplots(3,1,figsize = (5,12),constrained_layout=True,dpi=dpi)
			if 'SV4' in dsCS1.keys():
				s = ax[0].pcolormesh( Y/1000,Z/ABLHm,sv4_fluc[time,:,:,indx],cmap='Oranges',vmin=0.01,vmax=10,norm="log") 
				plt.colorbar(s,ax=ax[0])
			ax[0].tick_params(axis='both',labelbottom=False)
			ax[0].set_title(r"$sv_4'$",loc='right')
			s = ax[1].pcolormesh( Y/1000,Z/ABLHm,sv1_fluc[time,:,:,indx],cmap='Reds',vmin=0.01,vmax=10,norm="log") 
			plt.colorbar(s,ax=ax[1])
			ax[1].tick_params(axis='both',labelbottom=False)
			ax[1].set_title(r"$sv_1'$",loc='right')
			s = ax[2].pcolormesh( Y/1000,Z/ABLHm,sv3_fluc[time,:,:,indx],cmap='Blues',vmin=0.1,vmax=500,norm="log") 
			plt.colorbar(s,ax=ax[2])
			ax[2].set_xlabel('Y (km)')
			ax[2].set_title(r"$sv_3'$",loc='right')
			for axe in ax:
				axe.set_ylim([0,1.2])
				axe.set_xlim([Y[0]/1000,Y[-1]/1000])
				axe.set_aspect(2.0)
				axe.set_ylabel('z/zi')
			fig.savefig(path_save2+'CS1_'+name+'_YZ_sv_i'+str(indx)+'.png')
		
	if 'obj_frac' in L_plot:
		# profiles at selected X in L_atX
		if True:
			print('			- object fraction at X='+str(L_atX)+' km')
			for atX in L_atX:
				indx = nearest(X.values,atX*1000)
				fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
				#ax.plot( np.ones(F_up.shape) - (F_up+F_sub+F_down+F_env),Z/ABLH,c='k',label='residual',ls='--') # residus = 1 - other
				ax.plot( F_up[:,indx],Z/ABLHm	,c='red',label='updrafts')
				ax.plot( F_sub[:,indx],Z/ABLHm	,c='purple',label='sub. shells')
				ax.plot( F_down[:,indx],Z/ABLHm	,c='green',label='downdrafts')
				ax.plot( F_env[:,indx],Z/ABLHm	,c='grey',label='env')
				if 'SV4' in dsCS1.keys():
					ax.plot( F_up2[:,indx],Z/ABLHm	,c='orange',label='updrafts 2')
					ax.plot( F_sub2[:,indx],Z/ABLHm	,c='pink',label='sub. shells 2')
				ax.plot( (F_up + F_sub + F_down + F_env + F_up2 + F_sub2)[:,indx],Z/ABLHm,c='k',label='total')
				ax.set_title('Object area cover over entire domain (S1) at X='+str(atX)+'km')
				ax.set_ylabel('z / <zi>x')
				ax.legend(loc='upper right')
				ax.set_ylim([0,1.2])
				ax.set_xlim([0,0.7])
				ax.grid()
				fig.savefig(path_save2+'CS1_'+name+'_cover_fraction_i'+str(indx)+'.png')
		# culumated area cover of each structure
		if False:
			print('			- cumulated meanTXZ cover')
			alpha = 1
			fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
			ax.fill_between(X/1000, (part_up+part_up2+part_sub+part_sub2+part_down+part_env)*100,color='grey',alpha=alpha,label='env')
			ax.fill_between(X/1000, (part_up+part_up2+part_sub+part_sub2+part_down)*100,color='green',alpha=alpha,label='downdrafts')
			ax.fill_between(X/1000, (part_up+part_up2+part_sub+part_sub2)*100,color='pink',alpha=alpha,label='sub. shell 2')
			ax.fill_between(X/1000, (part_up+part_up2+part_sub)*100,color='purple',alpha=alpha,label='sub. shell')
			ax.fill_between(X/1000, (part_up+part_up2)*100,color='orange',alpha=alpha,label='updrafts 2')
			ax.fill_between(X/1000, (part_up)*100,color='red',alpha=alpha,label='updrafts')
			ax.set_xlim([X[0]/1000,X[-1]/1000])
			ax.set_ylim([0,50])
			ax.set_xlabel('X (km)')
			ax.set_ylabel('%')
			ax.set_title('meanTXZ cover of each structures')
			fig.savefig(path_save2+'CS1_'+name+'_Xcumcover.png')	
			
			fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
			ax.plot(X/1000, part_env*100,color='grey',label='env')
			ax.plot(X/1000, part_down*100,color='green',label='downdrafts')
			ax.plot(X/1000, part_sub2*100,color='pink',label='sub. shell 2')
			ax.plot(X/1000, part_sub*100,color='purple',label='sub. shell')
			ax.plot(X/1000, part_up2*100,color='orange',label='updrafts 2')
			ax.plot(X/1000, part_up*100,color='red',label='updrafts')
			ax.set_xlim([X[0]/1000,X[-1]/1000])
			ax.set_ylim([0,20])
			ax.set_xlabel('X (km)')
			ax.set_ylabel('%')
			ax.set_title('meanTXZ cover of each structures')
			fig.savefig(path_save2+'CS1_'+name+'_Xcover.png')
		# cover at altitudes
		if False:
			print('			- meanTX cover at several altitudes') # a adapter avec SV4
			frac_zi = [0.2,0.5,1] # here zi = ABLHm but is somewhat representative of the ABLH(x)
			alpha = 1
			for z in frac_zi:
				indz = nearest(Z.values,z*ABLHm)
				fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
				ax.fill_between(X/1000, (F_up+F_up2+F_sub+F_sub2+F_down+F_env)[indz,:]*100,color='grey',alpha=alpha,label='env')
				ax.fill_between(X/1000, (F_up+F_up2+F_sub+F_sub2+F_down)[indz,:]*100,color='green',alpha=alpha,label='downdrafts')
				ax.fill_between(X/1000, (F_up+F_up2+F_sub+F_sub2)[indz,:]*100,color='pink',alpha=alpha,label='sub. shell 2')
				ax.fill_between(X/1000, (F_up+F_up2+F_sub)[indz,:]*100,color='purple',alpha=alpha,label='sub. shell')
				ax.fill_between(X/1000, (F_up+F_up2)[indz,:]*100,color='orange',alpha=alpha,label='updrafts 2')
				ax.fill_between(X/1000, F_up[indz,:]*100,color='red',alpha=alpha,label='updrafts')
				ax.set_xlim([X[0]/1000,X[-1]/1000])
				ax.set_ylim([0,60])
				ax.set_xlabel('X (km)')
				ax.set_ylabel('%')
				ax.set_title('meanTX cover of each structures at z/zi='+str(z))
				fig.savefig(path_save2+'CS1_'+name+'_Xcumcover_k'+str(indz)+'.png')	
				
				fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
				ax.plot(X/1000, F_env[indz,:]*100,color='grey',label='env')
				ax.plot(X/1000, F_down[indz,:]*100,color='green',label='downdrafts')
				ax.plot(X/1000, F_sub[indz,:]*100,color='purple',label='sub. shell')
				ax.plot(X/1000, F_up[indz,:]*100,color='red',label='updrafts')
				if 'SV4' in dsCS1.keys():
					ax.plot(X/1000, F_sub2[indz,:]*100,color='pink',label='sub. shell 2')
					ax.plot(X/1000, F_up2[indz,:]*100,color='orange',label='updrafts 2')
				ax.set_xlim([X[0]/1000,X[-1]/1000])
				ax.set_ylim([0,20])
				ax.set_xlabel('X (km)')
				ax.set_ylabel('%')
				ax.set_title('meanTXZ cover of each structures at z/zi='+str(z))
				fig.savefig(path_save2+'CS1_'+name+'_Xcover_k'+str(indz)+'.png')
				
				
	# film XZ avec le champ de vent en vecteurs et les objets
	if 'obj_movie' in L_plot: 
		print('			- object movie')
		fps = 10
		path_save_frames = path_save2+'frames/'
		# XZ plot -------------------------------------------------------------
		CHOIX = 'objects'
		atpos = atyXZ*1000
		vect_param = {'stepx':2,'stepy':3,
				'headwidth':2,'headaxislength':4,
				'scale':50,'vect_ref':0.5}	
		params = {'RVT':{'VAR':RV,
				'cmap':'Blues',
				'bornes':{'x':[10,15],'y':[0,800],'var':[10,13]},
				'coeff':1000},
			'objects':{'VAR':global_mask,
				'cmap':c.ListedColormap(['white','r','purple','g','grey']),
				'bornes':{'x':[10,15],'y':[0,800],'var':[0,4]},
				'coeff':1} }
		# ---------------------------------------------------------------------			
		cmap = params[CHOIX]['cmap']
		bornes = params[CHOIX]['bornes']
		coeff = params[CHOIX]['coeff']
		VAR = params[CHOIX]['VAR']
		MovieMaker_Vector(VAR,U-Um,W,vect_param,Time,X,Y,Z,'XZ',atpos,
				cmap,coeff=coeff,bornes=bornes,dpi=200,fps=fps,
				path_save=path_save2,path_save_frames=path_save_frames)	
	
		
			
	# * profiles of fluxes at selected X points
	# * plot XZ avec 5 fig: pcolormesh de chaque contrib + total des structures coherentes
	# 		mais plot en anomaly de contribution par rapport à X=0 par ex.
	
	if 'XY' in L_plot:
		for atziXY in L_atziXY:
			print('			- XY plots at z/zi =',str(atziXY))
			if TURB_COND=='ITURB2':
				cmap = c.ListedColormap(['white','r','purple','g','grey'])
				vmin,vmax = 0,4
			elif TURB_COND=='C10':
				cmap = c.ListedColormap(['white','r','purple','g','grey','orange','pink'])
				vmin,vmax = 0,6
			indx1,indx2 = nearest(X.values,7*1000),nearest(X.values,17*1000)
			indzzi = nearest(Z.values,atziXY*ABLHm)
			stepx,stepy = 1,1 # number of skiped cell to draw arrows
			fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=300)	
			ax.set_title('Objects and wind - mean wind',loc='right')
			ax.pcolormesh( X[indx1:indx2]/1000,Y/1000,global_mask[indt,indzzi,:,indx1:indx2],cmap=cmap,vmin=vmin,vmax=vmax)
			Q = ax.quiver(X[indx1:indx2:stepx]/1000,Y[::stepy]/1000,(U-Um)[indt,indzzi,::stepy,indx1:indx2:stepx],(V-Vm)[indt,indzzi,::stepy,indx1:indx2:stepx],
					angles='uv',pivot='middle',width=0.001,headwidth=3,headaxislength=2,headlength=2,scale=100)
			ax.quiverkey(Q, 0.9, 0.05, 1, '1 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
			ax.set_ylim([0,2])
			ax.set_xlim([X[indx1]/1000,X[indx2]/1000])
			ax.set_aspect(1.0)
			ax.set_xlabel('X (km)')
			ax.set_ylabel('Y (km)')
			fig.savefig(path_save2+'CS1_'+name+'_XY_zzi'+str(atziXY)+'.png')
	# XZ at selected Y
	

def Buoy_in_structures(X,Z,SST,dsref,crit_value,TURB_COND,SVT_type,RV_DIFF,DIR,L_atX,L_zzi,chunksNOHALO,path_saving,dpi):
	"""
	This procedure is plotting the buoyancy term of the W budget for a specific coherent structure.
	(for now only for updrafts but can be modified easily)
	
	And the evolution with X of the thtv from inside the structures vs environnement.
	
	INPUTS
		- X,Z		: dimensions of MNH domain
		- SST		: 1D sst
		- dsref		: reference simulations
		- crit_value: to represent the front
		- TURB_COND : turbulent condition to use
		- SVT_type 	: type of tracer to use, RV or SVT
		- RV_DIFF	: if SVT_type==RV
		- DIR		: direction of the plot, 'x' or 'z'
		- L_atX		: list of X coordinates to plots profiles
		- L_zzi		: list of Z/zi coordinates
		- chunksNOHALO : for // computation
		- path_saving 	: abs path to store figures
		- dpi 			: dot per inches for figures
		
	OUTPUT
		- plot of buoyancy in updraft in the 'DIR' direction
		- evolution of THTV in X at specific z/zi (boolean to change manually)
		
	"""
	K = 5/100 # minimum area cover to plot profiles
	ABLH = 600
	if SVT_type=='SVT': RV_DIFF='MEAN'
	NAME = 'S1'
	# Opening file
	if TURB_COND=='ITURB2':
		dsCS1 = xr.open_dataset('DATA_turb/06W_CS1_S1_ITURB2_RVMIXED_keep.nc',chunks=chunksNOHALO)
	else:
		dsCS1 = xr.open_dataset('DATA_turb/06W_CS1_'+NAME+'_'+TURB_COND+'_'+SVT_type+RV_DIFF+'.nc',chunks=chunksNOHALO) # to get 'updrafts' detection
	dsCS1cold = xr.open_dataset('DATA_turb/06W_CS1_cold_'+TURB_COND+'_'+SVT_type+RV_DIFF+'.nc',chunks=chunksNOHALO)
	dsCS1warm = xr.open_dataset('DATA_turb/06W_CS1_warm_'+TURB_COND+'_'+SVT_type+RV_DIFF+'.nc',chunks=chunksNOHALO)
	indt = -1 # last time for ref files
	cmap_warm ='Reds'
	cmap_cold ='winter'
	# building save path
	path_save2 = path_saving+TURB_COND
	if TURB_COND=='C10':
		mCS = dsCS1.attrs['mCS']
		gammaRV = dsCS1.attrs['gammaRv']
		path_save2 = path_save2+'_m'+str(mCS)+'_g'+str(gammaRV*100)
	elif TURB_COND=='ITURB2':
		gammaTurb2 = dsCS1.attrs['gammaTurb2']
		path_save2 = path_save2+'_g'+str(gammaTurb2*100)
	elif TURB_COND=='ITURB':
		gammaTurb1 = dsCS1.attrs['gammaTurb1']
		path_save2 = path_save2+ '_g'+str(gammaTurb1*100)
	elif TURB_COND=='EC':
		gammaEc = dsCS1.attrs['gammaEc']
		path_save2 = path_save2 +'_g'+str(gammaEc*100)
	if SVT_type=='RV':
		path_save2 = path_save2+'_'+RV_DIFF+'_'+SVT_type+'/'
	else:
		path_save2 = path_save2+'_'+SVT_type+'/'
	path_save2 = path_save2 + NAME+'/'
	if not os.path.isdir(path_save2): # creat folder if it doesnt exist
		os.makedirs(path_save2)
		
	# Variables
	is_up = xr.where(dsCS1.global_objects==1,1,0)	
	is_up_refW = xr.where(dsCS1warm.global_objects==1,1,0)	
	is_up_refC = xr.where(dsCS1cold.global_objects==1,1,0)
	if 'is_up2' in dsCS1.keys():
		is_up = np.logical_or(is_up,xr.where(dsCS1.global_objects==5,1,0))
	
	F_up = is_up.mean(dim=['time','nj']) 
	F_up_warm = is_up_refW.mean(dim=['ni','nj'])
	F_up_cold = is_up_refC.mean(dim=['ni','nj'])
	THTV = dsCS1.THTV
	THTV_up = THTV.where(is_up==True).mean(dim=['time','nj'])
	#THTV_ref = THTV.mean(dim=['time','nj'])
	THTV_ref = THTV.mean(dim=['time','nj'])
	B = g*(THTV_up/THTV_ref - 1)
	THTV_warm = dsCS1warm.THTV
	THTV_up_warm = THTV_warm.where(is_up_refW==1).mean(dim=['ni','nj'])
	THTV_ref_warm = THTV_warm.mean(dim=['ni','nj'])
	#ABLH_warm = dsref['warm']['nomean']['Misc']['BL_H'][indt].values
	ABLH_warm = Z[THTV_warm.mean(dim=['ni','nj']).differentiate('level').argmax().values].values
	B_warm = g*(THTV_up_warm/THTV_ref_warm - 1)
	THTV_cold = dsCS1cold.THTV
	THTV_up_cold = THTV_cold.where(is_up_refC==1).mean(dim=['ni','nj'])
	THTV_ref_cold = THTV_cold.mean(dim=['ni','nj'])
	#ABLH_cold = dsref['cold']['nomean']['Misc']['BL_H'][indt].values
	ABLH_cold = Z[THTV_cold.mean(dim=['ni','nj']).differentiate('level').argmax().values].values
	
	#Z[int(dsCS1[case].THTV.mean(dim=['ni','nj']).differentiate('level').argmax())].values
	
	B_cold = g*(THTV_up_cold/THTV_ref_cold - 1)
	# plot
	BLIM = [-0.006,0.006]
	if 'x' in DIR:	
		L_indx = []
		for x in L_atX:
			L_indx.append(nearest(X.values,x*1000))
		colorsX = DISCRETIZED_2CMAP_2(cmap_cold,cmap_warm,L_atX*1000,SST,crit_value,X.values)	
		fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
		ax.vlines(0,0,3,colors='grey',alpha=0.5)
#		ax.plot(B_cold[:].where(F_up_cold>K),Z/ABLH_cold,c='blue',label='ref: cold',ls='--')
#		ax.plot(B_warm[:].where(F_up_warm>K),Z/ABLH_warm,c='red',label='ref: warm',ls='--')
		ax.plot(np.ma.masked_where(F_up_cold<=K,B_cold),Z/ABLH_cold,c='blue',label='ref: cold',ls='--')
		ax.plot(np.ma.masked_where(F_up_warm<=K,B_warm),Z/ABLH_warm,c='red',label='ref: warm',ls='--')
		for kx,indx in enumerate(L_indx):
			#ax.plot(B[:,indx].where(F_up[:,indx]>K),Z/ABLH,c=colorsX[kx],label='X='+str(L_atX[kx])+'km')
			ax.plot(np.ma.masked_where(F_up.isel(ni=indx)<=K,B.isel(ni=indx)),Z/ABLH,c=colorsX[kx],label='X='+str(L_atX[kx])+'km')
			
		ax.legend()
		ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
		ax.xaxis.major.formatter._useMathText = True
		ax.set_ylabel(r'z/$z_i$')
		ax.set_xlabel(r'g*(<$\theta_v$>$_{up}$/$<\theta_v>$-1) (m.s$^{-2}$)')
		ax.set_ylim([0,1.2])
		ax.set_xlim(BLIM)
		ax.grid()
		fig.savefig(path_save2+'CS1_B_inX.png')
	if 'z' in DIR:
		L_indz = []
		for zzi in L_zzi:
			L_indz.append(nearest(Z.values/ABLH,zzi))
		cmap = mpl.colormaps.get_cmap('jet')
		fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)	
		ax.hlines(0,0,X[-1]/1000,colors='grey',alpha=0.5)
#		for kz,indz in enumerate(L_indz): # isolated for loop to get the hlines under everything
#			ax.hlines(B_cold[indz],0,X[-1]/1000,colors='blue',ls='--',alpha=0.5)
#			ax.hlines(B_warm[indz],0,X[-1]/1000,colors='red',ls='--',alpha=0.5)
		for kz,indz in enumerate(L_indz):
			ax.plot(X/1000,B[indz,:],c=cmap(kz/len(L_zzi)),label=str(L_zzi[kz]))
		ax.legend()
		ax.set_xlim(X[0]/1000,X[-1]/1000)
		ax.set_ylim(BLIM)
		ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		ax.xaxis.major.formatter._useMathText = True
		ax.set_ylabel(r'g*(<$\theta_v$>$_{up}$/$<\theta_v>$-1) (m.s$^{-2}$)'+'\nat some z/zi')
		ax.set_xlabel('X (km)')
		fig.savefig(path_save2+'CS1_B_inZ.png')
		
	# X evolution of THTV at some z/zi
	if False:
		fig, ax = plt.subplots(len(L_zzi),1,figsize = (5,10),constrained_layout=True,dpi=dpi)	
		YLIM = [0.65,0.9]
		for k,zzi in enumerate(L_zzi):
			indz = nearest(Z.values,zzi*ABLH)
			ax[k].plot(X/1000,THTV_up[indz,:]-297,c='r',label='in Up')
			ax[k].plot(X/1000,THTV_ref[indz,:]-297,c='k',label='mean')
			ax[k].set_title('z/zi='+str(zzi),loc='right')
			ax[k].set_xlim(X[0]/1000,X[-1]/1000)
			ax[k].tick_params(axis='both',labelbottom=False)
			ax[k].set_ylim(YLIM)	
		ax[-1].tick_params(axis='both',labelbottom=True)
		ax[-1].set_xlabel('X (km)')
		ax[0].set_title(r'$\theta_v$ (K) - 297K')
		ax[0].legend()
		fig.savefig(path_save2+'CS1_THTVup_vs_THTVmean_X.png')
	
def Contrib_flux_Tau(chunksNOHALO,data,data_mean,param_COND,L_TURB_COND,SVT_type,RV_DIFF,path_CS1,dpi):
	"""
	This procedure is plotting the contribution of the different coherent structures detected by a chosen conditonal sampling.
	The input data should have different radio decay for the passive tracers (same tau for all tracers)
	
	INPUTS
		- chunksNOHALO 	: chunks sizes when MNH halo has already been removed
		- data 			: 3D fields, with conditional sampling filter, data is ouput of 'build_CS1' from module_building_files
		- data_mean 	: mean fields from 000 diachronic file, opened with 'Open_LES_MEAN' from module_building_files
		- param_COND 	: parameters for conditional samplings
		- L_TURB_COND 	: choice of conditional sampling
		- SVT_type 		: type of tracer used, 'SVT' or 'RV'
		- RV_DIFF 		: if SVT_type=='RV', choose between using fluctuations of RV
							with respect to horizontal mean value or mean mixed layer value
		- path_CS1 		: where to save the CS files and the plot
		- dpi 			: for figures
		
	OUTPUT
		a plot with 2 figures (one for flux uw and one for wthtv):
			- Y axis : contribution to the flux for each structure
			- X axis : radio decay constante (tau)
	
	NOTE 07/08/24 : tau has been changed for nu to be consistent with paper notation
	
	"""
	
	L_name = ['up','sub','down','env','all']
	L_col = ['red','purple','green','grey','black']
	Tau = np.array([1,4,7,10,12,15,20,30,40])
	Tauref = 15
	indTauref = nearest(Tau,Tauref)
	# 1) building CS for each files
	SEUIL_ML = 0.5
	indt = -1				# time index for ref simus, -1 <=> t=+3h
	build_CS1(nhalo,data,data_mean,param_COND,
		L_TURB_COND,SVT_type,RV_DIFF,0.5,-1,path_CS1)
	for case in data.keys(): data[case].close() # closing OUT files
	# 2) getting flux decomposition
	TURB_COND = L_TURB_COND[0]
	dsCS = {'1min':xr.open_dataset(path_CS1+'CS1_1min_'+TURB_COND+'_'+SVT_type+RV_DIFF+'_m1.nc',chunks=chunksNOHALO),
			'4min':xr.open_dataset(path_CS1+'CS1_4min_'+TURB_COND+'_'+SVT_type+RV_DIFF+'_m1.nc',chunks=chunksNOHALO),
			'7min':xr.open_dataset(path_CS1+'CS1_7min_'+TURB_COND+'_'+SVT_type+RV_DIFF+'_m1.nc',chunks=chunksNOHALO),
			'10min':xr.open_dataset(path_CS1+'CS1_10min_'+TURB_COND+'_'+SVT_type+RV_DIFF+'_m1.nc',chunks=chunksNOHALO),
			'12min':xr.open_dataset(path_CS1+'CS1_12min_'+TURB_COND+'_'+SVT_type+RV_DIFF+'_m1.nc',chunks=chunksNOHALO),
			'15min':xr.open_dataset(path_CS1+'CS1_15min_'+TURB_COND+'_'+SVT_type+RV_DIFF+'_m1.nc',chunks=chunksNOHALO),
			'20min':xr.open_dataset(path_CS1+'CS1_20min_'+TURB_COND+'_'+SVT_type+RV_DIFF+'_m1.nc',chunks=chunksNOHALO),
			'30min':xr.open_dataset(path_CS1+'CS1_20min_'+TURB_COND+'_'+SVT_type+RV_DIFF+'_m1.nc',chunks=chunksNOHALO),
			'40min':xr.open_dataset(path_CS1+'CS1_20min_'+TURB_COND+'_'+SVT_type+RV_DIFF+'_m1.nc',chunks=chunksNOHALO)}
	flx_uw = np.zeros((len(dsCS.keys()),5)) # 9 tau and 5 structures (objects + their sum)
	flx_wthtv = np.zeros((len(dsCS.keys()),5)) # 9 tau and 5 structures (objects + their sum)
	for i,case in enumerate(dsCS.keys()):
		ds = dsCS[case]
		# pathsave
		path_save2 = path_CS1 + 'Sensitivity_tau_' + TURB_COND
		if TURB_COND=='C10':
			mCS = ds.attrs['mCS']
			gammaRV = ds.attrs['gammaRv']
			path_save2 = path_save2+'_m'+str(mCS)+'_g'+str(gammaRV*100)
		elif TURB_COND=='ITURB2':
			gammaTurb2 = ds.attrs['gammaTurb2']
			path_save2 = path_save2+'_g'+str(gammaTurb2*100)
		elif TURB_COND=='ITURB':
			gammaTurb1 = ds.attrs['gammaTurb1']
			path_save2 = path_save2+ '_g'+str(gammaTurb1*100)
		elif TURB_COND=='EC':
			gammaEc = ds.attrs['gammaEc']
			path_save2 = path_save2 +'_g'+str(gammaEc*100)
		if SVT_type=='RV':
			path_save2 = path_save2+'_'+RV_DIFF+'_'+SVT_type+'.png'
		else:
			path_save2 = path_save2+'_'+SVT_type+'.png'
		
		uw = (ds.UT - ds.UTm)*ds.WT
		wthtv = ds.WT*(ds.THTV - ds.THTVm)
		uw_m = uw.mean(dim=['ni','nj'])
		wthtv_m = wthtv.mean(dim=['ni','nj'])
		uw_obj = {}
		wthtv_obj = {}
		uw_obj['all'],wthtv_obj['all'] = np.zeros((5)),np.zeros((5))
		gTHTv = ds.THTVm[:,0,0].differentiate('level')
		ABLHm = ds.level[gTHTv.argmax(dim='level').values].mean().values
		indzi = nearest(ds.level.values,1.1*ABLHm)
		sum1,sum2 = np.zeros(len(ds.level)),np.zeros(len(ds.level))
		for k,struc in enumerate(['up','sub','down','env']):
			# first flux decomposition
			temp1 =	compute_flx_contrib(uw,[ds['is_'+struc]],meanDim=['ni','nj'])
			temp2 =	compute_flx_contrib(wthtv,[ds['is_'+struc]],meanDim=['ni','nj'])
			sum1 = sum1 + temp1
			sum2 = sum2 + temp2
			# then vertical average
			flx_uw[i,k] = mean_vertical_contrib(temp1,uw_m,indzi)
			flx_wthtv[i,k] = mean_vertical_contrib(temp2,wthtv_m,indzi)
		# get contribution from all detected structures
		flx_uw[i,-1] = mean_vertical_contrib(sum1,uw_m,indzi)
		flx_wthtv[i,-1] = mean_vertical_contrib(sum2,wthtv_m,indzi)
	# 3) Plotting
	fig, ax = plt.subplots(2,1,figsize = (5,8),constrained_layout=True,dpi=dpi)
	ax1 = inset_axes(ax[0], width="60%", height="50%", loc=4,borderpad=2)
	ax2 = inset_axes(ax[1], width="60%", height="50%", loc=4,borderpad=2)
	for k in range(len(L_name)):
		if L_name[k]!='env':
			ax[0].plot(Tau,(flx_uw[:,k]-flx_uw[indTauref,k])*100,c=L_col[k],label=L_name[k])
			ax1.plot(Tau,(flx_uw[:,k]-flx_uw[indTauref,k])*100,c=L_col[k],label=L_name[k])
			ax[1].plot(Tau,(flx_wthtv[:,k]-flx_wthtv[indTauref,k])*100,c=L_col[k],label=L_name[k])
			ax2.plot(Tau,(flx_wthtv[:,k]-flx_wthtv[indTauref,k])*100,c=L_col[k],label=L_name[k])
	ax[0].set_ylabel('Contribution anomaly \n'+r'(% over $\nu$='+str(Tauref)+'min)')
	ax[0].set_title(r"$< \~ u \~w >$",loc='left')
	ax[0].legend(loc='upper right')
	ax[1].set_xlabel(r'$\nu$ (min)')
	ax[1].set_ylabel('Contribution anomaly \n'+r'(% over $\nu$='+str(Tauref)+'min)')
	ax[1].set_title(r"$< \~ w \~ \theta_v>$",loc='left')
	#fig.suptitle(r'Sensitivity tests on $\tau$')
	ax1.set_ylim([-5,5])
	ax2.set_ylim([-5,5])
	fig.savefig(path_save2)
	
	# absolute values
#	fig, ax = plt.subplots(2,1,figsize = (7,8),constrained_layout=True,dpi=dpi)
#	ax1 = inset_axes(ax[0], width="80%", height="50%", loc=4,borderpad=2)
#	ax2 = inset_axes(ax[1], width="80%", height="50%", loc=4,borderpad=2)
#	for k in range(len(L_name)):
#		ax[0].plot(Tau,(flx_uw[:,k])*100,c=L_col[k],label=L_name[k])
#		ax1.plot(Tau,(flx_uw[:,k])*100,c=L_col[k],label=L_name[k])
#		ax[1].plot(Tau,(flx_wthtv[:,k])*100,c=L_col[k],label=L_name[k])
#		ax2.plot(Tau,(flx_wthtv[:,k])*100,c=L_col[k],label=L_name[k])
#	ax[0].set_ylabel('object contribution (%)')
#	ax[0].set_title(r"$\overline{u'w'}$",loc='left')
#	ax[0].legend(loc='upper right')
#	ax[1].set_xlabel(r'$\tau$ (min)')
#	ax[1].set_ylabel('object contribution (%)')
#	ax[1].set_title(r"$\overline{w'\theta_v'}$",loc='left')
#	fig.suptitle(r'Sensitivity tests on $\tau$')
#	ax1.set_ylim([-5,5])
#	ax2.set_ylim([-5,5])
	#fig.savefig(path_save2)

def Hovmoller_up_down_cover(X,Z,N_timeO,chunksNOHALO,path_save,dpi):
	"""Plot Hovmöller diagram of Y averaged cover of updrafts and downdrafts
	
	question : how far is the mean ['time','ni','nj'] to the mean ['ni','nj'] at each instant ?
				
	INPUTS
		- X 			: X dimension of domain
		- Z 			: Z dimension of domain
		- N_timeO 		: Number of OUTPUT files = number of time step saved
		- chunksNOHALO 	: for dask, chunks when halo has been removed
		- path_save 	: where to save figures
		- dpi			: for figures
	
	OUTPUT
		- Hovmoller plots (X,Time) of 'updraft' and 'downdraft' cover
				
	"""
	dsITURB = xr.open_dataset('DATA_turb/06W_CS1_S1_ITURB2_RVMIXED.nc',chunks=chunksNOHALO)
	dsC10 = xr.open_dataset('DATA_turb/06W_CS1_S1_C10_SVTMEAN.nc',chunks=chunksNOHALO)
	atZ = 100 #m 
	indz = nearest(Z.values,atZ)
	atZ = np.round(Z[indz].values,2)
	
	is_down1 = dsITURB.is_down.mean(dim='nj').isel(level=indz)
	is_down2 = dsC10.is_down.mean(dim='nj').isel(level=indz)
	is_up1 = dsITURB.is_up.mean(dim='nj').isel(level=indz)
	#is_up2 = ( dsC10.is_up + dsC10.is_up2 - dsC10.is_up*dsC10.is_up2 ).mean(dim='nj').isel(level=indz) # logical OR
	is_up2 = np.logical_or(dsC10.is_up,dsC10.is_up2).mean(dim='nj').isel(level=indz) # logical OR
	
	fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	s = ax[0].pcolormesh(X/1000,np.arange(0,N_timeO,1),is_up1,vmin=0,vmax=0.5,cmap='jet')
	plt.colorbar(s,ax=ax[0])
	s = ax[1].pcolormesh(X/1000,np.arange(0,N_timeO,1),is_down1,vmin=0,vmax=0.5,cmap='jet')
	plt.colorbar(s,ax=ax[1])
	ax[1].set_xlabel('X (km)')
	ax[0].set_xlabel('X (km)')
	ax[0].set_ylabel('time (t*30s+2h)')
	ax[0].set_title('updraft',loc='right')
	ax[1].set_title('downdraft',loc='right')
	fig.suptitle('Object cover with ITURB2, z='+str(atZ)+'m')
	fig.savefig('PNG_CAS06W/Hovmoller_obj_cover_ITURB2_z100m.png')
	
	fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	s = ax[0].pcolormesh(X/1000,np.arange(0,N_timeO,1),is_up2,vmin=0,vmax=0.5,cmap='jet')
	plt.colorbar(s,ax=ax[0])
	s = ax[1].pcolormesh(X/1000,np.arange(0,N_timeO,1),is_down2,vmin=0,vmax=0.5,cmap='jet')
	plt.colorbar(s,ax=ax[1])
	ax[1].set_xlabel('X (km)')
	ax[0].set_xlabel('X (km)')
	ax[0].set_ylabel('time (t*30s+2h)')
	ax[0].set_title('updraft (both tracer)',loc='right')
	ax[1].set_title('downdraft',loc='right')
	fig.suptitle('Object cover with C10, z='+str(atZ)+'m')
	fig.savefig(path_save)	
	
def top_hat_decomposition(X,Z,dsflx,L_atX,case,TURB,SV_type,RVdiff,chunksNOHALO,path_save,dpi):
	"""
	This procedure is splitting resolved flux into a top hat contribution and intra-structure variability
		for now it is only for u'w' flux
	
	following Wang and Stevens 2000 "Top-Hat Representation of Turbulence Statistics in Cloud-Topped Boundary Layers:
										 Large Eddy Simulation Study" 
									
				-> eq 4a
	
	INPUTS :
		- X 		: X dimension, to be used for S1
		- Z 		: Z dimension, altitude
		- L_atX 	: list of X position to plot the profiles, only for S1
		- case		: 'warm' or 'cold' or 'S1'
		- TURB 		: turbulent condition for conditional sampling 'ITURB2' or 'C10'
		- SV_type 	: type of tracer used for conditional sampling, 'SVT' or 'RV'
		- RVdiff 	: if SV_type=RV, 'MEAN' or 'MIXED'
		- chunksNOHALO : for dask, chunks when halo has been removed
		- dpi 		: for figures
	
			
	OUTS:
		- 1 plot with tophat part and intra-variability. If case='S1', the plot is at 'L_atX' positions
	
	NOTE:
		- for alpha['other'] = 1-alpha['up']-alpha['down'], i checked
	"""
	dsCS1 = xr.open_dataset('DATA_turb/06W_CS1_'+case+'_'+TURB+'_'+SV_type+RVdiff+'.nc',chunks=chunksNOHALO)
	colors = {'up':['red','orange'],'down':['green','chartreuse'],'other':['blue','cyan']}
	Xlim = [-1.5,0.3]
	Ylim = [0,1.2]
	zi = 600
	# path save for fig
	path_save2 = path_save + TURB
	if TURB=='C10':
		mCS = dsCS1.attrs['mCS']
		gammaRV = dsCS1.attrs['gammaRv']
		path_save2 = path_save2+'_m'+str(mCS)+'_g'+str(gammaRV*100)
	elif TURB=='ITURB2':
		gammaTurb2 = dsCS1.attrs['gammaTurb2']
		path_save2 = path_save2+'_g'+str(gammaTurb2*100)
	if SV_type=='RV':
		path_save2 = path_save2+'_'+RVdiff+'_'+SV_type+'/'
	else:
		path_save2 = path_save2+'_SVT/'
	path_save2 = path_save2 + case +'/tophat'
	
	# on S1
	if case=='S1':
		U,W = dsCS1.UT,dsCS1.WT
		Um,Wm = dsCS1.UTm[1,:,1,:],dsCS1.WTm[1,:,1,:]
		u_fluc,w_fluc = U-Um,W-Wm
		uw_m = (u_fluc*w_fluc).mean(dim=['time','nj'])
		uw_m_total = dsflx.FLX_UW
		u_star = 0.211 # m/s
		# getting the masks
		obj_all = dsCS1.global_objects
		if SV_type=='SVT':
			is_up1 = xr.where(obj_all == 1,1,0)
			is_up2 = xr.where(obj_all == 5,1,0)
			is_up = np.logical_or(is_up1,is_up2)
		else:
			is_up = xr.where(obj_all == 1,1,0)
		is_down = xr.where(obj_all == 3,1,0)
		is_other = np.logical_not( np.logical_or(is_up,is_down) )
		d_str = {'up':is_up,'down':is_down,'other':is_other}
		# splitting into top hat and intra variability
		TopH,Intra = {},{}
		summ= xr.zeros_like(uw_m)
		for structure in d_str.keys():
			Ui = U.where(d_str[structure]==1).mean(dim=['time','nj'])
			Wi = W.where(d_str[structure]==1).mean(dim=['time','nj'])
			alphai = d_str[structure].mean(dim=['time','nj']) 
			TopH[structure] = alphai * (Ui-Um)*Wi
			Intra[structure] = alphai * ( (U-Ui)*(W-Wi) ).where(d_str[structure]==1).mean(dim=['time','nj'])
			summ = summ + TopH[structure] + Intra[structure]
		# plot
		for atX in L_atX:
			indx = nearest(X.values,atX*1000)
			fig, ax = plt.subplots(1,1,figsize = (4,5),constrained_layout=True,dpi=dpi)
			for structure in d_str.keys():
				ax.plot(TopH[structure][:,indx]/u_star**2,Z/zi,c=colors[structure][0],label=structure+' TopH')
				ax.plot(Intra[structure][:,indx]/u_star**2,Z/zi,c=colors[structure][0],label=structure+' Intra',ls=(0, (5, 1)))
			#ax.plot((summ)[:,indx]/u_star**2,Z/600,c='pink',label='total') # to check that total = mean
			#ax.plot(uw_m[:,indx]/u_star**2,Z/600,c='k',label='mean',ls='--') # only resolved flux
			ax.plot(uw_m_total[:,indx]/u_star**2,Z/zi,c='k',label='mean',ls='--') # total flux
			ax.set_ylim(Ylim)
			ax.set_xlim(Xlim)
			ax.set_xlabel(r"$<\~ u \~ w>$/u*²")
			ax.set_ylabel('z / zi')
			ax.legend()
			ax.set_title('S1 at X='+str(atX)+'km',loc='right')
			ax.xaxis.set_major_locator(MultipleLocator(0.5))
			ax.grid()
			ax.xaxis.label.set_fontsize(13)
			ax.yaxis.label.set_fontsize(13)
			fig.savefig(path_save2+'_i'+str(indx)+'.png')
			
	# on ref
	if (case=='warm' or case=='cold'):		
		is_up = dsCS1.is_up
		not_up = np.logical_not(is_up)
		U,W = dsCS1.UT,dsCS1.WT
		Um = dsCS1.UTm[:,1,1]
		u_fluc,w_fluc = U-Um,W
		uw_m = (u_fluc*w_fluc).mean(dim=['ni','nj'])
		obj_all = dsCS1.global_objects
		is_up = xr.where(obj_all == 1,1,0)
		is_down = xr.where(obj_all == 3,1,0)
		is_other = np.logical_not( np.logical_or(is_up,is_down) )
		d_str = {'up':is_up,'down':is_down,'other':is_other}
		# splitting into top hat and intra variability
		TopH,Intra = {},{}
		summ= xr.zeros_like(uw_m)
		for structure in d_str.keys():
			Ui = U.where(d_str[structure]==1).mean(dim=['ni','nj'])
			Wi = W.where(d_str[structure]==1).mean(dim=['ni','nj'])
			alphai = d_str[structure].mean(dim=['ni','nj']) 
			TopH[structure] = alphai * (Ui-Um)*Wi
			Intra[structure] = alphai * ( (U-Ui)*(W-Wi) ).where(d_str[structure]==1).mean(dim=['ni','nj'])
			summ = summ + TopH[structure] + Intra[structure]
		if case=='warm':
			u_star = 0.231 # m/s
			zi = 683 # m
		elif case=='cold':
			u_star = 0.2161
			zi = 507 
		# plot
		fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
		for structure in d_str.keys():
			ax.plot(TopH[structure][:]/u_star**2,Z/zi,c=colors[structure][0],label=structure+' TopH')
			ax.plot(Intra[structure][:]/u_star**2,Z/zi,c=colors[structure][1],label=structure+' Intra')
		#ax.plot((summ)[:]/u_star**2,Z/zi,c='pink',label='total') # to check that total = mean
		ax.plot(uw_m[:]/u_star**2,Z/zi,c='k',label='mean',ls='--')
		ax.set_ylim(Ylim)
		ax.set_xlim(Xlim)
		ax.set_xlabel("u'w'/u*²")
		ax.set_ylabel('z/zi')
		ax.legend()
		ax.grid()
		ax.set_title(case+' ref',loc='right')	
		fig.savefig(path_save2+'.png')
	
def std_tracers(X,Z,dsCS1,dpi):
	"""
	Plot standard deviation along dimension 'time' and 'Y' of each tracer for S1 sim
	
	INPUT
		- X 	: X dimension
		- Z 	: Z dimension
		- dsCS1 : file with conditional sampling for S1
		- dpi 	: for figures
	
	OUTPUT
		- XZ plot of std of each tracers
	
	NOTE
		- output image to be saved manually
	"""	
	std1 = dsCS1.SV1.std(dim=['time','nj'])
	std4 = dsCS1.SV4.std(dim=['time','nj'])
	std3 = dsCS1.SV3.std(dim=['time','nj'])
	
	fig, ax = plt.subplots(3,1,figsize = (10,6),constrained_layout=True,dpi=dpi)
	s = ax[0].pcolormesh(X/1000,Z/600,std1,cmap='jet',norm="log",vmin=0.01,vmax=10)
	plt.colorbar(s,ax=ax[0])
	ax[0].set_title('std for SV1 (cold)',loc='right')
	s = ax[1].pcolormesh(X/1000,Z/600,std4,cmap='jet',norm="log",vmin=0.01,vmax=10)
	plt.colorbar(s,ax=ax[1])
	ax[1].set_title('std for SV4 (warm)',loc='right')
	s = ax[2].pcolormesh(X/1000,Z/600,std3,cmap='jet',norm="log",vmin=0.01,vmax=100)
	cb = plt.colorbar(s,ax=ax[2])
	cb.ax.minorticks_on()
	
	
	y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
	cb.ax.yaxis.set_minor_locator(y_minor)
	cb.ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
	
	ax[2].set_xlabel('X (km)')
	ax[2].set_title('std for SV3 (topABL)',loc='right')
	for axe in ax:
		axe.set_ylabel('z/zi')
		axe.set_ylim([0,1.2])
	
def transport_time_for_SV4(X,Z,dsCS1,dpi):
	"""
	# temps nécessaire pour aller à une altitude donnée transporté par les thermiques.
	# goal : savoir si les thermiques à la transition sont alimentés par le sv4 et si oui depuis cb de temps/distance (= où est l'origine de cet updraft)
	# result : sv4 est injecté dans les thermiques qui sont advectés sur la zone chaude.
	
	INPUTS
		- X 	: X dimension
		- Z 	: Z dimension
		- dsCS1 : file with conditional sampling for S1
		- dpi 	: for figures
	
	OUTPUT
		- trajectory of a air particule starting at atX and moving inside an mean updraft up to atZ.
		
		
	"""	
	is_up1 = dsCS1.is_up
	is_up2 = dsCS1.is_up2
	is_up = np.logical_or(is_up1,is_up2)
	W_upm = dsCS1.WT.where(is_up==1).mean(dim=['time','nj'])
	
	fig, ax = plt.subplots(1,1,figsize = (6,3),constrained_layout=True,dpi=dpi)
	s=ax.pcolormesh(X/1000,Z/600,W_upm,cmap='jet',vmin=-1,vmax=1)
	plt.colorbar(s,ax=ax)
	ax.set_ylabel('z/zi')
	ax.set_xlabel('X (km)')
	ax.set_ylim([0,1.2])
	ax.set_title('Vitesse vertical moyenne dans les updrafts (m/s)')
	
	atX = 10000
	atZ = 0.4 # *zi
	dx = 50
	indz = nearest(Z.values,atZ*600)
	indx = nearest(X.values,atX)
	U = 5.5 # m/s
	Wp = W_upm[:,indx] # profile of W somewhat representative of the updraft velocity
	
	x1 = np.arange(0,10,0.05)
	h = np.zeros(x1.shape)
	for i in range(1,len(x1)):
		iz = nearest(Z.values,h[i-1])
		h[i] = h[i-1] + dx*Wp[iz]/U

	fig, ax = plt.subplots(1,1,figsize = (4,4),constrained_layout=True,dpi=dpi)
	ax.plot(x1,h/600,c='k')
	ax.set_ylabel('z/zi of tracer front in updraft')
	ax.set_xlabel('X=UT advected distance (U=5.5m/s)')
	
def u_and_w_fluc_in_updrafts(X,Z,dsCS1,dataSST,K,crit_value,L_atX,path_save,dpi):
	"""As the name suggest, plots u' and w' fluctuations with respect to the mean,
		at some X positions
		
	INPUT
		- X 			: X dimension
		- Z 			: Z dimension
		- dsCS1 		: file with conditional sampling of S1 simulation
		- dataSST 	: 1D SST
		- K		: threshold on the surface covered for a structure to plot profiles
		- crit_value : threshold value of SST to consider a position to be on 'cold' or 'warm' SST
		- L_atX		: List of X positions (km)
		- path_save 	: where to save figures
		- dpi		: for figures
	
	OUTPUT
		- 2 plot, 2 subplots each. W and U fluctuations inside 'updrafts' and inside 'other' (=1-updrafts) 	
			
	"""	
	map1 = 'winter'
	map2 = 'Reds'
	colorsX = DISCRETIZED_2CMAP_2(map1,map2,L_atX*1000,dataSST,crit_value,X.values)
	fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	fig2, ax2 = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	for case in dsCS1.keys():
		if case=='S1':
			is_up1 = xr.where(dsCS1[case].global_objects==1,1,0)
			is_up2 = xr.where(dsCS1[case].global_objects==5,1,0)
			is_up = np.logical_or(is_up1,is_up2)
			is_other =  np.logical_not(is_up)
			F_up = is_up.mean(dim=['time','nj'])
			F_other = is_other.mean(dim=['time','nj'])
			#VAR_up_p = np.ma.masked_where(F_up<=K, VAR_up_ma.mean(axis=(1,2)) )
			for k,atX in enumerate(L_atX):
				indx = nearest(X.values,atX*1000)
				W_up = (dsCS1[case].WT-dsCS1[case].WTm).where(is_up).mean(dim=['time','nj']).isel(ni=indx)
				U_up = (dsCS1[case].UT-dsCS1[case].UTm).where(is_up).mean(dim=['time','nj']).isel(ni=indx)
				W_other = (dsCS1[case].WT-dsCS1[case].WTm).where(is_other).mean(dim=['time','nj']).isel(ni=indx)
				U_other = (dsCS1[case].UT-dsCS1[case].UTm).where(is_other).mean(dim=['time','nj']).isel(ni=indx)
				ax[0].plot( np.ma.masked_where(F_up.isel(ni=indx)<=K,W_up), Z/ABLH_S1,c=colorsX[k],label='S1 x='+str(atX)+'km')
				ax[1].plot( np.ma.masked_where(F_up.isel(ni=indx)<=K,U_up), Z/ABLH_S1,c=colorsX[k],label='S1 x='+str(atX)+'km')
				ax2[0].plot( np.ma.masked_where(F_other.isel(ni=indx)<=K,W_other), Z/ABLH_S1,c=colorsX[k],label='S1 x='+str(atX)+'km')
				ax2[1].plot( np.ma.masked_where(F_other.isel(ni=indx)<=K,U_other), Z/ABLH_S1,c=colorsX[k],label='S1 x='+str(atX)+'km')
		else:
			if case=='warm':
				color = 'red'
			else:
				color ='blue'
			is_up = xr.where(dsCS1[case].global_objects==1,1,0)
			is_other = np.logical_not(is_up)
			F_up = is_up.mean(dim=['ni','nj'])
			F_other = is_other.mean(dim=['ni','nj'])
			zi = Z[int(dsCS1[case].THTV.mean(dim=['ni','nj']).differentiate('level').argmax())].values
			W_up = dsCS1[case].WT.where(is_up).mean(dim=['ni','nj'])
			U_up = (dsCS1[case].UT-dsCS1[case].UTm).where(is_up).mean(dim=['ni','nj'])
			W_other = dsCS1[case].WT.where(is_other).mean(dim=['ni','nj'])
			U_other = (dsCS1[case].UT-dsCS1[case].UTm).where(is_other).mean(dim=['ni','nj'])
			ax[0].plot( np.ma.masked_where(F_up<=K,W_up),Z/zi,c=color,label='ref: '+case,ls='--')
			ax[1].plot( np.ma.masked_where(F_up<=K,U_up),Z/zi,c=color,label='ref: '+case,ls='--')
			ax2[0].plot( np.ma.masked_where(F_other<=K,W_other),Z/zi,c=color,label='ref: '+case,ls='--')
			ax2[1].plot( np.ma.masked_where(F_other<=K,U_other),Z/zi,c=color,label='ref: '+case,ls='--')
	#ax[1].legend(loc='upper left') -> no need because already on the figure with updraft's buoyancy
	#ax2[1].legend(loc='upper left')
	ax[0].set_ylabel('z/$z_i$')
	ax2[0].set_ylabel('z/$z_i$')
	ax[0].set_xlabel(r"$\~w_{up}$ (m.s$^{-1}$)")
	ax[1].set_xlabel(r"$\~u_{up}$ (m.s$^{-1}$)")
	ax2[0].set_xlabel(r"$\~w_{other}$ (m.s$^{-1}$)")
	ax2[1].set_xlabel(r"$\~u_{other}$ (m.s$^{-1}$)")
	ax[0].set_xlim([-0.05,0.75])
	ax2[0].set_xlim([-0.2,0.05])
	ax[1].set_xlim([-0.75,0.05])
	ax2[1].set_xlim([-0.05,0.2])
	for k in range(len(ax)):
			ax[k].set_ylim([0,1.2])
			ax2[k].set_ylim([0,1.2])
			ax[k].grid()
			ax2[k].grid()
	fig.savefig(path_save+"_in_updrafts.png")
	fig2.savefig(path_save+"_in_other.png")
	


def C10_downdraft_detection_change(X,Z,Y,abs_path,chunksOUT,chunksNOHALO,nhalo,dpi):
	"""
	 problem : downdrafts not well detected by C10.
	 maybe the solution : change how SV3m is computed : mean('time','nj') -> mean('nj')
	 
	 to be compared before/after the modification :
		- X cover at Z/zi
		- Z cover at X
		- uw flux decomposition
		- wthtv flux decomposition
		- inst. file with downdrafts and updrafts only.
		- Hovmöller like Hovmoller_up_down_cover
	
	 Conlusion : not much change, the lack of downdraft detected with C10 is due something else
	 
	 INPUT
	 	- X 			: X dimension
		- Z 			: Z dimension
		- Y 			: Y dimension
		- abs_path 	: where the output of simulation is (=workdir)
		- chunksOUT 	: for dask, chunks when halo has not been removed
		- chunksNOHALO : for dask, chunks when halo has been removed
		- nhalo 		: MNH halo
		- dpi		: for figures
	 	
	 OUTPUT
	 	- 3 plots : * std_min before/after
	 			    * turbulent mask XZ at some t,y
	 			    * downdraft mask XZ at some t,y	
	 	- 4 plots : * X cover at some z/zi
	 				* Z cover at some X
	 				* uw flux decomposition
	 				* Hovmuller of downdraft cover (X,time)	 	
	"""
	
	NT = 121 # number of output files
				
	if False: # build the file
		start = time.time()
		print('building new downdraft file...')
		dsO = xr.open_mfdataset([abs_path+'CAS06_SST_is_Tini_SVT/FICHIERS_OUT/CAS06.1.002.OUT.'+f"{k:03}"+'.nc' for k in range(121,121+NT)],chunks=chunksOUT)
		mCS = 1
		gammaSV = 0.005
		
		# building the downdraft mask
		#	first the turbulent mask
		SV3 = dsO.SVT003[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] # 4D
		W = dsO.WT.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] # 4D
		SV3m = SV3.mean(dim=['ni','nj']) # 2D
		SV3m_3D = SV3.mean(dim=['nj']) # 3D
		std_sv3 = SV3.std(dim=['ni','nj']) # 2D
		std_sv3_3D = SV3.std(dim=['nj']) # 3D
		sv3_fluc = SV3 - SV3m_3D 
		 
		indz0,indzmax = 0, SV3m.argmax(dim='level').values + 2
		std_min = xr.zeros_like(SV3m[:,:])
		print('current time:')
		for indt in range(0,len(indzmax)):
			print(indt)
			integ = xr.ones_like(Z) * 999
			for indz in range(indz0,indzmax[indt],1):
				integ[dict(level=indz)] = std_sv3.isel(level=slice(indz,indzmax[indt]),time=indt).integrate('level') / (Z[indzmax[indt]].values-Z[indz].values)
			std_min[indt,:] = gammaSV * integ.where( integ!=999, other = integ.isel(level=indzmax[indt]-1) )
		print('std_min.shape',std_min.shape) # this should be Ntime*Nz, yes it is

		max_cond =  mCS * xr.where(std_sv3_3D > std_min, std_sv3_3D, std_min) #  <=> mCS.max(std_min,sv3.std(dim=['nj']))
		bool_turb = sv3_fluc > max_cond
		is_down =  np.logical_and( np.logical_and(bool_turb, sv3_fluc>0), W<0)
		
		data_vars = {'std_min':(['time','level'],std_min.data,{'long_name':'',
							'units':'std(sv3)'}),
						'max_cond':(['time','level','ni'],max_cond.data,{'long_name':'',
							'units':"mCS.max(std_min,sv3.std(dim=['nj']))"}),
						'std_3D':(['time','level','ni'],std_sv3_3D.data,{'long_name':'',
							'units':"SV3.mean(dim=['nj'])"}),
						'bool_turb':(['time','level','nj','ni'],bool_turb.data,{'long_name':'',
							'units':"sv3_fluc > max_cond"}),
						'is_down':(['time','level','nj','ni'],is_down.data,{'long_name':'',
							'units':"bool_turb & sv3_fluc>0 & W<0"}),
						'SV3_meanNJNI':(['time','level'],SV3m.data,{'long_name':'',
							'units':"SV3.mean(dim=['ni','nj'])"}),
						'SV3_meanNJ':(['time','level','ni'],SV3m_3D.data,{'long_name':'',
							'units':"SV3.mean(dim=['nj'])"}) }
		coords={'time':dsO.time,'level': Z,'nj':Y,'ni':X}
		ds_CS1 = xr.Dataset(data_vars=data_vars,coords=coords)
		ds_CS1.to_netcdf(path='DATA_turb/CAS06W_SVT_new_downdrafts.nc',mode='w')  
		ds_CS1.close()
		print('nb of sec to build file:',time.time() - start)
	ds = xr.open_dataset('DATA_turb/CAS06W_SVT_new_downdrafts.nc',chunks=chunksNOHALO)
	ds2 = xr.open_dataset('DATA_turb/06W_CS1_S1_C10_SVTMEAN.nc',chunks=chunksNOHALO)
	indt = -1
	indx1 = nearest(X.values,4000)
	indx2 = nearest(X.values,13000)
	indz = nearest(Z.values,100)
	indz2 = nearest(Z.values,300)
	indzi = nearest(Z.values,1.1*600)
	print('last time is :',ds.time[indt].values)
	std_min = ds.std_min
	std_sv3_3D = ds.std_3D
	max_cond = ds.max_cond
	bool_turb = ds.bool_turb
	is_down = ds.is_down
	
	is_down2 = ds2.is_down
	is_up = np.logical_or(ds2.is_up,ds2.is_up2)
	# PLOTS
	if False: # simple plots, to check that functions are working
		#std_min vs std
		fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
		ax.plot( std_min[indt,:],Z/600,c='k',label='std_min at that time' )
		ax.plot( std_sv3_3D[indt,:,indx1],Z/600,c='b',label='std at x=4km')
		ax.plot( std_sv3_3D[indt,:,indx2],Z/600,c='g',label='std at x=13km')
		ax.set_ylim([0,1.2])
		ax.set_xlabel('std_min')
		ax.set_ylabel('z/zi')
		
		# bool turb at indt=-1, indy=0
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
		s = ax.pcolormesh(X/1000,Z,bool_turb[indt,:,0,:],vmin=0,vmax=1,cmap='jet')
		plt.colorbar(s,ax=ax)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_title('bool_turb at indt=-1, indy=0')
		
		# downdraft obj at indt=-1, indy=0
		fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
		s = ax.pcolormesh(X/1000,Z,is_down[indt,:,0,:],vmin=0,vmax=1,cmap='jet')
		plt.colorbar(s,ax=ax)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_title('is_down at indt=-1, indy=0')
		
	if True: # synthetic plots
		#	- X cover at Z/zi
		F_cover = is_down.mean(dim=['time','nj'])
		F_cover2 = is_down2.mean(dim=['time','nj'])
		fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
		ax.plot(X/1000,F_cover2[indz,:]*100,c='k',label='old,z=100m')
		ax.plot(X/1000,F_cover[indz,:]*100,c='grey',label='new,z=100m')
		ax.plot(X/1000,F_cover2[indz2,:]*100,c='green',label='old,z=300m')
		ax.plot(X/1000,F_cover[indz2,:]*100,c='chartreuse',label='new,z=300m')
		ax.set_xlabel('X')
		ax.set_ylabel('downd cover (%)')
		ax.legend()
		ax.grid()
		ax.set_title('X cover at z')
		#	- Z cover at X
		fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
		ax.plot(F_cover2[:,indx2]*100,Z/600,c='k',label='old')
		ax.plot(F_cover[:,indx2]*100,Z/600,c='g',label='new')
		ax.set_xlabel('downd cover (%)')
		ax.set_ylabel('z/zi')
		ax.set_title('Z cover at X=13km')
		ax.legend()
		ax.set_ylim([0,1.2])
		ax.set_xlim([0,70])
		ax.grid()
		#	- uw flux decomposition
		u_star = 0.211 #m/s
		u_fluc = ds2.UT - ds2.UTm
		w_fluc = ds2.WT
		uw = u_fluc*w_fluc
		uw_down = uw.where(is_down==1).mean(dim=['time','nj'])
		uw_down2 = uw[:NT].where(is_down2==1).mean(dim=['time','nj']) # this is to be changed after full 121 files processed
		uw_up = uw.where(ds2.is_up==1).mean(dim=['time','nj'])
		part_down = compute_flx_contrib(uw[:NT],[is_down],meanDim=['time','nj']) # this is to be changed after full 121 files processed
		part_up,part_down2 = compute_flx_contrib(uw,[is_up,is_down2],meanDim=['time','nj']) 
		uw_mean = uw.mean(dim=['time','nj'])
		fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
		ax.plot(part_down[:,indx2]/u_star**2,Z/600,c='chartreuse',label='new down')
		ax.plot(part_down2[:,indx2]/u_star**2,Z/600,c='g',label='old down')
		ax.plot(part_up[:,indx2]/u_star**2,Z/600,c='r',label='up (2 tracers)')
		ax.plot(uw_mean[:,indx2]/u_star**2,Z/600,c='k',label='<uw>',ls='--')
		ax.plot( (part_down+part_up)[:,indx2]/u_star**2,Z/600,c='k',label='new down +up')
		ax.plot( (part_down2+part_up)[:,indx2]/u_star**2,Z/600,c='grey',label='old down +up')
		ax.set_ylim([0,1.2])
		ax.legend()
		ax.set_ylabel('z/zi')
		ax.set_xlabel("u'w'/u*²")
		ax.set_title('uw decomposition at x=13km')
		#	- wthtv flux decomposition
		#	- inst. file with downdrafts and updrafts only. -> this i'll look with ncview
		#	- Hovmöller like Hovmoller_up_down_cover
		fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
		s = ax[1].pcolormesh(X/1000,np.arange(0,len(ds.time),1),is_down.isel(level=indz).mean(dim='nj'),vmin=0,vmax=0.5,cmap='jet')
		ax[1].set_ylim([0,len(ds2.time)])
		plt.colorbar(s,ax=ax[1])
		ax[1].set_title('new down',loc='right')
		s = ax[0].pcolormesh(X/1000,np.arange(0,len(ds2.time),1),is_down2.isel(level=indz).mean(dim='nj'),vmin=0,vmax=0.5,cmap='jet')
		ax[0].set_ylim([0,len(ds2.time)])
		plt.colorbar(s,ax=ax[0])
		ax[0].set_title('old down',loc='right')
		fig.suptitle('down cover at z=100m')
		ax[1].set_xlabel('X (km)')
		ax[0].set_xlabel('X (km)')
		ax[0].set_ylabel('time (t*30s+2h)')
	
	ds.close()
	ds2.close()

def movie_coherent_structures(X,Y,Z,chunksNOHALO,L_TURB_COND,dataSST,SEUIL_ML,ini_t,ini_x,tmin,tmax,fps,stepy,stepz,Awidth,scale,path_save):
	"""
	This procedure is plotting the images needed to build a movie of the coherent structures evolution
	
	INPUTS:
		- X			: X dimension of the sim
		- Y			: Y dimension of the sim
		- Z			: Z dimension of the sim
		- L_TURB_COND : Choice of conditional sampling (C10 or ITURB2)
		- dataSST	: 1D SST(x)
		- SEUIL_ML	: thetav threshold to detect mixed layer
		- ini_t 	: integer, index of time. instant of interested to start from
		- ini_x 	: integer, index of ni. X position of interested to start from
		- tmin 		: how far back in time to look 
		- tmax 		: how far forward in time to look
		- fps 		: movie frame per seconde
		- stepy 	: vector field : skipping cells Y
		- stepz 	: vector field : skipping cells Z
		- Awidth 	: vector field : arrow width
		- scale 	: vector field : size of arrows
		- path_save : where to save images/movie
		
	OUPUTS : 
		- images to build the movie
		- command line to build the movie (with ffmpeg)
	"""	
	
	ds = xr.open_dataset('DATA_turb/S1_CS1_S1_C10_SVTMEAN.nc',chunks=chunksNOHALO)
	# getting global objects
	u_f = ds.UT-ds.UTm
	v_f = ds.VT-ds.VTm
	w_f = ds.WT-ds.WTm
	sv1_f = ds.SV1-ds.SV1m
	sv4_f = ds.SV4-ds.SV4m
	sv3_f = ds.SV3-ds.SV3m
	gTHTV = ds.THTVm[0,:,0,:].differentiate('level')
	RV,RVm = ds.RVT,ds.RVTm
	# getting advection velocity : for now is constant
	ABLH = 600 	# m
	dt = 30 	# s, OUT frequency
	dx = 50		# m, horizontal res
	indzzi = nearest(Z.values,1*ABLH)
	U = 6.53 # m/s, = dsmean.Um.isel(level=slice(0,indzzi)).mean(dim='ni').integrate('level') / ABLH
	fetch = int(np.round(U*dt/dx,0)) # to get integer
	
	# test : does the intg. advection velocity changes along x ? and if yes how much ?
	# TBD
	
	for TURB_COND in L_TURB_COND:
		path_save2 = path_save + 'T'+str(tmin)+'-T'+str(tmax)+'_t'+str(ini_t)+'_i'+str(ini_x)+'/'
		path_save_frames = path_save2 + 'frames_'+TURB_COND+'/'
		if not pathlib.Path(path_save_frames).is_dir():
			os.makedirs(path_save_frames)
		if TURB_COND=='C10':
			obj = ds.global_objects
			cmap = c.ListedColormap(['white','r','purple','g','grey','orange','pink'])
			c_range = np.arange(0,7,1)
			c_label = ['other','up','ss','down','env','up2','ss2']
			obj_max = 6.5
			
		print('	TURB_COND=',TURB_COND,'ini_t=',ini_t,'ini_x=',ini_x,'U=',U,',fetch=',fetch)
		L_t = np.zeros(tmax-tmin+1,dtype=np.int32)
		L_f = np.zeros(tmax-tmin+1,dtype=np.int32)
		RVmixed = np.zeros(tmax-tmin+1)
		
		print('	looking back up to tmin')
		t = ini_t
		f = ini_x
		while t >= tmin:
			indx = f 
			print('		time,indx =',t,indx)
			# computing rvmixed(x)
			ind1,ind2 = get_mixed_layer_indexes(Z,gTHTV[:,indx],SEUIL_ML)		
			RVmixed[t-tmin] = ( RVm.isel(level=slice(ind1,ind2+1),ni=indx,time=0,nj=0).integrate('level')/ (Z[ind2]-Z[ind1]) ).values
			L_t[t-tmin] = t
			L_f[t-tmin] = f
			t = t-1
			f = indx - fetch
			if f==0: f = 768-1 # cyclic condition
			
		print('	looking forward up to tmax')
		t = ini_t
		f = ini_x
		while t <= tmax:
			indx = f 
			print('		time,indx =',t,indx)
			# computing rvmixed(x)
			ind1,ind2 = get_mixed_layer_indexes(Z,gTHTV[:,indx],SEUIL_ML)		
			RVmixed[t-tmin] = ( RVm.isel(level=slice(ind1,ind2+1),ni=indx,time=0,nj=0).integrate('level')/ (Z[ind2]-Z[ind1]) ).values
			
			L_t[t-tmin] = t
			L_f[t-tmin] = f
			t = t+1
			f = indx + fetch
			if f==768-1: f = 0 # cyclic condition
		# Plot	
		print('	Plotting ...')
		for t in L_t:
			indx = L_f[t-tmin]
			fig = plt.figure(figsize = (15,8),layout="constrained",dpi=200)
			gs = GridSpec(4, 8, figure=fig)
			# -> SST
			ax1 = fig.add_subplot(gs[3,:])
			ax1.plot(X/1000,dataSST,c='k')
			ax1.scatter(X[indx]/1000,dataSST[indx],marker='x',c='k')
			ax1.set_ylabel('SST (K)')
			ax1.set_xlabel('X (km)')
			# -> obj
			ax2 = fig.add_subplot(gs[0:3,0:2])
			s = ax2.pcolormesh( Y/1000,Z/ABLH,obj[t,:,:,indx],cmap=cmap,vmin=-0.5,vmax=obj_max)
			Q = ax2.quiver(Y[::stepy]/1000,Z[::stepz]/ABLH,v_f[t,::stepz,::stepy,indx],w_f[t,::stepz,::stepy,indx],
					angles='uv',pivot='middle',width=Awidth,headwidth=3,headaxislength=2,headlength=2,scale=scale)
			cbar = plt.colorbar(s,ax=ax2,pad=0.05,orientation='horizontal',aspect=10)
			cbar.set_ticks(c_range)
			cbar.ax.set_xticklabels(c_label,rotation=45)
			ax2.set_aspect(2.0)
			ax2.quiverkey(Q, 0.05, 0.9, 1, '1 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
			ax2.set_ylabel('z/zi')
			ax2.set_ylim([0,1.2])
			ax2.set_title('objects',loc='right')
			# -> surface tracers
			ax3 = fig.add_subplot(gs[0:3,2:4])
			linthresh = 0.01
			vmextrm = 10
			rounding = 2 # this is linthresh = 10^(-rounding)
			sfx_tracer = xr.where(ds.SV1>ds.SV4,sv1_f.where(sv1_f>0),-sv4_f.where(sv4_f>0))
			NORM = c.SymLogNorm(linthresh=linthresh, linscale=0.03,vmin=-vmextrm, vmax=vmextrm, base=10)
			s = ax3.pcolormesh( Y/1000,Z/ABLH, sfx_tracer[t,:,:,indx],cmap='PuOr',norm=NORM) # 
			cb = plt.colorbar(s,ax=ax3,pad=0.05,orientation='horizontal',aspect=10)
			major_loc = np.round(cb.get_ticks(),rounding)
			minor_loc = Minor_ticks_symLog(major_loc,linthresh)
			if linthresh in cb.get_ticks():
				major_loc = np.delete(major_loc,[len(major_loc)//2-1,len(major_loc)//2+1])
			cb.set_ticks(minor_loc,minor=True)
			cb.set_ticks(major_loc)
			cb.set_ticklabels([str(np.abs(np.round(tick,rounding))) for tick in major_loc])
			ax3.set_ylim([0,1.2])
			ax3.set_aspect(2.0)
			ax3.set_title('surface tracers',loc='right')
			# -> top tracer
			ax4 = fig.add_subplot(gs[0:3,4:6])
			s = ax4.pcolormesh( Y/1000,Z/ABLH, sv3_f[t,:,:,indx],cmap='Blues',vmin=0.1,vmax=500,norm="log")
			plt.colorbar(s,ax=ax4,pad=0.05,orientation='horizontal',aspect=10)
			ax4.set_ylim([0,1.2])
			ax4.set_aspect(2.0)
			ax4.set_title('top tracer',loc='right')
			# -> rvmixed
			ax5 = fig.add_subplot(gs[0:3,6:])
			s = ax5.pcolormesh( Y/1000,Z/ABLH, (RV-RVmixed[t-tmin])[t,:,:,indx]*1000,cmap='BrBG',vmin=-1,vmax=1)
			plt.colorbar(s,ax=ax5,pad=0.05,orientation='horizontal',aspect=10)
			ax5.set_ylim([0,1.2])
			ax5.set_aspect(2.0)
			ax5.set_title(r'r$_v$ - r$_{v,mixed}$',loc='right')
			fig.suptitle('obj '+TURB_COND+', t='+str(t)+', i='+str(indx))
			fig.savefig(path_save_frames+f"{t:03}"+'.png')
			plt.close(fig)
		# building movie
		print(' Building movie with the following cmd:')
		print('ffmpeg -framerate '+str(fps)+' -start_number '+str(tmin)+' -i '+path_save_frames+'%03d.png '+path_save2+'movie_'+TURB_COND+'.mp4')
		#os.system('ffmpeg -framerate '+str(fps)+'-start_number '+str(tmin)+' -i '+path_save_frames+'%03d.png '+path_save+'movie.mp4')

def movie_coherent_structures_cleaner(X,Y,Z,chunksNOHALO,L_TURB_COND,dataSST,SEUIL_ML,ini_t,ini_x,tmin,tmax,fps,stepy,stepz,Awidth,scale,path_save):
	"""
	This procedure is plotting the images needed to build a movie of the coherent structures.
	Same as 'movie_coherent_structures' but with few plots and coherent structure categories.
	
	INPUTS:
		- X			: X dimension of the sim
		- Y			: Y dimension of the sim
		- Z			: Z dimension of the sim
		- L_TURB_COND : Choice of conditional sampling (C10 or ITURB2)
		- dataSST	: 1D SST(x)
		- SEUIL_ML	: thetav threshold to detect mixed layer
		- ini_t 	: integer, index of time. instant of interested to start from
		- ini_x 	: integer, index of ni. X position of interested to start from
		- tmin 		: how far back in time to look 
		- tmax 		: how far forward in time to look
		- fps 		: movie frame per seconde
		- stepy 	: vector field : skipping cells Y
		- stepz 	: vector field : skipping cells Z
		- Awidth 	: vector field : arrow width
		- scale 	: vector field : size of arrows
		- path_save : where to save images/movie
		
	OUPUTS : 
		- images to build the movie
		- command line to build the movie (with ffmpeg)
	"""
	BOOL_SST = False
	ds = xr.open_dataset('DATA_turb/S1_CS1_S1_C10_SVTMEAN_m1.nc',chunks=chunksNOHALO)
	# getting global objects
	u_f = ds.UT-ds.UTm
	v_f = ds.VT-ds.VTm
	w_f = ds.WT-ds.WTm
	gTHTV = ds.THTVm[0,:,0,:].differentiate('level')
	RV,RVm = ds.RVT,ds.RVTm
	THT,THTm = ds.THTV,ds.THTVm
	# getting advection velocity : for now is = constant
	ABLH = 600 	# m
	dt = 30 	# s, OUT frequency
	dx = 50		# m, horizontal res
	indzzi = nearest(Z.values,1*ABLH)
	U = 6.53 # m/s, = dsmean.Um.isel(level=slice(0,indzzi)).mean(dim='ni').integrate('level') / ABLH
	fetch = int(np.round(U*dt/dx,0)) # to get integer
	unity = xr.ones_like(RV)
	zeros = unity - 1
	for TURB_COND in L_TURB_COND:
		path_save2 = path_save + 'T'+str(tmin)+'-T'+str(tmax)+'_t'+str(ini_t)+'_i'+str(ini_x)+'/'
		path_save_frames = path_save2 + 'frames_clean_'+TURB_COND+'/'
		if not pathlib.Path(path_save_frames).is_dir():
			os.makedirs(path_save_frames)
		if TURB_COND=='C10':
			obj = ds.global_objects
			cmap = c.ListedColormap(['white','r','purple','orange','pink','g',])
			c_range = np.arange(0,6,1)
			c_label = ['other','up','ss','up2','ss2','down']
			obj_max = 5.5
			obj_clean = obj
			obj_clean = xr.where(obj==1,unity,zeros) # updrafts
			obj_clean = xr.where(obj==2,2*unity,obj_clean) # ss
			obj_clean = xr.where(obj==5,3*unity,obj_clean) # updrafts 2
			obj_clean = xr.where(obj==6,4*unity,obj_clean) # ss2
			obj_clean = xr.where(obj==3,5*unity,obj_clean) # downdrafts
		elif TURB_COND=='ITURB2':
			ds2 = xr.open_dataset('DATA_turb/06W_CS1_S1_ITURB2_RVMIXED.nc',chunks=chunksNOHALO)
			obj = ds2.global_objects
			cmap = c.ListedColormap(['white','r','g',])
			c_range = np.arange(0,3,1)
			c_label = ['other','up','down']
			obj_max = 2.5
			obj_clean = obj
			obj_clean = xr.where(obj==1,unity,zeros)
			obj_clean = xr.where(obj==3,2*unity,obj_clean)
			
		print('	TURB_COND=',TURB_COND,'ini_t=',ini_t,'ini_x=',ini_x,'U=',U,',fetch=',fetch)
		L_t = np.zeros(tmax-tmin+1,dtype=np.int32)
		L_f = np.zeros(tmax-tmin+1,dtype=np.int32)
		RVmixed = np.zeros(tmax-tmin+1)
		THTmixed = np.zeros(tmax-tmin+1)
		
		print('	looking back up to tmin')
		t = ini_t
		f = ini_x
		while t >= tmin:
			indx = f 
			print('		time,indx =',t,indx)
			# computing rvmixed(x)
			ind1,ind2 = get_mixed_layer_indexes(Z,gTHTV[:,indx],SEUIL_ML)		
			RVmixed[t-tmin] = ( RVm.isel(level=slice(ind1,ind2+1),ni=indx,time=0,nj=0).integrate('level')/ (Z[ind2]-Z[ind1]) ).values
			THTmixed[t-tmin] = ( THTm.isel(level=slice(ind1,ind2+1),ni=indx,time=0,nj=0).integrate('level')/ (Z[ind2]-Z[ind1]) ).values		
			L_t[t-tmin] = t
			L_f[t-tmin] = f
			t = t-1
			f = indx - fetch
			if f==0: f = 768-1 # cyclic condition
			
		print('	looking forward up to tmax')
		t = ini_t
		f = ini_x
		while t <= tmax:
			indx = f 
			print('		time,indx =',t,indx)
			# computing rvmixed(x)
			ind1,ind2 = get_mixed_layer_indexes(Z,gTHTV[:,indx],SEUIL_ML)		
			RVmixed[t-tmin] = ( RVm.isel(level=slice(ind1,ind2+1),ni=indx,time=0,nj=0).integrate('level')/ (Z[ind2]-Z[ind1]) ).values
			THTmixed[t-tmin] = ( THTm.isel(level=slice(ind1,ind2+1),ni=indx,time=0,nj=0).integrate('level')/ (Z[ind2]-Z[ind1]) ).values
			L_t[t-tmin] = t
			L_f[t-tmin] = f
			t = t+1
			f = indx + fetch
			if f==768-1: f = 0 # cyclic condition
		# Plot	
		print(' Plotting ...')
		for t in L_t:
			indx = L_f[t-tmin]
			
			if BOOL_SST:
				fig = plt.figure(figsize = (8,8),layout="constrained",dpi=200)
				gs = GridSpec(2, 4, figure=fig)
				# -> SST
				ax1 = fig.add_subplot(gs[2,:])
				ax1.plot(X/1000,dataSST,c='k')
				ax1.scatter(X[indx]/1000,dataSST[indx],marker='x',c='k')
				ax1.set_ylabel('SST (K)')
				ax1.set_xlabel('X (km)')
			else:
				fig = plt.figure(figsize = (7.5,6.1),layout="constrained",dpi=200)
				gs = GridSpec(1, 4, figure=fig)
			# -> obj
			ax2 = fig.add_subplot(gs[0,0:2])
			s = ax2.pcolormesh( Y/1000,Z/ABLH,obj_clean[t,:,:,indx],cmap=cmap,vmin=-0.5,vmax=obj_max)
			Q = ax2.quiver(Y[::stepy]/1000,Z[::stepz]/ABLH,v_f[t,::stepz,::stepy,indx],w_f[t,::stepz,::stepy,indx],
					angles='uv',pivot='middle',width=Awidth,headwidth=3,headaxislength=2,headlength=2,scale=scale)
			cbar = plt.colorbar(s,ax=ax2,pad=0.05,orientation='horizontal',aspect=10)
			cbar.set_ticks(c_range)
			cbar.ax.set_xticklabels(c_label,rotation=45)
			ax2.set_aspect(2.0)
			ax2.quiverkey(Q, 0.43, 0.17, 1, '1 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
			ax2.set_ylabel(r'z/$z_i$')
			ax2.set_ylim([0,1.2])
			ax2.set_xlabel('Y (km)')
			ax2.set_title('objects',loc='right')
			# -> thtmixed
			"""
			To add thtv, i would need to remove the warming trend from the data. 
			THTVm is a time average so if i plot thtv-thtvm it is not representative of instantaneous buoyancy
			"""
	#		ax3 = fig.add_subplot(gs[0:3,2:4])
	#		s = ax3.pcolormesh( Y/1000,Z/ABLH, (THT-THTmixed[t-tmin])[t,:,:,indx],cmap='bwr',vmin=-1,vmax=1) # 
	#		plt.colorbar(s,ax=ax3,pad=0.05,orientation='horizontal',aspect=10)
	#		ax3.set_ylim([0,1.2])
	#		ax3.set_aspect(2.0)
	#		ax3.set_title(r'$\theta_v$ - $\theta_{v,mixed}$',loc='right')

			# -> rvmixed
			ax4 = fig.add_subplot(gs[0,2:])
			s = ax4.pcolormesh( Y/1000,Z/ABLH, (RV-RVmixed[t-tmin])[t,:,:,indx]*1000,cmap='BrBG',vmin=-1,vmax=1)
			plt.colorbar(s,ax=ax4,pad=0.05,orientation='horizontal',aspect=10)
			ax4.set_ylim([0,1.2])
			ax4.set_aspect(2.0)
			ax4.tick_params(axis='both',labelleft=False)
			ax4.set_title(r'r$_v$ - r$_{v,mixed}$ (g.kg$^{-1}$)',loc='right')
			fig.suptitle('obj '+TURB_COND+', t='+str(t)+', i='+str(indx))
			ax4.set_xlabel('Y (km)')
			fig.savefig(path_save_frames+f"{t:03}"+'.png')
			plt.close(fig)
		# building movie
		print(' Building movie with the following cmd:')
		print('ffmpeg -framerate '+str(fps)+' -start_number '+str(tmin)+' -i '+path_save_frames+'%03d.png '+path_save2+'movie_clean_'+TURB_COND+'.mp4')



def One_frame_from_movieCS_C10vsITURB2(X,Y,Z,chunksNOHALO,L_atX,stepy,stepz,indt,SEUIL_ML,path_save,dpi):
	"""
	trying to understand how 'downdrafts' from ITURB2 and 'downdrafts' from C10 are (or not) related.
	
	-> ITURB2 objects 	vs rvmixed vs sv3'
	-> C10 objects 		vs rvmixed vs sv3'
	
	Plotting slices of the simulations at X positions.
	"""
	ABLH = 600
	# data opening
	ds1 = xr.open_dataset('DATA_turb/06W_CS1_S1_C10_SVTMEAN.nc',chunks=chunksNOHALO)
	ds2 = xr.open_dataset('DATA_turb/06W_CS1_S1_ITURB2_RVMIXED.nc',chunks=chunksNOHALO)
	# pronostic variables, common for the 2 files
	U,V,W = ds1.UT,ds1.VT,ds1.WT
	THT,RV,THTV = ds1.THT,ds1.RVT,ds1.THTV
	SV1,SV3,SV4 = ds1.SV1,ds1.SV3,ds1.SV4
	Um,Vm,Wm = ds1.UTm,ds1.VTm,ds1.WTm
	THTm,RVm,THTVm = ds1.THTm,ds1.RVTm,ds1.THTVm
	SV1m,SV3m,SV4m = ds1.SV1m,ds1.SV3m,ds1.SV4m
	u_f,v_f,w_f = U-Um, V-Vm, W-Wm
	sv1_f,sv3_f,sv4_f = SV1-SV1m, SV3-SV3m, SV4-SV4m
	gTHTV = THTVm[0,:,0,:].differentiate('level')
	# objects
	OBJ1 = ds1.global_objects # 0 -> 6
	OBJ2 = ds2.global_objects # 0 -> 4
	
	Awidth = 0.002
	scale = 70
	print('time is:',ds1.time[indt].values)
	for atX in L_atX:
		indx = nearest(X.values,atX*1000)
		print('i =',indx,'X=',X[indx].values/1000)
		path_save2 = path_save + str(indx) + '.png'
		# Computation of rvmixed(x)
		ind1,ind2 = get_mixed_layer_indexes(Z,gTHTV[:,indx],SEUIL_ML)		
		RVmixed = ( RVm.isel(level=slice(ind1,ind2+1),ni=indx,time=0,nj=0).integrate('level')/ (Z[ind2]-Z[ind1]) ).values
		# Plot
		fig, ax = plt.subplots(2,2,figsize = (10,10),constrained_layout=True,dpi=dpi)
		# 	obj from ITURB2
		ax[0,0].pcolormesh( Y/1000,Z/ABLH,OBJ2[indt,:,:,indx],cmap=c.ListedColormap(['white','r','purple','g','grey']),vmin=0,vmax=4)
		Q = ax[0,0].quiver(Y[::stepy]/1000,Z[::stepz]/ABLH,(V-Vm)[indt,::stepz,::stepy,indx],W[indt,::stepz,::stepy,indx],
				angles='uv',pivot='middle',width=Awidth,headwidth=3,headaxislength=2,headlength=2,scale=scale)
		#,width=0.001,headwidth=3,headaxislength=2,headlength=2,scale=100
		#,headwidth=2,headaxislength=4,scale=3
		ax[0,0].quiverkey(Q, 0.45, 0.6, 1, '1 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
		ax[0,0].set_ylim([0,1.2])
		#ax[0,0].set_xlim([Y[0]/1000,Y[-1]/1000])
		ax[0,0].set_aspect(2.0)
		ax[0,0].set_ylabel('z/zi')
		ax[0,0].set_title('obj ITURB2',loc='right')
		# 	obj from C10
		ax[0,1].pcolormesh( Y/1000,Z/ABLH,OBJ1[indt,:,:,indx],cmap=c.ListedColormap(['white','r','purple','g','grey','orange','pink']),vmin=0,vmax=6)
		Q = ax[0,1].quiver(Y[::stepy]/1000,Z[::stepz]/ABLH,(V-Vm)[indt,::stepz,::stepy,indx],W[indt,::stepz,::stepy,indx],
				angles='uv',pivot='middle',width=Awidth,headwidth=3,headaxislength=2,headlength=2,scale=scale)
		ax[0,1].set_ylim([0,1.2])
		#ax[0,1].set_xlim([Y[0]/1000,Y[-1]/1000])
		ax[0,1].set_aspect(2.0)
		ax[0,1].set_title('obj C10',loc='right')
		# 	rvmixed
		s = ax[1,0].pcolormesh( Y/1000,Z/ABLH, (RV-RVmixed)[indt,:,:,indx]*1000,cmap='BrBG',vmin=-1,vmax=1)
		plt.colorbar(s,ax=ax[1,0])
		ax[1,0].set_ylim([0,1.2])
		ax[1,0].set_xlabel('Y (km)')
		ax[1,0].set_ylabel('z/zi')
		ax[1,0].set_title(r'r$_{v,mixed}$',loc='right')
		# 	sv3'
		s = ax[1,1].pcolormesh( Y/1000,Z/ABLH, sv3_f[indt,:,:,indx],cmap='Blues',vmin=0.1,vmax=500,norm="log")
		plt.colorbar(s,ax=ax[1,1])
		ax[1,1].set_ylim([0,1.2])
		ax[1,1].set_xlabel('Y (km)')
		ax[1,1].set_title(r"s$_{v3}'$",loc='right') 
		fig.savefig(path_save2)	
		
def CS1_ref_time_mean(nhalo):
	"""testing temporal mean for CS1 warm (WIP)"""
	path_in = 'CAS10_SVT/FICHIERS_OUT/*'
	dsO = xr.open_mfdataset(path_in)
	dsB = xr.open_dataset('CAS10_SVT/CAS10.1.002.002.nc')
	case = 'warm'
	indt = -1 
	# getting the data
	U = dsO.UT.interp({'ni_u':dsO.ni})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# grid : 2
	V = dsO.VT.interp({'nj_v':dsO.nj})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# grid : 3
	W = dsO.WT.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] # grid : 4
	SV1 = dsO.SVCS000[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	SV2 = dsO.SVCS001[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	SV3 = dsO.SVCS002[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
	U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
	V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
	Um = data_mean[case]['nomean']['Mean']['MEAN_U'][indt,:]
	Vm = data_mean[case]['nomean']['Mean']['MEAN_V'][indt,:]
	Wm = data_mean[case]['nomean']['Mean']['MEAN_W'][indt,:]
	ABLH = data_mean[case]['nomean']['Misc']['BL_H'][indt]
	SV1m = data_mean[case]['nomean']['Mean']['MEAN_SV'][0,indt,:]
	SV2m = data_mean[case]['nomean']['Mean']['MEAN_SV'][1,indt,:]
	SV3m = data_mean[case]['nomean']['Mean']['MEAN_SV'][2,indt,:]
	SV1_2m = data_mean[case]['nomean']['Resolved']['RES_SV2'][0,indt,:] # no sbg sv'² in mnh
	SV2_2m = data_mean[case]['nomean']['Resolved']['RES_SV2'][1,indt,:]
	SV3_2m = data_mean[case]['nomean']['Resolved']['RES_SV2'][2,indt,:]
	SV1m = SV1m.rename(new_name_or_name_dict={'level_les':'level'})
	SV2m = SV2m.rename(new_name_or_name_dict={'level_les':'level'})
	SV3m = SV3m.rename(new_name_or_name_dict={'level_les':'level'})
	SV1_2m = SV1_2m.rename(new_name_or_name_dict={'level_les':'level'})
	SV2_2m = SV2_2m.rename(new_name_or_name_dict={'level_les':'level'})
	SV3_2m = SV3_2m.rename(new_name_or_name_dict={'level_les':'level'}) 
	Um,Vm,Wm,RVm,THTm,THTVm,Em,RV2m,ABLH= Complete_dim_like([Um,Vm,Wm,RVm,THTm,THTVm,Em,RV2m,ABLH]	 ,U[:,:,:,:])
	SV1m,SV2m,SV3m,SV1_2m,SV2_2m,SV3_2m = Complete_dim_like([SV1m,SV2m,SV3m,SV1_2m,SV2_2m,SV3_2m],U[:,:,:,:])
	raise Exception('Code to be terminated for safety')	
	
def C10vsITURB2(X,Z,Y,chunksNOHALO,dpi):
	"""
	 verifying the hypothesis of core + intromission zone 
	 both for uppdraft and downdrafts.
	
	 hypothesis : C10 detect only core while ITURB2 is detecting core+int. zone
	 results : seems ok for updraft but not at all for downdraft
	 Methods:
	 - XY plot, with contours of both detections for both structures
	 - a way of quantifying the distance between the boundaries of the same structures but detected by the 2 CS.
	
	INPUT
		- X 	: X dimension
		- Z 	: Z dimension
		- Y 	: Y dimension
		- chunksNOHALO : for dask, chunks when halo has been removed
		- dpi 	: for figures
	
	OUTPUT
		- For all height in L_atzzi, XY plot of objects (background ITURB2, contours C10) with anomaly of wind as a vector field.
		- Counts of vertical velocity for the object 'updraft' and 'downdraft' for the two conditional samplings
		
	NOTE
		- figures to be saved manually
		- for the histogram, count over (X2-X1) km, Y and time.
	"""
	
	dsC10 = xr.open_dataset('DATA_turb/06W_CS1_S1_C10_SVTMEAN.nc',chunks=chunksNOHALO)
	dsITURB = xr.open_dataset('DATA_turb/06W_CS1_S1_ITURB2_RVMIXED.nc',chunks=chunksNOHALO)
	obj = {'C10':dsC10.global_objects,'ITURB2':dsITURB.global_objects}
	U,Um = dsC10.UT,dsC10.UTm
	V,Vm = dsC10.VT,dsC10.VTm
	W = dsC10.WT
	ABLH = 600
	# 1) XY plot, with contours of both detections for both structures
	# -> very similar between the CS but in the mixed layer (bc of how is defined rvmixed ??)
	#		it suggests that the distinction is in the mixed layer
	if True:
		X1,X2 = 10,15 # km
		L_atzzi = [0.2,0.5,0.8,1.0] # in [0,1]
		indt = -1
		stepx,stepy = 2,2
		
		
		indx1 = nearest(X.values,X1*1000)
		indx2 = nearest(X.values,X2*1000)
		
		for atzzi in L_atzzi:
			indz = nearest(Z.values,atzzi*ABLH)
			fig, ax = plt.subplots(2,1,figsize = (10,10),constrained_layout=True,dpi=dpi)
			
			s = ax[0].pcolormesh(X[indx1:indx2]/1000,Y/1000,obj['ITURB2'][indt,indz,:,indx1:indx2],cmap = c.ListedColormap(['white','r','purple','g','grey']),alpha=0.5)
			#plt.colorbar(s,ax=ax)
			ax[0].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==1,1,0),levels=[0.55],colors=['r'],linewidths=1.0)
			ax[0].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==2,1,0),levels=[0.55],colors=['purple'],linewidths=1.0)
			ax[0].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==3,1,0),levels=[0.55],colors=['green'],linewidths=1.0)
			ax[0].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==4,1,0),levels=[0.55],colors=['grey'],linewidths=1.0)
			ax[0].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==5,1,0),levels=[0.55],colors=['orange'],linewidths=1.0)
			ax[0].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==6,1,0),levels=[0.55],colors=['pink'],linewidths=1.0)
			ax[0].set_ylabel('Y (km)')
			ax[0].set_aspect('equal')
			ax[0].set_title('back: ITURB2, contours:C10. At zzi='+str(atzzi),loc='right')
			ax[1].pcolormesh( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,W[indt,indz,:,indx1:indx2],cmap='coolwarm',vmin=-1,vmax=1)
			ax[1].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==1,1,0),levels=[0.55],colors=['r'],linewidths=1.0)
			ax[1].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==2,1,0),levels=[0.55],colors=['purple'],linewidths=1.0)
			ax[1].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==3,1,0),levels=[0.55],colors=['green'],linewidths=1.0)
			ax[1].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==4,1,0),levels=[0.55],colors=['grey'],linewidths=1.0)
			ax[1].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==5,1,0),levels=[0.55],colors=['orange'],linewidths=1.0)
			ax[1].contour( X[indx1:indx2]/1000+0.05/2, Y/1000+0.05/2,xr.where(obj['C10'][indt,indz,:,indx1:indx2]==6,1,0),levels=[0.55],colors=['pink'],linewidths=1.0)
			Q = ax[1].quiver(X[indx1:indx2:stepx]/1000,Y[::stepy]/1000,(U-Um)[indt,indz,::stepy,indx1:indx2:stepx],(V-Vm)[indt,indz,::stepy,indx1:indx2:stepx],
					angles='uv',pivot='middle',headwidth=2,headaxislength=4,scale=30)
			ax[1].quiverkey(Q, 0.9, 0.05, 0.5, '0.5 m/s', labelpos='E',coordinates='figure',angle=0) # Reference arrow horizontal
			ax[1].set_ylabel('Y (km)')
			ax[1].set_xlabel('X (km)')
			ax[1].set_title('Wind (arrows:UV,background:W(m/s)). At zzi='+str(atzzi),loc='right')
	
	# distribution of W to check if C10 has stronger updrafts/downdrafts
	if True:
#					W_up = {'C10':W.where(np.logical_or(obj['C10']==1,obj['C10']==5),other=-999),'ITURB2':W.where(obj['ITURB2']==1,other=-999) }
#					W_down = {'C10':W.where(obj['C10']==3,other=-999),'ITURB2':W.where(obj['ITURB2']==3,other=-999) }
		W_up = {'C10':W.where(np.logical_or(obj['C10']==1,obj['C10']==5)),'ITURB2':W.where(obj['ITURB2']==1) }
		W_down = {'C10':W.where(obj['C10']==3),'ITURB2':W.where(obj['ITURB2']==3) }
		Nbins = 50
		indt = -1
		X1,X2 = 10,15 # km
		L_atzzi = [0.2,0.5,0.8,1.0] # in [0,1]
		indx1 = nearest(X.values,X1*1000)
		indx2 = nearest(X.values,X2*1000)
		
		for atzzi in L_atzzi:
			indz = nearest(Z.values,atzzi*ABLH)
			fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
			# np.hist
#						ax.hist(W_up['C10'][indt,indz,:,indx1:indx2].values.flatten(),		density=True,bins=Nbins,range=(-2.0,2.0), histtype="step",color='r',label='upC10')
#						ax.hist(W_up['ITURB2'][indt,indz,:,indx1:indx2].values.flatten(),	density=True,bins=Nbins,range=(-2.0,2.0), histtype="step",color='orange',label='upITURB2')
#						ax.hist(W_down['C10'][indt,indz,:,indx1:indx2].values.flatten(),	density=True,bins=Nbins,range=(-2.0,2.0), histtype="step",color='g',label='downC10')
#						ax.hist(W_down['ITURB2'][indt,indz,:,indx1:indx2].values.flatten(),	density=True,bins=Nbins,range=(-2.0,2.0), histtype="step",color='chartreuse',label='downITURB2')
			# kde estimation of pdf
#						sns.kdeplot(ax=ax,data=W_up['C10'][indt,indz,:,indx1:indx2].values.flatten(),c='r',label='upC10' )
#						sns.kdeplot(ax=ax,data=W_up['ITURB2'][indt,indz,:,indx1:indx2].values.flatten(),c='orange',label='upITURB2' )
#						sns.kdeplot(ax=ax,data=W_down['C10'][indt,indz,:,indx1:indx2].values.flatten(),c='g',label='downC10' )
#						sns.kdeplot(ax=ax,data=W_down['ITURB2'][indt,indz,:,indx1:indx2].values.flatten(),c='chartreuse',label='downITURB2' )
			# kde estimation of hist
			sns.histplot(ax=ax,bins=Nbins,binrange=[-2.5,2.5],data=W_up['C10'][:,indz,:,indx1:indx2].values.flatten(),color='r',label='upC10',element="poly",fill=False )
			sns.histplot(ax=ax,bins=Nbins,binrange=[-2.5,2.5],data=W_up['ITURB2'][:,indz,:,indx1:indx2].values.flatten(),color='orange',label='upITURB2',element="poly",fill=False )
			sns.histplot(ax=ax,bins=Nbins,binrange=[-2.5,2.5],data=W_down['C10'][:,indz,:,indx1:indx2].values.flatten(),color='g',label='downC10',element="poly",fill=False )
			sns.histplot(ax=ax,bins=Nbins,binrange=[-2.5,2.5],data=W_down['ITURB2'][:,indz,:,indx1:indx2].values.flatten(),color='chartreuse',label='downITURB2',element="poly",fill=False )
			#ax.set_ylim([0,2.5])
			ax.set_xlim([-2.5,2.5])
			ax.set_ylabel('Occurences')
			ax.set_xlabel('W (m/s)')
			ax.set_title(r'X $\in$ ['+str(X1)+','+str(X2)+']km z/zi='+str(atzzi))
			ax.legend()

	
def CS_m_sensitivity(X,Y,Z,data,data_mean,dsflx,TURB_COND,L_choice,chunksNOHALO,i,t,atX,path_save,path_CS1,dpi):
	"""This procedure is looking at the sensitivity of TURB_COND conditional sampling to m
	
	
	A point is considered 'turbulent' if:
	-> for 'ITURB3'	I_turb > m*I_turb_min
	-> for 'C10' 	sv' > m.max(std_min,std)
		
		with I_turb = sqrt(2/3*E) / sqrt(Um**2+Vm**2+Wm**2)
		and I_turb_min the vertically integrated value of <I_turb>ty (see Integ_min3)
		and std_min the vertically integrated value of <std(sv)>ty (see Integ_min3)
		
	INPUTS: 
		- X 		: X dimension of the domain
		- Y 		: Y dimension of the domain
		- Z 		: Z dimension of the domain
		- data		: data to build C10 file 
		- data_mean	: mean fields
		- dsflx 	:  dataset with mean fluxes of S1
		- TURB_COND : 'C10' or 'ITURB3'
		- L_choice 	: what plot to do, see OUTPUTS
		- chunksNOHALO : chunks when halo has already been removed
		- i 		: ni index for YZ plot (not used yet)
		- t 		: time index for YZ plot (not used yet)
		- atX 		: in km, where to plot mean profiles
		- path_save : where to save the plots
		- path_CS1	: where to save the .nc
		- dpi 		: for the figures
		
	OUTPUTS:
		if '1': profiles of fluxes at atX km
		if '2': profiles of fluxes + top hat decomposition  at atX km
		if '3': cover at atX km
	"""
	#path_save = path_save + 'test_m'+TURB_COND+'/'
	
	# building files for different m values
	#  default parameters
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
	
	for m in [0.25,0.5]:
		param_COND['C10'] = (m,0.005)
		build_CS1(nhalo,data,data_mean,param_COND,L_TURB_COND,SVT_type,RV_DIFF,SEUIL_ML,indt,path_CS1)
	
	# data
	dsC10 = xr.open_dataset('DATA_turb/S1_CS1_S1_C10_SVTMEAN_m1.nc',chunks=chunksNOHALO) # DATA_turb/06W_CS1_S1_C10_SVTMEAN.nc

	data = {'0.25':xr.open_dataset('DATA_turb/S1_CS1_S1_C10_SVTMEAN_m0.25.nc',chunks=chunksNOHALO),
			'0.5':xr.open_dataset('DATA_turb/S1_CS1_S1_C10_SVTMEAN_m0.5.nc',chunks=chunksNOHALO),
			'1':xr.open_dataset('DATA_turb/S1_CS1_S1_C10_SVTMEAN_m1.nc',chunks=chunksNOHALO)}	
	Nc,figsize = 3,(12,6)
	
	ABLH = 600 	# m, ABLH of S1
	normX = 4 	# km		
	
	indx = nearest(X.values,atX*1000)
	# variables are the same, only the way turbulent structures are detetected is different
	U,Um = dsC10.UT,dsC10.UTm
	V,Vm = dsC10.VT,dsC10.VTm
	W,Wm = dsC10.WT,dsC10.WTm
	THT,THTm = dsC10.THT,dsC10.THTm
	THTV,THTVm = dsC10.THTV,dsC10.THTVm
	RV,RVm = dsC10.RVT,dsC10.RVTm
	E,Em = dsC10.E,dsC10.Em
	SV1,SV1m = dsC10.SV1,dsC10.SV1m
	SV3,SV3m = dsC10.SV3,dsC10.SV3m
	SV4,SV4m = dsC10.SV4,dsC10.SV4m
	u_f = U-Um
	v_f = V-Vm
	w_f = W-Wm
	thtv_f = THTV-THTVm
	#sv1_f = SV1-SV1m
	#sv4_f = SV4-SV4m
	#sv3_f = SV3-SV3m
	uw = u_f*w_f
	wthtv = w_f*thtv_f
	uw_mean = dsflx.FLX_UW #- dsflx.FLX_UW_s # uw_mean_tot = dsflx.FLX_UW
	wthtv_mean = dsflx.FLX_THvW
	Iturb = np.sqrt(2/3*E) / np.sqrt(Um**2+Vm**2+Wm**2)
	
	
	indnormX = nearest(X.values,normX*1000)
	u_star = dsC10.u_star[indnormX].values
	Qv0 = dsC10.Qv0[indnormX].values
	indzi = nearest(Z.values,1.1*ABLH)
	
	""" 0) """ # YZ plots, TBD
	for strm in data.keys():
		m = float(strm)
		
							
	# profiles of fluxes
	if '1' in L_choice:
		print('profiles of uw and wthtv at X='+str(atX)+'km')
		fig1, ax1 = plt.subplots(1,Nc,figsize = figsize,constrained_layout=True,dpi=dpi)
		fig2, ax2 = plt.subplots(1,Nc,figsize = figsize,constrained_layout=True,dpi=dpi)
		for k,strm in enumerate(data.keys()):
			print('	m='+strm)
			ds = data[strm]
			m = float(strm)
			obj = ds.global_objects
			is_up = xr.where(obj==1,1,0)
			is_sub = xr.where(obj==2,1,0)
			is_down = xr.where(obj==3,1,0)
			is_env = xr.where(obj==4,1,0)
			uw_up_p,uw_sub_p,uw_down_p,uw_env_p = compute_flx_contrib(uw,[is_up,is_sub,is_down,is_env],meanDim=['time','nj'])
			flx_summ = (uw_up_p + uw_sub_p + uw_down_p + uw_env_p ).isel(ni=indx)
			wthtv_up_p,wthtv_sub_p,wthtv_down_p,wthtv_env_p = compute_flx_contrib(wthtv,[is_up,is_sub,is_down,is_env],meanDim=['time','nj'])	
			flx_summ_wthtv = (wthtv_up_p + wthtv_sub_p + wthtv_down_p + wthtv_env_p ).isel(ni=indx)
			if TURB_COND=='C10':
				is_up2 = xr.where(obj==5,1,0)
				is_sub2 = xr.where(obj==6,1,0)
				uw_up_p2,uw_sub_p2 = compute_flx_contrib(uw,[is_up2,is_sub2],meanDim=['time','nj'])
				flx_summ = flx_summ + ( uw_up_p2 + uw_sub_p2 ).isel(ni=indx)
				flx_part_up2 	= mean_vertical_contrib((uw_up_p2).isel(ni=indx),(uw_mean).isel(ni=indx),indzi).values
				flx_part_sub2 	= mean_vertical_contrib((uw_sub_p2).isel(ni=indx),(uw_mean).isel(ni=indx),indzi).values
				wthtv_up_p2,wthtv_sub_p2 = compute_flx_contrib(wthtv,[is_up2,is_sub2],meanDim=['time','nj'])
				flx_summ_wthtv = (wthtv_up_p2 + wthtv_sub_p2 ).isel(ni=indx)	
				flx_part_up2 	= mean_vertical_contrib((wthtv_up_p2).isel(ni=indx),	(wthtv_mean).isel(ni=indx),indzi).values
				flx_part_sub2 	= mean_vertical_contrib((wthtv_sub_p2).isel(ni=indx),	(wthtv_mean).isel(ni=indx),indzi).values
			# uw
			norm = u_star**2
			flx_part_up 	= mean_vertical_contrib((uw_up_p).isel(ni=indx),(uw_mean).isel(ni=indx),indzi).values
			flx_part_sub 	= mean_vertical_contrib((uw_sub_p).isel(ni=indx),(uw_mean).isel(ni=indx),indzi).values
			flx_part_down 	= mean_vertical_contrib((uw_down_p).isel(ni=indx),(uw_mean).isel(ni=indx),indzi).values
			flx_part_env 	= mean_vertical_contrib((uw_env_p).isel(ni=indx),(uw_mean).isel(ni=indx),indzi).values
			flx_obj_over_all = mean_vertical_contrib(flx_summ,(uw_mean).isel(ni=indx),indzi).values
			# -> plot
			ax1[k].plot( uw_up_p[:,indx]/norm,Z/ABLH	,c='red'	,label='updrafts ('		+str(np.round(flx_part_up*100,1))	+'%)')
			ax1[k].plot( uw_sub_p[:,indx]/norm,Z/ABLH	,c='purple'	,label='sub. shells ('	+str(np.round(flx_part_sub*100,1))	+'%)')
			ax1[k].plot( uw_down_p[:,indx]/norm,Z/ABLH,c='green'	,label='downdrafts ('	+str(np.round(flx_part_down*100,1))	+'%)')
			ax1[k].plot( uw_env_p[:,indx]/norm,Z/ABLH	,c='grey'	,label='env ('			+str(np.round(flx_part_env*100,1))	+'%)')
			if TURB_COND=='C10':
				ax1[k].plot( uw_up_p2[:,indx]/norm,Z/ABLH	,c='orange'	,label='updrafts2 ('		+str(np.round(flx_part_up2*100,1))	+'%)')
				ax1[k].plot( uw_sub_p2[:,indx]/norm,Z/ABLH	,c='pink'	,label='sub. shells2 ('	+str(np.round(flx_part_sub2*100,1))	+'%)')
			ax1[k].plot( flx_summ[:]/norm,Z/ABLH		,c='k'		,label='all ('			+str(np.round(flx_obj_over_all*100,1))+'%)')
			ax1[k].plot( uw_mean[:,indx]/norm,Z/ABLH	,c='k'		,label='mean',ls='--')
			ax1[k].set_xlabel(r"$<\~ u \~ w>$/u*²")
			ax1[k].set_ylim([0,1.2])
			ax1[k].set_xlim([-1.5,0.3])
			ax1[k].grid()
			ax1[k].set_title('m='+strm,loc='right')
			# wthv
			norm = Qv0			
			flx_part_up 	= mean_vertical_contrib((wthtv_up_p).isel(ni=indx),	(wthtv_mean).isel(ni=indx),indzi).values
			flx_part_sub 	= mean_vertical_contrib((wthtv_sub_p).isel(ni=indx),	(wthtv_mean).isel(ni=indx),indzi).values
			flx_part_down 	= mean_vertical_contrib((wthtv_down_p).isel(ni=indx),	(wthtv_mean).isel(ni=indx),indzi).values
			flx_part_env 	= mean_vertical_contrib((wthtv_env_p).isel(ni=indx),	(wthtv_mean).isel(ni=indx),indzi).values
			flx_obj_over_all = mean_vertical_contrib(flx_summ_wthtv,(wthtv_mean).isel(ni=indx),indzi).values
			# -> plot
			ax2[k].plot( wthtv_up_p[:,indx]/norm,Z/ABLH	,c='red'	,label='updrafts ('		+str(np.round(flx_part_up*100,1))	+'%)')
			ax2[k].plot( wthtv_sub_p[:,indx]/norm,Z/ABLH	,c='purple'	,label='sub. shells ('	+str(np.round(flx_part_sub*100,1))	+'%)')
			ax2[k].plot( wthtv_down_p[:,indx]/norm,Z/ABLH	,c='green'	,label='downdrafts ('	+str(np.round(flx_part_down*100,1))	+'%)')
			ax2[k].plot( wthtv_env_p[:,indx]/norm,Z/ABLH	,c='grey'	,label='env ('			+str(np.round(flx_part_env*100,1))	+'%)')
			if TURB_COND=='C10':
				ax2[k].plot( wthtv_up_p2[:,indx]/norm,Z/ABLH	,c='orange'	,label='updrafts2 ('		+str(np.round(flx_part_up*100,1))	+'%)')
				ax2[k].plot( wthtv_sub_p2[:,indx]/norm,Z/ABLH	,c='pink'	,label='sub. shells2 ('	+str(np.round(flx_part_sub*100,1))	+'%)')
			ax2[k].plot( flx_summ_wthtv[:]/norm,Z/ABLH		,c='k'		,label='all ('			+str(np.round(flx_obj_over_all*100,1))+'%)')
			ax2[k].plot( wthtv_mean[:,indx]/norm,Z/ABLH		,c='k'		,label='mean',ls='--')
			ax2[k].set_title('m='+strm,loc='right')
			ax2[k].set_xlabel(r"$<\~ w \~ \theta_v>$/$Q_v^*$")
			ax2[k].set_ylim([0,1.2])
			ax2[k].set_xlim([-1.1,3.5])
			ax2[k].grid()
			ax1[k].legend(loc='upper left')
			ax2[k].legend(loc='upper right')
		ax1[0].set_ylabel('z / <zi>x')
		ax2[0].set_ylabel('z / <zi>x')
		fig1.savefig(path_save+'turbulent_fluxes_uw_x'+str(atX)+'km.png')
		fig2.savefig(path_save+'turbulent_fluxes_wthtv_x'+str(atX)+'km.png')
	
	
				
	if '2' in L_choice:
		print('profiles of uw and wthtv at X='+str(atX)+'km, with tophat decomposition for up/down/other')
		color = {'up':'red',
				'down':'green',
				'other':'blue'}
				
		# linestyles is same length as number of m tested
		if TURB_COND=='ITURB3':
			# ls = ['-','--',(5, (10, 3)),'-.',':',] if 5 values of m
			ls = ['-','--','dotted'] # if 3 values of m
		elif TURB_COND=='C10':
			ls = ['dotted','--','-'] 	
		
		fig, ax = plt.subplots(1,2,figsize = (7,5),constrained_layout=True,dpi=dpi)
		N = len(data.keys())
		for k,strm in enumerate(data.keys()):
			print('	m='+strm)
			ds = data[strm]
			m = float(strm)
			obj = ds.global_objects
			is_up = xr.where(obj==1,1,0)
			is_sub = xr.where(obj==2,1,0)
			is_down = xr.where(obj==3,1,0)
			is_env = xr.where(obj==4,1,0)
			if TURB_COND=='C10':
				is_up = np.logical_or( xr.where(obj==1,1,0),xr.where(obj==5,1,0) )
			# uw
			norm = u_star**2
			is_other = np.logical_not( np.logical_or(is_up,is_down) )
			d_str = {'up':is_up,'down':is_down,'other':is_other}
			# splitting into top hat and intra variability
			TopH,Intra = {},{}
			summ= xr.zeros_like(uw_mean)
			for structure in d_str.keys():
				Ui = U.where(d_str[structure]==1).mean(dim=['time','nj'])
				Wi = W.where(d_str[structure]==1).mean(dim=['time','nj'])
				alphai = d_str[structure].mean(dim=['time','nj']) 
				TopH[structure] = alphai * (Ui-Um[0,:,0,:])*Wi
				Intra[structure] = alphai * ( (U-Ui)*(W-Wi) ).where(d_str[structure]==1).mean(dim=['time','nj'])
				summ = summ + TopH[structure] + Intra[structure]
			for structure in d_str.keys():
				if m==1:
					ax[0].plot( TopH[structure][:,indx]/u_star**2,Z/ABLH,c=color[structure] ,ls=ls[k],label=structure)
				else:
					ax[0].plot( TopH[structure][:,indx]/u_star**2,Z/ABLH,c=color[structure] ,ls=ls[k])
				ax[1].plot( Intra[structure][:,indx]/u_star**2,Z/ABLH,c=color[structure] ,ls=ls[k])
		ax[0].plot(uw_mean[:,indx]/u_star**2,Z/ABLH,c='k',label='mean')
		ax[1].plot(uw_mean[:,indx]/u_star**2,Z/ABLH,c='k',label='mean')	
		ax[0].set_title('Top-hat')
		ax[1].set_title('Intra-variability')
		ax[0].set_xlabel(r"$<\~ u \~ w>$/u*²")
		ax[1].set_xlabel(r"$<\~ u \~ w>$/u*²")
		ax[0].set_ylabel(r'z/z$_i$')
		ax[0].legend()
		for axe in ax:
			axe.set_ylim([0,1.2])
			axe.set_xlim([-1.5,0.3])
			axe.xaxis.set_major_locator(MultipleLocator(0.5))
			axe.grid()
			axe.xaxis.label.set_fontsize(13)
			axe.yaxis.label.set_fontsize(13)
		# saving	
		fig.savefig(path_save+'turbulent_fluxes_uw_x'+str(atX)+'km_tophat.png')
		
	if '3' in L_choice:
		print('profiles of objet covers')
		fig1, ax1 = plt.subplots(1,Nc,figsize = figsize,constrained_layout=True,dpi=dpi)
		for k,strm in enumerate(data.keys()):
			print('	m='+strm)
			ds = data[strm]
			m = float(strm)
			obj = ds.global_objects
			is_up = xr.where(obj==1,1,0)
			is_sub = xr.where(obj==2,1,0)
			is_down = xr.where(obj==3,1,0)
			is_env = xr.where(obj==4,1,0)
			F_up = is_up.mean(dim=['time','nj']) 
			F_sub = is_sub.mean(dim=['time','nj']) 
			F_down = is_down.mean(dim=['time','nj'])  
			F_env = is_env.mean(dim=['time','nj']) 
			summ = F_up+F_sub+F_down+F_env
			if TURB_COND=='C10':
				is_up2 = xr.where(obj==5,1,0)
				is_sub2 = xr.where(obj==6,1,0)
				F_up2 = is_up2.mean(dim=['time','nj']) 
				F_sub2 = is_sub2.mean(dim=['time','nj']) 
				summ = summ + F_up2+F_sub2
			ax1[k].plot( F_up[:,indx],Z/ABLH	,c='red',label='updrafts')
			ax1[k].plot( F_sub[:,indx],Z/ABLH	,c='purple',label='sub. shells')
			ax1[k].plot( F_down[:,indx],Z/ABLH	,c='green',label='downdrafts')
			ax1[k].plot( F_env[:,indx],Z/ABLH	,c='grey',label='env')
			if TURB_COND=='C10':
				ax1[k].plot( F_up2[:,indx],Z/ABLH	,c='orange',label='updrafts2')
				ax1[k].plot( F_sub2[:,indx],Z/ABLH	,c='pink',label='sub. shells2')
			ax1[k].plot( summ[:,indx],Z/ABLH	,c='k',label='total of obj')
			ax1[k].set_ylim([0,1.2])
			ax1[k].grid()
			ax1[k].set_title('m='+strm,loc='right')
		ax1[0].set_ylabel('z/zi')
		ax1[0].legend()
		fig1.savefig(path_save+'cover_x'+str(atX)+'km.png')
	
	
	
def updraft_charateristics(X,Z,dsCS1,dsCS1warm,dsCS1cold,dataSST,crit_value,Tstart,Tstop,window,BLIM,L_atX,K,path_saving,dpi):
	"""Plots buoyancy, w' and u' inside updrafts from the conditionaly sampled updraft by C10.
		
		INPUTS:
			- TBD
		
	"""
	indt = -1 # last time for ref files
	path_save2 = path_saving+'C10_m'+str(dsCS1.attrs['mCS'])+'_g'+str(dsCS1.attrs['gammaRv']*100) + '_SVT/S1/'
	if not os.path.isdir(path_save2): # creat folder if it doesnt exist
		os.makedirs(path_save2)
	
	cmap_warm ='Reds'
	cmap_cold ='winter'
	colorsX = DISCRETIZED_2CMAP_2(cmap_cold,cmap_warm,L_atX*1000,dataSST,crit_value,X.values)	
	
	L_indx = []
	for x in L_atX:
		L_indx.append(nearest(X.values,x*1000))		
	
	# Variables
	is_up = xr.where(dsCS1.global_objects==1,1,0)	
	is_up_refW = xr.where(dsCS1warm.global_objects==1,1,0)	
	is_up_refC = xr.where(dsCS1cold.global_objects==1,1,0)
	is_up = np.logical_or(is_up,xr.where(dsCS1.global_objects==5,1,0))
	F_up = MeanTurb(is_up,Tstart,Tstop,window)
	F_up_warm = is_up_refW.mean(dim=['ni','nj'])
	F_up_cold = is_up_refC.mean(dim=['ni','nj'])
	# BUOYANCY
	# -> S1
	THTV = dsCS1.THTV
	THTV_up = MeanTurb(THTV.where(is_up==True),Tstart,Tstop,window)
	THTV_ref = MeanTurb(THTV,Tstart,Tstop,window)
	B = g*(THTV_up/THTV_ref - 1)
	# -> warm ref
	THTV_warm = dsCS1warm.THTV
	THTV_up_warm = THTV_warm.where(is_up_refW==1).mean(dim=['ni','nj'])
	THTV_ref_warm = THTV_warm.mean(dim=['ni','nj'])
	ABLH_warm = Z[THTV_warm.mean(dim=['ni','nj']).differentiate('level').argmax().values].values
	B_warm = g*(THTV_up_warm/THTV_ref_warm - 1)
	# -> cold ref
	THTV_cold = dsCS1cold.THTV
	THTV_up_cold = THTV_cold.where(is_up_refC==1).mean(dim=['ni','nj'])
	THTV_ref_cold = THTV_cold.mean(dim=['ni','nj'])
	ABLH_cold = Z[THTV_cold.mean(dim=['ni','nj']).differentiate('level').argmax().values].values
	B_cold = g*(THTV_up_cold/THTV_ref_cold - 1)
	# FLUCTUATIONS in updrafts
	# -> S1
	W_up = MeanTurb( (dsCS1.WT-dsCS1.WTm).where(is_up), Tstart,Tstop,window)
	U_up = MeanTurb( (dsCS1.UT-dsCS1.UTm).where(is_up), Tstart,Tstop,window)
	# -> warm ref
	W_up_w = (dsCS1warm.WT-dsCS1warm.WTm).where(is_up_refW).mean(dim=['ni','nj'])
	U_up_w = (dsCS1warm.UT-dsCS1warm.UTm).where(is_up_refW).mean(dim=['ni','nj'])
	# -> cold ref
	W_up_c = (dsCS1cold.WT-dsCS1cold.WTm).where(is_up_refC).mean(dim=['ni','nj'])
	U_up_c = (dsCS1cold.UT-dsCS1cold.UTm).where(is_up_refC).mean(dim=['ni','nj'])
	
	# PLOT
	fig, ax = plt.subplots(1,3,figsize = (10,5),constrained_layout=True,dpi=dpi)	
	# -> Buoyancy 
	ax[0].vlines(0,0,3,colors='grey',alpha=0.5)
	ax[0].plot(np.ma.masked_where(F_up_cold<=K,B_cold)*1000,Z/ABLH_cold,c='blue',label='ref: cold',ls='--')
	ax[0].plot(np.ma.masked_where(F_up_warm<=K,B_warm)*1000,Z/ABLH_warm,c='red',label='ref: warm',ls='--')
	for kx,indx in enumerate(L_indx):
		ax[0].plot(np.ma.masked_where(F_up.isel(ni=indx)<=K,B.isel(ni=indx))*1000,Z/ABLH_S1,c=colorsX[kx],label='X='+str(L_atX[kx])+'km')
	ax[0].legend()
	#ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	#ax[0].xaxis.major.formatter._useMathText = True
	ax[0].set_ylabel(r'z/z$_i$')
	ax[0].set_xlabel(r'g.(<$\theta_v$>$_{up}$/$<\theta_v>$-1)') # (10$^{-3}$ m.s$^{-2}$)
	ax[0].set_xlim(BLIM)
	# -> W
	ax[1].plot( np.ma.masked_where(F_up_warm<=K,W_up_w),Z/ABLH_warm,c='r',label='ref: warm',ls='--')
	ax[1].plot( np.ma.masked_where(F_up_cold<=K,W_up_c),Z/ABLH_cold,c='b',label='ref: cold',ls='--')
	for kx,indx in enumerate(L_indx):
		ax[1].plot( np.ma.masked_where(F_up.isel(ni=indx)<=K,W_up.isel(ni=indx)), Z/ABLH_S1,c=colorsX[kx],label='S1 x='+str(L_atX[kx])+'km')
	ax[1].set_xlabel(r"$\~w_{up}$") #  (m.s$^{-1}$)
	ax[1].set_xlim([-0.05,0.75])
	#ax[1].tick_params(axis='both',labelleft=False)
	# -> U
	ax[2].plot( np.ma.masked_where(F_up_warm<=K,U_up_w),Z/ABLH_warm,c='r',label='ref: warm',ls='--')
	ax[2].plot( np.ma.masked_where(F_up_cold<=K,U_up_c),Z/ABLH_cold,c='b',label='ref: cold',ls='--')
	for kx,indx in enumerate(L_indx):
		ax[2].plot( np.ma.masked_where(F_up.isel(ni=indx)<=K,U_up.isel(ni=indx)), Z/ABLH_S1,c=colorsX[kx],label='S1 x='+str(L_atX[kx])+'km')
	ax[2].set_xlabel(r"$\~u_{up}$") #  (m.s$^{-1}$)
	ax[2].set_xlim([-0.75,0.05])
	#ax[2].tick_params(axis='both',labelleft=False)
	
	for axe in ax.flatten():
		axe.set_ylim([0,1.2])
		axe.grid()	
		axe.xaxis.label.set_fontsize(13)
		axe.yaxis.label.set_fontsize(13)
		
	fig.savefig(path_save2+'Updrafts_characteristics.png')




def uw_decomposition_10_13_23_km(X,Z,dsCS1,dsref,dsflx,L_atX,PLOT_REF,PLOT_CONTRIB,path_saving,dpi):
	"""This procedure is plotting the uw flux decomposition by the conditional sampling C10
	
	INPUTS :
		TBD
	"""

	indt_c = -1
	indt_w = -1
	l_indx = [nearest(X.values,atx*1000) for atx in L_atX]
	indzi = nearest(Z.values,1.1*ABLH_S1)
	# saving path
	path_save2 = path_saving+'C10_m'+str(dsCS1.attrs['mCS'])+'_g'+str(dsCS1.attrs['gammaRv']*100) + '_SVT/S1/'
	if not os.path.isdir(path_save2): # creat folder if it doesnt exist
		os.makedirs(path_save2)
	# normalization	
	normX = 4 # km
	indnormX = nearest(X.values,normX*1000)
	u_star = dsCS1.u_star[indnormX].values #  = 0.211
	# Flux profiles	
	u_fluc = dsCS1.UT - dsCS1.UTm
	w_fluc = dsCS1.WT - dsCS1.WTm
	uw = u_fluc*w_fluc
	uw_mean= uw.mean(dim=['time','nj']).compute()
	uw_mean_tot = dsflx.FLX_UW
	Ones = xr.ones_like(u_fluc)
	Zeros = xr.zeros_like(u_fluc)
	is_up = xr.where( dsCS1.global_objects==1,Ones,Zeros)
	is_sub = xr.where( dsCS1.global_objects==2,Ones,Zeros )
	is_down = xr.where( dsCS1.global_objects==3,Ones,Zeros )
	is_env = xr.where( dsCS1.global_objects==4,Ones,Zeros )
	is_up2 = xr.where( dsCS1.global_objects==5,Ones,Zeros )
	is_sub2 = xr.where( dsCS1.global_objects==6,Ones,Zeros )	
	uw_mean= uw.mean(dim=['time','nj']).compute()
	uw_mean_tot = dsflx.FLX_UW
	uw_up_p,uw_sub_p,uw_up_p2,uw_sub_p2,uw_down_p,uw_env_p = compute_flx_contrib(uw,[is_up,is_sub,is_up2,is_sub2,is_down,is_env],meanDim=['time','nj'])
	flx_summ = (uw_up_p + uw_sub_p + uw_up_p2 + uw_sub_p2 + uw_down_p + uw_env_p ).isel(ni=l_indx)
	# vertically integrated contributions
	flx_part_up 	= mean_vertical_contrib((uw_up_p).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
	flx_part_sub 	= mean_vertical_contrib((uw_sub_p).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
	flx_part_up2 	= mean_vertical_contrib((uw_up_p2).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
	flx_part_sub2 	= mean_vertical_contrib((uw_sub_p2).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
	flx_part_down 	= mean_vertical_contrib((uw_down_p).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
	flx_part_env 	= mean_vertical_contrib((uw_env_p).isel(ni=l_indx),(uw_mean).isel(ni=l_indx),indzi).values
	flx_obj_over_all = mean_vertical_contrib(flx_summ,(uw_mean).isel(ni=l_indx),indzi).values
	
	# PLOT
	fig, ax = plt.subplots(1,3,figsize = (10,5),constrained_layout=True,dpi=dpi)
	for i,indx in enumerate(l_indx):
		norm = u_star**2
		if PLOT_REF: # if True, plots the uw profils from references
			uw_c = dsref['cold']['nomean']['Resolved']['RES_WU'][indt_c,:] + dsref['cold']['nomean']['Subgrid']['SBG_WU'][indt_c,:] 
			uw_w = dsref['warm']['nomean']['Resolved']['RES_WU'][indt_c,:] + dsref['warm']['nomean']['Subgrid']['SBG_WU'][indt_c,:]
			gTHTV_w = Compute_THTV( dsref['warm']['nomean']['Mean']['MEAN_TH'],
									dsref['warm']['nomean']['Mean']['MEAN_RV'])[indt_w].differentiate('level_les')
			gTHTV_c = Compute_THTV( dsref['cold']['nomean']['Mean']['MEAN_TH'],
									dsref['cold']['nomean']['Mean']['MEAN_RV'])[indt_c].differentiate('level_les')
			zi_c,  zi_w = ( Z[gTHTV_c.argmax('level_les').values].values, 
							Z[gTHTV_w.argmax('level_les').values].values )
			ax[i].plot( -uw_c/uw_c[0],Z/zi_c, c='b', label='mean refC')
			ax[i].plot( -uw_w/uw_w[0],Z/zi_w, c='r', label='mean refW')
		if PLOT_CONTRIB: # if False, plot only the total contribution of coherent structures
			ax[i].plot( uw_up_p[:,indx]/norm,Z/ABLH_S1	,c='red'	,label='up ('		+str(np.round(flx_part_up[i]*100,1))	+'%)')
			ax[i].plot( uw_sub_p[:,indx]/norm,Z/ABLH_S1	,c='purple'	,label='ss ('	+str(np.round(flx_part_sub[i]*100,1))	+'%)') # sub. shells
			ax[i].plot( uw_up_p2[:,indx]/norm,Z/ABLH_S1	,c='orange'	,label='up 2 ('	+str(np.round(flx_part_up2[i]*100,1))	+'%)')
			ax[i].plot( uw_sub_p2[:,indx]/norm,Z/ABLH_S1,c='pink'	,label='ss 2 ('+str(np.round(flx_part_sub2[i]*100,1))	+'%)')
			ax[i].plot( uw_down_p[:,indx]/norm,Z/ABLH_S1,c='green'	,label='down ('	+str(np.round(flx_part_down[i]*100,1))	+'%)')
			#ax[i].plot( uw_env_p[:,indx]/norm,Z/ABLH_S1,c='grey'	,label='env ('			+str(np.round(flx_part_env[i]*100,1))	+'%)')
			ax[i].plot( flx_summ[:,i]/norm,Z/ABLH_S1	,c='k'		,label='all ('			+str(np.round(flx_obj_over_all[i]*100,1))+'%)')
		#ax[i].plot( uw_mean[:,indx]/norm,Z/ABLH_S1	,c='k'		,label='mean',ls='--') # only resolved flux
		ax[i].plot( uw_mean_tot[:,indx]/norm,Z/ABLH_S1	,c='k'		,label='mean S1',ls='--')
		ax[i].set_title("X="+str(L_atX[i])+"km",loc='right')
		ax[i].set_xlabel("$<\~ u \~ w>$/u*²")
		ax[i].legend(loc='upper left')
		ax[i].set_ylim([0,1.2])
		ax[i].set_xlim([-1.5,0.3])
		ax[i].grid(True)
		ax[i].xaxis.label.set_fontsize(13)
		ax[i].yaxis.label.set_fontsize(13)
	ax[0].set_ylabel(r'z/z$_i$')
	fig.savefig(path_save2+'uw_flux_decomposition_10_13_23_km.png')




























	
	
	
	
	
	
	
	
	
	
	
