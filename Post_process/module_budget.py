# To be used with analyse.py 
import xarray as xr
from module_tools import *
import numpy as np
import os
import matplotlib.pyplot as plt

def BUDGET_by_NAM(X,Z,ds000,dsBU,VAR_BU,BORNES,cmap,dpi,path_budget_nam):
	"""This procedure is meant to plot the most important terms in
		the budget done by the namelist NAM_BUDGET of MNH56
		
		- dsBU is the diachronic file containing all budgets
		- VAR_BU is the dic containing selected variables
		- BORNES is the vmin vmax of each variables of which the budget is plotted
		- path_budget_nam is the location of the saved images
	"""
	# attention à bien interpoler au pt de masse pour U,W (pas V car on moyenne en Y)
	x,z = X[:-1],Z[:-1]
	# U
	fig, ax = plt.subplots(1,3,figsize = (15,5),constrained_layout=True,dpi=dpi)
	axe = ax.flatten()
	ds = dsBU['UU']
	unit = ds.AVEF.units
	fig.suptitle('U budget by NAM_BUDGET ('+unit+')')
	for k,var in enumerate(VAR_BU['UU']):
		variable = VAR_BU['UU'][k]
		axe[k].set_title(var)
		VARU_i = ds[variable][0,:,:].interp({'cart_ni_u':x}) #ds000.ni[nhalo:-nhalo]
		s = axe[k].pcolormesh(X/1000,Z,VARU_i,cmap='rainbow',vmin=BORNES['U'][0],vmax=BORNES['U'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_ylim([0,600])
	fig.savefig(path_budget_nam+'U.png')
	# V
	fig, ax = plt.subplots(1,3,figsize = (15,5),constrained_layout=True,dpi=dpi)
	axe = ax.flatten()
	ds = dsBU['VV']
	unit = ds.AVEF.units
	fig.suptitle('V budget by NAM_BUDGET ('+unit+') x10')
	for k,var in enumerate(VAR_BU['VV']): 
		variable = VAR_BU['VV'][k]
		if var !='FRCCOR':
			axe[k].set_title(var)
			s = axe[k].pcolormesh(x/1000,z,ds[variable][0,:,:]*10,cmap='rainbow',vmin=BORNES['V'][0],vmax=BORNES['V'][1])
		else:
			axe[k].set_title('COR')
			FRCCOR = ds.FRC[0,:,:] + ds.COR[0,:,:]
			s = axe[k].pcolormesh(x/1000,z,FRCCOR*10,cmap='rainbow',vmin=BORNES['V'][0],vmax=BORNES['V'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_ylim([0,600])
	fig.savefig(path_budget_nam+'V.png')
	# W
	fig, ax = plt.subplots(2,2,figsize = (7.5,5),constrained_layout=True,dpi=dpi)
	axe = ax.flatten()
	ds = dsBU['WW']
	unit = ds.AVEF.units
	fig.suptitle('W budget by NAM_BUDGET ('+unit+')')
	for k,var in enumerate(VAR_BU['WW']): #VAR_BU['WW']
		variable = VAR_BU['WW'][k]
		axe[k].set_title(var)
		if var !='BOY':
			s = axe[k].pcolormesh(x/1000,Z,ds[variable][0,:,:].interp({'cart_level_w':Z}),cmap='rainbow',vmin=BORNES['W'][0],vmax=BORNES['W'][1])
		else:
			#BOY = ds.GRAV.interp({'cart_level_w':Z})[0,:,:].values + (ds.PRES[0,:,:].interp({'cart_level_w':Z}).values+9.81*ds000.RHODREF[1:-1,0,1:-2].values)
			#s = axe[k].pcolormesh(x/1000,Z,BOY,cmap='rainbow',vmin=BORNES['W'][0],vmax=BORNES['W'][1])
			BOY = ds.GRAV[0,:,:] + ds.PRES[0,:,:]
			s = axe[k].pcolormesh(x/1000,Z,BOY.interp({'cart_level_w':Z}),cmap='rainbow',vmin=BORNES['W'][0],vmax=BORNES['W'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_ylim([0,600])
	fig.savefig(path_budget_nam+'W.png')
	# TH
	fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	axe = ax.flatten()
	ds = dsBU['TH']
	unit = ds.AVEF.units
	fig.suptitle('TH budget by NAM_BUDGET ('+unit+')')
	for k,var in enumerate(VAR_BU['TH']): #
		variable = VAR_BU['TH'][k]
		axe[k].set_title(var)
		s = axe[k].pcolormesh(x/1000,z,ds[variable][0,:,:],cmap='rainbow',vmin=BORNES['TH'][0],vmax=BORNES['TH'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_ylim([0,600])			
	ax[0].set_ylabel('Altitude (m)')
	ax[0].set_xlabel('X (km)')
	ax[1].set_xlabel('X (km)')
	fig.savefig(path_budget_nam+'TH.png')
	# RV
	fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	axe = ax.flatten()
	ds = dsBU['RV']
	unit = ds.AVEF.units
	fig.suptitle('RV budget by NAM_BUDGET ('+unit+')')
	for k,var in enumerate(VAR_BU['RV']): # ds.keys()
		variable = VAR_BU['RV'][k]
		axe[k].set_title(var)
		s = axe[k].pcolormesh(x/1000,z,ds[variable][0,:,:]*1000,cmap='rainbow',vmin=BORNES['RV'][0],vmax=BORNES['RV'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_ylim([0,600])			
	ax[0].set_ylabel('Altitude (m)')
	ax[0].set_xlabel('X (km)')
	ax[1].set_xlabel('X (km)')
	fig.savefig(path_budget_nam+'RV.png')
	
def VERIF_TENDENCY_NAM(X,Z,dsBU,ds000,VAR_BU,dpi):
	"""This procedure is meant to plot the tendency obtained from NAM_BUDGET,
		with Da/DT as the sum of all terms versus the end-begining/deltaT
		a is a pronostic variable.
	"""
	x,z = X[:-1],Z[:-1]
	vmin,vmax=-0.002,0.002
	deltaT = 14400
	L_VAR = ['UU','VV','WW','TH','RV']
	for big_var in L_VAR:
		print(big_var)
		ini,end = dsBU[big_var].INIF[0,:,:],dsBU[big_var].ENDF[0,:,:]
		if big_var=='UU':
			DaDT = (end - ini).interp({'cart_ni_u':x})
		elif big_var=='WW':
			DaDT = (end - ini).interp({'cart_level_w':z})
		else:
			DaDT = end - ini
		DaDT = DaDT / deltaT
		SUM = np.zeros(dsBU[big_var].AVEF[0,:,:].shape)
		for k,var in enumerate(VAR_BU[big_var]):
			print('	',var)
			if var=='FRCCOR':
				term = dsBU[big_var]['FRC'][0,:,:] + dsBU[big_var]['COR'][0,:,:]
			elif var=='BOY':
				term = dsBU[big_var]['GRAV'][0,:,:] + dsBU[big_var]['PRES'][0,:,:]
			else:
				term = dsBU[big_var][var][0,:,:]
			if big_var=='UU':
				term = term.interp({'cart_ni_u':x})
			elif big_var=='WW':
				term = term.interp({'cart_level_w':z})
			SUM = SUM + term
		fig, ax = plt.subplots(1,3,figsize = (15,5),constrained_layout=True,dpi=dpi)
		fig.suptitle('D('+big_var+')/DT, SUM, D('+big_var+')/DT-SUM')
		s = ax[0].pcolormesh(x/1000,z,DaDT,cmap='rainbow',vmin=vmin,vmax=vmax)
		plt.colorbar(s,ax=ax[0])
		s = ax[1].pcolormesh(x/1000,z,SUM,cmap='rainbow',vmin=vmin,vmax=vmax)
		plt.colorbar(s,ax=ax[1])
		s = ax[2].pcolormesh(x/1000,z,DaDT-SUM,cmap='bwr') #,vmin=vmin,vmax=vmax
		plt.colorbar(s,ax=ax[2])
	
def COMPARE_MEAN_FIELDS(dsmean,dsBU,X,Z):
	"""This procedure compare mean fields computed from 2 sources: 
		- by hand with OUTPUT files
		- by namelist NAM_BUDGETS
	"""
	dU = dsmean.Um[:-1,:-1].values-dsBU['UU'].AVEF[0,:,:].values
	dV = dsmean.Vm[:-1,:-1].values-dsBU['VV'].AVEF[0,:,:].values
	dW = dsmean.Wm[:-1,:-1].values-dsBU['WW'].AVEF[0,:,:].values
	N = 40
	Nt = 767*159 # = Nx*Nz
	U_h = np.histogram(dU,N)
	V_h = np.histogram(dV,N)
	W_h = np.histogram(dW,N)
	print('		Mean difference U,V,W :',np.round(dU.mean(),5),np.round(dV.mean(),5),np.round(dW.mean(),5))
	print('		Std for U,V,W         :',np.round(np.std(dU),5),np.round(np.std(dV),5),np.round(np.std(dW),5))
	fig, ax = plt.subplots(3,1,figsize = (5,10),constrained_layout=True,dpi=100)
	ax[1].set_ylabel('Altitude (m)')
	ax[2].set_xlabel('X (km)')
	ax[0].set_title('U,V,W difference by hand - by NAM_BUDGET')
	s = ax[0].pcolormesh(X[:-1]/1000,Z[:-1],dU,cmap='bwr',vmin=-0.05,vmax=0.05)
	plt.colorbar(s,ax=ax[0])
	s = ax[1].pcolormesh(X[:-1]/1000,Z[:-1],dV,cmap='bwr',vmin=-0.05,vmax=0.05)
	plt.colorbar(s,ax=ax[1])
	s = ax[2].pcolormesh(X[:-1]/1000,Z[:-1],dW,cmap='bwr',vmin=-0.05,vmax=0.05)
	plt.colorbar(s,ax=ax[2])
	fig, ax = plt.subplots(3,1,figsize = (5,10),constrained_layout=True,dpi=100)
	ax[0].set_title('PDF dU,dV,dW (by hand - by NAM_BUDGET')
	ax[0].plot(U_h[1][:-1],U_h[0]/Nt,c='k')
	ax[1].plot(V_h[1][:-1],V_h[0]/Nt,c='k')
	ax[2].plot(W_h[1][:-1],W_h[0]/Nt,c='k')
	for axe in ax.flatten():
		axe.set_xlim([-0.05,0.05])
def BUDGET_by_HAND(X,Z,ds_hbudget,VAR_BU,BORNES,cmap,path_budget_hand,dpi):
	"""This procedure is meant to plot the budget terms
	   of U,V,W,Total TKE with the terms computed by hand 
	   with OUTPUT files from the MNH simulation.
	   
	"""
	x,z = X[:-1],Z[:-1]
	
	# U
	fig, ax = plt.subplots(2,3,figsize = (10,5),constrained_layout=True,dpi=dpi)
	axe=ax.flatten()
	for k,term in enumerate(VAR_BU['U']):
		s = axe[k].pcolormesh(x/1000,z,ds_hbudget[term],cmap=cmap,vmin=BORNES['U'][0],vmax=BORNES['U'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_title(term+' ('+ds_hbudget[term].units+')')
		axe[k].set_ylim([0,600])
	fig.savefig(path_budget_hand+'U.png')
	# V
	fig, ax = plt.subplots(2,3,figsize = (10,5),constrained_layout=True,dpi=dpi)
	axe=ax.flatten()
	for k,term in enumerate(VAR_BU['V']):
		s = axe[k].pcolormesh(x/1000,z,ds_hbudget[term],cmap=cmap,vmin=BORNES['V'][0],vmax=BORNES['V'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_title(term+' ('+ds_hbudget[term].units+')')
		axe[k].set_ylim([0,600])
	axe[-1].set_axis_off()
	fig.savefig(path_budget_hand+'V.png')
	# W
	fig, ax = plt.subplots(2,3,figsize = (10,5),constrained_layout=True,dpi=dpi)
	axe=ax.flatten()
	for k,term in enumerate(VAR_BU['W']):
		if term=='w_boytotale':
			data = ds_hbudget['w_pres'] + ds_hbudget['w_grav'] + ds_hbudget['w_cor']
			UNITS = ds_hbudget['w_pres'].units
		else:
			data = ds_hbudget[term]
			UNITS = ds_hbudget[term].units
		s = axe[k].pcolormesh(x/1000,z,data,cmap=cmap,vmin=BORNES['W'][0],vmax=BORNES['W'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_title(term+' ('+UNITS+')')
		axe[k].set_ylim([0,600])
	fig.savefig(path_budget_hand+'W.png')
	# Total TKE
	fig, ax = plt.subplots(2,4,figsize = (15,5),constrained_layout=True,dpi=dpi)
	axe=ax.flatten()
	for k,term in enumerate(VAR_BU['ET']):
		s = axe[k].pcolormesh(x/1000,z,ds_hbudget[term],cmap=cmap,vmin=BORNES['ET'][0],vmax=BORNES['ET'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_title(term+' ('+ds_hbudget[term].units+')')
		axe[k].set_ylim([0,600])
	axe[-1].set_axis_off()
	fig.savefig(path_budget_hand+'TKE.png')
	# Resolved TKE
	fig, ax = plt.subplots(2,4,figsize = (15,5),constrained_layout=True,dpi=dpi)
	axe=ax.flatten()
	for k,term in enumerate(VAR_BU['E']):
		s = axe[k].pcolormesh(x/1000,z,ds_hbudget[term],cmap=cmap,vmin=BORNES['E'][0],vmax=BORNES['E'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_title(term+' ('+ds_hbudget[term].units+')')
		axe[k].set_ylim([0,600])
	axe[-1].set_axis_off()
	fig.savefig(path_budget_hand+'E_r.png')
	# Subgrid TKE
	fig, ax = plt.subplots(2,4,figsize = (15,5),constrained_layout=True,dpi=dpi)
	axe=ax.flatten()
	for k,term in enumerate(VAR_BU['TKET']):
		s = axe[k].pcolormesh(x/1000,z,ds_hbudget[term],cmap=cmap,vmin=BORNES['TKET'][0],vmax=BORNES['TKET'][1])
		plt.colorbar(s,ax=axe[k])
		axe[k].set_title(term+' ('+ds_hbudget[term].units+')')
		axe[k].set_ylim([0,600])
	fig.savefig(path_budget_hand+'E_s.png')
	
def VERIF_TENDENCY_HAND(X,Z,ds_hbudget,VAR_BU,dpi,path_budget_hand):
	x,z = X[:-1],Z[:-1]
	VAR_BU = {'U':['u_cor','u_hadv','u_hturb','u_pres','u_vadv','u_vturb'],
		'V':['v_cor','v_hadv','v_hturb','v_vadv','v_vturb'],
		'W':['w_hadv','w_hturb','w_pres','w_vadv','w_vturb','w_grav','w_cor'],
		'ET':['ET_DIFF','ET_DISS','ET_HADV','ET_HDP','ET_TP','ET_VADV','ET_VDP']}
	fig, ax = plt.subplots(2,2,figsize = (10,5),constrained_layout=True,dpi=dpi)
	axe = ax.flatten()
	for k,key in enumerate(VAR_BU.keys()):
		SOMME = np.zeros(ds_hbudget['u_cor'].shape)
		for var in VAR_BU[key]:
			SOMME = SOMME + ds_hbudget[var]
		s = axe[k].pcolormesh(x/1000,z,SOMME,vmin=-0.001,vmax=0.001,cmap='bwr')
		plt.colorbar(s,ax=axe[k])
		axe[k].set_title('TENDENCY of '+key)
		axe[k].set_ylim([0,600])
	fig.savefig(path_budget_hand+'TENDENCY.png')		



def X_BUDGET_NAM(dsBU,X,Z,height,VAR_BU,COLORS,BORNES,path_budget_nam,dpi):
	"""This procedure is plotting the dominants terms in budgets from NAM_BUDGET
		with respect to X position, at differents heights
	"""
	x = X[:-1]
	deltaT = 14400
	indz =  []
	for z in height:
		indz.append(np.argmin(np.abs(Z.values-z)))
	for var in ['UU','VV','WW','RV','TH']:
			units,coeff = dsBU[var].AVEF.units,1.0
			if var=='RV':
				units,coeff='g kg-1',1000
			fig, ax = plt.subplots(len(height),1,figsize = (10,10),constrained_layout=True,dpi=dpi)
			for k,jz in enumerate(indz):
				for term in VAR_BU[var]:
					if term=='FRCCOR':
						data = dsBU[var]['COR'][0,jz,:] + dsBU[var]['FRC'][0,jz,:]
					elif term=='BOY':
						data = dsBU[var]['PRES'][0,jz,:] + dsBU[var]['GRAV'][0,jz,:]
					elif term=='TEND':
						data = (dsBU[var]['ENDF'][0,jz,:] - dsBU[var]['INIF'][0,jz,:])/deltaT								
					else:
						data = dsBU[var][term][0,jz,:] 
					ax[k].plot(x/1000,data*coeff,color=COLORS[term],label=term)
				ax[k].set_title('z='+str(np.round(Z[jz].values,1))+'m',loc='right')
				ax[k].set_ylim(BORNES[var])
				ax[k].set_xlim([(x/1000)[0],(x/1000)[-1]])
			ax[0].legend(loc='upper right')
			fig.suptitle(var+' budget ('+units+')')			
			fig.savefig(path_budget_nam+'X_'+var+'.png')	

def X_BUDGET_HAND(dataSST,dsB,ds_hbudget,X,Z,height,VAR_BU,COLORS,BORNES,Factor,path_budget_hand,dpi):
	"""This procedure is plotting the dominants terms in budgets from hand made budget
		with respect to X position, at differents heights
		
		INPUTS : TBD
	"""
	

	#x = X[:-1]
	x=X
	indz =  []
	EXCLUDE = ['u_hadv','u_vadv','u_hturb','ET_HADV','THT_HADV','THT_VADV','THT_HTURB']
	translate = {'u_cor':'coriolis','u_pres':'pressure','u_vturb':'stress divergence'}
	for z in height:
		indz.append(np.argmin(np.abs(Z.values-z)))
	for var in ['u']: # ['u','v','w','ET','tht']
		print(var)
		#units = ds_hbudget[VAR_BU[var][0]].units
		if var in ['u','v','w']:
			units = r'm.s$^{-2}$'
			adv = ds_hbudget[var+'_hadv']
		elif var=='ET':
			units = r'm$^2$.s$^{-3}$'
			adv = ds_hbudget['ET_HADV']
		elif var=='tht':
			units = r'K.s$^{-1}$'
			adv = ds_hbudget['THT_HADV']
		else:
			units = 'no units'
			adv = 0
		fig, axe = plt.subplots(2,2,figsize = (10,6),constrained_layout=True,dpi=dpi)
		ax = axe.flatten()
		SUM = np.zeros((len(indz),ds_hbudget['u_cor'].shape[1]))
		for term in VAR_BU[var]:
			print('	'+term)
			for j,jz in enumerate(indz):	
				k = -j
				if term=='w_boytotale':
					data = ds_hbudget['w_pres'][indz[k],:] + ds_hbudget['w_grav'][indz[k],:]
				else:
					data = ds_hbudget[term][indz[k],:] 
				data = data
				SUM[k] = SUM[k] + data
				if var=='ET':
					color = COLORS[term[3:]]
				elif var=='tht':
					color = COLORS[term[4:]]
				else:
					color = COLORS[term[2:]]
				if term not in EXCLUDE:
					ax[k].plot(x/1000,data*Factor,color=color,label=translate[term])
					
				#if k!=len(indz)-1:
				#	ax[k].tick_params(axis='both',labelbottom=False)
				ax[k].set_title('z='+str(height[k])+'m',loc='right', y=0.88)
				ax[k].set_ylim(BORNES[var])
				ax[k].set_xlim([(x/1000)[0],(x/1000)[-1]])
				Add_SST_bar(X.values/1000,np.linspace(BORNES[var][0],BORNES[var][1],200),4,dataSST,ax[k])
				ax[k].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
				ax[k].yaxis.major.formatter._useMathText = True
				ax[k].set_xlabel('X (km)')		
		for j,jz in enumerate(indz):
			k = -j
			if var=='tht':
				ax[k].plot(x/1000,- (SUM[k,:] - adv[indz[k],:])*Factor,color='k',label=r'-D$\theta$/Dt')
			elif var=='u':
				ax[k].plot(x/1000,- (SUM[k,:] - adv[indz[k],:])*Factor,color='k',label='-D<U>/Dt') # ,ls='-.'
			ax[k].hlines(0,0,(x/1000)[-1],colors='grey',ls='--')			
			ax[k].xaxis.label.set_fontsize(13)
			ax[k].yaxis.label.set_fontsize(13)
		ax[-1].legend(fontsize=11,loc='lower right')
		
		# zones
		ax[0].vlines(5,-4,4,colors='k',linestyles='--')
		ax[0].vlines(15,-4,4,colors='k',linestyles='--')
		ax[0].vlines(25,-4,4,colors='k',linestyles='--')

		fig.savefig(path_budget_hand+'X_'+var+'.png')	
		
		
		
def PROFILES_BUDGET_HAND(ds_hbudget,X,Z,VAR_BU,BORNES,COLORS,NORM,path_budget_hand,dpi,zones):
	"""The purpose of this procedure is to plot the profiles of the budget terms
		inside the domain defined by 'zones'.
		
		INPUTS : TBD
	"""
	ABLH = 600 #m
	EXCLUDE = ['u_hadv','u_vadv','u_hturb','ET_HADV','THT_HADV','THT_VADV','THT_HTURB']
	if NORM:
		zi = ABLH
		NAMEY,NAMESAVE,YLIM = 'z/zi','_NORM',[0,1.1]
	else:
		zi = 1
		NAMEY,NAMESAVE,YLIM = 'Altitude (m)','',[0,600]
	colors_sgs = {'e_ADVR_s':'orange','e_DIFF_s':'blue',
					'e_DISS_s':'magenta','e_DPR_s':'chartreuse',
					'e_HADVM_s':'sienna','e_HDPM_s':'lime',
					'e_TP_s':'red','e_VADVM_s':'tan',
					'e_VDPM_s':'green'}
	for var in VAR_BU.keys():
		print(var)
		if var=='e' and NORM: YLIM = [0,0.3]
		elif var=='e' and not NORM: YLIM = [0,0.3*ABLH]
		units = ds_hbudget[VAR_BU[var][0]].units
		fig, ax = plt.subplots(1,len(zones),figsize = (5*len(zones),5),constrained_layout=True,dpi=dpi)	
		for k,ZONE in enumerate(zones):
			ax[k].vlines(0,0,600,colors='grey',linestyle='--')
			print('	'+str(zones[ZONE])+'km')
			indx = np.argmin(np.abs(X.values-zones[ZONE]*1000))
			SUM = np.zeros(ds_hbudget['u_cor'].shape[0])
			for term in VAR_BU[var]:
				print('		'+term)
				if term=='w_boytotale':
					data =  ds_hbudget['w_pres'][:,indx] + ds_hbudget['w_grav'][:,indx] 
				else:
					data =  ds_hbudget[term][:,indx] 
				SUM = SUM + data
				if var=='ET':
					color = COLORS[term[3:]]
					adv = ds_hbudget['ET_HADV'][:,indx]
				elif var=='e':
					color = colors_sgs[term]
				elif var in ['u','v','w']:
					color = COLORS[term[2:]]
					adv = ds_hbudget[var+'_hadv'][:,indx]
				elif var=='tht':
					color = COLORS[term[4:]]
					adv = ds_hbudget['THT_HADV'][:,indx]
				if term not in EXCLUDE:
					ax[k].plot(data,Z/zi,color=color,label=term)
			ax[k].plot(- (SUM[:] - adv[:]),Z/zi,color='k',label='-D'+var+'/Dt')
			ax[k].set_title('X at '+str(zones[ZONE])+'km',loc='right',size='small',pad=3)
			ax[k].set_xlim(BORNES[var])
			ax[k].set_ylim(YLIM)
			ax[k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax[k].xaxis.major.formatter._useMathText = True
			if k>0:
				ax[k].tick_params(axis='both',labelleft=False)
		ax[0].legend(loc='upper right')
		ax[0].set_ylabel(NAMEY)
		fig.suptitle(var+' budget ('+units+')')	
		fig.savefig(path_budget_hand+'Z_'+var+NAMESAVE+'.png')	
	
def PROFILES_FLX_BUDGET(ds_flxbudget,dsmean,X,Z,VAR_BU,BORNES,COLORS,path_budget,dpi,zones):
	"""The purpose of this procedure is to plot the profiles of the budget terms
		of the fluxes budget
		inside the domain defined by 'zones'.
	"""	
	#Z = Z[:-1]	
	for var in ['uw','wtht','wrv','wthtv']:
		print(var)
		
		fig, ax = plt.subplots(1,len(zones),figsize = (5*len(zones),5),constrained_layout=True,dpi=dpi)	
		for k,ZONE in enumerate(zones):
			ax[k].vlines(0,0,600,colors='grey',linestyle='--')
			print('	'+ZONE)
			indx = np.argmin(np.abs(X.values-zones[ZONE]*1000))
			SUM = np.zeros(ds_flxbudget['uw_PRES'].shape[0])
			for term in VAR_BU[var]:
				print('		'+term)
				if var=='wthtv': # w'thtv' = thtm*1.61*w'rv' +  w'tht'*(1+1.61*rvm)
					thtm = dsmean.THTm[:,indx]
					rvm = dsmean.RVTm[:,indx]
					data = ( (1+1.61*rvm)*ds_flxbudget['wtht'+term[len(var):]][:,indx] 
						+ thtm*1.61*ds_flxbudget['wrv'+term[len(var):]][:,indx] 
						)
					units = ds_flxbudget[VAR_BU['wtht'][0]].units
				else:
					units = ds_flxbudget[VAR_BU[var][0]].units
					data =  ds_flxbudget[term][:,indx] 
				SUM = SUM + data
				color = COLORS[term[len(var)+1:]]
				ax[k].plot(data,Z,color=color,label=term)
			ax[k].plot(SUM,Z,color='k',label='Sum',ls='-.')
			ax[k].set_title('X at '+str(zones[ZONE])+'km',loc='right',size='small',pad=3)
			ax[k].set_xlim(BORNES[var])
			ax[k].set_ylim([0,600])
			ax[k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			ax[k].xaxis.major.formatter._useMathText = True
			if k>0:
				ax[k].tick_params(axis='both',labelleft=False)
			if var=='uw':
				ax[k].set_xlabel('GAIN        LOSS')
			elif var=='wrv':
				ax[k].set_xlabel('LOSS        GAIN')
		ax[0].legend(loc='upper right')
		ax[0].set_ylabel('Altitude (m)')
		fig.suptitle(var+' budget ('+units+')')	
		fig.savefig(path_budget+'Z_'+var+'.png')	
	
def split_TR_uw(X,Z,dsO,dsmean,path_budget_hand,dpi):
	"""
	"""
	Tstart = dsmean.attrs['Tstart']
	Tstop = dsmean.attrs['Tstop']
	window = dsmean.attrs['window']
	Time = dsO.time
	U = dsO.UT.interp({'ni_u':dsO.ni})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 		# grid : 2
	W = dsO.WT.interp({'level_w':dsO.level})[:,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# grid : 4
	U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
	Um = dsmean.Um[:,:] # here the halo has already been removed (in procedure build_mean)
	Wm = dsmean.Wm[:,:]
#	Um3D = Um.expand_dims(dim={"nj":dsO.nj,"time":Time},axis=(2,0))
#	Wm3D = Wm.expand_dims(dim={"nj":dsO.nj,"time":Time},axis=(2,0))
	Um3D,Wm3D = Complete_dim_like([Um,Wm],U)
	u_fluc = (U - Um3D)
	w_fluc = (W - Wm3D)
	uwu = u_fluc*w_fluc*u_fluc
	uww = u_fluc*w_fluc*w_fluc
	H_TR = - np.gradient(MeanTurb(uwu,Tstart,Tstop,window),X,axis=1)
	V_TR = - np.gradient(MeanTurb(uww,Tstart,Tstop,window),Z,axis=0)
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	s = ax.pcolormesh(X/1000,Z,H_TR,cmap='bwr',vmin=-0.001,vmax=0.001)
	plt.colorbar(s,ax=ax)
	ax.set_ylabel('Altitude (m)')
	ax.set_xlabel('X (km)')
	ax.set_title(r"-d$\overline{u'w'u'}$/dx")
	ax.set_ylim([0,600])
	fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
	s = ax.pcolormesh(X/1000,Z,V_TR,cmap='bwr',vmin=-0.001,vmax=0.001)
	plt.colorbar(s,ax=ax)
	ax.set_ylabel('Altitude (m)')
	ax.set_xlabel('X (km)')
	ax.set_title(r"-d$\overline{u'w'w'}$/dz") # <--------- this is the important term
	ax.set_ylim([0,600])
	fig.savefig(path_budget_hand+'uw_TR_V.png')
	
def XZ_FLX_BUDGET(ds_flxbudget,dsmean,X,Z,VAR_BU,BORNES,path_budget_hand,dpi,zones):
	"""Plots the XZ budget terms of turbulent flux budgets
	"""
	print('TBD ?')	
	
	
def PROFILES_BUDGET_HAND_ref(dsref,Inst_warm,Inst_cold,Z,VAR_BU,BORNES,COLORS,zones,path_save,dpi):
	"""
	Plot profiles of the budgets terms for the reference simulations, at different X positions.
	
	INPUTS:TBD
	"""
	indt = -1 # last time is used in reference simulations
	
	figU, axU = plt.subplots(1,2,figsize = (7,5),constrained_layout=True,dpi=dpi)	
	
	for k,case in enumerate(['cold','warm']):
		if case=='cold': Inst=Inst_cold 
		else:Inst=Inst_warm # ref state is mean state : <>xy
		# getting data
		U = Inst.UT.interp({'ni_u':Inst.ni})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		V = Inst.VT.interp({'nj_v':Inst.nj})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		W = Inst.WT.interp({'level_w':Inst.level})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		THT = Inst.THT[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		THTV = Compute_THTV(Inst)[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		U = U.rename(new_name_or_name_dict={'nj_u':'nj'})
		V = V.rename(new_name_or_name_dict={'ni_v':'ni'})
		UW_HFLX = Inst.UW_HFLX.interp({'level_w':Inst.level,'ni_u':Inst.ni})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# Grid : 6
		UW_VFLX = Inst.UW_VFLX.interp({'level_w':Inst.level})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]					# Grid : 4
		VW_HFLX = Inst.VW_HFLX.interp({'level_w':Inst.level,'nj_v':Inst.nj})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo] 	# Grid : 7
		VW_VFLX = Inst.VW_VFLX.interp({'level_w':Inst.level})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]					# Grid : 4
		THW_FLX = Inst.THW_FLX.interp({'level_w':Inst.level})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]					# Grid : 4
		RCONSW_FLX = Inst.RCONSW_FLX.interp({'level_w':Inst.level})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]			# Grid : 4
		UV_FLX = Inst.UV_FLX.interp({'ni_u':Inst.ni,'nj_v':Inst.nj})[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		U_VAR = Inst.U_VAR[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]		# Grid : 1
		V_VAR = Inst.V_VAR[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]		# Grid : 1
		W_VAR = Inst.W_VAR[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]		# Grid : 1
		UW_HFLX = UW_HFLX.rename(new_name_or_name_dict={'nj_u':'nj'})
		VW_HFLX = VW_HFLX.rename(new_name_or_name_dict={'ni_v':'ni'})
		UW_FLX = UW_HFLX + UW_VFLX
		VW_FLX = VW_HFLX + VW_VFLX
		THvW_FLX = THW_FLX*THTV/THT + 0.61*THT*RCONSW_FLX 
		P = Inst.PABST[0,nhalo:-nhalo,nhalo:-nhalo,nhalo:-nhalo]
		dpdz = P.differentiate('level')
		ABLH = dsref[case]['nomean']['Misc']['BL_H'][indt].values
		Um = U.mean(dim=['ni','nj'])
		Vm = V.mean(dim=['ni','nj'])
		THTVm = THTV.mean(dim=['ni','nj'])
		Pm = P.mean(dim=['ni','nj'])
		# fluctuations and total fluxes
		u_fluc,v_fluc,w_fluc = (U-Um),(V-Vm),W
		thtv_fluc = THTV-THTVm
		uv = (u_fluc*v_fluc) + UV_FLX
		uw = (u_fluc*w_fluc) + UW_FLX
		vw = (v_fluc*w_fluc) + VW_FLX
		uu = (u_fluc*u_fluc) + U_VAR
		vv = (v_fluc*v_fluc) + V_VAR
		ww = ( w_fluc*w_fluc) + W_VAR
		wthtv = (thtv_fluc*w_fluc) + THvW_FLX
		
		# U budget
		if 'U' in VAR_BU:
			print('U '+case)
			#u_hadv = - Um*Um.differentiate('ni')
			#u_vadv = - Wm*Um.differentiate('level')
			#u_hturb = - ( uu.mean(dim=['ni','nj']) ).differentiate('ni')
			u_vturb = - ( uw.mean(dim=['ni','nj']) ).differentiate('level')
			#u_pres = - Pm.differentiate('ni')
			u_cor = f*Vm
			# plot
			axU[k].vlines(0,0,600,colors='grey',linestyle='--')
			SUM = u_cor + u_vturb
			axU[k].plot(u_cor,Z/ABLH,color=COLORS['cor'],label='u_cor')
			#axU[k].plot(u_hadv,Z/ABLH,color=COLORS['hadv'],label='u_hadv') # no adv !
			#axU[k].plot(u_pres,Z/ABLH,color=COLORS['pres'],label='u_pres')
			axU[k].plot(u_vturb,Z/ABLH,color=COLORS['vturb'],label='u_vturb')
			axU[k].plot(SUM,Z/ABLH,color='k',label='Sum',ls='-.')
			axU[k].set_title(case)
			axU[k].set_xlim(BORNES['U'])
			axU[k].set_ylim([0,1.1])
			axU[k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
			axU[k].xaxis.major.formatter._useMathText = True
			axU[k].legend(loc='upper right')
			axU[k].set_ylabel('z/zi')
			
			
	# closing figures and saving		
	figU.suptitle('U budget (m s-2)')	
	figU.savefig(path_save+case+'_Z_U_NORM.png')	
			
		# V budget
		# W budget
		# ET budget : available in 000
		
		
	
	
	
	
	
	
	
	
	
