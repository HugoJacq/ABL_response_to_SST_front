# python setup.py
import os

"""
This script is building the namlist and executables for a MesoNH simulation.

The case studied here is a change in radioactive decay sensitivity study.
	
"""



L_nu = [1,4,7,10,12,15,20,30,40] # in minutes
seg_length = '3600.' # for paper : 3600.
HPC = True


if HPC:
	path_profile = '. /home/cnrm_other/ge/mrmc/jacqueth/MNH-V5-6-2/conf/profile_mesonh-LXifort-R8I4-MNH-V5-6-2-2tracers_SBGinOUT-MPIAUTO-O3'
	core = 512
	node = 4
	run_line = 'time ${MPIRUN} MESONH${XYZ} #belenos cmd'
else:
	path_profile = '. /home/jacqhugo/MNH-V5-6-2/conf/profile_mesonh-LXgfortran-R8I4-MNH-V5-6-2-2tracers_SBGinOUT-MPIAUTO-DEBUG'
	core = 16
	node = 4 # not used
	run_line = '${TIMEOUT} time -v ${MPIRUN} MESONH${XYZ} # local cmd'


def Check_file(name):
	"""Check if file is here,
		if yes : rename it by adding OLD at its end
		if no : create it
	"""
	if os.path.exists(name):
		os.system('mv '+name+' '+name+'OLD')
	else:
		os.system('touch '+name)


for nu in L_nu:
	if nu<10 and nu>=1:
		NAME = 'NU'+str(nu)+'mi'
	else:
		NAME = 'NU'+str(nu)+'m'

	txt_exseg1 = """&NAM_CONFZ LMNH_MPI_BSEND = .FALSE.,MPI_BUFFER_SIZE=100, NZ_VERB=5 /
&NAM_CONFIO LCDF4=.TRUE., LLFIOUT=.FALSE., LLFIREAD=.FALSE. / ! Type of file
&NAM_LUNITn  CINIFILE = "CAS09.1.001.002",	! Name of file from which to start the segment
	     CINIFILEPGD="INIT_CANAL_PGD" /      ! Name of PGD
	     
&NAM_DYNn    XTSTEP = 1., 		! Timestep
             CPRESOPT = "ZRESI",	! Pressure solver
             XT4DIFU=100.,		! Diffusion time for momentum
             LHORELAX_UVWTH=.FALSE.,	! Relaxation in case LBC='OPEN'
	     NRIMX=1,			! Number of relaxation points in X
             NRIMY=1,			! Number of relaxation points in Y
             XRIMKMAX=0.01/		! K relaxation coefficient
             
&NAM_ADVn CUVW_ADV_SCHEME = 'CEN4TH', 	! Centered order 4 for U,V,W
	  CTEMP_SCHEME='RKC4', 		! RungeKuta 4 for time
	  	! 4*dx of effective resolution
          CMET_ADV_SCHEME = "PPM_01",	! Piecewise Parabolic Method for THT,TKE,Scalar 
          CSV_ADV_SCHEME="PPM_01"/	! Piecewise Parabolic Method for tracer 
          
&NAM_PARAMn  CTURB = "TKEL",		! 1.5 order closure (TKE and LM)
             CRAD = "NONE", 		! Radiation
             CCLOUD= "ICE3", 		! Cloud 
             CSCONV= "NONE",		! Param. shallow convection
             CDCONV= "NONE" /		! Param. deep convection
             
&NAM_PARAM_ICE LWARM=.TRUE./		! ICE3 namelist              
             
&NAM_SEAFLUXn CSEA_FLUX="COARE3",CSEA_ALB="UNIF"/  ! Surface scheme (see Surfex)

&NAM_LBCn    CLBCX = 2*"CYCL",		! LBC X direction
             CLBCY = 2*"CYCL"/		! LBC Y direction
             
&NAM_TURBn   XIMPL = 1.,		! 1=full implicit, 0=full explicit
             CTURBLEN = "DEAR",		! Turbulent mixing length
             CTURBDIM = "3DIM",		! 3D or 1D
             LRMC01 = T,		! Separate mixing and dissipative mixing lengths
             LTURB_FLX = T,		! Turbulent flux stored in BACKUPs (THW_FLX, ...)
             LTURB_DIAG = T,		! Turbulent diag (TKE_DISS, ...)
             LSIG_CONV = F,		! Compute Sigmas due to subgrid condensation
             LSUBG_COND = F /		! Flag for subgrid condensation
             
&NAM_CONF    CCONF="RESTA",		! RESTA or START
             LFLAT = T,			! flat terrain
             CEQNSYS = "DUR",		! system of equation, Durran
             LFORCING = T,		! use the forcing defined in ZFRC
             NMODEL = 1,		! Number of nested models
             NVERB = 1,			! verbose, 10 is maximum
             CEXP = '"""+NAME+"""',		! name of experiment
             CSEG = "003",		! name of segment
             NHALO=1 /			! halo for parallel computation
             
&NAM_DYN     XSEGLEN = """+seg_length+""",	 	! length of segment
             XASSELIN = 0.2,		! Asselin temporal filter
             LCORIO = T,		! T=Earth's rotation is taken into account
             XALKTOP = 0.005, 		! Top sponge layer coefficient
             XALZBOT = 1800., 		! Altitude of the begining of sponge layer
             LNUMDIFU = T /		! Flag for num. diffusion (for CEN4TH)
 
&NAM_FRC  LTEND_THRV_FRC= F,		! Flag to use THT and RV tendencies
          LVERT_MOTION_FRC= F,		! Flag to use large scale vertical transport
          LRELAX_THRV_FRC=F,		! Flag to relax to ZFRC values of THT and RV
          LGEOST_UV_FRC=T,		! Flag to use ZFRC values of U,V as geo wind
          LRELAX_UV_FRC=F/ 		! Flag to relax to ZFRC values of U,V
         
&NAM_LES LLES_MEAN=.TRUE.,	 !Umoyen
	 LLES_RESOLVED =.TRUE.,  !turbulence résolue
	 LLES_SUBGRID = .TRUE.,  !turbulence sous maille
	 CBL_HEIGHT_DEF = "DTH", !CBL height definition
	 XLES_TEMP_SAMPLING = """+seg_length+"""/
         
&NAM_CONDSAMP 	LCONDSAMP = .TRUE.	! Flag to activate conditional sampling
	NCONDSAMP = 3,		! Number of conditional samplings
	XRADIO = 3*"""+str(nu*60)+""".,	! Period of radioactive decay (15min)
	!XSCAL =3*1.,		! Scaling factor (1)
	XHEIGHT_BASE = 50,	! Height below Cbase where the 2nd tracer is released
	XDEPTH_BASE = 0.,	! Depth on which the 2nd tracer is released
	XHEIGHT_TOP = 50,	! Height above Ctop where the 3rd tracer is released
	XDEPTH_TOP = 50,	! Depth on which the 3rd tracer is released
	NFINDTOP = 1		! see doc
	!XTHVP =0.25,		! if NFINDTOP = 2, threshold to detect ABLH base on thtv
	LTPLUS = .FALSE./	! see doc
	  	
&NAM_BACKUP XBAK_TIME_FREQ(1) = """+seg_length+""", 	! Frequency of output of all variables
            XBAK_TIME_FREQ_FIRST(1)= """+seg_length+""" /	! First time 
&NAM_OUTPUT 				
	COUT_VAR(1,1)='RVT',
	COUT_VAR(1,2)='THT',
	COUT_VAR(1,3)='TKET',
	COUT_VAR(1,4)='UT',
	COUT_VAR(1,5)='VT',
	COUT_VAR(1,6)='WT',
	COUT_VAR(1,7)='PABST',
	XOUT_TIME_FREQ(1)=30.,		! Frequency of output of pronostics variables
					! 	and subgrid flux (custom modification)
	LOUT_BEG=.FALSE.,		! Save at begining of segment
	LOUT_END=.FALSE.,		! Save at end of segment
	COUT_DIR='FICHIERS_OUT',	! Directory to save the files
	XOUT_TIME_FREQ_FIRST(1) = """+seg_length+"""	! Time at which the saving starts
	LOUT_REDUCE_FLOAT_PRECISION(1)=.TRUE./	! Flag to use simple precision
"""
	
	txt_run = """#!/bin/bash
#SBATCH -J NU"""+str(nu)+"""
#SBATCH -N """+str(node)+"""           # nodes number (=NBP)   
#SBATCH -n """+str(core)+"""       # CPUs number (on all nodes) (=NBP*TPN) 
#SBATCH -o NU"""+str(nu)+""".eo%j   #
#SBATCH -e NU"""+str(nu)+""".eo%j   #
#SBATCH -t 01:30:00    # time limit

# Echo des commandes
set -x
ulimit -c 0
ulimit -s unlimited
# Arret du job des la premiere erreur

# Nom de la machine
hostname

ln -sf ../../RefC/CAS09.1.001.002.nc .
ln -sf ../../RefC/CAS09.1.001.002.des .
ln -sf ../../RefC/INIT_CANAL_PGD.nc .
ln -sf ../../RefC/INIT_CANAL_PGD.des .

"""+path_profile+"""

mkdir -p FICHIERS_OUT

set -e
export MPIRUN="Mpirun -np """+str(core)+""" " 
export TIMEOUT="timelimit -t 86400" #259200=3j 86400=1j

rm -f file_for_xtransfer pipe_name

ls -lrt
#time $MPIRUN ~escobar/MPI/iping_ipong_bi_10it_check_network_intel2019_ompi4051 # pour voir si un noeud a un probleme
"""+run_line+"""
#${TIMEOUT} time -v ${MPIRUN} idrmem MESONH${XYZ} PAS EN PRODUCTION !!
mv -f OUTPUT_LISTING1  OUTPUT_LISTING1_mnh
mv -f OUTPUT_LISTING0  OUTPUT_LISTING0_mnh

rm -f REMAP*
ls -lrt"""
	
	# if nu=15min, then its RefC so no need to rerun the simulation
	if nu!=15:
	
		# create directories
		os.system('mkdir -p ../'+str(nu)+'min')
		
		# run_mesonh
		Check_file('../'+str(nu)+'min/run_mesonh')
		run = open('../'+str(nu)+'min/run_mesonh', 'w')
		run.write(txt_run)
		run.close()
		os.system('chmod +x ../'+str(nu)+'min/run_mesonh')
		
		# EXSEG1.nam
		Check_file('../'+str(nu)+'min/EXSEG1.nam')
		exseg = open('../'+str(nu)+'min/EXSEG1.nam', 'w')
		exseg.write(txt_exseg1)
		exseg.close()
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	