# Atmospheric Response to an Oceanic Submesoscale SST Front : a Coherent Structure Analysis

![alt text](https://github.com/HugoJacq/ABL_response_to_SST_front/blob/main/S1_Ly8km_LastTime_UVW_XY_lanes.png?raw=true)

**Figure 2.** Snapshot at t=+3h of (a) zonal and (b) meridional wind at first model level and (c) vertical velocity in the middle of the ABL. The colored line at the bottom of the panels indicates the SST with blue representing cold SST (296.55 K) and red warm SST (298.05 K). The cold-to-warm and warm-to-cold fronts are located at X = 5 km and X = 25 km respectively.

[1. Introduction ](#1-introduction)

[2. Producing the data](#2-producing-the-data)

[2.1 Installation of MesoNH](#21-installation-of-mesonh)

[2.2 Running the simulations](#22-running-the-simulations)

[3. Plotting the figures](#3-plotting-the-figures)

[Licence and contact](#licence-and-contact)

## 1. Introduction 

This repo contains scripts, namlists and everything needed to reproduce figures from the paper "Atmosphere Response To An Oceanic Sub-mesoscale SST Front" avalaible [here](DOI to be added).
Steps to reproduce the simulations are detailed, but you will need a strong computer ! I used 512 CPUs for the simulations. You can also directly download data from zenodo : [part1](https://zenodo.org/records/13321540), [part2](https://zenodo.org/records/13321834). Scripts and Namlists are also [archived](https://zenodo.org/records/13341467).
Figures are produced using Python, Xarray and Matplotlib while the simulations were done with MesoNH.

If you need help running the scripts or running the simulations, please open an issue and i'll try to have a look.

The plan is :
- Produce the data (install MesoNH, run the simulations)
- Plot the figures (environment, running the scripts)

Diskspace used:
- about 105Go for the simulations (compressed)
- 200Go for the post process

For your information, producing the simulation data emitted 115 kg of equivalent carbon (including tests cases).

(The paper has been submitted to JGR: Atmosphere and is under peer review.)

## 2. Producing the data
![alt text](http://mesonh.aero.obs-mip.fr/mesonh57/Welcome?action=AttachFile&do=get&target=LogoMesoNH.jpg)

### 2.1 Installation of MesoNH

The code used is MesoNH 5.6.2 ([Lafore et al. 2018](https://doi.org/10.5194/gmd-11-1929-2018)) and it can be downloaded from [here](http://mesonh.aero.obs-mip.fr/mesonh56/Download). Look for the row with "MASDEV5-6 BUG2" and click on the tar link.
Basic installation instructions will be given here but you can also look at the 'A-Install' file inside the archive. It is assumed that you are using a linux machine, with a sourced version of mpi and a fortran compiler.

Once you have downloaded the archive, uncompress it somewhere that will not be erased (typically the $HOME directory, and not the $WORKDIR on supercomputer):

```
cd ~
tar xvfz MNH-V5-6-2.tar.gz
cd MNH-V5-6-2/src
```

If you use Debian or Ubuntu, you should already have gfortran. If you do not have any mpi architecture, a tutorial is available [here](http://mesonh.aero.obs-mip.fr/mesonh56/MesonhTEAMFAQ/PC_Linux) to install one.
Then, set the following variable to be able to run the code in parallel:
```
export VER_MPI=MPIAUTO
```
Before compiling, run:
```
./configure
```
and then load the file that has just been created, something like:
```
. ../conf/profile_mesonh-LXgfortran-R8I4-MNH-V5-6-2-MPIAUTO-02
```
Finally, launch the compilation (on 8 cpus for example) with
```
make -j 8
```
And link all the executable with
```
make installmaster
```

Ok fine, you now have a working installation of MesoNH. The results from the paper also use a user made modification that do two things:
- output subgrid flux
- emit two tracers at the surface depending on the SST

In a new command window, move the files from [here](https://github.com/HugoJacq/ABL_response_to_SST_front/tree/main/MNH_user_modif_code) to the folder `2tracers_AND_SBGinOUT` in the source directory of MesoNH:
```
cd ~
cd MNH-V5-6-2/src
mkdir 2tracers_AND_SBGinOUT
cp 'path/to/the/file' 2tracers_AND_SBGinOUT
```

Now you will compile these part of the code. No need to recompile everything. Create a config file that takes into account the user modification with:
```
export VER_USER=2tracers_AND_SBGinOUT
export VER_MPI=MPIAUTO
./configure
```
and then load the new profile (in a new tab)
```
. ../conf/profile_mesonh-LXgfortran-R8I4-MNH-V5-6-2-2tracers_AND_SBGinOUT-MPIAUTO-02
```
compile 
```
make -j 8 user
make installuser
```

MesoNH 5.6.1 is now installed, with the user modification required.

### 2.2 Running the simulations

**Workflow**

MesoNH uses Namlists to provide inputs to a simulation. The simulations that we want to produce are idealized case and so there are three steps when preparing a simulation: 
- Initialize the 3D fields with user-provided profiles (PRE_IDEAL.nam)
- Change the SST for S1 (replace_sst.py)
- Run a spinup period (EXSEG1.nam.spinup)
- Run the simulation (EXSEG1.nam.run)

Namlits can be found [here](https://github.com/HugoJacq/ABL_response_to_SST_front/tree/main/Namlists). Three simulations are done : one with the SST front (S1), and two homogeneous simulations (refC and refW).
. For each step, you will need to adapt:  
- the workload manager (slurm, ...)
- the MesoNH profile with the profile you created in the last section
- the number of CPUs for the 3rd and 4th step (I used 512)

Note on the hardware: I used 512 CPU to run the simulations, producing roughly 100 Go (77 Go for S1, 4 Go for RefC, 4 Go for RefW and 6 Go for the nu sensitivity study).

**Launching the simulations**

First make sure you have the proper folder structure with namlists:
```
S1/
│   PRE_IDEA1.nam
│   EXSEG1.nam.spinup
│   EXSEG1.nam.run
|   replaceSST.py
|   run_mesonh
|   run_prep_ideal
|   run_spinup
RefW/
│   PRE_IDEA1.nam
|   EXSEG1.nam
|   run_mesonh
|   run_prep_ideal
RefC/
│   PRE_IDEA1.nam
|   EXSEG1.nam
|   run_mesonh
|   run_prep_ideal
radio_decay_sensitivity/
└───Namlist_injector/
    |   setup.py
```

Then, run the program (e.g. the initialization part) with `./run_prep_ideal` on your PC or something like `sbatch run_prep_ideal` on a supercomputer.
I advise to run each step at once.

For S1:
```
./run_prep_ideal
python replace_sst.py
./run_spinup
./run_mesonh
```

For RefC and RefW:
```
./run_prep_ideal
./run_mesonh
```
For radioactive decay (nu) sensitivity analysis: \
In the folder `radio_decay_sensitivity`, you will find another folder called `Namlist_injector`. Inside, a python script will create the namlists to run the sensitivity analysis.
Before running the commands below, modify in `setup.py` your mesonh profile (lines 18->27) and if needed the header of the string `txt_run` according to your HPC format (line 148).
Then, simply type: 
```
cd radio_decay_sensitivity/Namlist_injector
python setup.py
```
You will now have one folder for each radioactive decay in the parent folder. The simulations are re-doing the last hour of RefC, changing only the radioactive decay time. To run the simulations for each radioactive decay time,
go the each directory and launch the executable. For example, for $\nu$ = 7min, run
```
cd ../7min
./run_mesonh
```
or on a supercomputer
```
cd ../7min
sbatch run_mesonh
```

**Outputs description**

For each simulations, you have in the main folder:
- Initial 3D fields `INIT_CANAL_SST.nc` and surface fields `INIT_CANAL_PGD.nc`
- BACKUP files `CAS06.1.001*` (spinup) or `CAS06.1.002*` (main run)
- Logs of the simulations in `OUTPUT_LISTING1_ideal` and `OUTPUT_LISTING1_mnh`
In the folder `FICHIERS_OUT`,
- High frequency outputs, `CAS06.1.001*` for spinup and `CAS06.1.002*` for main run

For reference simulations, you will have an added file `CAS06.1.001.000.nc` (diachronic file) which contains output from online LES computations (mean, fluxes, ...)

Here is the tree of the output of data, with the files needed for the post-process step:
```
S1/
│   CAS06.1.002.002.nc
│   INIT_CANAL_SST.nc    
└───FICHIERS_OUT/
    │   CAS06.1.002.OUT.001.nc
    │   [...]
    │   CAS06.1.002.OUT.121.nc
RefW/
│   CAS10.1.001.000.nc
|   CAS10.1.001.003.nc
└───FICHIERS_OUT/
    │   CAS10.1.001.OUT.001.nc 
RefC/
│   CAS09.1.001.000.nc
|   CAS09.1.001.003.nc
└───FICHIERS_OUT/
    │   CAS09.1.001.OUT.001.nc
radio_decay_sensitivity/
└───Namlist_injector/
    |   setup.py
└───1min/
└───4min/
|   [...]
└───40min/
    └───FICHIERS_OUT/
        |   NU40m.1.003.OUT.001.nc
```
Finally, ouputs have been compressed to reduce diskspace with the command:
```
ncks -O -7 -L 1 --ppc default=4#PABST=7#RVT=5#THT=6 oldfile newfile
```
And packed with the command (for example for S1):
```
tar -hzcvf S1.tar.gz S1/
```

You now have the data necessary to post process the simulation !

## 3. Plotting the figures
  ### 3.1 Required packages

Packages necessary to run the post process (the version indicated is the one I used) :

- Python  (3.10.10)
- Xarray (with dask) (2024.10.0)
- SciPy (1.14.1)
- Matplotlib (3.9.2)
- Numpy (2.1.3)
- cmocean (4.0.3)
  
### 3.2 What is the plan ?
Plotting the figures requires some data post processing. In a first time, a few files are built. Then data from those files are used to plot the figures.
The user cannot plot figures without these files. There are height python script that you can found [here](https://github.com/HugoJacq/ABL_response_to_SST_front/tree/main/Post_process):
- `analyse.py` is the main program. This is where you can chose what to plot.
- `module_building_files.py` were function to produce files for the post process
- `module_cst.py` gather all constants like gravitational acceleration, gas constants, ...
- `module_phy.py` where all procedures called in `analyse.py` are defined.
- `module_tool.py` is a toolbox that gather small functions used in `module_phy.py`
- `module_CS.py` is a file where you can find functions related to the conditional sampling
- `module_CS_func.py` are small functions used in `module_CS.py`
- `module_budget.py` where budgets are computed
### 3.3 Plotting
You will need to modify the path where the data is stored in `analyse.py` in the first lines. By default, the program will only build the postprocess files and will not plot any figures.
Building those files can be long depending on your machine. Then, let's say you want to plot figure 1, set `NUM_FIG=1`, and if you want to plot all figures, set `NUM_FIG=-1`.
Figures will be saved in the current directory.

To start the postprocess/plotting, simply type
```
python analyse.py
```

**Notes**

Due to the intrinsec chaotic nature of turbulence, the figure 2 and 7 cannot be reproduced exactly: the small differences of computing architecture introduce slightly different boundary and initial conditions and so the flow instantaneous structure is different.
Also, please be aware that the figure with the snapshot of two plumes AND the sensibility tests of tau are done with the simulation of width 2km (setting NJMAX=40 instead of NJMAX=160, every other namlist entry is the same.).

## Licence and contact

If you reuse this work, please cite it ! This repository is under MIT license, Zenodo archives are under Creative Commons Attribution 4.0 International license.

(here citation paper when accepted)

Jacquet, H. (2024). Output of simulations for "Atmosphere Response to an Oceanic Sub-mesoscale SST Front: A Coherent Structure Analysis": Part 1 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13321540

Jacquet, H. (2024). Output of simulations for "Atmosphere Response to an Oceanic Sub-mesoscale SST Front: A Coherent Structure Analysis": Part 2 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13321834

Jacquet, H. (2024). Scripts and namlists for "Atmosphere Response to an Oceanic Sub-mesoscale SST Front: A Coherent Structure Analysis". Zenodo. https://doi.org/10.5281/zenodo.13341467

If you have a problem with the code, please open an issue. If you want to discuss science, you can contact me at hugo.jacquet.work[at]gmail.com
