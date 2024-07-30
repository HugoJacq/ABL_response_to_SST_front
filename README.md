# Atmospheric Response To An Oceanic Sub-mesoscale SST Front


## 1. Introduction 

  This repo contains scripts, namlists and everything needed to reproduce figures from the paper "Atmosphere Response To An Oceanic Sub-mesoscale SST Front" avalaible [here](DOI to be added).
  Steps to reproduce the simulations are detailed, but you will need a beefy computer ! I used 512 CPUs for the simulations. You can also directly download data from zenodo : [part1](link), [part2](link).
  Figures are produced using Python, Xarray and Matplotlib while the simulations were done with MesoNH.

  If you need help running the scripts or running the simulations, please open an issue and i'll try to have a look.
  
  This is Work In Progress !

## 2. Producing the data
  ### 2.1 Installation of MesoNH
  
  The code used is MesoNH 5.6.1 ([Lafore et al. 2018](https://doi.org/10.5194/gmd-11-1929-2018)) and it can be downloaded from [here](http://mesonh.aero.obs-mip.fr/mesonh56/Download). Look for the row with "MASDEV5-6 BUG1" and click on the tar link.
  Basic installation instructions will be given here but you can also look at the 'A-Install' file inside the archive. It is assumed that you are using a linux machine, with a sourced version of mpi and a fortran compiler.
  
  % Here insert a image with arrows that describe the actions to do.
  
  Once you have downloaded the archive, uncompress it somewhere that will not be erased (typically the $HOME directory, and not the $WORKDIR on supercomputer):
  
  ```
  cd ~
  tar xvfz MNH-V5-6-1.tar.gz
  cd MNH-V5-6-1/src
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
  . ../conf/profile_mesonh-LXgfortran-R8I4-MNH-V5-6-1-MPIAUTO-02
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
  
  In a new command window, move the files from the folder [Insert path of github modif] to the folder `2tracers_AND_SBGinOUT` in the source directory of MesoNH:
  ```
  cd ~
  cd MNH-V5-6-1/src
  mkdir 2tracers_AND_SBGinOUT
  cp 'path/to/the/file' 2tracers_AND_SBGinOUT
  ```
  
  Now you will compile these part of the code. No need to recompile everything. Create a config file that takes into account the user modification with:
  ```
  export VER_USER=2tracers_AND_SBGinOUT
  export VER_MPI=MPIAUTO
  ./configure
  ```
  and then load the new profile
  ```
  . ../conf/profile_mesonh-LXgfortran-R8I4-MNH-V5-6-1-2tracers_AND_SBGinOUT-MPIAUTO-02
  ```
  compile 
  ```
  make -j 8 user
  make installuser
  ```

  MesoNH 5.6.1 is now installed, with the user modification required.

  ### 2.2 Running the simulations
  
  MesoNH uses Namlists to provide inputs to a simulation. The simulations that we want to produce are idealized case and so there are three steps when preparing a simulation: 
  - Initialize the 3D fields with user-provided profiles (PRE_IDEAL.nam)
  - Run a spinup period (EXSEG1_spinup.nam)
  - Run the simulation (EXSEG1_run.nam)
  
  Namlits can be found at [link to namlists]. Three simulations are done : one with the SST front (S1), and two homogeneous simulations (refC and refW).
  For each simulation, there is three folders: one for each step in the simulation. For each step, you will need to adapt:  
  - the workload manager (slurm, ...)
  - the MesoNH profile with the profile you created in the last section
  - the number of CPUs for the 2nd and 3rd step (i used 512)
  
  Then, run the program (e.g. the initialization part) with `./run_prep_ideal` on your PC or something like `sbatch run_prep_ideal` on a supercomputer.
  You now have the data necessary to post process the simulation !

## 3. Plotting the figures
  ### 3.1 Required packages

- Python
- Xarray
- SciPy
- Matplotlib
- Numpy
  
  ### 3.2 What is the plan ?
  Plotting the figures requires some data post processing. In a first time, a few files are built. Then data from those files are used to plot the figures.
  The user cannot plot figures without these files. There are four python script that you can found here [link to files]:
  - `analyse.py` is the main program. This is where you can chose what to plot.
  - `module_cst.py` gather all constants like gravitational acceleration, gas constants, ...
  - `module_phy.py` where all procedures called in `analyse.py` are defined.
  - `module_tool.py` is a toolbox that gather small functions used in `module_phy.py`
  ### 3.3 Plotting
  You will need to modify the path where the data is stored in `analyse.py` in the first lines. By default, the program will only build the postprocess files and will not plot any figures.
  Building those files can be long depending on your machine. Then, let's say you want to plot figure 1, set `FIG1=True`, and if you want to plot all figures, set `FIGALL=True`.
  Figures will be saved in the current directory.

  To start the postprocess/plotting, simply type
  ```
  python analyse.py
  ```
  
## 4. Licences



