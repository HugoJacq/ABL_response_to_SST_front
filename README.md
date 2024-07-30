# ABL_response_to_SST_front


## 1. Introduction 

This repo contains scripts, namlists and everything needed to reproduce figures from the paper "Atmosphere Response To An Oceanic Sub-mesoscale SST Front" avalaible [here](DOI to be added).
Steps to reproduce the simulations are detailed, but you will need a beefy computer ! I used 512 CPUs for the simulations.

This is Work In Progress !

## 2. Producing the data

### 2.1 Installation of MesoNH
The code used is MesoNH 5.6.1 [Lafore et al. 2018](https://doi.org/10.5194/gmd-11-1929-2018) and it can be downloaded from [here](http://mesonh.aero.obs-mip.fr/mesonh56/Download). Look for the row with "MASDEV5-6 BUG1" and click on the tar link.
Basic installation instructions will be given here but you can also look at the 'A-Install' file inside the archive. It is assumed that you are using a linux machine, with a sourced version of mpi and a fortran compiler.

% Here insert a image with arrows that describe the actions to do.

Once you have downloaded the archive, uncompress it somewhere that will not be erased (typically the $HOME directory, and not the $WORKDIR on supercomputer):

`code`

### 2.2 Preparing the simulations

### 2.3 Running the simulations

## 3. Plotting the figures

### 3.1 Required packages

### 3.2 What is the plan ?

### 3.3 Plotting

Plotting the figures requires some data post processing. In a first time, a few files are built. Then data from those files are used to plot the figures.
The user cannot plot figures without these files.


