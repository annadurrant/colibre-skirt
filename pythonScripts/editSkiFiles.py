import numpy as np
import subprocess
from multiprocessing import Pool
import yaml
import argparse
from datetime import datetime
import os
import shutil
import pandas as pd

skifileversion = "5.0"

parser = argparse.ArgumentParser(
    description="Prepare SKIRT simulation run files from sample list."
)

parser.add_argument(
    "simName",
    type=str,
    help="Simulation name.",
)

parser.add_argument(
    "outputDir",
    type=str,
    help="Name of output directory.",
)

parser.add_argument(
    "snap",
    type=int,
    help="Snapshot number.",
)

parser.add_argument(
    "--vIMF",
    action="store_true",
    help="Running in vIMF mode (default: false).",
)

parser.add_argument(
    "--distr",
    type=int,
    default=-1,
)


args = parser.parse_args()

#############################################
#### Define filepaths from parameter file ###
#############################################
dir_path =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
param_file = "SKIRT_parameters.yml"
if args.vIMF == True:
    param_file = "vimf_" + param_file
with open(f"{dir_path}/{param_file}","r") as stream:
    params = yaml.safe_load(stream)


simPath = params["InputFilepaths"]["simPath"].format(simName=args.simName)
sampleFolder = params["OutputFilepaths"]["sampleFolder"].format(simPath=simPath,rotation=params["ModelParameters"]["rotation"])
txtFilePath = params["OutputFilepaths"]["storeParticlesPath"].format(simPath=simPath,rotation=params["ModelParameters"]["rotation"]) # Path to the COLIBRE particle .txt files
SKIRTinputFilePath = params["OutputFilepaths"]["SKIRTinputFilePath"].format(simPath=simPath,rotation=params["ModelParameters"]["rotation"]) # Path where the SKIRT input files will be stored
SKIRToutputFilePath = params["OutputFilepaths"]["SKIRToutputFilePath"].format(simPath=simPath,rotation=params["ModelParameters"]["rotation"]) # Path where the SKIRT output files will be stored
skifilename = params["InputFilepaths"]["skiFilepath"].format(skifileversion=skifileversion)

snap = int(args.snap)
redshift_list = pd.read_csv(f"{simPath}/output_list.txt").to_numpy()[:,0]
redshift = redshift_list[int(snap)]

#########################################
######### Make output directories #######
#########################################
os.system(f"mkdir -p {os.path.dirname(SKIRToutputFilePath)}")
os.system(f"mkdir -p {SKIRToutputFilePath}")

SKIRToutputFilePath += args.outputDir
os.system(f"mkdir -p {SKIRToutputFilePath}")

#########################################
######### Set SKIRT parameters ##########
#########################################

old_stars_tmin = float(params["ModelParameters"]["starsMaxAge"]) # Minimum age in Myr for an evolved star particle. Also determines the TODDLERS averaging timescale

Npp = int(float(params["ModelParameters"]["photonPackets"])) # Number of photon packets
binTreeMaxLevel = params["ModelParameters"]["binTreeMaxLevel"] # Max refinement level of the spatial grid

scaleFactor = 1. / (1. + redshift) # Scale factor for the snapshot
SKIRTboxsize0 = float(params["ModelParameters"]["SKIRTboxsize"])
SKIRTboxsize = min(SKIRTboxsize0, SKIRTboxsize0 * 1.8 / 0.7 * scaleFactor) # Scale SKIRT box size akin to COLIBRE gravitational softening length, in kpc 

startTime = datetime.now()



#########################################
######### Read in galaxy sample #########
#########################################

# Sample list filepath, either total list or part of distributed list
if args.distr != -1:
    sampleFile = sampleFolder + "/sample_" + str(snap) + "/sample_" + str(snap) + "." + str(args.distr) + ".txt"
else:
    sampleFile = sampleFolder + "/sample_" + str(snap) + ".txt"

halo_IDs, Rstars, Mdusts, Rdusts = np.loadtxt(sampleFile, unpack = True, usecols = [0, 2, 3, 4])
SigmaDusts = Mdusts / (2 * np.pi * Rdusts**2) # Dust surface density

halo_IDs = halo_IDs.astype(int)


#########################################
######### Edit file per galaxy ##########
#########################################

for idx, haloID in enumerate(halo_IDs):

    SKIRTinputFiles = SKIRTinputFilePath + f"snap{snap}_ID{haloID}"
    skifilename_halo = f"snap{snap}_ID{haloID}.ski"

    if os.path.isfile(skifilename_halo):
        # if .ski file already exists, do not replace
        continue
    
    else:

        Rstar, SigmaDust = Rstars[idx], SigmaDusts[idx] # in kpc, Msun/kpc**2
        
        maxDustFraction = np.clip(
            10**(-0.5 - np.log10(SigmaDust)),
            a_min=10**(-6.5),
            a_max=10**(-4.5)
        )

        # copy file
        shutil.copyfile(skifilename, skifilename_halo)

        # open and edit .ski file
        with open(skifilename_halo, "r") as f:
            text = f.read()   

        replacements = {
            'maxLevel="0': f'maxLevel="{binTreeMaxLevel}',
            'dust.txt': f'{SKIRTinputFiles}_dust.txt',
            'minX="-0': f'minX="{-SKIRTboxsize / 2. * 1000.}', # in pc
            'maxX="0': f'maxX="{SKIRTboxsize / 2. * 1000.}',
            'minY="-0': f'minY="{-SKIRTboxsize / 2. * 1000.}',
            'maxY="0': f'maxY="{SKIRTboxsize / 2. * 1000.}',
            'minZ="-0': f'minZ="{-SKIRTboxsize / 2. * 1000.}',
            'maxZ="0': f'maxZ="{SKIRTboxsize / 2. * 1000.}',
            'maxDustFraction="0': f'maxDustFraction="{maxDustFraction}',
            'numPackets="0': f'numPackets="{Npp}',
            'old_stars': f'{SKIRTinputFiles}_old_stars',
            'starforming_gas': f'{SKIRTinputFiles}_starforming_gas',
            'Period0': f'Period{int(old_stars_tmin)}',
            'radius="1 Rstar': f'radius="{Rstar} kpc',
            'radius="3 Rstar': f'radius="{3.0 * Rstar} kpc',
            'radius="5 Rstar': f'radius="{5.0 * Rstar} kpc',
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        with open(skifilename_halo, "w") as f:
            f.write(text)
            
print(f"Elapsed time to generate ski files for snap {snap}:", datetime.now() - startTime)