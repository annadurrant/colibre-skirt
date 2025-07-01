"""
Run a set of SKIRT simulations for given halo indices.
Created by Andrea Gebek on 29.11.2024
"""

import numpy as np
import subprocess
from multiprocessing import Pool
import yaml
import argparse
import os

parser = argparse.ArgumentParser(
    description="Run a set of SKIRT simulations for given halo indices."
)

# Set simName if needed for output files
parser = argparse.ArgumentParser(
    description="Select COLIBRE halos and store some global information."
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
    "--snaps",
    type=int,
    nargs='+',
    help="<Required> Snapshot number(s).",
)

parser.add_argument(
        "--ID",
        type=int,
        default=None,
        help="HBT track ID to run SKIRT simulations for (default: None, uses sample.txt file for IDs).",
)

parser.add_argument(
        "--n_mpi",
        type=int,
        default=3,
        help="Number of SKIRT simulations you want to run in parallel (default: 3)",
)

parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="Number of threads to run each SKIRT simulation on (default: 4).",
)

parser.add_argument(
    "--vIMF",
    action="store_true",
    help="Running in vIMF mode (default: false).",
)

args = parser.parse_args()

# Define filepaths from parameter file
dir_path =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
param_file = 'SKIRT_parameters.yml'
if args.vIMF == True:
    param_file = 'vimf_' + param_file
with open(f'{dir_path}/{param_file}','r') as stream:
    params = yaml.safe_load(stream)


simPath = params['InputFilepaths']['simPath'].format(simName=args.simName)
sampleFolder = params['OutputFilepaths']['sampleFolder'].format(simPath=simPath,rotation=params['ModelParameters']['rotation'])
txtFilePath = params['OutputFilepaths']['storeParticlesPath'].format(simPath=simPath,rotation=params['ModelParameters']['rotation']) # Path to the COLIBRE particle .txt files
SKIRTinputFilePath = params['OutputFilepaths']['SKIRTinputFilePath'].format(simPath=simPath,rotation=params['ModelParameters']['rotation']) # Path where the SKIRT input files will be stored
SKIRToutputFilePath = params['OutputFilepaths']['SKIRToutputFilePath'].format(simPath=simPath,rotation=params['ModelParameters']['rotation']) # Path where the SKIRT output files will be stored

# Make output directories 
os.system(f'mkdir -p {SKIRTinputFilePath}')
os.system(f'mkdir -p {os.path.dirname(SKIRToutputFilePath)}')
os.system(f'mkdir -p {SKIRToutputFilePath}')

SKIRToutputFilePath += args.outputDir
os.system(f'mkdir -p {SKIRToutputFilePath}')

# Set list of snapshots to postprocess

Nprocesses = args.n_mpi

def preprocess(snapList):
    # Generate a list of SKIRT simulation names and run the necessary preprocessing steps

    skifilenames = []

    for snap in snapList:

        halo_IDs, Rstar = np.loadtxt(sampleFolder + '/sample_' + str(snap) + '.txt', unpack = True, usecols = [0, 2])
        halo_IDs = halo_IDs.astype(int)

        for idx, ID in enumerate(halo_IDs):

            if args.ID != None and ID != args.ID:
                continue

            skifilenames.append( 'snap' + str(snap) + '_ID' + str(ID) )

            # Save SKIRT input files
            subprocess.run(['python', f'{dir_path}/pythonScripts/saveSKIRTinput.py', str(snap), str(ID), txtFilePath, SKIRTinputFilePath, str(args.vIMF)])

            # Edit ski files

            subprocess.run(['python', f'{dir_path}/pythonScripts/editSkiFile.py', str(snap), str(ID), str(Rstar[idx]), txtFilePath, SKIRTinputFilePath, simPath])

    return skifilenames



def runSKIRT(skifilename):

    # Run skirt

    subprocess.run(['skirt', '-t', str(args.n_threads), '-b', skifilename]) # Run SKIRT with 4 threads (that's apparently quite optimal)
    # The -b option reduces the verbosity of the log (but the saved log file still contains all logging information)

    return skifilename

def postprocess(snapList):

    # Get the SKIRT output files and move them to the output folder

    for snap in snapList:

        halo_IDs = np.loadtxt(sampleFolder + '/sample_' + str(snap) + '.txt', unpack = True, usecols = 0).astype(int)

        for idx, ID in enumerate(halo_IDs):

            if args.ID != None and ID != args.ID:
                continue

            sim_name = 'snap' + str(snap) + '_ID' + str(ID)
            
            os.system(f'mv {sim_name}* {SKIRToutputFilePath}/')

def main():

    skifilenames = preprocess(args.snaps)

    with Pool(processes = Nprocesses) as pool:
        
        pool.map(runSKIRT, skifilenames)

    # postprocess(args.snaps)

if __name__=="__main__":

    main()