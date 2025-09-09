import numpy as np
import subprocess
from multiprocessing import Pool
import yaml
import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser(
    description="Run a set of SKIRT simulations for given halo indices."
)

# Set simName if needed for output files
parser = argparse.ArgumentParser(
    description="Collect SKIRT run files."
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

for snap in args.snaps:

    startTime = datetime.now()

    if args.distr != -1:
        sampleFile = sampleFolder + '/sample_' + str(snap) + '/sample_' + str(snap) + '.' + str(args.distr) + '.txt'
        halo_IDs, Rstar = np.loadtxt(sampleFile, unpack = True, usecols = [0, 2])
    else:
        halo_IDs,Rstar = np.loadtxt(sampleFolder + '/sample_' + str(snap) + '.txt', unpack = True, usecols = [0, 2])

    halo_IDs = halo_IDs.astype(int)

    for idx, ID in enumerate(halo_IDs):

        if args.ID != None and ID != args.ID:
            continue

        if idx%100 == 0:
            print(f'Running {idx}th halo in subset.',flush=True)

        skifilename = 'snap' + str(snap) + '_ID' + str(ID)

        if os.path.isfile(skifilename + '.ski'):
            continue
        
        else:

            # Save SKIRT input files
            subprocess.run(['python', f'{dir_path}/pythonScripts/saveSKIRTinput.py', str(snap), str(ID), txtFilePath, SKIRTinputFilePath, str(args.vIMF)])

            # Edit ski files

            subprocess.run(['python', f'{dir_path}/pythonScripts/editSkiFile.py', str(snap), str(ID), str(Rstar[idx]), txtFilePath, SKIRTinputFilePath, simPath, str(args.vIMF)])

    print(f'Elapsed time to save SKIRT input files and .ski files for snap {snap}:', datetime.now() - startTime)