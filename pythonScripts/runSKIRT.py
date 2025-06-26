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
parser.add_argument(
    "BoxSize",
    type=int,
    help="Boxsize of the simulation in Mpc.",
)

parser.add_argument(
    "Resolution",
    type=int,
    help="Particle mass resolution of the simulation in log10(M/Msun).",
)

parser.add_argument(
    "--snaps",
    type=int,
    required=True,
    nargs='+',
    help="<Required> Snapshot number(s).",
)

parser.add_argument(
    "--mode",
    type=str,
    default="Thermal", # Thermal AGN feedback with non-equilibrium chemistry
    help="Simulation mode (default: Thermal).",
)


parser.add_argument(
        "--nproc",
        type=int,
        default=3,
        help="Number of SKIRT simulations you want to run in parallel. Note that each SKIRT simulation runs with 4 threads by default.",
)

args = parser.parse_args()

sim = 'L{:03.0f}_m{:01.0f}'.format(args.BoxSize, args.Resolution)
simName = sim + '/' + args.mode

args = parser.parse_args()

# Define filepaths from parameter file
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(f'{dir_path}/../SKIRT_parameters.yml','r') as stream:
    params = yaml.safe_load(stream)

sim = 'L{:03.0f}_m{:01.0f}'.format(args.BoxSize, args.Resolution)

sampleFolder = params['SkirtFilepaths']['sampleFolder'].format(sim = sim) # Folder to the galaxy sample files
txtFilePath = params['SkirtFilepaths']['storeParticlesPath'].format(sim = sim) # Path to the COLIBRE particle .txt files
SKIRTinputFilePath = params['SkirtFilepaths']['SKIRTinputFilePath'].format(sim = sim) # Path where the SKIRT input files will be stored
SKIRToutputFilePath = params['SkirtFilepaths']['SKIRToutputFilePath'].format(sim = sim) # Path where the SKIRT output files will be stored


# Set list of snapshots to postprocess

Nprocesses = args.nproc

def preprocess(snapList):
    # Generate a list of SKIRT simulation names and run the necessary preprocessing steps

    skifilenames = []

    for snap in snapList:

        halo_IDs, Rstar = np.loadtxt(sampleFolder + '/sample_' + str(snap) + '.txt', unpack = True, usecols = [0, 2])
        halo_IDs = halo_IDs.astype(int)

        for idx, ID in enumerate(halo_IDs):

            skifilenames.append( 'snap' + str(snap) + '_ID' + str(ID) )

            # Save SKIRT input files

            subprocess.run(['python', f'{dir_path}/saveSKIRTinput.py', str(snap), str(ID), txtFilePath, SKIRTinputFilePath])

            # Edit ski files

            subprocess.run(['python', f'{dir_path}/editSkiFile.py', str(snap), str(ID), str(Rstar[idx]), txtFilePath, SKIRTinputFilePath])

    return skifilenames



def runSKIRT(skifilename):

    # Run skirt

    subprocess.run(['skirt', '-t', '4', '-b', skifilename]) # Run SKIRT with 4 threads (that's apparently quite optimal)
    # The -b option reduces the verbosity of the log (but the saved log file still contains all logging information)

    return skifilename

def postprocess(snapList):

    # Get the SKIRT output files and move them to the output folder

    for snap in snapList:

        halo_IDs = np.loadtxt(sampleFolder + '/sample_' + str(snap) + '.txt', unpack = True, usecols = 0).astype(int)

        for ID in halo_IDs:

            sim_name = 'snap' + str(snap) + '_ID' + str(ID)

            subprocess.run(['rm', sim_name + '.ski']) # Remove the SKIRT input file

            subprocess.run(['mv', sim_name + '_parameters.xml', SKIRToutputFilePath + sim_name + '_parameters.xml'])
            subprocess.run(['mv', sim_name + '_log.txt', SKIRToutputFilePath + sim_name + '_log.txt'])

            if os.path.isfile(sim_name + '_conv_convergence.dat'): # Check if the file exists (it only does if there is a SKIRT medium)
                subprocess.run(['mv', sim_name + '_conv_convergence.dat', SKIRToutputFilePath + sim_name + '_conv_convergence.dat'])
                
            subprocess.run(['mv', sim_name + '_lum_luminosities.dat', SKIRToutputFilePath + sim_name + '_lum_luminosities.dat'])
            subprocess.run(['mv', sim_name + '_SED_tot_sed.dat', SKIRToutputFilePath + sim_name + '_SED_tot.dat'])
            subprocess.run(['mv', sim_name + '_SED_10kpc_sed.dat', SKIRToutputFilePath + sim_name + '_SED_10kpc.dat'])
            subprocess.run(['mv', sim_name + '_SED_30kpc_sed.dat', SKIRToutputFilePath + sim_name + '_SED_30kpc.dat'])
            subprocess.run(['mv', sim_name + '_SED_50kpc_sed.dat', SKIRToutputFilePath + sim_name + '_SED_50kpc.dat'])
            subprocess.run(['mv', sim_name + '_SED_1Rstar_sed.dat', SKIRToutputFilePath + sim_name + '_SED_1Rstar.dat'])
            subprocess.run(['mv', sim_name + '_SED_3Rstar_sed.dat', SKIRToutputFilePath + sim_name + '_SED_3Rstar.dat'])
            subprocess.run(['mv', sim_name + '_SED_5Rstar_sed.dat', SKIRToutputFilePath + sim_name + '_SED_5Rstar.dat'])
    


def main():

    skifilenames = preprocess(args.snaps)

    with Pool(processes = Nprocesses) as pool:
        
        pool.map(runSKIRT, skifilenames)

    postprocess(args.snaps)

if __name__=="__main__":

    main()