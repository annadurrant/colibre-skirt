"""
Edit a ski file
"""

import numpy as np
import subprocess
import sys
import warnings
from datetime import datetime
import unyt
import os
import yaml

startTime = datetime.now()

# Global settings

snapNum = sys.argv[1]
haloID = sys.argv[2]
Rstar = unyt.unyt_quantity(float(sys.argv[3]), 'kpc')
SigmaDust = unyt.unyt_quantity(float(sys.argv[4]), 'Msun/kpc**2')

txtFilePath = sys.argv[5]
SKIRTinputFilePath = sys.argv[6]
simPath = sys.argv[7]
vIMF = sys.argv[8] == 'True' #whether to run in vIMF mode


# Define filepaths from parameter file

dir_path =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
param_file = 'SKIRT_parameters.yml'
if vIMF == True:
    param_file = 'vimf_' + param_file
with open(f'{dir_path}/{param_file}','r') as stream:
    params = yaml.safe_load(stream)


# Set parameters

old_stars_tmin = unyt.unyt_quantity(params['ModelParameters']['starsMaxAge'], 'Myr') # Minimum age in Myr for an evolved star particle. Also determines the TODDLERS averaging timescale

Npp_per_par = int(float(params['ModelParameters']['photonPackets'])) # Number of photon packets per star particle
binTreeMaxLevel = params['ModelParameters']['binTreeMaxLevel'] # Max refinement level of the spatial grid

f = open(txtFilePath + 'snap' + snapNum + '_' + 'ID' + haloID + '_stars.txt', 'r')
header = f.readline() # Read first header line
redshift = float(header.split(' ')[-1])
f.close()

scaleFactor = 1. / (1. + redshift) # Scale factor for the snapshot
SKIRTboxsize0 = float(params['ModelParameters']['SKIRTboxsize'])
SKIRTboxsize = unyt.unyt_quantity(min(SKIRTboxsize0, SKIRTboxsize0 * 1.8 / 0.7 * scaleFactor), 'kpc') # Scale SKIRT box size akin to COLIBRE gravitational softening length

skifileversion = '5.0'


# Edit ski file

def editSki(snapNum, haloID, Rstar, SigmaDust):

    SKIRTinputFiles = SKIRTinputFilePath + 'snap' + snapNum + '_ID' + haloID
    skifilename = params['InputFilepaths']['skiFilepath'].format(skifileversion=skifileversion)
    skifilename_halo = 'snap' + snapNum + '_ID' + haloID + '.ski'

    if os.path.isfile(skifilename_halo):
        subprocess.run(['mv', skifilename_halo, 'delete.me'])

    subprocess.run(['cp', skifilename, skifilename_halo]) # copy the skirt file for each galaxy

    subprocess.run(['perl', '-pi', '-e', 's/maxLevel=\"0/maxLevel=\"' + str(binTreeMaxLevel) + '/g', skifilename_halo])

    subprocess.run(['perl', '-pi', '-e', 's#dust.txt#' + SKIRTinputFiles + '_dust.txt#g', skifilename_halo])

    subprocess.run(['perl', '-pi', '-e', 's/minX=\"-0/minX=\"' + str(-SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
    subprocess.run(['perl', '-pi', '-e', 's/maxX=\"0/maxX=\"' + str(SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
    subprocess.run(['perl', '-pi', '-e', 's/minY=\"-0/minY=\"' + str(-SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
    subprocess.run(['perl', '-pi', '-e', 's/maxY=\"0/maxY=\"' + str(SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
    subprocess.run(['perl', '-pi', '-e', 's/minZ=\"-0/minZ=\"' + str(-SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
    subprocess.run(['perl', '-pi', '-e', 's/maxZ=\"0/maxZ=\"' + str(SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])

    maxDustFraction = np.clip(10**(-0.5 - np.log10(SigmaDust)), a_min = 10**(-6.5), a_max = 10**(-4.5))
    
    subprocess.run(['perl', '-pi', '-e', 's/maxDustFraction=\"0/maxDustFraction=\"' + str(maxDustFraction) + '/g', skifilename_halo])

    Npp = Npp_per_par * len(np.loadtxt(txtFilePath + 'snap' + snapNum + '_' + 'ID' + haloID + '_stars.txt'))
    subprocess.run(['perl', '-pi', '-e', 's/numPackets=\"0/numPackets=\"' + str(Npp) + '/g', skifilename_halo])

    subprocess.run(['perl', '-pi', '-e', 's#old_stars#' + SKIRTinputFiles + '_old_stars#g', skifilename_halo])
    subprocess.run(['perl', '-pi', '-e', 's#starforming_gas#' + SKIRTinputFiles + '_starforming_gas#g', skifilename_halo])
    subprocess.run(['perl', '-pi', '-e', 's/Period0/Period' + str(int(old_stars_tmin.to('Myr').value)) + '/g', skifilename_halo])

    subprocess.run(['perl', '-pi', '-e', 's/radius=\"1 Rstar/radius=\"' + str(Rstar.to('kpc').value) + ' kpc' +  '/g', skifilename_halo])
    subprocess.run(['perl', '-pi', '-e', 's/radius=\"3 Rstar/radius=\"' + str(3. * Rstar.to('kpc').value) + ' kpc' +  '/g', skifilename_halo])
    subprocess.run(['perl', '-pi', '-e', 's/radius=\"5 Rstar/radius=\"' + str(5. * Rstar.to('kpc').value) + ' kpc' +  '/g', skifilename_halo])

    return None

editSki(snapNum, haloID, Rstar, SigmaDust)

# print('Elapsed time to edit ski file and calculate dust surface density:', datetime.now() - startTime)