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
Rstar = float(sys.argv[3])

txtFilePath = sys.argv[4]
SKIRTinputFilePath = sys.argv[5]

vIMF = sys.argv[7] == 'True' #whether to run in vIMF mode

# Define filepaths from parameter file
dir_path =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
param_file = 'SKIRT_parameters.yml'
if vIMF == True:
    param_file = 'vimf_' + param_file
with open(f'{dir_path}/{param_file}','r') as stream:
    params = yaml.safe_load(stream)

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

def editSki(snapNum, haloID, Rstar):

    SKIRTinputFiles = SKIRTinputFilePath + 'snap' + snapNum + '_ID' + haloID

    skifilename = params['InputFilepaths']['skiFilepath'].format(skifileversion=skifileversion)

    skifilename_halo = 'snap' + snapNum + '_ID' + haloID + '.ski'

    if os.path.isfile(skifilename_halo):
        subprocess.run(['mv', skifilename_halo, 'delete.me'])


    # Calculate max dust fraction based on particle data

    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Ignore warning if file is empty
        gas_file = np.loadtxt(txtFilePath + 'snap' + snapNum + '_ID' + haloID + '_gas.txt') # Calculate dust surface density from the 
        # original gas particle data, to avoid issues with negative dust masses due to TODDLERS dust subtraction

        Npp = Npp_per_par * len(np.loadtxt(txtFilePath + 'snap' + snapNum + '_' + 'ID' + haloID + '_gas.txt'))
    

    if len(gas_file) > 0: 

        subprocess.run(['cp', skifilename, skifilename_halo]) # copy the skirt file for each galaxy

        if len(gas_file) == 1:

            maxDustFraction = 10**(-4.5)
    
        else: # 2 or more gas particles

            gas_file = np.atleast_2d(gas_file)

            dust_r = np.sqrt(gas_file[:, 0]**2 + gas_file[:, 1]**2 + gas_file[:, 2]**2) * 1e-3 # In kpc
            dust_m = np.sum(gas_file[:, 7:], axis = 1) # In Msun


            dustMasses_sorted = dust_m[np.argsort(dust_r)]

            idx_halfmass = np.min(np.argwhere((np.cumsum(dustMasses_sorted) / np.sum(dustMasses_sorted)) >= 0.5))

            dustHalfMassRadius = np.sort(dust_r)[idx_halfmass]

            dustHalfMass = (np.sum(dust_m) / 2.)

            SigmaDust = dustHalfMass / (np.pi * dustHalfMassRadius**2) # In solar masses / kpc^2

            maxDustFraction = np.clip(10**(-0.5 - np.log10(SigmaDust)), a_min = 10**(-6.5), a_max = 10**(-4.5))

            subprocess.run(['perl', '-pi', '-e', 's/maxLevel=\"0/maxLevel=\"' + str(binTreeMaxLevel) + '/g', skifilename_halo])

            subprocess.run(['perl', '-pi', '-e', 's#dust.txt#' + SKIRTinputFiles + '_dust.txt#g', skifilename_halo])

            subprocess.run(['perl', '-pi', '-e', 's/minX=\"-0/minX=\"' + str(-SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
            subprocess.run(['perl', '-pi', '-e', 's/maxX=\"0/maxX=\"' + str(SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
            subprocess.run(['perl', '-pi', '-e', 's/minY=\"-0/minY=\"' + str(-SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
            subprocess.run(['perl', '-pi', '-e', 's/maxY=\"0/maxY=\"' + str(SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
            subprocess.run(['perl', '-pi', '-e', 's/minZ=\"-0/minZ=\"' + str(-SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])
            subprocess.run(['perl', '-pi', '-e', 's/maxZ=\"0/maxZ=\"' + str(SKIRTboxsize.to('pc').value / 2.) + '/g', skifilename_halo])

            subprocess.run(['perl', '-pi', '-e', 's/maxDustFraction=\"0/maxDustFraction=\"' + str(maxDustFraction) + '/g', skifilename_halo])



        subprocess.run(['perl', '-pi', '-e', 's/numPackets=\"0/numPackets=\"' + str(Npp) + '/g', skifilename_halo])

        subprocess.run(['perl', '-pi', '-e', 's#old_stars#' + SKIRTinputFiles + '_old_stars#g', skifilename_halo])
        subprocess.run(['perl', '-pi', '-e', 's#starforming_gas#' + SKIRTinputFiles + '_starforming_gas#g', skifilename_halo])
        subprocess.run(['perl', '-pi', '-e', 's/Period0/Period' + str(int(old_stars_tmin.to('Myr').value)) + '/g', skifilename_halo])

        subprocess.run(['perl', '-pi', '-e', 's/radius=\"1 Rstar/radius=\"' + str(Rstar) + ' kpc' +  '/g', skifilename_halo])
        subprocess.run(['perl', '-pi', '-e', 's/radius=\"3 Rstar/radius=\"' + str(3. * Rstar) + ' kpc' +  '/g', skifilename_halo])
        subprocess.run(['perl', '-pi', '-e', 's/radius=\"5 Rstar/radius=\"' + str(5. * Rstar) + ' kpc' +  '/g', skifilename_halo])

    return None

editSki(snapNum, haloID, Rstar)

# print('Elapsed time to edit ski file and calculate dust surface density:', datetime.now() - startTime)