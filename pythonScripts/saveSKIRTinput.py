"""
Script to create SKIRT input .txt files, from already
stored .txt star and gas files.
Created by Andrea Gebek on 12.3.2025.
"""

import numpy as np
import unyt
import sys
import warnings
from datetime import datetime

startTime = datetime.now()

snapNum = sys.argv[1]
haloID = sys.argv[2]

txtFilePath = sys.argv[3]
SKIRTinputFilePath = sys.argv[4]

old_stars_tmin = unyt.unyt_quantity(10., 'Myr')

f = open(txtFilePath + 'snap' + snapNum + '_' + 'ID' + haloID + '_stars.txt', 'r')
header = f.readline() # Read first header line
redshift = float(header.split(' ')[-1])
f.close()

scaleFactor = 1. / (1. + redshift) # Scale factor for the snapshot
SKIRTboxsize = unyt.unyt_quantity(min(100., 100. * 1.8 / 0.7 * scaleFactor), 'kpc') # Scale SKIRT box size akin to COLIBRE gravitational softening length


# Star particles

with warnings.catch_warnings():
    warnings.simplefilter('ignore') # Ignore warning if file is empty
    stars_file = np.atleast_2d(np.loadtxt(txtFilePath + 'snap' + snapNum + '_' + 'ID' + haloID + '_stars.txt'))

if np.shape(stars_file) != (1, 0): # At least one star particle

    stars_x = unyt.unyt_array(stars_file[:, 0], 'pc')
    stars_y = unyt.unyt_array(stars_file[:, 1], 'pc')
    stars_z = unyt.unyt_array(stars_file[:, 2], 'pc')
    stars_sml = unyt.unyt_array(stars_file[:, 3], 'pc') # Smoothing length derived from neighbouring star particles
    stars_sml_fromGas = unyt.unyt_array(stars_file[:, 4], 'pc') # Smoothing length derived from neighbouring gas particles
    stars_M = unyt.unyt_array(stars_file[:, 5], 'Msun') # Initial stellar mass
    stars_Z = unyt.unyt_array(stars_file[:, 8], 'dimensionless')
    stars_age = unyt.unyt_array(stars_file[:, 9], 'yr')

    stars_birthDensity = unyt.unyt_array(stars_file[:, 7], 'Msun/pc**3')
    stars_SFR = 0.01 * np.sqrt(32. * unyt.physical_constants.G / (3. * np.pi)) * np.sqrt(stars_birthDensity) * stars_M # Based on the gas density at the birth of the star particle
    corr_factor = (old_stars_tmin - stars_age) / old_stars_tmin # Correction factor (see Joop's comment on slack)
    stars_SFR = stars_SFR.to('Msun/yr')

    old_stars_mask = (stars_age >= old_stars_tmin)

    old_stars_params = np.transpose([stars_x, stars_y, stars_z, stars_sml, stars_M, stars_Z, stars_age])[old_stars_mask, :]

    stars_SFE = unyt.unyt_array(np.full(len(stars_sml), 0.025), 'dimensionless') # Star-formation efficiency, 2.5%
    stars_n_cl = unyt.unyt_array(np.full(len(stars_sml), 320.), '1/cm**3') # Cloud density

    starforming_parentGas_params = np.transpose([stars_x, stars_y, stars_z, stars_sml_fromGas, stars_Z, stars_SFE, stars_n_cl, stars_SFR])[~old_stars_mask, :]

else:

    old_stars_params = np.array([])

old_stars_header = 'Column 1: x (pc)\n' + \
            'Column 2: y (pc)\n' + \
            'Column 3: z (pc)\n' + \
            'Column 4: smoothing length (pc)\n' + \
            'Column 5: initial stellar mass (Msun)\n' + \
            'Column 6: metallicity (1)\n' + \
            'Column 7: age (yr)\n'


np.savetxt(SKIRTinputFilePath + 'snap' + snapNum + '_ID' + haloID + '_old_stars.txt', old_stars_params, fmt = '%.6e', header = old_stars_header)


# Gas/dust particles
#
with warnings.catch_warnings():
    warnings.simplefilter('ignore') # Ignore warning if file is empty
    gas_file = np.atleast_2d(np.loadtxt(txtFilePath + 'snap' + snapNum + '_' + 'ID' + haloID + '_gas.txt'))

if  np.shape(gas_file) != (1, 0): # At least one gas particle


    gas_x = unyt.unyt_array(gas_file[:, 0], 'pc')
    gas_y = unyt.unyt_array(gas_file[:, 1], 'pc')
    gas_z = unyt.unyt_array(gas_file[:, 2], 'pc')
    gas_sml = unyt.unyt_array(gas_file[:, 3], 'pc')
    gas_Z = unyt.unyt_array(gas_file[:, 4], 'dimensionless')
    gas_SFR10Myr = unyt.unyt_array(gas_file[:, 9], 'Msun/yr')
    gas_Mdust = unyt.unyt_array(gas_file[:, 10:], 'Msun')

    # Dust

    dust_mask = (np.abs(gas_x.to('kpc').value) <= SKIRTboxsize.to('kpc').value / 2.) * (np.abs(gas_y.to('kpc').value) <= SKIRTboxsize.to('kpc').value / 2.) * (np.abs(gas_z.to('kpc').value) <= SKIRTboxsize.to('kpc').value / 2.)


    dust_params = np.transpose([gas_x, gas_y, gas_z, gas_sml,
                                gas_Mdust[:, 0], gas_Mdust[:, 1] + gas_Mdust[:, 2], gas_Mdust[:, 3], gas_Mdust[:, 4] + gas_Mdust[:, 5]])[dust_mask, :]
    # Note: Mg-rich and Fe-rich silicates are summed up, as they are not distinguished in our current SKIRT setup


    # SF gas

    gas_SFE = unyt.unyt_array(np.full(len(gas_sml), 0.025), 'dimensionless') # Star-formation efficiency, 2.5%
    gas_n_cl = unyt.unyt_array(np.full(len(gas_sml), 320.), '1/cm**3') # Cloud density

    starforming_gas_mask = (gas_SFR10Myr > 0.)

    starforming_gas_params = np.transpose([gas_x, gas_y, gas_z, gas_sml, gas_Z, gas_SFE, gas_n_cl, gas_SFR10Myr])[starforming_gas_mask, :]

else:

    starforming_gas_params = np.empty((0, 8)) # Needs to be a 2D array for concatenation later
    dust_params = np.array([])

starformingregions_params = np.concatenate((starforming_gas_params, starforming_parentGas_params))

starforming_gas_header = 'Column 1: x (pc)\n' + \
            'Column 2: y (pc)\n' + \
            'Column 3: z (pc)\n' + \
            'Column 4: smoothing length (pc)\n' + \
            'Column 5: metallicity (1)\n' + \
            'Column 6: star formation efficiency (1)\n' + \
            'Column 7: cloud density (1/cm3)\n' + \
            'Column 8: star formation rate averaged over 10 Myr (Msun/yr)\n'

np.savetxt(SKIRTinputFilePath + 'snap' + snapNum + '_ID' + haloID + '_starforming_gas.txt', starformingregions_params, fmt = '%.6e', header = starforming_gas_header)

dust_header = 'Column 1: x (pc)\n' + \
    'Column 2: y (pc)\n' + \
    'Column 3: z (pc)\n' + \
    'Column 4: smoothing length (pc)\n' + \
    'Column 5: dust mass large graphite (Msun)\n' + \
    'Column 6: dust mass large silicates (Msun)\n' + \
    'Column 7: dust mass small graphite (Msun)\n' + \
    'Column 8: dust mass small silicates (Msun)\n'

np.savetxt(SKIRTinputFilePath + 'snap' + snapNum + '_ID' + haloID + '_dust.txt', dust_params, fmt = '%.6e', header = dust_header)

print('Elapsed time to save SKIRT input files:', datetime.now() - startTime)