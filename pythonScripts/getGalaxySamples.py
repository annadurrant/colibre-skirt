"""
Script to select COLIBRE halos and store some global information.
Created by Andrea Gebek on 25.3.2025.
"""

import unyt
from swiftsimio import load as load_snapshot
import numpy as np
import yaml
import argparse
import os

# Set simName
parser = argparse.ArgumentParser(
    description="Select COLIBRE halos and store some global information."
)

parser.add_argument(
    "simName",
    type=str,
    help="Simulation name.",
)

parser.add_argument(
    "--snaps",
    type=int,
    nargs='+',
    help="<Required> Snapshot number(s).",
)

parser.add_argument(
    "--IDs",
    type=int,
    nargs='+',
    default=-1,
    help="Halo IDs to run SKIRT simulations for, based on HBT track IDs.",
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
sampleFolder = params['OutputFilepaths']['sampleFolder'].format(simPath=simPath)

# Make output directories 
os.system(f'mkdir -p {os.path.dirname(sampleFolder)}')
os.system(f'mkdir -p {sampleFolder}')

header = 'Column 1: Halo ID\n' + \
         'Column 2: Stellar mass (Msun)\n' + \
         'Column 3: Stellar half-mass radius (kpc)\n'

for snap in args.snaps:
    
    catalogue_file = params['InputFilepaths']['catalogueFile'].format(simPath=simPath,snap_nr=snap)
    catalogue = load_snapshot(catalogue_file)
   
    if args.vIMF == False:
        halo_track_IDs = catalogue.input_halos_hbtplus.track_id.value
    else: #required for now, I don't think older SOAP had hbt tracking ids included
        halo_track_IDs = catalogue.input_halos.halo_catalogue_index.value

    Mstar = unyt.unyt_array(catalogue.bound_subhalo.stellar_mass.to_physical())
    Rstar = unyt.unyt_array(catalogue.bound_subhalo.half_mass_radius_stars.to_physical())

    if args.IDs != -1:
        SEL = np.isin(halo_track_IDs, args.IDs)
    
    else:
        SEL = (Mstar >= unyt.unyt_quantity(float(params['SelectionCriteria']['minStellarMass']), 'Msun')) * (Mstar <= unyt.unyt_quantity(float(params['SelectionCriteria']['maxStellarMass']), 'Msun')) # Simple stellar mass selection. Replace this with 
        # your selection criteria.

        max_number = params['SelectionCriteria']['maxNumHalos']
        count = 0
        for sel_i,selection in enumerate(SEL):
            if selection == True and count == max_number:
                SEL[sel_i] = False
            elif selection == True:
                count += 1

    print(len(SEL[SEL]), 'galaxies selected in snapshot', snap)

    sample_file = np.vstack((halo_track_IDs, Mstar.to('Msun').value, Rstar.to('kpc').value)).T[SEL, :]

    np.savetxt(sampleFolder + 'sample_' + str(snap) + '.txt', sample_file, fmt = ['%d', '%.6e', '%.4f'], header = header)