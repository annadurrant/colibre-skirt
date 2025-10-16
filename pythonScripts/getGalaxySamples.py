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

parser.add_argument(
    "--nchunks",
    type=int,
    default=1,
    help="Number of chunks to split the galaxy sample into.",
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
         'Column 2: Total stellar mass (Msun)\n' + \
         'Column 3: Stellar half-mass radius (kpc)\n' + \
         'Column 4: 50-kpc exclusive-sphere dust mass (Msun)\n' + \
         'Column 5: Total dust mass (Msun)\n'

for snap in args.snaps:
    
    catalogue_file = params['InputFilepaths']['catalogueFile'].format(simPath=simPath,snap_nr=snap)
    catalogue = load_snapshot(catalogue_file)
   
    Mstar = unyt.unyt_array(catalogue.bound_subhalo.stellar_mass.to_physical())
    Rstar = unyt.unyt_array(catalogue.bound_subhalo.half_mass_radius_stars.to_physical())
    Mdust = unyt.unyt_array(catalogue.exclusive_sphere_50kpc.dust_small_grain_mass.to_physical() + catalogue.exclusive_sphere_50kpc.dust_large_grain_mass.to_physical())
    Rdust = unyt.unyt_array(catalogue.bound_subhalo.half_mass_radius_dust.to_physical())

    if args.IDs != -1:
        # Select based on specific input IDs
        halo_IDs = catalogue.input_halos_hbtplus.track_id.value
        SEL = np.isin(halo_IDs, args.IDs)
    
    else:

        halo_IDs = catalogue.input_halos.halo_catalogue_index.value

        if float(params['SelectionCriteria']['maxUVMagnitude']) != 0 :

                Luv = unyt.unyt_array(catalogue.projected_aperture_50kpc_projz.corrected_stellar_luminosity.to_physical()[:,4])
                minUV = 10 ** (-0.4 * unyt.unyt_quantity(float(params['SelectionCriteria']['maxUVMagnitude']), 'dimensionless'))

                SEL = (Luv >= minUV) 

        else:
            SEL = (Mstar >= unyt.unyt_quantity(float(params['SelectionCriteria']['minStellarMass']), 'Msun')) * \
                (Mstar <= unyt.unyt_quantity(float(params['SelectionCriteria']['maxStellarMass']), 'Msun'))

        # remove dust-free haloes
        SEL *= (Mdust > 0)


    ngal = len(SEL[SEL])
    print(ngal, 'galaxies selected in snapshot', snap)

    sample_file = np.vstack((
        halo_IDs, Mstar.to('Msun').value, 
        Rstar.to('kpc').value, 
        Mdust.to('Msun').value,
        Rdust.to('kpc').value, 
    )).T[SEL, :]

    # Split sample into nchunks for faster processing
    if args.nchunks > 1:
        nproc = args.nchunks
        gal_per_slice = int(ngal/nproc)
        gals_slice_collections = []
        for i in range(nproc):
            if i < nproc-1:
                particles_indicies = np.arange(i*gal_per_slice,(i+1)*gal_per_slice)
            else:
                particles_indicies = np.arange(i*gal_per_slice,ngal)
            gals_slice_collections.append(particles_indicies)

        os.system(f'mkdir -p {sampleFolder}/sample_{snap}')

        for id,collection in enumerate(gals_slice_collections):
            sample_slice = sample_file[collection[0]:collection[-1]+1]
            np.savetxt(sampleFolder + f'/sample_{snap}/sample_{snap}.{id}.txt',sample_slice, fmt = ['%d', '%.6e', '%.4f', '%.6e', '%.4f'], header = header)

    # But still save whole sample too

    np.savetxt(sampleFolder + 'sample_' + str(snap) + '.txt', sample_file, fmt = ['%d', '%.6e', '%.4f', '%.6e', '%.4f'], header = header)