"""
Script to create particle .txt files, based on Kyle Oman's
swiftgalaxy framework. As of November 2024, swiftgalaxy
needs to be installed from github (see https://swiftgalaxy.readthedocs.io/en/latest/getting_started/index.html).
Created by Andrea Gebek on 18.11.2024
"""

from swiftgalaxy.halo_catalogues import SOAP
import numpy as np
from swiftgalaxy.iterator import SWIFTGalaxies
from datetime import datetime
from swiftsimio.visualisation.smoothing_length.generate import generate_smoothing_lengths as gsl
from swiftsimio import load as load_snapshot
import unyt
import yaml
import argparse
import os
import h5py
import scipy
from swiftsimio.objects import cosmo_array, cosmo_factor, a
from variableIMF import get_imf_mode, imf_high_mass_slope
import warnings
warnings.filterwarnings("ignore")

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
storeParticlesPath = params['OutputFilepaths']['storeParticlesPath'].format(simPath=simPath,rotation=params['ModelParameters']['rotation']) # Folder where the .txt particle files are stored

z_list = np.loadtxt(simPath + '/output_list.txt', delimiter = ',', usecols = 0)     # Load the redshifts of the available snapshots

rotation_mode = params['ModelParameters']['rotation']

os.system(f'mkdir -p {storeParticlesPath}')

gas_header = 'Column 1: x (pc)\n' + \
    'Column 2: y (pc)\n' + \
    'Column 3: z (pc)\n' + \
    'Column 4: smoothing length (pc)\n' + \
    'Column 5: metallicity (1)\n' + \
    'Column 6: temperature (K)\n' + \
    'Column 7: mass density (Msun/pc3)\n' + \
    'Column 8: mass (Msun)\n' + \
    'Column 9: instantaneous star formation rate (Msun/yr)\n' + \
    'Column 10: star formation rate averaged over 10 Myr (Msun/yr)\n' + \
    'Column 11: dust mass large graphite (Msun)\n' + \
    'Column 12: dust mass large Mg silicates (Msun)\n' + \
    'Column 13: dust mass large Fe silicates (Msun)\n' + \
    'Column 14: dust mass small graphite (Msun)\n' + \
    'Column 15: dust mass small Mg silicates (Msun)\n' + \
    'Column 16: dust mass small Fe silicates (Msun)\n' + \
    'Column 17: IMF high mass slope (1)\n'

stars_header = 'Column 1: x (pc)\n' + \
            'Column 2: y (pc)\n' + \
            'Column 3: z (pc)\n' + \
            'Column 4: smoothing length (neighbouring star particles) (pc)\n' + \
            'Column 5: smoothing length (neighbouring gas particles) (pc)\n' + \
            'Column 6: initial stellar mass (Msun)\n' + \
            'Column 7: current stellar mass (Msun)\n' + \
            'Column 8: gas density at birth from parent gas particle (Msun/pc3)\n' + \
            'Column 9: metallicity (1)\n' + \
            'Column 10: age (yr)\n' + \
            'Column 11: IMF high mass slope (1)\n'

def attach_membership_info_to_sg_and_mask(sg, membership_filename):
    # Attaches SOAP membership information to SWIFTGalaxy object 
    # if SWIFTGalaxies could not be run with a virtual snapshot file.

    mfile = h5py.File(membership_filename, "r")
    for gname, ptype in zip(
        sg.metadata.present_group_names, sg.metadata.present_groups
    ):
        groupnr_bound = np.concatenate(
            [
                mfile[f"{ptype}/GroupNr_bound"][read_range[0] : read_range[1]]
                for read_range in getattr(sg.mask, f"_{gname}")
            ]
        )
        getattr(sg, gname)._particle_dataset.group_nr_bound = cosmo_array(
            groupnr_bound,
            unyt.dimensionless,
            comoving=True,
            cosmo_factor=cosmo_factor(a**0, sg.metadata.scale_factor),
        )

    mfile.close()
    extra_mask = sg.halo_catalogue._generate_bound_only_mask(sg)
    sg.mask_particles(extra_mask)
    return

def analysis(sg, halo_ID, snap):
    # this function can also have additional args & kwargs, if needed
    # it should only access the pre-loaded data fields

    # print('Saving txt files for halo ID:', halo_ID)

    z = z_list[snap] # Redshift of the snapshot

    if add_mem == True:
        attach_membership_info_to_sg_and_mask(
            sg, 
            membership_file
        )

    # Coordinates
    stars_coordinates = sg.stars.coordinates.to('pc').to_physical().value
    gas_coordinates = sg.gas.coordinates.to('pc').to_physical().value

    if rotation_mode != 'None':
        
        if rotation_mode == 'face_on':
            alignment_vector = [0, 0, 1]
        elif rotation_mode == 'edge_on':
            alignment_vector = [0, 1, 0]
        else:
            raise Warning('The rotation parameter can only be None, face_on or edge_on.')
        
        angular_momentum_vector /= -1 * np.linalg.norm(angular_momentum_vector)
        
        rotation_matrix,_ = scipy.spatial.transform.Rotation.align_vectors(alignment_vector, angular_momentum_vector)
        rotation_matrix.as_matrix()
        
        stars_coordinates = rotation_matrix.apply(stars_coordinates)
        gas_coordinates = rotation_matrix.apply(gas_coordinates)

    # Star particles
    #

    stars_x, stars_y, stars_z = stars_coordinates.T
    
    stars_sml_fromGas = sg.stars.smoothing_lengths.to('pc').to_physical() * 2.018932 # Using neighbouring gas particles
    try:
        # Recalculate stellar smoothing lengths, following COLIBRE tutorials
        stars_sml_fromStars = gsl((sg.stars.coordinates + sg.centre) % sg.metadata.boxsize, sg.metadata.boxsize,
                        kernel_gamma = 1.0, neighbours = 32, speedup_fac = 1, dimension = 3).to('pc').to_physical() # Using neighbouring star particles
        if np.inf in stars_sml_fromStars:
            stars_sml_fromStars = stars_sml_fromGas
    except:
        stars_sml_fromStars = stars_sml_fromGas

    stars_Z = sg.stars.metal_mass_fractions.to_physical()
    stars_Minit = sg.stars.initial_masses.to('Msun').to_physical()
    stars_Mcurr = sg.stars.masses.to('Msun').to_physical()
    stars_birthDensity = sg.stars.birth_densities.to('Msun/pc**3').to_physical()
    stars_age = sg.stars.ages.to('yr').to_physical()

    # set IMF high mass slope
    if args.vIMF == True:
        imf_mode = get_imf_mode(sg)
        if imf_mode == "Chabrier" or "Density":
            slope_variable = stars_birthDensity.to("g/cm**3") / unyt.mh.to("g")
        elif imf_mode == "Redshift":
            slope_variable = 1 / sg.stars.birth_scale_factors.value - 1 

        stars_imf_slopes = imf_high_mass_slope(sg, slope_variable)
    else:
        stars_imf_slopes = unyt.unyt_array(2.3 * np.ones_like(stars_age.value), "dimensionless")

    stars_params = np.transpose([stars_x, stars_y, stars_z, stars_sml_fromStars, stars_sml_fromGas, stars_Minit, stars_Mcurr, stars_birthDensity, stars_Z, stars_age, stars_imf_slopes])  

    np.savetxt(storeParticlesPath + 'snap' + str(snap) + '_ID' + str(halo_ID) + '_stars.txt', stars_params, fmt = '%.6e', header = 'Redshift: ' + str(z) + '\n' + stars_header)

    # Gas particles
    #

    gas_x, gas_y, gas_z = gas_coordinates.T
    gas_sml = sg.gas.smoothing_lengths.to('pc').to_physical() * 2.018932
    gas_Z = sg.gas.metal_mass_fractions.to_physical()
    gas_T = sg.gas.temperatures.to('K').to_physical()
    gas_rho = sg.gas.densities.to('Msun/pc**3').to_physical()
    gas_SFR = sg.gas.star_formation_rates.to_physical().to('Msun/yr') # Instantaneous SFRs
    gas_SFR_10Myr = sg.gas.averaged_star_formation_rates[:, 1].to_physical().to('Msun/yr') # 10-Myr averaged SFRs
    gas_M = sg.gas.masses.to('Msun').to_physical()
    DustSpecies = sg.gas.dust_mass_fractions.named_columns
    gas_fDust = np.array([getattr(sg.gas.dust_mass_fractions, name) for name in DustSpecies]).T
    dust_M = (gas_fDust * np.atleast_1d(gas_M)[:, np.newaxis].repeat(6, axis = 1)).to('Msun').to_physical()

    # set IMF high mass slope
    if args.vIMF == True:
        if imf_mode == "Chabrier" or "Density":
            slope_variable = gas_rho.to("g/cm**3") / unyt.mh.to("g")
        elif imf_mode == "Redshift":
            slope_variable = unyt.unyt_array(z * np.ones_like(gas_Z.value), "dimensionless") 

        gas_imf_slopes = imf_high_mass_slope(sg, slope_variable)
    else:
        gas_imf_slopes = unyt.unyt_array(2.3 * np.ones_like(gas_Z.value), "dimensionless")

    gas_params = np.transpose([gas_x, gas_y, gas_z, gas_sml, gas_Z, gas_T, gas_rho, gas_M, gas_SFR, gas_SFR_10Myr,
                            dust_M[:, 0], dust_M[:, 1], dust_M[:, 2], dust_M[:, 3], dust_M[:, 4], dust_M[:, 5], gas_imf_slopes])


    np.savetxt(storeParticlesPath + 'snap' + str(snap) + '_ID' + str(halo_ID) + '_gas.txt', gas_params, fmt = '%.6e', header = 'Redshift: ' + str(z) + '\n' + gas_header)

    return None


for snap in args.snaps:

    startTime = datetime.now()

    catalogue_file = params['InputFilepaths']['catalogueFile'].format(simPath=simPath,snap_nr=snap)
    virtual_snapshot_file = params['InputFilepaths']['virtualSnapshotFile'].format(simPath=simPath,snap_nr=snap)

    catalogue = load_snapshot(catalogue_file)

    # halo_IDs_all = catalogue.input_halos_hbtplus.track_id.value
    halo_IDs_all = catalogue.input_halos.halo_catalogue_index.value

    if args.distr != -1:
        sampleFile = sampleFolder + '/sample_' + str(snap) + '/sample_' + str(snap) + '.' + str(args.distr) + '.txt'
    else:
        sampleFile = sampleFolder + '/sample_' + str(snap) + '.txt'
    
    halo_IDs = np.loadtxt(sampleFile, usecols = 0)
    halo_IDs = halo_IDs.astype(int)

    SEL = np.isin(halo_IDs_all, halo_IDs)

    # In case the halo IDs from the sample .txt file are sorted differently for some reason,
    # take care of that here by resorting the halo IDs.
    indices = np.array([list(halo_IDs).index(i) for i in halo_IDs_all[SEL]])
    halo_IDs = halo_IDs[indices] # Re-sort halo IDs from the sample file to match the COLIBRE halo ordering

    halo_indices = np.where(SEL)[0]

    print(len(halo_IDs), 'galaxies in snapshot', snap, 'selected.')

    soap = SOAP(catalogue_file, soap_index = halo_indices)

    preload_fields = {'stars.coordinates', 'stars.smoothing_lengths', 'stars.metal_mass_fractions', 'stars.initial_masses', 'stars.ages',
                    'stars.birth_densities', 'stars.masses',
                    'gas.coordinates', 'gas.smoothing_lengths', 'gas.temperatures', 'gas.densities', 'gas.masses', 'gas.metal_mass_fractions',
                    'gas.star_formation_rates', 'gas.averaged_star_formation_rates',
                    'gas.dust_mass_fractions.GraphiteLarge', 'gas.dust_mass_fractions.MgSilicatesLarge', 'gas.dust_mass_fractions.FeSilicatesLarge',
                    'gas.dust_mass_fractions.GraphiteSmall', 'gas.dust_mass_fractions.MgSilicatesSmall', 'gas.dust_mass_fractions.FeSilicatesSmall'}

    if os.path.exists(virtual_snapshot_file):
        sgs = SWIFTGalaxies(
            virtual_snapshot_file,
            SOAP(
                catalogue_file,
                soap_index=halo_indices,
            ),
            preload=preload_fields,
        )
        add_mem = False

    else: 
        # virtual snapshot does not exist 
        # run SWIFTGalaxies without membership information first
        
        snapshot_file = params['InputFilepaths']['snapshotFile'].format(simPath=simPath,snap_nr=snap)
        membership_file = params['InputFilepaths']['membershipFile'].format(simPath=simPath,snap_nr=snap)
        
        sgs = SWIFTGalaxies(
            snapshot_file,
            SOAP(
                catalogue_file,
                soap_index=halo_indices,
                extra_mask=None,
            ),
            preload=preload_fields,
        )
        add_mem = True

        # and then add membership information
        for sg in sgs:
            attach_membership_info_to_sg_and_mask(sg, membership_file)

    # map accepts arguments `args` and `kwargs`, passed through to function, if needed
    sgs.map(analysis, args = list(zip(halo_IDs, np.full(len(halo_IDs), snap))))

    print('Elapsed time for snapshot', snap, ':', datetime.now() - startTime)