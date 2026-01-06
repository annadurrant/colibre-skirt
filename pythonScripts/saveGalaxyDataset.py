import numpy as np
from swiftsimio import load as load_snapshot
import unyt
import yaml
import argparse
import os
import h5py as h5
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing

# Set simName
parser = argparse.ArgumentParser(
    description="Convert SKIRT integrated FUV-luminosity output into hdf5 dataset."
)

parser.add_argument(
    "simName",
    type=str,
    help="Simulation name.",
)

parser.add_argument(
    "snap",
    type=int,
    help="<Required> Snapshot number.",
)

parser.add_argument(
    "outputDir",
    type=str,
    help="Name of output directory.",
)

parser.add_argument(
    "--SOAP",
    action="store_true",
    help="Also store data to SOAP catalogue.",
)

parser.add_argument(
    "--IDs",
    type=int,
    nargs='+',
    default=-1,
    help="Halo IDs to run SKIRT simulations for, based on HBT track IDs.",
)


args = parser.parse_args()

# Define filepaths from parameter file
dir_path =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
param_file = 'vimf_SKIRT_parameters.yml'
with open(f'{dir_path}/{param_file}','r') as stream:
    params = yaml.safe_load(stream)

simPath = params['InputFilepaths']['simPath'].format(simName=args.simName)
sampleFilepath = params['OutputFilepaths']['sampleFolder'].format(simPath=simPath)
SKIRToutputFilePath = params['OutputFilepaths']['SKIRToutputFilePath'].format(simPath=simPath,rotation=params['ModelParameters']['rotation'])
SKIRToutputFilePath += args.outputDir

# Load SOAP catalogue and required attributes for a SOAP dset
catalogue_file = params['InputFilepaths']['catalogueFile'].format(simPath=simPath,snap_nr=args.snap)

with h5.File(catalogue_file) as fi:
    halo_IDs = fi['InputHalos/HaloCatalogueIndex'][()]
fi.close()

def loop_luminosity(idx):

    idx=int(idx)

    file = SKIRToutputFilePath + f'/snap{args.snap}_ID{idx}_SED_tot_sed.dat'
    
    if os.path.isfile(file):
        dset_id = np.where(halo_IDs == idx)[0][0]

        wavelengths = np.loadtxt(file,usecols=0)
        sed_tot = 1e12 * np.loadtxt(file,usecols=1) / 3631 #Jy
        sed_10 = 1e12 * np.loadtxt( SKIRToutputFilePath + f'/snap{args.snap}_ID{idx}_SED_10kpc_sed.dat',usecols=1) / 3631 #Jy
        sed_50 = 1e12 * np.loadtxt( SKIRToutputFilePath + f'/snap{args.snap}_ID{idx}_SED_50kpc_sed.dat',usecols=1) / 3631 #Jy

        fuv_tot, fuv_10, fuv_50 = sed_tot[0], sed_10[0], sed_50[0] 

        # for beta slopes, convert from freq. flux to wavelength flux
        convert_fuv = (1/1350 - 1/1750) / (1750 - 1350) 
        convert_nuv = (1/1750 - 1/2800) / (2800 - 1750) 
        beta_50 = np.log10( sed_50[1]/sed_50[0] * convert_nuv/convert_fuv ) / np.log10(wavelengths[1]/wavelengths[0])
        # or is the conversion factor just between the central wavelengths?
        beta_50 = np.log10( sed_50[1]/sed_50[0] * (wavelengths[0]/wavelengths[1]**2) ) / np.log10(wavelengths[1]/wavelengths[0])
        

        return (dset_id, fuv_tot, fuv_10, fuv_50, beta_50)

def create_skirt_lum_dset():

    with h5.File(catalogue_file) as fi:
        soap_dset = fi['BoundSubhalo/CorrectedStellarLuminosity']
        attributes = {}
        for key in soap_dset.attrs:
            if key == 'Description':
                attributes[key] = 'Total dust-attenuated stellar luminosity in the GALEX FUV band, computed with SKIRT.'
            else:
                attributes[key] = soap_dset.attrs[key]
        
        halo_luminosities = np.array([
            soap_dset[()][:,0],
            fi['ProjectedAperture/10kpc/projz/CorrectedStellarLuminosity'][()][:,0],
            fi['ProjectedAperture/50kpc/projz/CorrectedStellarLuminosity'][()][:,0]
        ]).T
        
    fi.close()

    halo_luminosities_with_skirt = np.copy(halo_luminosities)

    # Read in SKIRT data and add to halo data
    # skirt_files = os.listdir(SKIRToutputFilePath)
    sample_file = np.loadtxt(sampleFilepath + f'sample_{args.snap}.txt')

    extinction = np.zeros_like(halo_luminosities_with_skirt)
    beta_slopes = np.zeros_like(halo_luminosities_with_skirt[:,0])

    with multiprocessing.Pool(processes=64) as pool:
        results = pool.map(loop_luminosity, sample_file[:,0])

    for dset_id, fuv_tot, fuv_10, fuv_50, beta_50 in results:
        skirt_lum_arr = np.array([fuv_tot, fuv_10, fuv_50])
        halo_luminosities_with_skirt[dset_id] = skirt_lum_arr
        extinction[dset_id] = -2.5 * np.log10( skirt_lum_arr / halo_luminosities[dset_id] )
        beta_slopes[dset_id] = beta_50

    print('Finished collecting SKIRT data and extinction factors.', flush=True)

    # Create hdf5 file and save data
    output_filepath = params['OutputFilepaths']['GalaxyLuminositiesFilepath'].format(simPath=simPath,snap_nr=args.snap)
    output_fi = h5.File(output_filepath,'a')

    grp_names = ['BoundSubhalo','ProjectedAperture/10kpc/projz','ProjectedAperture/50kpc/projz']

    # First save luminosities and extinction factors
    for gi, grp_name in enumerate(grp_names):

        grp = output_fi.require_group(grp_name)

        dset = grp.create_dataset('FUVStellarLuminosity',data=halo_luminosities_with_skirt[:,gi])

        for attribute in attributes:
            dset.attrs[attribute] = attributes[attribute]

        dset_extinct = grp.create_dataset('FUVExtinction',data=extinction[:,gi])

    # Also save beta slope
    grp = output_fi.require_group('ProjectedAperture/50kpc/projz')
    dset_beta = grp.create_dataset('BetaSlope',data=beta_slopes)

    output_fi.close()

    if args.SOAP == True:
        with h5.File(catalogue_file,'a') as dst_fi, h5.File(output_filepath,'a') as src_fi:
            src_fi.copy(src_fi['ProjectedAperture/50kpc/projz/FUVStellarLuminosity'],dst_fi['ProjectedAperture/50kpc/projz'],'CorrectedStellarLuminosityWithSKIRT')
        src_fi.close()
        dst_fi.close()

    print('Done.', flush=True)


def create_average_sml_dst():
    # Compute mass weighted smoothing lengths from particle files

    def loopSml(id):
        dset_id = np.where(halo_IDs == id)[0][0]
        stars_file = np.loadtxt(storeParticlesPath + f'/snap{args.snap}_ID{id}_stars.txt').T

        minitials = stars_file[5]
        smls = stars_file[3]
        massweighted_sml = minitials * smls / np.sum(minitials)
        smooothingLength = np.mean(massweighted_sml)

        return (dset_id, smooothingLength)

    SmoothingLengths = np.zeros_like(halo_IDs, dtype=float)

    with multiprocessing.Pool(processes=64) as pool:
        results = pool.map(loopSml, sample_IDs)

    for idx, value in results:
        SmoothingLengths[idx] = value

    output_fi = h5.File(output_filepath,'a')
    output_fi.create_dataset('BoundSubhalo/InitialMassWeightedSmoothingLength',data=SmoothingLengths)
    output_fi.close()

def check_error():

    n_err = 0
    redo_ids = []

    
    if args.IDs != -1:
        IDs = args.IDs
    else:
        IDs = np.loadtxt(sampleFilepath + f'sample_{args.snap}.txt')[:,0]

    errors = np.zeros_like(IDs, dtype = float)
    for i,idx in enumerate(IDs):

        stats_file = np.loadtxt(SKIRToutputFilePath + f'/snap{args.snap}_ID{int(idx)}_SED_tot_sedstats.dat')
        N = stats_file[0][1]
        w1 = stats_file[0][2]
        w2 = stats_file[0][3]
        err = np.sqrt( w2/w1**2 - 1/N )
        if err > 0.1:
            n_err += 1
            redo_ids.append(idx)
        errors[i] = err

    print(f'snapshot {args.snap}')
    print(f'There are {n_err} galaxies with R > 0.1.')
    if n_err > 0:
        print('These are:')
        for idx in redo_ids:
            print(int(idx))
    
    ids_with_errors = np.array([ IDs, errors]).T

    header = 'Column 1: Halo ID\n' + \
             'Column 2: Relative error R \n'
    
    np.savetxt(sampleFilepath + f'/relative_error_{args.snap}.txt', ids_with_errors, fmt = ['%d', '%.6f'], header = header)

# create_skirt_lum_dset()
check_error()