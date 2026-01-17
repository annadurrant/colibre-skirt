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
from functools import partial
from scipy.interpolate import interp1d
import statsmodels.api as sm

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
    "--IDs",
    type=int,
    nargs='+',
    default=-1,
)

parser.add_argument(
    "--nproc",
    type=int,
    default=64,
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

min_wavelength = 1450 * 1e-4 # micron
max_wavelength = 1550 * 1e-4

def top_hat_filter(
        wavelengths, # micron
        f_lambda, # flux per wavelength in W/m2/micron
        min_wavelength, 
        max_wavelength,
):
    speed_of_light_micron = unyt.c.to_value('m/s') * 1e6
    frequencies = speed_of_light_micron / wavelengths # Hz
    f_v = f_lambda * wavelengths / frequencies # W/m2/Hz

    f_v /= 1e-26 # SI to Jy 

    log_wavelengths = np.log10(wavelengths)
    log_min, log_max = np.log10(min_wavelength), np.log10(max_wavelength)

    f_v_intp = interp1d(log_wavelengths,f_v)
    f_v_refined = f_v_intp(np.linspace(log_min,log_max,1000))
    f_v_avg = np.mean(f_v_refined)

    lum = f_v_avg / 3631
    
    return lum


def uv_continuum_slope(
    wavelengths,  # micron
    f_y,  # flux per wavelength in W/m2/micron
    y0 = 0.1250, 
    y1 = 0.2600,
):
    if y0 < wavelengths[0]:
        y0 = wavelengths[0]

    wavelengths_intp = np.linspace(y0,y1,200)
    f_y_intp = interp1d(wavelengths,f_y)(wavelengths_intp)

    C = sm.add_constant(np.log10(wavelengths_intp))

    # Least Absolute Deviation (L1) regression
    model = sm.QuantReg(np.log10(f_y_intp), C)
    result = model.fit(q=0.5)

    logA, beta = result.params
    
    return beta

def loop_luminosity(
    idx,
    aperture_name='tot'
):
    idx = int(idx)
    dset_id = np.where(halo_IDs == idx)[0][0] 

    sed_file = np.loadtxt(SKIRToutputFilePath + f'/snap{args.snap}_ID{idx}_SED_{aperture_name}_sed.dat')
    wavelengths = sed_file[:,0] # in micron
    attenuated_sed = sed_file[:,1] # in W/m2/micron
    intrinsic_sed = sed_file[:,2] # in W/m2/micron

    # run top hat filter over seds and store
    intrinsic_luminosity = top_hat_filter(wavelengths,intrinsic_sed,min_wavelength,max_wavelength)
    attenuated_luminosity = top_hat_filter(wavelengths,attenuated_sed,min_wavelength,max_wavelength)

    # compute beta slope between end points
    beta = uv_continuum_slope(wavelengths, attenuated_sed)

    return (dset_id, intrinsic_luminosity, attenuated_luminosity, beta)

def create_skirt_lum_dset(
    aperture=50 # kpc
):
    print('Aperture size [kpc]:', aperture)

    if aperture == None:
        aperture_name = 'tot'
        group_name = 'BoundSubhalo'
    else:
        aperture_name = f'{aperture}kpc'
        group_name = f'ProjectedAperture/{aperture}kpc/projz'

    # Get intrinsic values from SOAP for faint' dust-free objects
    with h5.File(catalogue_file) as fi:
        soap_dset = fi[f'{group_name}/CorrectedStellarLuminosity']
        attributes = {}
        for key in soap_dset.attrs:
            if key == 'Description':
                attributes[key] = 'Total stellar luminosity for a top hat UV band [1450-1550 A], computed with SKIRT.'
            else:
                attributes[key] = soap_dset.attrs[key]
        
        intrinsic_luminosities = soap_dset[()][:,0]
    fi.close()

    # Create arrays to store results
    attenuated_luminosities = np.copy(intrinsic_luminosities)
    beta_slopes = np.zeros_like(intrinsic_luminosities)
    extinction = np.zeros_like(intrinsic_luminosities)

    # Loop over SKIRT IDs
    sample_IDs = np.loadtxt(sampleFilepath + f'sample_{args.snap}.txt')[:,0]

    with multiprocessing.Pool(processes=args.nproc) as pool:
        results = pool.map(partial(loop_luminosity,aperture_name=aperture_name), sample_IDs)

    for dset_id, lum_int, lum_att, beta in results:
        intrinsic_luminosities[dset_id] = lum_int
        attenuated_luminosities[dset_id] = lum_att
        extinction[dset_id] = -2.5 * np.log10( lum_att / lum_int )
        beta_slopes[dset_id] = beta

    print('Finished collecting SKIRT data and extinction factors.', flush=True)

    # Create hdf5 file and save data
    output_filepath = params['OutputFilepaths']['GalaxyLuminositiesFilepath'].format(simPath=simPath,snap_nr=args.snap)
    output_fi = h5.File(output_filepath,'a')

    grp = output_fi.require_group(group_name)

    try:
        dset = grp.create_dataset('IntrinsicUVLuminosity',data=intrinsic_luminosities)
        for attribute in attributes:
            dset.attrs[attribute] = attributes[attribute]

        dset = grp.create_dataset('AttenuatedUVLuminosity',data=attenuated_luminosities)
        for attribute in attributes:
            dset.attrs[attribute] = attributes[attribute]

        grp.create_dataset('UVExtinction',data=extinction)
        grp.create_dataset('BetaSlope',data=beta_slopes)
    
    except:
        del grp['BetaSlope']
        grp.create_dataset('BetaSlope',data=beta_slopes)

    output_fi.close()

    print('Done.', flush=True)


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

create_skirt_lum_dset(aperture=None)
create_skirt_lum_dset(aperture=10)
create_skirt_lum_dset(aperture=50)

    