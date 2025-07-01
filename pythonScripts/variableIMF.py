import numpy as np
import unyt

def get_imf_mode(snapshot):
    try:
       return str(snapshot.metadata.parameters["COLIBREFeedback:IMF_Scaling_Variable"])
    except:
        return "Chabrier"

def variable_slope(
    variable: unyt.unyt_array,
    alpha_min: float,
    alpha_max: float,
    sigma: float,
    pivot: float,
) -> unyt.unyt_array:
    """
    Computes the IMF high mass slope value at a given variable.

    Parameters:
    variable (unyt.unyt_array): The stellar birth variable.
    alpha_min (float): The minimum slope value.
    alpha_max (float): The maximum slope value.
    sigma (float): The dispersion parameter.
    pivot (float): The pivot value.

    Returns:
    unyt.unyt_array: The computed slope value as a dimensionless unyt array.
    """

    alpha = (alpha_min - alpha_max) / (1.0 + np.exp(sigma * np.log10( variable / pivot )) ) + alpha_max
    return unyt.unyt_array(alpha, "dimensionless")

def imf_high_mass_slope(snapshot, slope_variable):
    
    def get_snapshot_param_float(snapshot, param_name: str) -> float:
        try:
            return float(snapshot.metadata.parameters[param_name].decode("utf-8"))
        except KeyError:
            raise KeyError(f"Parameter {param_name} not found in snapshot metadata.")
        

    try:
        # Extract IMF parameters from snapshot metadata
        alpha_min = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:IMF_HighMass_slope_minimum"
        )
        alpha_max = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:IMF_HighMass_slope_maximum"
        )
        sigma = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:IMF_sigmoid_inverse_width"
        )
        pivot = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:IMF_sigmoid_pivot_CGS"
        )

        # Compute slope values
        slope_values = variable_slope(
            variable=slope_variable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma=sigma,
            pivot=pivot,
        )

        slope_values = unyt.unyt_array(
            -1 * slope_values, "dimensionless" # SKIRT assumes positive slope convetion
        )

    except KeyError as e:
        slope_values = unyt.unyt_array(
            2.3 * np.ones_like(slope_variable), "dimensionless"
        )

    return slope_values;



