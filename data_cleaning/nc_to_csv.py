import xarray as xr
import numpy as np
import pandas as pd

osd_data_og = "data/original/OSD1.nc"
pfl_data_og = "data/original/PFL.nc"
ctd_data_og = "data/original/CTD4.nc"
processed_dir = "data/processed/" 
VARIABLES = [
    'Temperature',
    'Salinity', 
    'Oxygen',
    'Pressure',
    'Phosphate',
    'Silicate',
    'Total_chlorophyll',
    'Biology',
    'Total_alkalinity',
    'Partial_pressure_of_carbon_dioxide',
    'Dissolved_inorganic_carbon',
    'Tritium',
    'Helium',
    'Delta_Carbon_13',
    'Delta_Carbon_14',
    'Delta_Helium_3',
    'Argon',
    'Neon',
    'Chlorofluorocarbon_11',
    'Chlorofluorocarbon_12',
    'Chlorofluorocarbon_113',
    'Delta_Oxygen_18'
]
def nc_convert(input_path, output_path, variables=None, exclude_vars=None):
    """
    Convert ragged netCDF to flat CSV format.
    """
    print(f"{input_path} -> {output_path}")
    
    # Open netcdf
    ds = xr.open_dataset(input_path)
    
    # Variables to exclude (typically metadata or non-numeric)
    if exclude_vars is None:
        exclude_vars = ['Primary_Investigator', 'Principal_Investigator']
    
    # Auto-detect variables if not provided
    if variables is None or len(variables) == 0:
        variables = []
        for var in ds.data_vars:
            if var.endswith("_row_size"):
                base_var = var.replace("_row_size", "")
                # Only include if it's a valid variable and not in exclude list
                if base_var in ds.data_vars and base_var not in exclude_vars:
                    variables.append(base_var)
        print(f"Auto-detected variables: {variables}")
    
    # Precompute all boundaries using cumsum on row_size variables
    row_boundaries = {}
    for var in variables:
        row_size_var = f"{var}_row_size"
        if row_size_var in ds.data_vars:
            # Handle NaN values in row_size
            row_sizes = ds[row_size_var].values
            row_sizes = np.nan_to_num(row_sizes, nan=0.0).astype(int)
            row_boundaries[var] = np.cumsum(row_sizes)
    
    # Handle z separately
    if "z_row_size" in ds.data_vars:
        z_sizes = ds["z_row_size"].values
        z_sizes = np.nan_to_num(z_sizes, nan=0.0).astype(int)
        z_boundaries = np.cumsum(z_sizes)
    else:
        z_boundaries = None
    
    # Create list to store cast dataframes
    cast_tables = []
    
    # Iterate along 'casts' dimension
    n_casts = len(ds["casts"].values)
    
    for castIndex in range(n_casts):
        # We'll use z as the primary length reference (or first variable if z not available)
        if z_boundaries is not None:
            start = 0 if castIndex == 0 else z_boundaries[castIndex - 1]
            end = z_boundaries[castIndex]
            n = end - start
            z_data = ds["z"].isel(z_obs=slice(start, end)).values
        else:
            # Use first variable as reference
            ref_var = variables[0] if variables else None
            if ref_var and ref_var in row_boundaries:
                start = 0 if castIndex == 0 else row_boundaries[ref_var][castIndex - 1]
                end = row_boundaries[ref_var][castIndex]
                n = end - start
                z_data = None
            else:
                continue
        
        if n == 0:
            continue  # Skip empty casts
        
        # Dictionary to hold this cast's data
        cast_data = {}
        
        # Add metadata first (per-cast scalars repeated n times)
        cast_data["castIndex"] = np.full(n, castIndex)
        
        if "wod_unique_cast" in ds:
            cast_data["wod_unique_cast"] = np.full(n, ds["wod_unique_cast"].isel(casts=castIndex).item())
        if "date" in ds:
            cast_data["date"] = np.full(n, ds["date"].isel(casts=castIndex).item())
        if "GMT_time" in ds:
            cast_data["GMT_time"] = np.full(n, ds["GMT_time"].isel(casts=castIndex).item())
        if "lat" in ds:
            cast_data["lat"] = np.full(n, ds["lat"].isel(casts=castIndex).item())
        if "lon" in ds:
            cast_data["lon"] = np.full(n, ds["lon"].isel(casts=castIndex).item())
        if "WMO_ID" in ds:
            cast_data["WMO_ID"] = np.full(n, ds["WMO_ID"].isel(casts=castIndex).item())
        
        # Add z (depth) if available
        if z_data is not None:
            cast_data["z"] = z_data
        
        # Extract all variables for this cast
        for var in variables:
            if var in row_boundaries:
                # Get slice boundaries
                start_var = 0 if castIndex == 0 else row_boundaries[var][castIndex - 1]
                end_var = row_boundaries[var][castIndex]
                n_var = end_var - start_var
                
                # Extract the data using the observation dimension
                obs_dim = f"{var}_obs"
                if obs_dim in ds[var].dims and n_var > 0:
                    var_data = ds[var].isel({obs_dim: slice(start_var, end_var)}).values
                    
                    # Pad or truncate to match n (the reference length)
                    if len(var_data) < n:
                        # Pad with NaN
                        padded_data = np.full(n, np.nan)
                        padded_data[:len(var_data)] = var_data
                        cast_data[var] = padded_data
                    elif len(var_data) > n:
                        # Truncate (shouldn't happen, but just in case)
                        cast_data[var] = var_data[:n]
                    else:
                        cast_data[var] = var_data
                else:
                    # No data for this variable in this cast
                    cast_data[var] = np.full(n, np.nan)
        
        # Create dataframe for this cast
        temp_table = pd.DataFrame(cast_data)
        cast_tables.append(temp_table)
        
        if (castIndex + 1) % 100 == 0:
            print(f"Processed {castIndex + 1}/{n_casts} casts")
    
    # Concatenate all casts
    full_table = pd.concat(cast_tables, ignore_index=True)
    
    # Save to CSV
    full_table.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Shape: {full_table.shape}, Columns: {list(full_table.columns)}")
    
    return full_table

# Process PFL data (auto-detects all variables)
#pfl_table = nc_convert(pfl_data_og, processed_dir + "PFL_preprocessed.csv")
#osd_table = nc_convert(osd_data_og, processed_dir+"OSD2_preprocessed.csv")
ctd_table = nc_convert(ctd_data_og, processed_dir+"CTD4_preprocessed.csv")