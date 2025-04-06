#python csv_to_hdf.py  csvfilename.csv
import numpy as np
import pandas as pd
import h5py
import sys
def process_and_save_to_hdf(csv_filepath, hdf_filepath):
    # Read the CSV file with headers
    df = pd.read_csv(csv_filepath)
    
    # Verify the expected columns are present
    expected_columns = ['m1', 'm2', 'chieff', 'VT']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError("CSV file must contain columns: m1, m2, Xieff, VT")
    
    # Get unique values for grids
    m1grid = np.sort(df['m1'].unique())
    m2grid = np.sort(df['m2'].unique())
    Xigrid = np.sort(df['Xieff'].unique())
    
    # Check if data has complete grid or needs symmetry
    expected_size = len(m1grid) * len(m2grid) * len(Xigrid)
    actual_size = len(df)
    
    # Initialize 3D array
    VT_3Dgrid = np.zeros((len(m1grid), len(m2grid), len(Xigrid)))
    
    if actual_size == expected_size:
        # Data is complete, just reshape
        # Need to sort the data properly before reshaping
        df_sorted = df.sort_values(by=['m1', 'm2', 'Xieff'])
        VT_3Dgrid = df_sorted['VT'].values.reshape((len(m1grid), len(m2grid), len(Xigrid)))
    else:
        # Data is incomplete, need to use symmetry
        print(f"Warning: Data incomplete. Expected {expected_size} points, got {actual_size}. Using symmetry.")
        
        # Create a dictionary for fast lookup
        data_dict = {}
        for _, row in df.iterrows():
            i = np.where(m1grid == row['m1'])[0][0]
            j = np.where(m2grid == row['m2'])[0][0]
            k = np.where(Xigrid == row['Xieff'])[0][0]
            data_dict[(i, j, k)] = row['VT']
        
        # Fill the array using symmetry where needed
        for i in range(len(m1grid)):
            for j in range(len(m2grid)):
                for k in range(len(Xigrid)):
                    if (i, j, k) in data_dict:
                        VT_3Dgrid[i, j, k] = data_dict[(i, j, k)]
                    elif (j, i, k) in data_dict:
                        VT_3Dgrid[i, j, k] = data_dict[(j, i, k)]
                    else:
                        raise ValueError(
                            f"Missing data for m1={m1grid[i]}, m2={m2grid[j]}, Xieff={Xigrid[k]} "
                            "and no symmetric point available"
                        )
    
    # Save to HDF5 file
    with h5py.File(hdf_filepath, 'w') as hf:
        hf.create_dataset('m1grid', data=m1grid)
        hf.create_dataset('m2grid', data=m1grid)  # As per your requirement, m2grid = m1grid
        hf.create_dataset('Xigrid', data=Xigrid)
        hf.create_dataset('VT_3Dgrid', data=VT_3Dgrid)
    
    print(f"Data successfully saved to {hdf_filepath}")

# Example usage:
process_and_save_to_hdf(sys.argv[1], 'output_VT_on_eval_grid.h5')
