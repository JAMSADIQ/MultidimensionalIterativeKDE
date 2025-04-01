import numpy as np
import csv
import os
#assuming one have this on CIT
from multiprocessing import Pool
import pandas as pd

def setup_grids():
    """Set up the parameter grids."""
    extended_grid = np.logspace(np.log10(5*5/105), np.log10(105), 299)
    m1_grid = extended_grid[124:]
    m2_grid = extended_grid[124:]
    chieff_grid = np.linspace(-1, 1, 100)
    return m1_grid, m2_grid, chieff_grid

def initialize_calculator():
    """Initialize the sensitive volume calculator."""
    import new_o123_class_found_inj_general as u_pdet
    run_fit = 'o3'
    run_dataset = 'o3'
    g = u_pdet.Found_injections(
        dmid_fun='Dmid_mchirp_fdmid_fspin', 
        emax_fun='emax_exp', 
        alpha_vary=None, 
        ini_files=None, 
        thr_far=1, 
        thr_snr=10
    )
    g.load_inj_set(run_dataset)
    g.get_opt_params(run_fit)
    g.set_shape_params()
    return g

def calculate_batch(params):
    """Calculate VT for a batch of parameters."""
    batch_id, param_batch = params
    g = initialize_calculator()
    results = []
    
    for m1val, m2val, Xival in param_batch:
        VT = g.total_sensitive_volume(m1val, m2val, chieff=Xival)
        results.append([m1val, m2val, Xival, VT])
    
    # Save batch results to temporary file
    temp_file = f'batch_results_{batch_id}.csv'
    with open(temp_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    
    return temp_file

def generate_parameter_combinations():
    """Generate parameter combinations considering symmetry."""
    m1_grid, m2_grid, chieff_grid = setup_grids()
    combinations = []
    
    for i, m1val in enumerate(m1_grid):
        for j, m2val in enumerate(m2_grid):
            # Only compute when m1 >= m2 to avoid redundancy
            if m1val >= m2val:
                for k, Xival in enumerate(chieff_grid):
                    combinations.append((m1val, m2val, Xival))
    
    return combinations

def split_into_batches(combinations, num_batches):
    """Split parameter combinations into batches."""
    batch_size = len(combinations) // num_batches
    if len(combinations) % num_batches != 0:
        batch_size += 1
    
    batches = []
    for i in range(0, len(combinations), batch_size):
        batch = combinations[i:i+batch_size]
        batches.append((i//batch_size, batch))
    
    return batches

def merge_results(temp_files, output_file):
    """Merge batch results and fill in symmetric cases."""
    # Read all temporary files and combine
    dfs = [pd.read_csv(file, header=None, names=['m1', 'm2', 'chieff', 'VT']) 
           for file in temp_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create symmetric entries
    symmetric_entries = []
    for _, row in combined_df.iterrows():
        if row['m1'] > row['m2']:  # Only create new entry if m1 > m2
            symmetric_entries.append({
                'm1': row['m2'],
                'm2': row['m1'],
                'chieff': row['chieff'],
                'VT': row['VT']
            })
    
    # Add symmetric entries
    if symmetric_entries:
        symmetric_df = pd.DataFrame(symmetric_entries)
        final_df = pd.concat([combined_df, symmetric_df], ignore_index=True)
    else:
        final_df = combined_df
    
    # Save to output file
    final_df.to_csv(output_file, index=False)
    
    # Clean up temporary files
    for file in temp_files:
        os.remove(file)

def run_parallel_computation(num_processes=4, output_file='saveVT_grid.csv'):
    """Run the computation in parallel across multiple processes."""
    # Generate parameter combinations
    combinations = generate_parameter_combinations()
    print(f"Total parameter combinations (with symmetry): {len(combinations)}")
    
    # Split into batches
    batches = split_into_batches(combinations, num_processes * 2)  # 2x batches per process for better load balancing
    print(f"Split into {len(batches)} batches")
    
    # Run in parallel
    with Pool(processes=num_processes) as pool:
        temp_files = pool.map(calculate_batch, batches)
    
    # Merge results
    merge_results(temp_files, output_file)
    print(f"Results merged and saved to {output_file}")

if __name__ == "__main__":
    # Set the number of processes based on available CPU cores
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    recommended_processes = max(1, num_cores - 1)  # Leave one core free
    
    run_parallel_computation(num_processes=recommended_processes)
