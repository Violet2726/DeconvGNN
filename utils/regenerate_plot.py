import argparse
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path to import STdGCN
sys.path.append(os.getcwd())

from STdGCN.visualization import save_interactive_assets

def generate_plot(dataset_name):
    print(f"Regenerating plot for {dataset_name}...")
    
    results_dir = f"./data/{dataset_name}/results"
    predict_path = os.path.join(results_dir, "predict_result.csv")
    coor_path = f"./data/{dataset_name}/combined/coordinates.csv"
    
    if not os.path.exists(predict_path):
        print(f"Error: {predict_path} not found. Have you run the training?")
        return

    # Load predictions
    predict_df = pd.read_csv(predict_path, index_col=0)
    cell_types = predict_df.columns.tolist()
    
    # Load coordinates
    # Format: index (Barcode), x, y
    coor_df = pd.read_csv(coor_path, header=0, index_col=0)
    
    # Align coordinates to predictions
    # Only keep spots that are in predictions
    common_indices = predict_df.index.intersection(coor_df.index)
    
    if len(common_indices) != len(predict_df):
        print(f"Warning: Mismatch in barcodes. Predict: {len(predict_df)}, Coor: {len(coor_df)}, Overlap: {len(common_indices)}")
    
    predict_df = predict_df.loc[common_indices]
    coor_df = coor_df.loc[common_indices]
    
    predict_vals = predict_df.values
    
    # Prepare inputs for save_interactive_assets
    coordinates = coor_df.copy()
    coordinates.columns = ['coor_X', 'coor_Y'] 
    
    try:
        save_interactive_assets(predict_vals, cell_type_list=cell_types, coordinates=coordinates, output_dir=results_dir)
        print("Success! High-resolution plot generated.")
        print(f"Image saved to: {os.path.join(results_dir, 'interactive_pie_background.png')}")
    except Exception as e:
        print(f"Error generating plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regenerate interactive plot without re-training')
    parser.add_argument('--dataset', type=str, default='CytAssist_11mm_FFPE_Mouse_Embryo',
                        help='Dataset name (e.g., V1_Mouse_Brain_Sagittal_Posterior)')
    args = parser.parse_args()
    
    generate_plot(args.dataset)
