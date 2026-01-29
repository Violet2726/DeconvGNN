import pandas as pd
import gzip
import os

def update_labels(dataset_name):
    print(f"Updating labels for {dataset_name}...")
    combined_dir = f"data/{dataset_name}/combined"
    label_path = os.path.join(combined_dir, "sc_label.tsv")
    backup_path = os.path.join(combined_dir, "sc_label_backup.tsv")
    metadata_path = "data/ref_mouse_cortex_allen/GSE115746_complete_metadata_28706-cells.csv.gz"

    if not os.path.exists(label_path):
        print(f"Error: {label_path} not found.")
        return

    # Backup original
    if not os.path.exists(backup_path):
        os.rename(label_path, backup_path)
        print(f"Original labels backed up to {backup_path}")
    else:
        # If backup exists, we read from backup to always start from a clean state
        pass

    # Read current cells (from backup)
    current_labels = pd.read_csv(backup_path, sep='\t')
    
    # Read metadata
    metadata = pd.read_csv(metadata_path, compression='gzip')
    # Metadata columns: 1: sample_name, 20: cell_subclass
    # We'll use sample_name as key
    metadata_map = metadata.set_index('sample_name')['cell_subclass'].to_dict()

    # Map labels
    # If not found in metadata, keep original class or set to 'Unknown'
    def map_detailed(row):
        cell_id = row['cell']
        detailed = metadata_map.get(cell_id)
        if pd.isna(detailed) or detailed == "" or detailed == "No Class":
            return row['cell_type'] # Fallback to original
        return detailed

    current_labels['cell_type'] = current_labels.apply(map_detailed, axis=1)
    
    # Save new labels
    current_labels.to_csv(label_path, sep='\t', index=False)
    print(f"Updated labels saved to {label_path}")
    print("New cell type distribution:")
    print(current_labels['cell_type'].value_counts().head(15))

if __name__ == "__main__":
    # Update for the posterior dataset shown in the user's image
    update_labels("V1_Mouse_Brain_Sagittal_Posterior")
    # Also update anterior and coronal if they exist and use the same reference
    update_labels("V1_Mouse_Brain_Sagittal_Anterior")
    update_labels("V1_Adult_Mouse_Brain_Coronal_Section_1")
    update_labels("CytAssist_11mm_FFPE_Mouse_Embryo")
