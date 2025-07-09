import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import re

def preprocess_user_dataset(user_csv_path: str, output_csv_path: str = "user_preprocessed_output.csv"):
    # Load user dataset
    user_df = pd.read_csv(user_csv_path)
    print("Loaded user dataset.")

    # ✅ Enforce required metadata columns
    required_cols = {"DRUG_ID", "DRUG_NAME", "CCLE_Name", "COSMIC_ID", "CANCER_TYPE", "ISOSMILES"}
    missing_required = required_cols - set(user_df.columns)

    if missing_required:
        raise ValueError(f"Missing required columns: {list(missing_required)}")

    # Validate presence of transcriptomics and proteomics features
    # Load reference feature sets
    ref_trans_cols = set(pd.read_csv("Preprocess_files/CCLE_Transcriptomics_cleaned.csv", nrows=0).columns)
    ref_prot_cols = set(pd.read_csv("Preprocess_files/CCLE_Proteomics_cleaned.csv", nrows=0).columns)

    # Exclude metadata columns from comparison
    metadata_cols = {"DRUG_ID", "DRUG_NAME", "CCLE_Name", "COSMIC_ID", "CANCER_TYPE", "ISOSMILES"}
    user_feature_cols = set(user_df.columns) - metadata_cols

    # Check for missing features
    missing_trans = ref_trans_cols - user_feature_cols
    missing_prot = ref_prot_cols - user_feature_cols

    if missing_trans or missing_prot:
        error_message = "You have to include both Transcriptomics and Proteomics features."
        if missing_trans:
            error_message += f"\nMissing transcriptomics features: {list(missing_trans)[:5]} ..."
        if missing_prot:
            error_message += f"\nMissing proteomics features: {list(missing_prot)[:5]} ..."
        raise ValueError(error_message)
    else:
        print("User dataset includes required omics features.")

        # ✅ Now check for null or non-numeric data
        skip_cols = {"ISOSMILES", "DRUG_NAME", "CCLE_Name", "CANCER_TYPE", "DRUG_ID", "COSMIC_ID"}

        for col in user_feature_cols - skip_cols:
            if user_df[col].isnull().any():
                raise ValueError(f"Column '{col}' contains null (missing) values.")
            if not pd.api.types.is_numeric_dtype(user_df[col]):
                raise ValueError(f"Column '{col}' contains non-numeric data.")
        
        print("All omics features are numeric and have no missing values.")

 
    
    # Load cleaned files
    print("Loading reference files...")

    # Determine path based on available columns
    if 'ISOSMILES' in user_df.columns:
        print("User dataset contains ISOSMILES. No merging needed.")
        final_merged = user_df.copy()

    final_merged = final_merged.drop_duplicates().reset_index(drop=True)

    # Generate Morgan fingerprints
    print("Generating Morgan fingerprints...")
    arr = []
    morgan_generator = AllChem.GetMorganGenerator(radius=2, fpSize=256)

    for smiles in final_merged['ISOSMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = morgan_generator.GetFingerprint(mol)
            fp_array = np.zeros((256,), dtype=np.int64)
            ConvertToNumpyArray(fp, fp_array)
            arr.append(fp_array)
        else:
            print(f"Invalid SMILES: {smiles}")
            arr.append(np.zeros((256,), dtype=np.int64))

    # Attach fingerprint data
    fingerprints_df = pd.DataFrame(arr)
    df_with_fps = final_merged.join(fingerprints_df)

    # Clean-up optional fields
    df_with_fps = df_with_fps.drop(columns=[col for col in ['PubCHEM', 'ISOSMILES'] if col in df_with_fps.columns])

    # Save output
    df_with_fps.to_csv(output_csv_path, index=False)
    print(f"Final preprocessed file saved to: {output_csv_path}")

    return df_with_fps
