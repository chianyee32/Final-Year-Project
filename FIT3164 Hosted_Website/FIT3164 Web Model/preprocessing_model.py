import os
import pandas as pd
import numpy as np
import requests                                # ← new
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

# ─── NEW: S3 URLs from environment ─────────────────────────────────────────────
CCLE_TRANSCRIPT_URL = os.environ.get("CCLE_TRANSCRIPT_URL")
CCLE_PROTEIN_URL    = os.environ.get("CCLE_PROTEIN_URL")


def _download_if_missing(url: str, dst: str):
    """
    Download `url` to local path `dst` if it does not already exist.
    """
    if url and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(1024*1024):
                f.write(chunk)
        print(f"Downloaded {os.path.basename(dst)} from {url}")


def preprocess_user_dataset(
    user_csv_path: str,
    output_csv_path: str = "user_preprocessed_output.csv"
) -> pd.DataFrame:
    """
    1) Load & validate user CSV
    2) Enforce required metadata columns
    3) Check for transcriptomics & proteomics features
    4) Generate 256‐bit Morgan fingerprints (radius=2)
    5) Save flattened CSV for prediction
    """

    # 0) Ensure reference CSVs are present (download from S3 if URLs set)
    _download_if_missing(
        CCLE_TRANSCRIPT_URL,
        os.path.join("Preprocess_files", "CCLE_Transcriptomics_cleaned.csv")
    )
    _download_if_missing(
        CCLE_PROTEIN_URL,
        os.path.join("Preprocess_files", "CCLE_Proteomics_cleaned.csv")
    )

    # 1) Load
    user_df = pd.read_csv(user_csv_path)
    print("Loaded user dataset.")

    # 2) Metadata columns
    required = {"DRUG_ID", "DRUG_NAME", "CCLE_Name", "COSMIC_ID", "CANCER_TYPE", "ISOSMILES"}
    missing = required - set(user_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3) Omics features check
    trans_ref = set(pd.read_csv(
        os.path.join("Preprocess_files", "CCLE_Transcriptomics_cleaned.csv"),
        nrows=0
    ).columns)
    prot_ref  = set(pd.read_csv(
        os.path.join("Preprocess_files", "CCLE_Proteomics_cleaned.csv"),
        nrows=0
    ).columns)
    feats = set(user_df.columns) - required

    missing_trans = trans_ref - feats
    missing_prot  = prot_ref  - feats
    if missing_trans or missing_prot:
        msg = "You have to include both transcriptomics and proteomics features."
        if missing_trans:
            msg += f"\nMissing transcriptomics (first 5): {list(missing_trans)[:5]}"
        if missing_prot:
            msg += f"\nMissing proteomics (first 5): {list(missing_prot)[:5]}"
        raise ValueError(msg)
    print("Omics features validated.")

    # Null / non-numeric check
    skip_cols = {"ISOSMILES","DRUG_NAME","CCLE_Name","CANCER_TYPE","DRUG_ID","COSMIC_ID"}
    for col in feats - skip_cols:
        if user_df[col].isnull().any():
            raise ValueError(f"Column '{col}' contains missing values.")
        if not pd.api.types.is_numeric_dtype(user_df[col]):
            raise ValueError(f"Column '{col}' must be numeric.")
    print("Numeric data check passed.")

    # Deduplicate
    final_df = user_df.drop_duplicates().reset_index(drop=True)

    # 4) Morgan fingerprints (radius=2, 256-bit)
    print("Generating Morgan fingerprints...")
    fps = []
    for smi in final_df["ISOSMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
            arr = np.zeros((256,), dtype=np.int64)
            ConvertToNumpyArray(bitvect, arr)
            fps.append(arr)
        else:
            print(f"Invalid SMILES: {smi}")
            fps.append(np.zeros((256,), dtype=np.int64))

    fps_df = pd.DataFrame(fps)

    # Merge and drop raw SMILES column
    out_df = pd.concat([final_df.reset_index(drop=True), fps_df], axis=1)
    drop_cols = [c for c in ("PubCHEM","ISOSMILES") if c in out_df.columns]
    out_df = out_df.drop(columns=drop_cols)

    # 5) Save
    out_df.to_csv(output_csv_path, index=False)
    print(f"Preprocessed file saved to: {output_csv_path}")

    return out_df
