"""
snob_effect_builder.py

Pipeline summary
----------------
1. Load pre‑computed DINO / DINOv2 features and image metadata.
2. Build a DataFrame linking each NFT to its collection contract.
3. Check which contracts already have sales stored in Postgres.
4. Use Alchemy to fetch sales for the remaining contracts.
5. Bulk‑insert those sales into `nfttosales_2`.

Prerequisites
-------------
* Postgres tables:
    - collectiontoaddress(slug, address)
    - nfttosales_2(contract, token_id, sale_price)
* Helper modules:
    feature_extract, psql_methods, alchemy_methods
Adjust SQL strings or paths as needed.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from more_itertools import chunked
from tqdm import tqdm

# Project‑specific helpers
import feature_extract            # helper with get_labels / get_filenames
import psql_methods as psql       # simple execute_commands wrapper
import alchemy_methods as alc     # has NFT_to_sales(contract, token_id)

DEFAULT_FEATURE_FILE = "testfeat.pth"   # created by earlier notebook run
CHUNK_SIZE = 10_000                     # rows per Alchemy request


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Build snob‑effect sales table")
    parser.add_argument(
        "--model_string",
        default="dinov2_vits14",
        help="DINOv2 model name used when features were extracted",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Directory containing the images used for feature extraction",
    )
    parser.add_argument(
        "--out_path",
        required=True,
        help="Directory that holds the saved feature file (testfeat.pth)",
    )
    parser.add_argument(
        "--feature_file",
        default=DEFAULT_FEATURE_FILE,
        help=f"Feature filename (default: {DEFAULT_FEATURE_FILE})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def load_features(data_path, features_dir, feature_file):
    """Load features, labels, and parsed filenames."""
    feat_path = Path(features_dir) / feature_file
    if not feat_path.is_file():
        raise FileNotFoundError(f"Expected feature file {feat_path} not found")

    features = torch.load(feat_path)
    labels = feature_extract.get_labels(data_path)
    files = feature_extract.get_filenames(data_path)
    return features, labels, files


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def slug_to_contract_map():
    """Return {collection_slug: contract_address} from Postgres."""
    rows = psql.execute_commands(["SELECT slug, address FROM collectiontoaddress"])
    return {slug: address for slug, address in rows}


def contracts_with_sales():
    """Return set of contract addresses already present in nfttosales_2."""
    rows = psql.execute_commands(["SELECT DISTINCT contract FROM nfttosales_2"])
    return {r[0] for r in rows}


def insert_sales(sales):
    """Bulk‑insert rows into nfttosales_2."""
    cmd = (
        "INSERT INTO nfttosales_2 (contract, token_id, sale_price)"
        " VALUES (%s, %s, %s)"
    )
    commands = [cmd] * len(sales)
    psql.execute_commands(commands, list(sales))


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def build_dataframe(features, labels, filenames, slug_to_contract):
    """Assemble tidy DataFrame for downstream processing."""
    return pd.DataFrame(
        {
            "Label": labels.tolist(),
            "Features": features.tolist(),
            "Collection": [f[0] for f in filenames],
            "NFT_num": [f[1] for f in filenames],
            "Contract": [slug_to_contract.get(f[0]) for f in filenames],
        }
    )


def harvest_and_store_sales(df, existing_contracts, chunk_size=CHUNK_SIZE):
    """Pull sales from Alchemy and insert any that aren’t yet in Postgres."""
    pairs = df.loc[df["Contract"].notna(), ["Contract", "NFT_num"]].to_numpy().tolist()
    pairs = [p for p in pairs if p[0] not in existing_contracts]

    logging.info(
        "Fetching sales for %d NFTs across %d new contracts",
        len(pairs),
        len({p[0] for p in pairs}),
    )

    for batch in chunked(pairs, chunk_size):
        sales = alc.NFT_to_sales(batch)  # returns (contract, token_id, price)
        insert_sales(sales)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
    )

    logging.info("Loading features")
    feats, lbls, files = load_features(args.data_path, args.out_path, args.feature_file)

    logging.info("Building DataFrame")
    slug_map = slug_to_contract_map()
    df = build_dataframe(feats, lbls, files, slug_map)

    logging.info("Checking existing sales")
    existing = contracts_with_sales()

    logging.info("Harvesting new sales")
    harvest_and_store_sales(df, existing)

    logging.info("Done!")


if __name__ == "__main__":
    main()