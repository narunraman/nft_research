#!/usr/bin/env python
# coding: utf-8
"""
graph_dataset_to_images.py

End‑to‑end pipeline that turns an NFT‑graph dataset into a folder of images.


Prerequisites
-------------
Database tables
    - ``nfttoimage(slug, token_id, url)``             – image URLs
    - ``collectiontoaddress(slug, address)``          – contract addresses

Custom helper modules
    feature_extract, psql_methods, image_utils, opensea_methods, alchemy_methods

Run
---
$ python graph_dataset_to_images.py

Edit the ``CONFIG`` block below for your paths and model choice.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#  Import project‑specific helpers                                            #
# --------------------------------------------------------------------------- #
sys.path.append("..")

import feature_extract                       # type: ignore
import psql_methods as psql                  # type: ignore
import image_utils as imgs                   # type: ignore
from opensea_methods import (                # type: ignore
    pull_nft_contracts,
    pull_nft_images,
    pull_image_from_url,
)
import alchemy_methods as alc                # type: ignore

# --------------------------------------------------------------------------- #
#  Config                                                                     #
# --------------------------------------------------------------------------- #
CONFIG: Dict[str, str] = {
    # Pickle containing list[str] of NFT collection slugs present in the graph
    "label_pickle": "../Graph_predictions/dataset_stor/graph_dataset_4/label_list.pkl",

    # Where image folders live.  graph_images/val/<slug>/<token>.png
    "image_root": "../Dino/images_features/images/val",

    # For logging API progress
    "log_file": "slug_url_logs.txt",


    # Final artefacts
    "raw_feature_dataframe": "graph_images_features_dataframe.pkl",
    "final_dataframe": "graph_images_dataframe.pkl",
}

# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #

def load_label_list(pickle_path: str | Path) -> List[str]:
    """Load the master list of NFT collection slugs."""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def get_existing_slugs() -> List[str]:
    """Return slugs that already have at least one image URL recorded."""
    rows = psql.execute_commands(["SELECT DISTINCT slug FROM nfttoimage"])
    return [r[0] for r in rows]


def contracts_dict() -> Dict[str, str | None]:
    """Return mapping slug → address (address may be None)."""
    rows = psql.execute_commands(["SELECT slug, address FROM collectiontoaddress"])
    return {slug: addr for slug, addr in rows}


def ensure_contract_addresses(slugs: Sequence[str], mapping: Dict[str, str | None]) -> None:
    """Fetch contract addresses for any slug lacking one and (optionally) save to DB."""
    missing = [s for s in slugs if mapping.get(s) in (None, "")]
    if not missing:
        return

    for slug in tqdm(missing, desc="Fetching missing contract addresses"):
        try:
            addr = pull_nft_contracts(slug)        # external helper
            # OPTIONAL – insert into DB here
            mapping[slug] = addr
        except Exception as exc:                   
            logging.error("Contract pull failed for %s – %s", slug, exc)


def fetch_image_urls(slugs: Sequence[str], limit_per_slug: int = 500) -> None:
    """Call the OpenSea API for each slug and harvest up to limit_per_slug URLs."""
    for slug in tqdm(slugs, desc="Pulling NFT image URLs"):
        try:
            pull_nft_images(slug, limit_toks=limit_per_slug)
        except Exception as exc:                   # noqa: BLE001
            logging.error("Image‑URL fetch failed for %s – %s", slug, exc)


def get_immediate_subdirectories(directory: str | Path) -> List[str]:
    """Return list of first‑level subdirectory names."""
    return [d.name for d in Path(directory).iterdir() if d.is_dir()]


def select_rows_to_download(
    label_list: Sequence[str],
    image_root: str | Path,
    num_per_collection: int = 50,
) -> List[Tuple[str, int, str]]:
    """Pick num_per_collection random NFT rows per slug that still need images."""
    completed = set(get_immediate_subdirectories(image_root))
    pending_collections = tuple(slug for slug in label_list if slug not in completed)

    sql = f"""WITH numbered_rows AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY slug ORDER BY RANDOM()) AS row_num
        FROM nfttoimage
        WHERE slug IN {pending_collections}
    )
    SELECT slug, token_id, url
    FROM numbered_rows
    WHERE row_num <= {num_per_collection};"""

    rows = psql.execute_commands([sql])
    return [(slug, token_id, url) for slug, token_id, url in rows]   # type: ignore[misc]


def download_images(rows: List[Tuple[str, int, str]], image_root: str | Path) -> None:
    """Spawn n_cores workers that download each (slug, token_id, url) triple."""
    import multiprocessing as mp  # ensure picklability inside function

    df = pd.DataFrame(rows, columns=["slug", "token_id", "url"])
    grouped = df.groupby("slug").apply(
        lambda x: (x["slug"].iloc[0], list(zip(x["token_id"], x["url"])))
    )
    args = list(grouped)

    nproc = mp.cpu_count()
    with mp.Pool(processes=nproc) as pool:
        for _ in pool.starmap(pull_image_from_url, args):
            pass  # pull_image_from_url already handles its own side effects

    # Clean up empty dirs (failed downloads)
    imgs.delete_empty_directories(image_root)


# --------------------------------------------------------------------------- #
#  Feature extraction & aggregation                                           #
# --------------------------------------------------------------------------- #

def build_feature_dataframe(cfg: dict) -> pd.DataFrame:
    """Load pre‑computed DINOv2 features + metadata into a DataFrame."""
    model = cfg["model_string"]
    raw_root = cfg["raw_image_root"]
    feat_dir = Path(cfg["feature_dir"]) / model
    features_fp = feat_dir / cfg["features_filename"]

    features = torch.load(features_fp)
    labels = feature_extract.get_labels(raw_root)            # type: ignore[attr-defined]
    file_names = feature_extract.get_filenames(raw_root)     # type: ignore[attr-defined]

    data = {
        "Label": labels.tolist(),
        "Features": features.tolist(),
        "Collection": [fn[0] for fn in file_names],
        "NFT_num": [fn[1] for fn in file_names],
    }
    df = pd.DataFrame(data)
    df.to_pickle(cfg["raw_feature_dataframe"])
    return df


def add_average_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average feature vector per label and merge back into df."""
    grouped = df.groupby("Label")
    avg_vectors = [
        (label, np.mean(group["Features"].tolist(), axis=0))
        for label, group in tqdm(grouped, desc="Averaging features")
    ]
    avg_df = pd.DataFrame(avg_vectors, columns=["Label", "AverageFeatureVector"])
    merged = pd.merge(df, avg_df, on="Label")
    return merged


# --------------------------------------------------------------------------- #
#  Main driver                                                                
# --------------------------------------------------------------------------- #

def main() -> None:
    """Run the full pipeline."""
    # ------------------------------------------------------------------- #
    #  0. House‑keeping                                                  #
    # ------------------------------------------------------------------- #
    logging.basicConfig(
        filename=CONFIG["log_file"],
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
    label_list = load_label_list(CONFIG["label_pickle"])
    logging.info("Loaded %d collection slugs", len(label_list))

    # ------------------------------------------------------------------- #
    #  1. Identify work to do                                            #
    # ------------------------------------------------------------------- #
    existing_slugs = set(get_existing_slugs())
    slugs_needed = [s for s in label_list if s not in existing_slugs]
    logging.info("%d collections still need URLs", len(slugs_needed))

    # Contracts first
    contract_map = contracts_dict()
    ensure_contract_addresses(slugs_needed, contract_map)

    # ------------------------------------------------------------------- #
    #  2. Pull image URLs                                                #
    # ------------------------------------------------------------------- #
    fetch_image_urls(slugs_needed, limit_per_slug=500)

    # ------------------------------------------------------------------- #
    #  3. Select & download actual images                                #
    # ------------------------------------------------------------------- #
    rows_to_pull = select_rows_to_download(label_list, CONFIG["image_root"], 50)
    logging.info("Will download %d images", len(rows_to_pull))
    download_images(rows_to_pull, CONFIG["image_root"])




if __name__ == "__main__":  # pragma: no cover
    main()
