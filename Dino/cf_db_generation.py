import feature_extract
import sys
sys.path.append("..")
import torch
import psql_methods as psql
import pickle
import alchemy_methods as alc
from tqdm import tqdm
import numpy as np
import image_utils as imgs
import opensea_methods as opse
import multiprocessing
import pandas as pd
import feature_utils as feat
import logging
#FILE used was top_100_pfps.txt
def format_slugs(slug_text_file):
    f = open(slug_text_file,'r')
    slugs = list(f)
    slugs = [x.strip('\n') for x in slugs]
    nfts_to_process = [x for x in nfts_to_process if x not in opse.SKIP_LIST]

def pull_urls(slugs,logging_file = 'counterfeit_url_logs.txt'):
    logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    for slug in tqdm(slugs):
        logging.info(f"Beginning slug {slug}")
        data.append(opse.pull_nft_images(slug,limit_toks=10000))
        logging.info(f"Finished slug {slug}")

#TODO this uses old databse structure
def save_urls(url_data):
    command = 'INSERT INTO nfttoimage (slug, token_id,url) VALUES (%s, %s, %s)"'
    psql.batch_insert(command,data_list)