from Openseas_Methods import pull_sales_data,make_graph
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import pygraphviz
import random
from pymongo import MongoClient
from tqdm import tqdm
from pymongo import UpdateOne
from itertools import combinations


client = MongoClient()
db =client.NFTDB
sales_data = db.salesCollection
transfers = db.transfersCollection
WallettoNFT = db.addresstoNFT

record = WallettoNFT.find({})
df = pd.DataFrame.from_records(record)


all_combos = defaultdict(lambda:0)
batch = len(df.NFTs) // 20
for i, nft in tqdm(enumerate(df.NFTs), position=0, leave=True):
    for comb in combinations(nft, r=2):
        if 'opensea-paymentassets' in comb or 'ens' in comb:
            continue
        else:
            sorted_comb = tuple(sorted(list(comb)))
            all_combos[str(sorted_comb)]+=1
    if (i % batch == 0 and i > 0) or i == len(df.NFTs) - 1:
        requests = []
        combos_filtered = {key:val for key, val in all_combos.items() if val >1}
        for key in tqdm(combos_filtered, desc='Generating Requests', position=0, leave=True):
            requests.append(UpdateOne({'_id': key}, { '$inc': { 'value': all_combos[key]}}, upsert=True))
        db.nftCombinations.bulk_write(requests)
        del requests
        print('Finished Writing')
        all_combos = defaultdict(lambda:0)