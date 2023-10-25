
from queue import Empty
import requests
from pymongo import MongoClient
from parsing_helpers import parse_sale_data, parse_nft_list, parse_listing_data
import time
import json
import networkx as nx
from collections import defaultdict
import datetime
from tqdm import tqdm

API_KEYS = ("-dkhE3R5RIFr9b9KUhpp4pqlQFzpPFzW")



    
def NFT_api_request(address, next_curr = "", API_KEY = "fa400fba3fd44574b659bb0372b8b5d9",timeout=3):
    no_response = True
    while(no_response):
        url = f"https://eth-mainnet.g.alchemy.com/nft/v3/-dkhE3R5RIFr9b9KUhpp4pqlQFzpPFzW/getNFTsForOwner?owner={address}&withMetadata=true&pageKey={next_curr}&pageSize=100"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        if response.status_code == 429:
            print('error')
            print(response)
            time.sleep(timeout)
        elif response.status_code!=200:
            print('error')
            print(response)
            return None
        else:
            no_response =False
    return response

def find_all_nfts_for_addresses(addresses,dbname='NFTDB',API_KEY = "fa400fba3fd44574b659bb0372b8b5d9"):
    #Find all NFTs currently held by an address and store this list and current timestamp to DB
    client = MongoClient()
    db = client[dbname]
    destination = db.addresstoNFT
    for address in tqdm(addresses):
        if destination.count_documents({'address':address})>0:
            continue
        nfts = []
        next_curr = ''
        while next_curr is not None:
            response = NFT_api_request(address,API_KEY=API_KEY,next_curr=next_curr)
            if response is not None:
                try:
                    nfts.append(response.json()['nfts'])
                except KeyError:
                    print(response.json())
                try:
                    next_curr = response.json()['next']
                except:
                    next_curr = None
            else:
                next_curr = None
        if nfts:
            destination.insert_one(parse_alchemy_nfts(response['ownedNfts'],address))
        
        
        
        

def person_detector(address):
#A few options, is the goal to cluster wallets that belong to the same person or to just flag wallets that are likely bots?
    return 0
