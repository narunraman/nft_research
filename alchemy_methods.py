
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
from Openseas_Methods import *
import logging


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

def labels_to_contracts(labels):
    contracts = []
    for label in tqdm(labels):
        contracts.append(pull_nft_contracts(label,no_save=True))
    return contracts
        
def contracts_to_owners(contracts,just_labels=False):
    #Expects a list of tuples of (slug,contract) or if just labels (slugs)
    label_to_owners = {}
    if just_labels:
        contracts = labels_to_contracts(contracts)
    for label,contract in tqdm(contracts):
        url = f"https://eth-mainnet.g.alchemy.com/nft/v3/-dkhE3R5RIFr9b9KUhpp4pqlQFzpPFzW/getOwnersForContract?contractAddress={contract}&withTokenBalances=false"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        try:
            label_to_owners[label] = response.json()['owners']
        except:
            print(f'Error on label {label}')
    return label_to_owners

def owners_to_NFT(wallets):
    #returns a list of (slug,contract)
    NFT_list = []
    for wall in tqdm(wallets):
        url = f"https://eth-mainnet.g.alchemy.com/nft/v3/-dkhE3R5RIFr9b9KUhpp4pqlQFzpPFzW/getNFTsForOwner?owner={wall}&withMetadata=true&pageSize=100"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        data = response.json()
        try:
            owned_nfts = data['ownedNfts']
        except:
            continue
        for nft in owned_nfts: 
            try:
                slug = nft['contract']['openSeaMetadata']['collectionSlug']
                contract = nft['contract']['address']
                NFT_list.append((slug,contract))

            except:
                continue
    return NFT_list

#Expects a list of tuples of contracts and token_ids
def NFT_to_sales(nfts):
    with_sales = []
    for contract,token_id in tqdm(nfts):
        if contract is None:
            continue
        url = f"https://eth-mainnet.g.alchemy.com/nft/v3/-dkhE3R5RIFr9b9KUhpp4pqlQFzpPFzW/getNFTSales?fromBlock=0&toBlock=latest&order=desc&contractAddress={contract}&tokenId={token_id}"
        
        headers = {"accept": "application/json"}
        
        response = requests.get(url, headers=headers)
        # print(response)
        data = response.json()
        try:
            sale_prices = []
            for sale in data["nftSales"]:
                sale_prices.append(float(sale['sellerFee']['amount'])*10**-18)
            avg_prices = sum(sale_prices)/len(sale_prices)
        except:
            if len(sale_prices)>1:
                avg_prices = sum(sale_prices)/len(sale_prices)
            else:
                continue
        with_sales.append((contract,token_id,avg_prices))
    return with_sales

def NFT_to_rarities(nfts):
    # Configure logging to write to a file
    logging.basicConfig(filename='error_log_alchemy.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    with_rarity = []
    failure_count=0
    dead_contracts = []
    for contract,token_id in tqdm(nfts):
        if contract is None or contract in dead_contracts:
            logging.error(f"Skipping {contract} and {token_id}")
            continue
        url = f"https://eth-mainnet.g.alchemy.com/nft/v3/-dkhE3R5RIFr9b9KUhpp4pqlQFzpPFzW/computeRarity?contractAddress={contract}&tokenId={token_id}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        # print(response)
        try:
            data = response.json()
            rarity = 1
            for sale in data["rarities"]:
                rarity*=float(sale["prevalence"])
        except Exception as e:
            failure_count+=1
            logging.error(f"An error occurred: {str(e)} while processing {contract} and {token_id}. Returned response {response}")
            dead_contracts.append(contract)
            if failure_count>1000:
                return with_rarity
            continue
        failure_count=0
        with_rarity.append((contract,token_id,rarity))
    return with_rarity


def person_detector(address):
#A few options, is the goal to cluster wallets that belong to the same person or to just flag wallets that are likely bots?
    return 0
