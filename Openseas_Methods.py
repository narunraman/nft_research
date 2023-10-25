
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
import urllib

API_KEYS = ("c113e12504b14e0185b714dcd72d6110", "55544646a70c491c80991e0666e7dbf6")

def event_api_request(event_type=None,collection_slug=None,account_address=None,next="", API_KEY =API_KEYS[0],after=None,before=None):
    no_response = True
    while(no_response):
        url = "https://api.opensea.io/api/v1/events"
        querystring = {
                    "only_opensea":"true",
                    "cursor":str(next)
                    }
        if event_type:
            querystring['event_type'] = event_type
        if collection_slug:
            querystring['collection_slug'] = collection_slug
        if account_address:
            querystring['account_address'] = account_address
        if after:
            querystring['occurred_after'] = after.strftime("%Y-%m-%d %H:%M:%S")
        if before:
            querystring['occurred_before'] = before.strftime("%Y-%m-%d %H:%M:%S")
        headers = {"Accept": "application/json",
                    "X-API-KEY": API_KEY}
        params =  urllib.parse.urlencode(querystring, quote_via=urllib.parse.quote)
        response = requests.request("GET", url, headers=headers, params=params)
        if response.status_code == 429:
            print('error')
            print(response)
            time.sleep(5)
        elif response.status_code!=200:
            print('error')
            print(response)
            return None
        else:
            no_response =False
    return response

def active_listing_api_request(collection_slug=None, next="",  API_KEY=API_KEYS[0]):
    no_response= True
    while(no_response):
        url = f'https://api.opensea.io/v2/listings/collection/{collection_slug}/all'
        querystring = {"cursor":str(next)}
        headers = {"Accept": "application/json", "X-API-KEY": API_KEY}
        response = requests.request("GET", url, headers = headers, params = querystring)
        if response.status_code == 429:
            print('error')
            print(response)
            time.sleep(5)
        elif response.status_code != 200:
            print('error')
            print(response)
            return None
        else:
            no_response = False
    return response

def Stats_api_request(collection, API_KEY = API_KEYS[0]):
    no_response = True
    while(no_response):
        url = f"https://api.opensea.io/api/v1/collection/{collection}/stats"
        headers = {
        "accept": "application/json",
        "X-API-KEY": API_KEY
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 429:
            print('error')
            print(response)
            time.sleep(3)
        elif response.status_code!=200:
            print('error')
            print(response)
            return None
        else:
            no_response =False
    return response



    
def NFT_api_request(address, next_curr = "", API_KEY = API_KEYS[0],timeout=3):
    no_response = True
    while(no_response):
        url = f"https://api.opensea.io/v2/chain/ethereum/account/{address}/nfts"
        headers = {
        "accept": "application/json",
        "X-API-KEY": API_KEY
        }
        querystring = {"next":str(next_curr)}
        response = requests.get(url, headers=headers,params=querystring)
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

def pull_listing_data(collection_slug=None, total_queries=100000, query_size=100, dbname='NFTDB', API_KEY=API_KEYS[0]):
    client = MongoClient()
    db = client[dbname]
    listing_collection = db.listingsCollection
    # complete_adds = db.completedAddressesListings
    complete_slugs = db.completedSlugsListings
    next_curr = ''
    if collection_slug:
        if complete_slugs.count_documents({'_id':collection_slug})>0:
            print("Completed Slug in database")
            return None
    while next_curr is not None:
        response = event_api_request(event_type='created',collection_slug=collection_slug,next=next_curr,API_KEY=API_KEY)
        if response is not None:
            # Get listings
            batch_listings = response.json()['asset_events']
            # Get cursor for next batch of listings
            next_curr = response.json()['next']
            if batch_listings == []:
                if collection_slug is not None:
                    complete_slugs.insert_one({'_id':collection_slug})
                    return None
                break
            
            # Parsing listing data
            parsed_listings = [parse_listing_data(sale) for sale in batch_listings]
            
            # Storing parsed data into MongoDB
            for parsed_listing in parsed_listings:
                # listing_collection.update_one({"$set": parsed_listing}, upsert=True)
                listing_collection.update_one({'_id': {'id': parsed_listing['id'],'token_id':parsed_listing['token_id'],'timestamp':parsed_listing['timestamp']}}, {"$set": parsed_listing}, upsert=True)
            time.sleep(0.5)
    if next_curr is None:
        if collection_slug is not None:
            complete_slugs.insert_one({'_id':collection_slug})
            return None
            

def pull_nft_stats(nft_list,dbname='NFTDB',API_KEY = API_KEYS[0]):
    client = MongoClient()
    db = client[dbname]
    stat_collection = db.NFTStats
    for nft in tqdm(nft_list):
        response = Stats_api_request(nft, API_KEY)
        response_dict = dict(response.json()['stats'])
        response_dict['slug']=nft
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
        response_dict['_id'] = {'slug':nft,'timestamp':timestamp}
        stat_collection.insert_one(response_dict)
        
def edge_counts_to_percent(edge_list,minimum =True,dbname='NFTDB'):
    client = MongoClient()
    db = client[dbname]
    stat_collection = db.NFTStats
    percent_weights = {}
    for edge in edge_list:
        NFT1 = edge[0][0]
        NFT2 = edge[0][1]
        try:
            num_owners1 = stat_collection.find_one({'slug':NFT1})["num_owners"]
            num_owners2 = stat_collection.find_one({'slug':NFT2})["num_owners"]
        except:
            continue
        if num_owners1 in [None,0]  or num_owners2 in [None,0]:
            continue
        if minimum:
            percent_weight = edge[1]/min(num_owners1,num_owners2)
        else:
            percent_weight = edge[1]/max(num_owners1,num_owners2)
        percent_weights[(NFT1,NFT2)] = percent_weight
    return percent_weights

def pull_sales_data(collection_slug=None,account_address=None, total_queries=100000,query_size = 50,event_type=None,dbname='NFTDB', API_KEY = API_KEYS[0],before=None,after=None):
    client = MongoClient()
    db = client[dbname]
    counter = 0
    if account_address:
        destination = db.addressSalesCollection
        complete_adds = db.completedAddresses
    elif event_type=='successful':
        destination = db.salesCollection
        complete_adds = db.completedAddresses
        complete_slugs = db.completedSlugs
        if before:
            complete_slugs = db.completedSlugsHistoric
    elif event_type=='transfer':
        destination = db.transferCollection
        complete_adds = db.completedAddressesTransfers
        complete_slugs = db.completedSlugsTransfers
    next_curr = ""
    if collection_slug:
        if complete_slugs.count_documents({'_id':collection_slug})>0:
            print("Completed Slug in database")
            return None
    if account_address:
        if complete_adds.count_documents({'_id':account_address})>0:
            print("Address already in database")
            return None
    while next_curr is not None:
        counter+=1
        response = event_api_request(event_type,collection_slug,account_address,next_curr,API_KEY=API_KEY,before=before,after=after)
        if response is not None:
            #Get sales
            batch_sales = response.json()['asset_events']
            #Get cursor for next batch of sales
            next_curr = response.json()['next']
            if batch_sales == []:
                if account_address is not None:
                    complete_adds.insert_one({'_id':account_address})
                    return None
                elif collection_slug is not None:
                    complete_slugs.insert_one({'_id':collection_slug})
                    return None
                break
            #Parsing sales data
            parsed_sales = [parse_sale_data(sale) for sale in batch_sales]
            #storing parsed data into MongoDB
            if counter%10==0:
                print(parsed_sales[0]['timestamp'])
                print("seller",parsed_sales[0]['seller_address'])
                print("buyer",parsed_sales[0]['buyer_address'])
            for parsed_sale in parsed_sales:
                destination.update_one({'_id': parsed_sale['_id']}, {"$set": parsed_sale}, upsert=True)
            time.sleep(0.5)
        else:
            return None
    if next_curr is None:
        if account_address is not None:
            complete_adds.insert_one({'_id':account_address})
            return None
        elif collection_slug is not None:
            complete_slugs.insert_one({'_id':collection_slug})
            return None
            
            
def collect_slugs(start_time = None,total_queries=1000):
    client = MongoClient()
    db = client.NFTDB
    slug_data = db.slugCollection
    url = 'https://api.opensea.io/v2/orders/ethereum/seaport/listings'
    next_cur = ""
    for i in range(0, total_queries):
        querystring = {
                    "cursor":str(next_cur)
                    }
        headers = {"Accept": "application/json",
                    "X-API-KEY": API_KEYS[0]}

        response = requests.request("GET", url, headers=headers, params=querystring)
        if response.status_code == 429:
            print('error')
            print(response)
            time.sleep(5)
        elif response.status_code!=200:
            print('error')
            print(response)
            break
        else:
            json_formatted_str = json.dumps(response.json(), indent=2)
#             print(json_formatted_str)
            #Getting sales data
            batch_listings = response.json()['orders']
            #Get cursor for next batch of sales
            next_cur = response.json()['next']
            if next_cur is None:
                print("End of Transaction History")
                break
            if batch_listings == []:
                print('empty batch')
                break
            #Parsing sales data
            try:
                parsed_slugs = [{'_id':listing['maker_asset_bundle']['assets'][0]['collection']['slug']} for listing in batch_listings]
            except:
                continue
            #storing parsed data into MongoDB
            print(parsed_slugs)
            for slug in parsed_slugs:
                slug_data.update_one(slug, {"$set": slug}, upsert=True)
            time.sleep(0.5)

def make_graph(documents):
    edge_dict = defaultdict(int)
    G = nx.DiGraph()
    addresses = set()
    for doc in documents:
        buy_address = doc['buyer_address']
        sell_address = doc['seller_address']
        if buy_address==None or sell_address==None:
            continue
        if buy_address not in addresses:
            G.add_node(buy_address)
            addresses.add(buy_address)
        if sell_address not in addresses:
            G.add_node(sell_address)
            addresses.add(sell_address)
#         link = (buy_address,sell_address)
#         link_sort = tuple(sorted(link))
#         edge_dict[link_sort] +=1
#     for key,val in edge_dict.items():
        G.add_edge(sell_address, buy_address,weight=1)
    return G

def make_nft_graph(documents,weight_div=1,skip_list=[],min_owners=0,dbname='NFTDB'):
    #Expects a list where each element is a list with a tuple of 2 NFTs and int representing number of wallets
    G = nx.Graph()
    client = MongoClient()
    db = client[dbname]
    stat_collection = db.NFTStats
    for doc in documents:
        NFT1 = doc[0][0]
        NFT2 = doc[0][1]
        if NFT1 in skip_list or NFT2 in skip_list:
            continue
        if min_owners>0:
            num_owners1 = stat_collection.find_one({'slug':NFT1})["num_owners"]
            num_owners2 = stat_collection.find_one({'slug':NFT2})["num_owners"]
            if num_owners1<min_owners or num_owners2<min_owners:
                continue
        G.add_node(NFT1)
        G.add_node(NFT2)
        G.add_edge(NFT1, NFT2,weight=doc[1]/weight_div)
    return G

def slug_to_trade_graph(slug):
    client = MongoClient()
    db = client.NFTDB
    events =db.salesCollection
    documents = list(events.find({'slug':slug}))
    G = make_graph(documents)
    return G



def address_to_trade_graph(address,split=False):
    client = MongoClient()
    db = client.NFTDB
    events =db.salesCollection
    documents1 = list(events.find({'seller_address':address}))
    if split:
        documents2= list(events.find({'buyer_address':address}))
    else:
        documents1+= list(events.find({'buyer_address':address}))
    G = make_graph(documents1)
    if not split:
        return G
    else:
        G2 = make_graph(documents2)
        return (G,G2)
    


def addresses_to_nft_graph(addresses):
    client = MongoClient()
    db = client.NFTDB
    events =db.ownerCollection
    for address in addresses:
        documents = list(events.find({'address':address}))
        edge_dict = defaultdict(int)
        G = nx.Graph()
        addresses = set()
        for doc in documents:
            buy_address = doc['buyer_address']
            sell_address = doc['seller_address']
            if buy_address not in addresses:
                G.add_node(buy_address)
                addresses.add(buy_address)
            if sell_address not in addresses:
                G.add_node(sell_address)
                addresses.add(sell_address)
            link = (buy_address,sell_address)
            link_sort = tuple(sorted(link))
            edge_dict[link_sort] +=1
        for key,val in edge_dict.items():
            G.add_edge(key[0], key[1], weight=val)
    return G

def get_addresses_from_slug(slug):
    client = MongoClient()
    db = client.NFTDB
    events =db.salesCollection
    documents = list(events.find({'slug':slug}))
    addresses = set()
    for doc in documents:
        buy_address = doc['buyer_address']
        sell_address = doc['seller_address']
        if buy_address not in addresses:
            addresses.add(buy_address)
        if sell_address not in addresses:
            addresses.add(sell_address)
    return addresses

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
            flat_nfts = [item for sublist in nfts for item in sublist]
            destination.insert_one(parse_nft_list(flat_nfts,address))
        
        
        
        

def person_detector(address):
#A few options, is the goal to cluster wallets that belong to the same person or to just flag wallets that are likely bots?
    return 0
