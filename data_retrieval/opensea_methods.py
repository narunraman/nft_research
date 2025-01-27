
from queue import Empty
from data_retrieval.parsing_helpers import parse_sale_data, parse_nft_list, parse_listing_data
import time
from datetime import datetime
from tqdm import tqdm
import data_retrieval.request_helpers as req
from data_retrieval.psql_methods import execute_commands,batch_insert

API_KEYS = ("c113e12504b14e0185b714dcd72d6110", "55544646a70c491c80991e0666e7dbf6","507741952944434a9234438b7707b358")
SKIP_LIST=['ens','base-introduced','fundrop-pass','gemesis','apecoin','dai-stablecoin','uniswap','1inch-token','rarible','emblem-vault','sewerpass','clonex-mintvial','sorare']



def pull_nft_stats(nft_list,API_KEY = API_KEYS[1]):
    response_list = []
    for nft in tqdm(nft_list):
        response = req.Stats_api_request(nft, API_KEY)
        try:
            response_dict = dict(response.json()['total'])
        except:
            continue
        response_dict['slug']=nft
        timestamp = datetime.now()
        response_dict['_id'] = {'slug':nft,'timestamp':timestamp}
        response_list.append(response_dict)
    return response_list


def pull_nft_images(slug,API_KEY = API_KEYS[0],limit_toks=10000):
    next_cur = ''
    data_list = []
    while next_cur is not None:
        response = req.image_api_request(slug,next_cur, API_KEY)
        next_cur = response.json().get('next',None)
        try:
            for record in response.json()['nfts']:
                response_dict = dict(record)
                if response_dict['image_url'] is None:
                    continue
                data = (slug,int(response_dict['identifier']),response_dict['image_url'])
                data_list.append(data)
            if len(data_list)>limit_toks:
                next_cur = None
        except:
            return None
        return data_list

def pull_nft_contracts(slug,API_KEY = API_KEYS[0]):
    next_cur = ''
    response = req.contract_api_request(slug,next_cur, API_KEY)
    try:
        response_dict = dict(response.json())
    except:
        print(f'{slug} failed to find contract')
        return None
    try:
        data = (slug,response_dict['contracts'][0]['address'])
    except:
        data = (slug,None)
    return data

def pull_nft_types(slug,API_KEY = API_KEYS[0]):
    response = req.contract_api_request(slug,next_cur, API_KEY)
    try:
        response_dict = dict(response.json())
    except:
        print(f'{slug} failed to find contract')
        return None
    try:
        data = (slug,response_dict['category'])
    except:
        data = (slug,None)
    return data

def pull_nft_dates(slug,no_save=False,API_KEY = API_KEYS[0]):
    next_cur = ''
    response = req.contract_api_request(slug,next_cur, API_KEY)
    try:
        response_dict = dict(response.json())
    except:
        print(f'{slug} failed to find contract')
        return None
    try:
        data = (slug,response_dict["created_date"])
    except:
        data = (slug,None)
    return data
        


def pull_sales_data(collection_slug,API_KEY = API_KEYS[0],before=None,after=None):
    counter = 0
    next_curr = ""
    total_sales = []
    max_time = int(time.time() * 1_000_000)
    while next_curr is not None:
        counter+=1
        response = req.sale_event_api_request(collection_slug,next=next_curr,API_KEY=API_KEY,before=before,after=after)
        if response is not None:
            #Get sales
            batch_sales = response.json()['asset_events']
            #Get cursor for next batch of sales
            next_curr = response.json()['next']
            if batch_sales == []:
                return total_sales
            #Parsing sales data
            parsed_sales = [tuple(parse_sale_data(sale).values()) for sale in batch_sales]
            #storing parsed data into MongoDB
            if parsed_sales[0][4]>=max_time:
                return total_sales
            else:
                max_time=parsed_sales[0][4]
            if counter%50==0:
                print(parsed_sales[0])
                print(datetime.utcfromtimestamp(parsed_sales[0][4]))
            total_sales += parsed_sales
        else:
            return total_sales
    if next_curr is None:
        return total_sales

def pull_nft_events(collection_contract,token,query_size = 50,API_KEY = API_KEYS[0]):
    total_sales = []
    response = req.nft_event_api_request(collection_contract,token,next="",API_KEY=API_KEY)
    if response is not None:
        #Get sales
        batch_sales = response.json()['asset_events']
        #Get cursor for next batch of sales
        if batch_sales == []:
            return total_sales
        #Parsing sales data
        parsed_sales = [tuple(parse_sale_data(sale).values()) for sale in batch_sales]
        #storing parsed data into MongoDB
        return parsed_sales

def pull_nft_rarity(collection_contract,token,API_KEY = API_KEYS[0]):
    response = req.nft_rarity_api_request(collection_contract,token,API_KEY=API_KEY)
    if response is not None:
        #Get sales
        rarity = response.json()['nft']['rarity']
        return rarity
        
def pull_wallet_sales_data(account_address,query_size = 50,API_KEY = API_KEYS[0],saved=False):
    counter = 0
    next_curr = ""
    total_sales = []
    max_time = int(time.time() * 1_000_000)
    command = "Insert into wall_sales (wallet, token_id, seller, buyer, timestamp, sale_price,payment_token,transaction) values (%s, %s, %s, %s, %s, %s, %s, %s)"
    while next_curr is not None:
        counter+=1
        response = req.wallet_event_api_request(account_address,next=next_curr,API_KEY=API_KEY)
        if response is not None:
            #Get sales
            batch_sales = response.json()['asset_events']
            #Get cursor for next batch of sales
            next_curr = response.json()['next']
            if batch_sales == []:
                return total_sales
            #Parsing sales data
            parsed_sales = [tuple(parse_sale_data(sale).values()) for sale in batch_sales]
            #storing parsed data into MongoDB
            if parsed_sales[0][4]>max_time:
                return total_sales
            else:
                max_time=parsed_sales[0][4]
            if counter%50==0:
                print(parsed_sales[0])
                print(datetime.utcfromtimestamp(parsed_sales[0][4]))
            total_sales += parsed_sales
        else:
            return total_sales
    if next_curr is None:
        return total_sales








        
        
        
    
