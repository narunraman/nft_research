import requests
import urllib
import time
API_KEYS = ("c113e12504b14e0185b714dcd72d6110", "55544646a70c491c80991e0666e7dbf6","507741952944434a9234438b7707b358")
def response_handler(url,querystring=None,headers=None):
    no_response = True
    while(no_response):
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
def sale_event_api_request(collection_slug,next="", API_KEY =API_KEYS[0],after=None,before=None):
    url = f"https://api.opensea.io/api/v2/events/collection/{collection_slug}"
    querystring = {
                "only_opensea":"true",
                "next":str(next)
                }
    querystring['event_type'] = "sale"
    if after:
        querystring['occurred_after'] = after.strftime("%Y-%m-%d %H:%M:%S")
    if before:
        querystring['occurred_before'] = before.strftime("%Y-%m-%d %H:%M:%S")
    headers = {"Accept": "application/json",
                "X-API-KEY": API_KEY}
    return response_handler(url,querystring,headers)

def wallet_event_api_request(account_address,next="", API_KEY =API_KEYS[0]):
    url = f"https://api.opensea.io/api/v2/events/accounts/{account_address}?event_type=transfer&event_type=sale"
    querystring = {
                "only_opensea":"true",
                "next":str(next)
                }
    headers = {"Accept": "application/json",
                "X-API-KEY": API_KEY}
    return response_handler(url,querystring,headers)

def nft_event_api_request(contract,token,next="", API_KEY =API_KEYS[0]):
    url = f"https://api.opensea.io/api/v2/events/chain/ethereum/contract/{contract}/nfts/{token}?event_type=transfer&event_type=sale"
    querystring = {
                "only_opensea":"true",
                "next":str(next)
                }
    headers = {"Accept": "application/json",
                "X-API-KEY": API_KEY}
    return response_handler(url,querystring,headers)
    
def active_listing_api_request(collection_slug=None, next="",  API_KEY=API_KEYS[0]):
    url = f'https://api.opensea.io/v2/listings/collection/{collection_slug}/all'
    querystring = {"cursor":str(next)}
    headers = {"Accept": "application/json", "X-API-KEY": API_KEY}
    return response_handler(url,querystring,headers)

def Stats_api_request(collection, API_KEY = API_KEYS[0],verbose=False):
    url = f"https://api.opensea.io/api/v2/collections/{collection}/stats"
    headers = {
    "accept": "application/json",
    "X-API-KEY": API_KEY
    }
    return response_handler(url,headers=headers)

def nft_rarity_api_request(contract,token_id, API_KEY = API_KEYS[0],verbose=True):
    url = f"https://api.opensea.io/api/v2/chain/ethereum/contract/{contract}/nfts/{token_id}"
    headers = {
    "accept": "application/json",
    "X-API-KEY": API_KEY
    }
    return response_handler(url,headers=headers)
    
def image_api_request(collection,next_curr, API_KEY = API_KEYS[0]):
    url = f"https://api.opensea.io/api/v2/collection/{collection}/nfts"
    headers = {
    "accept": "application/json",
    "X-API-KEY": API_KEY
    }
    querystring = {"next":str(next_curr),"limit":200}
    return response_handler(url,querystring,headers)

def contract_api_request(collection,next_curr=None, API_KEY = API_KEYS[0]):
    url = f"https://api.opensea.io/api/v2/collections/{collection}"
    headers = {
    "accept": "application/json",
    "X-API-KEY": API_KEY
    }
    querystring = {"next":str(next_curr),"limit":200}
    return response_handler(url,querystring,headers)


    
def wall_to_NFT_api_request(address, next_curr = "", API_KEY = API_KEYS[0],timeout=3):
    url = f"https://api.opensea.io/v2/chain/ethereum/account/{address}/nfts"
    headers = {
    "accept": "application/json",
    "X-API-KEY": API_KEY
    }
    querystring = {"next":str(next_curr)}
    return response_handler(url,querystring,headers)

def NFT_to_wall_api_request(address, next_curr = "", API_KEY = API_KEYS[0],timeout=3):
    url = f"https://api.opensea.io/v2/chain/ethereum/account/{address}/nfts"
    headers = {
    "accept": "application/json",
    "X-API-KEY": API_KEY
    }
    querystring = {"next":str(next_curr)}
    return response_handler(url,querystring,headers)
