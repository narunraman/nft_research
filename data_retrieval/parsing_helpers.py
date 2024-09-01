import datetime

def safeget(dct, keys):
    for key in keys:
        try:
            dct = dct[key]
        except:
            return None
    return dct

def parse_listing_data(listing_dict):
    result = {}
    event_data = {
        'username': ['from_account', 'user', 'username'],
        'lister_address': ['from_account', 'user', 'address'], # address of the account that listed
        
        'id': ['id'],
        'listing_contract_address': ['contract_address'], # check if this is real
        'timestamp': ['event_timestamp'],
        'created_date': ['created_date'],
        'duration': ['duration'],
        'auction_type': ['auction_type'],
        'starting_price': ['starting_price'],
        'ending_price': ['ending_price'],
        'private': ['is_private'],
        'owner_account': ['owner_account'],
        
        'token_id': ['asset', 'token_id'], # token id of the NFT
        'asset_id': ['asset', 'id'],
        'num_sales': ['asset', 'num_sales'],
        'asset_owner': ['asset', 'owner'], # deprecated but new 'top_ownerships' not found in asset object
        
        'total_supply': ['asset', 'asset_contract', 'total_supply'], # check if this is real
        'asset_contract': ['asset', 'asset_contract', 'address'], # on chain address of the contract
        'asset_contract_owner': ['asset', 'asset_contract', 'owner'],
        'seller_fees': ['asset', 'collection', 'fees', 'seller_fees'],
        
        'opensea_fees': ['asset', 'collection', 'fees', 'opensea_fees'],
        'slug': ['asset', 'collection', 'slug']
    }

    # for event in listing_dict:
    for key in event_data:
        result[key] = safeget(listing_dict, event_data[key])
        
    if result['seller_fees']: # {address: fee, address: fee} dictionary -> [[address, fee], ...]
        result['seller_fees'] = [[key, val] for key, val in result['seller_fees'].items()] 
    if result['opensea_fees']:
        result['opensea_fees'] = [[key, val] for key, val in result['opensea_fees'].items()]
            
    return result


def parse_active_listing_data(listing_dict):
    result = {}
    listing_keys = {
        '_id': ['order_hash'],
        'signature': ['protocol_data', 'signature'],
        'chain': ['chain'], 
        'auction_type': ['type'],
        'price': ['price', 'current', 'value'],
        'currency': ['price', 'current', 'currency'],
        'decimals': ['price', 'current', 'decimals']
    }
    protocol_keys = {
        'offerer': ['offerer'],
        'start_time': ['startTime'],
        'end_time': ['endTime'],
        'secondary_account': ['zone'],
        'start_amount': ['offer', 0, 'startAmount'],
        'end_amount': ['offer', 0, 'endAmount'],
        'offer_token': ['offer', 0, 'token'],
        'offerer_fee': ['consideration', 0, 'endAmount'],
        'opensea_fee': ['consideration', 1, 'endAmount'],
        'collection_fee': ['consideration', 2, 'endAmount']
    }
    
    for listing in listing_dict:
        try:
            protocol_data = listing['protocol_data']['parameters']
        except:
            continue
        
        for key in listing_keys:
            result[key] = safeget(listing, listing_keys[key])
        for key in protocol_keys: 
            result[key] = safeget(protocol_data, protocol_keys[key])
    return result
    
def parse_sale_data(sale_dict):
    event_type = sale_dict['event_type']
    if event_type=="sale":
        if sale_dict['nft'] != None:
            slug = sale_dict['nft']['collection']
            token_id = sale_dict['nft']['identifier']
        else:
            token_id=None
            slug =None
        seller_address = sale_dict.get('seller',None)
        buyer_address = sale_dict.get('buyer',None)
        timestamp = sale_dict.get('event_timestamp',None)
        transaction_hash = sale_dict.get('transaction',None)
        try:
            total_price = float(sale_dict['payment']['quantity'])*10**-18
        except:
            total_price = None
        try:
            payment_token = sale_dict['payment']['symbol']
        except:
            payment_token = None
            
        result = {
                  'slug':slug,
                  'token_id': token_id,
                  'seller_address': seller_address,
                  'buyer_address': buyer_address,
                  'timestamp': timestamp,
                  'total_price': total_price, 
                  'payment_token': payment_token,
                  'id': transaction_hash}
        
    elif event_type=="transfer":
        if sale_dict['nft'] != None:
            slug = sale_dict['nft']['collection']
            token_id = sale_dict['nft']['identifier']
        else:
            token_id=None
            slug =None
        seller_address = sale_dict.get('from_address',None)
        buyer_address = sale_dict.get('to_address',None)
        timestamp = sale_dict.get('event_timestamp',None)
        transaction_hash = sale_dict.get('transaction',None)

        result = {
                  'slug':slug,
                  'token_id': token_id,
                  'seller_address': seller_address,
                  'buyer_address': buyer_address,
                  'timestamp': timestamp,
                  'total_price': None, 
                  'payment_token': None,
                  'id': transaction_hash}
            
    return result

def parse_nft_list(nft_list,wallet):
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
    nft_set = set()
    nft_with_token = []
    for nft in nft_list:
        nft_set.add(nft['collection'])
        nft_with_token+= (nft['collection'],nft['identifier'])
    result = {'address':wallet,'timestamp':timestamp,'NFTs':list(nft_set),'NFTandToken':nft_with_token}
    return result

# def parse
        
    