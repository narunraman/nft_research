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
    auction_type = sale_dict['auction_type']
    if event_type=="successful":
        is_bundle = False
        slug = sale_dict['collection_slug']
        if sale_dict['asset'] != None:
            token_id = sale_dict['asset']['token_id']
        elif sale_dict['asset_bundle'] != None:
            token_id = [asset['token_id'] for asset in sale_dict['asset_bundle']['assets']]
            is_bundle = True
        else:
            token_id=None
        try:
            seller_address = sale_dict['seller']['address']
        except:
            seller_address = None
        try:
            buyer_address = sale_dict['winner_account']['address']
        except:
            buyer_address = None
        try:
            seller_username = sale_dict['seller']['user']['username']
        except:
            seller_username = None    
        try:
            buyer_username = sale_dict['winner_account']['user']['username']
        except:
            buyer_username = None
        try:
            timestamp = sale_dict['transaction']['timestamp']
        except:
            timestamp = None
        try:
            total_price = float(sale_dict['total_price'])*10**-18
        except:
            total_price = None
        try: 
            bid_amount = float(sale_dict['total_price'])*10**-18
        except:
            bid_amount = None
        try:
            payment_token = sale_dict['payment_token']['symbol']
            usd_price = float(sale_dict['payment_token']['usd_price'])
        except:
            payment_token = None
            usd_price = None
        try:
            transaction_hash = sale_dict['transaction']['transaction_hash']
        except:
            transaction_hash = None
        


        result = {'is_bundle': is_bundle,
                  'slug':slug,
                  'token_id': token_id,
                  'seller_address': seller_address,
                  'buyer_address': buyer_address,
                  'buyer_username': buyer_username,
                  'seller_username':seller_username,
                  'timestamp': timestamp,
                  'total_price': total_price, 
                  'payment_token': payment_token,
                  'usd_price': usd_price,
                  '_id': transaction_hash,
                 'event_type': event_type,
                 'auction_type': auction_type}
    elif event_type=='transfer':
        is_bundle = False
        slug = sale_dict['collection_slug']
        if sale_dict['asset'] != None:
            token_id = sale_dict['asset']['token_id']
        elif sale_dict['asset_bundle'] != None:
            token_id = [asset['token_id'] for asset in sale_dict['asset_bundle']['assets']]
            is_bundle = True
        else:
            token_id=None
        try:
            timestamp = sale_dict['transaction']['timestamp']
        except:
            timestamp = None
        try:
            transaction_hash = sale_dict['transaction']['transaction_hash']
        except:
            transaction_hash = None
        try:
            seller_address = sale_dict['transaction']['from_account']['address']
        except:
            seller_address = None
        try:
            buyer_address = sale_dict['transaction']['to_account']['address']
        except:
            buyer_address = None
        try:
            seller_username = sale_dict['transaction']['from_account']['user']['username']
        except:
            seller_username = None    
        try:
            buyer_username = sale_dict['transaction']['to_account']['user']['username']
        except:
            buyer_username = None
        result = {'is_bundle': is_bundle,
                  'slug':slug,
                  'token_id': token_id,
                  'seller_address': seller_address,
                  'buyer_address': buyer_address,
                  'buyer_username': buyer_username,
                  'seller_username':seller_username,
                  'timestamp': timestamp,
                  '_id': transaction_hash,
                 'event_type': event_type,
                 'auction_type': auction_type}
            
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
        
    