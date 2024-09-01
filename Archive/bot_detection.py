from tqdm import tqdm
import pandas as pd
import pymongo
from pymongo import MongoClient
import seaborn as sns
import requests
import pandas as pd


def add_addresses(df):
    #adds timestamp field
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace = True)
    #splits buy and sell into two dbs, then merges on timestamp and finds unique address
    df_address = pd.DataFrame()
    
    df_address['transaction_id'] = list(df['_id']) * 2
    df_address['address'] = list(df['buyer_address']) + list(df['seller_address'])
    df_address['timestamp'] = list(df['timestamp']) * 2
    df_address['timestamp_min'] = list(df['timestamp'].apply(lambda x: x.replace(second=0))) * 2
    df_address['slug'] = list(df['slug']) * 2
    df_address['token_id'] = list(df['token_id']) * 2
    df_address['type'] = ['buy'] * len(list(df['_id'])) + ['sell'] * len(list(df['_id']))
    return df_address


def gen_bot_df(df):
    def max_in_group(grouped_df, transaction_types):
        max_elm = 1
        try:
            for name, group in grouped_df:
                if len(group) > max_elm and transaction_types[0] in group['type'].unique() and transaction_types[1] in group['type'].unique():
                    max_elm = len(group)
        except:
            print(grouped_df)
        return max_elm
        
    destination = db.address_data
    addresses = df.groupby('address')
    for address, address_group in addresses:

        # counts number of transactions for an address
        num_transactions = int(address_group.count()[0])
        
        # for an address groups all buy and sell transactions within the same second: 
        # group is (address, timestamp)
        trans_sec = address_group.groupby('timestamp')
        max_trans_sec = max_in_group(trans_sec, ['buy', 'sell'])
        # max_trans_sec = 1
        # for name, group in trans_sec:
        #     if len(group) > max_trans_sec and 'buy' in group['type'].unique() and 'sell' in group['type'].unique():
        #         max_trans_sec = len(group)

        # for an address groups all buy and sell transactions for the same nft within the same second: 
        # group is (address, slug, token_id, timestamp)
        trans_sec_nft = address_group.groupby(['slug', 'token_id', 'timestamp'])
        max_trans_sec_nft = max_in_group(trans_sec_nft, ['buy', 'sell'])

        # for an address groups all buy and list actions for the same nft within the same second: 
        # group is (address, slug, token_id, timestamp)
        buy_list_sec_nft = address_group.groupby(['slug', 'token_id', 'timestamp'])
        max_buy_list_sec_nft = max_in_group(buy_list_sec_nft, ['buy', 'list'])

        
        trans_min = address_group.groupby('timestamp_min')
        max_trans_min = max_in_group(trans_sec, ['buy', 'sell'])

        trans_min_nft = address_group.groupby(['slug', 'token_id', 'timestamp_min'])
        max_trans_min_nft = max_in_group(trans_min_nft, ['buy', 'sell'])
        
        
        row = {
            '_id': address,
            'num_transactions': num_transactions,
            'max_trans_sec': max_trans_sec,
            'max_trans_sec_nft': max_trans_sec_nft,
            'max_buy_list_sec_nft': max_buy_list_sec_nft,
            'max_trans_min': max_trans_min,
            'max_trans_min_nft': max_trans_min_nft
        }
        destination.update_one({'_id': row['_id']}, {"$set": row}, upsert=True)



def main():
    client = MongoClient()
    #Pulls all sales converts to pandas
    db = client.NFTDB
    sales = db.salesCollection
    test = db.testdb
    record = sales.find({})
    df = pd.DataFrame.from_records(record)
    address_df = add_addresses(pd.DataFrame.from_records(db.salesCollection.find({})))
    address_df[['address', 'timestamp', 'timestamp_min', 'slug', 'token_id', 'type']]
    gen_bot_df(address_df)


if __name__ == "__main__":
    main()