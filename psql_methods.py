#!/usr/bin/python
from configparser import ConfigParser
import psycopg2
import psycopg2.extras

def config(filename='/global/scratch/tlundy/NFT_Research/nft_research/database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db

def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
		
        # create a cursor
        cur = conn.cursor()
        
	# execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
       
	# close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def execute_commands(commands,data_list=None,no_return=False):
    conn = None
    rows= []
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        if data_list is None:
            for command in commands:
                cur.execute(command)
        else:
            for command,data in zip(commands,data_list):
                cur.execute(command,data)
        if not no_return:
            rows = cur.fetchall()
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
        return rows
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def batch_insert(command,data_list):
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        cur.executemany(command, data_list)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def batch_insert_fast(command,data_list):
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        psycopg2.extras.execute_values(cur, command, data_list,page_size=5000)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            
def get_psql_type(field, collection_name):
    if field == None:
        return "text"
    elif 'str' in field:
        return "text"
    elif 'bigger float' in field:
        return 'NUMERIC(40, 15)'
    elif 'big float'in field:
        return 'NUMERIC(30, 5)'
    elif 'float' in field:
        return 'NUMERIC(35, 20)'
    elif 'BIGGERINT' in field:
        return 'NUMERIC(80)'
    elif 'BIGINT' in field:
        return 'BIGINT'
    elif 'int' in field:
        return 'INTEGER'
    elif 'bool' in field:
        return 'BOOLEAN'
    elif 'ARRAY' in field:
        return 'text []'
    elif 'json' in field:
        return 'json'
    elif 'TIMESTAMP' in field:
        return 'TIMESTAMP'
    elif 'BYTEA' in field:
        return "VARCHAR(255)"
        # return 'BYTEA'

def fields_to_string(fields_dict, collection_name):
    return ', '.join(
        [field_name + f" {get_psql_type(fields_dict[field_name], collection_name)}" 
         if field_name != '_id' 
         else field_name + f" {get_psql_type(fields_dict[field_name], collection_name)} PRIMARY KEY" 
         for field_name in fields_dict]
    )

def create_empty_psql_tables_from_mongo(mongo_dict):
    """ create tables in the PostgreSQL database"""
    commands = []
    for collection_name in mongo_dict:
        if 'test' in collection_name:
            continue
        commands.append(
            f"""
            CREATE TABLE {collection_name} (
                {fields_to_string(mongo_dict[collection_name], collection_name)}
            )
            """)

    execute_commands(commands)
    
def create_table(table_name, fields):
    """ create tables in the PostgreSQL database"""
    field_statements = [f"{field_name} {data_type}" for field_name, data_type in fields.items()]
    fields_sql = ", ".join(field_statements)
    create_table_command = f"CREATE TABLE IF NOT EXISTS {table_name} (id SERIAL PRIMARY KEY, {fields_sql});"

    # execute_commands(create_table_command)
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()

        # Execute the SQL command
        cursor.execute(create_table_command)
        
        # Commit the changes to the database
        conn.commit()
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        print(f"Table {table_name} created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while creating PostgreSQL table {table_name}", error)
    finally:
        # Ensure the connection is closed
        if conn is not None:
            conn.close()


def fill_psql_from_csv(filename):
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # copy table from csv
        with open(f'mongo_csvs/{filename}.csv', 'r') as f:
            next(f) # Skip the header row. 
            cur.copy_from(f, filename.lower(), sep=',', null='')
            #Commit Changes
            conn.commit()
            #Close connection
            conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


