#TODO move to a file
import requests
from PIL import Image
from io import BytesIO
import os
import pandas as pd
import logging

def pull_image_from_url(slug,nft_data,base_dir='embedding_test',log_dir='logs'):  
    # URL of the image you want to download 

    logging.basicConfig(filename=f'{log_dir}/{slug}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Specify the directory path you want to create
    directory_path = f"{base_dir}/val/{slug}/"
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Directory '{directory_path}' created successfully.")
    logging.info(f"{nft_data[0]}")
    for token_id,image_url in nft_data:
        try:
            # Send a GET request to the image URL
            response = requests.get(image_url)
        
            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Open the image using Pillow from the binary content of the response
                image = Image.open(BytesIO(response.content))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                new_size = (300, 300)
                resized_image = image.resize(new_size, Image.LANCZOS)
                # display(image)
                # Save the image as a JPEG file
                file_path = f"{base_dir}/val/{slug}/{token_id}.jpg"
                resized_image.save(file_path, "JPEG")
        
            else:
                logging.info(f"Failed to download the image. Status code: {response.status_code}")
    
        except Exception as e:
            logging.info(f"An error occurred: {e}")