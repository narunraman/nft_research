#TODO move to a file
import requests
from PIL import Image,ImageOps
from io import BytesIO
import os
import pandas as pd
import logging
import random

def pull_image_from_url(slug,nft_data,base_dir='expanded_images',log_dir='logs'):  
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
            elif response.status_code== 504: 
                logging.info(f"Failed to download the image. Status code: {response.status_code}")
                return None
            else:
                logging.info(f"Failed to download the image. Status code: {response.status_code}")
    
        except Exception as e:
            logging.info(f"An error occurred: {e}")

def get_all_filenames(directory):
    all_filenames = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            all_filenames.append(file_path)
    return all_filenames

def get_all_folder_names(directory):
    all_folder_names = []
    for root, dirs, files in os.walk(directory):
        for folder_name in dirs:
            folder_path = os.path.join(root, folder_name)
            all_folder_names.append(folder_path)
    return all_folder_names

def get_immediate_subdirectories(directory):
    subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return subdirectories

def get_immediate_files(directory):
    subdirectories = [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
    return subdirectories

def delete_empty_directories(root_directory):
    for root, dirs, files in os.walk(root_directory, topdown=False):
        for directory in dirs:
            full_path = os.path.join(root, directory)
            if not os.listdir(full_path):  # Check if the directory is empty
                os.rmdir(full_path)

# Function to create a 3x3 grid from randomly selected images
def slug_list_grid(root,slugs,label,out_directory,randomize=False,grid_size=3,border=False):
    # Select 9 directories at random
    if len(slugs)<grid_size*grid_size:
        print('Not Enough Slugs')
        return None
    elif randomize:
        selected_slugs = random.sample(directories, min(9, len(directories)))
    elif len(slugs)==grid_size**2:
        selected_slugs = slugs
    else:
        print('TOO MANY DIRECTORIES NEED TO SET RANDOMIZE TRUE')
        return None
    # Create a blank canvas for the grid
    middle = (grid_size**2-1)/2
    canvas_width = grid_size * 100  # Adjust this value based on the width of your images
    canvas_height = grid_size * 100  # Adjust this value based on the height of your images
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

    # Iterate over selected directories
    for i, slug in enumerate(selected_slugs):
        directory = os.path.join(root,slug)
        # Get a list of all image files in the directory
        image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]

        # Select one random image from the directory
        if image_files:
            selected_image = random.choice(image_files)
            img_path = os.path.join(directory, selected_image)
            img = Image.open(img_path)
            if i==middle and border:
                border_size = 10
                border_color = (255,255,0)  # Red color, change as needed
                
                # Add border to the image
                img = ImageOps.expand(img, border=(border_size, border_size), fill=border_color)
            img = img.resize((100, 100))  # Adjust this size based on your preference
            row = i // grid_size
            col = i % grid_size
            print(row,col,slug)
            canvas.paste(img, (col * 100, row * 100))

    # Save or display the resulting grid image
    canvas.save(os.path.join(out_directory, f'{label}_grid_image.jpg'))
    # canvas.show()

def show_random_image(root,slug):
    directory = os.path.join(root,slug)
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]
    # Select one random image from the directory
    if image_files:
        selected_image = random.choice(image_files)
        img_path = os.path.join(directory, selected_image)
        img = Image.open(img_path)
        display(img)

#Takes the raw output from nfttoimage db and converts to the input for pull_image_from_url
def db_to_args(rows_to_pull):
    # Column names for the DataFrame
    columns = ['slug', 'token_id', 'url']
    
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(rows_to_pull, columns=columns)
    grouped_data = df.groupby('slug').apply(lambda x: (x['slug'].iloc[0], list(zip(x['token_id'], x['url']))))
    args = list(grouped_data)
    return args