import import boto3
import sagemaker
import os
from sagemaker import get_execution_role

BUCKET_NAME = "rotten-tomatoes-tomato-leaf-datasets"
FOLDER_LIST = ['data/', 'LVD2021']

role = sagemaker.get_execution_role() # specifies your permissions to use AWS tools
session = sagemaker.Session() 
s3 = boto3.client('s3')

# Initialize the S3 client
s3_client = boto3.client('s3')

# Local destination folder
local_folder = './data/'
os.makedirs(local_folder, exist_ok=True)  # Create local folder if it doesn't exist

for folder in FOLDER_LIST:
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder)

    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            if file_key.endswith('/'):
                continue
            
            file_name = os.path.basename(file_key)  # Extract just the file name
            local_file_path = os.path.join(local_folder, file_name)
            
            print(f"Downloading {file_key} to {local_file_path}...")
            s3_client.download_file(BUCKET_NAME, file_key, local_file_path)
    else:
        print(f"No files found in {folder} on bucket {BUCKET_NAME}.")