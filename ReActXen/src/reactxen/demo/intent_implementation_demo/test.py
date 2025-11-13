import platform
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder
import os

from custom_tools.huggingface import base_huggingface

# test print information corresponding to a particular platform
# print(platform.architecture())
print(platform.machine())
print(platform.processor())


# Load dotenv files
load_dotenv()
print(f"Huggingface api key : {os.getenv('HF_APIKEY')}")


def push_data_to_huggingface():
    token = os.getenv("HF_APIKEY")

    HfFolder.save_token(token=token)
    api_instance = HfApi()
    api_instance.upload_folder(
        folder_path=(
            "/Users/ayandas/Desktop/research_ibm/Intent-Based-Industrial-Automation/data/CMAPSSData"
        ),
        repo_id="DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data",
        repo_type="dataset",
        token=token,
    )


# Simple test to see if environmental variables are loading as intended
# Access and print out the variables to see if they render
print(f"watsonx api key : {os.getenv('WATSONX_APIKEY')}")
print(f"brave search API Kye : {os.getenv('BRAVE_API_KEY')}")
# print(f"")

if __name__ == "__main__":
    push_data_to_huggingface()
