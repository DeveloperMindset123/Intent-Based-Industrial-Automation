import platform
from dotenv import load_dotenv
import os

from custom_tools.huggingface import base_huggingface

# test print information corresponding to a particular platform
# print(platform.architecture())
# print(platform.machine())
# print(platform.processor())

# Load dotenv files
load_dotenv()

# Simple test to see if environmental variables are loading as intended
# Access and print out the variables to see if they render
print(f"watsonx api key : {os.getenv('WATSONX_APIKEY')}")
print(f"brave search API Kye : {os.getenv('BRAVE_API_KEY')}")
# print(f"")
