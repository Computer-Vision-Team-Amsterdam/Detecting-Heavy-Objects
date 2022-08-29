import requests
from requests.auth import HTTPBasicAuth
import shutil

USERNAME = ""
PASSWORD = ""

#base_url = "https://3206eec333a04cc980799f75a593505a.objectstore.eu/processed/"
#
intermediate_url = "https://3206eec333a04cc980799f75a593505a.objectstore.eu/2016/03/17/TMX7315120208-000020/pano_0000_000000.jpg"
processed_url = "https://3206eec333a04cc980799f75a593505a.objectstore.eu/processed/2016/03/17/TMX7315120208-000020/" \
               "pano_0000_000000/equirectangular/panorama_4000.jpg"
response = requests.get(processed_url, stream=True, auth=HTTPBasicAuth(USERNAME, PASSWORD))
print(response.status_code)
if response.status_code == 200:
    print("authenticated")
filename = './test_pano_processed.jpg'
with open(filename, 'wb') as out_file:
    shutil.copyfileobj(response.raw, out_file)
del response
