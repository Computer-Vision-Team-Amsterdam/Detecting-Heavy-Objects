# version 1


from httpx import HTTPStatusError
from tqdm import tqdm
import sys
sys.path.append("..")

import panorama
from panorama.client import PanoramaClient
from panorama import models


BATCH_START = 108
BATCH_END = 200

for i in tqdm(range(BATCH_START, BATCH_END)):
    with open(f'../../data_azure/pano_ids/{i}.txt') as f:
        pano_ids = f.read().splitlines()
    f.close()

    for pano_id in pano_ids:
        try:
            pano = PanoramaClient.get_panorama(pano_id)
            # Download the corresponding image to your machine
            PanoramaClient.download_image(pano, size=models.ImageSize.MEDIUM,
                                          output_location="blurred_panos_from_cloudvps")
        except HTTPStatusError:
            print("image not found!")