# Image selection

For the blurring project, we create a dataset with unblurred images which are stored on CloudVPS.

We have to annotate this dataset and potentially extend in the future, so we keep track below of how we created this 
dataset.

The current blurring model has problems with blurring people of color, people that
are in squares, markets, restaurants with outside terraces. We query the panorama API based on these
locations.

### Locations and dates used in pano API queries

In ```find_crowded_panos_pipeline.py``` we loop through the rows in the **Crowded Locations Table** below and find the id's of the panoramas.
We save the id's of these panoramas. Next we use these ids to download the panoramas images
from CloudVPS. Images are downloaded to ```panoramas``` folder.


| Location name  |  Lat  | Long | Search radius (m)  | Date (yyyy-mm-dd)  | 
|---|---|---|---| --- |
| Dam Square  |  52.368519 |  4.902040 | 50  |  - |
| Leidseplein  | 52.363630  | 4.881880  | 50 | -  |
| Rembrandtplein  | 52.365990 | 4.896470  | 50 |  - |
| Museumplein  |  52.357430 |  4.881610 | 50  | -  |
| Waterlooplein  |  52.367760 | 4.902752  | 50 | - |

Input: ```crowded_locations.csv```

Output: ``crowded_pano_ids.csv``, ```panoramas/*.jpg```

Example usage:

```python find_crowded_panos_pipeline.py```

If you already have the list of panorama ids, i.e. ```crowded_pano_ids.csv```, simply run 
```python download_pano_from_cloudvps.py``` to download the corresponding images to the
```panoramas``` folder.