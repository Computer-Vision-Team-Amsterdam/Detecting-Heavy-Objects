## Steps  

1. 
  
1. Merge panoramas from the following sources, excluding pano* filenames   
- `05-16-2022_111258_UTC/annotations-first-iteration.json` --> Diana's and Laurens' annotations
- `05-16-2022_111258_UTC/annotations-second-iteration.json` --> Liska's annotations  
The 2 files can be found in AS environment [here-TODO]().   
  
Location: `annotations-projects / 07-25-2022_121324_UTC / annotations-for-retraining/`  
```python  
from annotations_utils import collect_pano_ids  
  
srcs = ["05-16-2022_111258_UTC/annotations-first-iteration.json",  
        "05-16-2022_111258_UTC/annotations-second-iteration.json"]  
  
# merge names of filenames from all annotation files  
file_names, ids = collect_pano_ids(srcs, exclude_prefix="pano")  
```   
Output:
```
Source 05-16-2022_111258_UTC/annotations-first-iteration.json has 1284 filenames.
Source 05-16-2022_111258_UTC/annotations-second-iteration.json has 431 filenames.
Total images: 1715
Nr images which start with pano: 370.
Nr UNIQUE images which start with pano: 367.
Nr images which start with TMX:  1345.
Nr UNIQUE images which start with TMX: 1326.
```
We have 22 duplicates. Also, the ids of the images are duplicated, since both files start indexing from 1. 

Assume correct annotations, then these duplicates do not represent a problem. 
We will cluster the images by location, so these duplicates will belong to the same subset.

---
2.  Create a single annotation file.

For this, we need to reindex the annotations. 
```
annotations-first-iteration.json starts with 
"images": [  
 {  "id": 1,  
     ....
  }]
annotations-second-iteration.json also starts with
 "images": [  
 {  "id": 1,  
     ....
  }]
```
  We want unique image `id` and annotation `id`.
  `annotations-first-iteration.json` has 1284 images and 1992 annotations. 
`annotations-second-iteration.json` has 431 images and 539 annotations.

Therefore:
-  reindex images from `annotations-second-iteration.json` from `[1, 432]` to `[1285, 1716]` -> `img-reindex-step = 1284`
- reindex annotations from `annotations-second-iteration.json` from `[1, 540]` to `[1993, 2532]`.  --> `ann-reindex-step = 1992`

```
Example: 
 "images": [  
 {  "id": 370 + img-reindex-step,  
     ....
  }]
"annotations": [{  
  "segmentation": [[...]],
  "id": 468 + ann-reindex-step,  
  "category_id": 1,  
  "image_id": 370 + img-reindex-step,  
  "area": 0.0,  
  "bbox": [...]  
},
...]
```

```python
import json  
  
src_first = "05-16-2022_111258_UTC/annotations-first-iteration.json"  
src_second = "05-16-2022_111258_UTC/annotations-second-iteration.json"  
img_reindex_step = 1284  
ann_reindex_step = 1992  
  
# read first file  
with open(src_first, "r") as read_file:  
  content_first = json.load(read_file)  
read_file.close()  
  
# collect images and annotations  
images_first = [image for image in content_first["images"]]  
annotations_first = [ann for ann in content_first["annotations"]]  
  
# read second file  
with open(src_second, "r") as read_file:  
  content_second = json.load(read_file)  
read_file.close()  
  
# collect images and annotations  
images_second = [image for image in content_second["images"]]  
annotations_second = [ann for ann in content_second["annotations"]]  
  
# reindex images from second file  
for i, image in enumerate(images_second):  
  image["id"] = image["id"] + img_reindex_step  
    images_second[i] = image  
  
# reindex annotations from second file  
for i, annotation in enumerate(annotations_second):  
  annotation["id"] = annotation["id"] + ann_reindex_step  
    annotation["image_id"] = annotation["image_id"] + img_reindex_step  
    content_second[i] = annotation  
  
# join images and annotations  
images_both = images_first + images_second  
annotations_both = annotations_first + annotations_second  
  
both = {"images": images_both, "annotations": annotations_both, "categories": content_first["categories"]}  
  
# ensure the last indices are as expected, i.e. 1716 and 2532  
assert images_both[-1]["id"] == len(images_both) + 1  
assert annotations_both[-1]["id"] == len(annotations_both) + 1  
  
# ensure we do not have duplicate image ids or annotations ids  
  
ids = [image["id"] for image in images_both]  
anns_ids = [ann["id"] for ann in annotations_both]  
assert len(ids) == len(set(ids))  
assert len(anns_ids) == len(set(anns_ids))  
  
# save joined annotation file  
with open("05-16-2022_111258_UTC/annotations-both.json", 'w') as f:  
  json.dump(both, f)
```
---
3. We reformat the annotation file from Azure AML to a correct COCO format  
```python    
from utils import DataFormatConverter  
from pathlib import Path  
  
src = "05-16-2022_111258_UTC/annotations-both.json"  
  
converter = DataFormatConverter(Path(src), output_dir="05-16-2022_111258_UTC")  
converter.convert_data(do_split=False) 
# data is saved in 05-16-2022_111258_UTC/annotations-both-converted.json
```  
---

4. Query panorama API and store metadata of the images in json  
   (so we don't send too many requests)  
  
```python  
from annotations_utils import collect_pano_ids  
from dataclass_wizard import Container  
from visualizations.unique_instance_prediction import append_geohash  
from visualizations.model import PointOfInterest  
from annotations_utils import get_filenames_metadata  
  
srcs = ["05-16-2022_111258_UTC/annotations-both-converted.json"]  
  
# merge names of filenames from all annotation files  
file_names, ids = collect_pano_ids(srcs, exclude_prefix="pano")  
  
# create Points of Interest list  
points = get_filenames_metadata(file_names, ids)  
  
# add geohash to Points of Interest list  
points_geohash = append_geohash(points)  
  
# save data to json  
# a `Container` object is just a wrapper around a Python `list`  
Container[PointOfInterest](points_geohash).to_json_file("05-16-2022_111258_UTC/metadata_for_map.json", indent=4)  
  
# now, read it back  
points_geohash = PointOfInterest.from_json_file('05-16-2022_111258_UTC/metadata_for_map.json')
# len(points_geohash) = 1345 --> 1000 + 345
```  
  
The `metadata_for_map.json` file can be found in the same location in the AS Storage Account.  

---
#### Some further explanations: 
We split the data in `train` and `validation`, with a `80-20` split. 
This roughly means 
- `80% out of 1715 = 1370 images for training` and 
- `20% out of 1715 = 345 images for validation`

We will place all 370 *pano* images in the `training` set. The remaining 1000 *TMX* training images we can plot on a map, since we know their metadata.

We split `train` and `validation` based on cluster location, so we will not have exactly 1370 and 345 images, but *roughly* these numbers, as we see below. 

---  
#### Some further analysis:

We cluster the points and get some statistics based on the clustering granularity  
```python   
from visualizations.model import PointOfInterest  
from visualizations.unique_instance_prediction import geo_clustering, get_points  
  
PREFIX_LENGTH = 6  
points_geohash = PointOfInterest.from_json_file('05-16-2022_111258_UTC/metadata_for_map.json')  
  
# add cluster_id to Points of Interest  
clustered_points, clusters = geo_clustering(container_locations=points_geohash,  
                                            prefix_length=PREFIX_LENGTH)  
  
  
points_per_cluster = []  
for geo_prefix, cluster_id in clusters.items():  
    points_subset = get_points(clustered_points, cluster_id=cluster_id)  
    nr_points = len(points_subset)  
    points_per_cluster.append(nr_points)  
  
mean_points_per_clusters = sum(points_per_cluster)/len(points_per_cluster)  
print(f"{50*'-'}STATS{50*'-'}")  
print(f"For prefix length {PREFIX_LENGTH} there are {len(clusters)} clusters.\n"  
      f"On average there are {mean_points_per_clusters} points in each clusters.\n\n")  
print(f"Number of points in each cluster: {points_per_cluster}")  
print("Names of clusters:")  
for geo_prefix, cluster_id in clusters.items():  
    print(geo_prefix)  
print("Vizualize these clusters at https://bhargavchippada.github.io/mapviz/.")  
print(f"{50*'-'}STATS{50*'-'}")   
```  
Output:
```
For prefix length 6 there are 75 clusters.
On average there are 17.9 points in each clusters.
```
---
5. Split the data into `train` and `validation`

```python
from visualizations.model import PointOfInterest  
from dataclass_wizard import Container  
from visualizations.unique_instance_prediction import geo_clustering  
from annotations_utils import split_pano_ids  
  
points_geohash = PointOfInterest.from_json_file('05-16-2022_111258_UTC/metadata_for_map.json')  
  
# add cluster_id to Points of Interest  
clustered_points, clusters = geo_clustering(container_locations=points_geohash,  
  prefix_length=6)  
  
# split Points of Interest into 3 subsets: train, val and test  
train_points, val_points = split_pano_ids(clustered_points, nr_clusters=len(clusters))  
  
Container[PointOfInterest](train_points).to_json_file("05-16-2022_111258_UTC/map_train_TMX.json", indent=4)  
Container[PointOfInterest](val_points).to_json_file("05-16-2022_111258_UTC/map_val_TMX.json", indent=4)
```
Output:
```
Number of panorama files: 1345
Train count is 1372, val count is 343.
After filename splitting, train count is 1008, val count is 337.
```
---

  
5. Plot `train` and  `validation` on a map.
  
  With the maps below we can check for an even spread of the clusters in both `train` and `validation` subset.
```python    
from visualizations.model import PointOfInterest  
from visualizations.unique_instance_prediction import color_generator, generate_map  
  
train_points = PointOfInterest.from_json_file('05-16-2022_111258_UTC/map_train_TMX.json')  
val_points = PointOfInterest.from_json_file('05-16-2022_111258_UTC/map_val_TMX.json')  
  
  
# array with N different colors for N different clusters  
# len(clusters) = 75  
# colors = color_generator(nr_colors=len(clusters))  
colors = color_generator(nr_colors=75)  
  
# create HTML map for each subset  
generate_map(vulnerable_bridges=[],  
  permit_locations=[],  
  detections=train_points,  
  name="05-16-2022_111258_UTC/Train",  
  colors=colors)  
generate_map(vulnerable_bridges=[],  
  permit_locations=[],  
  detections=val_points,  
  name="05-16-2022_111258_UTC/Validation",  
  colors=colors)
  
```  
  Let's also create a map with both `train` and `validation` sets to see how close the clusters are to each other.

```python
from visualizations.model import PointOfInterest  
from visualizations.unique_instance_prediction import color_generator, generate_map  
  
train_points = PointOfInterest.from_json_file('05-16-2022_111258_UTC/map_train_TMX.json')  
val_points = PointOfInterest.from_json_file('05-16-2022_111258_UTC/map_val_TMX.json')  
  
train_validation_points = train_points + val_points  
# array with N different colors for N different clusters  
# len(clusters) = 75  
# colors = color_generator(nr_colors=len(clusters))  
colors = color_generator(nr_colors=75)  
  
# create HTML map for each subset  
generate_map(vulnerable_bridges=[],  
  permit_locations=[],  
  detections=train_validation_points,  
  name="05-16-2022_111258_UTC/Train_Validation",  
  colors=colors)
```
  ---

6. Create `train` and `validation` COCO annotation files. 

  First 
  - place all *pano* filenames in the training annotations.
  Then
   - place entries from `map_train_TMX` in the `train` annotation file.
   - place entries from `map_val_TMX` in the `val` annotation file.
 
We can add PointOfInterest.id to get the ids, which are unique, instead of filenames. 
```python  
  
import json  
from visualizations.model import PointOfInterest  
  
src = '05-16-2022_111258_UTC/annotations-both-converted.json'  
  
with open(src, "r") as read_file:  
  content = json.load(read_file)  
read_file.close()  
  
# get all pano* images and their corresponding annotations from the main annotation file  
pano_images = [image for image in content["images"] if image["file_name"].startswith("pano")]  
pano_images_ids = [image["id"] for image in pano_images]  
pano_annotations = [annotation for annotation in content["annotations"] if annotation["image_id"] in pano_images_ids]  
  
assert len(pano_images_ids) == 370  
assert len(pano_images_ids) == len(set(pano_images_ids))  
  
# get all train TMX* images and their corresponding annotations from the main annotation file  
train_points_TMX = PointOfInterest.from_json_file('05-16-2022_111258_UTC/map_train_TMX.json')  
TMX_train_images_ids = [train_point.image_id for train_point in train_points_TMX]  
TMX_train_images = [image for image in content["images"] if image["id"] in TMX_train_images_ids]  
TMX_train_annotations = [annotation for annotation in content["annotations"] if annotation["image_id"] in TMX_train_images_ids]  
  
assert len(TMX_train_images) == 1008  
assert len(TMX_train_images_ids) == len(set(TMX_train_images_ids))  
  
  
train_images = pano_images + TMX_train_images  
train_annotations = pano_annotations + TMX_train_annotations  
  
# get all validation images and their corresponding annotations from the main annotation file  
val_points = PointOfInterest.from_json_file('05-16-2022_111258_UTC/map_val_TMX.json')  
val_images_ids = [val_point.image_id for val_point in val_points]  
val_images = [image for image in content["images"] if image["id"] in val_images_ids]  
val_annotations = [annotation for annotation in content["annotations"] if annotation["image_id"] in val_images_ids]  
  
# assert disjoint sets  
assert not bool(set(pano_images_ids) & set(TMX_train_images_ids))  
assert not bool(set(TMX_train_images_ids) & set(val_images_ids))  
  
# save to json files  
train = {"images": train_images, "annotations": train_annotations, "categories": content["categories"]}  
with open("05-16-2022_111258_UTC/containers-annotated-COCO-train.json", 'w') as f:  
  json.dump(train, f)  
  
val = {"images": val_images, "annotations": val_annotations, "categories": content["categories"]}  
with open("05-16-2022_111258_UTC/containers-annotated-COCO-val.json", 'w') as f:  
  json.dump(train, f)
```  
  


  

  
  
  
Next, we can filter the annotations files based on area, so we only keep the larger annotations