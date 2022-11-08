Steps

1. Merge panoramas from the following sources, excluding pano* filenames 
- `05-16-2022_111258_UTC/annotations-as-aml.json` --> Liska's annotations
- `05-16-2022_111258_UTC/containers-annotated-COCO-train.json` --> Train annotations from best model
- `05-16-2022_111258_UTC/containers-annotated-COCO-val.json` --> Val annotations from best model
- `05-16-2022_111258_UTC/containers-annotated-COCO-test.json` --> Test annotations from best model

The 4 files can be found in AS environment [here](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fb5d1b0e0-1ce4-40f9-87d5-cf3fde7a7b14%2FresourceGroups%2Fcvo-aml-p-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fcvodataweupgwapeg4pyiw5e/path/annotations-blob-container/etag/%220x8DA6E35E185702E%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None). 

Location: `annotations-projects / 07-25-2022_121324_UTC / annotations-for-retraining/`
```python
from annotations_utils import collect_pano_ids

srcs = ["05-16-2022_111258_UTC/annotations-as-aml.json",
        "05-16-2022_111258_UTC/containers-annotated-COCO-train.json",
        "05-16-2022_111258_UTC/containers-annotated-COCO-val.json",
        "05-16-2022_111258_UTC/containers-annotated-COCO-test.json"]

# merge names of filenames from all annotation files
file_names = collect_pano_ids(srcs, exclude_prefix="pano")
``` 

There are 1408 filenames that start with *TMX*.

There are 355 filenames that start with *pano*.

In total there are 1763 images.

All images that start with *pano* go to the training set.

So, if we want an 70-15-15 split, then this means 1234-264-264 images.



Out of the 863 training images, 355 will be the filenames starting with *pano*.


2. Query panorama API and store metadata of the images in json
   (so we don't send too many requests)

```python
from dataclass_wizard import Container
from visualizations.unique_instance_prediction import append_geohash
from visualizations.model import PointOfInterest
from annotations_utils import get_filenames_metadata


# create Points of Interest list
points = get_filenames_metadata(file_names)

# add geohash to Points of Interest list
points_geohash = append_geohash(points)

# save data to json
# a `Container` object is just a wrapper around a Python `list`
Container[PointOfInterest](points_geohash).to_json_file("points.json", indent=4)

# now, read it back
points_geohash = PointOfInterest.from_json_file('points.json')

```

The `points.json` file can be found in the same location in the AS Storage Account.

3. Cluster the points and get some statistics based on the clustering granularity
```python

from visualizations.model import PointOfInterest
from visualizations.unique_instance_prediction import geo_clustering, get_points

PREFIX_LENGTH = 5
points_geohash = PointOfInterest.from_json_file('points.json')

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

4. Plot train, validation, test on a map 

We split the set in `70-15-15`.

After filename splitting, train TMX count is 895, val TMX count is 265 and test TXM count is 248.
Therefore, we have a total of `895+355 = 1250` train images.

```python

from visualizations.model import PointOfInterest
from visualizations.unique_instance_prediction import geo_clustering, color_generator, generate_map
from annotations_utils import split_pano_ids

points_geohash = PointOfInterest.from_json_file('points.json')


# add cluster_id to Points of Interest
clustered_points, clusters = geo_clustering(container_locations=points_geohash,
                                            prefix_length=6)

# array with N different colors for N different clusters
colors = color_generator(nr_colors=len(clusters))

# create the HTML map
generate_map(vulnerable_bridges=[],
             permit_locations=[],
             detections=clustered_points,
             name="examples/Clustered_annotations",
             colors=colors)

# split Points of Interest into 3 subsets: train, val and test
train_points, val_points, test_points = split_pano_ids(clustered_points, nr_clusters=len(clusters))

# create HTML map for each subset
generate_map(vulnerable_bridges=[],
             permit_locations=[],
             detections=train_points,
             name="examples/Train",
             colors=colors)
generate_map(vulnerable_bridges=[],
             permit_locations=[],
             detections=val_points,
             name="examples/Validation",
             colors=colors)
generate_map(vulnerable_bridges=[],
             permit_locations=[],
             detections=test_points,
             name="examples/Test",
             colors=colors)

```

We have split the panoramas PointOfInterest starting with TMX in train, validation and test. 
We can also save them, so we can easily access them later.

```python
from visualizations.model import PointOfInterest
from dataclass_wizard import Container


Container[PointOfInterest](train_points).to_json_file("examples/points_train_TMX.json", indent=4)
Container[PointOfInterest](val_points).to_json_file("examples/points_val_TMX.json", indent=4)
Container[PointOfInterest](test_points).to_json_file("examples/points_test_TMX.json", indent=4)
```

5. Next, we create the annotation files based on the split.

First, place all *pano* filenames in the training annotations

```python

import json

src = "05-16-2022_111258_UTC/containers-annotated-COCO-train.json"

with open(src, "r") as read_file:
    content = json.load(read_file)
read_file.close()

images = [image for image in content["images"] if image["file_name"].split("/")[-1].startswith("pano")]
ids = [image["id"] for image in content["images"] if image["file_name"].split("/")[-1].startswith("pano")]
annotations = [ann for ann in content["annotations"] if ann["image_id"] in ids]

train = {"images": images, "annotations": annotations, "categories": content["categories"]}

# save
filename_output = f"0_{src.split('/')[-1]}"  # 0_containers-annotated-COCO-train.json
with open(f"examples/{filename_output}", 'w') as f:
    json.dump(train, f)

```

Secondly, we reformat the annotation file from Azure AML to a correct COCO format
```python

```

Thirdly, we reindex the annotation files. 
This is because the `annotations-as-aml.json` has ids starting from 1, but we already have id 1 in the 
previous annotations, so we find the last ids in the old annotations and reindex `annotations-as-aml.json` 
from that id.
```python
# get last id
```


Next, we add the *TMX* starting filenames to the annotation json files. 
Keep in mind that the train json is updated (there are already the *pano* files in there), whereas the val 
and test and created.

```python

```




Next, we can filter the annotations files based on area, so we only keep the larger annotations