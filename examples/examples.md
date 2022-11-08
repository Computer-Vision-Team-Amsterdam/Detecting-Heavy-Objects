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