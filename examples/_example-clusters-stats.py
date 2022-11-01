# python -m  examples._example-clusters-stats
# vizualize geohashes here: https://bhargavchippada.github.io/mapviz/

from visualizations.unique_instance_prediction import append_geohash, geo_clustering, get_points
from annotations_utils import get_filenames_metadata, collect_pano_ids

PREFIX_LENGTH = 8
srcs = ["tests/test_annotations-utils/annotations-as-aml.json",
        "tests/test_annotations-utils/containers-annotated-COCO-val.json"]

# merge names of filenames from all annotation files
file_names = collect_pano_ids(srcs, exclude_prefix="pano")

# create Points of Interest
points = get_filenames_metadata(file_names)

# add geohash to Points of Interest
points_geohash = append_geohash(points)

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