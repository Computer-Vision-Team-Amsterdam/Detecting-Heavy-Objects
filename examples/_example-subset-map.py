from visualizations.unique_instance_prediction import append_geohash, geo_clustering, color_generator, generate_map
from annotations_utils import collect_pano_ids, get_filenames_metadata, split_pano_ids


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

