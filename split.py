"""
This model performs splitting into train, validation and test based on the coordinates of the images.
"""

import json
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm

from visualizations.model import PointOfInterest
from visualizations.unique_instance_prediction import append_geohash, geo_clustering

from panorama.client import PanoramaClient  # pylint: disable=import-error




class CoordinateSplit:
    """
   This class clusters coordinates and splits the data into train, validation and test.

   We start from the dataset which was split based on features.
   We start there so we do not have to do all the other preprocessing on the images (2000x1000 vs 4000x2000) or on the
   annotation files (missing keys, area of 0)
   """

    def __init__(self, root: Path, anns_prefix: str):

        """
        param root: folder which contains train, val and test folders
        """

        self.root = root
        self.subsets: List[str] = ["train", "val", "test"]
        self.anns_prefix = anns_prefix
        self.subsets_new_split: List["str"] = ["train_new_split", "val_new_split", "test_new_split"]
        self.containers: List[PointOfInterest] = []
        self.n_moved = 0
        self.n_total: int = 0  # image counter
        self.n_train: int
        self.n_train_remaining: int
        self.n_val: int
        self.n_test: int
        self.n_moved: int

    def _create_new_folders(self):
        """
       Create folder structures where we put the new split dataset folders
       """
        for subset, subset_new in zip(self.subsets, self.subsets_new_split):
            Path(self.root, subset_new).mkdir(parents=True, exist_ok=True)

    def _move_faulty_panoramas(self):
        """
        We move panoramas that have name pano*** instead of TMX*** to the train_new_split folder.
        """

        # loop through images in the train, val and test folders
        for subset in tqdm(self.subsets, desc="Moving faulty panoramas to train", total=3):
            panoramas = Path(self.root, subset).glob("*.jpg")
            for panorama in panoramas:

                self.n_total = self.n_total + 1  # .glob returns a generator, we cannot just do len(panoramas)
                filename = panorama.parts[-1]
                if filename.startswith("pano"):
                    # move it to the new training folder
                    shutil.move(src=panorama,
                                dst=Path(self.root, "train_new_split", f"{filename}"))
                    self.n_moved = self.n_moved + 1
        print(f"{self.n_moved} files have been moved to {Path(self.root, 'train_new_split')} folder")

    def _get_predictions_objects(self):
        """
        We gather info about coordinates of images, so that we can later cluster them.
        """
        for subset in tqdm(self.subsets, desc="Creating prediction objects with coord metadata", total=3):
            panoramas = Path(self.root, subset).glob("*.jpg")
            for panorama in panoramas:
                panorama_id = panorama.parts[-1]
                panorama_object = PanoramaClient.get_panorama(panorama_id.split(".")[0])  # remove .jpg extension
                long, lat, _ = panorama_object.geometry.coordinates
                container = PointOfInterest(pano_id=panorama_id, coords=(lat, long))
                self.containers.append(container)

    def _get_clustered_prediction_objects(self):

        """
        Assign a geohashes to all coordinates, then cluster the geohashes as granular as we want
        """
        geohashed = append_geohash(self.containers)
        self.containers = geohashed

        # https://en.wikipedia.org/wiki/Geohash

        # TODO: check this out:
        #  https://github.com/ashwin711/proximityhash  - geohashes with circular border
        clustered, self.nr_clusters = geo_clustering(self.containers, prefix_length=8)  # ~ 19m error
        self.containers = clustered

    def _move_images_based_on_cluster(self):
        """
        Calculate how many images we should have in the train 70% , val 20% , test 10%
        Keep adding clustered images to the new split folders such that we reach 70-20-10 split
        """

        self.n_train = int(0.7 * self.n_total)
        self.n_val = int(0.2 * self.n_total)
        self.n_test = int(0.1 * self.n_total)

        print(f"70% out of  {self.n_total} is {self.n_train}")
        print(f"20% out of {self.n_total} is {self.n_val}")
        print(f"10% out of {self.n_total} is {self.n_test}")

        #self.n_train = self.n_train - self.n_moved
        print(f"{self.n_moved} files were already moved, {self.n_train-self.n_moved} are left for the training set.")

        def _get_cluster(id: int):
            single_cluster: List[PointOfInterest] = []
            for container in self.containers:
                if container.cluster == id:
                    single_cluster.append(container)

            return single_cluster

        def _subset_complete(src: str, to_add: int, max_size: int):
            """
            Check if folder is complete
            """
            is_complete = False
            folder_content = Path(self.root, src).glob("*.jpg")
            current_size = sum(1 for _ in folder_content)
            print(f"current size: {current_size}")
            if current_size + to_add > max_size:
                is_complete = True
                return is_complete
            return is_complete

        def _add_cluster_to_subset(cluster, dst_folder):

            def _find_and_move(image, destination):
                """
                Search for image in train, val and test folders, then move it
                """
                for subset in self.subsets:
                    folder_content = Path(self.root, subset).glob("*.jpg")
                    for file in folder_content:
                        filename = file.parts[-1]
                        if filename == image:
                            shutil.move(src=file, dst=destination)
                            return

            for element in cluster:
                img_name = element.pano_id
                destination = Path(self.root, dst_folder, img_name)
                _find_and_move(img_name, destination)

        curr = 0  # current cluster index tracker
        cluster = _get_cluster(curr)
        pbar = tqdm(desc="Adding clusters to subsets", total=self.nr_clusters)
        while curr < self.nr_clusters:  # use for progress bar
            print("train")
            while not _subset_complete("train_new_split", to_add=len(cluster), max_size=self.n_train):
                _add_cluster_to_subset(cluster=cluster, dst_folder="train_new_split")
                print(f"current: {curr}, max: {self.n_train}")
                curr = curr + 1
                pbar.update(1)
                cluster = _get_cluster(curr)
                print(f"cluster size: {len(cluster)}")

            print("val")
            cluster = _get_cluster(curr)
            while not _subset_complete("val_new_split", to_add=len(cluster), max_size=self.n_val):
                print(f"current cluster: {curr}, max: {self.n_val}")
                _add_cluster_to_subset(cluster=cluster, dst_folder="val_new_split")
                curr = curr + 1
                pbar.update(1)
                cluster = _get_cluster(curr)
                print(f"cluster size: {len(cluster)}")

            cluster = _get_cluster(curr)
            print("test")
            while not _subset_complete("test_new_split", to_add=len(cluster), max_size=self.n_test + len(cluster)):
                print(f"current cluster : {curr}, max: {self.n_test}")
                _add_cluster_to_subset(cluster=cluster, dst_folder="test_new_split")
                curr = curr + 1
                pbar.update(1)
                cluster = _get_cluster(curr)
                print(f"cluster size: {len(cluster)}")

            # add the rest to train

            #_add_cluster_to_subset(cluster=cluster, dst_folder="train_new_split")
            # skip the last 3 clusters
            curr = self.nr_clusters

    def _regenerate_annotation_files(self):
        """
        Since we moved files around, we must recreate the annotation files.
        We also store them in the corresponding new folders
        """

        # we open the old annotation files
        # open train:
        train_filename = self.anns_prefix + "train.json"
        train_anns = Path(self.root, "train", train_filename)
        with open(train_anns) as f:
            old_train_annotations = json.load(f)
        f.close()

        # open val
        val_filename = self.anns_prefix + "val.json"
        val_anns = Path(self.root, "val", val_filename)
        with open(val_anns) as f:
            old_val_annotations = json.load(f)
        f.close()

        # open test
        test_filename = self.anns_prefix + "test.json"
        test_anns = Path(self.root, "test", test_filename)
        with open(test_anns) as f:
            old_test_annotations = json.load(f)
        f.close()

        new_train_anns = {"images": [], "annotations": [], "categories": [{"id": 1, "name":"container"}]}
        new_val_anns = {"images": [], "annotations": [], "categories": [{"id": 1, "name":"container"}]}
        new_test_anns = {"images": [], "annotations": [], "categories": [{"id": 1, "name":"container"}]}

        # move all pano*.jpg to train annotations.json

        # FROM TRAIN
        for image in old_train_annotations["images"]:
            name = image["file_name"].split("/")[-1]
            if name.startswith("pano"):
                image["file_name"] = f"train/{name}"
                new_train_anns["images"].append(image)
                id = image["id"]
                # append annotations too
                for ann in old_train_annotations["annotations"]:
                    if ann["image_id"] == id:
                        new_train_anns["annotations"].append(ann)
        # FROM VAL
        for image in old_val_annotations["images"]:
            name = image["file_name"].split("/")[-1]
            if name.startswith("pano"):
                image["file_name"] = f"train/{name}"
                id = image["id"]
                # append annotations too
                for ann in old_train_annotations["annotations"]:
                    if ann["image_id"] == id:
                        new_train_anns["annotations"].append(ann)
        # FROM TEST
        for image in old_test_annotations["images"]:
            name = image["file_name"].split("/")[-1]
            if name.startswith("pano"):
                image["file_name"] = f"train/{name}"
                id = image["id"]
                # append annotations too
                for ann in old_train_annotations["annotations"]:
                    if ann["image_id"] == id:
                        new_train_anns["annotations"].append(ann)

        # in train_new_split
        # gather all filenames
        print("From train")
        train_folder = Path(self.root, "train_new_split").glob("*.jpg")
        for file in train_folder:
            filename = file.parts[-1]

            # search for them in the 3 old .jsons
            # SEARCH IN OLD_TRAIN.JSON
            for image in old_train_annotations["images"]:
                name = image["file_name"].split("/")[-1]
                if name == filename and name.startswith("T"):
                    image["file_name"] = f"train/{name}"
                    new_train_anns["images"].append(image)
                    id = image["id"]
                    for ann in old_train_annotations["annotations"]:
                        if ann["image_id"] == id:
                            new_train_anns["annotations"].append(ann)

            # SEARCH IN OLD_VAL.JSON
            for image in old_val_annotations["images"]:
                name = image["file_name"].split("/")[-1]
                if name == filename and name.startswith("T"):
                    image["file_name"] = f"train/{name}"
                    new_train_anns["images"].append(image)
                    id = image["id"]
                    for ann in old_val_annotations["annotations"]:
                        if ann["image_id"] == id:
                            new_train_anns["annotations"].append(ann)

            # SEARCH IN OLD_TEST.JSON
            for image in old_test_annotations["images"]:
                name = image["file_name"].split("/")[-1]
                if name == filename and name.startswith("T"):
                    image["file_name"] = f"train/{name}"
                    new_train_anns["images"].append(image)
                    id = image["id"]
                    for ann in old_test_annotations["annotations"]:
                        if ann["image_id"] == id:
                            new_train_anns["annotations"].append(ann)

        train_output_path = Path(self.root, "train_new_split", train_filename)
        with open(train_output_path, "w") as f:
            json.dump(new_train_anns, f)
        f.close()



        ######## VALIDATION ##########
        print("From val")
        # gather all filenames
        val_folder = Path(self.root, "val_new_split").glob("*.jpg")
        for file in val_folder:
            filename = file.parts[-1]

            # search for them in the 3 old .jsons
            # SEARCH IN OLD_TRAIN.JSON
            for image in old_train_annotations["images"]:
                name = image["file_name"].split("/")[-1]
                if name == filename and name.startswith("T"):
                    image["file_name"] = f"val/{name}"
                    new_val_anns["images"].append(image)
                    id = image["id"]
                    for ann in old_train_annotations["annotations"]:
                        if ann["image_id"] == id:
                            new_val_anns["annotations"].append(ann)

            # SEARCH IN OLD_VAL.JSON
            for image in old_val_annotations["images"]:
                name = image["file_name"].split("/")[-1]
                if name == filename and name.startswith("T"):
                    image["file_name"] = f"val/{name}"
                    new_val_anns["images"].append(image)
                    id = image["id"]
                    for ann in old_val_annotations["annotations"]:
                        if ann["image_id"] == id:
                            new_val_anns["annotations"].append(ann)

            # SEARCH IN OLD_TEST.JSON
            for image in old_test_annotations["images"]:
                name = image["file_name"].split("/")[-1]
                if name == filename and name.startswith("T"):
                    image["file_name"] = f"val/{name}"
                    new_val_anns["images"].append(image)
                    id = image["id"]
                    for ann in old_test_annotations["annotations"]:
                        if ann["image_id"] == id:
                            new_val_anns["annotations"].append(ann)

        val_output_path = Path(self.root, "val_new_split", val_filename)
        with open(val_output_path, "w") as f:
            json.dump(new_val_anns, f)
        f.close()


        ##### TEST #########
        # gather all filenames
        print("From test")
        test_folder = Path(self.root, "test_new_split").glob("*.jpg")
        for file in test_folder:
            filename = file.parts[-1]

            # search for them in the 3 old .jsons
            # SEARCH IN OLD_TRAIN.JSON
            for image in old_train_annotations["images"]:
                name = image["file_name"].split("/")[-1]
                if name == filename and name.startswith("T"):
                    image["file_name"] = f"test/{name}"
                    new_test_anns["images"].append(image)
                    id = image["id"]
                    for ann in old_train_annotations["annotations"]:
                        if ann["image_id"] == id:
                            new_test_anns["annotations"].append(ann)

            # SEARCH IN OLD_VAL.JSON
            for image in old_val_annotations["images"]:
                name = image["file_name"].split("/")[-1]
                if name == filename and name.startswith("T"):
                    image["file_name"] = f"test/{name}"
                    new_test_anns["images"].append(image)
                    id = image["id"]
                    for ann in old_val_annotations["annotations"]:
                        if ann["image_id"] == id:
                            new_test_anns["annotations"].append(ann)

            # SEARCH IN OLD_TEST.JSON
            for image in old_test_annotations["images"]:
                name = image["file_name"].split("/")[-1]
                if name == filename and name.startswith("T"):
                    image["file_name"] = f"test/{name}"
                    new_test_anns["images"].append(image)
                    id = image["id"]
                    for ann in old_test_annotations["annotations"]:
                        if ann["image_id"] == id:
                            new_test_anns["annotations"].append(ann)

        test_output_path = Path(self.root, "test_new_split", test_filename)
        with open(test_output_path, "w") as f:
            json.dump(new_test_anns, f)
        f.close()

    def main(self):
        self._create_new_folders()
        self._move_faulty_panoramas()
        self._get_predictions_objects()
        self._get_clustered_prediction_objects()
        self._move_images_based_on_cluster()
        self._regenerate_annotation_files()


if __name__ == "__main__":
    root = Path("/Users/dianaepureanu/Documents/Projects/versions_of_data/04-30-2022_092102_UTC")
    anns_prefix = "containers-annotated-COCO-"
    splitter = CoordinateSplit(root, anns_prefix)
    splitter.main()
