This repository is a clone of YOLOv5 (commit 454dae1301abb3fbf4fd1f54d5dc706cc69f8e7e), but applied for blurring panoramic imagery. The model is trained on two classes: pedestrians and license plates. 

To use blurring algorithm, install dependencies
```shell
pip install -r requirements.txt
```

To run blurring algorithm:
```python
python blur.py --weights/best.pt --source folder_with_raw_images --output_folder folder_blurred_images
```
