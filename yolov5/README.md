This repo is a clone of yolov5, but applied for blurring panoramic imagery. The model removes people and license plates from the input images.

To use blurring algorithm, install dependencies
```shell
pip install -r requirements.txt
```

To run blurring algorithm:
```python 
python blur.py --weights/best.pt --input_folder folder_with_raw_images --output_folder folder_blurred_images
```