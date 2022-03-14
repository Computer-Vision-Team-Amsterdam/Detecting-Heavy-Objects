## Blurring process
0. Create virtual environment.
```shell
conda env create -f det2.yml
```
Have the latest panorama version installed.

If not sure, run 

```shell
pip uninstall panorama
pip install git+ssh://git@github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.1

```
1. To download a batch of 100 images, run
```shell
conda activate det2
python download.py --f "{batch_number}"
```


Images are downloaded at ```data_azure/{batch_number}/images```.

2. To run yolov5 on the downloaded batch, run
```shell
cd yolov5
conda activate yolov5
python detect.py --source "../data_azure/{batch_number}/images" --name "{batch_number}"
conda deactivate
cd ..
```

Labels are stored in the same folder as the images.

3. To open the files with labelImg, run

```shell
labelImg "data_azure/{batch_number}/images"
```
4. To blur the images, run
```shell
python blur.py --source "{batch_number}"
```
Blurred images are stored at ```data_azure/{batch_number}/blurred```
