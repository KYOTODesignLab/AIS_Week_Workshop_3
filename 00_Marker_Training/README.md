# *sumizuke* for augmented craftsmanship 

Activate the environment:

`conda activate AIS26`

To use YOLO we need Ultralytics

`pip install ultralytics`

Step 1 : Take pictures and put the images in the 'images_to_label' folder
Step 2 : Resize images using `resize.py`
Step 2 : Prepare `data.yaml` with location of files, and the classes. DO IT BEFORE NEXT STEP!!!!
Step 3 : Split dataset to training and validation data using `split_dataset.py`. This will also define the classes for the next spep, saved in `labelImg\data\predefined_classes.txt`
Step 4 : Label images using the `labelimg/labelimg.exe`
Step 4 : Train data using `train_yolo_model`
Step 5 : Test on any image using `test_with_custom_img.py`
Step 6 : Deploy into projects using `interpreter.py`





