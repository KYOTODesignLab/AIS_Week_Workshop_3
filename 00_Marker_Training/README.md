# *sumizuke* for augmented craftsmanship 

Activate the environment:

`conda activate AIS26`

Step 1 : Take pictures and put the images in the `labelImg\marker_images\raw` folder
Step 2 : Resize images using `resize.py`
Step 3 : Prepare `data.yaml` with location of files, and the classes. DO IT BEFORE NEXT STEP!!!!
Step 4 : Split dataset to training and validation data using `split_dataset.py`. This will also define the classes for the next spep, saved in `labelImg\data\predefined_classes.txt`
Step 5 : Label images using the `labelimg/labelimg.exe`
Step 6 : Train data using `train_yolo_model`
Step 7 : Test on any image using `test_with_custom_img.py`







