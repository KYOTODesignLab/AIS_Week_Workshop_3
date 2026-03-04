# get all images in the folder and resize to 256x256
# save to the same folder as a copy

import os
import cv2

def resize_images(folder):
    for root, dirs, files in os.walk(os.path.join(folder, "raw")):
        print('Processing', root)
        for file in files:
            print('Processing', file)   
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.JPG'):
                img = cv2.imread(os.path.join(root, file))
                # Crop square
                h, w = img.shape[:2]
                if h > w:
                    img = img[(h-w)//2:(h+w)//2, :]
                else:
                    img = img[:, (w-h)//2:(w+h)//2]
                # Resize to 256x256
                img = cv2.resize(img, (640, 640))
                # save to the same folder as a copy
                cv2.imwrite(os.path.join(folder, "resized",'resized_' + file), img)
                print('Resized', file)

if __name__ == '__main__':
    folder = os.path.join(os.path.dirname(__file__), "images_to_label")
    print('Resizing images in the folder:', folder)
    resize_images(folder)
    print('All images in the folder have been resized to 320x320')
