import cv2
import numpy as np 
import os, sys
import threading


# Import the CCTDecodeRelease module
module_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
module_path = os.path.join(module_path, 'CCTDecode', 'CCTDecode')
if module_path not in sys.path:
    sys.path.append(module_path)

print(sys.path)


import CCTDecodeRelease as cct

marker_list = {"113": "011", "105": "012", "089": "013", "101": "014", "085": "015", "077": "016", "125": "017", "099": "018",
               "083": "019", "075": "01a", "123": "01b", "071": "01c", "119": "01d", "111": "01e", "095": "01f", "135": "021",
               "209": "022", "177": "023", "201": "024", "169": "025", "153": "026", "249": "027", "197": "028", "165": "029",
               "149": "02a", "245": "02b", "141": "02c", "237": "02d", "221": "02e", "189": "02f", "163": "031", "147": "032",
               "243": "033", "139": "034", "235": "035", "219": "036", "187": "037", "231": "039", "215": "03a", "183": "03b",
               "207": "03c", "175": "03d", "159": "03e", "255": "03f", "281": "044", "277": "045", "275": "046", "287": "047",
               "291": "048", "329": "049", "297": "04a", "489": "04b", "473": "04d", "441": "04e", "377": "04f", "293": "052",
               "485": "053", "469": "055", "437": "056", "373": "057", "461": "059", "429": "05a", "365": "05b", "413": "05c",
               "349": "05d", "317": "05e", "509": "05f", "399": "063", "467": "065", "435": "066", "371": "067", "459": "069",
               "427": "06a", "363": "06b", "411": "06c", "347": "06d", "315": "06e", "507": "06f", "423": "072", "359": "073",
               "407": "074", "343": "075", "311": "076", "503": "077", "335": "079", "303": "07a", "495": "07b", "479": "07d"}


cap = cv2.VideoCapture("/dev/video0")  # or use /dev/video0

if not cap.isOpened():
    print("Cannot open camera")
    exit()

from picamera2 import Picamera2  # type: ignore


camera = Picamera2()
config = camera.create_preview_configuration({'format': 'RGB888'})
camera.configure(config)
camera.start()

while True:
    frame = camera.capture_array()
    
    try:
        code_table, img = cct.CCT_extract(frame, 12, 0.85, 'black')
        if code_table:
            if str(code_table[0][0]) in marker_list:
                print('yey')
                image_id = marker_list[str(code_table[0][0])]
                print(image_id)


    except Exception as e:
        print(e)
        
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
