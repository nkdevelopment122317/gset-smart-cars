import os
from time import sleep

img_dir_str = "C:\\Users\\micro\\GSET OpenCV Scripts\\gset_smart_cars_training_data\\images\\training_video_2\\"
ann_dir_str = "C:\\Users\\micro\\GSET OpenCV Scripts\\gset_smart_cars_training_data\\annotations\\training_video_2\\"

imgs = []
anns = []

img_dir = os.fsencode(img_dir_str)
ann_dir = os.fsencode(ann_dir_str)

def populate_file_list(dir, ending, list_to_populate):
    for file in os.listdir(dir):
        filename = os.fsencode(file)
        if filename.endswith(b"." + str.encode(ending)):
            print("[INFO] " + filename.decode("utf-8").replace("." + ending, ""))
            list_to_populate.append(filename.decode("utf-8").replace("." + ending, ""))
        else:
            continue

populate_file_list(img_dir_str, "jpg", imgs)

sleep(1)

populate_file_list(ann_dir_str, "xml", anns)

print("[INFO] Done. Length of imgs list: " + str(len(imgs)))
print("[INFO] Done. Length of anns list: " + str(len(anns)))

imgs_to_delete = [i + ".jpg" for i in imgs if i not in anns]

sleep(1)

for img in imgs_to_delete:
    os.remove(img_dir_str + img)
    print("[INFO] Deleted image: " + img)

print("[INFO] Deleted unused images. Length of imgs_to_delete list: " + str(len(imgs_to_delete)))

