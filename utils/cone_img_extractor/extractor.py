#!/usr/bin/env python
import os
import json
from tqdm import tqdm
from glob import glob
from PIL import Image
import random

ann_path = "data/ann"
img_path = "data/img"


rand_shift = True
rand_shift_max = 0.25
no_cone_percentage = 0.2

class_id_dict = {
    "no_cone": 0,
    "yellow_cone": 1,
    "blue_cone": 2,
    "orange_cone": 3,
    "large_orange_cone": 4
}

ann_files = [os.path.basename(x) for x in glob(ann_path + "/*.json")]
img_files = [os.path.basename(x) for x in glob(img_path + "/*.jpg")] + [os.path.basename(x) for x in glob(img_path + "/*.png")]

print(ann_files)
print(img_files)

no_img_list = []

for ann_file in tqdm(ann_files):
    
    img_name = os.path.splitext(ann_file)[0]
    if img_name in img_files:
    
        f = open(ann_path + "/" + ann_file)
        data = json.load(f)
        img_to_load = img_path + "/" + img_name
        # print(img_to_load)
        img = Image.open(img_to_load)

        bb_cord_list = []
        bb_height_list = []

        for i, cone in enumerate(data['objects']):

            if cone["classTitle"] != "unknown_cone":

                x1 = cone["points"]["exterior"][0][0]
                y1 = cone["points"]["exterior"][0][1]
                x2 = cone["points"]["exterior"][1][0]
                y2 = cone["points"]["exterior"][1][1]

                if(rand_shift):
                    rand_shift_x = round(random.uniform(-rand_shift_max, rand_shift_max), 2)
                    rand_shift_y = round(random.uniform(-rand_shift_max, rand_shift_max), 2)
                    x1 = int(x1 + (x2-x1)*rand_shift_x)
                    y1 = int(y1 + (y2-y1)*rand_shift_y)
                    x2 = int(x2 + (x2-x1)*rand_shift_x)
                    y2 = int(y2 + (y2-y1)*rand_shift_y)


                # print("x1,y1,x2,y2: ", x1, ", ", y1, ", ", x2, ", ", y2)
                
                box = [x1, y1, x2, y2]
                crop_img = img.crop(box)
                bb_cord_list.append(box)
                bb_height_list.append(x2-x1)

                label_file = {}
                label_file['class_label'] = cone["classTitle"]
                label_file['class_id'] = class_id_dict[cone["classTitle"]]
                json_data = json.dumps(label_file)

                new_img_name = "extracted/img/" + os.path.splitext(img_name)[0] + "_" + str(i) + ".jpg"
                crop_img.save(new_img_name)

                new_json_name = "extracted/ann/" + os.path.splitext(img_name)[0] + "_" + str(i) + ".json"
                with open(new_json_name, "w") as outfile:
                    outfile.write(json_data)


        # Create images without cones
        if(no_cone_percentage > 0):
            num_cones = len(data['objects'])

            num_no_cones = int(num_cones * no_cone_percentage)

            no_cone_created = 0

            while no_cone_created < num_no_cones:

                height = data["size"]["height"]
                width = data["size"]["width"]

                xa = random.randint(0, width)
                ya = random.randint(0, height)
                h  = random.randint(min(bb_height_list), max(bb_height_list))
                w  = h * round(random.uniform(0.60, 0.80), 2)
                xb = xa + w
                yb = ya + h

                if(xb < width and yb < height):

                    max_iou = 0

                    for bb in bb_cord_list:
                        iou_bb_xa = max(xa, bb[0])
                        iou_bb_ya = max(ya, bb[1])
                        iou_bb_xb = min(xb, bb[2])
                        iou_bb_yb = min(yb, bb[3])

                        inter_area = abs((max(iou_bb_xb - iou_bb_xa, 0)) * max((iou_bb_yb - iou_bb_ya), 0))
                        iou = 0

                        if (inter_area != 0):
                            boxAArea = abs((xb - xa) * (yb - ya))
                            boxBArea = abs((bb[2] - bb[0]) * (bb[3] - bb[1]))

                            iou = inter_area / float(boxAArea + boxBArea - inter_area)
                            if(iou > max_iou):
                                max_iou = iou

                    if(max_iou <= 0.1):

                        crop_img = img.crop([xa,ya,xb,yb])

                        label_file = {}
                        label_file['class_label'] = "no_cone"
                        label_file['class_id'] = class_id_dict["no_cone"]
                        json_data = json.dumps(label_file)

                        identificator = str(num_cones + no_cone_created)
                        new_img_name = "extracted/img/" + os.path.splitext(img_name)[0] + "_" + identificator + ".jpg"
                        crop_img.save(new_img_name)

                        new_json_name = "extracted/ann/" + os.path.splitext(img_name)[0] + "_" + identificator + ".json"
                        with open(new_json_name, "w") as outfile:
                            outfile.write(json_data)

                        no_cone_created += 1


    else:
        no_img_list.append(img_name)