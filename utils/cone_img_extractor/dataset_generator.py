#!/usr/bin/env python
import os
import json
import random
from glob import glob
import shutil
import ruamel.yaml
from ruamel.yaml.scalarstring import SingleQuotedScalarString as sq

ROOT = os.path.dirname(os.path.realpath(__file__))
DATASET_NAME = "dataset"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
train_list = []
val_list = []
test_list = []

image_dir = ROOT + "/extracted/img"
label_dir = ROOT + "/extracted/ann"

ann_files = [os.path.basename(x) for x in glob(label_dir + "/*.json")]
img_files = [os.path.basename(x) for x in glob(image_dir + "/*.jpg")] + [os.path.basename(x) for x in glob(image_dir + "/*.png")]

print(ann_files)
print(img_files)


# Count number of unique labels and create a dictionary for label string
label_list = []
class_string_dict = {}
for file in ann_files:
    f = open(label_dir + "/" + file)
    data = json.load(f)

    label = data['class_id']

    if label not in label_list:
        label_list.append(label)
    class_string_dict[label] = data['class_label']

print(label_list)
print(len(label_list))

# Create list for each unique label
label_list_list = []
for i in range (0, len(label_list)):
    label_list_list.append([])

# Append file names to list based on the label
for file in ann_files:
    f = open(label_dir + "/" + file)
    data = json.load(f)

    label = data['class_id']

    label_list_list[label].append(file)

print(label_list_list)

# Split labels into train, val, test
for l_list in label_list_list:
    random.shuffle(l_list)
    train_list.extend(l_list[:int(len(l_list) * TRAIN_RATIO)])
    val_list.extend(l_list[int(len(l_list) * TRAIN_RATIO):int(len(l_list) * (TRAIN_RATIO + VAL_RATIO))])
    test_list.extend(l_list[int(len(l_list) * (TRAIN_RATIO + VAL_RATIO)):])

print(len(train_list))
print(len(val_list))
print(len(test_list))


# Copy files to train, val, test folders

for i, lab_list in enumerate([train_list, val_list, test_list]):
    sub_dir = ""
    if i == 0:
        sub_dir = "/train/"
    elif i == 1:
        sub_dir = "/valid/"
    elif i == 2:
        sub_dir = "/test/"

    new_img_dir = ROOT + "/datasets/" + DATASET_NAME + sub_dir + "labels"
    new_label_dir = ROOT + "/datasets/" + DATASET_NAME + sub_dir + "images"
    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)

    for label in lab_list:
        file_name = os.path.splitext(label)[0]

        shutil.copy(label_dir + "/" + label, ROOT + "/datasets/" + DATASET_NAME + sub_dir + "labels/" + label)

        shutil.copy(image_dir + "/" + file_name + ".jpg", ROOT + "/datasets/" + DATASET_NAME + sub_dir + "images/" + file_name + ".jpg")

# Create dataset.yaml
names_list = []
print("Length of label_list: ", len(label_list))
for x in range(0, len(label_list)):
    names_list.append((sq(class_string_dict[x])))
    print(names_list)

names_string = ruamel.yaml.comments.CommentedSeq(names_list)
names_string.fa.set_flow_style()

data = dict(
    path = "./",
    train = "data/" + DATASET_NAME + "/train",
    val = "data/" + DATASET_NAME + "/valid",
    test = "data/" + DATASET_NAME + "/test",
    nc = len(label_list),
    names = names_string
)

yaml = ruamel.yaml.YAML()
yaml.indent(sequence=4, offset=2)

yaml_file_name = ROOT + "/datasets/" + DATASET_NAME + ".yaml"
with open(yaml_file_name, 'w') as outfile:
    yaml.dump(data, outfile)
