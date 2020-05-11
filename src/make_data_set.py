import glob
import xml
from tqdm import tqdm
import random
import os
import shutil
import xml.etree.ElementTree as ET


def to_parseable(tree):
    t = ET.tostring(tree)
    t = t.lower()
    return ET.fromstring(t)

datapath = '../data/'
annotation_dir = datapath + '/processed/annotations/places_annot/'
images_dir = '../../../datasets/DeBoer/data/'
output_dir = datapath + '/processed/places_boer/'

annotations = glob.glob(annotation_dir + '*')

random.seed(666)
random.shuffle(annotations)

split_1 = int(0.8 * len(annotations))
split_2 = int(0.9 * len(annotations))

train_annotations = annotations[:split_1]
validation_annotations = annotations[split_1:]
# validation_annotations = annotations[split_1:split_2]
# test_annotations = annotations[split_2:]


types_ = ['train', 'test', 'validation']
for index, type_ in enumerate(types_):
        print("processing: {}".format(index))
        
        for xml in tqdm([train_annotations,
                         test_annotations,
                         validation_annotations][index]):
            tree = ET.parse(xml).getroot()
            tree = to_parseable(tree)
            filename = filename = tree.find("filename").text
            for elem in tree.iter():
                if elem.tag == 'name':
                    folder = elem.text
            if folder:
                os.makedirs(os.path.join(output_dir, type_, folder), exist_ok=True)
                shutil.copy(os.path.join(images_dir, filename), os.path.join(output_dir, type_, folder))
            else:
                pass