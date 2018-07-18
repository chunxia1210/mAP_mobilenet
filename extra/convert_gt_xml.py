import sys
import os
import glob
import xml.etree.ElementTree as ET
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# change directory to the one with the files to be changed
path_to_folder = '../ground-truth'
#print(path_to_folder)
os.chdir(path_to_folder)

labelmap_file = open('/data0/workspace/limei/code/mAP/extra/labelmap_voc.prototxt', 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(labelmap_file.read()), labelmap)


# old files (xml format) will be moved to a "backup" folder
## create the backup dir if it doesn't exist already
if not os.path.exists("backup"):
  os.makedirs("backup")


def get_displayname(labelmap, obj_sku_name):
    num_labels = len(labelmap.item)
    displaynames = []
    for i in range(0, num_labels):
        if obj_sku_name == labelmap.item[i].name:
            displaynames.append(labelmap.item[i].display_name)
            break
    return displaynames


# create VOC format files
xml_list = glob.glob('*.xml')
if len(xml_list) == 0:
  print("Error: no .xml files found in ground-truth")
  sys.exit()
for tmp_file in xml_list:
  #print(tmp_file)
  # 1. create new file (VOC format)
  with open(tmp_file.replace(".xml", ".txt"), "a") as new_f:
    root = ET.parse(tmp_file).getroot()
    for obj in root.findall('object'):
      obj_sku_name = obj.find('name').text

      obj_displayname = str(get_displayname(labelmap, obj_sku_name)).replace('[','').replace(']','').replace("'","")
      bndbox = obj.find('bndbox')
      left = bndbox.find('xmin').text
      top = bndbox.find('ymin').text
      right = bndbox.find('xmax').text
      bottom = bndbox.find('ymax').text
      new_f.write(obj_displayname + " " + left + " " + top + " " + right + " " + bottom + '\n')
  # 2. move old file (xml format) to backup
  os.rename(tmp_file, "backup/" + tmp_file)
print("Conversion completed!")
