import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ['target']  # 写自己类别
abs_path = os.getcwd()
print(abs_path)


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[2] + box[0]) / 2.0  # 中心点
    y = (box[3] + box[1]) / 2.0
    w = box[2] - box[0]  # 宽
    h = box[3] - box[1]  # 高
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open('dataset/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open('dataset/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)

        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),  # 左上角
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))  # 右下角
        b1, b2, b3, b4 = b  # (x1,y1,x2,y2)
        '''
          (x1,y1)
            --------------------
            |                  |
            |                  |
            |                  |
            |                  |
            |                  |
            --------------------(x2,y2)
        '''
        if b3 > w:
            b3 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
for image_set in sets:
    if not os.path.exists('dataset/labels/'):
        os.makedirs('dataset/labels/')
    image_ids = open('dataset/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('dataset/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(abs_path + '/dataset/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()