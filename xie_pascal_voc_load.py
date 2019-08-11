import cv2
import yolo.config as cfg
import numpy as np
import os
import xml.etree.ElementTree as ET


class pascal_voc_xie(object):
    def __init__(self):
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.obj_class = cfg.CLASSES
        self.flipped = cfg.FLIPPED
        self.batch_size = cfg.BATCH_SIZE
        self.train_index = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'trainval.txt')
        self.description_pic = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit', 'VOC2007', 'Annotations')
        self.pic_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.epoch = 1
        self.num = 0
        self.index = None
        self.prepare_index()


    def prepare_index(self):
        with open(self.train_index, 'r') as f:
            index = [index.strip() for index in f.readlines()]
            noflipped_mark = [0]*len(index)
            index_0 =list(zip(index, noflipped_mark))
            if self.flipped:
                flipped_mark = [1]*len(index)
                index_1 = list(zip(index, flipped_mark))
                self.index = index_0+index_1
            else:
                self.index = index_0




    def get_index(self):
            index = self.index[self.num: self.num + 45]
            self.num = self.num+45
            if self.num > len(self.index):
                self.epoch = + 1
                self.num = 0
                np.random.shuffle(self.index)
            return index

    def get_image(self, index):
        image = np.zeros([self.batch_size, self.image_size, self.image_size, 3])
        for i,  index in enumerate(index):
            if index[1] == 0:
                pic_name = os.path.join(self.pic_path, index[0]+'.jpg')
                pic = cv2.imread(pic_name)
                pic = cv2.resize(pic, (self.image_size, self.image_size))
                image[i, :, :, :] = pic
            else:
                pic_name = os.path.join(self.pic_path, index[0] + '.jpg')
                pic = cv2.imread(pic_name)
                pic = cv2.resize(pic, (self.image_size, self.image_size))
                image[i, :, ::-1, :] = pic

        return image

    def get_label(self, index):
        label = np.zeros([self.batch_size, self.cell_size, self.cell_size, 25 ])
        for i,  index in enumerate(index):
            filename = os.path.join(self.description_pic, index[0] + '.xml')
            tree = ET.parse(filename)
            size = tree.find('size')
            pic_w = float(size.find('width').text)
            pic_h = float(size.find('height').text)
            w_ratio = self.image_size/pic_w
            h_ratio = self.image_size/pic_h
            label_one_pic = np.zeros([self.cell_size, self.cell_size, 25])
            objs = tree.findall('object')
            for obj in objs:
                box_size = obj.find('bndbox')
                xmin = float(box_size.find('xmin').text)
                xmax = float(box_size.find('xmax').text)
                ymin = float(box_size.find('ymin').text)
                ymax = float(box_size.find('ymax').text)
                x_center = (xmax+xmin)/2
                y_center = (ymax+ymin)/2
                cell_h = int(y_center / pic_h * self.cell_size)
                cell_w = int(x_center / pic_w * self.cell_size)
                label_one_pic[cell_h, cell_w, 0] = 1
                x_center *= w_ratio
                y_center *= h_ratio
                w =(xmax-xmin)*w_ratio
                h =(ymax-ymin)*h_ratio
                #x_center = (1-index[1])*x_center + index[1]*(self.image_size - x_center)#是否需要翻转
                box = [x_center, y_center, w, h]
                label_one_pic[cell_h, cell_w, 1:5] = box
                class_name = str(obj.find('name').text)
                label_one_pic[cell_h, cell_w, 5+self.obj_class.index(class_name)] = 1
                if index[1] == 0:
                    pass
                elif index[1] == 1:#翻转
                    label_one_pic[cell_h, :, :] = label_one_pic[cell_h, ::-1, :]
                    label_one_pic[cell_h, cell_w, 1] = self.image_size -label_one_pic[cell_h, cell_w, 1]

            label[i, :, :, :] = label_one_pic
        return label

    def get(self):
        index =self.get_index()
        return self.get_image(index), self.get_label(index)


if __name__ == '__main__':
    solver = pascal_voc_xie()
    image, label =solver.get()
