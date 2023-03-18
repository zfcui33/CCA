import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random
import torchvision.datasets as datasets


from pathlib import Path

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])
def input_resize_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size), interpolation=3),
        transforms.Pad( 10, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop(size=tuple(size)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

# pytorch version of CVUSA loader
class University1652(torch.utils.data.Dataset):
    def __init__(self, mode = '', root = 'D://University1652//', same_area=True, print_bool=False, polar = '',args=None): 
        super(University1652, self).__init__()
        self.args = args
        self.root = root
        self.mode = mode
        self.sat_size = [256, 256]
        self.sat_size_default = [256, 256]
        self.uav_size = [256, 256]
        # if args.sat_res != 0:
        #     self.sat_size = [args.sat_res, args.sat_res]

        if print_bool:
            print(self.sat_size, self.uav_size)

        self.sat_ori_size = [512, 512]
        self.grd_ori_size = [512, 512]

        self.transform_query = input_transform(size=self.uav_size)
        self.transform_train_query = input_resize_transform(size=self.uav_size)
        self.transform_train_reference = input_resize_transform(size=self.sat_size)
        if len(polar) == 0:
            self.transform_reference = input_transform(size=self.sat_size)
        else:
            self.transform_reference = input_transform(size=[750,750]) # 750, 512
        self.to_tensor = transforms.ToTensor()

        self.train_list = self.root + 'train.csv'
        self.test_list = self.root + 'test.csv'

        self.__cur_id = 0  # for training
        self.train_uav_class_name = []
        self.train_uav_path = []
        self.train_sat_class_name = []
        self.train_sat_path = []
        with open(self.train_list, 'r') as file:
            for line in file:
                data = line.split(',')
                class_name = (data[0].split('//')[-1]).split('.')[0]
                for uav_class_idx in range(len(data)-2):
                    self.train_uav_path.append(data[uav_class_idx+1])
                    self.train_uav_class_name.append(class_name)
                # self.train_uav_class_name.append(class_name)
                # self.train_uav_path.append(data[1])
                self.train_sat_path.append(data[0])
                self.train_sat_class_name.append(class_name)
        self.data_size = len(self.train_uav_class_name)

        self.__cur_test_id = 0  # for training
        self.test_uav_class_name = []
        self.test_uav_path = []
        self.test_sat_class_name = []
        self.test_sat_path = []
        with open(self.test_list, 'r') as file:
            for line in file:
                data = line.split(',')
                class_name = (data[0].split('//')[-1]).split('.')[0]
                for test_uav_class_idx in range(len(data)-2):
                    self.test_uav_path.append(data[test_uav_class_idx+1])
                    self.test_uav_class_name.append(class_name)
                # self.test_uav_class_name.append(class_name)
                # self.test_uav_path.append(data[1])
                self.test_sat_path.append(data[0])
                self.test_sat_class_name.append(class_name)
        self.test_data_size = len(self.test_uav_class_name)


    def __getitem__(self, index, debug=False):
        if self.mode== 'train':
            # get uav images index in a batch
            uav_ids = index % len(self.train_uav_class_name)
            # get uav images
            img_query = Image.open(Path(self.root + self.train_uav_path[uav_ids])).convert('RGB')
            img_query = self.transform_query(img_query)
            # get class name list in uav name list
            get_class_name = self.train_uav_class_name[uav_ids]
            # get satellite name list corresponding to uav name list 
            sat_ids = self.train_sat_class_name.index(get_class_name)
            # get satellite images
            img_reference = Image.open(Path(self.root + self.train_sat_path[sat_ids])).convert('RGB')
            img_reference = self.transform_train_reference(img_reference)

            return img_query, img_reference, torch.tensor(uav_ids), torch.tensor(sat_ids), get_class_name, 0

        elif 'scan_val' in self.mode:
            get_class_name = self.test_uav_class_name[index] # class uav names list
            img_query = Image.open(Path(self.root + self.test_uav_path[index])).convert('RGB')
            img_query = self.transform_query(img_query)
            sat_ids = self.test_sat_class_name.index(get_class_name)
            img_reference = Image.open(Path(self.root + self.test_sat_path[sat_ids])).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            return img_query, img_reference, torch.tensor(index), torch.tensor(sat_ids), get_class_name, 0

        elif 'test_reference' in self.mode: # satellite
            # get sat image by index in test set
            img_reference = Image.open(Path(self.root + self.test_sat_path[index])).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            # get sat class name in test set
            sat_name = self.test_sat_class_name[index]
            uav_ids = self.test_uav_class_name.index(sat_name)
            return img_reference, torch.tensor(index), sat_name, uav_ids
            
        elif 'test_query' in self.mode: # uav
            # get uav image by index in test set
            img_query = Image.open(Path(self.root + self.test_uav_path[index])).convert('RGB')
            img_query = self.transform_query(img_query)
            # get uav image class name in test set
            test_cls_name = self.test_uav_class_name[index]
            # get the index of the coressponding class name in test satellite class name list
            sat_ids = self.test_sat_class_name.index(test_cls_name)
            return img_query, torch.tensor(index), torch.tensor(sat_ids),0
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_uav_class_name)
        elif 'scan_val' in self.mode:
            return len(self.test_uav_class_name)
        elif 'test_reference' in self.mode:
            return len(self.test_sat_class_name)
        elif 'test_query' in self.mode:
            return len(self.test_uav_class_name)
        else:
            print('not implemented!')
            raise Exception

    def test_Dataset(self):
        # print(len(self.train_uav_path))
        # print(self.train_sat_path[1])
        # print(len(self.train_uav_class_name))
        # print(self.train_sat_class_name[1])
        # print(self.test_uav_path[54])
        # print(self.test_sat_path[1])
        # print(self.test_uav_class_name[54])
        # print(self.test_sat_class_name[1])
        return


if __name__ == '__main__':
    dataset_Test = University1652()
    dataset_Test.test_Dataset()
    


