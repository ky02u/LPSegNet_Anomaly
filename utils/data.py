import os
from PIL import Image
import numpy as np
import cv2

import torch.utils.data as data
from sklearn.model_selection import train_test_split

"""
Load all data in a new folder call as data. 
"""
class PolypData(data.Dataset):
    def __init__(self, image_root, trainsize): #MODIFIQUE AQUI
        
        os.makedirs('./Dataset_FONDO/', exist_ok=True)
        #os.makedirs('./dataset_IGHO_2/masks_/', exist_ok=True)
        
        # self.fp_root  = []
        # for fp_videos in os.listdir(fp_root): 
        #     frames = os.listdir(fp_root + fp_videos + '/frames/')
        #     choice = np.random.choice(frames, 290, replace=False)
        #     for file in choice: 
        #         if file.endswith('.jpg') or file.endswith('.png'):
        #             self.fp_root.append(fp_root + fp_videos + '/frames/' + file)
        self.images_root = sorted([image_root + file for file in os.listdir(image_root) if file.endswith('.jpg') or file.endswith('.png')])
        self.read_json()

        #self.fp_root         = sorted(self.choice_no_polyp)
        # self.new_images_root = sorted(self.choice_polyp)
        # self.gts_root        = sorted([file.replace('Normal', 'MASKS') for file in self.choice_polyp])
        # self.gts_root = sorted([gt_root + file for file in os.listdir(gt_root) if file.endswith('.jpg') or file.endswith('.png')])
        
        # self.fp_root = [fp_root + file for file in os.listdir(fp_videos) if file.endswith('.jpg') or file.endswith('.png')]
        # self.fps_root = np.random.choice(self.fp_root, 1500)

        self.trainsize = trainsize
        
        #self.filter_files_np()
        #self.filter_files_wp()

    def read_json(self): 
        json_root = sorted([image_root + file for file in os.listdir(image_root) if file.endswith('.json')])
        if len(json_root) > 0:
            polyp_root = [i.replace('json', 'png') for i in json_root]
            # self.choice_polyp = np.random.choice(polyp_root, 281, replace=False)
            self.choice_polyp = polyp_root
            # np_root =  [i for i in self.images_root if i not in polyp_root]
        else: 
            np_root =  [i for i in self.images_root]
            self.choice_no_polyp = np.random.choice(np_root, 290, replace=False)
        
    #def filter_files_wp(self):
        #assert len(self.new_images_root) == len(self.gts_root)
        # images, gts = [], []
        
        for img_path in zip(self.images_root): #MODIFIQUE AQUI

            img = Image.open(img_path)
            #gt = Image.open(gt_path) MODIFIQUE AQUI
            
            new_img_path = './Dataset_FONDO/'
            #new_gt_path = './dataset_IGHO_2/masks_/' AQI
            name_video = img_path.split('/')[-3][11:]
            name_img = '1-' + name_video + '-' +  img_path.split('/')[-1]

            img.save(new_img_path + name_img)
            #gt.save(new_gt_path + name_img) AQUI

        #     if img.size == gt.size:
        #         images.append(new_img_path + name_img)
        #         gts.append(new_gt_path + name_img)
        # print(len(images))
        # self.images = images
        # self.gts = gts

    #def filter_files_np(self):
       # for i, fp_path in enumerate(sorted(self.fp_root)):
        #    img = Image.open(fp_path)
            # img_c = cv2.imread(fp_path)
         #   gt = np.zeros((img.size[1], img.size[0]))

          
          #  new_img_path = './dataset_IGHO_2/images/'
           # new_gt_path = './dataset_IGHO_2/masks/'
            
            
            #name_video = fp_path.split('/')[-3][11:]
            #name_img = '0-' + name_video + '-' + fp_path.split('/')[-1]

            #img.save(new_img_path + name_img)
            #cv2.imwrite(new_gt_path + name_img, gt)
            
            # if img_c.shape[:-1] == gt.shape:
            #     images.append(new_img_path + name_img)
            #     gts.append(new_gt_path + name_img)

"""
Test the PolypData class 
"""

# image_root = "/data/Datasets/Igho/Uncompressed/Videos/2021_NOVIEMBRE/polipo/2021-11-25_175942_606/Normal/"
#image_root = "/data/Datasets/Igho/Uncompressed/Videos/2021_AGOSTO/NP/Colonoscopy/Train/2021-08-23_105392/Normal/"
#gt_root = "/data/Datasets/Igho/Uncompressed/Videos/2021_AGOSTO/NP/Colonoscopy/Train/2021-08-23_105392/Normal/"
#fp_root = ""

# PolypData(image_root, gt_root, fp_root, trainsize=352)  ##COMENTAR


def generate_train_test_split(train_root, val_root):
    image_extensions = ('.jpg', '.png')

    train_images = [train_root + file for file in os.listdir(train_root) if file.endswith(image_extensions)]
    val_images = [val_root + file for file in os.listdir(val_root) if file.endswith(image_extensions)]
    print("Training size:", len(train_images))
    print("Validation size:", len(val_images))

    return train_images, val_images
