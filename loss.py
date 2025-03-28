import torch
import cv2
import numpy                as np
import matplotlib.pyplot    as plt
import torch.nn.functional  as F
import os
import torch.nn as nn
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler


def mae_loss(images, pred):
    Lambda = 100
    min_val_images = torch.min(images)
    max_val_images = torch.max(images)
    images_normalized = (images - min_val_images) / (max_val_images - min_val_images)
    min_val_pred = torch.min(pred)
    max_val_pred = torch.max(pred)
    pred_normalized = (pred - min_val_pred) / (max_val_pred - min_val_pred)
    mae = F.l1_loss(images, pred)      
    total_mae = Lambda * mae
    
    return total_mae


def mse_loss(images, pred):  
    Lambda = 100
    min_val_images = torch.min(images)
    max_val_images = torch.max(images)
    images_normalized = (images - min_val_images) / (max_val_images - min_val_images)
    min_val_pred = torch.min(pred)
    max_val_pred = torch.max(pred)
    pred_normalized = (pred - min_val_pred) / (max_val_pred - min_val_pred)
    criterion = nn.MSELoss() 
    mse = criterion(images, pred)
    total_mse = mse * Lambda

    return total_mse

def save_csv(data, filename, path):
    df = pd.DataFrame(data)
    file_path = os.path.join(path, filename)    
    df.to_csv(file_path, index=False)


def mae_loss_val(images, pred, names, names2):
    total_mae_background = 0
    total_mae_polyp = 0
    count_background = 0
    count_polyp = 0
    error_data_list = [] 
    error_list_final = []
    min_val_images = torch.min(images)
    max_val_images = torch.max(images)
    images_normalized = (images - min_val_images) / (max_val_images - min_val_images)
    min_val_pred = torch.min(pred)
    max_val_pred = torch.max(pred)
    pred_normalized = (pred - min_val_pred) / (max_val_pred - min_val_pred)

    for i, (prediction, name, name2) in enumerate(zip(pred_normalized, names, names2)):
        Lambda = 100
        pred_ = prediction
        image_ = images_normalized[i]
        mae = F.l1_loss(image_, pred_)
        mae = mae * Lambda

        if name == '0':
            total_mae_background += mae
            count_background += 1
        elif name == '1':
            total_mae_polyp += mae
            count_polyp += 1

        error_data = {
            'name2': name2,
            'error': mae,
            'label': name
        }
        error_data_list.append(error_data)
    
    error_list_final.append(error_data_list)
    #print(error_list_final)

    count_polyp = torch.tensor(count_polyp)
    if count_background > 0:
        cost_function_background = total_mae_background / count_background
    else:
        cost_function_background = 0
    if count_polyp > 0:
        cost_function_polyp = total_mae_polyp / count_polyp
    else:
        cost_function_polyp = 0

    cost_function_background = torch.tensor(cost_function_background)
    cost_function_polyp = torch.tensor(cost_function_polyp)
    
    return cost_function_background, cost_function_polyp, error_list_final


contador = 0

def save_images(images, pred, pred2, names2, pred_prom):
    global contador 

    imagen_path = '/data/danielortiz/danielortiz/polyps/Prueba_imagenes10/image'
    pred_path = '/data/danielortiz/danielortiz/polyps/Prueba_imagenes10/pred'
    preddesnorm_path = '/data/danielortiz/danielortiz/polyps/Prueba_imagenes10/pred_desnorm'
    predprom_path = '/data/danielortiz/danielortiz/polyps/Prueba_imagenes10/pred_prom'
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #print(names)
    #print(len(names), len(images), len(pred))

    for j in range(images.size(0)):
        image = images[j]    
        pred_image = pred[j]
        pred_image2 = pred2[j]
        pred_promedio = pred_prom[j]
        pred_promedio2 = torch.mean(pred_promedio, dim=0, keepdim=True)
        #print("MAXIMO Y MINIMO_PRED",pred_image.max(), pred_image.min())
        #print("MAXIMO Y MINIMO_IMAGE",image.max(), image.min())
        name = names2[j]
        image_np = image.detach().cpu().numpy()
        pred_np = pred_image.detach().cpu().numpy()
        pred_np2 = pred_image2.detach().cpu().numpy()
        pred_prom_np = pred_promedio2.detach().cpu().numpy()
        min_val_image = np.min(image_np)
        max_val_image = np.max(image_np)
        image_normalized = (image_np - min_val_image) / (max_val_image - min_val_image)
        image_np = image_normalized
        #image_np = image_np * std[:, None, None] + mean[:, None, None]
        min_val_pred = np.min(pred_np)
        max_val_pred = np.max(pred_np)
        pred_normalized = (pred_np - min_val_pred) / (max_val_pred - min_val_pred)
        min_val_pred2 = np.min(pred_np2)
        max_val_pred2 = np.max(pred_np2)
        pred_normalized2 = (pred_np2 - min_val_pred2) / (max_val_pred2 - min_val_pred2)
        pred_desnorm_np = pred_normalized 
        pred_desnorm_np2 = pred_normalized2
        #print(pred_desnorm_np.shape, pred_desnorm_np2.shape, image_np.shape)
        #print("MAXIMO Y MINOMOSDESNORM_PRED:",pred_desnorm_np.max(), pred_desnorm_np.min())
        #print("MAXIMO Y MINOMOSDESNORM_IMG:",image_np.max(), image_np.min())
        image_np = image_np.transpose(1, 2, 0)
        pred_np = pred_np.transpose(1, 2, 0)
        pred_np2 = pred_np2.transpose(1, 2, 0)
        pred_prom_np2 = pred_prom_np.transpose(1, 2, 0)
        pred_desnorm_np = pred_desnorm_np.transpose(1, 2, 0)
        pred_desnorm_np2 = pred_desnorm_np2.transpose(1, 2, 0)
        image_file = os.path.join(imagen_path, f"image_{contador}_{name}")
        pred_file = os.path.join(pred_path, f"pred_{contador}_{name}")
        pred_desnorm_file = os.path.join(preddesnorm_path, f"pred_desnorm_{contador}_{name}")
        predprom_file = os.path.join(predprom_path, f"pred_prom{contador}_{name}")
        h, w, c = image_np.shape
        image_rgb = np.zeros((h,w,c))
        image_rgb[:,:,2] = image_np[:,:,0]
        image_rgb[:,:,1] = image_np[:,:,1]
        image_rgb[:,:,0] = image_np[:,:,2]
        h2, w2, c2 = pred_desnorm_np.shape
        pred_rgb = np.zeros((h2,w2,c2))
        pred_rgb[:,:,2] = pred_desnorm_np[:,:,0]
        pred_rgb[:,:,1] = pred_desnorm_np[:,:,1]
        pred_rgb[:,:,0] = pred_desnorm_np[:,:,2]
        contador += 1     
        #print(image_rgb.shape, pred_np.shape)  
        cv2.imwrite(image_file, image_rgb*255)
        cv2.imwrite(pred_file, pred_np2) # SE GUARDA SUMA DE GAMMA, mediante una conv
        cv2.imwrite(pred_desnorm_file, pred_rgb*255)
        cv2.imwrite(predprom_file, pred_prom_np2)
        #print(pred_rgb.max(), pred_np.max(), image_rgb.max())

def error_csv(images, pred, names):
  error_data_list = []

  for i in range(len(images)):
    mae = F.l1_loss(images[i], pred[i], reduction='mean')
    mae = mae * 100

    error_data = {
      'name': names[i],
      'error': mae,
    }
    error_data_list.append(error_data)

  return error_data_list

def save_error_csv(error_data_list):
    error_data_list_as_lists = [[data['name'], data['error']] for data in error_data_list]
    with open('csv_error/error3.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(error_data_list_as_lists)

