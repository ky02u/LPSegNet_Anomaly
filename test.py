import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

import argparse
import torch
import torch.nn             as nn
import numpy                as np
import matplotlib.pyplot    as plt
from lib.PraNet_Res2Net_v7  import PraNet
from utils.dataloader       import test_dataset
from metrics                import metric_IOU, metrics_seg, calculate_mae, metrics_mae
from activations            import show_activations, generate_segmentation, generate_segmentation_numpy
from PIL import Image
import cv2
import csv

def extract_name(name):
    if len(name) == 5:
        new_name = '000' + name
    elif len(name) == 6: 
        new_name = '00' + name
    elif len(name) == 7: 
        new_name = '0' + name
    else: 
        new_name = name
    
    return new_name

def extract_name_number(name):
    if name <= 9:
        new_name = '00' + str(name)
    elif name>9 and name<=99: 
        new_name = '0' + str(name)
    else: 
        new_name = str(name)
    return new_name

def save_outputs(outputs, output_folder):    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, output in enumerate(outputs):
        output_path = os.path.join(output_folder, f"output_{i}.png")
        output = (output * 255).astype(np.uint8)
        cv2.imwrite(output_path, output)


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/data/jefelitman_pupils/danielortiz_tesis/danielortiz/polyps/snapshots/Prueba_anomaly_80epochs_1e6lr/80.pth')
parser.add_argument('--dataset', type=str, default='images')
opt = parser.parse_args()

datasets =  {
             'images': ['Kvasir', 'ETIS-Larib', 'CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'Kvasir'], 
            #  'images': ['CVC_cls'],
             'IGHO-AGO': ['2021-08-26_105781', '2021-08-30_106015_2', '2021-08-30_106025'], #'2021-08-23_105365', '2021-08-23_105381', '2021-08-23_105396', '2021-08-26_105774',  
             'IGHO-NOV': ['2021-11-25_134924_312', '2021-11-25_143810_249', '2021-11-25_151631_034', '2021-11-25_175942_606'],
             'IGHO-SEPT': ['2021-09-02-106236', '2021-09-02-106239', '2021-09-02-106262'], #'2021-09-02-106236', '2021-09-02-106239', '2021-09-02-106262', '2021-09-09-106762', '2021-09-09-106786', '2021-09-11-107014', '2021-09-11-107050' 
             'IGHO-JUN': ['2022-06-09_112525_362'],
             'CVC-video': ['Video1', 'Video2', 'Video3', 'Video4', 'Video5', 'Video6', 'Video7', 'Video8', 'Video9', 'Video10', 'Video11', 'Video12', 'Video13', 'Video15', 'Video16', 'Video17', 'Video18'],
             'ASU-Mayo': ['Video_2', 'Video_4', 'Video_24', 'Video_49', 'Video_52', 'Video_61', 'Video_66', 'Video_68', 'Video_69', 'Video_70'],
             'test': ['60_test_public']
            }

path =  {'images': '/data/jefelitman_pupils/danielortiz_tesis/danielortiz/lina/Polyps/data/TestDataset/',
         'IGHO-AGO': '/data/jefelitman_pupils/Igho/polipo_2021_AGOSTO/', 
         'IGHO-SEPT': '/data/jefelitman_pupils/Igho/polipo_2021_SEPTIEMBRE/', # 2021_AGOSTO 2021_NOVIEMBRE 2021_SEPTIEMBRE 2022_JUNIO
         'CVC-video': '/data/jefelitman_pupils/danielortiz_tesis/danielortiz/lina/Polyps/data/TestDataset/CVC-Video/',
         'ASU-Mayo': '/data/jefelitman_pupils/danielortiz_tesis/danielortiz/lina/Polyps/data/TestDataset/ASU-Mayo/',
         'IGHO-NOV': '/data/jefelitman_pupils/Igho/polipo_2021_NOVIEMBRE/',
         'IGHO-JUN': '/data/jefelitman_pupils/Igho/polipo/',
         'test': '/data/jefelitman_pupils/'
        }


hist_path = '/home/danielortiz/polyps/data/TestDataset/CVC_cls/valid.csv'
#hist_path = '/home/linamruiz/Polyps/tesismaestria/LPSegNet/dataset_cls/data/m_train/train_2.csv'
hist      = False
model_name = opt.pth_path.split('/')[2]
video_name = opt.dataset.split('/')

labels_total = []
iou_seg_total, iou_total = [], []
labels_fp = []
mae_total = []
data_by_dataset = {}

for index, _data_name in enumerate(datasets[opt.dataset]):

    iou_video, label_video, iou_seg_video = [], [], []   
    mae_total = []
    print(_data_name)
    #data_path = path['Dataset_FP_IGHO2'] + '{}'.format(_data_name)
    data_path = path[opt.dataset] + '{}'.format(_data_name) #USAR ESTE CUANDO NO SE USEN LA CARPETA DE FP
    save_path = './results/' + model_name + '/{}/seg_binary/'.format(_data_name)
    os.makedirs(save_path, exist_ok=True)

    model = PraNet(use_attention='PCM', mode_cls='max_pooling').cuda()
    # model = LPSegNet().cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    
    model.cuda()
    model.eval()

    image_root  = '{}/images/'.format(data_path)      #USAR ESTAS PARA DATASETS QUE NO SEAN DE IGHO o en su defecto el dataset para sacar FP
    gt_root     = '{}/masks/'.format(data_path)       

    #image_root  = '{}/Normal/'.format(data_path)      
    #gt_root     = '{}/MASKS/'.format(data_path)    
    #gt_root     = '{}/Masks/'.format(data_path)       #PARA AGOSTO 2021 USAR ESTA   

    test_loader = test_dataset(image_root, gt_root, opt.testsize, hist, hist_path)
    count = 0

    for i in range(test_loader.size):
        image, gt, name_img, name, label_histhology = test_loader.load_data() 
        image2              = np.asarray(image, np.float32) #imagen original con sus tamaños     
        gt              = np.asarray(gt, np.float32)
        label           = 1 if gt.max() == 255.0 else 0
        gt              /= (gt.max() + 1e-8) 
        #imagen_original = image_orig  
        #imagen_original = np.array(imagen_original)         
        image           = image.cuda() #Los tamaños no son los mismos pero coinciden los de la imagen y los de la pred!
        #mascara_original = gt   
        seg,_,_  = model(image) #pcm_t seg es la reconstruccion
        #print(type(image))
        #print(len(image))
        #print("Tamaño del primer tensor imagen:", image.size()) 
        #print(type(seg))
        #print(len(seg))
        #print("Tamaño del primer tensor seg:", seg.size()) 
        #print("Tamaño del primer tensor:", seg[0].size()) 
        #print(torch.min(seg), torch.max(seg))

        #output          = generate_segmentation_numpy(seg, gt, False)
        error_mae_imagen = calculate_mae(seg, image)
        mae_total.append(error_mae_imagen)
        #print(error_mae_imagen)
        #mae_total.append(error_mae_imagen)
        #iou_imagen = metric_IOU(output, gt, label)*100  # ES LO QUE USO PARA MI IF DE GUARDAR IMAGENES
        #iou_seg_total.append(round(metric_IOU(output, gt, label)*100,2))
        #iou_seg_video.append(round(metric_IOU(output, gt, label)*100,2))
        labels_total.append(label)
        label_video.append(label)
        #mascara_generada = output        
        #Este if guardo las imagenes y las mascaras
        #if (iou_imagen < 100 and label == 0):        
        #    h, w, c = imagen_original.shape
        #    image_rgb = np.zeros((h,w,c))
        #    image_rgb[:,:,2] = imagen_original[:,:,0]
        #    image_rgb[:,:,1] = imagen_original[:,:,1]
        #    image_rgb[:,:,0] = imagen_original[:,:,2]
        #    image_final = image_rgb
            #print(imagen_original.shape)      

        #    mascara_original = mascara_original*255
        #    mascara_generada = mascara_generada*255    
            #print(mascara_generada.shape)   
            #cv2.imwrite(carpeta_Normal + name_img, image_final)
            #cv2.imwrite(carpeta_Gt + name_img, mascara_original)
            #cv2.imwrite(carpeta_mascara_generada + name_img, mascara_generada)
                        
        #if label == 1: 
        #    iou_total.append(round(metric_IOU(output, gt, label)*100,2))
        #    iou_video.append(round(metric_IOU(output, gt, label)*100,2))
    data_by_dataset[_data_name] = {
        'labels': labels_total,
        'mae_errors': mae_total
    }

    iou_video         = np.array(iou_video)
    iou_seg_video     = np.array(iou_seg_video)
    label_video       = np.array(label_video)
    mae_total         = np.array(mae_total)
    acc, prec, recall, spec = metrics_mae(mae_total, label_video, _data_name)
    #print("IoU:\t\t", round(iou_video.mean(),2))
    np.save(save_path + 'iou_wp.npy',      iou_video)
    np.save(save_path + 'iou_np_wp.npy',   iou_seg_video)
    np.save(save_path + 'iou_label.npy',   label_video)

csv_folder = '/data/jefelitman_pupils/danielortiz_tesis/danielortiz/polyps/Pytorch-UNet-master/ASU-MAYO/'
os.makedirs(csv_folder, exist_ok=True)

for dataset_name, dataset_data in data_by_dataset.items():
    csv_filename = os.path.join(csv_folder, f'{dataset_name}_data.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'error'])
        for label, mae_error in zip(dataset_data['labels'], dataset_data['mae_errors']):
            writer.writerow([label, mae_error])


iou_total         = np.array(iou_total)
iou_seg_total     = np.array(iou_seg_total)
labels_total      = np.array(labels_total)
acc, prec, recall, spec = metrics_mae(iou_seg_total, labels_total, _data_name)
#print("IoU:\t\t", round(iou_total.mean(),2))
labels_fp = np.array(labels_fp)
#np.save(carpeta_fp + 'labels_fp', labels_total)


save = './results/' + model_name + '/IGHO/'
os.makedirs(save, exist_ok=True)
np.save(save + 'iou_agosto_wp.npy',      iou_total)
np.save(save + 'iou_agosto_total.npy',   iou_seg_total)
np.save(save + 'label_agosto_total.npy', labels_total)
