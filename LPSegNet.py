import os
import torch
import torch.nn.functional  as F
from torch.autograd import Variable
from tqdm           import tqdm
from utils.utils    import clip_gradient, adjust_lr 
from loss           import mae_loss, save_images, mae_loss_val, mse_loss, save_csv, error_csv, save_error_csv #MODIFIQUE ACA
import matplotlib.pyplot as plt



def train(train_loader, val_loader, model, fp, optimizer, epoch, train_size, lr, decay_rate, decay_epoch):    
    loss_train, loss_val = [], [] #MODIFIQUE ACA
    f            = open ('logits.txt','a')
    size_rates   = [0.75, 1, 1.25]
    stream_train = tqdm(range(epoch))

    for epoch_ in stream_train:

        adjust_lr(optimizer, lr, epoch_, decay_rate, decay_epoch)

        loss_t, loss_v, loss_t_mse = 0.0, 0.0, 0.0
        loss_v_background = 0  
        loss_v_polyp = 0
        total_v_background = 0
        total_t, total_v = 0.0, 0.0 #MODIFIQUE ESTAS LINEAS
        total_v_polyp = 0
        total_v_background = 0
        aux = 0
        aux2 = 0 

        model.train()
        for i, sample in enumerate(train_loader, start=1):
            for rate in size_rates:

                # ---- data prepare ----
                images, names, names2 = sample #CAMBIOS AQUIA    
                images = Variable(images).cuda()
                #gts = Variable(gts).cuda()
 
                # label2 = label.squeeze(1)  MODIFUQE ESTO, NO ES NECESARIO LOS LABELS!!
                #label2 = label.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                #label2 = label2.type(torch.FloatTensor)
                #label2 = Variable(label2.cuda())

                # ---- rescale ----
                trainsize = int(round(train_size*rate/32)*32)
                if rate != 1:
                    images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners = True)
                    #gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners = True) AQUI

                # ----- forward step ---
               
                loss, _, loss_mse,total = forward_step('train', model, images, optimizer, i, epoch_, names=None, names2 = None)                            
                loss_t += loss.detach().cpu().numpy()
                loss_t_mse += loss_mse.detach().cpu().numpy()
                #print(loss_t_mse)
                #print("loss:",loss_t)
                
            total_t += total 
            #print("total:",total_t)   

        model.eval()
        with torch.no_grad():
            for j, sample in enumerate(val_loader, 1):
                image, names, names2 = sample
                image = Variable(image).cuda()
                #gts_v = Variable(gts_v).cuda()
                # label_v = label_.squeeze(1)
                #label_v = label_.unsqueeze(1).unsqueeze(2).unsqueeze(3) CAMBIOS AQUI
                #label_v = label_v.type(torch.FloatTensor)
                #label_v = Variable(label_v.cuda())

                _, loss_val, _, total = forward_step('validation', model, image, optimizer, i, epoch_, names, names2)          
                mae_loss_val_background, mae_loss_val_polyp = loss_val
                #print(type(mae_loss_val_polyp))
                loss_v_background += mae_loss_val_background.detach().cpu().numpy()
                loss_v_polyp += mae_loss_val_polyp.detach().cpu().numpy()
                #print(loss_v_background, loss_v_polyp)
                total_v_background += total
                total_v_polyp += total
                


                #total_v += total

        #iouTrain = round(100*iou_t/(total_t*3),2)
        #iouVal   = round(100*iou_v/(total_v),2)
        #print("LOSS:", loss_t)
        #print("TOTAL",loss_t/i)
        #print("Valor i:", i)
        #print("Valor j:", j)

        
        stream_train.set_description("epoch: {} TrainLoss_MAE: {} \t TrainLoss_MSE: {} \tValLoss_Polyp: {} \t ValLoss_background: {}  ".format(epoch_+1,
                                    round(loss_t/i,2), round(loss_t_mse/i,2), round(loss_v_polyp/j,2), round(loss_v_background/j,2)))

        f.write("\nepoch: {}\t TrainLoss_MAE: {} \t TrainLoss_MSE: {} \t ValLoss_Polyp: {} \t ValLoss_background: {} ".format(epoch_+1,
                 round(loss_t/i,2), round(loss_t_mse/i,2), round(loss_v_polyp/j,2), round(loss_v_background/j,2)))


        loss_train.append(loss_t/i)
        loss_val.append(loss_v/j)
        #iou_train.append(iouTrain)
        #iou_val.append(iouVal)

        # ----- save weights -----
        # save_path = 'snapshots/KQV_29/'
        # os.makedirs(save_path, exist_ok=True)
        # torch.save(model.state_dict(), save_path + '%d.pth' % epoch_)
        # print('[Saving Snapshot:]', save_path + '%d.pth'% epoch_)


    print('Training Complete!')
    f.close()
    return model, loss_train


def forward_step(mode, model, images, optimizer, i, epoch, names, names2): #AQUI
  
    # ---- forward ----
    loss = 0
    loss_mse = 0
    loss_val = [0, 0]
    optimizer.zero_grad()
    pcm, pcm2, pcm_prom = model(images) #MODIFIQUE ACA pcm2 es la salida de gamma con una conv1x1, pcm es la salida de la conv 1x1x3
                                        #pcm_prom es la concatenacion de gama subida que sera promediada
    #print(pcm.shape)
    #print(pcm2.shape)
      

    #loss_cls           = F.binary_cross_entropy_with_logits(classifcation, label2, reduce='none')
    mae_loss_value =  mae_loss(images, pcm)   
    mse_loss_value = mse_loss(images, pcm)

    #print(mae_loss_value)
    Lista = []
    if names is not None and len(names) > 0:
        #print(names2)
        mae_loss_val_background, mae_loss_val_polyp, dictionary = mae_loss_val(images, pcm, names, names2)
        loss_val[0] = mae_loss_val_background
        loss_val[1] = mae_loss_val_polyp
        if(epoch == 79): #CAMBIAR Cuando cambian los epochs, si son 80 colocar 79
            #filename = 'error4.csv'
            #path = '/data/danielortiz/danielortiz/polyps/csv_error/' 
            #save_csv(dictionary, filename, path)
            #save_images(images, pcm, pcm2, names2, pcm_prom) 
            print("Lote", dictionary ) 
            
            #Lista.append(error_csv(images, pcm, names))
        #print()
        #save_error_csv(Lista)

 
     # ---- backward ---- 
    total = images.size(0)
    #print(total)
    loss  = mae_loss_value
    loss_mse = mse_loss_value
    #print(len(mae_list))

    if mode == 'train':
        loss.backward()
        clip_gradient(optimizer, grad_clip = 0.5)
        optimizer.step()
    # else: 
        # plt.imsave("val/rfb.png",  torch.round(torch.sigmoid(pcm)).detach().cpu().numpy()[0][0], cmap = 'gray')
    return loss, loss_val , loss_mse, total