# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/12/18 11:43:47
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
from option.options import options, config
from data.dataloader import get_dataloader
import torch
from model.model import TextImgPersonReidNet
from loss.Id_loss import Id_Loss
from loss.RankingLoss import RankLoss
from loss.InfoNCE import CrossEn
from torch import optim
import logging
import os
from test import Inference
from torch.autograd import Variable
from utils.read_write_data import save_checkpoint, load_checkpoint
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def train(opt, network, current_epoch, test_best, train_dataloader, val_img_dataloader, val_txt_dataloader, id_loss_fun, align_loss_fun, optimizer, scheduler):
    
    for epoch in range(current_epoch+1, opt.epoch+1):

        network.train()
        network.to(opt.device)

        id_loss_sum = 0
        Align_loss_sum = 0

        for times, [image, label, captions, same_id_captions] in enumerate(train_dataloader):

            image = Variable(image.to(opt.device))
            label = Variable(label.to(opt.device))

            img_global, patch_part, txt_global, word_part = network(image, captions)

            if(opt.ID_LOSS):
                id_loss_global = id_loss_fun(img_global, txt_global, label)
            else:
                 id_loss_global = 0

            id_loss = id_loss_global
            if(opt.loss_type == 'Rank'):
                txt_global_cr = []
                for i in same_id_captions:         
                    if(i != '0'):
                        txt_global_cr2, word_part_cr2 = network.txt_embedding(i)
                        txt_global_cr.append(txt_global_cr2)                     
                align_loss = align_loss_fun(img_global, txt_global, txt_global_cr, label, epoch >= opt.epoch_begin)

            elif(opt.loss_type == 'InfoNCE'):
                sim_i_2_t = network.get_similarity(img_global, patch_part, txt_global, word_part)
                sim_t_2_i = sim_i_2_t.t()
                loss_i_2_t = align_loss_fun(sim_i_2_t)     
                loss_t_2_i = align_loss_fun(sim_t_2_i)
                align_loss = (loss_t_2_i + loss_i_2_t) / 2
            Align_loss = align_loss

            optimizer.zero_grad()
            loss = (id_loss + Align_loss)

            loss.backward()
            optimizer.step()

            if (times + 1) % 50 == 0:
                logging.info("Epoch: %d/%d Step: [%d/%d], Align_loss: %.2f, id_loss: %.2f, Lr: %.6f"
                             % (epoch, opt.epoch, times + 1, len(train_dataloader), Align_loss, id_loss, scheduler.get_lr()[0]))

            Align_loss_sum += Align_loss
            id_loss_sum += id_loss

        Align_loss_avg = Align_loss_sum / (times + 1)
        id_loss_avg = id_loss_sum / (times + 1)

        logging.info("Epoch: %d/%d , Align_loss_per_epoch: %.2f, id_loss_per_epoch: %.2f"
                     % (epoch, opt.epoch, Align_loss_avg, id_loss_avg))

        logger.info(f'Checkpoint successfully saved in {epoch} epoch!')

        print(opt.model_name)
        network.eval()
        cmc, mAP, mINP, all_infer_time, infer_time_per_query  = Inference(opt, epoch, network, val_img_dataloader, val_txt_dataloader)
        str = "Epoch: {} Testing , t2i: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, mAP: {:.4}, mINP: {:.4}".format(epoch, cmc[0], cmc[4], cmc[9], mAP, mINP)
        
        logging.info(str)
        logging.info("This model inference time is {:.2f} Seconds \n".format(all_infer_time))
        logging.info('This model inference time per query is {:.2f} ms \n'.format(infer_time_per_query * 1000))

        if(cmc[0] > test_best):
            test_best = cmc[0]
            checkpoint_folder = os.path.join(opt.save_path, 'breakpoint/')
            if(opt.ID_LOSS):
                state = {
                    'model': network.state_dict(),
                    'epoch': epoch,
                    'test_best': test_best,
                    'id_loss': id_loss_fun.state_dict(),
                }
                
            else:
                state = {
                    'model': network.state_dict(),
                    'epoch': epoch,
                    'test_best': test_best,
                    'id_loss': 0,
                }   

            save_checkpoint(state, checkpoint_folder, opt.ID_LOSS)
            
        torch.cuda.empty_cache()  
        scheduler.step()

    logging.info("Maximum GPU Memory: {:.2f} GB\n".format(torch.cuda.max_memory_allocated(opt.device) * 1.0 / 1024 / 1024/ 1024))
    logging.info('Training Done')

if __name__ == '__main__':
    opt = options().opt

    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))

    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + opt.model_name

    config(opt)

    train_dataloader = get_dataloader(opt,"train")

    test_img_dataloader, test_txt_dataloader = get_dataloader(opt,"test")

    if(opt.ID_LOSS):
        id_loss_fun = Id_Loss(opt, opt.feature_length).to(opt.device)
    else:
        id_loss_fun = 0
    if(opt.loss_type == 'Rank'):
        align_loss_fun = RankLoss(opt)
    elif(opt.loss_type == 'InfoNCE'):
        align_loss_fun  = CrossEn()

    network = TextImgPersonReidNet(opt).to(opt.device)

    test_best_path = os.path.join(opt.save_path, 'breakpoint/test_best')
    if os.path.exists(test_best_path):       
        test_best = torch.load(test_best_path)
    else:
        test_best = 0

    current_epoch_path = os.path.join(opt.save_path, 'breakpoint/current_epoch')
    if(opt.ID_LOSS):
        id_loss_path = os.path.join(opt.save_path, 'breakpoint/id_loss')

    if os.path.exists(current_epoch_path):         #training from current epoch
        current_epoch = torch.load(current_epoch_path)
        if(opt.ID_LOSS):
            id_loss = load_checkpoint(id_loss_path)
            id_loss_fun.load_state_dict(id_loss)

        checkpoint_feature_path = os.path.join(opt.save_path, 'breakpoint/best_model') 
        if os.path.exists(checkpoint_feature_path):
            model_state = load_checkpoint(checkpoint_feature_path)
            network.load_state_dict(model_state)
            
            logger.info('Network successfully loaded from the {} epoch!'.format(current_epoch))    
  
    else:   #training from epoch 0
        current_epoch = 0
        logger.info("Initialize the Network!")
    logger.info("Initialize the Network!")
    ft_params = []
    param_groups = []

    if(opt.image_fintune):
        if(opt.img_model == 'ResNet50'):
            ft_params += list(map(id, network.ImageExtract.parameters()))
        else:
            ft_params += list(map(id, network.ImageExtract.cnn.parameters())) 
    if(opt.text_fintune):
        if(opt.txt_model == 'ResNet50'):
            ft_params += list(map(id, network.TextExtract.parameters()))
        else:
            ft_params += list(map(id, network.TextExtract.textExtractor.parameters())) 


    other_params = list(filter(lambda p: id(p) not in ft_params, network.parameters()))     #backbone fc
    if(opt.ID_LOSS):
        id_loss_params = list(id_loss_fun.parameters())   

    param_groups.append({'params': other_params, 'lr': opt.lr})     #10-4 0.0001 
    if(opt.ID_LOSS):
        param_groups.append({'params': id_loss_params, 'lr': 5*opt.lr})     #5*10-4 

    if(opt.image_fintune):
        if(opt.img_model == 'ResNet50'):
            param_groups.append({'params': network.ImageExtract.parameters(), 'lr': opt.lr * 0.1})  #10-5   
        else: 
            param_groups.append({'params': network.ImageExtract.cnn.parameters(), 'lr': opt.lr * 0.1})  #10-5
    if(opt.text_fintune):
        if(opt.txt_model == 'ResNet50'):
            param_groups.append({'params': network.TextExtract.parameters(), 'lr': opt.lr * 0.1}) 
        else:
            param_groups.append({'params': network.TextExtract.textExtractor.parameters(), 'lr': opt.lr * 0.1})   


    optimizer = optim.Adam(param_groups, betas=(opt.adam_alpha, opt.adam_beta))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.epoch_decay)

    train(opt, network, current_epoch, test_best, train_dataloader, test_img_dataloader, test_txt_dataloader, id_loss_fun, align_loss_fun, optimizer, scheduler)
