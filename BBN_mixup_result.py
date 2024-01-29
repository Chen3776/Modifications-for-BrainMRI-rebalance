
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import utils
from scheduler import create_scheduler
from optim.optim_factory_kad import create_optimizer
from dataset.dataset import MedKLIP_Dataset
# from models.model_MedKLIP import MedKLIP
from models.model_MedKLIP_before_fuse import MedKLIP as MedKLIP
from models.before_fuse import *
# from models.model_MedKLIP_attention_14class import MedKLIP as MedKLIP_14_atten

from models.tokenization_bert import BertTokenizer

from models.imageEncoder import ModelRes, ModelDense
from models.VIT_image_encoder.VIT_ie import VIT_ie
from transformers import AutoModel,AutoTokenizer

from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
import torch.nn.functional as F
from loss.loss import *
from dataset.sampler import *

def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if using multi-GPU.
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_tokenizer(tokenizer,target_text):
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length=128,return_tensors="pt")
    return target_tokenizer

def get_text_features(model,text_list,tokenizer,device,max_length):
    # text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    target_tokenizer = tokenizer(list(text_list), padding='max_length', truncation=True, max_length=max_length,return_tensors="pt").to(device)
    # text_features = model.encode_text(text_token)
    text_features = model(input_ids = target_tokenizer['input_ids'],attention_mask = target_tokenizer['attention_mask'])#(**encoded_inputs)
    text_features = text_features.last_hidden_state[:,0,:]
    # text_features = F.normalize(text_features, dim=-1)
    return text_features

def gen_entity_labels(entity):
    # entity [(b,1),(b,1),...]
    entity_labels = []
    for group in entity:
        b = len(group)
        a=np.eye(b)
        for i in range(b):
            text1 = sorted(group[i].split('[SEP]'))
            if len(text1) == 1 and text1[0] == "unspecified":
                continue
            for j in range(i+1,b):
                text2 = sorted(group[j].split('[SEP]'))
                a[i][j] = 1 if len(text1) == len(text2) and text1 == text2 else 0
                a[j][i] = a[i][j]
        c=1/a.sum(axis=-1)
        d = np.array([a[idx]*c[idx] for idx in range(len(a))])
        entity_labels.append(d)
    return torch.tensor(np.array(entity_labels))

def _get_bert_basemodel(bert_model_name):
    try:
        model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
        print("text feature extractor:", bert_model_name)
    except:
        raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

    for param in model.parameters():
        param.requires_grad = False

    return model


def compute_AUCs(gt, pred, n_class):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    for i in range(n_class):
        cur_gt = gt_np[:,i]
        cur_pred = pred_np[:,i]
        Mask = (( cur_gt!= -1) & ( cur_gt != 2)).squeeze()
        cur_gt = cur_gt[Mask]
        cur_pred = cur_pred[Mask]
        if (not 1 in cur_gt) or (not 0 in cur_gt):
            AUROCs.append(-1)
        else:
            AUROCs.append(roc_auc_score(cur_gt, cur_pred))
    return AUROCs

def evaluate(tensorboard):
    gt, pred = tensorboard["gt"],tensorboard["pred"]
    AUROCs = np.array(compute_AUCs(gt, pred,len(target_class)))
    mean_AUROC = np.mean(AUROCs)
    max_f1s = []
    accs = []
    for i in range(len(target_class)):
        gt_np = gt[:, i].cpu().numpy()
        pred_np = pred[:, i].detach().cpu().numpy() 
        Mask = (( gt_np!= -1) & ( gt_np != 2)).squeeze()
        gt_np = gt_np[Mask]
        pred_np = pred_np[Mask]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision # dot multiply for list
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np>max_f1_thresh))
    return AUROCs,accs,max_f1s,mean_AUROC

def get_weights(labels,Pc_list,Nc_list):

    score_list = []
    for i in range(labels.shape[0]):
        score_pos = 0.0
        score_neg = 0.0
        score_num_pos = 0
        score_num_neg = 0
        for c in range(len(Pc_list)):
            if labels[i,c] in [0, 1]:
                if labels[i,c] == 1:
                    score_pos += Pc_list[c]
                    score_num_pos += 1
                else:
                    score_neg += Nc_list[c]
                    score_num_neg += 1
        
        score = score_pos / score_num_pos + score_neg / score_num_neg
        score_list.append(score)
    weights = [1/x for x in score_list]
    return score_list, weights

def get_sample_wise_weight_from_auc_result(labels, auc_list):

    weights = []
    for i in range(labels.shape[0]):
        weight_pos = 0.0
        num_pos = 0
        for c in range(len(auc_list)):
            if labels[i,c] in [0, 1]:
                if labels[i,c] == 1:
                    weight_pos += 1 - auc_list[c]
                    num_pos += 1
        weight = weight_pos / num_pos
        weights.append(weight)

    return weights

def plot_weight(weights,save_path):
    sorted_weights = np.sort(weights)[::-1]
    normalized_weights = [x / sum(sorted_weights) for x in sorted_weights]
    plt.figure(figsize=(10, 6))  # 设置图形的大小
    plt.plot(normalized_weights, label='Weights')  # 绘制排序后的weights
    plt.title('Weights')
    plt.xlabel('Index')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.savefig(save_path)
    
def mixup_data(images1, images2, labels1, labels2, entity_features, device, alpha=1.0):
    '''Applies Mixup augmentation to a batch of images and labels.
    
    Args:
      x: Batch of images, a tensor of shape [B, C, H, W].
      y: Batch of labels, a tensor of shape [B].
      alpha: Mixup interpolation coefficient, float (default: 1.0).
    
    Returns:
      mixed_x: The mixed images.
      mixed_y: The mixed labels.
      lambda: The mixup ratio used for mixing.
    '''

    if alpha > 0:
        lambda_1 = np.random.beta(alpha, alpha)
        lambda_2 = np.random.beta(alpha, alpha)
    else:
        lambda_1 = lambda_2 = 1

    lambda_1 = max(lambda_1, 1 - lambda_1)
    lambda_2 = max(lambda_2, 1 - lambda_2)

    # half batch size = 8
    half_batch_size = images1[0].size()[0]
    index_sub1 = torch.randperm(half_batch_size).to(device)
    index_sub2 = torch.randperm(half_batch_size).to(device)
    # index = torch.randperm(half_batch_size * 2).to(device)

    mixed_images = []
    mixed_entity_features = []

    for i in range(len(images1)):
        images1[i] = images1[i].to(device)
        images2[i] = images2[i].to(device)

        mixed_image_subset1 = lambda_1 * images1[i] + (1 - lambda_1) * images2[i][index_sub2, :]
        mixed_image_subset2 = lambda_2 * images2[i] + (1 - lambda_2) * images1[i][index_sub1, :]
        mixed_image = torch.cat((mixed_image_subset1, mixed_image_subset2), dim=0)
        mixed_images.append(mixed_image)
    
    negative_label_mask1 = (labels1 == -1) | (labels2[index_sub2] == -1)    # mask of first half
    negative_label_mask2 = (labels2 == -1) | (labels1[index_sub1] == -1)    # mask of second half
    negative_label_mask = torch.cat((negative_label_mask1, negative_label_mask2), dim=0)

    mixed_labels_subset1 = lambda_1 * labels1 + (1 - lambda_1) * labels2[index_sub2] 
    mixed_labels_subset2 = lambda_2 * labels2 + (1 - lambda_2) * labels1[index_sub1]
    mixed_labels = torch.cat((mixed_labels_subset1, mixed_labels_subset2), dim=0)
    
    mixed_labels[negative_label_mask] = -1

    # entity_features: len 5 list, with 5 tensor shape 16, 768
    for i in range(len(entity_features)):

        mixed_subset1 = lambda_1 * entity_features[i][:8, :] + (1 - lambda_1) * entity_features[i][-8:, :]
        mixed_subset2 = lambda_2 * entity_features[i][-8:, :] + (1 - lambda_2) * entity_features[i][:8, :]
        mixed_entity_feature = torch.cat((mixed_subset1, mixed_subset2), dim=0)
        mixed_entity_features.append(mixed_entity_feature)

    return mixed_images, mixed_labels, mixed_entity_features, lambda_1, lambda_2

def train(model, image_encoder, text_encoder, fuseModule, tokenizer, datasets, Pc_list, Nc_list, optimizer, epoch, warmup_steps, device, scheduler, config, auc_list):
    mask_modal = config['mask_modal'] if 'mask_modal' in config else ""
    clip_loss = ClipLoss(mask_modal=mask_modal)
    model.train()
    if config['4_image_encoder']:
        for idx in range(len(image_encoder)):
            image_encoder[idx].train()
    else:
        image_encoder.train()

    label_list = np.load(config['train_label_file'])
    # train_sampler = UniformSampler(datasets,config['batch_size'],batch_clas_num=8) if 'uniform_sample' in config and config['uniform_sample'] else None
    # _ , weights = get_weights(label_list,Pc_list,Nc_list)
    weights = get_sample_wise_weight_from_auc_result(label_list, auc_list)

    if epoch % 5 == 1:
        Path(os.path.join(args.output_dir,'weight_figure')).mkdir(parents=True, exist_ok=True)
        save_weights_figure_path = os.path.join(args.output_dir,'weight_figure', f'weights_plot_epoch_{epoch}.png')
        plot_weight(weights,save_weights_figure_path)
    
    uniform_sampler = WeightedRandomSampler(weights=torch.ones(len(weights)), num_samples=len(weights)//2, replacement=True)
    weighted_sampler = WeightedRandomSampler(weights, num_samples=len(weights)//2, replacement=True)
    uniform_loader = DataLoader(
        datasets,
        batch_size=config['train_batch_size']//2,
        num_workers=4,
        pin_memory=True,
        sampler=uniform_sampler,
        shuffle=False,
        collate_fn=None,
        drop_last=True,
    )     
    weighted_loader = DataLoader(
        datasets,
        batch_size=config['train_batch_size']//2,
        num_workers=4,
        pin_memory=True,
        sampler=weighted_sampler,
        shuffle=False,
        collate_fn=None,
        drop_last=True,
    )     

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cl', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_clip', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(loss_ce=1.0)
    metric_logger.update(loss_cl=1.0)
    metric_logger.update(loss_clip=1.0)
    metric_logger.update(lr = optimizer.param_groups[0]['lr'])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 1   
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(uniform_loader)

    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)

    tensorboard = {
        "train_loss":[],
        "train_loss_ce":[],
        "train_loss_ce_former":[],
        "train_loss_ce_latter":[],
        "train_loss_cl":[],
        "train_loss_clip":[],
        "gt":gt,
        "pred":pred
    }

    json_book = json.load(open(config['disease_book'],'r'))
    json_order=json.load(open(config['disease_order'],'r'))
    disease_book = [json_book[i] for i in json_order]
    # disease_book = json_order
    ana_order=json.load(open(config['anatomy_order'],'r'))
    ana_book = [ 'It is located at ' + i for i in ana_order]

    text_features = get_text_features(text_encoder,disease_book,tokenizer,device,max_length=128)
    ana_features = get_text_features(text_encoder,ana_book,tokenizer,device,max_length=128)

    cl_excluded_disease = ['normal']
    if "exclude_class" in config and config["exclude_class"]:
        cl_excluded_disease += config["exclude_classes"]
        keep_class_dim = [json_order.index(i) for i in json_order if i not in config["exclude_classes"] ]   
    cl_class_dim = [json_order.index(i) for i in json_order if i not in cl_excluded_disease]
    
    # for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for (i, (sample1, sample2)) in enumerate(zip(uniform_loader, weighted_loader)):
        images1 = sample1['image']  # [(b,x,y,z),(b,x,y,z)]
        labels1 = sample1['label'].to(device)
        entity1 = sample1['entity'] # [(b,a1),(b, a2),...]
        images2 = sample2['image']  # [(b,x,y,z),(b,x,y,z)]
        labels2 = sample2['label'].to(device)
        entity2 = sample2['entity'] # [(b,a1),(b, a2),...]
        
        images = [torch.cat((a, b), dim=0) for a, b in zip(images1, images2)]
        labels = torch.cat((labels1,labels2),dim=0)
        entity = [a+b for a, b in zip(entity1, entity2)]

        B = labels.shape[0]

        cur_text_features = text_features.unsqueeze(0).repeat(B,1,1)
        cur_ana_features = ana_features.unsqueeze(0).repeat(B,1,1)
        entity_labels = gen_entity_labels(entity).float().to(device)
        entity_features = []
        for en in entity:
            entity_features.append(get_text_features(text_encoder,en,tokenizer,device,max_length=128)) # [(b,d),(b,d)]
        
        images, labels, entity_features, _a, _b = mixup_data(images1, images2, labels1, labels2, entity_features, device, alpha=1)
        
        optimizer.zero_grad()
        
        # entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)
        image_features = [] # image_features 4 b n d, image_features_pool 4 b d
        image_features_pool = []
        for idx,cur_image in enumerate(images):
            cur_image = cur_image.to(device)
            if config['4_image_encoder']:
                cur_image_encoder = image_encoder[idx]
                image_feature,image_feature_pool = cur_image_encoder(cur_image)
            else:
                image_feature,image_feature_pool = image_encoder(cur_image) 
            image_features.append(image_feature)
            image_features_pool.append(image_feature_pool)
        
        # before fuse
        fuse_image_feature, fuse_image_feature_pool = fuseModule(image_features)
        image_features_pool.append(fuse_image_feature_pool)

        if config['no_cl'] == False:
            logits, ll, cl_labels = model(fuse_image_feature,cur_text_features,cur_ana_features)
        else:
            logits = model(fuse_image_feature,cur_text_features,cur_ana_features)
        # logits.shape torch.Size([112, 1]) torch.Size([112, 1]) torch.Size([112])
        # ll.shape torch.Size([104, 8]) torch.Size([104]) torch.Size([104])
        # print('logits',logits.shape)

        tensorboard["gt"] = torch.cat((tensorboard["gt"], labels), 0)

        cl_mask_labels = labels[:,cl_class_dim]
        cl_mask_labels = cl_mask_labels.reshape(-1,1) # b*left_class_num,1

        B = labels.shape[0]

        if "exclude_class" in config and config["exclude_class"]:
            labels = labels[:,keep_class_dim]
        
        pred_class = logits.reshape(-1,len(all_target_class))
        
        # if config['num_classes']>13:
        #     label_former = labels[:,:13].reshape(-1,1)
        #     logit_former = pred_class[:,:13].reshape(-1,1)
        #     label_latter = labels[:,13:].reshape(-1,1)
        #     logit_latter = pred_class[:,13:].reshape(-1,1)
        #     # print("shape",label_former.shape,logit_former.shape,label_latter.shape,logit_latter.shape)
        #     Mask1 = ((label_former != -1) & (label_former != 2)).squeeze()
        #     label_former = label_former[Mask1].float()
        #     logit_former = logit_former[Mask1]
        #     Mask2 = ((label_latter != -1) & (label_latter != 2)).squeeze()
        #     label_latter = label_latter[Mask2].float()
        #     logit_latter = logit_latter[Mask2]
        #     loss_ce_former = F.binary_cross_entropy(logit_former[:,0],label_former[:,0])
        #     loss_ce_latter = F.binary_cross_entropy(logit_latter[:,0],label_latter[:,0])

        labels = labels.reshape(-1,1) # b*class_num,1

        Mask = ((labels != -1) & (labels != 2)).squeeze()

        labels = labels[Mask].float()
        logits = logits[Mask]

        # cur_target_class= target_class[Mask]

        # if config["la"]:
        #     class_p = class_p.unsqueeze(0).repeat(B,1,1)
        #     class_p = class_p.reshape(-1,class_p.shape[-1])
        #     logits = logits + class_p

        loss_ce = F.binary_cross_entropy(logits[:,0],labels[:,0])

        cl_mask = (cl_mask_labels == 1).squeeze()
        if config['no_cl'] == False:
            cl_labels = cl_labels[cl_mask].long()
            ll = ll[cl_mask]
            loss_cl = F.cross_entropy(ll,cl_labels)
        else:
            loss_cl = torch.tensor(0).to(device)

        # if args.use_entity_features:
        #     pred_class_text = model(entity_features.unsqueeze(1),text_features)
        #     loss_ce_text = ce_loss(pred_class_text.view(-1,2),label.view(-1))
        #     loss_ce = loss_ce_image + loss_ce_text
        # else:
        #     loss_ce = loss_ce_image

        if config['kad']:
            loss_clip = clip_loss(image_features_pool, entity_features, entity_labels) # 4 b d; 4 b d; 4 b b
            # loss_clip = clip_loss(fuse_image_feature_pool, entity_features, entity_labels)
        else:
            loss_clip = torch.tensor(0).to(device)

        loss_ce_ratio = config['ce_loss_ratio'] if 'ce_loss_ratio' in config else 1
        loss = loss_ce * loss_ce_ratio + loss_cl + loss_clip * config['kad_loss_ratio']

        
        # pred_class = logits.reshape(-1,len(target_class))
        # pred_class = pred_class[:,:,0]
        tensorboard["pred"] = torch.cat((tensorboard["pred"], pred_class), 0)
        tensorboard["train_loss"].append(loss.item())
        tensorboard["train_loss_ce"].append(loss_ce.item())
        if config['no_cl'] == False:
            tensorboard["train_loss_cl"].append(loss_cl.item())
        if config['kad']:
            tensorboard["train_loss_clip"].append(loss_clip.item())
        # if config['num_classes']>13:
        #     tensorboard["train_loss_ce_former"].append(loss_ce_former.item())
        #     tensorboard["train_loss_ce_latter"].append(loss_ce_latter.item())

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        
        scalar_step += 1
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_cl=loss_cl.item())
        metric_logger.update(loss_clip=loss_clip.item())

        # if epoch==0 and warmup_iterations!=0 and i%step_size==0 and i<=warmup_iterations: 
        #     scheduler.step(i//step_size)         
        metric_logger.update(lr = optimizer.param_groups[0]['lr'])

    # get mean loss for the epoch
    for i in tensorboard:
        if i == "gt" or i == "pred":
            continue
        tensorboard[i]=np.array(tensorboard[i]).mean() if len(tensorboard[i]) else 0
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, tensorboard #,loss_epoch.mean()

def valid(model, image_encoder, text_encoder, fuseModule, tokenizer, datasets, device,config):
    mask_modal = config['mask_modal'] if 'mask_modal' in config else ""
    clip_loss = ClipLoss(mask_modal=mask_modal)
    model.eval()
    if config['4_image_encoder']:
        for idx in range(len(image_encoder)):
            image_encoder[idx].eval()
    else:
        image_encoder.eval()
        
    data_loader = DataLoader(
        datasets,
        batch_size=config['val_batch_size'],
        num_workers=4,
        pin_memory=True,
        sampler=None,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
    )
    
    # temp = nn.Parameter(torch.ones([]) * config['temp'])   
    # val_scalar_step = epoch*len(data_loader)
    # val_loss = []
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    tensorboard = {
        # "val_loss":[],
        "val_loss_ce":[],
        # "val_loss_ce_former":[],
        # "val_loss_ce_latter":[],
        # "val_loss_cl":[],
        # "val_loss_clip":[],
        "gt":gt,
        "pred":pred
    }

    json_book = json.load(open(config['disease_book'],'r'))
    json_order=json.load(open(config['disease_order'],'r'))
    disease_book = [json_book[i] for i in json_order]
    # disease_book = json_order
    ana_order=json.load(open(config['anatomy_order'],'r'))
    ana_book = [ 'It is located at ' + i for i in ana_order]

    text_features = get_text_features(text_encoder,disease_book,tokenizer,device,max_length=128)
    ana_features = get_text_features(text_encoder,ana_book,tokenizer,device,max_length=128)

    cl_excluded_disease = ['normal']
    if "exclude_class" in config and config["exclude_class"]:
        cl_excluded_disease += config["exclude_classes"]
        keep_class_dim = [json_order.index(i) for i in json_order if i not in config["exclude_classes"] ]   
    cl_class_dim = [json_order.index(i) for i in json_order if i not in cl_excluded_disease]  
    
    Pc_pred = [0.0 for _ in range(len(json_order))]
    Pc_num = [0 for _ in range(len(json_order))]
    Nc_pred = [0.0 for _ in range(len(json_order))]
    Nc_num = [0 for _ in range(len(json_order))]

    for i, sample in enumerate(data_loader):

        images = sample['image']  # [(b,x,y,z),(b,x,y,z)]
        labels = sample['label'].to(device)
        # index = sample['index'].to(device)
        entity = sample['entity'] # [(b,a1),(b, a2),...]
        # report = sample["report"] # [b,whole]
        
        B = labels.shape[0]
        C = labels.shape[1]
        with torch.no_grad():
            cur_text_features = text_features.unsqueeze(0).repeat(B,1,1)
            cur_ana_features = ana_features.unsqueeze(0).repeat(B,1,1)
            entity_labels = gen_entity_labels(entity).float().to(device)
            entity_features = []
            for en in entity:
                entity_features.append(get_text_features(text_encoder,en,tokenizer,device,max_length=128)) # [(b,d),(b,d)]

            image_features = [] # image_features 4 b n d, image_features_pool 4 b d
            image_features_pool = []
            for idx,cur_image in enumerate(images):
                cur_image = cur_image.to(device)
                if config['4_image_encoder']:
                    cur_image_encoder = image_encoder[idx]
                    image_feature,image_feature_pool = cur_image_encoder(cur_image)
                else:
                    image_feature,image_feature_pool = image_encoder(cur_image) 
                image_features.append(image_feature)
                image_features_pool.append(image_feature_pool)
            
            # before fuse
            fuse_image_feature, fuse_image_feature_pool = fuseModule(image_features)
            image_features_pool.append(fuse_image_feature_pool)
            
            if config['no_cl'] == False:
                logits, ll, cl_labels = model(fuse_image_feature,cur_text_features,cur_ana_features)
            else:
                logits = model(fuse_image_feature,cur_text_features,cur_ana_features)
            
            tensorboard["gt"] = torch.cat((tensorboard["gt"], labels), 0)

            cl_mask_labels = labels[:,cl_class_dim]
            cl_mask_labels = cl_mask_labels.reshape(-1,1) # b*left_class_num,1

            if "exclude_class" in config and config["exclude_class"]:
                labels = labels[:,keep_class_dim]

            pred_class = logits.reshape(-1,len(all_target_class))

            # if config['num_classes']>13:
            #     label_former = labels[:,:13].reshape(-1,1)
            #     logit_former = pred_class[:,:13].reshape(-1,1)
            #     label_latter = labels[:,13:].reshape(-1,1)
            #     logit_latter = pred_class[:,13:].reshape(-1,1)
            #     Mask1 = ((label_former != -1) & (label_former != 2)).squeeze()
            #     label_former = label_former[Mask1].float()
            #     logit_former = logit_former[Mask1]
            #     Mask2 = ((label_latter != -1) & (label_latter != 2)).squeeze()
            #     label_latter = label_latter[Mask2].float()
            #     logit_latter = logit_latter[Mask2]
            #     loss_ce_former = F.binary_cross_entropy(logit_former[:,0],label_former[:,0])
            #     loss_ce_latter = F.binary_cross_entropy(logit_latter[:,0],label_latter[:,0])
            
            labels = labels.reshape(-1,1) # b*class_num,1
            
            for idx in range(labels.shape[0]):
                if labels[idx,0] == 1:
                    Pc_pred[idx % C] += logits[idx,0]
                    Pc_num[idx % C] += 1
                elif labels[idx,0] == 0:
                    Nc_pred[idx % C] += 1-logits[idx,0]
                    Nc_num[idx % C] += 1

            Mask = ((labels != -1) & (labels != 2)).squeeze()

            labels = labels[Mask].float()
            logits = logits[Mask]

            loss_ce = F.binary_cross_entropy(logits[:,0],labels[:,0])

            # cl_mask = (cl_mask_labels == 1).squeeze()

            # if config['no_cl'] == False:
            #     cl_labels = cl_labels[cl_mask].long()
            #     ll = ll[cl_mask]
            #     loss_cl = F.cross_entropy(ll,cl_labels)
            # else:
            #     loss_cl = torch.tensor(0).to(device)

            # if config['kad']:
            #     loss_clip = clip_loss(image_features_pool, entity_features, entity_labels) # 4 b d; 4 b d; 4 b b
            #     # loss_clip = clip_loss(fuse_image_feature_pool, entity_features, entity_labels)
            # else:
            #     loss_clip = torch.tensor(0).to(device)
    
            # loss_ce_ratio = config['ce_loss_ratio'] if 'ce_loss_ratio' in config else 1
            # loss = loss_ce * loss_ce_ratio + loss_cl + loss_clip * config['kad_loss_ratio']

            
            # pred_class = pred_class[:,:,0]
            tensorboard["pred"] = torch.cat((tensorboard["pred"], pred_class), 0)
            # val_loss.append(loss.item())
            # tensorboard["val_loss"].append(loss.item())
            tensorboard["val_loss_ce"].append(loss_ce.item())
            # if config['no_cl'] == False:
            #     tensorboard["val_loss_cl"].append(loss_cl.item())
            # if config['kad']:
            #     tensorboard["val_loss_clip"].append(loss_clip.item())
            # if config['num_classes']>13:
            #     tensorboard["val_loss_ce_former"].append(loss_ce_former.item())
            #     tensorboard["val_loss_ce_latter"].append(loss_ce_latter.item())
    
    # Pc_list = []
    # Nc_list = []
    # for i in range(len(Pc_num)):
    #     if Pc_num[i] == 0:
    #         print('num error')
    #         Pc_list.append(0.5)
    #     else:
    #         Pc_list.append((Pc_pred[i] / Pc_num[i]).item())
            
    # for i in range(len(Nc_num)):
    #     if Nc_num[i] == 0:
    #         print('num error')
    #         Nc_list.append(0.5)
    #     else:
    #         Nc_list.append((Nc_pred[i] / Nc_num[i]).item())
        
    for i in tensorboard:
        if i == "gt" or i == "pred":
            continue
        tensorboard[i]=np.array(tensorboard[i]).mean() if len(tensorboard[i]) else 0

    return tensorboard

def main(args, config):
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type('torch.FloatTensor')
    # cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs'] if 'warmup_epochs' in config['schedular'] else 0

    #### Dataset #### 
    print("Creating dataset")
    print("train file",config['train_file'])
    print("valid file",config['valid_file'])
    augment = True if 'augment' in config and config['augment'] else False
    mask_modal = config['mask_modal'] if 'mask_modal' in config else ""
    train_datasets = MedKLIP_Dataset(config['train_file'],config['label_file'],config['dis_label_file'],config['report_observe'], mode = 'train', augmentation=augment,mask_modal=mask_modal)
    
    val_datasets = MedKLIP_Dataset(config['valid_file'],config['label_file'],config['dis_label_file'],config['report_observe'],mode ='train',mask_modal=mask_modal)

    print("Creating model")

    if config['text_encoder'].startswith("emi"):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config['text_encoder'])

    if config['model_type']== 'resnet':
        image_encoder = ModelRes(config).to(device)
    elif config['model_type'] == 'densenet':
        image_encoder = ModelDense(config).to(device)
    elif config['model_type'] == 'VIT':
        image_encoder = VIT_ie(config).to(device)

    text_encoder = _get_bert_basemodel(config['text_encoder']).to(device)

    model = MedKLIP(config)
    
    fuseModule = beforeFuse(config).to(device) # before fusion

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(model, device_ids) 
    # model = model.module
    model = model.cuda(device=device_ids[0])

    image_encoder = nn.DataParallel(image_encoder, device_ids) 
    # image_encoder = image_encoder.module
    image_encoder = image_encoder.cuda(device=device_ids[0])

    if len(args.finetune_checkpoint):    
        checkpoint = torch.load(args.finetune_checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        for name, param in model.named_parameters():
            if "classifier" in name:
                print(name)
                param.requires_grad = True
                print("init",name)
                if 'weight' in name:
                    param.data.normal_(mean=0.0, std=0.02)
                elif 'bias' in name:
                    torch.nn.init.constant_(param,0)
                else:
                    print("param.shape",param.shape)
                    for i in range(len(param)):
                        torch.nn.init.normal_(param[i], mean=0.0, std=0.02)
            else:
                param.requires_grad = False 
        print('load finetune checkpoint from %s'%args.finetune_checkpoint)

    arg_opt = utils.AttrDict(config['optimizer'])
    # optimizer: {opt: adamW, lr: 1e-4, weight_decay: 0.02}
    # schedular: {sched: cosine, lr: 1e-4, epochs: 100, min_lr: 1e-5, decay_rate: 1, warmup_lr: 1e-5, warmup_epochs: 5, cooldown_epochs: 0}
    optimizer = create_optimizer(arg_opt, model, image_encoder, text_encoder, fuseModule)
    # optimizer = nn.DataParallel(optimizer,device_ids)
    # model = model.cuda(device=device_ids[0])
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    
    print("Start training")
    start_time = time.time()

    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))

    best_val_auc = float("-inf")
    
    dis_list = json.load(open(config['disease_order'],'r'))
    Pc_list = [1 / len(dis_list)] * len(dis_list)
    Nc_list = [1 / len(dis_list)] * len(dis_list)

    result_auc = [0.1] * 13

    for epoch in range(start_epoch, max_epoch):
        # if epoch>0:
        #     lr_scheduler.step(epoch+warmup_steps)
        # if epoch == 0 and warmup_steps == 0:
        #     lr_scheduler.step(epoch)

        lr_scheduler.step(epoch)
        
        train_stats, tensorboard_train = train(model, image_encoder, text_encoder, fuseModule, tokenizer, train_datasets, Pc_list, Nc_list, optimizer, epoch, warmup_steps, device, lr_scheduler, config, result_auc)

        writer.add_scalar('lr/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)

        tensorboard_val = valid(model, image_encoder, text_encoder, fuseModule, tokenizer, val_datasets, device, config)
        writer.add_scalars('loss_epoch',{'train_loss':tensorboard_train["train_loss"]}, epoch)
        
        # if config['num_classes']>13:
        #     content = {'train_loss':tensorboard_train["train_loss_ce"],\
        #                 "train_dis_loss":tensorboard_train["train_loss_ce_former"],\
        #                 "train_other_loss":tensorboard_train["train_loss_ce_latter"],\
        #                 "val_loss":tensorboard_val["val_loss_ce"],\
        #                 "val_dis_loss":tensorboard_val["val_loss_ce_former"],\
        #                 "val_other_loss":tensorboard_val["val_loss_ce_latter"]}
        #     writer.add_scalars('loss_ce_epoch',content, epoch)
        # else:
        writer.add_scalars('loss_ce_epoch',{'train_loss':tensorboard_train["train_loss_ce"],"val_loss":tensorboard_val["val_loss_ce"]}, epoch)
        
        if config['no_cl'] == False:
            writer.add_scalars('loss_cl_epoch',{'train_loss':tensorboard_train["train_loss_cl"]}, epoch)
        if config['kad']:
            writer.add_scalars('loss_clip_epoch',{'train_loss':tensorboard_train["train_loss_clip"]}, epoch)
        if 'global_local_loss' in config and config['global_local_loss']:
            writer.add_scalars('loss_global_local_epoch',{'train_loss':tensorboard_train["train_loss_global_local"]}, epoch)

        if utils.is_main_process():  
            image_encoder_params = [cur.state_dict() for cur in image_encoder] if config['4_image_encoder'] else image_encoder.state_dict()
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': tensorboard_val["val_loss_ce"],
                         'Pc_list': [round(num,2) for num in Pc_list], 'Nc_list': [round(num,2) for num in Nc_list]
                        }                     
            save_obj = {
                'model': model.state_dict(),
                'image_encoder': image_encoder_params,
                'fuseModule': fuseModule.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_state.pth'))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # train_metric = evaluate(tensorboard_train)
        val_metric = evaluate(tensorboard_val)
        # train_roc = {i:train_metric[0][idx] for idx,i in enumerate(target_class)}
        # train_acu = {i:train_metric[1][idx] for idx,i in enumerate(target_class)}
        # train_f1 = {i:train_metric[2][idx] for idx,i in enumerate(target_class)}
        val_roc = {i:val_metric[0][idx] for idx,i in enumerate(target_class)}
        val_acu = {i:val_metric[1][idx] for idx,i in enumerate(target_class)}
        val_f1 = {i:val_metric[2][idx] for idx,i in enumerate(target_class)}
        val_mean_auc = val_metric[3]
        # writer.add_scalars('train_metric/roc',train_roc, epoch)
        # writer.add_scalars('train_metric/acu',train_acu, epoch)
        # writer.add_scalars('train_metric/f1',train_f1, epoch)
        writer.add_scalars('val_metric/roc',val_roc, epoch)
        writer.add_scalars('val_metric/acu',val_acu, epoch)
        writer.add_scalars('val_metric/f1',val_f1, epoch)
        writer.add_scalar('val_mean_auc',val_mean_auc, epoch)

        # if epoch % 10 == 1 and epoch>1:
        if utils.is_main_process() and best_val_auc < val_mean_auc:
            image_encoder_params = image_encoder.state_dict()
            save_obj = {
                'model': model.state_dict(),
                'image_encoder': image_encoder_params,
                'fuseModule': fuseModule.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'best_val.pth')) 
            best_val_auc = val_mean_auc
            print("save best",epoch)

        result_auc = list(val_roc.values())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/remote-home/pengyichen/baseline_1/CODE/Pretrain/configs/config_one_fifth.yaml')
    parser.add_argument('--finetune_checkpoint', default='')
    parser.add_argument('--output_dir', default='/remote-home/pengyichen/baseline_1/CODE/Pretrain/output_dir/output_BBN_mixup_result')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='3', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ['NCCL_IB_DISABLE'] = '1'
    # os.environ['NCCL_P2P_DISABLE'] = '1'
    if args.gpu !='-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    all_target_class = json.load(open(config['disease_order'],'r'))
    target_class = all_target_class.copy()
    if "exclude_class" in config and config["exclude_class"]:
        keep_class_dim = [all_target_class.index(i) for i in all_target_class if i not in config["exclude_classes"] ]
        all_target_class = [target_class[i] for i in keep_class_dim]
        keep_class_dim = [target_class.index(i) for i in target_class if i not in config["exclude_classes"] ]
        target_class = [target_class[i] for i in keep_class_dim]
    main(args, config)