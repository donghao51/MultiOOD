from mmaction.apis import init_recognizer
import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_flow_audio import EPICDOMAIN

def ash_b(x, percentile=90):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x

def validate_one_step(model, clip, labels, flow, model_flow, spectrogram, audio_cls_model):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)
    spectrogram = spectrogram.unsqueeze(1).cuda()

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)  # 16,1024,8,14,14
        v_feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)

        f_feat = model_flow.module.backbone.get_feature(flow)  # 16,1024,8,14,14
        f_feat = model_flow.module.backbone.get_predict(f_feat)
        f_predict, f_emd = model_flow.module.cls_head(f_feat)

        _, audio_feat, _ = audio_model(spectrogram)
        audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
       
        if args.use_ash:
            v_emd = ash_b(v_emd.view(v_emd.size(0), -1, 1, 1))
            v_emd = v_emd.view(v_emd.size(0), -1)
            f_emd = ash_b(f_emd.view(f_emd.size(0), -1, 1, 1))
            f_emd = f_emd.view(f_emd.size(0), -1)
            audio_emd = ash_b(audio_emd.view(audio_emd.size(0), -1, 1, 1))
            audio_emd = audio_emd.view(audio_emd.size(0), -1)

        if args.use_react:
            v_emd = v_emd.clip(max=args.v_thr)
            v_emd = v_emd.view(v_emd.size(0), -1)
            f_emd = f_emd.clip(max=args.f_thr)
            f_emd = f_emd.view(f_emd.size(0), -1)
            audio_emd = audio_emd.clip(max=args.a_thr)
            audio_emd = audio_emd.view(audio_emd.size(0), -1)

        predict = mlp_cls(v_emd, audio_emd, f_emd)
        feature = torch.cat((v_emd, audio_emd, f_emd), dim=1)

    return predict, feature, v_predict, f_predict, audio_predict

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
  
    def forward(self, vfeat, afeat, ffeat):
        feat = torch.cat((vfeat, afeat, ffeat), dim=1)
        return self.enc_net(feat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/path/to/video_datasets/',
                        help='datapath')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--resumef", type=str, default='checkpoint.pt')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument('--use_ash', action='store_true')
    parser.add_argument('--use_react', action='store_true')
    parser.add_argument('--v_thr', type=float, default=0.8023215949535369,
                        help='v_thr')
    parser.add_argument('--f_thr', type=float, default=0.615705931186676,
                        help='f_thr')
    parser.add_argument('--a_thr', type=float, default=0.5100213885307313,
                        help='f_thr')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--near_ood', action='store_true')
    parser.add_argument("--dataset", type=str, default='Kinetics') # HMDB UCF Kinetics
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.use_react:
        percentile = 90
        if args.near_ood:
            feature_name = 'saved_files/id_'+args.dataset+'_near_ood_feature_' + args.appen + 'val.npy'
        else:
            feature_name = 'saved_files/id_'+args.dataset+'_feature_' + args.appen + 'val.npy'
        id_train_feature = np.load(feature_name)
        v_emd = id_train_feature[:, :2304]
        args.v_thr = np.percentile(v_emd.flatten(), percentile)
        a_emd = id_train_feature[:, 2304:2304+512]
        args.a_thr = np.percentile(a_emd.flatten(), percentile)
        f_emd = id_train_feature[:, 2304+512:]
        args.f_thr = np.percentile(f_emd.flatten(), percentile)

        args.appen = args.appen + 'react_'

    if args.use_ash:
        args.appen = args.appen + 'ash_'

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    v_dim = 2304
    f_dim = 2048
    a_dim = 512

    num_class = 129

    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    model_flow = init_recognizer(config_file_flow, device=device,use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
    cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)

    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar")
    audio_model.load_state_dict(checkpoint['model_state_dict'])
    audio_model = audio_model.cuda()
    audio_model.eval()

    audio_cls_model = AudioAttGenModule()
    audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    audio_cls_model.fc = nn.Linear(a_dim, num_class)
    audio_cls_model = audio_cls_model.cuda()

    mlp_cls = Encoder(input_dim=v_dim+f_dim+a_dim, out_dim=num_class)
    mlp_cls = mlp_cls.cuda()

    resume_file = args.resumef
    print("Resuming from ", resume_file)
    checkpoint = torch.load(resume_file)
    BestTestAcc = checkpoint['BestTestAcc']

    model.load_state_dict(checkpoint['model_state_dict'])
    model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
    mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])
    audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
    audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
        
    model.eval()
    model_flow.eval()
    audio_model.eval()
    audio_cls_model.eval()
    mlp_cls.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = EPICDOMAIN(split='train', eval=True, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=args.near_ood)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=False,
                                                pin_memory=(device.type == "cuda"), drop_last=False)

    val_dataset = EPICDOMAIN(split='val', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=args.near_ood)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=False,
                                                pin_memory=(device.type == "cuda"), drop_last=False)
    test_dataset = EPICDOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=args.near_ood)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=False,
                                                pin_memory=(device.type == "cuda"), drop_last=False)
    
    eval_dataset = EPICDOMAIN(split='eval', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=args.near_ood)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=False,
                                                pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader, 'eval': eval_dataloader}

    splits = ['test', 'eval', 'val']

    for split in splits:
        print(split)
        pred_list, conf_list, label_list, output_list, feature_list = [], [], [], [], []
        for clip, flow, spectrogram, labels in tqdm(dataloaders[split]):
            output, feature, output_v, output_f, output_a = validate_one_step(model, clip, labels, flow, model_flow, spectrogram, audio_cls_model)
            score = torch.softmax(output, dim=1)
            conf, pred = torch.max(score, dim=1)
            output_list.append(output.cpu())
            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(labels.cpu())
            feature_list.append(feature.cpu())

        output_list = torch.cat(output_list).numpy()
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        feature_list = torch.cat(feature_list).numpy()

        output_name = 'saved_files/id_'+args.dataset+'_near_ood_output_vfa_' + args.appen + split + '.npy'
        pred_name = 'saved_files/id_'+args.dataset+'_near_ood_pred_vfa_' + args.appen + split + '.npy'
        conf_name = 'saved_files/id_'+args.dataset+'_near_ood_conf_vfa_' + args.appen + split + '.npy'
        label_name = 'saved_files/id_'+args.dataset+'_near_ood_label_vfa_' + args.appen + split + '.npy'
        feature_name = 'saved_files/id_'+args.dataset+'_near_ood_feature_vfa_' + args.appen + split + '.npy'
    
        np.save(output_name, output_list)
        np.save(pred_name, pred_list)
        np.save(conf_name, conf_list)
        np.save(label_name, label_list)
        np.save(feature_name, feature_list)
