import numpy as np
from metrics import compute_all_metrics
import torch
import argparse
import faiss
import sklearn.covariance
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
            
    def forward(self, afeat, ffeat):
        feat = torch.cat((afeat, ffeat), dim=1)
        return self.enc_net(feat)

def generalized_entropy(softmax_id_val, gamma=0.1, M=20):
        probs =  softmax_id_val 
        probs_sorted = np.sort(probs, axis=1)[:,-M:]
        scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
        return -scores 

def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

parser = argparse.ArgumentParser()
parser.add_argument("--postprocessor", type=str, default='msp') # 'msp' 'ebo' 'maxlogit' 'Mahalanobis' 'ash' 'react' 'knn' 'gen' 'vim'
parser.add_argument("--appen", type=str, default='a2d_npmix_best_') # a2d_npmix_best_ a2d_npmix_best_ash_ a2d_npmix_best_react_
parser.add_argument("--dataset", type=str, default='HMDB') # HMDB Kinetics
parser.add_argument("--ood_dataset", type=str, default='UCF') # HMDB UCF Kinetics EPIC HAC
parser.add_argument("--path", type=str, default='HMDB-rgb-flow') # HMDB-rgb-flow EPIC-rgb-flow
parser.add_argument("--resume_file", type=str, default='HMDB-rgb-flow/models/checkpoint.pt') # for vim 'HMDB_far_ood_a2d_npmix.pt'
args = parser.parse_args()

if args.dataset == 'HMDB':
    num_classes = 43
elif args.dataset == 'Kinetics':
    num_classes = 229

if args.postprocessor == 'knn':
    if args.dataset == 'Kinetics':
        feature_name = args.path + '/saved_files/id_'+args.dataset+'_feature_' + args.appen + 'val.npy'
    else:
        feature_name = args.path + '/saved_files/id_'+args.dataset+'_feature_' + args.appen + 'train.npy'
    id_train_feature = np.load(feature_name)
    id_train_feature = normalizer(id_train_feature)

    index = faiss.IndexFlatL2(id_train_feature.shape[1])
    index.add(id_train_feature)

if args.postprocessor == 'Mahalanobis':
    if args.dataset == 'Kinetics':
        feature_name = args.path + '/saved_files/id_'+args.dataset+'_feature_' + args.appen + 'val.npy'
        label_name = args.path + '/saved_files/id_'+args.dataset+'_label_' + args.appen + 'val.npy'
    else:
        feature_name = args.path + '/saved_files/id_'+args.dataset+'_feature_' + args.appen + 'train.npy'
        label_name = args.path + '/saved_files/id_'+args.dataset+'_label_' + args.appen + 'train.npy'
        
    id_train_feature = np.load(feature_name)
    id_train_label = np.load(label_name)
    id_train_feature = torch.tensor(id_train_feature)
    id_train_label = torch.tensor(id_train_label)

    class_mean = []
    centered_data = []
    for c in range(num_classes):
        class_samples = id_train_feature[id_train_label.eq(c)]
        class_mean.append(class_samples.mean(0))
        centered_data.append(class_samples -
                                class_mean[c].view(1, -1))

    class_mean = torch.stack(
        class_mean)  # shape [#classes, feature dim]

    group_lasso = sklearn.covariance.EmpiricalCovariance(
        assume_centered=False)
    group_lasso.fit(
        torch.cat(centered_data).cpu().numpy().astype(np.float32))
    # inverse of covariance
    precision = torch.from_numpy(group_lasso.precision_).float()

if args.postprocessor == 'vim':
    if args.dataset == 'kinetics':
        feature_name = args.path + '/saved_files/id_'+args.dataset+'_feature_' + args.appen + 'val.npy'
    else:
        feature_name = args.path + '/saved_files/id_'+args.dataset+'_feature_' + args.appen + 'train.npy'
    id_train_feature = np.load(feature_name)

    vim_dim = 256
    fc = Encoder(id_train_feature.shape[1], num_classes)
    checkpoint = torch.load(args.resume_file, map_location=torch.device('cpu'))
    fc.load_state_dict(checkpoint['mlp_cls_state_dict'])
    vim_w = fc.enc_net.weight.cpu().detach().numpy()
    vim_b = fc.enc_net.bias.cpu().detach().numpy()
    id_train_logit = id_train_feature @ vim_w.T +vim_b

    vim_u = -np.matmul(pinv(vim_w), vim_b)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(id_train_feature - vim_u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    vim_NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[vim_dim:]]).T)

    vlogit_id_train = norm(np.matmul(id_train_feature - vim_u,
                                        vim_NS),
                            axis=-1)
    vim_alpha = id_train_logit.max(
        axis=-1).mean() / vlogit_id_train.mean()

split = 'test'
print(split)

output_name = args.path + '/saved_files/id_'+args.dataset+'_output_' + args.appen + split + '.npy'
pred_name = args.path + '/saved_files/id_'+args.dataset+'_pred_' + args.appen + split + '.npy'
conf_name = args.path + '/saved_files/id_'+args.dataset+'_conf_' + args.appen + split + '.npy'
label_name = args.path + '/saved_files/id_'+args.dataset+'_label_' + args.appen + split + '.npy'
feature_name = args.path + '/saved_files/id_'+args.dataset+'_feature_' + args.appen + split + '.npy'
id_output = np.load(output_name)
id_pred = np.load(pred_name)
id_conf = np.load(conf_name)
id_gt = np.load(label_name)
id_feature = np.load(feature_name)

ID_ACC = acc(id_pred, id_gt)
print("ID_ACC: ", ID_ACC)

if args.postprocessor == 'ebo' or args.postprocessor == 'ash' or args.postprocessor == 'react':
    temperature = 1.0
    id_output = torch.tensor(id_output)
    id_conf = temperature * torch.logsumexp(id_output / temperature, dim=1)
    id_conf = id_conf.numpy()
elif args.postprocessor == 'maxlogit':
    id_output = torch.tensor(id_output)
    id_conf, id_pred = torch.max(id_output, dim=1)
    id_pred = id_pred.numpy().astype(int)
    id_conf = id_conf.numpy()
elif args.postprocessor == 'knn':
    K = 10 
    id_feature = normalizer(id_feature)
    D, _ = index.search(
            id_feature,
            K,
        )
    kth_dist = -D[:, -1]
    id_conf = kth_dist
elif args.postprocessor == 'Mahalanobis':
    class_scores = torch.zeros((id_output.shape[0], num_classes))
    for c in range(num_classes):
        id_feature = torch.tensor(id_feature)
        tensor = id_feature - class_mean[c].view(1, -1)
        class_scores[:, c] = -torch.matmul(
            torch.matmul(tensor, precision), tensor.t()).diag()

    id_conf = torch.max(class_scores, dim=1)[0].numpy()
elif args.postprocessor == 'gen':
    id_output = torch.tensor(id_output)
    id_output_softmax = torch.softmax(id_output, dim=1).numpy()
    id_conf = generalized_entropy(id_output_softmax, M=num_classes)
elif args.postprocessor == 'vim':
    logit_id = id_feature @ vim_w.T + vim_b
    energy_id = logsumexp(logit_id, axis=-1)
    vlogit_id = norm(np.matmul(id_feature - vim_u, vim_NS),
                          axis=-1) * vim_alpha
    id_conf = -vlogit_id + energy_id

split = 'eval'
print(split)

output_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_output_' + args.appen + split + '.npy'
pred_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_pred_' + args.appen + split + '.npy'
conf_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_conf_' + args.appen + split + '.npy'
label_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_label_' + args.appen + split + '.npy'
feature_name = args.path + '/saved_files/id_'+args.dataset+'_ood_'+args.ood_dataset+'_feature_' + args.appen + split + '.npy'
ood_output = np.load(output_name)
ood_pred = np.load(pred_name)
ood_conf = np.load(conf_name)
ood_gt = np.load(label_name)
ood_feature = np.load(feature_name)

if args.postprocessor == 'ebo' or args.postprocessor == 'ash' or args.postprocessor == 'react':
    ood_output = torch.tensor(ood_output)
    ood_conf = temperature * torch.logsumexp(ood_output / temperature, dim=1)
    ood_conf = ood_conf.numpy()
elif args.postprocessor == 'maxlogit':
    ood_output = torch.tensor(ood_output)
    ood_conf, ood_pred = torch.max(ood_output, dim=1)
    ood_pred = ood_pred.numpy().astype(int)
    ood_conf = ood_conf.numpy()
elif args.postprocessor == 'knn':
    ood_feature = normalizer(ood_feature)
    D, _ = index.search(
            ood_feature,
            K,
        )
    kth_dist = -D[:, -1]
    ood_conf = kth_dist
elif args.postprocessor == 'Mahalanobis':
    class_scores = torch.zeros((ood_output.shape[0], num_classes))
    for c in range(num_classes):
        ood_feature = torch.tensor(ood_feature)
        tensor = ood_feature - class_mean[c].view(1, -1)
        class_scores[:, c] = -torch.matmul(
            torch.matmul(tensor, precision), tensor.t()).diag()

    ood_conf = torch.max(class_scores, dim=1)[0].numpy()
elif args.postprocessor == 'gen':
    ood_output = torch.tensor(ood_output)
    ood_output_softmax = torch.softmax(ood_output, dim=1).numpy()
    ood_conf = generalized_entropy(ood_output_softmax, M=num_classes)
elif args.postprocessor == 'vim':
    logit_ood = ood_feature @ vim_w.T + vim_b
    energy_ood = logsumexp(logit_ood, axis=-1)
    vlogit_ood = norm(np.matmul(ood_feature - vim_u, vim_NS),
                        axis=-1) * vim_alpha
    ood_conf = -vlogit_ood + energy_ood

ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
pred = np.concatenate([id_pred, ood_pred])
conf = np.concatenate([id_conf, ood_conf])
label = np.concatenate([id_gt, ood_gt])
ood_metrics = compute_all_metrics(conf, label, pred)

print("FPR@95: ", ood_metrics[0])
print("AUROC: ", ood_metrics[1])