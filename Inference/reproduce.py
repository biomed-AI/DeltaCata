from prepare import Reproduce_prepare_feature
import json
import argparse
import datetime
from tqdm import tqdm
from model import DeltaCata
from torch_geometric.loader import DataLoader
from data import ProteinGraphDataset

import numpy as np
import os, random, torch
import pandas as pd

from scipy.stats import pearsonr


def Reproduce(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = args.output
    feature_path = args.feature_path
    os.makedirs(output_path, exist_ok = True)    
    artifacts_path = args.artifacts_path
    
    task = args.task
    level = args.level

    num_workers = 8
    dropout=0.0
    node_input_dim = 1473
    edge_input_dim = 450
    hidden_dim = 128
    GNN_layers = 2
    batch_size = 4
    folds = 5
    kcat_num_att_layers= 2
    km_num_att_layers= 4
    num_att_layers = kcat_num_att_layers if task == 'kcat' else km_num_att_layers

    test_data_path = args.test_dataset_path
    test_dataset = ProteinGraphDataset(test_data_path,feature_path)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)

    models = []
    for fold in range(folds):
        state_dict = torch.load(artifacts_path + f'{task}_{level}/model_fold%s.ckpt'%fold, device)
        model = DeltaCata(node_input_dim, edge_input_dim, hidden_dim, GNN_layers, dropout, task, num_att_layers).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)


    test_pred_dict = {} 
    for data in tqdm(test_dataloader):
        wt_data,mut_data,smiles = data[0].to(device),data[1].to(device),data[2].to(device)
        with torch.no_grad():
            mut_data.mut_graph_mask=mut_data.mut_graph_mask12 if task == 'kcat' and level=='seq' else mut_data.mut_graph_mask10
            outputs = [model(wt_data,mut_data,smiles) for model in models] # [[b,2]*5]
            
            preds = [outputs[0] for outputs in outputs] # [[b,2]*5]
            preds = torch.stack(preds,0).mean(0).detach().cpu().numpy() 
            
        names = wt_data.name
        for i, mut in enumerate(names):
            test_pred_dict[mut] = preds[i]
            
    test_df = pd.read_csv(test_data_path)

    preds = []
    trues = []
    for _,row in test_df.iterrows():
        name = row['index']
        preds.append(test_pred_dict[name])
        trues.append(row[f'delta_{task}_log10'])
        
    log = open(args.output+f"reproduce_{task}_{level}.log",'w')
    log.write("Args: "+str(args)+"\n")
    pcc = round(pearsonr(trues, preds)[0], 4)
    log.write(f"test PCC: {pcc}\n")
    log.close()
    
    test_df[f'{task}_{level}_preds']=preds
    test_df.to_csv(args.output+f"reproduce_{task}_{level}.csv",index=False)


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



parser = argparse.ArgumentParser()
parser.add_argument("--task",type=str,default="kcat",help="kcat or km")
parser.add_argument("--level",type=str,default="mut",help="mut/seq for mutation- or sequence-level split")
parser.add_argument("--test_dataset_path", type=str, default='../Dataset/test_dataset/kcat_mut_test.csv',help="Path to a csv file of a test dataset")
parser.add_argument("--pdbs_path", type=str, default='../Dataset/test_pdbs/',help="Path where pdb files of the test datasets are stored")
parser.add_argument("--artifacts_path", type=str, default='./artifacts/',help="Path where model checkpoints are stored")
parser.add_argument("--feature_path", type=str, default='./features/reproduce_data/',help="Path where prepared features are saved")
parser.add_argument("--output", type=str, default='./output/',help="Path where output files are saved")
args = parser.parse_args()


if __name__ == '__main__':
    Seed_everything(seed=42)
    Reproduce_prepare_feature(args)
    Reproduce(args)
