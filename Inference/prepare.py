import json
import pandas as pd
import torch
import esm
import numpy as np
from tqdm import tqdm
from Bio import pairwise2
import os

AMINO_ACIDS = 'CDSQKIPTFAGHELRWVNYM'

def prepare_dms(nn_config):    
    sequence = nn_config['sequence']
    protein_name = nn_config['protein_name']
    mutants = []
    mutant_seqs = []
    for i in range(len(sequence)):  
        original_aa = sequence[i] 
        for aa in AMINO_ACIDS: 
            if aa != original_aa:
                mutant = original_aa+str(i+1)+aa
                mutant_sequence = sequence[:i] + aa + sequence[i+1:]
                mutants.append(mutant)  # mutant (start from 1)
                mutant_seqs.append(mutant_sequence)  # mutant sequence 
    data = {
        'index':list(range(len(mutants))),
        "protein_name":[protein_name for _ in mutants],
        "smiles":nn_config['smiles'],
        "mutant":mutants,
        "mutant_sequence":mutant_seqs,
        "seq_id":[protein_name+"_"+mut for mut in mutants]}
    dms_df = pd.DataFrame(data)
    dms_df.to_csv(nn_config['dms_path'],index=False)
    return dms_df


def prepare_mutant_df(nn_config,desired_mutants):    
    sequence = nn_config['sequence']
    protein_name = nn_config['protein_name']
    mutant_seqs = []
    mutants = desired_mutants.split(";")
    for mutant in mutants:  
        mutant_sequence = sequence
        for mut_site in mutant.split(","):
            from_aa,loc,to_aa = mut_site[0],mut_site[1:-1],mut_site[-1]
            loc = int(loc)-1 # convert 1-based loc to 0-based index
            assert from_aa == mutant_sequence[loc], f"wrong mutant:{mutant} with wrong wt_aa {from_aa} != {mutant_sequence[loc]} in {loc+1}"
            mutant_sequence = mutant_sequence[:loc] + to_aa + mutant_sequence[loc+1:]
        mutant_seqs.append(mutant_sequence)
    data = {
        "index":list(range(len(mutants))),
        "protein_name":[protein_name for _ in mutants],
        "smiles":nn_config['smiles'],
        "mutant":mutants,
        "mutant_sequence":mutant_seqs,
        "seq_id":[protein_name+"_"+mut for mut in mutants]}
    test_df = pd.DataFrame(data)
    test_df.to_csv(nn_config['dms_path'],index=False)
    return test_df


def prepare_esm_embedding(esm_input_data,Max_esm,Min_esm,save_path):  
    # Load ESM-2 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    model = model.to(device)
    
    with torch.no_grad():
        batch_size = 10
        for batch_idx,i in enumerate(range(0, len(esm_input_data), batch_size)):
            batch_data = esm_input_data[i:i + batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            print(f"ESM batch idx:{batch_idx}")

            batch_tokens=batch_tokens.to(device)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, ((seq_id,_),tokens_len) in enumerate(zip(batch_data,batch_lens)):
                raw_esm = token_representations[i, 1 : tokens_len - 1].detach().cpu().numpy()
                esm_norm = (raw_esm - Min_esm) / (Max_esm - Min_esm)
                torch.save(torch.tensor(esm_norm, dtype = torch.float32), save_path + seq_id + '.tensor')



def pdb2tensor(origin_pdb_path,protein_name,save_path):
    def get_pdb_xyz(pdb_file):
        current_pos = -1000
        X = []
        current_aa = {} # N, CA, C, O, R
        for line in pdb_file:
            if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
                if current_aa != {}:
                    R_group = []
                    for atom in current_aa:
                        if atom not in ["N", "CA", "C", "O"]:
                            R_group.append(current_aa[atom])
                    if R_group == []:
                        R_group = [current_aa["CA"]]
                    R_group = np.array(R_group).mean(0)
                    X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                    current_aa = {}
                if line[0:4].strip() != "TER":
                    current_pos = int(line[22:26].strip())

            if line[0:4].strip() == "ATOM":
                atom = line[13:16].strip()
                if atom != "H":
                    xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                    current_aa[atom] = xyz
        return np.array(X)
    
    with open(origin_pdb_path, "r") as f:
        X = get_pdb_xyz(f.readlines())
        
    torch.save(torch.tensor(X, dtype = torch.float32), save_path + protein_name + '.tensor')



def get_DSSP(ref_seq,pdb_path,protein_name,dssp_path,save_path):
    ########## Get DSSP ##########
    def process_dssp(dssp_file):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"
        SS_type = "HBEGITSC"
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

        with open(dssp_file, "r") as f:
            lines = f.readlines()

        seq = ""
        dssp_feature = []

        p = 0
        while lines[p].strip()[0] != "#":
            p += 1
        for i in range(p + 1, len(lines)):
            aa = lines[i][13]
            if aa == "!" or aa == "*":
                continue
            seq += aa
            SS = lines[i][16]
            if SS == " ":
                SS = "C"
            SS_vec = np.zeros(8)
            SS_vec[SS_type.find(SS)] = 1
            ACC = float(lines[i][34:38].strip())
            ASA = min(1, ACC / rASA_std[aa_type.find(aa)])
            dssp_feature.append(np.concatenate((np.array([ASA]), SS_vec)))

        return seq, dssp_feature

    def match_dssp(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        padded_item = np.zeros(9)

        new_dssp = []
        for aa in seq:
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0))

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])

        return matched_dssp

    
    os.system("{}mkdssp -i {} -o {}{}.dssp".format(dssp_path, pdb_path, save_path,protein_name))
    dssp_seq, dssp_matrix = process_dssp("{}{}.dssp".format(save_path,protein_name))
    if dssp_seq != ref_seq:
        dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)
    dssp_matrix = np.array(dssp_matrix)

    torch.save(torch.tensor(dssp_matrix, dtype = torch.float32), save_path + protein_name + '_dssp.tensor')
    os.system("rm {}{}.dssp".format(save_path,protein_name))



def Inference_prepare_feature(nn_config):
    nn_config['protein_name'] = nn_config['protein']['name']
    nn_config['sequence'] = nn_config['protein']['sequence']
    nn_config['smiles'] = nn_config['ligand']['SMILES']
    
    # step 0 makedirs
    feature_path = nn_config['feature_path']
    ESM_path = feature_path+"esm/"
    os.makedirs(ESM_path,exist_ok=True)
    nn_config['ESM_path'] = ESM_path
    
    structs_path = feature_path+"structs/"
    os.makedirs(structs_path,exist_ok=True)
    nn_config['structs_path'] = structs_path
    
    
    # step 1 prepare mutation file
    print("step 1 prepare mutation file")
    nn_config['dms_path'] = feature_path+"test.csv"
    if nn_config['mutant']=="dms": 
        test_df = prepare_dms(nn_config)
    else: 
        desired_mutants = nn_config["mutant"]
        test_df = prepare_mutant_df(nn_config,desired_mutants)

    
    protein_name = nn_config['protein_name']
    sequence = nn_config['sequence']
    
    
    # step2 get ESM embeddings
    print("step 2 prepare ESM embeddings")
    esm_feat_save_path = nn_config['ESM_path']
    Max_esm = np.load(nn_config['artifacts_path']+"esm_t33_Max.npy")
    Min_esm = np.load(nn_config['artifacts_path']+"esm_t33_Min.npy")
        
    # Prepare esm_input_data
    esm_input_data = [(row['seq_id'], row['mutant_sequence']) for _,row in test_df.iterrows()]
    esm_input_data.append((protein_name,sequence))   # wild type
    prepare_esm_embedding(esm_input_data,Max_esm,Min_esm,esm_feat_save_path)
    
    
    # step3 get dssp & pdb tensor
    print("step 3 prepare DSSP")
    pdb_path = nn_config['pdb_path']
    structs_save_path = nn_config['structs_path']
    dssp_path = nn_config['artifacts_path'] + "dssp-2.0.4/"
    
    pdb2tensor(pdb_path,protein_name,structs_save_path)
    get_DSSP(sequence,pdb_path,protein_name,dssp_path,structs_save_path)
    
    return nn_config



def Reproduce_prepare_feature(args):
    test_df = pd.read_csv(args.test_dataset_path)
    pdb_path = args.pdbs_path
    feature_path = args.feature_path
    
    # get esm emb
    print("step 1 prepare ESM embeddings")
    esm_feat_save_path = feature_path+"esm/"
    os.makedirs(esm_feat_save_path,exist_ok=True)
    
    Max_esm = np.load(args.artifacts_path+"esm_t33_Max.npy")
    Min_esm = np.load(args.artifacts_path+"esm_t33_Min.npy")
        
    # Prepare esm_input_data
    esm_input_data_wt_dict = {row['protein_name']:row['wildtype_sequence'] for _,row in test_df.iterrows()}
    esm_input_data_mut_dict = {row['seq_id']: row['mutant_sequence'] for _,row in test_df.iterrows()}
    esm_input_data=[(k,v) for k,v in esm_input_data_wt_dict.items()] + [(k,v) for k,v in esm_input_data_mut_dict.items()]
    prepare_esm_embedding(esm_input_data,Max_esm,Min_esm,esm_feat_save_path)
    
    # get pdb tensor & dssp
    print("step 2 prepare DSSP")
    structs_save_path = feature_path+"structs/"
    dssp_path = args.artifacts_path + "dssp-2.0.4/"
    os.makedirs(structs_save_path,exist_ok=True)
    for protein_name,sequence in esm_input_data_wt_dict.items():
        pdb2tensor(pdb_path+protein_name+".pdb",protein_name,structs_save_path)
        get_DSSP(sequence,pdb_path+protein_name+".pdb",protein_name,dssp_path,structs_save_path)