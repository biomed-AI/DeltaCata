import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pandas as pd


def get_maccs_keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Unable to create molecule object from SMILES")
        return None
    
    maccs_keys = MACCSkeys.GenMACCSKeys(mol)
    maccs_keys = list(maccs_keys)
    maccs_keys = torch.tensor(maccs_keys,dtype=torch.float32)
    return maccs_keys


class ProteinGraphDataset(data.Dataset):
    def __init__(self, data_path,feat_path):
        super(ProteinGraphDataset, self).__init__()       
        self.dataset = pd.read_csv(data_path)
        self.feat_path = feat_path
               
    def __len__(self): return len(self.dataset)
    
    def __getitem__(self, idx): return self._featurize_graph(idx)
    
    def _featurize_graph(self, idx):
        row = self.dataset.loc[idx]
        mut_seq_id = row['seq_id']
        protein_name = row['protein_name']
        mutants = row['mutant']
        
        with torch.no_grad():
            X = torch.load(self.feat_path+"structs/"+protein_name+".tensor")
            DSSP = torch.load(self.feat_path+"structs/"+protein_name+"_dssp.tensor")
            mut_esm = torch.load(self.feat_path+"esm/" + mut_seq_id + ".tensor")
            wt_esm = torch.load(self.feat_path+"esm/"+ protein_name + ".tensor")
            
            mut_pre_computed_node_feat = torch.cat([mut_esm, DSSP], dim=-1)
            wt_pre_computed_node_feat = torch.cat([wt_esm, DSSP], dim=-1)

            X_ca = X[:, 1]
            edge_index = torch_geometric.nn.radius_graph(X_ca, r=12.0, loop=True, max_num_neighbors = 1000, num_workers = 4)

            mut_graph_mask10 = torch.zeros(len(X),dtype=torch.bool)
            mut_graph_mask12 = torch.zeros(len(X),dtype=torch.bool)
            for mut in mutants.split(","):
                mut_loc = X_ca[int(mut[1:-1])-1]
                distance = torch.norm(X_ca-mut_loc,dim=1)
                mut_graph_mask10[distance<10]=True
                mut_graph_mask12[distance<12]=True
                
            smiles_feat = get_maccs_keys(row['smiles'])
            
        wt_graph_data = torch_geometric.data.Data(name=row['index'], node_feat=wt_pre_computed_node_feat, 
                                                X=X, edge_index=edge_index)
        mut_graph_data = torch_geometric.data.Data(name=row['index'], node_feat=mut_pre_computed_node_feat,
                            mut_graph_mask10=mut_graph_mask10, mut_graph_mask12=mut_graph_mask12)
        
        return wt_graph_data, mut_graph_data, smiles_feat


##############  Geometric Featurizer  ##############
def get_geo_feat(X, edge_index):
    pos_embeddings = _positional_embeddings(edge_index)
    node_angles = _get_angle(X)
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

    geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

    return geo_node_feat, geo_edge_feat


def _positional_embeddings(edge_index, num_embeddings=16):
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE

def _get_angle(X, eps=1e-7):
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles # dim = 12

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _get_distance(X, edge_index):
    atom_N = X[:,0]  # [L, 3]
    atom_Ca = X[:,1]
    atom_C = X[:,2]
    atom_O = X[:,3]
    atom_R = X[:,4]
    
    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1) # dim = [N, 10 * 16]
    atom_list = ["N", "Ca", "C", "O", "R"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1) # dim = [E, 25 * 16]
    return node_dist, edge_dist

def _get_direction_orientation(X, edge_index): # N, CA, C, O, R
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    node_j, node_i = edge_index
    
    t = F.normalize(X[:, [0,2,3,4]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1) # [L, 4 * 3]

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3]
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]

    return node_direction, edge_direction, edge_orientation

def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [N,3,3]
        Q [N,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q

