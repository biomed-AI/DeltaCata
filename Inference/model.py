import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from data import *
import math


class GNNLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


class TripletAttention(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.attention_dropout   = attention_dropout
        
        assert not (self.edge_width % self.num_heads),\
                'edge_width must be divisible by num_heads'
        self._dot_dim = self.edge_width//self.num_heads
        self._scale_factor = self._dot_dim ** -0.5
        
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        
        self.lin_QKV_in = nn.Linear(self.edge_width, self.edge_width*3)
        self.lin_EG_in  = nn.Linear(self.edge_width, self.num_heads*2)
        
        self.lin_QKV_out = nn.Linear(self.edge_width, self.edge_width*3)
        self.lin_EG_out  = nn.Linear(self.edge_width, self.num_heads*2)
        
        self.lin_O  = nn.Linear(self.edge_width*2, self.edge_width)
    
    def forward(self, e, mask):
        bsize, num_edges, _, embed_dim = e.shape
        e_ln = self.tri_ln_e(e)
        
        # Projections
        Q_in, K_in, V_in = self.lin_QKV_in(e_ln).chunk(3, dim=-1)
        E_in, G_in = self.lin_EG_in(e_ln).unsqueeze(2).chunk(2, dim=-1) # bi1kh
        
        Q_in = Q_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_in = K_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_in = V_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_in = Q_in * self._scale_factor
        H_in = torch.einsum('bijdh,bjkdh->bijkh', Q_in, K_in) + E_in
        
        mask_in = mask.unsqueeze(2)
        gates_in = torch.sigmoid(G_in+mask_in)
        A_in = torch.softmax(H_in+mask_in, dim=3) * gates_in

        if self.attention_dropout > 0:
            A_in = F.dropout(A_in, p=self.attention_dropout,
                             training=self.training, inplace=True)
        
        Va_in = torch.einsum('bijkh,bjkdh->bijdh', A_in, V_in)
        
        Q_out, K_out, V_out = self.lin_QKV_out(e_ln).chunk(3, dim=-1)
        E_out, G_out = self.lin_EG_out(e_ln).unsqueeze(3).chunk(2, dim=-1) # bki1h
        
        Q_out = Q_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_out = K_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_out = V_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_out = Q_out * self._scale_factor
        H_out = torch.einsum('bijdh,bkjdh->bkijh', Q_out, K_out) + E_out
        
        mask_out = mask.unsqueeze(3)
        gates_out = torch.sigmoid(G_out+mask_out)
        A_out = torch.softmax(H_out+mask_out, dim=1) * gates_out
        if self.attention_dropout > 0:
            A_out = F.dropout(A_out, p=self.attention_dropout,
                              training=self.training, inplace=True)
        
        Va_out = torch.einsum('bkijh,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1).view(bsize,num_edges,num_edges,embed_dim*2)
        e = self.lin_O(Va)
        return e


class TripletAttentionPackage(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super(TripletAttentionPackage, self).__init__()
        self.tgt = TripletAttention(edge_width,num_heads,attention_dropout=attention_dropout)

        self.drop = nn.Dropout(attention_dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(edge_width) for _ in range(2)])
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(edge_width, edge_width*4),
            nn.ReLU(),
            nn.Linear(edge_width*4, edge_width)
        )

    def forward(self,h_V,subgraph_mask,batch_id):
        assert subgraph_mask.dtype == torch.bool, "subgraph_mask must be torch.bool"
        BL,hid_dim = h_V.shape
        
        # step 1.
        sub_h_V = h_V[subgraph_mask] # [B*subgraph L, hid] 
        sub_batch_id = batch_id[subgraph_mask] # [B*subgraph L, ]

        sub_h_V_batch,sub_h_V_batch_mask = to_dense_batch(sub_h_V,sub_batch_id) # [B,subgraph L, hid]  [B, subgraph L] 

        # step 2.
        subgraph_edge = torch.einsum('bid,bjd->bijd', sub_h_V_batch, sub_h_V_batch) # [B, subgraph L, subgraph L, hid]
        subgraph_edge_mask = torch.einsum('bi,bj->bij', sub_h_V_batch_mask, sub_h_V_batch_mask) # [B, subgraph L, subgraph L,]
        subgraph_edge_mask = subgraph_edge_mask.unsqueeze(-1).to(torch.float) 
        subgraph_edge_mask_tgt = (1-subgraph_edge_mask)*torch.finfo(subgraph_edge_mask.dtype).min 

        # step 3.
        subgraph_edge = self.tgt(subgraph_edge,subgraph_edge_mask_tgt) # [B, subgraph L, subgraph L, hid]

        subgraph_edge = subgraph_edge.masked_fill(subgraph_edge_mask == 0, 0)  
        sub_h_V_batch = subgraph_edge.sum(dim=2)/subgraph_edge_mask.sum(dim=2) # [B, subgraph L, hid]
        
        dh = sub_h_V_batch.view([-1,hid_dim])[sub_h_V_batch_mask.view([-1])==1] # [B*subgraph L, hid] 
        
        # step 4.
        sub_h_V = self.norm[0](sub_h_V + self.drop(dh)) 
        
        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(sub_h_V)
        sub_h_V = self.norm[1](sub_h_V + self.drop(dh))

        h_V_new = h_V.clone()  
        h_V_new[subgraph_mask] = sub_h_V  
        h_V = h_V_new  
        
        return h_V


class Graph_encoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim,
                 num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.gnn_layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))

        self.tgt_layers = nn.ModuleList([TripletAttentionPackage(edge_width=hidden_dim,num_heads=4,attention_dropout=drop_rate) for _ in range(num_layers)])
        
    def forward(self, h_V, edge_index, h_E, batch_id,subgraph_mask):

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for gnn_layer,tgt_layer in zip(self.gnn_layers,self.tgt_layers):
            h_V, h_E = gnn_layer(h_V, edge_index, h_E, batch_id)
            h_V = tgt_layer(h_V,subgraph_mask,batch_id) 
        return h_V


class CrossAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        
        self.drop = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(2)])

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.scale = math.sqrt(hid_dim // n_heads)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(hid_dim, hid_dim*4),
            nn.ReLU(),
            nn.Linear(hid_dim*4, hid_dim)
        )

    def forward(self, smiles, key, value, batch):
        # smiles  [B, hid]
        
        Q = self.w_q(smiles[batch]) # [B*L,hid]  smiles
        K = self.w_k(key)   # [B*L,hid]  protein
        V = self.w_v(value) # [B*L,hid]  protein

        score = (Q*K).sum(dim=-1) / self.scale  # [B*L,]
        
        exp_score = torch.exp(score)
        exp_score_batch_sum = global_add_pool(exp_score,batch)[batch] 
        attention = exp_score/exp_score_batch_sum 
        attention = attention.unsqueeze(-1) # [B*L,1]
        dh = attention * V  # [B*L, hid]       
        dh = global_add_pool(dh,batch) # [B, hid]

        smiles = self.norm[0](smiles + self.drop(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(smiles)
        smiles = self.norm[1](smiles + self.drop(dh))
        
        return smiles,attention


class DeltaCata(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, dropout, task, num_att_layers,num_tgt_layers=2):
        super(DeltaCata, self).__init__()
        self.hidden_dim = hidden_dim

        self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, num_layers=num_layers, drop_rate=dropout)
        
        self.cross_attention_layers =  nn.ModuleList([CrossAttention(hid_dim=hidden_dim,n_heads=1,dropout=dropout) for _ in range(num_att_layers)])

        self.smiles_fc = nn.Linear(167, hidden_dim)
        
        self.task = task
        self.add_module("FC_{}1".format(task), nn.Linear(hidden_dim*4, hidden_dim, bias=True))
        self.add_module("FC_{}2".format(task), nn.Linear(hidden_dim, 1, bias=True))
        
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)        

    def forward(self, wt_data, mut_data, smiles):
        X = wt_data.X
        edge_index = wt_data.edge_index
        subgraph_mask = mut_data.mut_graph_mask
        wt_h_V,mut_h_V = wt_data.node_feat,mut_data.node_feat

        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            wt_h_V = wt_h_V + self.augment_eps * torch.randn_like(wt_h_V)
            mut_h_V = mut_h_V + self.augment_eps * torch.randn_like(mut_h_V)

        if wt_data.batch.shape != mut_data.batch.shape:
            print("error!")
        batch = wt_data.batch
        
        h_V_geo, h_E = get_geo_feat(X, edge_index)


        # wildtype GNN
        wt_h_V = torch.cat([wt_h_V, h_V_geo], dim=-1)
        wt_h_V = self.Graph_encoder(wt_h_V, edge_index, h_E, batch,subgraph_mask) # [num_residue, hidden_dim]
        # mutant GNN
        mut_h_V = torch.cat([mut_h_V, h_V_geo], dim=-1)
        mut_h_V = self.Graph_encoder(mut_h_V, edge_index, h_E, batch,subgraph_mask) # [num_residue, hidden_dim]
        
        delta_h_V = mut_h_V - wt_h_V  #  [num_residue, hidden_dim]


        # cross attention
        smiles = self.smiles_fc(smiles) # [B,hid]
        wt_smiles=smiles
        mut_smiles=smiles
        for layer in self.cross_attention_layers:
            mut_smiles,attention_mut = layer(mut_smiles,mut_h_V,mut_h_V,batch)
            wt_smiles,attention_wt = layer(wt_smiles,wt_h_V,wt_h_V,batch)
        delta_smiles = mut_smiles-wt_smiles


        # mutation effect prediction
        delta_h_V = torch.cat([
            global_max_pool(delta_h_V,batch),
            global_mean_pool(delta_h_V,batch), 
            delta_smiles,
            smiles,
        ],dim=-1)

        delta_outputs = F.elu(self._modules["FC_{}1".format(self.task)](delta_h_V))
        delta_outputs = self._modules["FC_{}2".format(self.task)](delta_outputs)
    
        return delta_outputs.view([-1]), delta_h_V
