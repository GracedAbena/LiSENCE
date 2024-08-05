#MULAGST_MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv,GATConv, GCNConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from models import attention, seqgraph,ann
import torch as th


def node_normalize(t):  #node_normalize is the same as norm diagram
    t = t.float()
    row_sums = t.sum(1) + 1e-12
    output = t / row_sums[:, None]
    output[th.isnan(output) | th.isinf(output)] = 0.0
    return output


# LiSENCE model  
class LiSENCE(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=35, num_features_xt=25, n_filters=16, output_dim=128,embed_dim=128, dropout=0.3):
    
        super(LiSENCE, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        
        D1_nn1 = Sequential(Linear(num_features_xd*10, num_features_xd*10), ReLU(), Linear(num_features_xd*10, num_features_xd*10))
        self.D1_conv1 = GINConv(D1_nn1)
        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10) 
        
        
        self.myconv_g1 = nn.Conv1d(in_channels=num_features_xd*10*2, out_channels=1500,kernel_size=1)#1st conv layer in MuLAG diagram
                       
        
        self.myconv_g2 = nn.Conv1d(in_channels=1500, out_channels=output_dim,kernel_size=1) #2nd conv layer in MuLAG diagram
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
           
        self.attn = attention.MultiAttention([9,15])   
        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        
        self.jola = seqgraph.SequenceGraph(in_channels=128, out_channels=n_filters)

        #self.fc1_xt = nn.Linear(16*35, output_dim)#this was from the baseline paper, replaced it with myconv_xt
        self.myconv_xt = nn.Conv1d(in_channels=560,
                       out_channels=128,
                       kernel_size=1)
          
        self.fc1 = nn.Linear(256, 512)  
        self.out = nn.Linear(512, self.n_output)

        self.fing_enc = ann.FingerEncoder()
 
        
    def forward(self, data):
        
        x, edge_index, edge_attr,batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.attn(x)
        x = node_normalize(x)
      
        target = data.target
        
        #Ligand Encoder Network - LEN
        x,w = self.conv1(x, edge_index,return_attention_weights= True)  # x's shape ([12544, 350]   
       
        x = self.D1_conv1(x,edge_index)  #GIN layer       
             
        x = self.relu(x)             
        
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #print(x.shape,"after concat")
             
        x = x.permute(1, 0).contiguous()  
        x = self.relu(self.myconv_g1(x)) 
        x = x.permute(1, 0).contiguous()
        #print(x.shape, "after fc_g1") 
        x = self.dropout(x)   
               
        x = x.permute(1, 0).contiguous()  
        x = self.relu(self.myconv_g2(x))
        x = x.permute(1, 0).contiguous()
        #
        ############################ Sequence Encoder Network - SEN ##########################
        embedded_xt = self.embedding_xt(target)
        
        embedded_xt = embedded_xt.permute(0,2,1)
        conv_xt = self.jola(embedded_xt) #applying the Joint local-Attention layers
        
        conv_xt = node_normalize(conv_xt)
        
        xt = conv_xt.reshape(-1, 16*35)
        #print(xt.shape,"shape of xt before fc1_xt")  #512,560 
                
        xt = xt.permute(1, 0).contiguous() 
        xt = self.myconv_xt(xt)
        xt = xt.permute(1, 0).contiguous()
        #print(xt.shape,"xt's shape after myconv_xt")

        # concat
        xc = torch.cat((x, xt), 1) #the concatenation of LEN and SEN
        xc = self.fc1(xc) 
        
        xc = self.relu(xc) 
        #xc = torch.cat([gmp(xc, batch), gap(xc, batch)], dim=1) 
        xc = self.dropout(xc)
        out = torch.sigmoid(self.out(xc))            
        
        
        return out,w
        
       












