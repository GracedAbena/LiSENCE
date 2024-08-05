
import torch
from torch import nn
import torch.nn.functional as F


from tkinter import Y
import torch
from torch import nn

class AttentionModel(nn.Module): 
#    def __init__(self, k_size) -> None:
#        super().__init__()
    def __init__(self): 
        super(AttentionModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        
        self.attention_weights = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        
        self.attention_weights = torch.softmax(x, dim=1)
        
        x = self.fc2(x)
        output = torch.softmax(x, dim=1)
        
        return output, self.attention_weights







def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

class FeatAttention(nn.Module):
    def __init__(self, k_size) -> None:
        super().__init__()

        #self.conv = nn.Conv1d(1, 1, k_size, 1, padding=int(k_size/2), bias=False, padding_mode='zeros')
        self.conv = nn.Conv2d(1, 1, k_size, 1, padding=int(k_size/2), bias=False, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # [N, F]
        y = x.unsqueeze(1)
        y = self.conv(y).squeeze(1)
        y = x * self.sigmoid(y)
        return y


class MultiAttention(nn.Module):
    def __init__(self, k_sizes) -> None:
        super().__init__()

        self.attens = nn.ModuleList()
        for k in k_sizes:
            self.attens.append(FeatAttention(k_size=k))
        
    def forward(self, x): #[N, F]
        w = 0
        for a in range(len(self.attens)):
            w += self.attens[a](x)
        #return x + outs#, outs
        return x + w, w


class MyMultiheadAttention(nn.Module): #got BEST VAL ACC:  0.8694598002133643 with this attention
 
    def __init__(self, input_dim, hidden_dim, num_heads): 
        super(MyMultiheadAttention, self).__init__()
    
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x) 
        
        # Splitting into multiple heads
        query = query.view(query.size(0), query.size(1), self.num_heads, -1).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.num_heads, -1).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.num_heads, -1).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        
        # Concatenate and project back
        attention_output = attention_output.transpose(1, 2).contiguous().view(x.size(0), -1, self.hidden_dim)
        output = self.output_projection(attention_output)
        
        return output


#class FCEncoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#
#        self.emb = nn.Embedding(42, 64)
#        self.lin1 = nn.Sequential(
#                        nn.Conv1d(64, 64, 7, 1, padding='same'),
#                        nn.BatchNorm1d(64),
#                        nn.Mish(True))
#
#        self.lin2 = nn.Sequential(
#                        nn.Conv1d(64, 128, 7, 1, padding='same'),
#                        nn.BatchNorm1d(128),
#                        nn.Mish(True))
#
#        self.lin3 = nn.Sequential(
#                        nn.Conv1d(128, 128, 7, 1, padding='same'),
#                        nn.BatchNorm1d(128),
#                        nn.Mish(True))
#
#    def forward(self, x):
#        x = self.emb(x).permute(0,2,1)
#        x = self.lin1(x)
#        x = self.lin2(x)
#        x = self.lin3(x).mean(-1)
#        return x



#import torch 
#import torch.nn as nn 
#import torch.nn.functional as F
#
#class GraphAttentionLayer(nn.Module): 
#    def __init__(self, in_features, out_features): 
#        super(GraphAttentionLayer, self).__init__() 
#        self.in_features = in_features
#        self.out_features = out_features
#    
#        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
#    
#        self.reset_parameters()
#    
#    def reset_parameters(self):
#        nn.init.xavier_uniform_(self.W.data, gain=1.414)
#        nn.init.xavier_uniform_(self.a.data, gain=1.414)
#    
#    def forward(self, input, adj):
#        h = torch.matmul(input, self.W)
#        N = h.size()[0]
#    
#        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1)
#        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(1).view(N, -1), negative_slope=0.2)
#        zero_vec = -9e15*torch.ones_like(e)
#        attention = torch.where(adj > 0, e, zero_vec)
#        attention = F.softmax(attention, dim=1)
#        attention = F.dropout(attention, p=0.6, training=self.training)
#        h_prime = torch.matmul(attention, h)
#    
#        return h_prime, attention
#
#class GraphAttentionNetwork(nn.Module): 
#    def __init__(self, in_features, out_features, n_heads): 
#       super(GraphAttentionNetwork, self).__init__()
#       self.attentions = nn.ModuleList([GraphAttentionLayer(in_features, out_features)]) #for in range(n_heads)])
#    
#    def forward(self, input, adj):
#        head_outputs = []
#        attentions = []
#        for attention_layer in self.attentions:
#            h_prime, attention = attention_layer(input, adj)
#            head_outputs.append(h_prime)
#            attentions.append(attention)
#        
#        output = torch.cat(head_outputs, dim=2)
#        attention_weights = torch.cat(attentions, dim=1)
#    
#        return output, attention_weights



#class SAGE(nn.Module):
#    def __init__(self, in_channels, hidden_channels, num_layers):
#        super(SAGE, self).__init__()
#        self.num_layers = num_layers
#        self.convs = nn.ModuleList()
#        
#        for i in range(num_layers):
#            in_channels = in_channels if i == 0 else hidden_channels
#            self.convs.append(SAGEConv(in_channels,
#                                   hidden_channels))
#    def forward(self, x, adjs):
#        for i, (edge_index, _, size) in enumerate(adjs):
#            x_target = x[:size[1]]  
#            x = self.convs[i]((x, x_target), edge_index)
#            if i != self.num_layers - 1:
#                x = x.relu()
#                x = F.dropout(x, p=0.5, training=self.training)
#        return x
#    def full_forward(self, x, edge_index):
#        for i, conv in enumerate(self.convs):
#            x = conv(x, edge_index)
#            if i != self.num_layers - 1:
#                x = x.relu()
#                x = F.dropout(x, p=0.5, training=self.training)
#        return x





import torch
import torch.nn as nn 
import torch.nn.functional as F


class GraphTransformer(nn.Module): 
    def __init__(self, in_features, out_features, num_heads=1): 
        super(GraphTransformer, self).__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
    
        self.attention_weights = None
    
        self.linear = nn.Linear(in_features, out_features * num_heads)
        self.attention = nn.MultiheadAttention(out_features, num_heads)
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(self, x, graph):
        x = self.linear(x)
        #x = x.permute(1, 0, 2)  # Reshape for multihead attention
        x = x.permute(1, 0)
        x, self.attention_weights = self.attention(x, x,x, need_weights=True)  #self.attention(x, x, x, need_weights=True)
        #x = x.permute(1, 0, 2)  # Restore shape
        x = x.permute(1, 0)
        x = self.layer_norm(x)
    
        return x, self.attention_weights








#
#class GraphTransformer(nn.Module): 
#    def __init__(self, in_features, out_features): 
#        super(GraphTransformer, self).__init__() 
#        self.in_features = in_features 
#        self.out_features = out_features 
#        self.linear_node = nn.Linear(in_features, out_features, bias=False)
#        self.linear_att = nn.Linear(out_features*2, 1, bias=False)
#    
#    def forward(self, x, adj_matrix):
#        x = self.linear_node(x)
#        x = F.relu(x)  # Apply non-linearity to node features
#        h_i = torch.unsqueeze(x, dim=1)
#    
#        N, M = adj_matrix.size(0), adj_matrix.size(1)
#        adj_matrix = adj_matrix.view(N, M, 1, 1)
#    
#        h_i = h_i.repeat(1, 1, M, 1)
#        h_j = h_i.permute(0, 2, 1, 3)
#        
#        a_input = torch.cat([h_i, h_j], dim=3)
#        e = self.linear_att(a_input).squeeze(3)
#        attention = F.softmax(e, dim=2)
#        attention = attention * adj_matrix
#        attention = attention / (torch.sum(attention, dim=2, keepdim=True) + 1e-8)
#        
#        x = torch.sum(attention * h_j, dim=2)
#        return x, attention





#import torch
#import torch.nn as nn 
#import torch.nn.functional as F 
#from torch_geometric.nn import MessagePassing
#
#class GraphTransformer(MessagePassing):
#   def init(self, in_channels, out_channels, heads): super(GraphTransformer, self).init(aggr='add') 
#     self.heads = heads 
#     self.lin = nn.Linear(in_channels, heads * out_channels)
#     self.att = nn.Parameter(torch.Tensor(1, heads, out_channels)) 
#     self.att.requiresGrad = True
#  
#  def forward(self, x, edge_index):
#      x = self.lin(x).view(-1, self.heads, out_channels)
#      edge_index, _ = self.preprocess(edge_index, num_nodes=x.size(0))
#      return self.propagate(edge_index, x=x)
#  
#  def update(self, aggr_out):
#      return aggr_out
#  
#  def message(self, x_i, x_j, edge_index_i, size_i):
#      x_j = x_j.view(-1, self.heads, out_channels)
#      alpha = (x_i * self.att).sum(dim=-1) + (x_j * self.att).sum(dim=-1)
#      alpha = F.leaky_relu(alpha, negative_slope=0.2)
#      alpha = softmax(alpha, edge_index_i)
#      alpha = alpha.view(-1, self.heads, 1)
#      return x_j * alpha







#import torch 
#import torch.nn as nn 
#import torch.nn.functional as F
#
#class MyGraphTransformer(nn.Module): 
#    def __init(self, in_channels, out_channels, heads): 
#        super(MyGraphTransformer, self).__init__() 
#        self.in_channels = in_channels 
#        self.out_channels = out_channels 
#        self.heads = heads
#    
#        self.linear = nn.Linear(in_channels, out_channels * heads)
#        self.att_linear = nn.Linear(in_channels, heads)
#        
#    def forward(self, x, adjacency_matrix):
#        batch_size, num_nodes, _ = x.size()
#        
#        x = self.linear(x).view(batch_size, num_nodes, self.heads, self.out_channels)
#        attention = self.att_linear(x).softmax(-1)
#        
#        adjacency_matrix = adjacency_matrix.unsqueeze(2).expand_as(attention)
#        attention = attention * adjacency_matrix
#        
#        attention = attention.masked_fill(attention == 0, float('-inf'))
#        attention = F.softmax(attention, dim=-1)
#        
#        x = (attention * x).sum(dim=2)
#        
#        return x, attention


#import torch
#import torch.nn as nn 
#import torch.nn.functional as F

#class GraphTransformer(nn.Module): 
#    def __init__(self, in_features, out_features, num_heads):
#        super(GraphTransformer, self).__init__() 
#        self.num_heads = num_heads
#        
#        self.linear = nn.Linear(in_features, out_features * num_heads)
#        self.attention_weights = None
#        #batch_size = data.batch
#    def forward(self, x, adj_matrix):
#        #batch_size, num_nodes, _ = x.size()
#        batch_size,num_nodes = x.size()
#        
#        # Linear transformation of input featu+res
#        x = self.linear(x)
#        
#        # Reshape into multiple heads
#        x = x.view(batch_size, num_nodes, self.num_heads, -1)
#        x = x.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_heads, num_nodes, out_features)
#        
#        # Attention mechanism
#        attention_scores = torch.matmul(x, x.permute(0, 1, 3, 2))  # (batch_size, num_heads, num_nodes, num_nodes)
#        attention_scores = torch.softmax(attention_scores, dim=-1)
#        self.attention_weights = attention_scores.detach().cpu().numpy()
#        
#        # Apply attention to input features
#        x = torch.matmul(attention_scores, x)  # (batch_size, num_heads, num_nodes, out_features)
#        
#        # Reshape back to original shape
#        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
#        
#        return x, self.attention_weights







#import torch 
#import torch.nn as nn 
#import torch.nn.functional as F
#
#class GraphAttentionModule(nn.Module): 
#    def __init__(self, in_features, out_features):
#        super(GraphAttentionModule, self).__init__()
#    
#        self.in_features = in_features
#        self.out_features = out_features
#        
#        self.linear = nn.Linear(in_features, out_features, bias=False)
#        self.attention_weights = nn.Parameter(torch.Tensor(out_features, 1))
#        
#        self.reset_parameters()
#        
#    def reset_parameters(self):
#        nn.init.xavier_uniform_(self.linear.weight)
#        nn.init.xavier_uniform_(self.attention_weights)
#        
#    def forward(self, input, adj_matrix):
#        linear_output = self.linear(input)
#        attention_scores = torch.matmul(linear_output, self.attention_weights)
#        attention_coefficients = F.softmax(attention_scores, dim=1)
#        
#        #output = torch.matmul(adj_matrix.transpose(1, 2), attention_coefficients * linear_output)
#        output = torch.matmul(adj_matrix.transpose(0, 1), attention_coefficients * linear_output)
#        output = output.squeeze()
#        
#        return output, attention_coefficients














#from tkinter import Y
#import torch
#from torch import nn

#
#import torch 
#import torch.nn as nn 
#import torch.nn.functional as F 
#from torch_geometric.nn import MessagePassing
#
#class MyAttentionModule(MessagePassing): 
#    def __init__(self, in_channels, out_channels, heads): 
#        super(MyAttentionModule, self).__init__(aggr='add') 
#        self.in_channels = in_channels 
#        self.out_channels = out_channels
#        self.heads = heads
#        
#        self.lin = nn.Linear(in_channels, heads * out_channels)
#        self.att = nn.Parameter(torch.Tensor(1, heads, 2*out_channels))
#        
#        self.reset_parameters()
#    
#    def reset_parameters(self):
#        nn.init.xavier_uniform_(self.lin.weight)
#        nn.init.xavier_uniform_(self.att)
#    
#    def forward(self, x, edge_index):
#        x = self.lin(x).view(-1, self.heads, self.out_channels)
#        return self.propagate(edge_index, x=x)
#    
#    def message(self, x_i, x_j, edge_index_i, edge_index_j):
#        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
#        alpha = F.leaky_relu(alpha, negative_slope=0.2)
#        
#        alpha = self.softmax(alpha, edge_index_i)
#        return x_j * alpha.view(-1, self.heads, 1)
#    
#    def softmax(self, alpha, edge_index_i):
#        num_nodes = torch.max(edge_index_i) + 1
#        alpha = alpha - alpha.max(dim=1, keepdim=True)[0]
#        alpha = torch.exp(alpha)
#        
#        alpha_sum = self.scatter_add(alpha, edge_index_i, dim=0, dim_size=num_nodes)
#        alpha = alpha / alpha_sum[edge_index_i].squeeze()
#        return alpha
#    
#    def update(self, aggr_out):
#        return aggr_out














#import torch 
#import torch.nn as nn 
#import torch.nn.functional as F
#
#class MyAttentionModule2(nn.Module): 
#    def __init__(self, in_dim, out_dim): 
#        super(MyAttentionModule2, self).__init__()
#    
#        self.in_dim = in_dim
#        self.out_dim = out_dim
#        
#        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))
#        self.a = nn.Parameter(torch.Tensor(2 * out_dim, 1))
#        
#        self.reset_parameters()
#    def reset_parameters(self): 
#    
#        nn.init.xavier_uniform_(self.W)
#        nn.init.xavier_uniform_(self.a)
#        
#    def forward(self, x, edge_index):
#        x = torch.matmul(x, self.W)
#        
#        src, dst = edge_index[0], edge_index[1]
#        x_i = torch.index_select(x, 0, src)
#        x_j = torch.index_select(x, 0, dst)
#        alpha = torch.cat([x_i, x_j], dim=1)
#        alpha = torch.matmul(alpha, self.a).squeeze()
#        alpha = F.leaky_relu(alpha, negative_slope=0.2)
#        alpha = torch.softmax(alpha, dim=0)
#        
#        h_j = torch.index_select(x, 0, dst)
#        h_j = alpha.unsqueeze(1) * h_j
#        h_j = torch.sum(h_j, dim=0)
#        
#        return h_j, alpha
    

#
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#
#class AttentionModule2(nn.Module):
#    def __init__(self, in_features, out_features, num_heads=1, dropout=0.6):
#        super(AttentionModule, self).__init__()
#        self.in_features = in_features
#        self.out_features = out_features
#        self.num_heads = num_heads
#        
#        # Linear transformations for attention mechanism
#        self.linear_layer = nn.Linear(in_features, out_features * num_heads, bias=False)
#        self.attention_weights = nn.Parameter(torch.Tensor(1, num_heads, out_features))
#        self.bias = nn.Parameter(torch.Tensor(out_features))
#        
#        # Dropout layer
#        self.dropout = nn.Dropout(dropout)
#        
#        # Initialization of parameters
#        nn.init.xavier_uniform_(self.attention_weights)
#        nn.init.zeros_(self.bias)
#        
#    def forward(self, x, edge_index):
#        # Linear transformation
#        x = self.linear_layer(x)
#        
#        # Reshaping for multi-head attention
#        x = x.view(-1, self.num_heads, self.out_features)
#        
#        # Attention mechanism
#        attention_scores = torch.matmul(x, self.attention_weights)
#        attention_scores = attention_scores.sum(dim=-1) / self.out_features
#        attention_scores = F.softmax(attention_scores, dim=-1)
#        
#        # Dropout
#        attention_scores = self.dropout(attention_scores)
#        
#        # Node features update
#        out = self.propagate(edge_index, x, attention_scores)
#        
#        # Adding bias
#        out = out + self.bias
#        
#        return out
#    
#    def propagate(self, edge_index, x, attention_scores):
#        src, dest = edge_index
#        
#        # Updating node features based on attention scores and neighboring node features
#        out = torch.zeros_like(x)
#        for i in range(self.num_heads):
#            out[:, i] = src[:, None]
#            out[:, i] = attention_scores[:, i][:, None] * x[src[:, None], i]
#        
#        # Aggregating neighboring features using summation
#        out = scatter_add(out, dest[:, None], dim=0, dim_size=x.size(0))
#        
#        return out
#
#
#
#
#class AttentionModule3(nn.Module): 
#    def __init__(self, in_features, out_features): 
#        super(AttentionModule, self).__init__()
#    
#        self.in_features = in_features
#        self.out_features = out_features
#        
#        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
#        self.a = nn.Parameter(torch.Tensor(2*out_features, 1))
#        
#        self.reset_parameters()
#        
#    def reset_parameters(self):
#        nn.init.xavier_uniform_(self.W)
#        nn.init.xavier_uniform_(self.a)
#        
#    def forward(self, x):
#        h = torch.mm(self.W, x.t())  # Linear transformation
#        N = h.size(1)
#        
#        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
#        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
#        
#        attention = F.softmax(e, dim=1)
#        
#        return attention
#
#
#
#
#def conv_init(conv):
#    if conv.weight is not None:
#        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
#    if conv.bias is not None:
#        nn.init.constant_(conv.bias, 0)
#
#






class FeatAttention(nn.Module):
    def __init__(self, k_size) -> None:
        super().__init__()

        self.conv = nn.Conv1d(1, 1, k_size, 1, padding=int(k_size/2), bias=False, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # [N, F]
        y = x.unsqueeze(1)
        y = self.conv(y).squeeze(1)
        y = x * self.sigmoid(y)
        return y


class MultiAttention(nn.Module):
    def __init__(self, k_sizes) -> None:
        super().__init__()

        self.attens = nn.ModuleList()
        for k in k_sizes:
            self.attens.append(FeatAttention(k_size=k))
        
    def forward(self, x): #[N, F]
        outs = 0
        for a in range(len(self.attens)):
            outs += self.attens[a](x)
        return x + outs

class FeatCorrelation(nn.Module):
    def __init__(self, N) -> None:
        super().__init__()

        self.tan = nn.Tanh()
        self.alpha = nn.Parameter(torch.zeros(1,), False)
        self.A = nn.Parameter(torch.zeros(N,N), False)

        self.conv1 = nn.Conv1d(1, 1, 1, 1)
        self.conv2 = nn.Conv1d(1, 1, 1, 1)
        self.conv = nn.Conv1d(1, 1, 1, 1)

        conv_init(self.conv1)
        conv_init(self.conv2)
        conv_init(self.conv)
        
    def forward(self, x): # [N,F]
        A1, A2 = self.conv1(x.unsqueeze(1)).squeeze(1), self.conv2(x.unsqueeze(1)).squeeze(1).permute(1,0)
        #A1, A2 = x, x.permute(1,0)
        A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
        A1 = self.conv(A1.unsqueeze(1)).squeeze(1)
        A1 = self.A + A1 * self.alpha
        #x = x + torch.matmul(x.permute(1,0), A1).permute(1,0)
        x = x + torch.einsum('vu,tu->tv', A1, x.permute(1,0)).permute(1,0)
        return x
        

if __name__ == '__main__':
    # net = MultiAttention([3,5])
    # x = torch.randn(300, 1000)
    # out = net(x)
    # print(out.shape)

    # x = torch.randn(80, 1000)
    # net = FeatCorrelation(600)
    # out = net(x)
    # print(out.shape)
    # print(out)
    print(torch.eye(5))






def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

class FeatAttention(nn.Module):
    def __init__(self, k_size) -> None:
        super().__init__()

        #self.conv = nn.Conv1d(1, 1, k_size, 1, padding=int(k_size/2), bias=False, padding_mode='zeros')
        self.conv = nn.Conv2d(1, 1, k_size, 1, padding=int(k_size/2), bias=False, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # [N, F]
        #y = x.unsqueeze(1)
#        y = self.conv(y).squeeze(1)
#        y = x * self.sigmoid(y)

        #y = self.conv(x)#.squeeze(1)
        #x = torch.tensor(x)
        x = x.clone().detach().requires_grad_(True)
        y = x * self.sigmoid(x)
        return y


class MultiAttention(nn.Module):
    def __init__(self, k_sizes) -> None:
        super().__init__()

        self.attens = nn.ModuleList()
        for k in k_sizes:
            self.attens.append(FeatAttention(k_size=k))
        
    def forward(self, x): #[N, F]
        outs = 0
        for a in range(len(self.attens)):
            outs += self.attens[a](x)
        return x + outs

class FeatCorrelation(nn.Module):
    def __init__(self, N) -> None:
        super().__init__()

        self.tan = nn.Tanh()
        self.alpha = nn.Parameter(torch.zeros(1,), False)
        self.A = nn.Parameter(torch.zeros(N,N), False)

        self.conv1 = nn.Conv1d(1, 1, 1, 1)
        self.conv2 = nn.Conv1d(1, 1, 1, 1)
        self.conv = nn.Conv1d(1, 1, 1, 1)

        conv_init(self.conv1)
        conv_init(self.conv2)
        conv_init(self.conv)
        
    def forward(self, x): # [N,F]
        A1, A2 = self.conv1(x.unsqueeze(1)).squeeze(1), self.conv2(x.unsqueeze(1)).squeeze(1).permute(1,0)
        #A1, A2 = x, x.permute(1,0)
        A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
        A1 = self.conv(A1.unsqueeze(1)).squeeze(1)
        A1 = self.A + A1 * self.alpha
        #x = x + torch.matmul(x.permute(1,0), A1).permute(1,0)
        x = x + torch.einsum('vu,tu->tv', A1, x.permute(1,0)).permute(1,0)
        return x
        

if __name__ == '__main__':
    # net = MultiAttention([3,5])
    # x = torch.randn(300, 1000)
    # out = net(x)
    # print(out.shape)

    # x = torch.randn(80, 1000)
    # net = FeatCorrelation(600)
    # out = net(x)
    # print(out.shape)
    # print(out)
    print(torch.eye(5))
