import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from utils import *
import matplotlib.pyplot as plt
from models.GIN_model import GINConvNet
from models.Multiview_LiSENCE import Multi_LiSENCE
from models.Dil_graph import Dil_graphNet
#from sampler import ImbalancedDatasetSampler


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    best_acc, best_loss = 0, 100
    train_loss, train_acc = 0, 0
    test_loss, test_acc = 0, 0    
    best_auc, test_auc = 0,0
    
    print(torch.__version__)
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_train = torch.Tensor()
    total_label = torch.Tensor()
    train_losses = []
    for batch_idx, data in enumerate(train_loader): 
        data = data.to(device)
        optimizer.zero_grad()
        output,w = model(data)
        loss = loss_fn(output, data.y.view(-1,1).float()).to(device)
        loss = torch.mean(loss).float()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))                                                                                                                                 	
    total_train = torch.cat((total_train, output.cpu()), 0)
    total_label = torch.cat((total_label, data.y.view(-1, 1).cpu()), 0)
    G_train = total_label.detach().numpy().flatten()
    P_train = total_train.detach().numpy().flatten()
    ret = [auc(G_train,P_train),pre(G_train,P_train),recall(G_train,P_train),f1(G_train,P_train),acc(G_train,P_train),mcc(G_train,P_train),spe(G_train,P_train)]
    print('train_auc',ret[0])
    print('train_pre',ret[1])
    print('train_recall',ret[2])
    print('train_f1',ret[3])
    print('train_acc',ret[4])
    print('train_mcc',ret[5])
    print('train_spe',ret[6])
    print('train_loss',np.average(train_losses))
    return G_train, P_train, np.average(train_losses)

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    losses = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output,w = model(data)
            loss = loss_fn(output, data.y.view(-1,1).float())
            loss = torch.mean(loss).float().to(device)
            losses.append(loss.item())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(),np.average(losses),w

# modeling = GINConvNet #Dil_graphNet #GINConvNet #GAT_GCN
# model_st = modeling.__name__
# model_st = 'GINConvNet'#'Dil_graphNet'#GINConvNet' #'GAT_GCN

modeling = Multi_LiSENCE #GINConvNet #GINConvNet #GAT_GCN
model_st = modeling.__name__
model_st = 'Multi_LiSENCE'#Multi_LiSENCE'#Dil_graphNet'#GINConvNet' #'GAT_GCN


cuda_name = "cuda:0"
print('cuda_name:', cuda_name)
    
TRAIN_BATCH_SIZE = 64#512
TEST_BATCH_SIZE = 64#512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
print('\nrunning on ', model_st + '_cyp')
processed_data_file_train = 'data/processed/cyp_train.pt'
processed_data_file_test_1a2 = 'data/processed/cyp_test_1a2.pt'
processed_data_file_valid = 'data/processed/cyp_valid.pt'
processed_data_file_test_2c9 = 'data/processed/cyp_test_2c9.pt'
processed_data_file_test_2c19 = 'data/processed/cyp_test_2c19.pt'
processed_data_file_test_2d6 = 'data/processed/cyp_test_2d6.pt'
processed_data_file_test_3a4 = 'data/processed/cyp_test_3a4.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_valid))):
    print('please run create_data.py to prepare data in pytorch format!')
else:
    train_data = TestbedDataset(root='data', dataset='cyp_train')
    test_1a2_data = TestbedDataset(root='data', dataset='cyp_test_1a2')
    valid_data = TestbedDataset(root='data', dataset='cyp_valid')
    test_2c9_data = TestbedDataset(root='data', dataset='cyp_test_2c9')
    test_2c19_data = TestbedDataset(root='data', dataset='cyp_test_2c19')
    test_2d6_data = TestbedDataset(root='data', dataset='cyp_test_2d6')
    test_3a4_data = TestbedDataset(root='data', dataset='cyp_test_3a4')
    train_set = pd.read_csv('cyp_data/cyp_train.csv')
    lables_unique, counts = np.unique(train_set['score'],return_counts = True)
    class_weights = [sum(counts)/ c for c in counts]
    example_weights = [class_weights[e] for e in train_set['score']]
    sampler = WeightedRandomSampler(example_weights, len(train_set['score']))
    # make data PyTorch mini-batch processing ready
    #train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, sampler=sampler)
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_1a2_loader = DataLoader(test_1a2_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_2c9_loader = DataLoader(test_2c9_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_2c19_loader = DataLoader(test_2c19_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_3a4_loader = DataLoader(test_3a4_data, batch_size=TEST_BATCH_SIZE,shuffle=False)
    test_2d6_loader = DataLoader(test_2d6_data, batch_size=TEST_BATCH_SIZE,shuffle=False)
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    print(model)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss = 100
    best_test_auc = 1000
    best_test_ci = 0
    best_epoch = -1
    patience = 30
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model_file_name = 'model_' + model_st + '_' + 'cyp.model'
    result_file_name = 'result_' + model_st + '_' + 'cyp.csv'
    train_losses=[]
    train_accs=[]
    valid_losses=[]
    valid_accs=[]
    best_acc= 0
    for epoch in range(NUM_EPOCHS):
        G_T,P_T,train_loss = train(model, device, train_loader, optimizer, epoch+1)
        print('predicting for valid data')
        G,P,loss_valid,w= predicting(model, device, valid_loader)
        loss_valid_value = loss_valid
        print('valid_loss',loss_valid)
        print('valid_auc',auc(G,P))
        print('valid_pre',pre(G,P))
        print('valid_recall',recall(G,P))
        print('valid_f1',f1(G,P))
        print('valid_acc',acc(G,P))
        print('valid_mcc',mcc(G,P))
        print('valid_spe',spe(G,P))


        if best_acc < acc(G,P):
                    #torch.save(network.state_dict(), save_path + "model.pth")
            best_acc = acc(G,P)
            best_acc = best_acc
            print("########################################################")
            print("BEST VAL ACC: ",best_acc)
            print("########################################################")
            #print(f"Epoch : {ep} | Loss : {train_loss} | Test Loss : {test_loss} | Acc : {train_acc} | Test Acc : {test_acc}")

        train_losses.append(np.array(train_loss))
        valid_losses.append(loss_valid)
        train_accs.append(acc(G_T,P_T))
        valid_accs.append(acc(G,P))
        b = pd.DataFrame({'value':G,'prediction':P})
        names = 'model_'+'value_validation'+'.csv'
        b.to_csv(names,sep=',') 
        early_stopping(loss_valid, model)
        if early_stopping.early_stop:
            print("Early stopping")
            torch.save(model.state_dict(), model_file_name)
            print('predicting for test data')
            G,P,loss_test_1a2,w_1a2 = predicting(model, device, test_1a2_loader)
            ret_1a2 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp_1a2 ',best_epoch,'auc',ret_1a2[0],'pre',ret_1a2[1],'recall',ret_1a2[2],'f1',ret_1a2[3],'acc',ret_1a2[4],'mcc',ret_1a2[5],'spe',ret_1a2[6])
            G,P,loss_test_2c9,w_2c9 = predicting(model, device, test_2c9_loader)
            ret_2c9 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp_2c9 ',best_epoch,'auc',ret_2c9[0],'pre',ret_2c9[1],'recall',ret_2c9[2],'f1',ret_2c9[3],'acc',ret_2c9[4],'mcc',ret_2c9[5],'spe',ret_2c9[6])
            G,P,loss_test_2c19,w_2c19 = predicting(model, device, test_2c19_loader)
            ret_2c19 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp_2c19 ',best_epoch,'auc',ret_2c19[0],'pre',ret_2c19[1],'recall',ret_2c19[2],'f1',ret_2c19[3],'acc',ret_2c19[4],'mcc',ret_2c19[5],'spe',ret_2c19[6])
            G,P,loss_test_2d6,w_2d6 = predicting(model, device, test_2d6_loader)
            ret_2d6 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp__2d6 ',best_epoch,'auc',ret_2d6[0],'pre',ret_2d6[1],'recall',ret_2d6[2],'f1',ret_2d6[3],'acc',ret_2d6[4],'mcc',ret_2d6[5],'spe',ret_2d6[6])
            G,P,loss_test_3a4,w_3a4 = predicting(model, device, test_3a4_loader)
            ret_3a4 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp_3a4 ',best_epoch,'auc',ret_3a4[0],'pre',ret_3a4[1],'recall',ret_3a4[2],'f1',ret_3a4[3],'acc',ret_3a4[4],'mcc',ret_3a4[5],'spe',ret_3a4[6])
            a = pd.DataFrame({'value':G,'prediction':P})
            name = 'model_'+'value_test'+'.csv'
            a.to_csv(name,sep=',')
            break
        else:
            print('no early stopping')
    df = pd.DataFrame({'train_loss':train_losses,'valid_loss':valid_losses,'train_accs':train_accs,'valid_accs':valid_accs})
    names = 'model_'+'loss_acc'+'.csv'
    df.to_csv(names,sep=',')


# import numpy as np
# import pandas as pd
# import sys, os
# from random import shuffle
# import torch
# import torch.nn as nn
# from torch.utils.data import WeightedRandomSampler
# #from models.gat import GATNet
# from models.gat_gcn import GAT_GCN
# from models.GIN_model import GINConvNet
# from utils import *
# import matplotlib.pyplot as plt
# #from sampler import ImbalancedDatasetSampler
# #import d2l
# #from torch_geometric.utils import degree
# #from matplotlib_inline import backend_inline
# import numpy as np


# #import dgl
# #import torch
# #from scipy.sparse import coo_matrix
# #
# ## Assume eweight is the attention tensor returned from a `GATConv` instance.
# #def weights(eweight):
# ##    eweight = torch.randn(3, 2, 1) # 3 edges, 2 heads
# ##    g = dgl.graph(([0, 1, 2], [1, 0, 1]))
# ##    num_nodes = g.num_nodes()
# ##    src, dst = g.edges(order='eid', form='uv')
# ##    edges = torch.stack([dst, src], dim=0)
# #    attention_adjs = []
# #    num_heads = eweight.shape[1]
# #    for head in range(num_heads):
# #        atten_head_adj = coo_matrix((eweight[:, head, 0], (dst, src)), shape=(num_nodes, num_nodes))
# #        attention_adjs.append(atten_head_adj)

# #def use_svg_display():
# #    """Use the svg format to display a plot in Jupyter.
# #
# #    Defined in :numref:`sec_calculus`"""
# #    backend_inline.set_matplotlib_formats('svg')
# #
# #
# #
# #def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
# #cmap='Reds'):
# ##"""Show heatmaps of matrices."""
# #    use_svg_display()
# #    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
# #    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
# #    sharex=True, sharey=True, squeeze=False)
# #    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
# #        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
# #            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
# #            if i == num_rows - 1:
# #                ax.set_xlabel(xlabel)
# #            if j == 0:
# #                ax.set_ylabel(ylabel)
# #            if titles:
# #                ax.set_title(titles[j])
# #    fig.colorbar(pcm, ax=axes, shrink=0.6)


# # training function at each epoch
# def train(model, device, train_loader, optimizer, epoch):
#     best_acc, best_loss = 0, 100
#     train_loss, train_acc = 0, 0
#     test_loss, test_acc = 0, 0
    
#     best_auc, test_auc = 0,0
    
#     print('Training on {} samples...'.format(len(train_loader.dataset)))
#     model.train()
#     total_train = torch.Tensor()
#     total_label = torch.Tensor()
#     train_losses = []
#     for batch_idx, data in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         #output,w = model(data)
#         output = model(data)
#         loss = loss_fn(output, data.y.view(-1,1).float()).to(device)
#         loss = torch.mean(loss).float()
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())
#         if batch_idx % LOG_INTERVAL == 0:
#             print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
#                                                                            batch_idx * len(data.x),
#                                                                            len(train_loader.dataset),
#                                                                            100. * batch_idx / len(train_loader),
#                                                                            loss.item()))                                                                                                                                 	
#     total_train = torch.cat((total_train, output.cpu()), 0)
#     total_label = torch.cat((total_label, data.y.view(-1, 1).cpu()), 0)
#     G_train = total_label.detach().numpy().flatten()
#     P_train = total_train.detach().numpy().flatten()
#     ret = [auc(G_train,P_train),pre(G_train,P_train),recall(G_train,P_train),f1(G_train,P_train),acc(G_train,P_train),mcc(G_train,P_train),spe(G_train,P_train)]
#     print('train_auc',ret[0])
#     print('train_pre',ret[1])
#     print('train_recall',ret[2])
#     print('train_f1',ret[3])
#     print('train_acc',ret[4])
#     print('train_mcc',ret[5])
#     print('train_spe',ret[6])
#     print('train_loss',np.average(train_losses))
    
       
#     return G_train, P_train, np.average(train_losses)


# def predicting(model, device, loader):
#     model.eval()
#     total_preds = torch.Tensor()
#     total_labels = torch.Tensor()
#     print('Make prediction for {} samples...'.format(len(loader.dataset)))
#     losses = []
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             #output,w = model(data) #original line of code 
#             output = model(data)
#             #show_heatmaps(w, xlabel='Keys', ylabel='Queries')

#             #print(w)
#             #output= model(data)
#             loss = loss_fn(output, data.y.view(-1,1).float())
#             loss = torch.mean(loss).float().to(device)
#             losses.append(loss.item())
#             total_preds = torch.cat((total_preds, output.cpu()), 0)
#             total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
#     #return total_labels.numpy().flatten(),total_preds.numpy().flatten(),np.average(losses),w
#     #return total_labels.numpy().flatten(),total_preds.numpy().flatten(),np.average(losses)


# modeling = GINConvNet #GAT_GCN
# #model_st = modeling.__name__
# model_st = 'GINConvNet' #'GAT_GCN'

# cuda_name = "cuda:0"
# print('cuda_name:', cuda_name)
    
# TRAIN_BATCH_SIZE = 512
# TEST_BATCH_SIZE = 512

# #TRAIN_BATCH_SIZE = 64
# #TEST_BATCH_SIZE = 64

# LR = 0.0005
# #LR = 0.0001
# LOG_INTERVAL = 20
# NUM_EPOCHS = 1000

# print('Learning rate: ', LR)
# print('Epochs: ', NUM_EPOCHS)

# # Main program: iterate over different datasets
# print('\nrunning on ', model_st + '_cyp')
# processed_data_file_train = 'data/processed/cyp_train.pt'
# processed_data_file_test_1a2 = 'data/processed/cyp_test_1a2.pt'
# processed_data_file_valid = 'data/processed/cyp_valid.pt'
# processed_data_file_test_2c9 = 'data/processed/cyp_test_2c9.pt'
# processed_data_file_test_2c19 = 'data/processed/cyp_test_2c19.pt'
# processed_data_file_test_2d6 = 'data/processed/cyp_test_2d6.pt'
# processed_data_file_test_3a4 = 'data/processed/cyp_test_3a4.pt'
# if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_valid))):
#     print('please run create_data.py to prepare data in pytorch format!')
# else:
#     train_data = TestbedDataset(root='data', dataset='cyp_train')
#     test_1a2_data = TestbedDataset(root='data', dataset='cyp_test_1a2')
#     valid_data = TestbedDataset(root='data', dataset='cyp_valid')
#     test_2c9_data = TestbedDataset(root='data', dataset='cyp_test_2c9')
#     test_2c19_data = TestbedDataset(root='data', dataset='cyp_test_2c19')
#     test_2d6_data = TestbedDataset(root='data', dataset='cyp_test_2d6')
#     test_3a4_data = TestbedDataset(root='data', dataset='cyp_test_3a4')
#     train_set = pd.read_csv('cyp_data/cyp_train.csv')
#     lables_unique, counts = np.unique(train_set['score'],return_counts = True)
#     class_weights = [sum(counts)/ c for c in counts]
#     example_weights = [class_weights[e] for e in train_set['score']]
#     sampler = WeightedRandomSampler(example_weights, len(train_set['score']))
#     # make data PyTorch mini-batch processing ready
#     #train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, sampler=sampler)
#     train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
#     valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
#     test_1a2_loader = DataLoader(test_1a2_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
#     test_2c9_loader = DataLoader(test_2c9_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
#     test_2c19_loader = DataLoader(test_2c19_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
#     test_3a4_loader = DataLoader(test_3a4_data, batch_size=TEST_BATCH_SIZE,shuffle=False)
#     test_2d6_loader = DataLoader(test_2d6_data, batch_size=TEST_BATCH_SIZE,shuffle=False)
#     # training the model
#     device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
#     model = modeling().to(device)
#     print(model)
#     loss_fn = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     best_loss = 100
#     best_test_auc = 1000
#     best_test_ci = 0
#     best_epoch = -1
#     patience = 30
#     early_stopping = EarlyStopping(patience=patience, verbose=True)
#     model_file_name = 'model_' + model_st + '_' + 'cyp.model'
#     result_file_name = 'result_' + model_st + '_' + 'cyp.csv'
#     train_losses=[]
#     train_accs=[]
#     valid_losses=[]
#     valid_accs=[]
    
    
    
    
#     best_acc= 0
    
    
#     for epoch in range(NUM_EPOCHS):
#         G_T,P_T,train_loss = train(model, device, train_loader, optimizer, epoch+1)
#         print('predicting for valid data')
#         G,P,loss_valid,w= predicting(model, device, valid_loader)
#         #G,P,loss_valid= predicting(model, device, valid_loader)
#         loss_valid_value = loss_valid
#         print('valid_loss',loss_valid)
#         print('valid_auc',auc(G,P))
#         print('valid_pre',pre(G,P))
#         print('valid_recall',recall(G,P))
#         print('valid_f1',f1(G,P))
#         print('valid_acc',acc(G,P))
#         print('valid_mcc',mcc(G,P))
#         print('valid_spe',spe(G,P))
        
# #        # Store accuracy scores and sample sizes ###########from this line wasn't part
# #        w = w.cpu()
# #        degrees = degree(w[0]).numpy()
# #            
# #        w = w.cuda()
# #        accuracies = valid_accs
# #        sizes = []
# #            
# #        # Accuracy for degrees between 0 and 5
# #        for i in range(0, 6):
# #          mask = np.where(degrees == i)[0]
# #          accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
# #          sizes.append(len(mask))
# #        
# #        # Accuracy for degrees > 5
# #        mask = np.where(degrees > 5)[0]
# #        accuracies.append(accuracy(out.argmax(dim=1)[mask], data.y[mask]))
# #        sizes.append(len(mask))
# #        
# #        # Bar plot
# #        fig, ax = plt.subplots(figsize=(18, 9))
# #        ax.set_xlabel('Node degree')
# #        ax.set_ylabel('Accuracy score')
# #        plt.bar(['0','1','2','3','4','5','>5'],
# #                accuracies,
# #                color='#0A047A')
# #        for i in range(0, 7):
# #            plt.text(i, accuracies[i], f'{accuracies[i]*100:.2f}%',
# #                     ha='center', color='#0A047A')
# #        for i in range(0, 7):
# #            plt.text(i, accuracies[i]//2, sizes[i],
# #                     ha='center', color='white')    ##### up to this line wasn't part
            
        
        
        
        
        
#         if best_acc < acc(G,P):
#             #torch.save(network.state_dict(), save_path + "model.pth")
#            best_acc = acc(G,P)
#            best_acc = best_acc
#         print("########################################################")
#         print("BEST VAL ACC: ",best_acc)
#         print("########################################################")
#         #print(f"Epoch : {ep} | Loss : {train_loss} | Test Loss : {test_loss} | Acc : {train_acc} | Test Acc : {test_acc}")
        
        
#         train_losses.append(np.array(train_loss))
#         valid_losses.append(loss_valid)
#         train_accs.append(acc(G_T,P_T))
#         valid_accs.append(acc(G,P))
#         b = pd.DataFrame({'value':G,'prediction':P})
#         names = 'model_'+'value_validation'+'.csv'
#         b.to_csv(names,sep=',') 
#         early_stopping(loss_valid, model)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             torch.save(model.state_dict(), model_file_name)
#             print('predicting for test data')
#             G,P,loss_test_1a2,w_1a2 = predicting(model, device, test_1a2_loader)
#             #G,P,loss_test_1a2= predicting(model, device, test_1a2_loader)
#             #w_1a2 = torch.corrcoef(torch.tensor(w_1a2))
#             #print(torch.corrcoef(w_1a2[0]))
#             #plt.imshow(w_1a2[0], cmap='hot', interpolation='nearest')
#             #plt.show()
            
#             #torch.save(w_1a2, 'w_1a2-file')
#             #torch.save(w_1a2, 'w_1a2-file')
            
# #            for att_mat in w_1a2:
# #              residual_att = torch.eye(att_mat.size(1))
# #              aug_att_mat = att_mat + residual_att
# #              aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
# #              joint_attentions = torch.zeros(aug_att_mat.size())
# #              joint_attentions[0] = aug_att_mat[0]
# #              for n in range(1, aug_att_mat.size(0)):
# #                  joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
# #              v = joint_attentions
# #              grid_size = int(np.sqrt(aug_att_mat.size(-1)))
# #              mask = v[0,1:].reshape(grid_size, grid_size).detach().numpy()
# #              mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
# #              result = (mask * im).astype("uint8")
# #              fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
# #              ax1.set_title('Original')
# #              ax2.set_title('Attention Map')
# #              _ = ax1.imshow(im)
# #              _ = ax2.imshow(result)
# #            #show_heatmaps(w_1a2, xlabel='Keys', ylabel='Queries')
            
#             #G,P,loss_test_1a2= predicting(model, device, test_1a2_loader)
#             ret_1a2 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
#             print('cyp_1a2 ',best_epoch,'auc',ret_1a2[0],'pre',ret_1a2[1],'recall',ret_1a2[2],'f1',ret_1a2[3],'acc',ret_1a2[4],'mcc',ret_1a2[5],'spe',ret_1a2[6])
            
#             G,P,loss_test_2c9,w_2c9 = predicting(model, device, test_2c9_loader)
#             #G,P,loss_test_2c9 = predicting(model, device, test_2c9_loader)
#             ret_2c9 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
#             print('cyp_2c9 ',best_epoch,'auc',ret_2c9[0],'pre',ret_2c9[1],'recall',ret_2c9[2],'f1',ret_2c9[3],'acc',ret_2c9[4],'mcc',ret_2c9[5],'spe',ret_2c9[6])
#             torch.save(w_2c9, 'w_2c9-file')
            
            
#             G,P,loss_test_2c19,w_2c19 = predicting(model, device, test_2c19_loader)
#             #G,P,loss_test_2c19 = predicting(model, device, test_2c19_loader)
#             ret_2c19 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
#             print('cyp_2c19 ',best_epoch,'auc',ret_2c19[0],'pre',ret_2c19[1],'recall',ret_2c19[2],'f1',ret_2c19[3],'acc',ret_2c19[4],'mcc',ret_2c19[5],'spe',ret_2c19[6])
#             torch.save(w_2c19, 'w_2c19-file')
            
#             G,P,loss_test_2d6,w_2d6 = predicting(model, device, test_2d6_loader)
#             #G,P,loss_test_2d6 = predicting(model, device, test_2d6_loader)
#             ret_2d6 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
#             print('cyp__2d6 ',best_epoch,'auc',ret_2d6[0],'pre',ret_2d6[1],'recall',ret_2d6[2],'f1',ret_2d6[3],'acc',ret_2d6[4],'mcc',ret_2d6[5],'spe',ret_2d6[6])
#             torch.save(w_2d6, 'w_2d6-file')
            
            
            
#             G,P,loss_test_3a4,w_3a4 = predicting(model, device, test_3a4_loader)
#             #G,P,loss_test_3a4 = predicting(model, device, test_3a4_loader)
#             ret_3a4 = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
#             print('cyp_3a4 ',best_epoch,'auc',ret_3a4[0],'pre',ret_3a4[1],'recall',ret_3a4[2],'f1',ret_3a4[3],'acc',ret_3a4[4],'mcc',ret_3a4[5],'spe',ret_3a4[6])
#             torch.save(w_3a4, 'w_3a4-file')
            
            
#             a = pd.DataFrame({'value':G,'prediction':P})
#             name = 'model_'+'value_test'+'.csv'
#             a.to_csv(name,sep=',')
#             break
#         else:
#             print('no early stopping')
#     df = pd.DataFrame({'train_loss':train_losses,'valid_loss':valid_losses,'train_accs':train_accs,'valid_accs':valid_accs})
#     names = 'model_'+'loss_acc'+'.csv'
#     df.to_csv(names,sep=',')

# torch.save(model.state_dict(), "GAT_GCN4.pth")