from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from itertools import combinations
import networkx
from tqdm import tqdm
from torch_geometric.utils.convert import from_networkx
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
import numpy as np

def input_from_networkx(G,num_val=0.1,num_test=0.2):
    pyg_graph = from_networkx(G)
    return pyg_graph
    
class GCN_Bin(torch.nn.Module):
    def __init__(self,num_bins=10):
        super().__init__()
        # self.conv1 = GCNConv(1, 64)
        # self.conv2 = GCNConv(64, 32)
        # self.conv3 = GCNConv(32, 16)
        # self.conv4 = GCNConv(16, 1)
        self.conv1 = GCNConv(1, 32)
        # self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.conv4 = GCNConv(16, num_bins+1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        output = self.conv4(x, edge_index)

        return output

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.conv4 = GCNConv(16, 1)
        # self.conv1 = GCNConv(1, 32)
        # self.conv2 = GCNConv(32, 16)
        # self.conv3 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        output = self.conv4(x, edge_index)


        return output

def train_node_classifier(model, loader, optimizer, criterion, n_epochs=200,print_every=100,bin=True,best_val=1000, outpath='train_model_slurm.pt'):
    # model.train()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    best_acc = best_val
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        for graph in loader:
            mask = graph.training_mask & graph.collection
            graph = graph.to(device)
            out = model(graph)
            loss = criterion(out[mask], graph.y[mask].unsqueeze(1))
            loss.backward()
            total_loss+=loss.item()
            optimizer.step()
            optimizer.zero_grad()
            # pred = out.argmax(dim=1)
            del out
            del graph
            del loss
        if epoch % print_every == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {total_loss:.3f}')
            total_acc = 0
            for graph in loader:
                acc = eval_node_classifier(model, graph, graph.val_mask,criterion,bin)
                total_acc += acc.item()        
                del acc
                del graph
            if total_acc < best_acc:
                torch.save(model, outpath)
                best_acc = total_acc
            print(f' Val Acc: {total_acc:.3f}')
            torch.cuda.empty_cache()


    return model

def test_acc(model,loader,criterion,bin=False):
    total_acc = 0
    for graph in loader:
        acc = eval_node_classifier(model, graph, graph.test_mask,criterion,bin)
        total_acc+=acc.item()
        del acc
        del graph
    print(f' Test Acc: {total_acc:.3f}')
    return total_acc
    
def eval_node_classifier(model, graph, mask,criterion,bin):
    mask = mask & graph.collection
    model.eval()
    torch.no_grad()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    if not bin:
        out = model(graph)
        loss = criterion(out[mask], graph.y[mask].unsqueeze(1))
    else:
        pred = model(graph).argmax(dim=1)
        correct = (pred[mask] == graph.y[mask]).sum()
        loss = int(correct) / int(mask.sum())
    del graph
    del out
    return loss
    
def plot_test(model,graph,mask,criterion,bin=False):
    mask = mask & graph.collection
    model.eval()
    torch.no_grad()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    out = model(graph)
    new_tensor = torch.full_like(out, 45)
    loss = criterion(out[mask], graph.y[mask].unsqueeze(1))
    r_loss = criterion(new_tensor[mask], graph.y[mask].unsqueeze(1))
    print(f"Model Loss:{loss}")
    print(f"Random Loss:{r_loss}")
    if bin:
        pred = model(graph).argmax(dim=1)
        y = pred[mask].cpu().detach()
    else:
        y= out[mask].squeeze(1).cpu().detach()
    x = graph.y[mask].cpu().detach()
    coefficients = np.polyfit(x, y, 1)
    line_of_best_fit = np.polyval(coefficients, x)
    # Plot the line of best fit
    plt.plot(x, line_of_best_fit, color='red', label='Line of Best Fit')
    plt.scatter(x, y,alpha=0.1)
    plt.ylim(top=100,bottom=0)
    # plt.xlim(top=0.2,bottom=0)
    return loss.item(),r_loss.item()