import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv


class BaseGNN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers=2,
                 task='classification',
                 num_classes=None,
                 dropout=0.5,
                 **kwargs):
        """
        Base class for GNN models
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output
            num_layers (int): Number of GNN layers
            task (str): 'classification' or 'regression'
            num_classes (int): Number of classes for classification
            dropout (float): Dropout rate
        """
        super(BaseGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.task = task
        self.dropout = dropout
        
        if self.task == 'classification':
            if num_classes is None:
                raise ValueError("num_classes must be specified for classification")
            self.num_classes = num_classes
        elif self.task not in ['classification', 'regression']:
            raise ValueError("Task must be either 'classification' or 'regression'")
        
        # Initialize layers
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # Input layer
        self.layers.append(self._create_gnn_layer(input_dim, hidden_dim, **kwargs))
        self.activations.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(self._create_gnn_layer(hidden_dim, hidden_dim, **kwargs))
            self.activations.append(nn.ReLU())
        
        # Output layer
        if num_layers > 1:
            self.layers.append(self._create_gnn_layer(hidden_dim, output_dim, **kwargs))
        
        # Task-specific output
        if self.task == 'classification':
            self.classifier = nn.Linear(output_dim, num_classes)
        else:  # regression
            self.regressor = nn.Linear(output_dim, output_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def _create_gnn_layer(self, in_dim, out_dim, **kwargs):
        """To be implemented by child classes"""
        raise NotImplementedError
    
    def forward(self, x, edge_index):
        # Message passing layers
        for i in range(self.num_layers - 1):
            x = self.layers[i](x, edge_index)
            x = self.activations[i](x)
            x = self.dropout_layer(x)
        
        # Last layer
        if self.num_layers >= 1:
            x = self.layers[-1](x, edge_index)
        
        # Task-specific output
        if self.task == 'classification':
            x = self.classifier(x)
            return F.log_softmax(x, dim=-1)
        else:  # regression
            x = self.regressor(x)
            return x
class VanillaGNNLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        
    def forward(self, x, adjacency):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency, x)  # Message passing
        return x

class VanillaGNN(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, num_layers=2,dropout=0.5):
        """
        Args:
            dim_in: Input feature dimension
            dim_h: Hidden layer dimension
            dim_out: Output dimension
            num_layers: Number of GNN layers (minimum 2)
        """
        super().__init__()
        self.num_layers = max(num_layers, 2)   # Ensure at least 2 layers
        self.dropout = dropout
        
        # Input layer
        if num_layers >2:
            self.layers = nn.ModuleList([VanillaGNNLayer(dim_in, dim_h* 2*(num_layers - 2))])
        else:
            self.layers = nn.ModuleList([VanillaGNNLayer(dim_in, dim_h)])
        
        # Hidden layers
        for _ in range(self.num_layers - 2,0,-1):
            
            self.layers.append(VanillaGNNLayer((dim_h* 2*(_)), dim_h * (_)))
        
        # Output layer
        self.layers.append(VanillaGNNLayer(dim_h, dim_out))
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def accuracy(self, y_pred, y_true):
        return (y_pred == y_true).sum().float() / len(y_true)
    
    def create_adjacency(self, edge_index, num_nodes):
        """Create dense adjacency matrix with self-loops"""
        adjacency = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        adjacency += torch.eye(num_nodes, device=edge_index.device)  # Add self-loops
        return adjacency
    
    def forward(self, x, adjacency):
        # First layer
        h = self.layers[0](x, adjacency)
        h = F.relu(h)
        
        # Hidden layers
        for layer in self.layers[1:-1]:
            h = layer(h, adjacency)
            h = F.relu(h)
            h = self.dropout_layer(h)  # Apply dropout after each hidden layer
        # Output layer (no activation)
        h = self.layers[-1](h, adjacency)
        return F.log_softmax(h, dim=1)
    
    def fit(self, data, epochs):
        adjacency = self.create_adjacency(data.edge_index, data.num_nodes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, adjacency)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = self.accuracy(out[data.train_mask].argmax(dim=1), 
                              data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epochs > 200:
                if epoch % 50 == 0:
                    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                    val_acc = self.accuracy(out[data.val_mask].argmax(dim=1), 
                                        data.y[data.val_mask])
                    print(f'Epoch {epoch:>3} | Loss: {loss.item():.3f} | Acc: {acc.item()*100:5.2f}% | '
                        f'Val Loss: {val_loss.item():.3f} | Val Acc: {val_acc.item()*100:5.2f}%')
            else:
                if epoch % 20 == 0:
                    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                    val_acc = self.accuracy(out[data.val_mask].argmax(dim=1), 
                                        data.y[data.val_mask])
                    print(f'Epoch {epoch:>3} | Loss: {loss.item():.3f} | Acc: {acc.item()*100:5.2f}% | '
                        f'Val Loss: {val_loss.item():.3f} | Val Acc: {val_acc.item()*100:5.2f}%')
    
    def test(self, data):
        adjacency = self.create_adjacency(data.edge_index, data.num_nodes)
        self.eval()
        with torch.no_grad():
            out = self(data.x, adjacency)
            acc = self.accuracy(out[data.test_mask].argmax(dim=1), 
                              data.y[data.test_mask])
        return acc.item()
    
class GCNClassifier(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, num_layers=2, dropout=0.5):
        """
        Flexible GCN classifier with configurable layers
        
        Args:
            dim_in: Input feature dimension
            dim_h: Hidden layer dimension
            dim_out: Output dimension (number of classes)
            num_layers: Total number of layers (minimum 2)
            dropout: Dropout probability (0 = no dropout)
        """
        super().__init__()
        self.num_layers = max(num_layers, 2)  # Ensure at least input and output layers
        self.dropout = dropout
        
        # Create GCN layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(dim_in, dim_h))
        

        # Hidden layers
        for _ in range(self.num_layers - 2,0,-1):
            
            self.convs.append(GCNConv(dim_h, dim_h))

        
        # Output layer
        self.convs.append(nn.Linear(dim_h, dim_out))
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(p=dropout)
        
    def accuracy(self, y_pred, y_true):
        return (y_pred == y_true).sum().float() / len(y_true)
    
    def forward(self, x, edge_index):
        # Input layer
        h = self.convs[0](x, edge_index)
        h = F.relu(h)
        h = self.dropout_layer(h)
        
        # Hidden layers
        for conv in self.convs[1:-1]:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout_layer(h)
        
        # Output layer
        h = self.convs[-1](h, edge_index)
        return F.log_softmax(h, dim=1)
    
    def fit(self, data, epochs=200, verbose=True):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), 
                                   lr=0.01, 
                                   weight_decay=5e-4)
        
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = self.accuracy(out[data.train_mask].argmax(dim=1), 
                              data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch % 20 == 0 or epoch == epochs):
                with torch.no_grad():
                    val_out = self(data.x, data.edge_index)
                    val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
                    val_acc = self.accuracy(val_out[data.val_mask].argmax(dim=1), 
                                          data.y[data.val_mask])
                    
                    print(f'Epoch {epoch:>3} | '
                          f'Loss: {loss.item():.3f} | '
                          f'Acc: {acc.item()*100:5.2f}% | '
                          f'Val Loss: {val_loss.item():.3f} | '
                          f'Val Acc: {val_acc.item()*100:5.2f}%')
    
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = self.accuracy(pred[data.test_mask], data.y[data.test_mask])
        return test_acc.item()