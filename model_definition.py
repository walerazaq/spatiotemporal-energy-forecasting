# Graph pooling

def graph_readout(x, method, batch):
    if method == 'mean':
        return global_mean_pool(x,batch)

    elif method == 'meanmax':
        x_mean = global_mean_pool(x,batch)
        x_max = global_max_pool(x,batch)
        return torch.cat((x_mean, x_max), dim=1)

    elif method == 'sum':
        return global_add_pool(x,batch)

    else:
        raise ValueError('Undefined readout opertaion')

# Model Architecture: EdgeGCN LSTM

class graphTS_model(nn.Module):
    def __init__(self, horizon, num_nodes, in_features, gcn_out_features, tr_hidden_size, mlp_hidden_size, readout):
        super().__init__()

        rd = 2 if readout=='meanmax' else 1

        self.edgeconv = EdgeConv(nn.Sequential(
            nn.Linear(in_features*2, in_features*2),
            nn.ReLU(), 
            nn.Linear(in_features*2, gcn_out_features)
        ))
        
        self.gcn = GCNConv(gcn_out_features, gcn_out_features)
        self.lstm = nn.LSTM(input_size=gcn_out_features*rd, hidden_size=tr_hidden_size, num_layers=1, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(tr_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, horizon) 
        )

        self.readout = readout

    def forward(self, x, adj, batch):
        
        # x shape: (batch_size * num_nodes, seq_len, in_features)
        _, seq_len, _ = x.shape
        
        out_gcn = []

        # Loop through sequence of graphs
        for i in range(seq_len):

            p = x[:, i, :]
        
            # Pass through EdgeConv & GCN
            p = self.edgeconv(p, adj)
            p = self.gcn(p, adj)

            # Pool features from node to graph level
            p = graph_readout(p, self.readout, batch)
            out_gcn.append(p)

        # Stack the output list along a new dimension and reshape for LSTM
        x = torch.stack(out_gcn, dim=0)  # New x shape: (seq_len, batch_size, gcn_out_features * rd)
        
        x = x.permute(1, 0, 2)  # Put batch first and sequence second
        x, _ = self.lstm(x)
        
        # Take the last time step of the LSTM output
        x = x[:, -1, :]

        # Pass through MLP
        x = self.mlp(x)

        # Flatten output
        x = x.reshape(-1)
        
        return x