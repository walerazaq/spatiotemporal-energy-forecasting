class PrepareDataset(torch.utils.data.Dataset):
    
    def __init__(self, folder_path, horizon, window_size=12, stride=3):

        self.folder_path = folder_path
        self.horizon = horizon
        self.window_size = window_size
        self.stride = stride
        
        # File paths for features and target
        self.files = {
            'P_kW': os.path.join(self.folder_path, 'P_kW.csv'),
            'S_kVA': os.path.join(self.folder_path, 'S_kVA.csv'),
            'std_P_kW': os.path.join(self.folder_path, 'std_P_kW.csv'),
            'std_S_kVA': os.path.join(self.folder_path, 'std_S_kVA.csv'),
            'target': os.path.join(self.folder_path, 'total_power.csv')
        }

        self.data = self.load_data()

    def load_data(self):
        data = []
        
        # Load the input features
        features = {}
        for name, path in self.files.items():
            if name != 'target':
                features[name] = pd.read_csv(path, index_col=0)  # First column is the timestamp
        
        # Load the target
        target = pd.read_csv(self.files['target'], index_col=0)

        # Verify all data have the same time index
        for key, df in features.items():
            assert (df.index == target.index).all(), f"Timestamp mismatch in {key}"
        
        # Extract node features at each timestamp
        feature_list = []
        for key in features:
            feature_list.append(features[key].values)  # Extract as numpy array
        
        # Stack features along a new axis (shape: [rows, num_nodes, num_features])
        features_array = np.stack(feature_list, axis=-1) 

        # Check for NaN values in features_array
        nan_fa = np.isnan(features_array)
        
        # If there are any NaNs, replace with 0
        if nan_fa.any():
            features_array[np.isnan(features_array)] = 0
        
        # Extract target values
        target_array = target.values

        # Apply sliding window and store tensor data
        for i in range(0, len(features_array) - self.window_size - self.horizon, self.stride):
            # Extract the window of data
            window_data = features['P_kW'][i:i + self.window_size]

            # Calculate the adjacency matrix for the current window
            adj_matrix = self.calculate_adj_matrix(window_data)

            # Convert to torch sparse tensor
            adj = torch.tensor(adj_matrix, dtype=torch.float32)
            adj = adj.to_sparse()
            adj = adj.to_sparse_csr()

            # Prepare the node features and target for this window
            X_tensor = torch.tensor(features_array[i:i + self.window_size], dtype=torch.float32)
            X_tensor = X_tensor.permute(1, 0, 2)  # Change the first dimension to the second for proper batching

            if self.horizon == 1:
                y_tensor = torch.tensor(target_array[i + self.window_size], dtype=torch.float32)
            else:
                y_tensor = torch.tensor(target_array[i + self.window_size:i + self.window_size + self.horizon], dtype=torch.float32)
            y_tensor = y_tensor.reshape(-1)  # Flatten the target array

            data.append(Data(x=X_tensor, y=y_tensor, adj=adj))

        return data

    def calculate_adj_matrix(self, window_data):
        """
        Calculate the adjacency matrix based on simultaneous machine activity
        within the given window data.
        """
        # Binary matrix: 1 if machine is active (power > 0), else 0
        binary_matrix = (window_data > 0).astype(int)

        # Number of machines (columns in the dataset)
        n_machines = binary_matrix.shape[1]

        # Initialize adjacency matrix
        adj_matrix = np.zeros((n_machines, n_machines))

        # Calculate simultaneous running time for each machine pair
        for i in range(n_machines):
            for j in range(n_machines):
                # Find times both machines were active
                both_active = np.sum((binary_matrix.iloc[:, i] == 1) & (binary_matrix.iloc[:, j] == 1))
                # Normalise by the total active time of the machine in row i
                total_active_i = np.sum(binary_matrix.iloc[:, i] == 1)
                adj_matrix[i, j] = both_active / total_active_i if total_active_i > 0 else 0

        return adj_matrix
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
