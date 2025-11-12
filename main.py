from torch_geometric.loader import DataLoader

from data_preprocessing import *
from model_definition import *
from training import *
from visualization import *

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    folder_path = r'C:\Users\USER\source\repos\walerazaq\spatiotemporal-energy-forecasting\cleaned_data'
    horizon = 3
    dataset = PrepareDataset(folder_path, horizon, window_size=12, stride=1)

    # Split data

    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    # Define data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1
    )

    # Initialise the model
    model = graphTS_model(horizon, num_nodes=10, in_features=4, gcn_out_features=8, tr_hidden_size=16, mlp_hidden_size=32, readout='meanmax')

    # Train model
    trainedModel = train(model, train_loader, val_loader, device)

    # Evaluate model
    mse, rmse, mae, pred, act = test_model(trainedModel, test_loader, device)

    plot_predictions(pred, act, start=0, end=None)

    print("done!")

if __name__ == "__main__":
    # import multiprocessing as mp
    # mp.freeze_support()                 # required on Windows
    main()
