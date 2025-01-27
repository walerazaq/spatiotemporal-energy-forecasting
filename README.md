# Spatiotemporal Framework for Forecasting Energy Consumption in Smart Manufacturing Systems

## Overview
This project introduces a novel spatiotemporal framework for forecasting energy consumption in smart manufacturing systems. The framework leverages a spatiotemporal neural network to model manufacturing systems as dynamic networks, effectively capturing their complex relationships. The solution aims to enhance the accuracy of energy forecasts, providing a significant improvement over existing methods.

## Key Features
- **Spatiotemporal Representation**: Models manufacturing systems as spatiotemporal networks to capture dynamic manufacturing patterns.
- **Novel Neural Network Architecture**: Utilises a combination of Graph Convolutional Networks (GCN), Edge Convolution layers, and Long Short-Term Memory (LSTM) networks.
- **Superior Performance**: Outperforms state-of-the-art methods with improvements of up to 53% in certain metrics.

## Dataset
The study utilises the publicly available High-resolution Industrial Production Energy (HIPE) dataset, which includes:
- **Source**: Karlsruhe Institute of Technology (KIT), Germany.
- **Data**: Smart meter readings from 10 machines recorded over 3 months.
- **Features**: Active power, apparent power, and their standard deviations over 10-minute intervals.

## Methodology
1. **Dynamic Network Construction**:
   - Machines are modelled as graph nodes.
   - Relationships between machines are captured in adjacency matrices.
2. **Spatiotemporal Neural Network**:
   - **Spatial Module**: Learns local dependencies using EdgeConv, GCN layers, and pooling operations.
   - **Temporal Module**: Captures temporal dependencies using LSTM layers.
3. **Forecasting**:
   - Predicts energy consumption for 1-step and 3-step horizons.
  
![Model Setup](https://github.com/user-attachments/assets/93bafcb7-50f9-4494-b1e4-5b67f8cfa47b)

## Results
The framework was validated against baseline models, including LSTM, GRU, and Spatio-Temporal Graph Convolutional Networks (STGCN). Key performance metrics:
- **1-Step Horizon**:
  - RMSE: 141.42
  - MAE: 28.69
- **3-Step Horizon**:
  - RMSE: 148.78
  - MAE: 34.77

### Benchmark Comparison
| Algorithm | 1-Step RMSE | 1-Step MAE | 3-Step RMSE | 3-Step MAE |
|-----------|-------------|------------|-------------|------------|
| LSTM      | 150.12      | 29.68      | 153.04      | 36.75      |
| GRU       | 152.16      | 31.23      | 156.85      | 39.90      |
| AR        | 156.12      | 51.86      | 170.19      | 74.34      |
| MLP       | 150.34      | 32.28      | 161.97      | 39.55      |
| STGCN     | 143.88      | 29.93      | 158.63      | 40.79      |
| **Proposed Framework** | **141.42** | **28.69** | **148.78** | **34.77** |

The proposed framework consistently outperformed all baseline models, demonstrating superior capability in capturing complex and dynamic spatiotemporal dependencies.


## Contributions
This work is supported by Innovate UK under the Smart Manufacturing Data Hub project (contract no. 10017032).

## Citation
If you use this framework in your work, please cite:
```
Sanni, A., Coleman, S., Kerr, D., & Quinn, J. (2025). Spatiotemporal Framework for Forecasting Energy Consumption in Smart Manufacturing Systems.
```
