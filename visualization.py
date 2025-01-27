def plot_predictions(pred, act, start=0, end=None):
    # Set default end to the length of the array if not provided
    if end is None or end > len(pred):
        end = len(pred)
    
    # Slice the arrays according to the input range
    
    pred_range = pred[start:end]
    act_range = act[start:end]

    # Plotting
    plt.figure(figsize=(15, 6))

    # Plot both actual and predicted values within the range
    plt.plot(act_range, label="Actual", color='b', linewidth=2)
    plt.plot(pred_range, label="Predicted", color='r', linestyle='--', linewidth=2)

    # Add labels and title
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Actual vs Predicted Values')
    plt.legend()

    # Show the plot
    plt.show()

plot_predictions(pred, act, start=0, end=None)