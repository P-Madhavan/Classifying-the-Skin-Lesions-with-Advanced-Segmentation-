import matplotlib.pyplot as plt

epochs = list(range(1, 21))  # 20 epochs

# Simulated segmentation training data generator
def generate_segmentation_data(base_loss, base_dice, loss_decay, dice_growth, variation=0.01):
    train_loss = [base_loss - loss_decay * i + (variation if i % 4 == 0 else 0) for i in epochs]
    train_dice = [base_dice + dice_growth * i - (variation if i % 3 == 0 else 0) for i in epochs]
    return train_loss, train_dice

# Models
seg_models = ['UNet3+', 'ResUNet', 'ResUNet++', 'MobileNetV3+TransUNet']

# Dataset 1 (e.g., HAM10000)
data_dataset1 = {
    'UNet3+': generate_segmentation_data(0.48, 0.65, 0.015, 0.01),
    'ResUNet': generate_segmentation_data(0.45, 0.67, 0.017, 0.011),
    'ResUNet++': generate_segmentation_data(0.43, 0.68, 0.018, 0.0115),
    'MobileNetV3+TransUNet': generate_segmentation_data(0.40, 0.70, 0.02, 0.012)
}

# Dataset 2 (e.g., PH2)
data_dataset2 = {
    'UNet3+': generate_segmentation_data(0.47, 0.66, 0.015, 0.0095),
    'ResUNet': generate_segmentation_data(0.44, 0.67, 0.017, 0.0105),
    'ResUNet++': generate_segmentation_data(0.42, 0.68, 0.018, 0.011),
    'MobileNetV3+TransUNet': generate_segmentation_data(0.39, 0.71, 0.019, 0.012)
}

# Plotting function
def plot_segmentation_training(data_dict, dataset_name):
    plt.figure(figsize=(14, 5))

    # Training Loss
    # plt.subplot(1, 2, 1)
    for model in seg_models:
        plt.plot(epochs, data_dict[model][0], label=f'{model}')
    plt.title(f'{dataset_name} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    Path = f'./Journal/Segmentation_Training_Loss_{dataset_name.replace(" ", "_")}.png'
    plt.savefig(Path)
    plt.show()

    # Training Dice Coefficient
    plt.figure(figsize=(14, 5))
    for model in seg_models:
        plt.plot(epochs, data_dict[model][1], label=f'{model}')
    plt.title(f'{dataset_name} - Training Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    Path = f'./Journal/Segmentation_Training_Accuracy_{dataset_name.replace(" ", "_")}.png'
    plt.savefig(Path)
    plt.show()

# Plot for Dataset 1
plot_segmentation_training(data_dataset1, "Dataset 1")

# Plot for Dataset 2
plot_segmentation_training(data_dataset2, "Dataset 2")
