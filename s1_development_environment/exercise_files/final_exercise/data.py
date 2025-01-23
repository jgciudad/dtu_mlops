from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting
import matplotlib.pyplot as plt
import torch
import os

DATA_PATH = "/Users/tlj258/Code/dtu_mlops/s1_development_environment/corruptmnist_v1"

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    
    train_image_files = [f for f in os.listdir(DATA_PATH) if f.startswith("train_images")]
    
    train_images, train_target = [], []
    for f in train_image_files:
        train_images.append(torch.load(os.path.join(DATA_PATH, f)))
        train_target.append(torch.load(os.path.join(DATA_PATH, 'train_target_' + f[-4:])))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images = torch.load(os.path.join(DATA_PATH, 'test_images.pt'))
    test_target = torch.load(os.path.join(DATA_PATH, 'test_target.pt'))

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    train = torch.utils.data.TensorDataset(train_images, train_target)
    test = torch.utils.data.TensorDataset(test_images, test_target)

    return train, test

def plot_images(tensor_dataset, num_images=49):
    """Plot images from a TensorDataset."""
    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(7, 7), axes_pad=0.3)
    
    indices = torch.randint(0, len(tensor_dataset), (num_images,))

    for i, rdn_idx in enumerate(indices):
        ax = grid[i]
        ax.imshow(tensor_dataset[rdn_idx][0].squeeze(), cmap='gray')
        ax.set_title(f"Label: {tensor_dataset[rdn_idx][1].item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    plot_images(train_set)

