import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, epochs: int = 5, batch_size: int = 32) -> None:
    """Train a model on MNIST."""
    print(lr)

    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()
    
    indices = torch.randperm(len(train_set)).tolist()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    
    losses = []
    for e in range(epochs):
        print(f"Epoch {e}")
        
        epoch_losses = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            images_batch = train_set[batch_indices][0]
            targets_batch = train_set[batch_indices][1]
            
            optimizer.zero_grad()
            output = model(images_batch)
            loss = loss_fn(output, targets_batch)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f"Loss: {losses[-1]}")
    
    print("Training complete")
    torch.save(model.state_dict(), "model.pth")

    plt.plot(range(epochs), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    plt.show()
        
        
@app.command()
def evaluate(model_checkpoint: str, batch_size: int = 32) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    
    _, test_set = corrupt_mnist()
    
    model.eval()
    for i in range(0, len(test_set), batch_size):
        images_batch = test_set[i:i + batch_size][0]
        
        output = model(images_batch)
        if i == 0:
            all_outputs = output
        else:
            all_outputs = torch.cat((all_outputs, output), dim=0)
            
    test_labels = test_set[:][1]
    _, predicted_labels = torch.max(all_outputs, 1)
    accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    
    
    
    

if __name__ == "__main__":
    app()
