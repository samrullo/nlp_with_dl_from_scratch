import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cpu")


def loss_fn(scores, targets):
    return torch.nn.functional.cross_entropy(scores, targets)


def compute_loss(model, data_loader, loss_fn):
    model.eval()

    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for contexts, targets in tqdm(data_loader, position=0, desc="Val Loss compute"):
            contexts = contexts.to(device)
            scores = model(contexts)
            loss = loss_fn(scores, targets)

            total_loss += loss.item()
            num_samples += contexts.size(0)
    avg_loss = total_loss / num_samples
    model.train()
    return avg_loss


def training_loop(
    n_epochs, optimizer, model, loss_fn, train_loader, val_loader, models_folder
):
    """
    train loop
    """
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(1, n_epochs + 1), position=0, desc="Epoch loop"):
        loss_train = 0
        for contexts, targets in tqdm(
            train_loader, position=0, desc="Within epoch iterations"
        ):
            contexts = contexts.to(device=device)
            targets = targets.to(device=device)
            scores = model(contexts)

            loss = loss_fn(scores, targets)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        loss_train_avg = loss_train / len(train_loader)

        print(f"Epoch {epoch} average train loss : {loss_train_avg}")

        with torch.no_grad():
            val_loss = compute_loss(model, val_loader,loss_fn)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                print(
                    f"epoch {epoch}, saving vae model with best loss of {best_val_loss}"
                )
                torch.save(
                    model.state_dict(), str(models_folder / f"vae_pytorch_best.pt")
                )
    return train_losses

def plot_losses(losses):
    plt.figure(figsize=(10,5))    
    plt.plot(losses)
    plt.title("Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

class MatMulPT(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
    
    def forward(self,x):
        return torch.mm(x, self.weights)

class SimpleCBOWPT(nn.Module):
    def __init__(self,vocab_size, hidden_dim,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_layer=MatMulPT(vocab_size,hidden_dim)
        self.out_layer=MatMulPT(hidden_dim,vocab_size)
    
    def forward(self,batch_x):
        h0 = self.in_layer(batch_x[:,0,:])
        h1 = self.in_layer(batch_x[:,1,:])
        h = (h0+h1)/2
        scores = self.out_layer(h)
        return scores

class CBOWDataset(Dataset):
    def __init__(self,contexts, targets) -> None:
        super().__init__()
        self.contexts = contexts
        self.targets = targets
    
    def __len__(self):
        return self.contexts.size(0)

    def __getitem__(self, index):
        return self.contexts[index], self.targets[index]
