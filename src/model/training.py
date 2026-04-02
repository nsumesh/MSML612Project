import torch
import torch.nn as nn
from pathlib import Path
from src.model.transformer import LapTimeTransformer
from src.data.dataset_file import get_dataloaders

def train_transformer(epochs, lr, batch_size=64, horizon=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loader, validation_loader, testing_loader = get_dataloaders(batch_size=batch_size)
    Path("models").mkdir(exist_ok=True)
    model = LapTimeTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_function = nn.MSELoss()

    best_validation_loss = float("inf")
    for i in range(epochs):
        model.train()
        training_loss = 0
        for data, label in training_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            prediction = output[:,-horizon:]
            loss = loss_function(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss+=loss.item()
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for data, label in validation_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                prediction = output[:,-horizon:]
                validation_loss+=loss_function(prediction, label).item()
        avg_val = validation_loss/len(validation_loader)
        if avg_val<best_validation_loss:
            best_validation_loss = avg_val
            torch.save(model.state_dict(), "models/best_model.pth")
        print(f"Epoch Number {i+1}, Training Loss : {training_loss/len(training_loader):.4f}, Validation Loss : {validation_loss/len(validation_loader):.4f}")

if __name__ == "__main__":
    train_transformer(epochs=50, lr=1e-3)
