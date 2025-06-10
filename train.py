import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CNN
import os
from dataset import emotional_classes, get_train_data, get_test_data
from utils import get_evaluation
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rich.table import Table
from rich.console import Console
from sklearn import metrics

model_id = 'Emotional_Recognition_1'
LEARNING_RATE = 0.0001
EPOCHS = 200
def main():
    log_dir = f"runs/speech2text_training/{model_id}"
    writer = SummaryWriter(log_dir)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    # --- Model Loading / Initialization ---
    model_path = f"models/{model_id}/model_latest.pth"
    # Check if a saved model checkpoint exists
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        # Load the existing model
        model = CNN.load(model_path).to(device)
    else:
        print("Initializing a new model")
        # Initialize a new CNN instance
        model = CNN().to(device)
    print(model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")
    # model.train()
    ##Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    data = get_train_data('data', batch_size= 128)
    test = get_test_data('data', batch_size=128)
    criterion = nn.CrossEntropyLoss()
    for i in range(EPOCHS):
        for idx,batch in enumerate(data):
            imgs = batch['array']
            labels = batch['label']
            imgs = imgs.to(device)
            labels = labels.to(device)
            # --- Forward Pass & Loss Calculation ---
            # Reset gradients
            optimizer.zero_grad()
            # Perform the forward pass through the model
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(labels.cpu(), output.cpu().detach().numpy(),
                                                list_metrics=["accuracy"])
            if idx + 1 == len(batch):
                print(f"Epoch: {i+1}/{EPOCHS}, Loss: {loss}")
                writer.add_scalar('Train/Loss', loss, i+1)
                writer.add_scalar('Train/Accuracy', training_metrics['accuracy'], i+1)
                writer.close()
        # model.eval()
        print(f"Saving model checkpoint at epoch {i+1}")
        os.makedirs(f"models/{model_id}", exist_ok=True)
                        # model.save(f"models/{model_id}/model_{steps}.pth")
        model.save(f"models/{model_id}/model_latest.pth")
        with torch.no_grad():
            for idx, batch in enumerate(test):
                imgs = batch['array'].to(device)
                labels = batch['label'].to(device)

                output = model(imgs)
                table = Table(title=f"Transcription Examples (Epoch: {i+1})")
                table.add_column("Example #", justify="right", style="cyan")
                table.add_column("Model Output", style="green")
                table.add_column("Ground Truth", style="yellow")
                for example_idx in range(4):
                                pred = np.argmax(output[example_idx].detach().cpu().numpy())
                                truth = labels[example_idx].detach().cpu().numpy()  # Ensure we don't exceed batch size
                                table.add_row(
                                    str(example_idx),
                                    emotional_classes[pred],
                                    batch['text'][example_idx]
                                )
                console = Console()
                console.print(table)

if __name__ == '__main__':
      main()

