import torch as pt
import torch.nn as nn
from torch.utils.data import random_split
from src.dataset import MolDataset
from torch.utils.data import DataLoader
from config import config_data, config_runtime
import joblib # to save scaler

def train(config_data, config_runtime):
    # from src.logger import Logger
    from model import NeuralNet
    # from src.data_encoding import all_resnames, selected_locations
    _ = pt.manual_seed(150)
    pt.set_num_threads(8)
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


    # read datasets
    N = 15000
    dataset = MolDataset("../../data/preprocessed/X.pk", 
                         y_datafile="../../data/preprocessed/Y.pk", normal = True)
    train_dataset, test_dataset = random_split(dataset, [len(dataset) - N, N])
    
    # log
    # print(f"length of the dataset is: {len(dataset)}")
    # logger.print(get_stat_from_dataset(dataset))
    print(f"Train: {len(train_dataset)}")
    # logger.print(get_stat_from_dataset(train_dataset))
    print(f"Test: {len(test_dataset)}")
    # logger.print(get_stat_from_dataset(test_dataset))

    n_categories = 3
    n_features = 42
    learning_rate = 1e-5
    n_hidden = 512
    # n_layers = config_runtime['layers']

    
    model = NeuralNet(n_features,n_hidden,n_categories).to(device)
    print(model)
    
    # loss_fn = nn.CrossEntropyLoss()
    class_weights = pt.Tensor([1.8, 1.8, 0.05])
    class_weights.to(device)
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)
    loss_fn.to(device)
    optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)

    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, Y) in enumerate(dataloader):
            # Compute prediction and loss
            x = X.to(device)
            y = Y[:,None,:].to(device)
            pred = model(x)
            # print(pred,y)
            loss = loss_fn(pred, y.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with pt.no_grad():
            for X, Y in dataloader:
                x = X.to(device)
                y = Y[:,None,:].to(device)
                pred = model(x)
                test_loss += loss_fn(pred, y.float()).item()
        #             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        #     correct /= size
        #     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print(f"Avg loss: {test_loss:>8f} \n")

    # train model
    train_dataloader = DataLoader(train_dataset, batch_size=2048,
                                  shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2048,
                                 shuffle=True, pin_memory=True)
    epochs = 150
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # logger.store_progress(0, is_train=True, epoch=t+1)
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    pt.save(model.state_dict(), "model.pt")
    
    # save scaler 
    scaler = dataset.get_scaler()
    scaler_filename = "scaler.save"
    joblib.dump(scaler, scaler_filename) 
    

if __name__ == '__main__':
    # train model
    train(config_data, config_runtime)