import math
import torch
import numpy as np
from typing import Dict


from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from src.NN import SimpleNN
from src.Utils import Load_Dataset, exchange, save_show


def nn_run(modelConfig,num_classes,input_size,X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor):
        device = torch.device(modelConfig["device"])
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=modelConfig["batch_size"], shuffle=True)
        model = SimpleNN(input_size, modelConfig["hidden_size1"], modelConfig["hidden_size2"],
                         modelConfig["hidden_size3"],
                         num_classes).double().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=modelConfig["learning_rate"])

        for epoch in range(modelConfig["nn_epochs"]):
            for i, (inputs, labels) in enumerate(train_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

        return accuracy,predicted


def train(modelConfig: Dict):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(modelConfig["device"])
    load_dataset = Load_Dataset()
    load_dataset.load(modelConfig)
    accuracys = []
    num_classes = load_dataset.n_clusters
    input_size = load_dataset.feature
    X_train_tensor = load_dataset.X_train_tensor.to(device)
    y_train_tensor = load_dataset.y_train_tensor.to(device)
    X_test_tensor = load_dataset.X_test_tensor.to(device)
    y_test_tensor = load_dataset.y_test_tensor.to(device)


    for i in range(modelConfig["step_epochs"]):
        if i == 0:
            predicted = y_train_tensor.clone()
        accuracy, predicted = nn_run(modelConfig, num_classes, input_size, X_train_tensor, predicted,
                                     X_test_tensor, y_test_tensor)
        print(i,"step: accuracy:",accuracy)
        X_train_tensor,X_test_tensor = exchange(X_train_tensor,X_test_tensor)
        y_train_tensor,y_test_tensor = exchange(y_train_tensor,y_test_tensor)
        accuracys.append(accuracy)

    save_show(accuracys,modelConfig)