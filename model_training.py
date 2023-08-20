import configparser
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import ray
from ray import tune, air
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

class FeatureDataset(Dataset):

    def __init__(self, file_name):

        # read csv files
        df = pd.read_csv(file_name)

        # load row data into variables
        x_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values

        # Feature Scaling
        sc = MinMaxScaler()
        x_train = sc.fit_transform(x_train)

        # converting to torch tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.long)

        self.X_train = torch.unsqueeze(self.X_train,1)
        self.y_train = torch.unsqueeze(self.y_train,1)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

class NeuralNetwork(nn.Module):

    def __init__(self, device, n_features, n_labels, batch_size, architecture):
        super(NeuralNetwork, self).__init__()

        self.hidden_size = 100 # hidden_layer_size
        self.n_layers = 1 # n_layers
        self.input_size = n_features
        self.output_size = n_labels
        self.batch_size = batch_size
        self.device = device
        self.architecture = architecture

        if(architecture == "MLP1" or architecture == "MLP2"):
            self.hidden_size = 100

            self.fc = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
            self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        elif(architecture == "RNN1" or architecture == "RNN2"):
            if(architecture == "RNN1"):
                self.hidden_size = 60
            if(architecture == "RNN2"):
                self.hidden_size = 100

            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)

        elif(architecture == "RNND"):
            self.hidden_size = 100

            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)
            self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        elif(architecture == "LSTM1"):

            self.hidden_size = 32
            self.n_layers = 2

            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)

        elif(architecture == "LSTMD"):

            self.hidden_size = 128
            self.n_layers = 2

            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)

            self.hidden_size = 32
            self.last_size = 128

            self.fc = nn.Linear(in_features=self.last_size, out_features=self.hidden_size)

            self.hidden_size = 10
            self.last_size = 32

            self.fc2 = nn.Linear(in_features=self.last_size, out_features=self.hidden_size)

        elif(architecture == "BLSTM1"):

            self.hidden_size = 12

            self.blstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=True)
            self.hidden_size = 24 

        self.output = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
        
    def forward(self, x):
        
        if(self.architecture in ("MLP1", "MLP2", "RNN1", "RNN2", "RNND", "LSTM1", "LSTMD")):
            if(self.architecture == "LSTMD"):
                self.hidden_size = 128

            h_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device)

            if(self.architecture in ("MLP1", "MLP2")):            
                out = self.fc(x)
                out = self.fc2(out)
                out = self.fc2(out)

                if(self.architecture == "MLP2"):
                    out = self.fc2(out)

            if(self.architecture in ("RNN1", "RNN2", "RNND")):
                out, h_n = self.rnn(x, h_0)

                if(self.architecture == "RNND"):
                    out = self.fc(out)
                    out = self.fc(out)
                    out = self.fc(out)

            if(self.architecture in ("LSTM1", "LSTMD")):
                out, h_n = self.lstm(x, (h_0, h_0))

                if(self.architecture == "LSTMD"):
                    out = self.fc(out)
                    out = self.fc2(out)

        elif (self.architecture in ("BLSTM1")):
            self.hidden_size = 12

            h_0 = torch.randn(2*self.n_layers, self.batch_size, self.hidden_size).to(self.device)
            out, h_n = self.blstm(x, (h_0, h_0))

        out = F.softmax(self.output(out), dim=2)

        return out

def train_model(model, optimizer, loader, epochs, batch_size, n_features, n_labels, device):

    loss_func = nn.CrossEntropyLoss()
    training_loss = []

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}", flush=True)

        current_loss = 0.0

        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)

            if(X.shape[0] != batch_size):
                continue

            optimizer.zero_grad()
            pred = model(X)

            y = torch.flatten(y)
            pred = torch.squeeze(pred)

            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()

        print(f"Current Loss: {current_loss}", flush=True)
        training_loss.append(current_loss)
        current_loss = 0.0

    return model, training_loss

def test_model(model, loader, batch_size, n_features, device):

    loss_func = nn.CrossEntropyLoss()

    num_batches = len(loader)
    total, correct, test_loss = 0, 0, 0
    confusion_matrix = []
    metrics = []

    for i in range(n_features):
        row = []
        for j in range(n_features):
            row.append(0)
        confusion_matrix.append(row)

    with torch.no_grad():
        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)

            if(X.shape[0] != batch_size):
                continue

            pred = model(X)

            y = torch.flatten(y)
            pred = torch.squeeze(pred)

            #total += y.size(0)
            #_, predicted = torch.max(pred.data, 1)
            #correct += (predicted == y).sum().item()

            prediction_list = pred.argmax(1).tolist()
            y_list = y.tolist()

            for i in range(len(y_list)):
                confusion_matrix[prediction_list[i]][y_list[i]] += 1
                total += 1
                if(prediction_list[i] == y_list[i]):
                    correct += 1

            test_loss += loss_func(pred, y).item()

    accuracy = correct/total

    test_loss /= num_batches

    return accuracy, test_loss, confusion_matrix, metrics

def objective(config):
    
    epochs=config["epochs"]
    batch_size=config["batch_size"]
    n_features=config["n_features"]
    n_labels=config["n_labels"]
    device=config["device"]
    lr=config["lr"]
    architecture=config["architecture"]
    dataset=config["dataset"]

    save_model = 0
    kfold = KFold(n_splits= 5, shuffle=True)

    train_set = FeatureDataset(dataset['dataset_folder'] + dataset['train_dataset_name'])
    test_set = FeatureDataset(dataset['dataset_folder'] + dataset['test_dataset_name'])

    for fold, (train_ids, validation_ids) in enumerate(kfold.split(train_set)):
        print(f"Fold {fold}\n", flush=True)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_subsampler)
        validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=validation_subsampler)

        model = NeuralNetwork(device=device, n_features=n_features, n_labels=n_labels, batch_size=batch_size, architecture=architecture).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        trained_model, training_loss = train_model(model, optimizer, train_loader, epochs=epochs, batch_size=batch_size, n_features=n_features, n_labels=n_labels, device=device)
        accuracy, validation_loss, confusion_matrix, metrics = test_model(trained_model, validation_loader, batch_size=batch_size, n_features=n_features, device=device)

        for i in range(n_labels):
            precision = confusion_matrix[i][i]/(sum(confusion_matrix[i])+1e-10)
            recall = confusion_matrix[i][i]/(sum(row[i] for row in confusion_matrix)+1e-10)
            metrics.insert(i, [precision, recall])

        if(accuracy > save_model):
            save_model = accuracy
            torch.save(model.state_dict(), 'model_tmp.pth')

        session.report({
            "fold": fold,
            "mean_accuracy": accuracy, 
            "metrics": metrics,
            "training_loss": training_loss,
            "validation_loss": validation_loss
    })

    best_model = NeuralNetwork(device=device, n_features=n_features, n_labels=n_labels, batch_size=batch_size, architecture=architecture).to(device)
    best_model.load_state_dict(torch.load('model_tmp.pth'))
    best_model.eval()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    accuracy, test_loss, confusion_matrix, metrics = test_model(best_model, test_loader, batch_size, n_features, device)

    for i in range(n_labels):
        precision = confusion_matrix[i][i]/(sum(confusion_matrix[i])+1e-10)
        recall = confusion_matrix[i][i]/(sum(row[i] for row in confusion_matrix)+1e-10)
        metrics.insert(i, [precision, recall])

    if(accuracy > save_model):
        save_model = accuracy
        torch.save(model.state_dict(), 'model_tmp.pth')

    print(f"\nCurrent grid (epochs:{epochs}, batch size:{batch_size}, lr:{lr}, architecture:{architecture}) testing results:")
    print(f"Accuracy: {100*accuracy:>0.2f}%")
    print(f"Average: {test_loss}")

    for i in range(n_labels):
        precision = confusion_matrix[i][i]/(sum(confusion_matrix[i])+1e-10)
        recall = confusion_matrix[i][i]/(sum(row[i] for row in confusion_matrix)+1e-10)
        metrics.insert(i, [precision, recall])

    print("Precision and Recall by label: ")
    print(f"{metrics}\n", flush=True)

def main():
    ray.init(log_to_driver=False)
    config = configparser.ConfigParser()
    config.read('config.txt')
    
    hyperparameters = config['hyperparameters']
    dataset = config['dataset']
    pc_specs = config['pc-specs']

    search_space = {
        "lr": tune.grid_search(list(map(float, hyperparameters['lr'].strip('][').split(',')))),
        "batch_size": tune.grid_search(list(map(int, hyperparameters['batch_size'].strip('][').split(',')))),
        "epochs": tune.grid_search(list(map(int, hyperparameters['epochs'].strip('][').split(',')))),
        #"device": "cuda" if torch.cuda.is_available() else "cpu",
        "device": "cpu",
        "n_features": 35,
        "n_labels": 11,
        "architecture": tune.grid_search(hyperparameters['architecture'].strip('][').split(',')),
        "dataset": dataset
    }

    reporter = CLIReporter(max_report_frequency=9999999, print_intermediate_tables=False)
    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective),
            resources = {"cpu": pc_specs['num_cpu'], "gpu": pc_specs['num_gpu']}
        ),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            #scheduler=scheduler,
        ),
        run_config=air.RunConfig(
            progress_reporter=reporter,
            log_to_file='tuning_test_records.txt'
        ),
        param_space=search_space
    )

    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)

if __name__ == '__main__':
    print("Starting!", flush=True)
    main()
