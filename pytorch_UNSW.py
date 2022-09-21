import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset

from ray import tune, air
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

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

    def __init__(self, device, n_features, n_labels, batch_size):
        super(NeuralNetwork, self).__init__()

        self.hidden_size = 50 # hidden_layer_size
        self.n_layers = 1 # n_layers
        self.input_size = n_features
        self.output_size = n_labels
        self.batch_size = batch_size
        self.device = device

        # RNN/GRU input:
        #   self.{rnn,gru} = nn.{RNN,GRU}(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)
        #   h_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device)
        #   out, h_n = self.{rnn,gru}(x, h_0)

        # BRNN/BGRU input:
        #   self.{brnn,bgru} = nn.{RNN,GRU}(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=True)
        #   Note: in_features of the following layer must be doubled
        #   h_0 = torch.randn(2*self.n_layers, self.batch_size, self.hidden_size).to(self.device)
        #   out, h_n = self.{rnn,gru}(x, h_0)

        # LSTM input:
        #   self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)
        #   h_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device)
        #   out, h_n = self.{rnn,gru}(x, (h_0, h_0))

        # BLSTM input:
        #   self.blstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=True)
        #   Note: in_features of the following layer must be doubled
        #   h_0 = torch.randn(2*self.n_layers, self.batch_size, self.hidden_size).to(self.device)
        #   out, h_n = self.{rnn,gru}(x, (h_0, h_0))

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.output = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):
        h_0 = torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device)

        out, h_n = self.gru(x, h_0)
        out = self.fc(out)
        out = self.fc2(out)
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

        #print(f"Current Loss: {current_loss}", flush=True)
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

    train_set = FeatureDataset('dataset_new/UNSW_2018_IoT_Botnet_Full5pc_Train_Small.csv')
    test_set = FeatureDataset('dataset_new/UNSW_2018_IoT_Botnet_Full5pc_Test_Small.csv')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    model = NeuralNetwork(device=device, n_features=n_features, n_labels=n_labels, batch_size=batch_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    trained_model, training_loss = train_model(model, optimizer, train_loader, epochs=epochs, batch_size=batch_size, n_features=n_features, n_labels=n_labels, device=device)
    accuracy, test_loss, confusion_matrix, metrics = test_model(trained_model, test_loader, batch_size=batch_size, n_features=n_features, device=device)

    for i in range(11):
        precision = confusion_matrix[i][i]/(sum(confusion_matrix[i])+1e-10)
        recall = confusion_matrix[i][i]/(sum(row[i] for row in confusion_matrix)+1e-10)
        metrics.insert(i, [precision, recall])

    session.report({
        "mean_accuracy": accuracy, 
        "metrics": metrics,
        "training_loss": training_loss,
        "test_loss": test_loss
    })

search_space = {
        "lr": tune.grid_search([1e-3]),
        "batch_size": tune.grid_search([100]),
        "epochs": tune.grid_search([5,10]),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_features": 35,
        "n_labels": 11
}

#algo = OptunaSearch()
reporter = CLIReporter(max_report_frequency=9999999, print_intermediate_tables=False)
scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2
)

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(objective),
        resources = {"cpu": 6, "gpu": 1}
    ),
    tune_config=tune.TuneConfig(
        metric="test_loss",
        mode="min",
        scheduler=scheduler,
    ),
    run_config=air.RunConfig(
        progress_reporter=reporter
    ),
    param_space=search_space,
)

results = tuner.fit()

print("Best config is:", results.get_best_result().config)
