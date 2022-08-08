import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

class FeatureDataset(Dataset):

    def __init__(self, file_name):

        # read csv files
        df = pd.read_csv(file_name)

        # remove the 5-tuple and non-numerical variables
        #df = df.drop(["pkSeqID","flgs","proto","saddr","daddr","sport","dport","state","attack","category","subcategory","smac","dmac","soui","doui","sco","dco"], axis=1)

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

        self.hidden_size = 60 # hidden_layer_size
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

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=2*self.hidden_size, out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.output = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x):
        h_0 = torch.randn(2*self.n_layers, self.batch_size, self.hidden_size).to(self.device)

        out, h_n = self.lstm(x, (h_0, h_0))
        out = self.fc(out)
        out = self.fc2(out)
        #out = self.fc2(out)
        #out = self.fc2(out)
        out = F.softmax(self.output(out), dim=2)

        return out

def train_model(loader, epochs, batch_size, n_features, n_labels, device):

    model = NeuralNetwork(device=device, n_features=n_features, n_labels=n_labels, batch_size=batch_size).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

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
        current_loss = 0.0

    return model

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

def main():

    k_folds = 3
    epochs = 200
    n_features = 18
    n_labels = 11
    batch_size = 100

    train_set = FeatureDataset('UNSW_2018_IoT_Botnet_Dataset_Balanced_Train_10M.csv')
    test_set = FeatureDataset('dataset_temp/UNSW_2018_IoT_Botnet_Dataset_Reduced.csv')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    save_model = 0
    results = {}
    kfold = KFold(n_splits=k_folds, shuffle=True)

    training_time = 0
    inference_time = 0

    # Training start

    if(device == "cuda"):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.time()

    for fold, (train_ids, validation_ids) in enumerate(kfold.split(train_set)):
        print(f"Fold {fold}\n", flush=True)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_subsampler)
        validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=validation_subsampler)

        model = train_model(train_loader, epochs, batch_size, n_features, n_labels, device)

        accuracy, validation_loss, confusion_matrix, metrics = test_model(model, validation_loader, batch_size, n_features, device)

        results[fold] = accuracy

        print(f"Accuracy: {100*accuracy:>0.2f}%")
        print("Average Loss: ", validation_loss)

        for i in range(n_labels):
            precision = confusion_matrix[i][i]/(sum(confusion_matrix[i])+1e-10)
            recall = confusion_matrix[i][i]/(sum(row[i] for row in confusion_matrix)+1e-10)
            metrics.insert(i, [precision, recall])

        print("Precision and Recall by label: ")
        print(f"{metrics}\n", flush=True)

        if(accuracy > save_model):
            save_model = accuracy
            torch.save(model.state_dict(), 'model_tmp.pth')
            #print("Model saved")

    if(device == "cuda"):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        training_time = start.elapsed_time(end)/1000
    else:
        end = time.time()
        training_time = end-start

    # Training end


    # Printing final training results

    print(f"K-fold Cross Validation Results for {k_folds} folds: ")

    final_sum = 0.0
    for key, value in results.items():
        print(f"Fold {key}: {100*value:>0.2f}%")
        final_sum += value

    print(f"Average: {(100*final_sum/len(results.items())):>0.2f}%")
    print(f"Total training time: {training_time} seconds", flush=True)


    # Testing start

    best_model = NeuralNetwork(device=device, n_features=n_features, n_labels=n_labels, batch_size=batch_size).to(device)
    best_model.load_state_dict(torch.load('model_tmp.pth'))
    best_model.eval()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    if(device == "cuda"):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.time()

    accuracy, test_loss, confusion_matrix, metrics = test_model(best_model, test_loader, batch_size, n_features, device)

    if(device == "cuda"):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end)/1000
    else:
        end = time.time()
        inference_time = end-start

    # Testing end


    # Printing final testing results

    print("\nTesting results:")
    print(f"Accuracy: {100*accuracy:>0.2f}%")
    print(f"Average: {test_loss}")

    for i in range(n_labels):
        precision = confusion_matrix[i][i]/(sum(confusion_matrix[i])+1e-10)
        recall = confusion_matrix[i][i]/(sum(row[i] for row in confusion_matrix)+1e-10)
        metrics.insert(i, [precision, recall])

    print("Precision and Recall by label: ")
    print(f"{metrics}\n", flush=True)

    print(f"Total inference time: {inference_time} seconds")
    print(f"\nDone!")

if __name__ == '__main__':
    print("Starting!", flush=True)
    main()