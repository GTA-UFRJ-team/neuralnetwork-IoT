import configparser
import copy
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import default_dynamic_qconfig, QConfigMapping

from sklearn.preprocessing import MinMaxScaler

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FeatureDataset(Dataset):

    def __init__(self, file_name):

        # read csv files
        df = pd.read_csv(file_name)
        self.df = df

        self.netinfo = df.loc[:, ['saddr', 'daddr', 'sport', 'dport', 'proto']].values

        # load row data into variables
        x_train = df.loc[:, ['stime', 'pkts', 'bytes', 'ltime', 'seq', 'dur', 'mean', 'stddev', 'sum', 'min', 'max', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate', 'flgs_number', 'proto_number', 'state_number', 'TnBPSrcIP', 'TnBPDstIP', 'TnP_PSrcIP', 'TnP_PDstIP', 'TnP_PerProto', 'TnP_Per_Dport', 'AR_P_Proto_P_SrcIP', 'AR_P_Proto_P_DstIP', 'N_IN_Conn_P_SrcIP', 'N_IN_Conn_P_DstIP', 'AR_P_Proto_P_Sport', 'AR_P_Proto_P_Dport', 'Pkts_P_State_P_Protocol_P_DestIP', 'Pkts_P_State_P_Protocol_P_SrcIP']].values

        # Feature Scaling
        sc = MinMaxScaler()
        x_train = sc.fit_transform(x_train)

        # converting to torch tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float)

        self.X_train = torch.unsqueeze(self.X_train,1)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx]
    
    def get_netinfo(self):
        return self.netinfo

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

def test_model(model, loader, device):

    prediction_list = []

    with torch.no_grad():
        for batch, X in enumerate(loader):
            X = X.to(device)

            pred = model(X)
            pred = torch.squeeze(pred)

            prediction_list.append(pred.argmax())

    return prediction_list

def process_csv_file(csv_file_path):
    print(f"Processing file: {csv_file_path}")

    try:
        time.sleep(5)

        test_set = FeatureDataset(csv_file_path)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, drop_last=True)
        network_info = test_set.get_netinfo()

        if quantize == "True":
            qconfig_mapping = (QConfigMapping()
            .set_object_type(nn.LSTM, default_dynamic_qconfig)
            .set_object_type(nn.Linear, default_dynamic_qconfig)
            )

            model_to_quantize = copy.deepcopy(best_model)
            prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, test_loader)

            quantized_model = convert_fx(prepared_model)

            s = time.time()
            prediction_list = test_model(quantized_model, test_loader, device)
            elapsed = time.time() - s
            print("Quantized elapsed time (seconds) ", elapsed)
            print("Quantized processing concluded\n")

        else:
            s = time.time()
            prediction_list = test_model(best_model, test_loader, device)
            elapsed = time.time() - s
            print("Elapsed time (seconds) ", elapsed)
            print("Processing concluded\n")

        mapping = {1: 'DoS-TCP', 2: 'DoS-UDP', 3: 'DoS-HTTP', 4: 'DDoS-TCP', 5: 'DDoS-UDP', 6: 'DDoS-HTTP', 7: 'Keylogging', 8: 'Data Exfiltration', 9: 'OS Fingerprinting', 10: 'Service Scan'}

        for index, prediction in enumerate(prediction_list):
            if prediction.item() != 0:
                print("Detectado ataque de tipo:", mapping[prediction.item()])
                print(f"Dados do ataque:\n IP de origem- {network_info[index][0]}\n IP de destino- {network_info[index][1]}\n Porta de origem- {network_info[index][2]}\n Porta de destino- {network_info[index][3]}\n Protocolo- {network_info[index][4]}\n")

    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty or has no valid data.")
    except pd.errors.ParserError:
        print("Error: Failed to parse the CSV file.")

class MyHandler(FileSystemEventHandler):

    def __init__(self):
        self.processing_flag = False

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith('.csv') and not self.processing_flag:
            csv_file_path = event.src_path
            self.processing_flag = True
            process_csv_file(csv_file_path)
            self.processing_flag = False

    def on_moved(self, event):
        if event.is_directory:
            return

        if event.dest_path.endswith('.csv') and not self.processing_flag:
            csv_file_path = event.dest_path
            self.processing_flag = True
            process_csv_file(csv_file_path)
            self.processing_flag = False

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.txt')

    model = config['model']
    quantization = config['quantization']

    n_features=35
    n_labels=11
    device="cpu"
    architecture=model['architecture']

    quantize=quantization['quantize']
    if not quantize:
        device="cuda" if torch.cuda.is_available() else "cpu"

    best_model = NeuralNetwork(device=device, n_features=n_features, n_labels=n_labels, batch_size=1, architecture=architecture).to(device)

    best_model.load_state_dict(torch.load(model['model_location'] + 'model_' + architecture.lower() + '.pth', map_location=torch.device('cpu')))
    best_model.eval()

    print("Starting observer")
    folder_to_watch = "old_data"

    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
