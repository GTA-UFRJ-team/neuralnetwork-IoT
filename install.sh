#!/bin/bash

apt install python3-pip flex bison argus-server libpcap-dev -y
pip3 install watchdog torch pandas scikit-learn numpy ray[tune] pyarrow fsspec

chmod +x processing.sh

wget https://github.com/openargus/clients/archive/refs/tags/v3.0.8.4.zip
unzip v3.0.8.4.zip
cd clients-3.0.8.4
./configure
make install
cd ..
rm v3.0.8.4.zip
rm -r clients-3.0.8.4

wget https://github.com/openargus/argus/archive/refs/heads/master-dev.zip
unzip master-dev.zip
cd argus-master-dev
./configure
make install
cd ..
rm master-dev.zip
rm -r argus-master-dev

cd dataset
wget https://gta.ufrj.br/~chagas/UNSW_2018_IoT_Botnet_Full5pc_Train_Small.csv
wget https://gta.ufrj.br/~chagas/UNSW_2018_IoT_Botnet_Full5pc_Test_Small.csv
cd ..

echo DL-SAFE installation complete