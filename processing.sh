#!/bin/bash

pause_interval=${1:-5}
iteration = 1

/usr/local/sbin/argus -P 561 -d -w argus_data.out

rm argus_data.csv
rm current_analysis/treated_data.csv
rm old_data/*
rm detection_results.txt

nohup python3 main_tool.py > detection_results.txt &

while true; do
	#mv argus_data.out current_analysis/
	#cd current_analysis/

	ra -r argus_data.out -s stime flgs proto saddr sport daddr dport pkts bytes state ltime seq dur mean stddev sum min max spkts dpkts sbytes dbytes rate srate drate -c , > argus_data.csv	
	cd current_analysis/
	python3 csv_processing.py

	sleep "$pause_interval"

	mv treated_data.csv "treated_data_${iteration}.csv"
	mv "treated_data_${iteration}.csv" ../old_data

	cd ..

	rm argus_data.csv

	((iteration++))
done


