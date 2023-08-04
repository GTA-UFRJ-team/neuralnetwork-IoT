import numpy as np
import pandas as pd
from datetime import datetime, date

df = pd.read_csv("../argus_data.csv")

df = df.rename(columns={'StartTime': 'stime', 'Flgs': 'flgs', 'Proto': 'proto', 'SrcAddr': 'saddr', 'Sport': 'sport', 'DstAddr': 'daddr', 'Dport': 'dport', 'TotPkts': 'pkts', 'TotBytes': 'bytes', 'State': 'state', 'LastTime': 'ltime', 'Seq': 'seq', 'Dur': 'dur', 'Mean': 'mean', 'StdDev': 'stddev', 'Sum': 'sum', 'Min': 'min', 'Max': 'max', 'SrcPkts': 'spkts', 'DstPkts': 'dpkts', 'SrcBytes': 'sbytes', 'DstBytes': 'dbytes', 'Rate': 'rate', 'SrcRate': 'srate', 'DstRate': 'drate'})


dt_now = date.today()
df['stime'] = pd.to_datetime(df['stime'])
df['datetime'] = pd.to_datetime(dt_now) + pd.to_timedelta(df['stime'].dt.strftime('%H:%M:%S.%f'))

df['unix_timestamp'] = (df['datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
df['unix_timestamp'] += df['stime'].dt.microsecond // 1000
df['stime'] = df['unix_timestamp']

df['ltime'] = pd.to_datetime(df['ltime'])
df['datetime'] = pd.to_datetime(dt_now) + pd.to_timedelta(df['ltime'].dt.strftime('%H:%M:%S.%f'))

df['unix_timestamp'] = (df['datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
df['unix_timestamp'] += df['ltime'].dt.microsecond // 1000
df['ltime'] = df['unix_timestamp']


flgs_dict = {
    'e': 1,
    'es': 2,
    'ed': 3,
    'e*': 4,
    'eg': 5,
    'eU': 6,
    'e&': 7,
    'et': 8,
    'eD': 9
}

df['flgs'] = df['flgs'].str.replace(' ', '')
df['flgs_number'] = df['flgs'].map(flgs_dict)
unknown_value_number = max(flgs_dict.values()) + 1
unknown_values = df['flgs'].loc[df['flgs_number'].isnull()].unique()

for unknown_value in unknown_values:
	flgs_dict[unknown_value] = unknown_value_number
	unknown_value_number += 1

df['flgs_number'] = df['flgs'].map(flgs_dict)

proto_dict = {
    'tcp': 1,
    'arp': 2,
    'udp': 3,
    'icmp': 4,
    'ipv6-icmp': 5
}

df['proto_number'] = df['proto'].map(proto_dict)
unknown_value_number = max(proto_dict.values()) + 1
unknown_values = df['proto'].loc[df['proto_number'].isnull()].unique()

for unknown_value in unknown_values:
	proto_dict[unknown_value] = unknown_value_number
	unknown_value_number += 1

df['proto_number'] = df['proto'].map(proto_dict)

state_dict = {
    'RST': 1,
    'CON': 2,
    'REQ': 3,
    'INT': 4,
    'URP': 5,
    'FIN': 6,
    'ACC': 7,
    'NRS': 8,
    'ECO': 9,
    'TST': 10,
    'MAS': 11
}

df['state_number'] = df['state'].map(state_dict)
unknown_value_number = max(state_dict.values()) + 1
unknown_values = df['state'].loc[df['state_number'].isnull()].unique()

for unknown_value in unknown_values:
	state_dict[unknown_value] = unknown_value_number
	unknown_value_number += 1

df['state_number'] = df['state'].map(state_dict)



df['TnBPSrcIP'] = df.groupby('saddr')['bytes'].transform('sum')

df['TnBPDstIP'] = df.groupby('daddr')['bytes'].transform('sum')

df['TnP_PSrcIP'] = df.groupby('saddr')['pkts'].transform('sum')

df['TnP_PDstIP'] = df.groupby('daddr')['pkts'].transform('sum')

df['TnP_PerProto'] = df.groupby('proto')['pkts'].transform('sum')

df['TnP_Per_Dport'] = df.groupby('dport')['pkts'].transform('sum')

grouped_sum_pkts = df.groupby(['saddr', 'proto'])['pkts'].transform('sum')
grouped_sum_dur = df.groupby(['saddr', 'proto'])['dur'].transform('sum')
df['AR_P_Proto_P_SrcIP'] = grouped_sum_pkts / grouped_sum_dur

grouped_sum_pkts = df.groupby(['daddr', 'proto'])['pkts'].transform('sum')
grouped_sum_dur = df.groupby(['daddr', 'proto'])['dur'].transform('sum')
df['AR_P_Proto_P_DstIP'] = grouped_sum_pkts / grouped_sum_dur

df['N_IN_Conn_P_SrcIP'] = df.groupby(['saddr', 'proto'])['saddr'].transform('count')

df['N_IN_Conn_P_DstIP'] = df.groupby(['daddr', 'proto'])['daddr'].transform('count')

grouped_sum_pkts = df.groupby(['sport', 'proto'])['pkts'].transform('sum')
grouped_sum_dur = df.groupby(['sport', 'proto'])['dur'].transform('sum')
df['AR_P_Proto_P_Sport'] = grouped_sum_pkts / grouped_sum_dur

grouped_sum_pkts = df.groupby(['dport', 'proto'])['pkts'].transform('sum')
grouped_sum_dur = df.groupby(['dport', 'proto'])['dur'].transform('sum')
df['AR_P_Proto_P_Dport'] = grouped_sum_pkts / grouped_sum_dur

df['Pkts_P_State_P_Protocol_P_DestIP'] = df.groupby(['daddr', 'state', 'proto'])['pkts'].transform('sum')

df['Pkts_P_State_P_Protocol_P_SrcIP'] = df.groupby(['saddr', 'state', 'proto'])['pkts'].transform('sum')

df = df.drop(columns=['flgs', 'state', 'datetime', 'unix_timestamp'])

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df.to_csv('treated_data.csv', index=False)
