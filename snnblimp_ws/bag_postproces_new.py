#!/usr/bin/env python3


import rosbag
import pandas as pd
from os import listdir
import numpy as np
import os


limit_u = True
limit_p = True
lim_p = 15
limit_d = True
lim_d = 15
limit_pd = True
lim_pd = 15

include_snn = True

rosbag_folder = "/home/tim/ros/snnblimp_ws/rosbag/new/"
if os.path.isdir(rosbag_folder + "csv"): pass
else: os.mkdir(rosbag_folder + "csv")

all_files = []
for file in os.listdir(rosbag_folder):
    if file.endswith(".bag"):
        all_files.append(file)

all_files = sorted(all_files)

for file in all_files: 
    bagfile = rosbag_folder + file


    bag = rosbag.Bag(bagfile)

    column_names = ['time','meas','ref','error','pid_p','pid_i','pid_d','pid_pd', 'u']
    df_final = pd.DataFrame(columns=column_names)
    
    for topic, msg, t in bag.read_messages(topics='/u_pid'):

        #convert the messages into variables
        meas = msg.meas
        ref = msg.ref
        u_p = msg.pe
        u_i  = msg.ie
        u_d = msg.de
        ts = t.to_sec()
    
    
        df_final = df_final.append(
            {'time': ts,
            'meas': meas,
            'ref':ref,
            'error': ref-meas,
            'pid_p': u_p,
            'pid_i': u_i,
            'pid_d': u_d,
            'pid_pd':u_p+u_d,
            'u':u_p+u_i+u_d},
            ignore_index=True
        )


    if include_snn:
        
        column_names_snn = ['snn_p','snn_i','snn_d','snn_pd','snn_pid']
        df_snn = pd.DataFrame(columns=column_names_snn)

        for topic, msg, t in bag.read_messages(topics='/u_snn'):
            #convert the messages into variables
            # snn_p = msg.snn_p
            snn_i = msg.snn_i
            # snn_d = msg.snn_d
            snn_pd  = msg.snn_pd
            # snn_pid = msg.snn_pid
            ts = t.to_sec()
        
        
            df_snn = df_snn.append(
                {'time': ts,
                # 'snn_p': snn_p,
                'snn_i':snn_i,
                # 'snn_d': snn_d,
                'snn_pd': snn_pd},
                # 'snn_pid': snn_pid},
                ignore_index=True
            )
        
        #Merge the two datasets
        df_final = pd.merge_asof(df_final,df_snn, on="time")

    #If it is an empty recording skip it
    if df_final.empty:
        continue

    #Map the dc motor between -10 and 10 and between [10-100] to [10-15]
    if limit_u:
        condition = df_final["u"] > 10
        df_final["u"][condition] = (df_final["u"][condition]-10)*5/90+10

        condition = df_final["u"] < -10
        df_final["u"][condition] = (df_final["u"][condition]+10)*5/90-10


    #Map the dc motor between -10 and 10 and between [10-100] to [10-15]
    if limit_p:
        condition = df_final["pid_p"] > lim_p
        df_final["pid_p"][condition] = (df_final["pid_p"][condition]-lim_p)*5/90+lim_p

        condition = df_final["pid_p"] < -lim_p
        df_final["pid_p"][condition] = (df_final["pid_p"][condition]+lim_p)*5/90-lim_p


    #Map the dc motor between -10 and 10 and between [10-100] to [10-15]
    if limit_d:
        condition = df_final["pid_d"] > lim_d
        df_final["pid_d"][condition] = (df_final["pid_d"][condition]-lim_d)*5/90+lim_d

        condition = df_final["pid_d"] < -lim_d
        df_final["pid_d"][condition] = (df_final["pid_d"][condition]+lim_d)*5/90-lim_d



    #Map the dc motor between -10 and 10 and between [10-100] to [10-15]
    if limit_pd:
        condition = df_final["pid_pd"] > lim_pd
        df_final["pid_pd"][condition] =(df_final["pid_pd"][condition]-lim_pd)*5/90+lim_pd

        condition = df_final["pid_pd"] < -lim_pd
        df_final["pid_pd"][condition] = (df_final["pid_pd"][condition]+lim_pd)*5/90-lim_pd
    


    df_final.dropna(inplace=True)
    df_final["time"] = df_final["time"] - df_final["time"].iloc[0]


    # remove first step which was not responding
    df_final = df_final.iloc[:9500]
    df_final["time"] = df_final["time"] - df_final["time"].iloc[0]


    file_csv = file.split(".")[0] + ".csv"
    df_final.to_csv(path_or_buf= rosbag_folder + "csv/" +file_csv, index=False)
 





