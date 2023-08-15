#!/usr/bin/env python3


import rosbag
import pandas as pd
from os import listdir
import numpy as np
import os

rosbag_folder = "/home/tim/ros/snnblimp_ws/rosbag/new/"
if os.path.isdir(rosbag_folder + "csv"): pass
else: os.mkdir(rosbag_folder + "csv")

all_files = []
for file in os.listdir(rosbag_folder):
    if file.endswith(".bag"):
        all_files.append(file)


for file in all_files: 
    bagfile = rosbag_folder + file

    time_offset = 0
    number_samples = 30

    h_ref         = True
    h_meas        = True
    optitrack     = True
    motor_control = True
    u_pid         = True
    u_snn         = True


    bag = rosbag.Bag(bagfile)

    # ---> Build RADAR dataframe: df_targets
    if h_meas:
        column_names = ['time','h_meas']
        df_meas = pd.DataFrame(columns=column_names)
        
        for topic, msg, t in bag.read_messages(topics='/tfmini_ros_node/TFmini'):
            meas = msg.data
            ts = t.to_sec()
            
            if ts > time_offset:
        
                df_meas = df_meas.append(
                    {'time': ts,
                    'h_meas': meas},
                    ignore_index=True)


    if h_ref:
        column_names = ['time','h_ref']
        df_ref = pd.DataFrame(columns=column_names)
        
        for topic, msg, t in bag.read_messages(topics='/h_ref'):
            ref = msg.data
            ts = t.to_sec()
            
            if ts > time_offset:
        
                df_ref = df_ref.append(
                    {'time': ts,
                    'h_ref': ref},
                    ignore_index=True)
                
    # ---> Build OPTITRACK dataframe: df_optitrack
    # if optitrack:

    #     column_names = ['time','h_meas']
    #     df_optitrack = pd.DataFrame(columns=column_names)
        
    #     for topic, msg, t in bag.read_messages(topics='/optitrack'):
    #         height = msg.position.z
    #         ts = t.to_sec()
            
    #         if ts > time_offset:
        
    #             df_optitrack = df_optitrack.append(
    #                 {'time': ts,
    #                  'h_meas': height},
    #                 ignore_index=True
    #             )
            
        
    if motor_control:
        # ---> Build MOTOR_CONTROL dataframe: df_motor
        column_names = ['time','ccw','cw','servo']
        df_motor = pd.DataFrame(columns=column_names)
        
        for topic, msg, t in bag.read_messages(topics='/motor_control'):
            ccw = msg.ccw_speed
            cw  = msg.cw_speed
            servo = msg.angle
            ts = t.to_sec()
        
            if ts > time_offset:
        
                df_motor = df_motor.append(
                    {'time': ts,
                    'ccw': ccw,
                    'cw': cw,
                    'servo': servo},
                    ignore_index=True
                )
    
    if u_pid:
        # ---> Build MOTOR_CONTROL dataframe: df_motor
        column_names = ['time','pid_p','pid_i','pid_d','pid_pd']
        df_pid = pd.DataFrame(columns=column_names)
        
        for topic, msg, t in bag.read_messages(topics='/u_pid'):
            u_p = msg.pe
            u_i  = msg.ie
            u_d = msg.de
            ts = t.to_sec()
        
            if ts > time_offset:
        
                df_pid = df_pid.append(
                    {'time': ts,
                    'pid_p': u_p,
                    'pid_i': u_i,
                    'pid_d': u_d,
                    'pid_pd':u_p+u_d},
                    ignore_index=True
                )
    
    if u_snn:
        # ---> Build MOTOR_CONTROL dataframe: df_motor
        column_names = ['time','snn_pd','snn_i']
        df_snn = pd.DataFrame(columns=column_names)
        
        for topic, msg, t in bag.read_messages(topics='/u_snn'):
            snn_pd = msg.snn_pd
            snn_i  = msg.snn_i
            ts = t.to_sec()
        
            if ts > time_offset:
        
                df_snn = df_snn.append(
                    {'time': ts,
                    'snn_pd': snn_pd,
                    'snn_i': snn_i},
                    ignore_index=True
                )
    # df_pid.to_csv(path_or_buf= rosbag_folder + "csv/" + "pid.csv", index=False)
    # # df_snn.to_csv(path_or_buf= rosbag_folder + "csv/" + "snn.csv", index=False)
    # df_motor.to_csv(path_or_buf= rosbag_folder + "csv/" + "motor.csv", index=False)
    # df_ref.to_csv(path_or_buf= rosbag_folder + "csv/" + "ref.csv", index=False)
    # df_meas.to_csv(path_or_buf= rosbag_folder + "csv/" + "meas.csv", index=False)
    if h_ref and optitrack and motor_control:
        # ---> Merge previous df's by closest TIME -> df_final
        # df_final = pd.merge_asof(df_ref, df_optitrack, on="time")
        df_final = pd.merge_asof(df_motor,df_ref, on="time")
        df_final = pd.merge_asof(df_final,df_meas, on="time")
        df_final = pd.merge_asof(df_final,df_pid, on="time")
        if u_snn:
            df_final = pd.merge_asof(df_final,df_snn, on="time")

    # elif not radar_targets and optitrack and motor_control:
    #     first_sec = int(df_optitrack["time"].iloc[0])
    #     final_sec = int(df_optitrack["time"].iloc[-1])
    #     time_samples = np.linspace(time_offset+first_sec, final_sec, number_samples*(final_sec-first_sec))
    #     df_samples = pd.DataFrame({"time": time_samples})
    #     df_final = pd.merge_asof(df_optitrack, df_motor, on="time")
    #     df_final = pd.merge_asof(df_samples, df_final, on="time")

    if df_final["cw"].equals(df_final["ccw"]):
        df_final.drop(["ccw"], axis=1, inplace=True)

    df_final.rename(columns={"cw": "dcmotor"}, inplace=True)

    #create a negative dc current
    condition = df_final["servo"] == 1
    df_final["dcmotor"][condition] = df_final["dcmotor"][condition] * (-1)

    #Map the dc motor between -10 and 10 and between [10-100] to [10-15]
    condition = df_final["dcmotor"] > 10
    df_final["dcmotor"][condition] = (df_final["dcmotor"][condition]-10)*5/90+10

    condition = df_final["dcmotor"] < -10
    df_final["dcmotor"][condition] = (df_final["dcmotor"][condition]+10)*5/90-10


    #Map the dc motor between -10 and 10 and between [10-100] to [10-15]
    lim_p = 15
    condition = df_final["pid_p"] > lim_p
    df_final["pid_p"][condition] = (df_final["pid_p"][condition]-lim_p)*5/90+lim_p

    condition = df_final["pid_p"] < -lim_p
    df_final["pid_p"][condition] = (df_final["pid_p"][condition]+lim_p)*5/90-lim_p


    #Map the dc motor between -10 and 10 and between [10-100] to [10-15]
    lim_d = 15
    condition = df_final["pid_d"] > lim_d
    df_final["pid_d"][condition] = (df_final["pid_d"][condition]-lim_d)*5/90+lim_d

    condition = df_final["pid_d"] < -lim_d
    df_final["pid_d"][condition] = (df_final["pid_d"][condition]+lim_d)*5/90-lim_d




    df_final["error"] = df_final["h_ref"] - df_final["h_meas"]

    for index, row in df_final.iterrows():
        if index ==0: df_final["pid_pd"][index] = 0
        else: 
            df_final["pid_pd"][index] = (df_final["error"][index] - df_final["error"][index-1])/0.2*12 + df_final["error"][index] * 10

    #Map the dc motor between -10 and 10 and between [10-100] to [10-15]
    lim_pd = 15
    condition = df_final["pid_pd"] > lim_pd
    df_final["pid_pd"][condition] =(df_final["pid_pd"][condition]-lim_pd)*5/90+lim_pd

    condition = df_final["pid_pd"] < -lim_pd
    df_final["pid_pd"][condition] = (df_final["pid_pd"][condition]+lim_pd)*5/90-lim_pd
    

    #Get rid of the servo column
    df_final.drop(["servo"], axis=1, inplace=True)

    # df_final["time"] = df_final["time"] - df_final["time"][0]
    df_final.dropna(inplace=True)
    df_final["time"] = df_final["time"] - df_final["time"].iloc[0]



    # remove first step which was not responding
    # df_final = df_final.iloc[1200:1700]
    # df_final["time"] = df_final["time"] - df_final["time"].iloc[0]

    # # Find the rows where the reference input changes
    # mask = df_final['h_ref'] != df_final['h_ref'].shift()
    # rows_with_change = df_final.index[mask]
    # rows_with_change = rows_with_change - rows_with_change[0]
    # print("Rows with change: ", rows_with_change)
    # df_final["ref_change_ind"] = -1
    # for i in range(rows_with_change.size):
    #     df_final["ref_change_ind"].iloc[i] = rows_with_change[i]
    # # df_final["ref_change_ind"] = rows_with_change


    df_final=df_final.reindex(columns=["time","dcmotor","h_ref","h_meas","pid_p","pid_i","pid_d","pid_pd","error","snn_pd","snn_i"])


    file_csv = file.split(".")[0] + ".csv"
    df_final.to_csv(path_or_buf= rosbag_folder + "csv/" +file_csv, index=False)
 





