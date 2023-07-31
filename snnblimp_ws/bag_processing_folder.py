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
        


    if h_ref and optitrack and motor_control:
        # ---> Merge previous df's by closest TIME -> df_final
        # df_final = pd.merge_asof(df_ref, df_optitrack, on="time")
        df_final = pd.merge_asof(df_motor,df_ref, on="time")
        df_final = pd.merge_asof(df_final,df_meas, on="time")

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
    condition = df_final["servo"] == 10
    df_final["dcmotor"][condition] = df_final["dcmotor"][condition] * (-1)



    #Get rid of the servo column
    df_final.drop(["servo"], axis=1, inplace=True)

    df_final["time"] = df_final["time"] - df_final["time"][0]
    df_final.dropna(inplace=True)

    file_csv = file.split(".")[0] + ".csv"
    df_final.to_csv(path_or_buf= rosbag_folder + "csv/" +file_csv, index=False)
 





