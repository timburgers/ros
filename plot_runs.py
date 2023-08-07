import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd
 
folder_path = "/home/tim/ros/snnblimp_ws/rosbag/new/csv/"
file_names = []
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_names.append(file)

ind = 0
for file in file_names:
    print(ind,") ", file)
    ind +=1
# file_names = ["all_2023-07-28-14-06-15.csv",
#             "all_2023-07-31-11-40-26.csv"]
            #   "all_2023-07-28-14-09-30.csv"]
            #   "all_2023-07-28-12-26-05.csv"]

# Create empty lists to store each column data


number_of_samples = []
for file in file_names:
    with open(folder_path + file, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Read and skip the header row
        total_rows = 0
        for _ in csv_reader:
            total_rows += 1
        number_of_samples.append(total_rows)

t = np.zeros([max(number_of_samples), len(file_names)])
u = np.zeros([max(number_of_samples), len(file_names)])
ref = np.zeros([max(number_of_samples), len(file_names)])
meas = np.zeros([max(number_of_samples), len(file_names)])
i = 0 

file_ind = 0
for file in file_names:
    with open(folder_path + file, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Read and skip the header row
        row_ind = 0
        for row in csv_reader:
           ti,ui,refi,measi,p,i,d,p_d ,error= row  # Unpack the row into separate variables
           t[row_ind,file_ind] =  ti  
           u[row_ind,file_ind] =  ui  
           ref[row_ind,file_ind] =  refi  
           meas[row_ind,file_ind] =  measi  
           row_ind +=1 

    file_ind +=1


# Calculate moving average
window_size = 10
i = 0

# Convert array of integers to pandas series
numbers_series = pd.DataFrame(u)
  
# Get the window of series
# of observations of specified window size
windows = numbers_series.rolling(window_size)
  
# Create a series of moving
# averages of each window
moving_averages = windows.mean()
# moving_averages.dropna(inplace=True)
# Convert pandas series back to list
moving_averages_numpy = moving_averages.to_numpy()

for i in range(len(file_names)):
    plt.plot(t[:,i],ref[:,i])
    plt.plot(t[:,i],meas[:,i],label = str(i))

    #
    plt.plot(t[:number_of_samples[i],i],u[:number_of_samples[i],i],label = str(i))
    plt.plot(t[:number_of_samples[i],i],moving_averages_numpy[:number_of_samples[i],i],label = str(i)+"_filtered")

plt.grid()
plt.legend()
plt.show()
