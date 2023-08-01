import matplotlib.pyplot as plt
import numpy as np
import csv
import os
 
folder_path = "/home/tim/ros/snnblimp_ws/rosbag/best/csv/"
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
           ti,ui,refi,measi= row  # Unpack the row into separate variables
           t[row_ind,file_ind] =  ti  
           u[row_ind,file_ind] =  ui  
           ref[row_ind,file_ind] =  refi  
           meas[row_ind,file_ind] =  measi  
           row_ind +=1 

    file_ind +=1
        

for i in range(len(file_names)):
    # plt.plot(t[:,i],ref[:,i])
    # plt.plot(t[:,i],meas[:,i],label = str(i))

    #
    plt.plot(t[:number_of_samples[i],i],u[:number_of_samples[i],i],label = str(i))

plt.grid()
plt.legend()
plt.show()
