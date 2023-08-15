import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd
 
folder_path = "/home/tim/ros/snnblimp_ws/rosbag/new/csv/"
Hz = 5
Kp = 10
Ki = 0.75
Kd = 12


plot_p = False
plot_i = False
plot_d = False
plot_pd = True


file_names = []
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_names.append(file)
file_names = sorted(file_names)


ind = 0
for file in file_names:
    print(ind,") ", file)
    ind +=1


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

t_arr = np.zeros([max(number_of_samples), len(file_names)])
u_arr = np.zeros([max(number_of_samples), len(file_names)])
ref_arr = np.zeros([max(number_of_samples), len(file_names)])
meas_arr = np.zeros([max(number_of_samples), len(file_names)])
i_arr = np.zeros([max(number_of_samples), len(file_names)])
p_arr = np.zeros([max(number_of_samples), len(file_names)])
d_arr = np.zeros([max(number_of_samples), len(file_names)])
pd_arr = np.zeros([max(number_of_samples), len(file_names)])
u_i_ideal = np.zeros([max(number_of_samples), len(file_names)])
u_pd_ideal = np.zeros([max(number_of_samples), len(file_names)])
snn_pd_out = np.zeros([max(number_of_samples), len(file_names)])
# snn_i_out = np.zeros([max(number_of_samples), len(file_names)])

i = 0 
file_ind = 0
for file in file_names:
    with open(folder_path + file, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Read and skip the header row
        row_ind = 0
        integral = 0
        error_prev =0
        for row in csv_reader:
           t,u,ref,meas,p,i,d,p_d,error,snn_pd, snn_i= row  # Unpack the row into separate variables
        #    ti,ui,refi,measi,p,i,d,p_d,error= row  # Unpack the row into separate variables 
        #    t,meas,ref,error,p,i,d,p_d,u= row   #New 
           t_arr[row_ind,file_ind] =  t  
           u_arr[row_ind,file_ind] =  u  
           ref_arr[row_ind,file_ind] =  ref  
           meas_arr[row_ind,file_ind] =  meas
           i_arr[row_ind,file_ind] =  i
           d_arr[row_ind,file_ind] = d
           p_arr[row_ind,file_ind] = p
           pd_arr[row_ind,file_ind] =  p_d  
           snn_pd_out[row_ind,file_ind] = snn_pd
        #    snn_i_out[row_ind,file_ind] = snn_i



        #    integral =  integral + (error*(1/Hz))
        #    u_i_ideal[row_ind,file_ind] = integral *Ki
           error = float(error)
           u_pd_ideal[row_ind,file_ind] = (error-error_prev)*Hz*Kd + error*Kp
           error_prev = error

           row_ind +=1 

    file_ind +=1



for i in range(len(file_names)):
    plt.title("Reference and height")
    plt.plot(t_arr[:-2,i],ref_arr[:-2,i])
    plt.plot(t_arr[:-2,i],meas_arr[:-2,i],label = file_names[i])
    # plt.plot(t[:,i],ref[:,i]-meas[:,i],label = str(i))
plt.legend()
plt.grid()
# plt.show()

if plot_p:
    plt.figure()
    for i in range(len(file_names)):
        plt.plot(t_arr[:number_of_samples[i],i],p_arr[:number_of_samples[i],i],label = str(i))
        plt.title("P controller")
    plt.grid()
    plt.legend()

if plot_d:
    plt.figure()
    for i in range(len(file_names)):
        plt.plot(t_arr[:number_of_samples[i],i],d_arr[:number_of_samples[i],i],label = str(i))
        plt.title("D controller")
    plt.grid()
    plt.legend()

if plot_pd:
    plt.figure()
    for i in range(len(file_names)):
        plt.plot(t_arr[:-2,i],(ref_arr[:-2,i]- meas_arr[:-2,i])*10,label = "error")
        plt.plot(t_arr[:number_of_samples[i],i],snn_pd_out[:number_of_samples[i],i],label = "actual_" + str(i))
        plt.plot(t_arr[:number_of_samples[i],i],u_pd_ideal[:number_of_samples[i],i],label = "ideal_"+str(i))
        plt.title("PD controller")
    plt.grid()
    plt.legend()

if plot_i:
    plt.figure()
    for i in range(len(file_names)):
        plt.plot(t_arr[:number_of_samples[i],i],i_arr[:number_of_samples[i],i],label = str(i))
        plt.title("I controller")
    plt.grid()
    plt.legend()

plt.show()



# plt.figure()
# for i in range(len(file_names)):
#     plt.plot(t[:number_of_samples[i],i],u_p[:number_of_samples[i],i],label = "u_p"+str(i))
#     # plt.plot(t[:number_of_samples[i],i],u_d[:number_of_samples[i],i],label = "u_d"+str(i))
#     # plt.plot(t[:number_of_samples[i],i],u_pd[:number_of_samples[i],i],label = "u_pd"+str(i))
#     # plt.plot(t[:number_of_samples[i],i],u_i[:number_of_samples[i],i],label = "u_i"+str(i))
#     # #
#     # plt.plot(t[:,i],ref[:,i]-meas[:,i],label = "ERROR")
#     # plt.plot(t[:number_of_samples[i],i],snn_pd_out[:number_of_samples[i],i],label = "snn_pd"+str(i))

# #     u_pd_ideal[:number_of_samples[i],i] = np.clip(u_pd_ideal[:number_of_samples[i],i],-10,10) 
# #     # plt.plot(t[:number_of_samples[i],i],u_pd_ideal[:number_of_samples[i],i],label = "u_pd_ideal"+str(i))
# #     # plt.plot(t[:number_of_samples[i],i],u_i_ideal[:number_of_samples[i],i],label = "u_i_ideal"+str(i))

# #     mse = np.mean((u_pd_ideal[:number_of_samples[i],i]-u_pd[:number_of_samples[i],i])**2)
#     # print(mse)