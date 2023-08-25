import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})
 
folder_path = "/home/tim/ros/snnblimp_ws/rosbag/new/csv/"
Hz = 10
Kp = 9
Ki = 0.1
Kd = 14
old_layout = False

plot_p = False
plot_i = True
plot_d = False
plot_pd = True
plot_ideal = True


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

# Init PID arrays
t_arr = np.zeros([max(number_of_samples), len(file_names)])
ref_arr = np.zeros([max(number_of_samples), len(file_names)])
meas_arr = np.zeros([max(number_of_samples), len(file_names)])
error_arr = np.zeros([max(number_of_samples), len(file_names)])
i_arr = np.zeros([max(number_of_samples), len(file_names)])
p_arr = np.zeros([max(number_of_samples), len(file_names)])
d_arr = np.zeros([max(number_of_samples), len(file_names)])
pd_arr = np.zeros([max(number_of_samples), len(file_names)])
u_arr = np.zeros([max(number_of_samples), len(file_names)])

# Init SNN arrays
snn_p_arr = np.zeros([max(number_of_samples), len(file_names)])
snn_i_arr = np.zeros([max(number_of_samples), len(file_names)])
snn_d_arr = np.zeros([max(number_of_samples), len(file_names)])
snn_pd_arr = np.zeros([max(number_of_samples), len(file_names)])
snn_pid_arr = np.zeros([max(number_of_samples), len(file_names)])

# Init Ideal arrays
i_ideal_arr = np.zeros([max(number_of_samples), len(file_names)])
d_ideal_arr = np.zeros([max(number_of_samples), len(file_names)])
pd_ideal_arr = np.zeros([max(number_of_samples), len(file_names)])
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
            
            if old_layout == True: 
                try: t,u,ref,meas,p,i,d,p_d,error = row
                except: t,u,ref,meas,p,i,d,p_d,error, snn_pd,snn_i = row
            elif len(row) == 6:     
                snn_p, snn_i, snn_d, snn_pd, snn_pid, t = row
                meas= 0 ;ref = 0; error=0;p=0;i=0;d=0;p_d=0;u = 0

            elif len(row) == 9:       t,meas,ref,error,p,i,d,p_d,u= row 
            elif len(row) == 14:    t,meas,ref,error,p,i,d,p_d,u, snn_p, snn_i, snn_d, snn_pd, snn_pid= row

            # Fill the standard arrays
            t_arr[row_ind,file_ind] =  t  
            meas_arr[row_ind,file_ind] =  meas
            ref_arr[row_ind,file_ind] =  ref  
            error_arr[row_ind,file_ind] = error
            p_arr[row_ind,file_ind] = p
            i_arr[row_ind,file_ind] =  i
            d_arr[row_ind,file_ind] = d
            pd_arr[row_ind,file_ind] =  p_d  
            u_arr[row_ind,file_ind] =  u  

            if len(row) ==14:
                snn_p_arr[row_ind,file_ind] =  snn_p
                snn_i_arr[row_ind,file_ind] =  snn_i
                snn_d_arr[row_ind,file_ind] =  snn_d
                snn_pd_arr[row_ind,file_ind] =  snn_pd
                snn_pid_arr[row_ind,file_ind] =  snn_pid  

            # Calculate the ideal responses
            error = float(error)
            integral =  integral + (error*(1/Hz))
            i_ideal_arr[row_ind,file_ind] = integral *Ki
            pd_ideal_arr[row_ind,file_ind] = (error-error_prev)*Hz*Kd + error*Kp
            d_ideal_arr[row_ind,file_ind] = (error-error_prev)*Hz*Kd 
            error_prev = error

            row_ind +=1 
    file_ind +=1


plt.plot(t_arr[:-2,0],ref_arr[:-2,0],color = "r", linestyle="--", label="Reference")
for i in range(len(file_names)):
    plt.title("SNN Controller PID (10Hz)")
    plt.plot(t_arr[:-2,i],meas_arr[:-2,i],label = file_names[i])
    # plt.plot(t[:,i],ref[:,i]-meas[:,i],label = str(i))
plt.legend()
plt.grid()
# plt.show()

if plot_p:
    plt.figure()
    for i in range(len(file_names)):
        plt.plot(t_arr[:number_of_samples[i],i],p_arr[:number_of_samples[i],i],label = "pid_" +str(i))
        plt.plot(t_arr[:number_of_samples[i],i],snn_p_arr[:number_of_samples[i],i],label ="snn_" + str(i))
        plt.plot(t_arr[:number_of_samples[i],i],error_arr[:number_of_samples[i],i]*(Kp-1),label ="ideal_" + str(i))
        plt.title("P controller")
    plt.grid()
    plt.legend()

if plot_d:
    plt.figure()
    for i in range(len(file_names)):
        plt.plot(t_arr[:number_of_samples[i],i],d_arr[:number_of_samples[i],i],label = "pid_" + str(i))
        plt.plot(t_arr[:number_of_samples[i],i],snn_d_arr[:number_of_samples[i],i],label ="snn_" + str(i))
        if plot_ideal:
            plt.plot(t_arr[:number_of_samples[i],i],d_ideal_arr[:number_of_samples[i],i],label = "ideal_"+str(i))
        plt.title("D controller")
    plt.grid()
    plt.legend()

if plot_pd:
    plt.figure()
    for i in range(len(file_names)):
        plt.plot(t_arr[:-2,i],(ref_arr[:-2,i]- meas_arr[:-2,i]),label = "error")
        plt.plot(t_arr[:number_of_samples[i],i],snn_pd_arr[:number_of_samples[i],i],label = "snn_" + str(i))
        if plot_ideal:
            plt.plot(t_arr[:number_of_samples[i],i],pd_ideal_arr[:number_of_samples[i],i],label = "ideal_"+str(i))
        plt.title("PD controller")
    plt.grid()
    plt.legend()

if plot_i:
    plt.figure()
    for i in range(len(file_names)):
        plt.plot(t_arr[:number_of_samples[i],i],i_arr[:number_of_samples[i],i],label = "pid_" +str(i))
        plt.plot(t_arr[:number_of_samples[i],i],snn_i_arr[:number_of_samples[i],i],label ="snn_" + str(i))
        if plot_ideal:
            plt.plot(t_arr[:number_of_samples[i],i],i_ideal_arr[:number_of_samples[i],i],label = "ideal_"+str(i))
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