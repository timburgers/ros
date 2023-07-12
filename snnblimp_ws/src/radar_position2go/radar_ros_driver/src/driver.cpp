/**
 * This file is part of the odroid_ros_dvs package - MAVLab TU Delft
 * 
 *   MIT License
 *
 *   Copyright (c) 2020 MAVLab TU Delft
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 * 
 * */

#include "radar_ros_driver/driver.h"

namespace radar_ros_driver {

    // Default Constructor
    RadarRosDriver::RadarRosDriver(ros::NodeHandle & nh, ros::NodeHandle nh_private) : nh_(nh){

        ns = ros::this_node::getNamespace();
        if (ns == "/"){
            ns = "/radar";
        }
        
        radar_pub_ = nh_.advertise<radar_msgs::Event>("radar", 10);
        radar_pub_plot_ = nh_.advertise<radar_msgs::MyPlot>("radar_plot", 10);
		radar_pub_targets_ = nh_.advertise<radar_targets_msgs::MyEventArray>("radar_targets", 10);

        myradar.setup();
        running_ = true;

        num_samples_per_chirp = (int)NUM_SAMPLES_PER_CHIRP;
        num_chirps = (int)NUM_OF_CHIRPS;
        range_fft_size = (int)RANGE_FFT_SIZE;
        dist_per_bin = (float_t)(myradar.dist_per_bin);

        ROS_INFO("Starting RADAR listener -> [DRIVER]...");
    }

    // Default Destructor
    RadarRosDriver::~RadarRosDriver(){

        if (running_){
            //TODO: close driver
            ROS_INFO("Shutting down RADAR listener ->[DRIVER]...");
            running_ = false;
        }
    }

    /**
     * Function that is executed continuously in the while::ros() loop with frequency FREQ (see driver.h)
     */
    void RadarRosDriver::readout(uint16_t count){

        if(running_){

            myradar.update();       

            // Raw radar message (I,Q, both antennas)
            radar_msgs::Event event_msg;

            event_msg.dimx = num_chirps;
            event_msg.dimy = num_samples_per_chirp;
            event_msg.ts = ros::Time::now(); 

            for(int i=0; i<num_samples_per_chirp*num_chirps; i++){
                adc_real_tx1rx1_f[i] = (float_t)(myradar.adc_real_tx1rx1[i]);
                adc_imag_tx1rx1_f[i] = (float_t)(myradar.adc_imag_tx1rx1[i]);
                adc_real_tx1rx2_f[i] = (float_t)(myradar.adc_real_tx1rx2[i]);
                adc_imag_tx1rx2_f[i] = (float_t)(myradar.adc_imag_tx1rx2[i]);

                event_msg.data_rx1_re.push_back(adc_real_tx1rx1_f[i]); 
                event_msg.data_rx1_im.push_back(adc_imag_tx1rx1_f[i]);
                event_msg.data_rx2_re.push_back(adc_real_tx1rx2_f[i]);
                event_msg.data_rx2_im.push_back(adc_imag_tx1rx2_f[i]);
            }

            radar_pub_.publish(event_msg);  

            // Targets radar message
            radar_targets_msgs::MyEvent single_target_msg;
            radar_targets_msgs::MyEventArray current_targets_msg;

            current_targets = myradar.current_targets;
            num_targets = current_targets.size(); 
            single_target_msg.ts = ros::Time::now();

            current_targets_msg.total_num_targets = num_targets;

            for(int i=0; i<num_targets; i++){
                single_target_msg.num_target       = i;
                single_target_msg.angle            = (float_t)(current_targets[i].angle);
                single_target_msg.speed            = (float_t)(current_targets[i].speed);
                single_target_msg.range            = (float_t)(current_targets[i].range);
                single_target_msg.strength         = (float_t)(current_targets[i].strength);
				single_target_msg.rx1_angle_arg_re = (float_t)(current_targets[i].rx1_angle_arg_re);
				single_target_msg.rx1_angle_arg_im = (float_t)(current_targets[i].rx1_angle_arg_im);
				single_target_msg.rx2_angle_arg_re = (float_t)(current_targets[i].rx2_angle_arg_re);
				single_target_msg.rx2_angle_arg_im = (float_t)(current_targets[i].rx2_angle_arg_im);

                current_targets_msg.target_events.push_back(single_target_msg);
            }

            radar_pub_targets_.publish(current_targets_msg);	
            
            // Test messages for plotting
            radar_msgs::MyPlot plot_msg;
            plot_msg.range_fft_size = range_fft_size;
            plot_msg.dist_per_bin = dist_per_bin;
            for(int i=0; i<range_fft_size; i++){
                range_tx1rx1_max_plot[i] = (float_t)(myradar.range_tx1rx1_max_plot[i]);
                range_tx1rx2_max_plot[i] = (float_t)(myradar.range_tx1rx2_max_plot[i]);
                plot_msg.max_range_rx1_vector.push_back(range_tx1rx1_max_plot[i]);
                plot_msg.max_range_rx2_vector.push_back(range_tx1rx2_max_plot[i]);
            }
            radar_pub_plot_.publish(plot_msg);
        }
    }
} // namespace