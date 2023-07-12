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

#ifndef DRIVER_H
#define DRIVER_H

#include "radar_ros_driver/ofxRadar24Ghz.h"

#include <ros/ros.h>
#include <string>
#include <sstream>

#include "radar_msgs/Event.h"
#include "radar_msgs/MyPlot.h"
#include "radar_targets_msgs/MyEvent.h"
#include "radar_targets_msgs/MyEventArray.h"

// Sampling frequency
#define FREQ 30.0

namespace radar_ros_driver {

    class RadarRosDriver {
        public:

            ofxRadar24Ghz myradar;

            RadarRosDriver(ros::NodeHandle & nh, ros::NodeHandle nh_private);
            virtual ~RadarRosDriver();
            void readout(uint16_t count);

            /*
            void LMA(uint32_t& window_size, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
                vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter);
            void LMA_afterFilter(uint32_t& window_size, uint32_t& num_targets, vector<vector<float_t>>& range_store, 
                vector<float_t>& range_filter);
            void EMA(uint32_t& window_size, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
                vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter, float_t& alpha);
            void EMA_afterFilter(uint32_t& window_size, uint32_t& num_targets, vector<vector<float_t>>& range_store, 
                vector<float_t>& range_filter, float_t& alpha);
            void MF (uint32_t& window_size, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
                vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter);
            */

        private: 

            ros::NodeHandle     nh_;
            ros::Publisher      radar_pub_;
            ros::Publisher      radar_pub_plot_;
            ros::Publisher      radar_pub_targets_;
            volatile bool       running_;
            std::string         ns;

            // Data size
            uint32_t num_samples_per_chirp;
            uint32_t num_chirps;
            uint32_t range_fft_size;
            // uint32_t MA_window_size;
            // uint32_t MF_window_size;
            float_t dist_per_bin;
            // float_t alpha;

            // Raw data 
            float adc_real_tx1rx1_f[NUM_CHIRPS_PER_FRAME*NUM_SAMPLES_PER_CHIRP];
            float adc_imag_tx1rx1_f[NUM_CHIRPS_PER_FRAME*NUM_SAMPLES_PER_CHIRP];
            float adc_real_tx1rx2_f[NUM_CHIRPS_PER_FRAME*NUM_SAMPLES_PER_CHIRP];
            float adc_imag_tx1rx2_f[NUM_CHIRPS_PER_FRAME*NUM_SAMPLES_PER_CHIRP];

            // Targets data
            vector<Measurement_elem_t> current_targets;
            uint32_t num_targets;

            // Averaged targets data
            // vector<vector<Measurement_elem_t>> targets_store;
            // vector<vector<float_t>> range_store;
            // vector<float_t> range_filter;

            // Additional plot data
            double range_tx1rx1_max_plot[RANGE_FFT_SIZE];
            double range_tx1rx2_max_plot[RANGE_FFT_SIZE];     
    };
} // namespace

#endif