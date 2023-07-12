#ifndef DRIVER_FILTER_H
#define DRIVER_FILTER_H

#include "radar_ros_driver/ofxRadar24Ghz.h"

#include <ros/ros.h>
#include <string>
#include <sstream>

#include "radar_targets_msgs/MyEvent.h"
#include "radar_targets_msgs/MyEventArray.h"

// Sampling frequency
#define FREQ 100.0

namespace radar_ros_driver {

    class RadarRosFilter {
        public:

            // Default Constructor + Destructor
            RadarRosFilter(ros::NodeHandle & nh, ros::NodeHandle nh_private);
            virtual ~RadarRosFilter();

            // Publisher Loop Function
            void readout(uint16_t count);

        private: 

            // ROS Attributes
            ros::NodeHandle nh_;
            ros::Subscriber radar_sub_targets_;
            ros::Publisher  radar_pub_filter_;
            volatile bool   running_;
            std::string     ns;

            // Callback Function
            void radarCallbackFilter(const radar_targets_msgs::MyEventArray::ConstPtr& msg);

            // Measurement + Filtering Attributes
            uint32_t MA_window_size;
            uint32_t MF_window_size;
            float_t alpha;
            //vector<Measurement_elem_t> current_targets;
            uint32_t num_targets;
            vector<Measurement_elem_t> current_targets_store;
            vector<vector<Measurement_elem_t>> filter_targets_store;
            vector<vector<float_t>> range_store;
            vector<float_t> range_filter;

            // Filtering Functions
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
            void MF_LMA(uint32_t& window_big, uint32_t& window_small, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
                vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter);
            void MF_EMA(uint32_t& window_big, uint32_t& window_small, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
                vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter, float_t& alpha);

            // Additional Functions
            vector<int> get_indexlist(uint32_t& window_big, uint32_t& window_small);

    };
} // namespace

#endif