#include "radar_ros_driver/driver_filter.h"
#include "radar_ros_driver/driver_filter_funcs.h"

namespace radar_ros_driver {

    // Default Constructor
	RadarRosFilter::RadarRosFilter(ros::NodeHandle & nh, ros::NodeHandle nh_private) : nh_(nh){

		radar_sub_targets_ = nh_.subscribe("radar_targets", 10, &RadarRosFilter::radarCallbackFilter, this);

        ns = ros::this_node::getNamespace();
        if (ns == "/"){
            ns = "/radar";
        }

        radar_pub_filter_ = nh_.advertise<radar_targets_msgs::MyEventArray>("radar_filter", 10); // Create new message type for this

        running_ = true;
        MA_window_size = (int)LMA_SIZE; //(int)EMA_SIZE
        MF_window_size = (int)MF_SIZE; 
        alpha = (float_t)EMA_ALPHA; 

        ROS_INFO("Starting [DRIVER] listener -> [DRIVER_FILTER]...");
	}

    // Default Destructor
	RadarRosFilter::~RadarRosFilter(){
		if (running_){
            //TODO: close RadarRosFilter
            ROS_INFO("Starting [DRIVER] listener -> [DRIVER_FILTER]...");
            running_ = false;
        }
	}

	void RadarRosFilter::radarCallbackFilter(const radar_targets_msgs::MyEventArray::ConstPtr& msg){

		num_targets = (int)(msg->total_num_targets);

        current_targets_store.clear();

		for(int i=0; i<num_targets; i++){

			radar_targets_msgs::MyEvent aux_msg   = msg->target_events[i];
            Measurement_elem_t aux_single_measure;

            aux_single_measure.range    = (float_t)aux_msg.range;
            aux_single_measure.speed    = (float_t)aux_msg.speed;
            aux_single_measure.angle    = (float_t)aux_msg.angle;
            aux_single_measure.strength = (float_t)aux_msg.strength;

            current_targets_store.push_back(aux_single_measure);
		}
	}

/**
     * Function that is executed continuously in the while::ros() loop with frequency FREQ (see driver.h)
     */
    void RadarRosFilter::readout(uint16_t count){

        if(running_){
            
            // Targets radar message
            radar_targets_msgs::MyEvent single_target_msg;
            radar_targets_msgs::MyEventArray current_targets_msg;

            //current_targets = myradar.current_targets;
            //num_targets = current_targets.size(); 
            single_target_msg.ts = ros::Time::now();

            current_targets_msg.total_num_targets = num_targets;

            for(int i=0; i<num_targets; i++){
                single_target_msg.num_target       = i;
                single_target_msg.angle            = (float_t)(current_targets_store[i].angle);
                single_target_msg.speed            = (float_t)(current_targets_store[i].speed);
                single_target_msg.range            = (float_t)(current_targets_store[i].range);
                single_target_msg.strength         = (float_t)(current_targets_store[i].strength);

                current_targets_msg.target_events.push_back(single_target_msg);
            }

            // Moving average computation
            current_targets_msg.MF_window_size = MF_window_size;
            current_targets_msg.MA_window_size = MA_window_size;

            //MF(this->MF_window_size, this->num_targets, this->current_targets_store, this->filter_targets_store, this->range_filter);
            //EMA_afterFilter(this->MA_window_size, this->num_targets, this->range_store, this->range_filter, this->alpha);

            MF_LMA(this->MF_window_size, this->MA_window_size, this->num_targets, this->current_targets_store, this->filter_targets_store, this->range_filter);
            //MF_EMA(this->MF_window_size, this->MA_window_size, this->num_targets, this->current_targets_store, this->filter_targets_store, this->range_filter, this->alpha);

            //LMA_afterFilter(this->MA_window_size, this->num_targets, this->range_store, this->range_filter);
            //EMA(this->MA_window_size, this->current_targets, this->targets_store,
            //    this->range_filter, this->alpha);

            //LMA(this->MA_window_size, this->num_targets, this->current_targets, this->targets_store, this->range_filter);

            current_targets_msg.range_filter = this->range_filter;
            current_targets_msg.range_filter_size = this->range_filter.size();

            radar_pub_filter_.publish(current_targets_msg);	
            
        }
    }
} // namespace
