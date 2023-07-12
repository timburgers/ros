#include <ros/ros.h>
#include "radar_ros_driver/driver_filter.h"

int main(int argc, char* argv[]){

    // Init
    ros::init(argc, argv, "radar_ros_filter");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    // Set frequency
    ros::Rate loop_rate(FREQ);

    // Set driver
    radar_ros_driver::RadarRosFilter* filter = new radar_ros_driver::RadarRosFilter(nh, nh_private);
    
    // Publish data
    int count = 0;
    while(ros::ok()){

        filter->readout(count);

        ros::spinOnce();
        loop_rate.sleep();
        ++count;

    }

    return 0;
}
