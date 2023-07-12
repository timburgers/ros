#ifndef DRIVER_FILTER_FUNCS_H
#define DRIVER_FILTER_FUNCS_H

#include "radar_ros_driver/driver_filter.h"

namespace radar_ros_driver {
    /**
     * Apply a Linear Moving Average smoothing filter to the target(s) range measurements coming from the radar
     *
     * @param[in]  window_size:      number of measurement points on which the average is calculated
     * @param[in]  current_targets:  vector of measurements from the current target detected NOW (not previous ones)
     * @param[out] targets_store:    matrix of target measurements that stores the values of all targets 
     *                               up to time (t-window_size+1). Concretely:                                   
     *                                                 --                                                   --
     *                                                 | target0_t-ws+1  target1_t-ws+1  ···  targetN_t-ws+1 |
     *                                                 |     ···            ···                  ···         |
     *                                                 | target0_t0      target1_t0      ···  targetN_t0     |
     *                                                 --                                                   --
     * @param[out] range_filter:     vector that stores the linear averaged range for each target
     */
    void RadarRosFilter::LMA(uint32_t& window_size, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
                vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter){

        //----------------------------------------------------------------------------------------------------------------------------------------//
        //------------!!!!!!!!!!!!!! TO DO [marina]: Get rid of THIS IF! [Initialize the vector size at the beginning] !!!!!!!!!!!!---------------//
        //----------------------------------------------------------------------------------------------------------------------------------------//

        // Fill in the 'targets_store' matrix based on the chosen number of points: 'window_size'
        if(targets_store.size() < window_size)
        {
            targets_store.push_back(current_targets);
        }
        else
        {
            targets_store.push_back(current_targets);
            targets_store.erase(targets_store.begin());

        vector<float_t> aux_range;
        // For each column (this is, target)
        for (int j = 0; j < num_targets; j++)
        {
            float_t tmp = 0;
            // For each row (this is, measurement in time)
            for (int i = 0; i < window_size; i++)
            {
                tmp+=targets_store[i][j].range;
            }
            tmp /= (float_t)window_size;
            aux_range.push_back((float_t)tmp);
        }
        range_filter = aux_range;
        }
    }

    /**
     * Apply an Linear Moving Average smoothing filter to the target(s) range measurements AFTER THE MF filter
     * 
     * [NOTE]: For more details on how this function works, see comments in LMA filter
     */
    void RadarRosFilter::LMA_afterFilter(uint32_t& window_size, uint32_t& num_targets, vector<vector<float_t>>& range_store, 
            vector<float_t>& range_filter){

        // Fill in the 'targets_store' matrix based on the chosen number of points: 'window_size'
        if(range_store.size() < window_size)
        {
            range_store.push_back(range_filter);
        }
        else
        {
            range_store.push_back(range_filter);
            range_store.erase(range_store.begin());
        
        /*
        ROS_INFO_STREAM("range_store window size -> "<<  range_store.size());
        ROS_INFO_STREAM("actual window size      -> "<<  window_size);
        ROS_INFO_STREAM("range_store target size -> "<<  range_store[0].size());
        ROS_INFO_STREAM("actual target size      -> "<<  num_targets);;
        */
       
        vector<float_t> aux_range;
        // For each column (this is, target)
        for (int j = 0; j < range_store[0].size(); j++)
        {
            float_t tmp = 0;
            // For each row (this is, measurement in time)
            for (int i = 0; i < window_size; i++)
            {
                tmp+=range_store[i][j];
            }
            tmp /= (float_t)window_size;
            aux_range.push_back((float_t)tmp);
        }
        range_filter = aux_range;
        }
    }

    /**
     * Apply an Exponential Moving Average smoothing filter to the target(s) range measurements coming from the radar
     *
     * @param[in]  window_size:      number of measurement points on which the average is calculated
     * @param[in]  current_targets:  vector of measurements from the current target detected NOW (not previous ones)
     * @param[in]  alpha:            parameter [ 2/(N+1) ] to choose the actual number of points for the EMA
     * @param[out] targets_store:    matrix of target measurements that stores the values of all targets 
     *                                  up to time (t-window_size+1).
     * @param[out] range_filter:     vector that stores the exponential averaged range for each target
     */
    void RadarRosFilter::EMA(uint32_t& window_size, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
                vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter, float_t& alpha){

        // Fill in the 'targets_store' matrix based on the chosen number of points: 'window_size'
        if(targets_store.size() < window_size)
        {
            targets_store.push_back(current_targets);
        }
        else
        {
            targets_store.push_back(current_targets);
            targets_store.erase(targets_store.begin());

        vector<float_t> aux_range;

        // For each column (this is, target)
        for (int j = 0; j < num_targets; j++) 
        {
            float_t s = targets_store[0][j].range;
            
            // For each row (this is, measurement in time)
            for (int i = 0; i < window_size; i++)
            {
                s = alpha*targets_store[i][j].range+(1-alpha)*s;
            }
            aux_range.push_back((float_t)s);
        }
        range_filter = aux_range;
        }
    }

    /**
     * Apply an Exponential Moving Average smoothing filter to the target(s) range measurements AFTER THE MF filter
     * 
     * [NOTE]: For more details on how this function works, see comments in EMA filter
     */
    void RadarRosFilter::EMA_afterFilter(uint32_t& window_size, uint32_t& num_targets, vector<vector<float_t>>& range_store, 
            vector<float_t>& range_filter, float_t& alpha){

        // Fill in the 'targets_store' matrix based on the chosen number of points: 'window_size'
        if(range_store.size() < window_size)
        {
            range_store.push_back(range_filter);
        }
        else
        {
            range_store.push_back(range_filter);
            range_store.erase(range_store.begin());

        vector<float_t> aux_range;

        // For each column (this is, target)
        for (int j = 0; j < range_store[0].size(); j++)
        {
            float_t s = range_store[0][j];

            // For each row (this is, measurement in time)
            for (int i = 0; i < window_size; i++)
            {
                s = alpha*range_store[i][j]+(1-alpha)*s;
            }
            aux_range.push_back((float_t)s);
        }
        range_filter = aux_range;
        }
    }

    /**
     * Apply a Median Filter to the target(s) range measurements coming from the radar to REMOVE OUTLIERS
     *
     * @param[in]  window_size:      number of measurement points on which the filter is applied
     * @param[in]  current_targets:  vector of measurements from the current target detected NOW (not previous ones)
     * @param[out] targets_store:    matrix of target measurements that stores the values of all targets 
     *                                 up to time (t-window_size+1).
     * @param[out] range_filter:     vector that stores the median filtered range for each target
     */
    void RadarRosFilter::MF(uint32_t& window_size, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
                vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter){

        // Fill in the 'targets_store' matrix based on the chosen number of points: 'window_size'

        //ROS_INFO_STREAM("targets_store.size -> " << targets_store.size());

        if(targets_store.size() < window_size)
        {
            targets_store.push_back(current_targets);
        }
        else
        {
            targets_store.push_back(current_targets);
            targets_store.erase(targets_store.begin());
        /*
        ROS_INFO_STREAM("targets_store.size() -> " << targets_store.size());
        ROS_INFO_STREAM("actual window size      -> "<<  window_size);
        ROS_INFO_STREAM("targets_store[0].size() -> " << targets_store[0].size());
        ROS_INFO_STREAM("actual target size      -> "<<  num_targets);
        */
        vector<float_t> aux_range;
        float_t tmp[window_size];
        int tmp_size = sizeof(tmp)/sizeof(float_t);

        //ROS_INFO_STREAM("tmp_size -> " << tmp_size);

        // For each column (this is, target)
        for (int j = 0; j < targets_store[0].size(); j++)
        {
            // For each row (this is, measurement in time)
            for (int i = 0; i < window_size; i++)
            {   
                tmp[i] = (float_t)targets_store[i][j].range;
            }
            sort(tmp,tmp+tmp_size);
            //ROS_INFO_STREAM("range_filter[" << j << "] -> " << (float_t)tmp[(int)(window_size/2+1)]);
            aux_range.push_back((float_t)tmp[(int)(window_size/2+1)]);
        }
        range_filter = aux_range;
        }
    }

    void RadarRosFilter::MF_LMA(uint32_t& window_big, uint32_t& window_small, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
            vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter){
        
        if(targets_store.size() < window_big)
        {
            targets_store.push_back(current_targets);
        }
        else
        {
            targets_store.push_back(current_targets);
            targets_store.erase(targets_store.begin());

        vector<float_t> aux_range;
        float_t tmp[window_big];
        int tmp_size = sizeof(tmp)/sizeof(float_t);

        //ROS_INFO_STREAM("tmp_size -> " << tmp_size);

        // For each column (this is, target)
        for (int j = 0; j < targets_store[0].size(); j++)
        {
            // For each row (this is, measurement in time)
            for (int i = 0; i < window_big; i++)
            {   
                tmp[i] = (float_t)targets_store[i][j].range;
            }
            sort(tmp,tmp+tmp_size);

            // Get subindex list
            vector<int> index_list = RadarRosFilter::get_indexlist(window_big, window_small); // TO DO [marina]: Calculate only once!! Put outside of the function!!

            float_t aux = 0;
            for (int m = 0; m < window_small; m++)
            {
                aux += tmp[index_list[m]];
            }
            aux /= window_small;
            //ROS_INFO_STREAM("range_filter[" << j << "] -> " << (float_t)tmp[(int)(window_size/2+1)]);
            aux_range.push_back((float_t)aux);
        }
        range_filter = aux_range;
        }
    }

    void RadarRosFilter::MF_EMA(uint32_t& window_big, uint32_t& window_small, uint32_t& num_targets, vector<Measurement_elem_t>& current_targets,
            vector<vector<Measurement_elem_t>>& targets_store, vector<float_t>& range_filter, float_t& alpha){
        
        if(targets_store.size() < window_big)
        {
            targets_store.push_back(current_targets);
        }
        else
        {
            targets_store.push_back(current_targets);
            targets_store.erase(targets_store.begin());

        vector<float_t> aux_range;
       
        //ROS_INFO_STREAM("tmp_size -> " << tmp_size);

        // For each column (this is, target)
        for (int j = 0; j < targets_store[0].size(); j++)
        {
            float_t tmp[window_big];
            int tmp_size = sizeof(tmp)/sizeof(float_t);

            // For each row (this is, measurement in time)
            for (int i = 0; i < window_big; i++)
            {   
                tmp[i] = (float_t)targets_store[i][j].range;
            }
            sort(tmp,tmp+tmp_size);

            // Get subindex list
            vector<int> index_list = RadarRosFilter::get_indexlist(window_big, window_small); // TO DO [marina]: Calculate only once!! Put outside of the function!!

            float_t s = tmp[index_list[0]];

            for (int m = 0; m < window_small; m++)
            {
                s = alpha*tmp[index_list[m]]+(1-alpha)*s;
            }
            //ROS_INFO_STREAM("range_filter[" << j << "] -> " << (float_t)tmp[(int)(window_size/2+1)]);
            aux_range.push_back((float_t)s);
        }
        range_filter = aux_range;
        }
    }

    /*
    vector<float_t> aux_range;
    // For each column (this is, target)
        for (int j = 0; j < num_targets; j++) 
        {
            float_t s = targets_store[0][j].range;
            
            // For each row (this is, measurement in time)
            for (int i = 0; i < window_size; i++)
            {
                s = alpha*targets_store[i][j].range+(1-alpha)*s;
            }
            aux_range.push_back((float_t)s);
        }
        range_filter = aux_range;
    */

    vector<int> RadarRosFilter::get_indexlist(uint32_t& window_big, uint32_t& window_small){

        vector<int> index_list;

        int start_index = (int)((window_big-window_small)/2);
        for (int i = 0; i < window_small; i++)
        {   
            //ROS_INFO_STREAM("subindexes -> " << start_index + i);
            index_list.push_back(start_index + i);
        }
        
        return index_list;
    }

} // namespace

#endif