#include "radar_ros_driver/ofxRadar24Ghz.h"
//#include <typeinfo>

//======================================================
void ofxRadar24Ghz::setup() {

	// via definitions
	num_chirps = NUM_CHIRPS_PER_FRAME;
	num_samples_per_chirp = NUM_SAMPLES_PER_CHIRP;
	esignalpart = E_SIGNAL_PART;
	rx_mask = RX_MASK;
	num_antennas = countSetBits(rx_mask);
	speed_of_light = SPEED_OF_LIGHT;// SPEED OF LIGHT

	// allocate memory for callbacks
	full_data_block = (float *)malloc(num_antennas * num_chirps * num_samples_per_chirp * 2 * sizeof(float) );	// I and Q
	temperature = (float *)malloc(1 * sizeof(float));
	frame_format_current = (Frame_Format_t *)malloc(1 * sizeof(Frame_Format_t));
	device_info = (Device_Info_t *)malloc(1 * sizeof(Device_Info_t));
	chirp_duration_ns = (uint32_t*)malloc(1*sizeof(uint32_t));
	min_frame_interval_us = (uint32_t*)malloc(1*sizeof(uint32_t));
	tx_power_001dBm = (uint32_t*)malloc(1*sizeof(uint32_t));
	distance_m = (double *)malloc(num_samples_per_chirp * sizeof(double));
	fmcw_cfg = (Fmcw_Configuration_t*)malloc(1 * sizeof(Fmcw_Configuration_t));
	tgt_range1 = (target_peak*)malloc(MAX_NUM_TARGETS * sizeof(target_peak));
	tgt_range2 = (target_peak*)malloc(MAX_NUM_TARGETS * sizeof(target_peak));
	// init index to be -1
	for(size_t tp=0;tp<MAX_NUM_TARGETS; tp++){
		tgt_range1[tp].index = -1;
		tgt_range1[tp].peak_val = 0.0;
		tgt_range2[tp].index = -1;
		tgt_range2[tp].peak_val = 0.0;
	}
	range_fft_spectrum_hist1 = (complex<double>*)malloc(RANGE_FFT_SIZE * sizeof(complex<double>) );
	range_fft_spectrum_hist2 = (complex<double>*)malloc(RANGE_FFT_SIZE * sizeof(complex<double>) );
	fft_1 = (complex<double>*)malloc(RANGE_FFT_SIZE * sizeof(complex<double>) );
	fft_2 =  (complex<double>*)malloc(RANGE_FFT_SIZE * sizeof(complex<double>) );
	//fft_doppler1 = (complex<double>*)malloc(DOPPLER_FFT_SIZE * sizeof(complex<double>) );

	// init to zero
	for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER < RANGE_FFT_SIZE; SAMPLE_NUMBER++){
		range_fft_spectrum_hist1[SAMPLE_NUMBER] = (0.0);
		range_fft_spectrum_hist2[SAMPLE_NUMBER] = (0.0);
		fft_1[SAMPLE_NUMBER] = (0.0);
		fft_2[SAMPLE_NUMBER] = (0.0);
	}
	// data
	adc_real_tx1rx1 = (double *)malloc(NUM_CHIRPS_PER_FRAME * NUM_SAMPLES_PER_CHIRP* sizeof(double));
	adc_imag_tx1rx1 = (double *)malloc(NUM_CHIRPS_PER_FRAME * NUM_SAMPLES_PER_CHIRP* sizeof(double));
	adc_real_tx1rx2 = (double *)malloc(NUM_CHIRPS_PER_FRAME * NUM_SAMPLES_PER_CHIRP* sizeof(double));
	adc_imag_tx1rx2 = (double *)malloc(NUM_CHIRPS_PER_FRAME * NUM_SAMPLES_PER_CHIRP* sizeof(double));

	// generals
	radar_handle = 0;
	num_of_ports = 0;
	res = -1;
	protocolHandle = 0;
	endpointRadarBase = 0;

	// START CONNECTION TO RADAR VIA USB
	startRadarUSB();

	Device_Info_t *this_device_infos = (Device_Info_t *) (device_info);
 	fC = ((double)this_device_infos->max_rf_frequency_kHz + (double)this_device_infos->min_rf_frequency_kHz)/2.0 * 1e3;

	fs = 426666; // Adcxmc configuration
	PRT = 0.0005;//chirp_duration_ns + DOWN_CHIRP_DURATION + CHIRP_TO_CHIRP_DELAY;
	BW = 200000000; // in HZ
	range_fft_size = RANGE_FFT_SIZE;
	doppler_fft_size = DOPPLER_FFT_SIZE;
	range_threshold = RANGE_THRESHOLD;
	doppler_threshold = DOPPLER_THRESHOLD;
	min_distance = MIN_DISTANCE; // m
	max_distance = MAX_DISTANCE;
	max_num_targets = MAX_NUM_TARGETS;
	lambda = SPEED_OF_LIGHT/fC;

	hz_to_mps_constant=lambda/2.0;
	if_scale= 16 * 3.3*range_fft_size/num_samples_per_chirp;
	// REANGE WINDOWING
	range_window_func = (double*)malloc(NUM_SAMPLES_PER_CHIRP*sizeof(double));
	for(double ij=0;ij<num_samples_per_chirp;ij++){
		double val = 2*blackman(ij, num_samples_per_chirp);
		range_window_func[(int)ij] = (val);
		//printf("val %f", val);
	}

	// DOPPLER
	dopper_window_func = (double*)malloc(DOPPLER_FFT_SIZE*sizeof(double));
	cheby_win(dopper_window_func, DOPPLER_FFT_SIZE, 120);

	r_max = NUM_SAMPLES_PER_CHIRP*SPEED_OF_LIGHT/ (2*BW); // max range!
	dist_per_bin = (double)r_max / (double)range_fft_size;
	double val_j;
	for(int jj=0.0; jj< range_fft_size; jj++){
		val_j = jj*dist_per_bin;
		array_bin_range.push_back(val_j);
	}
	//printf("\n%f\n",r_max);
	fD_max = (double)((1.0) / (2.0*(double)PRT));
	fD_per_bin = (double)fD_max /(double)(doppler_fft_size/2);
	for(int jj=0;jj< doppler_fft_size ;jj++){
		double val = (double)((double)jj - ((double)doppler_fft_size /2.0) - 1)*-1.0*(double)fD_per_bin*(double)hz_to_mps_constant;
		array_bin_fD.push_back(val);
	}

	// doppler FFT MATRIX, just one frame
	for(int h = 0; h < RANGE_FFT_SIZE; h++ ){
		vector<complex<double>> this_ll;
		for(unsigned int i=0; i< DOPPLER_FFT_SIZE; i++){
			complex<double> cv;
			cv =(0.0);
			//cv.imag(0.0);
			this_ll.push_back(cv);
		}
		range_doppler_tx1rx1.push_back(this_ll);
		range_doppler_tx1rx2.push_back(this_ll);
		rangeFFT1.push_back(this_ll);
		rangeFFT2.push_back(this_ll);
	}

	// TRACKING
	enable_mti_filtering = false;

	// loading from file
	isloaddata = false;
	file_loaded = false;
	repeat_mode = false;
	acq_started = false;

	if(!acq_started){
		// start acquisition
		// enable/disable automatic trigger
		if (AUTOMATIC_DATA_FRAME_TRIGGER){
			res = ep_radar_base_set_automatic_frame_trigger(protocolHandle,
															endpointRadarBase,
															AUTOMATIC_DATA_TRIGER_TIME_US);
		}else{
			res = ep_radar_base_set_automatic_frame_trigger(protocolHandle,
															endpointRadarBase,
															0);
		}
		if(res != -1){
			acq_started = true;
		}else{
			printf("CANNOT START ACQUISITION\n");
			islive = false;
		}
	}

}

//======================================================
double ofxRadar24Ghz::calculateBeatFreq(double distance_in_m, double bandwidth_hz, double speed_of_light, double ramp_time_s){
	return(2.0*distance_in_m*bandwidth_hz)/(speed_of_light*ramp_time_s);
}

//======================================================
void ofxRadar24Ghz::changeMTI(){
	enable_mti_filtering = !enable_mti_filtering;
}

//======================================================
void ofxRadar24Ghz::startRadarUSB(){

	// open COM port
	protocolHandle = radar_auto_connect();

	// get endpoint ids
	if (protocolHandle >= 0)
	{
		for (int i = 1; i <= protocol_get_num_endpoints(protocolHandle); ++i) {
			// current endpoint is radar base endpoint
			if (ep_radar_base_is_compatible_endpoint(protocolHandle, i) == 0) {
				endpointRadarBase = i;
				continue;
			}
		}
	}

	if (endpointRadarBase >= 0)
	{
		// compatible in all means
		uint32_t is_compatible = ep_radar_base_is_compatible_endpoint(protocolHandle,endpointRadarBase);

		print_status_code( protocolHandle, is_compatible);

		// callback get device info
		ep_radar_base_set_callback_device_info(this->get_device_info, device_info);
		// callback for frame format messages.
		ep_radar_base_set_callback_frame_format(this->received_frame_format, frame_format_current);
		// callback min frame interval
		ep_radar_base_set_callback_min_frame_interval(this->get_min_frame_interval, min_frame_interval_us);
		// callback chirp_duration
		ep_radar_base_set_callback_chirp_duration(this->get_chirp_duration, chirp_duration_ns);
		// register call back functions for adc data
		ep_radar_base_set_callback_data_frame(this->received_frame_data, full_data_block);//(void*) this);
		// register call back for tx power read
		ep_radar_base_set_callback_tx_power(this->get_tx_power, tx_power_001dBm);

		// get device info
		uint32_t dev_info_status = ep_radar_base_get_device_info(protocolHandle,endpointRadarBase);
		print_status_code( protocolHandle, dev_info_status);
		// get power
		int32_t answer = ep_radar_base_get_tx_power(protocolHandle, endpointRadarBase, 0);
		print_status_code(protocolHandle, answer);
		// get current frame format
		Frame_Format_t* frame_format_now;
		frame_format_now = (Frame_Format_t *)malloc(1 * sizeof(Frame_Format_t));
		this->get_frame_format(protocolHandle, endpointRadarBase, frame_format_now);

		/* If the frame format contains a 0, this makes no sense. */
		if ((frame_format_now->rx_mask == 0) ||
		  (frame_format_now->num_chirps_per_frame  == 0) ||
			(frame_format_now->num_samples_per_chirp == 0) ||
			  (frame_format_now->num_chirps_per_frame  > (uint32_t)num_chirps) ||
				(frame_format_now->num_samples_per_chirp > (uint32_t)num_samples_per_chirp))
		{
			printf("frame format error\n");
		}

		// set current frame format to 64 64
		frame_format_now->num_chirps_per_frame = num_chirps;
		frame_format_now->num_samples_per_chirp = num_samples_per_chirp;
		frame_format_now->eSignalPart = (Signal_Part_t)esignalpart;
		frame_format_now->rx_mask = rx_mask;
		int32_t jj =  ep_radar_base_set_frame_format(protocolHandle,endpointRadarBase,frame_format_now);
		this->get_frame_format(protocolHandle, endpointRadarBase, frame_format_now);
		print_status_code(protocolHandle, jj);

		//get chirp duration
		int32_t chirp_duration_status = ep_radar_base_get_chirp_duration(protocolHandle, endpointRadarBase);
		print_status_code( protocolHandle, chirp_duration_status);

		// get min frame interval
		uint32_t min_frame = ep_radar_base_get_min_frame_interval(protocolHandle, endpointRadarBase);
		print_status_code( protocolHandle, min_frame);

		// distance calculations
		Device_Info_t *this_device_info = (Device_Info_t *) (device_info);
		double bandwidth_hz = (double)(this_device_info->max_rf_frequency_kHz-this_device_info->min_rf_frequency_kHz)*1000.0;
		double ramp_time_s = (double)(*chirp_duration_ns)*1e-9;
		printf("bandwidth_Hz %f\n", bandwidth_hz);
		printf("ramp_time_s %f\n", ramp_time_s);
		printf("speed_of_light %f\n", speed_of_light);

		islive = true;
	}else{
		printf("RADAR INITIALIZATION FAILED\n");
		islive = false; // no radar
	}

}

//======================================================
int ofxRadar24Ghz::radar_auto_connect() {
	// usb connections
	num_of_ports = com_get_port_list(comp_port_list, (size_t)256);
	if (num_of_ports == 0)
	{
		return -1;
	}
	else
	{
		comport = strtok(comp_port_list, delim);
		while (num_of_ports > 0)
		{
			num_of_ports--;
			// open COM port
			radar_handle = protocol_connect(comport);
			if (radar_handle >= 0)
			{
				break;
			}
			comport = strtok(NULL, delim);
		}
		return radar_handle;
	}
}

//======================================================
void ofxRadar24Ghz::update() {

	// get raw data
	res = ep_radar_base_get_frame_data(protocolHandle,	endpointRadarBase,	1);
	//if(res != -1){
		// IF LIVE DATA
	for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t)num_chirps; CHIRP_NUM++){
		for (uint32_t ANTENNA_NUMBER = 0;ANTENNA_NUMBER < (uint32_t)num_antennas ; ANTENNA_NUMBER++){
			for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER < (uint32_t)num_samples_per_chirp; SAMPLE_NUMBER++){
				double this_adc_real = full_data_block[CHIRP_NUM*4*num_samples_per_chirp + (2*ANTENNA_NUMBER)*num_samples_per_chirp + SAMPLE_NUMBER]*if_scale;
				double this_adc_img  = full_data_block[CHIRP_NUM*4*num_samples_per_chirp + (2*ANTENNA_NUMBER+1)*num_samples_per_chirp + SAMPLE_NUMBER]*if_scale;

				if(ANTENNA_NUMBER == 0){
					adc_real_tx1rx1[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER] = (this_adc_real); // data out and scaled
					adc_imag_tx1rx1[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER] = (this_adc_img);   // data out and scaled

				}else if (ANTENNA_NUMBER == 1){
					adc_real_tx1rx2[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER] = (this_adc_real); // data out and scaled
					adc_imag_tx1rx2[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER] = (this_adc_img);   // data out and scaled
				}

			}
		}
	} // chirp
	//}else{
	//	printf("THERE IS A PROBLEM WITH DATA ACQUISITION\n");
	//	islive = false; // something has happed to the radar connection
	//}

	// MEAN REMOVAL ACROSS RANGE for RX1 and RX2
	double mean_real_tx1rx1[num_chirps];
	double mean_imag_tx1rx1[num_chirps];
	double mean_real_tx1rx2[num_chirps];
	double mean_imag_tx1rx2[num_chirps];
	// init to zero
	for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t)num_chirps; CHIRP_NUM++){
		mean_real_tx1rx1[CHIRP_NUM] = 0.0;
		mean_imag_tx1rx1[CHIRP_NUM] = 0.0;
		mean_real_tx1rx2[CHIRP_NUM] = 0.0;
		mean_imag_tx1rx2[CHIRP_NUM] = 0.0;
	}
	for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t)num_chirps; CHIRP_NUM++){
		for (uint32_t ANTENNA_NUMBER = 0;ANTENNA_NUMBER < (uint32_t)num_antennas ; ANTENNA_NUMBER++){
			for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER <(uint32_t) num_samples_per_chirp; SAMPLE_NUMBER++){
				if(ANTENNA_NUMBER == 0){
					mean_real_tx1rx1[CHIRP_NUM] += adc_real_tx1rx1[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER]; // data out and scaled
					mean_imag_tx1rx1[CHIRP_NUM] += adc_imag_tx1rx1[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER];   // data out and scaled
				}else if (ANTENNA_NUMBER == 1){
					mean_real_tx1rx2[CHIRP_NUM] += adc_real_tx1rx2[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER]; // data out and scaled
					mean_imag_tx1rx2[CHIRP_NUM] += adc_imag_tx1rx2[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER];   // data out and scaled
				}
			}
		}
	}

	// put frame in complex matrix
	for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t)num_chirps; CHIRP_NUM++){
		mean_real_tx1rx1[CHIRP_NUM] /= (double)(num_samples_per_chirp);
		mean_imag_tx1rx1[CHIRP_NUM] /= (double)(num_samples_per_chirp);
		mean_real_tx1rx2[CHIRP_NUM] /= (double)(num_samples_per_chirp);
		mean_imag_tx1rx2[CHIRP_NUM] /= (double)(num_samples_per_chirp);

		for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER < (uint32_t)num_samples_per_chirp; SAMPLE_NUMBER++){
			adc_real_tx1rx1[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER] -= mean_real_tx1rx1[CHIRP_NUM];
			adc_imag_tx1rx1[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER] -= mean_imag_tx1rx1[CHIRP_NUM];
			adc_real_tx1rx2[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER] -= mean_real_tx1rx2[CHIRP_NUM];
			adc_imag_tx1rx2[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER] -= mean_imag_tx1rx2[CHIRP_NUM];
		}
	}

	//range_fft_spectrum_hist for MTI filter enable
	double  alpha_mti = 1.0f / MTI_FILTER_LEN;
	double  beta_mti = (1.0f - alpha_mti);
	for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t)num_chirps; CHIRP_NUM++){
		// FFT IS PADDED WITH ZERO OR TRUNCATED
		// RANGE_FFT_SIZE
		for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER < RANGE_FFT_SIZE; SAMPLE_NUMBER++){
			// zero padding
			if(SAMPLE_NUMBER >= (uint32_t)num_samples_per_chirp){
				fft_1[SAMPLE_NUMBER] = (0.0);
				fft_2[SAMPLE_NUMBER] = (0.0);
			}else{
				double adc_r = adc_real_tx1rx1[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER]*range_window_func[SAMPLE_NUMBER];
				double adc_i = adc_imag_tx1rx1[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER]*range_window_func[SAMPLE_NUMBER];
				fft_1[SAMPLE_NUMBER].real(adc_r);
				fft_1[SAMPLE_NUMBER].imag(adc_i);//matrix_tx1rx1[SAMPLE_NUMBER][CHIRP_NUM];
				adc_r = adc_real_tx1rx2[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER]*range_window_func[SAMPLE_NUMBER];
				adc_i = adc_imag_tx1rx2[CHIRP_NUM*num_samples_per_chirp + SAMPLE_NUMBER]*range_window_func[SAMPLE_NUMBER];
				fft_2[SAMPLE_NUMBER].real(adc_r);
				fft_2[SAMPLE_NUMBER].imag(adc_i);
			}
		}
		FFT(fft_1, RANGE_FFT_SIZE, 1);
		FFT(fft_2, RANGE_FFT_SIZE, 1);
		for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER < RANGE_FFT_SIZE; SAMPLE_NUMBER++){
			rangeFFT1[SAMPLE_NUMBER][CHIRP_NUM] = fft_1[SAMPLE_NUMBER];
			rangeFFT2[SAMPLE_NUMBER][CHIRP_NUM] = fft_2[SAMPLE_NUMBER];
		}

		// applying MTI filtering
		if(enable_mti_filtering){
			for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER < RANGE_FFT_SIZE; SAMPLE_NUMBER++){

				rangeFFT1[SAMPLE_NUMBER][CHIRP_NUM] -= range_fft_spectrum_hist1[SAMPLE_NUMBER];
				rangeFFT2[SAMPLE_NUMBER][CHIRP_NUM] -= range_fft_spectrum_hist2[SAMPLE_NUMBER];

				range_fft_spectrum_hist1[SAMPLE_NUMBER] = ( alpha_mti * rangeFFT1[SAMPLE_NUMBER][CHIRP_NUM]+
												+ beta_mti * range_fft_spectrum_hist1[SAMPLE_NUMBER]);
				range_fft_spectrum_hist2[SAMPLE_NUMBER] = ( alpha_mti * rangeFFT2[SAMPLE_NUMBER][CHIRP_NUM]+
												+ beta_mti * range_fft_spectrum_hist2[SAMPLE_NUMBER]);

			}
		}
	}
	// RANGE TARGET DETECTION
	// detect the targets in range by applying constant amplitude threshold over range
	// data integration of range FFT over the chirps for target range detection

	double range_tx1rx1_max[RANGE_FFT_SIZE];
	double range_tx1rx2_max[RANGE_FFT_SIZE];

	for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER < RANGE_FFT_SIZE; SAMPLE_NUMBER++){
		double max_this_c1 = -1;
		double max_this_c2 = -1;
		for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t) num_chirps; CHIRP_NUM++){
			double	 tm1 = sqrt(pow(rangeFFT1[SAMPLE_NUMBER][CHIRP_NUM].real(),2)+
							pow(rangeFFT1[SAMPLE_NUMBER][CHIRP_NUM].imag(),2));
			double	 tm2 = sqrt(pow(rangeFFT2[SAMPLE_NUMBER][CHIRP_NUM].real(),2)+
												pow(rangeFFT2[SAMPLE_NUMBER][CHIRP_NUM].imag(),2));
			if(tm1 > max_this_c1){
				max_this_c1 = tm1;
			}
			if(tm2 > max_this_c2){
				max_this_c2 = tm2;
			}
		}
		range_tx1rx1_max[SAMPLE_NUMBER] = max_this_c1;
		range_tx1rx2_max[SAMPLE_NUMBER] = max_this_c2;

		this->range_tx1rx1_max_plot[SAMPLE_NUMBER] = range_tx1rx1_max[SAMPLE_NUMBER];
		this->range_tx1rx2_max_plot[SAMPLE_NUMBER] = range_tx1rx2_max[SAMPLE_NUMBER];
	}

	// PEAK SEARCH FIRST RESET PEAKS
	for(size_t tp=0;tp<MAX_NUM_TARGETS; tp++){
		tgt_range1[tp].index = -1;
		tgt_range1[tp].peak_val = 0.0;
		tgt_range2[tp].index = -1;
		tgt_range2[tp].peak_val = 0.0;
	}
	f_search_peak(range_tx1rx1_max, RANGE_FFT_SIZE, (double)RANGE_THRESHOLD,
			MAX_NUM_TARGETS, (double)MIN_DISTANCE,  (double)MAX_DISTANCE, (double)dist_per_bin, tgt_range1);
	// NUM PEAKS FROM ANT 1
	int num_peaks_ant_1 = 0;
	for(size_t tp=0;tp<MAX_NUM_TARGETS; tp++){
		if(tgt_range1[tp].index != -1){
			num_peaks_ant_1 += 1;
		}
	}
	f_search_peak(range_tx1rx2_max, RANGE_FFT_SIZE, (double)RANGE_THRESHOLD,
			MAX_NUM_TARGETS, (double)MIN_DISTANCE,  (double)MAX_DISTANCE, (double)dist_per_bin, tgt_range2);
	// NUM PEAKS FROM ANT 2
	int num_peaks_ant_2 = 0;
	for(size_t tp=0;tp<MAX_NUM_TARGETS; tp++){
		if(tgt_range2[tp].index != -1){
			num_peaks_ant_2 += 1;
		}
	}

	int max_t;
	int use_id=-1; // which antenna to use
	if(num_peaks_ant_1 < num_peaks_ant_2){
		max_t = num_peaks_ant_1;
		use_id = 0;
	}else{
		max_t = num_peaks_ant_2;
		use_id = 1;
	}

	// SLOW TIME PROCESSING
	// DOPPLER range_doppler_tx1rx1 [RANGE_FFT_SIZE][DOPPLER_FFT_SIZE]
	// COMPUTE MEAN ACROSS DOPPLER - only for targets i.e peaks
	vector<complex<double>> rx1_doppler_mean;
	vector<complex<double>> rx2_doppler_mean;

	if(num_peaks_ant_1 > 0 && num_peaks_ant_2 > 0){
		for(size_t tgt_=0; tgt_< (uint32_t)num_peaks_ant_1; tgt_++){
			complex<double> this_m(0.0,0.0);
			//complex<double> this_m2(0.0,0.0);
			for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t)num_chirps; CHIRP_NUM++){
				int bin_pos = tgt_range1[tgt_].index;
				this_m += rangeFFT1[bin_pos][CHIRP_NUM];
			}
			this_m.real( this_m.real() / (double)num_chirps);
			this_m.imag( this_m.imag() / (double)num_chirps);
			rx1_doppler_mean.push_back(this_m);
		}
		for(size_t tgt_=0; tgt_<(uint32_t)num_peaks_ant_2; tgt_++){
			complex<double> this_m(0.0,0.0);
			for (uint32_t CHIRP_NUM = 0; CHIRP_NUM <(uint32_t) num_chirps; CHIRP_NUM++){
				int bin_pos = tgt_range2[tgt_].index;
				this_m += rangeFFT2[bin_pos][CHIRP_NUM];
			}
			this_m.real( this_m.real() / (double)num_chirps);
			this_m.imag( this_m.imag() / (double)num_chirps);
			rx2_doppler_mean.push_back(this_m);
		}

		// MEAN REMOVAL
		for(size_t tgt_=0; tgt_<(uint32_t)num_peaks_ant_1; tgt_++){
			for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t)num_chirps; CHIRP_NUM++){
				int bin_pos = tgt_range1[tgt_].index;
				rangeFFT1[bin_pos][CHIRP_NUM] -= rx1_doppler_mean[tgt_];
			}
		}
		for(size_t tgt_=0; tgt_<(uint32_t)num_peaks_ant_2; tgt_++){
			for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t)num_chirps; CHIRP_NUM++){
				int bin_pos = tgt_range2[tgt_].index;
				rangeFFT2[bin_pos][CHIRP_NUM] -= rx2_doppler_mean[tgt_];
			}
		}

		// NOW WE FILL THE RANGE DOPPLER MAP
		//range_doppler_tx1rx1
		// -------------------- RX1
		// Window for the Doppler map and prepare for FFT
		// for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER < RANGE_FFT_SIZE; SAMPLE_NUMBER++){
		complex<double> range_fft_1[DOPPLER_FFT_SIZE];

		for(size_t tgt_=0; tgt_<(uint32_t)num_peaks_ant_1; tgt_++){
			for (uint32_t CHIRP_NUM = 0; CHIRP_NUM <(uint32_t) num_chirps; CHIRP_NUM++){
				int this_idx = tgt_range1[tgt_].index;
				if(this_idx < RANGE_FFT_SIZE and this_idx!= -1){
					range_fft_1[CHIRP_NUM].real(rangeFFT1[this_idx][CHIRP_NUM].real() * dopper_window_func[CHIRP_NUM]);
					range_fft_1[CHIRP_NUM].imag(rangeFFT1[this_idx][CHIRP_NUM].imag() * dopper_window_func[CHIRP_NUM]);
				}else{// up to DOPPLER_FFT_SIZE
					range_fft_1[CHIRP_NUM] =  (0.0);
				}
			}
			// FFT FIRST
			FFT(range_fft_1, DOPPLER_FFT_SIZE, 1);

			// flip before putting it in the doppler map ([half end | beg half] --> [beg half | half end])
			for(size_t THIS_DP=0; THIS_DP<DOPPLER_FFT_SIZE; THIS_DP++){
				int bin_pos = tgt_range1[tgt_].index;
				if(THIS_DP < (DOPPLER_FFT_SIZE/2)){
					range_doppler_tx1rx1[bin_pos][THIS_DP+(DOPPLER_FFT_SIZE/2)] = range_fft_1[THIS_DP];
				}else{
					range_doppler_tx1rx1[bin_pos][THIS_DP-(DOPPLER_FFT_SIZE/2)] = range_fft_1[THIS_DP];
				}
			}
		}
		// -------------------- RX2
		// Window for the Doppler map and prepare for FFT
		complex<double> range_fft_2[DOPPLER_FFT_SIZE];

		for(size_t tgt_=0; tgt_<(uint32_t)num_peaks_ant_2; tgt_++){

			for (uint32_t CHIRP_NUM = 0; CHIRP_NUM < (uint32_t)num_chirps; CHIRP_NUM++){
				int this_idx = tgt_range2[tgt_].index;
				if(this_idx < RANGE_FFT_SIZE and this_idx!= -1){
					range_fft_2[CHIRP_NUM].real(rangeFFT2[this_idx][CHIRP_NUM].real() * dopper_window_func[CHIRP_NUM]);
					range_fft_2[CHIRP_NUM].imag(rangeFFT2[this_idx][CHIRP_NUM].imag() * dopper_window_func[CHIRP_NUM]);
				}else{// up to DOPPLER_FFT_SIZE
					range_fft_2[CHIRP_NUM] =  (0.0);
				}
			}
			// FFT FIRST
			FFT(range_fft_2, DOPPLER_FFT_SIZE, 1);
			// flip before putting it in the doppler map ([half end | beg half] --> [beg half | half end])
			for(size_t THIS_DP=0; THIS_DP<DOPPLER_FFT_SIZE; THIS_DP++){
				int bin_pos = tgt_range2[tgt_].index;
				if(THIS_DP < DOPPLER_FFT_SIZE/2){
					range_doppler_tx1rx2[bin_pos][THIS_DP+(DOPPLER_FFT_SIZE/2)] = range_fft_2[THIS_DP];
				}else{
					range_doppler_tx1rx2[bin_pos][THIS_DP-(DOPPLER_FFT_SIZE/2)] = range_fft_2[THIS_DP];
				}
			}
		}

		// EXTRACTION OF INDICES FROM RANGE-DOPPLER MAP
		// TARGET MUST BE SEEN IN BOTH ANTENNAS - THIS CAN ALSO BE DIFFERENT, NO YET TARGET ID..
		int tgt_doppler_idx[max_t];
		complex<double> z1[max_t]; // Doppler range for targets -- to be used in MVDR or other super-resolution
		complex<double> z2[max_t];

		for(size_t NUM_TARGET=0; NUM_TARGET<(uint32_t)max_t; NUM_TARGET++){

			// find max val and dp index
			double max_dp=-1;
			int idx_dp = 0;
			for(size_t THIS_DP=0; THIS_DP<DOPPLER_FFT_SIZE; THIS_DP++){
				double tmp = sqrt(pow(range_doppler_tx1rx1[tgt_range1[NUM_TARGET].index][THIS_DP].real(), 2)+
						pow(range_doppler_tx1rx1[tgt_range1[NUM_TARGET].index][THIS_DP].imag(),2));
				if(max_dp<tmp){
					max_dp = tmp;
					idx_dp = THIS_DP;
				}
			}
			// consider the value of the range doppler map for the two receivers for targets with non
			// zero speed to compute angle of arrival.
			// for zero Doppler (targets with zero speed) calculate mean over Doppler to compute angle of arrival
			// index 17 corresponds to zero Doppler
			if(max_dp >= DOPPLER_THRESHOLD && idx_dp != DOPPLER_FFT_SIZE/2){
				tgt_doppler_idx[NUM_TARGET] = idx_dp;
				if(use_id == 0){
					z1[NUM_TARGET] = range_doppler_tx1rx1[tgt_range1[NUM_TARGET].index][idx_dp];//range_fft_1[tgt_range1[NUM_TARGET].index];//range_doppler_tx1rx1[tgt_range1[NUM_TARGET].index][idx_dp];
					z2[NUM_TARGET] = range_doppler_tx1rx2[tgt_range1[NUM_TARGET].index][idx_dp];//range_fft_2[tgt_range1[NUM_TARGET].index];//range_doppler_tx1rx2[tgt_range1[NUM_TARGET].index][idx_dp];
				}else if(use_id == 1){
					z1[NUM_TARGET] = range_doppler_tx1rx1[tgt_range2[NUM_TARGET].index][idx_dp];//range_fft_1[tgt_range2[NUM_TARGET].index];//range_doppler_tx1rx1[tgt_range2[NUM_TARGET].index][idx_dp];
					z2[NUM_TARGET] = range_doppler_tx1rx2[tgt_range2[NUM_TARGET].index][idx_dp];//range_fft_2[tgt_range2[NUM_TARGET].index];//range_doppler_tx1rx2[tgt_range2[NUM_TARGET].index][idx_dp];
				}
			}else{
				tgt_doppler_idx[NUM_TARGET] = DOPPLER_FFT_SIZE/2;//32
				z1[NUM_TARGET] = rx1_doppler_mean[NUM_TARGET];
				z2[NUM_TARGET] = rx2_doppler_mean[NUM_TARGET];
			}
		}
		// MEASUREMENT UPDATE
		vector<Measurement_elem_t> aux_current;
		for(size_t NUM_TARGET=0; NUM_TARGET<(uint32_t)max_t; NUM_TARGET++){
				Measurement_elem_t this_measure;
			if(use_id == 0){
				this_measure.strength = tgt_range1[NUM_TARGET].peak_val;
				this_measure.range = tgt_range1[NUM_TARGET].index*dist_per_bin;
				this_measure.speed = (tgt_doppler_idx[NUM_TARGET] - DOPPLER_FFT_SIZE/2) * -1.0*fD_per_bin * hz_to_mps_constant ;
				this_measure.rx1_angle_arg_re = z1[NUM_TARGET].real();
				this_measure.rx1_angle_arg_im = z1[NUM_TARGET].imag();
				this_measure.rx2_angle_arg_re = z2[NUM_TARGET].real();
				this_measure.rx2_angle_arg_im = z2[NUM_TARGET].imag();
				double this_angle = compute_angle(z1[NUM_TARGET], z2[NUM_TARGET], LAMBDA/ANTENNA_SPACING);
				this_measure.angle = this_angle;
			}else if(use_id == 1){
				this_measure.strength = tgt_range2[NUM_TARGET].peak_val;
				this_measure.range = tgt_range2[NUM_TARGET].index*dist_per_bin;
				this_measure.speed = (tgt_doppler_idx[NUM_TARGET] - DOPPLER_FFT_SIZE/2) * -1.0*fD_per_bin * hz_to_mps_constant ;
				this_measure.rx1_angle_arg_re = z1[NUM_TARGET].real();
				this_measure.rx1_angle_arg_im = z1[NUM_TARGET].imag();
				this_measure.rx2_angle_arg_re = z2[NUM_TARGET].real();
				this_measure.rx2_angle_arg_im = z2[NUM_TARGET].imag();
				double this_angle = compute_angle(z1[NUM_TARGET], z2[NUM_TARGET], LAMBDA/ANTENNA_SPACING);
				this_measure.angle = this_angle;
			}
			// Collect all detections
			aux_current.push_back(this_measure);
			// current_targets.push_back(this_measure);
		}

		current_targets = aux_current;
	}
}

//======================================================
double ofxRadar24Ghz::get_phase(float real, float imag)
{
	double phi;

	/* Phase angle (0 to 2Pi) */
	if((real > 0) && (imag >= 0))		// 1st quadrant
	{
		phi = atan((double)imag / (double)real);
	}
	else if((real < 0) && (imag >= 0))	// 2nd quadrant
	{
		phi = atan((double)imag / (double)real) + PI;
	}
	else if((real < 0) && (imag <= 0)) 	// 3rd quadrant
	{
		phi = atan((double)imag / (double)real) + PI;
	}
	else if((real > 0) && (imag <= 0)) 	// 4th quadrant
	{
		phi = atan((double)imag / (double)real) + 2*PI;
	}
	else if((real == 0) && (imag > 0))
	{
		phi = PI/2;
	}
	else if((real == 0) && (imag < 0))
	{
		phi = 3*PI/2;
	}
	else
	{
		phi = 0;	// Indeterminate
	}

	return(phi);
}

//======================================================
int ofxRadar24Ghz::compare_float(const void *a, const void *b){
	int retval = 0;

	float a_f = *(float*)a;
	float b_f = *(float*)b;

	if (a_f > b_f)
	{
		retval = 1;
	}
	else if (a_f < b_f)
	{
		retval = -1;
	}

	return retval;
}

//======================================================
void ofxRadar24Ghz::f_search_peak(double *fft_spectrum, int search_lenght, double threshold,
		int max_target_count, double min_distance,  double max_ditance, double dist_per_bin, target_peak * peak_idx ){

	// search for peaks
	int peak_cnt = 0;
	for(size_t n=3; n < (uint32_t)search_lenght-3; n++){
		//printf("searching for peaks n = %d\n", n);

		int fp_bin = n;
		int f1_bin = fp_bin -1;
		int f12_bin  = fp_bin -2;
		int fr_bin = fp_bin +1;
		int fr2_bin = fp_bin +2;

		double fp = fft_spectrum[fp_bin];
		double f1 = fft_spectrum[f1_bin];
		double f12 = fft_spectrum[f12_bin];
		double fr = fft_spectrum[fr_bin];
		double fr2 = fft_spectrum[fr2_bin];

		float peak_idxs = 0;
		uint32_t target_range;
		if(fp >= threshold && fp>= f12 && fp>=f1 && fp > fr && fp > fr2){

		    peak_idxs = (f12_bin * f12 + f1_bin * f1 + fp_bin * fp + fr_bin * fr + fr2_bin * fr2) / (f12 + f1 + fp + fr + fr2);
		    target_range = (uint32_t)((peak_idxs -1) * dist_per_bin);
			//double curr_range = (double) (fp_bin -1) * dist_per_bin;

			if(target_range >= min_distance && target_range <= max_ditance){

		        float fp_new;

		        if(peak_idxs > fp_bin)
		          fp_new = fp +(fr - fp) * (peak_idxs - fp_bin) / (fr_bin - fp_bin);
		        else
		          fp_new= f1 + (fp - f1) * (fp_bin - peak_idxs) / (fp_bin - f1_bin);

		        //target_info[num_of_targets].strength = fp_new;  // FFT magnitude level
		        //target_info[num_of_targets].range = target_range; // Range in centimeters (cm)
		        //target_info[num_of_targets].idx = (uint32_t)(peak_idxs + 0.5);  // index of FFT where target is detected (rounded)
		        //num_of_targets++;

		        peak_idx[peak_cnt].index = peak_idxs;
		        peak_idx[peak_cnt].peak_val = fp_new;
		        peak_cnt+=1;
				if(peak_cnt>=max_target_count){
					return;
					break;
				}
			}
		}
	}
	return;
}

////////////////////////////////////////////////////////////////////////////////////////////
// CALLBACKS
/* 24Ghz radar USB
 * Helper function for the ep_radar_base_get_frame_data
 * called every time ep_radar_base_get_frame_data method
 * is called to return measured time domain signals
 * */

//===========================================================================
void ofxRadar24Ghz::print_status_code( int32_t protocol_handle, int32_t status){

	//check status
	const char * hr_current_frame_format = protocol_get_status_code_description(protocol_handle, status);
	char buffer [1500];
	//int n;
	sprintf(buffer, "%s", hr_current_frame_format);
	printf("[%s]\n",buffer);


}

//===========================================================================
void ofxRadar24Ghz::get_frame_format( int32_t protocol_handle,
        uint8_t endpoint,
        Frame_Format_t* frame_format){

	// query
	int32_t current_frame_format = ep_radar_base_get_frame_format(protocolHandle, endpointRadarBase);
	print_status_code(protocolHandle, current_frame_format);

	// cast and read data format
	Frame_Format_t * frame_format_disp = (Frame_Format_t *) (frame_format_current);
	printf("num_chirps_per_frame %d\n", frame_format_disp->num_chirps_per_frame);
	printf("num_samples_per_chirp %d\n", frame_format_disp->num_samples_per_chirp);
	printf("rx_mask %d\n", frame_format_disp->rx_mask);
	printf("ONLY_I = 0 /  ONLY_Q = 1 / I_AND_Q = 2 %d\n", frame_format_disp->eSignalPart);

	frame_format->num_chirps_per_frame = frame_format_disp->num_chirps_per_frame;
	frame_format->num_samples_per_chirp = frame_format_disp->num_samples_per_chirp;
	frame_format->rx_mask = frame_format_disp->rx_mask;
	frame_format->eSignalPart = frame_format_disp->eSignalPart;

}

//===========================================================================
void ofxRadar24Ghz::received_frame_data(void* context,
						int32_t protocol_handle,
		                uint8_t endpoint,
						const Frame_Info_t* frame_info){

    float *full_data_block = (float *) (context);
    int num_ant = 2;
    if(frame_info->rx_mask == 3){
    	num_ant = 2;
    }

	for (uint32_t ANTENNA_NUMBER = 0; ANTENNA_NUMBER < (uint32_t)num_ant ; ANTENNA_NUMBER++){
		//uint32_t start = ant*frame_info->num_chirps*frame_info->num_samples_per_chirp*1
		for (uint32_t CHIRP_NUMBER = 0;CHIRP_NUMBER <  frame_info->num_chirps; CHIRP_NUMBER++){
			for (uint32_t SAMPLE_NUMBER = 0; SAMPLE_NUMBER < frame_info->num_samples_per_chirp; SAMPLE_NUMBER++)
			{
				if(frame_info->data_format != 0){
					const float * frame_start =  &frame_info->sample_data[CHIRP_NUMBER*num_ant*frame_info->num_samples_per_chirp*2];

					full_data_block[CHIRP_NUMBER*4*frame_info->num_samples_per_chirp + (2*ANTENNA_NUMBER)*frame_info->num_samples_per_chirp + SAMPLE_NUMBER] =
												frame_start[(2*ANTENNA_NUMBER)*frame_info->num_samples_per_chirp+SAMPLE_NUMBER];

					full_data_block[CHIRP_NUMBER*4*frame_info->num_samples_per_chirp + (2*ANTENNA_NUMBER+1)*frame_info->num_samples_per_chirp + SAMPLE_NUMBER] =
												frame_start[(2*ANTENNA_NUMBER+1)*frame_info->num_samples_per_chirp+SAMPLE_NUMBER];

				}else{
					printf("Not implemented: data format is real.. please check format.");
				}
			}
		}
	}
}

/* Function to get no of set bits in binary
   representation of positive integer n */
//===========================================================================
int ofxRadar24Ghz::countSetBits(unsigned int n){
	unsigned int count = 0;
	while (n) {
		count += n & 1;
		n >>= 1;
	}
	return count;
}

//===========================================================================
void ofxRadar24Ghz::received_temperature(void* context,
		int32_t protocol_handle,
        uint8_t endpoint,
		uint8_t temp_sensor,
        int32_t temperature_001C){

	//
    //float *temperature = (float *) (context);
    //printf("temperature %d:\n", frame_info->num_temp_sensors);

}

//===========================================================================
void ofxRadar24Ghz::received_frame_format(void* context,
		int32_t protocol_handle,
        uint8_t endpoint,
        const Frame_Format_t* frame_format){

	Frame_Format_t *frame_format_current = (Frame_Format_t *) (context);

	/*printf("num_chirps_per_frame %d\n", frame_format->num_chirps_per_frame);
	printf("num_samples_per_chirp %d\n", frame_format->num_samples_per_chirp);
	printf("rx_mask %d\n", frame_format->rx_mask);
	printf("ONLY_I = 0 /  ONLY_Q = 1 / I_AND_Q = 2 %d\n", frame_format->eSignalPart);*/

	frame_format_current->num_chirps_per_frame = frame_format->num_chirps_per_frame;
	frame_format_current->num_samples_per_chirp = frame_format->num_samples_per_chirp;
	frame_format_current->rx_mask = frame_format->rx_mask;
	frame_format_current->eSignalPart = frame_format->eSignalPart;


	//printf("data format is %d", frame_format->data_format);
	/*
	EP_RADAR_BASE_SIGNAL_ONLY_I  = 0,< Only the I signal is captured
                                           during radar data frame
                                           acquisition.
    EP_RADAR_BASE_SIGNAL_ONLY_Q  = 1, < Only the Q signal is captured
                                           during radar data frame
                                           acquisition.
    EP_RADAR_BASE_SIGNAL_I_AND_Q = 2  < Both, I and Q signal are captured
                                           as a complex signal during radar
                                           data frame acquisition. */

}

//===========================================================================
void ofxRadar24Ghz::get_device_info(void* context,
        int32_t protocol_handle,
        uint8_t endpoint,
		const Device_Info_t * device_info){

	Device_Info_t *this_device_info = (Device_Info_t *) (context);

	this_device_info->description = device_info->description;
	this_device_info->min_rf_frequency_kHz = device_info->min_rf_frequency_kHz;
	this_device_info->max_rf_frequency_kHz = device_info->max_rf_frequency_kHz;
	this_device_info->num_tx_antennas = device_info->num_tx_antennas;
	this_device_info->num_rx_antennas = device_info->num_rx_antennas;
	this_device_info->max_tx_power = device_info->max_tx_power;
	this_device_info->num_temp_sensors = device_info->num_temp_sensors;
	this_device_info->major_version_hw = device_info->major_version_hw;
	this_device_info->minor_version_hw = device_info->minor_version_hw;
	this_device_info->interleaved_rx = device_info->interleaved_rx;
	this_device_info->data_format = device_info->data_format;

	printf("max_tx_power %d\n", device_info->max_tx_power);
	printf("num_tx_antennas %d\n", device_info->num_tx_antennas);
	printf("num_rx_antennas %d\n", device_info->num_rx_antennas);
	printf("data_format %d interleaved_rx %d\n", device_info->data_format, device_info->interleaved_rx);
	printf("min_rf_frequency_kHz %d max_rf_frequency_kHz %d\n", device_info->min_rf_frequency_kHz, device_info->max_rf_frequency_kHz);
	printf("bandwidth %d kHz\n", device_info->max_rf_frequency_kHz-device_info->min_rf_frequency_kHz);
	printf("num_temp_sensors  %d\n", device_info->num_temp_sensors);
	printf("version %d-%d\n", device_info->major_version_hw, device_info->minor_version_hw);

}

/* * \param[in] context          The context data pointer, provided along with
 *                             the callback itself through
 *                             \ref ep_radar_base_set_callback_tx_power.
 * \param[in] protocol_handle  The handle of the connection, the sending
 *                             device is connected to.
 * \param[in] endpoint         The number of the endpoint that has sent the
 *                             message.
 * \param[in] tx_antenna       The number of the TX antenna from which the
 *                             power was measured.
 * \param[in] tx_power_001dBm  The power value in 0.001 dBm.*/
//===========================================================================
void ofxRadar24Ghz::get_tx_power(void* context,
			int32_t protocol_handle,
			uint8_t endpoint,
			uint8_t tx_antenna,
			int32_t tx_power_001dBm){

	uint32_t * power_set = (uint32_t *) context;
	*power_set = tx_power_001dBm;
	printf("power is set to %f dBm\n", (double)tx_power_001dBm*(1e-3));


}

//===========================================================================
void ofxRadar24Ghz::set_fmcw_conf(void* context,
            int32_t protocol_handle,
            uint8_t endpoint,
            const Fmcw_Configuration_t*
              fmcw_configuration){

	//Fmcw_Configuration_t *sd = (Fmcw_Configuration_t * )context;
	printf("lower_frequency_kHz %d \n", fmcw_configuration->lower_frequency_kHz);
	printf("upper_frequency_kHz %d \n", fmcw_configuration->upper_frequency_kHz);
	printf("tx_power %d \n", fmcw_configuration->tx_power);


}

//===========================================================================
void ofxRadar24Ghz::get_bw_sec(void* context,
		   int32_t protocol_handle,
		   uint8_t endpoint,
		   uint32_t bandwidth_per_second){

	uint32_t * bps = (uint32_t *) context;
	printf("bandwidth_per_second %d \n", bandwidth_per_second);
	* bps = bandwidth_per_second;

}

//===========================================================================
void ofxRadar24Ghz::get_chirp_duration(void* context,
        int32_t protocol_handle,
        uint8_t endpoint,
        uint32_t chirp_duration_ns){

	uint32_t * cd = (uint32_t *) context;
	printf("chirp Duration is %d ns\n", chirp_duration_ns);
	*cd = chirp_duration_ns;
}

//===========================================================================
void ofxRadar24Ghz::get_min_frame_interval(void* context,
        int32_t protocol_handle,
        uint8_t endpoint,
        uint32_t min_frame_interval_us){

	uint32_t * cd = (uint32_t *) context;
	printf("min_frame_interval is %d us\n", min_frame_interval_us);
	*cd = min_frame_interval_us;

}