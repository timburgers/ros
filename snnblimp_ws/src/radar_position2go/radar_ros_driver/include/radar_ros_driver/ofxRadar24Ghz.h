#ifndef OFXRADAR24GHZ_H
#define OFXRADAR24GHZ_H

#include <stdio.h>
#include <string.h>
#include <valarray>

// Radar - Host communication
#include "radar_ros_driver/EndpointCalibration.h"
#include "radar_ros_driver/EndpointTargetDetection.h"
#include "radar_ros_driver/EndpointRadarAdcxmc.h"
#include "radar_ros_driver/EndpointRadarBase.h"
#include "radar_ros_driver/EndpointRadarDoppler.h"
#include "radar_ros_driver/EndpointRadarFmcw.h"
#include "radar_ros_driver/EndpointRadarP2G.h"
#include "radar_ros_driver/EndpointRadarIndustrial.h"
#include "radar_ros_driver/Protocol.h"
#include "radar_ros_driver/COMPort.h"

// FFT
#include <iostream>
#include <complex>
#include <fstream>
#include <vector>

#include <algorithm>

// ROS
#include <ros/ros.h>


// Global Definitions
#define FRAME_PERIOD_MSEC 			(150U)    	    // Time period of one frame to capture
#define ANTENNA_SPACING 		    0.0062          // For angle estimation
#define LAMBDA 					    0.0124          // For angle estimation

#define RANGE_FFT_SIZE              256      //**1
#define DOWN_CHIRP_DURATION         0.0001  
#define CHIRP_TO_CHIRP_DELAY        0.0001
#define RANGE_THRESHOLD             800//200 //***
#define DOPPLER_THRESHOLD           50       //***
#define MIN_DISTANCE                0        //***
#define MAX_DISTANCE                4        //***
#define MAX_NUM_TARGETS             1        //***
#define INDEX_ZERO_DOPPLER 			17	 			// --------- !!!!!!!!!!!! WHAT IS THIS ?????????? !!!!!!!!!! ------------ //	
										 			// --------- !!!!!!!!!!!! WHAT IS THIS ?????????? !!!!!!!!!! ------------ //	

// YES I KNOW...  consider the value of the range doppler map for the two receivers for targets with non zero speed to compute angle of arrival.
// For zero Doppler (targets with zero speed) calculate mean over Doppler to compute angle of arrival index 17 corresponds to zero Doppler.

#define LMA_SIZE                    9		 //***
#define EMA_SIZE                    9       //***
#define EMA_ALPHA                   0.1      //***
#define MF_SIZE                     15       //***

#define NUM_OF_CHIRPS               16

#define AUTOMATIC_DATA_FRAME_TRIGGER 0		        // define if automatic trigger is active or not
#define AUTOMATIC_DATA_TRIGER_TIME_US (300)	        // get ADC data each 300us in not automatic trigger mode
#define SPEED_OF_LIGHT              2.998e8

#define CHIRP_DUR_NS                300000
#define	NUM_CHIRPS_PER_FRAME        16		 //**2
#define DOPPLER_FFT_SIZE            64	     //**2       // == NUM_CHIRPS_PER_FRAME!!!
#define	NUM_SAMPLES_PER_CHIRP       64		 //**1
#define E_SIGNAL_PART               2  			    //ONLY_I = 0 /  ONLY_Q = 1 / I_AND_Q = 2
#define RX_MASK 					3			
// Each available RX antenna is represented by a bit in this mask. 
// If a bit is set, the IF signal received through the according RX antenna is captured during chirp processing.
#define RX_NUM_ANTENNAS             2
#define PEAK_TRESHOLD               0.7				// --------- !!!!!!!!!!!! WHAT IS THIS ?????????? !!!!!!!!!! ------------ //	
										 			// --------- !!!!!!!!!!!! WHAT IS THIS ?????????? !!!!!!!!!! ------------ //

#define MIN_ANGLE_FOR_ASSIGNMENT    50.0			// --------- !!!!!!!!!!!! WHAT IS THIS ?????????? !!!!!!!!!! ------------ //	
										 			// --------- !!!!!!!!!!!! WHAT IS THIS ?????????? !!!!!!!!!! ------------ //
#define IGNORE_NAN		            (555U)
#define ANGLE_QUANTIZATION			(1U)		    // Enable and set the Number of degrees

#define MTI_FILTER_LEN              100
#define MAXIMUM_NUMBER_HISTORY      40

#define PI							3.14159265358979323846

/*
==============================================================================
   3. TYPES
==============================================================================
*/


/**
 * \brief Data structure for current measurements used in data association.
 * @{
 */
typedef struct
{
	uint16_t is_associated;
	float    strength;
	float    range;
	float    speed;
	float	 angle;
	float    rx1_angle_arg_re;
	float    rx1_angle_arg_im;
	float    rx2_angle_arg_re;
	float    rx2_angle_arg_im;
} Measurement_elem_t;

typedef struct
{
	int index;
	double peak_val;
}target_peak;


using namespace std;

class ofxRadar24Ghz {

	public:
		void setup();
		void update();

		// frame initialize memory
		int num_chirps;
		int num_samples_per_chirp;
		int esignalpart;
		int rx_mask;
		int num_antennas;
		int moving_average_size;

		int radar_handle = 0;
		int num_of_ports = 0;
		char comp_port_list[RANGE_FFT_SIZE];
		char* comport;
		const char *delim = ";";

		int res;
		int protocolHandle;
		int endpointRadarBase;
		bool acq_started;


		// Algorithm
		double speed_of_light;
		double fC;
		double PRT;
		int fs;
		int BW;
		int range_fft_size;
		int doppler_fft_size;
		int range_threshold;
		int doppler_threshold;
		int min_distance;
		int max_distance;
		int max_num_targets;
		double lambda;
		double hz_to_mps_constant;
		double if_scale;
		double *range_window_func;
		double *dopper_window_func;
		int r_max;
		double dist_per_bin;
		vector<double> array_bin_range;
		double fD_max;
		double fD_per_bin;
		vector<double> array_bin_fD;

		bool enable_mti_filtering;
		complex<double> * range_fft_spectrum_hist1; // FOR MTI
		complex<double> * range_fft_spectrum_hist2; // FOR MTI
		complex<double> * fft_1; // range FFT
		complex<double> * fft_2; // range FFT

		// FFT
		double *adc_real_tx1rx1;	// REAL
		double *adc_imag_tx1rx1;	// IMG
		double *adc_real_tx1rx2;	// REAL
		double *adc_imag_tx1rx2;	// IMG

		// ADC I AND Q two antennas
		float *full_data_block;
		float *temperature;
		void  *frame_format_current;
		void  *device_info;
		void  *fmcw_cfg;
		uint32_t  *chirp_duration_ns;
		uint32_t  *min_frame_interval_us;
		uint32_t  *tx_power_001dBm;
		// =========================================

		// FFT maps
		vector<vector<complex<double>>> range_tx1rx1;
		vector<vector<complex<double>>> range_tx1rx2;

		double * distance_m;

		// DOPPLER
		vector<vector<complex<double>>> range_doppler_tx1rx1;
		vector<vector<complex<double>>> range_doppler_tx1rx2;
		vector<vector<complex<double>>> rangeFFT1;
		vector<vector<complex<double>>> rangeFFT2;


		// END
		double range_tx1rx1_max_plot[RANGE_FFT_SIZE];
		double range_tx1rx2_max_plot[RANGE_FFT_SIZE];
		vector<Measurement_elem_t> current_targets; // here the content
		//vector<vector<Measurement_elem_t>> targets_MA;
		//vector<float> range_MA;

		int countSetBits(unsigned int n);

		void changeMTI();
		void changeLoadData();

		void f_search_peak(double * fft_spectrum, int search_lenght, double threshold, int max_target_count,
				double min_distance,  double max_distance, double dist_per_bin, target_peak *tgt_range);
		void startRadarUSB();

		target_peak *tgt_range1;
		target_peak *tgt_range2;


		double calculateBeatFreq(double distance_m, double bandwidth_hz, double speed_of_light, double ramp_time_s);

		ifstream bindayDataIn;

		bool isloaddata;
		bool file_loaded;
		bool repeat_mode;

		bool islive;

		int 	radar_auto_connect();  // just pick the only radar available
		void 	print_status_code( int32_t protocol_handle, int32_t status);

		double blackman(double i, double N) {
			double a0 = 0.42;
			double a1 = 0.5;
			double a2 = 0.08;
			double f = 6.283185307179586*i/(N-1);
			return a0 - a1 * cos(f) + a2*cos(2.0*f);
		}


		/***************************************************************************
		 calculate a chebyshev window of size N, store coeffs in out as in out
		-out should be array of size N
		-atten is the required sidelobe attenuation (e.g. if you want -60dB atten, use '60')
		***************************************************************************/
		void cheby_win(double *out, int N, float atten){
			int nn, i;
			double M, n, sum = 0, max=0;
			double tg = pow(10,atten/20);  /* 1/r term [2], 10^gamma [2] */
			double x0 = cosh((1.0/(N-1))*acosh(tg));
			M = (N-1)/2;
			if(N%2==0) M = M + 0.5; /* handle even length windows */
			for(nn=0; nn<(N/2+1); nn++){
				n = nn-M;
				sum = 0;
				for(i=1; i<=M; i++){
					sum += cheby_poly(N-1,x0*cos(PI*i/N))*cos(2.0*n*PI*i/N);
				}
				out[nn] = tg + 2*sum;
				out[N-nn-1] = out[nn];
				if(out[nn]>max)max=out[nn];
			}
			for(nn=0; nn<N; nn++) out[nn] /= max; /* normalise everything */
			return;
		}

		/**************************************************************************
		This function computes the chebyshev polyomial T_n(x)
		***************************************************************************/
		double cheby_poly(int n, double x){
			double res;
			if (fabs(x) <= 1) res = cos(n*acos(x));
			else              res = cosh(n*acosh(x));
			return res;
		}

		static int compare_float(const void* a, const void* b);

		double compute_angle(complex<double> z1, complex<double> z2, float wave_length_ant_spacing_ratio){
			double angle_rx1 = atan2(z1.imag(), z1.real()); // phase of received signal for rx1
			double angle_rx2 = atan2(z2.imag(), z2.real()); // phase of received signal for rx2

			double d_phi = angle_rx1 - angle_rx2;
			if(d_phi <= 0){
				d_phi = d_phi + 2*PI;
			}
			d_phi = d_phi - PI;
			double target_angle = asin(d_phi * wave_length_ant_spacing_ratio  / (2*PI));

			return (target_angle*(180/PI)); // deg

		}

		double get_phase(float real, float imag);

		/**
		 * \brief  This function computes the FFt signal out of raw ADC samples.
		 *
		 *  Internally it computes mean of respective I & Q signal and subtract it before applying IF scaling and Windowing.
		 *  Afterwards computes the FFT signal and returns the Nf number of complex samples.
		 *
		 * \param[in]	*i_data		Pointer of type signed 16-bit integer, containing the address of the I data buffer
		 * \param[in]	*q_data		Pointer of type signed 16-bit integer, containing the address of the Q data buffer
		 * \param[in]	Nd			Unsigned 16-bit integer, containing the size of raw ADC IQ data buffer
		 * \param[in]	Nf			Unsigned 16-bit integer, containing the size of FFT complex values array
		 * \param[in]	if_scale	Floating point scale applied to the FFT spectrum to enhance the visibility of targets
		 * \param[in]	fft_type	Complex or Real input FFT to be computed defined by \ref FFT_Input_t
		 * \param[in]	fft_direction	Fast or Slow FFT to be computed defined by \ref FFT_Direction_t
		 *
		 * \param[out]  *i_mean		Pointer to a floating point value, containing the mean of the I channel
		 * \param[out]  *q_mean		Pointer to a floating point value, containing the mean of the Q channel
		 * \param[out]  *complex_fft_signal		Pointer to a floating point array, to return the complex FFT signal in interleaved I&Q format.
		 *
		 */
		//void compute_fft_signal(float* i_data, float* q_data, uint16_t Nd, uint16_t Nf, float if_scale,
		//						FFT_Input_t fft_type, FFT_Direction_t fft_direction,
		//						float* i_mean, float* q_mean, float* complex_fft_signal);

		/**
		 * \brief  This function computes the FFt spectrum out of raw ADC samples.
		 *
		 *  Internally it computes mean of respective I & Q signal and subtract it before applying IF scaling and Windowing.
		 *  Afterwards computes the FFT signal and returns the Nf number of real samples as FFT spectrum.
		 *
		 * \param[in]	*fft_input_signal		Pointer of type float, containing the address of the Complex FFT signal with interleaved IQ
		 * \param[in]	Nf						Unsigned 32-bit integer, containing the size of FFT complex values array
		 *
		 * \param[out]  *fft_output_spectrum	Pointer to a floating point array, to return the real valued FFT spectrum.
		 *
		 */
		//void compute_fft_spectrum(float* fft_input_signal, uint32_t Nf, float * fft_output_spectrum);


		////////////////////////////////////
		// callback functions COMMUNICATION
		////////////////////////////////////

		// query frame format
		void get_frame_format(int32_t protocol_handle,
				uint8_t endpoint,
				Frame_Format_t* frame_format);


		static void 	received_frame_data(void* context,
							int32_t protocol_handle,
							uint8_t endpoint,
							const Frame_Info_t* frame_info);

		static void received_frame_format(void* context,
				int32_t protocol_handle,
				uint8_t endpoint,
				const Frame_Format_t* frame_format);

		static void received_temperature(void* context,
				int32_t protocol_handle,
				uint8_t endpoint,
				uint8_t temp_sensor,
				int32_t temperature_001C);

		static void get_chirp_duration(void* context,
				int32_t protocol_handle,
				uint8_t endpoint,
				uint32_t chirp_duration_ns);

		static void get_device_info(void* context,
				int32_t protocol_handle,
				uint8_t endpoint,
				const Device_Info_t* device_info);

		static void get_tx_power(void* context,
				int32_t protocol_handle,
				uint8_t endpoint,
				uint8_t tx_antenna,
				int32_t tx_power_001dBm);


		static void get_bw_sec(void* context,
			int32_t protocol_handle,
			uint8_t endpoint,
			uint32_t bandwidth_per_second);


		static void set_fmcw_conf(void* context,
				int32_t protocol_handle,
				uint8_t endpoint,
				const Fmcw_Configuration_t*
				fmcw_configuration);

		static void get_min_frame_interval(void* context,
										int32_t protocol_handle,
										uint8_t endpoint,
										uint32_t min_frame_interval_us);


		// INLINE DEFINED DSP
		int log2fft(int N)    //funzione per calcolare il logaritmo in base 2 di un intero
		{
		int k = N, i = 0;
		while(k) {
			k >>= 1;
			i++;
		}
		return i - 1;
		}

		int checkfft(int n)    //usato per controllare se il numero di componenti del vettore di input è una potenza di 2
		{
		return n > 0 && (n & (n - 1)) == 0;
		}

		int reversefft(int N, int n)    //calcola il reverse number di ogni intero n rispetto al numero massimo N
		{
		int j, p = 0;
		for(j = 1; j <= log2fft(N); j++) {
			if(n & (1 << (log2fft(N) - j)))
			p |= 1 << (j - 1);
		}
		return p;
		}

		void ordinafft(complex<double>* f1, int N)     //dispone gli elementi del vettore ordinandoli per reverse order
		{
		complex<double> f2[N];
		for(int i = 0; i < N; i++)
			f2[i] = f1[reversefft(N, i)];
		for(int j = 0; j < N; j++)
			f1[j] = f2[j];
		}

		void transformfft(complex<double>* f, int N)     //calcola il vettore trasformato
		{
			ordinafft(f, N);    //dapprima lo ordina col reverse order
			complex<double> W[N / 2]; //vettore degli zeri dell'unità.
										//Prima N/2-1 ma genera errore con ciclo for successivo
									//in quanto prova a copiare in una zona non allocata "W[N/2-1]"
			W[1] = polar(1., -2. * M_PI / N);
			W[0] = 1;
			for(int i = 2; i < N / 2; i++)
				W[i] = pow(W[1], i);
			int n = 1;
			int a = N / 2;
			for(int j = 0; j < log2fft(N); j++) {
				for(int i = 0; i < N; i++) {
				if(!(i & n)) {
					/*ad ogni step di raddoppiamento di n, vengono utilizzati gli indici */
					/*'i' presi alternativamente a gruppetti di n, una volta si e una no.*/
					complex<double> temp = f[i];
					complex<double> Temp = W[(i * a) % (n * a)] * f[i + n];
					f[i] = temp + Temp;
					f[i + n] = temp - Temp;
				}
				}
				n *= 2;
				a = a / 2;
			}
		}

		void FFT(complex<double>* f, int N, double d)
		{
		transformfft(f, N);
		for(int i = 0; i < N; i++)
			f[i] *= d; //moltiplica il vettore per il passo in modo da avere il vettore trasformato effettivo
		}
	};

#endif