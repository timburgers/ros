#include <ros/ros.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <string>
// #include <sensor_msgs/Range.h>
#include "tfmini_ros/Range.h"
#include <std_msgs/Float32.h>

namespace benewake
{
  class TFmini
  {
    public:
      TFmini(std::string _name, int _baudRate);
      ~TFmini(){};
      float getDist();
      void closePort();

      unsigned char dataBuf[7];

    private:
      std::string portName_;
      int baudRate_;
      int serial_;

      bool readData(unsigned char *_buf, int _nRead);
  };
}
