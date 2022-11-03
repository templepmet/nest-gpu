#ifndef NON_BLOCKING_TIMER_H
#define NON_BLOCKING_TIMER_H

#include <cuda.h>
#include <string>

class NonBlockingTimer
{

private:
  char *label;
  
  // host
  double time_h;
  double start_h;
  bool is_start_h;

  // device
  double time_d;
  cudaEvent_t start_d;
  cudaEvent_t stop_d;
  bool is_start_d;
  bool is_async; // is running gpu

  void startRecordHost();

  void stopRecordHost();

  void startRecordDevice();

  void stopRecordDevice();

  void _addElapsedTime();

public:
  NonBlockingTimer(const char *label);

  ~NonBlockingTimer();

  void startRecord();

  void stopRecord();

  double getTimeHost();

  double getTimeDevice();

  double getTime();
};

#endif // NON_BLOCKING_TIMER_H;
