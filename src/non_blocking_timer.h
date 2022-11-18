#ifndef NON_BLOCKING_TIMER_H
#define NON_BLOCKING_TIMER_H

#include <cuda.h>
#include <string>
#include <queue>

class CudaEventPair
{
public:
  cudaEvent_t start;
  cudaEvent_t stop;

  CudaEventPair();
  ~CudaEventPair();
};

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
  bool is_start_d;
  std::queue<CudaEventPair *> used_queue;
  std::queue<CudaEventPair *> available_queue;

  void startRecordHost();

  void stopRecordHost();

  void startRecordDevice();

  void stopRecordDevice();

  void _consumeRecord(bool is_sync);

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
