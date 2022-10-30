#ifndef TIMER_ASYNC_H
#define TIMER_ASYNC_H

#include <cassert>
#include <iostream>
#include <cuda.h>

#include "getRealTime.h"

class NonBlockingTimer
{

private:
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

  void _addElapsedTime()
  {
    float milliseconds;
    if (cudaEventQuery(stop_d) != cudaSuccess)
    {
      cudaEventSynchronize(stop_d);
    }
    cudaEventElapsedTime(&milliseconds, start_d, stop_d);
    time_d += milliseconds / 1e3;
  }

public:
  NonBlockingTimer()
  {
    time_h = 0;
    start_h = 0;
    is_start_h = false;

    time_d = 0;
    cudaEventCreate(&start_d);
    cudaEventCreate(&stop_d);
    is_start_d = false;
    is_async = false;
  }

  ~NonBlockingTimer()
  {
    cudaEventDestroy(start_d);
    cudaEventDestroy(stop_d);
  }

  void startRecordDevice()
  {
    assert(!is_start_d);
    is_start_d = true;

    if (is_async)
    {
      _addElapsedTime();
      is_async = false;
    }
    cudaEventRecord(start_d);
  }

  void stopRecordDevice()
  {
    assert(is_start_d);
    is_start_d = false;

    cudaEventRecord(stop_d);
    is_async = true;
  }

  void startRecordHost()
  {
    assert(!is_start_h);
    is_start_h = true;
    start_h = getRealTime();
  }

  void stopRecordHost()
  {
    assert(is_start_h);
    is_start_h = false;
    time_h += (getRealTime() - start_h);
  }

  void startRecord()
  {
    startRecordDevice();
    startRecordHost();
  }

  void stopRecord()
  {
    stopRecordDevice();
    stopRecordHost();
  }

  double getTimeHost()
  {
    assert(!is_start_h);
    return time_h;
  }

  double getTimeDevice()
  {
    assert(!is_start_d);
    if (is_async)
    {
      _addElapsedTime();
      is_async = false;
    }
    return time_d;
  }

  double getTime()
  {
    return getTimeHost();
  }
};

#endif // TIMER_ASYNC_H;
