#include "non_blocking_timer.h"

#include <cassert>
#include <cuda.h>
#include "getRealTime.h"

NonBlockingTimer::NonBlockingTimer()
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

NonBlockingTimer::~NonBlockingTimer()
{
  cudaEventDestroy(start_d);
  cudaEventDestroy(stop_d);
}

void NonBlockingTimer::startRecordHost()
{
  assert(!is_start_h);
  is_start_h = true;
  start_h = getRealTime();
}

void NonBlockingTimer::stopRecordHost()
{
  assert(is_start_h);
  is_start_h = false;
  time_h += (getRealTime() - start_h);
}

void NonBlockingTimer::startRecordDevice()
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

void NonBlockingTimer::stopRecordDevice()
{
  assert(is_start_d);
  is_start_d = false;
  cudaEventRecord(stop_d);
  is_async = true;
}

void NonBlockingTimer::_addElapsedTime()
{
  float milliseconds;
  if (cudaEventQuery(stop_d) != cudaSuccess)
  {
    cudaEventSynchronize(stop_d);
  }
  cudaEventElapsedTime(&milliseconds, start_d, stop_d);
  time_d += milliseconds / 1e3;
}

void NonBlockingTimer::startRecord()
{
  startRecordHost();
  startRecordDevice();
}

void NonBlockingTimer::stopRecord()
{
  stopRecordHost();
  stopRecordDevice();
}

double NonBlockingTimer::getTimeHost()
{
  assert(!is_start_h);
  return time_h;
}

double NonBlockingTimer::getTimeDevice()
{
  assert(!is_start_d);
  if (is_async)
  {
    _addElapsedTime();
    is_async = false;
  }
  return time_d;
}

double NonBlockingTimer::getTime()
{
  return getTimeHost();
}
