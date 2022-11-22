#include "non_blocking_timer.h"

#include <string>
#include <cassert>
#include "getRealTime.h"

#include <cuda.h>
#include <nvtx3/nvToolsExt.h>


CudaEventPair::CudaEventPair()
{
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
}

CudaEventPair::~CudaEventPair()
{
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


NonBlockingTimer::NonBlockingTimer(const char *label)
{
  this->label = (char*)label;
  time_h = 0;
  start_h = 0;
  is_start_h = false;
  time_d = 0;
  is_start_d = false;
  available_queue.push(new CudaEventPair());
}

NonBlockingTimer::~NonBlockingTimer()
{
  while (!available_queue.empty()) {
    CudaEventPair *cep = available_queue.front();
    available_queue.pop();
    delete cep;
  }
  while (!used_queue.empty()) {
    CudaEventPair *cep = used_queue.front();
    used_queue.pop();
    delete cep;
  }
}

void NonBlockingTimer::startRecordHost()
{
  nvtxRangePush(label);
  assert(!is_start_h);
  is_start_h = true;
  start_h = getRealTime();
}

void NonBlockingTimer::stopRecordHost()
{
  assert(is_start_h);
  is_start_h = false;
  time_h += (getRealTime() - start_h);
  nvtxRangePop();
}

void NonBlockingTimer::startRecordDevice()
{
  assert(!is_start_d);
  is_start_d = true;

  // consume
  _consumeRecord(false);

  // produce
  if (available_queue.empty()) {
    available_queue.push(new CudaEventPair());
  }

  CudaEventPair *cep = available_queue.front();
  cudaEventRecord(cep->start);
}

void NonBlockingTimer::stopRecordDevice()
{
  assert(is_start_d);
  is_start_d = false;

  assert(!available_queue.empty());
  CudaEventPair *cep = available_queue.front();
  available_queue.pop();
  used_queue.push(cep);
  cudaEventRecord(cep->stop);
}

void NonBlockingTimer::_consumeRecord(bool is_sync)
{
  float milliseconds;
  while (!used_queue.empty()) {
    CudaEventPair *cep = used_queue.front();
    if (cudaEventQuery(cep->stop) != cudaSuccess) {
      if (is_sync) {
        cudaEventSynchronize(cep->stop);
      }
      else {
        break;
      }
    }
    cudaEventElapsedTime(&milliseconds, cep->start, cep->stop);
    time_d += milliseconds / 1e3;
    used_queue.pop();
    available_queue.push(cep);
  }
}

void NonBlockingTimer::startRecord()
{
  startRecordHost();
  startRecordDevice();
}

void NonBlockingTimer::stopRecord()
{
  stopRecordDevice();
  stopRecordHost();
}

double NonBlockingTimer::getTimeHost()
{
  assert(!is_start_h);
  return time_h;
}

double NonBlockingTimer::getTimeDevice()
{
  assert(!is_start_d);
  _consumeRecord(true);
  return time_d;
}

double NonBlockingTimer::getTime()
{
  return getTimeHost() + getTimeDevice();
}
