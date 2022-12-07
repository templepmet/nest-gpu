#include "non_blocking_timer.h"

#include <string>
#include <cassert>
#include "getRealTime.h"

#include <cuda.h>
#include <nvtx3/nvToolsExt.h>


NonBlockingTimer *SpikeBufferUpdate_timer;
NonBlockingTimer *poisson_generator_timer;
NonBlockingTimer *neuron_Update_timer;
NonBlockingTimer *copy_ext_spike_timer;
NonBlockingTimer *SendExternalSpike_timer;
NonBlockingTimer *SendSpikeToRemote_timer;
NonBlockingTimer *RecvSpikeFromRemote_timer;
NonBlockingTimer *CopySpikeFromRemote_timer;
NonBlockingTimer *MpiBarrier_timer;
NonBlockingTimer *copy_spike_timer;
NonBlockingTimer *ClearGetSpikeArrays_timer;
NonBlockingTimer *NestedLoop_timer;
NonBlockingTimer *GetSpike_timer;
NonBlockingTimer *SpikeReset_timer;
NonBlockingTimer *ExternalSpikeReset_timer;
NonBlockingTimer *RevSpikeBufferUpdate_timer;
NonBlockingTimer *BufferRecSpikeTimes_timer;
NonBlockingTimer *Other_timer;

double RecvSpikeWait_time; // comm_wait


CudaEventPair::CudaEventPair()
{
  cudaEventCreate(&start_e);
  cudaEventCreate(&stop_e);
}

CudaEventPair::~CudaEventPair()
{
  cudaEventDestroy(start_e);
  cudaEventDestroy(stop_e);
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
  cudaEventRecord(cep->start_e);
}

void NonBlockingTimer::stopRecordDevice()
{
  assert(is_start_d);
  is_start_d = false;

  assert(!available_queue.empty());
  CudaEventPair *cep = available_queue.front();
  available_queue.pop();
  used_queue.push(cep);
  cudaEventRecord(cep->stop_e);
}

void NonBlockingTimer::_consumeRecord(bool is_sync)
{
  float milliseconds;
  while (!used_queue.empty()) {
    CudaEventPair *cep = used_queue.front();
    if (cudaEventQuery(cep->stop_e) != cudaSuccess) {
      if (is_sync) {
        cudaEventSynchronize(cep->stop_e);
      }
      else {
        break;
      }
    }
    cudaEventElapsedTime(&milliseconds, cep->start_e, cep->stop_e);
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
