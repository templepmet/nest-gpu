#ifndef NON_BLOCKING_TIMER_H
#define NON_BLOCKING_TIMER_H

#include <cuda.h>
#include <string>
#include <queue>

class CudaEventPair
{
public:
  cudaEvent_t start_e;
  cudaEvent_t stop_e;

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

  void _consumeRecord(bool is_sync);

public:
  NonBlockingTimer(const char *label);

  ~NonBlockingTimer();

  void startRecordHost();

  void stopRecordHost();

  void startRecordDevice();

  void stopRecordDevice();

  void startRecord();

  void stopRecord();

  double getTimeHost();

  double getTimeDevice();

  double getTime();
};

extern NonBlockingTimer *SpikeBufferUpdate_timer;
extern NonBlockingTimer *poisson_generator_timer;
extern NonBlockingTimer *neuron_Update_timer;
extern NonBlockingTimer *copy_ext_spike_timer;
extern NonBlockingTimer *SendExternalSpike_timer;
extern NonBlockingTimer *SendSpikeToRemote_timer;
extern NonBlockingTimer *RecvSpikeFromRemote_timer;
extern NonBlockingTimer *CopySpikeFromRemote_timer;
extern NonBlockingTimer *MpiBarrier_timer;
extern NonBlockingTimer *copy_spike_timer;
extern NonBlockingTimer *ClearGetSpikeArrays_timer;
extern NonBlockingTimer *NestedLoop_timer;
extern NonBlockingTimer *GetSpike_timer;
extern NonBlockingTimer *SpikeReset_timer;
extern NonBlockingTimer *ExternalSpikeReset_timer;
extern NonBlockingTimer *RevSpikeBufferUpdate_timer;
extern NonBlockingTimer *BufferRecSpikeTimes_timer;
extern NonBlockingTimer *Blocking_timer;

extern double RecvWait_time; // comm_wait

#endif // NON_BLOCKING_TIMER_H;
