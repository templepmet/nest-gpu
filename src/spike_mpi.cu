/*
 *  This file is part of NESTGPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NESTGPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NESTGPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NESTGPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <fstream>
#include <list>
#include <string>
#include <vector>

#include "cuda_error.h"
#include "getRealTime.h"
#include "spike_buffer.h"

#include "connect_mpi.h"
#include "mode.h"
#include "non_blocking_timer.h"
#include "scan.h"
#include "spike_mpi.h"

__device__ int locate(int val, int *data, int n);

// Simple kernel for pushing remote spikes in local spike buffers
// Version with spike multiplicity array (spike_height)
__global__ void PushSpikeFromRemote(int n_spikes, int *spike_buffer_id,
                                    float *spike_height) {
    int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_spike < n_spikes) {
        int isb = spike_buffer_id[i_spike];
        float height = spike_height[i_spike];
        PushSpike(isb, height);
    }
}

// Simple kernel for pushing remote spikes in local spike buffers
// Version without spike multiplicity array (spike_height)
__global__ void PushSpikeFromRemote(int n_spikes, int *spike_buffer_id) {
    int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_spike < n_spikes) {
        int isb = spike_buffer_id[i_spike];
        PushSpike(isb, 1.0);
    }
}

// convert node group indexes to spike buffer indexes
// by adding the index of the first node of the node group
__global__ void AddOffset(int n_spikes, int *spike_buffer_id,
                          int i_remote_node_0) {
    int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_spike < n_spikes) {
        spike_buffer_id[i_spike] += i_remote_node_0;
    }
}

__constant__ bool NESTGPUMpiFlag;

#ifdef HAVE_MPI

__device__ int NExternalTargetHost;
__device__ int MaxSpikePerHost;

int *d_ExternalSpikeNum;
__device__ int *ExternalSpikeNum;

int *d_ExternalSpikeSourceNode;  // [MaxSpikeNum];
__device__ int *ExternalSpikeSourceNode;

float *d_ExternalSpikeHeight;  // [MaxSpikeNum];
__device__ float *ExternalSpikeHeight;

int *d_ExternalTargetSpikeNum;
__device__ int *ExternalTargetSpikeNum;

int *d_ExternalTargetSpikeNodeId;
__device__ int *ExternalTargetSpikeNodeId;

float *d_ExternalTargetSpikeHeight;
__device__ float *ExternalTargetSpikeHeight;

// for overlap
int *d_ExternalTargetSpikeMinDelay;
__device__ int *ExternalTargetSpikeMinDelay;

int *d_NExternalNodeTargetHost;
__device__ int *NExternalNodeTargetHost;

int **d_ExternalNodeTargetHostId;
__device__ int **ExternalNodeTargetHostId;

int **d_ExternalNodeId;
__device__ int **ExternalNodeId;

int **d_ExternalNodeMinDelay;
__device__ int **ExternalNodeMinDelay;

// int *d_ExternalSourceSpikeNum;
//__device__ int *ExternalSourceSpikeNum;

int *d_ExternalSourceSpikeNodeId;
__device__ int *ExternalSourceSpikeNodeId;

float *d_ExternalSourceSpikeHeight;
__device__ float *ExternalSourceSpikeHeight;

int *d_ExternalTargetSpikeCumul;
int *d_ExternalTargetSpikeNodeIdJoin;

int *h_ExternalTargetSpikeNum;
int *h_ExternalTargetSpikeCumul;
int *h_ExternalSourceSpikeNum;
int *h_ExternalTargetSpikeNodeId;
int *h_ExternalSourceSpikeNodeId;
int *h_ExternalTargetSpikeMinDelay;  // for overlap

// Alltoall
int *h_ExternalSourceSpikeNum_recvcounts;
int *h_ExternalSourceSpikeCumul_rdispls;
MPI_Request *send_mpi_request;
MPI_Status *send_mpi_status;
MPI_Request *recv_mpi_request;
MPI_Status *recv_mpi_status;

// Overlap
int tag_immed = 0;
int tag_delay = 1;
std::vector<std::vector<int>> h_ExternalTargetSpikeNodeId_immed;
std::vector<std::vector<int>> h_ExternalTargetSpikeNodeId_delay;
std::vector<std::vector<int>> h_ExternalSourceSpikeNodeId_immed;
std::vector<std::vector<int>> h_ExternalSourceSpikeNodeId_delay;
std::vector<int> h_ExternalTargetSpikeNodeId_immed_count;
std::vector<int> h_ExternalTargetSpikeNodeId_delay_count;
std::vector<int> h_ExternalSourceSpikeNodeId_immed_count;
std::vector<int> h_ExternalSourceSpikeNodeId_delay_count;
std::vector<MPI_Request> send_mpi_request_immed;
std::vector<MPI_Request> send_mpi_request_delay;
std::vector<MPI_Request> recv_mpi_request_immed;
std::vector<MPI_Request> recv_mpi_request_delay;
std::vector<MPI_Status> send_mpi_status_immed;
std::vector<MPI_Status> send_mpi_status_delay;
std::vector<MPI_Status> recv_mpi_status_immed;
std::vector<MPI_Status> recv_mpi_status_delay;

// int *h_ExternalSpikeNodeId;
//  float *h_ExternalSpikeHeight;

// Push in a dedicated array the spikes that must be sent externally
__device__ void PushExternalSpike(int i_source, float height) {
    int pos = atomicAdd(ExternalSpikeNum, 1);
    if (pos >= MaxSpikePerHost) {
        printf("Number of spikes larger than MaxSpikePerHost: %d\n",
               MaxSpikePerHost);
        *ExternalSpikeNum = MaxSpikePerHost;
        return;
    }
    ExternalSpikeSourceNode[pos] = i_source;
    ExternalSpikeHeight[pos] = height;
}

// Properly organize the spikes that must be sent externally
__global__ void SendExternalSpike() {
    int i_spike = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_spike < *ExternalSpikeNum) {
        int i_source = ExternalSpikeSourceNode[i_spike];
        float height = ExternalSpikeHeight[i_spike];
        int Nth = NExternalNodeTargetHost[i_source];

        for (int ith = 0; ith < Nth; ith++) {
            int target_host_id = ExternalNodeTargetHostId[i_source][ith];
            int remote_node_id = ExternalNodeId[i_source][ith];
            int min_delay = ExternalNodeMinDelay[i_source][ith];
            int pos = atomicAdd(&ExternalTargetSpikeNum[target_host_id], 1);
            ExternalTargetSpikeNodeId[target_host_id * MaxSpikePerHost + pos] =
                remote_node_id;
            ExternalTargetSpikeHeight[target_host_id * MaxSpikePerHost + pos] =
                height;
            ExternalTargetSpikeMinDelay[target_host_id * MaxSpikePerHost +
                                        pos] = min_delay;
        }
    }
}

// reset external spike counters
__global__ void ExternalSpikeReset() {
    *ExternalSpikeNum = 0;
    for (int ith = 0; ith < NExternalTargetHost; ith++) {
        ExternalTargetSpikeNum[ith] = 0;
    }
}

// initialize external spike arrays
int ConnectMpi::ExternalSpikeInit(int n_node, int n_hosts,
                                  int max_spike_per_host, int i_remote_node_0) {
    SendSpikeToRemote_MPI_time_ = 0;
    RecvSpikeFromRemote_MPI_time_ = 0;
    SendSpikeToRemote_CUDAcp_time_ = 0;
    RecvSpikeFromRemote_CUDAcp_time_ = 0;
    JoinSpike_time_ = 0;

    RecvSpikeWait_time = 0;  // comm_wait

    int *h_NExternalNodeTargetHost = new int[n_node];
    int **h_ExternalNodeTargetHostId = new int *[n_node];
    int **h_ExternalNodeId = new int *[n_node];
    int **h_ExternalNodeMinDelay = new int *[n_node];

    // h_ExternalSpikeNodeId = new int[max_spike_per_host];
    h_ExternalTargetSpikeNum = new int[n_hosts];
    h_ExternalTargetSpikeCumul = new int[n_hosts + 1];
    h_ExternalSourceSpikeNum = new int[n_hosts];
    h_ExternalTargetSpikeNodeId = new int[n_hosts * (max_spike_per_host + 1)];
    h_ExternalSourceSpikeNodeId = new int[n_hosts * (max_spike_per_host + 1)];
    h_ExternalTargetSpikeMinDelay =
        new int[n_hosts * (max_spike_per_host + 1)];  // for overlap

    // Alltoall
    h_ExternalSourceSpikeNum_recvcounts = new int[n_hosts];
    h_ExternalSourceSpikeCumul_rdispls = new int[n_hosts];
    for (int i = 0; i < n_hosts; ++i) {
        h_ExternalSourceSpikeNum_recvcounts[i] = max_spike_per_host;
        h_ExternalSourceSpikeCumul_rdispls[i] = max_spike_per_host * i;
    }
    h_ExternalSourceSpikeNum_recvcounts[mpi_id_] = 0;
    send_mpi_request = new MPI_Request[n_hosts];
    send_mpi_status = new MPI_Status[n_hosts];
    recv_mpi_request = new MPI_Request[n_hosts];
    recv_mpi_status = new MPI_Status[n_hosts];

    // Overlap
    if (isMode("comm_overlap")) {
        h_ExternalTargetSpikeNodeId_immed.resize(
            n_hosts, std::vector<int>(max_spike_per_host));
        h_ExternalTargetSpikeNodeId_delay.resize(
            n_hosts, std::vector<int>(max_spike_per_host));
        h_ExternalSourceSpikeNodeId_immed.resize(
            n_hosts, std::vector<int>(max_spike_per_host));
        h_ExternalSourceSpikeNodeId_delay.resize(
            n_hosts, std::vector<int>(max_spike_per_host));
        h_ExternalTargetSpikeNodeId_immed_count.resize(n_hosts, 0);
        h_ExternalTargetSpikeNodeId_delay_count.resize(n_hosts, 0);
        h_ExternalSourceSpikeNodeId_immed_count.resize(n_hosts, 0);
        h_ExternalSourceSpikeNodeId_delay_count.resize(n_hosts, 0);
        send_mpi_request_immed.resize(n_hosts);
        send_mpi_request_delay.resize(n_hosts);
        recv_mpi_request_immed.resize(n_hosts);
        recv_mpi_request_delay.resize(n_hosts);
        send_mpi_status_immed.resize(n_hosts);
        send_mpi_status_delay.resize(n_hosts);
        recv_mpi_status_immed.resize(n_hosts);
        recv_mpi_status_delay.resize(n_hosts);
    }

    // h_ExternalSpikeHeight = new float[max_spike_per_host];

    gpuErrchk(cudaMalloc(&d_ExternalSpikeNum, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_ExternalSpikeSourceNode,
                         max_spike_per_host * sizeof(int)));
    gpuErrchk(
        cudaMalloc(&d_ExternalSpikeHeight, max_spike_per_host * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_ExternalTargetSpikeNum, n_hosts * sizeof(int)));

    // printf("n_hosts, max_spike_per_host: %d %d\n", n_hosts,
    // max_spike_per_host);

    gpuErrchk(cudaMalloc(&d_ExternalTargetSpikeNodeId,
                         n_hosts * max_spike_per_host * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_ExternalTargetSpikeHeight,
                         n_hosts * max_spike_per_host * sizeof(float)));
    // gpuErrchk(cudaMalloc(&d_ExternalSourceSpikeNum, n_hosts*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_ExternalSourceSpikeNodeId,
                         n_hosts * max_spike_per_host * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_ExternalSourceSpikeHeight,
                         n_hosts * max_spike_per_host * sizeof(float)));
    gpuErrchk(
        cudaMalloc(&d_ExternalTargetSpikeMinDelay,
                   n_hosts * max_spike_per_host * sizeof(int)));  // for overlap

    gpuErrchk(cudaMalloc(&d_ExternalTargetSpikeNodeIdJoin,
                         n_hosts * max_spike_per_host * sizeof(int)));
    gpuErrchk(
        cudaMalloc(&d_ExternalTargetSpikeCumul, (n_hosts + 1) * sizeof(int)));

    gpuErrchk(cudaMalloc(&d_NExternalNodeTargetHost, n_node * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_ExternalNodeTargetHostId, n_node * sizeof(int *)));
    gpuErrchk(cudaMalloc(&d_ExternalNodeId, n_node * sizeof(int *)));
    gpuErrchk(cudaMalloc(&d_ExternalNodeMinDelay,
                         n_node * sizeof(int *)));  // for overlap

    if (isMode("comm_overlap")) {
        // delayも調整する
        ExchangeExternalMinDelay(n_node, n_hosts, i_remote_node_0);
    }

    for (int i_source = 0; i_source < n_node; i_source++) {
        std::vector<ExternalConnectionNode> *conn =
            &extern_connection_[i_source];
        int Nth = conn->size();
        h_NExternalNodeTargetHost[i_source] = Nth;
        if (Nth > 0) {
            gpuErrchk(cudaMalloc(&h_ExternalNodeTargetHostId[i_source],
                                 Nth * sizeof(int)));
            gpuErrchk(
                cudaMalloc(&h_ExternalNodeId[i_source], Nth * sizeof(int)));
            gpuErrchk(cudaMalloc(&h_ExternalNodeMinDelay[i_source],
                                 Nth * sizeof(int)));
            int *target_host_arr = new int[Nth];
            int *node_id_arr = new int[Nth];
            int *min_delay_arr = new int[Nth];
            for (int ith = 0; ith < Nth; ith++) {
                target_host_arr[ith] = conn->at(ith).target_host_id;
                node_id_arr[ith] = conn->at(ith).remote_node_id;
                min_delay_arr[ith] = conn->at(ith).min_delay;
            }
            cudaMemcpy(h_ExternalNodeTargetHostId[i_source], target_host_arr,
                       Nth * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(h_ExternalNodeId[i_source], node_id_arr,
                       Nth * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(h_ExternalNodeMinDelay[i_source], min_delay_arr,
                       Nth * sizeof(int), cudaMemcpyHostToDevice);
            delete[] target_host_arr;
            delete[] node_id_arr;
            delete[] min_delay_arr;
        }
    }

    cudaMemcpy(d_NExternalNodeTargetHost, h_NExternalNodeTargetHost,
               n_node * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ExternalNodeTargetHostId, h_ExternalNodeTargetHostId,
               n_node * sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ExternalNodeId, h_ExternalNodeId, n_node * sizeof(int *),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ExternalNodeMinDelay, h_ExternalNodeMinDelay,
               n_node * sizeof(int *), cudaMemcpyHostToDevice);

    DeviceExternalSpikeInit<<<1, 1>>>(
        n_hosts, max_spike_per_host, d_ExternalSpikeNum,
        d_ExternalSpikeSourceNode, d_ExternalSpikeHeight,
        d_ExternalTargetSpikeNum, d_ExternalTargetSpikeNodeId,
        d_ExternalTargetSpikeHeight, d_ExternalTargetSpikeMinDelay,
        d_NExternalNodeTargetHost, d_ExternalNodeTargetHostId, d_ExternalNodeId,
        d_ExternalNodeMinDelay);
    delete[] h_NExternalNodeTargetHost;
    delete[] h_ExternalNodeTargetHostId;
    delete[] h_ExternalNodeId;
    delete[] h_ExternalNodeMinDelay;

    return 0;
}

// exchange external_node min_delay in all processes
int ConnectMpi::ExchangeExternalMinDelay(int n_node, int n_hosts,
                                         int i_remote_node_0) {
    std::vector<int> n_ex_node_per_host(n_hosts, 0);
    std::vector<int> n_ex_node_per_host_cumul(n_hosts + 1, 0);
    for (int i = 0; i < n_node; ++i) {
        std::vector<ExternalConnectionNode> *conn = &extern_connection_[i];
        for (int j = 0; j < conn->size(); ++j) {
            n_ex_node_per_host[conn->at(j).target_host_id]++;
        }
    }
    for (int i = 0; i < n_hosts; ++i) {
        n_ex_node_per_host_cumul[i + 1] =
            n_ex_node_per_host_cumul[i] + n_ex_node_per_host[i];
    }
    int n_ex_node = n_ex_node_per_host_cumul[n_hosts];
    std::vector<int> idx_per_host = n_ex_node_per_host_cumul;
    std::vector<int> remote_node_id_sendbuf(n_ex_node + 1);
    for (int i = 0; i < n_node; ++i) {
        std::vector<ExternalConnectionNode> *conn = &extern_connection_[i];
        for (int j = 0; j < conn->size(); ++j) {
            int idx = idx_per_host[conn->at(j).target_host_id]++;
            remote_node_id_sendbuf[idx] = conn->at(j).remote_node_id;
        }
    }
    std::vector<int> n_ex_node_per_host_recv(n_hosts, 0);
    std::vector<int> n_ex_node_per_host_cumul_recv(n_hosts + 1, 0);
    MPI_Alltoall(&n_ex_node_per_host[0], 1, MPI_INT,
                 &n_ex_node_per_host_recv[0], 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < n_hosts; ++i) {
        n_ex_node_per_host_cumul_recv[i + 1] =
            n_ex_node_per_host_cumul_recv[i] + n_ex_node_per_host_recv[i];
    }
    int n_ex_node_recv = n_ex_node_per_host_cumul_recv[n_hosts];
    std::vector<int> remote_node_id_recvbuf(n_ex_node_recv + 1);
    MPI_Alltoallv(&remote_node_id_sendbuf[0], &n_ex_node_per_host[0],
                  &n_ex_node_per_host_cumul[0], MPI_INT,
                  &remote_node_id_recvbuf[0], &n_ex_node_per_host_recv[0],
                  &n_ex_node_per_host_cumul_recv[0], MPI_INT, MPI_COMM_WORLD);

    std::vector<int> remote_node_mindelay_sendbuf(n_ex_node_recv + 1);
    std::vector<int> remote_node_mindelay_recvbuf(n_ex_node + 1);
    for (int i = 0; i < n_ex_node_recv; ++i) {
        int i_node = remote_node_id_recvbuf[i];
        int i_source = i_node + i_remote_node_0;
        int min_delay = 12345;
        for (ConnGroup &cg : net_connection_->connection_[i_source]) {
            int delay = cg.delay;
            min_delay = std::min(min_delay, delay);
        }
        assert(min_delay < 12345);
        remote_node_mindelay_sendbuf[i] = min_delay;
        if (min_delay > 0) {
            // 重複しない？：RemoteConnectionごとにRemoteNodeが作られるので重複しないはず
            for (ConnGroup &cg : net_connection_->connection_[i_source]) {
                cg.delay -= 1;
            }
        }
    }
    MPI_Alltoallv(&remote_node_mindelay_sendbuf[0], &n_ex_node_per_host_recv[0],
                  &n_ex_node_per_host_cumul_recv[0], MPI_INT,
                  &remote_node_mindelay_recvbuf[0], &n_ex_node_per_host[0],
                  &n_ex_node_per_host_cumul[0], MPI_INT, MPI_COMM_WORLD);

    idx_per_host = n_ex_node_per_host_cumul;  // reset
    for (int i = 0; i < n_node; ++i) {
        std::vector<ExternalConnectionNode> *conn = &extern_connection_[i];
        for (int j = 0; j < conn->size(); ++j) {
            int idx = idx_per_host[conn->at(j).target_host_id]++;
            conn->at(j).min_delay = remote_node_mindelay_recvbuf[idx];
        }
    }

    return 0;
}

// initialize external spike array pointers in the GPU
__global__ void DeviceExternalSpikeInit(
    int n_hosts, int max_spike_per_host, int *ext_spike_num,
    int *ext_spike_source_node, float *ext_spike_height,
    int *ext_target_spike_num, int *ext_target_spike_node_id,
    float *ext_target_spike_height, int *ext_target_spike_delay,
    int *n_ext_node_target_host, int **ext_node_target_host_id,
    int **ext_node_id, int **ext_node_delay)

{
    NExternalTargetHost = n_hosts;
    MaxSpikePerHost = max_spike_per_host;
    ExternalSpikeNum = ext_spike_num;
    ExternalSpikeSourceNode = ext_spike_source_node;
    ExternalSpikeHeight = ext_spike_height;
    ExternalTargetSpikeNum = ext_target_spike_num;
    ExternalTargetSpikeNodeId = ext_target_spike_node_id;
    ExternalTargetSpikeHeight = ext_target_spike_height;
    ExternalTargetSpikeMinDelay = ext_target_spike_delay;
    NExternalNodeTargetHost = n_ext_node_target_host;
    ExternalNodeTargetHostId = ext_node_target_host_id;
    ExternalNodeId = ext_node_id;
    ExternalNodeMinDelay = ext_node_delay;
    *ExternalSpikeNum = 0;
    for (int ith = 0; ith < NExternalTargetHost; ith++) {
        ExternalTargetSpikeNum[ith] = 0;
    }
}

// Send spikes to remote MPI processes
int ConnectMpi::SendSpikeToRemote(int n_hosts, int max_spike_per_host) {
    Other_timer->startRecord();

    int tag = 1;

    gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNum, d_ExternalTargetSpikeNum,
                         n_hosts * sizeof(int), cudaMemcpyDeviceToHost));

    Other_timer->stopRecord();
    PackSendSpike_timer->startRecord();

    int n_spike_tot = JoinSpikes(n_hosts, max_spike_per_host);

    gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNodeId,
                         d_ExternalTargetSpikeNodeIdJoin,
                         n_spike_tot * sizeof(int), cudaMemcpyDeviceToHost));
    PackSendSpike_timer->stopRecord();

    SendRecvSpikeRemote_immed_timer->startRecordHost();
    for (int ih = 0; ih < n_hosts; ih++) {
        int array_idx = h_ExternalTargetSpikeCumul[ih];
        int n_spikes = h_ExternalTargetSpikeCumul[ih + 1] - array_idx;
        MPI_Isend(&h_ExternalTargetSpikeNodeId[array_idx], n_spikes, MPI_INT,
                  ih, tag, MPI_COMM_WORLD, &send_mpi_request[ih]);
    }
    // MPI_Waitall(n_hosts, send_mpi_request, send_mpi_status);
    SendRecvSpikeRemote_immed_timer->stopRecordHost();

    if (isMode("dump_comm_dist")) {
        std::string filename = "comm/send_" + std::to_string(mpi_id_) + ".txt";
        std::ofstream ofs(filename, std::ios::app);
        for (int i = 0; i < n_hosts; ++i) {
            if (i > 0) ofs << ",";
            int spikes = h_ExternalTargetSpikeCumul[i + 1] -
                         h_ExternalTargetSpikeCumul[i];
            ofs << spikes;
        }
        ofs << std::endl;
        ofs.close();
    }

    return 0;
}

// Receive spikes from remote MPI processes
int ConnectMpi::RecvSpikeFromRemote(int n_hosts, int max_spike_per_host) {
    int tag = 1, count;

    SendRecvSpikeRemote_immed_timer->startRecordHost();
    for (int i = 0; i < n_hosts; i++) {
        MPI_Irecv(h_ExternalSourceSpikeNodeId + (i * max_spike_per_host),
                  max_spike_per_host, MPI_INT, i, tag, MPI_COMM_WORLD,
                  &recv_mpi_request[i]);
    }
    MPI_Waitall(n_hosts, recv_mpi_request, recv_mpi_status);
    for (int i = 0; i < n_hosts; ++i) {
        MPI_Get_count(&recv_mpi_status[i], MPI_INT, &count);
        h_ExternalSourceSpikeNum[i] = count;
    }
    h_ExternalSourceSpikeNum[mpi_id_] = 0;
    SendRecvSpikeRemote_immed_timer->stopRecordHost();

    if (isMode("dump_comm_dist")) {
        std::string filename = "comm/recv_" + std::to_string(mpi_id_) + ".txt";
        std::ofstream ofs(filename, std::ios::app);
        for (int i = 0; i < n_hosts; ++i) {
            if (i > 0) ofs << ",";
            int spikes = h_ExternalSourceSpikeNum[i];
            ofs << spikes;
        }
        ofs << std::endl;
        ofs.close();
    }

    return 0;
}

// Send Receive spikes from remote MPI processes
int ConnectMpi::SendRecvSpikeRemoteOverlap(int n_hosts, int max_spike_per_host,
                                           long long it_, long long Nt_) {
    Other_timer->startRecord();

    gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNum, d_ExternalTargetSpikeNum,
                         n_hosts * sizeof(int), cudaMemcpyDeviceToHost));

    Other_timer->stopRecord();
    PackSendSpike_timer->startRecord();

    int n_spike_tot = JoinSpikes(n_hosts, max_spike_per_host);

    gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNodeId,
                         d_ExternalTargetSpikeNodeIdJoin,
                         n_spike_tot * sizeof(int), cudaMemcpyDeviceToHost));

    // ここから
    gpuErrchk(cudaMemcpy(
        h_ExternalTargetSpikeMinDelay, d_ExternalTargetSpikeMinDelay,
        (n_hosts * max_spike_per_host) * sizeof(int), cudaMemcpyDeviceToHost));
    std::fill(h_ExternalTargetSpikeNodeId_immed_count.begin(),
              h_ExternalTargetSpikeNodeId_immed_count.end(), 0);
    std::fill(h_ExternalTargetSpikeNodeId_delay_count.begin(),
              h_ExternalTargetSpikeNodeId_delay_count.end(), 0);
    for (int i = 0; i < n_hosts; ++i) {
        int idx = h_ExternalTargetSpikeCumul[i];
        int n_spikes = h_ExternalTargetSpikeCumul[i + 1] - idx;
        for (int j = 0; j < n_spikes; ++j) {
            if (h_ExternalTargetSpikeMinDelay[i * max_spike_per_host + j] ==
                0) {
                h_ExternalTargetSpikeNodeId_immed
                    [i][h_ExternalTargetSpikeNodeId_immed_count[i]++] =
                        h_ExternalTargetSpikeNodeId[idx + j];
            } else {
                h_ExternalTargetSpikeNodeId_delay
                    [i][h_ExternalTargetSpikeNodeId_delay_count[i]++] =
                        h_ExternalTargetSpikeNodeId[idx + j];
            }
        }
    }
    // ここをGPUで処理
    PackSendSpike_timer->stopRecord();

    SendRecvSpikeRemote_immed_timer->startRecordHost();
    // immed
    for (int i = 0; i < n_hosts; ++i) {
        MPI_Isend(h_ExternalTargetSpikeNodeId_immed[i].data(),
                  h_ExternalTargetSpikeNodeId_immed_count[i], MPI_INT, i,
                  tag_immed, MPI_COMM_WORLD, &send_mpi_request_immed[i]);
    }
    for (int i = 0; i < n_hosts; ++i) {
        MPI_Irecv(h_ExternalSourceSpikeNodeId_immed[i].data(),
                  max_spike_per_host, MPI_INT, i, tag_immed, MPI_COMM_WORLD,
                  &recv_mpi_request_immed[i]);
    }
    MPI_Waitall(n_hosts, &send_mpi_request_immed[0], &send_mpi_status_immed[0]);
    MPI_Waitall(n_hosts, &recv_mpi_request_immed[0], &recv_mpi_status_immed[0]);
    for (int i = 0; i < n_hosts; i++) {
        MPI_Get_count(&recv_mpi_status_immed[i], MPI_INT,
                      &h_ExternalSourceSpikeNodeId_immed_count[i]);
    }
    SendRecvSpikeRemote_immed_timer->stopRecordHost();

    // delay
    if (it_ > 0) {  // overlap
        SendRecvSpikeRemote_delay_timer->startRecordHost();
        MPI_Waitall(n_hosts, &send_mpi_request_delay[0],
                    &send_mpi_status_delay[0]);
        MPI_Waitall(n_hosts, &recv_mpi_request_delay[0],
                    &recv_mpi_status_delay[0]);
        for (int i = 0; i < n_hosts; i++) {
            MPI_Get_count(&recv_mpi_status_delay[i], MPI_INT,
                          &h_ExternalSourceSpikeNodeId_delay_count[i]);
        }
        SendRecvSpikeRemote_delay_timer->stopRecordHost();
    }

    UnpackRecvSpike_timer->startRecordHost();
    for (int i = 0; i < n_hosts; i++) {
        int count_immed = h_ExternalSourceSpikeNodeId_immed_count[i];
        int count_delay = h_ExternalSourceSpikeNodeId_delay_count[i];
        std::copy(&h_ExternalSourceSpikeNodeId_immed[i][0],
                  &h_ExternalSourceSpikeNodeId_immed[i][count_immed],
                  h_ExternalSourceSpikeNodeId + (i * max_spike_per_host));
        std::copy(&h_ExternalSourceSpikeNodeId_delay[i][0],
                  &h_ExternalSourceSpikeNodeId_delay[i][count_delay],
                  h_ExternalSourceSpikeNodeId + (i * max_spike_per_host) +
                      count_immed);
        h_ExternalSourceSpikeNum[i] = count_immed + count_delay;
    }
    h_ExternalSourceSpikeNum[mpi_id_] = 0;
    UnpackRecvSpike_timer->stopRecordHost();

    SendRecvSpikeRemote_delay_timer->startRecordHost();
    for (int i = 0; i < n_hosts; ++i) {
        MPI_Isend(h_ExternalTargetSpikeNodeId_delay[i].data(),
                  h_ExternalTargetSpikeNodeId_delay_count[i], MPI_INT, i,
                  tag_delay, MPI_COMM_WORLD, &send_mpi_request_delay[i]);
    }
    for (int i = 0; i < n_hosts; ++i) {
        MPI_Irecv(h_ExternalSourceSpikeNodeId_delay[i].data(),
                  max_spike_per_host, MPI_INT, i, tag_delay, MPI_COMM_WORLD,
                  &recv_mpi_request_delay[i]);
    }
    if (it_ == Nt_ - 1) {  // final receive
        MPI_Waitall(n_hosts, &send_mpi_request_delay[0],
                    &send_mpi_status_delay[0]);
        MPI_Waitall(n_hosts, &recv_mpi_request_delay[0],
                    &recv_mpi_status_delay[0]);
        for (int i = 0; i < n_hosts; i++) {
            MPI_Get_count(&recv_mpi_status_delay[i], MPI_INT,
                          &h_ExternalSourceSpikeNodeId_delay_count[i]);
        }
    }
    SendRecvSpikeRemote_delay_timer->stopRecordHost();

    return 0;
}

// Send spikes to remote MPI processes
int ConnectMpi::SendSpikeToRemoteCuda(int n_hosts, int max_spike_per_host) {
    Other_timer->startRecord();

    int tag = 1;

    gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNum, d_ExternalTargetSpikeNum,
                         n_hosts * sizeof(int),
                         cudaMemcpyDeviceToHost));  // necessary for send counts

    Other_timer->stopRecord();
    SendSpikeToRemote_timer->startRecord();

    for (int i = 0; i < n_hosts; i++) {
        MPI_Isend(d_ExternalTargetSpikeNodeId + (i * max_spike_per_host),
                  h_ExternalTargetSpikeNum[i], MPI_INT, i, tag, MPI_COMM_WORLD,
                  &send_mpi_request[i]);
    }
    MPI_Waitall(n_hosts, send_mpi_request, send_mpi_status);

    SendSpikeToRemote_timer->stopRecord();

    return 0;
}

// Receive spikes from remote MPI processes
int ConnectMpi::RecvSpikeFromRemoteCuda(int n_hosts, int max_spike_per_host) {
    int tag = 1, count;

    for (int i = 0; i < n_hosts; i++) {
        MPI_Irecv(d_ExternalSourceSpikeNodeId + (i * max_spike_per_host),
                  max_spike_per_host, MPI_INT, i, tag, MPI_COMM_WORLD,
                  &recv_mpi_request[i]);
    }
    MPI_Waitall(n_hosts, recv_mpi_request, recv_mpi_status);

    for (int i = 0; i < n_hosts; ++i) {
        MPI_Get_count(&recv_mpi_status[i], MPI_INT, &count);
        h_ExternalSourceSpikeNum[i] = count;
    }
    h_ExternalSourceSpikeNum[mpi_id_] = 0;

    return 0;
}

// AlltoAll spikes for remote MPI processes
int ConnectMpi::AlltoallvSpikeforRemote(int n_hosts, int max_spike_per_host) {
    Other_timer->startRecord();
    gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNum, d_ExternalTargetSpikeNum,
                         n_hosts * sizeof(int), cudaMemcpyDeviceToHost));
    Other_timer->stopRecord();

    SendSpikeToRemote_timer->startRecord();
    // pack the spikes in GPU memory and copy them to CPU
    int n_spike_tot = JoinSpikes(n_hosts, max_spike_per_host);
    // copy spikes from GPU to CPU memory
    gpuErrchk(cudaMemcpy(h_ExternalTargetSpikeNodeId,
                         d_ExternalTargetSpikeNodeIdJoin,
                         n_spike_tot * sizeof(int), cudaMemcpyDeviceToHost));
    SendSpikeToRemote_timer->stopRecord();

    RecvSpikeFromRemote_timer->startRecordHost();
    for (int i = n_hosts - 1; i >= 0; --i) {
        for (int j = h_ExternalTargetSpikeCumul[i + 1] - 1;
             j >= h_ExternalTargetSpikeCumul[i]; --j) {
            h_ExternalTargetSpikeNodeId[j + i + 1] =
                h_ExternalTargetSpikeNodeId[j];
        }
        h_ExternalTargetSpikeNodeId[h_ExternalTargetSpikeCumul[i] + i] =
            h_ExternalTargetSpikeNum[i];
    }
    for (int i = 0; i < n_hosts; ++i) {
        h_ExternalTargetSpikeNum[i]++;
        h_ExternalTargetSpikeCumul[i] += i;
    }
    h_ExternalSourceSpikeNum_recvcounts[mpi_id_] = 1;
    // only alltoallv
    MPI_Alltoallv(h_ExternalTargetSpikeNodeId, h_ExternalTargetSpikeNum,
                  h_ExternalTargetSpikeCumul, MPI_INT,
                  h_ExternalSourceSpikeNodeId,
                  h_ExternalSourceSpikeNum_recvcounts,
                  h_ExternalSourceSpikeCumul_rdispls, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < n_hosts; ++i) {
        h_ExternalSourceSpikeNum[i] =
            h_ExternalSourceSpikeNodeId[i * max_spike_per_host];
        for (int j = 0; j < h_ExternalSourceSpikeNum[i]; ++j) {
            h_ExternalSourceSpikeNodeId[i * max_spike_per_host + j] =
                h_ExternalSourceSpikeNodeId[i * max_spike_per_host + j + 1];
        }
    }

    // alltoall + alltoallv
    // MPI_Alltoall(h_ExternalTargetSpikeNum, 1, MPI_INT,
    //             h_ExternalSourceSpikeNum, 1, MPI_INT, MPI_COMM_WORLD);
    // MPI_Alltoallv(h_ExternalTargetSpikeNodeId, h_ExternalTargetSpikeNum,
    // h_ExternalTargetSpikeCumul, MPI_INT,
    //   h_ExternalSourceSpikeNodeId, h_ExternalSourceSpikeNum,
    //   h_ExternalSourceSpikeCumul_rdispls, MPI_INT, MPI_COMM_WORLD);
    RecvSpikeFromRemote_timer->stopRecordHost();

    if (isMode("dump_comm_dist")) {
        {
            std::string filename =
                "comm/send_" + std::to_string(mpi_id_) + ".txt";
            std::ofstream ofs(filename, std::ios::app);
            for (int i = 0; i < n_hosts; ++i) {
                if (i > 0) ofs << ",";
                int spikes = h_ExternalTargetSpikeNum[i];
                ofs << spikes;
            }
            ofs << std::endl;
            ofs.close();
        }
        {
            std::string filename =
                "comm/recv_" + std::to_string(mpi_id_) + ".txt";
            std::ofstream ofs(filename, std::ios::app);
            for (int i = 0; i < n_hosts; ++i) {
                if (i > 0) ofs << ",";
                int spikes = h_ExternalSourceSpikeNum[i];
                ofs << spikes;
            }
            ofs << std::endl;
            ofs.close();
        }
    }

    return 0;
}

// pack spikes received from remote MPI processes
// and copy them to GPU memory
int ConnectMpi::CopySpikeFromRemote(int n_hosts, int max_spike_per_host,
                                    int i_remote_node_0) {
    double time_mark = getRealTime();
    int n_spike_tot = 0;
    // loop on MPI proc
    for (int i_host = 0; i_host < n_hosts; i_host++) {
        int n_spike = h_ExternalSourceSpikeNum[i_host];
        for (int i_spike = 0; i_spike < n_spike; i_spike++) {
            // pack spikes received from remote MPI processes
            h_ExternalSourceSpikeNodeId[n_spike_tot] =
                h_ExternalSourceSpikeNodeId[i_host * max_spike_per_host +
                                            i_spike];
            n_spike_tot++;
        }
    }
    JoinSpike_time_ += (getRealTime() - time_mark);

    if (n_spike_tot > 0) {
        time_mark = getRealTime();
        // Memcopy will be synchronized with AddOffset kernel
        // copy to GPU memory packed spikes from remote MPI proc
        gpuErrchk(cudaMemcpyAsync(
            d_ExternalSourceSpikeNodeId, h_ExternalSourceSpikeNodeId,
            n_spike_tot * sizeof(int), cudaMemcpyHostToDevice));
        RecvSpikeFromRemote_CUDAcp_time_ += (getRealTime() - time_mark);
        // convert node group indexes to spike buffer indexes
        // by adding the index of the first node of the node group
        AddOffset<<<(n_spike_tot + 1023) / 1024, 1024>>>(
            n_spike_tot, d_ExternalSourceSpikeNodeId, i_remote_node_0);
        // push remote spikes in local spike buffers
        PushSpikeFromRemote<<<(n_spike_tot + 1023) / 1024, 1024>>>(
            n_spike_tot, d_ExternalSourceSpikeNodeId);
    }
    gpuErrchk(cudaPeekAtLastError());

    if (isMode("dump_syndelay")) {
        int mpi_id;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);

        std::vector<long long> delay_hist;
        for (int i = 0; i < n_spike_tot; ++i) {
            int i_node = h_ExternalSourceSpikeNodeId[i];
            int i_source = i_node + i_remote_node_0;
            for (ConnGroup &cg : net_connection_->connection_[i_source]) {
                int delay = cg.delay;
                if ((int)delay_hist.size() < delay + 1) {
                    delay_hist.resize(delay + 1, 0);
                }
                delay_hist[delay] += cg.target_vect.size();
            }
        }
        std::string filename =
            "syndelay/spike_" + std::to_string(mpi_id) + ".txt";
        std::ofstream ofs(filename, std::ios::app);
        if ((int)delay_hist.size() > 0) {
            for (int i = 0; i < (int)delay_hist.size(); ++i) {
                if (i > 0) ofs << ",";
                ofs << delay_hist[i];
            }
        } else {
            ofs << 0;
        }
        ofs << std::endl;
        ofs.close();
    }

    return n_spike_tot;
}

// pack spikes received from remote MPI processes
// and copy them to GPU memory
int ConnectMpi::CopySpikeFromRemoteCuda(int n_hosts, int max_spike_per_host,
                                        int i_remote_node_0) {
    int n_spike_tot = 0;
    for (int i_host = 0; i_host < n_hosts; i_host++) {
        n_spike_tot += h_ExternalSourceSpikeNum[i_host];
    }
    // bug: necessary packing head

    printf("rank %d: n_spike_tot=%d\n", mpi_id_, n_spike_tot);

    if (n_spike_tot > 0) {
        AddOffset<<<(n_spike_tot + 1023) / 1024, 1024>>>(
            n_spike_tot, d_ExternalSourceSpikeNodeId, i_remote_node_0);
        PushSpikeFromRemote<<<(n_spike_tot + 1023) / 1024, 1024>>>(
            n_spike_tot, d_ExternalSourceSpikeNodeId);
    }

    return n_spike_tot;
}

// pack the spikes in GPU memory that must be sent externally
__global__ void JoinSpikeKernel(int n_hosts, int *ExternalTargetSpikeCumul,
                                int *ExternalTargetSpikeNodeId,
                                int *ExternalTargetSpikeNodeIdJoin,
                                int n_spike_tot, int max_spike_per_host) {
    // parallel implementation of nested loop
    // outer loop index i_host = 0, ... , n_hosts
    // inner loop index i_spike = 0, ... , ExternalTargetSpikeNum[i_host];
    // array_idx is the index in the packed spike array
    int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (array_idx < n_spike_tot) {
        int i_host = locate(array_idx, ExternalTargetSpikeCumul, n_hosts + 1);
        while ((i_host < n_hosts) && (ExternalTargetSpikeCumul[i_host + 1] ==
                                      ExternalTargetSpikeCumul[i_host])) {
            i_host++;
            if (i_host == n_hosts) return;
        }
        int i_spike = array_idx - ExternalTargetSpikeCumul[i_host];
        // packed spike array
        ExternalTargetSpikeNodeIdJoin[array_idx] =
            ExternalTargetSpikeNodeId[i_host * max_spike_per_host + i_spike];
    }
}

// pack the spikes in GPU memory that must be sent externally
// and copy them to CPU memory
int ConnectMpi::JoinSpikes(int n_hosts, int max_spike_per_host) {
    double time_mark = getRealTime();
    // the index in the packed array can be computed from the MPI proc index
    // and from the spike index using  a cumulative sum (prefix scan)
    // of the number of spikes per MPI proc
    // the cumulative sum is done both in CPU and in GPU
    prefix_scan(d_ExternalTargetSpikeCumul, d_ExternalTargetSpikeNum,
                n_hosts + 1, true);
    h_ExternalTargetSpikeCumul[0] = 0;
    for (int ih = 0; ih < n_hosts; ih++) {
        h_ExternalTargetSpikeCumul[ih + 1] =
            h_ExternalTargetSpikeCumul[ih] + h_ExternalTargetSpikeNum[ih];
    }
    int n_spike_tot = h_ExternalTargetSpikeCumul[n_hosts];

    if (n_spike_tot > 0) {
        // pack the spikes in GPU memory
        JoinSpikeKernel<<<(n_spike_tot + 1023) / 1024, 1024>>>(
            n_hosts, d_ExternalTargetSpikeCumul, d_ExternalTargetSpikeNodeId,
            d_ExternalTargetSpikeNodeIdJoin, n_spike_tot, max_spike_per_host);

        gpuErrchk(cudaPeekAtLastError());
    }

    JoinSpike_time_ += (getRealTime() - time_mark);

    return n_spike_tot;
}

#endif
