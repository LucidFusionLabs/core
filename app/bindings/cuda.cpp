/*
 * $Id: cuda.cpp 1335 2014-12-02 04:13:46Z justin $
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cuda_runtime.h>
#include "cuda/lfcuda.h"
#include "speech/hmm.h"
#include "speech/speech.h"

namespace LFL {
void PrintCUDAProperties(cudaDeviceProp *prop) {
  DEBUGf("Major revision number:         %d", prop->major);
  DEBUGf("Minor revision number:         %d", prop->minor);
  DEBUGf("Name:                          %s", prop->name);
  DEBUGf("Total global memory:           %u", prop->totalGlobalMem);
  DEBUGf("Total shared memory per block: %u", prop->sharedMemPerBlock);
  DEBUGf("Total registers per block:     %d", prop->regsPerBlock);
  DEBUGf("Warp size:                     %d", prop->warpSize);
  DEBUGf("Maximum memory pitch:          %u", prop->memPitch);
  DEBUGf("Maximum threads per block:     %d", prop->maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i) DEBUGf("Maximum dimension %d of block: %d", i, prop->maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i) DEBUGf("Maximum dimension %d of grid:  %d", i, prop->maxGridSize[i]);
  DEBUGf("Clock rate:                    %d", prop->clockRate);
  DEBUGf("Total constant memory:         %u", prop->totalConstMem);
  DEBUGf("Texture alignment:             %u", prop->textureAlignment);
  DEBUGf("Concurrent copy and execution: %s", (prop->deviceOverlap ? "Yes" : "No"));
  DEBUGf("Number of multiprocessors:     %d", prop->multiProcessorCount);
  DEBUGf("Kernel execution timeout:      %s", (prop->kernelExecTimeoutEnabled ? "Yes" : "No"));
}

int CUDA::Init() {
  INFO("CUDA::Init()");
  FLAGS_enable_cuda = 0;

  int cuda_devices = 0;
  cudaError_t err;
  if ((err = cudaGetDeviceCount(&cuda_devices)) != cudaSuccess)
  { ERROR("cudaGetDeviceCount error ", cudaGetErrorString(err)); return 0; }

  cudaDeviceProp prop;
  for (int i=0; i<cuda_devices; i++) {
    if ((err = cudaGetDeviceProperties(&prop, i)) != cudaSuccess) { ERROR("cudaGetDeviceProperties error ", err); return 0; }
    if (FLAGS_debug) PrintCUDAProperties(&prop);
    if (strstr(prop.name, "Emulation")) continue;
    FLAGS_enable_cuda=1;
  }

  if (FLAGS_enable_cuda) {
    INFO("CUDA device detected, enabling acceleration: enable_cuda(", FLAGS_enable_cuda, ") devices ", cuda_devices);
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
    cuda_init_hook();
  }
  else INFO("no CUDA devices detected ", cuda_devices);
  return 0;
}

}; // namespace LFL
