/*
 * $Id$
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

#include "app/app.h"
#include "app/math.h"
#include "app/speech.h"
#include "app/cuda/speech.h"

#define cudaMemcpyDeviceToHost 1
#define cudaMemcpyHostToDevice 1
#define __device__
#define __global__
#define __shared__

void __syncthreads() {}
int cudaMalloc(void **ptr, int size) { *ptr = malloc(size); return !*ptr; }
void *cudaMemcpy(void *dst, const void *src, int len, int flag) { return memcpy(dst,src,len); }

CudaAppInterface CAppI;
CudaAcousticModel *CAMinst;
CudaAcousticModel::CudaAcousticModel(int numstates, int k, int d) : map(0), states(numstates), K(k), D(d) {}
CudaAcousticModel *CudaAcousticModel::load(AcousticModel *model) { return 0; }

/*
 *
 * paste here
 *
 */

void cuda_init_hook() {}
void cuda_calc_emissions(double *emission, bool *active, HMM *model, double *observation) {}

Bind binds[] = { { 0, 0, 0, 0 } };
Asset assets[] = { { 0, 0, 0, 0, 0, 0, 0, 0, } };
SoundAsset soundassets[] = { { 0, 0, 0, 0, 0 } };
int frame(unsigned clicks, unsigned mic_samples, bool cam_sample, unsigned events, int flag) { return 0; }

HMM::State *test_AM_iter(AcousticModel *model, int *id) { return model->nextState(id); }
HMM::State *test_AM_state(AcousticModel *model, int id) { return model->state(id); }

#undef main
int main(int argc, char *argv[]) {

    CAppI.log = log;
    CAppI.map_set = IntMap::set;
    CAppI.map_get = IntMap::get;
    CAppI.AM_iter = test_AM_iter;
    CAppI.AM_state = test_AM_state;

    const char *modeldir = "/corpus/model/";
    AcousticModel *model = new AcousticModel(); int lastiter;
    if ((lastiter = model->read(modeldir))<0) { error("read %s %d", modeldir, lastiter); return -1; }
    info("loaded iter %d, %d models, %d states", lastiter, model->modelCount, model->stateCount);
    CAMinst = CudaAcousticModel::load(model);
    
    return 0;
}

