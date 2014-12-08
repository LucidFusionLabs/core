/*
 * $Id: lfcuda.cu 501 2012-03-23 07:40:54Z justin $
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

#include <string>
#include <vector>
using namespace std;

#define Now() 0
#ifdef _WIN32
#define INFINITY HUGE_VAL
typedef void* HANDLE;
#endif

typedef          char  int8_t;
typedef unsigned char uint8_t;

typedef          long  int32_t;
typedef unsigned long uint32_t;

typedef          long long  int64_t;
typedef unsigned long long uint64_t;

typedef long long Time;
typedef void* Mutex;
struct Iter;
int sprint(char *out, int len, const char *fmt, ...);
char *sprintmp(const char *fmt, ...);
int doubleSort(double a, double b);
int doubleSort (const void *a, const void *b);
unsigned fnv32(const void *buf, unsigned len=0, unsigned hval=0);
int next_multiple_of_two(int input, int align);

/* *** LFL LOG IMPORT *** */
extern "C" void lfapp_log(int level, const char *file, int line, const char *fmt, ...);
struct LogLevel { enum { Fatal=-1, Error=0, Info=3, Debug=7 }; int l; };
#define  info(fmt,...) lfapp_log(LogLevel::Info,  __FILE__, __LINE__, fmt, __VA_ARGS__)
#define debug(fmt,...) lfapp_log(LogLevel::Debug, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define error(fmt,...) lfapp_log(LogLevel::Error, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define   err(fmt,...) lfapp_log(LogLevel::Error, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define fatal(fmt,...) lfapp_log(LogLevel::Fatal, __FILE__, __LINE__, fmt, __VA_ARGS__)
/* *** LFL LOG IMPORT *** */

#define LFL_FAKE_STL
#include "lfapp/lftypes.h"
#include "lfapp/math.h"
#include "lfapp/hmm.h"
#include "lfapp/audio.h"
#include "lfapp/speech.h"
#include "lfcuda.h"

void cuda_init_hook() { info("%s", "cuda init"); }

void cuda_error_check(const char *file, int line, const char *message) {
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) return;
    error("cuda error: %s: %s\n", message, cudaGetErrorString(err));
}

void cuda_malloc(const char *file, int line, void **out, int size) { cudaMalloc(out, size); cuda_error_check(file, line, "malloc"); }
void cuda_memcpy(const char *file, int line, void *out, const void *in, int len, cudaMemcpyKind flag) { cudaMemcpy(out, in, len, flag); cuda_error_check(file, line, "memcpy"); }
#define cuda_malloc(out, size) cuda_malloc(__FILE__, __LINE__, out, size)
#define cuda_memcpy(out, in, len, flag) cuda_memcpy(__FILE__, __LINE__, out, in, len, flag)
#define cuda_error_check(message) cuda_error_check(__FILE__, __LINE__, message)

int cuda_double_import(const double *in, int count, fpt *out) {
    fpt convbuf[16384]; int len=count*sizeof(fpt);
    for (int i=0; i<count; i++) convbuf[i] = in[i];	
    cuda_memcpy(out, convbuf, len, cudaMemcpyHostToDevice);
    return len;
}

int cuda_Matrix_import(Matrix *in, fpt *out, bool transpose) {
    fpt convbuf[16384]; int len=in->M*in->N*sizeof(fpt);
    if (transpose) in = in->transpose();
    MatrixIter(in) convbuf[i*in->N+j] = in->m[i*in->N+j];	
    cuda_memcpy(out, convbuf, len, cudaMemcpyHostToDevice);
    if (transpose) delete in;
    return len;
}

__device__ fpt cuda_logadd(fpt x, fpt y) {
    if (x<y) return y + log1p(exp(x - y));
    else     return x + log1p(exp(y - x));
}

__global__ void cuda_gmmPdfEval(CudaAcousticModel cam) {
    int ind=cam.beam[blockIdx.x], kind=ind*cam.K, kdind=ind*cam.K*cam.D, ki=threadIdx.x;
    __shared__ fpt sum[CudaAcousticModel::MaxK];
    fpt dist2=0, prob;

    for (int i=0; i<cam.D; i++) {
        int kdi = i*cam.K + ki;
        fpt diff = cam.mean[kdind+kdi] - cam.obv[i];
        fpt p = diff*diff;
        p /= cam.covar[kdind+kdi];
        dist2 += p;
    }

    prob = cam.norm[kind+ki] - 0.5 * dist2 + cam.prior[kind+ki];
    sum[ki] = prob;
    __syncthreads();

    for (unsigned s=cam.K/2; s>0; s>>=1) {
        if (ki < s) sum[ki] = cuda_logadd(sum[ki], sum[ki+s]);
        __syncthreads();
    }

    cam.posterior[blockIdx.x*cam.K + ki] = prob;

    if (!ki) cam.result[blockIdx.x] = sum[0];
}

CudaAcousticModel::CudaAcousticModel(int numstates, int k, int d) : states(numstates), K(k), D(d) {
    kelsize = K*sizeof(fpt);
    kdelsize = kelsize*D;
    ksize = states*kelsize;
    kdsize = states*kdelsize;

    cuda_malloc((void**)&mean, kdsize);
    cuda_malloc((void**)&covar, kdsize);
    cuda_malloc((void**)&prior, ksize);
    cuda_malloc((void**)&norm, ksize);

    cuda_malloc((void**)&obv, MaxObvLen*D*sizeof(fpt));	
    cuda_malloc((void**)&beam, MaxBeamWidth*sizeof(int));
    cuda_malloc((void**)&result, MaxBeamWidth*sizeof(fpt));
    cuda_malloc((void**)&posterior, MaxBeamWidth*K*sizeof(fpt));

    resout = (fpt*)malloc(MaxBeamWidth*sizeof(fpt));
    postout = (fpt*)malloc(MaxBeamWidth*K*sizeof(fpt));
}

int CudaAcousticModel::load(CudaAcousticModel *out, AcousticModel::Compiled *in) {
    int K=out->K, D=out->D, ind=0;

    AcousticModel::StateCollection::Iterator iter;
    for (in->beginState(&iter); !iter.done; in->nextState(&iter), ind++) {
        AcousticModel::State *s = iter.v;

        if (ind >= out->states) return 0;
        if (cuda_Matrix_import(&s->emission.mean, out->mean+ind*K*D, 1) != out->kdelsize) return 0;
        if (cuda_Matrix_import(&s->emission.diagcov, out->covar+ind*K*D, 1) != out->kdelsize) return 0;
        if (cuda_Matrix_import(&s->emission.prior, out->prior+ind*K, 1) != out->kelsize) return 0;
        if (cuda_Matrix_import(&s->emission.norm, out->norm+ind*K, 1) != out->kelsize) return 0;
    }
    return 0;
}

int CudaAcousticModel::calcEmissions(CudaAcousticModel *CAM, const double *observation, const int *beam, int framebeamwidth, double *emissionOut, double *posteriorOut) {
    if (!CAM) { error("no CUDA Acoustic Model %p", CAM); return -1; }
    if (!framebeamwidth) { error("CUDA calc emissions called with %d beamwidth", framebeamwidth); return -1; }

    cuda_memcpy(CAM->beam, beam, framebeamwidth*sizeof(int), cudaMemcpyHostToDevice);
    cuda_double_import(observation, CAM->D, CAM->obv);

    cuda_gmmPdfEval<<<framebeamwidth, CAM->K>>>(*CAM);
    cuda_error_check("cuda_gmmPdfEval");

    if (1)            cuda_memcpy(CAM->resout,  CAM->result,    framebeamwidth*sizeof(fpt),        cudaMemcpyDeviceToHost);
    if (posteriorOut) cuda_memcpy(CAM->postout, CAM->posterior, framebeamwidth*CAM->K*sizeof(fpt), cudaMemcpyDeviceToHost);

    if (1)            for (int i=0; i<framebeamwidth;        i++) emissionOut[i]  = CAM->resout[i];
    if (posteriorOut) for (int i=0; i<framebeamwidth*CAM->K; i++) posteriorOut[i] = CAM->postout[i] - emissionOut[i/CAM->K];

    return 0;
}

