/*
 * $Id: lfcuda.h 496 2011-12-12 01:56:48Z justin $
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

#ifndef __LFL_LFAPP_LFCUDA_H__
#define __LFL_LFAPP_LFCUDA_H__

typedef float fpt;
extern void cuda_init_hook();

struct CudaAcousticModel {
    int states, K, D, ksize, kdsize, kelsize, kdelsize, *beam;
    fpt *mean, *covar, *prior, *norm, *obv, *result, *resout, *posterior, *postout;

    CudaAcousticModel(int numstates, int k, int d);
    static const int MaxK=64, MaxObvLen=4096, MaxBeamWidth=16384;

    static int load(CudaAcousticModel *out, AcousticModel::Compiled *in);
    static int calcEmissions(CudaAcousticModel *model, const double *observation, const int *beam, int framebeamwidth, double *emissionOut, double *posteriorOut);
};

#endif
