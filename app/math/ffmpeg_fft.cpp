/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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

#include "lfapp/lfapp.h"

extern "C" {
#include <libavcodec/avfft.h>
};

namespace LFL {
void FFT(float *out, int fftlen) {
  int bits = WhichLog2(fftlen);
  CHECK_GT(bits, 0);
  RDFTContext *fftctx = av_rdft_init(bits, DFT_R2C);
  av_rdft_calc(fftctx, out);
  av_rdft_end(fftctx);
}

void IFFT(float *out, int fftlen) {
  int bits = WhichLog2(fftlen);
  CHECK_GT(bits, 0);
  RDFTContext *fftctx = av_rdft_init(bits, IDFT_C2R);
  av_rdft_calc(fftctx, &fftbuf[0]);
  av_rdft_end(fftctx);
}
}; // namespace LFL
