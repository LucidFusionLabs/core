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

#include "opencv/cxcore.h"

namespace LFL {
void Invert(const Matrix *in, Matrix *out) {
  cv::Mat cvI(in->M,   in->N, CV_64FC1,  in->m);
  cv::Mat cvO(out->M, out->N, CV_64FC1, out->m);
  cvO = cvI.inv();
}

void SVD(const Matrix *A, Matrix *D, Matrix *U, Matrix *V) {
  CvMat cvA, cvD, cvU, cvV;
  cvInitMatHeader(&cvA, A->M, A->N, CV_64FC1, A->m);
  cvInitMatHeader(&cvD, D->M, D->N, CV_64FC1, D->m);
  cvInitMatHeader(&cvU, U->M, U->N, CV_64FC1, U->m);
  cvInitMatHeader(&cvV, V->M, V->N, CV_64FC1, V->m);
  cvSVD(&cvA, &cvD, &cvU, &cvV, CV_SVD_U_T|CV_SVD_V_T);
}
}; // namespace LFL
