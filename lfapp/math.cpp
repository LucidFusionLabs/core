/*
 * $Id: math.cpp 1335 2014-12-02 04:13:46Z justin $
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

#ifdef LFL_OPENCV
#include "opencv/cxcore.h"
#endif

#include "lfapp/lfapp.h"

namespace LFL {
bool v4::operator<(const v4 &c) const { SortMacro4(x, c.x, y, c.y, z, c.z, w, c.w); }

double Squared(double n) { return n*n; }
float Decimals(float n) { return n - (int)n; }
float Rand(float a, float b) { return ((b-a)*((float)::rand()/RAND_MAX))+a; }
unsigned long long Rand64() { return (unsigned long long)::rand()<<32 | ::rand(); }

void Clamp(float *x, float floor, float ceil) { *x = Clamp(*x, floor, ceil); }
float Clamp(float x, float floor, float ceil) {
    if (x < floor) return floor; if (x > ceil) return ceil; return x;
}
v3 Clamp(const v3& x, float floor, float ceil) {
    return v3(Clamp(x.x, floor, ceil), Clamp(x.y, floor, ceil), Clamp(x.z, floor, ceil));
}
v4 Clamp(const v4& x, float floor, float ceil) {
    return v4(Clamp(x.x, floor, ceil), Clamp(x.y, floor, ceil), Clamp(x.z, floor, ceil), Clamp(x.w, floor, ceil));
}

int RoundF(float f, bool round_point_five_up) {
    if (round_point_five_up) return (int)((f - (int)f) >= 0.5 ? f+1 : f); 
    else                     return (int)((f - (int)f) >  0.5 ? f+1 : f); 
}

int OpaqueIntPtrSort(const void *a, const void *b) { return *(int *)a - *(int *)b; }

int DimCheck(const char *alg, int d1, int d2) {
    if (d1 != d2) { ERROR(alg, ".dim ", d1, " != ", d2); return 0; }
    return 1;
}

int NextPowerOfTwo(int input) {
    int value = 1;
    while ( value < input ) value <<= 1;
    return value;
}

bool IsPowerOfTwo(unsigned x) { return x && (x & (~x + 1)) == x; }
int NextMultipleOfPowerOfTwo(int input, int align) { return (input+align-1) & ~(align-1); }
void *NextMultipleOfPowerOfTwo(void *input, int align) { return (void*)(((unsigned long long)input+align-1) & ~(align-1)); }
int NextMultipleOfN(int input, int N) { return (input % N) ? (input/N+1)*N : input; }
int PrevMultipleOfN(int input, int N) { return (input % N) ? (input/N  )*N : input; }

int WhichLog2(int n) {
    bool power_of_two = (n & (n-1)) == 0;
    if (!n || !power_of_two) return -1;

    static int wl2_pwr[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
    static int wl2_exp[] = { 0, 1, 2, 3, 4,  5,  6,  7,   8,   9,   10,   11,   12,   13,   14,    15,    16    };

    int *which = (int*)bsearch(&n, wl2_pwr, sizeof(wl2_pwr)/sizeof(int), sizeof(int), OpaqueIntPtrSort);
    if (!which) return -1;
    return wl2_exp[which - wl2_pwr];
}

int FloorLog2(int n) {
    for (int i=0; i<31; i++) if (n < (1<<i)) return i;
    return -1;
}

int IsPrime(int n) {
    int r = (int)sqrtf(n);
    for (int i=2; i<=r; i++) if (!(n % i)) return 0;
    return 1;
}

int NextPrime(int n) {
    for (/**/; !IsPrime(n); n++) {}
    return n;
}

double RadianToDegree(float rad) { return rad * 180 / M_PI; }
double DegreeToRadian(float deg) { return deg * M_PI / 180; }
double HZToMel(double f) { return 2595 * log10(f/700 + 1); }
double MelToHZ(double f) { return 700 * (pow(10, f/2595) - 1); }
double HZToBark(double f) { return 6 * asinh(f/600); }
double BarkToHZ(double f) { return 600 * sinh(f/6); }
float CMToInch(float f) { return f * 0.393701;  }
float MMToInch(float f) { return f * 0.0393701; }
float InchToCM(float f) { return f / 0.393701;  }
float InchToMM(float f) { return f / 0.0393701; }

double CumulativeAvg(double last, double next, int count) { return (last*(count-1) + next) / count; }

/* interpolate f(x) : 0<=x<=1 */
double ArrayAsFunc(double *Array, int ArrayLen, double x) {
    double xi=x*(ArrayLen-1); int xind=(int)xi;
    return (Array[xind+1] - Array[xind]) * (xi-xind) + Array[xind]; 
}

#if 0
double MatrixAsFunc(Matrix *m, double x, double y)
{
    double yi=y*(m->M-1), xi=x*(m->N-1);
    int yind=yi, xind=xi;
    double *row = m->row(yind);
    return row[xind];
}
#else
/* bilinearly interpolate m as f(x,y) : 0<=x<=1 */
double MatrixAsFunc(Matrix *m, double x, double y) {
    double yi=y*(m->M-1), xi=x*(m->N-1);
    int yind=(int)yi, xind=(int)xi;
    int yp=yind!=m->M-1, xp=xind!=m->N-1;
    double *row1 = m->row(yind), *row2 = m->row(yind+yp);
    double q11 = row1[xind], q12 = row1[xind+xp];
    double q21 = row2[xind], q22 = row2[xind+xp];

    double m1 = (q12 - q11) * (xi-xind) + q11;
    double m2 = (q22 - q21) * (xi-xind) + q21;

    return (m2 - m1) * (yi-yind) + m1;
}
#endif

int Levenshtein(const vector<int> &source, const vector<int> &target, vector<pair<int, int> > *alignment) {
    matrix<int> mat(source.size()+1, target.size()+1);
    matrix<pair<int, int> > bt(mat.M, mat.N, pair<int,int>());
    MatrixRowIter(&mat) mat.row(i)[0] = i;
    MatrixColIter(&mat) mat.row(0)[j] = j;

    for (int i=1; i<mat.M; i++) {
        for (int j=1; j<mat.N; j++) {
            pair<int,int> above(i-1, j), left(i, j-1), diag(i-1, j-1), min = diag;
            int cost = source[i-1] == target[j-1] ? 0 : 1, cell = mat.row(diag.first)[diag.second] + cost;
            if (mat.row(above.first)[above.second]+1 < cell) { min = above; cost=1; };
            if (mat.row(left.first)[left.second]+1 < cell) { min = left; cost=1; }
            cell = mat.row(min.first)[min.second] + cost;
            mat.row(i)[j] = cell;
            bt.row(i)[j] = min;
        }
    }
    return mat.row(mat.M-1)[mat.N-1];
}

double LevinsonDurbin(int order, const double *ac, double *ref, double *lpc) {
    double r, mmse = ac[0]; int i, j;
    if (!ac[0]) { for (i=0; i<order; i++) ref[i] = 0; return 0; }

    for (i=0; i<order; i++) {
        /* sum reflection coefficient */
        r = -ac[i + 1];
        for (j=0; j<i; j++) r -= lpc[j] * ac[i - j];
        ref[i] = r /= mmse;

        /* update LPC and total error */
        lpc[i] = r;
        for (j=0; j<i/2; j++) {
            double tmp  = lpc[j];
            lpc[j]     += r * lpc[i-1-j];
            lpc[i-1-j] += r * tmp;
        }
        if (i % 2) lpc[j] += lpc[j] * r;
        mmse *= 1.0 - r * r;
    }
    return mmse;
}

double Sinc(double x) { return x ? sin(M_PI*x)/(M_PI*x) : 1; }

double Hamming(int n, int i) { return 0.54 - 0.46 * cos(2*M_PI*i/(n-1)); }

double AmplitudeRatioDecibels(float a1, float a2) { return 20*log10(1e-20+a1/a2); }

#ifdef _WIN32
double log1p (double x) { double y=1+x, z=y-1; return log(y)-(z-x)/y; }
#endif

double LogAdd(double *x, int n) {
    const double *xi, *xe = x + n;
    double x_max = -INFINITY, x_minus_max_sum_exp = 0;
    for (xi = x; xi < xe; xi++) if (*xi > x_max) x_max = *xi;
    for (xi = x; xi < xe; xi++) x_minus_max_sum_exp += exp(*xi - x_max);
    return x_max + log(x_minus_max_sum_exp);
}
double LogAdd(double x, double y) {
    if (x<y) return y + log1p(exp(x - y));
    else     return x + log1p(exp(y - x));
}
double LogAdd(double *x, double y) { return (*x = LogAdd(*x, y)); }
float LogAdd(float  *x, float y)  { return (*x = LogAdd(*x, y)); }
unsigned LogAdd(unsigned *x, unsigned y) { FATAL("not implemented"); }
double LogSub(double *x, double y) { return (*x = LogSub(*x, y)); }
double LogSub(double x, double y) { return x + log1p(-exp(y - x)); }

double asinh(double x) { return log(x + sqrt(x*x + 1)); }
double acosh(double x) { return log(x + sqrt(x*x - 1)); }
double atanh(double x) { return (x >= 0.5) ? (0.5 * log((1+x)/(1-x))) : (0.5 * log((2*x)+(2*x)*x / (1-x))); }

#if 0
float  &fft_r(float  *a, int fftlen, int index) { return a[index]; }
double &fft_r(double *a, int fftlen, int index) { return a[index]; }
float  &fft_i(float  *a, int fftlen, int index) { static double z=0; return !index ? z : a[fftlen-index]; }
double &fft_i(double *a, int fftlen, int index) { static float z=0; return !index ? z : a[fftlen-index]; }
const float  &fft_r(const float  *a, int fftlen, int index) { return a[index]; }
const double &fft_r(const double *a, int fftlen, int index) { return a[index]; }
const float  &fft_i(const float  *a, int fftlen, int index) { static double z=0; return !index ? z : a[fftlen-index]; }
const double &fft_i(const double *a, int fftlen, int index) { static float z=0; return !index ? z : a[fftlen-index]; }
#else
float  &fft_r(float  *a, int fftlen, int index) { return a[index*2]; }
double &fft_r(double *a, int fftlen, int index) { return a[index*2]; }
float  &fft_i(float  *a, int fftlen, int index) { return a[index*2+1]; }
double &fft_i(double *a, int fftlen, int index) { return a[index*2+1]; }
const float  &fft_r(const float  *a, int fftlen, int index) { return a[index*2]; }
const double &fft_r(const double *a, int fftlen, int index) { return a[index*2]; }
const float  &fft_i(const float  *a, int fftlen, int index) { return a[index*2+1]; }
const double &fft_i(const double *a, int fftlen, int index) { return a[index*2+1]; }
#endif
double fft_abs2(int fftlen, int index, float r, float i) {
    if (index == 0 || index == fftlen/2) return r*r;
    else return r*r + i*i;
}
double fft_abs2(const float  *a, int fftlen, int index) { return fft_abs2(fftlen, index, fft_r(a, fftlen, index), fft_i(a, fftlen, index)); }
double fft_abs2(const double *a, int fftlen, int index) { return fft_abs2(fftlen, index, fft_r(a, fftlen, index), fft_i(a, fftlen, index)); }

Matrix *IDFT(int rows, int cols) {
    Matrix *m = new Matrix(rows, cols);
    MatrixIter(m) {
        m->row(i)[j] = !j ? 1.0 : cos(M_PI * i * j / (cols-1)) * (j == cols-1 ? 1 : 2);
    }
    return m;
}

/* DCT3 is transpose of DCT2 */
Matrix *DCT2(int rows, int cols) {
    Matrix *m = new Matrix(rows, cols);
    MatrixIter(m) { 
        m->row(i)[j] = cos(M_PI * i * (-1+(j+1)*2) / (2*cols)) * sqrt((double)2/cols);	
    }
    return m;
}

double MahalDist2(const double *mean, const double *diagcovar, const double *vec, int D) {
    double sum=0;
    for (int i=0; i<D; i++) {
        double diff=mean[i]-vec[i];
        sum += (diff*diff)/diagcovar[i];
    }
    return sum;
}

double DiagDet(const double *diagmat, int D) {
    double det=0;
    for (int i=0; i<D; i++) det += log(diagmat[i]);
    return det;
}

double GausNormC(const double *diagcovar, int D) {
    static const double log_tau = log(M_TAU);
    return -0.5 * (log_tau*D + DiagDet(diagcovar, D));
}

double GausPdfEval(const double *mean, const double *diagcovar, const double *normC, const double *observation, int D) {
    return (normC ? *normC : GausNormC(diagcovar, D)) - 0.5 * MahalDist2(mean, diagcovar, observation, D);
}

double GmmPdfEval(const Matrix *means, const Matrix *diagcovar,
                  const double *obv, const double *prior, const double *normC, double *outPosteriors) {
    int K=means->M, D=means->N;
    double total=-INFINITY, *prob=(double *)alloca(K*sizeof(double));
    MatrixRowIter(means) {
        prob[i] = GausPdfEval(means->row(i), diagcovar->row(i), normC?&normC[i]:0, obv, D);
        if (prior) prob[i] += prior[i];
        LogAdd(&total, prob[i]); 
    }
    if (outPosteriors) {
        MatrixRowIter(means) { outPosteriors[i] = prob[i] - total; }
    }
    return total;
}

void MeanNormalizeRows(const Matrix *in, Matrix *out) {
    double *mean = new double[in->N];
    MatrixIter(in) mean[j] += in->row(i)[j];
    MatrixColIter(in) mean[j] /= in->M;
    MatrixIter(out) out->row(i)[j] = in->row(i)[j] - mean[j];
    delete [] mean;
}

void Invert(const Matrix *in, Matrix *out) {
#ifndef LFL_OPENCV
    ERROR("not implemented");
#else
    cv::Mat cvI(in->M,   in->N, CV_64FC1,  in->m);
    cv::Mat cvO(out->M, out->N, CV_64FC1, out->m);
    cvO = cvI.inv();
#endif
}

void SVD(const Matrix *A, Matrix *D, Matrix *U, Matrix *V) {
#ifndef LFL_OPENCV
    ERROR("not implemented");
#else
    CvMat cvA, cvD, cvU, cvV;
    cvInitMatHeader(&cvA, A->M, A->N, CV_64FC1, A->m);
    cvInitMatHeader(&cvD, D->M, D->N, CV_64FC1, D->m);
    cvInitMatHeader(&cvU, U->M, U->N, CV_64FC1, U->m);
    cvInitMatHeader(&cvV, V->M, V->N, CV_64FC1, V->m);
    cvSVD(&cvA, &cvD, &cvU, &cvV, CV_SVD_U_T|CV_SVD_V_T);
#endif
}

Matrix *PCA(const Matrix *obv, Matrix *projected, double *var) {
    Matrix A(obv->M, obv->N), D(obv->N, 1), U(obv->M, obv->M), V(obv->N, obv->N);
    MeanNormalizeRows(obv, &A);

    double scale = sqrt((double)obv->M-1);
    MatrixIter(&A) A.row(i)[j] /= scale;
    SVD(&A, &D, &U, &V);
    MatrixIter(&A) A.row(i)[j] *= scale;

    if (!Matrix::Mult(&V, &A, projected, mTrnpB|mTrnpC)) FATAL("mult failed");

    if (var) MatrixRowIter(&D) var[i] = pow(D.row(i)[0], 2);
    return V.Clone();
}

}; // namespace LFL
