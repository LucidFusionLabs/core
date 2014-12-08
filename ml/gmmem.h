/*
 * $Id: gmmem.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef __LFL_ML_GMMEM_H__
#define __LFL_ML_GMMEM_H__
namespace LFL {

/* Gaussian Mixture Model Expectation Maximization */
struct GMMEM {
    GMM *mixture;
    Matrix *accums, *newmeans;
    double *denoms, totalprob;
    int count, mode;
    struct Mode { enum { Means=1, Cov=2 }; int m; };
    bool FullVariance;
    const char *name;

    ~GMMEM() { delete accums; delete newmeans; delete [] denoms; }
    GMMEM(GMM *Mixture, bool fullVariance=0, const char *Name="") : mixture(Mixture), accums(new Matrix(mixture->mean.M, mixture->mean.N)), newmeans(new Matrix(accums->M, accums->N)), denoms(new double[accums->M]), mode(0), FullVariance(fullVariance), name(Name) { reset(); }

    void reset() {
        MatrixRowIter(accums) {
            MatrixColIter(accums) { accums->row(i)[j] = 0; }
            MatrixColIter(newmeans) { newmeans->row(i)[j] = 0; }
            denoms[i] = -INFINITY;
        }
        count = 0;
        totalprob = 0;
    }

    void add_features(Matrix *features) {
        if (!dim_check("GMEM", features->N, newmeans->N)) return;
        for (int i=0; i<features->M; i++) add_feature(features->row(i));
    }
    static void add_features(const char *fn, Matrix *, Matrix *features, const char *transcript, void *arg) { return ((GMMEM*)arg)->add_features(features); }

    void add_feature(double *feature) { 
        int K = newmeans->M, D = newmeans->N;
        double *posteriors = (double*)alloca(K*sizeof(double));
        double prob = mixture->PDF(feature, posteriors);

        if (mode == Mode::Means) {
            double *featscaled=(double*)alloca(D*sizeof(double));

            MatrixRowIter(&mixture->mean) {
                Vector::mult(feature, exp(posteriors[i]), featscaled, D);
                Vector::add(newmeans->row(i), featscaled, D);

                logadd(&denoms[i], posteriors[i]);
 
                if (!FullVariance) {
                    Vector::mult(featscaled, feature, D);
                    Vector::add(accums->row(i), featscaled, D);
                }
            }
            count++;
            totalprob += prob;
        }
        else if (mode == Mode::Cov) {
            double *diff=(double*)alloca(D*sizeof(double));

            MatrixRowIter(newmeans) {
                Vector::sub(newmeans->row(i), feature, diff, D);
                Vector::mult(diff, diff, D);
                Vector::mult(diff, exp(posteriors[i]), D);
                Vector::add(accums->row(i), diff, D);
            }
            if (!FullVariance) FATAL("incompatible modes ", -1);
        }
    }
    static void add_feature(double *feature, void *arg) { return ((GMMEM*)arg)->add_feature(feature); }

    void complete() {
        if (!count) return;
        int N = newmeans->N;
        double lc = log((double)count);

        if (mode == Mode::Means) {
            MatrixRowIter(newmeans) {
                double dn = exp(denoms[i]);
                if (dn < FLAGS_CovarFloor) dn = FLAGS_CovarFloor;

                Vector::div(newmeans->row(i), dn, N); /* update newmeans */
            }
        }
        else if (mode == Mode::Cov) {
            int D = newmeans->N;
            double *x = (double*)alloca(D*sizeof(double));
            MatrixRowIter(accums) {
                double dn = exp(denoms[i]);
                if (dn < FLAGS_CovarFloor) dn = FLAGS_CovarFloor;

                if (mixture->prior.m) {
                    float v = denoms[i] - lc;
                    mixture->prior.row(i)[0] = v < FLAGS_PriorFloor ? FLAGS_PriorFloor : v; /* update priors */
                }

                Vector::assign(mixture->mean.row(i), newmeans->row(i), N); /* update means */
                Vector::div(accums->row(i), dn, mixture->diagcov.row(i), N); /* update covariance */

                if (!FullVariance) {
                    Vector::mult(mixture->mean.row(i), mixture->mean.row(i), x, D);
                    Vector::sub(mixture->diagcov.row(i), x, D);
                }

                MatrixColIter(&mixture->diagcov) if (mixture->diagcov.row(i)[j] < FLAGS_CovarFloor) mixture->diagcov.row(i)[j] = FLAGS_CovarFloor;

                mixture->computeNorms();
            }
            reset(); /* full reset */
        }
    }
};

}; // namespace LFL
#endif // __LFL_ML_GMMEM_H__
