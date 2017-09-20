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

#ifndef LFL_ML_GMMEM_H__
#define LFL_ML_GMMEM_H__
namespace LFL {

/* Gaussian Mixture Model Expectation Maximization */
struct GMMEM {
  struct Mode { enum { Means=1, Cov=2 }; int m; };
  GMM *mixture;
  unique_ptr<Matrix> accums, newmeans;
  vector<double> denoms, posteriors, featscaled, diff, xv;
  double totalprob;
  int count, mode=0;
  bool full_variance;
  const char *name;
  GMMEM(GMM *m, bool fv=0, const char *n="") : mixture(m), accums(make_unique<Matrix>(mixture->mean.M, mixture->mean.N)),
  newmeans(make_unique<Matrix>(accums->M, accums->N)), denoms(accums->M, 0), full_variance(fv), name(n) { Reset(); }

  void Reset() {
    MatrixRowIter(accums) {
      MatrixColIter(accums) { accums->row(i)[j] = 0; }
      MatrixColIter(newmeans) { newmeans->row(i)[j] = 0; }
      denoms[i] = -INFINITY;
    }
    count = 0;
    totalprob = 0;
  }

  void AddFeatures(Matrix *features) {
    if (!DimCheck("GMEM", features->N, newmeans->N)) return;
    for (int i=0; i<features->M; i++) AddFeature(features->row(i));
  }

  void AddFeature(double *feature) { 
    int K = newmeans->M, D = newmeans->N;
    posteriors.resize(K);
    double prob = mixture->PDF(feature, posteriors.data());

    if (mode == Mode::Means) {
      featscaled.resize(D);

      MatrixRowIter(&mixture->mean) {
        Vector::Mult(feature, exp(posteriors[i]), &featscaled[0], D);
        Vector::Add(newmeans->row(i), featscaled.data(), D);

        LogAdd(&denoms[i], posteriors[i]);

        if (!full_variance) {
          Vector::Mult(&featscaled[0], feature, D);
          Vector::Add(accums->row(i), featscaled.data(), D);
        }
      }
      count++;
      totalprob += prob;
    }
    else if (mode == Mode::Cov) {
      diff.resize(D);

      MatrixRowIter(newmeans) {
        Vector::Sub(newmeans->row(i), feature, &diff[0], D);
        Vector::Mult(&diff[0], diff.data(), D);
        Vector::Mult(&diff[0], exp(posteriors[i]), D);
        Vector::Add(accums->row(i), diff.data(), D);
      }
      if (!full_variance) FATAL("incompatible modes ", -1);
    }
  }

  void Complete() {
    if (!count) return;
    int N = newmeans->N;
    double lc = log((double)count);

    if (mode == Mode::Means) {
      MatrixRowIter(newmeans) {
        double dn = exp(denoms[i]);
        if (dn < FLAGS_CovarFloor) dn = FLAGS_CovarFloor;

        Vector::Div(newmeans->row(i), dn, N); /* update newmeans */
      }
    }
    else if (mode == Mode::Cov) {
      int D = newmeans->N;
      MatrixRowIter(accums) {
        double dn = exp(denoms[i]);
        if (dn < FLAGS_CovarFloor) dn = FLAGS_CovarFloor;

        if (mixture->prior.m) {
          float v = denoms[i] - lc;
          mixture->prior.row(i)[0] = v < FLAGS_PriorFloor ? FLAGS_PriorFloor : v; /* update priors */
        }

        Vector::Assign(mixture->mean.row(i), newmeans->row(i), N); /* update means */
        Vector::Div(accums->row(i), dn, mixture->diagcov.row(i), N); /* update covariance */

        if (!full_variance) {
          xv.resize(D);
          Vector::Mult(mixture->mean.row(i), mixture->mean.row(i), &xv[0], D);
          Vector::Sub(mixture->diagcov.row(i), xv.data(), D);
        }

        MatrixColIter(&mixture->diagcov) if (mixture->diagcov.row(i)[j] < FLAGS_CovarFloor) mixture->diagcov.row(i)[j] = FLAGS_CovarFloor;

        mixture->ComputeNorms();
      }
      Reset(); /* full reset */
    }
  }
};

}; // namespace LFL
#endif // LFL_ML_GMMEM_H__
