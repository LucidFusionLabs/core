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

#ifndef LFL_ML_SAMPLE_H__
#define LFL_ML_SAMPLE_H__
namespace LFL {

struct SampleExtent {
  int D=0, count=0;
  vector<double> vec_min, vec_max;

  void Complete() {}
  void AddFeatures(Matrix *features) {
    if (!vec_min.size()) {
      D = features->N;
      vec_min = vector<double>(D, INFINITY);
      vec_max = vector<double>(D, -INFINITY);
    }
    if (!DimCheck("SampleExtents", features->N, D)) return;

    MatrixIter(features) {
      if (features->row(i)[j] < vec_min[j]) vec_min[j] = features->row(i)[j];
      if (features->row(i)[j] > vec_max[j]) vec_max[j] = features->row(i)[j];
    }
    count += features->M;
  }
};

struct SampleMean {
  int D=0, count=0;
  vector<double> vec;

  void Complete() { Vector::Div(&vec[0], count, D); }

  void AddFeatures(Matrix *features) {
    if (!vec.size()) { D = features->N; vec = vector<double>(D, 0); }
    if (!DimCheck("SampleMean", features->N, D)) return;

    MatrixIter(features) { vec[j] += features->row(i)[j]; }
    count += features->M;
  }
};

struct SampleCovariance {
  int K, D;
  unique_ptr<Matrix> accums, diagnol;
  vector<int> count;
  vector<double> diff;
  Matrix *model;

  SampleCovariance(Matrix *M) : K(M->M), D(M->N), count(K, 0), model(M),
  accums(make_unique<Matrix>(K, D)) { Reset(); }

  void Reset() { memset(accums->m, 0, accums->bytes); memset(&count[0], 0, K*sizeof(int)); }

  void AddFeatures(Matrix *features) {
    if (!DimCheck("SampleCovariance", features->N, D)) return;
    for (int i=0; i<features->M; i++) AddFeature(features->row(i));
  }

#if 1
  void AddFeature(double *feature) {
    int minindex;
    double mindist;
    diff.resize(D);
    KMeans::NearestNeighbor(model, feature, &minindex, &mindist);

    Vector::Sub(model->row(minindex), feature, diff.data(), D);
    Vector::Mult(&diff[0], diff.data(), D);
    Vector::Add(accums->row(minindex), diff.data(), D);
    count[minindex]++;
  }
#else
  void AddFeature(double *feature) {
    diff.resize(D);
    for (int i=0; i<K; i++) {
      Vector::sub(model->row(i), feature, diff.get(), D);
      Vector::mult(&diff[0], diff.get(), D);
      Vector::add(accums->row(i), diff.get(), D);
      count[i]++;
    }
  }
#endif

  void Complete() {
    diagnol = make_unique<Matrix>(model->M, model->N);

    for (int k=0; k<K; k++) {
      if (!count[k]) { Vector::Assign(accums->row(k), FLAGS_CovarFloor, D); continue; }

      Vector::Div(accums->row(k), count[k], diagnol->row(k), D);
    }
    Reset();
  }
};

struct SampleProb {
  Matrix *means, *diagcovar;
  double prob;
  SampleProb(Matrix *Means, Matrix *Var) : means(Means), diagcovar(Var) { Reset(); }

  void Reset() { prob=-INFINITY; }
  void Complete() {}

  void AddFeatures(Matrix *features) {
    if (!DimCheck("SampleProb", features->N, means->N)) return;
    for (int i=0; i<features->M; i++) AddFeature(features->row(i));
  }

  void AddFeature(double *feature) {
    double p = GmmPdfEval(means, diagcovar, feature);
    DEBUG("prob = ", exp(p));
    prob += p;
  }
};

}; // namespace LFL
#endif // LFL_ML_SAMPLE_H__
