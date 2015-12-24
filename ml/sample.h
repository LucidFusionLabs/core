/*
 * $Id: sample.h 1306 2014-09-04 07:13:16Z justin $
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
  double *vec_min=0, *vec_max=0;

  SampleExtent() {}
  ~SampleExtent() { delete vec_min; delete vec_max; }
  void Complete() {}

  void AddFeatures(Matrix *features) {
    if (!vec_min) {
      D = features->N;
      vec_min = new double[D]; Vector::Assign(vec_min, INFINITY, D);
      vec_max = new double[D]; Vector::Assign(vec_max, -INFINITY, D);
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
  int D, count;
  double *vec;

  SampleMean() : D(0), count(0), vec(0) {}
  ~SampleMean() { delete vec; }

  void Complete() { Vector::Div(vec, count, D); }

  void AddFeatures(Matrix *features) {
    if (!vec) { D = features->N; vec = new double[D](); }
    if (!DimCheck("SampleMean", features->N, D)) return;

    MatrixIter(features) { vec[j] += features->row(i)[j]; }
    count += features->M;
  }
};

struct SampleCovariance {
  int K, D, *count;
  Matrix *model, *accums, *diagnol;

  ~SampleCovariance() { delete []count; delete accums; delete diagnol; }
  SampleCovariance(Matrix *M) : K(M->M), D(M->N), count(new int[K]), model(M), accums(new Matrix(K, D)), diagnol(0) { Reset(); }

  void Reset() { memset(accums->m,0,accums->bytes); memset(count,0,K*sizeof(int)); }

  void AddFeatures(Matrix *features) {
    if (!DimCheck("SampleCovariance", features->N, D)) return;
    for (int i=0; i<features->M; i++) AddFeature(features->row(i));
  }

#if 1
  void AddFeature(double *feature) {
    int minindex; double mindist, *diff=(double*)alloca(D*sizeof(double));
    KMeans::NearestNeighbor(model, feature, &minindex, &mindist);

    Vector::Sub(model->row(minindex), feature, diff, D);
    Vector::Mult(diff, diff, D);
    Vector::Add(accums->row(minindex), diff, D);
    count[minindex]++;
  }
#else
  void AddFeature(double *feature) {
    double *diff=(double*)alloca(D*sizeof(double));
    for (int i=0; i<K; i++) {
      Vector::sub(model->row(i), feature, diff, D);
      Vector::mult(diff, diff, D);
      Vector::add(accums->row(i), diff, D);
      count[i]++;
    }
  }
#endif

  void Complete() {
    if (diagnol) delete diagnol;
    diagnol = new Matrix(model->M, model->N);

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

  ~SampleProb() {}
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
