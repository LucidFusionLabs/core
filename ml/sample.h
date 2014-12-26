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

#ifndef __LFL_ML_SAMPLE_H__
#define __LFL_ML_SAMPLE_H__
namespace LFL {

struct SampleExtent {
    int D, count;
    double *vec_min, *vec_max;

    SampleExtent() : D(0), count(0), vec_min(0), vec_max(0) {}
    ~SampleExtent() { delete vec_min; delete vec_max; }
    void complete() {}

    void add_features(Matrix *features) {
        if (!vec_min) {
            D = features->N;
            vec_min = new double[D]; vec_max = new double[D];
            Vector::Assign(vec_min, INFINITY, D); Vector::Assign(vec_max, -INFINITY, D);
        }
        if (!dim_check("SampleExtents", features->N, D)) return;

        MatrixIter(features) {
            if (features->row(i)[j] < vec_min[j]) vec_min[j] = features->row(i)[j];
            if (features->row(i)[j] > vec_max[j]) vec_max[j] = features->row(i)[j];
        }
        count += features->M;
    }
    static void add_label_features(int label, Matrix *features, void *arg) { ((SampleExtent*)arg)->add_features(features); }
    static void add_mfcc_features(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript, void *arg) { ((SampleExtent*)arg)->add_features(features); }
};

struct SampleMean {
    int D, count;
    double *vec;

    SampleMean() : D(0), count(0), vec(0) {}
    ~SampleMean() { delete vec; }

    void complete() { Vector::Div(vec, count, D); }

    void add_features(Matrix *features) {
        if (!vec) { D = features->N; vec = new double[D](); }
        if (!dim_check("SampleMean", features->N, D)) return;

        MatrixIter(features) { vec[j] += features->row(i)[j]; }
        count += features->M;
    }
    static void add_label_features(int label, Matrix *features, void *arg) { ((SampleMean*)arg)->add_features(features); }
    static void add_features(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript, void *arg) { ((SampleMean*)arg)->add_features(features); }
};

struct SampleCovariance {
    int K, D, *count;
    Matrix *model, *accums, *diagnol;

    ~SampleCovariance() { delete []count; delete accums; delete diagnol; }
    SampleCovariance(Matrix *M) : K(M->M), D(M->N), count(new int[K]), model(M), accums(new Matrix(K, D)), diagnol(0) { reset(); }

    void reset() { memset(accums->m,0,accums->bytes); memset(count,0,K*sizeof(int)); }
    
    void add_features(Matrix *features) {
        if (!dim_check("SampleCovariance", features->N, D)) return;
        for (int i=0; i<features->M; i++) add_feature(features->row(i));
    }
    static void add_features(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript, void *arg) { return ((SampleCovariance*)arg)->add_features(features); }
#if 1
    void add_feature(double *feature) {
        int minindex; double mindist, *diff=(double*)alloca(D*sizeof(double));
        KMeans::nearest_neighbor(model, feature, &minindex, &mindist);

        Vector::Sub(model->row(minindex), feature, diff, D);
        Vector::Mult(diff, diff, D);
        Vector::Add(accums->row(minindex), diff, D);
        count[minindex]++;
    }
#else
    void add_feature(double *feature) {
        double *diff=(double*)alloca(D*sizeof(double));
        for (int i=0; i<K; i++) {
            Vector::sub(model->row(i), feature, diff, D);
            Vector::mult(diff, diff, D);
            Vector::add(accums->row(i), diff, D);
            count[i]++;
        }
    }
#endif
    static void add_feature(double *feature, void *arg) { return ((SampleCovariance*)arg)->add_feature(feature); }

    void complete() {
        if (diagnol) delete diagnol;
        diagnol = new Matrix(model->M, model->N);

        for (int k=0; k<K; k++) {
            if (!count[k]) { Vector::Assign(accums->row(k), FLAGS_CovarFloor, D); continue; }

            Vector::Div(accums->row(k), count[k], diagnol->row(k), D);
        }
        reset();
    }
};

struct SampleProb {
    Matrix *means, *diagcovar;
    double prob;

    ~SampleProb() {}
    SampleProb(Matrix *Means, Matrix *Var) : means(Means), diagcovar(Var) { reset(); }

    void reset() { prob=-INFINITY; }
    
    void add_features(Matrix *features) {
        if (!dim_check("SampleProb", features->N, means->N)) return;
        for (int i=0; i<features->M; i++) add_feature(features->row(i));
    }
    static void add_features(const char *fn, Matrix *, Matrix *features, const char *transcript, void *arg) { ((SampleProb*)arg)->add_features(features); }

    void add_feature(double *feature) {
        double p = gmmPdfEval(means, diagcovar, feature);
        DEBUG("prob = ", exp(p));
        prob += p;
    }
    static void add_feature(double *feature, void *arg) { return ((SampleProb*)arg)->add_feature(feature); }

    void complete() {}
};

}; // namespace LFL
#endif // __LFL_ML_SAMPLE_H__
