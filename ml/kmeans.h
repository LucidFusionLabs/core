/*
 * $Id: kmeans.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef __LFL_ML_KMEANS_H__
#define __LFL_ML_KMEANS_H__
namespace LFL {

DECLARE_FLAG(CovarFloor, double);
DECLARE_FLAG(PriorFloor, double);

struct KMeans {
    int K, D, *count;
    Matrix *means, *accums, *prior;
    double totaldist;

    KMeans(int k, int d) : K(k), D(d), count(new int[K]), means(new Matrix(K,D)), accums(new Matrix(K,D)), prior(new Matrix(K,1)) { reset(); }
    ~KMeans() { delete []count; delete means; delete accums; delete prior; }

    void reset() { memset(accums->m,0,accums->bytes); memset(count,0,K*sizeof(int)); totaldist=0; }
    
    void add_features(Matrix *features) {
        if (!dim_check("KMeans", features->N, D)) return;
        for (int i=0; i<features->M; i++) add_feature(features->row(i));
    }
    static void add_features(const char *fn, Matrix *, Matrix *features, const char *transcript, void *arg) { ((KMeans*)arg)->add_features(features); }

    void add_feature(double *feature) {
        double mindist; int minindex;
        nearest_neighbor(means, feature, &minindex, &mindist);

        Vector::add(accums->row(minindex), feature, D);
        count[minindex]++;
        totaldist += mindist;
    }
    static void add_feature(double *feature, void *arg) { ((KMeans*)arg)->add_feature(feature); }
    
    static void nearest_neighbor(Matrix *model, double *vectorIn, int *minindexOut, double *mindistOut, double *distOut=0) {
        double mindist; int minindex=-1;
        for (int k=0; k<model->M; k++) {
            double distance = Vector::dist2(model->row(k), vectorIn, model->N);
            if (minindex<0 || distance < mindist) { mindist=distance; minindex=k; }
            if (distOut) distOut[k] = distance;
        }
        *minindexOut=minindex; *mindistOut=mindist;
    }

    void complete() {
        int totalcount = 0;
        for (int k=0; k<K; k++) totalcount += count[k];
        INFO("Kmeans complete totaldist = ", totaldist, ", totalcount = ", totalcount);

        for (int k=0; k<K; k++) {
            if (!count[k]) ERROR("kmeans div0");
            else Vector::div(accums->row(k), count[k], means->row(k), D);

            prior->row(k)[0] = totalcount ? log((double)count[k] / totalcount) : FLAGS_PriorFloor;
        }
        reset();
    }
};

struct KMeansInit {
    KMeans *kmeans;
    int features, count, *pick;

    ~KMeansInit() { delete [] pick; }

    KMeansInit(KMeans *out, int feats) : kmeans(out), features(feats), count(0), pick(new int[kmeans->K]) {
        if (!features) {
            ERROR("KMeans::Init called with ", features, " features");
            for (int k=0; k<kmeans->K; k++) Vector::assign(kmeans->means->row(k), 0.0, kmeans->D);
            return; 
        }

        for (int i=0, j=0; i<kmeans->K; i++) { 
            pick[i] = rand(0, features-1); /* choose uniform randomly */

            /* dupe check */
            for (j=0; j<i; j++) if (pick[j] == pick[i]) break;
            if (j != i) i--;
        }    
    }

    void add_feature(double *feature) {
        for (int k=0; k<kmeans->K; k++) {
            if (count != pick[k]) continue;

            double *mean = kmeans->means->row(k);
            Vector::assign(mean, feature, kmeans->D);

            string s;
            for (int l=0; l<kmeans->D; l++) StrAppend(&s, mean[l], ", ");
            INFO("chose mean ", k, " (", count, ") [", s, "]");
        }
        count++;
    }
    static void add_feature(double *feature, void *arg) { ((KMeansInit*)arg)->add_feature(feature); }

    void add_features(Matrix *features) {
        if (!dim_check("KMeans.init", features->N, kmeans->D)) return;
        for (int i=0; i<features->M; i++) add_feature(features->row(i));
    }
    static void add_features(const char *fn, Matrix *, Matrix *features, const char *transcript, void *arg) { ((KMeansInit*)arg)->add_features(features); }
};

}; // namespace LFL
#endif // __LFL_ML_KMEANS_H__
