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

#ifndef LFL_ML_KMEANS_H__
#define LFL_ML_KMEANS_H__
namespace LFL {

DECLARE_FLAG(CovarFloor, double);
DECLARE_FLAG(PriorFloor, double);

struct KMeans {
  int K, D;
  vector<int> count;
  unique_ptr<Matrix> means, accums, prior;
  double totaldist;

  KMeans(int k, int d) : K(k), D(d), count(0, K), means(make_unique<Matrix>(K,D)),
  accums(make_unique<Matrix>(K,D)), prior(make_unique<Matrix>(K,1)) { Reset(); }

  void Reset() {
    memset(accums->m, 0, accums->bytes);
    memset(&count[0], 0, K*sizeof(int));
    totaldist = 0; 
  }

  void AddFeatures(Matrix *features) {
    if (!DimCheck("KMeans", features->N, D)) return;
    for (int i=0; i<features->M; i++) AddFeature(features->row(i));
  }

  void AddFeature(double *feature) {
    double mindist; int minindex;
    NearestNeighbor(means.get(), feature, &minindex, &mindist);

    Vector::Add(accums->row(minindex), feature, D);
    count[minindex]++;
    totaldist += mindist;
  }

  static void NearestNeighbor(Matrix *model, double *vectorIn, int *minindexOut, double *mindistOut, double *distOut=0) {
    double mindist; int minindex=-1;
    for (int k=0; k<model->M; k++) {
      double distance = Vector::Dist2(model->row(k), vectorIn, model->N);
      if (minindex<0 || distance < mindist) { mindist=distance; minindex=k; }
      if (distOut) distOut[k] = distance;
    }
    *minindexOut=minindex; *mindistOut=mindist;
  }

  void Complete() {
    int totalcount = 0;
    for (int k=0; k<K; k++) totalcount += count[k];
    INFO("Kmeans complete totaldist = ", totaldist, ", totalcount = ", totalcount);

    for (int k=0; k<K; k++) {
      if (!count[k]) ERROR("kmeans div0");
      else Vector::Div(accums->row(k), count[k], means->row(k), D);

      prior->row(k)[0] = totalcount ? log((double)count[k] / totalcount) : FLAGS_PriorFloor;
    }
    Reset();
  }
};

struct KMeansInit {
  KMeans *kmeans;
  vector<int> pick;
  int features, count=0;

  KMeansInit(KMeans *out, int feats) : kmeans(out), pick(kmeans->K, 0), features(feats) {
    if (!features) {
      ERROR("KMeans::Init called with ", features, " features");
      for (int k=0; k<kmeans->K; k++) Vector::Assign(kmeans->means->row(k), 0.0, kmeans->D);
      return; 
    }

    for (int i=0, j=0; i<kmeans->K; i++) { 
      pick[i] = Rand(0, features-1); /* choose uniform randomly */

      /* dupe check */
      for (j=0; j<i; j++) if (pick[j] == pick[i]) break;
      if (j != i) i--;
    }    
  }

  void AddFeature(double *feature) {
    for (int k=0; k<kmeans->K; k++) {
      if (count != pick[k]) continue;

      double *mean = kmeans->means->row(k);
      Vector::Assign(mean, feature, kmeans->D);

      string s;
      for (int l=0; l<kmeans->D; l++) StrAppend(&s, mean[l], ", ");
      INFO("chose mean ", k, " (", count, ") [", s, "]");
    }
    count++;
  }

  void AddFeatures(Matrix *features) {
    if (!DimCheck("KMeans.init", features->N, kmeans->D)) return;
    for (int i=0; i<features->M; i++) AddFeature(features->row(i));
  }
};

}; // namespace LFL
#endif // LFL_ML_KMEANS_H__
