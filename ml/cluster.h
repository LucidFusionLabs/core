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

#ifndef LFL_ML_CLUSTER_H__
#define LFL_ML_CLUSTER_H__
namespace LFL {

DECLARE_FLAG(Initialize, int);
DECLARE_FLAG(MaxIterations, int);

struct ClusterAlgorithm { enum { KMeans=1, GMMEM=2 }; };

int Features2Cluster(AssetLoading *loader, const char *featdir, const char *modeldir, int algo) {
  unique_ptr<Matrix> model(make_unique<Matrix>()), mcov;
  int lastiter = 0;

  if (FLAGS_Initialize) { /* intialize */
    SampleMean mean;
    int init_means = FLAGS_Initialize;
    FeatCorpus::FeatIter(featdir, [&](const char*, Matrix*, Matrix *f, const char*) { mean.AddFeatures(f); });
    INFO("mean calculated, ", mean.count, " features, ", mean.D, " dimensions");

    KMeans kmeans(init_means, mean.D);
    KMeansInit kmeansinit(&kmeans, mean.count);
    FeatCorpus::FeatIter(featdir, [&](const char*, Matrix*, Matrix *f, const char*) { kmeansinit.AddFeatures(f); });
    INFO("means initialized ", init_means, " means (", kmeansinit.count, "/", mean.count, ")");
    model = move(kmeans.means);
    MatrixFile(model.get(), "").WriteVersioned(modeldir, "Cluster", "means", lastiter);
  }
  else { /* must load means from file */
    if ((lastiter = MatrixFile::ReadVersioned(modeldir, "Cluster", "means", &model, 0)) < 0)
    { ERROR("no such model KMeans.means"); return 0; }

    INFO("means loaded ", model->M, " means from disk, iteration ", lastiter);
  }

  if (algo == ClusterAlgorithm::KMeans) { /* k-means */
    KMeans kmeans(model->M, model->N);
    kmeans.means->AssignL(model.get());

    for (int i=0; i<FLAGS_MaxIterations; i++) { /* loop */
      FeatCorpus::FeatIter(featdir, [&](const char*, Matrix*, Matrix *f, const char*) { kmeans.AddFeatures(f); });
      double totaldist = kmeans.totaldist;
      if (loader->lifetime->run) kmeans.Complete(); else return 0;
      MatrixFile(kmeans.means.get()).WriteVersioned(modeldir, "Cluster", "means", ++lastiter);
      INFO("k-means iteration ", lastiter, " completed, totaldist = ", totaldist);
    }

    model = move(kmeans.means);
  }

  if (1) { /* if no covariance matrix, take sample covariance */
    if (lastiter > MatrixFile::ReadVersioned(modeldir, "Cluster", "diagcovar", &mcov, 0)) {
      SampleCovariance SV(model.get());
      FeatCorpus::FeatIter(featdir, [&](const char*, Matrix*, Matrix *f, const char*) { SV.AddFeatures(f); });
      if (loader->lifetime->run) SV.Complete(); else return 0;
      INFO("computed sample covariance ", lastiter);
      MatrixFile(SV.diagnol.get()).WriteVersioned(modeldir, "Cluster", "diagcovar", lastiter);
      mcov = move(SV.diagnol);
    }
  }

  if (algo == ClusterAlgorithm::GMMEM) { /* gaussian mixture model - expectation maximization */
    unique_ptr<Matrix> prior;
    if (1) {
      prior = make_unique<Matrix>(model->M, 1);
      for (int i=0; i<prior->M; i++) prior->row(i)[0] = log(1.0 / prior->M);
      MatrixFile(prior.get(), "").WriteVersioned(modeldir, "Cluster", "prior", lastiter);
    }

    GMM gmm;
    gmm.AssignDataPtr(model->M, model->N, model->m, mcov->m, prior->m);
    GMMEM EM(&gmm, true);

    for (int i=0; i<FLAGS_MaxIterations; i++) {
      EM.mode = GMMEM::Mode::Means;
      FeatCorpus::FeatIter(featdir, [&](const char*, Matrix*, Matrix *f, const char*) { EM.AddFeatures(f); });
      if (loader->lifetime->run) EM.Complete(); else return 0;

      EM.mode = GMMEM::Mode::Cov;
      FeatCorpus::FeatIter(featdir, [&](const char*, Matrix*, Matrix *f, const char*) { EM.AddFeatures(f); });
      double totalprob = EM.totalprob; int totalcount = EM.count;
      if (loader->lifetime->run) EM.Complete(); else return 0;

      MatrixFile(&EM.mixture->mean)   .WriteVersioned(modeldir, "Cluster", "means",   ++lastiter);
      MatrixFile(&EM.mixture->diagcov).WriteVersioned(modeldir, "Cluster", "diagcovar", lastiter);
      if (prior) MatrixFile(&EM.mixture->prior).WriteVersioned(modeldir, "Cluster", "prior", lastiter);
      INFO("GMM-EM iteration ", lastiter, " completed, totalprob=", totalprob, " PP=", totalprob/-totalcount);
    }
  }

  if (1) {
    SampleProb SP(model.get(), mcov.get());
    FeatCorpus::FeatIter(featdir, [&](const char*, Matrix*, Matrix *f, const char*) { SP.AddFeatures(f); });
    if (loader->lifetime->run) SP.Complete(); else return 0;
    INFO("ran sampleprob");
  }
  return 0;
}

}; // namespace LFL
#endif // LFL_ML_CLUSTER_H__
