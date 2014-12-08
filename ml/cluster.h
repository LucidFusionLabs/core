/*
 * $Id: cluster.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef __LFL_ML_CLUSTER_H__
#define __LFL_ML_CLUSTER_H__
namespace LFL {

DECLARE_FLAG(Initialize, int);
DECLARE_FLAG(MaxIterations, int);

struct ClusterAlgorithm { enum { KMeans=1, GMMEM=2 }; };

int features2cluster(const char *featdir, const char *modeldir, int algo) {
    Matrix *model=0, *mcov=0;
    int lastiter=0;

    if (FLAGS_Initialize) { /* intialize */
        SampleMean mean; int InitMeans=FLAGS_Initialize;
        FeatCorpus::feat_iter(featdir, SampleMean::add_features, &mean);
        INFO("mean calculated, ", mean.count, " features, ", mean.D, " dimensions");

        KMeans kmeans(InitMeans, mean.D);
        KMeansInit kmeansinit(&kmeans, mean.count);
        FeatCorpus::feat_iter(featdir, KMeansInit::add_features, &kmeansinit);
        INFO("means initialized ", InitMeans, " means (", kmeansinit.count, "/", mean.count, ")");
        model=kmeans.means; kmeans.means=0;
        MatrixFile::WriteFile(modeldir,"Cluster","means",model,lastiter);
    }
    else { /* must load means from file */
        if ((lastiter = MatrixFile::ReadFile(modeldir, "Cluster", "means", &model, 0)) < 0)
        { ERROR("no such model KMeans.means"); return 0; }

        INFO("means loaded ", model->M, " means from disk, iteration ", lastiter);
    }

    if (algo == ClusterAlgorithm::KMeans) { /* k-means */
        KMeans kmeans(model->M, model->N);
        kmeans.means->assignL(model);

        for (int i=0; i<FLAGS_MaxIterations; i++) { /* loop */
            FeatCorpus::feat_iter(featdir, KMeans::add_features, &kmeans);
            double totaldist = kmeans.totaldist;
            if (Running()) kmeans.complete(); else return 0;
            MatrixFile::WriteFile(modeldir, "Cluster", "means", kmeans.means, ++lastiter);
            INFO("k-means iteration ", lastiter, " completed, totaldist = ", totaldist);
        }

        model=kmeans.means; kmeans.means=0;
    }

    if (1) { /* if no covariance matrix, take sample covariance */
        if (lastiter > MatrixFile::ReadFile(modeldir, "Cluster", "diagcovar", &mcov, 0)) {
            SampleCovariance SV(model);
            FeatCorpus::feat_iter(featdir, SampleCovariance::add_features, &SV);
            if (Running()) SV.complete(); else return 0;
            INFO("computed sample covariance ", lastiter);
            MatrixFile::WriteFile(modeldir, "Cluster", "diagcovar", SV.diagnol, lastiter);
            mcov=SV.diagnol; SV.diagnol=0;
        }
    }

    if (algo == ClusterAlgorithm::GMMEM) { /* gaussian mixture model - expectation maximization */
        Matrix *prior = 0;
        if (1) {
            prior = new Matrix(model->M, 1);
            for (int i=0; i<prior->M; i++) prior->row(i)[0] = log((double)1/prior->M);
            MatrixFile::WriteFile(modeldir, "Cluster", "prior", prior, lastiter);
        }

        GMM gmm;
        gmm.assignDataPtr(model->M, model->N, model->m, mcov->m, prior->m);
        GMMEM EM(&gmm, true);

        for (int i=0; i<FLAGS_MaxIterations; i++) {
            EM.mode = GMMEM::Mode::Means;
            FeatCorpus::feat_iter(featdir, GMMEM::add_features, &EM);
            if (Running()) EM.complete(); else return 0;

            EM.mode = GMMEM::Mode::Cov;
            FeatCorpus::feat_iter(featdir, GMMEM::add_features, &EM);
            double totalprob = EM.totalprob; int totalcount = EM.count;
            if (Running()) EM.complete(); else return 0;

            MatrixFile::WriteFile(modeldir, "Cluster", "means",     &EM.mixture->mean,  ++lastiter);
            MatrixFile::WriteFile(modeldir, "Cluster", "diagcovar", &EM.mixture->diagcov, lastiter);
            if (prior) MatrixFile::WriteFile(modeldir, "Cluster", "prior", &EM.mixture->prior, lastiter);
            INFO("GMM-EM iteration ", lastiter, " completed, totalprob=", totalprob, " PP=", totalprob/-totalcount);
        }
    }

    if (1) {
        SampleProb SP(model, mcov);
        FeatCorpus::feat_iter(featdir, SampleProb::add_features, &SP);
        if (Running()) SP.complete(); else return 0;
        INFO("ran sampleprob");
    }
    return 0;
}

}; // namespace LFL
#endif // __LFL_ML_CLUSTER_H__
