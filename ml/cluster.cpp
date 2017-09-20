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

#include "core/app/network.h"
#include "core/ml/hmm.h"

namespace LFL {
DEFINE_string(input,         "",    "Input matrix");
DEFINE_string(inputlabel,    "",    "Input label matrix");
DEFINE_string(Transform,     "",    "Input transform matrix");
DEFINE_int   (Project,       0,     "Project input to N dimensions");
DEFINE_bool  (MeanNormalize, false, "Mean normalize input");

Matrix *in=0, *intx=0;
vector<string> *inlabel=0;
string inhdr, intxhdr, inlabelhdr;

struct FeatCorpus {
  typedef function<void(const char*, Matrix*, Matrix*, const char*)> FeatCB;
  static int FeatIter(const char *featdir, FeatCB cb) { cb("", 0, in, ""); return 0; }
};
}; // namespace LFL

#include "core/ml/kmeans.h"
#include "core/ml/gmmem.h"
#include "core/ml/sample.h"
#include "core/ml/cluster.h"

namespace LFL {
DEFINE_int   (Initialize,     0,     "Initialize models");
DEFINE_int   (MaxIterations,  50,    "Initialize models");
DEFINE_double(CovarFloor,     1e-6,  "Model covariance floor");
DEFINE_double(PriorFloor,     -16,   "Model prior and mixture weight floor");

DEFINE_int   (choosen,        0,     "Choose every N from input");
DEFINE_bool  (normalize,      false, "Normalize values during Choose Every N");
DEFINE_bool  (pca,            false, "Perform principal components analysis on input");
DEFINE_bool  (cluster,        false, "Cluster input");
DEFINE_bool  (classifynn,     false, "Classify nearest neighbor");

void Print(Matrix *m, int cols=0, vector<string> *l=0, int *indexmap=0) {
  if (!cols) cols = m->N;
  MatrixRowIter(m) {
    string v; int ind = indexmap ? indexmap[i] : i;
    for (int j=0; j<cols; j++) v += StringPrintf("%f, ", m->row(ind)[j]);
    INFO(v, " (", l ? (*l)[ind] : "", ")");
  }
}

int WritePloticus(string tofile, const Matrix *in, vector<string> *l, int *indmap, string (*plot)(const Matrix *, vector<string> *, int *)) {
  LocalFile f(tofile.c_str(), "w");
  string v = plot(in, l, indmap);
  return f.Write(v.data(), v.size()) == v.size();
}

vector<int> indexmap1D(const Matrix *in) {
  vector<double> vals;
  MatrixRowIter(in) val.push_back(in->row(i)[0]);

  vector<HMM::SortPair> sortMap(HMM::SortPair(), in->M);
  HMM::SortPairs(vals, 0, sortMap, in->M);

  vector<int> ret;
  MatrixRowIter(in) ret.push_back(sortMap[i].ind);
  return ret;
}

string tostrPloticus1D(const Matrix *in, vector<string> *l, int *indexmap) {
  double min = INFINITY, max = -INFINITY;
  MatrixRowIter(in) {
    if (in->row(i)[0] < min) min = in->row(i)[0];
    if (in->row(i)[0] > max) max = in->row(i)[0];
  }

  string v = StringPrintf("#proc areadef\n"
                          "title: Members, by date enrolled\n"
                          "rectangle: 1 3 6 3.7\n"
                          "autowidth: 0.3  3.0  8.0\n"
                          "xscaletype: linear\n"
                          "xrange: %f %f\n"
                          "yrange: 0 80\n"
                          "#proc getdata\n"
                          "data:\n", min, max);

  MatrixRowIter(in) {
    int ind = indexmap ? indexmap[i] : i;
    v += StringPrintf("%s %f\n", (*l)[ind].c_str(), in->row(ind)[0]);
  }

  v += StringPrintf("\n"
                    "#proc scatterplot\n"
                    "xfield: 2\n"
                    "labelfield: 1\n"
                    "textdetails: size=5 color=blue\n"
                    "ylocation: @AREABOTTOM+0.3\n"
                    "cluster: yes\n"
                    "verticaltext: yes\n"
                   );

  return v;
}

}; // namespace LFL
using namespace LFL;

extern "C" LFApp *MyAppCreate(int argc, const char* const* argv) {
  app = CreateApplication(argc, argv).release();
  app->focused = CreateWindow(app).release();
  return app;
}

extern "C" int MyAppMain() {
  if (app->Create(__FILE__)) return -1;

  if (!FLAGS_input.size()) FATAL("nothing to do");
  if (MatrixFile::Read(FLAGS_input, &in, &inhdr)) FATAL("MatrixFile::read ", strerror(errno));
  if (FLAGS_Transform.size() && MatrixFile::Read(FLAGS_Transform, &intx, &intxhdr)) FATAL("MatrixFile::read ", strerror(errno));

  if (FLAGS_inputlabel.size()) {
    if (StringFile::Read(FLAGS_inputlabel, &inlabel, &inlabelhdr)) FATAL("StringFile::read ", strerror(errno));
    if (in->M != inlabel->size()) FATAL("data/label mismatch ", in->M, " != ", inlabel->size());
  }

  INFO("input:");

  if (intx) {
    Matrix input(in->M, in->N);
    if (FLAGS_MeanNormalize) MeanNormalizeRows(in, &input);
    else MatrixIter(&input) input.row(i)[j] = in->row(i)[j];

    Matrix projected(in->M, in->N);
    Matrix::Mult(intx, &input, &projected, mTrnpB|mTrnpC);

    MatrixIter(in) in->row(i)[j] = projected.row(i)[j];
  }

  if (FLAGS_Project) in->N = FLAGS_Project;

  Print(in, -1, inlabel);

  if (FLAGS_choosen) {
    Matrix chosen(in->M / FLAGS_choosen, in->N);
    vector<string> label;
    MatrixRowIter(in) {
      if (Rand<int>() % FLAGS_choosen) continue;
      int ind = label.size();
      if (ind >= chosen.M) break;

      double sum = Vec<double>::Sum(in->row(i), in->N);
      MatrixColIter(in) chosen.row(ind)[j] = in->row(i)[j] / (FLAGS_normalize ? sum : 1);

      label.push_back(inlabel ? (*inlabel)[i] : "");
    }
    chosen.M -= (chosen.M - label.size());

    string hdr="choosen output", matfile="choosen.matrix", labfile="choosenlabel.matrix";
    if (MatrixFile(&chosen, hdr).Write(matfile, matfile)) FATAL("error writing ", matfile);
    if (StringFile(&label,  hdr).Write(labfile, labfile)) FATAL("error writing ", labfile);

    INFO("wrote ", matfile, " and ", labfile);
    return 0;
  }

  if (FLAGS_pca) {
    Matrix projected(in->M, in->N);
    vector<double> variance(in->N, 0);
    Matrix *pca = PCA(in, &projected, variance);
    int *indexmap = indexmap1D(&projected);

    string hdr1="PCA", fn1="PCA.matrix", hdr2="projected", fn2="projected.matrix";
    if (MatrixFile(pca,        hdr1).Write(fn1, fn1)) FATAL("error writing ", fn1);
    if (MatrixFile(&projected, hdr2).Write(fn2, fn2)) FATAL("error writing ", fn2);

#if 0
    Matrix::print(pca, "PCA");
    INFO("variance:");
    Vector::print(variance, in->N);
    Matrix::print(&projected, "projected");
#endif

    INFO("output:");
    Print(&projected, 1, inlabel, indexmap);

    WritePloticus(app->savedir + "plot.htm", &projected, inlabel, indexmap, tostrPloticus1D);
  }

  string modeldir = app->savedir;

  if (FLAGS_cluster) Features2Cluster("", modeldir.c_str(), ClusterAlgorithm::KMeans);

  if (FLAGS_classifynn) {
    int lastiter; Matrix *model=0, *mcov=0;
    if ((lastiter = MatrixFile::ReadVersioned(modeldir.c_str(), "Cluster", "means",    &model)) < 0) FATAL("cluster means");
    if (lastiter  > MatrixFile::ReadVersioned(modeldir.c_str(), "Cluster", "diagcovar", &mcov))      FATAL("cluster diagcov");

    MatrixRowIter(in) {
      int minindex; double mindist;
      KMeans::NearestNeighbor(model, in->row(i), &minindex, &mindist);
      INFO("CLUSTER=", minindex, " LABEL=", (*inlabel)[i]);
    }
  }

  return 0;
}
