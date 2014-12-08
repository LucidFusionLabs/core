/*
 * $Id: trainer.cpp 1306 2014-09-04 07:13:16Z justin $
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

#include "lfapp/lfapp.h"
#include "lfapp/network.h"
#include "ml/hmm.h"

using namespace LFL;

DEFINE_string(input,         "",    "Input matrix");
DEFINE_string(inputlabel,    "",    "Input label matrix");
DEFINE_string(Transform,     "",    "Input transform matrix");
DEFINE_int   (Project,       0,     "Project input to N dimensions");
DEFINE_bool  (MeanNormalize, false, "Mean normalize input");

Matrix *in=0, *intx=0;
StringFile inlabel;
string inhdr, intxhdr, inlabelhdr;

struct FeatCorpus {
    typedef void (*FeatCB)(const char *, Matrix *, Matrix *, const char *, void *);
    static int feat_iter(const char *featdir, FeatCB cb, void *arg) { cb("", 0, in, "", arg); return 0; }
};

#include "ml/kmeans.h"
#include "ml/gmmem.h"
#include "ml/sample.h"
#include "ml/cluster.h"

namespace LFL {
DEFINE_int   (Initialize,     0,     "Initialize models");
DEFINE_int   (MaxIterations,  50,    "Initialize models");
DEFINE_double(CovarFloor,     1e-6,  "Model covariance floor");
DEFINE_double(PriorFloor,     -16,   "Model prior and mixture weight floor");
}; // namespace LFL

DEFINE_int   (choosen,        0,     "Choose every N from input");
DEFINE_bool  (normalize,      false, "Normalize values during Choose Every N");
DEFINE_bool  (pca,            false, "Perform principal components analysis on input");
DEFINE_bool  (cluster,        false, "Cluster input");
DEFINE_bool  (classifynn,     false, "Classify nearest neighbor");

void print(Matrix *m, int cols=0, StringFile *l=0, int *indexmap=0) {
    if (!cols) cols = m->N;
    MatrixRowIter(m) {
        string v; int ind = indexmap ? indexmap[i] : i;
        for (int j=0; j<cols; j++) v += StringPrintf("%f, ", m->row(ind)[j]);
        INFO(v, " (", (l && l->lines) ? l->line[ind] : "", ")");
    }
}

string tostrPloticus1D(const Matrix *in, StringFile *l, int *indexmap);

int writePloticus(string tofile, const Matrix *in, StringFile *l, int *indmap, string (*plot)(const Matrix *, StringFile *, int *)) {
    LocalFile f(tofile.c_str(), "w");
    string v = plot(in, l, indmap);
    return f.write(v.data(), v.size()) == v.size();
}

int *indexmap1D(const Matrix *in) {
    double *vals = new double [in->M];
    MatrixRowIter(in) vals[i] = in->row(i)[0];

    HMM::SortPair *sortMap = new HMM::SortPair[in->M];
    HMM::sortPairs(vals, 0, sortMap, in->M);

    int *ret = new int[in->M];
    MatrixRowIter(in) ret[i] = sortMap[i].ind;

    delete [] sortMap;
    delete [] vals;
    return ret;
}

extern "C" {
int main(int argc, const char *argv[]) {
    app->logfilename = StrCat(dldir(), "trainer.txt");
    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }

    if (!FLAGS_input.size()) FATAL("nothing to do");
    if (MatrixFile::read(inhdr, &in, FLAGS_input.c_str())) FATAL("MatrixFile::read ", strerror(errno));
    if (FLAGS_Transform.size() && MatrixFile::read(intxhdr, &intx, FLAGS_Transform.c_str())) FATAL("MatrixFile::read ", strerror(errno));

    if (FLAGS_inputlabel.size()) {
        if (StringFile::read(inlabelhdr, &inlabel, FLAGS_inputlabel.c_str())) FATAL("StringFile::read ", strerror(errno));
        if (in->M != inlabel.lines) FATAL("data/label mismatch ", in->M, " != ", inlabel.lines);
    }

    INFO("input:");
        
    if (intx) {
        Matrix input(in->M, in->N);
        if (FLAGS_MeanNormalize) MeanNormalizeRows(in, &input);
        else MatrixIter(&input) input.row(i)[j] = in->row(i)[j];

        Matrix projected(in->M, in->N);
        Matrix::mult(intx, &input, &projected, mTrnpB|mTrnpC);

        MatrixIter(in) in->row(i)[j] = projected.row(i)[j];
    }

    if (FLAGS_Project) in->N = FLAGS_Project;

    print(in, -1, &inlabel);

    if (FLAGS_choosen) {
        Matrix chosen(in->M / FLAGS_choosen, in->N);
        vector<string> label;
        MatrixRowIter(in) {
           if (rand() % FLAGS_choosen) continue;
           int ind = label.size();
           if (ind >= chosen.M) break;

           double sum = Vec<double>::sum(in->row(i), in->N);
           MatrixColIter(in) chosen.row(ind)[j] = in->row(i)[j] / (FLAGS_normalize ? sum : 1);

           label.push_back(inlabel.lines ? inlabel.line[i] : "");
        }
        chosen.M -= (chosen.M - label.size());

        string hdr="choosen output", matfile="choosen.matrix", labfile="choosenlabel.matrix";
        if (MatrixFile::write(hdr, &chosen, matfile.c_str(), matfile.c_str())) FATAL("error writing ", matfile);
        if (StringFile::write(hdr, label, labfile.c_str(), labfile.c_str())) FATAL("error writing ", labfile);

        INFO("wrote ", matfile, " and ", labfile);
        return 0;
    }

    if (FLAGS_pca) {
        Matrix projected(in->M, in->N);
        double *variance = (double*)alloca(in->N * sizeof(double));
        Matrix *pca = PCA(in, &projected, variance);
        int *indexmap = indexmap1D(&projected);

        string hdr1="PCA", fn1="PCA.matrix", hdr2="projected", fn2="projected.matrix";
        if (MatrixFile::write(hdr1, pca, fn1.c_str(), fn1.c_str())) FATAL("error writing ", fn1);
        if (MatrixFile::write(hdr2, &projected, fn2.c_str(), fn2.c_str())) FATAL("error writing ", fn2);

#if 0
        Matrix::print(pca, "PCA");
        INFO("variance:");
        Vector::print(variance, in->N);
        Matrix::print(&projected, "projected");
#endif

        INFO("output:");
        print(&projected, 1, &inlabel, indexmap);

        writePloticus(string(dldir()) + "plot.htm", &projected, &inlabel, indexmap, tostrPloticus1D);
    }

    string modeldir = dldir();

    if (FLAGS_cluster) features2cluster("", modeldir.c_str(), ClusterAlgorithm::KMeans);

    if (FLAGS_classifynn) {
        int lastiter; Matrix *model=0, *mcov=0;
        if ((lastiter = MatrixFile::ReadFile(modeldir.c_str(), "Cluster", "means", &model, 0)) < 0) FATAL("cluster means");
        if (lastiter > MatrixFile::ReadFile(modeldir.c_str(), "Cluster", "diagcovar", &mcov, 0)) FATAL("cluster diagcov");

        MatrixRowIter(in) {
            int minindex; double mindist;
            KMeans::nearest_neighbor(model, in->row(i), &minindex, &mindist);
            INFO("CLUSTER=", minindex, " LABEL=", inlabel.line[i]);
        }
    }

    return 0;
}
};

string tostrPloticus1D(const Matrix *in, StringFile *l, int *indexmap) {
    double min = INFINITY, max = -INFINITY;
    MatrixRowIter(in) {
        if (in->row(i)[0] < min) min = in->row(i)[0];
        if (in->row(i)[0] > max) max = in->row(i)[0];
    }

    string v = StringPrintf(
            "#proc areadef\n"
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
        v += StringPrintf("%s %f\n", l->line[ind], in->row(ind)[0]);
    }

    v += StringPrintf(
            "\n"
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

