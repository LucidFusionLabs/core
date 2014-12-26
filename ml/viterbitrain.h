/*
 * $Id: viterbitrain.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef __LFL_ML_VITERBITRAIN_H__
#define __LFL_ML_VITERBITRAIN_H__
namespace LFL {

DECLARE_FLAG(interactive, bool);

struct ViterbiTrain {
    AcousticModel::Compiled *model;
    typedef double (*ViterbF)(AcousticModel::Compiled *utterance, Matrix *observations, Matrix *path, int InitMax, double beamWidth, int flag);
    ViterbF viterbF;

    MatrixArchiveIn UttPathsIn;
    MatrixArchiveOut UttPathsOut;
    
    double totalprob, accumprob, beamwidth;
    int totalcount, accumcount;

    struct Mode { enum { Means=1, Cov=2, UpdateModel=3 }; }; int mode;
    bool UsePrior, UseTransition, FullVariance, MustReachFinal;
    int HMMFlag;

    struct Accum {
        virtual ~Accum() {}
        virtual void add_mean(double *features) = 0;
        virtual void add_cov (double *features) = 0;
        virtual void complete_mean() = 0;
        virtual void complete_cov () = 0;
        virtual void reset_mean() = 0;
        virtual void reset_cov () = 0;
        virtual Matrix *prior() const = 0;
        virtual Matrix *mean () const = 0;  
        virtual Matrix *cov  () const = 0;
        virtual int count() const = 0;
        virtual double prob() const = 0;
    } **accum;

    typedef Accum *(*AccumFactory)(GMM *m, bool FullVariance, const char *name);

    struct KMeansAccum : public Accum {
        KMeans *means;
        SampleCovariance *covs;
        KMeansInit *init;
        int stage, feats;

        ~KMeansAccum() { delete covs; delete means; delete init; }
        KMeansAccum(KMeans *KM) : means(KM), covs(new SampleCovariance(KM->means)), init(0), stage(means->K == 1 ? 2 : 0), feats(0) {}
        static Accum *create(GMM *m, bool, const char *name) { return new KMeansAccum(new KMeans(m->mean.M, m->mean.N)); }
        
        void add_mean(double *feature) {
            if      (stage == 0) feats++;
            else if (stage == 1) init->add_feature(feature);
            else if (stage == 2) means->add_feature(feature);
        }
        void complete_mean() {
            if      (stage == 0) { init=new KMeansInit(means, feats); stage++; }
            else if (stage == 1) { stage++; }
            else if (stage == 2) { means->complete(); }
        }
        void add_cov (double *feature) { SampleCovariance::add_feature(feature, covs); }
        void complete_cov () { covs->complete(); }
        void reset_mean() { means->reset(); }
        void reset_cov () { covs->reset(); }
        Matrix *mean() const { return means->means; }
        Matrix *cov () const { return covs->diagnol; }
        Matrix *prior() const { return means->prior; }
        int count() const { int ret=0; for (int i=0; i<means->K; i++) ret+=means->count[i]; return ret; }
        double prob() const { return means->totaldist; }
    };

    struct GMMAccum : public Accum {
        GMM mixture;
        GMMEM EM;

        GMMAccum(Matrix *M, Matrix *V, Matrix *P, bool FullVariance, const char *Name) : mixture(M, V, P), EM(&mixture, FullVariance, Name) { EM.mode = GMMEM::Mode::Means; }
        static Accum *create(GMM *m, bool FullVariance, const char *name) { return new GMMAccum(&m->mean, &m->diagcov, &m->prior, FullVariance, name); }

        void add_mean(double *feature) { EM.add_feature(feature); } 
        void add_cov (double *feature) { EM.add_feature(feature); }
        void complete_mean() { EM.complete(); EM.mode=GMMEM::Mode::Cov; }
        void complete_cov () { EM.complete(); EM.mode=GMMEM::Mode::Means; }
        void reset_mean() {}
        void reset_cov () {}
        Matrix *mean() const { return &EM.mixture->mean; }
        Matrix *cov () const { return &EM.mixture->diagcov; }
        Matrix *prior() const { return &EM.mixture->prior; }
        int count() const { return EM.count; }
        double prob() const { return isnan(EM.totalprob) ? 0 : EM.totalprob; }
    };

    ViterbiTrain(AcousticModel::Compiled *m, double beamWidth, bool usePrior=0, bool useTransition=0, bool fullVariance=0, double mustReachFinal=0, AccumFactory accumCreate=GMMAccum::create)
        : model(m), mode(0), UsePrior(usePrior), UseTransition(useTransition), FullVariance(fullVariance), MustReachFinal(mustReachFinal), HMMFlag(0), viterbF(0), accumprob(0), accumcount(0), beamwidth(beamWidth), accum(new Accum*[model->states])
    {
        if (UsePrior) HMMFlag |= AcousticHMM::Flag::UsePrior;
        if (UseTransition) HMMFlag |= AcousticHMM::Flag::UseTransit;

        for (int i=0; i<model->states; i++) {
            AcousticModel::State *s = &model->state[i];
            if (s->val.emission_index != i) FATAL("ViterbiTrain emission_index mismatch ", s->val.emission_index, " != ", i);

            accum[i] = accumCreate(&s->emission, FullVariance, s->name.c_str());
        }
        reset();
    }
    
    ~ViterbiTrain() { for(int i=0; i<model->states; i++) delete accum[i]; delete [] accum; }

    void reset(bool free=0) { 
        for (int i=0; i<model->states; i++) {
            accum[i]->reset_mean();
            accum[i]->reset_cov();
        }
        totalprob=0; totalcount=0;
    }

    void run(const char *modeldir, const char *featdir, int write, int iterO, int iterI, string uttPathsInFile, string uttPathsOutFile) {
        /* start */
        run_start(modeldir, write, iterO, iterI, uttPathsInFile, uttPathsOutFile);

        /* thunk iteration */
        int count = 0;
        INFO("corpus = ", UttPathsIn.file ? "PathCorpus" : "UttCorpus");
        if (!UttPathsIn.file) count = FeatCorpus::feat_iter(featdir, ViterbiTrain::add_features, this);
        else count = PathCorpus::path_iter(featdir, &UttPathsIn, ViterbiTrain::add_path, this);

        /* finish */
        run_finish(iterO, iterI, count);
    }

    void run_start(const char *modeldir, int write, int iterO, int iterI, string uttPathsInFile, string uttPathsOutFile) {
        INFO("enter ViterbiTrain iteration ", iterO, "-", iterI, " (mode=", mode, ")");
        string name = string(modeldir) + StringPrintf("vp.%04d.matlist", iterO);

        if (uttPathsInFile.size()) UttPathsIn.Open(modeldir + uttPathsInFile);
        if (uttPathsOutFile.size()) UttPathsOut.Open(modeldir + uttPathsOutFile);

        /* default output */
        if (write && !UttPathsOut.file && UttPathsIn.Filename() != name) {
            if (UttPathsOut.Open(name)) ERROR("open: ", name, ": ", strerror(errno));
        }

        /* default input */
        if (!write && !UttPathsIn.file) {
            if (UttPathsIn.Open(name)) FATAL("open ", name, " failed");
        }
    }

    void run_finish(int iterO, int iterI, int count) {
        /* close */
        UttPathsIn.Close();
        UttPathsOut.Close();

        INFO("exit ViterbiTrain iteration ", iterO, "-", iterI, ", utterances = ", count);
    }

    void add_features(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript) {
        /* build utterance model from transcript */
        AcousticModel::Compiled *hmm = AcousticModel::fromUtterance(model, transcript, UseTransition);
        return add_features(hmm, fn, MFCC, features, transcript);
    }
    void add_features(AcousticModel::Compiled *hmm, const char *fn, Matrix *MFCC, Matrix *features, const char *transcript) {
        if (!hmm) return DEBUG("utterance construct failed: ", transcript);
        if (!dim_check("VT", features->N, hmm->state[0].emission.mean.N)) return;

        MatrixIter(features) if (!isfinite(features->row(i)[j]))
            return ERRORf("feat '%s' (%s) has [%d,%d] = %f", transcript, fn, i, j, features->row(i)[j]);

        /* find viterbi path */
        Matrix viterbi(features->M, 1); Timer vtime;
        double vprob = viterbF(hmm, features, &viterbi, 2, beamwidth, HMMFlag);
        if (!isfinite(vprob)) { ERROR(vprob, " vprob for '", transcript, "', skipping (processed=", totalcount, ")"); delete hmm; return; }

        if (MustReachFinal) {
            int finalstate = viterbi.lastrow()[0];
            if (finalstate < hmm->states-2) { ERRORf("final state not reached (%d < %d) for '%s', skipping (processed=%d)", finalstate, hmm->states-2, transcript, totalcount); delete hmm; return; }
        }

        /* write & process path */
        if (UttPathsOut.file) PathCorpus::add_path(&UttPathsOut, &viterbi, fn);
        add_path(hmm, &viterbi, vprob, vtime.time(), MFCC, features, transcript);
        delete hmm;
    }
    static void add_features(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript, void *arg) { return ((ViterbiTrain*)arg)->add_features(fn, MFCC, features, transcript); }

    void add_path(AcousticModel::Compiled *hmm, Matrix *viterbi, double vprob, double vtime, Matrix *MFCC, Matrix *features, const char *transcript) {
        if (viterbi->M != features->M) { ERROR("viterbi length mismatch ", viterbi->M, " != ", features->M); }

        /* optionally build utterance model from transcript */
        AcousticModel::Compiled *myhmm = 0;
        if (!hmm) hmm = myhmm = AcousticModel::fromUtterance(model, transcript, UseTransition);
        if (!hmm) return ERROR("utterance decode failed: ", transcript);

        /* visualize viterbi path as spectogram segmentation */
        if (FLAGS_lfapp_video) Decoder::visualizeFeatures(hmm, MFCC, viterbi, vprob, vtime, FLAGS_interactive);

        /* for each viterbi state occupancy and observation */
        MatrixRowIter(viterbi) {

            /* get emission index from viterbi occupancy index in utterance model */
            int vind = viterbi->row(i)[0];
            if (vind < 0 || vind >= hmm->states) FATAL("oob vind ", vind);
            int emission_index = hmm->state[vind].val.emission_index;

            /* count */
            if      (mode == Mode::Means) accum[emission_index]->add_mean(features->row(i));
            else if (mode == Mode::Cov)   accum[emission_index]->add_cov (features->row(i));
        }

        if (mode == Mode::Means) { totalprob += vprob; totalcount++; }
        delete myhmm;
    }
    static void add_path(AcousticModel::Compiled *hmm, Matrix *viterbi, double vprob, double vtime, Matrix *MFCC, Matrix *features, const char *transcript, void *arg) { return ((ViterbiTrain*)arg)->add_path(hmm, viterbi, vprob, vtime, MFCC, features, transcript); }

    void complete() {
        if (mode == Mode::Means) { accumprob=0; accumcount=0; }

        for (int i=0; i<model->states; i++) {
            if (mode == Mode::Means) {
                double prob = accum[i]->prob();
                accumprob += prob;
                accumcount++;
                if (0) INFO("VT complete ", model->state[i].name, " accumprob = ", prob);

                accum[i]->complete_mean();
            }
            else if (mode == Mode::Cov) {
                accum[i]->complete_cov();
            }
            else if (mode == Mode::UpdateModel) {
                AcousticModel::State *s = &model->state[i];

                s->emission.mean.AssignL(accum[i]->mean());

                s->emission.diagcov.AssignL(accum[i]->cov());
                if (accum[i]->prior()) s->emission.prior.AssignL(accum[i]->prior());
                s->emission.computeNorms();

                s->val.samples = accum[i]->count();
            }
        }
        if (FLAGS_lfapp_cuda && mode == Mode::UpdateModel) AcousticModel::toCUDA(model);
    }
};

}; // namespace LFL
#endif // __LFL_ML_VITERBITRAIN_H__
