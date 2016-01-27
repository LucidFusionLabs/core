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

#ifndef LFL_ML_VITERBITRAIN_H__
#define LFL_ML_VITERBITRAIN_H__
namespace LFL {

DECLARE_FLAG(interactive, bool);

struct ViterbiTrain {
  struct Mode { enum { Means=1, Cov=2, UpdateModel=3 }; };

  struct Accum {
    virtual ~Accum() {}
    virtual void AddMean(double *features) = 0;
    virtual void AddCov (double *features) = 0;
    virtual void CompleteMean() = 0;
    virtual void CompleteCov () = 0;
    virtual void ResetMean() = 0;
    virtual void ResetCov () = 0;
    virtual Matrix *Prior() const = 0;
    virtual Matrix *Mean () const = 0;  
    virtual Matrix *Cov  () const = 0;
    virtual int Count() const = 0;
    virtual double Prob() const = 0;
  };

  struct KMeansAccum : public Accum {
    KMeans *means;
    SampleCovariance *covs;
    KMeansInit *init;
    int stage, feats;

    ~KMeansAccum() { delete covs; delete means; delete init; }
    KMeansAccum(KMeans *KM) : means(KM), covs(new SampleCovariance(KM->means)), init(0), stage(means->K == 1 ? 2 : 0), feats(0) {}
    static Accum *Create(GMM *m, bool, const char *name) { return new KMeansAccum(new KMeans(m->mean.M, m->mean.N)); }

    void AddMean(double *feature) {
      if      (stage == 0) feats++;
      else if (stage == 1) init->AddFeature(feature);
      else if (stage == 2) means->AddFeature(feature);
    }
    void CompleteMean() {
      if      (stage == 0) { init=new KMeansInit(means, feats); stage++; }
      else if (stage == 1) { stage++; }
      else if (stage == 2) { means->Complete(); }
    }
    void AddCov(double *feature) { covs->AddFeature(feature); }
    void CompleteCov () { covs->Complete(); }
    void ResetMean() { means->Reset(); }
    void ResetCov () { covs->Reset(); }
    Matrix *Mean() const { return means->means; }
    Matrix *Cov () const { return covs->diagnol; }
    Matrix *Prior() const { return means->prior; }
    int Count() const { int ret=0; for (int i=0; i<means->K; i++) ret+=means->count[i]; return ret; }
    double Prob() const { return means->totaldist; }
  };

  struct GMMAccum : public Accum {
    GMM mixture;
    GMMEM EM;

    GMMAccum(Matrix *M, Matrix *V, Matrix *P, bool full_variance, const char *Name) : mixture(M, V, P), EM(&mixture, full_variance, Name) { EM.mode = GMMEM::Mode::Means; }
    static Accum *Create(GMM *m, bool full_variance, const char *name) { return new GMMAccum(&m->mean, &m->diagcov, &m->prior, full_variance, name); }

    void AddMean(double *feature) { EM.AddFeature(feature); } 
    void AddCov (double *feature) { EM.AddFeature(feature); }
    void CompleteMean() { EM.Complete(); EM.mode=GMMEM::Mode::Cov; }
    void CompleteCov () { EM.Complete(); EM.mode=GMMEM::Mode::Means; }
    void ResetMean() {}
    void ResetCov () {}
    Matrix *Mean() const { return &EM.mixture->mean; }
    Matrix *Cov () const { return &EM.mixture->diagcov; }
    Matrix *Prior() const { return &EM.mixture->prior; }
    int Count() const { return EM.count; }
    double Prob() const { return isnan(EM.totalprob) ? 0 : EM.totalprob; }
  };

  typedef Accum *(*AccumFactory)(GMM *m, bool full_variance, const char *name);
  typedef double (*ViterbF)(AcousticModel::Compiled *utterance, Matrix *observations, Matrix *path, int InitMax, double beamWidth, int flag);

  int mode;
  AcousticModel::Compiled *model;
  ViterbF viterbF;

  MatrixArchiveInputFile utt_paths_in;
  MatrixArchiveOutputFile utt_paths_out;

  double totalprob, accumprob, beamwidth;
  int totalcount, accumcount;

  bool use_prior, use_transition, full_variance, must_reach_final;
  int hmm_flag;
  Accum **accum;

  ~ViterbiTrain() { for(int i=0; i<model->states; i++) delete accum[i]; delete [] accum; }
  ViterbiTrain(AcousticModel::Compiled *m, double bw, bool up=0, bool ut=0, bool fv=0, double mrf=0, AccumFactory accum_create=GMMAccum::Create)
    : model(m), mode(0), use_prior(up), use_transition(ut), full_variance(fv), must_reach_final(mrf), hmm_flag(0), viterbF(0), accumprob(0), accumcount(0), beamwidth(bw), accum(new Accum*[model->states])
  {
    if (use_prior)      hmm_flag |= AcousticHMM::Flag::UsePrior;
    if (use_transition) hmm_flag |= AcousticHMM::Flag::UseTransit;

    for (int i=0; i<model->states; i++) {
      AcousticModel::State *s = &model->state[i];
      if (s->val.emission_index != i) FATAL("ViterbiTrain emission_index mismatch ", s->val.emission_index, " != ", i);

      accum[i] = accum_create(&s->emission, full_variance, s->name.c_str());
    }
    Reset();
  }

  void Reset(bool free=0) { 
    for (int i=0; i<model->states; i++) {
      accum[i]->ResetMean();
      accum[i]->ResetCov();
    }
    totalprob=0; totalcount=0;
  }

  void Run(const char *modeldir, const char *featdir, int write, int iterO, int iterI, string utt_paths_in_file, string utt_paths_out_file) {
    /* start */
    RunStart(modeldir, write, iterO, iterI, utt_paths_in_file, utt_paths_out_file);

    /* thunk iteration */
    int count = 0;
    INFO("corpus = ", utt_paths_in.file ? "PathCorpus" : "UttCorpus");
    if (!utt_paths_in.file) count = FeatCorpus::FeatIter(featdir, bind(&ViterbiTrain::AddFeatures, this, _1, _2, _3, _4));
    else count = PathCorpus::PathIter(featdir, &utt_paths_in, bind(&ViterbiTrain::AddPath, this, _1, _2, _3, _4, _5, _6, _7));

    /* finish */
    RunFinish(iterO, iterI, count);
  }

  void RunStart(const char *modeldir, int write, int iterO, int iterI, string utt_paths_in_file, string utt_paths_out_file) {
    INFO("enter ViterbiTrain iteration ", iterO, "-", iterI, " (mode=", mode, ")");
    string name = string(modeldir) + StringPrintf("vp.%04d.matlist", iterO);

    if (utt_paths_in_file .size()) utt_paths_in .Open(modeldir + utt_paths_in_file);
    if (utt_paths_out_file.size()) utt_paths_out.Open(modeldir + utt_paths_out_file);

    /* default output */
    if (write && !utt_paths_out.file && utt_paths_in.Filename() != name) {
      if (utt_paths_out.Open(name)) ERROR("open: ", name, ": ", strerror(errno));
    }

    /* default input */
    if (!write && !utt_paths_in.file) {
      if (utt_paths_in.Open(name)) FATAL("open ", name, " failed");
    }
  }

  void RunFinish(int iterO, int iterI, int count) {
    /* close */
    utt_paths_in.Close();
    utt_paths_out.Close();

    INFO("exit ViterbiTrain iteration ", iterO, "-", iterI, ", utterances = ", count);
  }

  void AddFeatures(const char *fn, Matrix *MFCC, Matrix *features, const char *transcript) {
    /* build utterance model from transcript */
    AcousticModel::Compiled *hmm = AcousticModel::FromUtterance(model, transcript, use_transition);
    return AddUtterance(hmm, fn, MFCC, features, transcript);
  }

  void AddUtterance(AcousticModel::Compiled *hmm, const char *fn, Matrix *MFCC, Matrix *features, const char *transcript) {
    if (!hmm) return DEBUG("utterance construct failed: ", transcript);
    if (!DimCheck("VT", features->N, hmm->state[0].emission.mean.N)) return;

    MatrixIter(features) if (!isfinite(features->row(i)[j]))
      return ERRORf("feat '%s' (%s) has [%d,%d] = %f", transcript, fn, i, j, features->row(i)[j]);

    /* find viterbi path */
    Matrix viterbi(features->M, 1); Timer vtime;
    double vprob = viterbF(hmm, features, &viterbi, 2, beamwidth, hmm_flag);
    if (!isfinite(vprob)) { ERROR(vprob, " vprob for '", transcript, "', skipping (processed=", totalcount, ")"); delete hmm; return; }

    if (must_reach_final) {
      int finalstate = viterbi.lastrow()[0];
      if (finalstate < hmm->states-2) { ERRORf("final state not reached (%d < %d) for '%s', skipping (processed=%d)", finalstate, hmm->states-2, transcript, totalcount); delete hmm; return; }
    }

    /* write & process path */
    if (utt_paths_out.file) PathCorpus::AddPath(&utt_paths_out, &viterbi, fn);
    AddPath(hmm, &viterbi, vprob, vtime.GetTime(), MFCC, features, transcript);
    delete hmm;
  }

  void AddPath(AcousticModel::Compiled *hmm, Matrix *viterbi, double vprob, Time vtime, Matrix *MFCC, Matrix *features, const char *transcript) {
    if (viterbi->M != features->M) { ERROR("viterbi length mismatch ", viterbi->M, " != ", features->M); }

    /* optionally build utterance model from transcript */
    AcousticModel::Compiled *myhmm = 0;
    if (!hmm) hmm = myhmm = AcousticModel::FromUtterance(model, transcript, use_transition);
    if (!hmm) return ERROR("utterance decode failed: ", transcript);

    /* visualize viterbi path as spectogram segmentation */
    if (FLAGS_lfapp_video) Decoder::VisualizeFeatures(hmm, MFCC, viterbi, vprob, vtime, FLAGS_interactive);

    /* for each viterbi state occupancy and observation */
    MatrixRowIter(viterbi) {

      /* get emission index from viterbi occupancy index in utterance model */
      int vind = viterbi->row(i)[0];
      if (vind < 0 || vind >= hmm->states) FATAL("oob vind ", vind);
      int emission_index = hmm->state[vind].val.emission_index;

      /* count */
      if      (mode == Mode::Means) accum[emission_index]->AddMean(features->row(i));
      else if (mode == Mode::Cov)   accum[emission_index]->AddCov (features->row(i));
    }

    if (mode == Mode::Means) { totalprob += vprob; totalcount++; }
    delete myhmm;
  }

  void Complete() {
    if (mode == Mode::Means) { accumprob=0; accumcount=0; }

    for (int i=0; i<model->states; i++) {
      if (mode == Mode::Means) {
        double prob = accum[i]->Prob();
        accumprob += prob;
        accumcount++;
        if (0) INFO("VT complete ", model->state[i].name, " accumprob = ", prob);

        accum[i]->CompleteMean();
      }
      else if (mode == Mode::Cov) {
        accum[i]->CompleteCov();
      }
      else if (mode == Mode::UpdateModel) {
        AcousticModel::State *s = &model->state[i];

        s->emission.mean.AssignL(accum[i]->Mean());

        s->emission.diagcov.AssignL(accum[i]->Cov());
        if (accum[i]->Prior()) s->emission.prior.AssignL(accum[i]->Prior());
        s->emission.ComputeNorms();

        s->val.samples = accum[i]->Count();
      }
    }
    if (FLAGS_lfapp_cuda && mode == Mode::UpdateModel) AcousticModel::ToCUDA(model);
  }
};

}; // namespace LFL
#endif // LFL_ML_VITERBITRAIN_H__
