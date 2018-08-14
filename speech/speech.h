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

#ifndef LFL_CORE_SPEECH_SPEECH_H__
#define LFL_CORE_SPEECH_SPEECH_H__
namespace LFL {

DECLARE_string(feat_type);
DECLARE_bool(feat_dither);
DECLARE_bool(feat_preemphasis);
DECLARE_double(feat_preemphasis_filter);
DECLARE_double(feat_minfreq);
DECLARE_double(feat_maxfreq);
DECLARE_int(feat_window);
DECLARE_int(feat_hop);
DECLARE_int(feat_melbands);
DECLARE_int(feat_cepcoefs);
DECLARE_bool(triphone_model);
DECLARE_bool(speech_recognition_debug);

struct Phoneme { 
  enum {
#define LFL_LANGUAGE_ENGLISH
#undef  XX
#define XX(phone) phone,
#include "core/speech/phones.h"
  };
  static const char *Name(char phoneId);
  static char Id(const char *phone, int len=0);
};
#define PhonemeIter() for (int phone=0; phone<LFL_PHONES; phone++)

struct PronunciationDict {
  typedef map<string, int> Map;
  Map word_pronunciation;
  StringAlloc val;

  static PronunciationDict *Instance(AssetLoading*);
  static int ReadDictionary(StringIter *in, PronunciationDict *out);
  static int ReadPronunciation(const char *in, int len, char *phoneIdOut, char *accentOut, int outlen);

  const char *Pronounce(const char *word);
  int Pronounce(const char *utterance, const char **words, const char **accents, int *phones, int max);
};

struct Features {
  static int filter_zeroth, deltas, deltadeltas, mean_normalization, variance_normalization, lpccoefs, barkbands;
  static double rastaB[5], rastaA[2];

  static vector<double> LifterMatrixROSA(int n, double L, bool inverse=false);
  static vector<double> LifterMatrixHTK(int n, double L, bool inverse=false);
  static unique_ptr<Matrix> EqualLoudnessCurve(int outrows, double max);
  static unique_ptr<Matrix> FFT2Bark(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate);
  static unique_ptr<Matrix> FFT2Mel(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate);
  static unique_ptr<Matrix> Mel2FFT(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate);
  static unique_ptr<Matrix> PLP(const RingSampler::Handle *in, Matrix *out=0, vector<StatefulFilter> *rastaFilter=0, Allocator *alloc=0);
  static unique_ptr<Matrix> MFCC(const RingSampler::Handle *in, Matrix *out=0, Allocator *alloc=0);
  static unique_ptr<RingSampler> InvPLP(const Matrix *in, int samplerate, Allocator *alloc=0);
  static unique_ptr<RingSampler> InvMFCC(const Matrix *in, int samplerate, const Matrix *f0=0);

  static unique_ptr<Matrix> FilterZeroth(unique_ptr<Matrix> features);
  static void MeanAndVarianceNormalization(int D, double *feat, const double *mean, const double *var);
  static void DeltaCoefficients(int D, const double *n2, double *f, const double *p2);
  static void DeltaDeltaCoefficients(int D, const double *n3, const double *n1, double *f,  const double *p1, const double *p3);
  static void PatchDeltaCoefficients(int D, const double *in, double *out1, double *out2); 
  static void PatchDeltaDeltaCoefficients(int D, const double *in, double *out1, double *out2, double *out3);
  static unique_ptr<Matrix> DeltaCoefficients(unique_ptr<Matrix> features, bool deltadelta=true);

  struct Flag { enum { Full=1, Storable=2 }; int f; };
  static unique_ptr<Matrix> FromAsset(SoundAsset *wav, int flag);
  static unique_ptr<Matrix> FromFeat(unique_ptr<Matrix> features, int flag);
  static unique_ptr<Matrix> FromFeat(unique_ptr<Matrix> features, int flag, bool FilterZeroth, bool Deltas, bool DeltaDeltas, bool MeanNorm, bool VarNorm);
  static unique_ptr<Matrix> FromBuf(const RingSampler::Handle *in, Matrix *out=0, vector<StatefulFilter> *filter=0, Allocator *alloc=0);
  static unique_ptr<RingSampler> Reverse(const Matrix *in, int samplerate, const Matrix *f0=0, Allocator *alloc=0);
  static int Dimension();
};

struct AcousticModel {
  struct State;
  struct StateCollection { 
    virtual ~StateCollection() {}
    virtual Matrix *TiedStates() { return 0; }
    virtual Matrix *PhoneTx() { return 0; }

    virtual int GetStateCount() = 0;
    virtual State *GetState(unsigned stateID) = 0;

    struct Iterator { int k; State *v; char impl[2048]; bool done; };
    virtual void BeginState(Iterator *iter) = 0;
    virtual void NextState(Iterator *iter) = 0;
  };
  struct Compiled;

  struct State {
    Allocator *alloc;
    string name;
    double prior, txself;
    Matrix transition;
    GMM emission;
    struct Val { int samples, emission_index; } val;

    ~State() {}
    State() : alloc(Singleton<NullAllocator>::Set()), prior(-INFINITY), txself(-INFINITY) { val.samples=val.emission_index=0; }
    unsigned Id() { return fnv32(name.c_str()); }
    void AssignPtr(State *s) {
      if (!s) FATAL("assign ", s);
      name = s->name;
      prior = s->prior;
      txself = s->txself;
      transition.AssignDataPtr(s->transition.M, s->transition.N, s->transition.m);
      emission.AssignDataPtr(&s->emission);
      val = s->val;
    }

    static double *Transit(Compiled *model, Matrix *trans, State *Rstate, unsigned *to=0);
    static double *Transit(Matrix *trans, unsigned Rstate);
    static int TransitionSort(const void *a, const void *b);
    static void SortTransitionMap(Matrix *trans);
    static void SortTransitionMap(double *trans, int M);
    static int LocalizeTransitionMap(Matrix *transI, Matrix *transO, Compiled *scope, int scopeState=0, int minSamples=0, Compiled *dubdref=0, const char *srcname=0);
    static unsigned Tied(Matrix *tiedstates, int pp, int p, int pn, int k);
    static void Assign(State *out, int *outind, int outsize, StateCollection *ps, int pp, int p, int pn, int states, int val=0); 
    static bool TriphoneConnect(int ap, int a, int an, int bp, int b, int bn) {
      bool asil = !ap && !a && !an, bsil = !bp && !b && !bn;
      return (bp == a && b == an && !bsil && !asil) || (asil && !bp && !bsil) || (bsil && !an && !asil);
    }
  };

  struct Compiled : public StateCollection {
    struct NameCB : public LFL::NameCB {
      Compiled *model;
      NameCB(Compiled *M) : model(M) {}
      string Name(int ind) {
        if (ind < 0 || ind > model->state.size()) FATAL("oob ind ", ind);
        return model->state[ind].name;
      }
    };

    vector<State> state;
    Matrix *tiedstates=0;
    unique_ptr<Matrix> phonetx;
    NameCB nameCB;

    Compiled(int N=0) : nameCB(this) { if (N) Open(N); }
    void Open(int N) { state = vector<State>(N, State()); }

    int GetStateCount() { return state.size(); }
    State *GetState(unsigned stateID) { return 0; }
    void BeginState(Iterator *iter) { iter->done = 0; *reinterpret_cast<int*>(iter->impl) = 0; NextState(iter); }
    void NextState(Iterator *iter) {
      int *impl = reinterpret_cast<int*>(iter->impl);
      if (*impl >= state.size()) { iter->done = 1; return; }
      iter->k = (*impl)++;
      iter->v = &state[iter->k];
    }
    Matrix *TiedStates() { return tiedstates; }
    Matrix *PhoneTx() { return phonetx.get(); }
  };

  const static int StatesPerPhone=3;
  static int ParseName(const string &name, int *phonemeStateOut=0, int *prevPhoneOut=0, int *nextPhoneOut=0);

  static string Name(int phoneme, int phonemeState) { return StringPrintf("Model_%s_State_%02d", Phoneme::Name(phoneme), phonemeState); }

  static string Name(int pp, int p, int np, int ps) {
    if (!pp && !np) return Name(p, ps);
    return StringPrintf("Model_%s_%s_%s_State_%02d", Phoneme::Name(pp), Phoneme::Name(p), Phoneme::Name(np), ps);
  }

  static bool IsSilence(const string &name) { return name == "Model_SIL_State_00"; }
  static bool IsSilence(unsigned id)        { return id == fnv32("Model_SIL_State_00"); }

  static bool TriphoneSort(const string &x, const string &y) {
    int xn, xp, yn, yp, cmp = ParseName(x.c_str(), &xp, &xn) - ParseName(y.c_str(), &yp, &yn);
    if (cmp) return cmp < 0;
    if ((cmp = xp - yp)) return cmp < 0;
    return (xn - yn) < 0;
  }

  static unique_ptr<Compiled> FullyConnected(Compiled *model);
  static unique_ptr<Compiled> FromUtterance(AssetLoading *loader, Compiled *model, const char *transcript, bool UseTransit);
  static unique_ptr<Compiled> FromUtterance3(AssetLoading *loader, Compiled *model, const char *transcript, bool UseTransit);
  static unique_ptr<Compiled> FromUtterance1(AssetLoading *loader, Compiled *model, const char *transcript, bool UseTransit);
  static unique_ptr<Compiled> FromModel1(StateCollection *model, bool rewriteTransitions);

  static void LoadFlags(const char *flags);
  static string Flags();

  static int Write(StateCollection *, const char *name, const char *dir, int iteration, int minSamples);
  static int ToCUDA(AcousticModel::Compiled *);
};

struct AcousticModelFile : public AcousticModel::Compiled {
  unique_ptr<Matrix> initial, mean, covar, prior, transit, tied;
  unique_ptr<vector<string>> names;
  HashMatrix map;
  AcousticModelFile() : map(0, 4) {}
  ~AcousticModelFile() { Reset(); }
  void Reset() { initial.reset(); mean.reset(); covar.reset(); prior.reset(); transit.reset(); tied.reset(); names.reset(); map.map.reset(); }
  AcousticModel::State *GetState(unsigned hash) { double *he = GetHashEntry(hash); return he ? &state[int(he[1])] : 0; }
  double *GetHashEntry(unsigned hash) { return map.Get(hash); }
  Matrix *TiedStates() { return tied.get(); }
  int Open(FileSystem*, const char *name, const char *dir, int lastiter=-1, bool rebuildTransit=0);
};

struct AcousticModelBuilder : public AcousticModel::StateCollection {
  typedef map<int, unique_ptr<AcousticModel::State>> Map;
  Map statemap;

  int GetStateCount() { return statemap.size(); }
  AcousticModel::State *GetState(unsigned stateID) { Map::iterator i = statemap.find(stateID); return i != statemap.end() ? (*i).second.get() : 0; }

  void BeginState(Iterator *iter) { iter->done=0; Map::iterator *i = reinterpret_cast<Map::iterator*>(iter->impl); *i = statemap.begin(); AssignKV(iter, i); }
  void NextState (Iterator *iter) {               Map::iterator *i = reinterpret_cast<Map::iterator*>(iter->impl); (*i)++;                AssignKV(iter, i); }

  void AssignKV(Iterator *iter, Map::iterator *i) {
    if (*i == statemap.end()) iter->done = 1;
    else { iter->k = (**i).first; iter->v = (**i).second.get(); }
  }

  void Add(unique_ptr<AcousticModel::State> s, int id) { statemap[id] = move(s); };
  void Add(unique_ptr<AcousticModel::State> s) { Add(move(s), s->Id()); }
};

struct AcousticHMM {
  struct Flag { enum { UsePrior=1, UseTransit=2, UseContentIndependentTransit=4, Visualize=8, Interactive=16 }; };

  struct TransitMap : public HMM::TransitMap {
    AcousticModel::Compiled *model;
    bool use_transition_prob;
#define TransitCols 3
#define TC_Self 0
#define TC_Edge 1
#define TC_Cost 2
    TransitMap(AcousticModel::Compiled *M, bool UT) : model(M), use_transition_prob(UT) {}

    void Begin(Iterator *iter, HMM::ActiveState *active, HMM::ActiveState::Iterator *Lstate) {
      iter->impl1 = Void(&model->state[Lstate->index / active->NBest]);
      iter->impl2 = 0;
      iter->state2 = 0;
      iter->done = 0;
      Next(iter);
    }
    void Next(Iterator *iter) {
      AcousticModel::State *s = static_cast<AcousticModel::State*>(iter->impl1);
      if (iter->impl2 >= s->transition.M) { iter->done=1; return; }
      double *tr = s->transition.row(iter->impl2++);
      iter->state = tr[TC_Edge];
      iter->cost = use_transition_prob ? tr[TC_Cost] : 0;
    }
    int Id(int Lstate) { return model->state[Lstate].val.emission_index; }
  };

  struct EmissionArray : public HMM::Emission {
    AcousticModel::Compiled *model;
    Matrix *observed;
    bool use_prior_prob;
    int time_index;

    Allocator *alloc;
    double *emission;

    ~EmissionArray() { if (alloc) alloc->Free(emission); }
    EmissionArray(AcousticModel::Compiled *M, Matrix *Observed, bool UsePrior, Allocator *Alloc=0) : model(M),
    observed(Observed), use_prior_prob(UsePrior), time_index(0), alloc(Alloc?Alloc:Singleton<MallocAllocator>::Set()),
    emission(static_cast<double*>(alloc->Malloc(sizeof(double)*model->state.size()))) {}

    double *Observation(int t) { return observed->row(t); }
    int Observations() { return observed->M; }
    double Prior(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) { return use_prior_prob ? model->state[state->index].prior : 0; }
    double *Posterior(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) { return 0; }
    double Prob(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) { return emission[state->index]; }
    double Prob(TransitMap::Iterator *iter) { return emission[iter->state]; }
    void Calc(HMM::ActiveState *active, int t) { time_index=t; Calc(model, active, time_index, observed->row(t), emission); }
    void CalcNext(HMM::ActiveState *active, HMM::TransitMap *, int t) { return Calc(active, t); /* needs BeamWidth >= NumStates */ }
    static void Calc(AcousticModel::Compiled *model, HMM::ActiveState *active, int time_index, const double *observed, double *emission, double *posterior=0, double *cudaposterior=0);
  };

  struct EmissionMatrix : public HMM::Emission {
    AcousticModel::Compiled *model;
    Matrix *observed;
    bool use_prior_prob;
    int K, time_index;

    Allocator *alloc;
    Matrix emission, emissionPosterior;
    bool *calcd;
    double *cudaPosterior;

    ~EmissionMatrix() { if (!alloc) return; alloc->Free(calcd); alloc->Free(cudaPosterior); }
    EmissionMatrix(AcousticModel::Compiled *M, Matrix *Observed, bool UsePrior, Allocator *Alloc=0) : model(M), observed(Observed), use_prior_prob(UsePrior),
    K(model->state[0].emission.mean.M), time_index(0), alloc(Alloc?Alloc:Singleton<MallocAllocator>::Set()),
    emission(observed->M, model->state.size(), 0, 0, Alloc), emissionPosterior(observed->M, model->state.size()*K, 0, 0, Alloc),
    calcd(static_cast<bool*>(alloc->Malloc(observed->M))), cudaPosterior(static_cast<double*>(alloc->Malloc(model->state.size()*K*sizeof(double))))
    { memset(calcd, 0, observed->M);  }

    double *Observation(int t) { return observed->row(t); }
    int Observations() { return observed->M; }
    double Prior(HMM::ActiveState *active, HMM::ActiveState::Iterator *iter) { return use_prior_prob ? model->state[iter->index].prior : 0; }
    double Prob(HMM::ActiveState *active, HMM::ActiveState::Iterator *iter) { return emission.row(time_index)[iter->index]; }
    double *Posterior(HMM::ActiveState *active, HMM::ActiveState::Iterator *iter) { return emissionPosterior.row(time_index) + iter->index*K; }
    double Prob(TransitMap::Iterator *iter) { return emission.row(time_index)[iter->state]; }
    void Calc(HMM::ActiveState *active, int t) {
      time_index=t; if (calcd[t]) return;
      EmissionArray::Calc(model, active, time_index, observed->row(t), emission.row(t), emissionPosterior.row(t), cudaPosterior);
      calcd[t]=1;
    }
    void CalcNext(HMM::ActiveState *active, HMM::TransitMap *, int t) { return Calc(active, t); /* needs BeamWidth >= NumStates */ }
  };

  static double ForwardBackward(AcousticModel::Compiled *model, Matrix *observations, int InitMax, double beamWidth, HMM::BaumWelchAccum *BWaccum=0, int flag=0);

  static double Viterbi(AcousticModel::Compiled *model, Matrix *observations, Matrix *path, int InitMax, double beamWidth, int flag);
  static double UniformViterbi(AcousticModel::Compiled *model, Matrix *observations, Matrix *viterbi, int Unused1, double Unused2, int Unused3);

  static void PrintLattice(AcousticModel::Compiled *hmm, AcousticModel::Compiled *model=0);
};

struct FeatureSink {
  virtual ~FeatureSink() {}
  FeatureSink() : responseCB(0), inputlen(0) {}

  struct DecodedWord {
    string text;
    long long beg, end;
    DecodedWord(const char *t=0, long long B=0, long long E=0) : text(t), beg(B), end(E) {}
    DecodedWord(const DecodedWord &copy) : text(copy.text), beg(copy.beg), end(copy.end) {}
  };
  typedef deque<DecodedWord> DecodedWords;
  DecodedWords decode;

  typedef function<void(DecodedWords &decoded, int inputlen)> ResponseCB;
  ResponseCB responseCB;
  int inputlen;

  virtual int Write(const Matrix *features, long long timestamp, bool flush=0, ResponseCB cb=0) = 0;
  virtual void Flush() = 0;
  virtual bool Connected() = 0;
};

struct Decoder {
  static unique_ptr<Matrix> DecodeFile(AssetLoading*, AcousticModel::Compiled *model, const char *filename, double beamWidth);
  static unique_ptr<Matrix> DecodeFeatures(AcousticModel::Compiled *model, Matrix *features, double beamWidth, int flag=0, Window *vis=0);
  static void VisualizeFeatures(Window*, AcousticModel::Compiled *model, Matrix *features, Matrix *viterbi, double vprob, Time time, bool interactive);

  struct PhoneIter {
    const AcousticModel::Compiled *model;
    const Matrix *m;
    int beg, end, phone;

    PhoneIter(const AcousticModel::Compiled *Model, const Matrix *M) : model(Model), m(M), beg(0), end(0) { Next(); }

    bool Done() { return !m || beg >= m->M; }
    void Next() {
      beg = end;
      if (Done()) return;
      if ((phone = PhoneID(beg)) < 0) { beg=end=m->M; return; }
      for (/**/; end<m->M; end++) if (PhoneID(end) != phone) break;
    }
    int PhoneID(int offset) {
      int ind = int(m->row(offset)[0]);
      if (ind < 0 || ind >= model->state.size()) FATAL("oob ind ", ind, " (0, ", model->state.size(), ")");
      int pp, np;
      return AcousticModel::ParseName(model->state[ind].name.c_str(), 0, &pp, &np);
    }
  };

  static string Transcript(const AcousticModel::Compiled *model, const Matrix *viterbi, Allocator *alloc=0);
};

#ifdef LFL_CORE_APP_GL_VIEW_H__
struct PhoneticSegmentationGUI : public View {
  struct Segment {
    string name; int beg, end; Box win; bool hover;
    Segment(const string &n, int b, int e) : name(n), beg(b), end(e), hover(0) {}
  };
  vector<Segment> segments;
  unique_ptr<Geometry> geometry;
  string sound_asset_name;
  int sound_asset_len;

  PhoneticSegmentationGUI(Window *W, FeatureSink::DecodedWords &decoded, int len, const string &AN) :
    View(W), sound_asset_name(AN), sound_asset_len(len) {
    Activate();
    for (int i=0, l=decoded.size(); i<l; i++)
      segments.push_back(Segment(decoded[i].text, decoded[i].beg, decoded[i].end));
  }

  PhoneticSegmentationGUI(Window *W, AcousticModel::Compiled *model, Matrix *decoded, const string &AN) :
    View(W), sound_asset_name(AN), sound_asset_len(decoded ? decoded->M : 0) {
    Activate();
    for (Decoder::PhoneIter iter(model, decoded); !iter.Done(); iter.Next()) {
      if (!iter.phone) continue;
      segments.push_back(Segment(StrCat("phone: ", Phoneme::Name(iter.phone)), iter.beg, iter.end));
    }
  }

  void Layout(bool flip) {
    vector<v2> verts;
    for (int i=0, l=segments.size(); i<l; i++) {
      int beg = segments[i].beg, end = segments[i].end, len = end-beg, total = sound_asset_len;

      int wb = !flip ? float(beg)/total*box.w : float(beg)/total*box.h;
      verts.push_back(!flip ? v2(wb, 0)      : v2(0,     wb));
      verts.push_back(!flip ? v2(wb, -box.h) : v2(box.w, wb));

      int we = !flip ? float(end)/total*box.w : float(end)/total*box.h;
      verts.push_back(!flip ? v2(we, 0)      : v2(0,     we));
      verts.push_back(!flip ? v2(we, -box.h) : v2(box.w, we));

      segments[i].win = !flip ? Box(wb, -box.h, float(len)/total*box.w, box.h) : Box(0, wb, box.w, float(len)/total*box.h);
      mouse.AddHoverBox(segments[i].win, Callback([&,i](){ segments[i].hover = !segments[i].hover; })); 
      mouse.AddClickBox(segments[i].win, Callback([&,beg,len](){ Play(beg, len); }));
    }

    if (verts.size()) geometry = make_unique<Geometry>(Geometry::Primitive::Lines, verts.size(), &verts[0],
                                                       nullptr, nullptr, Color(1.0,1.0,1.0));
  }

  void Frame(Box win, Font *font, bool flip=false) {
    box = win;
    if (!geometry.get() && segments.size()) Layout(flip);

    if (geometry.get()) {
      geometry->SetPosition(win.TopLeft());
      root->gd->DisableTexture();
      Scene::Select(root->gd, geometry.get());
      Scene::Draw(root->gd, geometry.get(), 0);
    }

    for (int i = 0; i < segments.size(); i++) {
      if (!segments[i].hover) continue;
      font->Draw(root->gd, segments[i].name, point(root->gl_w * (!flip ? .85 : .15), root->gl_h * (!flip ? .15 : .85)));
      break;
    }
  }

  void Play(int beg, int len) {
    if (root->parent->audio->Out.size()) return;
    vector<string> args;
    args.push_back(sound_asset_name);
    args.push_back(StrCat(beg*FLAGS_feat_hop));
    args.push_back(StrCat(len*FLAGS_feat_hop)); 
    root->shell->play(args);
  }
};
#endif /* LFL_CORE_APP_GL_VIEW_H__ */

int Resynthesize(Audio *s, const SoundAsset *sa);

}; // namespace LFL
#endif // LFL_CORE_SPEECH_SPEECH_H__
