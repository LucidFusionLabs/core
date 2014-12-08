/*
 * $Id: speech.h 1312 2014-10-14 01:39:53Z justin $
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

#ifndef __LFL_LFAPP_SPEECH_H__
#define __LFL_LFAPP_SPEECH_H__
namespace LFL {

DECLARE_bool(TriphoneModel);

/* phonemes */
struct Phoneme { 
    enum {
#define LFL_LANGUAGE_ENGLISH
#undef  XX
#define XX(phone) phone,
#include "speech/phones.h"
    };
    static const char *name(char phoneId);
    static char id(const char *phone, int len=0);
};
#define PhonemeIter() for (int phone=0; phone<LFL_PHONES; phone++)

/* pronunciation dictionary */
struct PronunciationDict {
    typedef map<string, int> Map;
    Map word_pronunciation;
    ReallocHeap val;

    static PronunciationDict *instance();
    static int readDictionary(Iter *in, PronunciationDict *out);
    static int readPronunciation(const char *in, int len, char *phoneIdOut, char *accentOut, int outlen);

    const char *pronounce(const char *word);
    int pronounce(const char *utterance, const char **words, const char **accents, int *phones, int max);
};

struct SoundAsset;

/* features */
struct Features {
    static int FilterZeroth, Deltas, DeltaDeltas, MeanNormalization, VarianceNormalization;
    static Matrix *filterZeroth(Matrix *features);

    static void meanAndVarianceNormalization(int D, double *feat, const double *mean, const double *var);
    static void deltaCoefficients(int D, const double *n2, double *f, const double *p2);
    static void deltaDeltaCoefficients(int D, const double *n3, const double *n1, double *f,  const double *p1, const double *p3);
    static void patchDeltaCoefficients(int D, const double *in, double *out1, double *out2); 
    static void patchDeltaDeltaCoefficients(int D, const double *in, double *out1, double *out2, double *out3);
    static Matrix *deltaCoefficients(Matrix *features, bool deltadelta=true);

    struct Flag { enum { Full=1, Storable=2 }; int f; };
    static Matrix *fromAsset(SoundAsset *wav, int flag);
    static Matrix *fromFeat(Matrix *features, int flag);
    static Matrix *fromFeat(Matrix *features, int flag, bool FilterZeroth, bool Deltas, bool DeltaDeltas, bool MeanNorm, bool VarNorm);
    static Matrix *fromBuf(const RingBuf::Handle *in, Matrix *out=0, vector<Filter> *filter=0, Allocator *alloc=0);
    static RingBuf *reverse(const Matrix *in, int samplerate, const Matrix *f0=0, Allocator *alloc=0);
    static int dimension();
};

/* acoustic model */
struct AcousticModel {
    struct State;
    struct StateCollection { 
        virtual ~StateCollection() {}
        virtual Matrix *tiedStates() { return 0; }
        virtual Matrix *phoneTx() { return 0; }

        virtual int getStateCount() = 0;
        virtual State *getState(unsigned stateID) = 0;

        struct Iterator { int k; State *v; char impl[2048]; bool done; };
        virtual void beginState(Iterator *iter) = 0;
        virtual void nextState(Iterator *iter) = 0;
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
        State() : alloc(Singleton<NullAlloc>::Get()), prior(-INFINITY), txself(-INFINITY) { val.samples=val.emission_index=0; }
        unsigned id() { return fnv32(name.c_str()); }
        void assignPtr(State *s) {
            if (!s) FATAL("assign ", s);
            name = s->name;
            prior = s->prior;
            txself = s->txself;
            transition.assignDataPtr(s->transition.M, s->transition.N, s->transition.m);
            emission.assignDataPtr(&s->emission);
            val = s->val;
        }

        static double *transit(Compiled *model, Matrix *trans, State *Rstate, unsigned *to=0);
        static double *transit(Matrix *trans, unsigned Rstate);
        static int transitionSort(const void *a, const void *b);
        static void sortTransitionMap(Matrix *trans);
        static void sortTransitionMap(double *trans, int M);
        static int localizeTransitionMap(Matrix *transI, Matrix *transO, Compiled *scope, int scopeState=0, int minSamples=0, Compiled *dubdref=0, const char *srcname=0);
        static unsigned tied(Matrix *tiedstates, int pp, int p, int pn, int k);
        static void assign(State *out, int *outind, int outsize, StateCollection *ps, int pp, int p, int pn, int states, int val=0); 
        static bool triphoneConnect(int ap, int a, int an, int bp, int b, int bn) {
            bool asil = !ap && !a && !an, bsil = !bp && !b && !bn;
            return (bp == a && b == an && !bsil && !asil) || (asil && !bp && !bsil) || (bsil && !an && !asil);
        }
    };

    struct Compiled : public StateCollection {
        State *state;
        int states;
        Matrix *tiedstates, *phonetx;
        
        struct NameCB : public LFL::NameCB {
            Compiled *model;
            NameCB(Compiled *M) : model(M) {}
            string name(int ind) {
                if (ind < 0 || ind > model->states) FATAL("oob ind ", ind);
                return model->state[ind].name;
            }
        } nameCB;

        ~Compiled() { delete [] state; }
        Compiled(int N=0) : state(0), states(0), tiedstates(0), phonetx(0), nameCB(this) { if (N) open(N); }
        void open(int N) { delete [] state; states=N; state=new State[states]; }

        int getStateCount() { return states; }
        State *getState(unsigned stateID) { return 0; }
        void beginState(Iterator *iter) { iter->done = 0; *((int*)iter->impl) = 0; nextState(iter); }
        void nextState(Iterator *iter) {
            int *impl = (int*)iter->impl;
            if (*impl >= states) { iter->done = 1; return; }
            iter->k = (*impl)++;
            iter->v = &state[iter->k];
        }
        Matrix *tiedStates() { return tiedstates; }
        Matrix *phoneTx() { return phonetx; }
    };

    const static int StatesPerPhone=3;
    static int parseName(const string &name, int *phonemeStateOut=0, int *prevPhoneOut=0, int *nextPhoneOut=0);

    static string name(int phoneme, int phonemeState) { return StringPrintf("Model_%s_State_%02d", Phoneme::name(phoneme), phonemeState); }

    static string name(int pp, int p, int np, int ps) {
        if (!pp && !np) return name(p, ps);
        return StringPrintf("Model_%s_%s_%s_State_%02d", Phoneme::name(pp), Phoneme::name(p), Phoneme::name(np), ps);
    }

    static bool isSilence(const string &name) { return name == "Model_SIL_State_00"; }
    static bool isSilence(unsigned id)        { return id == fnv32("Model_SIL_State_00"); }

    static bool triphoneSort(const string &x, const string &y) {
        int xn, xp, yn, yp, cmp = parseName(x.c_str(), &xp, &xn) - parseName(y.c_str(), &yp, &yn);
        if (cmp) return cmp < 0;
        if ((cmp = xp - yp)) return cmp < 0;
        return (xn - yn) < 0;
    }

    static Compiled *fullyConnected(Compiled *model);
    static Compiled *fromUtterance(Compiled *model, const char *transcript, bool UseTransit);
    static Compiled *fromUtterance3(Compiled *model, const char *transcript, bool UseTransit);
    static Compiled *fromUtterance1(Compiled *model, const char *transcript, bool UseTransit);
    static Compiled *fromModel1(StateCollection *model, bool rewriteTransitions);

    static void loadflags(const char *flags);
    static string flags();

    static int write(StateCollection *, const char *name, const char *dir, int iteration, int minSamples);
    static int toCUDA(AcousticModel::Compiled *);
};

struct AcousticModelFile : public AcousticModel::Compiled {
    Matrix *initial, *mean, *covar, *prior, *transit, *map, *tied;
    StringFile names;

    ~AcousticModelFile() { reset(); }
    AcousticModelFile() : initial(0), mean(0), covar(0), prior(0), transit(0), map(0), tied(0) {}
    void reset() { delete initial; initial=0; delete mean; mean=0; delete covar; covar=0; delete prior; prior=0; delete transit; transit=0; delete map; map=0; delete tied; tied=0; names.clear(); }
    AcousticModel::State *getState(unsigned hash) { double *he = getHashEntry(hash); return he ? &state[(int)he[1]] : 0; }
    double *getHashEntry(unsigned hash) { return HashMatrix::get(map, hash, 4); }
    Matrix *tiedStates() { return tied; }
    int read(const char *name, const char *dir, int lastiter=-1, bool rebuildTransit=0);
};

struct AcousticModelBuilder : public AcousticModel::StateCollection {
    typedef map<int, AcousticModel::State*> Map;
    Map statemap;

    AcousticModelBuilder() {} 
    ~AcousticModelBuilder() {
        for (Map::iterator i = statemap.begin(); i != statemap.end(); i++) {
            AcousticModel::State *s = (*i).second;
        }
    }

    int getStateCount() { return statemap.size(); }
    AcousticModel::State *getState(unsigned stateID) { Map::iterator i = statemap.find(stateID); return i != statemap.end() ? (*i).second : 0; }

    void beginState(Iterator *iter) { iter->done=0; Map::iterator *i = (Map::iterator*)iter->impl; *i = statemap.begin(); assignKV(iter, i); }
    void nextState (Iterator *iter) {               Map::iterator *i = (Map::iterator*)iter->impl; (*i)++;                assignKV(iter, i); }

    void assignKV(Iterator *iter, Map::iterator *i) {
        if (*i == statemap.end()) iter->done = 1;
        else { iter->k = (**i).first; iter->v = (**i).second; }
    }

    void add(AcousticModel::State *s, int id) { statemap[id] = s; };
    void add(AcousticModel::State *s) { add(s, s->id()); }
    void rem(AcousticModel::State *s) { rem(s->id()); }
    void rem(int id) { statemap.erase(id); }
};

struct AcousticHMM {
    struct Flag { enum { UsePrior=1, UseTransit=2, UseContentIndependentTransit=4, Visualize=8, Interactive=16 }; };

    struct TransitMap : public HMM::TransitMap {
        AcousticModel::Compiled *model;
        bool UseTransitionProb;
#define TransitCols 3
#define TC_Self 0
#define TC_Edge 1
#define TC_Cost 2
        TransitMap(AcousticModel::Compiled *M, bool UseTransition) : model(M), UseTransitionProb(UseTransition) {}

        void begin(Iterator *iter, HMM::ActiveState *active, HMM::ActiveState::Iterator *Lstate) {
            iter->impl1 = (void*)&model->state[Lstate->index / active->NBest];
            iter->impl2 = 0;
            iter->state2 = 0;
            iter->done = 0;
            next(iter);
        }
        void next(Iterator *iter) {
            AcousticModel::State *s = (AcousticModel::State*)iter->impl1;
            if (iter->impl2 >= s->transition.M) { iter->done=1; return; }
            double *tr = s->transition.row(iter->impl2++);
            iter->state = tr[TC_Edge];
            iter->cost = UseTransitionProb ? tr[TC_Cost] : 0;
        }
        int id(int Lstate) { return model->state[Lstate].val.emission_index; }
    };

    struct EmissionArray : public HMM::Emission {
        AcousticModel::Compiled *model;
        Matrix *observed;
        bool UsePriorProb;
        int time_index;

        Allocator *alloc;
        double *emission;

        ~EmissionArray() { if (alloc) alloc->free(emission); }
        EmissionArray(AcousticModel::Compiled *M, Matrix *Observed, bool UsePrior, Allocator *Alloc=0) : model(M),
            observed(Observed), UsePriorProb(UsePrior), time_index(0), alloc(Alloc?Alloc:Singleton<MallocAlloc>::Get()),
            emission((double*)alloc->malloc(sizeof(double)*model->states)) {}

        double *observation(int t) { return observed->row(t); }
        int observations() { return observed->M; }
        double prior(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) { return UsePriorProb ? model->state[state->index].prior : 0; }
        double *posterior(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) { return 0; }
        double prob(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) { return emission[state->index]; }
        double prob(TransitMap::Iterator *iter) { return emission[iter->state]; }
        void calc(HMM::ActiveState *active, int t) { time_index=t; calc(model, active, time_index, observed->row(t), emission); }
        void calcNext(HMM::ActiveState *active, HMM::TransitMap *, int t) { return calc(active, t); /* needs BeamWidth >= NumStates */ }
        static void calc(AcousticModel::Compiled *model, HMM::ActiveState *active, int time_index, const double *observed, double *emission, double *posterior=0, double *cudaposterior=0);
    };

    struct EmissionMatrix : public HMM::Emission {
        AcousticModel::Compiled *model;
        Matrix *observed;
        bool UsePriorProb;
        int K, time_index;

        Allocator *alloc;
        Matrix emission, emissionPosterior;
        bool *calcd;
        double *cudaPosterior;

        ~EmissionMatrix() { if (!alloc) return; alloc->free(calcd); alloc->free(cudaPosterior); }
        EmissionMatrix(AcousticModel::Compiled *M, Matrix *Observed, bool UsePrior, Allocator *Alloc=0) : model(M), observed(Observed), UsePriorProb(UsePrior),
            K(model->state[0].emission.mean.M), time_index(0), alloc(Alloc?Alloc:Singleton<MallocAlloc>::Get()),
            emission(observed->M, model->states, 0, 0, Alloc), emissionPosterior(observed->M, model->states*K, 0, 0, Alloc),
            calcd((bool*)alloc->malloc(observed->M)), cudaPosterior((double*)alloc->malloc(model->states*K*sizeof(double)))
            { memset(calcd, 0, observed->M);  }

        double *observation(int t) { return observed->row(t); }
        int observations() { return observed->M; }
        double prior(HMM::ActiveState *active, HMM::ActiveState::Iterator *iter) { return UsePriorProb ? model->state[iter->index].prior : 0; }
        double prob(HMM::ActiveState *active, HMM::ActiveState::Iterator *iter) { return emission.row(time_index)[iter->index]; }
        double *posterior(HMM::ActiveState *active, HMM::ActiveState::Iterator *iter) { return emissionPosterior.row(time_index) + iter->index*K; }
        double prob(TransitMap::Iterator *iter) { return emission.row(time_index)[iter->state]; }
        void calc(HMM::ActiveState *active, int t) {
            time_index=t; if (calcd[t]) return;
            EmissionArray::calc(model, active, time_index, observed->row(t), emission.row(t), emissionPosterior.row(t), cudaPosterior);
            calcd[t]=1;
        }
        void calcNext(HMM::ActiveState *active, HMM::TransitMap *, int t) { return calc(active, t); /* needs BeamWidth >= NumStates */ }
    };

    static double forwardBackward(AcousticModel::Compiled *model, Matrix *observations, int InitMax, double beamWidth, HMM::BaumWelchAccum *BWaccum=0, int flag=0);

    static double viterbi(AcousticModel::Compiled *model, Matrix *observations, Matrix *path, int InitMax, double beamWidth, int flag);
    static double uniformViterbi(AcousticModel::Compiled *model, Matrix *observations, Matrix *viterbi, int Unused1, double Unused2, int Unused3);

    static void printLattice(AcousticModel::Compiled *hmm, AcousticModel::Compiled *model=0);
};

/* client network interface */
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

    typedef void (*ResponseCB)(DecodedWords &decoded, int inputlen);
    ResponseCB responseCB;
    int inputlen;

    virtual int Write(const Matrix *features, long long timestamp, bool flush=0, ResponseCB cb=0) = 0;
    virtual void Flush() = 0;
    virtual bool connected() = 0;
};

/* decoder */
struct Decoder {
    static Matrix *decodeFile(AcousticModel::Compiled *model, const char *filename, double beamWidth);
    static Matrix *decodeFeatures(AcousticModel::Compiled *model, Matrix *features, double beamWidth, int flag=0);
    static void visualizeFeatures(AcousticModel::Compiled *model, Matrix *features, Matrix *viterbi, double vprob, double time, bool interactive);

    struct PhoneIter {
        const AcousticModel::Compiled *model;
        const Matrix *m;
        int beg, end, phone;

        PhoneIter(const AcousticModel::Compiled *Model, const Matrix *M) : model(Model), m(M), beg(0), end(0) { next(); }

        bool done() { return !m || beg >= m->M; }
        void next() {
            beg = end;
            if (done()) return;
            if ((phone = phoneID(beg)) < 0) { beg=end=m->M; return; }
            for (/**/; end<m->M; end++) if (phoneID(end) != phone) break;
        }
        int phoneID(int offset) {
            int ind = (int)m->row(offset)[0];
            if (ind < 0 || ind >= model->states) FATAL("oob ind ", ind, " (0, ", model->states, ")");
            int pp, np;
            return AcousticModel::parseName(model->state[ind].name.c_str(), 0, &pp, &np);
        }
    };

    static string transcript(const AcousticModel::Compiled *model, const Matrix *viterbi, Allocator *alloc=0);
};

#ifdef __LFL_LFAPP_GUI_H__
struct PhoneticSegmentationGUI : public GUI {
    struct Segment {
        string name; int beg, end; Box win; bool hover;
        Segment(const string &n, int b, int e) : name(n), beg(b), end(e), hover(0) {}
    };
    vector<Segment> segments;
    auto_ptr<Geometry> geometry;
    string sound_asset_name;
    int sound_asset_len;

    PhoneticSegmentationGUI(LFL::Window *w, FeatureSink::DecodedWords &decoded, int len, const string &AN) : GUI(w), sound_asset_name(AN), sound_asset_len(len) {
        for (int i=0, l=decoded.size(); i<l; i++)
            segments.push_back(Segment(decoded[i].text, decoded[i].beg, decoded[i].end));
    }
    PhoneticSegmentationGUI(LFL::Window *w, AcousticModel::Compiled *model, Matrix *decoded, const string &AN) : GUI(w), sound_asset_name(AN), sound_asset_len(decoded ? decoded->M : 0) {
        for (Decoder::PhoneIter iter(model, decoded); !iter.done(); iter.next()) {
            if (!iter.phone) continue;
            segments.push_back(Segment(StrCat("phone: ", Phoneme::name(iter.phone)), iter.beg, iter.end));
        }
    }

    void Layout(bool flip) {
        vector<v2> verts;
        for (int i=0, l=segments.size(); i<l; i++) {
            int beg = segments[i].beg, end = segments[i].end, len = end-beg, total = sound_asset_len;

            int wb = !flip ? (float)beg/total*box.w : (float)beg/total*box.h;
            verts.push_back(!flip ? v2(wb, 0)      : v2(0,     wb));
            verts.push_back(!flip ? v2(wb, -box.h) : v2(box.w, wb));

            int we = !flip ? (float)end/total*box.w : (float)end/total*box.h;
            verts.push_back(!flip ? v2(we, 0)      : v2(0,     we));
            verts.push_back(!flip ? v2(we, -box.h) : v2(box.w, we));

            segments[i].win = !flip ? Box(wb, -box.h, (float)len/total*box.w, box.h) : Box(0, wb, box.w, (float)len/total*box.h);
            mouse.AddHoverBox(segments[i].win, Callback([&,i](){ segments[i].hover = !segments[i].hover; })); 
            mouse.AddClickBox(segments[i].win, Callback([&,beg,len](){ Play(beg, len); }));
        }

        if (verts.size()) geometry = auto_ptr<Geometry>
            (new Geometry(GraphicsDevice::Lines, verts.size(), &verts[0], 0, 0, Color(1.0,1.0,1.0)));
    }

    void Frame(Box win, Font *font, bool flip=false) {
        box = win;
        mouse.Activate();
        if (!geometry.get() && segments.size()) Layout(flip);

        if (geometry.get()) {
            geometry->SetPosition(win.TopLeft());
            screen->gd->DisableTexture();
            Scene::Select(geometry.get());
            Scene::Draw(geometry.get(), 0);
        }

        for (int i = 0; i < segments.size(); i++) {
            if (!segments[i].hover) continue;
            font->Draw(segments[i].name, point(screen->width * (!flip ? .85 : .15), screen->height * (!flip ? .15 : .85)));
            break;
        }
    }

    void Play(int beg, int len) {
        if (app->audio.Out.size()) return;
        vector<string> args;
        args.push_back(sound_asset_name);
        args.push_back(StrCat(beg*FLAGS_feat_hop));
        args.push_back(StrCat(len*FLAGS_feat_hop)); 
        app->shell.play(args);
    }
};
#endif /* __LFL_LFAPP_GUI_H__ */

int resynthesize(Audio *s, const SoundAsset *sa);

}; // namespace LFL
#endif // __LFL_LFAPP_SPEECH_H__
