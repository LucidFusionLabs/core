/*
 * $Id: recognition.h 1336 2014-12-08 09:29:59Z justin $
 * Copyright (C) 2009 Lucid Fusion Labs
 */

#ifndef __LFL_SPEECH_RECOGNITION_H__
#define __LFL_SPEECH_RECOGNITION_H__
namespace LFL {

DECLARE_FLAG(LanguageModelWeight, double);
DECLARE_FLAG(WordInsertionPenalty, double);

struct RecognitionModel {
    LogSemiring semiring;
    AcousticModelFile acousticModel;
    WFST::AcousticModelAlphabet recognitionNetworkIn;
    WFST::StringFileAlphabet recognitionNetworkOut;
    WFST recognitionNetwork, grammar;
    WFST::Composer::BigramReachable reachable;
    WFST::Composer composer;
    int emissions;

    struct TokenNameCB : public LFL::TokenNameCB<HMM::Token> {
        RecognitionModel *model;
        TokenNameCB(RecognitionModel *M) : model(M) {}
        string name(HMM::Token &state) { 
            string name = model->acousticModel.state[state.emission_index].name;
            if (state.out) name += string("-") + model->recognitionNetworkOut.name(state.out);
            return name;
        }
    } nameCB;

    RecognitionModel() : recognitionNetworkIn(&acousticModel), recognitionNetwork(&semiring), grammar(&semiring), nameCB(this) {}

    int read(const char *name, const char *dir, int WantIter=-1) {
        int amiter, recogiter, ret;
        if ((amiter = acousticModel.read("AcousticModel", dir, WantIter)) < 0) { ERROR("acousticModel ", dir, " ", amiter); return -1; }
        emissions = acousticModel.states;

        if ((recogiter = recognitionNetwork.read("recognition", dir, WantIter)) < 0) { ERROR("recognition ", dir, " ", recogiter); return -1; }
        if ((ret = recognitionNetworkOut.read(dir, "recognition", "out", recogiter))) { ERROR("recognition out ", dir, " ", ret, " != ", 0); return -1; }
        if (recogiter != (ret = grammar.read("grammar", dir, recogiter))) { ERROR("grammar ", dir, " ", ret, " != ", recogiter); return -1; }

        recognitionNetwork.A = &recognitionNetworkIn;
        recognitionNetwork.B = &recognitionNetworkOut;

        grammar.A = &recognitionNetworkOut;
        grammar.B = &recognitionNetworkOut;

        reachable.clear();
        reachable.precompute(&recognitionNetwork, &grammar, false);

        composer.T1 = &recognitionNetwork;
        composer.T2 = &grammar;
        composer.transition_filter = WFST::Composer::trivial_filter;
        return 0;
    }

    int predict(int source) {
        WFST::State::ChainFilter cfilter(recognitionNetwork.E);
        WFST::State::InvertFilter cifilter(&cfilter);

        WFST::ShortestDistance::PathMap dist;
        WFST::shortestDistance(dist, Singleton<TropicalSemiring>::Get(), recognitionNetwork.E,
                               source, &cifilter, 0, 0, WFST::ShortestDistance::Transduce, 32);

        Semiring *K = Singleton<TropicalSemiring>::Get();
        int out=0; double val=K->Zero(); 
        for (WFST::ShortestDistance::PathMap::iterator i = dist.begin(); i != dist.end(); i++)
            if ((*i).second.out.size() == 1 && K->Add((*i).second.D, val) != val) { 
                out = (*i).second.out[0];
                val = (*i).second.D;
            }

        return out;
    }
};

struct RecognitionHMM {
    struct TransitMap : public HMM::TransitMap {
        RecognitionModel *model;
        bool UseTransitionProb;
        TransitMap(RecognitionModel *M, bool UseTransit=false) : model(M), UseTransitionProb(UseTransit) {}

        virtual void begin(Iterator *iter, HMM::ActiveState *active, HMM::ActiveState::Iterator *LstateI) {
            HMM::TokenPasser<HMM::Token> *beam = (HMM::TokenPasser<HMM::Token>*)active;
            HMM::Token *L = &beam->active[LstateI->impl-1];
            iter->done = 0;
            iter->state = LstateI->index / active->NBest;
            iter->state2 = L->ind2;
            iter->emission_index = L->emission_index;
            iter->out = L->out;
            iter->cost = model->acousticModel.state[iter->emission_index].txself;
            iter->impl1 = (void*)L;
            iter->impl2 = -1;
            iter->impl5 = L->ind2;
            iter->impl6 = 0;
        }
        virtual void next(Iterator *iter) {
            WFST::TransitMap::Iterator iter2;
            if (iter->impl2 < 0) {
                model->recognitionNetwork.E->begin(&iter2, iter->state);
                translateIterOut(iter, &iter2);

                /* impl6 = true if multiple transitions */
                model->recognitionNetwork.E->next(&iter2);
                iter->impl6 = !iter2.done;
            }
            else {
                translateIterIn(iter, &iter2);
                model->recognitionNetwork.E->next(&iter2);
                translateIterOut(iter, &iter2);
            }
        }
        void translateIterIn(Iterator *iter, WFST::TransitMap::Iterator *iter2) {
            iter2->done = iter->done;
            iter2->impl1 = iter->impl2;
            iter2->impl2 = iter->impl3;
            iter2->impl3 = -1;
            iter2->prevState = iter->impl4;
        }
        void translateIterOut(Iterator *iter, WFST::TransitMap::Iterator *iter2) {
            iter->impl2 = iter2->impl1;
            iter->impl3 = iter2->impl2;
            iter->impl4 = iter2->prevState;
            if ((iter->done = iter2->done)) return;

            iter->state = iter2->nextState;
            iter->state2 = iter->impl5;
            iter->emission_index = iter2->in;
            iter->out = iter2->out;
            iter->cost = UseTransitionProb ? -iter2->weight /* negative log prob */ : 0;
        }
        int id(int Lstate) { return Lstate; }
    };
    
    struct DynamicComposer : public TransitMap { 
        DynamicComposer(RecognitionModel *M, bool UseTransit=false) : TransitMap(M, UseTransit) {}
        virtual void next(Iterator *iter) {
            TransitMap::next(iter);
            if (iter->done || !iter->out) return;

            HMM::Token *t = (HMM::Token*)iter->impl1;
            WFST::Composer::trip q(t->ind, t->ind2, 0);
            WFST::Edge e1(t->ind, iter->state, iter->emission_index, iter->out, iter->cost), e2;
            int matched = model->composer.compose_right(q.second, e1.out, e1, 0, q.third, &e2);
            if (!matched) matched = model->reachable(q.second, e1.out, 0, &e2);
            if (matched != 1) { ERROR("dynamicComposer: no match for ", q.second, ", ", e1.out, " (", matched, ")"); return; }

            iter->state2 = e2.nextState;
            iter->cost += FLAGS_LanguageModelWeight * -e2.weight;
            // INFO("tx ", model->recognitionNetworkOut.name(e1.prevState), " -> ", model->recognitionNetworkOut.name(e1.out), " @ ", e2.weight);
        }
    };

    struct Emission : public HMM::Emission {
        RecognitionModel *model;
        HMM::Observation *observed;
        TransitMap *transit;
        bool UsePriorProb;

        Allocator *alloc;
        HMM::ActiveStateIndex beam;
        double *emission;

        ~Emission() { if (alloc) alloc->Free(emission); }
        Emission(RecognitionModel *M, HMM::Observation *O, TransitMap *T, Allocator *Alloc=0, bool UsePrior=false) :
            model(M), observed(O), transit(T), UsePriorProb(UsePrior),
            alloc(Alloc?Alloc:Singleton<MallocAlloc>::Get()), beam(model->emissions, 1, model->emissions, 0, alloc), emission((double*)alloc->Malloc(model->emissions*sizeof(double))) {}

        double *observation(int t) { return observed->observation(t); }
        int observations() { return observed->observations(); }

        double *posterior(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) { return 0; }
        double prior(HMM::ActiveState *actiae, HMM::ActiveState::Iterator *state) { return 0; }

        double prob(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) {
            int emission_index = ((HMM::TokenPasser<HMM::Token>*)active)->active[state->impl-1].emission_index;
            return emission[emission_index];
        }

        double prob(TransitMap::Iterator *iter) { return emission[iter->emission_index]; }

        void calc(HMM::ActiveState *active, int t) {
            if (t) FATAL("unexpected ", t); /* bootstrap t=0 */

            HMM::TokenPasser<HMM::Token> *beam = (HMM::TokenPasser<HMM::Token>*)active;
            beam->count = 1;
            beam->active[0].ind = 0;
            beam->active[0].ind2 = 0;
            beam->active[0].val = -INFINITY;
            beam->active[0].emission_index = 0;
            beam->active[0].out = 0;
            beam->active[0].backtrace = 0;
            beam->active[0].tstate = 0;
            beam->active[0].steps = 1;

            calcNext(active, transit, t);
        }

        void calcNext(HMM::ActiveState *active, HMM::TransitMap *transit, int t) {
            Timer t1;
            active->time_index = t;
            memset(beam.active, -1, model->emissions*sizeof(int));

            HMM::ActiveState::Iterator LstateI;
            for (active->begin(t, &LstateI); !LstateI.done; active->next(&LstateI)) {
                int emission_index = ((HMM::TokenPasser<HMM::Token>*)active)->active[LstateI.impl-1].emission_index;
                beam.active[emission_index] = emission_index;

                HMM::TransitMap::Iterator RstateI;
                for (transit->begin(&RstateI, active, &LstateI); !RstateI.done; transit->next(&RstateI)) 
                    beam.active[RstateI.emission_index] = RstateI.emission_index;
            }

            sort(beam.active, beam.active + model->emissions, intSort);
            for (beam.count=0; beam.count < model->emissions && beam.active[beam.count] >= 0; beam.count++) {}

            Timer t2;
            for (int i=0; i<model->emissions; i++) emission[i] = -INFINITY;
            AcousticHMM::EmissionArray::calc(&model->acousticModel, &beam, t, observation(t), emission);
            // INFO("calc ", beam.count, " in " t1.time()*1000, " ms (emitcalc ", t2.time()*1000, " ms));
        }

        static bool intSort(const int x, const int y) { return x > y; }
    };

    struct TokenPasser : public HMM::TokenPasser<HMM::Token> {
        RecognitionModel *model;

        TokenPasser(RecognitionModel *M, int NS, int NB, int BW, HMM::TokenBacktrace<HMM::Token> *BT, int Scale=100, Allocator *Alloc=0) :
            HMM::TokenPasser<HMM::Token>(NS, NB, BW, BT, Scale, Alloc), model(M) {}

        virtual double *out(int time_index, ActiveState::Iterator *left, TransitMap::Iterator *right, double **trace) {
            double *ret = HMM::TokenPasser<HMM::Token>::out(time_index, left, right, trace);
            if (!ret) return 0;

            HMM::Token *parent = &active[left->impl-1], *t = &nextActive[nextCount-1];
            if (t->tstate && right->impl6) { t->tstate=0; t->steps=0; }
            if (t->out) t->tstate = 1;
            return ret;
        }
    };

    static double viterbi(RecognitionModel *model, Matrix *observations, matrix<HMM::Token> *path, double beamWidth, bool UseTransit=0, LFL::TokenNameCB<HMM::Token> *ncb=0) {
        int NBest=1, M=observations->M;
        DynamicComposer transit(model, UseTransit);
        HMM::ObservationMatrix om(observations);
        Emission emit(model, &om, &transit);

        matrix<HMM::Token> backtrace(M, beamWidth, HMM::Token());
        HMM::TokenBacktraceMatrix<HMM::Token> tbt(&backtrace);
        TokenPasser beam(model, 1, NBest, beamWidth, &tbt);

        int endstate = HMM::forward(&beam, &transit, &emit, &beam, &beam, 0);
        HMM::Token::tracePath(path, &backtrace, endstate, M);
        return backtrace.row(M-1)[endstate].val;
    }

    static void printLattice(RecognitionModel *recognize) {}
};

struct Recognizer {
    static matrix<HMM::Token> *decodeFile(RecognitionModel *model, const char *fn, double beamWidth=0,
                                          bool UseTransit=0, double *vprobout=0, Matrix **MFCCout=0, TokenNameCB<HMM::Token> *nameCB=0) {
        SoundAsset input("input", fn, 0, 0, 0, 0);
        input.Load();
        if (!input.wav) { ERROR(fn, " not found"); return 0; }

        if (MFCCout) *MFCCout = Features::fromAsset(&input, Features::Flag::Storable);
        Matrix *features = Features::fromAsset(&input, Features::Flag::Full);
        matrix<HMM::Token> *ret = decodeFeatures(model, features, beamWidth, UseTransit, vprobout, nameCB);
        delete features;
        return ret;
    }

    static matrix<HMM::Token> *decodeFeatures(RecognitionModel *model, Matrix *features, double beamWidth,
                                              bool UseTransit=0, double *vprobout=0, TokenNameCB<HMM::Token> *nameCB=0) {
        if (!DimCheck("decodeFeatures", features->N, model->acousticModel.state[0].emission.mean.N)) return 0;
        matrix<HMM::Token> *viterbi = new matrix<HMM::Token>(features->M, 1, HMM::Token());
        double vprob = RecognitionHMM::viterbi(model, features, viterbi, beamWidth, UseTransit, nameCB);
        if (vprobout) *vprobout = vprob;
        return viterbi;
    }

    static AcousticModel::Compiled *decodedAcousticModel(RecognitionModel *model, matrix<HMM::Token> *viterbi, Matrix *vout) {
        int last = -1, count = 0;
        MatrixRowIter(viterbi) {
            int recogState = viterbi->row(i)[0].ind;
            if (recogState == last) continue;
            last = recogState;
            count++;
        }

        AcousticModel::Compiled *hmm = new AcousticModel::Compiled(count);
        last = -1; count = 0;
        MatrixRowIter(viterbi) {
            vout->row(i)[0] = count-1;
            int emission_index = viterbi->row(i)[0].emission_index;
            if (emission_index == last) continue;

            vout->row(i)[0] = count;
            last = emission_index;
            int pp, pn, phone = AcousticModel::parseName(model->acousticModel.state[emission_index].name, 0, &pp, &pn);
            AcousticModel::State::assign(hmm->state, &count, hmm->states, &model->acousticModel, 0, phone, 0, 1);
        }
        if (count > hmm->states) FATAL("overflow ", count, " > ", hmm->states);
        return hmm;
    }

    struct WordIter {
        const RecognitionModel *model;
        const matrix<HMM::Token> *m;
        int beg, end, impl, adjustforward;
        string word;

        WordIter(const RecognitionModel *Model, const matrix<HMM::Token> *paths) : model(Model), m(paths), beg(0), end(0), impl(0), adjustforward(0) { next(); }

        bool done() { return impl >= m->M; }
        void next() {
            for (/**/; impl<m->M; impl++) {
                const HMM::Token *t = m->row(impl), *t2 = 0;
                if (!t->out) continue;

                word = model->recognitionNetworkOut.name(t->out);
                beg = impl++ - t->steps + adjustforward;
                adjustforward = 0;

                for (/**/; impl<m->M; impl++) {
                    if (!(t2 = m->row(impl))->ind) break;
                    if (t2->tstate) continue;

                    int adjustbegin = impl, tps = AcousticModel::StatesPerPhone-1, ps = -1;
                    if (impl) {
                        if (!(t2 = m->row(impl-1))->ind) break;
                        ps = phonemestate(t2);
                    }

                    if (!impl || ps != tps) {
                        ps = seekphonemestate(tps);
                        if (ps != tps) break;
                    }

                    ps = seekphonemestate(0);
                    adjustforward = impl - adjustbegin;
                    break;
                }

                end = impl-1;
                break;
            }
        }

        int phonemestate(const HMM::Token *t) {
            int ps = -1;
            AcousticModel::parseName(model->acousticModel.state[t->emission_index].name, &ps);
            return ps;
        }

        int seekphonemestate(int tps) {
            int ps = -1;
            for (const HMM::Token *t; impl<m->M; impl++) 
                if (!(t = m->row(impl))->ind || (ps = phonemestate(t)) == tps) break;
            return ps;
        }
    };

    static string transcript(const RecognitionModel *model, const matrix<HMM::Token> *viterbi, bool annotate=false) {
        Recognizer::WordIter iter(model, viterbi); string v;
        for (/**/; !iter.done(); iter.next()) {
            if (!iter.word.size()) continue;
            v += iter.word;
            if (annotate) StringAppendf(&v, "-%d(%d,%d) ", iter.word.c_str(), iter.end-iter.beg, iter.beg, iter.end);
            else v += " ";
        }
        return v;
    }

    static double wordErrorRate(const RecognitionModel *model, string gold, string x) {
        vector<int> A, B;
        LFL::StringWordIter aw(gold.c_str()), bw(x.c_str());
        for (const char *w = aw.Next(); w; w = aw.Next()) A.push_back(model->recognitionNetworkOut.id(tolower(w).c_str()));
        for (const char *w = bw.Next(); w; w = bw.Next()) B.push_back(model->recognitionNetworkOut.id(tolower(w).c_str()));
        return (double)Levenshtein(A, B) / A.size();
    }
};

}; // namespace LFL
#endif // __LFL_SPEECH_RECOGNITION_H__
