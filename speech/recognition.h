/*
 * $Id: aed.h 1330 2014-11-06 03:04:15Z justin $
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

#ifndef LFL_SPEECH_RECOGNITION_H__
#define LFL_SPEECH_RECOGNITION_H__
namespace LFL {

DECLARE_FLAG(language_model_weight, double);
DECLARE_FLAG(word_insertion_penalty, double);

struct RecognitionModel {
  LogSemiring semiring;
  AcousticModelFile acoustic_model;
  WFST::AcousticModelAlphabet recognition_network_in;
  WFST::StringFileAlphabet recognition_network_out;
  WFST recognition_network, grammar;
  WFST::Composer::BigramReachable reachable;
  WFST::Composer composer;
  int emissions;

  struct TokenNameCB : public LFL::TokenNameCB<HMM::Token> {
    RecognitionModel *model;
    TokenNameCB(RecognitionModel *M) : model(M) {}
    string Name(HMM::Token &state) { 
      string name = model->acoustic_model.state[state.emission_index].name;
      if (state.out) name += string("-") + model->recognition_network_out.Name(state.out);
      return name;
    }
  } nameCB;

  RecognitionModel() : recognition_network_in(&acoustic_model), recognition_network(&semiring), grammar(&semiring), nameCB(this) {}

  int Read(const char *name, const char *dir, int want_iter=-1) {
    int amiter, recogiter, ret;
    if ((amiter = acoustic_model.Open("AcousticModel", dir, want_iter)) < 0) { ERROR("acoustic_model ", dir, " ", amiter); return -1; }
    emissions = acoustic_model.states;

    if ((recogiter = recognition_network.Read("recognition", dir, want_iter)) < 0) { ERROR("recognition ", dir, " ", recogiter); return -1; }
    if ((ret = recognition_network_out.Read(dir, "recognition", "out", recogiter))) { ERROR("recognition out ", dir, " ", ret, " != ", 0); return -1; }
    if (recogiter != (ret = grammar.Read("grammar", dir, recogiter))) { ERROR("grammar ", dir, " ", ret, " != ", recogiter); return -1; }

    recognition_network.A = &recognition_network_in;
    recognition_network.B = &recognition_network_out;

    grammar.A = &recognition_network_out;
    grammar.B = &recognition_network_out;

    reachable.Clear();
    reachable.PreCompute(&recognition_network, &grammar, false);

    composer.T1 = &recognition_network;
    composer.T2 = &grammar;
    composer.transition_filter = WFST::Composer::TrivialFilter;
    return 0;
  }

  int Predict(int source) {
    WFST::State::ChainFilter cfilter(recognition_network.E);
    WFST::State::InvertFilter cifilter(&cfilter);

    WFST::ShortestDistance::PathMap dist;
    WFST::ShortestDistance(dist, Singleton<TropicalSemiring>::Get(), recognition_network.E,
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
    bool use_transition_prob;
    TransitMap(RecognitionModel *M, bool UT=false) : model(M), use_transition_prob(UT) {}

    virtual void Begin(Iterator *iter, HMM::ActiveState *active, HMM::ActiveState::Iterator *LstateI) {
      HMM::TokenPasser<HMM::Token> *beam = static_cast<HMM::TokenPasser<HMM::Token>*>(active);
      HMM::Token *L = &beam->active[LstateI->impl-1];
      iter->done = 0;
      iter->state = LstateI->index / active->NBest;
      iter->state2 = L->ind2;
      iter->emission_index = L->emission_index;
      iter->out = L->out;
      iter->cost = model->acoustic_model.state[iter->emission_index].txself;
      iter->impl1 = Void(L);
      iter->impl2 = -1;
      iter->impl5 = L->ind2;
      iter->impl6 = 0;
    }
    virtual void Next(Iterator *iter) {
      WFST::TransitMap::Iterator iter2;
      if (iter->impl2 < 0) {
        model->recognition_network.E->Begin(&iter2, iter->state);
        TranslateIterOut(iter, &iter2);

        /* impl6 = true if multiple transitions */
        model->recognition_network.E->Next(&iter2);
        iter->impl6 = !iter2.done;
      }
      else {
        TranslateIterIn(iter, &iter2);
        model->recognition_network.E->Next(&iter2);
        TranslateIterOut(iter, &iter2);
      }
    }
    void TranslateIterIn(Iterator *iter, WFST::TransitMap::Iterator *iter2) {
      iter2->done = iter->done;
      iter2->impl1 = iter->impl2;
      iter2->impl2 = iter->impl3;
      iter2->impl3 = -1;
      iter2->prevState = iter->impl4;
    }
    void TranslateIterOut(Iterator *iter, WFST::TransitMap::Iterator *iter2) {
      iter->impl2 = iter2->impl1;
      iter->impl3 = iter2->impl2;
      iter->impl4 = iter2->prevState;
      if ((iter->done = iter2->done)) return;

      iter->state = iter2->nextState;
      iter->state2 = iter->impl5;
      iter->emission_index = iter2->in;
      iter->out = iter2->out;
      iter->cost = use_transition_prob ? -iter2->weight /* negative log prob */ : 0;
    }
    int Id(int Lstate) { return Lstate; }
  };

  struct DynamicComposer : public TransitMap { 
    DynamicComposer(RecognitionModel *M, bool UT=false) : TransitMap(M, UT) {}
    virtual void Next(Iterator *iter) {
      TransitMap::Next(iter);
      if (iter->done || !iter->out) return;

      HMM::Token *t = FromVoid<HMM::Token*>(iter->impl1);
      WFST::Composer::trip q(t->ind, t->ind2, 0);
      WFST::Edge e1(t->ind, iter->state, iter->emission_index, iter->out, iter->cost), e2;
      int matched = model->composer.ComposeRight(q.second, e1.out, e1, 0, q.third, &e2);
      if (!matched) matched = model->reachable(q.second, e1.out, 0, &e2);
      if (matched != 1) { ERROR("dynamicComposer: no match for ", q.second, ", ", e1.out, " (", matched, ")"); return; }

      iter->state2 = e2.nextState;
      iter->cost += FLAGS_language_model_weight * -e2.weight;
      // INFO("tx ", model->recognition_network_out.name(e1.prevState), " -> ", model->recognition_network_out.name(e1.out), " @ ", e2.weight);
    }
  };

  struct Emission : public HMM::Emission {
    RecognitionModel *model;
    HMM::ObservationInterface *observed;
    TransitMap *transit;
    bool use_prior_prob;

    Allocator *alloc;
    HMM::ActiveStateIndex beam;
    double *emission;

    ~Emission() { if (alloc) alloc->Free(emission); }
    Emission(RecognitionModel *M, HMM::ObservationInterface *O, TransitMap *T, Allocator *Alloc=0, bool UP=false) :
      model(M), observed(O), transit(T), use_prior_prob(UP),
      alloc(Alloc?Alloc:Singleton<MallocAllocator>::Get()), beam(model->emissions, 1, model->emissions, 0, alloc),
      emission(FromVoid<double*>(alloc->Malloc(model->emissions*sizeof(double)))) {}

    double *Observation(int t) { return observed->Observation(t); }
    int Observations() { return observed->Observations(); }

    double *Posterior(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) { return 0; }
    double Prior(HMM::ActiveState *actiae, HMM::ActiveState::Iterator *state) { return 0; }

    double Prob(HMM::ActiveState *active, HMM::ActiveState::Iterator *state) {
      int emission_index = static_cast<HMM::TokenPasser<HMM::Token>*>(active)->active[state->impl-1].emission_index;
      return emission[emission_index];
    }

    double Prob(TransitMap::Iterator *iter) { return emission[iter->emission_index]; }

    void Calc(HMM::ActiveState *active, int t) {
      if (t) FATAL("unexpected ", t); /* bootstrap t=0 */

      HMM::TokenPasser<HMM::Token> *beam = static_cast<HMM::TokenPasser<HMM::Token>*>(active);
      beam->count = 1;
      beam->active[0].ind = 0;
      beam->active[0].ind2 = 0;
      beam->active[0].val = -INFINITY;
      beam->active[0].emission_index = 0;
      beam->active[0].out = 0;
      beam->active[0].backtrace = 0;
      beam->active[0].tstate = 0;
      beam->active[0].steps = 1;

      CalcNext(active, transit, t);
    }

    void CalcNext(HMM::ActiveState *active, HMM::TransitMap *transit, int t) {
      Timer t1;
      active->time_index = t;
      memset(beam.active, -1, model->emissions*sizeof(int));

      HMM::ActiveState::Iterator LstateI;
      for (active->Begin(t, &LstateI); !LstateI.done; active->Next(&LstateI)) {
        int emission_index = static_cast<HMM::TokenPasser<HMM::Token>*>(active)->active[LstateI.impl-1].emission_index;
        beam.active[emission_index] = emission_index;

        HMM::TransitMap::Iterator RstateI;
        for (transit->Begin(&RstateI, active, &LstateI); !RstateI.done; transit->Next(&RstateI)) 
          beam.active[RstateI.emission_index] = RstateI.emission_index;
      }

      sort(beam.active, beam.active + model->emissions, IntSort);
      for (beam.count=0; beam.count < model->emissions && beam.active[beam.count] >= 0; beam.count++) {}

      Timer t2;
      for (int i=0; i<model->emissions; i++) emission[i] = -INFINITY;
      AcousticHMM::EmissionArray::Calc(&model->acoustic_model, &beam, t, Observation(t), emission);
      // INFO("calc ", beam.count, " in " t1.time()*1000, " ms (emitcalc ", t2.time()*1000, " ms));
    }

    static bool IntSort(const int x, const int y) { return x > y; }
  };

  struct TokenPasser : public HMM::TokenPasser<HMM::Token> {
    RecognitionModel *model;

    TokenPasser(RecognitionModel *M, int NS, int NB, int BW, HMM::TokenBacktrace<HMM::Token> *BT, int Scale=100, Allocator *Alloc=0) :
      HMM::TokenPasser<HMM::Token>(NS, NB, BW, BT, Scale, Alloc), model(M) {}

    virtual double *Out(int time_index, ActiveState::Iterator *left, TransitMap::Iterator *right, double **trace) {
      double *ret = HMM::TokenPasser<HMM::Token>::Out(time_index, left, right, trace);
      if (!ret) return 0;

      HMM::Token *parent = &active[left->impl-1], *t = &nextActive[nextCount-1];
      if (t->tstate && right->impl6) { t->tstate=0; t->steps=0; }
      if (t->out) t->tstate = 1;
      return ret;
    }
  };

  static double Viterbi(RecognitionModel *model, Matrix *observations, matrix<HMM::Token> *path, double beam_width, bool use_transit=0, LFL::TokenNameCB<HMM::Token> *ncb=0) {
    int NBest=1, M=observations->M;
    DynamicComposer transit(model, use_transit);
    HMM::ObservationMatrix om(observations);
    Emission emit(model, &om, &transit);

    matrix<HMM::Token> backtrace(M, beam_width, HMM::Token());
    HMM::TokenBacktraceMatrix<HMM::Token> tbt(&backtrace);
    TokenPasser beam(model, 1, NBest, beam_width, &tbt);

    int endstate = HMM::Forward(&beam, &transit, &emit, &beam, &beam, 0);
    HMM::Token::TracePath(path, &backtrace, endstate, M);
    return backtrace.row(M-1)[endstate].val;
  }

  static void PrintLattice(RecognitionModel *recognize) {}
};

struct Recognizer {
  static matrix<HMM::Token> *DecodeFile(RecognitionModel *model, const char *fn, double beam_width=0,
                                        bool use_transit=0, double *vprobout=0, Matrix **MFCCout=0, TokenNameCB<HMM::Token> *nameCB=0) {
    SoundAsset input("input", fn, 0, 0, 0, 0);
    input.Load();
    if (!input.wav) { ERROR(fn, " not found"); return 0; }

    if (MFCCout) *MFCCout = Features::FromAsset(&input, Features::Flag::Storable);
    Matrix *features = Features::FromAsset(&input, Features::Flag::Full);
    matrix<HMM::Token> *ret = DecodeFeatures(model, features, beam_width, use_transit, vprobout, nameCB);
    delete features;
    return ret;
  }

  static matrix<HMM::Token> *DecodeFeatures(RecognitionModel *model, Matrix *features, double beam_width,
                                            bool use_transit=0, double *vprobout=0, TokenNameCB<HMM::Token> *nameCB=0) {
    if (!DimCheck("decodeFeatures", features->N, model->acoustic_model.state[0].emission.mean.N)) return 0;
    matrix<HMM::Token> *viterbi = new matrix<HMM::Token>(features->M, 1, HMM::Token());
    double vprob = RecognitionHMM::Viterbi(model, features, viterbi, beam_width, use_transit, nameCB);
    if (vprobout) *vprobout = vprob;
    return viterbi;
  }

  static AcousticModel::Compiled *DecodedAcousticModel(RecognitionModel *model, matrix<HMM::Token> *viterbi, Matrix *vout) {
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
      int pp, pn, phone = AcousticModel::ParseName(model->acoustic_model.state[emission_index].name, 0, &pp, &pn);
      AcousticModel::State::Assign(hmm->state, &count, hmm->states, &model->acoustic_model, 0, phone, 0, 1);
    }
    if (count > hmm->states) FATAL("overflow ", count, " > ", hmm->states);
    return hmm;
  }

  struct WordIter {
    const RecognitionModel *model;
    const matrix<HMM::Token> *m;
    int beg, end, impl, adjustforward;
    string word;

    WordIter(const RecognitionModel *Model, const matrix<HMM::Token> *paths) : model(Model), m(paths), beg(0), end(0), impl(0), adjustforward(0) { Next(); }

    bool Done() { return impl >= m->M; }
    void Next() {
      for (/**/; impl<m->M; impl++) {
        const HMM::Token *t = m->row(impl), *t2 = 0;
        if (!t->out) continue;

        word = model->recognition_network_out.Name(t->out);
        beg = impl++ - t->steps + adjustforward;
        adjustforward = 0;

        for (/**/; impl<m->M; impl++) {
          if (!(t2 = m->row(impl))->ind) break;
          if (t2->tstate) continue;

          int adjustbegin = impl, tps = AcousticModel::StatesPerPhone-1, ps = -1;
          if (impl) {
            if (!(t2 = m->row(impl-1))->ind) break;
            ps = PhonemeState(t2);
          }

          if (!impl || ps != tps) {
            ps = SeekPhonemeState(tps);
            if (ps != tps) break;
          }

          ps = SeekPhonemeState(0);
          adjustforward = impl - adjustbegin;
          break;
        }

        end = impl-1;
        break;
      }
    }

    int PhonemeState(const HMM::Token *t) {
      int ps = -1;
      AcousticModel::ParseName(model->acoustic_model.state[t->emission_index].name, &ps);
      return ps;
    }

    int SeekPhonemeState(int tps) {
      int ps = -1;
      for (const HMM::Token *t; impl<m->M; impl++) 
        if (!(t = m->row(impl))->ind || (ps = PhonemeState(t)) == tps) break;
      return ps;
    }
  };

  static string Transcript(const RecognitionModel *model, const matrix<HMM::Token> *viterbi, bool annotate=false) {
    Recognizer::WordIter iter(model, viterbi); string v;
    for (/**/; !iter.Done(); iter.Next()) {
      if (!iter.word.size()) continue;
      v += iter.word;
      if (annotate) StringAppendf(&v, "-%d(%d,%d) ", iter.word.c_str(), iter.end-iter.beg, iter.beg, iter.end);
      else v += " ";
    }
    return v;
  }

  static double WordErrorRate(const RecognitionModel *model, string gold, string x) {
    vector<int> A, B;
    LFL::StringWordIter aw(gold), bw(x);
    for (string w = aw.NextString(); !aw.Done(); w = aw.NextString()) A.push_back(model->recognition_network_out.Id(tolower(w).c_str()));
    for (string w = bw.NextString(); !bw.Done(); w = bw.NextString()) B.push_back(model->recognition_network_out.Id(tolower(w).c_str()));
    return double(Levenshtein(A, B)) / A.size();
  }
};

}; // namespace LFL
#endif // LFL_SPEECH_RECOGNITION_H__
