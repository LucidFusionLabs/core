/*
 * $Id: hmm.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef LFL_LFAPP_HMM_H__
#define LFL_LFAPP_HMM_H__
namespace LFL {

struct NameCB {
  virtual string Name(int id) = 0;
};

template <class X> struct TokenNameCB {
  virtual string Name(X &id) = 0;
};

struct HMM {
  template <class T> struct SortPairT {
    T val; int ind;
    SortPairT() : val(-INFINITY), ind(-1) {}

    bool operator<(const SortPairT &r) const {
      int ret = DoubleSort(val, r.val);
      if      (ret < 0) return true;
      else if (ret > 0) return false;
      else {
        if      (ind < r.ind) return false;
        else if (ind > r.ind) return true;
        else return false;
      }
    }
  };
  typedef SortPairT<double> SortPair;

  static void SortPairs(double *alpha, double *backtrace, SortPair *sortMap, int num) { 
    for (int i=0; i<num; i++) {
      sortMap[i].val = alpha[i];
      sortMap[i].ind = backtrace ? backtrace[i] : i;
    }
    sort(sortMap, sortMap + num);
  }

  static void SortBest(double *alpha, double *backtrace, int NBest) {
    vector<SortPair> sortMap(NBest);
    SortPairs(alpha, backtrace, &sortMap[0], NBest);

    for (int i=0; i<NBest; i++) {
      alpha[i] = sortMap[i].val;
      backtrace[i] = sortMap[i].ind;
    }
  }

  struct ActiveState {
    int NumStates, NBest, BeamWidth, time_index;
    Allocator *alloc;

    ActiveState(int NS, int NB, int BW, Allocator *Alloc) : NumStates(NS), NBest(NB), BeamWidth(BW), time_index(-1), alloc(Alloc ? Alloc : Singleton<MallocAllocator>::Get()) {}
    virtual ~ActiveState() {}

    struct Iterator {
      int index, impl; bool done;
      Iterator() : index(-1), impl(-1), done(0) {}
      Iterator(int ind) : index(ind), impl(ind+1), done(0) {}
      string DebugString() const { return StrCat("ASit ", index, ", ", impl, ", ", done); }
    };

    virtual int Size() { return BeamWidth; }
    virtual void Next(Iterator *iter) = 0;
    virtual int Update(double *alpha) = 0;

    virtual void PrintTrellis(double *row, int rowind, NameCB *nameCB) { 
      string v = StrCat("t[", rowind, "]: ");
      for (int state=0; state<NumStates; state++) {
        for (int k=0; k<NBest; k++) {
          double val = row[state*NBest + k];
          if (isinf(val)) continue;
          StrAppend(&v, nameCB->Name(state+1), " [", k, "] = ", val);
        }
      }
      INFO(v);
    }

    void Reset(Iterator *iter) { iter->index=0; iter->impl=0; iter->done=0; }
    void Begin(int t, Iterator *iter) { time_index=t; Reset(iter); Next(iter); }
  };

  struct TransitMap {
    struct Iterator {
      unsigned state, state2, emission_index, out; float cost; bool done;
      void *impl1; int impl2, impl3, impl4, impl5, impl6;
      string tostr() { return StringPrintf("TXit %d, %d, %d, %d, %f, %d, %p, %d %d %d %d %d %d", state, state2, emission_index, out, cost, done, impl1, impl2, impl3, impl4, impl5, impl6); }
    };

    virtual void Begin(Iterator *iter, ActiveState *active, ActiveState::Iterator *Lstate) = 0;
    virtual void Next(Iterator *iter) = 0;

    virtual int Id(int Lstate) = 0;
  };

  struct ObservationInterface {
    virtual double *Observation(int t) = 0;
    virtual int Observations() = 0;
  };

  struct Emission : public ObservationInterface {
    virtual void Calc(ActiveState *active, int t) = 0;
    virtual void CalcNext(ActiveState *active, TransitMap *transit, int t) = 0;

    virtual double  Prior    (ActiveState *active, ActiveState::Iterator *state) = 0;
    virtual double *Posterior(ActiveState *active, ActiveState::Iterator *state) = 0;
    virtual double  Prob     (ActiveState *active, ActiveState::Iterator *state) = 0;

    virtual double Prob(TransitMap::Iterator *iter) = 0;
  };

  struct Beam {
    virtual double  *In(int t, ActiveState::Iterator *l=0, TransitMap::Iterator *r=0)                   = 0;
    virtual double *Out(int t, ActiveState::Iterator *l=0, TransitMap::Iterator *r=0, double **trace=0) = 0;
  };

  struct Algorithm {
    struct Interface { 
      virtual void Apply(ActiveState::Iterator *, TransitMap::Iterator *, double prob, int NBest, double *out, double *traceout) = 0;
    };
    struct Forward : public Interface {
      void Apply(ActiveState::Iterator *, TransitMap::Iterator *, double prob, int NBest, double *out, double *) { LogAdd(out, prob); }
    };
    struct Viterbi : public Interface {
      void Apply(ActiveState::Iterator *Lstate, TransitMap::Iterator *Rstate, double prob, int NBest, double *out, double *trace) {
        return Apply(NBest, prob, Lstate->index, out, trace);
      }
      void Apply(int NBest, double prob, int backtrace, double *out, double *trace) {
        if (DoubleSort(out[NBest-1], prob) > 0) {
          out[NBest-1] = prob;
          if (trace) trace[NBest-1] = backtrace;
          if (NBest > 1) SortBest(out, trace, NBest);
        }
      }
    };
  };

  struct ActiveStateIndex : public ActiveState {
    int *active, count, init_max;
    ~ActiveStateIndex() { alloc->Free(active); }
    ActiveStateIndex(int NS, int NB, int BW, int InitMax=0, Allocator *Alloc=0) :
      ActiveState(NS, NB, BW, Alloc), active(static_cast<int*>(alloc->Malloc(sizeof(int)*BeamWidth))),
      count(0), init_max(InitMax) { if (!active) FATAL("alloc failed"); }

    int Size() { 
      if (!time_index) return init_max ? init_max : NumStates;
      return count;
    }

    void Next(Iterator *iter) {
      if (!time_index) {
        if (iter->impl >= NumStates || (init_max && iter->impl >= init_max)) { iter->done=1; return; }
        iter->index = NBest * iter->impl++;
      }
      else {
        if (iter->impl >= count) { iter->done=1; return; }
        iter->index = active[iter->impl++];
      }
    }

    virtual int Update(double *alpha) {
      int num = NumStates*NBest;
      vector<SortPair> sortMap(num);
      SortPairs(alpha, 0, &sortMap[0], num);
      for (count=0; count<num && count<BeamWidth; count++) active[count] = sortMap[count].ind;
      return sortMap[0].ind;
    }
  };

  struct TransitMapMatrix : public TransitMap {
    const Matrix *transit;
    TransitMapMatrix(const Matrix *T) : transit(T) {}

    void Begin(Iterator *iter, ActiveState *active, ActiveState::Iterator *Lstate) {
      iter->impl1 = Void(transit->row(Lstate->index / active->NBest));
      iter->impl2 = 0;
      iter->state2 = 0;
      iter->done = 0;
      return Next(iter);
    }
    void Next(Iterator *iter) {
      if (iter->impl2 >= transit->N) { iter->done=1; return; }
      iter->state = iter->impl2++;
      iter->cost = static_cast<double*>(iter->impl1)[iter->state];
    }
    int Id(int Lstate) { return Lstate; }
  };

  struct ObservationMatrix : public ObservationInterface {
    Matrix *observed;
    ObservationMatrix(Matrix *O) : observed(O) {}
    double *Observation(int t) { return observed->row(t); }
    int Observations() { return observed->M; }
  };

  struct ObservationPtr : public ObservationInterface {
    double *v; int num;
    ObservationPtr() : v(0), num(1) {}
    double *Observation(int t) { return v; }
    int Observations() { return num; }
  };

  struct EmissionMatrix : public Emission {
    Matrix *emission;
    double *row, *priorRow;
    int *obind, obs;

    EmissionMatrix(Matrix *emit, double *Prior=0, int *ObservationIndex=0, int Observations=0) : emission(emit), row(0), priorRow(Prior), obind(ObservationIndex), obs(Observations) {}

    double *Observation(int t) { return 0; }
    int Observations() { return obs ? obs : emission->M; }
    void Calc(ActiveState *active, int ti) { row = emission->row(obind ? obind[ti] : ti); }
    void CalcNext(ActiveState *active, TransitMap *transit, int ti) { Calc(active, ti); }
    virtual double Prior(ActiveState *active, ActiveState::Iterator *state) { return priorRow ? priorRow[state->index / active->NBest] : 0; }
    double *Posterior(ActiveState *active, ActiveState::Iterator *state) { return 0; }
    double Prob(ActiveState *active, ActiveState::Iterator *state) { return row[state->index / active->NBest]; }
    double Prob(TransitMap::Iterator *iter) { return row[iter->state]; }
  };

  struct BeamMatrix : public Beam {
    Matrix *lambda, *backtrace;
    int NBest;
    BeamMatrix(Matrix *L, Matrix *BT, int NB) : lambda(L), backtrace(BT), NBest(NB) {}

    double  *In(int time_index, ActiveState::Iterator *left, TransitMap::Iterator *) { return &lambda->row(time_index-1)[left ? left->index : 0]; }
    double *Out(int time_index, ActiveState::Iterator *, TransitMap::Iterator *right, double **trace) {
      int ind = right ? right->state*NBest : 0;
      if (trace) *trace = backtrace ? &backtrace->row(time_index)[ind] : 0;
      return &lambda->row(time_index)[ind];
    }
  };

  struct Token : public SortPair {
    int ind2, out; double backtrace; unsigned short emission_index, tstate; int steps;
    Token() : ind2(0), out(0), backtrace(-1), emission_index(-1), tstate(0), steps(0) {}
    bool operator==(const Token &r) { return ind2 == r.ind2 && out == r.out && backtrace == r.backtrace && emission_index == r.emission_index && tstate == r.tstate && steps == r.steps; }
    bool operator!=(const Token &r) { return !(*this == r); }

    static bool Merge(const Token &l, const Token &r) { return l.ind == r.ind && l.ind2 == r.ind2 && l.out == r.out && l.emission_index == r.emission_index; }
    static bool CmpInd(const Token &l, const Token &r) {
      if      (l.ind < r.ind) return true;
      else if (l.ind > r.ind) return false;

      else if (l.ind2 < r.ind2) return true;
      else if (l.ind2 > r.ind2) return false;

      else if (l.emission_index < r.emission_index) return true;
      else if (l.emission_index > r.emission_index) return false;

      else if (l.out < r.out) return true;
      else if (l.out > r.out) return false;

      else return l.val > r.val;
    }
    static int TracePath(matrix<Token> *viterbi, matrix<Token> *backtrace, int endindex, int len, bool returnMerged=false) {
      Token *t = &backtrace->row(len-1)[endindex];
      viterbi->row(len-1)[0] = *t;

      for (int i=len-1; i>0; i--) {
        t = &backtrace->row(i-1)[int(t->backtrace)];
        if (returnMerged && viterbi->row(i-1)[0] == *t) return i-1;
        viterbi->row(i-1)[0] = *t;
      }
      return 0;
    }
    static void PrintViterbi(matrix<Token> *vp, NameCB *nameCB) {
      string v = StrCat("vitebri(", vp->M, "): ");
      MatrixRowIter(vp) {
        int id = vp->row(i)->ind;
        StringAppendf(&v, "%03d : %s, ", i, nameCB->Name(id).c_str());
        if ((i+1) % 32 == 0) { INFO(v); v.clear(); }
      }
      if (v.size()) INFO(v);
    }
    static void PrintViterbi(matrix<Token> *vp, TokenNameCB<Token> *nameCB) {
      string v = StrCat("vitebri(", vp->M, "): ");
      MatrixRowIter(vp) {
        StringAppendf(&v, "%03d : %s, ", i, nameCB->Name(vp->row(i)[0]).c_str());
        if ((i+1) % 32 == 0) { INFO(v); v.clear(); }
      }
      if (v.size()) INFO(v);
    }
  };

  template <class T> struct TokenBacktrace { virtual T *row() = 0; };

  template <class T> struct TokenBacktracePtr : public TokenBacktrace<T> {
    T *v;
    TokenBacktracePtr() : v(0) {}
    T *row() { return v; }
  };

  template <class T> struct TokenBacktraceMatrix : public TokenBacktrace<T> {
    matrix<T> *bt; int count;
    TokenBacktraceMatrix(matrix<T> *BT) : bt(BT), count(0) {}
    T *row() { return bt->row(count++ % bt->M); }
  };

  template <class T> struct TokenPasser : public ActiveState, public Beam, public Algorithm::Interface {
    TokenBacktrace<T> *backtrace;
    int num, count, nextCount;
    Algorithm::Viterbi *viterbi;
    T *active, *nextActive;

    virtual ~TokenPasser() { if (alloc) { alloc->Free(active); alloc->Free(nextActive); } }
    TokenPasser(int NS, int NB, int BW, TokenBacktrace<T> *BT, int Scale=100, Allocator *Alloc=0) : ActiveState(NS, NB, BW, Alloc), backtrace(BT), num(BeamWidth*Scale),
    count(0), nextCount(0), viterbi(Singleton<Algorithm::Viterbi>::Get()), active(static_cast<T*>(alloc->Malloc(sizeof(T)*num))), nextActive(static_cast<T*>(alloc->Malloc(sizeof(T)*num))) {
      if (!active) FATAL("alloc failed");
      ClearNext();
    }
    void ClearNext() { for (int i=0; i<num; i++) nextActive[i].val = -INFINITY; }

    /* ActiveState */
    virtual int Size() { return !time_index ? NumStates : count; }
    virtual void Next(Iterator *iter) {
      if (!time_index) { /* patch t=0 */
        count = NumStates;
        if (iter->impl >= count) { iter->done=1; return; }
        active[iter->impl].ind = iter->impl;
      }
      else if (iter->impl >= count) { iter->done=1; return; }
      iter->index = active[iter->impl++].ind;
    }

    virtual int Update(double *) {
      if (backtrace) memcpy(backtrace->row(), active, BeamWidth*sizeof(T));
      if (!time_index) return 0;
      if (!nextCount) return count=0; 

      /* sort by index */
      sort(nextActive, nextActive + nextCount, T::CmpInd);

      /* compact */
      T last;
      for (int i=0; i<nextCount; i++) {
        if (T::Merge(nextActive[i], last)) nextActive[i].val = -INFINITY;
        else last = nextActive[i];
      }

      /* sort by value */
      sort(nextActive, nextActive + nextCount);

      count = nextCount = 0;
      for (int i=0; i<BeamWidth && !isinf(nextActive[i].val); i++) active[count++] = nextActive[i];
      ClearNext();
      return 0;
    }

    /* Beam */
    virtual double  *In(int time_index, ActiveState::Iterator *left, TransitMap::Iterator *) { return &active[left->impl-1].val; }
    virtual double *Out(int time_index, ActiveState::Iterator *left, TransitMap::Iterator *right, double **trace) {
      if (!right) return 0;
      if (nextCount >= num) FATAL("beam sort width ", num, " exceeded");
      T *parent = &active[left->impl-1];
      T *t = &nextActive[nextCount++];
      t->ind = right->state * NBest;
      t->ind2 = right->state2;
      t->tstate = parent->tstate;
      t->emission_index = right->emission_index;
      t->out = right->out;
      t->steps = parent->ind ? parent->steps + 1 : 0;
      if (trace) *trace = &t->backtrace;
      return &t->val;
    }

    /* Algorithm */
    virtual void Apply(ActiveState::Iterator *Lstate, TransitMap::Iterator *, double prob, int NBest, double *out, double *trace) {
      return viterbi->Apply(NBest, prob, Lstate->impl-1, out, trace);
    }
  };

  static int Forward(ActiveState *active, TransitMap *transit, Emission *emission, Beam *beam, Algorithm::Interface *algo, NameCB *nameCB=0, bool init=1, int T=0)
  {
    int NBest=active->NBest, endindex=0, len=emission->Observations();
    for (int i=0; i<len; i++) {
      int t = T + i;
      if (!t && init) emission->Calc(active, t);
      else emission->CalcNext(active, transit, t);

      ActiveState::Iterator LstateI;
      for (active->Begin(t, &LstateI); !LstateI.done; active->Next(&LstateI)) {

        if (!t && init) { *beam->In(t+1, &LstateI) = emission->Prior(active, &LstateI) + emission->Prob(active, &LstateI); continue; }

        TransitMap::Iterator RstateI;
        for (transit->Begin(&RstateI, active, &LstateI); !RstateI.done; transit->Next(&RstateI)) {

          double last_prob = *beam->In(t, &LstateI, &RstateI);
          if (isinf(last_prob)) continue;

          double trans = RstateI.cost;
          double emit = emission->Prob(&RstateI);

          double prob = last_prob + trans + emit;
          double *emited, *trace, *out = beam->Out(t, &LstateI, &RstateI, &trace);

          algo->Apply(&LstateI, &RstateI, prob, NBest, out, trace);
        }
      }

      if (nameCB) active->PrintTrellis(beam->Out(t), t, nameCB);
      endindex = active->Update(beam->Out(t));
    }

    return endindex;
  }

  static int Viterbi(Matrix *emission, Matrix *transition, Matrix *lambda, Matrix *backtrace,
                     int NBest, int BeamWidth, NameCB *nameCB)
  {
    ActiveStateIndex active(lambda->N/NBest, NBest, BeamWidth);
    TransitMapMatrix transit(transition);
    EmissionMatrix emit(emission);
    BeamMatrix beam(lambda, backtrace, NBest);
    return Forward(&active, &transit, &emit, &beam, Singleton<Algorithm::Viterbi>::Get(), nameCB);
  }

  struct BaumWelchAccum {
    virtual ~BaumWelchAccum() {}
    virtual void AddPrior(int state, double pi) = 0;
    virtual void AddEmission(int state, const double *feature, const double *emissionPosterior, double emissionTotal, double gamma) = 0;
    virtual void AddTransition(int Lstate, int Rstate, double xi) = 0;
  };

  struct BaumWelchWrapper : public BaumWelchAccum {
    struct Prior { 
      int state; double pi;
      Prior(int S, double P) : state(S), pi(P) {}
    };
    typedef vector<Prior> PriorList;
    PriorList prior;
    virtual void AddPrior(int state, double pi) { prior.push_back(Prior(state, pi)); }

    struct Emission {
      int state; const double *feature, *emissionPosterior; double emissionTotal, gamma;
      Emission(int S, const double *F, const double *EP, double ET, double G) : state(S), feature(F), emissionPosterior(EP), emissionTotal(ET), gamma(G) {}
    };
    typedef vector<Emission> EmissionList;
    EmissionList emission;
    virtual void AddEmission(int state, const double *feature, const double *emissionPosterior, double emissionTotal, double gamma) { emission.push_back(Emission(state, feature, emissionPosterior, emissionTotal, gamma)); }

    struct Transition {
      int Lstate, Rstate; double xi;
      Transition(int LS, int RS, double X) : Lstate(LS), Rstate(RS), xi(X) {}
    };
    typedef vector<Transition> TransitionList;
    TransitionList transition;
    virtual void AddTransition(int Lstate, int Rstate, double xi) { transition.push_back(Transition(Lstate, Rstate, xi)); }

    void Add(BaumWelchAccum *out) {
      for (PriorList::iterator i = prior.begin(); i != prior.end(); i++) out->AddPrior((*i).state, (*i).pi);
      for (EmissionList::iterator i = emission.begin(); i != emission.end(); i++) out->AddEmission((*i).state, (*i).feature, (*i).emissionPosterior, (*i).emissionTotal, (*i).gamma);
      for (TransitionList::iterator i = transition.begin(); i != transition.end(); i++) out->AddTransition((*i).Lstate, (*i).Rstate, (*i).xi);
    }
  };

  static double ForwardBackward(ActiveState *active, TransitMap *transit, Emission *emission,
                                Matrix *alpha, Matrix *beta, Matrix *gamma, Matrix *xi,
                                double *backwardTotalOut=0, BaumWelchAccum *BWaccum=0)
  {
    int NumStates=alpha->N, len=emission->Observations();

    /* go forward */
    BeamMatrix beam(alpha, 0, active->NBest);
    int final_state = Forward(active, transit, emission, &beam, Singleton<Algorithm::Forward>::Get());

    double forwardTotal = -INFINITY, backwardTotal = -INFINITY;
    for (int j=0; j<NumStates; j++) LogAdd(&forwardTotal, alpha->row(len-1)[j]);
    for (int j=0; j<NumStates; j++) beta->row(len-1)[j] = 0;
    if (isnan(forwardTotal) || isinf(forwardTotal)) return forwardTotal;

    /* go backward */
    for (int i=len-1; i>=0; i--) {
      if (i!=len-1) emission->CalcNext(active, transit, i);

      ActiveState::Iterator LstateI;
      for (active->Begin(i, &LstateI); !LstateI.done; active->Next(&LstateI)) {

        int Lstate = LstateI.index;
        double prob = alpha->row(i)[Lstate] + beta->row(i)[Lstate]; 
        double LstateProb = prob - forwardTotal;
        gamma->row(i)[Lstate] = LstateProb;
        if (BWaccum && !isinf(LstateProb)) {
          if (!i) BWaccum->AddPrior(transit->Id(Lstate), LstateProb);
          BWaccum->AddEmission(transit->Id(Lstate), emission->Observation(i), emission->Posterior(active, &LstateI), emission->Prob(active, &LstateI), LstateProb);
        }
        if (!i) { if (!isinf(prob)) LogAdd(&backwardTotal, prob); continue; }

        TransitMap::Iterator RstateI;
        for (transit->Begin(&RstateI, active, &LstateI); !RstateI.done; transit->Next(&RstateI)) {

          int Rstate = RstateI.state;
          float trans = RstateI.cost;
          float last_prob = beta->row(i)[Rstate];
          float emit = emission->Prob(&RstateI);

          float prob = last_prob + trans + emit;
          if (isinf(prob)) continue;
          LogAdd(&beta->row(i-1)[Lstate], prob);

          prob += alpha->row(i-1)[Lstate] - forwardTotal;
          LogAdd(&xi->row(Lstate)[Rstate], prob);
          if (BWaccum) BWaccum->AddTransition(transit->Id(Lstate), transit->Id(Rstate), prob);
        }
      }

      if (!i) break;
      active->Update(beta->row(i-1));
    }

    if (backwardTotalOut) *backwardTotalOut = backwardTotal;
    return forwardTotal;
  }

  static double ForwardBackward(Matrix *emission, Matrix *transition, Matrix *alpha, Matrix *beta, Matrix *gamma, Matrix *xi, double *backwardTotal,
                                int NBest, int BeamWidth, NameCB *nameCB)
  {
    ActiveStateIndex active(alpha->N/NBest, NBest, BeamWidth);
    TransitMapMatrix transit(transition);
    EmissionMatrix emit(emission);
    return ForwardBackward(&active, &transit, &emit, alpha, beta, gamma, xi, backwardTotal);
  }

  static void TracePath(Matrix *viterbi, Matrix *backtrace, int endindex, int len) {
    viterbi->row(len-1)[0] = endindex;

    for (int i=len-1; i>0; i--) {
      int ind = int(viterbi->row(i)[0]);
      viterbi->row(i-1)[0] = backtrace->row(i)[ind];
    }
  }

  static double TracePath(Matrix *lambda, Matrix *backtrace, int *path, int endindex, int len, int NumStates, int NBest) {
    float final_prob = lambda->row(len-1)[endindex];
    path[len-1] = endindex;
    for (int i=len-1; i>0; i--) {
      int cur = path[i];
      path[i-1] = backtrace->row(i)[cur];
    }
    return final_prob;
  }

  static void PrintViterbi(Matrix *vp, NameCB *nameCB) {
    string v = StrCat("vitebri(", vp->M, "): ");
    MatrixRowIter(vp) {
      int id = int(vp->row(i)[0]);
      StringAppendf(&v, "%03d : %s, ", i, nameCB->Name(id).c_str());
      if ((i+1) % 32 == 0) { INFO(v); v.clear(); }
    }
    if (v.size()) INFO(v);
  }

  static void PrintForwardBackward(Matrix *alpha, Matrix *beta, Matrix *gamma, Matrix *xi) {
    Matrix::Print(alpha, "alpha");
    Matrix::Print(beta, "beta");

    INFO("gamma (", gamma->M, ",", gamma->N, ")");
    MatrixRowIter(gamma) {
      double sum=0; string s;
      MatrixColIter(gamma) {
        double v = exp(gamma->row(i)[j]);
        sum += v;
        StrAppend(&s, v, ", ");
      }
      INFO("row(", i, ") sum=", sum, " { ", s, " }");
    }
    Matrix::Print(gamma, "gamma");

    Matrix::Print(xi, "xi");
  }
};

struct TestHMM { 
  struct ActiveState : public HMM::ActiveStateIndex {
    ActiveState(int states) : ActiveStateIndex(states, 1, states) { count = states; for (int i=0; i<states; i++) active[i] = i; }
    virtual int update(double *alpha) { return 0; }
  };

  static void ForwardBackward(HMM::ActiveState *active, HMM::TransitMap *transit, HMM::Emission *emit, int M, int N) {
    Matrix alpha(M, N, -INFINITY), beta(M, N, -INFINITY), gamma(M, N, -INFINITY), xi(N, N, -INFINITY);
    HMM::BaumWelchWrapper BWwrapper;
    double bp, fp=HMM::ForwardBackward(active, transit, emit, &alpha, &beta, &gamma, &xi, &bp, &BWwrapper);

    INFO("fp=", fp, " bp=", bp);
    HMM::PrintForwardBackward(&alpha, &beta, &gamma, &xi);

    INFO("xi = ", BWwrapper.transition.size(), " (reverse order timewise)");
    for (HMM::BaumWelchWrapper::TransitionList::iterator it = BWwrapper.transition.begin(); it != BWwrapper.transition.end(); it++) 
      INFO((*it).Lstate, " -> ", (*it).Rstate, " : ", (*it).xi);
  }

  static void Test() {
    /* Example of the Baum-Welch Algorithm (http://www.indiana.edu/~iulg/moss/hmmcalculations.pdf)
     *
     *   />>\   />>.7>>\   />>\
     * .3     s          t     .9
     *   \<</   \<<.1<</   \<</
     *
     * Starting probability of s is .85, of t is .15. In s, Pr(A) = .4, Pr(B) = .6. In t, Pr(A) = .5, Pr(B) = .5.
     *
     * We take the set Y of unanalyzed words to be {ABBA,BAB}, and c to be given by c(ABBA) = 10, c(BAB) = 20.
     */

    /* s=0, t=1 */
    double prior[] = { log(.85), log(.15) };
    Matrix transit(2, 2);
    transit.row(0)[0] = log(.3); transit.row(0)[1] = log(.7);
    transit.row(1)[0] = log(.1); transit.row(1)[1] = log(.9);
    HMM::TransitMapMatrix transition(&transit);

    /* A=0, B=1 */
    Matrix emit(2, 2);
    emit.row(0)[0] = log(.4); emit.row(0)[1] = log(.5);
    emit.row(1)[0] = log(.6); emit.row(1)[1] = log(.5);
    HMM::EmissionMatrix emission(&emit, prior);
    ActiveState active(2);

    /* ABBA */
    int utt1[] = { 0, 1, 1, 0 }, utt1len = sizeof(utt1)/sizeof(int);
    emission.obind = utt1;
    emission.obs = utt1len;
    ForwardBackward(&active, &transition, &emission, utt1len, 2);

    /* BAB */
    int utt2[] = { 1, 0, 1 }, utt2len = sizeof(utt2)/sizeof(int);
    emission.obind = utt2;
    emission.obs = utt2len;
    ForwardBackward(&active, &transition, &emission, utt2len, 2);
  }
};

}; // namespace LFL
#endif // LFL_LFAPP_HMM_H__
