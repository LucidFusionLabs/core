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

#ifndef LFL_SPEECH_WFST_H__
#define LFL_SPEECH_WFST_H__
namespace LFL {

/* Weighted Finite State Transducer */
struct WFST {
  enum { NULL1=0, BLOCK=-2, NULL2=-3 }; /* labels */

  struct IOAlphabet {
    virtual int Id(string s) const = 0;
    virtual string Name(int id) const = 0;
    virtual int Size() const = 0;
    virtual const char *FactoryName() const { return 0; }

    struct Iterator { int id; const char *name; bool done; int impl; };
    virtual void Begin(Iterator *) const = 0;
    virtual void Next(Iterator *) const = 0;

    static const unsigned auxiliary_symbol_prefix=0x7f, auxiliary_symbol_shift=24, auxiliary_symbol_spaces=3, auxiliary_symbol_space=0xffffff;
    static unsigned AuxiliarySymbolOffset(unsigned id) { return id & auxiliary_symbol_space; }
    static unsigned AuxiliarySymbolId(unsigned offset, int auxnum=0) { return ((auxiliary_symbol_prefix - auxnum) << auxiliary_symbol_shift) | (offset & auxiliary_symbol_space); }
    static bool AuxiliarySymbol(unsigned id) {
      for (int i=0; i<auxiliary_symbol_spaces; i++) {
        int auxiliary_symbol_pref = (auxiliary_symbol_prefix - i) << auxiliary_symbol_shift;
        if ((id & auxiliary_symbol_pref) == auxiliary_symbol_pref) return true;
      }
      return false;
    }
  };

  struct AlphabetBuilder : public IOAlphabet {
    typedef map<string, int> AMap;
    typedef vector<string> Strvec;
    AMap amap;
    Strvec tostr;
    IOAlphabet *aux;

    AlphabetBuilder(IOAlphabet *Aux=0) : aux(Aux) { tostr.push_back(""); amap[""] = 0; /* null */ } 

    int Id(string s) const { AMap::const_iterator it = amap.find(s); return it != amap.end() ? (*it).second : -1; }
    string Name(int id) const {
      if (aux && IOAlphabet::AuxiliarySymbol(id)) return aux->Name(IOAlphabet::AuxiliarySymbolOffset(id));
      if (id < 0 || id >= tostr.size()) { ERROR("bad id: ", id, " ", aux, " ", IOAlphabet::AuxiliarySymbolOffset(id)); return ""; }
      return tostr[id];
    }
    int Size() const { return tostr.size(); }
    void Begin(Iterator *iter) const { iter->impl = 0; iter->done = 0; Next(iter); }
    void Next(Iterator *iter) const { if ((iter->id = iter->impl++) >= tostr.size()) iter->done = 1; else iter->name = tostr[iter->id].c_str(); }
    int Add(string s) { 
      int ret = Id(s);
      if (ret >= 0) return ret;
      ret = tostr.size();
      tostr.push_back(s);
      amap[s] = ret;
      return ret;
    }
  };

  struct StringFileAlphabet : public IOAlphabet {
    unique_ptr<vector<string>> str;
    HashMatrix map;
    IOAlphabet *aux;

    static const int HashBuckets = 5, HashValues = 2;
    StringFileAlphabet(vector<string> *S=0, Matrix *M=0, IOAlphabet *Aux=0) : str(S), map(M, HashValues), aux(Aux) {}
    virtual ~StringFileAlphabet() { Reset(); }
    void Reset() { str.reset(); map.map.reset(); }

    int Id(string s) const { return Id(fnv32(s.c_str(), s.size())); }
    int Id(unsigned hash) const { const double *row = map.Get(hash); return row ? row[1] : -1; }
    string Name(int id) const {
      if (aux && IOAlphabet::AuxiliarySymbol(id)) return aux->Name(IOAlphabet::AuxiliarySymbolOffset(id));
      if (id < 0 || id >= str->size()) { ERROR("bad id: ", id); return ""; }
      return (*str)[id];
    }
    int Size() const { return str->size(); }
    void Begin(Iterator *iter) const { iter->impl = 0; iter->done = 0; Next(iter); }
    void Next(Iterator *iter) const { if ((iter->id = iter->impl++) >= str->size()) iter->done = 1; else iter->name = (*str)[iter->id].c_str(); }

    int Read(const char *dir, const char *name1, const char *name2, int iteration) {
      unique_ptr<Matrix> mapdata;
      if (StringFile::ReadVersioned(dir, name1, name2, &str,     0, iteration)<0) { ERROR(name2, ".", iteration, ".in.string"); return -1; }
      if (MatrixFile::ReadVersioned(dir, name1, name2, &mapdata, 0, iteration)<0) { ERROR(name2, ".", iteration, ".in.matrix"); return -1; }
      map.map = move(mapdata);
      return 0;
    }

    static int Write(IOAlphabet *A, const char *dir, const char *name1, const char *name2, int iteration) {
      LocalFile out(string(dir) + MatrixFile::Filename(name1, name2, "string", iteration), "w");
      MatrixFile::WriteHeader(&out, BaseName(out.Filename()), A->FactoryName() ? A->FactoryName() : "", A->Size(), 1);
      if (A->FactoryName()) return 0;

      Matrix map_data(NextPrime(A->Size()*4), HashBuckets * HashValues);
      HashMatrix map(&map_data, HashValues);
      for (int i=0, l=A->Size(); i<l; i++) {
        string name = A->Name(i);
        StringFile::WriteRow(&out, name.c_str());
        double *he = map.Set(fnv32(name.c_str()));
        if (!he) FATAL("Matrix hash collision: ", name);
        he[1] = i;
      }
      if (MatrixFile(map.map.get()).WriteVersioned(dir, name1, name2, iteration) < 0) return ERRORv(-1, name1, " write map");
      return 0;
    }
  };

  struct PhonemeAlphabet : public IOAlphabet {
    IOAlphabet *aux;
    PhonemeAlphabet(IOAlphabet *Aux=0) : aux(Aux) {}
    int Id(string s) const { return Phoneme::Id(s.c_str(), s.size()); }
    string Name(int id) const {
      if (aux && IOAlphabet::AuxiliarySymbol(id)) return aux->Name(IOAlphabet::AuxiliarySymbolOffset(id));
      const char *n = Phoneme::Name(id); return n ? n : StrCat(id);
    }
    int Size() const { return LFL_PHONES; }
    void Begin(Iterator *iter) const { iter->impl = 0; iter->done = 0; Next(iter); }
    void Next(Iterator *iter) const { if ((iter->id = iter->impl++) >= LFL_PHONES) iter->done = 1; else iter->name = Name(iter->id).c_str(); }
    const char *FactoryName() const { return "phoneme"; }
  };

  struct AcousticModelAlphabet : public IOAlphabet {
    AcousticModel::Compiled *AM;
    IOAlphabet *aux;
    AcousticModelAlphabet(AcousticModel::Compiled *am, IOAlphabet *Aux=0) : AM(am), aux(Aux) {}
    int Id(string s) const { return Phoneme::Id(s.c_str(), s.size()); }
    string Name(int id) const {
      if (aux && IOAlphabet::AuxiliarySymbol(id)) return aux->Name(IOAlphabet::AuxiliarySymbolOffset(id));
      return (id < 0 || id >= AM->state.size()) ? "" : AM->state[id].name;
    }
    int Size() const { return AM->state.size(); }
    void Begin(Iterator *iter) const { iter->impl = 0; iter->done = 0; Next(iter); }
    void Next(Iterator *iter) const { if ((iter->id = iter->impl++) >= AM->state.size()) iter->done = 1; else iter->name = Name(iter->id).c_str(); }
    const char *FactoryName() const { return "AcousticModel"; }
  };

  struct StateSet {
    virtual ~StateSet() {}
    virtual int States() = 0;
    virtual int State(int ind) = 0;
    virtual int Find(int state) = 0;

    virtual void Clear() = 0;
    virtual void Add(int state) = 0;
    virtual void Join(vector<int> &merged) = 0;
  };

  struct Statevec : public vector<int>, StateSet {
    Statevec() {}
    Statevec(const Statevec &copy) : vector<int>(copy) {}
    Statevec(const vector<int> &copy) : vector<int>(copy) {}
    Statevec(const set<int> &copy) { for (set<int>::const_iterator i = copy.begin(); i != copy.end(); i++) push_back(*i); }
    Statevec(StateSet *copy) { for (int i=0, l=copy->States(); i<l; i++) push_back(copy->State(i)); Sort(); }

    int States() { return size(); }
    int State(int ind) {
      if (ind < 0 || ind >= size()) { ERROR("oob set index: ", ind); return -1; }
      return (*this)[ind];
    }
    int Find(int state) {
      vector<int>::iterator i = lower_bound(begin(), end(), state);
      return (i == end() || (*i) != state) ? -1 : i - begin();
    }
    void Sort() {
      sort(begin(), end());
      erase(unique(begin(), end()), end());
    }
    void Clear() { vector<int>::clear(); }
    void Add(int state) { push_back(state); }
    void SubtractIDs(int n) { for (vector<int>::iterator i = begin(); i != end(); i++) (*i) -= n; }
    void Join(vector<int> &merged) {
      Statevec copy = *this;
      clear();
      for (Statevec::iterator i = copy.begin(); i != copy.end(); i++)
        if ((*i) >= 0 && (*i) < merged.size() && merged[(*i)] >= 0)
          push_back(merged[(*i)]);
      Sort();
    }
  };

  struct StateSetMatrix : public StateSet {
    unique_ptr<Matrix> set;
    StateSetMatrix(unique_ptr<Matrix> S) : set(move(S)) {}
    virtual ~StateSetMatrix() {}

    int States() { return set ? set->M : 0; }
    int State(int ind) {
      if (ind < 0 || ind >= States()) { ERROR("oob set index: ", ind); return -1; }
      return set->row(ind)[0];
    }
    int Find(int state) {
      double x = state, *row = static_cast<double*>(bsearch(&x, set->m, set->M, sizeof(double), DoubleSortR));
      return row ? row - set->m : -1;
    }
    void Clear()                   { FATAL(this, " function not implemented"); }
    void Add(int)                  { FATAL(this, " function not implemented"); }
    void Join(vector<int> &merged) { FATAL(this, " function not implemented"); }

    static int Write(StateSet *S, const char *dir, const char *name1, const char *name2, int iteration) {
      LocalFile out(string(dir) + MatrixFile::Filename(name1, name2, "matrix", iteration), "w");
      MatrixFile::WriteHeader(&out, BaseName(out.Filename()), "", S->States(), 1);
      for (int i=0, l=S->States(); i<l; i++) {
        double row[] = { double(S->State(i)) };
        MatrixFile::WriteRow(&out, row, 1);
      }
      return 0;
    }
  };

  struct Edge {
    static const int Cols = 5;
    int prevState, nextState, in, out; double weight;

    Edge() : prevState(-1), nextState(-1), in(-1), out(-1), weight(INFINITY) {}
    Edge(int L, int R, int I, int O, double W) : prevState(L), nextState(R), in(I), out(O), weight(W) {}
    Edge &operator=(const Edge &e) { prevState=e.prevState; nextState=e.nextState; in=e.in; out=e.out; weight=e.weight; return *this; }

    Edge(const double *row) { Assign(row); }
    void Assign(const double *row) { prevState = row[0]; nextState = row[1]; in = row[2]; out = row[3]; weight = row[4]; }

    string Name() { return StrCat(prevState, " ", nextState, " ", in, " ", out, " ", weight); }
    string Name(IOAlphabet *A, IOAlphabet *B) { return StrCat(prevState, " ", nextState, " ", A->Name(in), " ", B->Name(out), " ", weight); }

    string ToStr(const char *fmt) const {
      string v, vi(sizeof(int), 0); int viv;
      for (const char *fi=fmt; *fi; fi++) { 
        if      (*fi == 'i') viv = in;
        else if (*fi == 'o') viv = out;
        else if (*fi == 'n') viv = nextState;
        else if (*fi == 'w') viv = weight * 100;
        else FATAL("unknown fmt: ", *fi);
        memcpy(&vi[0], &viv, sizeof(int));
        v += vi;
      }
      return v;
    }

    bool operator<(const Edge &r) const { return LessThan(*this, r); }
    static bool LessThan(const Edge &l, const Edge &r) { return Cmp(l, r, false, true, true); }
    static bool Reverse(const Edge &l, const Edge &r) { return Cmp(l, r, true, true, true); }

    static bool Cmp(const Edge &l, const Edge &r, bool flipPrevNext, bool considerOut, bool considerWeight) {
      if (!flipPrevNext) {
        if      (l.prevState < r.prevState) return true;
        else if (l.prevState > r.prevState) return false;
      }
      else {
        if      (l.nextState < r.nextState) return true;
        else if (l.nextState > r.nextState) return false;
      }

      if      (l.in < r.in) return true;
      else if (l.in > r.in) return false;

      if (considerOut) {
        if      (l.out < r.out) return true;
        else if (l.out > r.out) return false;
      }
      if (considerWeight) {
        if      (l.weight < r.weight) return true;
        else if (l.weight > r.weight) return false;
      }

      if (flipPrevNext) {
        if      (l.nextState < r.nextState) return true;
        else if (l.nextState > r.nextState) return false;
      }
      else {
        if      (l.prevState < r.prevState) return true;
        else if (l.prevState > r.prevState) return false;
      }
      return false;
    }

    struct Filter {
      virtual void Begin(int state) {}
      virtual bool operator()(Edge &e) = 0;
    };

    struct LoopFilter    : public Filter { bool operator()(Edge &e) { return e.prevState == e.nextState; } };
    struct InNullFilter  : public Filter { bool operator()(Edge &e) { return !e.in; } };
    struct OutNullFilter : public Filter { bool operator()(Edge &e) { return !e.out; } };
    struct NullFilter    : public Filter { bool operator()(Edge &e) { return !e.in && !e.out && !e.weight; } };

    struct InvertFilter  : public Filter {
      Filter *F;
      InvertFilter(Filter *f) : F(f) {}
      bool operator()(Edge &e) { return !(*F)(e); }
    }; 

    struct SetFilter     : public Filter {
      StateSet *S; bool match;
      SetFilter(StateSet *s) : S(s) {}
      void Begin(int state) { match = S->Find(state) >= 0; }
      bool operator()(Edge &e) { return match; }
    };

    struct F2NotIFilter  : public Filter {
      StateSet *I, *F; bool final;
      F2NotIFilter(StateSet *i, StateSet *f) : I(i), F(f) {}
      void Begin(int state) { final = F->Find(state) >= 0; }
      bool operator()(Edge &e) { return final && I->Find(e.nextState) < 0; }
    };

    struct ShiftInputFilter : public Filter {
      StateSet *S; bool match; int from, to;
      ShiftInputFilter(StateSet *s, int F, int T) : S(s), from(F), to(T) {}
      void Begin(int state) { match = S->Find(state) >= 0; }
      bool operator()(Edge &e) {
        if (!match) return false;
        if (e.in != from) {
          ERROR(e.prevState, " -> ", e.nextState, " expected ", from, " input got ", e.in, ", skipping");
          return false;
        }
        e.in = to;
        return false;
      }
    };
  };

  struct TransitMap {
    virtual ~TransitMap() {}
    struct Iterator : public Edge {
      bool done; int impl1, impl2, impl3;
      Iterator() : done(0), impl1(0), impl2(0), impl3(0) {}
    };

    virtual void Begin(Iterator *iter, int state, int input=-1) = 0;
    virtual void Next(Iterator *iter) = 0;

    virtual int States() = 0;
    virtual int Transitions() = 0;
    virtual int Transitions(int state, int input = -1) {
      TransitMap::Iterator iter; int count=0;
      for (Begin(&iter, state, input); !iter.done; Next(&iter)) count++;
      return count;
    }

    virtual void Compact(Semiring *K, StateSet *I=0, StateSet *F=0) = 0;
    virtual void Join(vector<int> &merged, Semiring *K, StateSet *I=0, StateSet *F=0) = 0;
  };

  struct TransitMapMatrix : public TransitMap {
    unique_ptr<Matrix> state, transit;
    static const int HashBuckets = 5, HashValues = 2, StateValues = 4;
    static const int rowsize = sizeof(double) * Edge::Cols;

    TransitMapMatrix(unique_ptr<Matrix> State, unique_ptr<Matrix> Transit) :
      state(move(State)), transit(move(Transit)) {}

    static int SourceInputRowSort(const void *R1, const void *R2) { 
      const double *r1 = static_cast<const double*>(R1), *r2 = static_cast<const double*>(R2);
      if      (r1[0] < r2[0]) return -1; else if (r2[0] < r1[0]) return 1;
      else if (r1[2] < r2[2]) return -1; else if (r2[2] < r1[2]) return 1;
      return 0;
    }

    void Begin(Iterator *iter, int stateIndex, int input = -1) {
      if (stateIndex < 0 || stateIndex >= state->M) { iter->done = 1; return; }
      double *staterow = state->row(stateIndex);
      int transitIndex = staterow[1], transitCount = staterow[2];

      if (transitIndex < 0 || transitIndex >= transit->M) { iter->done = 1; return; }
      double *row = transit->row(transitIndex);

      if (input != -1) {
        double seek[] = { double(stateIndex), -1, double(input), -1, -1 }, *found; 
        found = static_cast<double*>(bsearch(&seek, row, transitCount, sizeof(double)*Edge::Cols, SourceInputRowSort));
        if (!found) { iter->done=1; return; }
        while (found >= row && found[0] == stateIndex && found[2] == input) found -= Edge::Cols;
        transitIndex += (found - row) / Edge::Cols + 1;
        row = transit->row(transitIndex);
      }

      iter->done = 0;
      iter->impl1 = transitIndex+1;
      iter->impl2 = transitCount;
      iter->impl3 = input;
      iter->Assign(row);
    }

    void Next(Iterator *iter) {
      if (iter->impl1 < 0 || iter->impl1 >= transit->M) { iter->done = 1; return; }
      int transitIndex = iter->impl1++;

      double *row = transit->row(transitIndex);
      if (iter->prevState != row[0]) { iter->done = 1; return; }
      iter->Assign(row);

      if (iter->impl3 != -1 && iter->impl3 != iter->in) { iter->done = 1; return; }
    }

    int States() { return state ? state->M : 0; }
    int Transitions() { return transit ? transit->M : 0; }
    int Transitions(int state, int input = -1) {
      if (input != -1) return TransitMap::Transitions(state, input);
      TransitMap::Iterator iter;
      Begin(&iter, state, input);
      if (iter.done) return 0;
      return iter.impl2;
    }

    void Compact(                     Semiring *K, StateSet *I=0, StateSet *F=0) { FATAL(this, " function not implemented"); }
    void Join   (vector<int> &merged, Semiring *K, StateSet *I=0, StateSet *F=0) { FATAL(this, " function not implemented"); }

    static int Write(TransitMap *M, const char *dir, const char *name, int iteration) {
      LocalFile state  (string(dir) + MatrixFile::Filename(name, "state",      "matrix", iteration), "w");
      LocalFile transit(string(dir) + MatrixFile::Filename(name, "transition", "matrix", iteration), "w");

      MatrixFile::WriteHeader(&state,   BaseName(state.Filename()),   "", M->States(),      StateValues);
      MatrixFile::WriteHeader(&transit, BaseName(transit.Filename()), "", M->Transitions(), Edge::Cols);

      int transits = 0;
      Matrix map(NextPrime(M->Transitions()*4), HashBuckets * HashValues);
      for (int i=0, l=M->States(); i<l; i++) {
        double row[] = { double(i), double(transits), double(M->Transitions(i)), 0 };

        Iterator iter; int count=0;
        for (M->Begin(&iter, i); !iter.done; M->Next(&iter)) {
          double row[] = { double(iter.prevState), double(iter.nextState), double(iter.in), double(iter.out), double(iter.weight) };
          MatrixFile::WriteRow(&transit, row, Edge::Cols);
          transits++;
          count++;
        }

        if (!count) row[1] = -1;
        MatrixFile::WriteRow(&state, row, StateValues);
      }
      return 0;
    }
  };

  struct TransitMapBuilder : public TransitMap {
    typedef vector<Edge> Edgevec;
    Edgevec transits;
    int maxStateInd;
    bool reverse;

    TransitMapBuilder(bool rev=false) : maxStateInd(-1), reverse(rev) {}
    TransitMapBuilder(const TransitMapBuilder &r) : transits(r.transits), maxStateInd(r.maxStateInd), reverse(r.reverse) {}
    TransitMapBuilder(TransitMap *E, bool (*filter)(const Edge &)=0) : maxStateInd(-1), reverse(false) {
      for (int i=0, l=E->States(); i<l; i++) {
        TransitMap::Iterator iter;
        for (E->Begin(&iter, i); !iter.done; E->Next(&iter))  {
          if (filter && filter(iter)) continue;
          Add(iter.prevState, iter.nextState, iter.in, iter.out, iter.weight);
        }
      }
    }
    void Begin(Iterator *iter, int state, int input = -1) {
      Edge lower(reverse ? -1 : state, reverse ? state : -1, input, -1, -1);
      Edgevec::iterator it = lower_bound(transits.begin(), transits.end(), lower, reverse ? Edge::Reverse : Edge::LessThan);
      if (Done(it, state, input)) { iter->done = 1; return; }
      *static_cast<Edge*>(iter) = (*it++);
      iter->impl1 = it - transits.begin();
      iter->impl2 = state;
      iter->impl3 = input;
      iter->done = 0;
    }
    void Next(Iterator *iter) {
      Edgevec::iterator it = transits.begin() + iter->impl1;
      if (Done(it, iter->impl2, iter->impl3)) { iter->done = 1; return; }
      *static_cast<Edge*>(iter) = (*it++);
      iter->impl1 = it - transits.begin();
    }
    bool Done(Edgevec::iterator it, int state, int input) {
      return it == transits.end() || (!reverse && state != (*it).prevState) ||
        (reverse && state != (*it).nextState) || (input != -1 && input != (*it).in);
    }

    int States() { return maxStateInd+1; }
    int Transitions() { return transits.size(); }

    void Add(int L, int R, int I, int O, double W) { Edge e(L, R, I, O, W); return Add(e); }
    void Add(const Edge &e) {
      if (e.nextState > maxStateInd) maxStateInd = e.nextState;
      if (e.prevState > maxStateInd) maxStateInd = e.prevState;
      transits.push_back(e);
    }
    void Add(const Edgevec &ev) { for (Edgevec::const_iterator i = ev.begin(); i != ev.end(); i++) Add(*i); }
    void Sort() { sort(transits.begin(), transits.end(), reverse ? Edge::Reverse : Edge::LessThan); }
    void SubtractIDs(int n) { for (Edgevec::iterator i = transits.begin(); i != transits.end(); i++) { (*i).prevState -= n; (*i).nextState -= n; } maxStateInd -= n; }
    int MaxID() { int ret=-1; for (Edgevec::iterator i = transits.begin(); i != transits.end(); i++) { if ((*i).prevState > ret) ret = (*i).prevState; if ((*i).nextState > ret) ret = (*i).nextState; } return ret; }

    void Realloc(vector<int> &merged, int plusN=0) {
      merged.resize(MaxID()+plusN+1);
      for (int i=0; i<merged.size(); i++) merged[i] = -1;

      for (Edgevec::iterator i = transits.begin(); i != transits.end(); i++) {
        merged[(*i).prevState] = (*i).prevState + plusN;
        merged[(*i).nextState] = (*i).nextState + plusN;
      }
    }

    void Compact(Semiring *K, StateSet *I=0, StateSet *F=0) {
      vector<int> merged;
      Realloc(merged);
      Join(merged, K, I, F);
    }

    void Join(vector<int> &merged, Semiring *K, StateSet *I=0, StateSet *F=0) {
      vector<int> merged2;
      merged2.resize(merged.size());

      int ind=0, outstates=0;
      for (int i=0, l=merged.size(); i<l; i++) {
        merged2[i] = i == merged[i] ? ind++ : -1;
        outstates += merged2[i] >= 0;
      }

      for (int i=0, l=merged.size(); i<l; i++) {
        if (merged[i] < 0 || merged[i] >= merged.size()) continue;
        merged[i] = merged2[merged[i]];
        if (merged[i] < 0 || merged[i] >= merged.size()) FATAL("oob merge ind ", merged[i], " (", i, " ", merged2[i], ")");
      }

      Join(merged, merged2, K, I, F);
    }

    void Join(vector<int> &merged, vector<int> &filter, Semiring *K, StateSet *I=0, StateSet *F=0) {
      TransitMapBuilder copy = *this;
      transits.clear();
      maxStateInd = -1;
      copy.Sort();

      for (int i=0, l=merged.size(); i<l; i++) {
        if (filter[i] < 0) continue;
        TransitMap::Iterator edge;
        for (copy.Begin(&edge, i); !edge.done; copy.Next(&edge)) {
          if (merged[edge.nextState] < 0) continue;
          Add(merged[i], merged[edge.nextState], edge.in, edge.out, edge.weight);
        }
      }

      Sort();
      if (I) I->Join(merged); 
      if (F) F->Join(merged); 
    }

    void Invert() {
      for (Edgevec::iterator i = transits.begin(); i != transits.end(); i++) swap((*i).in, (*i).out);
      Sort();
    }

    void Union(Semiring *K, StateSet *I, StateSet *F=0) {
      vector<int> merged;
      Realloc(merged, 1);

      vector<int> filter = merged;
      filter[filter.size()-1] = -1;
      Join(merged, filter, 0, F);

      for (int i=0, l=I->States(); i<l; i++)
        Add(0, I->State(i)+1, 0, 0, K ? K->One() : 0);

      I->Clear();
      I->Add(0);
      Sort();
    }
  };

  struct Residual {
    double residualWeight;
    vector<int> out, residualOut;

    Residual(double r=INFINITY, vector<int> *ResidualOut=0) : residualWeight(r) { if (ResidualOut) residualOut = *ResidualOut; }
    Residual(const Residual &copy) : residualWeight(copy.residualWeight), out(copy.out), residualOut(copy.residualOut) {}
    void AddResidualOutput(int O) { if (O && (!out.size() || (out.size() && out[out.size()-1] != O))) out.push_back(O); }

    void SubtractResidualOutput(const vector<int> &prefix) { 
      if (prefix.size()) out.erase(out.begin(), out.begin() + min(prefix.size(), out.size()));
      residualOut = out;
      out.clear();
    }
    void PrependResidualOutput() {
      vector<int> copyOut = out;
      out.clear();
      for (vector<int>::iterator i = residualOut.begin(); i != residualOut.end(); i++) out.push_back(*i);
      for (vector<int>::iterator i =     copyOut.begin(); i !=     copyOut.end(); i++) out.push_back(*i);
    }
    string Print(string v="") {
      v += " out: "; for (vector<int>::iterator i =         out.begin(); i !=         out.end(); i++) StrAppend(&v, (*i), "-");
      v += " r/o: "; for (vector<int>::iterator i = residualOut.begin(); i != residualOut.end(); i++) StrAppend(&v, (*i), "-");
      return v;
    }
  };

  struct ResidualSubset : public map<int, Residual> {
    double residualWeightSum;

    ResidualSubset(double r=INFINITY) : residualWeightSum(r) {}
    ResidualSubset(const ResidualSubset &copy) : map<int, Residual>(copy), residualWeightSum(copy.residualWeightSum) {}
    ResidualSubset(int state, Residual residual) { Add(state, residual); }
    void Add(int state, Residual residual) { (*this)[state] = residual; }

    bool operator<(const ResidualSubset &r) const {
      if      (size() < r.size()) return true;
      else if (size() > r.size()) return false;

      for (map<int, Residual>::const_iterator i = begin(), j = r.begin(); i != end() && j != r.end(); i++, j++) {
        if      ((*i).first < (*j).first) return true;
        else if ((*i).first > (*j).first) return false;
        if      ((*i).second.residualOut < (*j).second.residualOut) return true;
        else if ((*i).second.residualOut > (*j).second.residualOut) return false;
      }
      return false;
    }

    vector<int> LongestCommonPrefix(int maxlen) {
      vector<int> ret;
      if (!size()) return ret;

      for (int i=0; i<maxlen; i++) {
        bool assigned = false; int lastElement = -1;

        for (map<int, Residual>::iterator it = begin(); it != end(); it++) {
          if (i >= (*it).second.out.size()) break;

          if (!assigned) { assigned = true; lastElement = (*it).second.out[i]; }
          else if (lastElement != (*it).second.out[i]) return ret;
        }
        if (!assigned) break;
        ret.push_back(lastElement);
      }
      return ret;
    }

    vector<int> ShiftLongestCommonPrefix(int maxlen) {
      for (ResidualSubset::iterator it = begin(); it != end(); it++) (*it).second.PrependResidualOutput();

      vector<int> out = LongestCommonPrefix(maxlen);
      for (ResidualSubset::iterator it = begin(); it != end(); it++) (*it).second.SubtractResidualOutput(out);
      return out;
    }

    int ShiftLongestCommonPrefixOne() {
      vector<int> out = ShiftLongestCommonPrefix(1);
      return out.size() ? out[0] : 0;
    }

    void DivideWeightsBySum(Semiring *K) {
      for (ResidualSubset::iterator it = begin(); it != end(); it++) K->Div(&(*it).second.residualWeight, residualWeightSum);
    }

    string Print(string v="") {
      v += " = ";
      for(map<int, Residual>::iterator i = begin(); i != end(); i++) StrAppend(&v, (*i).first, " (", (*i).second.Print(), ")");
      return v;
    }
  };

  struct State {
    struct Filter { virtual bool operator()(int s) = 0; };
    struct SetFilter : public Filter {
      StateSet *S;
      SetFilter(StateSet *s) : S(s) {}
      bool operator()(int state) { return S->Find(state) >= 0; }
    };

    struct ChainFilter : public Filter {
      TransitMap *E;
      ChainFilter(TransitMap *e) : E(e) {}
      bool operator()(int state) {
        TransitMap::Iterator it; int count = 0;
        for (E->Begin(&it, state); !it.done; E->Next(&it)) if (count++) return false;
        return true;
      }
    };

    struct InvertFilter : public Filter {
      Filter *F;
      InvertFilter(Filter *f) : F(f) {}
      bool operator()(int state) { return !(*F)(state); }
    };

    struct OrFilter : public Filter {
      Filter *A, *B;
      OrFilter(Filter *a, Filter *b) : A(a), B(b) {}
      bool operator()(int state) { return (*A)(state) || (*B)(state); }
    };

    struct LabelSet {
      virtual string Name(int id) { return StrCat("S", id); }
    };

    struct LabelMap : public LabelSet, public map<int, string> {
      virtual string Name(int id) { map<int, string>::iterator it = find(id); return it != end() ? (*it).second.c_str() : ""; }
    };
  };

  Semiring *K;
  IOAlphabet *A, *B;
  unique_ptr<StateSet> I, F;
  unique_ptr<TransitMap> E;

  virtual ~WFST() {}
  WFST(Semiring *k, IOAlphabet *a=0, IOAlphabet *b=0, unique_ptr<StateSet> i=unique_ptr<StateSet>(),
       unique_ptr<StateSet> f=unique_ptr<StateSet>(), unique_ptr<TransitMap> e=unique_ptr<TransitMap>()) : 
    K(k), A(a), B(b), I(move(i)), F(move(f)), E(move(e)) {}

  void Reset() { E.reset(); I.reset(); F.reset(); }
  void Replace(unique_ptr<TransitMap> e, unique_ptr<StateSet> i, unique_ptr<StateSet> f) { E=move(e); I=move(i); F=move(f); }

  void Print(const char *name) {
    INFO("WFST ", name, ": ");
    for (int i=0, l=E->States(); i<l; i++) {
      TransitMap::Iterator edge;
      string v = StrCat(i, ": ");
      for (E->Begin(&edge, i); !edge.done; E->Next(&edge)) {
        string ain=A->Name(edge.in), aout=B->Name(edge.out);
        StringAppendf(&v, "%d -%s:%s-> %d %f, ", edge.prevState, ain.c_str(), aout.c_str(), edge.nextState, edge.weight);
        if (v.size() >= 1024) { INFO(v); v = "        "; }
      }
      if (v.size()) INFO(v);
    }
  }

  string ToStrGraphViz(State::LabelSet *label=0) {
    State::LabelSet defaultLabel;
    if (!label) label = &defaultLabel;
    string v = GraphViz::DigraphHeader("WFST");

    for (int i=0, l=I->States(); i<l; i++) GraphViz::AppendNode(&v, label->Name(I->State(i)));

    v += GraphViz::NodeStyle("solid");
    v += GraphViz::NodeShape("doublecircle");

    for (int i=0, l=F->States(); i<l; i++) GraphViz::AppendNode(&v, label->Name(F->State(i)));

    v += GraphViz::NodeShape("circle");

    for (int i=0, l=E->States(); i<l; i++) {
      TransitMap::Iterator edge;
      for (E->Begin(&edge, i); !edge.done; E->Next(&edge)) 
        GraphViz::AppendEdge(&v, label->Name(edge.prevState), label->Name(edge.nextState),
                                 StrCat(A->Name(edge.in), " : ", B->Name(edge.out), " / ",
                                        StringPrintf("%.2f", edge.weight)));
    }

    return v + GraphViz::Footer();
  }

  void PrintGraphViz() { string v = ToStrGraphViz(); printf("%s", v.c_str()); fflush(stdout); }

  int WriteGraphViz(string tofile, State::LabelSet *label=0) {
    LocalFile f(tofile.c_str(), "w"); string v=ToStrGraphViz(label);
    return f.Write(v.data(), v.size()) == v.size();
  }

  int Read(const char *name, const char *dir, int lastiter=-1) {
    Reset();

    unique_ptr<Matrix> state, transit;
    lastiter = MatrixFile::ReadVersioned(dir, name, "state", &state, 0, lastiter);
    if (!state) { ERROR("no WFST: ", name); return -1; }
    if (MatrixFile::ReadVersioned(dir, name, "transition", &transit, 0, lastiter)<0) { ERROR(name, ".", lastiter, ".transition.matrix"); return -1; }
    E = make_unique<TransitMapMatrix>(move(state), move(transit));

    unique_ptr<Matrix> initial, final;
    if (MatrixFile::ReadVersioned(dir, name, "initial", &initial, 0, lastiter)<0) { ERROR(name, ".", lastiter, ".initial.matrix"); return -1; }
    if (MatrixFile::ReadVersioned(dir, name, "final",   &final,   0, lastiter)<0) { ERROR(name, ".", lastiter, ".final.matrix"  ); return -1; }
    I = make_unique<StateSetMatrix>(move(initial));
    F = make_unique<StateSetMatrix>(move(final));
    return lastiter;
  }

  static int Write(WFST *t, const char *name, const char *dir, int iteration, bool wA=true, bool wB=true) {
    if (wA && StringFileAlphabet::Write(t->A, dir, name, "in", iteration)) return -1;
    if (wB && StringFileAlphabet::Write(t->B, dir, name, "out", iteration)) return -1;
    if (StateSetMatrix::Write(t->I.get(), dir, name, "initial", iteration)) return -1;
    if (StateSetMatrix::Write(t->F.get(), dir, name, "final", iteration)) return -1;
    if (TransitMapMatrix::Write(t->E.get(), dir, name, iteration)) return -1;
    return 0;
  }

  static void EdgeFilter(WFST *t, Edge::Filter *filter, TransitMapBuilder::Edgevec *filtered=0) {
    auto E = make_unique<TransitMapBuilder>();
    auto I = make_unique<Statevec>(t->I.get());
    auto F = make_unique<Statevec>(t->F.get());
    for (int i=0, l=t->E->States(); i<l; i++) {
      filter->Begin(i);
      TransitMap::Iterator edge;
      for (t->E->Begin(&edge, i); !edge.done; t->E->Next(&edge)) 
        if (!(*filter)(edge)) E->Add(edge);
        else if (filtered) filtered->push_back(edge);
    }
    E->Sort();
    t->Replace(move(E), move(I), move(F));
  }

  struct ShortestDistance {
    enum { Transduce=1, Reverse=2, SourceReset=4, Trace=8 };
    struct Path {
      double D, R; int depth; vector<int> out, traceback;
      Path(double zero=INFINITY) : D(zero), R(zero), depth(-1) {}
      Path(const Path &copy) : D(copy.D), R(copy.R), depth(copy.depth), out(copy.out), traceback(copy.traceback) {}
    };
    virtual Path *Get(int state) = 0;
    typedef map<int, Path> PathMap;

    set<int> reachable;
    void Successful(StateSet *F, vector<int> &out) {
      for (set<int>::iterator i = reachable.begin(); i != reachable.end(); i++)
        if (F->Find((*i)) >= 0) out.push_back((*i));
    }
  };

  struct ShortestDistanceVector : public ShortestDistance {
    vector<Path> path;
    ShortestDistanceVector(Semiring *K, int size) { path.resize(size, ShortestDistance::Path(K->Zero())); }
    Path *Get(int s) { return &path[s]; }
  };

  struct ShortestDistanceMapPtr : public ShortestDistance {
    Semiring *K;
    PathMap *path;
    ShortestDistanceMapPtr(Semiring *k, PathMap *p) : K(k), path(p) {}
    Path *Get(int s) { return &FindOrInsert(*path, s, ShortestDistance::Path(K->Zero()))->second; }
  };

  struct ShortestDistanceMap : public ShortestDistanceMapPtr {
    PathMap impl;
    ShortestDistanceMap(Semiring *k) : ShortestDistanceMapPtr(k, &impl) {}
    void Clear() { impl.clear(); }
  };

  static void ShortestDistance(ShortestDistance::PathMap &reachable, Semiring *K, TransitMap *E, int S,
                               State::Filter *sfilter=0, Edge::Filter *efilter=0, vector<int> *in=0, int flag=0, int maxiter=0)
  {
    Statevec SS;
    SS.push_back(S);
    ShortestDistanceMapPtr search(K, &reachable);
    ShortestDistance(&search, K, E, &SS, sfilter, efilter, in, flag, maxiter);
  }

  /* Semiring Frameworks and Algorithms for Shortest-Distance Problems http://www.cs.nyu.edu/~mohri/postscript/jalc.ps */
  static void ShortestDistance(struct ShortestDistance *search, Semiring *K, TransitMap *E, StateSet *S,
                               State::Filter *sfilter=0, Edge::Filter *efilter=0, vector<int> *in=0, int flag=0, int maxiter=0)
  {
    bool transduce = flag & ShortestDistance::Transduce, reverse = flag & ShortestDistance::Reverse, sourcereset = flag & ShortestDistance::SourceReset, trace = flag & ShortestDistance::Trace;
    deque<int> queue;
    for (int i=0, l=S->States(); i<l; i++) {
      int source = S->State(i);
      ShortestDistance::Path *path = search->Get(source);
      path->D = path->R = K->One();
      path->depth = 0;
      search->reachable.insert(source);
      queue.push_back(source);
    }
    int initialQueueSize = queue.size(), iterations = 0;

    while (queue.size()) {
      int q = queue.front();
      queue.pop_front();
      if (sfilter && (*sfilter)(q)) continue;
      ShortestDistance::Path *path = search->Get(q);

      double r = path->R;
      int depth = path->depth;
      vector<int> out = path->out, traceback = path->traceback;
      int nextin = (in && depth < in->size()) ? (*in)[depth] : 0;

      path->R = K->Zero();
      if (iterations++ < initialQueueSize && sourcereset) *path = ShortestDistance::Path(K->Zero());
      if (maxiter && iterations >= maxiter) break;

      TransitMap::Iterator edge;
      for (E->Begin(&edge, q); !edge.done; E->Next(&edge)) {
        if (in && edge.in && edge.in != nextin) continue;
        if (efilter && (*efilter)(edge)) continue;

        double rw = K->Mult(r, edge.weight);
        int state2 = reverse ? edge.prevState : edge.nextState;
        ShortestDistance::Path *path2 = search->Get(state2);

        if (path2->D != K->Add(path2->D, rw)) {
          K->Add(&path2->D, rw);
          K->Add(&path2->R, rw);

          path2->depth = depth + (edge.in != 0);
          if (transduce) { path2->out = out; if (edge.out) path2->out.push_back(edge.out); }
          if (trace) { path2->traceback = traceback; path2->traceback.push_back(edge.prevState); }

          if (search->reachable.find(state2) == search->reachable.end()) {
            search->reachable.insert(state2);
            queue.push_back(state2);
          }
        }
      }
    }
  }

  struct Builder {
    unique_ptr<TransitMapBuilder> E;
    unique_ptr<Statevec> I, F;
    Builder(bool open=true) { if (open) Open(); }
    void Open() { E = make_unique<TransitMapBuilder>(); I = make_unique<Statevec>(); F = make_unique<Statevec>(); }
    void Free() { E.reset(); I.reset(); F.reset(); }
    void Reopen() { Free(); Open(); }
  };

  struct Composer { /* A Generalized Composition Algorithm for Weighted Finite-State Transducers http://research.google.com/pubs/archive/36838.pdf */
    typedef Triple<int, int, int> trip;
    map<trip, int> states;
    set<trip> done;
    deque<trip> S;
    enum { T1MatchNull=1, T2MatchNull=2, FreeT1=4, FreeT2=8 };

    struct Reachable {
      virtual void PreCompute(WFST *t1, WFST *t2, bool r1=true, bool r2=true) = 0;
      virtual int operator()(int r1s, int r2s, Edge *singleMatchOut=0) = 0;
      virtual int operator()(int r2s, int r2i, int matches, Edge *singleMatchOut=0) = 0;
    };

    WFST *T1, *T2;
    State::LabelMap *label;
    Builder out;
    Reachable *reachable;

    int nextid, flag, (*transition_filter)(Reachable *, Edge&, Edge&, int q3);
    const static int defaultflag = T1MatchNull | T2MatchNull | FreeT1 | FreeT2;

    Composer() { Clear(); }
    void Clear() { T1=0; T2=0; label=0; out.Free(); reachable=0; nextid=0; flag=0; transition_filter=0; }

    struct SingleSourceNullClosure : public Reachable {
      WFST *T1, *T2;
      typedef pair<int, double> State; 
      typedef vector<State> States;
      vector<States> R1, R2;
      bool R1matchany, R2matchany;

      virtual States *R1Closure(int r1s) { return &R1[r1s]; }
      virtual States *R2Closure(int r2s) { return &R2[r2s]; }

      SingleSourceNullClosure(bool r1any=0, bool r2any=0) : T1(0), T2(0), R1matchany(0), R2matchany(0) {}
      void Clear() { R1.clear(); R2.clear(); }

      void PreCompute(WFST *t1, WFST *t2, bool r1=true, bool r2=true) {
        T1=t1; T2=t2;
        if (!R1matchany && r1) {
          R1.reserve(t1->E->States());
          for (int i=0, l=t1->E->States(); i<l; i++) {
            States out_reachable;
            Edge::OutNullFilter onf;
            NullClosure(t1->E.get(), i, &onf, &out_reachable);
            R1.push_back(out_reachable);
          }
        }
        if (!R2matchany && r2) {
          R2.reserve(t2->E->States());
          for (int i=0, l=t2->E->States(); i<l; i++) {
            States in_reachable;
            Edge::InNullFilter inf;
            NullClosure(t2->E.get(), i, &inf, &in_reachable);
            R2.push_back(in_reachable);
          }
        }
      }

      static void NullClosure(TransitMap *E, int source, Edge::Filter *filter, States *out) {
        Edge::InvertFilter ifilter(filter);
        ShortestDistance::PathMap dist;
        ShortestDistance(dist, Singleton<TropicalSemiring>::Set(), E, source, 0, &ifilter);
        for (ShortestDistance::PathMap::iterator it = dist.begin(); it != dist.end(); it++)
          out->push_back(State((*it).first, (*it).second.D));
      }

      int operator()(int r2s, int r2i, int matches, Edge *singleMatchOut=0) {
        States *R2ms = R2Closure(r2s); TransitMap::Iterator edge2;
        for (States::iterator j = R2ms->begin(); j != R2ms->end(); j++) {
          matches += T2->E->Transitions((*j).first, r2i);
          if (matches > 1) return matches;
          if (matches == 1) {
            if (!singleMatchOut) return matches;
            T2->E->Begin(&edge2, (*j).first, r2i);
            edge2.weight = T2->K->Mult(edge2.weight, (*j).second);
            *singleMatchOut = edge2;
          }
        }
        return matches; 
      }

      int operator()(int r1s, int r2s, Edge *singleMatchOut=0) {
        States *R1ms = R1Closure(r1s), *R2ms = R2Closure(r2s);
        if (R1matchany || R2matchany) return (R1matchany && R2ms->size()) || (R2matchany && R1ms->size());

        TransitMap::Iterator edge1; int matches = 0;
        for (States::iterator i = R1ms->begin(); i != R1ms->end(); i++)
          for (T1->E->Begin(&edge1, (*i).first); !edge1.done; T1->E->Next(&edge1)) {
            if (!edge1.out) continue;
            matches += (*this)(r2s, edge1.out, matches, singleMatchOut);
            if (matches > 1 || (matches && !singleMatchOut)) return matches;
          }

        return matches;
      };
    };

    struct BigramReachable : public SingleSourceNullClosure {
      SingleSourceNullClosure::States states;
      virtual void PreCompute(WFST *t1, WFST *t2, bool uu1=true, bool uu2=true) { T1=t1; T2=t2; }
      virtual States *R1Closure(int r1s) { return 0; }
      virtual States *R2Closure(int r2s) {
        TransitMap::Iterator edge2;
        T2->E->Begin(&edge2, r2s, 0);
        if (edge2.done || edge2.nextState) FATAL(r2s, " missing tx to 0 (", edge2.nextState, ", ", edge2.done, ")");
        states.clear();
        states.push_back(SingleSourceNullClosure::State(0, edge2.weight));
        return &states;
      }
      int operator()(int r2s, int r2i, int matches, Edge *singleMatchOut=0) {
        if (r2s || !r2i) return SingleSourceNullClosure::operator()(r2s, r2i, matches, singleMatchOut);
        TransitMapMatrix *E = static_cast<TransitMapMatrix*>(T2->E.get());
        TransitMap::Iterator edge2;
        E->Begin(&edge2, r2s);
        double *row = E->transit->row(edge2.impl1 + r2i - 2);
        if (singleMatchOut) singleMatchOut->Assign(row);
        return 1; 
      }
    };

    static int TrivialFilter(Reachable *R, Edge &e1, Edge &e2, int q3) {
      if (e1.out == e2.in) return 0;
      else                 return BLOCK;
    }

    static int EpsilonMatchingFilter(Reachable *R, Edge &e1, Edge &e2, int q3) {
      if      (e1.out > 0 && e1.out == e2.in)                return 0;
      else if (e1.out == NULL1 && e2.in == NULL1 && q3 == 0) return 0;
      else if (e1.out == NULL1 && e2.in == NULL2 && q3 != 1) return 2;
      else if (e1.out == NULL2 && e2.in == NULL1 && q3 != 2) return 1;
      return BLOCK;
    }

    static int LabelReachabilityFilter(Reachable *R, Edge &e1, Edge &e2, int q3) {
      if      (e1.out == e2.in && e2.in != NULL2) return 0;
      else if (e1.out == NULL1 && e2.in == NULL2) { if ((*R)(e1.prevState, e2.prevState)) return 0; }
      else if (e1.out == NULL2 && e2.in == NULL1) { if ((*R)(e1.prevState, e2.prevState)) return 0; }
      return BLOCK;
    }

    static int PushingLabelReachabilityFilter(Reachable *R, Edge &e1, Edge &e2, int q3) {
      int matches;
      if      (e1.out == e2.in && e2.in != NULL2) return 0;
      else if (e1.out == NULL1 && e2.in == NULL2) {
        if ((matches = (*R)(e1.prevState, e2.prevState)) > 1) return 0;
      }
      else if (e1.out == NULL2 && e2.in == NULL1) {
        if ((matches = (*R)(e1.prevState, e2.prevState)) > 1) return 0;
      }
      return BLOCK;
    }

    static bool ComposedStateMember(trip &q, StateSet *S1, StateSet *S2, Statevec &s3) {
      if (search(s3.begin(), s3.end(), &q.third, &q.third) == s3.end()) return false;
      if (S1->Find(q.first) < 0) return false;
      return S2->Find(q.second) >= 0;
    }

    int TripId(trip &t) {
      int lastid = nextid;
      int tid = FindOrInsert(states, t, nextid, &nextid)->second;
      if (lastid != nextid && label) (*label)[lastid] = StrCat(t.first, ",", t.second);
      return tid;
    }

    unique_ptr<WFST> Compose(WFST *t1, WFST *t2, string filter, Reachable *R=0, State::LabelMap *L=0, int Flag=0) {
      out.Reopen();
      T1=t1; T2=t2; reachable=R; label=L; flag=Flag?Flag:defaultflag;
      if (!T1->E->States() || !T2->E->States()) return Complete();

      Statevec i3, Q3;
      i3.push_back(0);
      Q3.push_back(0); Q3.push_back(BLOCK);
      bool precompute_reachable = false;

      if (filter == "trivial") transition_filter = TrivialFilter;
      else if (filter == "epsilon-matching") {
        Q3.push_back(1); Q3.push_back(2);
        transition_filter = EpsilonMatchingFilter;
      }
      else if (filter == "label-reachability") {
        transition_filter = LabelReachabilityFilter;
        precompute_reachable = true;
      }
      else if (filter == "pushing-label-reachability") {
        transition_filter = PushingLabelReachabilityFilter;
        precompute_reachable = true;
      }
      else FATAL("unknown filter ", filter);

      sort(i3.begin(), i3.end());
      sort(Q3.begin(), Q3.end());

      if (precompute_reachable) {
        if (!R) FATAL("missing R ", R);
        R->PreCompute(T1, T2);
      }

      for (Statevec::iterator fit = i3.begin(); fit != i3.end(); fit++) { /* cartesian product I1 x I2 x i3 */
        for (int i=0, il=T1->I->States(); i<il; i++) {
          for (int j=0, jl=T2->I->States(); j<jl; j++) {
            trip q(T1->I->State(i), T2->I->State(j), (*fit));
            done.insert(q);
            S.push_back(q);
          }
        }
      }

      while (S.size()) {
        trip q = S.front();
        S.pop_front();
        int qid = TripId(q);

        if (ComposedStateMember(q, T1->I.get(), T2->I.get(), i3)) out.I->push_back(qid);
        if (ComposedStateMember(q, T1->F.get(), T2->F.get(), Q3)) out.F->push_back(qid);

        Compose(q, qid);
        if (flag & T1MatchNull) ComposeNullLeft(q, qid);
        if (flag & T2MatchNull) ComposeNullRight(q, qid);
      }
      return Complete();
    }

    void Compose(trip &q, int qid) {
      TransitMap::Iterator edge1;
      for (T1->E->Begin(&edge1, q.first); !edge1.done; T1->E->Next(&edge1))
        ComposeRight(q.second, edge1.out, edge1, qid, q.third);
    }

    void ComposeNullLeft(trip &q, int qid) {
      Edge e2nl(q.second, q.second, NULL2, NULL1, T2->K->One());
      ComposeLeft(q.first, e2nl, qid, q.third);
    }

    void ComposeNullRight(trip &q, int qid, Edge *singleMatchOut=0) {
      Edge e1nl(q.first, q.first, NULL1, NULL2, T2->K->One());
      ComposeRight(q.second, NULL1, e1nl, qid, q.third, singleMatchOut);
    }

    int ComposeLeft(int state, Edge &e2, int qid, int q3) {
      TransitMap::Iterator e1; int matches=0;
      for (T1->E->Begin(&e1, state); !e1.done; T1->E->Next(&e1)) {
        Edge e2copy = e2;
        if ((q3 = transition_filter(reachable, e1, e2copy, q3)) == BLOCK) continue;
        if (out.E) Add(e1, e2copy, qid, q3);
        matches++;
      }
      return matches;
    }

    int ComposeRight(int state, int input, Edge &e1, int qid, int q3, Edge *singleMatchOut=0) {
      TransitMap::Iterator e2; int matches=0;
      for (T2->E->Begin(&e2, state, input); !e2.done; T2->E->Next(&e2)) {
        Edge e1copy = e1;
        if ((q3 = transition_filter(reachable, e1copy, e2, q3)) == BLOCK) continue;
        if (out.E) Add(e1copy, e2, qid, q3);
        if (++matches == 1 && singleMatchOut) *singleMatchOut = e2;
      }
      return matches;
    }

    void Add(Edge &e1, Edge &e2, int qid, int q3) {
      trip p(e1.nextState, e2.nextState, q3);
      int pid = TripId(p);
      out.E->Add(qid, pid, e1.in, e2.out, T2->K->Mult(e1.weight, e2.weight));
      if (done.find(p) == done.end()) { done.insert(p); S.push_back(p); }
    }

    unique_ptr<WFST> Complete() {
      auto ret = make_unique<WFST>(T2->K, T1->A, T2->B, move(out.I), move(out.F), move(out.E));
      if (flag & FreeT1) delete T1;
      if (flag & FreeT2) delete T2;
      Clear();
      return ret;
    }
  };

  static void Invert(TransitMapBuilder *E, IOAlphabet**A, IOAlphabet**B) { E->Invert(); if (A && B) swap(*A, *B); }

  static void Union(Semiring *K, TransitMapBuilder *E, StateSet *I, StateSet *F=0) { E->Union(K, I, F); }

  static void Closure(Semiring *K, TransitMapBuilder *E, StateSet *I, StateSet *F) {
    if (I->States() != 1) FATAL("expected 1 intial state, got ", I->States());
    for (int i=0, l=F->States(); i<l; i++) E->Add(F->State(i), I->State(0), 0, 0, K->One());
    E->Sort();
  }

  static void NullRemovedClosure(Semiring *K, TransitMapBuilder *E, StateSet *I, StateSet *F) {
    if (I->States() != 1) FATAL("expected 1 intial state, got ", I->States());

    vector<int> merged;
    E->Realloc(merged);

    for (int i=0, l=F->States(); i<l; i++) {
      int state = F->State(i); 
      TransitMap::Iterator edge;
      E->Begin(&edge, state);
      if (!edge.done) FATAL("expected final state ", state, " to have 0 transitions");
      merged[state] = I->State(0);
    }

    E->Join(merged, K, I, F);
  }

  static unique_ptr<WFST> ConCat(WFST *t1, WFST *t2) { /* add_edge(x,y) : t1.F X t2.I */ return 0; }

  static unique_ptr<WFST> Compose(WFST *t1, WFST *t2, string filter="label-reachability", Composer::Reachable *R=0, State::LabelMap *label=0, int flag=0) {
    Composer::SingleSourceNullClosure reachable;
    return Composer().Compose(t1, t2, filter, R ? R : &reachable, label, flag);
  }

  /* Weighted Transducer Determinization from p-Subsequentiable Transducers http://www.cs.nyu.edu/~mohri/pub/psub2.ps */
  static void Determinize(WFST *t, Semiring *K=0) {
    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>();
    unique_ptr<Statevec> I = make_unique<Statevec>(), F = make_unique<Statevec>();
    int startid = t->E->States(), nextid = startid;
    if (!K) K = t->K;

    deque<ResidualSubset> S;
    map<ResidualSubset, int> idmap;
    set<ResidualSubset> done;

    /* for each initial state */
    for (int i=0, l=t->I->States(); i<l; i++) {
      ResidualSubset s = ResidualSubset(t->I->State(i), K->One());
      S.push_back(s);
      done.insert(s);
      I->push_back(nextid);
      idmap[s] = nextid++;
    }

    while (S.size()) {
      /* de-queue */
      ResidualSubset p = S.front();
      S.pop_front();
      map<ResidualSubset, int>::iterator id1 = idmap.find(p);
      if (id1 == idmap.end()) FATAL("missing id ", -1);

      /* input label to target set */
      map<int, ResidualSubset> X;

      /* for each state p in subset p' */
      for (ResidualSubset::iterator i = p.begin(); i != p.end(); i++) {
        int state = (*i).first; double pw = (*i).second.residualWeight; bool inserted;

        /* for each transition from state p */
        TransitMap::Iterator edge; bool edges = false;
        for (t->E->Begin(&edge, state); !edge.done; t->E->Next(&edge)) {

          /* find input label */
          map<int, ResidualSubset>::iterator j = FindOrInsert(X, edge.in, ResidualSubset(K->Zero()));
          edges = true;

          /* update input label residual sum */
          double pqw = K->Mult(pw, edge.weight);
          K->Add(&(*j).second.residualWeightSum, pqw);

          /* add to target set */
          Residual qr(K->Zero(), &(*i).second.residualOut);
          ResidualSubset::iterator q = FindOrInsert((*j).second, edge.nextState, qr, &inserted);

          /* update target out */
          if (inserted && edge.out) (*q).second.out.push_back(edge.out);

          /* update target residual weight */
          K->Add(&(*q).second.residualWeight, pqw);

          /* target out conlict check */
          if ((!(*q).second.out.size() && edge.out) || ((*q).second.out.size() && (*q).second.out[(*q).second.out.size()-1] != edge.out)) 
            ERROR("conflicting out ", edge.out, " != ", (*q).second.out.size() ? (*q).second.out[(*q).second.out.size()-1] : -1,
                  " (", t->B->Name(edge.out), " != ", t->B->Name((*q).second.out[0]), ")");

          /* residual out conflict check */
          if ((*q).second.residualOut != (*i).second.residualOut) 
            ERROR("conflicting out ", (*q).second.residualOut.size() ? t->B->Name((*q).second.residualOut[0]).c_str() : "<none>",
                  " ", (*i).second.residualOut.size() ? t->B->Name((*i).second.residualOut[0]).c_str() : "<none>");
        }

        if (!edges && (*i).second.residualOut.size()) {

          /* find input label mapping for null/epsilon */
          map<int, ResidualSubset>::iterator j = FindOrInsert(X, 0, ResidualSubset(pw), &inserted);
          if (inserted) {

            /* all states are reachable by null transition to self */
            (*j).second[state] = Residual(pw, &(*i).second.residualOut);
          }

          /* update target residual weight */
          ResidualSubset::iterator q = (*j).second.begin();
          K->Add(&(*q).second.residualWeight, pw);

          /* target out conlict check */
          if ((*q).second.out.size())
            ERROR("conflicting out ", (*q).second.out[0], " (", t->B->Name((*q).second.out[0]), ")");

          /* residual out conflict check */
          if ((*q).second.residualOut != (*i).second.residualOut) 
            ERROR("conflicting out ", (*q).second.residualOut.size() ? t->B->Name((*q).second.residualOut[0]).c_str() : "<none>",
                  " ", (*i).second.residualOut.size() ? t->B->Name((*i).second.residualOut[0]).c_str() : "<none>");
        }
      }

      /* for each input label */
      for (map<int, ResidualSubset>::iterator j = X.begin(); j != X.end(); j++) {

        /* update residual outs and determine common label (if any) */
        int out = (*j).second.ShiftLongestCommonPrefixOne();
        (*j).second.DivideWeightsBySum(K);

        /* determine target */
        map<ResidualSubset, int>::iterator id2 = FindOrInsert(idmap, (*j).second, nextid, &nextid);

        /* add edge */
        Edge e((*id1).second, (*id2).second, (*j).first, out, (*j).second.residualWeightSum);
        E->Add(e);

        /* check done */
        if (done.find((*j).second) != done.end()) continue;
        done.insert((*j).second);

        /* add final */
        for (ResidualSubset::iterator q = (*j).second.begin(); q != (*j).second.end(); q++) 
          if (t->F->Find((*q).first) >= 0) { F->push_back((*id2).second); break; }

        /* en-queue */
        S.push_back((*j).second);
      }
    }

    E->SubtractIDs(startid); I->SubtractIDs(startid); F->SubtractIDs(startid);
    E->Sort(); I->Sort(); F->Sort();
    t->Replace(move(E), move(I), move(F));
  }

  /* Speech Recognition With Weighted Finite-State Transducers http://www.cs.nyu.edu/~mohri/pub/hbka.pdf */
  static void PushWeights(Semiring *K, WFST *t, StateSet *startFrom, bool sourcereset=false) {
    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>(t->E.get());
    unique_ptr<Statevec> I = make_unique<Statevec>(t->I.get()), F = make_unique<Statevec>(t->F.get());

    E->reverse = true;
    E->Sort();

    ShortestDistanceVector search(K, t->E->States());
    int flag = ShortestDistance::Reverse | (sourcereset ? ShortestDistance::SourceReset : 0);
    ShortestDistance(&search, K, E.get(), startFrom, 0, 0, 0, flag);

    for (TransitMapBuilder::Edgevec::iterator it = E->transits.begin(); it != E->transits.end(); it++) {
      double dp = search.path[(*it).prevState].D, dn = search.path[(*it).nextState].D;

      if (dp != K->Zero())
        (*it).weight = K->Div(K->Mult((*it).weight, dn), dp);
    }

    E->reverse = false;
    E->Sort();

    t->Replace(move(E), move(I), move(F));
  }

  int Minimize() { /* XXX Hopcroft Algorithm is O(|I|*n*log(n)) while this (due to the pathological case) is O(n^2*log(n)) */
    for (int iter=0; /**/; iter++) {
      vector<int> merged;
      vector<pair<string, int> > tx;
      tx.resize(E->States());
      merged.resize(E->States());

      for (int i=0, l=E->States(); i<l; i++) {
        string v = F->Find(i) >= 0 ? "1" : "0";
        TransitMap::Iterator iter;
        for (E->Begin(&iter, i); !iter.done; E->Next(&iter)) v += iter.ToStr("ionw");
        tx[i] = pair<string, int>(v, i);
        merged[i] = i;
      }

      sort(tx.begin(), tx.end());
      int mergecount = 0;
      for (int i=0, l=E->States(); i<l; i++) if (i+1<l && tx[i].first == tx[i+1].first) { merged[tx[i+1].second] = merged[tx[i].second]; mergecount++; }
      if (!mergecount) return iter;

      E->Join(merged, K, I.get(), F.get());
    } 
  }

  /* Generic null-Removal and Input null-Normalization Algorithms for Weighted Transducer http://www.cs.nyu.edu/~mohri/pub/ijfcs.pdf */
  static void RemoveNulls(WFST *t) { Edge::NullFilter nullf; RemoveNulls(t, &nullf); }
  static void RemoveNulls(WFST *t, Edge::Filter *filter) {
    if (!t->E->States()) return;
    Edge::InvertFilter ifilter(filter);

    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>();
    unique_ptr<Statevec> I = make_unique<Statevec>(t->I.get());
    set<int> final;
    for (int i=0, l=t->F->States(); i<l; i++) final.insert(t->F->State(i));

    for (int i=0, l=t->E->States(); i<l; i++) {
      bool finalState = final.find(i) != final.end();

      TransitMap::Iterator iter;
      for (t->E->Begin(&iter, i); !iter.done; t->E->Next(&iter))
        if (!(*filter)(iter))
          E->Add(iter.prevState, iter.nextState, iter.in, iter.out, iter.weight);

      ShortestDistance::PathMap dist;
      ShortestDistance(dist, Singleton<TropicalSemiring>::Set(), t->E.get(), i, 0, &ifilter);
      for (ShortestDistance::PathMap::iterator it = dist.begin(); it != dist.end(); it++) {
        if ((*it).first == i) continue;
        if (!finalState && final.find((*it).first) != final.end()) { final.insert(i); finalState = true; }

        TransitMap::Iterator iter2;
        for (t->E->Begin(&iter2, (*it).first); !iter2.done; t->E->Next(&iter2))
          if (!(*filter)(iter2))
            E->Add(i, iter2.nextState, iter2.in, iter2.out, t->K->Mult((*it).second.D, iter2.weight));
      }
    }

    auto F = make_unique<Statevec>(final);
    E->Compact(t->K, I.get(), F.get());
    t->Replace(move(E), move(I), move(F));
  }

  /* cross between determinize and removeNulls */
  static void NormalizeInput(WFST *t) {
    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>();
    unique_ptr<Statevec> I = make_unique<Statevec>(), F = make_unique<Statevec>();
    int startid = t->E->States(), nextid = startid;

    deque<ResidualSubset> S;
    set<ResidualSubset> done;
    map<ResidualSubset, int> idmap;
    map<ResidualSubset, int>::iterator id1, id2;

    Edge::InNullFilter inullf, *filter = &inullf;
    Edge::InvertFilter ifilter(filter);

    /* for each initial state */
    for (int i=0, l=t->I->States(); i<l; i++) {
      ResidualSubset s = ResidualSubset(t->I->State(i), t->K->One());
      S.push_back(s);
      done.insert(s);
      I->push_back(nextid);
      idmap[s] = nextid++;
    }

    while (S.size()) {
      /* de-queue */
      ResidualSubset p = S.front();
      S.pop_front();
      if (p.size() != 1 || (id1 = idmap.find(p)) == idmap.end()) FATAL("corruption ", -1);
      int source = (*p.begin()).first;
      Residual &pr = (*p.begin()).second;

      /* same as removeNulls */
      TransitMap::Iterator iter;
      for (t->E->Begin(&iter, source); !iter.done; t->E->Next(&iter)) 
        if (!(*filter)(iter)) {
          ResidualSubset q = ResidualSubset(iter.nextState, pr);
          (*q.begin()).second.AddResidualOutput(iter.out);
          int out = q.ShiftLongestCommonPrefixOne();

          id2 = FindOrInsert(idmap, q, nextid, &nextid);
          E->Add((*id1).second, (*id2).second, iter.in, out, iter.weight);
          if (done.find(q) == done.end()) { S.push_back(q); done.insert(q); }
        }

      /* same as removeNulls */
      ShortestDistance::PathMap dist;
      ShortestDistance(dist, Singleton<TropicalSemiring>::Set(), t->E.get(), source, 0, &ifilter, 0, ShortestDistance::Transduce);
      for (ShortestDistance::PathMap::iterator it = dist.begin(); it != dist.end(); it++) {
        if ((*it).first == source) continue;

        TransitMap::Iterator iter2;
        for (t->E->Begin(&iter2, (*it).first); !iter2.done; t->E->Next(&iter2))
          if (!(*filter)(iter2)) {
            ResidualSubset q = ResidualSubset(iter2.nextState, pr);
            (*q.begin()).second.out = (*it).second.out;
            (*q.begin()).second.AddResidualOutput(iter2.out);
            int out = q.ShiftLongestCommonPrefixOne();

            id2 = FindOrInsert(idmap, q, nextid, &nextid);
            E->Add((*id1).second, (*id2).second, iter2.in, out, t->K->Mult(iter2.weight, (*it).second.D));
            if (done.find(q) == done.end()) { S.push_back(q); done.insert(q); }
          }
      }

      if (t->F->Find(source) < 0) continue;
      else if (!(*p.begin()).second.residualOut.size()) { F->push_back((*id1).second); continue; }
      else {
        int out = p.ShiftLongestCommonPrefixOne();
        id2 = FindOrInsert(idmap, p, nextid, &nextid);
        E->Add((*id1).second, (*id2).second, 0, out, t->K->One());
        S.push_back(p); done.insert(p);
      }
    }

    E->SubtractIDs(startid); I->SubtractIDs(startid); F->SubtractIDs(startid);
    E->Sort(); I->Sort(); F->Sort();
    t->Replace(move(E), move(I), move(F));
  }

  /* remove any state with all outgoing null inputs and all incoming null outputs - introduces non-determinism */
  static void NormalizeTail(WFST *t) {
    if (!t->E->States()) return;

    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>(t->E.get());
    unique_ptr<Statevec> I = make_unique<Statevec>(t->I.get()), F = make_unique<Statevec>(t->F.get());

    vector<int> merged;
    E->Realloc(merged);

    E->reverse = true;
    E->Sort();

    for (int i=0, l=t->E->States(); i<l; i++) {
      /* find state with all outgoing null inputs */
      TransitMap::Iterator iter1, iter2; int outcount = 0;
      for (t->E->Begin(&iter1, i); !iter1.done; t->E->Next(&iter1)) if (iter1.in) break; else outcount++;
      if (!iter1.done || !outcount) continue;

      /* check state has all incoming null outputs */
      for (E->Begin(&iter2, i); !iter2.done; E->Next(&iter2)) if (iter2.out) break;
      if (!iter2.done) { ERROR("state ", i, " has null transitions but cannot be split"); continue; }

      /* ok we can remove it */
      merged[i] = -1;
      vector<Edge> addedge;

      /* pair null outputs with null inputs */
      for (t->E->Begin(&iter1, i); !iter1.done; t->E->Next(&iter1)) 
        for (E->Begin(&iter2, i); !iter2.done; E->Next(&iter2))
          addedge.push_back(Edge(iter2.prevState, iter1.nextState, iter2.in, iter1.out, iter1.weight));

      for (vector<Edge>::iterator it = addedge.begin(); it != addedge.end(); it++) E->Add(*it);
    }

    E->reverse = false;
    E->Join(merged, t->K, I.get(), F.get());
    t->Replace(move(E), move(I), move(F));
  }

  static void RemoveNonCoAccessible(WFST *t) { return RemoveNonAccessible(t, true); }
  static void RemoveNonAccessible(WFST *t, bool reverse=false) {
    if (!t->E->States()) return;

    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>(t->E.get());
    unique_ptr<Statevec> I = make_unique<Statevec>(t->I.get()), F = make_unique<Statevec>(t->F.get());
    StateSet *startFrom = t->I.get();
    if (reverse) { startFrom = t->F.get(); E->reverse = true; E->Sort(); }

    ShortestDistanceVector search(Singleton<TropicalSemiring>::Set(), t->E->States());
    int flag = reverse ? ShortestDistance::Reverse : 0;
    ShortestDistance(&search, Singleton<TropicalSemiring>::Set(), E.get(), startFrom, 0, 0, 0, flag);

    vector<int> accessible;
    accessible.resize(t->E->States(), -1);
    for (set<int>::iterator i = search.reachable.begin(); i != search.reachable.end(); i++)
      accessible[(*i)] = (*i);

    if (reverse) { E->reverse = false; E->Sort(); }
    E->Join(accessible, t->K, I.get(), F.get());
    t->Replace(move(E), move(I), move(F));
  }

  static void RemoveSelfLoops(WFST *t) { Edge::LoopFilter loopf; EdgeFilter(t, &loopf); }
  static void RemoveNonInitialFinalTransitions(WFST *t) { Edge::F2NotIFilter f2nif(t->I.get(), t->F.get()); EdgeFilter(t, &f2nif); }
  static void RemoveFinalTransitions(WFST *t, TransitMapBuilder::Edgevec *out=0) { Edge::SetFilter sf(t->F.get()); EdgeFilter(t, &sf, out); }
  static void ShiftFinalTransitions(WFST *t, int from, int to) { Edge::ShiftInputFilter sf(t->F.get(), from, to); EdgeFilter(t, &sf); }

  /* G = grammar */
  static unique_ptr<WFST> Grammar(Semiring *K, IOAlphabet *A, LanguageModel *LM, StringIter *vocab) {
    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>();
    unique_ptr<Statevec> I = make_unique<Statevec>(), F = make_unique<Statevec>();
    map<unsigned, int> idmap;
    int state = 0;
    I->push_back(state++);

    for (const char *word=vocab->Next(); word; word=vocab->Next()) {
      unsigned wordhash = fnv32(word);
      int LM_priorInd, LM_transitInd, LM_count=0, LM_transits=0;
      int out = A->Id(word);
      if (out < 0) continue;

      if (!LM->Get(wordhash, &LM_priorInd, &LM_transitInd)) { DEBUG("LM missing '%s'", word); LM_count=1; }
      if (!LM_count) {
        LM_transits = LM->Transits(LM_transitInd, wordhash);
        LM_count = LM->Occurrences(LM_priorInd);
      }

      int prevState = FindOrInsert(idmap, wordhash, state, &state)->second;
      E->Add(0, prevState, out, out, -log(double(LM_count) / LM->total));
      E->Add(prevState, 0, 0, 0, -log(1.0 / (LM_transits+1)));

      for (int i=0; i<LM_transits; i++) {
        double *tr = LM->transit->row(LM_transitInd + i);
        unsigned wordhash2 = tr[TC_Edge];

        int LM_priorInd2, out2;
        if (!LM->Get(wordhash2, &LM_priorInd2)) continue;
        string word2 = LM->Name(LM_priorInd2);
        if ((out2 = A->Id(word2.c_str())) < 0) continue;

        int nextState = FindOrInsert(idmap, wordhash2, state, &state)->second;
        E->Add(prevState, nextState, out2, out2, -tr[TC_Cost]);
      }
    }
    E->Sort();
    return make_unique<WFST>(K, A, A, move(I), move(F), move(E));
  }

  /* L = pronunciation lexicon */
  static unique_ptr<WFST> PronunciationLexicon(Semiring *K, IOAlphabet *A, AlphabetBuilder *B, AlphabetBuilder *aux, PronunciationDict *dict, StringIter *vocab, LanguageModel *LM=0) {
    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>();
    unique_ptr<Statevec> I = make_unique<Statevec>(), F = make_unique<Statevec>();
    map<string, int> homophone;
    int state = 0, max_homophones = 0, id;

    for (const char *word=vocab->Next(); word; word=vocab->Next()) {
      const char *pronunciation = dict->Pronounce(word);
      if (!pronunciation || !*pronunciation) { ERROR("no pronuciation for existing word '", word, "'"); continue; }

      /* disambiguate homophones */ 
      map<string, int>::iterator hpi = FindOrInsert(homophone, string(pronunciation), int(0));
      (*hpi).second++;
      for (int i=max_homophones; i<(*hpi).second; i++) if ((id = aux->Add(StrCat("#", i+1))) != i+1) FATAL("mismatch ", id, " != ", i+1);

      I->push_back(state);
      for (int j=0, l=strlen(pronunciation); j<l; j++) {
        if (pronunciation[j] < 0 || pronunciation[j] >= LFL_PHONES) FATAL("oob phone ", word, " ", j, "/", l, " = ", pronunciation[j]);

        bool out_on_first = true, out_on_this = (!out_on_first && j == l-1) || (out_on_first && !j);

        double weight = K->One(); int priorInd;
        if (out_on_this && LM && LM->Get(word, &priorInd)) /* unigram LM */
          weight = -log(double(LM->Occurrences(priorInd)) / LM->total);

        int nextstate = state+1;
        E->Add(state++, nextstate, pronunciation[j], out_on_this ? B->Add(word) : 0, weight);
      }

      int nextState = state+1;
      E->Add(state++, nextState, IOAlphabet::AuxiliarySymbolId((*hpi).second), 0, K->One());
      F->push_back(state++);
    }

    auto L = make_unique<WFST>(K, A, B, move(I), move(F), move(E));
    INFOf("pronunciation WFST contains %d states (%d initial, %d final) and %d transits", L->E->States(), L->I->States(), L->F->States(), L->E->Transitions());
    return L;
  }

  static unique_ptr<WFST> TestPronunciationLexicon(Semiring *K, IOAlphabet *A, IOAlphabet *B) {
    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>();
    unique_ptr<Statevec> I = make_unique<Statevec>(), F = make_unique<Statevec>();
    int aux1 = IOAlphabet::AuxiliarySymbolId(1);
    int word1=B->Id("cat"), word2=B->Id("dog");
    I->push_back(0); I->push_back(1);
    E->Add(0, 2, 1, word1, K->One());
    E->Add(1, 3, 2, word2, K->One());
    E->Add(2, 4, 2,     0, K->One());
    E->Add(3, 5, 1,     0, K->One());
    E->Add(4, 6, aux1,  0, K->One());
    E->Add(5, 7, aux1,  0, K->One());
    F->push_back(6); F->push_back(7);
    return make_unique<WFST>(K, A, B, move(I), move(F), move(E));
  }

  /* C = context dependency transducer */
  static unique_ptr<WFST> ContextDependencyTransducer(Semiring *K, IOAlphabet *A, AlphabetBuilder *B, IOAlphabet *aux=0) {
    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>();
    unique_ptr<Statevec> I = make_unique<Statevec>(), F = make_unique<Statevec>();

    for (int i=1; i<LFL_PHONES; i++) {
      for (int j=0; j<LFL_PHONES; j++) {
        int state = (i-1)*LFL_PHONES+j+1;
        E->Add(0, state, i, B->Add(AcousticModel::Name(0,i,j,0)), K->One());
        if (!j) continue;

        for (int k=0; k<LFL_PHONES; k++) {
          int nextstate = (j-1)*LFL_PHONES+k+1;
          E->Add(state, nextstate, j, B->Add(AcousticModel::Name(i,j,k,0)), K->One());
          if (!k) F->push_back(nextstate);
        }
      }
    }

    E->Sort();
    F->Sort();
    I->push_back(0);
    return make_unique<WFST>(K, A, B, move(I), move(F), move(E));
  }

  static void ContextDependencyRebuildFinal(WFST *t) {
    auto F = make_unique<Statevec>();
    for (int i=0; i<t->E->States(); i++) {
      TransitMap::Iterator iter;
      for (t->E->Begin(&iter, i); !iter.done; t->E->Next(&iter)) {
        int pp, pn, p = AcousticModel::ParseName(t->A->Name(iter.in).c_str(), 0, &pp, &pn);
        if (!pn) F->push_back(iter.nextState);
      }
    }
    F->Sort();
    t->F = move(F);
  }

  /* auxiliary symbols */
  static void AddAuxiliarySymbols(TransitMapBuilder *E, IOAlphabet *aux, double weight) {
    for (int i=0; i<E->States(); i++) AddAuxiliarySymbols(E, aux, i, weight);
    E->Sort();
  }

  static void AddAuxiliarySymbols(TransitMapBuilder *E, IOAlphabet *aux, StateSet *S, double weight) {
    for (int i=0; i<S->States(); i++) AddAuxiliarySymbols(E, aux, S->State(i), weight);
    E->Sort();
  }

  static void AddAuxiliarySymbols(TransitMapBuilder *E, IOAlphabet *aux, int state, double weight) {
    IOAlphabet::Iterator iter;
    for (aux->Begin(&iter); !iter.done; aux->Next(&iter)) if (iter.id)
      E->Add(state, state, IOAlphabet::AuxiliarySymbolId(iter.id), IOAlphabet::AuxiliarySymbolId(iter.id), weight);
  }

  static void AddAuxiliarySymbols(vector<int> *R, IOAlphabet *aux) {
    IOAlphabet::Iterator iter;
    for (aux->Begin(&iter); !iter.done; aux->Next(&iter)) if (iter.id) R->push_back(IOAlphabet::AuxiliarySymbolId(iter.id));
    sort(R->begin(), R->end());
  }

  static void ReplaceAuxiliarySymbols(TransitMapBuilder *E) {
    for (TransitMapBuilder::Edgevec::iterator i = E->transits.begin(); i != E->transits.end(); i++) {
      if (IOAlphabet::AuxiliarySymbol((*i).in))  (*i).in  = 0;
      if (IOAlphabet::AuxiliarySymbol((*i).out)) (*i).out = 0;
    }
    E->Sort();
  }

  /* H = hidden markov model transducer */
  static unique_ptr<WFST> HMMTransducer(Semiring *K, AcousticModelAlphabet *A, IOAlphabet *B, vector<int> *reachable=0) {
    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>();
    unique_ptr<Statevec> I = make_unique<Statevec>(), F = make_unique<Statevec>();
    IOAlphabet::Iterator iter;
    int state = 0;
    if (reachable) reachable->resize(B->Size());

    for (B->Begin(&iter); !iter.done; B->Next(&iter)) {
      if (!iter.id) continue;
      if (reachable) reachable->push_back(iter.id);

      int pp, pn, ps, p = AcousticModel::ParseName(iter.name, &ps, &pp, &pn);
      if (ps != 0) FATAL("unexpected ", iter.name, " (", iter.id, ") ", ps, " != ", 0);

      I->push_back(state);
      for (int j=0; j<AcousticModel::StatesPerPhone; j++) {
        unsigned hash = AcousticModel::State::Tied(A->AM->TiedStates(), pp, p, pn, j);
        int in = A->AM->GetState(hash)->val.emission_index;
        int nextstate = state+1;
        E->Add(state++, nextstate, in, !j ? iter.id : 0, K->One());
      }
      F->push_back(state++);
    }

    if (reachable) sort(reachable->begin(), reachable->end());
    return make_unique<WFST>(K, A, B, move(I), move(F), move(E));
  }

  /* HW = hidden markov model weight transducer */
  static unique_ptr<WFST> HMMWeightTransducer(Semiring *K, AcousticModelAlphabet *A) {
    unique_ptr<TransitMapBuilder> E = make_unique<TransitMapBuilder>();
    unique_ptr<Statevec> I = make_unique<Statevec>(), F = make_unique<Statevec>();

    for (int i=0, l=A->AM->state.size(); i<l; i++) {
      HMM::ActiveStateIndex active(1,1,1);
      HMM::ActiveState::Iterator src(i);
      AcousticHMM::TransitMap tm(A->AM, true);
      AcousticHMM::TransitMap::Iterator iter;
      for (tm.Begin(&iter, &active, &src); !iter.done; tm.Next(&iter)) {
        unsigned hash = iter.state;
        int o = A->AM->GetState(hash)->val.emission_index;
        if (i == o) continue;
        E->Add(i, o, o, o, -iter.cost /* negative log prob */);
      }
    }

    I->push_back(0);
    E->Compact(K, I.get());
    for (int i=0, l=E->States(); i<l; i++) F->push_back(i);

    return make_unique<WFST>(K, A, A, move(I), move(F), move(E));
  }
};

}; // namespace LFL
#endif // LFL_SPEECH_WFST_H__
