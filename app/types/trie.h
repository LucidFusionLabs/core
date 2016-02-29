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

#ifndef LFL_LFAPP_TYPES_TRIE_H__
#define LFL_LFAPP_TYPES_TRIE_H__
namespace LFL {

template <class K> struct TrieNode {
  int val_ind=0, next_ind=0;
  unsigned short next_len=0, next_i=0;
  TrieNode(int V=0) : val_ind(V) {}
};

template <class K> struct TrieNodeNext {
  K key;
  int next; 
  TrieNodeNext(const K &k=K(), int n=0) : key(k), next(n) {}
  bool operator<(const TrieNodeNext &n) const { SortImpl1(key, n.key) } 
};

template <class K, class V> struct TrieNodeValue {
  V val;
  TrieNodeValue(const basic_string<K>&, int, const V &v) : val(v) {}
};

template <class K, class V, class TN = TrieNode<K>, class TV = TrieNodeValue<K, V> > struct Trie {
  typedef Trie<K, V, TN, TV> Self;
  typedef basic_string<K> String;
  typedef TrieNodeNext<K> Next;
  typedef TN Node;
  typedef TV Value;

  struct SortedInputConstructor {
    struct Prenode {
      K key;
      int node, next_len; 
      Prenode(const K &k, int n) : key(k), node(n), next_len(0) {}
    };
    typedef int  (SortedInputConstructor::*InsertCB)(const String &k, const V*);
    typedef void (SortedInputConstructor::*CompleteCB)(const Prenode&);

    Self *out;
    int insert_id=0;
    bool patricia;
    vector<Prenode> stack;
    InsertCB     insert_cb = &SortedInputConstructor::Phase1Insert;
    CompleteCB complete_cb = &SortedInputConstructor::Phase1Complete;

    template <class I> SortedInputConstructor(Self *O, I b, I e, bool P=true) : out(O), patricia(P) {
      stack.emplace_back(K(), (this->*insert_cb)(String(), NULL));
      Input(b, e);
      insert_cb   = &SortedInputConstructor::Phase2Insert;
      complete_cb = &SortedInputConstructor::Phase2Complete;
      stack.emplace_back(K(), (this->*insert_cb)(String(), NULL));
      Input(b, e);
    }

    template <class I> void Input(I b, I e) { 
      for (auto i = b; i != e; /**/) {
        const String &k = (*i).first;
        const V      &v = (*i).second;
        Input(k, ++i != e ? (*i).first : String(), v);
      }
      while (stack.size()) (this->*complete_cb)(PopBack(stack));
      out->head = 1;
    }

    void Input(const String &in, const String &lookahead, const V &v) {
      PopPrefixesNotMatching(in);
      AddPrefixesOf(in, lookahead);
      stack.emplace_back(in[stack.size()-1], (this->*insert_cb)(in, &v));
    }

    int Phase1Insert(const String &in, const V *v) { 
      if (v) out->val.push_back(Value(in, stack.size(), *v)); 
      out->data.push_back(Node(v ? out->val.size() : 0));
      return out->data.size();
    }

    void Phase1Complete(const Prenode &n) {
      if (stack.size()) stack.back().next_len++;
      out->data[n.node-1].next_len = n.next_len;
    }

    int Phase2Insert(const String &leaf, const V *v) { return ++insert_id; }
    void Phase2Complete(const Prenode &n) {
      out->ComputeStateFromChildren(&out->data[n.node-1]);
      if (!stack.size()) return;
      Node *p = &out->data[stack.back().node-1];
      CHECK_LT(p->next_i, p->next_len);
      if (!p->next_ind) {
        p->next_ind = out->next.size() + 1;
        out->next.resize(out->next.size() + p->next_len);
      }
      Next *next = &out->next[p->next_ind-1];
      next[p->next_i  ].key  = n.key;
      next[p->next_i++].next = n.node;
    }

    void PopPrefixesNotMatching(const String &in) {
      auto si = stack.begin() + 1, se = stack.end();
      for (auto i = in.begin(), e = in.end(); i != e && si != se; ++i, ++si) if (*i != si->key) break;
      for (int pop = se - si; pop > 0; pop--) (this->*complete_cb)(PopBack(stack));
    }

    void AddPrefixesOf(const String &in, const String &lookahead) {
      int add_prefixes;
      if (patricia) {
        int matching_lookahead = MismatchOffset(in.begin(), in.end(), lookahead.begin(), lookahead.end()); 
        add_prefixes = matching_lookahead - stack.size();
      } else add_prefixes = in.size() - stack.size();
      for (int i=0; i<add_prefixes; i++) {
        stack.emplace_back(in[stack.size()-1], (this->*insert_cb)(String(), NULL));
      }
    }
  };

  vector<Node> data;
  vector<Next> next;
  vector<Value> val;
  int head=0;
  Trie() {}
  template <class I> Trie(I b, I e) { SortedInputConstructor(this, b, e, false); }

  virtual Node *DeadEnd(const String &in, int leaf_ind, Node *n) { return 0; }
  virtual void ComputeStateFromChildren(Node *n) {}

  Node *Query(const String &s) {
    if (!head) return 0;
    if (s.empty()) return &data[head-1];
    Node *n = &data[head-1];
    for (auto c = s.begin(), ce = s.end(); c != ce; ++c)
      if (int child = GetChildIndex(n, *c)) n = &data[child-1];
      else return DeadEnd(s, c-s.begin(), n);
    return n;
  }

  int GetChildIndex(Node *n, K key) {
    Next *t = &next[n->next_ind-1], *te = t + n->next_len;
    auto tn = lower_bound(t, te, Next(key));
    return (tn != te && tn->key == key) ? tn->next : 0;
  }
};

// Patricia Trie

template <class K, class V> struct PatriciaTrieNodeValue {
  V val;
  basic_string<K> leaf;
  PatriciaTrieNodeValue(const basic_string<K> &s, int leaf_ind, const V &v) : val(v), leaf(s.substr(leaf_ind)) {}
};

template <class K, class V, class TN = TrieNode<K>, class TV = PatriciaTrieNodeValue<K, V> >
struct PatriciaTrie : public Trie<K, V, TN, TV> {
  typedef Trie<K, V, TN, TV> Parent;
  template <class I> PatriciaTrie(I b, I e) { typename Parent::SortedInputConstructor(this, b, e, true); }

  virtual typename Parent::Node *DeadEnd(const typename Parent::String &key, int leaf_ind,
                                         typename Parent::Node *n) {
    if (!n->val_ind) return 0;
    typename Parent::Value *v = &Parent::val[n->val_ind-1];
    if (key.substr(leaf_ind) != v->leaf) return 0;
    return n;
  }
};

// Patricia Completer

template <class K, class V> struct PatriciaCompleterNodeValue {
  V val;
  basic_string<K> key;
  PatriciaCompleterNodeValue(const basic_string<K> &s, int, const V &v) : val(v), key(s) {}
};

template <class K, const int C> struct PatriciaCompleterNode : public TrieNode<K> {
  static const int completions = C;
  int completion[completions];
  PatriciaCompleterNode(int V=0) : TrieNode<K>(V) { memzero(completion); }
};

template <class K, class V, class TN = PatriciaCompleterNode<K, 10>, class TV = PatriciaCompleterNodeValue<K, V> >
struct PatriciaCompleter : public Trie<K, V, TN, TV> {
  typedef Trie<K, V, TN, TV> Parent;
  function<int(const V*)> completion_sort_cb;
  template <class I> PatriciaCompleter(I b, I e) : completion_sort_cb([](const V *v){ return *v; }) {
    typename Parent::SortedInputConstructor(this, b, e, true);
  }

  virtual typename Parent::Node *DeadEnd(const typename Parent::String &key, int leaf_ind,
                                         typename Parent::Node *n) {
    if (!n->val_ind) return 0;
    typename Parent::Value *v = &Parent::val[n->val_ind-1];
    if (key.substr(leaf_ind) != v->key.substr(leaf_ind)) return 0;
    return n;
  }

  virtual void ComputeStateFromChildren(typename Parent::Node *n) {
    TopN<pair<int, int> > completion(TN::completions);
    if (n->val_ind) AddCompletion(&completion, n->val_ind);
    for (typename Parent::Next *t = &Parent::next[n->next_ind-1], *te = t + n->next_len; t != te; ++t) {
      auto nn = &Parent::data[t->next-1]; 
      if (nn->val_ind) AddCompletion(&completion, nn->val_ind);
    }
    int ind = 0;
    for (auto i : completion.data) n->completion[ind++] = i.second;
  }

  void AddCompletion(TopN<pair<int, int> > *out, int val_ind) const {
    out->Insert(pair<int, int>(-completion_sort_cb(&Parent::val[val_ind-1].val), val_ind));
  }
};

// Aho-Corasick Finite State Machine

template <class K> struct AhoCorasickFSMNode : public TrieNode<K> {
  int fail=-1, out_ind=-1, out_len=0;
  AhoCorasickFSMNode(int V=0) : TrieNode<K>(V) {}
};

template <class K = char> struct AhoCorasickFSM {
  struct PatternTable : public vector<int> {
    PatternTable(int s) : vector<int>(s) { std::iota(begin(), end(), 0); }
  };
  typedef Trie<K, int, AhoCorasickFSMNode<K> > FSM;
  typedef typename FSM::String String;
  typedef IterPair<typename vector<String>::const_iterator, typename PatternTable::const_iterator> InputPair;

  PatternTable pattern;
  FSM fsm;
  vector<int> out;
  int match_state=0, max_pattern_size=0;

  AhoCorasickFSM(const vector<String> &sorted_in) : pattern(sorted_in.size()),
  fsm(InputPair(sorted_in.begin(), pattern.begin()), InputPair(sorted_in.end(), pattern.end())),
  match_state(fsm.head) {
    for (int i=0, l=sorted_in.size(); i != l; ++i)
      Max(&max_pattern_size, (pattern[i] = sorted_in[i].size()));
    if (!fsm.head || !fsm.data.size()) return;
    std::queue<int> q;
    auto n = &fsm.data[fsm.head-1];
    n->fail = fsm.head;
    for (auto t = &fsm.next[n->next_ind-1], te = t+n->next_len; t != te; ++t) {
      q.push(t->next);
      auto nn = &fsm.data[t->next-1]; 
      nn->fail = fsm.head;
      AddOutput(t->next, nn, NULL);
    }
    while (q.size()) {
      int node_ind = PopFront(q);
      n = &fsm.data[node_ind-1];
      for (auto t = &fsm.next[n->next_ind-1], te = t+n->next_len; t != te; ++t) {
        q.push(t->next);
        int state = n->fail;
        for (;;) {
          auto fn = &fsm.data[state-1];
          if (int child = fsm.GetChildIndex(fn, t->key)) { state = child; break; }
          else if (state == fsm.head) break;
          state = fn->fail;
        }
        auto fn = &fsm.data[state-1];
        auto nn = &fsm.data[t->next-1]; 
        nn->fail = state;
        AddOutput(t->next, nn, fn);
      }
    }
  }

  void Reset() { match_state = fsm.head; }
  void AddOutput(int nid, typename FSM::Node *n, typename FSM::Node *f) {
    int start_size = out.size();
    if (n->val_ind) out.push_back(fsm.val[n->val_ind-1].val);
    if (f) for (int i=0, l=f->out_len; i != l; ++i) out.push_back(out[f->out_ind+i]);
    if ((n->out_len = out.size() - start_size)) n->out_ind = start_size;
  }

  template <class I> I MatchOne(I s, I b, I e, vector<Regex::Result> *result) {
    for (auto c = b; c != e; ++c) {
      for (;;) {
        auto n = &fsm.data[match_state-1];
        if (int child = fsm.GetChildIndex(n, *c)) { match_state = child; break; }
        else if (match_state == fsm.head) break;
        match_state = n->fail;
      }
      auto n = &fsm.data[match_state-1];
      if (!result || !n->out_len) continue;
      for (auto i = &out[n->out_ind], e = i + n->out_len; i != e; ++i) {
        int pattern_len = pattern[*i], offset = c - s - pattern_len + 1;
        result->push_back(Regex::Result(offset, offset + pattern_len));
      }
      return ++c;
    }
    return e;
  }

  template <class I> void Match(I s, I e, vector<Regex::Result> *result) {
    for (auto b = s; b != e; /**/) b = MatchOne(s, b, e, result);
  }

  void Match(const String &text, vector<Regex::Result> *result) { Match(text.begin(), text.end(), result); }
};

// StringMatcher

template <class K=char, class M=AhoCorasickFSM<K> > struct StringMatcher {
  typedef basic_string<K> String;
  struct iterator {
    StringMatcher *parent;
    typename String::const_iterator s, e, b, nb;
    bool match_begin=0, match_begin_next=0, match_end=0;
    int size=0;

    iterator(StringMatcher *p, const String &in)
      : parent(p), s(in.begin()), e(in.end()) { if ((b=nb=s) != e) ++(*this); }

    bool MatchBegin() const { return match_begin; }
    bool Matching  () const { return match_begin || match_end || (parent->in_match && !match_begin_next); }
    bool MatchEnd  () const { return match_end; }

    iterator &operator++() {
      match_begin = match_begin_next;
      match_begin_next = match_end = 0;
      if ((b = nb) == e) return *this;
      if (parent->in_match) {
        if ((nb = std::find_if(b, e, parent->match_end_condition)) != e) { parent->in_match=0; match_end=1; }
      } else {
        vector<Regex::Result> result;
        nb = parent->impl ? parent->impl->MatchOne(s, b, e, &result) : e;
        if (result.size()) { parent->in_match=1; parent->match=result[0]; match_begin_next=1; }
      }
      size = nb - b;
      return *this;
    }
  };

  M *impl;
  bool in_match=0;
  string match_buf;
  Regex::Result match;
  int last_input_size=0;
  int (*match_end_condition)(int) = &isspace;
  function<void(const String&)> match_cb;
  StringMatcher(M *m=0) : impl(m) {}

  iterator Begin(const String &in) {
    if (in_match) match -= last_input_size;
    last_input_size = in.size();
    return iterator(this, in);
  }

  /// Match works with arbitrary segmentation, but filtering needs appropriate buffering
  void Match(const String &in, string *filtered=0) {
    for (auto i = Begin(in); i.b != i.e; ++i) {
      if (i.MatchBegin()) {
        match_buf.clear();
        if (filtered) { int ms=match.end-match.begin, fs=filtered->size(); if (fs >= ms) filtered->resize(fs-ms); }
      }
      if (i.Matching()) match_buf.append(i.b, i.nb);
      else if (filtered) filtered->append(i.b, i.nb);
      if (i.MatchEnd()) match_cb(match_buf);
    }
  }
};

}; // namespace LFL
#endif // LFL_LFAPP_TYPES_TRIE_H__
