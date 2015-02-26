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

#ifndef __LFL_LFAPP_TRIE_H__
#define __LFL_LFAPP_TRIE_H__
namespace LFL {

template <class K> struct TrieNode {
    int val_ind=0, next_ind=0;
    unsigned short next_len=0, next_i=0;
    TrieNode(int V=0) : val_ind(V) {}
};

template <class K> struct TrieNodeNext {
    K key; int next; 
    TrieNodeNext(const K &k=K(), int n=0) : key(k), next(n) {}
    bool operator<(const TrieNodeNext &n) const { SortImpl1(key, n.key) } 
};

template <class K, class V> struct TrieNodeValue {
    V val;
    TrieNodeValue(const basic_string<K>&, const V &v) : val(v) {}
};

template <class K, class V, class TN = TrieNode<K>, class TV = TrieNodeValue<K, V> > struct Trie {
    typedef Trie<K, V, TN, TV> Self;
    typedef basic_string<K> String;
    typedef TrieNodeNext<K> Next;
    typedef TN Node;
    typedef TV Value;

    struct SortedInputConstructor {
        struct Prenode {
            K key; int node, next_len; 
            Prenode(const K &k, int n) : key(k), node(n), next_len(0) {}
        };
        typedef int  (SortedInputConstructor::*InsertCB)(const String &k, const V*);
        typedef void (SortedInputConstructor::*CompleteCB)(const Prenode&);

        Self *out;
        int insert_id=0;
        bool patricia=true;
        vector<Prenode> stack;
        InsertCB     insert_cb = &SortedInputConstructor::Phase1Insert;
        CompleteCB complete_cb = &SortedInputConstructor::Phase1Complete;

        template <class I> SortedInputConstructor(Self *O, I b, I e, bool P=true) : out(O), patricia(P) {
            stack.emplace_back(K(), (this->*insert_cb)("", NULL));
            Input(b, e);
            insert_cb   = &SortedInputConstructor::Phase2Insert;
            complete_cb = &SortedInputConstructor::Phase2Complete;
            stack.emplace_back(K(), (this->*insert_cb)("", NULL));
            Input(b, e);
        }
        template <class I> void Input(I b, I e) { 
            for (auto i = b; i != e; /**/) {
                const String &k = i->first;
                const V      &v = i->second;
                Input(k, ++i != e ? i->first : "", v);
            }
            while (stack.size()) (this->*complete_cb)(PopBack(stack));
            out->head = 1;
        }
        void Input(const String &in, const String &lookahead, const V &v) {
            PopPrefixesNotMatching(in);
            AddPrefixesOf(in, lookahead);
            stack.emplace_back(in[stack.size()-1], (this->*insert_cb)(in.substr(stack.size()), &v));
        }

        int Phase1Insert(const String &leaf, const V *v) { 
            if (v) out->val.push_back(Value(leaf, *v)); 
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
                add_prefixes = matching_lookahead - stack.size() - (matching_lookahead == in.size());
            } else add_prefixes = in.size() - stack.size();
            for (int i=0; i<add_prefixes; i++) {
                stack.emplace_back(in[stack.size()-1], (this->*insert_cb)("", NULL));
            }
        }
    };

    vector<Node> data;
    vector<Next> next;
    vector<Value> val;
    int head=0;
    template <class I> Trie(I b, I e) { SortedInputConstructor(this, b, e, Patricia()); }

    virtual bool Patricia() const { return 0; }
    virtual Node *DeadEnd(const String &leaf, Node *n) { return 0; }
    virtual void ComputeStateFromChildren(Node *n) {}

    Node *Query(const String &s) {
        if (!head) return 0;
        if (s.empty()) return &data[head-1];
        Node *n = &data[head-1];
        for (auto c = s.begin(), ce = s.end(); c != ce; ++c) {
            Next *t = &next[n->next_ind-1], *te = t + n->next_len;
            auto tn = lower_bound(t, te, Next(*c));
            if (tn == te || tn->key != *c) return DeadEnd(s.substr(c-s.begin()), n);
            n = &data[tn->next-1];
        }
        return n;
    }
};

template <class K, class V> struct PatriciaTrieNodeValue {
    V val;
    basic_string<K> leaf;
    PatriciaTrieNodeValue(const basic_string<K> &s, const V &v) : val(v), leaf(s) {}
};

template <class K, class V, class TN = TrieNode<K>, class TV = PatriciaTrieNodeValue<K, V> >
struct PatriciaTrie : public Trie<K, V, TN, TV> {
    typedef Trie<K, V, TN, TV> Parent;
    template <class I> PatriciaTrie(I b, I e) : Trie<K, V, TN, TV>(b, e) {}

    virtual bool Patricia() const { return 1; }
    virtual typename Parent::Node *DeadEnd(const typename Parent::String &leaf, typename Parent::Node *n) {
        if (!n->val_ind) return 0;
        typename Parent::Value *v = &Parent::val[n->val_ind-1];
        if (leaf != v->leaf) return 0;
        return n;
    }
};

struct AhoCorasickMatcher {
    unsigned fail;
    AhoCorasickMatcher(const vector<string> &sorted_in) {
    }
};

}; // namespace LFL
#endif // __LFL_LFAPP_TRIE_H__
