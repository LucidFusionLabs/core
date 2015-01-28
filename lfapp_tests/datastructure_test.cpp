/*
 * $Id: lfapp.cpp 1309 2014-10-10 19:20:55Z justin $
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

#include "gtest/gtest.h"
#include "lfapp/lfapp.h"

GTEST_API_ int main(int argc, const char **argv) {
    testing::InitGoogleTest(&argc, (char**)argv);
    LFL::FLAGS_default_font = LFL::FakeFont::Filename();
    CHECK_EQ(LFL::app->Create(argc, argv, __FILE__), 0);
    return RUN_ALL_TESTS();
}

namespace LFL {
DEFINE_int(size,         1024*1024, "Test size"); 
DEFINE_int(rand_key_min, 0,         "Min key");
DEFINE_int(rand_key_max, 10000000,  "Max key");

string GraphVizFooter() { return "}\r\n"; }
string GraphVizHeader(const string &name) {
    return StrCat("digraph ", name, " {\r\n"
                  "rankdir=LR;\r\n"
                  "size=\"8,5\"\r\n"
                  "node [style = solid];\r\n"
                  "node [shape = circle];\r\n");
}

struct MyEnvironment : public ::testing::Environment {
    vector<int> rand_db, incr_db, decr_db, sorted_rand_db, sorted_incr_db, sorted_decr_db;
    vector<Triple<string, vector<int>*, vector<int>*> > db;

    virtual ~MyEnvironment() {}
    virtual void TearDown() { INFO(Singleton<PerformanceTimers>::Get()->DebugString()); }
    virtual void SetUp() { 
        unordered_set<int> rand_unique;
        for (int i=0; i<FLAGS_size; i++) {
            int ri = Rand(FLAGS_rand_key_min, FLAGS_rand_key_max);
            if (!rand_unique.insert(ri).second) { i--; continue; }
            rand_db.push_back(ri);
        }
        for (int i=0; i<FLAGS_size; i++) incr_db.push_back(i);
        for (int i=0; i<FLAGS_size; i++) decr_db.push_back(FLAGS_size-i);

        sorted_rand_db = rand_db; sort(sorted_rand_db.begin(), sorted_rand_db.end());
        sorted_incr_db = incr_db; sort(sorted_incr_db.begin(), sorted_incr_db.end());
        sorted_decr_db = decr_db; sort(sorted_decr_db.begin(), sorted_decr_db.end());

        db.emplace_back("rand", &rand_db, &sorted_rand_db);
        db.emplace_back("incr", &incr_db, &sorted_incr_db);
        db.emplace_back("decr", &decr_db, &sorted_decr_db);
    }
};

MyEnvironment* const my_env = (MyEnvironment*)::testing::AddGlobalTestEnvironment(new MyEnvironment);

template <class K, class V> struct SkipList {
    struct Node {
        K key; int val_ind, next_ind, leveldown_ind;
        Node(K k=K(), int VI=-1, int NI=-1, int LDI=-1) : key(k), val_ind(VI), next_ind(NI), leveldown_ind(LDI) {}
    };
    FreeListVector<V> val;
    FreeListVector<Node> node;
    int count=0, head_ind=-1;

    void Insert(const K &k, const V &v) {
        vector<pair<int, int> > zipper;
        zipper.reserve(64);
        int ind = FindNode(k, &zipper), next_ind;
        if (ind < 0 || (next_ind = node[ind].next_ind) < 0) {}
        else if (const Node *n = &node[next_ind]) if (k == n->key) { val[n->val_ind] = v; return; }
        int val_ind = val.Insert(v), level = RandomLevel();
        for (int i=0, zl=zipper.size(), last_ind=-1; i<level; i++) {
            int zi = i < zl ? zl-i-1 : -1;
            last_ind = node.Insert(Node(k, val_ind, zi < 0 ? -1 : zipper[zi].second, last_ind));
            if (zi >= 0) node[zipper[zi].first].next_ind = last_ind;
            else head_ind = node.Insert(Node(-1, -1, last_ind, head_ind));
        }
        count++;
    }
    bool Erase(const K &k) {
        vector<pair<int, int> > zipper;
        zipper.reserve(64);
        int ind = FindNode(k, &zipper), next_ind, val_ind, erase_ind;
        if (ind < 0 || (next_ind = node[ind].next_ind) < 0) return false;
        const Node *n = &node[next_ind];
        if (k != n->key) return false;
        val.Erase((val_ind = n->val_ind));
        for (int i=zipper.size()-1; i>=0; i--) {
            Node *p = &node[zipper[i].first];
            if ((erase_ind = p->next_ind) < 0 || (n = &node[erase_ind])->val_ind != val_ind) break;
            p->next_ind = n->next_ind;
            node.Erase(erase_ind);
        }
        CHECK(count);
        return count--;
    }
    V *Find(const K &k) {
        int ind = FindNode(k), next_ind;
        if (ind < 0 || (next_ind = node[ind].next_ind) < 0) return 0;
        const Node *n = &node[next_ind];
        return k == n->key ? &val[n->val_ind] : 0;
    }
    int FindNode(const K &k, vector<pair<int, int> > *zipper=0) const {
        int ind = head_ind, next_ind;
        for (int leveldown_ind; ind >= 0; ind = leveldown_ind) {
            const Node *n = &node[ind], *nn;
            while (n->next_ind >= 0 && (nn = &node[n->next_ind])->key < k) { ind = n->next_ind; n = nn; }
            if (zipper) zipper->emplace_back(ind, n->next_ind);
            if ((leveldown_ind = n->leveldown_ind) < 0) break;
        }
        return ind;
    }
    string DebugString() const {
        string v;
        for (int i=0, ind=head_ind; ind >= 0; ind = node[ind].leveldown_ind, i--) {
            StrAppend(&v, "\n", i, ": H");
            for (int pi = node[ind].next_ind; pi >= 0; pi = node[pi].next_ind)
                StrAppend(&v, " ", node[pi].val_ind < 0 ? "-1" : StrCat(val[node[pi].val_ind]));
        } return v;
    }
    static int RandomLevel() { static float logP = log(.5); return 1 + (int)(log(Rand(0,1)) / logP); }
};

struct AVLTreeZipper {
    int sum=0;
    vector<pair<int, bool> > path;
    AVLTreeZipper() { path.reserve(64); }
    void Reset() { sum=0; path.clear(); }
};

template <class K, class V> struct AVLTreeNode {
    enum { Left, Right };
    K key; unsigned val:30, left:30, right:30, height:6;
    AVLTreeNode(K k, unsigned v) : key(k), val(v), left(0), right(0), height(1) {}
    void SetChildren(unsigned l, unsigned r) { left=l; right=r; }
    void SwapKV(AVLTreeNode *n) { swap(key, n->key); unsigned v=val; val=n->val; n->val=v; }
    virtual bool LessThan(const K &k, int ind, AVLTreeZipper *z) const { return key < k; }
    virtual bool MoreThan(const K &k, int ind, AVLTreeZipper *z) const { return k < key; }
    virtual void ComputeStateFromChildren(const AVLTreeNode<K,V> *lc, const AVLTreeNode<K,V> *rc) {
        height = max(lc ? lc->height : 0, rc ? rc->height : 0) + 1;
    }
};

template <class K, class V, class Node = AVLTreeNode<K,V> > struct AVLTree {
    struct Query {
        const K key; const V *val; V *ret; AVLTreeZipper *z;
        Query(const K &k, const V *v=0, AVLTreeZipper *Z=0) : key(k), val(v), ret(0), z(Z) {}
    };

    FreeListVector<Node> node;
    FreeListVector<V> val;
    bool update_on_dup_insert=1;
    int head=0;

    const V* Find  (const K &k) const       { Query q=GetQuery(k); int vind = FindNode  (head, &q); return vind ? &val[vind-1] : 0; }
    V*       Find  (const K &k)             { Query q=GetQuery(k); int vind = FindNode  (head, &q); return vind ? &val[vind-1] : 0; }
    V*       Insert(const K &k, const V &v) { Query q=GetQuery(k, &v); head = InsertNode(head, &q); return q.ret; }
    bool     Erase (const K &k)             { Query q=GetQuery(k);     head = EraseNode (head, &q); return q.ret; }

    virtual Query GetQuery(const K &k, const V *v=0) { return Query(k, v); }
    virtual K GetCreateNodeKey(const Query *q) const { return q->key; }
    virtual int ResolveInsertCollision(int ind, Query *q) { 
        Node *n = &node[ind-1];
        if (update_on_dup_insert) q->ret = &(val[n->val] = *q->val);
        return ind;
    }

    int FindNode(int ind, Query *q) const {
        if (!ind) return 0;
        const Node *n = &node[ind-1];
        if      (n->MoreThan(q->key, ind, q->z)) return FindNode(n->left,  q);
        else if (n->LessThan(q->key, ind, q->z)) return FindNode(n->right, q);
        else return n->val+1;
    }
    int InsertNode(int ind, Query *q) {
        if (!ind) return CreateNode(q);
        Node *n = &node[ind-1];
        if      (n->MoreThan(q->key, ind, q->z)) node[ind-1].left  = InsertNode(n->left,  q);
        else if (n->LessThan(q->key, ind, q->z)) node[ind-1].right = InsertNode(n->right, q);
        else return ResolveInsertCollision(ind, q);
        return Balance(ind);
    }
    int CreateNode(Query *q) {
        int val_ind = val.Insert(*q->val);
        q->ret = &val[val_ind];
        return node.Insert(Node(GetCreateNodeKey(q), val_ind))+1;
    }
    int EraseNode(int ind, Query *q) {
        if (!ind) return 0;
        Node *n = &node[ind-1];
        if      (n->MoreThan(q->key, ind, q->z)) n->left  = EraseNode(n->left,  q);
        else if (n->LessThan(q->key, ind, q->z)) n->right = EraseNode(n->right, q);
        else {
            q->ret = &val[n->val];
            int left = n->left, right = n->right;
            val.Erase(n->val);
            node.Erase(ind-1);
            if (!right) return left;
            ind = GetMinNode(right);
            right = EraseMinNode(right);
            node[ind-1].SetChildren(left, right);
        }
        return Balance(ind);
    }
    int EraseMinNode(int ind) {
        Node *n = &node[ind-1];
        if (!n->left) return n->right;
        n->left = EraseMinNode(n->left);
        return Balance(ind);
    }
    int Balance(int ind) {
        Node *n = &node[ind-1];
        ComputeStateFromChildren(n);
        int nbal = GetBalance(ind);
        if (GetBalance(ind) > 1) {
            if (GetBalance(n->right) < 0) n->right = RotateRight(n->right);
            return RotateLeft(ind);
        } else if (nbal < -1) {
            if (GetBalance(n->left) > 0) n->left = RotateLeft(n->left);
            return RotateRight(ind);
        } else return ind;
    }
    int RotateLeft(int ind) {
        int right_ind;
        Node *n = &node[ind-1], *o = &node[(right_ind = n->right)-1];
        n->right = o->left;
        o->left = ind;
        ComputeStateFromChildren(n);
        ComputeStateFromChildren(o);
        return right_ind;
    }
    int RotateRight(int ind) {
        int left_ind;
        Node *n = &node[ind-1], *o = &node[(left_ind = n->left)-1];
        n->left = o->right;
        o->right = ind;
        ComputeStateFromChildren(n);
        ComputeStateFromChildren(o);
        return left_ind;
    }
    void ComputeStateFromChildren(Node *n) {
        n->ComputeStateFromChildren(n->left ? &node[n->left -1] : 0, n->right ? &node[n->right-1] : 0);
    }
    int GetMinNode(int ind) const {
        const Node *n = &node[ind-1];
        return n->left ? GetMinNode(n->left) : ind;
    }
    int GetBalance(int ind) const {
        const Node *n = &node[ind-1];
        return (n->right ? node[n->right-1].height : 0) -
               (n->left  ? node[n->left -1].height : 0);
    }

    virtual string DebugString(const string &name=string()) const {
        string ret = GraphVizHeader(StrCat("AVLTree", name.size()?"_":"", name));
        PrintNodes(head, &ret);
        PrintEdges(head, &ret);
        return ret + GraphVizFooter();
    }
    virtual void PrintNodes(int ind, string *out) const {
        if (!ind) return;
        const Node *n = &node[ind-1];
        PrintNodes(n->left, out);
        StrAppend(out, "node [label = \"", n->key, " ", n->val, "\"];\r\n");
        PrintNodes(n->right, out);
    }
    virtual void PrintEdges(int ind, string *out) const {
        if (!ind) return;
        const Node *n = &node[ind-1], *l=n->left?&node[n->left-1]:0, *r=n->right?&node[n->right-1]:0;
        if (l) { PrintEdges(n->left, out);  StrAppend(out, "\"", n->key, "\" -> \"", l->key, "\" [ label = \"left\"  ];\r\n"); }
        if (r) { PrintEdges(n->right, out); StrAppend(out, "\"", n->key, "\" -> \"", r->key, "\" [ label = \"right\" ];\r\n"); }
    }
};

template <class K, class V> struct PrefixSumKeyedAVLTreeNode : public AVLTreeNode<K,V> {
    typedef AVLTreeNode<K,V> Parent;
    typedef PrefixSumKeyedAVLTreeNode<K,V> Self;
    int left_sum=0, right_sum=0;
    PrefixSumKeyedAVLTreeNode(K k=K(), unsigned v=0) : AVLTreeNode<K,V>(k,v) {}
    virtual bool LessThan(const K &k, int ind, AVLTreeZipper *z) const {
        if (!((z->sum + left_sum + Parent::key) < k)) return 0;
        z->sum += left_sum + Parent::key;
        return 1;
    }
    virtual bool MoreThan(const K &k, int ind, AVLTreeZipper *z) const {
        if (!(k < (z->sum + left_sum + Parent::key))) return 0;
        return 1;
    }
    virtual void ComputeStateFromChildren(const Self *lc, const Self *rc) {
        Parent::ComputeStateFromChildren(lc, rc);
        left_sum  = lc ? (lc->left_sum + lc->right_sum + lc->key) : 0;
        right_sum = rc ? (rc->left_sum + rc->right_sum + rc->key) : 0;
    }
};

template <class K, class V, class N = PrefixSumKeyedAVLTreeNode<K,V> >
struct PrefixSumKeyedAVLTree : public AVLTree<K,V,N> {
    typedef AVLTree<K,V,N> Parent;
    mutable AVLTreeZipper z;
    virtual K NodeValue(const V *v) const { return 1; }
    virtual K GetCreateNodeKey(const typename Parent::Query *q) const { return NodeValue(q->ret); }
    virtual typename Parent::Query GetQuery(const K &k, const V *v=0) {
        z.Reset();
        return typename Parent::Query(k + (v ? 0 : 1), v, &z);
    }
    virtual int ResolveInsertCollision(int ind, typename Parent::Query *q) { 
        Parent::node[ind-1].left = ResolveInsertCollision(Parent::node[ind-1].left, ind, q);
        return Parent::Balance(ind);
    }
    virtual int ResolveInsertCollision(int ind, int dup_ind, typename Parent::Query *q) {
        if (!ind) {
            int new_ind = Parent::CreateNode(q);
            Parent::node[new_ind-1].SwapKV(&Parent::node[dup_ind-1]);
            return new_ind;
        }
        Parent::node[ind-1].right = ResolveInsertCollision(Parent::node[ind-1].right, dup_ind, q);
        return Parent::Balance(ind);
    }
    virtual void PrintNodes(int ind, string *out) const {
        if (!ind) return;
        const N *n = &Parent::node[ind-1];
        const V *v = &Parent::val[n->val];
        PrintNodes(n->left, out);
        StrAppend(out, "node [label = \"", *v, " v:", n->key,
                  "\nlsum:", n->left_sum, " rsum:", n->right_sum, "\"];\r\n\"", *v, "\";\r\n");
        PrintNodes(n->right, out);
    }
    virtual void PrintEdges(int ind, string *out) const {
        if (!ind) return;
        const N *n = &Parent::node[ind-1], *l=n->left?&Parent::node[n->left-1]:0, *r=n->right?&Parent::node[n->right-1]:0;
        const V *v = &Parent::val[n->val], *lv = l?&Parent::val[l->val]:0, *rv = r?&Parent::val[r->val]:0;
        if (l) { PrintEdges(n->left,  out); StrAppend(out, "\"", *v, "\" -> \"", *lv, "\" [ label = \"left\"  ];\r\n"); }
        if (r) { PrintEdges(n->right, out); StrAppend(out, "\"", *v, "\" -> \"", *rv, "\" [ label = \"right\" ];\r\n"); }
    }
    void LoadFromSortedVal() {
        CHECK_EQ(0, Parent::node.size());
        Parent::head = BuildTreeFromSortedVal(0, Parent::val.size()-1);
    }
    int BuildTreeFromSortedVal(int beg_val_ind, int end_val_ind) {
        if (end_val_ind < beg_val_ind) return 0;
        int mid_val_ind = (beg_val_ind + end_val_ind) / 2;
        int ind = Parent::node.Insert(N(NodeValue(&Parent::val[mid_val_ind]), mid_val_ind))+1;
        Parent::node[ind-1].left  = BuildTreeFromSortedVal(beg_val_ind,   mid_val_ind-1);
        Parent::node[ind-1].right = BuildTreeFromSortedVal(mid_val_ind+1, end_val_ind);
        Parent::ComputeStateFromChildren(&Parent::node[ind-1]);
        return ind;
    }
};

struct RedBlackTreeZipper {
    int sum=0;
    vector<pair<int, bool> > path;
    RedBlackTreeZipper() { path.reserve(64); }
    void Reset() { sum=0; path.clear(); }
};

template <class K, class V> struct RedBlackTreeNode {
    enum { Left, Right };
    K key; unsigned val:31, left, right, parent, color:1;
    RedBlackTreeNode(K k, unsigned v) : key(k), val(v), left(0), right(0), parent(0), color(0) {}
    void Assign(K k, unsigned v) { key=k; val=v; }
    virtual bool LessThan(const K &k, int ind, RedBlackTreeZipper *z) const { return key < k; }
    virtual bool MoreThan(const K &k, int ind, RedBlackTreeZipper *z) const { return k < key; }
};

template <class K, class V, class Node = RedBlackTreeNode<K,V> > struct RedBlackTree {
    enum Color { Red, Black };

    FreeListVector<Node> node;
    FreeListVector<V> val;
    int head=0;

    const V* Find (const K &k) const { const Node *n = FindNode(k); return n ? &val[n->val] : 0; }
    /**/  V* Find (const K &k)       { const Node *n = FindNode(k); return n ? &val[n->val] : 0; }
    const Node *FindNode(const K &k, RedBlackTreeZipper *z=0) const {
        for (int ind = head; ind;) {
            const Node *n = &node[ind-1];
            if      (n->MoreThan(k, ind, z)) ind = n->left;
            else if (n->LessThan(k, ind, z)) ind = n->right;
            else return n;
        } return 0;
    }

    V* Insert(const K &k, const V &v, RedBlackTreeZipper *z=0) {
        Node *new_node;
        int new_ind = node.Insert(Node(k, 0))+1, val_ind;
        if (int ind = head) {
            for (;;) {
                Node *n = &node[ind-1];
                if      (n->MoreThan(k, ind, z)) { if (!n->left)  { n->left  = new_ind; break; } ind = n->left;  }
                else if (n->LessThan(k, ind, z)) { if (!n->right) { n->right = new_ind; break; } ind = n->right; }
                else                             { node.Erase(new_ind-1); return &(val[n->val] = v); }
            }
            (new_node = &node[new_ind-1])->parent = ind;
        } else new_node = &node[(head = new_ind)-1];
        new_node->val = val_ind = val.Insert(v);
        InsertBalance(new_ind);
        return &val[val_ind];
    }
    void InsertBalance(int ind) {
        int p_ind, gp_ind, u_ind;
        Node *n = &node[ind-1], *p, *gp, *u;
        if (!(p_ind = n->parent)) { n->color = Black; return; }
        if ((p = &node[p_ind-1])->color == Black) return;
        gp = &node[(gp_ind = p->parent)-1];
        if ((u_ind = GetSibling(gp, p_ind)) && (u = &node[u_ind-1])->color == Red) {
            p->color = u->color = Black;
            gp->color = Red;
            return InsertBalance(gp_ind);
        }
        do {
            if      (ind == p->right && p_ind == gp->left)  { RotateLeft (p_ind); ind = node[ind-1].left; }
            else if (ind == p->left  && p_ind == gp->right) { RotateRight(p_ind); ind = node[ind-1].right; }
            else break;
            p  = &node[(p_ind  = GetParent(ind))-1];
            gp = &node[(gp_ind = p->parent)     -1];
        } while (0);
        p->color = Black;
        gp->color = Red;
        if (ind == p->left && p_ind == gp->left) RotateRight(gp_ind);
        else                                     RotateLeft (gp_ind);
    }

    bool Erase(const K &k, RedBlackTreeZipper *z=0) {
        Node *n = (Node*)FindNode(k, z);
        if (!n) return false;
        val.Erase(n->val);
        int ind = n - &node.data[0] + 1, m_ind;
        if (n->left && n->right) {
            Node *m = &node[(m_ind = GetMaxNode(n->left))-1];
            n->Assign(m->key, m->val);
            n = &node[(ind = m_ind)-1];
        }
        int child = X_or_Y(n->right, n->left);
        if (n->color == Black) {
            n->color = GetColor(child);
            EraseBalance(ind);
        }
        ReplaceNode(ind, child);
        node.Erase(ind-1);
        return true;
    }
    void EraseBalance(int ind) {
        int p_ind, s_ind;
        Node *n = &node[ind-1], *p, *s;
        if (!n->parent) return;
        p = &node[(p_ind = n->parent)-1];
        s = &node[(s_ind = GetSibling(p, ind))-1];
        if (s->color == Red) {
            p->color = Red;
            s->color = Black;
            if (ind == p->left) RotateLeft (p_ind);
            else                RotateRight(p_ind);
            s = &node[(s_ind = GetSibling(p, ind))-1];
        }
        if (s->color == Black) {
            int csl = GetColor(s->left), csr = GetColor(s->right);
            if (p->color == Black && csl == Black && csr == Black) {
                s->color = Red;
                return EraseBalance(p_ind);
            }
            if (p->color == Red && csl == Black && csr == Black) {
                s->color = Red;
                p->color = Black;
                return;
            }
            if      (ind == p->left  && csl == Red   && csr == Black) { s->color = Red; node[s->left -1].color = Black; RotateRight(s_ind); }
            else if (ind == p->right && csl == Black && csr == Red  ) { s->color = Red; node[s->right-1].color = Black; RotateLeft (s_ind); }
            s = &node[(s_ind = GetSibling(p, ind))-1];
        }
        s->color = p->color;
        p->color = Black;
        if (ind == p->left) { node[s->right-1].color = Black; RotateLeft (p_ind); }
        else                { node[s->left -1].color = Black; RotateRight(p_ind); }
    }

    void RotateLeft (int ind) {
        int right_ind;
        Node *n = &node[ind-1], *o = &node[(right_ind = n->right)-1];
        ReplaceNode(ind, right_ind);
        n->right = o->left;
        if (o->left) node[o->left-1].parent = ind;
        o->left = ind;
        n->parent = right_ind;
    }
    void RotateRight(int ind) {
        int left_ind;
        Node *n = &node[ind-1], *o = &node[(left_ind = n->left)-1];
        ReplaceNode(ind, left_ind);
        n->left = o->right;
        if (o->right) node[o->right-1].parent = ind;
        o->right = ind;
        n->parent = left_ind;
    }
    void ReplaceNode(int ind, int new_ind) {
        Node *n = &node[ind-1];
        if (!n->parent) head = new_ind;
        else if (Node *parent = &node[n->parent-1]) {
            if (ind == parent->left) parent->left  = new_ind;
            else                     parent->right = new_ind;
        }
        if (new_ind) node[new_ind-1].parent = n->parent;
    }

    int GetColor  (int ind) const { return ind ? node[ind-1].color : Black; }
    int GetParent (int ind) const { return node[ind-1].parent; }
    int GetSibling(int ind) const { return GetSibling(&node[GetParent(ind)-1], ind); }
    int GetMaxNode(int ind) const {
        const Node *n = &node[ind-1];
        return n->right ? GetMaxNode(n->right) : ind;
    }

    virtual string DebugString(const string &name=string()) const {
        string ret = GraphVizHeader(StrCat("RedBlackTree", name.size()?"_":"", name));
        StrAppend(&ret, "node [color = black];\r\n"); PrintNodes(head, Black, &ret);
        StrAppend(&ret, "node [color = red];\r\n");   PrintNodes(head, Red,   &ret);
        PrintEdges(head, &ret);
        return ret + GraphVizFooter();
    }
    virtual void PrintNodes(int ind, int color, string *out) const {
        if (!ind) return;
        const Node *n = &node[ind-1];
        PrintNodes(n->left, color, out);
        if (n->color == color) StrAppend(out, "\"", n->key, "\";\r\n");
        PrintNodes(n->right, color, out);
    }
    virtual void PrintEdges(int ind, string *out) const {
        if (!ind) return;
        const Node *n = &node[ind-1], *l=n->left?&node[n->left-1]:0, *r=n->right?&node[n->right-1]:0;
        if (l) { PrintEdges(n->left, out);  StrAppend(out, "\"", n->key, "\" -> \"", l->key, "\" [ label = \"left\"  ];\r\n"); }
        if (r) { PrintEdges(n->right, out); StrAppend(out, "\"", n->key, "\" -> \"", r->key, "\" [ label = \"right\" ];\r\n"); }
    }

    static int GetSibling(const Node *parent, int ind) { return ind == parent->left ? parent->right : parent->left; }
};

template <class K, class V> struct PrefixSumKeyedRedBlackTreeNode : public RedBlackTreeNode<K,V> {
    PrefixSumKeyedRedBlackTreeNode(K k=K(), unsigned v=0) : RedBlackTreeNode<K,V>(k,v) {}
};

template <class K, class V, class Node = PrefixSumKeyedRedBlackTreeNode<K,V> >
struct PrefixSumKeyedRedBlackTree : public RedBlackTree<K,V,Node> {
};

TEST(DatastructureTest, StdMultiMap) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    int dup_val[] = { 5, 2, 1 }, sum_dup_val = 8, j;
    for (auto i : my_env->db) {
        auto &db = *i.second, &sorted_db = *i.third;
        std::multimap<int, int> m;
        std::multimap<int, int>::const_iterator mi;
        int ctid = timers->Create(StrCat("std::mmap ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("std::mmap ", i.first, " query1"));
        int itid = timers->Create(StrCat("std::mmap ", i.first, " iterU1")), iind=0;
        int dtid = timers->Create(StrCat("std::mmap ", i.first, " del   "));
        int rtid = timers->Create(StrCat("std::mmap ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("std::mmap ", i.first, " ins2  "));
        int Dtid = timers->Create(StrCat("std::mmap ", i.first, " del2  ")); 

        timers->AccumulateTo(ctid); for (auto i : db) for (j=0; j<3; j++) m.insert(pair<int,int>(i, i+dup_val[j]));
        timers->AccumulateTo(qtid); for (auto i : db) { mi=m.find(i); int sum=0; for (j=0; j<3; j++, mi++) { EXPECT_NE(m.end(), mi); if (mi!=m.end()) sum += mi->second; } EXPECT_EQ(i*3+sum_dup_val, sum); }
        timers->AccumulateTo(itid); for (auto i : m) {}
        timers->AccumulateTo(dtid); for (int i=0, l=db.size()/2; i<l; i++) EXPECT_EQ(3, m.erase(db[i]));
        timers->AccumulateTo(rtid);
        for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
            if (i < hl) { EXPECT_EQ(m.end(), m.find(db[i])); continue; }
            mi=m.find(db[i]); int sum=0; for (j=0; j<3; j++, mi++) { EXPECT_NE(m.end(), mi); if (mi!=m.end()) sum += mi->second; } EXPECT_EQ(db[i]*3+sum_dup_val, sum);
        }
        timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) for (j=0; j<3; j++) m.insert(pair<int,int>(db[i], db[i]+dup_val[j]));
        timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_EQ(3, m.erase(i));
        timers->AccumulateTo(0);
    }
}

TEST(DatastructureTest, StdUnorderedMap) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    for (auto i : my_env->db) {
        auto &db = *i.second, &sorted_db = *i.third;
        unordered_map<int, int> m;
        unordered_map<int, int>::const_iterator mi;
        int ctid = timers->Create(StrCat("std::umap ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("std::umap ", i.first, " query1"));
        int itid = timers->Create(StrCat("std::umap ", i.first, " iterU1")), iind=0;
        int dtid = timers->Create(StrCat("std::umap ", i.first, " del   "));
        int rtid = timers->Create(StrCat("std::umap ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("std::umap ", i.first, " ins2  "));
        int Dtid = timers->Create(StrCat("std::umap ", i.first, " del2  ")); 

        timers->AccumulateTo(ctid); for (auto i : db) m[i] = i;
        timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE(m.end(), (mi=m.find(i))); if (mi!=m.end()) EXPECT_EQ(i, mi->second); }
        timers->AccumulateTo(itid); for (auto i : m) {}
        timers->AccumulateTo(dtid); for (int i=0, l=db.size()/2; i<l; i++) EXPECT_EQ(1, m.erase(db[i]));
        timers->AccumulateTo(rtid);
        for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
            if (i < hl) EXPECT_EQ(m.end(), m.find(db[i]));
            else { EXPECT_NE(m.end(), (mi=m.find(db[i]))); if (mi!=m.end()) EXPECT_EQ(db[i], mi->second); }
        }
        timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) m[db[i]] = db[i];
        timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_EQ(1, m.erase(i));
        timers->AccumulateTo(0);
    }
}

TEST(DatastructureTest, StdMap) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    for (auto i : my_env->db) {
        auto &db = *i.second, &sorted_db = *i.third;
        map<int, int> m;
        map<int, int>::const_iterator mi;
        int ctid = timers->Create(StrCat("std::map  ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("std::map  ", i.first, " query1"));
        int itid = timers->Create(StrCat("std::map  ", i.first, " iterO1")), iind=0;
        int dtid = timers->Create(StrCat("std::map  ", i.first, " del   "));
        int rtid = timers->Create(StrCat("std::map  ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("std::map  ", i.first, " ins2  "));
        int Dtid = timers->Create(StrCat("std::map  ", i.first, " del2  ")); 

        timers->AccumulateTo(ctid); for (auto i : db) m[i] = i;
        timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE(m.end(), (mi=m.find(i))); if (mi!=m.end()) EXPECT_EQ(i, mi->second); }
        timers->AccumulateTo(itid); for (auto i : m)  { EXPECT_EQ(sorted_db[iind], i.first); EXPECT_EQ(sorted_db[iind], i.second); iind++; }
        timers->AccumulateTo(dtid); for (int i=0, l=db.size()/2; i<l; i++) EXPECT_EQ(1, m.erase(db[i]));
        timers->AccumulateTo(rtid);
        for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
            if (i < hl) EXPECT_EQ(m.end(), m.find(db[i]));
            else { EXPECT_NE(m.end(), (mi=m.find(db[i]))); if (mi!=m.end()) EXPECT_EQ(db[i], mi->second); }
        }
        timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) m[db[i]] = db[i];
        timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_EQ(1, m.erase(i));
        timers->AccumulateTo(0);
    }
}

#ifdef LFL_JUDY
TEST(DatastructureTest, JudyArray) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    for (auto i : my_env->db) {
        auto &db = *i.second, &sorted_db = *i.third;
        judymap<int, int> m;
        judymap<int, int>::const_iterator mi;
        int ctid = timers->Create(StrCat("JudyArray ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("JudyArray ", i.first, " query1"));
        int itid = timers->Create(StrCat("JudyArray ", i.first, " iterO1")), iind=0;
        int dtid = timers->Create(StrCat("JudyArray ", i.first, " del   "));
        int rtid = timers->Create(StrCat("JudyArray ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("JudyArray ", i.first, " ins2  "));

        timers->AccumulateTo(ctid); for (auto i : db) m[i] = i;
        timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE(m.end(), (mi=m.find(i))); if (mi!=m.end()) EXPECT_EQ(i, (int)(long)(*mi).second); }
        timers->AccumulateTo(itid); for (auto i : m)  { EXPECT_EQ(sorted_db[iind], i.first); EXPECT_EQ(sorted_db[iind], i.second); iind++; }
        timers->AccumulateTo(dtid); for (int i=0, l=db.size()/2; i<l; i++) EXPECT_EQ(1, m.erase(db[i]));
        timers->AccumulateTo(rtid);
        for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
            if (i < hl) EXPECT_EQ(m.end(), m.find(db[i]));
            else { EXPECT_NE(m.end(), (mi=m.find(db[i]))); if (mi!=m.end()) EXPECT_EQ(db[i], (*mi).second); }
        }
        timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) m[db[i]] = db[i];
        timers->AccumulateTo(0);
    }
}
#endif

TEST(DatastructureTest, SkipList) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    for (auto i : my_env->db) {
        auto &db = *i.second;
        SkipList<int, int> t;
        int ctid = timers->Create(StrCat("SkipList  ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("SkipList  ", i.first, " query1"));
        int dtid = timers->Create(StrCat("SkipList  ", i.first, " del   ")); 
        int rtid = timers->Create(StrCat("SkipList  ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("SkipList  ", i.first, " ins2  ")), *v; 
        int Dtid = timers->Create(StrCat("SkipList  ", i.first, " del2  ")); 

        timers->AccumulateTo(ctid); for (auto i : db) t.Insert(i, i);
        timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE((int*)0, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
        timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_EQ(true, t.Erase(db[i]));
        timers->AccumulateTo(rtid);
        for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
            if (i < hl) EXPECT_EQ(0, t.Find(db[i]));
            else { EXPECT_NE((int*)0, (v=t.Find(db[i]))); if (v) EXPECT_EQ(db[i], *v); }
        }
        timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) t.Insert(db[i], db[i]);
        timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_EQ(true, t.Erase(i));
        timers->AccumulateTo(0);
    }
}

TEST(DatastructureTest, AVLTree) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    for (auto i : my_env->db) {
        auto &db = *i.second;
        AVLTree<int, int> t;
        int ctid = timers->Create(StrCat("AVLTree   ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("AVLTree   ", i.first, " query1"));
        int dtid = timers->Create(StrCat("AVLTree   ", i.first, " del   ")); 
        int rtid = timers->Create(StrCat("AVLTree   ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("AVLTree   ", i.first, " ins2  ")), *v; 
        int Dtid = timers->Create(StrCat("AVLTree   ", i.first, " del2  ")); 

        timers->AccumulateTo(ctid); for (auto i : db) { EXPECT_NE((int*)0, t.Insert(i, i)); }
        timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE((int*)0, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
        timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_EQ(true, t.Erase(db[i]));
        timers->AccumulateTo(rtid);
        for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
            if (i < hl) EXPECT_EQ(0, t.Find(db[i]));
            else { EXPECT_NE((int*)0, (v=t.Find(db[i]))); if (v) EXPECT_EQ(db[i], *v); }
        }
        timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) t.Insert(db[i], db[i]);
        timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_EQ(true, t.Erase(i));
        timers->AccumulateTo(0);
    }
}

TEST(DatastructureTest, PrefixSumKeyedAVLTree) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    for (auto i : my_env->db) {
        auto &db = *i.second;
        if (i.first != "incr") continue;
        int ctid = timers->Create(StrCat("PSAVLTree ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("PSAVLTree ", i.first, " query1"));
        int dtid = timers->Create(StrCat("PSAVLTree ", i.first, " del   ")); 
        int rtid = timers->Create(StrCat("PSAVLTree ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("PSAVLTree ", i.first, " ins2  ")), *v; 
        int Rtid = timers->Create(StrCat("PSAVLTree ", i.first, " query3"));
        int Dtid = timers->Create(StrCat("PSAVLTree ", i.first, " del2  ")); 
        int Stid = timers->Create(StrCat("PSAVLTree ", i.first, " insS  ")); 

        {
            PrefixSumKeyedAVLTree<int, int> t;
            timers->AccumulateTo(ctid); for (auto i : db) { EXPECT_NE((int*)0, t.Insert(i, i)); }
            timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE((int*)0, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
            timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_EQ(true, t.Erase(0));
            timers->AccumulateTo(rtid); for (int i=0, hl=db.size()/2; i<hl; i++) { EXPECT_NE((int*)0, (v=t.Find(db[i]))); if (v) EXPECT_EQ(db[hl+i], *v); }
            timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) t.Insert(0, db[i]);
            timers->AccumulateTo(Rtid); for (auto i : db) { EXPECT_NE((int*)0, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
            timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_EQ(true, t.Erase(0));
            timers->AccumulateTo(0);
        }
        {
            PrefixSumKeyedAVLTree<int, int> t;
            timers->AccumulateTo(Stid); for (auto i : db) t.val.Insert(i); t.LoadFromSortedVal();
            timers->AccumulateTo(0);
            for (auto i : db) { EXPECT_NE((int*)0, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
        }
    }
}

TEST(DatastructureTest, RedBlackTree) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    for (auto i : my_env->db) {
        auto &db = *i.second;
        RedBlackTree<int, int> t;
        int ctid = timers->Create(StrCat("RBTree    ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("RBTree    ", i.first, " query1"));
        int dtid = timers->Create(StrCat("RBTree    ", i.first, " del   ")); 
        int rtid = timers->Create(StrCat("RBTree    ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("RBTree    ", i.first, " ins2  ")), *v; 
        int Dtid = timers->Create(StrCat("RBTree    ", i.first, " del2  ")); 
                                                    
        timers->AccumulateTo(ctid); for (auto i : db) t.Insert(i, i);
        timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE((int*)0, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
        timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_EQ(true, t.Erase(db[i]));
        timers->AccumulateTo(rtid);
        for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
            if (i < hl) EXPECT_EQ(0, t.Find(db[i]));
            else { EXPECT_NE((int*)0, (v=t.Find(db[i]))); if (v) EXPECT_EQ(db[i], *v); }
        }
        timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) t.Insert(db[i], db[i]);
        timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_EQ(true, t.Erase(i));
        timers->AccumulateTo(0);
    }
}
}; // namespace LFL
