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

void PrintGraphVizFooter() { printf("}\r\n"); }
void PrintGraphVizHeader(const char *name) {
    printf("digraph %s {\r\n"
           "rankdir=LR;\r\n"
           "size=\"8,5\"\r\n"
           "node [style = solid];\r\n"
           "node [shape = circle];\r\n", name);
}

struct MyEnvironment : public ::testing::Environment {
    vector<int> rand_db, incr_db, decr_db;
    vector<pair<string, vector<int>*> > db;

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

        db.emplace_back("rand", &rand_db);
        db.emplace_back("incr", &incr_db);
        db.emplace_back("decr", &decr_db);
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

template <class K, class V> struct AVLTreeNode {
    K key; unsigned val:30, left:30, right:30, height:6;
    V sum=V();
    AVLTreeNode(K k=K(), unsigned v=0) : key(k), val(v), left(0), right(0), height(1) {}
    void SetChildren(unsigned l, unsigned r) { left=l; right=r; }
};

template <class K, class V, class Node = AVLTreeNode<K,V> > struct AVLTree {
    struct Query {
        const K &key; const V *val; V *ret;
        Query(const K &k, const V *v=0) : key(k), val(v), ret(0) {}
    };

    FreeListVector<Node> node;
    FreeListVector<V> val;
    int head=0;

    V*   Find  (const K &k)             { Query q(k);     return Find  (head, &q); }
    V*   Insert(const K &k, const V &v) { Query q(k, &v); head = Insert(head, &q); return q.ret; }
    bool Erase (const K &k)             { Query q(k);     head = Erase (head, &q); return q.ret; }
    void Print() const { PrintGraphVizHeader("AVLTree"); Print(head); PrintGraphVizFooter(); }

    V *Find(int ind, Query *q) {
        if (!ind) return 0;
        Node *n = &node[ind-1];
        if      (q->key < n->key) return Find(n->left,  q);
        else if (n->key < q->key) return Find(n->right, q);
        else return &val[n->val];
    }
    int Insert(int ind, Query *q) {
        if (!ind) return Create(q);
        Node *n = &node[ind-1];
        if      (q->key < n->key) node[ind-1].left  = Insert(n->left,  q);
        else if (n->key < q->key) node[ind-1].right = Insert(n->right, q);
        else { q->ret = &(val[n->val] = *q->val); return ind; }
        return Balance(ind);
    }
    int Create(Query *q) {
        int val_ind = val.Insert(*q->val);
        q->ret = &val[val_ind];
        return node.Insert(Node(q->key, val_ind))+1;
    }
    int Erase(int ind, Query *q) {
        if (!ind) return 0;
        Node *n = &node[ind-1];
        if      (q->key < n->key) n->left  = Erase(n->left,  q);
        else if (n->key < q->key) n->right = Erase(n->right, q);
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
        n->height = GetHeight(n);
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
        n->height = GetHeight(n);
        o->height = GetHeight(o);
        return right_ind;
    }
    int RotateRight(int ind) {
        int left_ind;
        Node *n = &node[ind-1], *o = &node[(left_ind = n->left)-1];
        n->left = o->right;
        o->right = ind;
        n->height = GetHeight(n);
        o->height = GetHeight(o);
        return left_ind;
    }
    int GetMinNode(int ind) const {
        const Node *n = &node[ind-1];
        return n->left ? GetMinNode(n->left) : ind;
    }
    int GetHeight(Node *n) const {
        return max(n->left  ? node[n->left -1].height : 0,
                   n->right ? node[n->right-1].height : 0) + 1;
    }
    int GetBalance(int ind) const {
        const Node *n = &node[ind-1];
        return (n->right ? node[n->right-1].height : 0) -
               (n->left  ? node[n->left -1].height : 0);
    }
    void Print(int ind) const {
        if (!ind) return;
        const Node *n = &node[ind-1], *l=n->left?&node[n->left-1]:0, *r=n->right?&node[n->right-1]:0;
        if (l) { Print(n->left);  printf("\"%d\" -> \"%d\" [ label = \"left\"  ];\r\n", n->key, l->key); }
        if (r) { Print(n->right); printf("\"%d\" -> \"%d\" [ label = \"right\" ];\r\n", n->key, r->key); }
    }
};

template <class K, class V> struct PrefixSumAVLTreeNode : public AVLTreeNode<K,V> {
    PrefixSumAVLTreeNode(K k=K(), unsigned v=0) : AVLTreeNode<K,V>(k,v) {}
};

template <class K, class V, class Node = PrefixSumAVLTreeNode<K,V> >
struct PrefixSumAVLTree : public AVLTree<K,V,Node> {
};

TEST(DatastructureTest, StdMap) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    for (auto i : my_env->db) {
        auto &db = *i.second;
        map<int, int> m;
        map<int, int>::const_iterator mi;
        int ctid = timers->Create(StrCat("std::map ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("std::map ", i.first, " query1"));
        int dtid = timers->Create(StrCat("std::map ", i.first, " del   "));
        int rtid = timers->Create(StrCat("std::map ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("std::map ", i.first, " ins2  "));

        timers->AccumulateTo(ctid); for (auto i : db) m[i] = i;
        timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE(m.end(), (mi=m.find(i))); if (mi!=m.end()) EXPECT_EQ(i, mi->second); }
        timers->AccumulateTo(dtid); for (int i=0, l=db.size()/2; i<l; i++) EXPECT_EQ(1, m.erase(db[i]));
        timers->AccumulateTo(rtid);
        for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
            if (i < hl) EXPECT_EQ(m.end(), m.find(db[i]));
            else { EXPECT_NE(m.end(), (mi=m.find(db[i]))); if (mi!=m.end()) EXPECT_EQ(db[i], mi->second); }
        }
        timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) m[db[i]] = db[i];
        timers->AccumulateTo(0);
    }
}

TEST(DatastructureTest, PrefixSumAVLTree) {
    PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
    for (auto i : my_env->db) {
        auto &db = *i.second;
        PrefixSumAVLTree<int, int> at;
        int ctid = timers->Create(StrCat("AVLTree  ", i.first, " ins1  "));
        int qtid = timers->Create(StrCat("AVLTree  ", i.first, " query1"));
        int dtid = timers->Create(StrCat("AVLTree  ", i.first, " del   ")); 
        int rtid = timers->Create(StrCat("AVLTree  ", i.first, " query2"));
        int Ctid = timers->Create(StrCat("AVLTree  ", i.first, " ins2  ")), *v; 

        timers->AccumulateTo(ctid); for (auto i : db) at.Insert(i, i);
        timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE((int*)0, (v=at.Find(i))); if (v) EXPECT_EQ(i, *v); }
        timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_EQ(true, at.Erase(db[i]));
        timers->AccumulateTo(rtid);
        for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
            if (i < hl) EXPECT_EQ(0, at.Find(db[i]));
            else { EXPECT_NE((int*)0, (v=at.Find(db[i]))); if (v) EXPECT_EQ(db[i], *v); }
        }
        timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) at.Insert(db[i], db[i]);
        timers->AccumulateTo(0);
    }
}
}; // namespace LFL
