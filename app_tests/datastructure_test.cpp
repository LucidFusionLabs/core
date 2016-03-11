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

#include "gtest/gtest.h"
#include "core/app/app.h"

extern "C" void MyAppCreate() {
  LFL::FLAGS_default_font = LFL::FakeFontEngine::Filename();
  LFL::app = new LFL::Application();
  LFL::screen = new LFL::Window();
}

extern "C" int MyAppMain(int argc, const char* const* argv) {
  testing::InitGoogleTest(&argc, const_cast<char**>(argv));
  if (!LFL::app) MyAppCreate();
  CHECK_EQ(0, LFL::app->Create(argc, argv, __FILE__));
  return RUN_ALL_TESTS();
}

namespace LFL {
DEFINE_int(size,         1024*1024, "Test size"); 
DEFINE_int(rand_key_min, 0,         "Min key");
DEFINE_int(rand_key_max, 10000000,  "Max key");

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

MyEnvironment* const my_env = dynamic_cast<MyEnvironment*>(::testing::AddGlobalTestEnvironment(new MyEnvironment));

template <class K, class V> struct SkipList {
  struct Node {
    K key; int val_ind, next_ind, leveldown_ind;
    Node(K k=K(), int VI=-1, int NI=-1, int LDI=-1) : key(k), val_ind(VI), next_ind(NI), leveldown_ind(LDI) {}
  };
  FreeListVector<V> val;
  FreeListVector<Node> node;
  int count=0, head_ind=-1;

  void Insert(const K &k, V &&v) {
    vector<pair<int, int> > zipper;
    zipper.reserve(64);
    int ind = FindNode(k, &zipper), next_ind;
    if (ind < 0 || (next_ind = node[ind].next_ind) < 0) {}
    else if (const Node *n = &node[next_ind]) if (k == n->key) { val[n->val_ind] = v; return; }
    int val_ind = val.Insert(forward<V>(v)), level = RandomLevel();
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

  static int RandomLevel() { static float logP = log(.5); return 1 + int(log(Rand(0.0, 1.0)) / logP); }
};

template <class K, class V> struct AVLTreeNode {
  enum { Left, Right };
  typedef K Key;
  typedef K Value;
  K key; unsigned val:30, left:30, right:30, height:6;
  AVLTreeNode(K k, unsigned v) : key(k), val(v), left(0), right(0), height(1) {}
  void SetChildren(unsigned l, unsigned r) { left=l; right=r; }
  void SwapKV(AVLTreeNode *n) { swap(key, n->key); unsigned v=val; val=n->val; n->val=v; }
  virtual void ComputeAnnotationFromChildren(const AVLTreeNode<K,V> *lc, const AVLTreeNode<K,V> *rc) {
    height = max(lc ? lc->height : 0, rc ? rc->height : 0) + 1;
  }
};

template <class Node> struct AVLTreeZipper {
  virtual bool LessThan(const Node &n, const typename Node::Key &k) { return n.key < k; }
  virtual bool MoreThan(const Node &n, const typename Node::Key &k) { return k < n.key; }
};

template <class K, class V, class Node = AVLTreeNode<K,V>, class Zipper = AVLTreeZipper<Node> > struct AVLTree {
  struct Query {
    const K key; V *ret; Zipper z;
    Query(const K &k) : key(k), ret(0) {}
  };

  FreeListVector<Node> node;
  FreeListVector<V> val;
  bool update_on_dup_insert=1;
  int head=0;

  const V* Find  (const K &k) const  { Query q=GetQuery(k); int vind = FindNode  (head, &q); return vind ? &val[vind-1] : 0; }
  V*       Find  (const K &k)        { Query q=GetQuery(k); int vind = FindNode  (head, &q); return vind ? &val[vind-1] : 0; }
  bool     Erase (const K &k)        { Query q=GetQuery(k);     head = EraseNode (head, &q); return q.ret; }
  V*       Insert(const K &k, V &&v) { Query q=GetQuery(k, &v); head = InsertNode(head, &q, forward<V>(v)); return q.ret; }

  virtual Query GetQuery(const K &k, const V *v=0) { return Query(k); }
  virtual K GetCreateNodeKey(const Query *q) const { return q->key; }
  virtual int ResolveInsertCollision(int ind, Query *q, V &&v) { 
    Node *n = &node[ind-1];
    if (update_on_dup_insert) q->ret = &(val[n->val] = forward<V>(v));
    return ind;
  }

  int FindNode(int ind, Query *q) const {
    if (!ind) return 0;
    const Node &n = node[ind-1];
    if      (q->z.MoreThan(n, q->key)) return FindNode(n.left,  q);
    else if (q->z.LessThan(n, q->key)) return FindNode(n.right, q);
    else return n.val+1;
  }

  int InsertNode(int ind, Query *q, V &&v) {
    if (!ind) return CreateNode(q, forward<V>(v));
    const Node &n = node[ind-1];
    if      (q->z.MoreThan(n, q->key)) { int li = InsertNode(n.left,  q, forward<V>(v)); node[ind-1].left  = li; }
    else if (q->z.LessThan(n, q->key)) { int ri = InsertNode(n.right, q, forward<V>(v)); node[ind-1].right = ri; }
    else return ResolveInsertCollision(ind, q, forward<V>(v));
    return Balance(ind);
  }

  int CreateNode(Query *q, V &&v) {
    int val_ind = val.Insert(forward<V>(v));
    q->ret = &val[val_ind];
    return node.Insert(Node(GetCreateNodeKey(q), val_ind))+1;
  }

  int EraseNode(int ind, Query *q) {
    if (!ind) return 0;
    const Node &n = node[ind-1];
    if      (q->z.MoreThan(n, q->key)) { int li = EraseNode(n.left,  q); node[ind-1].left  = li; }
    else if (q->z.LessThan(n, q->key)) { int ri = EraseNode(n.right, q); node[ind-1].right = ri; }
    else {
      q->ret = &val[n.val];
      int left = n.left, right = n.right;
      val.Erase(n.val);
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
    ComputeAnnotationFromChildren(n);
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
    ComputeAnnotationFromChildren(n);
    ComputeAnnotationFromChildren(o);
    return right_ind;
  }

  int RotateRight(int ind) {
    int left_ind;
    Node *n = &node[ind-1], *o = &node[(left_ind = n->left)-1];
    n->left = o->right;
    o->right = ind;
    ComputeAnnotationFromChildren(n);
    ComputeAnnotationFromChildren(o);
    return left_ind;
  }

  void ComputeAnnotationFromChildren(Node *n) {
    n->ComputeAnnotationFromChildren(n->left ? &node[n->left -1] : 0, n->right ? &node[n->right-1] : 0);
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
    string ret = GraphVizFile::DigraphHeader(StrCat("AVLTree", name.size()?"_":"", name));
    PrintNodes(head, &ret);
    PrintEdges(head, &ret);
    return ret + GraphVizFile::Footer();
  }

  virtual void PrintNodes(int ind, string *out) const {
    if (!ind) return;
    const Node *n = &node[ind-1];
    PrintNodes(n->left, out);
    GraphVizFile::AppendNode(out, StrCat(n->key), StrCat(n->key, " ", n->val));
    PrintNodes(n->right, out);
  }

  virtual void PrintEdges(int ind, string *out) const {
    if (!ind) return;
    const Node *n = &node[ind-1], *l=n->left?&node[n->left-1]:0, *r=n->right?&node[n->right-1]:0;
    const V    *v = &val[n->val], *lv = l?&val[l->val]:0, *rv = r?&val[r->val]:0;
    if (l) { PrintEdges(n->left,  out); StrAppend(out, "\"", *v, "\" -> \"", *lv, "\" [ label = \"left\"  ];\r\n"); }
    if (r) { PrintEdges(n->right, out); StrAppend(out, "\"", *v, "\" -> \"", *rv, "\" [ label = \"right\" ];\r\n"); }
  }
};

template <class K, class V, class Node, class Finger>
struct AVLFingerTree : public AVLTree<K,V,Node,Finger> {
  typedef AVLTree<K,V,Node,Finger> Parent;

  virtual K GetCreateNodeKey(const typename Parent::Query *q) const { return q->z.GetValue(*q->ret); }
  virtual typename Parent::Query GetQuery(const K &k, const V *v=0) { return typename Parent::Query(k + (v ? 0 : 1)); }
  virtual int ResolveInsertCollision(int ind, typename Parent::Query *q, V &&v) { 
    int li = ResolveInsertCollision(Parent::node[ind-1].left, ind, q, forward<V>(v));
    Parent::node[ind-1].left = li;
    return Parent::Balance(ind);
  }

  virtual int ResolveInsertCollision(int ind, int dup_ind, typename Parent::Query *q, V &&v) {
    if (!ind) {
      int new_ind = Parent::CreateNode(q, forward<V>(v));
      Parent::node[new_ind-1].SwapKV(&Parent::node[dup_ind-1]);
      return new_ind;
    }
    int ri = ResolveInsertCollision(Parent::node[ind-1].right, dup_ind, q, forward<V>(v));
    Parent::node[ind-1].right = ri;
    return Parent::Balance(ind);
  }

  void LoadFromSortedVal() {
    Finger z;
    CHECK_EQ(0, Parent::node.size());
    Parent::head = BuildTreeFromSortedVal(z, 0, Parent::val.size()-1);
  }

  int BuildTreeFromSortedVal(const Finger &z, int beg_val_ind, int end_val_ind) {
    if (end_val_ind < beg_val_ind) return 0;
    int mid_val_ind = (beg_val_ind + end_val_ind) / 2;
    int ind = Parent::node.Insert(Node(z.GetValue(Parent::val[mid_val_ind]), mid_val_ind))+1;
    int li = BuildTreeFromSortedVal(z, beg_val_ind,   mid_val_ind-1);
    int ri = BuildTreeFromSortedVal(z, mid_val_ind+1, end_val_ind);
    Parent::node[ind-1].left  = li;
    Parent::node[ind-1].right = ri;
    Parent::ComputeAnnotationFromChildren(&Parent::node[ind-1]);
    return ind;
  }
};

template <class K, class V> struct PrefixSumKeyedAVLTreeNode : public AVLTreeNode<K,V> {
  typedef K Key;
  typedef V Value;
  typedef PrefixSumKeyedAVLTreeNode<K,V> Self;
  int left_sum=0, right_sum=0;
  PrefixSumKeyedAVLTreeNode(K k=K(), unsigned v=0) : AVLTreeNode<K,V>(k,v) {}
  virtual void ComputeAnnotationFromChildren(const Self *lc, const Self *rc) {
    AVLTreeNode<K,V>::ComputeAnnotationFromChildren(lc, rc);
    left_sum  = lc ? (lc->left_sum + lc->right_sum + lc->key) : 0;
    right_sum = rc ? (rc->left_sum + rc->right_sum + rc->key) : 0;
  }
};

template <class Node> struct PrefixSumKeyedAVLTreeZipper {
  int sum=0;
  virtual typename Node::Key GetValue(const Node &n) const {
    return 1;
  }

  virtual bool LessThan(const Node &n, const typename Node::Key &k) {
    if (!((sum + n.left_sum + n.key) < k)) return 0;
    sum += n.left_sum + n.key;
    return 1;
  }

  virtual bool MoreThan(const Node &n, const typename Node::Key &k) {
    if (!(k < (sum + n.left_sum + n.key))) return 0;
    return 1;
  }
};

template <class K, class V, class Node = PrefixSumKeyedAVLTreeNode<K,V>, class Finger = PrefixSumKeyedAVLTreeZipper<Node> >
struct PrefixSumKeyedAVLTree : public AVLFingerTree<K,V,Node,Finger> {
  typedef AVLFingerTree<K,V,Node,Finger> Parent;
  virtual void PrintNodes(int ind, string *out) const {
    if (!ind) return;
    const Node *n = &Parent::node[ind-1];
    const V    *v = &Parent::val[n->val];
    PrintNodes(n->left, out);
    StrAppend(out, "node [label = \"", *v, " v:", n->key, "\nlsum:", n->left_sum, " rsum:", n->right_sum, "\"];\r\n\"", *v, "\";\r\n");
    PrintNodes(n->right, out);
  }
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

    timers->AccumulateTo(ctid); for (auto i : db) t.Insert(i, move(i));
    timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE(nullptr, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
    timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_TRUE(t.Erase(db[i]));
    timers->AccumulateTo(rtid);
    for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
      if (i < hl) EXPECT_EQ(0, t.Find(db[i]));
      else { EXPECT_NE(nullptr, (v=t.Find(db[i]))); if (v) EXPECT_EQ(db[i], *v); }
    }
    timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) t.Insert(db[i], move(db[i]));
    timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_TRUE(t.Erase(i));
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

    timers->AccumulateTo(ctid); for (auto i : db) { EXPECT_NE(nullptr, t.Insert(i, move(i))); }
    timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE(nullptr, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
    timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_TRUE(t.Erase(db[i]));
    timers->AccumulateTo(rtid);
    for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
      if (i < hl) EXPECT_EQ(0, t.Find(db[i]));
      else { EXPECT_NE(nullptr, (v=t.Find(db[i]))); if (v) EXPECT_EQ(db[i], *v); }
    }
    timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) t.Insert(db[i], move(db[i]));
    timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_TRUE(t.Erase(i));
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
      timers->AccumulateTo(ctid); for (auto i : db) { EXPECT_NE(nullptr, t.Insert(i, move(i))); }
      timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE(nullptr, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
      timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_TRUE(t.Erase(0));
      timers->AccumulateTo(rtid); for (int i=0, hl=db.size()/2; i<hl; i++) { EXPECT_NE(nullptr, (v=t.Find(db[i]))); if (v) EXPECT_EQ(db[hl+i], *v); }
      timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) t.Insert(0, move(db[i]));
      timers->AccumulateTo(Rtid); for (auto i : db) { EXPECT_NE(nullptr, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
      timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_TRUE(t.Erase(0));
      timers->AccumulateTo(0);
    }
    {
      PrefixSumKeyedAVLTree<int, int> t;
      timers->AccumulateTo(Stid); for (auto i : db) t.val.Insert(move(i)); t.LoadFromSortedVal();
      timers->AccumulateTo(0);
      for (auto i : db) { EXPECT_NE(nullptr, (v=t.Find(i))); if (v) EXPECT_EQ(i, *v); }
    }
  }
}

TEST(DatastructureTest, RedBlackTree) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
  for (auto i : my_env->db) {
    auto &db = *i.second, &sorted_db = *i.third;
    RedBlackTree<int, int> t;
    RedBlackTree<int, int>::Iterator ti;
    int ctid = timers->Create(StrCat("RBTree    ", i.first, " ins1  "));
    int qtid = timers->Create(StrCat("RBTree    ", i.first, " query1"));
    int itid = timers->Create(StrCat("RBTree    ", i.first, " iterO1")), iind=0;
    int dtid = timers->Create(StrCat("RBTree    ", i.first, " del   ")); 
    int rtid = timers->Create(StrCat("RBTree    ", i.first, " query2"));
    int Ctid = timers->Create(StrCat("RBTree    ", i.first, " ins2  ")), *v; 
    int Dtid = timers->Create(StrCat("RBTree    ", i.first, " del2  ")); 
    int Stid = timers->Create(StrCat("RBTree    ", i.first, " insHS ")); 
    int Rtid = timers->Create(StrCat("RBTree    ", i.first, " queryS"));

    {                                                
      timers->AccumulateTo(ctid); for (auto i : db) t.Insert(i, move(i));
      timers->AccumulateTo(0);    t.CheckProperties(); CHECK_EQ(0, (--t.Begin()).val);
      timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE(nullptr, (ti=t.Find(i)).val); if (ti.val) EXPECT_EQ(i, *ti.val); }
      timers->AccumulateTo(itid); for (ti = t. Begin(); ti.ind; ++ti) {         EXPECT_EQ(sorted_db[iind], ti.key); EXPECT_EQ(sorted_db[iind], *ti.val); iind++; }
      timers->AccumulateTo(0);    for (ti = t.RBegin(); ti.ind; --ti) { iind--; EXPECT_EQ(sorted_db[iind], ti.key); EXPECT_EQ(sorted_db[iind], *ti.val);         }
      timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_TRUE(t.Erase(db[i]));
      timers->AccumulateTo(0);    t.CheckProperties();
      timers->AccumulateTo(rtid);
      for (int i=0, l=db.size(), hl=l/2; i<l; i++) {
        if (i < hl) EXPECT_EQ(nullptr, t.Find(db[i]).val);
        else { EXPECT_NE(nullptr, (ti=t.Find(db[i])).val); if (ti.val) EXPECT_EQ(db[i], *ti.val); }
      }
      timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) t.Insert(db[i], move(db[i]));
      timers->AccumulateTo(0);    t.CheckProperties();
      timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_TRUE(t.Erase(i));
      timers->AccumulateTo(0);    t.CheckProperties();
    }
    {
      RedBlackTree<int, int> t;
      timers->AccumulateTo(Stid);
      t.LoadFromSortedArrays(&sorted_db[0], &sorted_db[0], sorted_db.size()/2);
      for (int l=db.size(), i=l/2; i<l; i++) t.Insert(sorted_db[i], move(sorted_db[i]));
      timers->AccumulateTo(Rtid); for (auto i : db) { EXPECT_NE(nullptr, (ti=t.Find(i)).val); if (ti.val) EXPECT_EQ(i, *ti.val); }
      timers->AccumulateTo(0);    t.CheckProperties();
    }
  }
}

TEST(DatastructureTest, PrefixSumKeyedRedBlackTree) {
  PerformanceTimers *timers = Singleton<PerformanceTimers>::Get();
  for (auto i : my_env->db) {
    auto &db = *i.second;
    if (i.first != "incr") continue;
    int ctid = timers->Create(StrCat("PSRBTree  ", i.first, " ins1  "));
    int qtid = timers->Create(StrCat("PSRBTree  ", i.first, " query1"));
    int itid = timers->Create(StrCat("PSRBTree  ", i.first, " iterO1")), iind=0;
    int dtid = timers->Create(StrCat("PSRBTree  ", i.first, " del   ")); 
    int rtid = timers->Create(StrCat("PSRBTree  ", i.first, " query2"));
    int Ctid = timers->Create(StrCat("PSRBTree  ", i.first, " ins2  ")), *v; 
    int Rtid = timers->Create(StrCat("PSRBTree  ", i.first, " query3"));
    int Dtid = timers->Create(StrCat("PSRBTree  ", i.first, " del2  ")); 
    int Stid = timers->Create(StrCat("PSRBTree  ", i.first, " insS  ")); 

    {
      PrefixSumKeyedRedBlackTree<int, int> t;
      PrefixSumKeyedRedBlackTree<int, int>::Iterator ti;
      timers->AccumulateTo(ctid); for (auto i : db) { EXPECT_NE(nullptr, t.Insert(i, move(i)).val); }
      timers->AccumulateTo(0);    t.CheckProperties();
      timers->AccumulateTo(qtid); for (auto i : db) { EXPECT_NE(nullptr, (ti=t.      Find(i)).val); if (ti.val) EXPECT_EQ(i, *ti.val); }
      timers->AccumulateTo(0);    for (auto i : db) { EXPECT_NE(nullptr, (ti=t.LowerBound(i)).val); if (ti.val) EXPECT_EQ(i, *ti.val); }
      timers->AccumulateTo(itid); for (ti = t. Begin(); ti.ind; ++ti) {         EXPECT_EQ(db[iind], ti.key); EXPECT_EQ(db[iind], *ti.val); iind++; }
      timers->AccumulateTo(0);    for (ti = t.RBegin(); ti.ind; --ti) { iind--; EXPECT_EQ(db[iind], ti.key); EXPECT_EQ(db[iind], *ti.val);         }
      timers->AccumulateTo(dtid); for (int i=0, hl=db.size()/2; i<hl; i++) EXPECT_TRUE(t.Erase(0));
      timers->AccumulateTo(0);    t.CheckProperties();
      timers->AccumulateTo(rtid); for (int i=0, hl=db.size()/2; i<hl; i++) { EXPECT_NE(nullptr, (ti=t.Find(db[i])).val); if (ti.val) EXPECT_EQ(db[hl+i], *ti.val); }
      timers->AccumulateTo(Ctid); for (int i=db.size()/2-1; i>=0; i--) t.Insert(0, move(db[i]));
      timers->AccumulateTo(0);    t.CheckProperties();
      timers->AccumulateTo(Rtid); for (auto i : db) { EXPECT_NE(nullptr, (ti=t.Find(i)).val); if (ti.val) EXPECT_EQ(i, *ti.val); }
      timers->AccumulateTo(0);    for (auto i : db) { EXPECT_NE(nullptr, (ti=t.Find(i)).val); if (ti.val) EXPECT_EQ(i, *ti.val); }
      timers->AccumulateTo(Dtid); for (auto i : db) EXPECT_TRUE(t.Erase(0));
      timers->AccumulateTo(0);    t.CheckProperties();
      timers->AccumulateTo(0);
    }
    {
      PrefixSumKeyedRedBlackTree<int, int> t;
      PrefixSumKeyedRedBlackTree<int, int>::Iterator ti;
      timers->AccumulateTo(Stid); for (auto i : db) t.val.Insert(move(i)); t.LoadFromSortedVal();
      timers->AccumulateTo(0);    t.CheckProperties();
      for (auto i : db) { EXPECT_NE(nullptr, (ti=t.Find(i)).val); if (ti.val) EXPECT_EQ(i, *ti.val); }
      t.node_value_cb = bind([&]() { return 10; });
      t.Update(0,   0); for (int i=1; i<=10; i++) { EXPECT_NE(nullptr, (ti=t.LowerBound(    i)).val); if (ti.val) EXPECT_EQ(db[1], *ti.val); }
      t.Update(9+4, 0); for (int i=1; i<=10; i++) { EXPECT_NE(nullptr, (ti=t.LowerBound(    i)).val); if (ti.val) EXPECT_EQ(db[1], *ti.val); }
      /**/              for (int i=1; i<=10; i++) { EXPECT_NE(nullptr, (ti=t.LowerBound(9+4+i)).val); if (ti.val) EXPECT_EQ(db[5], *ti.val); }
      for (auto i : db) { EXPECT_NE(nullptr, (ti=t.      Find(i+(i>0?9:0)+(i>4?9:0))).val); if (ti.val) EXPECT_EQ(i, *ti.val); }
      for (auto i : db) { EXPECT_NE(nullptr, (ti=t.LowerBound(i+(i>0?9:0)+(i>4?9:0))).val); if (ti.val) EXPECT_EQ(i, *ti.val); }
      iind=0; for (ti = t. Begin(); ti.ind; ++ti) {         EXPECT_EQ(db[iind]+(iind>0?9:0)+(iind>4?9:0), ti.key); EXPECT_EQ(db[iind], *ti.val); iind++; }
      /**/    for (ti = t.RBegin(); ti.ind; --ti) { iind--; EXPECT_EQ(db[iind]+(iind>0?9:0)+(iind>4?9:0), ti.key); EXPECT_EQ(db[iind], *ti.val);         }
    }
  }
}

TEST(DatastructureTest, RedBlackIntervalTree) {
  RedBlackIntervalTree<int,int> t;
  RedBlackIntervalTree<int,int>::ConstIterator i;
  t.Insert(pair<int,int>(3, 10), 0);
  t.Insert(pair<int,int>(4, 11), 1);
  t.Insert(pair<int,int>(5, 9),  2);

  i = t.IntersectAll(2);  EXPECT_EQ(nullptr, i.val);

  i = t.IntersectAll(3);  EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(0, *i.val);
  t.IntersectAllNext(&i); EXPECT_EQ(nullptr, i.val);

  i = t.IntersectAll(4);  EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(1, *i.val);
  t.IntersectAllNext(&i); EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(0, *i.val);
  t.IntersectAllNext(&i); EXPECT_EQ(nullptr, i.val);

  i = t.IntersectAll(5);  EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(1, *i.val);
  t.IntersectAllNext(&i); EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(0, *i.val);
  t.IntersectAllNext(&i); EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(2, *i.val);
  t.IntersectAllNext(&i); EXPECT_EQ(nullptr, i.val);

  i = t.IntersectAll(9);  EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(1, *i.val);
  t.IntersectAllNext(&i); EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(0, *i.val);
  t.IntersectAllNext(&i); EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(2, *i.val);
  t.IntersectAllNext(&i); EXPECT_EQ(nullptr, i.val);

  i = t.IntersectAll(10); EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(1, *i.val);
  t.IntersectAllNext(&i); EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(0, *i.val);
  t.IntersectAllNext(&i); EXPECT_EQ(nullptr, i.val);

  i = t.IntersectAll(11); EXPECT_NE(nullptr, i.val); if (i.val) EXPECT_EQ(1, *i.val);
  t.IntersectAllNext(&i); EXPECT_EQ(nullptr, i.val);

  i = t.IntersectAll(12); EXPECT_EQ(nullptr, i.val);
}

}; // namespace LFL
