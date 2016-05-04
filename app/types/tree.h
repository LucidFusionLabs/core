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

#ifndef LFL_CORE_APP_TYPES_TREE_H__
#define LFL_CORE_APP_TYPES_TREE_H__
namespace LFL {

template <class K, class V> struct RedBlackTreeNode {
  enum { Left, Right };
  typedef K Key;
  typedef V Value;
  K key;
  unsigned val:31, left, right, parent, color:1;
  RedBlackTreeNode(K k, unsigned v, unsigned P, bool C=0) : key(k), val(v), left(0), right(0), parent(P), color(C) {}
  void SwapKV(RedBlackTreeNode *n) { swap(key, n->key); unsigned v=val; val=n->val; n->val=v; }
  void ComputeAnnotationFromChildren(const RedBlackTreeNode*, const RedBlackTreeNode*) {}
};

template <class Node> struct RedBlackTreeZipper {
  RedBlackTreeZipper(bool update=0, bool by_index=0, bool end_key=0) {} 
  string             DebugString()                                              const { return ""; }
  int                GetIndex   (const Node &n)                                 const { return -1; }
  typename Node::Key GetKey     (const Node &n)                                 const { return n.key; }
  typename Node::Key GetBegKey  (const Node &n)                                 const { return n.key; }
  typename Node::Key GetEndKey  (const Node &n)                                 const { return n.key; }
  int                WalkLeft   (const Node &n)                                       { return n.left; }
  int                WalkRight  (const Node &n)                                       { return n.right; }
  int                UnwalkRight(const Node &n)                                       { return n.parent; }
  bool               LessThan   (const Node &n, int ind, const typename Node::Key &k) { return n.key < k; }
  bool               MoreThan   (const Node &n, int ind, const typename Node::Key &k) { return k < n.key; }
};

template <class K, class V, class Node = RedBlackTreeNode<K,V>, class Zipper = RedBlackTreeZipper<Node>, template <class T> class Storage = FreeListVector>
struct RedBlackTree {
  enum Color { Red, Black };
  struct Iterator { 
    RedBlackTree *tree; int ind; K key; V *val; Zipper zipper;
    Iterator(RedBlackTree *T=0, int I=0)        : tree(T), ind(I), key(), val(0) {                  if (ind) LoadKV(); }
    Iterator(RedBlackTree *T, int I, Zipper &z) : tree(T), ind(I), key(), val(0) { swap(z, zipper); if (ind) LoadKV(); }
    Iterator& operator--() { if (!ind) *this = tree->RBegin(); else if ((ind = tree->DecrementNode(ind, &zipper))) LoadKV(); else val = 0; return *this; }
    Iterator& operator++() { CHECK(ind);                            if ((ind = tree->IncrementNode(ind, &zipper))) LoadKV(); else val = 0; return *this; }
    bool operator!=(const Iterator &i) { return tree != i.tree || ind != i.ind; }
    void LoadKV() { if (const Node *n = &tree->node[ind-1]) { key = zipper.GetBegKey(*n); val = &tree->val[n->val]; } }
    int GetIndex() const { if (const Node *n = &tree->node[ind-1]) return zipper.GetIndex(*n); return -1; }
    int GetBegKey() const { if (const Node *n = &tree->node[ind-1]) return zipper.GetBegKey(*n); return -1; }
    int GetEndKey() const { if (const Node *n = &tree->node[ind-1]) return zipper.GetEndKey(*n); return -1; }
    int GetAnchor() const { return ind; }
  };
  struct ConstIterator { 
    const RedBlackTree *tree; int ind; K key; const V *val; Zipper zipper;
    ConstIterator(const RedBlackTree *T=0, int I=0)        : tree(T), ind(I), key(), val(0) {                  if (ind) LoadKV(); }
    ConstIterator(const RedBlackTree *T, int I, Zipper &z) : tree(T), ind(I), key(), val(0) { swap(z, zipper); if (ind) LoadKV(); }
    ConstIterator(const Iterator &i) : tree(i.tree), ind(i.ind), key(i.key), val(i.val), zipper(i.zipper) {}
    ConstIterator& operator-- () { if (!ind) *this = tree->RBegin(); else if ((ind = tree->DecrementNode(ind, &zipper))) LoadKV(); else val = 0; return *this; }
    ConstIterator& operator++ () { CHECK(ind);                            if ((ind = tree->IncrementNode(ind, &zipper))) LoadKV(); else val = 0; return *this; }
    bool operator!=(const ConstIterator &i) { return tree != i.tree || ind != i.ind; }
    void LoadKV() { if (const Node *n = &tree->node[ind-1]) { key = zipper.GetBegKey(*n); val = &tree->val[n->val]; } }
    int GetIndex() const { if (const Node *n = &tree->node[ind-1]) return zipper.GetIndex(*n); return -1; }
    int GetBegKey() const { if (const Node *n = &tree->node[ind-1]) return zipper.GetBegKey(*n); return -1; }
    int GetEndKey() const { if (const Node *n = &tree->node[ind-1]) return zipper.GetEndKey(*n); return -1; }
    int GetAnchor() const { return ind; }
  };
  struct Query {
    const K key; Zipper z;
    Query(const K &k, bool update=0, bool by_index=0, bool end_key=0) : key(k), z(update, by_index, end_key) {}
  };

  Storage<Node> node;
  Storage<V>    val;
  int head=0, count=0;

  int size() const { return count; }
  void Clear() { head=count=0; node.Clear(); val.Clear(); }
  /**/ Iterator  Begin()       { Zipper z; int n = head ? GetMinNode(head)     : 0; return      Iterator(this, n, z); }
  ConstIterator  Begin() const { Zipper z; int n = head ? GetMinNode(head)     : 0; return ConstIterator(this, n, z); }
  /**/ Iterator RBegin()       { Zipper z; int n = head ? GetMaxNode(head, &z) : 0; return      Iterator(this, n, z); }
  ConstIterator RBegin() const { Zipper z; int n = head ? GetMaxNode(head, &z) : 0; return ConstIterator(this, n, z); }

  K  GetAnchorKey(int id) const { return node[id-1].key; }
  V *GetAnchorVal(int id) { return &val[node[id-1].val]; }
  virtual      Iterator GetAnchorIter(int id)       { Zipper z; return      Iterator(this, id, z); }
  virtual ConstIterator GetAnchorIter(int id) const { Zipper z; return ConstIterator(this, id, z); }

  bool          Erase (const K &k)             { Query q(k, 1); return EraseNode(&q); }
  /**/ Iterator Insert(const K &k, V &&v)      { Query q(k, 1); int n=InsertNode(&q, forward<V>(v)); return Iterator     (this, n, q.z); }
  ConstIterator Find       (const K &k) const  { Query q(k);    int n=FindNode      (&q);            return ConstIterator(this, n, q.z); }
  /**/ Iterator Find       (const K &k)        { Query q(k);    int n=FindNode      (&q);            return Iterator     (this, n, q.z); }
  ConstIterator LowerBound (const K &k) const  { Query q(k);    int n=LowerBoundNode(&q);            return ConstIterator(this, n, q.z); }
  /**/ Iterator LowerBound (const K &k)        { Query q(k);    int n=LowerBoundNode(&q);            return Iterator     (this, n, q.z); }
  ConstIterator LesserBound(const K &k) const  { Query q(k);    int n=LowerBoundNode(&q);            ConstIterator i(this, n, q.z); if (i.val && k < i.key) --i; return i; }
  /**/ Iterator LesserBound(const K &k)        { Query q(k);    int n=LowerBoundNode(&q);            Iterator      i(this, n, q.z); if (i.val && k < i.key) --i; return i; }
  ConstIterator SecondBound(const K &k) const  { Query q(k,0,0,1); int n=LowerBoundNode(&q);         return ConstIterator(this, n, q.z); }
  /**/ Iterator SecondBound(const K &k)        { Query q(k,0,0,1); int n=LowerBoundNode(&q);         return Iterator     (this, n, q.z); }

  ConstIterator FindIndex(const K &k) const { Query q(k,0,1); int n=FindNode(&q); return ConstIterator(this, n, q.z); }
  /**/ Iterator FindIndex(const K &k)       { Query q(k,0,1); int n=FindNode(&q); return Iterator     (this, n, q.z); }

  virtual K GetCreateNodeKey(const Query *q, const V*) const { return q->key; }
  virtual void ComputeAnnotationFromChildrenOnPath(Query *q) {}
  virtual int ResolveInsertCollision(int ind, Query *q, V &&v) { 
    Node *n = &node[ind-1];
    if (bool update_on_dup_insert = 1) val[n->val] = move(v);
    return ind;
  }

  int FindNode(Query *q) const {
    for (int ind=head; ind;) {
      const Node &n = node[ind-1];
      if      (q->z.MoreThan(n, ind, q->key)) ind = q->z.WalkLeft (n);
      else if (q->z.LessThan(n, ind, q->key)) ind = q->z.WalkRight(n);
      else return ind;
    } return 0;
  }

  int LowerBoundNode(Query *q) const {
    int ind = head;
    while (ind) {
      const Node &n = node[ind-1];
      if      (q->z.MoreThan(n, ind, q->key)) { if (!n.left) break; ind = q->z.WalkLeft(n); }
      else if (q->z.LessThan(n, ind, q->key)) {
        if (!n.right) { ind = IncrementNode(ind, &q->z); break; }
        ind = q->z.WalkRight(n);
      } else break;
    }
    return ind;
  }

  int WalkBackwardsToRoot(int in, Zipper *z) const {
    for (int ind = in; ind && ind != head; /**/) {
      int p_ind = GetParent(ind);
      const Node &p = node[p_ind-1];
      if      (ind == p.left)  { ind = p_ind; }
      else if (ind == p.right) { ind = p_ind; z->WalkRight(p); }
      else FATAL("corrupt");
    }
    return in;
  }

  int IncrementNode(int ind, Zipper *z) const {
    const Node *n = &node[ind-1], *p;
    if (n->right) for (n = &node[(ind = z->WalkRight(*n))-1]; n->left; n = &node[(ind = n->left)-1]) {}
    else {
      int p_ind = GetParent(ind);
      while (p_ind && ind == (p = &node[p_ind-1])->right) { ind=p_ind; p_ind=z->UnwalkRight(*p); }
      ind = p_ind;
    }
    return ind;
  }

  int DecrementNode(int ind, Zipper *z) const {
    const Node *n = &node[ind-1], *p;
    if (n->left) for (n = &node[(ind = n->left)-1]; n->right; n = &node[(ind = z->WalkRight(*n))-1]) {}
    else {
      int p_ind = GetParent(ind);
      while (p_ind && ind == (p = &node[p_ind-1])->left) { ind=p_ind; p_ind=p->parent; }
      if ((ind = p_ind)) z->UnwalkRight(*p); 
    }
    return ind;
  }

  int InsertNode(Query *q, V &&v) {
    Node *new_node;
    int new_ind = node.Insert(Node(GetCreateNodeKey(q, &v), 0, 0))+1;
    if (int ind = head) {
      for (;;) {
        Node *n = &node[ind-1];
        if      (q->z.MoreThan(*n, ind, q->key)) { if (!n->left)  { n->left  = new_ind; break; } ind = q->z.WalkLeft (*n); }
        else if (q->z.LessThan(*n, ind, q->key)) { if (!n->right) { n->right = new_ind; break; } ind = q->z.WalkRight(*n); }
        else { node.Erase(new_ind-1); return ResolveInsertCollision(ind, q, forward<V>(v)); }
      }
      (new_node = &node[new_ind-1])->parent = ind;
    } else new_node = &node[(head = new_ind)-1];
    new_node->val = val.Insert(forward<V>(v));
    ComputeAnnotationFromChildrenOnPath(q);
    InsertBalance(new_ind);
    count++;
    return new_ind;
  }

  void InsertBalance(int ind) {
    int p_ind, gp_ind, u_ind;
    Node *n = &node[ind-1], *p, *gp, *u;
    if (!(p_ind = n->parent)) { n->color = Black; return; }
    if ((p = &node[p_ind-1])->color == Black || !(gp_ind = p->parent)) return;
    gp = &node[gp_ind-1];
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

  bool EraseNode(Query *q) {
    Node *n;
    int ind = head;
    while (ind) {
      n = &node[ind-1];
      if      (q->z.MoreThan(*n, ind, q->key)) ind = q->z.WalkLeft (*n);
      else if (q->z.LessThan(*n, ind, q->key)) ind = q->z.WalkRight(*n);
      else break;
    }
    if (!ind) return false;
    int orig_ind = ind, max_path = 0;
    val.Erase(n->val);
    if (n->left && n->right) {
      Zipper z;
      n->SwapKV(&node[(ind = GetMaxNode((max_path = n->left), &z))-1]);
      n = &node[ind-1];
    }
    bool balance = n->color == Black;
    int child = X_or_Y(n->left, n->right), parent = n->parent;
    ReplaceNode(ind, child);
    node.Erase(ind-1);
    if (child == head && head) node[head-1].color = Black;
    if (max_path) ComputeAnnotationFromChildrenOnMaxPath(max_path);
    if (max_path) ComputeAnnotationFromChildren(&node[orig_ind-1]);
    ComputeAnnotationFromChildrenOnPath(q);
    if (balance) EraseBalance(child, parent);
    count--;
    return true;
  }

  void EraseBalance(int ind, int p_ind) {
    if (!p_ind) return;
    Node *p = &node[p_ind-1];
    int s_ind = GetSibling(p, ind);
    if (!s_ind) return;
    Node *s = &node[s_ind-1];
    if (s->color == Red) {
      p->color = Red;
      s->color = Black;
      if (ind == p->left) RotateLeft (p_ind);
      else                RotateRight(p_ind);
      s = &node[(s_ind = GetSibling(p, ind))-1];
    }
    if (s->color == Black) {
      bool slr = GetColor(s->left)  == Red, slb = !slr, lc = ind == p->left;
      bool srr = GetColor(s->right) == Red, srb = !srr, rc = ind == p->right;
      if (p->color == Black && slb && srb) { s->color = Red; return EraseBalance(p_ind, GetParent(p_ind)); }
      if (p->color == Red   && slb && srb) { s->color = Red; p->color = Black; return; }
      if      (lc && slr && srb) { s->color = Red; SetColor(s->left , Black); RotateRight(s_ind); }
      else if (rc && slb && srr) { s->color = Red; SetColor(s->right, Black); RotateLeft (s_ind); }
      s = &node[(s_ind = GetSibling(p, ind))-1];
    }
    s->color = p->color;
    p->color = Black;
    if (ind == p->left) { SetColor(s->right, Black); RotateLeft (p_ind); }
    else                { SetColor(s->left,  Black); RotateRight(p_ind); }
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

  void RotateLeft(int ind) {
    int right_ind;
    Node *n = &node[ind-1], *o = &node[(right_ind = n->right)-1];
    ReplaceNode(ind, right_ind);
    n->right = o->left;
    if (o->left) node[o->left-1].parent = ind;
    o->left = ind;
    n->parent = right_ind;
    ComputeAnnotationFromChildren(n);
    ComputeAnnotationFromChildren(o);
  }

  void RotateRight(int ind) {
    int left_ind;
    Node *n = &node[ind-1], *o = &node[(left_ind = n->left)-1];
    ReplaceNode(ind, left_ind);
    n->left = o->right;
    if (o->right) node[o->right-1].parent = ind;
    o->right = ind;
    n->parent = left_ind;
    ComputeAnnotationFromChildren(n);
    ComputeAnnotationFromChildren(o);
  }

  void ComputeAnnotationFromChildren(Node *n) {
    n->ComputeAnnotationFromChildren(n->left ? &node[n->left-1] : 0, n->right ? &node[n->right-1] : 0);
  }

  void ComputeAnnotationFromChildrenOnMaxPath(int ind) {
    Node *n = &node[ind-1];
    if (n->right) ComputeAnnotationFromChildrenOnMaxPath(n->right);
    ComputeAnnotationFromChildren(n);
  }

  void SetColor(int ind, int color) { node[ind-1].color = color; }
  int GetColor  (int ind) const { return ind ? node[ind-1].color : Black; }
  int GetParent (int ind) const { return node[ind-1].parent; }
  int GetSibling(int ind) const { return GetSibling(&node[GetParent(ind)-1], ind); }
  int GetMinNode(int ind) const {
    const Node *n = &node[ind-1];
    return n->left ? GetMinNode(n->left) : ind;
  }

  int GetMaxNode(int ind, Zipper *z) const {
    const Node &n = node[ind-1];
    return n.right ? GetMaxNode(z->WalkRight(n), z) : ind;
  }

  struct LoadFromSortedArraysQuery { const K *k; V *v; int max_height; };
  virtual void LoadFromSortedArrays(const K *k, V *v, int n) {
    CHECK_EQ(0, node.size());
    LoadFromSortedArraysQuery q = { k, v, WhichLog2(NextPowerOfTwo(n, true)) };
    count = n;
    head = BuildTreeFromSortedArrays(&q, 0, n-1, 1);
  }

  virtual int BuildTreeFromSortedArrays(LoadFromSortedArraysQuery *q, int beg_ind, int end_ind, int h) {
    if (end_ind < beg_ind) return 0;
    CHECK_LE(h, q->max_height);
    int mid_ind = (beg_ind + end_ind) / 2, color = ((h>1 && h == q->max_height) ? Red : Black);
    int node_ind = node.Insert(Node(q->k[mid_ind], val.Insert(move(q->v[mid_ind])), 0, color))+1;
    int left_ind  = BuildTreeFromSortedArrays(q, beg_ind,   mid_ind-1, h+1);
    int right_ind = BuildTreeFromSortedArrays(q, mid_ind+1, end_ind,   h+1);
    Node *n = &node[node_ind-1];
    n->left  = left_ind;
    n->right = right_ind;
    if (left_ind)  node[ left_ind-1].parent = node_ind; 
    if (right_ind) node[right_ind-1].parent = node_ind; 
    ComputeAnnotationFromChildren(&node[node_ind-1]);
    return node_ind;
  }

  void CheckProperties() const {
    CHECK_EQ(Black, GetColor(head));
    CheckNoAdjacentRedNodes(head);
    int same_black_length = -1;
    CheckEveryPathToLeafHasSameBlackLength(head, 0, &same_black_length);
  }

  void CheckNoAdjacentRedNodes(int ind) const {
    if (!ind) return;
    const Node *n = &node[ind-1];
    CheckNoAdjacentRedNodes(n->left);
    CheckNoAdjacentRedNodes(n->right);
    if (n->color == Black) return;
    CHECK_EQ(Black, GetColor(n->parent));
    CHECK_EQ(Black, GetColor(n->parent));
    CHECK_EQ(Black, GetColor(n->parent));
  }

  void CheckEveryPathToLeafHasSameBlackLength(int ind, int black_length, int *same_black_length) const {
    if (!ind) {
      if (*same_black_length < 0) *same_black_length = black_length;
      CHECK_EQ(*same_black_length, black_length);
      return;
    }
    const Node *n = &node[ind-1];
    if (n->color == Black) black_length++;
    CheckEveryPathToLeafHasSameBlackLength(n->left,  black_length, same_black_length);
    CheckEveryPathToLeafHasSameBlackLength(n->right, black_length, same_black_length);
  }

  virtual string DebugString(const string &name=string()) const {
    string ret = GraphVizFile::DigraphHeader(StrCat("RedBlackTree", name.size()?"_":"", name));
    ret += GraphVizFile::NodeColor("black"); PrintNodes(head, Black, &ret);
    ret += GraphVizFile::NodeColor("red");   PrintNodes(head, Red,   &ret);
    PrintEdges(head, &ret);
    return ret + GraphVizFile::Footer();
  }

  virtual void PrintNodes(int ind, int color, string *out) const {
    if (!ind) return;
    const Node *n = &node[ind-1];
    PrintNodes(n->left, color, out);
    if (n->color == color) GraphVizFile::AppendNode(out, StrCat(n->key));
    PrintNodes(n->right, color, out);
  }

  virtual void PrintEdges(int ind, string *out) const {
    if (!ind) return;
    const Node *n = &node[ind-1], *l=n->left?&node[n->left-1]:0, *r=n->right?&node[n->right-1]:0;
    if (l) { PrintEdges(n->left, out);  GraphVizFile::AppendEdge(out, StrCat(n->key), StrCat(l->key), "left" ); }
    if (r) { PrintEdges(n->right, out); GraphVizFile::AppendEdge(out, StrCat(n->key), StrCat(r->key), "right"); }
  }

  static int GetSibling(const Node *parent, int ind) { return ind == parent->left ? parent->right : parent->left; }
};

template <class K, class V, class Node, class Finger>
struct RedBlackFingerTree : public RedBlackTree<K, V, Node, Finger> {
  typedef RedBlackTree<K, V, Node, Finger> Parent;
  function<K     (const V*)> node_value_cb;
  function<string(const V*)> node_print_cb;
  RedBlackFingerTree() : 
    node_value_cb([](const V *v){ return 1;          }), 
    node_print_cb([](const V *v){ return StrCat(*v); }) {}

  virtual K GetCreateNodeKey(const typename Parent::Query *q, const V *v) const { return node_value_cb(v); }
  virtual int ResolveInsertCollision(int ind, typename Parent::Query *q, V &&v) { 
    int val_ind = Parent::val.Insert(forward<V>(v)), p_ind = ind;
    int new_ind = Parent::node.Insert(Node(GetCreateNodeKey(q, &v), val_ind, 0))+1;
    Node *n = &Parent::node[ind-1], *nn = &Parent::node[new_ind-1];
    if (!n->left) n->left = new_ind;
    else { Finger z; Parent::node[(p_ind = Parent::GetMaxNode(n->left, &z))-1].right = new_ind; }
    nn->parent = p_ind;
    Parent::ComputeAnnotationFromChildrenOnMaxPath(n->left);
    Parent::ComputeAnnotationFromChildren(n);
    ComputeAnnotationFromChildrenOnPath(q);
    Parent::InsertBalance(new_ind);
    Parent::count++;
    return new_ind;
  }

  virtual void ComputeAnnotationFromChildrenOnPath(typename Parent::Query *q) {
    for (auto i = q->z.path.rbegin(), e = q->z.path.rend(); i != e; ++i) 
      Parent::ComputeAnnotationFromChildren(&Parent::node[i->first-1]);
  }

  void LoadFromSortedVal() {
    int n = Parent::val.size();
    CHECK_EQ(0, Parent::node.size());
    Parent::count = Parent::val.size();
    Parent::head = BuildTreeFromSortedVal(0, n-1, 1, WhichLog2(NextPowerOfTwo(n, true)));
  }

  int BuildTreeFromSortedVal(int beg_val_ind, int end_val_ind, int h, int max_h) {
    if (end_val_ind < beg_val_ind) return 0;
    CHECK_LE(h, max_h);
    int mid_val_ind = (beg_val_ind + end_val_ind) / 2, color = ((h>1 && h == max_h) ? Parent::Red : Parent::Black);
    int ind = Parent::node.Insert(Node(node_value_cb(&Parent::val[mid_val_ind]), mid_val_ind, 0, color))+1;
    int left_ind  = BuildTreeFromSortedVal(beg_val_ind,   mid_val_ind-1, h+1, max_h);
    int right_ind = BuildTreeFromSortedVal(mid_val_ind+1, end_val_ind,   h+1, max_h);
    Node *n = &Parent::node[ind-1];
    n->left  = left_ind;
    n->right = right_ind;
    if (left_ind)  Parent::node[ left_ind-1].parent = ind;
    if (right_ind) Parent::node[right_ind-1].parent = ind;
    Parent::ComputeAnnotationFromChildren(&Parent::node[ind-1]);
    return ind;
  }

  V *Update(const K &k, const V &v) {
    typename Parent::Query q(k, 1);
    Node *n;
    for (int ind = Parent::head; ind; ) {
      n = &Parent::node[ind-1];
      if      (q.z.MoreThan(*n, ind, q.key)) ind = q.z.WalkLeft (*n);
      else if (q.z.LessThan(*n, ind, q.key)) ind = q.z.WalkRight(*n);
      else {
        n->key = node_value_cb(&v);
        ComputeAnnotationFromChildrenOnPath(&q);
        return &Parent::val[n->val];
      }
    } return 0;
  }

  virtual void PrintEdges(int ind, string *out) const {
    if (!ind) return;
    const Node *n = &Parent::node[ind-1], *l=n->left?&Parent::node[n->left-1]:0, *r=n->right?&Parent::node[n->right-1]:0;
    string v = node_print_cb(&Parent::val[n->val]), lv = l?node_print_cb(&Parent::val[l->val]):"", rv = r?node_print_cb(&Parent::val[r->val
                                                                                                                        ]):"";
    if (l) { PrintEdges(n->left,  out); StrAppend(out, "\"", v, "\" -> \"", lv, "\" [ label = \"left\"  ];\r\n"); }
    if (r) { PrintEdges(n->right, out); StrAppend(out, "\"", v, "\" -> \"", rv, "\" [ label = \"right\" ];\r\n"); }
  }
};

// RedBlackIntervalTree 

template <class K, class V> struct RedBlackIntervalTreeNode {
  enum { Left, Right };
  typedef pair<K, K> Key;
  typedef V          Value;
  Key key;
  K left_max=0, right_max=0, left_min=0, right_min=0;
  unsigned val:31, left, right, parent, color:1;
  RedBlackIntervalTreeNode(pair<K,K> k, unsigned v, unsigned P, bool C=0)
    : key(k), val(v), left(0), right(0), parent(P), color(C) {}

  void SwapKV(RedBlackIntervalTreeNode *n) { swap(key, n->key); unsigned v=val; val=n->val; n->val=v; }
  void ComputeAnnotationFromChildren(const RedBlackIntervalTreeNode *lc, const RedBlackIntervalTreeNode *rc) {
    left_max  = lc ? max(lc->key.second, max(lc->left_max, lc->right_max)) : 0;
    right_max = rc ? max(rc->key.second, max(rc->left_max, rc->right_max)) : 0;
    left_min  = lc ? ComputeMinFromChild(lc) : 0;
    right_min = lc ? ComputeMinFromChild(rc) : 0;
  }

  static K ComputeMinFromChild(const RedBlackIntervalTreeNode *n) {
    K ret = n->key.first;
    if (n->left_min)  ret = min(ret, n->left_min);
    if (n->right_min) ret = min(ret, n->right_min);
    return ret;
  }
};

template <class Node> struct RedBlackIntervalTreeFinger {
  typename Node::Key query;
  vector<pair<int, int> > path;
  RedBlackIntervalTreeFinger(bool update=0, bool by_index=0, bool end_key=0) {}
  RedBlackIntervalTreeFinger(const typename Node::Key::first_type &q, int head) : query(q,q) { path.reserve(64); path.emplace_back(head, 0); }
  string             DebugString()                                              const { return ""; }
  int                GetIndex   (const Node &n)                                 const { return -1; }
  typename Node::Key GetKey     (const Node &n)                                 const { return n.key; }
  typename Node::Key GetBegKey  (const Node &n)                                 const { return n.key; }
  typename Node::Key GetEndKey  (const Node &n)                                 const { return n.key; }
  int                WalkLeft   (const Node &n)                                       { return n.left; }
  int                WalkRight  (const Node &n)                                       { return n.right; }
  int                UnwalkRight(const Node &n)                                       { return n.parent; }
  bool               LessThan   (const Node &n, int ind, const typename Node::Key &k) { return n.key.first < k.first; }
  bool               MoreThan   (const Node &n, int ind, const typename Node::Key &k) { return k.first < n.key.first; }
};

template <class K, class V, class Node = RedBlackIntervalTreeNode<K, V>, class Finger = RedBlackIntervalTreeFinger<Node> >
struct RedBlackIntervalTree : public RedBlackTree<pair<K, K>, V, Node, Finger> {
  typedef RedBlackTree<pair<K, K>, V, Node, Finger> Parent;

  const V *IntersectOne(const K &k) const { int n=IntersectOneNode(k); return n ? &Parent::val[Parent::data[n-1].val] : 0; }
  V *IntersectOne(const K &k)       { int n=IntersectOneNode(k); return n ? &Parent::val[Parent::data[n-1].val] : 0; }

  typename Parent::ConstIterator IntersectAll(const K &k) const { Finger z(k, Parent::head); int n=IntersectAllNode(&z); return typename Parent::ConstIterator(this, n, z); }
  typename Parent::Iterator      IntersectAll(const K &k)       { Finger z(k, Parent::head); int n=IntersectAllNode(&z); return typename Parent::Iterator     (this, n, z); }
  void IntersectAllNext(typename Parent::ConstIterator *i) const { if ((i->ind = IntersectAllNode(&i->zipper))) i->LoadKV(); else i->val=0; }
  void IntersectAllNext(typename Parent::Iterator      *i)       { if ((i->ind = IntersectAllNode(&i->zipper))) i->LoadKV(); else i->val=0; }

  int IntersectOneNode(const K &q) const {
    const Node *n;
    for (int ind = Parent::head; ind; /**/) {
      n = &Parent::node[ind-1];
      if (n->key.first <= q && q < n->key.second) return ind;
      ind = (n->left && q <= n->left_max) ? n->left : n->right;
    } return 0;
  }

  int IntersectAllNode(Finger *z) const {
    const Node *n;
    while (z->path.size()) {
      auto &i = z->path.back();
      n = &Parent::node[i.first-1];
      if (i.second == 0 && ++i.second) if (Intersect(z->query, n->key)) return i.first;
      if (i.second == 1 && ++i.second && n->left  && z->query.first  <= n->left_max)  { z->path.emplace_back(n->left,  0); continue; }
      if (i.second == 2 && ++i.second && n->right && z->query.second >= n->right_min) { z->path.emplace_back(n->right, 0); continue; }
      z->path.pop_back();
    } return 0;
  }

  static bool Intersect(const pair<K,K> &q, const pair<K,K> &p) {
    return q.first <= p.second && p.first <= q.second;
  }
};

// Prefix-Sum-Keyed Red-Black Tree

template <class K, class V> struct PrefixSumKeyedRedBlackTreeNode {
  enum { Left, Right };
  typedef K Key;
  typedef V Value;
  K key, left_sum=0, right_sum=0, left_count=0, right_count=0;
  unsigned val:31, left, right, parent, color:1;
  PrefixSumKeyedRedBlackTreeNode(K k, unsigned v, unsigned P, bool C=0) : key(k), val(v), left(0), right(0), parent(P), color(C) {}
  void SwapKV(PrefixSumKeyedRedBlackTreeNode *n) { swap(key, n->key); unsigned v=val; val=n->val; n->val=v; }
  void ComputeAnnotationFromChildren(const PrefixSumKeyedRedBlackTreeNode *lc, const PrefixSumKeyedRedBlackTreeNode *rc) {
    left_sum    = lc ? (lc->left_sum   + lc->right_sum   + lc->key) : 0;
    right_sum   = rc ? (rc->left_sum   + rc->right_sum   + rc->key) : 0;
    left_count  = lc ? (lc->left_count + lc->right_count + 1)       : 0;
    right_count = rc ? (rc->left_count + rc->right_count + 1)       : 0;
  }
};

template <class Node> struct PrefixSumKeyedRedBlackTreeFinger {
  bool update=0, by_index=0, end_key=0;
  typename Node::Key sum=0, count=0;
  vector<pair<int, bool>> path;
  PrefixSumKeyedRedBlackTreeFinger(bool U=0, bool I=0, bool E=0) :
    update(U), by_index(I), end_key(E) { if (update) path.reserve(64); }

  string DebugString() const { 
    string p;
    for (auto i : path) StrAppend(&p, p.size()?", ":"", i.first);
    return StrCat("PrefixSumFinger: sum=", sum, ", update=", update, ", path={", p, "}");
  }

  int                GetIndex   (const Node &n) const { return count + n.left_count; }
  typename Node::Key GetKey     (const Node &n) const { return by_index ? GetIndex(n) : (end_key ? GetEndKey(n) : GetBegKey(n)); }
  typename Node::Key GetBegKey  (const Node &n) const { return sum + n.left_sum; }
  typename Node::Key GetEndKey  (const Node &n) const { return sum + n.left_sum + n.key; }
  int                WalkLeft   (const Node &n)       { return n.left; }
  int                WalkRight  (const Node &n)       { sum += (n.left_sum + n.key); count += (n.left_count + 1); return n.right; }
  int                UnwalkRight(const Node &n)       { sum -= (n.left_sum + n.key); count -= (n.left_count + 1); return n.parent; }

  bool LessThan(const Node &n, int ind, const typename Node::Key &k) {
    if (!(GetKey(n) < k)) return 0;
    if (update) path.emplace_back(ind, Node::Left);
    return 1;
  }

  bool MoreThan(const Node &n, int ind, const typename Node::Key &k) {
    if (!(k < GetKey(n))) return 0;
    if (update) path.emplace_back(ind, Node::Right);
    return 1;
  }
};

template <class K, class V, class Node = PrefixSumKeyedRedBlackTreeNode<K, V>, class Finger = PrefixSumKeyedRedBlackTreeFinger<Node> >
struct PrefixSumKeyedRedBlackTree : public RedBlackFingerTree<K, V, Node, Finger> {
  typedef RedBlackFingerTree<K, V, Node, Finger> Parent;

  typename Parent::     Iterator GetAnchorIter(int id)       { typename Parent::     Iterator it(this); it.ind=Parent::WalkBackwardsToRoot(id, &it.zipper); it.LoadKV(); return it; }
  typename Parent::ConstIterator GetAnchorIter(int id) const { typename Parent::ConstIterator it(this); it.ind=Parent::WalkBackwardsToRoot(id, &it.zipper); it.LoadKV(); return it; }

  int Total() const {
    auto i = Parent::RBegin();
    if (i.ind) i.zipper.WalkRight(Parent::node[i.ind-1]);
    return i.zipper.sum;
  }

  virtual void PrintNodes(int ind, int color, string *out) const {
    if (!ind) return;
    const Node *n = &Parent::node[ind-1];
    string v = Parent::node_print_cb(&Parent::val[n->val]);
    PrintNodes(n->left, color, out);
    if (n->color == color) StrAppend(out, "node [label = \"", v, " v:", n->key, "\nlsum:", n->left_sum,
                                     " rsum:", n->right_sum, "\"];\r\n\"", v, "\";\r\n");
    PrintNodes(n->right, color, out);
  }
};

}; // namespace LFL
#endif // LFL_CORE_APP_TYPES_TREE_H__
