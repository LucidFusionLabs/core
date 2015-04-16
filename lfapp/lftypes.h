/*
 * $Id: lftypes.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef __LFL_LFAPP_LFTYPES_H__
#define __LFL_LFAPP_LFTYPES_H__

#ifdef LFL_JUDY
#include "judymap.h"
#endif

#define SortImpl1(x1, y2) return x1 < y2;
#define SortImpl2(x1, y1, x2, y2)  \
    if      (x1 < y1) return true;  \
    else if (x1 > y1) return false; \
    else return x2 < y2;
#define SortImpl3(x1, y1, x2, y2, x3, y3) \
    if      (x1 < y1) return true;  \
    else if (x1 > y1) return false; \
    if      (x2 < y2) return true;  \
    else if (x2 > y2) return false; \
    else return x3 < y3;
#define SortImpl4(x1, y1, x2, y2, x3, y3, x4, y4) \
    if      (x1 < y1) return true;  \
    else if (x1 > y1) return false; \
    if      (x2 < y2) return true;  \
    else if (x2 > y2) return false; \
    if      (x3 < y3) return true;  \
    else if (x3 > y3) return false; \
    else return x4 < y4;

namespace LFL {
struct typed_ptr {
    int type; void *value;
    typed_ptr() : type(0), value(0) {}
    typed_ptr(int T, void *P) : type(T), value(P) {}
};

template <class X> int TypeId()   { static int ret = fnv32(typeid(X).name()); return ret; }
template <class X> int TypeId(X*) { static int ret = fnv32(typeid(X).name()); return ret; }
template <class X> typed_ptr TypePointer(X* v) { return typed_ptr(TypeId<X>(), v); }

struct RefCounter {
    int count=0;
    virtual void AddRef() { count++; }
    virtual void DelRef() { count--; }
};

struct RefSet {
    unordered_set<RefCounter*> refs;
    virtual ~RefSet() { for (auto i : refs) i->DelRef(); }
    void Insert(RefCounter *x) { auto i = refs.insert(x); if (i.second) x->AddRef(); }
};

template <class X> static void Replace(X** p, X* r) { delete (*p); (*p) = r; }
template <class X> static void AllocReplace(X** p, X* r) { if (*p && (*p)->alloc) (*p)->alloc->free(*p); *p = r; }

template <class X> typename X::iterator Insert(X &m, const typename X::key_type &k, const typename X::mapped_type &v) {
    return m.insert(typename X::value_type(k, v)).first;
}

template <typename K, typename V> typename map<K, V>::iterator FindOrInsert(map<K, V> &m, K k, V v, int *inserted) {
    auto ret = m.insert(typename map<K, V>::value_type(k, v));
    if (inserted) (*inserted)++;
    return ret.first;
}
template <typename K, typename V> typename map<K, V>::iterator FindOrInsert(map<K, V> &m, K k, V v, bool *inserted=0) {
    auto ret = m.insert(typename map<K, V>::value_type(k, v));
    if (inserted) *inserted = ret.second;
    return ret.first;
}
template <typename K, typename V> typename map<K, V*>::iterator FindOrInsert(map<K, V*> &m, K k, V* v, bool *inserted=0) {
    auto ret = m.insert(typename map<K, V*>::value_type(k, v));
    if (inserted) *inserted = ret.second;
    return ret.first;
}
template <typename K, typename V> typename map<K, V>::iterator FindOrInsert(map<K, V> &m, K k, bool *inserted=0) {
    auto i = m.find(k);
    if (i != m.end()) return i;
    if (inserted) *inserted = 1;
    auto ret = m.insert(typename map<K, V>::value_type(k, V()));
    return ret.first;
}
template <typename K, typename V> typename map<K, V*>::iterator FindOrInsert(map<K, V*> &m, K k, bool *inserted=0) {
    auto ret = m.insert(typename map<K, V*>::value_type(k, 0));
    if (ret.second) ret.first->second = new V();
    if (inserted) *inserted = ret.second;
    return ret.first;
}

template <typename K, typename V> typename unordered_map<K, V>::iterator FindOrInsert(unordered_map<K, V> &m, K k, bool *inserted=0) {
    auto i = m.find(k);
    if (i != m.end()) return i;
    if (inserted) *inserted = 1;
    auto ret = m.insert(typename unordered_map<K, V>::value_type(k, V()));
    return ret.first;
}
template <typename K, typename V> typename unordered_map<K, V>::iterator FindOrInsert(unordered_map<K, V> &m, K k, V v, bool *inserted) {
    auto ret = m.insert(typename unordered_map<K, V>::value_type(k, v));
    if (inserted) *inserted = ret.second;
    return ret.first;
}

template <typename X> typename X::mapped_type FindOrNull(const X &m, const typename X::key_type &k) {
    typename X::const_iterator iter = m.find(k);
    return (iter != m.end()) ? iter->second : 0;
}

template <typename X> typename X::mapped_type FindOrDefault(const X &m, const typename X::key_type &k, const typename X::mapped_type &v) {
    typename X::const_iterator iter = m.find(k);
    return (iter != m.end()) ? iter->second : v;
}

template <typename X> typename X::mapped_type FindOrDie(const char *file, int line, const X &m, const typename X::key_type &k) {
    typename X::const_iterator iter = m.find(k);
    if (iter == m.end()) Log(LFApp::Log::Fatal, file, line, StrCat("FindOrDie(", k, ")"));
    return iter->second;
}
#define FindOrDie(m, k) FindOrDie(__FILE__, __LINE__, m, k)

template <class I, class T> I LesserBound(I first, I last, const T& v, bool strict=false) {
    I i = lower_bound(first, last, v);
    if (i == last || i == first) return last;
    return (!strict && v < *i) ? i : --i;
}

template <typename RI> typename RI::iterator_type ForwardIteratorFromReverse(RI ri) { return (++ri).base(); }

template <class X> basic_string<X> Substr(const basic_string<X> &in, size_t o=0, size_t l=basic_string<X>::npos) {
    return o < in.size() ? in.substr(o, l) : basic_string<X>();
}

template <typename X> bool Contains(const X &c, const typename X::key_type &k) { return c.find(k) != c.end(); }
template <typename X> void EnsureSize(X &x, int n) { if (x.size() < n) x.resize(n); }

template <typename X> typename X::value_type &PushFront(X &v, const typename X::value_type &x) { v.push_front(x); return v.front(); }
template <typename X> typename X::value_type &PushBack (X &v, const typename X::value_type &x) { v.push_back (x); return v.back (); }
template <typename X> typename X::value_type  PopBack  (X &v) { typename X::value_type ret = v.back (); v.pop_back (); return ret; }
template <typename X> typename X::value_type  PopFront (X &v) { typename X::value_type ret = v.front(); v.pop_front(); return ret; }
template <typename X> typename std::queue<X>::value_type PopFront(std::queue<X> &v) { typename std::queue<X>::value_type ret = v.front(); v.pop(); return ret; }

template <class X>       X *VectorGet(      vector<X> &x, int n) { return (n >= 0 && n < x.size()) ? &x[n] : 0; }
template <class X> const X *VectorGet(const vector<X> &x, int n) { return (n >= 0 && n < x.size()) ? &x[n] : 0; }
template <class X> X *VectorEnsureElement(vector<X> &x, int n) { EnsureSize(x, n+1); return &x[n]; }
template <class X> X *VectorCheckElement(vector<X> &x, int n) { CHECK_LT(n, x.size()); return &x[n]; }
template <typename X, class Y> void VectorAppend(vector<X> &out, const Y& begin, const Y& end) {
    out.insert(out.end(), begin, end);
}

template <class X> void VectorClear(vector<X > *v) { v->clear(); }
template <class X> void VectorClear(vector<X*> *v) {
    for (typename vector<X*>::const_iterator i = v->begin(); i != v->end(); ++i) delete (*i);
    v->clear();
}
template <class X> void VectorErase(X* v, int ind) {
    CHECK_RANGE(ind, 0, v->size());
    v->erase(v->begin()+ind, v->begin()+ind+1);
}
template <class X, class I> I VectorEraseIterSwapBack(X* v, I i) {
    bool last = (i+1) == v->end();
    if (!last) swap(*i, v->back());
    v->pop_back();
    return last ? v->end() : i;
}
template <class X> void VectorEraseSwapBack(X* v, int ind) {
    CHECK_RANGE(ind, 0, v->size());
    VectorEraseIterSwapBack(v, v->begin()+ind);
}
template <class X> int VectorEraseByValue(vector<X> *v, const X& x) {
    int orig_size = v->size();
    v->erase(LFL_STL_NAMESPACE::remove(v->begin(), v->end(), x), v->end());
    return orig_size - v->size();
}

template <class X> X BackOrDefault (const vector<X> &a)                    { return a.size() ? a.back () : X(); }
template <class X> X FrontOrDefault(const vector<X> &a)                    { return a.size() ? a.front() : X(); }
template <class X> X IndexOrDefault(const vector<X> &a, int n)             { return n < a. size() ?   a [n] : X(); }
template <class X> X IndexOrDefault(      vector<X> *a, int n)             { return n < a->size() ? (*a)[n] : X(); }
template <class X> X IndexOrDefault(const vector<X> &a, int n, const X& b) { return n < a. size() ?   a [n] : b; }

template <class X> void InsertOrErase(X *v, const typename X::value_type &val, bool insert) {
    if (insert) v->insert(val);
    else        v->erase(val);
}

template <class I1, class I2> size_t MismatchOffset(I1 first1, I1 last1, I2 first2, I2 last2) {
    int ret = 0;
    while (first1 != last1 && *first1 == *first2) { ++first1; ++first2; ++ret; }
    return ret;
}

template <class X> void FilterValues(X *v, const typename X::value_type &val) {
    for (typename X::iterator i = v->begin(); i != v->end(); /**/) {
        if (*i == val) i = v->erase(i);
        else i++;
    }
}

template <class X> void Move(X *to, const X *from, int size) {
    if      (from == to) return;
    else if (from <  to) { for (int i=size-1; i >= 0; i--) to[i] = from[i]; }
    else                 { for (int i=0; i <= size-1; i++) to[i] = from[i]; }
} 

template <class X> void Move(X &buf, int to_ind, int from_ind, int size) {
    if      (from_ind == to_ind) return;
    else if (from_ind <  to_ind) { for (int i=size-1; i >= 0; i--) buf[to_ind+i] = buf[from_ind+i]; }
    else                         { for (int i=0; i <= size-1; i++) buf[to_ind+i] = buf[from_ind+i]; }
} 

template <class X, class Y> void Move(X &buf, int to_ind, int from_ind, int size, const function<void(Y&,Y&)> &move_cb) {
    if      (from_ind == to_ind) return;
    else if (from_ind <  to_ind) { for (int i=size-1; i >= 0; i--) move_cb(buf[to_ind+i], buf[from_ind+i]); }
    else                         { for (int i=0; i <= size-1; i++) move_cb(buf[to_ind+i], buf[from_ind+i]); }
} 

template <class X> struct ScopedValue {
    X *v, ov;
    ScopedValue(X *V, X nv) : v(V), ov(V ? *V : X()) { *v = nv; }
    ~ScopedValue() { if (v) *v = ov; }
};

template <class X> struct ScopedDeltaTracker {
    X *v, ov; function<X()> f;
    ScopedDeltaTracker(X *V, const function<X()> &F) : v(V), ov(V ? F() : X()), f(F) {}
    ~ScopedDeltaTracker() { if (v) (*v) += (f() - ov); }
};

template <class T1, class T2, class T3> struct Triple {
    T1 first; T2 second; T3 third;
    Triple() {}
    Triple(const T1 &t1, const T2 &t2, const T3 &t3) : first(t1), second(t2), third(t3) {}
    Triple(const Triple &copy) : first(copy.first), second(copy.second), third(copy.third) {}
    const Triple &operator=(const Triple &r) { first=r.first; second=r.second; third=r.third; return *this; }
    bool operator<(const Triple &r) const { SortImpl3(first, r.first, second, r.second, third, r.third); }
};

template <class T1, class T2, class T3, class T4> struct Quadruple {
    T1 first; T2 second; T3 third; T4 fourth;
    Quadruple() {}
    Quadruple(const T1 &t1, const T2 &t2, const T3 &t3, const T4 &t4) : first(t1), second(t2), third(t3), fourth(t4) {}
    Quadruple(const Quadruple &copy) : first(copy.first), second(copy.second), third(copy.third), fourth(copy.fourth) {}
    const Quadruple &operator=(const Quadruple &r) { first=r.first; second=r.second; third=r.third; fourth=r.fourth; return *this; }
    bool operator<(const Quadruple &r) const { SortImpl4(first, r.first, second, r.second, third, r.third, fourth, r.fourth); }
};

template <class I1, class I2> struct IterPair {
    typedef pair<const typename I1::value_type&, const typename I2::value_type&> value_type;
    I1 i1; I2 i2;
    IterPair(I1 x1, I2 x2) : i1(x1), i2(x2) {}
    IterPair& operator++() { ++i1; ++i2; return *this; }
    IterPair& operator++(int) { IterPair v(*this); ++(*this); return v; }
    value_type operator*() const { return value_type(*i1, *i2); }
    bool operator!=(const IterPair &x) const { return i1 != x.i1; }
};

template <class X> struct FreeListVector {
    vector<X> data;
    vector<int> free_list;

    int size() const { return data.size(); }
    const X& back() const { return data.back(); }
    const X& operator[](int i) const { return data[i]; }
    /**/  X& operator[](int i)       { return data[i]; }

    virtual void Clear() { data.clear(); free_list.clear(); }
    virtual int Insert(const X &v) {
        if (free_list.empty()) {
            data.push_back(v);
            return data.size()-1;
        } else {
            int free_ind = PopBack(free_list);
            data[free_ind] = v;
            return free_ind;
        }
    }
    virtual void Erase(unsigned ind) {
        CHECK_LT(ind, data.size());
        free_list.push_back(ind);
    }
};

template <class X> struct IterableFreeListVector : public FreeListVector<X> { /// Expects X.deleted
    virtual void Erase(unsigned ind) {
        FreeListVector<X>::Erase(ind);
        FreeListVector<X>::data[ind].deleted = 1;
    }
};

template <class X, class Alloc = std::allocator<X> > struct FreeListBlockAllocator {
    typedef pair<X*, int> Block;
    const int block_size;
    Alloc alloc;
    vector<Block> data;
    vector<int> free_list;

    ~FreeListBlockAllocator() { Clear(); }
    FreeListBlockAllocator(int bs=1024) : block_size(bs) {}

    int size() const { return data.empty() ? 0 : (data.size()-1) * block_size + data.back().second; }
    const X& back() const { return data.back().first[data.back().second-1]; }
    const X& operator[](int i) const { return data[GetBlockFromIndex(i)].first[GetOffsetFromIndex(i)]; }
    /**/  X& operator[](int i)       { return data[GetBlockFromIndex(i)].first[GetOffsetFromIndex(i)]; }

    int MakeIndex(int block, int offset) const { return block * block_size + offset; }
    int GetBlockFromIndex(int ind) const { return ind / block_size; }
    int GetOffsetFromIndex(int ind) const { return ind % block_size; }

    void Clear() {
        for (auto b : data) {
            for (X *j = b.first, *je = j + b.second; j != je; ++j) alloc.destroy(j);
            alloc.deallocate(b.first, block_size);
        }
        data.clear();
        free_list.clear();
    }

    int Insert(const X &v) {
        if (free_list.empty()) {
            if (!data.size() || data.back().second == block_size)
                data.emplace_back(alloc.allocate(block_size), 0);
            Block *b = &data.back();
            alloc.construct(&b->first[b->second], v);
            return MakeIndex(data.size()-1, b->second++);
        } else {
            int free_ind = PopBack(free_list);
            X *j = &(*this)[free_ind];
            alloc.destroy(j);
            alloc.construct(j, v);
            return free_ind;
        }
    }

    void Erase(unsigned ind) { free_list.push_back(ind); }
};

template <typename X> struct ArraySegmentIter {
    const X *buf; int i, ind, len, cur_start; X cur_attr;
    ArraySegmentIter(const basic_string<X> &B) : buf(B.size() ? &B[0] : 0), i(-1), ind(0), len(B.size()) { Increment(); }
    ArraySegmentIter(const vector      <X> &B) : buf(B.size() ? &B[0] : 0), i(-1), ind(0), len(B.size()) { Increment(); }
    ArraySegmentIter(const X *B, int L)        : buf(B),                    i(-1), ind(0), len(L)        { Increment(); }
    const X *Data() const { return &buf[cur_start]; }
    int Length() const { return ind - cur_start; }
    bool Done() const { return cur_start == len; }
    void Update() { cur_start = ind; if (!Done()) cur_attr = buf[ind]; }
    void Increment() { Update(); while (ind != len && cur_attr == buf[ind]) ind++; i++; }
};

template <typename X, typename Y, Y (X::*Z)> struct ArrayMemberSegmentIter {
    typedef function<void (const X&)> CB;
    typedef function<bool (const Y&, const Y&)> Cmp;
    const X *buf; int i, ind, len, cur_start; Y cur_attr; CB cb; Cmp cmp;
    ArrayMemberSegmentIter(const vector<X> &B)              : buf(B.size() ? &B[0] : 0), i(-1), ind(0), len(B.size()),         cmp(equal_to<Y>()) { Increment(); }
    ArrayMemberSegmentIter(const X *B, int L)               : buf(B),                    i(-1), ind(0), len(L),                cmp(equal_to<Y>()) { Increment(); }
    ArrayMemberSegmentIter(const X *B, int L, const CB &Cb) : buf(B),                    i(-1), ind(0), len(L),        cb(Cb), cmp(equal_to<Y>()) { Increment(); }
    const X *Data() const { return &buf[cur_start]; }
    int Length() const { return ind - cur_start; }
    bool Done() const { return cur_start == len; }
    void Update() { cur_start = ind; if (!Done()) cur_attr = buf[ind].*Z; }
    void Increment() { Update(); while (ind != len && cmp(cur_attr, buf[ind].*Z)) { if (cb) cb(buf[ind]); ind++; } i++; }
};

template <typename X, typename Y, Y (X::*Z1), Y (X::*Z2)> struct ArrayMemberPairSegmentIter {
    typedef function<void (const X&)> CB;
    typedef function<bool (const Y&, const Y&)> Cmp;
    const X *buf; int i, ind, len, cur_start; Y cur_attr1, cur_attr2; CB cb; Cmp cmp;
    ArrayMemberPairSegmentIter(const vector<X> &B)              : buf(B.size() ? &B[0] : 0), i(-1), ind(0), len(B.size()),         cmp(equal_to<Y>()) { Increment(); }
    ArrayMemberPairSegmentIter(const X *B, int L)               : buf(B),                    i(-1), ind(0), len(L),                cmp(equal_to<Y>()) { Increment(); }
    ArrayMemberPairSegmentIter(const X *B, int L, const CB &Cb) : buf(B),                    i(-1), ind(0), len(L),        cb(Cb), cmp(equal_to<Y>()) { Increment(); }
    const X *Data() const { return &buf[cur_start]; }
    int Length() const { return ind - cur_start; }
    bool Done() const { return cur_start == len; }
    void Update() { cur_start = ind; if (!Done()) { const X& b = buf[ind]; cur_attr1 = b.*Z1; cur_attr2 = b.*Z2; } }
    void Increment() { Update(); while (ind != len) { const X& b = buf[ind]; if (!cmp(cur_attr1, b.*Z1) || !cmp(cur_attr2, b.*Z2)) break; if (cb) cb(buf[ind]); ind++; } i++; }
};

template <typename X, typename Y, Y (X::*Z)() const> struct ArrayMethodSegmentIter {
    typedef function<void (const X&)> CB;
    typedef function<bool (const Y&, const Y&)> Cmp;
    const X *buf; int i, ind, len, cur_start; Y cur_attr; CB cb; Cmp cmp;
    ArrayMethodSegmentIter(const vector<X> &B)              : buf(B.size() ? &B[0] : 0), i(-1), ind(0), len(B.size()),         cmp(equal_to<Y>()) { Increment(); }
    ArrayMethodSegmentIter(const X *B, int L)               : buf(B),                    i(-1), ind(0), len(L),                cmp(equal_to<Y>()) { Increment(); }
    ArrayMethodSegmentIter(const X *B, int L, const CB &Cb) : buf(B),                    i(-1), ind(0), len(L),        cb(Cb), cmp(equal_to<Y>()) { Increment(); }
    const X *Data() const { return &buf[cur_start]; }
    int Length() const { return ind - cur_start; }
    bool Done() const { return cur_start == len; }
    void Update() { cur_start = ind; if (!Done()) cur_attr = (buf[ind].*Z)(); }
    void Increment() { Update(); while (ind != len && cmp(cur_attr, (buf[ind].*Z)())) { if (cb) cb(buf[ind]); ind++; } i++; }
};

template <class X> struct FlattenedArrayValues {
    typedef pair<int, int> Iter;
    typedef function<int(X*, int)> GetValCB;
    X *data; int l; GetValCB get_val;
    FlattenedArrayValues(X *D, int L, GetValCB cb) : data(D), l(L), get_val(cb) {}

    Iter LastIter() { return Iter(X_or_1(l)-1, l ? get_val(data, l-1)-1 : 0); }
    void AdvanceIter(Iter *o, int n) {
        if (!l) return;
        for (int i=abs(n), d; i; i-=d) {
            int v = get_val(data, o->first);
            if (n > 0) {
                d = min(i, v - o->second);
                if ((o->second += d) >= v) {
                    if (o->first >= l-1) { *o = LastIter(); return; }
                    *o = Iter(o->first+1, 0);
                }
            } else {
                d = min(i, o->second+1);
                if ((o->second -= d) < 0) {
                    if (o->first <= 0) { *o = Iter(0, 0); return; }
                    *o = Iter(o->first-1, get_val(data, o->first-1)-1);
                }
            }
        }
    }
    int Distance(Iter i1, Iter i2, int maxdist=0) {
        int dist = 0;
        if (i2 < i1) swap(i1, i2);
        for (int i = i1.first; i <= i2.first; i++) {
            int v = get_val(data, i);
            dist += v;
            if (i == i1.first) dist -= i1.second;
            if (i == i2.first) dist -= (v - i2.second);
            if (maxdist && dist >= maxdist) return maxdist;
        }
        return dist;
    }
};

template <class X> struct MessageQueue {
    condition_variable cv;
    bool use_cv = 0;
    deque<X> queue;
    mutex lock;

    void Write(const X &x) {
        ScopedMutex sm(lock);
        queue.push_back(x);
        if (use_cv) cv.notify_one();
    }
    bool NBRead(X* out) {
        if (queue.empty()) return false;
        ScopedMutex sm(lock);
        if (queue.empty()) return false;
        *out = PopBack(queue);
        return true;
    }
    X Read() {
        unique_lock<mutex> ul(lock);
        cv.wait(ul, [this](){ return !this->queue.empty(); } );
        return PopBack(queue);
    }
};

struct CallbackQueue : public MessageQueue<Callback*> {
    void Shutdown() { Write(new Callback()); }
    void HandleMessage(Callback *cb) { (*cb)(); delete cb; }
    void HandleMessages() { Callback *cb=0; while (NBRead(&cb)) HandleMessage(cb); }
    void HandleMessagesLoop() { for (use_cv=1; GetLFApp()->run; ) HandleMessage(Read()); }
};

struct CallbackList {
    bool dirty;
    vector<Callback> data;
    CallbackList() : dirty(0) {}
    int Size() const { return data.size(); }
    void Clear() { dirty=0; data.clear(); }
    void Run() { for (auto i = data.begin(); i != data.end(); ++i) (*i)(); Clear(); }
    void Add(const Callback &cb) { data.push_back(cb); dirty=1; }
    void Add(const CallbackList &cb) { data.insert(data.end(), cb.data.begin(), cb.data.end()); dirty=1; }
};
#define CallbackListAdd(cblist, ...) (cblist)->Add(bind(__VA_ARGS__))
#define CallbackListsAdd(cblists, ...) for(CallbackList **cbl=(cblists); *cbl; cbl++) CallbackListAdd(*cbl, __VA_ARGS__)

template <class X> struct TopN {
    const int num;
    set<X> data;
    TopN(int n) : num(n) {}

    void Insert(const X &v) {
        if (data.size() < num) data.insert(v);
        else if (v < *data.rbegin()) {
            data.erase(ForwardIteratorFromReverse(data.rbegin()));
            data.insert(v);
        }
    }
};

struct RingIndex {
    int size, back, count, total;
    RingIndex(int S=0) : size(S), back(0), count(0), total(0) {}
    static int Wrap(int i, int s) { if ((i = s ? (i % s) : 0) < 0) i += s; return i; }
    static int WrapOver(int i, int s) { return i == s ? i : Wrap(i, s); }
    int Back() const { return Index(-1); }
    int Front() const { return Index(-count); }
    int Index(int i) const { return AbsoluteIndex(back + i); }
    int IndexOf(int i) const { return i - back - (i >= back ? size : 0); }
    int AbsoluteIndex(int i) const { return Wrap(i, size); }
    void IncrementSize(int n) { count = min(size, count + n); total += n; } 
    void DecrementSize(int n) { total = max(0, total - n); if (total < size) count = total; }
    void PopBack  (int n) { back = Index(-n); DecrementSize(n); }
    void PushBack (int n) { back = Index(+n); IncrementSize(n); }
    void PushFront(int n) { back = Index(-max(0, count+n-size)); IncrementSize(n); }
    void PopFront (int n) { DecrementSize(n); }
    void Clear() { back=count=total=0; }
};

template <class X> struct RingVector {
    RingIndex ring;
    vector<X> data;
    RingVector(int S=0) : ring(S), data(S) {}
    X&       operator[](int i)       { return data[ring.Index(i)]; }
    const X& operator[](int i) const { return data[ring.Index(i)]; }
    virtual int Size() const { return ring.count; }
    virtual int IndexOf(const X *v) const { return ring.IndexOf(v - &data[0]); }
    virtual X   *Back ()         { return &data[ring.Back()]; }
    virtual X   *Front()         { return &data[ring.Front()]; }
    virtual X   *PushBack ()     { ring.PushBack (1); return Back(); }
    virtual X   *PushFront()     { ring.PushFront(1); return Front(); }
    virtual void PopFront(int n) { ring.PopFront(n); }
    virtual void PopBack (int n) { ring.PopBack (n); }
    virtual void Clear() { ring.Clear(); }
};

struct RingBuf {
    RingIndex ring;
    int samples_per_sec, width, bytes;
    Allocator *alloc;
    char *buf;
    microseconds *stamp;
    ~RingBuf() { alloc->Free((void*)buf); alloc->Free((void*)stamp); }
    RingBuf(int SPS=0, int SPB=0, int Width=0, Allocator *Alloc=0) : samples_per_sec(0), width(0), bytes(0),
    alloc(Alloc?Alloc:Allocator::Default()), buf(0), stamp(0) { if (SPS) Resize(SPS, X_or_Y(SPB, SPS), Width); }

    int Bucket(int n) const { return ring.AbsoluteIndex(n); }
    virtual void *Ind(int index) const { return (void *)(buf + index*width); }

    void Resize(int SPS, int SPB, int Width=0);
    enum WriteFlag { Peek=1, Stamp=2 };
    virtual void *Write(int writeFlag=0, microseconds timestamp=microseconds(-1));

    int Dist(int indexB, int indexE) const;
    int Since(int index, int Next=-1) const;
    virtual void *Read(int index, int Next=-1) const;
    virtual microseconds ReadTimestamp(int index, int Next=-1) const;

    struct Handle : public Vec<float> {
        RingBuf *sb; int next, nlen;
        Handle() : sb(0), next(-1), nlen(-1) {}
        Handle(RingBuf *SB, int Next=-1, int Len=-1) : sb(SB), next((sb && Next != -1)?sb->Bucket(Next):-1), nlen(Len) {}
        virtual int Rate() const { return sb->samples_per_sec; }
        virtual int Len() const { return nlen >= 0 ? nlen : sb->ring.size; }
        virtual float *Index(int index) { return (float *)sb->Ind(index); }
        virtual float Ind(int index) const { return *(float *)sb->Ind(index); }
        virtual void Write(float v, int flag=0, microseconds ts=microseconds(-1)) { *(float *)(sb->Write(flag, ts)) = v; }
        virtual void Write(float *v, int flag=0, microseconds ts=microseconds(-1)) { *(float *)(sb->Write(flag, ts)) = *v; }
        virtual float Read(int index) const { return *(float *)sb->Read(index, next); }
        virtual float *ReadAddr(int index) const { return (float *)sb->Read(index, next); } 
        virtual microseconds ReadTimestamp(int index) const { return sb->ReadTimestamp(index, next); }
        void CopyFrom(const RingBuf::Handle *src);
    };

    template <typename X> struct HandleT : public Handle {
        HandleT() {}
        HandleT(RingBuf *SB, int Next=-1, int Len=-1) : Handle(SB, Next, Len) {}
        virtual float Ind(int index) const { return *(X *)sb->Ind(index); }
        virtual float Read(int index) const { return *(X *)sb->Read(index, next); }
        virtual void Write(float  v, int flag=0, microseconds ts=microseconds(-1)) { *(X *)(sb->Write(flag, ts)) =  v; }
        virtual void Write(float *v, int flag=0, microseconds ts=microseconds(-1)) { *(X *)(sb->Write(flag, ts)) = *v; }
    };

    struct DelayHandle : public Handle {
        int delay;
        DelayHandle(RingBuf *SB, int next, int Delay) : Handle(SB, next+Delay), delay(Delay) {}
        virtual float Read(int index) const {
            if ((index < 0 && index >= -delay) || (index > 0 && index >= Len() - delay)) return 0;
            return Handle::Read(index);
        }
    };

    struct WriteAheadHandle : public Handle {
        WriteAheadHandle(RingBuf *SB) : Handle(SB, SB->ring.back) {}
        virtual void Write(float v, int flag=0, microseconds ts=microseconds(-1)) { *(float*)Write(flag, ts) = v; }
        virtual void *Write(int flag=0, microseconds ts=microseconds(-1)) {
            void *ret = sb->Ind(next);
            if (flag & Stamp) sb->stamp[next] = (ts != microseconds(-1)) ? ts : Now();
            if (!(flag & Peek)) next = (next+1) % sb->ring.size;
            return ret;
        }
        void Commit() { sb->ring.back = sb->Bucket(next); }
    };

    template <class T=double> struct MatrixHandleT : public matrix<T> {
        RingBuf *sb;
        int next;
        MatrixHandleT(RingBuf *SB, int Next, int Rows) : sb(SB), next((sb && Next != -1)?sb->Bucket(Next):-1) { 
            matrix<T>::bytes = sb->width;
            matrix<T>::M = Rows?Rows:sb->ring.size;
            matrix<T>::N = matrix<T>::bytes/sizeof(T);
        }

        T             * row(int i)       { return (T*)sb->Read(i, next); }
        const T       * row(int i) const { return (T*)sb->Read(i, next); }
        Complex       *crow(int i)       { return (Complex*)sb->Read(i, next); }
        const Complex *crow(int i) const { return (Complex*)sb->Read(i, next); }
    };
    typedef MatrixHandleT<double> MatrixHandle;

    template <class T=double> struct RowMatHandleT {
        RingBuf *sb;
        int next;
        matrix<T> wrap;
        void Init() { wrap.M=1; wrap.bytes=sb->width; wrap.N=wrap.bytes/sizeof(T); }

        ~RowMatHandleT() { wrap.m=0; }
        RowMatHandleT() : sb(0), next(0) {}
        RowMatHandleT(RingBuf *SB) : sb(SB), next(-1) { Init(); }
        RowMatHandleT(RingBuf *SB, int Next) : sb(SB), next((sb && Next != -1)?sb->Bucket(Next):-1) { Init(); }

        matrix<T> *Ind(int index) { wrap.m = (double *)sb->Ind(index); return &wrap; }
        matrix<T> *Read(int index) { wrap.m = (double *)sb->Read(index, next); return &wrap; }
        double *ReadRow(int index) { wrap.m = (double *)sb->Read(index, next); return wrap.row(0); }
        matrix<T> *Write(int flag=0, microseconds ts=microseconds(-1)) { wrap.m = (double *)sb->Write(flag, ts); return &wrap; }
        microseconds ReadTimestamp(int index) const { return sb->ReadTimestamp(index, next); }
    };
    typedef RowMatHandleT<double> RowMatHandle;
};

struct ColMatPtrRingBuf : public RingBuf {
    Matrix *wrap;
    int col;
    ColMatPtrRingBuf(Matrix *m, int ci, int SPS=0) : wrap(m), col(ci)  {
        samples_per_sec = SPS;
        ring.size = m->M;
        width = sizeof(double);
        bytes = width * ring.size;
        ring.back = 0;
    }
    virtual void *Ind(int index) const { return &wrap->row(index)[col]; }
    virtual void *Read(int index, int Next=-1) const { return &wrap->row(index)[col]; }
    virtual void *Write(int writeFlag=0, microseconds timestamp=microseconds(-1)) { return &wrap->row(ring.back++)[col]; } 
    virtual microseconds ReadTimestamp(int index, int Next=-1) const { return microseconds(index); }
};

struct BloomFilter {
    int M, K;
    unsigned char *buf;
    ~BloomFilter() { free(buf); }
    BloomFilter(int m, int k, unsigned char *B) : M(m), K(k), buf(B) {}
    static BloomFilter *Empty(int M, int K) { int s=M/8+1; BloomFilter *bf = new BloomFilter(M, K, (unsigned char*)calloc(s, 1)); return bf; }
    static BloomFilter *Full (int M, int K) { int s=M/8+1; BloomFilter *bf = new BloomFilter(M, K, (unsigned char*)malloc(s)); memset(bf->buf, ~0, s); return bf; }

    void Set  (long long val) { return Set  ((      unsigned char *)&val, sizeof(long long)); }
    void Clear(long long val) { return Clear((      unsigned char *)&val, sizeof(long long)); }
    int  Get  (long long val) { return Get  ((const unsigned char *)&val, sizeof(long long)); }
    int  Not  (long long val) { return Not  ((const unsigned char *)&val, sizeof(long long)); }

    void Set  (      unsigned char *val, size_t len) {        ForEachBitBucketDo(BitString::Set,   val, len); }
    void Clear(      unsigned char *val, size_t len) {        ForEachBitBucketDo(BitString::Clear, val, len); }
    int  Get  (const unsigned char *val, size_t len) { return ForEachBitBucketDo(BitString::Get,   val, len); }
    int  Not  (const unsigned char *val, size_t len) { return ForEachBitBucketDo(BitString::Not,   val, len); }

    int ForEachBitBucketDo(int (*op)(unsigned char *b, int bucket), const unsigned char *val, size_t len) {
        unsigned long long h = fnv64(val, len);
        unsigned h1 = (h >> 32) % M, h2 = (h & 0xffffffff) % M;
        for (int i = 1; i <= K; i++) {
            h1 = (h1 + h2) % M;
            h2 = (h2 +  i) % M;
            if (!op(buf, h1)) return 0;
        }
        return 1;
    }
};

struct Toggler {
    enum { Default = 1, OneShot = 2 };
    int mode;
    bool *target;
    Toggler(bool *T, int M=Default) : target(T), mode(M) {}
    bool Toggle() {
        if (*target && mode == OneShot) return false;
        else { *target = !*target; return true; }
    }
};

template <class X> struct CategoricalVariable {
    int ind;
    vector<X> data;
    CategoricalVariable(int n, ...) : ind(0), data(n) {
        va_list ap;
        va_start(ap, n);
        for (auto &di : data) di = va_arg(ap, X);
        va_end(ap);
    }
    X Cur() const { return data[ind]; }
    bool Enabled() const { return ind != 0; }
    int Match(X x) const { for (int i=0, n=data.size(); i<n; i++) if (x == data[i]) return i; return -1; }
    X Next() { return data[(ind = (ind+1) % data.size())]; }
    void On()  { ind = 1; }
    void Off() { ind = 0; }
};
}; // namespace LFL

#include "lfapp/tree.h"
#include "lfapp/trie.h"

#endif // __LFL_LFAPP_LFTYPES_H__
