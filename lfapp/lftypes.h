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

#ifdef WIN32
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
static __forceinline int ffsl(long x) {
	unsigned long i;
	if (_BitScanForward(&i, x)) return (i + 1);
	return (0);
}
static __forceinline int ffs(int x) { return ffsl(x); }
#endif

namespace LFL {
template <typename K, typename V> typename map<K, V>::iterator FindOrInsert(map<K, V> &m, K k, V v, int *inserted) {
    LFL_STL_NAMESPACE::pair<typename map<K, V>::iterator, bool> ret = m.insert(typename map<K, V>::value_type(k, v));
    if (inserted) (*inserted)++;
    return ret.first;
}

template <typename K, typename V> typename map<K, V>::iterator FindOrInsert(map<K, V> &m, K k, V v, bool *inserted=0) {
    LFL_STL_NAMESPACE::pair<typename map<K, V>::iterator, bool> ret = m.insert(typename map<K, V>::value_type(k, v));
    if (inserted) *inserted = ret.second;
    return ret.first;
}

template <typename K, typename V> typename map<K, V*>::iterator FindOrInsert(map<K, V*> &m, K k, V* v, bool *inserted=0) {
    LFL_STL_NAMESPACE::pair<typename map<K, V*>::iterator, bool> ret = m.insert(typename map<K, V*>::value_type(k, v));
    if (inserted) *inserted = ret.second;
    return ret.first;
}

template <typename K, typename V> typename map<K, V*>::iterator FindOrInsert(map<K, V*> &m, K k, bool *inserted=0) {
    LFL_STL_NAMESPACE::pair<typename map<K, V*>::iterator, bool> ret = m.insert(typename map<K, V*>::value_type(k, 0));
    if (ret.second) ret.first->second = new V();
    if (inserted) *inserted = ret.second;
    return ret.first;
}

template <typename K, typename V> typename unordered_map<K, V>::iterator FindOrInsert(unordered_map<K, V> &m, K k, V v, bool *inserted) {
    LFL_STL_NAMESPACE::pair<typename unordered_map<K, V>::iterator, bool> ret = m.insert(typename unordered_map<K, V>::value_type(k, v));
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
    if (iter == m.end()) lfapp_log(LogLevel::Fatal, file, line, StrCat("FindOrDie(", k, ")"));
    return iter->second;
}
#define FindOrDie(m, k) FindOrDie(__FILE__, __LINE__, m, k)

template <typename X> bool Contains(const X &c, const typename X::key_type &k) { return c.find(k) != c.end(); }
template <typename X> void EnsureSize(X &x, int n) { if (x.size() < n) x.resize(n); }

template <typename X> typename X::value_type &PushFront(X &v, const typename X::value_type &x) { v.push_front(x); return v.front(); }
template <typename X> typename X::value_type &PushBack (X &v, const typename X::value_type &x) { v.push_back (x); return v.back (); }
template <typename X> typename X::value_type  PopFront (X &v) { typename X::value_type ret = v.front(); v.pop_front(); return ret; }
template <typename X> typename X::value_type  PopBack  (X &v) { typename X::value_type ret = v.back (); v.pop_back (); return ret; }

template <class X>       X *VectorGet(      vector<X> &x, int n) { return n < x.size() ? &x[n] : 0; }
template <class X> const X *VectorGet(const vector<X> &x, int n) { return n < x.size() ? &x[n] : 0; }
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

template <class X> X IndexOrDefault(const vector<X> &a, int n)             { return n < a.size() ? a[n] : X(); }
template <class X> X IndexOrDefault(const vector<X> &a, int n, const X& b) { return n < a.size() ? a[n] : b; }

template <class X> void ValueFilter(X *v, const typename X::value_type &val) {
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
    else if (from_ind <  to_ind) { for (int i=size-1; i >= 0; i--) buf[to_ind + i] = buf[from_ind + i]; }
    else                         { for (int i=0; i <= size-1; i++) buf[to_ind + i] = buf[from_ind + i]; }
} 

template <class T1, class T2, class T3> struct Triple {
    T1 first; T2 second; T3 third;
    Triple() {}
    Triple(const T1 &t1, const T2 &t2, const T3 &t3) : first(t1), second(t2), third(t3) {}
    Triple(const Triple &copy) : first(copy.first), second(copy.second), third(copy.third) {}
    const Triple &operator=(const Triple &r) { first=r.first; second=r.second; third=r.third; return *this; }
    bool operator<(const Triple &r) const { 
        if      (first  < r.first)  return true;
        else if (first  > r.first)  return false;
        else if (second < r.second) return true;
        else if (second > r.second) return false;
        else return third < r.third;
    }
};

template <class T1, class T2, class T3, class T4> struct Quadruple {
    T1 first; T2 second; T3 third; T4 fourth;
    Quadruple() {}
    Quadruple(const T1 &t1, const T2 &t2, const T3 &t3, const T4 &t4) : first(t1), second(t2), third(t3), fourth(t4) {}
    Quadruple(const Quadruple &copy) : first(copy.first), second(copy.second), third(copy.third), fourth(copy.fourth) {}
    const Quadruple &operator=(const Quadruple &r) { first=r.first; second=r.second; third=r.third; fourth=r.fourth; return *this; }
    bool operator<(const Quadruple &r) const { 
        if      (first  < r.first)  return true;
        else if (first  > r.first)  return false;
        else if (second < r.second) return true;
        else if (second > r.second) return false;
        else if (third  < r.third) return true;
        else if (third  > r.third) return false;
        else return fourth < r.fourth;
    }
};

template <class X> struct FreeListVector : public vector<X> {
    vector<int> free_list;
    void Clear() { this->clear(); free_list.clear(); }
    int Insert(const X &v) {
        if (free_list.empty()) {
            this->push_back(v);
            return this->size()-1;
        } else {
            int free_ind = PopBack(free_list);
            (*this)[free_ind] = v;
            return free_ind;
        }
    }
    void Erase(unsigned ind) {
        CHECK_LT(ind, this->size());
        (*this)[ind].deleted = 1; // expectes X.deleted
        free_list.push_back(ind);
    }
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

template <class X> struct ValueSet {
    int i, c; X *d;
    int match(X x) { for (int j=0; j<c; j++) if (x == d[j]) return j; return -1; }
    ValueSet(int n, ...) : i(0), c(n), d(new X[c]) { va_list ap; va_start(ap, n); for (int j=0; j<c; j++) d[j] = va_arg(ap, X); va_end(ap); }
    ~ValueSet() { delete[] d; }
    X cur() { return d[i]; }
    X next() { i=(i+1)%c; return d[i]; }
    bool enabled() { return i!=0; }
    void on() { i=1; }
    void off(){ i=0; }
};

struct AnyBoolSet {
    int count;
    AnyBoolSet() : count(0) {}
    bool Get() const { return count; }
};

struct AnyBoolElement {
    AnyBoolSet *parent; bool val;
    AnyBoolElement(AnyBoolSet *P, bool v=0) : parent(P), val(0) { Set(v); }
    virtual ~AnyBoolElement() { Set(0); }
    bool Get() const { return val; }
    void Set(bool v) {
        if (val == v) return;
        if (v) parent->count++;
        else   parent->count--;
        val = v;
    }
};

struct ToggleBool {
    enum { Default = 1, OneShot = 2 };
    bool *toggle; int mode;
    ToggleBool(bool *t, int m=Default) : toggle(t), mode(m) {}
    bool Toggle() { if (*toggle && mode == OneShot) return false; *toggle = !*toggle; return true; }
};

struct ReallocHeap {
    char *heap; int size, len;
    ~ReallocHeap(); 
    ReallocHeap(int startSize=65536);
    void reset() { len=0; }
    int alloc(int len);
};

struct Bit {
    static int Count(unsigned long long n) { int c; for (c=0; n; c++) n &= (n-1); return c; }
    static void Indices(unsigned long long in, int *o) { 
        for (int b, n = in & 0xffffffff; n; /**/) { b = ffs(n)-1; n ^= (1<<b); *o++ = b;    }
        for (int b, n = in >> 32       ; n; /**/) { b = ffs(n)-1; n ^= (1<<b); *o++ = b+32; }
        *o++ = -1;
    }
};

struct BitField {
    static int Clear(      unsigned char *b, int bucket) {          b[bucket/8] &= ~(1 << (bucket % 8)); return 1; }
    static int Set  (      unsigned char *b, int bucket) {          b[bucket/8] |=  (1 << (bucket % 8)); return 1; }
    static int Get  (const unsigned char *b, int bucket) { return   b[bucket/8] &   (1 << (bucket % 8));           }
    static int Not  (const unsigned char *b, int bucket) { return !(b[bucket/8] &   (1 << (bucket % 8)));          }
    static int Get  (      unsigned char *b, int bucket) { return Get((const unsigned char*)b, bucket); }
    static int Not  (      unsigned char *b, int bucket) { return Not((const unsigned char*)b, bucket); }

    static int FirstSet  (const unsigned char *b, int l) { for (int i=0;   i<l;  i++) { unsigned char c=b[i]; if (c != 0)   return i*8 + ffs( c)-1; } return -1; }
    static int FirstClear(const unsigned char *b, int l) { for (int i=0;   i<l;  i++) { unsigned char c=b[i]; if (c != 255) return i*8 + ffs(~c)-1; } return -1; }
    static int  LastClear(const unsigned char *b, int l) { for (int i=l-1; i>=0; i--) { unsigned char c=b[i]; if (c != 255) return i*8 + ffs(~c)-1; } return -1; }
};

struct BloomFilter {
    int M, K; unsigned char *buf;
    ~BloomFilter() { free(buf); }
    BloomFilter(int m, int k, unsigned char *B) : M(m), K(k), buf(B) {}
    static BloomFilter *Empty(int M, int K) { int s=M/8+1; BloomFilter *bf = new BloomFilter(M, K, (unsigned char*)calloc(s, 1)); return bf; }
    static BloomFilter *Full (int M, int K) { int s=M/8+1; BloomFilter *bf = new BloomFilter(M, K, (unsigned char*)malloc(s)); memset(bf->buf, ~0, s); return bf; }

    void Set  (long long val) { return Set  ((      unsigned char *)&val, sizeof(long long)); }
    void Clear(long long val) { return Clear((      unsigned char *)&val, sizeof(long long)); }
    int  Get  (long long val) { return Get  ((const unsigned char *)&val, sizeof(long long)); }
    int  Not  (long long val) { return Not  ((const unsigned char *)&val, sizeof(long long)); }

    void Set  (      unsigned char *val, size_t len) {        ForEachBitBucketDo(BitField::Set,   val, len); }
    void Clear(      unsigned char *val, size_t len) {        ForEachBitBucketDo(BitField::Clear, val, len); }
    int  Get  (const unsigned char *val, size_t len) { return ForEachBitBucketDo(BitField::Get,   val, len); }
    int  Not  (const unsigned char *val, size_t len) { return ForEachBitBucketDo(BitField::Not,   val, len); }

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

struct RingIndex {
    int size, back, count, total;
    RingIndex(int S=0) : size(S), back(0), count(0), total(0) {}
    int Back() const { return Index(-1); }
    int Front() const { return Index(-count); }
    int Index(int i) const { return AbsoluteIndex(back + i); }
    int AbsoluteIndex(int i) const { if ((i = i % size) < 0) i += size; return i; }
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
    virtual X   *PushBack ()     { ring.PushBack (1); return &data[ring.Back()]; }
    virtual X   *PushFront()     { ring.PushFront(1); return &data[ring.Front()]; }
    virtual void PopFront(int n) { ring.PopFront(n); }
    virtual void PopBack (int n) { ring.PopBack (n); }
    virtual void Clear() { ring.Clear(); data.clear(); }
};

struct RingBuf {
    RingIndex ring;
    int samplesPerSec, width, bytes;
    Allocator *alloc; char *buf; Time *stamp;
    ~RingBuf() { alloc->free((void*)buf); alloc->free((void*)stamp); }
    RingBuf(int SPS=0, int SPB=0, int Width=0, Allocator *Alloc=0) : samplesPerSec(0), width(0), bytes(0), alloc(Alloc?Alloc:Singleton<MallocAlloc>::Get()), buf(0), stamp(0)
    { if (SPS) resize(SPS, X_or_Y(SPB, SPS), Width); }

    int bucket(int n) const { return ring.AbsoluteIndex(n); }
    virtual void *ind(int index) const { return (void *)(buf + index*width); }

    void resize(int SPS, int SPB, int Width=0);
    enum WriteFlag { Peek=1, Stamp=2 };
    virtual void *write(int writeFlag=0, Time timestamp=-1);

    int dist(int indexB, int indexE) const;
    int since(int index, int Next=-1) const;
    virtual void *read(int index, int Next=-1) const;
    virtual Time readtimestamp(int index, int Next=-1) const;

    struct Handle : public Vec<float> {
        RingBuf *sb; int next, nlen;
        Handle() : sb(0), next(-1), nlen(-1) {}
        Handle(RingBuf *SB, int Next=-1, int Len=-1) : sb(SB), next((sb && Next != -1)?sb->bucket(Next):-1), nlen(Len) {}
        virtual int rate() const { return sb->samplesPerSec; }
        virtual int len() const { return nlen >= 0 ? nlen : sb->ring.size; }
        virtual float *index(int index) { return (float *)sb->ind(index); }
        virtual float ind(int index) const { return *(float *)sb->ind(index); }
        virtual void write(float v, int flag=0, Time ts=-1) { *(float *)(sb->write(flag, ts)) = v; }
        virtual void write(float *v, int flag=0, Time ts=-1) { *(float *)(sb->write(flag, ts)) = *v; }
        virtual float read(int index) const { return *(float *)sb->read(index, next); }
        virtual float *readaddr(int index) const { return (float *)sb->read(index, next); } 
        virtual Time readtimestamp(int index) const { return sb->readtimestamp(index, next); }
        void CopyFrom(const RingBuf::Handle *src);
    };

    template <typename X> struct handle : public Handle {
        handle() {}
        handle(RingBuf *SB, int Next=-1, int Len=-1) : Handle(SB, Next, Len) {}
        virtual float ind(int index) const { return *(X *)sb->ind(index); }
        virtual float read(int index) const { return *(X *)sb->read(index, next); }
        virtual void write(float v, int flag=0, Time ts=-1) { *(X *)(sb->write(flag, ts)) = v; }
        virtual void write(float *v, int flag=0, Time ts=-1) { *(X *)(sb->write(flag, ts)) = *v; }
    };

    struct DelayHandle : public Handle {
        int delay;
        DelayHandle(RingBuf *SB, int next, int Delay) : Handle(SB, next+Delay), delay(Delay) {}
        virtual float read(int index) const {
            if ((index < 0 && index >= -delay) || (index > 0 && index >= len() - delay)) return 0;
            return Handle::read(index);
        }
    };

    struct WriteAheadHandle : public Handle {
        WriteAheadHandle(RingBuf *SB) : Handle(SB, SB->ring.back) {}
        virtual void write(float v, int flag=0, Time ts=-1) { *(float*)write(flag, ts) = v; }
        virtual void *write(int flag=0, Time ts=-1) {
            void *ret = sb->ind(next);
            if (flag & Stamp) sb->stamp[next] = ts ? ts : Now() * 1000;
            if (!(flag & Peek)) next = (next+1) % sb->ring.size;
            return ret;
        }
        void commit() { sb->ring.back = sb->bucket(next); }
    };

    template <class T=double> struct matrixHandle : public matrix<T> {
        RingBuf *sb; int next;

        matrixHandle(RingBuf *SB, int Next, int Rows) : sb(SB), next((sb && Next != -1)?sb->bucket(Next):-1) { 
            matrix<T>::bytes = sb->width;
            matrix<T>::M = Rows?Rows:sb->ring.size;
            matrix<T>::N = matrix<T>::bytes/sizeof(T);
        }

        T             * row(int i)       { return (T*)sb->read(i, next); }
        const T       * row(int i) const { return (T*)sb->read(i, next); }
        Complex       *crow(int i)       { return (Complex*)sb->read(i, next); }
        const Complex *crow(int i) const { return (Complex*)sb->read(i, next); }
    };
    typedef matrixHandle<double> MatrixHandle;

    template <class T=double> struct rowMatHandle {
        RingBuf *sb; int next;
        matrix<T> wrap;
        void init() { wrap.M=1; wrap.bytes=sb->width; wrap.N=wrap.bytes/sizeof(T); }

        ~rowMatHandle() { wrap.m=0; }
        rowMatHandle() : sb(0), next(0) {}
        rowMatHandle(RingBuf *SB) : sb(SB), next(-1) { init(); }
        rowMatHandle(RingBuf *SB, int Next) : sb(SB), next((sb && Next != -1)?sb->bucket(Next):-1) { init(); }

        matrix<T> *ind(int index) { wrap.m = (double *)sb->ind(index); return &wrap; }
        matrix<T> *read(int index) { wrap.m = (double *)sb->read(index, next); return &wrap; }
        double *readrow(int index) { wrap.m = (double *)sb->read(index, next); return wrap.row(0); }
        matrix<T> *write(int flag=0, Time ts=-1) { wrap.m = (double *)sb->write(flag, ts); return &wrap; }
        Time readtimestamp(int index) const { return sb->readtimestamp(index, next); }
    };
    typedef rowMatHandle<double> RowMatHandle;
};

struct ColMatPtrRingBuf : public RingBuf {
    Matrix *wrap; int col;
    ColMatPtrRingBuf(Matrix *m, int ci, int SPS=0) : wrap(m), col(ci)  {
        samplesPerSec = SPS;
        ring.size = m->M;
        width = sizeof(double);
        bytes = width * ring.size;
        ring.back = 0;
    }
    virtual void *ind(int index) const { return &wrap->row(index)[col]; }
    virtual void *read(int index, int Next=-1) const { return &wrap->row(index)[col]; }
    virtual void *write(int writeFlag=0, Time timestamp=-1) { return &wrap->row(ring.back++)[col]; } 
    virtual Time readtimestamp(int index, int Next=-1) const { return index; }
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

template <class X> struct AssetMapT {
    bool loaded;
    vector<X> vec;
    map<string, X*> amap;
    AssetMapT() : loaded(0) {}
    void Add(const X &a) { CHECK(!loaded); vec.push_back(a); }
    void Unloaded(X *a) { if (!a->name.empty()) amap.erase(a->name); }
    void Load(X *a) { a->parent = this; if (!a->name.empty()) amap[a->name] = a; a->Load(); }
    void Load() { CHECK(!loaded); for (int i=0; i<vec.size(); i++) Load(&vec[i]); loaded=1; }
    X *operator()(const string &an) { return FindOrNull(amap, an); }
};
typedef AssetMapT<     Asset>      AssetMap;
typedef AssetMapT<SoundAsset> SoundAssetMap;
typedef AssetMapT<MovieAsset> MovieAssetMap;

struct Serializable {
    struct Stream;

    virtual int type() const = 0;
    virtual int size() const = 0;
    virtual int hdrsize() const = 0;
    virtual int in(const Stream *i) = 0;
    virtual void out(Stream *o) const = 0;

    string ToString(unsigned short seq=0);
    void ToString(string *out, unsigned short seq=0);
    void ToString(char *buf, int len, unsigned short seq=0);

    struct Header {
        static const int size = 4;
        unsigned short id, seq;

        void out(Stream *o) const;
        void in(const Stream *i);
    };

    bool HdrCheck(int content_len) { return content_len >= Header::size + hdrsize(); }
    bool    Check(int content_len) { return content_len >= Header::size +    size(); }
    bool HdrCheck(const Stream *is) { return HdrCheck(is->len()); }
    bool    Check(const Stream *is) { return    Check(is->len()); }
    int      Read(const Stream *is) { if (!HdrCheck(is)) return -1; return in(is); }

    struct Stream {
        virtual unsigned char  *n8()            = 0;
        virtual unsigned short *n16()           = 0;
        virtual unsigned       *n32()           = 0;
        virtual char           *get(int size=0) = 0;
        virtual char           *end()           = 0;

        char *buf; int size; mutable int offset; mutable bool error;
        Stream(char *B, int S) : buf(B), size(S), offset(0), error(0) {}

        int len() const { return size; }
        int pos() const { return offset; };
        int remaining() const { return size - offset; }
        const char *start() const { return buf; }
        const char *advance(int n=0) const { return get(n); }
        const char           *end() const { return buf + size; }
        const unsigned char  *n8()  const { unsigned char  *ret = (unsigned char *)(buf+offset); offset += 1;   if (offset > size) { error=1; return 0; } return ret; }
        const unsigned short *n16() const { unsigned short *ret = (unsigned short*)(buf+offset); offset += 2;   if (offset > size) { error=1; return 0; } return ret; }
        const unsigned       *n32() const { unsigned       *ret = (unsigned      *)(buf+offset); offset += 4;   if (offset > size) { error=1; return 0; } return ret; }
        const char           *get(int len=0) const { char  *ret = (char          *)(buf+offset); offset += len; if (offset > size) { error=1; return 0; } return ret; }

        void String(const char *buf, int len) { char *v = (char*)get(len); if (v) memcpy(v, buf, len); }
        void String(const string &in) { char *v = (char*)get(in.size()); if (v) memcpy(v, in.c_str(), in.size()); }

        void N8 (const unsigned char  &in) { unsigned char  *v =                 n8();  if (v) *v = in; }
        void N8 (const          char  &in) {          char  *v = (char*)         n8();  if (v) *v = in; }
        void N16(const unsigned short &in) { unsigned short *v =                 n16(); if (v) *v = in; }
        void N16(const          short &in) {          short *v = (short*)        n16(); if (v) *v = in; }
        void N32(const unsigned int   &in) { unsigned int   *v =                 n32(); if (v) *v = in; }
        void N32(const          int   &in) {          int   *v = (int*)          n32(); if (v) *v = in; }
        void N32(const unsigned long  &in) { unsigned long  *v = (unsigned long*)n32(); if (v) *v = in; }
        void N32(const          long  &in) {          long  *v = (long*)         n32(); if (v) *v = in; }

        void Ntohs(const unsigned short &in) { unsigned short *v =         n16(); if (v) *v = ntohs(in); }
        void Htons(const unsigned short &in) { unsigned short *v =         n16(); if (v) *v = htons(in); }
        void Ntohs(const          short &in) {          short *v = (short*)n16(); if (v) *v = ntohs(in); }
        void Htons(const          short &in) {          short *v = (short*)n16(); if (v) *v = htons(in); }
        void Ntohl(const unsigned int   &in) { unsigned int   *v =         n32(); if (v) *v = ntohl(in); }
        void Htonl(const unsigned int   &in) { unsigned int   *v =         n32(); if (v) *v = htonl(in); }
        void Ntohl(const          int   &in) {          int   *v = (int*)  n32(); if (v) *v = ntohl(in); }
        void Htonl(const          int   &in) {          int   *v = (int*)  n32(); if (v) *v = htonl(in); }

        void Htons(unsigned short *out) const { const unsigned short *v =         n16(); *out = v ? htons(*v) : 0; }
        void Ntohs(unsigned short *out) const { const unsigned short *v =         n16(); *out = v ? ntohs(*v) : 0; }
        void Htons(         short *out) const { const          short *v = (short*)n16(); *out = v ? htons(*v) : 0; }
        void Ntohs(         short *out) const { const          short *v = (short*)n16(); *out = v ? ntohs(*v) : 0; }
        void Htonl(unsigned int   *out) const { const unsigned int   *v =         n32(); *out = v ? htonl(*v) : 0; }
        void Ntohl(unsigned int   *out) const { const unsigned int   *v =         n32(); *out = v ? ntohl(*v) : 0; }
        void Htonl(         int   *out) const { const          int   *v = (int*)  n32(); *out = v ? htonl(*v) : 0; }
        void Ntohl(         int   *out) const { const          int   *v = (int*)  n32(); *out = v ? ntohl(*v) : 0; }

        void N8 (unsigned char  *out) const { const unsigned char  *v =                 n8();  *out = v ? *v : 0; }
        void N8 (         char  *out) const { const          char  *v = (char*)         n8();  *out = v ? *v : 0; }
        void N16(unsigned short *out) const { const unsigned short *v =                 n16(); *out = v ? *v : 0; }
        void N16(         short *out) const { const          short *v = (short*)        n16(); *out = v ? *v : 0; }
        void N32(unsigned int   *out) const { const unsigned int   *v =                 n32(); *out = v ? *v : 0; }
        void N32(         int   *out) const { const          int   *v = (int*)          n32(); *out = v ? *v : 0; }
        void N32(unsigned long  *out) const { const unsigned long  *v = (unsigned long*)n32(); *out = v ? *v : 0; }
        void N32(         long  *out) const { const          long  *v = (long*)         n32(); *out = v ? *v : 0; }
    };

    struct ConstStream : public Stream {
        ConstStream(const char *B, int S) : Stream((char*)B, S) {}
        char           *end()          { FATAL(this, ": ConstStream write"); return 0; }
        unsigned char  *n8()           { FATAL(this, ": ConstStream write"); return 0; }
        unsigned short *n16()          { FATAL(this, ": ConstStream write"); return 0; }
        unsigned       *n32()          { FATAL(this, ": ConstStream write"); return 0; }
        char           *get(int len=0) { FATAL(this, ": ConstStream write"); return 0; }
    };

    struct MutableStream : public Stream {
        MutableStream(char *B, int S) : Stream(B, S) {}
        char           *end() { return buf + size; }
        unsigned char  *n8()  { unsigned char  *ret = (unsigned char *)(buf+offset); offset += 1;   if (offset > size) { error=1; return 0; } return ret; }
        unsigned short *n16() { unsigned short *ret = (unsigned short*)(buf+offset); offset += 2;   if (offset > size) { error=1; return 0; } return ret; }
        unsigned       *n32() { unsigned       *ret = (unsigned      *)(buf+offset); offset += 4;   if (offset > size) { error=1; return 0; } return ret; }
        char           *get(int len=0) { char  *ret = (char          *)(buf+offset); offset += len; if (offset > size) { error=1; return 0; } return ret; }
    };
};

struct ProtoHeader {
    int flag, len; 
    ProtoHeader() : len(0) { set_flag(0); }
    ProtoHeader(int f) : len(0) { set_flag(f); }
    ProtoHeader(const char *text) {
        memcpy(&flag, text, sizeof(int));
        memcpy(&len, text+sizeof(int), sizeof(int));
        validate();
    }
    void validate() const { if (((flag>>16)&0xffff) != magic) FATAL("magic check"); }
    void set_len(int v) { validate(); len = v; }
    void set_flag(unsigned short v) { flag = (magic<<16) | v; }
    unsigned short get_flag() const { return flag & 0xffff; }
    static const int size = sizeof(int)*2, magic = 0xfefe;
};

/* file types */

struct ProtoFile {
    File *file; int read_offset, write_offset; bool done;

    ~ProtoFile() { delete file; }
    ProtoFile(const char *fn=0) : file(0) { open(fn); }

    bool opened() { return file && file->opened(); }
    void open(const char *fn);
    int add(const Proto *msg, int status);
    bool update(int offset, const ProtoHeader *ph, const Proto *msg);
    bool update(int offset, int status);
    bool get(Proto *out, int offset, int status=-1);
    bool next(Proto *out, int *offsetOut=0, int status=-1);
    bool next(ProtoHeader *hdr, Proto *out, int *offsetOut=0, int status=-1);
};

struct StringFile {
    ReallocHeap b; char **line; int lines;

    StringFile() { clear(); }
    ~StringFile() { clear(); }
    void clear() { b.reset(); line=0; lines=0; }
    void print(const char *name, bool nl=1);

    static int read(string &hdrout, StringFile *out, const char *path, int header=1);
    static int read(string &hdrout, StringFile *out, IterWordIter *word, int header);

    static int writeRow(File *file, const char *rowval);
    static int write(const string &hdr, const vector<string> &out, File *file, const char *name);
    static int write(const string &hdr, const vector<string> &out, const char *path, const char *name);

    static int ReadFile(const char *Dir, const char *Class, const char *Var, StringFile *out, string *flagOut, int iteration=-1);
    static int WriteFile(const char *Dir, const char *Class, const char *Var, const vector<string> &model, int iteration, const char *transcript=0);
};

struct SettingsFile {
    static const char *VarType() { return "string"; }
    static const char *VarName() { return "settings"; }
    static const char *Separator() { return " = "; }

    static int read(const char *dir, const char *name);
    static int write(const vector<string> &fields, const char *dir, const char *name);
};

struct MatrixFile {
    Matrix *F; string T;

    ~MatrixFile() { delete F; }
    MatrixFile() { clear(); }
    MatrixFile(Matrix *features, const char *transcript) : F(features), T(transcript) {}

    void clear() { F=0; T.clear(); }
    const char *text() { return T.c_str(); }

    struct Header { enum { NONE=0, DIM_PLUS=1, DIM=2 }; };
    int read(const char *path, int header=1) { return read(T, &F, path, header); }
    int read(IterWordIter *word, int header) { return read(T, &F, word, header); }
    int read_binary(const char *path) { return read_binary(T, &F, path); }

    int write(const char *path, const char *name) { return write(T, F, path, name); }
    int write(File *file, const char *name) { return write(T, F, file, name); }

    struct BinaryHeader{ int magic, M, N, name, transcript, data, unused1, unused2; };
    int write_binary(const char *path, const char *name) { return write_binary(T, F, path, name); }
    int write_binary(File *file, const char *name) { return write_binary(T, F, file, name); }

    static string filename(const char *Class, const char *Var, const char *Suffix, int iteration);
    static int findHighestIteration(const char *Dir, const char *Class, const char *Var, const char *Suffix);
    static int findHighestIteration(const char *Dir, const char *Class, const char *Var, const char *Suffix1, const char *Suffix2);

    static int readHeader(string &hdrout, IterWordIter *word);
    static int readDimensions(IterWordIter *word, int *M, int *N);
    static int read(string &hdrout, Matrix **out, const char *path, int header=1, int (*IsSpace)(int)=0);
    static int read(string &hdrout, Matrix **out, IterWordIter *word, int header);
    static int read_binary(string &hdrout, Matrix **out, const char *path);

    static int writeHeader(File *file, const char *name, const char *hdr, int M, int N);
    static int writeBinaryHeader(File *file, const char *name, const char *hdr, int M, int N);
    static int writeRow(File *file, const double *row, int N, bool lastrow=0);
    static int write(const string &hdr, Matrix *out, File *file, const char *name);
    static int write(const string &hdr, Matrix *out, const char *path, const char *name);
    static int write_binary(const string &hdr, Matrix *out, File *file, const char *name);
    static int write_binary(const string &hdr, Matrix *out, const char *path, const char *name);

    static int ReadFile(const char *Dir, const char *Class, const char *Var, Matrix **modelOut, string *flagOut, int iteration=-1);
    static int WriteFile(const char *Dir, const char *Class, const char *Var, Matrix *model, int iteration, const char *transcript=0);
    static int WriteFileBinary(const char *Dir, const char *Class, const char *Var, Matrix *model, int iteration, const char *transcript=0);
};

struct MatrixArchiveOut {
    File *file;

    ~MatrixArchiveOut();
    MatrixArchiveOut(const char *name=0);

    void close();
    int open(const char *name);
    int open(string name) { return open(name.c_str()); }
    int write(const string &hdr, Matrix*, const char *name);
    int write(const MatrixFile *f, const char *name) { return write(f->T, f->F, name); }
};

struct MatrixArchiveIn {
    IterWordIter *file; int index;

    ~MatrixArchiveIn();
    MatrixArchiveIn(const char *name=0);

    void close();
    int open(const char *name);
    int open(string name) { return open(name.c_str()); }
    int read(string &hdrout, Matrix **out);
    int read(MatrixFile *f) { return read(f->T, &f->F); }
    int skip();
    const char *filename();
    static int count(const char *name);
};

struct HashMatrix {
#define get_impl(T) \
        unsigned ind = hash % map->M; \
        T *hashrow = map->row(ind); \
        for (int k=0; k<map->N/VPE; k++) { \
            if (hashrow[k*VPE] != hash) continue; \
            return &hashrow[k*VPE]; \
        } \
    return 0;
    static       double *get(/**/  Matrix *map, unsigned hash, int VPE) { get_impl(      double); }
    static const double *get(const Matrix *map, unsigned hash, int VPE) { get_impl(const double); }

    static double *set(Matrix *map, unsigned hash, int VPE) {
        unsigned ind = hash % map->M;
        double *hashrow = map->row(ind);
        for (int k=0; k<map->N/VPE; k++) {
            if (hashrow[k*VPE]) {
                if (hashrow[k*VPE] == hash) { ERROR("hash collision or duplicate insert ", hash); break; }
                continue;
            }
            hashrow[k*VPE] = hash;
            return &hashrow[k*VPE];
        }
        return 0;
    }
    static double *set_binary(File *lf, int M, int N, int hdr_size, unsigned hash, int VPE, double *hashrow) {
        unsigned ind = hash % M, row_size = N * sizeof(double), offset = hdr_size + ind * row_size;
        if (lf->seek(offset, File::Whence::SET) != offset) { ERROR("seek: ", offset);   return 0; } 
        if (lf->read(hashrow, row_size)       != row_size) { ERROR("read: ", row_size); return 0; }

        for (int k=0; k<N/VPE; k++) {
            if (hashrow[k*VPE]) {
                if (hashrow[k*VPE] == hash) { ERROR("hash collision or duplicate insert ", hash); break; }
                continue;
            }
            int hri_offset = offset + k*VPE * sizeof(double);
            if (lf->seek(hri_offset, File::Whence::SET) != hri_offset) { ERROR("seek: ", hri_offset); return 0; } 
            hashrow[k*VPE] = hash;
            return &hashrow[k*VPE];
        }
        return 0;
    }
    static void set_binary_flush(File *lf, int VPE, const double *hashrow) {
        int write_size = VPE * sizeof(double);
        if (lf->write(hashrow, write_size) != write_size) ERROR("read: ", write_size);
    }
};

struct HashMatrix64 {
#define hash_split(hash, hhash, lhash) unsigned hhash = hash>>32, lhash = hash&0xffffffff;
#undef  get_impl
#define get_impl(T) \
        unsigned ind = hash % map->M; \
        T *hashrow = map->row(ind); \
        for (int k=0; k<map->N/VPE; k++) { \
            int hri = k*VPE; \
            if (hashrow[hri] != hhash || hashrow[hri+1] != lhash) continue; \
            return &hashrow[hri]; \
        } \
    return 0;
    static double *get(Matrix *map, unsigned long long hash, int VPE) {
        hash_split(hash, hhash, lhash);
        get_impl(double);
    }
    static const double *get(const Matrix *map, unsigned long long hash, int VPE) {
        hash_split(hash, hhash, lhash);
        get_impl(const double);
    }

    static double *set(Matrix *map, unsigned long long hash, int VPE) {
        hash_split(hash, hhash, lhash);
        long long ind = hash % map->M;
        double *hashrow = map->row(ind);
        for (int k=0; k<map->N/VPE; k++) {
            int hri = k*VPE;
            if (hashrow[hri]) {
                if (hashrow[hri] == hhash && hashrow[hri+1] == lhash) { ERROR("hash collision or duplicate insert ", hhash, " ", lhash); break; }
                continue;
            }
            hashrow[hri] = hhash;
            hashrow[hri+1] = lhash;
            return &hashrow[hri];
        }
        return 0;
    }

    static double *set_binary(File *lf, int M, int N, int hdr_size, unsigned long long hash, int VPE, double *hashrow) {
        hash_split(hash, hhash, lhash);
        long long ind = hash % M, row_size = N * sizeof(double), offset = hdr_size + ind * row_size, ret;
        if ((ret = lf->seek(offset, File::Whence::SET)) != offset) { ERROR("seek: ", offset,   " != ", ret); return 0; } 
        if ((ret = lf->read(hashrow, row_size))       != row_size) { ERROR("read: ", row_size, " != ", ret); return 0; }

        for (int k=0; k<N/VPE; k++) {
            int hri = k*VPE;
            if (hashrow[hri]) {
                if (hashrow[hri] == hhash && hashrow[hri+1] == lhash) { ERROR("hash collision or duplicate insert ", hhash, " ", lhash); break; }
                continue;
            }
            int hri_offset = offset + hri * sizeof(double);
            if ((ret = lf->seek(hri_offset, File::Whence::SET)) != hri_offset) { ERROR("seek: ", hri_offset, " != ", ret); return 0; } 
            hashrow[hri] = hhash;
            hashrow[hri+1] = lhash;
            return &hashrow[hri];
        }
        return 0;
    }

    static void set_binary_flush(LocalFile *lf, int VPE, const double *hashrow) {
        int write_size = VPE * sizeof(double);
        if (lf->write(hashrow, write_size) != write_size) ERROR("read: ", write_size);
    }
};
#undef get_impl

}; // namespace LFL
#endif // __LFL_LFAPP_LFTYPES_H__
