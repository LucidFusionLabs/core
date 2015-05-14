/*
 * $Id: judymap.h 743 2013-09-19 12:36:55Z justin $
 *
 * STL style container for http://judy.sourceforge.net/
 *
 * This code is free as in beer.
 */

#ifndef __LFL_LFAPP_JUDY_H__
#define __LFL_LFAPP_JUDY_H__

/* DESCRIPTION: 
Implements std::map according to http: //www.cplusplus.com/reference/stl/map/
EXCEPT FOR: operator=, swap, rbegin, rend, key_comp, value_comp, erase(beg, end), equal_range, get_alloactor, and max_size

-* QUICK START:
using namespace std;

#ifdef USING_JUDY
#define Map(K, V) judymap<K, V>
#else
#define Map(K, V) map<K, V>
#endif

-* EXAMPLE:

typedef judysmap<string, int> map_t;

map_t test;
test["cat"] = 33;
test["dog"] = 55;

for (map_t::iterator i = test.begin(); i != test.end(); i++) {
    INFO("%s %d\n", (*i).first.c_str(), (*i).second);
    if (i == test.begin()) (*i).second = 99;
}

for (map_t::const_iterator i = test.begin(); i != test.end(); i++)
    INFO("%s %d\n", (*i).first.c_str(), (*i).second);
*/

#ifndef StaticAssert /* from Modern C++ Design: Generic Programming and Design Patterns Applied by Andrei Alexandrescu */
template <bool> class StaticAssertCheck { public: StaticAssertCheck(...) {} };
template <> class StaticAssertCheck<false> {};
#define StaticAssert(test, errormsg) do { \
    struct ERROR_ ## errormsg {}; \
    typedef StaticAssertCheck<(test)!=0> assertCheck; \
    assertCheck check = assertCheck(ERROR_ ## errormsg()); \
    if (sizeof(check)) {}; \
} while(0)
#endif

struct judymap_impl {
    void *impl;

    judymap_impl() : impl(0) {}
    ~judymap_impl();

    void clear();
    int remove(void *k);
    void set(void *k, void *v);
    void **set(void *k);
    void **get(void *k);
    int size() const;

    struct iterator {
        judymap_impl *impl;
        void *k, **v;
        bool done;

        iterator(judymap_impl *Impl) : impl(Impl), k(0), v(0), done(0) {}
        iterator(judymap_impl *Impl, void *K) : impl(Impl), k(K) { done = !(v = impl->get(k)); }
        bool operator==(const iterator &r) const { return (done && r.done) || (!done && !r.done && k == r.k); }
        void operator++(int) { impl->next(this); }
    };

    void begin(iterator *iter) const;
    void next(iterator *iter) const;
    void next(iterator *iter, void *k) const;
    void assignKV(iterator *iter, void *arg) const;
};

template <class X, class Y> struct judymap {
    judymap_impl impl;
    judymap() {
        StaticAssert(sizeof(X) <= sizeof(void *), INVALID_KEY_TYPE);
        StaticAssert(sizeof(Y) <= sizeof(void *), INVALID_VALUE_TYPE);
    }

    typedef judymap<X, Y> this_type;
    typedef X             key_type;
    typedef Y             value_type;

    struct pair {
        key_type first;
        value_type &second;
        pair(key_type k, value_type *v) : first(k), second(*v) {}
    };

    struct const_pair {
        const key_type first;
        const value_type &second;
        const_pair(key_type k, value_type *v) : first(k), second(*v) {}
    };

    struct const_iterator;

    struct iterator {
        judymap_impl::iterator impl;

        iterator()                                   : impl(0) {}
        iterator(const iterator &r)                  : impl(r.impl) {}
        iterator(const this_type *parent, void *k=0) : impl((judymap_impl*)&parent->impl, k) {}

        iterator& operator=(const iterator &r) { impl=r.impl; return *this; }

        pair       operator*()       { return       pair((key_type)(long long)impl.k, (value_type*)impl.v); } /* non const dereference */
        const_pair operator*() const { return const_pair((key_type)(long long)impl.k, (value_type*)impl.v); }

        bool operator==(const       iterator &r) const { return   impl == r.impl;  }
        bool operator!=(const       iterator &r) const { return !(impl == r.impl); }
        bool operator==(const const_iterator &r) const { return   impl == r.impl;  }
        bool operator!=(const const_iterator &r) const { return !(impl == r.impl); }

        void operator++()    { impl++; }
        void operator++(int) { impl++; }
    };

    struct const_iterator {
        judymap_impl::iterator impl;
        
        const_iterator()                                   : impl(0) {}
        const_iterator(const const_iterator &r)            : impl(r.impl) {}
        const_iterator(const iterator &r)                  : impl(r.impl) {} /* cross copy constructor */
        const_iterator(const this_type *parent, void *k=0) : impl((judymap_impl*)&parent->impl, k) {}

        const_iterator& operator=(const const_iterator &r) { impl=r.impl; return *this; }

        const_pair operator*() const { return const_pair((key_type)(long)impl.k, (value_type*)impl.v); }

        bool operator==(const const_iterator &r) const { return   impl == r.impl;  }
        bool operator!=(const const_iterator &r) const { return !(impl == r.impl); }
        bool operator==(const       iterator &r) const { return   impl == r.impl;  }
        bool operator!=(const       iterator &r) const { return !(impl == r.impl); }

        void operator++()    { impl++; }
        void operator++(int) { impl++; }
    };

    int size() const { return impl.size(); }
    bool empty() const { return !size(); }

    void clear() { impl.clear(); }

    iterator       begin()       { iterator       iter(this); impl.begin(&iter.impl); return iter; }
    const_iterator begin() const { const_iterator iter(this); impl.begin(&iter.impl); return iter; }

    iterator       end()       { iterator       iter(this); iter.impl.done=1; return iter; }
    const_iterator end() const { const_iterator iter(this); iter.impl.done=1; return iter; }

    iterator       find(const key_type &k)       { iterator       iter(this, (void*)(long)k); return iter; }
    const_iterator find(const key_type &k) const { const_iterator iter(this, (void*)(long)k); return iter; }

    iterator       lower_bound(const key_type &k)       { iterator       iter(this, (void*)(long)k); if (!iter.impl.done) return iter; impl.next(&iter.impl, (void*)(long)k); return iter; }
    const_iterator lower_bound(const key_type &k) const { const_iterator iter(this, (void*)(long)k); if (!iter.impl.done) return iter; impl.next(&iter.impl, (void*)(long)k); return iter; }

    iterator       upper_bound(const key_type &k)       { iterator       iter(this, (void*)(long)k);                                   impl.next(&iter.impl, (void*)(long)k); return iter; }
    const_iterator upper_bound(const key_type &k) const { const_iterator iter(this, (void*)(long)k);                                   impl.next(&iter.impl, (void*)(long)k); return iter; }

    void erase(iterator it) { impl.remove(it.impl.k); }
    int  erase(const key_type &k) { return impl.remove((void*)(long)k); }

    value_type& operator[](X k) { return *(value_type *)impl.set((void*)(long)k); }
};

struct judysmap_impl {
    void *impl;

    ~judysmap_impl();
    judysmap_impl() : impl(0) {}

    void clear();
    int remove(const char *);    
    void set(const char *, void *val);
    void **set(const char *);
    void **get(const char *);

    struct iterator {
        judysmap_impl *impl;
        static const int keymax = 1024;
        char k[keymax];
        void **v;
        bool done;

        iterator(judysmap_impl *Impl) : impl(Impl), v(0), done(0) { k[0]=0; }
        iterator(judysmap_impl *Impl, const char *K) : impl(Impl), v(0), done(0) { strncpy(k, K, sizeof(k)); done = !(v = impl->get(k)); }
        iterator(const iterator &r) : impl(r.impl), v(r.v), done(r.done) { strncpy(k, r.k, sizeof(k)); }

        bool operator==(const iterator &r) { return (done && r.done) || (!done && !r.done && !strcmp(k, r.k)); }
        void operator++(int) { impl->next(this); }
    };

    void begin(iterator *) const;
    void next(iterator *) const;
    void next(iterator *, const char *k) const;
    void assignKV(iterator *, void *) const;
};

template <class Y> struct judymap<string, Y> {
    judysmap_impl impl;
    judymap() { 
        StaticAssert(sizeof(Y) <= sizeof(void*), INVALID_VALUE_TYPE);
    }

    typedef judymap<string, Y> this_type;
    typedef string             key_type;
    typedef Y                  value_type;
    
    struct fakestring { /* emulate std::string */
        const char *v;
        fakestring(const char *V=0) : v(V) {}
        const char *data() const { return v; }
        const char *c_str() const { return v; }
        int size() const { return strlen(v); }
        int length() const { return strlen(v); }
        bool empty() const { return !v || !*v; }
        const char& operator[](size_t pos) const { return v[pos]; }
        string substr(size_t pos=0, size_t n=string::npos) const { return string(v, pos, n); } /* assumes using namespace std */
    };
    
    struct pair {
        fakestring first;
        value_type &second;
        pair(const char *k, value_type *v) : first(k), second(*v) {}
    };
    
    struct const_pair {
        const fakestring first;
        const value_type &second;
        const_pair(const char *k, value_type *v) : first(k), second(*v) {}
    };

    struct iterator {
        judysmap_impl::iterator impl;

        iterator()                  : impl(0)      {}
        iterator(const iterator &r) : impl(r.impl) {}

        iterator(const this_type *parent, string k) : impl((judysmap_impl*)&parent->impl, k.c_str()) {}
        iterator(const this_type *parent)           : impl((judysmap_impl*)&parent->impl) {}

        iterator& operator=(const iterator &r) { impl=r.impl; return *this; }

        pair       operator*()       { return       pair(impl.k, (value_type*)impl.v); } /* non const dereference */
        const_pair operator*() const { return const_pair(impl.k, (value_type*)impl.v); }

        bool operator==(const iterator &r) { return   impl == r.impl;  }
        bool operator!=(const iterator &r) { return !(impl == r.impl); }

        void operator++()    { impl++; }
        void operator++(int) { impl++; }
    };

    struct const_iterator {
        judysmap_impl::iterator impl;

        const_iterator()                        : impl(0)      {}
        const_iterator(const const_iterator &r) : impl(r.impl) {}
        const_iterator(const       iterator &r) : impl(r.impl) {} /* cross copy constructor */

        const_iterator(const this_type *parent, string k) : impl((judysmap_impl*)&parent->impl, k.c_str()) {}
        const_iterator(const this_type *parent)           : impl((judysmap_impl*)&parent->impl) {}

        const_iterator& operator=(const const_iterator &r) { impl=r.impl; return *this; }

        const_pair operator*() const { return const_pair(impl.k, (value_type *)impl.v); }

        bool operator==(const const_iterator &r) { return   impl == r.impl;  }
        bool operator!=(const const_iterator &r) { return !(impl == r.impl); }

        void operator++()    { impl++; }
        void operator++(int) { impl++; }
    };

    void clear() { impl.clear(); }

    iterator       begin()       { iterator       iter(this); impl.begin(&iter.impl); return iter; }
    const_iterator begin() const { const_iterator iter(this); impl.begin(&iter.impl); return iter; }

    iterator       end()       { iterator       iter(this); iter.impl.done=1; return iter; }
    const_iterator end() const { const_iterator iter(this); iter.impl.done=1; return iter; }

    iterator       find(const key_type &k)       { iterator       iter(this, k); return iter; }
    const_iterator find(const key_type &k) const { const_iterator iter(this, k); return iter; }

    iterator       lower_bound(const key_type &k)       { iterator       iter(this, k); if (!iter.impl.done) return iter; impl.next(&iter.impl, k.c_str()); return iter; }
    const_iterator lower_bound(const key_type &k) const { const_iterator iter(this, k); if (!iter.impl.done) return iter; impl.next(&iter.impl, k.c_str()); return iter; }

    iterator       upper_bound(const key_type &k)       { iterator       iter(this, k);                                   impl.next(&iter.impl, k.c_str()); return iter; }
    const_iterator upper_bound(const key_type &k) const { const_iterator iter(this, k);                                   impl.next(&iter.impl, k.c_str()); return iter; }

    void erase(iterator it) { impl.remove((*it).first.c_str()); }
    int  erase(const key_type &k) { return impl.remove(k.c_str()); }

    value_type& operator[](const key_type &k) { return *(value_type *)impl.set(k.c_str()); }
};
#endif /* __LFL_LFAPP_JUDY_H__ */

#ifdef LFL_INCLUDE_JUDY_IMPL
#include "Judy.h"

judymap_impl::~judymap_impl() { clear(); }
void judymap_impl::clear() { if (!impl) return; int rc; JLFA(rc, impl); impl=0; }
int judymap_impl::remove(void *k) { Word_t key=(Word_t)k; int rc; JLD(rc, impl, key); return rc; }
void judymap_impl::set(void *k, void *v) { Word_t key=(Word_t)k; void *val; JLI(val, impl, key); *(void**)val = v; }
void **judymap_impl::set(void *k) { Word_t key=(Word_t)k; void *val; JLI(val, impl, key); return (void**)val; }
void **judymap_impl::get(void *k) { Word_t key=(Word_t)k; void *val; JLG(val, impl, key); return (void**)val; }
int judymap_impl::size() const { if (!impl) return 0; int rc; JLC(rc, impl, 0, -1); return rc; }

#define MyJudyPtrMapIter (*(Word_t*)&iter->k)
void judymap_impl::begin(iterator *iter) const { MyJudyPtrMapIter=0; void *val; JLF(val, impl, MyJudyPtrMapIter); assignKV(iter, val); }
void judymap_impl::next(iterator *iter) const { void *val; JLN(val, impl, MyJudyPtrMapIter); assignKV(iter, val); }
void judymap_impl::next(iterator *iter, void *k) const { MyJudyPtrMapIter = (Word_t)k; next(iter); }
void judymap_impl::assignKV(iterator *iter, void *arg) const { if ((iter->done = !arg)) return; iter->k = (void*)MyJudyPtrMapIter; iter->v = (void **)arg; }

judysmap_impl::~judysmap_impl() { clear(); }
void judysmap_impl::clear() { if (!impl) return; int rc; JSLFA(rc, impl); impl=0; }
int judysmap_impl::remove(const char* k) { int rc; JSLD(rc, impl, (const unsigned char *)k); return rc; }
void judysmap_impl::set(const char* k, void *v) { void *val; JSLI(val, impl, (const unsigned char *)k); *(void**)val = v; }
void **judysmap_impl::set(const char *k) { void *val; JSLI(val, impl, (const unsigned char *)k); return (void**)val; }
void **judysmap_impl::get(const char* k) { void *val; JSLG(val, impl, (const unsigned char *)k); return (void**)val; }

#define MyJudyStrMapIter ((unsigned char*)iter->k)
void judysmap_impl::begin(iterator *iter) const { *MyJudyStrMapIter=0; void *val; JSLF(val, impl, MyJudyStrMapIter); assignKV(iter, val); }
void judysmap_impl::next(iterator *iter) const { void *val; JSLN(val, impl, MyJudyStrMapIter); assignKV(iter, val); }
void judysmap_impl::next(iterator *iter, const char *key) const { strncpy((char*)MyJudyStrMapIter, key, sizeof(MyJudyStrMapIter)); next(iter); }
void judysmap_impl::assignKV(iterator *iter, void *arg) const { if ((iter->done = !arg)) return; iter->v = (void **)arg; }
#endif /* LFL_INCLUDE_JUDY_IMPL */
