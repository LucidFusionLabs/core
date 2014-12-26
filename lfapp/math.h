/*
 * $Id: math.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef __LFL_LFAPP_MATH_H__
#define __LFL_LFAPP_MATH_H__
namespace LFL {

template <class X> struct V2 {
    X x, y;
    V2() : x(0), y(0) {}
    V2(X xin, X yin) : x(xin), y(yin) {}
    template <class Y> V2(const V2<Y> &c) : x((X)c.x), y((Y)c.y) {}
    operator X *() { return &x; }
    operator const X *() const { return &x; }
    bool operator == (const V2<X> &r) const { return x == r.x && y == r.y; }
    bool operator != (const V2<X> &r) const { return !(*this == r); }
    V2<X> operator - () const { return V2<X>(-x, -y); }
    V2<X> operator * (X r) const { V2<X> ret=*this; ret *= r; return ret; }
    V2<X> operator * (const V2<X> &dm) const { V2<X> ret=*this; ret.x*=dm.x; ret.y*=dm.y; return ret; }
    V2<X> operator - (const V2<X> &v)  const { V2<X> ret=*this; ret -= v; return ret; }
    V2<X> operator + (const V2<X> &v)  const { V2<X> ret=*this; ret += v; return ret; }
    void operator += (const V2<X> &v) { x += v.x; y += v.y; }
    void operator -= (const V2<X> &v) { x -= v.x; y -= v.y; }
    void operator *= (X f)            { x *= f;   y *= f;   }
    void norm() { float l=len(); if (!l) return; x /= l; y /= l; }
    float len() const { return sqrt(x*x + y*y); }
    string DebugString() const { return StrCat("v(", x, ", ", y, ")"); }
};
typedef V2<float> v2;
typedef V2<int> point;

struct v3 {
    float x, y, z;
    v3() : x(0), y(0), z(0) {}
    v3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
    void set(float X, float Y, float Z) { x=X; y=Y; z=Z; }
    operator float *() { return &x; }
    operator const float *() const { return &x; }
    bool operator == (const v3 &r) const { return x == r.x && y == r.y && z == r.z; }
    bool operator != (const v3 &r) const { return !(*this == r); }
    v3 operator - () const { return v3(-x, -y, -z); }
    v3 operator * (float r) const { v3 ret=*this; ret.scale(r); return ret; }
    v3 operator * (const v3 &dm) const { v3 ret=*this; ret.x*=dm.x; ret.y*=dm.y; ret.z*=dm.z; return ret; }
    v3 operator - (const v3 &v) const { v3 ret=*this; ret.sub(v); return ret; }
    v3 operator + (const v3 &v) const { v3 ret=*this; ret.add(v); return ret; }
    void operator += (const v3 &v) { add(v); }
    void add(v3 v) { x += v.x; y += v.y; z += v.z; }
    void sub(v3 v) { x -= v.x; y -= v.y; z -= v.z; }
    void scale(float f) { x *= f; y *= f; z *= f; }
    float len() { return sqrt(x*x + y*y + z*z); }
    void norm() { float l=len(); if (!l) return; x /= l; y /= l; z /= l; }
    string DebugString() const { return StrCat("v(", x, ", ", y, ", ", z, ")"); }
    static v3 norm(v3 q) { float l=q.len(); if (!l) return q; q.x /= l; q.y /= l; q.z /= l; return q; }
    static float dot(v3 q, v3 p) { return q.x*p.x + q.y*p.y + q.z*p.z; }
    static float dist2(v3 q, v3 p) { v3 d = q - p; return dot(d, d); }
    static v3 cross(v3 q, v3 p) { return v3(q.y*p.z - q.z*p.y, q.z*p.x - q.x*p.z, q.x*p.y - q.y*p.x); }
    static v3 normal(v3 a, v3 b, v3 c) { v3 q=c, p=c; q.sub(a); p.sub(b); q = v3::cross(q, p); q.norm(); return q; }
    static v3 rand() {
        float phi = LFL::rand(0, M_TAU), costheta = LFL::rand(-1, 1), rho = sqrt(1 - pow(costheta, 2));
        return v3(rho*cos(phi), rho*sin(phi), costheta);
    }
};

struct v4 {
    float x, y, z, w;
    v4() : x(0), y(0), z(0), w(0) {}
    v4(const v3& xyz, float W) : x(xyz.x), y(xyz.y), z(xyz.z), w(W) {}
    v4(float X, float Y, float Z, float W) : x(X), y(Y), z(Z), w(W) {}
    v4(const float *v) : x(v[0]), y(v[1]), z(v[2]), w(v[3]) {}
    v4 operator * (float r) { v4 ret=*this; ret.scale(r); return ret; }
    v4 operator * (const v4 &dm) { v4 ret=*this; ret.x*=dm.x; ret.y*=dm.y; ret.z*=dm.z; ret.w*=dm.w; return ret; }
    v4 operator / (float r) { v4 ret=*this; ret.scale(1/r); return ret; }
    v4 operator + (const v4 &v) { v4 ret=*this; ret.add(v); return ret; }
    v4 operator - (const v4 &v) { v4 ret=*this; ret.sub(v); return ret; }
    void add(v4 v) { x += v.x; y += v.y; z += v.z; w += v.w; }
    void sub(v4 v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; }
    void scale(float f) { x *= f; y *= f; z *= f; w *= f; }
    float len() { return sqrt(x*x + y*y + z*z + w*w); }
    void norm() { float l=len(); if (!l) return; scale(1/l); }
    v3 xyz() const { return v3(x, y, z); }
    operator float *() { return &x; }
    operator const float *() const { return &x; }
    bool operator<(const v4 &c) const;
};

struct m33 {
    v3 m[3]; m33() {}
    v3 &operator[](unsigned i) { return m[i]; }
    const v3 &operator[](unsigned i) const { return m[i]; }
    static m33 rotaxis(float f, float x, float y, float z) {
        float s=sin(f), c=cos(f); m33 m;
        m[0][0] =    c + x*x*(1-c); m[1][0] =  z*s + y*x*(1-c); m[2][0] = -y*s + z*x*(1-c);
        m[0][1] = -z*s + x*y*(1-c); m[1][1] =    c + y*y*(1-c); m[2][1] =  x*s + z*y*(1-c);
        m[0][2] =  y*s + x*z*(1-c); m[1][2] = -x*s + y*z*(1-c); m[2][2] =    c + z*z*(1-c);
        return m;
    }
    v3 transform(const v3 &v) const {
        v3 ret;
        ret.x=m[0][0]*v.x + m[1][0]*v.y + m[2][0]*v.z;
        ret.y=m[0][1]*v.x + m[1][1]*v.y + m[2][1]*v.z;
        ret.z=m[0][2]*v.x + m[1][2]*v.y + m[2][2]*v.z;
        return ret;
    }
};

struct m44 {
    v4 m[4]; m44() {}
    m44(const int *in) { assign(in); }
    m44(const float *in) { assign(in); }
    m44(const m44 &in) { assign(in); }
    m44(const m33 &in) { assign(in); }
    v4 &operator[](unsigned i) { return m[i]; }
    const v4 &operator[](unsigned i) const { return m[i]; }
    void assign(const m44   &in) { for (int i=0; i<4; i++) m[i] = in.m[i]; }
    void assign(const m33   &in) { for (int i=0; i<3; i++) m[i] = v4(in[i], 0); m[3] = v4(0,0,0,1); }
    void assign(const float *in) { int k=0; for (int i=0; i<4; i++) for (int j=0; j<4; j++, k++) m[i][j] = in[k]; }
    void assign(const int   *in) { int k=0; for (int i=0; i<4; i++) for (int j=0; j<4; j++, k++) m[i][j] = in[k]; }
    void mult(const m44 &in) { m44 result; mult(in, *this, &result); assign(result); }
    void Print(const string &name) const { INFOf("%s = { %f,%f,%f,%f, %f,%f,%f,%f, %f,%f,%f,%f, %f,%f,%f,%f }\n", name.c_str(),
                                               m[0][0], m[0][1], m[0][2], m[0][3],
                                               m[1][0], m[1][1], m[1][2], m[1][3],
                                               m[2][0], m[2][1], m[2][2], m[2][3],
                                               m[3][0], m[3][1], m[3][2], m[3][3]); }
    v4 transform(const v4 &v) const {
        v4 ret;
        ret.x=m[0][0]*v.x + m[1][0]*v.y + m[2][0]*v.z + m[3][0]*v.w;
        ret.y=m[0][1]*v.x + m[1][1]*v.y + m[2][1]*v.z + m[3][1]*v.w;
        ret.z=m[0][2]*v.x + m[1][2]*v.y + m[2][2]*v.z + m[3][2]*v.w;
        ret.w=m[0][3]*v.x + m[1][3]*v.y + m[2][3]*v.z + m[3][3]*v.w;
        return ret;
    }
    static void mult(const m44 &A, const m44 &B, m44 *C) {
        for (int i=0; i<4; i++) for (int j=0; j<4; j++) { ((*C)[i])[j] = 0; for (int k=0; k<4; k++) ((*C)[i])[j] += (A[i])[k] * (B[k])[j]; }
    }
    static m44 identity() { float v[] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }; return m44(v); };
    static m44 translate(float x, float y, float z) { float v[] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1 }; return m44(v); }
    static m44 scale(float x, float y, float z) { float v[] = { x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1 }; return m44(v); }
    static m44 rotate(float angle, float x, float y, float z) { return m44(m33::rotaxis(angle, x, y, z)); }
    static m44 ortho(float l, float r, float b, float t, float nv, float fv) {
        float v[] = { 2/(r-l),0,0,0, 0,2/(t-b),0,0, 0,0,-2/(nv-fv),0, -(r+l)/(r-l), -(t+b)/(t-b), -(fv+nv)/(fv-nv), 1 };
        return m44(v);
    }
    static m44 frustum(float l, float r, float b, float t, float nv, float fv) { 
        float v[] = { 2*nv/(r-l),0,0,0, 0,2*nv/(t-b),0,0, (r+l)/(r-l), (t+b)/(t-b), -(fv+nv)/(fv-nv), -1, 0,0,-2*fv*nv/(fv-nv),0 };
        return m44(v);
    }
};

struct Plane {
    float a, b, c, d;
    Plane(v3 pos, v3 ort) { From(pos, ort); }
    Plane(v3 p1, v3 p2, v3 p3) { v3 norm = Normal(p1, p2, p3); From(p1, norm); }
    Plane(float a, float b, float c, float d) { Assign(a, b, c, d); }

    void Assign(float A, float B, float C) { a=A; b=B; c=C; }
    void Assign(float A, float B, float C, float D) { a=A; b=B; c=C; d=D; }
    void From(v3 pos, v3 ort) {
        ort.norm();
        Assign(ort.x, ort.y, ort.z);
        d = -a*pos.x -b*pos.y -c*pos.z;
    }
    float Distance(v3 p, bool norm=true) {
        float num = (a*p.x + b*p.y + c*p.z + d);
        if (!norm) return num;
        return num / sqrt(a*a + b*b + c*c);
    }
    static v3 Normal(v3 p1, v3 p2, v3 p3) { return v3::cross(p3-p2, p1-p2); }
};

#undef Complex
struct Complex {
    double r, i;

    void Add(Complex n) { r += n.r; i += n.i; }
    void Sub(Complex n) { r -= n.r; i -= n.i; }
    void Mult(Complex n) {
        Complex l = *this;
        r = l.r*n.r - l.i*n.i;
        i = l.r*n.i + l.i*n.r;
    }
    Complex Conjugate() const { Complex ret = {r,-i}; return ret; }

    static Complex Add(Complex a, Complex b) { a.Add(b); return a; }
    static Complex Sub(Complex a, Complex b) { a.Sub(b); return a; }
    static Complex Mult(Complex a, Complex b) { a.Mult(b); return a; }
};

double logadd(double *, double);
float logadd(float *, float);
unsigned logadd(unsigned *, unsigned);

template <class X> struct Vec {
    virtual ~Vec() {}
    virtual int Len() const = 0;
    virtual X Read(int i) const = 0;

    static string Str(const X *v1,              int D) { string line; for (int i=0; i<D; i++) line += (line.size() ? ", " : "") + Typed::Str(v1[i]); return line; }
    static void Print(const X *v1,              int D) { INFO(Str(v1, D)); }
    static X    Dist2(const X *v1, const X *v2, int D) { X ret=0; for (int i=0; i<D; i++) ret += squared(v1[i]-v2[i]); return ret; }
    static X      Dot(const X *v1, const X *v2, int D) { X ret=0; for (int i=0; i<D; i++) ret += v1[i] * v2[i];        return ret; }
    static void Add (const X *v1, const X *v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] + v2[i]; }
    static void Add (const X *v1,       X  v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] + v2;    }
    static void Sub (const X *v1, const X *v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] - v2[i]; }
    static void Sub (const X *v1,       X  v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] - v2;    }
    static void Mult(const X *v1, const X *v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] * v2[i]; }
    static void Mult(const X *v1,       X  v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] * v2;    }
    static void Div (const X *v1, const X *v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] / v2[i]; }
    static void Div (const X *v1,       X  v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] / v2;    }
    static void Add   (X *v1, const X *v2, int D) { return Add (v1,v2,v1,D); }
    static void Add   (X *v1,       X  v2, int D) { return Add (v1,v2,v1,D); }
    static void Sub   (X *v1, const X *v2, int D) { return Sub (v1,v2,v1,D); }
    static void Sub   (X *v1,       X  v2, int D) { return Sub (v1,v2,v1,D); }
    static void Mult  (X *v1, const X *v2, int D) { return Mult(v1,v2,v1,D); }
    static void Mult  (X *v1,       X  v2, int D) { return Mult(v1,v2,v1,D); }
    static void Div   (X *v1, const X *v2, int D) { return Div (v1,v2,v1,D); }
    static void Div   (X *v1,       X  v2, int D) { return Div (v1,v2,v1,D); }
    static void Exp   (X *v1, const X *v2, int D) { for (int i=0; i<D; i++) v1[i] = ::exp(v2[i]); }
    static void Assign(X *v1, const X *v2, int D) { for (int i=0; i<D; i++) v1[i] = v2[i]; }
    static void Assign(X *v1,       X  v2, int D) { for (int i=0; i<D; i++) v1[i] = v2; }
    static void Log   (X *v1, const X *v2, int D) { for (int i=0; i<D; i++) v1[i] = ::log(v2[i]); }
    static void LogAdd(X *v1, const X *v2, int D) { for (int i=0; i<D; i++) LFL::logadd(&v1[i], v2[i]); }
    static void Exp   (      X *v1, int D) { exp(v1, v1, D); }
    static void Log   (      X *v1, int D) { log(v1, v1, D); }
    static double Sum (const X *v1, int D) { X ret=0; for (int i=0; i<D; i++) ret += v1[i]; return ret; }
    static double Min (const X *v1, int D) { X ret=(X) INFINITY; for (int i=0; i<D; i++) if (v1[i] < ret) ret = v1[i]; return (double)ret; }
    static double Max (const X *v1, int D) { X ret=(X)-INFINITY; for (int i=0; i<D; i++) if (v1[i] > ret) ret = v1[i]; return (double)ret; }
    static int  MaxInd(const X *v1, int D) { X mx =(X)-INFINITY; int ret=-1; for (int i=0; i<D; i++) if (v1[i] > mx) { mx = v1[i]; ret = i; } return ret; }
};
typedef Vec<double> Vector;

enum { mTrnpA=1, mTrnpB=1<<1, mTrnpC=1<<2, mDelA=1<<3, mDelB=1<<4, mZeroOnly=1<<5, mNeg=1<<6 };
#define MatrixRowIter(m) for(int i=0;i<(m)->M;i++)
#define MatrixColIter(m) for(int j=0;j<(m)->N;j++)
#define MatrixIter(m) MatrixRowIter(m) MatrixColIter(m)

template <class T=double> struct matrix {
    struct Flag { enum { Complex=1 }; int f; };
    int M, N, flag; long long bytes;
    Allocator *alloc;
    T *m;

    matrix() : M(0), N(0), flag(0), bytes(0), alloc(Singleton<MallocAlloc>::Get()), m(0) {}
    matrix(int Mrows, int Ncols, T InitialVal=T(), int Flag=0, Allocator *Alloc=0) : m(0) { Open(Mrows, Ncols, InitialVal, Flag, Alloc); }
    matrix(const matrix<T> &copy) : m(0), alloc(0) { Open(copy.M, copy.N, T(), copy.flag); AssignL(&copy); }
    matrix(const char *bitmap, int Mrows, int Ncols, int Flag=0, int Chans=1, int ChanInd=0, Allocator *Alloc=0) : m(0) { Open(Mrows, Ncols, bitmap, Flag, Alloc); }
    matrix& operator=(const matrix &copy) { Open(copy.M, copy.N, T(), copy.flag); AssignL(&copy); return *this; }
    virtual ~matrix() { alloc->free(m); }

    virtual T             *row(int i)        { if (i>=M||i<0) return 0; return &m[i*N*((flag&Flag::Complex)?2:1)]; }
    virtual const T       *row(int i) const  { if (i>=M||i<0) return 0; return &m[i*N*((flag&Flag::Complex)?2:1)]; }
    virtual Complex       *crow(int i)       { if (i>=M||i<0) return 0; return (Complex*)&m[i*N*2]; }
    virtual const Complex *crow(int i) const { if (i>=M||i<0) return 0; return (Complex*)&m[i*N*2]; }

    virtual T       *lastrow()       { return row(M-1); }
    virtual const T *lastrow() const { return row(M-1); }
 
    void AddRows(int rows, bool prepend=false) {
        CHECK_GE(M+rows, 0);
        M += rows;
        bytes = M*N*sizeof(T)*((flag&Flag::Complex)?2:1);
        void *pre = m;
        if (!alloc) FATAL("null alloc: ", alloc, ", ", rows, ", ", bytes);
        if (!(m = (T*)alloc->realloc(m, bytes))) FATALf("matrix %s failed: %p (%p) %lld (%d * %d * %lld)", alloc->name(), m, pre, bytes, M, N, bytes/(M*N));
        if (rows > 0) {
            if (prepend) {
                int rowbytes = bytes/M;
                memmove((char*)m + rowbytes*rows, m, rowbytes*(M-rows));
                for (int i=0; i<rows; i++) { MatrixColIter(this) { row(i)[j] = T(); } }
            } else {
                for (int i=M-rows; i<M; i++) { MatrixColIter(this) { row(i)[j] = T(); } }
            }
        }
    }
    void AddCols(unsigned cols, bool prepend=false) {
        matrix<T> new_matrix(M, N+cols, 0, flag, alloc);
        if (!prepend) new_matrix.AssignR(this);
        else MatrixIter(this) new_matrix.row(i)[j+cols] = row(i)[j];
        Typed::Swap(m, new_matrix.m);
        Typed::MinusPlus(&new_matrix.N, &N, (int)cols);
        bytes = M*N*sizeof(T)*((flag&Flag::Complex)?2:1);
    }

    void Assign(int Mrows, int Ncols, long long Bytes, int Flag, Allocator *Alloc) { M=Mrows; N=Ncols; bytes=Bytes; flag=Flag; alloc=Alloc; }
    void AssignL(const matrix *m) { MatrixIter(this) row(i)[j] = m->row(i)[j]; }
    void AssignR(const matrix *m) { MatrixIter(m)    row(i)[j] = m->row(i)[j]; }
    void AssignL(const matrix *m, int flag) { bool neg=flag&mNeg; MatrixIter(this) row(i)[j] = neg ? Typed::Negate(m->row(i)[j]) : m->row(i)[j]; CompleteOperation(m, 0, 0, flag); }
    void AssignR(const matrix *m, int flag) { bool neg=flag&mNeg; MatrixIter(m)    row(i)[j] = neg ? Typed::Negate(m->row(i)[j]) : m->row(i)[j]; CompleteOperation(m, 0, 0, flag); }
    void AssignDiagonal(double v) { MatrixRowIter(this) row(i)[i] = v; }
    void Absorb(matrix *nm) { 
        if (m) { alloc->free(m); m=0; }
        Assign(nm->M, nm->N, nm->M*nm->N*sizeof(T), nm->flag, nm->alloc);
        m = nm->m;
        nm->m = 0;
        delete nm;
    }
    void AssignDataPtr(int nm, int nn, T *nv, Allocator *Alloc=0) {
        if (m) { alloc->free(m); m=0; }
        if (!Alloc) Alloc = Singleton<NullAlloc>::Get();
        Assign(nm, nn, nm*nn*sizeof(T), 0, Alloc);
        m = nv;
    }
    void Open(int Mrows, int Ncols, T InitialVal=0, int Flag=0, Allocator *Alloc=0) {
        if (m) { alloc->free(m); m=0; }
        long long bytes = Mrows*Ncols*sizeof(T)*((Flag&Flag::Complex)?2:1);
        if (!Alloc) {
            Alloc = Singleton<MallocAlloc>::Get();
#if defined(__linux__) && !defined(LFL_ANDROID)
            if (bytes >= (1<<30)) Alloc = MMapAlloc::open("/dev/zero", true, false, bytes);
#endif
        }
        Assign(Mrows, Ncols, bytes, Flag, Alloc);
        AddRows(0);
        MatrixIter(this) { row(i)[j] = InitialVal; }
    }
    void Open(int Mrows, int Ncols, const char *bitmap, int Flag=0, Allocator *Alloc=0) {
        if (m) { alloc->free(m); m=0; }
        if (!Alloc) Alloc = Singleton<MallocAlloc>::Get();
        Assign(Mrows, Ncols, Mrows*Ncols*sizeof(T), Flag, Alloc);
        AddRows(0);
        MatrixIter(this) { row(i)[j] = (unsigned char)bitmap[i + j*M]; }
    }
    void Clear() {
        if (m) { alloc->free(m); m=0; }
        M = N = 0;
        bytes = 0;
    }
    matrix *Clone() const {
        if (!this) return 0;
        matrix *ret = new matrix(M,N,flag);
        MatrixIter(ret) { ret->row(i)[j] = row(i)[j]; }
        return ret;
    }

    matrix *Transpose(int flag=0) const { return matrix::Transpose(this, flag); }
    matrix *Mult(matrix *B, int flag=0) const { return matrix::Mult(this, B, flag); }
    void MultdiagR(double *diagonalmatrix, int len=0) { /* this = this * diagnolmatrix */
        if (len && len != N) FATAL("mismatch ", len, " != ", N);
        MatrixIter(this) row(i)[j] *= diagonalmatrix[j];
    }

    static matrix *Transpose(const matrix *A, int flag=0) {
        matrix *C = new matrix(A->N, A->M);
        MatrixIter(A) C->row(j)[i] = A->row(i)[j];
        if (flag & mDelA) delete A;
        return C;
    }
    static matrix *Add(const matrix *A, const matrix *B, matrix *C, int flag=0) {
        if (A->M != B->M || B->M != C->M || A->N != B->N || B->N != C->N) { ERROR("add ", A->M, ", ", A->N, ", ", B->M, ", ", B->N, ", ", C->M, ", ", C->N); return 0; }
        MatrixIter(A) C->row(i)[j] = A->row(i)[j] + B->row(i)[j];
        return CompleteOperation(A, B, C, flag);
    }
    static matrix *Sub(const matrix *A, const matrix *B, matrix *C, int flag=0) {
        if (A->M != B->M || B->M != C->M || A->N != B->N || B->N != C->N) { ERROR("sub ", A->M, ", ", A->N, ", ", B->M, ", ", B->N, ", ", C->M, ", ", C->N); return 0; }
        MatrixIter(A) C->row(i)[j] = A->row(i)[j] - B->row(i)[j];
        return CompleteOperation(A, B, C, flag);
    }
    static matrix *Mult(const matrix *A, const matrix *B, int flag=0) {
        matrix *C = new matrix((flag & mTrnpA)?A->N:A->M, (flag & mTrnpB)?B->M:B->N);
        return matrix::Mult(A, B, C, flag);
    }
    static matrix *Mult(const matrix *A, const matrix *B, matrix *C, int flag=0) {
        bool trnpA = flag & mTrnpA, trnpB = flag & mTrnpB, trnpC = flag & mTrnpC, neg = flag & mNeg;
        int AM = trnpA ? A->N : A->M, AN = trnpA ? A->M : A->N;
        int BM = trnpB ? B->N : B->M, BN = trnpB ? B->M : B->N;
        int CM = trnpC ? C->N : C->M, CN = trnpC ? C->M : C->N;
        if (AN != BM || CM != AM || CN != BN) { ERROR("mult ", AM, ", ", AN, ", ", BM, ", ", BN, ", ", CM, ", ", CN); return 0; }

        for (int i=0; i<AM; i++) {
            for (int j=0; j<BN; j++) {
                int ind1 = trnpC ? j : i, ind2 = trnpC ? i : j;
                C->row(ind1)[ind2] = 0;

                for (int k=0; k<AN; k++) {
                    T av = trnpA ? A->row(k)[i] : A->row(i)[k];
                    T bv = trnpB ? B->row(j)[k] : B->row(k)[j];
                    C->row(ind1)[ind2] += (neg ? -av : av) * bv;
                }
            }
        }
        return CompleteOperation(A, B, C, flag);
    }
    static matrix *Convolve(const matrix *A, const matrix *B, matrix *C, int flag=0) {
        if (A->M != C->M || A->N != C->N || (B->M % 2) != 1 || (B->N % 2) != 1) { ERROR("convolve ", A->M, ", ", A->N, " ", B->M, ", ", B->N, " ", C->M, ", ", C->N); return 0; }
        bool zero_only = flag & mZeroOnly;
        MatrixIter(A) {
            double ret = 0;
            int si = i - B->M/2, sj = j - B->N/2;
            if (zero_only && A->row(i)[j]) { C->row(i)[j] = A->row(i)[j]; continue; }

            for (int bi=0; bi < B->M; bi++) {
                int ii = si + bi;
                if (ii < 0 || ii >= A->M) continue;
                for (int bj=0; bj < B->N; bj++) {
                    int jj = sj + bj;
                    if (jj < 0 || jj >= A->N) continue;
                    ret += A->row(ii)[jj] * B->row(bi)[bj];
                }
            }
            C->row(i)[j] = ret;
        }
        return CompleteOperation(A, B, C, flag);
    }
    static matrix *CompleteOperation(const matrix *A, const matrix *B, matrix *C, int flag) {
        if (flag & mDelA) delete A;
        if (flag & mDelB) delete B;
        return C;
    }
    static T Max(const matrix *A) {
        T ret = -INFINITY;
        MatrixIter(A) ret = max(ret, A->row(i)[j]);
        return ret;
    }
    static void Print(const matrix *A, const string &name) {
        INFO(name, " Matrix(", A->M, ",", A->N, ") = ");
        MatrixRowIter(A) Vec<T>::Print(A->row(i), A->N);
    }
};
typedef matrix<double> Matrix;

struct RollingAvg { 
    int window, index, count, init, accum;
    unsigned total, *buf; float dev, *dbuf;

    ~RollingAvg() { free(buf); free(dbuf); }
    RollingAvg(int w) : window(w), index(0), count(0), init(0), accum(0),
        total(0), buf((unsigned*)calloc(window, sizeof(unsigned))),
        dev(0),   dbuf((float*)  calloc(window, sizeof(float))) {}

    void add(unsigned n) {
        unsigned ov = buf[index], nv = n + accum;
        float od = dbuf[index], nd = pow(avg() - nv, 2);

        buf[index] = nv;
        dbuf[index] = nd;

        index = (index + 1) % window;
        total += nv - ov;
        dev += nd - od;

        if (count < window) count++;
        accum = 0;
    }
    double _min() const { return Vec<unsigned>::Min(buf, count); }
    double _max() const { return Vec<unsigned>::Max(buf, count); }
    double avg() const { return count ? total / count : 0; }
    double sumavg() const { return count ? Vec<unsigned>::Sum(buf, count) / count : 0; }
    float stddev() const { return count ? sqrt(dev) : 0; }
    float fps() const { return 1000.0/avg(); }
};

/* util */
double squared(double n);
float decimals(float n);
float rand(float a, float b);
unsigned long long rand64();
float clamp(float x, float floor, float ceil);
void clamp(float *x, float floor, float ceil);
v3 clamp(const v3& x, float floor, float ceil);
v4 clamp(const v4& x, float floor, float ceil);
int round_f(float f, bool round_point_five_up=false);
int dim_check(const char *log, int d1, int d2);
int next_multiple_of_power_of_two(int input, int align);
void *next_multiple_of_power_of_two(void *input, int align);
int prev_multiple_of_n(int input, int N);
int next_multiple_of_n(int input, int N);
int next_power_of_two(int n);
bool is_power_of_two(unsigned n);
int which_log2(int n);
int floor_log2(int n);
int is_prime(int n);
int next_prime(int n);

/* conversion */
double radian2degree(float rad);
double degree2radian(float deg);
double hz2mel(double f);
double mel2hz(double f);
double hz2bark(double f);
double bark2hz(double f);
float cm2inch(float f);
float mm2inch(float f);
float inch2cm(float f);
float inch2mm(float f);

/* cumulative average */
double avg_cumulative(double last, double next, int count);

/* interpolate f(x) : 0<=x<=1 */
double arrayAsFunc(double *Array, int ArrayLen, double x);

/* bilinearly interpolate f(x) : 0<=x<=1 */
double matrixAsFunc(Matrix *m, double x, double y);

/* edit distance */
int levenshtein(const vector<int> &source, const vector<int> &target, vector<LFL_STL_NAMESPACE::pair<int, int> > *alignment=0);

/* solve autoCorrelation toeplitz system for LPC */
double levinsondurbin(int order, double const *autoCorrelation, double *reflectionOut, double *lpcOut);

#ifdef _WIN32
/* logarithm of 1+x */
double log1p(double x);
#endif

/* log domain add */
double logadd(double *log_x, int n);
double logadd(double log_a, double log_b);
double logadd(double *log_a, double log_b);

/* log domain subtract */
double logsub(double log_a, double log_b);
double logsub(double *log_a, double log_b);

/* hyperbolic arc */
double asinh(double x);
double acosh(double x);
double atanh(double x);

/* normalized sinc function */
double sinc(double x);

/* hamming window */
double hamming(int n, int i);

/* fft index real */
float &fft_r(float *a, int fftlen, int index);
double &fft_r(double *a, int fftlen, int index);
const float &fft_r(const float *a, int fftlen, int index);
const double &fft_r(const double *a, int fftlen, int index);

/* fft index imaginary */
float &fft_i(float *a, int fftlen, int index);
double &fft_i(double *a, int fftlen, int index);
const float &fft_i(const float *a, int fftlen, int index);
const double &fft_i(const double *a, int fftlen, int index);

/* absolute value of fft index squared */
double fft_abs2(const float *a, int fftlen, int index);
double fft_abs2(const double *a, int fftlen, int index);

/* convert to decibels */
double fft_abs_to_dB(float fft_abs);
double fft_abs2_to_dB(float fft_abs2);

/* inverse discrete fourier transform */
Matrix *IDFT(int rows, int cols);

/* discrete cosine transform type 2 */
Matrix *DCT2(int rows, int cols);

/* mahalanobis distance squared */ 
double mahaldist2(const double *mean, const double *diagcovar, const double *vec, int D);

/* log determinant of diagnol matrix */
double diagdet(const double *diagmat, int D);

/* gaussian normalizing constant */
double gausNormC(const double *diagcovar, int D);

/* gaussian log(PDF) eval */
double gausPdfEval(const double *mean, const double *diagcovar, const double *normC, const double *observation, int D);

/* guassian mixture log(PDF) eval */
double gmmPdfEval(const Matrix *means, const Matrix *covariance, const double *observation,
                  const double *prior=0, const double *normC=0, double *outPosteriors=0);

/* gaussian mixture model */
struct GMM {
    Matrix mean, diagcov, prior, norm;

    GMM() {}
    GMM(Matrix *M, Matrix *V, Matrix *P) : mean(*M), diagcov(*V), prior(*P) { computeNorms(); }
    GMM(int K, int D) : mean(K, D), diagcov(K, D), prior(K, 1) { computeNorms(); }

    void assignDataPtr(GMM *m) { assignDataPtr(m->mean.M, m->mean.N, m->mean.m, m->diagcov.m, m->prior.m, m->norm.m); }

    void assignDataPtr(int M, int N, double *m, double *v, double *p, double *n=0) {
        mean.AssignDataPtr(M, N, m);
        diagcov.AssignDataPtr(M, N, v);
        prior.AssignDataPtr(M, 1, p);
        if (n) norm.AssignDataPtr(M, 1, n);
        else computeNorms();
    }

    void computeNorms() {
        if (!norm.m) norm.Open(diagcov.M, 1);
        MatrixRowIter(&diagcov) norm.row(i)[0] = gausNormC(diagcov.row(i), diagcov.N);
    };

    double PDF(const double *observation, double *posteriors=0) {
        return gmmPdfEval(&mean, &diagcov, observation, prior.m ? prior.m : 0, norm.m ? norm.m : 0, posteriors);
    }
};

/* semirings */
struct Semiring {
    virtual double zero() = 0;
    virtual double one() = 0;
    virtual double add(double l, double r) = 0;
    virtual double mult(double l, double r) = 0;
    virtual double div(double l, double r) = 0;
    virtual bool approxequal(double l, double r) = 0;
    double add(double *l, double r) { return (*l = add(*l, r)); }
    double mult(double *l, double r) { return (*l = mult(*l, r)); }
    double div(double *l, double r) { return (*l = div(*l, r)); }
};

struct LogSemiring : public Semiring {
    double zero() { return INFINITY; }
    double one() { return 0; }
    double add(double l, double r) { return -logadd(-l, -r); }
    double mult(double l, double r) { return l + r; }
    double div(double l, double r) { return l - r; }
    bool approxequal(double l, double r) { return Equal(l, r); }
};

struct TropicalSemiring : public Semiring {
    double zero() { return INFINITY; }
    double one() { return 0; }
    double add(double l, double r) { return min(l, r); }
    double mult(double l, double r) { return l + r; }
    double div(double l, double r) { return l - r; }
    bool approxequal(double l, double r) { return Equal(l, r); }
};

struct BooleanSemiring : public Semiring {
    double zero() { return 0; }
    double one() { return 1; }
    double add(double l, double r) { return l || r; }
    double mult(double l, double r) { return l && r; }
    double div(double l, double r) { FATAL("not implemented: ", -1); }
    bool approxequal(double l, double r) { return l == r; }
};

struct DiscreteDistribution {
    typedef map<double, void*> Table;
    Table table;
    double sum;
    int samples;
    void *lastval;
    DiscreteDistribution() : lastval(0) { Clear(); }

    int Size() const { return table.size(); }
    void Clear() { table.clear(); sum=0; samples=0; }
    void Add(double p, void *v) { table[sum] = v; sum += p; lastval = v; }
    void Prepare() { table[sum] = lastval; }
    void *Sample() {
        samples++;
        float rv = rand(0, sum);
        Table::const_iterator i = table.lower_bound(rv);
        if (i == table.end()) FATAL("lower_bound ", rv, " ", sum);
        return i->second;
    }
};

/* subtract each columns mean from each row */
void MeanNormalizeRows(const Matrix *in, Matrix *out);

/* invert matrix */
void Invert(const Matrix *in, Matrix *out);

/* singular value decomposition: A = U D V^T */
void SVD(const Matrix *A, Matrix *D, Matrix *U, Matrix *V);

/* principal components analysis */
Matrix *PCA(const Matrix *obv, Matrix *projected, double *var=0);

}; // namespace LFL
#endif // __LFL_LFAPP_MATH_H__
