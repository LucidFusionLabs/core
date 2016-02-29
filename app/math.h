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

#ifndef LFL_LFAPP_MATH_H__
#define LFL_LFAPP_MATH_H__

namespace LFL {
int NextMultipleOfPowerOfTwo(int input, int align);
void *NextMultipleOfPowerOfTwo(void *input, int align);
inline int NextMultipleOf4 (int n) { return NextMultipleOfPowerOfTwo(n, 4); }
inline int NextMultipleOf8 (int n) { return NextMultipleOfPowerOfTwo(n, 8); }
inline int NextMultipleOf16(int n) { return NextMultipleOfPowerOfTwo(n, 16); }
inline int NextMultipleOf32(int n) { return NextMultipleOfPowerOfTwo(n, 32); }
inline int NextMultipleOf64(int n) { return NextMultipleOfPowerOfTwo(n, 64); }

template <typename X> typename enable_if<is_integral<X>::value, X>::type
Rand(X rmin = 0, X rmax = numeric_limits<X>::max()) {
  return std::uniform_int_distribution<X>(rmin, rmax)(ThreadLocalStorage::Get()->rand_eng);
}

template <typename X> typename enable_if<is_floating_point<X>::value, X>::type
Rand(X rmin = 0, X rmax = numeric_limits<X>::max()) {
  return std::uniform_real_distribution<X>(rmin, rmax)(ThreadLocalStorage::Get()->rand_eng);
}

template <class Generator> string RandBytes(int n, Generator &g) {
  string ret(NextMultipleOfPowerOfTwo(n, sizeof(uint32_t)), 0);
  std::uniform_int_distribution<uint32_t> dist(0, numeric_limits<uint32_t>::max());
  for (char *p = &ret[0], *e = p + ret.size(); p != e; p += sizeof(uint32_t)) 
  { uint32_t rv = dist(g); memcpy(p, &rv, sizeof(uint32_t)); }
  ret.resize(n);
  return ret;
}

inline int rand() { return Rand<int>(); }
inline unsigned long long Rand64() { return Rand<unsigned long long>(); }
inline bool Equal(float a, float b, float eps=1e-6) { return fabs(a-b) < eps; }

template <class X> static X Negate(X x) { return x ? -x : x; }
template <class X> static bool Min(X *a, X b) { if (b >= *a) return 0; *a = b; return 1; }
template <class X> static bool Max(X *a, X b) { if (b <= *a) return 0; *a = b; return 1; }
template <class X> static bool Within(X x, X a, X b) { return x >= a && x <= b; }
template <class X> static bool Changed     (X* p, const X& r) { bool ret = *p != r;       if (ret) *p = r; return ret; }
template <class X> static bool EqualChanged(X* p, const X& r) { bool ret = !Equal(*p, r); if (ret) *p = r; return ret; }
template <class X> static X    ChangedDiff (X* p, const X& r) { X ret = r - *p;           if (ret) *p = r; return ret; }
template <class X> static void MinusPlus(X *m, X* p, X v) { *m -= v; *p += v; }

template <class X> struct V2 {
  X x, y;
  V2() : x(0), y(0) {}
  V2(X xin, X yin) : x(xin), y(yin) {}
  template <class Y> V2(const V2<Y> &c) : x(X(c.x)), y(Y(c.y)) {}
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
  void Norm() { float l=Len(); if (!l) return; x /= l; y /= l; }
  float Len() const { return sqrt(x*x + y*y); }
  bool Zero() const { return !x && !y; }
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
  v3 operator * (float r) const { v3 ret=*this; ret.Scale(r); return ret; }
  v3 operator * (const v3 &dm) const { v3 ret=*this; ret.x*=dm.x; ret.y*=dm.y; ret.z*=dm.z; return ret; }
  v3 operator - (const v3 &v) const { v3 ret=*this; ret.Sub(v); return ret; }
  v3 operator + (const v3 &v) const { v3 ret=*this; ret.Add(v); return ret; }
  void operator += (const v3 &v) { Add(v); }
  void Add(v3 v) { x += v.x; y += v.y; z += v.z; }
  void Sub(v3 v) { x -= v.x; y -= v.y; z -= v.z; }
  void Scale(float f) { x *= f; y *= f; z *= f; }
  float Len() { return sqrt(x*x + y*y + z*z); }
  void Norm() { float l=Len(); if (!l) return; x /= l; y /= l; z /= l; }
  string DebugString() const { return StrCat("v(", x, ", ", y, ", ", z, ")"); }
  static v3 Norm(v3 q) { float l=q.Len(); if (!l) return q; q.x /= l; q.y /= l; q.z /= l; return q; }
  static float Dot(const v3 &q, const v3 &p) { return q.x*p.x + q.y*p.y + q.z*p.z; }
  static float Dist2(const v3 &q, const v3 &p) { v3 d = q - p; return Dot(d, d); }
  static v3 Cross(v3 q, v3 p) { return v3(q.y*p.z - q.z*p.y, q.z*p.x - q.x*p.z, q.x*p.y - q.y*p.x); }
  static v3 Normal(v3 a, v3 b, v3 c) { v3 q=c, p=c; q.Sub(a); p.Sub(b); q = v3::Cross(q, p); q.Norm(); return q; }
  static v3 Rand();
};

struct v4 {
  float x, y, z, w;
  v4() : x(0), y(0), z(0), w(0) {}
  v4(const v3& xyz, float W) : x(xyz.x), y(xyz.y), z(xyz.z), w(W) {}
  v4(float X, float Y, float Z, float W) : x(X), y(Y), z(Z), w(W) {}
  v4(const float *v) : x(v[0]), y(v[1]), z(v[2]), w(v[3]) {}
  operator float *() { return &x; }
  operator const float *() const { return &x; }
  bool operator == (const v4 &r) const { return x == r.x && y == r.y && z == r.z && w == r.w; }
  bool operator != (const v4 &r) const { return !(*this == r); }
  bool operator<(const v4 &c) const;
  string DebugString() const { return StrCat("(", x, ",", y, ",", z, ",", w, ")"); }
  v4 operator * (float r) { v4 ret=*this; ret.Scale(r); return ret; }
  v4 operator * (const v4 &dm) { v4 ret=*this; ret.x*=dm.x; ret.y*=dm.y; ret.z*=dm.z; ret.w*=dm.w; return ret; }
  v4 operator / (float r) { v4 ret=*this; ret.Scale(1/r); return ret; }
  v4 operator + (const v4 &v) { v4 ret=*this; ret.Add(v); return ret; }
  v4 operator - (const v4 &v) { v4 ret=*this; ret.Sub(v); return ret; }
  void Add(v4 v) { x += v.x; y += v.y; z += v.z; w += v.w; }
  void Sub(v4 v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; }
  void Scale(float f) { x *= f; y *= f; z *= f; w *= f; }
  float Len() { return sqrt(x*x + y*y + z*z + w*w); }
  void Norm() { float l=Len(); if (!l) return; Scale(1/l); }
  v3 XYZ() const { return v3(x, y, z); }
};

struct m33 {
  v3 m[3];
  m33() {}
  /**/  v3 &operator[](unsigned i)       { return m[i]; }
  const v3 &operator[](unsigned i) const { return m[i]; }
  static m33 RotAxis(float f, const v3 &v) { return RotAxis(f, v.x, v.y, v.z); }
  static m33 RotAxis(float f, float x, float y, float z) {
    m33 m;
    float s=sin(f), c=cos(f);
    m[0][0] =    c + x*x*(1-c); m[1][0] =  z*s + y*x*(1-c); m[2][0] = -y*s + z*x*(1-c);
    m[0][1] = -z*s + x*y*(1-c); m[1][1] =    c + y*y*(1-c); m[2][1] =  x*s + z*y*(1-c);
    m[0][2] =  y*s + x*z*(1-c); m[1][2] = -x*s + y*z*(1-c); m[2][2] =    c + z*z*(1-c);
    return m;
  }
  v3 Transform(const v3 &v) const {
    v3 ret;
    ret.x = m[0][0]*v.x + m[1][0]*v.y + m[2][0]*v.z;
    ret.y = m[0][1]*v.x + m[1][1]*v.y + m[2][1]*v.z;
    ret.z = m[0][2]*v.x + m[1][2]*v.y + m[2][2]*v.z;
    return ret;
  }
};

struct m44 {
  v4 m[4];
  m44() {}
  m44(const int   *in) { Assign(in); }
  m44(const float *in) { Assign(in); }
  m44(const m44   &in) { Assign(in); }
  m44(const m33   &in) { Assign(in); }
  /**/  v4 &operator[](unsigned i)       { return m[i]; }
  const v4 &operator[](unsigned i) const { return m[i]; }
  void Assign(const m44   &in) { for (int i=0; i<4; i++) m[i] = in.m[i]; }
  void Assign(const m33   &in) { for (int i=0; i<3; i++) m[i] = v4(in[i], 0); m[3] = v4(0,0,0,1); }
  void Assign(const float *in) { int k=0; for (int i=0; i<4; i++) for (int j=0; j<4; j++, k++) m[i][j] = in[k]; }
  void Assign(const int   *in) { int k=0; for (int i=0; i<4; i++) for (int j=0; j<4; j++, k++) m[i][j] = in[k]; }
  void Mult(const m44 &in) { m44 result; Mult(in, *this, &result); Assign(result); }
  void Print(const string &name) const { INFOf("%s = { %f,%f,%f,%f, %f,%f,%f,%f, %f,%f,%f,%f, %f,%f,%f,%f }\n", name.c_str(),
                                               m[0][0], m[0][1], m[0][2], m[0][3],
                                               m[1][0], m[1][1], m[1][2], m[1][3],
                                               m[2][0], m[2][1], m[2][2], m[2][3],
                                               m[3][0], m[3][1], m[3][2], m[3][3]); }
  v4 Transform(const v4 &v) const {
    v4 ret;
    ret.x=m[0][0]*v.x + m[1][0]*v.y + m[2][0]*v.z + m[3][0]*v.w;
    ret.y=m[0][1]*v.x + m[1][1]*v.y + m[2][1]*v.z + m[3][1]*v.w;
    ret.z=m[0][2]*v.x + m[1][2]*v.y + m[2][2]*v.z + m[3][2]*v.w;
    ret.w=m[0][3]*v.x + m[1][3]*v.y + m[2][3]*v.z + m[3][3]*v.w;
    return ret;
  }

  static bool Invert(const m44 &A, m44 *out=0, float *det_out=0);
  static void Mult(const m44 &A, const m44 &B, m44 *C) {
    for (int i=0; i<4; i++) for (int j=0; j<4; j++) { ((*C)[i])[j] = 0; for (int k=0; k<4; k++) ((*C)[i])[j] += (A[i])[k] * (B[k])[j]; }
  }
  static m44 Identity() { float v[] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }; return m44(v); };
  static m44 Translate(float x, float y, float z) { float v[] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1 }; return m44(v); }
  static m44 Scale(float x, float y, float z) { float v[] = { x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1 }; return m44(v); }
  static m44 Rotate(float angle, float x, float y, float z) { return m44(m33::RotAxis(angle, x, y, z)); }
  static m44 Ortho(float l, float r, float b, float t, float nv, float fv) {
    float v[] = { 2/(r-l),0,0,0, 0,2/(t-b),0,0, 0,0,-2/(nv-fv),0, -(r+l)/(r-l), -(t+b)/(t-b), -(fv+nv)/(fv-nv), 1 };
    return m44(v);
  }
  static m44 Frustum(float l, float r, float b, float t, float nv, float fv) { 
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
    ort.Norm();
    Assign(ort.x, ort.y, ort.z);
    d = -a*pos.x -b*pos.y -c*pos.z;
  }

  float Distance(v3 p, bool norm=true) {
    float num = (a*p.x + b*p.y + c*p.z + d);
    if (!norm) return num;
    return num / sqrt(a*a + b*b + c*c);
  }

  static v3 Normal(v3 p1, v3 p2, v3 p3) { return v3::Cross(p3-p2, p1-p2); }
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

  static Complex Add (Complex a, Complex b) { a.Add (b); return a; }
  static Complex Sub (Complex a, Complex b) { a.Sub (b); return a; }
  static Complex Mult(Complex a, Complex b) { a.Mult(b); return a; }
};

double LogAdd(double *, double);
float LogAdd(float *, float);
unsigned LogAdd(unsigned *, unsigned);
double Squared(double n);

template <class X> struct Vec {
  virtual ~Vec() {}
  virtual int Len() const = 0;
  virtual X Read(int i) const = 0;

  static string Str (const X *v1, int D)                  { string line; for (int i=0; i<D; i++) StrAppend(&line, (line.size() ? ", " : ""), ToString(         v1[i])); return line; }
  static string Str (const X *v1, int D, const char *fmt) { string line; for (int i=0; i<D; i++) StrAppend(&line, (line.size() ? ", " : ""), StringPrintf(fmt, v1[i])); return line; }
  static void Print (const X *v1,              int D) { INFO(Str(v1, D)); }
  static X    Dist2 (const X *v1, const X *v2, int D) { X ret=0; for (int i=0; i<D; i++) ret += Squared(v1[i]-v2[i]); return ret; }
  static X    Dot   (const X *v1, const X *v2, int D) { X ret=0; for (int i=0; i<D; i++) ret += v1[i] * v2[i];        return ret; }
  static bool Equals(const X *v1, const X *v2, int D) { for (int i=0; i<D; i++) if (v1[i] != v2[i]) return false; return true; }
  static void Add   (const X *v1, const X *v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] + v2[i]; }
  static void Add   (const X *v1,       X  v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] + v2;    }
  static void Sub   (const X *v1, const X *v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] - v2[i]; }
  static void Sub   (const X *v1,       X  v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] - v2;    }
  static void Mult  (const X *v1, const X *v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] * v2[i]; }
  static void Mult  (const X *v1,       X  v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] * v2;    }
  static void Div   (const X *v1, const X *v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] / v2[i]; }
  static void Div   (const X *v1,       X  v2, X *vOut, int D) { for (int i=0; i<D; i++) vOut[i] = v1[i] / v2;    }
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
  static void LogAdd(X *v1, const X *v2, int D) { for (int i=0; i<D; i++) LFL::LogAdd(&v1[i], v2[i]); }
  static void Exp   (      X *v1, int D) { exp(v1, v1, D); }
  static void Log   (      X *v1, int D) { log(v1, v1, D); }
  static double Sum (const X *v1, int D) { X ret=0; for (int i=0; i<D; i++) ret += v1[i]; return ret; }
  static double Min (const X *v1, int D) { X ret=X( INFINITY); for (int i=0; i<D; i++) if (v1[i] < ret) ret = v1[i]; return double(ret); }
  static double Max (const X *v1, int D) { X ret=X(-INFINITY); for (int i=0; i<D; i++) if (v1[i] > ret) ret = v1[i]; return double(ret); }
  static int MaxInd (const X *v1, int D) { X mx =X(-INFINITY); int ret=-1; for (int i=0; i<D; i++) if (v1[i] > mx) { mx = v1[i]; ret = i; } return ret; }
  template <class Y> static void Assign(X *v1, Y *v2, int D) { for (int i=0; i<D; i++) v1[i] = v2[i]; }
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

  matrix() : M(0), N(0), flag(0), bytes(0), alloc(Allocator::Default()), m(0) {}
  matrix(int Mrows, int Ncols, T InitialVal=T(), int Flag=0, Allocator *Alloc=0) : m(0) { Open(Mrows, Ncols, InitialVal, Flag, Alloc); }
  matrix(const matrix<T> &copy) : m(0), alloc(0) { Open(copy.M, copy.N, T(), copy.flag); AssignL(&copy); }
  matrix(const char *bitmap, int Mrows, int Ncols, int Flag=0, int Chans=1, int ChanInd=0, Allocator *Alloc=0) : m(0) { Open(Mrows, Ncols, bitmap, Flag, Alloc); }
  matrix& operator=(const matrix &copy) { Open(copy.M, copy.N, T(), copy.flag); AssignL(&copy); return *this; }
  virtual ~matrix() { alloc->Free(m); }

  virtual T             *row(int i)        { if (i>=M||i<0) return 0; return &m[i*N*((flag&Flag::Complex)?2:1)]; }
  virtual const T       *row(int i) const  { if (i>=M||i<0) return 0; return &m[i*N*((flag&Flag::Complex)?2:1)]; }
  virtual Complex       *crow(int i)       { if (i>=M||i<0) return 0; return reinterpret_cast<Complex*>(&m[i*N*2]); }
  virtual const Complex *crow(int i) const { if (i>=M||i<0) return 0; return reinterpret_cast<Complex*>(&m[i*N*2]); }

  virtual T       *lastrow()       { return row(M-1); }
  virtual const T *lastrow() const { return row(M-1); }

  void AddRows(int rows, bool prepend=false) {
    CHECK_GE(M+rows, 0);
    M += rows;
    bytes = M*N*sizeof(T)*((flag&Flag::Complex)?2:1);
    void *pre = m;
    if (!alloc) FATAL("null alloc: ", alloc, ", ", rows, ", ", bytes);
    if (!(m = reinterpret_cast<T*>(alloc->Realloc(m, bytes))))
      FATALf("matrix alloc failed: %p (%p) %lld (%d * %d * %lld)", m, pre, bytes, M, N, bytes/(M*N));
    if (rows > 0) {
      if (prepend) {
        int rowbytes = bytes/M;
        memmove(reinterpret_cast<char*>(m) + rowbytes*rows, m, rowbytes*(M-rows));
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
    swap(m, new_matrix.m);
    MinusPlus(&new_matrix.N, &N, int(cols));
    bytes = M*N*sizeof(T)*((flag&Flag::Complex)?2:1);
  }

  void Assign(int Mrows, int Ncols, long long Bytes, int Flag, Allocator *Alloc) { M=Mrows; N=Ncols; bytes=Bytes; flag=Flag; alloc=Alloc; }
  void AssignL(const matrix *m) { MatrixIter(this) row(i)[j] = m->row(i)[j]; }
  void AssignR(const matrix *m) { MatrixIter(m)    row(i)[j] = m->row(i)[j]; }
  void AssignL(const matrix *m, int flag) { bool neg=flag&mNeg; MatrixIter(this) row(i)[j] = neg ? Negate(m->row(i)[j]) : m->row(i)[j]; CompleteOperation(m, 0, 0, flag); }
  void AssignR(const matrix *m, int flag) { bool neg=flag&mNeg; MatrixIter(m)    row(i)[j] = neg ? Negate(m->row(i)[j]) : m->row(i)[j]; CompleteOperation(m, 0, 0, flag); }
  void AssignDiagonal(double v) { MatrixRowIter(this) row(i)[i] = v; }

  void Absorb(matrix *nm) { 
    if (m) { alloc->Free(m); m=0; }
    Assign(nm->M, nm->N, nm->M*nm->N*sizeof(T), nm->flag, nm->alloc);
    m = nm->m;
    nm->m = 0;
    delete nm;
  }

  void AssignDataPtr(int nm, int nn, T *nv, Allocator *Alloc=0) {
    if (m) { alloc->Free(m); m=0; }
    if (!Alloc) Alloc = Singleton<NullAllocator>::Get();
    Assign(nm, nn, nm*nn*sizeof(T), 0, Alloc);
    m = nv;
  }

  void Open(int Mrows, int Ncols, T InitialVal=0, int Flag=0, Allocator *Alloc=0) {
    if (m) { alloc->Free(m); m=0; }
    long long bytes = Mrows*Ncols*sizeof(T)*((Flag&Flag::Complex)?2:1);
    if (!Alloc) {
      Alloc = Allocator::Default();
#if defined(__linux__) && !defined(LFL_ANDROID)
      // if (bytes >= (1<<30)) Alloc = MMapAlloc::Open("/dev/zero", true, false, bytes);
#endif
    }
    Assign(Mrows, Ncols, bytes, Flag, Alloc);
    AddRows(0);
    MatrixIter(this) { row(i)[j] = InitialVal; }
  }

  void Open(int Mrows, int Ncols, const char *bitmap, int Flag=0, Allocator *Alloc=0) {
    if (m) { alloc->Free(m); m=0; }
    if (!Alloc) Alloc = Allocator::Default();
    Assign(Mrows, Ncols, Mrows*Ncols*sizeof(T), Flag, Alloc);
    AddRows(0);
    MatrixIter(this) { row(i)[j] = uint8_t(bitmap[i + j*M]); }
  }

  void Clear() {
    if (m) { alloc->Free(m); m=0; }
    M = N = 0;
    bytes = 0;
  }

  matrix *Clone() const {
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

template <class X> struct RollingAvg { 
  X total=0, accum=0;
  float dev=0;
  vector<X> buf;
  vector<float> dbuf;
  int window=0, index=0, count=0;
  RollingAvg(int W) : buf(W, 0), dbuf(W, 0), window(W) {}

  double Min   () const { return Vec<X>::Min(&buf[0], count); }
  double Max   () const { return Vec<X>::Max(&buf[0], count); }
  double Avg   () const { return count ? total / count : 0; }
  double SumAvg() const { return count ? Vec<X>::Sum(&buf[0], count) / count : 0; }
  float  StdDev() const { return count ? sqrt(dev) : 0; }
  float  FPS   () const { return 1000.0/Avg(); }

  void Add(X n) {
    X     ov = buf [index], nv = n + accum;
    float od = dbuf[index], nd = pow(Avg() - nv, 2);
    buf [index] = nv;
    dbuf[index] = nd;
    index = (index + 1) % window;
    total += nv - ov;
    dev   += nd - od;
    if (count < window) count++;
    accum = 0;
  }
};

/* util */
float Decimals(float n);
float Clamp(float x, float floor, float ceil);
void Clamp(float *x, float floor, float ceil);
v3 Clamp(const v3& x, float floor, float ceil);
v4 Clamp(const v4& x, float floor, float ceil);
int Sign(float f);
int RoundF(float f, bool round_point_five_up=false);
int RoundUp(float f);
int RoundDown(float f);
int RoundHigher(float f);
int RoundLower(float f);
int DimCheck(const char *log, int d1, int d2);
int PrevMultipleOfN(int input, int N);
int NextMultipleOfN(int input, int N);
int NextPowerOfTwo(int n, bool strict=false);
bool IsPowerOfTwo(unsigned n);
int WhichLog2(int n);
int FloorLog2(int n);
int IsPrime(int n);
int NextPrime(int n);
int DoubleSort(double a, double b);
int DoubleSort (const void *a, const void *b);
int DoubleSortR(const void *a, const void *b);

/* conversion */
double RadianToDegree(float rad);
double DegreeToRadian(float deg);
double HZToMel(double f);
double MelToHZ(double f);
double HZToBark(double f);
double BarkToHZ(double f);
float CMToInch(float f);
float MMToInch(float f);
float InchToCM(float f);
float InchToMM(float f);

/* cumulative average */
double CumulativeAvg(double last, double next, int count);

/* interpolate f(x) : 0<=x<=1 */
double ArrayAsFunc(double *Array, int ArrayLen, double x);

/* bilinearly interpolate f(x) : 0<=x<=1 */
double MatrixAsFunc(Matrix *m, double x, double y);

/* edit distance */
int Levenshtein(const vector<int> &source, const vector<int> &target, vector<LFL_STL_NAMESPACE::pair<int, int> > *alignment=0);

/* solve autoCorrelation toeplitz system for LPC */
double LevinsonDurbin(int order, double const *autoCorrelation, double *reflectionOut, double *lpcOut);

#ifdef _WIN32
/* logarithm of 1+x */
double log1p(double x);
#endif

/* log domain add */
double LogAdd(double *log_x, int n);
double LogAdd(double log_a, double log_b);
double LogAdd(double *log_a, double log_b);

/* log domain subtract */
double LogSub(double log_a, double log_b);
double LogSub(double *log_a, double log_b);

/* hyperbolic arc */
double asinh(double x);
double acosh(double x);
double atanh(double x);

/* normalized sinc function */
double Sinc(double x);

/* hamming window */
double Hamming(int n, int i);

/* convert to decibels */
double AmplitudeRatioDecibels(float a1, float a2);

void FFT(float *out, int fftlen);
void IFFT(float *out, int fftlen);

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

/* inverse discrete fourier transform */
Matrix *IDFT(int rows, int cols);

/* discrete cosine transform type 2 */
Matrix *DCT2(int rows, int cols);

/* mahalanobis distance squared */ 
double MahalDist2(const double *mean, const double *diagcovar, const double *vec, int D);

/* log determinant of diagnol matrix */
double DiagDet(const double *diagmat, int D);

/* gaussian normalizing constant */
double GausNormC(const double *diagcovar, int D);

/* gaussian log(PDF) eval */
double GausPdfEval(const double *mean, const double *diagcovar, const double *normC, const double *observation, int D);

/* guassian mixture log(PDF) eval */
double GmmPdfEval(const Matrix *means, const Matrix *covariance, const double *observation,
                  const double *prior=0, const double *normC=0, double *outPosteriors=0);

/* gaussian mixture model */
struct GMM {
  Matrix mean, diagcov, prior, norm;
  GMM() {}
  GMM(int K, int D) : mean(K, D), diagcov(K, D), prior(K, 1) { ComputeNorms(); }
  GMM(Matrix *M, Matrix *V, Matrix *P) : mean(*M), diagcov(*V), prior(*P) { ComputeNorms(); }

  void AssignDataPtr(GMM *m) { AssignDataPtr(m->mean.M, m->mean.N, m->mean.m, m->diagcov.m, m->prior.m, m->norm.m); }
  void AssignDataPtr(int M, int N, double *m, double *v, double *p, double *n=0) {
    mean.AssignDataPtr(M, N, m);
    diagcov.AssignDataPtr(M, N, v);
    prior.AssignDataPtr(M, 1, p);
    if (n) norm.AssignDataPtr(M, 1, n);
    else ComputeNorms();
  }

  void ComputeNorms() {
    if (!norm.m) norm.Open(diagcov.M, 1);
    MatrixRowIter(&diagcov) norm.row(i)[0] = GausNormC(diagcov.row(i), diagcov.N);
  }

  double PDF(const double *observation, double *posteriors=0) {
    return GmmPdfEval(&mean, &diagcov, observation, prior.m ? prior.m : 0, norm.m ? norm.m : 0, posteriors);
  }
};

/* semirings */
struct Semiring {
  virtual double Zero() = 0;
  virtual double One () = 0;
  virtual double Add (double l, double r) = 0;
  virtual double Mult(double l, double r) = 0;
  virtual double Div (double l, double r) = 0;
  virtual bool ApproxEqual(double l, double r) = 0;
  double Add (double *l, double r) { return (*l = Add (*l, r)); }
  double Mult(double *l, double r) { return (*l = Mult(*l, r)); }
  double Div (double *l, double r) { return (*l = Div (*l, r)); }
};

struct LogSemiring : public Semiring {
  double Zero() { return INFINITY; }
  double One () { return 0; }
  double Add (double l, double r) { return -LogAdd(-l, -r); }
  double Mult(double l, double r) { return l + r; }
  double Div (double l, double r) { return l - r; }
  bool ApproxEqual(double l, double r) { return Equal(l, r); }
};

struct TropicalSemiring : public Semiring {
  double Zero() { return INFINITY; }
  double One () { return 0; }
  double Add (double l, double r) { return min(l, r); }
  double Mult(double l, double r) { return l + r; }
  double Div (double l, double r) { return l - r; }
  bool ApproxEqual(double l, double r) { return Equal(l, r); }
};

struct BooleanSemiring : public Semiring {
  double Zero() { return 0; }
  double One () { return 1; }
  double Add (double l, double r) { return l || r; }
  double Mult(double l, double r) { return l && r; }
  double Div (double l, double r) { FATAL("not implemented: ", -1); }
  bool ApproxEqual(double l, double r) { return l == r; }
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
    float rv = Rand(0.0, sum);
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

/* arbitrary precision integers */
#if defined(LFL_COMMONCRYPTO) && 0
typedef CCBigNumRef BigNum;
typedef void* BigNumContext;
#elif defined(LFL_OPENSSL)
typedef BIGNUM* BigNum;
typedef BN_CTX* BigNumContext;
typedef int       ECDef;
typedef EC_GROUP* ECGroup;
typedef EC_POINT* ECPoint;
typedef EC_KEY*   ECPair;
#else
typedef void* BigNum;
typedef void* BigNumContext;
typedef void* ECDef;
typedef void* ECGroup;
typedef void* ECPoint;
typedef void* ECPair;
#endif
BigNum        NewBigNum();
BigNumContext NewBigNumContext();
void FreeBigNumContext(BigNumContext c);
void FreeBigNum(BigNum n);
void BigNumModExp(BigNum v, BigNum a, BigNum e, BigNum m, BigNumContext);
void BigNumSetValue(BigNum v, int val);
void BigNumGetData(BigNum v, char *out);
BigNum BigNumSetData(BigNum v, const StringPiece &data);
BigNum BigNumRand(BigNum v, int bits, int top, int bottom);
int BigNumDataSize(BigNum v);
int BigNumSignificantBits(BigNum v);
ECPoint NewECPoint(ECGroup);
void FreeECPoint(ECPoint);
void FreeECPair(ECPair);
ECGroup GetECPairGroup(ECPair);
ECPoint GetECPairPubKey(ECPair);
bool SetECPairPubKey(ECPair, ECPoint);
string ECPointGetData(ECGroup, ECPoint, BigNumContext);
int ECPointDataSize(ECGroup, ECPoint, BigNumContext);
void ECPointGetData(ECGroup, ECPoint, char *out, int len, BigNumContext);
void ECPointSetData(ECGroup, ECPoint out, const StringPiece &data);

}; // namespace LFL
#endif // LFL_LFAPP_MATH_H__
