/*
 * $Id: video.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef __LFL_LFAPP_VIDEO_H__
#define __LFL_LFAPP_VIDEO_H__

namespace LFL {
DECLARE_bool(gd_debug);
DECLARE_bool(swap_axis);
DECLARE_int(dots_per_inch);
DECLARE_float(field_of_view);
DECLARE_float(near_plane);
DECLARE_float(far_plane);
DECLARE_string(font_engine);
DECLARE_string(default_font);
DECLARE_string(default_font_family);
DECLARE_int(default_font_size);
DECLARE_int(default_font_flag);
DECLARE_int(default_missing_glyph);
DECLARE_bool(atlas_dump);
DECLARE_string(atlas_font_sizes);
DECLARE_int(scale_font_height);
DECLARE_int(add_font_size);

struct DrawMode { enum { _3D=0, _2D=1, NullOp=2 }; int m; };
struct TexGen { enum { LINEAR=1, REFLECTION=2 }; };

struct Depth {
    enum { _16=1 }; 
    static int OpenGLID(int d);
};

struct CubeMap {
    enum { PX=1, NX=2, PY=3, NY=4, PZ=5, NZ=6 };
    static int OpenGLID(int target);
};

struct ColorChannel {
    enum { Red=1, Green=2, Blue=3, Alpha=4 };
    static int PixelOffset(int c);
};

struct Pixel {
    enum { RGB32=1, BGR32=2, RGBA=3,
           RGB24=4, BGR24=5, 
           RGB555=6, BGR555=7, RGB565=8, BGR565=9,
           YUV410P=10, YUV420P=11, YUYV422=12, YUVJ420P=13, YUVJ422P=14, YUVJ444P=15,
           GRAY8=16, GRAYA8=17, LCD=18 };

    static const char *Name(int id);
    static int size(int p);
    static int OpenGLID(int p);

#ifdef LFL_PNG
    static int FromPngId(int fmt);
    static int ToPngId(int fmt);
#endif

#ifdef LFL_FFMPEG
    static int FromFFMpegId(int fmt);
    static int ToFFMpegId(int fmt);
#endif
};

struct Color {
    float x[4];
    Color() { r()=0; g()=0; b()=0; a()=1; }
    Color(const float *f) { r()=f[0]; g()=f[1]; b()=f[2]; a()=f[3]; }
    Color(double R, double G, double B) { r()=R; g()=G; b()=B; a()=1.0; }
    Color(double R, double G, double B, double A) { r()=R; g()=G; b()=B; a()=A; }
    Color(int R, int G, int B) { r()=R/255.0; g()=G/255.0; b()=B/255.0; a()=1.0; }
    Color(int R, int G, int B, int A) { r()=R/255.0; g()=G/255.0; b()=B/255.0; a()=A/255.0; }
    Color(unsigned v, bool has_alpha=true) { r()=((v>>16)&0xff)/255.0; g()=((v>>8)&0xff)/255.0; b()=(v&0xff)/255.0; a()=(has_alpha ? ((v>>24)&0xff)/255.0 : 1.0); };
    Color(const StringPiece &hs) { *this = Color(strtoul(hs.data(), 0, 16), false); }
    Color(const Color &c, double A) { *this = c; a() = A; }
    Color operator+(const Color &y) const { Color ret = *this; for (int i=0;i<4;i++) ret.x[i] += y.x[i]; return ret; }
    Color operator-(const Color &y) const { Color ret = *this; for (int i=0;i<4;i++) ret.x[i] -= y.x[i]; return ret; }
    bool operator< (const Color &y) const { SortMacro4(x[0], y.x[0], x[1], y.x[1], x[2], y.x[2], x[3], y.x[3]); }
    bool operator==(const Color &y) const { return R()==y.R() && G()==y.G() && B()==y.B() && A()==y.A(); }
    bool operator!=(const Color &y) const { return !(*this == y); }
    string DebugString() const { return HexString(); }
    string IntString() const { return StrCat("Color(", R(), ",", G(), ",", B(), ",", A(), ")"); }
    string HexString() const { return StringPrintf("%02X%02X%02X", R(), G(), B()); }
    const float &r() const { return x[0]; }
    const float &g() const { return x[1]; }
    const float &b() const { return x[2]; }     
    const float &a() const { return x[3]; }
    float       &r()       { return x[0]; }     
    float       &g()       { return x[1]; }
    float       &b()       { return x[2]; }     
    float       &a()       { return x[3]; }
    int          R() const { return RoundF(x[0]*255); } 
    int          G() const { return RoundF(x[1]*255); }
    int          B() const { return RoundF(x[2]*255); } 
    int          A() const { return RoundF(x[3]*255); }
    Color r(float v) const { Color c=*this; c.r() = v; return c; }
    Color g(float v) const { Color c=*this; c.g() = v; return c; }
    Color b(float v) const { Color c=*this; c.b() = v; return c; }
    Color a(float v) const { Color c=*this; c.a() = v; return c; }
    void scale(float f) { r() = Clamp(r()*f, 0, 1); g() = Clamp(g()*f, 0, 1); b() = Clamp(b()*f, 0, 1); }
    void ToHSV(float *h, float *s, float *v) const {
        float M = Typed::Max(r(), Typed::Max(g(), b()));
        float m = Typed::Min(r(), Typed::Min(g(), b()));
        float C = M - m;
        if (!C) { *h = *s = 0; *v = m; return; }

        *v = M;
        *s = C / M;

        if      (r() == m) *h = 3 - (g() - b()) / C;
        else if (g() == m) *h = 5 - (b() - r()) / C;
        else               *h = 1 - (r() - g()) / C;

        *h *= 60;
        if (*h < 0) *h += 360;
    }
    static Color FromHSV(float h, float s, float v) {
        if (s == 0) return Color(v, v, v);
        while (h >= 360) h -= 360;
        while (h <    0) h += 360;

        float hf = Decimals(h / 60);
        float p = v * (1 - s);
        float q = v * (1 - s * hf);
        float t = v * (1 - s * (1 - hf));

        if      (h < 60)  return Color(v, t, p);
        else if (h < 120) return Color(q, v, p);
        else if (h < 180) return Color(p, v, t);
        else if (h < 240) return Color(p, q, v);
        else if (h < 300) return Color(t, p, v);
        else              return Color(v, p, q);
    }
    static Color Red  (float v=1.0) { return Color(  v, 0.0, 0.0, 0.0); }
    static Color Green(float v=1.0) { return Color(0.0,   v, 0.0, 0.0); }
    static Color Blue (float v=1.0) { return Color(0.0, 0.0,   v, 0.0); }
    static Color Alpha(float v=1.0) { return Color(0.0, 0.0, 0.0,   v); }
    static Color fade(float v) {
        Color color;
        if      (v < 0.166667f) color = Color(1.0, v * 6.0, 0.0);
        else if (v < 0.333333f) color = Color((1/3.0 - v) * 6.0, 1.0, 0.0);
        else if (v < 0.5f)      color = Color(0.0, 1.0, (v - 1/3.0) * 6);
        else if (v < 0.666667f) color = Color(0.0, (2/3.0 - v) * 6, 1.0);
        else if (v < 0.833333f) color = Color((v - 2/3.0) * 6, 0.0, 1.0);
        else                    color = Color(1.0, 0.0, (1 - v)*6.0);
        for (int i = 0; i < 4; i++) color.x[i] = min(1.0f, max(color.x[i], 0.0f));
        return color;
    }
    static Color Interpolate(Color l, Color r, float mix) { l.scale(mix); r.scale(1-mix); return add(l,r); }
    static Color add(const Color &l, const Color &r) {
        return Color(Clamp(l.r()+r.r(), 0, 1), Clamp(l.g()+r.g(), 0, 1), Clamp(l.b()+r.b(), 0, 1), Clamp(l.a()+r.a(), 0, 1));
    }
    static Color white, black, red, green, blue, cyan, yellow, magenta, grey90, grey80, grey70, grey60, grey50, grey40, grey30, grey20, grey10;
};

struct Material {
    Color ambient, diffuse, specular, emissive;
    Material() {}
#if 0
    void SetLightColor(const Color &color) {
        diffuse = specular = ambient = color;
        ambient.scale(.2);
        emissive = Color::black;
    }
    void SetMaterialColor(const Color &color) {
        diffuse = ambient = color;
        specular = Color::white;
        emissive = Color::black;
    }
#else
    void SetLightColor(const Color &color) {
        diffuse = ambient = color;
        diffuse.scale(.9);
        ambient.scale(.5);
        specular = emissive = Color::black;
    }
    void SetMaterialColor(const Color &color) {
        diffuse = ambient = color;
        specular = emissive = Color::black;
    }
#endif
};

struct Light { v4 pos; Material color; };

struct Border {
    int top, right, bottom, left;
    Border() : top(0), right(0), bottom(0), left(0) {}
    Border(int T, int R, int B, int L) : top(T), right(R), bottom(B), left(L) {}
    Border &operator+=(const Border &v) { top+=v.top; right+=v.right; bottom+=v.bottom; left+=v.left; return *this; }
    Border &operator-=(const Border &v) { top-=v.top; right-=v.right; bottom-=v.bottom; left-=v.left; return *this; }
    int Width() const { return right + left; }
    int Height() const { return top + bottom; }
    Border TopBottom() const { return Border(top, 0, bottom, 0); }
    Border LeftRight() const { return Border(0, right, 0, left); }
    string DebugString() const { return StrCat("{", top, ", ", right, ", ", bottom, ", ", left, "}"); }
};

struct Box {
    int x, y, w, h;
    void clear() { x=y=w=h=0; }
    Box() : x(0), y(0), w(0), h(0) {}
    Box(int W, int H) : x(0), y(0), w(W), h(H) {}
    Box(const point &D) : x(0), y(0), w(D.x), h(D.y) {}
    Box(int X, int Y, int W, int H) : x(X), y(Y), w(W), h(H) {}
    Box(const point &P, int W, int H) : x(P.x), y(P.y), w(W), h(H) {}
    Box(float X, float Y, float W, float H, bool round) {
        if (round) { x=RoundF(X); y=RoundF(Y); w=RoundF(W); h=RoundF(H); }
        else       { x= (int)(X); y= (int)(Y); w= (int)(W); h= (int)(H); }
    }
    Box(const float *v4, bool round) {
        if (round) { x=RoundF(v4[0]); y=RoundF(v4[1]); w=RoundF(v4[2]); h=RoundF(v4[3]); }
        else       { x= (int)(v4[0]); y= (int)(v4[1]); w= (int)(v4[2]); h= (int)(v4[3]); }
    }
    virtual const FloatContainer *AsFloatContainer() const { return 0; }
    virtual       FloatContainer *AsFloatContainer()       { return 0; }
    virtual float baseleft (float py, float ph, int *ao=0) const { if (ao) *ao=-1; return x;   }
    virtual float baseright(float py, float ph, int *ao=0) const { if (ao) *ao=-1; return x+w; }
    virtual string DebugString() const;
    point Position () const { return point(x, y); }
    point Dimension() const { return point(w, h); }
    void SetPosition (const v2 &p)    { x=p.x; y=p.y; }
    void SetPosition (const point &p) { x=p.x; y=p.y; }
    void SetDimension(const v2 &p)    { w=p.x; h=p.y; }
    void SetDimension(const point &p) { w=p.x; h=p.y; }
    Box operator+(const point &p) const { return Box(x+p.x, y+p.y, w, h); }
    Box operator-(const point &p) const { return Box(x-p.x, y-p.y, w, h); }
    Box &operator+=(const point &p) { x+=p.x; y+=p.y; return *this; }
    Box &operator-=(const point &p) { x-=p.x; y-=p.y; return *this; }
    int top    () const { return y+h; }
    int right  () const { return x+w; }
    int centerX() const { return x+w/2; }
    int centerY() const { return y+h/2; }
    v2  center () const { return v2(centerX(), centerY()); }
    Box center(const Box &w) const { return Box(centerX(w.w), centerY(w.h), w.w, w.h); }
    int centerX (int cw)   const { return x+(w-cw)/2; }
    int centerY (int ch)   const { return y+(h-ch)/2; }
    int percentX(float px) const { return x+w*px; }
    int percentY(float py) const { return y+h*py; }
    bool Empty() const { return !w && !h; }
    bool within(const point &p) const { return p.x >= x && p.x <= x+w && p.y >= y && p.y <= y+h; }
    bool operator==(const Box &c) const { return x == c.x && y == c.y && w == c.w && h == c.h; }
    bool operator!=(const Box &c) const { return !(*this == c); }
    bool operator<(const Box &c) const { SortMacro4(x, c.x, y, c.y, w, c.w, h, c.h); }
    void scale(float xf, float yf) { x = RoundF(x*xf); w = RoundF(w*xf); y = RoundF(y*yf); h = RoundF(h*yf); }
    void swapaxis(int width, int height) { x += w; y += h; Typed::Swap(x,y); Typed::Swap(w,h); y = width - y; x = height - x; } 
    void AddBorder(const Border &b) { *this = AddBorder(*this, b); }
    void DelBorder(const Border &b) { *this = DelBorder(*this, b); }
    Box Intersect(const Box &w) const { Box ret(max(x, w.x), max(y, w.y), min(right(), w.right()), min(top(), w.top())); ret.w -= ret.x; ret.h -= ret.y; return (ret.w >= 0 && ret.h >= 0) ? ret : Box(); }
    Box BottomLeft(const Box &sub) const { return Box(x+sub.x, y+sub.y,           sub.w, sub.h); }
    Box    TopLeft(const Box &sub) const { return Box(x+sub.x, top()-sub.y-sub.h, sub.w, sub.h); }
    point  TopLeft () const { return point(x,       top()); }
    point  TopRight() const { return point(right(), top()); }
    void Draw(const float *texcoord=0) const;
    void DrawCrimped(const float *texcoord, int orientation, float scrollX=0, float scrollY=0) const;

    static float ScrollCrimped(float tex0, float tex1, float scroll, float *min, float *mid1, float *mid2, float *max);
    static bool   VerticalIntersect(const Box &w1, const Box &w2) { return w1.y < (w2.y + w2.h) && w2.y < (w1.y + w1.h); }
    static bool HorizontalIntersect(const Box &w1, const Box &w2) { return w1.x < (w2.x + w2.w) && w2.x < (w1.x + w1.w); }
    static Box Add(const Box &w, const point &p) { return Box(w.x+p.x, w.y+p.y, w.w, w.h); }
    static Box AddBorder(const Box &w, int xb, int yb) { return Box(w.x-RoundF(xb/2.0, 1), w.y-RoundF(yb/2.0, 1), max(0,w.w+xb), max(0,w.h+yb)); }
    static Box DelBorder(const Box &w, int xb, int yb) { return Box(w.x+RoundF(xb/2.0, 1), w.y-RoundF(yb/2.0, 1), max(0,w.w-xb), max(0,w.h-yb)); }
    static Box AddBorder(const Box &w, int tb, int rb, int bb, int lb) { return Box(w.x-lb, w.y-bb, max(0,w.w+lb+rb), max(0,w.h+tb+bb)); }
    static Box DelBorder(const Box &w, int tb, int rb, int bb, int lb) { return Box(w.x+lb, w.y+bb, max(0,w.w-lb-rb), max(0,w.h-tb-bb)); }
    static Box AddBorder(const Box &w, const Border &b) { return AddBorder(w, b.top, b.right, b.bottom, b.left); }
    static Box DelBorder(const Box &w, const Border &b) { return DelBorder(w, b.top, b.right, b.bottom, b.left); }
    static Box TopBorder(const Box &w, const Border &b) { return Box(w.x, w.top()-b.top, w.w, b.top); }
    static Box BotBorder(const Box &w, const Border &b) { return Box(w.x, w.y,           w.w, b.bottom); }
};

struct Box3 {
    Box v[3];
    Box3() {}
    Box3(const Box *container, const point &pb, const point &pe, int first_line_height, int last_line_height, int mid_region_height, int lines) {
        v[1].h = v[2].h = 0; int ll=1+(lines>1);
        if (pe.y == pb.y) v[0]  = Box(pb.x, pb.y, pe.x               - pb.x, first_line_height);
        else {            v[0]  = Box(pb.x, pb.y, container->right() - pb.x, first_line_height);
            if (lines>1){ v[1]  = Box(container->x, pb.y - mid_region_height, container->w, mid_region_height); CHECK(last_line_height); }
            if (lines)    v[ll] = Box(container->x, pe.y,              pe.x - container->x, last_line_height);
        }
    }
    Box       &operator[](int i)       { return v[i]; }
    const Box &operator[](int i) const { return v[i]; }
    bool Null() const { return !v[0].h; }
    void Clear() { for (int i=0; i<3; i++) v[i].clear(); }
    string DebugString() const { string ret = "Box3{"; for (int i=0; i<3; i++) if (!i || v[i].h) StrAppend(&ret, v[i].DebugString(), ", "); return ret + "}"; }
    void AddBorder(const Border &b, Box3 *out) const { for (int i=0; i<3; i++) if (!i || v[i].h) out->v[i] = Box::AddBorder(v[i], b); }
    void DelBorder(const Border &b, Box3 *out) const { for (int i=0; i<3; i++) if (!i || v[i].h) out->v[i] = Box::DelBorder(v[i], b); }
    void Draw(const Color *c) const;
    bool VerticalIntersect(const Box &w) const {
        for (int i=0; i<3; i++) if (v[i].h && Box::VerticalIntersect(v[i], w)) return true;
        return false;
    }
    Box BoundingBox() const {
        int min_x = v[0].x, min_y = v[0].y, max_x = v[0].x + v[0].w, max_y = v[0].y + v[0].h;
        if (v[1].h) { min_x = min(min_x, v[1].x); min_y = min(min_y, v[1].y); max_x = max(max_x, v[1].x + v[1].w); max_y = max(max_y, v[1].y + v[1].h); }
        if (v[2].h) { min_x = min(min_x, v[2].x); min_y = min(min_y, v[2].y); max_x = max(max_x, v[2].x + v[2].w); max_y = max(max_y, v[2].y + v[2].h); }
        return Box(min_x, min_y, max_x - min_x, max_y - min_y);
    }
};

struct Drawable {
    struct Box {
        LFL::Box box; const Drawable *drawable; int attr_id, line_id;
        Box(                   const Drawable *D=0, int A=0, int L=-1) :         drawable(D), attr_id(A), line_id(L) {}
        Box(const LFL::Box &B, const Drawable *D=0, int A=0, int L=-1) : box(B), drawable(D), attr_id(A), line_id(L) {}
        typedef ArrayMemberPairSegmentIter<Box, int, &Box::attr_id, &Box::line_id> Iterator;
    };
    struct Attr { 
        Font *font=0; const Color *fg=0, *bg=0; const Texture *tex=0; const LFL::Box *scissor=0;
        bool underline=0, overline=0, midline=0, blink=0;
        Attr(Font *F=0, const Color *FG=0, const Color *BG=0, bool UL=0) : font(F), fg(FG), bg(BG), underline(UL) {}
        bool operator==(const Attr &y) const { return font==y.font && fg==y.fg && bg==y.bg && tex==y.tex && scissor==y.scissor && underline==y.underline && overline==y.overline && midline==y.midline && blink==y.blink; }
        bool operator!=(const Attr &y) const { return !(*this == y); }
        void Clear() { font=0; fg=bg=0; tex=0; scissor=0; underline=overline=midline=blink=0; }
    };
    struct AttrSource { virtual Attr GetAttr(int attr_id) const = 0; };
    struct AttrVec : public AttrSource, public vector<Attr> {
        Attr current;
        AttrSource *source=0;
        Attr GetAttr(int attr_id) const { return source ? source->GetAttr(attr_id) : (*this)[attr_id-1]; }
        int GetAttrId(const Attr &v)
        { CHECK(!source); if (empty() || this->back() != v) push_back(v); return size(); }
    };
    virtual int  Id()                              const { return 0; }
    virtual int  Layout(const Attr *, LFL::Box *B) const { return B ? B->w : 0; }
    virtual void Draw  (const LFL::Box &B)         const = 0;
};

struct DrawableNullOp : public Drawable { void Draw(const LFL::Box &B) const {} };
#define DrawableNop() Singleton<DrawableNullOp>::Get()

struct Texture : public Drawable {
    unsigned ID; int width, height, pf, cubemap; float coord[4]; unsigned char *buf; bool buf_owner;
    Texture(int w=0, int h=0, int PF=Pixel::RGBA, unsigned id=0) : ID(id), width(w), height(h), cubemap(0), pf(PF), buf(0), buf_owner(1) { Coordinates(coord,1,1,1,1); }
    Texture(int w,   int h,   int PF,          unsigned char *B) : ID(0),  width(w), height(h), cubemap(0), pf(PF), buf(B), buf_owner(0) { Coordinates(coord,1,1,1,1); }
    Texture(const Texture &t) : ID(t.ID), width(t.width), height(t.height), pf(t.pf), cubemap(t.cubemap), buf(t.buf), buf_owner(buf?0:1) { memcpy(&coord, t.coord, sizeof(coord)); }
    virtual ~Texture() { ClearBuffer(); }
    void Bind() const;
    int PixelSize() const { return Pixel::size(pf); }
    int LineSize() const { return width * PixelSize(); }
    int GLPixelType() const { return Pixel::OpenGLID(pf); }
    int GLTexType() const { return CubeMap::OpenGLID(cubemap); }
    void ClearGL();
    void ClearBuffer() { if (buf_owner) delete [] buf; buf = 0; buf_owner = 1; }
    unsigned char *NewBuffer() const { return new unsigned char [width * height * PixelSize()](); }
    unsigned char *RenewBuffer() { ClearBuffer(); buf = NewBuffer(); return buf; }

    struct Flag { enum { CreateGL=1, CreateBuf=2, FlipY=4, Resample=8 }; };
    void Create      (int W, int H, int PF=0) { Resize(W, H, PF, Flag::CreateGL); }
    void CreateBacked(int W, int H, int PF=0) { Resize(W, H, PF, Flag::CreateGL | Flag::CreateBuf); }
    void Resize(int W, int H, int PF=0, int flag=0);

    void AssignBuffer(Texture *t, bool become_owner=0) { AssignBuffer(t->buf, t->width, t->height, t->pf, become_owner); if (become_owner) t->buf_owner=0; }
    void AssignBuffer(      unsigned char *B, int W, int H, int PF, bool become_owner=0) { buf=B; width=W; height=H; pf=PF; buf_owner=become_owner; }
    void LoadBuffer  (const unsigned char *B, int W, int H, int PF, int linesize, int flag=0);
    void UpdateBuffer(const unsigned char *B, int W, int H, int PF, int linesize, int flag=0);
    void UpdateBuffer(const unsigned char *B, int X, int Y, int W, int H, int PF, int linesize, int blit_flag=0);

    void LoadGL  (const unsigned char *B, int W, int H, int PF, int linesize, int flag=0);
    void UpdateGL(const unsigned char *B, int X, int Y, int W, int H, int flag=0);
    void UpdateGL(int X, int Y, int W, int H, int flag=0) { return UpdateGL(buf?(buf+(Y*width+X)*PixelSize()):0, X, Y, W, H, flag); }
    void UpdateGL() { UpdateGL(0, 0, width, height); }
    void LoadGL() { LoadGL(buf, width, height, pf, LineSize()); }

    virtual int Id() const { return 0; }
    virtual int Layout(const point &p, Box *out) const { *out = LFL::Box(p, width, height); return width; } 
    virtual void Draw(const LFL::Box &B) const { Bind(); B.Draw(coord); }
    virtual void DrawCrimped(const LFL::Box &B, int ort, float sx, float sy) const { Bind(); B.DrawCrimped(coord, ort, sx, sy); }

    void Screenshot();
    void ToIplImage(_IplImage *out);
#ifdef __APPLE__
    CGContextRef CGBitMap();
    CGContextRef CGBitMap(int X, int Y, int W, int H);
#endif
    static void Coordinates(float *texcoord, int w, int h, int wd, int hd);
    static const int CoordMinX, CoordMinY, CoordMaxX, CoordMaxY;
};

struct DepthTexture {
    unsigned ID; int width, height, df;
    DepthTexture(int w=0, int h=0, int DF=Depth::_16, unsigned id=0) : ID(id), width(w), height(h), df(DF) {}

    struct Flag { enum { CreateGL=1 }; };
    void Create(int W, int H, int DF=0) { Resize(W, H, DF, Flag::CreateGL); }
    void Resize(int W, int H, int DF=0, int flag=0);
};

struct FrameBuffer {
    unsigned ID; int width, height; DepthTexture depth; Texture tex;
    FrameBuffer(int w=0, int h=0, unsigned id=0) : ID(id), width(w), height(h) {}

    struct Flag { enum { CreateGL=1, CreateTexture=2, CreateDepthTexture=4, ReleaseFB=8 }; };
    void Create(int W, int H, int flag=0) { Resize(W, H, Flag::CreateGL | flag); }
    void Resize(int W, int H, int flag=0);

    void AllocTexture(Texture *out);
    void AllocTexture(unsigned *out) { Texture tex; AllocTexture(&tex); *out = tex.ID; } 
    void AllocDepthTexture(DepthTexture *out);

    void Attach(int ct=0, int dt=0);
    void Release();
    void Render(FrameCB cb);
};

struct Shader {
    string name;

    static const int MaxVertexAttrib = 4;
    int unused_attrib_slot[MaxVertexAttrib];

    bool dirty_material, dirty_light_pos[4], dirty_light_color[4];
    int ID, slot_position, slot_normal, slot_tex, slot_color, uniform_modelview, uniform_modelviewproj, uniform_tex,
        uniform_cubetex, uniform_normalon, uniform_texon, uniform_coloron, uniform_cubeon, uniform_colordefault,
        uniform_material_ambient, uniform_material_diffuse, uniform_material_specular, uniform_material_emission,
        uniform_light0_pos, uniform_light0_ambient, uniform_light0_diffuse, uniform_light0_specular;
    Shader()
        : ID(0), slot_position(-1), slot_normal(-1), slot_tex(-1), slot_color(-1), uniform_modelviewproj(-1), uniform_tex(-1),
        uniform_cubetex(-1), uniform_normalon(-1), uniform_texon(-1), uniform_coloron(-1), uniform_cubeon(-1), uniform_colordefault(-1),
        uniform_material_ambient(-1), uniform_material_diffuse(-1), uniform_material_specular(-1), uniform_material_emission(-1),
        uniform_light0_pos(-1), uniform_light0_ambient(-1), uniform_light0_diffuse(-1), uniform_light0_specular(-1) {
        dirty_material=0; memzeros(dirty_light_pos); memzeros(dirty_light_color);
    }

    static int Create(const string &name, const string &vertex_shader, const string &fragment_shader, const string &defines, Shader *out);
    int GetUniformIndex(const string &name);
    void SetUniform1i(const string &name, float v);
    void SetUniform1f(const string &name, float v);
    void SetUniform2f(const string &name, float v1, float v2);
    void SetUniform3fv(const string &name, const float *v);
    void SetUniform3fv(const string &name, int n, const float *v);

    static void SetGlobalUniform1f(const string &name, float v);
    static void SetGlobalUniform2f(const string &name, float v1, float v2);
};

struct Video : public Module {
    Shader shader_default, shader_normals, shader_cubemap, shader_cubenorm;
    Module *impl = 0;
    int Init();
    int Free();
    int Flush();
    void CreateGraphicsDevice();
};

extern Window *screen;
struct Window : public NativeWindow {
    GraphicsDevice *gd;
    point mouse;
    string caption;
    BindMap *binds;
    Console *console;
    Browser *browser_window;
    GUI *gui_root;
    RollingAvg fps;
    Dialog *top_dialog;
    vector<Dialog*> dialogs;
    Entity *cam;
    vector<GUI*> mouse_gui;
    vector<KeyboardGUI*> keyboard_gui;

    Window();
    virtual ~Window();
    void InitConsole();
    void ClearEvents();
    void ClearGesture();
    void Reshape(int w, int h);
    void Reshaped(int w, int h);
    void SwapAxis();

    void DeactivateMouseGUIs();
    void ClearMouseGUIEvents();
    void DeactivateKeyboardGUIs();
    void ClearKeyboardGUIEvents();
    void DrawDialogs();

    LFL::Box Box() const { return LFL::Box(0, 0, width, height); }
    LFL::Box Box(float xs, float ys) const { return LFL::Box(0, 0, width*xs, width*ys); }
    LFL::Box Box(float xp, float yp, float xs, float ys, float xbl=0, float ybt=0, float xbr=-INFINITY, float ybb=-INFINITY) const {
        if (isinf(xbr)) xbr = xbl;
        if (isinf(ybb)) ybb = ybt;
        return LFL::Box(width  * (xp + xbl),
                        height * (yp + ybb),
                        width  * xs - width  * (xbl + xbr),
                        height * ys - height * (ybt + ybb), false);
    }

    typedef unordered_map<void*, Window*> WindowMap;
    static WindowMap active;
    static Window *Get(void *id);
    static bool Create(Window *W);
    static void Close(Window *W);
    static void MakeCurrent(Window *W);
};

struct GraphicsDevice {
    static const int Float, Points, Lines, LineLoop, Triangles, TriangleStrip, Texture2D, UnsignedInt;
    static const int Ambient, Diffuse, Specular, Emission, Position;
    static const int One, SrcAlpha, OneMinusSrcAlpha, OneMinusDstColor;
    static const int Fill, Line, Point;

    int draw_mode = 0;
    vector<Color> default_color;
    vector<vector<Box> > scissor_stack;
    GraphicsDevice() : scissor_stack(1) {}

    virtual void Init() = 0;
    virtual bool ShaderSupport() = 0;
    virtual void EnableTexture() = 0;
    virtual void DisableTexture() = 0;
    virtual void EnableLighting() = 0;
    virtual void DisableLighting() = 0;
    virtual void EnableNormals() = 0;
    virtual void DisableNormals() = 0;
    virtual void EnableVertexColor() = 0;
    virtual void DisableVertexColor() = 0;
    virtual void EnableLight(int n) = 0;
    virtual void DisableLight(int n) = 0;
    virtual void DisableCubeMap() = 0;
    virtual void BindCubeMap(int n) = 0;
    virtual void TextureGenLinear() = 0;
    virtual void TextureGenReflection() = 0;
    virtual void Material(int t, float *color) = 0;
    virtual void Light(int n, int t, float *color) = 0;
    virtual void BindTexture(int t, int n) = 0;
    virtual void ActiveTexture(int n) = 0;
    virtual void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) = 0;
    virtual void VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) = 0;
    virtual void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) = 0;
    virtual void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) = 0;
    virtual void Color4f(float r, float g, float b, float a) = 0;
    virtual void UpdateColor() = 0;
    virtual void MatrixProjection() = 0;
    virtual void MatrixModelview() = 0;
    virtual void LoadIdentity() = 0;
    virtual void PushMatrix() = 0;
    virtual void PopMatrix() = 0;
    virtual void GetMatrix(m44 *out) = 0;
    virtual void PrintMatrix() = 0;
    virtual void Scalef(float x, float y, float z) = 0;
    virtual void Rotatef(float angle, float x, float y, float z) = 0;
    virtual void Ortho(float l, float r, float b, float t, float nv, float fv) = 0;
    virtual void Frustum(float l, float r, float b, float t, float nv, float fv) = 0;
    virtual void Mult(const float *m) = 0;
    virtual void Translate(float x, float y, float z) = 0;
    virtual void UseShader(Shader *shader) = 0;
    virtual void Draw(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) = 0;
    virtual void DrawArrays(int t, int o, int n) = 0;

    // Shader interface
    int CreateProgram();
    int CreateShader(int t);
    void ShaderSource(int shader, int count, const char **source, int *len);
    void CompileShader(int shader);
    void AttachShader(int prog, int shader);
    void BindAttribLocation(int prog, int loc, const string &name);
    void LinkProgram(int prog);
    void GetProgramiv(int p, int t, int *out);
    void GetIntegerv(int t, int *out);
    int GetAttribLocation(int prog, const string &name);
    int GetUniformLocation(int prog, const string &name);
    void Uniform1i(int u, int v);
    void Uniform1f(int u, float v);
    void Uniform2f(int u, float v1, float v2);
    void Uniform3fv(int u, int n, const float *v);

    // Common layer
    void Clear();
    void ClearColor(const Color &c);
    void FillColor(const Color &c) { DisableTexture(); SetColor(c); };
    void SetColor(const Color &c) { Color4f(c.r(), c.g(), c.b(), c.a()); }
    void PushColor(const Color &c) { PushColor(); SetColor(c); }
    void PushColor();
    void PopColor();
    void PointSize(float n);
    void LineWidth(float n);
    void GenTextures(int t, int n, unsigned *out);
    void DelTextures(int n, const unsigned *id);
    void EnableDepthTest();
    void DisableDepthTest();
    void EnableBlend();
    void DisableBlend();
    void BlendMode(int sm, int tm);
    void RestoreViewport(int drawmode);
    void DrawMode(int drawmode, bool flush=1);
    void DrawMode(int drawmode, int W, int H, bool flush=1);
    void EnableLayering() { DisableDepthTest(); DisableLighting(); EnableBlend(); EnableTexture(); }
    void LookAt(const v3 &pos, const v3 &targ, const v3 &up);
    void ViewPort(Box w);
    void Scissor(Box w);
    void PushScissor(Box w);
    void PopScissor();
    void PushScissorStack();
    void PopScissorStack();

    static int VertsPerPrimitive(int gl_primtype);
};

struct Scissor {
    Scissor(int x, int y, int w, int h) { screen->gd->PushScissor(Box(x, y, w, h)); }
    Scissor(Box w) { screen->gd->PushScissor(w); }
    ~Scissor()     { screen->gd->PopScissor(); }
};

struct ScissorStack {
    ScissorStack()  { screen->gd->PushScissorStack(); }
    ~ScissorStack() { screen->gd->PopScissorStack(); }
};

struct ScopedDrawMode {
    int prev_mode; bool nop;
    ScopedDrawMode(int dm) : prev_mode(screen->gd->draw_mode), nop(dm == DrawMode::NullOp) { if (!nop) screen->gd->DrawMode(dm,        0); }
    ~ScopedDrawMode()                                                                      { if (!nop) screen->gd->DrawMode(prev_mode, 0); }
};

struct VideoResamplerInterface {
    int s_fmt, d_fmt, s_width, d_width, s_height, d_height;
    VideoResamplerInterface() : s_fmt(0), d_fmt(0), s_width(0), d_width(0), s_height(0), d_height(0) {}
    virtual ~VideoResamplerInterface() {}
    virtual bool Opened()                                                                                           = 0;
    virtual void Open(int sw, int sh, int sf, int dw, int dh, int df)                                               = 0;
    virtual void Resample(const unsigned char *s, int sls, unsigned char *d, int dls, bool flip_x=0, bool flip_y=0) = 0;
};

struct SimpleVideoResampler : public VideoResamplerInterface {
    virtual ~SimpleVideoResampler() {}
    virtual bool Opened();
    virtual void Open(int sw, int sh, int sf, int dw, int dh, int df);
    virtual void Resample(const unsigned char *s, int sls, unsigned char *d, int dls, bool flip_x=0, bool flip_y=0);

    static bool Supports(int fmt);
    static void CopyPixel(int s_fmt, int d_fmt, const unsigned char *sp, unsigned char *dp, bool sxb, bool sxe, int flag=0);
    static void RGB2BGRCopyPixels(unsigned char *dst, const unsigned char *src, int l, int bpp);

    struct Flag { enum { FlipY=1, TransparentBlack=2, ZeroOnly=4 }; };

    static void Blit(const unsigned char *src, unsigned char *dst, int w, int h,
                     int sf, int sls, int sx, int sy,
                     int df, int dls, int dx, int dy, int flag=0);

    static void Filter(unsigned char *dst, int w, int h,
                       int pf, int ls, int x, int y, Matrix *kernel, int channel, int flag=0);

    static void ColorChannelToMatrix(const unsigned char *buf, int w, int h,
                                     int pw, int ls, int x, int y, Matrix *out, int po);
    static void MatrixToColorChannel(const Matrix *M, int w, int h,
                                     int pw, int ls, int x, int y, unsigned char *out, int po);
};

#ifdef LFL_FFMPEG
struct FFMPEGVideoResampler : public SimpleVideoResampler {
    void *conv; bool simple_resampler_passthru;
    FFMPEGVideoResampler() : conv(0), simple_resampler_passthru(0) {}
    virtual ~FFMPEGVideoResampler();
    virtual bool Opened();
    virtual void Open(int sw, int sh, int sf, int dw, int dh, int df);
    virtual void Resample(const unsigned char *s, int sls, unsigned char *d, int dls, bool flip_x=0, bool flip_y=0);
};
typedef FFMPEGVideoResampler VideoResampler;
#else
typedef SimpleVideoResampler VideoResampler;
#endif

struct HorizontalExtentTracker {
    int x, right, count;
    HorizontalExtentTracker() { Clear(); }
    void Clear() { x = INT_MAX; right = INT_MIN; count = 0; }
    void AddDrawableBox(const Drawable::Box &b) { AddBox(b.box); }
    void AddBox(const Box &b) { Typed::Min(&x, b.x); Typed::Max(&right, b.right()); count++; }
    Box Get(int y, int h) { Box ret = count ? Box(x, y, right-x, h) : Box(0, y, 0, h); Clear(); return ret; }
};

struct VerticalExtentTracker {
    int y, top, count;
    VerticalExtentTracker() { Clear(); }
    void Clear() { y = INT_MAX; top = INT_MIN; count = 0; }
    void AddDrawableBox(const Drawable::Box &b) { AddBox(b.box); }
    void AddBox(const Box &b) { Typed::Min(&y, b.y); Typed::Max(&top, b.top()); count++; }
    Box Get(int x, int w) { Box ret = count ? Box(x, y, w, top-y) : Box(x, 0, w, 0); Clear(); return ret; }
};

struct BoxExtentTracker {
    HorizontalExtentTracker x;
    VerticalExtentTracker   y;
    void Clear() { x.Clear(); y.Clear(); }
    void AddDrawableBox(const Drawable::Box &b) { AddBox(b.box); }
    void AddBox(const Box &b) { x.AddBox(b); y.AddBox(b); }
    Box Get() { return x.Get(y.y, y.top-y.y); }
};

struct BoxRun {
    Drawable::Attr attr;
    ArrayPiece<Drawable::Box> data;
    const Box *line;
    BoxRun(const Drawable::Box *buf=0, int len=0)                                        :          data(buf, len), line(0) {}
    BoxRun(const Drawable::Box *buf,   int len, const Drawable::Attr &A, const Box *L=0) : attr(A), data(buf, len), line(L) {}
    string Text() const { string t(data.size(), 0); for (int i=0; i<t.size(); i++) t[i] = data.buf[i].drawable ? data.buf[i].drawable->Id() : 0; return t; }
    string DebugString() const { return StrCat("BoxRun='", Text(), "'"); }

    typedef function<void    (const Drawable *,  const Box &)> DrawCB;
    static void DefaultDrawCB(const Drawable *d, const Box &w) { d->Draw(w); }
    point Draw(point p, DrawCB = &DefaultDrawCB);
	void draw(point p) { Draw(p); }

    typedef function<void              (const Box &)> DrawBackgroundCB;
    static void DefaultDrawBackgroundCB(const Box &w) { w.Draw(); }
    void DrawBackground(point p, DrawBackgroundCB = &DefaultDrawBackgroundCB);
};

struct BoxArray {
    vector<Drawable::Box> data;
    Drawable::AttrVec attr;
    vector<Box> line;
    vector<int> line_ind;
    int height;
    BoxArray() : height(0) { Clear(); }
    
    int Size() const { return data.size(); }
    string Text() const { return data.size() ? BoxRun(&data[0], data.size()).Text() : ""; }
    point Position(int o) const { 
        CHECK_GE(o, 0); bool last = o >= Size(); Box e;
        const Box &b = data.size() ? data[last ? data.size()-1 : o].box : e;
        return last ? b.TopRight() : b.TopLeft();
    }

    void Clear() { data.clear(); attr.clear(); line.clear(); line_ind.clear(); height=0; }
    BoxArray *Reset() { Clear(); return this; }

    Drawable::Box &PushBack(const Box &box, const Drawable::Attr &cur_attr, Drawable *drawable, int *ind_out=0) { return PushBack(box, attr.GetAttrId(cur_attr), drawable, ind_out); }
    Drawable::Box &PushBack(const Box &box, int                   cur_attr, Drawable *drawable, int *ind_out=0) {
        if (ind_out) *ind_out = data.size();
        return LFL::PushBack(data, Drawable::Box(box, drawable, cur_attr, line.size()));
    }

    void InsertAt(int o, const BoxArray &x) { InsertAt(o, x.data); }
    void InsertAt(int o, const vector<Drawable::Box> &x) {
        CHECK_EQ(0, line_ind.size());
        point p(x.size() ? (x.back().box.right() - x.front().box.x) : 0, 0);
        auto i = data.insert(data.begin()+o, x.begin(), x.end()) + x.size();
        for (; i != data.end(); ++i) i->box += p;
    }
    void Erase(int o, size_t l=UINT_MAX, bool shift=false) { 
        if (!l || data.size() <= o) return;
        if (shift) CHECK_EQ(0, line_ind.size());
        vector<Drawable::Box>::iterator b = data.begin() + o, e = data.begin() + min(o+l, data.size());
        point p(shift ? ((e-1)->box.right() - b->box.x) : 0, 0);
        auto i = data.erase(b, e);
        if (shift) for (; i != data.end(); ++i) i->box -= p;
    }

    point Draw(point p) {
        point e;
        for (Drawable::Box::Iterator iter(data); !iter.Done(); iter.Increment())
            e = BoxRun(iter.Data(), iter.Length(), attr.GetAttr(iter.cur_attr1), VectorGet(line, iter.cur_attr2)).Draw(p);
        return e;
    }
    string DebugString() const {
        string ret = StrCat("BoxArray H=", height, " ");
        for (Drawable::Box::Iterator iter(data); !iter.Done(); iter.Increment()) 
            StrAppend(&ret, "R", iter.i, "(", BoxRun(iter.Data(), iter.Length()).DebugString(), "), ");
        return ret;
    }
    void GetGlyphFromCoords(const point &p, int *glyph_index_out, Box *glyph_box_out) {}
    int   GetLineFromCoords(const point &p) { return 0; }

    struct RollbackState { size_t data_size, attr_size, line_size; int height; };
    RollbackState GetRollbackState() const { return { data.size(), attr.size(), line.size(), height }; }
    void Rollback(const RollbackState &s) { data.resize(s.data_size); attr.resize(s.attr_size); line.resize(s.line_size); height=s.height; }
};

struct FontDesc {
    enum { Bold=1, Italic=2, Mono=4 };
};

struct FontInterface {
    int size=0, height=0, max_top=0, max_width=0, fixed_width=0, flag=0, missing_glyph=0; bool mono=0;
    FontInterface(int S=0, int F=0) : size(S), flag(F), missing_glyph(FLAGS_default_missing_glyph), mono(F & FontDesc::Mono) {}
    int MaxBottom() const { return max_top ? height - max_top : 0; }
    int FixedWidth() const { return X_or_Y(fixed_width, mono ? max_width : 0); }
};

struct Font : public FontInterface {
    Color fg; bool mix_fg; float scale;
    Font()                             :                             mix_fg(0), scale(0) {}
    Font(int S, const Color &C, int F) : FontInterface(S, F), fg(C), mix_fg(0), scale(0) {}
    virtual ~Font() {}
    virtual Font *Clone(int pointsize, Color fg, int flag=0);
    static  Font *OpenAtlas(const string &name, int size, Color c, int fonts_flag);

    void Select();
    void Scale(int new_size);

    struct Glyph : public Drawable {
        int id; Texture tex; int top, left;
        Glyph() : id(0), top(0), left(0) {}

        bool operator<(const Glyph &y) const { return id < y.id; }
        int  ToArray  (      double *out, int l);
        void FromArray(const double *in,  int l);

        virtual int Id() const { return id; }
        virtual void Draw(const LFL::Box &B) const { return tex.Draw(B); }
        virtual int Layout(const Drawable::Attr *attr, LFL::Box *out) const {
            float scale = attr->font->scale;
            int center_width = attr->font->mono ? attr->font->max_width : 0;
            int gw=XY_or_Y(scale, tex.width), gh=XY_or_Y(scale, tex.height), gt=XY_or_Y(scale, top);
            *out = LFL::Box(center_width ? (center_width - gw) / 2 : 0, (gt ? gt - gh : 0), gw, gh);
            return X_or_Y(center_width, gw);
        }
    };
    struct Glyphs {
        vector<Glyph>              table;
        map<int, Glyph>            index;
        vector<shared_ptr<Atlas> > atlas;
        Font                      *primary;
        Glyphs(Font *P=0, Atlas *A=0) : table(128), primary(P) { if (A) atlas.push_back(shared_ptr<Atlas>(A)); }
    };
    shared_ptr<Glyphs> glyph;

#   define GlyphTableIter(f) for (auto i = (f)->glyph->table.begin(); i != (f)->glyph->table.end(); ++i)
#   define GlyphIndexIter(f) for (auto i = (f)->glyph->index.begin(); i != (f)->glyph->index.end(); ++i)

    virtual Glyph *FindGlyph        (unsigned gind);
    virtual Glyph *FindOrInsertGlyph(unsigned gind);
    virtual Glyph *LoadGlyph        (unsigned gind) { return &glyph->table[missing_glyph]; }
    void DrawGlyph(int g, const Box &w, int orientation=1) { return FindGlyph(g)->tex.Draw(w); }

    struct Flag {
        enum {
            NoWrap=1<<6, GlyphBreak=1<<7, AlignCenter=1<<8, AlignRight=1<<9, 
            Underline=1<<10, Overline=1<<11, Midline=1<<12, Blink=1<<13,
            Uppercase=1<<14, Lowercase=1<<15, Capitalize=1<<16, Clipped=1<<17,
            AssignFlowX=1<<18
        };
        static int Orientation(int f) { return f & 0xf; };
    };

    template <class X> void Size(const StringPieceT<X> &text, Box *out, int width=0, int *lines_out=0);
    /**/               void Size(const StringPiece     &text, Box *out, int width=0, int *lines_out=0) { return Size<char> (                text,  out, width, lines_out); }
    /**/               void Size(const String16Piece   &text, Box *out, int width=0, int *lines_out=0) { return Size<short>(                text,  out, width, lines_out); }
    template <class X> void Size(const X               *text, Box *out, int width=0, int *lines_out=0) { return Size(StringPiece::Unbounded(text), out, width, lines_out); }

    template <class X> int Width(const StringPieceT<X> &text) { Box b; Size(text, &b); CHECK_EQ(b.h, height); return b.w; }
    /**/               int Width(const StringPiece     &text) { return Width<char> (text); }
    /**/               int Width(const String16Piece   &text) { return Width<short>(text); }
    template <class X> int Width(const X               *text) { return Width(StringPiece::Unbounded(text)); }

    template <class X> void Encode(const StringPieceT<X> &text, const Box &box, BoxArray *out, int draw_flag=0, int attr_id=0);
    /**/               void Encode(const StringPiece     &text, const Box &box, BoxArray *out, int draw_flag=0, int attr_id=0) { return Encode<char> (                text,  box, out, draw_flag, attr_id); }
    /**/               void Encode(const String16Piece   &text, const Box &box, BoxArray *out, int draw_flag=0, int attr_id=0) { return Encode<short>(                text,  box, out, draw_flag, attr_id); }
    template <class X> void Encode(const X               *text, const Box &box, BoxArray *out, int draw_flag=0, int attr_id=0) { return Encode(StringPiece::Unbounded(text), box, out, draw_flag, attr_id); }

    template <class X> int Draw(const StringPieceT<X> &text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw<X>    (                text,  Box(cp.x,cp.y+height,0,0), lb, draw_flag); }
    /**/               int Draw(const StringPiece     &text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw<char> (                text,  Box(cp.x,cp.y+height,0,0), lb, draw_flag); }
    /**/               int Draw(const String16Piece   &text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw<short>(                text,  Box(cp.x,cp.y+height,0,0), lb, draw_flag); }
    template <class X> int Draw(const X               *text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw(StringPiece::Unbounded(text), Box(cp.x,cp.y+height,0,0), lb, draw_flag); }
    
    template <class X> int Draw(const StringPieceT<X> &text, const Box &box, vector<Box> *lb=0, int draw_flag=0);
    /**/               int Draw(const StringPiece     &text, const Box &box, vector<Box> *lb=0, int draw_flag=0) { return Draw<char>(                 text,  box, lb, draw_flag); }
    /**/               int Draw(const String16Piece   &text, const Box &box, vector<Box> *lb=0, int draw_flag=0) { return Draw<short>(                text,  box, lb, draw_flag); }
    template <class X> int Draw(const X               *text, const Box &box, vector<Box> *lb=0, int draw_flag=0) { return Draw(StringPiece::Unbounded(text), box, lb, draw_flag); }
};

struct FakeFont : public Font {
    static const char *Filename() { return "__FakeFontFilename__"; }
    FakeFont() {
        glyph = shared_ptr<Glyphs>(new Glyphs(this));
        fg = Color::white; fixed_width = max_width = 8; height = 14; size = 10; 
        for (int i = 0; i < 128; i++) { 
            glyph->table[i].id = i;
            glyph->table[i].tex.width = 8;
            glyph->table[i].tex.height = 14;
        }
    }
};

struct TTFFont : public Font {
    struct Resource {
        string content, name; FT_FaceRec_ *face; int flag;
        Resource() : face(0), flag(0) {}
        Resource(FT_FaceRec_ *FR, const string &N, int F=0) : face(FR), name(N), flag(F) {}
        Resource(const string &C, const string &N, int F=0) : content(C), face(0), name(N), flag(F) {}
        virtual ~Resource();
    };
    shared_ptr<Resource> resource;
    TTFFont(const shared_ptr<Resource> &R, int S, const Color &C, int F) : resource(R), Font(S,C,F) {}
    virtual Font *Clone(int pointsize, Color fg, int flag=0);
    virtual Glyph *LoadGlyph(unsigned gind);

    static void Init();
    struct Flag { enum { WriteAtlas=1, Outline=2 }; };
    static Font *OpenFile  (const string &filename, const string &name, int size, Color c, int flag, int ttf_flag);
    static Font *OpenBuffer(const shared_ptr<Resource> &R,              int size, Color c, int flag);
    static Font *Open      (const shared_ptr<Resource> &R,              int size, Color c, int flag);
};

#ifdef __APPLE__
struct CoreTextFont : public Font {
    struct Resource {
        string name; CGFontRef cgfont; int flag;
        Resource(const char *N=0, CGFontRef FR=0, int F=0) : name(BlankNull(N)), cgfont(FR), flag(F) {}
    };
    shared_ptr<Resource> resource;
    CoreTextFont(const shared_ptr<Resource> &R, int S, const Color &C, int F) : resource(R), Font(S,C,F) {}
    virtual Font *Clone(int pointsize, Color fg, int flag=0);
    virtual Glyph *LoadGlyph(unsigned gind);

    static void Init();
    struct Flag { enum { WriteAtlas=1 }; };
    static Font *Open(const string &name,            int size, Color c, int flag, int ct_flag);
    static Font *Open(const shared_ptr<Resource> &R, int size, Color c, int flag);
};
#endif

struct Fonts {
    typedef map<unsigned, Font*> FontSizeMap;
    typedef map<unsigned, FontSizeMap> FontColorMap;
    typedef map<string, FontColorMap> FontMap;
    FontMap font_map;

    struct Family { set<string> normal, bold, italic, bold_italic; };
    typedef map<string, Family> FamilyMap;
    FamilyMap family_map;

    static string FontName(const string &filename, int pointsize, Color fg, int flag);
    static unsigned FontColor(Color fg, int FontFlag);
    static int ScaledFontSize(int pointsize);

    static Font *Default();
    static Font *Fake();

    static Font *Insert        (Font*, const string &filename, const string &family, int pointsize, const Color &fg, int flag);
    static Font *InsertAtlas   (/**/   const string &filename, const string &family, int pointsize, const Color &fg, int flag);
    static Font *InsertFreetype(/**/   const string &filename, const string &family, int pointsize, const Color &fg, int flag);
    static Font *InsertCoreText(/**/   const string &filename, const string &family, int pointsize, const Color &fg, int flag);

    static Font *Get(FontColorMap *colors,                         int pointsize, Color fg, int flag=-1);
    static Font *Get(const string &filename,                       int pointsize, Color fg, int flag=-1);
    static Font *Get(const string &filename, const string &family, int pointsize, Color fg, int flag=-1);
};

struct FloatContainer : public Box {
    struct Float : public Box {
        bool inherited, stacked; void *val;
        Float() : inherited(0), stacked(0), val(0) {}
        Float(const point &p, int W=0, int H=0, void *V=0) : Box(p, W, H), inherited(0), stacked(0), val(V) {}
        Float(const Box   &w,                   void *V=0) : Box(w),       inherited(0), stacked(0), val(V) {}
        Float(const Float &f, const point &p) : Box(p, f.w, f.h), inherited(f.inherited), stacked(f.stacked), val(f.val) {}
        virtual string DebugString() const { return StrCat("Float{", Box::DebugString(), ", inherited=", inherited, ", stacked=", stacked, ", val=", (void*)val, "}"); }
        static void MarkInherited(vector<Float> *t) { for (auto i = t->begin(); i != t->end(); ++i) i->inherited=1; }
    };
    vector<Float> float_left, float_right;
    FloatContainer() {}
    FloatContainer(const Box &W) : Box(W) {}
    FloatContainer &operator=(const Box &W) { x=W.x; y=W.y; w=W.w; h=W.h; return *this; }

    virtual string DebugString() const;
    virtual const FloatContainer *AsFloatContainer() const { return this; }
    virtual       FloatContainer *AsFloatContainer()       { return this; }
    virtual float baseleft(float py, float ph, int *adjacent_out=0) const {
        int max_left = x;
        basedir(py, ph, &float_left, adjacent_out, [&](const Box &b){ return Typed::Max(&max_left, b.right()); });
        return max_left - x;
    }
    virtual float baseright(float py, float ph, int *adjacent_out=0) const { 
        int min_right = x + w;
        basedir(py, ph, &float_right, adjacent_out, [&](const Box &b){ return Typed::Min(&min_right, b.x); });
        return min_right - x;
    }
    void basedir(float py, float ph, const vector<Float> *float_target, int *adjacent_out, function<bool (const Box&)> filter_cb) const {
        if (adjacent_out) *adjacent_out = -1;
        for (int i = 0; i < float_target->size(); i++) {
            const Float &f = (*float_target)[i];
            if ((f.y + 0  ) >= (py + ph)) continue;
            if ((f.y + f.h) <= (py + 0 )) break;
            if (filter_cb(f) && adjacent_out) *adjacent_out = i; 
        }
    }

    int CenterFloatWidth(int fy, int fh) const { return baseright(fy, fh) - baseleft(fy, fh); }
    int FloatHeight() const {
        int min_y = 0;
        for (auto i = float_left .begin(); i != float_left .end(); ++i) if (!i->inherited) Typed::Min(&min_y, i->y);
        for (auto i = float_right.begin(); i != float_right.end(); ++i) if (!i->inherited) Typed::Min(&min_y, i->y);
        return -min_y;
    }
    int ClearFloats(int fy, int fh, bool clear_left, bool clear_right) const {
        if (!clear_left && !clear_right) return 0;
        int fl = -1, fr = -1, sy = fy, ch;
        while (clear_left || clear_right) {
            if (clear_left)  { baseleft (fy, fh, &fl); if (fl >= 0) Typed::Min(&fy, float_left [fl].Position().y - fh); }
            if (clear_right) { baseright(fy, fh, &fr); if (fr >= 0) Typed::Min(&fy, float_right[fr].Position().y - fh); }
            if ((!clear_left || fl<0) && (!clear_right || fr<0)) break;
        }
        return max(0, sy - fy);
    }

    FloatContainer *Reset() { Clear(); return this; }
    void Clear() { float_left.clear(); float_right.clear(); }

    void AddFloat(int fy, int fw, int fh, bool right_or_left, LFL::DOM::Node *v, Box *out_box) {
        for (;;) {
            int adjacent_ind = -1, opposite_ind = -1;
            int base_left  = baseleft (fy, fh, !right_or_left ? &adjacent_ind : &opposite_ind);
            int base_right = baseright(fy, fh,  right_or_left ? &adjacent_ind : &opposite_ind);
            int fx = right_or_left ? (base_right - fw) : base_left;
            Float *adjacent_float = (adjacent_ind < 0) ? 0 : &(!right_or_left ? float_left : float_right)[adjacent_ind];
            Float *opposite_float = (opposite_ind < 0) ? 0 : &( right_or_left ? float_left : float_right)[opposite_ind];
            if (((adjacent_float || opposite_float) && (fx < base_left || (fx + fw) > base_right)) ||
                (adjacent_float && adjacent_float->stacked)) {
                if (adjacent_float) adjacent_float->stacked = 1;
                point afp((X_or_Y(adjacent_float, opposite_float)->Position()));
                fy = afp.y - fh;
                continue;
            }
            *out_box = Box(fx, fy, fw, fh);
            break;
        }
        vector<Float> *float_target = right_or_left ? &float_right : &float_left;
        float_target->push_back(Float(out_box->Position(), out_box->w, out_box->h, v));
        sort(float_target->begin(), float_target->end(), FloatContainer::Compare);
    }

    int InheritFloats(const FloatContainer *parent) {
        Copy(parent->float_left,  &float_left,  -TopLeft(), 1, 1);
        Copy(parent->float_right, &float_right, -TopLeft(), 1, 1);
        return parent->float_left.size() + parent->float_right.size();
    }
    int AddFloatsToParent(FloatContainer *parent) {
        int count = 0;
        count += Copy(float_left,  &parent->float_left,  TopLeft(), 0, 0);
        count += Copy(float_right, &parent->float_right, TopLeft(), 0, 0);
        Float::MarkInherited(&float_left);
        Float::MarkInherited(&float_right);
        return count;
    }

    static bool Compare(const Box &lw, const Box &rw) { return pair<int,int>(lw.top(), lw.h) > pair<int,int>(rw.top(), rw.h); }
    static int Copy(const vector<Float> &s, vector<Float> *d, const point &dc, bool copy_inherited, bool mark_inherited) {
        int count = 0;
        if (!s.size()) return count;
        for (int i=0; i<s.size(); i++) {
            if (!copy_inherited && s[i].inherited) continue;
            d->push_back(Float(s[i], s[i].Position() + dc));
            if (mark_inherited) (*d)[d->size()-1].inherited = 1;
            count++;
        }
        sort(d->begin(), d->end(), Compare);
        return count;
    }
};

struct Flow {
    struct Layout {
        bool wrap_lines=1, word_break=1, align_center=0, align_right=0, ignore_newlines=0;
        int char_spacing=0, word_spacing=0, line_height=0, valign_offset=0;
        int (*char_tf)(int)=0, (*word_start_char_tf)(int)=0;
    } layout;
    point p; 
    BoxArray *out;
    const Box *container;
    Drawable::Attr cur_attr;
    int adj_float_left=-1, adj_float_right=-1;
    struct CurrentLine { int out_ind, beg, end, base, ascent, descent, height; bool fresh; } cur_line;
    struct CurrentWord { int out_ind, len;                                     bool fresh; } cur_word;
    enum class State { OK=1, NEW_WORD=2, NEW_LINE=3 } state=State::OK;
    int max_line_width=0;

    Flow(BoxArray *O) : Flow(0, 0, O) {}
    Flow(const Box *W=0, Font *F=0, BoxArray *O=0, Layout *L=0) :
        layout(*(L?L:Singleton<Layout>::Get())), out(O), container(W?W:Singleton<Box>::Get())
        { memzero(cur_line); memzero(cur_word); SetFont(F); SetCurrentLineBounds(); cur_line.fresh=1; }

    struct RollbackState {
        point p; Drawable::Attr attr; CurrentLine line; CurrentWord word; State state; int max_line_width;
        BoxArray::RollbackState out_state; 
    };
    RollbackState GetRollbackState() { return { p, cur_attr, cur_line, cur_word, state, max_line_width, out->GetRollbackState() }; }
    void Rollback(const RollbackState &s) { p=s.p; cur_attr=s.attr; cur_line=s.line; cur_word=s.word; state=s.state; max_line_width=s.max_line_width; out->Rollback(s.out_state); }
    string DebugString() const {
        return StrCat("Flow{ p=", p.DebugString(), ", container=", container->DebugString(), "}");
    }

    void SetFGColor(const Color *C) { cur_attr.fg = C; }
    void SetBGColor(const Color *C) { cur_attr.bg = C; }
    void SetAtlas(Font *F) { cur_attr.font = F; }
    void SetFont(Font *F) {
        if (!(cur_attr.font = F)) return;
        int prev_height = cur_line.height, prev_ascent = cur_line.ascent, prev_descent = cur_line.descent;
        Typed::Max(&cur_line.height,  F->height);
        Typed::Max(&cur_line.ascent,  X_or_Y(F->max_top, F->height));
        Typed::Max(&cur_line.descent, F->MaxBottom());
        UpdateCurrentLine(cur_line.height-prev_height, cur_line.ascent-prev_ascent, cur_line.descent-prev_descent);
    }
    void SetMinimumAscent(int line_ascent) {
        int prev_height = cur_line.height, prev_ascent = cur_line.ascent;
        Typed::Max(&cur_line.ascent, line_ascent);
        Typed::Max(&cur_line.height, cur_line.ascent + cur_line.descent);
        UpdateCurrentLine(cur_line.height-prev_height, cur_line.ascent-prev_ascent, 0);
    }
    void UpdateCurrentLine(int height_delta, int ascent_delta, int descent_delta) {
        int baseline_delta = height_delta/2.0 - ascent_delta/2.0 + descent_delta/2.0;
        p.y -= height_delta;
        cur_line.base += baseline_delta;
        if (out) MoveCurrentLine(point(0, -height_delta + baseline_delta));
    }

    int Height() const { return -p.y - (cur_line.fresh ? cur_line.height : 0); }
    Box CurrentLineBox() const { return Box(cur_line.beg, p.y, p.x - cur_line.beg, cur_line.height); }
    int LayoutLineHeight() const { return X_or_Y(layout.line_height, cur_attr.font ? cur_attr.font->height : 0); }

    void AppendVerticalSpace(int h) {
        if (h <= 0) return;
        if (!cur_line.fresh) AppendNewline();
        p.y -= h;
        SetCurrentLineBounds();
    }
    void AppendBlock(int w, int h, Box *box_out) {
        AppendVerticalSpace(h);
        *box_out = Box(0, p.y + cur_line.height, w, h);
    }
    void AppendBlock(int w, int h, const Border &b, Box *box_out) {
        AppendBlock(w + b.Width(), h + (h ? b.Height() : 0), box_out);
        *box_out = Box::DelBorder(*box_out, h ? b : b.LeftRight());
    }
    void AppendRow(float x=0, float w=0, Box *box_out=0) { AppendBox(x, container->w*w, cur_line.height, box_out); }
    void AppendBoxArrayText(const BoxArray &in) {
        bool attr_fwd = in.attr.source;
        for (Drawable::Box::Iterator iter(in.data); !iter.Done(); iter.Increment()) {
            if (!attr_fwd) cur_attr = in.attr.GetAttr(iter.cur_attr1);
            AppendText(BoxRun(iter.Data(), iter.Length()).Text(), attr_fwd ? iter.cur_attr1 : 0);
        }
    }

    int AppendBox(float x, int w, int h, Drawable *drawable) { p.x=container->w*x; return AppendBox(w, h, drawable); }
    int AppendBox(/**/     int w, int h, Drawable *drawable) { 
        AppendBox(&out->PushBack(Box(0,0,w,h), cur_attr, drawable));
        return out->data.size()-1;
    }

    void AppendBox(float x, int w, int h, Box *box_out) { p.x=container->w*x; AppendBox(w, h, box_out); }
    void AppendBox(/**/     int w, int h, Box *box_out) {
        Drawable::Box box(Box(0,0,w,h), 0, out ? out->attr.GetAttrId(cur_attr) : 0, out ? out->line.size() : -1);
        AppendBox(&box);
        if (box_out) *box_out = box.box;
    }
    void AppendBox(int w, int h, const Border &b, Box *box_out) {
        AppendBox(w + b.Width(), h + (h ? b.Height() : 0), box_out);
        if (box_out) *box_out = Box::DelBorder(*box_out, h ? b : b.LeftRight());
    }

    void AppendBox(Drawable::Box *box) {
        point bp = box->box.Position();
        SetMinimumAscent(box->box.h);
        if (!box->box.w) box->box.SetPosition(p);
        else {
            box->box.SetPosition(bp);
            cur_word.len = box->box.w;
            cur_word.fresh = 1;
            AppendBoxOrChar(0, box, box->box.h);
        }
        cur_word.len = 0;
    }

    /**/               void AppendText(float x, const StringPiece     &text) { p.x=container->w*x; AppendText<char> (                text,  0); }
    /**/               void AppendText(float x, const String16Piece   &text) { p.x=container->w*x; AppendText<short>(                text,  0); }
    template <class X> void AppendText(float x, const X               *text) { p.x=container->w*x; AppendText(StringPiece::Unbounded(text), 0); }
    template <class X> void AppendText(float x, const StringPieceT<X> &text) { p.x=container->w*x; AppendText<X>    (                text,  0); }

    /**/               void AppendText(const StringPiece     &text, int attr_id=0) { AppendText<char> (                text,  attr_id); }
    /**/               void AppendText(const String16Piece   &text, int attr_id=0) { AppendText<short>(                text,  attr_id); }
    template <class X> void AppendText(const X               *text, int attr_id=0) { AppendText(StringPiece::Unbounded(text), attr_id); }
    template <class X> void AppendText(const StringPieceT<X> &text, int attr_id=0) {
        if (!attr_id) attr_id = out->attr.GetAttrId(cur_attr);
        out->data.reserve(out->data.size() + text.size());
        int initial_out_lines = out->line.size(), line_start_ind = 0, c_bytes = 0, ci_bytes = 0;
        for (const X *p = text.data(); !text.end(p); p += c_bytes) {
            int c = UTF<X>::ReadGlyph(text, p, &c_bytes);
            if (AppendChar(c, attr_id, &PushBack(out->data, Drawable::Box())) == State::NEW_WORD) {
                for (const X *pi=p; *pi && notspace(*pi); pi += ci_bytes) {
                    int ci = UTF<X>::ReadGlyph(text, pi, &ci_bytes);
                    cur_word.len += XY_or_Y(cur_attr.font->scale, cur_attr.font->FindGlyph(ci)->tex.width);
                }
                AppendChar(c, attr_id, &out->data.back());
            }
        }
    }

    State AppendChar(int c, int attr_id, Drawable::Box *box) {
        if (layout.char_tf) c = layout.char_tf(c);
        if (state == State::NEW_WORD && layout.word_start_char_tf) c = layout.word_start_char_tf(c);
        Typed::Max(&cur_line.height, cur_attr.font->height);
        box->drawable = cur_attr.font->FindGlyph(c);
        box->attr_id = attr_id;
        box->line_id = out ? out->line.size() : -1;
        return AppendBoxOrChar(c, box, cur_attr.font->height);
    }
    State AppendBoxOrChar(int c, Drawable::Box *box, int h) {
        bool space = isspace(c);
        if (space) cur_word.len = 0;
        for (int i=0; layout.wrap_lines && i<1000; i++) {
            bool wrap = 0;
            if (!cur_word.len) cur_word.fresh = 1;
            if (!layout.word_break) wrap = cur_line.end && p.x + box->box.w > cur_line.end;
            else if (cur_word.fresh && !space) {
                if (!cur_word.len) return state = State::NEW_WORD;
                wrap = cur_word.len && cur_line.end && (p.x + cur_word.len > cur_line.end);
            }
            if (wrap && !(cur_line.fresh && adj_float_left == -1 && adj_float_right == -1)) {
                if (cur_line.fresh) { /* clear floats */ } 
                AppendNewline(h);
                continue;
            }
            break;
        }
        cur_line.fresh = 0;
        cur_word.fresh = 0;
        if (c == '\n') { if (!layout.ignore_newlines) AppendNewline(); return State::OK; }

        int advance = box->drawable ? box->drawable->Layout(&cur_attr, &box->box) : box->box.w;
        box->box.y += cur_line.base;
        box->box += p;
        p.x += advance;
        return state = State::OK;
    }

    void AppendNewlines(int n) { for (int i=0; i<n; i++) AppendNewline(); }
    State AppendNewline(int need_height=0, bool next_glyph_preadded=1) {
        if (out) {        
            AlignCurrentLine();
            out->line.push_back(CurrentLineBox());
            out->line_ind.push_back(out ? Typed::Max<int>(0, out->data.size()-next_glyph_preadded) : 0);
            out->height += out->line.back().h;
            if (out->data.size() > cur_line.out_ind)
                Typed::Max(&max_line_width, out->data.back().box.right() - out->data[cur_line.out_ind].box.x);
        }
        cur_line.fresh = 1;
        cur_line.height = cur_line.ascent = cur_line.descent = 0;
        cur_line.out_ind = out ? out->data.size() : 0;
        SetMinimumAscent(max(need_height, LayoutLineHeight()));
        SetCurrentLineBounds();
        return state = State::NEW_LINE;
    }

    void AlignCurrentLine() {
        if (cur_line.out_ind >= out->data.size() || (!layout.align_center && !layout.align_right)) return;
        int line_size = cur_line.end - cur_line.beg, line_min_x, line_max_x;
        GetCurrentLineExtents(&line_min_x, &line_max_x);
        int line_len = line_max_x - line_min_x, align = 0;
        if      (layout.align_center) align = (line_size - line_len)/2;
        else if (layout.align_right)  align = (line_size - line_len);
        if (align) MoveCurrentLine(point(align, 0));
    }
    void MoveCurrentLine(const point &dx) { 
        for (auto i = out->data.begin() + cur_line.out_ind; i != out->data.end(); ++i) i->box += dx;
    }
    void GetCurrentLineExtents(int *min_x, int *max_x) { 
        *min_x=INT_MAX; *max_x=INT_MIN;
        for (auto i = out->data.begin() + cur_line.out_ind; i != out->data.end(); ++i) { Typed::Min(min_x, i->box.x); Typed::Max(max_x, i->box.right()); } 
    }
    void SetCurrentLineBounds() {
        cur_line.beg = container->baseleft (p.y, cur_line.height, &adj_float_left)  - container->x;
        cur_line.end = container->baseright(p.y, cur_line.height, &adj_float_right) - container->x;
        p.x = cur_line.beg;
    }
    void Complete() { if (!cur_line.fresh) AppendNewline(); }
};

#define TableFlowColIter(t) for (int j=0, cols=(t)->column.size(); j<cols; j++) if (TableFlow::Column *cj = &(t)->column[j])
struct TableFlow {
    struct Column {
        int width=0, min_width=0, max_width=0, last_ended_y=0, remaining_rowspan=0, remaining_height=0; void *remaining_val=0;
        void ResetHeight() { last_ended_y=remaining_rowspan=remaining_height=0; remaining_val=0; }
        void AddHeight(int height, int rowspan, void *val=0) {
            CHECK(!remaining_height && !remaining_rowspan);
            remaining_height=height; remaining_rowspan=rowspan; remaining_val=val;
        }
    };
    Flow *flow;
    vector<Column> column;
    int col_skipped=0, cell_width=0, max_cell_height=0, split_cell_height=0;
    TableFlow(Flow *F=0) : flow(F) {}

    void Select() { flow->layout.wrap_lines=0; }
    void SetMinColumnWidth(int j, int width, int colspan=1) {
        EnsureSize(column, j+colspan);
        if (width) for (int v=width/colspan, k=j; k<j+colspan; k++) Typed::Max(&column[k].width, v);
    }
    Column *SetCellDim(int j, int width, int colspan=1, int rowspan=1) {
        while (VectorEnsureElement(column, j+col_skipped)->remaining_rowspan) col_skipped++;
        SetMinColumnWidth(j+col_skipped, width, colspan);
        for (int k = 0; k < colspan; k++) column[j+col_skipped+k].remaining_rowspan += rowspan;
        return &column[j+col_skipped];
    }
    void NextRowDim() { col_skipped=0; TableFlowColIter(this) if (cj->remaining_rowspan) cj->remaining_rowspan--; }
    int ComputeWidth(int fixed_width) {
        int table_width = 0, auto_width_cols = 0, sum_column_width = 0;
        TableFlowColIter(this) { cj->ResetHeight(); if (cj->width) sum_column_width += cj->width; else auto_width_cols++; }
        if (fixed_width) {
            table_width = max(fixed_width, sum_column_width);
            int remaining = table_width - sum_column_width;
            if (remaining > 0) {
                if (auto_width_cols) { TableFlowColIter(this) if (!cj->width) cj->width += remaining/auto_width_cols; }
                else                 { TableFlowColIter(this)                 cj->width += remaining/cols; }
            }
        } else { 
            int min_table_width = 0, max_table_width = 0;
            TableFlowColIter(this) { 
                min_table_width += max(cj->width, cj->min_width);
                max_table_width += max(cj->width, cj->max_width);
            }
            bool maxfits = max_table_width < flow->container->w;
            table_width = maxfits ? max_table_width : min_table_width;
            TableFlowColIter(this) {
                cj->width = maxfits ? max(cj->width, cj->max_width) : max(cj->width, cj->min_width);
            }
        }
        return table_width;
    }
    void AppendCell(int j, Box *out, int colspan=1) {
        TableFlow::Column *cj = 0;
        for (;;col_skipped++) {
            if (!(cj = VectorCheckElement(column, j+col_skipped))->remaining_rowspan) break;
            flow->AppendBox(cj->width, 0, (Box*)0);
        }
        cell_width = 0;
        CHECK_LE(j+col_skipped+colspan, column.size());
        for (int k=j+col_skipped, l=k+colspan; k<l; k++) cell_width += column[k].width;
        flow->AppendBox(cell_width, 0, out);
    }
    void SetCellHeight(int j, int cellheight, void *cell, int colspan=1, int rowspan=1) {
        column[j+col_skipped].AddHeight(cellheight, rowspan, cell);
        for (int k = 1; k < colspan; k++) column[j+col_skipped+k].remaining_rowspan = rowspan;
        col_skipped += colspan-1;

        if (rowspan == 1)   max_cell_height = max(max_cell_height,   cellheight);
        else              split_cell_height = max(split_cell_height, cellheight / rowspan);
    }
    int AppendRow() {
        if (!max_cell_height) max_cell_height = split_cell_height;
        TableFlowColIter(this) {
            cj->remaining_rowspan = max(0, cj->remaining_rowspan - 1);
            if (!cj->remaining_rowspan) Typed::Max(&max_cell_height, cj->remaining_height);
        }
        TableFlowColIter(this) {
            int subtracted = min(max_cell_height, cj->remaining_height);
            if (subtracted) cj->remaining_height -= subtracted;
        }
        flow->AppendBox(1, max_cell_height, (Box*)0);
        flow->AppendNewline();
        int ret = max_cell_height;
        col_skipped = cell_width = max_cell_height = split_cell_height = 0;
        return ret;
    }
};

struct Atlas {
    Texture tex; Box dim; Flow flow;
    Atlas(unsigned T, int W, int H=0) : tex(W, H?H:W, Pixel::RGBA, T), dim(tex.width, tex.height), flow(&dim) {}
    bool Add(int *x_out, int *y_out, float *out_texcoord, int w, int h, int max_height=0);
    void Update(const string &name, Font *glyphs, bool dump);

    static void WriteGlyphFile(const string &name, Font *glyphs);
    static void MakeFromPNGFiles(const string &name, const vector<string> &png, int atlas_dim, Font **glyphs_out);
    static void SplitIntoPNGFiles(const string &input_png_fn, const map<int, v4> &glyphs, const string &dir_out);
    static int Dimension(int n, int w, int h) { return 1 << max(8,FloorLog2(sqrt((w+4)*(h+4)*n))); }
};

}; // namespace LFL
#endif // __LFL_LFAPP_VIDEO_H__
