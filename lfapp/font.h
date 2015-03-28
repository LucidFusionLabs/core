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

#ifndef __LFL_LFAPP_FONT_H__
#define __LFL_LFAPP_FONT_H__

namespace LFL {
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
DECLARE_int(glyph_table_size);

struct FontDesc {
    enum { Bold=1, Italic=2, Mono=4, Outline=8 };
    string name, family;
    int size, flag;
    Color fg, bg;
    FontDesc(const string &n="", const string &fam="", int s=0,
             const Color &fgc=Color::white,
             const Color &bgc=Color::clear, int f=-1) :
        name(n), family(fam), size(s), fg(fgc), bg(bgc), flag(f == -1 ? FLAGS_default_font_flag : f) {}
    bool operator==(const FontDesc &x) const {
        return name == x.name && family == x.family && size == x.size && flag == x.flag && fg == x.fg && bg == x.bg;
    }
};

}; // namespace LFL
namespace std {
    template <> struct hash<LFL::FontDesc> {
        size_t operator()(const LFL::FontDesc &v) const {
            unsigned h = 0, fgc = v.fg.AsUnsigned(), bgc = v.bg.AsUnsigned();
            h = LFL::fnv32(v.name  .data(), v.name  .size(), h);
            h = LFL::fnv32(v.family.data(), v.family.size(), h);
            h = LFL::fnv32(&v.size,         sizeof(v.size),  h);
            h = LFL::fnv32(&v.flag,         sizeof(v.flag),  h);
            h = LFL::fnv32(&fgc,            sizeof(fgc),     h);
            h = LFL::fnv32(&bgc,            sizeof(bgc),     h);
            return h;
        }
    };
}; // namespace std;
namespace LFL {

struct FontEngine {
    struct Resource { virtual ~Resource() {} };
    virtual const char *Name() = 0;
    virtual bool  Init(const FontDesc&) { return true; }
    virtual Font *Open(const FontDesc&) = 0;
    virtual bool  HaveGlyph (Font *f, unsigned short) { return true; }
    virtual int   InitGlyphs(Font *f,       Glyph *g, int n) = 0;
    virtual int   LoadGlyphs(Font *f, const Glyph *g, int n) = 0;
};

struct Glyph : public Drawable {
    unsigned short id=0;
    short bearing_x=0, bearing_y=0, advance=0;
    union Internal {
        struct FreeType { int id; }                                          freetype;
        struct CoreText { int id; float origin_x, origin_y, width, height; } coretext;
    } internal;
    mutable Texture tex;
    mutable bool ready=0;
    Glyph() { memzero(internal); }

    bool operator<(const Glyph &y) const { return id < y.id; }
    void FromArray(const double *in,  int l);
    int  ToArray  (      double *out, int l);

    virtual int  Id() const { return id; }
    virtual int  Ascender   (const LFL::Box *b, const Drawable::Attr *a=0) const;
    virtual int  Advance    (const LFL::Box *b, const Drawable::Attr *a=0) const;
    virtual int  LeftBearing(                   const Drawable::Attr *a=0) const;
    virtual int  Layout     (      LFL::Box *b, const Drawable::Attr *a=0) const;
    virtual void Draw       (const LFL::Box &b, const Drawable::Attr *a=0) const;
};

struct GlyphCache {
    typedef function<void(const Box&, unsigned char *buf, int linesize, int pf)> FilterCB;
    Box dim;
    Texture tex;
    Flow *flow=0;
    CGContextRef cgcontext=0;
    GlyphCache(unsigned T, int W, int H=0);
    ~GlyphCache();

    bool Add(point *out, float *out_texcoord, int w, int h, int max_height=0);
    void Upload(const Glyph *g, const point &p, const unsigned char *buf, int linesize, int pf, const FilterCB &f=FilterCB());
    void Upload(const Glyph *g, const point &p, CGFontRef cgfont, int size);
};

struct GlyphMap {
    int table_start=0;
    vector<Glyph>             table;
    unordered_map<int, Glyph> index;
    shared_ptr<GlyphCache>    cache;
    GlyphMap(const shared_ptr<GlyphCache> &C = shared_ptr<GlyphCache>()) : table(FLAGS_glyph_table_size), cache(C) {
        for (auto b = table.begin(), g = b, e = table.end(); g != e; ++g) g->id = table_start + g - b;
    }
};
#define GlyphTableIter(f) for (auto i = (f)->glyph->table.begin(); i != (f)->glyph->table.end(); ++i)
#define GlyphIndexIter(f) for (auto i = (f)->glyph->index.begin(); i != (f)->glyph->index.end(); ++i)

struct Font {
    struct DrawFlag {
        enum {
            NoWrap=1<<6, GlyphBreak=1<<7, AlignCenter=1<<8, AlignRight=1<<9, 
            Underline=1<<10, Overline=1<<11, Midline=1<<12, Blink=1<<13,
            Uppercase=1<<14, Lowercase=1<<15, Capitalize=1<<16, Clipped=1<<17,
            AssignFlowX=1<<18
        };
        static int Orientation(int f) { return f & 0xf; };
    };

    short size=0, ascender=0, descender=0, max_width=0, fixed_width=-1, missing_glyph=0;
    bool mono=0, mix_fg=0;
    float scale=0;
    int flag=0;
    Color fg, bg;
    RefCounter ref;
    FontEngine *engine;
    shared_ptr<GlyphMap> glyph;
    shared_ptr<FontEngine::Resource> resource;

    virtual ~Font() {}
    Font(FontEngine *E, const FontDesc &D, const shared_ptr<FontEngine::Resource> &R) :
        size(D.size), missing_glyph(FLAGS_default_missing_glyph), mono(D.flag & FontDesc::Mono),
        flag(D.flag), fg(D.fg), bg(D.bg), engine(E), resource(R) {}

    short Height() const { return ascender + descender; }
    short FixedWidth() const { return X_or_Y(fixed_width, mono ? max_width : 0); }

    Glyph *FindGlyph        (unsigned short gind);
    Glyph *FindOrInsertGlyph(unsigned short gind);

    void Select();
    void DrawGlyph(int g, const Box &w) { Select(); Drawable::Attr a(this); FindGlyph(g)->Draw(w, &a); }
    int GetGlyphWidth(int g) { return RoundXY_or_Y(scale, FindGlyph(g)->advance); }

    void UpdateMetrics(const Glyph *g) {
        int descent = g->tex.height - g->bearing_y;
        if (g->advance && fixed_width == -1)         fixed_width = g->advance;
        if (g->advance && fixed_width != g->advance) fixed_width = 0;
        if (g->advance   > max_width) max_width = g->advance;
        if (g->bearing_y > ascender)  ascender  = g->bearing_y;
        if (descent      > descender) descender = descent;
    }

    template <class X> void Size(const StringPieceT<X> &text, Box *out, int width=0, int *lines_out=0);
    /**/               void Size(const string          &text, Box *out, int width=0, int *lines_out=0) { return Size(StringPiece           (text), out, width, lines_out); }
    /**/               void Size(const String16        &text, Box *out, int width=0, int *lines_out=0) { return Size(String16Piece         (text), out, width, lines_out); }
    template <class X> void Size(const X               *text, Box *out, int width=0, int *lines_out=0) { return Size(StringPiece::Unbounded(text), out, width, lines_out); }

    template <class X> int Lines(const StringPieceT<X> &text, int width) { if (!width) return 1; Box b; Size(text, &b, width); return b.h / Height(); }
    /**/               int Lines(const string          &text, int width) { return Lines(StringPiece           (text), width); }
    /**/               int Lines(const String16        &text, int width) { return Lines(String16Piece         (text), width); }
    template <class X> int Lines(const X               *text, int width) { return Lines(StringPiece::Unbounded(text), width); }

    template <class X> int Width(const StringPieceT<X> &text) { Box b; Size(text, &b); if (b.w) CHECK_EQ(b.h, Height()); return b.w; }
    /**/               int Width(const string          &text) { return Width(StringPiece           (text)); }
    /**/               int Width(const String16        &text) { return Width(String16Piece         (text)); }
    template <class X> int Width(const X               *text) { return Width(StringPiece::Unbounded(text)); }

    template <class X> void Encode(const StringPieceT<X> &text, const Box &box, BoxArray *out, int draw_flag=0, int attr_id=0);
    /**/               void Encode(const string          &text, const Box &box, BoxArray *out, int draw_flag=0, int attr_id=0) { return Encode(StringPiece           (text), box, out, draw_flag, attr_id); }
    /**/               void Encode(const String16        &text, const Box &box, BoxArray *out, int draw_flag=0, int attr_id=0) { return Encode(String16Piece         (text), box, out, draw_flag, attr_id); }
    template <class X> void Encode(const X               *text, const Box &box, BoxArray *out, int draw_flag=0, int attr_id=0) { return Encode(StringPiece::Unbounded(text), box, out, draw_flag, attr_id); }

    template <class X> int Draw(const StringPieceT<X> &text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw<X>    (                text,  Box(cp.x,cp.y+Height(),0,0), lb, draw_flag); }
    /**/               int Draw(const string          &text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw(StringPiece           (text), Box(cp.x,cp.y+Height(),0,0), lb, draw_flag); }
    /**/               int Draw(const String16        &text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw(String16Piece         (text), Box(cp.x,cp.y+Height(),0,0), lb, draw_flag); }
    template <class X> int Draw(const X               *text, point cp,       vector<Box> *lb=0, int draw_flag=0) { return Draw(StringPiece::Unbounded(text), Box(cp.x,cp.y+Height(),0,0), lb, draw_flag); }
    
    template <class X> int Draw(const StringPieceT<X> &text, const Box &box, vector<Box> *lb=0, int draw_flag=0);
    /**/               int Draw(const string          &text, const Box &box, vector<Box> *lb=0, int draw_flag=0) { return Draw(StringPiece           (text), box, lb, draw_flag); }
    /**/               int Draw(const String16        &text, const Box &box, vector<Box> *lb=0, int draw_flag=0) { return Draw(String16Piece         (text), box, lb, draw_flag); }
    template <class X> int Draw(const X               *text, const Box &box, vector<Box> *lb=0, int draw_flag=0) { return Draw(StringPiece::Unbounded(text), box, lb, draw_flag); }
};

struct FakeFontEngine : public FontEngine {
    FontDesc fake_font_desc;
    Font fake_font;
    FakeFontEngine() : fake_font(this, fake_font_desc, shared_ptr<FontEngine::Resource>()) {
        fake_font_desc.size = 10;
        fake_font.fixed_width = fake_font.max_width = 8;
        fake_font.ascender = 9;
        fake_font.descender = 5;
        fake_font.glyph = shared_ptr<GlyphMap>(new GlyphMap(shared_ptr<GlyphCache>(new GlyphCache(0, 0))));
        InitGlyphs(&fake_font, &fake_font.glyph->table[0], fake_font.glyph->table.size());
    }
    virtual const char *Name() { return "FakeFontEngine"; }
    virtual Font *Open(const FontDesc&) { return &fake_font; }
    virtual int  LoadGlyphs(Font *f, const Glyph *g, int n) { return n; }
    virtual int  InitGlyphs(Font *f,       Glyph *g, int n) {
        for (Glyph *e = g + n; g != e; ++g) {
            g->tex.height = g->bearing_y = fake_font.Height();
            g->tex.width  = g->advance   = fake_font.fixed_width;
        } return n;
    }
    static const char *Filename() { return "__FakeFontFilename__"; }
};

struct AtlasFontEngine : public FontEngine {
    struct Resource : public FontEngine::Resource { Font *primary=0; };
    typedef map<unsigned, Font*> FontSizeMap;
    typedef map<unsigned, FontSizeMap> FontFlagMap;
    typedef map<unsigned, FontFlagMap> FontColorMap;
    typedef map<string, FontColorMap> FontMap;
    FontMap font_map;
    virtual const char *Name() { return "AtlasFontEngine"; }
    virtual bool  Init(const FontDesc&);
    virtual Font *Open(const FontDesc&);
    virtual bool  HaveGlyph (Font *f, unsigned short) { return false; }
    virtual int   InitGlyphs(Font *f,       Glyph *g, int n) { return n; }
    virtual int   LoadGlyphs(Font *f, const Glyph *g, int n) { return n; }

    static Font *OpenAtlas(const FontDesc&);
    static void WriteAtlas(const string &name, Font *glyphs, Texture *t);
    static void WriteAtlas(const string &name, Font *glyphs);
    static void WriteGlyphFile(const string &name, Font *glyphs);
    static void MakeFromPNGFiles(const string &name, const vector<string> &png, int atlas_dim, Font **glyphs_out);
    static void SplitIntoPNGFiles(const string &input_png_fn, const map<int, v4> &glyphs, const string &dir_out);
    static int Dimension(int n, int w, int h) { return 1 << max(8,FloorLog2(sqrt((w+4)*(h+4)*n))); }
};

#ifdef LFL_FREETYPE
struct FreeTypeFontEngine : public FontEngine {
    struct Resource : public FontEngine::Resource {
        string name, content;
        FT_FaceRec_ *face=0;
        virtual ~Resource();
        Resource(FT_FaceRec_ *FR=0, const string &N="", string *C=0) : face(FR), name(N) { if (C) swap(*C, content); }
    };
    unordered_map<string, shared_ptr<Resource> > resource;
    GlyphCache::FilterCB subpixel_filter = &FreeTypeFontEngine::SubPixelFilter;
    virtual const char *Name() { return "FreeTypeFontEngine"; }
    virtual bool  Init(const FontDesc&);
    virtual Font *Open(const FontDesc&);
    virtual int   InitGlyphs(Font *f,       Glyph *g, int n);
    virtual int   LoadGlyphs(Font *f, const Glyph *g, int n);

    static void Init();
    static void SubPixelFilter(const Box &b, unsigned char *buf, int linesize, int pf);
    static Resource *OpenFile  (const FontDesc&);
    static Resource *OpenBuffer(const FontDesc&, string *content);
};
#endif

#ifdef __APPLE__
struct CoreTextFontEngine : public FontEngine {
    struct Resource : public FontEngine::Resource {
        string name;
        CGFontRef cgfont;
        int flag;
        virtual ~Resource();
        Resource(const char *N=0, CGFontRef CGF=0, int F=0) : name(BlankNull(N)), cgfont(CGF), flag(F) {}
    };
    unordered_map<string, shared_ptr<Resource> > resource;
    virtual const char *Name() { return "CoreTextFontEngine"; }
    virtual Font *Open(const FontDesc&);
    virtual int   InitGlyphs(Font *f,       Glyph *g, int n);
    virtual int   LoadGlyphs(Font *f, const Glyph *g, int n);

    struct Flag { enum { WriteAtlas=1 }; };
    static Font *Open(const string &name,            int size, Color c, int flag, int ct_flag);
    static Font *Open(const shared_ptr<Resource> &R, int size, Color c, int flag);
    static void GetSubstitutedFont(CTFontRef, unsigned short gid, CGFontRef *cgout, CTFontRef *ctout, int *id_out);
    static void AssignGlyph(Glyph *out, const CGRect &bounds, struct CGSize &advance);
};
#endif

struct Fonts {
    struct Family {
        vector<string> normal, bold, italic, bold_italic;
        void Add(const string &n, int flag) {
            bool is_bold = flag & FontDesc::Bold, is_italic = flag & FontDesc::Italic;
            if (is_bold && is_italic) bold_italic.push_back(n);
            else if (is_bold)         bold       .push_back(n);
            else if (is_italic)       italic     .push_back(n);
            else                      normal     .push_back(n);
        }
    };
    unordered_map<FontDesc, Font*> desc_map;
    unordered_map<string, Family> family_map;
    Font *FindOrInsert(FontEngine *engine, const FontDesc &d);

    static FontEngine *DefaultFontEngine();
    static Font *Default();
    static Font *Fake();
    static Font *GetByDesc(FontEngine*, FontDesc);
    template <class... Args> static Font *Get(Args&&... args) { return GetByDesc(DefaultFontEngine(), FontDesc(args...)); }
    static int ScaledFontSize(int pointsize);
};

}; // namespace LFL
#endif // __LFL_LFAPP_FONT_H__
