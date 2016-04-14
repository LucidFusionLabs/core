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

#ifndef LFL_CORE_APP_FONT_H__
#define LFL_CORE_APP_FONT_H__

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
DECLARE_int(glyph_table_start);
DECLARE_bool(subpixel_fonts);

struct FontDesc {
  enum { Bold=1, Italic=2, Mono=4, Outline=8, Shadow=16 };
  struct Engine {
    enum { Default=0, Atlas=1, FreeType=2, CoreText=3, GDI=4 };
    static int Parse(const string &s) {
      if      (s == "atlas")    return Atlas;
      else if (s == "freetype") return FreeType;
      else if (s == "coretext") return CoreText;
      else if (s == "gdi")      return GDI;
      return Default;
    };
  };
  struct Hasher {
    size_t operator()(const FontDesc &v) const {
      size_t h = fnv32(v.name  .data(), v.name  .size(), 0);
      /**/   h = fnv32(v.family.data(), v.family.size(), h);
      /**/   h = fnv32(&v.size,         sizeof(v.size),  h);
      return     fnv32(&v.flag,         sizeof(v.flag),  h);
    }
  };
  struct ColoredHasher {
    size_t operator()(const FontDesc &v) const {
      size_t h = Hasher()(v), fgc = v.fg.AsUnsigned(), bgc = v.bg.AsUnsigned();
      /**/ h = fnv32(&fgc, sizeof(fgc), h);
      return   fnv32(&bgc, sizeof(bgc), h);
    }
  };
  struct Equal {
    bool operator()(const FontDesc &x, const FontDesc &y) const {
      return x.name == y.name && x.family == y.family && x.size == y.size && x.flag == y.flag;
    }
  };
  struct ColoredEqual {
    bool operator()(const FontDesc &x, const FontDesc &y) const {
      return Equal()(x, y) && x.fg == y.fg && x.bg == y.bg;
    } 
  };

  string name, family;
  int size, flag, engine;
  Color fg, bg;
  bool unicode;

  FontDesc(const IPC::FontDescription&);
  FontDesc(const string &n="", const string &fam="", int s=0,
           const Color &fgc=Color::white,
           const Color &bgc=Color::clear, int f=-1, bool U=1, int E=0) :
    family(fam), size(s), flag(f == -1 ? FLAGS_default_font_flag : f), engine(E), fg(fgc), bg(bgc), unicode(U)
    {
      string engine_proto;
      name = ParseProtocol(n.data(), &engine_proto);
      if (!engine_proto.empty()) engine = Engine::Parse(engine_proto);
    }

  string Filename() const {
    return StrCat(name, ",", size, ",", fg.R(), ",", fg.G(), ",", fg.B(), ",", flag);
  }
  string DebugString() const {
    return StrCat(name, " (", family, ") ", size, " ", fg.DebugString(), " ", bg.DebugString(), " ", flag);
  }

  static FontDesc Default() {
    return FontDesc(FLAGS_default_font, FLAGS_default_font_family, FLAGS_default_font_size,
                    Color::white, Color::clear, FLAGS_default_font_flag);
  }
};

struct FontEngine {
  struct Resource { virtual ~Resource() {} };
  virtual const char*      Name() = 0;
  virtual void             Shutdown() {}
  virtual void             SetDefault() {}
  virtual bool             Init(const FontDesc&) { return true; }
  virtual unique_ptr<Font> Open(const FontDesc&) = 0;
  virtual bool             HaveGlyph (Font *f, char16_t) { return true; }
  virtual int              InitGlyphs(Font *f,       Glyph *g, int n) = 0;
  virtual int              LoadGlyphs(Font *f, const Glyph *g, int n) = 0;
  virtual string           DebugString(Font *f) const = 0;
};

struct Glyph : public Drawable {
  char16_t id=0;
  short bearing_x=0, bearing_y=0, advance=0;
  bool wide=0, space=0;
  union Internal {
    struct FreeType  { int id; }                                                   freetype;
    struct IPCClient { int id; }                                                   ipcclient;
    struct CoreText  { int id; float origin_x, origin_y, width, height, advance; } coretext;
  } internal;
  mutable Texture tex;
  mutable bool ready=0;
  Glyph() { memzero(internal); tex.owner=0; }

  bool operator<(const Glyph &y) const { return id < y.id; }
  void FromMetrics(const GlyphMetrics &m);
  void FromArray(const double *in,  int l);
  int  ToArray  (      double *out, int l);

  virtual int  Id()    const { return id; }
  virtual int  TexId() const { return tex.ID; }
  virtual bool Wide()  const { return wide; }
  virtual int  Baseline   (const LFL::Box *b, const Drawable::Attr *a=0) const;
  virtual int  Ascender   (const LFL::Box *b, const Drawable::Attr *a=0) const;
  virtual int  Advance    (const LFL::Box *b, const Drawable::Attr *a=0) const;
  virtual int  LeftBearing(                   const Drawable::Attr *a=0) const;
  virtual int  Layout     (      LFL::Box *b, const Drawable::Attr *a=0) const;
  virtual void Draw       (const LFL::Box &b, const Drawable::Attr *a=0) const;
};

struct GlyphMetrics {
  char16_t id=0;
  short width=0, height=0, bearing_x=0, bearing_y=0, advance=0;
  bool wide=0, space=0;
  int tex_id=0;
  GlyphMetrics() {}
  GlyphMetrics(const Glyph &g) : id(g.id), width(g.tex.width), height(g.tex.height), bearing_x(g.bearing_x),
    bearing_y(g.bearing_y), advance(g.advance), wide(g.wide), space(g.space), tex_id(g.tex.ID) {}
};

struct GlyphCache {
  typedef function<void(const Box&, unsigned char *buf, int linesize, int pf)> FilterCB;
  Box dim;
  Texture tex;
  unique_ptr<Flow> flow;
#ifdef LFL_APPLE
  CGContextRef cgcontext=0;
#endif
#ifdef LFL_WINDOWS
  HDC hdc=0;
#endif
  vector<const Glyph*> glyph;
  int max_width=128, max_height=128;
  GlyphCache(unsigned T, int W, int H=0);
  ~GlyphCache();

  bool ShouldCacheGlyph(const Texture &t) const {
    CHECK_LT(t.width,  1024*1024);
    CHECK_LT(t.height, 1024*1024);
    return t.width < max_width && t.height < max_height;
  }

  void Clear();
  bool Add(point *out, float *out_texcoord, int w, int h, int max_height=0);
  void Load(const Font*, const Glyph*, const unsigned char *buf, int linesize, int pf, const FilterCB &f=FilterCB());
#ifdef LFL_APPLE
  void Load(const Font*, const Glyph*, CGFontRef cgfont, int size);
#endif
#ifdef LFL_WINDOWS
  void Load(const Font*, const Glyph*, HFONT hfont, int size, HDC dc);
#endif

  static shared_ptr<GlyphCache> Get() {
    static shared_ptr<GlyphCache> inst = make_shared<GlyphCache>(0, 512);
    if (!inst->tex.ID) inst->tex.Create(inst->tex.width, inst->tex.height);
    return inst;
  }
};

struct GlyphMap {
  int table_start;
  vector<Glyph>             table;
  unordered_map<int, Glyph> index;
  shared_ptr<GlyphCache>    cache;
  GlyphMap(shared_ptr<GlyphCache> C = shared_ptr<GlyphCache>()) :
    table_start(FLAGS_glyph_table_start), table(FLAGS_glyph_table_size), cache(move(C))
  { for (auto b = table.begin(), g = b, e = table.end(); g != e; ++g) g->id = table_start + (g - b); }
};

struct Font {
  struct DrawFlag {
    enum {
      NoWrap=1<<6, GlyphBreak=1<<7, AlignCenter=1<<8, AlignRight=1<<9, 
      Underline=1<<10, Overline=1<<11, Midline=1<<12, Blink=1<<13,
      Uppercase=1<<14, Lowercase=1<<15, Capitalize=1<<16, Clipped=1<<17,
      DontAssignFlowP=1<<18, DontCompleteFlow=1<<19
    };
    static int Orientation(int f) { return f & 0xf; };
  };

  short size=0, ascender=0, descender=0, max_width=0, fixed_width=-1, missing_glyph=0;
  bool mono=0, mix_fg=0, has_bg=0, fix_metrics=0;
  float scale=0;
  int flag=0;
  Color fg, bg;
  RefCounter ref;
  const FontDesc *desc=0;
  FontEngine* const engine;
  shared_ptr<GlyphMap> glyph;
  shared_ptr<FontEngine::Resource> resource;

  virtual ~Font() {}
  Font(FontEngine *E, const FontDesc &D, shared_ptr<FontEngine::Resource> R) :
    size(D.size), missing_glyph(FLAGS_default_missing_glyph), mono(D.flag & FontDesc::Mono),
    flag(D.flag), fg(D.fg), bg(D.bg), engine(E), resource(move(R)) {}

  short Height() const { return ascender + descender; }
  short FixedWidth() const { return X_or_Y(fixed_width, mono ? max_width : 0); }

  Glyph *FindGlyph        (char16_t gind);
  Glyph *FindOrInsertGlyph(char16_t gind);

  void Select();
  void UpdateMetrics(Glyph *g);
  void SetMetrics(short a, short d, short mw, short fw, short mg, bool mf, bool hb, bool fm, float f)
  { ascender=a; descender=d; max_width=mw; fixed_width=fw; missing_glyph=mg; mix_fg=mf; has_bg=hb; fix_metrics=fm; scale=f; }
  int GetGlyphWidth(int g) { return RoundXY_or_Y(scale, FindGlyph(g)->advance); }
  void DrawGlyph(int g, const Box &w) { return DrawGlyphWithAttr(g, w, Drawable::Attr(this)); }
  void DrawGlyphWithAttr(int g, const Box &w, const Drawable::Attr&);

  template <class X> void Size(const StringPieceT<X> &text, Box *out, int width=0, int flag=0, int *lines_out=0);
  /**/               void Size(const string          &text, Box *out, int width=0, int flag=0, int *lines_out=0) { return Size(StringPiece           (text), out, width, flag, lines_out); }
  /**/               void Size(const String16        &text, Box *out, int width=0, int flag=0, int *lines_out=0) { return Size(String16Piece         (text), out, width, flag, lines_out); }
  template <class X> void Size(const X               *text, Box *out, int width=0, int flag=0, int *lines_out=0) { return Size(StringPiece::Unbounded(text), out, width, flag, lines_out); }

  template <class X> int Lines(const StringPieceT<X> &text, int width, bool word_wrap=1) { if (!width) return 1; Box b; Size(text, &b, width, word_wrap?0:DrawFlag::GlyphBreak); return b.h / Height(); }
  /**/               int Lines(const string          &text, int width, bool word_wrap=1) { return Lines(StringPiece           (text), width, word_wrap); }
  /**/               int Lines(const String16        &text, int width, bool word_wrap=1) { return Lines(String16Piece         (text), width, word_wrap); }
  template <class X> int Lines(const X               *text, int width, bool word_wrap=1) { return Lines(StringPiece::Unbounded(text), width, word_wrap); }

  template <class X> int Width(const StringPieceT<X> &text) { Box b; Size(text, &b); if (b.w) CHECK_EQ(b.h, Height()); return b.w; }
  /**/               int Width(const string          &text) { return Width(StringPiece           (text)); }
  /**/               int Width(const String16        &text) { return Width(String16Piece         (text)); }
  template <class X> int Width(const X               *text) { return Width(StringPiece::Unbounded(text)); }

  template <class X> void Shape(const StringPieceT<X> &text, const Box &box, DrawableBoxArray *out, int draw_flag=0, int attr_id=0);
  /**/               void Shape(const string          &text, const Box &box, DrawableBoxArray *out, int draw_flag=0, int attr_id=0) { return Shape(StringPiece           (text), box, out, draw_flag, attr_id); }
  /**/               void Shape(const String16        &text, const Box &box, DrawableBoxArray *out, int draw_flag=0, int attr_id=0) { return Shape(String16Piece         (text), box, out, draw_flag, attr_id); }
  template <class X> void Shape(const X               *text, const Box &box, DrawableBoxArray *out, int draw_flag=0, int attr_id=0) { return Shape(StringPiece::Unbounded(text), box, out, draw_flag, attr_id); }

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
  static const int size = 10, fixed_width = 8, ascender = 9, descender = 5;
  static const unsigned char wide_glyph_begin = 0xf0, wide_glyph_end = 0xff;
  FontDesc fake_font_desc;
  Font fake_font;
  FakeFontEngine();
  virtual const char *Name() { return "FakeFontEngine"; }
  virtual unique_ptr<Font> Open(const FontDesc&) { return make_unique<Font>(fake_font); }
  virtual int LoadGlyphs(Font *f, const Glyph *g, int n) { return n; }
  virtual int InitGlyphs(Font *f,       Glyph *g, int n);
  virtual string DebugString(Font *f) const { return "FakeFontEngineFont"; }
  static const char *Filename() { return "__FakeFontFilename__"; }
};

struct AtlasFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource { Font *primary=0; };
  typedef map<size_t, Font*> FontSizeMap;
  typedef map<size_t, FontSizeMap> FontFlagMap;
  typedef map<size_t, FontFlagMap> FontColorMap;
  typedef map<string, FontColorMap> FontMap;
  FontMap font_map;
  bool in_init=0;

  virtual const char*      Name() { return "AtlasFontEngine"; }
  virtual void             SetDefault();
  virtual bool             Init(const FontDesc&);
  virtual unique_ptr<Font> Open(const FontDesc&);
  virtual bool             HaveGlyph (Font *f, char16_t) { return false; }
  virtual int              InitGlyphs(Font *f,       Glyph *g, int n) { return n; }
  virtual int              LoadGlyphs(Font *f, const Glyph *g, int n) { return n; }
  virtual string           DebugString(Font *f) const;

  static Font *OpenAtlas(const FontDesc&);
  static void WriteAtlas(const string &name, Font *glyphs, Texture *t);
  static void WriteAtlas(const string &name, Font *glyphs);
  static void WriteGlyphFile(const string &name, Font *glyphs);
  static void MakeFromPNGFiles(const string &name, const vector<string> &png, const point &atlas_dim, Font **glyphs_out);
  static void SplitIntoPNGFiles(const string &input_png_fn, const map<int, v4> &glyphs, const string &dir_out);
  static int Dimension(int n, int w, int h) { return 1 << max(8,FloorLog2(sqrt((w+4)*(h+4)*n))); }
};

struct FreeTypeFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource {
    string name, content;
    FT_FaceRec_ *face=0;
    virtual ~Resource();
    Resource(FT_FaceRec_ *FR=0, const string &N="", string *C=0) : face(FR), name(N) { if (C) swap(*C, content); }
  };
  unordered_map<FontDesc, Font*, FontDesc::Hasher, FontDesc::Equal> font_map;
  unordered_map<string, shared_ptr<Resource> > resource;
  GlyphCache::FilterCB subpixel_filter = &FreeTypeFontEngine::SubPixelFilter;

  virtual const char*      Name() { return "FreeTypeFontEngine"; }
  virtual void             SetDefault();
  virtual bool             Init(const FontDesc&);
  virtual unique_ptr<Font> Open(const FontDesc&);
  virtual int              InitGlyphs(Font *f,       Glyph *g, int n);
  virtual int              LoadGlyphs(Font *f, const Glyph *g, int n);
  virtual string           DebugString(Font *f) const;

  static void Init();
  static void SubPixelFilter(const Box &b, unsigned char *buf, int linesize, int pf);
  static Resource *OpenFile  (const FontDesc&);
  static Resource *OpenBuffer(const FontDesc&, string *content);
};

#ifdef LFL_APPLE
struct CoreTextFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource {
    string name;
    CGFontRef cgfont;
    hb_face_t *hb_face=0;
    int flag;
    virtual ~Resource();
    Resource(const char *N=0, CGFontRef CGF=0, int F=0) : name(BlankNull(N)), cgfont(CGF), flag(F) {}
  };
  unordered_map<string, shared_ptr<Resource> > resource;

  virtual const char*      Name() { return "CoreTextFontEngine"; }
  virtual void             SetDefault();
  virtual unique_ptr<Font> Open(const FontDesc&);
  virtual int              InitGlyphs(Font *f,       Glyph *g, int n);
  virtual int              LoadGlyphs(Font *f, const Glyph *g, int n);
  virtual string           DebugString(Font *f) const;

  struct Flag { enum { WriteAtlas=1 }; };
  static Font *Open(const string &name, int size, Color c, int flag, int ct_flag);
  static void GetSubstitutedFont(Font*, CTFontRef, char16_t gid, CGFontRef *cgout, CTFontRef *ctout, int *id_out);
  static void AssignGlyph(Glyph *out, const CGRect &bounds, struct CGSize &advance);
  static v2 GetAdvanceBounds(Font*);
};
#else
struct CoreTextFontEngine {};
#endif

#ifdef LFL_WINDOWS
struct GDIFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource {
    string name;
    HFONT hfont;
    int flag;
    virtual ~Resource();
    Resource(const char *N = 0, HFONT H = 0, int F = 0) : name(BlankNull(N)), hfont(H), flag(F) {}
  };
  unordered_map<string, shared_ptr<Resource> > resource;
  HDC hdc=0;
  GDIFontEngine();
  ~GDIFontEngine();

  virtual const char*      Name() { return "GDIFontEngine"; }
  virtual void             Shutdown();
  virtual void             SetDefault();
  virtual unique_ptr<Font> Open(const FontDesc&);
  virtual int              InitGlyphs(Font *f, Glyph *g, int n);
  virtual int              LoadGlyphs(Font *f, const Glyph *g, int n);
  virtual string           DebugString(Font *f) const;

  struct Flag { enum { WriteAtlas = 1 }; };
  static Font *Open(const string &name, int size, Color c, int flag, int ct_flag);
  static bool GetSubstitutedFont(Font *f, HFONT hfont, char16_t glyph_id, HDC hdc, HFONT *hfontout);
  static void AssignGlyph(Glyph *out, const ::SIZE &bounds, const ::SIZE &advance);
};
#else
struct GDIFontEngine {};
#endif

struct IPCClientFontEngine : public FontEngine {
  struct Resource : public FontEngine::Resource { int id; Resource(int I=0) : id(I) {} };
  virtual const char *Name() { return "IPCClientFontEngine"; }
  virtual unique_ptr<Font> Open(const FontDesc&);
  virtual int InitGlyphs(Font *f, Glyph *g, int n);
  virtual int LoadGlyphs(Font *f, const Glyph *g, int n);
  string DebugString(Font *f) const;
  int OpenSystemFontResponse(Font *f, const IPC::OpenSystemFontResponse*, const MultiProcessBuffer&);
  static int GetId(Font *f);
};

struct IPCServerFontEngine : public FontEngine {
  virtual const char *Name() { return "IPCServerFontEngine"; }
  virtual unique_ptr<Font> Open(const FontDesc&);
  virtual int InitGlyphs(Font *f, Glyph *g, int n);
  virtual int LoadGlyphs(Font *f, const Glyph *g, int n);
  string DebugString(Font *f) const;
};

struct Fonts {
  static int InitFontWidth();
  static int InitFontHeight();
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

  FontEngine *default_font_engine=0;
  LazyInitializedPtr<FakeFontEngine> fake_engine;
  LazyInitializedPtr<AtlasFontEngine> atlas_engine;
  LazyInitializedPtr<FreeTypeFontEngine> freetype_engine;
  LazyInitializedPtr<CoreTextFontEngine> coretext_engine;
  LazyInitializedPtr<GDIFontEngine> gdi_engine;
  LazyInitializedPtr<IPCClientFontEngine> ipc_client_engine;
  LazyInitializedPtr<IPCServerFontEngine> ipc_server_engine;
  unordered_map<FontDesc, unique_ptr<Font>, FontDesc::ColoredHasher, FontDesc::ColoredEqual> desc_map;
  unordered_map<string, Family> family_map;
  Font *default_font=0;

  Font *Find        (                    const FontDesc &d);
  Font *Insert      (FontEngine *engine, const FontDesc &d);
  Font *FindOrInsert(FontEngine *engine, const FontDesc &d);

  FontEngine *GetFontEngine(int engine_type);
  FontEngine *DefaultFontEngine();
  Font *Fake();
  Font *GetByDesc(FontDesc);
  template <class... Args> Font *Get(Args&&... args) { return GetByDesc(FontDesc(forward<Args>(args)...)); }
  Font *Change(Font*, int new_size, const Color &new_fg, const Color &new_bg, int new_flag=0);
  int ScaledFontSize(int pointsize);
  void ResetGL();
  void LoadDefaultFonts();
  void LoadConsoleFont(const string &name, const vector<int> &sizes = vector<int>(1, 32));
};

struct FontRef {
  Font *ptr=0;
  FontDesc desc;
  FontRef(Font *F) { SetFont(F); }
  FontRef(const FontDesc &d=FontDesc(), bool load=true) : desc(d) { if (load) Load(); }

  Font *Load();
  operator Font* () const { return ptr; }
  Font* operator->() const { return ptr; }
  void SetFont(Font *F) { if ((ptr = F)) desc = *ptr->desc; }
};

struct DejaVuSansFreetype {
  static void SetDefault();
  static void Load();
};

}; // namespace LFL
#endif // LFL_CORE_APP_FONT_H__
