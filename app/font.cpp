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

#include "core/app/gl/view.h"
#include "core/app/ipc.h"

namespace LFL {
#if defined(LFL_WINDOWS)
DEFINE_string(font_engine, "gdi",      "[atlas,freetype,gdi]");
#elif defined(LFL_APPLE)
DEFINE_string(font_engine, "coretext", "[atlas,freetype,coretext]");
#elif defined(LFL_ANDROID)
DEFINE_string(font_engine, "android",  "[atlas,freetype,android]");
#elif defined(LFL_LINUX)
DEFINE_string(font_engine, "fc",       "[atlas,freetype,fc]");
#elif defined(LFL_FREETYPE)
DEFINE_string(font_engine, "freetype", "[atlas,freetype]");
#else
DEFINE_string(font_engine, "atlas",    "[atlas,freetype]");
#endif
DEFINE_string(font, "", "Default font");
DEFINE_string(font_family, "sans-serif", "Default font family");
DEFINE_int(font_size, 16, "Default font size");
DEFINE_int(font_flag, FontDesc::Mono, "Default font flag");
DEFINE_int(missing_glyph, 127, "Default glyph returned for missing requested glyph");
DEFINE_bool(atlas_dump, false, "Dump .png files for every font");
DEFINE_string(atlas_font_sizes, "32", "Load font atlas CSV sizes");
DEFINE_int(glyph_table_start, 32, "Glyph table start value");
DEFINE_int(glyph_table_size, 96, "Use array for glyphs [x=glyph_stable_start, x+glyph_table_size)");
DEFINE_bool(subpixel_fonts, false, "Treat RGB components as subpixels, tripling width");
DEFINE_int(scale_font_height, 0, "Scale font when height != scale_font_height");
DEFINE_int(add_font_size, 0, "Increase all font sizes by add_font_size");

void Glyph::FromArray(const double *in, int l) {
  CHECK_GE(l, 10);
  id           = int(in[0]);  advance      = int(in[1]);
  tex.width    = int(in[2]);  tex.height   = int(in[3]);
  bearing_y    = int(in[4]);  bearing_x    = int(in[5]);
  tex.coord[0] =     in[6];   tex.coord[1] =     in[7];
  tex.coord[2] =     in[8];   tex.coord[3] =     in[9];
}

void Glyph::FromMetrics(const GlyphMetrics &m) {
  id=m.id; tex.width=m.width; tex.height=m.height;
  bearing_x=m.bearing_x; bearing_y=m.bearing_y; advance=m.advance;
  wide=m.wide; space=m.space; tex.ID=m.tex_id;
}

int Glyph::ToArray(double *out, int l) {
  CHECK_GE(l, 10);
  out[0] = id;            out[1] = advance;
  out[2] = tex.width;     out[3] = tex.height;
  out[4] = bearing_y;     out[5] = bearing_x;
  out[6] = tex.coord[0];  out[7] = tex.coord[1];
  out[8] = tex.coord[2];  out[9] = tex.coord[3];
  return sizeof(double)*10;
}

int Glyph::Baseline(const LFL::Box *b, const Drawable::Attr *a) const {
  if (!a || !a->font) return tex.Baseline(b,a);
  return b->h - RoundXY_or_Y(a->font->scale, bearing_y);
}

int Glyph::Ascender(const LFL::Box *b, const Drawable::Attr *a) const {
  if (!a || !a->font) return tex.Ascender(b,a);
  return a->font->ascender;
}

int Glyph::Advance(const LFL::Box *b, const Drawable::Attr *a) const {
  if (!a || !a->font) return tex.Advance (b,a); 
  int cw = (!a->font->fixed_width && a->font->mono) ? a->font->max_width : 0;
  return cw ? cw : RoundXY_or_Y(a->font->scale, advance);
}

int Glyph::LeftBearing(const Drawable::Attr *a) const {
  if (!a || !a->font) return tex.LeftBearing(a);
  int cw = (!a->font->fixed_width && a->font->mono) ? a->font->max_width : 0;
  return RoundXY_or_Y(a->font->scale, bearing_x) + (cw ? RoundF((cw - RoundXY_or_Y(a->font->scale, advance))/2.0) : 0);
}

int Glyph::Layout(LFL::Box *out, const Drawable::Attr *attr) const {
  if (!attr || !attr->font) return tex.Layout(out, attr);
  float scale = attr->font->scale;
  int center_width = (!attr->font->fixed_width && attr->font->mono) ? attr->font->max_width : 0;
  int gw = RoundXY_or_Y(scale, tex.width), gh = RoundXY_or_Y(scale, tex.height);
  int bx = RoundXY_or_Y(scale, bearing_x), by = RoundXY_or_Y(scale, bearing_y);
  int ax = RoundXY_or_Y(scale, advance);
  *out = LFL::Box(bx + (center_width ? RoundF((center_width - ax) / 2.0) : 0), by-gh, gw, gh);
  return X_or_Y(center_width, ax);
}

void Glyph::Draw(GraphicsContext *gc, const LFL::Box &b) const {
  if (space) return;
  if (!gc->attr || !gc->attr->font) return tex.Draw(gc, b);
  if (!ready) gc->attr->font->engine->LoadGlyphs(gc->attr->font, this, 1);
  if (tex.buf) gc->gd->DrawPixels(b, tex);
  else gc->DrawTexturedBox(b, tex.coord);
}

void FillColor::Draw(GraphicsContext *gc, const LFL::Box &b) const {
  if (!ready) {
    Texture fill(gc->gd, 2, 2);
    fill.RenewBuffer();
    SimpleVideoResampler::Fill(fill.buf, fill.width, fill.height, fill.pf, fill.LineSize(), 0, 0,
                               Color(internal.fillcolor.color));

    tex.width = fill.width;
    tex.height = fill.height;
    if (auto cache = app->fonts->rgba_glyph_cache.get()) {
      cache->Load(0, this, fill.buf, fill.LineSize(), fill.pf);
      tex.coord[Texture::minx_coord_ind] += 1.0/cache->tex.width;
      tex.coord[Texture::miny_coord_ind] += 1.0/cache->tex.height;
      tex.coord[Texture::maxx_coord_ind] -= 1.0/cache->tex.width;
      tex.coord[Texture::maxy_coord_ind] -= 1.0/cache->tex.height;
      ready = true;
    }
  }
  gc->DrawTexturedBox(b, tex.coord);
}

GlyphCache::~GlyphCache() {}
GlyphCache::GlyphCache(unsigned T, int W, int H) :
  dim(W, H ? H : W), tex(nullptr, dim.w, dim.h, Texture::preferred_pf, T), flow(make_unique<Flow>(&dim)) {}

void GlyphCache::Clear(bool reopen) {
  INFO("GlyphCache::Clear reopen=", reopen);    
  flow = make_unique<Flow>(&dim);
  for (auto g : glyph) g->ready = false;
  glyph.clear();
  if (reopen) {
    if (tex.buf) tex.RenewBuffer();
    else         tex.RenewGL();
  } else {
    if (tex.buf) tex.RenewBuffer();
    else         tex.ClearGL();
  }
}

bool GlyphCache::Add(point *out, float *texcoord, int w, int h, int max_height) {
  Box box;
  flow->SetMinimumAscent(max_height);
  flow->AppendBox(w, h, Border(2,2,2,2), &box);

  *out = point(box.x, box.y + flow->container->top());
  if (out->x < 0 || out->x + w > tex.width ||
      out->y < 0 || out->y + h > tex.height) {
    Clear();
    return Add(out, texcoord, w, h, max_height);
  }

  texcoord[Texture::minx_coord_ind] =     float(out->x    ) / tex.width;
  texcoord[Texture::miny_coord_ind] = 1 - float(out->y + h) / tex.height;
  texcoord[Texture::maxx_coord_ind] =     float(out->x + w) / tex.width;
  texcoord[Texture::maxy_coord_ind] = 1 - float(out->y    ) / tex.height;
  return true;
}

void GlyphCache::Load(const Font *f, const Glyph *g, const unsigned char *buf, int linesize, int spf, const GlyphCache::FilterCB &filter) {
  if (!g->tex.width || !g->tex.height) return;
  point p;
  bool cache_glyph = ShouldCacheGlyph(g->tex);
  if (cache_glyph) {
    CHECK(Add(&p, g->tex.coord, g->tex.width, g->tex.height, f ? f->Height() : g->tex.height));
    glyph.push_back(g);
  }
  if (tex.buf && cache_glyph) {
    tex.UpdateBuffer(buf, Box(p, g->tex.width, g->tex.height), spf, linesize);
    if (filter) filter(Box(p, g->tex.Dimension()), tex.buf, tex.LineSize(), tex.pf);
  } else {
    g->tex.pf = tex.pf;
    g->tex.LoadBuffer(buf, g->tex.Dimension(), spf, linesize, Texture::Flag::FlipY);
    if (filter) filter(Box(g->tex.Dimension()), g->tex.buf, g->tex.LineSize(), g->tex.pf);
    // PngWriter::Write(StringPrintf("glyph%06x.png", g->id), g->tex);
    if (cache_glyph) {
      tex.UpdateGL(g->tex.buf, Box(p, g->tex.Dimension()), 0, Texture::Flag::FlipY); 
      g->tex.ClearBuffer();
    }
  }
}

FontDesc::FontDesc(const IPC::FontDescription &d) :
  FontDesc(d.name() ? d.name()->data() : "", d.family() ? d.family()->data() : "", d.size(),
           d.fg() ? Color(d.fg()->r(), d.fg()->g(), d.fg()->b(), d.fg()->a()) : Color::white,
           d.bg() ? Color(d.bg()->r(), d.bg()->g(), d.bg()->b(), d.bg()->a()) : Color::clear, d.flag(), d.unicode(), d.engine()) {}

Glyph *Font::FindOrInsertGlyph(char16_t ind) {
  if (ind >= glyph->table_start) {
    int table_ind = ind - glyph->table_start;
    if (table_ind < glyph->table.size()) return &glyph->table[table_ind];
  }
  return &glyph->index[ind];
}

Glyph *Font::FindGlyph(char16_t ind) {
  if (ind >= glyph->table_start) {
    int table_ind = ind - glyph->table_start;
    if (table_ind < glyph->table.size()) return &glyph->table[table_ind];
  }
  auto i = glyph->index.find(ind);
  if (i != glyph->index.end()) return &i->second;
  bool zwnbsp = ind == Unicode::zero_width_non_breaking_space, nbsp = zwnbsp || ind == Unicode::non_breaking_space;
  if (!nbsp && !engine->HaveGlyph(this, ind)) {
    int table_ind = missing_glyph - glyph->table_start;
    CHECK_LT(table_ind, glyph->table.size());
    return &glyph->table[table_ind];
  }
  Glyph *g = &glyph->index[ind];
  g->id = nbsp ? ' ' : ind;
  engine->InitGlyphs(this, g, 1);
  if (nbsp) g->id = ind;
  if (zwnbsp) g->advance = g->tex.width = g->tex.height = 0;
  return g;
}

void Font::UpdateMetrics(Glyph *g) {
  if (fix_metrics) {
    if (int fixed_width = FixedWidth()) {
      if (g->advance > fixed_width) {
        if ((g->wide = g->advance > (fixed_width * 1.4) && g->id != Unicode::replacement_char))
          g->advance = fixed_width * 2;
        else g->advance = fixed_width;
      } else if (g->advance < fixed_width) g->advance = fixed_width;
    }
    return;
  }
  int descent = g->tex.height - g->bearing_y;
  if (g->advance && fixed_width == -1)         fixed_width = g->advance;
  if (g->advance && fixed_width != g->advance) fixed_width = 0;
  if (g->advance   > max_width) max_width = g->advance;
  if (g->bearing_y > ascender)  ascender  = g->bearing_y;
  if (descent      > descender) descender = descent;
}

void Font::Select(GraphicsDevice *gd) {
  gd->EnableTexture();
  glyph->cache->tex.Bind();
  if      (mix_fg)                { gd->EnableBlend(); gd->SetColor(fg); }
  else if (has_bg && bg.a() == 1) { gd->DisableBlend(); }
}

void Font::DrawGlyph(GraphicsDevice *d, int g, const Box &w) {
  Drawable::Attr a(this);
  GraphicsContext gc(d, &a);
  return DrawGlyphWithAttr(&gc, g, w);
}

void Font::DrawGlyphWithAttr(GraphicsContext *gc, int gid, const Box &w) {
  Select(gc->gd);
  Glyph *g = FindGlyph(gid);
  g->Draw(gc, w);
}

template <class X> void Font::Size(const StringPieceT<X> &text, Box *out, int maxwidth, int flag, int *lines_out) {
  vector<Box> line_box;
  int lines = Draw(text, Box(0,0,maxwidth,0), &line_box, flag | DrawFlag::Clipped);
  if (lines_out) *lines_out = lines;
  *out = Box(0, 0, 0, lines * Height());
  for (int i=0; i<line_box.size(); i++) out->w = max(out->w, line_box[i].w);
}

template <class X> void Font::Shape(const StringPieceT<X> &text, const Box &box, DrawableBoxArray *out, int draw_flag, int attr_id) {
  Flow flow(&box, this, out);
  if (!(draw_flag & DrawFlag::DontAssignFlowP)) flow.p = box.Position();
  flow.layout.wrap_lines     = !(draw_flag & DrawFlag::NoWrap) && box.w;
  flow.layout.word_break     = !(draw_flag & DrawFlag::GlyphBreak);
  flow.layout.align_center   =  (draw_flag & DrawFlag::AlignCenter);
  flow.layout.align_right    =  (draw_flag & DrawFlag::AlignRight);
  flow.layout.pad_wide_chars = FixedWidth();
  if (!attr_id) {
    flow.cur_attr.underline  =  (draw_flag & DrawFlag::Underline);
    flow.cur_attr.overline   =  (draw_flag & DrawFlag::Overline);
    flow.cur_attr.midline    =  (draw_flag & DrawFlag::Midline);
    flow.cur_attr.blink      =  (draw_flag & DrawFlag::Blink);
  }
  if      (draw_flag & DrawFlag::Uppercase)  flow.layout.char_tf = ::toupper;
  else if (draw_flag & DrawFlag::Lowercase)  flow.layout.char_tf = ::tolower;
  if      (draw_flag & DrawFlag::Capitalize) flow.layout.word_start_char_tf = ::toupper;
  flow.AppendText(text, attr_id);
  if (!(draw_flag & DrawFlag::DontCompleteFlow)) flow.Complete();
}

template <class X> int Font::Draw(const StringPieceT<X> &text, const Box &box, vector<Box> *lb, int draw_flag) {
  DrawableBoxArray out;
  Shape(text, box, &out, draw_flag | DrawFlag::DontAssignFlowP);
  if (lb) *lb = out.line;
  if (!(draw_flag & DrawFlag::Clipped)) out.Draw(app->focused->gd, box.TopLeft());
  return max(size_t(1), out.line.size());
}

template void Font::Size  <char>    (const StringPiece   &text, Box *out, int maxwidth, int flag, int *lines_out);
template void Font::Size  <char16_t>(const String16Piece &text, Box *out, int maxwidth, int flag, int *lines_out);
template void Font::Shape <char>    (const StringPiece   &text, const Box &box, DrawableBoxArray *out, int draw_flag, int attr_id);
template void Font::Shape <char16_t>(const String16Piece &text, const Box &box, DrawableBoxArray *out, int draw_flag, int attr_id);
template int  Font::Draw  <char>    (const StringPiece   &text, const Box &box, vector<Box> *lb, int draw_flag);
template int  Font::Draw  <char16_t>(const String16Piece &text, const Box &box, vector<Box> *lb, int draw_flag);

FakeFontEngine::FakeFontEngine() : fake_font_desc(Filename(), "", FLAGS_font_size), 
  fake_font(this, fake_font_desc, shared_ptr<FontEngine::Resource>()) {
  fake_font.desc = &fake_font_desc;
  fake_font.fixed_width = fake_font.max_width = fixed_width;
  fake_font.ascender = ascender;
  fake_font.descender = descender;
  fake_font.glyph = make_shared<GlyphMap>(make_shared<GlyphCache>(0, 0));
  InitGlyphs(&fake_font, &fake_font.glyph->table[0], fake_font.glyph->table.size());
  for (char16_t wide_glyph_id = wide_glyph_begin, e = wide_glyph_end + 1; wide_glyph_id != e; ++wide_glyph_id) {
    Glyph *wg = fake_font.FindGlyph(wide_glyph_id);
    wg->wide = 1;
    wg->advance *= 2;
  }
}

int FakeFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
  for (Glyph *e = g + n; g != e; ++g) {
    g->tex.height = g->bearing_y = fake_font.Height();
    g->tex.width  = g->advance   = fake_font.fixed_width;
  } return n;
}

Fonts::~Fonts() {
  // if (fc_engine.ptr) fc_engine.ptr->Shutdown();
}

void Fonts::SelectFillColor(GraphicsDevice *gd) {
  gd->Color4f(1, 1, 1, 1);
  gd->EnableTexture();
  if (auto cache = rgba_glyph_cache.get()) cache->tex.Bind();
  gd->DisableBlend();
}

FillColor *Fonts::GetFillColor(const Color &c) {
  bool inserted = false;
  auto it = LFL::FindOrInsert(color_map, c.AsUnsigned(), &inserted);
  if (inserted) it->second.internal.fillcolor.color = c.AsUnsigned();
  return &it->second;
}

int Fonts::InitFontWidth() {
#if defined(LFL_WINDOWS)
  return 8;
#elif defined(LFL_APPLE)
  return 9;
#else
  return 10;
#endif
}

int Fonts::InitFontHeight() {
#if defined(LFL_WINDOWS)
  return 17;
#elif defined(LFL_APPLE)
  return 20;
#else
  return 18;
#endif
}

FontEngine *Fonts::GetFontEngine(int engine_type) {
  switch (engine_type) {
    case FontDesc::Engine::Atlas:    return atlas_engine.get();
    case FontDesc::Engine::FreeType: return freetype_engine.get();
#if defined(LFL_WINDOWS)
    case FontDesc::Engine::GDI:      return gdi_engine.get();
#elif defined(LFL_APPLE)
    case FontDesc::Engine::CoreText: return coretext_engine.get();
#elif defined(LFL_ANDROID)
    case FontDesc::Engine::Android:  return android_engine.get();
#elif defined(LFL_LINUX)
    case FontDesc::Engine::FC:       return fc_engine.get();
#endif
    case FontDesc::Engine::Default:  return DefaultFontEngine();
  } return DefaultFontEngine();
}

FontEngine *Fonts::DefaultFontEngine() {
  if (!default_font_engine) {
    if      (FLAGS_font_engine == "atlas")      default_font_engine = atlas_engine.get();
    else if (FLAGS_font_engine == "freetype")   default_font_engine = freetype_engine.get();
#if defined(LFL_WINDOWS)
    else if (FLAGS_font_engine == "gdi")        default_font_engine = gdi_engine.get();
#elif defined(LFL_APPLE)
    else if (FLAGS_font_engine == "coretext")   default_font_engine = coretext_engine.get();
#elif defined(LFL_ANDROID)
    else if (FLAGS_font_engine == "android")    default_font_engine = android_engine.get();
#elif defined(LFL_LINUX)
    else if (FLAGS_font_engine == "fc")         default_font_engine = fc_engine.get();
#endif                                          
    else                                        default_font_engine = fake_engine.get();
  }
  return default_font_engine;
}

Font *Fonts::Fake() { return &fake_engine.get()->fake_font; }

Font *Fonts::Find(const FontDesc &d) {
  if (d.name.empty()) return 0;
  auto di = desc_map.find(d);
  return (di != desc_map.end()) ? di->second.get() : 0;
}

Font *Fonts::Insert(FontEngine *engine, const FontDesc &d) {
  if (d.name.size()) {
    if (Font *new_font = engine->Open(d).release()) {
      auto di = desc_map.insert(decltype(desc_map)::value_type(d, unique_ptr<Font>(new_font))).first;
      di->second->desc = &di->first;
      return new_font;
    }
  }
  ERROR(engine->Name(), " Open: ", d.name, " ", d.size, " ", d.fg.DebugString(), " ", d.flag, " failed");
  return NULL;
}

Font *Fonts::FindOrInsert(FontEngine *engine, const FontDesc &d) {
  if (Font *f = Find(d)) return f;
  return Insert(engine, d);
}

Font *Fonts::GetByDesc(FontDesc d) {
  d.size = ScaledFontSize(d.size);
  if (Font *f = Find(d)) return f;

  if (d.name == FakeFontEngine::Filename()) return Fake();
  FontEngine *engine = GetFontEngine(d.engine);
  Font *f = Insert(engine, d);
  if (f || d.family.empty()) return f;

  auto fi = family_map.find(d.family);
  if (fi == family_map.end()) return 0;

  bool bold = d.flag & FontDesc::Bold, italic = d.flag & FontDesc::Italic;
  if (bold && italic && fi->second.bold_italic.size()) { d.name = *fi->second.bold_italic.begin(); if ((f = FindOrInsert(engine, d))) return f; }
  if (bold &&           fi->second.bold       .size()) { d.name = *fi->second.bold       .begin(); if ((f = FindOrInsert(engine, d))) return f; }
  if (italic &&         fi->second.italic     .size()) { d.name = *fi->second.italic     .begin(); if ((f = FindOrInsert(engine, d))) return f; }
  if (                  fi->second.normal     .size()) { d.name = *fi->second.normal     .begin(); if ((f = FindOrInsert(engine, d))) return f; }

  ERROR("open Font ", d.DebugString(), " failed");
  return 0;
}

Font *Fonts::Change(Font *in, int new_size, const Color &new_fg, const Color &new_bg, int new_flag) {
  static Font *fake_font = Fonts::Fake();
  if (in == fake_font) return fake_font;
  if (!in->desc) return 0;
  FontDesc d = *in->desc;
  d.size = new_size ? new_size : in->size;
  d.fg   = new_fg;
  d.bg   = new_bg;
  d.flag = new_flag;
  return GetByDesc(d);
}

int Fonts::ScaledFontSize(int pointsize) {
  if (FLAGS_scale_font_height) {
    float ratio = float(app->focused->height) / FLAGS_scale_font_height;
    pointsize = RoundF(pointsize * ratio);
  }
  return pointsize + FLAGS_add_font_size;
}

void Fonts::ResetGL(int flag) {
  bool reload = flag & ResetGLFlag::Reload, forget = (flag & ResetGLFlag::Delete) == 0;
  unordered_set<GlyphMap*> maps;
  unordered_set<GlyphCache*> caches;
  for (auto &i : desc_map) {
    auto f = i.second.get();
    if (f->engine == atlas_engine.get() && f == dynamic_cast<AtlasFontEngine::Resource*>(f->resource.get())->primary) {
      if (!forget) f->glyph->cache->tex.Clear();
      f->glyph->cache->tex = Texture();
      if (!reload) continue;
      Asset::LoadTexture(StrCat(f->desc->Filename(), ".0000.png"), &f->glyph->cache->tex);
      if (!f->glyph->cache->tex.ID) ERROR("Reset font failed");
    } else maps.insert(f->glyph.get());
  }
  for (auto m : maps) if (auto c = m->cache.get()) caches.insert(c);
  for (auto c : caches) {
    if (forget) c->tex.owner = false;
    c->Clear(reload);
  }
}

void Fonts::LoadDefaultFonts() {
  FontEngine *font_engine = DefaultFontEngine();
  vector<string> atlas_font_size;
  Split(FLAGS_atlas_font_sizes, iscomma, &atlas_font_size);
  for (int i=0; i<atlas_font_size.size(); i++) {
    int size = atoi(atlas_font_size[i].c_str());
    font_engine->Init(FontDesc(FLAGS_font, FLAGS_font_family, size,
                               Color::white, Color::clear, FLAGS_font_flag));
  }

  FontEngine *atlas_engine = app->fonts->atlas_engine.get();
  atlas_engine->Init(FontDesc("MenuAtlas", "", 0, Color::white, Color::clear, 0, false));

  if (FLAGS_console && FLAGS_font_engine != "atlas" && FLAGS_font_engine != "freetype")
    LoadConsoleFont(FLAGS_console_font.empty() ? "VeraMoBd.ttf" : FLAGS_console_font);
}

void Fonts::LoadConsoleFont(const string &name, const vector<int> &sizes) {
  auto atlas_font_engine = atlas_engine.get();
  FLAGS_console_font = StrCat("atlas://", name);
  FLAGS_atlas_font_sizes.clear();
  for (auto size : sizes) {
    StrAppend(&FLAGS_atlas_font_sizes, FLAGS_atlas_font_sizes.empty()?"":",", size);
    atlas_font_engine->Init(FontDesc(name, "", size, Color::white, Color::clear, FLAGS_console_font_flag));
  }
}

Font *FontRef::Load() { return (ptr = app->fonts->Get(desc)); }

void DejaVuSansFreetype::SetDefault() {
  FLAGS_font_engine = "freetype";
  FLAGS_font = "DejaVuSans.ttf";
  FLAGS_font_family = "sans-serif";
  FLAGS_font_flag = 0;
  FLAGS_atlas_font_sizes = "32";
}

void DejaVuSansFreetype::Load() { 
  vector<string> atlas_font_size;
  Split(FLAGS_atlas_font_sizes, iscomma, &atlas_font_size);
  FontEngine *freetype = app->fonts->freetype_engine.get();
  for (int i=0; i<atlas_font_size.size(); i++) {
    int size = ::atoi(atlas_font_size[i].c_str());
    freetype->Init(FontDesc("DejaVuSans.ttf",                "sans-serif", size, Color::white, Color::clear, 0));
    freetype->Init(FontDesc("DejaVuSans-Bold.ttf",           "sans-serif", size, Color::white, Color::clear, FontDesc::Bold));
    freetype->Init(FontDesc("DejaVuSans-Oblique.ttf",        "sans-serif", size, Color::white, Color::clear, FontDesc::Italic));
    freetype->Init(FontDesc("DejaVuSans-BoldOblique.ttf",    "sans-serif", size, Color::white, Color::clear, FontDesc::Italic | FontDesc::Bold));
    freetype->Init(FontDesc("DejaVuSansMono.ttf",            "monospace",  size, Color::white, Color::clear, 0));
    freetype->Init(FontDesc("DejaVuSansMono-Bold.ttf",       "monospace",  size, Color::white, Color::clear, FontDesc::Bold));
    freetype->Init(FontDesc("DejaVuSerif.ttf",               "serif",      size, Color::white, Color::clear, 0));
    freetype->Init(FontDesc("DejaVuSerif-Bold.ttf",          "serif",      size, Color::white, Color::clear, FontDesc::Bold));
    freetype->Init(FontDesc("DejaVuSansMono-Oblique.ttf",    "cursive",    size, Color::white, Color::clear, 0));
    freetype->Init(FontDesc("DejaVuSerifCondensed-Bold.ttf", "fantasy",    size, Color::white, Color::clear, 0));
  }
}

}; // namespace LFL
