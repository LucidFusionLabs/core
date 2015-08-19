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

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"

#ifdef WIN32
#include <mlang.h>
IMLangFontLink2 *fontlink=0;
#endif

#ifdef __APPLE__
#import <CoreText/CTFont.h>
#import <CoreText/CTLine.h>
#import <CoreText/CTRun.h>
#import <CoreText/CTStringAttributes.h>
#import <CoreFoundation/CFAttributedString.h>
#import <CoreGraphics/CGBitmapContext.h> 
#ifdef LFL_HARFBUZZ
#include "harfbuzz/hb-coretext.h"
#endif
extern "C" void ConvertColorFromGenericToDeviceRGB(const float *i, float *o);
inline CFStringRef ToCFStr(const string &n) { return CFStringCreateWithCString(0, n.data(), kCFStringEncodingUTF8); }
inline string FromCFStr(CFStringRef in) {
    string ret(CFStringGetMaximumSizeForEncoding(CFStringGetLength(in), kCFStringEncodingUTF8), 0);
    if (!CFStringGetCString(in, (char*)ret.data(), ret.size(), kCFStringEncodingUTF8)) return string();
    ret.resize(strlen(ret.data()));
    return ret;
}
inline CFAttributedStringRef ToCFAStr(CTFontRef ctfont, char16_t glyph_id) {
    const CFStringRef attr_key[] = { kCTFontAttributeName };
    const CFTypeRef attr_val[] = { ctfont };
    CFDictionaryRef attr = CFDictionaryCreate
        (kCFAllocatorDefault, (const void**)&attr_key, (const void**)&attr_val, sizeofarray(attr_key),
         &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFStringRef str = CFStringCreateWithCharacters(kCFAllocatorDefault, reinterpret_cast<unsigned short*>(&glyph_id), 1);
    CFAttributedStringRef astr = CFAttributedStringCreate(kCFAllocatorDefault, str, attr);
    CFRelease(attr);
    CFRelease(str);
    return astr;
}
#endif

#ifdef LFL_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_LCD_FILTER_H
FT_Library ft_library;
#endif

namespace LFL {
#if defined(WIN32)
DEFINE_string(font_engine, "gdi",      "[atlas,freetype,coretext,gdi]");
#elif defined(__APPLE__)
DEFINE_string(font_engine, "coretext", "[atlas,freetype,coretext,gdi]");
#elif defined(LFL_ANDROID)
DEFINE_string(font_engine, "atlas",    "[atlas,freetype,coretext,gdi]");
#else
DEFINE_string(font_engine, "freetype", "[atlas,freetype,coretext,gdi]");
#endif
DEFINE_string(default_font, "", "Default font");
DEFINE_string(default_font_family, "sans-serif", "Default font family");
DEFINE_int(default_font_size, 16, "Default font size");
DEFINE_int(default_font_flag, FontDesc::Mono, "Default font flag");
DEFINE_int(default_missing_glyph, 127, "Default glyph returned for missing requested glyph");
DEFINE_bool(atlas_dump, false, "Dump .png files for every font");
DEFINE_string(atlas_font_sizes, "32", "Load font atlas CSV sizes");
DEFINE_int(glyph_table_start, 32, "Glyph table start value");
DEFINE_int(glyph_table_size, 96, "Use array for glyphs [x=glyph_stable_start, x+glyph_table_size]");
DEFINE_bool(subpixel_fonts, false, "Treat RGB components as subpixels, tripling width");
DEFINE_int(scale_font_height, 0, "Scale font when height != scale_font_height");
DEFINE_int(add_font_size, 0, "Increase all font sizes by add_font_size");

void Glyph::FromArray(const double *in, int l) {
    CHECK_GE(l, 10);
    id        = (int)in[0]; advance      = (int)in[1]; tex.width    = (int)in[2]; tex.height   = (int)in[3]; bearing_y    = (int)in[4];
    bearing_x = (int)in[5]; tex.coord[0] =      in[6]; tex.coord[1] =      in[7]; tex.coord[2] =      in[8]; tex.coord[3] =      in[9];
}

int Glyph::ToArray(double *out, int l) {
    CHECK_GE(l, 10);
    out[0] = id;        out[1] = advance;      out[2] = tex.width;    out[3] = tex.height;   out[4] = bearing_y;
    out[5] = bearing_x; out[6] = tex.coord[0]; out[7] = tex.coord[1]; out[8] = tex.coord[2]; out[9] = tex.coord[3];
    return sizeof(double)*10;
}

int Glyph::LeftBearing(                   const Drawable::Attr *a) const { return (a && a->font) ? RoundXY_or_Y(a->font->scale, bearing_x) : tex.LeftBearing(a); }
int Glyph::Advance    (const LFL::Box *b, const Drawable::Attr *a) const { return (a && a->font) ? RoundXY_or_Y(a->font->scale, advance)   : tex.Advance (b,a); }
int Glyph::Ascender   (const LFL::Box *b, const Drawable::Attr *a) const { return (a && a->font) ? a->font->ascender                       : tex.Ascender(b,a); }

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

void Glyph::Draw(const LFL::Box &b, const Drawable::Attr *a) const {
    if (space) return;
    if (!a || !a->font) return tex.Draw(b, a);
    if (!ready) a->font->engine->LoadGlyphs(a->font, this, 1);
    if (tex.buf) screen->gd->DrawPixels(b, tex);
    else         b.Draw(tex.coord);
}

GlyphCache::GlyphCache(unsigned T, int W, int H) : dim(W, H ? H : W), tex(dim.w, dim.h, Texture::preferred_pf, T), flow(new Flow(&dim)) {}
GlyphCache::~GlyphCache() { delete flow; }

void GlyphCache::Clear() {
    delete flow;
    flow = new Flow(&dim);
    for (auto g : glyph) g->ready = false;
    glyph.clear();
    if      (tex.buf) tex.RenewBuffer();
    else if (tex.ID)  tex.RenewGL();
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

    texcoord[Texture::CoordMinX] =     (float)(out->x    ) / tex.width;
    texcoord[Texture::CoordMinY] = 1 - (float)(out->y + h) / tex.height;
    texcoord[Texture::CoordMaxX] =     (float)(out->x + w) / tex.width;
    texcoord[Texture::CoordMaxY] = 1 - (float)(out->y    ) / tex.height;
    return true;
}

void GlyphCache::Load(const Font *f, const Glyph *g, const unsigned char *buf, int linesize, int spf, const GlyphCache::FilterCB &filter) {
    if (!g->tex.width || !g->tex.height) return;
    point p;
    bool cache_glyph = ShouldCacheGlyph(g->tex);
    if (cache_glyph) {
        CHECK(Add(&p, g->tex.coord, g->tex.width, g->tex.height, f->Height()));
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
            tex.UpdateGL(g->tex.buf, Box(p, g->tex.Dimension()), Texture::Flag::FlipY); 
            g->tex.ClearBuffer();
        }
    }
}

#ifdef __APPLE__
void GlyphCache::Load(const Font *f, const Glyph *g, CGFontRef cgfont, int size) {
    if (!g->internal.coretext.id || !g->tex.width || !g->tex.height) return;
    point p;
    bool cache_glyph = ShouldCacheGlyph(g->tex);
    if (cache_glyph) {
        CHECK(Add(&p, g->tex.coord, g->tex.width, g->tex.height, f->Height()));
        glyph.push_back(g);
    }
    CGGlyph cg = g->internal.coretext.id;
    const Color &fg = f->fg, &bg = f->bg;
    if (tex.buf && cache_glyph) {
        CGPoint point = CGPointMake(p.x - g->bearing_x, tex.height - p.y - g->bearing_y);
        CGContextSetRGBFillColor(cgcontext, bg.r(), bg.g(), bg.b(), bg.a());
        CGContextFillRect(cgcontext, CGRectMake(p.x, tex.height - p.y - g->tex.height, g->tex.width, g->tex.height));
        CGContextSetRGBFillColor(cgcontext, fg.r(), fg.g(), fg.b(), fg.a());
        CGContextSetFont(cgcontext, cgfont);
        CGContextSetFontSize(cgcontext, size);
        CGContextShowGlyphsAtPositions(cgcontext, &cg, &point, 1);
    } else {
        g->tex.pf = tex.pf;
        g->tex.RenewBuffer();
        CGContextRef context = g->tex.CGBitMap();
        CGPoint point = CGPointMake(-g->bearing_x, -RoundLower(g->internal.coretext.origin_y));
        CGContextSetRGBFillColor(context, bg.r(), bg.g(), bg.b(), bg.a());
        CGContextFillRect(context, CGRectMake(0, 0, g->tex.width, g->tex.height));
        CGContextSetRGBFillColor(context, fg.r(), fg.g(), fg.b(), fg.a());
        CGContextSetFont(context, cgfont);
        CGContextSetFontSize(context, size);
        CGContextShowGlyphsAtPositions(context, &cg, &point, 1);
        CGContextRelease(context);
        g->tex.FlipBufferY();
        if (cache_glyph) {
            tex.UpdateGL(g->tex.buf, Box(p, g->tex.Dimension()), Texture::Flag::FlipY); 
            g->tex.ClearBuffer();
        }
        // INFOf("LoadGlyph U+%06x '%c' texID=%d %s point(%f,%f)", g->id, g->id, tex.ID, f->desc->DebugString().c_str(), point.x, point.y);
    }
}
#endif

#ifdef WIN32
void GlyphCache::Load(const Font *f, const Glyph *g, HFONT hfont, int size, HDC dc) {
  if (!g->id || !g->tex.width || !g->tex.height) return;
  point p;
  wchar_t b[] = { g->Id(), 0 };
  bool cache_glyph = ShouldCacheGlyph(g->tex);
  if (cache_glyph) {
    CHECK(Add(&p, g->tex.coord, g->tex.width, g->tex.height, f->Height()));
    glyph.push_back(g);
  }
  const Color &fg = f->fg, &bg = f->bg;
  if (tex.buf && cache_glyph) {
    SetTextColor(hdc, RGB(f->fg.R(), f->fg.G(), f->fg.B()));
    SetBkColor(hdc, RGB(f->bg.R(), f->bg.G(), f->bg.B()));
    SetTextAlign(hdc, TA_TOP);
    HGDIOBJ pf = SelectObject(hdc, hfont);
    CHECK(ExtTextOutW(hdc, p.x - g->bearing_x, p.y - g->bearing_y, ETO_OPAQUE, NULL, b, 1, NULL));
    SelectObject(hdc, pf);
  } else {
    g->tex.pf = tex.pf;
    HBITMAP hbitmap = g->tex.CreateGDIBitMap(dc);
    HGDIOBJ pf = SelectObject(dc, hfont), pbm = SelectObject(dc, hbitmap);
    SetTextColor(dc, RGB(f->fg.R(), f->fg.G(), f->fg.B()));
    SetBkColor(dc, RGB(f->bg.R(), f->bg.G(), f->bg.B()));
    SetTextAlign(dc, TA_TOP);
    CHECK(ExtTextOutW(dc, 0, 0, ETO_OPAQUE, NULL, b, 1, NULL));
    GdiFlush();
    g->tex.FlipBufferY();
    if (cache_glyph) {
      tex.UpdateGL(g->tex.buf, Box(p, g->tex.Dimension()), Texture::Flag::FlipY);
      g->tex.ClearBuffer();
    }
    SelectObject(dc, pf);
    SelectObject(dc, pbm);
    DeleteObject(hbitmap);
    // INFOf("LoadGlyph U+%06x '%c' texID=%d %s point(%f,%f)", g->id, g->id, tex.ID, f->desc->DebugString().c_str(), point.x, point.y);
  }
}
#endif

Glyph *Font::FindOrInsertGlyph(char16_t gind) {
    unsigned ind = gind - glyph->table_start;
    return ind < glyph->table.size() ? &glyph->table[ind] : &glyph->index[gind];
}

Glyph *Font::FindGlyph(char16_t gind) {
    unsigned ind = gind - glyph->table_start;
    if (ind < glyph->table.size()) return &glyph->table[ind];
    auto i = glyph->index.find(gind);
    if (i != glyph->index.end()) return &i->second;
    bool zwnbsp = gind == Unicode::zero_width_non_breaking_space, nbsp = zwnbsp || gind == Unicode::non_breaking_space;
    if (!nbsp && !engine->HaveGlyph(this, gind)) {
        ind = missing_glyph - glyph->table_start;
        CHECK_LT(ind, glyph->table.size());
        return &glyph->table[ind];
    }
    Glyph *g = &glyph->index[gind];
    g->id = nbsp ? ' ' : gind;
    engine->InitGlyphs(this, g, 1);
    if (nbsp) g->id = gind;
    if (zwnbsp) g->advance = g->tex.width = g->tex.height = 0;
    return g;
}

void Font::UpdateMetrics(Glyph *g) {
    if (fix_metrics) {
        if (int fixed_width = FixedWidth()) {
            if (g->advance > (fixed_width)) {
                if ((g->wide = g->advance > (fixed_width * 1.4) && g->id != Unicode::replacement_char))
                    g->advance = fixed_width * 2;
                else g->advance = fixed_width;
            }
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

void Font::Select() {
    screen->gd->EnableLayering();
    glyph->cache->tex.Bind();
    if      (mix_fg)                screen->gd->SetColor(fg);
    else if (has_bg && bg.a() == 1) screen->gd->DisableBlend();
}

template <class X> void Font::Size(const StringPieceT<X> &text, Box *out, int maxwidth, int *lines_out) {
    vector<Box> line_box;
    int lines = Draw(text, Box(0,0,maxwidth,0), &line_box, DrawFlag::Clipped);
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
    if (!(draw_flag & DrawFlag::Clipped)) out.Draw(box.TopLeft());
    return out.line.size();
}

template void Font::Size  <char>    (const StringPiece   &text, Box *out, int maxwidth, int *lines_out);
template void Font::Size  <char16_t>(const String16Piece &text, Box *out, int maxwidth, int *lines_out);
template void Font::Shape <char>    (const StringPiece   &text, const Box &box, DrawableBoxArray *out, int draw_flag, int attr_id);
template void Font::Shape <char16_t>(const String16Piece &text, const Box &box, DrawableBoxArray *out, int draw_flag, int attr_id);
template int  Font::Draw  <char>    (const StringPiece   &text, const Box &box, vector<Box> *lb, int draw_flag);
template int  Font::Draw  <char16_t>(const String16Piece &text, const Box &box, vector<Box> *lb, int draw_flag);

FakeFontEngine::FakeFontEngine() : fake_font(this, fake_font_desc, shared_ptr<FontEngine::Resource>()) {
    fake_font_desc.size = size;
    fake_font.fixed_width = fake_font.max_width = fixed_width;
    fake_font.ascender = ascender;
    fake_font.descender = descender;
    fake_font.glyph = shared_ptr<GlyphMap>(new GlyphMap(shared_ptr<GlyphCache>(new GlyphCache(0, 0))));
    InitGlyphs(&fake_font, &fake_font.glyph->table[0], fake_font.glyph->table.size());
    for (char16_t wide_glyph_id = wide_glyph_begin, e = wide_glyph_end + 1; wide_glyph_id != e; ++wide_glyph_id) {
        Glyph *wg = fake_font.FindGlyph(wide_glyph_id);
        wg->wide = 1;
        wg->advance *= 2;
    }
}

int FakeFontEngine::InitGlyphs(Font *f,       Glyph *g, int n) {
    for (Glyph *e = g + n; g != e; ++g) {
        g->tex.height = g->bearing_y = fake_font.Height();
        g->tex.width  = g->advance   = fake_font.fixed_width;
    } return n;
}

string AtlasFontEngine::DebugString(Font *f) const {
    return StrCat("AtlasFont(", f->desc->DebugString(), "), H=", f->Height(), ", FW=", f->fixed_width);
}

bool AtlasFontEngine::Init(const FontDesc &d) {
    if (Font *f = OpenAtlas(d)) {
        FontDesc de = d;
        de.engine = FontDesc::Engine::Atlas;
        font_map[d.name][d.fg.AsUnsigned()][d.flag][d.size] = f;
        return Fonts::GetByDesc(de);
    }
    return false;
}

Font *AtlasFontEngine::Open(const FontDesc &d) {
    FontMap::iterator fi = font_map.find(d.name);
    if (fi == font_map.end() || !fi->second.size()) return ERRORv(nullptr, "AtlasFont ", d.DebugString(), " not found");

    bool is_fg_white = d.fg == Color::white;
    int max_ci = 2 - is_fg_white;
    for (int ci = 0; ci < max_ci; ++ci) {
        bool last_ci = ci == (max_ci - 1);
        const Color *c = ci ? &Color::white : &d.fg;
        FontColorMap::iterator i;
        FontFlagMap::iterator j;
        FontSizeMap::iterator k;
        FontSizeMap::reverse_iterator l;
        Font *f = 0;

        if ((i = fi->second.find(c->AsUnsigned())) == fi->second.end() || !i->second.size()) continue;
        if ((j = i->second.find(d.flag))           ==  i->second.end() || !j->second.size()) continue;

        if      ((k = j->second.lower_bound(d.size)) != j->second.end())  f = k->second;
        else if ((l = j->second.rbegin())            != j->second.rend()) f = l->second;

        if (!f) continue;
        if (!ci && f->size == d.size) return f;

        Font *primary      = static_cast<Resource*>(f->resource.get())->primary;
        Font *ret          = new Font(this, d, f->resource);
        ret->mix_fg        = ci;
        ret->mono          = primary->mono;
        ret->glyph         = primary->glyph;
        ret->missing_glyph = primary->missing_glyph;
        ret->scale         = (float)ret->size / primary->size;
        ret->ascender      = RoundF(primary->ascender    * ret->scale);
        ret->descender     = RoundF(primary->descender   * ret->scale);
        ret->max_width     = RoundF(primary->max_width   * ret->scale);
        ret->fixed_width   = RoundF(primary->fixed_width * ret->scale);
        return ret;
    }
    return ERRORv(nullptr, "AtlasFont ", d.DebugString(), " clone failed");
}

Font *AtlasFontEngine::OpenAtlas(const FontDesc &d) {
    Texture tex;
    string fn = d.Filename();
    Asset::LoadTexture(StrCat(fn, ".0000.png"), &tex);
    if (!tex.ID) return ERRORv(nullptr, "load ", fn, ".0000.png failed");

    MatrixFile gm;
    unique_ptr<File> gmfile(Asset::OpenFile(MatrixFile::Filename(VersionedFileName(app->assetdir.c_str(), fn.c_str(), "glyphs"), "matrix", 0)));
    if (gmfile && gm.Read(gmfile.get())) return ERRORv(nullptr, "load ", d.name, ".0000.glyphs.matrix failed");

    Resource *resource = new Resource();
    Font *ret = new Font(Singleton<AtlasFontEngine>::Get(), d, shared_ptr<FontEngine::Resource>(resource));
    ret->glyph = shared_ptr<GlyphMap>(new GlyphMap(shared_ptr<GlyphCache>(new GlyphCache(tex.ID, tex.width, tex.height))));
    GlyphCache *cache = ret->glyph->cache.get();
    resource->primary = ret;

    float max_t = 0, max_u = 0;
    MatrixRowIter(gm.F) {
        int glyph_ind = (int)gm.F->row(i)[0];
        Glyph *g = ret->FindOrInsertGlyph(glyph_ind);
        g->FromArray(gm.F->row(i), gm.F->N);
        g->tex.ID = tex.ID;
        if (d.unicode) g->space = isspace(g->id) || g->id == Unicode::non_breaking_space || g->id == Unicode::zero_width_non_breaking_space;
        if (!g->advance) {
            g->advance = g->tex.width;
            g->bearing_y = g->tex.height;
            g->bearing_x = 0;
        }
        ret->UpdateMetrics(g);
        max_t = max(max(max_t, g->tex.coord[0]), g->tex.coord[2]);
        max_u = max(max(max_u, g->tex.coord[1]), g->tex.coord[3]);
    }

    if (ret->fixed_width < 0) ret->fixed_width = 0;
    cache->flow->SetMinimumAscent(ret->Height());
    cache->flow->p.x =  max_t * cache->tex.width;
    cache->flow->p.y = -max_u * cache->tex.height;

    INFO("OpenAtlas ", d.name, ", texID=", tex.ID, ", height=", ret->Height(), ", fixed_width=", ret->fixed_width);
    return ret;
}

void AtlasFontEngine::WriteAtlas(const string &name, Font *f) { WriteAtlas(name, f, &f->glyph->cache->tex); }
void AtlasFontEngine::WriteAtlas(const string &name, Font *f, Texture *t) {
    LocalFile lf(app->assetdir + name + ".0000.png", "w");
    PngWriter::Write(&lf, *t);
    INFO("wrote ", lf.Filename());
    WriteGlyphFile(name, f);
}

void AtlasFontEngine::WriteGlyphFile(const string &name, Font *f) {
    int glyph_count = 0, glyph_out = 0;
    GlyphTableIter(f) if (i->       tex.width && i->       tex.height) glyph_count++;
    GlyphIndexIter(f) if (i->second.tex.width && i->second.tex.height) glyph_count++;

    Matrix *gm = new Matrix(glyph_count, 10);
    GlyphTableIter(f) if (i->       tex.width && i->       tex.height) i->       ToArray(gm->row(glyph_out++), gm->N);
    GlyphIndexIter(f) if (i->second.tex.width && i->second.tex.height) i->second.ToArray(gm->row(glyph_out++), gm->N);
    MatrixFile(gm, "").WriteVersioned(VersionedFileName(app->assetdir.c_str(), name.c_str(), "glyphs"), 0);
}

void AtlasFontEngine::MakeFromPNGFiles(const string &name, const vector<string> &png, const point &atlas_dim, Font **glyphs_out) {
    Font *ret = new Font(Singleton<AtlasFontEngine>::Get(), FontDesc(name), shared_ptr<FontEngine::Resource>());
    ret->glyph = shared_ptr<GlyphMap>(new GlyphMap(shared_ptr<GlyphCache>(new GlyphCache(0, atlas_dim.x, atlas_dim.y))));
    EnsureSize(ret->glyph->table, png.size());

    GlyphCache *cache = ret->glyph->cache.get();
    cache->tex.RenewBuffer();

    for (int i = 0, skipped = 0; i < png.size(); ++i) {
        LocalFile in(png[i], "r");
        if (!in.Opened()) { INFO("Skipped: ", png[i]); skipped++; continue; }
        Glyph *out = &ret->glyph->table[i - skipped];
        out->id = i - skipped;

        if (PngReader::Read(&in, &out->tex)) { skipped++; continue; }
        Max(&ret->ascender, (short)out->tex.height);

        point cache_p;
        CHECK(cache->Add(&cache_p, out->tex.coord, out->tex.width, out->tex.height, ret->Height()));
        SimpleVideoResampler::Blit(out->tex.buf, cache->tex.buf, out->tex.width, out->tex.height,
                                   out->tex.pf,   out->tex.LineSize(),   0,       0,
                                   cache->tex.pf, cache->tex.LineSize(), cache_p.x, cache_p.y);
        out->tex.ClearBuffer();
    }

    WriteAtlas(name, ret, &cache->tex);
    cache->tex.LoadGL();
    cache->tex.ClearBuffer();

    if (glyphs_out) *glyphs_out = ret;
    else delete ret;
}

void AtlasFontEngine::SplitIntoPNGFiles(const string &input_png_fn, const map<int, v4> &glyphs, const string &dir_out) {
    LocalFile in(input_png_fn, "r");
    if (!in.Opened()) return ERROR("open: ", input_png_fn);

    Texture png;
    if (PngReader::Read(&in, &png)) return ERROR("read: ", input_png_fn);

    int count=0;
    for (map<int, v4>::const_iterator i = glyphs.begin(); i != glyphs.end(); ++i) {
        unsigned gx1 = RoundF(i->second.x * png.width), gy1 = RoundF((1 - i->second.y) * png.height);
        unsigned gx2 = RoundF(i->second.z * png.width), gy2 = RoundF((1 - i->second.w) * png.height);
        unsigned gw = gx2 - gx1, gh = gy1 - gy2;
        if (gw <= 0 || gh <= 0) continue;

        Texture glyph;
        glyph.Resize(gw, gh, Texture::preferred_pf, Texture::Flag::CreateBuf);
        SimpleVideoResampler::Blit(png.buf, glyph.buf, glyph.width, glyph.height,
                                   png  .pf, png  .LineSize(), gx1, gy2,
                                   glyph.pf, glyph.LineSize(), 0,   0);

        LocalFile lf(dir_out + StringPrintf("glyph%03d.png", i->first), "w");
        CHECK(lf.Opened());
        PngWriter::Write(&lf, glyph);
    }
}

#ifdef LFL_FREETYPE
FreeTypeFontEngine::Resource::~Resource() {
    if (face) FT_Done_Face(face);
}

string FreeTypeFontEngine::DebugString(Font *f) const {
    return StrCat("TTTFont(", f->desc->DebugString(), "), H=", f->Height(), ", FW=", f->fixed_width);
}

bool FreeTypeFontEngine::Init(const FontDesc &d) {
    if (Contains(resource, d.name)) return true;
    string content = Asset::FileContents(d.name);
    if (Resource *r = OpenBuffer(d, &content)) {
        bool fixed_width = FT_IS_FIXED_WIDTH(r->face);
        resource[d.name] = shared_ptr<Resource>(r);
        return true;
    }
    return false;
}

void FreeTypeFontEngine::Init() {
    static bool init = false;
    if (!init && (init=true)) {
        int error;
        if ((error = FT_Init_FreeType(&ft_library))) ERROR("FT_Init_FreeType: ", error);
    }
}

void FreeTypeFontEngine::SubPixelFilter(const Box &b, unsigned char *buf, int linesize, int pf) {
    Matrix kernel(3, 3, 1/8.0); 
    SimpleVideoResampler::Filter(buf, b.w, b.h, pf, linesize, b.x, b.y, &kernel, ColorChannel::Alpha, SimpleVideoResampler::Flag::ZeroOnly);
}

FreeTypeFontEngine::Resource *FreeTypeFontEngine::OpenFile(const FontDesc &d) {
    Init(); 
    int error;
    FT_FaceRec_ *face = 0;
    if ((error = FT_New_Face(ft_library, d.name.c_str(), 0, &face))) { ERROR("FT_New_Face: ",       error); return 0; }
    if ((error = FT_Select_Charmap(face, FT_ENCODING_UNICODE)))      { ERROR("FT_Select_Charmap: ", error); return 0; }
    FT_Library_SetLcdFilter(ft_library, FLAGS_subpixel_fonts ? FT_LCD_FILTER_LIGHT : FT_LCD_FILTER_NONE);
    return new Resource(face, d.name);
}

FreeTypeFontEngine::Resource *FreeTypeFontEngine::OpenBuffer(const FontDesc &d, string *content) {
    Init();
    int error;
    Resource *r = new Resource(0, d.name, content);
    if ((error = FT_New_Memory_Face(ft_library, (const FT_Byte*)r->content.data(), r->content.size(), 0, &r->face))) { ERROR("FT_New_Memory_Face: ", error); delete r; return 0; }
    if ((error = FT_Select_Charmap(r->face, FT_ENCODING_UNICODE)))                                                   { ERROR("FT_Select_Charmap: ",  error); delete r; return 0; }
    FT_Library_SetLcdFilter(ft_library, FLAGS_subpixel_fonts ? FT_LCD_FILTER_LIGHT : FT_LCD_FILTER_NONE);
    return r;
}

int FreeTypeFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
    int count = 0, error;
    FT_FaceRec_ *face = static_cast<Resource*>(f->resource.get())->face;
    FT_Int32 flags = FT_LOAD_RENDER | (FLAGS_subpixel_fonts ? FT_LOAD_TARGET_LCD : 0) | FT_LOAD_FORCE_AUTOHINT;
    if ((error = FT_Set_Pixel_Sizes(face, 0, f->size))) { ERROR("FT_Set_Pixel_Sizes(", f->size, ") = ", error); return 0; }

    for (Glyph *e = g + n; g != e; ++g, count++) {
        if (!(g->internal.freetype.id = FT_Get_Char_Index(face, g->id))) { /* assign missing glyph? */ continue; }
        if ((error = FT_Load_Glyph(face, g->internal.freetype.id, flags))) { ERROR("FT_Load_Glyph(", g->internal.freetype.id, ") = ", error); continue; }
        if (( FLAGS_subpixel_fonts && face->glyph->bitmap.pixel_mode != FT_PIXEL_MODE_LCD) ||
            (!FLAGS_subpixel_fonts && face->glyph->bitmap.pixel_mode != FT_PIXEL_MODE_GRAY))
        { ERROR("glyph bitmap pixel_mode ", face->glyph->bitmap.pixel_mode); continue; }

        g->tex.width  = face->glyph->bitmap.width / (FLAGS_subpixel_fonts ? 3 : 1);
        g->tex.height = face->glyph->bitmap.rows;
        g->bearing_x  = face->glyph->bitmap_left;
        g->bearing_y  = face->glyph->bitmap_top;
        g->advance    = RoundF(face->glyph->advance.x/64.0);
        g->space      = isspace(g->id) || g->id == Unicode::non_breaking_space || g->id == Unicode::zero_width_non_breaking_space;
        f->UpdateMetrics(g);
    }
    return count;
}

int FreeTypeFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) {
    int count = 0, spf = FLAGS_subpixel_fonts ? Pixel::LCD : Pixel::GRAY8, error;
    bool outline = f->flag & FontDesc::Outline;
    GlyphCache *cache = f->glyph->cache.get();
    FT_FaceRec_ *face = static_cast<Resource*>(f->resource.get())->face;
    FT_Int32 flags = FT_LOAD_RENDER | (FLAGS_subpixel_fonts ? FT_LOAD_TARGET_LCD : 0) | FT_LOAD_FORCE_AUTOHINT;
    if ((error = FT_Set_Pixel_Sizes(face, 0, f->size))) { ERROR("FT_Set_Pixel_Sizes(", f->size, ") = ", error); return false; }

    for (const Glyph *e = g + n; g != e; ++g, count++) {
        g->ready = true;
        if (!g->internal.freetype.id) continue;
        if ((error = FT_Load_Glyph(face, g->internal.freetype.id, flags))) { ERROR("FT_Load_Glyph(", g->internal.freetype.id, ") = ", error); continue; }
        cache->Load(f, g, face->glyph->bitmap.buffer, face->glyph->bitmap.pitch, spf,
                    FLAGS_subpixel_fonts ? subpixel_filter : GlyphCache::FilterCB());
    }
    return count;
}

Font *FreeTypeFontEngine::Open(const FontDesc &d) {
    auto fi = font_map.find(d);
    if (fi != font_map.end()) {
        Font *ret = new Font(*fi->second);
        ret->fg = d.fg;
        ret->bg = d.bg;
        return ret;
    }

    auto ri = resource.find(d.name);
    if (ri == resource.end()) return 0;
    Font *ret = new Font(this, d, ri->second);
    ret->glyph = shared_ptr<GlyphMap>(new GlyphMap());
    int count = InitGlyphs(ret, &ret->glyph->table[0], ret->glyph->table.size());
    ret->fix_metrics = true;
    ret->mix_fg = true;

    bool new_cache = false, pre_load = false;
    GlyphCache *cache =
        new_cache ? new GlyphCache(0, AtlasFontEngine::Dimension(ret->max_width, ret->Height(), count)) : GlyphCache::Get();
    ret->glyph->cache = shared_ptr<GlyphCache>(cache);
    if (new_cache) cache->tex.RenewBuffer();
    if (pre_load) LoadGlyphs(ret, &ret->glyph->table[0], ret->glyph->table.size());
    if (FLAGS_atlas_dump) AtlasFontEngine::WriteAtlas(d.Filename(), ret, &cache->tex);
    if (new_cache) {
        cache->tex.LoadGL();
        cache->tex.ClearBuffer();
    }

    font_map[d] = ret;
    return ret;
}
#endif /* LFL_FREETYPE */

#ifdef __APPLE__
CoreTextFontEngine::Resource::~Resource() {
    if (cgfont) CFRelease(cgfont);
}

string CoreTextFontEngine::DebugString(Font *f) const {
    return StrCat("CoreTextFont(", f->desc->DebugString(), "), H=", f->Height(), " fixed_width=", f->fixed_width, " mono=", f->mono?f->max_width:0, " advance_bounds = ", GetAdvanceBounds(f).DebugString());
}

int CoreTextFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
    CGSize advance;
    Resource *resource = static_cast<Resource*>(f->resource.get());
    CTFontRef ctfont = CTFontCreateWithGraphicsFont(resource->cgfont, f->size, 0, 0);
    if (bool no_substitution = false) {
        vector<UniChar> ascii (n);
        vector<CGGlyph> glyphs(n);
        vector<CGRect>  bounds(n);
        CGRect  *b  = &bounds[0];
        CGGlyph *cg = &glyphs[0];

        for (int i=0; i<n; ++i) ascii[i] = g[i].id;
        CHECK(CTFontGetGlyphsForCharacters(ctfont, &ascii[0], &glyphs[0], glyphs.size()));
        CTFontGetBoundingRectsForGlyphs(ctfont, kCTFontDefaultOrientation, &glyphs[0], &bounds[0], glyphs.size());

        for (Glyph *e = g + n; g != e; ++g, ++b, ++cg) {
            CTFontGetAdvancesForGlyphs(ctfont, kCTFontDefaultOrientation, cg, &advance, 1);
            AssignGlyph(g, *b, advance);
            g->internal.coretext.id = *cg;
            f->UpdateMetrics(g);
        }
    } else for (Glyph *e = g + n; g != e; ++g) {
        CGRect b;
        CTFontRef substituted_ctfont;
        GetSubstitutedFont(f, ctfont, g->id, 0, &substituted_ctfont, &g->internal.coretext.id);
        CGGlyph cg = g->internal.coretext.id;
        CTFontGetBoundingRectsForGlyphs(substituted_ctfont, kCTFontDefaultOrientation, &cg, &b,       1);
        CTFontGetAdvancesForGlyphs     (substituted_ctfont, kCTFontDefaultOrientation, &cg, &advance, 1);
        CFRelease(substituted_ctfont);
        AssignGlyph(g, b, advance);
        f->UpdateMetrics(g);
    }
    CFRelease(ctfont);
    return n;
}

int CoreTextFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) {
    GlyphCache *cache = f->glyph->cache.get();
    Resource *resource = static_cast<Resource*>(f->resource.get());
    CTFontRef ctfont = CTFontCreateWithGraphicsFont(resource->cgfont, f->size, 0, 0);
    for (const Glyph *e = g + n; g != e; ++g) {
        g->ready = true;
        Resource substituted;
        GetSubstitutedFont(f, ctfont, g->id, &substituted.cgfont, 0, 0);
        cache->Load(f, g, substituted.cgfont, f->size);
    }
    CFRelease(ctfont);
    return n;
}

Font *CoreTextFontEngine::Open(const FontDesc &d) {
    bool inserted = 0;
    auto ri = FindOrInsert(resource, d.name, &inserted);
    if (inserted) {
        CFStringRef cfname = ToCFStr(d.name);
        ri->second = shared_ptr<Resource>(new Resource());
        if (!(ri->second->cgfont = CGFontCreateWithFontName(cfname))) { CFRelease(cfname); resource.erase(d.name); return 0; }
#ifdef LFL_HARFBUZZ
        ri->second->hb_face = hb_coretext_face_create(ri->second->cgfont);
#endif
        CFRelease(cfname);
    }

    CTFontRef ctfont = CTFontCreateWithGraphicsFont(ri->second->cgfont, d.size, 0, 0);
    CGFloat ascent = CTFontGetAscent(ctfont), descent = CTFontGetDescent(ctfont), leading = CTFontGetLeading(ctfont);
    CFRelease(ctfont);

    Font *ret = new Font(this, d, ri->second);
    ret->glyph = shared_ptr<GlyphMap>(new GlyphMap());
    ret->ascender = RoundUp(ascent);
    ret->descender = RoundUp(descent) + RoundDown(leading);
    int count = InitGlyphs(ret, &ret->glyph->table[0], ret->glyph->table.size());
    ret->fix_metrics = true;
    ret->has_bg = true;
    // ConvertColorFromGenericToDeviceRGB(d.fg.x, ret->fg.x);
    // ConvertColorFromGenericToDeviceRGB(d.bg.x, ret->bg.x);

    bool new_cache = false, pre_load = false;
    GlyphCache *cache =
        new_cache ? new GlyphCache(0, AtlasFontEngine::Dimension(ret->max_width, ret->Height(), count)) : GlyphCache::Get();
    ret->glyph->cache = shared_ptr<GlyphCache>(cache);
    if (new_cache) {
        cache->tex.RenewBuffer();
        cache->cgcontext = cache->tex.CGBitMap();
    }
    if (pre_load) LoadGlyphs(ret, &ret->glyph->table[0], ret->glyph->table.size());
    if (FLAGS_atlas_dump) AtlasFontEngine::WriteAtlas(d.Filename(), ret, &cache->tex);
    if (new_cache) {
        cache->tex.LoadGL();
        CFRelease(cache->cgcontext);
        cache->cgcontext = 0;
        cache->tex.ClearBuffer();
    }
    return ret;
}

void CoreTextFontEngine::GetSubstitutedFont(Font *f, CTFontRef ctfont, char16_t glyph_id,
                                            CGFontRef *cgout, CTFontRef *ctout, int *id_out) {
    CFAttributedStringRef astr = ToCFAStr(ctfont, glyph_id);
    CTLineRef line = CTLineCreateWithAttributedString(astr);
    CFArrayRef runs = CTLineGetGlyphRuns(line);
    CHECK_EQ(1, CFArrayGetCount(runs));
    CTRunRef run = (CTRunRef)CFArrayGetValueAtIndex(runs, 0);
    int glyphs = CTRunGetGlyphCount(run);
    CHECK_GT(glyphs, 0);
    if (glyphs != 1) ERROR("CoreTextFontEngine ", glyphs, " glyphs for codepoint ", glyph_id);
    if (id_out) {
        CGGlyph cgg;
        CTRunGetGlyphs(run, CFRangeMake(0,1), &cgg);
        *id_out = cgg;
    }
    CFDictionaryRef run_attr = CTRunGetAttributes(run);
    ctfont = (CTFontRef)CFDictionaryGetValue(run_attr, kCTFontAttributeName);
    if (ctout) CFRetain((*ctout = ctfont));
    if (cgout) *cgout = CTFontCopyGraphicsFont(ctfont, NULL);
    CFRelease(line);
    CFRelease(astr);
}

void CoreTextFontEngine::AssignGlyph(Glyph *g, const CGRect &bounds, struct CGSize &advance) {
    float x_extent = bounds.origin.x + bounds.size.width, y_extent = bounds.origin.y + bounds.size.height;
    g->bearing_x  = RoundDown(bounds.origin.x);
    g->bearing_y  = RoundHigher(y_extent);
    g->tex.width  = RoundUp(x_extent - g->bearing_x);
    g->tex.height = g->bearing_y - RoundLower(bounds.origin.y);
    g->advance    = RoundF(advance.width);
    g->space      = isspace(g->id) || g->id == Unicode::non_breaking_space || g->id == Unicode::zero_width_non_breaking_space;
    g->internal.coretext.origin_x = bounds.origin.x;
    g->internal.coretext.origin_y = bounds.origin.y;
    g->internal.coretext.width    = bounds.size.width;
    g->internal.coretext.height   = bounds.size.height;
    g->internal.coretext.advance  = advance.width;
}

v2 CoreTextFontEngine::GetAdvanceBounds(Font *f) {
  v2 ret(INFINITY, -INFINITY);
  for (auto b = f->glyph->table.begin(), e = f->glyph->table.end(), g = b; g != e; ++g) {
    if (g->internal.coretext.advance) Min(&ret.x, static_cast<float>(g->internal.coretext.advance));
    if (1)                            Max(&ret.y, static_cast<float>(g->internal.coretext.advance));
  }
  return ret;
}
#endif /* __APPLE__ */

#ifdef WIN32
GDIFontEngine::Resource::~Resource() {
  if (hfont) DeleteObject(hfont);
}

GDIFontEngine::~GDIFontEngine() { DeleteDC(hdc);}
GDIFontEngine::GDIFontEngine() : hdc(CreateCompatibleDC(NULL)) {
  CHECK(!fontlink)
  CoCreateInstance(CLSID_CMultiLanguage, NULL, CLSCTX_ALL, IID_IMLangFontLink2, (void**)&fontlink);
}

void GDIFontEngine::Shutdown() {
  if (fontlink) fontlink->Release();
}

string GDIFontEngine::DebugString(Font *f) const {
  return StrCat("GDIFont(", f->desc->DebugString(), "), H=", f->Height(), " fixed_width=", f->fixed_width, " mono=", f->mono ? f->max_width : 0);
}

int GDIFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
  GlyphCache *cache = f->glyph->cache.get();
  Resource *resource = static_cast<Resource*>(f->resource.get());
  HGDIOBJ pf = SelectObject(hdc, resource->hfont);
  SIZE s, advance;
  
  for (Glyph *e = g + n; g != e; ++g) {
    wchar_t c = g->Id();
    HFONT substituted_font = 0;
    if (GetSubstitutedFont(f, resource->hfont, g->Id(), hdc, &substituted_font)) SelectObject(hdc, substituted_font);
    CHECK(GetTextExtentPoint32W(hdc, &c, 1, &s));
    if (substituted_font) { SelectObject(hdc, resource->hfont); fontlink->ReleaseFont(substituted_font); }
    AssignGlyph(g, s, advance);
    f->UpdateMetrics(g);
  }

  SelectObject(hdc, pf);
  return n;
}

int GDIFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) {
  GlyphCache *cache = f->glyph->cache.get();
  Resource *resource = static_cast<Resource*>(f->resource.get());
  HGDIOBJ pf = SelectObject(hdc, resource->hfont);
  for (const Glyph *e = g + n; g != e; ++g) {
    g->ready = true;
    HFONT substituted_font = 0;
    GetSubstitutedFont(f, resource->hfont, g->Id(), hdc, &substituted_font);
    cache->Load(f, g, X_or_Y(substituted_font, resource->hfont), f->size, hdc);
    if (substituted_font) fontlink->ReleaseFont(substituted_font);
  }
  SelectObject(hdc, pf);
  return n;
}

Font *GDIFontEngine::Open(const FontDesc &d) {
  bool inserted = 0;
  auto ri = FindOrInsert(resource, StrCat(d.name,",",d.size,",",d.flag), &inserted);
  if (inserted) {
    ri->second = shared_ptr<Resource>(new Resource());
    if (!(ri->second->hfont = CreateFont(d.size, 0, 0, 0, (d.flag & FontDesc::Bold) ? FW_BOLD : FW_NORMAL, d.flag & FontDesc::Italic, 0, 0,
                                         DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, VARIABLE_PITCH,
                                         d.name.c_str()))) { resource.erase(d.name); return 0; }
  }

  TEXTMETRIC tm;
  HBITMAP hbitmap = 0;
  HGDIOBJ pf = SelectObject(hdc, ri->second->hfont), pbm=0;
  GetTextMetrics(hdc, &tm);

  Font *ret = new Font(this, d, ri->second);
  ret->glyph = shared_ptr<GlyphMap>(new GlyphMap());
  ret->ascender = tm.tmAscent + tm.tmDescent;
  ret->descender = 0;
  int count = InitGlyphs(ret, &ret->glyph->table[0], ret->glyph->table.size()); 
  ret->fix_metrics = true;
  ret->has_bg = true;

  bool new_cache = false, pre_load = false;
  GlyphCache *cache =
    new_cache ? new GlyphCache(0, AtlasFontEngine::Dimension(ret->max_width, ret->Height(), count)) : GlyphCache::Get();
  ret->glyph->cache = shared_ptr<GlyphCache>(cache);
  if (new_cache) {
    pbm = SelectObject((cache->hdc = hdc), (hbitmap = cache->tex.CreateGDIBitMap(hdc)));
  }
  if (pre_load) LoadGlyphs(ret, &ret->glyph->table[0], ret->glyph->table.size());
  if (FLAGS_atlas_dump) AtlasFontEngine::WriteAtlas(d.Filename(), ret, &cache->tex);
  if (new_cache) {
    GdiFlush();
    cache->tex.LoadGL();
    cache->tex.ClearBuffer();
    SelectObject(hdc, pbm);
    DeleteObject(hbitmap);
    cache->hdc = 0;
  }
  SelectObject(hdc, pf);
  return ret;
}

bool GDIFontEngine::GetSubstitutedFont(Font *f, HFONT hfont, char16_t glyph_id, HDC hdc, HFONT *hfontout) {
  *hfontout = 0;
  long processed = 0;
  WCHAR c = glyph_id;
  DWORD orig_code_pages = 0, replaced_code_pages = 0;
  if (!fontlink || FAILED(fontlink->GetFontCodePages(hdc, hfont, &orig_code_pages))) return false;
  if (FAILED(fontlink->GetStrCodePages(&c, 1, orig_code_pages, &replaced_code_pages, &processed))) return false;
  if (replaced_code_pages & orig_code_pages) return false;
  bool ret = !FAILED(fontlink->MapFont(hdc, replaced_code_pages, replaced_code_pages ? 0 : c, hfontout));
  return ret;
}

void GDIFontEngine::AssignGlyph(Glyph *g, const ::SIZE &bounds, const ::SIZE &advance) {
  g->bearing_x = 0;
  g->bearing_y = bounds.cy;
  g->tex.width = bounds.cx;
  g->tex.height = bounds.cy;
  g->advance = bounds.cx;
  g->space = isspace(g->id) || g->id == Unicode::non_breaking_space || g->id == Unicode::zero_width_non_breaking_space;
}
#endif /* WIN32 */

FontEngine *Fonts::GetFontEngine(int engine_type) {
    switch (engine_type) {
        case FontDesc::Engine::Atlas:    return Singleton<AtlasFontEngine>   ::Get();
#ifdef LFL_FREETYPE
        case FontDesc::Engine::FreeType: return Singleton<FreeTypeFontEngine>::Get();
#endif
#ifdef __APPLE__
        case FontDesc::Engine::CoreText: return Singleton<CoreTextFontEngine>::Get();
#endif
#ifdef WIN32
        case FontDesc::Engine::GDI:      return Singleton<GDIFontEngine>::Get();
#endif
        case FontDesc::Engine::Default:  return DefaultFontEngine();
    } return DefaultFontEngine();
}

FontEngine *Fonts::DefaultFontEngine() {
    static FontEngine *default_font_engine = 0;
    if (!default_font_engine) {
        if      (FLAGS_font_engine == "atlas")    default_font_engine = Singleton<AtlasFontEngine>   ::Get();
#ifdef LFL_FREETYPE
        else if (FLAGS_font_engine == "freetype") default_font_engine = Singleton<FreeTypeFontEngine>::Get();
#endif
#ifdef __APPLE__
        else if (FLAGS_font_engine == "coretext") default_font_engine = Singleton<CoreTextFontEngine>::Get();
#endif
#ifdef WIN32
        else if (FLAGS_font_engine == "gdi")      default_font_engine = Singleton<GDIFontEngine>     ::Get();
#endif
        else                                      default_font_engine = Singleton<FakeFontEngine>    ::Get();
    }
    return default_font_engine;
}

Font *Fonts::Fake() { return Singleton<FakeFontEngine>::Get()->Open(FontDesc()); }
Font *Fonts::Default() {
    static Font *default_font = 0;
    if (!default_font) default_font = Fonts::Get(FLAGS_default_font, FLAGS_default_font_family, FLAGS_default_font_size);
    return default_font;
}

Font *Fonts::Find(const FontDesc &d) {
    if (d.name.empty()) return 0;
    auto di = desc_map.find(d);
    return (di != desc_map.end()) ? di->second : 0;
}

Font *Fonts::Insert(FontEngine *engine, const FontDesc &d) {
    if (d.name.size()) {
        if (Font *new_font = engine->Open(d)) {
            auto di = desc_map.insert(decltype(desc_map)::value_type(d, new_font)).first;
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
    Fonts *inst = Singleton<Fonts>::Get();
    d.size = ScaledFontSize(d.size);
    if (Font *f = inst->Find(d)) return f;

    if (d.name == FakeFontEngine::Filename()) return Fake();
    FontEngine *engine = GetFontEngine(d.engine);
    Font *f = inst->Insert(engine, d);
    if (f || d.family.empty()) return f;

    auto fi = inst->family_map.find(d.family);
    if (fi == inst->family_map.end()) return 0;

    bool bold = d.flag & FontDesc::Bold, italic = d.flag & FontDesc::Italic;
    if (bold && italic && fi->second.bold_italic.size()) { d.name = *fi->second.bold_italic.begin(); if ((f = inst->FindOrInsert(engine, d))) return f; }
    if (bold &&           fi->second.bold       .size()) { d.name = *fi->second.bold       .begin(); if ((f = inst->FindOrInsert(engine, d))) return f; }
    if (italic &&         fi->second.italic     .size()) { d.name = *fi->second.italic     .begin(); if ((f = inst->FindOrInsert(engine, d))) return f; }
    if (                  fi->second.normal     .size()) { d.name = *fi->second.normal     .begin(); if ((f = inst->FindOrInsert(engine, d))) return f; }

    ERROR("open Font ", d.DebugString(), " failed");
    return 0;
}

Font *Fonts::Change(Font *in, int new_size, const Color &new_fg, const Color &new_bg, int new_flag) {
    static Font *fake_font = Fonts::Fake();
    if (in == fake_font) return fake_font;
    if (!in->desc) return 0;
    FontDesc d = *in->desc;
    d.size = new_size;
    d.fg   = new_fg;
    d.bg   = new_bg;
    d.flag = new_flag;
    return GetByDesc(d);
}

int Fonts::ScaledFontSize(int pointsize) {
    if (FLAGS_scale_font_height) {
        float ratio = (float)screen->height / FLAGS_scale_font_height;
        pointsize = (int)(pointsize * ratio);
    }
    return pointsize + FLAGS_add_font_size;
}

}; // namespace LFL
