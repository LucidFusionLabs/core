/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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

#include <mlang.h>
IMLangFontLink2 *fontlink=0;

namespace LFL {
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
      tex.UpdateGL(g->tex.buf, Box(p, g->tex.Dimension()), 0, Texture::Flag::FlipY);
      g->tex.ClearBuffer();
    }
    SelectObject(dc, pf);
    SelectObject(dc, pbm);
    DeleteObject(hbitmap);
    // INFOf("LoadGlyph U+%06x '%c' texID=%d %s point(%f,%f)", g->id, g->id, tex.ID, f->desc->DebugString().c_str(), point.x, point.y);
  }
}

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

void GDIFontEngine::SetDefault() {
  FLAGS_font_engine = "gdi";
  FLAGS_font = "Consolas";
  FLAGS_font_size = 17;
  // FLAGS_font_flag = FontDesc::Bold | FontDesc::Mono;
}

string GDIFontEngine::DebugString(Font *f) const {
  return StrCat("GDIFont(", f->desc->DebugString(), "), H=", f->Height(), " fixed_width=", f->fixed_width, " mono=", f->mono ? f->max_width : 0);
}

int GDIFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
  GlyphCache *cache = f->glyph->cache.get();
  Resource *resource = dynamic_cast<Resource*>(f->resource.get());
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
  Resource *resource = dynamic_cast<Resource*>(f->resource.get());
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

unique_ptr<Font> GDIFontEngine::Open(const FontDesc &d) {
  bool inserted = 0;
  auto ri = FindOrInsert(resource, StrCat(d.name,",",d.size,",",d.flag), &inserted);
  if (inserted) {
    ri->second = make_shared<Resource>();
    if (!(ri->second->hfont = CreateFont(d.size, 0, 0, 0, (d.flag & FontDesc::Bold) ? FW_BOLD : FW_NORMAL, d.flag & FontDesc::Italic, 0, 0,
                                         DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, VARIABLE_PITCH,
                                         d.name.c_str()))) { resource.erase(d.name); return nullptr; }
  }

  TEXTMETRIC tm;
  HBITMAP hbitmap = 0;
  HGDIOBJ pf = SelectObject(hdc, ri->second->hfont), pbm=0;
  GetTextMetrics(hdc, &tm);

  unique_ptr<Font> ret = make_unique<Font>(this, d, ri->second);
  ret->glyph = make_shared<GlyphMap>();
  ret->ascender = tm.tmAscent + tm.tmDescent;
  ret->descender = 0;
  int count = InitGlyphs(ret.get(), &ret->glyph->table[0], ret->glyph->table.size()); 
  ret->fix_metrics = true;
  ret->has_bg = true;

  bool new_cache = false, pre_load = false;
  ret->glyph->cache =
    (!new_cache ? app->fonts->GetGlyphCache() :
     make_shared<GlyphCache>(0, AtlasFontEngine::Dimension(ret->max_width, ret->Height(), count)));
  GlyphCache *cache = ret->glyph->cache.get();

  if (new_cache) {
    pbm = SelectObject((cache->hdc = hdc), (hbitmap = cache->tex.CreateGDIBitMap(hdc)));
  }
  if (pre_load) LoadGlyphs(ret.get(), &ret->glyph->table[0], ret->glyph->table.size());
  if (FLAGS_atlas_dump) AtlasFontEngine::WriteAtlas(d.Filename(), ret.get(), &cache->tex);
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

}; // namespace LFL
