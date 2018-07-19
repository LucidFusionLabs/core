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

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_LCD_FILTER_H

namespace LFL {
static FT_Library ft_library;

FreeTypeFontEngine::Resource::~Resource() {
  if (face) FT_Done_Face(face);
}

string FreeTypeFontEngine::DebugString(Font *f) const {
  return StrCat("TTTFont(", f->desc->DebugString(), "), H=", f->Height(), ", FW=", f->fixed_width);
}

void FreeTypeFontEngine::SetDefault() {
  FLAGS_font_engine = "freetype";
  FLAGS_font = "VeraMoBd.ttf"; // "DejaVuSansMono-Bold.ttf";
  FLAGS_missing_glyph = 42;
}

bool FreeTypeFontEngine::Init(const FontDesc &d) {
  if (Contains(resource, d.name)) return true;
  string content = d.name.data()[0] == '/' ? LocalFile(d.name, "r").Contents() : parent->loader->FileContents(d.name);
  if (auto r = OpenBuffer(d, &content)) {
    resource[d.name] = shared_ptr<Resource>(r.release());
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

unique_ptr<FreeTypeFontEngine::Resource> FreeTypeFontEngine::OpenFile(const FontDesc &d) {
  Init(); 
  int error;
  FT_FaceRec_ *face = 0;
  if ((error = FT_New_Face(ft_library, d.name.c_str(), 0, &face))) { ERROR("FT_New_Face: ",       error); return 0; }
  if ((error = FT_Select_Charmap(face, FT_ENCODING_UNICODE)))      { ERROR("FT_Select_Charmap: ", error); return 0; }
  FT_Library_SetLcdFilter(ft_library, FLAGS_subpixel_fonts ? FT_LCD_FILTER_LIGHT : FT_LCD_FILTER_NONE);
  return make_unique<Resource>(face, d.name);
}

unique_ptr<FreeTypeFontEngine::Resource> FreeTypeFontEngine::OpenBuffer(const FontDesc &d, string *content) {
  Init();
  int error;
  auto r = make_unique<Resource>(nullptr, d.name, content);
  if ((error = FT_New_Memory_Face(ft_library, (const FT_Byte*)r->content.data(), r->content.size(), 0, &r->face))) return ERRORv(nullptr, "FT_New_Memory_Face: ", error);
  if ((error = FT_Select_Charmap(r->face, FT_ENCODING_UNICODE))) return ERRORv(nullptr, "FT_Select_Charmap: ", error);
  FT_Library_SetLcdFilter(ft_library, FLAGS_subpixel_fonts ? FT_LCD_FILTER_LIGHT : FT_LCD_FILTER_NONE);
  return r;
}

int FreeTypeFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
  int count = 0, error;
  FT_FaceRec_ *face = dynamic_cast<Resource*>(f->resource.get())->face;
  FT_Int32 flags = FT_LOAD_RENDER | (FLAGS_subpixel_fonts ? FT_LOAD_TARGET_LCD : 0); // | FT_LOAD_FORCE_AUTOHINT;
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
  FT_FaceRec_ *face = dynamic_cast<Resource*>(f->resource.get())->face;
  FT_Int32 flags = FT_LOAD_RENDER | (FLAGS_subpixel_fonts ? FT_LOAD_TARGET_LCD : 0); // | FT_LOAD_FORCE_AUTOHINT;
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

unique_ptr<Font> FreeTypeFontEngine::Open(const FontDesc &d) {
  auto fi = font_map.find(d);
  if (fi != font_map.end()) {
    unique_ptr<Font> ret = make_unique<Font>(*fi->second);
    ret->fg = d.fg;
    ret->bg = d.bg;
    return ret;
  }

  auto ri = resource.find(d.name);
  if (ri == resource.end()) return 0;
  unique_ptr<Font> ret = make_unique<Font>(this, d, ri->second);
  ret->glyph = make_shared<GlyphMap>(parent->parent);
  int count = InitGlyphs(ret.get(), &ret->glyph->table[0], ret->glyph->table.size());
  ret->fix_metrics = true;
  ret->mix_fg = true;

  bool new_cache = false, pre_load = false;
  ret->glyph->cache = 
    (!new_cache ? parent->GetGlyphCache() :
     make_shared<GlyphCache>(parent->parent, 0, AtlasFontEngine::Dimension(ret->max_width, ret->Height(), count)));
  GlyphCache *cache = ret->glyph->cache.get();

  if (new_cache) cache->tex.RenewBuffer();
  if (pre_load) LoadGlyphs(ret.get(), &ret->glyph->table[0], ret->glyph->table.size());
  if (FLAGS_atlas_dump) AtlasFontEngine::WriteAtlas(parent->appinfo, d.Filename(), ret.get(), &cache->tex);
  if (new_cache) {
    cache->tex.LoadGL();
    cache->tex.ClearBuffer();
  }

  font_map[d] = ret.get();
  return ret;
}

}; // namespace LFL
