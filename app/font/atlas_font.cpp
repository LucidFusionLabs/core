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

namespace LFL {
extern FlagOfType<string> FLAGS_font_engine_;
extern FlagOfType<string> FLAGS_font_;
extern FlagOfType<int> FLAGS_font_size_;

AtlasFontEngine::AtlasFontEngine(Fonts *f) : FontEngine(f) {}

string AtlasFontEngine::DebugString(Font *f) const {
  return StrCat("AtlasFont(", f->desc->DebugString(), "), H=", f->Height(), ", FW=", f->fixed_width);
}

void AtlasFontEngine::SetDefault() {
  FLAGS_font_engine = "atlas";
  FLAGS_font = "VeraMoBd.ttf";
  FLAGS_missing_glyph = 42;
  // FLAGS_font_size = 32;
}

bool AtlasFontEngine::Init(const FontDesc &d) {
  if (Font *f = OpenAtlas(parent, d)) {
    ScopedReentryGuard scoped(&in_init);
    font_map[d.name][d.fg][d.flag][d.size] = f;
    FontDesc de = d;
    de.engine = FontDesc::Engine::Atlas;
    Font *ret = parent->GetByDesc(de);
    if (ret != f) {} // leaks on exit()
    return ret;
  }
  return false;
}

unique_ptr<Font> AtlasFontEngine::Open(const FontDesc &d) {
  FontMap::iterator fi = font_map.find(d.name);
  if (fi == font_map.end() || !fi->second.size()) return ERRORv(nullptr, "AtlasFont ", d.DebugString(), " not found");

  bool is_fg_white = d.fg == Color::white;
  int max_ci = 2 - is_fg_white;
  for (int ci = 0; ci < max_ci; ++ci) {
    bool last_ci = ci == (max_ci - 1);
    ColorDesc c = ci ? ColorDesc(Color::white) : d.fg;
    FontColorMap::iterator i;
    FontFlagMap::iterator j;
    FontSizeMap::iterator k;
    FontSizeMap::reverse_iterator l;
    Font *f = 0;

    if ((i = fi->second.find(c))     == fi->second.end() || !i->second.size()) continue;
    if ((j = i->second.find(d.flag)) ==  i->second.end() || !j->second.size()) continue;

    if      ((k = j->second.lower_bound(d.size)) != j->second.end())  f = k->second;
    else if ((l = j->second.rbegin())            != j->second.rend()) f = l->second;

    if (!f) continue;
    if (!ci && f->size == d.size) {
      if (in_init) return unique_ptr<Font>(f);
      ERROR("OpenDuplicate ", d.DebugString());
    }

    unique_ptr<Font> ret = make_unique<Font>(this, d, f->resource);
    Font *primary        = dynamic_cast<Resource*>(f->resource.get())->primary;
    ret->mix_fg          = primary->mix_fg || ci;
    ret->mono            = primary->mono;
    ret->glyph           = primary->glyph;
    ret->missing_glyph   = primary->missing_glyph;
    ret->scale           = float(ret->size) / primary->size;
    ret->ascender        = RoundF(primary->ascender    * ret->scale);
    ret->descender       = RoundF(primary->descender   * ret->scale);
    ret->max_width       = RoundF(primary->max_width   * ret->scale);
    ret->fixed_width     = RoundF(primary->fixed_width * ret->scale);
    return ret;
  }
  return ERRORv(nullptr, "AtlasFont ", d.DebugString(), " clone failed");
}

Font *AtlasFontEngine::OpenAtlas(Fonts *fonts, const FontDesc &d) {
  Texture tex(fonts->loader->window);
  string fn = d.Filename();
  fonts->loader->LoadTexture(StrCat(fn, ".0000.png"), &tex);
  if (!tex.ID) return ERRORv(nullptr, "load ", fn, ".0000.png failed");

  MatrixFile gm;
  unique_ptr<File> gmfile(fonts->loader->OpenFile(MatrixFile::Filename(VersionedFileName(fonts->appinfo->assetdir.c_str(), fn.c_str(), "glyphs"), "matrix", 0)));
  if (gmfile && gm.Read(gmfile.get())) return ERRORv(nullptr, "load ", d.name, ".0000.glyphs.matrix failed");

  tex.owner = false;
  Resource *resource = new Resource();
  Font *ret = new Font(fonts->atlas_engine.get(fonts), d, shared_ptr<FontEngine::Resource>(resource));
  ret->glyph = make_shared<GlyphMap>(fonts->loader->window, make_shared<GlyphCache>(fonts->loader->window, tex.ID, tex.width, tex.height));
  ret->mix_fg = Color(d.bg).a() != 1.0;
  GlyphCache *cache = ret->glyph->cache.get();
  resource->primary = ret;

  float max_t = 0, max_u = 0;
  MatrixRowIter(gm.F) {
    int glyph_ind = int(gm.F->row(i)[0]);
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

  INFO("OpenAtlas ", d.DebugString(), ", texID=", tex.ID, ", height=", ret->Height(), ", fixed_width=", ret->fixed_width);
  return ret;
}

void AtlasFontEngine::WriteAtlas(ApplicationInfo *appinfo, const string &name, Font *f) { WriteAtlas(appinfo, name, f, &f->glyph->cache->tex); }
void AtlasFontEngine::WriteAtlas(ApplicationInfo *appinfo, const string &name, Font *f, Texture *t) {
  LocalFile lf(appinfo->assetdir + name + ".0000.png", "w");
  PngWriter::Write(&lf, *t);
  INFO("wrote ", lf.Filename());
  WriteGlyphFile(appinfo, name, f);
}

void AtlasFontEngine::WriteGlyphFile(ApplicationInfo *appinfo, const string &name, Font *f) {
  int glyph_count = 0, glyph_out = 0;
  for (auto &i : f->glyph->table) if (i.       tex.width && i.       tex.height) glyph_count++;
  for (auto &i : f->glyph->index) if (i.second.tex.width && i.second.tex.height) glyph_count++;

  unique_ptr<Matrix> gm = make_unique<Matrix>(glyph_count, 10);
  for (auto &i : f->glyph->table) if (i.       tex.width && i.       tex.height) i.       ToArray(gm->row(glyph_out++), gm->N);
  for (auto &i : f->glyph->index) if (i.second.tex.width && i.second.tex.height) i.second.ToArray(gm->row(glyph_out++), gm->N);
  MatrixFile(gm.release(), "").
    WriteVersioned(VersionedFileName(appinfo->assetdir.c_str(), name.c_str(), "glyphs"), 0);
}

void AtlasFontEngine::MakeFromPNGFiles(Fonts *fonts, const string &name, const vector<string> &png, const point &atlas_dim, Font **glyphs_out) {
  Font *ret = new Font(fonts->atlas_engine.get(fonts), FontDesc(name), shared_ptr<FontEngine::Resource>());
  ret->glyph = make_shared<GlyphMap>(fonts->loader->window, make_shared<GlyphCache>(fonts->loader->window, 0, atlas_dim.x, atlas_dim.y));
  for (int i=ret->glyph->table.size(), l=png.size(); i < l; ++i) ret->glyph->table.emplace_back(fonts->loader->window);

  GlyphCache *cache = ret->glyph->cache.get();
  cache->tex.RenewBuffer();

  for (int i = 0, skipped = 0; i < png.size(); ++i) {
    LocalFile in(png[i], "r");
    if (!in.Opened()) { INFO("Skipped: ", png[i]); skipped++; continue; }
    Glyph *out = &ret->glyph->table[i - skipped];
    out->id = i - skipped;

    if (PngReader::Read(&in, &out->tex)) { skipped++; continue; }
    Max(&ret->ascender, int16_t(out->tex.height));

    point cache_p;
    CHECK(cache->Add(&cache_p, out->tex.coord, out->tex.width, out->tex.height, ret->Height()));
    SimpleVideoResampler::Blit(out->tex.buf, cache->tex.buf, out->tex.width, out->tex.height,
                               out->tex.pf,   out->tex.LineSize(),   0,       0,
                               cache->tex.pf, cache->tex.LineSize(), cache_p.x, cache_p.y);
    out->tex.ClearBuffer();
  }

  WriteAtlas(fonts->appinfo, name, ret, &cache->tex);
  cache->tex.LoadGL();
  cache->tex.ClearBuffer();

  if (glyphs_out) *glyphs_out = ret;
  else delete ret;
}

void AtlasFontEngine::SplitIntoPNGFiles(GraphicsDeviceHolder *parent, const string &input_png_fn, const map<int, v4> &glyphs, const string &dir_out) {
  LocalFile in(input_png_fn, "r");
  if (!in.Opened()) return ERROR("open: ", input_png_fn);

  Texture png(parent);
  if (PngReader::Read(&in, &png)) return ERROR("read: ", input_png_fn);

  int count=0;
  for (map<int, v4>::const_iterator i = glyphs.begin(); i != glyphs.end(); ++i) {
    unsigned gx1 = RoundF(i->second.x * png.width), gy1 = RoundF((1 - i->second.y) * png.height);
    unsigned gx2 = RoundF(i->second.z * png.width), gy2 = RoundF((1 - i->second.w) * png.height);
    unsigned gw = gx2 - gx1, gh = gy1 - gy2;
    if (gw <= 0 || gh <= 0) continue;

    Texture glyph(parent);
    glyph.Resize(gw, gh, Texture::preferred_pf, Texture::Flag::CreateBuf);
    SimpleVideoResampler::Blit(png.buf, glyph.buf, glyph.width, glyph.height,
                               png  .pf, png  .LineSize(), gx1, gy2,
                               glyph.pf, glyph.LineSize(), 0,   0);

    LocalFile lf(dir_out + StringPrintf("glyph%03d.png", i->first), "w");
    CHECK(lf.Opened());
    PngWriter::Write(&lf, glyph);
  }
}

}; // namespace LFL
