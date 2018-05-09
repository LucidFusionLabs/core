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

#include <fontconfig/fontconfig.h>

namespace LFL {
extern FlagOfType<string> FLAGS_font_engine_;
extern FlagOfType<string> FLAGS_font_;
extern FlagOfType<int> FLAGS_font_size_;

static FcConfig *fc_config;

void FCFontEngine::Shutdown() { FcFini(); }

string FCFontEngine::DebugString(Font *f) const {
  return StrCat("FCFont(", f->desc->DebugString(), "), H=", f->Height(), " fixed_width=", f->fixed_width, " mono=", f->mono?f->max_width:0);
}

void FCFontEngine::SetDefault() {
  if (!FLAGS_font_engine_.override) FLAGS_font_engine = "fc";
  if (!FLAGS_font_.override) FLAGS_font = "default";
  if (!FLAGS_font_size_.override) FLAGS_font_size = 15;
}

int FCFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
  static FreeTypeFontEngine *ttf_engine = app->fonts->freetype_engine.get();
  int ret = ttf_engine->InitGlyphs(f, g, n);
  for (Glyph *e = g + n; g != e; ++g) {
    if (g->internal.freetype.id) continue;
    FcPattern *pattern = FcPatternCreate();
    FcCharSet *char_set = FcCharSetCreate();
    FcCharSetAddChar(char_set, g->id);
    FcPatternAddCharSet(pattern, FC_CHARSET, char_set);
    FcCharSetDestroy(char_set);

    FcChar8 *file=0;
    FcResult result;
    FcConfigSubstitute(fc_config, pattern, FcMatchPattern);
    FcDefaultSubstitute(pattern);
    pattern = FcFontMatch(fc_config, pattern, &result);
    if (FcPatternGetString(pattern, FC_FILE, 0, &file) != FcResultMatch)
    { ERROR("no substitute for ", g->id); FcPatternDestroy(pattern); continue; }
    string fn = MakeSigned(file);
    FcPatternDestroy(pattern);

    bool inserted = 0;
    auto ri = FindOrInsert(resource, fn, &inserted);
    if (inserted) {
      FontDesc ttf = f ? *f->desc : FontDesc();
      ttf.name = fn;
      ri->second = make_shared<Resource>(OpenTTF(ttf));
      if (!ri->second->font) { ERROR("load substitute failed: ", fn); resource.erase(fn); continue; }
    }

    Font *sub = ri->second->font.get();
    ttf_engine->InitGlyphs(sub, g, 1);
    g->internal.freetype.substitute = sub;
    f->UpdateMetrics(g);
  }
  return ret;
}

int FCFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) {
  static FreeTypeFontEngine *ttf_engine = app->fonts->freetype_engine.get();
  for (const Glyph *e = g + n; g != e; ++g)
    ttf_engine->LoadGlyphs(X_or_Y(g->internal.freetype.substitute, f), g, 1);
  return n;
}

void FCFontEngine::Init() {
  static bool init = false;
  if (!init && (init=true)) {
    FcInit();
    fc_config = FcInitLoadConfigAndFonts();
  }
}

unique_ptr<Font> FCFontEngine::Open(const FontDesc &d) {
  Init();
  FontDesc ttf = d;
  if (ttf.name.size() && ttf.name[0] != '/') {
    FcPattern *pattern = d.name == "default" ? FcPatternCreate() : FcNameParse(MakeUnsigned(d.name.c_str()));
    if (d.flag & FontDesc::Mono) FcPatternAddString(pattern, FC_FAMILY, MakeUnsigned("Mono"));
    if (d.flag & FontDesc::Bold) FcPatternAddInteger(pattern, FC_WEIGHT, FC_WEIGHT_BOLD);

    FcChar8 *file=0;
    FcResult result;
    FcConfigSubstitute(fc_config, pattern, FcMatchPattern);
    FcDefaultSubstitute(pattern);
    pattern = FcFontMatch(fc_config, pattern, &result);
    if (FcPatternGetString(pattern, FC_FILE, 0, &file) != FcResultMatch) {
      FcPatternDestroy(pattern);
      return ERRORv(nullptr, "FcPatternGetString ", d.name, ": no file");
    }

    ttf.name = MakeSigned(file);
    FcPatternDestroy(pattern);
  }
  return OpenTTF(ttf);
}

unique_ptr<Font> FCFontEngine::OpenTTF(const FontDesc &ttf) {
  static FreeTypeFontEngine *ttf_engine = app->fonts->freetype_engine.get();
  if (!ttf_engine->Init(ttf)) { ERROR("ttf init failed for ", ttf.DebugString()); return nullptr; }
  unique_ptr<Font> ret = ttf_engine->Open(ttf);
  ret->engine = this;
  return ret;
}

}; // namespace LFL
