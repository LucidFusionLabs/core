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

#include <fontconfig/fontconfig.h>

namespace LFL {
static FcConfig *fc_config;

void FCFontEngine::Shutdown() {}

string FCFontEngine::DebugString(Font *f) const {
  return StrCat("FCFont(", f->desc->DebugString(), "), H=", f->Height(), " fixed_width=", f->fixed_width, " mono=", f->mono?f->max_width:0);
}

void FCFontEngine::SetDefault() {
  FLAGS_font_engine = "fc";
  FLAGS_default_font = "default";
  FLAGS_default_font_size = 15;
}

int FCFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
  return n;
}

int FCFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) {
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
  FcResult result;
  FcChar8 *file=0;
  FcPattern *pattern = d.name == "default" ? FcPatternCreate() : FcNameParse(MakeUnsigned(d.name.c_str()));
  // FcPatternAddString(pattern, FC_FAMILY, MakeUnsigned("mono"));
  FcDefaultSubstitute(pattern);
  FcConfigSubstitute(fc_config, pattern, FcMatchFont);
  pattern = FcFontMatch(fc_config, pattern, &result);
  if (FcPatternGetString(pattern, FC_FILE, 0, &file) != FcResultMatch)
    return ERRORv(nullptr, "FcPatternGetString ", d.name, ": no file");
  FontDesc ttf = d;
  ttf.name = MakeSigned(file);
  auto ttf_engine = app->fonts->freetype_engine.get();
  if (!ttf_engine->Init(ttf)) FATAL("ttf init failed for ", ttf.DebugString());
  return ttf_engine->Open(ttf);
}

}; // namespace LFL
