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

namespace LFL {
FreeTypeFontEngine::Resource::~Resource() {}
string FreeTypeFontEngine::DebugString(Font *f) const { return "NullFreeTypeFont"; }
void FreeTypeFontEngine::SetDefault() {}
bool FreeTypeFontEngine::Init(const FontDesc &d) { return false; }
void FreeTypeFontEngine::Init() {}
void FreeTypeFontEngine::SubPixelFilter(const Box &b, unsigned char *buf, int linesize, int pf) {}
FreeTypeFontEngine::Resource *FreeTypeFontEngine::OpenFile(const FontDesc &d) { return 0; }
FreeTypeFontEngine::Resource *FreeTypeFontEngine::OpenBuffer(const FontDesc &d, string *content) { return 0; }
int FreeTypeFontEngine::InitGlyphs(Font *f, Glyph *g, int n) { return 0; }
int FreeTypeFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) { return 0; }
unique_ptr<Font> FreeTypeFontEngine::Open(const FontDesc &d) { return nullptr; }

#ifdef LFL_LINUX
void FCFontEngine::Shutdown() {}
string FCFontEngine::DebugString(Font *f) const {}
void FCFontEngine::SetDefault() {}
int FCFontEngine::InitGlyphs(Font *f, Glyph *g, int n) { return 0; }
int FCFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) { return 0; }
void FCFontEngine::Init() {}
unique_ptr<Font> FCFontEngine::Open(const FontDesc &d) { return nullptr; }
#endif

}; // namespace LFL
