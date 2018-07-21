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
#include "core/web/browser/browser.h"

namespace LFL {
unique_ptr<Font> IPCClientFontEngine::Open(const FontDesc &d) {
  unique_ptr<Font> ret = make_unique<Font>(this, d, make_shared<Resource>());
  ret->glyph = make_shared<GlyphMap>(parent->parent);
  ret->glyph->cache = parent->GetGlyphCache();
  dispatch->main_process->OpenSystemFont(d, bind(&IPCClientFontEngine::OpenSystemFontResponse, this, ret.get(), _1, _2));
  if (1 && dispatch->main_process->browser) dispatch->main_process->WaitAllOpenSystemFontResponse();
  return ret;
}

int IPCClientFontEngine::OpenSystemFontResponse(Font *f, const IPC::OpenSystemFontResponse *res, const MultiProcessBuffer &mpb) {
  if (!res) return IPC::Error;
  dynamic_cast<Resource*>(f->resource.get())->id = res->font_id();
  f->SetMetrics(res->ascender(), res->descender(), res->max_width(), res->fixed_width(), res->missing_glyph(),
                res->mix_fg(), res->has_bg(), res->fix_metrics(), res->scale());
  f->glyph->table_start = res->start_glyph_id();
  f->glyph->table.clear();
  for (int i=0, l=res->num_glyphs(); i != l; ++i) f->glyph->table.emplace_back(parent->parent);
  GlyphMetrics *g = reinterpret_cast<GlyphMetrics*>(mpb.buf);
  for (int i=0, l=f->glyph->table.size(); i<l; i++) f->glyph->table[i].FromMetrics(g[i]);
  if (dispatch->main_process) if (auto html = dispatch->main_process->browser->doc.DocElement()) html->SetStyleDirty();
  return IPC::Done;
}

int IPCClientFontEngine::GetId(Font *f) { return dynamic_cast<Resource*>(f->resource.get())->id; }
int IPCClientFontEngine::InitGlyphs(Font *f, Glyph *g, int n) { return 0; }
int IPCClientFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) { return 0; }
string IPCClientFontEngine::DebugString(Font *f) const { return ""; }

unique_ptr<Font> IPCServerFontEngine::Open(const FontDesc&) { return nullptr; }
int IPCServerFontEngine::InitGlyphs(Font *f, Glyph *g, int n) { return 0; }
int IPCServerFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) { return 0; }
string IPCServerFontEngine::DebugString(Font *f) const { return ""; }

}; // namespace LFL
