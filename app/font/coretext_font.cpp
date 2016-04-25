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

inline CFStringRef ToCFStr(const LFL::string &n) { return CFStringCreateWithCString(0, n.data(), kCFStringEncodingUTF8); }
inline LFL::string FromCFStr(CFStringRef in) {
  LFL::string ret(CFStringGetMaximumSizeForEncoding(CFStringGetLength(in), kCFStringEncodingUTF8), 0);
  if (!CFStringGetCString(in, &ret[0], ret.size(), kCFStringEncodingUTF8)) return LFL::string();
  ret.resize(strlen(ret.data()));
  return ret;
}

inline CFAttributedStringRef ToCFAStr(CTFontRef ctfont, char16_t glyph_id) {
  const void* attr_key[] = { kCTFontAttributeName };
  const void* attr_val[] = { ctfont };
  CFDictionaryRef attr = CFDictionaryCreate
    (kCFAllocatorDefault, attr_key, attr_val, sizeofarray(attr_key),
     &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFStringRef str = CFStringCreateWithCharacters(kCFAllocatorDefault, LFL::MakeUnsigned(&glyph_id), 1);
  CFAttributedStringRef astr = CFAttributedStringCreate(kCFAllocatorDefault, str, attr);
  CFRelease(attr);
  CFRelease(str);
  return astr;
}

namespace LFL {
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
    if (f->flag & FontDesc::Shadow) {
      // CGContextSetShadow(context, CGSizeMake(15, -20), 5); 
      CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
      CGFloat cv[] = { 0, 0, 0, 1 };
      CGColorRef c = CGColorCreate(cs, cv);
      CGContextSetShadowWithColor(context, CGSizeMake(15, -20), 5, c);
      CGColorRelease(c);
      CGColorSpaceRelease(cs); 
    }
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

CoreTextFontEngine::Resource::~Resource() {
  if (cgfont) CFRelease(cgfont);
}

string CoreTextFontEngine::DebugString(Font *f) const {
  return StrCat("CoreTextFont(", f->desc->DebugString(), "), H=", f->Height(), " fixed_width=", f->fixed_width, " mono=", f->mono?f->max_width:0, " advance_bounds = ", GetAdvanceBounds(f).DebugString());
}

void CoreTextFontEngine::SetDefault() {
  FLAGS_font_engine = "coretext";
#ifdef LFL_IPHONE
  FLAGS_default_font = "Menlo-Bold";
  FLAGS_default_font_size = 12;
#else
  FLAGS_default_font = "Monaco";
  FLAGS_default_font_size = 15;
#endif
}

int CoreTextFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
  CGSize advance;
  Resource *resource = dynamic_cast<Resource*>(f->resource.get());
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
  Resource *resource = dynamic_cast<Resource*>(f->resource.get());
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

unique_ptr<Font> CoreTextFontEngine::Open(const FontDesc &d) {
  bool inserted = 0;
  auto ri = FindOrInsert(resource, d.name, &inserted);
  if (inserted) {
    CFStringRef cfname = ToCFStr(d.name);
    ri->second = make_shared<Resource>();
    if (!(ri->second->cgfont = CGFontCreateWithFontName(cfname))) { CFRelease(cfname); resource.erase(d.name); return 0; }
#ifdef LFL_HARFBUZZ
    ri->second->hb_face = hb_coretext_face_create(ri->second->cgfont);
#endif
    CFRelease(cfname);
  }

  CTFontRef ctfont = CTFontCreateWithGraphicsFont(ri->second->cgfont, d.size, 0, 0);
  CGFloat ascent = CTFontGetAscent(ctfont), descent = CTFontGetDescent(ctfont), leading = CTFontGetLeading(ctfont);
  CFRelease(ctfont);

  unique_ptr<Font> ret = make_unique<Font>(this, d, ri->second);
  ret->glyph = make_shared<GlyphMap>();
  ret->ascender = RoundUp(ascent);
  ret->descender = RoundUp(descent) + RoundDown(leading);
  int count = InitGlyphs(ret.get(), &ret->glyph->table[0], ret->glyph->table.size());
  ret->fix_metrics = true;
  ret->has_bg = true;
  // ConvertColorFromGenericToDeviceRGB(d.fg.x, ret->fg.x);
  // ConvertColorFromGenericToDeviceRGB(d.bg.x, ret->bg.x);

  bool new_cache = false, pre_load = false;
  ret->glyph->cache =
    (!new_cache ? app->fonts->GetGlyphCache() :
     make_shared<GlyphCache>(0, AtlasFontEngine::Dimension(ret->max_width, ret->Height(), count)));
  GlyphCache *cache = ret->glyph->cache.get();

  if (new_cache) {
    cache->tex.RenewBuffer();
    cache->cgcontext = cache->tex.CGBitMap();
  }
  if (pre_load) LoadGlyphs(ret.get(), &ret->glyph->table[0], ret->glyph->table.size());
  if (FLAGS_atlas_dump) AtlasFontEngine::WriteAtlas(d.Filename(), ret.get(), &cache->tex);
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
  CTRunRef run = reinterpret_cast<CTRunRef>(CFArrayGetValueAtIndex(runs, 0));
  int glyphs = CTRunGetGlyphCount(run);
  CHECK_GT(glyphs, 0);
  if (glyphs != 1) ERROR("CoreTextFontEngine ", glyphs, " glyphs for codepoint ", glyph_id);
  if (id_out) {
    CGGlyph cgg;
    CTRunGetGlyphs(run, CFRangeMake(0,1), &cgg);
    *id_out = cgg;
  }
  CFDictionaryRef run_attr = CTRunGetAttributes(run);
  ctfont = reinterpret_cast<CTFontRef>(CFDictionaryGetValue(run_attr, kCTFontAttributeName));
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
    if (g->internal.coretext.advance) Min(&ret.x, float(g->internal.coretext.advance));
    if (1)                            Max(&ret.y, float(g->internal.coretext.advance));
  }
  return ret;
}

}; // namespace LFL
