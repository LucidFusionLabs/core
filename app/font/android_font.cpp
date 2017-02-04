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
extern FlagOfType<string> FLAGS_font_engine_;
extern FlagOfType<string> FLAGS_font_;
extern FlagOfType<int> FLAGS_font_size_;

struct UnicodeBlock {
  struct Block {
    uint16_t beg, end; string name;
    Block(uint16_t b, uint16_t e, string n) : beg(b), end(e), name(n) {}
    bool operator<(const Block &x) { return end < x.end; }
  };

  static string Get(uint16_t v) {
    static vector<Block> blocks{
      Block(0x0020,  0x007F,  "Basic Latin"),
      Block(0x00A0,  0x00FF,  "Latin-1 Supplement"),
      Block(0x0100,  0x017F,  "Latin Extended-A"),
      Block(0x0180,  0x024F,  "Latin Extended-B"),
      Block(0x0250,  0x02AF,  "IPA Extensions"),
      Block(0x02B0,  0x02FF,  "Spacing Modifier Letters"),
      Block(0x0300,  0x036F,  "Combining Diacritical Marks"),
      Block(0x0370,  0x03FF,  "Greek and Coptic"),
      Block(0x0400,  0x04FF,  "Cyrillic"),
      Block(0x0500,  0x052F,  "Cyrillic Supplementary"),
      Block(0x0530,  0x058F,  "Armenian"),
      Block(0x0590,  0x05FF,  "Hebrew"),
      Block(0x0600,  0x06FF,  "Arabic"),
      Block(0x0700,  0x074F,  "Syriac"),
      Block(0x0780,  0x07BF,  "Thaana"),
      Block(0x0900,  0x097F,  "Devanagari"),
      Block(0x0980,  0x09FF,  "Bengali"),
      Block(0x0A00,  0x0A7F,  "Gurmukhi"),
      Block(0x0A80,  0x0AFF,  "Gujarati"),
      Block(0x0B00,  0x0B7F,  "Oriya"),
      Block(0x0B80,  0x0BFF,  "Tamil"),
      Block(0x0C00,  0x0C7F,  "Telugu"),
      Block(0x0C80,  0x0CFF,  "Kannada"),
      Block(0x0D00,  0x0D7F,  "Malayalam"),
      Block(0x0D80,  0x0DFF,  "Sinhala"),
      Block(0x0E00,  0x0E7F,  "Thai"),
      Block(0x0E80,  0x0EFF,  "Lao"),
      Block(0x0F00,  0x0FFF,  "Tibetan"),
      Block(0x1000,  0x109F,  "Myanmar"),
      Block(0x10A0,  0x10FF,  "Georgian"),
      Block(0x1100,  0x11FF,  "Hangul Jamo"),
      Block(0x1200,  0x137F,  "Ethiopic"),
      Block(0x13A0,  0x13FF,  "Cherokee"),
      Block(0x1400,  0x167F,  "Unified Canadian Aboriginal Syllabics"),
      Block(0x1680,  0x169F,  "Ogham"),
      Block(0x16A0,  0x16FF,  "Runic"),
      Block(0x1700,  0x171F,  "Tagalog"),
      Block(0x1720,  0x173F,  "Hanunoo"),
      Block(0x1740,  0x175F,  "Buhid"),
      Block(0x1760,  0x177F,  "Tagbanwa"),
      Block(0x1780,  0x17FF,  "Khmer"),
      Block(0x1800,  0x18AF,  "Mongolian"),
      Block(0x1900,  0x194F,  "Limbu"),
      Block(0x1950,  0x197F,  "Tai Le"),
      Block(0x19E0,  0x19FF,  "Khmer Symbols"),
      Block(0x1D00,  0x1D7F,  "Phonetic Extensions"),
      Block(0x1E00,  0x1EFF,  "Latin Extended Additional"),
      Block(0x1F00,  0x1FFF,  "Greek Extended"),
      Block(0x2000,  0x206F,  "General Punctuation"),
      Block(0x2070,  0x209F,  "Superscripts and Subscripts"),
      Block(0x20A0,  0x20CF,  "Currency Symbols"),
      Block(0x20D0,  0x20FF,  "Combining Diacritical Marks for Symbols"),
      Block(0x2100,  0x214F,  "Letterlike Symbols"),
      Block(0x2150,  0x218F,  "Number Forms"),
      Block(0x2190,  0x21FF,  "Arrows"),
      Block(0x2200,  0x22FF,  "Mathematical Operators"),
      Block(0x2300,  0x23FF,  "Miscellaneous Technical"),
      Block(0x2400,  0x243F,  "Control Pictures"),
      Block(0x2440,  0x245F,  "Optical Character Recognition"),
      Block(0x2460,  0x24FF,  "Enclosed Alphanumerics"),
      Block(0x2500,  0x257F,  "Box Drawing"),
      Block(0x2580,  0x259F,  "Block Elements"),
      Block(0x25A0,  0x25FF,  "Geometric Shapes"),
      Block(0x2600,  0x26FF,  "Miscellaneous Symbols"),
      Block(0x2700,  0x27BF,  "Dingbats"),
      Block(0x27C0,  0x27EF,  "Miscellaneous Mathematical Symbols-A"),
      Block(0x27F0,  0x27FF,  "Supplemental Arrows-A"),
      Block(0x2800,  0x28FF,  "Braille Patterns"),
      Block(0x2900,  0x297F,  "Supplemental Arrows-B"),
      Block(0x2980,  0x29FF,  "Miscellaneous Mathematical Symbols-B"),
      Block(0x2A00,  0x2AFF,  "Supplemental Mathematical Operators"),
      Block(0x2B00,  0x2BFF,  "Miscellaneous Symbols and Arrows"),
      Block(0x2E80,  0x2EFF,  "CJK Radicals Supplement"),
      Block(0x2F00,  0x2FDF,  "Kangxi Radicals"),
      Block(0x2FF0,  0x2FFF,  "Ideographic Description Characters"),
      Block(0x3000,  0x303F,  "CJK Symbols and Punctuation"),
      Block(0x3040,  0x309F,  "Hiragana"),
      Block(0x30A0,  0x30FF,  "Katakana"),
      Block(0x3100,  0x312F,  "Bopomofo"),
      Block(0x3130,  0x318F,  "Hangul Compatibility Jamo"),
      Block(0x3190,  0x319F,  "Kanbun"),
      Block(0x31A0,  0x31BF,  "Bopomofo Extended"),
      Block(0x31F0,  0x31FF,  "Katakana Phonetic Extensions"),
      Block(0x3200,  0x32FF,  "Enclosed CJK Letters and Months"),
      Block(0x3300,  0x33FF,  "CJK Compatibility"),
      Block(0x3400,  0x4DBF,  "CJK Unified Ideographs Extension A"),
      Block(0x4DC0,  0x4DFF,  "Yijing Hexagram Symbols"),
      Block(0x4E00,  0x9FFF,  "CJK Unified Ideographs"),
      Block(0xA000,  0xA48F,  "Yi Syllables"),
      Block(0xA490,  0xA4CF,  "Yi Radicals"),
      Block(0xAC00,  0xD7AF,  "Hangul Syllables"),
      Block(0xD800,  0xDB7F,  "High Surrogates"),
      Block(0xDB80,  0xDBFF,  "High Private Use Surrogates"),
      Block(0xDC00,  0xDFFF,  "Low Surrogates"),
      Block(0xE000,  0xF8FF,  "Private Use Area"),
      Block(0xF900,  0xFAFF,  "CJK Compatibility Ideographs"),
      Block(0xFB00,  0xFB4F,  "Alphabetic Presentation Forms"),
      Block(0xFB50,  0xFDFF,  "Arabic Presentation Forms-A"),
      Block(0xFE00,  0xFE0F,  "Variation Selectors"),
      Block(0xFE20,  0xFE2F,  "Combining Half Marks"),
      Block(0xFE30,  0xFE4F,  "CJK Compatibility Forms"),
      Block(0xFE50,  0xFE6F,  "Small Form Variants"),
      Block(0xFE70,  0xFEFF,  "Arabic Presentation Forms-B"),
      Block(0xFF00,  0xFFEF,  "Halfwidth and Fullwidth Forms"),
      Block(0xFFF0,  0xFFFF,  "Specials"),
#if 0
      Block(0x10000, 0x1007F, "Linear B Syllabary"),
      Block(0x10080, 0x100FF, "Linear B Ideograms"),
      Block(0x10100, 0x1013F, "Aegean Numbers"),
      Block(0x10300, 0x1032F, "Old Italic"),
      Block(0x10330, 0x1034F, "Gothic"),
      Block(0x10380, 0x1039F, "Ugaritic"),
      Block(0x10400, 0x1044F, "Deseret"),
      Block(0x10450, 0x1047F, "Shavian"),
      Block(0x10480, 0x104AF, "Osmanya"),
      Block(0x10800, 0x1083F, "Cypriot Syllabary"),
      Block(0x1D000, 0x1D0FF, "Byzantine Musical Symbols"),
      Block(0x1D100, 0x1D1FF, "Musical Symbols"),
      Block(0x1D300, 0x1D35F, "Tai Xuan Jing Symbols"),
      Block(0x1D400, 0x1D7FF, "Mathematical Alphanumeric Symbols"),
      Block(0x20000, 0x2A6DF, "CJK Unified Ideographs Extension B"),
      Block(0x2F800, 0x2FA1F, "CJK Compatibility Ideographs Supplement"),
      Block(0xE0000, 0xE007F, "Tags"),
#endif
    };
    auto it = lower_bound(blocks.begin(), blocks.end(), Block(0, v, ""));
    return (it != blocks.end() && v >= it->beg) ? it->name : "";
  }
};

void AndroidFontEngine::Shutdown() {}
void AndroidFontEngine::Init() {
  static bool init = false;
  if (!init && (init=true)) {}
}

string AndroidFontEngine::DebugString(Font *f) const {
  return StrCat("AndroidFont(", f->desc->DebugString(), "), H=", f->Height(), " fixed_width=", f->fixed_width, " mono=", f->mono?f->max_width:0);
}

void AndroidFontEngine::SetDefault() {
  if (!FLAGS_font_engine_.override) FLAGS_font_engine = "android";
  if (!FLAGS_font_.override) FLAGS_font = "default";
  if (!FLAGS_font_size_.override) FLAGS_font_size = 15;
}

int AndroidFontEngine::InitGlyphs(Font *f, Glyph *g, int n) {
  static FreeTypeFontEngine *ttf_engine = app->fonts->freetype_engine.get();
  int ret = ttf_engine->InitGlyphs(f, g, n);
  for (Glyph *e = g + n; g != e; ++g) {
    if (g->internal.freetype.id) continue;
    string name = "DroidSans", lang = UnicodeBlock::Get(g->id), fn;
    do {
      if (lang.size()) if (LocalFile::IsFile((fn = StrCat("/system/fonts/DroidSans", lang, "-Regular.ttf")))) break;
      if (lang.size()) if (LocalFile::IsFile((fn = StrCat("/system/fonts/NotoSans",  lang, "-Regular.ttf")))) break;
      /**/             if (LocalFile::IsFile((fn = StrCat("/system/fonts/DroidSansFallback.ttf"))))           break;
      fn = "";
    } while(0);
    if (fn.empty()) { INFO("missing glyph ", g->id); continue; }

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
    if (g->internal.freetype.id) { g->internal.freetype.substitute = sub; continue; }
    else { INFO("missing glyph ", g->id); continue; }
  }
  return ret;
}

int AndroidFontEngine::LoadGlyphs(Font *f, const Glyph *g, int n) {
  static FreeTypeFontEngine *ttf_engine = app->fonts->freetype_engine.get();
  for (const Glyph *e = g + n; g != e; ++g)
    ttf_engine->LoadGlyphs(X_or_Y(g->internal.freetype.substitute, f), g, 1);
  return n;
}

unique_ptr<Font> AndroidFontEngine::Open(const FontDesc &d) {
  Init();
  FontDesc ttf = d;
  if (ttf.name.size() && ttf.name[0] != '/') do {
    string name = ttf.name;
    if (name == "default") name = "DroidSans";
    if (d.flag & FontDesc::Mono) if (LocalFile::IsFile((ttf.name = StrCat("/system/fonts/", name, "Mono.ttf"))))     break;
    if (d.flag & FontDesc::Bold) if (LocalFile::IsFile((ttf.name = StrCat("/system/fonts/", name, "-Bold.ttf"))))    break;
    /**/                         if (LocalFile::IsFile((ttf.name = StrCat("/system/fonts/", name, ".ttf"))))         break;
    /**/                         if (LocalFile::IsFile((ttf.name = StrCat("/system/fonts/", name, "-Regular.ttf")))) break;
    ERROR("no system font named ", name);
    return nullptr;
  } while(0);
  return OpenTTF(ttf);
}

unique_ptr<Font> AndroidFontEngine::OpenTTF(const FontDesc &ttf) {
  static FreeTypeFontEngine *ttf_engine = app->fonts->freetype_engine.get();
  if (!ttf_engine->Init(ttf)) { ERROR("ttf init failed for ", ttf.DebugString()); return nullptr; }
  unique_ptr<Font> ret = ttf_engine->Open(ttf);
  ret->engine = this;
  return ret;
}

}; // namespace LFL
