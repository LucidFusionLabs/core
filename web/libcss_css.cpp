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
#include "core/web/browser.h"
#include "core/web/document.h"
extern "C" {
#include "libcss/libcss.h"
};
#include "core/web/libcss.h"

namespace LFL {
namespace DOM {
const int BackgroundAttachment::Fixed  = CSS_BACKGROUND_ATTACHMENT_FIXED;
const int BackgroundAttachment::Scroll = CSS_BACKGROUND_ATTACHMENT_SCROLL;

const int BackgroundRepeat::RepeatX  = CSS_BACKGROUND_REPEAT_REPEAT_X;
const int BackgroundRepeat::RepeatY  = CSS_BACKGROUND_REPEAT_REPEAT_Y;
const int BackgroundRepeat::Repeat   = CSS_BACKGROUND_REPEAT_REPEAT;
const int BackgroundRepeat::NoRepeat = CSS_BACKGROUND_REPEAT_NO_REPEAT;

const int BorderCollapse::Separate = CSS_BORDER_COLLAPSE_SEPARATE;
const int BorderCollapse::Collapse = CSS_BORDER_COLLAPSE_COLLAPSE;

const int BorderWidth::Thin   = CSS_BORDER_WIDTH_THIN;
const int BorderWidth::Medium = CSS_BORDER_WIDTH_MEDIUM;
const int BorderWidth::Thick  = CSS_BORDER_WIDTH_THICK;

const int BorderStyle::None   = CSS_BORDER_STYLE_NONE;
const int BorderStyle::Hidden = CSS_BORDER_STYLE_HIDDEN;
const int BorderStyle::Dotted = CSS_BORDER_STYLE_DOTTED;
const int BorderStyle::Dashed = CSS_BORDER_STYLE_DASHED;
const int BorderStyle::Solid  = CSS_BORDER_STYLE_SOLID;
const int BorderStyle::Double = CSS_BORDER_STYLE_DOUBLE;
const int BorderStyle::Groove = CSS_BORDER_STYLE_GROOVE;
const int BorderStyle::Ridge  = CSS_BORDER_STYLE_RIDGE;
const int BorderStyle::Inset  = CSS_BORDER_STYLE_INSET;
const int BorderStyle::Outset = CSS_BORDER_STYLE_OUTSET;

const int CaptionSide::Top    = CSS_CAPTION_SIDE_TOP;
const int CaptionSide::Bottom = CSS_CAPTION_SIDE_BOTTOM;

const int Clear::None  = CSS_CLEAR_NONE;
const int Clear::Left  = CSS_CLEAR_LEFT;
const int Clear::Right = CSS_CLEAR_RIGHT;
const int Clear::Both  = CSS_CLEAR_BOTH;

const int Cursor::Auto      = CSS_CURSOR_AUTO;
const int Cursor::Default   = CSS_CURSOR_DEFAULT;
const int Cursor::NEResize  = CSS_CURSOR_NE_RESIZE;
const int Cursor::WResize   = CSS_CURSOR_W_RESIZE;
const int Cursor::Move      = CSS_CURSOR_MOVE;
const int Cursor::Pointer   = CSS_CURSOR_POINTER;
const int Cursor::EResize   = CSS_CURSOR_E_RESIZE;
const int Cursor::NWResize  = CSS_CURSOR_NW_RESIZE;
const int Cursor::Text      = CSS_CURSOR_TEXT;
const int Cursor::Crosshair = CSS_CURSOR_CROSSHAIR;
const int Cursor::SEResize  = CSS_CURSOR_SE_RESIZE;
const int Cursor::Wait      = CSS_CURSOR_WAIT;
const int Cursor::Progress  = CSS_CURSOR_PROGRESS;
const int Cursor::SResize   = CSS_CURSOR_S_RESIZE;
const int Cursor::Help      = CSS_CURSOR_HELP;
const int Cursor::NResize   = CSS_CURSOR_N_RESIZE;
const int Cursor::SWResize  = CSS_CURSOR_SW_RESIZE;

const int Direction::LTR = CSS_DIRECTION_LTR;
const int Direction::RTL = CSS_DIRECTION_RTL;

const int Display::None        = CSS_DISPLAY_NONE;
const int Display::Row         = CSS_DISPLAY_TABLE_ROW;
const int Display::HeaderGroup = CSS_DISPLAY_TABLE_HEADER_GROUP;
const int Display::Inline      = CSS_DISPLAY_INLINE;
const int Display::Cell        = CSS_DISPLAY_TABLE_CELL;
const int Display::FooterGroup = CSS_DISPLAY_TABLE_FOOTER_GROUP;
const int Display::Block       = CSS_DISPLAY_BLOCK;
const int Display::Column      = CSS_DISPLAY_TABLE_COLUMN;
const int Display::InlineBlock = CSS_DISPLAY_INLINE_BLOCK;
const int Display::RunIn       = CSS_DISPLAY_RUN_IN;
const int Display::Caption     = CSS_DISPLAY_TABLE_CAPTION;
const int Display::InlineTable = CSS_DISPLAY_INLINE_TABLE;
const int Display::ListItem    = CSS_DISPLAY_LIST_ITEM;
const int Display::RowGroup    = CSS_DISPLAY_TABLE_ROW_GROUP;
const int Display::Table       = CSS_DISPLAY_TABLE;
const int Display::ColGroup    = CSS_DISPLAY_TABLE_COLUMN_GROUP;

const int EmptyCells::Show = CSS_EMPTY_CELLS_SHOW;
const int EmptyCells::Hide = CSS_EMPTY_CELLS_HIDE;

const int Float::Left  = CSS_FLOAT_LEFT;
const int Float::Right = CSS_FLOAT_RIGHT;
const int Float::None  = CSS_FLOAT_NONE; 

const int FontFamily::Serif     = CSS_FONT_FAMILY_SERIF;
const int FontFamily::SansSerif = CSS_FONT_FAMILY_SANS_SERIF;
const int FontFamily::Cursive   = CSS_FONT_FAMILY_CURSIVE;
const int FontFamily::Fantasy   = CSS_FONT_FAMILY_FANTASY;
const int FontFamily::Monospace = CSS_FONT_FAMILY_MONOSPACE;

const int FontSize::XXSmall = CSS_FONT_SIZE_XX_SMALL;
const int FontSize::XLarge  = CSS_FONT_SIZE_X_LARGE;
const int FontSize::XSmall  = CSS_FONT_SIZE_X_SMALL;
const int FontSize::XXLarge = CSS_FONT_SIZE_XX_LARGE;
const int FontSize::Small   = CSS_FONT_SIZE_SMALL;
const int FontSize::Larger  = CSS_FONT_SIZE_LARGER;
const int FontSize::Medium  = CSS_FONT_SIZE_MEDIUM;
const int FontSize::Smaller = CSS_FONT_SIZE_SMALLER;
const int FontSize::Large   = CSS_FONT_SIZE_LARGE;

const int FontStyle::Normal  = CSS_FONT_STYLE_NORMAL;
const int FontStyle::Italic  = CSS_FONT_STYLE_ITALIC;
const int FontStyle::Oblique = CSS_FONT_STYLE_OBLIQUE;

const int FontVariant::Normal    = CSS_FONT_VARIANT_NORMAL;
const int FontVariant::SmallCaps = CSS_FONT_VARIANT_SMALL_CAPS;

const int FontWeight::Normal  = CSS_FONT_WEIGHT_NORMAL;
const int FontWeight::_400    = CSS_FONT_WEIGHT_400;
const int FontWeight::Bold    = CSS_FONT_WEIGHT_BOLD;
const int FontWeight::_500    = CSS_FONT_WEIGHT_500;
const int FontWeight::Bolder  = CSS_FONT_WEIGHT_BOLDER;
const int FontWeight::_600    = CSS_FONT_WEIGHT_600;
const int FontWeight::Lighter = CSS_FONT_WEIGHT_LIGHTER;
const int FontWeight::_700    = CSS_FONT_WEIGHT_700;
const int FontWeight::_100    = CSS_FONT_WEIGHT_100;
const int FontWeight::_800    = CSS_FONT_WEIGHT_800;
const int FontWeight::_200    = CSS_FONT_WEIGHT_200;
const int FontWeight::_900    = CSS_FONT_WEIGHT_900;
const int FontWeight::_300    = CSS_FONT_WEIGHT_300;

const int ListStylePosition::Inside  = CSS_LIST_STYLE_POSITION_INSIDE;
const int ListStylePosition::Outside = CSS_LIST_STYLE_POSITION_OUTSIDE;

const int ListStyleType::None               = CSS_LIST_STYLE_TYPE_NONE;
const int ListStyleType::LowerLatin         = CSS_LIST_STYLE_TYPE_LOWER_LATIN;
const int ListStyleType::Disc               = CSS_LIST_STYLE_TYPE_DISC;
const int ListStyleType::UpperLatin         = CSS_LIST_STYLE_TYPE_UPPER_LATIN;
const int ListStyleType::Circle             = CSS_LIST_STYLE_TYPE_CIRCLE;
const int ListStyleType::Armenian           = CSS_LIST_STYLE_TYPE_ARMENIAN;
const int ListStyleType::Square             = CSS_LIST_STYLE_TYPE_SQUARE;
const int ListStyleType::Georgian           = CSS_LIST_STYLE_TYPE_GEORGIAN;
const int ListStyleType::Decimal            = CSS_LIST_STYLE_TYPE_DECIMAL;
const int ListStyleType::LowerAlpha         = CSS_LIST_STYLE_TYPE_LOWER_ALPHA;
const int ListStyleType::LowerRoman         = CSS_LIST_STYLE_TYPE_LOWER_ROMAN;
const int ListStyleType::UpperAlpha         = CSS_LIST_STYLE_TYPE_UPPER_ALPHA;
const int ListStyleType::UpperRoman         = CSS_LIST_STYLE_TYPE_UPPER_ROMAN;
const int ListStyleType::DecimalLeadingZero = CSS_LIST_STYLE_TYPE_DECIMAL_LEADING_ZERO;
const int ListStyleType::LowerGreek         = CSS_LIST_STYLE_TYPE_LOWER_GREEK;

const int Overflow::Visible = CSS_OVERFLOW_VISIBLE;
const int Overflow::Scroll  = CSS_OVERFLOW_SCROLL;
const int Overflow::Hidden  = CSS_OVERFLOW_HIDDEN;
const int Overflow::Auto    = CSS_OVERFLOW_AUTO;

const int PageBreak::Auto        = CSS_BREAK_AFTER_AUTO;
const int PageBreak::Page        = CSS_BREAK_AFTER_PAGE;
const int PageBreak::Avoid       = CSS_BREAK_AFTER_AVOID;
const int PageBreak::Column      = CSS_BREAK_AFTER_COLUMN;
const int PageBreak::Always      = CSS_BREAK_AFTER_ALWAYS;
const int PageBreak::AvoidPage   = CSS_BREAK_AFTER_AVOID_PAGE;
const int PageBreak::Left        = CSS_BREAK_AFTER_LEFT;
const int PageBreak::AvoidColumn = CSS_BREAK_AFTER_AVOID_COLUMN;
const int PageBreak::Right       = CSS_BREAK_AFTER_RIGHT;

const int Position::Static   = CSS_POSITION_STATIC;
const int Position::Relative = CSS_POSITION_RELATIVE;
const int Position::Absolute = CSS_POSITION_ABSOLUTE;
const int Position::Fixed    = CSS_POSITION_FIXED;

const int TableLayout::Auto  = CSS_TABLE_LAYOUT_AUTO;
const int TableLayout::Fixed = CSS_TABLE_LAYOUT_FIXED;

const int TextAlign::Left    = CSS_TEXT_ALIGN_LEFT;
const int TextAlign::Center  = CSS_TEXT_ALIGN_CENTER;
const int TextAlign::Right   = CSS_TEXT_ALIGN_RIGHT;
const int TextAlign::Justify = CSS_TEXT_ALIGN_JUSTIFY;

const int TextDecoration::None        = CSS_TEXT_DECORATION_NONE;
const int TextDecoration::Blink       = CSS_TEXT_DECORATION_BLINK;
const int TextDecoration::LineThrough = CSS_TEXT_DECORATION_LINE_THROUGH;
const int TextDecoration::Overline    = CSS_TEXT_DECORATION_OVERLINE;
const int TextDecoration::Underline   = CSS_TEXT_DECORATION_UNDERLINE;

const int TextTransform::Capitalize = CSS_TEXT_TRANSFORM_CAPITALIZE;
const int TextTransform::Lowercase  = CSS_TEXT_TRANSFORM_LOWERCASE;
const int TextTransform::Uppercase  = CSS_TEXT_TRANSFORM_UPPERCASE;
const int TextTransform::None       = CSS_TEXT_TRANSFORM_NONE;

const int UnicodeBidi::Normal   = CSS_UNICODE_BIDI_NORMAL;
const int UnicodeBidi::Embed    = CSS_UNICODE_BIDI_EMBED;
const int UnicodeBidi::Override = CSS_UNICODE_BIDI_BIDI_OVERRIDE;

const int VerticalAlign::Baseline   = CSS_VERTICAL_ALIGN_BASELINE;
const int VerticalAlign::TextTop    = CSS_VERTICAL_ALIGN_TEXT_TOP;
const int VerticalAlign::Sub        = CSS_VERTICAL_ALIGN_SUB;
const int VerticalAlign::Middle     = CSS_VERTICAL_ALIGN_MIDDLE;
const int VerticalAlign::Super      = CSS_VERTICAL_ALIGN_SUPER;
const int VerticalAlign::Bottom     = CSS_VERTICAL_ALIGN_BOTTOM;
const int VerticalAlign::Top        = CSS_VERTICAL_ALIGN_TOP;
const int VerticalAlign::TextBottom = CSS_VERTICAL_ALIGN_TEXT_BOTTOM;

const int Visibility::Visible  = CSS_VISIBILITY_VISIBLE;
const int Visibility::Hidden   = CSS_VISIBILITY_HIDDEN;
const int Visibility::Collapse = CSS_VISIBILITY_COLLAPSE;

const int WhiteSpace::Normal  = CSS_WHITE_SPACE_NORMAL;
const int WhiteSpace::PreWrap = CSS_WHITE_SPACE_PRE_WRAP;
const int WhiteSpace::Pre     = CSS_WHITE_SPACE_PRE;
const int WhiteSpace::PreLine = CSS_WHITE_SPACE_PRE_LINE;
const int WhiteSpace::Nowrap  = CSS_WHITE_SPACE_NOWRAP;

#include "core/web/css_common.h"
}; // namespace DOM

lwc_string *LibCSS_String::Intern(const string   &s) {                             lwc_string *v=0; CHECK_EQ(lwc_error_ok, lwc_intern_string(s.data(), s.size(), &v)); return v; }
lwc_string *LibCSS_String::Intern(const String16 &s) { string S=String::ToUTF8(s); lwc_string *v=0; CHECK_EQ(lwc_error_ok, lwc_intern_string(S.data(), S.size(), &v)); return v; }
LFL::DOM::DOMString LibCSS_String::ToString(lwc_string *s) { return LFL::DOM::DOMString(lwc_string_data(s), lwc_string_length(s)); }
string LibCSS_String::ToUTF8String(lwc_string *s) { return String::ToUTF8(ToString(s)); }

int LibCSS_Unit::ToPrimitive(int n) {
  switch (n) {
    case CSS_UNIT_PX:  return LFL::DOM::CSSPrimitiveValue::CSS_PX;         break;
    case CSS_UNIT_EX:  return LFL::DOM::CSSPrimitiveValue::CSS_EXS;        break;
    case CSS_UNIT_EM:  return LFL::DOM::CSSPrimitiveValue::CSS_EMS;        break;
    case CSS_UNIT_IN:  return LFL::DOM::CSSPrimitiveValue::CSS_IN;         break;
    case CSS_UNIT_CM:  return LFL::DOM::CSSPrimitiveValue::CSS_CM;         break;
    case CSS_UNIT_MM:  return LFL::DOM::CSSPrimitiveValue::CSS_MM;         break;
    case CSS_UNIT_PT:  return LFL::DOM::CSSPrimitiveValue::CSS_PT;         break;
    case CSS_UNIT_PC:  return LFL::DOM::CSSPrimitiveValue::CSS_PC;         break;
    case CSS_UNIT_PCT: return LFL::DOM::CSSPrimitiveValue::CSS_PERCENTAGE; break;
  } return 0;
}

css_error LibCSS_StyleSheet::Resolve(void *pw, const char *base, lwc_string *rel, lwc_string **abs) { *abs = lwc_string_ref(rel); return CSS_OK; }
css_error LibCSS_StyleSheet::Import(void *pw, css_stylesheet *parent, lwc_string *url, uint64_t media) {
  LFL::DOM::Document *D = static_cast<StyleSheet*>(pw)->ownerDocument;
  if (D) D->parser->OpenStyleImport(LibCSS_String::ToUTF8String(url)); // XXX use parent
  INFO("libcss Import ", LibCSS_String::ToString(url));
  return CSS_INVALID; // CSS_OK;
}
css_error LibCSS_StyleSheet::Color(void *pw, lwc_string *name, css_color *color) {
  // INFO("libcss Color ", LibCSS_String::ToString(name)); // none, medium, ridge, solid, thin, threedface, no-repeat, center
  return CSS_INVALID; // CSS_OK;
}
css_error LibCSS_StyleSheet::Font(void *pw, lwc_string *name, css_system_font *system_font) {
  INFO("libcss Font ", LibCSS_String::ToString(name));
  return CSS_INVALID; // CSS_OK;
}

StyleSheet::~StyleSheet() {
  CHECK_EQ(CSS_OK, css_stylesheet_destroy(sheet));
  delete params;
}

StyleSheet::StyleSheet(LFL::DOM::Document *D, const char *U, const char *CS,
                       bool in_line, bool quirks, const char *Content) :
  ownerDocument(D), content_url(BlankNull(U)), character_set(BlankNull(CS)) {
  params = new css_stylesheet_params();
  params->params_version = CSS_STYLESHEET_PARAMS_VERSION_1;
  params->level = CSS_LEVEL_DEFAULT;
  params->title = NULL;
  params->url = content_url.c_str();
  params->charset = !character_set.empty() ? character_set.c_str() : 0;
  params->inline_style = in_line;
  params->allow_quirks = quirks;
  params->resolve = LibCSS_StyleSheet::Resolve;
  params->import = LibCSS_StyleSheet::Import;
  params->color = LibCSS_StyleSheet::Color;
  params->font = LibCSS_StyleSheet::Font;
  params->resolve_pw = params->import_pw = params->color_pw = params->font_pw = this;
  CHECK_EQ(CSS_OK, css_stylesheet_create(params, &sheet));
  if (Content) { Parse(Content); Done(); }
}

void StyleSheet::Parse(const char *content, int content_len) {
  css_error code = css_stylesheet_append_data(sheet, MakeUnsigned(content), content_len);
  CHECK(code == CSS_OK || code == CSS_NEEDDATA);
}

void StyleSheet::Done() { CHECK_EQ(CSS_OK, css_stylesheet_data_done(sheet)); }

StyleSheet *StyleSheet::Default() {
  static StyleSheet ret(0, "", "UTF-8", 0, 0, LFL::CSS::Default);
  return &ret;
}

bool ComputedStyle::Computed() const { return style; }
void ComputedStyle::Reset() {
  if (style) CHECK_EQ(CSS_OK, css_select_results_destroy(style));
  style = 0;
}

LFL::DOM::Display              ComputedStyle::Display             () const { return LFL::DOM::Display(css_computed_display(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, is_root)); }
LFL::DOM::Clear                ComputedStyle::Clear               () const { return LFL::DOM::Clear(css_computed_clear(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::Float                ComputedStyle::Float               () const { return LFL::DOM::Float(css_computed_float(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::Position             ComputedStyle::Position            () const { return LFL::DOM::Position(css_computed_position(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::Overflow             ComputedStyle::Overflow            () const { return LFL::DOM::Overflow(css_computed_overflow(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::TextAlign            ComputedStyle::TextAlign           () const { return LFL::DOM::TextAlign(css_computed_text_align(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::Direction            ComputedStyle::Direction           () const { return LFL::DOM::Direction(css_computed_direction(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::FontStyle            ComputedStyle::FontStyle           () const { return LFL::DOM::FontStyle(css_computed_font_style(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::EmptyCells           ComputedStyle::EmptyCells          () const { return LFL::DOM::EmptyCells(css_computed_empty_cells(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::FontWeight           ComputedStyle::FontWeight          () const { return LFL::DOM::FontWeight(css_computed_font_weight(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::Visibility           ComputedStyle::Visibility          () const { return LFL::DOM::Visibility(css_computed_visibility(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::WhiteSpace           ComputedStyle::WhiteSpace          () const { return LFL::DOM::WhiteSpace(css_computed_white_space(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::CaptionSide          ComputedStyle::CaptionSide         () const { return LFL::DOM::CaptionSide(css_computed_caption_side(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::FontVariant          ComputedStyle::FontVariant         () const { return LFL::DOM::FontVariant(css_computed_font_variant(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::TableLayout          ComputedStyle::TableLayout         () const { return LFL::DOM::TableLayout(css_computed_table_layout(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::UnicodeBidi          ComputedStyle::UnicodeBidi         () const { return LFL::DOM::UnicodeBidi(css_computed_unicode_bidi(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::BorderStyle          ComputedStyle::OutlineStyle        () const { return LFL::DOM::BorderStyle(css_computed_outline_style(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::TextTransform        ComputedStyle::TextTransform       () const { return LFL::DOM::TextTransform(css_computed_text_transform(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::TextDecoration       ComputedStyle::TextDecoration      () const { return LFL::DOM::TextDecoration(css_computed_text_decoration(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::BorderCollapse       ComputedStyle::BorderCollapse      () const { return LFL::DOM::BorderCollapse(css_computed_border_collapse(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::ListStyleType        ComputedStyle::ListStyleType       () const { return LFL::DOM::ListStyleType(css_computed_list_style_type(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::ListStylePosition    ComputedStyle::ListStylePosition   () const { return LFL::DOM::ListStylePosition(css_computed_list_style_position(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::PageBreak            ComputedStyle::PageBreakAfter      () const { return LFL::DOM::PageBreak(css_computed_page_break_after(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::PageBreak            ComputedStyle::PageBreakBefore     () const { return LFL::DOM::PageBreak(css_computed_page_break_before(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::PageBreak            ComputedStyle::PageBreakInside     () const { return LFL::DOM::PageBreak(css_computed_page_break_inside(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::BorderStyle          ComputedStyle::BorderTopStyle      () const { return LFL::DOM::BorderStyle(css_computed_border_top_style(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::BorderStyle          ComputedStyle::BorderLeftStyle     () const { return LFL::DOM::BorderStyle(css_computed_border_left_style(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::BorderStyle          ComputedStyle::BorderRightStyle    () const { return LFL::DOM::BorderStyle(css_computed_border_right_style(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::BorderStyle          ComputedStyle::BorderBottomStyle   () const { return LFL::DOM::BorderStyle(css_computed_border_bottom_style(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::BackgroundRepeat     ComputedStyle::BackgroundRepeat    () const { return LFL::DOM::BackgroundRepeat(css_computed_background_repeat(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::BackgroundAttachment ComputedStyle::BackgroundAttachment() const { return LFL::DOM::BackgroundAttachment(css_computed_background_attachment(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0)); }
LFL::DOM::RGBColor             ComputedStyle::BackgroundColor     () const { css_color c; return css_computed_background_color   (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &c) == CSS_BACKGROUND_COLOR_COLOR ? LFL::DOM::RGBColor(c) : LFL::DOM::RGBColor(); }
LFL::DOM::RGBColor             ComputedStyle::BorderBottomColor   () const { css_color c; return css_computed_border_bottom_color(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &c) == CSS_BORDER_COLOR_COLOR     ? LFL::DOM::RGBColor(c) : LFL::DOM::RGBColor(); }
LFL::DOM::RGBColor             ComputedStyle::BorderLeftColor     () const { css_color c; return css_computed_border_left_color  (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &c) == CSS_BORDER_COLOR_COLOR     ? LFL::DOM::RGBColor(c) : LFL::DOM::RGBColor(); }
LFL::DOM::RGBColor             ComputedStyle::BorderRightColor    () const { css_color c; return css_computed_border_right_color (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &c) == CSS_BORDER_COLOR_COLOR     ? LFL::DOM::RGBColor(c) : LFL::DOM::RGBColor(); }
LFL::DOM::RGBColor             ComputedStyle::BorderTopColor      () const { css_color c; return css_computed_border_top_color   (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &c) == CSS_BORDER_COLOR_COLOR     ? LFL::DOM::RGBColor(c) : LFL::DOM::RGBColor(); }
LFL::DOM::RGBColor             ComputedStyle::OutlineColor        () const { css_color c; return css_computed_outline_color      (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &c) == CSS_OUTLINE_COLOR_COLOR    ? LFL::DOM::RGBColor(c) : LFL::DOM::RGBColor(); }
LFL::DOM::RGBColor             ComputedStyle::Color               () const { css_color c; return css_computed_color              (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &c) == CSS_COLOR_COLOR            ? LFL::DOM::RGBColor(c) : LFL::DOM::RGBColor(); }
LFL::DOM::Cursor               ComputedStyle::Cursor              () const { lwc_string **n=0; LFL::DOM::Cursor     ret(css_computed_cursor     (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &n)); for (; n && *n; n++) ret.url .push_back(LibCSS_String::ToUTF8String(*n)); return ret; }
LFL::DOM::FontFamily           ComputedStyle::FontFamily          () const { lwc_string **n=0; LFL::DOM::FontFamily ret(css_computed_font_family(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &n)); for (; n && *n; n++) ret.name.push_back(LibCSS_String::ToUTF8String(*n)); return ret; }
LFL::DOM::CSSStringValue       ComputedStyle::BackgroundImage     () const { lwc_string *v=0; css_computed_background_image (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &v); return v ? LFL::DOM::CSSStringValue(LibCSS_String::ToString(v)) : LFL::DOM::CSSStringValue(); }
LFL::DOM::CSSNumericValuePair  ComputedStyle::BackgroundPosition  () const { css_fixed x,y; css_unit xu,yu; return css_computed_background_position(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &x, &xu, &y, &yu) == CSS_BACKGROUND_POSITION_SET ? LFL::DOM::CSSNumericValuePair(FIXTOFLT(x), LibCSS_Unit::ToPrimitive(xu), 0, FIXTOFLT(y), LibCSS_Unit::ToPrimitive(yu), LFL::DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight) : LFL::DOM::CSSNumericValuePair(); }
LFL::DOM::Rect                 ComputedStyle::Clip                () const { css_computed_clip_rect r;      return css_computed_clip             (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &r)      == CSS_CLIP_RECT       ? LFL::DOM::Rect(r.top_auto, FIXTOFLT(r.top), LibCSS_Unit::ToPrimitive(r.tunit), r.right_auto, FIXTOFLT(r.right), LibCSS_Unit::ToPrimitive(r.runit), r.bottom_auto, FIXTOFLT(r.bottom), LibCSS_Unit::ToPrimitive(r.bunit), r.left_auto, FIXTOFLT(r.left), LibCSS_Unit::ToPrimitive(r.lunit)) : LFL::DOM::Rect(); }
LFL::DOM::CSSNumericValue      ComputedStyle::Opacity             () const { css_fixed s;                   return css_computed_opacity          (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s)      == CSS_OPACITY_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LFL::DOM::CSSPrimitiveValue::CSS_NUMBER) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::Orphans             () const { css_fixed s;                   return css_computed_orphans          (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s)      == CSS_ORPHANS_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LFL::DOM::CSSPrimitiveValue::CSS_NUMBER) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::Widows              () const { css_fixed s;                   return css_computed_widows           (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s)      == CSS_WIDOWS_SET      ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LFL::DOM::CSSPrimitiveValue::CSS_NUMBER) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::MinHeight           () const { css_fixed s; css_unit su;      return css_computed_min_height       (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su) == CSS_MIN_HEIGHT_SET  ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::MinWidth            () const { css_fixed s; css_unit su;      return css_computed_min_width        (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su) == CSS_MIN_WIDTH_SET   ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::PaddingTop          () const { css_fixed s; css_unit su;      return css_computed_padding_top      (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su) == CSS_PADDING_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::PaddingRight        () const { css_fixed s; css_unit su;      return css_computed_padding_right    (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su) == CSS_PADDING_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::PaddingBottom       () const { css_fixed s; css_unit su;      return css_computed_padding_bottom   (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su) == CSS_PADDING_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::PaddingLeft         () const { css_fixed s; css_unit su;      return css_computed_padding_left     (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su) == CSS_PADDING_SET     ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::TextIndent          () const { css_fixed s; css_unit su;      return css_computed_text_indent      (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su) == CSS_TEXT_INDENT_SET ? LFL::DOM::CSSNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Top                 () const { css_fixed s; css_unit su; int ret = css_computed_top                (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_TOP_SET             ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_TOP_AUTO); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Bottom              () const { css_fixed s; css_unit su; int ret = css_computed_bottom             (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_BOTTOM_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_BOTTOM_AUTO); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Left                () const { css_fixed s; css_unit su; int ret = css_computed_left               (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_LEFT_SET            ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_LEFT_AUTO); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Right               () const { css_fixed s; css_unit su; int ret = css_computed_right              (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_RIGHT_SET           ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_RIGHT_AUTO); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Width               () const { css_fixed s; css_unit su; int ret = css_computed_width              (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_WIDTH_SET           ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_WIDTH_AUTO); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Height              () const { css_fixed s; css_unit su; int ret = css_computed_height             (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_HEIGHT_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_HEIGHT_AUTO); }
LFL::DOM::FontSize             ComputedStyle::FontSize            () const { css_fixed s; css_unit su; int ret = css_computed_font_size          (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_FONT_SIZE_DIMENSION ? LFL::DOM::FontSize           (FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::FontSize(ret); }
LFL::DOM::VerticalAlign        ComputedStyle::VerticalAlign       () const { css_fixed s; css_unit su; int ret = css_computed_vertical_align     (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_VERTICAL_ALIGN_SET  ? LFL::DOM::VerticalAlign      (FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::VerticalAlign(ret); }
LFL::DOM::BorderWidth          ComputedStyle::BorderTopWidth      () const { css_fixed w; css_unit wu; int ret = css_computed_border_top_width   (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
LFL::DOM::BorderWidth          ComputedStyle::BorderRightWidth    () const { css_fixed w; css_unit wu; int ret = css_computed_border_right_width (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
LFL::DOM::BorderWidth          ComputedStyle::BorderBottomWidth   () const { css_fixed w; css_unit wu; int ret = css_computed_border_bottom_width(style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
LFL::DOM::BorderWidth          ComputedStyle::BorderLeftWidth     () const { css_fixed w; css_unit wu; int ret = css_computed_border_left_width  (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
LFL::DOM::BorderWidth          ComputedStyle::OutlineWidth        () const { css_fixed w; css_unit wu; int ret = css_computed_outline_width      (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &w, &wu); return ret == CSS_BORDER_WIDTH_WIDTH  ? LFL::DOM::BorderWidth        (FIXTOFLT(w), LibCSS_Unit::ToPrimitive(wu)) : LFL::DOM::BorderWidth(ret); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::MarginBottom        () const { css_fixed s; css_unit su; int ret = css_computed_margin_bottom      (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_MARGIN_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_MARGIN_AUTO); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::MarginLeft          () const { css_fixed s; css_unit su; int ret = css_computed_margin_left        (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_MARGIN_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_MARGIN_AUTO); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::MarginRight         () const { css_fixed s; css_unit su; int ret = css_computed_margin_right       (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_MARGIN_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_MARGIN_AUTO); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::MarginTop           () const { css_fixed s; css_unit su; int ret = css_computed_margin_top         (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_MARGIN_SET          ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSAutoNumericValue(ret == CSS_MARGIN_AUTO); }
LFL::DOM::CSSNoneNumericValue  ComputedStyle::MaxHeight           () const { css_fixed s; css_unit su; int ret = css_computed_max_height         (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_MAX_HEIGHT_SET      ? LFL::DOM::CSSNoneNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNoneNumericValue(ret == CSS_MAX_HEIGHT_NONE); }
LFL::DOM::CSSNoneNumericValue  ComputedStyle::MaxWidth            () const { css_fixed s; css_unit su; int ret = css_computed_max_width          (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_MAX_WIDTH_SET       ? LFL::DOM::CSSNoneNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su)) : LFL::DOM::CSSNoneNumericValue(ret == CSS_MAX_WIDTH_NONE); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::ZIndex              () const { css_fixed s;              int ret = css_computed_z_index            (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s);      return ret == CSS_Z_INDEX_SET         ? LFL::DOM::CSSAutoNumericValue(FIXTOFLT(s), LFL::DOM::CSSPrimitiveValue::CSS_NUMBER) : LFL::DOM::CSSAutoNumericValue(ret == CSS_Z_INDEX_AUTO); }
LFL::DOM::CSSNormNumericValue  ComputedStyle::WordSpacing         () const { css_fixed s; css_unit su; int ret = css_computed_word_spacing       (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_WORD_SPACING_SET    ? LFL::DOM::CSSNormNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su), LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth) : LFL::DOM::CSSNormNumericValue(ret == CSS_WORD_SPACING_NORMAL,   LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth); }
LFL::DOM::CSSNormNumericValue  ComputedStyle::LetterSpacing       () const { css_fixed s; css_unit su; int ret = css_computed_letter_spacing     (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); return ret == CSS_LETTER_SPACING_SET  ? LFL::DOM::CSSNormNumericValue(FIXTOFLT(s), LibCSS_Unit::ToPrimitive(su), LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth) : LFL::DOM::CSSNormNumericValue(ret == CSS_LETTER_SPACING_NORMAL, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth); }
LFL::DOM::CSSNormNumericValue  ComputedStyle::LineHeight          () const { css_fixed s; css_unit su; int ret = css_computed_line_height        (style ? style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0, &s, &su); bool dim = ret == CSS_LINE_HEIGHT_DIMENSION; return (ret == CSS_LINE_HEIGHT_NUMBER || dim) ? LFL::DOM::CSSNormNumericValue(FIXTOFLT(s), dim ? LibCSS_Unit::ToPrimitive(su) : LFL::DOM::CSSPrimitiveValue::CSS_NUMBER, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontHeight) : LFL::DOM::CSSNormNumericValue(ret == CSS_LINE_HEIGHT_NORMAL, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontHeight); }

const LFL::DOM::CSSValue *ComputedStyle::getPropertyCSSValue(const DOMString &n) const { return 0; };
DOMString ComputedStyle::getPropertyPriority(const DOMString &n) const { return ""; }
DOMString ComputedStyle::removeProperty(const DOMString &n) { ERROR("not implemented"); return ""; }
int ComputedStyle::length() const { return Singleton<LFL::CSS::PropertyNames>::Get()->size(); };
DOMString ComputedStyle::item(int index) const { CHECK_RANGE(index, 0, length()); return (*Singleton<LFL::CSS::PropertyNames>::Get())[index]; }

DOMString ComputedStyle::getPropertyValue(const DOMString &k) const {
  string name = String::ToUTF8(k);
  if (0) {}
#undef  XX
#define XX(n) else if (name == #n)
#undef  YY
#define YY(n)   return n().cssText();
#undef  ZZ
#define ZZ(n)
#include "core/web/css_properties.h"
  return "";
}

void ComputedStyle::setProperty(const DOMString &k, const DOMString &v, const DOMString &priority) {
  string name = String::ToUTF8(k), val = StrCat(String::ToUTF8(v), ";");
  INFO("ComputedStyle.setProperty(", name, ", ", val, ")"); 
  if (0) {}
#undef  XX
#define XX(n) else if (name == #n)
#undef  YY
#define YY(n)
#undef  ZZ
#define ZZ(n)   override_style = StrCat("\n", UpdateKVLine(override_style, #n, val, ':')); 
#include "core/web/css_properties.h"
  else return;
  if (auto html = node->ownerDocument->documentElement()) html->SetStyleDirty();
  node->ClearComputedInlineStyle();
}

css_select_handler *LibCSS_StyleContext::SelectHandler() {
  static css_select_handler select_handler = {
    CSS_SELECT_HANDLER_VERSION_1, NodeName, NodeClasses, NodeId, NamedAncestorNode, NamedParentNode, NamedSiblingNode,
    NamedGenericSiblingNode, ParentNode, SiblingNode, NodeHasName, NodeHasClass, NodeHasId, NodeHasAttribute,
    NodeHasAttributeEqual, NodeHasAttributeDashmatch, NodeHasAttributeIncludes, NodeHasAttributePrefix,
    NodeHasAttributeSuffix, NodeHasAttributeSubstring, NodeIsRoot, NodeCountSiblings, NodeIsEmpty, NodeIsLink,
    NodeIsVisited, NodeIsHover, NodeIsActive, NodeIsFocus, NodeIsEnabled, NodeIsDisabled, NodeIsChecked,
    NodeIsTarget, NodeIsLang, NodePresentationalHint, UADefaultForProperty, ComputeFontSize, SetLibCSSNodeData,
    GetLibCSSNodeData
  };
  return &select_handler;
}

css_error LibCSS_StyleContext::NodeName(void *pw, void *n, css_qname *qname) {
  qname->name = LibCSS_String::Intern(static_cast<LFL::DOM::Node*>(n)->nodeName());
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeClasses(void *pw, void *n, lwc_string ***classes_out, uint32_t *n_classes) {
  *n_classes = 0;
  *classes_out = NULL;
  DOM::Node *node = static_cast<DOM::Node*>(n), *attr = 0;
  if (!(attr = node->getAttributeNode("class"))) return CSS_OK;

  vector<DOM::DOMString> classes;
  Split(attr->nodeValue(), isspace, &classes);
  if (!classes.size()) return CSS_OK;

  node->render->style.class_cache.resize(classes.size());
  *classes_out = &node->render->style.class_cache[0];
  *n_classes = classes.size();
  for (int i=0; i<classes.size(); i++) (*classes_out)[i] = LibCSS_String::Intern(classes[i]);
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeId(void *pw, void *n, lwc_string **id) {
  *id = NULL;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n), *attr = 0;
  if ((attr = node->getAttributeNode("id"))) *id = LibCSS_String::Intern(attr->nodeValue());
  return CSS_OK;
}

css_error LibCSS_StyleContext::NamedAncestorNode(void *pw, void *n, const css_qname *qname, void **ancestor) {
  *ancestor = NULL;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n);
  LFL::DOM::DOMString query = LibCSS_String::ToString(qname->name);
  for (node = node->parentNode; node; node = node->parentNode) {
    if (node->nodeName() == query && node->AsElement()) { *ancestor = node; break; }
  }
  return CSS_OK;
}

css_error LibCSS_StyleContext::NamedParentNode(void *pw, void *n, const css_qname *qname, void **parent) {
  *parent = NULL;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n);
  for (node = node->parentNode; node && !node->AsElement(); node = node->parentNode) {}
  if (node && node->nodeName() == LibCSS_String::ToString(qname->name)) *parent = node;
  return CSS_OK;
}

css_error LibCSS_StyleContext::NamedSiblingNode(void *pw, void *n, const css_qname *qname, void **sibling) {
  *sibling = NULL;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n);
  for (node = node->previousSibling(); node && !node->AsElement(); node = node->previousSibling()) {}
  if (node && node->nodeName() == LibCSS_String::ToString(qname->name)) *sibling = node;
  return CSS_OK;
}

css_error LibCSS_StyleContext::NamedGenericSiblingNode(void *pw, void *n, const css_qname *qname, void **sibling) {
  *sibling = NULL;
  LFL::DOM::DOMString query = LibCSS_String::ToString(qname->name);
  for (LFL::DOM::Node *i = static_cast<LFL::DOM::Node*>(n)->previousSibling(); i; i = i->previousSibling()) {
    if (i->nodeName() == query && i->AsElement()) { *sibling = i; break; }
  }
  return CSS_OK;
}

css_error LibCSS_StyleContext::ParentNode(void *pw, void *n, void **parent) {
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n);
  for (node = node->parentNode; node && !node->AsElement(); node = node->parentNode) {}
  *parent = node;
  return CSS_OK;
}

css_error LibCSS_StyleContext::SiblingNode(void *pw, void *n, void **sibling) {
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n);
  for (node = node->previousSibling(); node && !node->AsElement(); node = node->previousSibling()) {}
  *sibling = node;
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasName(void *pw, void *n, const css_qname *qname, bool *match) {
  *match = LibCSS_String::ToString(qname->name) == ((LFL::DOM::Node*)n)->nodeName();
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasClass(void *pw, void *n, lwc_string *name, bool *match) {
  *match = false;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n), *attr = 0;
  if (!(attr = node->getAttributeNode("class"))) return CSS_OK;

  vector<LFL::DOM::DOMString> classes;
  Split(attr->nodeValue(), isspace, &classes);
  if (!classes.size()) return CSS_OK;

  LFL::DOM::DOMString query = LibCSS_String::ToString(name);
  for (int i=0; i<classes.size(); i++) if (classes[i] == query) { *match = true; break; }
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasId(void *pw, void *n, lwc_string *name, bool *match) {
  *match = false;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n), *attr = 0;
  if ((attr = node->getAttributeNode("id"))) *match = attr->nodeValue() == LibCSS_String::ToString(name);
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasAttribute(void *pw, void *n, const css_qname *qname, bool *match) {
  *match = static_cast<LFL::DOM::Node*>(n)->getAttributeNode(LibCSS_String::ToString(qname->name));
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasAttributeEqual(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
  *match = false;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n), *attr = 0;
  if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name))))
    *match = attr->nodeValue() == LibCSS_String::ToString(value);
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasAttributeDashmatch(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
  *match = false;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n), *attr = 0;
  if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name)))) {
    LFL::DOM::DOMString attr_val = attr->nodeValue(), seek_val = LibCSS_String::ToString(value);
    *match = (seek_val == attr_val) ||
      (seek_val == attr_val.substr(0, seek_val.size()) && attr_val[seek_val.size()-1] == '-');
  } 
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasAttributeIncludes(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
  *match = false;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n), *attr = 0;
  if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name)))) {
    LFL::DOM::DOMString attr_val = attr->nodeValue(), seek_val = LibCSS_String::ToString(value);
    *match = seek_val == attr_val.substr(0, attr_val.find(' '));
  } 
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasAttributePrefix(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
  *match = false;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n), *attr = 0;
  if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name))))
    *match = PrefixMatch(attr->nodeValue(), LibCSS_String::ToString(value));
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasAttributeSuffix(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
  *match = false;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n), *attr = 0;
  if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name))))
    *match = SuffixMatch(attr->nodeValue(), LibCSS_String::ToString(value));
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeHasAttributeSubstring(void *pw, void *n, const css_qname *qname, lwc_string *value, bool *match) {
  *match = false;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n), *attr = 0;
  if ((attr = node->getAttributeNode(LibCSS_String::ToString(qname->name))))
    *match = attr->nodeValue().find(LibCSS_String::ToString(value)) != string::npos;
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeIsRoot(void *pw, void *n, bool *match) {
  LFL::DOM::Node *node = (LFL::DOM::Node*)n;
  *match = !node->parentNode || node->parentNode->AsDocument();
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeCountSiblings(void *pw, void *n, bool same_name, bool after, int32_t *count) {
  *count = 0;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n);
  LFL::DOM::DOMString query = same_name ? node->nodeName() : "";
  while ((node = after ? node->nextSibling() : node->previousSibling())) {
    if (!node->AsElement() || (same_name && node->nodeName() != query)) continue;
    (*count)++;
  }
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeIsEmpty(void *pw, void *n, bool *match) {
  *match = true;
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n);
  for (int i = 0; i < node->childNodes.length(); i++)
    if (node->childNodes.item(i)->AsElement() || node->childNodes.item(i)->AsText()) { *match = false; break; }
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeIsLink(void *pw, void *n, bool *match) {
  LFL::DOM::Node *node = static_cast<LFL::DOM::Node*>(n);
  *match = node->htmlElementType == LFL::DOM::HTML_ANCHOR_ELEMENT && node->getAttributeNode("href");
  return CSS_OK;
}

css_error LibCSS_StyleContext::NodeIsVisited (void *pw, void *n, bool *match) { *match = false; return CSS_OK; }
css_error LibCSS_StyleContext::NodeIsHover   (void *pw, void *n, bool *match) { *match = false; return CSS_OK; }
css_error LibCSS_StyleContext::NodeIsActive  (void *pw, void *n, bool *match) { *match = false; return CSS_OK; }
css_error LibCSS_StyleContext::NodeIsFocus   (void *pw, void *n, bool *match) { *match = false; return CSS_OK; }
css_error LibCSS_StyleContext::NodeIsEnabled (void *pw, void *n, bool *match) { *match = false; return CSS_OK; }
css_error LibCSS_StyleContext::NodeIsDisabled(void *pw, void *n, bool *match) { *match = false; return CSS_OK; }
css_error LibCSS_StyleContext::NodeIsChecked (void *pw, void *n, bool *match) { *match = false; return CSS_OK; }
css_error LibCSS_StyleContext::NodeIsTarget  (void *pw, void *n, bool *match) { *match = false; return CSS_OK; }
css_error LibCSS_StyleContext::NodeIsLang    (void *pw, void *n, lwc_string *lang, bool *match) { *match = false; return CSS_OK; }
css_error LibCSS_StyleContext::NodePresentationalHint(void *pw, void *node, uint32_t property, css_hint *hint) { return CSS_PROPERTY_NOT_SET; }
css_error LibCSS_StyleContext::UADefaultForProperty(void *pw, uint32_t property, css_hint *hint) {
  if      (property == CSS_PROP_COLOR)        { hint->data.color = 0xFF000000; hint->status = CSS_COLOR_COLOR; }
  else if (property == CSS_PROP_FONT_FAMILY)  { hint->data.strings = NULL;     hint->status = CSS_FONT_FAMILY_SANS_SERIF; }
  else if (property == CSS_PROP_QUOTES)       { hint->data.strings = NULL;     hint->status = CSS_QUOTES_NONE; }
  else if (property == CSS_PROP_VOICE_FAMILY) { hint->data.strings = NULL;     hint->status = 0; }
  else return CSS_INVALID;
  return CSS_OK;
}

css_error LibCSS_StyleContext::ComputeFontSize(void *pw, const css_hint *parent, css_hint *size) {
  static css_hint_length sizes[] = {
    { FLTTOFIX(6.75), CSS_UNIT_PT },
    { FLTTOFIX(7.50), CSS_UNIT_PT },
    { FLTTOFIX(9.75), CSS_UNIT_PT },
    { FLTTOFIX(12.0), CSS_UNIT_PT },
    { FLTTOFIX(13.5), CSS_UNIT_PT },
    { FLTTOFIX(18.0), CSS_UNIT_PT },
    { FLTTOFIX(24.0), CSS_UNIT_PT }
  };
  const css_hint_length *parent_size;
  /* Grab parent size, defaulting to medium if none */
  if (parent == NULL) {
    parent_size = &sizes[CSS_FONT_SIZE_MEDIUM - 1];
  } else {
    CHECK_EQ(CSS_FONT_SIZE_DIMENSION, parent->status);
    CHECK_NE(CSS_UNIT_EM, parent->data.length.unit);
    CHECK_NE(CSS_UNIT_EX, parent->data.length.unit);
    parent_size = &parent->data.length;
  }
  CHECK_NE(size->status, CSS_FONT_SIZE_INHERIT);
  if (size->status < CSS_FONT_SIZE_LARGER) {
    /* Keyword -- simple */
    size->data.length = sizes[size->status - 1];
  } else if (size->status == CSS_FONT_SIZE_LARGER) {
    /** \todo Step within table, if appropriate */
    size->data.length.value = FMUL(parent_size->value, FLTTOFIX(1.2));
    size->data.length.unit = parent_size->unit;
  } else if (size->status == CSS_FONT_SIZE_SMALLER) {
    /** \todo Step within table, if appropriate */
    size->data.length.value = FMUL(parent_size->value, FLTTOFIX(1.2));
    size->data.length.unit = parent_size->unit;
  } else if (size->data.length.unit == CSS_UNIT_EM || size->data.length.unit == CSS_UNIT_EX) {
    size->data.length.value = FMUL(size->data.length.value, parent_size->value);
    if (size->data.length.unit == CSS_UNIT_EX) size->data.length.value = FMUL(size->data.length.value, FLTTOFIX(0.6));
    size->data.length.unit = parent_size->unit;
  } else if (size->data.length.unit == CSS_UNIT_PCT) {
    size->data.length.value = FDIV(FMUL(size->data.length.value, parent_size->value), FLTTOFIX(100));
    size->data.length.unit  = parent_size->unit;
  }
  size->status = CSS_FONT_SIZE_DIMENSION;
  return CSS_OK;
}

css_error LibCSS_StyleContext::SetLibCSSNodeData(void *pw, void *n, void *libcss_node_data) {
  /* Since we're not storing it, ensure node data gets deleted */
  css_libcss_node_data_handler(SelectHandler(), CSS_NODE_DELETED, pw, n, NULL, libcss_node_data);
  return CSS_OK;
}

css_error LibCSS_StyleContext::GetLibCSSNodeData(void *pw, void *n, void **libcss_node_data) {
  *libcss_node_data = NULL;
  return CSS_OK;
}

StyleContext::StyleContext(LFL::DOM::HTMLDocument *D) : doc(D) { CHECK_EQ(CSS_OK, css_select_ctx_create(&select_ctx)); }
StyleContext::~StyleContext() { CHECK_EQ(CSS_OK, css_select_ctx_destroy(select_ctx)); }
void StyleContext::AppendSheet(StyleSheet *ss) { CHECK_EQ(CSS_OK, css_select_ctx_append_sheet(select_ctx, ss->sheet, CSS_ORIGIN_AUTHOR, CSS_MEDIA_ALL)); }

void StyleContext::Match(ComputedStyle *out, LFL::DOM::Node *node, const ComputedStyle *parent_sheet, StyleSheet *inline_style) {
  out->Reset();
  CHECK_EQ(CSS_OK, css_select_style(select_ctx, node, CSS_MEDIA_SCREEN,
                                    inline_style ? inline_style->sheet : 0,
                                    LibCSS_StyleContext::SelectHandler(), this, &out->style));


  auto os = out->style ? out->style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0;
  lwc_string **n;
  css_fixed s;
  css_unit su;
  css_color c;

  out->color_not_inherited   = css_computed_color           (os, &c) == CSS_COLOR_COLOR;
  out->bgcolor_not_inherited = css_computed_background_color(os, &c) == CSS_BACKGROUND_COLOR_COLOR;
  out->font_not_inherited = out->color_not_inherited || out->bgcolor_not_inherited ||
    (css_computed_font_family (os, &n)      != CSS_FONT_FAMILY_INHERIT) ||
    (css_computed_font_size   (os, &s, &su) != CSS_FONT_SIZE_INHERIT)   ||
    (css_computed_font_style  (os)          != CSS_FONT_STYLE_INHERIT)  ||
    (css_computed_font_weight (os)          != CSS_FONT_WEIGHT_INHERIT) ||
    (css_computed_font_variant(os)          != CSS_FONT_VARIANT_INHERIT);

  if (parent_sheet) {
    auto ps = parent_sheet->style ? parent_sheet->style->styles[CSS_PSEUDO_ELEMENT_NONE] : 0;
    CHECK_EQ(CSS_OK, css_computed_style_compose(ps, os, LibCSS_StyleContext::ComputeFontSize, NULL, os));
  }
}

int StyleContext::FontFaces(const string &face_name, vector<LFL::DOM::FontFace> *out) {
  out->clear();
  lwc_string *n=0; css_select_font_faces_results *v=0; uint32_t l;
  int rc = css_select_font_faces(select_ctx, CSS_MEDIA_SCREEN, LibCSS_String::Intern(face_name), &v);
  for (int i=0; rc == CSS_OK && v && i<v->n_font_faces; i++) {
    LFL::DOM::FontFace ff;
    ff.style  = LFL::DOM::FontStyle (css_font_face_font_style( v->font_faces[i]));
    ff.weight = LFL::DOM::FontWeight(css_font_face_font_weight(v->font_faces[i]));
    n=0; css_font_face_get_font_family(v->font_faces[i], &n); if (n) ff.family.name.push_back(LibCSS_String::ToUTF8String(n));
    l=0; css_font_face_count_srcs(v->font_faces[i], &l);
    for (int j=0; j<l; j++) {
      const css_font_face_src *src=0; css_font_face_get_src(v->font_faces[i], j, &src);
      if (css_font_face_src_location_type(src) == CSS_FONT_FACE_LOCATION_TYPE_UNSPECIFIED) continue;
      n=0; if (src) css_font_face_src_get_location(src, &n);
      if (n) ff.source.push_back(LibCSS_String::ToUTF8String(n));
    }
    out->push_back(ff);
  }
  if (v) css_select_font_faces_results_destroy(v);
  return out->size();
}

}; // namespace LFL
