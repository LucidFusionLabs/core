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

#include "core/app/gl/view.h"
#include "core/web/dom.h"
#include "core/web/css.h"

namespace LFL {
namespace DOM {
const int BackgroundAttachment::Fixed  = 1;
const int BackgroundAttachment::Scroll = 2;

const int BackgroundRepeat::RepeatX  = 1;
const int BackgroundRepeat::RepeatY  = 2;
const int BackgroundRepeat::Repeat   = 3;
const int BackgroundRepeat::NoRepeat = 4;

const int BorderCollapse::Separate = 1;
const int BorderCollapse::Collapse = 2;

const int BorderWidth::Thin   = 1;
const int BorderWidth::Medium = 2;
const int BorderWidth::Thick  = 3;

const int BorderStyle::None   = 1;
const int BorderStyle::Hidden = 2;
const int BorderStyle::Dotted = 3;
const int BorderStyle::Dashed = 4;
const int BorderStyle::Solid  = 5;
const int BorderStyle::Double = 6;
const int BorderStyle::Groove = 7;
const int BorderStyle::Ridge  = 8;
const int BorderStyle::Inset  = 9;
const int BorderStyle::Outset = 10;

const int CaptionSide::Top    = 1;
const int CaptionSide::Bottom = 2;

const int Clear::None  = 1;
const int Clear::Left  = 2;
const int Clear::Right = 3;
const int Clear::Both  = 4;

const int Cursor::Auto      = 1;
const int Cursor::Default   = 2;
const int Cursor::NEResize  = 3;
const int Cursor::WResize   = 4;
const int Cursor::Move      = 5;
const int Cursor::Pointer   = 6;
const int Cursor::EResize   = 7;
const int Cursor::NWResize  = 8;
const int Cursor::Text      = 9;
const int Cursor::Crosshair = 10;
const int Cursor::SEResize  = 11;
const int Cursor::Wait      = 12;
const int Cursor::Progress  = 13;
const int Cursor::SResize   = 14;
const int Cursor::Help      = 15;
const int Cursor::NResize   = 16;
const int Cursor::SWResize  = 17;

const int Direction::LTR = 1;
const int Direction::RTL = 2;

const int Display::None        = 1;
const int Display::Row         = 2;
const int Display::HeaderGroup = 3;
const int Display::Inline      = 4;
const int Display::Cell        = 5;
const int Display::FooterGroup = 6;
const int Display::Block       = 7;
const int Display::Column      = 8;
const int Display::InlineBlock = 9;
const int Display::RunIn       = 10;
const int Display::Caption     = 11;
const int Display::InlineTable = 12;
const int Display::ListItem    = 13;
const int Display::RowGroup    = 14;
const int Display::Table       = 15;
const int Display::ColGroup    = 16;

const int EmptyCells::Show = 1;
const int EmptyCells::Hide = 2;

const int Float::Left  = 1;
const int Float::Right = 2;
const int Float::None  = 3; 

const int FontFamily::Serif     = 1;
const int FontFamily::SansSerif = 2;
const int FontFamily::Cursive   = 3;
const int FontFamily::Fantasy   = 4;
const int FontFamily::Monospace = 5;

const int FontSize::XXSmall = 1;
const int FontSize::XLarge  = 2;
const int FontSize::XSmall  = 3;
const int FontSize::XXLarge = 4;
const int FontSize::Small   = 5;
const int FontSize::Larger  = 6;
const int FontSize::Medium  = 7;
const int FontSize::Smaller = 8;
const int FontSize::Large   = 9;

const int FontStyle::Normal  = 1;
const int FontStyle::Italic  = 2;
const int FontStyle::Oblique = 3;

const int FontVariant::Normal    = 1;
const int FontVariant::SmallCaps = 2;

const int FontWeight::Normal  = 1;
const int FontWeight::_400    = 2;
const int FontWeight::Bold    = 3;
const int FontWeight::_500    = 4;
const int FontWeight::Bolder  = 5;
const int FontWeight::_600    = 6;
const int FontWeight::Lighter = 7;
const int FontWeight::_700    = 8;
const int FontWeight::_100    = 9;
const int FontWeight::_800    = 10;
const int FontWeight::_200    = 11;
const int FontWeight::_900    = 12;
const int FontWeight::_300    = 13;

const int ListStylePosition::Inside  = 1;
const int ListStylePosition::Outside = 2;

const int ListStyleType::None               = 1;
const int ListStyleType::LowerLatin         = 2;
const int ListStyleType::Disc               = 3;
const int ListStyleType::UpperLatin         = 4;
const int ListStyleType::Circle             = 5;
const int ListStyleType::Armenian           = 6;
const int ListStyleType::Square             = 7;
const int ListStyleType::Georgian           = 8;
const int ListStyleType::Decimal            = 9;
const int ListStyleType::LowerAlpha         = 10;
const int ListStyleType::LowerRoman         = 11;
const int ListStyleType::UpperAlpha         = 12;
const int ListStyleType::UpperRoman         = 13;
const int ListStyleType::DecimalLeadingZero = 14;
const int ListStyleType::LowerGreek         = 15;

const int Overflow::Visible = 1;
const int Overflow::Scroll  = 2;
const int Overflow::Hidden  = 3;
const int Overflow::Auto    = 4;

const int PageBreak::Auto        = 1;
const int PageBreak::Page        = 2;
const int PageBreak::Avoid       = 3;
const int PageBreak::Column      = 4;
const int PageBreak::Always      = 5;
const int PageBreak::AvoidPage   = 6;
const int PageBreak::Left        = 7;
const int PageBreak::AvoidColumn = 8;
const int PageBreak::Right       = 9;

const int Position::Static   = 1;
const int Position::Relative = 2;
const int Position::Absolute = 3;
const int Position::Fixed    = 4;

const int TableLayout::Auto  = 1;
const int TableLayout::Fixed = 2;

const int TextAlign::Left    = 1;
const int TextAlign::Center  = 2;
const int TextAlign::Right   = 3;
const int TextAlign::Justify = 4;

const int TextDecoration::None        = 1;
const int TextDecoration::Blink       = (1<<1);
const int TextDecoration::LineThrough = (1<<2);
const int TextDecoration::Overline    = (1<<3);
const int TextDecoration::Underline   = (1<<4);

const int TextTransform::Capitalize = 1;
const int TextTransform::Lowercase  = 2;
const int TextTransform::Uppercase  = 3;
const int TextTransform::None       = 4;

const int UnicodeBidi::Normal   = 1;
const int UnicodeBidi::Embed    = 2;
const int UnicodeBidi::Override = 3;

const int VerticalAlign::Baseline   = 1;
const int VerticalAlign::TextTop    = 2;
const int VerticalAlign::Sub        = 3;
const int VerticalAlign::Middle     = 4;
const int VerticalAlign::Super      = 5;
const int VerticalAlign::Bottom     = 6;
const int VerticalAlign::Top        = 7;
const int VerticalAlign::TextBottom = 8;

const int Visibility::Visible  = 1;
const int Visibility::Hidden   = 2;
const int Visibility::Collapse = 3;

const int WhiteSpace::Normal  = 1;
const int WhiteSpace::PreWrap = 2;
const int WhiteSpace::Pre     = 3;
const int WhiteSpace::PreLine = 4;
const int WhiteSpace::Nowrap  = 5;

#include "core/web/css_common.h"
}; // namespace DOM

StyleSheet *StyleSheet::Default() { static StyleSheet ret(0); return &ret; }
StyleSheet::~StyleSheet() {}
StyleSheet::StyleSheet(LFL::DOM::Document *D, const char *U, const char *T, bool in_line, bool quirks, const char *Content) : ownerDocument(D) {}
void StyleSheet::Parse(const char *content, int content_len) {}
void StyleSheet::Done() {}

void ComputedStyle::Reset() {}
bool ComputedStyle::IsLink() const {
  for (LFL::DOM::Node *n = node ? node->parentNode : 0; n; n = n->parentNode)
    if (n->htmlElementType == LFL::DOM::HTML_ANCHOR_ELEMENT) return true;
  return false;
}

LFL::DOM::Display ComputedStyle::Display() const {
  switch (node->htmlElementType) {
    case LFL::DOM::HTML_BLOCK_QUOTE_ELEMENT:    case LFL::DOM::HTML_HR_ELEMENT:
    case LFL::DOM::HTML_CANVAS_ELEMENT:         case LFL::DOM::HTML_MENU_ELEMENT:
    case LFL::DOM::HTML_DIV_ELEMENT:            case LFL::DOM::HTML_O_LIST_ELEMENT:
    case LFL::DOM::HTML_D_LIST_ELEMENT:         case LFL::DOM::HTML_PARAGRAPH_ELEMENT:
    case LFL::DOM::HTML_FIELD_SET_ELEMENT:      case LFL::DOM::HTML_PRE_ELEMENT:
    case LFL::DOM::HTML_FORM_ELEMENT:           case LFL::DOM::HTML_TABLE_ELEMENT:
    case LFL::DOM::HTML_HEADING_ELEMENT:        case LFL::DOM::HTML_U_LIST_ELEMENT:
      return LFL::DOM::Display(LFL::DOM::Display::Block);
  } return LFL::DOM::Display(LFL::DOM::Display::Inline);
}

LFL::DOM::Clear                ComputedStyle::Clear               () const { return LFL::DOM::Clear(LFL::DOM::Clear::None); }
LFL::DOM::Float                ComputedStyle::Float               () const { return LFL::DOM::Float(LFL::DOM::Float::None); }
LFL::DOM::Position             ComputedStyle::Position            () const { return LFL::DOM::Position(LFL::DOM::Position::Static); }
LFL::DOM::Overflow             ComputedStyle::Overflow            () const { return LFL::DOM::Overflow(LFL::DOM::Overflow::Visible); }
LFL::DOM::TextAlign            ComputedStyle::TextAlign           () const { return LFL::DOM::TextAlign(LFL::DOM::TextAlign::Left); }
LFL::DOM::Direction            ComputedStyle::Direction           () const { return LFL::DOM::Direction(LFL::DOM::Direction::LTR); }
LFL::DOM::FontStyle            ComputedStyle::FontStyle           () const { return LFL::DOM::FontStyle(LFL::DOM::FontStyle::Normal); }
LFL::DOM::EmptyCells           ComputedStyle::EmptyCells          () const { return LFL::DOM::EmptyCells(LFL::DOM::EmptyCells::Show); }
LFL::DOM::FontWeight           ComputedStyle::FontWeight          () const { return LFL::DOM::FontWeight(LFL::DOM::FontWeight::Normal); }
LFL::DOM::Visibility           ComputedStyle::Visibility          () const { return LFL::DOM::Visibility(LFL::DOM::Visibility::Visible); }
LFL::DOM::WhiteSpace           ComputedStyle::WhiteSpace          () const { return LFL::DOM::WhiteSpace(LFL::DOM::WhiteSpace::Normal); }
LFL::DOM::CaptionSide          ComputedStyle::CaptionSide         () const { return LFL::DOM::CaptionSide(LFL::DOM::CaptionSide::Top); }
LFL::DOM::FontVariant          ComputedStyle::FontVariant         () const { return LFL::DOM::FontVariant(LFL::DOM::FontVariant::Normal); }
LFL::DOM::TableLayout          ComputedStyle::TableLayout         () const { return LFL::DOM::TableLayout(LFL::DOM::TableLayout::Auto); }
LFL::DOM::UnicodeBidi          ComputedStyle::UnicodeBidi         () const { return LFL::DOM::UnicodeBidi(LFL::DOM::UnicodeBidi::Normal); }
LFL::DOM::BorderStyle          ComputedStyle::OutlineStyle        () const { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
LFL::DOM::TextTransform        ComputedStyle::TextTransform       () const { return LFL::DOM::TextTransform(LFL::DOM::TextTransform::None); }
LFL::DOM::TextDecoration       ComputedStyle::TextDecoration      () const { return LFL::DOM::TextDecoration(IsLink() ? LFL::DOM::TextDecoration::Underline : LFL::DOM::TextDecoration::None); }
LFL::DOM::BorderCollapse       ComputedStyle::BorderCollapse      () const { return LFL::DOM::BorderCollapse(LFL::DOM::BorderCollapse::Separate); }
LFL::DOM::ListStyleType        ComputedStyle::ListStyleType       () const { return LFL::DOM::ListStyleType(LFL::DOM::ListStyleType::Disc); }
LFL::DOM::ListStylePosition    ComputedStyle::ListStylePosition   () const { return LFL::DOM::ListStylePosition(LFL::DOM::ListStylePosition::Outside); }
LFL::DOM::PageBreak            ComputedStyle::PageBreakAfter      () const { return LFL::DOM::PageBreak(LFL::DOM::PageBreak::Auto); }
LFL::DOM::PageBreak            ComputedStyle::PageBreakBefore     () const { return LFL::DOM::PageBreak(LFL::DOM::PageBreak::Auto); }
LFL::DOM::PageBreak            ComputedStyle::PageBreakInside     () const { return LFL::DOM::PageBreak(LFL::DOM::PageBreak::Auto); }
LFL::DOM::BorderWidth          ComputedStyle::BorderTopWidth      () const { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
LFL::DOM::BorderWidth          ComputedStyle::BorderLeftWidth     () const { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
LFL::DOM::BorderWidth          ComputedStyle::BorderRightWidth    () const { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
LFL::DOM::BorderWidth          ComputedStyle::BorderBottomWidth   () const { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
LFL::DOM::BackgroundRepeat     ComputedStyle::BackgroundRepeat    () const { return LFL::DOM::BackgroundRepeat(LFL::DOM::BackgroundRepeat::Repeat); }
LFL::DOM::BackgroundAttachment ComputedStyle::BackgroundAttachment() const { return LFL::DOM::BackgroundAttachment(LFL::DOM::BackgroundAttachment::Scroll); }
LFL::DOM::RGBColor             ComputedStyle::BackgroundColor     () const { return LFL::DOM::RGBColor(0x00000000); }
LFL::DOM::RGBColor             ComputedStyle::BorderBottomColor   () const { return LFL::DOM::RGBColor(0x00000000); }
LFL::DOM::RGBColor             ComputedStyle::BorderLeftColor     () const { return LFL::DOM::RGBColor(0x00000000); }
LFL::DOM::RGBColor             ComputedStyle::BorderRightColor    () const { return LFL::DOM::RGBColor(0x00000000); }
LFL::DOM::RGBColor             ComputedStyle::BorderTopColor      () const { return LFL::DOM::RGBColor(0x00000000); }
LFL::DOM::RGBColor             ComputedStyle::OutlineColor        () const { return LFL::DOM::RGBColor(0x00000000); }
LFL::DOM::RGBColor             ComputedStyle::Color               () const { return LFL::DOM::RGBColor(0xffffffff); }
LFL::DOM::Cursor               ComputedStyle::Cursor              () const { return LFL::DOM::Cursor(LFL::DOM::Cursor::Auto); }
LFL::DOM::FontFamily           ComputedStyle::FontFamily          () const { return LFL::DOM::FontFamily(FLAGS_font); }
LFL::DOM::CSSStringValue       ComputedStyle::BackgroundImage     () const { return LFL::DOM::CSSStringValue(); }
LFL::DOM::CSSNumericValuePair  ComputedStyle::BackgroundPosition  () const { return LFL::DOM::CSSNumericValuePair(); }
LFL::DOM::Rect                 ComputedStyle::Clip                () const { return LFL::DOM::Rect(); }
LFL::DOM::CSSNumericValue      ComputedStyle::Opacity             () const { return LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::Orphans             () const { return LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::Widows              () const { return LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::MinHeight           () const { return LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::MinWidth            () const { return LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSNumericValue      ComputedStyle::PaddingTop          () const { return LFL::DOM::CSSNumericValue    (0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
LFL::DOM::CSSNumericValue      ComputedStyle::PaddingRight        () const { return LFL::DOM::CSSNumericValue    (0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
LFL::DOM::CSSNumericValue      ComputedStyle::PaddingBottom       () const { return LFL::DOM::CSSNumericValue    (0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
LFL::DOM::CSSNumericValue      ComputedStyle::PaddingLeft         () const { return LFL::DOM::CSSNumericValue    (0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
LFL::DOM::CSSNumericValue      ComputedStyle::TextIndent          () const { return LFL::DOM::CSSNumericValue(); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Top                 () const { return LFL::DOM::CSSAutoNumericValue(true); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Bottom              () const { return LFL::DOM::CSSAutoNumericValue(true); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Left                () const { return LFL::DOM::CSSAutoNumericValue(true); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Right               () const { return LFL::DOM::CSSAutoNumericValue(true); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Width               () const { return LFL::DOM::CSSAutoNumericValue(true); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::Height              () const { return LFL::DOM::CSSAutoNumericValue(true); }
LFL::DOM::FontSize             ComputedStyle::FontSize            () const { return LFL::DOM::FontSize(LFL::DOM::FontSize::Medium); }
LFL::DOM::VerticalAlign        ComputedStyle::VerticalAlign       () const { return LFL::DOM::VerticalAlign(LFL::DOM::VerticalAlign::Baseline); }
LFL::DOM::BorderStyle          ComputedStyle::BorderTopStyle      () const { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
LFL::DOM::BorderStyle          ComputedStyle::BorderRightStyle    () const { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
LFL::DOM::BorderStyle          ComputedStyle::BorderBottomStyle   () const { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
LFL::DOM::BorderStyle          ComputedStyle::BorderLeftStyle     () const { return LFL::DOM::BorderStyle(LFL::DOM::BorderStyle::None); }
LFL::DOM::BorderWidth          ComputedStyle::OutlineWidth        () const { return LFL::DOM::BorderWidth(LFL::DOM::BorderWidth::Medium); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::MarginBottom        () const { return LFL::DOM::CSSAutoNumericValue(0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::MarginLeft          () const { return LFL::DOM::CSSAutoNumericValue(0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::MarginRight         () const { return LFL::DOM::CSSAutoNumericValue(0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::MarginTop           () const { return LFL::DOM::CSSAutoNumericValue(0, LFL::DOM::CSSPrimitiveValue::CSS_PX); }
LFL::DOM::CSSNoneNumericValue  ComputedStyle::MaxHeight           () const { return LFL::DOM::CSSNoneNumericValue(true); }
LFL::DOM::CSSNoneNumericValue  ComputedStyle::MaxWidth            () const { return LFL::DOM::CSSNoneNumericValue(true); }
LFL::DOM::CSSAutoNumericValue  ComputedStyle::ZIndex              () const { return LFL::DOM::CSSAutoNumericValue(true); }
LFL::DOM::CSSNormNumericValue  ComputedStyle::WordSpacing         () const { return LFL::DOM::CSSNormNumericValue(true, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth); }
LFL::DOM::CSSNormNumericValue  ComputedStyle::LetterSpacing       () const { return LFL::DOM::CSSNormNumericValue(true, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth); }
LFL::DOM::CSSNormNumericValue  ComputedStyle::LineHeight          () const { return LFL::DOM::CSSNormNumericValue(true, LFL::DOM::CSSPrimitiveValue::PercentRefersTo::FontHeight); }

int                       ComputedStyle::length() const { return  0; }
LFL::DOM::DOMString       ComputedStyle::item(int index) const { return ""; }
LFL::DOM::DOMString       ComputedStyle::getPropertyValue   (const LFL::DOM::DOMString &n) const { return ""; }
const LFL::DOM::CSSValue *ComputedStyle::getPropertyCSSValue(const LFL::DOM::DOMString &n) const { return  0; }
LFL::DOM::DOMString       ComputedStyle::getPropertyPriority(const LFL::DOM::DOMString &n) const { return ""; }
LFL::DOM::DOMString       ComputedStyle::removeProperty     (const LFL::DOM::DOMString &n) { ERROR("not implemented"); return ""; }
void                      ComputedStyle::setProperty(const LFL::DOM::DOMString &n, const LFL::DOM::DOMString &v, const LFL::DOM::DOMString &priority) { ERROR("not implemented"); }

StyleContext::~StyleContext() {}
StyleContext::StyleContext(LFL::DOM::HTMLDocument *D) {}
void StyleContext::AppendSheet(StyleSheet *ss) {}
void StyleContext::Match(ComputedStyle *out, LFL::DOM::Node *node, const ComputedStyle *parent_sheet, StyleSheet *inline_style) {}
int StyleContext::FontFaces(const string &face_name, vector<LFL::DOM::FontFace> *out) { out->clear(); return 0; }

}; // namespace LFL
