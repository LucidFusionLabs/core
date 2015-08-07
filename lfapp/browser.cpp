/*
 * $Id: gui.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "lfapp/ipc.h"
#include "lfapp/browser.h"
#include "../crawler/html.h"
#include "../crawler/document.h"

#ifdef LFL_QT
#include <QWindow>
#include <QWebView>
#include <QWebFrame>
#include <QWebElement>
#include <QMouseEvent>
#include <QApplication>
#endif

#ifdef LFL_BERKELIUM
#include "berkelium/Berkelium.hpp"
#include "berkelium/Window.hpp"
#include "berkelium/WindowDelegate.hpp"
#include "berkelium/Context.hpp"
#endif

namespace LFL {
BrowserInterface *CreateDefaultBrowser(Window *W, Asset *a, int w, int h) {
  BrowserInterface *ret = 0;
  if ((ret = CreateQTWebKitBrowser(a)))        return ret;
  if ((ret = CreateBerkeliumBrowser(a, w, h))) return ret;
  return new Browser(W, Box(w, h));
}

int PercentRefersTo(unsigned short prt, Flow *inline_context) {
  switch (prt) {
    case DOM::CSSPrimitiveValue::PercentRefersTo::FontWidth:           return inline_context->cur_attr.font->max_width;
    case DOM::CSSPrimitiveValue::PercentRefersTo::FontHeight:          return inline_context->cur_attr.font->Height();
    case DOM::CSSPrimitiveValue::PercentRefersTo::LineHeight:          return inline_context->layout.line_height;
    case DOM::CSSPrimitiveValue::PercentRefersTo::ContainingBoxHeight: return inline_context->container->h;
  }; return inline_context->container->w;
}

#ifdef LFL_LIBCSS
css_error StyleSheet::Import(void *pw, css_stylesheet *parent, lwc_string *url, uint64_t media) {
  LFL::DOM::Document *D = (LFL::DOM::Document*)((StyleSheet*)pw)->ownerDocument;
  if (D) D->parser->OpenStyleImport(LibCSS_String::ToUTF8String(url)); // XXX use parent
  INFO("libcss Import ", LibCSS_String::ToString(url));
  return CSS_INVALID; // CSS_OK;
}
#endif

string DOM::Node::HTML4Style() { 
  string ret;
  if (Node *a = getAttributeNode("align")) {
    const DOMString &v = a->nodeValue();
    if      (StringEquals(v, "right"))  StrAppend(&ret, "margin-left: auto;\n");
    else if (StringEquals(v, "center")) StrAppend(&ret, "margin-left: auto;\nmargin-right: auto;\n");
  }
  if (Node *a = getAttributeNode("background")) StrAppend(&ret, "background-image: ", a->nodeValue(), ";\n");
  if (Node *a = getAttributeNode("bgcolor"))    StrAppend(&ret, "background-color: ", a->nodeValue(), ";\n");
  if (Node *a = getAttributeNode("border"))     StrAppend(&ret, "border-width: ",     a->nodeValue(), "px;\n");
  if (Node *a = getAttributeNode("color"))      StrAppend(&ret, "color: ",            a->nodeValue(), ";\n");
  if (Node *a = getAttributeNode("height"))     StrAppend(&ret, "height: ",           a->nodeValue(), "px;\n");
  if (Node *a = getAttributeNode("text"))       StrAppend(&ret, "color: ",            a->nodeValue(), ";\n");
  if (Node *a = getAttributeNode("width"))      StrAppend(&ret, "width: ",            a->nodeValue(), "px;\n");
  return ret;
}

DOM::Node *DOM::Node::cloneNode(bool deep) {
  FATAL("not implemented");
}

DOM::Renderer *DOM::Node::AttachRender() {
  if (!render) render = AllocatorNew(ownerDocument->alloc, (DOM::Renderer), (this));
  return render;
}

void DOM::Element::setAttribute(const DOMString &name, const DOMString &value) {
  DOM::Attr *attr = AllocatorNew(ownerDocument->alloc, (DOM::Attr), (ownerDocument));
  attr->name = name;
  attr->value = value;
  setAttributeNode(attr);
}

float DOM::CSSPrimitiveValue::ConvertToPixels(float n, unsigned short u, unsigned short prt, Flow *inline_context) {
  switch (u) {
    case CSS_PX:         return          n;
    case CSS_EXS:        return          n  * inline_context->cur_attr.font->max_width;
    case CSS_EMS:        return          n  * inline_context->cur_attr.font->size;
    case CSS_IN:         return          n  * FLAGS_dots_per_inch;
    case CSS_CM:         return CMToInch(n) * FLAGS_dots_per_inch;
    case CSS_MM:         return MMToInch(n) * FLAGS_dots_per_inch;
    case CSS_PC:         return          n  /   6 * FLAGS_dots_per_inch;
    case CSS_PT:         return          n  /  72 * FLAGS_dots_per_inch;
    case CSS_PERCENTAGE: return          n  / 100 * LFL::PercentRefersTo(prt, inline_context);
  }; return 0;
}

float DOM::CSSPrimitiveValue::ConvertFromPixels(float n, unsigned short u, unsigned short prt, Flow *inline_context) {
  switch (u) {
    case CSS_PX:         return          n;
    case CSS_EXS:        return          n / inline_context->cur_attr.font->max_width;
    case CSS_EMS:        return          n / inline_context->cur_attr.font->size;
    case CSS_IN:         return          n / FLAGS_dots_per_inch;
    case CSS_CM:         return InchToCM(n / FLAGS_dots_per_inch);
    case CSS_MM:         return InchToMM(n / FLAGS_dots_per_inch);
    case CSS_PC:         return          n *   6 / FLAGS_dots_per_inch;
    case CSS_PT:         return          n *  72 / FLAGS_dots_per_inch;
    case CSS_PERCENTAGE: return          n * 100 / LFL::PercentRefersTo(prt, inline_context);
  }; return 0;                            
}

int DOM::FontSize::getFontSizeValue(Flow *flow) {
  int base = FLAGS_default_font_size; float scale = 1.2;
  switch (attr) {
    case XXSmall: return base/scale/scale/scale;    case XLarge:  return base*scale*scale;
    case XSmall:  return base/scale/scale;          case XXLarge: return base*scale*scale*scale;
    case Small:   return base/scale;                case Larger:  return flow->cur_attr.font->size*scale;
    case Medium:  return base;                      case Smaller: return flow->cur_attr.font->size/scale;
    case Large:   return base*scale;
  }
  return getPixelValue(flow);
}

void DOM::HTMLTableSectionElement::deleteRow(long index) { return parentTable->deleteRow(index); }
DOM::HTMLElement *DOM::HTMLTableSectionElement::insertRow(long index) { return parentTable->insertRow(index); }

void DOM::Renderer::UpdateStyle(Flow *F) {
  DOM::Node *n = style.node;
  CHECK(n->parentNode);
  style.is_root = n->parentNode == n->ownerDocument;
  CHECK(style.is_root || n->parentNode->render);
  DOM::Attr *inline_style = n->getAttributeNode("style");
  n->ownerDocument->style_context->Match(&style, n, style.is_root ? 0 : &n->parentNode->render->style,
                                         inline_style ? inline_style->nodeValue().c_str() : 0);

  DOM::Display              display        = style.Display();
  DOM::Position             position       = style.Position();
  DOM::Float                _float         = style.Float();
  DOM::Clear                clear          = style.Clear();
  DOM::Visibility           visibility     = style.Visibility();
  DOM::Overflow             overflow       = style.Overflow();
  DOM::TextAlign            textalign      = style.TextAlign();
  DOM::TextTransform        texttransform  = style.TextTransform();
  DOM::TextDecoration       textdecoration = style.TextDecoration();
  DOM::CSSStringValue       bgimage        = style.BackgroundImage();
  DOM::BackgroundRepeat     bgrepeat       = style.BackgroundRepeat();
  DOM::BackgroundAttachment bgattachment   = style.BackgroundAttachment();
  DOM::VerticalAlign        valign         = style.VerticalAlign();
  DOM::CSSAutoNumericValue  zindex         = style.ZIndex();
  DOM::BorderStyle          os             = style.OutlineStyle();
  DOM::BorderStyle          bs_top         = style.BorderTopStyle();
  DOM::BorderStyle          bs_bottom      = style.BorderBottomStyle();
  DOM::BorderStyle          bs_left        = style.BorderLeftStyle();
  DOM::BorderStyle          bs_right       = style.BorderRightStyle();

  display_table_element = display.TableElement();
  display_table         = display.v  == DOM::Display::Table;
  display_inline_table  = display.v  == DOM::Display::InlineTable;
  display_block         = display.v  == DOM::Display::Block;
  display_inline        = display.v  == DOM::Display::Inline;
  display_inline_block  = display.v  == DOM::Display::InlineBlock;
  display_list_item     = display.v  == DOM::Display::ListItem;
  display_none          = display.v  == DOM::Display::None || display.v == DOM::Display::Column || display.v == DOM::Display::ColGroup;

  position_relative = position.v == DOM::Position::Relative;
  position_absolute = position.v == DOM::Position::Absolute;
  position_fixed    = position.v == DOM::Position::Fixed;
  positioned = position_relative || position_absolute || position_fixed;

  if (position_absolute || position_fixed) {
    floating = float_left = float_right = 0;
  } else {
    float_left  = _float.v == DOM::Float::Left;
    float_right = _float.v == DOM::Float::Right;
    floating    = float_left || float_right;
  }

  if (position_absolute) {
    DOM::Rect r = style.Clip();
    if ((clip = !r.Null())) {
      clip_rect = Box(r.left .getPixelValue(F), r.top   .getPixelValue(F),
                      r.right.getPixelValue(F), r.bottom.getPixelValue(F));
      clip_rect.w = max(0, clip_rect.w - clip_rect.x);
      clip_rect.h = max(0, clip_rect.h - clip_rect.y);
    }
  } else clip = 0;

  overflow_auto   = overflow.v == DOM::Overflow::Auto;
  overflow_scroll = overflow.v == DOM::Overflow::Scroll;
  overflow_hidden = overflow.v == DOM::Overflow::Hidden;

  normal_flow = !position_absolute && !position_fixed && !floating && !display_table_element && !style.is_root;
  inline_block = display_inline_block || display_inline_table;
  bool block_level_element = display_block || display_list_item || display_table;
  establishes_block = style.is_root || floating || position_absolute || position_fixed ||
    inline_block || display_table_element ||
    (block_level_element && (overflow_hidden || overflow_scroll || overflow_auto));
  block_level_box = block_level_element | establishes_block;
  if (!block_level_box) overflow_auto = overflow_scroll = overflow_hidden = 0;

  right_to_left    = style.Direction     ().v == DOM::Direction::RTL;
  border_collapse  = style.BorderCollapse().v == DOM::BorderCollapse::Collapse;
  clear_left       = (clear.v == DOM::Clear::Left  || clear.v == DOM::Clear::Both);
  clear_right      = (clear.v == DOM::Clear::Right || clear.v == DOM::Clear::Both);
  hidden           = (visibility.v == DOM::Visibility::Hidden || visibility.v == DOM::Visibility::Collapse);
  valign_top       = (valign.v == DOM::VerticalAlign::Top || valign.v == DOM::VerticalAlign::TextTop);
  valign_mid       = (valign.v == DOM::VerticalAlign::Middle);
  valign_px        = (!valign.Null() && !valign.attr) ? valign.getPixelValue(F) : 0;
  bgrepeat_x       = (bgrepeat.v == DOM::BackgroundRepeat::Repeat || bgrepeat.v == DOM::BackgroundRepeat::RepeatX);
  bgrepeat_y       = (bgrepeat.v == DOM::BackgroundRepeat::Repeat || bgrepeat.v == DOM::BackgroundRepeat::RepeatY);
  bgfixed          = bgattachment  .v == DOM::BackgroundAttachment::Fixed;
  textalign_center = textalign     .v == DOM::TextAlign::Center;
  textalign_right  = textalign     .v == DOM::TextAlign::Right;
  underline        = textdecoration.v == DOM::TextDecoration::Underline;
  overline         = textdecoration.v == DOM::TextDecoration::Overline;
  midline          = textdecoration.v == DOM::TextDecoration::LineThrough;
  blink            = textdecoration.v == DOM::TextDecoration::Blink;
  uppercase        = texttransform .v == DOM::TextTransform::Uppercase;
  lowercase        = texttransform .v == DOM::TextTransform::Lowercase;
  capitalize       = texttransform .v == DOM::TextTransform::Capitalize;

  if (n->nodeType == DOM::TEXT_NODE) {
    color = style.Color().v;
    color.a() = 1.0;
  }
  if (style.bgcolor_not_inherited) {
    background_color = style.BackgroundColor().v;
  }

  os            = os       .Null() ? 0 : os.v;
  bs_t          = bs_top   .Null() ? 0 : bs_top.v;
  bs_b          = bs_bottom.Null() ? 0 : bs_bottom.v;
  bs_l          = bs_left  .Null() ? 0 : bs_left.v;
  bs_r          = bs_right .Null() ? 0 : bs_right.v;
  border_top    = Color(style.BorderTopColor   ().v, 1.0);
  border_left   = Color(style.BorderLeftColor  ().v, 1.0);
  border_right  = Color(style.BorderRightColor ().v, 1.0);
  border_bottom = Color(style.BorderBottomColor().v, 1.0);
  outline       = Color(style.OutlineColor     ().v, 1.0);

  if (!bgimage.Null() && !background_image)
    background_image = n->ownerDocument->parser->OpenImage(String::ToUTF8(bgimage.v));

  DOM::CSSNormNumericValue lineheight = style.LineHeight(), charspacing = style.LetterSpacing(), wordspacing = style.WordSpacing();
  lineheight_px  = (!lineheight .Null() && !lineheight. _norm) ? lineheight .getPixelValue(F) : 0;
  charspacing_px = (!charspacing.Null() && !charspacing._norm) ? charspacing.getPixelValue(F) : 0;
  wordspacing_px = (!wordspacing.Null() && !wordspacing._norm) ? wordspacing.getPixelValue(F) : 0;
}

Font *DOM::Renderer::UpdateFont(Flow *F) {
  DOM::FontFamily  font_family  = style.FontFamily();
  DOM::FontSize    font_size    = style.FontSize();
  DOM::FontStyle   font_style   = style.FontStyle();
  DOM::FontWeight  font_weight  = style.FontWeight();
  DOM::FontVariant font_variant = style.FontVariant();

  int font_size_px = font_size.getFontSizeValue(F);
  int font_flag = ((font_weight.v == DOM::FontWeight::_700 || font_weight.v == DOM::FontWeight::_800 || font_weight.v == DOM::FontWeight::_900 || font_weight.v == DOM::FontWeight::Bold || font_weight.v == DOM::FontWeight::Bolder) ? FontDesc::Bold : 0) |
    ((font_style.v == DOM::FontStyle::Italic || font_style.v == DOM::FontStyle::Oblique) ? FontDesc::Italic : 0);
  Font *font = font_family.attr ? Fonts::Get("", String::ToUTF8(font_family.cssText()), font_size_px, Color::white, Color::clear, font_flag) : 0;
  for (int i=0; !font && i<font_family.name.size(); i++) {
    vector<DOM::FontFace> ff;
    style.node->ownerDocument->style_context->FontFaces(font_family.name[i], &ff);
    for (int j=0; j<ff.size(); j++) {
      // INFO(j, " got font face: ", ff[j].family.name.size() ? ff[j].family.name[0] : "<NO1>",
      //      " ", ff[j].source.size() ? ff[j].source[0] : "<NO2>");
    }
  }
  return font ? font : Fonts::Get(FLAGS_default_font, "", font_size_px, Color::white);
}

void DOM::Renderer::UpdateDimensions(Flow *F) {
  DOM::CSSAutoNumericValue     width = style.   Width(),     height = style.   Height();
  DOM::CSSNoneNumericValue max_width = style.MaxWidth(), max_height = style.MaxHeight();
  DOM::CSSAutoNumericValue ml = style.MarginLeft(), mr = style.MarginRight();
  DOM::CSSAutoNumericValue mt = style.MarginTop(),  mb = style.MarginBottom();

  bool no_auto_margin = position_absolute || position_fixed || display_table_element;
  width_percent = width .IsPercent();
  width_auto    = width ._auto;
  height_auto   = height._auto;
  ml_auto       = ml    ._auto && !no_auto_margin;
  mr_auto       = mr    ._auto && !no_auto_margin;
  mt_auto       = mt    ._auto && !no_auto_margin;
  mb_auto       = mb    ._auto && !no_auto_margin;
  width_px      = width_auto  ? 0 : width .getPixelValue(F);
  height_px     = height_auto ? 0 : height.getPixelValue(F);
  ml_px         = ml_auto     ? 0 : ml    .getPixelValue(F);
  mr_px         = mr_auto     ? 0 : mr    .getPixelValue(F);
  mt_px         = mt_auto     ? 0 : mt    .getPixelValue(F);
  mb_px         = mb_auto     ? 0 : mb    .getPixelValue(F);
  bl_px         = !bs_l ? 0 : style.BorderLeftWidth  ().getPixelValue(F);
  br_px         = !bs_r ? 0 : style.BorderRightWidth ().getPixelValue(F);
  bt_px         = !bs_t ? 0 : style.BorderTopWidth   ().getPixelValue(F);
  bb_px         = !bs_b ? 0 : style.BorderBottomWidth().getPixelValue(F);
  o_px         = style.OutlineWidth     ().getPixelValue(F);
  pl_px         = style.PaddingLeft      ().getPixelValue(F);
  pr_px         = style.PaddingRight     ().getPixelValue(F);
  pt_px         = style.PaddingTop       ().getPixelValue(F);
  pb_px         = style.PaddingBottom    ().getPixelValue(F);

  bool include_floats = block_level_box && establishes_block && !floating;
  if (include_floats && width_auto) box.w = box.CenterFloatWidth(F->p.y, F->cur_line.height) - MarginWidth(); 
  else                              box.w = width_auto ? WidthAuto(F) : width_px;
  if (!max_width._none)             box.w = min(box.w, max_width.getPixelValue(F));

  if (!width_auto) UpdateMarginWidth(F, width_px);

  if (block_level_box && (width_auto || width_percent)) {
    if (normal_flow) shrink = inline_block || style.node->parentNode->render->shrink;
    else             shrink = width_auto && (floating || position_absolute || position_fixed);
  } else               shrink = normal_flow ? style.node->parentNode->render->shrink : 0;
}

void DOM::Renderer::UpdateMarginWidth(Flow *F, int w) {
  if (ml_auto && mr_auto) ml_px = mr_px = MarginCenterAuto(F, w);
  else if       (ml_auto) ml_px =         MarginLeftAuto  (F, w);
  else if       (mr_auto) mr_px =         MarginRightAuto (F, w);
}

int DOM::Renderer::ClampWidth(int w) {
  DOM::CSSNoneNumericValue max_width = style.MaxWidth();
  if (!max_width._none) w = min(w, max_width.getPixelValue(flow));
  return max(w, style.MinWidth().getPixelValue(flow));
}

void DOM::Renderer::UpdateBackgroundImage(Flow *F) {
  int bgpercent_x = 0, bgpercent_y = 0; bool bgdone_x = 0, bgdone_y = 0;
  int iw = background_image->width, ih = background_image->height; 
  DOM::CSSNumericValuePair bgposition = style.BackgroundPosition();
  if (!bgposition.Null()) {
    if (bgposition.first .IsPercent()) bgpercent_x = bgposition.first .v; else { bgposition_x = bgposition.first .getPixelValue(F);      bgdone_x = 0; }
    if (bgposition.second.IsPercent()) bgpercent_y = bgposition.second.v; else { bgposition_y = bgposition.second.getPixelValue(F) + ih; bgdone_y = 0; }
  }
  if (!bgdone_x) { float p=bgpercent_x/100.0; bgposition_x = box.w*p - iw*p; }
  if (!bgdone_y) { float p=bgpercent_y/100.0; bgposition_y = box.h*p + ih*(1-p); }
}

void DOM::Renderer::UpdateFlowAttributes(Flow *F) {
  if (charspacing_px)       F->layout.char_spacing       = charspacing_px;
  if (wordspacing_px)       F->layout.word_spacing       = wordspacing_px;
  if (lineheight_px)        F->layout.line_height        = lineheight_px;
  if (valign_px)            F->layout.valign_offset      = valign_px;
  if (1)                    F->layout.align_center       = textalign_center;
  if (1)                    F->layout.align_right        = textalign_right;
  if (1)                    F->cur_attr.underline        = underline;
  if (1)                    F->cur_attr.overline         = overline;
  if (1)                    F->cur_attr.midline          = midline;
  if (1)                    F->cur_attr.blink            = blink;
  if      (capitalize)      F->layout.word_start_char_tf = ::toupper;
  if      (uppercase)       F->layout.char_tf            = ::toupper;
  else if (lowercase)       F->layout.char_tf            = ::tolower;
  else                      F->layout.char_tf            = 0;
  if (valign_top) {}
  if (valign_mid) {}
}

void DOM::Renderer::PushScissor(const Box &w) {
  if (!tiles) return;
  if (!tile_context_opened) { tile_context_opened=1; tiles->ContextOpen(); }
  TilesPreAdd (tiles, &Tiles::PushScissor, tiles, w);
  TilesPostAdd(tiles, &GraphicsDevice::PopScissor, screen->gd);
}

void DOM::Renderer::Finish() {
  if (tile_context_opened) tiles->ContextClose();
  tile_context_opened = 0;
}

Browser::Document::~Document() { delete parser; }
Browser::Document::Document(Window *W, const Box &V) : gui(W, V), parser(new DocumentParser(this)),
  v_scrollbar(&gui, Box(V.w, V.h)),
  h_scrollbar(&gui, Box(V.w, V.h), Widget::Scrollbar::Flag::AttachedHorizontal), alloc(1024*1024) {}

void Browser::Document::Clear() {
  delete js_context;
  VectorClear(&style_sheet);
  gui.Clear();
  v_scrollbar.LayoutAttached(gui.box);
  h_scrollbar.LayoutAttached(gui.box);
  alloc.Reset();
  node = AllocatorNew(&alloc, (DOM::HTMLDocument), (parser, &alloc, &gui));
  node->style_context = AllocatorNew(&alloc, (StyleContext), (node));
  node->style_context->AppendSheet(StyleSheet::Default());
  js_context = CreateV8JSContext(js_console, node);
}

Browser::Browser(Window *W, const Box &V) : doc(W, V) {
  if (Font *maf = Fonts::Get("MenuAtlas", "", 0, Color::black, Color::clear, 0, 0)) {
    missing_image = maf->FindGlyph(0)->tex;
    missing_image.width = missing_image.height = 16;
  }
  doc.Clear(); 
}

void Browser::Navigate(const string &url) {
  if (layers.empty()) return SystemBrowser::Open(url.c_str());
}

void Browser::Open(const string &url) {
  doc.parser->OpenFrame(url, (DOM::Frame*)NULL);
}

void Browser::KeyEvent(int key, bool down) {}
void Browser::MouseMoved(int x, int y) { mouse=point(x,y); }
void Browser::MouseButton(int b, bool d) { doc.gui.Input(b, mouse, d, 1); }
void Browser::MouseWheel(int xs, int ys) {}
void Browser::AnchorClicked(DOM::HTMLAnchorElement *anchor) {
  Navigate(String::ToUTF8(anchor->getAttribute("href")));
}

bool Browser::Dirty(Box *VP) {
  if (layers.empty()) return true;
  DOM::Node *n = doc.node->documentElement();
  Box viewport = Viewport();
  bool resized = viewport.w != VP->w || viewport.h != VP->h;
  if (resized && n) n->render->layout_dirty = true;
  return n && (n->render->layout_dirty || n->render->style_dirty);
}

void Browser::Draw(Box *VP) { return Draw(VP, Dirty(VP)); }
void Browser::Draw(Box *VP, bool dirty) {
  if (!VP || !doc.node || !doc.node->documentElement()) return;
  doc.gui.box = *VP;
  bool tiles = layers.size();
  int v_scrolled = doc.v_scrollbar.scrolled * doc.v_scrollbar.doc_height;
  int h_scrolled = doc.h_scrollbar.scrolled * 1000; // doc.v_scrollbar.doc_height;
  if (dirty) {
    DOM::Renderer *html_render = doc.node->documentElement()->render;
    html_render->tiles = tiles ? layers[0] : 0;
    html_render->child_bg.Reset();
    Flow flow(html_render->box.Reset(), 0, html_render->child_box.Reset());
    flow.p.y -= tiles ? 0 : v_scrolled;
    Draw(&flow, Viewport().TopLeft());
    doc.height = html_render->box.h;
    if (tiles) layers.Update();
  }
  for (int i=0; i<layers.size(); i++) layers[i]->Draw(Viewport(), !i ? h_scrolled : 0, !i ? -v_scrolled : 0);
}

void Browser::Draw(Flow *flow, const point &displacement) {
  initial_displacement = displacement;
  DrawNode(flow, doc.node->documentElement(), initial_displacement);
  if (layers.empty()) DrawScrollbar();
}

void Browser::DrawScrollbar() {
  doc.gui.Draw();
  doc.v_scrollbar.SetDocHeight(doc.height);
  doc.v_scrollbar.Update();
  doc.h_scrollbar.Update();
  doc.gui.Activate();
}

void Browser::DrawNode(Flow *flow, DOM::Node *n, const point &displacement_in) {
  if (!n) return;
  DOM::Renderer *render = n->render;
  ComputedStyle *style = &render->style;
  bool is_body  = n->htmlElementType == DOM::HTML_BODY_ELEMENT;
  bool is_table = n->htmlElementType == DOM::HTML_TABLE_ELEMENT;
  if (render->style_dirty || render->layout_dirty) { ScissorStack ss; LayoutNode(flow, n, 0); }
  if (render->display_none) return;

  if (!render->done_positioned) {
    render->done_positioned = 1;
    if (render->positioned) {
      DOM::CSSAutoNumericValue pt = style->Top(),  pb = style->Bottom();
      DOM::CSSAutoNumericValue pl = style->Left(), pr = style->Right();
      int ptp = pt.Null() ? 0 : pt.getPixelValue(flow), pbp = pb.Null() ? 0 : pb.getPixelValue(flow);
      int plp = pl.Null() ? 0 : pl.getPixelValue(flow), prp = pr.Null() ? 0 : pr.getPixelValue(flow);
      if (render->position_relative) { 
        if      (ptp) render->box.y -= ptp;
        else if (pbp) render->box.y += pbp;
        if      (plp) render->box.x += plp;
        else if (prp) render->box.x -= prp;
      } else {
        CHECK(!render->floating);
        Box viewport = Viewport();
        const Box *cont =
          (render->position_fixed || render->absolute_parent == doc.node->documentElement()) ?
          &viewport : &render->absolute_parent->render->box;

        if (render->height_auto && !pt._auto && !pb._auto) render->box.h = cont->h - render->MarginHeight() - ptp - pbp;
        if (render->width_auto  && !pl._auto && !pr._auto) render->box.w = cont->w - render->MarginWidth () - plp - prp;

        Box margin = Box::AddBorder(render->box, render->MarginOffset());
        if      (!pt._auto) margin.y =  0       - ptp - margin.h;
        else if (!pb._auto) margin.y = -cont->h + pbp;
        if      (!pl._auto) margin.x =  0       + plp;
        else if (!pr._auto) margin.x =  cont->w - prp - margin.w;
        render->box = Box::DelBorder(margin, render->MarginOffset());
      }
    }
  }

  point displacement = displacement_in + (render->block_level_box ? render->box.TopLeft() : point());
  if (render_log)              UpdateRenderLog(n);
  if (render->clip)            render->PushScissor(render->border.TopLeft(render->clip_rect) + displacement);
  if (render->overflow_hidden) render->PushScissor(render->content                           + displacement);
  if (!render->hidden) {
    if (render->tiles) {
      render->tiles->AddDrawableBoxArray(render->child_bg,  displacement);
      render->tiles->AddDrawableBoxArray(render->child_box, displacement);
    } else {
      render->child_bg .Draw(displacement);
      render->child_box.Draw(displacement);
    }
  }

  if (is_table) {
    DOM::HTMLTableElement *table = n->AsHTMLTableElement();
    HTMLTableRowIter(table)
      if (!row->render->Dirty()) HTMLTableColIter(row) 
        if (!cell->render->Dirty()) DrawNode(render->flow, cell, displacement);
  } else {
    DOM::NodeList *children = &n->childNodes;
    for (int i=0, l=children->length(); i<l; i++) {
      DOM::Node *child = children->item(i);
      if (!child->render->done_floated) DrawNode(render->flow, child, displacement);
    }
  }

  if (render->block_level_box) {
    for (int i=0; i<render->box.float_left .size(); i++) {
      if (render->box.float_left[i].inherited) continue;
      DOM::Node *child = (DOM::Node*)render->box.float_left[i].val;
      if (!child->render->done_positioned) {
        EX_EQ(render->box.float_left[i].w, child->render->margin.w);
        EX_EQ(render->box.float_left[i].h, child->render->margin.h);
        child->render->box.SetPosition(render->box.float_left[i].Position() - child->render->MarginPosition());
      }
      DrawNode(render->flow, child, displacement);
    }
    for (int i=0; i<render->box.float_right.size(); i++) {
      if (render->box.float_right[i].inherited) continue;
      DOM::Node *child = (DOM::Node*)render->box.float_right[i].val;
      if (!child->render->done_positioned) {
        EX_EQ(render->box.float_right[i].w, child->render->margin.w);
        EX_EQ(render->box.float_right[i].h, child->render->margin.h);
        child->render->box.SetPosition(render->box.float_right[i].Position() - child->render->MarginPosition());
      }
      DrawNode(render->flow, child, displacement);
    }
    if (render_log) render_log->indent -= 2;
  }
  render->Finish();
}

DOM::Node *Browser::LayoutNode(Flow *flow, DOM::Node *n, bool reflow) {
  if (!n) return 0;
  DOM::Renderer *render = n->render;
  ComputedStyle *style = &render->style;
  bool is_table = n->htmlElementType == DOM::HTML_TABLE_ELEMENT;
  bool table_element = render->display_table_element;

  if (!reflow) {
    if (render->style_dirty) render->UpdateStyle(flow);
    if (!style->is_root) render->tiles = n->parentNode->render->tiles;
    render->done_positioned = render->done_floated = 0;
    render->max_child_i = -1;
  } else if (render->done_floated) return 0;
  if (render->display_none) { render->style_dirty = render->layout_dirty = 0; return 0; }

  if (render->position_absolute) {
    DOM::Node *ap = n->parentNode;
    while (ap && ap->render && (!ap->render->positioned || !ap->render->block_level_box)) ap = ap->parentNode;
    render->absolute_parent = (ap && ap->render) ? ap : doc.node->documentElement();
  } else render->absolute_parent = 0;

  render->parent_flow = render->absolute_parent ? render->absolute_parent->render->flow : flow;
  if (!table_element) {
    render->box = style->is_root ? Viewport() : Box(max(0, flow->container->w), 0);
    render->UpdateDimensions(render->parent_flow);
  }

  render->clear_height = flow->container->AsFloatContainer()->ClearFloats(flow->p.y, flow->cur_line.height, render->clear_left, render->clear_right);
  if (render->clear_height) flow->AppendVerticalSpace(render->clear_height);

  if (style->font_not_inherited || !(render->child_flow.cur_attr.font = flow->cur_attr.font)) render->child_flow.cur_attr.font = render->UpdateFont(flow);
  if (style->is_root) flow->SetFont(render->child_flow.cur_attr.font);

  if (render->block_level_box) {
    if (!table_element) {
      if (!flow->cur_line.fresh && (render->normal_flow || render->position_absolute) && 
          !render->inline_block && !render->floating && !render->position_absolute) flow->AppendNewlines(1);
      render->box.y = flow->p.y;
    }
    render->child_flow = Flow(&render->box, render->child_flow.cur_attr.font, render->child_box.Reset());
    render->flow = &render->child_flow;
    if (!reflow) {
      render->box.Reset();
      if (render->normal_flow && !render->inline_block) render->box.InheritFloats(flow->container->AsFloatContainer());
      if (render->position_fixed && layers.size()) render->tiles = layers[1];
    }
  } else render->flow = render->parent_flow;
  render->UpdateFlowAttributes(render->flow);

  if (n->nodeType == DOM::TEXT_NODE) {
    DOM::Text *T = n->AsText();
    if (1)                            render->flow->SetFGColor(&render->color);
    // if (style->bgcolor_not_inherited) render->flow->SetBGColor(&render->background_color);
    render->flow->AppendText(T->data);
  } else if ((n->htmlElementType == DOM::HTML_IMAGE_ELEMENT) ||
             (n->htmlElementType == DOM::HTML_INPUT_ELEMENT && StringEquals(n->AsElement()->getAttribute("type"), "image"))) {
    Texture *tex = n->htmlElementType == DOM::HTML_IMAGE_ELEMENT ? n->AsHTMLImageElement()->tex.get() : n->AsHTMLInputElement()->image_tex.get();
    bool missing = !tex || !tex->ID;
    if (missing) tex = &missing_image;

    point dim(n->render->width_px, n->render->height_px);
    if      (!dim.x && !dim.y)                dim   = point(tex->width, tex->height);
    if      ( dim.x && !dim.y && tex->width ) dim.y = RoundF((float)tex->height/tex->width *dim.x);
    else if (!dim.x &&  dim.y && tex->height) dim.x = RoundF((float)tex->width /tex->height*dim.y);

    bool add_margin = !n->render->floating && !n->render->position_absolute && !n->render->position_fixed;
    Border margin = add_margin ? n->render->MarginOffset() : Border();
    if (missing) {
      point d(max(0, dim.x - (int)tex->width), max(0, dim.y - (int)tex->height));
      margin += Border(d.y/2, d.x/2, d.y/2, d.x/2);
      dim -= d;
    }
    render->flow->SetFGColor(&Color::white);
    render->flow->AppendBox(dim.x, dim.y, tex);
    render->box = render->flow->out->data.back().box;
  } 
  else if (n->htmlElementType == DOM::HTML_BR_ELEMENT) render->flow->AppendNewlines(1);
  else if (is_table) LayoutTable(render->flow, n->AsHTMLTableElement());

  vector<Flow::RollbackState> child_flow_state;
  DOM::NodeList *children = &n->childNodes;
  for (int i=0, l=children->length(); !is_table && i<l; i++) {
    if (render->block_level_box) child_flow_state.push_back(render->flow->GetRollbackState());

    DOM::Node *child = children->item(i);
    bool reflow_child = i <= render->max_child_i;
    if (render->style_dirty  && !reflow_child) child->render->style_dirty  = 1;
    if (render->layout_dirty && !reflow_child) child->render->layout_dirty = 1;
    DOM::Node *descendent_float = LayoutNode(render->flow, child, reflow_child);
    Max(&render->max_child_i, i);
    if (!descendent_float) continue;
    if (!render->block_level_box) return descendent_float;

    Box df_margin;
    DOM::Renderer *dfr = descendent_float->render;
    CHECK_EQ(dfr->parent_flow, render->flow);
    int fy = render->flow->p.y - max(0, dfr->margin.h - render->flow->cur_line.height);
    render->box.AddFloat(fy, dfr->margin.w, dfr->margin.h, dfr->float_right, descendent_float, &df_margin);
    dfr->done_floated = 1;

    bool same_line = !render->flow->cur_line.fresh;
    int descendent_float_newlines = render->flow->out->line.size();
    while (child_flow_state.size() && render->flow->out->line.size() == descendent_float_newlines) {
      render->flow->Rollback(PopBack(child_flow_state));
      same_line = (render->flow->out->line.size() == descendent_float_newlines && !render->flow->cur_line.fresh);
    }
    EX_EQ(same_line, false);
    i = child_flow_state.size() - 1;
  }

  if (render->block_level_box) {
    render->flow->Complete();
    int block_height = X_or_Y(render->height_px, max(render->flow->Height(), render->establishes_block ? render->box.FloatHeight() : 0));
    int block_width  = render->shrink ? render->ClampWidth(render->flow->max_line_width) : render->box.w;
    if (render->normal_flow) {
      if (render->inline_block) render->parent_flow->AppendBox  (block_width, block_height, render->MarginOffset(), &render->box);
      else                      render->parent_flow->AppendBlock(block_width, block_height, render->MarginOffset(), &render->box); 
      if (!render->establishes_block) render->box.AddFloatsToParent((FloatContainer*)render->parent_flow->container->AsFloatContainer());
    } else {
      render->box.SetDimension(point(block_width, block_height));
      if (style->is_root || render->position_absolute || table_element) render->box.y -= block_height;
    }
  }

  if (render->background_image) render->UpdateBackgroundImage(flow);
  if (style->bgcolor_not_inherited || render->inline_block || render->clip || render->floating
      || render->bs_t || render->bs_b || render->bs_r || render->bs_l) {
    render->content = Box(0, -render->box.h, render->box.w, render->box.h);
    render->padding = Box::AddBorder(render->content, render->PaddingOffset());
    render->border  = Box::AddBorder(render->content, render->BorderOffset());
    render->margin  = Box::AddBorder(render->content, render->MarginOffset());
  }
  LayoutBackground(n);

  render->style_dirty = render->layout_dirty = 0;
  return (render->floating && !render->done_floated) ? n : 0;
}

void Browser::LayoutBackground(DOM::Node *n) {
  bool is_body = n->htmlElementType == DOM::HTML_BODY_ELEMENT;
  DOM::Renderer *render = n->render;
  ComputedStyle *style = &render->style;
  Flow flow(0, 0, render->child_bg.Reset());

  const Box                                                         *box = &render->content;
  if      (is_body)                                                  box = &render->margin;
  else if (render->block_level_box || render->display_table_element) box = &render->padding;

  if (style->bgcolor_not_inherited && render->background_color.A()) {
    if (is_body) screen->gd->ClearColor(Color(render->background_color, 0.0));
    else {
      flow.SetFGColor(&render->background_color);
      flow.out->PushBack(*box, flow.cur_attr, Singleton<BoxFilled>::Get());
    }
  } else if (is_body) screen->gd->ClearColor(Color(1.0, 1.0, 1.0, 0.0));

  if (render->background_image && render->background_image->width && 
      render->background_image->height) {
    Box bgi(box->x     + render->bgposition_x, 
            box->top() - render->bgposition_y, 
            render->background_image->width,
            render->background_image->height);

    flow.cur_attr.Clear();
    flow.cur_attr.scissor = box;
    // flow.cur_attr.tex = &render->background_image;

    int start_x = !n->render->bgrepeat_x ? bgi.x       : box->x - bgi.w + (n->render->bgposition_x % bgi.w);
    int start_y = !n->render->bgrepeat_y ? bgi.y       : box->y - bgi.h + (n->render->bgposition_y % bgi.h);
    int end_x   = !n->render->bgrepeat_x ? bgi.right() : box->right();
    int end_y   = !n->render->bgrepeat_y ? bgi.top  () : box->top  ();
    for     (int y = start_y; y < end_y; y += bgi.h) { bgi.y = y;
      for (int x = start_x; x < end_x; x += bgi.w) { bgi.x = x;
        flow.out->PushBack(bgi, flow.cur_attr, render->background_image.get());
      }
    }
  }

  flow.cur_attr.Clear();
  if (render->bs_t) {
    flow.SetFGColor(&render->border_top);
    flow.out->PushBack(Box(render->border.x, render->padding.top(), render->border.w, render->border.top() - render->padding.top()),
                       flow.cur_attr, Singleton<BoxFilled>::Get());
  }
  if (render->bs_b) {
    flow.SetFGColor(&render->border_bottom);
    flow.out->PushBack(Box(render->border.x, render->border.y, render->border.w, render->padding.y - render->border.y),
                       flow.cur_attr, Singleton<BoxFilled>::Get());
  }
  if (render->bs_r) {
    flow.SetFGColor(&render->border_right);
    flow.out->PushBack(Box(render->padding.right(), render->border.y, render->border.right() - render->padding.right(), render->border.h),
                       flow.cur_attr, Singleton<BoxFilled>::Get());
  }
  if (render->bs_l) {
    flow.SetFGColor(&render->border_left);
    flow.out->PushBack(Box(render->border.x, render->border.y, render->padding.x - render->border.x, render->border.h),
                       flow.cur_attr, Singleton<BoxFilled>::Get());
  }
}

void Browser::LayoutTable(Flow *flow, DOM::HTMLTableElement *n) {
  UpdateTableStyle(flow, n);
  TableFlow table(flow);
  table.Select();

  HTMLTableIterColGroup(n)
    table.SetMinColumnWidth(j, col->render->MarginBoxWidth(), X_or_1(atoi(col->getAttribute("span"))));

  HTMLTableRowIter(n) {
    HTMLTableColIter(row) {
      DOM::Renderer *cr = cell->render;
      TableFlow::Column *cj = table.SetCellDim(j, cr->MarginBoxWidth(), cr->cell_colspan, cr->cell_rowspan);
      if (!n->render->width_auto || !cr->width_auto) continue;
      cr->box = Box(0,0); LayoutNode(flow, cell, 0); Max(&cj->max_width, cr->flow->max_line_width + cr->MarginWidth()); 
      cr->box = Box(1,0); LayoutNode(flow, cell, 0); Max(&cj->min_width, cr->flow->max_line_width + cr->MarginWidth());
    }
    table.NextRowDim();
  }

  n->render->box.w = table.ComputeWidth(n->render->width_auto ? 0 : n->render->width_px);
  if (n->render->width_auto) n->render->UpdateMarginWidth(n->render->parent_flow, n->render->box.w);

  HTMLTableRowIter(n) {
    HTMLTableColIter(row) {
      DOM::Renderer *cr = cell->render;
      table.AppendCell(j, &cr->box, cr->cell_colspan);
      cr->box.DelBorder(cr->MarginOffset().LeftRight());
      LayoutNode(flow, cell, 0);
      table.SetCellHeight(j, cr->box.h + cr->MarginHeight(), cell, cr->cell_colspan, cr->cell_rowspan);
    }
    row->render->row_height = table.AppendRow();
    row->render->layout_dirty = 0;
  }
}

void Browser::UpdateTableStyle(Flow *flow, DOM::Node *n) {
  for (int i=0, l=n->childNodes.length(); i<l; i++) {
    DOM::Node *child = n->childNodes.item(i);
    DOM::HTMLTableCellElement *cell = child->AsHTMLTableCellElement();
    if (n->render->style_dirty) child->render->style_dirty = 1;
    if (!child->AsHTMLTableRowElement() && !child->AsHTMLTableSectionElement() &&
        !child->AsHTMLTableColElement() && !child->AsHTMLTableCaptionElement() && !cell) continue;

    if (child->render->style_dirty) {
      child->render->UpdateStyle(flow);
      child->render->display_table_element = 1;
      child->render->tiles = child->parentNode->render->tiles;
      child->render->UpdateDimensions(flow);
      if (cell) {
        cell->render->cell_colspan = X_or_1(atoi(cell->getAttribute("colspan")));
        cell->render->cell_rowspan = X_or_1(atoi(cell->getAttribute("rowspan")));
      }
    }
    UpdateTableStyle(flow, child);
  }
  n->render->style_dirty = 0;
}

void Browser::UpdateRenderLog(DOM::Node *n) {
  DOM::Renderer *render = n->render;
  if (render_log->data.empty()) {
    CHECK(render->style.is_root);
    Box viewport = Viewport();
    render_log->indent = 2;
    render_log->data = StrCat("layer at (0,0) size ",          viewport.w, "x", viewport.h,
                              "\n  RenderView at (0,0) size ", viewport.w, "x", viewport.h,
                              "\nlayer at (0,0) size ", render->box.w, "x", render->box.h, "\n");
  }
  if (render->block_level_box) {
    StrAppend(&render_log->data, string(render_log->indent, ' '), "RenderBlock {", toupper(n->nodeName()), "} at (",
              render->box.x, ",", ScreenToWebKitY(render->box), ") size ", render->box.w, "x", render->box.h, "\n");
    render_log->indent += 2;
  }
  if (n->nodeType == DOM::TEXT_NODE) {
    StrAppend(&render_log->data, string(render_log->indent, ' '),
              "text run at (", render->box.x, ",",    ScreenToWebKitY(render->box),
              ") width ",      render->box.w, ": \"", n->nodeValue(), "\"\n");
  }
}

struct BrowserController : public InputController {
  BrowserInterface *browser;
  BrowserController(BrowserInterface *B) : browser(B) {}
  void Input(InputEvent::Id event, bool down) {
    int key = InputEvent::GetKey(event);
    if (key)                                browser->KeyEvent(key, down);
    else if (event == Mouse::Event::Motion) browser->MouseMoved(screen->mouse.x, screen->mouse.y);
    else if (event == Mouse::Event::Wheel)  browser->MouseWheel(0, down*32);
    else                                    browser->MouseButton(event, down);
  }
};

#ifdef LFL_QT
extern QApplication *lfl_qapp;
class QTWebKitBrowser : public QObject, public BrowserInterface {
  Q_OBJECT
  public:
    Asset* asset;
    int W, H, lastx, lasty, lastd;
    QWebPage page;

    QTWebKitBrowser(Asset *a) : lastx(0), lasty(0), lastd(0) {
      asset = a;
      Resize(screen->width, screen->height);
      lfl_qapp->connect(&page, SIGNAL(repaintRequested(const QRect&)),           this, SLOT(ViewUpdate(const QRect &)));
      lfl_qapp->connect(&page, SIGNAL(scrollRequested(int, int, const QRect &)), this, SLOT(Scroll(int, int, const QRect &)));
    }
    void Resize(int win, int hin) {
      W = win; H = hin;
      page.setViewportSize(QSize(W, H));
      if (!asset->tex.ID) asset->tex.CreateBacked(W, H);
      else                asset->tex.Resize(W, H);
    }
    void Open(const string &url) { page.mainFrame()->load(QUrl(url.c_str())); }
    void MouseMoved(int x, int y) {
      lastx = x; lasty = screen->height - y; 
      QMouseEvent me(QEvent::MouseMove, QPoint(lastx, lasty), Qt::NoButton,
                     lastd ? Qt::LeftButton : Qt::NoButton, Qt::NoModifier);
      QApplication::sendEvent(&page, &me);
    }
    void MouseButton(int b, bool d) {
      lastd = d;
      QMouseEvent me(QEvent::MouseButtonPress, QPoint(lastx, lasty), Qt::LeftButton,
                     lastd ? Qt::LeftButton : Qt::NoButton, Qt::NoModifier);
      QApplication::sendEvent(&page, &me);
    }
    void MouseWheel(int xs, int ys) {}
    void KeyEvent(int key, bool down) {}
    void BackButton() { page.triggerAction(QWebPage::Back); }
    void ForwardButton() { page.triggerAction(QWebPage::Forward);  }
    void RefreshButton() { page.triggerAction(QWebPage::Reload); }
    string GetURL() { return page.mainFrame()->url().toString().toLocal8Bit().data(); }
    void Draw(Box *w) {
      asset->tex.DrawCrimped(*w, 1, 0, 0);
      QPoint scroll = page.currentFrame()->scrollPosition();
      QWebHitTestResult hit_test = page.currentFrame()->hitTestContent(QPoint(lastx, lasty));
      QWebElement hit = hit_test.element();
      QRect rect = hit.geometry();
      Box hitwin(rect.x() - scroll.x(), w->h-rect.y()-rect.height()+scroll.y(), rect.width(), rect.height());
      screen->gd->SetColor(Color::blue);
      hitwin.Draw();
    }
  public slots:
    void Scroll(int dx, int dy, const QRect &rect) { ViewUpdate(QRect(0, 0, W, H)); }
    void ViewUpdate(const QRect &dirtyRect) {
      QImage image(page.viewportSize(), QImage::Format_ARGB32);
      QPainter painter(&image);
      page.currentFrame()->render(&painter, dirtyRect);
      painter.end();

      int instride = image.bytesPerLine(), bpp = Pixel::size(asset->tex.pf);
      int mx = dirtyRect.x(), my = dirtyRect.y(), mw = dirtyRect.width(), mh = dirtyRect.height();
      const unsigned char *in = (const unsigned char *)image.bits();
      unsigned char *buf = asset->tex.buf;
      screen->gd->BindTexture(GraphicsDevice::Texture2D, asset->tex.ID);
      for (int j = 0; j < mh; j++) {
        int src_ind = (j + my) * instride + mx * bpp;
        int dst_ind = ((j + my) * asset->tex.width + mx) * bpp;
        if (true) VideoResampler::RGB2BGRCopyPixels(buf+dst_ind, in+src_ind, mw, bpp);
        else                                 memcpy(buf+dst_ind, in+src_ind, mw*bpp);
      }
      asset->tex.UpdateGL(Box(0, my, asset->tex.width, dirtyRect.height()));
    }
};

#include "browser.moc"

BrowserInterface *CreateQTWebKitBrowser(Asset *a) { return new QTWebKitBrowser(a); }
#else /* LFL_QT */
BrowserInterface *CreateQTWebKitBrowser(Asset *a) { return 0; }
#endif /* LFL_QT */

#ifdef LFL_BERKELIUM
struct BerkeliumModule : public Module {
  BerkeliumModule() { app->LoadModule(this); }
  int Frame(unsigned t) { Berkelium::update(); return 0; }
  int Free() { Berkelium::destroy(); return 0; }
  int Init() {
    const char *homedir = LFAppDownloadDir();
    INFO("berkelium init");
    Berkelium::init(
#ifdef _WIN32
                    homedir ? Berkelium::FileString::point_to(wstring(homedir).c_str(), wstring(homedir).size()) :
#else
                    homedir ? Berkelium::FileString::point_to(homedir, strlen(homedir)) :
#endif
                    Berkelium::FileString::empty());
    return 0;
  }
};

struct BerkeliumBrowser : public BrowserInterface, public Berkelium::WindowDelegate {
  Asset *asset; int W, H; Berkelium::Window *window;
  BerkeliumBrowser(Asset *a, int win, int hin) : asset(a), window(0) { 
    Singleton<BerkeliumModule>::Get();
    Berkelium::Context* context = Berkelium::Context::create();
    window = Berkelium::Window::create(context);
    delete context;
    window->setDelegate(this);
    resize(win, hin); 
  }

  void Draw(Box *w) { glOverlay(*w, asset, 1); }
  void Open(const string &url) { window->navigateTo(Berkelium::URLString::point_to(url.data(), url.length())); }
  void MouseMoved(int x, int y) { window->mouseMoved((float)x/screen->width*W, (float)(screen->height-y)/screen->height*H); }
  void MouseButton(int b, bool down) {
    if (b == BIND_MOUSE1 && down) {
      int click_x = screen->mouse.x/screen->width*W, click_y = (screen->height - screen->mouse.y)/screen->height*H;
      string js = WStringPrintf(L"lfapp_browser_click(document.elementFromPoint(%d, %d).innerHTML)", click_x, click_y);
      window->executeJavascript(Berkelium::WideString::point_to((wchar_t *)js.c_str(), js.size()));
    }
    window->mouseButton(b == BIND_MOUSE1 ? 0 : 1, down);
  }
  void MouseWheel(int xs, int ys) { window->mouseWheel(xs, ys); }
  void KeyEvent(int key, bool down) {
    int bk = -1;
    switch (key) {
      case Key::PageUp:    bk = 0x21; break;
      case Key::PageDown:  bk = 0x22; break;
      case Key::Left:      bk = 0x25; break;
      case Key::Up:        bk = 0x26; break;
      case Key::Right:     bk = 0x27; break;
      case Key::Down:      bk = 0x28; break;
      case Key::Delete:    bk = 0x2E; break;
      case Key::Backspace: bk = 0x2E; break;
      case Key::Return:    bk = '\r'; break;
      case Key::Tab:       bk = '\t'; break;
    }

    int mods = ctrl_key_down() ? Berkelium::CONTROL_MOD : 0 | shift_key_down() ? Berkelium::SHIFT_MOD : 0;
    window->keyEvent(down, mods, bk != -1 ? bk : key, 0);
    if (!down || bk != -1) return;

    wchar_t wkey[2] = { key, 0 };
    window->textEvent(wkey, 1);
  }

  void Resize(int win, int hin) {
    W = win; H = hin;
    window->resize(W, H);
    if (!asset->texID) PixelBuffer::create(W, H, asset);
    else               asset->tex.resize(W, H, asset->texcoord);
  }

  virtual void onPaint(Berkelium::Window* wini, const unsigned char *in_buf, const Berkelium::Rect &in_rect,
                       size_t num_copy_rects, const Berkelium::Rect* copy_rects, int dx, int dy, const Berkelium::Rect& scroll_rect) {
    unsigned char *buf = asset->tex.buf;
    int bpp = Pixel::size(asset->tex.pf);
    screen->gd->BindTexture(GL_TEXTURE_2D, asset->texID);

    if (dy < 0) {
      const Berkelium::Rect &r = scroll_rect;
      for (int j = -dy, h = r.height(); j < h; j++) {
        int src_ind = ((j + r.top()     ) * asset->tex.w + r.left()) * bpp;
        int dst_ind = ((j + r.top() + dy) * asset->tex.w + r.left()) * bpp;
        memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
      }
    } else if (dy > 0) {
      const Berkelium::Rect &r = scroll_rect;
      for (int j = r.height() - dy; j >= 0; j--) {
        int src_ind = ((j + r.top()     ) * asset->tex.w + r.left()) * bpp;
        int dst_ind = ((j + r.top() + dy) * asset->tex.w + r.left()) * bpp;                
        memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
      }
    }
    if (dx) {
      const Berkelium::Rect &r = scroll_rect;
      for (int j = 0, h = r.height(); j < h; j++) {
        int src_ind = ((j + r.top()) * asset->tex.w + r.left()     ) * bpp;
        int dst_ind = ((j + r.top()) * asset->tex.w + r.left() - dx) * bpp;
        memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
      }
    }
    for (int i = 0; i < num_copy_rects; i++) {
      const Berkelium::Rect &r = copy_rects[i];
      for(int j = 0, h = r.height(); j < h; j++) {
        int dst_ind = ((j + r.top()) * asset->tex.w + r.left()) * bpp;
        int src_ind = ((j + (r.top() - in_rect.top())) * in_rect.width() + (r.left() - in_rect.left())) * bpp;
        memcpy(buf+dst_ind, in_buf+src_ind, r.width()*bpp);
      }
      asset->tex.flush(0, r.top(), asset->tex.w, r.height());        
    }

    // for (Berkelium::Window::FrontToBackIter iter = wini->frontIter(); 0 && iter != wini->frontEnd(); iter++) { Berkelium::Widget *w = *iter; }
  }

  virtual void onCreatedWindow(Berkelium::Window *win, Berkelium::Window *newWindow, const Berkelium::Rect &initialRect) { newWindow->setDelegate(this); }
  virtual void onAddressBarChanged(Berkelium::Window *win, Berkelium::URLString newURL) { INFO("onAddressBarChanged: ", newURL.data()); }
  virtual void onStartLoading(Berkelium::Window *win, Berkelium::URLString newURL) { INFO("onStartLoading: ", newURL.data()); }
  virtual void onLoad(Berkelium::Window *win) { INFO("onLoad"); }
  virtual void onJavascriptCallback(Berkelium::Window *win, void* replyMsg, Berkelium::URLString url, Berkelium::WideString funcName, Berkelium::Script::Variant *args, size_t numArgs) {
    string fname = WideStringToString(funcName);
    if (fname == "lfapp_browser_click" && numArgs == 1 && args[0].type() == args[0].JSSTRING) {
      string a1 = WideStringToString(args[0].toString());
      INFO("jscb: ", fname, " ", a1);
    } else {
      INFO("jscb: ", fname);
    }
    if (replyMsg) win->synchronousScriptReturn(replyMsg, numArgs ? args[0] : Berkelium::Script::Variant());
  }
  virtual void onRunFileChooser(Berkelium::Window *win, int mode, Berkelium::WideString title, Berkelium::FileString defaultFile) { win->filesSelected(NULL); }
  virtual void onNavigationRequested(Berkelium::Window *win, Berkelium::URLString newURL, Berkelium::URLString referrer, bool isNewWindow, bool &cancelDefaultAction) {}

  static string WideStringToString(const Berkelium::WideString& in) {
    string out; out.resize(in.size());
    for (int i = 0; i < in.size(); i++) out[i] = in.data()[i];     
    return out;
  }
};

BrowserInterface *CreateBerkeliumBrowser(Asset *a, int W, int H) {
  BerkeliumBrowser *browser = new BerkeliumBrowser(a, W, H);
  Berkelium::WideString click = Berkelium::WideString::point_to(L"lfapp_browser_click");
  browser->window->bind(click, Berkelium::Script::Variant::bindFunction(click, true));
  return browser;
}
#else /* LFL_BERKELIUM */
BrowserInterface *CreateBerkeliumBrowser(Asset *a, int W, int H) { return 0; }
#endif /* LFL_BERKELIUM */

}; // namespace LFL
