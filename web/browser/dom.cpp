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
#include "core/web/browser/document.h"

namespace LFL {
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
  if (!render) render = ownerDocument->alloc->New<DOM::Renderer>(this);
  return render;
}

void DOM::Node::SetLayoutDirty() { if (render) render->layout_dirty = true; }
void DOM::Node::SetStyleDirty()  { if (render) render->style_dirty  = true; }
void DOM::Node::ClearComputedInlineStyle() { if (render) { render->inline_style.Reset(); render->inline_style_sheet.reset(); } }

void DOM::Element::setAttribute(const DOMString &name, const DOMString &value) {
  DOM::Attr *attr = ownerDocument->alloc->New<DOM::Attr>(ownerDocument);
  attr->name = name;
  attr->value = value;
  setAttributeNode(attr);
}

void DOM::HTMLTableSectionElement::deleteRow(long index) { return parentTable->deleteRow(index); }
DOM::HTMLElement *DOM::HTMLTableSectionElement::insertRow(long index) { return parentTable->insertRow(index); }

void DOM::Renderer::UpdateStyle(Flow *F) {
  DOM::Node *n = style.node;
  CHECK(n->parentNode);
  style.is_root = inline_style.is_root = n->parentNode == n->ownerDocument;
  CHECK(style.is_root || n->parentNode->render);
  Renderer *parent_render = (n->parentNode && n->parentNode->render) ? n->parentNode->render : 0;

  if (!inline_style_sheet) {
    DOM::Attr *inline_style_attr = n->getAttributeNode("style");
    string style_text = StrCat(n->HTML4Style(), inline_style_attr ? inline_style_attr->nodeValue().c_str() : "",
                               inline_style.override_style);
    if (!style_text.empty()) inline_style_sheet =
      make_unique<LFL::StyleSheet>(n->ownerDocument, "", "UTF-8", true, false, style_text.c_str());
  }

  ComputeStyle(n->ownerDocument->style_context, &style, style.is_root ? 0 : &parent_render->style);

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

  if (style.color_not_inherited || !parent_render)                     color = style.Color().v;
  else                                                                 color = parent_render->color;

  if (style.bgcolor_not_inherited && style.BackgroundColor().v.a()==1) solid_background_color = style.BackgroundColor().v;
  else if (!parent_render)                                             solid_background_color = style.BackgroundColor().v;
  else                                                                 solid_background_color = parent_render->solid_background_color;

  if (style.bgcolor_not_inherited || !parent_render)                   background_color = style.BackgroundColor().v;
  else                                                                 background_color = parent_render->background_color;

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
  over_background_image = background_image.get() || (parent_render && parent_render->over_background_image);

  DOM::CSSNormNumericValue lineheight = style.LineHeight(), charspacing = style.LetterSpacing(), wordspacing = style.WordSpacing();
  lineheight_px  = (!lineheight .Null() && !lineheight. _norm) ? lineheight .getPixelValue(F) : 0;
  charspacing_px = (!charspacing.Null() && !charspacing._norm) ? charspacing.getPixelValue(F) : 0;
  wordspacing_px = (!wordspacing.Null() && !wordspacing._norm) ? wordspacing.getPixelValue(F) : 0;
}

Font *DOM::Renderer::UpdateFont(Flow *F, Fonts *fonts, Window *win) {
  DOM::FontFamily  font_family  = style.FontFamily();
  DOM::FontSize    font_size    = style.FontSize();
  DOM::FontStyle   font_style   = style.FontStyle();
  DOM::FontWeight  font_weight  = style.FontWeight();
  DOM::FontVariant font_variant = style.FontVariant();

  int font_size_px = font_size.getFontSizeValue(F);
  int font_flag = ((font_weight.v == DOM::FontWeight::_700 || font_weight.v == DOM::FontWeight::_800 || font_weight.v == DOM::FontWeight::_900 || font_weight.v == DOM::FontWeight::Bold || font_weight.v == DOM::FontWeight::Bolder) ? FontDesc::Bold : 0) |
    ((font_style.v == DOM::FontStyle::Italic || font_style.v == DOM::FontStyle::Oblique) ? FontDesc::Italic : 0);
  Font *font = 0; // font_family.attr ? Fonts::Get("", String::ToUTF8(font_family.cssText()), font_size_px, Color::white, Color::clear, font_flag) : 0;
  for (int i=0; !font && i<font_family.name.size(); i++) {
    vector<DOM::FontFace> ff;
    style.node->ownerDocument->style_context->FontFaces(font_family.name[i], &ff);
    for (int j=0; j<ff.size(); j++) {
      // INFO(j, " got font face: ", ff[j].family.name.size() ? ff[j].family.name[0] : "<NO1>",
      //      " ", ff[j].source.size() ? ff[j].source[0] : "<NO2>");
    }
  }
  return font ? font : fonts->Get
    (win->gl_h, FLAGS_font, "", font_size_px, color,
     over_background_image ? background_color : solid_background_color); 
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
  o_px          = style.OutlineWidth     ().getPixelValue(F);
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
  } else             shrink = normal_flow ? style.node->parentNode->render->shrink : 0;
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
  if (!tile_context_opened && (tile_context_opened=1)) tiles->ContextOpen();
  tiles->AddScissor(w);
}

void DOM::Renderer::Finish() {
  if (tile_context_opened) tiles->ContextClose();
  tile_context_opened = 0;
}

Browser::Document::~Document() {}
Browser::Document::Document(ThreadDispatcher *D, AssetLoading *L, Fonts *F, GraphicsDeviceHolder *G, SocketServices *N, const Box &V) : alloc(1024*1024),
  dispatch(D), loader(L), fonts(F), parser(make_unique<DocumentParser>(G, N, this)) {}

void Browser::Document::Clear() {
  js_context.reset();
  style_sheet.clear();
  alloc.Reset();
  node = alloc.New<DOM::HTMLDocument>(parser.get(), &alloc);
  node->style_context        = alloc.New<StyleContext>(node);
  node->inline_style_context = alloc.New<StyleContext>(node);
  node->style_context       ->AppendSheet(StyleSheet::Default());
  node->inline_style_context->AppendSheet(StyleSheet::Default());
  js_context = JSContext::Create(js_console.get(), node);
  active_input = 0;
}

}; // namespace LFL
