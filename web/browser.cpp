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

#include "core/app/gui.h"
#include "core/app/ipc.h"
#include "core/web/browser.h"
#include "core/web/document.h"

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

#ifdef BROWSER_DEBUG
#define BrowserDebug(...) printf(__VA_ARGS__)
#else
#define BrowserDebug(...)
#endif

namespace LFL {
unique_ptr<BrowserInterface> CreateDefaultBrowser(GUI *g, int w, int h) {
  unique_ptr<BrowserInterface> ret;
  if ((ret = CreateQTWebKitBrowser (g, w, h))) return ret;
  if ((ret = CreateBerkeliumBrowser(g, w, h))) return ret;
  return make_unique<Browser>(g, Box(w, h));
}

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

Font *DOM::Renderer::UpdateFont(Flow *F) {
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
  return font ? font : app->fonts->Get(FLAGS_font, "", font_size_px, color,
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
Browser::Document::Document(const Box &V) : parser(make_unique<DocumentParser>(this)), alloc(1024*1024) {}

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

Browser::Browser(GUI *gui, const Box &V) :
  doc(V), v_scrollbar(gui), h_scrollbar(gui, Widget::Slider::Flag::AttachedHorizontal) {
  if (Font *maf = app->fonts->Get("MenuAtlas", "", 0, Color::white, Color::clear, 0, 0)) {
    missing_image = maf->FindGlyph(0)->tex;
    missing_image.width = missing_image.height = 16;
    missing_image.owner = false;
  }
  doc.parser->redirect_cb = bind(&Browser::Open, this, _1);
  doc.Clear(); 
}

void Browser::Navigate(const string &url) {
  if (!layers) return app->OpenSystemBrowser(url);
  else Open(url);
}

void Browser::Open(const string &url) {
  if (app->render_process) return app->RunInNetworkThread(bind(&ProcessAPIClient::Navigate, app->render_process.get(), url));
  doc.parser->OpenFrame(url, nullptr);
  if (app->main_process) { app->main_process->SetURL(url); app->main_process->SetTitle(url); }
  else                   { SetURLText(url); app->focused->SetCaption(url); }
}

void Browser::KeyEvent(int key, bool down) {
  if (app->render_process) app->RunInNetworkThread(bind(&ProcessAPIClient::KeyPress, app->render_process.get(), key, down));
  else {
    if (auto n = doc.node->documentElement()) EventNode(n, initial_displacement, key);
    if (down && doc.active_input && doc.active_input->tiles) {
      if (!doc.active_input->HandleSpecialKey(key)) doc.active_input->Input(key);
      doc.active_input->tiles->Run(TilesInterface::RunFlag::DontClear);
    }
  }
}

void Browser::MouseMoved(int x, int y) {
  if (app->render_process || (!app->render_process && !app->main_process)) y -= app->focused->height+viewport.top()+VScrolled();
  if (app->render_process) { app->RunInNetworkThread(bind(&ProcessAPIClient::MouseMove, app->render_process.get(), x, y, x-mouse.x, y-mouse.y)); mouse=point(x,y); }
  else if (auto n = doc.node->documentElement()) { mouse=point(x,y); EventNode(n, initial_displacement, Mouse::Event::Motion); }
}

void Browser::MouseButton(int b, bool d, int x, int y) {
  // doc.gui.Input(b, mouse, d, 1);
  if (app->render_process || (!app->render_process && !app->main_process)) y -= app->focused->height+viewport.top()+VScrolled();
  if (app->render_process) { app->RunInNetworkThread(bind(&ProcessAPIClient::MouseClick, app->render_process.get(), b, d, x, y)); mouse=point(x,y); }
  else if (auto n = doc.node->documentElement()) { mouse=point(x,y); EventNode(n, initial_displacement, Mouse::ButtonID(b)); }
}

void Browser::MouseWheel(int xs, int ys) {}
void Browser::AnchorClicked(DOM::HTMLAnchorElement *anchor) {
  const string &d = doc.node->domain;
  string href = String::ToUTF8(anchor->getAttribute("href"));
  if      (href.size() > 1 && href[0] == '/' && href[1] == '/') href = StrCat(d.substr(0, d.find(':')), ":", href);
  else if (href.size()     && href[0] == '/')                   href = StrCat(d,                             href);
  Navigate(href);
}

void Browser::SetClearColor(const Color &c) {
  if (app->main_process) app->main_process->SetClearColor(c);
  else                   app->focused->gd->ClearColor(c);
}

void Browser::SetViewport(int w, int h) {
  viewport.SetDimension(point(w, h));
  if (app->render_process)               app->RunInNetworkThread(bind(&ProcessAPIClient::SetViewport, app->render_process.get(), w, h));
  else if (auto html = doc.DocElement()) html->SetLayoutDirty(); 
}

void Browser::Layout(const Box &b) {
  if (dim.x == b.w && dim.y == b.h) return;
  dim = (viewport = b).Dimension();
  Widget::Slider::AttachContentBox(&viewport, &v_scrollbar, &h_scrollbar);
  SetViewport(viewport.w, viewport.h);
}

void Browser::Draw(const Box &b) {
  if (!app->render_process && (!doc.node || !doc.node->documentElement())) return;
  int v_scrolled = VScrolled(), h_scrolled = HScrolled(); 
  if (!layers) { Render(v_scrolled); UpdateScrollbar(); }
  else {
    if (app->render_process) { IPCTrace("Browser::Draw ipc_buffer_size: %zd\n", app->render_process->ipc_buffer.size()); }
    else if (doc.Dirty()) Render();
    layers->node[0].scrolled = point(h_scrolled, -v_scrolled);
    layers->Draw(viewport + b.TopLeft(), point(0, -viewport.h));
  }
}

void Browser::UpdateScrollbar() {
  v_scrollbar.SetDocHeight(doc.height);
  v_scrollbar.Update();
  h_scrollbar.Update();
}

void Browser::Render(bool screen_coords, int v_scrolled) {
  INFO(app->main_process ? "Render" : "Main", " process Browser::Render, requested: ", doc.parser->requested, ", completed:",
       doc.parser->completed, ", outstanding: ", doc.parser->outstanding.size());
  DOM::Renderer *html_render = doc.node->documentElement()->render;
  html_render->tiles = layers ? layers->layer[0].get() : nullptr;
  html_render->child_bg.Reset();
  if (layers) layers->ClearLayerNodes();
  Flow flow(html_render->box.Reset(), 0, html_render->child_box.Reset());
  flow.p.y -= layers ? 0 : v_scrolled;
  Paint(&flow, screen_coords ? Viewport().TopLeft() : point());
  doc.height = html_render->box.h;
  if (layers) layers->Update();
}

void Browser::PaintTile(int x, int y, int z, int flag, const MultiProcessPaintResource &paint) {
  CHECK(layers && layers->layer.size());
  if (z < 0 || z >= layers->layer.size()) return;
  TilesIPCServer *tiles = dynamic_cast<TilesIPCServer*>(layers->layer[z].get());
  CHECK(tiles);
  tiles->Select();
  tiles->RunTile(y, x, flag, tiles->GetTile(x, y), paint);
  tiles->Release();
}

bool Browser::EventNode(DOM::Node *n, const point &displacement_in, InputEvent::Id type) {
  auto render = n->render;
  if (!render) return true;
  Box b = (render->block_level_box ? render->box : render->inline_box[0]) + displacement_in;
  BrowserDebug("%s mouse %s box %s\n", n->DebugString().c_str(), mouse.DebugString().c_str(), b.DebugString().c_str());
  if (!b.within(mouse)) return true;

  if (auto e = n->AsHTMLAnchorElement()) { AnchorClicked(e); return false; }
  else if (auto e = n->AsHTMLInputElement()) {
    if (e->text && type == Mouse::ButtonID(1)) doc.active_input = e->text.get();
  }
  bool is_table = n->htmlElementType == DOM::HTML_TABLE_ELEMENT;
  point displacement = displacement_in + (render->block_level_box ? render->box.TopLeft() : point());
  if (is_table) { if (!VisitTableChildren (n, [&](DOM::Node *child) { return EventNode(child, displacement, type); })) return false; }
  else          { if (!VisitChildren      (n, [&](DOM::Node *child) { return EventNode(child, displacement, type); })) return false; }
  if (render->block_level_box) { if (!VisitFloatingChildren(n, [&](DOM::Node *child, const FloatContainer::Float &b) {
                                                            return EventNode(child, displacement, type); })) return false; }
  return true;
}

void Browser::Paint(Flow *flow, const point &displacement) {
  initial_displacement = displacement;
  PaintNode(flow, doc.node->documentElement(), initial_displacement);
}

void Browser::PaintNode(Flow *flow, DOM::Node *n, const point &displacement_in) {
  if (!n) return;
  DOM::Renderer *render = n->render;
  ComputedStyle *style = &render->style;
  bool is_body  = n->htmlElementType == DOM::HTML_BODY_ELEMENT;
  bool is_table = n->htmlElementType == DOM::HTML_TABLE_ELEMENT;
  if (render->style_dirty || render->layout_dirty) { ScissorStack ss(app->focused->gd); LayoutNode(flow, n, 0); }
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
  BrowserDebug("PaintNode %s box %s displacement %s\n", n->DebugString().c_str(), Box(render->box).DebugString().c_str(), displacement.DebugString().c_str());
  if (render_log)              UpdateRenderLog(n, displacement_in);
  if (render->clip)            render->PushScissor(render->border.TopLeft(render->clip_rect) + displacement);
  if (render->overflow_hidden) render->PushScissor(render->content                           + displacement);
  if (render->establishes_layer) layers->AddLayerNode(1, render->content + displacement, 1);
  if (!render->hidden) {
    if (render->tiles) {
      render->tiles->AddDrawableBoxArray(render->child_bg,  displacement);
      render->tiles->AddDrawableBoxArray(render->child_box, displacement);
    } else {
      render->child_bg .Draw(app->focused->gd, displacement);
      render->child_box.Draw(app->focused->gd, displacement);
    }
  }
  if (auto input = n->AsHTMLInputElement()) {
    if (input->text) input->text->AssignTarget(render->tiles, render->box.Position() + displacement);
  }

  if (is_table) VisitTableChildren (n, [&](DOM::Node *child) { PaintNode(render->flow, child, displacement); return true; });
  else          VisitChildren      (n, [&](DOM::Node *child) { PaintNode(render->flow, child, displacement); return true; });

  if (render->block_level_box) {
    VisitFloatingChildren
      (n, [&](DOM::Node *child, const FloatContainer::Float &b) {
        if (!child->render->done_positioned) {
          EX_EQ(b.w, child->render->margin.w);
          EX_EQ(b.h, child->render->margin.h);
          child->render->box.SetPosition(b.Position() - child->render->MarginPosition());
        }
        PaintNode(render->flow, child, displacement);
        return true;
      });
    if (render_log) render_log->indent -= 2;
  }
  render->Finish();
}

DOM::Node *Browser::LayoutNode(Flow *flow, DOM::Node *n, bool reflow) {
  if (!n) return 0;
  DOM::Renderer *render = n->render;
  ComputedStyle *style = &render->style;
  bool is_table = n->htmlElementType == DOM::HTML_TABLE_ELEMENT;
  bool is_input = n->htmlElementType == DOM::HTML_INPUT_ELEMENT;
  bool is_image = n->htmlElementType == DOM::HTML_IMAGE_ELEMENT;
  bool table_element = render->display_table_element;

  if (!reflow) {
    if (render->style_dirty) render->UpdateStyle(flow);
    if (!style->is_root) render->tiles = n->parentNode->render->tiles;
    render->done_positioned = render->done_floated = render->establishes_layer = 0;
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

  if (style->font_not_inherited || !(render->child_flow.cur_attr.font = flow->cur_attr.font))
    flow->SetFont((render->child_flow.cur_attr.font = render->UpdateFont(flow)));

  if (render->block_level_box) {
    if (!table_element) {
      if (!flow->cur_line.fresh && (render->normal_flow || render->position_absolute) && 
          !render->inline_block && !render->floating && !render->position_absolute) flow->AppendNewlines(1);
      render->box.y = -flow->Height();
    }
    render->child_flow = Flow(&render->box, render->child_flow.cur_attr.font, render->child_box.Reset());
    render->flow = &render->child_flow;
    if (!reflow) {
      render->box.Reset();
      if (render->normal_flow && !render->inline_block) render->box.InheritFloats(flow->container->AsFloatContainer());
      if (render->position_fixed && layers && Changed(&render->tiles, layers->layer[1].get())) render->establishes_layer = true;
    }
  } else render->flow = render->parent_flow;
  render->UpdateFlowAttributes(render->flow);
  point sp = render->flow->p;
  int beg_out_ind = render->flow->out->Size(), beg_line_ind = render->flow->out->line_ind.size();
  BrowserDebug("LayoutNode %s blb %d font %s sp %s\n", n->DebugString().c_str(), render->block_level_box, render->flow->cur_attr.font->desc->DebugString().c_str(), sp.DebugString().c_str());

  if (n->nodeType == DOM::TEXT_NODE) {
    if (1)                                render->flow->SetFGColor(&Color::white); // &render->color);
    // if (style->bgcolor_not_inherited)  render->flow->SetBGColor(&render->background_color);
    render->flow->AppendText(n->AsText()->data);
  } else if (is_image || (is_input && StringEquals(n->AsElement()->getAttribute("type"), "image"))) {
    Texture *tex = is_image ? n->AsHTMLImageElement()->tex.get() : n->AsHTMLInputElement()->image_tex.get();
    bool missing = !tex || !tex->ID;
    if (missing) tex = &missing_image;

    point dim(n->render->width_px, n->render->height_px);
    if      (!dim.x && !dim.y)                dim   = point(tex->width, tex->height);
    if      ( dim.x && !dim.y && tex->width ) dim.y = RoundF(float(tex->height)/tex->width *dim.x);
    else if (!dim.x &&  dim.y && tex->height) dim.x = RoundF(float(tex->width) /tex->height*dim.y);

    bool add_margin = !n->render->floating && !n->render->position_absolute && !n->render->position_fixed;
    Border margin = add_margin ? n->render->MarginOffset() : Border();
    if (missing) {
      point d(max(0, dim.x - int(tex->width)), max(0, dim.y - int(tex->height)));
      margin += Border(d.y/2, d.x/2, d.y/2, d.x/2);
      dim -= d;
    }
    render->flow->SetFGColor(&Color::white);
    render->flow->AppendBox(dim.x, dim.y, tex);
    render->box = render->flow->out->data.back().box;
  } else if (is_input) {
    string t = n->AsElement()->getAttribute("type");
    if (t == "" || t == "text") {
      DOM::HTMLInputElement *input = n->AsHTMLInputElement();
      if (!input->text) input->text = make_unique<TiledTextBox>();
      point dim(X_or_Y(n->render->width_px, 256), X_or_Y(n->render->height_px, 16));
      render->flow->AppendBox(dim.x, dim.y, &render->box);
      input->text->cmd_fb.SetDimensions(render->box.w, render->box.h,
                                        (input->text->style.font.ptr = render->flow->cur_attr.font));
      render->inline_box[0] = render->box;
    } else if (t == "submit") {
      point dim(X_or_Y(n->render->width_px, 128), X_or_Y(n->render->height_px, 16));
      render->flow->AppendBox(dim.x, dim.y, &render->box);
      render->inline_box[0] = render->box;
    }
  } else if (n->htmlElementType == DOM::HTML_BR_ELEMENT) render->flow->AppendNewlines(1);
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
      if (!render->establishes_block) render->box.AddFloatsToParent(const_cast<FloatContainer*>(render->parent_flow->container->AsFloatContainer()));
    } else {
      render->box.SetDimension(point(block_width, block_height));
      if (style->is_root || render->position_absolute || table_element) render->box.y -= block_height;
    }
  } else if (!is_input) {
    int end_out_ind = flow->out->Size(), end_line_ind = flow->out->line_ind.size();
    int first_line_height = flow->cur_attr.font->Height(), last_line_height = flow->cur_attr.font->Height();
    Box gb = beg_out_ind ? flow->out->data[beg_out_ind-1].box : (flow->out->data.size() ? flow->out->data.begin()->box : Box());
    Box ge = end_out_ind ? flow->out->data[end_out_ind-1].box : Box();
    render->inline_box = Box3(Box(flow->container->w, flow->container->h),
                              gb.Position(), ge.Position() + point(ge.w, 0),
                              first_line_height, last_line_height);
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

  if (style->bgcolor_not_inherited && render->background_color.a()) {
    if (is_body) SetClearColor(render->background_color);
    else {
      flow.SetFGColor(&render->background_color);
      flow.out->PushBack(*box, flow.cur_attr, Singleton<BoxFilled>::Get());
    }
  } else if (is_body) SetClearColor(Color(1.0, 1.0, 1.0, 0.0));

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

void Browser::UpdateRenderLog(DOM::Node *n, const point &displacement) {
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
    Box box = render->box + displacement;
    StrAppend(&render_log->data, string(render_log->indent, ' '), "RenderBlock {", toupper(n->nodeName()), "} at (");
    StrAppend(&render_log->data, box.x, ",", ToWebKitY(box), ") size ", box.w, "x", box.h, "\n");
    render_log->indent += 2;
  }
  if (n->nodeType == DOM::TEXT_NODE) {
    Box inline_box = render->inline_box[0] + displacement;
    StrAppend(&render_log->data, string(render_log->indent, ' '), "text run at (", inline_box.x, ",", ToWebKitY(inline_box));
    StrAppend(&render_log->data, ") width ", inline_box.w, " height ", inline_box.h, " color ", render->color.DebugString());
    StrAppend(&render_log->data, ": \"", n->nodeValue(), "\"\n");
  }
}

bool Browser::VisitChildren(DOM::Node *n, const function<bool(DOM::Node*)> &f) {
  DOM::NodeList *children = &n->childNodes;
  for (int i=0, l=children->length(); i<l; i++) {
    DOM::Node *child = children->item(i);
    if (!child->render->done_floated) if (!f(child)) return false;
  }
  return true;
}

bool Browser::VisitTableChildren(DOM::Node *n, const function<bool(DOM::Node*)> &f) {
  DOM::HTMLTableElement *table = n->AsHTMLTableElement();
  HTMLTableRowIter(table)
    if (!row->render->Dirty()) HTMLTableColIter(row) 
      if (!cell->render->Dirty()) if (!f(cell)) return false;
  return true;
}

bool Browser::VisitFloatingChildren(DOM::Node *n, const function<bool(DOM::Node*, const FloatContainer::Float&)> &f) {
  auto render = n->render;
  if (!render) return true;
  for (auto &i : render->box.float_left)  if (!i.inherited) if (!f(static_cast<DOM::Node*>(i.val), i)) return false;
  for (auto &i : render->box.float_right) if (!i.inherited) if (!f(static_cast<DOM::Node*>(i.val), i)) return false;
  return true;
}

#ifdef LFL_QT
extern QApplication *lfl_qapp;
class QTWebKitBrowser : public QObject, public BrowserInterface {
  Q_OBJECT
  public:
  QWebPage page;
  Texture tex;
  point dim, mouse;
  bool mouse1_down=0;

  QTWebKitBrowser(int w, int h) {
    Resize(w, h);
    lfl_qapp->connect(&page, SIGNAL(repaintRequested(const QRect&)),           this, SLOT(ViewUpdate(const QRect &)));
    lfl_qapp->connect(&page, SIGNAL(scrollRequested(int, int, const QRect &)), this, SLOT(Scroll(int, int, const QRect &)));
  }
  void Resize(int win, int hin) {
    page.setViewportSize(QSize((dim.x = win), (dim.y = hin)));
    if (!tex.ID) tex.CreateBacked(dim.x, dim.y);
    else         tex.Resize(dim.x, dim.y);
  }
  void Open(const string &url) { page.mainFrame()->load(QUrl(url.c_str())); }
  void MouseMoved(int x, int y) {
    mouse = point(x, screen->height - y);
    QMouseEvent me(QEvent::MouseMove, QPoint(mouse.x, mouse.y), Qt::NoButton, mouse1_down ? Qt::LeftButton : Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(&page, &me);
  }
  void MouseButton(int b, bool d, int x, int y) {
    if (b != 1) return;
    mouse1_down = d;
    QMouseEvent me(QEvent::MouseButtonPress, QPoint(mouse.x, mouse.y), Qt::LeftButton, mouse1_down ? Qt::LeftButton : Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(&page, &me);
  }
  void MouseWheel(int xs, int ys) {}
  void KeyEvent(int key, bool down) {}
  void BackButton() { page.triggerAction(QWebPage::Back); }
  void ForwardButton() { page.triggerAction(QWebPage::Forward);  }
  void RefreshButton() { page.triggerAction(QWebPage::Reload); }
  string GetURL() const { return page.mainFrame()->url().toString().toLocal8Bit().data(); }
  void Draw(const Box &w) {
    tex.DrawCrimped(w, 1, 0, 0);
    QPoint scroll = page.currentFrame()->scrollPosition();
    QWebHitTestResult hit_test = page.currentFrame()->hitTestContent(QPoint(mouse.x, mouse.y));
    QWebElement hit = hit_test.element();
    QRect rect = hit.geometry();
    Box hitwin(rect.x() - scroll.x(), w.h-rect.y()-rect.height()+scroll.y(), rect.width(), rect.height());
    screen->gd->SetColor(Color::blue);
    hitwin.Draw();
  }

  public slots:
  void Scroll(int dx, int dy, const QRect &rect) { ViewUpdate(QRect(0, 0, dim.x, dim.y)); }
  void ViewUpdate(const QRect &dirtyRect) {
    QImage image(page.viewportSize(), QImage::Format_ARGB32);
    QPainter painter(&image);
    page.currentFrame()->render(&painter, dirtyRect);
    painter.end();

    int instride = image.bytesPerLine(), bpp = Pixel::Size(tex.pf);
    int mx = dirtyRect.x(), my = dirtyRect.y(), mw = dirtyRect.width(), mh = dirtyRect.height();
    const unsigned char *in = (const unsigned char *)image.bits();
    screen->gd->BindTexture(GraphicsDevice::Texture2D, tex.ID);

    unsigned char *buf = tex.buf;
    for (int j = 0; j < mh; j++) {
      int src_ind = (j + my) * instride + mx * bpp;
      int dst_ind = ((j + my) * tex.width + mx) * bpp;
      if (false) VideoResampler::RGB2BGRCopyPixels(buf+dst_ind, in+src_ind, mw, bpp);
      else                                  memcpy(buf+dst_ind, in+src_ind, mw*bpp);
    }
    tex.UpdateGL(Box(0, my, tex.width, dirtyRect.height()));
    if (!screen->target_fps) app->scheduler.Wakeup(0);
  }
};

#include "browser.moc"

unique_ptr<BrowserInterface> CreateQTWebKitBrowser(GUI *g, int w, int h) { return make_unique<QTWebKitBrowser>(w, h); }
#else /* LFL_QT */
unique_ptr<BrowserInterface> CreateQTWebKitBrowser(GUI *g, int w, int h) { return nullptr; }
#endif /* LFL_QT */

#ifdef LFL_BERKELIUM
struct BerkeliumModule : public Module {
  BerkeliumModule() { app->LoadModule(this); }
  int Frame(unsigned t) { Berkelium::update(); return 0; }
  int Free() { Berkelium::destroy(); return 0; }
  int Init() {
    const char *homedir = app->savedir;
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
  int W, H;
  Texture *tex;
  Berkelium::Window *window;
  BerkeliumBrowser(Texture *t, int win, int hin) : tex(t), window(0) { 
    Singleton<BerkeliumModule>::Get();
    Berkelium::Context* context = Berkelium::Context::create();
    window = Berkelium::Window::create(context);
    delete context;
    window->setDelegate(this);
    resize(win, hin); 
  }

  void Resize(int win, int hin) {
    window->resize((W = win), (H = hin));
    if (!tex->ID) tex->CreateBacked(W, H);
    else          tex->Resize(W, H);
  }
  void Draw(const Box &w) { tex->DrawCrimped(w, 1, 0, 0); }
  void Open(const string &url) { window->navigateTo(Berkelium::URLString::point_to(url.data(), url.length())); }
  void MouseMoved(int x, int y) { window->mouseMoved((float)x/screen->width*W, (float)(screen->height-y)/screen->height*H); }
  void MouseButton(int b, bool down, int x, int y) {
    if (b == 1 && down) {
      int click_x = screen->mouse.x/screen->width*W, click_y = (screen->height - screen->mouse.y)/screen->height*H;
      basic_string<wchar_t> js = WStringPrintf(L"lfapp_browser_click(document.elementFromPoint(%d, %d).innerHTML)", click_x, click_y);
      window->executeJavascript(Berkelium::WideString::point_to(js.data(), js.size()));
    }
    window->mouseButton(b == 1 ? 0 : 1, down);
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

  virtual void onPaint(Berkelium::Window* wini, const unsigned char *in_buf, const Berkelium::Rect &in_rect,
                       size_t num_copy_rects, const Berkelium::Rect* copy_rects, int dx, int dy, const Berkelium::Rect& scroll_rect) {
    unsigned char *buf = tex->buf;
    int bpp = Pixel::size(tex->pf);
    screen->gd->BindTexture(GL_TEXTURE_2D, tex->ID);

    if (dy < 0) {
      const Berkelium::Rect &r = scroll_rect;
      for (int j = -dy, h = r.height(); j < h; j++) {
        int src_ind = ((j + r.top()     ) * tex->w + r.left()) * bpp;
        int dst_ind = ((j + r.top() + dy) * tex->w + r.left()) * bpp;
        memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
      }
    } else if (dy > 0) {
      const Berkelium::Rect &r = scroll_rect;
      for (int j = r.height() - dy; j >= 0; j--) {
        int src_ind = ((j + r.top()     ) * tex->w + r.left()) * bpp;
        int dst_ind = ((j + r.top() + dy) * tex->w + r.left()) * bpp;                
        memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
      }
    }
    if (dx) {
      const Berkelium::Rect &r = scroll_rect;
      for (int j = 0, h = r.height(); j < h; j++) {
        int src_ind = ((j + r.top()) * tex->w + r.left()     ) * bpp;
        int dst_ind = ((j + r.top()) * tex->w + r.left() - dx) * bpp;
        memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
      }
    }
    for (int i = 0; i < num_copy_rects; i++) {
      const Berkelium::Rect &r = copy_rects[i];
      for(int j = 0, h = r.height(); j < h; j++) {
        int dst_ind = ((j + r.top()) * tex->w + r.left()) * bpp;
        int src_ind = ((j + (r.top() - in_rect.top())) * in_rect.width() + (r.left() - in_rect.left())) * bpp;
        memcpy(buf+dst_ind, in_buf+src_ind, r.width()*bpp);
      }
      tex->flush(0, r.top(), tex->w, r.height());        
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

unique_ptr<BrowserInterface> CreateBerkeliumBrowser(GUI *g, int W, int H) {
  unique_ptr<BerkeliumBrowser> browser = make_unique<BerkeliumBrowser>(a, W, H);
  Berkelium::WideString click = Berkelium::WideString::point_to(L"lfapp_browser_click");
  browser->window->bind(click, Berkelium::Script::Variant::bindFunction(click, true));
  return browser;
}
#else /* LFL_BERKELIUM */
unique_ptr<BrowserInterface> CreateBerkeliumBrowser(GUI *g, int W, int H) { return nullptr; }
#endif /* LFL_BERKELIUM */

}; // namespace LFL
