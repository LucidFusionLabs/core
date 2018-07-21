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

#ifdef BROWSER_DEBUG
#define BrowserDebug(...) printf(__VA_ARGS__)
#else
#define BrowserDebug(...)
#endif

namespace LFL {
Browser::Browser(ThreadDispatcher *d, Window *w, AssetLoading *l, Fonts *f, SocketServices *n, ProcessAPI *A, SystemBrowser *B, View *v, const Box &b) :
  doc(d, l, f, w, n, b), dispatch(d), window(w), missing_image(w->parent), v_scrollbar(v), h_scrollbar(v, Widget::Slider::Flag::AttachedHorizontal), system_browser(B) {
  if (Font *maf = f->Get(window->GD()->parent->gl_h, "MenuAtlas", "", 0, Color::white, Color::clear, 0, 0)) {
    missing_image = maf->FindGlyph(0)->tex;
    missing_image.width = missing_image.height = 16;
    missing_image.owner = false;
  }
  doc.parser->redirect_cb = bind(&Browser::Open, this, _1);
  doc.Clear(); 
}

void Browser::Navigate(const string &url) {
  if (!layers) return system_browser->OpenSystemBrowser(url);
  else Open(url);
}

void Browser::Open(const string &url) {
  if (dispatch->render_process) return dispatch->RunInNetworkThread(bind(&ProcessAPIClient::Navigate, dispatch->render_process.get(), url));
  doc.parser->OpenFrame(url, nullptr);
  if (dispatch->main_process) { dispatch->main_process->SetURL(url); dispatch->main_process->SetTitle(url); }
  else                        { SetURLText(url); window->GD()->parent->SetCaption(url); }
}

void Browser::KeyEvent(int key, bool down) {
  if (dispatch->render_process) dispatch->RunInNetworkThread(bind(&ProcessAPIClient::KeyPress, dispatch->render_process.get(), key, down));
  else {
    if (auto n = doc.node->documentElement()) EventNode(n, initial_displacement, key);
    if (down && doc.active_input && doc.active_input->tiles) {
      if (!doc.active_input->HandleSpecialKey(key)) doc.active_input->Input(key);
      doc.active_input->tiles->Run(TilesInterface::RunFlag::DontClear);
    }
  }
}

void Browser::MouseMoved(int x, int y) {
  if (dispatch->render_process || (!dispatch->render_process && !dispatch->main_process)) y -= window->GD()->parent->gl_h+viewport.top()+VScrolled();
  if (dispatch->render_process) { dispatch->RunInNetworkThread(bind(&ProcessAPIClient::MouseMove, dispatch->render_process.get(), x, y, x-mouse.x, y-mouse.y)); mouse=point(x,y); }
  else if (auto n = doc.node->documentElement()) { mouse=point(x,y); EventNode(n, initial_displacement, Mouse::Event::Motion); }
}

void Browser::MouseButton(int b, bool d, int x, int y) {
  // doc.gui.Input(b, mouse, d, 1);
  if (dispatch->render_process || (!dispatch->render_process && !dispatch->main_process)) y -= window->GD()->parent->gl_h+viewport.top()+VScrolled();
  if (dispatch->render_process) { dispatch->RunInNetworkThread(bind(&ProcessAPIClient::MouseClick, dispatch->render_process.get(), b, d, x, y)); mouse=point(x,y); }
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
  if (dispatch->main_process) dispatch->main_process->SetClearColor(c);
  else                        window->GD()->ClearColor(c);
}

void Browser::SetViewport(int w, int h) {
  viewport.SetDimension(point(w, h));
  if (dispatch->render_process)          dispatch->RunInNetworkThread(bind(&ProcessAPIClient::SetViewport, dispatch->render_process.get(), w, h));
  else if (auto html = doc.DocElement()) html->SetLayoutDirty(); 
}

void Browser::Layout(const Box &b) {
  if (dim.x == b.w && dim.y == b.h) return;
  dim = (viewport = b).Dimension();
  Widget::Slider::AttachContentBox(&viewport, &v_scrollbar, &h_scrollbar);
  SetViewport(viewport.w, viewport.h);
}

void Browser::Draw(const Box &b) {
  if (!dispatch->render_process && (!doc.node || !doc.node->documentElement())) return;
  int v_scrolled = VScrolled(), h_scrolled = HScrolled(); 
  if (!layers) { Render(v_scrolled); UpdateScrollbar(); }
  else {
    if (dispatch->render_process) { IPCTrace("Browser::Draw ipc_buffer_size: %zd\n", dispatch->render_process->ipc_buffer.size()); }
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
  INFO(dispatch->main_process ? "Render" : "Main", " process Browser::Render, requested: ", doc.parser->requested, ", completed:",
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
  auto gd = window->GD();
  DOM::Renderer *render = n->render;
  ComputedStyle *style = &render->style;
  bool is_body  = n->htmlElementType == DOM::HTML_BODY_ELEMENT;
  bool is_table = n->htmlElementType == DOM::HTML_TABLE_ELEMENT;
  if (render->style_dirty || render->layout_dirty) { ScissorStack ss(gd); LayoutNode(flow, n, 0); }
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
      render->child_bg .Draw(gd, displacement);
      render->child_box.Draw(gd, displacement);
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
    flow->SetFont((render->child_flow.cur_attr.font = render->UpdateFont(flow, doc.fonts, window)));

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
      if (!input->text) input->text = make_unique<TiledTextBox>(window);
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
    render->content = render->box.RelativeCoordinatesBox();
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
      flow.out->PushBack(*box, flow.cur_attr, Singleton<BoxFilled>::Set());
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
                       flow.cur_attr, Singleton<BoxFilled>::Set());
  }
  if (render->bs_b) {
    flow.SetFGColor(&render->border_bottom);
    flow.out->PushBack(Box(render->border.x, render->border.y, render->border.w, render->padding.y - render->border.y),
                       flow.cur_attr, Singleton<BoxFilled>::Set());
  }
  if (render->bs_r) {
    flow.SetFGColor(&render->border_right);
    flow.out->PushBack(Box(render->padding.right(), render->border.y, render->border.right() - render->padding.right(), render->border.h),
                       flow.cur_attr, Singleton<BoxFilled>::Set());
  }
  if (render->bs_l) {
    flow.SetFGColor(&render->border_left);
    flow.out->PushBack(Box(render->border.x, render->border.y, render->padding.x - render->border.x, render->border.h),
                       flow.cur_attr, Singleton<BoxFilled>::Set());
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

unique_ptr<BrowserInterface> CreateDefaultBrowser(ThreadDispatcher *d, Window *g, AssetLoading *l, Fonts *f, SocketServices *n, ProcessAPI *a, SystemBrowser *b, View *v, int w, int h) {
  unique_ptr<BrowserInterface> ret;
  // if ((ret = CreateQTWebKitBrowser (v, w, h))) return ret;
  // if ((ret = CreateBerkeliumBrowser(v, w, h))) return ret;
  return make_unique<Browser>(d, g, l, f, n, a, b, v, Box(w, h));
}

}; // namespace LFL
