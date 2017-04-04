/*
 * $Id: gui.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef LFL_CORE_APP_BROWSER_H__
#define LFL_CORE_APP_BROWSER_H__

#include "core/web/dom.h"
#include "core/web/css.h"
#include "core/web/html.h"

namespace LFL {

struct BrowserController : public InputController {
  BrowserInterface *browser;
  int last_mx=0, last_my=0;
  BrowserController(BrowserInterface *B) : browser(B) { active=1; }
  void Button(InputEvent::Id event, bool down) {
    int key = InputEvent::GetKey(event);
    if (key)                                browser->KeyEvent(key, down);
    else if (event == Mouse::Event::Wheel)  browser->MouseWheel(0, down*32);
    else if (event == Mouse::Event::Click2) browser->MouseButton(2, down, last_mx, last_my);
    else if (event == Mouse::Event::Click) {
      Window *screen = app->focused;
      if (down) screen->active_textbox = 0;
      browser->MouseButton(1, down, screen->mouse.x, screen->mouse.y);
    }
  }
  void Moved(InputEvent::Id event, point p, point d) {
    if (event == Mouse::Event::Motion) browser->MouseMoved((last_mx = p.x), (last_my = p.y));
  }
};

namespace DOM {
struct Renderer : public Object {
  Box3 inline_box;
  FloatContainer box;
  ComputedStyle style, inline_style;
  unique_ptr<LFL::StyleSheet> inline_style_sheet;
  bool style_dirty=1, layout_dirty=1, establishes_layer=0;
  Flow *flow=0, *parent_flow=0, child_flow;
  DOM::Node *absolute_parent=0;
  shared_ptr<Texture> background_image;
  DrawableBoxArray child_box, child_bg;
  TilesInterface *tiles=0;

  Box content, padding, border, margin, clip_rect;
  Color color, background_color, solid_background_color, border_top, border_bottom, border_right, border_left, outline;

  bool display_table_element=0, display_table=0, display_inline_table=0, display_block=0, display_inline=0;
  bool display_inline_block=0, display_list_item=0, display_none=0, block_level_box=0, establishes_block=0;
  bool position_relative=0, position_absolute=0, position_fixed=0, positioned=0, float_left=0, float_right=0, floating=0, normal_flow=0;
  bool done_positioned=0, done_floated=0, textalign_center=0, textalign_right=0, underline=0, overline=0, midline=0, blink=0, uppercase=0, lowercase=0, capitalize=0, valign_top=0, valign_mid=0;
  bool bgfixed=0, bgrepeat_x=0, bgrepeat_y=0, border_collapse=0, clear_left=0, clear_right=0, right_to_left=0, hidden=0, inline_block=0;
  bool width_percent=0, width_auto=0, height_auto=0, ml_auto=0, mr_auto=0, mt_auto=0, mb_auto=0;
  bool overflow_hidden=0, overflow_scroll=0, overflow_auto=0, clip=0, shrink=0, tile_context_opened=0, over_background_image=0;
  int width_px=0, height_px=0, ml_px=0, mr_px=0, mt_px=0, mb_px=0, bl_px=0, br_px=0, bt_px=0, bb_px=0, pl_px=0, pr_px=0, pt_px=0, pb_px=0, o_px=0;
  int lineheight_px=0, charspacing_px=0, wordspacing_px=0, valign_px=0, bgposition_x=0, bgposition_y=0, bs_t=0, bs_b=0, bs_r=0, bs_l=0, os=0;
  int clear_height=0, row_height=0, cell_colspan=0, cell_rowspan=0, extra_cell_height=0, max_child_i=-1;

  Renderer(Node *N) : style(N), inline_style(N) {}

  void  UpdateStyle(Flow *F);
  Font* UpdateFont(Flow *F);
  void  UpdateDimensions(Flow *F);
  void  UpdateMarginWidth(Flow *F, int w);
  void  UpdateBackgroundImage(Flow *F);
  void  UpdateFlowAttributes(Flow *F);
  int ClampWidth(int w);
  void PushScissor(const Box &w);
  void Finish();

  bool Dirty() const { return style_dirty || layout_dirty; }
  int TopBorderOffset    () const { return pt_px + bt_px; }
  int RightBorderOffset  () const { return pr_px + br_px; }
  int BottomBorderOffset () const { return pb_px + bb_px; }
  int LeftBorderOffset   () const { return pl_px + bl_px; }
  int TopMarginOffset    () const { return mt_px + TopBorderOffset(); }
  int RightMarginOffset  () const { return mr_px + RightBorderOffset(); }
  int BottomMarginOffset () const { return mb_px + BottomBorderOffset(); }
  int LeftMarginOffset   () const { return ml_px + LeftBorderOffset(); }
  point MarginPosition   () const { return -point(LeftMarginOffset(), BottomMarginOffset()); }
  Border PaddingOffset   () const { return Border(pt_px, pr_px, pb_px, pl_px); }
  Border BorderOffset    () const { return Border(TopBorderOffset(), RightBorderOffset(), BottomBorderOffset(), LeftBorderOffset()); }
  Border MarginOffset    () const { return Border(TopMarginOffset(), RightMarginOffset(), BottomMarginOffset(), LeftMarginOffset()); }
  int MarginWidth        () const { return LeftMarginOffset() + RightMarginOffset(); }
  int MarginHeight       () const { return TopMarginOffset () + BottomMarginOffset(); }
  int MarginBoxWidth     () const { return width_auto ? 0 : (width_px + MarginWidth()); }
  int WidthAuto       (Flow *flow)        const { return max(0, flow->container->w - MarginWidth()); }
  int MarginCenterAuto(Flow *flow, int w) const { return max(0, flow->container->w - bl_px - pl_px - w - br_px - pr_px) / 2; }
  int MarginLeftAuto  (Flow *flow, int w) const { return max(0, flow->container->w - bl_px - pl_px - w - RightMarginOffset()); } 
  int MarginRightAuto (Flow *flow, int w) const { return max(0, flow->container->w - br_px - pr_px - w - LeftMarginOffset()); }
  void ComputeStyle(StyleContext *style_context, ComputedStyle *out, const ComputedStyle *parent=0) const {
    style_context->Match(out, style.node, parent, inline_style_sheet.get());
  }
}; }; // namespace DOM

struct Browser : public BrowserInterface {
  struct Document {
    DOM::BlockChainObjectAlloc alloc;
    DOM::HTMLDocument *node=0;
    string content_type, char_set;
    vector<unique_ptr<StyleSheet>> style_sheet;
    unique_ptr<DocumentParser> parser;
    unique_ptr<JSContext> js_context;
    unique_ptr<Console> js_console;
    TiledTextBox *active_input=0;
    int height=0;

    ~Document();
    Document(const Box &V=Box());
    const DOM::Element *DocElement() const { return node ? node->documentElement() : 0; }
    /**/  DOM::Element *DocElement()       { return node ? node->documentElement() : 0; }
    bool Dirty() const { if (auto html = DocElement()) return html->render->layout_dirty || html->render->style_dirty; return 0; }
    void Clear();
  };
  struct RenderLog {
    string data; int indent=0;
    void Clear() { data.clear(); indent=0; }
  };

  Document doc;
  unique_ptr<LayersInterface> layers;
  RenderLog *render_log=0;
  Texture missing_image;
  point dim, mouse, initial_displacement;
  Box viewport;
  Widget::Slider v_scrollbar, h_scrollbar;
  StringCB url_cb;

  Browser(View *v=0, const Box &b=Box());

  void Navigate(const string &url);
  void Open(const string &url);
  void KeyEvent(int key, bool down);
  void MouseMoved(int x, int y);
  void MouseButton(int b, bool d, int x, int y);
  void MouseWheel(int xs, int ys);
  void BackButton() {}
  void ForwardButton() {}
  void RefreshButton() {}
  void AnchorClicked(DOM::HTMLAnchorElement *anchor);
  void SetClearColor(const Color &c);
  void SetViewport(int w, int h);
  void SetDocsize(int w, int h) { /*doc.width=w;*/ doc.height=h; } 
  Box Viewport() const { return Box(viewport.Dimension()); }
  int VScrolled() const { return v_scrollbar.scrolled * X_or_Y(v_scrollbar.doc_height, 1000); }
  int HScrolled() const { return h_scrollbar.scrolled * 1000; }
  void InitLayers(unique_ptr<LayersInterface> l) { CHECK(!layers); (layers = move(l))->Init(app->focused->gd, 2); }
  void PaintTile(int x, int y, int z, int flag, const MultiProcessPaintResource &paint);
  string GetURL() const { return String::ToUTF8(doc.node->URL); }
  void SetURLText(const string &s) { if (url_cb) url_cb(s); }

  void Layout(const Box &b);
  void Draw(const Box &b);
  void UpdateScrollbar();
  void Render(bool screen_coords=0, int v_scrolled=0);

  bool       EventNode       (DOM::Node*, const point &displacement, InputEvent::Id);
  void       Paint           (Flow *flow, const point &displacement);
  void       PaintNode       (Flow *flow, DOM::Node*, const point &displacement);
  DOM::Node *LayoutNode      (Flow *flow, DOM::Node*, bool reflow);
  void       LayoutBackground(DOM::Node*);
  void       LayoutTable     (Flow *flow, DOM::HTMLTableElement *n);
  void       UpdateTableStyle(Flow *flow, DOM::Node *n);
  void       UpdateRenderLog (DOM::Node *n, const point &displacement);

  bool VisitChildren        (DOM::Node *n, const function<bool(DOM::Node*)>&);
  bool VisitTableChildren   (DOM::Node *n, const function<bool(DOM::Node*)>&);
  bool VisitFloatingChildren(DOM::Node *n, const function<bool(DOM::Node*, const FloatContainer::Float&)>&);

  static int ToWebKitY(const Box &w) { return -w.y - w.h; }
};

unique_ptr<BrowserInterface> CreateQTWebKitBrowser (View *v, int w=1024, int h=1024);
unique_ptr<BrowserInterface> CreateBerkeliumBrowser(View *v, int w=1024, int h=1024);
unique_ptr<BrowserInterface> CreateDefaultBrowser  (View *v, int w=1024, int h=1024);

}; // namespace LFL
#endif // LFL_CORE_APP_BROWSER_H__
