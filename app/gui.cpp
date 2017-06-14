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

namespace LFL {
#ifdef LFL_MOBILE
DEFINE_bool(multitouch, true, "Touchscreen controls");
#else
DEFINE_bool(multitouch, false, "Touchscreen controls");
#endif
DEFINE_bool(console, false, "Enable dropdown lfapp console");
DEFINE_string(console_font, "", "Console font, blank for default_font");
DEFINE_int(console_font_flag, FontDesc::Mono, "Console font flag");
DEFINE_bool(draw_grid, false, "Draw lines intersecting mouse x,y");
DEFINE_FLAG(testbox, Box, Box(), "Test box; change via console: testbox x,y,w,h");
DEFINE_FLAG(testcolor, Color, Color::red, "Test color; change via console: testcolor hexval");

void View::UpdateBox(const Box &b, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box = b;
  if (input_box_ind >= 0) mouse.hit     [input_box_ind].box = b;
}

void View::UpdateBoxX(int x, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.x = x;
  if (input_box_ind >= 0) mouse.hit     [input_box_ind].box.x = x;
}

void View::UpdateBoxY(int y, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.y = y;
  if (input_box_ind >= 0) mouse.hit     [input_box_ind].box.y = y;
}

void View::IncrementBoxY(int y, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.y += y;
  if (input_box_ind >= 0) mouse.hit     [input_box_ind].box.y += y;
}

void View::Draw() {
  if (child_box.data.empty()) Layout();
  child_box.Draw(root->gd, box.TopLeft());
}

void Widget::Button::Layout(Flow *flow, Font *f) {
  flow->SetFont(0);
  flow->SetFGColor(&Color::white);
  LayoutComplete(flow, f, flow->out->data[flow->AppendBox(box.w, box.h, image)].box);
}

void Widget::Button::LayoutBox(Flow *flow, Font *f, const Box &b) {
  flow->SetFont(0);
  flow->SetFGColor(&Color::white);
  if (image) flow->out->PushBack(b, flow->cur_attr, image, &drawbox_ind);
  LayoutComplete(flow, f, b);
}

void Widget::Button::LayoutComplete(Flow *flow, Font *f, const Box &b) {
  font = f;
  SetBox(b);
  if (solid) {
    flow->SetFont(0);
    flow->SetFGColor(solid);
    flow->out->PushBack(box, flow->cur_attr, Singleton<BoxFilled>::Get());
  }
  if (outline) {
    flow->SetFont(0);
    flow->SetFGColor(outline);
    flow->out->PushBack(box, flow->cur_attr, Singleton<BoxOutline>::Get());
  } else if (outline_topleft || outline_bottomright) {
    flow->SetFont(0);
    flow->cur_attr.line_width = outline_w;
    if (outline_topleft) {
      flow->SetFGColor(outline_topleft);
      flow->out->PushBack(box, flow->cur_attr, Singleton<BoxTopLeftOutline>::Get());
    }
    if (outline_bottomright) {
      flow->SetFGColor(outline_bottomright);
      flow->out->PushBack(box, flow->cur_attr, Singleton<BoxBottomRightOutline>::Get());
    }
    flow->cur_attr.line_width=0;
  }
  if (!text.empty()) {
    Box dim(box.Dimension()), tb;
    font->Size(text, &tb);
    textsize = tb.Dimension();

    point save_p = flow->p;
    flow->SetFont(font);
    flow->SetFGColor(0);
    flow->p = box.Position() + point(dim.centerX(textsize.x),
                                     v_align == VAlign::Center ? dim.centerY(textsize.y) : v_offset);
    flow->AppendText(text);
    flow->p = save_p;
  }
}

Widget::Slider::Slider(View *V, int f) : Interface(V), flag(f),
  menuicon(app->fonts->Get("MenuAtlas", "", 0, Color::white, Color::clear, 0)) {}

void Widget::Slider::LayoutAttached(const Box &w) {
  track = w;
  int aw = dot_size, ah = dot_size;
  bool flip = flag & Flag::Horizontal;
  if (flip) track.h = ah;
  else { track.x += track.w - aw; track.w = aw; }
  if (flag & Flag::NoCorner) {
    if (flip) track.w -= aw;
    else { track.h -= ah; track.y += ah; }
  }
  Layout(aw, ah, flip);
}

void Widget::Slider::Layout(int, int, bool flip) {
  if (outline_topleft && outline_bottomright) {
    track.DelBorder(Border(outline_w, 0, 0, outline_w));
    int attr_id = view->child_box.attr.GetAttrId(Drawable::Attr(NullPointer<Font>(), outline_topleft, nullptr, false, true, outline_w));
    view->child_box.PushBack(track, attr_id, Singleton<BoxTopLeftOutline>::Get());
    attr_id = view->child_box.attr.GetAttrId(Drawable::Attr(NullPointer<Font>(), outline_bottomright, nullptr, false, true, outline_w));
    view->child_box.PushBack(track, attr_id, Singleton<BoxBottomRightOutline>::Get());
    track.DelBorder(Border(0, outline_w, outline_w, 0));
  }

  Box arrow_down = track;
  if (flip) { arrow_down.w = track.h; track.x += track.h; }
  else      { arrow_down.h = track.w; track.y += track.w; }

  Box scroll_dot = arrow_down, arrow_up = track;
  if (flip) { arrow_up.w = track.h; track.w -= 2*track.h; arrow_up.x += track.w; }
  else      { arrow_up.h = track.w; track.h -= 2*track.w; arrow_up.y += track.h; }

  if (1) {
    int attr_id = view->child_box.attr.GetAttrId(Drawable::Attr(menuicon, &Color::white, nullptr, false, true));
    if (arrows) view->child_box.PushBack(arrow_up,   attr_id, menuicon ? menuicon->FindGlyph(flip ? 2 : 4) : 0);
    if (arrows) view->child_box.PushBack(arrow_down, attr_id, menuicon ? menuicon->FindGlyph(flip ? 3 : 1) : 0);
    if (1)      view->child_box.PushBack(scroll_dot, attr_id, menuicon ? menuicon->FindGlyph(           5) : 0, &drawbox_ind);

    if (1)      AddDragBox (scroll_dot, MouseController::CB(bind(&Slider::DragScrollDot, this)));
    if (arrows) AddClickBox(arrow_up,   MouseController::CB(bind(flip ? &Slider::ScrollDown : &Slider::ScrollUp,   this)));
    if (arrows) AddClickBox(arrow_down, MouseController::CB(bind(flip ? &Slider::ScrollUp   : &Slider::ScrollDown, this)));
  }
  Update(true);
}

void Widget::Slider::Update(bool force) {
  if (!app->input || !app->input->MouseButton1Down()) dragging = false;
  if (!dragging && !dirty && !force) return;
  bool flip = flag & Flag::Horizontal;
  if (dragging) {
    if (flip) scrolled = Clamp(    float(view->RelativePosition(view->root->mouse).x - track.x) / track.w, 0.0f, 1.0f);
    else      scrolled = Clamp(1 - float(view->RelativePosition(view->root->mouse).y - track.y) / track.h, 0.0f, 1.0f);
  }
  if (flip) { int aw = arrows ? dot_size : 0; view->UpdateBoxX(track.x          + int((track.w - aw) * scrolled), drawbox_ind, IndexOrDefault(hitbox, 0, -1)); }
  else      { int ah = arrows ? dot_size : 0; view->UpdateBoxY(track.top() - ah - int((track.h - ah) * scrolled), drawbox_ind, IndexOrDefault(hitbox, 0, -1)); }
  dirty = false;
}

float Widget::Slider::AddScrollDelta(float cur_val) {
  scrolled = Clamp(cur_val + ScrollDelta(), 0.0f, 1.0f);
  if (EqualChanged(&last_scrolled, scrolled)) dirty = 1;
  return scrolled;
}
  
void Widget::Slider::AttachContentBox(Box *b, Slider *vs, Slider *hs) {
  if (vs) { vs->LayoutAttached(*b); }
  if (hs) { hs->LayoutAttached(*b); MinusPlus(&b->h, &b->y, hs->dot_size); }
  if (vs) b->w -= vs->dot_size;
}

void Widget::Divider::ApplyConstraints() {
  if (max_size >= 0) size = min(size, max_size);
  if (min_size >  0) size = max(size, min_size);
}

void Widget::Divider::LayoutDivideTop(const Box &in, Box *top, Box *bottom, int offset) {
  ApplyConstraints();
  changed = 0;
  direction = 1;
  *top = *bottom = in;
  bottom->h -= (top->h = size);
  top->y = bottom->top();
  AddDragBox(Box(bottom->x, bottom->top()-1 + offset, bottom->w, 3),
             MouseController::CoordCB(bind(&Widget::Divider::DragCB, this, _1, _2, _3, _4)));
}

void Widget::Divider::LayoutDivideBottom(const Box &in, Box *top, Box *bottom, int offset) {
  ApplyConstraints();
  changed = 0;
  *top = *bottom = in;
  MinusPlus(&top->h, &top->y, size);
  bottom->h = size;
  AddDragBox(Box(bottom->x, bottom->top()-1 + offset, bottom->w, 3),
             MouseController::CoordCB(bind(&Widget::Divider::DragCB, this, _1, _2, _3, _4)));
}

void Widget::Divider::LayoutDivideLeft(const Box &in, Box *right, Box *left, int offset) {
  ApplyConstraints();
  changed = 0;
  direction = 1;
  *left = *right = in;
  MinusPlus(&right->w, &right->x, size);
  left->w = size;
  AddDragBox(Box(right->x-1, right->y + offset, 3, right->h),
             MouseController::CoordCB(bind(&Widget::Divider::DragCB, this, _1, _2, _3, _4)));
}

void Widget::Divider::LayoutDivideRight(const Box &in, Box *left, Box *right, int offset) {
  ApplyConstraints();
  changed = 0;
  *left = *right = in;
  left->w -= (right->w = size);
  right->x = left->right();
  AddDragBox(Box(right->x-1, right->y + offset, 3, right->h),
             MouseController::CoordCB(bind(&Widget::Divider::DragCB, this, _1, _2, _3, _4)));
}

void Widget::Divider::DragCB(int b, point p, point d, int down) {
  int v = (horizontal ? p.y : -p.x) * (direction ? -1 : 1);
  if (!changing && down) Assign(&start, &start_size, v, size);
  size = max(0, (start_size + v - start));
  Assign(&changing, &changed, bool(down), true);
}

int TextBox::Colors::SetDefaultAttr(int da) const {
  return Style::SetFGColorIndex(Style::SetBGColorIndex(da, background_index), normal_index);
}

const Drawable::Attr *TextBox::Style::GetAttr(int attr) const {
  if (colors) {
    bool italic = attr & Italic, bold = attr & Bold;
    int fg_index = GetFGColorIndex(attr), bg_index = GetBGColorIndex(attr);
    const Color *fg = colors->GetColor(italic ? colors->background_index : ((bold && fg_index == colors->normal_index) ? colors->bold_index : fg_index));
    const Color *bg = colors->GetColor(italic ? colors->normal_index     : bg_index);
    if (attr & Reverse) swap(fg, bg);
    last_attr.font = app->fonts->Change(font, font->size, *fg, *bg, font->flag);
    last_attr.bg = bg == colors->GetColor(colors->background_index) ? 0 : bg; // &font->bg;
  }
  last_attr.underline = attr & Underline;
  return &last_attr;
}

TextBox::Control::Control(TextBox::Line *P, View *V, const Box3 &b, string v, MouseControllerCallback cb) :
  Interface(V), box(b), val(move(v)), line(P) {
  AddClickBox(b, move(cb));
#ifndef LFL_MOBILE
  AddHoverBox(b, MouseController::CoordCB(bind(&Control::Hover, this, _1, _2, _3, _4)));
#endif
  del_hitbox = true;
}

void TextBox::LineData::AddControlsDelta(int delta_y) {
  for (auto &i : controls) {
    i.second->box += point(0, delta_y);
    for (auto &j : i.second->hitbox) i.second->view->IncrementBoxY(delta_y, -1, j);
  }
}

int TextBox::Line::Erase(int x, int l) {
  if (!(l = max(0, min(Size() - x, l)))) return 0;
  TokenProcessor<DrawableBox> update;
  bool token_processing = parent->token_processing;
  if (token_processing) {
    update.Init(data->glyphs, x, data->glyphs.Substr(x, l), l, bind(&TextBox::UpdateToken, parent, this, _1, _2, _3, &update));
    update.SetNewLineBoundaryConditions(!x ? update.nw : update.lbw,
                                        x + l == data->glyphs.Size() ? update.pw : update.lew);
    update.ProcessUpdate(data->glyphs);
  }
  data->glyphs.Erase(x, l, true);
  data->flow.p.x = data->glyphs.Position(data->glyphs.Size()).x;
  if (update.nw) update.ni -= l;
  if (token_processing) update.ProcessResult();
  return l;
}

template <class X> int TextBox::Line::InsertTextAt(int x, const StringPieceT<X> &v, int attr) {
  if (!v.size()) return 0;
  DrawableBoxArray b;
  b.attr.source = data->glyphs.attr.source;
  EncodeText(&b, data->glyphs.Position(x).x, v, attr);
  return b.Size() ? InsertTextAt(x, v, b) : 0;
}

template <class X> int TextBox::Line::InsertTextAt(int x, const StringPieceT<X> &v, const DrawableAnnotation &attr) {
  if (!v.size()) return 0;
  DrawableBoxArray b;
  b.attr.source = data->glyphs.attr.source;
  EncodeText(&b, data->glyphs.Position(x).x, v, attr, parent->default_attr);
  return b.Size() ? InsertTextAt(x, v, b) : 0;
}

template <class X> int TextBox::Line::InsertTextAt(int x, const StringPieceT<X> &v, const DrawableBoxArray &b) {
  int ret = b.Size();
  TokenProcessor<DrawableBox> update;
  bool token_processing = parent->token_processing, append = x == Size();
  if (token_processing) {
    update.Init(data->glyphs, x, b, 0, bind(&TextBox::UpdateToken, parent, this, _1, _2, _3, &update));
    update.SetNewLineBoundaryConditions(!x ? update.sw : update.lbw,
                                        x == data->glyphs.Size()-1 ? update.ew : update.lew);
    update.ProcessResult();
  }

  data->glyphs.InsertAt(x, b);
  data->flow.p.x = data->glyphs.Position(data->glyphs.Size()).x;
  if (!append && update.nw) update.ni += ret;

  if (token_processing) update.ProcessUpdate(data->glyphs);
  return ret;
}

template <class X> int TextBox::Line::OverwriteTextAt(int x, const StringPieceT<X> &v, int attr) {
  int size = Size(), pad = max(0, x + v.len - size), grow = 0;
  if (pad) data->flow.AppendText(basic_string<X>(pad, ' '), attr);

  DrawableBoxArray b;
  b.attr.source = data->glyphs.attr.source;
  EncodeText(&b, data->glyphs.Position(x).x, v, attr);
  if (!(size = b.Size())) return 0;
  if (size - v.len > 0 && (grow = max(0, x + size - Size())))
    data->flow.AppendText(basic_string<X>(grow, ' '), attr);
  ArrayPiece<DrawableBox> orun(&data->glyphs[x], size), nrun(&b[0], size);

  TokenProcessor<DrawableBox> update;
  bool token_processing = parent->token_processing;
  if (token_processing) {
    update.Init(data->glyphs, x, orun, size, bind(&TextBox::UpdateToken, parent, this, _1, _2, _3, &update));
    update.FindBoundaryConditions(nrun, &update.osw, &update.oew);
    update.SetNewLineBoundaryConditions(!x ? update.osw : update.lbw,
                                        x + size == data->glyphs.Size() ? update.oew : update.lew);
    update.ProcessUpdate(data->glyphs);
  }
  data->glyphs.OverwriteAt(x, b.data);
  data->flow.p.x = data->glyphs.Position(data->glyphs.Size()).x;
  if (token_processing) {
    update.PrepareOverwrite(nrun);
    update.ProcessUpdate(data->glyphs);
  }
  return size;
}

template <class X> int TextBox::Line::UpdateText(int x, const StringPieceT<X> &v, int attr, int max_width, bool *append_out, int mode) {
  bool append = 0, insert_mode = mode == -1 ? parent->insert_mode : mode;
  int size = Size(), ret = 0;
  if (insert_mode) {
    if (size < x)                 data->flow.AppendText(basic_string<X>(x - size, ' '), attr);
    if ((append = (Size() == x))) ret  = AppendText  (   v, attr);
    else                          ret  = InsertTextAt(x, v, attr);
    if (max_width)                       Erase(max_width);
  } else {
    data->flow.cur_attr.font = parent->style.font;
    ret = OverwriteTextAt(x, v, attr);
  }
  if (append_out) *append_out = append;
  return ret;
}

template int TextBox::Line::UpdateText<char>      (int x, const StringPiece   &v, int attr, int max_width, bool *append, int);
template int TextBox::Line::UpdateText<char16_t>  (int x, const String16Piece &v, int attr, int max_width, bool *append, int);
template int TextBox::Line::InsertTextAt<char>    (int x, const StringPiece   &v, int attr);
template int TextBox::Line::InsertTextAt<char16_t>(int x, const String16Piece &v, int attr);
template int TextBox::Line::InsertTextAt<char>    (int x, const StringPiece   &v, const DrawableAnnotation &attr);
template int TextBox::Line::InsertTextAt<char16_t>(int x, const String16Piece &v, const DrawableAnnotation &attr);
template int TextBox::Line::InsertTextAt<char>    (int x, const StringPiece   &v, const DrawableBoxArray &b);
template int TextBox::Line::InsertTextAt<char16_t>(int x, const String16Piece &v, const DrawableBoxArray &b);

void TextBox::Line::Layout(Box win, bool flush) {
  if (data->box.w == win.w && !flush) return;
  data->box = win;
  ScopedDeltaTracker<int> SWLT(cont ? &cont->wrapped_lines : 0, bind(&Line::Lines, this));
  DrawableBoxArray b;
  swap(b, data->glyphs);
  Clear();
  data->glyphs.attr.source = b.attr.source;
  data->flow.AppendBoxArrayText(b);
}

point TextBox::Line::Draw(point pos, int relayout_width, int g_offset, int g_len, const Box *scissor) {
  if (relayout_width >= 0) Layout(relayout_width);
  data->glyphs.Draw(parent->root->gd, (p = pos), g_offset, g_len, scissor);
  return p - point(0, parent->style.font->Height() + data->glyphs.height);
}

TextBox::Lines::Lines(TextBox *P, int N) : RingVector<Line>(N), parent(P), wrapped_lines(N),
  move_cb (bind(&Line::Move,  _1, _2)),
  movep_cb(bind(&Line::MoveP, _1, _2)) { for (auto &i : data) i.Init(parent, this); }
  
void TextBox::Lines::Resize(int s) {
  int d = s - ring.size;
  if (!d) return Clear();
  RingVector::Resize((wrapped_lines = s));
  if (d>0) for (auto i=data.begin()+ring.size-d, e=data.end(); i != e; ++i)
  { i->Init(parent, this); i->data->glyphs.attr.source = attr_source; }
}

void TextBox::Lines::SetAttrSource(Drawable::AttrSource *s) {
  attr_source = s;
  for (auto &i : data) i.data->glyphs.attr.source = s;
}

TextBox::Line *TextBox::Lines::InsertAt(int dest_line, int lines, int dont_move_last) {
  CHECK(lines);
  CHECK_LT(dest_line, 0);
  int clear_dir = 1;
  if (dest_line != -1) Move<Lines,Line>(*this, dest_line+lines, dest_line, -dest_line - lines - dont_move_last, movep_cb);
  else if ((clear_dir = -1)) { 
    ring.PushBack(lines);
    for (int scrollback_start_line = parent->GetFrameBuffer()->h / parent->style.font->Height(), i=0;
         i<lines; i++) (*this)[-scrollback_start_line-i-1].data->controls.clear();
  }
  for (int i=0; i<lines; i++) (*this)[dest_line + i*clear_dir].Clear();
  return &(*this)[dest_line];
}

TextBox::LinesFrameBuffer *TextBox::LinesFrameBuffer::Attach(TextBox::LinesFrameBuffer **last_fb) {
  if (*last_fb != this) fb.Attach();
  return *last_fb = this;
}

int TextBox::LinesFrameBuffer::SizeChanged(int W, int H, Font *font, const Color *bgc) {
  lines = max(only_grow ? lines : 0,
              partial_last_line ? RoundUp(float(H) / font->Height()) : (H / font->Height()));
  return RingFrameBuffer::SizeChanged(max(only_grow ? w : 0, W),
                                      max(only_grow ? h : 0, lines * font->Height()), font, bgc);
}

void TextBox::LinesFrameBuffer::Update(TextBox::Line *l, int flag) {
  if (!(flag & Flag::NoLayout)) l->Layout(wrap ? w : 0, flag & Flag::Flush);
  RingFrameBuffer::Update(l, Box(w, l->Lines() * font_height), paint_cb, true);
}

void TextBox::LinesFrameBuffer::OverwriteUpdate(Line *l, int xo, int wlo, int wll, int flag) {
  Update(l, flag);
}

int TextBox::LinesFrameBuffer::PushFrontAndUpdate(TextBox::Line *l, int xo, int wlo, int wll, int flag) {
  if (!(flag & Flag::NoLayout)) l->Layout(wrap ? w : 0, flag & Flag::Flush);
  int wl = max(0, l->Lines() - wlo), lh = (wll ? min(wll, wl) : wl) * font_height;
  if (!lh) return 0;
  Box b(xo, wl * font_height - lh, w, lh);
  return RingFrameBuffer::PushFrontAndUpdate(l, b, paint_cb, !(flag & Flag::NoVWrap)) / font_height;
}

int TextBox::LinesFrameBuffer::PushBackAndUpdate(TextBox::Line *l, int xo, int wlo, int wll, int flag) {
  if (!(flag & Flag::NoLayout)) l->Layout(wrap ? w : 0, flag & Flag::Flush);
  int wl = max(0, l->Lines() - wlo), lh = (wll ? min(wll, wl) : wl) * font_height;
  if (!lh) return 0;
  Box b(xo, wlo * font_height, w, lh);
  return RingFrameBuffer::PushBackAndUpdate(l, b, paint_cb, !(flag & Flag::NoVWrap)) / font_height;
}

void TextBox::LinesFrameBuffer::PushFrontAndUpdateOffset(TextBox::Line *l, int lo) {
  Update(l, RingFrameBuffer::BackPlus(point(0, font_height + lo * -font_height)));
  RingFrameBuffer::AdvancePixels(-l->Lines() * font_height);
}

void TextBox::LinesFrameBuffer::PushBackAndUpdateOffset(TextBox::Line *l, int lo) {
  Update(l, RingFrameBuffer::BackPlus(point(0, lo * font_height)));
  RingFrameBuffer::AdvancePixels(l->Lines() * font_height);
}

point TextBox::LinesFrameBuffer::Paint(TextBox::Line *l, point lp, const Box &b, int offset, int len) {
  auto p = l->parent;
  GraphicsContext gc(p->root->gd);
  Box sb(lp.x, lp.y - b.h, b.w, b.h);
  app->fonts->SelectFillColor(gc.gd);
  app->fonts->GetFillColor(p->bg_color ? *p->bg_color : Color::black)->Draw(&gc, sb);
  l->Draw(lp + b.Position(), -1, offset, len, &sb);
  return point(lp.x, lp.y-b.h);
}

void TextBox::LinesFrameBuffer::DrawAligned(const Box &b, point adjust) {
  RingFrameBuffer::Draw(align_top_or_bot ? (b.TopLeft() - point(0, h)) : b.Position(), adjust, false);
}

TextBox::LineUpdate::~LineUpdate() {
  if (!fb->lines || (flag & DontUpdate)) v->Layout(fb->wrap ? fb->w : 0);
  else if (flag & PushFront) { if (o) fb->PushFrontAndUpdateOffset(v,o); else fb->PushFrontAndUpdate(v); }
  else if (flag & PushBack)  { if (o) fb->PushBackAndUpdateOffset (v,o); else fb->PushBackAndUpdate (v); }
  else fb->Update(v);
}

TextBox::TextBox(Window *W, const FontRef &F, int LC) : View(W), style(F), cmd_fb(W?W->gd:0), cmd_last(LC) {
  if (style.font.Load()) cmd_line.GetAttrId(Drawable::Attr(style.font));
  layout.pad_wide_chars = 1;
  cmd_line.Init(this, 0);
}

point TextBox::RelativePosition(const point &in) const {
  auto fb = GetFrameBuffer();
  point p = in - (fb->align_top_or_bot ? (box.TopLeft() - point(0, fb->h)) : box.Position());
  if (clip) if (p.y < clip->bottom || p.y > box.h - clip->top) return p - point(0, box.h);
  return fb->BackPlus(p);
}

void TextBox::Enter() {
  string cmd = String::ToUTF8(Text16());
  AssignInput("");
  if (!cmd.empty()) AddHistory(cmd);
  if (!cmd.empty() || run_blank_cmd) Run(cmd);
  if (deactivate_on_enter) Deactivate();
}

void TextBox::SetColors(Colors *C) {
  style.colors = C;
  default_attr = style.colors->SetDefaultAttr(default_attr);
  bg_color = style.colors->GetColor(style.colors->background_index);
}

void TextBox::UpdateLineFB(Line *L, LinesFrameBuffer *fb, int flag) {
  fb->fb.Attach();
  ScopedClearColor scc(fb->fb.gd, bg_color);
  ScopedDrawMode drawmode(fb->fb.gd, DrawMode::_2D);
  fb->OverwriteUpdate(L, 0, 0, 0, flag);
  fb->fb.Release();
}

void TextBox::Draw(const Box &b) {
  if (cmd_fb.SizeChanged(b.w, b.h, style.font, bg_color)) {
    cmd_fb.p = point(0, style.font->Height());
    cmd_line.Draw(point(0, cmd_line.Lines() * style.font->Height()), cmd_fb.w);
    cmd_fb.SizeChangedDone();
  }
  // gd->PushColor();
  // gd->SetColor(cmd_color);
  cmd_fb.Draw(b.Position(), point());
  if (Active()) DrawCursor(b.Position() + cursor.p);
  // gd->PopColor();
}

void TextBox::DrawCursor(point p, Shader *shader) {
  float scale = shader ? X_or_Y(shader->scale, 1) : 1;
  GraphicsContext gc(root->gd);
  if (cursor.type == Cursor::Block) {
    gc.gd->EnableBlend();
    gc.gd->BlendMode(GraphicsDevice::OneMinusDstColor, GraphicsDevice::OneMinusSrcAlpha);
    gc.gd->FillColor(cmd_color);
    gc.DrawTexturedBox(Box(p.x, p.y - style.font->Height(), style.font->max_width, style.font->Height()) * scale);
    gc.gd->BlendMode(GraphicsDevice::SrcAlpha, GraphicsDevice::One);
    gc.gd->DisableBlend();
  } else {
    bool blinking = false;
    Time now = Now(), elapsed; 
    if (Active() && (elapsed = now - cursor.blink_begin) > cursor.blink_time) {
      if (elapsed > cursor.blink_time * 2) cursor.blink_begin = now;
      else blinking = true;
    }
    if (blinking) style.font->Draw("_", (p - point(0, style.font->Height())) * scale);
  }
}

void TextBox::UpdateToken(Line *L, int word_offset, int word_len, int update_type, const TokenProcessor<DrawableBox>*) {
  const DrawableBoxArray &glyphs = L->data->glyphs;
  CHECK_LE(word_offset + word_len, glyphs.Size());
  string text = DrawableBoxRun(&glyphs[word_offset], word_len).Text();
  UpdateLongToken(L, word_offset, L, word_offset+word_len-1, text, update_type);
}

void TextBox::UpdateLongToken(Line *BL, int beg_offset, Line *EL, int end_offset, const string &text, int update_type) {
  StringPiece textp(text);
  int offset = 0, url_offset = -1;
  for (; textp.len>1 && MatchingParens(*textp.buf, *textp.rbegin()); offset++, textp.buf++, textp.len -= 2) {}
  if (int punct = LengthChar(textp.buf, ispunct, textp.len)) { offset += punct; textp.buf += punct; }
  if      (textp.len > 7 && PrefixMatch(textp.buf, "http://"))  url_offset = offset + 7;
  else if (textp.len > 8 && PrefixMatch(textp.buf, "https://")) url_offset = offset + 8;
  if (url_offset >= 0) {
    if (update_type < 0) BL->data->controls.erase(beg_offset);
    else {
      string url = offset ? textp.str() : text;
      auto control = AddUrlBox(BL, beg_offset, EL, end_offset, url, [=](){ app->OpenSystemBrowser(url); });
      if (new_link_cb) new_link_cb(control);
    }
  }
}

shared_ptr<TextBox::Control> TextBox::AddUrlBox(Line *BL, int beg_offset, Line *EL, int end_offset, string val, Callback cb) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  int fb_h = fb->h, fh = style.font->Height(), adjust_y = BL->data->outside_scroll_region ? -fb_h : 0;
  Box gb = Box(BL->data->glyphs[beg_offset].box).SetY(BL->p.y - fh + adjust_y);
  Box ge = Box(EL->data->glyphs[end_offset].box).SetY(EL->p.y - fh + adjust_y);
  Box3 box(Box(fb->w, fb_h), gb.Position(), ge.Position() + point(ge.w, 0), fh, fh);
  return Insert(BL->data->controls, beg_offset, make_shared<Control>
                (BL, this, box, move(val), MouseControllerCallback(move(cb))))->second;
}

point TiledTextBox::PaintCB(Line *l, point lp, const Box &b) {
  GraphicsContext gc(root->gd);
  Box box = b + offset;
  tiles->ContextOpen();
  tiles->AddScissor(box);

  // tiles->SetAttr(&a);
  tiles->InitDrawBox(box.Position());
  tiles->DrawBox(&gc, Singleton<BoxFilled>::Get(), box);

  tiles->AddDrawableBoxArray(l->data->glyphs, lp + b.TopLeft() + offset);
  tiles->ContextClose();
  return point(lp.x, lp.y-b.h);
}

void TextBox::AddHistory(const string &cmd) {
  cmd_last.ring.PushBack(1);
  cmd_last[(cmd_last_ind = -1)] = cmd;
}

int TextBox::ReadHistory(const string &dir, const string &name) {
  StringFile history;
  VersionedFileName vfn(dir.c_str(), name.c_str(), "history");
  if (history.ReadVersioned(vfn) < 0) { ERROR("no ", name, " history"); return -1; }
  for (int i=0, l=history.Lines(); i<l; i++) AddHistory((*history.F)[l-1-i]);
  return 0;
}

int TextBox::WriteHistory(const string &dir, const string &name, const string &hdr) {
  if (!cmd_last.Size()) return 0;
  LocalFile history(dir + MatrixFile::Filename(name, "history", "string", 0), "w");
  if (!history.Opened()) return -1;
  MatrixFile::WriteHeader(&history, BaseName(history.fn), hdr, cmd_last.ring.count, 1);
  for (int i=0; i<cmd_last.ring.count; i++) StringFile::WriteRow(&history, cmd_last[-1-i]);
  return 0;
}

/* TextArea */

TextArea::TextArea(Window *W, const FontRef &F, int S, int LC) :
  TextBox(W, F, LC), line(this, S), line_fb(W->gd) {
  InitSelection();
#ifdef LFL_MOBILE
  drag_cb = [=](int, point p, point d, int down){
    if (selection.explicitly_initiated) return false;
    if (selection.Update(p, down)) selection.beg_click_time = Now();
    else if (!down && abs(selection.beg_click.y - selection.end_click.y) < 12
             && (Now() - selection.beg_click_time).count() > 200) {
      app->ShowSystemContextMenu
        (MenuItemVec{ 
         MenuItem{"", "Copy", [=](){ selection.explicitly_initiated = true; } },
         MenuItem{"", "Paste" }, MenuItem{"", "Keyboard" } });
      return true;
    }
#if 0
    if (d.y) {
      float sl = float(d.y) / style.font->Height();
      v_scrolled = Clamp(v_scrolled + sl * PercentOfLines(1), 0.0f, 1.0f);
      UpdateScrolled();
      if (!W->target_fps) app->scheduler.Wakeup(root);
    }
#endif
    return true;
  };
#endif
}

void TextArea::Write(const StringPiece &s, bool update_fb, bool release_fb) {
  if (!app->MainThread()) return app->RunInMainThread(bind(&TextArea::WriteCB, this, s.str(), update_fb, release_fb));
  write_last = Now();
  bool wrap = Wrap();
  int update_flag = LineFBPushBack(), sl;
  LinesFrameBuffer *fb = GetFrameBuffer();
  ScopedClearColor scc(fb->fb.gd, update_fb ? bg_color : NULL);
  ScopedDrawMode drawmode(fb->fb.gd, update_fb ? DrawMode::_2D : DrawMode::NullOp);
  if (update_fb && fb->lines) fb->fb.Attach();
  StringLineIter add_lines(s, StringLineIter::Flag::BlankLines);
  for (const char *add_line = add_lines.Next(); add_line; add_line = add_lines.Next()) {
    bool append = !write_newline && add_lines.first && add_lines.CurrentLength() && line.ring.count;
    Line *l = append ? &line[-1] : line.InsertAt(-1);
    if (!append) { l->Clear(); sl = 0; }
    else sl = l->Lines();
    if (write_timestamp) l->AppendText(StrCat(logtime(Now()), " "), cursor.attr);
    l->AppendText(StringPiece(add_line, add_lines.CurrentLength()), cursor.attr);
    if (int dl = l->Layout(wrap ? fb->w : 0) - sl) {
      if (start_line) { start_line += dl; end_line += dl; }
      if (scrolled_lines) v_scrolled = float(scrolled_lines += dl) / (WrappedLines()-1);
    }
    if (!update_fb || start_line) continue;
    LineUpdate(&line[-start_line-1], fb, (!append ? update_flag : 0));
  }
  if (update_fb && release_fb && fb->lines) fb->fb.Release();
}

void TextArea::SetDimension(int w, int h) {
  box.w = w;
  box.h = h;
  extra_height = line_fb.font_height ? (box.h % line_fb.font_height) : 0;
}

void TextArea::Resized(const Box &b, bool font_size_changed) {
  SetDimension(b.w, b.h);
  Box gui_box(0,-b.h,b.w,b.h*2);
  if (selection.gui_ind >= 0) UpdateBox(gui_box, -1, selection.gui_ind);
  for (auto &i : resize_gui_ind) UpdateBox(gui_box, -1, i);
  UpdateLines(last_v_scrolled, 0, 0, 0);
  UpdateCursor();
  Redraw(false, font_size_changed);
}

void TextArea::CheckResized(const Box &b) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  if (int c = fb->SizeChanged(b.w, b.h, style.font, bg_color))                 { Resized(b, c > 1); fb->SizeChangedDone(); }
  else if (needs_redraw && !(needs_redraw = 0)) { fb->BeginSizeChange(bg_color); Resized(b, c > 1); fb->SizeChangedDone(); }
  else if (box.w != b.w || box.h != b.h)                                       { SetDimension(b.w, b.h); UpdateCursor(); }
}

void TextArea::Redraw(bool attach, bool relayout) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  ScopedClearColor scc(fb->fb.gd, bg_color);
  ScopedDrawMode drawmode(fb->fb.gd, DrawMode::_2D);
  int fb_flag = LinesFrameBuffer::Flag::NoVWrap | (relayout ? LinesFrameBuffer::Flag::Flush : 0);
  int lines = start_line_adjust + skip_last_lines, font_height = style.font->Height();
  int (LinesFrameBuffer::*update_cb)(Line*, int, int, int, int) =
    reverse_line_fb ? &LinesFrameBuffer::PushBackAndUpdate
                    : &LinesFrameBuffer::PushFrontAndUpdate;
  fb->p = reverse_line_fb ? point(0, fb->h - start_line_adjust * font_height) 
                          : point(0, start_line_adjust * font_height);
  if (attach) { fb->fb.Attach(); fb->fb.gd->Clear(); }
  for (int i=start_line; i<line.ring.count && lines < fb->lines; i++) {
    Line *L = &line[-i-1];
    int py = L->p.y;
    lines += (fb->*update_cb)(L, -line_left, 0, fb->lines - lines, fb_flag);
    L->data->AddControlsDelta(L->p.y - py);
  }
  fb->p = point(0, fb->h);
  if (attach) { fb->scroll = v2(); fb->fb.Release(); }
}

int TextArea::UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  pair<int, int> old_first_line(start_line, -start_line_adjust), new_first_line, new_last_line;
  FlattenedArrayValues<TextBox::Lines>
    flattened_lines(&line, line.Size(), bind(&TextArea::LayoutBackLine, this, _1, _2));
  flattened_lines.AdvanceIter(&new_first_line, (scrolled_lines = RoundF(v_scrolled * (WrappedLines()-1))));
  flattened_lines.AdvanceIter(&(new_last_line = new_first_line), fb->lines-1);
  LayoutBackLine(&line, new_last_line.first);
  bool up = new_first_line < old_first_line;
  int dist = flattened_lines.Distance(new_first_line, old_first_line, fb->lines-1);
  if (first_offset) *first_offset = up ?  start_line_cutoff :  end_line_adjust;
  if (first_ind)    *first_ind    = up ? -start_line-1      : -end_line-1;
  if (first_len)    *first_len    = up ? -start_line_adjust :  end_line_cutoff;
  start_line        =  new_first_line.first;
  start_line_adjust = -new_first_line.second;
  end_line          =  new_last_line.first;
  end_line_adjust   =  new_last_line.second+1;
  end_line_cutoff   = line[  -end_line-1].Lines() - end_line_adjust;
  start_line_cutoff = line[-start_line-1].Lines() + start_line_adjust;
  return dist * (up ? -1 : 1);
}

void TextArea::UpdateScrolled() {
  int max_w = 4000;
  bool h_changed = Wrap() ? 0 : EqualChanged(&last_h_scrolled, h_scrolled);
  bool v_updated=0, v_changed = EqualChanged(&last_v_scrolled, v_scrolled);
  if (v_changed) {
    int first_ind = 0, first_offset = 0, first_len = 0;
    int dist = UpdateLines(v_scrolled, &first_ind, &first_offset, &first_len);
    if ((v_updated = dist)) {
      if (h_changed) UpdateHScrolled(RoundF(max_w * h_scrolled), false);
      if (1)         UpdateVScrolled(abs(dist), dist<0, first_ind, first_offset, first_len);
    }
  }
  if (h_changed && !v_updated) UpdateHScrolled(RoundF(max_w * h_scrolled), true);
}

void TextArea::UpdateHScrolled(int x, bool update_fb) {
  line_left = x;
  if (!update_fb) return;
  Redraw(true);
}

void TextArea::UpdateVScrolled(int dist, bool up, int ind, int first_offset, int first_len) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  if (dist >= fb->lines) Redraw(true);
  else {
    bool front = up == reverse_line_fb, decr = front != reverse_line_fb;
    int wl = 0, (LinesFrameBuffer::*update_cb)(Line*, int, int, int, int) =
      front ? &LinesFrameBuffer::PushFrontAndUpdate : &LinesFrameBuffer::PushBackAndUpdate;
    ScopedDrawMode drawmode(fb->fb.gd, DrawMode::_2D);
    ScopedClearColor scc(fb->fb.gd, bg_color);
    fb->fb.Attach();
    if (first_len)  wl += (fb->*update_cb)(&line[ind], -line_left, first_offset, min(dist, first_len), 0); 
    while (wl<dist) wl += (fb->*update_cb)(&line[decr ? --ind : ++ind], -line_left, 0, dist-wl, 0);
    fb->fb.Release();
  }
}

void TextArea::ChangeColors(Colors *C) {
  SetColors(C);
  Redraw(true, true);
}

void TextArea::Draw(const Box &b, int flag, Shader *shader) {
  float scale = 0;
  GraphicsContext gc(root->gd);
  if (shader) {
    scale = shader->scale;
    glShadertoyShader(gc.gd, shader);
    shader->SetUniform1i("iChannelFlip", 0);
    shader->SetUniform4f("iTargetBox", 0, 0,   XY_or_Y(scale, gc.gd->TextureDim(line_fb.w)), 
                                               XY_or_Y(scale, gc.gd->TextureDim(line_fb.h)));
    shader->SetUniform3f("iChannelResolution", XY_or_Y(scale, gc.gd->TextureDim(line_fb.w)), 
                                               XY_or_Y(scale, gc.gd->TextureDim(line_fb.h)), 1);
    shader->SetUniform2f("iChannelScroll", XY_or_Y(scale, -line_fb.scroll.x * line_fb.w),
                                           XY_or_Y(scale, -line_fb.scroll.y * line_fb.h - line_fb.align_top_or_bot * extra_height) - b.y);
    shader->SetUniform2f("iChannelModulus", line_fb.fb.tex.coord[Texture::maxx_coord_ind],
                                            line_fb.fb.tex.coord[Texture::maxy_coord_ind]);
  }
  int font_height = style.font->Height();
  LinesFrameBuffer *fb = GetFrameBuffer();
  if (flag & DrawFlag::CheckResized) CheckResized(b);
  if (clip) gc.gd->PushScissor(Box::DelBorder(b, *clip, scale));
  else if (extra_height) gc.gd->PushScissor(b);
  fb->DrawAligned(b, point(0, max(0, CommandLines()-1) * font_height));
  if (clip || extra_height) gc.gd->PopScissor();
  if (flag & DrawFlag::DrawCursor) DrawCursor(b.Position() + cursor.p);
  if (selection.gui_ind >= 0) box.SetPosition(b.Position());
  if (selection.changing) DrawSelection();
  if (!clip && hover_control) DrawHoverLink(b);
}

void TextArea::DrawHoverLink(const Box &b) {
  bool outside_scroll_region = hover_control->line->data->outside_scroll_region;
  int fb_h = line_fb.h;
  for (const auto &i : hover_control->box) {
    if (!i.w || !i.h) continue;
    point p = i.BottomLeft();
    p.y = outside_scroll_region ? (p.y + fb_h) : RingIndex::Wrap(p.y + line_fb.scroll.y * fb_h, fb_h);
    glLine(root->gd, p + point(b.x, b.y), point(i.BottomRight().x, p.y) + point(b.x, b.y), &Color::white);
  }
  if (hover_control_cb) hover_control_cb(hover_control);
}

bool TextArea::GetGlyphFromCoordsOffset(const point &p, Selection::Point *out, int sl, int sla) {
  int fh = style.font->Height(), targ = Clamp(reverse_line_fb ? ((box.h - p.y) / fh) : (p.y / fh), 0, line_fb.lines-1);
  for (int i = sl, lines = sla, ll; i < line.ring.count && lines < line_fb.lines; i++, lines += ll) {
    Line *L = &line[-i-1];
    if (lines + (ll = L->Lines()) <= targ) continue;
    L->data->glyphs.GetGlyphFromCoords(p, &out->char_ind, &out->glyph, reverse_line_fb ? targ-lines : lines+ll-targ-1);
    out->glyph.y = targ * fh;
    out->line_ind = i;
    return true;
  }
  return false;
}

void TextArea::InitSelection() {
  resize_gui_ind.push_back(mouse.AddWheelBox
    (Box(), MouseController::CoordCB
     ([=](int button, point p, point d, int down) { if (d.y > 0) ScrollUp(); else ScrollDown(); })));
  selection.gui_ind = mouse.AddDragBox
    (Box(), MouseController::CoordCB(bind(&TextArea::DragCB, this, _1, _2, _3, _4)));
}

void TextArea::DrawSelection() {
  GraphicsContext gc(root->gd);
  gc.gd->EnableBlend();
  gc.gd->FillColor(selection_color);
  gc.DrawBox3(selection.box, box.BottomLeft() + point(0, line_fb.align_top_or_bot ? extra_height : 0));
  gc.gd->SetColor(Color::white);
}

void TextArea::DragCB(int b, point p, point d, int down) {
  if (!line.ring.size) return;
  if (down) { if (!Active()) Activate(); }
  if (drag_cb && drag_cb(b, p, d, down)) return;

  Selection *s = &selection;
  point last_end_click = s->end_click;
  bool start = s->Update(root->mouse - box.BottomLeft() + point(line_left, 0), down);
  if (start) { 
    GetGlyphFromCoords(s->beg_click, &s->beg);
    s->Begin(v_scrolled);
    if (selection_cb) selection_cb(s->beg);
  } else {
#ifndef LFL_MOBILE
    int add_scroll = 0;
    if      (s->end_click.y>box.top()) add_scroll = max(0, s->end_click.y-max(box.top(), last_end_click.y));
    else if (s->end_click.y<box.y)     add_scroll = min(0, s->end_click.y-min(box.y,     last_end_click.y));
    if (add_scroll) {
      AddVScroll(reverse_line_fb ? -add_scroll : add_scroll);
      selection.scrolled += add_scroll;
    }
#endif
    GetGlyphFromCoords(s->end_click, &s->end);
  }

  bool swap = (s->beg.line_ind == s->end.line_ind && s->end.char_ind < s->beg.char_ind) ||
    (reverse_line_fb ? (s->end.line_ind < s->beg.line_ind) : (s->beg.line_ind < s->end.line_ind));
  if (!s->changing) {
    if (v_scrolled != selection.start_v_scrolled) {
      v_scrolled = selection.start_v_scrolled;
      UpdateScrolled();
      Redraw();
    }
    CopyText(swap ? s->end : s->beg, swap ? s->beg : s->end);
    selection.explicitly_initiated = false;
    s->box = Box3();
  } else {
    LinesFrameBuffer *fb = GetFrameBuffer();
    int fh = style.font->Height(), h = box.h;
    Box gb = swap ? s->end.glyph : s->beg.glyph;
    Box ge = swap ? s->beg.glyph : s->end.glyph;
    if (reverse_line_fb) { gb.y=h-gb.y-fh; ge.y=h-ge.y-fh; }
    if      (selection.scrolled > 0) ge.y -= selection.scrolled * fh;
    else if (selection.scrolled < 0) gb.y -= selection.scrolled * fh;
    s->box = Box3(Box(fb->w, h), gb.Position(), ge.BottomRight(), fh, fh);
  }
}

void TextArea::CopyText(const Selection::Point &beg, const Selection::Point &end) {
  string copy_text = CopyText(beg.line_ind, beg.char_ind, end.line_ind, end.char_ind, true);
  if (!copy_text.empty()) app->SetClipboardText(copy_text);
}

string TextArea::CopyText(int beg_line_ind, int beg_char_ind, int end_line_ind, int end_char_ind, bool add_nl) {
  String16 copy_text;
  bool one_line = beg_line_ind == end_line_ind;
  int d = reverse_line_fb ? 1 : -1;
  if (d < 0) { CHECK_LE(end_line_ind, beg_line_ind); }
  else       { CHECK_LE(beg_line_ind, end_line_ind); }

  for (int i = beg_line_ind; /**/; i += d) {
    Line *l = &line[-i-1];
    int len = l->Size();
    if (i == beg_line_ind) {
      if (!l->Size() || beg_char_ind < 0) len = -1;
      else {
        len = (one_line && end_char_ind >= 0) ? (end_char_ind + 1) : l->Size();
        copy_text += Substr(l->Text16(), beg_char_ind, max(0, len - beg_char_ind));
      }
    } else if (i == end_line_ind) {
      len = (end_char_ind >= 0) ? end_char_ind+1 : l->Size();
      copy_text += Substr(l->Text16(), 0, len);
    } else copy_text += l->Text16();

    if (add_nl && len == l->Size() && !l->data->wrapped) copy_text += String16(1, '\n');
    if (i == end_line_ind) break;
  }
  return String::ToUTF8(copy_text);
}

/* TextView */

int TextView::UpdateLines(float vs, int *first_ind, int *first_offset, int *first_len) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  bool width_changed = last_fb_width != fb->w, wrap = Wrap(), init = !wrapped_lines;
  if (width_changed) {
    last_fb_width = fb->w;
    if (wrap || init) UpdateMapping(fb->w, last_update_mapping_flag);
  }

  bool resized = (width_changed && wrap) || last_fb_lines != fb->lines;
  int new_first_line = RoundF(vs * (wrapped_lines - 1)), new_last_line = new_first_line + fb->lines;
  int dist = resized ? fb->lines : abs(new_first_line - last_first_line);
  if (!dist || Empty()) return 0;

  bool redraw = dist >= fb->lines;
  if (redraw) { line.Resize(fb->lines); line_fb.p=point(); fb_wrapped_lines=0; }

  bool up = !redraw && new_first_line < last_first_line;
  if (first_offset) *first_offset = up ?  start_line_cutoff : end_line_adjust;
  if (first_len)    *first_len    = up ? -start_line_adjust : end_line_cutoff;

  pair<int, int> read_lines;
  if (dist < fb->lines) {
    if (up) read_lines = pair<int, int>(new_first_line, dist);
    else    read_lines = pair<int, int>(new_last_line - dist, dist);
  } else    read_lines = pair<int, int>(new_first_line, fb->lines);

  bool head_read = new_first_line == read_lines.first;
  bool tail_read = new_last_line  == read_lines.first + read_lines.second;
  CHECK(head_read || tail_read);
  if (head_read && tail_read) CHECK(redraw);
  if (redraw) CHECK(head_read && tail_read);
  if (up) { CHECK(head_read); }
  else    { CHECK(tail_read); }

  bool short_read = !(head_read && tail_read), shorten_read = short_read && head_read && start_line_adjust;
  int past_end_lines = max(0, min(dist, read_lines.first + read_lines.second - wrapped_lines));
  read_lines.second = max(0, read_lines.second - past_end_lines);

  if      ( up && dist <= -start_line_adjust) { start_line_adjust += dist; read_lines.second=past_end_lines=0; }
  else if (!up && dist <=  end_line_cutoff)   { end_line_cutoff   -= dist; read_lines.second=past_end_lines=0; }

  last_fb_lines = fb->lines;
  last_first_line = new_first_line;
  int added = UpdateMappedLines(read_lines, up, head_read, tail_read, short_read, shorten_read);

  Line *L = 0;
  if (!up) for (int i=0; i<past_end_lines; i++, added++) { 
    int removed = (wrap && line.ring.Full()) ? line.Back()->Lines() : 0;
    (L = line.PushFront())->Clear();
    fb_wrapped_lines += L->Layout(wrap ? fb->w : 0, true) - removed;
    if (removed) line.ring.DecrementSize(1);
  }

  CHECK_LE(line.ring.count, line.ring.size);
  if (wrap && !redraw) {
    for (;;) {
      int ll = (L = up ? line.Front() : line.Back())->Lines();
      if (fb_wrapped_lines + (up ? start_line_adjust : -end_line_cutoff) - ll < fb->lines) break;
      fb_wrapped_lines -= ll;
      if (up) line.PopFront(1);
      else    line.PopBack (1);
    }
    if (up) end_line_cutoff   =  (fb_wrapped_lines + start_line_adjust - fb->lines);
    else    start_line_adjust = -(fb_wrapped_lines - end_line_cutoff   - fb->lines);
  }

  end_line_adjust   = line.Front()->Lines() - end_line_cutoff;
  start_line_cutoff = line.Back ()->Lines() + start_line_adjust;
  if (first_ind) *first_ind = up ? -added-1 : -line.Size()+added;
  return dist * (up ? -1 : 1);
}

/* PropertyView */

PropertyView::PropertyView(Window *W, const FontRef &F) : TextView(W, F),
  property_line(&NodeIndex::GetLines, &NodeIndex::GetString),
  menuicon_white(FontDesc("MenuAtlas", "", 0, Color::white, Color::clear, 0, 0, FontDesc::Engine::Atlas)),
  menuicon_black(FontDesc("MenuAtlas", "", 0, Color::black, Color::clear, 0, 0, FontDesc::Engine::Atlas)) {
  cursor_enabled = 0;
  cmd_color = Color(Color::black, .5);
  selection_cb = bind(&PropertyView::SelectionCB, this, _1);
  if (F->desc->bg.A()) bg_color = &F->desc->bg;
  Activate();
}

void PropertyView::Draw(const Box &b, int flag, Shader *shader) {
  TextArea::Draw(b, flag, shader);
  GraphicsContext gc(root->gd);
  int selected_line_no_offset = selected_line_no - last_first_line, fh = style.font->Height();
  if (Within(selected_line_no_offset, 0, GetFrameBuffer()->lines)) {
    gc.gd->EnableBlend();
    gc.gd->FillColor(selected_color);
    gc.DrawTexturedBox(Box(b.x, b.top()-(selected_line_no_offset+1)*fh, b.w, fh));
  }
}

void PropertyView::VisitExpandedChildren(Id id, const Node::Visitor &cb, int depth) {
  if (!id) return;
  auto n = GetNode(id);
  if (depth) cb(id, n, depth);
  if (n->expanded) for (auto i : n->child) VisitExpandedChildren(i, cb, 1+(depth ? depth : n->depth));
}

void PropertyView::UpdateMapping(int width, int flag) {
  last_update_mapping_flag = flag;
  wrapped_lines = 0;
  property_line.Clear();
  VisitExpandedChildren(root_id, [&](Id id, Node *n, int d)
   { if (d) GetNode(id)->depth = d-1; property_line.val.Insert(NodeIndex(id)); wrapped_lines++; });
  property_line.LoadFromSortedVal();
}

int PropertyView::UpdateMappedLines(pair<int, int> read_lines, bool up, bool head_read, bool tail_read,
                                    bool short_read, bool shorten_read) {
  if (!read_lines.second) return 0;
  int added = 0, last_read_line = read_lines.first + read_lines.second - 1, fh = style.font->Height();
  LineMap::ConstIterator lib, lie;
  CHECK((lib = property_line.SecondBound(read_lines.first+1)).val);
  for (lie = lib; lie.val && lie.key <= last_read_line; ++lie) {}

  Line *L;
  if (up) for (auto li=lie; li!=lib;       added++, fb_wrapped_lines++) LayoutLine(line.PushBack(), *(--li).val, line_fb.BackPlus(point(0,  fh*(added+1))));
  else    for (auto li=lib; li!=lie; ++li, added++, fb_wrapped_lines++) LayoutLine(line.PushFront(), *li.val,    line_fb.BackPlus(point(0, -fh*added), true));
  return added;
}

void PropertyView::LayoutLine(Line *L, const NodeIndex &ni, const point &p) {
  Node *n = GetNode(ni.id);
  L->Clear();
  Flow *flow = &L->data->flow;
  flow->layout.wrap_lines = 0;

  int fw = style.font->max_width;
  Box control(2 + fw * n->depth * 2, -style.font->ascender, fw, fw);
  if (n->control) {
    int control_attr_id = flow->out->attr.GetAttrId(Drawable::Attr(menuicon_black, NULL, NULL, false, true));
    flow->out->PushBack(control, control_attr_id, menuicon_black->FindGlyph(n->expanded ? 11 : 12));
    auto i = Insert(L->data->controls, 0, make_shared<Control>
                    (L, this, control + p, "", MouseControllerCallback(MouseController::CoordCB
                      (bind(&PropertyView::HandleNodeControlClicked, this, ni.id, _1, _2, _3, _4)), true)));
  }

  if (n->icon) {
    int icon_attr_id = flow->out->attr.GetAttrId(Drawable::Attr(menuicon_white, NULL, NULL, false, false));
    flow->out->PushBack(control + point(2*fw, 0), icon_attr_id, n->icon);
  }
  flow->p.x += control.right() + (n->icon ? 3*fw : fw);
  flow->SetFont(style.font);
  flow->AppendText(n->text);
}

void PropertyView::HandleNodeControlClicked(Id id, int b, point p, point d, int down) {
  if (!down) return;
  auto n = GetNode(id);
  auto fb = GetFrameBuffer();
  int h = fb->h, fh = style.font->Height(), depth = n->depth, added = 0;
  // int line_no = RingIndex::Wrap((h - fb->scroll.y * h - p.y), h) / fh, child_line_no = line_no + 1;
  int line_no = last_first_line + (box.top() - root->mouse.y) / fh, child_line_no = line_no + 1;
  if (line_no == selected_line_no) selected_line_no = -1;

  if ((n->expanded = !n->expanded)) {
    VisitExpandedChildren(id, [&](Id id, Node *n, int d){
      GetNode(id)->depth = d;
      property_line.Insert(child_line_no + added++, NodeIndex(id));
      wrapped_lines = AddWrappedLines(wrapped_lines, 1);
    });
  } else {
    while (auto li = property_line.SecondBound(child_line_no+1).val) {
      if (GetNode(li->id)->depth <= depth) break;
      HandleCollapsed(li->id);
      property_line.Erase(child_line_no);
      wrapped_lines = AddWrappedLines(wrapped_lines, -1);
    }
  }
  RefreshLines();
  Redraw();
}

void PropertyView::SelectionCB(const Selection::Point &p) {
  int line_no = last_first_line + (box.top() - root->mouse.y) / style.font->Height();
  if (line_no == selected_line_no && selected_line_clicked_cb)
    if (auto li = property_line.SecondBound(line_no+1).val) selected_line_clicked_cb(this, li->id);
  selected_line_no = line_no;
  if (line_selected_cb)
    if (auto li = property_line.SecondBound(line_no+1).val) line_selected_cb(this, li->id);
}

/* DirectoryTree */

void DirectoryTree::VisitExpandedChildren(Id id, const Node::Visitor &cb, int depth) {
  if (!id) return;
  auto n = GetNode(id);
  if (depth) cb(id, n, depth);
  else depth = n->depth;
  if (!n->expanded) return;
  string dirname = n->val;
  DirectoryIter dir(dirname, -1);
  while(const char *fn = dir.Next()) {
    string pn = StrCat(dirname, fn);
    VisitExpandedChildren(pn.back() == '/' ? AddDir(pn) : AddFile(pn), cb, 1+depth);
  }
}

/* Console */

Console::Console(Window *W, const FontRef &F, const Callback &C) : TextArea(W, F, 200, 50),
  animating_cb(C), full_height(root->height * .4) {
  line_fb.wrap = write_timestamp = 1;
  line_fb.align_top_or_bot = false;
  SetToggleKey(Key::Backquote);
  bg_color = &Color::clear;
  cursor.type = Cursor::Underline;
}

void Console::StartAnimating() {
  bool last_animating = animating;
  Time now = Now(), elapsed = now - anim_begin;
  anim_begin = now - (elapsed < anim_time ? anim_time - elapsed : Time(0));
  animating = (elapsed = now - anim_begin) < anim_time;
  if (animating && !last_animating && animating_cb) animating_cb();
}

void Console::Draw() {
  drawing = 1;
  Time now = Now(), elapsed;
  bool active = Active(), last_animating = animating;
  int h = active ? full_height : 0;
  if ((animating = (elapsed = now - anim_begin) < anim_time)) {
    if (active) h = full_height * (  double(elapsed.count())/anim_time.count());
    else        h = full_height * (1-double(elapsed.count())/anim_time.count());
  }
  if (!animating) {
    if (last_animating && animating_cb) animating_cb();
    if (!active) { drawing = 0; return; }
  }
  scissor = (animating && bottom_or_top) ? &(scissor_buf = Box(0, root->y, root->width, h)) : 0;
  Draw(Box(0, root->y + (bottom_or_top ? 0 : root->height - h), root->width, full_height));
}

void Console::Draw(const Box &b, int flag, Shader *shader) {
  GraphicsContext gc(root->gd);
  if (scissor) gc.gd->PushScissor(*scissor);
  int fh = style.font->Height();
  Box tb(b.x, b.y + fh, b.w, b.h - fh);
  if (blend) gc.gd->EnableBlend(); 
  else       gc.gd->DisableBlend();
  if (blend) { gc.gd->FillColor(color); gc.DrawTexturedBox(b); gc.gd->SetColor(Color::white); }
  TextBox::Draw(b);
  TextArea::Draw(tb, flag & ~DrawFlag::DrawCursor, shader);
  if (scissor) gc.gd->PopScissor();
}

/* Dialog */

Dialog::Dialog(Window *W, float w, float h, int flag) : View(W),
  color(85,85,85,220), title_gradient{Color(127,0,0), Color(0,0,127), Color(0,0,178), Color(208,0,127)},
  font(FontDesc(FLAGS_font, "", 14, Color(Color::white,.8), Color::clear, FLAGS_font_flag)),
  menuicon(FontDesc("MenuAtlas", "", 0, Color::white, Color::clear, 0)), deleted_cb([=]{ deleted=true; })
{
  box = root->Box().center(root->Box(w, h));
  fullscreen = flag & Flag::Fullscreen;
  Activate();
}

void Dialog::LayoutTabbed(int tab, const Box &b, const point &tab_dim, MouseController *oc, DrawableBoxArray *od) {
  fullscreen = tabbed = true;
  box = b;
  Layout();
  LayoutTitle(Box(box.x+tab*tab_dim.x, 0, tab_dim.x, tab_dim.y), oc, od);
}

void Dialog::Layout() {
  ResetView();
  int fh = font->Height();
  content = Box(0, -box.h, box.w, box.h + ((fullscreen && !tabbed) ? 0 : -fh));
  if (fullscreen) return;
  LayoutTitle(Box(box.w, fh), &mouse, &child_box);
  LayoutReshapeControls(box.Dimension(), &mouse);
}

void Dialog::LayoutTitle(const Box &b, MouseController *oc, DrawableBoxArray *od) {
  title = Box(0, -b.h, b.w, b.h);
  Box close = Box(b.w-10, title.top()-10, 10, 10);
  oc->AddClickBox(close + b.Position(), deleted_cb);

  int attr_id = od->attr.GetAttrId(Drawable::Attr(menuicon));
  od->PushBack(close, attr_id, menuicon ? menuicon->FindGlyph(0) : 0);
  if (title_text.size()) {
    Box title_text_size;
    font->Size(title_text, &title_text_size);
    font->Shape(title_text, Box(title.centerX(title_text_size.w), title.centerY(title_text_size.h), 0, 0), od);
  }
}

void Dialog::LayoutReshapeControls(const point &dim, MouseController *oc) {
  resize_left   = Box(0,       -dim.y, 3, dim.y);
  resize_right  = Box(dim.x-3, -dim.y, 3, dim.y);
  resize_bottom = Box(0,       -dim.y, dim.x, 3);

  oc->AddClickBox(resize_left,   MouseController::CB(bind(&Dialog::Reshape, this, &resizing_left)));
  oc->AddClickBox(resize_right,  MouseController::CB(bind(&Dialog::Reshape, this, &resizing_right)));
  oc->AddClickBox(resize_bottom, MouseController::CB(bind(&Dialog::Reshape, this, &resizing_bottom)));
  oc->AddClickBox(title,         MouseController::CB(bind(&Dialog::Reshape, this, &moving)));
}

bool Dialog::HandleReshape(Box *outline) {
  bool resizing = resizing_left || resizing_right || resizing_bottom;
  if (moving) box.SetPosition(win_start + root->mouse - mouse_start);
  if (moving || resizing) *outline = box;
  if (resizing_left)   MinusPlus(&outline->x, &outline->w, max(-outline->w + min_width,  int(mouse_start.x - root->mouse.x)));
  if (resizing_bottom) MinusPlus(&outline->y, &outline->h, max(-outline->h + min_height, int(mouse_start.y - root->mouse.y)));
  if (resizing_right)  outline->w += max(-outline->w + min_width, int(root->mouse.x - mouse_start.x));
  if (!app->input->MouseButton1Down()) {
    if (resizing) { box = *outline; Layout(); }
    moving = resizing_left = resizing_right = resizing_bottom = 0;
  }
  return moving || resizing_left || resizing_right || resizing_bottom;
}

void Dialog::Draw() {
  GraphicsContext gc(root->gd);
  if (fullscreen) { child_box.Draw(gc.gd, box.TopLeft()); return; }

  Box outline;
  bool reshaping = HandleReshape(&outline);
  if (child_box.data.empty() && !reshaping) Layout();

  gc.gd->FillColor(color);
  gc.DrawTexturedBox(box);
  gc.gd->SetColor(Color::white);
  if (reshaping) BoxOutline().Draw(&gc, outline);

  gc.gd->EnableVertexColor();
  DrawGradient(box.TopLeft());
  gc.gd->DisableVertexColor();

  child_box.Draw(gc.gd, box.TopLeft());
}

void DialogTab::Draw(GraphicsDevice *gd, const Box &b, const point &tab_dim, const vector<DialogTab> &t) {
  gd->FillColor(Color::grey70);
  GraphicsContext::DrawTexturedBox1(gd, Box(b.x, b.y+b.h-tab_dim.y, b.w, tab_dim.y));
  gd->EnableVertexColor();
  for (int i=0, l=t.size(); i<l; ++i) t[i].dialog->DrawGradient(b.TopLeft() + point(i*tab_dim.x, 0));
  gd->DisableVertexColor();
  gd->EnableTexture();
  for (int i=0, l=t.size(); i<l; ++i) t[i].child_box.Draw(gd, b.TopLeft() + point(i*tab_dim.x, 0));
}

TabbedDialogInterface::TabbedDialogInterface(View *v, const point &d) : view(v), tab_dim(d) {}
void TabbedDialogInterface::Layout() {
  for (auto b=tab_list.begin(), e=tab_list.end(), t=b; t != e; ++t) {
    int tab_no = t-b;
    t->dialog->LayoutTabbed(tab_no, box, tab_dim, &view->mouse, &t->child_box);
    view->mouse.AddClickBox(t->dialog->title + point(box.x + tab_no*tab_dim.x, 0),
                            MouseController::CB(bind(&TabbedDialogInterface::SelectTabIndex, this, tab_no)));
  }
  if (tab_list.size() && tab_list.size() * tab_dim.x > box.w) {
    auto t = tab_list.begin();
    int fh = t->dialog->font->Height();
    Font *menuicon = t->dialog->menuicon.ptr;
    int attr_id = view->child_box.attr.GetAttrId(Drawable::Attr(menuicon, &Color::white, nullptr, false, true));
    view->child_box.PushBack(Box(box.x,          -fh, fh, fh), attr_id, menuicon ? menuicon->FindGlyph(6) : 0);
    view->child_box.PushBack(Box(box.right()-fh, -fh, fh, fh), attr_id, menuicon ? menuicon->FindGlyph(7) : 0);
  }
}

void MessageBoxDialog::Layout() {
  Dialog::Layout();
  Box dim(box.Dimension());
  Flow flow(&dim, font, &child_box);
  flow.p = point(dim.centerX(message_size.w), dim.centerY(message_size.h) - dim.h);
  flow.AppendText(message);
}

void MessageBoxDialog::Draw() {
  Scissor scissor(root->gd, box);
  Dialog::Draw();
}

SliderDialog::SliderDialog(Window *w, const string &t, const SliderDialog::UpdatedCB &cb, float scrolled, float total, float inc) :
  Dialog(w, .33, .09), updated(cb), slider(this, Widget::Slider::Flag::Horizontal) {
  title_text = t;
  slider.scrolled = scrolled;
  slider.doc_height = total;
  slider.increment = inc;
}

FlagSliderDialog::FlagSliderDialog(Window *w, const string &fn, float total, float inc) :
  SliderDialog(w, fn, bind(&FlagSliderDialog::Updated, this, _1), atof(Singleton<FlagMap>::Get()->Get(fn)) / total, total, inc),
  flag_name(fn), flag_map(Singleton<FlagMap>::Get()) {}

void Dialog::MessageBox(const string &n) {
  app->ReleaseMouseFocus();
  app->focused->AddDialog(make_unique<MessageBoxDialog>(app->focused, n));
}

void Dialog::TextureBox(const string &n) {
  app->ReleaseMouseFocus();
  app->focused->AddDialog(make_unique<TextureBoxDialog>(app->focused, n));
}

#ifdef LFL_BOOST
}; // namespace LFL
#include <boost/graph/fruchterman_reingold.hpp>
#include <boost/graph/random_layout.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/simple_point.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/progress.hpp>
#include <boost/shared_ptr.hpp>

using namespace boost;

struct BoostForceDirectedLayout {
  typedef adjacency_list<listS, vecS, undirectedS, property<vertex_name_t, std::string> > Graph;
  typedef graph_traits<Graph>::vertex_descriptor Vertex;

  typedef boost::rectangle_topology<> topology_type;
  typedef topology_type::point_type point_type;
  typedef std::vector<point_type> PositionVec;
  typedef iterator_property_map<PositionVec::iterator, property_map<Graph, vertex_index_t>::type> PositionMap;

  Graph g;
  vector<Vertex> v;

  void Clear() { g.clear(); v.clear(); }
  void AssignVertexPositionToTargetCenter(const HelperGUI *gui, PositionMap *position, int vertex_offset) {
    for (int i = 0; i < gui->label.size(); ++i) {
      (*position)[v[i*2+vertex_offset]][0] = gui->label[i].target_center.x;
      (*position)[v[i*2+vertex_offset]][1] = gui->label[i].target_center.y;
    }
  }
  void AssignVertexPositionToLabelCenter(const HelperGUI *gui, PositionMap *position, int vertex_offset) {
    for (int i = 0; i < gui->label.size(); ++i) {
      (*position)[v[i*2+vertex_offset]][0] = gui->label[i].label_center.x;
      (*position)[v[i*2+vertex_offset]][1] = gui->label[i].label_center.y;
    }
  }
  void Layout(HelperGUI *gui) {
    Clear();

    for (int i = 0; i < gui->label.size()*2; ++i) v.push_back(add_vertex(StringPrintf("%d", i), g));
    for (int i = 0; i < gui->label.size()  ; ++i) add_edge(v[i*2], v[i*2+1], g);

    PositionVec position_vec(num_vertices(g));
    PositionMap position(position_vec.begin(), get(vertex_index, g));

    minstd_rand gen;
    topology_type topo(gen, 0, 0, gui->root->width, gui->root->height);
    if (0) random_graph_layout(g, position, topo);
    else AssignVertexPositionToLabelCenter(gui, &position, 1);

    for (int i = 0; i < 300; i++) {
      AssignVertexPositionToTargetCenter(gui, &position, 0);
      fruchterman_reingold_force_directed_layout(g, position, topo, cooling(linear_cooling<double>(1)));
    }

    for (int i = 0; i < gui->label.size(); ++i) {
      HelperGUI::Label *l = &gui->label[i];
      l->label_center.x = position[v[i*2+1]][0];
      l->label_center.y = position[v[i*2+1]][1];
      l->AssignLabelBox();
    }
  }
};
namespace LFL {
#endif // LFL_BOOST

void HelperView::ForceDirectedLayout() {
#ifdef LFL_BOOST
  BoostForceDirectedLayout().Layout(this);
#endif
}

HelperView::Label::Label(const Box &w, const string &d, int h, Font *f, const point &p) :
  target(w), target_center(target.center()), hint(h), description(d) {
  label_center = target_center;
  if      (h == Hint::UP   || h == Hint::UPLEFT   || h == Hint::UPRIGHT)   label_center.y += p.y;
  else if (h == Hint::DOWN || h == Hint::DOWNLEFT || h == Hint::DOWNRIGHT) label_center.y -= p.y;
  if      (h == Hint::UPRIGHT || h == Hint::DOWNRIGHT)                     label_center.x += p.x;
  else if (h == Hint::UPLEFT  || h == Hint::DOWNLEFT)                      label_center.x -= p.x;
  f->Size(description.c_str(), &label);
  AssignLabelBox();
}

void HelperView::Draw() {
  GraphicsContext gc(root->gd);
  for (auto i = label.begin(); i != label.end(); ++i) {
    glLine(root->gd, point(i->label_center.x, i->label_center.y),
                     point(i->target_center.x, i->target_center.y), &font->fg);
    gc.gd->FillColor(Color::black);
    gc.DrawTexturedBox(Box::AddBorder(i->label, 4, 0));
    font->Draw(i->description, point(i->label.x, i->label.y));
  }
}

}; // namespace LFL
