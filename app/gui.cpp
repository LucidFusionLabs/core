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

void GUI::UpdateBox(const Box &b, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box = b;
  if (input_box_ind >= 0) mouse.hit     [input_box_ind].box = b;
}

void GUI::UpdateBoxX(int x, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.x = x;
  if (input_box_ind >= 0) mouse.hit     [input_box_ind].box.x = x;
}

void GUI::UpdateBoxY(int y, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.y = y;
  if (input_box_ind >= 0) mouse.hit     [input_box_ind].box.y = y;
}

void GUI::IncrementBoxY(int y, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.y += y;
  if (input_box_ind >= 0) mouse.hit     [input_box_ind].box.y += y;
}

void GUI::Draw() {
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

Widget::Slider::Slider(GUI *Gui, int f) : Interface(Gui), flag(f),
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
    int attr_id = gui->child_box.attr.GetAttrId(Drawable::Attr(NullPointer<Font>(), outline_topleft, nullptr, false, true, outline_w));
    gui->child_box.PushBack(track, attr_id, Singleton<BoxTopLeftOutline>::Get());
    attr_id = gui->child_box.attr.GetAttrId(Drawable::Attr(NullPointer<Font>(), outline_bottomright, nullptr, false, true, outline_w));
    gui->child_box.PushBack(track, attr_id, Singleton<BoxBottomRightOutline>::Get());
    track.DelBorder(Border(0, outline_w, outline_w, 0));
  }

  Box arrow_down = track;
  if (flip) { arrow_down.w = track.h; track.x += track.h; }
  else      { arrow_down.h = track.w; track.y += track.w; }

  Box scroll_dot = arrow_down, arrow_up = track;
  if (flip) { arrow_up.w = track.h; track.w -= 2*track.h; arrow_up.x += track.w; }
  else      { arrow_up.h = track.w; track.h -= 2*track.w; arrow_up.y += track.h; }

  if (1) {
    int attr_id = gui->child_box.attr.GetAttrId(Drawable::Attr(menuicon, &Color::white, nullptr, false, true));
    gui->child_box.PushBack(arrow_up,   attr_id, menuicon ? menuicon->FindGlyph(flip ? 2 : 4) : 0);
    gui->child_box.PushBack(arrow_down, attr_id, menuicon ? menuicon->FindGlyph(flip ? 3 : 1) : 0);
    gui->child_box.PushBack(scroll_dot, attr_id, menuicon ? menuicon->FindGlyph(           5) : 0, &drawbox_ind);

    AddDragBox (scroll_dot, MouseController::CB(bind(&Slider::DragScrollDot, this)));
    AddClickBox(arrow_up,   MouseController::CB(bind(flip ? &Slider::ScrollDown : &Slider::ScrollUp,   this)));
    AddClickBox(arrow_down, MouseController::CB(bind(flip ? &Slider::ScrollUp   : &Slider::ScrollDown, this)));
  }
  Update(true);
}

void Widget::Slider::Update(bool force) {
  if (!app->input || !app->input->MouseButton1Down()) dragging = false;
  if (!dragging && !dirty && !force) return;
  bool flip = flag & Flag::Horizontal;
  int aw = dot_size, ah = dot_size;
  if (dragging) {
    if (flip) scrolled = Clamp(    float(gui->RelativePosition(gui->root->mouse).x - track.x) / track.w, 0.0f, 1.0f);
    else      scrolled = Clamp(1 - float(gui->RelativePosition(gui->root->mouse).y - track.y) / track.h, 0.0f, 1.0f);
  }
  if (flip) gui->UpdateBoxX(track.x          + int((track.w - aw) * scrolled), drawbox_ind, IndexOrDefault(hitbox, 0, -1));
  else      gui->UpdateBoxY(track.top() - ah - int((track.h - ah) * scrolled), drawbox_ind, IndexOrDefault(hitbox, 0, -1));
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

TextBox::Control::Control(TextBox::Line *P, GUI *G, const Box3 &b, string v, MouseControllerCallback cb) :
  Interface(G), box(b), val(move(v)), line(P) {
  AddClickBox(b, move(cb));
#ifndef LFL_MOBILE
  AddHoverBox(b, MouseController::CoordCB(bind(&Control::Hover, this, _1, _2, _3, _4)));
#endif
  del_hitbox = true;
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

TextBox::TextBox(Window *W, const FontRef &F, int LC) : GUI(W), style(F), cmd_last(LC), cmd_fb(W?W->gd:0) {
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

void TextBox::DrawCursor(point p) {
  GraphicsContext gc(root->gd);
  if (cursor.type == Cursor::Block) {
    gc.gd->EnableBlend();
    gc.gd->BlendMode(GraphicsDevice::OneMinusDstColor, GraphicsDevice::OneMinusSrcAlpha);
    gc.gd->FillColor(cmd_color);
    gc.DrawTexturedBox(Box(p.x, p.y - style.font->Height(), style.font->max_width, style.font->Height()));
    gc.gd->BlendMode(GraphicsDevice::SrcAlpha, GraphicsDevice::One);
    gc.gd->DisableBlend();
  } else {
    bool blinking = false;
    Time now = Now(), elapsed; 
    if (Active() && (elapsed = now - cursor.blink_begin) > cursor.blink_time) {
      if (elapsed > cursor.blink_time * 2) cursor.blink_begin = now;
      else blinking = true;
    }
    if (blinking) style.font->Draw("_", p - point(0, style.font->Height()));
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
    if (selection.Update(p, down)) app->ShowSystemContextMenu
      (MenuItemVec{ MenuItem{"", "Copy", [=](){ selection.explicitly_initiated = true; } } });
    if (d.y) {
      float sl = float(d.y) / style.font->Height();
      v_scrolled = Clamp(v_scrolled + sl * PercentOfLines(1), 0.0f, 1.0f);
      UpdateScrolled();
      if (!W->target_fps) app->scheduler.Wakeup(root);
    }
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
  if (selection.gui_ind >= 0) UpdateBox(Box(0,-b.h,b.w,b.h*2), -1, selection.gui_ind);
  if (context_gui_ind   >= 0) UpdateBox(Box(0,-b.h,b.w,b.h*2), -1, context_gui_ind);
  if (wheel_gui_ind     >= 0) UpdateBox(Box(0,-b.h,b.w,b.h*2), -1, wheel_gui_ind);
  UpdateLines(last_v_scrolled, 0, 0, 0);
  UpdateCursor();
  Redraw(false, font_size_changed);
}

void TextArea::CheckResized(const Box &b) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  if (int c = fb->SizeChanged(b.w, b.h, style.font, bg_color)) { Resized(b, c > 1); fb->SizeChangedDone(); }
  else if (box.w != b.w || box.h != b.h) { SetDimension(b.w, b.h); UpdateCursor(); }
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
  for (int i=start_line; i<line.ring.count && lines < fb->lines; i++)
    lines += (fb->*update_cb)(&line[-i-1], -line_left, 0, fb->lines - lines, fb_flag);
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
  GraphicsContext gc(root->gd);
  if (shader) {
    float scale = shader->scale;
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
  if (clip) gc.gd->PushScissor(Box::DelBorder(b, *clip));
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
  wheel_gui_ind = mouse.AddWheelBox
    (Box(), MouseController::CoordCB
     ([=](int button, point p, point d, int down) { if (d.y > 0) ScrollUp(); else ScrollDown(); }));
  selection.gui_ind = mouse.AddDragBox
    (Box(), MouseController::CoordCB(bind(&TextArea::DragCB, this, _1, _2, _3, _4)));
}

void TextArea::DrawSelection() {
  GraphicsContext gc(root->gd);
  gc.gd->EnableBlend();
  gc.gd->FillColor(selection_color);
  gc.DrawBox3(selection.box, box.BottomLeft());
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
    int add_scroll = 0;
    if      (s->end_click.y>box.top()) add_scroll = max(0, s->end_click.y-max(box.top(), last_end_click.y));
    else if (s->end_click.y<box.y)     add_scroll = min(0, s->end_click.y-min(box.y,     last_end_click.y));
    if (add_scroll) {
      AddVScroll(reverse_line_fb ? -add_scroll : add_scroll);
      selection.scrolled += add_scroll;
    }
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

/* Editor */

Editor::SyntaxColors::SyntaxColors(const string &n, const vector<Rule> &rules) :
  name(n), color(3, Color()) {
  normal_index=0; bold_index=1; background_index=2;
  map<Color, pair<int, int>> seen_fg, seen_bg;
  for (auto &r : rules) {
    auto fs = &seen_fg[r.fg], bs = &seen_bg[r.bg];
    if (!fs->second++) fs->first = PushBackIndex(color, r.fg);
    if (!bs->second++) bs->first = PushBackIndex(color, r.bg);
    style[r.name] = Style::SetColorIndex(r.style, fs->first, bs->first);
  }
  auto i = style.find("Normal");
  if (i != style.end()) {
    color[normal_index] = color[bold_index] = color[Style::GetFGColorIndex(i->second)];
    color[background_index]                 = color[Style::GetBGColorIndex(i->second)];
  } else {
    ERROR("SyntaxColors ", name, " no Normal");
    int max_fg=0, max_bg=0, max_fg_ind=0, max_bg_ind=0;
    for (auto &f : seen_fg) if (Max(&max_fg, f.second.second)) max_fg_ind = f.second.first;
    for (auto &b : seen_bg) if (Max(&max_bg, b.second.second)) max_bg_ind = b.second.first;
    color[normal_index] = color[bold_index] = color[max_fg_ind];
    color[background_index]                 = color[max_bg_ind];
  }
  for (auto &c : color) if (c == Color::clear) c = color[background_index];
}

// base16 by chriskempson
Editor::Base16DefaultDarkSyntaxColors::Base16DefaultDarkSyntaxColors() :
  SyntaxColors("Base16DefaultDarkSyntaxColors", {
    { "Boolean",      Color("dc9656"), Color("181818"), 0 },
    { "Character",    Color("ab4642"), Color("181818"), 0 },
    { "Comment",      Color("585858"), Color("181818"), 0 },
    { "Conditional",  Color("ba8baf"), Color("181818"), 0 },
    { "Constant",     Color("dc9656"), Color("181818"), 0 },
    { "Define",       Color("ba8baf"), Color("181818"), 0 },
    { "Delimiter",    Color("a16946"), Color("181818"), 0 },
    { "Float",        Color("dc9656"), Color("181818"), 0 },
    { "Function",     Color("7cafc2"), Color("181818"), 0 },
    { "Identifier",   Color("ab4642"), Color("181818"), 0 },
    { "Include",      Color("7cafc2"), Color("181818"), 0 },
    { "Keyword",      Color("ba8baf"), Color("181818"), 0 },
    { "Label",        Color("f7ca88"), Color("181818"), 0 },
    { "Normal",       Color("d8d8d8"), Color("181818"), 0 },
    { "Number",       Color("dc9656"), Color("181818"), 0 },
    { "Operator",     Color("d8d8d8"), Color("181818"), 0 },
    { "PreCondition", Color("ba8baf"), Color("181818"), 0 },
    { "PreProc",      Color("f7ca88"), Color("181818"), 0 },
    { "Repeat",       Color("f7ca88"), Color("181818"), 0 },
    { "Special",      Color("86c1b9"), Color("181818"), 0 },
    { "SpecialChar",  Color("a16946"), Color("181818"), 0 },
    { "Statement",    Color("ab4642"), Color("181818"), 0 },
    { "StatusLine",   Color("b8b8b8"), Color("383838"), 0 },
    { "StorageClass", Color("f7ca88"), Color("181818"), 0 },
    { "String",       Color("a1b56c"), Color("181818"), 0 },
    { "Structure",    Color("ba8baf"), Color("181818"), 0 },
    { "Tag",          Color("f7ca88"), Color("181818"), 0 },
    { "Todo",         Color("f7ca88"), Color("282828"), 0 },
    { "Type",         Color("f7ca88"), Color("181818"), 0 },
    { "Typedef",      Color("f7ca88"), Color("181818"), 0 }
  }) {}

Editor::~Editor() {}
Editor::Editor(Window *W, const FontRef &F, File *I) : TextView(W, F),
  file_line(&LineOffset::GetLines, &LineOffset::GetString),
  annotation_cb([](const LineMap::ConstIterator&, const String16&, bool, int cs, int){ static DrawableAnnotation a; return cs ? nullptr : &a; }) {
  cmd_color = Color(Color::black, .5);
  edits.free_func = [](String16 *v) { v->clear(); };
  selection_cb = bind(&Editor::SelectionCB, this, _1);
  if (I) Init(I);
}

const String16 *Editor::ReadLine(const Editor::LineMap::Iterator &i, String16 *buf) {
  if (int e = (max(0, -i.val->file_size))) return &edits[e-1];
  *buf = String::ToUTF16(file->ReadString(i.val->file_offset, i.val->file_size));
  return buf;
}

void Editor::SetWrapMode(const string &n) {
  bool lines_mode = (n == "lines"), words_mode = (n == "words"), should_wrap = lines_mode || words_mode;
  SetShouldWrap(should_wrap, should_wrap ? words_mode : layout.word_break);
}

void Editor::SetShouldWrap(bool w, bool word_break) {
  layout.word_break = word_break;
  line_fb.wrap = w;
  Reload();
  Redraw(true);
}

void Editor::HistUp() {
  if (cursor.i.y > 0) { if (!--cursor.i.y && start_line_adjust) return AddVScroll(start_line_adjust); }
  else {
    cursor.i.y = 0;
    LineOffset *lo = 0;
    if (Wrap()) lo = (--file_line.SecondBound(cursor_start_line_number+1)).val;
    return AddVScroll((lo ? -lo->wrapped_lines : -1) + start_line_adjust);
  }
  UpdateCursorLine();
  UpdateCursor();
}

void Editor::HistDown() {
  bool last=0;
  if (cursor_start_line_number_offset + cursor_glyphs->Lines() < line_fb.lines) {
    last = ++cursor.i.y >= line.Size()-1;
  } else {
    AddVScroll(end_line_cutoff + 1);
    cursor.i.y = line.Size() - 1;
    last = 1; 
  }
  UpdateCursorLine();
  UpdateCursor();
  if (last) UpdateCursorX(cursor.i.x);
}

void Editor::SelectionCB(const Selection::Point &p) {
  cursor.i.y = p.line_ind;
  UpdateCursorLine();
  cursor.i.x = p.char_ind >= 0 ? p.char_ind : CursorGlyphsSize();
  UpdateCursor();
}

void Editor::UpdateMapping(int width, int flag) {
  wrapped_lines = syntax_parsed_anchor = syntax_parsed_line_index = 0;
  last_update_mapping_flag = flag;
  file_line.Clear();
  file->Reset();
  NextRecordReader nr(file.get());
  int ind = 0, offset = 0, wrap = Wrap(), ll;
  for (const char *l = nr.NextLineRaw(&offset); l; l = nr.NextLineRaw(&offset)) {
    wrapped_lines += (ll = wrap ? TextArea::style.font->Lines(l, width, layout.word_break) : 1);
    file_line.val.Insert(LineOffset(offset, nr.record_len, ll, flag ? ind++ : -1));
  }
  file_line.LoadFromSortedVal();
  // XXX update cursor position
}

int Editor::UpdateMappedLines(pair<int, int> read_lines, bool up, bool head_read, bool tail_read,
                              bool short_read, bool shorten_read) {
  bool wrap = Wrap();
  int read_len = 0;
  IOVector rv;
  LineMap::Iterator lib, lie;
  if (read_lines.second) {
    CHECK((lib = file_line.SecondBound(read_lines.first+1)).val);
    if (wrap) {
      if (head_read) start_line_adjust = min(0, lib.key - last_first_line);
      if (short_read && tail_read && end_line_cutoff) ++lib;
    }
    int last_read_line = read_lines.first + read_lines.second - 1;
    for (lie = lib; lie.val && lie.key <= last_read_line; ++lie) {
      auto v = lie.val;
      if (shorten_read && !(lie.key + v->wrapped_lines <= last_read_line)) break;
      if (v->file_size >= 0) read_len += rv.Append({v->file_offset, v->file_size+1});
    }
    if (wrap && tail_read) {
      LineMap::Iterator i = lie.ind ? lie : file_line.RBegin();
      end_line_cutoff = max(0, (--i).key + i.val->wrapped_lines - last_first_line - last_fb_lines);
    }
  }

  string buf(read_len, 0);
  if (read_len) CHECK_EQ(buf.size(), file->ReadIOV(&buf[0], rv.data(), rv.size()));

  int added = 0, bo = 0, width = wrap ? GetFrameBuffer()->w : 0, ll, l, e;
  bool first_line = true;
  Line *L = 0;
  if (up) {
    vector<tuple<String16, const String16*, const DrawableAnnotation*, LineMap::Iterator>> line_data;
    for (auto li = lie; li != lib; bo += l + !e) {
      l = (e = max(0, -(--li).val->file_size)) ? 0 : li.val->file_size;
      if (e) line_data.emplace_back(String16(), &edits[e-1], nullptr, li);
      else   line_data.emplace_back(String::ToUTF16(buf.data()+read_len-bo-l-1, l), nullptr, nullptr, li);
    }
    for (auto ldi = line_data.rbegin(), lde = line_data.rend(); ldi != lde; ++ldi) {
      auto t = tuple_get<1>(*ldi);
      tuple_get<2>(*ldi) = annotation_cb(tuple_get<3>(*ldi), t ? *t : tuple_get<0>(*ldi), first_line, 0, 0);
      first_line = 0;
    }
    added = line_data.size();
    for (auto &ld : line_data) {
      auto t = tuple_get<1>(ld);
      int removed = (wrap && line.ring.Full()) ? line.Front()->Lines() : 0;
      (L = line.PushBack())->AssignText(t ? *t : tuple_get<0>(ld), *tuple_get<2>(ld));
      fb_wrapped_lines += (ll = L->Layout(width, true)) - removed;
      if (removed) line.ring.DecrementSize(1);
      CHECK_EQ(tuple_get<3>(ld).val->wrapped_lines, ll);
    }
  } else {
    for (auto li = lib; li != lie; ++li, bo += l + !e, added++, first_line = 0) {
      l = (e = max(0, -li.val->file_size)) ? 0 : li.val->file_size;
      int removed = (wrap && line.ring.Full()) ? line.Back()->Lines() : 0;
      if (e) { const auto &t = edits[e-1];                 (L = line.PushFront())->AssignText(t, *annotation_cb(li, t, first_line, 0, 0)); }
      else   { auto t = String::ToUTF16(buf.data()+bo, l); (L = line.PushFront())->AssignText(t, *annotation_cb(li, t, first_line, 0, 0)); }
      fb_wrapped_lines += (ll = L->Layout(width, true)) - removed;
      if (removed) line.ring.DecrementSize(1);
      CHECK_EQ(li.val->wrapped_lines, ll);
    }
  }
  return added;
}

int Editor::UpdateLines(float vs, int *first_ind, int *first_offset, int *first_len) {
  if (!opened) return 0;
  int ret = TextView::UpdateLines(vs, first_ind, first_offset, first_len);
  if (!file_line.size()) {
    line.Resize((wrapped_lines = 1));
    line.PushBack()->Clear();
    file_line.Insert(1, LineOffset());
  }
  UpdateCursorLine();
  UpdateCursor();
  return ret;
}

void Editor::UpdateCursorLine() {
  int capped_y = min(cursor.i.y, min(line.Size(), wrapped_lines - last_first_line) - 1);
  cursor_start_line_number = last_first_line + start_line_adjust;
  if (!Wrap()) cursor_start_line_number += (cursor.i.y = capped_y);
  else for (cursor.i.y = 0; cursor.i.y < capped_y && cursor_start_line_number < wrapped_lines-1; cursor.i.y++)
    cursor_start_line_number += line[-1-cursor.i.y].Lines();

  cursor_glyphs = &line[-1-cursor.i.y];
  cursor.i.x = min(cursor.i.x, CursorGlyphsSize());
  cursor_start_line_number_offset = cursor_start_line_number - last_first_line;
  if (!Wrap()) { CHECK_EQ(cursor.i.y, cursor_start_line_number_offset); }

  auto it = file_line.SecondBound(cursor_start_line_number+1);
  if (!it.ind) FATAL("missing line number: ", cursor_start_line_number, " size=", file_line.size(), ", wl=", wrapped_lines); 
  cursor_line_index = it.GetIndex();
  cursor_anchor = it.GetAnchor();
  cursor_offset = it.val;
}

void Editor::UpdateCursor() {
  cursor.p = !cursor_glyphs ? point() : cursor_glyphs->data->glyphs.Position(cursor.i.x) +
    point(0, box.h - cursor_start_line_number_offset * style.font->Height());
}

void Editor::UpdateCursorX(int x) {
  cursor.i.x = min(x, CursorGlyphsSize());
  if (!Wrap()) return UpdateCursor();
  int wl = cursor_start_line_number_offset + cursor_glyphs->data->glyphs.GetLineFromIndex(x);
  if (wl >= line_fb.lines) { AddVScroll(wl-line_fb.lines+1); cursor.i=point(x, line.Size()-1); }
  else if (wl < 0)         { AddVScroll(wl);                 cursor.i=point(x, 0); }
  UpdateCursorLine();
  UpdateCursor();
}

int Editor::CursorLinesChanged(const String16 &b, int add_lines) {
  cursor_glyphs->AssignText(b);
  int ll = cursor_glyphs->Layout(GetFrameBuffer()->w, true);
  int d = ChangedDiff(&cursor_offset->wrapped_lines, ll);
  if (d) { CHECK_EQ(cursor_offset, file_line.Update(cursor_start_line_number, *cursor_offset)); }
  if (int a = d + add_lines) wrapped_lines = AddWrappedLines(wrapped_lines, a);
  return d;
}

int Editor::ModifyCursorLine() {
  CHECK(cursor_offset);
  if (cursor_offset->file_size >= 0) cursor_offset->file_size = -edits.Insert(cursor_glyphs->Text16())-1;
  return -cursor_offset->file_size-1;
}

void Editor::Modify(char16_t c, bool erase, bool undo_or_redo) {
  if (!cursor_glyphs || !cursor_offset) return;
  int edit_id = ModifyCursorLine();
  String16 *b = &edits[edit_id];
  CHECK_LE(cursor.i.x, cursor_glyphs->Size());
  CHECK_LE(cursor.i.x, b->size());
  CHECK_EQ(cursor_offset->wrapped_lines, cursor_glyphs->Lines());
  bool wrap = Wrap(), erase_line = erase && !cursor.i.x;
  if (!cursor_start_line_number && erase_line) return;

  if (!undo_or_redo && !erase_line)
    RecordModify(point(cursor.i.x, cursor_line_index), erase, !erase ? c : (*b)[cursor.i.x-1]);
  modified = Now();
  if (modified_cb) modified_cb();
  if (!erase_line && cursor_line_index < syntax_parsed_line_index) MarkCursorLineFirstDirty();
  cursor_offset->colored = false;

  if (erase_line) {
    file_line.Erase(cursor_start_line_number);
    wrapped_lines = AddWrappedLines(wrapped_lines, -cursor_glyphs->Lines());
    HistUp();
    if (cursor_line_index < syntax_parsed_line_index) MarkCursorLineFirstDirty();
    b = &edits[ModifyCursorLine()];
    bool chomped = !undo_or_redo && b->size() && b->back() == '\r';
    if (chomped) b->pop_back();
    int x = b->size();
    b->append(edits[edit_id]);
    edits.Erase(edit_id);
    if (wrap) CursorLinesChanged(*b);
    UpdateCursorX(x);
    RefreshLines();
    if (!undo_or_redo) {
      if (1)       RecordModify(point(cursor.i.x + chomped, cursor_line_index), true, '\n');
      if (chomped) RecordModify(point(cursor.i.x + 1,       cursor_line_index), true, '\r');
    }
    Redraw(true);
  } else if (!erase && c == '\n') {
    String16 a = b->substr(cursor.i.x);
    b->resize(cursor.i.x);
    if (wrap) CursorLinesChanged(*b, 1);
    else wrapped_lines = AddWrappedLines(wrapped_lines, 1);
    file_line.Insert(cursor_start_line_number + cursor_offset->wrapped_lines, LineOffset());
    if (wrap) RefreshLines();
    cursor.i.x = 0;
    HistDown();
    CHECK_EQ(0, cursor_offset->file_offset);
    CHECK_EQ(0, cursor_offset->file_size);
    CHECK_EQ(1, cursor_offset->wrapped_lines);
    if (wrap) CursorLinesChanged(a);
    swap(edits[ModifyCursorLine()], a);
    RefreshLines();
    if (newline_cb) newline_cb();
    Redraw(true);
  } else {
    if (erase) b->erase(cursor.i.x-1, 1);
    else       b->insert(cursor.i.x, 1, c);
    const DrawableAnnotation *annotation = nullptr;
    if (!wrap && !(annotation = annotation_cb
                   (file_line.GetAnchorIter(cursor_anchor), *b, 0, erase ? -1 : 1, cursor.i.x))) {
      int attr_id = cursor.i.x ? cursor_glyphs->data->glyphs[cursor.i.x-1].attr_id : default_attr;
      if (erase)  cursor_glyphs->Erase          (cursor.i.x-1, 1);
      else if (0) cursor_glyphs->OverwriteTextAt(cursor.i.x, String16(1, c), attr_id);
      else        cursor_glyphs->InsertTextAt   (cursor.i.x, String16(1, c), attr_id);
      UpdateLineFB(cursor_glyphs, GetFrameBuffer(), 0);
    } 
    if (erase) CursorLeft();
    if (wrap) {
      CursorLinesChanged(*b);
      RefreshLines();
      Redraw(true);
    } else if (annotation) {
      // XXX keep reparsing till reach same signature or last displayed line, then only redraw those
      RefreshLines();
      Redraw(true);
    }
    if (!erase) CursorRight();
  }
}

int Editor::Save() {
  IOVector rv;
  for (LineMap::ConstIterator i = file_line.Begin(); i.ind; ++i)
    rv.Append({ i.val->file_offset, i.val->file_size + (i.val->file_size >= 0) });
  int ret = file->Rewrite(ArrayPiece<IOVec>(rv.data(), rv.size()),
    function<string(int)>([&](int i){ return String::ToUTF8(edits.data[i]) + "\n"; }));
  Reload();
  Redraw(true);
  saved_version_number = version_number;
  return ret;
}

int Editor::SaveTo(File *out) {
  IOVector rv;
  for (LineMap::ConstIterator i = file_line.Begin(); i.ind; ++i)
    rv.Append({ i.val->file_offset, i.val->file_size + (i.val->file_size >= 0) });
  return file->Rewrite(ArrayPiece<IOVec>(rv.data(), rv.size()),
    function<string(int)>([&](int i){ return String::ToUTF8(edits.data[i]) + "\n"; }), out);
}

bool Editor::CacheModifiedText(bool force) {
  if (!force && version_number == saved_version_number) return false;
  if (version_number != cached_text_version_number) {
    cached_text = make_shared<BufferFile>(string());
    SaveTo(cached_text.get());
    cached_text_version_number = version_number;
  }
  return true;
}

void Editor::RecordModify(const point &p, bool erase, char16_t c) {
  if (version_number.offset < version.size()) { version_number.major++; version.resize(version_number.offset); }
  if (version.size() && c != '\n') {
    Modification &m = version.back();
    if (m.p.y == p.y && m.erase == erase && !(m.data.size() == 1 && m.data[0] == '\n') &&
        m.p.x + m.data.size() * (erase ? -1 : 1) == p.x) { m.data.append(1, c); return; }
  }
  version.push_back({ p, erase, String16(1, c) });
  version_number.offset = version.size();
}

bool Editor::WalkUndo(bool backwards) {
  if (backwards) { if (!version_number.offset)                  return false; }
  else           { if (version_number.offset >= version.size()) return false; }

  const Modification &m = version[backwards ? --version_number.offset : version_number.offset++];
  bool nl = m.data.size() == 1 && m.data[0] == '\n';
  point target = (nl && (backwards != m.erase)) ? point(0, m.p.y + 1) :
    point(m.p.x + ((backwards && !nl) ? (m.erase ? -m.data.size() : m.data.size()) : 0), m.p.y);
  CHECK(ScrollTo(target.y, target.x));
  CHECK_EQ(target.y, cursor_line_index);
  CHECK_EQ(target.x, cursor.i.x);

  if (backwards) for (auto i=m.data.rbegin(), e=m.data.rend(); i!=e; ++i) Modify(*i, !m.erase, true);
  else           for (auto i=m.data.begin(),  e=m.data.end();  i!=e; ++i) Modify(*i,  m.erase, true);
  return true;
}

bool Editor::ScrollTo(int line_index, int x) {
  int target_offset = min(line_index, last_fb_lines/2), wrapped_line_no = -1;
  if (!Wrap()) {
    if (line_index < 0 || line_index >= file_line.size()) return false;
    wrapped_line_no = line_index;
  } else {
    LineMap::ConstIterator li = file_line.FindIndex(line_index);
    if (!li.val) return false;
    wrapped_line_no = li.GetBegKey();
  }

  cursor.i.y = 0;
  if (wrapped_line_no >= last_first_line &&
      wrapped_line_no < last_first_line + last_fb_lines) target_offset = wrapped_line_no - last_first_line;
  else SetVScroll(wrapped_line_no - target_offset);

  if (!Wrap()) cursor.i.y = target_offset;
  else for (int i=start_line, o=start_line_adjust; o<target_offset; i++, cursor.i.y++) o += line[-1-i].Lines();

  UpdateCursorLine();
  UpdateCursorX(x);
  return true;
}
 
/* Terminal */

#ifdef  LFL_TERMINAL_DEBUG
#define TerminalDebug(...) ERRORf(__VA_ARGS__)
#define TerminalTrace(...) DebugPrintf("%s", StrCat(logtime(Now()), " ", StringPrintf(__VA_ARGS__)).c_str())
#else
#define TerminalDebug(...)
#define TerminalTrace(...)
#endif

Terminal::StandardVGAColors::StandardVGAColors() { 
  c[0] = Color(  0,   0,   0);  c[ 8] = Color( 85,  85,  85);
  c[1] = Color(170,   0,   0);  c[ 9] = Color(255,  85,  85);
  c[2] = Color(  0, 170,   0);  c[10] = Color( 85, 255,  85);
  c[3] = Color(170,  85,   0);  c[11] = Color(255, 255,  85);
  c[4] = Color(  0,   0, 170);  c[12] = Color( 85,  85, 255);
  c[5] = Color(170,   0, 170);  c[13] = Color(255,  85, 255);
  c[6] = Color(  0, 170, 170);  c[14] = Color( 85, 255, 255);
  c[7] = Color(170, 170, 170);  c[15] = Color(255, 255, 255);
  c[normal_index]     = c[7];
  c[bold_index]       = c[15];
  c[background_index] = c[0];
}

/// Solarized palette by Ethan Schoonover
Terminal::SolarizedDarkColors::SolarizedDarkColors() { 
  c[0] = Color(  7,  54,  66); /*base02*/   c[ 8] = Color(  0,  43,  54); /*base03*/
  c[1] = Color(220,  50,  47); /*red*/      c[ 9] = Color(203,  75,  22); /*orange*/
  c[2] = Color(133, 153,   0); /*green*/    c[10] = Color( 88, 110, 117); /*base01*/
  c[3] = Color(181, 137,   0); /*yellow*/   c[11] = Color(101, 123, 131); /*base00*/
  c[4] = Color( 38, 139, 210); /*blue*/     c[12] = Color(131, 148, 150); /*base0*/
  c[5] = Color(211,  54, 130); /*magenta*/  c[13] = Color(108, 113, 196); /*violet*/
  c[6] = Color( 42, 161, 152); /*cyan*/     c[14] = Color(147, 161, 161); /*base1*/
  c[7] = Color(238, 232, 213); /*base2*/    c[15] = Color(253, 246, 227); /*base3*/
  c[normal_index]     = c[12];
  c[bold_index]       = c[14];
  c[background_index] = c[8];
}

Terminal::SolarizedLightColors::SolarizedLightColors() { 
  c[0] = Color(238, 232, 213); /*base2*/    c[ 8] = Color(253, 246, 227); /*base3*/
  c[1] = Color(220,  50,  47); /*red*/      c[ 9] = Color(203,  75,  22); /*orange*/
  c[2] = Color(133, 153,   0); /*green*/    c[10] = Color(147, 161, 161); /*base1*/
  c[3] = Color(181, 137,   0); /*yellow*/   c[11] = Color(131, 148, 150); /*base0*/
  c[4] = Color( 38, 139, 210); /*blue*/     c[12] = Color(101, 123, 131); /*base00*/
  c[5] = Color(211,  54, 130); /*magenta*/  c[13] = Color(108, 113, 196); /*violet*/
  c[6] = Color( 42, 161, 152); /*cyan*/     c[14] = Color( 88, 110, 117); /*base01*/
  c[7] = Color(  7,  54,  66); /*base02*/   c[15] = Color(  0,  43,  54); /*base03*/
  c[normal_index]     = c[12];
  c[bold_index]       = c[14];
  c[background_index] = c[8];
}

Terminal::Terminal(ByteSink *O, Window *W, const FontRef &F, const point &dim) :
  TextArea(W, F, 200, 0), sink(O), fb_cb(bind(&Terminal::GetFrameBuffer, this, _1)) {
  CHECK(style.font->fixed_width || (style.font->flag & FontDesc::Mono));
  wrap_lines = write_newline = insert_mode = 0;
  line.SetAttrSource(&style);
  line_fb.align_top_or_bot = cmd_fb.align_top_or_bot = false;
  line_fb.partial_last_line = cmd_fb.partial_last_line = false;
  SetColors(Singleton<SolarizedDarkColors>::Get());
  cursor.attr = default_attr;
  token_processing = 1;
  cmd_prefix = "";
  SetTerminalDimension(dim.x, dim.y);
  Activate();
}

int Terminal::GetFrameY(int y) const { return (term_height - y + 1) * style.font->Height(); }
int Terminal::GetCursorY(int y) const { return GetFrameY(y) + (line_fb.align_top_or_bot ? extra_height : 0); }
int Terminal::GetCursorX(int x, int y) const {
  const Line *l = GetTermLine(y);
  return x <= l->Size() ? l->data->glyphs.Position(x-1).x : ((x-1) * style.font->FixedWidth());
}

Terminal::LinesFrameBuffer *Terminal::GetFrameBuffer(const Line *l) {
  int i = line.IndexOf(l);
  return ((-i-1 < start_line || term_height+i < skip_last_lines) ? cmd_fb : line_fb).Attach(&last_fb);
}

void Terminal::Resized(const Box &b, bool font_size_changed) {
  int old_term_width = term_width, old_term_height = term_height;
  SetTerminalDimension(b.w / style.font->FixedWidth(), b.h / style.font->Height());
  TerminalDebug("Resized %d, %d <- %d, %d", term_width, term_height, old_term_width, old_term_height);
  bool grid_changed = term_width != old_term_width || term_height != old_term_height;
  if (grid_changed || first_resize) if (sink) sink->IOCtlWindowSize(term_width, term_height); 

  int height_dy = term_height - old_term_height;
  if (!line_fb.only_grow) {
    if      (height_dy > 0) TextArea::Write(string(height_dy, '\n'), 0);
    else if (height_dy < 0 && term_cursor.y < old_term_height) line.PopBack(-height_dy);
  } else {
    if      (height_dy > 0) { for (int i=0; i<height_dy; i++) line.PushFront(); }
    else if (height_dy < 0) line.PopFront(-height_dy);
    term_cursor.y += height_dy;
  }

  term_cursor.x = min(term_cursor.x, max(1, term_width));
  term_cursor.y = min(term_cursor.y, max(1, term_height));
  TextArea::Resized(b, font_size_changed);
  if (clip) clip = UpdateClipBorder();
  ResizedLeftoverRegion(b.w, b.h);
}

void Terminal::ResizedLeftoverRegion(int w, int h, bool update_fb) {
  if (!cmd_fb.SizeChanged(w, h, style.font, bg_color)) return;
  if (update_fb) {
    for (int i=0; i<start_line;      i++) MoveToOrFromScrollRegion(&cmd_fb, &line[-i-1],           point(0, GetFrameY(term_height-i)), LinesFrameBuffer::Flag::Flush);
    for (int i=0; i<skip_last_lines; i++) MoveToOrFromScrollRegion(&cmd_fb, &line[-term_height+i], point(0, GetFrameY(i+1)),           LinesFrameBuffer::Flag::Flush);
  }
  cmd_fb.SizeChangedDone();
  last_fb = 0;
}

void Terminal::MoveToOrFromScrollRegion(TextBox::LinesFrameBuffer *fb, TextBox::Line *l, const point &p, int flag) {
  int plpy = l->p.y;
  fb->Update(l, p, flag);
  bool last_outside_scroll_region = l->data->outside_scroll_region;
  if ((l->data->outside_scroll_region = fb != &line_fb) != last_outside_scroll_region) {
    int delta_y = plpy - l->p.y + line_fb.h * (l->data->outside_scroll_region ? -1 : 1);
    for (auto &i : l->data->controls) {
      i.second->box += point(0, delta_y);
      for (auto &j : i.second->hitbox) i.second->gui->IncrementBoxY(delta_y, -1, j);
    }
  }
}

void Terminal::SetScrollRegion(int b, int e, bool release_fb) {
  if (b<0 || e<0 || e>term_height || b>e) { TerminalDebug("%d-%d outside 1-%d", b, e, term_height); return; }
  int prev_region_beg = scroll_region_beg, prev_region_end = scroll_region_end, font_height = style.font->Height();
  scroll_region_beg = b;
  scroll_region_end = e;
  bool no_region = !scroll_region_beg || !scroll_region_end || (scroll_region_beg == 1 && scroll_region_end == term_height);
  skip_last_lines = no_region ? 0 : scroll_region_beg - 1;
  start_line_adjust = start_line = no_region ? 0 : term_height - scroll_region_end;
  clip = no_region ? 0 : UpdateClipBorder();
  ResizedLeftoverRegion(line_fb.w, line_fb.h);
  int   prev_beg_or_1=X_or_1(prev_region_beg),     prev_end_or_ht=X_or_Y(  prev_region_end, term_height);
  int scroll_beg_or_1=X_or_1(scroll_region_beg), scroll_end_or_ht=X_or_Y(scroll_region_end, term_height);

  if (scroll_beg_or_1 != prev_beg_or_1 || prev_end_or_ht != scroll_end_or_ht) GetPrimaryFrameBuffer();
  for (int i =  scroll_beg_or_1; i <    prev_beg_or_1; i++) MoveToOrFromScrollRegion(&line_fb, GetTermLine(i),   line_fb.BackPlus(point(0, (term_height-i+1)*font_height)), LinesFrameBuffer::Flag::NoLayout);
  for (int i =   prev_end_or_ht; i < scroll_end_or_ht; i++) MoveToOrFromScrollRegion(&line_fb, GetTermLine(i+1), line_fb.BackPlus(point(0, (term_height-i)  *font_height)), LinesFrameBuffer::Flag::NoLayout);

  if (prev_beg_or_1 < scroll_beg_or_1 || scroll_end_or_ht < prev_end_or_ht) GetSecondaryFrameBuffer();
  for (int i =    prev_beg_or_1; i < scroll_beg_or_1; i++) MoveToOrFromScrollRegion(&cmd_fb, GetTermLine(i),   point(0, GetFrameY(i)),   LinesFrameBuffer::Flag::NoLayout);
  for (int i = scroll_end_or_ht; i <  prev_end_or_ht; i++) MoveToOrFromScrollRegion(&cmd_fb, GetTermLine(i+1), point(0, GetFrameY(i+1)), LinesFrameBuffer::Flag::NoLayout);
  if (release_fb) { cmd_fb.fb.Release(); last_fb=0; }
}

void Terminal::SetTerminalDimension(int w, int h) {
  term_width  = max(line_fb.only_grow ? term_width  : 0, w);
  term_height = max(line_fb.only_grow ? term_height : 0, h);
  ScopedClearColor scc(line_fb.fb.gd, bg_color);
  if (!line.Size()) TextArea::Write(string(term_height, '\n'), 0);
}

Border *Terminal::UpdateClipBorder() {
  int font_height = style.font->Height();
  clip_border.bottom = font_height * start_line_adjust;
  clip_border.top    = font_height * skip_last_lines;
  if (line_fb.align_top_or_bot) { if (clip_border.bottom) clip_border.bottom += extra_height; }
  else                          { if (clip_border.top)    clip_border.top    += extra_height; }
  return &clip_border;
}

void Terminal::MoveLines(int sy, int ey, int dy, bool move_fb_p) {
  if (sy == ey) return;
  CHECK_LT(sy, ey);
  int line_ind = GetTermLineIndex(sy), scroll_lines = ey - sy + 1, ady = abs(dy), sdy = (dy > 0 ? 1 : -1);
  Move(line, line_ind + (dy>0 ? dy : 0), line_ind + (dy<0 ? -dy : 0), scroll_lines - ady, move_fb_p ? line.movep_cb : line.move_cb);
  for (int i = 0, cy = (dy>0 ? sy : ey); i < ady; i++) GetTermLine(cy + i*sdy)->Clear();
}

void Terminal::Scroll(int sl) {
  bool up = sl<0;
  if (!clip) return up ? PushBackLines(-sl) : PushFrontLines(sl);
  int beg_y = scroll_region_beg, offset = up ? start_line           : skip_last_lines;
  int end_y = scroll_region_end, flag   = up ? LineUpdate::PushBack : LineUpdate::PushFront;
  MoveLines(beg_y, end_y, sl, true);
  GetPrimaryFrameBuffer();
  for (int i=0, l=abs(sl); i<l; i++)
    LineUpdate(GetTermLine(up ? (end_y-l+i+1) : (beg_y+l-i-1)), &line_fb, flag, offset);
}

void Terminal::UpdateToken(Line *L, int word_offset, int word_len, int update_type, const TokenProcessor<DrawableBox> *token) {
  CHECK_LE(word_offset + word_len, L->data->glyphs.Size());
  Line *BL = L, *EL = L, *NL;
  int in_line_ind = -line.IndexOf(L)-1, beg_line_ind = in_line_ind, end_line_ind = in_line_ind;
  int beg_offset = word_offset, end_offset = word_offset + word_len - 1, new_offset, gs;

  for (; !beg_offset && beg_line_ind < term_height; beg_line_ind++, beg_offset = term_width - new_offset, BL = NL) {
    const DrawableBoxArray &glyphs = (NL = &line[-beg_line_ind-1-1])->data->glyphs;
    if ((gs = glyphs.Size()) != term_width || !(new_offset = RLengthChar(&glyphs[gs-1], notspace, gs))) break;
  }
  for (; end_offset == term_width-1 && end_line_ind >= 0; end_line_ind--, end_offset = new_offset - 1, EL = NL) {
    const DrawableBoxArray &glyphs = (NL = &line[-beg_line_ind-1+1])->data->glyphs;
    if (!(new_offset = LengthChar(glyphs.data.data(), notspace, glyphs.Size()))) break;
  }

  string text = CopyText(beg_line_ind, beg_offset, end_line_ind, end_offset, 0);
  if (update_type < 0) UpdateLongToken(BL, beg_offset, EL, end_offset, text, update_type);

  if (BL != L && !word_offset && token->lbw != token->nlbw)
    UpdateLongToken(BL, beg_offset, &line[-in_line_ind-1-1], term_width-1,
                    CopyText(beg_line_ind, beg_offset, in_line_ind+1, term_width-1, 0), update_type * -10);
  if (EL != L && word_offset + word_len == term_width && token->lew != token->nlew)
    UpdateLongToken(&line[-in_line_ind-1+1], 0, EL, end_offset,
                    CopyText(in_line_ind-1, 0, end_line_ind, end_offset, 0), update_type * -11);

  if (update_type > 0) UpdateLongToken(BL, beg_offset, EL, end_offset, text, update_type);
}

void Terminal::Draw(const Box &b, int flag, Shader *shader) {
  GraphicsContext gc(root->gd);
  TextArea::Draw(b, flag & ~DrawFlag::DrawCursor, shader);
  if (shader) shader->SetUniform2f("iChannelScroll", 0, XY_or_Y(shader->scale, -b.y));
  if (clip) {
    { Scissor s(gc.gd, Box::TopBorder(b, *clip)); cmd_fb.DrawAligned(b, point()); }
    { Scissor s(gc.gd, Box::BotBorder(b, *clip)); cmd_fb.DrawAligned(b, point()); }
    if (hover_control) DrawHoverLink(b);
  }
  if (flag & DrawFlag::DrawCursor) TextBox::DrawCursor(b.Position() + cursor.p);
  if (extra_height && !shader) {
    ScopedFillColor c(gc.gd, *bg_color);
    gc.DrawTexturedBox(Box(b.x, b.y + (line_fb.align_top_or_bot ? 0 : line_fb.h), b.w, extra_height));
  }
  if (selection.changing) DrawSelection();
}

void Terminal::Write(const StringPiece &s, bool update_fb, bool release_fb) {
  if (!app->MainThread()) return app->RunInMainThread(bind(&Terminal::WriteCB, this, s.str(), update_fb, release_fb));
  TerminalTrace("Terminal: Write('%s', %zd)", CHexEscapeNonAscii(s.str()).c_str(), s.size());
  line_fb.fb.gd->DrawMode(DrawMode::_2D, 0);
  ScopedClearColor scc(line_fb.fb.gd, bg_color);
  last_fb = 0;
  for (int i = 0; i < s.len; i++) {
    const unsigned char c = *(s.begin() + i);
    if (c == 0x18 || c == 0x1a) { /* CAN or SUB */ parse_state = State::TEXT; continue; }
    if (parse_state == State::ESC) {
      parse_state = State::TEXT; // default next state
      TerminalTrace("ESC %c", c);
      if (c >= '(' && c <= '+') {
        parse_state = State::CHARSET;
        parse_charset = c;
      } else switch (c) {
        case '[':
          parse_state = State::CSI; // control sequence introducer
          parse_csi.clear(); break;
        case ']':
          parse_state = State::OSC; // operating system command
          parse_osc.clear();
          parse_osc_escape = false; break;
        case '=': case '>':                        break; // application or normal keypad
        case 'c': ResetTerminal();                 break;
        case 'D': Newline();                       break;
        case 'M': NewTopline();                    break;
        case '7': saved_term_cursor = term_cursor; break;
        case '8': term_cursor = point(Clamp(saved_term_cursor.x, 1, term_width),
                                      Clamp(saved_term_cursor.y, 1, term_height));
        default: TerminalDebug("unhandled escape %c (%02x)", c, c);
      }
    } else if (parse_state == State::CHARSET) {
      TerminalTrace("charset G%d %c", 1+parse_charset-'(', c);
      parse_state = State::TEXT;

    } else if (parse_state == State::OSC) {
      if (!parse_osc_escape) {
        if (c == 0x1b) { parse_osc_escape = 1; continue; }
        if (c != 0x07) { parse_osc       += c; continue; }
      }
      else if (c != 0x5c) { TerminalDebug("within-OSC-escape %c (%02x)", c, c); parse_state = State::TEXT; continue; }
      parse_state = State::TEXT;

      if (parse_osc.size() > 2 && parse_osc[1] == ';' && Within(parse_osc[0], '0', '2')) root->SetCaption(parse_osc.substr(2));
      else TerminalDebug("unhandled OSC %s", parse_osc.c_str());

    } else if (parse_state == State::CSI) {
      // http://en.wikipedia.org/wiki/ANSI_escape_code#CSI_codes
      if (c < 0x40 || c > 0x7e) { parse_csi += c; continue; }
      TerminalTrace("CSI %s%c (cur=%d,%d)", parse_csi.c_str(), c, term_cursor.x, term_cursor.y);
      parse_state = State::TEXT;

      int parsed_csi=0, parse_csi_argc=0, parse_csi_argv[16];
      unsigned char parse_csi_argv00 = parse_csi.empty() ? 0 : (isdigit(parse_csi[0]) ? 0 : parse_csi[0]);
      for (/**/; Within<int>(parse_csi[parsed_csi], 0x20, 0x2f); parsed_csi++) {}
      StringPiece intermed(parse_csi.data(), parsed_csi);

      memzeros(parse_csi_argv);
      bool parse_csi_arg_done = 0;
      // http://www.inwap.com/pdp10/ansicode.txt 
      for (/**/; Within<int>(parse_csi[parsed_csi], 0x30, 0x3f); parsed_csi++) {
        if (parse_csi[parsed_csi] <= '9') { // 0x30 == '0'
          AccumulateAsciiDigit(&parse_csi_argv[parse_csi_argc], parse_csi[parsed_csi]);
          parse_csi_arg_done = 0;
        } else if (parse_csi[parsed_csi] <= ';') {
          parse_csi_arg_done = 1;
          parse_csi_argc++;
        } else continue;
      }
      if (!parse_csi_arg_done) parse_csi_argc++;

      switch (c) {
        case '@': {
          LineUpdate l(GetCursorLine(), fb_cb);
          l->UpdateText(term_cursor.x-1, StringPiece(string(X_or_1(parse_csi_argv[0]), ' ')), cursor.attr, term_width, 0, 1);
        } break;
        case 'A': term_cursor.y = max(term_cursor.y - X_or_1(parse_csi_argv[0]), 1);           break;
        case 'B': term_cursor.y = min(term_cursor.y + X_or_1(parse_csi_argv[0]), term_height); break;
        case 'C': term_cursor.x = min(term_cursor.x + X_or_1(parse_csi_argv[0]), term_width);  break;
        case 'D': term_cursor.x = max(term_cursor.x - X_or_1(parse_csi_argv[0]), 1);           break;
        case 'd': term_cursor.y = Clamp(parse_csi_argv[0], 1, term_height);                    break;
        case 'G': term_cursor.x = Clamp(parse_csi_argv[0], 1, term_width);                     break;
        case 'H': term_cursor = point(Clamp(parse_csi_argv[1], 1, term_width),
                                      Clamp(parse_csi_argv[0], 1, term_height)); break;
        case 'Z': TabPrev(parse_csi_argv[0]); break;
        case 'J': {
          int clear_beg_y = 1, clear_end_y = term_height;
          if      (parse_csi_argv[0] == 0) { LineUpdate(GetCursorLine(), fb_cb)->Erase(term_cursor.x-1);  clear_beg_y = term_cursor.y; }
          else if (parse_csi_argv[0] == 1) { LineUpdate(GetCursorLine(), fb_cb)->Erase(0, term_cursor.x); clear_end_y = term_cursor.y; }
          else if (parse_csi_argv[0] == 2) { ClearTerminal(); break; }
          for (int i = clear_beg_y; i <= clear_end_y; i++) LineUpdate(GetTermLine(i), fb_cb)->Clear();
        } break;
        case 'K': {
          LineUpdate l(GetCursorLine(), fb_cb);
          if      (parse_csi_argv[0] == 0) l->Erase(term_cursor.x-1);
          else if (parse_csi_argv[0] == 1) l->Erase(0, term_cursor.x);
          else if (parse_csi_argv[0] == 2) l->Clear();
        } break;
        case 'S': case 'T': Scroll((c == 'T' ? 1 : -1) * X_or_1(parse_csi_argv[0])); break;
        case 'L': case 'M': { /* insert or delete lines */
          int sl = (c == 'L' ? 1 : -1) * X_or_1(parse_csi_argv[0]);
          int beg_y = term_cursor.y, end_y = X_or_Y(scroll_region_end, term_height);
          if (beg_y == X_or_1(scroll_region_beg)) { Scroll(sl); break; }
          if (clip && beg_y < scroll_region_beg)
          { TerminalDebug("y=%s scrollregion=%d-%d", term_cursor.DebugString().c_str(), scroll_region_beg, scroll_region_end); break; }
          MoveLines(beg_y, end_y, sl, false);
          GetPrimaryFrameBuffer();
          for (int i=beg_y; i<=end_y; i++) LineUpdate(GetTermLine(i), &line_fb);
        } break;
        case 'P': {
          LineUpdate l(GetCursorLine(), fb_cb);
          int erase = max(1, parse_csi_argv[0]);
          l->Erase(term_cursor.x-1, erase);
        } break;
        case 'h': { // set mode
          int mode = parse_csi_argv[0];
          if      (parse_csi_argv00 == 0   && mode ==    4) { insert_mode = true;           }
          else if (parse_csi_argv00 == 0   && mode ==   34) { /* steady cursor */           }
          else if (parse_csi_argv00 == '?' && mode ==    1) { /* guarded area tx = all */   }
          else if (parse_csi_argv00 == '?' && mode ==    7) { /* wrap around mode */        }
          else if (parse_csi_argv00 == '?' && mode ==   25) { cursor_enabled = true;        }
          else if (parse_csi_argv00 == '?' && mode ==   47) { /* alternate screen buffer */ }
          else if (parse_csi_argv00 == '?' && mode == 1034) { /* meta mode: 8th bit on */   }
          else if (parse_csi_argv00 == '?' && mode == 1049) { /* save screen */             }
          else TerminalDebug("unhandled CSI-h mode = %d av00 = %c i= %s", mode, parse_csi_argv00, intermed.str().c_str());
        } break;
        case 'l': { // reset mode
          int mode = parse_csi_argv[0];
          if      (parse_csi_argv00 == 0   && mode ==    3) { /* 80 column mode */                 }
          else if (parse_csi_argv00 == 0   && mode ==    4) { insert_mode = false;                 }
          else if (parse_csi_argv00 == 0   && mode ==   34) { /* blink cursor */                   }
          else if (parse_csi_argv00 == '?' && mode ==    1) { /* guarded area tx = unprot only */  }
          else if (parse_csi_argv00 == '?' && mode ==    7) { /* no wrap around mode */            }
          else if (parse_csi_argv00 == '?' && mode ==   12) { /* steady cursor */                  }
          else if (parse_csi_argv00 == '?' && mode ==   25) { cursor_enabled = false;              }
          else if (parse_csi_argv00 == '?' && mode ==   47) { /* normal screen buffer */           }
          else if (parse_csi_argv00 == '?' && mode == 1049) { /* restore screen */                 }
          else TerminalDebug("unhandled CSI-l mode = %d av00 = %c i= %s", mode, parse_csi_argv00, intermed.str().c_str());
        } break;
        case 'm':
        for (int i=0; i<parse_csi_argc; i++) {
          int sgr = parse_csi_argv[i]; // select graphic rendition
          if      (sgr >= 30 && sgr <= 37) cursor.attr = Style::SetFGColorIndex(cursor.attr, sgr-30);
          else if (sgr >= 40 && sgr <= 47) cursor.attr = Style::SetBGColorIndex(cursor.attr, sgr-40);
          else switch(sgr) {
            case 0:         cursor.attr  =  default_attr;    break;
            case 1:         cursor.attr |=  Style::Bold;      break;
            case 3:         cursor.attr |=  Style::Italic;    break;
            case 4:         cursor.attr |=  Style::Underline; break;
            case 5: case 6: cursor.attr |=  Style::Blink;     break;
            case 7:         cursor.attr |=  Style::Reverse;   break;
            case 22:        cursor.attr &= ~Style::Bold;      break;
            case 23:        cursor.attr &= ~Style::Italic;    break;
            case 24:        cursor.attr &= ~Style::Underline; break;
            case 25:        cursor.attr &= ~Style::Blink;     break;
            case 27:        cursor.attr &= ~Style::Reverse;   break;
            case 39:        cursor.attr = Style::SetFGColorIndex(cursor.attr, style.colors->normal_index);     break;
            case 49:        cursor.attr = Style::SetBGColorIndex(cursor.attr, style.colors->background_index); break;
            default:        TerminalDebug("unhandled SGR %d", sgr);
          }
        } break;
        case 'p':
          if (parse_csi_argv00 == '!') { /* soft reset http://vt100.net/docs/vt510-rm/DECSTR */
            insert_mode = false;
            SetScrollRegion(1, term_height);
          }
          else TerminalDebug("Unhandled CSI-p %c", parse_csi_argv00);
          break;
        case 'r':
          if (parse_csi_argc == 2) SetScrollRegion(parse_csi_argv[0], parse_csi_argv[1]);
          else TerminalDebug("invalid scroll region argc %d", parse_csi_argc);
          break;
        default:
          TerminalDebug("unhandled CSI %s%c", parse_csi.c_str(), c);
      }
    } else {
      // http://en.wikipedia.org/wiki/C0_and_C1_control_codes#C0_.28ASCII_and_derivatives.29
      bool C0_control = (c >= 0x00 && c <= 0x1f) || c == 0x7f;
      bool C1_control = (c >= 0x80 && c <= 0x9f);
      if (C0_control || C1_control) {
        TerminalTrace("C0/C1 control: %02x", c);
        FlushParseText();
      }
      if (C0_control) switch(c) {
        case '\a':   TerminalDebug("%s", "bell");                     break; // bell
        case '\b':   term_cursor.x = max(term_cursor.x-1, 1);         break; // backspace
        case '\t':   TabNext(1);                                      break; // tab 
        case '\r':   term_cursor.x = 1;                               break; // carriage return
        case '\x1b': parse_state = State::ESC;                        break;
        case '\x14': case '\x15': case '\x7f':                        break; // shift charset in, out, delete
        case '\n':   case '\v':   case '\f':   Newline(newline_mode); break; // line feed, vertical tab, form feed
        default:                               TerminalDebug("unhandled C0 control %02x", c);
      } else if (0 && C1_control) {
        if (0) {}
        else TerminalDebug("unhandled C1 control %02x", c);
      } else {
        parse_text += c;
      }
    }
  }
  FlushParseText();
  UpdateCursor();
  line_fb.fb.Release();
  last_fb = 0;
}

void Terminal::FlushParseText() {
  if (parse_text.empty()) return;
  bool append = 0;
  int consumed = 0, write_size = 0, update_size = 0;
  CHECK_GT(term_cursor.x, 0);
  style.font.ptr = style.GetAttr(cursor.attr)->font;
  String16 input_text = String::ToUTF16(parse_text, &consumed);
  TerminalTrace("Terminal: (cur=%d,%d) FlushParseText('%s').size = [%zd, %d]", term_cursor.x, term_cursor.y,
                CHexEscapeNonAscii(StringPiece(parse_text.data(), consumed)).c_str(), input_text.size(), consumed);
  for (int wrote = 0; wrote < input_text.size(); wrote += write_size) {
    if (wrote) Newline(true);
    else if (term_cursor.x > term_width) { GetCursorLine()->data->wrapped = true; Newline(true); }
    Line *l = GetCursorLine();
    LinesFrameBuffer *fb = GetFrameBuffer(l);
    int remaining = input_text.size() - wrote, o = term_cursor.x-1, tl = term_width - o;
    write_size = min(remaining, tl);
    if (write_size == tl && remaining > write_size) l->data->wrapped = true;
    String16Piece input_piece(input_text.data() + wrote, write_size);
    update_size = l->UpdateText(o, input_piece, cursor.attr, term_width, &append);
    TerminalTrace("Terminal: FlushParseText: UpdateText(%d, %d, '%s').size = [%d, %d] attr=%d",
                  term_cursor.x, term_cursor.y, String::ToUTF8(input_piece).c_str(), write_size, update_size, cursor.attr);
    if (!update_size) continue;
    l->Layout();
    if (!fb->lines) continue;
    int s = l->Size(), ol = s - o, sx = l->data->glyphs.LeftBound(o), ex = l->data->glyphs.RightBound(s-1);
    if (append) l->Draw(l->p, -1, o, ol);
    else LinesFrameBuffer::Paint(l, point(sx, l->p.y), Box(-sx, 0, ex - sx, fb->font_height), o, ol);
  }
  term_cursor.x += update_size;
  parse_text.erase(0, consumed);
}

void Terminal::Newline(bool carriage_return) {
  if (clip && term_cursor.y == scroll_region_end) Scroll(-1);
  else if (term_cursor.y == term_height) { if (!clip) PushBackLines(1); }
  else term_cursor.y = min(term_height, term_cursor.y+1);
  if (carriage_return) term_cursor.x = 1;
}

void Terminal::NewTopline() {
  if (clip && term_cursor.y == scroll_region_beg) Scroll(1);
  else if (term_cursor.y == 1) { if (!clip) PushFrontLines(1); }
  else term_cursor.y = max(1, term_cursor.y-1);
}

void Terminal::TabNext(int n) { term_cursor.x = min(NextMultipleOfN(term_cursor.x, tab_width) + 1, term_width); }
void Terminal::TabPrev(int n) {
  if (tab_stop.size()) TerminalDebug("%s", "variable tab stop not implemented");
  else if ((term_cursor.x = max(PrevMultipleOfN(term_cursor.x - max(0, n-1)*tab_width - 2, tab_width), 1)) != 1) term_cursor.x++;
}

void Terminal::Redraw(bool attach, bool relayout) {
  bool prev_clip = clip;
  int prev_scroll_beg = scroll_region_beg, prev_scroll_end = scroll_region_end;
  SetScrollRegion(1, term_height, true);
  TextArea::Redraw(true, relayout);
  if (prev_clip) SetScrollRegion(prev_scroll_beg, prev_scroll_end, true);
}

void Terminal::ResetTerminal() {
  term_cursor.x = term_cursor.y = 1;
  scroll_region_beg = scroll_region_end = 0;
  clip = 0;
  ClearTerminal();
}

void Terminal::ClearTerminal() {
  for (int i=1; i<=term_height; ++i) GetTermLine(i)->Clear();
  Redraw(true);
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

Dialog::Dialog(Window *W, float w, float h, int flag) : GUI(W),
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
  ResetGUI();
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

TabbedDialogInterface::TabbedDialogInterface(GUI *g, const point &d): gui(g), tab_dim(d) {}
void TabbedDialogInterface::Layout() {
  for (auto b=tab_list.begin(), e=tab_list.end(), t=b; t != e; ++t) {
    int tab_no = t-b;
    t->dialog->LayoutTabbed(tab_no, box, tab_dim, &gui->mouse, &t->child_box);
    gui->mouse.AddClickBox(t->dialog->title + point(box.x + tab_no*tab_dim.x, 0),
                           MouseController::CB(bind(&TabbedDialogInterface::SelectTabIndex, this, tab_no)));
  }
  if (tab_list.size() && tab_list.size() * tab_dim.x > box.w) {
    auto t = tab_list.begin();
    int fh = t->dialog->font->Height();
    Font *menuicon = t->dialog->menuicon.ptr;
    int attr_id = gui->child_box.attr.GetAttrId(Drawable::Attr(menuicon, &Color::white, nullptr, false, true));
    gui->child_box.PushBack(Box(box.x,          -fh, fh, fh), attr_id, menuicon ? menuicon->FindGlyph(6) : 0);
    gui->child_box.PushBack(Box(box.right()-fh, -fh, fh, fh), attr_id, menuicon ? menuicon->FindGlyph(7) : 0);
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

void HelperGUI::ForceDirectedLayout() {
#ifdef LFL_BOOST
  BoostForceDirectedLayout().Layout(this);
#endif
}

HelperGUI::Label::Label(const Box &w, const string &d, int h, Font *f, const point &p) :
  target(w), target_center(target.center()), hint(h), description(d) {
  label_center = target_center;
  if      (h == Hint::UP   || h == Hint::UPLEFT   || h == Hint::UPRIGHT)   label_center.y += p.y;
  else if (h == Hint::DOWN || h == Hint::DOWNLEFT || h == Hint::DOWNRIGHT) label_center.y -= p.y;
  if      (h == Hint::UPRIGHT || h == Hint::DOWNRIGHT)                     label_center.x += p.x;
  else if (h == Hint::UPLEFT  || h == Hint::DOWNLEFT)                      label_center.x -= p.x;
  f->Size(description.c_str(), &label);
  AssignLabelBox();
}

void HelperGUI::Draw() {
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
