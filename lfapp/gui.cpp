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

#ifdef LFL_QT
#include <QMessageBox>
#endif

#ifdef LFL_LIBCLANG
#include "clang-c/Index.h"
static string GetClangString(const CXString &s) { string ret=LFL::BlankNull(clang_getCString(s)); clang_disposeString(s); return ret; }
#endif

namespace LFL {
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
DEFINE_bool(multitouch, true, "Touchscreen controls");
#else
DEFINE_bool(multitouch, false, "Touchscreen controls");
#endif
DEFINE_bool(lfapp_console, false, "Enable dropdown lfapp console");
DEFINE_string(lfapp_console_font, "", "Console font, blank for default_font");
DEFINE_int(lfapp_console_font_flag, FontDesc::Mono, "Console font flag");
DEFINE_bool(draw_grid, false, "Draw lines intersecting mouse x,y");

void GUI::UpdateBox(const Box &b, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box = b;
  if (input_box_ind >= 0) hit           [input_box_ind].box = b;
}

void GUI::UpdateBoxX(int x, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.x = x;
  if (input_box_ind >= 0) hit           [input_box_ind].box.x = x;
}

void GUI::UpdateBoxY(int y, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.y = y;
  if (input_box_ind >= 0) hit           [input_box_ind].box.y = y;
}

void GUI::IncrementBoxY(int y, int draw_box_ind, int input_box_ind) {
  if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.y += y;
  if (input_box_ind >= 0) hit           [input_box_ind].box.y += y;
}

void GUI::Draw() {
  if (child_box.data.empty()) Layout();
  child_box.Draw(box.TopLeft());
}

bool GUI::ToggleActive() {
  bool ret = toggle_active.Toggle();
  active ? Activate() : Deactivate();
  return ret;
}

void Widget::Button::Layout(Flow *flow) {
  flow->SetFont(0);
  flow->SetFGColor(&Color::white);
  LayoutComplete(flow, flow->out->data[flow->AppendBox(box.w, box.h, drawable)].box);
}

void Widget::Button::LayoutBox(Flow *flow, const Box &b) {
  flow->SetFont(0);
  flow->SetFGColor(&Color::white);
  if (drawable) flow->out->PushBack(b, flow->cur_attr, drawable, &drawbox_ind);
  LayoutComplete(flow, b);
}

void Widget::Button::LayoutComplete(Flow *flow, const Box &b) {
  SetBox(b);
  if (outline) {
    flow->SetFont(0);
    flow->SetFGColor(outline);
    flow->out->PushBack(box, flow->cur_attr, Singleton<BoxOutline>::Get());
  }
  if (!text.empty()) {
    point save_p = flow->p;
    flow->SetFont(font);
    flow->SetFGColor(0);
    flow->p = box.Position() + point(Box(0, 0, box.w, box.h).centerX(textsize.x), 0);
    flow->AppendText(text);
    flow->p = save_p;
  }
}

Widget::Scrollbar::Scrollbar(GUI *Gui, Box window, int f) : Interface(Gui), win(window), flag(f), menuicon(Fonts::Get("MenuAtlas", "", 0, Color::white, Color::clear, 0)) {
  if (win.w && win.h) { if (f & Flag::Attached) LayoutAttached(win); else LayoutFixed(win); }
}

void Widget::Scrollbar::LayoutAttached(const Box &w) {
  win = w;
  int aw = dot_size, ah = dot_size;
  bool flip = flag & Flag::Horizontal;
  if (flip) win.h = ah;
  else { win.x += win.w - aw; win.w = aw; }
  if (flag & Flag::NoCorner) {
    if (flip) win.w -= aw;
    else { win.h -= ah; win.y += ah; }
  }
  Layout(aw, ah, flip);
}

void Widget::Scrollbar::Layout(int aw, int ah, bool flip) {
  Box arrow_down = win;
  if (flip) { arrow_down.w = aw; win.x += aw; }
  else      { arrow_down.h = ah; win.y += ah; }

  Box scroll_dot = arrow_down, arrow_up = win;
  if (flip) { arrow_up.w = aw; win.w -= 2*aw; arrow_up.x += win.w; }
  else      { arrow_up.h = ah; win.h -= 2*ah; arrow_up.y += win.h; }

  if (gui) {
    int attr_id = gui->child_box.attr.GetAttrId(Drawable::Attr(menuicon, NULL, NULL, false, true));
    gui->child_box.PushBack(arrow_up,   attr_id, menuicon ? menuicon->FindGlyph(flip ? 2 : 4) : 0);
    gui->child_box.PushBack(arrow_down, attr_id, menuicon ? menuicon->FindGlyph(flip ? 3 : 1) : 0);
    gui->child_box.PushBack(scroll_dot, attr_id, menuicon ? menuicon->FindGlyph(           5) : 0, &drawbox_ind);

    AddDragBox(scroll_dot, MouseController::CB(bind(&Scrollbar::DragScrollDot, this)));
    AddClickBox(arrow_up,   MouseController::CB(bind(flip ? &Scrollbar::ScrollDown : &Scrollbar::ScrollUp,   this)));
    AddClickBox(arrow_down, MouseController::CB(bind(flip ? &Scrollbar::ScrollUp   : &Scrollbar::ScrollDown, this)));
  }
  Update(true);
}

void Widget::Scrollbar::Update(bool force) {
  if (!app->input || !app->input->MouseButton1Down()) dragging = false;
  if (!dragging && !dirty && !force) return;
  bool flip = flag & Flag::Horizontal;
  int aw = dot_size, ah = dot_size;
  if (dragging) {
    if (flip) scrolled = Clamp(    (float)(gui->MousePosition().x - win.x) / win.w, 0, 1);
    else      scrolled = Clamp(1 - (float)(gui->MousePosition().y - win.y) / win.h, 0, 1);
  }
  if (flip) gui->UpdateBoxX(win.x          + (int)((win.w - aw) * scrolled), drawbox_ind, IndexOrDefault(hitbox, 0, -1));
  else      gui->UpdateBoxY(win.top() - ah - (int)((win.h - ah) * scrolled), drawbox_ind, IndexOrDefault(hitbox, 0, -1));
  dirty = false;
}

float Widget::Scrollbar::AddScrollDelta(float cur_val) {
  scrolled = Clamp(cur_val + ScrollDelta(), 0, 1);
  if (EqualChanged(&last_scrolled, scrolled)) dirty = 1;
  return scrolled;
}
  
void Widget::Scrollbar::AttachContentBox(Box *b, Scrollbar *vs, Scrollbar *hs) {
  if (vs) { vs->LayoutAttached(*b); }
  if (hs) { hs->LayoutAttached(*b); MinusPlus(&b->h, &b->y, vs->dot_size); }
  if (vs) b->w -= vs->dot_size;
}

void KeyboardGUI::AddHistory(const string &cmd) {
  lastcmd.ring.PushBack(1);
  lastcmd[(lastcmd_ind = -1)] = cmd;
}

int KeyboardGUI::WriteHistory(const string &dir, const string &name, const string &hdr) {
  if (!lastcmd.Size()) return 0;
  LocalFile history(dir + MatrixFile::Filename(name, "history", "string", 0), "w");
  MatrixFile::WriteHeader(&history, BaseName(history.fn), hdr, lastcmd.ring.count, 1);
  for (int i=0; i<lastcmd.ring.count; i++) StringFile::WriteRow(&history, lastcmd[-1-i]);
  return 0;
}

int KeyboardGUI::ReadHistory(const string &dir, const string &name) {
  StringFile history;
  VersionedFileName vfn(dir.c_str(), name.c_str(), "history");
  if (history.ReadVersioned(vfn) < 0) { ERROR("no ", name, " history"); return -1; }
  for (int i=0, l=history.Lines(); i<l; i++) AddHistory((*history.F)[l-1-i]);
  return 0;
}

TextGUI::Link::Link(TextGUI::Line *P, GUI *G, const Box3 &b, const string &U) : Interface(G), box(b), link(U), line(P) {
  AddClickBox(b, MouseController::CB(bind(&Link::Visit, this)));
  AddHoverBox(b, MouseController::CoordCB(bind(&Link::Hover, this, _1, _2, _3, _4)));
  del_hitbox = true;
}

int TextGUI::Line::Erase(int x, int l) {
  if (!(l = max(0, min(Size() - x, l)))) return 0;
  bool token_processing = parent->token_processing;
  LineTokenProcessor update(token_processing ? this : 0, x, DrawableBoxRun(&data->glyphs[x], l), l);
  if (token_processing) {
    update.SetNewLineBoundaryConditions(!x ? update.nw : update.lbw, x + l == update.line_size ? update.pw : update.lew);
    update.ProcessUpdate();
  }
  data->glyphs.Erase(x, l, true);
  data->flow.p.x = data->glyphs.Position(data->glyphs.Size()).x;
  if (update.nw) update.ni -= l;
  if (token_processing) update.ProcessResult();
  return l;
}

template <class X> int TextGUI::Line::InsertTextAt(int x, const StringPieceT<X> &v, int attr) {
  if (!v.size()) return 0;
  DrawableBoxArray b;
  b.attr.source = data->glyphs.attr.source;
  EncodeText(&b, data->glyphs.Position(x).x, v, attr);
  return b.Size() ? InsertTextAt(x, v, b) : 0;
}

template <class X> int TextGUI::Line::InsertTextAt(int x, const StringPieceT<X> &v, const Flow::TextAnnotation &attr) {
  if (!v.size()) return 0;
  DrawableBoxArray b;
  b.attr.source = data->glyphs.attr.source;
  EncodeText(&b, data->glyphs.Position(x).x, v, attr, parent->default_attr);
  return b.Size() ? InsertTextAt(x, v, b) : 0;
}

template <class X> int TextGUI::Line::InsertTextAt(int x, const StringPieceT<X> &v, const DrawableBoxArray &b) {
  int ret = b.Size();
  bool token_processing = parent->token_processing, append = x == Size();
  LineTokenProcessor update(token_processing ? this : 0, x, DrawableBoxRun(&b[0], ret), 0);
  if (token_processing) {
    update.SetNewLineBoundaryConditions(!x ? update.sw : update.lbw, x == update.line_size-1 ? update.ew : update.lew);
    update.ProcessResult();
  }

  data->glyphs.InsertAt(x, b);
  data->flow.p.x = data->glyphs.Position(data->glyphs.Size()).x;
  if (!append && update.nw) update.ni += ret;

  if (token_processing) update.ProcessUpdate();
  return ret;
}

template <class X> int TextGUI::Line::OverwriteTextAt(int x, const StringPieceT<X> &v, int attr) {
  int size = Size(), pad = max(0, x + v.len - size), grow = 0;
  if (pad) data->flow.AppendText(basic_string<X>(pad, ' '), attr);

  DrawableBoxArray b;
  b.attr.source = data->glyphs.attr.source;
  EncodeText(&b, data->glyphs.Position(x).x, v, attr);
  if (!(size = b.Size())) return 0;
  if (size - v.len > 0 && (grow = max(0, x + size - Size())))
    data->flow.AppendText(basic_string<X>(grow, ' '), attr);
  DrawableBoxRun orun(&data->glyphs[x], size), nrun(&b[0], size);

  bool token_processing = parent->token_processing;
  LineTokenProcessor update(token_processing ? this : 0, x, orun, size);
  if (token_processing) {
    update.FindBoundaryConditions(nrun, &update.osw, &update.oew);
    update.SetNewLineBoundaryConditions(!x ? update.osw : update.lbw, x + size == update.line_size ? update.oew : update.lew);
    update.ProcessUpdate();
  }
  data->glyphs.OverwriteAt(x, b.data);
  data->flow.p.x = data->glyphs.Position(data->glyphs.Size()).x;
  if (token_processing) {
    update.PrepareOverwrite(nrun);
    update.ProcessUpdate();
  }
  return size;
}

template <class X> int TextGUI::Line::UpdateText(int x, const StringPieceT<X> &v, int attr, int max_width, bool *append_out, int mode) {
  bool append = 0, insert_mode = mode == -1 ? parent->insert_mode : mode;
  int size = Size(), ret = 0;
  if (insert_mode) {
    if (size < x)                 data->flow.AppendText(basic_string<X>(x - size, ' '), attr);
    if ((append = (Size() == x))) ret  = AppendText  (   v, attr);
    else                          ret  = InsertTextAt(x, v, attr);
    if (max_width)                       Erase(max_width);
  } else {
    data->flow.cur_attr.font = parent->font;
    ret = OverwriteTextAt(x, v, attr);
  }
  if (append_out) *append_out = append;
  return ret;
}

template int TextGUI::Line::UpdateText<char>      (int x, const StringPiece   &v, int attr, int max_width, bool *append, int);
template int TextGUI::Line::UpdateText<char16_t>  (int x, const String16Piece &v, int attr, int max_width, bool *append, int);
template int TextGUI::Line::InsertTextAt<char>    (int x, const StringPiece   &v, int attr);
template int TextGUI::Line::InsertTextAt<char16_t>(int x, const String16Piece &v, int attr);
template int TextGUI::Line::InsertTextAt<char>    (int x, const StringPiece   &v, const Flow::TextAnnotation &attr);
template int TextGUI::Line::InsertTextAt<char16_t>(int x, const String16Piece &v, const Flow::TextAnnotation &attr);
template int TextGUI::Line::InsertTextAt<char>    (int x, const StringPiece   &v, const DrawableBoxArray &b);
template int TextGUI::Line::InsertTextAt<char16_t>(int x, const String16Piece &v, const DrawableBoxArray &b);

void TextGUI::Line::Layout(Box win, bool flush) {
  if (data->box.w == win.w && !flush) return;
  data->box = win;
  ScopedDeltaTracker<int> SWLT(cont ? &cont->wrapped_lines : 0, bind(&Line::Lines, this));
  DrawableBoxArray b;
  swap(b, data->glyphs);
  data->glyphs.attr.source = b.attr.source;
  Clear();
  data->flow.AppendBoxArrayText(b);
}

point TextGUI::Line::Draw(point pos, int relayout_width, int g_offset, int g_len) {
  if (relayout_width >= 0) Layout(relayout_width);
  data->glyphs.Draw((p = pos), g_offset, g_len);
  return p - point(0, parent->font->Height() + data->glyphs.height);
}

TextGUI::LineTokenProcessor::LineTokenProcessor(TextGUI::Line *l, int o, const DrawableBoxRun &V, int Erase)
  : L(l), x(o), line_size(L?L->Size():0), erase(Erase) {
    if (!L) return;
    const DrawableBoxArray &glyphs = L->data->glyphs;
    const Drawable *p=0, *n=0;
    CHECK_LE(x, line_size);
    LoadV(V);
    ni = x + (Erase ? Erase : 0);
    nw = ni<line_size && (n=glyphs[ni ].drawable) && !isspace(n->Id());
    pw = x >0         && (p=glyphs[x-1].drawable) && !isspace(p->Id());
    pi = x - pw;
    if ((pw && nw) || (pw && sw)) FindPrev(glyphs);
    if ((pw && nw) || (nw && ew)) FindNext(glyphs);
    FindBoundaryConditions(DrawableBoxRun(&glyphs[0], line_size), &lbw, &lew);
  }

void TextGUI::LineTokenProcessor::ProcessUpdate() {
  int tokens = 0, vl = v.Size();
  if (!vl) return;

  StringWordIterT<DrawableBox> word(v.data.buf, v.data.len, isspace, 0);
  for (const DrawableBox *w = word.Next(); w; w = word.Next(), tokens++) {
    int start_offset = w - v.data.buf, end_offset = start_offset + word.cur_len;
    bool first = start_offset == 0, last = end_offset == v.data.len;
    if (first && last && pw && nw) L->parent->UpdateToken(L, pi, ni-pi+1,                             erase ? -1 : 1, this);
    else if (first && pw)          L->parent->UpdateToken(L, pi, x+end_offset-pi,                     erase ? -2 : 2, this);
    else if (last && nw)           L->parent->UpdateToken(L, x+start_offset, ni-x-start_offset+1,     erase ? -3 : 3, this);
    else                           L->parent->UpdateToken(L, x+start_offset, end_offset-start_offset, erase ? -4 : 4, this);
  }
  if ((!tokens || overwrite) && vl) {
    const DrawableBoxArray &glyphs = L->data->glyphs;
    if (pw && !sw && osw) { FindPrev(glyphs); L->parent->UpdateToken(L, pi, x-pi,        erase ? -5 : 5, this); }
    if (nw && !ew && oew) { FindNext(glyphs); L->parent->UpdateToken(L, x+vl, ni-x-vl+1, erase ? -6 : 6, this); }
  }
}

void TextGUI::LineTokenProcessor::ProcessResult() {
  const DrawableBoxArray &glyphs = L->data->glyphs;
  if      (pw && nw) L->parent->UpdateToken(L, pi, ni - pi + 1, erase ? 7 : -7, this);
  else if (pw && sw) L->parent->UpdateToken(L, pi, x  - pi,     erase ? 8 : -8, this);
  else if (nw && ew) L->parent->UpdateToken(L, x,  ni - x + 1,  erase ? 9 : -9, this);
}

void TextGUI::LineTokenProcessor::FindBoundaryConditions(const DrawableBoxRun &v, bool *sw, bool *ew) {
  *sw = v.Size() && !isspace(v.First().Id());
  *ew = v.Size() && !isspace(v.Last ().Id());
}

TextGUI::Lines::Lines(TextGUI *P, int N) : RingVector<Line>(N), parent(P), wrapped_lines(N),
  move_cb (bind(&Line::Move,  _1, _2)),
  movep_cb(bind(&Line::MoveP, _1, _2)) { for (auto &i : data) i.Init(P, this); }

TextGUI::Line *TextGUI::Lines::InsertAt(int dest_line, int lines, int dont_move_last) {
  CHECK(lines);
  CHECK_LT(dest_line, 0);
  int clear_dir = 1;
  if (dest_line != -1) Move<Lines,Line>(*this, dest_line+lines, dest_line, -dest_line - lines - dont_move_last, movep_cb);
  else if ((clear_dir = -1)) { 
    ring.PushBack(lines);
    for (int scrollback_start_line = parent->GetFrameBuffer()->h / parent->font->Height(), i=0; i<lines; i++)
      for (auto &l : (*this)[-scrollback_start_line-i-1].data->links) l.second->DelHitBox();
  }
  for (int i=0; i<lines; i++) (*this)[dest_line + i*clear_dir].Clear();
  return &(*this)[dest_line];
}

TextGUI::LinesFrameBuffer *TextGUI::LinesFrameBuffer::Attach(TextGUI::LinesFrameBuffer **last_fb) {
  if (*last_fb != this) fb.Attach();
  return *last_fb = this;
}

bool TextGUI::LinesFrameBuffer::SizeChanged(int W, int H, Font *font) {
  lines = H / font->Height();
  return RingFrameBuffer::SizeChanged(W, H, font);
}

void TextGUI::LinesFrameBuffer::Update(TextGUI::Line *l, int flag) {
  if (!(flag & Flag::NoLayout)) l->Layout(wrap ? w : 0, flag & Flag::Flush);
  RingFrameBuffer::Update(l, Box(w, l->Lines() * font_height), paint_cb, true);
}

void TextGUI::LinesFrameBuffer::OverwriteUpdate(Line *l, int xo, int wlo, int wll, int flag) {
  Update(l, flag);
}

int TextGUI::LinesFrameBuffer::PushFrontAndUpdate(TextGUI::Line *l, int xo, int wlo, int wll, int flag) {
  if (!(flag & Flag::NoLayout)) l->Layout(wrap ? w : 0, flag & Flag::Flush);
  int wl = max(0, l->Lines() - wlo), lh = (wll ? min(wll, wl) : wl) * font_height;
  if (!lh) return 0;
  Box b(xo, wl * font_height - lh, w, lh);
  return RingFrameBuffer::PushFrontAndUpdate(l, b, paint_cb, !(flag & Flag::NoVWrap)) / font_height;
}

int TextGUI::LinesFrameBuffer::PushBackAndUpdate(TextGUI::Line *l, int xo, int wlo, int wll, int flag) {
  if (!(flag & Flag::NoLayout)) l->Layout(wrap ? w : 0, flag & Flag::Flush);
  int wl = max(0, l->Lines() - wlo), lh = (wll ? min(wll, wl) : wl) * font_height;
  if (!lh) return 0;
  Box b(xo, wlo * font_height, w, lh);
  return RingFrameBuffer::PushBackAndUpdate(l, b, paint_cb, !(flag & Flag::NoVWrap)) / font_height;
}

void TextGUI::LinesFrameBuffer::PushFrontAndUpdateOffset(TextGUI::Line *l, int lo) {
  Update(l, RingFrameBuffer::BackPlus(point(0, font_height + lo * -font_height)));
  RingFrameBuffer::AdvancePixels(-l->Lines() * font_height);
}

void TextGUI::LinesFrameBuffer::PushBackAndUpdateOffset(TextGUI::Line *l, int lo) {
  Update(l, RingFrameBuffer::BackPlus(point(0, lo * font_height)));
  RingFrameBuffer::AdvancePixels(l->Lines() * font_height);
}

point TextGUI::LinesFrameBuffer::Paint(TextGUI::Line *l, point lp, const Box &b, int offset, int len) {
  Scissor scissor(lp.x, lp.y - b.h, b.w, b.h);
  screen->gd->Clear();
  l->Draw(lp + b.Position(), -1, offset, len);
  return point(lp.x, lp.y-b.h);
}

point TextGUI::LinesGUI::MousePosition() const {
  point p = screen->mouse - box.BottomLeft();
  if (const Border *clip = parent->clip) if (p.y < clip->bottom || p.y > box.h - clip->top) return p - point(0, box.h);
  return parent->GetFrameBuffer()->BackPlus(p);
}

TextGUI::LineUpdate::~LineUpdate() {
  if (!fb->lines || (flag & DontUpdate)) v->Layout(fb->wrap ? fb->w : 0);
  else if (flag & PushFront) { if (o) fb->PushFrontAndUpdateOffset(v,o); else fb->PushFrontAndUpdate(v); }
  else if (flag & PushBack)  { if (o) fb->PushBackAndUpdateOffset (v,o); else fb->PushBackAndUpdate (v); }
  else fb->Update(v);
}

const Drawable::Attr *TextGUI::GetAttr(int attr) const {
  const Color *fg=0, *bg=0;
  if (colors) {
    bool italic = attr & Attr::Italic, bold = attr & Attr::Bold;
    int fg_index = Attr::GetFGColorIndex(attr), bg_index = Attr::GetBGColorIndex(attr);
    fg = &colors->c[italic ? colors->bg_index     : ((bold && fg_index == colors->normal_index) ? colors->bold_index : fg_index)];
    bg = &colors->c[italic ? colors->normal_index : bg_index];
    if (attr & Attr::Reverse) swap(fg, bg);
  }
  last_attr.font = Fonts::Change(font, font->size, *fg, *bg, font->flag);
  last_attr.bg = bg == &colors->c[colors->bg_index] ? 0 : bg; // &font->bg;
  last_attr.underline = attr & Attr::Underline;
  return &last_attr;
}

void TextGUI::Enter() {
  string cmd = String::ToUTF8(Text16());
  AssignInput("");
  if (!cmd.empty()) { AddHistory(cmd); Run(cmd); }
  if (deactivate_on_enter) active = false;
}

void TextGUI::SetColors(Colors *C) {
  colors = C;
  Attr::SetFGColorIndex(&default_attr, colors->normal_index);
  Attr::SetBGColorIndex(&default_attr, colors->bg_index);
  bg_color = &colors->c[colors->bg_index];
}

void TextGUI::UpdateLineFB(Line *L, LinesFrameBuffer *fb) {
  fb->fb.Attach();
  ScopedClearColor scc(bg_color);
  ScopedDrawMode drawmode(DrawMode::_2D);
  fb->OverwriteUpdate(L);
  fb->fb.Release();
}

void TextGUI::Draw(const Box &b) {
  if (cmd_fb.SizeChanged(b.w, b.h, font)) {
    cmd_fb.p = point(0, font->Height());
    cmd_line.Draw(point(0, cmd_line.Lines() * font->Height()), cmd_fb.w);
    cmd_fb.SizeChangedDone();
  }
  // screen->gd->PushColor();
  // screen->gd->SetColor(cmd_color);
  cmd_fb.Draw(b.Position(), point());
  DrawCursor(b.Position() + cursor.p);
  // screen->gd->PopColor();
}

void TextGUI::DrawCursor(point p) {
  if (cursor.type == Cursor::Block) {
    screen->gd->EnableBlend();
    screen->gd->BlendMode(GraphicsDevice::OneMinusDstColor, GraphicsDevice::OneMinusSrcAlpha);
    screen->gd->FillColor(cmd_color);
    Box(p.x, p.y - font->Height(), font->max_width, font->Height()).Draw();
    screen->gd->BlendMode(GraphicsDevice::SrcAlpha, GraphicsDevice::One);
    screen->gd->DisableBlend();
  } else {
    bool blinking = false;
    Time now = Now(), elapsed; 
    if (active && (elapsed = now - cursor.blink_begin) > cursor.blink_time) {
      if (elapsed > cursor.blink_time * 2) cursor.blink_begin = now;
      else blinking = true;
    }
    if (blinking) font->Draw("_", p);
  }
}

void TextGUI::UpdateToken(Line *L, int word_offset, int word_len, int update_type, const LineTokenProcessor*) {
  const DrawableBoxArray &glyphs = L->data->glyphs;
  CHECK_LE(word_offset + word_len, glyphs.Size());
  string text = DrawableBoxRun(&glyphs[word_offset], word_len).Text();
  UpdateLongToken(L, word_offset, L, word_offset+word_len-1, text, update_type);
}

void TextGUI::UpdateLongToken(Line *BL, int beg_offset, Line *EL, int end_offset, const string &text, int update_type) {
  StringPiece textp(text);
  int offset = 0, url_offset = -1, fh = font->Height();
  for (; textp.len>1 && MatchingParens(*textp.buf, *textp.rbegin()); offset++, textp.buf++, textp.len -= 2) {}
  if (int punct = LengthChar(textp.buf, ispunct, textp.len)) { offset += punct; textp.buf += punct; }
  if      (textp.len > 7 && PrefixMatch(textp.buf, "http://"))  url_offset = offset + 7;
  else if (textp.len > 8 && PrefixMatch(textp.buf, "https://")) url_offset = offset + 8;
  if (url_offset >= 0) {
    if (update_type < 0) BL->data->links.erase(beg_offset);
    else {
      LinesFrameBuffer *fb = GetFrameBuffer();
      int fb_h = fb->Height(), adjust_y = BL->data->outside_scroll_region ? -fb_h : 0;
      Box gb = Box(BL->data->glyphs[beg_offset].box).SetY(BL->p.y - fh + adjust_y);
      Box ge = Box(EL->data->glyphs[end_offset].box).SetY(EL->p.y - fh + adjust_y);
      Box3 box(Box(fb->Width(), fb_h), gb.Position(), ge.Position() + point(ge.w, 0), fh, fh);
      auto i = Insert(BL->data->links, beg_offset, shared_ptr<Link>(new Link(BL, &mouse_gui, box, offset ? textp.str() : text)));
      if (new_link_cb) new_link_cb(i->second);
    }
  }
}

point TilesTextGUI::PaintCB(Line *l, point lp, const Box &b) {
  tiles->AddDrawableBoxArray(l->data->glyphs, lp + b.Position() + offset);
  return point(lp.x, lp.y-b.h);
}

/* TextArea */

void TextArea::Write(const StringPiece &s, bool update_fb, bool release_fb) {
  if (!MainThread()) return RunInMainThread(new Callback(bind(&TextArea::WriteCB, this, s.str(), update_fb, release_fb)));
  write_last = Now();
  bool wrap = Wrap();
  int update_flag = LineFBPushBack();
  LinesFrameBuffer *fb = GetFrameBuffer();
  ScopedClearColor scc(update_fb ? bg_color : NULL);
  ScopedDrawMode drawmode(update_fb ? DrawMode::_2D : DrawMode::NullOp);
  if (update_fb && fb->lines) fb->fb.Attach();
  StringLineIter add_lines(s, StringLineIter::Flag::BlankLines);
  for (const char *add_line = add_lines.Next(); add_line; add_line = add_lines.Next()) {
    bool append = !write_newline && add_lines.first && add_lines.CurrentLength() && line.ring.count;
    Line *l = append ? &line[-1] : line.InsertAt(-1);
    if (!append) { l->Clear(); if (start_line) { start_line++; end_line++; } }
    if (write_timestamp) l->AppendText(StrCat(logtime(Now()), " "), cursor.attr);
    l->AppendText(StringPiece(add_line, add_lines.CurrentLength()), cursor.attr);
    l->Layout(wrap ? fb->w : 0);
    if (scrolled_lines) v_scrolled = (float)++scrolled_lines / (WrappedLines()-1);
    if (!update_fb || start_line) continue;
    LineUpdate(&line[-start_line-1], fb, (!append ? update_flag : 0));
  }
  if (update_fb && release_fb && fb->lines) fb->fb.Release();
}

void TextArea::Resized(const Box &b) {
  if (selection.enabled) {
    mouse_gui.box.SetDimension(b.Dimension());
    mouse_gui.UpdateBox(Box(0,-b.h,b.w,b.h*2), -1, selection.gui_ind);
  }
  UpdateLines(last_v_scrolled, 0, 0, 0);
  UpdateCursor();
  Redraw(false);
}

void TextArea::CheckResized(const Box &b) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  if (fb->SizeChanged(b.w, b.h, font)) { Resized(b); fb->SizeChangedDone(); }
}

void TextArea::Redraw(bool attach) {
  ScopedClearColor scc(bg_color);
  ScopedDrawMode drawmode(DrawMode::_2D);
  LinesFrameBuffer *fb = GetFrameBuffer();
  int fb_flag = LinesFrameBuffer::Flag::NoVWrap | LinesFrameBuffer::Flag::Flush;
  int lines = start_line_adjust + skip_last_lines, font_height = font->Height();
  int (LinesFrameBuffer::*update_cb)(Line*, int, int, int, int) =
    reverse_line_fb ? &LinesFrameBuffer::PushBackAndUpdate
                    : &LinesFrameBuffer::PushFrontAndUpdate;
  fb->p = reverse_line_fb ? point(0, fb->Height() - start_line_adjust * font_height) 
                          : point(0, start_line_adjust * font_height);
  if (attach) { fb->fb.Attach(); screen->gd->Clear(); }
  for (int i=start_line; i<line.ring.count && lines < fb->lines; i++)
    lines += (fb->*update_cb)(&line[-i-1], -line_left, 0, fb->lines - lines, fb_flag);
  fb->p = point(0, fb->Height());
  if (attach) { fb->scroll = v2(); fb->fb.Release(); }
}

int TextArea::UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  pair<int, int> old_first_line(start_line, -start_line_adjust), new_first_line, new_last_line;
  FlattenedArrayValues<TextGUI::Lines>
    flattened_lines(&line, line.Size(), bind(&TextArea::LayoutBackLine, this, _1, _2));
  flattened_lines.AdvanceIter(&new_first_line, (scrolled_lines = RoundF(v_scrolled * (WrappedLines()-1))));
  flattened_lines.AdvanceIter(&(new_last_line = new_first_line), fb->lines-1);
  LayoutBackLine(&line, new_last_line.first);
  bool up = new_first_line < old_first_line;
  int dist = flattened_lines.Distance(new_first_line, old_first_line, fb->lines-1);
  if (first_offset) *first_offset = up ?  start_line_cutoff+1 :  end_line_adjust+1;
  if (first_ind)    *first_ind    = up ? -start_line-1        : -end_line-1;
  if (first_len)    *first_len    = up ? -start_line_adjust   :  end_line_cutoff;
  start_line        =  new_first_line.first;
  start_line_adjust = -new_first_line.second;
  end_line          =  new_last_line.first;
  end_line_adjust   =  new_last_line.second;
  end_line_cutoff   = line[  -end_line-1].Lines() -  new_last_line.second - 1;
  start_line_cutoff = line[-start_line-1].Lines() - new_first_line.second - 1;
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
    ScopedDrawMode drawmode(DrawMode::_2D);
    ScopedClearColor scc(bg_color);
    fb->fb.Attach();
    if (first_len)  wl += (fb->*update_cb)(&line[ind], -line_left, first_offset, min(dist, first_len), 0); 
    while (wl<dist) wl += (fb->*update_cb)(&line[decr ? --ind : ++ind], -line_left, 0, dist-wl, 0);
    fb->fb.Release();
  }
}

void TextArea::ChangeColors(Colors *C) {
  SetColors(C);
  // for (int i=1; i<=term_height; ++i) if (Line *L = GetTermLine(i)) L->Layout(L->data->box, true);
  Redraw();
}

void TextArea::Draw(const Box &b, int flag, Shader *shader) {
  if (shader) {
    float scale = shader->scale;
    glShadertoyShader(shader);
    shader->SetUniform3f("iChannelResolution", XY_or_Y(scale, b.w), XY_or_Y(scale, b.h), 1);
    shader->SetUniform2f("iScroll", XY_or_Y(scale, -line_fb.scroll.x * line_fb.w),
                         XY_or_Y(scale, -line_fb.scroll.y * line_fb.h - b.y));
  }
  int font_height = font->Height();
  LinesFrameBuffer *fb = GetFrameBuffer();
  if (flag & DrawFlag::CheckResized) CheckResized(b);
  if (clip) screen->gd->PushScissor(Box::DelBorder(b, *clip));
  fb->Draw(b.Position(), point(0, CommandLines() * font_height));
  if (clip) screen->gd->PopScissor();
  if (flag & DrawFlag::DrawCursor) DrawCursor(b.Position() + cursor.p);
  if (selection.enabled) mouse_gui.box.SetPosition(b.Position());
  if (selection.changing) DrawSelection();
  if (!clip && hover_link) DrawHoverLink(b);
}

void TextArea::DrawHoverLink(const Box &b) {
  bool outside_scroll_region = hover_link->line->data->outside_scroll_region;
  int fb_h = line_fb.Height();
  for (const auto &i : hover_link->box) {
    if (!i.w || !i.h) continue;
    point p = i.BottomLeft();
    p.y = outside_scroll_region ? (p.y + fb_h) : RingIndex::Wrap(p.y + line_fb.scroll.y * fb_h, fb_h);
    glLine(p, point(i.BottomRight().x, p.y), &Color::white);
  }
  if (hover_link_cb) hover_link_cb(hover_link);
}

bool TextArea::GetGlyphFromCoordsOffset(const point &p, Selection::Point *out, int sl, int sla) {
  LinesFrameBuffer *fb = GetFrameBuffer();
  int h = fb->Height(), fh = font->Height();
  int targ = reverse_line_fb ? ((h - p.y) / fh - sla) : (p.y / fh + sla);
  for (int i=sl, lines=0, ll; i<line.ring.count && lines<line_fb.lines; i++, lines += ll) {
    Line *L = &line[-i-1];
    if (lines + (ll = L->Lines()) <= targ) continue;
    L->data->glyphs.GetGlyphFromCoords(p, &out->char_ind, &out->glyph, targ - lines);
    out->glyph.y = lines * fh;
    out->line_ind = i;
    return true;
  }
  return false;
}

void TextArea::InitSelection() {
  mouse_gui.Activate();
  selection.gui_ind = mouse_gui.AddDragBox
    (Box(), MouseController::CoordCB(bind(&TextArea::DragCB, this, _1, _2, _3, _4)));
}

void TextArea::DrawSelection() {
  screen->gd->EnableBlend();
  screen->gd->FillColor(selection_color);
  selection.box.Draw(mouse_gui.box.BottomLeft());
}

void TextArea::DragCB(int button, int, int, int down) {
  point p = screen->mouse - mouse_gui.box.BottomLeft();
  LinesFrameBuffer *fb = GetFrameBuffer();
  Selection *s = &selection;
  if (!(s->changing = down)) {
    bool swap = (!reverse_line_fb && s->end < s->beg) || (reverse_line_fb && s->beg < s->end);
    CopyText(swap ? s->end : s->beg, swap ? s->beg : s->end);
    s->changing_previously = 0;
    return;
  }

  int scp = s->changing_previously, fh = font->Height(), h = fb->Height();
  if (scp) GetGlyphFromCoords((s->end.click = point(line_left+p.x, p.y)), &s->end);
  else   { GetGlyphFromCoords((s->beg.click = point(line_left+p.x, p.y)), &s->beg); s->end = s->beg; }

  bool swap = (!reverse_line_fb && s->end < s->beg) || (reverse_line_fb && s->beg < s->end);
  Box gb = swap ? s->end.glyph : s->beg.glyph;
  Box ge = swap ? s->beg.glyph : s->end.glyph;
  if (reverse_line_fb) { gb.y=h-gb.y-fh; ge.y=h-ge.y-fh; }
  point gbp = !reverse_line_fb ? gb.Position() : gb.Position() + point(gb.w, 0);
  point gep =  reverse_line_fb ? ge.Position() : ge.Position() + point(ge.w, 0);
  s->box = Box3(Box(fb->Width(), fb->Height()), gbp, gep, fh, fh);
  s->changing_previously = s->changing;
}

void TextArea::CopyText(const Selection::Point &beg, const Selection::Point &end) {
  string copy_text = CopyText(beg.line_ind, beg.char_ind, end.line_ind, end.char_ind, true);
  if (!copy_text.empty()) Clipboard::Set(copy_text);
}

string TextArea::CopyText(int beg_line_ind, int beg_char_ind, int end_line_ind, int end_char_ind, bool add_nl) {
  String16 copy_text;
  bool one_line = beg_line_ind == end_line_ind;
  int bc = (one_line && reverse_line_fb) ? end_char_ind : beg_char_ind;
  int ec = (one_line && reverse_line_fb) ? beg_char_ind : end_char_ind;

  for (int i = beg_line_ind, d = reverse_line_fb ? 1 : -1; /**/; i += d) {
    Line *l = &line[-i-1];
    int len = l->Size();
    if (i == beg_line_ind) {
      if (!l->Size() || bc < 0) len = -1;
      else {
        len = (one_line && ec >= 0) ? ec+1 : l->Size();
        copy_text += Substr(l->Text16(), bc, max(0, len - bc));
      }
    } else if (i == end_line_ind) {
      len = (end_char_ind >= 0) ? end_char_ind+1 : l->Size();
      copy_text += Substr(l->Text16(), 0, len);
    } else copy_text += l->Text16();

    if (add_nl && len == l->Size()) copy_text += String16(1, '\n');
    if (i == end_line_ind) break;
  }
  return String::ToUTF8(copy_text);
}

/* Editor */

Editor::Editor(Window *W, Font *F, File *I, bool Wrap) : TextArea(W, F), file(I) {
  reverse_line_fb = 1;
  opened = file && file->Opened();
  cmd_color = Color(Color::black, .5);
  line_fb.wrap = Wrap;
  file_line.node_value_cb = &LineOffset::GetLines;
  file_line.node_print_cb = &LineOffset::GetString;
}

void Editor::UpdateWrappedLines(int cur_font_size, int width) {
  wrapped_lines = 0;
  file_line.Clear();
  file->Reset();
  int offset = 0, wrap = Wrap(), ll;
  for (const char *l = file->NextLineRaw(&offset); l; l = file->NextLineRaw(&offset)) {
    wrapped_lines += (ll = wrap ? TextArea::font->Lines(l, width) : 1);
    file_line.val.Insert(LineOffset(offset, file->nr.record_len, ll));
  }
  file_line.LoadFromSortedVal();
}

int Editor::UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len) {
  if (!opened) return 0;
  LinesFrameBuffer *fb = GetFrameBuffer();
  bool width_changed = last_fb_width != fb->w, wrap = Wrap(), init = !wrapped_lines;
  if (width_changed) {
    last_fb_width = fb->w;
    if (wrap || init) UpdateWrappedLines(TextArea::font->size, fb->w);
    if (init) UpdateAnnotation();
  }

  bool resized = (width_changed && wrap) || last_fb_lines != fb->lines;
  int new_first_line = RoundF(v_scrolled * (wrapped_lines - 1)), new_last_line = new_first_line + fb->lines;
  int dist = resized ? fb->lines : abs(new_first_line - last_first_line), read_len = 0, bo = 0, l;
  if (!dist || !file_line.size()) return 0;

  bool redraw = dist >= fb->lines;
  if (redraw) { line.Clear(); fb_wrapped_lines = 0; }

  bool up = !redraw && new_first_line < last_first_line;
  if (first_offset) *first_offset = up ?  start_line_cutoff : end_line_adjust;
  if (first_len)    *first_len    = up ? -start_line_adjust : end_line_cutoff;

  pair<int, int> read_lines;
  if (dist < fb->lines) {
    if (up) read_lines = pair<int, int>(new_first_line, dist);
    else    read_lines = pair<int, int>(new_first_line + fb->lines - dist, dist);
  } else    read_lines = pair<int, int>(new_first_line, fb->lines);

  bool head_read = new_first_line == read_lines.first;
  bool tail_read = new_last_line  == read_lines.first + read_lines.second;
  bool short_read = !(head_read && tail_read), shorten_read = short_read && head_read && start_line_adjust;
  int past_end_lines = max(0, min(dist, read_lines.first + read_lines.second - wrapped_lines)), added = 0;
  read_lines.second = max(0, read_lines.second - past_end_lines);

  if      ( up && dist <= -start_line_adjust) { start_line_adjust += dist; read_lines.second=past_end_lines=0; }
  else if (!up && dist <=  end_line_cutoff)   { end_line_cutoff   -= dist; read_lines.second=past_end_lines=0; }

  LineMap::ConstIterator lib, lie;
  if (read_lines.second) {
    CHECK((lib = file_line.LesserBound(read_lines.first)).val);
    if (wrap) {
      if (head_read) start_line_adjust = min(0, lib.key - new_first_line);
      if (short_read && tail_read && end_line_cutoff) ++lib;
    }
    int last_read_line = read_lines.first + read_lines.second - 1;
    for (lie = lib; lie.val && lie.key <= last_read_line; ++lie) {
      auto v = lie.val;
      if (shorten_read && !(lie.key + v->wrapped_lines <= last_read_line)) break;
      if (v->size >= 0) read_len += (v->size + 1);
    }
    if (wrap && tail_read) {
      LineMap::ConstIterator i = lie;
      end_line_cutoff = max(0, (--i).key + i.val->wrapped_lines - new_last_line);
    }
  }

  string buf((read_len = X_or_1(read_len)-1), 0);
  if (read_len) {
    file->Seek(lib.val->offset, File::Whence::SET);
    CHECK_EQ(buf.size(), file->Read(&buf[0], buf.size()));
  }

  Line *L = 0;
  if (up) for (LineMap::ConstIterator li = lie; li != lib; bo += l + (L != 0), added++) {
    l = (--li).val->size;
    int e = li.val->modified;
    if (e) (L = line.PushBack())->AssignText(edits[e-1],                                     Flow::TextAnnotation(annotation.data(), PieceIndex()));
    else   (L = line.PushBack())->AssignText(StringPiece(buf.data() + read_len - bo - l, l), Flow::TextAnnotation(annotation.data(), li.val->annotation));
    fb_wrapped_lines += L->Layout(wrap ? fb->w : 0, true);
  }
  else for (LineMap::ConstIterator li = lib; li != lie; ++li, bo += l+1, added++) {
    l = li.val->size;
    int e = li.val->modified;
    if (e) (L = line.PushFront())->AssignText(edits[e-1],                      Flow::TextAnnotation(annotation.data(), PieceIndex()));
    else   (L = line.PushFront())->AssignText(StringPiece(buf.data() + bo, l), Flow::TextAnnotation(annotation.data(), li.val->annotation));
    fb_wrapped_lines += L->Layout(wrap ? fb->w : 0, true);
  }
  if (!up) for (int i=0; i<past_end_lines; i++, added++) { 
    (L = line.PushFront())->Clear();
    fb_wrapped_lines += L->Layout(wrap ? fb->w : 0, true);
  }

  CHECK_LT(line.ring.count, line.ring.size);
  if (!redraw) {
    for (bool first=1;;first=0) {
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

  last_fb_lines = fb->lines;
  last_first_line = new_first_line;
  UpdateCursor();
  UpdateCursorLine();
  return dist * (up ? -1 : 1);
}

void Editor::UpdateCursor() {
  cursor.i.y = min(cursor.i.y, line_fb.lines-1);
  cursor.i.x = min(cursor.i.x, GetCursorLine()->Size());
  cursor.p = point(GetCursorLine()->data->glyphs.Position(cursor.i.x).x,
                   line_fb.Height() - cursor.i.y*font->Height());
}

#ifdef LFL_LIBCLANG
struct ClangTokenVisitor {
  typedef function<void(ClangTokenVisitor*, int, int, int)> TokenCB;
  string filename, compile_command, working_directory;
  point last_token;
  TokenCB cb;
  ClangTokenVisitor(const string &f, const string &cc, const string &wd, const TokenCB &c) :
    filename(f), compile_command(cc), working_directory(wd), cb(c) {}

  void Visit() {
    vector<string> argv;
    vector<const char*> av = { "-xc++", "-std=c++11" };
    Split(compile_command, isspace, &argv);
    for (int i=1; i<(int)argv.size()-4; i++) if (!PrefixMatch(argv[i], "-O") && !PrefixMatch(argv[i], "-m")) av.push_back(argv[i].data());
    chdir(working_directory.c_str());
    CXIndex index = clang_createIndex(0, 0);
    CXTranslationUnit tu = clang_parseTranslationUnit(index, filename.c_str(), av.data(), av.size(), 0, 0, CXTranslationUnit_None);

    CXToken* tokens=0;
    unsigned num_tokens=0, by=0, bx=0, ey=0, ex=0;
    CXFile cf = clang_getFile(tu, filename.c_str());
    CXSourceRange sr = clang_getRange(clang_getLocationForOffset(tu, cf, 0),
                                      clang_getLocationForOffset(tu, cf, LocalFile(filename, "r").Size()));
    clang_tokenize(tu, sr, &tokens, &num_tokens);
    for (int i = 0; i < num_tokens; i++) {
      sr = clang_getTokenExtent(tu, tokens[i]);
      clang_getSpellingLocation(clang_getRangeStart(sr), NULL, &by, &bx, NULL);
      clang_getSpellingLocation(clang_getRangeEnd  (sr), NULL, &ey, &ex, NULL);

      if (1)                  CHECK_LE(last_token.y, (int)by);
      if (by == last_token.y) CHECK_LT(last_token.x, (int)bx);
      cb(this, clang_getTokenKind(tokens[i]), by, bx);
      last_token = point(bx, by);
    }

    clang_disposeTranslationUnit(tu);
    clang_disposeIndex(index);
  }
};
#endif

void Editor::UpdateAnnotation() {
#ifdef LFL_LIBCLANG
  if (!project) return;
  string filename = file->Filename();
  auto rule = project->build_rules.find(filename);
  bool have_rule = rule != project->build_rules.end();

  int fl_ind = 1;
  auto fl = file_line.Begin();
  LineOffset *file_line_data = 0;
  ClangTokenVisitor(filename, have_rule ? rule->second.cmd : "", have_rule ? rule->second.dir : "", ClangTokenVisitor::TokenCB([&]
    (ClangTokenVisitor *v, int kind, int line, int column) {
      int a = default_attr;
      switch (kind) {
        case CXToken_Punctuation:                                     break;
        case CXToken_Keyword:     Attr::SetFGColorIndex(&a, 5);       break;
        case CXToken_Identifier:  Attr::SetFGColorIndex(&a, 13);      break;
        case CXToken_Literal:     Attr::SetFGColorIndex(&a, 1);       break;
        case CXToken_Comment:     Attr::SetFGColorIndex(&a, 6);       break;
        default:                  ERROR("unknown token kind ", kind); break;
      }
      if (line != v->last_token.y) {
        if (v->last_token.y+1 < line && a != default_attr) PushBack(annotation, {0, a});
        if (fl_ind < line)
          for (++fl_ind, ++fl; fl.ind && fl_ind < line; ++fl_ind, ++fl) fl.val->annotation = PieceIndex(annotation.size()-1, 1);
        if (!fl.ind) return;
        (file_line_data = fl.val)->annotation.offset = annotation.size();
      }
      annotation.emplace_back(column-1, a);
      file_line_data->annotation.len++;
    })).Visit();
#endif
}

void Editor::Modify(bool erase, int c) {
  Line *L = GetCursorLine();
  if (!L || !cursor_line) return;
  if (cursor_line->modified <= 0) cursor_line->modified = edits.Insert(L->Text16())+1;
  String16 *b = &edits[cursor_line->modified-1];
  CHECK_LE(cursor.i.x, L->Size());
  CHECK_LE(cursor.i.x, b->size());
  if (erase)  b->erase(cursor.i.x, 1);
  else        b->insert(cursor.i.x, 1, c);
  if (erase)  L->Erase          (cursor.i.x, 1);
  else if (0) L->OverwriteTextAt(cursor.i.x, String16(1, c), default_attr);
  else        L->InsertTextAt   (cursor.i.x, String16(1, c), default_attr);
  UpdateLineFB(L, GetFrameBuffer());
}

/* Terminal */

#ifdef  LFL_TERMINAL_DEBUG
#define TerminalDebug(...) ERRORf(__VA_ARGS__)
#define TerminalTrace(...) printf("%s", StrCat(logtime(Now()), " ", StringPrintf(__VA_ARGS__)).c_str())
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
  c[normal_index] = c[7];
  c[bold_index]   = c[15];
  c[bg_index]     = c[0];
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
  c[normal_index] = c[12];
  c[bold_index]   = c[14];
  c[bg_index]     = c[8];
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
  c[normal_index] = c[12];
  c[bold_index]   = c[14];
  c[bg_index]     = c[8];
}

Terminal::Terminal(ByteSink *O, Window *W, Font *F) : TextArea(W, F), sink(O), fb_cb(bind(&Terminal::GetFrameBuffer, this, _1)) {
  CHECK(F->fixed_width || (F->flag & FontDesc::Mono));
  wrap_lines = write_newline = insert_mode = 0;
  line.SetAttrSource(this);
  SetColors(Singleton<SolarizedDarkColors>::Get());
  cursor.attr = default_attr;
  token_processing = 1;
  cmd_prefix = "";
}

void Terminal::Resized(const Box &b) {
  int old_term_width = term_width, old_term_height = term_height;
  SetDimension(b.w / font->FixedWidth(), b.h / font->Height());
  TerminalDebug("Resized %d, %d <- %d, %d\n", term_width, term_height, old_term_width, old_term_height);
  bool grid_changed = term_width != old_term_width || term_height != old_term_height;
  if (grid_changed || first_resize) if (sink) sink->IOCtlWindowSize(term_width, term_height); 

  int height_dy = term_height - old_term_height;
  if      (height_dy > 0) TextArea::Write(string(height_dy, '\n'), 0);
  else if (height_dy < 0 && term_cursor.y < old_term_height) line.PopBack(-height_dy);

  term_cursor.x = min(term_cursor.x, term_width);
  term_cursor.y = min(term_cursor.y, term_height);
  TextArea::Resized(b);
  if (clip) clip = UpdateClipBorder();
  ResizedLeftoverRegion(b.w, b.h);
}

void Terminal::ResizedLeftoverRegion(int w, int h, bool update_fb) {
  if (!cmd_fb.SizeChanged(w, h, font)) return;
  if (update_fb) {
    for (int i=0; i<start_line;      i++) MoveToOrFromScrollRegion(&cmd_fb, &line[-i-1],             point(0,GetCursorY(term_height-i)), LinesFrameBuffer::Flag::Flush);
    for (int i=0; i<skip_last_lines; i++) MoveToOrFromScrollRegion(&cmd_fb, &line[-line_fb.lines+i], point(0,GetCursorY(i+1)),           LinesFrameBuffer::Flag::Flush);
  }
  cmd_fb.SizeChangedDone();
  last_fb = 0;
}

void Terminal::MoveToOrFromScrollRegion(TextGUI::LinesFrameBuffer *fb, TextGUI::Line *l, const point &p, int flag) {
  int plpy = l->p.y;
  fb->Update(l, p, flag);
  l->data->outside_scroll_region = fb != &line_fb;
  int delta_y = plpy - l->p.y + line_fb.Height() * (l->data->outside_scroll_region ? -1 : 1);
  for (auto &i : l->data->links) {
    i.second->box += point(0, delta_y);
    for (auto &j : i.second->hitbox) i.second->gui->IncrementBoxY(delta_y, -1, j);
  }
}

void Terminal::SetScrollRegion(int b, int e, bool release_fb) {
  if (b<0 || e<0 || e>term_height || b>e) { TerminalDebug("%d-%d outside 1-%d\n", b, e, term_height); return; }
  int prev_region_beg = scroll_region_beg, prev_region_end = scroll_region_end, font_height = font->Height();
  scroll_region_beg = b;
  scroll_region_end = e;
  bool no_region = !scroll_region_beg || !scroll_region_end || (scroll_region_beg == 1 && scroll_region_end == term_height);
  skip_last_lines = no_region ? 0 : scroll_region_beg - 1;
  start_line_adjust = start_line = no_region ? 0 : term_height - scroll_region_end;
  clip = no_region ? 0 : UpdateClipBorder();
  ResizedLeftoverRegion(line_fb.w, line_fb.h, false);

  if (release_fb) { last_fb=0; screen->gd->DrawMode(DrawMode::_2D, 0); }
  int   prev_beg_or_1=X_or_1(prev_region_beg),     prev_end_or_ht=X_or_Y(  prev_region_end, term_height);
  int scroll_beg_or_1=X_or_1(scroll_region_beg), scroll_end_or_ht=X_or_Y(scroll_region_end, term_height);

  if (scroll_beg_or_1 != prev_beg_or_1 || prev_end_or_ht != scroll_end_or_ht) GetPrimaryFrameBuffer();
  for (int i =  scroll_beg_or_1; i <    prev_beg_or_1; i++) MoveToOrFromScrollRegion(&line_fb, GetTermLine(i),   line_fb.BackPlus(point(0, (term_height-i+1)*font_height)), LinesFrameBuffer::Flag::NoLayout);
  for (int i =   prev_end_or_ht; i < scroll_end_or_ht; i++) MoveToOrFromScrollRegion(&line_fb, GetTermLine(i+1), line_fb.BackPlus(point(0, (term_height-i)  *font_height)), LinesFrameBuffer::Flag::NoLayout);

  if (prev_beg_or_1 < scroll_beg_or_1 || scroll_end_or_ht < prev_end_or_ht) GetSecondaryFrameBuffer();
  for (int i =    prev_beg_or_1; i < scroll_beg_or_1; i++) MoveToOrFromScrollRegion(&cmd_fb, GetTermLine(i),   point(0, GetCursorY(i)),   LinesFrameBuffer::Flag::NoLayout);
  for (int i = scroll_end_or_ht; i <  prev_end_or_ht; i++) MoveToOrFromScrollRegion(&cmd_fb, GetTermLine(i+1), point(0, GetCursorY(i+1)), LinesFrameBuffer::Flag::NoLayout);
  if (release_fb) cmd_fb.fb.Release();
}

void Terminal::SetDimension(int w, int h) {
  term_width  = w;
  term_height = h;
  ScopedClearColor scc(bg_color);
  if (!line.Size()) TextArea::Write(string(term_height, '\n'), 0);
}

Border *Terminal::UpdateClipBorder() {
  int font_height = font->Height();
  clip_border.top    = font_height * skip_last_lines;
  clip_border.bottom = font_height * start_line_adjust;
  return &clip_border;
}

void Terminal::MoveLines(int sy, int ey, int dy, bool move_fb_p) {
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

void Terminal::UpdateToken(Line *L, int word_offset, int word_len, int update_type, const LineTokenProcessor *token) {
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
  TextArea::Draw(b, false, shader);
  if (shader) shader->SetUniform2f("iScroll", 0, XY_or_Y(shader->scale, -b.y));
  if (clip) {
    { Scissor s(Box::TopBorder(b, *clip)); cmd_fb.Draw(b.Position(), point(), false); }
    { Scissor s(Box::BotBorder(b, *clip)); cmd_fb.Draw(b.Position(), point(), false); }
    if (hover_link) DrawHoverLink(b);
  }
  if (flag & DrawFlag::DrawCursor) TextGUI::DrawCursor(b.Position() + cursor.p);
  if (selection.changing) DrawSelection();
}

void Terminal::Write(const StringPiece &s, bool update_fb, bool release_fb) {
  if (!MainThread()) return RunInMainThread(new Callback(bind(&Terminal::WriteCB, this, s.str(), update_fb, release_fb)));
  TerminalTrace("Terminal: Write('%s', %zd)\n", CHexEscapeNonAscii(s.str()).c_str(), s.size());
  screen->gd->DrawMode(DrawMode::_2D, 0);
  ScopedClearColor scc(bg_color);
  last_fb = 0;
  for (int i = 0; i < s.len; i++) {
    const unsigned char c = *(s.begin() + i);
    if (c == 0x18 || c == 0x1a) { /* CAN or SUB */ parse_state = State::TEXT; continue; }
    if (parse_state == State::ESC) {
      parse_state = State::TEXT; // default next state
      TerminalTrace("ESC %c\n", c);
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
        case 'c': Reset();                         break;
        case 'D': Newline();                       break;
        case 'M': NewTopline();                    break;
        case '7': saved_term_cursor = term_cursor; break;
        case '8': term_cursor = point(Clamp(saved_term_cursor.x, 1, term_width),
                                      Clamp(saved_term_cursor.y, 1, term_height));
        default: TerminalDebug("unhandled escape %c (%02x)\n", c, c);
      }
    } else if (parse_state == State::CHARSET) {
      TerminalTrace("charset G%d %c\n", 1+parse_charset-'(', c);
      parse_state = State::TEXT;

    } else if (parse_state == State::OSC) {
      if (!parse_osc_escape) {
        if (c == 0x1b) { parse_osc_escape = 1; continue; }
        if (c != 0x07) { parse_osc       += c; continue; }
      }
      else if (c != 0x5c) { TerminalDebug("within-OSC-escape %c (%02x)\n", c, c); parse_state = State::TEXT; continue; }
      parse_state = State::TEXT;

      if (parse_osc.size() > 2 && parse_osc[1] == ';' && Within(parse_osc[0], '0', '2')) screen->SetCaption(parse_osc.substr(2));
      else TerminalDebug("unhandled OSC %s\n", parse_osc.c_str());

    } else if (parse_state == State::CSI) {
      // http://en.wikipedia.org/wiki/ANSI_escape_code#CSI_codes
      if (c < 0x40 || c > 0x7e) { parse_csi += c; continue; }
      TerminalTrace("CSI %s%c (cur=%d,%d)\n", parse_csi.c_str(), c, term_cursor.x, term_cursor.y);
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
          else if (parse_csi_argv[0] == 2) { Clear(); break; }
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
          { TerminalDebug("y=%s scrollregion=%d-%d\n", term_cursor.DebugString().c_str(), scroll_region_beg, scroll_region_end); break; }
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
          else TerminalDebug("unhandled CSI-h mode = %d av00 = %c i= %s\n", mode, parse_csi_argv00, intermed.str().c_str());
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
          else TerminalDebug("unhandled CSI-l mode = %d av00 = %c i= %s\n", mode, parse_csi_argv00, intermed.str().c_str());
        } break;
        case 'm':
        for (int i=0; i<parse_csi_argc; i++) {
          int sgr = parse_csi_argv[i]; // select graphic rendition
          if      (sgr >= 30 && sgr <= 37) Attr::SetFGColorIndex(&cursor.attr, sgr-30);
          else if (sgr >= 40 && sgr <= 47) Attr::SetBGColorIndex(&cursor.attr, sgr-40);
          else switch(sgr) {
            case 0:         cursor.attr  =  default_attr;                              break;
            case 1:         cursor.attr |=  Attr::Bold;                                break;
            case 3:         cursor.attr |=  Attr::Italic;                              break;
            case 4:         cursor.attr |=  Attr::Underline;                           break;
            case 5: case 6: cursor.attr |=  Attr::Blink;                               break;
            case 7:         cursor.attr |=  Attr::Reverse;                             break;
            case 22:        cursor.attr &= ~Attr::Bold;                                break;
            case 23:        cursor.attr &= ~Attr::Italic;                              break;
            case 24:        cursor.attr &= ~Attr::Underline;                           break;
            case 25:        cursor.attr &= ~Attr::Blink;                               break;
            case 27:        cursor.attr &= ~Attr::Reverse;                             break;
            case 39:        Attr::SetFGColorIndex(&cursor.attr, colors->normal_index); break;
            case 49:        Attr::SetBGColorIndex(&cursor.attr, colors->bg_index);     break;
            default:        TerminalDebug("unhandled SGR %d\n", sgr);
          }
        } break;
        case 'p':
          if (parse_csi_argv00 == '!') { /* soft reset http://vt100.net/docs/vt510-rm/DECSTR */
            insert_mode = false;
            SetScrollRegion(1, term_height);
          }
          else TerminalDebug("Unhandled CSI-p %c\n", parse_csi_argv00);
          break;
        case 'r':
          if (parse_csi_argc == 2) SetScrollRegion(parse_csi_argv[0], parse_csi_argv[1]);
          else TerminalDebug("invalid scroll region argc %d\n", parse_csi_argc);
          break;
        default:
          TerminalDebug("unhandled CSI %s%c\n", parse_csi.c_str(), c);
      }
    } else {
      // http://en.wikipedia.org/wiki/C0_and_C1_control_codes#C0_.28ASCII_and_derivatives.29
      bool C0_control = (c >= 0x00 && c <= 0x1f) || c == 0x7f;
      bool C1_control = (c >= 0x80 && c <= 0x9f);
      if (C0_control || C1_control) {
        TerminalTrace("C0/C1 control: %02x\n", c);
        FlushParseText();
      }
      if (C0_control) switch(c) {
        case '\a':   TerminalDebug("%s", "bell");              break; // bell
        case '\b':   term_cursor.x = max(term_cursor.x-1, 1);  break; // backspace
        case '\t':   TabNext(1);                               break; // tab 
        case '\r':   term_cursor.x = 1;                        break; // carriage return
        case '\x1b': parse_state = State::ESC;                 break;
        case '\x14': case '\x15': case '\x7f':                 break; // shift charset in, out, delete
        case '\n':   case '\v':   case '\f':   Newline();      break; // line feed, vertical tab, form feed
        default:                               TerminalDebug("unhandled C0 control %02x\n", c);
      } else if (0 && C1_control) {
        if (0) {}
        else TerminalDebug("unhandled C1 control %02x\n", c);
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
  font = GetAttr(cursor.attr)->font;
  String16 input_text = String::ToUTF16(parse_text, &consumed);
  TerminalTrace("Terminal: (cur=%d,%d) FlushParseText('%s').size = [%zd, %d]\n", term_cursor.x, term_cursor.y,
                StringPiece(parse_text.data(), consumed).str().c_str(), input_text.size(), consumed);
  for (int wrote = 0; wrote < input_text.size(); wrote += write_size) {
    if (wrote || term_cursor.x > term_width) Newline(true);
    Line *l = GetCursorLine();
    LinesFrameBuffer *fb = GetFrameBuffer(l);
    int remaining = input_text.size() - wrote, o = term_cursor.x-1;
    write_size = min(remaining, term_width - o);
    String16Piece input_piece(input_text.data() + wrote, write_size);
    update_size = l->UpdateText(o, input_piece, cursor.attr, term_width, &append);
    TerminalTrace("Terminal: FlushParseText: UpdateText(%d, %d, '%s').size = [%d, %d] attr=%d\n",
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
  if (clip && term_cursor.y == scroll_region_end) { Scroll(-1); last_fb=&line_fb; }
  else if (term_cursor.y == term_height) { if (!clip) { PushBackLines(1); last_fb=&line_fb; } }
  else term_cursor.y = min(term_height, term_cursor.y+1);
  if (carriage_return) term_cursor.x = 1;
}

void Terminal::NewTopline() {
  if (clip && term_cursor.y == scroll_region_beg) { Scroll(1); last_fb=&line_fb; }
  else if (term_cursor.y == 1) { if (!clip) PushFrontLines(1); last_fb=&line_fb; }
  else term_cursor.y = max(1, term_cursor.y-1);
}

void Terminal::TabNext(int n) { term_cursor.x = min(NextMultipleOfN(term_cursor.x, tab_width) + 1, term_width); }
void Terminal::TabPrev(int n) {
  if (tab_stop.size()) TerminalDebug("%s\n", "variable tab stop not implemented");
  else if ((term_cursor.x = max(PrevMultipleOfN(term_cursor.x - max(0, n-1)*tab_width - 2, tab_width), 1)) != 1) term_cursor.x++;
}

void Terminal::Clear() {
  for (int i=1; i<=term_height; ++i) GetTermLine(i)->Clear();
  Redraw(true);
}

void Terminal::Redraw(bool attach) {
  bool prev_clip = clip;
  int prev_scroll_beg = scroll_region_beg, prev_scroll_end = scroll_region_end;
  SetScrollRegion(1, term_height, true);
  TextArea::Redraw(true);
  if (prev_clip) SetScrollRegion(prev_scroll_beg, prev_scroll_end, true);
}

void Terminal::Reset() {
  term_cursor.x = term_cursor.y = 1;
  scroll_region_beg = scroll_region_end = 0;
  clip = 0;
  Clear();
}

/* Console */

bool Console::Toggle() {
  if (!TextGUI::Toggle()) return false;
  bool last_animating = animating;
  Time now = Now(), elapsed = now - anim_begin;
  anim_begin = now - (elapsed < anim_time ? anim_time - elapsed : Time(0));
  animating = (elapsed = now - anim_begin) < anim_time;
  if (animating && !last_animating && animating_cb) animating_cb();
  return true;
}

void Console::Draw() {
  if (!ran_startcmd && (ran_startcmd = 1)) if (startcmd.size()) Run(startcmd);

  drawing = 1;
  Time now=Now(), elapsed;
  bool last_animating = animating;
  int h = active ? (int)(screen->height*screen_percent) : 0;
  if ((animating = (elapsed = now - anim_begin) < anim_time)) {
    if (active) h = (int)(screen->height*(  (double)elapsed.count()/anim_time.count())*screen_percent);
    else        h = (int)(screen->height*(1-(double)elapsed.count()/anim_time.count())*screen_percent);
  }
  if (!animating) {
    if (last_animating && animating_cb) animating_cb();
    if (!active) { drawing = 0; return; }
  }

  screen->gd->FillColor(color);
  if (blend) screen->gd->EnableBlend(); 
  else       screen->gd->DisableBlend();

  int y = bottom_or_top ? 0 : screen->height-h;
  Box(0, y, screen->width, h).Draw();

  screen->gd->SetColor(Color::white);
  TextArea::Draw(Box(0, y, screen->width, h), DrawFlag::DrawCursor | DrawFlag::CheckResized);
  TextGUI::Draw(Box(0, y, screen->width, font->Height()));
}

/* Dialog */

Dialog::Dialog(float w, float h, int flag) : GUI(screen), font(Fonts::Get(FLAGS_default_font, "", 14, Color::white)), menuicon(Fonts::Get("MenuAtlas", "", 0, Color::white, Color::clear, 0)) {
  screen->dialogs.push_back(this);
  box = screen->Box().center(screen->Box(w, h));
  fullscreen = flag & Flag::Fullscreen;
  Activate();
}

void Dialog::Layout() {
  Reset();
  if (fullscreen) return;
  title         = Box(0,       0,      box.w, screen->height*.05);
  resize_left   = Box(0,       -box.h, 3,     box.h);
  resize_right  = Box(box.w-3, -box.h, 3,     box.h);
  resize_bottom = Box(0,       -box.h, box.w, 3);

  Box close = Box(box.w-10, title.top()-10, 10, 10);
  AddClickBox(resize_left,   MouseController::CB(bind(&Dialog::Reshape,     this, &resizing_left)));
  AddClickBox(resize_right,  MouseController::CB(bind(&Dialog::Reshape,     this, &resizing_right)));
  AddClickBox(resize_bottom, MouseController::CB(bind(&Dialog::Reshape,     this, &resizing_bottom)));
  AddClickBox(title,         MouseController::CB(bind(&Dialog::Reshape,     this, &moving)));
  AddClickBox(close,         MouseController::CB(bind(&Dialog::MarkDeleted, this)));

  int attr_id = child_box.attr.GetAttrId(Drawable::Attr(menuicon));
  child_box.PushBack(close, attr_id, menuicon ? menuicon->FindGlyph(0) : 0);

  if (title_text.size()) {
    Box title_text_size;
    font->Size(title_text, &title_text_size);
    font->Shape(title_text, Box(title.centerX(title_text_size.w), title.centerY(title_text_size.h), 0, 0), &child_box);
  }
}

void Dialog::Draw() {
  bool resizing = resizing_left || resizing_right || resizing_top || resizing_bottom;
  if (child_box.data.empty() && !resizing) Layout();
  if (moving) box.SetPosition(win_start + screen->mouse - mouse_start);

  Box outline = BoxAndTitle();
  static const int min_width = 50, min_height = 1;
  if (resizing_left)   MinusPlus(&outline.x, &outline.w, max(-outline.w + min_width,            (int)(mouse_start.x - screen->mouse.x)));
  if (resizing_bottom) MinusPlus(&outline.y, &outline.h, max(-outline.h + min_height + title.h, (int)(mouse_start.y - screen->mouse.y)));
  if (resizing_right)  outline.w += max(-outline.w + min_width, (int)(screen->mouse.x - mouse_start.x));

  if (!app->input->MouseButton1Down()) {
    if (resizing) {
      box = Box(outline.x, outline.y, outline.w, outline.h - title.h);
      Layout();
    }
    moving = resizing_left = resizing_right = resizing_top = resizing_bottom = 0;
  }

  screen->gd->FillColor(color);
  box.Draw();

  screen->gd->SetColor(color + Color(0,0,0,(int)(color.A()*.25)));
  (title + box.TopLeft()).Draw();
  screen->gd->SetColor(Color::white);

  if (moving || resizing_left || resizing_right || resizing_top || resizing_bottom)
    BoxOutline().Draw(outline);

  child_box.Draw(box.TopLeft());
}

void Dialog::BringToFront() {
  if (screen->top_dialog == this) return;
  for (vector<Dialog*>::iterator i = screen->dialogs.begin(); i != screen->dialogs.end(); ++i) (*i)->zsort++; zsort = 0;
  sort(screen->dialogs.begin(), screen->dialogs.end(), LessThan);
  screen->top_dialog = this;
}

void MessageBoxDialog::Draw() {
  Dialog::Draw();
  { Scissor scissor(box); font->Draw(message, point(box.centerX(messagesize.w), box.centerY(messagesize.h)));  }
}

SliderDialog::SliderDialog(const string &t, const SliderDialog::UpdatedCB &cb, float scrolled, float total, float inc) :
  Dialog(.3, .05), updated(cb), slider(this, Box(), Widget::Scrollbar::Flag::Horizontal) {
  title_text = t;
  slider.scrolled = scrolled;
  slider.doc_height = total;
  slider.increment = inc;
}

SliderFlagDialog::SliderFlagDialog(const string &fn, float total, float inc) :
  SliderDialog(fn, bind(&SliderFlagDialog::Updated, this, _1), atof(Singleton<FlagMap>::Get()->Get(fn)) / total, total, inc),
  flag_name(fn), flag_map(Singleton<FlagMap>::Get()) {}

EditorDialog::EditorDialog(Window *W, Font *F, File *I, float w, float h, int flag) :
  Dialog(w, h, flag), editor(W, F, I, flag & Flag::Wrap), v_scrollbar(this, Box(), Widget::Scrollbar::Flag::AttachedNoCorner),
  h_scrollbar(this, Box(), Widget::Scrollbar::Flag::AttachedHorizontalNoCorner) {}

void EditorDialog::Layout() {
  Dialog::Layout();
  Widget::Scrollbar::AttachContentBox(&(content_box = Box(0, -box.h, box.w, box.h)), &v_scrollbar, editor.Wrap() ? NULL : &h_scrollbar);
}

void EditorDialog::Draw() {
  bool wrap = editor.Wrap();
  if (1)     editor.active = fullscreen || screen->top_dialog == this;
  if (1)     editor.v_scrolled = v_scrollbar.AddScrollDelta(editor.v_scrolled);
  if (!wrap) editor.h_scrolled = h_scrollbar.AddScrollDelta(editor.h_scrolled);
  if (1)     editor.UpdateScrolled();
  if (1)     Dialog::Draw();
  if (1)     editor.Draw(content_box + box.TopLeft(), Editor::DrawFlag::DrawCursor | Editor::DrawFlag::CheckResized);
  if (1)     v_scrollbar.Update();
  if (!wrap) h_scrollbar.Update();
}

#ifdef LFL_QT
void Dialog::MessageBox(const string &n) {
  Mouse::ReleaseFocus();
  QMessageBox *msg = new QMessageBox();
  msg->setAttribute(Qt::WA_DeleteOnClose);
  msg->setText("MesssageBox");
  msg->setInformativeText(n.c_str());
  msg->setModal(false);
  msg->open();
}
void Dialog::TextureBox(const string &n) {}

#else /* LFL_QT */

void Dialog::MessageBox(const string &n) {
  Mouse::ReleaseFocus();
  new MessageBoxDialog(n);
}
void Dialog::TextureBox(const string &n) {
  Mouse::ReleaseFocus();
  new TextureBoxDialog(n);
}
#endif /* LFL_QT */

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
    topology_type topo(gen, 0, 0, screen->width, screen->height);
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

HelperGUI::Label::Label(const Box &w, const string &d, int h, Font *f, float ly, float lx) :
  target(w), target_center(target.center()), hint(h), description(d) {
  lx *= screen->width; ly *= screen->height;
  label_center = target_center;
  if      (h == Hint::UP   || h == Hint::UPLEFT   || h == Hint::UPRIGHT)   label_center.y += ly;
  else if (h == Hint::DOWN || h == Hint::DOWNLEFT || h == Hint::DOWNRIGHT) label_center.y -= ly;
  if      (h == Hint::UPRIGHT || h == Hint::DOWNRIGHT)                     label_center.x += lx;
  else if (h == Hint::UPLEFT  || h == Hint::DOWNLEFT)                      label_center.x -= lx;
  f->Size(description.c_str(), &label);
  AssignLabelBox();
}

void HelperGUI::Draw() {
  for (auto i = label.begin(); i != label.end(); ++i) {
    glLine(point(i->label_center.x, i->label_center.y),
           point(i->target_center.x, i->target_center.y), &font->fg);
    screen->gd->FillColor(Color::black);
    Box::AddBorder(i->label, 4, 0).Draw();
    font->Draw(i->description, point(i->label.x, i->label.y));
  }
}

}; // namespace LFL
