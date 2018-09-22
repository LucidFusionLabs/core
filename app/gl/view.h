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

#ifndef LFL_CORE_APP_GL_VIEW_H__
#define LFL_CORE_APP_GL_VIEW_H__

#include "core/app/flow.h"

namespace LFL {
DECLARE_bool(multitouch);
DECLARE_bool(draw_grid);
DECLARE_bool(console);
DECLARE_string(console_font);
DECLARE_int(console_font_flag);
DECLARE_FLAG(testbox, Box);

struct View : public Drawable {
  const char *name;
  Window *root;
  Box box;
  MouseController mouse;
  DrawableBoxArray child_box;
  vector<View*> child_view;
  View *parent=0;
  bool visible=1, active=0;

  View(Window *R, const char *n, const Box &B=Box()) : name(n), root(R), box(B), mouse(this) {}
  virtual ~View() {}

  DrawableBoxArray *ResetView() { ClearView(); return &child_box; }
  void ClearView() { child_view.clear(); child_box.Clear(); mouse.Clear(); }
  void UpdateBox(const Box &b, int draw_box_ind, int input_box_ind);
  void UpdateBoxX(int x, int draw_box_ind, int input_box_ind);
  void UpdateBoxY(int y, int draw_box_ind, int input_box_ind);
  void IncrementBoxY(int y, int draw_box_ind, int input_box_ind);

  virtual bool Activate() { if ( active) return 0; active = 1; return 1; }
  virtual bool Deactivate() { if (!active) return 0; active = 0; return 1; }
  virtual bool NotActive(const point &p) const { return !active; }
  virtual bool ToggleActive() { if ((active = !active)) Activate(); else Deactivate(); return active; }
  virtual void SetParentView(View *v) { parent = v; }
  virtual void AppendChildView(View *v) { child_view.push_back(v); v->SetParentView(this); }
  virtual void AppendChildView(View *v, const Box &b) { v->box = b; AppendChildView(v); }
  virtual MouseControllerInterface *GetMouseController() { return &mouse; }
  virtual KeyboardControllerInterface *GetKeyboardController() { return nullptr; }

  virtual point RelativePosition(const point &p) const { return p - box.TopLeft(); }
  virtual point RootPosition(const point &p) const;
  virtual void HandleTextMessage(const string &s) {}

  virtual void SetLayoutDirty() { child_box.Clear(); child_view.clear(); }
  virtual View *Layout(Flow *flow=nullptr) { return this; }
  virtual void Draw(GraphicsContext*, const LFL::Box &b) const override { const_cast<View*>(this)->Draw(b.TopLeft()); }
  virtual void Draw(const point &p);
  virtual void ResetGL(int flag) {}
};

struct Widget {
  struct Interface {
    View *view;
    vector<int> hitbox;
    int drawbox_ind=-1;
    bool del_hitbox=0;
    virtual ~Interface() { if (del_hitbox) DelHitBox(); }
    Interface(View *v) : view(v) {}

    void AddClickBox(const Box  &b, MouseControllerCallback cb) {                   hitbox.push_back(view->mouse.AddClickBox(b, move(cb))); }
    void AddHoverBox(const Box  &b, MouseControllerCallback cb) {                   hitbox.push_back(view->mouse.AddHoverBox(b, move(cb))); }
    void AddDragBox (const Box  &b, MouseControllerCallback cb) {                   hitbox.push_back(view->mouse.AddDragBox (b, move(cb))); }
    void AddClickBox(const Box3 &t, MouseControllerCallback cb) { for (auto &b : t) hitbox.push_back(view->mouse.AddClickBox(b, move(cb))); }
    void AddHoverBox(const Box3 &t, MouseControllerCallback cb) { for (auto &b : t) hitbox.push_back(view->mouse.AddHoverBox(b, move(cb))); }
    void AddDragBox (const Box3 &t, MouseControllerCallback cb) { for (auto &b : t) hitbox.push_back(view->mouse.AddDragBox (b, move(cb))); }
    void DelHitBox() { for (auto &i : hitbox) view->mouse.hit.Erase(i); hitbox.clear(); }
    MouseController::HitBox &GetHitBox(int i=0) const { return view->mouse.hit[hitbox[i]]; }
    Box GetHitBoxBox(int i=0) const { return Box::Add(GetHitBox(i).box, view->box.TopLeft()); }
    DrawableBox *GetDrawBox() const { return drawbox_ind >= 0 ? VectorGet(view->child_box.data, drawbox_ind) : 0; }
  };

  struct Button : public Interface {
    Box box;
    Font *font=0;
    string text;
    point textsize;
    Drawable *image=0;
    BoxOutline box_outline;
    Color *solid=0, *outline=0, *outline_topleft=0, *outline_bottomright=0;
    MouseControllerCallback cb;
    bool init=0, hover=0;
    int decay=0, v_align=Align::VerticalCenter, v_offset=0, outline_w=3;
    Button() : Interface(0) {}
    Button(View *V, Drawable *I, const string &T, const MouseControllerCallback &CB)
      : Interface(V), text(T), image(I), cb(CB), init(1) {}

    Color *SetOutlineColor(Color *c) { return (outline_topleft = outline_bottomright = c); }
    void SetBox(const Box &b) { box=b; hitbox.clear(); AddClickBox(box, cb); init=0; }
    void EnableHover() { AddHoverBox(box, MouseController::CB(bind(&Button::ToggleHover, this))); }
    void ToggleHover() { hover = !hover; }

    void Layout        (Flow *flow, Font *f, const point &d) { box.SetDimension(d); Layout(flow, f); }
    void Layout        (Flow *flow, Font *f);
    void LayoutBox     (Flow *flow, Font *f, const Box &b);
    void LayoutComplete(Flow *flow, Font *f, const Box &b);
  };

  struct Slider : public Interface {
    struct Flag { enum { Attached=1, Horizontal=2, NoCorner=4, AttachedHorizontal=Attached|Horizontal,
      AttachedNoCorner=Attached|NoCorner, AttachedHorizontalNoCorner=AttachedHorizontal|NoCorner }; };
    Box track;
    int flag=0, doc_height=200, dot_size=15, outline_w=1;
    float scrolled=0, last_scrolled=0, min_value=0, max_value=0, increment=20;
    Color color=Color(15, 15, 15, 55), *outline_topleft=&Color::grey80, *outline_bottomright=&Color::grey50;
    FontRef menuicon;
    bool changed=0, arrows=1;
    StringCB changed_cb;
    virtual ~Slider() {}
    Slider(View *V, int f=Flag::Attached);

    float Percent() const;
    bool HasChanged();
    void LayoutFixed(const Box &w);
    void LayoutAttached(const Box &w);
    void Layout(int aw, int ah, bool flip);
    void OnChanged();
    void UpdateDotPosition();
    void SetDocHeight(int v) { doc_height = v; }
    void DragScrollDot(int, point, point, int);
    void SetScrolled(float v);
    void ScrollUp();
    void ScrollDown();
    float ScrollDelta();
    float AddScrollDelta(float cur_val);
    static void AttachContentBox(Box *b, Slider *vs, Slider *hs);
  };
  
  struct Divider : public Interface {
    int size=0, start=0, start_size=0, min_size=0, max_size=-1;
    bool horizontal=1, direction=0, changing=0, changed=0;
    Divider(View *V, bool H, int S) : Interface(V), size(S), horizontal(H) {}
    void ApplyConstraints();
    void LayoutDivideTop   (const Box &in, Box *top,  Box *bottom, int offset=0);
    void LayoutDivideBottom(const Box &in, Box *top,  Box *bottom, int offset=0);
    void LayoutDivideLeft  (const Box &in, Box *left, Box *right,  int offset=0);
    void LayoutDivideRight (const Box &in, Box *left, Box *right,  int offset=0);
    void DragCB(int b, point p, point d, int down);
  };
};

struct TextBox : public View, public TextboxController {
  struct Line;
  struct Lines;
  struct Colors {
    int normal_index=0, bold_index=0, background_index=7;
    int SetDefaultAttr(int da=0) const;
    virtual const Color *GetColor(int) const = 0;
  };

  struct Style : public Drawable::AttrSource {
    enum { Bold=1<<16, Underline=1<<17, Blink=1<<18, Reverse=1<<19, Italic=1<<20, Link=1<<21 };
    static int GetFGColorIndex(int a) { int c = a & 0xff; return c | (((a & Bold) && c<16) ? (1<<3) : 0); }
    static int GetBGColorIndex(int a) { return (a>>8) & 0xff; }
    static int SetFGColorIndex(int a, int c) { return (a & ~0x00ff) | ((c & 0xff)     ); }
    static int SetBGColorIndex(int a, int c) { return (a & ~0xff00) | ((c & 0xff) << 8); }
    static int SetColorIndex(int a, int fg, int bg) { return SetFGColorIndex(SetBGColorIndex(a, bg), fg); } 
    FontRef font;
    Colors *colors=0;
    mutable Drawable::Attr last_attr;
    Style(const FontRef &F) : font(F) {}
    virtual const Drawable::Attr *GetAttr(int attr) const override;
  };

  struct Control : public Widget::Interface {
    Box3 box;
    string val;
    Line *line=0;
    shared_ptr<Texture> image;
    Control(Line *P, View *V, const Box3 &b, string, MouseControllerCallback);
    virtual ~Control() { if (line->parent->hover_control == this) line->parent->hover_control = 0; }
    void Hover(int, point, point, int down) { line->parent->hover_control = down ? this : 0; }
  };

  struct LineData {
    Box box;
    Flow flow;
    DrawableBoxArray glyphs;
    bool wrapped=0, outside_scroll_region=0;
    unordered_map<int, shared_ptr<Control>> controls;
    void Clear() { controls.clear(); glyphs.Clear(); wrapped=0; }
    void AddControlsDelta(int delta_y);
  };

  struct Line {
    point p;
    TextBox *parent=0;
    TextBox::Lines *cont=0;
    shared_ptr<LineData> data;
    Line() : data(make_shared<LineData>()) {}
    Line &operator=(const Line &s) { data=s.data; return *this; }
    const DrawableBox& operator[](int i) const { return data->glyphs[i]; }
    static void Move (Line &t, Line &s) { swap(t.data, s.data); }
    static void MoveP(Line &t, Line &s) { swap(t.data, s.data); t.p=s.p; }

    void Init(TextBox *P, TextBox::Lines *C) { parent=P; cont=C; data->flow=InitFlow(&data->glyphs); }
    Flow InitFlow(DrawableBoxArray *out) { return Flow(&data->box, parent->style.font, out, &parent->layout); }
    int GetAttrId(const Drawable::Attr &a) { return data->glyphs.attr.GetAttrId(a); }
    int Size () const { return data->glyphs.Size(); }
    int Lines() const { return 1+data->glyphs.line.size(); }
    String16 Text16() const { return data->glyphs.Text16(); }
    void Clear() { data->Clear(); data->flow = InitFlow(&data->glyphs); }
    int Erase(int o, int l=INT_MAX);
    int AssignText(const StringPiece   &s, int                     a=0) { Clear(); return AppendText(s, a); }
    int AssignText(const String16Piece &s, int                     a=0) { Clear(); return AppendText(s, a); }
    int AssignText(const StringPiece   &s, const DrawableAnnotation &a) { Clear(); return AppendText(s, a); }
    int AssignText(const String16Piece &s, const DrawableAnnotation &a) { Clear(); return AppendText(s, a); }
    int AppendText(const StringPiece   &s, int                     a=0) { return InsertTextAt(Size(), s, a); }
    int AppendText(const String16Piece &s, int                     a=0) { return InsertTextAt(Size(), s, a); }
    int AppendText(const StringPiece   &s, const DrawableAnnotation &a) { return InsertTextAt(Size(), s, a); }
    int AppendText(const String16Piece &s, const DrawableAnnotation &a) { return InsertTextAt(Size(), s, a); }
    template <class X> int OverwriteTextAt(int o, const StringPieceT<X> &s, int a=0);
    template <class X> int InsertTextAt   (int o, const StringPieceT<X> &s, int a=0);
    template <class X> int InsertTextAt   (int o, const StringPieceT<X> &s, const DrawableAnnotation&);
    template <class X> int InsertTextAt   (int o, const StringPieceT<X> &s, const DrawableBoxArray&);
    template <class X> int UpdateText     (int o, const StringPieceT<X> &s, int a, int max_width=0, bool *append=0, int insert_mode=-1);
    int OverwriteTextAt(int o, const string   &s, int a=0) { return OverwriteTextAt<char>    (o, s, a); }
    int OverwriteTextAt(int o, const String16 &s, int a=0) { return OverwriteTextAt<char16_t>(o, s, a); }
    int InsertTextAt(int o, const string   &s, int a=0) { return InsertTextAt<char>    (o, s, a); }
    int InsertTextAt(int o, const String16 &s, int a=0) { return InsertTextAt<char16_t>(o, s, a); }
    int UpdateText(int o, const string   &s, int attr, int max_width=0, bool *append=0) { return UpdateText<char>    (o, s, attr, max_width, append); }
    int UpdateText(int o, const String16 &s, int attr, int max_width=0, bool *append=0) { return UpdateText<char16_t>(o, s, attr, max_width, append); }
    void EncodeText(DrawableBoxArray *o, int x, const StringPiece   &s, int a=0)                               { Flow f=InitFlow(o); f.p.x=x; f.AppendText(s,a); }
    void EncodeText(DrawableBoxArray *o, int x, const String16Piece &s, int a=0)                               { Flow f=InitFlow(o); f.p.x=x; f.AppendText(s,a); }
    void EncodeText(DrawableBoxArray *o, int x, const StringPiece   &s, const DrawableAnnotation &a, int da=0) { Flow f=InitFlow(o); f.p.x=x; f.AppendText(s,a,da); }
    void EncodeText(DrawableBoxArray *o, int x, const String16Piece &s, const DrawableAnnotation &a, int da=0) { Flow f=InitFlow(o); f.p.x=x; f.AppendText(s,a,da); }
    int Layout(int width=0, bool flush=0) { Layout(Box(0,0,width,0), flush); return Lines(); }
    void Layout(Box win, bool flush=0);
    point Draw(point pos, int relayout_width=-1, int g_offset=0, int g_len=-1, const Box *scissor=0);
  };

  struct Lines : public RingVector<Line> {
    TextBox *parent;
    int wrapped_lines;
    Drawable::AttrSource *attr_source=0;
    function<void(Line&, Line&)> move_cb, movep_cb;
    Lines(TextBox *P, int N);

    void Resize(int s) override;
    void SetAttrSource(Drawable::AttrSource *s);
    Line *PushFront() override { Line *l = RingVector::PushFront(); l->Clear(); return l; }
    Line *InsertAt(int dest_line, int lines=1, int dont_move_last=0);
    static int GetBackLineLines(const Lines &l, int i) { return l[-i-1].Lines(); }
  };

  struct LinesFrameBuffer : public RingFrameBuffer<Line> {
    typedef function<LinesFrameBuffer*(const Line*)> FromLineCB;
    struct Flag { enum { NoLayout=1, NoVWrap=2, Flush=4 }; };
    PaintCB paint_cb = &LinesFrameBuffer::PaintCB;
    int lines=0;
    bool align_top_or_bot=1, partial_last_line=1, wrap=0, only_grow=0;
    LinesFrameBuffer(GraphicsDeviceHolder *d) : RingFrameBuffer(d) {}
    LinesFrameBuffer *Attach(LinesFrameBuffer **last_fb);
    virtual int SizeChanged(int W, int H, Font *font, ColorDesc bgc) override;
    tvirtual void Clear(Line *l) { RingFrameBuffer::Clear(l, Box(w, l->Lines() * font_height), true); }
    tvirtual void Update(Line *l, int flag=0);
    tvirtual void Update(Line *l, const point &p, int flag=0) { l->p=p; Update(l, flag); }
    tvirtual void OverwriteUpdate(Line *l, int xo=0, int wlo=0, int wll=0, int flag=0);
    tvirtual int PushFrontAndUpdate(Line *l, int xo=0, int wlo=0, int wll=0, int flag=0);
    tvirtual int PushBackAndUpdate (Line *l, int xo=0, int wlo=0, int wll=0, int flag=0);
    tvirtual void PushFrontAndUpdateOffset(Line *l, int lo);
    tvirtual void PushBackAndUpdateOffset (Line *l, int lo); 
    tvirtual void DrawAligned(const Box &b, point adjust);
    static point PaintCB(Line *l, point lp, const Box &b) { return Paint(l, lp, b); }
    static point Paint  (Line *l, point lp, const Box &b, int offset=0, int len=-1);
  };

  struct LineUpdate {
    enum { PushBack=1, PushFront=2, DontUpdate=4 }; 
    Line *v; LinesFrameBuffer *fb; int flag, o;
    LineUpdate(Line *V=0, LinesFrameBuffer *FB=0, int F=0, int O=0) : v(V), fb(FB), flag(F), o(O) {}
    LineUpdate(Line *V, const LinesFrameBuffer::FromLineCB &cb, int F=0, int O=0) : v(V), fb(cb(V)), flag(F), o(O) {}
    ~LineUpdate();
    Line *operator->() const { return v; }
  };

  struct Cursor {
    enum { Underline=1, Block=2 };
    int type=Block, attr=0;
    Time blink_time=Time(333), blink_begin=Time(0);
    point i, p;
  };

  struct Selection : public DragTracker {
    struct Point { 
      Box glyph;
      int line_ind=0, char_ind=0;
      string DebugString() const { return StrCat("Selection::Point(l=", line_ind, ", c=", char_ind, ", b=", glyph.DebugString(), ")"); }
    };
    Point beg, end;
    Box3 box;
    String16 text;
    float start_v_scrolled=0;
    int gui_ind=-1, scrolled=0;
    bool explicitly_initiated=0;
    void Begin(float s) { end=beg; start_v_scrolled=s; scrolled=0; text.clear(); }
  };

  StringCB runcb;
  Style style;
  Flow::Layout layout;
  Cursor cursor;
  Selection selection;
  Line cmd_line;
  LinesFrameBuffer cmd_fb;
  string cmd_prefix="> ";
  RingVector<string> cmd_last;
  vector<int> resize_gui_ind;
  Color cmd_color=Color::white, selection_color=Color(Color::grey70, 0.5);
  bool needs_redraw=0, deactivate_on_enter=0, clear_on_enter=1, token_processing=0, insert_mode=1, run_blank_cmd=0;
  int start_line=0, end_line=0, start_line_adjust=0, skip_last_lines=0, default_attr=0, cmd_last_ind=-1;
  function<void(const Selection::Point&)> selection_cb;
  function<void(const shared_ptr<Control>&)> new_link_cb;
  function<void(Control*)> hover_control_cb;
  Control *hover_control=0;
  const Border *clip=0;
  ColorDesc bg_color=0;

  TextBox(Window *W, const FontRef &F=FontRef(), int LC=10);
  virtual ~TextBox() { if (root) Deactivate(); }

  virtual point RelativePosition(const point&) const override;
  virtual void Run(const string &cmd) { if (runcb) runcb(cmd); }
  virtual int CommandLines() const { return 0; }
  virtual bool Active() const { return root->active_textbox == this; }
  virtual bool NotActive(const point &p) const override { return !box.RelativeCoordinatesBox().within(p) && mouse.drag.empty(); }
  virtual bool Activate() override { if ( Active()) return 0; if (auto g = dynamic_cast<View*>(root->active_textbox)) g->Deactivate(); root->active_textbox = this; return 1; }
  virtual bool Deactivate() override { if (!Active()) return 0; root->active_textbox = root->default_textbox(); return 1; }
  virtual bool ToggleActive() override { if (!Active()) Activate(); else Deactivate(); return Active(); }
  virtual void Input(char k) override { cmd_line.UpdateText(cursor.i.x++, String16(1, *MakeUnsigned<char>(&k)), cursor.attr); UpdateCommandFB(); UpdateCursor(); }
  virtual void Erase()       override { if (!cursor.i.x) return; cmd_line.Erase(--cursor.i.x, 1); UpdateCommandFB(); UpdateCursor(); }
  virtual void CursorRight() override { UpdateCursorX(min(cursor.i.x+1, cmd_line.Size())); }
  virtual void CursorLeft()  override { UpdateCursorX(max(cursor.i.x-1, 0)); }
  virtual void Home()        override { UpdateCursorX(0); }
  virtual void End()         override { UpdateCursorX(cmd_line.Size()); }
  virtual void HistUp()      override { if (int c=cmd_last.ring.count) { AssignInput(cmd_last[cmd_last_ind]); cmd_last_ind=max(cmd_last_ind-1, -c); } }
  virtual void HistDown()    override { if (int c=cmd_last.ring.count) { AssignInput(cmd_last[cmd_last_ind]); cmd_last_ind=min(cmd_last_ind+1, -1); } }
  virtual void Enter()       override;
  virtual void Tab()         override {}

  virtual String16 Text16() const { return cmd_line.Text16(); }
  virtual void AssignInput(const string &text) { cmd_line.AssignText(text); UpdateCommandFB(); UpdateCursorX(cmd_line.Size()); }
  void SetColors(Colors *C);

  virtual const LinesFrameBuffer *GetFrameBuffer() const { return &cmd_fb; }
  virtual       LinesFrameBuffer *GetFrameBuffer()       { return &cmd_fb; }
  virtual void ResetGL(int flag) override { cmd_fb.ResetGL(flag); needs_redraw=true; }
  virtual void UpdateCursorX(int x) { cursor.i.x = x; UpdateCursor(); }
  virtual void UpdateCursor() { cursor.p = cmd_line.data->glyphs.Position(cursor.i.x) + point(0, style.font->Height()); }
  virtual void UpdateCommandFB() { UpdateLineFB(&cmd_line, &cmd_fb); }
  virtual void UpdateLineFB(Line *L, LinesFrameBuffer *fb, int flag=0);
  virtual void Draw(GraphicsContext*, const LFL::Box &b) const override { const_cast<TextBox*>(this)->Draw(b); }
  virtual void Draw(const Box &b);
  virtual void DrawCursor(point p, Shader *shader=0);
  virtual void UpdateToken(Line*, int word_offset, int word_len, int update_type, const TokenProcessor<DrawableBox>*);
  virtual void UpdateLongToken(Line *BL, int beg_offset, Line *EL, int end_offset, const string &text, int update_type);
  virtual shared_ptr<Control> AddUrlBox(Line *BL, int beg_offset, Line *EL, int end_offset, string v, Callback cb);

  void AddHistory(const string &cmd);
  int  ReadHistory(FileSystem*, const string &dir, const string &name);
  int  WriteHistory(const string &dir, const string &name, const string &hdr);
};

struct UnbackedTextBox : public TextBox {
  UnbackedTextBox(Window *W, const FontRef &F=FontRef()) : TextBox(W, F) {}
  virtual void UpdateCommandFB() override {}
};

struct TiledTextBox : public TextBox {
  point offset;
  TilesInterface *tiles=0;
  TiledTextBox(Window *W, const FontRef &F=FontRef()) : TextBox(W, F) { cmd_fb.paint_cb = bind(&TiledTextBox::PaintCB, this, _1, _2, _3); }
  void AssignTarget(TilesInterface *T, const point &p) { tiles=T; offset=p; }
  point PaintCB(Line *l, point lp, const Box &b);
};

struct TextArea : public TextBox {
  Lines line;
  LinesFrameBuffer line_fb;
  Time write_last=Time(0);
  bool wrap_lines=1, write_timestamp=0, write_newline=1, reverse_line_fb=0, cursor_enabled=1;
  int line_left=0, end_line_adjust=0, start_line_cutoff=0, end_line_cutoff=0;
  int extra_height=0, scroll_inc=10, scrolled_lines=0;
  float v_scrolled=0, h_scrolled=0, last_v_scrolled=0, last_h_scrolled=0;
  function<bool(int, point, point, int)> drag_cb;

  TextArea(Window *W, const FontRef &F, int S, int LC);
  virtual ~TextArea() {}

  /// Write() is thread-safe.
  virtual void Write(const StringPiece &s, bool update_fb=true, bool release_fb=true);
  virtual void WriteCB(const string &s, bool update_fb, bool release_fb) { return Write(s, update_fb, release_fb); }
  virtual void PageUp  () override { AddVScroll(-scroll_inc); }
  virtual void PageDown() override { AddVScroll( scroll_inc); }
  virtual void ScrollUp  () { PageUp(); }
  virtual void ScrollDown() { PageDown(); }
  virtual void SetDimension(int w, int h);
  virtual void Resized(const Box &b, bool font_size_changed=false);
  virtual void CheckResized(const Box &b);

  virtual void Redraw(bool attach=true, bool relayout=false);
  virtual void UpdateScrolled();
  virtual void UpdateHScrolled(int x, bool update_fb=true);
  virtual void UpdateVScrolled(int dist, bool reverse, int first_ind, int first_offset, int first_len);
  virtual int UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len);
  virtual int WrappedLines() const { return line.wrapped_lines; }
  virtual const LinesFrameBuffer *GetFrameBuffer() const override { return &line_fb; }
  virtual       LinesFrameBuffer *GetFrameBuffer()       override { return &line_fb; }
  virtual void ResetGL(int flag) override { line_fb.ResetGL(flag); TextBox::ResetGL(flag); }
  void ChangeColors(Colors *C);

  struct DrawFlag { enum { DrawCursor=1, CheckResized=2, Default=DrawCursor|CheckResized }; };
  virtual void Draw(const Box &w, int flag=DrawFlag::Default, Shader *shader=0);
  virtual void DrawHoverLink(const Box &w);
  virtual bool GetGlyphFromCoords(const point &p, Selection::Point *out) { return GetGlyphFromCoordsOffset(p, out, start_line, start_line_adjust); }
  bool GetGlyphFromCoordsOffset(const point &p, Selection::Point *out, int sl, int sla);

  bool Wrap() const { return line_fb.wrap; }
  int LineFBPushBack () const { return reverse_line_fb ? LineUpdate::PushFront : LineUpdate::PushBack;  }
  int LineFBPushFront() const { return reverse_line_fb ? LineUpdate::PushBack  : LineUpdate::PushFront; }
  float PercentOfLines(int n) const { return float(n) / (WrappedLines()-1); }
  void AddVScroll(int n) { v_scrolled = Clamp(v_scrolled + PercentOfLines(n), 0.0f, 1.0f); UpdateScrolled(); }
  void SetVScroll(int n) { v_scrolled = Clamp(0          + PercentOfLines(n), 0.0f, 1.0f); UpdateScrolled(); }
  int LayoutBackLine(Lines *l, int i) { return Wrap() ? (*l)[-i-1].Layout(line_fb.w) : 1; }
  int AddWrappedLines(int wl, int n) { last_v_scrolled = (v_scrolled *= float(wl)/(wl+n)); return wl+n; }

  void InitSelection();
  void DrawSelection();
  void DragCB(int button, point p, point d, int down);
  void CopyText(const Selection::Point &beg, const Selection::Point &end);
  string CopyText(int beg_line_ind, int beg_char_ind, int end_line_end, int end_char_ind, bool add_nl);
  void InitContextMenu(const MouseController::CB &cb)    { resize_gui_ind.push_back(mouse.AddRightClickBox(box, cb)); }
  void InitWheelMenu(const MouseController::CoordCB &cb) { resize_gui_ind.push_back(mouse.AddWheelBox(box, cb)); }
};

struct TextView : public TextArea {
  int wrapped_lines=0, fb_wrapped_lines=0;
  int last_fb_width=0, last_fb_lines=0, last_first_line=0, last_update_mapping_flag=0;
  TextView(Window *W, const FontRef &F=FontRef()) : TextArea(W, F, 0, 0) { reverse_line_fb=1; }

  virtual int WrappedLines() const override { return wrapped_lines; }
  virtual int UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len) override;
  virtual int RefreshLines() { last_fb_lines=0; return UpdateLines(last_v_scrolled, 0, 0, 0); }
  virtual void Reload() { last_fb_width=0; wrapped_lines=0; RefreshLines(); }

  virtual bool Empty() const { return true; }
  virtual void UpdateMapping(int width, int flag=0) {}
  virtual int UpdateMappedLines(pair<int, int>, bool, bool, bool, bool, bool) { return 0; }
};

struct PropertyView : public TextView {
  typedef size_t Id;
  typedef vector<Id> Children;
  typedef function<void(PropertyView*, Id)> PropertyCB; 
  struct Node {
    typedef function<void(Id, Node*, int)> Visitor;
    Drawable *icon;
    string text, val;
    Children child;
    bool control=1, expanded=0;
    int depth=0;
    Node(Drawable *I, const string &T, const string &V, bool C=1) : icon(I), text(T), val(V), control(C) {}
    Node(Drawable *I=0, const string &T=string(), const Children &C=Children()) : icon(I), text(T), child(C) {}
  };
  struct NodeIndex {
    Id id;
    NodeIndex(Id I=0) : id(I) {}
    static string GetString(const NodeIndex *v) { return ""; }
    static int    GetLines (const NodeIndex *v) { return 1; }
  };
  typedef PrefixSumKeyedRedBlackTree<int, NodeIndex> LineMap;

  Id root_id=0;
  LineMap property_line;
  FontRef menuicon_white, menuicon_black;
  int selected_line_no=-1;
  Color selected_color=Color(Color::blue, 0.2);
  PropertyCB line_selected_cb, selected_line_clicked_cb;
  PropertyView(Window *W, const FontRef &F=FontRef());

  virtual       Node* GetNode(Id)       = 0;
  virtual const Node* GetNode(Id) const = 0;

  virtual bool Empty() const override { return !property_line.size(); }
  virtual void Clear() { property_line.Clear(); root_id=0; selected_line_no=-1; }
  virtual bool Deactivate() override { bool ret = TextBox::Deactivate(); selected_line_no=-1; return ret; }
  virtual void SetRoot(Id id) { GetNode((root_id = id))->expanded = true; }
  virtual void Draw(const Box &w, int flag=DrawFlag::Default, Shader *shader=0) override;
  virtual void VisitExpandedChildren(Id id, const Node::Visitor &cb, int depth=0);
  virtual void HandleCollapsed(Id id) {}
  virtual void Input(char k) override {}
  virtual void Erase()       override {}
  virtual void CursorRight() override {}
  virtual void CursorLeft()  override {}
  virtual void Home()        override {}
  virtual void End()         override {}
  virtual void HistUp()      override {}
  virtual void HistDown()    override {}
  virtual void Enter()       override {}

  void UpdateMapping(int width, int flag=0) override;
  int UpdateMappedLines(pair<int, int>, bool, bool, bool, bool, bool) override;
  void LayoutLine(Line *L, const NodeIndex &n, const point &p);
  void HandleNodeControlClicked(Id id, int b, point p, point d, int down);
  void SelectionCB(const Selection::Point &p);
};

struct PropertyTree : public PropertyView {
  FreeListVector<Node> tree;
  using PropertyView::PropertyView;

  /**/  Node* GetNode(Id id)       override { return &tree[id-1]; }
  const Node* GetNode(Id id) const override { return &tree[id-1]; }
  void Clear() override { PropertyView::Clear(); tree.Clear(); }
  template <class... Args> Id AddNode(Args&&... args) { return 1+tree.Insert(Node(forward<Args>(args)...)); }
};

struct DirectoryTree : public PropertyTree {
  FileSystem *fs;
  DirectoryTree(Window *W, const FontRef &F=FontRef()) : PropertyTree(W, F), fs(&W->parent->localfs) {}
  DirectoryTree(Window *W, FileSystem *FS, const FontRef &F=FontRef()) : PropertyTree(W, F), fs(FS) {}
  void Open(const string &p) { tree.Clear(); SetRoot(AddDir(p)); Reload(); }
  Id AddDir (const string &p) { return AddNode(menuicon_white->FindGlyph(13), BaseName(StringPiece(p.data(), p.size()?p.size()-1:0)), p); }
  Id AddFile(const string &p) { return AddNode(menuicon_white->FindGlyph(14), BaseName(p), p, 0); }
  virtual void VisitExpandedChildren(Id id, const Node::Visitor &cb, int depth=0) override;
  virtual void HandleCollapsed(Id id) override { tree.Erase(id-1); }
};

struct Console : public TextArea {
  Color color=Color(25,60,130,120);
  Callback animating_cb;
  Time anim_time=Time(333), anim_begin=Time(0);
  bool animating=0, drawing=0, bottom_or_top=0, blend=1;
  int full_height;
  Box *scissor=0, scissor_buf;

  Console(Window *W, const FontRef &F, const Callback &C=Callback());
  Console(Window *W, const Callback &C=Callback()) :
    Console(W, FontRef(W, FontDesc(A_or_B(FLAGS_console_font, FLAGS_font), "", 9, Color::white,
                                   Color::clear, FLAGS_console_font_flag)), C) {}

  virtual ~Console() {}
  virtual int CommandLines() const override { return cmd_line.Lines(); }
  virtual void Run(const string &in) override;
  virtual void PageUp  () override { TextArea::PageDown(); }
  virtual void PageDown() override { TextArea::PageUp(); }
  virtual bool Activate()   override { bool ret = TextBox::Activate();   StartAnimating(); return ret; }
  virtual bool Deactivate() override { bool ret = TextBox::Deactivate(); StartAnimating(); return ret; }
  virtual void Draw(const Box &b, int flag=DrawFlag::Default, Shader *shader=0) override;
  virtual void Draw(const point &p) override;
  void StartAnimating();
};

struct Dialog : public View {
  struct Flag { enum { None=0, Fullscreen=1, Next=2 }; };
  static const int min_width = 50, min_height = 25;
  Color color, title_gradient[4];
  FontRef font, menuicon;
  Box title, content, resize_left, resize_right, resize_bottom, close;
  bool deleted=0, moving=0, resizing_left=0, resizing_right=0, resizing_bottom=0, fullscreen=0, tabbed=0;
  point mouse_start, win_start;
  Callback deleted_cb;
  string title_text;
  int zsort=0;

  Dialog(Window*, const char *n, float w, float h, int flag=0);
  virtual ~Dialog() {}
  virtual void TakeFocus() {}
  virtual void LoseFocus() {}
  virtual View *Layout(Flow *flow=nullptr) override;
  virtual void Draw(const point &p) override;

  void LayoutTabbed(int, const Box &b, const point &d, MouseController*, DrawableBoxArray*);
  void LayoutTitle(const Box &b, MouseController*, DrawableBoxArray*);
  void LayoutReshapeControls(const point &d, MouseController*);
  bool HandleReshape(Box *outline);
  void DrawGradient(const point &p) const { GraphicsContext::DrawGradientBox1(root->gd, (title + p), title_gradient); }
  void Reshape(bool *down) { mouse_start = root->mouse; win_start = point(box.x, box.y); *down = 1; }

  static bool LessThan(const unique_ptr<Dialog> &l, const unique_ptr<Dialog> &r) { return l->zsort < r->zsort; }
  static void MessageBox(Window*, const string &text);
  static void TextureBox(Window*, const string &text);
};

struct DialogTab {
  Dialog *dialog;
  DrawableBoxArray child_box;
  DialogTab(Dialog *D=0) : dialog(D) {}
  bool operator<(const DialogTab &x) const { return dialog < x.dialog; }
  bool operator==(const DialogTab &x) const { return dialog == x.dialog; }
  static void Draw(GraphicsDevice*, const Box &b, const point &tab_dim, const vector<DialogTab>&);
};

struct TabbedDialogInterface {
  View *view;
  Box box;
  point tab_dim;
  vector<DialogTab> tab_list;
  TabbedDialogInterface(View *V, const point &d=point(200,16));
  virtual void SelectTabIndex(size_t i) {}
  virtual View *Layout(Flow *flow=nullptr);
  virtual void Draw(const point &p) { DialogTab::Draw(view->root->gd, box, tab_dim, tab_list); }
};

template <class D=Dialog> struct TabbedDialog : public TabbedDialogInterface {
  D *top=0;
  unordered_set<D*> tabs;
  using TabbedDialogInterface::TabbedDialogInterface;
  D *FirstTab() const { return tab_list.size() ? dynamic_cast<D*>(tab_list.begin()->dialog) : 0; }
  int TabIndex(D *t) const { for (auto b=tab_list.begin(), e=tab_list.end(), i=b; i!=e; ++i) if (*i == t) return i-b; return -1; }
  void AddTab(D *t) { tabs.insert(t); tab_list.emplace_back(t); SelectTab(t); }
  void DelTab(D *t) { tabs.erase(t); VectorEraseByValue(&tab_list, DialogTab(t)); if (top == t) SelectTab(FirstTab()); }
  void SelectTab(D *t) { if (top) top->LoseFocus(); if ((top = t)) { view->child_view={t}; t->TakeFocus(); } }
  void SelectTabIndex(size_t i) override { CHECK_LT(i, tab_list.size()); SelectTab(dynamic_cast<D*>(tab_list[i].dialog)); }
  void SelectNextTab() { if (top) SelectTabIndex(RingIndex::Wrap(TabIndex(top)+1, tab_list.size())); }
  void SelectPrevTab() { if (top) SelectTabIndex(RingIndex::Wrap(TabIndex(top)-1, tab_list.size())); }
  void Draw(const point &p) override { TabbedDialogInterface::Draw(p); if (top) top->Draw(p); }
};

struct MessageBoxDialog : public Dialog {
  string message;
  Box message_size;
  MessageBoxDialog(Window *w, const string &m) :
    Dialog(w, "MessageBoxDialog", .25, .2), message(m) { font->Size(message, &message_size); }
  View *Layout(Flow *flow=nullptr) override;
  void Draw(const point &p) override;
};

struct TextureBoxDialog : public Dialog {
  Texture tex;
  TextureBoxDialog(Window *w, const string &m) :
    Dialog(w, "TextureBoxDialog", .33, .33), tex(w->parent) { tex.ID = ::atoi(m.c_str()); tex.owner = false; }
  void Draw(const point &p) override { Dialog::Draw(p); tex.DrawGD(root->gd, content + box.TopLeft()); }
};

struct SliderDialog : public Dialog {
  typedef function<void(Widget::Slider*)> UpdatedCB;
  string title;
  UpdatedCB updated;
  Widget::Slider slider;
  SliderDialog(Window *w, const string &title="", const UpdatedCB &cb=UpdatedCB(),
               float scrolled=0, float total=100, float inc=1);
  View *Layout(Flow *flow=nullptr) override { Dialog::Layout(flow); slider.LayoutFixed(content); return this; }
  void Draw(const point &p) override { Dialog::Draw(p); if (updated && slider.HasChanged()) updated(&slider); }
};

struct FlagSliderDialog : public SliderDialog {
  string flag_name;
  FlagMap *flag_map;
  FlagSliderDialog(Window *w, const string &fn, float total=100, float inc=1);
  virtual void Updated(Widget::Slider *s) { flag_map->Set(flag_name, StrCat(s->Percent())); }
};

template <class X> struct TextViewDialogT  : public Dialog {
  X view;
  Widget::Slider v_scrollbar, h_scrollbar;
  TextViewDialogT(Window *W, const char *n, const FontRef &F, float w=0.5, float h=.5, int flag=0) :
    Dialog(W, n, w, h, flag), view(W, F), v_scrollbar(this, Widget::Slider::Flag::AttachedNoCorner),
    h_scrollbar(this, Widget::Slider::Flag::AttachedHorizontalNoCorner) {}
  View *Layout(Flow *flow=nullptr) override {
    Dialog::Layout(flow);
    Widget::Slider::AttachContentBox(&content, &v_scrollbar, view.Wrap() ? nullptr : &h_scrollbar);
    child_view.push_back(&view);
    return this;
  }
  void Draw(const point &p) override { 
    bool wrap = view.Wrap();
    if (1)     view.v_scrolled = v_scrollbar.AddScrollDelta(view.v_scrolled);
    if (!wrap) view.h_scrolled = h_scrollbar.AddScrollDelta(view.h_scrolled);
    if (1)     view.UpdateScrolled();
    if (1)     Dialog::Draw(p);
    if (1)     view.Draw(content + box.TopLeft(), TextArea::DrawFlag::CheckResized |
                         (view.cursor_enabled ? TextArea::DrawFlag::DrawCursor : 0));
    if (1)     v_scrollbar.UpdateDotPosition();
    if (!wrap) h_scrollbar.UpdateDotPosition();
  }
  void TakeFocus() override { view.Activate(); }
  void LoseFocus() override { view.Deactivate(); }
};

struct PropertyTreeDialog : public TextViewDialogT<PropertyTree> {
  PropertyTreeDialog(Window *W, const FontRef &F, float w=0.5, float h=.5, int flag=0) :
    TextViewDialogT(W, "PropertyTreeDialog", F, w, h, flag) {}
};

struct DirectoryTreeDialog : public TextViewDialogT<DirectoryTree> {
  DirectoryTreeDialog(Window *W, const FontRef &F, float w=0.5, float h=.5, int flag=0) :
    TextViewDialogT(W, "DirectoryTreeDialog", F, w, h, flag) {}
};

struct HelperView : public View {
  FontRef font;
  HelperView(Window *W) : View(W, "HelperView"), font(W, FontDesc(FLAGS_font, "", 9, Color::white)) {}
  struct Hint { enum { UP, UPLEFT, UPRIGHT, DOWN, DOWNLEFT, DOWNRIGHT }; };
  struct Label {
    Box target, label;
    v2 target_center, label_center;
    int hint; string description;
    Label(const Box &w, const string &d, int h, Font *f, const point &p);
    void AssignLabelBox() { label.x = label_center.x - label.w/2; label.y = label_center.y - label.h/2; }
  };
  vector<Label> label;
  void AddLabel(const Box &w, const string &d, int h, const point &p) { label.emplace_back(w, d, h, font, p); }
  bool Activate() override { if (active) return 0; active=1; /* ForceDirectedLayout(); */ return 1; }
  void ForceDirectedLayout();
  void Draw(const point &p) override;
};

struct BindMap : public View, public MouseControllerInterface, public KeyboardControllerInterface {
  unordered_set<Bind> data, down;
  function<void(point, point)> move_cb;
  BindMap(Window *w) : View(w, "BindMap") { Activate(); }

  template <class... Args> void Add(Args&&... args) { AddBind(Bind(forward<Args>(args)...)); }
  void AddBind(const Bind &b) { data.insert(b); }
  void Repeat(unsigned clicks) { for (auto b : down) b.Run(clicks); }
  void Button(InputEvent::Id event, bool d);
  string DebugString() const override { string v="{ "; for (auto b : data) StrAppend(&v, b.key, " "); return v + "}"; }

  MouseControllerInterface    *GetMouseController   () override { return this; }
  KeyboardControllerInterface *GetKeyboardController() override { return this; }

  int SendKeyEvent(InputEvent::Id, bool down)                                            override;
  int SendMouseEvent(InputEvent::Id, const point &p, const point &d, int down, int flag) override;
  int SendWheelEvent(InputEvent::Id, const v2    &p, const v2    &d, bool begin)         override;
};

}; // namespace LFL
#endif // LFL_CORE_APP_GL_VIEW_H__
