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

#ifndef LFL_LFAPP_GUI_H__
#define LFL_LFAPP_GUI_H__
namespace LFL {

DECLARE_bool(multitouch);
DECLARE_bool(draw_grid);
DECLARE_bool(console);
DECLARE_string(console_font);
DECLARE_int(console_font_flag);

struct GUI {
  Box box;
  DrawableBoxArray child_box;
  MouseController mouse;
  bool active=0;
  GUI(const Box &B=Box()) : box(B) {}
  virtual ~GUI() {}

  DrawableBoxArray *Reset() { Clear(); return &child_box; }
  void Clear() { child_box.Clear(); mouse.Clear(); }
  void UpdateBox(const Box &b, int draw_box_ind, int input_box_ind);
  void UpdateBoxX(int x, int draw_box_ind, int input_box_ind);
  void UpdateBoxY(int y, int draw_box_ind, int input_box_ind);
  void IncrementBoxY(int y, int draw_box_ind, int input_box_ind);

  virtual void Activate() { active = 1; }
  virtual void Deactivate() { active = 0; }
  virtual bool NotActive() const { return !active; }
  virtual bool ToggleActive() { if ((active = !active)) Activate(); else Deactivate(); return active; }
  virtual point RelativePosition(const point &p) const { return p - box.TopLeft(); }
  virtual void SetLayoutDirty() { child_box.Clear(); }
  virtual void Layout(const Box &b) { box=b; Layout(); }
  virtual void Layout() {}
  virtual void Draw();
  virtual void ResetGL() {}
  virtual void HandleTextMessage(const string &s) {}
};

struct Widget {
  struct Interface {
    GUI *gui;
    vector<int> hitbox;
    int drawbox_ind=-1;
    bool del_hitbox=0;
    virtual ~Interface() { if (del_hitbox) DelHitBox(); }
    Interface(GUI *g) : gui(g) {}

    void AddClickBox(const Box  &b, const MouseControllerCallback &cb) {                   hitbox.push_back(gui->mouse.AddClickBox(b, cb)); }
    void AddHoverBox(const Box  &b, const MouseControllerCallback &cb) {                   hitbox.push_back(gui->mouse.AddHoverBox(b, cb)); }
    void AddDragBox (const Box  &b, const MouseControllerCallback &cb) {                   hitbox.push_back(gui->mouse.AddDragBox (b, cb)); }
    void AddClickBox(const Box3 &t, const MouseControllerCallback &cb) { for (auto &b : t) hitbox.push_back(gui->mouse.AddClickBox(b, cb)); }
    void AddHoverBox(const Box3 &t, const MouseControllerCallback &cb) { for (auto &b : t) hitbox.push_back(gui->mouse.AddHoverBox(b, cb)); }
    void AddDragBox (const Box3 &t, const MouseControllerCallback &cb) { for (auto &b : t) hitbox.push_back(gui->mouse.AddDragBox (b, cb)); }
    void DelHitBox() { for (auto i = hitbox.begin(), e = hitbox.end(); i != e; ++i) gui->mouse.hit.Erase(*i); hitbox.clear(); }
    MouseController::HitBox &GetHitBox(int i=0) const { return gui->mouse.hit[hitbox[i]]; }
    Box GetHitBoxBox(int i=0) const { return Box::Add(GetHitBox(i).box, gui->box.TopLeft()); }
    DrawableBox *GetDrawBox() const { return drawbox_ind >= 0 ? VectorGet(gui->child_box.data, drawbox_ind) : 0; }
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
    int decay=0;
    Button() : Interface(0) {}
    Button(GUI *G, Drawable *I, const string &T, const MouseControllerCallback &CB)
      : Interface(G), text(T), image(I), cb(CB), init(1) {}

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
    Box win;
    int flag=0, doc_height=200, dot_size=15;
    float scrolled=0, last_scrolled=0, increment=20;
    Color color=Color(15, 15, 15, 55);
    Font *menuicon=0;
    bool dragging=0, dirty=0;
    virtual ~Slider() {}
    Slider(GUI *Gui, Box window=Box(), int f=Flag::Attached);

    float Percent() const { return scrolled * doc_height; }
    void LayoutFixed(const Box &w) { win = w; Layout(dot_size, dot_size, flag & Flag::Horizontal); }
    void LayoutAttached(const Box &w);
    void Layout(int aw, int ah, bool flip);
    void Update(bool force=false);
    void SetDocHeight(int v) { doc_height = v; }
    void DragScrollDot() { dragging = true; dirty = true; }
    void ScrollUp  () { scrolled -= increment / doc_height; Clamp(&scrolled, 0, 1); dirty=true; }
    void ScrollDown() { scrolled += increment / doc_height; Clamp(&scrolled, 0, 1); dirty=true; }
    float ScrollDelta() { float ret=scrolled-last_scrolled; last_scrolled=scrolled; return ret; }
    float AddScrollDelta(float cur_val);
    static void AttachContentBox(Box *b, Slider *vs, Slider *hs);
  };
};

struct TextBox : public GUI, public KeyboardController, public Drawable::AttrSource {
  struct Attr {
    enum { Bold=1<<16, Underline=1<<17, Blink=1<<18, Reverse=1<<19, Italic=1<<20, Link=1<<21 };
    static void SetFGColorIndex(int *a, int c) { *a = (*a & ~0x00ff) | ((c & 0xff)     ); }
    static void SetBGColorIndex(int *a, int c) { *a = (*a & ~0xff00) | ((c & 0xff) << 8); }
    static int GetFGColorIndex(int a) { int c = a & 0xff; return c | (((a & Bold) && c<16) ? (1<<3) : 0); }
    static int GetBGColorIndex(int a) { return (a>>8) & 0xff; }
  };

  struct Colors {
    static const int normal_index=16, bold_index=17, bg_index=18;
    Color c[16 + 3];
  };
  struct StandardVGAColors    : public Colors { StandardVGAColors(); };
  struct SolarizedDarkColors  : public Colors { SolarizedDarkColors(); };
  struct SolarizedLightColors : public Colors { SolarizedLightColors(); };

  struct Line;
  struct Lines;
  typedef function<void(const string&)> RunCB;

  struct Link : public Widget::Interface {
    Box3 box;
    string link;
    Line *line=0;
    shared_ptr<Texture> image;
    Link(Line *P, GUI *G, const Box3 &b, const string &U);
    virtual ~Link() { if (line->parent->hover_link == this) line->parent->hover_link = 0; }
    void Hover(int, int, int, int down) { line->parent->hover_link = down ? this : 0; }
    void Visit() { app->OpenSystemBrowser(link); }
  };

  struct LineData {
    Box box;
    Flow flow;
    DrawableBoxArray glyphs;
    bool outside_scroll_region=0;
    unordered_map<int, shared_ptr<Link> > links;
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
    Flow InitFlow(DrawableBoxArray *out) { return Flow(&data->box, parent->font, out, &parent->layout); }
    int GetAttrId(const Drawable::Attr &a) { return data->glyphs.attr.GetAttrId(a); }
    int Size () const { return data->glyphs.Size(); }
    int Lines() const { return 1+data->glyphs.line.size(); }
    String16 Text16() const { return data->glyphs.Text16(); }
    void Clear() { data->links.clear(); data->glyphs.Clear(); data->flow=InitFlow(&data->glyphs); }
    int Erase(int o, int l=INT_MAX);
    int AssignText(const StringPiece   &s, int                       a=0) { Clear(); return AppendText(s, a); }
    int AssignText(const String16Piece &s, int                       a=0) { Clear(); return AppendText(s, a); }
    int AssignText(const StringPiece   &s, const Flow::TextAnnotation &a) { Clear(); return AppendText(s, a); }
    int AssignText(const String16Piece &s, const Flow::TextAnnotation &a) { Clear(); return AppendText(s, a); }
    int AppendText(const StringPiece   &s, int                       a=0) { return InsertTextAt(Size(), s, a); }
    int AppendText(const String16Piece &s, int                       a=0) { return InsertTextAt(Size(), s, a); }
    int AppendText(const StringPiece   &s, const Flow::TextAnnotation &a) { return InsertTextAt(Size(), s, a); }
    int AppendText(const String16Piece &s, const Flow::TextAnnotation &a) { return InsertTextAt(Size(), s, a); }
    template <class X> int OverwriteTextAt(int o, const StringPieceT<X> &s, int a=0);
    template <class X> int InsertTextAt   (int o, const StringPieceT<X> &s, int a=0);
    template <class X> int InsertTextAt   (int o, const StringPieceT<X> &s, const Flow::TextAnnotation&);
    template <class X> int InsertTextAt   (int o, const StringPieceT<X> &s, const DrawableBoxArray&);
    template <class X> int UpdateText     (int o, const StringPieceT<X> &s, int a, int max_width=0, bool *append=0, int insert_mode=-1);
    int OverwriteTextAt(int o, const string   &s, int a=0) { return OverwriteTextAt<char>    (o, s, a); }
    int OverwriteTextAt(int o, const String16 &s, int a=0) { return OverwriteTextAt<char16_t>(o, s, a); }
    int InsertTextAt(int o, const string   &s, int a=0) { return InsertTextAt<char>    (o, s, a); }
    int InsertTextAt(int o, const String16 &s, int a=0) { return InsertTextAt<char16_t>(o, s, a); }
    int UpdateText(int o, const string   &s, int attr, int max_width=0, bool *append=0) { return UpdateText<char>    (o, s, attr, max_width, append); }
    int UpdateText(int o, const String16 &s, int attr, int max_width=0, bool *append=0) { return UpdateText<char16_t>(o, s, attr, max_width, append); }
    void EncodeText(DrawableBoxArray *o, int x, const StringPiece   &s, int a=0)                                 { Flow f=InitFlow(o); f.p.x=x; f.AppendText(s,a); }
    void EncodeText(DrawableBoxArray *o, int x, const String16Piece &s, int a=0)                                 { Flow f=InitFlow(o); f.p.x=x; f.AppendText(s,a); }
    void EncodeText(DrawableBoxArray *o, int x, const StringPiece   &s, const Flow::TextAnnotation &a, int da=0) { Flow f=InitFlow(o); f.p.x=x; f.AppendText(s,a,da); }
    void EncodeText(DrawableBoxArray *o, int x, const String16Piece &s, const Flow::TextAnnotation &a, int da=0) { Flow f=InitFlow(o); f.p.x=x; f.AppendText(s,a,da); }
    int Layout(int width=0, bool flush=0) { Layout(Box(0,0,width,0), flush); return Lines(); }
    void Layout(Box win, bool flush=0);
    point Draw(point pos, int relayout_width=-1, int g_offset=0, int g_len=-1);
  };

  struct Lines : public RingVector<Line> {
    TextBox *parent;
    int wrapped_lines;
    function<void(Line&, Line&)> move_cb, movep_cb;
    Lines(TextBox *P, int N);

    void SetAttrSource(Drawable::AttrSource *s) { for (int i=0; i<ring.size; i++) (*this)[i].data->glyphs.attr.source = s; }
    Line *PushFront() { Line *l = RingVector<Line>::PushFront(); l->Clear(); return l; }
    Line *InsertAt(int dest_line, int lines=1, int dont_move_last=0);
    static int GetBackLineLines(const Lines &l, int i) { return l[-i-1].Lines(); }
  };

  struct LinesFrameBuffer : public RingFrameBuffer<Line> {
    typedef function<LinesFrameBuffer*(const Line*)> FromLineCB;
    struct Flag { enum { NoLayout=1, NoVWrap=2, Flush=4 }; };
    PaintCB paint_cb = &LinesFrameBuffer::PaintCB;
    int lines=0;
    LinesFrameBuffer(GraphicsDevice *d) : RingFrameBuffer(d) {}
    LinesFrameBuffer *Attach(LinesFrameBuffer **last_fb);
    virtual bool SizeChanged(int W, int H, Font *font, const Color *bgc);
    virtual int Height() const { return lines * font_height; }
    tvirtual void Clear(Line *l) { RingFrameBuffer::Clear(l, Box(w, l->Lines() * font_height), true); }
    tvirtual void Update(Line *l, int flag=0);
    tvirtual void Update(Line *l, const point &p, int flag=0) { l->p=p; Update(l, flag); }
    tvirtual void OverwriteUpdate(Line *l, int xo=0, int wlo=0, int wll=0, int flag=0);
    tvirtual int PushFrontAndUpdate(Line *l, int xo=0, int wlo=0, int wll=0, int flag=0);
    tvirtual int PushBackAndUpdate (Line *l, int xo=0, int wlo=0, int wll=0, int flag=0);
    tvirtual void PushFrontAndUpdateOffset(Line *l, int lo);
    tvirtual void PushBackAndUpdateOffset (Line *l, int lo); 
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
    bool enabled=1;
    int gui_ind=-1;
    struct Point { 
      int line_ind=0, char_ind=0; Box glyph;
      bool operator<(const Point &c) const { SortImpl2(c.glyph.y, glyph.y, glyph.x, c.glyph.x); }
      string DebugString() const { return StrCat("Selection::Point(l=", line_ind, ", c=", char_ind, ", b=", glyph.DebugString(), ")"); }
    } beg, end;
    Box3 box;
  };

  RunCB runcb;
  FontRef font;
  Flow::Layout layout;
  Cursor cursor;
  Selection selection;
  Line cmd_line;
  LinesFrameBuffer cmd_fb;
  string cmd_prefix="> ";
  RingVector<string> cmd_last;
  Color cmd_color=Color::white, selection_color=Color(Color::grey70, 0.5);
  bool deactivate_on_enter=0, token_processing=0, insert_mode=1, run_blank_cmd=0;
  int start_line=0, end_line=0, start_line_adjust=0, skip_last_lines=0, default_attr=0, cmd_last_ind=-1;
  function<void(const Selection::Point&)> selection_cb;
  function<void(const shared_ptr<Link>&)> new_link_cb;
  function<void(Link*)> hover_link_cb;
  Link *hover_link=0;
  const Border *clip=0;
  const Colors *colors=0;
  const Color *bg_color=0;
  mutable Drawable::Attr last_attr;

  TextBox(GraphicsDevice *d, const FontRef &F=FontRef(), int LC=10) : font(F), cmd_last(LC), cmd_fb(d)
  { if (font.Load()) cmd_line.GetAttrId(Drawable::Attr(font)); layout.pad_wide_chars=1; cmd_line.Init(this,0); }

  virtual ~TextBox() {}
  virtual point RelativePosition(const point&) const;
  virtual const Drawable::Attr *GetAttr(int attr) const;
  virtual int CommandLines() const { return 0; }
  virtual void Run(const string &cmd) { if (runcb) runcb(cmd); }
  virtual bool Active() const { return screen->active_textbox == this; }
  virtual bool NotActive() const { return !box.within(screen->mouse); }
  virtual void Activate()   { if (!Active()) { if (auto g=screen->active_textbox) g->Deactivate(); screen->active_textbox=this; } }
  virtual void Deactivate() { if (Active()) screen->active_textbox = screen->default_textbox(); }
  virtual bool ToggleActive() { if (!Active()) Activate(); else Deactivate(); return Active(); }
  virtual void Input(char k) { cmd_line.UpdateText(cursor.i.x++, String16(1, *MakeUnsigned<char>(&k)), cursor.attr); UpdateCommandFB(); UpdateCursor(); }
  virtual void Erase()       { if (!cursor.i.x) return; cmd_line.Erase(--cursor.i.x, 1); UpdateCommandFB(); UpdateCursor(); }
  virtual void CursorRight() { UpdateCursorX(min(cursor.i.x+1, cmd_line.Size())); }
  virtual void CursorLeft()  { UpdateCursorX(max(cursor.i.x-1, 0)); }
  virtual void Home()        { UpdateCursorX(0); }
  virtual void End()         { UpdateCursorX(cmd_line.Size()); }
  virtual void HistUp()      { if (int c=cmd_last.ring.count) { AssignInput(cmd_last[cmd_last_ind]); cmd_last_ind=max(cmd_last_ind-1, -c); } }
  virtual void HistDown()    { if (int c=cmd_last.ring.count) { AssignInput(cmd_last[cmd_last_ind]); cmd_last_ind=min(cmd_last_ind+1, -1); } }
  virtual void Enter();

  virtual String16 Text16() const { return cmd_line.Text16(); }
  virtual void AssignInput(const string &text) { cmd_line.AssignText(text); UpdateCommandFB(); UpdateCursorX(cmd_line.Size()); }
  void SetColors(Colors *C);

  virtual const LinesFrameBuffer *GetFrameBuffer() const { return &cmd_fb; }
  virtual       LinesFrameBuffer *GetFrameBuffer()       { return &cmd_fb; }
  virtual void ResetGL() { cmd_fb.ResetGL(); }
  virtual void UpdateCursorX(int x) { cursor.i.x = x; UpdateCursor(); }
  virtual void UpdateCursor() { cursor.p = cmd_line.data->glyphs.Position(cursor.i.x) + point(0, font->Height()); }
  virtual void UpdateCommandFB() { UpdateLineFB(&cmd_line, &cmd_fb); }
  virtual void UpdateLineFB(Line *L, LinesFrameBuffer *fb, int flag=0);
  virtual void Draw(const Box &b);
  virtual void DrawCursor(point p);
  virtual void UpdateToken(Line*, int word_offset, int word_len, int update_type, const TokenProcessor<DrawableBox>*);
  virtual void UpdateLongToken(Line *BL, int beg_offset, Line *EL, int end_offset, const string &text, int update_type);

  void AddHistory  (const string &cmd);
  int  ReadHistory (const string &dir, const string &name);
  int  WriteHistory(const string &dir, const string &name, const string &hdr);
};

struct UnbackedTextBox : public TextBox {
  UnbackedTextBox(const FontRef &F=FontRef()) : TextBox(0, F) {}
  virtual void UpdateCommandFB() {}
};

struct TiledTextBox : public TextBox {
  point offset;
  TilesInterface *tiles=0;
  TiledTextBox(const FontRef &F=FontRef()) : TextBox(0, F) { cmd_fb.paint_cb = bind(&TiledTextBox::PaintCB, this, _1, _2, _3); }
  void AssignTarget(TilesInterface *T, const point &p) { tiles=T; offset=p; }
  point PaintCB(Line *l, point lp, const Box &b);
};

struct TextArea : public TextBox {
  Lines line;
  LinesFrameBuffer line_fb;
  Time write_last=Time(0);
  bool wrap_lines=1, write_timestamp=0, write_newline=1, reverse_line_fb=0;
  int line_left=0, end_line_adjust=0, start_line_cutoff=0, end_line_cutoff=0;
  int scroll_inc=10, scrolled_lines=0;
  float v_scrolled=0, h_scrolled=0, last_v_scrolled=0, last_h_scrolled=0;

  TextArea(GraphicsDevice *D, const FontRef &F=FontRef(), int S=200, int LC=10) :
    TextBox(D, F, LC), line(this, S), line_fb(D) { if (selection.enabled) InitSelection(); }
  virtual ~TextArea() {}

  /// Write() is thread-safe.
  virtual void Write(const StringPiece &s, bool update_fb=true, bool release_fb=true);
  virtual void WriteCB(const string &s, bool update_fb, bool release_fb) { return Write(s, update_fb, release_fb); }
  virtual void PageUp  () { AddVScroll(-scroll_inc); }
  virtual void PageDown() { AddVScroll( scroll_inc); }
  virtual void ScrollUp  () { PageUp(); }
  virtual void ScrollDown() { PageDown(); }
  virtual void Resized(const Box &b);
  virtual void CheckResized(const Box &b);
  virtual void SetDimension(int w, int h) {}

  virtual void Redraw(bool attach=true);
  virtual void UpdateScrolled();
  virtual void UpdateHScrolled(int x, bool update_fb=true);
  virtual void UpdateVScrolled(int dist, bool reverse, int first_ind, int first_offset, int first_len);
  virtual int UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len);
  virtual int WrappedLines() const { return line.wrapped_lines; }
  virtual const LinesFrameBuffer *GetFrameBuffer() const { return &line_fb; }
  virtual       LinesFrameBuffer *GetFrameBuffer()       { return &line_fb; }
  virtual void ResetGL() { line_fb.ResetGL(); TextBox::ResetGL(); }
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
  void AddVScroll(int n) { v_scrolled = Clamp(v_scrolled + PercentOfLines(n), 0, 1); UpdateScrolled(); }
  void SetVScroll(int n) { v_scrolled = Clamp(0          + PercentOfLines(n), 0, 1); UpdateScrolled(); }
  int LayoutBackLine(Lines *l, int i) { return Wrap() ? (*l)[-i-1].Layout(line_fb.w) : 1; }

  void InitSelection();
  void DrawSelection();
  void DragCB(int button, int x, int y, int down);
  void CopyText(const Selection::Point &beg, const Selection::Point &end);
  string CopyText(int beg_line_ind, int beg_char_ind, int end_line_end, int end_char_ind, bool add_nl);
};

struct Editor : public TextArea {
  struct LineOffset { 
    long long offset;
    int size, wrapped_lines;
    PieceIndex annotation;
    LineOffset(int O=0, int S=0, int WL=1) : offset(O), size(S), wrapped_lines(WL) {}
    static string GetString(const LineOffset *v) { return StrCat(v->offset); }
    static int    GetLines (const LineOffset *v) { return v->wrapped_lines; }
    static int VectorGetLines(const vector<LineOffset> &v, int i) { return v[i].wrapped_lines; }
  };
  typedef PrefixSumKeyedRedBlackTree<int, LineOffset> LineMap;

  shared_ptr<File> file;
  LineMap file_line;
  FreeListVector<String16> edits;
  Line *cursor_line=0;
  LineOffset *cursor_offset=0;
  vector<pair<int,int>> annotation;
  int last_fb_width=0, last_fb_lines=0, last_first_line=0;
  int wrapped_lines=0, fb_wrapped_lines=0;
  int cursor_line_number=0, cursor_line_number_offset=0;
  bool opened=0;
  IDE::Project *project=0;
  unique_ptr<IDE::File> ide_file;
  Editor(GraphicsDevice *D, const FontRef &F=FontRef(), File *I=0, bool Wrap=0);

  void Input(char k)  { Modify(false, k); }
  void Enter()        { Modify(false, '\r'); }
  void Erase()        { Modify(true,  0); }
  void CursorLeft()   { UpdateCursorX(max(cursor.i.x-1, 0)); }
  void CursorRight()  { UpdateCursorX(min(cursor.i.x+1, CursorLineSize())); }
  void Home()         { UpdateCursorX(0); }
  void End()          { UpdateCursorX(CursorLineSize()); }
  void HistUp();
  void HistDown();
  void SelectionCB(const Selection::Point&);

  int WrappedLines() const { return wrapped_lines; }
  int CursorLineSize() const { return cursor_line ? cursor_line->Size() : 0; }
  void ToggleShouldWrap() { SetShouldWrap(!line_fb.wrap); }
  void SetShouldWrap(bool);
  void AddWrappedLines(int n);
  void UpdateWrappedLines(int cur_font_size, int width);
  void Reload() { last_fb_width=0; wrapped_lines=0; RefreshLines(); }
  int RefreshLines() { last_fb_lines=0; return UpdateLines(last_v_scrolled, 0, 0, 0); }
  int UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len);
  void UpdateCursor();
  void UpdateCursorLine();
  void UpdateCursorX(int x);
  int CursorLinesChanged(const String16 &b, int add_lines=0);
  int ModifyCursorLine();
  void Modify(bool erase, int c);
  int Save();
  FileNameAndOffset FindDefinition(const point &p);
  void UpdateAnnotation();
};

struct Terminal : public TextArea {
  struct State { enum { TEXT=0, ESC=1, CSI=2, OSC=3, CHARSET=4 }; };
  struct ByteSink {
    virtual int Write(const StringPiece &b) = 0;
    virtual void IOCtlWindowSize(int w, int h) {}
  };
  struct Controller : public ByteSink {
    bool ctrl_down=0, frame_on_keyboard_input=0;
    virtual ~Controller() {}
    virtual int Open(TextArea*) = 0;
    virtual StringPiece Read() = 0;
    virtual void Close() {}
    virtual void Dispose() {}
  };

  ByteSink *sink=0;
  int term_width=0, term_height=0, parse_state=State::TEXT;
  int scroll_region_beg=0, scroll_region_end=0, tab_width=8;
  string parse_text, parse_csi, parse_osc;
  unsigned char parse_charset=0;
  bool parse_osc_escape=0, cursor_enabled=1, first_resize=1;
  point term_cursor=point(1,1), saved_term_cursor=point(1,1);
  LinesFrameBuffer::FromLineCB fb_cb;
  LinesFrameBuffer *last_fb=0;
  Border clip_border;
  set<int> tab_stop;

  Terminal(ByteSink *O, GraphicsDevice *D, const FontRef &F=FontRef());
  virtual ~Terminal() {}
  virtual void Resized(const Box &b);
  virtual void ResizedLeftoverRegion(int w, int h, bool update_fb=true);
  virtual void SetScrollRegion(int b, int e, bool release_fb=false);
  virtual void SetDimension(int w, int h);
  virtual void Draw(const Box &b, int flag=DrawFlag::Default, Shader *shader=0);
  virtual void Write(const StringPiece &s, bool update_fb=true, bool release_fb=true);
  virtual void Input(char k) {                       sink->Write(StringPiece(&k, 1)); }
  virtual void Erase      () { char k = 0x7f;        sink->Write(StringPiece(&k, 1)); }
  virtual void Enter      () { char k = '\r';        sink->Write(StringPiece(&k, 1)); }
  virtual void Tab        () { char k = '\t';        sink->Write(StringPiece(&k, 1)); }
  virtual void Escape     () { char k = 0x1b;        sink->Write(StringPiece(&k, 1)); }
  virtual void HistUp     () { char k[] = "\x1bOA";  sink->Write(StringPiece( k, 3)); }
  virtual void HistDown   () { char k[] = "\x1bOB";  sink->Write(StringPiece( k, 3)); }
  virtual void CursorRight() { char k[] = "\x1bOC";  sink->Write(StringPiece( k, 3)); }
  virtual void CursorLeft () { char k[] = "\x1bOD";  sink->Write(StringPiece( k, 3)); }
  virtual void PageUp     () { char k[] = "\x1b[5~"; sink->Write(StringPiece( k, 4)); }
  virtual void PageDown   () { char k[] = "\x1b[6~"; sink->Write(StringPiece( k, 4)); }
  virtual void Home       () { char k[] = "\x1bOH";  sink->Write(StringPiece( k, 3)); }
  virtual void End        () { char k[] = "\x1bOF";  sink->Write(StringPiece( k, 3)); }
  virtual void MoveToOrFromScrollRegion(LinesFrameBuffer *fb, Line *l, const point &p, int flag);
  virtual void UpdateCursor() { cursor.p = point(GetCursorX(term_cursor.x, term_cursor.y), GetCursorY(term_cursor.y)); }
  virtual void UpdateToken(Line*, int word_offset, int word_len, int update_type, const TokenProcessor<DrawableBox>*);
  virtual bool GetGlyphFromCoords(const point &p, Selection::Point *out) { return GetGlyphFromCoordsOffset(p, out, clip ? 0 : start_line, 0); }
  virtual void ScrollUp  () { TextArea::PageDown(); }
  virtual void ScrollDown() { TextArea::PageUp(); }
  int GetCursorX(int x, int y) const {
    const Line *l = GetTermLine(y);
    return x <= l->Size() ? l->data->glyphs.Position(x-1).x : ((x-1) * font->FixedWidth());
  }
  int GetCursorY(int y) const { return (term_height - y + 1) * font->Height(); }
  int GetTermLineIndex(int y) const { return -term_height + y-1; }
  const Line *GetTermLine(int y) const { return &line[GetTermLineIndex(y)]; }
  /**/  Line *GetTermLine(int y)       { return &line[GetTermLineIndex(y)]; }
  Line *GetCursorLine() { return GetTermLine(term_cursor.y); }
  LinesFrameBuffer *GetPrimaryFrameBuffer()   { return line_fb.Attach(&last_fb); }
  LinesFrameBuffer *GetSecondaryFrameBuffer() { return cmd_fb .Attach(&last_fb); }
  LinesFrameBuffer *GetFrameBuffer(const Line *l) {
    int i = line.IndexOf(l);
    return ((-i-1 < start_line || term_height+i < skip_last_lines) ? cmd_fb : line_fb).Attach(&last_fb);
  }
  void PushBackLines (int n) { TextArea::Write(string(n, '\n'), true, false); }
  void PushFrontLines(int n) {
    for (int i=0; i<n; ++i) LineUpdate(line.InsertAt(-term_height, 1, start_line_adjust), GetPrimaryFrameBuffer(), LineUpdate::PushFront);
  }
  Border *UpdateClipBorder();
  void MoveLines(int sy, int ey, int dy, bool move_fb_p);
  void Scroll(int sl);
  void FlushParseText();
  void Newline(bool carriage_return=false);
  void NewTopline();
  void TabNext(int n);
  void TabPrev(int n);
  void Clear();
  void Redraw(bool attach=true);
  void Reset();
};

struct Console : public TextArea {
  Color color=Color(25,60,130,120);
  double screen_percent=.4;
  Callback animating_cb;
  Time anim_time=Time(333), anim_begin=Time(0);
  bool animating=0, drawing=0, bottom_or_top=0, blend=1;
  Box *scissor=0, scissor_buf;

  Console(GraphicsDevice *D, const FontRef &F, const Callback &C=Callback()) : TextArea(D, F, 200, 50), animating_cb(C)
  { line_fb.wrap=write_timestamp=1; SetToggleKey(Key::Backquote); bg_color=&Color::clear; cursor.type = Cursor::Underline; }
  Console(GraphicsDevice *D, const Callback &C=Callback()) :
    Console(D, FontDesc(A_or_B(FLAGS_console_font, FLAGS_default_font), "", 9, Color::white,
                        Color::clear, FLAGS_console_font_flag), C) {}

  virtual ~Console() {}
  virtual int CommandLines() const { return cmd_line.Lines(); }
  virtual void Run(const string &in) { screen->shell->Run(in); }
  virtual void PageUp  () { TextArea::PageDown(); }
  virtual void PageDown() { TextArea::PageUp(); }
  virtual void Activate() { TextBox::Activate(); StartAnimating(); }
  virtual void Deactivate() { TextBox::Deactivate(); StartAnimating(); }
  virtual void Draw(const Box &b, int flag=DrawFlag::Default, Shader *shader=0);
  virtual void Draw();
  void StartAnimating();
};

struct Dialog : public GUI {
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

  Dialog(GraphicsDevice*, float w, float h, int flag=0);
  virtual ~Dialog() {}
  virtual void Layout();
  virtual void Draw();
  virtual void TakeFocus() {}
  virtual void LoseFocus() {}

  void LayoutTabbed(int, const Box &b, const point &d, MouseController*, DrawableBoxArray*);
  void LayoutTitle(const Box &b, MouseController*, DrawableBoxArray*);
  void LayoutReshapeControls(const point &d, MouseController*);
  bool HandleReshape(Box *outline);
  void DrawGradient(const point &p) const { (title + p).DrawGradient(title_gradient); }
  void Reshape(bool *down) { mouse_start = screen->mouse; win_start = point(box.x, box.y); *down = 1; }

  static bool LessThan(const unique_ptr<Dialog> &l, const unique_ptr<Dialog> &r) { return l->zsort < r->zsort; }
  static void MessageBox(const string &text);
  static void TextureBox(const string &text);
};

struct DialogTab {
  Dialog *dialog;
  DrawableBoxArray child_box;
  DialogTab(Dialog *D=0) : dialog(D) {}
  bool operator<(const DialogTab &x) const { return dialog < x.dialog; }
  bool operator==(const DialogTab &x) const { return dialog == x.dialog; }
  static void Draw(const Box &b, const point &tab_dim, const vector<DialogTab>&);
};

template <class D=Dialog> struct TabbedDialog {
  GUI *gui;
  Box box;
  D *top=0;
  point tab_dim;
  unordered_set<D*> tabs;
  vector<DialogTab> tab_list;
  TabbedDialog(GUI *g, const point &d=point(200,16)) : gui(g), tab_dim(d) {}

  D *FirstTab() const { return tab_list.size() ? dynamic_cast<D*>(tab_list.begin()->dialog) : 0; }
  void AddTab(D *t) { tabs.insert(t); tab_list.emplace_back(t); SelectTab(t); }
  void DelTab(D *t) { tabs.erase(t); VectorEraseByValue(&tab_list, DialogTab(t)); ReleaseTab(t); }
  void SelectTab(D *t) { (top = t)->TakeFocus(); }
  void ReleaseTab(D *t) { if (top == t) { top=0; t->LoseFocus(); if ((top = FirstTab())) top->TakeFocus(); } }
  void Draw() { DialogTab::Draw(box, tab_dim, tab_list); if (top) top->Draw(); }
  void Layout() {
    for (auto b=tab_list.begin(), e=tab_list.end(), t=b; t != e; ++t)
      t->dialog->LayoutTabbed(t-b, box, tab_dim, &gui->mouse, &t->child_box);
  }
};

struct MessageBoxDialog : public Dialog {
  string message;
  Box messagesize;
  MessageBoxDialog(GraphicsDevice *d, const string &m) :
    Dialog(d, .25, .2), message(m) { font->Size(message, &messagesize); }
  void Draw();
};

struct TextureBoxDialog : public Dialog {
  Texture tex;
  TextureBoxDialog(GraphicsDevice *d, const string &m) :
    Dialog(d, .33, .33) { tex.ID = ::atoi(m.c_str()); }
  void Draw() { Dialog::Draw(); tex.Draw(content + box.TopLeft()); }
};

struct SliderDialog : public Dialog {
  typedef function<void(Widget::Slider*)> UpdatedCB;
  string title;
  UpdatedCB updated;
  Widget::Slider slider;
  SliderDialog(GraphicsDevice *d, const string &title="", const UpdatedCB &cb=UpdatedCB(),
               float scrolled=0, float total=100, float inc=1);
  void Layout() { Dialog::Layout(); slider.LayoutFixed(content); }
  void Draw() { Dialog::Draw(); if (slider.dirty) { slider.Update(); if (updated) updated(&slider); } }
};

struct FlagSliderDialog : public SliderDialog {
  string flag_name;
  FlagMap *flag_map;
  FlagSliderDialog(GraphicsDevice *d, const string &fn, float total=100, float inc=1);
  virtual void Updated(Widget::Slider *s) { flag_map->Set(flag_name, StrCat(s->Percent())); }
};

struct EditorDialog : public Dialog {
  struct Flag { enum { Wrap=Dialog::Flag::Next }; };
  Editor editor;
  Widget::Slider v_scrollbar, h_scrollbar;
  EditorDialog(GraphicsDevice *d, const FontRef &F, File *I, float w=.5, float h=.5, int flag=0);
  void Layout();
  void Draw();
  void TakeFocus() { editor.Activate(); }
  void LoseFocus() { editor.Deactivate(); }
};

struct HelperGUI : public GUI {
  FontRef font;
  HelperGUI() : font(FontDesc(FLAGS_default_font, "", 9, Color::white)) {}
  struct Hint { enum { UP, UPLEFT, UPRIGHT, DOWN, DOWNLEFT, DOWNRIGHT }; };
  struct Label {
    Box target, label;
    v2 target_center, label_center;
    int hint; string description;
    Label(const Box &w, const string &d, int h, Font *f, float ly, float lx);
    void AssignLabelBox() { label.x = label_center.x - label.w/2; label.y = label_center.y - label.h/2; }
  };
  vector<Label> label;
  void AddLabel(const Box &w, const string &d, int h, float ly=.05, float lx=.02) { label.push_back(Label(w, d, h, font, ly, lx)); }
  void Activate() { active=1; /* ForceDirectedLayout(); */ }
  void ForceDirectedLayout();
  void Draw();
};

}; // namespace LFL
#endif // LFL_LFAPP_GUI_H__
