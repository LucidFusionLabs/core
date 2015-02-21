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

#ifndef __LFL_LFAPP_GUI_H__
#define __LFL_LFAPP_GUI_H__
namespace LFL {

DECLARE_bool(multitouch);
DECLARE_bool(draw_grid);

struct GUI : public MouseController {
    Box box;
    Window *parent;
    BoxArray child_box;
    ToggleBool toggle_active;
    GUI(Window *W=0, const Box &B=Box()) : box(B), parent(W), toggle_active(&active)
    { if (parent) parent->mouse_gui.push_back(this); }
    virtual ~GUI() { if (parent) VectorEraseByValue(&parent->mouse_gui, this); }

    point MousePosition() const { return screen->mouse - box.TopLeft(); }
    BoxArray *Reset() { Clear(); return &child_box; }
    void Clear() { child_box.Clear(); MouseController::Clear(); }
    void UpdateBox(const Box &b, int draw_box_ind, int input_box_ind) {
        if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box = b;
        if (input_box_ind >= 0) hit           [input_box_ind].box = b;
    }
    void UpdateBoxX(int x, int draw_box_ind, int input_box_ind) {
        if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.x = x;
        if (input_box_ind >= 0) hit           [input_box_ind].box.x = x;
    }
    void UpdateBoxY(int y, int draw_box_ind, int input_box_ind) {
        if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.y = y;
        if (input_box_ind >= 0) hit           [input_box_ind].box.y = y;
    }

    virtual void SetLayoutDirty() { child_box.Clear(); }
    virtual void Layout(const Box &b) { box=b; Layout(); }
    virtual void Layout() {}
    virtual void Draw() {
        if (child_box.data.empty()) Layout();
        child_box.Draw(box.TopLeft());
    }
    virtual void HandleTextMessage(const string &s) {}

    virtual bool ToggleActive() {
        bool ret = toggle_active.Toggle();
        active ? Activate() : Deactivate();
        return ret;
    }
    virtual void ToggleConsole() { if (!active) app->shell.console(vector<string>()); }
};

struct Widget {
    struct Interface {
        GUI *gui;
        int drawbox_ind=1;
        vector<int> hitbox;
        bool del_hitbox=0;
        virtual ~Interface() { if (del_hitbox) DelHitBox(); }
        Interface(GUI *g) : gui(g) {}

        void AddClickBox(const Box &w, const MouseController::Callback &cb) { hitbox.push_back(gui->AddClickBox(w, cb)); }
        void AddHoverBox(const Box &w, const MouseController::Callback &cb) { hitbox.push_back(gui->AddHoverBox(w, cb)); }
        void AddDragBox (const Box &w, const MouseController::Callback &cb) { hitbox.push_back(gui->AddDragBox (w, cb)); }
        void DelHitBox() { for (vector<int>::const_iterator i = hitbox.begin(); i != hitbox.end(); ++i) gui->hit.Erase(*i); hitbox.clear(); }
        MouseController::HitBox &GetHitBox(int i=0) const { return gui->hit[hitbox[i]]; }
        Box GetHitBoxBox(int i=0) const { return Box::Add(GetHitBox(i).box, gui->box.TopLeft()); }
        Drawable::Box *GetDrawBox() const { return drawbox_ind >= 0 ? VectorGet(gui->child_box.data, drawbox_ind) : 0; }
    };
    struct Vector : public vector<Interface*> {
        virtual ~Vector() {}
    };
    struct Window : public Box, Interface {
        virtual ~Window() {}
        Window(GUI *Gui, Box window) : Box(window), Interface(Gui) {}
        Window(GUI *Gui, int X, int Y, int W, int H) : Interface(Gui) { x=X; y=Y; w=W; h=H; }
    };
    struct Button : public Interface {
        Box box; Drawable *drawable=0;
        Font *font=0; string text; point textsize;
        MouseController::Callback cb;
        bool init=0, hover=0; int decay=0;
        Color *outline=0; string link;
        Button() : Interface(0) {}
        Button(GUI *G, Drawable *D, Font *F, const string &T, const MouseController::Callback &CB)
            : Interface(G), drawable(D), font(F), text(T), cb(CB), init(1) { if (F && T.size()) SetText(T); }

        void SetText(const string &t) { text = t; Box w; font->Size(text, &w); textsize = w.Dimension(); }
        void EnableHover() { AddHoverBox(box, MouseController::CB(bind(&Button::ToggleHover, this))); }
        void ToggleHover() { hover = !hover; }
        void Visit() { SystemBrowser::Open(link.c_str()); }

        void Layout(Flow *flow, const point &d) { box.SetDimension(d); Layout(flow); }
        void Layout(Flow *flow) { 
            flow->SetFGColor(&Color::white);
            LayoutComplete(flow, flow->out->data[flow->AppendBox(box.w, box.h, drawable)].box);
        }
        void LayoutBox(Flow *flow, const Box &b) {
            flow->SetFGColor(&Color::white);
            if (drawable) flow->out->PushBack(b, flow->cur_attr, drawable, &drawbox_ind);
            LayoutComplete(flow, b);
        }
        void LayoutComplete(Flow *flow, const Box &b) {
            box = b;
            hitbox.clear();
            AddClickBox(box, cb);
            if (outline) {
                flow->SetFont(0);
                flow->SetFGColor(outline);
                flow->out->PushBack(box, flow->cur_attr, Singleton<BoxOutline>::Get());
            }
            point save_p = flow->p;
            flow->SetFont(font);
            flow->SetFGColor(0);
            flow->p = box.Position() + point(Box(0, 0, box.w, box.h).centerX(textsize.x), 0);
            flow->AppendText(text);
            flow->p = save_p;
            init = 0;
        }
    };
    struct Scrollbar : public Interface {
        struct Flag { enum { Attached=1, Horizontal=2, AttachedHorizontal=Attached|Horizontal }; };
        Box win;
        int flag=0, doc_height=200, dot_size=25;
        float scrolled=0, last_scrolled=0, increment=20;
        Color color=Color(15, 15, 15, 55);
        Font *menuicon2=0;
        bool dragging=0, dirty=0;
        virtual ~Scrollbar() {}
        Scrollbar(GUI *Gui, Box window=Box(), int f=Flag::Attached) : Interface(Gui), win(window), flag(f),
        menuicon2(Fonts::Get("MenuAtlas2", 0, Color::black, 0)) {
            if (win.w && win.h) { if (f & Flag::Attached) LayoutAttached(win); else LayoutFixed(win); } 
        }

        void LayoutFixed(const Box &w) { win = w; Layout(dot_size, dot_size, flag & Flag::Horizontal); }
        void LayoutAttached(const Box &w) {
            win = w;
            win.y = -win.h;
            int aw = dot_size, ah = dot_size;
            bool flip = flag & Flag::Horizontal;
            if (!flip) { win.x += win.w - aw - 1; win.w = aw; }
            else win.h = ah;
            Layout(aw, ah, flip);
        }
        void Layout(int aw, int ah, bool flip) {
            Box arrow_down = win;
            if (flip) { arrow_down.w = aw; win.x += aw; }
            else      { arrow_down.h = ah; win.y += ah; }

            Box scroll_dot = arrow_down, arrow_up = win;
            if (flip) { arrow_up.w = aw; win.w -= 2*aw; arrow_up.x += win.w; }
            else      { arrow_up.h = ah; win.h -= 2*ah; arrow_up.y += win.h; }

            if (gui) {
                int attr_id = gui->child_box.attr.GetAttrId(Drawable::Attr());
                gui->child_box.PushBack(arrow_up,   attr_id, menuicon2 ? menuicon2->FindGlyph(flip ? 64 : 66) : 0);
                gui->child_box.PushBack(arrow_down, attr_id, menuicon2 ? menuicon2->FindGlyph(flip ? 65 : 61) : 0);
                gui->child_box.PushBack(scroll_dot, attr_id, menuicon2 ? menuicon2->FindGlyph(            72) : 0, &drawbox_ind);

                AddClickBox(scroll_dot, MouseController::CB(bind(&Scrollbar::DragScrollDot, this)));
                AddClickBox(arrow_up,   MouseController::CB(bind(flip ? &Scrollbar::ScrollDown : &Scrollbar::ScrollUp,   this)));
                AddClickBox(arrow_down, MouseController::CB(bind(flip ? &Scrollbar::ScrollUp   : &Scrollbar::ScrollDown, this)));
            }
            Update(true);
        }
        void Update(bool force=false) {
            if (!app->input.MouseButton1Down()) dragging = false;
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
        void SetDocHeight(int v) { doc_height = v; }
        void DragScrollDot() { dragging = true; dirty = true; }
        void ScrollUp  () { scrolled -= increment / doc_height; Clamp(&scrolled, 0, 1); dirty=true; }
        void ScrollDown() { scrolled += increment / doc_height; Clamp(&scrolled, 0, 1); dirty=true; }
        float ScrollDelta() { float ret=scrolled-last_scrolled; last_scrolled=scrolled; return ret; }
        float AddScrollDelta(float cur_val) { 
            scrolled = Clamp(cur_val + ScrollDelta(), 0, 1);
            if (Typed::EqualChanged(&last_scrolled, scrolled)) dirty = 1;
            return scrolled;
        }
    };
};

struct KeyboardGUI : public KeyboardController {
    typedef function<void(const string &text)> RunCB;
    Window *parent;
    ToggleBool toggle_active;
    Bind toggle_bind;
    RunCB runcb;
    RingVector<string> lastcmd;
    int lastcmd_ind=-1;
    KeyboardGUI(Window *W, Font *F, int LastCommands=50)
        : parent(W), toggle_active(&active), lastcmd(LastCommands) { parent->keyboard_gui.push_back(this); }
    virtual ~KeyboardGUI() { if (parent) VectorEraseByValue(&parent->keyboard_gui, this); }
    virtual void Enable() { active = true; }
    virtual bool Toggle() { return toggle_active.Toggle(); }
    virtual void Run(string cmd) { if (runcb) runcb(cmd); }
    virtual void SetToggleKey(int TK, int TM=ToggleBool::Default) { toggle_bind.key=TK; toggle_active.mode=TM; }

    void AddHistory  (const string &cmd);
    int  ReadHistory (const string &dir, const string &name);
    int  WriteHistory(const string &dir, const string &name, const string &hdr);
};

struct TextGUI : public KeyboardGUI {
    struct Lines;
    struct Link {
        Widget::Button widget;
        DOM::Attr image_src;
        DOM::HTMLImageElement image;
        Link(GUI *G, const string &url) : widget(G, 0, 0, "", MouseController::CB(bind(&Widget::Button::Visit, &widget))),
        image_src(0), image(0) { widget.link=url; widget.del_hitbox=true; widget.EnableHover(); }
    };
    typedef function<void(Link*)> LinkCB;
    struct LineData {
        BoxArray glyphs;
        Box box; Flow flow;
        vector<shared_ptr<Link> > links;
    };
    struct Line {
        point p;
        Lines *cont=0;
        TextGUI *parent=0;
        shared_ptr<LineData> data;
        Line() : data(new LineData()) {}
        Line &operator=(const Line &s) { data=s.data; return *this; }
        static void Move (Line &t, Line &s) { swap(t.data, s.data); }
        static void MoveP(Line &t, Line &s) { swap(t.data, s.data); t.p=s.p; }

        int GetAttrId(const Drawable::Attr &a) { return data->glyphs.attr.GetAttrId(a); }
        void InitFlow() { data->flow = Flow(&data->box, parent->font, &data->glyphs, &parent->layout); }
        void Init(Lines *C, TextGUI *P) { cont=C; parent=P; InitFlow(); }
        int Size () const { return data->glyphs.Size(); }
        int Lines() const { return 1+data->glyphs.line.size(); }
        string Text() const { return data->glyphs.Text(); }
        void Clear() { data->glyphs.Clear(); InitFlow(); }
        void Erase(int o, unsigned l=UINT_MAX) { data->glyphs.Erase(o, l, true); data->flow.p.x = BackOrDefault(data->glyphs.data).box.right(); }
        void InsertTextAt(int o, const StringPiece   &s, int a=0) { data->glyphs.InsertAt(o, EncodeText(s, a, data->glyphs.Position(o))); }
        void InsertTextAt(int o, const String16Piece &s, int a=0) { data->glyphs.InsertAt(o, EncodeText(s, a, data->glyphs.Position(o))); }
        void AppendText  (       const StringPiece   &s, int a=0) { data->flow.AppendText(s, a); }
        void AppendText  (       const String16Piece &s, int a=0) { data->flow.AppendText(s, a); }
        void AssignText  (       const StringPiece   &s, int a=0) { Clear(); AppendText(s, a); }
        void AssignText  (       const String16Piece &s, int a=0) { Clear(); AppendText(s, a); }
        vector<Drawable::Box> EncodeText(const StringPiece   &s, int a, const point &p) { BoxArray b; parent->font->Encode(s, Box(p,0,0), &b, Font::Flag::AssignFlowX, a); return b.data; }
        vector<Drawable::Box> EncodeText(const String16Piece &s, int a, const point &p) { BoxArray b; parent->font->Encode(s, Box(p,0,0), &b, Font::Flag::AssignFlowX, a); return b.data; }
        void UpdateAttr(int ind, int len, int a) {}
        void Layout(Box win, bool flush=0) {
            if (data->box.w == win.w && !flush) return;
            data->box = win;
            ScopedDeltaTracker<int> SWLT(cont ? &cont->wrapped_lines : 0, bind(&Line::Lines, this));
            BoxArray b;
            swap(b, data->glyphs);
            data->glyphs.attr.source = b.attr.source;
            Clear();
            data->flow.AppendBoxArrayText(b);
        }
        bool UpdateText(int x, const String16Piece &v, int attr, int max_width=0) {
            int size = Size(); bool append = size <= x;
            if (size < x) AppendText(string(x - size, ' '), attr);
            else if (!parent->insert_mode) Erase(x, v.size());
            if (size == x) AppendText  (   v, attr);
            else           InsertTextAt(x, v, attr);
            if (parent->insert_mode && max_width) Erase(max_width);
            return append;
        }
        void UpdateAttr(int ind, int len) {
            if (!parent || !parent->clickable_links) return;
#if 0
            for (int i = 0, len = 0; i < ret.size(); i += len) {
                for (/**/; i < ret.size(); i++)
                    if (PrefixMatch(s.data() + i, "http://") ||
                        PrefixMatch(s.data() + i, "https://")) break;
                if (i == ret.size()) break;
                const short *start = s.data() + i, *end = nextchar(start, isspace);
                len = end ? (end - start) : (s.size() - i);
                UpdateAttr(i, len, attr | Attr::Link);
                // Link *link = new Link(parent->gui, ::String::ToUTF8(start, len));
                // links.push_back(link);
                // if (parent->new_link_cb) parent->new_link_cb(link);
            }
#endif
        }
        int Layout(int width=0, bool flush=0) { Layout(Box(0,0,width,0), flush); return Lines(); }
        point Draw(point pos, int relayout_width=-1, int g_offset=0, int g_len=-1) {
            if (relayout_width >= 0) Layout(relayout_width);
            data->glyphs.Draw((p = pos), g_offset, g_len);
            return p - point(0, parent->font->height + data->glyphs.height);
        }
    };
    struct Lines : public RingVector<Line> {
        int wrapped_lines;
        function<void(Line&, Line&)> move_cb, movep_cb;
        Lines(TextGUI *P, int N) :
            RingVector<Line>(N), wrapped_lines(N),
            move_cb (bind(&Line::Move,  _1, _2)), 
            movep_cb(bind(&Line::MoveP, _1, _2)) { for (auto &i : data) i.Init(this, P); }

        Line *PushFront() { Line *l = RingVector<Line>::PushFront(); l->Clear(); return l; }
        Line *InsertAt(int dest_line, int lines=1, int dont_move_last=0) {
            CHECK(lines); CHECK_LT(dest_line, 0);
            int clear_dir = 1;
            if (dest_line == -1) { ring.PushBack(lines); clear_dir = -1; }
            else Move(*this, dest_line+lines, dest_line, -dest_line-lines-dont_move_last, move_cb);
            for (int i=0; i<lines; i++) (*this)[dest_line + i*clear_dir].Clear();
            return &(*this)[dest_line];
        }
        static int GetBackLineLines(const Lines &l, int i) { return l[-i-1].Lines(); }
    };
    struct LinesFrameBuffer : public RingFrameBuffer<Line> {
        typedef function<LinesFrameBuffer*(const Line*)> FromLineCB;
        struct Flag { enum { NoLayout=1, NoVWrap=2, Flush=4 }; };
        PaintCB paint_cb = &LinesFrameBuffer::PaintCB;
        int lines=0;

        LinesFrameBuffer *Attach(LinesFrameBuffer **last_fb);
        virtual bool SizeChanged(int W, int H, Font *font);
        virtual int Height() const { return lines * font_height; }
        tvirtual void Clear(Line *l) { RingFrameBuffer::Clear(l, Box(w, l->Lines() * font_height), true); }
        tvirtual void Update(Line *l, int flag=0);
        tvirtual void Update(Line *l, const point &p, int flag=0) { l->p=p; Update(l, flag); }
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
        ~LineUpdate() {
            if (!fb->lines || (flag & DontUpdate)) v->Layout(fb->wrap ? fb->w : 0);
            else if (flag & PushFront) { if (o) fb->PushFrontAndUpdateOffset(v,o); else fb->PushFrontAndUpdate(v); }
            else if (flag & PushBack)  { if (o) fb->PushBackAndUpdateOffset (v,o); else fb->PushBackAndUpdate (v); }
            else fb->Update(v);
        }
        Line *operator->() const { return v; }
    };
    struct Cursor {
        enum { Underline=1, Block=2 };
        int type=Underline, blink_time=333, attr=0;
        Time blink_begin=0;
        point i, p;
    };
    struct Selection {
        bool enabled=1, changing=0, changing_previously=0;
        struct Point { 
            int line_ind=0, char_ind=0; point click; Box glyph;
            bool operator<(const Point &c) const { SortImpl2(c.glyph.y, glyph.y, glyph.x, c.glyph.x); }
            string DebugString() const { return StrCat("i=", click.DebugString(), " l=", line_ind, " c=", char_ind, " b=", glyph.DebugString()); }
        } beg, end;
        Box3 box;
    };

    Font *font;
    Flow::Layout layout;
    Cursor cursor;
    Selection selection;
    Line cmd_line;
    LinesFrameBuffer cmd_fb;
    string cmd_prefix="> ";
    Color cmd_color=Color::white, selection_color=Color(Color::grey70, 0.5);
    bool deactivate_on_enter=0, clickable_links=0, insert_mode=1;
    int start_line=0, end_line=0, start_line_adjust=0, skip_last_lines=0;
    TextGUI(Window *W, Font *F) : KeyboardGUI(W, F), font(F)
    { cmd_line.Init(0,this); cmd_line.GetAttrId(Drawable::Attr(F)); }

    virtual ~TextGUI() {}
    virtual int CommandLines() const { return 0; }
    virtual void Input(char k) { cmd_line.UpdateText(cursor.i.x++, String16(1, k), cursor.attr); UpdateCommandFB(); UpdateCursor(); }
    virtual void Erase()       { if (!cursor.i.x) return;       cmd_line.Erase(--cursor.i.x, 1); UpdateCommandFB(); UpdateCursor(); }
    virtual void CursorRight() { cursor.i.x = min(cursor.i.x+1, cmd_line.Size()); UpdateCursor(); }
    virtual void CursorLeft()  { cursor.i.x = max(cursor.i.x-1, 0);               UpdateCursor(); }
    virtual void Home()        { cursor.i.x = 0;                                  UpdateCursor(); }
    virtual void End()         { cursor.i.x = cmd_line.Size();                    UpdateCursor(); }
    virtual void HistUp()      { if (int c=lastcmd.ring.count) { AssignInput(lastcmd[lastcmd_ind]); lastcmd_ind=max(lastcmd_ind-1, -c); } }
    virtual void HistDown()    { if (int c=lastcmd.ring.count) { AssignInput(lastcmd[lastcmd_ind]); lastcmd_ind=min(lastcmd_ind+1, -1); } }
    virtual void Enter();

    virtual string Text() const { return cmd_line.Text(); }
    virtual void AssignInput(const string &text)
    { cmd_line.AssignText(text); cursor.i.x=cmd_line.Size(); UpdateCommandFB(); UpdateCursor(); }

    virtual void UpdateCursor() { cursor.p = cmd_line.data->glyphs.Position(cursor.i.x); }
    virtual void UpdateCommandFB() { 
        cmd_fb.fb.Attach();
        ScopedDrawMode drawmode(DrawMode::_2D);
        cmd_fb.PushBackAndUpdate(&cmd_line); // cmd_fb.OverwriteUpdate(&cmd_line, cursor.x)
        cmd_fb.fb.Release();
    }
    virtual void Draw(const Box &b);
    virtual void DrawCursor(point p);
};

struct TextArea : public TextGUI {
    Lines line;
    LinesFrameBuffer line_fb;
    GUI mouse_gui;
    Time write_last=0;
    const Border *clip=0;
    bool wrap_lines=1, write_timestamp=0, write_newline=1, reverse_line_fb=0;
    int line_left=0, end_line_adjust=0, start_line_cutoff=0, end_line_cutoff=0;
    int scroll_inc=10, scrolled_lines=0;
    float v_scrolled=0, h_scrolled=0, last_v_scrolled=0, last_h_scrolled=0;
    LinkCB new_link_cb, hover_link_cb;

    TextArea(Window *W, Font *F, int S=200) : TextGUI(W, F), line(this, S), mouse_gui(W) {}
    virtual ~TextArea() {}

    /// Write() is thread-safe.
    virtual void Write(const string &s, bool update_fb=true, bool release_fb=true);
    virtual void PageUp  () { v_scrolled = Clamp(v_scrolled - (float)scroll_inc/(WrappedLines()-1), 0, 1); UpdateScrolled(); }
    virtual void PageDown() { v_scrolled = Clamp(v_scrolled + (float)scroll_inc/(WrappedLines()-1), 0, 1); UpdateScrolled(); }
    virtual void Resized(int w, int h);

    virtual void Redraw(bool attach=true);
    virtual void UpdateScrolled();
    virtual void UpdateHScrolled(int x, bool update_fb=true);
    virtual void UpdateVScrolled(int dist, bool reverse, int first_ind, int first_offset, int first_len);
    virtual int UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len);
    virtual int WrappedLines() const { return line.wrapped_lines; }
    virtual LinesFrameBuffer *GetFrameBuffer() { return &line_fb; }

    virtual void Draw(const Box &w, bool cursor);
    virtual void DrawWithShader(const Box &w, bool cursor, Shader *shader)
    { glTimeResolutionShader(shader); Draw(w, cursor); screen->gd->UseShader(0); }

    bool Wrap() const { return line_fb.wrap; }
    int LineFBPushBack () const { return reverse_line_fb ? LineUpdate::PushFront : LineUpdate::PushBack;  }
    int LineFBPushFront() const { return reverse_line_fb ? LineUpdate::PushBack  : LineUpdate::PushFront; }
    int LayoutBackLine(Lines *l, int i) { return Wrap() ? (*l)[-i-1].Layout(line_fb.w) : 1; }

    void DrawSelection();
    void ClickCB(int button, int x, int y, int down);
    bool GetGlyphFromCoords(const point &p, Selection::Point *out);
    void CopyText(const Selection::Point &beg, const Selection::Point &end);
};

struct Editor : public TextArea {
    struct LineOffset { 
        long long offset; int size, wrapped_lines;
        LineOffset(int O=0, int S=0, int WL=1) : offset(O), size(S), wrapped_lines(WL) {}
        static string GetString(const LineOffset *v) { return StrCat(v->offset); }
        static int    GetLines (const LineOffset *v) { return v->wrapped_lines; }
        static int VectorGetLines(const vector<LineOffset> &v, int i) { return v[i].wrapped_lines; }
    };
    typedef PrefixSumKeyedRedBlackTree<int, LineOffset> LineMap;

    shared_ptr<File> file;
    LineMap file_line;
    FreeListVector<string> edits;
    int last_fb_width=0, last_fb_lines=0, last_first_line=0, wrapped_lines=0, fb_wrapped_lines=0;

    Editor(Window *W, Font *F, File *I, bool Wrap=0) : TextArea(W, F), file(I) {
        reverse_line_fb = 1;
        line_fb.wrap = Wrap;
        file_line.node_value_cb = &LineOffset::GetLines;
        file_line.node_print_cb = &LineOffset::GetString;
    }

    int WrappedLines() const { return wrapped_lines; }
    void UpdateWrappedLines(int cur_font_size, int width);
    int UpdateLines(float v_scrolled, int *first_ind, int *first_offset, int *first_len);
    void Draw(const Box &box) { TextArea::Draw(box, true); }
};

struct Terminal : public TextArea, public Drawable::AttrSource {
    struct State { enum { TEXT=0, ESC=1, CSI=2, OSC=3, CHARSET=4 }; };
    struct Attr {
        enum { Bold=1<<8, Underline=1<<9, Blink=1<<10, Reverse=1<<11, Italic=1<<12, Link=1<<13 };
        static void SetFGColorIndex(int *a, int c) { *a = (*a & ~0x0f) | ((c & 0xf)     ); }
        static void SetBGColorIndex(int *a, int c) { *a = (*a & ~0xf0) | ((c & 0xf) << 4); }
        static int GetFGColorIndex(int a) { return (a & 0xf) | ((a & Bold) ? (1<<3) : 0); }
        static int GetBGColorIndex(int a) { return (a>>4) & 0xf; }
    };
    struct Colors { Color c[16]; int normal_index, bold_index, bg_index; };
    struct StandardVGAColors : public Colors {
        StandardVGAColors() { 
            c[0] = Color(  0,   0,   0); c[ 8] = Color( 85,  85,  85);
            c[1] = Color(170,   0,   0); c[ 9] = Color(255,  85,  85);
            c[2] = Color(  0, 170,   0); c[10] = Color( 85, 255,  85);
            c[3] = Color(170,  85,   0); c[11] = Color(255, 255,  85);
            c[4] = Color(  0,   0, 170); c[12] = Color( 85,  85, 255);
            c[5] = Color(170,   0, 170); c[13] = Color(255,  85, 255);
            c[6] = Color(  0, 170, 170); c[14] = Color( 85, 255, 255);
            c[7] = Color(170, 170, 170); c[15] = Color(255, 255, 255);
            bg_index = 0; normal_index = 7; bold_index = 15;
        }
    };
    /// Solarized palette by Ethan Schoonover
    struct SolarizedColors : public Colors {
        SolarizedColors() { 
            c[0] = Color(  7,  54,  66); c[ 8] = Color(  0,  43,  54);
            c[1] = Color(220,  50,  47); c[ 9] = Color(203,  75,  22);
            c[2] = Color(133, 153,   0); c[10] = Color( 88, 110, 117);
            c[3] = Color(181, 137,   0); c[11] = Color(101, 123, 131);
            c[4] = Color( 38, 139, 210); c[12] = Color(131, 148, 150);
            c[5] = Color(211,  54, 130); c[13] = Color(108, 113, 196);
            c[6] = Color( 42, 161, 152); c[14] = Color(147, 161, 161);
            c[7] = Color(238, 232, 213); c[15] = Color(253, 246, 227);
            bg_index = 8; normal_index = 12; bold_index = 12;
        }
    };

    int fd, term_width=0, term_height=0, parse_state=State::TEXT, default_cursor_attr=0;
    int scroll_region_beg=0, scroll_region_end=0;
    string parse_text, parse_csi, parse_osc;
    unsigned char parse_charset=0;
    bool parse_osc_escape=0, cursor_enabled=1;
    point term_cursor=point(1,1), saved_term_cursor=point(1,1);
    LinesFrameBuffer::FromLineCB fb_cb;
    LinesFrameBuffer *last_fb=0;
    Border clip_border;
    Colors *colors=0;
    Color *bg_color=0;

    Terminal(int FD, Window *W, Font *F) :
        TextArea(W, F), fd(FD), fb_cb(bind(&Terminal::GetFrameBuffer, this, _1)) {
        CHECK(F->fixed_width || (F->flag & FontDesc::Mono));
        wrap_lines = write_newline = insert_mode = 0;
        for (int i=0; i<line.ring.size; i++) line[i].data->glyphs.attr.source = this;
        SetColors(Singleton<StandardVGAColors>::Get());
        cursor.attr = default_cursor_attr;
        cursor.type = Cursor::Block;
        clickable_links = 1;
        cmd_prefix = "";
    }
    virtual ~Terminal() {}
    virtual void Resized(int w, int h);
    virtual void ResizedLeftoverRegion(int w, int h, bool update_fb=true);
    virtual void SetScrollRegion(int b, int e, bool release_fb=false);
    virtual void Draw(const Box &b, bool draw_cursor);
    virtual void Write(const string &s, bool update_fb=true, bool release_fb=true);
    virtual void Input(char k) {                       write(fd, &k, 1); }
    virtual void Erase      () { char k = 0x7f;        write(fd, &k, 1); }
    virtual void Enter      () { char k = '\r';        write(fd, &k, 1); }
    virtual void Tab        () { char k = '\t';        write(fd, &k, 1); }
    virtual void Escape     () { char k = 0x1b;        write(fd, &k, 1); }
    virtual void HistUp     () { char k[] = "\x1bOA";  write(fd,  k, 3); }
    virtual void HistDown   () { char k[] = "\x1bOB";  write(fd,  k, 3); }
    virtual void CursorRight() { char k[] = "\x1bOC";  write(fd,  k, 3); }
    virtual void CursorLeft () { char k[] = "\x1bOD";  write(fd,  k, 3); }
    virtual void PageUp     () { char k[] = "\x1b[5~"; write(fd,  k, 4); }
    virtual void PageDown   () { char k[] = "\x1b[6~"; write(fd,  k, 4); }
    virtual void Home       () { char k = 'A' - 0x40;  write(fd, &k, 1); }
    virtual void End        () { char k = 'E' - 0x40;  write(fd, &k, 1);  }
    virtual void UpdateCursor() { cursor.p = point(GetCursorX(term_cursor.x), GetCursorY(term_cursor.y)); }
    virtual Drawable::Attr GetAttr(int attr) const {
        Color *fg = colors ? &colors->c[Attr::GetFGColorIndex(attr)] : 0;
        Color *bg = colors ? &colors->c[Attr::GetBGColorIndex(attr)] : 0;
        if (attr & Attr::Reverse) Typed::Swap(fg, bg);
        return Drawable::Attr(font, fg, bg, attr & Attr::Underline);
    }
    int GetCursorX(int x) const { return (x - 1) * font->FixedWidth(); }
    int GetCursorY(int y) const { return (term_height - y + 1) * font->height; }
    int GetTermLineIndex(int y) const { return -term_height + y-1; }
    Line *GetTermLine(int y) { return &line[GetTermLineIndex(y)]; }
    Line *GetCursorLine() { return GetTermLine(term_cursor.y); }
    LinesFrameBuffer *GetPrimaryFrameBuffer()   { return line_fb.Attach(&last_fb); }
    LinesFrameBuffer *GetSecondaryFrameBuffer() { return cmd_fb .Attach(&last_fb); }
    LinesFrameBuffer *GetFrameBuffer(const Line *l) {
        int i = line.IndexOf(l);
        return ((-i-1 < start_line || term_height-i < skip_last_lines) ? cmd_fb : line_fb).Attach(&last_fb);
    }
    void SetColors(Colors *C) {
        colors = C;
        Attr::SetFGColorIndex(&default_cursor_attr, colors->normal_index);
        Attr::SetBGColorIndex(&default_cursor_attr, colors->bg_index);
        bg_color = &colors->c[colors->bg_index];
    }
    void Scroll(int sy, int ey, int dy, bool move_fb_p) {
        CHECK_LT(sy, ey);
        int line_ind = GetTermLineIndex(sy), scroll_lines = ey - sy + 1, ady = abs(dy), sdy = (dy > 0 ? 1 : -1);
        Move(line, line_ind + (dy>0 ? dy : 0), line_ind + (dy<0 ? -dy : 0), scroll_lines - ady, move_fb_p ? line.movep_cb : line.move_cb);
        for (int i = 0, cy = (dy>0 ? sy : ey); i < ady; i++) GetTermLine(cy + i*sdy)->Clear();
    }
    void FlushParseText();
    void Newline(bool carriage_return=false);
    void NewTopline();
};

struct Console : public TextArea {
    string startcmd;
    double screenPercent=.4;
    Color color=Color(25,60,130,120);
    int animTime=333; Time animBegin=0;
    bool animating=0, drawing=0, bottom_or_top=0, blend=1, ran_startcmd=0;
    Console(Window *W, Font *F) : TextArea(W,F)
    { line_fb.wrap=write_timestamp=1; SetToggleKey(Key::Backquote); }

    virtual ~Console() {}
    virtual int CommandLines() const { return cmd_line.Lines(); }
    virtual void Run(string in) { app->shell.Run(in); }
    virtual void PageUp  () { TextArea::PageDown(); }
    virtual void PageDown() { TextArea::PageUp(); }
    virtual bool Toggle() {
        if (!TextGUI::Toggle()) return false;
        Time elapsed = Now()-animBegin;
        animBegin = Now() - (elapsed<animTime ? animTime-elapsed : 0);
        return true;
    }
    virtual void Draw() {
        if (!ran_startcmd && (ran_startcmd = 1)) if (startcmd.size()) Run(startcmd);

        drawing = 1;
        Time now=Now(), elapsed;
        int h = active ? (int)(screen->height*screenPercent) : 0;
        if ((animating = (elapsed=now-animBegin) < animTime)) {
            if (active) h = (int)(screen->height*(  (double)elapsed/animTime)*screenPercent);
            else        h = (int)(screen->height*(1-(double)elapsed/animTime)*screenPercent);
        } else if (!active) { drawing = 0; return; }
        
        screen->gd->FillColor(color);
        if (blend) screen->gd->EnableBlend(); 
        else       screen->gd->DisableBlend();

        int y = bottom_or_top ? 0 : screen->height-h;
        Box(0, y, screen->width, h).Draw();

        screen->gd->SetColor(Color::white);
        TextArea::Draw(Box(0, y, screen->width, h), true);
    }
    int WriteHistory(const string &dir, const string &name)
    { return KeyboardGUI::WriteHistory(dir, name, startcmd); }
};

struct Dialog : public GUI {
    struct Flag { enum { None=0, Fullscreen=1, Next=2 }; };
    Font *font=0;
    Color color=Color(25,60,130,220);
    Box title, resize_left, resize_right, resize_bottom, close;
    bool deleted=0, moving=0, resizing_left=0, resizing_right=0, resizing_top=0, resizing_bottom=0, fullscreen=0;
    point mouse_start, win_start;
    int zsort=0;
    Dialog(float w, float h, int flag=0) : GUI(screen), font(Fonts::Get(FLAGS_default_font, 14, Color::white)) {
        screen->dialogs.push_back(this);
        box = screen->Box().center(screen->Box(w, h));
        fullscreen = flag & Flag::Fullscreen;
        active = true;
        Layout();
    }
    virtual ~Dialog() {}
    virtual void Draw();
    virtual void Layout() {
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
    }
    void BringToFront() {
        if (screen->top_dialog == this) return;
        for (vector<Dialog*>::iterator i = screen->dialogs.begin(); i != screen->dialogs.end(); ++i) (*i)->zsort++; zsort = 0;
        sort(screen->dialogs.begin(), screen->dialogs.end(), LessThan);
        screen->top_dialog = this;
    }
    Box BoxAndTitle() const { return Box(box.x, box.y, box.w, box.h + title.h); }
    void Reshape(bool *down) { mouse_start = screen->mouse; win_start = point(box.x, box.y); *down = 1; }
    void MarkDeleted() { deleted = 1; }

    static bool LessThan(const Dialog *l, const Dialog *r) { return l->zsort < r->zsort; }
    static void MessageBox(const string &text);
    static void TextureBox(const string &text);
};

struct MessageBoxDialog : public Dialog {
    string message;
    Box messagesize;
    MessageBoxDialog(const string &m) : Dialog(.25, .1), message(m) { font->Size(message, &messagesize); }
    void Draw() {
        Dialog::Draw();
        { Scissor scissor(box); font->Draw(message, point(box.centerX(messagesize.w), box.centerY(messagesize.h)));  }
    }
};

struct TextureBoxDialog : public Dialog {
    Texture tex;
    TextureBoxDialog(const string &m) : Dialog(.25, .1) { tex.ID = ::atoi(m.c_str()); }
    void Draw() { Dialog::Draw(); tex.Draw(box); }
};

struct SliderTweakDialog : public Dialog {
    string flag_name;
    FlagMap *flag_map;
    Box flag_name_size;
    Widget::Scrollbar slider;
    SliderTweakDialog(const string &fn, float total=100, float inc=1) : Dialog(.3, .05),
    flag_name(fn), flag_map(Singleton<FlagMap>::Get()), slider(this, Box(), Widget::Scrollbar::Flag::Horizontal) {
        slider.increment = inc;
        slider.doc_height = total;
        slider.scrolled = atof(flag_map->Get(flag_name).c_str()) / total;
        font->Size(flag_name, &flag_name_size);
    }
    void Draw() { 
        Dialog::Draw();
        // slider.Draw(win);
        if (slider.dirty) {
            slider.dirty = 0;
            flag_map->Set(flag_name, StrCat(slider.scrolled * slider.doc_height));
        }
        font->Draw(flag_name, point(title.centerX(flag_name_size.w), title.centerY(flag_name_size.h)));
    }
};

struct EditorDialog : public Dialog {
    struct Flag { enum { Wrap=Dialog::Flag::Next }; };
    Editor editor;
    Widget::Scrollbar v_scrollbar, h_scrollbar;
    EditorDialog(Window *W, Font *F, File *I, float w=.5, float h=.5, int flag=0) : Dialog(w, h, flag), editor(W, F, I, flag & Flag::Wrap),
    v_scrollbar(this), h_scrollbar(this, Box(), Widget::Scrollbar::Flag::AttachedHorizontal) {}

    void Layout() {
        Dialog::Layout();
        if (1)              v_scrollbar.LayoutAttached(box.Dimension());
        if (!editor.Wrap()) h_scrollbar.LayoutAttached(box.Dimension());
    }
    void Draw() {
        Dialog::Draw();
        bool wrap = editor.Wrap();
        if (1)     editor.active = screen->top_dialog == this;
        if (1)     editor.v_scrolled = v_scrollbar.AddScrollDelta(editor.v_scrolled);
        if (!wrap) editor.h_scrolled = h_scrollbar.AddScrollDelta(editor.h_scrolled);
        if (1)     editor.UpdateScrolled();
        if (1)     editor.Draw(box);
        if (1)     GUI::Draw();
        if (1)     v_scrollbar.Update();
        if (!wrap) h_scrollbar.Update();
    }
};

namespace DOM {
struct Renderer : public Object {
    FloatContainer box;
    ComputedStyle style;
    bool style_dirty=1, layout_dirty=1;
    Flow *flow=0, *parent_flow=0, child_flow;
    DOM::Node *absolute_parent=0;
    Asset *background_image=0;
    BoxArray child_box, child_bg;
    Tiles *tiles=0;

    Box content, padding, border, margin, clip_rect;
    Color color, background_color, border_top, border_bottom, border_right, border_left, outline;

    bool display_table_element=0, display_table=0, display_inline_table=0, display_block=0, display_inline=0;
    bool display_inline_block=0, display_list_item=0, display_none=0, block_level_box=0, establishes_block=0;
    bool position_relative=0, position_absolute=0, position_fixed=0, positioned=0, float_left=0, float_right=0, floating=0, normal_flow=0;
    bool done_positioned=0, done_floated=0, textalign_center=0, textalign_right=0, underline=0, overline=0, midline=0, blink=0, uppercase=0, lowercase=0, capitalize=0, valign_top=0, valign_mid=0;
    bool bgfixed=0, bgrepeat_x=0, bgrepeat_y=0, border_collapse=0, clear_left=0, clear_right=0, right_to_left=0, hidden=0, inline_block=0;
    bool width_percent=0, width_auto=0, height_auto=0, ml_auto=0, mr_auto=0, mt_auto=0, mb_auto=0, overflow_hidden=0, overflow_scroll=0, overflow_auto=0, clip=0, shrink=0, tile_context_opened=0;
    int width_px=0, height_px=0, ml_px=0, mr_px=0, mt_px=0, mb_px=0, bl_px=0, br_px=0, bt_px=0, bb_px=0, pl_px=0, pr_px=0, pt_px=0, pb_px=0, o_px=0;
    int lineheight_px=0, charspacing_px=0, wordspacing_px=0, valign_px=0, bgposition_x=0, bgposition_y=0, bs_t=0, bs_b=0, bs_r=0, bs_l=0, os=0;
    int clear_height=0, row_height=0, cell_colspan=0, cell_rowspan=0, extra_cell_height=0, max_child_i=-1;

    Renderer(Node *N) : style(N) {}

    void  UpdateStyle(Flow *F);
    Font* UpdateFont(Flow *F);
    void  UpdateDimensions(Flow *F);
    void  UpdateMarginWidth(Flow *F, int w);
    void  UpdateBackgroundImage(Flow *F);
    void  UpdateFlowAttributes(Flow *F);
    int ClampWidth(int w);
    void PushScissor(const Box &w);
    void Finish();

    bool Dirty() { return style_dirty || layout_dirty; }
    void InputActivate() { style.node->ownerDocument->gui->Activate(); }
    int TopBorderOffset    () { return pt_px + bt_px; }
    int RightBorderOffset  () { return pr_px + br_px; }
    int BottomBorderOffset () { return pb_px + bb_px; }
    int LeftBorderOffset   () { return pl_px + bl_px; }
    int TopMarginOffset    () { return mt_px + TopBorderOffset(); }
    int RightMarginOffset  () { return mr_px + RightBorderOffset(); }
    int BottomMarginOffset () { return mb_px + BottomBorderOffset(); }
    int LeftMarginOffset   () { return ml_px + LeftBorderOffset(); }
    point MarginPosition   () { return -point(LeftMarginOffset(), BottomMarginOffset()); }
    Border PaddingOffset   () { return Border(pt_px, pr_px, pb_px, pl_px); }
    Border BorderOffset    () { return Border(TopBorderOffset(), RightBorderOffset(), BottomBorderOffset(), LeftBorderOffset()); }
    Border MarginOffset    () { return Border(TopMarginOffset(), RightMarginOffset(), BottomMarginOffset(), LeftMarginOffset()); }
    int MarginWidth        () { return LeftMarginOffset() + RightMarginOffset(); }
    int MarginHeight       () { return TopMarginOffset () + BottomMarginOffset(); }
    int MarginBoxWidth     () { return width_auto ? 0 : (width_px + MarginWidth()); }
    int WidthAuto       (Flow *flow)        { return max(0, flow->container->w - MarginWidth()); }
    int MarginCenterAuto(Flow *flow, int w) { return max(0, flow->container->w - bl_px - pl_px - w - br_px - pr_px) / 2; }
    int MarginLeftAuto  (Flow *flow, int w) { return max(0, flow->container->w - bl_px - pl_px - w - RightMarginOffset()); } 
    int MarginRightAuto (Flow *flow, int w) { return max(0, flow->container->w - br_px - pr_px - w - LeftMarginOffset()); } 
}; }; // namespace DOM

struct Browser : public BrowserInterface {
    struct Document {
        DOM::BlockChainObjectAlloc alloc;
        DOM::HTMLDocument *node=0;
        string content_type, char_set;
        vector<StyleSheet*> style_sheet;
        DocumentParser *parser=0;
        JSContext *js_context=0;
        Console *js_console=0;
        int height=0;
        GUI gui;
        Widget::Scrollbar v_scrollbar, h_scrollbar;

        ~Document();
        Document(Window *W=0, const Box &V=Box());
        void Clear();
    };
    struct RenderLog { string data; int indent; };

    Layers layers;
    Document doc;
    RenderLog *render_log=0;
    Asset missing_image;
    point mouse, initial_displacement;
    Browser(Window *W=0, const Box &V=Box());

    Box Viewport() const { return doc.gui.box; }
    void Navigate(const string &url);
    void Open(const string &url);
    void KeyEvent(int key, bool down);
    void MouseMoved(int x, int y);
    void MouseButton(int b, bool d);
    void MouseWheel(int xs, int ys);
    void BackButton() {}
    void ForwardButton() {}
    void RefreshButton() {}
    void AnchorClicked(DOM::HTMLAnchorElement *anchor);
    void InitLayers() { layers.Init(2); }
    string GetURL() { return String::ToUTF8(doc.node->URL); }

    bool Dirty(Box *viewport);
    void Draw(Box *viewport);
    void Draw(Box *viewport, bool dirty);
    void Draw(Flow *flow, const point &displacement);
    void DrawScrollbar();

    void       DrawNode        (Flow *flow, DOM::Node*, const point &displacement);
    DOM::Node *LayoutNode      (Flow *flow, DOM::Node*, bool reflow);
    void       LayoutBackground(            DOM::Node*);
    void       LayoutTable     (Flow *flow, DOM::HTMLTableElement *n);
    void       UpdateTableStyle(Flow *flow, DOM::Node *n);
    void       UpdateRenderLog (            DOM::Node *n);

    static int ScreenToWebKitY(const Box &w) { return -w.y - w.h; }
};

BrowserInterface *CreateQTWebKitBrowser(Asset *a);
BrowserInterface *CreateBerkeliumBrowser(Asset *a, int w=1024, int h=1024);
BrowserInterface *CreateDefaultBrowser(Window *W, Asset *a, int w=1024, int h=1024);

struct HelperGUI : public GUI {
    HelperGUI(Window *W) : GUI(W), font(Fonts::Get(FLAGS_default_font, 9, Color::white)) {}
    Font *font;
    struct Hint { enum { UP, UPLEFT, UPRIGHT, DOWN, DOWNLEFT, DOWNRIGHT }; };
    struct Label {
        Box target, label; v2 target_center, label_center;
        int hint; string description;
        Label(const Box &w, const string &d, int h, Font *f, float ly, float lx) :
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
        void AssignLabelBox() { label.x = label_center.x - label.w/2; label.y = label_center.y - label.h/2; }
    };
    vector<Label> label;
    void AddLabel(const Box &w, const string &d, int h, float ly=.05, float lx=.02) { label.push_back(Label(w, d, h, font, ly, lx)); }
    void Activate() { active=1; /* ForceDirectedLayout(); */ }
    void ForceDirectedLayout();
    void Draw();
};

struct GChartsHTML {
    static string JSFooter() { return "}\ngoogle.setOnLoadCallback(drawVisualization);\n</script>\n"; }
    static string JSHeader() {
        return
            "<script type=\"text/javascript\" src=\"//www.google.com/jsapi\"></script>\n"
            "<script type=\"text/javascript\">\n"
            "google.load('visualization', '1', {packages: ['corechart']});\n"
            "function drawVisualization() {\n"
            "var data; var ac;\n";
    };
    static string JSAreaChart(const string &div_id, int width, int height,
                              const string &title, const string &vaxis_label, const string &haxis_label,
                              const vector<vector<string> > &table) {
        string ret = "data = google.visualization.arrayToDataTable([\n";
        for (int i = 0; i < table.size(); i++) {
            const vector<string> &l = table[i];
            ret += "[";
            for (int j = 0; j < l.size(); j++) StrAppend(&ret, l[j], ", ");
            ret += "],\n";
        };
        StrAppend(&ret, "]);\nac = new google.visualization.AreaChart(document.getElementById('", div_id, "'));\n");
        StrAppend(&ret, "ac.draw(data, {\ntitle : '", title, "',\n");
        StrAppend(&ret, "isStacked: true,\nwidth: ", width, ",\nheight: ", height, ",\n");
        StrAppend(&ret, "vAxis: {title: \"", vaxis_label, "\"},\nhAxis: {title: \"", haxis_label, "\"}\n");
        StrAppend(&ret, "});\n");
        return ret;
    }
    static string DivElement(const string &div_id, int width, int height) {
        return StrCat("<div id=\"", div_id, "\" style=\"width: ", width, "px; height: ", height, "px;\"></div>");
    }
};

struct DeltaSampler {
    typedef long long Value;
    struct Entry : public vector<Value> {
        void Assign(const vector<const Value*> &in) { resize(in.size()); for (int i=0; i<size(); i++) (*this)[i] = *in[i]; }
        void Subtract(const Entry &e) {      CHECK_EQ(size(), e.size()); for (int i=0; i<size(); i++) (*this)[i] -=  e[i]; }
        void Divide  (float v)        {                                  for (int i=0; i<size(); i++) (*this)[i] = static_cast<long long>((*this)[i] / v); }
    };
    Timer timer; int interval; vector<string> label; vector<const Value*> input; vector<Entry> data; Entry cur;
    DeltaSampler(int I, const vector<const Value*> &in, const vector<string> &l) : interval(I), label(l), input(in) { cur.Assign(input); }
    void Update() {
        if (timer.GetTime() < interval) return;
        float buckets = (float)timer.GetTime(true) / interval;
        if (fabs(buckets - 1.0) > .5) ERROR("buckets = ", buckets);
        Entry last = cur;
        cur.Assign(input);
        Entry diff = cur;
        diff.Subtract(last);
        data.push_back(diff);
    }
};

struct DeltaGrapher {
    static void JSTable(const DeltaSampler &sampler, vector<vector<string> > *out, int window_size) {
        vector<string> v; v.push_back("'Time'");
        for (int i=0; i<sampler.label.size(); i++) v.push_back(StrCat("'", sampler.label[i], "'"));
        out->push_back(v);

        v.clear();
        for (int j=0; j<sampler.label.size()+1; j++) v.push_back("0");
        out->push_back(v);

        for (int l=sampler.data.size(), i=max(0,l-window_size); i<l; i++) {
            const DeltaSampler::Entry &le = sampler.data[i];
            v.clear(); v.push_back(StrCat(i+1));
            for (int j=0; j<le.size(); j++) v.push_back(StrCat(le[j]));
            out->push_back(v);
        }
    }
};

}; // namespace LFL
#endif // __LFL_LFAPP_GUI_H__
