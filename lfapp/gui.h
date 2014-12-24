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

struct GUI {
    Box box;
    Window *parent;
    BoxArray child_box;
    MouseController mouse;
    bool display=0; ToggleBool toggleDisplay;
    GUI()                        :         parent(0), toggleDisplay(&display) {}
    GUI(Window *W)               :         parent(W), toggleDisplay(&display) { parent->mouse_gui.push_back(this); }
    GUI(Window *W, const Box &B) : box(B), parent(W), toggleDisplay(&display) { parent->mouse_gui.push_back(this); }
    virtual ~GUI() { if (parent) VectorEraseByValue(&parent->mouse_gui, this); }

    point MousePosition() const { return screen->mouse - box.TopLeft(); }
    BoxArray *Reset() { Clear(); return &child_box; }
    void Clear() { child_box.Clear(); mouse.Clear(); }
    void UpdateBox(const Box &b, int draw_box_ind, int input_box_ind) {
        if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box = b;
        if (input_box_ind >= 0) mouse.hit     [input_box_ind].box = b;
    }
    void UpdateBoxX(int x, int draw_box_ind, int input_box_ind) {
        if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.x = x;
        if (input_box_ind >= 0) mouse.hit     [input_box_ind].box.x = x;
    }
    void UpdateBoxY(int y, int draw_box_ind, int input_box_ind) {
        if (draw_box_ind  >= 0) child_box.data[draw_box_ind ].box.y = y;
        if (input_box_ind >= 0) mouse.hit     [input_box_ind].box.y = y;
    }

    virtual void SetLayoutDirty() { child_box.Clear(); }
    virtual void Layout() {}
    virtual void Draw(const Box &b) { box=b; Draw(); }
    virtual void Draw() {
        mouse.Activate();
        if (child_box.data.empty()) Layout();
        child_box.Draw(box.TopLeft());
    }
    virtual void HandleTextMessage(const string &s) {}

    virtual void EnableDisplay() { display = true; }
    virtual bool ToggleDisplay() {
        bool ret = toggleDisplay.Toggle();
        display ? ToggleDisplayOn() : ToggleDisplayOff();
        return ret;
    }
    virtual void ToggleDisplayOn() { display=1; }
    virtual void ToggleDisplayOff() { display=0; }
    virtual void ToggleConsole() { if (!display) app->shell.console(vector<string>()); }
};

struct KeyboardGUI : public KeyboardController {
    typedef function<void(const string &text)> RunCB;
    RunCB runcb;
    string startcmd;
    RingVector<string> lastcmd;
    int lastcmd_ind=-1;
    Bind toggle_bind;
    ToggleBool toggle_active;
    Window *parent;
    KeyboardGUI(Window *W, Font *F, int TK=0, int TM=ToggleBool::Default, int LastCommands=50)
        : toggle_active(&active, TM), lastcmd(LastCommands), parent(W)
    { parent->keyboard_gui.push_back(this); toggle_bind.key=TK; }
    virtual ~KeyboardGUI() { if (parent) VectorEraseByValue(&parent->keyboard_gui, this); }
    virtual void Enable() { active = true; }
    virtual bool Toggle() { return toggle_active.Toggle(); }
    virtual void Run(string cmd) { if (runcb) runcb(cmd); }

    void AddHistory(string cmd);
    int WriteHistory(const char *dir, const char *name);
    int ReadHistory(const char *dir, const char *name);
};

struct TextGUI : public KeyboardGUI {
    struct Link;
    struct Lines;
    struct LineData {
        BoxArray glyphs;
        String16 text, text_attr;
        vector<shared_ptr<Link> > links;
    };
    struct Line {
        point p;
        TextGUI *parent;
        shared_ptr<LineData> data;
        Line(TextGUI *P=0) : parent(P), data(new LineData()) {}
        Line &operator=(const Line &s) { data=s.data; return *this; }
        static void Move (Line &t, Line &s) { swap(t.data, s.data); }
        static void MoveP(Line &t, Line &s) { swap(t.data, s.data); t.p=s.p; }
#if 0
        int Size () const { return glyphs.Size(); }
        int Lines() const { return max(1, glyphs.line.size()); }
        string Text() const { return glyphs.Text(); }
        void Clear() { glyphs.Clear(); }
        void Erase(int o, unsigned l=UINT_MAX) { return glyphs.Erase(o, l); }
        void InsertTextAt(int o, const string   &s, int a=-1) { glyphs.InsertAt(o,                  EncodeText(s, a)); }
        void InsertTextAt(int o, const String16 &s, int a=-1) { glyphs.InsertAt(o,                  EncodeText(s, a)); }
        void AppendText  (       const string   &s, int a=-1) { glyphs.InsertAt(glyphs.data.size(), EncodeText(s, a)); }
        void AppendText  (       const String16 &s, int a=-1) { glyphs.InsertAt(glyphs.data.size(), EncodeText(s, a)); }
        void AssignText  (       const string   &s, int a=-1) { glyphs.Clear(); AppendText(s, a); }
        void AssignText  (       const String16 &s, int a=-1) { glyphs.Clear(); AppendText(s, a); }
        vector<Drawable::Box> EncodeText(const string   &s, int a=-1) { BoxArray b; parent->font->Encode(s, Box(), &b, 0, a); return b.data; }
        vector<Drawable::Box> EncodeText(const String16 &s, int a=-1) { BoxArray b; parent->font->Encode(s, Box(), &b, 0, a); return b.data; }
        void UpdateAttr(int ind, int len, int a) {}
        void Layout(Box win) {}
#else
        int Size () const { return data->glyphs.Size(); }
        int Lines() const { return max(1, data->glyphs.line.size()); }
        string Text() const { return String::ToUTF8(data->text); }
        void Clear() { data->text.clear(); data->text_attr.clear(); data->glyphs.Clear(); }
        void Erase(int o, unsigned long long l=String16::npos) { if (data->text.size() > o) { data->text.erase(o, l);  data->text_attr.erase(o, l); } }
        void InsertTextAt(int o, const string   &s, int a=0) { String16 v=String::ToUTF16(s); data->text.insert(o, v); data->text_attr.insert(o, String16(v.size(), a)); }
        void InsertTextAt(int o, const String16 &s, int a=0) {                                data->text.insert(o, s); data->text_attr.insert(o, String16(s.size(), a)); }
        void AppendText  (       const string   &s, int a=0) { String16 v=String::ToUTF16(s); data->text.append(v);    data->text_attr.append(   String16(v.size(), a)); }
        void AppendText  (       const String16 &s, int a=0) {                                data->text.append(s);    data->text_attr.append(   String16(s.size(), a)); }
        void AssignText  (       const string   &s, int a=0) { data->text = String::ToUTF16(s); data->text_attr = String16(data->text.size(), a); }
        void AssignText  (       const String16 &s, int a=0) { data->text = s;                  data->text_attr = String16(data->text.size(), a); }
        void UpdateAttr(int ind, int len, int a) { Vec<short>::assign((short*)data->text_attr.data() + ind, a, len); }
        void Layout(Box win) {
            data->glyphs.Clear();
            Flow flow(&win, parent->font, &data->glyphs, &parent->layout);
            for (ArraySegmentIter<short> iter(data->text_attr); !iter.Done(); iter.Increment()) {
                if (parent->clickable_links) {}
                flow.AppendText(String16Piece(&data->text[iter.cur_start], iter.Length()), iter.cur_attr);
            }
            flow.Complete();
            CHECK_EQ(data->text.size(), data->glyphs.Size());
        }
#endif
        void UpdateText(int x, const String16 &v, int attr, int max_width=0) {
            if (Size() < x) AppendText(string(x - Size(), ' '), attr);
            if (!parent->insert_mode) Erase       (x, v.size());
            if (1)                    InsertTextAt(x, v, attr);
            if (parent->insert_mode && max_width) Erase(max_width);
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
        void Layout(int width=0) { return Layout(Box(0,0,width,0)); }
        point Draw(point pos, bool relayout, int relayout_width) {
            if (relayout) Layout(relayout_width);
            data->glyphs.Draw((p = pos));
            return p - point(0, max(parent->font->height, data->glyphs.height));
        }
    };
    struct Lines : public RingVector<Line> {
        function<void(Line&, Line&)> move_cb, movep_cb;
        Lines(TextGUI *P, int ML=200) : RingVector<Line>(ML),
        move_cb (bind(&Line::Move,  _1, _2)), 
        movep_cb(bind(&Line::MoveP, _1, _2)) { for (auto &i : data) i.parent = P; }

        Line *PushFront() { Line *l = RingVector<Line>::PushFront(); l->Clear(); return l; }
        Line *InsertAt(int dest_line, int lines=1, int dont_move_last=0) {
            CHECK(lines); CHECK_LT(dest_line, 0);
            int clear_dir = 1;
            if (dest_line == -1) { ring.PushBack(lines); clear_dir = -1; }
            else Move(*this, dest_line+lines, dest_line, -dest_line-lines-dont_move_last, move_cb);
            for (int i=0; i<lines; i++) (*this)[dest_line + i*clear_dir].Clear();
            return &(*this)[dest_line];
        }
    };
    struct LinesFrameBuffer : public RingFrameBuffer {
        typedef function<point(Line*, point, const Box&)> PaintCB;
        typedef function<LinesFrameBuffer*(const Line*)> FromLineCB;
        int lines=0;
        PaintCB paint_cb;
        LinesFrameBuffer() : paint_cb(bind(&LinesFrameBuffer::Paint, this, _1, _2, _3)) {}

        int Height() const { return lines * font_height; }
        bool SizeChanged(int W, int H, Font *font) {
            lines = H / font->height;
            return RingFrameBuffer::SizeChanged(W, H, font);
        }
        LinesFrameBuffer *Attach(LinesFrameBuffer **last_fb) {
            if (*last_fb != this) fb.Attach();
            return *last_fb = this;
        }
        void Clear(Line *l, bool vwrap=true) {
            RingFrameBuffer::Clear(l, Box(0, l->Lines() * font_height), vwrap);
        }
        void Update(Line *l, const point &p, bool lo=true, bool vw=true) { l->p=p; Update(l, lo, vw); }
        void Update(Line *l, bool layout=true, bool vwrap=true) {
            if (layout) l->Layout(wrap ? w : 0);
            RingFrameBuffer::Update(l, Box(0, l->Lines() * font_height), paint_cb, vwrap);
        }
        int PushFrontAndUpdate(Line *l, int max_lines=0, bool vwrap=true, bool reverse=false) {
            l->Layout(wrap ? w : 0);
            int lh = (max_lines ? min(max_lines, l->Lines()) : l->Lines()) * font_height;
            Box b(0, !reverse ? (l->Lines() * font_height - lh) : 0, 0, lh);
            return RingFrameBuffer::PushFrontAndUpdate(l, b, paint_cb, vwrap) / font_height;
        }
        int PushBackAndUpdate(Line *l, int max_lines=0, bool vwrap=true, bool reverse=false) {
            l->Layout(wrap ? w : 0);
            int lh = (max_lines ? min(max_lines, l->Lines()) : l->Lines()) * font_height;
            Box b(0, reverse ? (l->Lines() * font_height - lh) : 0, 0, lh);
            return RingFrameBuffer::PushBackAndUpdate(l, b, paint_cb, vwrap) / font_height;
        }
        void PushFrontAndUpdateOffset(Line *l, int lo, bool vwrap=true) {
            Update(l, RingFrameBuffer::BackPlus(point(0, (1 + lo) * font_height)));
            RingFrameBuffer::AdvancePixels(-l->Lines() * font_height);
        }
        void PushBackAndUpdateOffset(Line *l, int lo, bool vwrap=true) {
            Update(l, RingFrameBuffer::BackPlus(point(0, lo * font_height)));
            RingFrameBuffer::AdvancePixels(l->Lines() * font_height);
        }
        point Paint(Line *l, point lp, const Box &b) {
            Scissor scissor(0, lp.y - b.h, w, b.h);
            screen->gd->Clear();
            point ret = l->Draw(lp + b.Position(), 0, 0);
            return ret;
        }
    };
    struct LineUpdate {
        enum { PushBack=1, PushFront=2, DontUpdate=4 }; 
        Line *v; LinesFrameBuffer *fb; int flag, o;
        LineUpdate(Line *V=0, LinesFrameBuffer *FB=0, int F=0, int O=0) : v(V), fb(FB), flag(F), o(O) {}
        LineUpdate(Line *V, const LinesFrameBuffer::FromLineCB &cb, int F=0, int O=0) : v(V), fb(cb(V)), flag(F), o(O) {}
        ~LineUpdate() {
            if (!fb->lines || (flag & DontUpdate)) v->Layout();
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

    Font *font;
    Flow::Layout layout;
    Cursor cursor;
    Line cmd_line;
    LinesFrameBuffer cmd_fb;
    string cmd_prefix="> ";
    Color cmd_color=Color::white;
    bool deactivate_on_enter=0, clickable_links=0, insert_mode=0;
    int scroll_inc=100, scrolled_lines=0, adjust_lines=0, skip_last_lines=0, start_line=0;
    TextGUI(Window *W, Font *F, int TK=0, int TM=ToggleBool::Default) : KeyboardGUI(W, F, TK, TM), font(F), cmd_line(this) {}

    virtual ~TextGUI() {}
    virtual int CommandLines() const { return 0; }
    virtual void Input(char k) { cmd_line.UpdateText(cursor.i.x++, String16(1, k), cursor.attr); UpdateCommandFB(); UpdateCursor(); }
    virtual void Erase()       { if (!cursor.i.x) return;       cmd_line.Erase(--cursor.i.x, 1); UpdateCommandFB(); UpdateCursor(); }
    virtual void CursorRight() { cursor.i.x = min(cursor.i.x+1, cmd_line.Size()); UpdateCursor(); }
    virtual void CursorLeft()  { cursor.i.x = max(cursor.i.x-1, 0);               UpdateCursor(); }
    virtual void Home()        { cursor.i.x = 0;                                  UpdateCursor(); }
    virtual void End()         { cursor.i.x = cmd_line.Size();                    UpdateCursor(); }
    virtual void HistUp()      { if (int c=lastcmd.ring.count) { AssignInput(lastcmd[lastcmd_ind]); lastcmd_ind=max(lastcmd_ind-1, -c); cursor.i.x = cmd_line.Size(); } }
    virtual void HistDown()    { if (int c=lastcmd.ring.count) { AssignInput(lastcmd[lastcmd_ind]); lastcmd_ind=min(lastcmd_ind+1, -1); cursor.i.x = cmd_line.Size(); } }
    virtual void Enter();

    virtual string Text() const { return cmd_line.Text(); }
    virtual void AssignInput(const string &text) { cmd_line.AssignText(text); UpdateCommandFB(); UpdateCursor(); }
    virtual void UpdateCursor() {
        bool right = cursor.i.x >= cmd_line.data->glyphs.Size();
        const Box &b = !cmd_line.data->glyphs.data.size() ? Box() :
            cmd_line.data->glyphs.data[right ? cmd_line.data->glyphs.data.size()-1 : cursor.i.x].box;
        cursor.p = right ? b.TopRight() : b.TopLeft();
    }
    virtual void UpdateCommandFB() { 
        cmd_fb.fb.Attach();
        ScopedDrawMode drawmode(DrawMode::_2D);
        cmd_fb.PushBackAndUpdate(&cmd_line); // cmd_fb.OverwriteUpdate(&cmd_line, cursor.x)
        cmd_fb.fb.Release();
    }
    virtual void Draw(const Box &b);
    virtual void DrawCursor(point p);
};

struct Widget {
    struct Interface {
        GUI *gui;
        int drawbox_ind=1;
        vector<int> hitbox;
        bool del_hitbox=0;
        virtual ~Interface() { if (del_hitbox) DelHitBox(); }
        Interface(GUI *g) : gui(g) {}

        void AddClickBox(const Box &w, const MouseController::Callback &cb) { hitbox.push_back(gui->mouse.hit.Insert(MouseController::HitBox(MouseController::Event::Click, w, cb))); }
        void AddHoverBox(const Box &w, const MouseController::Callback &cb) { hitbox.push_back(gui->mouse.hit.Insert(MouseController::HitBox(MouseController::Event::Hover, w, cb))); }
        void DelHitBox() { for (vector<int>::const_iterator i = hitbox.begin(); i != hitbox.end(); ++i) gui->mouse.hit.Erase(*i); hitbox.clear(); }
        MouseController::HitBox &GetHitBox(int i=0) const { return gui->mouse.hit[hitbox[i]]; }
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
        void Visit() { SystemBrowser::open(link.c_str()); }

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
        float scrolled=0, increment=20;
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
                if (flip) scrolled = clamp(    (float)(gui->MousePosition().x - win.x) / win.w, 0, 1);
                else      scrolled = clamp(1 - (float)(gui->MousePosition().y - win.y) / win.h, 0, 1);
            }
            if (flip) gui->UpdateBoxX(win.x          + (int)((win.w - aw) * scrolled), drawbox_ind, IndexOrDefault(hitbox, 0, -1));
            else      gui->UpdateBoxY(win.top() - ah - (int)((win.h - ah) * scrolled), drawbox_ind, IndexOrDefault(hitbox, 0, -1));
            dirty = false;
        }
        void SetDocHeight(int v) { doc_height = v; }
        void ScrollUp  () { scrolled -= increment / doc_height; clamp(&scrolled, 0, 1); dirty=true; }
        void ScrollDown() { scrolled += increment / doc_height; clamp(&scrolled, 0, 1); dirty=true; }
        void DragScrollDot() { dragging = true; dirty = true; }
    };
};

struct TextArea : public TextGUI {
    struct Link {
        Widget::Button widget;
        DOM::Attr image_src;
        DOM::HTMLImageElement image;
        Link(GUI *G, const string &url) : widget(G, 0, 0, "", MouseController::CB(bind(&Widget::Button::Visit, &widget))),
        image_src(0), image(0) {
            widget.link = url;
            widget.del_hitbox = true;
            widget.EnableHover();
        }
    };
    typedef function<void(Link*)> LinkCB;

    Lines line;
    LinesFrameBuffer line_fb;
    GUI mouse_gui;
    Time write_last=0;
    const Border *clip=0;
    bool wrap_lines=1, write_timestamp=0, write_newline=1;
    bool selection_changing=0, selection_changing_previously=0;
    point selection_beg, selection_end;
    LinkCB new_link_cb, hover_link_cb;

    TextArea(Window *W, Font *F, int TK=0, int TM=ToggleBool::Default) : TextGUI(W, F, TK, TM), line(this), mouse_gui(W) {}
    virtual ~TextArea() {}

    /// Write() is thread-safe.
    virtual void Write(const string &s, bool update_fb=true, bool release_fb=true);
    virtual void PageUp();
    virtual void PageDown();
    virtual void Resized(int w, int h);

    virtual void Draw(const Box &w, bool cursor);
    // void UpdateLineOffsets(bool size_changed, bool cursor);
    // void UpdateWrapping(int width, bool new_lines_only);
    // void DrawOrCopySelection();

    void ClickCB(int button, int x, int y, int down) {
        if (1)    selection_changing = down;
        if (down) selection_beg = point(x, y);
        else      selection_end = point(x, y);
    }
};

struct Editor : public TextArea {
    struct LineOffset { 
        int offset, size, font_size, wrapped_line_number; float width; 
        LineOffset(int O=0, int S=0, int FS=0, int WLN=0, float W=0) :
            offset(O), size(S), font_size(FS), wrapped_line_number(WLN), width(W) {}
        bool operator<(const LineOffset &l) const { return wrapped_line_number < l.wrapped_line_number; }
    };
    typedef pair<int, int> LineOffsetSegment, WrappedLineOffset;
    WrappedLineOffset first_line, last_line;
    int last_fb_lines=0, wrapped_lines=0;
    float last_v_scrolled=0;
    shared_ptr<File> file;
    vector<LineOffset> file_line;
    Editor(Window *W, Font *F, File *I) : TextArea(W, F), file(I) { /*line_fb.wrap=1;*/ BuildLineMap(); }

    bool Wrap() const { return line_fb.wrap; }
    void BuildLineMap() {
        int ind=0, offset=0;
        for (const char *l = file->nextlineraw(&offset); l; l = file->nextlineraw(&offset))
            file_line.push_back(LineOffset(offset, file->nr.record_len, TextArea::font->size,
                                           ind++, TextArea::font->Width(l)));
    }
    void UpdateWrappedLines(int cur_font_size, int box_width) {
        wrapped_lines = 0;
        for (auto &l : file_line) {
            wrapped_lines += 1 + l.width * cur_font_size / l.font_size / box_width;
            l.wrapped_line_number = wrapped_lines;
        }
    }
    void GetWrappedLineOffset(float percent, WrappedLineOffset *out) const {
        if (!Wrap()) { *out = WrappedLineOffset(percent * file_line.size(), 0); return; }
        int wrapped_line = percent * wrapped_lines;
        auto it = lower_bound(file_line.begin(), file_line.end(), LineOffset(0,0,0,wrapped_line));
        if (it == file_line.end()) { *out = WrappedLineOffset(file_line.size(), 0); return; }
        int wrapped_line_index = it - file_line.begin();
        *out = WrappedLineOffset(wrapped_line_index, wrapped_line - wrapped_line_index - 1);
    }
    int Distance(const WrappedLineOffset &o, bool reverse) {
        int dist = 0; // for (int i=a_first_line; i<=b_first_line; i++) dist += 
        return abs(o.first - first_line.first);
    }
    void UpdateLines(float v_scrolled, float h_scrolled);
    void Draw(const Box &box, float v_scrolled, float h_scrolled) {
        TextArea::Draw(box, true);
        UpdateLines(v_scrolled, h_scrolled);
    }
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

    Terminal(int FD, Window *W, Font *F, int TK=0, int TM=ToggleBool::Default) :
        TextArea(W, F, TK, TM), fd(FD), fb_cb(bind(&Terminal::GetFrameBuffer, this, _1)) {
        CHECK(F->fixed_width || (F->flag & FontDesc::Mono));
        wrap_lines = write_newline = 0;
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
    }
    void Scroll(int sy, int ey, int dy, bool move_fb_p) {
        CHECK_LT(sy, ey);
        int line_ind = GetTermLineIndex(sy), scroll_lines = ey - sy + 1, ady = abs(dy), sdy = (dy > 0 ? 1 : -1);
        Move(line, line_ind + (dy>0 ? dy : 0), line_ind + (dy<0 ? -dy : 0), scroll_lines - ady, move_fb_p ? line.movep_cb : line.move_cb);
        for (int i = 0, cy = (dy>0 ? sy : ey); i < ady; i++) GetTermLine(cy + i*sdy)->Clear();
    }
    void FlushParseText();
    void Newline(bool carriage_return=false);
};

struct Console : public TextArea {
    double screenPercent=.4;
    Color color=Color(25,60,130,120);
    int animTime=333; Time animBegin=0;
    bool animating=0, drawing=0, bottom_or_top=0, blend=1;
    Console(Window *W, Font *F) : TextArea(W,F, Key::Backquote) { write_timestamp = 1; }

    virtual ~Console() {}
    virtual int CommandLines() const { return cmd_line.Lines(); }
    virtual void Run(string in) { app->shell.Run(in); }
    virtual bool Toggle() {
        if (!TextGUI::Toggle()) return false;
        Time elapsed = Now()-animBegin;
        animBegin = Now() - (elapsed<animTime ? animTime-elapsed : 0);
        return true;
    }
    virtual void Draw() {
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
};

struct Dialog : public GUI {
    Font *font=0;
    Color color=Color(25,60,130,220);
    Box title, resize_left, resize_right, resize_bottom, close;
    bool deleted=0, moving=0, resizing_left=0, resizing_right=0, resizing_top=0, resizing_bottom=0;
    point mouse_start, win_start;
    int zsort=0;
    Dialog(float w, float h) : GUI(screen), font(Fonts::Get(FLAGS_default_font, 14, Color::white)) {
        screen->dialogs.push_back(this);
        box = screen->Box().center(screen->Box(w, h));
        Layout();
    }
    virtual ~Dialog() {}
    virtual void Draw();
    virtual void Layout() {
        Reset();
        title         = Box(0,       0,      box.w, screen->height*.05);
        resize_left   = Box(0,       -box.h, 3,     box.h);
        resize_right  = Box(box.w-3, -box.h, 3,     box.h);
        resize_bottom = Box(0,       -box.h, box.w, 3);

        Box close = Box(box.w-10, title.top()-10, 10, 10);
        mouse.AddClickBox(resize_left,   MouseController::CB(bind(&Dialog::Reshape,     this, &resizing_left)));
        mouse.AddClickBox(resize_right,  MouseController::CB(bind(&Dialog::Reshape,     this, &resizing_right)));
        mouse.AddClickBox(resize_bottom, MouseController::CB(bind(&Dialog::Reshape,     this, &resizing_bottom)));
        mouse.AddClickBox(title,         MouseController::CB(bind(&Dialog::Reshape,     this, &moving)));
        mouse.AddClickBox(close,         MouseController::CB(bind(&Dialog::MarkDeleted, this)));
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
    Editor editor;
    Widget::Scrollbar v_scrollbar, h_scrollbar;
    EditorDialog(Window *W, Font *F, File *I) : Dialog(.5, .5), editor(W, F, I),
    v_scrollbar(this), h_scrollbar(this, Box(), Widget::Scrollbar::Flag::AttachedHorizontal) {}

    void Layout() {
        Dialog::Layout();
        if (1)              v_scrollbar.LayoutAttached(box.Dimension());
        if (!editor.Wrap()) h_scrollbar.LayoutAttached(box.Dimension());
    }
    void Draw() {
        Dialog::Draw();
        editor.Draw(box, v_scrollbar.scrolled, h_scrollbar.scrolled);
        GUI::Draw();
        if (1)              v_scrollbar.Update();
        if (!editor.Wrap()) h_scrollbar.Update();
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
    void InputActivate() { style.node->ownerDocument->gui->mouse.Activate(); }
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

struct SimpleBrowser : public Browser {
    GUI gui;
    Font *font;
    Layers layers;
    point initial_displacement;
    Widget::Scrollbar v_scrollbar, h_scrollbar;
    set<void*> outstanding;
    int requested=0, completed=0, mx=0, my=0;
    DOM::BlockChainObjectAlloc alloc;
    struct Document {
        DOM::HTMLDocument *node;
        string content_type, char_set;
        vector<StyleSheet*> style_sheet;
        int height;
        Document() : node(0), height(0) {}
    } doc;
    JSContext *js_context=0;
    Console *js_console=0;
    Asset missing_image;
    unordered_map<string, Asset*> image_cache;
    struct RenderLog { string data; int indent; } *render_log=0;

    SimpleBrowser()                          :            font(0), v_scrollbar(&gui, Box()),         h_scrollbar(&gui, Box(),         Widget::Scrollbar::Flag::AttachedHorizontal), alloc(1024*1024) { Construct(); Clear(); }
    SimpleBrowser(Window *W, Font *F, Box V) : gui(W, V), font(F), v_scrollbar(&gui, Box(V.w, V.h)), h_scrollbar(&gui, Box(V.w, V.h), Widget::Scrollbar::Flag::AttachedHorizontal), alloc(1024*1024) { Construct(); Clear(); }
    ~SimpleBrowser() { Clear(); delete js_context; } 

    void Construct() { if (Font *maf = Fonts::Get("MenuAtlas1", 0, Color::black, 0)) { missing_image.tex = maf->glyph->table[12].tex; missing_image.tex.width = missing_image.tex.height = 16; } }
    bool Running(void *h) { return outstanding.find(h) != outstanding.end(); }
    Box Viewport() const { return gui.box; }
    void Clear();
    void Navigate(const string &url);
    void OpenHTML(const string &content);
    void Open(const string &url) { Open(url, (DOM::Frame*)NULL); }
    void Open(const string &url, DOM::Frame *frame);
    void Open(const string &url, DOM::Node *target);
    Asset *OpenImage(const string &url);
    void OpenStyleImport(const string &url);
    void KeyEvent(int key, bool down);
    void MouseMoved(int x, int y);
    void MouseButton(int b, bool d);
    void MouseWheel(int xs, int ys);
    void BackButton() {}
    void ForwardButton() {}
    void RefreshButton() {}
    void AnchorClicked(DOM::HTMLAnchorElement *anchor);
    void InitLayers() { layers.Init(2); gui.mouse.dont_deactivate=1; }
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

Browser *CreateQTWebKitBrowser(Asset *a);
Browser *CreateBerkeliumBrowser(Asset *a, int w=1024, int h=1024);
Browser *CreateDefaultBrowser(Window *W, Asset *a, int w=1024, int h=1024);

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
    void ToggleDisplayOn() { display=1; /* ForceDirectedLayout(); */ }
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
        if (timer.time() < interval) return;
        float buckets = (float)timer.time(true) / interval;
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
