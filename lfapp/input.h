/*
 * $Id: input.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef __LFL_LFAPP_INPUT_H__
#define __LFL_LFAPP_INPUT_H__
namespace LFL {

struct InputEvent {
    typedef long long Id;
    static int GetKey(Id event) { return event & 0xffffffff; }
};

struct Key {
    typedef long long Mod;
    struct Modifier { static const InputEvent::Id Ctrl, Cmd; }; /// On PC Alt=Cmd
    static const int Escape, Return, Up, Down, Left, Right, LeftShift, RightShift, LeftCtrl, RightCtrl, LeftCmd, RightCmd;
    static const int Tab, Space, Backspace, Delete, Quote, Backquote, PageUp, PageDown, Home, End;
    static const int F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12;
};

struct Mouse {
    struct Button { static const InputEvent::Id _1, _2; };
    struct Event  { static const InputEvent::Id Motion, Wheel, Button1, Button2; };
    static InputEvent::Id ButtonID(int button) {
        switch (button) {
            case 1: return Button::_1;
            case 2: return Button::_2;
        } return 0;
    }
    static void GrabFocus();
    static void ReleaseFocus();
};

struct InputController {
    struct Events { int total; };
    Events events;
    bool active=0;
    InputController() { ClearEvents(); screen->input_bind.push_back(this); }
    virtual ~InputController() { VectorEraseByValue(&screen->input_bind, this); }

    void ClearEvents() { memzero(events); }
    virtual void Activate  () { active = 1; }
    virtual void Deactivate() { active = 0; }
    virtual void Input(InputEvent::Id event, bool down) {}
};

struct KeyboardController {
    struct Events { int total; };
    Events events;
    bool active=0;
    KeyboardController() { ClearEvents(); }

    void ClearEvents() { memzero(events); }
    virtual void Activate  () { active = 1; }
    virtual void Deactivate() { active = 0; }
    virtual void Input(const string &s) { for (int i=0; i<s.size(); i++) Input(s[i]); }
    virtual void Input(char key) {}
    virtual void Enter      () {}
    virtual void Erase      () {}
    virtual void CursorLeft () {}
    virtual void CursorRight() {}
    virtual void PageUp     () {}
    virtual void PageDown   () {}
    virtual void Home       () {}
    virtual void End        () {}
    virtual void HistUp     () {}
    virtual void HistDown   () {}
    virtual void Tab        () {}
    virtual void Escape     () {}
};

struct MouseController {
    typedef function<void()> CB;
    typedef function<bool()> BoolCB;
    typedef function<void(int, int, int, int)> CoordCB;

    struct Event { enum { Click=1, Hover=2, Drag=3 }; };
    struct Events { int total, click, hover; };

    struct Callback {
        enum { NONE=0, CB_VOID=1, CB_BOOL=2, CB_COORD=3 } type;
        UNION FunctionPointer {
            CB      cb_void;
            BoolCB  cb_bool;
            CoordCB cb_coord;
            FunctionPointer() {}
            ~FunctionPointer() {}
        } cb;
        ~Callback() { Destruct(); }
        Callback()                  : type(NONE) {}
        Callback(const CB       &c) : type(CB_VOID)  { new (&cb.cb_void)  CB     (c); }
        Callback(const BoolCB   &c) : type(CB_BOOL)  { new (&cb.cb_bool)  BoolCB (c); }
        Callback(const CoordCB  &c) : type(CB_COORD) { new (&cb.cb_coord) CoordCB(c); }
        Callback(const Callback &c) { Assign(c); }
        Callback &operator=(const Callback &c) { Destruct(); Assign(c); return *this; }
        void Destruct() {
            switch(type) {
                case CB_VOID:  cb.cb_void .~CB();      break;
                case CB_BOOL:  cb.cb_bool .~BoolCB();  break;
                case CB_COORD: cb.cb_coord.~CoordCB(); break;
                default:                               break;
            }
        }
        void Assign(const Callback &c) {
            switch ((type = c.type)) {
                case CB_VOID:  new (&cb.cb_void)  CB     (c.cb.cb_void);  break;
                case CB_BOOL:  new (&cb.cb_bool)  BoolCB (c.cb.cb_bool);  break;
                case CB_COORD: new (&cb.cb_coord) CoordCB(c.cb.cb_coord); break;
                default:                                                  break;
            }
        }
        bool Run(const point &p, int button, int down) {
            bool ret = 1;
            switch (type) {
                case CB_VOID:  cb.cb_void();                        break;
                case CB_BOOL:  ret = cb.cb_bool();                  break;
                case CB_COORD: cb.cb_coord(button, p.x, p.y, down); break;
                default:                                            break;
            } return ret;
        }
    };

    struct HitBox {
        Box box;
        int evtype, val=0;
        bool active=1, deleted=0, run_only_if_first=0;
        Callback CB;
        HitBox(int ET=0, const Box &b=Box(), const Callback &cb=Callback()) : box(b), evtype(ET), CB(cb) {}
    };

    IterableFreeListVector<HitBox> hit;
    unordered_set<int> drag;
    Events events;
    bool active=0;
    virtual ~MouseController() { Clear(); }

    virtual void Clear() { hit.Clear(); ClearEvents(); }
    virtual void ClearEvents() { memzero(events); }
    virtual void Activate() { active = 1; }
    virtual void Deactivate() { active = 0; }
    virtual bool NotActive() const { return !active; }
    virtual int AddClickBox(const Box &w, const Callback &cb) { return hit.Insert(HitBox(Event::Click, w, cb)); }
    virtual int AddHoverBox(const Box &w, const Callback &cb) { return hit.Insert(HitBox(Event::Hover, w, cb)); }
    virtual int AddDragBox (const Box &w, const Callback &cb) { return hit.Insert(HitBox(Event::Drag,  w, cb)); }
    virtual int Input(InputEvent::Id, const point &p, int down, int flag);
};

struct Input : public Module {
    bool left_shift_down=0, right_shift_down=0, left_ctrl_down=0, right_ctrl_down=0;
    bool left_cmd_down=0, right_cmd_down=0, mouse_but1_down=0, mouse_but2_down=0;
    vector<Callback> queued_input;
    mutex queued_input_mutex;
    Module *impl=0;

    void QueueKey(int key, bool down) {
        ScopedMutex sm(queued_input_mutex);
        queued_input.push_back(bind([&](){ KeyPress(key, down); }));
    }
    void QueueMouseClick(int button, bool down, const point &p) {
        ScopedMutex sm(queued_input_mutex);
        queued_input.push_back(bind([&](){ MouseClick(button, down, p); }));
    }
    void QueueMouseMovement(const point &p, const point &d) {
        ScopedMutex sm(queued_input_mutex);
        queued_input.push_back(bind([&](){ MouseMove(p, d); }));
    }
    void QueueMouseWheel(int dw) {
        ScopedMutex sm(queued_input_mutex);
        queued_input.push_back(bind([&](){ MouseWheel(dw); }));
    }
    
    bool ShiftKeyDown() const { return left_shift_down || right_shift_down; }
    bool CtrlKeyDown() const { return left_ctrl_down || right_ctrl_down; }
    bool CmdKeyDown() const { return left_cmd_down || right_cmd_down; }
    bool MouseButton1Down() const { return mouse_but1_down; }
    bool MouseButton2Down() const { return mouse_but2_down; }

    int Init();
    int Frame(unsigned time);
    int DispatchQueuedInput();

    int  KeyPress(int key, bool down);
    int  KeyEventDispatch(InputEvent::Id event, bool down);

    int  MouseMove(const point &p, const point &d);
    int  MouseWheel(int dw);
    int  MouseClick(int button, bool down, const point &p);
    int  MouseEventDispatch(InputEvent::Id event, const point &p, int down);

    static point TransformMouseCoordinate(point p) {
        if (FLAGS_swap_axis) p = point(screen->width - p.y, p.x);
        return point(p.x, screen->height - p.y);
    }
};

struct Bind {
    typedef function<void()> CB;
    typedef function<void(unsigned)> TimeCB;
    enum { NONE=0, CB_VOID=1, CB_TIME=2 } cb_type;
    UNION FunctionPointer {
        CB     cb_void; 
        TimeCB cb_time; 
        FunctionPointer() {}
        ~FunctionPointer() {}
    } cb;
    InputEvent::Id key;
    Bind(InputEvent::Id K=0, Key::Mod M=0)               : cb_type(NONE),    key(K|M) {}
    Bind(InputEvent::Id K,             const CB     &Cb) : cb_type(CB_VOID), key(K|0) { new (&cb.cb_void)     CB(Cb); }
    Bind(InputEvent::Id K, Key::Mod M, const CB     &Cb) : cb_type(CB_VOID), key(K|M) { new (&cb.cb_void)     CB(Cb); }
    Bind(InputEvent::Id K,             const TimeCB &Cb) : cb_type(CB_TIME), key(K|0) { new (&cb.cb_time) TimeCB(Cb); }
    Bind(InputEvent::Id K, Key::Mod M, const TimeCB &Cb) : cb_type(CB_TIME), key(K|M) { new (&cb.cb_time) TimeCB(Cb); }
    Bind(const Bind &c) { Assign(c); }
    Bind &operator=(const Bind &c) { Destruct(); Assign(c); return *this; }
    bool operator<(const Bind &c) const { SortImpl1(key, c.key); }
    bool operator==(const Bind &c) const { return key == c.key; }

    void Destruct() {
        switch (cb_type) {
            case CB_VOID: cb.cb_void.~CB();     break;
            case CB_TIME: cb.cb_time.~TimeCB(); break;
            case NONE: break;
        }
    }
    void Assign(const Bind &c) {
        key = c.key;
        cb_type = c.cb_type;
        switch (cb_type) {
            case CB_VOID: new (&cb.cb_void)     CB(c.cb.cb_void); break;
            case CB_TIME: new (&cb.cb_time) TimeCB(c.cb.cb_time); break;
            case NONE: break;
        }
    }
    void Run(unsigned t) const {
        switch (cb_type) {
            case CB_VOID: cb.cb_void();  break;
            case CB_TIME: cb.cb_time(t); break;
            case NONE: break;
        }
    }
};

}; // namespace LFL
namespace std {
    template <> struct ::std::hash<LFL::Bind> {
        size_t operator()(const LFL::Bind &v) const { return ::std::hash<LFL::InputEvent::Id>()(v.key); }
    };
}; // namespace std;
namespace LFL {

struct BindMap : public InputController {
    unordered_set<Bind> data, down;
    BindMap() { active = 1; }
    void Add(const Bind &b) { data.insert(b); }
    void Repeat(unsigned clicks) { for (auto b : down) b.Run(clicks); }
    void Input(InputEvent::Id event, bool d) {
        auto b = data.find(event);
        if (b == data.end()) return;
        if (b->cb_type == Bind::CB_TIME) { Bind r=*b; r.key=InputEvent::GetKey(r.key); InsertOrErase(&down, r, d); }
        else if (d) b->Run(0);
    }
    string DebugString() const { string v="{ "; for (auto b : data) StrAppend(&v, b.key, " "); return v + "}"; }
};

struct Shell {
    typedef function<void(const vector<string>&)> CB;
    struct Command { 
        string name; CB cb;
        Command(const string &N, const CB &Cb) : name(N), cb(Cb) {}
    };
    vector<Command> command;
    AssetMap       *assets;
    SoundAssetMap  *soundassets;
    MovieAssetMap  *movieassets;
    Shell(AssetMap *AM=0, SoundAssetMap *SAM=0, MovieAssetMap *MAM=0);

    Asset      *asset     (const string &n);
    SoundAsset *soundasset(const string &n);
    MovieAsset *movieasset(const string &n);

    bool FGets();
    void Run(const string &text);

    void quit(const vector<string>&);
    void mousein(const vector<string>&);
    void mouseout(const vector<string>&);
    void console(const vector<string>&);
    void consolecolor(const vector<string>&);
    void showkeyboard(const vector<string>&);
    void clipboard(const vector<string>&);
    void startcmd(const vector<string>&);
    void dldir(const vector<string>&);
    void screenshot(const vector<string>&);

    void fillmode(const vector<string>&);
    void grabmode(const vector<string>&);
    void texmode (const vector<string>&);
    void swapaxis(const vector<string>&);
    void campos(const vector<string>&);
    void play     (const vector<string>&);
    void playmovie(const vector<string>&);
    void loadsound(const vector<string>&);
    void loadmovie(const vector<string>&);
    void copy(const vector<string>&);
    void snap(const vector<string>&);
    void filter   (const vector<string>&);
    void fftfilter(const vector<string>&);
    void f0(const vector<string>&);
    void sinth(const vector<string>&);
    void writesnap(const vector<string>&);
    void fps(const vector<string>&);
    void wget(const vector<string>&);
    void MessageBox(const vector<string>&);
    void TextureBox(const vector<string>&);
    void Slider    (const vector<string>&);
    void Edit      (const vector<string>&);

    void cmds (const vector<string>&);
    void flags(const vector<string>&);
    void binds(const vector<string>&);
};

}; // namespace LFL
#endif // __LFL_LFAPP_INPUT_H__
