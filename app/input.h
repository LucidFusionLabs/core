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

#ifndef LFL_CORE_APP_INPUT_H__
#define LFL_CORE_APP_INPUT_H__
namespace LFL {

struct InputEvent {
  typedef long long Id;
  static int GetKey(Id event) { return event & 0xffffffff; }
  static const char *Name(Id event);
};

struct Key {
  typedef long long Mod;
  struct Modifier {
    static const InputEvent::Id Shift, Ctrl, Cmd; /// On PC Alt=Cmd
    struct ID { enum { Shift=1, Ctrl=2, Cmd=4 }; };
    static Mod FromID(int id) { return Mod(id) << 32; }
  };
  static const int Escape, Return, Up, Down, Left, Right, LeftShift, RightShift, LeftCtrl, RightCtrl, LeftCmd, RightCmd;
  static const int Tab, Space, Backspace, Delete, Quote, Backquote, PageUp, PageDown, Home, End, Insert;
  static const int F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12;
  static int CtrlModified(int k);
  static int ShiftModified(int k);
};

struct Mouse {
  struct Button { static const InputEvent::Id _1, _2; };
  struct Event  {
    static const InputEvent::Id Motion, Motion2, Click, Click2, DoubleClick, DoubleClick2, Wheel, Zoom, Swipe; 
  };
  static InputEvent::Id ButtonID(int button);
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

  ~Bind() { Destruct(); }
  Bind(InputEvent::Id K=0, Key::Mod M=0)               : cb_type(NONE),    key(K|M) {}
  Bind(InputEvent::Id K,             const CB     &Cb) : cb_type(CB_VOID), key(K|0) { new(&cb.cb_void)     CB(Cb); }
  Bind(InputEvent::Id K, Key::Mod M, const CB     &Cb) : cb_type(CB_VOID), key(K|M) { new(&cb.cb_void)     CB(Cb); }
  Bind(InputEvent::Id K,             const TimeCB &Cb) : cb_type(CB_TIME), key(K|0) { new(&cb.cb_time) TimeCB(Cb); }
  Bind(InputEvent::Id K, Key::Mod M, const TimeCB &Cb) : cb_type(CB_TIME), key(K|M) { new(&cb.cb_time) TimeCB(Cb); }
  Bind(const Bind &c) { Assign(c); }
  Bind &operator=(const Bind &c) { Destruct(); Assign(c); return *this; }
  bool operator<(const Bind &c) const { SortImpl1(key, c.key); }
  bool operator==(const Bind &c) const { return key == c.key; }

  void Destruct();
  void Assign(const Bind &c);
  void Run(unsigned t) const;
};

}; // namespace LFL
namespace std {
  template <> struct hash<LFL::Bind> {
    size_t operator()(const LFL::Bind &v) const { return ::std::hash<LFL::InputEvent::Id>()(v.key); }
  };
}; // namespace std;
namespace LFL {

struct KeyboardControllerInterface {
  virtual int SendKeyEvent(InputEvent::Id, bool down) = 0;
};

struct MouseControllerInterface {
  virtual int SendMouseEvent(InputEvent::Id, const point &p, const point &d, int down, int flag) = 0;
  virtual int SendWheelEvent(InputEvent::Id, const v2    &p, const v2    &d, bool begin)         = 0;
};

struct KeyboardController : public KeyboardControllerInterface {
  Bind toggle_bind;
  bool toggle_enabled=0, keydown_events_only=1;
  virtual ~KeyboardController() {}
  virtual void SetToggleKey(int TK) { toggle_bind.key=TK; toggle_enabled=true; }
  virtual int SendKeyEvent(InputEvent::Id, bool down) { return 0; }
};

struct TextboxController : public KeyboardController {
  Clipboard *clipboard;
  TextboxController(Clipboard *C=0) : clipboard(C) {}
  int HandleSpecialKey(InputEvent::Id event);
  virtual int SendKeyEvent(InputEvent::Id event, bool down);
  virtual void InputString(const string &s) { for (int i=0; i<s.size(); i++) Input(s[i]); }
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

struct MouseControllerCallback {
  typedef function<void()> CB;
  typedef function<bool()> BoolCB;
  typedef function<void(int, v2, v2, int)> ScaleCB;
  typedef function<void(int, point, point, int)> CoordCB;
  enum { NONE=0, CB_VOID=1, CB_BOOL=2, CB_COORD=3, CB_SCALE=4 } type;
  UNION FunctionPointer {
    CB      cb_void;
    BoolCB  cb_bool;
    CoordCB cb_coord;
    ScaleCB cb_scale;
    FunctionPointer() {}
    ~FunctionPointer() {}
  };

  FunctionPointer cb;
  ThreadDispatcher *run_from_message_loop=nullptr;

  ~MouseControllerCallback() { Destruct(); }
  MouseControllerCallback()                                          : type(NONE) {}
  MouseControllerCallback(const CB       &c, ThreadDispatcher *mt=0) : type(CB_VOID)  { new(&cb.cb_void)  CB     (c); run_from_message_loop=mt; }
  MouseControllerCallback(const BoolCB   &c, ThreadDispatcher *mt=0) : type(CB_BOOL)  { new(&cb.cb_bool)  BoolCB (c); run_from_message_loop=mt; }
  MouseControllerCallback(const CoordCB  &c, ThreadDispatcher *mt=0) : type(CB_COORD) { new(&cb.cb_coord) CoordCB(c); run_from_message_loop=mt; }
  MouseControllerCallback(const ScaleCB  &c, ThreadDispatcher *mt=0) : type(CB_SCALE) { new(&cb.cb_scale) ScaleCB(c); run_from_message_loop=mt; }
  MouseControllerCallback(const MouseControllerCallback &c) { Assign(c); }
  MouseControllerCallback &operator=(const MouseControllerCallback &c) { Destruct(); Assign(c); return *this; }

  void Destruct();
  void Assign(const MouseControllerCallback &c);
  template <class X> bool Run(X p, X d, int button, int down, bool wrote=false);
};

struct MouseController : public MouseControllerInterface {
  typedef MouseControllerCallback::CB CB;
  typedef MouseControllerCallback::BoolCB BoolCB;
  typedef MouseControllerCallback::CoordCB CoordCB;
  typedef MouseControllerCallback::ScaleCB ScaleCB;
  struct Event {
    enum { Click=1, RightClick=2, Hover=3, Drag=4, Wheel=5, Zoom=6 }; 
    static const char *Name(int e);
  };
  struct HitBox {
    Box box;
    int evtype, val=0;
    bool active=1, deleted=0, run_only_if_first=0;
    MouseControllerCallback CB;
    HitBox(int ET=0, const Box &b=Box(), MouseControllerCallback cb=MouseControllerCallback()) : box(b), evtype(ET), CB(move(cb)) {}
  };

  View *parent_view;
  IterableFreeListVector<HitBox, &HitBox::deleted> hit;
  unordered_set<int> drag;
  vector<int> hover;

  MouseController(View *V=0) : parent_view(V) {}
  virtual ~MouseController() { Clear(); }
  virtual void Clear() { hit.Clear(); }
  virtual int AddClickBox     (const Box &w, MouseControllerCallback cb) { return hit.Insert(HitBox(Event::Click,      w, move(cb))); }
  virtual int AddRightClickBox(const Box &w, MouseControllerCallback cb) { return hit.Insert(HitBox(Event::RightClick, w, move(cb))); }
  virtual int AddZoomBox      (const Box &w, MouseControllerCallback cb) { return hit.Insert(HitBox(Event::Zoom,       w, move(cb))); }
  virtual int AddWheelBox     (const Box &w, MouseControllerCallback cb) { return hit.Insert(HitBox(Event::Wheel,      w, move(cb))); }
  virtual int AddHoverBox     (const Box &w, MouseControllerCallback cb) { return hit.Insert(HitBox(Event::Hover,      w, move(cb))); }
  virtual int AddDragBox      (const Box &w, MouseControllerCallback cb) { return hit.Insert(HitBox(Event::Drag,       w, move(cb))); }
  virtual int SendMouseEvent(InputEvent::Id, const point &p, const point &d, int down, int flag);
  virtual int SendWheelEvent(InputEvent::Id, const v2    &p, const v2    &d, bool begin);
  template <class X> int SendEvent(InputEvent::Id, const X &p, const X &d, int down, int flag);
};

struct DragTracker {
  bool changing=0;
  point beg_click, end_click;
  Time beg_click_time;
  bool Update(const point &p, bool down);
};

struct Input : public Module {
  struct InputCB {
    enum { KeyPress=1, MouseClick=2, MouseMove=3, MouseWheel=4, MouseZoom=5, MouseSwipe=6 };
    int type;
    UNION Data {
      struct { int   x, y, a, b; } iv;
      struct { float x, y, a, b; } fv;
    } data;
    bool begin;
    InputCB(int T=0, int X=0, int Y=0, int A=0, int B=0, bool b=0) : type(T), begin(b) { data.iv.x=X; data.iv.y=Y; data.iv.a=A; data.iv.b=B; }
    InputCB(int T, float X, float Y, float A, float B, bool b, bool round) : type(T), begin(b) { data.fv.x=X; data.fv.y=Y; data.fv.a=A; data.fv.b=B; }
  };

  Clipboard *clipboard;
  WindowHolder *window;
  ThreadDispatcher *dispatcher;
  bool left_shift_down = 0, right_shift_down = 0, left_ctrl_down = 0, right_ctrl_down = 0;
  bool left_cmd_down = 0, right_cmd_down = 0, mouse_but1_down = 0, mouse_but2_down = 0;
  vector<InputCB> queued_input;
  mutex queued_input_mutex;
  Input(Clipboard *C, WindowHolder *W, ThreadDispatcher *D) : clipboard(C), window(W), dispatcher(D) {}

  void QueueKeyPress(int key, int mod, bool down);
  void QueueMouseClick(int button, bool down, const point &p);
  void QueueMouseMovement(const point &p, const point &d);
  void QueueMouseSwipe(const point &p, const point &d);
  void QueueMouseWheel(const v2 &p, const v2 &d);
  void QueueMouseZoom(const v2 &p, const v2 &d, bool begin);

  bool ShiftKeyDown() const { return left_shift_down || right_shift_down; }
  bool CtrlKeyDown() const { return left_ctrl_down || right_ctrl_down; }
  bool CmdKeyDown() const { return left_cmd_down || right_cmd_down; }
  bool MouseButton1Down() const { return mouse_but1_down; }
  bool MouseButton2Down() const { return mouse_but2_down; }
  void ClearButtonsDown();

  int Init() override;
  int DispatchQueuedInput(bool event_on_keyboard_input, bool event_on_mouse_input);

  int KeyPress(int key, int mod, bool down);
  int KeyEventDispatch(InputEvent::Id event, bool down);
  int KeyEventDispatchView(InputEvent::Id event, int down, View *v, int *active_guis);

  int MouseMove(const point &p, const point &d);
  int MouseSwipe(const point &p, const point &d);
  int MouseWheel(const v2 &p, const v2 &d);
  int MouseZoom(const v2 &p, const v2 &d, bool begin);
  int MouseClick(int button, bool down, const point &p);
  int MouseEventDispatch(InputEvent::Id event, const point &p, const point &d, int down);
  int MouseEventDispatchView(InputEvent::Id event, const point &p, const point &d, int down, View *v, int *active_guis);

  static point TransformMouseCoordinate(Window*, point p);
};

}; // namespace LFL
#endif // LFL_CORE_APP_INPUT_H__
