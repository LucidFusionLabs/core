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

#ifndef LFL_LFAPP_INPUT_H__
#define LFL_LFAPP_INPUT_H__
namespace LFL {

struct InputEvent {
  typedef long long Id;
  static int GetKey(Id event) { return event & 0xffffffff; }
  static const char *Name(Id event);
};

struct Key {
  typedef long long Mod;
  struct Modifier { static const InputEvent::Id Ctrl, Cmd; }; /// On PC Alt=Cmd
  static const int Escape, Return, Up, Down, Left, Right, LeftShift, RightShift, LeftCtrl, RightCtrl, LeftCmd, RightCmd;
  static const int Tab, Space, Backspace, Delete, Quote, Backquote, PageUp, PageDown, Home, End;
  static const int F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12;
  static int CtrlModified(int k);
  static int ShiftModified(int k);
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
  template <> struct hash<LFL::Bind> {
    size_t operator()(const LFL::Bind &v) const { return ::std::hash<LFL::InputEvent::Id>()(v.key); }
  };
}; // namespace std;
namespace LFL {

struct InputController {
  bool active=0;
  virtual ~InputController() {}
  virtual void Activate  () { active = 1; }
  virtual void Deactivate() { active = 0; }
  virtual void Input(InputEvent::Id event, bool down) {}
};

struct BindMap : public InputController {
  unordered_set<Bind> data, down;
  BindMap() { active = 1; }
  template <class... Args> void Add(Args&&... args) { AddBind(Bind(forward<Args>(args)...)); }
  void AddBind(const Bind &b) { data.insert(b); }
  void Repeat(unsigned clicks) { for (auto b : down) b.Run(clicks); }
  void Input(InputEvent::Id event, bool d) {
    auto b = data.find(event);
    if (b == data.end()) return;
    if (b->cb_type == Bind::CB_TIME) { Bind r=*b; r.key=InputEvent::GetKey(r.key); InsertOrErase(&down, r, d); }
    else if (d) b->Run(0);
  }
  string DebugString() const { string v="{ "; for (auto b : data) StrAppend(&v, b.key, " "); return v + "}"; }
};

struct KeyboardController {
  Bind toggle_bind;
  bool toggle_once=0;
  virtual ~KeyboardController() {}
  virtual int HandleSpecialKey(InputEvent::Id);
  virtual void SetToggleKey(int TK, bool TO=0) { toggle_bind.key=TK; toggle_once=TO; }
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

struct MouseControllerCallback {
  typedef function<void()> CB;
  typedef function<bool()> BoolCB;
  typedef function<void(int, int, int, int)> CoordCB;
  enum { NONE=0, CB_VOID=1, CB_BOOL=2, CB_COORD=3 } type;
  UNION FunctionPointer {
    CB      cb_void;
    BoolCB  cb_bool;
    CoordCB cb_coord;
    FunctionPointer() {}
    ~FunctionPointer() {}
  };

  FunctionPointer cb;
  bool run_from_message_loop=false;

  ~MouseControllerCallback() { Destruct(); }
  MouseControllerCallback()                                 : type(NONE) {}
  MouseControllerCallback(const CB       &c, bool mt=false) : type(CB_VOID)  { new (&cb.cb_void)  CB     (c); run_from_message_loop=mt; }
  MouseControllerCallback(const BoolCB   &c, bool mt=false) : type(CB_BOOL)  { new (&cb.cb_bool)  BoolCB (c); run_from_message_loop=mt; }
  MouseControllerCallback(const CoordCB  &c, bool mt=false) : type(CB_COORD) { new (&cb.cb_coord) CoordCB(c); run_from_message_loop=mt; }
  MouseControllerCallback(const MouseControllerCallback &c) { Assign(c); }
  MouseControllerCallback &operator=(const MouseControllerCallback &c) { Destruct(); Assign(c); return *this; }

  void Destruct();
  void Assign(const MouseControllerCallback &c);
  bool Run(const point &p, int button, int down, bool wrote=false);
};

struct MouseController {
  typedef MouseControllerCallback::CB CB;
  typedef MouseControllerCallback::BoolCB BoolCB;
  typedef MouseControllerCallback::CoordCB CoordCB;
  struct Event { enum { Click=1, Hover=2, Drag=3 }; };
  struct HitBox {
    Box box;
    int evtype, val=0;
    bool active=1, deleted=0, run_only_if_first=0;
    MouseControllerCallback CB;
    HitBox(int ET=0, const Box &b=Box(), const MouseControllerCallback &cb=MouseControllerCallback()) : box(b), evtype(ET), CB(cb) {}
  };

  IterableFreeListVector<HitBox, &HitBox::deleted> hit;
  unordered_set<int> drag;
  vector<int> hover;

  virtual ~MouseController() { Clear(); }
  virtual void Clear() { hit.Clear(); }
  virtual int AddClickBox(const Box &w, const MouseControllerCallback &cb) { return hit.Insert(HitBox(Event::Click, w, cb)); }
  virtual int AddHoverBox(const Box &w, const MouseControllerCallback &cb) { return hit.Insert(HitBox(Event::Hover, w, cb)); }
  virtual int AddDragBox (const Box &w, const MouseControllerCallback &cb) { return hit.Insert(HitBox(Event::Drag,  w, cb)); }
  virtual int Input(InputEvent::Id, const point &p, int down, int flag);
};

struct DragTracker {
  bool changing=0;
  point beg_click, end_click;

  bool Update(const point &p, bool down) {
    bool start = !changing && down;
    if (start) beg_click = p;
    end_click = p;
    changing = down;
    return start;
  }
};

struct InputModule : public Module {
  virtual int Init(Window *w) { return 0; }
};

struct Input : public InputModule {
  struct InputCB {
    enum { KeyPress=1, MouseClick=2, MouseMove=3, MouseWheel=4 };
    int type, x, y, a, b;
    InputCB(int T=0, int X=0, int Y=0, int A=0, int B=0) : type(T), x(X), y(Y), a(A), b(B) {}
  };

  bool left_shift_down = 0, right_shift_down = 0, left_ctrl_down = 0, right_ctrl_down = 0;
  bool left_cmd_down = 0, right_cmd_down = 0, mouse_but1_down = 0, mouse_but2_down = 0;
  vector<InputCB> queued_input;
  mutex queued_input_mutex;
  unique_ptr<InputModule> impl;
  Bind paste_bind;

  void QueueKeyPress(int key, bool down) {
    ScopedMutex sm(queued_input_mutex);
    queued_input.emplace_back(InputCB::KeyPress, 0, 0, key, down);
  }
  void QueueMouseClick(int button, bool down, const point &p) {
    ScopedMutex sm(queued_input_mutex);
    queued_input.emplace_back(InputCB::MouseClick, p.x, p.y, button, down);
  }
  void QueueMouseMovement(const point &p, const point &d) {
    ScopedMutex sm(queued_input_mutex);
    queued_input.emplace_back(InputCB::MouseMove, p.x, p.y, d.x, d.y);
  }
  void QueueMouseWheel(const point &p, const point &d) {
    ScopedMutex sm(queued_input_mutex);
    queued_input.emplace_back(InputCB::MouseWheel, p.x, p.y, d.x, d.y);
  }

  bool ShiftKeyDown() const { return left_shift_down || right_shift_down; }
  bool CtrlKeyDown() const { return left_ctrl_down || right_ctrl_down; }
  bool CmdKeyDown() const { return left_cmd_down || right_cmd_down; }
  bool MouseButton1Down() const { return mouse_but1_down; }
  bool MouseButton2Down() const { return mouse_but2_down; }
  void ClearButtonsDown();

  int Init();
  int Init(Window*);
  int Frame(unsigned time);
  int DispatchQueuedInput(bool event_on_keyboard_input, bool event_on_mouse_input);

  int KeyPress(int key, bool down);
  int KeyEventDispatch(InputEvent::Id event, bool down);

  int MouseMove(const point &p, const point &d);
  int MouseWheel(const point &p, const point &d);
  int MouseClick(int button, bool down, const point &p);
  int MouseEventDispatch(InputEvent::Id event, const point &p, int down);

  static point TransformMouseCoordinate(point p);
};

}; // namespace LFL
#endif // LFL_LFAPP_INPUT_H__
