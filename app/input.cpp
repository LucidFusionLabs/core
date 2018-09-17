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

#include "core/app/gl/view.h"

#ifdef LFL_DEBUG
#define InputDebug(...)         if (FLAGS_input_debug)           DebugPrintf(__VA_ARGS__);
#define InputDebugIfDown(...)   if (FLAGS_input_debug && down)   DebugPrintf(__VA_ARGS__);
#define InputDebugIfEvents(...) if (FLAGS_input_debug && events) DebugPrintf(__VA_ARGS__);
#else
#define InputDebug(...)
#define InputDebugIfDown(...)
#define InputDebugIfEvents(...)
#endif

namespace LFL {
DEFINE_float(ksens, 4, "Keyboard sensitivity");
DEFINE_float(msens, 1, "Mouse sensitivity");
DEFINE_int(invert, 1, "Invert mouse [1|-1]");
DEFINE_int(keyboard_repeat, 50, "Keyboard repeat in milliseconds");
DEFINE_int(keyboard_delay, 180, "Keyboard delay until repeat in milliseconds");
DEFINE_bool(input_debug, false, "Debug input events");

const InputEvent::Id Key::Modifier::Shift       = 1LL<<32;
const InputEvent::Id Key::Modifier::Ctrl        = 1LL<<33;
const InputEvent::Id Key::Modifier::Cmd         = 1LL<<34;
const InputEvent::Id Mouse::Button::_1          = 1LL<<35;
const InputEvent::Id Mouse::Button::_2          = 1LL<<36;
const InputEvent::Id MouseEvent                 = 1LL<<37;
const InputEvent::Id Mouse::Event::Motion       = MouseEvent+0;
const InputEvent::Id Mouse::Event::Motion2      = MouseEvent+1;
const InputEvent::Id Mouse::Event::Wheel        = MouseEvent+2;
const InputEvent::Id Mouse::Event::Zoom         = MouseEvent+3;
const InputEvent::Id Mouse::Event::Swipe        = MouseEvent+4;
const InputEvent::Id Mouse::Event::Click        = Mouse::Button::_1;
const InputEvent::Id Mouse::Event::Click2       = Mouse::Button::_2;
const InputEvent::Id Mouse::Event::DoubleClick  = Mouse::Button::_1+1;
const InputEvent::Id Mouse::Event::DoubleClick2 = Mouse::Button::_2+1;

const char *InputEvent::Name(InputEvent::Id event) {
  switch (event) {
    case Mouse::Event::Motion:       return "MouseMotion";
    case Mouse::Event::Motion2:      return "MouseMotion2";
    case Mouse::Event::Click:        return "Click";
    case Mouse::Event::Click2:       return "Click2";
    case Mouse::Event::DoubleClick:  return "DoubleClick";
    case Mouse::Event::DoubleClick2: return "DoubleClick2";
    case Mouse::Event::Wheel:        return "MouseWheel";
    case Mouse::Event::Zoom:         return "MouseZoom";
    case Mouse::Event::Swipe:        return "MouseSwipe";
    default:                         return "Unknown";
  }
}

int Key::CtrlModified(int k) {
  if (isalpha(k)) k = ::toupper(k);
  return (k >= 'A' && k <= '_') ? k - 0x40 : k;
}

int Key::ShiftModified(int k) {
  if (isalpha(k)) k = ::toupper(k);
  else switch(k) {
    case '\'': k='"'; break;
    case '\\': k='|'; break;
    case  '-': k='_'; break;
    case  ';': k=':'; break;
    case  ',': k='<'; break;
    case  '.': k='>'; break;
    case  '/': k='?'; break;
    case  '=': k='+'; break;
    case  '1': k='!'; break;
    case  '2': k='@'; break;
    case  '3': k='#'; break;
    case  '4': k='$'; break;
    case  '5': k='%'; break;
    case  '6': k='^'; break;
    case  '7': k='&'; break;
    case  '8': k='*'; break;
    case  '9': k='('; break;
    case  '0': k=')'; break;
    case  '[': k='{'; break;
    case  ']': k='}'; break;
    case  '`': k='~'; break;
  }
  return k;
}

#if 0
struct KeyRepeater {
  static const int repeat_keys=512;
  unordered_set<int> keys_down;
  bool key_down[repeat_keys], key_delay[repeat_keys];
  Time key_down_repeat[repeat_keys];
  KeyRepeater() { memzero(key_down); memzero(key_delay); memzero(key_down_repeat); }

  void KeyChange(int key, int down) {
    if (key < 0 || key >= repeat_keys) return;
    if      ( down && !key_down[key]) { keys_down.insert(key); key_down[key]=1; key_delay[key]=0; key_down_repeat[key]=Now(); }
    else if (!down &&  key_down[key]) { keys_down.erase (key); key_down[key]=0;                                               }
  }
  void Repeat(Input *input, unsigned clicks) {
    Time now = Now();
    for (auto i = keys_down.begin(); i != keys_down.end(); ++i) {
      int elapsed = now - key_down_repeat[*i], delay = key_delay[*i];
      if ((!delay && elapsed < FLAGS_keyboard_delay) ||
          ( delay && elapsed < FLAGS_keyboard_repeat)) continue;
      for (int j=0, max_repeat=10; elapsed >= FLAGS_keyboard_repeat; ++j) {
        if (!delay) { delay=1; key_delay[*i]=true; elapsed -= FLAGS_keyboard_delay; }
        else        {                              elapsed -= FLAGS_keyboard_repeat; }
        if (j < max_repeat) input->KeyEventDispatch(*i, true);
      }
      key_down_repeat[*i] = now - elapsed;
    }
  }
};
#endif

InputEvent::Id Mouse::ButtonID(int button) {
  switch (button) {
    case 1: return Button::_1;
    case 2: return Button::_2;
  } return 0;
}

void Bind::Destruct() {
  switch (cb_type) {
    case CB_VOID: cb.cb_void.~CB();     break;
    case CB_TIME: cb.cb_time.~TimeCB(); break;
    case NONE: break;
  }
}

void Bind::Assign(const Bind &c) {
  key = c.key;
  cb_type = c.cb_type;
  switch (cb_type) {
    case CB_VOID: new(&cb.cb_void)     CB(c.cb.cb_void); break;
    case CB_TIME: new(&cb.cb_time) TimeCB(c.cb.cb_time); break;
    case NONE: break;
  }
}

void Bind::Run(unsigned t) const {
  switch (cb_type) {
    case CB_VOID: cb.cb_void();  break;
    case CB_TIME: cb.cb_time(t); break;
    case NONE: break;
  }
}

void BindMap::Button(InputEvent::Id event, bool d) {
  auto b = data.find(event);
  if (b == data.end()) return;
  if (b->cb_type == Bind::CB_TIME) { Bind r=*b; r.key=InputEvent::GetKey(r.key); InsertOrErase(&down, r, d); }
  else if (d) b->Run(0);
}

int TextboxController::HandleSpecialKey(InputEvent::Id event) {
  if      (event == Key::Backspace) { Erase();       return 1; }
  else if (event == Key::Delete)    { Erase();       return 1; }
  else if (event == Key::Return)    { Enter();       return 1; }
  else if (event == Key::Left)      { CursorLeft();  return 1; }
  else if (event == Key::Right)     { CursorRight(); return 1; }
  else if (event == Key::Up)        { HistUp();      return 1; }
  else if (event == Key::Down)      { HistDown();    return 1; }
  else if (event == Key::PageUp)    { PageUp();      return 1; }
  else if (event == Key::PageDown)  { PageDown();    return 1; }
  else if (event == Key::Home)      { Home();        return 1; }
  else if (event == Key::End)       { End();         return 1; }
  else if (event == Key::Tab)       { Tab();         return 1; }
  else if (event == Key::Escape)    { Escape();      return 1; }
  return 0;
}

int TextboxController::SendKeyEvent(InputEvent::Id event, bool down) {
  int key = InputEvent::GetKey(event);
  bool shift_down = event & Key::Modifier::Shift, ctrl_down = event & Key::Modifier::Ctrl,
       cmd_down = event & Key::Modifier::Cmd;
  InputDebugIfDown("TextboxController::Input %s %d %d %d %d",
                   InputEvent::Name(event), key, shift_down, ctrl_down, cmd_down);

  if (toggle_bind.key == event && !toggle_once) return 0;

  if (clipboard && event == clipboard->paste_bind.key) { InputString(clipboard->GetClipboardText()); return 1; }
  if (HandleSpecialKey(event)) return 1;

  if (cmd_down) return 0;
  if (key >= 128) { InputDebug("TextboxController::Input unhandled key %lld", event); return 0; }

  if (shift_down) key = Key::ShiftModified(key);
  if (ctrl_down)  key = Key::CtrlModified(key);

  Input(key);
  return 1;
}

bool DragTracker::Update(const point &p, bool down) {
  bool start = !changing && down;
  if (start) beg_click = p;
  end_click = p;
  changing = down;
  return start;
}

void Input::QueueKeyPress(int key, int mod, bool down) {
  ScopedMutex sm(queued_input_mutex);
  queued_input.emplace_back(InputCB::KeyPress, key, mod, down, 0);
}

void Input::QueueMouseClick(int button, bool down, const point &p) {
  ScopedMutex sm(queued_input_mutex);
  queued_input.emplace_back(InputCB::MouseClick, p.x, p.y, button, down);
}

void Input::QueueMouseMovement(const point &p, const point &d) {
  ScopedMutex sm(queued_input_mutex);
  queued_input.emplace_back(InputCB::MouseMove, p.x, p.y, d.x, d.y);
}

void Input::QueueMouseSwipe(const point &p, const point &d) {
  ScopedMutex sm(queued_input_mutex);
  queued_input.emplace_back(InputCB::MouseSwipe, p.x, p.y, d.x, d.y);
}

void Input::QueueMouseWheel(const v2 &p, const v2 &d) {
  ScopedMutex sm(queued_input_mutex);
  queued_input.emplace_back(InputCB::MouseWheel, p.x, p.y, d.x, d.y);
}

void Input::QueueMouseZoom(const v2 &p, const v2 &d, bool begin) {
  ScopedMutex sm(queued_input_mutex);
  queued_input.emplace_back(InputCB::MouseZoom, p.x, p.y, d.x, d.y, begin, false);
}

point Input::TransformMouseCoordinate(Window *w, point p) {
  if (FLAGS_swap_axis) p = point(w->gl_w - p.y, p.x);
  return point(p.x, w->gl_h - p.y);
}

void Input::ClearButtonsDown() {
  Window *screen = window->focused;
  if (left_shift_down)  { KeyPress(Key::LeftShift,  0, 0); left_shift_down = 0; }
  if (right_shift_down) { KeyPress(Key::RightShift, 0, 0); left_shift_down = 0; }
  if (left_ctrl_down)   { KeyPress(Key::LeftCtrl,   0, 0); left_ctrl_down  = 0; }
  if (right_ctrl_down)  { KeyPress(Key::RightCtrl,  0, 0); right_ctrl_down = 0; }
  if (left_cmd_down)    { KeyPress(Key::LeftCmd,    0, 0); left_cmd_down   = 0; }
  if (right_cmd_down)   { KeyPress(Key::RightCmd,   0, 0); right_cmd_down  = 0; }
  if (mouse_but1_down)  { MouseClick(1, 0, screen->mouse); mouse_but1_down = 0; }
  if (mouse_but2_down)  { MouseClick(2, 0, screen->mouse); mouse_but2_down = 0; }
}

int Input::Init() {
  INFO("Input::Init()");
  return 0;
}

#if 0
int Input::GestureFrame(unsinged clicks) {
  if (screen->gesture_swipe_up)   { if (screen->console && screen->console->active) screen->console->PageUp();   }
  if (screen->gesture_swipe_down) { if (screen->console && screen->console->active) screen->console->PageDown(); }
  screen->gesture_swipe_up = screen->gesture_swipe_down = 0;
  screen->gesture_tap[0] = screen->gesture_tap[1] = screen->gesture_dpad_stop[0] = screen->gesture_dpad_stop[1] = 0;
  screen->gesture_dpad_dx[0] = screen->gesture_dpad_dx[1] = screen->gesture_dpad_dy[0] = screen->gesture_dpad_dy[1] = 0;
}
#endif

int Input::DispatchQueuedInput(bool event_on_keyboard_input, bool event_on_mouse_input) {
  vector<InputCB> icb;
  {
    ScopedMutex sm(queued_input_mutex);
    swap(icb, queued_input);
  }
  int events = 0, v = 0;
  for (auto &i : icb)
    switch (i.type) { 
      case InputCB::KeyPress:   v = KeyPress  (i.data.iv.x, i.data.iv.y, i.data.iv.a);                               if (event_on_keyboard_input) events += v; break;
      case InputCB::MouseClick: v = MouseClick(i.data.iv.a, i.data.iv.b, point(i.data.iv.x, i.data.iv.y));           if (event_on_mouse_input)    events += v; break;
      case InputCB::MouseMove:  v = MouseMove (point(i.data.iv.x, i.data.iv.y), point(i.data.iv.a, i.data.iv.b));    if (event_on_mouse_input)    events += v; break;
      case InputCB::MouseSwipe: v = MouseSwipe(point(i.data.iv.x, i.data.iv.y), point(i.data.iv.a, i.data.iv.b));    if (event_on_mouse_input)    events += v; break;
      case InputCB::MouseWheel: v = MouseWheel(v2(i.data.fv.x, i.data.fv.y), v2(i.data.fv.a, i.data.fv.b));          if (event_on_mouse_input)    events += v; break;
      case InputCB::MouseZoom:  v = MouseZoom (v2(i.data.fv.x, i.data.fv.y), v2(i.data.fv.a, i.data.fv.b), i.begin); if (event_on_mouse_input)    events += v; break;
    }
  return events;
}

int Input::KeyPress(int key, int mod, bool down) {
  if (!dispatcher->run) return 0;
#ifdef LFL_DEBUG
  if (!dispatcher->MainThread()) ERROR("Input::KeyPress() called from thread ", Thread::GetId());
#endif

  if      (key == Key::LeftShift)   left_shift_down = down;
  else if (key == Key::RightShift) right_shift_down = down;
  else if (key == Key::LeftCtrl)     left_ctrl_down = down;
  else if (key == Key::RightCtrl)   right_ctrl_down = down;
  else if (key == Key::LeftCmd)       left_cmd_down = down;
  else if (key == Key::RightCmd)     right_cmd_down = down;

  InputEvent::Id event = key;
  if (mod) event |= Key::Modifier::FromID(mod);
  else {
    if (CtrlKeyDown ()) event |= Key::Modifier::Ctrl;
    if (CmdKeyDown  ()) event |= Key::Modifier::Cmd;
    if (ShiftKeyDown()) event |= Key::Modifier::Shift;
  }

  int fired = KeyEventDispatch(event, down);
  if (fired) return fired;

  Window *screen = window->focused;
  for (auto &g : screen->input)
    if (g->active) g->Button(event, down); 

  return 0;
}

int Input::KeyEventDispatch(InputEvent::Id event, bool down) {
  KeyboardController *g = window->focused ? window->focused->active_textbox : 0;
  if (!g || (!down && g->keydown_events_only)) return 0;
  return g->SendKeyEvent(event, down);
}

int Input::MouseMove(const point &p, const point &d) {
  if (!dispatcher->run) return 0;
  Window *screen = window->focused;
  int fired = MouseEventDispatch(Mouse::Event::Motion, p, d, MouseButton1Down());
  if (!screen->grab_mode.Enabled()) return fired;

  for (auto &g : screen->input)
    if (g->active) g->Move(Mouse::Event::Motion, p, d); 

  return fired;
}

int Input::MouseSwipe(const point &p, const point &d) {
  int events = 0;
  if (!dispatcher->run || !window->focused) return events;
  if (auto mc = window->focused->active_controller) {
    if ((events = mc->SendWheelEvent(Mouse::Event::Swipe, v2(p.x, p.y), v2(d.x, d.y), 0))) { 
      InputDebug("Input::MouseWheel sent MouseController[%p] events = %d", mc, events);
      return events;
    }
  }
  return events;
}

int Input::MouseWheel(const v2 &p, const v2 &d) {
  int events = 0;
  if (!dispatcher->run || !window->focused) return events;
  if (auto mc = window->focused->active_controller) {
    if ((events = mc->SendWheelEvent(Mouse::Event::Wheel, p, d, 0))) { 
      InputDebug("Input::MouseWheel sent MouseController[%p] events = %d", mc, events);
      return events;
    }
  }
  return events;
}

int Input::MouseZoom(const v2 &p, const v2 &d, bool begin) {
  int events = 0;
  if (!dispatcher->run || !window->focused) return events;
  if (auto mc = window->focused->active_controller) {
    if ((events = mc->SendWheelEvent(Mouse::Event::Zoom, p, d, begin))) { 
      InputDebug("Input::MouseZoom sent MouseController[%p] events = %d", mc, events);
      return events;
    }
  }
  return events;
}

int Input::MouseClick(int button, bool down, const point &p) {
  if (!dispatcher->run) return 0;
  InputEvent::Id event = Mouse::ButtonID(button);
  if      (event == Mouse::Button::_1) mouse_but1_down = down;
  else if (event == Mouse::Button::_2) mouse_but2_down = down;
  // event |= (CtrlKeyDown() ? Key::Modifier::Ctrl : 0) | (CmdKeyDown() ? Key::Modifier::Cmd : 0);

  int fired = MouseEventDispatch(event, p, point(), down);
  if (fired) return fired;

  Window *screen = window->focused;
  for (auto i = screen->input.begin(), e = screen->input.end(); i != e; ++i)
    if ((*i)->active) (*i)->Button(event, down);

  return fired;
}

int Input::MouseEventDispatch(InputEvent::Id event, const point &pp, const point &d, int down) {
  Window *w = window->focused;
  if      (event == clipboard->paste_bind.key) return KeyEventDispatch(event, down);
  else if (event == Mouse::Event::Wheel) w->mouse_wheel = pp;
  else                                   w->mouse       = pp;
  InputDebug("Input::MouseEventDispatch %s %s down=%d",
             InputEvent::Name(event), w->mouse.DebugString().c_str(), down);

  int fired = 0, active_guis = 0, events;
  Dialog *bring_to_front = 0;
  for (auto i = w->dialogs.begin(); i != w->dialogs.end(); /**/) {
    Dialog *g = i->get();
    point p = g->RelativePosition(pp);
    if (g->NotActive(p)) { i++; continue; }
    fired += g->mouse.SendMouseEvent(event, p, d, down, 0);
    if (g->deleted) { w->GiveDialogFocusAway(g); i = w->dialogs.erase(i); continue; }
    if (event == Mouse::Event::Click && down && g->box.within(w->mouse)) { bring_to_front = g; break; }
    i++;
  }
  if (bring_to_front) w->BringDialogToFront(bring_to_front);

  if (auto mc = w->active_controller) {
    if ((events = mc->SendMouseEvent(event, mc->parent_view ? mc->parent_view->RelativePosition(pp) : pp, d, down, 0))) {
      InputDebug("Input::MouseEventDispatch sent MouseController[%p] %s events = %d", mc, InputEvent::Name(event), events);
      return events;
    }
  } else for (auto b = w->view.begin(), e = w->view.end(), i = b; i != e; ++i) {
    if ((events = MouseEventDispatchView(event, pp - w->Box().TopLeft(), d, down, *i, &active_guis))) {
      InputDebug("Input::MouseEventDispatch sent View[%d] %s events = %d", i - b, InputEvent::Name(event), events);
      return events;
    }
  }

  InputDebugIfDown("Input::MouseEventDispatch %s fired=%d, guis=%d/%zd",
                   w->mouse.DebugString().c_str(), fired, active_guis, w->view.size());
  return fired;
}

int Input::MouseEventDispatchView(InputEvent::Id event, const point &pp, const point &d, int down, View *v, int *active_views) {
  point p = v->RelativePosition(pp);
  if (v->NotActive(p)) return 0;
  else (*active_views)++;
  int events = v->mouse.SendMouseEvent(event, p, d, down, 0);
  InputDebugIfDown("Input::MouseEventDispatchView %s fired=%d, view=%s", p.DebugString().c_str(), events, v->name);
  for (auto i=v->child_view.begin(), e=v->child_view.end(); i != e; ++i)
    events += MouseEventDispatchView(event, p, d, down, *i, active_views);
  return events;
}

void MouseControllerCallback::Destruct() {
  switch(type) {
    case CB_VOID:  cb.cb_void .~CB();      break;
    case CB_BOOL:  cb.cb_bool .~BoolCB();  break;
    case CB_COORD: cb.cb_coord.~CoordCB(); break;
    case CB_SCALE: cb.cb_scale.~ScaleCB(); break;
    default:                               break;
  }
}

void MouseControllerCallback::Assign(const MouseControllerCallback &c) {
  run_from_message_loop = c.run_from_message_loop;
  switch ((type = c.type)) {
    case CB_VOID:  new(&cb.cb_void)  CB     (c.cb.cb_void);  break;
    case CB_BOOL:  new(&cb.cb_bool)  BoolCB (c.cb.cb_bool);  break;
    case CB_COORD: new(&cb.cb_coord) CoordCB(c.cb.cb_coord); break;
    case CB_SCALE: new(&cb.cb_scale) ScaleCB(c.cb.cb_scale); break;
    default:                                                 break;
  }
}

template <class X> bool MouseControllerCallback::Run(X p, X d, int button, int down, bool wrote) {
  if (run_from_message_loop && !wrote) { run_from_message_loop->RunInMainThread([=]{ Run(p, d, button, down, true); }); return false; }
  bool ret = 1;
  switch (type) {
    case CB_VOID:  cb.cb_void();                                                              break;
    case CB_BOOL:  ret = cb.cb_bool();                                                        break;
    case CB_COORD: cb.cb_coord(button, V2Type<X>::GetPoint(p), V2Type<X>::GetPoint(d), down); break;
    case CB_SCALE: cb.cb_scale(button, V2Type<X>::GetV2   (p), V2Type<X>::GetV2   (d), down); break;
    default:                                                                                  break;
  }
  return ret;
}

template bool MouseControllerCallback::Run(v2    p, v2    d, int button, int down, bool wrote);
template bool MouseControllerCallback::Run(point p, point d, int button, int down, bool wrote);

const char *MouseController::Event::Name(int e) {
  switch(e) {
    case Click:      return "Click";
    case RightClick: return "RightClick";
    case Hover:      return "Hover";
    case Drag:       return "Drag";
    case Wheel:      return "Wheel";
    case Zoom:       return "Zoom";
    default:         return "";
  }
}

int MouseController::SendMouseEvent(InputEvent::Id event, const point &p, const point &dp, int down, int flag) {
  return SendEvent(event, p, dp, down, flag);
}

int MouseController::SendWheelEvent(InputEvent::Id event, const v2 &p, const v2 &d, bool begin) {
  return SendEvent(event, p, d, 1, 0);
}

template <class X> int MouseController::SendEvent(InputEvent::Id event, const X &p, const X &dp, int down, int flag) {
  int fired = 0, boxes_checked = 0;
  bool but1  = event == Mouse::Event::Click;
  bool but2  = event == Mouse::Event::Click2;
  bool wheel = event == Mouse::Event::Wheel;
  bool zoom  = event == Mouse::Event::Zoom;
  point pp = V2Type<X>::GetPoint(p);

  for (auto h = hover.begin(); h != hover.end(); /**/) {
    auto e = &hit.data[*h];
    if (!e->deleted && e->box.within(pp)) { ++h; continue; }
    h = VectorEraseIterSwapBack(&hover, h);
    if (e->deleted) continue;
    e->val = 0;
    e->CB.Run(p, dp, event, 0);
    fired++;
  }

  for (auto e = hit.data.rbegin(), ee = hit.data.rend(); e != ee; ++e) {
    bool thunk = 0, e_hover = e->evtype == Event::Hover;

    if (e->deleted || !e->active || (e_hover && e->val) || 
        (!down && (e->evtype == Event::Click || e->evtype == Event::RightClick) &&
         e->CB.type != MouseControllerCallback::CB_COORD && e->CB.type != MouseControllerCallback::CB_SCALE)) continue;

    InputDebug("%s check %s within %s (%s)", InputEvent::Name(event), p.DebugString().c_str(),
               e->box.DebugString().c_str(), Event::Name(e->evtype));
    boxes_checked++;
    if (e->box.within(pp)) {
      if (e->run_only_if_first && fired) continue;
      switch (e->evtype) { 
        case Event::Click:      if (but1)         { thunk=1; } break;
        case Event::RightClick: if (but2)         { thunk=1; } break;
        case Event::Drag:       if (but1 && down) { thunk=1; } break;
        case Event::Hover:      if ((e->val = 1)) { thunk=1; } break;
        case Event::Wheel:      if (wheel)        { thunk=1; } break;
        case Event::Zoom:       if (zoom)         { thunk=1; } break;
      }
    }

    if (thunk) {
      InputDebugIfDown("MouseController::Input %s RunCB %s",
                       p.DebugString().c_str(), e->box.DebugString().c_str());
      if (!e->CB.Run(p, dp, event, e_hover ? 1 : down)) continue;

      if (e_hover) hover.push_back(ForwardIteratorFromReverse(e) - hit.data.begin());
      fired++;

      if (e->evtype == Event::Drag && down) drag.insert(ForwardIteratorFromReverse(e) - hit.data.begin());
      if (flag) break;
    }
  }

  if (event == Mouse::Event::Motion) { for (auto d : drag) if (hit.data[d].CB.Run(p, dp, event, down)) fired++; }
  else if (!down && but1)            { for (auto d : drag) if (hit.data[d].CB.Run(p, dp, event, down)) fired++; drag.clear(); }

  InputDebugIfDown("MouseController::Input %s fired=%d, checked %zd of %zd hitboxes",
                   p.DebugString().c_str(), fired, boxes_checked, hit.data.size());
  return fired;
}

}; // namespace LFL
