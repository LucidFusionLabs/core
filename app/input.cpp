/*
 * $Id: input.cpp 1328 2014-11-04 09:35:46Z justin $
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

const InputEvent::Id Key::Modifier::Shift  = 1LL<<32;
const InputEvent::Id Key::Modifier::Ctrl   = 1LL<<33;
const InputEvent::Id Key::Modifier::Cmd    = 1LL<<34;
const InputEvent::Id Mouse::Button::_1     = 1LL<<35;
const InputEvent::Id Mouse::Button::_2     = 1LL<<36;
const InputEvent::Id MouseEvent            = 1LL<<37;
const InputEvent::Id Mouse::Event::Motion  = MouseEvent+0;
const InputEvent::Id Mouse::Event::Wheel   = MouseEvent+1;
const InputEvent::Id Mouse::Event::Button1 = Mouse::Button::_1;
const InputEvent::Id Mouse::Event::Button2 = Mouse::Button::_2;

const char *InputEvent::Name(InputEvent::Id event) {
  switch (event) {
    case Mouse::Event::Motion:  return "MouseMotion";
    case Mouse::Event::Wheel:   return "MouseWheel";
    case Mouse::Event::Button1: return "MouseButton1";
    case Mouse::Event::Button2: return "MouseButton2";
    default:                    return "Unknown";
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
  void Repeat(unsigned clicks) {
    Time now = Now();
    for (auto i = keys_down.begin(); i != keys_down.end(); ++i) {
      int elapsed = now - key_down_repeat[*i], delay = key_delay[*i];
      if ((!delay && elapsed < FLAGS_keyboard_delay) ||
          ( delay && elapsed < FLAGS_keyboard_repeat)) continue;
      for (int j=0, max_repeat=10; elapsed >= FLAGS_keyboard_repeat; ++j) {
        if (!delay) { delay=1; key_delay[*i]=true; elapsed -= FLAGS_keyboard_delay; }
        else        {                              elapsed -= FLAGS_keyboard_repeat; }
        if (j < max_repeat) app->input->KeyEventDispatch(*i, true);
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
    case CB_VOID: new (&cb.cb_void)     CB(c.cb.cb_void); break;
    case CB_TIME: new (&cb.cb_time) TimeCB(c.cb.cb_time); break;
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

void Input::QueueMouseWheel(const point &p, const point &d) {
  ScopedMutex sm(queued_input_mutex);
  queued_input.emplace_back(InputCB::MouseWheel, p.x, p.y, d.x, d.y);
}

point Input::TransformMouseCoordinate(point p) {
  if (FLAGS_swap_axis) p = point(screen->width - p.y, p.x);
  return point(p.x, screen->height - p.y);
}

void Input::ClearButtonsDown() {
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
#ifdef LFL_APPLE
  paste_bind = Bind('v', Key::Modifier::Cmd);
#else
  paste_bind = Bind('v', Key::Modifier::Ctrl);
#endif
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
      case InputCB::KeyPress:   v = KeyPress  (i.x, i.y, i.a);                    if (event_on_keyboard_input) events += v; break;
      case InputCB::MouseClick: v = MouseClick(i.a, i.b, point(i.x, i.y));        if (event_on_mouse_input)    events += v; break;
      case InputCB::MouseMove:  v = MouseMove (point(i.x, i.y), point(i.a, i.b)); if (event_on_mouse_input)    events += v; break;
      case InputCB::MouseWheel: v = MouseMove (point(i.x, i.y), point(i.a, i.b)); if (event_on_mouse_input)    events += v; break;
    }
  return events;
}

int Input::KeyPress(int key, int mod, bool down) {
  if (!app->run) return 0;
#ifdef LFL_DEBUG
  if (!app->MainThread()) ERROR("Input::KeyPress() called from thread ", Thread::GetId());
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

  for (auto &g : screen->input)
    if (g->active) g->Button(event, down); 

  return 0;
}

int Input::KeyEventDispatch(InputEvent::Id event, bool down) {
  if (!down) return 0;
  int key = InputEvent::GetKey(event);
  bool shift_down = event & Key::Modifier::Shift, ctrl_down = event & Key::Modifier::Ctrl,
       cmd_down = event & Key::Modifier::Cmd;
  InputDebugIfDown("Input::KeyEventDispatch %s %d %d %d %d",
                   InputEvent::Name(event), key, shift_down, ctrl_down, cmd_down);

  if (KeyboardController *g = screen->active_textbox) do {
    if (g->toggle_bind.key == event && !g->toggle_once) return 0;

    if (event == paste_bind.key) { g->Input(app->GetClipboardText()); return 1; }
    if (HandleSpecialKey(event, g)) return 1;

    if (cmd_down) return 0;
    if (key >= 128) { InputDebug("Input::KeyEventDispatch unhandled key %lld", event); break; }

    if (shift_down) key = Key::ShiftModified(key);
    if (ctrl_down)  key = Key::CtrlModified(key);

    g->Input(key);
    return 1;
  } while(0);

  return 0;
}

int Input::HandleSpecialKey(InputEvent::Id event, KeyboardController *g) {
  if      (event == Key::Backspace) { g->Erase();       return 1; }
  else if (event == Key::Delete)    { g->Erase();       return 1; }
  else if (event == Key::Return)    { g->Enter();       return 1; }
  else if (event == Key::Left)      { g->CursorLeft();  return 1; }
  else if (event == Key::Right)     { g->CursorRight(); return 1; }
  else if (event == Key::Up)        { g->HistUp();      return 1; }
  else if (event == Key::Down)      { g->HistDown();    return 1; }
  else if (event == Key::PageUp)    { g->PageUp();      return 1; }
  else if (event == Key::PageDown)  { g->PageDown();    return 1; }
  else if (event == Key::Home)      { g->Home();        return 1; }
  else if (event == Key::End)       { g->End();         return 1; }
  else if (event == Key::Tab)       { g->Tab();         return 1; }
  else if (event == Key::Escape)    { g->Escape();      return 1; }
  return 0;
}

int Input::MouseMove(const point &p, const point &d) {
  if (!app->run) return 0;
  int fired = MouseEventDispatch(Mouse::Event::Motion, p, MouseButton1Down());
  if (!screen->grab_mode.Enabled()) return fired;

  for (auto &g : screen->input)
    if (g->active) g->Move(Mouse::Event::Motion, p, d); 

  return fired;
}

int Input::MouseWheel(const point &p, const point &d) {
  if (!app->run) return 0;
  int fired = MouseEventDispatch(Mouse::Event::Wheel, screen->mouse, d.y);
  return fired;
}

int Input::MouseClick(int button, bool down, const point &p) {
  if (!app->run) return 0;
  InputEvent::Id event = Mouse::ButtonID(button);
  if      (event == Mouse::Button::_1) mouse_but1_down = down;
  else if (event == Mouse::Button::_2) mouse_but2_down = down;
  // event |= (CtrlKeyDown() ? Key::Modifier::Ctrl : 0) | (CmdKeyDown() ? Key::Modifier::Cmd : 0);

  int fired = MouseEventDispatch(event, p, down);
  if (fired) return fired;

  for (auto i = screen->input.begin(), e = screen->input.end(); i != e; ++i)
    if ((*i)->active) (*i)->Button(event, down);

  return fired;
}

int Input::MouseEventDispatch(InputEvent::Id event, const point &p, int down) {
  if      (event == paste_bind.key)      return KeyEventDispatch(event, down);
  else if (event == Mouse::Event::Wheel) screen->mouse_wheel = p;
  else                                   screen->mouse       = p;
  InputDebug("Input::MouseEventDispatch %s %s down=%d",
             InputEvent::Name(event), screen->mouse.DebugString().c_str(), down);

  int fired = 0, active_guis = 0, events;
  Dialog *bring_to_front = 0;
  for (auto i = screen->dialogs.begin(); i != screen->dialogs.end(); /**/) {
    Dialog *g = i->get();
    if (g->NotActive(screen->mouse)) { i++; continue; }
    fired += g->mouse.Input(event, g->RelativePosition(screen->mouse), down, 0);
    if (g->deleted) { screen->GiveDialogFocusAway(g); i = screen->dialogs.erase(i); continue; }
    if (event == Mouse::Event::Button1 && down && g->box.within(screen->mouse)) { bring_to_front = g; break; }
    i++;
  }
  if (bring_to_front) screen->BringDialogToFront(bring_to_front);

  for (auto i = screen->gui.begin(), e = screen->gui.end(); i != e; ++i) {
    if ((events = MouseEventDispatchGUI(event, p, down, *i, &active_guis))) {
      InputDebug("Input::MouseEventDispatch sent GUI[%td] events = %d", i - screen->gui.begin(), events);
      return events;
    }
  }

  InputDebugIfDown("Inut::MouseEventDispatch %s fired=%d, guis=%d/%zd",
                   screen->mouse.DebugString().c_str(), fired, active_guis, screen->gui.size());
  return fired;
}

int Input::MouseEventDispatchGUI(InputEvent::Id event, const point &p, int down, GUI *g, int *active_guis) {
  if (g->NotActive(screen->mouse)) return 0;
  else (*active_guis)++;
  int events = g->mouse.Input(event, g->RelativePosition(screen->mouse), down, 0);
  if (!events && g->child_gui) return MouseEventDispatchGUI(event, p, down, g->child_gui, active_guis);
  return events;
}

void MouseControllerCallback::Destruct() {
  switch(type) {
    case CB_VOID:  cb.cb_void .~CB();      break;
    case CB_BOOL:  cb.cb_bool .~BoolCB();  break;
    case CB_COORD: cb.cb_coord.~CoordCB(); break;
    default:                               break;
  }
}

void MouseControllerCallback::Assign(const MouseControllerCallback &c) {
  run_from_message_loop = c.run_from_message_loop;
  switch ((type = c.type)) {
    case CB_VOID:  new (&cb.cb_void)  CB     (c.cb.cb_void);  break;
    case CB_BOOL:  new (&cb.cb_bool)  BoolCB (c.cb.cb_bool);  break;
    case CB_COORD: new (&cb.cb_coord) CoordCB(c.cb.cb_coord); break;
    default:                                                  break;
  }
}

bool MouseControllerCallback::Run(const point &p, int button, int down, bool wrote) {
  if (run_from_message_loop && !wrote) { app->RunInMainThread([=]{ Run(p, button, down, true); }); return false; }
  bool ret = 1;
  switch (type) {
    case CB_VOID:  cb.cb_void();                        break;
    case CB_BOOL:  ret = cb.cb_bool();                  break;
    case CB_COORD: cb.cb_coord(button, p.x, p.y, down); break;
    default:                                            break;
  }
  return ret;
}

int MouseController::Input(InputEvent::Id event, const point &p, int down, int flag) {
  int fired = 0, boxes_checked = 0;
  bool but1 = event == Mouse::Event::Button1;
  bool but2 = event == Mouse::Event::Button2;

  for (auto h = hover.begin(); h != hover.end(); /**/) {
    auto e = &hit.data[*h];
    if (!e->deleted && e->box.within(p)) { ++h; continue; }
    h = VectorEraseIterSwapBack(&hover, h);
    if (e->deleted) continue;
    e->val = 0;
    e->CB.Run(p, event, 0);
    fired++;
  }

  for (auto e = hit.data.rbegin(), ee = hit.data.rend(); e != ee; ++e) {
    bool thunk = 0, e_hover = e->evtype == Event::Hover;

    if (e->deleted || !e->active || (e_hover && e->val) || 
        (!down && (e->evtype == Event::Click || e->evtype == Event::RightClick) &&
         e->CB.type != MouseControllerCallback::CB_COORD)) continue;

    InputDebug("check %s within %s", p.DebugString().c_str(), e->box.DebugString().c_str());
    boxes_checked++;
    if (e->box.within(p)) {
      if (e->run_only_if_first && fired) continue;
      switch (e->evtype) { 
        case Event::Click:      if (but1)         { thunk=1; } break;
        case Event::RightClick: if (but2)         { thunk=1; } break;
        case Event::Drag:       if (but1 && down) { thunk=1; } break;
        case Event::Hover:      if ((e->val = 1)) { thunk=1; } break;
      }
    }

    if (thunk) {
      InputDebugIfDown("MouseController::Input %s RunCB %s",
                       p.DebugString().c_str(), e->box.DebugString().c_str());
      if (!e->CB.Run(p, event, e_hover ? 1 : down)) continue;

      if (e_hover) hover.push_back(ForwardIteratorFromReverse(e) - hit.data.begin());
      fired++;

      if (e->evtype == Event::Drag && down) drag.insert(ForwardIteratorFromReverse(e) - hit.data.begin());
      if (flag) break;
    }
  }

  if (event == Mouse::Event::Motion) { for (auto d : drag) if (hit.data[d].CB.Run(p, event, down)) fired++; }
  else if (!down && but1)            { for (auto d : drag) if (hit.data[d].CB.Run(p, event, down)) fired++; drag.clear(); }

  InputDebugIfDown("MouseController::Input %s fired=%d, checked %zd of %zd hitboxes",
                   screen->mouse.DebugString().c_str(), fired, boxes_checked, hit.data.size());
  return fired;
}

}; // namespace LFL
