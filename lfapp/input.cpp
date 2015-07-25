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

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"

#ifdef LFL_QT
#include <QtOpenGL>
#endif

#ifdef LFL_WXWIDGETS
#include <wx/wx.h>
#endif

#ifdef LFL_ANDROID
#include <android/log.h>
#endif

#ifdef LFL_WININPUT
#include <windowsx.h>
#endif

#ifdef LFL_GLFWINPUT
#include "GLFW/glfw3.h"
#endif

#ifdef LFL_SDLINPUT
#include "SDL.h"
extern "C" {
#ifdef LFL_IPHONE
#include "SDL_uikitkeyboard.h"
#endif
};
#endif

namespace LFL {
DEFINE_float(ksens, 4, "Keyboard sensitivity");
DEFINE_float(msens, 1, "Mouse sensitivity");
DEFINE_int(invert, 1, "Invert mouse [1|-1]");
DEFINE_int(keyboard_repeat, 50, "Keyboard repeat in milliseconds");
DEFINE_int(keyboard_delay, 180, "Keyboard delay until repeat in milliseconds");
DEFINE_bool(input_debug, false, "Debug input events");

const InputEvent::Id Key::Modifier::Ctrl   = 1LL<<32;
const InputEvent::Id Key::Modifier::Cmd    = 1LL<<33;
const InputEvent::Id Mouse::Button::_1     = 1LL<<34;
const InputEvent::Id Mouse::Button::_2     = 1LL<<35;
const InputEvent::Id MouseEvent            = 1LL<<36;
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
        if (j < max_repeat) app->input.KeyEventDispatch(*i, true);
      }
      key_down_repeat[*i] = now - elapsed;
    }
  }
};
#endif

#if !defined(LFL_ANDROIDINPUT) && !defined(LFL_IPHONEINPUT)
void TouchDevice::OpenKeyboard() {}
void TouchDevice::CloseKeyboard() {}
Box TouchDevice::GetKeyboardBox() { return Box(); }
#endif

#if !defined(LFL_OSXINPUT) && !defined(LFL_WININPUT) && !defined(LFL_ANDROIDINPUT) && !defined(LFL_IPHONEINPUT) && !defined(LFL_QT) && !defined(LFL_WXWIDGETS) && !defined(LFL_GLFWINPUT) && !defined(LFL_SDLINPUT)
const int Key::Escape     = -1;
const int Key::Return     = -2;
const int Key::Up         = -3;
const int Key::Down       = -4;
const int Key::Left       = -5;
const int Key::Right      = -6;
const int Key::LeftShift  = -7;
const int Key::RightShift = -8;
const int Key::LeftCtrl   = -9;
const int Key::RightCtrl  = -10;
const int Key::LeftCmd    = -11;
const int Key::RightCmd   = -12;
const int Key::Tab        = -13;
const int Key::Space      = -14;
const int Key::Backspace  = -15;
const int Key::Delete     = -16;
const int Key::Quote      = -17;
const int Key::Backquote  = -18;
const int Key::PageUp     = -19;
const int Key::PageDown   = -20;
const int Key::F1         = -21;
const int Key::F2         = -22;
const int Key::F3         = -23;
const int Key::F4         = -24;
const int Key::F5         = -25;
const int Key::F6         = -26;
const int Key::F7         = -27;
const int Key::F8         = -28;
const int Key::F9         = -29;
const int Key::F10        = -30;
const int Key::F11        = -31;
const int Key::F12        = -32;
const int Key::Home       = -33;
const int Key::End        = -34;

string Clipboard::Get() { return ""; }
void Clipboard::Set(const string &s) {}
void Mouse::GrabFocus() {}
void Mouse::ReleaseFocus() {}
#endif

#ifdef LFL_ANDROIDINPUT
struct AndroidInputModule : public InputModule {
  int Frame(unsigned clicks) { return 0; }
};

const int Key::Escape     = -1;
const int Key::Return     = 10;
const int Key::Up         = -3;
const int Key::Down       = -4;
const int Key::Left       = -5;
const int Key::Right      = -6;
const int Key::LeftShift  = -7;
const int Key::RightShift = -8;
const int Key::LeftCtrl   = -9;
const int Key::RightCtrl  = -10;
const int Key::LeftCmd    = -11;
const int Key::RightCmd   = -12;
const int Key::Tab        = -13;
const int Key::Space      = -14;
const int Key::Backspace  = 0;
const int Key::Delete     = -16;
const int Key::Quote      = '\'';
const int Key::Backquote  = '`';
const int Key::PageUp     = -19;
const int Key::PageDown   = -20;
const int Key::F1         = -21;
const int Key::F2         = -22;
const int Key::F3         = -23;
const int Key::F4         = -24;
const int Key::F5         = -25;
const int Key::F6         = -26;
const int Key::F7         = -27;
const int Key::F8         = -28;
const int Key::F9         = -29;
const int Key::F10        = -30;
const int Key::F11        = -31;
const int Key::F12        = -32;
const int Key::Home       = -33;
const int Key::End        = -34;

static bool android_keyboard_toggled = false;

string Clipboard::Get() { return ""; }
void Clipboard::Set(const string &s) {}
void TouchDevice::OpenKeyboard()  { if ( android_keyboard_toggled) return; AndroidToggleKeyboard(); android_keyboard_toggled=1; }
void TouchDevice::CloseKeyboard() { if (!android_keyboard_toggled) return; AndroidToggleKeyboard(); android_keyboard_toggled=0; }
void Mouse::GrabFocus() {}
void Mouse::ReleaseFocus() {}
#endif

#ifdef LFL_IPHONEINPUT
extern "C" int iPhoneSetExtraScale(bool);
extern "C" int iPhoneSetMultisample(bool);
extern "C" void iPhoneShowKeyboard();
extern "C" void iPhoneHideKeyboard();
extern "C" void iPhoneHideKeyboardAfterReturn(bool v);
extern "C" void iPhoneGetKeyboardBox(int *x, int *y, int *w, int *h);
extern "C" void iPhoneCreateToolbar(int n, const char **name, const char **val);
extern "C" void iPhoneToggleToolbarButton(const char *n);

struct IPhoneInputModule : public InputModule {
  int Frame(unsigned clicks) { return 0; }
};

const int Key::Escape     = -1;
const int Key::Return     = 10;
const int Key::Up         = -3;
const int Key::Down       = -4;
const int Key::Left       = -5;
const int Key::Right      = -6;
const int Key::LeftShift  = -7;
const int Key::RightShift = -8;
const int Key::LeftCtrl   = -9;
const int Key::RightCtrl  = -10;
const int Key::LeftCmd    = -11;
const int Key::RightCmd   = -12;
const int Key::Tab        = -13;
const int Key::Space      = -14;
const int Key::Backspace  = 8;
const int Key::Delete     = -16;
const int Key::Quote      = -17;
const int Key::Backquote  = '~';
const int Key::PageUp     = -19;
const int Key::PageDown   = -20;
const int Key::F1         = -21;
const int Key::F2         = -22;
const int Key::F3         = -23;
const int Key::F4         = -24;
const int Key::F5         = -25;
const int Key::F6         = -26;
const int Key::F7         = -27;
const int Key::F8         = -28;
const int Key::F9         = -29;
const int Key::F10        = -30;
const int Key::F11        = -31;
const int Key::F12        = -32;
const int Key::Home       = -33;
const int Key::End        = -34;

string Clipboard::Get() { return ""; }
void Clipboard::Set(const string &s) {}
int TouchDevice::SetExtraScale(bool v) { return iPhoneSetExtraScale(v); }
int TouchDevice::SetMultisample(bool v) { return iPhoneSetMultisample(v); }
void TouchDevice::OpenKeyboard() { iPhoneShowKeyboard(); }
void TouchDevice::CloseKeyboard() { iPhoneHideKeyboard(); }
void TouchDevice::CloseKeyboardAfterReturn(bool v) { iPhoneHideKeyboardAfterReturn(v); } 
Box  TouchDevice::GetKeyboardBox() { Box ret; iPhoneGetKeyboardBox(&ret.x, &ret.y, &ret.w, &ret.h); return ret; }
void TouchDevice::ToggleToolbarButton(const string &n) { iPhoneToggleToolbarButton(n.c_str()); }
void Mouse::GrabFocus() {}
void Mouse::ReleaseFocus() {}
#endif

#ifdef LFL_OSXINPUT
extern "C" void OSXGrabMouseFocus();
extern "C" void OSXReleaseMouseFocus();
extern "C" void OSXSetMousePosition(void*, int x, int y);
extern "C" void OSXClipboardSet(const char *v);
extern "C" const char *OSXClipboardGet();

const int Key::Escape = 0x81;
const int Key::Return = '\r';
const int Key::Up = 0xBE;
const int Key::Down = 0xBD;
const int Key::Left = 0xBB;
const int Key::Right = 0xBC;
const int Key::LeftShift = 0x83;
const int Key::RightShift = 0x87;
const int Key::LeftCtrl = 0x86;
const int Key::RightCtrl = 0x89;
const int Key::LeftCmd = 0x82;
const int Key::RightCmd = -12;
const int Key::Tab = '\t';
const int Key::Space = ' ';
const int Key::Backspace = 0x80;
const int Key::Delete = -16;
const int Key::Quote = '\'';
const int Key::Backquote = '`';
const int Key::PageUp = 0xB4;
const int Key::PageDown = 0xB9;
const int Key::F1 = 0xBA;
const int Key::F2 = 0xB8;
const int Key::F3 = 0xA8;
const int Key::F4 = 0xB6;
const int Key::F5 = 0xA5;
const int Key::F6 = 0xA6;
const int Key::F7 = 0xA7;
const int Key::F8 = 0xA9;
const int Key::F9 = 0xAA;
const int Key::F10 = 0xAF;
const int Key::F11 = 0xAB;
const int Key::F12 = 0xB0;
const int Key::Home = 0xB3;
const int Key::End = 0xB7;

void Clipboard::Set(const string &s) { OSXClipboardSet(s.c_str()); }
string Clipboard::Get() { const char *v = OSXClipboardGet(); string ret = v; free((void*)v); return ret; }
void Mouse::ReleaseFocus() { OSXReleaseMouseFocus(); app->grab_mode.Off(); screen->cursor_grabbed = 0; }
void Mouse::GrabFocus() {
  OSXGrabMouseFocus();    app->grab_mode.On();  screen->cursor_grabbed = 1;
  OSXSetMousePosition(screen->id, screen->width / 2, screen->height / 2);
}
#endif // LFL_OSXINPUT

#ifdef LFL_WININPUT
struct WinInputModule {
  static int GetKeyCode(unsigned char k) {
    const unsigned short key_code[] = {
      0xF00, /*null*/              0xF01, /*Left mouse*/     0xF02, /*Right mouse*/     0xF03, /*Control-break*/
      0xF04, /*Middle mouse*/      0xF05, /*X1 mouse*/       0xF06, /*X2 mouse*/        0xF07, /*Undefined*/
      '\b',  /*BACKSPACE key*/     '\t',  /*TAB key*/        0xF0A, /*Reserved*/        0xF0B, /*Reserved*/
      0xF0C, /*CLEAR key*/         '\r',  /*ENTER key*/      0xF0E, /*Undefined*/       0xF0F, /*Undefined*/
      0xF10, /*SHIFT key*/         0xF11, /*CTRL key*/       0xF12, /*ALT key*/         0xF13, /*PAUSE key*/
      0xF14, /*CAPS LOCK key*/     0xF15, /*IME Kana mode*/  0xF16, /*Undefined*/       0xF17, /*IME Junja mode*/
      0xF18, /*IME final mode*/    0xF19, /*IME Hanja mode*/ 0xF1A, /*Undefined*/       '\x1b',/*ESC key*/
      0xF1C, /*IME convert*/       0xF1D, /*IME nonconvert*/ 0xF1E, /*IME accept*/      0xF1F, /*IME mode change*/
      ' ',   /*SPACEBAR*/          0xF21, /*PAGE UP key*/    0xF22, /*PAGE DOWN key*/   0xF23, /*END key*/
      0xF24, /*HOME key*/          0xF25, /*LEFT ARROW key*/ 0xF26, /*UP ARROW key*/    0xF27, /*RIGHT ARROW key*/
      0xF28, /*DOWN ARROW key*/    0xF29, /*SELECT key*/     0xF2A, /*PRINT key*/       0xF2B, /*EXECUTE key*/
      0xF2C, /*PRINT SCREEN key*/  0xF2D, /*INS key*/        0xF2E, /*DEL key*/         0xF2F, /*HELP key*/
      '0',   /*0 key*/             '1',   /*1 key*/          '2',   /*2 key*/           '3',   /*3 key*/
      '4',   /*4 key*/             '5',   /*5 key*/          '6',   /*6 key*/           '7',   /*7 key*/
      '8',   /*8 key*/             '9',   /*9 key*/          0xF3A, /*undefined*/       0xF3B, /*undefined*/
      0xF3C, /*undefined*/         0xF3D, /*undefined*/      0xF3E, /*undefined*/       0xF3F, /*undefined*/
      0xF40, /*undefined*/         'a',   /*A key*/          'b',   /*B key*/           'c',   /*C key*/
      'd',   /*D key*/             'e',   /*E key*/          'f',   /*F key*/           'g',   /*G key*/
      'h',   /*H key*/             'i',   /*I key*/          'j',   /*J key*/           'k',   /*K key*/
      'l',   /*L key*/             'm',   /*M key*/          'n',   /*N key*/           'o',   /*O key*/
      'p',   /*P key*/             'q',   /*Q key*/          'r',   /*R key*/           's',   /*S key*/
      't',   /*T key*/             'u',   /*U key*/          'v',   /*V key*/           'w',   /*W key*/
      'x',   /*X key*/             'y',   /*Y key*/          'z',   /*Z key*/           0xF5B, /*Left Windows key*/
      0xF5C, /*Right Windows key*/ 0xF5D, /*Apps-key*/       0xF5E, /*Reserved*/        0xF5F, /*Computer Sleep*/
      0xF60, /*keypad 0 key*/      0xF61, /*keypad 1 key*/   0xF62, /*keypad 2 key*/    0xF63, /*keypad 3 key*/
      0xF64, /*keypad 4 key*/      0xF65, /*keypad 5 key*/   0xF66, /*keypad 6 key*/    0xF67, /*keypad 7 key*/
      0xF68, /*keypad 8 key*/      0xF69, /*keypad 9 key*/   0xF6A, /*Multiply key*/    0xF6B, /*Add key*/
      0xF6C, /*Separator key*/     0xF6D, /*Subtract key*/   0xF6E, /*Decimal key*/     0xF6F, /*Divide key*/
      0xF70, /*F1 key*/            0xF71, /*F2 key*/         0xF72, /*F3 key*/          0xF73, /*F4 key*/
      0xF74, /*F5 key*/            0xF75, /*F6 key*/         0xF76, /*F7 key*/          0xF77, /*F8 key*/
      0xF78, /*F9 key*/            0xF79, /*F10 key*/        0xF7A, /*F11 key*/         0xF7B, /*F12 key*/
      0xF7C, /*F13 key*/           0xF7D, /*F14 key*/        0xF7E, /*F15 key*/         0xF7F, /*F16 key*/
      0xF80, /*F17 key*/           0xF81, /*F18 key*/        0xF82, /*F19 key*/         0xF83, /*F20 key*/
      0xF84, /*F21 key*/           0xF85, /*F22 key*/        0xF86, /*F23 key*/         0xF87, /*F24 key*/
      0xF88, /*Unassigned*/        0xF89, /*Unassigned*/     0xF8A, /*Unassigned*/      0xF8B, /*Unassigned*/
      0xF8C, /*Unassigned*/        0xF8D, /*Unassigned*/     0xF8E, /*Unassigned*/      0xF8F, /*Unassigned*/
      0xF90, /*NUM LOCK key*/      0xF91, /*SCROLL LOCK*/    0xF92, /*OEM specific*/    0xF93, /*OEM specific*/
      0xF94, /*OEM specific*/      0xF95, /*OEM specific*/   0xF96, /*OEM specific*/    0xF97, /*Unassigned*/
      0xF98, /*Unassigned*/        0xF99, /*Unassigned*/     0xF9A, /*Unassigned*/      0xF9B, /*Unassigned*/
      0xF9C, /*Unassigned*/        0xF9D, /*Unassigned*/     0xF9E, /*Unassigned*/      0xF9F, /*Unassigned*/
      0xFA0, /*Left SHIFT key*/    0xFA1, /*Right SHIFT*/    0xFA2, /*Left CONTROL*/    0xFA3, /*Right CONTROL*/
      0xFA4, /*Left MENU key*/     0xFA5, /*Right MENU*/     0xFA6, /*Browser Back*/    0xFA7, /*Browser Forward*/
      0xFA8, /*Browser Refresh*/   0xFA9, /*Browser Stop*/   0xFAA, /*Browser Search*/  0xFAB, /*Browser Favorites*/
      0xFAC, /*Browser Home*/      0xFAD, /*Volume Mute*/    0xFAE, /*Volume Down*/     0xFAF, /*Volume Up*/
      0xFB0, /*Next Track key*/    0xFB1, /*Previous Track*/ 0xFB2, /*Stop Media*/      0xFB3, /*Play/Pause Media*/
      0xFB4, /*Start Mail key*/    0xFB5, /*Select Media*/   0xFB6, /*Start App-1 key*/ 0xFB7, /*Start Application 2*/
      0xFB8, /*Reserved*/          0xFB9, /*Reserved*/       ';',   /*the ';:' key*/    '=',  /*the '+' key*/
      ',',   /*the ',' key*/       '-',   /*the '-' key*/    '.',   /*the '.' key*/     '/',  /*the '/?' key*/
      '\`',  /*the '`~' key*/      0xFC1, /*Reserved*/       0xFC2, /*Reserved*/        0xFC3, /*Reserved*/
      0xFC4, /*Reserved*/          0xFC5, /*Reserved*/       0xFC6, /*Reserved*/        0xFC7, /*Reserved*/
      0xFC8, /*Reserved*/          0xFC9, /*Reserved*/       0xFCA, /*Reserved*/        0xFCB, /*Reserved*/
      0xFCC, /*Reserved*/          0xFCD, /*Reserved*/       0xFCE, /*Reserved*/        0xFCF, /*Reserved*/
      0xFD0, /*Reserved*/          0xFD1, /*Reserved*/       0xFD2, /*Reserved*/        0xFD3, /*Reserved*/
      0xFD4, /*Reserved*/          0xFD5, /*Reserved*/       0xFD6, /*Reserved*/        0xFD7, /*Reserved*/
      0xFD8, /*Unassigned*/        0xFD9, /*Unassigned*/     0xFDA, /*Unassigned*/      '[',  /*the '[{' key*/
      '\\',  /*the '\|' key*/      ']',  /*the ']}' key*/    '\'',  /*'quote' key*/     0xFDF, /*misc*/
      0xFE0, /*Reserved*/          0xFE1, /*OEM specific*/   0xFE2, /*RT 102 bracket*/  0xFE3, /*OEM specific*/
      0xFE4, /*OEM specific*/      0xFE5, /*IME PROCESS*/    0xFE6, /*OEM specific*/    0xFE7, /*Used for Unicode*/
      0xFE8, /*Unassigned*/        0xFE9, /*OEM specific*/   0xFEA, /*OEM specific*/    0xFEB, /*OEM specific*/
      0xFEC, /*OEM specific*/      0xFED, /*OEM specific*/   0xFEE, /*OEM specific*/    0xFEF, /*OEM specific*/
      0xFF0, /*OEM specific*/      0xFF1, /*OEM specific*/   0xFF2, /*OEM specific*/    0xFF3, /*OEM specific*/
      0xFF4, /*OEM specific*/      0xFF5, /*OEM specific*/   0xFF6, /*Attn key*/        0xFF7, /*CrSel key*/
      0xFF8, /*ExSel key*/         0xFF9, /*Erase EOF key*/  0xFFA, /*Play key*/        0xFFB, /*Zoom key*/
      0xFFC, /*Reserved*/          0xFFD, /*PA1 key*/        0xFFE, /*Clear key*/       0xFFF, /*Unused*/
    };
    return key_code[k];
  }
  static void UpdateMousePosition(const LPARAM &lParam, point *p, point *d) {
    WinWindow *win = static_cast<WinWindow*>(screen->impl);
    *p = point(GET_X_LPARAM(lParam), screen->height - GET_Y_LPARAM(lParam));
    *d = *p - win->prev_mouse_pos;
    win->prev_mouse_pos = *p;
  }
};

void WinApp::CreateClass() {
  WNDCLASS wndClass;
  wndClass.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
  wndClass.lpfnWndProc = &WinApp::WndProc;
  wndClass.cbClsExtra = 0;
  wndClass.cbWndExtra = 0;
  wndClass.hInstance = hInst;
  wndClass.hIcon = LoadIcon(hInst, "IDI_APP_ICON");
  wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
  wndClass.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
  wndClass.lpszMenuName = NULL;
  wndClass.lpszClassName = app->name.c_str();
  if (!RegisterClass(&wndClass)) ERROR("RegisterClass: ", GetLastError());
}

int WinApp::MessageLoop() {
  INFOf("WinApp::MessageLoop %p", screen);
  CoInitialize(NULL);
  MSG msg;
  while (app->run) {
    if (app->run && FLAGS_target_fps) LFAppFrame();
    while (app->run && PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))         { TranslateMessage(&msg); DispatchMessage(&msg); }
    if (app->run && !FLAGS_target_fps) if (GetMessage(&msg, NULL, 0, 0)) { TranslateMessage(&msg); DispatchMessage(&msg); }
  }
  app->Free();
  CoUninitialize();
  return msg.wParam;
}

LRESULT APIENTRY WinApp::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
  WinWindow *win = static_cast<WinWindow*>(screen->impl);
  PAINTSTRUCT ps;
  POINT cursor;
  point p, d;
  int ind;
  switch (message) {
  case WM_CREATE:                      return 0;
  case WM_DESTROY:                     LFAppShutdown(); PostQuitMessage(0); return 0;
  case WM_SIZE:                        WindowReshaped(LOWORD(lParam), HIWORD(lParam)); app->scheduler.Wakeup(0); return 0;
  case WM_KEYUP:   case WM_SYSKEYUP:   if (KeyPress(WinInputModule::GetKeyCode(wParam), 0) && win->frame_on_keyboard_input) app->scheduler.Wakeup(0); return 0;
  case WM_KEYDOWN: case WM_SYSKEYDOWN: if (KeyPress(WinInputModule::GetKeyCode(wParam), 1) && win->frame_on_keyboard_input) app->scheduler.Wakeup(0); return 0;
  case WM_LBUTTONDOWN:                 if (MouseClick(1, 1, win->prev_mouse_pos.x, win->prev_mouse_pos.y) && win->frame_on_mouse_input) app->scheduler.Wakeup(0); return 0;
  case WM_LBUTTONUP:                   if (MouseClick(1, 0, win->prev_mouse_pos.x, win->prev_mouse_pos.y) && win->frame_on_mouse_input) app->scheduler.Wakeup(0); return 0;
  case WM_RBUTTONDOWN:                 if (MouseClick(2, 1, win->prev_mouse_pos.x, win->prev_mouse_pos.y) && win->frame_on_mouse_input) app->scheduler.Wakeup(0); return 0;
  case WM_RBUTTONUP:                   if (MouseClick(2, 0, win->prev_mouse_pos.x, win->prev_mouse_pos.y) && win->frame_on_mouse_input) app->scheduler.Wakeup(0); return 0;
  case WM_MOUSEMOVE:                   WinInputModule::UpdateMousePosition(lParam, &p, &d); if (MouseMove(p.x, p.y, d.x, d.y) && win->frame_on_mouse_input) app->scheduler.Wakeup(0); return 0;
  case WM_COMMAND:                     if ((ind = wParam - win->start_msg_id) >= 0) if (ind < win->menu_cmds.size()) ShellRun(win->menu_cmds[ind].c_str()); return 0;
  case WM_CONTEXTMENU:                 if (win->menu) { GetCursorPos(&cursor); TrackPopupMenu(win->context_menu, TPM_LEFTALIGN|TPM_TOPALIGN, cursor.x, cursor.y, 0, hWnd, NULL); } return 0;
  case WM_PAINT:                       BeginPaint((HWND)screen->id, &ps); if (!FLAGS_target_fps) LFAppFrame(); EndPaint((HWND)screen->id, &ps); return 0;
  case WM_SIZING:                      return win->resize_increment.Zero() ? 0 : win->RestrictResize(wParam, reinterpret_cast<LPRECT>(lParam));
  case WM_USER:                        if (!FLAGS_target_fps) LFAppFrame(); return 0;
  case WM_KILLFOCUS:                   app->input.ClearButtonsDown(); return 0;
  default:                             break;
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

bool WinWindow::RestrictResize(int m, RECT *r) {
  point in(r->right - r->left, r->bottom - r->top);
  RECT w = { 0, 0, in.x, in.y };
  AdjustWindowRect(&w, GetWindowLong((HWND)screen->id, GWL_STYLE), menubar);
  point extra((w.right - w.left) - in.x, (w.bottom - w.top) - in.y), content = in - extra;
  switch (m) {
    case WMSZ_TOP:         r->top    -= (NextMultipleOfN(content.y, resize_increment.y) - content.y); break;
    case WMSZ_BOTTOM:      r->bottom += (NextMultipleOfN(content.y, resize_increment.y) - content.y); break;
    case WMSZ_LEFT:        r->left   -= (NextMultipleOfN(content.x, resize_increment.x) - content.x); break;
    case WMSZ_RIGHT:       r->right  += (NextMultipleOfN(content.x, resize_increment.x) - content.x); break;
    case WMSZ_BOTTOMLEFT:  r->bottom += (NextMultipleOfN(content.y, resize_increment.y) - content.y); 
                           r->left   -= (NextMultipleOfN(content.x, resize_increment.x) - content.x); break;
    case WMSZ_BOTTOMRIGHT: r->right  += (NextMultipleOfN(content.x, resize_increment.x) - content.x);
                           r->bottom += (NextMultipleOfN(content.y, resize_increment.y) - content.y); break;
    case WMSZ_TOPLEFT:     r->top    -= (NextMultipleOfN(content.y, resize_increment.y) - content.y);
                           r->left   -= (NextMultipleOfN(content.x, resize_increment.x) - content.x); break;
    case WMSZ_TOPRIGHT:    r->right  += (NextMultipleOfN(content.x, resize_increment.x) - content.x);
                           r->top    -= (NextMultipleOfN(content.y, resize_increment.y) - content.y); break;
  }
  return true;
}

const int Key::Escape     = '\x1b';
const int Key::Return     = '\r';
const int Key::Up         = 0xf00 | VK_UP;
const int Key::Down       = 0xf00 | VK_DOWN;
const int Key::Left       = 0xf00 | VK_LEFT;
const int Key::Right      = 0xf00 | VK_RIGHT;
const int Key::LeftShift  = 0xf00 | VK_SHIFT; // VK_LSHIFT;
const int Key::RightShift = 0xf00 | VK_RSHIFT;
const int Key::LeftCtrl   = 0xf00 | VK_CONTROL; // VK_LCONTROL;
const int Key::RightCtrl  = 0xf00 | VK_RCONTROL;
const int Key::LeftCmd    = 0xf00 | VK_MENU;
const int Key::RightCmd   = -12;
const int Key::Tab        = 0xf00 | VK_TAB;
const int Key::Space      = 0xf00 | VK_SPACE;
const int Key::Backspace  = '\b';
const int Key::Delete     = 0xf00 | VK_DELETE;
const int Key::Quote      = '\'';
const int Key::Backquote  = '`';
const int Key::PageUp     = 0xf00 | VK_PRIOR;
const int Key::PageDown   = 0xf00 | VK_NEXT;
const int Key::F1         = 0xf00 | VK_F1;
const int Key::F2         = 0xf00 | VK_F2;
const int Key::F3         = 0xf00 | VK_F3;
const int Key::F4         = 0xf00 | VK_F4;
const int Key::F5         = 0xf00 | VK_F5;
const int Key::F6         = 0xf00 | VK_F6;
const int Key::F7         = 0xf00 | VK_F7;
const int Key::F8         = 0xf00 | VK_F8;
const int Key::F9         = 0xf00 | VK_F9;
const int Key::F10        = 0xf00 | VK_F10;
const int Key::F11        = 0xf00 | VK_F11;
const int Key::F12        = 0xf00 | VK_F12;
const int Key::Home       = 0xf00 | VK_HOME;
const int Key::End        = 0xf00 | VK_END;

void Mouse::ReleaseFocus() {}
void Mouse::GrabFocus() {}
void Clipboard::Set(const string &in) {
  String16 s = String::ToUTF16(in);
  if (!OpenClipboard(NULL)) return;
  EmptyClipboard();
  HGLOBAL hg = GlobalAlloc(GMEM_MOVEABLE, (s.size()+1)*2);
  if (!hg) { CloseClipboard(); return; }
  memcpy(GlobalLock(hg), s.c_str(), (s.size()+1)*2);
  GlobalUnlock(hg);
  SetClipboardData(CF_UNICODETEXT, hg);
  CloseClipboard();
  GlobalFree(hg);
}
string Clipboard::Get() {
  string ret;
  if (!OpenClipboard(NULL)) return "";
  const HANDLE hg = GetClipboardData(CF_UNICODETEXT);
  if (!hg) { CloseClipboard(); return ""; }
  ret = String::ToUTF8(reinterpret_cast<short*>(GlobalLock(hg)));
  GlobalUnlock(hg);
  CloseClipboard();
  return ret;
}
#endif // LFL_WININPUT

#ifdef LFL_QT
struct QTInputModule : public InputModule {
  bool grabbed = 0;
  int Frame(unsigned clicks) {
    app->input.DispatchQueuedInput();
    return 0;
  }
};

const int Key::Escape     = Qt::Key_Escape;
const int Key::Return     = Qt::Key_Return;
const int Key::Up         = Qt::Key_Up;
const int Key::Down       = Qt::Key_Down;
const int Key::Left       = Qt::Key_Left;
const int Key::Right      = Qt::Key_Right;
const int Key::LeftShift  = Qt::Key_Shift;
const int Key::RightShift = -8;
const int Key::LeftCtrl   = Qt::Key_Meta;
const int Key::RightCtrl  = -10;
const int Key::LeftCmd    = Qt::Key_Control;
const int Key::RightCmd   = -12;
const int Key::Tab        = Qt::Key_Tab;
const int Key::Space      = Qt::Key_Space;
const int Key::Backspace  = Qt::Key_Backspace;
const int Key::Delete     = Qt::Key_Delete;
const int Key::Quote      = Qt::Key_Apostrophe;
const int Key::Backquote  = Qt::Key_QuoteLeft;
const int Key::PageUp     = Qt::Key_PageUp;
const int Key::PageDown   = Qt::Key_PageDown;
const int Key::F1         = Qt::Key_F1;
const int Key::F2         = Qt::Key_F2;
const int Key::F3         = Qt::Key_F3;
const int Key::F4         = Qt::Key_F4;
const int Key::F5         = Qt::Key_F5;
const int Key::F6         = Qt::Key_F6;
const int Key::F7         = Qt::Key_F7;
const int Key::F8         = Qt::Key_F8;
const int Key::F9         = Qt::Key_F9;
const int Key::F10        = Qt::Key_F10;
const int Key::F11        = Qt::Key_F11;
const int Key::F12        = Qt::Key_F12;
const int Key::Home       = Qt::Key_Home;
const int Key::End        = Qt::Key_End;

string Clipboard::Get() { return ""; }
void Clipboard::Set(const string &s) {}
#endif /* LFL_QT */

#ifdef LFL_WXWIDGETS
struct QTInputModule : public InputModule {
  bool grabbed = 0;
  int Frame(unsigned clicks) {
    app->input.DispatchQueuedInput();
    return 0;
  }
};

const int Key::Escape     = WXK_ESCAPE;
const int Key::Return     = WXK_RETURN;
const int Key::Up         = WXK_UP;
const int Key::Down       = WXK_DOWN;
const int Key::Left       = WXK_LEFT;
const int Key::Right      = WXK_RIGHT;
const int Key::LeftShift  = WXK_SHIFT;
const int Key::RightShift = -8;
const int Key::LeftCtrl   = WXK_ALT;
const int Key::RightCtrl  = -10;
const int Key::LeftCmd    = WXK_CONTROL;
const int Key::RightCmd   = -12;
const int Key::Tab        = WXK_TAB;
const int Key::Space      = WXK_SPACE;
const int Key::Backspace  = WXK_BACK;
const int Key::Delete     = WXK_DELETE;
const int Key::Quote      = '\'';
const int Key::Backquote  = '`';
const int Key::PageUp     = WXK_PAGEUP;
const int Key::PageDown   = WXK_PAGEDOWN;
const int Key::F1         = WXK_F1;
const int Key::F2         = WXK_F2;
const int Key::F3         = WXK_F3;
const int Key::F4         = WXK_F4;
const int Key::F5         = WXK_F5;
const int Key::F6         = WXK_F6;
const int Key::F7         = WXK_F7;
const int Key::F8         = WXK_F8;
const int Key::F9         = WXK_F9;
const int Key::F10        = WXK_F10;
const int Key::F11        = WXK_F11;
const int Key::F12        = WXK_F12;
const int Key::Home       = WXK_HOME;
const int Key::End        = WXK_END;

string Clipboard::Get() { return ""; }
void Clipboard::Set(const string &s) {}
#endif /* LFL_WXWIDGETS */

#ifdef LFL_GLFWINPUT
struct GLFWInputModule : public InputModule {
  int Init(Window *W) { InitWindow((GLFWwindow*)screen->id); return 0; }
  int Frame(unsigned clicks) { glfwPollEvents(); return 0; }

  static void InitWindow(GLFWwindow *W) {
    glfwSetInputMode          (W, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetWindowCloseCallback(W, WindowClose);
    glfwSetWindowSizeCallback (W, WindowSize);
    glfwSetKeyCallback        (W, Key);
    glfwSetMouseButtonCallback(W, MouseClick);
    glfwSetCursorPosCallback  (W, MousePosition);
    glfwSetScrollCallback     (W, MouseWheel);
  }
  static bool LoadScreen(GLFWwindow *W) {
    if (!SetNativeWindowByID(W)) return 0;
    screen->events.input++;
    return 1;
  }
  static void WindowSize (GLFWwindow *W, int w, int h) { if (!LoadScreen(W)) return; screen->Reshaped(w, h); }
  static void WindowClose(GLFWwindow *W)               { if (!LoadScreen(W)) return; Window::Close(screen); }
  static void Key(GLFWwindow *W, int k, int s, int a, int m) {
    if (!LoadScreen(W)) return;
    app->input.KeyPress((unsigned)k < 256 && isalpha((unsigned)k) ? ::tolower((unsigned)k) : k,
                        (a == GLFW_PRESS || a == GLFW_REPEAT));
  }
  static void MouseClick(GLFWwindow *W, int b, int a, int m) {
    if (!LoadScreen(W)) return;
    app->input.MouseClick(MouseButton(b), a == GLFW_PRESS, screen->mouse);
  }
  static void MousePosition(GLFWwindow *W, double x, double y) {
    if (!LoadScreen(W)) return;
    point p = Input::TransformMouseCoordinate(point(x, y));
    app->input.MouseMove(p, p - screen->mouse);
  }
  static void MouseWheel(GLFWwindow *W, double x, double y) {
    if (!LoadScreen(W)) return;
    point p(x, y);
    app->input.MouseWheel(p, p -screen->mouse_wheel);
  }
  static unsigned MouseButton(int b) {
    switch (b) {
      case GLFW_MOUSE_BUTTON_1: return 1;
      case GLFW_MOUSE_BUTTON_2: return 2;
      case GLFW_MOUSE_BUTTON_3: return 3;
      case GLFW_MOUSE_BUTTON_4: return 4;
    } return 0;
  }
};

const int Key::Escape     = GLFW_KEY_ESCAPE;
const int Key::Return     = GLFW_KEY_ENTER;
const int Key::Up         = GLFW_KEY_UP;
const int Key::Down       = GLFW_KEY_DOWN;
const int Key::Left       = GLFW_KEY_LEFT;
const int Key::Right      = GLFW_KEY_RIGHT;
const int Key::LeftShift  = GLFW_KEY_LEFT_SHIFT;
const int Key::RightShift = GLFW_KEY_RIGHT_SHIFT;
const int Key::LeftCtrl   = GLFW_KEY_LEFT_CONTROL;
const int Key::RightCtrl  = GLFW_KEY_RIGHT_CONTROL;
#ifdef __APPLE__
const int Key::LeftCmd    = GLFW_KEY_LEFT_SUPER;
const int Key::RightCmd   = GLFW_KEY_RIGHT_SUPER;
#else
const int Key::LeftCmd    = GLFW_KEY_LEFT_ALT;
const int Key::RightCmd   = GLFW_KEY_RIGHT_ALT;
#endif
const int Key::Tab        = GLFW_KEY_TAB;
const int Key::Space      = GLFW_KEY_SPACE;
const int Key::Backspace  = GLFW_KEY_BACKSPACE;
const int Key::Delete     = GLFW_KEY_DELETE;
const int Key::Quote      = '\'';
const int Key::Backquote  = '`';
const int Key::PageUp     = GLFW_KEY_PAGE_UP;
const int Key::PageDown   = GLFW_KEY_PAGE_DOWN;
const int Key::F1         = GLFW_KEY_F1;
const int Key::F2         = GLFW_KEY_F2;
const int Key::F3         = GLFW_KEY_F3;
const int Key::F4         = GLFW_KEY_F4;
const int Key::F5         = GLFW_KEY_F5;
const int Key::F6         = GLFW_KEY_F6;
const int Key::F7         = GLFW_KEY_F7;
const int Key::F8         = GLFW_KEY_F8;
const int Key::F9         = GLFW_KEY_F9;
const int Key::F10        = GLFW_KEY_F10;
const int Key::F11        = GLFW_KEY_F11;
const int Key::F12        = GLFW_KEY_F12;
const int Key::Home       = GLFW_KEY_HOME;
const int Key::End        = GLFW_KEY_END;

string Clipboard::Get()                { return glfwGetClipboardString((GLFWwindow*)screen->id); }
void   Clipboard::Set(const string &s) {        glfwSetClipboardString((GLFWwindow*)screen->id, s.c_str()); }
void Mouse::GrabFocus()    { glfwSetInputMode((GLFWwindow*)screen->id, GLFW_CURSOR, GLFW_CURSOR_DISABLED); app->grab_mode.On();  screen->cursor_grabbed=true;  }
void Mouse::ReleaseFocus() { glfwSetInputMode((GLFWwindow*)screen->id, GLFW_CURSOR, GLFW_CURSOR_NORMAL);   app->grab_mode.Off(); screen->cursor_grabbed=false; }
#endif

#ifdef LFL_SDLINPUT
struct SDLInputModule : public InputModule {
  int Frame(unsigned clicks) {
    SDL_Event ev; point mp;
    SDL_GetMouseState(&mp.x, &mp.y);
    bool mouse_moved = false;

    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_QUIT) app->run = false;
      else if (ev.type == SDL_WINDOWEVENT) {
        if (ev.window.event == SDL_WINDOWEVENT_FOCUS_GAINED ||
            ev.window.event == SDL_WINDOWEVENT_SHOWN ||
            ev.window.event == SDL_WINDOWEVENT_RESIZED ||
            ev.window.event == SDL_WINDOWEVENT_CLOSE) {
          CHECK((screen = Window::Get((void*)(long)ev.window.windowID)));
          Window::MakeCurrent(screen);
        }
        if      (ev.window.event == SDL_WINDOWEVENT_RESIZED) screen->Reshape(ev.window.data1, ev.window.data2);
        else if (ev.window.event == SDL_WINDOWEVENT_CLOSE) Window::Close(screen);
      }
      else if (ev.type == SDL_KEYDOWN) app->input.KeyPress(ev.key.keysym.sym, 1);
      else if (ev.type == SDL_KEYUP)   app->input.KeyPress(ev.key.keysym.sym, 0);
      else if (ev.type == SDL_MOUSEMOTION) {
        app->input.MouseMove(mp, point(ev.motion.xrel, ev.motion.yrel));
        mouse_moved = true;
      }
      else if (ev.type == SDL_MOUSEBUTTONDOWN) app->input.MouseClick(ev.button.button, 1, point(ev.button.x, ev.button.y));
      else if (ev.type == SDL_MOUSEBUTTONUP)   app->input.MouseClick(ev.button.button, 0, point(ev.button.x, ev.button.y));
      // else if (ev.type == SDL_ACTIVEEVENT && ev.active.state & SDL_APPACTIVE) { if ((minimized = ev.active.gain)) return 0; }
      if (screen) screen->events.input++;
    }

#ifndef __APPLE__
    if (mouse_moved && screen->cursor_grabbed) {
      SDL_WarpMouseInWindow((SDL_Window*)screen->id, width/2, height/2);
      while(SDL_PollEvent(&ev)) { /* do nothing */ }
    }
#endif
    return 0;
  }
};

const int Key::Escape     = SDLK_ESCAPE;
const int Key::Return     = SDLK_RETURN;
const int Key::Up         = SDLK_UP;
const int Key::Down       = SDLK_DOWN;
const int Key::Left       = SDLK_LEFT;
const int Key::Right      = SDLK_RIGHT;
const int Key::LeftShift  = SDLK_LSHIFT;
const int Key::RightShift = SDLK_RSHIFT;
const int Key::LeftCtrl   = SDLK_LCTRL;
const int Key::RightCtrl  = SDLK_RCTRL;
const int Key::LeftCmd    = SDLK_LGUI;
const int Key::RightCmd   = SDLK_RGUI;
const int Key::Tab        = SDLK_TAB;
const int Key::Space      = SDLK_SPACE;
const int Key::Backspace  = SDLK_BACKSPACE;
const int Key::Delete     = SDLK_DELETE;
const int Key::Quote      = SDLK_QUOTE;
const int Key::Backquote  = SDLK_BACKQUOTE;
const int Key::PageUp     = SDLK_PAGEUP;
const int Key::PageDown   = SDLK_PAGEDOWN;
const int Key::F1         = SDLK_F1;
const int Key::F2         = SDLK_F2;
const int Key::F3         = SDLK_F3;
const int Key::F4         = SDLK_F4;
const int Key::F5         = SDLK_F5;
const int Key::F6         = SDLK_F6;
const int Key::F7         = SDLK_F7;
const int Key::F8         = SDLK_F8;
const int Key::F9         = SDLK_F9;
const int Key::F10        = SDLK_F10;
const int Key::F11        = SDLK_F11;
const int Key::F12        = SDLK_F12;
const int Key::Home       = SDLK_HOME;
const int Key::End        = SDLK_END;

string Clipboard::Get() { return SDL_GetClipboardText(); }
void Clipboard::Set(const string &s) { SDL_SetClipboardText(s.c_str()); }
void TouchDevice::CloseKeyboard() {
#ifdef LFL_IPHONE 
  SDL_iPhoneKeyboardHide((SDL_Window*)screen->id);
#endif
}
void TouchDevice::OpenKeyboard() {
#ifdef LFL_IPHONE 
  SDL_iPhoneKeyboardShow((SDL_Window*)screen->id);
#endif
}
void Mouse::GrabFocus()    { SDL_ShowCursor(0); SDL_SetWindowGrab((SDL_Window*)screen->id, SDL_TRUE);  SDL_SetRelativeMouseMode(SDL_TRUE);  app->grab_mode.On();  screen->cursor_grabbed=true; }
void Mouse::ReleaseFocus() { SDL_ShowCursor(1); SDL_SetWindowGrab((SDL_Window*)screen->id, SDL_FALSE); SDL_SetRelativeMouseMode(SDL_FALSE); app->grab_mode.Off(); screen->cursor_grabbed=false; }
#endif /* LFL_SDLINPUT */

void TouchDevice::AddToolbar(const vector<pair<string, string>>&items) {
  vector<const char *> k, v;
  for (auto &i : items) { k.push_back(i.first.c_str()); v.push_back(i.second.c_str()); }
#ifdef LFL_IPHONEINPUT
  iPhoneCreateToolbar(items.size(), &k[0], &v[0]);
#endif
}

void Input::ClearButtonsDown() {
  if (left_shift_down)  { KeyPress(Key::LeftShift,  0);    left_shift_down = 0; }
  if (right_shift_down) { KeyPress(Key::RightShift, 0);    left_shift_down = 0; }
  if (left_ctrl_down)   { KeyPress(Key::LeftCtrl,   0);    left_ctrl_down  = 0; }
  if (right_ctrl_down)  { KeyPress(Key::RightCtrl,  0);    right_ctrl_down = 0; }
  if (left_cmd_down)    { KeyPress(Key::LeftCmd,    0);    left_cmd_down   = 0; }
  if (right_cmd_down)   { KeyPress(Key::RightCmd,   0);    right_cmd_down  = 0; }
  if (mouse_but1_down)  { MouseClick(1, 0, screen->mouse); mouse_but1_down = 0; }
  if (mouse_but2_down)  { MouseClick(2, 0, screen->mouse); mouse_but2_down = 0; }
}

int Input::Init() {
  INFO("Input::Init()");
#ifdef __APPLE__
  paste_bind = Bind('v', Key::Modifier::Cmd);
#else
  paste_bind = Bind('v', Key::Modifier::Ctrl);
#endif

#if defined(LFL_QT)
  impl = new QTInputModule();
#elif defined(LFL_GLFWINPUT)
  impl = new GLFWInputModule();
#elif defined(LFL_SDLINPUT)
  impl = new SDLInputModule();
#elif defined(LFL_ANDROIDINPUT)
  impl = new AndroidInputModule();
#elif defined(LFL_IPHONEINPUT)
  impl = new IPhoneInputModule();
#endif
  return 0;
}

int Input::Init(Window *W) {
  return impl ? impl->Init(W) : 0;
}

int Input::Frame(unsigned clicks) {
  if (impl) impl->Frame(clicks);
  if (screen) { // XXX support multiple windows
    if (screen->binds) screen->binds->Repeat(clicks);
    if (screen->gesture_swipe_up)   { if (screen->console && screen->console->active) screen->console->PageUp();   }
    if (screen->gesture_swipe_down) { if (screen->console && screen->console->active) screen->console->PageDown(); }
    screen->ClearGesture();
  }
  return 0;
}

int Input::DispatchQueuedInput() {
  vector<Callback> icb;
  {
    ScopedMutex sm(queued_input_mutex);
    swap(icb, queued_input);
  }
  int ret = icb.size();
  for (auto i = icb.begin(); i != icb.end(); ++i) {
    (*i)();
    if (screen) screen->events.input++;
  }
  return ret;
}

int Input::KeyPress(int key, bool down) {
  // if (!MainThread()) ERROR("KeyPress() called from thread ", Thread::GetId());

  switch (key) {
    case Key::LeftShift:   left_shift_down = down; break;
    case Key::RightShift: right_shift_down = down; break;
    case Key::LeftCtrl:     left_ctrl_down = down; break;
    case Key::RightCtrl:   right_ctrl_down = down; break;
    case Key::LeftCmd:       left_cmd_down = down; break;
    case Key::RightCmd:     right_cmd_down = down; break;
  }

  InputEvent::Id event = key | (CtrlKeyDown() ? Key::Modifier::Ctrl : 0) | (CmdKeyDown() ? Key::Modifier::Cmd : 0);
  int fired = KeyEventDispatch(event, down);
  screen->events.key++;
  screen->events.gui += fired;
  if (fired) return fired;

  for (auto g = screen->input_bind.begin(); g != screen->input_bind.end(); ++g)
    if ((*g)->active) (*g)->Input(event, down); 

  return fired;
}

int Input::KeyEventDispatch(InputEvent::Id event, bool down) {
  if (!down) return 0;
  int key = InputEvent::GetKey(event);
  bool shift_down = ShiftKeyDown(), ctrl_down = CtrlKeyDown(), cmd_down = CmdKeyDown();

  if (FLAGS_input_debug && down)
    INFO("KeyEvent ", InputEvent::Name(event), " ", key, " ", shift_down, " ", ctrl_down, " ", cmd_down);

  for (auto it = screen->keyboard_gui.begin(); it != screen->keyboard_gui.end(); ++it) {
    KeyboardGUI *g = *it;
    if (!g->active) continue;
    if (g->toggle_bind.key == event && g->toggle_active.mode != Toggler::OneShot) return 0;

    g->events.total++;
    if (event == paste_bind.key) { g->Input(Clipboard::Get()); return 1; }
    switch (event) {
      case Key::Backspace: g->Erase();       return 1;
      case Key::Delete:    g->Erase();       return 1;
      case Key::Return:    g->Enter();       return 1;
      case Key::Left:      g->CursorLeft();  return 1;
      case Key::Right:     g->CursorRight(); return 1;
      case Key::Up:        g->HistUp();      return 1;
      case Key::Down:      g->HistDown();    return 1;
      case Key::PageUp:    g->PageUp();      return 1;
      case Key::PageDown:  g->PageDown();    return 1;
      case Key::Home:      g->Home();        return 1;
      case Key::End:       g->End();         return 1;
      case Key::Tab:       g->Tab();         return 1;
      case Key::Escape:    g->Escape();      return 1;
    }

    if (cmd_down) { g->events.total--; return 0; }
    if (key >= 128) { g->events.total--; /* ERROR("unhandled key ", event); */ continue; }

    if (shift_down) key = Key::ShiftModified(key);
    if (ctrl_down)  key = Key::CtrlModified(key);

    g->Input(key);
    return 1;
  }
  return 0;
}

int Input::MouseMove(const point &p, const point &d) {
  int fired = MouseEventDispatch(Mouse::Event::Motion, p, MouseButton1Down());
  screen->events.mouse_move++;
  screen->events.gui += fired;
  if (!app->grab_mode.Enabled()) return fired;
  if (d.x<0) screen->cam->YawLeft  (-d.x); else if (d.x>0) screen->cam->YawRight(d.x);
  if (d.y<0) screen->cam->PitchDown(-d.y); else if (d.y>0) screen->cam->PitchUp (d.y);
  return fired;
}

int Input::MouseWheel(const point &p, const point &d) {
  int fired = MouseEventDispatch(Mouse::Event::Wheel, screen->mouse, d.y);
  screen->events.mouse_wheel++;
  screen->events.gui += fired;
  return fired;
}

int Input::MouseClick(int button, bool down, const point &p) {
  InputEvent::Id event = Mouse::ButtonID(button);
  if      (event == Mouse::Button::_1) mouse_but1_down = down;
  else if (event == Mouse::Button::_2) mouse_but2_down = down;

  int fired = MouseEventDispatch(event, p, down);
  screen->events.mouse_click++;
  screen->events.gui += fired;
  if (fired) return fired;

  for (auto g = screen->input_bind.begin(); g != screen->input_bind.end(); ++g)
    if ((*g)->active) (*g)->Input(event, down);

  return fired;
}

int Input::MouseEventDispatch(InputEvent::Id event, const point &p, int down) {
  if      (event == paste_bind.key)      return KeyEventDispatch(event, down);
  else if (event == Mouse::Event::Wheel) screen->mouse_wheel = p;
  else                                   screen->mouse       = p;

  if (FLAGS_input_debug && down)
    INFO("MouseEvent ", InputEvent::Name(event), " ", screen->mouse.DebugString());

  int fired = 0;
  for (auto g = screen->mouse_gui.begin(); g != screen->mouse_gui.end(); ++g) {
    if ((*g)->NotActive()) continue;
    fired += (*g)->Input(event, (*g)->MousePosition(), down, 0);
  }

  vector<Dialog*> removed;
  Dialog *bring_to_front = 0;
  for (auto i = screen->dialogs.begin(); i != screen->dialogs.end(); /**/) {
    Dialog *gui = (*i);
    if (gui->NotActive()) { i++; continue; }
    fired += gui->Input(event, screen->mouse, down, 0);
    if (gui->deleted) { delete gui; i = screen->dialogs.erase(i); continue; }
    if (event == Mouse::Event::Button1 && down && gui->BoxAndTitle().within(screen->mouse)) { bring_to_front = *i; break; }
    i++;
  }
  if (bring_to_front) bring_to_front->BringToFront();

  if (FLAGS_input_debug && down) INFO("MouseEvent ", screen->mouse.DebugString(), " fired=", fired, ", guis=", screen->mouse_gui.size());
  return fired;
}

int MouseController::Input(InputEvent::Id event, const point &p, int down, int flag) {
  int fired = 0;
  bool but1 = event == Mouse::Event::Button1;

  for (auto h = hover.begin(); h != hover.end(); /**/) {
    auto e = &hit.data[*h];
    if (!e->deleted && e->box.within(p)) { ++h; continue; }
    h = VectorEraseIterSwapBack(&hover, h);
    if (e->deleted) continue;
    e->val = 0;
    e->CB.Run(p, event, 0);
    events.hover++;
    fired++;
  }

  for (auto e = hit.data.rbegin(), ee = hit.data.rend(); e != ee; ++e) {
    bool thunk = 0, e_hover = e->evtype == Event::Hover;

    if (e->deleted || !e->active || (e_hover && e->val) || 
        (!down && e->evtype == Event::Click && e->CB.type != MouseControllerCallback::CB_COORD)) continue;

    if (e->box.within(p)) {
      if (e->run_only_if_first && fired) continue;
      switch (e->evtype) { 
        case Event::Click: if (but1)         { thunk=1; } break;
        case Event::Drag:  if (but1 && down) { thunk=1; } break;
        case Event::Hover: if ((e->val = 1)) { thunk=1; } break;
      }
    }

    if (thunk) {
      if (FLAGS_input_debug && down) INFO("MouseController::Input ", p.DebugString(), " ", e->box.DebugString());
      if (!e->CB.Run(p, event, e_hover ? 1 : down)) continue;

      if (1)       { events.total++; }
      if (e_hover) { events.hover++; hover.push_back(ForwardIteratorFromReverse(e) - hit.data.begin()); }
      else         { events.click++; }
      fired++;

      if (e->evtype == Event::Drag && down) drag.insert(ForwardIteratorFromReverse(e) - hit.data.begin());
      if (flag) break;
    }
  }

  if (event == Mouse::Event::Motion) { for (auto d : drag) if (hit.data[d].CB.Run(p, event, down)) fired++; }
  else if (!down && but1)            { for (auto d : drag) if (hit.data[d].CB.Run(p, event, down)) fired++; drag.clear(); }

  if (FLAGS_input_debug && down) INFO("MouseController::Input ", screen->mouse.DebugString(), " fired=", fired, ", hitboxes=", hit.data.size());
  return fired;
}

Shell::Shell(AssetMap *AM, SoundAssetMap *SAM, MovieAssetMap *MAM) : assets(AM), soundassets(SAM), movieassets(MAM) {
  command.push_back(Command("quit",       bind(&Shell::quit,         this, _1)));
  command.push_back(Command("cmds",       bind(&Shell::cmds,         this, _1)));
  command.push_back(Command("binds",      bind(&Shell::binds,        this, _1)));
  command.push_back(Command("flags",      bind(&Shell::flags,        this, _1)));
  command.push_back(Command("conscolor",  bind(&Shell::consolecolor, this, _1)));
  command.push_back(Command("clipboard",  bind(&Shell::clipboard,    this, _1)));
  command.push_back(Command("startcmd",   bind(&Shell::startcmd,     this, _1)));
  command.push_back(Command("dldir",      bind(&Shell::dldir,        this, _1)));
  command.push_back(Command("screenshot", bind(&Shell::screenshot,   this, _1)));
  command.push_back(Command("fillmode",   bind(&Shell::fillmode,     this, _1)));
  command.push_back(Command("texmode",    bind(&Shell::texmode,      this, _1)));
  command.push_back(Command("swapaxis",   bind(&Shell::swapaxis,     this, _1)));
  command.push_back(Command("campos",     bind(&Shell::campos,       this, _1)));
  command.push_back(Command("filter",     bind(&Shell::filter,       this, _1)));
  command.push_back(Command("fftfilter",  bind(&Shell::filter,       this, _1)));
  command.push_back(Command("f0",         bind(&Shell::f0,           this, _1)));
  command.push_back(Command("sinth",      bind(&Shell::sinth,        this, _1)));
  command.push_back(Command("play",       bind(&Shell::play,         this, _1)));
  command.push_back(Command("playmovie",  bind(&Shell::playmovie,    this, _1)));
  command.push_back(Command("loadsound",  bind(&Shell::loadsound,    this, _1)));
  command.push_back(Command("loadmovie",  bind(&Shell::loadmovie,    this, _1)));
  command.push_back(Command("copy",       bind(&Shell::copy,         this, _1)));
  command.push_back(Command("snap",       bind(&Shell::snap,         this, _1)));
  command.push_back(Command("writesnap",  bind(&Shell::writesnap,    this, _1)));
  command.push_back(Command("fps",        bind(&Shell::fps,          this, _1)));
  command.push_back(Command("wget",       bind(&Shell::wget,         this, _1)));
  command.push_back(Command("messagebox", bind(&Shell::MessageBox,   this, _1)));
  command.push_back(Command("texturebox", bind(&Shell::TextureBox,   this, _1)));
  command.push_back(Command("edit",       bind(&Shell::Edit,         this, _1)));
  command.push_back(Command("slider",     bind(&Shell::Slider,       this, _1)));
}

Asset      *Shell::asset     (const string &n) { return assets      ? (*     assets)(n) : 0; }
SoundAsset *Shell::soundasset(const string &n) { return soundassets ? (*soundassets)(n) : 0; }
MovieAsset *Shell::movieasset(const string &n) { return movieassets ? (*movieassets)(n) : 0; }

bool Shell::FGets() {
  char buf[1024];
  if (!LFL::FGets(buf, sizeof(buf))) return false;
  ChompNewline(buf, strlen(buf));
  Shell::Run(buf);
  return true;
}

void Shell::Run(const string &text) {
  if (!MainThread()) return RunInMainThread(new Callback(bind(&Shell::Run, this, text)));

  string cmd;
  vector<string> arg;
  Split(text, isspace, isquote, &arg);
  if (arg.size()) { cmd = arg[0]; arg.erase(arg.begin()); }
  if (cmd.empty()) return;

  for (auto i = command.begin(); i != command.end(); ++i) {
    if (StringEquals(i->name, cmd)) {
      i->cb(arg);
      return;
    }
  }

  FlagMap *flags = Singleton<FlagMap>::Get();
  for (auto i = flags->flagmap.begin(); i != flags->flagmap.end(); ++i) {
    Flag *flag = (*i).second;
    if (StringEquals(flag->name, cmd)) {
      flag->Update(arg.size() ? arg[0].c_str() : "");
      INFO(flag->name, " = ", flag->Get());
      return;
    }
  }
  INFO("unkown cmd '", cmd, "'");
}

void Shell::mousein (const vector<string>&) { Mouse::GrabFocus(); }
void Shell::mouseout(const vector<string>&) { Mouse::ReleaseFocus(); }

void Shell::quit(const vector<string>&) { app->run = false; }
void Shell::console(const vector<string>&) { screen->console->Toggle(); }
void Shell::showkeyboard(const vector<string>&) { TouchDevice::OpenKeyboard(); }

void Shell::clipboard(const vector<string> &a) {
  if (a.empty()) INFO(Clipboard::Get());
  else Clipboard::Set(Join(a, " "));
}

void Shell::consolecolor(const vector<string>&) {
  delete screen->console->font;
  screen->console->font = Fonts::Get(FLAGS_default_font, "", 9, Color::black);
}

void Shell::startcmd(const vector<string> &a) {
  if (a.empty()) return;
  screen->console->startcmd = Join(a, " ");
}

void Shell::dldir(const vector<string>&) { INFO(LFAppDownloadDir()); }

void Shell::screenshot(const vector<string> &a) {
  if (a.empty()) return INFO("usage: screenshot <file> [tex_id]");
  Texture tex;
  if (a.size() == 1) tex.Screenshot();
  else               tex.DumpGL(atoi(a[1]));
  LocalFile lf(a[0], "w");
  PngWriter::Write(&lf, tex);
}

void Shell::fillmode(const vector<string>&) {
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
  // glPolygonMode(GL_FRONT_AND_BACK, app->fillMode.next());
#endif
}

void Shell::grabmode(const vector<string> &a) { if (app->grab_mode.Next()) mousein(a); else mouseout(a); }
void Shell::texmode(const vector<string>&) { if (app->tex_mode.Next()) screen->gd->EnableTexture(); else screen->gd->DisableTexture(); }
void Shell::swapaxis(const vector<string>&) { screen->SwapAxis(); }

void Shell::campos(const vector<string>&) {
  INFO("camMain.pos=",  screen->cam->pos.DebugString(),
       " camMain.ort=", screen->cam->ort.DebugString(),
       " camMain.up=",  screen->cam->up .DebugString());
}

void Shell::snap(const vector<string> &arg) {
  Asset      *a  = asset     (arg.size() ? arg[0] : "snap"); 
  SoundAsset *sa = soundasset(arg.size() ? arg[0] : "snap");
  if (a && sa) { app->audio.Snapshot(sa); glSpectogram(sa, a); }
}

void Shell::play(const vector<string> &arg) {
  SoundAsset *sa     = arg.size() > 0 ? soundasset(arg[0]) : soundasset("snap");
  int         offset = arg.size() > 1 ?       atoi(arg[1]) : -1;
  int         len    = arg.size() > 2 ?       atoi(arg[2]) : -1;
  if (sa) app->audio.QueueMix(sa, MixFlag::Reset, offset, len);
}

void Shell::playmovie(const vector<string> &arg) {
  MovieAsset *ma = arg.size() ? movieasset(arg[0]) : 0;
  if (ma) ma->Play(0);
}

void Shell::loadsound(const vector<string> &arg) {
  static int id = 1;
  if (arg.empty()) return;
  SoundAsset *a = new SoundAsset();
  a->filename = arg[0];
  a->name = StrCat("sa", id++);
  a->Load();
  INFO("loaded ", a->name);
}

void Shell::loadmovie(const vector<string> &arg) {
  static int id = 1;
  if (arg.empty()) return;
  MovieAsset *ma = new MovieAsset();
  SoundAsset *a = &ma->audio;
  ma->audio.name = ma->video.name = StrCat("ma", id++);
  ma->Load(arg[0].c_str());
  INFO("loaded ", ma->name);
  if (a->wav) INFO("loaded ", a->name, " : ", a->filename, " chans=", a->channels, " sr=", a->sample_rate, " ", a->wav->ring.size);
}

void Shell::copy(const vector<string> &arg) {
  SoundAsset *src = 0, *dst = 0;
  if (!(src = app->shell.soundasset(arg.size() > 0 ? arg[0] : "")) ||
      !(dst = app->shell.soundasset(arg.size() > 1 ? arg[1] : ""))) { INFO("copy <src> <dst>"); return; }

  INFOf("copy %s %d %d %d %s %d %d %d",
        src->name.c_str(), src->sample_rate, src->channels, src->seconds,
        dst->name.c_str(), dst->sample_rate, dst->channels, dst->seconds);

  RingBuf::Handle srch(src->wav), dsth(dst->wav);
  dsth.CopyFrom(&srch);
}

void shell_filter(const vector<string> &arg, bool FFTfilter, int taps, int hop=0) {
  SoundAsset *sa=0; vector<double> filter; double cutoff=0;

  if (arg.size() > 0) sa     = app->shell.soundasset(arg[0]);
  if (arg.size() > 2) cutoff = atof(arg[2]);
  if (arg.size() > 1) {
    filter.resize(taps);
    if        (arg[1] == "low") {
      for (int i=0; i<taps; i++) filter[i] = LowPassFilter(taps, i, (int)cutoff);
    } else if (arg[1] == "high") {
      for (int i=0; i<taps; i++) filter[i] = HighPassFilter(taps, i, (int)cutoff);
    } else if (arg[1] == "preemph") {
      taps = 2;
      filter = PreEmphasisFilter();
    }
  }
  if (arg.size() > 3) { taps = atoi(arg[3]); hop = taps/2; }

  if (!sa || filter.empty() || !taps) {
    INFO("filter <asset> <low,high> <cutoff> [taps]");
    return;
  }

  RingBuf filtered(sa->wav->samples_per_sec, sa->wav->ring.size);
  RingBuf::Handle I(sa->wav), O(&filtered);

  if (FFTfilter) {
    FFTFilterCompile(taps, &filter[0]);
    if (FFTFilter(&I, &O, taps, hop, &filter[0])) return;
  }
  else {
    if (LFL::Filter(&I, &O, taps, &filter[0])) return;
  }

  if (1) {
    int N=20; string b="input = ";
    for (int i=0; i<N; i++) StringAppendf(&b, "x[%d]=%f, ", i, I.Read(i));
    INFO(b);

    b = "output = ";
    for (int i=0; i<N; i++) StringAppendf(&b, "y[%d]=%f, ", i, O.Read(i));
    INFO(b);
  }

  app->audio.QueueMixBuf(&O);
}

void Shell::filter   (const vector<string> &arg) { shell_filter(arg, false, 16); }
void Shell::fftfilter(const vector<string> &arg) { shell_filter(arg, true, FLAGS_feat_window, FLAGS_feat_hop); }

void Shell::f0(const vector<string> &arg) {
  SoundAsset *sa=0; int offset=0; int method=F0EstmMethod::Default;

  if (arg.size() > 0) sa = app->shell.soundasset(arg[0]);
  if (arg.size() > 1) offset = atoi(arg[1]);
  if (arg.size() > 2) method = atoi(arg[2]);

  if (!sa || !sa->wav || sa->wav->ring.size < offset+FLAGS_feat_window) {
    INFO("f0 <asset> <offset>");
    return;
  }

  if (offset) {
    RingBuf::Handle I(sa->wav, offset);
    float f0 = FundamentalFrequency(&I, FLAGS_feat_window, 0, method);
    INFO("f0 = (", sa->name, ":", offset, ") = ", f0);    
  }
  else {
    RingBuf::Handle I(sa->wav, offset);
    Matrix *f0 = F0Stream(&I, 0, FLAGS_feat_window, FLAGS_feat_hop, method);
    for (int i=0; i<f0->M; /**/) {
      char buf[1024]; int len=0;
      for (int j=0; j<20 && i<f0->M; j++,i++) len += sprint(buf+len, sizeof(buf)-len, "%.2f, ", f0->row(i)[0]);
      INFO(buf);
    }
  }
}

void Shell::sinth(const vector<string> &a) { 
  int hz[3] = { 440, 0, 0};
  for (int i=0; i<sizeofarray(hz) && i<a.size(); i++) hz[i] = atof(a[i]);
  Sinthesize(&app->audio, hz[0], hz[1], hz[2]);
}

void Shell::writesnap(const vector<string> &a) {
  SoundAsset *sa = app->shell.soundasset(a.size() ? a[0] : "snap");
  if (sa) {
    string filename = StrCat(LFAppDownloadDir(), "snap.wav"); 
    RingBuf::Handle B(sa->wav);
    LocalFile lf(filename, "r");
    WavWriter w(&lf);
    int ret = w.Write(&B);
    INFO("wrote ", filename, " ret ", ret);
  }
}

void Shell::fps(const vector<string>&) { INFO("FPS ", FPS()); }

void Shell::wget(const vector<string> &a) {
  if (a.empty()) return;
  Singleton<HTTPClient>::Get()->WGet(a[0]);
}

void Shell::MessageBox(const vector<string> &a) { Dialog::MessageBox(Join(a, " ")); }
void Shell::TextureBox(const vector<string> &a) { Dialog::TextureBox(a.size() ? a[0] : ""); }

void Shell::Slider(const vector<string> &a) {
  if (a.empty()) { INFO("slider <flag_name> [total] [inc]"); return; }
  string flag_name = a[0];
  float total = a.size() >= 1 ? atof(a[1]) : 0;
  float inc   = a.size() >= 2 ? atof(a[2]) : 0;
  new SliderFlagDialog(flag_name, total ? total : 100, inc ? inc : 1);
}

void Shell::Edit(const vector<string> &a) {
  string s = LocalFile::FileContents(StrCat(app->assetdir, "lfapp_vertex.glsl"));
  new EditorDialog(screen, Fonts::Default(), new BufferFile(s));
}

void Shell::cmds(const vector<string>&) {
  for (auto i = command.begin(); i != command.end(); ++i) INFO(i->name);
}

void Shell::flags(const vector<string>&) { Singleton<FlagMap>::Get()->Print(); }

void Shell::binds(const vector<string>&) { }

}; // namespace LFL
