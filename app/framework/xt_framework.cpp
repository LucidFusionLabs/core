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

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <X11/IntrinsicP.h>
#include <X11/ShellP.h>
#include <GL/glx.h>

namespace LFL {
static const int XKeyPress = KeyPress, XButton1 = Button1;
#undef KeyPress
#undef Button1
const int Key::Escape = XK_Escape;
const int Key::Return = XK_Return;
const int Key::Up = XK_Up;
const int Key::Down = XK_Down;
const int Key::Left = XK_Left;
const int Key::Right = XK_Right;
const int Key::LeftShift = XK_Shift_L;
const int Key::RightShift = XK_Shift_R;
const int Key::LeftCtrl = XK_Control_L;
const int Key::RightCtrl = XK_Control_R;
const int Key::LeftCmd = XK_Alt_L;
const int Key::RightCmd = XK_Alt_R;
const int Key::Tab = XK_Tab;
const int Key::Space = ' ';
const int Key::Backspace = XK_BackSpace;
const int Key::Delete = XK_Delete;
const int Key::Quote = '\'';
const int Key::Backquote = '`';
const int Key::PageUp = XK_Page_Up;
const int Key::PageDown = XK_Page_Down;
const int Key::F1 = XK_F1;
const int Key::F2 = XK_F2;
const int Key::F3 = XK_F3;
const int Key::F4 = XK_F4;
const int Key::F5 = XK_F5;
const int Key::F6 = XK_F6;
const int Key::F7 = XK_F7;
const int Key::F8 = XK_F8;
const int Key::F9 = XK_F9;
const int Key::F10 = XK_F10;
const int Key::F11 = XK_F11;
const int Key::F12 = XK_F12;
const int Key::Home = XK_Home;
const int Key::End = XK_End;
static int GetKeyCodeFromXEvent(Display *display, const XEvent &xev) {
  return XKeycodeToKeysym(display, xev.xkey.keycode, xev.xkey.state & ShiftMask);
}

struct XTInputModule : public InputModule {
  int Frame(unsigned clicks) {
    Display *display = static_cast<Display*>(screen->surface);
    XEvent xev;
    while (XtAppPending) {
      XtAppNextEvent();
    }
    return 0;
  }
};
void Application::SetClipboardText(const string &s) {}
string Application::GetClipboardText() {}
void Application::ReleaseMouseFocus() {}
void Application::GrabMouseFocus() {}

struct XTVideoModule : public Module {
  Widget toplevel = 0;
  char *argv[2] = { 0, 0 };
  int argc = 1;

  int Init() {
    XtAppContext xt_app;
    argv[0] = &app->progname[0];
    toplevel = XtOpenApplication(&xt_app, screen->caption.c_str(), NULL, 0, &argc, argv,
                                 NULL, applicationShellWidgetClass, NULL, 0);
    INFO("XTideoModule::Init()");
    return app->CreateWindow(screen) ? 0 : -1;
  }
};
bool Application::CreateWindow(Window *W) {
  XTVideoModule *video = dynamic_cast<XTVideoModule*>(app->video->impl.get());
  W->surface = XtDisplay((::Widget)W->impl);
  W->id = XmCreateFrame(video->toplevel, "frame", NULL, 0);
  W->impl = video->toplevel;
  return true;
}
void Application::CloseWindow(Window *W) {}
void Application::MakeCurrentWindow(Window *W) {}

extern "C" void *LFAppCreatePlatformModule() { return new IPhoneVideoModule(); }

}; // namespace LFL
