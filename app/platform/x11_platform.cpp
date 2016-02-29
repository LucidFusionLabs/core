/*
 * $Id: video.cpp 1336 2014-12-08 09:29:59Z justin $
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
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/glx.h>

namespace LFL {
struct X11VideoModule : public Module {
  Display *display = 0;
  XVisualInfo *vi = 0;
  int Init() {
    GLint dbb = FLAGS_depth_buffer_bits;
    GLint att[] = { GLX_RGBA, GLX_DEPTH_SIZE, dbb, GLX_DOUBLEBUFFER, None };
    if (!(display = XOpenDisplay(NULL))) return ERRORv(-1, "XOpenDisplay");
    if (!(vi = glXChooseVisual(display, 0, att))) return ERRORv(-1, "glXChooseVisual");
    app->scheduler.system_event_socket = ConnectionNumber(display);
    app->scheduler.AddWaitForeverSocket(app->scheduler.system_event_socket, SocketSet::READABLE, 0);
    SystemNetwork::SetSocketCloseOnExec(app->scheduler.system_event_socket, true);
    INFO("X11VideoModule::Init()");
    return app->CreateWindow(screen) ? 0 : -1;
  }
  int Free() {
    XFree(vi);
    XCloseDisplay(display);
    return 0;
  }
};

bool Application::CreateWindow(Window *W) {
  X11VideoModule *video = dynamic_cast<X11VideoModule*>(app->video->impl.get());
  ::Window root = DefaultRootWindow(video->display);
  XSetWindowAttributes swa;
  swa.colormap = XCreateColormap(video->display, root, video->vi->visual, AllocNone);
  swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | PointerMotionMask |
    ButtonPressMask | ButtonReleaseMask | StructureNotifyMask;
  ::Window win = XCreateWindow(video->display, root, 0, 0, W->width, W->height, 0, video->vi->depth,
                               InputOutput, video->vi->visual, CWColormap | CWEventMask, &swa);
  Atom protocols[] = { XInternAtom(video->display, "WM_DELETE_WINDOW", 0) };
  XSetWMProtocols(video->display, win, protocols, sizeofarray(protocols));
  if (!(W->id = (void*)(win))) return ERRORv(false, "XCreateWindow");
  XMapWindow(video->display, win);
  XStoreName(video->display, win, W->caption.c_str());
  if (!(W->gl = glXCreateContext(video->display, video->vi, NULL, GL_TRUE))) return ERRORv(false, "glXCreateContext");
  W->surface = video->display;
  windows[W->id] = W;
  MakeCurrentWindow(W);
  return true;
}

void Application::CloseWindow(Window *W) {
  Display *display = GetTyped<Display*>(W->surface);
  glXMakeCurrent(display, None, NULL);
  glXDestroyContext(display, GetTyped<GLXContext>(W->gl));
  XDestroyWindow(display, GetTyped<::Window>(W->id));
  windows.erase(W->id);
  if (windows.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}

void Application::MakeCurrentWindow(Window *W) {
  glXMakeCurrent(GetTyped<Display*>(W->surface), GetTyped<::Window>(W->id), GetTyped<GLXContext>(W->gl));
}

void Window::Reshape(int w, int h) {
  X11VideoModule *video = dynamic_cast<X11VideoModule*>(app->video->impl.get());
  XWindowChanges resize;
  resize.width = w;
  resize.height = h;
  XConfigureWindow(video->display, GetTyped<::Window*>(id), CWWidth|CWHeight, &resize);
}

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

struct X11InputModule : public InputModule {
  int Frame(unsigned clicks) {
    Display *display = static_cast<Display*>(screen->surface);
    static const Atom delete_win = XInternAtom(display, "WM_DELETE_WINDOW", 0);
    XEvent xev;
    while (XPending(display)) {
      XNextEvent(display, &xev);
      switch (xev.type) {
        case XKeyPress:       if (KeyPress(GetKeyCodeFromXEvent(display, xev), 1)) app->EventDrivenFrame(0); break;
        case KeyRelease:      if (KeyPress(GetKeyCodeFromXEvent(display, xev), 0)) app->EventDrivenFrame(0); break;
        case ButtonPress:     if (screen && MouseClick(xev.xbutton.button, 1, xev.xbutton.x, screen->height-xev.xbutton.y)) app->EventDrivenFrame(0); break;
        case ButtonRelease:   if (screen && MouseClick(xev.xbutton.button, 0, xev.xbutton.x, screen->height-xev.xbutton.y)) app->EventDrivenFrame(0); break;
        case MotionNotify:    if (screen) { point p(xev.xmotion.x, screen->height-xev.xmotion.y); if (app->input->MouseMove(p, p - screen->mouse)) app->EventDrivenFrame(0); } break;
        case ConfigureNotify: if (screen && xev.xconfigure.width != screen->width || xev.xconfigure.height != screen->height) { screen->Reshaped(xev.xconfigure.width, xev.xconfigure.height); app->EventDrivenFrame(0); } break;
        case ClientMessage:   if (xev.xclient.data.l[0] == delete_win) WindowClosed(); break;
        case Expose:          app->EventDrivenFrame(0);
        default:              continue;
      }
    }
    return 0;
  }
};

void Application::SetClipboardText(const string &s) {}
string Application::GetClipboardText() {}
void Application::ReleaseMouseFocus() {}
void Application::GrabMouseFocus() {}

int Video::Swap() {
  screen->gd->Flush();
  glXSwapBuffers((Display*)screen->surface, (::Window)screen->id);
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

extern "C" void *LFAppCreatePlatformModule() { return new IPhoneVideoModule(); }

}; // namespace LFL
