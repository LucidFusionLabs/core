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
#include <GL/glx.h>
static const int XKeyPress = KeyPress, XButton1 = Button1;
#undef KeyPress
#undef Button1

namespace LFL {
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
const int Key::Insert = XK_Insert;

static int GetKeyCodeFromXEvent(Display *display, const XEvent &xev) {
  return XKeycodeToKeysym(display, xev.xkey.keycode, xev.xkey.state & ShiftMask);
}

struct X11Window : public Window {
  ::Window win;
  Display *surface;
  GLXContext gl;
  bool frame_on_mouse_input=0, frame_on_keyboard_input=0;
  X11Window(Application *a) : Window(a) {}

  bool Reshape(int w, int h) override {
    XWindowChanges resize;
    resize.width = w;
    resize.height = h;
    XConfigureWindow(surface, win, CWWidth|CWHeight, &resize);
    return true;
  }
  
  void Wakeup(int) override {
    if (!parent->scheduler.wait_forever) return;
    XEvent exp;
    exp.type = Expose;
    exp.xexpose.window = win;
    XSendEvent(surface, exp.xexpose.window, 0, ExposureMask, &exp);
  }

  int Swap() override {
    gd->Flush();
    glXSwapBuffers(surface, win);
    gd->CheckForError(__FILE__, __LINE__);
    return 0;
  }

  void SetCaption(const string &v) override {
    XChangeProperty(surface, win,
                    XInternAtom(surface, "_NET_WM_NAME", False),
                    XInternAtom(surface, "UTF8_STRING", False),
                    8, PropModeReplace, MakeUnsigned(v.data()), v.size());
  }

  void SetResizeIncrements(float x, float y) override {}
  void SetTransparency(float v) override {}
};

struct X11Framework : public Framework {
  Application *app;
  Display *display = 0;
  XVisualInfo *vi = 0;
  XErrorEvent error;
  int (*old_error_handler)(Display *, XErrorEvent *);
  X11Framework(Application *a) : app(a) {}

  int Init() override {
    GLint dbb = FLAGS_depth_buffer_bits;
    GLint att[] = { GLX_RGBA, GLX_DEPTH_SIZE, dbb, GLX_DOUBLEBUFFER, None };
    if (!(display = XOpenDisplay(NULL))) return ERRORv(-1, "XOpenDisplay");
    if (!(vi = glXChooseVisual(display, 0, att))) return ERRORv(-1, "glXChooseVisual");
    app->scheduler.system_event_socket = ConnectionNumber(display);
    app->scheduler.AddMainWaitSocket(app->focused, app->scheduler.system_event_socket, SocketSet::READABLE);
    SystemNetwork::SetSocketCloseOnExec(app->scheduler.system_event_socket, true);
    INFO("X11Framework::Init()");
    return CreateWindow(app, app->focused) ? 0 : -1;
  }

  int Free() override {
    XFree(vi);
    XCloseDisplay(display);
    return 0;
  }

  unique_ptr<Window> ConstructWindow(Application *app) override { return make_unique<X11Window>(app); }
  void StartWindow(Window*) override {}

  bool CreateWindow(WindowHolder *H, Window *W) override {
    auto w = dynamic_cast<X11Window*>(W);
    auto fw = dynamic_cast<X11Framework*>(app->framework.get());
    ::Window root = DefaultRootWindow(fw->display);
    XSetWindowAttributes swa;
    swa.colormap = XCreateColormap(fw->display, root, fw->vi->visual, AllocNone);
    swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | PointerMotionMask |
      ButtonPressMask | ButtonReleaseMask | StructureNotifyMask;
    w->win = XCreateWindow(fw->display, root, 0, 0, W->gl_w, W->gl_h, 0, fw->vi->depth,
                           InputOutput, fw->vi->visual, CWColormap | CWEventMask, &swa);
    Atom protocols[] = { XInternAtom(fw->display, "WM_DELETE_WINDOW", 0) };
    XSetWMProtocols(fw->display, w->win, protocols, sizeofarray(protocols));
    if (!(w->id = Void(w->win))) return ERRORv(false, "XCreateWindow");
    XMapWindow(fw->display, w->win);
    XStoreName(fw->display, w->win, w->caption.c_str());
    GLXContext share = app->windows.size() ? dynamic_cast<X11Window*>(app->windows.begin()->second)->gl : nullptr;

    TrapError();
    if (!(w->gl = glXCreateContext(fw->display, fw->vi, share, GL_TRUE))) return ERRORv(false, "glXCreateContext");
    XSync(fw->display, false);
    if (int error_code = UntrapError()) {
      string text(256, 0);
      if (!XGetErrorText(fw->display, error_code, &text[0], text.size())) text.clear();
      return ERRORv(false, StringPrintf("glXCreateContext: error_code=%d %s", error_code, text.data()));
    }

    w->surface = fw->display;
    app->windows[W->id] = W;
    app->MakeCurrentWindow(W);
    return true;
  }

  int Frame(unsigned clicks) override {
    int events = 0;
    auto screen = dynamic_cast<X11Window*>(app->focused);
    auto display = screen->surface;
    static const Atom delete_win = XInternAtom(display, "WM_DELETE_WINDOW", 0);

    XEvent xev;
    while (XPending(display)) {
      XNextEvent(display, &xev);
      if (app->windows.size() > 1) {
        app->SetFocusedWindowByID(Void(xev.xany.window));
        screen = dynamic_cast<X11Window*>(app->focused);
        display = screen->surface;
      }

      switch (xev.type) {
        case XKeyPress:       if (app->input->KeyPress(GetKeyCodeFromXEvent(display, xev), 0, 1) && screen->frame_on_keyboard_input) { screen->frame_pending = true; events++; } break;
        case KeyRelease:      if (app->input->KeyPress(GetKeyCodeFromXEvent(display, xev), 0, 0) && screen->frame_on_keyboard_input) { screen->frame_pending = true; events++; } break;
        case ButtonPress:     if (app->input->MouseClick(xev.xbutton.button, 1, point(xev.xbutton.x, screen->gl_h-xev.xbutton.y)) && screen->frame_on_mouse_input) { screen->frame_pending = true; events++; } break;
        case ButtonRelease:   if (app->input->MouseClick(xev.xbutton.button, 0, point(xev.xbutton.x, screen->gl_h-xev.xbutton.y)) && screen->frame_on_mouse_input) { screen->frame_pending = true; events++; } break;
        case MotionNotify:    { point p(xev.xmotion.x, screen->gl_h-xev.xmotion.y); if (app->input->MouseMove(p, p - screen->mouse) && screen->frame_on_mouse_input) { screen->frame_pending = true; events++; } } break;
        case ConfigureNotify: if (xev.xconfigure.width != screen->gl_w || xev.xconfigure.height != screen->gl_h) { point d(xev.xconfigure.width, xev.xconfigure.height); screen->Reshaped(d, d); screen->frame_pending = true; events++; } break;
        case ClientMessage:   if (xev.xclient.data.l[0] == delete_win) app->CloseWindow(screen); break;
        case Expose:          { screen->frame_pending = true; events++; } break;
        default:              continue;
      }
    }

    return events;
  }

  void TrapError() { trapped_error_code = 0; old_error_handler = XSetErrorHandler(TrapErrorHandler); }
  int UntrapError() { XSetErrorHandler(old_error_handler); return trapped_error_code; }

  static int trapped_error_code;
  static int TrapErrorHandler(Display*, XErrorEvent *error) {
    if (!trapped_error_code) trapped_error_code = error->error_code;
    return 0;
  }
};

int X11Framework::trapped_error_code = 0;

string Clipboard::GetClipboardText() { return string(); }
void Clipboard::SetClipboardText(const string &s) {}

void MouseFocus::GrabMouseFocus() {}
void MouseFocus::ReleaseMouseFocus() {}

void TouchKeyboard::ToggleTouchKeyboard() {}
void TouchKeyboard::OpenTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboardAfterReturn(bool v) {}
void TouchKeyboard::SetTouchKeyboardTiled(bool v) {}

void ThreadDispatcher::RunCallbackInMainThread(Callback cb) {
  message_queue.Write(make_unique<Callback>(move(cb)).release());
  if (!FLAGS_target_fps) wakeup->Wakeup();
}

void WindowHolder::MakeCurrentWindow(Window *W) {
  auto w = dynamic_cast<X11Window*>(W);
  glXMakeCurrent(w->surface, w->win, w->gl);
}

int Application::SetExtraScale(bool v) { return false; }
void Application::SetDownScale(bool v) {}
void Application::SetTitleBar(bool v) {}
void Application::SetKeepScreenOn(bool v) {}
void Application::SetAutoRotateOrientation(bool v) {}
void Application::SetVerticalSwipeRecognizer(int touches) {}
void Application::SetHorizontalSwipeRecognizer(int touches) {}
void Application::SetPanRecognizer(bool enabled) {}
void Application::SetPinchRecognizer(bool enabled) {}
void Application::ShowSystemStatusBar(bool v) {}
void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &choose_cb) {}
void Application::SetTheme(const string &v) {}
int Application::Suspended() { return 0; }

void Application::CloseWindow(Window *W) {
  auto w = dynamic_cast<X11Window*>(W);
  Display *display = w->surface;
  glXMakeCurrent(display, None, NULL);
  glXDestroyContext(display, w->gl);
  XDestroyWindow(display, w->win);
  windows.erase(W->id);
  if (windows.empty()) run = false;
  if (window_closed_cb) window_closed_cb(W);
  focused = nullptr;
  if (windows.size() == 1) SetFocusedWindow(windows.begin()->second);
}

FrameScheduler::FrameScheduler(WindowHolder *w) :
  window(w), maxfps(&FLAGS_target_fps), rate_limit(1), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(1), run_main_loop(1) {}
  
bool FrameScheduler::DoMainWait(bool only_pool) {
  bool ret = false;
  main_wait_sockets.Select(only_pool ? 0 : -1);
  for (auto i = main_wait_sockets.socket.begin(); i != main_wait_sockets.socket.end(); /**/) {
    iter_socket = i->first;
    auto w = static_cast<Window*>(i->second.third);
    auto f = static_cast<function<bool()>*>(i++->second.second);
    if (iter_socket == system_event_socket) ret = true;
    else { if (f) if ((*f)()) { w->frame_pending = true; ret = true; } }
  }
  iter_socket = InvalidSocket;
  return ret;
}

void FrameScheduler::UpdateWindowTargetFPS(Window *w) {}
void FrameScheduler::AddMainWaitMouse(Window *w) { dynamic_cast<X11Window*>(w)->frame_on_mouse_input = 1; }
void FrameScheduler::DelMainWaitMouse(Window *w) { dynamic_cast<X11Window*>(w)->frame_on_mouse_input = 0;  }
void FrameScheduler::AddMainWaitKeyboard(Window *w) { dynamic_cast<X11Window*>(w)->frame_on_keyboard_input = 1; }
void FrameScheduler::DelMainWaitKeyboard(Window *w) { dynamic_cast<X11Window*>(w)->frame_on_keyboard_input = 0; }

void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()> f) {
  if (fd == InvalidSocket) return;
  main_wait_sockets.Add(fd, flag, f ? new function<bool()>(move(f)) : nullptr, w);
}

void FrameScheduler::DelMainWaitSocket(Window *w, Socket fd) {
  if (fd == InvalidSocket) return;
  if (iter_socket != InvalidSocket)
    CHECK_EQ(iter_socket, fd) << "Can only remove current socket from wait callback";
  auto it = main_wait_sockets.socket.find(fd);
  if (it == main_wait_sockets.socket.end()) return;
  if (auto f = static_cast<function<bool()>*>(it->second.second)) delete f;
  main_wait_sockets.Del(fd);
}

unique_ptr<Framework> Framework::Create(Application *a) { return make_unique<X11Framework>(a); }
unique_ptr<TimerInterface> SystemToolkit::CreateTimer(Callback cb) { return nullptr; }
unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(Window *w, AlertItemVec items) { return nullptr; }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(Window*, const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(Window*, const string &title, MenuItemVec items) { return nullptr; }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(Window*, MenuItemVec items) { return nullptr; }

extern "C" int main(int argc, const char* const* argv) {
  auto app = MyAppCreate(argc, argv);
  return MyAppMain(app);
}

}; // namespace LFL
