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

struct X11FrameworkModule : public Module {
  Display *display = 0;
  XVisualInfo *vi = 0;

  int Init() {
    GLint dbb = FLAGS_depth_buffer_bits;
    GLint att[] = { GLX_RGBA, GLX_DEPTH_SIZE, dbb, GLX_DOUBLEBUFFER, None };
    if (!(display = XOpenDisplay(NULL))) return ERRORv(-1, "XOpenDisplay");
    if (!(vi = glXChooseVisual(display, 0, att))) return ERRORv(-1, "glXChooseVisual");
    app->scheduler.system_event_socket = ConnectionNumber(display);
    app->scheduler.AddMainWaitSocket(app->focused, app->scheduler.system_event_socket, SocketSet::READABLE);
    SystemNetwork::SetSocketCloseOnExec(app->scheduler.system_event_socket, true);
    INFO("X11VideoModule::Init()");
    return Video::CreateWindow(app->focused) ? 0 : -1;
  }

  int Free() {
    XFree(vi);
    XCloseDisplay(display);
    return 0;
  }

  int Frame(unsigned clicks) {
    Window *screen = app->focused;
    Display *display = GetTyped<Display*>(screen->surface);
    static const Atom delete_win = XInternAtom(display, "WM_DELETE_WINDOW", 0);
    XEvent xev;
    while (XPending(display)) {
      XNextEvent(display, &xev);
      if (app->windows.size() > 1) SetLFAppWindowByID(Void(xev.xany.window));
      switch (xev.type) {
        case XKeyPress:       if (app->input->KeyPress(GetKeyCodeFromXEvent(display, xev), 0, 1)) app->EventDrivenFrame(0); break;
        case KeyRelease:      if (app->input->KeyPress(GetKeyCodeFromXEvent(display, xev), 0, 0)) app->EventDrivenFrame(0); break;
        case ButtonPress:     if (screen && app->input->MouseClick(xev.xbutton.button, 1, point(xev.xbutton.x, screen->height-xev.xbutton.y))) app->EventDrivenFrame(0); break;
        case ButtonRelease:   if (screen && app->input->MouseClick(xev.xbutton.button, 0, point(xev.xbutton.x, screen->height-xev.xbutton.y))) app->EventDrivenFrame(0); break;
        case MotionNotify:    if (screen) { point p(xev.xmotion.x, screen->height-xev.xmotion.y); if (app->input->MouseMove(p, p - screen->mouse)) app->EventDrivenFrame(0); } break;
        case ConfigureNotify: if (screen && (xev.xconfigure.width != screen->width || xev.xconfigure.height != screen->height)) { screen->Reshaped(Box(xev.xconfigure.width, xev.xconfigure.height)); app->EventDrivenFrame(0); } break;
        case ClientMessage:   if (xev.xclient.data.l[0] == delete_win) LFL::app->CloseWindow(screen); break;
        case Expose:          app->EventDrivenFrame(0);
        default:              continue;
      }
    }
    return 0;
  }
};

void Application::CloseWindow(Window *W) {
  Display *display = GetTyped<Display*>(W->surface);
  glXMakeCurrent(display, None, NULL);
  glXDestroyContext(display, GetTyped<GLXContext>(W->gl));
  XDestroyWindow(display, ::Window(W->id.v));
  windows.erase(W->id.v);
  if (windows.empty()) run = false;
  if (window_closed_cb) window_closed_cb(W);
  focused = nullptr;
  if (windows.size() == 1) SetLFAppWindow(windows.begin()->second);
}

void Application::MakeCurrentWindow(Window *W) {
  glXMakeCurrent(GetTyped<Display*>(W->surface), ::Window(W->id.v), GetTyped<GLXContext>(W->gl));
}

void Application::ReleaseMouseFocus() {}
void Application::GrabMouseFocus() {}

void Application::SetClipboardText(const string &s) {}
string Application::GetClipboardText() { return string(); }
void Application::ShowSystemContextMenu(const vector<MenuItem>&items) {}

void Application::OpenTouchKeyboard(bool) {}
void Application::SetTouchKeyboardTiled(bool v) {}
void Application::SetAutoRotateOrientation(bool v) {}
void Application::SetDownScale(bool) {}

void Window::SetCaption(const string &v) {}
void Window::SetResizeIncrements(float x, float y) {}
void Window::SetTransparency(float v) {}
bool Window::Reshape(int w, int h) {
  auto fw = dynamic_cast<X11FrameworkModule*>(app->framework.get());
  XWindowChanges resize;
  resize.width = w;
  resize.height = h;
  XConfigureWindow(fw->display, ::Window(id.v), CWWidth|CWHeight, &resize);
  return true;
}

bool Video::CreateWindow(Window *W) {
  auto *fw = dynamic_cast<X11FrameworkModule*>(app->framework.get());
  ::Window root = DefaultRootWindow(fw->display);
  XSetWindowAttributes swa;
  swa.colormap = XCreateColormap(fw->display, root, fw->vi->visual, AllocNone);
  swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | PointerMotionMask |
    ButtonPressMask | ButtonReleaseMask | StructureNotifyMask;
  ::Window win = XCreateWindow(fw->display, root, 0, 0, W->width, W->height, 0, fw->vi->depth,
                               InputOutput, fw->vi->visual, CWColormap | CWEventMask, &swa);
  Atom protocols[] = { XInternAtom(fw->display, "WM_DELETE_WINDOW", 0) };
  XSetWMProtocols(fw->display, win, protocols, sizeofarray(protocols));
  if (!(W->id.v = Void(win))) return ERRORv(false, "XCreateWindow");
  XMapWindow(fw->display, win);
  XStoreName(fw->display, win, W->caption.c_str());
  GLXContext share = app->windows.size() ? GetTyped<GLXContext>(app->windows.begin()->second->gl) : nullptr;
  if (!(W->gl = MakeTyped(glXCreateContext(fw->display, fw->vi, share, GL_TRUE))).v)
    return ERRORv(false, "glXCreateContext");
  W->surface = MakeTyped(fw->display);
  app->windows[W->id.v] = W;
  app->MakeCurrentWindow(W);
  return true;
}

void Video::StartWindow(Window*) {}
int Video::Swap() {
  Window *screen = app->focused;
  screen->gd->Flush();
  glXSwapBuffers(GetTyped<Display*>(screen->surface), ::Window(screen->id.v));
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void *Video::BeginGLContextCreate(Window *W) {}
void *Video::CompleteGLContextCreate(Window *W, void *gl_context) {
  X11VideoModule *video = dynamic_cast<X11VideoModule*>(app->video->impl.get());
  GLXContext glc = glXCreateContext(video->display, video->vi, GetTyped<GLXContext>(W->gl), GL_TRUE);
  glXMakeCurrent(video->display, (::Window)(W->id), glc);
  return glc;
}

FrameScheduler::FrameScheduler() :
  maxfps(&FLAGS_target_fps), wakeup_thread(&frame_mutex, &wait_mutex), rate_limit(1), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(1), run_main_loop(1) {}
  
bool FrameScheduler::DoMainWait() {
  unordered_set<Window*> wokeup;
  main_wait_sockets.Select(-1);
  for (auto &s : main_wait_sockets.socket)
    if (main_wait_sockets.GetReadable(s.first))
      if (s.first != system_event_socket) wokeup.insert(static_cast<Window*>(s.second.second));
  for (auto w : wokeup) app->scheduler.Wakeup(w);
  return false;
}

bool FrameScheduler::WakeupIn(Window *w, Time interval, bool force) {}
void FrameScheduler::Wakeup(Window *w) { 
  if (wait_forever && w) {
    XEvent exp;
    exp.type = Expose;
    exp.xexpose.window = ::Window(w->id.v);
    XSendEvent(GetTyped<Display*>(w->surface), exp.xexpose.window, 0, ExposureMask, &exp);
  }
}

void FrameScheduler::UpdateWindowTargetFPS(Window *w) {}
void FrameScheduler::AddMainWaitMouse(Window *w) { }
void FrameScheduler::DelMainWaitMouse(Window *w) {  }
void FrameScheduler::AddMainWaitKeyboard(Window *w) {  }
void FrameScheduler::DelMainWaitKeyboard(Window *w) {  }
void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()>) {
  if (fd == InvalidSocket) return;
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, w);
  main_wait_sockets.Add(fd, flag, w);
}
void FrameScheduler::DelMainWaitSocket(Window *w, Socket fd) {
  if (fd == InvalidSocket) return;
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
  main_wait_sockets.Del(fd);
}

Application *CreateApplication(int ac, const char* const* av) { return new Application(ac, av); }
unique_ptr<Module> CreateFrameworkModule() { return make_unique<X11FrameworkModule>(); }

extern "C" int main(int argc, const char* const* argv) {
  MyAppCreate(argc, argv);
  return MyAppMain();
}

}; // namespace LFL
