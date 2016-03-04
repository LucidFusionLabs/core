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
#include <wx/wx.h>
#include <wx/glcanvas.h>

namespace LFL {
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

string Application::GetClipboardText() { return ""; }
void Application::SetClipboardText(const string &s) {}

}; // namespace LFL
struct LFLWxWidgetCanvas : public wxGLCanvas {
  wxGLContext *context=0;
  NativeWindow *screen=0;
  bool frame_on_keyboard_input=0, frame_on_mouse_input=0;
  virtual ~LFLWxWidgetCanvas() { delete context; }
  LFLWxWidgetCanvas(NativeWindow *s, wxFrame *parent, int *args) :
    wxGLCanvas(parent, wxID_ANY, args, wxDefaultPosition, wxDefaultSize, wxFULL_REPAINT_ON_RESIZE),
    context((wxGLContext*)s->gl), screen(s) {}
  void OnPaint(wxPaintEvent& event) {
    wxPaintDC(this);
    SetCurrent(*context);
    if (LFL::app->run) LFAppFrame();
    else exit(0);
  }
  void OnMouseMove(wxMouseEvent& event) {
    SetNativeWindow(screen);
    LFL::point p = GetMousePosition(event);
    int fired = LFL::app->input.MouseMove(p, p - LFL::screen->mouse);
    if (fired && frame_on_mouse_input) Refresh();
  }
  void OnMouseDown(wxMouseEvent& event) { OnMouseClick(1, true,  GetMousePosition(event)); }
  void OnMouseUp  (wxMouseEvent& event) { OnMouseClick(1, false, GetMousePosition(event)); }
  void OnMouseClick(int button, bool down, const LFL::point &p) {
    SetNativeWindow(screen);
    int fired = LFL::app->input.MouseClick(button, down, p);
    if (fired && frame_on_mouse_input) Refresh();
  }
  void OnKeyDown(wxKeyEvent& event) { OnKeyEvent(GetKeyCode(event), true); }
  void OnKeyUp  (wxKeyEvent& event) { OnKeyEvent(GetKeyCode(event), false); }
  void OnKeyEvent(int key, bool down) {
    SetNativeWindow(screen);
    int fired = key ? KeyPress(key, down) : 0;
    if (fired && frame_on_keyboard_input) Refresh();
  }
  static int GetKeyCode(wxKeyEvent& event) {
    int key = event.GetUnicodeKey();
    if (key == WXK_NONE) key = event.GetKeyCode();
    return key < 256 && isalpha(key) ? ::tolower(key) : key;
  }
  static LFL::point GetMousePosition(wxMouseEvent& event) {
    return LFL::Input::TransformMouseCoordinate(LFL::point(event.GetX(), event.GetY()));
  }
  DECLARE_EVENT_TABLE()
};
BEGIN_EVENT_TABLE(LFLWxWidgetCanvas, wxGLCanvas)
  EVT_PAINT    (LFLWxWidgetCanvas::OnPaint)
  EVT_KEY_DOWN (LFLWxWidgetCanvas::OnKeyDown)
  EVT_KEY_UP   (LFLWxWidgetCanvas::OnKeyUp)
  EVT_LEFT_DOWN(LFLWxWidgetCanvas::OnMouseDown)
  EVT_LEFT_UP  (LFLWxWidgetCanvas::OnMouseUp)
  EVT_MOTION   (LFLWxWidgetCanvas::OnMouseMove)
END_EVENT_TABLE()

struct LFLWxWidgetFrame : public wxFrame {
  LFLWxWidgetCanvas *canvas=0;
  LFLWxWidgetFrame(LFL::Window *w) : wxFrame(NULL, wxID_ANY, wxString::FromUTF8(w->caption.c_str())) {
    int args[] = { WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, FLAGS_depth_buffer_bits, 0 };
    canvas = new LFLWxWidgetCanvas(w, this, args);
    SetClientSize(w->width, w->height);
    wxMenu *menu = new wxMenu;
    menu->Append(wxID_NEW);
    menu->Append(wxID_CLOSE);
    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append(menu, wxT("&Window"));
    SetMenuBar(menuBar);
    if (w->gl) Show();
  }
  void OnClose(wxCommandEvent& event) { Close(true); }
  void OnNewWindow(wxCommandEvent& event) { LFL::app->CreateNewWindow(); }
  wxDECLARE_EVENT_TABLE();
};
wxBEGIN_EVENT_TABLE(LFLWxWidgetFrame, wxFrame)
  EVT_MENU(wxID_NEW, LFLWxWidgetFrame::OnNewWindow)
  EVT_MENU(wxID_CLOSE, LFLWxWidgetFrame::OnClose)
wxEND_EVENT_TABLE()

struct LFLWxWidgetApp : public wxApp {
  virtual bool OnInit() override {
    if (!wxApp::OnInit()) return false;
    vector<string> ab;
    vector<const char *> av;
    for (int i=0; i<argc; i++) {
      ab.push_back(argv[i].utf8_str().data());
      av.push_back(ab.back().c_str());
    }
    av.push_back(0);
    INFOf("WxWidgetsModule::Main argc=%d\n", argc);
    int ret = LFLWxWidgetsMain(argc, &av[0]);
    if (ret) exit(ret);
    INFOf("%s", "WxWidgetsModule::Main done");
    ((wxGLCanvas*)LFL::screen->id)->GetParent()->Show();
    return TRUE;
  }
  int OnExit() override {
    return wxApp::OnExit();
  }
};
#undef main
IMPLEMENT_APP(LFLWxWidgetApp)

namespace LFL {
struct WxWidgetsVideoModule : public Module {
  int Init() {
    INFOf("WxWidgetsVideoModule::Init() %p", screen);
    CHECK(app->CreateWindow(screen));
    return 0;
  }
};
bool Application::CreateWindow(Window *W) {
  if (!windows.empty()) W->gl = windows.begin()->second->gl;
  LFLWxWidgetCanvas *canvas = (new LFLWxWidgetFrame(W))->canvas;
  if ((W->id = canvas)) windows[W->id] = W;
  if (!W->gl) W->gl = canvas->context = new wxGLContext(canvas);
  MakeCurrentWindow(W);
  return true; 
}
void Application::MakeCurrentWindow(Window *W) { 
  LFLWxWidgetCanvas *canvas = (LFLWxWidgetCanvas*)W->id;
  canvas->SetCurrent(*canvas->context);
}
void Application::CloseWindow(Window *W) {
  windows.erase(W->id);
  if (windows.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}
void Window::Reshape(int w, int h) { GetTyped<wxGLCanvas*>(id)->SetSize(w, h); }
void Mouse::GrabFocus()    {}
void Mouse::ReleaseFocus() {}

int Video::Swap() {
  screen->gd->Flush();
  ((wxGLCanvas*)screen->id)->SwapBuffers();
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void FrameScheduler::DoWait() {}
void FrameScheduler::Setup() { rate_limit = synchronize_waits = monolithic_frame = run_main_loop = 0; }
void FrameScheduler::Wakeup(void *opaque) { if (wait_forever && screen && wait_forever_thread) GetTyped<wxGLCanvas*>(screen->id)->Refresh(); }
void FrameScheduler::AddWaitForeverMouse() {}
void FrameScheduler::DelWaitForeverMouse() {}
void FrameScheduler::AddWaitForeverKeyboard() {}
void FrameScheduler::DelWaitForeverKeyboard() {}
void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) { if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, val); }
void FrameScheduler::DelWaitForeverSocket(Socket fd) { if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd); }

extern "C" void *LFAppCreatePlatformModule() { return new IPhoneVideoModule(); }

}; // namespace LFL