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

namespace LFL {
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
const int Key::Insert     = -35;

const int Texture::updatesystemimage_pf = Pixel::RGB24;

struct NullWindow : public Window {
  void SetCaption(const string &c) {}
  void SetResizeIncrements(float x, float y) {}
  void SetTransparency(float v) {}
  bool Reshape(int w, int h) { return false; }
};

struct NullFrameworkModule : public Module {
  int Init() {
    INFO("NullFrameworkModule::Init()");
    app->focused->id = app->focused;
    app->windows[app->focused->id] = app->focused;
    return 0;
  }
};

struct NullAlertView : public AlertViewInterface {
  void Hide() {}
  void Show(const string &arg) {}
  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {}
  string RunModal(const string &arg) { return string(); }
};

struct NullMenuView : public MenuViewInterface {
  void Show() {}
};

struct NullPanelView : public PanelViewInterface {
  void Show() {}
  void SetTitle(const string &title) {}
};

struct NullNag : public NagInterface {
};

int Application::Suspended() { return 0; }
void Application::RunCallbackInMainThread(Callback cb) {
  message_queue.Write(new Callback(move(cb)));
  if (!FLAGS_target_fps) scheduler.Wakeup(focused);
}

void Application::MakeCurrentWindow(Window *W) {}
void Application::CloseWindow(Window *W) {
  windows.erase(W->id);
  if (windows.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  app->focused = 0;
}

void Application::LoseFocus() {}
void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}

string Application::GetClipboardText() { return ""; }
void Application::SetClipboardText(const string &s) {}

void Application::SetAppFrameEnabled(bool) {}
void Application::OpenTouchKeyboard() {}
void Application::CloseTouchKeyboard() {}
void Application::ToggleTouchKeyboard() {}
void Application::CloseTouchKeyboardAfterReturn(bool v) {}
void Application::SetTouchKeyboardTiled(bool v) {}
void Application::SetAutoRotateOrientation(bool v) {}
void Application::ShowSystemStatusBar(bool v) {}
void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB&) {}
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB&) {}
void Application::ShowSystemContextMenu(const vector<MenuItem>&items) {}
void Application::SetKeepScreenOn(bool v) {}
void Application::SetTheme(const string &v) {}

bool Video::CreateWindow(Window *W) { 
  app->windows[W->id] = W;
  return true;
}
void Video::StartWindow(Window*) {}
int Video::Swap() { return 0; }

FrameScheduler::FrameScheduler() :
  maxfps(&FLAGS_target_fps), wakeup_thread(&frame_mutex, &wait_mutex), rate_limit(0), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(1), run_main_loop(1) {}

bool FrameScheduler::DoMainWait() { return false; }
void FrameScheduler::Wakeup(Window*, int) {}
void FrameScheduler::UpdateWindowTargetFPS(Window*) {}
void FrameScheduler::AddMainWaitMouse(Window*) {}
void FrameScheduler::DelMainWaitMouse(Window*) {}
void FrameScheduler::AddMainWaitKeyboard(Window*) {}
void FrameScheduler::DelMainWaitKeyboard(Window*) {}
void FrameScheduler::AddMainWaitSocket(Window*, Socket fd, int flag, function<bool()>) {}
void FrameScheduler::DelMainWaitSocket(Window*, Socket fd) {}

extern "C" int main(int argc, const char *argv[]) {
  MyAppCreate(argc, argv);
  return MyAppMain();
}

Window *Window::Create() { return new NullWindow(); }
Application *CreateApplication(int ac, const char* const* av) { return new Application(ac, av); }
unique_ptr<Module> CreateFrameworkModule() { return unique_ptr<NullFrameworkModule>(); }
unique_ptr<TimerInterface> SystemToolkit::CreateTimer(Callback cb) { return nullptr; }
unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(AlertItemVec items) { return make_unique<NullAlertView>(); }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(const string &title, MenuItemVec items) { return make_unique<NullMenuView>(); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(vector<MenuItem> items) { return nullptr; }
unique_ptr<NagInterface> SystemToolkit::CreateNag(const string &id, int min_days, int min_uses, int min_events, int remind_days) { return make_unique<NullNag>(); }

}; // namespace LFL
