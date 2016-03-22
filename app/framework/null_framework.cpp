/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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

struct NullFrameworkModule : public Module {
  int Init() {
    INFO("NullFrameworkModule::Init()");
    screen->id = MakeTyped(screen);
    app->windows[screen->id.v] = screen;
    return 0;
  }
};

string Application::GetClipboardText() { return ""; }
void Application::SetClipboardText(const string &s) {}
void Application::LoseFocus() {}
void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}
void Application::OpenTouchKeyboard() {}
void Application::LaunchNativeContextMenu(const vector<MenuItem>&items) {}
void Application::MakeCurrentWindow(Window *W) {}
void Application::CloseWindow(Window *W) {
  windows.erase(W->id.v);
  if (windows.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}

void Window::SetCaption(const string &v) {}
void Window::SetResizeIncrements(float x, float y) {}
void Window::SetTransparency(float v) {}
void Window::Reshape(int w, int h) {}

bool Video::CreateWindow(Window *W) { 
  app->windows[W->id.v] = W;
  return true;
}
void Video::StartWindow(Window*) {}
int Video::Swap() { return 0; }

bool FrameScheduler::DoWait() { return false; }
void FrameScheduler::Setup() { rate_limit = synchronize_waits = wait_forever_thread = 0; }
void FrameScheduler::Wakeup(void *opaque) {}
void FrameScheduler::UpdateWindowTargetFPS(Window*) {}
void FrameScheduler::AddWaitForeverMouse() {}
void FrameScheduler::DelWaitForeverMouse() {}
void FrameScheduler::AddWaitForeverKeyboard() {}
void FrameScheduler::DelWaitForeverKeyboard() {}
void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) {}
void FrameScheduler::DelWaitForeverSocket(Socket fd) {}

extern "C" int main(int argc, const char *argv[]) {
  MyAppCreate();
  return MyAppMain(argc, argv);
}

unique_ptr<Module> CreateFrameworkModule() { return unique_ptr<NullFrameworkModule>(); }

}; // namespace LFL
