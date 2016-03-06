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

#include "core/app/app.h"
#include "core/app/bindings/jni.h"
#include <android/log.h>

namespace LFL {
struct AndroidFrameworkModule : public Module {
  bool frame_on_keyboard_input = 0, frame_on_mouse_input = 0;
  int Frame(unsigned clicks) { return app->input->DispatchQueuedInput(frame_on_keyboard_input, frame_on_mouse_input); }
};

const int Key::Escape     = 0xE100;
const int Key::Return     = 10;
const int Key::Up         = 0xE101;
const int Key::Down       = 0xE102;
const int Key::Left       = 0xE103;
const int Key::Right      = 0xE104;
const int Key::LeftShift  = -7;
const int Key::RightShift = -8;
const int Key::LeftCtrl   = 0xE105;
const int Key::RightCtrl  = 0xE106;
const int Key::LeftCmd    = 0xE107;
const int Key::RightCmd   = 0xE108;
const int Key::Tab        = 0xE109;
const int Key::Space      = ' ';
const int Key::Backspace  = '\b';
const int Key::Delete     = -16;
const int Key::Quote      = '\'';
const int Key::Backquote  = '`';
const int Key::PageUp     = 0xE10A;
const int Key::PageDown   = 0xE10B;
const int Key::F1         = 0xE10C;
const int Key::F2         = 0xE10D;
const int Key::F3         = 0xE10E;
const int Key::F4         = 0xE10F;
const int Key::F5         = 0xE110;
const int Key::F6         = 0xE111;
const int Key::F7         = 0xE112;
const int Key::F8         = 0xE113;
const int Key::F9         = 0xE114;
const int Key::F10        = -30;
const int Key::F11        = -31;
const int Key::F12        = -32;
const int Key::Home       = -33;
const int Key::End        = -34;

extern "C" void AndroidSetFrameOnKeyboardInput(int v) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_keyboard_input = v; }
extern "C" void AndroidSetFrameOnMouseInput   (int v) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_mouse_input    = v; }

string Application::GetClipboardText() { return ""; }
void Application::SetClipboardText(const string &s) {}
int  Application::SetExtraScale(bool v) {}
int  Application::SetMultisample(bool v) {}
void Application::OpenTouchKeyboard()  { AndroidShowOrHideKeyboard(1); }
void Application::CloseTouchKeyboard() { AndroidShowOrHideKeyboard(0); }
void Application::CloseTouchKeyboardAfterReturn(bool v) {} 
Box  Application::GetTouchKeyboardBox() { return Box(); }
void Application::ToggleToolbarButton(const string &n) {}
void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}

struct AndroidVideoModule : public Module {
  int Init() {
    INFO("AndroidVideoModule::Init()");
    if (AndroidVideoInit(&app->opengles_version)) return -1;
    CHECK(!screen->id.v);
    screen->id = MakeTyped(screen);
    app->windows[screen->id.v] = screen;
    return 0;
  }
};

void Application::CloseWindow(Window *W) {}
void Application::MakeCurrentWindow(Window *W) {}

bool Video::CreateWindow(Window *W) { return true; }
int Video::Swap() {
  screen->gd->Flush();
  AndroidVideoSwap();
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

unique_ptr<Module> CreateFrameworkModule() { return make_unique<AndroidFrameworkModule>(); }

}; // namespace LFL
