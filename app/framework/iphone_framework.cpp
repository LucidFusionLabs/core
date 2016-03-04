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

namespace LFL {
extern "C" int iPhoneSetExtraScale(bool);
extern "C" int iPhoneSetMultisample(bool);
extern "C" void iPhoneShowKeyboard();
extern "C" void iPhoneHideKeyboard();
extern "C" void iPhoneHideKeyboardAfterReturn(bool v);
extern "C" void iPhoneGetKeyboardBox(int *x, int *y, int *w, int *h);
extern "C" void iPhoneCreateToolbar(int n, const char **name, const char **val);
extern "C" void iPhoneToggleToolbarButton(const char *n);

extern "C" void iPhoneTriggerFrame(void*);
extern "C" bool iPhoneTriggerFrameIn(void*, int ms, bool force);
extern "C" void iPhoneClearTriggerFrameIn(void *O);
extern "C" void iPhoneUpdateTargetFPS(void*);
extern "C" void iPhoneAddWaitForeverMouse(void*);
extern "C" void iPhoneDelWaitForeverMouse(void*);
extern "C" void iPhoneAddWaitForeverKeyboard(void*);
extern "C" void iPhoneDelWaitForeverKeyboard(void*);
extern "C" void iPhoneAddWaitForeverSocket(void*, int fd);
extern "C" void iPhoneDelWaitForeverSocket(void*, int fd);

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

string Application::GetClipboardText() { return ""; }
void Application::SetClipboardText(const string &s) {}
int  Applicateion::SetExtraScale(bool v) { return iPhoneSetExtraScale(v); }
int  Applicateion::SetMultisample(bool v) { return iPhoneSetMultisample(v); }
void Application::OpenTouchKeyboard() { iPhoneShowKeyboard(); }
void Application::CloseTouchKeyboard() { iPhoneHideKeyboard(); }
void Application::CloseTouchKeyboardAfterReturn(bool v) { iPhoneHideKeyboardAfterReturn(v); } 
Box  Application::GetTouchKeyboardBox() { Box ret; iPhoneGetKeyboardBox(&ret.x, &ret.y, &ret.w, &ret.h); return ret; }
void Application::ToggleToolbarButton(const string &n) { iPhoneToggleToolbarButton(n.c_str()); }
void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}

extern "C" void iPhoneVideoSwap();
struct IPhoneVideoModule : public Module {
  int Init() {
    INFO("IPhoneVideoModule::Init()");
    CHECK(!screen->id);
    NativeWindowInit();
    NativeWindowSize(&screen->width, &screen->height);
    CHECK(screen->id);
    windows[screen->id] = screen;
    return 0;
  }
};
bool Application::CreateWindow(Window *W) { return false; }
void Application::CloseWindow(Window *W) {}
void Application::MakeCurrentWindow(Window *W) {}

int Video::Swap() {
  screen->gd->Flush();
  iPhoneVideoSwap();
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void FrameScheduler::DoWait() {}
void FrameScheduler::Setup() { rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = 0; }
void FrameScheduler::Wakeup(void*) { iPhoneTriggerFrame(screen->id); }
bool FrameScheduler::WakeupIn(void *opaque, Time interval, bool force) { return iPhoneTriggerFrameIn(screen->id, interval.count(), force); }
void FrameScheduler::ClearWakeupIn() { iPhoneClearTriggerFrameIn(screen->id); }
void FrameScheduler::UpdateWindowTargetFPS(Window *w) { iPhoneUpdateTargetFPS(screen->id); }
void FrameScheduler::AddWaitForeverMouse() { iPhoneAddWaitForeverMouse(screen->id); }
void FrameScheduler::DelWaitForeverMouse() { iPhoneDelWaitForeverMouse(screen->id); }
void FrameScheduler::AddWaitForeverKeyboard() { iPhoneAddWaitForeverKeyboard(screen->id); }
void FrameScheduler::DelWaitForeverKeyboard() { iPhoneDelWaitForeverKeyboard(screen->id); }
void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, val);
  if (!wait_forever_thread) { CHECK_EQ(SocketSet::READABLE, flag); iPhoneAddWaitForeverSocket(screen->id, fd); }
}
void FrameScheduler::DelWaitForeverSocket(Socket fd) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
  CHECK(screen->id);
  iPhoneDelWaitForeverSocket(screen->id, fd);
}

extern "C" void *LFAppCreatePlatformModule() { return new IPhoneVideoModule(); }

}; // namespace LFL
