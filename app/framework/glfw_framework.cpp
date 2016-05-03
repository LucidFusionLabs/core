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

#include "GLFW/glfw3.h"

namespace LFL {
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
#ifdef LFL_APPLE
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

struct GLFWFrameworkModule : public Module {
  int Init() {
    INFO("GLFWVideoModule::Init");
    CHECK(Video::CreateWindow(screen));
    app->MakeCurrentWindow(screen);
    glfwSwapInterval(1);
    return 0;
  }
  int Free() {
    glfwTerminate();
    return 0;
  }

  int Init(Window *W) { InitWindow(GetTyped<GLFWwindow*>(screen->id)); return 0; }
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
    return 1;
  }

  static void WindowSize (GLFWwindow *W, int w, int h) { if (!LoadScreen(W)) return; screen->Reshaped(w, h); }
  static void WindowClose(GLFWwindow *W)               { if (!LoadScreen(W)) return; app->CloseWindow(screen); }
  static void Key(GLFWwindow *W, int k, int s, int a, int m) {
    if (!LoadScreen(W)) return;
    app->input->KeyPress((unsigned)k < 256 && isalpha((unsigned)k) ? ::tolower((unsigned)k) : k,
                         (a == GLFW_PRESS || a == GLFW_REPEAT));
  }

  static void MouseClick(GLFWwindow *W, int b, int a, int m) {
    if (!LoadScreen(W)) return;
    app->input->MouseClick(MouseButton(b), a == GLFW_PRESS, screen->mouse);
  }

  static void MousePosition(GLFWwindow *W, double x, double y) {
    if (!LoadScreen(W)) return;
    point p = Input::TransformMouseCoordinate(point(x, y));
    app->input->MouseMove(p, p - screen->mouse);
  }

  static void MouseWheel(GLFWwindow *W, double x, double y) {
    if (!LoadScreen(W)) return;
    point p(x, y);
    app->input->MouseWheel(p, p -screen->mouse_wheel);
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

string Application::GetClipboardText()                { return glfwGetClipboardString(GetTyped<GLFWwindow*>(screen->id)); }
void   Application::SetClipboardText(const string &s) {        glfwSetClipboardString(GetTyped<GLFWwindow*>(screen->id), s.c_str()); }
void Application::GrabMouseFocus()    { glfwSetInputMode(GetTyped<GLFWwindow*>(screen->id), GLFW_CURSOR, GLFW_CURSOR_DISABLED); app->grab_mode.On();  screen->cursor_grabbed=true;  }
void Application::ReleaseMouseFocus() { glfwSetInputMode(GetTyped<GLFWwindow*>(screen->id), GLFW_CURSOR, GLFW_CURSOR_NORMAL);   app->grab_mode.Off(); screen->cursor_grabbed=false; }
void Application::OpenTouchKeyboard() {}
void Application::LoseFocus() {}

void Application::MakeCurrentWindow(Window *W) {
  glfwMakeContextCurrent(GetTyped<GLFWwindow*>(W->id));
  screen = W;
}

void Application::CloseWindow(Window *W) {
  windows.erase(W->id.v);
  if (windows.empty()) run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  glfwDestroyWindow(GetTyped<GLFWwindow*>(W->id));
  screen = 0;
}

void Window::SetCaption(const string &v) {}
void Window::SetResizeIncrements(float x, float y) {}
void Window::SetTransparency(float v) {}
void Window::Reshape(int w, int h) { glfwSetWindowSize(GetTyped<GLFWwindow*>(id), w, h); }

bool Video::CreateWindow(Window *W) {
  GLFWwindow *share = app->windows.empty() ? 0 : GetTyped<GLFWwindow*>(app->windows.begin()->second->id);
  if (!(W->id = MakeTyped(glfwCreateWindow(W->width, W->height, W->caption.c_str(), 0, share))).v) return ERRORv(false, "glfwCreateWindow");
  app->windows[W->id.v] = W;
  return true;
}

void Video::StartWindow(Window *W) {}
int Video::Swap() {
  screen->gd->Flush();
  glfwSwapBuffers(GetTyped<GLFWwindow*>(screen->id));
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void FrameScheduler::DoWait() { glfwWaitEvents(); }
void FrameScheduler::Setup() {}
void FrameScheduler::Wakeup(void*) { if (wait_forever && screen && wait_forever_thread) glfwPostEmptyEvent(); }
void FrameScheduler::UpdateWindowTargetFPS(Window *w) {}
void FrameScheduler::AddWaitForeverMouse() {}
void FrameScheduler::DelWaitForeverMouse() {}
void FrameScheduler::AddWaitForeverKeyboard() {}
void FrameScheduler::DelWaitForeverKeyboard() {}
void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) { if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, val); }
void FrameScheduler::DelWaitForeverSocket(Socket fd) { if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd); }

unique_ptr<Module> CreateFrameworkModule() {
  ONCE({ if (FLAGS_enable_video) {
    INFO("LFAppCreatePlatformModule: glfwInit()");
    if (!glfwInit()) return ERRORv(nullptr, "glfwInit: ", strerror(errno));
  }});
  return make_unique<GLFWFrameworkModule>();
}

extern "C" int main(int argc, const char* const* argv) {
  MyAppCreate();
  return MyAppMain(argc, argv);
}

}; // namespace LFL
