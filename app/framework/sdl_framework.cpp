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

#include "SDL.h"
extern "C" {
#ifdef LFL_ANDROID
#include "SDL_androidvideo.h"
#endif
#ifdef LFL_IPHONE
#include "SDL_uikitkeyboard.h"
#endif
};
#ifdef EMSCRIPTEN
#include <emscripten/emscripten.h>
#endif

namespace LFL {
const int Key::Escape     = SDLK_ESCAPE;
const int Key::Return     = SDLK_RETURN;
const int Key::Up         = SDLK_UP;
const int Key::Down       = SDLK_DOWN;
const int Key::Left       = SDLK_LEFT;
const int Key::Right      = SDLK_RIGHT;
const int Key::LeftShift  = SDLK_LSHIFT;
const int Key::RightShift = SDLK_RSHIFT;
const int Key::LeftCtrl   = SDLK_LCTRL;
const int Key::RightCtrl  = SDLK_RCTRL;
const int Key::LeftCmd    = SDLK_LGUI;
const int Key::RightCmd   = SDLK_RGUI;
const int Key::Tab        = SDLK_TAB;
const int Key::Space      = SDLK_SPACE;
const int Key::Backspace  = SDLK_BACKSPACE;
const int Key::Delete     = SDLK_DELETE;
const int Key::Quote      = SDLK_QUOTE;
const int Key::Backquote  = SDLK_BACKQUOTE;
const int Key::PageUp     = SDLK_PAGEUP;
const int Key::PageDown   = SDLK_PAGEDOWN;
const int Key::F1         = SDLK_F1;
const int Key::F2         = SDLK_F2;
const int Key::F3         = SDLK_F3;
const int Key::F4         = SDLK_F4;
const int Key::F5         = SDLK_F5;
const int Key::F6         = SDLK_F6;
const int Key::F7         = SDLK_F7;
const int Key::F8         = SDLK_F8;
const int Key::F9         = SDLK_F9;
const int Key::F10        = SDLK_F10;
const int Key::F11        = SDLK_F11;
const int Key::F12        = SDLK_F12;
const int Key::Home       = SDLK_HOME;
const int Key::End        = SDLK_END;

struct SDLFrameworkModule : public Module {
  int Init() {
    INFO("SDLFrameworkModule::Init");
    int SDL_Init_Flag = 0;
    SDL_Init_Flag |= (FLAGS_enable_video ? SDL_INIT_VIDEO : 0);
    SDL_Init_Flag |= (FLAGS_enable_audio ? SDL_INIT_AUDIO : 0);
    if (SDL_Init(SDL_Init_Flag) < 0) return ERRORv(-1, "SDL_Init: ", SDL_GetError());
    CHECK(Video::CreateWindow(screen));
    app->MakeCurrentWindow(screen);
    SDL_GL_SetSwapInterval(1);
    // SDL_StartTextInput();
    return 0;
  }

  int Free() {
    SDL_Quit();
    return 0;
  }

  int Frame(unsigned clicks) {
    bool mouse_moved = false;
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_QUIT) app->run = false;
      else if (ev.type == SDL_WINDOWEVENT) {
#ifndef LFL_EMSCRIPTEN
        if (ev.window.event == SDL_WINDOWEVENT_FOCUS_GAINED ||
            ev.window.event == SDL_WINDOWEVENT_SHOWN ||
            ev.window.event == SDL_WINDOWEVENT_RESIZED ||
            ev.window.event == SDL_WINDOWEVENT_CLOSE) {
          CHECK((screen = app->GetWindow(Void(size_t(ev.window.windowID)))));
          app->MakeCurrentWindow(screen);
        }
#endif
        if      (ev.window.event == SDL_WINDOWEVENT_RESIZED) screen->Reshape(ev.window.data1, ev.window.data2);
        else if (ev.window.event == SDL_WINDOWEVENT_CLOSE) app->CloseWindow(screen);
      }
      else if (ev.type == SDL_KEYDOWN) {
        app->input->KeyPress(GetKey(ev.key.keysym.sym), 1);
      } else if (ev.type == SDL_KEYUP) {
        app->input->KeyPress(GetKey(ev.key.keysym.sym), 0);
      } else if (ev.type == SDL_MOUSEMOTION) {
        app->input->MouseMove(Input::TransformMouseCoordinate(point(ev.motion.x, ev.motion.y)),
                              point(ev.motion.xrel, -ev.motion.yrel));
        mouse_moved = true;
      }
      else if (ev.type == SDL_MOUSEBUTTONDOWN) app->input->MouseClick(ev.button.button, 1, Input::TransformMouseCoordinate(point(ev.button.x, ev.button.y)));
      else if (ev.type == SDL_MOUSEBUTTONUP)   app->input->MouseClick(ev.button.button, 0, Input::TransformMouseCoordinate(point(ev.button.x, ev.button.y)));
      // else if (ev.type == SDL_ACTIVEEVENT && ev.active.state & SDL_APPACTIVE) { if ((minimized = ev.active.gain)) return 0; }
    }

#if 0 // ndef __APPLE__
    if (mouse_moved && screen->cursor_grabbed) {
      SDL_WarpMouseInWindow(GetTyped<SDL_Window*>(screen->id), screen->width/2, screen->height/2);
      while(SDL_PollEvent(&ev)) { /* do nothing */ }
    }
#endif
    return 0;
  }

  int GetKey(int sym) {
#ifdef LFL_EMSCRIPTEN
    switch (sym) {
      case SDL_SCANCODE_LSHIFT: return Key::LeftShift;
      case SDL_SCANCODE_RSHIFT: return Key::RightShift;
      default:                  break;
    }
    return SDL_GetKeyFromScancode(SDL_ScanCode(sym));
#else
    return sym;
#endif
  }
};

void Window::SetCaption(const string &v) {}
void Window::SetResizeIncrements(float x, float y) {}
void Window::SetTransparency(float v) {}
void Window::Reshape(int w, int h) { SDL_SetWindowSize(GetTyped<SDL_Window*>(id), w, h); }

void Application::MakeCurrentWindow(Window *W) {
  if (SDL_GL_MakeCurrent(GetTyped<SDL_Window*>(W->id), GetTyped<SDL_GLContext>(W->gl)) < 0)
    ERROR("SDL_GL_MakeCurrent: ", SDL_GetError());
  screen = W; 
}

void Application::CloseWindow(Window *W) {
  auto w = GetTyped<SDL_Window*>(W->id);
  SDL_GL_MakeCurrent(NULL, NULL);
  windows.erase(Void(size_t(SDL_GetWindowID(w))));
  if (windows.empty()) {
    app->run = false;
    SDL_GL_DeleteContext(GetTyped<SDL_GLContext>(W->gl));
  }
  SDL_DestroyWindow(w);
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}

string Application::GetClipboardText() { return SDL_GetClipboardText(); }
void Application::SetClipboardText(const string &s) { SDL_SetClipboardText(s.c_str()); }
void Application::CloseTouchKeyboard() {
#ifdef LFL_IPHONE 
  SDL_iPhoneKeyboardHide((SDL_Window*)screen->id);
#endif
}
void Application::OpenTouchKeyboard() {
#ifdef LFL_IPHONE 
  SDL_iPhoneKeyboardShow((SDL_Window*)screen->id);
#endif
}

int Application::GetVolume() { return 0; }
int Application::GetMaxVolume() { return 0; }
void Application::SetVolume(int v) {}
void Application::ShowAds() {}
void Application::HideAds() {}

void Application::LoseFocus() {}
void Application::GrabMouseFocus()    { SDL_ShowCursor(0); SDL_SetWindowGrab(GetTyped<SDL_Window*>(screen->id), SDL_TRUE);  SDL_SetRelativeMouseMode(SDL_TRUE);  app->grab_mode.On();  screen->cursor_grabbed=true; }
void Application::ReleaseMouseFocus() { SDL_ShowCursor(1); SDL_SetWindowGrab(GetTyped<SDL_Window*>(screen->id), SDL_FALSE); SDL_SetRelativeMouseMode(SDL_FALSE); app->grab_mode.Off(); screen->cursor_grabbed=false; }

bool Video::CreateWindow(Window *W) {
  int createflag = SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
#ifdef LFL_MOBILE
  createflag |= SDL_WINDOW_BORDERLESS;
  int bitdepth[] = { 5, 6, 5 };
#else
  int bitdepth[] = { 8, 8, 8 };
#endif
  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, bitdepth[0]);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, bitdepth[1]);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, bitdepth[2]);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, FLAGS_depth_buffer_bits);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  if (!(W->id = MakeTyped(SDL_CreateWindow(W->caption.c_str(), SDL_WINDOWPOS_UNDEFINED,
                                           SDL_WINDOWPOS_UNDEFINED, W->width, W->height, createflag))).v)
    return ERRORv(false, "SDL_CreateWindow: ", SDL_GetError());

  auto w = GetTyped<SDL_Window*>(W->id);
  if (!app->windows.empty()) W->gl = app->windows.begin()->second->gl;
  else if (!(W->gl = MakeTyped(SDL_GL_CreateContext(w))).v)
    return ERRORv(false, "SDL_GL_CreateContext: ", SDL_GetError());

  SDL_Surface* icon = SDL_LoadBMP(Asset::FileName("icon.bmp").c_str());
  SDL_SetWindowIcon(w, icon);
  app->windows[Void(size_t(SDL_GetWindowID(w)))] = W;
  return true;
}

void Video::StartWindow(Window *W) {}
int Video::Swap() {
  screen->gd->Flush();
  SDL_GL_SwapWindow(GetTyped<SDL_Window*>(screen->id));
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void FrameScheduler::Setup() {
#ifdef EMSCRIPTEN
  run_main_loop = false;
#endif
}

bool FrameScheduler::DoWait() { return SDL_WaitEvent(NULL); }
void FrameScheduler::UpdateWindowTargetFPS(Window *w) {}
void FrameScheduler::Wakeup(void *opaque) {
  if (wait_forever && screen && wait_forever_thread) {
    static int my_event_type = SDL_RegisterEvents(1);
    CHECK_GE(my_event_type, 0);
    SDL_Event event;
    SDL_zero(event);
    event.type = my_event_type;
    SDL_PushEvent(&event);
  }
}

void FrameScheduler::AddWaitForeverMouse() {}
void FrameScheduler::DelWaitForeverMouse() {}
void FrameScheduler::AddWaitForeverKeyboard() {}
void FrameScheduler::DelWaitForeverKeyboard() {}
void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) { if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, val); }
void FrameScheduler::DelWaitForeverSocket(Socket fd) { if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd); }

unique_ptr<Module> CreateFrameworkModule() { return make_unique<SDLFrameworkModule>(); }

extern "C" int main(int argc, const char* const* argv) {
  MyAppCreate();
  int ret = MyAppMain(argc, argv);
  if (ret) return ret;
#ifdef EMSCRIPTEN
  emscripten_set_main_loop(LFAppTimerDrivenFrame, 0, 0);
#endif
  return 0;
}

}; // namespace LFL
