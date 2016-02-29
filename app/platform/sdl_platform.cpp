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
#include "SDL.h"
#ifdef LFL_ANDROID
extern "C" {
#include "SDL_androidvideo.h"
#ifdef LFL_IPHONE
#include "SDL_uikitkeyboard.h"
#endif
};
#endif

namespace LFL {
/* struct NativeWindow { SDL_Window* id; SDL_GLContext gl; SDL_Surface *surface; }; */
struct SDLVideoModule : public Module {
  int Init() {
    INFO("SFLVideoModule::Init");
    CHECK(app->CreateWindow(screen));
    app->MakeCurrentWindow(screen);
    SDL_GL_SetSwapInterval(1);
    return 0;
  }
  int Free() {
    SDL_Quit();
    return 0;
  }
};
bool Application::CreateWindow(Window *W) {
  int createFlag = SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
#ifdef LFL_MOBILE
  createFlag |= SDL_WINDOW_BORDERLESS;
  int bitdepth[] = { 5, 6, 5 };
#else
  int bitdepth[] = { 8, 8, 8 };
#endif
  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, bitdepth[0]);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, bitdepth[1]);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, bitdepth[2]);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, FLAGS_depth_buffer_bits);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  if (!(W->id = SDL_CreateWindow(W->caption.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, W->width, W->height, createFlag)))
    return ERRORv(false, "SDL_CreateWindow: ",     SDL_GetError());

  if (!windows.empty()) W->gl = windows.begin()->second->gl;
  else if (!(W->gl = SDL_GL_CreateContext((SDL_Window*)W->id)))
    return ERRORv(false, "SDL_GL_CreateContext: ", SDL_GetError());

  SDL_Surface* icon = SDL_LoadBMP(StrCat(app->assetdir, "icon.bmp").c_str());
  SDL_SetWindowIcon((SDL_Window*)W->id, icon);

  windows[(void*)(long)SDL_GetWindowID((SDL_Window*)W->id)] = W;
  return true;
}
void Application::MakeCurrentWindow(Window *W) {
  if (SDL_GL_MakeCurrent((SDL_Window*)W->id, W->gl) < 0) ERROR("SDL_GL_MakeCurrent: ", SDL_GetError());
  screen = W; 
}
void Application::CloseWindow(Window *W) {
  SDL_GL_MakeCurrent(NULL, NULL);
  windows.erase((void*)(long)SDL_GetWindowID((SDL_Window*)W->id));
  if (windows.empty()) {
    app->run = false;
    SDL_GL_DeleteContext(W->gl);
  }
  SDL_DestroyWindow((SDL_Window*)W->id);
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}
void Window::Reshape(int w, int h) { SDL_SetWindowSize(GetTyped<SDL_Window*>(id), w, h); }

struct SDLInputModule : public InputModule {
  int Frame(unsigned clicks) {
    SDL_Event ev; point mp;
    SDL_GetMouseState(&mp.x, &mp.y);
    bool mouse_moved = false;

    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_QUIT) app->run = false;
      else if (ev.type == SDL_WINDOWEVENT) {
        if (ev.window.event == SDL_WINDOWEVENT_FOCUS_GAINED ||
            ev.window.event == SDL_WINDOWEVENT_SHOWN ||
            ev.window.event == SDL_WINDOWEVENT_RESIZED ||
            ev.window.event == SDL_WINDOWEVENT_CLOSE) {
          CHECK((screen = Window::Get((void*)(long)ev.window.windowID)));
          Window::MakeCurrent(screen);
        }
        if      (ev.window.event == SDL_WINDOWEVENT_RESIZED) screen->Reshape(ev.window.data1, ev.window.data2);
        else if (ev.window.event == SDL_WINDOWEVENT_CLOSE) Window::Close(screen);
      }
      else if (ev.type == SDL_KEYDOWN) app->input->KeyPress(ev.key.keysym.sym, 1);
      else if (ev.type == SDL_KEYUP)   app->input->KeyPress(ev.key.keysym.sym, 0);
      else if (ev.type == SDL_MOUSEMOTION) {
        app->input->MouseMove(mp, point(ev.motion.xrel, ev.motion.yrel));
        mouse_moved = true;
      }
      else if (ev.type == SDL_MOUSEBUTTONDOWN) app->input->MouseClick(ev.button.button, 1, point(ev.button.x, ev.button.y));
      else if (ev.type == SDL_MOUSEBUTTONUP)   app->input->MouseClick(ev.button.button, 0, point(ev.button.x, ev.button.y));
      // else if (ev.type == SDL_ACTIVEEVENT && ev.active.state & SDL_APPACTIVE) { if ((minimized = ev.active.gain)) return 0; }
    }

#ifndef __APPLE__
    if (mouse_moved && screen->cursor_grabbed) {
      SDL_WarpMouseInWindow((SDL_Window*)screen->id, width/2, height/2);
      while(SDL_PollEvent(&ev)) { /* do nothing */ }
    }
#endif
    return 0;
  }
};

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
void Application::GrabMouseFocus()    { SDL_ShowCursor(0); SDL_SetWindowGrab((SDL_Window*)screen->id, SDL_TRUE);  SDL_SetRelativeMouseMode(SDL_TRUE);  app->grab_mode.On();  screen->cursor_grabbed=true; }
void Application::ReleaseMouseFocus() { SDL_ShowCursor(1); SDL_SetWindowGrab((SDL_Window*)screen->id, SDL_FALSE); SDL_SetRelativeMouseMode(SDL_FALSE); app->grab_mode.Off(); screen->cursor_grabbed=false; }

int Video::Swap() {
  screen->gd->Flush();
  SDL_GL_SwapWindow((SDL_Window*)screen->id);
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

extern "C" void *LFAppCreatePlatformModule() {
  ONCE({ if (FLAGS_lfapp_audio || FLAGS_lfapp_video) {
    int SDL_Init_Flag = 0;
    SDL_Init_Flag |= (FLAGS_lfapp_video ? SDL_INIT_VIDEO : 0);
    SDL_Init_Flag |= (FLAGS_lfapp_audio ? SDL_INIT_AUDIO : 0);
    INFO("LFAppCreatePlatformModule: SDL_init()");
    if (SDL_Init(SDL_Init_Flag) < 0) return ERRORv(nullptr, "SDL_Init: ", SDL_GetError());
  }});
  return new SDLVideoModule();
}

}; // namespace LFL
