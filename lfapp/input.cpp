/*
 * $Id: input.cpp 1328 2014-11-04 09:35:46Z justin $
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
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"

#ifdef LFL_QT
#include <QtOpenGL>
#endif

#ifdef LFL_ANDROID
#include <android/log.h>
#endif

#ifdef LFL_GLFWINPUT
#include "GLFW/glfw3.h"
#endif

#ifdef LFL_SDLINPUT
#include "SDL.h"
extern "C" {
#ifdef LFL_IPHONE
#include "SDL_uikitkeyboard.h"
#endif
};
#endif

namespace LFL {
DEFINE_float(ksens, 4, "Keyboard sensitivity");
DEFINE_float(msens, 1, "Mouse sensitivity");
DEFINE_int(invert, 1, "Invert mouse [1|-1]");
DEFINE_int(keyboard_repeat, 50, "Keyboard repeat in milliseconds");
DEFINE_int(keyboard_delay, 180, "Keyboard delay until repeat in milliseconds");
DEFINE_bool(input_debug, false, "Debug input events");

#if !defined(LFL_ANDROID) && !defined(LFL_IPHONE) && !defined(LFL_GLFWINPUT) && !defined(LFL_SDLINPUT) && !defined(LFL_QT)
int Key::Escape     = 0;
int Key::Return     = 0;
int Key::Up         = 0;
int Key::Down       = 0;
int Key::Left       = 0;
int Key::Right      = 0;
int Key::LeftShift  = 0;
int Key::RightShift = 0;
int Key::LeftCtrl   = 0;
int Key::RightCtrl  = 0;
int Key::LeftCmd    = 0;
int Key::RightCmd   = 0;
int Key::Tab        = 0;
int Key::Space      = 0;
int Key::Backspace  = 0;
int Key::Delete     = 0;
int Key::Quote      = 0;
int Key::Backquote  = 0;
int Key::PageUp     = 0;
int Key::PageDown   = 0;
int Key::F1         = 0;
int Key::F2         = 0;
int Key::F3         = 0;
int Key::F4         = 0;
int Key::F5         = 0;
int Key::F6         = 0;
int Key::F7         = 0;
int Key::F8         = 0;
int Key::F9         = 0;
int Key::F10        = 0;
int Key::F11        = 0;
int Key::F12        = 0;
int Key::Home       = 0;
int Key::End        = 0;

const char *Clipboard::get() { return ""; }
void Clipboard::set(const char *s) {}
void TouchDevice::openKeyboard() {}
void TouchDevice::closeKeyboard() {}
void Mouse::grabFocus() {}
void Mouse::releaseFocus() {}
#endif

#ifdef LFL_ANDROID
struct AndroidInputModule : public Module {
    int Frame(unsigned clicks) { return android_input(clicks); }
};

bool android_keyboard_toggled = false;

int Key::Escape     = -1;
int Key::Return     = 10;
int Key::Up         = -1;
int Key::Down       = -1;
int Key::Left       = -1;
int Key::Right      = -1;
int Key::LeftShift  = -1;
int Key::RightShift = -1;
int Key::LeftCtrl   = -1;
int Key::RightCtrl  = -1;
int Key::LeftCmd    = -1;
int Key::RightCmd   = -1;
int Key::Tab        = -1;
int Key::Space      = -1;
int Key::Backspace  = 0;
int Key::Delete     = -1;
int Key::Quote      = '\'';
int Key::Backquote  = '`';
int Key::PageUp     = -1;
int Key::PageDown   = -1;
int Key::F1         = -1;
int Key::F2         = -1;
int Key::F3         = -1;
int Key::F4         = -1;
int Key::F5         = -1;
int Key::F6         = -1;
int Key::F7         = -1;
int Key::F8         = -1;
int Key::F9         = -1;
int Key::F10        = -1;
int Key::F11        = -1;
int Key::F12        = -1;
int Key::Home       = -1;
int Key::End        = -1;

const char *Clipboard::get() { return ""; }
void Clipboard::set(const char *s) {}
void TouchDevice::openKeyboard()  { if ( android_keyboard_toggled) return; android_toggle_keyboard(); android_keyboard_toggled=1; }
void TouchDevice::closeKeyboard() { if (!android_keyboard_toggled) return; android_toggle_keyboard(); android_keyboard_toggled=0; }
void Mouse::grabFocus() {}
void Mouse::releaseFocus() {}
#endif

#ifdef LFL_IPHONE
struct IPhoneInputModule : public Module {
    int Frame(unsigned clicks) { return iphone_input(clicks); }
};

int Key::Escape     = -1;
int Key::Return     = 10;
int Key::Up         = -1;
int Key::Down       = -1;
int Key::Left       = -1;
int Key::Right      = -1;
int Key::LeftShift  = -1;
int Key::RightShift = -1;
int Key::LeftCtrl   = -1;
int Key::RightCtrl  = -1;
int Key::LeftCmd    = -1;
int Key::RightCmd   = -1;
int Key::Tab        = -1;
int Key::Space      = -1;
int Key::Backspace  = 8;
int Key::Delete     = -1;
int Key::Quote      = -1;
int Key::Backquote  = '~';
int Key::PageUp     = -1;
int Key::PageDown   = -1;
int Key::F1         = -1;
int Key::F2         = -1;
int Key::F3         = -1;
int Key::F4         = -1;
int Key::F5         = -1;
int Key::F6         = -1;
int Key::F7         = -1;
int Key::F8         = -1;
int Key::F9         = -1;
int Key::F10        = -1;
int Key::F11        = -1;
int Key::F12        = -1;
int Key::Home       = -1;
int Key::End        = -1;

int iphone_show_keyboard();
const char *Clipboard::get() { return ""; }
void Clipboard::set(const char *s) {}
void TouchDevice::openKeyboard() { iphone_show_keyboard(); }
void TouchDevice::closeKeyboard() {}
void Mouse::grabFocus() {}
void Mouse::releaseFocus() {}
#endif

#ifdef LFL_QT
struct QTInputModule : public Module {
    bool grabbed = 0;
    int Frame(unsigned clicks) {
        app->input.DispatchQueuedInput();
        return 0;
    }
};

int Key::Escape     = Qt::Key_Escape;
int Key::Return     = Qt::Key_Return;
int Key::Up         = Qt::Key_Up;
int Key::Down       = Qt::Key_Down;
int Key::Left       = Qt::Key_Left;
int Key::Right      = Qt::Key_Right;
int Key::LeftShift  = Qt::Key_Shift;
int Key::RightShift = 0;
int Key::LeftCtrl   = Qt::Key_Meta;
int Key::RightCtrl  = 0;
int Key::LeftCmd    = Qt::Key_Control;
int Key::RightCmd   = 0;
int Key::Tab        = Qt::Key_Tab;
int Key::Space      = Qt::Key_Space;
int Key::Backspace  = Qt::Key_Backspace;
int Key::Delete     = Qt::Key_Delete;
int Key::Quote      = Qt::Key_Apostrophe;
int Key::Backquote  = Qt::Key_QuoteLeft;
int Key::PageUp     = Qt::Key_PageUp;
int Key::PageDown   = Qt::Key_PageDown;
int Key::F1         = Qt::Key_F1;
int Key::F2         = Qt::Key_F2;
int Key::F3         = Qt::Key_F3;
int Key::F4         = Qt::Key_F4;
int Key::F5         = Qt::Key_F5;
int Key::F6         = Qt::Key_F6;
int Key::F7         = Qt::Key_F7;
int Key::F8         = Qt::Key_F8;
int Key::F9         = Qt::Key_F9;
int Key::F10        = Qt::Key_F10;
int Key::F11        = Qt::Key_F11;
int Key::F12        = Qt::Key_F12;
int Key::Home       = Qt::Key_Home;
int Key::End        = Qt::Key_End;

const char *Clipboard::get() { return ""; }
void Clipboard::set(const char *s) {}
void TouchDevice::openKeyboard() {}
void TouchDevice::closeKeyboard() {}
#endif /* LFL_QT */

#ifdef LFL_GLFWINPUT
struct GLFWInputModule : public Module {
    static double mx, my, mw;
    GLFWInputModule(GLFWwindow *W) {
        glfwSetInputMode          (W, GLFW_STICKY_KEYS, GL_TRUE);
        glfwSetWindowCloseCallback(W, WindowClose);
        glfwSetWindowSizeCallback (W, WindowSize);
        glfwSetKeyCallback        (W, Key);
        glfwSetMouseButtonCallback(W, MouseClick);
        glfwSetCursorPosCallback  (W, MousePosition);
        glfwSetScrollCallback     (W, MouseWheel);
    }
    int Frame(unsigned clicks) { glfwPollEvents(); return 0; }
    static bool LoadScreen (GLFWwindow *W) { if (!(screen = Window::Get(W))) return 0; screen->events.input++; return 1; }
    static void WindowSize (GLFWwindow *W, int w, int h) { if (!LoadScreen(W)) return; Window::MakeCurrent(screen); screen->Reshaped(w, h); }
    static void WindowClose(GLFWwindow *W)               { if (!LoadScreen(W)) return; Window::MakeCurrent(screen); Window::Close(screen); }
    static void Key(GLFWwindow *W, int k, int s, int a, int m) {
        if (a == GLFW_REPEAT || !LoadScreen(W)) return;
        app->input.KeyPress((unsigned)k < 256 && isalpha((unsigned)k) ? ::tolower((unsigned)k) : k, a == GLFW_PRESS, 0, 0);
    }
    static void MouseClick(GLFWwindow *W, int b, int a, int m) {
        if (!LoadScreen(W)) return;
        app->input.MouseClick(MouseButton(b), a == GLFW_PRESS, mx, my);
    }
    static void MousePosition(GLFWwindow *W, double x, double y) {
        if (!LoadScreen(W)) return;
        app->input.MouseMove(x, y, x - mx, y - my);
        mx=x; my=y;
    }
    static void MouseWheel(GLFWwindow *W, double x, double y) {
        if (!LoadScreen(W)) return;
        app->input.MouseWheel(y - mw, 0, 0, 0);
        mw=y;
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

double GLFWInputModule::mx;
double GLFWInputModule::my;
double GLFWInputModule::mw;

int Key::Escape     = GLFW_KEY_ESCAPE;
int Key::Return     = GLFW_KEY_ENTER;
int Key::Up         = GLFW_KEY_UP;
int Key::Down       = GLFW_KEY_DOWN;
int Key::Left       = GLFW_KEY_LEFT;
int Key::Right      = GLFW_KEY_RIGHT;
int Key::LeftShift  = GLFW_KEY_LEFT_SHIFT;
int Key::RightShift = GLFW_KEY_RIGHT_SHIFT;
int Key::LeftCtrl   = GLFW_KEY_LEFT_CONTROL;
int Key::RightCtrl  = GLFW_KEY_RIGHT_CONTROL;
int Key::LeftCmd    = GLFW_KEY_LEFT_SUPER;
int Key::RightCmd   = GLFW_KEY_RIGHT_SUPER;
int Key::Tab        = GLFW_KEY_TAB;
int Key::Space      = GLFW_KEY_SPACE;
int Key::Backspace  = GLFW_KEY_BACKSPACE;
int Key::Delete     = GLFW_KEY_DELETE;
int Key::Quote      = '\'';
int Key::Backquote  = '`';
int Key::PageUp     = GLFW_KEY_PAGE_UP;
int Key::PageDown   = GLFW_KEY_PAGE_DOWN;
int Key::F1         = GLFW_KEY_F1;
int Key::F2         = GLFW_KEY_F2;
int Key::F3         = GLFW_KEY_F3;
int Key::F4         = GLFW_KEY_F4;
int Key::F5         = GLFW_KEY_F5;
int Key::F6         = GLFW_KEY_F6;
int Key::F7         = GLFW_KEY_F7;
int Key::F8         = GLFW_KEY_F8;
int Key::F9         = GLFW_KEY_F9;
int Key::F10        = GLFW_KEY_F10;
int Key::F11        = GLFW_KEY_F11;
int Key::F12        = GLFW_KEY_F12;
int Key::Home       = GLFW_KEY_HOME;
int Key::End        = GLFW_KEY_END;

void TouchDevice::openKeyboard() {}
void TouchDevice::closeKeyboard() {}
const char *Clipboard::get()              { return glfwGetClipboardString((GLFWwindow*)screen->id   ); }
void        Clipboard::set(const char *s) {        glfwSetClipboardString((GLFWwindow*)screen->id, s); }
void Mouse::grabFocus()    { glfwSetInputMode((GLFWwindow*)screen->id, GLFW_CURSOR, GLFW_CURSOR_DISABLED); app->grabMode.on();  screen->cursor_grabbed=true;  }
void Mouse::releaseFocus() { glfwSetInputMode((GLFWwindow*)screen->id, GLFW_CURSOR, GLFW_CURSOR_NORMAL);   app->grabMode.off(); screen->cursor_grabbed=false; }
#endif

#ifdef LFL_SDLINPUT
struct SDLInputModule : public Module {
    int Frame(unsigned clicks) {
        SDL_Event ev; int mx, my;
        SDL_GetMouseState(&mx, &my);
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
            else if (ev.type == SDL_KEYDOWN) app->input.KeyPress(ev.key.keysym.sym, 1, 0, 0);
            else if (ev.type == SDL_KEYUP)   app->input.KeyPress(ev.key.keysym.sym, 0, 0, 0);
            else if (ev.type == SDL_MOUSEMOTION) {
                app->input.MouseMove(mx, my, ev.motion.xrel, ev.motion.yrel);
                mouse_moved = true;
            }
            else if (ev.type == SDL_MOUSEBUTTONDOWN) app->input.MouseClick(ev.button.button, 1, ev.button.x, ev.button.y);
            else if (ev.type == SDL_MOUSEBUTTONUP)   app->input.MouseClick(ev.button.button, 0, ev.button.x, ev.button.y);
            // else if (ev.type == SDL_ACTIVEEVENT && ev.active.state & SDL_APPACTIVE) { if ((minimized = ev.active.gain)) return 0; }
            screen->events.input++;
        }

#ifndef __APPLE__
        if (mouse_moved && cursor_grabbed) {
            SDL_WarpMouseInWindow((SDL_Window*)screen->id, width/2, height/2);
            while(SDL_PollEvent(&ev)) { /* do nothing */ }
        }
#endif
        return 0;
    }
};

int Key::Escape     = SDLK_ESCAPE;
int Key::Return     = SDLK_RETURN;
int Key::Up         = SDLK_UP;
int Key::Down       = SDLK_DOWN;
int Key::Left       = SDLK_LEFT;
int Key::Right      = SDLK_RIGHT;
int Key::LeftShift  = SDLK_LSHIFT;
int Key::RightShift = SDLK_RSHIFT;
int Key::LeftCtrl   = SDLK_LCTRL;
int Key::RightCtrl  = SDLK_RCTRL;
int Key::LeftCmd    = SDLK_LGUI;
int Key::RightCmd   = SDLK_RGUI;
int Key::Tab        = SDLK_TAB;
int Key::Space      = SDLK_SPACE;
int Key::Backspace  = SDLK_BACKSPACE;
int Key::Delete     = SDLK_DELETE;
int Key::Quote      = SDLK_QUOTE;
int Key::Backquote  = SDLK_BACKQUOTE;
int Key::PageUp     = SDLK_PAGEUP;
int Key::PageDown   = SDLK_PAGEDOWN;
int Key::F1         = SDLK_F1;
int Key::F2         = SDLK_F2;
int Key::F3         = SDLK_F3;
int Key::F4         = SDLK_F4;
int Key::F5         = SDLK_F5;
int Key::F6         = SDLK_F6;
int Key::F7         = SDLK_F7;
int Key::F8         = SDLK_F8;
int Key::F9         = SDLK_F9;
int Key::F10        = SDLK_F10;
int Key::F11        = SDLK_F11;
int Key::F12        = SDLK_F12;
int Key::Home       = SDLK_HOME;
int Key::End        = SDLK_END;

const char *Clipboard::get() { return SDL_GetClipboardText(); }
void Clipboard::set(const char *s) { SDL_SetClipboardText(s); }
void TouchDevice::closeKeyboard() {
#ifdef LFL_IPHONE 
    SDL_iPhoneKeyboardHide((SDL_Window*)screen->id);
#endif
}
void TouchDevice::openKeyboard() {
#ifdef LFL_IPHONE 
    SDL_iPhoneKeyboardShow((SDL_Window*)screen->id);
#endif
}
void Mouse::grabFocus()    { SDL_ShowCursor(0); SDL_SetWindowGrab((SDL_Window*)screen->id, SDL_TRUE);  SDL_SetRelativeMouseMode(SDL_TRUE);  app->grabMode.on();  screen->cursor_grabbed=true; }
void Mouse::releaseFocus() { SDL_ShowCursor(1); SDL_SetWindowGrab((SDL_Window*)screen->id, SDL_FALSE); SDL_SetRelativeMouseMode(SDL_FALSE); app->grabMode.off(); screen->cursor_grabbed=false; }
#endif /* LFL_SDLINPUT */

int Input::Init() {
#if defined(LFL_QT)
    impl = new QTInputModule();
#elif defined(LFL_GLFWINPUT)
    impl = new GLFWInputModule((GLFWwindow*)screen->id);
#elif defined(LFL_SDLINPUT)
    impl = new SDLInputModule();
#elif defined(LFL_ANDROID)
    impl = new AndroidInputModule();
#elif defined(LFL_IPHONE)
    impl = new IPhoneInputModule();
#endif
    return 0;
}

int Input::Frame(unsigned clicks) {
    if (impl) impl->Frame(clicks);
    if (screen) {
        KeyPressRepeat(clicks);
        if (screen->binds) screen->binds->Repeat(clicks);
    }
    return 0;
}

int Input::DispatchQueuedInput() {
    vector<Callback> icb;
    {
        ScopedMutex sm(queued_input_mutex);
        icb = queued_input;
        queued_input.clear();
    }
    int ret = icb.size();
    for (vector<Callback>::iterator i = icb.begin(); i != icb.end(); ++i) {
        (*i)();
        if (screen) screen->events.input++;
    }
    return ret;
}

void Input::KeyPress(int key, int down, int, int) {
    if (FLAGS_keyboard_repeat && key >= 0 && key < repeat_keys) {
        if      ( down && !key_down[key]) { keys_down.insert(key); key_down[key]=1; key_delay[key]=0; key_down_repeat[key]=Now(); }
        else if (!down &&  key_down[key]) { keys_down.erase (key); key_down[key]=0;                                               }
    }

    if      (key == Key::LeftShift)   left_shift_down = down;
    else if (key == Key::RightShift) right_shift_down = down;
    else if (key == Key::LeftCtrl)     left_ctrl_down = down;
    else if (key == Key::RightCtrl)   right_ctrl_down = down;
    else if (key == Key::LeftCmd)       left_cmd_down = down;
    else if (key == Key::RightCmd)     right_cmd_down = down;

    key_mod = (CtrlKeyDown() ? Key::Modifier::Ctrl : 0) | (CmdKeyDown() ? Key::Modifier::Cmd : 0);

    int fired = KeyEventDispatch(key, key_mod, down);
    screen->events.key++;
    screen->events.gui += fired;
    if (fired) return;

    fired = screen->binds ? screen->binds->Run(key, key_mod, down) : 0;
    screen->events.bind += fired;
}

void Input::KeyPressRepeat(unsigned clicks) {
    Time now = Now();
    for (unordered_set<int>::const_iterator i = keys_down.begin(); i != keys_down.end(); ++i) {
        int elapsed = now - key_down_repeat[*i], delay = key_delay[*i];
        if ((!delay && elapsed < FLAGS_keyboard_delay) ||
            ( delay && elapsed < FLAGS_keyboard_repeat)) continue;

        for (int j=0, max_repeat=10; elapsed >= FLAGS_keyboard_repeat; ++j) {
            if (!delay) { delay=1; key_delay[*i]=true; elapsed -= FLAGS_keyboard_delay; }
            else        {                              elapsed -= FLAGS_keyboard_repeat; }

            if (j >= max_repeat) continue;
            KeyEventDispatch(*i, 0, false);
            KeyEventDispatch(*i, 0, true);
        }
        key_down_repeat[*i] = now - elapsed;
    }
}

int Input::KeyEventDispatch(int key, int keymod, bool down) {
    if (screen->browser_window) screen->browser_window->KeyEvent(key, down);
    if (!down) return 0;

    if (ShiftKeyDown() && key < 256) {
        if (isalpha(key)) key = ::toupper(key);
        else switch(key) {
            case '\'': key='"'; break;
            case '\\': key='|'; break;
            case  '-': key='_'; break;
            case  ';': key=':'; break;
            case  ',': key='<'; break;
            case  '.': key='>'; break;
            case  '/': key='?'; break;
            case  '=': key='+'; break;
            case  '1': key='!'; break;
            case  '2': key='@'; break;
            case  '3': key='#'; break;
            case  '4': key='$'; break;
            case  '5': key='%'; break;
            case  '6': key='^'; break;
            case  '7': key='&'; break;
            case  '8': key='*'; break;
            case  '9': key='('; break;
            case  '0': key=')'; break;
            case  '[': key='{'; break;
            case  ']': key='}'; break;
            case  '`': key='~'; break;
        }
    }

    if (CtrlKeyDown() && key < 256) {
        if (isalpha(key)) key = ::toupper(key);
        if (key >= 'A' && key <= '_') key -= 0x40;
    }

    for (set<KeyboardGUI*>::iterator it = screen->keyboard_gui.begin(); it != screen->keyboard_gui.end(); ++it) {
        KeyboardGUI *g = *it;
        if (!g->active) continue;

        if (g->toggle_bind.Match(key, keymod) && g->toggle_active.mode != ToggleBool::OneShot) return 0;
        g->events.total++;

        if      (key == Key::Backspace || key == Key::Delete) { g->Erase();                  return 1; }
        else if (key == Key::Return)                          { g->Enter();                  return 1; }
        else if (key == Key::Left)                            { g->CursorLeft();             return 1; }
        else if (key == Key::Right)                           { g->CursorRight();            return 1; }
        else if (key == Key::Up)                              { g->HistUp();                 return 1; }
#ifdef LFL_IPHONE                                                                            
        else if (key == '6' && shift_key_down())              { g->HistUp();                 return 1; }
#endif                                                                                       
        else if (key == Key::Down)                            { g->HistDown();               return 1; }
        else if (key == Key::PageUp)                          { g->PageUp();                 return 1; }
        else if (key == Key::PageDown)                        { g->PageDown();               return 1; }
        else if (key == Key::Home)                            { g->Home();                   return 1; }
        else if (key == Key::End)                             { g->End();                    return 1; }
        else if (key == Key::Tab)                             { g->Tab();                    return 1; }
        else if (key == Key::Escape)                          { g->Escape();                 return 1; }
#ifdef __APPLE__
        else if (key == 'v' && CmdKeyDown())                  { g->Input(Clipboard::get());  return 1; }
#else                                                     
        else if (key == 'v' && CtrlKeyDown())                 { g->Input(Clipboard::get());  return 1; }
#endif
        else if (CmdKeyDown()) /* skip command keys */        { g->events.total--;           return 0; }
        else if (key >= 0 && key<128)                         { g->Input(key);               return 1; }
        else {
            g->events.total--;
            // ERROR("unhandled key ", key);
        }
    }
    return 0;
}

void Input::MouseMove(int x, int y, int dx, int dy) {
    screen->events.mouse_move++;
    screen->events.gui += MouseEventDispatch(Bind::MOUSEMOTION, x, y, 0);
    if (!app->grabMode.enabled()) return;
    if (dx<0) screen->camMain->YawLeft  (-dx); else if (dx>0) screen->camMain->YawRight(dx);
    if (dy<0) screen->camMain->PitchDown(-dy); else if (dy>0) screen->camMain->PitchUp (dy);
}

void Input::MouseWheel(int dw, int, int, int) {
    screen->events.mouse_wheel++;
    if (screen->browser_window) screen->browser_window->MouseWheel(0, dw*32);
}

void Input::MouseClick(int button, int down, int x, int y) {
    int key = MouseButtonID(button);
    if      (key == Bind::MOUSE1) mouse_but1_down = down;
    else if (key == Bind::MOUSE2) mouse_but2_down = down;

    int fired = MouseEventDispatch(key, x, y, down);
    screen->events.mouse_click++;
    screen->events.gui += fired;
    if (fired) return;

    fired = screen->binds ? screen->binds->Run(key, 0, down) : 0;
    screen->events.bind += fired;
}

int Input::MouseEventDispatch(int button, int X, int Y, int down) {
    screen->mouse = TransformMouseCoordinate(point(X, Y));
    if (FLAGS_input_debug && down) INFO("MouseEvent ", screen->mouse.DebugString());

    if (screen->browser_window) {
        if (button == Bind::MOUSEMOTION) screen->browser_window->MouseMoved(screen->mouse.x, screen->mouse.y);
        else                             screen->browser_window->MouseButton(button, down);
    }

    int fired = 0;
    for (set<GUI*>::iterator g = screen->mouse_gui.begin(); g != screen->mouse_gui.end(); ++g)
        if ((*g)->mouse.active) fired += (*g)->mouse.Input(button, (*g)->MousePosition(), down, 0);

    vector<Dialog*> removed;
    Dialog *bring_to_front = 0;
    for (vector<Dialog*>::iterator i = screen->dialogs.begin(); i != screen->dialogs.end(); /**/) {
        Dialog *gui = (*i);
        if (!gui->mouse.active) { i++; continue; }
        fired += gui->mouse.Input(button, screen->mouse, down, 0);
        if (gui->deleted) { delete gui; i = screen->dialogs.erase(i); continue; }
        if (button == Bind::MOUSE1 && down && gui->BoxAndTitle().within(screen->mouse)) { bring_to_front = *i; break; }
        i++;
    }
    if (bring_to_front) bring_to_front->BringToFront();

    if (FLAGS_input_debug && down) INFO("MouseEvent ", screen->mouse.DebugString(), " fired=", fired, ", guis=", screen->mouse_gui.size());
    return fired;
}

int MouseController::Input(int button, const point &p, int down, int flag) {
    int fired = 0;
    for (vector<HitBox>::iterator e = hit.begin(); e != hit.end(); ++e) {
        if (e->deleted || !e->active ||
            (!down && e->evtype == Event::Click && e->CB.type != Callback::CB_COORD)) continue;

        bool thunk = 0;
        if (e->box.within(p)) {
            if (e->run_only_if_first && fired) continue;
            if      (e->evtype == Event::Click && button == Bind::MOUSE1) thunk=1;
            else if (e->evtype == Event::Hover && !e->val) { e->val=1; thunk=1; }
        }
        else {
            if (e->evtype == Event::Hover && e->val) { e->val=0; thunk=1; }
        }

        if (thunk) {
            if (FLAGS_input_debug && down) INFO("MouseController::Input ", p.DebugString(), " ", e->box.DebugString());
            e->CB.Run(p, button, down);

            if (1)                         events.total++;
            if (e->evtype == Event::Hover) events.hover++;
            else                           events.click++;
            fired++;

            if (flag) break;
        }
    }
    if (FLAGS_input_debug && down) INFO("MouseController::Input ", screen->mouse.DebugString(), " fired=", fired, ", hitboxes=", hit.size());
    return fired;
}

}; // namespace LFL
