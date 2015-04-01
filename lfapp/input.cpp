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
#include "lfapp/flow.h"
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

const InputEvent::Id Key::Modifier::Ctrl   = 1LL<<32;
const InputEvent::Id Key::Modifier::Cmd    = 1LL<<33;
const InputEvent::Id Mouse::Button::_1     = 1LL<<34;
const InputEvent::Id Mouse::Button::_2     = 1LL<<35;
const InputEvent::Id MouseEvent            = 1LL<<36;
const InputEvent::Id Mouse::Event::Motion  = MouseEvent+0;
const InputEvent::Id Mouse::Event::Wheel   = MouseEvent+1;
const InputEvent::Id Mouse::Event::Button1 = Mouse::Button::_1;
const InputEvent::Id Mouse::Event::Button2 = Mouse::Button::_2;

const char *InputEvent::Name(InputEvent::Id event) {
    switch (event) {
        case Mouse::Event::Motion:  return "MouseMotion";
        case Mouse::Event::Wheel:   return "MouseWheel";
        case Mouse::Event::Button1: return "MouseButton1";
        case Mouse::Event::Button2: return "MouseButton2";
        default:                    return "Unknown";
    }
}

#if 0
struct KeyRepeater {
    static const int repeat_keys=512;
    unordered_set<int> keys_down;
    bool key_down[repeat_keys], key_delay[repeat_keys];
    Time key_down_repeat[repeat_keys];
    KeyRepeater() { memzero(key_down); memzero(key_delay); memzero(key_down_repeat); }

    void KeyChange(int key, int down) {
        if (key < 0 || key >= repeat_keys) return;
        if      ( down && !key_down[key]) { keys_down.insert(key); key_down[key]=1; key_delay[key]=0; key_down_repeat[key]=Now(); }
        else if (!down &&  key_down[key]) { keys_down.erase (key); key_down[key]=0;                                               }
    }
    void Repeat(unsigned clicks) {
        Time now = Now();
        for (auto i = keys_down.begin(); i != keys_down.end(); ++i) {
            int elapsed = now - key_down_repeat[*i], delay = key_delay[*i];
            if ((!delay && elapsed < FLAGS_keyboard_delay) ||
                ( delay && elapsed < FLAGS_keyboard_repeat)) continue;
            for (int j=0, max_repeat=10; elapsed >= FLAGS_keyboard_repeat; ++j) {
                if (!delay) { delay=1; key_delay[*i]=true; elapsed -= FLAGS_keyboard_delay; }
                else        {                              elapsed -= FLAGS_keyboard_repeat; }
                if (j < max_repeat) app->input.KeyEventDispatch(*i, true);
            }
            key_down_repeat[*i] = now - elapsed;
        }
    }
};
#endif

#if !defined(LFL_ANDROIDINPUT) && !defined(LFL_IPHONEINPUT) && !defined(LFL_GLFWINPUT) && !defined(LFL_SDLINPUT) && !defined(LFL_QT) && !defined(LFL_OSXINPUT)
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

string Clipboard::Get() { return ""; }
void Clipboard::Set(const string &s) {}
void TouchDevice::OpenKeyboard() {}
void TouchDevice::CloseKeyboard() {}
void Mouse::GrabFocus() {}
void Mouse::ReleaseFocus() {}
#endif

#ifdef LFL_ANDROIDINPUT
struct AndroidInputModule : public Module {
    int Frame(unsigned clicks) { return android_input(clicks); }
};

static bool android_keyboard_toggled = false;

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
const int Key::Backspace  = 0;
const int Key::Delete     = -16;
const int Key::Quote      = '\'';
const int Key::Backquote  = '`';
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

string Clipboard::Get() { return ""; }
void Clipboard::Set(const string &s) {}
void TouchDevice::OpenKeyboard()  { if ( android_keyboard_toggled) return; android_toggle_keyboard(); android_keyboard_toggled=1; }
void TouchDevice::CloseKeyboard() { if (!android_keyboard_toggled) return; android_toggle_keyboard(); android_keyboard_toggled=0; }
void Mouse::GrabFocus() {}
void Mouse::ReleaseFocus() {}
#endif

#ifdef LFL_IPHONEINPUT
extern "C" int iPhoneInput(unsigned clicks, unsigned *events);

struct IPhoneInputModule : public Module {
    int Frame(unsigned clicks) { return iPhoneInput(clicks); }
};

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

int iphone_show_keyboard();
string Clipboard::Get() { return ""; }
void Clipboard::Set(const string &s) {}
void TouchDevice::OpenKeyboard() { iphone_show_keyboard(); }
void TouchDevice::CloseKeyboard() {}
void Mouse::GrabFocus() {}
void Mouse::ReleaseFocus() {}
#endif

#ifdef LFL_QT
struct QTInputModule : public Module {
    bool grabbed = 0;
    int Frame(unsigned clicks) {
        app->input.DispatchQueuedInput();
        return 0;
    }
};

const int Key::Escape     = Qt::Key_Escape;
const int Key::Return     = Qt::Key_Return;
const int Key::Up         = Qt::Key_Up;
const int Key::Down       = Qt::Key_Down;
const int Key::Left       = Qt::Key_Left;
const int Key::Right      = Qt::Key_Right;
const int Key::LeftShift  = Qt::Key_Shift;
const int Key::RightShift = -8;
const int Key::LeftCtrl   = Qt::Key_Meta;
const int Key::RightCtrl  = -10;
const int Key::LeftCmd    = Qt::Key_Control;
const int Key::RightCmd   = -12;
const int Key::Tab        = Qt::Key_Tab;
const int Key::Space      = Qt::Key_Space;
const int Key::Backspace  = Qt::Key_Backspace;
const int Key::Delete     = Qt::Key_Delete;
const int Key::Quote      = Qt::Key_Apostrophe;
const int Key::Backquote  = Qt::Key_QuoteLeft;
const int Key::PageUp     = Qt::Key_PageUp;
const int Key::PageDown   = Qt::Key_PageDown;
const int Key::F1         = Qt::Key_F1;
const int Key::F2         = Qt::Key_F2;
const int Key::F3         = Qt::Key_F3;
const int Key::F4         = Qt::Key_F4;
const int Key::F5         = Qt::Key_F5;
const int Key::F6         = Qt::Key_F6;
const int Key::F7         = Qt::Key_F7;
const int Key::F8         = Qt::Key_F8;
const int Key::F9         = Qt::Key_F9;
const int Key::F10        = Qt::Key_F10;
const int Key::F11        = Qt::Key_F11;
const int Key::F12        = Qt::Key_F12;
const int Key::Home       = Qt::Key_Home;
const int Key::End        = Qt::Key_End;

string Clipboard::Get() { return ""; }
void Clipboard::Set(const string &s) {}
void TouchDevice::OpenKeyboard() {}
void TouchDevice::CloseKeyboard() {}
#endif /* LFL_QT */

#ifdef LFL_GLFWINPUT
struct GLFWInputModule : public Module {
    static point mp;
    static double mw;
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
        if (!LoadScreen(W)) return;
        app->input.KeyPress((unsigned)k < 256 && isalpha((unsigned)k) ? ::tolower((unsigned)k) : k,
                            (a == GLFW_PRESS || a == GLFW_REPEAT));
    }
    static void MouseClick(GLFWwindow *W, int b, int a, int m) {
        if (!LoadScreen(W)) return;
        app->input.MouseClick(MouseButton(b), a == GLFW_PRESS, mp);
    }
    static void MousePosition(GLFWwindow *W, double x, double y) {
        if (!LoadScreen(W)) return;
        point p = Input::TransformMouseCoordinate(point(x, y));
        app->input.MouseMove(p, p - mp);
        mp = p;
    }
    static void MouseWheel(GLFWwindow *W, double x, double y) {
        if (!LoadScreen(W)) return;
        app->input.MouseWheel(y - mw);
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

point  GLFWInputModule::mp;
double GLFWInputModule::mw;

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
#ifdef __APPLE__
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

void TouchDevice::OpenKeyboard() {}
void TouchDevice::CloseKeyboard() {}
string Clipboard::Get()                { return glfwGetClipboardString((GLFWwindow*)screen->id); }
void   Clipboard::Set(const string &s) {        glfwSetClipboardString((GLFWwindow*)screen->id, s.c_str()); }
void Mouse::GrabFocus()    { glfwSetInputMode((GLFWwindow*)screen->id, GLFW_CURSOR, GLFW_CURSOR_DISABLED); app->grab_mode.On();  screen->cursor_grabbed=true;  }
void Mouse::ReleaseFocus() { glfwSetInputMode((GLFWwindow*)screen->id, GLFW_CURSOR, GLFW_CURSOR_NORMAL);   app->grab_mode.Off(); screen->cursor_grabbed=false; }
#endif

#ifdef LFL_SDLINPUT
struct SDLInputModule : public Module {
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
            else if (ev.type == SDL_KEYDOWN) app->input.KeyPress(ev.key.keysym.sym, 1);
            else if (ev.type == SDL_KEYUP)   app->input.KeyPress(ev.key.keysym.sym, 0);
            else if (ev.type == SDL_MOUSEMOTION) {
                app->input.MouseMove(mp, point(ev.motion.xrel, ev.motion.yrel));
                mouse_moved = true;
            }
            else if (ev.type == SDL_MOUSEBUTTONDOWN) app->input.MouseClick(ev.button.button, 1, point(ev.button.x, ev.button.y));
            else if (ev.type == SDL_MOUSEBUTTONUP)   app->input.MouseClick(ev.button.button, 0, point(ev.button.x, ev.button.y));
            // else if (ev.type == SDL_ACTIVEEVENT && ev.active.state & SDL_APPACTIVE) { if ((minimized = ev.active.gain)) return 0; }
            if (screen) screen->events.input++;
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

string Clipboard::Get() { return SDL_GetClipboardText(); }
void Clipboard::Set(const string &s) { SDL_SetClipboardText(s.c_str()); }
void TouchDevice::CloseKeyboard() {
#ifdef LFL_IPHONE 
    SDL_iPhoneKeyboardHide((SDL_Window*)screen->id);
#endif
}
void TouchDevice::OpenKeyboard() {
#ifdef LFL_IPHONE 
    SDL_iPhoneKeyboardShow((SDL_Window*)screen->id);
#endif
}
void Mouse::GrabFocus()    { SDL_ShowCursor(0); SDL_SetWindowGrab((SDL_Window*)screen->id, SDL_TRUE);  SDL_SetRelativeMouseMode(SDL_TRUE);  app->grab_mode.On();  screen->cursor_grabbed=true; }
void Mouse::ReleaseFocus() { SDL_ShowCursor(1); SDL_SetWindowGrab((SDL_Window*)screen->id, SDL_FALSE); SDL_SetRelativeMouseMode(SDL_FALSE); app->grab_mode.Off(); screen->cursor_grabbed=false; }
#endif /* LFL_SDLINPUT */

#ifdef LFL_OSXINPUT
extern "C" void OSXGrabMouseFocus();
extern "C" void OSXReleaseMouseFocus();
extern "C" void OSXSetMousePosition(void*, int x, int y);
extern "C" void OSXClipboardSet(const char *v);
extern "C" const char *OSXClipboardGet();

const int Key::Escape     = 0x81;
const int Key::Return     = '\r';
const int Key::Up         = 0xBE;
const int Key::Down       = 0xBD;
const int Key::Left       = 0xBB;
const int Key::Right      = 0xBC;
const int Key::LeftShift  = 0x83;
const int Key::RightShift = 0x87;
const int Key::LeftCtrl   = 0x86;
const int Key::RightCtrl  = 0x89;
const int Key::LeftCmd    = 0x82;
const int Key::RightCmd   = -12;
const int Key::Tab        = '\t';
const int Key::Space      = ' ';
const int Key::Backspace  = 0x80;
const int Key::Delete     = -16;
const int Key::Quote      = '\'';
const int Key::Backquote  = '`';
const int Key::PageUp     = 0xB4;
const int Key::PageDown   = 0xB9;
const int Key::F1         = 0xBA;
const int Key::F2         = 0xB8;
const int Key::F3         = 0xA8;
const int Key::F4         = 0xB6;
const int Key::F5         = 0xA5;
const int Key::F6         = 0xA6;
const int Key::F7         = 0xA7;
const int Key::F8         = 0xA9;
const int Key::F9         = 0xAA;
const int Key::F10        = 0xAF;
const int Key::F11        = 0xAB;
const int Key::F12        = 0xB0;
const int Key::Home       = 0xB3;
const int Key::End        = 0xB7;

void TouchDevice::OpenKeyboard() {}
void TouchDevice::CloseKeyboard() {}
void Clipboard::Set(const string &s) { OSXClipboardSet(s.c_str()); }
string Clipboard::Get() { const char *v=OSXClipboardGet(); string ret=v; free((void*)v); return ret; }
void Mouse::ReleaseFocus() { OSXReleaseMouseFocus(); app->grab_mode.Off(); screen->cursor_grabbed=0; }
void Mouse::GrabFocus   () { OSXGrabMouseFocus();    app->grab_mode.On();  screen->cursor_grabbed=1;
    OSXSetMousePosition(screen->id, screen->width/2, screen->height/2);
}
#endif // LFL_OSXINPUT

int Input::Init() {
#if defined(LFL_QT)
    impl = new QTInputModule();
#elif defined(LFL_GLFWINPUT)
    impl = new GLFWInputModule((GLFWwindow*)screen->id);
#elif defined(LFL_SDLINPUT)
    impl = new SDLInputModule();
#elif defined(LFL_ANDROIDINPUT)
    impl = new AndroidInputModule();
#elif defined(LFL_IPHONEINPUT)
    impl = new IPhoneInputModule();
#endif
    return 0;
}

int Input::Frame(unsigned clicks) {
    if (impl) impl->Frame(clicks);
    if (screen) { // XXX support multiple windows
        if (screen->binds) screen->binds->Repeat(clicks);
        if (screen->gesture_swipe_up)   { if (screen->console && screen->console->active) screen->console->PageUp();   }
        if (screen->gesture_swipe_down) { if (screen->console && screen->console->active) screen->console->PageDown(); }
        screen->ClearGesture();
    }
    return 0;
}

int Input::DispatchQueuedInput() {
    vector<Callback> icb;
    {
        ScopedMutex sm(queued_input_mutex);
        swap(icb, queued_input);
    }
    int ret = icb.size();
    for (auto i = icb.begin(); i != icb.end(); ++i) {
        (*i)();
        if (screen) screen->events.input++;
    }
    return ret;
}

int Input::KeyPress(int key, bool down) {
    // if (!MainThread()) ERROR("KeyPress() called from thread ", Thread::GetId());

    switch (key) {
        case Key::LeftShift:   left_shift_down = down; break;
        case Key::RightShift: right_shift_down = down; break;
        case Key::LeftCtrl:     left_ctrl_down = down; break;
        case Key::RightCtrl:   right_ctrl_down = down; break;
        case Key::LeftCmd:       left_cmd_down = down; break;
        case Key::RightCmd:     right_cmd_down = down; break;
    }

    InputEvent::Id event = key | (CtrlKeyDown() ? Key::Modifier::Ctrl : 0) | (CmdKeyDown() ? Key::Modifier::Cmd : 0);
    int fired = KeyEventDispatch(event, down);
    screen->events.key++;
    screen->events.gui += fired;
    if (fired) return fired;

    for (auto g = screen->input_bind.begin(); g != screen->input_bind.end(); ++g)
        if ((*g)->active) (*g)->Input(event, down); 

    return fired;
}

int Input::KeyEventDispatch(InputEvent::Id event, bool down) {
    if (!down) return 0;
    int key = InputEvent::GetKey(event);
    bool shift_down = ShiftKeyDown(), ctrl_down = CtrlKeyDown(), cmd_down = CmdKeyDown();
    for (auto it = screen->keyboard_gui.begin(); it != screen->keyboard_gui.end(); ++it) {
        KeyboardGUI *g = *it;
        if (!g->active) continue;
        if (g->toggle_bind.key == event && g->toggle_active.mode != ToggleBool::OneShot) return 0;

        g->events.total++;
#ifdef __APPLE__
        if      (event == Bind('v', Key::Modifier::Cmd).key)  { g->Input(Clipboard::Get()); return 1; }
#else
        if      (event == Bind('v', Key::Modifier::Ctrl).key) { g->Input(Clipboard::Get()); return 1; }
#endif
#ifdef LFL_IPHONE
        else if (event == '6' && shift_down) { g->HistUp(); return 1; }
#endif
        switch (event) {
            case Key::Backspace: g->Erase();       return 1;
            case Key::Delete:    g->Erase();       return 1;
            case Key::Return:    g->Enter();       return 1;
            case Key::Left:      g->CursorLeft();  return 1;
            case Key::Right:     g->CursorRight(); return 1;
            case Key::Up:        g->HistUp();      return 1;
            case Key::Down:      g->HistDown();    return 1;
            case Key::PageUp:    g->PageUp();      return 1;
            case Key::PageDown:  g->PageDown();    return 1;
            case Key::Home:      g->Home();        return 1;
            case Key::End:       g->End();         return 1;
            case Key::Tab:       g->Tab();         return 1;
            case Key::Escape:    g->Escape();      return 1;
        }

        if (cmd_down) { g->events.total--; return 0; }
        if (key >= 128) { g->events.total--; /* ERROR("unhandled key ", event); */ continue; }

        if (shift_down) {
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

        if (ctrl_down) {
            if (isalpha(key)) key = ::toupper(key);
            if (key >= 'A' && key <= '_') key -= 0x40;
        }

        g->Input(key);
        return 1;
    }
    return 0;
}

int Input::MouseMove(const point &p, const point &d) {
    int fired = MouseEventDispatch(Mouse::Event::Motion, p, MouseButton1Down());
    screen->events.mouse_move++;
    screen->events.gui += fired;
    if (!app->grab_mode.Enabled()) return fired;
    if (d.x<0) screen->cam->YawLeft  (-d.x); else if (d.x>0) screen->cam->YawRight(d.x);
    if (d.y<0) screen->cam->PitchDown(-d.y); else if (d.y>0) screen->cam->PitchUp (d.y);
    return fired;
}

int Input::MouseWheel(int dw) {
    int fired = MouseEventDispatch(Mouse::Event::Wheel, screen->mouse, dw);
    screen->events.mouse_wheel++;
    screen->events.gui += fired;
    return fired;
}

int Input::MouseClick(int button, bool down, const point &p) {
    InputEvent::Id event = Mouse::ButtonID(button);
    if      (event == Mouse::Button::_1) mouse_but1_down = down;
    else if (event == Mouse::Button::_2) mouse_but2_down = down;

    int fired = MouseEventDispatch(event, p, down);
    screen->events.mouse_click++;
    screen->events.gui += fired;
    if (fired) return fired;

    for (auto g = screen->input_bind.begin(); g != screen->input_bind.end(); ++g)
        if ((*g)->active) (*g)->Input(event, down);

    return fired;
}

int Input::MouseEventDispatch(InputEvent::Id event, const point &p, int down) {
    screen->mouse = p;
    if (FLAGS_input_debug && down)
        INFO("MouseEvent ", InputEvent::Name(event), " ", screen->mouse.DebugString());

    int fired = 0;
    for (auto g = screen->mouse_gui.begin(); g != screen->mouse_gui.end(); ++g) {
        if ((*g)->NotActive()) continue;
        fired += (*g)->Input(event, (*g)->MousePosition(), down, 0);
    }

    vector<Dialog*> removed;
    Dialog *bring_to_front = 0;
    for (auto i = screen->dialogs.begin(); i != screen->dialogs.end(); /**/) {
        Dialog *gui = (*i);
        if (gui->NotActive()) { i++; continue; }
        fired += gui->Input(event, screen->mouse, down, 0);
        if (gui->deleted) { delete gui; i = screen->dialogs.erase(i); continue; }
        if (event == Mouse::Event::Button1 && down && gui->BoxAndTitle().within(screen->mouse)) { bring_to_front = *i; break; }
        i++;
    }
    if (bring_to_front) bring_to_front->BringToFront();

    if (FLAGS_input_debug && down) INFO("MouseEvent ", screen->mouse.DebugString(), " fired=", fired, ", guis=", screen->mouse_gui.size());
    return fired;
}

int MouseController::Input(InputEvent::Id event, const point &p, int down, int flag) {
    int fired = 0;
    bool but1 = event == Mouse::Event::Button1;

    for (auto h = hover.begin(); h != hover.end(); /**/) {
        auto e = &hit.data[*h];
        if (e->box.within(p)) { ++h; continue; }
        h = VectorEraseIterSwapBack(&hover, h);
        e->val = 0;
        e->CB.Run(p, event, 0);
        events.hover++;
        fired++;
    }

    for (auto e = hit.data.rbegin(), ee = hit.data.rend(); e != ee; ++e) {
        bool thunk = 0, e_hover = e->evtype == Event::Hover;

        if (e->deleted || !e->active || (e_hover && e->val) || 
            (!down && e->evtype == Event::Click && e->CB.type != Callback::CB_COORD)) continue;

        if (e->box.within(p)) {
            if (e->run_only_if_first && fired) continue;
            switch (e->evtype) { 
                case Event::Click: if (but1)         { thunk=1; } break;
                case Event::Drag:  if (but1 && down) { thunk=1; } break;
                case Event::Hover: if ((e->val = 1)) { thunk=1; } break;
            }
        }

        if (thunk) {
            if (FLAGS_input_debug && down) INFO("MouseController::Input ", p.DebugString(), " ", e->box.DebugString());
            if (!e->CB.Run(p, event, e_hover ? 1 : down)) continue;

            if (1)       { events.total++; }
            if (e_hover) { events.hover++; hover.push_back(ForwardIteratorFromReverse(e) - hit.data.begin()); }
            else         { events.click++; }
            fired++;

            if (e->evtype == Event::Drag && down) drag.insert(ForwardIteratorFromReverse(e) - hit.data.begin());
            if (flag) break;
        }
    }

    if (event == Mouse::Event::Motion) { for (auto d : drag) if (hit.data[d].CB.Run(p, event, down)) fired++; }
    else if (!down && but1)            { for (auto d : drag) if (hit.data[d].CB.Run(p, event, down)) fired++; drag.clear(); }

    if (FLAGS_input_debug && down) INFO("MouseController::Input ", screen->mouse.DebugString(), " fired=", fired, ", hitboxes=", hit.data.size());
    return fired;
}

Shell::Shell(AssetMap *AM, SoundAssetMap *SAM, MovieAssetMap *MAM) : assets(AM), soundassets(SAM), movieassets(MAM) {
    command.push_back(Command("quit",       bind(&Shell::quit,         this, _1)));
    command.push_back(Command("cmds",       bind(&Shell::cmds,         this, _1)));
    command.push_back(Command("binds",      bind(&Shell::binds,        this, _1)));
    command.push_back(Command("flags",      bind(&Shell::flags,        this, _1)));
    command.push_back(Command("conscolor",  bind(&Shell::consolecolor, this, _1)));
    command.push_back(Command("clipboard",  bind(&Shell::clipboard,    this, _1)));
    command.push_back(Command("startcmd",   bind(&Shell::startcmd,     this, _1)));
    command.push_back(Command("dldir",      bind(&Shell::dldir,        this, _1)));
    command.push_back(Command("screenshot", bind(&Shell::screenshot,   this, _1)));
    command.push_back(Command("fillmode",   bind(&Shell::fillmode,     this, _1)));
    command.push_back(Command("texmode",    bind(&Shell::texmode,      this, _1)));
    command.push_back(Command("swapaxis",   bind(&Shell::swapaxis,     this, _1)));
    command.push_back(Command("campos",     bind(&Shell::campos,       this, _1)));
    command.push_back(Command("filter",     bind(&Shell::filter,       this, _1)));
    command.push_back(Command("fftfilter",  bind(&Shell::filter,       this, _1)));
    command.push_back(Command("f0",         bind(&Shell::f0,           this, _1)));
    command.push_back(Command("sinth",      bind(&Shell::sinth,        this, _1)));
    command.push_back(Command("play",       bind(&Shell::play,         this, _1)));
    command.push_back(Command("playmovie",  bind(&Shell::playmovie,    this, _1)));
    command.push_back(Command("loadsound",  bind(&Shell::loadsound,    this, _1)));
    command.push_back(Command("loadmovie",  bind(&Shell::loadmovie,    this, _1)));
    command.push_back(Command("copy",       bind(&Shell::copy,         this, _1)));
    command.push_back(Command("snap",       bind(&Shell::snap,         this, _1)));
    command.push_back(Command("writesnap",  bind(&Shell::writesnap,    this, _1)));
    command.push_back(Command("fps",        bind(&Shell::fps,          this, _1)));
    command.push_back(Command("wget",       bind(&Shell::wget,         this, _1)));
    command.push_back(Command("messagebox", bind(&Shell::MessageBox,   this, _1)));
    command.push_back(Command("texturebox", bind(&Shell::TextureBox,   this, _1)));
    command.push_back(Command("edit",       bind(&Shell::Edit,         this, _1)));
    command.push_back(Command("slider",     bind(&Shell::Slider,       this, _1)));
}

Asset      *Shell::asset     (const string &n) { return assets      ? (*     assets)(n) : 0; }
SoundAsset *Shell::soundasset(const string &n) { return soundassets ? (*soundassets)(n) : 0; }
MovieAsset *Shell::movieasset(const string &n) { return movieassets ? (*movieassets)(n) : 0; }

bool Shell::FGets() {
    char buf[1024];
    if (!LFL::FGets(buf, sizeof(buf))) return false;
    chompnl(buf, strlen(buf));
    Shell::Run(buf);
    return true;
}

void Shell::Run(const string &text) {
    if (!MainThread()) return RunInMainThread(new Callback(bind(&Shell::Run, this, text)));

    string cmd;
    vector<string> arg;
    Split(text, isspace, isquote, &arg);
    if (arg.size()) { cmd = arg[0]; arg.erase(arg.begin()); }
    if (cmd.empty()) return;

    for (auto i = command.begin(); i != command.end(); ++i) {
        if (StringEquals(i->name, cmd)) {
            i->cb(arg);
            return;
        }
    }

    FlagMap *flags = Singleton<FlagMap>::Get();
    for (auto i = flags->flagmap.begin(); i != flags->flagmap.end(); ++i) {
        Flag *flag = (*i).second;
        if (StringEquals(flag->name, cmd)) {
            flag->Update(arg.size() ? arg[0].c_str() : "");
            INFO(flag->name, " = ", flag->Get());
            return;
        }
    }
    INFO("unkown cmd '", cmd, "'");
}

void Shell::mousein (const vector<string>&) { Mouse::GrabFocus(); }
void Shell::mouseout(const vector<string>&) { Mouse::ReleaseFocus(); }

void Shell::quit(const vector<string>&) { app->run = false; }
void Shell::console(const vector<string>&) { screen->console->Toggle(); }
void Shell::showkeyboard(const vector<string>&) { TouchDevice::OpenKeyboard(); }

void Shell::clipboard(const vector<string> &a) {
    if (a.empty()) INFO(Clipboard::Get());
    else Clipboard::Set(Join(a, " "));
}

void Shell::consolecolor(const vector<string>&) {
    delete screen->console->font;
    screen->console->font = Fonts::Get(FLAGS_default_font, "", 9, Color::black);
}

void Shell::startcmd(const vector<string> &a) {
    if (a.empty()) return;
    screen->console->startcmd = Join(a, " ");
}

void Shell::dldir(const vector<string>&) { INFO(LFAppDownloadDir()); }

void Shell::screenshot(const vector<string> &a) {
    if (a.empty()) return INFO("usage: screenshot <file> [tex_id]");
    Texture tex;
    if (a.size() == 1) tex.Screenshot();
    else               tex.DumpGL(atoi(a[1]));
    LocalFile lf(a[0], "w");
    PngWriter::Write(&lf, tex);
}

void Shell::fillmode(const vector<string>&) {
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
    // glPolygonMode(GL_FRONT_AND_BACK, app->fillMode.next());
#endif
}

void Shell::grabmode(const vector<string> &a) { if (app->grab_mode.Next()) mousein(a); else mouseout(a); }
void Shell::texmode(const vector<string>&) { if (app->tex_mode.Next()) screen->gd->EnableTexture(); else screen->gd->DisableTexture(); }
void Shell::swapaxis(const vector<string>&) { screen->SwapAxis(); }

void Shell::campos(const vector<string>&) {
    INFO("camMain.pos=",  screen->cam->pos.DebugString(),
         " camMain.ort=", screen->cam->ort.DebugString(),
         " camMain.up=",  screen->cam->up .DebugString());
}

void Shell::snap(const vector<string> &arg) {
    Asset      *a  = asset     (arg.size() ? arg[0] : "snap"); 
    SoundAsset *sa = soundasset(arg.size() ? arg[0] : "snap");
    if (a && sa) { app->audio.Snapshot(sa); glSpectogram(sa, a); }
}

void Shell::play(const vector<string> &arg) {
    SoundAsset *sa     = arg.size() > 0 ? soundasset(arg[0]) : soundasset("snap");
    int         offset = arg.size() > 1 ?       atoi(arg[1]) : -1;
    int         len    = arg.size() > 2 ?       atoi(arg[2]) : -1;
    if (sa) app->audio.QueueMix(sa, MixFlag::Reset, offset, len);
}

void Shell::playmovie(const vector<string> &arg) {
    MovieAsset *ma = arg.size() ? movieasset(arg[0]) : 0;
    if (ma) ma->Play(0);
}

void Shell::loadsound(const vector<string> &arg) {
    static int id = 1;
    if (arg.empty()) return;
    SoundAsset *a = new SoundAsset();
    a->filename = arg[0];
    a->name = StrCat("sa", id++);
    a->Load();
    INFO("loaded ", a->name);
}

void Shell::loadmovie(const vector<string> &arg) {
    static int id = 1;
    if (arg.empty()) return;
    MovieAsset *ma = new MovieAsset();
    SoundAsset *a = &ma->audio;
    ma->audio.name = ma->video.name = StrCat("ma", id++);
    ma->Load(arg[0].c_str());
    INFO("loaded ", ma->name);
    if (a->wav) INFO("loaded ", a->name, " : ", a->filename, " chans=", a->channels, " sr=", a->sample_rate, " ", a->wav->ring.size);
}

void Shell::copy(const vector<string> &arg) {
    SoundAsset *src = 0, *dst = 0;
    if (!(src = app->shell.soundasset(arg.size() > 0 ? arg[0] : "")) ||
        !(dst = app->shell.soundasset(arg.size() > 1 ? arg[1] : ""))) { INFO("copy <src> <dst>"); return; }

    INFOf("copy %s %d %d %d %s %d %d %d",
         src->name.c_str(), src->sample_rate, src->channels, src->seconds,
         dst->name.c_str(), dst->sample_rate, dst->channels, dst->seconds);

    RingBuf::Handle srch(src->wav), dsth(dst->wav);
    dsth.CopyFrom(&srch);
}

void shell_filter(const vector<string> &arg, bool FFTfilter, int taps, int hop=0) {
    SoundAsset *sa=0; vector<double> filter; double cutoff=0;

    if (arg.size() > 0) sa     = app->shell.soundasset(arg[0]);
    if (arg.size() > 2) cutoff = atof(arg[2]);
    if (arg.size() > 1) {
        filter.resize(taps);
        if        (arg[1] == "low") {
            for (int i=0; i<taps; i++) filter[i] = LowPassFilter(taps, i, (int)cutoff);
        } else if (arg[1] == "high") {
            for (int i=0; i<taps; i++) filter[i] = HighPassFilter(taps, i, (int)cutoff);
        } else if (arg[1] == "preemph") {
            taps = 2;
            filter = PreEmphasisFilter();
        }
    }
    if (arg.size() > 3) { taps = atoi(arg[3]); hop = taps/2; }

    if (!sa || filter.empty() || !taps) {
        INFO("filter <asset> <low,high> <cutoff> [taps]");
        return;
    }

    RingBuf filtered(sa->wav->samplesPerSec, sa->wav->ring.size);
    RingBuf::Handle I(sa->wav), O(&filtered);
    
    if (FFTfilter) {
        FFTFilterCompile(taps, &filter[0]);
        if (FFTFilter(&I, &O, taps, hop, &filter[0])) return;
    }
    else {
        if (LFL::Filter(&I, &O, taps, &filter[0])) return;
    }

    if (1) {
        int N=20; string b="input = ";
        for (int i=0; i<N; i++) StringAppendf(&b, "x[%d]=%f, ", i, I.Read(i));
        INFO(b);

        b = "output = ";
        for (int i=0; i<N; i++) StringAppendf(&b, "y[%d]=%f, ", i, O.Read(i));
        INFO(b);
    }

    app->audio.QueueMixBuf(&O);
}

void Shell::filter   (const vector<string> &arg) { shell_filter(arg, false, 16); }
void Shell::fftfilter(const vector<string> &arg) { shell_filter(arg, true, FLAGS_feat_window, FLAGS_feat_hop); }

void Shell::f0(const vector<string> &arg) {
    SoundAsset *sa=0; int offset=0; int method=F0EstmMethod::Default;

    if (arg.size() > 0) sa = app->shell.soundasset(arg[0]);
    if (arg.size() > 1) offset = atoi(arg[1]);
    if (arg.size() > 2) method = atoi(arg[2]);

    if (!sa || !sa->wav || sa->wav->ring.size < offset+FLAGS_feat_window) {
        INFO("f0 <asset> <offset>");
        return;
    }

    if (offset) {
        RingBuf::Handle I(sa->wav, offset);
        float f0 = FundamentalFrequency(&I, FLAGS_feat_window, 0, method);
        INFO("f0 = (", sa->name, ":", offset, ") = ", f0);    
    }
    else {
        RingBuf::Handle I(sa->wav, offset);
        Matrix *f0 = F0Stream(&I, 0, FLAGS_feat_window, FLAGS_feat_hop, method);
        for (int i=0; i<f0->M; /**/) {
            char buf[1024]; int len=0;
            for (int j=0; j<20 && i<f0->M; j++,i++) len += sprint(buf+len, sizeof(buf)-len, "%.2f, ", f0->row(i)[0]);
            INFO(buf);
        }
    }
}

void Shell::sinth(const vector<string> &a) { 
    int hz[3] = { 440, 0, 0};
    for (int i=0; i<sizeofarray(hz) && i<a.size(); i++) hz[i] = atof(a[i]);
    Sinthesize(&app->audio, hz[0], hz[1], hz[2]);
}

void Shell::writesnap(const vector<string> &a) {
    SoundAsset *sa = app->shell.soundasset(a.size() ? a[0] : "snap");
    if (sa) {
        string filename = StrCat(LFAppDownloadDir(), "snap.wav"); 
        RingBuf::Handle B(sa->wav);
        LocalFile lf(filename, "r");
        WavWriter w(&lf);
        int ret = w.Write(&B);
        INFO("wrote ", filename, " ret ", ret);
    }
}

void Shell::fps(const vector<string>&) { INFO("FPS ", FPS()); }

void Shell::wget(const vector<string> &a) {
    if (a.empty()) return;
    Singleton<HTTPClient>::Get()->WGet(a[0]);
}

void Shell::MessageBox(const vector<string> &a) { Dialog::MessageBox(Join(a, " ")); }
void Shell::TextureBox(const vector<string> &a) { Dialog::TextureBox(a.size() ? a[0] : ""); }

void Shell::Slider(const vector<string> &a) {
    if (a.empty()) { INFO("slider <flag_name> [total] [inc]"); return; }
    string flag_name = a[0];
    float total = a.size() >= 1 ? atof(a[1]) : 0;
    float inc   = a.size() >= 2 ? atof(a[2]) : 0;
    new SliderTweakDialog(flag_name, total ? total : 100, inc ? inc : 1);
}

void Shell::Edit(const vector<string> &a) {
    string s = LocalFile::FileContents(StrCat(ASSETS_DIR, "lfapp_vertex.glsl"));
    new EditorDialog(screen, Fonts::Default(), new BufferFile(s));
}

void Shell::cmds(const vector<string>&) {
    for (auto i = command.begin(); i != command.end(); ++i) INFO(i->name);
}

void Shell::flags(const vector<string>&) { Singleton<FlagMap>::Get()->Print(); }

void Shell::binds(const vector<string>&) { }

}; // namespace LFL
