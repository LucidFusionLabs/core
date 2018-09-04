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

#import <ScreenSaver/ScreenSaver.h>
#import <OpenGL/gl.h>
#include <AppKit/NSColor.h>
#include <AppKit/NSColorSpace.h>
#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/osx_common.h"

namespace LFL {
struct OSXScreensaverWindow;
};

@interface SaverView : ScreenSaverView
  @property (nonatomic, assign) LFL::OSXScreensaverWindow *screen;
  @property (nonatomic, retain) NSOpenGLPixelFormat *pixel_format;
  @property (nonatomic, retain) NSOpenGLContext *context;
@end

namespace LFL {
struct OSXScreensaverWindow : public Window {
  SaverView *view=0;
  NSOpenGLContext *gl=0;
  OSXScreensaverWindow(Application *app) : Window(app) {}
  virtual ~OSXScreensaverWindow() { ClearChildren(); }

  void SetCaption(const string &c) override {}
  void SetResizeIncrements(float x, float y) override {}
  void SetTransparency(float v) override {}
  bool Reshape(int w, int h) override { return true; }
  void Wakeup(int flag=0) override {}

  int Swap() override {
    gd->Flush();
    // [view.context flushBuffer];
    CGLFlushDrawable([view.context CGLContextObj]);
    gd->CheckForError(__FILE__, __LINE__);
    return 0;
  }
};
}; // namespace LFL

static LFL::Application *saver_app;
static bool ran_saver_main;

@implementation SaverView
  {
    BOOL needs_reshape, seen_frame;
  };

  - (void)dealloc {
    INFOf("Dealloc SaverView: %p", self);
    if (_screen) _screen->parent->CloseWindow(_screen);
    [super dealloc];
  }

  - (id)initWithFrame:(NSRect)frame isPreview:(BOOL)isPreview {
    self = [super initWithFrame:frame isPreview:isPreview];
    if (self) {
      if (!saver_app) {
        static const char *argv[] = { "Screensaver", nullptr };
        saver_app = static_cast<LFL::Application*>(MyAppCreate(1, argv));
      }
      INFOf("Creating SaverView: %p", self);
    }
    return self;
  }

  - (void)animateOneFrame {
    [self setNeedsDisplay:YES];
  }

  - (void)drawRect:(NSRect)rect { 
    INFOf("drawRect %p first=%d", self, !seen_frame);
    if (!seen_frame && (seen_frame = true)) {
      NSOpenGLPixelFormatAttribute attributes[] = { NSOpenGLPFADepthSize, LFL::FLAGS_depth_buffer_bits, 0 };
      _pixel_format = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];

      NSOpenGLContext *prev_context = saver_app->focused ?
        dynamic_cast<LFL::OSXScreensaverWindow*>(saver_app->focused)->gl : nullptr;
      INFOf("creating context sharing %p", prev_context);
      _context = [[NSOpenGLContext alloc] initWithFormat:_pixel_format shareContext:prev_context];
      needs_reshape = YES;
      GLint swapInt = 1;
      [_context setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];
      [_context setView:self];
      [_context makeCurrentContext];

      if (!ran_saver_main && (ran_saver_main = true)) {
        int ret = MyAppMain(saver_app);
        if (ret) exit(ret);
        INFOf("SaverView=%p Main ret=%d\n", self, ret);
      } else {
        auto w = saver_app->CreateNewWindow();
        saver_app->MakeCurrentWindow(w);
        if (!prev_context) saver_app->ResetGL(LFL::ResetGLFlag::Delete | LFL::ResetGLFlag::Reload);
      }

      _screen = dynamic_cast<LFL::OSXScreensaverWindow*>(saver_app->focused);
      _screen->gl = _context;
    }

    auto app = _screen->parent;
    if (app->focused != static_cast<LFL::Window*>(_screen)) {
      INFOf("Changing focus to %p", self);
      app->SetFocusedWindow(_screen);
    }

    if (needs_reshape && !(needs_reshape=NO)) {
      [_context update];
      app->SetFocusedWindow(_screen);
      float screen_w = [self frame].size.width, screen_h = [self frame].size.height;
      _screen->Reshaped(LFL::point(screen_w, screen_h), LFL::Box(0, 0, int(screen_w), int(screen_h)));
    }
    app->EventDrivenFrame(true, true);
  }
@end

namespace LFL {
const int Key::Escape = 0x81;
const int Key::Return = '\r';
const int Key::Up = 0xBE;
const int Key::Down = 0xBD;
const int Key::Left = 0xBB;
const int Key::Right = 0xBC;
const int Key::LeftShift = 0x83;
const int Key::RightShift = 0x87;
const int Key::LeftCtrl = 0x86;
const int Key::RightCtrl = 0x89;
const int Key::LeftCmd = 0x82;
const int Key::RightCmd = -12;
const int Key::Tab = '\t';
const int Key::Space = ' ';
const int Key::Backspace = 0x80;
const int Key::Delete = -16;
const int Key::Quote = '\'';
const int Key::Backquote = '`';
const int Key::PageUp = 0xB4;
const int Key::PageDown = 0xB9;
const int Key::F1 = 0xBA;
const int Key::F2 = 0xB8;
const int Key::F3 = 0xA8;
const int Key::F4 = 0xB6;
const int Key::F5 = 0xA5;
const int Key::F6 = 0xA6;
const int Key::F7 = 0xA7;
const int Key::F8 = 0xA9;
const int Key::F9 = 0xAA;
const int Key::F10 = 0xAF;
const int Key::F11 = 0xAB;
const int Key::F12 = 0xB0;
const int Key::Home = 0xB3;
const int Key::End = 0xB7;
const int Key::Insert = -1;

const int Texture::updatesystemimage_pf = Pixel::RGB24;

struct OSXScreensaverFrameworkModule : public Framework {
  WindowHolder *window;
  OSXScreensaverFrameworkModule(WindowHolder *w) : window(w) {}

  int Init() override {
    INFO("OSXScreensaverFrameworkModule::Init()");
    CHECK(CreateWindow(window, window->focused));
    window->MakeCurrentWindow(window->focused);
    return 0;
  }

  unique_ptr<Window> ConstructWindow(Application *app) override { return make_unique<OSXScreensaverWindow>(app); }
  void StartWindow(Window *W) override {}
  bool CreateWindow(WindowHolder *H, Window *W) override { return true; }
};

void TouchKeyboard::ToggleTouchKeyboard() {}
void TouchKeyboard::OpenTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboardAfterReturn(bool v) {}
void TouchKeyboard::SetTouchKeyboardTiled(bool v) {}

void MouseFocus::ReleaseMouseFocus() {}
void MouseFocus::GrabMouseFocus() {}

void Clipboard::SetClipboardText(const string &s) {}
string Clipboard::GetClipboardText() { return string(); }

void ThreadDispatcher::RunCallbackInMainThread(Callback cb) {
  message_queue.Write(make_unique<Callback>(move(cb)).release());
  if (!FLAGS_target_fps) wakeup->Wakeup();
}

void WindowHolder::SetAppFrameEnabled(bool) {}
void WindowHolder::MakeCurrentWindow(Window *W) { 
  if (!(focused = W)) return;
  [dynamic_cast<OSXScreensaverWindow*>(W)->view.context makeCurrentContext];
  if (W->gd) W->gd->MarkDirty();
}

void Application::CloseWindow(Window *W) {
  windows.erase(W->id);
  if (window_closed_cb) window_closed_cb(W);
  focused = 0;
}

void Application::LoseFocus() {}
int Application::Suspended() { return 0; }
int Application::SetExtraScale(bool v) { return false; }
void Application::SetDownScale(bool v) {}
void Application::SetTitleBar(bool v) {}
void Application::SetKeepScreenOn(bool v) {}
void Application::SetAutoRotateOrientation(bool v) {}
void Application::SetVerticalSwipeRecognizer(int touches) {}
void Application::SetHorizontalSwipeRecognizer(int touches) {}
void Application::SetPanRecognizer(bool enabled) {}
void Application::SetPinchRecognizer(bool enabled) {}
void Application::ShowSystemStatusBar(bool v) {}
void Application::SetTheme(const string &v) {}
void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &choose_cb) {}
void Application::ShowSystemFileChooser(bool choose_files, bool choose_dirs, bool choose_multi, const StringVecCB &choose_cb) {}
void Application::ShowSystemContextMenu(const MenuItemVec &items) {}

FrameScheduler::FrameScheduler(WindowHolder *w) :
  window(w), maxfps(&FLAGS_target_fps), rate_limit(0), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(0), run_main_loop(0) {}

bool FrameScheduler::DoMainWait(bool only_poll) { return false; }
void FrameScheduler::UpdateWindowTargetFPS(Window *w) { }
void FrameScheduler::AddMainWaitMouse(Window *w) {}
void FrameScheduler::DelMainWaitMouse(Window *w) {}
void FrameScheduler::AddMainWaitKeyboard(Window *w) {}
void FrameScheduler::DelMainWaitKeyboard(Window *w) {}
void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()> cb) {}
void FrameScheduler::DelMainWaitSocket(Window *w, Socket fd) {}

unique_ptr<Framework> Framework::Create(Application *a) { return make_unique<OSXScreensaverFrameworkModule>(a); }
unique_ptr<TimerInterface> SystemToolkit::CreateTimer(Callback cb) { return make_unique<AppleTimer>(move(cb)); }
unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(Window *w, AlertItemVec items) { return nullptr; }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(Window*, const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(Window*, const string &title, MenuItemVec items) { return nullptr; }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(Window*, MenuItemVec items) { return nullptr; }

}; // namespace LFL
