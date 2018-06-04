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

#import <Cocoa/Cocoa.h>
#import <OpenGL/gl.h>
#include <AppKit/NSColor.h>
#include <AppKit/NSColorSpace.h>
#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/osx_common.h"

@implementation GameView
  {
    NSOpenGLContext *context;
    NSOpenGLPixelFormat *pixel_format;

    NSTimer *runloop_timer;
    CVDisplayLinkRef displayLink;

    NSPoint prev_mouse_pos;
    BOOL initialized, needs_reshape, needs_frame;
    BOOL use_timer, use_display_link, video_thread_init;
    BOOL cmd_down, ctrl_down, shift_down;
  };

  - (void)dealloc {
    if (runloop_timer) ERRORf("%s", "runloop_timer remaining");
    CVDisplayLinkRelease(displayLink);
    [pixel_format release];
    [context release];
    [super dealloc];
  }

  - (id)initWithFrame:(NSRect)frame pixelFormat:(NSOpenGLPixelFormat*)format {
    [super initWithFrame:frame];
    _main_wait_fh = [[NSMutableDictionary alloc] init];
    pixel_format = [format retain];
    use_timer = _should_close = 1;
    return self;
  }

  + (NSOpenGLPixelFormat *)defaultPixelFormat {
    NSOpenGLPixelFormatAttribute attributes[] = { NSOpenGLPFADepthSize, LFL::FLAGS_depth_buffer_bits, 0 };
    return [[[NSOpenGLPixelFormat alloc] initWithAttributes:attributes] autorelease];
  }

  - (NSOpenGLPixelFormat *)pixelFormat { return pixel_format; }
  - (BOOL)isOpaque { return YES; }
  - (BOOL)acceptsFirstResponder { return YES; }
  - (BOOL)isInitialized { return initialized; }
  - (void)setFrameSize:(NSSize)s { [super setFrameSize:s]; needs_reshape=YES; [self update]; }

  - (void)viewDidChangeBackingProperties {
    float ms = [[NSScreen mainScreen] backingScaleFactor];
    self.layer.contentsScale = [[self window] backingScaleFactor];
    INFOf("viewDidChangeBackingProperties %f %f", self.layer.contentsScale, ms);
  }

  - (void)update { [context update]; }
  - (void)lockFocus {
    [super lockFocus];
    CGLLockContext([context CGLContextObj]);
    if ([context view] != self) [context setView:self];
    _screen->parent->SetFocusedWindow(_screen);
    if (needs_reshape) { [self reshape]; needs_reshape=NO; }
  }

  - (void)unlockFocus {
    [super unlockFocus];
    CGLUnlockContext([context CGLContextObj]);
  }

  - (void)prepareOpenGL {
    GLint swapInt = 1, opaque = 0;
    // [self setWantsBestResolutionOpenGLSurface:YES];
    [[self openGLContext] setValues:&swapInt forParameter:NSOpenGLCPSwapInterval];
    // [[self openGLContext] setValues:&opaque forParameter:NSOpenGLCPSurfaceOpacity];
    if (use_display_link) {
      CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
      CVDisplayLinkSetOutputCallback(displayLink, &MyDisplayLinkCallback, self);
      CGLContextObj cglContext = [[self openGLContext] CGLContextObj];
      // CGLEnable(cglContext, kCGLCEMPEngine);
      CGLPixelFormatObj cglPixelFormat = [[self pixelFormat] CGLPixelFormatObj];
      CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink, cglContext, cglPixelFormat);
    }
  }

  - (NSOpenGLContext *)openGLContext {
    if (context == nil) {
      context = [self createGLContext];
      dynamic_cast<LFL::OSXWindow*>(_screen)->gl = context;
      needs_reshape = YES;
    }
    return context;
  }

  - (NSOpenGLContext *)createGLContext {
    NSOpenGLContext *prev_context = dynamic_cast<LFL::OSXWindow*>(_screen->parent->focused)->gl;
    NSOpenGLContext *ret = [[NSOpenGLContext alloc] initWithFormat:pixel_format shareContext:prev_context];
    [ret setView:self];
    return ret;
  }

  - (void)startThread:(bool)first {
    initialized = true;
    if (!LFL::FLAGS_enable_video) {
      if (first) INFOf("OSXModule impl = %s", "PassThru");
      exit(_screen->parent->MainLoop());
    } else if (_screen->target_fps == 0) {
      if (first) INFOf("OSXModule impl = %s", "MainWait");
      [self setNeedsDisplay:YES];
    } else if (use_display_link) {
      if (first) INFOf("OSXModule impl = %s", "DisplayLink");
      CVDisplayLinkStart(displayLink);
    } else if (use_timer) {
      if (first) INFOf("OSXModule impl = %s", "NSTimer");
      runloop_timer = [NSTimer timerWithTimeInterval: 1.0/_screen->target_fps
          target:self selector:@selector(runloopTimerFired:) userInfo:nil repeats:YES];
      [[NSRunLoop currentRunLoop] addTimer:runloop_timer forMode:NSDefaultRunLoopMode];
      // [[NSRunLoop currentRunLoop] addTimer:runloop_timer forMode:NSEventTrackingRunLoopMode];
    }
  }

  - (void)stopThread {
    if (displayLink) { CVDisplayLinkStop(displayLink); displayLink = nil; }
    if (runloop_timer) { [runloop_timer invalidate]; runloop_timer = nil; }
  }

  - (void)runloopTimerFired:(id)sender { [self setNeedsDisplay:YES]; }
  - (void)drawRect:(NSRect)dirtyRect { if (use_timer) [self getFrameForTime:0]; }
  - (void)getFrameForTime:(const CVTimeStamp*)outputTime {
    static bool first_callback = 1;
    auto a = _screen->parent;
    if (first_callback && !(first_callback = 0)) a->SetMainThread();
    if (!initialized) return;
    if (a->run) a->EventDrivenFrame(true, true);
    else [[NSApplication sharedApplication] terminate:self];
  }
   
  - (void)reshape {
    if (!initialized) return;
    [context update];
    _screen->parent->SetFocusedWindow(_screen);
    float screen_w = [self frame].size.width, screen_h = [self frame].size.height;
    _screen->Reshaped(LFL::point(screen_w, screen_h), LFL::Box(0, 0, int(screen_w), int(screen_h)));
  }

  - (void)addMainWaitSocket:(int)fd callback:(std::function<bool()>)cb {
    [_main_wait_fh
      setObject: [[[ObjcFileHandleCallback alloc] initWithCB:move(cb) forWindow:self.window.contentView
        fileDescriptor:fd] autorelease]
      forKey:    [NSNumber numberWithInt: fd]];
  }

  - (void)delMainWaitSocket: (int)fd {
    [_main_wait_fh removeObjectForKey: [NSNumber numberWithInt: fd]];
  }

  - (void)callbackRun:(id)sender { [(ObjcCallback*)[sender representedObject] run]; }
  - (void)mouseDown:(NSEvent*)e { [self mouseClick:e down:1]; }
  - (void)mouseUp:  (NSEvent*)e { [self mouseClick:e down:0]; }
  - (void)mouseClick:(NSEvent*)e down:(bool)d {
    auto a = _screen->parent;
    a->SetFocusedWindow(_screen);
    NSPoint p = [e locationInWindow];
    int fired = a->input->MouseClick(1+ctrl_down, d, LFL::point(p.x, p.y));
    if (fired && _frame_on_mouse_input) [self setNeedsDisplay:YES]; 
    prev_mouse_pos = p;
  }

  - (void)mouseDragged:(NSEvent*)e { [self mouseMove:e drag:1]; }
  - (void)mouseMoved:  (NSEvent*)e { [self mouseMove:e drag:0]; }
  - (void)mouseMove: (NSEvent*)e drag:(bool)drag {
    auto a = _screen->parent;
    a->SetFocusedWindow(_screen);
    if (_screen->cursor_grabbed) {
      a->input->MouseMove(LFL::point(prev_mouse_pos.x, prev_mouse_pos.y), LFL::point([e deltaX], -[e deltaY]));
    } else {
      NSPoint p=[e locationInWindow], d=NSMakePoint(p.x-prev_mouse_pos.x, p.y-prev_mouse_pos.y);
      int fired = a->input->MouseMove(LFL::point(p.x, p.y), LFL::point(d.x, d.y));
      if (fired && _frame_on_mouse_input) [self setNeedsDisplay:YES]; 
      prev_mouse_pos = p;
    }
  }

  - (void)magnifyWithEvent:(NSEvent*)event {
    CGFloat p_scale = [event magnification] + 1.0;
    auto a = _screen->parent;
    LFL::v2 d(p_scale, p_scale);
    int fired = a->input->MouseZoom(a->focused ? a->focused->mouse : LFL::point(), d, 0);
    if (fired && _frame_on_mouse_input) [self setNeedsDisplay:YES]; 
  }

  - (void)swipeWithEvent:(NSEvent*)event {
    LFL::v2 d([event deltaX], [event deltaY]);
    int fired = _screen->parent->input->MouseWheel(d, d);
    if (fired && _frame_on_mouse_input) [self setNeedsDisplay:YES]; 
  }

  - (void)keyDown:(NSEvent *)theEvent { [self keyPress:theEvent down:1]; }
  - (void)keyUp:  (NSEvent *)theEvent { [self keyPress:theEvent down:0]; }
  - (void)keyPress:(NSEvent *)theEvent down:(bool)d {
    auto a = _screen->parent;
    a->SetFocusedWindow(_screen);
    int c = getKeyCode([theEvent keyCode]);
    int fired = c ? a->input->KeyPress(c, 0, d) : 0;
    if (fired && _frame_on_keyboard_input) [self setNeedsDisplay:YES]; 
  }

  - (void)flagsChanged:(NSEvent *)theEvent {
    int flags = [theEvent modifierFlags];
    bool cmd = flags & NSEventModifierFlagCommand, ctrl = flags & NSEventModifierFlagControl, shift = flags & NSEventModifierFlagShift;
    auto a = _screen->parent;
    if (cmd   != cmd_down)   { a->input->KeyPress(0x82, 0, cmd);   cmd_down   = cmd;   }
    if (ctrl  != ctrl_down)  { a->input->KeyPress(0x86, 0, ctrl);  ctrl_down  = ctrl;  }
    if (shift != shift_down) { a->input->KeyPress(0x83, 0, shift); shift_down = shift; }
  }

  - (void)clearKeyModifiers {
    auto a = _screen->parent;
    if (cmd_down)   { a->input->KeyPress(0x82, 0, 0); cmd_down   = 0; }
    if (ctrl_down)  { a->input->KeyPress(0x86, 0, 0); ctrl_down  = 0; }
    if (shift_down) { a->input->KeyPress(0x83, 0, 0); shift_down = 0; }
  }

  static int getKeyCode(int c) {
    static unsigned char keyCode[] = {
      'a',  /* 0x00: kVK_ANSI_A */             's',  /* 0x01: kVK_ANSI_S */               'd',  /* 0x02: kVK_ANSI_D */
      'f',  /* 0x03: kVK_ANSI_F */             'h',  /* 0x04: kVK_ANSI_H */               'g',  /* 0x05: kVK_ANSI_G */
      'z',  /* 0x06: kVK_ANSI_Z */             'x',  /* 0x07: kVK_ANSI_X */               'c',  /* 0x08: kVK_ANSI_C */
      'v',  /* 0x09: kVK_ANSI_V */             0,    /* 0x0A: */                          'b',  /* 0x0B: kVK_ANSI_B */
      'q',  /* 0x0C: kVK_ANSI_Q */             'w',  /* 0x0D: kVK_ANSI_W */               'e',  /* 0x0E: kVK_ANSI_E */
      'r',  /* 0x0F: kVK_ANSI_R */             'y',  /* 0x10: kVK_ANSI_Y */               't',  /* 0x11: kVK_ANSI_T */
      '1',  /* 0x12: kVK_ANSI_1 */             '2',  /* 0x13: kVK_ANSI_2 */               '3',  /* 0x14: kVK_ANSI_3 */
      '4',  /* 0x15: kVK_ANSI_4 */             '6',  /* 0x16: kVK_ANSI_6 */               '5',  /* 0x17: kVK_ANSI_5 */
      '=',  /* 0x18: kVK_ANSI_Equal */         '9',  /* 0x19: kVK_ANSI_9 */               '7',  /* 0x1A: kVK_ANSI_7 */
      '-',  /* 0x1B: kVK_ANSI_Minus */         '8',  /* 0x1C: kVK_ANSI_8 */               '0',  /* 0x1D: kVK_ANSI_0 */
      ']',  /* 0x1E: kVK_ANSI_RightBracket */  'o',  /* 0x1F: kVK_ANSI_O */               'u',  /* 0x20: kVK_ANSI_U */
      '[',  /* 0x21: kVK_ANSI_LeftBracket */   'i',  /* 0x22: kVK_ANSI_I */               'p',  /* 0x23: kVK_ANSI_P */
      '\r', /* 0x24: kVK_Return */             'l',  /* 0x25: kVK_ANSI_L */               'j',  /* 0x26: kVK_ANSI_J */
      '\'', /* 0x27: kVK_ANSI_Quote */         'k',  /* 0x28: kVK_ANSI_K */               ';',  /* 0x29: kVK_ANSI_Semicolon */
      '\\', /* 0x2A: kVK_ANSI_Backslash */     ',',  /* 0x2B: kVK_ANSI_Comma */           '/',  /* 0x2C: kVK_ANSI_Slash */ 
      'n',  /* 0x2D: kVK_ANSI_N */             'm',  /* 0x2E: kVK_ANSI_M */               '.',  /* 0x2F: kVK_ANSI_Period */ 
      '\t', /* 0x30: kVK_Tab */                ' ',  /* 0x31: kVK_Space */                '`',  /* 0x32: kVK_ANSI_Grave */             
      0x80, /* 0x33: kVK_Delete */             0,    /* 0x34: */                          0x81, /* 0x35: kVK_Escape */ 
      0,    /* 0x36: */                        0x82, /* 0x37: kVK_Command */              0x83, /* 0x38: kVK_Shift */ 
      0x84, /* 0x39: kVK_CapsLock */           0x85, /* 0x3A: kVK_Option */               0x86, /* 0x3B: kVK_Control */ 
      0x87, /* 0x3C: kVK_RightShift */         0x88, /* 0x3D: kVK_RightOption */          0x89, /* 0x3E: kVK_RightControl */ 
      0x8A, /* 0x3F: kVK_Function */           0x8B, /* 0x40: kVK_F17 */                  0x8C, /* 0x41: kVK_ANSI_KeypadDecimal */
      0,    /* 0x42: */                        0x8D, /* 0x43: kVK_ANSI_KeypadMultiply */  0,    /* 0x44: */
      0x8E, /* 0x45: kVK_ANSI_KeypadPlus */    0x8F, /* 0x46: */                          0x90, /* 0x47: kVK_ANSI_KeypadClear */ 
      0x91, /* 0x48: kVK_VolumeUp */           0x92, /* 0x49: kVK_VolumeDown */           0x93, /* 0x4A: kVK_Mute */
      0x94, /* 0x4B: kVK_ANSI_KeypadDivide */  0x95, /* 0x4C: kVK_ANSI_KeypadEnter */     0,    /* 0x4D: */
      0x96, /* 0x4E: kVK_ANSI_KeypadMinus */   0x97, /* 0x4F: kVK_F18 */                  0x98, /* 0x50: kVK_F19 */
      0x99, /* 0x51: kVK_ANSI_KeypadEquals */  0x9A, /* 0x52: kVK_ANSI_Keypad0 */         0x9B, /* 0x53: kVK_ANSI_Keypad1 */ 
      0x9C, /* 0x54: kVK_ANSI_Keypad2 */       0x9D, /* 0x55: kVK_ANSI_Keypad3 */         0x9E, /* 0x56: kVK_ANSI_Keypad4 */ 
      0x9F, /* 0x57: kVK_ANSI_Keypad5 */       0xA0, /* 0x58: kVK_ANSI_Keypad6 */         0xA1, /* 0x59: kVK_ANSI_Keypad7 */            
      0xA2, /* 0x5A: kVK_F20 */                0xA3, /* 0x5B: kVK_ANSI_Keypad8 */         0xA4, /* 0x5C: kVK_ANSI_Keypad9 */ 
      0,    /* 0x5D: */                        0,    /* 0x5E: */                          0,    /* 0x5F: */
      0xA5, /* 0x60: kVK_F5 */                 0xA6, /* 0x61: kVK_F6 */                   0xA7, /* 0x62: kVK_F7 */ 
      0xA8, /* 0x63: kVK_F3 */                 0xA9, /* 0x64: kVK_F8 */                   0xAA, /* 0x65: kVK_F9 */ 
      0,    /* 0x66: */                        0xAB, /* 0x67: kVK_F11 */                  0,    /* 0x68: */
      0xAC, /* 0x69: kVK_F13 */                0xAD, /* 0x6A: kVK_F16 */                  0xAE, /* 0x6B: kVK_F14 */
      0,    /* 0x6C: */                        0xAF, /* 0x6D: kVK_F10 */                  0,    /* 0x6E: */
      0xB0, /* 0x6F: kVK_F12 */                0,    /* 0x70: */                          0xB1, /* 0x71: kVK_F15 */
      0xB2, /* 0x72: kVK_Help */               0xB3, /* 0x73: kVK_Home */                 0xB4, /* 0x74: kVK_PageUp */
      0xB5, /* 0x75: kVK_ForwardDelete */      0xB6, /* 0x76: kVK_F4 */                   0xB7, /* 0x77: kVK_End */
      0xB8, /* 0x78: kVK_F2 */                 0xB9, /* 0x79: kVK_PageDown */             0xBA, /* 0x7A: kVK_F1 */
      0xBB, /* 0x7B: kVK_LeftArrow */          0xBC, /* 0x7C: kVK_RightArrow */           0xBD, /* 0x7D: kVK_DownArrow */
      0xBE, /* 0x7E: kVK_UpArrow */            0     /* 0x7f: */                            
    };
    return (c >= 0 && c < 128) ? keyCode[c] : 0;
  }

  static CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp* now, const CVTimeStamp* outputTime, CVOptionFlags flagsIn, CVOptionFlags* flagsOut, void* displayLinkContext) {
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    [static_cast<GameView*>(displayLinkContext) getFrameForTime:outputTime];
    [pool release];
    return kCVReturnSuccess;
  }
@end

@implementation GameContainerView
  {
    LFL::Application *app;
  }

  - (id)initWithFrame:(NSRect)frame andApp:(LFL::Application*)a {
    self = [super initWithFrame: frame];
    app = a;
    return self;
  }

  // - (void)windowDidBecomeKey:(NSNotification *)notification {}
  // - (void)windowDidResignKey:(NSNotification *)notification {}
  // - (void)windowDidBecomeMain:(NSNotification *)notification {}
  - (void)windowDidResignMain:(NSNotification *)notification { [_gameView clearKeyModifiers]; }

  - (BOOL)windowShouldClose:(id)sender { return _gameView.should_close; }
  - (void)windowWillClose:(NSNotification *)notification { 
    [_gameView stopThread];
    auto wh = _gameView.screen->parent;
    wh->SetFocusedWindow(_gameView.screen);
    wh->CloseWindow(_gameView.screen);
    if (wh->windows.empty()) app->Exit();
  }

  - (void)objcWindowSelect { _gameView.screen->parent->SetFocusedWindow(_gameView.screen); }
  - (void)objcWindowFrame { [_gameView getFrameForTime:nil]; } 
@end

// AppDelegate
@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate
  {
    LFL::Application *app;
  }

  - (id) initWithApp: (LFL::Application*) a {
    self = [super init];
    app = a;
    return self;
  }

  - (void)applicationWillTerminate: (NSNotification *)aNotification {}
  - (void)applicationDidFinishLaunching: (NSNotification *)aNotification {
    [[NSUserDefaults standardUserDefaults] registerDefaults:@{ @"NSApplicationCrashOnExceptions": @YES }];
    // [[NSFileManager defaultManager] changeCurrentDirectoryPath: [[NSBundle mainBundle] resourcePath]];
    int ret = MyAppMain(app);
    if (ret) exit(ret);
    INFOf("OSXFramework::Main ret=%d\n", ret);
  }

  - (GameView*)createWindow: (int)w height:(int)h nativeWindow:(LFL::Window*)s {
    NSWindow *window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, w, h)
                                         styleMask:NSWindowStyleMaskClosable|NSWindowStyleMaskMiniaturizable|NSWindowStyleMaskResizable|NSWindowStyleMaskTitled
                                         backing:NSBackingStoreBuffered defer:NO];

    GameContainerView *container_view = [[[GameContainerView alloc] initWithFrame:window.frame andApp:app] autorelease];
    container_view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [container_view setAutoresizesSubviews:YES];
    [window setContentView:container_view];
    [window setDelegate:container_view];

    GameView *view = [[[GameView alloc] initWithFrame:container_view.frame pixelFormat:GameView.defaultPixelFormat] autorelease];
    view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    view.screen = s;
    [[view openGLContext] setView:view];
    [container_view addSubview: view];
    container_view.gameView = view;

    [window center];
    [window makeKeyAndOrderFront:nil];
    [window setMinSize:NSMakeSize(256, 256)];
    [window setReleasedWhenClosed: YES];
    if (1) {
      window.acceptsMouseMovedEvents = YES;
      [window makeFirstResponder:view];
    }
    [NSApp activateIgnoringOtherApps:YES];
    INFOf("OSXVideoModule: AppDelegate::createWindow(%d, %d).id = %p", w, h, view);
    return view;
  }

  - (void)destroyWindow: (NSWindow*)window {
    [window setContentView:nil];
    [window close];
  }

  - (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)theApplication {
    return YES;
  }
@end

@interface OSXAlert : NSObject<NSTextFieldDelegate>
  @property (nonatomic, retain) NSAlert     *alert;
  @property (nonatomic, retain) NSTextField *input;
  @property (nonatomic)         bool         add_text;
  @property (nonatomic)         std::string  style;
  @property (nonatomic, assign) LFL::StringCB cancel_cb, confirm_cb;
@end

@implementation OSXAlert
  - (id)init:(LFL::AlertItemVec) kv {
    CHECK_EQ(4, kv.size());
    CHECK_EQ("style", kv[0].first);
    _style      = move(kv[0].second);
    _cancel_cb  = move(kv[2].cb);
    _confirm_cb = move(kv[3].cb);
    _alert = [[NSAlert alloc] init];
    [_alert addButtonWithTitle: [NSString stringWithUTF8String: kv[3].first.c_str()]];
    [_alert addButtonWithTitle: [NSString stringWithUTF8String: kv[2].first.c_str()]];
    [_alert setMessageText:     [NSString stringWithUTF8String: kv[1].first.c_str()]];
    [_alert setInformativeText: [NSString stringWithUTF8String: kv[1].second.c_str()]];
    [_alert setAlertStyle:NSAlertStyleWarning];
    if ((_add_text = _style == "textinput")) {
      _input = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 200, 24)];
      _input.delegate = self;
      [_alert setAccessoryView: _input];
    } else if ((_add_text = _style == "pwinput")) {
      _input = [[NSSecureTextField alloc] initWithFrame:NSMakeRect(0, 0, 200, 24)];
      _input.delegate = self;
      [_alert setAccessoryView: _input];
    }

    return self;
  }

  - (void)alertDidEnd:(NSAlert *)alert returnCode:(NSInteger)returnCode contextInfo:(void *)contextInfo {
    if (returnCode != NSAlertFirstButtonReturn) { if (_cancel_cb) _cancel_cb(""); }
    else { if (_confirm_cb) _confirm_cb(_add_text ? [[_input stringValue] UTF8String] : ""); }
  }
  
  - (void)modalAlertDidEnd:(NSAlert *)alert returnCode:(NSInteger)returnCode contextInfo:(void *)contextInfo {
    NSWindow *window = (NSWindow*)contextInfo;
    [NSApp endSheet: window];
  }

  - (void)controlTextDidChange:(NSNotification *)notification {}
  - (void)controlTextDidEndEditing:(NSNotification *)notification {}
@end

@interface OSXPanel : NSObject
  @property(readonly, assign) NSWindow *window;
  - (void) show;
  - (void) addTextField:(NSTextField*)tf withCB:(LFL::StringCB)cb;
@end

@implementation OSXPanel
  {
    std::vector<std::pair<NSButton*,    LFL::StringCB>> buttons;
    std::vector<std::pair<NSTextField*, LFL::StringCB>> textfields;
  }

  - (id) initWithBox: (const LFL::Box&) b {
    self = [super init];
    _window = [[NSPanel alloc] initWithContentRect:NSMakeRect(b.x, b.y, b.w, b.h) 
      styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
      backing:NSBackingStoreBuffered defer:YES];
    [_window makeFirstResponder:nil];
    return self;
  }

  - (void) show {
    [_window setLevel:NSFloatingWindowLevel];
    [_window center];
    [_window makeKeyAndOrderFront:NSApp];
    [_window retain];
  }

  - (void) addButton: (NSButton*)button withCB: (LFL::StringCB)cb {
    [button setTarget: self];
    [button setAction: @selector(buttonAction:)];
    [button setTag: buttons.size()];
    buttons.emplace_back(button, move(cb));
  }
  
  - (void) addTextField: (NSTextField*)textfield withCB: (LFL::StringCB)cb {
    [textfield setTarget: self];
    [textfield setAction: @selector(textFieldAction:)];
    [textfield setTag: textfields.size()];
    textfields.emplace_back(textfield, move(cb));
  }

  - (void) buttonAction: (id)sender {
    int tag = [sender tag];
    CHECK_RANGE(tag, 0, buttons.size());
    buttons[tag].second([[sender stringValue] UTF8String]);
  }

  - (void) textFieldAction: (id)sender {
    int tag = [sender tag];
    CHECK_RANGE(tag, 0, textfields.size());
    textfields[tag].second([[sender stringValue] UTF8String]);
  }
@end

@interface OSXFontChooser : NSObject
@end

@implementation OSXFontChooser
  {
    NSFont *font;
    LFL::StringVecCB font_change_cb;
  }

  - (void)selectFont: (const char *)name size:(int)s cb:(LFL::StringVecCB)v {
    font_change_cb = move(v);
    font = [NSFont fontWithName:[NSString stringWithUTF8String:name] size:s];
    NSFontManager *fontManager = [NSFontManager sharedFontManager];
    [fontManager setSelectedFont:font isMultiple:NO];
    [fontManager setTarget:self];
    [fontManager orderFrontFontPanel:self];
  }

  - (void)changeFont:(id)sender {
    font = [sender convertFont:font];
    float size = [[[font fontDescriptor] objectForKey:NSFontSizeAttribute] floatValue];
    font_change_cb(LFL::StringVec{ [[font fontName] UTF8String], LFL::StrCat(size) });
  }
@end


extern "C" void ConvertColorFromGenericToDeviceRGB(const float *i, float *o) {
  double ib[4], ob[4], *ii = ib, *oi = ob;
  for (auto e = i + 4; i != e; ) *ii++ = *i++;
  NSColor *gen_color = [NSColor colorWithColorSpace:[NSColorSpace genericRGBColorSpace] components:ib count:4];
  NSColor *dev_color = [gen_color colorUsingColorSpaceName:NSDeviceRGBColorSpace];
  [dev_color getComponents: ob];
  for (auto e = o + 4; o != e; ) *o++ = *oi++;
}

extern "C" void OSXCreateSystemApplicationMenu() {
  NSMenuItem *item; 
  NSMenu *menu = [[NSMenu alloc] initWithTitle:@""];
  NSString *app_name = [[NSRunningApplication currentApplication] localizedName];
  item = [menu addItemWithTitle:[@"About " stringByAppendingString:app_name]
    action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
  [menu addItem:[NSMenuItem separatorItem]];
  item = [menu addItemWithTitle:@"New Window" action:nil keyEquivalent:@"n"];
  item = [menu addItemWithTitle:[@"Hide " stringByAppendingString:app_name]
    action:@selector(hide:) keyEquivalent:@"h"];
  item = [menu addItemWithTitle:@"Hide Others"
    action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
  [item setKeyEquivalentModifierMask:(NSEventModifierFlagOption|NSEventModifierFlagCommand)];
  item = [menu addItemWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""];
  [menu addItem:[NSMenuItem separatorItem]];
  item = [menu addItemWithTitle: [@"Quit " stringByAppendingString:app_name]
    action:@selector(terminate:) keyEquivalent:@"q"];
  item = [[NSMenuItem alloc] initWithTitle:@"" action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
}

static void AddNSMenuItems(NSMenu *menu, LFL::vector<LFL::MenuItem> items) {
  NSMenuItem *item;
  for (auto &i : items) { 
    const LFL::string &k=i.shortcut, &n=i.name;
    if (n == "<separator>") { [menu addItem:[NSMenuItem separatorItem]]; continue; }

    NSString *key = nil;
    if      (k == "<left>")  { unichar fk = NSLeftArrowFunctionKey;  key = [NSString stringWithCharacters:&fk length:1]; }
    else if (k == "<right>") { unichar fk = NSRightArrowFunctionKey; key = [NSString stringWithCharacters:&fk length:1]; }
    else if (k == "<up>"   ) { unichar fk = NSUpArrowFunctionKey;    key = [NSString stringWithCharacters:&fk length:1]; }
    else if (k == "<down>")  { unichar fk = NSDownArrowFunctionKey;  key = [NSString stringWithCharacters:&fk length:1]; }
    else key = [NSString stringWithUTF8String: k.c_str()];

    item = [menu addItemWithTitle: [NSString stringWithUTF8String: n.c_str()]
                 action:           (i.cb ? @selector(callbackRun:) : nil)
                 keyEquivalent:    key];
    if (i.cb) [item setRepresentedObject: [[ObjcCallback alloc] initWithCB: move(i.cb)]];
  }
}

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

struct OSXFrameworkModule : public Framework {
  WindowHolder *window;
  OSXFrameworkModule(WindowHolder *w) : window(w) {}

  int Init() override {
    INFO("OSXFrameworkModule::Init()");
    CHECK(CreateWindow(window, window->focused));
    window->MakeCurrentWindow(window->focused);
    return 0;
  }

  unique_ptr<Window> ConstructWindow(Application *app) override { return make_unique<OSXWindow>(app); }
  void StartWindow(Window *W) override { [dynamic_cast<OSXWindow*>(W)->view startThread:true]; }

  bool CreateWindow(WindowHolder *H, Window *W) override { 
    [dynamic_cast<OSXWindow*>(H->focused)->view clearKeyModifiers];
    W->id = (dynamic_cast<OSXWindow*>(W)->view =
             [static_cast<AppDelegate*>([NSApp delegate]) createWindow:W->gl_w height:W->gl_h nativeWindow:W]);
    if (W->id) H->windows[W->id] = W;
    W->SetCaption(W->caption);
    return true; 
  }

  // void *BeginGLContextCreate(Window *W) { return 0; }
  // void *CompleteGLContextCreate(Window *W, void *gl_context) { return [dynamic_cast<OSXWindow*>(W)->view createGLContext]; }
};

struct OSXAlertView : public AlertViewInterface {
  WindowHolder *win;
  OSXAlert *alert;
  ~OSXAlertView() { [alert release]; }
  OSXAlertView(WindowHolder *w, AlertItemVec items) : win(w), alert([[OSXAlert alloc] init: move(items)]) {}

  void Hide() {}
  void Show(const string &arg) {
    NSWindow *w = [dynamic_cast<OSXWindow*>(win->focused)->view window];
    if (alert.add_text) [alert.input setStringValue: MakeNSString(arg)];
    [alert.alert beginSheetModalForWindow:w completionHandler:^(NSModalResponse n) {
      [alert alertDidEnd:alert.alert returnCode:n contextInfo:w];
    } ];
    [dynamic_cast<OSXWindow*>(win->focused)->view  clearKeyModifiers];
  }

  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
    NSWindow *w = [dynamic_cast<OSXWindow*>(win->focused)->view window];
    alert.confirm_cb = move(confirm_cb);
    [alert.alert setMessageText: MakeNSString(title)];
    [alert.alert setInformativeText: MakeNSString(msg)];
    [alert.alert beginSheetModalForWindow:w completionHandler:^(NSModalResponse n) {
      [alert alertDidEnd:alert.alert returnCode:n contextInfo:w];
    } ];
    [dynamic_cast<OSXWindow*>(win->focused)->view  clearKeyModifiers];
  }

  string RunModal(const string &arg) {
    NSWindow *w = [dynamic_cast<OSXWindow*>(win->focused)->view window];
    if (alert.add_text) [alert.input setStringValue: MakeNSString(arg)];
    [alert.alert beginSheetModalForWindow:w completionHandler:^(NSModalResponse n) {
      [alert modalAlertDidEnd:alert.alert returnCode:n contextInfo:w];
    } ];
    [NSApp runModalForWindow: w];
    return alert.add_text ? GetNSString([alert.input stringValue]) : "";
  }
};

struct OSXMenuView : public MenuViewInterface {
  NSMenu *menu;
  ~OSXMenuView() { [menu release]; }
  OSXMenuView(const string &title, MenuItemVec items) :
    menu([[NSMenu alloc] initWithTitle: MakeNSString(title)]) {
    AddNSMenuItems(menu, move(items));
    NSMenuItem *item = [[NSMenuItem alloc] initWithTitle: MakeNSString(title) action:nil keyEquivalent:@""];
    [item setSubmenu: menu];
    [[NSApp mainMenu] addItem: item];
    [menu release];
    [item release];
  }
  void Show() {}
};

struct OSXPanelView : public PanelViewInterface {
  OSXPanel *panel;
  ~OSXPanelView() { [panel release]; }
  OSXPanelView(const Box &b, const string &title, PanelItemVec items) :
    panel([[OSXPanel alloc] initWithBox: b]) {
    [[panel window] setTitle: [NSString stringWithUTF8String: title.c_str()]];
    for (auto &i : items) {
      const Box &b = i.box;
      const string &t = i.type;
      if (t == "textbox") {
        NSTextField *textfield = [[NSTextField alloc] initWithFrame:NSMakeRect(b.x, b.y, b.w, b.h)];
        [panel addTextField: textfield withCB: move(i.cb)];
        [[[panel window] contentView] addSubview: textfield];
      } else if (PrefixMatch(t, "button:")) {
        NSButton *button = [[[NSButton alloc] initWithFrame:NSMakeRect(b.x, b.y, b.w, b.h)] autorelease];
        [panel addButton: button withCB: move(i.cb)];
        [[[panel window] contentView] addSubview: button];
        [button setTitle: [NSString stringWithUTF8String: t.substr(7).c_str()]];
        [button setButtonType:NSMomentaryLightButton];
        [button setBezelStyle:NSRoundedBezelStyle];
      } else ERROR("unknown panel item ", t);
    }
  }

  void Show() { [panel show]; }
  void SetTitle(const string &title) { 
    [[panel window] setTitle: [NSString stringWithUTF8String: title.c_str()]];
  }
};

void OSXWindow::SetCaption(const string &v) { 
  [view window].title = [NSString stringWithUTF8String:v.c_str()];
}

void OSXWindow::SetResizeIncrements(float x, float y) {
  [[view window] setContentResizeIncrements:
    NSMakeSize((resize_increment_x = x), (resize_increment_y = y))];
}

void OSXWindow::SetTransparency(float v) { 
  [[view window] setAlphaValue: 1.0 - v];
}

bool OSXWindow::Reshape(int w, int h) {
  NSWindow *window = [view window];
  NSRect frame = [window frame], x;
  LFL::Box b(frame.origin.x, frame.origin.y, w, h);
  if (resize_increment_x || resize_increment_y) {
    NSUInteger styleMask = [window styleMask];
    x = [NSWindow frameRectForContentRect: NSMakeRect(b.x, b.y, b.w, b.h) styleMask: styleMask];
    x = [window constrainFrameRect: x toScreen: [NSScreen mainScreen]];
    x = [NSWindow contentRectForFrameRect: x styleMask: styleMask];
    LFL::Box constrained(x.origin.x, x.origin.y, x.size.width, x.size.height); 
    if (b != constrained) {
      b = constrained;
      if (resize_increment_x) b.w -= (b.w % resize_increment_x);
      if (resize_increment_y) b.h -= (b.h % resize_increment_y);
    }
  }
  [window setContentSize:NSMakeSize(b.w, b.h)];
  return true;
}

void OSXWindow::Wakeup(int flag) {
  if (parent->scheduler.wait_forever) {
    if (flag & WakeupFlag::InMainThread) [view setNeedsDisplay:YES];
    else [view performSelectorOnMainThread:@selector(setNeedsDisplay:) withObject:@YES waitUntilDone:NO];
  }
}

int OSXWindow::Swap() {
  gd->Flush();
  // [[view openGLContext] flushBuffer];
  CGLFlushDrawable([[view openGLContext] CGLContextObj]);
  gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void ThreadDispatcher::RunCallbackInMainThread(Callback cb) {
  message_queue.Write(make_unique<Callback>(move(cb)).release());
  if (!FLAGS_target_fps) wakeup->Wakeup();
}

void WindowHolder::MakeCurrentWindow(Window *W) { 
  if (!(focused = W)) return;
  [[dynamic_cast<OSXWindow*>(W)->view openGLContext] makeCurrentContext];
  if (W->gd) W->gd->MarkDirty();
}

int Application::Suspended() { return 0; }
void Application::CloseWindow(Window *W) {
  windows.erase(W->id);
  if (windows.empty()) run = false;
  if (window_closed_cb) window_closed_cb(W);
  // [static_cast<AppDelegate*>([NSApp delegate]) destroyWindow: [dynamic_cast<OSXWindow*>(W)->view window]];
  focused = 0;
}

void Application::LoseFocus() { [dynamic_cast<OSXWindow*>(focused)->view clearKeyModifiers]; }

void MouseFocus::ReleaseMouseFocus() { 
  CGDisplayShowCursor(kCGDirectMainDisplay);
  CGAssociateMouseAndMouseCursorPosition(true);
  window->focused->grab_mode.Off();
  window->focused->cursor_grabbed = 0;
}

void MouseFocus::GrabMouseFocus() {
  CGDisplayHideCursor(kCGDirectMainDisplay);
  CGAssociateMouseAndMouseCursorPosition(false);
  window->focused->grab_mode.On(); 
  window->focused->cursor_grabbed = 1;
  CGWarpMouseCursorPosition
    (NSPointToCGPoint([[dynamic_cast<OSXWindow*>(window->focused)->view window] convertRectToScreen:
                      NSMakeRect(window->focused->gl_w / 2, window->focused->gl_h / 2, 0, 0)].origin));
}

void Clipboard::SetClipboardText(const string &s) {
  NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
  [pasteboard clearContents];
  [pasteboard declareTypes:[NSArray arrayWithObject:NSPasteboardTypeString] owner:nil];
  [pasteboard setString:[[NSString alloc] initWithUTF8String:s.c_str()] forType:NSPasteboardTypeString];
}

string Clipboard::GetClipboardText() { 
  NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
  NSString *v = [pasteboard stringForType:NSPasteboardTypeString];
  return [v UTF8String];
}

void TouchKeyboard::ToggleTouchKeyboard() {}
void TouchKeyboard::OpenTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboardAfterReturn(bool v) {}
void TouchKeyboard::SetTouchKeyboardTiled(bool v) {}

void WindowHolder::SetAppFrameEnabled(bool) {}
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

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &choose_cb) {
  static OSXFontChooser *font_chooser = [OSXFontChooser alloc];
  [font_chooser selectFont:cur_font.name.c_str() size:cur_font.size cb:choose_cb];
}

void Application::ShowSystemFileChooser(bool choose_files, bool choose_dirs, bool choose_multi, const StringVecCB &choose_cb) {
  NSOpenPanel *panel = [NSOpenPanel openPanel];
  [panel setCanChooseFiles:choose_files];
  [panel setCanChooseDirectories:choose_dirs];
  [panel setAllowsMultipleSelection:choose_multi];
  NSInteger clicked = [panel runModal];
  LoseFocus();
  if (clicked != NSModalResponseOK) return;

  StringVec arg;
  for (NSURL *url in [panel URLs]) arg.emplace_back([[url absoluteString] UTF8String]);
  if (arg.size()) choose_cb(arg);
}

void Application::ShowSystemContextMenu(const MenuItemVec &items) {
  NSEvent *event = [NSEvent mouseEventWithType: NSEventTypeLeftMouseDown
                            location:           NSMakePoint(focused->mouse.x, focused->mouse.y)
                            modifierFlags:      NSEventMaskLeftMouseDown
                            timestamp:          0
                            windowNumber:       [[dynamic_cast<OSXWindow*>(focused)->view window] windowNumber]
                            context:            [[dynamic_cast<OSXWindow*>(focused)->view window] graphicsContext]
                            eventNumber:        0
                            clickCount:         1
                            pressure:           1];

  NSMenuItem *item;
  NSMenu *menu = [[NSMenu alloc] init];
  for (auto &i : items) {
    const char *k = i.shortcut.c_str(), *n = i.name.c_str();
    if (!strcmp(n, "<separator>")) { [menu addItem:[NSMenuItem separatorItem]]; continue; }
    item = [menu addItemWithTitle: [NSString stringWithUTF8String: n]
                 action:           (i.cb ? @selector(callbackRun:) : nil)
                 keyEquivalent:    [NSString stringWithUTF8String: k]];
    if (i.cb) [item setRepresentedObject: [[ObjcCallback alloc] initWithCB: i.cb]];
  }

  [NSMenu popUpContextMenu:menu withEvent:event forView:dynamic_cast<OSXWindow*>(focused)->view];
  [dynamic_cast<OSXWindow*>(focused)->view clearKeyModifiers];
}

FrameScheduler::FrameScheduler(WindowHolder *w) :
  window(w), maxfps(&FLAGS_target_fps), rate_limit(0), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(0), run_main_loop(0) {}

bool FrameScheduler::DoMainWait(bool only_poll) { return false; }
void FrameScheduler::UpdateWindowTargetFPS(Window *w) { 
  [dynamic_cast<OSXWindow*>(w)->view stopThread];
  [dynamic_cast<OSXWindow*>(w)->view startThread:false];
}

void FrameScheduler::AddMainWaitMouse(Window *w) { dynamic_cast<OSXWindow*>(w)->view.frame_on_mouse_input = 1; }
void FrameScheduler::DelMainWaitMouse(Window *w) { dynamic_cast<OSXWindow*>(w)->view.frame_on_mouse_input = 0; }
void FrameScheduler::AddMainWaitKeyboard(Window *w) { dynamic_cast<OSXWindow*>(w)->view.frame_on_keyboard_input = 1; }
void FrameScheduler::DelMainWaitKeyboard(Window *w) { dynamic_cast<OSXWindow*>(w)->view.frame_on_keyboard_input = 0; }
void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()> cb) {
  if (fd == InvalidSocket) return;
  if (!wait_forever_thread) {
    CHECK_EQ(SocketSet::READABLE, flag);
    [dynamic_cast<OSXWindow*>(w)->view addMainWaitSocket:fd callback:move(cb)];
  }
}

void FrameScheduler::DelMainWaitSocket(Window *w, Socket fd) {
  if (fd == InvalidSocket) return;
  CHECK(w->id);
  [dynamic_cast<OSXWindow*>(w)->view delMainWaitSocket: fd];
}

unique_ptr<Framework> Framework::Create(Application *a) { return make_unique<OSXFrameworkModule>(a); }
unique_ptr<TimerInterface> SystemToolkit::CreateTimer(Callback cb) { return make_unique<AppleTimer>(move(cb)); }
unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(Window *w, AlertItemVec items) { return make_unique<OSXAlertView>(w->parent, move(items)); }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(Window*, const Box &b, const string &title, PanelItemVec items) { return make_unique<OSXPanelView>(b, title, move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(Window*, const string &title, MenuItemVec items) { return make_unique<OSXMenuView>(title, move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(Window*, MenuItemVec items) {
  NSMenuItem *item; 
  NSMenu *menu = [[NSMenu alloc] initWithTitle:@"Edit"];
  item = [menu addItemWithTitle:@"Copy"  action:@selector(copy:)  keyEquivalent:@"c"];
  item = [menu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
  AddNSMenuItems(menu, move(items));
  item = [[NSMenuItem alloc] initWithTitle:@"Edit" action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
  return make_unique<OSXMenuView>("", MenuItemVec());
}

}; // namespace LFL

extern "C" int main(int argc, const char** argv) {
  auto app = static_cast<LFL::Application*>(MyAppCreate(argc, argv));
  AppDelegate *app_delegate = [[AppDelegate alloc] initWithApp: app];
  [[NSApplication sharedApplication] setDelegate: app_delegate];
  [NSApp setMainMenu:[[NSMenu alloc] init]];
  OSXCreateSystemApplicationMenu();
  return NSApplicationMain(argc, argv);
}
