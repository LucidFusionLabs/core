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

#import <Cocoa/Cocoa.h>
#import <OpenGL/gl.h>

#include "core/app/app.h"

static int osx_argc = 0, osx_evcount = 0;
static const char **osx_argv = 0;

// GameView
@interface GameView : NSView<NSWindowDelegate>
  + (NSOpenGLPixelFormat *)defaultPixelFormat;
  - (void)update;
@end

@implementation GameView
  {
    LFApp *app;
    NativeWindow *screen;

    NSWindow *window;
    NSOpenGLContext  *context;
    NSOpenGLPixelFormat *pixel_format;
    NSFileHandle *wait_forever_fh;

    NSTimer *runloop_timer, *trigger_timer;
    NSTimeInterval trigger_timer_start;
    CVDisplayLinkRef displayLink;

    NSPoint prev_mouse_pos;
    BOOL initialized, needs_reshape, needs_frame;
    BOOL use_timer, use_display_link, video_thread_init;
    BOOL cmd_down, ctrl_down, shift_down;
    BOOL frame_on_keyboard_input, frame_on_mouse_input, should_close;
  };

  - (void)dealloc {
    if (trigger_timer) [self clearTriggerTimer];
    if (runloop_timer) ERRORf("%s", "runloop_timer remaining");
    if (wait_forever_fh) [self delWaitForeverSocket: [wait_forever_fh fileDescriptor]];
    CVDisplayLinkRelease(displayLink);
    [pixel_format release];
    [context release];
    [super dealloc];
  }
  - (id)initWithFrame:(NSRect)frame pixelFormat:(NSOpenGLPixelFormat*)format {
    [super initWithFrame:frame];
    pixel_format = [format retain];
    app = GetLFApp();
    use_timer = should_close = 1;
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
  // - (void)windowDidBecomeKey:(NSNotification *)notification {}
  // - (void)windowDidResignKey:(NSNotification *)notification {}
  // - (void)windowDidBecomeMain:(NSNotification *)notification {}
  - (void)windowDidResignMain:(NSNotification *)notification { [self clearKeyModifiers]; }
  - (void)setWindow:(NSWindow*)w { window = w; }
  - (void)setScreen:(NativeWindow*)s { screen = s; }
  - (void)setFrameSize:(NSSize)s { [super setFrameSize:s]; needs_reshape=YES; [self update]; }
  - (void)setFrameOnMouseInput:(bool)v { frame_on_mouse_input = v; }
  - (void)setFrameOnKeyboardInput:(bool)v { frame_on_keyboard_input = v; }
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
    SetNativeWindow(screen);
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
      screen->gl = LFL::MakeTyped(context);
      needs_reshape = YES;
    }
    return context;
  }
  - (NSOpenGLContext *)createGLContext {
    NSOpenGLContext *prev_context = LFL::GetTyped<NSOpenGLContext*>(GetNativeWindow()->gl);
    NSOpenGLContext *ret = [[NSOpenGLContext alloc] initWithFormat:pixel_format shareContext:prev_context];
    [ret setView:self];
    return ret;
  }
  - (void)startThread:(bool)first {
    initialized = true;
    if (!LFL::FLAGS_lfapp_input) {
      if (first) INFOf("OSXModule impl = %s", "PassThru");
      exit(LFAppMainLoop());
    } else if (screen->target_fps == 0) {
      if (first) INFOf("OSXModule impl = %s", "WaitForever");
      [self setNeedsDisplay:YES];
    } else if (use_display_link) {
      if (first) INFOf("OSXModule impl = %s", "DisplayLink");
      CVDisplayLinkStart(displayLink);
    } else if (use_timer) {
      if (first) INFOf("OSXModule impl = %s", "NSTimer");
      runloop_timer = [NSTimer timerWithTimeInterval: 1.0/screen->target_fps
          target:self selector:@selector(runloopTimerFired:) userInfo:nil repeats:YES];
      [[NSRunLoop currentRunLoop] addTimer:runloop_timer forMode:NSDefaultRunLoopMode];
      // [[NSRunLoop currentRunLoop] addTimer:runloop_timer forMode:NSEventTrackingRunLoopMode];
    }
  }
  - (void)stopThread {
    if (displayLink) { CVDisplayLinkStop(displayLink); displayLink = nil; }
    if (runloop_timer) { [runloop_timer invalidate]; runloop_timer = nil; }
  }
  - (void)stopThreadAndExit { [self stopThread]; exit(0); }
  - (void)runloopTimerFired:(id)sender { [self setNeedsDisplay:YES]; }
  - (void)triggerTimerFired:(id)sender { [self clearTriggerTimer]; needs_frame=1; [self setNeedsDisplay:YES]; }
  - (void)clearTriggerTimer { needs_frame=0; [trigger_timer invalidate]; trigger_timer = nil; }
  - (bool)triggerFrameIn:(int)ms force:(bool)force {
    if (needs_frame && !(needs_frame=0)) { [self clearTriggerTimer]; return false; }
    NSTimeInterval now = [NSDate timeIntervalSinceReferenceDate];
    if (trigger_timer) {
      int remaining = [[trigger_timer userInfo] intValue] - (now - trigger_timer_start)*1000.0;
      if (remaining <= ms && !force) return true;
      [self clearTriggerTimer]; 
    }
    trigger_timer = [NSTimer timerWithTimeInterval: ms/1000.0
      target:self selector:@selector(triggerTimerFired:) userInfo:[NSNumber numberWithInt:ms] repeats:NO];
    [[NSRunLoop currentRunLoop] addTimer:trigger_timer forMode:NSDefaultRunLoopMode];
    [trigger_timer retain];
    trigger_timer_start = now;
    return true;
  }
  - (void)drawRect:(NSRect)dirtyRect { if (use_timer) [self getFrameForTime:0]; }
  - (void)getFrameForTime:(const CVTimeStamp*)outputTime {
    static bool first_callback = 1;
    if (first_callback && !(first_callback = 0)) SetLFAppMainThread();
    if (!initialized) return;
    if (app->run) LFAppFrame(true);
    else [self stopThreadAndExit];
  }
  - (void)reshape {
    if (!initialized) return;
    [context update];
    SetNativeWindow(screen);
    float screen_w = [self frame].size.width, screen_h = [self frame].size.height;
    WindowReshaped(int(screen_w), int(screen_h));
  }
  - (BOOL)windowShouldClose:(id)sender { return should_close; }
  - (void)windowWillClose:(NSNotification *)notification { 
    SetNativeWindow(screen);
    WindowClosed();
    [self stopThread];
  }
  - (void)setWaitForeverSocket: (int)fd {
    if (wait_forever_fh) FATALf("wait_forever_fh already set: %p", wait_forever_fh);
    wait_forever_fh = [[NSFileHandle alloc] initWithFileDescriptor:fd];
    [[NSNotificationCenter defaultCenter] addObserver:self
      selector:@selector(fileDataAvailable:) name:NSFileHandleDataAvailableNotification object:wait_forever_fh];
    [wait_forever_fh waitForDataInBackgroundAndNotify];
  }
  - (void)delWaitForeverSocket: (int)fd {
    if (!wait_forever_fh || [wait_forever_fh fileDescriptor] != fd) FATALf("del mismatching wait_forever_fh %o", wait_forever_fh);
    [[NSNotificationCenter defaultCenter] removeObserver:self
      name:NSFileHandleDataAvailableNotification object:wait_forever_fh];
    // [wait_forever_fh closeFile];
    wait_forever_fh = nil;
  }
  - (void)fileDataAvailable: (NSNotification *)notification { 
    NSFileHandle *fh = static_cast<NSFileHandle*>([notification object]);
    if (fh != wait_forever_fh) return;
    SetNativeWindow(screen);
    // [self setNeedsDisplay:YES]; 
    [self getFrameForTime:nil];
    [fh waitForDataInBackgroundAndNotify];
  }
  - (void)fileReadCompleted: (NSNotification *)notification {}
  - (void)shellRun: (id)sender { ShellRun([[sender representedObject] UTF8String]); }
  - (void)mouseDown:(NSEvent*)e { [self mouseClick:e down:1]; }
  - (void)mouseUp  :(NSEvent*)e { [self mouseClick:e down:0]; }
  - (void)mouseClick:(NSEvent*)e down:(bool)d {
    SetNativeWindow(screen);
    NSPoint p = [e locationInWindow];
    int fired = MouseClick(1, d, p.x, p.y);
    if (fired && frame_on_mouse_input) [self setNeedsDisplay:YES]; 
    prev_mouse_pos = p;
  }
  - (void)mouseDragged:(NSEvent*)e { [self mouseMove:e drag:1]; }
  - (void)mouseMoved:  (NSEvent*)e { [self mouseMove:e drag:0]; }
  - (void)mouseMove: (NSEvent*)e drag:(bool)drag {
    SetNativeWindow(screen);
    if (screen->cursor_grabbed) MouseMove(prev_mouse_pos.x, prev_mouse_pos.y, [e deltaX], -[e deltaY]);
    else {
      NSPoint p=[e locationInWindow], d=NSMakePoint(p.x-prev_mouse_pos.x, p.y-prev_mouse_pos.y);
      int fired = MouseMove(p.x, p.y, d.x, d.y);
      if (fired && frame_on_mouse_input) [self setNeedsDisplay:YES]; 
      prev_mouse_pos = p;
    }
  }
  - (void)keyDown:(NSEvent *)theEvent { [self keyPress:theEvent down:1]; }
  - (void)keyUp:  (NSEvent *)theEvent { [self keyPress:theEvent down:0]; }
  - (void)keyPress:(NSEvent *)theEvent down:(bool)d {
    SetNativeWindow(screen);
    int c = getKeyCode([theEvent keyCode]);
    int fired = c ? KeyPress(c, d) : 0;
    if (fired && frame_on_keyboard_input) [self setNeedsDisplay:YES]; 
  }
  - (void)flagsChanged:(NSEvent *)theEvent {
    int flags = [theEvent modifierFlags];
    bool cmd = flags & NSCommandKeyMask, ctrl = flags & NSControlKeyMask, shift = flags & NSShiftKeyMask;
    if (cmd   != cmd_down)   { KeyPress(0x82, cmd);   cmd_down   = cmd;   }
    if (ctrl  != ctrl_down)  { KeyPress(0x86, ctrl);  ctrl_down  = ctrl;  }
    if (shift != shift_down) { KeyPress(0x83, shift); shift_down = shift; }
  }
  - (void)clearKeyModifiers {
    if (cmd_down)   { KeyPress(0x82, 0); cmd_down   = 0; }
    if (ctrl_down)  { KeyPress(0x86, 0); ctrl_down  = 0; }
    if (shift_down) { KeyPress(0x83, 0); shift_down = 0; }
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

// AppDelegate
@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate
  {
    NSFont *font;
    NSString *font_change_cmd;
  }
  - (void)applicationWillTerminate: (NSNotification *)aNotification {}
  - (void)applicationDidFinishLaunching: (NSNotification *)aNotification {
    INFOf("OSXModule::Main argc=%d\n", osx_argc);
    // [[NSFileManager defaultManager] changeCurrentDirectoryPath: [[NSBundle mainBundle] resourcePath]];
    int ret = MyLFAppMain(osx_argc, osx_argv);
    if (ret) exit(ret);
    INFOf("%s", "OSXModule::Main done");
  }
  - (GameView*)createWindow: (int)w height:(int)h nativeWindow:(NativeWindow*)s {
    NSWindow *window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, w, h)
                                         styleMask:NSClosableWindowMask|NSMiniaturizableWindowMask|NSResizableWindowMask|NSTitledWindowMask
                                         backing:NSBackingStoreBuffered defer:NO];
    GameView *view = [[[GameView alloc] initWithFrame:window.frame pixelFormat:GameView.defaultPixelFormat] autorelease];
    [view setScreen:s];
    [view setWindow:window];
    [[view openGLContext] setView:view];
    [window setContentView:view];
    [window setDelegate:view];
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
    GetLFApp()->run = false;
    return YES;
  }
  - (void)selectFont: (const char *)name size:(int)s cmd:(const char*)v {
    font = [NSFont fontWithName:[NSString stringWithUTF8String:name] size:s];
    font_change_cmd = [NSString stringWithUTF8String:v];
    NSFontManager *fontManager = [NSFontManager sharedFontManager];
    [fontManager setSelectedFont:font isMultiple:NO];
    [fontManager setDelegate:self];
    [fontManager setTarget:self];
    [fontManager orderFrontFontPanel:self];
  }
  - (void)changeFont:(id)sender {
    font = [sender convertFont:font];
    float size = [[[font fontDescriptor] objectForKey:NSFontSizeAttribute] floatValue];
    ShellRun([[NSString stringWithFormat:@"%@ %@ %f", font_change_cmd, [font fontName], size] UTF8String]);
  }
@end

extern "C" void NativeWindowInit() {}
extern "C" void NativeWindowQuit() {}
extern "C" int NativeWindowOrientation() { return 1; }
extern "C" void NativeWindowSize(int *width, int *height) {}

extern "C" typed_ptr OSXCreateWindow(int w, int h, struct NativeWindow *nw) {
  [LFL::GetTyped<GameView*>(GetNativeWindow()->id) clearKeyModifiers];
  return LFL::MakeTyped([static_cast<AppDelegate*>([NSApp delegate]) createWindow:w height:h nativeWindow:nw]);
}

extern "C" void OSXDestroyWindow(typed_ptr O) {
  [static_cast<AppDelegate*>([NSApp delegate]) destroyWindow: [LFL::GetTyped<GameView*>(O) window]];
}

extern "C" void OSXSetWindowResizeIncrements(typed_ptr O, float x, float y) {
  [[LFL::GetTyped<GameView*>(O) window] setContentResizeIncrements: NSMakeSize(x, y)];
}

extern "C" void OSXSetWindowTransparency(typed_ptr O, float v) {
  [[LFL::GetTyped<GameView*>(O) window] setAlphaValue: 1.0 - v];
}

extern "C" typed_ptr OSXCreateGLContext(typed_ptr O) {
  return LFL::MakeTyped([LFL::GetTyped<GameView*>(O) createGLContext]);
}

extern "C" void OSXDeleteGLContext(typed_ptr O) {
}

extern "C" void OSXStartWindow(typed_ptr O) {
  [LFL::GetTyped<GameView*>(O) startThread:true];
}

extern "C" void OSXMakeWindowCurrent(typed_ptr O) {
  [[LFL::GetTyped<GameView*>(O) openGLContext] makeCurrentContext];
}

extern "C" void OSXSetWindowSize(typed_ptr O, int W, int H) {
  [[LFL::GetTyped<GameView*>(O) window] setContentSize:NSMakeSize(W, H)];
}

extern "C" void OSXSetWindowTitle(typed_ptr O, const char *v) {
  [LFL::GetTyped<GameView*>(O) window].title = [NSString stringWithUTF8String:v];
}

extern "C" void OSXVideoSwap(typed_ptr O) {
  // [[LFL::GetTyped<GameView*>(O) openGLContext] flushBuffer];
  CGLFlushDrawable([[LFL::GetTyped<GameView*>(O) openGLContext] CGLContextObj]);
}

extern "C" void OSXGrabMouseFocus() { 
  CGDisplayHideCursor(kCGDirectMainDisplay);
  CGAssociateMouseAndMouseCursorPosition(false);
}

extern "C" void OSXReleaseMouseFocus() {
  CGDisplayShowCursor(kCGDirectMainDisplay);
  CGAssociateMouseAndMouseCursorPosition(true);
}

extern "C" void OSXSetMousePosition(typed_ptr O, int x, int y) {
  CGWarpMouseCursorPosition
    (NSPointToCGPoint([[LFL::GetTyped<GameView*>(O) window] convertRectToScreen:NSMakeRect(x, y, 0, 0)].origin));
}

extern "C" void OSXClipboardSet(const char *v) {
  NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
  [pasteboard clearContents];
  [pasteboard declareTypes:[NSArray arrayWithObject:NSPasteboardTypeString] owner:nil];
  [pasteboard setString:[[NSString alloc] initWithUTF8String:v] forType:NSPasteboardTypeString];
}

extern "C" char *OSXClipboardGet() {
  NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
  NSString *v = [pasteboard stringForType:NSPasteboardTypeString];
  return strdup([v UTF8String]);
}

extern "C" void OSXTriggerFrame(typed_ptr O) {
  [LFL::GetTyped<GameView*>(O) performSelectorOnMainThread:@selector(setNeedsDisplay:) withObject:@YES waitUntilDone:NO];
}

extern "C" bool OSXTriggerFrameIn(typed_ptr O, int ms, bool force) {
  return [LFL::GetTyped<GameView*>(O) triggerFrameIn:ms force:force];
}

extern "C" void OSXClearTriggerFrameIn(typed_ptr O) {
  [LFL::GetTyped<GameView*>(O) clearTriggerTimer];
}

extern "C" void OSXUpdateTargetFPS(typed_ptr O) {
  [LFL::GetTyped<GameView*>(O) stopThread];
  [LFL::GetTyped<GameView*>(O) startThread:false];
}

extern "C" void OSXAddWaitForeverMouse(typed_ptr O) { [LFL::GetTyped<GameView*>(O) setFrameOnMouseInput:1]; }
extern "C" void OSXDelWaitForeverMouse(typed_ptr O) { [LFL::GetTyped<GameView*>(O) setFrameOnMouseInput:0]; }
extern "C" void OSXAddWaitForeverKeyboard(typed_ptr O) { [LFL::GetTyped<GameView*>(O) setFrameOnKeyboardInput:1]; }
extern "C" void OSXDelWaitForeverKeyboard(typed_ptr O) { [LFL::GetTyped<GameView*>(O) setFrameOnKeyboardInput:0]; }
extern "C" void OSXAddWaitForeverSocket(typed_ptr O, int fd) { [LFL::GetTyped<GameView*>(O) setWaitForeverSocket: fd]; }
extern "C" void OSXDelWaitForeverSocket(typed_ptr O, int fd) { [LFL::GetTyped<GameView*>(O) delWaitForeverSocket: fd]; }

extern "C" void OSXCreateNativeApplicationMenu() {
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
  [item setKeyEquivalentModifierMask:(NSAlternateKeyMask|NSCommandKeyMask)];
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

extern "C" void OSXCreateNativeEditMenu() {
  NSMenuItem *item; 
  NSMenu *menu = [[NSMenu alloc] initWithTitle:@"Edit"];
  item = [menu addItemWithTitle:@"Copy"  action:@selector(copy:)  keyEquivalent:@"c"];
  item = [menu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
  item = [[NSMenuItem alloc] initWithTitle:@"Edit" action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
}

extern "C" void OSXCreateNativeMenu(const char *title_text, int n, const char **key, const char **name, const char **val) {
  NSMenuItem *item;
  NSString *title = [NSString stringWithUTF8String: title_text];
  NSMenu *menu = [[NSMenu alloc] initWithTitle: title];
  for (int i=0; i<n; i++) {
    if (!strcmp(name[i], "<seperator>")) { [menu addItem:[NSMenuItem separatorItem]]; continue; }
    item = [menu addItemWithTitle: [NSString stringWithUTF8String: name[i]]
                 action:           (val[i][0] ? @selector(shellRun:) : nil)
                 keyEquivalent:    [NSString stringWithUTF8String: key[i]]];
    [item setRepresentedObject: [NSString stringWithUTF8String: val[i]]];
  }
  item = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
}

extern "C" void OSXLaunchNativeContextMenu(typed_ptr O, int x, int y, int n, const char **key, const char **name, const char **val) {
  NSEvent *event = [NSEvent mouseEventWithType: NSLeftMouseDown
                            location:           NSMakePoint(x, y)
                            modifierFlags:      NSLeftMouseDownMask
                            timestamp:          0
                            windowNumber:       [[LFL::GetTyped<GameView*>(O) window] windowNumber]
                            context:            [[LFL::GetTyped<GameView*>(O) window] graphicsContext]
                            eventNumber:        0
                            clickCount:         1
                            pressure:           1];

  NSMenuItem *item;
  NSMenu *menu = [[NSMenu alloc] init];
  for (int i=0; i<n; i++) {
    if (!strcmp(name[i], "<seperator>")) { [menu addItem:[NSMenuItem separatorItem]]; continue; }
    item = [menu addItemWithTitle: [NSString stringWithUTF8String: name[i]]
                 action:           (val[i][0] ? @selector(shellRun:) : nil)
                 keyEquivalent:    [NSString stringWithUTF8String: key[i]]];
    [item setRepresentedObject: [NSString stringWithUTF8String: val[i]]];
  }

  [NSMenu popUpContextMenu:menu withEvent:event forView:LFL::GetTyped<GameView*>(O)];
  [LFL::GetTyped<GameView*>(O) clearKeyModifiers];
}

extern "C" void OSXLaunchNativeFontChooser(const char *cur_font, int size, const char *change_font_cmd) {
  [static_cast<AppDelegate*>([NSApp delegate]) selectFont:cur_font size:size cmd:change_font_cmd];
}

extern "C" void OSXLaunchNativeFileChooser(bool choose_files, bool choose_dirs, bool choose_multi, const char *open_files_cmd) {
  NSOpenPanel *panel = [NSOpenPanel openPanel];
  [panel setCanChooseFiles:choose_files];
  [panel setCanChooseDirectories:choose_dirs];
  [panel setAllowsMultipleSelection:choose_multi];
  NSInteger clicked = [panel runModal];
  [LFL::GetTyped<GameView*>(GetNativeWindow()->id) clearKeyModifiers];
  if (clicked != NSFileHandlingPanelOKButton) return;

  std::string start = open_files_cmd, run = start;
  for (NSURL *url in [panel URLs]) {
    run.append(" ");
    run.append([[url absoluteString] UTF8String]);
  }
  if (run.size() > start.size()) ShellRun(run.c_str());
}

namespace LFL {
extern "C" typed_ptr OSXCreateWindow(int W, int H, struct NativeWindow *nw);
extern "C" void OSXDestroyWindow(typed_ptr O);
extern "C" void OSXMakeWindowCurrent(typed_ptr O);
extern "C" void OSXSetWindowSize(typed_ptr, int W, int H);
extern "C" void OSXSetWindowTitle(typed_ptr, const char *v);
extern "C" void OSXSetWindowResizeIncrements(typed_ptr, float x, float y);
extern "C" void OSXSetWindowTransparency(typed_ptr, float v);
extern "C" typed_ptr OSXCreateGLContext(typed_ptr);
extern "C" void OSXVideoSwap(typed_ptr);

extern "C" void OSXGrabMouseFocus();
extern "C" void OSXReleaseMouseFocus();
extern "C" void OSXSetMousePosition(typed_ptr, int x, int y);
extern "C" void OSXClipboardSet(const char *v);
extern "C" char *OSXClipboardGet();

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

struct OSXVideoModule : public Module {
  int Init() {
    INFO("OSXVideoModule::Init()");
    NativeWindowInit();
    NativeWindowSize(&screen->width, &screen->height);
    CHECK(app->CreateWindow(screen));
    return 0;
  }
};

bool Application::CreateWindow(Window *W) { 
  W->id = OSXCreateWindow(W->width, W->height, W);
  if (W->id.value) windows[W->id.value] = W;
  OSXSetWindowTitle(W->id, W->caption.c_str());
  return true; 
}
void Application::MakeCurrentWindow(Window *W) { 
  if (W) OSXMakeWindowCurrent((screen = W)->id);
}
void Application::CloseWindow(Window *W) {
  windows.erase(W->id.value);
  if (windows.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  // OSXDestroyWindow(W->id);
  screen = 0;
}

void Window::SetCaption(const string &v) { OSXSetWindowTitle(id, v.c_str()); }
void Window::SetResizeIncrements(float x, float y) { OSXSetWindowResizeIncrements(id, x, y); }
void Window::SetTransparency(float v) { OSXSetWindowTransparency(id, v); }
void Window::Reshape(int w, int h) { OSXSetWindowSize(id, w, h); }

void Application::SetClipboardText(const string &s) { OSXClipboardSet(s.c_str()); }
string Application::GetClipboardText() { char *v = OSXClipboardGet(); string ret = v; free(reinterpret_cast<void*>(v)); return ret; }
void Application::ReleaseMouseFocus() { OSXReleaseMouseFocus(); app->grab_mode.Off(); screen->cursor_grabbed = 0; }
void Application::GrabMouseFocus()    { OSXGrabMouseFocus();    app->grab_mode.On();  screen->cursor_grabbed = 1;
  OSXSetMousePosition(screen->id, screen->width / 2, screen->height / 2);
}

void Video::StartWindow() { OSXStartWindow(screen->id); }
void *Video::BeginGLContextCreate(Window *W) { return 0; }
void *Video::CompleteGLContextCreate(Window *W, void *gl_context) {
  return OSXCreateGLContext(W->id).value;
}
int Video::Swap() {
  screen->gd->Flush();
  OSXVideoSwap(screen->id);
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void FrameScheduler::DoWait() {}
void FrameScheduler::Setup() { rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = run_main_loop = 0; }
void FrameScheduler::Wakeup(void *opaque) { if (wait_forever && screen) OSXTriggerFrame(screen->id); }
bool FrameScheduler::WakeupIn(void *opaque, Time interval, bool force) { return OSXTriggerFrameIn(screen->id, interval.count(), force); }
void FrameScheduler::ClearWakeupIn() { OSXClearTriggerFrameIn(screen->id); }
void FrameScheduler::AddWaitForeverMouse() { OSXAddWaitForeverMouse(screen->id); }
void FrameScheduler::DelWaitForeverMouse() { OSXDelWaitForeverMouse(screen->id); }
void FrameScheduler::AddWaitForeverKeyboard() { OSXAddWaitForeverKeyboard(screen->id); }
void FrameScheduler::DelWaitForeverKeyboard() { OSXDelWaitForeverKeyboard(screen->id); }
void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, val);
  if (!wait_forever_thread) { CHECK_EQ(SocketSet::READABLE, flag); OSXAddWaitForeverSocket(screen->id, fd); }
}
void FrameScheduler::DelWaitForeverSocket(Socket fd) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
  CHECK(screen->id.value);
  OSXDelWaitForeverSocket(screen->id, fd);
}

}; // namespace LFL

extern "C" void *LFAppCreatePlatformModule() { return new LFL::OSXVideoModule(); }

#undef main
extern "C" int main(int argc, const char **argv) {
  osx_argc = argc;
  osx_argv = argv;
  AppDelegate *app_delegate = [[AppDelegate alloc] init];
  [[NSApplication sharedApplication] setDelegate: app_delegate];
  [NSApp setMainMenu:[[NSMenu alloc] init]];
  OSXCreateNativeApplicationMenu();
  return NSApplicationMain(argc, argv);
}
