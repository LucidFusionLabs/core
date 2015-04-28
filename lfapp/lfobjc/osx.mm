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

#include <stdlib.h>
#include "../lfexport.h"

#ifdef LFL_OSXVIDEO
#import <Cocoa/Cocoa.h>
#import <OpenGL/gl.h>

extern "C" int OSXMain(int argc, const char **argv);

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

        NSTimer *timer;
        CVDisplayLinkRef displayLink;

        NSPoint prev_mouse_pos;
        BOOL initialized, needs_reshape;
        BOOL use_timer, use_display_link, video_thread_init;
        BOOL cmd_down, ctrl_down, shift_down;
        BOOL frame_on_keyboard_input, frame_on_mouse_input, should_close;
    };

    - (void)dealloc {
        if (timer)           FATALf("%s", "timer");
        if (wait_forever_fh) FATALf("%s", wait_forever_fh);
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
        NSOpenGLPixelFormatAttribute attributes[] = { 0 };
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
    - (void)setFrame:(NSRect)frame { [super setFrame:frame]; needs_reshape=YES; [self update]; }
    - (void)setFrameOnMouseInput:(bool)v { frame_on_mouse_input = v; }
    - (void)setFrameOnKeyboardInput:(bool)v { frame_on_keyboard_input = v; }
    - (void)update { [context update]; }
    - (void)lockFocus {
        [super lockFocus];
        CGLLockContext([context CGLContextObj]);
        [context setView:self];
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
            screen->gl = context;
            needs_reshape = YES;
        }
        return context;
    }
    - (NSOpenGLContext *)createGLContext {
        NSOpenGLContext *prev_context = (NSOpenGLContext*)GetNativeWindow()->gl;
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
            timer = [NSTimer timerWithTimeInterval: 1.0/screen->target_fps
                target:self selector:@selector(timerFired:) userInfo:nil repeats:YES];
            [[NSRunLoop currentRunLoop] addTimer:timer forMode:NSDefaultRunLoopMode];
            // [[NSRunLoop currentRunLoop] addTimer:timer forMode:NSEventTrackingRunLoopMode];
        }
    }
    - (void)viewDidChangeBackingProperties {
        float ms = [[NSScreen mainScreen] backingScaleFactor];
        self.layer.contentsScale = [[self window] backingScaleFactor];
        printf("view did change backing %f main window %f\n", self.layer.contentsScale, ms);
    }
    - (void)stopThread {
        if (displayLink) { CVDisplayLinkStop(displayLink); displayLink = nil; }
        if (timer) { [timer invalidate]; timer = nil; }
    }
    - (void)stopThreadAndExit { [self stopThread]; exit(0); }
    - (void)timerFired:(id)sender { [self setNeedsDisplay:YES]; }
    - (void)drawRect:(NSRect)dirtyRect { if (use_timer) [self getFrameForTime:0]; }
    - (void)getFrameForTime:(const CVTimeStamp*)outputTime {
        static bool first_callback = 1;
        if (first_callback && !(first_callback = 0)) SetLFAppMainThread();
        if (!initialized) return;
        if (app->run) LFAppFrame();
        else [self stopThreadAndExit];
    }
    - (void)reshape {
        if (!initialized) return;
        [context update];
        SetNativeWindow(screen);
        float screen_w = [self frame].size.width, screen_h = [self frame].size.height;
        WindowReshaped((int)screen_w, (int)screen_h);
    }
    - (BOOL)windowShouldClose:(id)sender { return should_close; }
    - (void)windowWillClose:(NSNotification *)notification { 
        SetNativeWindow(screen);
        WindowClosed();
        [self stopThread];
        [self autorelease];
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
        NSFileHandle *fh = (NSFileHandle*) [notification object];
        if (fh != wait_forever_fh) return;
        SetNativeWindow(screen);
        // [self setNeedsDisplay:YES]; 
        [self getFrameForTime:nil];
        [fh waitForDataInBackgroundAndNotify];
    }
    - (void)fileReadCompleted: (NSNotification *)notification {}
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
            'a',  /* 0x00: kVK_ANSI_A */               's',  /* 0x01: kVK_ANSI_S */                 'd',  /* 0x02: kVK_ANSI_D */
            'f',  /* 0x03: kVK_ANSI_F */               'h',  /* 0x04: kVK_ANSI_H */                 'g',  /* 0x05: kVK_ANSI_G */
            'z',  /* 0x06: kVK_ANSI_Z */               'x',  /* 0x07: kVK_ANSI_X */                 'c',  /* 0x08: kVK_ANSI_C */
            'v',  /* 0x09: kVK_ANSI_V */               0,    /* 0x0A: */                            'b',  /* 0x0B: kVK_ANSI_B */
            'q',  /* 0x0C: kVK_ANSI_Q */               'w',  /* 0x0D: kVK_ANSI_W */                 'e',  /* 0x0E: kVK_ANSI_E */
            'r',  /* 0x0F: kVK_ANSI_R */               'y',  /* 0x10: kVK_ANSI_Y */                 't',  /* 0x11: kVK_ANSI_T */
            '1',  /* 0x12: kVK_ANSI_1 */               '2',  /* 0x13: kVK_ANSI_2 */                 '3',  /* 0x14: kVK_ANSI_3 */
            '4',  /* 0x15: kVK_ANSI_4 */               '6',  /* 0x16: kVK_ANSI_6 */                 '5',  /* 0x17: kVK_ANSI_5 */
            '=',  /* 0x18: kVK_ANSI_Equal */           '9',  /* 0x19: kVK_ANSI_9 */                 '7',  /* 0x1A: kVK_ANSI_7 */
            '-',  /* 0x1B: kVK_ANSI_Minus */           '8',  /* 0x1C: kVK_ANSI_8 */                 '0',  /* 0x1D: kVK_ANSI_0 */
            ']',  /* 0x1E: kVK_ANSI_RightBracket */    'o',  /* 0x1F: kVK_ANSI_O */                 'u',  /* 0x20: kVK_ANSI_U */
            '[',  /* 0x21: kVK_ANSI_LeftBracket */     'i',  /* 0x22: kVK_ANSI_I */                 'p',  /* 0x23: kVK_ANSI_P */
            '\r', /* 0x24: kVK_Return */               'l',  /* 0x25: kVK_ANSI_L */                 'j',  /* 0x26: kVK_ANSI_J */
            '\'', /* 0x27: kVK_ANSI_Quote */           'k',  /* 0x28: kVK_ANSI_K */                 ';',  /* 0x29: kVK_ANSI_Semicolon */
            '\\', /* 0x2A: kVK_ANSI_Backslash */       ',',  /* 0x2B: kVK_ANSI_Comma */             '/',  /* 0x2C: kVK_ANSI_Slash */ 
            'n',  /* 0x2D: kVK_ANSI_N */               'm',  /* 0x2E: kVK_ANSI_M */                 '.',  /* 0x2F: kVK_ANSI_Period */ 
            '\t', /* 0x30: kVK_Tab */                  ' ',  /* 0x31: kVK_Space */                  '`',  /* 0x32: kVK_ANSI_Grave */             
            0x80, /* 0x33: kVK_Delete */               0,    /* 0x34: */                            0x81, /* 0x35: kVK_Escape */ 
            0,    /* 0x36: */                          0x82, /* 0x37: kVK_Command */                0x83, /* 0x38: kVK_Shift */ 
            0x84, /* 0x39: kVK_CapsLock */             0x85, /* 0x3A: kVK_Option */                 0x86, /* 0x3B: kVK_Control */ 
            0x87, /* 0x3C: kVK_RightShift */           0x88, /* 0x3D: kVK_RightOption */            0x89, /* 0x3E: kVK_RightControl */ 
            0x8A, /* 0x3F: kVK_Function */             0x8B, /* 0x40: kVK_F17 */                    0x8C, /* 0x41: kVK_ANSI_KeypadDecimal */
            0,    /* 0x42: */                          0x8D, /* 0x43: kVK_ANSI_KeypadMultiply */    0,    /* 0x44: */
            0x8E, /* 0x45: kVK_ANSI_KeypadPlus */      0x8F, /* 0x46: */                            0x90, /* 0x47: kVK_ANSI_KeypadClear */ 
            0x91, /* 0x48: kVK_VolumeUp */             0x92, /* 0x49: kVK_VolumeDown */             0x93, /* 0x4A: kVK_Mute */
            0x94, /* 0x4B: kVK_ANSI_KeypadDivide */    0x95, /* 0x4C: kVK_ANSI_KeypadEnter */       0,    /* 0x4D: */
            0x96, /* 0x4E: kVK_ANSI_KeypadMinus */     0x97, /* 0x4F: kVK_F18 */                    0x98, /* 0x50: kVK_F19 */
            0x99, /* 0x51: kVK_ANSI_KeypadEquals */    0x9A, /* 0x52: kVK_ANSI_Keypad0 */           0x9B, /* 0x53: kVK_ANSI_Keypad1 */ 
            0x9C, /* 0x54: kVK_ANSI_Keypad2 */         0x9D, /* 0x55: kVK_ANSI_Keypad3 */           0x9E, /* 0x56: kVK_ANSI_Keypad4 */ 
            0x9F, /* 0x57: kVK_ANSI_Keypad5 */         0xA0, /* 0x58: kVK_ANSI_Keypad6 */           0xA1, /* 0x59: kVK_ANSI_Keypad7 */            
            0xA2, /* 0x5A: kVK_F20 */                  0xA3, /* 0x5B: kVK_ANSI_Keypad8 */           0xA4, /* 0x5C: kVK_ANSI_Keypad9 */ 
            0,    /* 0x5D: */                          0,    /* 0x5E: */                            0,    /* 0x5F: */
            0xA5, /* 0x60: kVK_F5 */                   0xA6, /* 0x61: kVK_F6 */                     0xA7, /* 0x62: kVK_F7 */ 
            0xA8, /* 0x63: kVK_F3 */                   0xA9, /* 0x64: kVK_F8 */                     0xAA, /* 0x65: kVK_F9 */ 
            0,    /* 0x66: */                          0xAB, /* 0x67: kVK_F11 */                    0,    /* 0x68: */
            0xAC, /* 0x69: kVK_F13 */                  0xAD, /* 0x6A: kVK_F16 */                    0xAE, /* 0x6B: kVK_F14 */
            0,    /* 0x6C: */                          0xAF, /* 0x6D: kVK_F10 */                    0,    /* 0x6E: */
            0xB0, /* 0x6F: kVK_F12 */                  0,    /* 0x70: */                            0xB1, /* 0x71: kVK_F15 */
            0xB2, /* 0x72: kVK_Help */                 0xB3, /* 0x73: kVK_Home */                   0xB4, /* 0x74: kVK_PageUp */
            0xB5, /* 0x75: kVK_ForwardDelete */        0xB6, /* 0x76: kVK_F4 */                     0xB7, /* 0x77: kVK_End */
            0xB8, /* 0x78: kVK_F2 */                   0xB9, /* 0x79: kVK_PageDown */               0xBA, /* 0x7A: kVK_F1 */
            0xBB, /* 0x7B: kVK_LeftArrow */            0xBC, /* 0x7C: kVK_RightArrow */             0xBD, /* 0x7D: kVK_DownArrow */
            0xBE, /* 0x7E: kVK_UpArrow */              0     /* 0x7f: */                            
        };
        return (c >= 0 && c < 128) ? keyCode[c] : 0;
    }
    static CVReturn MyDisplayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp* now, const CVTimeStamp* outputTime, CVOptionFlags flagsIn, CVOptionFlags* flagsOut, void* displayLinkContext) {
        NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
        [(GameView*)displayLinkContext getFrameForTime:outputTime];
        [pool release];
        return kCVReturnSuccess;
    }
@end

// AppDelegate
@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate
    - (void)applicationWillTerminate: (NSNotification *)aNotification {}
    - (void)applicationDidFinishLaunching: (NSNotification *)aNotification {
        INFOf("OSXModule::Main argc=%d\n", osx_argc);
        // [[NSFileManager defaultManager] changeCurrentDirectoryPath: [[NSBundle mainBundle] resourcePath]];
        int ret = OSXMain(osx_argc, osx_argv);
        if (ret) exit(ret);
        INFOf("%s", "OSXModule::Main done");
        [(GameView*)GetNativeWindow()->id startThread:true];
    }
    - (void*)createWindow: (int)w height:(int)h nativeWindow:(NativeWindow*)s {
        NSWindow *window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, w, h)
                                             styleMask:NSClosableWindowMask|NSMiniaturizableWindowMask|NSResizableWindowMask|NSTitledWindowMask
                                             backing:NSBackingStoreBuffered defer:NO];
        GameView *view = [[GameView alloc] initWithFrame:window.frame pixelFormat:GameView.defaultPixelFormat];
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
@end

extern "C" void NativeWindowInit() {}
extern "C" void NativeWindowQuit() {}
extern "C" int NativeWindowOrientation() { return 1; }
extern "C" void NativeWindowSize(int *width, int *height) {}

extern "C" void *OSXCreateWindow(int w, int h, struct NativeWindow *nw) {
    [(GameView*)GetNativeWindow()->id clearKeyModifiers];
    return [(AppDelegate*)[NSApp delegate] createWindow:w height:h nativeWindow:nw];
}
extern "C" void OSXDestroyWindow(void *O) {
    [(AppDelegate*)[NSApp delegate] destroyWindow: [(GameView*)O window] ];
}

extern "C" void *OSXCreateGLContext(void *O) {
    return [(GameView*)O createGLContext];
}
extern "C" void OSXDeleteGLContext(void *O) {
}

extern "C" void OSXStartWindow(void *O) {
    [(GameView*)O startThread:true];
}
extern "C" void OSXMakeWindowCurrent(void *O) {
    [[(GameView*)O openGLContext] makeCurrentContext];
}
extern "C" void OSXSetWindowSize(void *O, int W, int H) {
    [[(GameView*)O window] setContentSize:NSMakeSize(W, H)];
}
extern "C" void OSXVideoSwap(void *O) {
    // [[(GameView*)O openGLContext] flushBuffer];
    CGLFlushDrawable([[(GameView*)O openGLContext] CGLContextObj]);
}

extern "C" void OSXGrabMouseFocus() { 
    CGDisplayHideCursor(kCGDirectMainDisplay);
    CGAssociateMouseAndMouseCursorPosition(false);
}
extern "C" void OSXReleaseMouseFocus() {
    CGDisplayShowCursor(kCGDirectMainDisplay);
    CGAssociateMouseAndMouseCursorPosition(true);
}
extern "C" void OSXSetMousePosition(void *O, int x, int y) {
    CGWarpMouseCursorPosition
        (NSPointToCGPoint([[(GameView*)O window] convertRectToScreen:NSMakeRect(x, y, 0, 0)].origin));
}

extern "C" void OSXClipboardSet(const char *v) {
    NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
    [pasteboard clearContents];
    [pasteboard declareTypes:[NSArray arrayWithObject:NSPasteboardTypeString] owner:nil];
    [pasteboard setString:[[NSString alloc] initWithUTF8String:v] forType:NSPasteboardTypeString];
}
extern "C" const char *OSXClipboardGet() {
    NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
    NSString *v = [pasteboard stringForType:NSPasteboardTypeString];
    return strdup([v UTF8String]);
}

extern "C" void OSXTriggerFrame(void *O) {
    [(GameView*)O performSelectorOnMainThread:@selector(setNeedsDisplay:) withObject:@YES waitUntilDone:NO];
}

extern "C" void OSXUpdateTargetFPS(void *O) {
    [(GameView*)O stopThread];
    [(GameView*)O startThread:false];
}

extern "C" void OSXAddWaitForeverMouse(void *O) { [(GameView*)O setFrameOnMouseInput:1]; }
extern "C" void OSXDelWaitForeverMouse(void *O) { [(GameView*)O setFrameOnMouseInput:0]; }
extern "C" void OSXAddWaitForeverKeyboard(void *O) { [(GameView*)O setFrameOnKeyboardInput:1]; }
extern "C" void OSXDelWaitForeverKeyboard(void *O) { [(GameView*)O setFrameOnKeyboardInput:0]; }
extern "C" void OSXAddWaitForeverSocket(void *O, int fd) { [(GameView*)O setWaitForeverSocket: fd]; }
extern "C" void OSXDelWaitForeverSocket(void *O, int fd) { [(GameView*)O delWaitForeverSocket: fd]; }

extern "C" void OSXCreateApplicationMenus() {
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

    menu = [[NSMenu alloc] initWithTitle:@"Edit"];
    item = [menu addItemWithTitle:@"Copy"  action:@selector(copy:)  keyEquivalent:@"c"];
    item = [menu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
    item = [[NSMenuItem alloc] initWithTitle:@"Edit" action:nil keyEquivalent:@""];
    [item setSubmenu: menu];
    [[NSApp mainMenu] addItem: item];
    [menu release];
    [item release];

    menu = [[NSMenu alloc] initWithTitle:@"View"];
    item = [menu addItemWithTitle:@"Zoom In"  action:nil keyEquivalent:@"+"];
    item = [menu addItemWithTitle:@"Zoom Out" action:nil keyEquivalent:@"-"];
    item = [[NSMenuItem alloc] initWithTitle:@"View" action:nil keyEquivalent:@""];
    [item setSubmenu: menu];
    [[NSApp mainMenu] addItem: item];
    [menu release];
    [item release];
}

extern "C" int main(int argc, const char **argv) {
    osx_argc = argc; osx_argv = argv;
    AppDelegate *app_delegate = [[AppDelegate alloc] init];
    [[NSApplication sharedApplication] setDelegate: app_delegate];
    [NSApp setMainMenu:[[NSMenu alloc] init]];
    OSXCreateApplicationMenus();
    return NSApplicationMain(argc, argv);
}
#endif // LFL_OSXVIDEO
