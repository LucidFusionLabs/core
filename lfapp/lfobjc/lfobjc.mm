/*
 * $Id: lfobjc.mm 1299 2011-12-11 21:39:12Z justin $
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

#include <stdio.h>
#include <Foundation/NSThread.h>

#include "../lfexport.h"

@interface MyThread : NSObject {}
@property int (*CB)(void*);
@property void *arg;
@end
@implementation MyThread
- (void) myMain { _CB(_arg); }
@end

void NativeThreadStart(int (*CB)(void *), void *arg) {
    MyThread *thread = [MyThread alloc];
    [thread setCB:  CB];
    [thread setArg: arg];
    [NSThread detachNewThreadSelector:@selector(myMain) toTarget:thread withObject:nil];
}

#ifndef LFL_IPHONE
#import <AppKit/NSFont.h>
#import <AppKit/NSFontManager.h>

extern "C" int OSXReadFonts(void *FontsIter, void (*FontsIterAdd)(void *fi, const char *name, const char *family, int flag)) {
    NSFontManager *nsFontManager = [NSFontManager sharedFontManager];
    [nsFontManager retain];

    NSArray *fontArray = [nsFontManager availableFonts];
    NSUInteger totalFonts = [fontArray count];
    for (int i = 0; i < totalFonts; i++) {
        NSString *str = [fontArray objectAtIndex:i];
        FontsIterAdd(FontsIter, [str UTF8String], "", 0);
    }

    [nsFontManager release];
    return 0;
}
#endif // LFL_IPHONE

#ifdef LFL_OSXVIDEO
#import <Cocoa/Cocoa.h>
#import <OpenGL/gl.h>

extern "C" int OSXMain(int argc, const char **argv);

static int osx_argc = 0, osx_evcount = 0;
static const char **osx_argv = 0;

// GameView
@interface GameView : NSOpenGLView<NSWindowDelegate>
@end

@implementation GameView
    LFApp *app = 0;
    NSWindow *window = 0;
    NativeWindow *screen = 0;
    NSTimer *timer = 0;
    CVDisplayLinkRef displayLink;
    NSPoint prev_mouse_pos;
    bool use_timer = 1, use_display_link = 0, video_thread_init = 0;
    bool cmd_down = 0, ctrl_down = 0, shift_down = 0;
    bool frame_on_keyboard_input = 0, frame_on_mouse_input = 0;

    - (BOOL)acceptsFirstResponder { return YES; }
    - (void)setWindow:(NSWindow*)w { window = w; }
    - (void)setFrameOnMouseInput:(bool)v { frame_on_mouse_input = v; }
    - (void)setFrameOnKeyboardInput:(bool)v { frame_on_keyboard_input = v; }
    - (void)dealloc {
        CVDisplayLinkRelease(displayLink);
        [super dealloc];
    }
    - (void)prepareOpenGL {
        GLint swapInt = 1, opaque = 0;
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
    - (void)startThread {
        if (!LFL::FLAGS_lfapp_input) {
            INFOf("OSXModule impl = %s", "PassThru");
            exit(LFAppMainLoop());
        } else if (LFL::FLAGS_target_fps == 0) {
            INFOf("OSXModule impl = %s", "WaitForever");
            [self setNeedsDisplay:YES];
        } else if (use_display_link) {
            INFOf("OSXModule impl = %s", "DisplayLink");
            CVDisplayLinkStart(displayLink);
        } else if (use_timer) {
            INFOf("OSXModule impl = %s", "NSTimer");
            timer = [NSTimer timerWithTimeInterval: 1.0/LFL::FLAGS_target_fps
                target:self selector:@selector(timerFired:) userInfo:nil repeats:YES];
            [[NSRunLoop currentRunLoop] addTimer:timer forMode:NSDefaultRunLoopMode];
            // [[NSRunLoop currentRunLoop] addTimer:timer forMode:NSEventTrackingRunLoopMode];
        }
    }
    - (void)stopThread {
        if (displayLink) CVDisplayLinkStop(displayLink);
        if (timer) { [timer invalidate]; timer = nil; }
    }
    - (void)stopThreadAndExit { [self stopThread]; exit(0); }
    - (void)timerFired:(id)sender { [self setNeedsDisplay:YES]; }
    - (void)drawRect:(NSRect)dirtyRect { if (use_timer) [self getFrameForTime:0]; }
    - (void)getFrameForTime:(const CVTimeStamp*)outputTime {
        if (!video_thread_init && (video_thread_init = 1)) {
            [[self openGLContext] makeCurrentContext];
            SetLFAppMainThread();
            app = GetLFApp();
            screen = GetNativeWindow();
        }
        if (!app->initialized) return;
        if (app->run) LFAppFrame();
        else [self stopThreadAndExit];
    }
    - (void)reshape {
        float screen_w = [self frame].size.width, screen_h = [self frame].size.height;
        Reshaped((int)screen_w, (int)screen_h);
    }
    - (void)fileDataAvailable: (NSNotification *)notification { 
        // [self setNeedsDisplay:YES]; 
        [self getFrameForTime:nil];
        NSFileHandle *fh = (NSFileHandle*) [notification object];
        [fh waitForDataInBackgroundAndNotify];
    }
    - (void)fileReadCompleted: (NSNotification *)notification {}
    - (void)mouseDown:(NSEvent*)e { [self mouseClick:e down:1]; }
    - (void)mouseUp  :(NSEvent*)e { [self mouseClick:e down:0]; }
    - (void)mouseClick:(NSEvent*)e down:(bool)d {
        NSPoint p = [e locationInWindow];
        int fired = MouseClick(1, d, p.x, p.y);
        if (fired && frame_on_mouse_input) [self setNeedsDisplay:YES]; 
        prev_mouse_pos = p;
    }
    - (void)mouseDragged:(NSEvent*)e { [self mouseMove:e drag:1]; }
    - (void)mouseMoved:  (NSEvent*)e { [self mouseMove:e drag:0]; }
    - (void)mouseMove: (NSEvent*)e drag:(bool)drag {
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
        [(GameView*)GetNativeWindow()->id startThread];
    }
    - (void)createWindow: (int)w height:(int)h {
        INFOf("OSXVideoModule: AppDelegate::createWindow(%d, %d)", w, h);
        NSWindow *window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, w, h)
                                             styleMask:NSClosableWindowMask|NSMiniaturizableWindowMask|NSResizableWindowMask|NSTitledWindowMask
                                             backing:NSBackingStoreBuffered defer:NO];
        GameView *view = [[GameView alloc] initWithFrame:window.frame pixelFormat:GameView.defaultPixelFormat];
        [view setWindow:window];
        [[view openGLContext] setView:view];
        [window setContentView:view];
        [window center];
        [window makeKeyAndOrderFront:nil];
        if (1) {
            window.acceptsMouseMovedEvents = YES;
            [window makeFirstResponder:view];
        }
        GetNativeWindow()->id = view;
        [NSApp activateIgnoringOtherApps:YES];
    }
    - (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)theApplication {
        GetLFApp()->run = false;
        return YES;
    }
@end

extern "C" void NativeWindowInit() {}
extern "C" void NativeWindowQuit() {}
extern "C" int NativeWindowOrientation() { return 1; }
extern "C" void NativeWindowSize(int *width, int *height) {
    [(AppDelegate*)[NSApp delegate] createWindow:*width height:*height];
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

extern "C" void OSXAddWaitForeverMouse(void *O) { [(GameView*)O setFrameOnMouseInput:1]; }
extern "C" void OSXDelWaitForeverMouse(void *O) { [(GameView*)O setFrameOnMouseInput:0]; }
extern "C" void OSXAddWaitForeverKeyboard(void *O) { [(GameView*)O setFrameOnKeyboardInput:1]; }
extern "C" void OSXDelWaitForeverKeyboard(void *O) { [(GameView*)O setFrameOnKeyboardInput:0]; }
extern "C" void OSXAddWaitForeverSocket(void *O, int fd) {
    NSFileHandle *fh = [[NSFileHandle alloc] initWithFileDescriptor:fd];
    // [[NSNotificationCenter defaultCenter] addObserver:(GameView*)O
    //     selector:@selector(fileReadCompleted:) name:NSFileHandleReadCompletionNotification object:nil];
    // [fh readInBackgroundAndNotify];
    [[NSNotificationCenter defaultCenter] addObserver:(GameView*)O
        selector:@selector(fileDataAvailable:) name:NSFileHandleDataAvailableNotification object:nil];
    [fh waitForDataInBackgroundAndNotify];
}

extern "C" int main(int argc, const char **argv) {
    osx_argc = argc; osx_argv = argv;
    AppDelegate *app_delegate = [[AppDelegate alloc] init];
    [[NSApplication sharedApplication] setDelegate: app_delegate];
    return NSApplicationMain(argc, argv);
}
#endif // LFL_OSXVIDEO

#ifdef LFL_IPHONE
#import <UIKit/UIKit.h>
#import <UIKit/UIScreen.h>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/EAGLDrawable.h>
#ifdef LFL_GLES2
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#endif // LFL_GLES2
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#import <QuartzCore/QuartzCore.h>
#import "EAGLView.h"

#ifndef LFL_IPHONESIM 
#import <AVFoundation/AVFoundation.h>  /* IOS4+ only */
/* AVFoundation Class Reference: http://developer.apple.com/library/ios/#documentation/AVFoundation/Reference/AVCaptureSession_Class/Reference/Reference.html */
#endif // LFL_IPHONESIM

struct IPhoneKeyCode { enum { Backspace = 8, Return = 10 }; };
extern "C" int iPhoneMain(int argc, const char **argv);

static int iphone_argc = 0, iphone_evcount = 0;
static const char **iphone_argv = 0;
static NSString *documentsDirectory = nil;

// LFUIView
@interface LFUIView : UIView {}
@end

@implementation LFUIView
    - (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
        UITouch *touch = [touches anyObject];
        UIView *view = [touch view];
        int dpind = view.frame.origin.y == 0;
        CGPoint position = [touch locationInView:view];
        if (!dpind) {
            position.x += view.frame.origin.x;
            position.y += view.frame.origin.y;
        }
        gui_gesture_dpad_x[dpind] = position.x;
        gui_gesture_dpad_y[dpind] = position.y;
        click(1, 1, (int)position.x, (int)position.y);
    }
    - (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {}
    - (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
        UITouch *touch = [touches anyObject];
        UIView *view = [touch view];
        int dpind = view.frame.origin.y == 0;
        CGPoint position = [touch locationInView:view];
        if (!dpind) {
            position.x += view.frame.origin.x;
            position.y += view.frame.origin.y;
        }
        gui_gesture_dpad_stop[dpind] = 1;
        gui_gesture_dpad_x[dpind] = 0;
        gui_gesture_dpad_y[dpind] = 0;
        click(1, 0, (int)position.x, (int)position.y);
    }
    - (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
        UITouch *touch = [touches anyObject];
        UIView *view = [touch view];
        int dpind = view.frame.origin.y == 0;
        CGPoint position = [touch locationInView:view];
        if (!dpind) {
            position.x += view.frame.origin.x;
            position.y += view.frame.origin.y;
        }
        gui_gesture_dpad_x[dpind] = position.x;
        gui_gesture_dpad_y[dpind] = position.y;
    }
@end

// LFViewController
@interface LFViewController : UIViewController<UITextFieldDelegate> {}
@end

@implementation LFViewController
    - (BOOL)textField: (UITextField *)textField shouldChangeCharactersInRange:(NSRange)range replacementString:(NSString *)string {
        int l = [string length];
        if (!l) handleKey(IPhoneKeyCode::Backspace);
        for (int i = 0, l = [string length]; i < l; i++) {
            unichar k = [string characterAtIndex: i];
            handleKey(k);
        }
    }
    - (BOOL)textFieldShouldReturn: (UITextField *)textField {
        handleKey(IPhoneKeyCode::Return);
        [textField resignFirstResponder];
        return YES;
    }
    - (void)handleKey: (int)k { key(k, 1, 0, 0); key(k, 0, 0, 0); iphone_evcount++; }
@end

// LFUIApplication
@interface LFUIApplication : NSObject <UIApplicationDelegate> {}
    @property (nonatomic, retain) UIWindow *window;
    @property (nonatomic, retain) EAGLView *view;
    @property (nonatomic, retain) LFViewController *viewController;
    @property (nonatomic, retain) UIView *lview, *rview;
    @property (nonatomic, retain) UITextField *textField;
    + (LFUIApplication *) sharedAppDelegate;
    - (void) swapBuffers;
@end

@implementation LFUIApplication
    @synthesize window, viewController, view, lview, rview, textField;
    + (LFUIApplication *)sharedAppDelegate {
        return (LFUIApplication *)[[UIApplication sharedApplication] delegate];
    }
    - (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
        self.window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];
        
        [UIApplication sharedApplication].statusBarHidden = YES;
        [UIApplication sharedApplication].idleTimerDisabled = YES;
        
        self.viewController = [[LFViewController alloc] init];
        self.viewController.wantsFullScreenLayout = YES;
        
        CGRect wbounds = [self.window bounds];
        self.view = [[EAGLView alloc] initWithFrame:wbounds
            pixelFormat:kEAGLColorFormatRGB565	// kEAGLColorFormatRGBA8
            depthFormat:0                       // GL_DEPTH_COMPONENT16_OES
            preserveBackbuffer:false];

        // left view  
        CGRect lrect = CGRectMake(0, 0, wbounds.size.width, wbounds.size.height/2);
        self.lview = [[LFUIView alloc] initWithFrame:lrect];
        // self.lview.backgroundColor = [UIColor greenColor];
        // self.lview.alpha = 0.3f;
        [self.view addSubview:self.lview];
        
        // right view  
        CGRect rrect = CGRectMake(0, wbounds.size.height/2, wbounds.size.width, wbounds.size.height/2);
        self.rview = [[LFUIView alloc] initWithFrame:rrect];
        // self.rview.backgroundColor = [UIColor blueColor];
        // self.rview.alpha = 0.3f;
        [self.view addSubview:self.rview];

        // text view for keyboard display
        self.textField = [[UITextField alloc] initWithFrame: CGRectZero];
        self.textField.delegate = self.viewController;
        self.textField.text = [NSString stringWithFormat:@"default"];
        [self.view addSubview:self.textField];

        [self.viewController setView:self.view]; 
        [self.window addSubview:self.view];
        [self.view setCurrentContext];

        [[NSFileManager defaultManager] changeCurrentDirectoryPath: [[NSBundle mainBundle] resourcePath]];
        iPhoneMain(iphone_argc, iphone_argv);

        [self performSelector:@selector(postFinishLaunch) withObject:nil afterDelay:0.0];
        return YES;
    }
    - (void)postFinishLaunch { LFAppMain(); }
    - (void)applicationWillTerminate:(UIApplication *)application {}
    - (void)applicationWillResignActive:(UIApplication*)application {}
    - (void)applicationDidBecomeActive:(UIApplication*)application {}
    - (void)swapBuffers {
        [self.view swapBuffers];
        [self.window makeKeyAndVisible];
    }
    - (void)showKeyboard { [self.textField becomeFirstResponder]; }
@end

@interface LFApp : NSObject { }
    - (int) orientation;
    - (id) init;
    + (LFApp *) sharedApp;
    - (void) initNotifications;
    - (void) shutdownNotifications;
    - (void) initGestureRecognizers;
    - (void) shutdownGestureRecognizers;
@end

@implementation LFApp
    static LFApp *gLFapp = nil;
    static int currentOrientation = 0;

    - (id) init {
        fprintf(stderr, "LFApp init\n");
        self = [super init];
        if (!self) return nil;
        [self initNotifications];
        [self initGestureRecognizers];
        return self;
    }
    - (void) dealloc {
        fprintf(stderr, "dealloc LFApp\n");
        [self shutdownNotifications];
        [self shutdownGestureRecognizers];
        [super dealloc];
    }
    + (void) shutdown {
        fprintf(stderr, "shutdown LFApp\n");
        if (gLFapp != nil) { [gLFapp release]; gLFapp = nil; }
    }
    + (LFApp*) sharedApp {
        if (gLFapp == nil) {
            fprintf(stderr, "sharedApp alloc/init LFApp\n");
            gLFapp = [[[LFApp alloc] init] retain];
        }
        return gLFapp;
    }
    - (void) shutdownNotifications {
        fprintf(stderr, "shutdown notifications\n");
        [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
        [[NSNotificationCenter defaultCenter] removeObserver:self name:@"UIDeviceOrientationDidChangeNotification" object:nil]; 
    }
    - (void) orientationChanged: (id)sender {
        fprintf(stderr, "notification of new orientation: %d -> %d \n", currentOrientation, [[UIDevice currentDevice] orientation]);
        currentOrientation = [[UIDevice currentDevice] orientation];
    }
    - (int) orientation {
        fprintf(stderr, "status bar orientation: %d -> %d\n", currentOrientation, [[UIApplication sharedApplication] statusBarOrientation]);
        currentOrientation = [[UIApplication sharedApplication] statusBarOrientation];
        return currentOrientation;
    }
    - (void) initNotifications {
        fprintf(stderr, "init notifications\n");
        [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications]; 
        [[NSNotificationCenter defaultCenter] addObserver:self
            selector:@selector(orientationChanged:)
            name:@"UIDeviceOrientationDidChangeNotification" object:nil];
    }
    - (void) initGestureRecognizers {
        UIWindow *window = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
        fprintf(stderr, "UIWindow frame: %s", [NSStringFromCGRect(window.frame) cString]);

        UISwipeGestureRecognizer *up = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(doubleSwipeUp:)];
        up.direction = UISwipeGestureRecognizerDirectionUp;
        up.numberOfTouchesRequired = 2;
        [window addGestureRecognizer:up];
        [up release];

        UISwipeGestureRecognizer *down = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(doubleSwipeDown:)];
        down.direction = UISwipeGestureRecognizerDirectionDown;
        down.numberOfTouchesRequired = 2;
        [window addGestureRecognizer:down];
        [down release];
#if 0
        // one-finger press and drag sends control stick movements- up, down, left, right
        UIPanGestureRecognizer *pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(panGesture:)];
        pan.minimumNumberOfTouches = 1;
        pan.maximumNumberOfTouches = 1;
        [[LFUIApplication sharedAppDelegate].lview addGestureRecognizer:pan];
        [pan release]; pan = nil;

        pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(panGesture:)];
        pan.minimumNumberOfTouches = 1;
        pan.maximumNumberOfTouches = 1;
        [[LFUIApplication sharedAppDelegate].rview addGestureRecognizer:pan];
        [pan release]; pan = nil;

        // one-finger hit sends jump movement
        UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(tapGesture:)];
        [[LFUIApplication sharedAppDelegate].view addGestureRecognizer:tap];
        [tap release];
#endif
    }
    - (void) shutdownGestureRecognizers {
        UIWindow *window = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
        NSArray *gestures = [window gestureRecognizers];
        for (int i = 0; i < [gestures count]; i++)
            [window removeGestureRecognizer:[gestures objectAtIndex:i]];
    }
    - (void) doubleSwipeUp:   (id)sender { gui_gesture_swipe_up   = 1; }
    - (void) doubleSwipeDown: (id)sender { gui_gesture_swipe_down = 1; }
    - (void) tapGesture: (UITapGestureRecognizer *)tapGestureRecognizer {
        UIView *view = [tapGestureRecognizer view];
        CGPoint position = [tapGestureRecognizer locationInView:view];
        int dpind = view.frame.origin.y == 0;

        click(1, 1, (int)position.x, (int)position.y);
        gui_gesture_tap[dpind] = 1;
        gui_gesture_dpad_x[dpind] = position.x;
        gui_gesture_dpad_y[dpind] = position.y;
    }
    - (void) panGesture: (UIPanGestureRecognizer *)panGestureRecognizer {
        UIView *view = [panGestureRecognizer view];
        int dpind = view.frame.origin.y == 0;

        if (panGestureRecognizer.state == UIGestureRecognizerStateChanged) {
            // CGPoint velocity = [panGestureRecognizer translationInView:view];
            CGPoint velocity = [panGestureRecognizer velocityInView:view];
            if (fabs(velocity.x) > 15 || fabs(velocity.y) > 15) {
                gui_gesture_dpad_dx[dpind] = velocity.x;
                gui_gesture_dpad_dy[dpind] = velocity.y;
            }
            // CGPoint position = [panGestureRecognizer locationInView:view];
            CGPoint position = [panGestureRecognizer locationOfTouch:0 inView:view];
            if (!dpind) {
                position.x += view.frame.origin.x;
                position.y += view.frame.origin.y;
            }
            gui_gesture_dpad_x[dpind] = position.x;
            gui_gesture_dpad_y[dpind] = position.y;
            // fprintf(stderr, "gest %f %f %f %f\n", position.x, position.y, velocity.x, velocity.y);
            // fprintf(stderr, "origin %f %f \n", view.frame.origin.x, view.frame.origin.y);
        }
        else if (panGestureRecognizer.state == UIGestureRecognizerStateEnded) {
            gui_gesture_dpad_stop[dpind] = 1;
            gui_gesture_dpad_x[dpind] = 0;
            gui_gesture_dpad_y[dpind] = 0;
            // CGPoint position = [panGestureRecognizer locationInView:view];
            // fprintf(stderr, "gest %f %f stop\n", position.x, position.y);
        }
    }
@end

extern "C" void NativeWindowInit() {
    LFApp *lfApp = [LFApp sharedApp];
}

extern "C" void NativeWindowQuit() {
    if (documentsDirectory != nil) { [documentsDirectory release]; documentsDirectory = nil; }
}

extern "C" void NativeWindowSize(int *width, int *height) {
    CGRect rect = [[UIScreen mainScreen] bounds];
    [UIApplication sharedApplication].statusBarHidden = YES;
    *width = rect.size.width;
    *height = rect.size.height;
}

extern "C" int NativeWindowOrientation() { return currentOrientation; }

extern "C" int iPhoneVideoSwap() {
    [[LFUIApplication sharedAppDelegate] swapBuffers];
    return 0;
}

extern "C" int iPhoneShowKeyboard() {
    [[LFUIApplication sharedAppDelegate] showKeyboard];
    return 0;
}

extern "C" int iPhoneInput(unsigned clicks, unsigned *events) {
    SInt32 result;
    do {
        result = CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, TRUE);
    } while (result == kCFRunLoopRunHandledSource);

    if (events) *events = iphone_evcount = 0;
    iphone_evcount = 0;
    return 0;
}

extern "C" int iPhoneOpenBrowser(const char *url_text) {
    NSString *url_string = [[NSString alloc] initWithUTF8String: url_text];
    NSURL *url = [NSURL URLWithString: url_string];  
    [[UIApplication sharedApplication] openURL:url];
    return 0;
}

extern "C" void *iPhoneLoadMusicAsset(const char *filename) {
#ifdef LFL_IPHONESIM
    return 0;
#else // LFL_IPHONESIM
    NSError *error;
    NSString *fn = [NSString stringWithCString:filename encoding:NSASCIIStringEncoding];
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithFormat:@"%@/%@", [[NSBundle mainBundle] resourcePath], fn]];
    AVAudioPlayer *audioPlayer = [[AVAudioPlayer alloc] initWithContentsOfURL:url error:&error];
    if (audioPlayer == nil) NSLog([error description]);
    return audioPlayer;
#endif // LFL_IPHONESIM
}

extern "C" void iPhonePlayMusic(void *handle) {
    AVAudioPlayer *audioPlayer = (AVAudioPlayer*)handle;
    [audioPlayer play];
}

extern "C" void iPhonePlayBackgroundMusic(void *handle) {
    AVAudioPlayer *audioPlayer = (AVAudioPlayer*)handle;
    audioPlayer.numberOfLoops = -1;
    [audioPlayer play];
}

extern "C" char *iPhoneDocumentPath() {
    if (documentsDirectory == nil) {
        NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
        documentsDirectory = [[paths objectAtIndex:0] copy];
    }
    return strdup([documentsDirectory UTF8String]);
}

extern "C" int iPhoneReadDir(const char *path, int dirs,
                             void *DirectoryIter, void (*DirectoryIterAdd)(void *di, const char *k, int)) {
    NSString *dirName = [[NSString alloc] initWithUTF8String:path];
    NSArray *dirContents = [[NSFileManager defaultManager] directoryContentsAtPath:dirName];

    for (NSString *fileName in dirContents) {
        NSString *fullPath = [dirName stringByAppendingPathComponent:fileName];

        BOOL isDir;
        if (![[NSFileManager defaultManager] fileExistsAtPath:fullPath isDirectory:&isDir]) continue;
        if (dirs >= 0 && isDir != dirs) continue;

        if (isDir) fileName = [NSString stringWithFormat:@"%@%@", fileName, @"/"];

        DirectoryIterAdd(DirectoryIter, [fileName UTF8String], 1);
    }
    return 0;
}

extern "C" int main(int ac, const char **av) {
    iphone_argc = ac; iphone_argv = av;
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    int ret = UIApplicationMain(iphone_argc, iphone_argv, nil, @"LFUIApplication");
    [pool release];
    return ret;
}

#endif /* LFL_IPHONE */
