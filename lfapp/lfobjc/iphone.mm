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

#ifdef LFL_IPHONE
#import <UIKit/UIKit.h>
#import <UIKit/UIScreen.h>
#import <GLKit/GLKit.h>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/EAGLDrawable.h>
#ifdef LFL_GLES2
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#endif // LFL_GLES2
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#import <QuartzCore/QuartzCore.h>
#ifndef LFL_IPHONESIM 
#import <AVFoundation/AVFoundation.h>
#endif // LFL_IPHONESIM

extern "C" int iPhoneMain(int argc, const char **argv);

struct IPhoneKeyCode { enum { Backspace = 8, Return = 10 }; };
static NSString *iphone_documents_directory = nil;
static const char **iphone_argv = 0;
static int iphone_argc = 0;

@interface MyTouchView : UIView {}
@end

@interface LFViewController : GLKViewController {}
@end

@implementation LFViewController
    - (void)viewWillAppear:(BOOL)animated { 
        [super viewWillAppear:animated];
        [self setPaused:YES];
    }
@end

@interface LFUIApplication : NSObject <UIApplicationDelegate, GLKViewDelegate, GLKViewControllerDelegate, UITextFieldDelegate> {}
    @property (nonatomic, retain) UIWindow *window;
    @property (nonatomic, retain) LFViewController *controller;
    @property (nonatomic, retain) GLKView *view;
    @property (nonatomic, retain) UIView *lview, *rview;
    @property (nonatomic, retain) UITextField *textField;
    @property BOOL resign_textfield_on_return, frame_on_keyboard_input, frame_on_mouse_input;
    + (LFUIApplication *) sharedAppDelegate;
@end

@implementation LFUIApplication
    {
        NativeWindow *screen;
        int current_orientation;
        CGRect keyboard_frame;
    }
    @synthesize window, controller, view, lview, rview, textField;

    + (LFUIApplication *)sharedAppDelegate { return (LFUIApplication *)[[UIApplication sharedApplication] delegate]; }
    - (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
        CGRect wbounds = [[UIScreen mainScreen] bounds];
        self.window = [[UIWindow alloc] initWithFrame:wbounds];

        EAGLContext *context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
        [EAGLContext setCurrentContext:context];
        self.view = [[GLKView alloc] initWithFrame:wbounds];
        self.view.context = context;
        self.view.delegate = self;

        self.controller = [[LFViewController alloc] initWithNibName:nil bundle:nil];
        self.controller.wantsFullScreenLayout = YES;
        self.controller.delegate = self;
        self.controller.resumeOnDidBecomeActive = NO;
        [self.controller setView:self.view]; 

        // left touch view
        CGRect lrect = CGRectMake(0, 0, wbounds.size.width, wbounds.size.height/2);
        self.lview = [[MyTouchView alloc] initWithFrame:lrect];
        // self.lview.backgroundColor = [UIColor greenColor];
        // self.lview.alpha = 0.3f;
        [self.view addSubview:self.lview];
        
        // right touch view
        CGRect rrect = CGRectMake(0, wbounds.size.height/2, wbounds.size.width, wbounds.size.height/2);
        self.rview = [[MyTouchView alloc] initWithFrame:rrect];
        // self.rview.backgroundColor = [UIColor blueColor];
        // self.rview.alpha = 0.3f;
        [self.view addSubview:self.rview];

        // text view for keyboard display
        _resign_textfield_on_return = YES;
        _frame_on_keyboard_input = _frame_on_mouse_input = YES;
        self.textField = [[UITextField alloc] initWithFrame: CGRectZero];
        self.textField.delegate = self;
        self.textField.text = [NSString stringWithFormat:@"default"];
        self.textField.autocorrectionType = UITextAutocorrectionTypeNo;
        self.textField.autocapitalizationType = UITextAutocapitalizationTypeNone;
        [self.view addSubview:self.textField];

        [UIApplication sharedApplication].statusBarHidden = YES;
        [UIApplication sharedApplication].idleTimerDisabled = YES;
        self.view.enableSetNeedsDisplay = TRUE;
        self.window.rootViewController = self.controller;
        [self.window addSubview:self.view];
        [self.window makeKeyAndVisible];

        [[NSFileManager defaultManager] changeCurrentDirectoryPath: [[NSBundle mainBundle] resourcePath]];
        NSLog(@"iPhoneMain argc=%d", iphone_argc);
        iPhoneMain(iphone_argc, iphone_argv);
        INFOf("didFinishLaunchingWithOptions, views: %p, %p, %p", self.view, self.lview, self.rview);

        screen = GetNativeWindow();
        [self initNotifications];
        [self initGestureRecognizers];
        return YES;
    }
    - (void)applicationWillTerminate:(UIApplication *)application {
        [self shutdownNotifications];
        [self shutdownGestureRecognizers];
    }
    - (void)applicationWillResignActive:(UIApplication*)application {}
    - (void)applicationDidBecomeActive:(UIApplication*)application {}
    - (void)glkView:(GLKView *)view drawInRect:(CGRect)rect { LFAppFrame(); }
    - (void)glkViewControllerUpdate:(GLKViewController *)controller {}

    - (void)initNotifications {
        [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications]; 
        NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
        INFOf("init notifications %p", center);
        [center addObserver:self selector:@selector(orientationChanged:) name:@"UIDeviceOrientationDidChangeNotification" object:nil];
        [center addObserver:self selector:@selector(keyboardDidShow:)    name:@"UIKeyboardDidShowNotification"            object:nil];
        [center addObserver:self selector:@selector(keyboardWillHide:)   name:@"UIKeyboardWillHideNotification"           object:nil];
    }
    - (void)shutdownNotifications {
        INFOf("%s", "shutdown notifications");
        [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
        [[NSNotificationCenter defaultCenter] removeObserver:self]; 
    }

    - (void)showKeyboard { [self.textField becomeFirstResponder]; }
    - (void)hideKeyboard { [self.textField resignFirstResponder]; }
    - (CGRect) getKeyboardFrame { return keyboard_frame; }
    - (void)keyboardWillHide:(NSNotification *)notification {}
    - (void)keyboardDidShow:(NSNotification *)notification {
        NSDictionary *userInfo = [notification userInfo];
        CGRect rect = [[userInfo objectForKey:UIKeyboardFrameEndUserInfoKey] CGRectValue];
        keyboard_frame = [self.window convertRect:rect toView:nil];
    }

    - (BOOL)textField: (UITextField *)textField shouldChangeCharactersInRange:(NSRange)range replacementString:(NSString *)string {
        int l = [string length];
        if (!l) [self handleKey:IPhoneKeyCode::Backspace];
        for (int i = 0, l = [string length]; i < l; i++) {
            unichar k = [string characterAtIndex: i];
            [self handleKey: k];
        }
        return YES;
    }
    - (BOOL)textFieldShouldReturn: (UITextField *)tf {
        [self handleKey:IPhoneKeyCode::Return];
        if (_resign_textfield_on_return) [tf resignFirstResponder];
        return YES;
    }
    - (void)handleKey: (int)k {
        int fired = 0;
        fired += KeyPress(k, 1);
        fired += KeyPress(k, 0);
        if (fired && _frame_on_keyboard_input) [self.view setNeedsDisplay];
    }

    - (int)getOrientation { return current_orientation; }
    - (void)orientationChanged: (id)sender {
        INFOf("notification of new orientation: %d -> %d", current_orientation, [[UIDevice currentDevice] orientation]);
        current_orientation = [[UIDevice currentDevice] orientation];
    }
    - (int)orientation {
        INFOf("status bar orientation: %d -> %d", current_orientation, [[UIApplication sharedApplication] statusBarOrientation]);
        current_orientation = [[UIApplication sharedApplication] statusBarOrientation];
        return current_orientation;
    }

    - (void)initGestureRecognizers {
        UIWindow *win = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
        INFOf("UIWindow frame: %s", [NSStringFromCGRect(win.frame) cString]);

        UISwipeGestureRecognizer *up = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(doubleSwipeUp:)];
        up.direction = UISwipeGestureRecognizerDirectionUp;
        up.numberOfTouchesRequired = 2;
        [win addGestureRecognizer:up];
        [up release];

        UISwipeGestureRecognizer *down = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(doubleSwipeDown:)];
        down.direction = UISwipeGestureRecognizerDirectionDown;
        down.numberOfTouchesRequired = 2;
        [win addGestureRecognizer:down];
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
    - (void)shutdownGestureRecognizers {
        UIWindow *win = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
        NSArray *gestures = [win gestureRecognizers];
        for (int i = 0; i < [gestures count]; i++)
            [win removeGestureRecognizer:[gestures objectAtIndex:i]];
    }
    - (void)doubleSwipeUp:   (id)sender { screen->gesture_swipe_up   = 1; }
    - (void)doubleSwipeDown: (id)sender { screen->gesture_swipe_down = 1; }
    - (void)tapGesture: (UITapGestureRecognizer *)tapGestureRecognizer {
        UIView *v = [tapGestureRecognizer view];
        CGPoint position = [tapGestureRecognizer locationInView:v];
        int dpind = v.frame.origin.y == 0;
        screen->gesture_tap[dpind] = 1;
        screen->gesture_dpad_x[dpind] = position.x;
        screen->gesture_dpad_y[dpind] = position.y;
        int fired = MouseClick(1, 1, (int)position.x, screen->height - (int)position.y);
        if (fired && _frame_on_mouse_input) [self.view setNeedsDisplay];
    }
    - (void)panGesture: (UIPanGestureRecognizer *)panGestureRecognizer {
        UIView *v = [panGestureRecognizer view];
        int dpind = v.frame.origin.y == 0;

        if (panGestureRecognizer.state == UIGestureRecognizerStateChanged) {
            // CGPoint velocity = [panGestureRecognizer translationInView:v];
            CGPoint velocity = [panGestureRecognizer velocityInView:v];
            if (fabs(velocity.x) > 15 || fabs(velocity.y) > 15) {
                screen->gesture_dpad_dx[dpind] = velocity.x;
                screen->gesture_dpad_dy[dpind] = velocity.y;
            }
            // CGPoint position = [panGestureRecognizer locationInView:v];
            CGPoint position = [panGestureRecognizer locationOfTouch:0 inView:v];
            if (!dpind) {
                position.x += v.frame.origin.x;
                position.y += v.frame.origin.y;
            }
            screen->gesture_dpad_x[dpind] = position.x;
            screen->gesture_dpad_y[dpind] = position.y;
            // INFOf("gest %f %f %f %f", position.x, position.y, velocity.x, velocity.y);
            // INFOf("origin %f %f", v.frame.origin.x, v.frame.origin.y);
        }
        else if (panGestureRecognizer.state == UIGestureRecognizerStateEnded) {
            screen->gesture_dpad_stop[dpind] = 1;
            screen->gesture_dpad_x[dpind] = 0;
            screen->gesture_dpad_y[dpind] = 0;
            // CGPoint position = [panGestureRecognizer locationInView:v];
            // INFOf("gest %f %f stop", position.x, position.y);
        }
    }
@end

@implementation MyTouchView
    {
        NativeWindow *screen;
        LFUIApplication *app;
    }
    - (id)initWithFrame:(CGRect)aRect {
        if (!(self = [super initWithFrame:aRect])) return self;
        screen = GetNativeWindow();
        app = [LFUIApplication sharedAppDelegate];
        return self;
    }
    - (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
        UITouch *touch = [touches anyObject];
        UIView *v = [touch view];
        CGPoint position = [touch locationInView:v];
        int dpind = v.frame.origin.y == 0;
        if (!dpind) position = CGPointMake(position.x + v.frame.origin.x, position.y + v.frame.origin.y);
        screen->gesture_dpad_x[dpind] = position.x;
        screen->gesture_dpad_y[dpind] = position.y;
        int fired = MouseClick(1, 1, (int)position.x, screen->height - (int)position.y);
        if (fired && app.frame_on_mouse_input) [self.superview setNeedsDisplay];
    }
    - (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {}
    - (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
        UITouch *touch = [touches anyObject];
        UIView *v = [touch view];
        CGPoint position = [touch locationInView:v];
        int dpind = v.frame.origin.y == 0;
        if (!dpind) position = CGPointMake(position.x + v.frame.origin.x, position.y + v.frame.origin.y);
        screen->gesture_dpad_stop[dpind] = 1;
        screen->gesture_dpad_x[dpind] = 0;
        screen->gesture_dpad_y[dpind] = 0;
        int fired = MouseClick(1, 0, (int)position.x, screen->height - (int)position.y);
        if (fired && app.frame_on_mouse_input) [self.superview setNeedsDisplay];
    }
    - (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
        UITouch *touch = [touches anyObject];
        UIView *v = [touch view];
        CGPoint position = [touch locationInView:v];
        int dpind = v.frame.origin.y == 0;
        if (!dpind) position = CGPointMake(position.x + v.frame.origin.x, position.y + v.frame.origin.y);
        screen->gesture_dpad_x[dpind] = position.x;
        screen->gesture_dpad_y[dpind] = position.y;
    }
@end

extern "C" void NativeWindowInit() { 
    NativeWindow *screen = GetNativeWindow();
    screen->opengles_version = 2;
    screen->id = [[LFUIApplication sharedAppDelegate] view];
}
extern "C" int NativeWindowOrientation() { return [[LFUIApplication sharedAppDelegate] getOrientation]; }
extern "C" void NativeWindowQuit() {
    if (iphone_documents_directory != nil) { [iphone_documents_directory release]; iphone_documents_directory = nil; }
}
extern "C" void NativeWindowSize(int *width, int *height) {
    CGRect rect = [[UIScreen mainScreen] bounds];
    [UIApplication sharedApplication].statusBarHidden = YES;
    *width = rect.size.width;
    *height = rect.size.height;
    INFOf("NativeWindowSize %d %d", *width, *height);
}

extern "C" int iPhoneVideoSwap() { return 0; }
extern "C" void iPhoneShowKeyboard() { [[LFUIApplication sharedAppDelegate] showKeyboard]; }
extern "C" void iPhoneHideKeyboard() { [[LFUIApplication sharedAppDelegate] hideKeyboard]; }
extern "C" void iPhoneHideKeyboardAfterReturn(bool v) { [LFUIApplication sharedAppDelegate].resign_textfield_on_return = v; }
extern "C" void iPhoneGetKeyboardBox(int *x, int *y, int *w, int *h) {
    NativeWindow *screen = GetNativeWindow();
    CGRect rect = [[LFUIApplication sharedAppDelegate] getKeyboardFrame];
    *x = rect.origin.x;
    *y = rect.origin.y + rect.size.height - screen->height;
    *w = rect.size.width;
    *h = rect.size.height;
}

extern "C" void iPhoneTriggerFrame(void *O) {
    dispatch_async(dispatch_get_main_queue(), ^{ [(GLKView*)O setNeedsDisplay]; });
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
    AVAudioPlayer *audio_player = [[AVAudioPlayer alloc] initWithContentsOfURL:url error:&error];
    if (audio_player == nil) NSLog(@"%@", [error description]);
    return audio_player;
#endif // LFL_IPHONESIM
}

extern "C" void iPhonePlayMusic(void *handle) {
#ifndef LFL_IPHONESIM
    AVAudioPlayer *audioPlayer = (AVAudioPlayer*)handle;
    [audioPlayer play];
#endif
}

extern "C" void iPhonePlayBackgroundMusic(void *handle) {
#ifndef LFL_IPHONESIM
    AVAudioPlayer *audioPlayer = (AVAudioPlayer*)handle;
    audioPlayer.numberOfLoops = -1;
    [audioPlayer play];
#endif
}

extern "C" char *iPhoneDocumentPath() {
    if (iphone_documents_directory == nil) {
        NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
        iphone_documents_directory = [[paths objectAtIndex:0] copy];
    }
    return strdup([iphone_documents_directory UTF8String]);
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

extern "C" void iPhoneLog(const char *text) {
    NSString *t = [[NSString alloc] initWithUTF8String: text];
    NSLog(@"%@", t);
}

extern "C" int main(int ac, const char **av) {
    iphone_argc = ac;
    iphone_argv = av;
    NSLog(@"%@", @"lfapp_lfobjc_iphone_main");
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    int ret = UIApplicationMain(ac, (char**)av, nil, @"LFUIApplication");
    [pool release];
    return ret;
}

#endif /* LFL_IPHONE */
