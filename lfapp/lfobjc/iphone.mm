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
#import <AVFoundation/AVFoundation.h>
#endif // LFL_IPHONESIM

struct IPhoneKeyCode { enum { Backspace = 8, Return = 10 }; };
static const char **iphone_argv = 0;
static int iphone_argc = 0, iphone_evcount = 0;
static NSString *iphone_documents_directory = nil;
extern "C" int iPhoneMain(int argc, const char **argv);

// LFUIView
@interface LFUIView : UIView {}
@end

@implementation LFUIView
    {
        NativeWindow *screen;
    }
    - (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
        UITouch *touch = [touches anyObject];
        UIView *view = [touch view];
        int dpind = view.frame.origin.y == 0;
        CGPoint position = [touch locationInView:view];
        if (!dpind) {
            position.x += view.frame.origin.x;
            position.y += view.frame.origin.y;
        }
        screen->gesture_dpad_x[dpind] = position.x;
        screen->gesture_dpad_y[dpind] = position.y;
        MouseClick(1, 1, (int)position.x, (int)position.y);
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
        screen->gesture_dpad_stop[dpind] = 1;
        screen->gesture_dpad_x[dpind] = 0;
        screen->gesture_dpad_y[dpind] = 0;
        MouseClick(1, 0, (int)position.x, (int)position.y);
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
        screen->gesture_dpad_x[dpind] = position.x;
        screen->gesture_dpad_y[dpind] = position.y;
    }
@end

// LFViewController
@interface LFViewController : UIViewController<UITextFieldDelegate> {}
@end

@implementation LFViewController
    - (BOOL)textField: (UITextField *)textField shouldChangeCharactersInRange:(NSRange)range replacementString:(NSString *)string {
        int l = [string length];
        if (!l) [self handleKey:IPhoneKeyCode::Backspace];
        for (int i = 0, l = [string length]; i < l; i++) {
            unichar k = [string characterAtIndex: i];
            [self handleKey: k];
        }
        return YES;
    }
    - (BOOL)textFieldShouldReturn: (UITextField *)textField {
        [self handleKey:IPhoneKeyCode::Return];
        [textField resignFirstResponder];
        return YES;
    }
    - (void)handleKey: (int)k { KeyPress(k, 1); KeyPress(k, 0); iphone_evcount++; }
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

@interface LFApplication : NSObject { }
    - (int) orientation;
    - (id) init;
    + (LFApplication *) sharedApp;
    - (void) initNotifications;
    - (void) shutdownNotifications;
    - (void) initGestureRecognizers;
    - (void) shutdownGestureRecognizers;
@end

@implementation LFApplication
    {
        int current_orientation;
        NativeWindow *screen;
    }
    static LFApplication *gLFapp = 0;
    - (id) init {
        fprintf(stderr, "LFApp init\n");
        self = [super init];
        if (!self) return nil;
        screen = GetNativeWindow();
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
    + (LFApplication*) sharedApp {
        if (gLFapp == nil) {
            fprintf(stderr, "sharedApp alloc/init LFApp\n");
            gLFapp = [[[LFApplication alloc] init] retain];
        }
        return gLFapp;
    }
    - (void) shutdownNotifications {
        fprintf(stderr, "shutdown notifications\n");
        [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
        [[NSNotificationCenter defaultCenter] removeObserver:self name:@"UIDeviceOrientationDidChangeNotification" object:nil]; 
    }
    - (void) orientationChanged: (id)sender {
        fprintf(stderr, "notification of new orientation: %d -> %d \n", current_orientation, [[UIDevice currentDevice] orientation]);
        current_orientation = [[UIDevice currentDevice] orientation];
    }
    - (int) orientation {
        fprintf(stderr, "status bar orientation: %d -> %d\n", current_orientation, [[UIApplication sharedApplication] statusBarOrientation]);
        current_orientation = [[UIApplication sharedApplication] statusBarOrientation];
        return current_orientation;
    }
    - (int) getOrientation { return current_orientation; }
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
    - (void) doubleSwipeUp:   (id)sender { screen->gesture_swipe_up   = 1; }
    - (void) doubleSwipeDown: (id)sender { screen->gesture_swipe_down = 1; }
    - (void) tapGesture: (UITapGestureRecognizer *)tapGestureRecognizer {
        UIView *view = [tapGestureRecognizer view];
        CGPoint position = [tapGestureRecognizer locationInView:view];
        int dpind = view.frame.origin.y == 0;

        MouseClick(1, 1, (int)position.x, (int)position.y);
        screen->gesture_tap[dpind] = 1;
        screen->gesture_dpad_x[dpind] = position.x;
        screen->gesture_dpad_y[dpind] = position.y;
    }
    - (void) panGesture: (UIPanGestureRecognizer *)panGestureRecognizer {
        UIView *view = [panGestureRecognizer view];
        int dpind = view.frame.origin.y == 0;

        if (panGestureRecognizer.state == UIGestureRecognizerStateChanged) {
            // CGPoint velocity = [panGestureRecognizer translationInView:view];
            CGPoint velocity = [panGestureRecognizer velocityInView:view];
            if (fabs(velocity.x) > 15 || fabs(velocity.y) > 15) {
                screen->gesture_dpad_dx[dpind] = velocity.x;
                screen->gesture_dpad_dy[dpind] = velocity.y;
            }
            // CGPoint position = [panGestureRecognizer locationInView:view];
            CGPoint position = [panGestureRecognizer locationOfTouch:0 inView:view];
            if (!dpind) {
                position.x += view.frame.origin.x;
                position.y += view.frame.origin.y;
            }
            screen->gesture_dpad_x[dpind] = position.x;
            screen->gesture_dpad_y[dpind] = position.y;
            // fprintf(stderr, "gest %f %f %f %f\n", position.x, position.y, velocity.x, velocity.y);
            // fprintf(stderr, "origin %f %f \n", view.frame.origin.x, view.frame.origin.y);
        }
        else if (panGestureRecognizer.state == UIGestureRecognizerStateEnded) {
            screen->gesture_dpad_stop[dpind] = 1;
            screen->gesture_dpad_x[dpind] = 0;
            screen->gesture_dpad_y[dpind] = 0;
            // CGPoint position = [panGestureRecognizer locationInView:view];
            // fprintf(stderr, "gest %f %f stop\n", position.x, position.y);
        }
    }
@end

extern "C" void NativeWindowInit() {
    LFApplication *lfApp = [LFApplication sharedApp];
}

extern "C" void NativeWindowQuit() {
    if (iphone_documents_directory != nil) { [iphone_documents_directory release]; iphone_documents_directory = nil; }
}

extern "C" void NativeWindowSize(int *width, int *height) {
    CGRect rect = [[UIScreen mainScreen] bounds];
    [UIApplication sharedApplication].statusBarHidden = YES;
    *width = rect.size.width;
    *height = rect.size.height;
}

extern "C" int NativeWindowOrientation() { return [[LFApplication sharedApp] getOrientation]; }

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
    AVAudioPlayer *audio_player = [[AVAudioPlayer alloc] initWithContentsOfURL:url error:&error];
    if (audio_player == nil) NSLog(@"%@", [error description]);
    return audio_player;
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

extern "C" int main(int ac, const char **av) {
    iphone_argc = ac;
    iphone_argv = av;
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    int ret = UIApplicationMain(iphone_argc, (char**)iphone_argv, nil, @"LFUIApplication");
    [pool release];
    return ret;
}

#endif /* LFL_IPHONE */
