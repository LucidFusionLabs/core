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

static int DoNothing(void *) { return 0; }

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
extern "C" void NativeWindowInit() {
#if 0
    NativeThreadStart(DoNothing, 0);
    BOOL mt1 = [NSThread isMultiThreaded];
    printf("NSthread multithread %d\n", mt1);
#endif
}
extern "C" void NativeWindowQuit() {}
extern "C" void NativeWindowSize(int *widthOut, int *heightOut) {}
extern "C" int NativeWindowOrientation() { return 1; }

#import <AppKit/NSFont.h>
#import <AppKit/NSFontManager.h>
int NSFMreadfonts(void *FontsIter, void (*FontsIterAdd)(void *fi, const char *name, const char *family, int flag)) {
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

#else // LFL_IPHONE

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

int iphone_evcount = 0;
extern int gui_gesture_swipe_up, gui_gesture_swipe_down, gui_gesture_tap[2], gui_gesture_dpad_stop[2];
extern float gui_gesture_dpad_x[2], gui_gesture_dpad_y[2], gui_gesture_dpad_dx[2], gui_gesture_dpad_dy[2];
extern void key(int button, int down, int, int);
extern void click(int button, int down, int x, int y);

extern "C" int iphone_main(int argc, char **argv);
extern int lfapp_main();

static int iphone_argc;
static char **iphone_argv;

int main(int ac, char **av) {
	iphone_argc = ac; iphone_argv = av;
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    int retVal = UIApplicationMain(iphone_argc, iphone_argv, nil, @"LFUIApplication");
    [pool release];
    return retVal;
}

// LFViewController
@interface LFViewController : UIViewController<UITextFieldDelegate> {}
@end

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

struct IPhoneKeyCode { enum { Backspace = 8, Return = 10 }; };
#define handle_key(k) { key((k), 1, 0, 0); key((k), 0, 0, 0); iphone_evcount++; }
@implementation LFViewController
    - (BOOL)textField: (UITextField *)textField shouldChangeCharactersInRange:(NSRange)range replacementString:(NSString *)string {
        int l = [string length];
        if (!l) handle_key(IPhoneKeyCode::Backspace);
        for (int i = 0, l = [string length]; i < l; i++) {
            unichar k = [string characterAtIndex: i];
            handle_key(k);
        }
    }
    - (BOOL)textFieldShouldReturn: (UITextField *)textField {
        handle_key(IPhoneKeyCode::Return);
        [textField resignFirstResponder];
        return YES;
    }
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
            depthFormat:0						// GL_DEPTH_COMPONENT16_OES
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
        iphone_main(iphone_argc, iphone_argv);

        [self performSelector:@selector(postFinishLaunch) withObject:nil afterDelay:0.0];
        return YES;
    }
    - (void)postFinishLaunch {
        lfapp_main();
    }
    - (void)applicationWillTerminate:(UIApplication *)application {}
    - (void)applicationWillResignActive:(UIApplication*)application {}
    - (void)applicationDidBecomeActive:(UIApplication*)application {}
    - (void)swapBuffers {
        [self.view swapBuffers];
        [self.window makeKeyAndVisible];
    }
    - (void)showKeyboard { [self.textField becomeFirstResponder]; }
@end

int iphone_video_swap() {
	[[LFUIApplication sharedAppDelegate] swapBuffers];
	return 0;
}

int iphone_show_keyboard() {
    [[LFUIApplication sharedAppDelegate] showKeyboard];
    return 0;
}

int iphone_input(unsigned clicks, unsigned *events) {
    SInt32 result;
    do {
        result = CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, TRUE);
    } while (result == kCFRunLoopRunHandledSource);

    if (events) *events = iphone_evcount = 0;
    iphone_evcount = 0;
    return 0;
}

int iphone_open_browser(const char *url_text) {
    NSString *url_string = [[NSString alloc] initWithUTF8String: url_text];
    NSURL *url = [NSURL URLWithString: url_string];  
    [[UIApplication sharedApplication] openURL:url];
    return 0;
}

void *iphone_load_music_asset(const char *filename) {
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

void iphone_play_music(void *handle) {
    AVAudioPlayer *audioPlayer = (AVAudioPlayer*)handle;
    [audioPlayer play];
}

void iphone_play_background_music(void *handle) {
    AVAudioPlayer *audioPlayer = (AVAudioPlayer*)handle;
    audioPlayer.numberOfLoops = -1;
    [audioPlayer play];
}

@interface LFApp : NSObject { }
    - (int) orientation;
    - (id) init;
    + (LFApp *) sharedApp;
    - (void) initNotifications;
    - (void) shutdownNotifications;
    - (void) initGestureRecognizers;
    - (void) shutdownGestureRecognizers;
@end

static int currentOrientation = 0;
@implementation LFApp
    static LFApp *gLFapp = nil;
   
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
   
    + (LFApp *) sharedApp {
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

    - (void) doubleSwipeUp: (id)sender {
        gui_gesture_swipe_up = 1;
    }

    - (void) doubleSwipeDown: (id)sender {
        gui_gesture_swipe_down = 1;
    }
	
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

static NSString *documentsDirectory = nil;

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

char *NSFMDocumentPath() {
    if (documentsDirectory == nil) {
        NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
        documentsDirectory = [[paths objectAtIndex:0] copy];
    }
    return strdup([documentsDirectory UTF8String]);
}

int NSFMreaddir(const char *path, int dirs, void *DirectoryIter, void (*DirectoryIterAdd)(void *di, const char *k, int)) {

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

#endif /* LFL_IPHONE */
