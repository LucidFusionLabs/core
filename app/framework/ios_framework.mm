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

#import <UIKit/UIKit.h>
#import <UIKit/UIScreen.h>
#import <GLKit/GLKit.h>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/EAGLDrawable.h>
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#import <QuartzCore/QuartzCore.h>
#import <AVFoundation/AVFoundation.h>

#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/ios_common.h"

namespace LFL {
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
const int Key::Insert     = -35;

static int                ios_argc = 0;
static const char* const* ios_argv = 0;
};

@implementation LFUIApplication
  {
    CGFloat scale, pinch_scale;
    CGPoint pinch_point;
    CGRect keyboard_frame;
    bool want_extra_scale;
    int current_orientation, target_fps;
    UIBackgroundTaskIdentifier bgTask;
  }
  @synthesize window, controller, view, lview, rview, text_field;

  + (LFUIApplication *)sharedAppDelegate { return (LFUIApplication *)[[UIApplication sharedApplication] delegate]; }

  - (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    _resign_textfield_on_return = YES;
    _main_wait_fh = [[NSMutableDictionary alloc] init];
    scale = [[UIScreen mainScreen] scale];
    CGRect wbounds = [[UIScreen mainScreen] bounds];

    MyAppCreate(LFL::ios_argc, LFL::ios_argv);
    self.window = [[[UIWindow alloc] initWithFrame:wbounds] autorelease];
    if (_title) {
      _title_bar = [[UINavigationBar alloc] initWithFrame:CGRectMake(0, 0, CGRectGetWidth(wbounds), 44)];
      [_title_bar setTintColor:[UIColor whiteColor]];
      UINavigationItem *item = [[UINavigationItem alloc] init];
      item.title = LFL::MakeNSString(LFL::app->focused ? LFL::app->focused->caption : LFL::app->name);
      [_title_bar setItems:@[item]];
      [window addSubview: _title_bar];

      int title_height = _title_bar.bounds.size.height;
      wbounds.origin.y    += title_height;
      wbounds.size.height -= title_height;
    }

    EAGLContext *context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
    [EAGLContext setCurrentContext:context];
    self.view = [[[GLKView alloc] initWithFrame:wbounds] autorelease];
    self.view.context = context;
    self.view.delegate = self;
    self.view.enableSetNeedsDisplay = TRUE;
    self.view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight |
      UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleLeftMargin |
      UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleTopMargin;
    [context release];

    self.controller = [[[LFViewController alloc] initWithNibName:nil bundle:nil] autorelease];
    self.controller.delegate = self.controller;
    [self.controller setView:self.view]; 
    self.controller.resumeOnDidBecomeActive = NO;
    // self.controller.wantsFullScreenLayout = YES;
    _top_controller = self.controller;

    [UIApplication sharedApplication].idleTimerDisabled = YES;
    self.window.rootViewController = self.controller;
    [self.window addSubview:self.view];

    // left touch view
    CGRect lrect = CGRectMake(0, 0, wbounds.size.width, wbounds.size.height/2);
    self.lview = [[[MyTouchView alloc] initWithFrame:lrect] autorelease];
    // self.lview.backgroundColor = [UIColor greenColor];
    // self.lview.alpha = 0.3f;
    [self.view addSubview:self.lview];
    
    // right touch view
    CGRect rrect = CGRectMake(0, wbounds.size.height/2, wbounds.size.width, wbounds.size.height/2);
    self.rview = [[[MyTouchView alloc] initWithFrame:rrect] autorelease];
    // self.rview.backgroundColor = [UIColor blueColor];
    // self.rview.alpha = 0.3f;
    [self.view addSubview:self.rview];
    [self initNotifications];

    // text view for keyboard display
    self.text_field = [[[MyTextField alloc] initWithFrame: CGRectZero] autorelease];
    self.text_field.delegate = self;
    self.text_field.text = [NSString stringWithFormat:@"default"];
    self.text_field.autocorrectionType = UITextAutocorrectionTypeNo;
    self.text_field.autocapitalizationType = UITextAutocapitalizationTypeNone;
    [self.window addSubview:self.text_field];

    [[NSFileManager defaultManager] changeCurrentDirectoryPath: [[NSBundle mainBundle] resourcePath]];
    NSLog(@"iOSMain argc=%d", LFL::app->argc);
    MyAppMain();
    INFOf("didFinishLaunchingWithOptions, views: %p, %p, %p, csf=%f", self.view, self.lview, self.rview, scale);

    [self.window makeKeyAndVisible];
    return YES;
  }

  - (void)applicationWillTerminate:(UIApplication *)application {
    [self shutdownNotifications];
    [self shutdownGestureRecognizers];
  }

  - (void)applicationWillResignActive:(UIApplication*)application {}
  - (void)applicationDidBecomeActive:(UIApplication*)application {}
  - (void)applicationWillEnterForeground:(UIApplication *)application{}

  - (void)applicationDidEnterBackground:(UIApplication *)application {
    bgTask = [application beginBackgroundTaskWithName:@"MyTask" expirationHandler:^{
      [application endBackgroundTask:bgTask];
      bgTask = UIBackgroundTaskInvalid;
    }];

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                     [application endBackgroundTask:bgTask];
                     bgTask = UIBackgroundTaskInvalid;
                   });
  }

  - (void)glkView:(GLKView *)v drawInRect:(CGRect)rect {
    LFL::Window *screen = LFL::app->focused;
    if (!_screen_width || !_screen_height || !screen) return;
    if (screen->y != _screen_y || screen->width != _screen_width || screen->height != _screen_height)
      WindowReshaped(0, _screen_y, _screen_width, _screen_height);

    LFAppFrame(true); 
  }

  - (void)updateTargetFPS: (int)fps {
    target_fps = fps;
    INFOf("updateTargetFPS: %d", target_fps);
    [self.controller setPaused:(!target_fps)];
  }

  - (CGRect)getFrame { return self.view.frame; }
  - (CGFloat)getScale { return (want_extra_scale ? scale : 1); }
  - (int)updateScale: (bool)v { want_extra_scale=v; [self updateGLKViewScale]; return v ? scale : 1; }
  - (void)downScale: (bool)v { _downscale=v; [self updateGLKViewScale]; }
  - (void)updateGLKViewScale { self.view.contentScaleFactor = _downscale ? 1 : [self getScale]; }
  - (int)updateGLKMultisample: (bool)v { 
    self.view.drawableMultisample = v ? GLKViewDrawableMultisample4X : GLKViewDrawableMultisampleNone;
    return v ? 4 : 0;
  }

  - (void)initNotifications {
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications]; 
    NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
    INFOf("init notifications %p", center);
    [center addObserver:self selector:@selector(orientationChanged:) name:@"UIDeviceOrientationDidChangeNotification" object:nil];
    [center addObserver:self selector:@selector(keyboardWillShow:)   name:@"UIKeyboardWillShowNotification"           object:nil];
    [center addObserver:self selector:@selector(keyboardWillHide:)   name:@"UIKeyboardWillHideNotification"           object:nil];
  }

  - (void)shutdownNotifications {
    INFOf("%s", "shutdown notifications");
    [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
    [[NSNotificationCenter defaultCenter] removeObserver:self]; 
  }

  - (CGRect) getKeyboardFrame { return keyboard_frame; }
  - (bool)isKeyboardFirstResponder { return [self.text_field isFirstResponder]; }
  - (void)hideKeyboard { [self.text_field resignFirstResponder]; }
  - (void)showKeyboard {
    if ([self.text_field isFirstResponder]) [self.text_field resignFirstResponder];
    [self.text_field becomeFirstResponder];
  }

  - (void)keyboardWillHide:(NSNotification *)notification {
    keyboard_frame = CGRectMake(0, 0, 0, 0);
    [self.controller updateToolbarFrame];
  }

  - (void)keyboardWillShow:(NSNotification *)notification {
    NSDictionary *userInfo = [notification userInfo];
    CGRect rect = [[userInfo objectForKey:UIKeyboardFrameEndUserInfoKey] CGRectValue];
    keyboard_frame = [self.window convertRect:rect toView:nil];
    [self.controller updateToolbarFrame];
  }

  - (BOOL)textField: (UITextField *)textField shouldChangeCharactersInRange:(NSRange)range replacementString:(NSString *)string {
    int l = [string length];
    // if (!l) [self handleKey: LFL::Key::Backspace];
    for (int i = 0, l = [string length]; i < l; i++) {
      unichar k = [string characterAtIndex: i];
      [self handleKey: k];
    }
    return YES;
  }

  - (BOOL)textFieldShouldReturn: (UITextField *)tf {
    [self handleKey: LFL::Key::Return];
    if (_resign_textfield_on_return) [tf resignFirstResponder];
    return YES;
  }

  - (void)handleKey: (int)k {
    int fired = 0;
    fired += KeyPress(k, 0, 1);
    fired += KeyPress(k, 0, 0);
    if (fired && _frame_on_keyboard_input) [self.view setNeedsDisplay];
  }

  - (int)getOrientation { return current_orientation; }
  - (void)orientationChanged: (id)sender {
    UIDeviceOrientation new_orientation = [[UIDevice currentDevice] orientation];
    bool landscape = UIDeviceOrientationIsLandscape(new_orientation);
    INFOf("notification of new orientation: %d -> %d landscape=%d", current_orientation, new_orientation, landscape);
    current_orientation = new_orientation;
  }

  - (void)initVerticalSwipeGestureRecognizers:(int)touches {
    UIWindow *win = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
    UISwipeGestureRecognizer *up = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(swipeUp:)];
    up.direction = UISwipeGestureRecognizerDirectionUp;
    up.numberOfTouchesRequired = touches;
    [win addGestureRecognizer:up];
    [up release];

    UISwipeGestureRecognizer *down = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(swipeDown:)];
    down.direction = UISwipeGestureRecognizerDirectionDown;
    down.numberOfTouchesRequired = touches;
    [win addGestureRecognizer:down];
    [down release];
  }

  - (void)initHorizontalSwipeGestureRecognizers:(int)touches {
    UIWindow *win = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
    UISwipeGestureRecognizer *left = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(swipeLeft:)];
    left.direction = UISwipeGestureRecognizerDirectionLeft;
    left.numberOfTouchesRequired = touches;
    [win addGestureRecognizer:left];
    [left release];

    UISwipeGestureRecognizer *right = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(swipeRight:)];
    right.direction = UISwipeGestureRecognizerDirectionRight;
    right.numberOfTouchesRequired = touches;
    [win addGestureRecognizer:right];
    [right release];
  }

  - (void)initPanGestureRecognizers {
    UIPanGestureRecognizer *pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(panGesture:)];
    pan.minimumNumberOfTouches = 1;
    pan.maximumNumberOfTouches = 1;
    [[LFUIApplication sharedAppDelegate].lview addGestureRecognizer:pan];
    [pan release];

    pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(panGesture:)];
    pan.minimumNumberOfTouches = 1;
    pan.maximumNumberOfTouches = 1;
    [[LFUIApplication sharedAppDelegate].rview addGestureRecognizer:pan];
    [pan release];
  }

  - (void)initTapGestureRecognizers {
    UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(tapGesture:)];
    [[LFUIApplication sharedAppDelegate].view addGestureRecognizer:tap];
    [tap release];
  }

  - (void)initPinchGestureRecognizers {
    UIWindow *win = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
    UIPinchGestureRecognizer *pinch = [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(pinchGesture:)];
    [win addGestureRecognizer:pinch];
    [pinch release];
  }

  - (void)shutdownGestureRecognizers {
    UIWindow *win = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
    NSArray *gestures = [win gestureRecognizers];
    for (int i = 0; i < [gestures count]; i++)
      [win removeGestureRecognizer:[gestures objectAtIndex:i]];
  }

  - (void)swipeUp:    (id)sender { LFL::point p(0,  1); if (LFL::app->input->MouseWheel(p, p) && _frame_on_mouse_input) [self.view setNeedsDisplay]; }
  - (void)swipeDown:  (id)sender { LFL::point p(0, -1); if (LFL::app->input->MouseWheel(p, p) && _frame_on_mouse_input) [self.view setNeedsDisplay]; }
  - (void)swipeLeft:  (id)sender { LFL::point p(-1, 0); if (LFL::app->input->MouseWheel(p, p) && _frame_on_mouse_input) [self.view setNeedsDisplay]; }
  - (void)swipeRight: (id)sender { LFL::point p( 1, 0); if (LFL::app->input->MouseWheel(p, p) && _frame_on_mouse_input) [self.view setNeedsDisplay]; }

  - (void)tapGesture: (UITapGestureRecognizer *)tapGestureRecognizer {
    UIView *v = [tapGestureRecognizer view];
    CGPoint position = [tapGestureRecognizer locationInView:v];
    int dpind = v.frame.origin.y == 0;
    if (auto s = LFL::app->focused) {
      s->gesture_tap[dpind] = 1;
      s->gesture_dpad_x[dpind] = position.x;
      s->gesture_dpad_y[dpind] = position.y;
      int fired = MouseClick(1, 1, (int)position.x, s->y + s->height - (int)position.y);
      if (fired && _frame_on_mouse_input) [self.view setNeedsDisplay];
    }
  }

  - (void)panGesture: (UIPanGestureRecognizer *)panGestureRecognizer {
    UIView *v = [panGestureRecognizer view];
    int dpind = v.frame.origin.y == 0;
    if (auto s = LFL::app->focused) {
      if (panGestureRecognizer.state == UIGestureRecognizerStateChanged) {
        // CGPoint velocity = [panGestureRecognizer translationInView:v];
        CGPoint velocity = [panGestureRecognizer velocityInView:v];
        if (fabs(velocity.x) > 15 || fabs(velocity.y) > 15) {
          s->gesture_dpad_dx[dpind] = velocity.x;
          s->gesture_dpad_dy[dpind] = velocity.y;
        }
        // CGPoint position = [panGestureRecognizer locationInView:v];
        CGPoint position = [panGestureRecognizer locationOfTouch:0 inView:v];
        if (!dpind) {
          position.x += v.frame.origin.x;
          position.y += v.frame.origin.y;
        }
        s->gesture_dpad_x[dpind] = position.x;
        s->gesture_dpad_y[dpind] = position.y;
        // INFOf("gest %f %f %f %f", position.x, position.y, velocity.x, velocity.y);
        // INFOf("origin %f %f", v.frame.origin.x, v.frame.origin.y);
      } else if (panGestureRecognizer.state == UIGestureRecognizerStateEnded) {
        s->gesture_dpad_stop[dpind] = 1;
        s->gesture_dpad_x[dpind] = 0;
        s->gesture_dpad_y[dpind] = 0;
        // CGPoint position = [panGestureRecognizer locationInView:v];
        // INFOf("gest %f %f stop", position.x, position.y);
      }
    }
  }

  - (void)pinchGesture: (UIPinchGestureRecognizer*)pinch {
    if (pinch.state == UIGestureRecognizerStateBegan) {
      pinch_scale = 1.0;
      pinch_point = [pinch locationInView: window];
    }
    CGFloat p_scale = 1.0 + (pinch_scale - pinch.scale);
    LFL::v2 p(pinch_point.x, pinch_point.y), d(p_scale, p_scale);
    int fired = LFL::app->input->MouseZoom(p, d);
    if (fired && _frame_on_mouse_input) [self.view setNeedsDisplay];
    pinch_scale = p_scale;
  }

  - (void)addMainWaitSocket:(int)fd callback:(std::function<bool()>)cb {
    [_main_wait_fh
      setObject: [[[ObjcFileHandleCallback alloc] initWithCB:move(cb) forWindow:self fileDescriptor:fd] autorelease]
      forKey:    [NSNumber numberWithInt: fd]];
  }

  - (void)delMainWaitSocket: (int)fd {
    [_main_wait_fh removeObjectForKey: [NSNumber numberWithInt: fd]];
  }

  - (void)objcWindowSelect {}
  - (void)objcWindowFrame { [self.view setNeedsDisplay]; }
@end

@implementation LFViewController
  {
    LFUIApplication *uiapp;
    bool overlap_keyboard;
  }

  - (void)viewWillAppear:(BOOL)animated { 
    [super viewWillAppear:animated];
    [self setPaused:YES];
    uiapp = [LFUIApplication sharedAppDelegate];
  }

  - (void)viewDidLoad {
    [super viewDidLoad];
  }

  - (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
  }

  - (void)viewWillLayoutSubviews {
    [super viewWillLayoutSubviews];
  }
  
  - (void)viewDidLayoutSubviews {
    CGFloat s = [uiapp getScale];
    CGRect bounds = self.view.bounds;
    int y = overlap_keyboard ? 0 : (s * [self getKeyboardToolbarFrame].size.height);
    uiapp.screen_y      = y;
    uiapp.screen_width  = s * bounds.size.width;
    uiapp.screen_height = s * bounds.size.height - y;
    [uiapp.view setNeedsDisplay];
    [super viewDidLayoutSubviews];
  }

  - (void)glkViewControllerUpdate:(GLKViewController *)controller {}
  - (BOOL)shouldAutorotate { return _autorotate; }
  - (void)setOverlapKeyboard:(bool)v { overlap_keyboard = v; }

  - (CGRect)getKeyboardToolbarFrame {
    CGRect kbd = [[LFUIApplication sharedAppDelegate] getKeyboardFrame];
    return CGRectMake(kbd.origin.x, kbd.origin.y, kbd.size.width, kbd.size.height + [IOSToolbar getBottomHeight]);
  }

  - (void)updateToolbarFrame {
    [IOSToolbar updateFrame];
    [self.view setNeedsLayout];
  }
@end

@implementation MyTouchView
  {
    LFUIApplication *uiapp;
  }

  - (id)initWithFrame:(CGRect)aRect {
    if (!(self = [super initWithFrame:aRect])) return self;
    uiapp = [LFUIApplication sharedAppDelegate];
    return self;
  }

  - (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
    UITouch *touch = [touches anyObject];
    UIView *v = [touch view];
    CGPoint position = [touch locationInView:v];
    int dpind = v.frame.origin.y == 0, scale = [uiapp getScale];
    if (!dpind) position = CGPointMake(scale * (position.x + v.frame.origin.x), scale * (position.y + v.frame.origin.y));
    else        position = CGPointMake(scale * position.x, scale * position.y);
    if (auto s = LFL::app->focused) {
      s->gesture_dpad_x[dpind] = position.x;
      s->gesture_dpad_y[dpind] = position.y;
      int fired = LFL::app->input->MouseClick(1, 1, LFL::point(position.x, s->y + s->height - (int)position.y));
      if (fired && uiapp.frame_on_mouse_input) [self.superview setNeedsDisplay];
    }
  }

  - (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {}
  - (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
    UITouch *touch = [touches anyObject];
    UIView *v = [touch view];
    CGPoint position = [touch locationInView:v];
    int dpind = v.frame.origin.y == 0, scale = [uiapp getScale];
    if (!dpind) position = CGPointMake(scale * (position.x + v.frame.origin.x), scale * (position.y + v.frame.origin.y));
    else        position = CGPointMake(scale * position.x, scale * position.y);
    if (auto s = LFL::app->focused) {
      s->gesture_dpad_stop[dpind] = 1;
      s->gesture_dpad_x[dpind] = 0;
      s->gesture_dpad_y[dpind] = 0;
      int fired = MouseClick(1, 0, (int)position.x, s->y + s->height - (int)position.y);
      if (fired && uiapp.frame_on_mouse_input) [self.superview setNeedsDisplay];
    }
  }

  - (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
    UITouch *touch = [touches anyObject];
    UIView *v = [touch view];
    CGPoint position = [touch locationInView:v];
    int dpind = v.frame.origin.y == 0, scale = [uiapp getScale];
    if (!dpind) position = CGPointMake(scale * (position.x + v.frame.origin.x), scale * (position.y + v.frame.origin.y));
    else        position = CGPointMake(scale * position.x, scale * position.y);
    if (auto s = LFL::app->focused) {
      s->gesture_dpad_x[dpind] = position.x;
      s->gesture_dpad_y[dpind] = position.y;
    }
  }
@end

@implementation MyTextField
  - (void)deleteBackward {
    [super deleteBackward];
    [[LFUIApplication sharedAppDelegate] handleKey: LFL::Key::Backspace];
  }
@end

namespace LFL {
struct iOSFrameworkModule : public Module {
  int Init() {
    INFO("iOSFrameworkModule::Init()");
    if (auto s = app->focused) {
      CHECK(!s->id.v);
      s->id = LFL::MakeTyped([[LFUIApplication sharedAppDelegate] view]);
      CHECK(s->id.v);
      app->windows[s->id.v] = s;
    }
    LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
    CGFloat scale = [uiapp getScale];
    CGRect rect = [uiapp getFrame];
    int w = rect.size.width * scale, h = rect.size.height * scale;
    if (auto s = app->focused) Assign(&s->width, &s->height, w, h);
    INFOf("iOSFrameworkModule::Init %d %d vs %d %d", w, h, [uiapp view].drawableWidth, [uiapp view].drawableHeight);
    return 0;
  }
};

struct iOSAssetLoader : public SimpleAssetLoader {
  virtual void UnloadAudioFile(void *h) {}
  virtual void *LoadAudioFile(File*) { return 0; }
  virtual void *LoadAudioFileNamed(const string &asset_fn) {
    NSError *error;
    NSString *fn = [NSString stringWithCString:Asset::FileName(asset_fn).c_str() encoding:NSASCIIStringEncoding];
    NSURL *url = [NSURL fileURLWithPath: fn];
    AVAudioPlayer *audio_player = [[AVAudioPlayer alloc] initWithContentsOfURL:url error:&error];
    if (audio_player == nil) ERRORf("iOS Asset Load %s: %s", [fn UTF8String], [[error description] UTF8String]);
    return audio_player;
  }

  virtual void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) { a->handle = handle; }
  virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }
};

void Application::MakeCurrentWindow(Window *W) {}
void Application::CloseWindow(Window *W) {}

void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}

string Application::GetClipboardText() { return [[UIPasteboard generalPasteboard].string UTF8String]; }
void Application::SetClipboardText(const string &s) {
  if (auto ns = MakeNSString(s)) [UIPasteboard generalPasteboard].string = ns;
}

void Application::OpenTouchKeyboard() { [[LFUIApplication sharedAppDelegate] showKeyboard]; }
void Application::CloseTouchKeyboard() { [[LFUIApplication sharedAppDelegate] hideKeyboard]; }
void Application::CloseTouchKeyboardAfterReturn(bool v) { [LFUIApplication sharedAppDelegate].resign_textfield_on_return = v; }
void Application::SetTouchKeyboardTiled(bool v) { [[LFUIApplication sharedAppDelegate].controller setOverlapKeyboard: !v]; }
void Application::ToggleTouchKeyboard() {
  if ([[LFUIApplication sharedAppDelegate] isKeyboardFirstResponder]) CloseTouchKeyboard();
  else OpenTouchKeyboard();
}

void Application::SetAutoRotateOrientation(bool v) { [LFUIApplication sharedAppDelegate].controller.autorotate = v; }
int Application::SetMultisample(bool v) { return [[LFUIApplication sharedAppDelegate] updateGLKMultisample:v]; }
int Application::SetExtraScale(bool v) { return [[LFUIApplication sharedAppDelegate] updateScale:v]; }
void Application::SetDownScale(bool v) { [[LFUIApplication sharedAppDelegate] downScale:v]; }
void Application::SetTitleBar(bool v) { [LFUIApplication sharedAppDelegate].title = true; }
void Application::SetKeepScreenOn(bool v) {}
void Application::SetVerticalSwipeRecognizer(int touches) { [[LFUIApplication sharedAppDelegate] initVerticalSwipeGestureRecognizers: touches]; }
void Application::SetHorizontalSwipeRecognizer(int touches) { [[LFUIApplication sharedAppDelegate] initHorizontalSwipeGestureRecognizers: touches]; }
void Application::SetPanRecognizer(bool enabled) { [[LFUIApplication sharedAppDelegate] initPanGestureRecognizers]; }
void Application::SetPinchRecognizer(bool enabled) { [[LFUIApplication sharedAppDelegate] initPinchGestureRecognizers]; }

void Window::SetResizeIncrements(float x, float y) {}
void Window::SetTransparency(float v) {}
bool Window::Reshape(int w, int h) { return false; }

void Window::SetCaption(const string &v) {
  auto title_bar = [LFUIApplication sharedAppDelegate].title_bar;
  if (!title_bar) return;
  UINavigationItem *newItem = title_bar.items[0];
  newItem.title = MakeNSString(v);
}

bool Video::CreateWindow(Window *w) { return false; }
void Video::StartWindow(Window *w) {
  [[LFUIApplication sharedAppDelegate] updateTargetFPS: w->target_fps];
}

int Video::Swap() {
  app->focused->gd->Flush();
  app->focused->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

bool FrameScheduler::DoMainWait() { return false; }
void FrameScheduler::Setup() { rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = run_main_loop = 0; }
void FrameScheduler::Wakeup(Window *w) {
  dispatch_async(dispatch_get_main_queue(), ^{ [GetTyped<GLKView*>(w->id) setNeedsDisplay]; });
}

bool FrameScheduler::WakeupIn(Window*, Time interval, bool force) { return false; }
void FrameScheduler::ClearWakeupIn(Window*) {}

void FrameScheduler::UpdateWindowTargetFPS(Window *w) {
  [[LFUIApplication sharedAppDelegate] updateTargetFPS: w->target_fps];
}

void FrameScheduler::AddMainWaitMouse(Window*) {
  [LFUIApplication sharedAppDelegate].frame_on_mouse_input = YES;
}

void FrameScheduler::DelMainWaitMouse(Window*) {
  [LFUIApplication sharedAppDelegate].frame_on_mouse_input = NO;
}

void FrameScheduler::AddMainWaitKeyboard(Window*) {
  [LFUIApplication sharedAppDelegate].frame_on_keyboard_input = YES;
}

void FrameScheduler::DelMainWaitKeyboard(Window*) {
 [LFUIApplication sharedAppDelegate].frame_on_keyboard_input = NO;
}

void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()> cb) {
  if (!wait_forever_thread) {
    CHECK_EQ(SocketSet::READABLE, flag);
    [[LFUIApplication sharedAppDelegate] addMainWaitSocket:fd callback:move(cb)];
  }
}

void FrameScheduler::DelMainWaitSocket(Window *w, Socket fd) {
  [[LFUIApplication sharedAppDelegate] delMainWaitSocket: fd];
}

unique_ptr<Module> CreateFrameworkModule() { return make_unique<iOSFrameworkModule>(); }
unique_ptr<AssetLoaderInterface> CreateAssetLoader() { return make_unique<iOSAssetLoader>(); }

extern "C" int main(int ac, const char* const* av) {
  ios_argc = ac;
  ios_argv = av;
  NSLog(@"%@", @"lfl_app_ios_framework_main");
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  int ret = UIApplicationMain(ac, const_cast<char**>(av), nil, @"LFUIApplication");
  [pool release];
  return ret;
}

}; // namespace LFL
