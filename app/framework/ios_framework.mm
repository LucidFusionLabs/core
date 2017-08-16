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
#include "core/imports/appirater/Appirater.h"

#include <netinet/in.h>

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

const int Texture::updatesystemimage_pf = Pixel::RGB24;

static int                ios_argc = 0;
static const char* const* ios_argv = 0;
};

static void ReachabilityCallback(SCNetworkReachabilityRef target, SCNetworkReachabilityFlags flags, void *info) {
  bool wifi = false;
  if ((flags & kSCNetworkReachabilityFlagsReachable) == 0) wifi = false;
  else {
    if ((flags & kSCNetworkReachabilityFlagsConnectionRequired) == 0) wifi = true;
    if ((((flags & kSCNetworkReachabilityFlagsConnectionOnDemand) != 0) ||
         (flags & kSCNetworkReachabilityFlagsConnectionOnTraffic) != 0) &&
         (flags & kSCNetworkReachabilityFlagsInterventionRequired) == 0) wifi = true;
    if ((flags & kSCNetworkReachabilityFlagsIsWWAN) == kSCNetworkReachabilityFlagsIsWWAN) wifi = false;
  }
  [LFUIApplication sharedAppDelegate].wifi = wifi;
}

@implementation LFUIWindow
@end

@implementation LFUIApplication
  {
    bool want_extra_scale, has_become_active;
    LFL::Callback bg_task_cb;
    UIBackgroundTaskIdentifier bg_task;
    SCNetworkReachabilityFlags reachability_flags;
  }

  + (LFUIApplication *)sharedAppDelegate { return (LFUIApplication *)[[UIApplication sharedApplication] delegate]; }

  + (CGSize)currentWindowSize {
    return [LFUIApplication sizeInOrientation:[UIApplication sharedApplication].statusBarOrientation];
  }

  + (CGSize)sizeInOrientation:(UIInterfaceOrientation)orientation {
    UIApplication *application = [UIApplication sharedApplication];
    CGSize size = [UIScreen mainScreen].bounds.size;
    if (UIInterfaceOrientationIsLandscape(orientation))
      size = CGSizeMake(size.height, size.width);
    // if (application.statusBarHidden == NO)
    //  size.height -= MIN(application.statusBarFrame.size.width, application.statusBarFrame.size.height);
    return size;
  }

  - (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    _main_wait_fh = [[NSMutableDictionary alloc] init];
    _scale = [[UIScreen mainScreen] scale];
    CGRect wbounds = [[UIScreen mainScreen] bounds];

    MyAppCreate(LFL::ios_argc, LFL::ios_argv);
    _window = [[LFUIWindow alloc] initWithFrame:wbounds];

    if (_show_title) {
      _title_bar = [[UINavigationBar alloc] initWithFrame:CGRectMake(0, 0, CGRectGetWidth(wbounds), 44)];
      [_title_bar setTintColor:[UIColor whiteColor]];
      UINavigationItem *item = [[UINavigationItem alloc] init];
      item.title = LFL::MakeNSString(LFL::app->focused ? LFL::app->focused->caption : LFL::app->name);
      [_title_bar setItems:@[item]];
      [_window addSubview: _title_bar];

      int title_height = _title_bar.bounds.size.height;
      wbounds.origin.y    += title_height;
      wbounds.size.height -= title_height;
    }

    EAGLContext *context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
    [EAGLContext setCurrentContext:context];

    _glk_controller = [[LFGLKViewController alloc] initWithNibName:nil bundle:nil];
    _glk_controller.delegate = _glk_controller;
    _glk_controller.resumeOnDidBecomeActive = NO;
    // _glk_controller.wantsFullScreenLayout = YES;

    _glk_view = [[LFGLKView alloc] initWithFrame:wbounds];
    _glk_view.context = context;
    _glk_view.delegate = _glk_controller;
    _glk_view.enableSetNeedsDisplay = TRUE;
    _glk_view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight |
      UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleLeftMargin |
      UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleTopMargin;
    _glk_view.resign_textfield_on_return = YES;
    [context release];
    _glk_controller.view = _glk_view;

    _controller = [[LFViewController alloc] initWithNibName:nil bundle:nil];
    _top_controller = _controller;
    _root_controller = _controller;
    _window.rootViewController = _controller;
    [_controller addChildViewController: _glk_controller];
    // _glk_controller.view.frame = [_controller frameForContentController];
    [_controller.view addSubview:_glk_controller.view];
    [_glk_controller didMoveToParentViewController: _controller];
    [_controller initNotifications];

    // left touch view
    CGRect lrect = CGRectMake(0, 0, wbounds.size.width/2, wbounds.size.height);
    _lview = [[MyTouchView alloc] initWithFrame:lrect];
    _lview.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight | UIViewAutoresizingFlexibleRightMargin;
    // _lview.backgroundColor = [UIColor greenColor];
    // _lview.alpha = 0.3f;
    [_glk_view addSubview:_lview];

    // right touch view
    CGRect rrect = CGRectMake(wbounds.size.width/2, 0, wbounds.size.width/2, wbounds.size.height);
    _rview = [[MyTouchView alloc] initWithFrame:rrect];
    _rview.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight | UIViewAutoresizingFlexibleLeftMargin;
    // _rview.backgroundColor = [UIColor blueColor];
    // _rview.alpha = 0.3f;
    [_glk_view addSubview:_rview];

    // monitor reachability
    struct sockaddr_in zeroAddress;
    bzero(&zeroAddress, sizeof(zeroAddress));
    zeroAddress.sin_len = sizeof(zeroAddress);
    zeroAddress.sin_family = AF_INET;
    _reachability = SCNetworkReachabilityCreateWithAddress(kCFAllocatorDefault,
                                                           (const struct sockaddr *)&zeroAddress);
    SCNetworkReachabilityContext reachable_context = {0, self, NULL, NULL, NULL};
    if (!SCNetworkReachabilitySetCallback(_reachability, ReachabilityCallback, &reachable_context) ||
        !SCNetworkReachabilityScheduleWithRunLoop(_reachability, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode) ||
        !SCNetworkReachabilityGetFlags(_reachability, &reachability_flags))
        { NSLog(@"SCNetworkReachabilityScheduleWithRunLoop failed"); CFRelease(_reachability); _reachability=nil; }
    else ReachabilityCallback(_reachability, reachability_flags, reachable_context.info);

    [[NSFileManager defaultManager] changeCurrentDirectoryPath: [[NSBundle mainBundle] resourcePath]];
    NSLog(@"iOSMain argc=%d", LFL::app->argc);
    MyAppMain();

    INFOf("didFinishLaunchingWithOptions, views: %p, %p, %p, csf=%f", _glk_view, _lview, _rview, _scale);
    [_window makeKeyAndVisible];
    return YES;
  }

  - (BOOL)application:(UIApplication *)application openURL:(NSURL *)url sourceApplication:(NSString *)sourceApplication annotation:(id)annotation {
    LFL::string urltext = LFL::GetNSString([url absoluteString]);
    INFO("openURL ", urltext);
    if (LFL::app->open_url_cb) LFL::app->open_url_cb(urltext);
    return YES;
  }

  - (void)applicationWillTerminate:(UIApplication *)application {
    [_controller shutdownNotifications];
    [_controller shutdownGestureRecognizers];
  }

  - (void)applicationWillResignActive:(UIApplication*)application {
  }

  - (void)applicationDidBecomeActive:(UIApplication*)application {
    if (!has_become_active && (has_become_active = true)) {}
  }

  - (void)applicationDidEnterBackground:(UIApplication *)application {
    LFL::app->suspended = true;
    if (LFL::app->focused->unfocused_cb) LFL::app->focused->unfocused_cb();
    while (LFL::app->message_queue.HandleMessages()) {}
    LFL::app->ResetGL(LFL::ResetGLFlag::Delete);
    glFinish();

    if (bg_task_cb) {
      bg_task = [application beginBackgroundTaskWithName:@"MyTask" expirationHandler:^{
        [application endBackgroundTask:bg_task];
        bg_task = UIBackgroundTaskInvalid;
      }];
      dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                       if (bg_task_cb) bg_task_cb();
                       [application endBackgroundTask:bg_task];
                       bg_task = UIBackgroundTaskInvalid;
                     });
    }
  }

  - (void)applicationWillEnterForeground:(UIApplication *)application{
    LFL::app->suspended = false;
    if (bg_task != UIBackgroundTaskInvalid) {
      [application endBackgroundTask:bg_task];
      bg_task = UIBackgroundTaskInvalid;
    }
    LFL::app->focused->gd->default_framebuffer = 0;
    _glk_controller.needs_gl_reset = true;
    if (LFL::app->focused->focused_cb) LFL::app->focused->focused_cb();
    [_glk_view setNeedsDisplay];
  }

  - (void)application:(UIApplication *)application performFetchWithCompletionHandler:(nonnull void (^)(UIBackgroundFetchResult))completionHandler {
    completionHandler(UIBackgroundFetchResultNoData);
  }

  - (CGRect)getFrame { return _glk_view.frame; }
  - (CGFloat)getScale { return (want_extra_scale ? _scale : 1); }
  - (bool)isKeyboardFirstResponder { return [_glk_view isFirstResponder]; }

  - updateTargetFPS: (int)fps {
    LFL::FLAGS_target_fps = fps;
    INFOf("updateTargetFPS: %d", LFL::FLAGS_target_fps);
    [_glk_controller setPaused:(!LFL::FLAGS_target_fps)];
  }

  - (int)updateScale: (bool)v { want_extra_scale=v; [self updateGLKViewScale]; return v ? _scale : 1; }
  - (void)downScale: (bool)v { _downscale=v; [self updateGLKViewScale]; }
  - (void)updateGLKViewScale { _glk_view.contentScaleFactor = _downscale ? 1 : [self getScale]; }
  - (int)updateGLKMultisample: (bool)v { 
    _glk_view.drawableMultisample = v ? GLKViewDrawableMultisample4X : GLKViewDrawableMultisampleNone;
    return v ? 4 : 0;
  }

  - (void)hideKeyboard {
    if ([_glk_view isFirstResponder])
      [_glk_view performSelector:@selector(resignFirstResponder) withObject:nil afterDelay:0];
    [_controller performSelector:@selector(becomeFirstResponder) withObject:nil afterDelay:0];
  }

  - (void)showKeyboard {
    [_glk_view performSelector:@selector(becomeFirstResponder) withObject:nil afterDelay:0];
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
  - (void)objcWindowFrame { [_glk_view setNeedsDisplay]; }
  - (void)setExtendedBackgroundTaskCB:(LFL::Callback)cb { bg_task_cb = move(cb); }

  - (UIViewController*)findTopViewController {
    return [LFUIApplication findTopViewControllerWithRootViewController: _top_controller];
  }

  + (UIViewController*)findTopViewControllerWithRootViewController:(UIViewController*)rootViewController {
    if ([rootViewController isKindOfClass:[UITabBarController class]]) {
      UITabBarController* tabBarController = (UITabBarController*)rootViewController;
      return [LFUIApplication findTopViewControllerWithRootViewController:tabBarController.selectedViewController];
    } else if ([rootViewController isKindOfClass:[UINavigationController class]]) {
      UINavigationController* navigationController = (UINavigationController*)rootViewController;
      return [LFUIApplication findTopViewControllerWithRootViewController:navigationController.visibleViewController];
    } else if (rootViewController.presentedViewController) {
      UIViewController* presentedViewController = rootViewController.presentedViewController;
      return [LFUIApplication findTopViewControllerWithRootViewController:presentedViewController];
    } else if (rootViewController.childViewControllers.count > 0) {
      return [LFUIApplication  findTopViewControllerWithRootViewController:rootViewController.childViewControllers.lastObject];
    } else {
      return rootViewController;
    }
  }
@end

@implementation LFGLKView
  - (BOOL)hasText { return YES; }
  - (BOOL)canBecomeFirstResponder { return YES; }
  - (UITextAutocorrectionType)autocorrectionType { return UITextAutocorrectionTypeNo; }
  - (void)deleteBackward { [self handleKey: LFL::Key::Backspace]; }
  - (void)insertText:(NSString*)string {
    for (int i = 0, l = [string length]; i < l; i++) {
      unichar k = [string characterAtIndex: i];
      [self handleKey: k];
      if (k == '\n' && _resign_textfield_on_return) [self resignFirstResponder];
    }
  }

  - (void)handleKey: (int)k {
    int fired = 0;
    fired += LFL::app->input->KeyPress(k, 0, 1);
    fired += LFL::app->input->KeyPress(k, 0, 0);
    if (fired && _frame_on_keyboard_input) [self setNeedsDisplay];
  }

  - (BOOL)canPerformAction:(SEL)action withSender:(id)sender {
    return action == @selector(copy:) || action == @selector(paste:) || action == @selector(toggleKeyboard:);
  }

  - (void)copy:(id)sender { if (_copy_cb) _copy_cb(); }
  - (void)paste:(id)sender { [self insertText: [UIPasteboard generalPasteboard].string]; }
  - (void)toggleKeyboard:(id)sender { LFL::app->ToggleTouchKeyboard(); }
@end

@implementation LFGLKViewController
  {
    LFUIApplication *uiapp;
  }

  - (void)viewWillAppear:(BOOL)animated { 
    [super viewWillAppear:animated];
    if (!LFL::FLAGS_target_fps) [self setPaused:YES];
    uiapp = [LFUIApplication sharedAppDelegate];
  }
  
  - (void)viewDidLayoutSubviews {
    CGFloat s = [uiapp getScale];
    CGRect bounds = self.view.bounds;
    bool overlap_keyboard = false;
    int kb_h = [uiapp.controller getKeyboardToolbarFrame].size.height, y = overlap_keyboard ? 0 : (s * kb_h);
    uiapp.gl_y = y;
    uiapp.gl_w = s * bounds.size.width;
    uiapp.gl_h = s * bounds.size.height - y;
    uiapp.screen_w = s * bounds.size.width;
    uiapp.screen_h = s * bounds.size.height;
    // INFO("LFGLKViewController viewDidLayoutSubviews kb_h=", kb_h, " y=", y, " h=", uiapp.gl_h, " bounds.h=", bounds.size.height, " scale=", float(s));
    [self.view setNeedsDisplay];
    [super viewDidLayoutSubviews];
  }

  - (void)glkViewControllerUpdate:(GLKViewController *)controller {}

  - (void)glkView:(GLKView *)v drawInRect:(CGRect)rect {
    LFL::Window *screen = LFL::app->focused;
    if (!screen->gd->default_framebuffer)
      screen->gd->GetIntegerv(LFL::GraphicsDevice::FramebufferBinding, &screen->gd->default_framebuffer);
    if (_needs_gl_reset && !(_needs_gl_reset = false)) {
      LFL::app->ResetGL(LFL::ResetGLFlag::Delete | LFL::ResetGLFlag::Reload);
      LFL::FrameBuffer(screen->gd).Release();
    }
    bool draw_frame = (uiapp.top_controller == uiapp.root_controller || uiapp.overlay_top_controller) &&
      !uiapp.frame_disabled && uiapp.gl_w && uiapp.gl_h && screen;
    if (draw_frame && (screen->gl_y != uiapp.gl_y || screen->gl_w != uiapp.gl_w || screen->gl_h != uiapp.gl_h))
      LFL::app->focused->Reshaped(LFL::point(uiapp.screen_w, uiapp.screen_h),
                                  LFL::Box(0, uiapp.gl_y, uiapp.gl_w, uiapp.gl_h));
    LFL::app->EventDrivenFrame(true, draw_frame);
  }
@end

@implementation LFViewController
  {
    LFUIApplication *uiapp;
    bool overlap_keyboard, interface_rotation, status_bar_hidden;
    UIInterfaceOrientation current_orientation, next_orientation;
    CGRect keyboard_frame;
    CGFloat pinch_scale;
    CGPoint pinch_point;
  }

  - (BOOL)canBecomeFirstResponder { return YES; }
  - (BOOL)prefersStatusBarHidden { return status_bar_hidden; }
  // - (UIView*)inputAccessoryView { return _input_accessory_toolbar; }

  - (CGRect)getKeyboardFrame { return keyboard_frame; }
  - (CGRect)getKeyboardToolbarFrame {
    int bottom_height = !_showing_keyboard ? [IOSToolbar getBottomHeight] : 0;
    CGRect kbd = [self getKeyboardFrame];
    CGRect kbdtb = CGRectMake(kbd.origin.x, kbd.origin.y, kbd.size.width, kbd.size.height + bottom_height);
    // INFO("getKeyboardToolbarFrame: ", LFL::GetCGRect(kbdtb).DebugString(), " keyboardFrame: ", LFL::GetCGRect(kbd).DebugString());
    return kbdtb;
  }

  - (void)setOverlapKeyboard:(bool)v { overlap_keyboard = v; }
  - (void)showStatusBar:(BOOL)v {
    if (status_bar_hidden == !v) return;
    status_bar_hidden = !v;
    [self setNeedsStatusBarAppearanceUpdate]; 
  }

  - (void)viewWillAppear:(BOOL)animated { 
    [super viewWillAppear:animated];
    uiapp = [LFUIApplication sharedAppDelegate];
  }

  - (void)viewDidLoad {
    [super viewDidLoad];
  }

  - (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    _ready = true;
    current_orientation = next_orientation = [UIApplication sharedApplication].statusBarOrientation;
    if (uiapp.top_controller != uiapp.root_controller &&
        uiapp.root_controller.presentedViewController != uiapp.top_controller)
      [uiapp.root_controller presentViewController:uiapp.top_controller animated:YES completion:nil];
  }

  - (void)viewWillLayoutSubviews {
    [super viewWillLayoutSubviews];
  }

  - (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
  }

  - (void)willRotateToInterfaceOrientation:(UIInterfaceOrientation)toInterfaceOrientation duration:(NSTimeInterval)duration {
    INFO("willRotateToInterfaceOrientation: orientation: ", int(toInterfaceOrientation), " ", LFL::GetCGRect(self.view.frame).DebugString());
    interface_rotation = true;
    next_orientation = toInterfaceOrientation;
  }

  - (void)didRotateFromInterfaceOrientation:(UIInterfaceOrientation)fromInterfaceOrientation {
                                  CGRect frame = self.view.frame;
    // LFL::string view_hierarchy = LFL::GetNSString([uiapp.window recursiveDescription] );
    INFO("didRotateFromInterfaceOrientation: orientation: ", int(fromInterfaceOrientation), " ", LFL::GetCGRect(self.view.frame).DebugString());
    interface_rotation = false;
    current_orientation = next_orientation;
  }

  - (void)initNotifications {
    [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications]; 
    NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
    INFOf("init notifications %p", center);
    [center addObserver:self selector:@selector(keyboardWillChangeFrame:) name:@"UIKeyboardWillChangeFrameNotification"     object:nil];
  }

  - (void)shutdownNotifications {
    INFOf("%s", "shutdown notifications");
    [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
    [[NSNotificationCenter defaultCenter] removeObserver:self]; 
  }

  - (void)keyboardWillChangeFrame:(NSNotification *)notification {
    UIWindow            *window   = [[[UIApplication sharedApplication] windows] firstObject];
    NSTimeInterval       duration = [notification.userInfo[UIKeyboardAnimationDurationUserInfoKey] doubleValue];
    UIViewAnimationCurve curve    = UIViewAnimationCurve([notification.userInfo[UIKeyboardAnimationCurveUserInfoKey] integerValue]);
    CGRect               endFrame = [notification.userInfo[UIKeyboardFrameEndUserInfoKey] CGRectValue];

    bool show_or_hide = CGRectContainsRect(window.frame, endFrame); // CGRectIntersectsRect(window.frame, endFrame);
    if (show_or_hide) [self kbWillShowOrHide:YES withRect:[self.view convertRect:endFrame fromView:nil] andDuration:duration andCurve:curve];
    else              [self kbWillShowOrHide:NO  withRect:CGRectMake(0,0,0,0)                                 andDuration:duration andCurve:curve];
  }

  - (void)kbWillShowOrHide:(BOOL)show_or_hide withRect:(CGRect)rect andDuration:(NSTimeInterval)interval andCurve:(UIViewAnimationCurve)curve {
    if (/*interface_rotation || */ CGRectEqualToRect(rect, keyboard_frame)) return;
    if (show_or_hide && !rect.size.height) return;
    if (_input_accessory_toolbar) _input_accessory_toolbar.hidden = show_or_hide;
    _showing_keyboard = show_or_hide;
    keyboard_frame = rect;
    INFO("kbWillShowOrHide: ", bool(show_or_hide), " ", LFL::GetCGRect(keyboard_frame).DebugString(), " scale=", uiapp.scale);
    bool presented_controller = uiapp.top_controller != uiapp.root_controller;
    if (presented_controller) [uiapp.top_controller.view setNeedsLayout];
    [uiapp.glk_view setNeedsLayout];
#if 0
    [UIView animateWithDuration:interval animations:^{
      if (presented_controller) [uiapp.top_controller.view layoutIfNeeded];
      [uiapp.glk_view layoutIfNeeded];
    }];
#endif
  }

  - (void)shutdownGestureRecognizers {
    UIWindow *win = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
    NSArray *gestures = [win gestureRecognizers];
    for (int i = 0; i < [gestures count]; i++)
      [win removeGestureRecognizer:[gestures objectAtIndex:i]];
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
    [[LFUIApplication sharedAppDelegate].glk_view addGestureRecognizer:tap];
    [tap release];
  }

  - (void)initPinchGestureRecognizers {
    UIWindow *win = [[[UIApplication sharedApplication] windows] objectAtIndex:0];
    UIPinchGestureRecognizer *pinch = [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(pinchGesture:)];
    [[LFUIApplication sharedAppDelegate].glk_view addGestureRecognizer:pinch];
    [pinch release];
  }

  - (void)swipeUp:    (UISwipeGestureRecognizer*)sender { [self swipe:LFL::point(0,  1) withSender:sender]; }
  - (void)swipeDown:  (UISwipeGestureRecognizer*)sender { [self swipe:LFL::point(0, -1) withSender:sender]; }
  - (void)swipeLeft:  (UISwipeGestureRecognizer*)sender { [self swipe:LFL::point(-1, 0) withSender:sender]; }
  - (void)swipeRight: (UISwipeGestureRecognizer*)sender { [self swipe:LFL::point( 1, 0) withSender:sender]; }
  - (void)swipe:(LFL::point)p withSender:(UISwipeGestureRecognizer*)sender {
    if (LFL::app->input->MouseSwipe(p, p) && uiapp.frame_on_mouse_input) [uiapp.glk_view setNeedsDisplay];
  }

  - (void)tapGesture: (UITapGestureRecognizer *)tapGestureRecognizer {
    UIView *v = [tapGestureRecognizer view];
    CGPoint position = [tapGestureRecognizer locationInView:v];
    int dpind = v.frame.origin.x == 0;
    if (auto s = LFL::app->focused) {
#if 0
      s->gesture_tap[dpind] = 1;
      s->gesture_dpad_x[dpind] = position.x;
      s->gesture_dpad_y[dpind] = position.y;
#endif
      int fired = LFL::app->input->MouseClick(1, 1, LFL::point(position.x, s->gl_y + s->gl_h - position.y));
      if (fired && uiapp.frame_on_mouse_input) [uiapp.glk_view setNeedsDisplay];
    }
  }

  - (void)panGesture: (UIPanGestureRecognizer *)panGestureRecognizer {
    if (panGestureRecognizer.state == UIGestureRecognizerStateEnded ||
        panGestureRecognizer.state == UIGestureRecognizerStateFailed ||
        panGestureRecognizer.state == UIGestureRecognizerStateCancelled) {
      if (auto s = LFL::app->focused) { 
        int fired = LFL::app->input->MouseClick(1, 0, s->mouse);
        if (fired && uiapp.frame_on_mouse_input) [uiapp.glk_view setNeedsDisplay];
      }
      return;
    }
    if (![panGestureRecognizer numberOfTouches]) return;
    UIView *v = [panGestureRecognizer view];
    CGPoint position = [panGestureRecognizer locationOfTouch:0 inView:v];
    CGPoint velocity = [panGestureRecognizer velocityInView:v];
    int dpind = v.frame.origin.x == 0;
    if (!dpind) position = CGPointMake(uiapp.scale * (position.x + v.frame.origin.x), uiapp.scale * (position.y + v.frame.origin.y));
    else        position = CGPointMake(uiapp.scale * position.x, uiapp.scale * position.y);
    if (auto s = LFL::app->focused) { 
      LFL::point pos = LFL::Floor(LFL::GetCGPoint(position));
      pos.y = s->gl_y + s->gl_h - pos.y;
      int fired = panGestureRecognizer.state == UIGestureRecognizerStateBegan ?
        LFL::app->input->MouseClick(1, 1, pos) :
        LFL::app->input->MouseMove(pos, pos - LFL::app->focused->mouse);
      if (fired && uiapp.frame_on_mouse_input) [uiapp.glk_view setNeedsDisplay];
    }
  }
    #if 0
    if (auto s = LFL::app->focused) {
      if (panGestureRecognizer.state == UIGestureRecognizerStateChanged) {
        // CGPoint velocity = [panGestureRecognizer translationInView:v];
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
  #endif

  - (void)pinchGesture: (UIPinchGestureRecognizer*)pinch {
    auto s = LFL::app->focused;
    if (!s || [pinch numberOfTouches] < 2) return;
    bool begin = pinch.state == UIGestureRecognizerStateBegan;
    if (begin) {
      UIView *v = [pinch view];
      CGPoint point0 = [pinch locationOfTouch:0 inView:v];
      CGPoint point1 = [pinch locationOfTouch:1 inView:v];
      point0.y = s->gl_y + s->gl_h - point0.y;
      point1.y = s->gl_y + s->gl_h - point1.y;
      pinch_point = CGPointMake((point0.x + point1.x) / 2.0, (point0.y + point1.y) / 2.0);
      pinch_scale = 1.0;
      _pinch_occurring = true;
    } else if (pinch.state == UIGestureRecognizerStateEnded ||
               pinch.state == UIGestureRecognizerStateCancelled ||
               pinch.state == UIGestureRecognizerStateFailed) {
      _pinch_occurring = false;
    }
    CGFloat p_scale = 1.0 + (pinch_scale - pinch.scale);
    LFL::v2 p(pinch_point.x, s->gl_y + s->gl_h - pinch_point.y), d(p_scale, p_scale);
    int fired = LFL::app->input->MouseZoom(p, d, begin);
    if (fired && uiapp.frame_on_mouse_input) [uiapp.glk_view setNeedsDisplay];
    pinch_scale = pinch.scale;
  }

  - (BOOL)canPerformAction:(SEL)action withSender:(id)sender {
    return action == @selector(copy:) || action == @selector(paste:) || action == @selector(toggleKeyboard:);
  }

  - (void)copy:(id)sender { if (uiapp.glk_view.copy_cb) uiapp.glk_view.copy_cb(); }
  - (void)paste:(id)sender { [uiapp.glk_view paste: sender]; }
  - (void)toggleKeyboard:(id)sender { LFL::app->ToggleTouchKeyboard(); }
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
    // if (uiapp.controller.pinch_occurring) return;
    UITouch *touch = [touches anyObject];
    UIView *v = [touch view];
    CGPoint position = [touch locationInView:v];
    int dpind = v.frame.origin.x == 0, scale = [uiapp getScale];
    if (!dpind) position = CGPointMake(scale * (position.x + v.frame.origin.x), scale * (position.y + v.frame.origin.y));
    else        position = CGPointMake(scale * position.x, scale * position.y);
    if (auto s = LFL::app->focused) {
#if 0
      s->gesture_dpad_x[dpind] = position.x;
      s->gesture_dpad_y[dpind] = position.y;
#endif
      LFL::point p(position.x, s->gl_y + s->gl_h - (int)position.y);
      int fired = LFL::app->input->MouseClick(1, 1, p);
      if (fired && uiapp.frame_on_mouse_input) [self.superview setNeedsDisplay];
    }
  }

  - (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {}
  - (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
    UITouch *touch = [touches anyObject];
    UIView *v = [touch view];
    CGPoint position = [touch locationInView:v];
    int dpind = v.frame.origin.x == 0, scale = [uiapp getScale];
    if (!dpind) position = CGPointMake(scale * (position.x + v.frame.origin.x), scale * (position.y + v.frame.origin.y));
    else        position = CGPointMake(scale * position.x, scale * position.y);
    if (auto s = LFL::app->focused) {
#if 0
      s->gesture_dpad_stop[dpind] = 1;
      s->gesture_dpad_x[dpind] = 0;
      s->gesture_dpad_y[dpind] = 0;
#endif
      LFL::point p(position.x, s->gl_y + s->gl_h - position.y);
      int fired = LFL::app->input->MouseClick(1, 0, p);
      if (fired && uiapp.frame_on_mouse_input) [self.superview setNeedsDisplay];
    }
  }

#if 0
  - (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
    UITouch *touch = [touches anyObject];
    UIView *v = [touch view];
    CGPoint position = [touch locationInView:v];
    int dpind = v.frame.origin.x == 0, scale = [uiapp getScale];
    if (!dpind) position = CGPointMake(scale * (position.x + v.frame.origin.x), scale * (position.y + v.frame.origin.y));
    else        position = CGPointMake(scale * position.x, scale * position.y);
    if (auto s = LFL::app->focused) {
#if 0
      s->gesture_dpad_x[dpind] = position.x;
      s->gesture_dpad_y[dpind] = position.y;
#endif
      LFL::point pos = LFL::Floor(LFL::GetCGPoint(position));
      pos.y = s->y + s->height - pos.y;
      int fired = LFL::app->input->MouseMove(pos, LFL::app->focused ? (pos - LFL::app->focused->mouse) : pos);
      if (fired && uiapp.frame_on_mouse_input) [self.superview setNeedsDisplay];
    }
  }
#endif
@end

@implementation IOSPicker
  - (id)initWithColumns:(LFL::PickerItem)in andTheme:(const LFL::string&)theme {
    self = [super init];
    return [self finishInit:in withTheme:theme];
  }

  - (id)initWithColumns:(LFL::PickerItem)in andTheme:(const LFL::string&)theme andFrame:(CGRect)r {
    self = [super initWithFrame:r];
    return [self finishInit:in withTheme:theme];
  }

  - (id)finishInit:(LFL::PickerItem)in withTheme:(const LFL::string&)theme {
    item = move(in);
    super.delegate = self;
    super.showsSelectionIndicator = YES;
    super.hidden = NO;
    super.layer.borderColor = [UIColor grayColor].CGColor;
    super.layer.borderWidth = 4;
    [self setTheme: theme];
    return self;
  }

  - (void)setTheme: (const LFL::string&)x {
    if ((dark_theme = (x == "Dark"))) [self setBackgroundColor:[UIColor colorWithWhite:0.249 alpha:.5]];
    else                              [self setBackgroundColor:[UIColor whiteColor]];
  }

  - (void)pickerView:(UIPickerView *)pV didSelectRow:(NSInteger)row inComponent:(NSInteger)component {
    CHECK_RANGE(component, 0, item.data.size());
    if (item.picked.size() != item.data.size()) item.picked.resize(item.data.size());
    item.picked[component] = row;
    if (item.cb) { if (item.cb(&item)) [self removeFromSuperview]; }
  }

  - (LFL::PickerItem*)getItem { return &item; }
  - (NSInteger)numberOfComponentsInPickerView:(UIPickerView *)pickerView { return item.data.size(); }
  - (NSInteger)pickerView:(UIPickerView *)pickerView numberOfRowsInComponent:(NSInteger)component { 
    CHECK_RANGE(component, 0, item.data.size());
    return item.data[component].size();
  }

  - (NSString *)pickerView:(UIPickerView *)pickerView titleForRow:(NSInteger)row forComponent:(NSInteger)component {
    CHECK_RANGE(component, 0, item.data.size());
    CHECK_RANGE(row, 0, item.data[component].size());
    return [NSString stringWithUTF8String: item.data[component][row].c_str()];
  }

  - (NSAttributedString *)pickerView:(UIPickerView *)pickerView attributedTitleForRow:(NSInteger)row forComponent:(NSInteger)component {
    CHECK_RANGE(component, 0, item.data.size());
    CHECK_RANGE(row, 0, item.data[component].size());
    NSString *title = [NSString stringWithUTF8String: item.data[component][row].c_str()];
    NSAttributedString *attString = 
      [[NSAttributedString alloc] initWithString:title attributes:@{
        NSForegroundColorAttributeName:(dark_theme ? [UIColor whiteColor] : [UIColor blackColor]) }];
    return attString;
  }

  - (void)selectRows:(const LFL::StringVec&)v {
    if (item.picked.size() != item.data.size()) item.picked.resize(item.data.size());
    CHECK_EQ(item.picked.size(), v.size());
    for (int col=0, l=v.size(); col != v.size(); ++col) {
      const LFL::string &name = v[col];
      for (auto b = item.data[col].begin(), e = item.data[col].end(), i = b; i != e; ++i) {
        if (*i == name) { item.picked[col] = i - b; break; }
      }
      [self selectRow:item.picked[col] inComponent:col animated:NO];
    }
  }
@end

@implementation IOSFontPicker
  {
    LFL::StringVecCB font_change_cb;
  }

  - (id)initWithTheme: (const LFL::string&)theme {
    LFL::PickerItem p;
    p.cb = [=](LFL::PickerItem *x) -> bool {
      if (font_change_cb) font_change_cb(LFL::StringVec{x->data[0][x->picked[0]], x->data[1][x->picked[1]]});
      return true;
    };
    [IOSFontPicker getSystemFonts:     &LFL::PushBack(p.data, {})];
    [IOSFontPicker getSystemFontSizes: &LFL::PushBack(p.data, {})];
    self = [super initWithColumns: move(p) andTheme: theme];
    return self;
  }

  - (void)selectFont:(const std::string&)name size:(int)s cb:(LFL::StringVecCB)v {
    font_change_cb = move(v);
    [IOSFontPicker selectFont:name withPicker:self size:s];
  }

  + (void)getSystemFonts:(std::vector<std::string>*)out {
    NSArray *families = [UIFont familyNames];
    for (NSString *family_name in families) {
      NSArray *fonts = [UIFont fontNamesForFamilyName:family_name];
      for (NSString *font_name in fonts) out->push_back([font_name UTF8String]);
    }
  }

  + (void)getSystemFontSizes:(std::vector<std::string>*)out {
    for (int i=0; i<64; ++i) out->push_back(LFL::StrCat(i+1));
  }

  + (void)selectFont:(const std::string&)name withPicker:(IOSPicker*)picker size:(int)s {
    [picker selectRows:LFL::StringVec{ name, LFL::StrCat(s) }];
  }
@end

@implementation IOSAlert
  - (id)init:(const LFL::AlertItemVec&) kv {
    self = [super init];
    CHECK_EQ(4, kv.size());
    CHECK_EQ("style", kv[0].first);
    _style       = kv[0].second;
    _cancel_cb   = kv[2].cb;
    _confirm_cb  = kv[3].cb;

    if (kv[2].first.empty() && kv[3].first.empty()) {
      CGRect bounds = [[UIScreen mainScreen] bounds];
      CGFloat cornerRadius = 7, flashWidth = 150, flashHeight = 30;

      _flash = [[UIView alloc] initWithFrame:
        CGRectMake((bounds.size.width - flashWidth) / 2, (bounds.size.height - flashHeight) / 2, flashWidth, flashHeight)];
      _flash.autoresizingMask = UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleLeftMargin |
        UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleBottomMargin;
      _flash.layer.cornerRadius = cornerRadius;
      _flash.layer.borderColor = [[UIColor colorWithRed:198.0/255.0 green:198.0/255.0 blue:198.0/255.0 alpha:1.0f] CGColor];
      _flash.layer.borderWidth = 1;
      _flash.layer.shadowRadius = cornerRadius + 5;
      _flash.layer.shadowOpacity = 0.1f;
      _flash.layer.shadowOffset = CGSizeMake(0 - (cornerRadius+5)/2, 0 - (cornerRadius+5)/2);
      _flash.layer.shadowColor = [UIColor blackColor].CGColor;
      _flash.layer.shadowPath = [UIBezierPath bezierPathWithRoundedRect:_flash.bounds cornerRadius:_flash.layer.cornerRadius].CGPath;

      CAGradientLayer *gradient = [CAGradientLayer layer];
      gradient.cornerRadius = cornerRadius;
      gradient.frame = _flash.bounds;
      gradient.colors = [NSArray arrayWithObjects:
        (id)[[UIColor colorWithRed:218.0/255.0 green:218.0/255.0 blue:218.0/255.0 alpha:1.0f] CGColor],
        (id)[[UIColor colorWithRed:233.0/255.0 green:233.0/255.0 blue:233.0/255.0 alpha:1.0f] CGColor],
        (id)[[UIColor colorWithRed:218.0/255.0 green:218.0/255.0 blue:218.0/255.0 alpha:1.0f] CGColor],
        nil];
      [_flash.layer insertSublayer:gradient atIndex:0];

      _flash_text = [[UILabel alloc] initWithFrame: CGRectMake(10, 0, flashWidth-20, 30)];
      _flash_text.font = [UIFont boldSystemFontOfSize:18.0];
      _flash_text.autoresizingMask = _flash.autoresizingMask;
      _flash_text.textAlignment = NSTextAlignmentCenter;
      [_flash addSubview: _flash_text];

    } else {
      self.alert   = [UIAlertController
        alertControllerWithTitle: LFL::MakeNSString(kv[1].first)
        message:                  LFL::MakeNSString(kv[1].second)
        preferredStyle:           UIAlertControllerStyleAlert];
      
      if ((_add_text = _style == "textinput")) {
        [_alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        }];
      } else if ((_add_text = _style == "pwinput")) {
        [_alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
          textField.secureTextEntry = YES;
        }];
      }
      
      if (!kv[2].first.empty()) {
        UIAlertAction *a = [UIAlertAction
          actionWithTitle: LFL::MakeNSString(kv[2].first)
          style:           UIAlertActionStyleCancel
          handler:         ^(UIAlertAction *action) {
            _done = true;
            if (_cancel_cb) _cancel_cb("");
          }];
        [_alert addAction:a];
      }
      
      if (!kv[3].first.empty()) {
        UIAlertAction *a = [UIAlertAction
          actionWithTitle: LFL::MakeNSString(kv[3].first)
          style:           UIAlertActionStyleDefault
          handler:         ^(UIAlertAction *action) {
            _done = true;
            if (_confirm_cb) _confirm_cb(_add_text ? LFL::GetNSString(_alert.textFields[0].text) : "");
          }];
        [_alert addAction:a];
      }
    }

    return self;
  }

  - (void)show: (bool)show_or_hide {
    LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
    _done = !show_or_hide;
    if (_flash) [_flash removeFromSuperview];
    if (show_or_hide) {
      if (!_flash) [uiapp.top_controller presentViewController:_alert animated:YES completion:nil];
      else [uiapp.top_controller.view addSubview: _flash];
    } else if (!_flash) [_alert dismissViewControllerAnimated:YES completion:nil];
  }
@end

@interface IOSMenu : NSObject<UIActionSheetDelegate>
  {
    std::vector<LFL::MenuItem> menu;
  }
  @property (nonatomic, retain) UIActionSheet *actions;
@end

@implementation IOSMenu
  - (id)init:(const std::string&)title_text items:(LFL::MenuItemVec)item {
    self = [super init];
    menu = move(item);
    NSString *title = LFL::MakeNSString(title_text);
    _actions = [[UIActionSheet alloc] initWithTitle:title delegate:self
      cancelButtonTitle:@"Cancel" destructiveButtonTitle:nil otherButtonTitles:nil];
    for (auto &i : menu) [_actions addButtonWithTitle: LFL::MakeNSString(i.name)];
    return self;
  }

  - (void)actionSheet:(UIActionSheet *)actions clickedButtonAtIndex:(NSInteger)buttonIndex {
    if (buttonIndex < 1 || buttonIndex > menu.size()) { ERRORf("invalid buttonIndex %d size=%d", buttonIndex, menu.size()); return; }
    if (menu[buttonIndex-1].cb) menu[buttonIndex-1].cb();
  }
@end

namespace LFL {
struct iOSFrameworkModule : public Module {
  int Init() {
    INFO("iOSFrameworkModule::Init()");
    if (auto s = dynamic_cast<iOSWindow*>(app->focused)) {
      CHECK(!s->id);
      s->id = (s->glkview = [[LFUIApplication sharedAppDelegate] glk_view]);
      CHECK(s->id);
      app->windows[s->id] = s;
    }
    LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
    CGFloat scale = [uiapp getScale];
    CGRect rect = [uiapp getFrame];
    int w = rect.size.width * scale, h = rect.size.height * scale;
    if (auto s = app->focused) Assign(&s->gl_w, &s->gl_h, w, h);
    INFOf("iOSFrameworkModule::Init %d %d vs %d %d", w, h, [uiapp glk_view].drawableWidth, [uiapp glk_view].drawableHeight);
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

struct iOSAlertView : public AlertViewInterface {
  IOSAlert *alert;
  ~iOSAlertView() { [alert release]; }
  iOSAlertView(AlertItemVec items) : alert([[IOSAlert alloc] init: move(items)]) {}

  void Hide() { [alert show: false]; }
  void Show(const string &arg) {
    if (alert.add_text) alert.alert.textFields[0].text = MakeNSString(arg);
    [alert show: true];
  }

  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
    alert.confirm_cb = move(confirm_cb);
    if (alert.flash) alert.flash_text.text = LFL::MakeNSString(title);
    else {
      alert.alert.title = MakeNSString(title);
      alert.alert.message = MakeNSString(msg);
    }
    Show(arg);
  }

  string RunModal(const string &arg) {
    if (alert.add_text) alert.alert.textFields[0].text = MakeNSString(arg);
    alert.done = false;
    [alert show: true];
    NSRunLoop *rl = [NSRunLoop currentRunLoop];
    // do { [rl runMode:NSRunLoopCommonModes beforeDate:[NSDate distantFuture]]; }
    do { [rl runMode:NSRunLoopCommonModes beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.3]]; }
    while(!alert.done);
    return alert.add_text ? GetNSString(alert.alert.textFields[0].text) : "";
  }
};

struct iOSMenuView : public MenuViewInterface {
  IOSMenu *menu;
  ~iOSMenuView() { [menu release]; }
  iOSMenuView(const string &t, MenuItemVec i) : menu([[IOSMenu alloc] init:t items:move(i)]) {}
  void Show() { [menu.actions showInView: [UIApplication sharedApplication].keyWindow]; }
};

struct iOSNag : public NagInterface {
  iOSNag(const string &id, int min_days, int min_uses, int min_events, int remind_days) {
    [Appirater setAppId: MakeNSString(id)];
    [Appirater setDaysUntilPrompt: min_days];
    [Appirater setUsesUntilPrompt: min_uses];
    [Appirater setSignificantEventsUntilPrompt: min_events];
    [Appirater setTimeBeforeReminding: remind_days];
    [Appirater setDebug: NO];
    [Appirater appLaunched:YES];
  }
};

int Application::Suspended() { return 0; }
void Application::RunCallbackInMainThread(Callback cb) {
#if 0
  ObjcCallback *ocb = [[ObjcCallback alloc] initWithCB: move(cb)];
  [ocb performSelectorOnMainThread:@selector(runAndRelease) withObject:nil waitUntilDone:NO];
#else
  message_queue.Write(new Callback(move(cb)));
  if (!FLAGS_target_fps && !app->suspended) scheduler.Wakeup(focused);
#endif
}

void Application::MakeCurrentWindow(Window *W) {}
void Application::CloseWindow(Window *W) {}

void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}

string Application::GetClipboardText() { return GetNSString([UIPasteboard generalPasteboard].string); }
void Application::SetClipboardText(const string &s) {
  if (auto ns = MakeNSString(s)) [UIPasteboard generalPasteboard].string = ns;
}

void Application::OpenTouchKeyboard() { [[LFUIApplication sharedAppDelegate] showKeyboard]; }
void Application::CloseTouchKeyboard() { [[LFUIApplication sharedAppDelegate] hideKeyboard]; }
void Application::CloseTouchKeyboardAfterReturn(bool v) { [LFUIApplication sharedAppDelegate].glk_view.resign_textfield_on_return = v; }
void Application::SetTouchKeyboardTiled(bool v) { [[LFUIApplication sharedAppDelegate].controller setOverlapKeyboard: !v]; }
void Application::ToggleTouchKeyboard() {
  if ([[LFUIApplication sharedAppDelegate] isKeyboardFirstResponder]) CloseTouchKeyboard();
  else OpenTouchKeyboard();
}

bool Application::GetAppFrameEnabled() { return ![LFUIApplication sharedAppDelegate].frame_disabled; }
void Application::SetAppFrameEnabled(bool v) { [LFUIApplication sharedAppDelegate].frame_disabled = !v; INFO("frame enabled = ", v); }
void Application::SetAutoRotateOrientation(bool v) {}
int Application::SetMultisample(bool v) { return [[LFUIApplication sharedAppDelegate] updateGLKMultisample:v]; }
int Application::SetExtraScale(bool v) { return [[LFUIApplication sharedAppDelegate] updateScale:v]; }
void Application::SetDownScale(bool v) { [[LFUIApplication sharedAppDelegate] downScale:v]; }
void Application::SetTitleBar(bool v) { [LFUIApplication sharedAppDelegate].show_title = v; }
void Application::SetKeepScreenOn(bool v) { [UIApplication sharedApplication].idleTimerDisabled = v; }
void Application::SetExtendedBackgroundTask(Callback cb) { [[LFUIApplication sharedAppDelegate] setExtendedBackgroundTaskCB: move(cb)]; }
void Application::SetVerticalSwipeRecognizer(int touches) { [[LFUIApplication sharedAppDelegate].controller initVerticalSwipeGestureRecognizers: touches]; }
void Application::SetHorizontalSwipeRecognizer(int touches) { [[LFUIApplication sharedAppDelegate].controller initHorizontalSwipeGestureRecognizers: touches]; }
void Application::SetPanRecognizer(bool enabled) { [[LFUIApplication sharedAppDelegate].controller initPanGestureRecognizers]; }
void Application::SetPinchRecognizer(bool enabled) { [[LFUIApplication sharedAppDelegate].controller initPinchGestureRecognizers]; }
void Application::ShowSystemStatusBar(bool v) { [[LFUIApplication sharedAppDelegate].controller showStatusBar:v]; }
void Application::SetTheme(const string &v) {}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &cb) {
  static IOSFontPicker *font_chooser = [[IOSFontPicker alloc] initWithTheme: "Light"];
  [font_chooser selectFont:cur_font.name size:cur_font.size cb:cb];
  [[LFUIApplication sharedAppDelegate].glk_view addSubview: font_chooser];
}

void Application::ShowSystemContextMenu(const MenuItemVec &items) {
  auto uiapp = [LFUIApplication sharedAppDelegate];
  UIMenuController* mc = [UIMenuController sharedMenuController];
  vector<UIMenuItem*> menuitems;
  for (auto &i : items) {
    if      (i.name == "Copy") { uiapp.glk_view.copy_cb = i.cb; continue; }
    else if (i.name == "Keyboard") {
      string title = [[LFUIApplication sharedAppDelegate] isKeyboardFirstResponder] ? "Hide Keyboard" : "Show Keyboard";
      UIMenuItem *mi = [[UIMenuItem alloc] initWithTitle: MakeNSString(title) action:@selector(toggleKeyboard:)];
      menuitems.push_back(mi);
    }
  }
  if (menuitems.size())
    mc.menuItems = [NSArray arrayWithObjects:&menuitems[0] count:menuitems.size()];

  auto w = app->focused;
  float s = [uiapp getScale];
  CGRect rect = CGRectMake(w->mouse.x / s, (w->gl_h + w->gl_y - w->mouse.y) / s, w->default_font->Height(), 100);
  [mc setTargetRect:rect inView:dynamic_cast<iOSWindow*>(w)->glkview];
  [mc setMenuVisible:![mc isMenuVisible] animated:TRUE];
}

void iOSWindow::SetResizeIncrements(float x, float y) {}
void iOSWindow::SetTransparency(float v) {}
bool iOSWindow::Reshape(int w, int h) { return false; }
void iOSWindow::SetCaption(const string &v) {
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

FrameScheduler::FrameScheduler() :
  maxfps(&FLAGS_target_fps), wakeup_thread(&frame_mutex, &wait_mutex), rate_limit(0), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(0), run_main_loop(0) {}

bool FrameScheduler::DoMainWait() { return false; }
void FrameScheduler::Wakeup(Window *w, int) {
  dispatch_async(dispatch_get_main_queue(), ^{ [dynamic_cast<iOSWindow*>(w)->glkview setNeedsDisplay]; });
}

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
  [LFUIApplication sharedAppDelegate].glk_view.frame_on_keyboard_input = YES;
}

void FrameScheduler::DelMainWaitKeyboard(Window*) {
 [LFUIApplication sharedAppDelegate].glk_view.frame_on_keyboard_input = NO;
}

void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()> cb) {
  if (fd == InvalidSocket) return;
  if (!wait_forever_thread) {
    CHECK_EQ(SocketSet::READABLE, flag);
    [[LFUIApplication sharedAppDelegate] addMainWaitSocket:fd callback:move(cb)];
  }
}

void FrameScheduler::DelMainWaitSocket(Window *w, Socket fd) {
  if (fd == InvalidSocket) return;
  [[LFUIApplication sharedAppDelegate] delMainWaitSocket: fd];
}

Window *Window::Create() { return new iOSWindow(); }
Application *CreateApplication(int ac, const char* const* av) { return new Application(ac, av); }
unique_ptr<Module> CreateFrameworkModule() { return make_unique<iOSFrameworkModule>(); }
unique_ptr<AssetLoaderInterface> CreateAssetLoader() { return make_unique<iOSAssetLoader>(); }
unique_ptr<TimerInterface> SystemToolkit::CreateTimer(Callback cb) { return make_unique<AppleTimer>(move(cb)); }
unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(AlertItemVec items) { return make_unique<iOSAlertView>(move(items)); }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(const string &title, MenuItemVec items) { return make_unique<iOSMenuView>(title, move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(vector<MenuItem> items) { return nullptr; }
unique_ptr<NagInterface> SystemToolkit::CreateNag(const string &id, int min_days, int min_uses, int min_events, int remind_days) { return make_unique<iOSNag>(id, min_days, min_uses, min_events, remind_days); }

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
