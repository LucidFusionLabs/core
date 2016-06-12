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
#ifndef LFL_IOS_SIM 
#import <AVFoundation/AVFoundation.h>
#endif

#include "core/app/app.h"
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
};

@implementation LFUIApplication
  {
    CGFloat scale;
    CGRect keyboard_frame;
    NSFileHandle *wait_forever_fh;
    bool restart_wait_forever_fh, want_extra_scale;
    int current_orientation, target_fps;
  }
  @synthesize window, controller, view, lview, rview, textField;

  + (LFUIApplication *)sharedAppDelegate { return (LFUIApplication *)[[UIApplication sharedApplication] delegate]; }
  - (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    CGRect wbounds = [[UIScreen mainScreen] bounds];
    scale = [[UIScreen mainScreen] scale];
    self.window = [[[UIWindow alloc] initWithFrame:wbounds] autorelease];

    EAGLContext *context = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
    [EAGLContext setCurrentContext:context];
    self.view = [[[GLKView alloc] initWithFrame:wbounds] autorelease];
    self.view.context = context;
    self.view.delegate = self;
    self.view.enableSetNeedsDisplay = TRUE;
    self.view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight |
      UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin | 
      UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleBottomMargin;
    [context release];

    self.controller = [[[LFViewController alloc] initWithNibName:nil bundle:nil] autorelease];
    self.controller.delegate = self.controller;
    [self.controller setView:self.view]; 
    self.controller.resumeOnDidBecomeActive = NO;
    // self.controller.wantsFullScreenLayout = YES;

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
    _resign_textfield_on_return = YES;
    self.textField = [[[MyTextField alloc] initWithFrame: CGRectZero] autorelease];
    self.textField.delegate = self;
    self.textField.text = [NSString stringWithFormat:@"default"];
    self.textField.autocorrectionType = UITextAutocorrectionTypeNo;
    self.textField.autocapitalizationType = UITextAutocapitalizationTypeNone;
    [self.window addSubview:self.textField];

    [[NSFileManager defaultManager] changeCurrentDirectoryPath: [[NSBundle mainBundle] resourcePath]];
    NSLog(@"iPhoneMain argc=%d", LFL::app->argc);
    MyAppMain();
    INFOf("didFinishLaunchingWithOptions, views: %p, %p, %p, csf=%f", self.view, self.lview, self.rview, scale);

    [self.window makeKeyAndVisible];
    [self initGestureRecognizers];
    return YES;
  }

  - (void)applicationWillTerminate:(UIApplication *)application {
    [self shutdownNotifications];
    [self shutdownGestureRecognizers];
  }

  - (void)applicationWillResignActive:(UIApplication*)application {}
  - (void)applicationDidBecomeActive:(UIApplication*)application {}

  - (void)glkView:(GLKView *)v drawInRect:(CGRect)rect {
    if (!_screen_width || !_screen_height) return;
    if (LFL::screen->width != _screen_width || LFL::screen->height != _screen_height)
      WindowReshaped(0, _screen_y, _screen_width, _screen_height);

    LFAppFrame(true); 

    if (wait_forever_fh && restart_wait_forever_fh && !(restart_wait_forever_fh=0))
        [wait_forever_fh waitForDataInBackgroundAndNotify];
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
  - (bool)isKeyboardFirstResponder { return [self.textField isFirstResponder]; }
  - (void)showKeyboard { [self.textField becomeFirstResponder]; }
  - (void)hideKeyboard { [self.textField resignFirstResponder]; }
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
    fired += KeyPress(k, 1);
    fired += KeyPress(k, 0);
    if (fired && _frame_on_keyboard_input) [self.view setNeedsDisplay];
  }

  - (int)getOrientation { return current_orientation; }
  - (void)orientationChanged: (id)sender {
    UIDeviceOrientation new_orientation = [[UIDevice currentDevice] orientation];
    bool landscape = UIDeviceOrientationIsLandscape(new_orientation);
    INFOf("notification of new orientation: %d -> %d landscape=%d", current_orientation, new_orientation, landscape);
    current_orientation = new_orientation;
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

  - (void)doubleSwipeUp:   (id)sender { LFL::screen->gesture_swipe_up   = 1; }
  - (void)doubleSwipeDown: (id)sender { LFL::screen->gesture_swipe_down = 1; }
  - (void)tapGesture: (UITapGestureRecognizer *)tapGestureRecognizer {
    UIView *v = [tapGestureRecognizer view];
    CGPoint position = [tapGestureRecognizer locationInView:v];
    int dpind = v.frame.origin.y == 0;
    LFL::screen->gesture_tap[dpind] = 1;
    LFL::screen->gesture_dpad_x[dpind] = position.x;
    LFL::screen->gesture_dpad_y[dpind] = position.y;
    int fired = MouseClick(1, 1, (int)position.x, LFL::screen->y + LFL::screen->height - (int)position.y);
    if (fired && _frame_on_mouse_input) [self.view setNeedsDisplay];
  }

  - (void)panGesture: (UIPanGestureRecognizer *)panGestureRecognizer {
    UIView *v = [panGestureRecognizer view];
    int dpind = v.frame.origin.y == 0;

    if (panGestureRecognizer.state == UIGestureRecognizerStateChanged) {
      // CGPoint velocity = [panGestureRecognizer translationInView:v];
      CGPoint velocity = [panGestureRecognizer velocityInView:v];
      if (fabs(velocity.x) > 15 || fabs(velocity.y) > 15) {
          LFL::screen->gesture_dpad_dx[dpind] = velocity.x;
          LFL::screen->gesture_dpad_dy[dpind] = velocity.y;
      }
      // CGPoint position = [panGestureRecognizer locationInView:v];
      CGPoint position = [panGestureRecognizer locationOfTouch:0 inView:v];
      if (!dpind) {
          position.x += v.frame.origin.x;
          position.y += v.frame.origin.y;
      }
      LFL::screen->gesture_dpad_x[dpind] = position.x;
      LFL::screen->gesture_dpad_y[dpind] = position.y;
      // INFOf("gest %f %f %f %f", position.x, position.y, velocity.x, velocity.y);
      // INFOf("origin %f %f", v.frame.origin.x, v.frame.origin.y);
    }
    else if (panGestureRecognizer.state == UIGestureRecognizerStateEnded) {
      LFL::screen->gesture_dpad_stop[dpind] = 1;
      LFL::screen->gesture_dpad_x[dpind] = 0;
      LFL::screen->gesture_dpad_y[dpind] = 0;
      // CGPoint position = [panGestureRecognizer locationInView:v];
      // INFOf("gest %f %f stop", position.x, position.y);
    }
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
    [[NSNotificationCenter defaultCenter] removeObserver:self name:NSFileHandleDataAvailableNotification object:wait_forever_fh];
    // [wait_forever_fh closeFile];
    [wait_forever_fh release];
    wait_forever_fh = nil;
    restart_wait_forever_fh = false;
  }

  - (void)fileDataAvailable: (NSNotification *)notification {
    NSFileHandle *fh = (NSFileHandle*) [notification object];
    if (fh != wait_forever_fh) return;
    restart_wait_forever_fh = true;
    [self.view setNeedsDisplay];
  }
@end

@implementation LFViewController
  {
    LFUIApplication *uiapp;
    UIToolbar *toolbar;
    int toolbar_height;
    bool overlap_keyboard;
    std::unordered_map<std::string, void*> toolbar_titles;
    std::unordered_map<void*, std::string> toolbar_cmds;
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
    CGFloat s     = [uiapp getScale];
    CGRect bounds = [[UIScreen mainScreen] bounds];
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

  - (void)addToolbar:(int)n key:(const char**)k val:(const char**)v {
    if (toolbar) FATALf("addToolbar w toolbar=%p", toolbar);
    NSMutableArray *items = [[NSMutableArray alloc] init];
    UIBarButtonItem *spacer = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemFlexibleSpace target:nil action:nil];
    for (int i=0; i<n; i++) {
      if (i) [items addObject: spacer];
      NSString *K = [NSString stringWithUTF8String: k[i]];
      UIBarButtonItem *item =
        [[UIBarButtonItem alloc] initWithTitle:[NSString stringWithFormat:@"%@\U0000FE0E", K]
        style:UIBarButtonItemStyleBordered target:self action:@selector(onClick:)];
      [items addObject:item];
      toolbar_titles[k[i]] = item;
      toolbar_cmds[item] = v[i];
      [item release];
    }
    toolbar_height = 30;
    toolbar = [[UIToolbar alloc]initWithFrame: [self getToolbarFrame]];
    // [toolbar setBarStyle:UIBarStyleBlackTranslucent];
    [toolbar setItems:items];
    [items release];
    [spacer release];
    [uiapp.window addSubview:toolbar];
  }

  - (void)updateToolbarFrame {
    if (toolbar) toolbar.frame = [self getToolbarFrame];
    [self.view setNeedsLayout];
  }

  - (CGRect)getToolbarFrame {
    CGRect bounds = [[UIScreen mainScreen] bounds], kbd = [[LFUIApplication sharedAppDelegate] getKeyboardFrame];
    return CGRectMake(0, bounds.size.height - kbd.size.height - toolbar_height, bounds.size.width, toolbar_height);
  }

  - (CGRect)getKeyboardToolbarFrame {
    CGRect kbd = [[LFUIApplication sharedAppDelegate] getKeyboardFrame];
    return CGRectMake(kbd.origin.x, kbd.origin.y, kbd.size.width, kbd.size.height + toolbar_height);
  }

  - (void)toggleToolbarButton:(id)sender {
    if (![sender isKindOfClass:[UIBarButtonItem class]]) FATALf("unknown sender: %p", sender);
    UIBarButtonItem *item = (UIBarButtonItem*)sender;
    if (item.style != UIBarButtonItemStyleDone) { item.style = UIBarButtonItemStyleDone;     item.tintColor = [UIColor colorWithRed:0.8 green:0.8 blue:0.8 alpha:.8]; }
    else                                        { item.style = UIBarButtonItemStyleBordered; item.tintColor = nil; }
  }

  - (void)toggleToolbarButtonWithTitle:(const char *)k {
    auto it = toolbar_titles.find(k);
    if (it != toolbar_titles.end()) [self toggleToolbarButton: (id)(UIBarButtonItem*)it->second];
  }

  - (void)onClick:(id)sender {
    auto it = toolbar_cmds.find(sender);
    if (it != toolbar_cmds.end()) {
      ShellRun(it->second.c_str());
      if (it->second.substr(0,6) == "toggle") [self toggleToolbarButton:sender];
    }
    [self resignFirstResponder];
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
    LFL::screen->gesture_dpad_x[dpind] = position.x;
    LFL::screen->gesture_dpad_y[dpind] = position.y;
    int fired = MouseClick(1, 1, (int)position.x, LFL::screen->y + LFL::screen->height - (int)position.y);
    if (fired && uiapp.frame_on_mouse_input) [self.superview setNeedsDisplay];
  }

  - (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event {}
  - (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
    UITouch *touch = [touches anyObject];
    UIView *v = [touch view];
    CGPoint position = [touch locationInView:v];
    int dpind = v.frame.origin.y == 0, scale = [uiapp getScale];
    if (!dpind) position = CGPointMake(scale * (position.x + v.frame.origin.x), scale * (position.y + v.frame.origin.y));
    else        position = CGPointMake(scale * position.x, scale * position.y);
    LFL::screen->gesture_dpad_stop[dpind] = 1;
    LFL::screen->gesture_dpad_x[dpind] = 0;
    LFL::screen->gesture_dpad_y[dpind] = 0;
    int fired = MouseClick(1, 0, (int)position.x, LFL::screen->y + LFL::screen->height - (int)position.y);
    if (fired && uiapp.frame_on_mouse_input) [self.superview setNeedsDisplay];
  }

  - (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
    UITouch *touch = [touches anyObject];
    UIView *v = [touch view];
    CGPoint position = [touch locationInView:v];
    int dpind = v.frame.origin.y == 0, scale = [uiapp getScale];
    if (!dpind) position = CGPointMake(scale * (position.x + v.frame.origin.x), scale * (position.y + v.frame.origin.y));
    else        position = CGPointMake(scale * position.x, scale * position.y);
    LFL::screen->gesture_dpad_x[dpind] = position.x;
    LFL::screen->gesture_dpad_y[dpind] = position.y;
  }
@end

@implementation MyTextField
  - (void)deleteBackward {
    [super deleteBackward];
    [[LFUIApplication sharedAppDelegate] handleKey: LFL::Key::Backspace];
  }
@end

namespace LFL {
struct iPhoneFrameworkModule : public Module {
  int Init() {
    INFO("iPhoneFrameworkModule::Init()");
    CHECK(!screen->id.v);
    screen->id = LFL::MakeTyped([[LFUIApplication sharedAppDelegate] view]);
    CHECK(screen->id.v);
    app->windows[screen->id.v] = screen;

    LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
    CGFloat scale = [uiapp getScale];
    CGRect rect = [uiapp getFrame];
    screen->width = rect.size.width * scale;
    screen->height = rect.size.height * scale;
    INFOf("viewWillLayoutSubviews %d %d vs %d %d",
          screen->width, screen->height, [uiapp view].drawableWidth, [uiapp view].drawableHeight);
    return 0;
  }
};

struct iPhoneAssetLoader : public SimpleAssetLoader {
  virtual void UnloadAudioFile(void *h) {}
  virtual void *LoadAudioFile(File*) { return 0; }
  virtual void *LoadAudioFileNamed(const string &filename) {
#ifdef LFL_IOS_SIM
    return 0;
#else // LFL_IOS_SIM
    NSError *error;
    NSString *fn = [NSString stringWithCString:filename.c_str() encoding:NSASCIIStringEncoding];
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithFormat:@"%@/%@", [[NSBundle mainBundle] resourcePath], fn]];
    AVAudioPlayer *audio_player = [[AVAudioPlayer alloc] initWithContentsOfURL:url error:&error];
    if (audio_player == nil) NSLog(@"%@", [error description]);
    return audio_player;
#endif // LFL_IOS_SIM
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
  [UIPasteboard generalPasteboard].string = [NSString stringWithUTF8String:s.c_str()];
}

void Application::AddToolbar(const vector<pair<string, string>>&items) {
  vector<const char *> k, v;
  for (auto &i : items) { k.push_back(i.first.c_str()); v.push_back(i.second.c_str()); }
  [[LFUIApplication sharedAppDelegate].controller addToolbar:items.size() key:&k[0] val:&v[0]];
}

void Application::ToggleToolbarButton(const string &n) { 
  [[LFUIApplication sharedAppDelegate].controller toggleToolbarButtonWithTitle:n.c_str()];
}

void Application::OpenTouchKeyboard() { [[LFUIApplication sharedAppDelegate] showKeyboard]; }
void Application::CloseTouchKeyboard() { [[LFUIApplication sharedAppDelegate] hideKeyboard]; }
void Application::CloseTouchKeyboardAfterReturn(bool v) { [LFUIApplication sharedAppDelegate].resign_textfield_on_return = v; }
void Application::SetTouchKeyboardTiled(bool v) { [[LFUIApplication sharedAppDelegate].controller setOverlapKeyboard: !v]; }
bool Application::GetTouchKeyboardOpened() { return [[LFUIApplication sharedAppDelegate] isKeyboardFirstResponder]; }
Box Application::GetTouchKeyboardBox() {
  Box ret; 
  LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
  CGRect rect = [uiapp.controller getKeyboardToolbarFrame];
  CGFloat scale = [uiapp getScale];
  return Box(scale * rect.origin.x,
             scale * (rect.origin.y + rect.size.height) - screen->height,
             scale * rect.size.width,
             scale * rect.size.height);
}

void Application::SetAutoRotateOrientation(bool v) { [LFUIApplication sharedAppDelegate].controller.autorotate = v; }
int Application::SetMultisample(bool v) { return [[LFUIApplication sharedAppDelegate] updateGLKMultisample:v]; }
int Application::SetExtraScale(bool v) { return [[LFUIApplication sharedAppDelegate] updateScale:v]; }
void Application::SetDownScale(bool v) { [[LFUIApplication sharedAppDelegate] downScale:v]; }

void Window::SetResizeIncrements(float x, float y) {}
void Window::SetTransparency(float v) {}
void Window::SetCaption(const string &v) {}
void Window::Reshape(int w, int h) {}

bool Video::CreateWindow(Window *w) { return false; }
void Video::StartWindow(Window *w) {
  [[LFUIApplication sharedAppDelegate] updateTargetFPS: w->target_fps];
}

int Video::Swap() {
  screen->gd->Flush();
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

bool FrameScheduler::DoWait() { return false; }
void FrameScheduler::Setup() { rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = 0; }
void FrameScheduler::Wakeup(Window *w) {
  dispatch_async(dispatch_get_main_queue(), ^{ [GetTyped<GLKView*>(w->id) setNeedsDisplay]; });
}

bool FrameScheduler::WakeupIn(Window*, Time interval, bool force) { return false; }
void FrameScheduler::ClearWakeupIn(Window*) {}

void FrameScheduler::UpdateWindowTargetFPS(Window *w) {
  [[LFUIApplication sharedAppDelegate] updateTargetFPS: w->target_fps];
}

void FrameScheduler::AddWaitForeverMouse(Window*) {
  [LFUIApplication sharedAppDelegate].frame_on_mouse_input = YES;
}

void FrameScheduler::DelWaitForeverMouse(Window*) {
  [LFUIApplication sharedAppDelegate].frame_on_mouse_input = NO;
}

void FrameScheduler::AddWaitForeverKeyboard(Window*) {
  [LFUIApplication sharedAppDelegate].frame_on_keyboard_input = YES;
}

void FrameScheduler::DelWaitForeverKeyboard(Window*) {
 [LFUIApplication sharedAppDelegate].frame_on_keyboard_input = NO;
}

void FrameScheduler::AddWaitForeverSocket(Window *w, Socket fd, int flag) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, w);
  if (!wait_forever_thread) {
    CHECK_EQ(SocketSet::READABLE, flag);
    [[LFUIApplication sharedAppDelegate] setWaitForeverSocket: fd];
  }
}

void FrameScheduler::DelWaitForeverSocket(Window *w, Socket fd) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
  [[LFUIApplication sharedAppDelegate] delWaitForeverSocket: fd];
}

unique_ptr<Module> CreateFrameworkModule() { return make_unique<iPhoneFrameworkModule>(); }
unique_ptr<AssetLoaderInterface> CreateAssetLoader() { return make_unique<iPhoneAssetLoader>(); }

extern "C" int main(int ac, const char* const* av) {
  MyAppCreate(ac, av);
  NSLog(@"%@", @"lfl_app_iphone_framework_main");
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  int ret = UIApplicationMain(ac, const_cast<char**>(av), nil, @"LFUIApplication");
  [pool release];
  return ret;
}

}; // namespace LFL
