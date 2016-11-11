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
#import <QuartzCore/QuartzCore.h>
#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/osx_common.h"

static std::vector<NSImage*> app_images;

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
    [_alert setAlertStyle:NSWarningAlertStyle];
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
    NSWindow *window = [LFL::GetTyped<GameView*>(LFL::app->focused->id) window];
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
      styleMask:NSTitledWindowMask | NSClosableWindowMask 
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

@interface OSXToolbar : NSObject
  @property (nonatomic, retain) NSToolbar *toolbar;
  + (int)getBottomHeight;
@end

@implementation OSXToolbar
  + (int)getBottomHeight { return 0; }
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
    [fontManager setDelegate:self];
    [fontManager setTarget:self];
    [fontManager orderFrontFontPanel:self];
  }

  - (void)changeFont:(id)sender {
    font = [sender convertFont:font];
    float size = [[[font fontDescriptor] objectForKey:NSFontSizeAttribute] floatValue];
    font_change_cb(LFL::StringVec{ [[font fontName] UTF8String], LFL::StrCat(size) });
  }
@end

@interface OSXNavigation : NSResponder
  @property (nonatomic, retain) NSViewController *rootViewController;
  @property (nonatomic, retain) NSMutableArray *viewControllers;
@end

@implementation OSXNavigation
  {
    CATransition *transition;
  }

  - (id)init { 
    self = [super init];
    self.viewControllers = [[NSMutableArray alloc] init];
    transition = [CATransition animation];
    [transition setType:kCATransitionPush];
    [transition setSubtype:kCATransitionFromRight];
    NSDictionary *animations = [NSDictionary dictionaryWithObject:transition forKey:@"subviews"];
    NSView *contentView = LFL::GetTyped<GameView*>(LFL::app->focused->id).window.contentView;
    [contentView setAnimations:animations];
    return self;
  }

  - (void)pushViewController:(NSViewController*)vc animated:(BOOL)anim {
    [self popView];
    [self.viewControllers addObject: vc];
    [self pushView];
  }

  - (NSViewController*)popViewControllerAnimated:(BOOL)anim {
    int count = [self.viewControllers count];
    if (!count) return nil;
    [transition setSubtype:kCATransitionFromLeft];
    NSViewController *topViewController = [self.viewControllers objectAtIndex: count-1];
    [self popView];
    [self.viewControllers removeLastObject];
    [self pushView];
    return topViewController;
  }

  - (void)pushView {
    int count = [self.viewControllers count];
    CHECK(count);
    NSViewController *topViewController = [self.viewControllers objectAtIndex: count-1];
    NSView *contentView = LFL::GetTyped<GameView*>(LFL::app->focused->id).window.contentView;
    [[contentView animator] addSubview:topViewController.view];
    [topViewController.view setFrameOrigin:CGPointMake(0, 0)];
  }

  - (void)popView {
    int count = [self.viewControllers count];
    if (!count) return;
    NSViewController *topViewController = [self.viewControllers objectAtIndex: count-1];
    [topViewController.view removeFromSuperview];
  }
@end

@interface OSXTable : NSViewController<NSTableViewDataSource, NSTableViewDelegate>
  @property (nonatomic, retain) NSTableView *tableView;
  @property (nonatomic, assign) LFL::SystemTableView *lfl_self;
@end

@implementation OSXTable
  - (void)loadView {
    NSView *contentView = LFL::GetTyped<GameView*>(LFL::app->focused->id).window.contentView;
    self.view = [[NSView alloc] initWithFrame: contentView.frame];
    [self.view addSubview:self.tableView];
  }

  - (id)init: (LFL::SystemTableView*)lself withTitle:(const std::string&)title andStyle:(const std::string&)style { 
    self = [super init];
    NSView *contentView = LFL::GetTyped<GameView*>(LFL::app->focused->id).window.contentView;
    self.tableView = [[NSTableView alloc] initWithFrame: contentView.frame];
    [self.tableView setDelegate:self];
    [self.tableView setDataSource:self];

    NSTableColumn * column1 = [[NSTableColumn alloc] initWithIdentifier:@"Col1"];
    NSTableColumn * column2 = [[NSTableColumn alloc] initWithIdentifier:@"Col2"];
    [column1 setWidth:252];
    [column2 setWidth:198];
    [self.tableView addTableColumn:column1];
    [self.tableView addTableColumn:column2];
    [column1 release];
    [column2 release];
    return self;
  }

  - (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView { return 0; }
  - (id)tableView:(NSTableView *)tableView objectValueForTableColumn:(NSTableColumn *)tableColumn row:(NSInteger)row {
    return nil; 
  }
@end

@interface OSXTextView : NSViewController<NSTextViewDelegate>
  @property (nonatomic, retain) NSTextView *textView;
  @property (nonatomic, retain) NSString *text;
@end

@implementation OSXTextView
  - (id)initWithTitle:(NSString*)title andText:(NSString*)t {
    self = [super init];
    self.title = title;
    self.text = t;
    return self;
  }
@end

namespace LFL {
static void AddNSMenuItems(NSMenu *menu, vector<MenuItem> items) {
  NSMenuItem *item;
  for (auto &i : items) { 
    const string &k=i.shortcut, &n=i.name;
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

SystemAlertView::~SystemAlertView() { if (auto alert = FromVoid<OSXAlert*>(impl)) [alert release]; }
SystemAlertView::SystemAlertView(AlertItemVec items) { impl = [[OSXAlert alloc] init: move(items)]; }

void SystemAlertView::Show(const string &arg) {
  auto alert = FromVoid<OSXAlert*>(impl);
  if (alert.add_text) [alert.input setStringValue: MakeNSString(arg)];
  [alert.alert beginSheetModalForWindow:[GetTyped<GameView*>(app->focused->id) window] modalDelegate:alert
    didEndSelector:@selector(alertDidEnd:returnCode:contextInfo:) contextInfo:nil];
  [GetTyped<GameView*>(app->focused->id) clearKeyModifiers];
}

void SystemAlertView::ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
  auto alert = FromVoid<OSXAlert*>(impl);
  alert.confirm_cb = move(confirm_cb);
  [alert.alert setMessageText: MakeNSString(title)];
  [alert.alert setInformativeText: MakeNSString(msg)];
  [alert.alert beginSheetModalForWindow:[GetTyped<GameView*>(app->focused->id) window] modalDelegate:alert
    didEndSelector:@selector(alertDidEnd:returnCode:contextInfo:) contextInfo:nil];
  [GetTyped<GameView*>(app->focused->id) clearKeyModifiers];
}

string SystemAlertView::RunModal(const string &arg) {
  auto alert = FromVoid<OSXAlert*>(impl);
  NSWindow *window = [GetTyped<GameView*>(app->focused->id) window];
  if (alert.add_text) [alert.input setStringValue: MakeNSString(arg)];
  [alert.alert beginSheetModalForWindow:window modalDelegate:alert
    didEndSelector:@selector(modalAlertDidEnd:returnCode:contextInfo:) contextInfo:nil];
  [NSApp runModalForWindow: window];
  return alert.add_text ? GetNSString([alert.input stringValue]) : "";
}

SystemMenuView::~SystemMenuView() { if (auto menu = FromVoid<NSMenu*>(impl)) [menu release]; }
SystemMenuView::SystemMenuView(const string &title_text, MenuItemVec items) {
  NSString *title = [NSString stringWithUTF8String: title_text.c_str()];
  NSMenu *menu = [[NSMenu alloc] initWithTitle: title];
  AddNSMenuItems(menu, move(items));
  NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
}

void SystemMenuView::Show() {}
unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(MenuItemVec items) {
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
  return make_unique<SystemMenuView>(nullptr);
}

SystemPanelView::~SystemPanelView() { if (auto panel = FromVoid<OSXPanel*>(impl)) [panel release]; }
SystemPanelView::SystemPanelView(const Box &b, const string &title, PanelItemVec items) {
  OSXPanel *panel = [[OSXPanel alloc] initWithBox: b];
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
  impl = panel;
}

void SystemPanelView::Show() { [FromVoid<OSXPanel*>(impl) show]; }
void SystemPanelView::SetTitle(const string &title) { 
  [[FromVoid<OSXPanel*>(impl) window] setTitle: [NSString stringWithUTF8String: title.c_str()]];
}

SystemToolbarView::~SystemToolbarView() {}
SystemToolbarView::SystemToolbarView(MenuItemVec items) : impl(0) {}
void SystemToolbarView::Show(bool show_or_hide) {}
void SystemToolbarView::ToggleButton(const string &n) {}

SystemTableView::~SystemTableView() { if (auto table = FromVoid<OSXTable*>(impl)) [table release]; }
SystemTableView::SystemTableView(const string &title, const string &style, TableItemVec items, int second_col) :
  impl([[OSXTable alloc] init:this withTitle:title andStyle:style]) {}

void SystemTableView::DelNavigationButton(int align) {}
void SystemTableView::AddNavigationButton(int align, const TableItem &item) {}
void SystemTableView::AddToolbar(SystemToolbarView *t) {
}

void SystemTableView::Show(bool show_or_hide) {
}

void SystemTableView::AddRow(int section, TableItem item) {}
string SystemTableView::GetKey(int section, int row) { return ""; }
int SystemTableView::GetTag(int section, int row) { return 0; }
void SystemTableView::SetTag(int section, int row, int val) {}
void SystemTableView::SetValue(int section, int row, const string &val) {}
void SystemTableView::SetHidden(int section, int row, bool val) {}
void SystemTableView::SetTitle(const string &title) {}
PickerItem *SystemTableView::GetPicker(int section, int row) { return 0; }
StringPairVec SystemTableView::GetSectionText(int section) { return StringPairVec(); }
void SystemTableView::SetEditableSection(int section, int start_row, LFL::IntIntCB cb) {
}

void SystemTableView::SelectRow(int section, int row) {
} 

void SystemTableView::BeginUpdates() {}
void SystemTableView::EndUpdates() {}
void SystemTableView::SetDropdown(int section, int row, int val) {}
void SystemTableView::SetSectionValues(int section, const StringVec &item) {}
void SystemTableView::ReplaceSection(int section, const string &h, int image, int flag, TableItemVec item, Callback add_button) {}

SystemTextView::~SystemTextView() { if (auto view = FromVoid<OSXTextView*>(impl)) [view release]; }
SystemTextView::SystemTextView(const string &title, File *f) : SystemTextView(title, f ? f->Contents() : "") {}
SystemTextView::SystemTextView(const string &title, const string &text) :
  impl([[OSXTextView alloc] initWithTitle:MakeNSString(title) andText:[[[NSString alloc]
       initWithBytes:text.data() length:text.size() encoding:NSASCIIStringEncoding] autorelease]]) {}

SystemNavigationView::~SystemNavigationView() { if (auto nav = FromVoid<OSXNavigation*>(impl)) [nav release]; }
SystemNavigationView::SystemNavigationView() : impl([[OSXNavigation alloc] init]) {}
void SystemNavigationView::Show(bool show_or_hide) {
  auto nav = FromVoid<OSXNavigation*>(impl);
  NSWindow *window = [GetTyped<GameView*>(app->focused->id) window];
  if ((shown = show_or_hide)) {
    if (root->show_cb) root->show_cb();
    // [window.windowController presentViewController:nav animator:nil];
  } else {
    // [window.windowController dismissViewController:nav];
  }
}

SystemTableView *SystemNavigationView::Back() {
  for (NSViewController *c in [FromVoid<OSXNavigation*>(impl).viewControllers reverseObjectEnumerator]) {
    if ([c isKindOfClass:[OSXTable class]])
      if (auto lself = static_cast<OSXTable*>(c).lfl_self) return lself;
  } 
  return nullptr;
}

void SystemNavigationView::PushTableView(SystemTableView *t) {
  if (!root) root = t;
  if (t->show_cb) t->show_cb();
  [FromVoid<OSXNavigation*>(impl) pushViewController: FromVoid<OSXTable*>(t->impl) animated: YES];
}

void SystemNavigationView::PushTextView(SystemTextView *t) {
  if (t->show_cb) t->show_cb();
  [FromVoid<OSXNavigation*>(impl) pushViewController: FromVoid<OSXTextView*>(t->impl) animated: YES];
}

void SystemNavigationView::PopToRoot() {}
void SystemNavigationView::PopAll() {}

void SystemNavigationView::PopView(int n) {
  for (int i = 0; i != n; ++i)
    [FromVoid<OSXNavigation*>(impl) popViewControllerAnimated: (i == n - 1)];
}

SystemAdvertisingView::SystemAdvertisingView() {}
void SystemAdvertisingView::Show() {}
void SystemAdvertisingView::Hide() {}

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
  app->LoseFocus();
  if (clicked != NSFileHandlingPanelOKButton) return;

  StringVec arg;
  for (NSURL *url in [panel URLs]) arg.emplace_back([[url absoluteString] UTF8String]);
  if (arg.size()) choose_cb(arg);
}

void Application::ShowSystemContextMenu(const MenuItemVec &items) {
  NSEvent *event = [NSEvent mouseEventWithType: NSLeftMouseDown
                            location:           NSMakePoint(app->focused->mouse.x, app->focused->mouse.y)
                            modifierFlags:      NSLeftMouseDownMask
                            timestamp:          0
                            windowNumber:       [[LFL::GetTyped<GameView*>(app->focused->id) window] windowNumber]
                            context:            [[LFL::GetTyped<GameView*>(app->focused->id) window] graphicsContext]
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

  [NSMenu popUpContextMenu:menu withEvent:event forView:LFL::GetTyped<GameView*>(app->focused->id)];
  [LFL::GetTyped<GameView*>(app->focused->id) clearKeyModifiers];
}

void Application::OpenSystemBrowser(const string &url_text) {
  CFURLRef url = CFURLCreateWithBytes(0, MakeUnsigned(url_text.c_str()), url_text.size(), kCFStringEncodingASCII, 0);
  if (url) { LSOpenCFURLRef(url, 0); CFRelease(url); }
}

int Application::LoadSystemImage(const string &n) {
  NSImage *image = [[NSImage alloc] initWithContentsOfFile: MakeNSString(StrCat(app->assetdir, "../", n)) ];
  if (!image) return 0;
  app_images.push_back(image);
  return app_images.size();
}

string Application::GetSystemDeviceName() {
  string ret(1024, 0);
  if (gethostname(&ret[0], ret.size())) return "";
  ret.resize(strlen(ret.c_str()));
  return ret;
}

Connection *Application::ConnectTCP(const string &hostport, int default_port, Connection::CB *connected_cb, bool background_services) {
#if 1
  INFO("Application::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = ", background_services);
  if (background_services) return new NSURLSessionStreamConnection(hostport, default_port, connected_cb ? move(*connected_cb) : Connection::CB());
  else return app->net->tcp_client->Connect(hostport, default_port, connected_cb);
#else
  INFO("Application::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = false"); 
  return app->net->tcp_client->Connect(hostport, default_port, connected_cb);
#endif
}

}; // namespace LFL
