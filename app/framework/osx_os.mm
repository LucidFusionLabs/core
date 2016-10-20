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
#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/osx_common.h"

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

@interface FontChooser : NSObject
@end

@implementation FontChooser
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
  [FromVoid<OSXAlert*>(impl).alert setMessageText: MakeNSString(title)];
  [FromVoid<OSXAlert*>(impl).alert setInformativeText: MakeNSString(msg)];
  string ret = RunModal(arg);
  if (confirm_cb) confirm_cb(move(ret));
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

SystemAdvertisingView::SystemAdvertisingView() {}
void SystemAdvertisingView::Show() {}
void SystemAdvertisingView::Hide() {}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &choose_cb) {
  static FontChooser *font_chooser = [FontChooser alloc];
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

string Application::GetSystemDeviceName() {
  string ret(1024, 0);
  if (gethostname(&ret[0], ret.size())) return "";
  ret.resize(strlen(ret.c_str()));
  return ret;
}

}; // namespace LFL
