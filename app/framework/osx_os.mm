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
  @property (nonatomic)         std::string  style, cancel_cmd, confirm_cmd;
@end

@implementation OSXAlert
  - (id)init:(const std::vector<std::pair<std::string, std::string>>&) kv {
    CHECK_EQ(4, kv.size());
    CHECK_EQ("style", kv[0].first);
    _style       = kv[0].second;
    _cancel_cmd  = kv[2].second;
    _confirm_cmd = kv[3].second;
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
    }
    return self;
  }

  - (void)alertDidEnd:(NSAlert *)alert returnCode:(NSInteger)returnCode contextInfo:(void *)contextInfo {
    if (_add_text) {
      ShellRun(returnCode == NSAlertFirstButtonReturn ?
               LFL::StrCat(_confirm_cmd, " ", [[_input stringValue] UTF8String]).c_str() :
               _cancel_cmd.c_str());
    } else {
      ShellRun(returnCode == NSAlertFirstButtonReturn ? _confirm_cmd.c_str() : _cancel_cmd.c_str());
    }
  }

  - (void)controlTextDidChange:(NSNotification *)notification {}
  - (void)controlTextDidEndEditing:(NSNotification *)notification {}
@end

@interface OSXPanel : NSObject
  @property(readonly, assign) NSWindow *window;
  - (void) show;
  - (void) addTextField: (NSTextField*) tf withCommand: (const std::string&) cmd;
@end

@implementation OSXPanel
  {
    std::vector<std::pair<NSButton*,    std::string>> buttons;
    std::vector<std::pair<NSTextField*, std::string>> textfields;
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

  - (void) addButton: (NSButton*) button withCommand: (const std::string&) cmd {
    [button setTarget: self];
    [button setAction: @selector(buttonAction:)];
    [button setTag: buttons.size()];
    buttons.emplace_back(button, cmd);
  }
  
  - (void) addTextField: (NSTextField*) textfield withCommand: (const std::string&) cmd {
    [textfield setTarget: self];
    [textfield setAction: @selector(textFieldAction:)];
    [textfield setTag: textfields.size()];
    textfields.emplace_back(textfield, cmd);
  }

  - (void) buttonAction: (id)sender {
    int tag = [sender tag];
    CHECK_RANGE(tag, 0, buttons.size());
    ShellRun(LFL::StrCat(buttons[tag].second, " ", [[sender stringValue] UTF8String]).c_str());
  }

  - (void) textFieldAction: (id)sender {
    int tag = [sender tag];
    CHECK_RANGE(tag, 0, textfields.size());
    ShellRun(LFL::StrCat(textfields[tag].second, " ", [[sender stringValue] UTF8String]).c_str());
  }
@end

@interface FontChooser : NSObject
@end

@implementation FontChooser
  {
    NSFont *font;
    NSString *font_change_cmd;
  }

  - (void)selectFont: (const char *)name size:(int)s cmd:(const char*)v {
    font = [NSFont fontWithName:[NSString stringWithUTF8String:name] size:s];
    font_change_cmd = [NSString stringWithUTF8String:v];
    NSFontManager *fontManager = [NSFontManager sharedFontManager];
    [fontManager setSelectedFont:font isMultiple:NO];
    [fontManager setDelegate:self];
    [fontManager setTarget:self];
    [fontManager orderFrontFontPanel:self];
  }

  - (void)changeFont:(id)sender {
    font = [sender convertFont:font];
    float size = [[[font fontDescriptor] objectForKey:NSFontSizeAttribute] floatValue];
    ShellRun([[NSString stringWithFormat:@"%@ %@ %f", font_change_cmd, [font fontName], size] UTF8String]);
  }
@end

namespace LFL {
static void AddNSMenuItems(NSMenu *menu, const vector<MenuItem>&items) {
  NSMenuItem *item;
  for (auto &i : items) { 
    const string &k=tuple_get<0>(i), &n=tuple_get<1>(i), &v=tuple_get<2>(i);
    if (n == "<seperator>") { [menu addItem:[NSMenuItem separatorItem]]; continue; }

    NSString *key = nil;
    if      (k == "<left>")  { unichar fk = NSLeftArrowFunctionKey;  key = [NSString stringWithCharacters:&fk length:1]; }
    else if (k == "<right>") { unichar fk = NSRightArrowFunctionKey; key = [NSString stringWithCharacters:&fk length:1]; }
    else if (k == "<up>"   ) { unichar fk = NSUpArrowFunctionKey;    key = [NSString stringWithCharacters:&fk length:1]; }
    else if (k == "<down>")  { unichar fk = NSDownArrowFunctionKey;  key = [NSString stringWithCharacters:&fk length:1]; }
    else key = [NSString stringWithUTF8String: k.c_str()];

    item = [menu addItemWithTitle: [NSString stringWithUTF8String: n.c_str()]
                 action:           (v[0] ? @selector(shellRun:) : nil)
                 keyEquivalent:    key];
    [item setRepresentedObject: [NSString stringWithUTF8String: v.c_str()]];
  }
}

NativeAlert::~NativeAlert() { if (auto alert = FromVoid<OSXAlert*>(impl)) [alert release]; }
NativeAlert::NativeAlert(const string &name, const vector<pair<string, string>>&items) {
  impl = [[OSXAlert alloc] init: items];
}

void NativeAlert::Show(const string &arg) {
  auto alert = FromVoid<OSXAlert*>(impl);
  if (alert.add_text) [alert.input setStringValue: [NSString stringWithUTF8String: arg.c_str()]];
  [alert.alert beginSheetModalForWindow:[GetTyped<GameView*>(screen->id) window] modalDelegate:alert
    didEndSelector:@selector(alertDidEnd:returnCode:contextInfo:) contextInfo:nil];
  [GetTyped<GameView*>(screen->id) clearKeyModifiers];
}

NativeMenu::~NativeMenu() { if (auto menu = FromVoid<NSMenu*>(impl)) [menu release]; }
NativeMenu::NativeMenu(const string &title_text, const vector<MenuItem>&items) {
  NSString *title = [NSString stringWithUTF8String: title_text.c_str()];
  NSMenu *menu = [[NSMenu alloc] initWithTitle: title];
  AddNSMenuItems(menu, items);
  NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
}

unique_ptr<NativeMenu> NativeMenu::CreateEditMenu(const vector<MenuItem>&items) {
  NSMenuItem *item; 
  NSMenu *menu = [[NSMenu alloc] initWithTitle:@"Edit"];
  item = [menu addItemWithTitle:@"Copy"  action:@selector(copy:)  keyEquivalent:@"c"];
  item = [menu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
  AddNSMenuItems(menu, items);
  item = [[NSMenuItem alloc] initWithTitle:@"Edit" action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
  return make_unique<NativeMenu>(nullptr);
}

NativePanel::~NativePanel() { if (auto panel = FromVoid<OSXPanel*>(impl)) [panel release]; }
NativePanel::NativePanel(const Box &b, const string &title, const vector<PanelItem> &items) {
  OSXPanel *panel = [[OSXPanel alloc] initWithBox: b];
  [[panel window] setTitle: [NSString stringWithUTF8String: title.c_str()]];
  for (auto &i : items) {
    const Box &b = tuple_get<1>(i);
    const string &t = tuple_get<0>(i), &cmd = tuple_get<2>(i);
    if (t == "textbox") {
      NSTextField *textfield = [[NSTextField alloc] initWithFrame:NSMakeRect(b.x, b.y, b.w, b.h)];
      [panel addTextField: textfield withCommand: cmd];
      [[[panel window] contentView] addSubview: textfield];
    } else if (PrefixMatch(t, "button:")) {
      NSButton *button = [[[NSButton alloc] initWithFrame:NSMakeRect(b.x, b.y, b.w, b.h)] autorelease];
      [panel addButton: button withCommand: cmd];
      [[[panel window] contentView] addSubview: button];
      [button setTitle: [NSString stringWithUTF8String: t.substr(7).c_str()]];
      [button setButtonType:NSMomentaryLightButton];
      [button setBezelStyle:NSRoundedBezelStyle];
    } else ERROR("unknown panel item ", t);
  }
  impl = panel;
}

void NativePanel::Show() { [FromVoid<OSXPanel*>(impl) show]; }
void NativePanel::SetTitle(const string &title) { 
  [[FromVoid<OSXPanel*>(impl) window] setTitle: [NSString stringWithUTF8String: title.c_str()]];
}

void Application::ShowNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd) {
  static FontChooser *font_chooser = [FontChooser alloc];
  [font_chooser selectFont:cur_font.name.c_str() size:cur_font.size cmd:choose_cmd.c_str()];
}

void Application::ShowNativeFileChooser(bool choose_files, bool choose_dirs, bool choose_multi,
                                          const string &choose_cmd) {
  NSOpenPanel *panel = [NSOpenPanel openPanel];
  [panel setCanChooseFiles:choose_files];
  [panel setCanChooseDirectories:choose_dirs];
  [panel setAllowsMultipleSelection:choose_multi];
  NSInteger clicked = [panel runModal];
  app->LoseFocus();
  if (clicked != NSFileHandlingPanelOKButton) return;

  string start = choose_cmd, run = start;
  for (NSURL *url in [panel URLs]) {
    run.append(" ");
    run.append([[url absoluteString] UTF8String]);
  }
  if (run.size() > start.size()) ShellRun(run.c_str());
}

void Application::ShowNativeContextMenu(const vector<MenuItem>&items) {
  NSEvent *event = [NSEvent mouseEventWithType: NSLeftMouseDown
                            location:           NSMakePoint(screen->mouse.x, screen->mouse.y)
                            modifierFlags:      NSLeftMouseDownMask
                            timestamp:          0
                            windowNumber:       [[LFL::GetTyped<GameView*>(screen->id) window] windowNumber]
                            context:            [[LFL::GetTyped<GameView*>(screen->id) window] graphicsContext]
                            eventNumber:        0
                            clickCount:         1
                            pressure:           1];

  NSMenuItem *item;
  NSMenu *menu = [[NSMenu alloc] init];
  for (auto &i : items) {
    const char *k = tuple_get<0>(i).c_str(), *n = tuple_get<1>(i).c_str(), *v = tuple_get<2>(i).c_str();
    if (!strcmp(n, "<seperator>")) { [menu addItem:[NSMenuItem separatorItem]]; continue; }
    item = [menu addItemWithTitle: [NSString stringWithUTF8String: n]
                 action:           (v[0] ? @selector(shellRun:) : nil)
                 keyEquivalent:    [NSString stringWithUTF8String: k]];
    [item setRepresentedObject: [NSString stringWithUTF8String: v]];
  }

  [NSMenu popUpContextMenu:menu withEvent:event forView:LFL::GetTyped<GameView*>(screen->id)];
  [LFL::GetTyped<GameView*>(screen->id) clearKeyModifiers];
}

void Application::OpenSystemBrowser(const string &url_text) {
  CFURLRef url = CFURLCreateWithBytes(0, MakeUnsigned(url_text.c_str()), url_text.size(), kCFStringEncodingASCII, 0);
  if (url) { LSOpenCFURLRef(url, 0); CFRelease(url); }
}

void Application::ShowAds() {}
void Application::HideAds() {}

}; // namespace LFL
