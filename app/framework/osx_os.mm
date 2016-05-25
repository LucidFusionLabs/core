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

@interface NativePanel : NSObject
  @property(readonly, assign) NSWindow *window;
  - (void) show;
  - (void) addTextField: (NSTextField*) tf withCommand: (const std::string&) cmd;
@end

@implementation NativePanel
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

  + (NativePanel*) addPanelNamed: (const std::string&) n withBox: (const LFL::Box&) b {
    if (panels.find(n) != panels.end()) return nullptr;
    NativePanel *ret = [[NativePanel alloc] initWithBox: b];
    panels[n] = ret;
    return ret;
  }

  + (bool) showPanelNamed: (const std::string&) n {
    auto it = panels.find(n);
    if (it == panels.end()) return false;
    [it->second show];
    return true;
  }

  + (bool) setPanelTitle: (const std::string&) title withName: (const std::string&) n {
    auto it = panels.find(n);
    if (it == panels.end()) return false;
    [[it->second window] setTitle: [NSString stringWithUTF8String: title.c_str()]];
    return true;
  }

  static std::unordered_map<std::string, NativePanel*> panels;
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
string GetNSDocumentDirectory() {
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
  return [[paths objectAtIndex:0] UTF8String];
}

static void AddNSMenuItems(NSMenu *menu, const vector<MenuItem>&items) {
  NSMenuItem *item;
  for (auto &i : items) { 
    const char *k=tuple_get<0>(i).c_str(), *n=tuple_get<1>(i).c_str(), *v=tuple_get<2>(i).c_str();
    if (!strcmp(n, "<seperator>")) { [menu addItem:[NSMenuItem separatorItem]]; continue; }
    item = [menu addItemWithTitle: [NSString stringWithUTF8String: n]
                 action:           (v[0] ? @selector(shellRun:) : nil)
                 keyEquivalent:    [NSString stringWithUTF8String: k]];
    [item setRepresentedObject: [NSString stringWithUTF8String: v]];
  }
}

void Application::AddNativeMenu(const string &title_text, const vector<MenuItem>&items) {
  NSString *title = [NSString stringWithUTF8String: title_text.c_str()];
  NSMenu *menu = [[NSMenu alloc] initWithTitle: title];
  AddNSMenuItems(menu, items);
  NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
}

void Application::AddNativeEditMenu(const vector<MenuItem>&items) {
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
}

void Application::LaunchNativePanel(const string &n) { [NativePanel showPanelNamed: n]; }
void Application::SetNativePanelTitle(const string &n, const string &title) { [NativePanel setPanelTitle: title withName: n]; }
void Application::AddNativePanel(const string &name, const Box &b, const string &title, const vector<PanelItem> &items) {
  NativePanel *panel = [NativePanel addPanelNamed: name withBox: b];
  CHECK(panel);
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
}

void Application::LaunchNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd) {
  static FontChooser *font_chooser = [FontChooser alloc];
  [font_chooser selectFont:cur_font.name.c_str() size:cur_font.size cmd:choose_cmd.c_str()];
}

void Application::LaunchNativeFileChooser(bool choose_files, bool choose_dirs, bool choose_multi,
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

void Application::OpenSystemBrowser(const string &url_text) {
  CFURLRef url = CFURLCreateWithBytes(0, MakeUnsigned(url_text.c_str()), url_text.size(), kCFStringEncodingASCII, 0);
  if (url) { LSOpenCFURLRef(url, 0); CFRelease(url); }
}

}; // namespace LFL
