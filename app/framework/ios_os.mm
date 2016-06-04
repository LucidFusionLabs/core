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
#import <GLKit/GLKit.h>

#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/ios_common.h"

@interface NativeMenu : NSObject
@end

@implementation NativeMenu
  + (void)addMenu:(const char*)title_text items:(const std::vector<LFL::MenuItem>&)item {
    NSString *title = [NSString stringWithUTF8String: title_text];
    menu_tags[[title hash]] = title_text;
    auto menu = &menus[title_text];
    for (auto &i : item) menu->emplace_back(tuple_get<1>(i), tuple_get<2>(i)); 
  }

  + (void)launchMenu:(const char*)title_text {
    auto it = menus.find(title_text);
    if (it == menus.end()) { ERRORf("unknown menu: %s", title_text); return; }
    NSString *title = [NSString stringWithUTF8String: title_text];
    UIActionSheet *actions = [[UIActionSheet alloc] initWithTitle:title delegate:self
      cancelButtonTitle:@"Cancel" destructiveButtonTitle:nil otherButtonTitles:nil];
    for (auto &i : it->second) [actions addButtonWithTitle:[NSString stringWithUTF8String: i.first.c_str()]];
    actions.tag = [title hash];
    [actions showInView:[UIApplication sharedApplication].keyWindow];
    [actions release];
  }

  + (void)actionSheet:(UIActionSheet *)actions clickedButtonAtIndex:(NSInteger)buttonIndex {
    auto tag_it = menu_tags.find(actions.tag);
    if (tag_it == menu_tags.end()) { ERRORf("unknown tag: %d", actions.tag); return; }
    auto it = menus.find(tag_it->second);
    if (it == menus.end()) { ERRORf("unknown menu: %s", tag_it->second.c_str()); return; }
    if (buttonIndex < 1 || buttonIndex > it->second.size()) { ERRORf("invalid buttonIndex %d size=%d", buttonIndex, it->second.size()); return; }
    ShellRun(it->second[buttonIndex-1].second.c_str());
  }

  static std::unordered_map<int, std::string> menu_tags;
  static std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> menus;
@end

@interface NativePicker : NSObject<UIPickerViewDelegate>
  {
    std::vector<std::vector<std::string>> columns;
    std::vector<int> picked_row;
  }
  @property (nonatomic, retain) UIPickerView *picker;
  - (bool)didSelect;
@end

@implementation NativePicker
  - (id)init {
    self = [super init];
    _picker = [[UIPickerView alloc] init];
    _picker.delegate = self;
    _picker.showsSelectionIndicator = YES;
    _picker.hidden = NO;
    _picker.layer.borderColor = [UIColor grayColor].CGColor;
    _picker.layer.borderWidth = 4;
    [_picker setBackgroundColor:[UIColor whiteColor]];
    return self;
  }

  - (void)pickerView:(UIPickerView *)pV didSelectRow:(NSInteger)row inComponent:(NSInteger)component {
    CHECK_RANGE(component, 0, columns.size());
    if (picked_row.size() != columns.size()) picked_row.resize(columns.size());
    picked_row[component] = row;
    if ([self didSelect]) [_picker removeFromSuperview];
  }

  - (bool)didSelect { return false; }
  - (NSInteger)numberOfComponentsInPickerView:(UIPickerView *)pickerView { return columns.size(); }
  - (NSInteger)pickerView:(UIPickerView *)pickerView numberOfRowsInComponent:(NSInteger)component { 
    CHECK_RANGE(component, 0, columns.size());
    return columns[component].size();
  }

  - (NSString *)pickerView:(UIPickerView *)pickerView titleForRow:(NSInteger)row forComponent:(NSInteger)component {
    CHECK_RANGE(component, 0, columns.size());
    CHECK_RANGE(row, 0, columns[component].size());
    return [NSString stringWithUTF8String: columns[component][row].c_str()];
  }
@end

@interface FontChooser : NativePicker
@end

@implementation FontChooser
  {
    std::string font_change_cmd;
  }

  - (id)init {
    self = [super init];
    columns.push_back({});
    NSArray *families = [UIFont familyNames];
    for (NSString *family_name in families) {
      NSArray *fonts = [UIFont fontNamesForFamilyName:family_name];
      for (NSString *font_name in fonts) columns.back().push_back([font_name UTF8String]);
    }
    columns.push_back({});
    for (int i=0; i<64; ++i) columns.back().push_back(LFL::StrCat(i+1));
    return self;
  }

  - (void)selectFont: (const std::string &)name size:(int)s cmd:(const std::string &)v {
    font_change_cmd = v;
    if (picked_row.size() != columns.size()) picked_row.resize(columns.size());
    for (auto b = columns[0].begin(), e = columns[0].end(), i = b; i != e; ++i)
      if (*i == name) picked_row[0] = i - b;
    picked_row[1] = LFL::Clamp(s-1, 1, 64);
    [self.picker selectRow:picked_row[0] inComponent:0 animated:NO];
    [self.picker selectRow:picked_row[1] inComponent:1 animated:NO];
  }

  - (bool)didSelect {
    ShellRun(LFL::StrCat(font_change_cmd, " ",
                         columns[0][picked_row[0]], " ", columns[1][picked_row[1]]).c_str());
    return true;
  }
@end

@interface NativeKeychain : NSObject
  + (void)save:(NSString *)service data:(id)data;
  + (id)load:(NSString *)service;
@end

@implementation NativeKeychain
  + (NSMutableDictionary *)getKeychainQuery:(NSString *)service {
    return [NSMutableDictionary dictionaryWithObjectsAndKeys:(id)kSecClassGenericPassword, (id)kSecClass, service,
           (id)kSecAttrService, service, (id)kSecAttrAccount, (id)kSecAttrAccessibleAfterFirstUnlock, (id)kSecAttrAccessible, nil];
  }

  + (void)save:(NSString *)service data:(id)data {
    NSMutableDictionary *keychainQuery = [self getKeychainQuery:service];
    SecItemDelete((CFDictionaryRef)keychainQuery);
    [keychainQuery setObject:[NSKeyedArchiver archivedDataWithRootObject:data] forKey:(id)kSecValueData];
    SecItemAdd((CFDictionaryRef)keychainQuery, NULL);
  }

  + (id)load:(NSString *)service {
    id ret = nil;
    NSMutableDictionary *keychainQuery = [self getKeychainQuery:service];
    [keychainQuery setObject:(id)kCFBooleanTrue forKey:(id)kSecReturnData];
    [keychainQuery setObject:(id)kSecMatchLimitOne forKey:(id)kSecMatchLimit];
    CFDataRef keyData = NULL;
    if (SecItemCopyMatching((CFDictionaryRef)keychainQuery, (CFTypeRef *)&keyData) == noErr) {
      @try { ret = [NSKeyedUnarchiver unarchiveObjectWithData:(NSData *)keyData]; }
      @catch (NSException *e) { NSLog(@"Unarchive of %@ failed: %@", service, e); }
      @finally {}
    }
    if (keyData) CFRelease(keyData);
    return ret;
  }
@end

namespace LFL {
void Application::AddNativeEditMenu(const vector<MenuItem>&items) {}
void Application::AddNativeMenu(const string &title, const vector<MenuItem>&items) {
  [NativeMenu addMenu:title.c_str() items:items];
}

void Application::LaunchNativeMenu(const string &title) {
  [NativeMenu launchMenu:title.c_str()];
}

void Application::LaunchNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd) {
  static FontChooser *font_chooser = [[FontChooser alloc] init];
  [font_chooser selectFont:cur_font.name size:cur_font.size cmd:choose_cmd];
  [[LFUIApplication sharedAppDelegate].view addSubview: font_chooser.picker];
}

void Application::OpenSystemBrowser(const string &url_text) {
  NSString *url_string = [[NSString alloc] initWithUTF8String: url_text.c_str()];
  NSURL *url = [NSURL URLWithString: url_string];
  [[UIApplication sharedApplication] openURL:url];
  [url_string release];
}

void Application::SavePassword(const string &h, const string &u, const string &pw_in) {
  NSString *k = [[NSString stringWithFormat:@"%s://%s@%s", name.c_str(), u.c_str(), h.c_str()] retain];
  NSMutableString *pw = [[NSMutableString stringWithUTF8String: pw_in.c_str()] retain];
  UIAlertController *alertController = [UIAlertController alertControllerWithTitle:@"LTerminal Keychain"
    message:[NSString stringWithFormat:@"Save password for %s@%s?", u.c_str(), h.c_str()] preferredStyle:UIAlertControllerStyleAlert];
  UIAlertAction *actionNo  = [UIAlertAction actionWithTitle:@"No"  style:UIAlertActionStyleDefault handler: nil];
  UIAlertAction *actionYes = [UIAlertAction actionWithTitle:@"Yes" style:UIAlertActionStyleDefault
    handler:^(UIAlertAction *){
      [NativeKeychain save:k data:pw];
      [pw replaceCharactersInRange:NSMakeRange(0, [pw length]) withString:[NSString stringWithFormat:@"%*s", [pw length], ""]];
      [pw release];
      [k release];
    }];
  [alertController addAction:actionYes];
  [alertController addAction:actionNo];
  [[LFUIApplication sharedAppDelegate].controller presentViewController:alertController animated:YES completion:nil];
}

bool Application::LoadPassword(const string &h, const string &u, string *pw_out) {
  NSString *k  = [NSString stringWithFormat:@"%s://%s@%s", name.c_str(), u.c_str(), h.c_str()];
  NSString *pw = [NativeKeychain load: k];
  if (pw) pw_out->assign([pw UTF8String]);
  else    pw_out->clear();
  return  pw_out->size();
}

void Application::ShowAds() {}
void Application::HideAds() {}

}; // namespace LFL
