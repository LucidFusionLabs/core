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

@interface NativeAlert : NSObject<UIAlertViewDelegate>
  @property (nonatomic, retain) UIAlertView *alert;
@end

@implementation NativeAlert
  {
    bool add_text;
    std::string style, cancel_cmd, confirm_cmd;
  }

  - (id)init:(const std::vector<std::pair<std::string, std::string>>&) kv {
    CHECK_EQ(4, kv.size());
    CHECK_EQ("style", kv[0].first);
    style       = kv[0].second;
    cancel_cmd  = kv[2].second;
    confirm_cmd = kv[3].second;
    _alert      = [[UIAlertView alloc]
      initWithTitle:     [NSString stringWithUTF8String: kv[1].first .c_str()]
      message:           [NSString stringWithUTF8String: kv[1].second.c_str()]
      delegate:          self
      cancelButtonTitle: [NSString stringWithUTF8String: kv[2].first.c_str()]
      otherButtonTitles: [NSString stringWithUTF8String: kv[3].first.c_str()], nil];
    if ((add_text = style == "textinput")) _alert.alertViewStyle = UIAlertViewStylePlainTextInput;
    return self;
  }

  - (void)alertView:(UIAlertView *)alertView clickedButtonAtIndex:(NSInteger)buttonIndex {
    if (add_text) {
      ShellRun(buttonIndex ?
               LFL::StrCat(confirm_cmd, " ", [[alertView textFieldAtIndex:0].text UTF8String]).c_str() :
               cancel_cmd.c_str());
    } else {
      ShellRun(buttonIndex ? confirm_cmd.c_str() : cancel_cmd.c_str());
    }
  }

  + (void)addAlert:(const std::string&)name items:(const std::vector<std::pair<std::string, std::string>>&) kv {
    alerts[name] = [[NativeAlert alloc] init: kv];
  }

  + (void)showAlert:(const std::string&)name arg:(const std::string&)a {
    auto alert = alerts[name];
    [alert.alert show];
    if (alert->add_text) [alert.alert textFieldAtIndex:0].text = [NSString stringWithUTF8String: a.c_str()];
  }

  static std::unordered_map<std::string, NativeAlert*> alerts;
@end

@interface NativeMenu : NSObject
@end

@implementation NativeMenu
  + (void)addMenu:(const std::string&)title_text items:(const std::vector<LFL::MenuItem>&)item {
    NSString *title = [NSString stringWithUTF8String: title_text.c_str()];
    menu_tags[[title hash]] = title_text;
    auto menu = &menus[title_text];
    for (auto &i : item) menu->emplace_back(tuple_get<1>(i), tuple_get<2>(i)); 
  }

  + (void)launchMenu:(const std::string&)title_text {
    auto it = menus.find(title_text);
    if (it == menus.end()) { ERRORf("unknown menu: %s", title_text.c_str()); return; }
    NSString *title = [NSString stringWithUTF8String: title_text.c_str()];
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

@implementation NativeToolbar
  {
    UIToolbar *toolbar;
    int toolbar_height;
    std::unordered_map<std::string, void*> toolbar_titles;
    std::unordered_map<void*, std::string> toolbar_cmds;
  }

  - (id)init: (const std::vector<std::pair<std::string, std::string>>&) kv {
    NSMutableArray *items = [[NSMutableArray alloc] init];
    UIBarButtonItem *spacer = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemFlexibleSpace target:nil action:nil];
    for (int i=0, l=kv.size(); i<l; i++) {
      if (i) [items addObject: spacer];
      NSString *K = [NSString stringWithUTF8String: kv[i].first.c_str()];
      UIBarButtonItem *item =
        [[UIBarButtonItem alloc] initWithTitle:[NSString stringWithFormat:@"%@\U0000FE0E", K]
        style:UIBarButtonItemStylePlain target:self action:@selector(onClick:)];
      [items addObject:item];
      toolbar_titles[kv[i].first] = item;
      toolbar_cmds[item] = kv[i].second;
      [item release];
    }
    toolbar_height = 30;
    toolbar = [[UIToolbar alloc] initWithFrame: [self getToolbarFrame]];
    // [toolbar setBarStyle:UIBarStyleBlackTranslucent];
    [toolbar setItems:items];
    [items release];
    [spacer release];
    return self;
  }

  - (CGRect)getToolbarFrame {
    CGRect bounds = [[UIScreen mainScreen] bounds], kbd = [[LFUIApplication sharedAppDelegate] getKeyboardFrame];
    return CGRectMake(0, bounds.size.height - kbd.size.height - toolbar_height, bounds.size.width, toolbar_height);
  }

  - (void)toggleButton:(id)sender {
    if (![sender isKindOfClass:[UIBarButtonItem class]]) FATALf("unknown sender: %p", sender);
    UIBarButtonItem *item = (UIBarButtonItem*)sender;
    if (item.style != UIBarButtonItemStyleDone) { item.style = UIBarButtonItemStyleDone;     item.tintColor = [UIColor colorWithRed:0.8 green:0.8 blue:0.8 alpha:.8]; }
    else                                        { item.style = UIBarButtonItemStyleBordered; item.tintColor = nil; }
  }

  - (void)onClick:(id)sender {
    auto it = toolbar_cmds.find(sender);
    if (it != toolbar_cmds.end()) {
      ShellRun(it->second.c_str());
      if (it->second.substr(0,6) == "toggle") [self toggleButton:sender];
    }
    [[LFUIApplication sharedAppDelegate].controller resignFirstResponder];
  }

  + (void)addToolbar:(const std::string&)name items:(const std::vector<std::pair<std::string, std::string>>&)kv {
    toolbars[name] = [[NativeToolbar alloc] init: kv];
  }

  + (void)showToolbar:(const std::string&)name {
    auto toolbar = toolbars[name];
    show_bottom.push_back(toolbar);
    [[LFUIApplication sharedAppDelegate].window addSubview: toolbar->toolbar];
  }

  + (int)getBottomHeight {
    int ret = 0;
    for (auto t : show_bottom) ret += t->toolbar_height;
    return ret;
  }

  + (void)updateFrame {
    for (auto t : show_bottom) t->toolbar.frame = [t getToolbarFrame];
  }

  + (void)toggleToolbarButton:(const std::string&)name withTitle:(const std::string&)k {
    auto toolbar = toolbars[name];
    auto it = toolbar->toolbar_titles.find(k);
    if (it != toolbar->toolbar_titles.end()) [toolbar toggleButton: (id)(UIBarButtonItem*)it->second];
  }

  static std::unordered_map<std::string, NativeToolbar*> toolbars;
  static std::vector<NativeToolbar*> show_bottom, show_top;
@end

@interface NativeTable : NSObject<UITableViewDelegate, UITableViewDataSource>
  {
    std::vector<std::string> rows;
  }
  @property (nonatomic, retain) UITableView *table;
  @property (nonatomic, retain) UIView *header;
  @property (nonatomic, retain) UILabel *header_label;
@end

@implementation NativeTable
  {
    int section_index;
    std::vector<std::vector<LFL::MenuItem>> data;
  }

  - (id)init: (const std::string&)title items:(const std::vector<LFL::MenuItem>&)item {
    data.emplace_back();
    for (auto i : item) {
      if (tuple_get<0>(i) == "<seperator>") {
        data.emplace_back();
        section_index++;
      } else {
        data[section_index].push_back(i);
      }
    }

    self = [super init];
    _table = [[UITableView alloc] initWithFrame: [LFUIApplication sharedAppDelegate].view.bounds
              style:UITableViewStyleGrouped];
    _table.delegate = self;
    _table.dataSource = self;
    _table.separatorInset = UIEdgeInsetsZero;
    [_table setSeparatorStyle:UITableViewCellSeparatorStyleSingleLine];
    [_table setSeparatorColor:[UIColor blackColor]];

    _header_label = [[UILabel alloc] initWithFrame:CGRectZero];
    UIFontDescriptor *font_desc = [_header_label.font.fontDescriptor
      fontDescriptorWithSymbolicTraits:UIFontDescriptorTraitBold];
    _header_label.font = [UIFont fontWithDescriptor:font_desc size:0];
    _header_label.text = [NSString stringWithUTF8String: title.c_str()];
    _header_label.textAlignment = NSTextAlignmentCenter;
    [_header_label sizeToFit];

    int label_height = _header_label.frame.size.height;
    int header_height = fmax(label_height + 10, _table.sectionHeaderHeight);
    _header_label.frame = CGRectMake(0, (header_height - label_height) / 2,
                                     _table.frame.size.width, label_height);
    _header = [[UIView alloc] initWithFrame:CGRectMake(0, 0, _table.frame.size.width, header_height)];
    [_header addSubview: _header_label];
    _table.tableHeaderView = _header;
    return self;
  }

  - (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView { return data.size(); }
  - (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    return data[section].size();
  }

  - (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    static NSString *cellIdentifier = @"cellIdentifier";
    UITableViewCell *cell = [self.table dequeueReusableCellWithIdentifier:cellIdentifier];
    if (cell == nil) {
      CHECK_LT(indexPath.section, data.size());
      CHECK_LT(indexPath.row, data[indexPath.section].size());
      cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleDefault reuseIdentifier:cellIdentifier];
      cell.textLabel.text = [NSString stringWithUTF8String: tuple_get<0>(data[indexPath.section][indexPath.row]).c_str()];
    }
    return cell;
  }

  - (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    INFOf("select row %d", indexPath.row);
    [_table removeFromSuperview];
  }

  + (void)addTable:(const std::string&)title items:(const std::vector<LFL::MenuItem>&)item {
    NativeTable *table = [[NativeTable alloc] init: title items: item];
    tables[title] = table;
  }

  + (void)launchTable:(const std::string&)title_text {
    auto it = tables.find(title_text);
    if (it == tables.end()) { ERRORf("unknown menu: %s", title_text.c_str()); return; }
    [[LFUIApplication sharedAppDelegate].view addSubview: it->second->_table];
  }

  static std::unordered_map<std::string, NativeTable*> tables;
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
void Application::AddNativeAlert(const string &name, const vector<pair<string, string>>&items) {
  [NativeAlert addAlert:name items:items];
}

void Application::LaunchNativeAlert(const string &name, const string &arg) {
  [NativeAlert showAlert:name arg:arg];
}

void Application::AddNativeEditMenu(const vector<MenuItem>&items) {}
void Application::AddNativeMenu(const string &title, const vector<MenuItem>&items) {
  [NativeMenu addMenu:title items:items];
}

void Application::LaunchNativeMenu(const string &title) {
  [NativeMenu launchMenu:title];
}

void Application::AddToolbar(const string &title, const vector<pair<string, string>>&items) {
  [NativeToolbar addToolbar:title items:items];
}

void Application::ShowToolbar(const string &title, bool v) {
  if (v) [NativeToolbar showToolbar: title];
}

void Application::ToggleToolbarButton(const string &title, const string &n) { 
  [NativeToolbar toggleToolbarButton:title withTitle:n];
}

void Application::AddNativeTable(const string &title, const vector<MenuItem>&items) {
  [NativeTable addTable:title items:items];
}

void Application::LaunchNativeTable(const string &title) {
  [NativeTable launchTable:title];
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
