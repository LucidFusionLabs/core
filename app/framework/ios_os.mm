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

@interface IOSAlert : NSObject<UIAlertViewDelegate>
  @property (nonatomic, retain) UIAlertView *alert;
  @property (nonatomic)         bool         add_text;
  @property (nonatomic)         std::string  style, cancel_cmd, confirm_cmd;
@end

@implementation IOSAlert
  - (id)init:(const LFL::StringPairVec&) kv {
    CHECK_EQ(4, kv.size());
    CHECK_EQ("style", kv[0].first);
    _style       = kv[0].second;
    _cancel_cmd  = kv[2].second;
    _confirm_cmd = kv[3].second;
    _alert       = [[UIAlertView alloc]
      initWithTitle:     [NSString stringWithUTF8String: kv[1].first .c_str()]
      message:           [NSString stringWithUTF8String: kv[1].second.c_str()]
      delegate:          self
      cancelButtonTitle: [NSString stringWithUTF8String: kv[2].first.c_str()]
      otherButtonTitles: [NSString stringWithUTF8String: kv[3].first.c_str()], nil];
    if ((_add_text = _style == "textinput")) _alert.alertViewStyle = UIAlertViewStylePlainTextInput;
    return self;
  }

  - (void)alertView:(UIAlertView *)alertView clickedButtonAtIndex:(NSInteger)buttonIndex {
    if (_add_text) {
      ShellRun(buttonIndex ?
               LFL::StrCat(_confirm_cmd, " ", [[alertView textFieldAtIndex:0].text UTF8String]).c_str() :
               _cancel_cmd.c_str());
    } else {
      ShellRun(buttonIndex ? _confirm_cmd.c_str() : _cancel_cmd.c_str());
    }
  }
@end

@interface IOSMenu : NSObject<UIActionSheetDelegate>
  {
    std::vector<LFL::MenuItem> menu;
  }
  @property (nonatomic, retain) UIActionSheet *actions;
@end

@implementation IOSMenu
  - (id)init:(const std::string&)title_text items:(const std::vector<LFL::MenuItem>&)item {
    menu = item;
    NSString *title = [NSString stringWithUTF8String: title_text.c_str()];
    _actions = [[UIActionSheet alloc] initWithTitle:title delegate:self
      cancelButtonTitle:@"Cancel" destructiveButtonTitle:nil otherButtonTitles:nil];
    for (auto &i : menu)
      [_actions addButtonWithTitle:[NSString stringWithUTF8String: tuple_get<1>(i).c_str()]];
    return self;
  }

  - (void)actionSheet:(UIActionSheet *)actions clickedButtonAtIndex:(NSInteger)buttonIndex {
    if (buttonIndex < 1 || buttonIndex > menu.size()) { ERRORf("invalid buttonIndex %d size=%d", buttonIndex, menu.size()); return; }
    ShellRun(tuple_get<2>(menu[buttonIndex-1]).c_str());
  }
@end

@implementation IOSToolbar
  {
    UIToolbar *toolbar;
    int toolbar_height;
    std::unordered_map<std::string, void*> toolbar_titles;
    std::unordered_map<void*, std::string> toolbar_cmds;
  }

  - (id)init: (const LFL::StringPairVec&) kv {
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
    toolbar_height = 44;
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

  - (void)toggleButtonNamed: (const std::string&) n {
    auto it = toolbar_titles.find(n);
    if (it != toolbar_titles.end()) [self toggleButton: (id)(UIBarButtonItem*)it->second];
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

  - (void)show: (bool)show_or_hide {
    if (show_or_hide) {
      show_bottom.push_back(self);
      [[LFUIApplication sharedAppDelegate].window addSubview: toolbar];
    } else {
      LFL::VectorEraseByValue(&show_bottom, self);
      [toolbar removeFromSuperview];
    }
  }

  + (int)getBottomHeight {
    int ret = 0;
    for (auto t : show_bottom) ret += t->toolbar_height;
    return ret;
  }

  + (void)updateFrame {
    for (auto t : show_bottom) t->toolbar.frame = [t getToolbarFrame];
  }

  static std::vector<IOSToolbar*> show_bottom, show_top;
@end

@interface IOSTable : UITableViewController
  {
    std::vector<std::string> rows;
  }
  @property (nonatomic, retain) UIView *header;
  @property (nonatomic, retain) UILabel *header_label;
  @property (nonatomic, assign) IOSToolbar *toolbar;
@end

@implementation IOSTable
  {
    int section_index;
    std::vector<std::vector<LFL::MenuItem>> data;
  }

  - (void)load: (const std::string&)title items:(const std::vector<LFL::MenuItem>&)item {
    data.emplace_back();
    for (auto i : item) {
      if (tuple_get<0>(i) == "<seperator>") {
        data.emplace_back();
        section_index++;
      } else {
        data[section_index].push_back(i);
      }
    }

    self.title = [NSString stringWithUTF8String: title.c_str()];
    self.tableView.separatorInset = UIEdgeInsetsZero;
    [self.tableView setSeparatorStyle:UITableViewCellSeparatorStyleSingleLine];
    [self.tableView setSeparatorColor:[UIColor blackColor]];
  }

  - (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView { return data.size(); }
  - (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    return data[section].size();
  }

  - (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    static NSString *cellIdentifier = @"cellIdentifier";
    UITableViewCell *cell = [self.tableView dequeueReusableCellWithIdentifier:cellIdentifier];
    if (cell == nil) {
      CHECK_LT(indexPath.section, data.size());
      CHECK_LT(indexPath.row, data[indexPath.section].size());
      cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleDefault reuseIdentifier:cellIdentifier];
      cell.textLabel.text = [NSString stringWithUTF8String: tuple_get<0>(data[indexPath.section][indexPath.row]).c_str()];
    }
    return cell;
  }

  - (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    CHECK_LT(indexPath.section, data.size());
    CHECK_LT(indexPath.row, data[indexPath.section].size());
    ShellRun(tuple_get<2>(data[indexPath.section][indexPath.row]).c_str());
  }

  - (void)show:(bool)show_or_hide {
    if (show_or_hide) [[LFUIApplication sharedAppDelegate].view addSubview: self.tableView];
    else [self.tableView removeFromSuperview];
  }

  - (void)viewWillAppear:   (BOOL)animated { if (_toolbar) [_toolbar show: true];  }
  - (void)viewWillDisappear:(BOOL)animated { if (_toolbar) [_toolbar show: false]; }
@end

@interface IOSNavigation : NSObject<UINavigationControllerDelegate>
  @property (nonatomic, retain) UINavigationController *controller;
@end

@implementation IOSNavigation
  - (id)init: (IOSTable*)root_controller {
    _controller = [[UINavigationController alloc] initWithRootViewController: root_controller];
    [_controller setToolbarHidden:YES animated:YES];
    return self;
  }
@end

@interface IOSPicker : NSObject<UIPickerViewDelegate>
  {
    std::vector<std::vector<std::string>> columns;
    std::vector<int> picked_row;
  }
  @property (nonatomic, retain) UIPickerView *picker;
  - (bool)didSelect;
@end

@implementation IOSPicker
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

@interface IOSFontPicker : IOSPicker
@end

@implementation IOSFontPicker
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

@interface IOSKeychain : NSObject
  + (void)save:(NSString *)service data:(id)data;
  + (id)load:(NSString *)service;
@end

@implementation IOSKeychain
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
SystemAlertWidget::~SystemAlertWidget() { if (auto alert = FromVoid<IOSAlert*>(impl)) [alert release]; }
SystemAlertWidget::SystemAlertWidget(const StringPairVec &items) : impl([[IOSAlert alloc] init: items]) {}
void SystemAlertWidget::Show(const string &arg) {
  auto alert = FromVoid<IOSAlert*>(impl);
  [alert.alert show];
  if (alert.add_text) [alert.alert textFieldAtIndex:0].text = [NSString stringWithUTF8String: arg.c_str()];
}

SystemMenuWidget::~SystemMenuWidget() { if (auto menu = FromVoid<IOSMenu*>(impl)) [menu release]; }
SystemMenuWidget::SystemMenuWidget(const string &t, const vector<MenuItem> &i) : impl([[IOSMenu alloc] init:t items:i]) {}
void SystemMenuWidget::Show() { [FromVoid<IOSMenu*>(impl).actions showInView:[UIApplication sharedApplication].keyWindow]; }
unique_ptr<SystemMenuWidget> SystemMenuWidget::CreateEditMenu(const vector<MenuItem> &items) { return nullptr; }

SystemToolbarWidget::~SystemToolbarWidget() { if (auto toolbar = FromVoid<IOSToolbar*>(impl)) [toolbar release]; }
SystemToolbarWidget::SystemToolbarWidget(const StringPairVec &items) : impl([[IOSToolbar alloc] init: items]) {}
void SystemToolbarWidget::Show(bool show_or_hide) { [FromVoid<IOSToolbar*>(impl) show:show_or_hide]; }
void SystemToolbarWidget::ToggleButton(const string &n) { [FromVoid<IOSToolbar*>(impl) toggleButtonNamed: n]; }

SystemTableWidget::~SystemTableWidget() { if (auto table = FromVoid<IOSTable*>(impl)) [table release]; }
SystemTableWidget::SystemTableWidget(const string &title, const vector<MenuItem>&items) {
  auto table = [[IOSTable alloc] initWithStyle: UITableViewStyleGrouped];
  [table load:title items:items];
  impl = table;
}
void SystemTableWidget::AddToolbar(SystemToolbarWidget *t) { [FromVoid<IOSTable*>(impl) setToolbar: FromVoid<IOSToolbar*>(t->impl)]; }
void SystemTableWidget::Show(bool show_or_hide) { [FromVoid<IOSTable*>(impl) show:show_or_hide]; }

SystemNavigationWidget::~SystemNavigationWidget() { if (auto nav = FromVoid<IOSNavigation*>(impl)) [nav release]; }
SystemNavigationWidget::SystemNavigationWidget(SystemTableWidget *r) : impl([[IOSNavigation alloc] init: FromVoid<IOSTable*>(r->impl)]) {}
void SystemNavigationWidget::Show(bool show_or_hide) {
  auto nav = FromVoid<IOSNavigation*>(impl);
  if (show_or_hide) {
    [[LFUIApplication sharedAppDelegate].controller presentViewController: nav.controller
      animated:YES completion:nil];
  } else {
    [[LFUIApplication sharedAppDelegate].controller dismissViewControllerAnimated:YES completion:nil];
  }
}

void SystemNavigationWidget::PushTable(SystemTableWidget *t) {
  [FromVoid<IOSNavigation*>(impl).controller pushViewController: FromVoid<IOSTable*>(t->impl) animated: YES];
}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const string &choose_cmd) {
  static IOSFontPicker *font_chooser = [[IOSFontPicker alloc] init];
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
      [IOSKeychain save:k data:pw];
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
  NSString *pw = [IOSKeychain load: k];
  if (pw) pw_out->assign([pw UTF8String]);
  else    pw_out->clear();
  return  pw_out->size();
}

void Application::ShowAds() {}
void Application::HideAds() {}

String16 Application::GetLocalizedString16(const char *key) { return String16(); }
string Application::GetLocalizedString(const char *key) {
  NSString *localized = 
    [[NSBundle mainBundle] localizedStringForKey: [NSString stringWithUTF8String: key] value:nil table:nil];
  if (!localized) return StrCat("<missing localized: ", key, ">");
  else            return [localized UTF8String];
}

String16 Application::GetLocalizedInteger16(int number) { return String16(); }
string Application::GetLocalizedInteger(int number) {
  static NSNumberFormatter *formatter=0;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    formatter = [[NSNumberFormatter alloc] init];
    [formatter setNumberStyle:NSNumberFormatterDecimalStyle];
    [formatter setMaximumFractionDigits:0];
    [formatter setMinimumIntegerDigits:1];
    [formatter setLocale:[NSLocale autoupdatingCurrentLocale]];
  });
  return [[formatter stringFromNumber: [NSNumber numberWithLong:number]] UTF8String];
}

}; // namespace LFL
