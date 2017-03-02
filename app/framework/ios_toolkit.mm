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

static std::vector<UIImage*> app_images;
struct iOSTableItem { enum { GUILoaded=LFL::TableItem::Flag::User1 }; };

@implementation IOSButton
  - (void)buttonClicked:(IOSButton*)sender {
    if (sender.cb) sender.cb();
  }
@end

@implementation IOSBarButtonItem
  - (IBAction)buttonClicked:(IOSBarButtonItem*)sender {
    if (sender.cb) sender.cb();
  }
@end

@interface IOSSegmentedControl : UISegmentedControl
  @property (nonatomic, assign) LFL::StringCB changed_cb;
@end
@implementation IOSSegmentedControl
@end

@interface IOSTextField : UITextField
  @property (nonatomic) bool modified;
  - (void)textFieldDidChange:(IOSTextField*)sender;
@end
@implementation IOSTextField
  - (void)textFieldDidChange:(IOSTextField*)sender { _modified = true; }
@end

@implementation IOSAlert
  - (id)init:(const LFL::AlertItemVec&) kv {
    self = [super init];
    CHECK_EQ(4, kv.size());
    CHECK_EQ("style", kv[0].first);
    _style       = kv[0].second;
    _cancel_cb   = kv[2].cb;
    _confirm_cb  = kv[3].cb;
    _alert       = [[UIAlertView alloc]
      initWithTitle:     [NSString stringWithUTF8String: kv[1].first .c_str()]
      message:           [NSString stringWithUTF8String: kv[1].second.c_str()]
      delegate:          self
      cancelButtonTitle: [NSString stringWithUTF8String: kv[2].first.c_str()]
      otherButtonTitles: [NSString stringWithUTF8String: kv[3].first.c_str()], nil];
    if      ((_add_text = _style == "textinput")) _alert.alertViewStyle = UIAlertViewStylePlainTextInput;
    else if ((_add_text = _style == "pwinput"))   _alert.alertViewStyle = UIAlertViewStyleSecureTextInput;
#if 0
    UISwitch *onoff = [[UISwitch alloc] init];
    [_alert setValue:onoff forKey:@"accessoryView"];
    [onoff release];
#endif
    return self;
  }

  - (void)alertView:(UIAlertView *)alertView clickedButtonAtIndex:(NSInteger)buttonIndex {
    if (buttonIndex) { if (_confirm_cb) _confirm_cb(_add_text ? [[alertView textFieldAtIndex:0].text UTF8String] : ""); }
    else             { if (_cancel_cb)  _cancel_cb(""); }
    _done = true;
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

@implementation IOSToolbar
  {
    std::unordered_map<std::string, int> toolbar_titles;
    LFL::MenuItemVec data;
    std::vector<bool> toggled;
  }

  - (id)init: (LFL::MenuItemVec)kv {
    self = [super init];
    data = move(kv);
    toggled.resize(data.size());
    for (auto b = data.begin(), e = data.end(), i = b; i != e; ++i) toolbar_titles[i->shortcut] = i - b;
    _toolbar  = [self createUIToolbar: [self getToolbarFrame] first:true];
    _toolbar2 = [self createUIToolbar: [self getToolbarFrame] first:false];
    return self;
  }
  
  - (NSMutableArray*)createUIToolbarItems:(BOOL)first {
    NSMutableArray *items = [[NSMutableArray alloc] init];
    UIBarButtonItem *spacer = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemFlexibleSpace target:nil action:nil];
    for (int i=0, l=data.size(); i<l; i++) {
      if (i || l == 1) [items addObject: spacer];
      NSString *K = [NSString stringWithUTF8String: data[i].shortcut.c_str()];
      UIBarButtonItem *item;
      if (int icon = data[i].image) {
          CHECK_LE(icon, app_images.size());
          item = [[UIBarButtonItem alloc] initWithImage: app_images[icon - 1]
            style:UIBarButtonItemStylePlain target:self action:(first ? @selector(onClick:) : @selector(onClick2:))];
      } else {
        item = [[UIBarButtonItem alloc] initWithTitle:(([K length] && LFL::isascii([K characterAtIndex:0])) ? [NSString stringWithFormat:@"%@", K] : [NSString stringWithFormat:@"%@\U0000FE0E", K])
          style:UIBarButtonItemStylePlain target:self action:(first ? @selector(onClick:) : @selector(onClick2:))];
      }
      [item setTag:i];
      [items addObject:item];
      [item release];
    }
    if (data.size() == 1) [items addObject: spacer];
    [spacer release];
    return items;
  }

  - (UIToolbar*)createUIToolbar:(CGRect)rect first:(BOOL)first {
    NSMutableArray *items = [self createUIToolbarItems: first];
    UIToolbar *tb = [[UIToolbar alloc] initWithFrame: rect];
    // [tb setBarStyle:UIBarStyleBlackTranslucent];
    tb.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleTopMargin;
    [tb setItems:items];
    [items release];
    return tb;
  }

  - (CGRect)getToolbarFrame {
    int tbh = 44;
    auto uiapp = [LFUIApplication sharedAppDelegate];
    CGRect bounds = [uiapp.glk_view bounds], kbb = [uiapp.controller getKeyboardFrame];
    return CGRectMake(0, bounds.size.height - kbb.size.height - tbh, bounds.size.width, tbh);
  }

  - (void)toggleButtonNamed: (const std::string&) n {
    auto it = toolbar_titles.find(n);
    if (it != toolbar_titles.end()) [self toggleButton: it->second];
  }

  - (void)toggleButton:(int)ind {
    CHECK_RANGE(ind, 0, toggled.size());
    bool is_on = toggled[ind] = !toggled[ind];
    [self toggleButton:ind onToolbar:_toolbar  isToggled:is_on];
    [self toggleButton:ind onToolbar:_toolbar2 isToggled:is_on];
  }

  - (void)toggleButton:(int)ind onToolbar:(UIToolbar*)tb isToggled:(bool)is_on {
    CHECK_RANGE(ind*2, 0, [tb.items count]);
    UIBarButtonItem *item = (UIBarButtonItem*)[tb.items objectAtIndex:ind*2];
    item.tintColor = is_on ? [UIColor colorWithRed:0.8 green:0.8 blue:0.8 alpha:.8] : nil;
  }

  - (void)onClick:(id)sender {
    if (![sender isKindOfClass:[UIBarButtonItem class]]) FATALf("unknown sender: %p", sender);
    UIBarButtonItem *item = (UIBarButtonItem*)sender;
    CHECK_RANGE(item.tag, 0, data.size());
    auto &b = data[item.tag];
    if (b.cb) b.cb();
    if (b.name == "toggle") [self toggleButton: item.tag];
  }

  - (void)onClick2:(id)sender {
    [self onClick: sender];
    [[LFUIApplication sharedAppDelegate].controller resignFirstResponder];
  }

  - (void)show: (bool)show_or_hide {
    auto uiapp = [LFUIApplication sharedAppDelegate];
    uiapp.controller.input_accessory_toolbar = show_or_hide ? _toolbar : nil;
    uiapp.text_field.inputAccessoryView = show_or_hide ? _toolbar2 : nil;
    if ([uiapp.text_field isFirstResponder]) [uiapp.text_field reloadInputViews];
    [_toolbar removeFromSuperview];
    if (show_or_hide) {
      uiapp.controller.input_accessory_toolbar.hidden = uiapp.controller.showing_keyboard;
      [uiapp.glk_view addSubview: _toolbar];
    }
  }

  + (int)getBottomHeight {
    auto uiapp = [LFUIApplication sharedAppDelegate];
    if (!uiapp.controller.input_accessory_toolbar) return 0;
    return uiapp.controller.input_accessory_toolbar.frame.size.height;
  }
@end

@interface IOSPicker : UIPickerView<UIPickerViewDelegate>
  {
    LFL::PickerItem item;
  }
  - (LFL::PickerItem*)getItem;
@end

@implementation IOSPicker
  - (id)initWithColumns:(LFL::PickerItem)in {
    self = [super init];
    return [self finishInit:in];
  }

  - (id)initWithColumns:(LFL::PickerItem)in andFrame:(CGRect)r {
    self = [super initWithFrame:r];
    return [self finishInit:in];
  }

  - (id)finishInit:(LFL::PickerItem)in {
    item = move(in);
    super.delegate = self;
    super.showsSelectionIndicator = YES;
    super.hidden = NO;
    super.layer.borderColor = [UIColor grayColor].CGColor;
    super.layer.borderWidth = 4;
    [self setBackgroundColor:[UIColor whiteColor]];
    return self;
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

@interface IOSFontPicker : IOSPicker
@end

@implementation IOSFontPicker
  {
    LFL::StringVecCB font_change_cb;
  }

  - (id)init {
    LFL::PickerItem p;
    p.cb = [=](LFL::PickerItem *x) -> bool {
      if (font_change_cb) font_change_cb(LFL::StringVec{x->data[0][x->picked[0]], x->data[1][x->picked[1]]});
      return true;
    };
    [IOSFontPicker getSystemFonts:     &LFL::PushBack(p.data, {})];
    [IOSFontPicker getSystemFontSizes: &LFL::PushBack(p.data, {})];
    self = [super initWithColumns: move(p)];
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

@implementation IOSNavigation
  - (void) viewDidLoad {
    [super viewDidLoad];
    INFO("IOSNavigation viewDidLoad: frame=", LFL::GetCGRect(self.view.frame).DebugString());
  }

  - (void) viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    INFO("IOSNavigation viewDidAppear: frame=", LFL::GetCGRect(self.view.frame).DebugString());
  }

  - (BOOL)shouldAutorotate { return self.topViewController.shouldAutorotate; }
  - (NSUInteger)supportedInterfaceOrientations { return self.topViewController.supportedInterfaceOrientations; }
@end

@implementation IOSTable
  {
    std::vector<LFL::TableSection> data;
    int double_section_row_height;
    bool dark_theme, change_selected_row_background;
  }

  - (id)initWithStyle: (UITableViewStyle)style {
    self = [super initWithNibName:nil bundle:nil];
    _tableView = [[UITableView alloc] initWithFrame: [LFUIApplication sharedAppDelegate].controller.view.frame style:style];
    _tableView.autoresizingMask = UIViewAutoresizingFlexibleHeight | UIViewAutoresizingFlexibleWidth;
    _tableView.delegate = self;
    _tableView.dataSource = self;
    _orig_bg_color = _tableView.backgroundColor;
    _orig_separator_color = _tableView.separatorColor;
    [self.view addSubview: _tableView];
    return self;
  }

  - (void)setTheme:(const std::string&)n {
    if ((dark_theme = (n == "Dark"))) {
      _tableView.backgroundColor = [UIColor colorWithWhite:0.249 alpha:1.000];
      _tableView.separatorColor = [UIColor colorWithWhite:0.664 alpha:1.000];
    } else {
      _tableView.backgroundColor = _orig_bg_color;
      _tableView.separatorColor = _orig_separator_color;
    }
    if (self.isViewLoaded && _tableView.window) [_tableView reloadData];
    else _needs_reload = true;
  }

  - (void)load:(LFL::SystemTableView*)lself withTitle:(const std::string&)title withStyle:(const std::string&)sty items:(std::vector<LFL::TableSection>)item {
    _lfl_self = lself;
    _style = sty;
    _needs_reload = true;
    _editable_section = _editable_start_row = double_section_row_height = -1;
    data = move(item);
    self.title = LFL::MakeNSString(title);
    if (_style != "indent") self.tableView.separatorInset = UIEdgeInsetsZero;
    if (_style == "big") {
      double_section_row_height = 0;
      change_selected_row_background = true;
    }
    [self.tableView setSeparatorStyle:UITableViewCellSeparatorStyleSingleLine];
    [self.tableView setSeparatorColor:[UIColor grayColor]];
  }

  - (void)addRow:(int)section withItem:(LFL::TableItem)item {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    data[section].item.emplace_back(move(item));
    NSIndexPath *path = [NSIndexPath indexPathForRow:data[section].item.size()-1 inSection:section];
    [self.tableView insertRowsAtIndexPaths:@[path] withRowAnimation:UITableViewRowAnimationNone];
  }

  - (void)replaceSection:(int)section items:(std::vector<LFL::TableItem>)item header:(LFL::TableItem)h flag:(int)f {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    data[section] = LFL::TableSection(move(h), f);
    data[section].item = move(item);
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex: section]
      withRowAnimation:UITableViewRowAnimationNone];
  }

  - (void)checkExists:(int)section row:(int)r {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_LT(r, data[section].item.size());
  }

  - (void)setHeader:(int)section header:(LFL::TableItem)h {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    data[section].header = move(h);
  }

  - (void)setKey:(int)section row:(int)r val:(const std::string&)v {
    [self checkExists:section row:r];
    auto &ci = data[section].item[r];
    ci.key = v;
  }

  - (std::string)getKey:(int)section row:(int)r {
    [self checkExists:section row:r];
    return data[section].item[r].key;
  }

  - (int)getTag:(int)section row:(int)r {
    [self checkExists:section row:r];
    return data[section].item[r].tag;
  }

  - (void)setTag:(int)section row:(int)r val:(int)v {
    [self checkExists:section row:r];
    data[section].item[r].tag = v;
  }

  - (void)setHidden:(int)section row:(int)r val:(bool)v {
    [self checkExists:section row:r];
    data[section].item[r].hidden = v;
  }

  - (void)setValue:(int)section row:(int)r val:(const std::string&)v {
    [self checkExists:section row:r];
    auto &ci = data[section].item[r];
    ci.val = v;
  }

  - (void)setSelected:(int)section row:(int)r val:(int)v {
    [self checkExists:section row:r];
    auto &ci = data[section].item[r];
    ci.selected = v;
  }

  - (void)replaceRow:(int)section row:(int)r val:(LFL::TableItem)v {
    [self checkExists:section row:r];
    data[section].item[r] = move(v);
    NSIndexPath *p = [NSIndexPath indexPathForRow:r inSection:section];
    [self.tableView reloadRowsAtIndexPaths:@[p] withRowAnimation:UITableViewRowAnimationNone];
  }

  - (void)setSectionValues:(int)section items:(const LFL::StringVec&)item {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_EQ(item.size(), data[section].item.size());
    for (int i=0, l=data[section].item.size(); i != l; ++i) [self setValue:section row:i val:item[i]];
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex: section]
      withRowAnimation:UITableViewRowAnimationNone];
  }

  - (void)applyChangeList:(const LFL::TableSection::ChangeList&)changes {
    LFL::TableSection::ApplyChangeList(changes, &data, [=](const LFL::TableSection::Change &d){
      NSIndexPath *p = [NSIndexPath indexPathForRow:d.row inSection:d.section];
      [self.tableView reloadRowsAtIndexPaths:@[p] withRowAnimation:UITableViewRowAnimationNone];
    });
  }

  - (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView { return data.size(); }
  - (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    return data[section].item.size();
  }

  - (CGRect)getCellFrame:(int)labelWidth {
    return CGRectMake(0, 0, self.tableView.frame.size.width - 110 - labelWidth, 44);
  }

  - (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)path {
    if (path.section >= data.size() || path.row >= data[path.section].item.size()) return tableView.rowHeight;
    const auto &ci = data[path.section].item[path.row];
    if (ci.hidden) return 0;
    else if ((ci.flags & iOSTableItem::GUILoaded) &&
             (ci.type == LFL::TableItem::Picker || ci.type == LFL::TableItem::FontPicker)) return ci.height;
    else if (double_section_row_height == path.section) return tableView.rowHeight * 2;
    else if (ci.flags & LFL::TableItem::Flag::SubText) return UITableViewAutomaticDimension;
    else return tableView.rowHeight;
  }
  
  - (void)clearNavigationButton:(int)align {
    if (align == LFL::HAlign::Right) self.navigationItem.rightBarButtonItem = nil;
    else                             self.navigationItem.leftBarButtonItem  = nil;
  }

  - (void)loadNavigationButton:(const LFL::TableItem&)item withAlign:(int)align {
    if (item.key == "Edit") {
      self.navigationItem.rightBarButtonItem = [self editButtonItem];
    } else {
      IOSBarButtonItem *button = [[IOSBarButtonItem alloc] init];
      button.cb = item.cb;
      button.title = LFL::MakeNSString(item.key);
      [button setTarget:button];
      [button setAction:@selector(buttonClicked:)];
      if (align == LFL::HAlign::Right) self.navigationItem.rightBarButtonItem = button;
      else                             self.navigationItem.leftBarButtonItem  = button;
      [button release];
    }
  }

  - (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)path {
    static NSString *cellIdentifier = @"cellIdentifier";
    UITableViewCell *cell = [self.tableView dequeueReusableCellWithIdentifier:cellIdentifier];
    if (cell) { [cell release]; cell = nil; }
    if (cell == nil) {
      int row = path.row, section = path.section;
      CHECK_LT(section, data.size());
      CHECK_LT(row, data[section].item.size());
      auto &ci = data[section].item[row];
      ci.flags |= iOSTableItem::GUILoaded;
      bool subtext      = ci.flags & LFL::TableItem::Flag::SubText;
      bool is_selected_row = section == _selected_section && row == _selected_row;
      UIColor *blue  = [UIColor colorWithRed: 0.0/255 green:122.0/255 blue:255.0/255 alpha:1];

      cell = [[UITableViewCell alloc] initWithStyle: (subtext ? UITableViewCellStyleSubtitle : UITableViewCellStyleDefault)
        reuseIdentifier:cellIdentifier];
      cell.selectionStyle = UITableViewCellSelectionStyleNone;
      if      (ci.bg_a)                                           [cell setBackgroundColor:[UIColor colorWithRed:ci.bg_r/255.0 green:ci.bg_g/255.0 blue:ci.bg_b/255.0 alpha:ci.bg_a/255.0]];
      else if (change_selected_row_background && is_selected_row) [cell setBackgroundColor:[UIColor lightGrayColor]];
      else if (dark_theme)                                        [cell setBackgroundColor: _tableView.backgroundColor];

      if (ci.type != LFL::TableItem::Button) {
        if (int icon = ci.left_icon) {
          CHECK_LE(icon, app_images.size());
          cell.imageView.image = app_images[icon - 1]; 
        }
      }

      if (ci.dropdown_key.size() && !(ci.flags & LFL::TableItem::Flag::FixDropdown)) {
        int w = 10, x = [LFL::MakeNSString(ci.key) sizeWithAttributes:
          @{NSFontAttributeName:[UIFont boldSystemFontOfSize:[UIFont labelFontSize]]}].width + 20 +
          (cell.imageView.image ? 60 : 0);
        IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
        button.frame = CGRectMake(0, 0, x+w, 40.0);
        if (ci.cb) {
          [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
          button.cb = [=](){ auto &item = data[section].item[row]; if (item.cb) item.cb(); };
        }
        [cell.contentView addSubview: button];
      }

      bool textinput=0, numinput=0, pwinput=0;
      if ((textinput = ci.type == LFL::TableItem::TextInput) || (numinput = ci.type == LFL::TableItem::NumberInput)
          || (pwinput = ci.type == LFL::TableItem::PasswordInput)) {
        cell.textLabel.text = LFL::MakeNSString(ci.key);
        [cell.textLabel sizeToFit];

        IOSTextField *textfield = [[IOSTextField alloc] initWithFrame:
          [self getCellFrame:cell.textLabel.frame.size.width]];
        textfield.autoresizingMask = UIViewAutoresizingFlexibleHeight;
        // textfield.adjustsFontSizeToFitWidth = YES;
        textfield.autoresizesSubviews = YES;
        textfield.autocorrectionType = UITextAutocorrectionTypeNo;
        textfield.autocapitalizationType = UITextAutocapitalizationTypeNone;
        textfield.clearButtonMode = UITextFieldViewModeNever;
        if      (pwinput)  textfield.secureTextEntry = YES;
        else if (numinput) textfield.keyboardType = UIKeyboardTypeNumberPad;
        else               textfield.keyboardType = UIKeyboardTypeDefault;
        textfield.returnKeyType = UIReturnKeyDone;
        // textfield.layer.cornerRadius = 10.0;
        // [textfield setBorderStyle: UITextBorderStyleRoundedRect];
        [textfield addTarget:self action:@selector(textFieldDidChange:) 
          forControlEvents:UIControlEventEditingChanged];

        if (ci.HasPlaceholderValue()) [textfield setPlaceholder: LFL::MakeNSString(ci.GetPlaceholderValue())];
        else if (ci.val.size())       [textfield setText:        LFL::MakeNSString(ci.val)];

        textfield.textAlignment = NSTextAlignmentRight;
        cell.accessoryView = textfield;
        if (is_selected_row) [textfield becomeFirstResponder];
        [textfield release];

      } else if (ci.type == LFL::TableItem::Selector) {
        NSArray *itemArray = LFL::MakeNSStringArray(LFL::Split(ci.val, ','));
        IOSSegmentedControl *segmented_control = [[IOSSegmentedControl alloc] initWithItems:itemArray];
        segmented_control.selectedSegmentIndex = ci.selected;
        if (ci.right_cb) segmented_control.changed_cb = ci.right_cb;
        [segmented_control addTarget:self action:@selector(segmentedControlClicked:)
          forControlEvents: UIControlEventValueChanged];
        if (ci.flags & LFL::TableItem::Flag::HideKey) {
          segmented_control.frame = cell.frame;
          segmented_control.autoresizingMask = UIViewAutoresizingFlexibleWidth;
          [cell.contentView addSubview:segmented_control];
        } else {
          cell.textLabel.text = LFL::MakeNSString(ci.key);
          [cell.textLabel sizeToFit];
          cell.accessoryView = segmented_control;
        }
        [segmented_control release]; 

      } else if (ci.type == LFL::TableItem::Picker || ci.type == LFL::TableItem::FontPicker) {
        LFL::PickerItem item;
        if (ci.type == LFL::TableItem::Picker) item = *ci.picker;
        else {
          [IOSFontPicker getSystemFonts:     &LFL::PushBack(item.data, {})];
          [IOSFontPicker getSystemFontSizes: &LFL::PushBack(item.data, {})];
        }
        item.cb = [=](LFL::PickerItem *x) -> bool {
          [self pickerPicked:x withSection:section andRow:row];
          return false;
        };

        int picker_cols = item.data.size();
        IOSPicker *picker = [[IOSPicker alloc] initWithColumns: move(item)];
        picker.autoresizingMask = UIViewAutoresizingFlexibleWidth;
        if (row > 0) {
          LFL::StringVec v, joined(picker_cols, "");
          LFL::Split(data[section].item[row-1].val, LFL::isspace, &v);
          LFL::Join(&joined, v, " ", false); 
          [picker selectRows: joined];
        }
        [cell.contentView addSubview:picker];
        ci.height = picker.frame.size.height;

      } else if (ci.type == LFL::TableItem::Button) {
        if (int icon = ci.left_icon) {
          CHECK_LE(icon, app_images.size());
          UIImage *image = app_images[icon - 1]; 
          IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
          [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
          button.cb = [=](){ auto &item = data[section].item[row]; if (item.cb) item.cb(); };
          button.frame = cell.frame;
          int spacing = -10, target_height = 40, margin = fabs(button.frame.size.height - target_height) / 2;
          [button setTitleColor:blue forState:UIControlStateNormal];
          [button setTitle:LFL::MakeNSString(ci.key) forState:UIControlStateNormal];
          [button setImage:image forState:UIControlStateNormal];
          button.imageView.contentMode = UIViewContentModeScaleAspectFit;
          [button setTitleEdgeInsets:UIEdgeInsetsMake(0, spacing, 0, 0)];
          [button setImageEdgeInsets:UIEdgeInsetsMake(margin, 0, margin, spacing)];
          [cell.contentView addSubview:button];
          [button release];
        } else {
          if (ci.bg_a) [cell.textLabel setFont:[UIFont boldSystemFontOfSize:[UIFont labelFontSize]]];
          cell.textLabel.text = LFL::MakeNSString(ci.key);
          cell.textLabel.textAlignment = NSTextAlignmentCenter;
          [cell.textLabel sizeToFit];
        }

      } else if (ci.type == LFL::TableItem::Toggle) {
        UISwitch *onoff = [[UISwitch alloc] init];
        onoff.on = ci.val == "1";
        [onoff addTarget: self action: @selector(switchFlipped:) forControlEvents: UIControlEventValueChanged];
        cell.textLabel.text = LFL::MakeNSString(ci.key);
        cell.accessoryView = onoff;
        [onoff release];

      } else if (ci.type == LFL::TableItem::Label) {
        cell.textLabel.text = LFL::MakeNSString(ci.key);
        if (subtext) {
          cell.detailTextLabel.text = LFL::MakeNSString(ci.val);
          cell.detailTextLabel.lineBreakMode = NSLineBreakByWordWrapping;
          cell.detailTextLabel.numberOfLines = 0;
        } else {
          [cell.textLabel sizeToFit];
          UILabel *label = [[UILabel alloc] initWithFrame: [self getCellFrame: cell.textLabel.frame.size.width]];
          label.text = LFL::MakeNSString(ci.val);
          label.adjustsFontSizeToFitWidth = TRUE;
          label.textAlignment = NSTextAlignmentRight;
          cell.accessoryView = label;
          [label release];
        }

      } else {
        cell.textLabel.text = LFL::MakeNSString(ci.key);
      }

      if (ci.dropdown_key.size() && cell.textLabel.text.length &&
          !(ci.flags & LFL::TableItem::Flag::FixDropdown)) {
        cell.textLabel.textColor = blue;
        cell.textLabel.text = [NSString stringWithFormat:@"%@  \U000002C5", cell.textLabel.text];
      }

      if (int icon = ci.right_icon) {
        CHECK_LE(icon, app_images.size());
        UIImage *image = app_images[icon - 1]; 
        IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
        button.frame = CGRectMake(0, 0, 40, 40);
        [button setImage:image forState:UIControlStateNormal];
        button.cb = LFL::bind(ci.right_cb, "");
        [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
        if (cell.accessoryView) [cell.accessoryView addSubview: button];
        else cell.accessoryView = button;
      } else if (ci.right_text.size()) {
        UILabel *label = [[UILabel alloc] init];
        label.textColor = self.view.tintColor;
        label.text = LFL::MakeNSString(ci.right_text);
        [label sizeToFit];
        if (cell.accessoryView) [cell.accessoryView addSubview: label];
        else cell.accessoryView = label;
        [label release];
      }
    }
    return cell;
  }

  - (void)tableView:(UITableView *)tableView commitEditingStyle:(UITableViewCellEditingStyle)editingStyle forRowAtIndexPath:(NSIndexPath *)path {
    [self checkExists:path.section row:path.row];
    if (editingStyle == UITableViewCellEditingStyleDelete) {
      if (_delete_row_cb) _delete_row_cb(path.row, data[path.section].item[path.row].tag);
      data[path.section].item.erase(data[path.section].item.begin() + path.row);
      [tableView beginUpdates];
      [tableView deleteRowsAtIndexPaths:@[path] withRowAnimation:UITableViewRowAnimationNone];
      [tableView endUpdates];
    }
  }

  - (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    return LFL::MakeNSString(data[section].header.key);
  }

  - (CGFloat)tableView:(UITableView *)tableView heightForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    if (int h = data[section].header_height) return h;
    if (int image = data[section].header.left_icon) {
      UIImageView *image_view = [[UIImageView alloc] initWithImage: app_images[image - 1]];
      data[section].header_height = 44 + image_view.frame.size.height;
      [image_view release];
      return data[section].header_height;
    }
    return UITableViewAutomaticDimension;
  }

  - (UIView *)tableView:(UITableView *)tableView viewForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    UIView *headerView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, tableView.frame.size.width, 44)];
    headerView.autoresizingMask = UIViewAutoresizingFlexibleWidth;
    if (data[section].header.hidden) return headerView;

    if (int image = data[section].header.left_icon) {
      UIImageView *image_view = [[UIImageView alloc] initWithImage: app_images[image - 1]];
      data[section].header_height = 44 + image_view.frame.size.height;
      image_view.contentMode = UIViewContentModeCenter;
      image_view.center = CGPointMake(headerView.frame.size.width/2, image_view.frame.size.height/2);
      image_view.autoresizingMask = UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleLeftMargin;
      headerView.frame = CGRectMake(0, 0, tableView.frame.size.width, data[section].header_height);
      [headerView addSubview: image_view];
      [image_view release];
    }

    int key_size = data[section].header.key.size();
    if (key_size) {
      UILabel *label = [[UILabel alloc] initWithFrame:
        CGRectMake(50, headerView.frame.size.height-1-21-11, headerView.frame.size.width-100,
                   (data[section].header.flags & LFL::TableItem::Flag::SubText) ? 44 : 21)];
      label.text = LFL::MakeNSString(data[section].header.key);
      label.textAlignment = NSTextAlignmentCenter;
      label.autoresizingMask = UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleLeftMargin;
      label.lineBreakMode = NSLineBreakByWordWrapping;
      label.numberOfLines = 0;
      [headerView addSubview:label];
      [label release];
    }

    bool edit_button = data[section].flag & LFL::TableSection::Flag::EditButton;
    if (edit_button || data[section].header.right_text.size()) {
      IOSButton *button = [IOSButton buttonWithType:UIButtonTypeSystem];
      if (edit_button) {
        [button setTitle:@"Edit" forState:UIControlStateNormal];
        [button addTarget:self action:@selector(toggleEditMode:) forControlEvents:UIControlEventTouchUpInside];
      } else {
        button.showsTouchWhenHighlighted = TRUE;
        button.cb = [=](){ auto &item = data[section].header; if (item.right_cb) item.right_cb(""); };
        [button setTitle:LFL::MakeNSString(data[section].header.right_text) forState:UIControlStateNormal];
        [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
      }
      [button sizeToFit];
      [button setFrame:CGRectMake(tableView.frame.size.width - button.frame.size.width - 11,
                                  key_size ? 11 : -11, button.frame.size.width, 21)];
      button.autoresizingMask = UIViewAutoresizingFlexibleLeftMargin;
      [headerView addSubview:button];
    }
    return headerView;
  }

  - (void)toggleEditMode:(UIButton*)button {
    [self.tableView setEditing:!self.tableView.editing animated:YES];
    if (self.tableView.editing) [button setTitle:@"Done" forState:UIControlStateNormal];
    else                        [button setTitle:@"Edit" forState:UIControlStateNormal];
    [button sizeToFit];
    [button setFrame:CGRectMake(self.tableView.frame.size.width - button.frame.size.width - 11, 11, button.frame.size.width, 21)];
  }

  - (BOOL)tableView:(UITableView *)tableView canEditRowAtIndexPath:(NSIndexPath *)path {
    return path.section == _editable_section;
  }

  - (UITableViewCellEditingStyle)tableView:(UITableView *)tableView editingStyleForRowAtIndexPath:(NSIndexPath *)path {
    if (path.section != _editable_section || path.row < _editable_start_row) return UITableViewCellEditingStyleNone;
    [self checkExists:path.section row:path.row];
    return (data[path.section].flag & LFL::TableSection::Flag::EditableIfHasTag && !data[path.section].item[path.row].tag) ?
      UITableViewCellEditingStyleNone : UITableViewCellEditingStyleDelete;
  }

  - (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)path {
    [self checkExists:path.section row:path.row];
    [self.tableView deselectRowAtIndexPath:path animated:NO];
    _selected_row = path.row;
    _selected_section = path.section;
    auto &ci = data[path.section].item[path.row];
    if (ci.type == LFL::TableItem::Command || ci.type == LFL::TableItem::Button) {
      if (ci.cb) ci.cb();
    } else if (ci.type == LFL::TableItem::Label && path.row + 1 < data[path.section].item.size()) {
      auto &next_ci = data[path.section].item[path.row+1];
      if (next_ci.type == LFL::TableItem::Picker ||
          next_ci.type == LFL::TableItem::FontPicker) {
        next_ci.hidden = !next_ci.hidden;
        [self reloadRowAtIndexPath:path withRowAnimation:UITableViewRowAnimationNone];
      }
    }
  }

  - (void)show:(bool)show_or_hide {
    auto uiapp = [LFUIApplication sharedAppDelegate];
    if (show_or_hide) [uiapp.glk_view addSubview: self.tableView];
    else              [self.tableView removeFromSuperview];
  }

  - (void)willMoveToParentViewController:(UIViewController *)parent {
    if (parent == nil && _lfl_self && _lfl_self->hide_cb) _lfl_self->hide_cb();
  }

  - (void)viewWillDisappear:(BOOL)animated { [super viewWillDisappear: animated]; }
  - (void)viewWillAppear:(BOOL)animated {
    if (_needs_reload && !(_needs_reload = false)) [_tableView reloadData];
    [super viewWillAppear:animated];
  }

  - (void)textFieldDidChange:(IOSTextField*)sender {
    _lfl_self->changed = true;
    [sender textFieldDidChange: sender];
  }

  - (void)segmentedControlClicked:(IOSSegmentedControl *)segmented_control {
    _lfl_self->changed = true;
    if (segmented_control.changed_cb) segmented_control.changed_cb
      (LFL::GetNSString([segmented_control titleForSegmentAtIndex: segmented_control.selectedSegmentIndex]));
  }

  - (IBAction) switchFlipped: (UISwitch*) onoff {
    _lfl_self->changed = true;
  }

  - (void)pickerPicked:(LFL::PickerItem*)x withSection:(int)section andRow:(int)row {
    _lfl_self->changed = true;
    data[section].item[row].hidden = true;
    [self.tableView beginUpdates];
    [self.tableView reloadRowsAtIndexPaths:@[[NSIndexPath indexPathForRow:row inSection:section]]
      withRowAnimation:UITableViewRowAnimationNone];
    if (row > 0) {
      data[section].item[row-1].val = x->PickedString();
      [self.tableView reloadRowsAtIndexPaths:@[[NSIndexPath indexPathForRow:row-1 inSection:section]]
       withRowAnimation:UITableViewRowAnimationTop];
    }
    [self.tableView endUpdates];
  }

  - (LFL::PickerItem*)getPicker:(int)section row:(int)r {
    [self checkExists:section row:r];
    NSIndexPath *path = [NSIndexPath indexPathForRow:r inSection:section];
    UITableViewCell *cell = [self.tableView cellForRowAtIndexPath:path];
    IOSPicker *picker_control = [[cell.contentView subviews] lastObject];
    return [picker_control getItem];
  }
  
  - (std::string)getVal:(int)section row:(int)r {
    NSIndexPath *path = [NSIndexPath indexPathForRow: r inSection: section];
    UITableViewCell *cell = [self.tableView cellForRowAtIndexPath: path];
    auto &ci = data[path.section].item[path.row];

    std::string val;
    if (!(ci.flags & iOSTableItem::GUILoaded)) {
      if (ci.val.size() && ci.val[0] != 1) val = ci.val[0] == 2 ? ci.val.substr(1) : ci.val;
    } else if ((ci.type == LFL::TableItem::TextInput) || (ci.type == LFL::TableItem::NumberInput) ||
               (ci.type == LFL::TableItem::PasswordInput)) {
      IOSTextField *textfield = LFL::objc_dynamic_cast<IOSTextField>(cell.accessoryView);
      val = LFL::GetNSString(textfield.text);
      if (val.empty() && !textfield.modified) {
        if (ci.val.size() && ci.val[0] != 1) val = ci.val[0] == 2 ? ci.val.substr(1) : ci.val;
      }
    } else if (ci.type == LFL::TableItem::Label) {
      UILabel *label = LFL::objc_dynamic_cast<UILabel>(cell.accessoryView);
      val = LFL::GetNSString(label.text);
    } else if (ci.type == LFL::TableItem::Selector) {
      UISegmentedControl *segmented_control;
      if (ci.flags & LFL::TableItem::Flag::HideKey) segmented_control = [[cell.contentView subviews] lastObject];
      else segmented_control = LFL::objc_dynamic_cast<UISegmentedControl>(cell.accessoryView);
      val = LFL::GetNSString([segmented_control titleForSegmentAtIndex: segmented_control.selectedSegmentIndex]);
    } else if (ci.type == LFL::TableItem::Toggle) {
      UISwitch *onoff = LFL::objc_dynamic_cast<UISwitch>(cell.accessoryView);
      val = onoff.on ? "1" : "";
    } else if (ci.type == LFL::TableItem::Picker || ci.type == LFL::TableItem::FontPicker) {
      IOSPicker *picker_control = [[cell.contentView subviews] lastObject];
      val = [picker_control getItem]->PickedString();
    }
    return val;
  }

  - (LFL::StringPairVec)dumpDataForSection: (int)ind {
    LFL::StringPairVec ret;
    CHECK_LT(ind, data.size());
    for (int i=0, l=data[ind].item.size(); i != l; i++) {
      auto &ci = data[ind].item[i];
      if (ci.dropdown_key.size()) ret.emplace_back(ci.dropdown_key, ci.key);
      ret.emplace_back(ci.key, [self getVal:ind row:i]);
    }
    return ret;
  }

  - (void)reloadRowAtIndexPath:(NSIndexPath*)path withRowAnimation:(UITableViewRowAnimation)animation {
    [self.tableView beginUpdates];
    [self.tableView reloadRowsAtIndexPaths:@[path] withRowAnimation:animation];
    [self.tableView endUpdates];
  }
@end

@interface IOSTextView : UIViewController<UITextViewDelegate>
  @property (nonatomic, retain) UITextView *textView;
  @property (nonatomic, retain) NSString *text;
@end

@implementation IOSTextView
  - (id)initWithTitle:(NSString*)title andText:(NSString*)t {
    self = [super init];
    self.title = title;
    self.text = t;
    return self;
  }

  - (void)viewDidLoad {
    self.textView = [[[UITextView alloc] initWithFrame:self.view.frame] autorelease];
    self.textView.delegate = self;
    self.textView.textColor = [UIColor blackColor];
    //self.textView.backgroundColor = [UIColor grayColor];
    self.textView.text = self.text;
    [self.view addSubview:self.textView];
  }
@end

namespace LFL {
struct iOSAlertView : public SystemAlertView {
  IOSAlert *alert;
  ~iOSAlertView() { [alert release]; }
  iOSAlertView(AlertItemVec items) : alert([[IOSAlert alloc] init: move(items)]) {}

  void Show(const string &arg) {
    if (alert.add_text) [alert.alert textFieldAtIndex:0].text = MakeNSString(arg);
    [alert.alert show];
  }

  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
    alert.confirm_cb = move(confirm_cb);
    alert.alert.title = MakeNSString(title);
    alert.alert.message = MakeNSString(msg);
    if (alert.add_text) [alert.alert textFieldAtIndex:0].text = MakeNSString(arg);
    [alert.alert show];
  }

  string RunModal(const string &arg) {
    if (alert.add_text) [alert.alert textFieldAtIndex:0].text = MakeNSString(arg);
    alert.done = false;
    [alert.alert show];
    NSRunLoop *rl = [NSRunLoop currentRunLoop];
    // do { [rl runMode:NSRunLoopCommonModes beforeDate:[NSDate distantFuture]]; }
    do { [rl runMode:NSRunLoopCommonModes beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.3]]; }
    while(!alert.done);
    return alert.add_text ? GetNSString([alert.alert textFieldAtIndex:0].text) : "";
  }
};

struct iOSMenuView : public SystemMenuView {
  IOSMenu *menu;
  ~iOSMenuView() { [menu release]; }
  iOSMenuView(const string &t, MenuItemVec i) : menu([[IOSMenu alloc] init:t items:move(i)]) {}
  void Show() { [menu.actions showInView: [UIApplication sharedApplication].keyWindow]; }
};

struct iOSToolbarView : public SystemToolbarView {
  IOSToolbar *toolbar;
  ~iOSToolbarView() { [toolbar release]; }
  iOSToolbarView(MenuItemVec items) : toolbar([[IOSToolbar alloc] init: move(items)]) {}
  void Show(bool show_or_hide) { [toolbar show:show_or_hide]; }
  void ToggleButton(const string &n) { [toolbar toggleButtonNamed: n]; }
};

struct iOSTextView : public SystemTextView {
  IOSTextView *view;
  ~iOSTextView() { [view release]; }
  iOSTextView(const string &title, File *f) : iOSTextView(title, f ? f->Contents() : "") {}
  iOSTextView(const string &title, const string &text) :
    view([[IOSTextView alloc] initWithTitle:MakeNSString(title) andText:[[[NSString alloc]
         initWithBytes:text.data() length:text.size() encoding:NSASCIIStringEncoding] autorelease]]) {}
};

struct iOSNavigationView : public SystemNavigationView {
  IOSNavigation *nav;
  ~iOSNavigationView() { [nav release]; }
  iOSNavigationView() : nav([[IOSNavigation alloc] initWithNavigationBarClass:nil toolbarClass:nil]) {
    [nav setToolbarHidden:YES animated:NO];
  }

  void Show(bool show_or_hide) {
    LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
    if ((shown = show_or_hide)) {
      if (root->show_cb) root->show_cb();
      INFO("LFViewController.presentViewController IOSNavigation frame=", LFL::GetCGRect(uiapp.controller.view.frame).DebugString());
      uiapp.top_controller = nav;
      if (uiapp.controller.presentedViewController != nav)
        [uiapp.controller presentViewController:nav animated:YES completion:nil];
    } else {
      INFO("LFViewController.dismissViewController ", GetNSString(NSStringFromClass([uiapp.controller.presentedViewController class])), " frame=", LFL::GetCGRect(uiapp.controller.view.frame).DebugString());
      uiapp.top_controller = uiapp.root_controller;
      [uiapp.controller dismissViewControllerAnimated:YES completion:nil];
    }
  }

  SystemTableView *Back() {
    for (UIViewController *c in [nav.viewControllers reverseObjectEnumerator]) {
      if ([c isKindOfClass:[IOSTable class]])
        if (auto lself = static_cast<IOSTable*>(c).lfl_self) return lself;
    } 
    return nullptr;
  }

  void PushTableView(SystemTableView *t) {
    if (!root && last_root == t) { root = t; return; }
    if (t->show_cb) t->show_cb();
    [nav pushViewController: dynamic_cast<iOSTableView*>(t)->table animated: YES];
    if (root) return;
    root = t;
    int children = [nav.viewControllers count];
    if (children == 1) return;
    CHECK_EQ(2, children);
    NSMutableArray *vc = [nav.viewControllers mutableCopy];
    [vc removeObjectAtIndex:0];
    nav.viewControllers = vc;
  }

  void PushTextView(SystemTextView *t) {
    if (t->show_cb) t->show_cb();
    [nav pushViewController: dynamic_cast<iOSTextView*>(t)->view animated: YES];
  }

  void PopToRoot() {
    if (root) [nav popToRootViewControllerAnimated: YES];
  }

  void PopAll() {
    if (!root) return;
    last_root = root;
    root = 0;
    [nav popToRootViewControllerAnimated: NO];
    if (last_root && last_root->hide_cb) last_root->hide_cb();
  }

  void PopView(int n) {
    for (int i = 0; i != n; ++i)
      [nav popViewControllerAnimated: (i == n - 1)];
  }
};

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &cb) {
  static IOSFontPicker *font_chooser = [[IOSFontPicker alloc] init];
  [font_chooser selectFont:cur_font.name size:cur_font.size cb:cb];
  [[LFUIApplication sharedAppDelegate].glk_view addSubview: font_chooser];
}

void Application::ShowSystemContextMenu(const MenuItemVec &items) {
  auto uiapp = [LFUIApplication sharedAppDelegate];
  UIMenuController* mc = [UIMenuController sharedMenuController];
  vector<UIMenuItem*> menuitems;
  for (auto &i : items) {
    if      (i.name == "Copy") { uiapp.text_field.copy_cb = i.cb; continue; }
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
  CGRect rect = CGRectMake(w->mouse.x / s, (w->height + w->y - w->mouse.y) / s, w->default_font->Height(), 100);
  [mc setTargetRect:rect inView:dynamic_cast<iOSWindow*>(w)->glkview];
  [mc setMenuVisible:![mc isMenuVisible] animated:TRUE];
}

int Application::LoadSystemImage(const string &n) {
  if (n.empty()) {
    app_images.push_back(nullptr);
    return app_images.size();
  }
  UIImage *image = [UIImage imageNamed:MakeNSString(StrCat("drawable-xhdpi/", n, ".png"))];
  if (!image) return 0;
  [image retain];
  app_images.push_back(image);
  return app_images.size();
}

void Application::UpdateSystemImage(int n, Texture &t) {
  CHECK_RANGE(n-1, 0, app_images.size());
  CGImageRef image = MakeCGImage(t);
  if (app_images[n-1]) [app_images[n-1] release];
  if (!(app_images[n-1] = [[UIImage alloc] initWithCGImage:image])) ERROR("UpdateSystemImage failed");
  CGImageRelease(image);
}

iOSTableView::~iOSTableView() { [table release]; }
iOSTableView::iOSTableView(const string &title, const string &style, TableItemVec items) :
  table([[IOSTable alloc] initWithStyle: UITableViewStyleGrouped]) {
    [table load:this withTitle:title withStyle:style items:TableSection::Convert(move(items))];
  }

void iOSTableView::DelNavigationButton(int align) { return [table clearNavigationButton:align]; }
void iOSTableView::AddNavigationButton(int align, const TableItem &item) { return [table loadNavigationButton:item withAlign:align]; }

void iOSTableView::SetToolbar(SystemToolbarView *t) {
  int toolbar_height = 44;
  CGRect frame = table.tableView.frame;
  if (table.toolbar) {
    [table.toolbar removeFromSuperview];
    table.toolbar = nil;
    frame.size.height += toolbar_height;
  }
  if (t) {
    table.toolbar = [dynamic_cast<iOSToolbarView*>(t)->toolbar createUIToolbar:
      CGRectMake(frame.origin.x, frame.origin.y+frame.size.height-toolbar_height, frame.size.width, toolbar_height) first: YES];
    [table.view addSubview: table.toolbar];
    frame.size.height -= toolbar_height;
  }
  table.tableView.frame = frame;
}

void iOSTableView::Show(bool show_or_hide) {
  if (show_or_hide && show_cb) show_cb();
  [table show:show_or_hide];
}

void iOSTableView::AddRow(int section, TableItem item) { return [table addRow:section withItem:move(item)]; }
string iOSTableView::GetKey(int section, int row) { return [table getKey:section row:row]; }
string iOSTableView::GetValue(int section, int row) { return [table getVal:section row:row]; }
int iOSTableView::GetTag(int section, int row) { return [table getTag:section row:row]; }
void iOSTableView::SetHeader(int section, TableItem h) { return [table setHeader:section header:move(h)]; }
void iOSTableView::SetKey(int section, int row, const string &val) { [table setKey:section row:row val:val]; }
void iOSTableView::SetTag(int section, int row, int val) { [table setTag:section row:row val:val]; }
void iOSTableView::SetValue(int section, int row, const string &val) { [table setValue:section row:row val:val]; }
void iOSTableView::SetSelected(int section, int row, int val) { [table setSelected:section row:row val:val]; }
void iOSTableView::SetHidden(int section, int row, bool val) { [table setHidden:section row:row val:val]; }
void iOSTableView::SetTitle(const string &title) { table.title = LFL::MakeNSString(title); }
void iOSTableView::SetTheme(const string &theme) { [table setTheme:theme]; }
PickerItem *iOSTableView::GetPicker(int section, int row) { return [table getPicker:section row:row]; }
StringPairVec iOSTableView::GetSectionText(int section) { return [table dumpDataForSection:section]; }
void iOSTableView::SetEditableSection(int section, int start_row, LFL::IntIntCB cb) {
  table.delete_row_cb = move(cb);
  table.editable_section = section;
  table.editable_start_row = start_row;
}

void iOSTableView::SelectRow(int section, int row) {
  table.selected_section = section;
  table.selected_row = row;
} 

void iOSTableView::BeginUpdates() { [table.tableView beginUpdates]; }
void iOSTableView::EndUpdates() { [table.tableView endUpdates]; }
void iOSTableView::SetSectionValues(int section, const StringVec &item) { [table setSectionValues:section items:item]; }
void iOSTableView::ApplyChangeList(const TableSection::ChangeList &changes) { [table applyChangeList:changes]; }
void iOSTableView::ReplaceRow(int section, int row, TableItem h) { [table replaceRow:section row:row val:move(h)]; }
void iOSTableView::ReplaceSection(int section, TableItem h, int flag, TableItemVec item)
{ [table replaceSection:section items:move(item) header:move(h) flag:flag]; }

unique_ptr<SystemAlertView> SystemAlertView::Create(AlertItemVec items) { return make_unique<iOSAlertView>(move(items)); }
unique_ptr<SystemPanelView> SystemPanelView::Create(const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<SystemToolbarView> SystemToolbarView::Create(MenuItemVec items) { return make_unique<iOSToolbarView>(move(items)); }
unique_ptr<SystemMenuView> SystemMenuView::Create(const string &title, MenuItemVec items) { return make_unique<iOSMenuView>(title, move(items)); }
unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(vector<MenuItem> items) { return nullptr; }
unique_ptr<SystemTableView> SystemTableView::Create(const string &title, const string &style, TableItemVec items) { return make_unique<iOSTableView>(title, style, move(items)); }
unique_ptr<SystemTextView> SystemTextView::Create(const string &title, File *file) { return make_unique<iOSTextView>(title, file); }
unique_ptr<SystemTextView> SystemTextView::Create(const string &title, const string &text) { return make_unique<iOSTextView>(title, text); }
unique_ptr<SystemNavigationView> SystemNavigationView::Create() { return make_unique<iOSNavigationView>(); }

}; // namespace LFL
