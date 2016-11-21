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

@interface IOSButton : UIButton 
  @property (nonatomic, assign) LFL::Callback cb;
  - (void)buttonClicked:(IOSButton*)sender;
@end
@implementation IOSButton
  - (void)buttonClicked:(IOSButton*)sender {
    if (sender.cb) sender.cb();
  }
@end

@interface IOSBarButtonItem : UIBarButtonItem
  @property (nonatomic, assign) LFL::Callback cb;
  - (IBAction)buttonClicked:(IOSBarButtonItem*)sender;
@end
@implementation IOSBarButtonItem
  - (IBAction)buttonClicked:(IOSBarButtonItem*)sender {
    if (sender.cb) sender.cb();
  }
@end

@interface IOSSegmentedControl : UISegmentedControl
  @property (copy) void (^changed)(const std::string&);
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

@interface IOSAlert : NSObject<UIAlertViewDelegate>
  @property (nonatomic, retain) UIAlertView *alert;
  @property (nonatomic)         bool         add_text, done;
  @property (nonatomic)         std::string  style;
  @property (nonatomic, assign) LFL::StringCB cancel_cb, confirm_cb;
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
    menu[buttonIndex-1].cb();
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
    _toolbar = [self createUIToolbar: [self getToolbarFrame]];
    _toolbar2 = [self createUIToolbar: [self getToolbarFrame]];
    return self;
  }

  - (UIToolbar*)createUIToolbar:(CGRect)rect {
    NSMutableArray *items = [[NSMutableArray alloc] init];
    UIBarButtonItem *spacer = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemFlexibleSpace target:nil action:nil];
    for (int i=0, l=data.size(); i<l; i++) {
      if (i) [items addObject: spacer];
      NSString *K = [NSString stringWithUTF8String: data[i].shortcut.c_str()];
      UIBarButtonItem *item =
        [[UIBarButtonItem alloc] initWithTitle:(([K length] && LFL::isascii([K characterAtIndex:0])) ? [NSString stringWithFormat:@"%@", K] : [NSString stringWithFormat:@"%@\U0000FE0E", K])
        style:UIBarButtonItemStylePlain target:self action:@selector(onClick:)];
      [item setTag:i];
      [items addObject:item];
      [item release];
    }

    UIToolbar *tb = [[UIToolbar alloc] initWithFrame: rect];
    // [tb setBarStyle:UIBarStyleBlackTranslucent];
    tb.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleTopMargin;
    [tb setItems:items];
    [items release];
    [spacer release];
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
    b.cb();
    if (b.name == "toggle") [self toggleButton: item.tag];
    [[LFUIApplication sharedAppDelegate].controller resignFirstResponder];
  }

  - (void)show: (bool)show_or_hide {
    auto uiapp = [LFUIApplication sharedAppDelegate];
    uiapp.controller.input_accessory_toolbar = show_or_hide ? _toolbar : nil;
    uiapp.text_field.inputAccessoryView = show_or_hide ? _toolbar2 : nil;
    [uiapp.text_field reloadInputViews];
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
      font_change_cb(LFL::StringVec{x->data[0][x->picked[0]], x->data[1][x->picked[1]]});
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

@interface IOSNavigation : UINavigationController<UINavigationControllerDelegate>
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

@interface IOSTable : UITableViewController
  @property (nonatomic, retain) UIView *header;
  @property (nonatomic, retain) UILabel *header_label;
  @property (nonatomic, retain) UINavigationController *modal_nav;
  @property (nonatomic, assign) IOSToolbar *toolbar;
  @property (nonatomic, assign) LFL::SystemTableView *lfl_self;
  @property (nonatomic, assign) LFL::IntIntCB delete_row_cb;
  @property (nonatomic)         std::string style;
  @property (nonatomic)         int editable_section, editable_start_row, selected_section, selected_row,
                                    second_col;
  @property (copy)              void (^changed)(const std::string&);
  @property (copy)              void (^completed)();
@end

@implementation IOSTable
  {
    std::vector<LFL::Table> data;
    std::vector<IOSTable*> dropdowns;
  }

  - (void)load:(LFL::SystemTableView*)lself withTitle:(const std::string&)title withStyle:(const std::string&)sty items:(std::vector<LFL::Table>)item {
    _lfl_self = lself;
    _style = sty;
    _editable_section = _editable_start_row = -1;
    data = move(item);
    self.title = LFL::MakeNSString(title);

    if (_style != "indent") self.tableView.separatorInset = UIEdgeInsetsZero;
    [self.tableView setSeparatorStyle:UITableViewCellSeparatorStyleSingleLine];
    [self.tableView setSeparatorColor:[UIColor grayColor]];
    if (_style == "modal" || _style == "dropdown") {
      _modal_nav = [[UINavigationController alloc] initWithRootViewController: self];
      _modal_nav.modalTransitionStyle = UIModalTransitionStyleCoverVertical;
    }

    if (_second_col < 0) {
      UIFont *font = [UIFont boldSystemFontOfSize:[UIFont labelFontSize]];
      NSDictionary *attr = @{NSFontAttributeName:font, NSForegroundColorAttributeName:[UIColor blackColor]};
      for (auto &d : data) {
        for (auto &i : d.item) {
          const CGSize size = [LFL::MakeNSString(i.key) sizeWithAttributes:attr];
          if (size.width > _second_col) _second_col = size.width;
        }
      }
      _second_col += 30;
    }
  }

  - (void)addRow:(int)section withItem:(LFL::TableItem)item {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    data[section].item.emplace_back(move(item));
    NSIndexPath *path = [NSIndexPath indexPathForRow:data[section].item.size()-1 inSection:section];
    [self.tableView insertRowsAtIndexPaths:@[path] withRowAnimation:UITableViewRowAnimationNone];
  }

  - (void)replaceSection:(int)section items:(std::vector<LFL::TableItem>)item header:(const std::string&)h image:(int)im flag:(int)f addbutton:(LFL::Callback)addb {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    data[section] = LFL::Table(h, im, f, move(addb));
    data[section].item = move(item);
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex: section]
      withRowAnimation:UITableViewRowAnimationNone];
  }

  - (void)checkExists:(int)section row:(int)r {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_LT(r, data[section].item.size());
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
    if (!ci.loaded) return;
    if (ci.type == LFL::TableItem::DropdownKey || ci.type == LFL::TableItem::DropdownValue ||
        ci.type == LFL::TableItem::FixedDropdown) {
      std::vector<std::string> vv=LFL::Split(v, ',');
      CHECK_RANGE(ci.ref, 0, dropdowns.size());
      auto dropdown_table = dropdowns[ci.ref];
      CHECK_EQ(dropdown_table->data[0].item.size(), vv.size()-1) << ": " << v;
      for (int j=1; j < vv.size(); ++j) dropdown_table->data[0].item[j-1].val = vv[j];
    } else if (ci.type == LFL::TableItem::Picker) {
    }
  }

  - (void)setDropdown:(int)section row:(int)r index:(int)ind {
    [self checkExists:section row:r];
    NSIndexPath *path = [NSIndexPath indexPathForRow:r inSection:section];
    UITableViewCell *cell = [self.tableView cellForRowAtIndexPath: path];
    int type=0;
    const std::string *k=0, *v=0;
    auto &ci = data[path.section].item[path.row];
    [self loadCellItem:cell withPath:path withItem:&ci outK:&k outT:&type outV:&v];
    if (ci.type == LFL::TableItem::DropdownKey || ci.type == LFL::TableItem::DropdownValue ||
        ci.type == LFL::TableItem::FixedDropdown) {
      CHECK_RANGE(ci.ref, 0, dropdowns.size());
      auto dropdown_table = dropdowns[ci.ref];
      CHECK_EQ(0, dropdown_table.selected_section);
      CHECK_LT(ind, dropdown_table->data[0].item.size());
      dropdown_table.selected_row = ind;
      [self.tableView reloadRowsAtIndexPaths:@[path] withRowAnimation:UITableViewRowAnimationNone];
      if (dropdown_table.changed) dropdown_table.changed(dropdown_table->data[0].item[ind].key);
    } 
  }

  - (void)setSectionValues:(int)section items:(const LFL::StringVec&)item {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_EQ(item.size(), data[section].item.size());
    for (int i=0, l=data[section].item.size(); i != l; ++i) [self setValue:section row:i val:item[i]];
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex: section]
      withRowAnimation:UITableViewRowAnimationNone];
  }

  - (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView { return data.size(); }
  - (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    return data[section].item.size();
  }

  - (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)path {
    if (path.section >= data.size() || path.row >= data[path.section].item.size()) return tableView.rowHeight;
    const auto &ci = data[path.section].item[path.row];
    if (ci.hidden) return 0;
    else if (ci.gui_loaded &&
             (ci.type == LFL::TableItem::Picker || ci.type == LFL::TableItem::FontPicker)) return ci.height;
    else return tableView.rowHeight;
  }

  - (CGRect)getCellFrame {
    if (_second_col) return CGRectMake(_second_col, 0, self.tableView.frame.size.width - _second_col - 30, 44);
    else             return CGRectMake(0, 0, 200, 44);
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

  - (void)loadCellItem:(UITableViewCell*)cell withPath:(NSIndexPath*)path withItem:(LFL::TableItem*)item
                       outK:(const std::string**)ok outT:(int*)ot outV:(const std::string**)ov {
    if (!item->loaded) {
      item->loaded = true;
      if (item->type == LFL::TableItem::DropdownKey || item->type == LFL::TableItem::DropdownValue ||
          item->type == LFL::TableItem::FixedDropdown) {
        item->ref = dropdowns.size();
        auto dropdown_table = [[IOSTable alloc] initWithStyle: UITableViewStyleGrouped];
        [dropdown_table load:nullptr withTitle:item->key withStyle:"dropdown" items:LFL::Table::Convert(item->MoveChildren())];
        [dropdown_table loadNavigationButton:
          LFL::TableItem("Back", LFL::TableItem::Button, "", "", 0, 0, 0, [=](){ [dropdown_table show: false]; })
          withAlign: LFL::HAlign::Left];
        if (item->depends.size()) dropdown_table.changed = [self makeChangedCB: *item];
        dropdown_table.completed = ^{ [self reloadRowAtIndexPath:path withRowAnimation:UITableViewRowAnimationNone]; };
        dropdowns.push_back(dropdown_table);
      }
    }

    const LFL::TableItem *ret;
    bool dropdown_value = item->type == LFL::TableItem::DropdownValue;
    if (dropdown_value || item->type == LFL::TableItem::DropdownKey || item->type == LFL::TableItem::FixedDropdown) {
      CHECK_RANGE(item->ref, 0, dropdowns.size());
      auto dropdown_table = dropdowns[item->ref];
      CHECK_EQ(0, dropdown_table.selected_section);
      CHECK_RANGE(dropdown_table.selected_row, 0, dropdown_table->data[0].item.size());
      ret = &dropdown_table->data[0].item[dropdown_table.selected_row];
    } else ret = item;

    bool parent_dropdown = _style == "dropdown";
    *ok = dropdown_value  ? &item->key                         : &ret->key;
    *ot = parent_dropdown ? 0                                  :  ret->type;
    *ov = parent_dropdown ? LFL::Singleton<std::string>::Get() : &ret->val;
  }

  - (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)path {
    static NSString *cellIdentifier = @"cellIdentifier";
    UITableViewCell *cell = [self.tableView dequeueReusableCellWithIdentifier:cellIdentifier];
    if (cell) { [cell release]; cell = nil; }
    if (cell == nil) {
      int type = 0, row = path.row, section = path.section;
      CHECK_LT(section, data.size());
      CHECK_LT(row, data[section].item.size());
      cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleDefault reuseIdentifier:cellIdentifier];
      cell.selectionStyle = UITableViewCellSelectionStyleNone;

      const std::string *k=0, *v=0;
      auto &compiled_item = data[section].item[row];
      [self loadCellItem:cell withPath:path withItem:&compiled_item outK:&k outT:&type outV:&v];
      compiled_item.gui_loaded = true;
      UIColor *blue = [UIColor colorWithRed:0.0/255 green:122.0/255 blue:255.0/255 alpha:1];

      if (type != LFL::TableItem::Button) {
        if (int icon = compiled_item.left_icon) {
          CHECK_LE(icon, app_images.size());
          cell.imageView.image = app_images[icon - 1]; 
        }
      }

      if (compiled_item.type == LFL::TableItem::DropdownKey || compiled_item.type == LFL::TableItem::DropdownValue) {
        int w = 10, x = _second_col ? _second_col - w : [LFL::MakeNSString(*k) sizeWithAttributes:
          @{NSFontAttributeName:[UIFont boldSystemFontOfSize:[UIFont labelFontSize]]}].width + 20 +
          (cell.imageView.image ? 60 : 0);
        IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
        button.frame = CGRectMake(0, 0, x+w, 40.0);
        if (compiled_item.cb) {
          [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
          button.cb = [=](){ data[section].item[row].cb(); };
        } else {
          [button addTarget:self action:@selector(dropDownClicked:) forControlEvents:UIControlEventTouchUpInside];
        }
        [button setTag: compiled_item.ref];
        [cell.contentView addSubview: button];
      }

      bool textinput=0, numinput=0, pwinput=0;
      if ((textinput = type == LFL::TableItem::TextInput) || (numinput = type == LFL::TableItem::NumberInput)
          || (pwinput = type == LFL::TableItem::PasswordInput)) {
        IOSTextField *textfield = [[IOSTextField alloc] initWithFrame: [self getCellFrame]];
        textfield.autoresizingMask = UIViewAutoresizingFlexibleHeight;
        textfield.adjustsFontSizeToFitWidth = YES;
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

        if (v->size() && ((*v)[0] == 1 || (*v)[0] == 2)) [textfield setPlaceholder: LFL::MakeNSString(v->substr(1))];
        else if (v->size())                              [textfield setText:        LFL::MakeNSString(*v)];

        cell.textLabel.text = LFL::MakeNSString(*k);
        if (_second_col) [cell.contentView addSubview: textfield];
        else {
          textfield.textAlignment = NSTextAlignmentRight;
          cell.accessoryView = textfield;
        }
        if (section == _selected_section && row == _selected_row) [textfield becomeFirstResponder];
        [textfield release];

      } else if (type == LFL::TableItem::Selector) {
        NSArray *itemArray = LFL::MakeNSStringArray(LFL::Split(*v, ','));
        IOSSegmentedControl *segmented_control = [[IOSSegmentedControl alloc] initWithItems:itemArray];
        segmented_control.frame = cell.frame;
        segmented_control.autoresizingMask = UIViewAutoresizingFlexibleWidth;
        segmented_control.selectedSegmentIndex = 0; 
        if (compiled_item.depends.size()) 
          segmented_control.changed = [self makeChangedCB: compiled_item];
        [segmented_control addTarget:self action:@selector(segmentedControlClicked:)
          forControlEvents: UIControlEventValueChanged];
        [cell.contentView addSubview:segmented_control];
        [segmented_control release]; 

      } else if (type == LFL::TableItem::Picker || type == LFL::TableItem::FontPicker) {
        LFL::PickerItem item;
        if (type == LFL::TableItem::Picker) item = *compiled_item.picker;
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
        compiled_item.height = picker.frame.size.height;

      } else if (type == LFL::TableItem::Button) {
        if (int icon = compiled_item.left_icon) {
          CHECK_LE(icon, app_images.size());
          UIImage *image = app_images[icon - 1]; 
          IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
          [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
          button.cb = [=](){ data[section].item[row].cb(); };
          button.frame = cell.frame;
          int spacing = -10, target_height = 40, margin = fabs(button.frame.size.height - target_height) / 2;
          // [button setFont:[UIFont boldSystemFontOfSize:[UIFont labelFontSize]]];
          [button setTitleColor:blue forState:UIControlStateNormal];
          [button setTitle:LFL::MakeNSString(*k) forState:UIControlStateNormal];
          [button setImage:image forState:UIControlStateNormal];
          button.imageView.contentMode = UIViewContentModeScaleAspectFit;
          [button setTitleEdgeInsets:UIEdgeInsetsMake(0, spacing, 0, 0)];
          [button setImageEdgeInsets:UIEdgeInsetsMake(margin, 0, margin, spacing)];
          [cell.contentView addSubview:button];
          [button release];
        } else {
          cell.textLabel.text = LFL::MakeNSString(*k);
          cell.textLabel.textAlignment = NSTextAlignmentCenter;
        }

      } else if (type == LFL::TableItem::Toggle) {
        UISwitch *onoff = [[UISwitch alloc] init];
        onoff.on = *v == "1";
        [onoff addTarget: self action: @selector(switchFlipped:) forControlEvents: UIControlEventValueChanged];
        cell.textLabel.text = LFL::MakeNSString(*k);
        cell.accessoryView = onoff;
        [onoff release];

      } else if (type == LFL::TableItem::Label) {
        UILabel *label = [[UILabel alloc] initWithFrame: [self getCellFrame]];
        label.text = LFL::MakeNSString(*v);
        label.adjustsFontSizeToFitWidth = TRUE;
        cell.textLabel.text = LFL::MakeNSString(*k);
        if (_second_col) [cell.contentView addSubview: label];
        else {
          label.textAlignment = NSTextAlignmentRight;
          cell.accessoryView = label;
        }
        [label release];

      } else {
        cell.textLabel.text = LFL::MakeNSString(*k);
      }

      if (compiled_item.type == LFL::TableItem::DropdownKey && cell.textLabel.text.length) {
        cell.textLabel.textColor = blue;
        cell.textLabel.text = [NSString stringWithFormat:@"%@  \U000002C5", cell.textLabel.text];
      }

      if (int icon = compiled_item.right_icon) {
        CHECK_LE(icon, app_images.size());
        UIImage *image = app_images[icon - 1]; 
        IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
        button.frame = CGRectMake(0, 0, 40, 40);
        [button setImage:image forState:UIControlStateNormal];
        button.cb = compiled_item.right_icon_cb;
        [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
        if (cell.accessoryView) [cell.accessoryView addSubview: button];
        else cell.accessoryView = button;
      } else if (compiled_item.right_text.size()) {
        UILabel *label = [[UILabel alloc] init];
        label.textColor = self.view.tintColor;
        label.text = LFL::MakeNSString(compiled_item.right_text);
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
      _delete_row_cb(path.row, data[path.section].item[path.row].tag);
      data[path.section].item.erase(data[path.section].item.begin() + path.row);
      [tableView beginUpdates];
      [tableView deleteRowsAtIndexPaths:@[path] withRowAnimation:UITableViewRowAnimationNone];
      [tableView endUpdates];
    }
  }

  - (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    return LFL::MakeNSString(data[section].header);
  }

  - (CGFloat)tableView:(UITableView *)tableView heightForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    if (int h = data[section].header_height) return h;
    if (int image = data[section].image) {
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

    if (int image = data[section].image) {
      UIImageView *image_view = [[UIImageView alloc] initWithImage: app_images[image - 1]];
      data[section].header_height = 44 + image_view.frame.size.height;
      image_view.contentMode = UIViewContentModeCenter;
      image_view.center = CGPointMake(headerView.frame.size.width/2, image_view.frame.size.height/2);
      image_view.autoresizingMask = UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleLeftMargin;
      headerView.frame = CGRectMake(0, 0, tableView.frame.size.width, data[section].header_height);
      [headerView addSubview: image_view];
      [image_view release];
    }

    if (data[section].header.size()) {
      UILabel *label = [[UILabel alloc] initWithFrame:
        CGRectMake(50, headerView.frame.size.height-1-21-11, headerView.frame.size.width-100, 21)];
      label.text = LFL::MakeNSString(data[section].header);
      label.textAlignment = NSTextAlignmentCenter;
      label.autoresizingMask = UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleLeftMargin;
      [headerView addSubview:label];
      [label release];
    }

    if (data[section].add_cb) {
      IOSButton *button = [IOSButton buttonWithType:UIButtonTypeSystem];
      [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
      [button setTitle:@"Add" forState:UIControlStateNormal];
      [button sizeToFit];
      [button setFrame:CGRectMake(11, 11, button.frame.size.width, 21)];
      button.autoresizingMask = UIViewAutoresizingFlexibleRightMargin;
      [headerView addSubview:button];
    }

    if (data[section].flag & LFL::SystemTableView::EditButton) {
      UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
      [button addTarget:self action:@selector(toggleEditMode:) forControlEvents:UIControlEventTouchUpInside];
      [button setTitle:@"Edit" forState:UIControlStateNormal];
      [button sizeToFit];
      [button setFrame:CGRectMake(tableView.frame.size.width - button.frame.size.width - 11, 11, button.frame.size.width, 21)];
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
    return (data[path.section].flag & LFL::SystemTableView::EditableIfHasTag && !data[path.section].item[path.row].tag) ?
      UITableViewCellEditingStyleNone : UITableViewCellEditingStyleDelete;
  }

  - (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)path {
    [self checkExists:path.section row:path.row];
    [self.tableView deselectRowAtIndexPath:path animated:NO];
    _selected_row = path.row;
    _selected_section = path.section;
    auto &compiled_item = data[path.section].item[path.row];
    if (_modal_nav) {
      [self show: false];
      if (_changed) {
        [self.tableView beginUpdates];
        _changed(compiled_item.key);
        [self.tableView endUpdates];
      }
      if (_completed) _completed();
    }
    if (compiled_item.type == LFL::TableItem::Command || compiled_item.type == LFL::TableItem::Button) {
      compiled_item.cb();
    } else if (compiled_item.type == LFL::TableItem::Label && path.row + 1 < data[path.section].item.size()) {
      auto &next_compiled_item = data[path.section].item[path.row+1];
      if (next_compiled_item.type == LFL::TableItem::Picker ||
          next_compiled_item.type == LFL::TableItem::FontPicker) {
        next_compiled_item.hidden = !next_compiled_item.hidden;
        [self reloadRowAtIndexPath:path withRowAnimation:UITableViewRowAnimationNone];
      }
    }
  }

  - (void)show:(bool)show_or_hide {
    auto uiapp = [LFUIApplication sharedAppDelegate];
    if (show_or_hide) {
      if (_modal_nav) {
        if (auto nav = LFL::objc_dynamic_cast<IOSNavigation>(uiapp.top_controller)) [nav pushViewController:self animated:YES];
        else [uiapp.top_controller presentViewController:self.modal_nav animated:YES completion:nil];
      } else [uiapp.glk_view addSubview: self.tableView];
    } else {
      if (_modal_nav) {
        if (auto nav = LFL::objc_dynamic_cast<IOSNavigation>(uiapp.top_controller)) [nav popViewControllerAnimated:YES];
        else [uiapp.top_controller dismissViewControllerAnimated:YES completion:nil];
      } else [self.tableView removeFromSuperview];
    }
  }

  - (void)viewWillAppear:   (BOOL)animated { [super viewWillAppear:    animated]; if (_toolbar) [_toolbar show: true];  }
  - (void)viewWillDisappear:(BOOL)animated { [super viewWillDisappear: animated]; if (_toolbar) [_toolbar show: false]; }

  - (void)textFieldDidChange:(IOSTextField*)sender {
    _lfl_self->changed = true;
    [sender textFieldDidChange: sender];
  }

  - (void)dropDownClicked:(IOSButton *)sender {
    _lfl_self->changed = true;
    int dropdown_ind = sender.tag;
    CHECK_RANGE(dropdown_ind, 0, dropdowns.size());
    auto dropdown_table = dropdowns[dropdown_ind];
    [dropdown_table show:true];
  }

  - (void)segmentedControlClicked:(IOSSegmentedControl *)segmented_control {
    _lfl_self->changed = true;
    if (segmented_control.changed) segmented_control.changed
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

  - (void)willMoveToParentViewController:(UIViewController *)parent {
    if (parent == nil && _lfl_self && _lfl_self->hide_cb) _lfl_self->hide_cb();
  }

  - (LFL::PickerItem*)getPicker:(int)section row:(int)r {
    [self checkExists:section row:r];
    NSIndexPath *path = [NSIndexPath indexPathForRow:r inSection:section];
    UITableViewCell *cell = [self.tableView cellForRowAtIndexPath:path];
    IOSPicker *picker_control = [[cell.contentView subviews] lastObject];
    return [picker_control getItem];
  }

  - (LFL::StringPairVec)dumpDataForSection: (int)ind {
    LFL::StringPairVec ret;
    CHECK_LT(ind, data.size());
    for (int i=0, l=data[ind].item.size(); i != l; i++) {
      NSIndexPath *path = [NSIndexPath indexPathForRow: i inSection: ind];
      UITableViewCell *cell = [self.tableView cellForRowAtIndexPath: path];
      int type=0;
      const std::string *k=0, *v=0;
      auto &compiled_item = data[path.section].item[path.row];
      [self loadCellItem:cell withPath:path withItem:&compiled_item outK:&k outT:&type outV:&v];

      if (compiled_item.type == LFL::TableItem::DropdownKey || compiled_item.type == LFL::TableItem::DropdownValue ||
          compiled_item.type == LFL::TableItem::FixedDropdown) {
        CHECK_RANGE(compiled_item.ref, 0, dropdowns.size());
        auto dropdown_table = dropdowns[compiled_item.ref];
        CHECK_EQ(0, dropdown_table.selected_section);
        CHECK_LT(dropdown_table.selected_row, dropdown_table->data[0].item.size());
        ret.emplace_back(LFL::GetNSString(dropdown_table.title),
                         dropdown_table->data[0].item[dropdown_table.selected_row].key);
      }

      std::string val;
      if (!compiled_item.gui_loaded) {
          if (v->size() && (*v)[0] != 1) val = (*v)[0] == 2 ? v->substr(1) : *v;
      } else if ((type == LFL::TableItem::TextInput) || (type == LFL::TableItem::NumberInput) ||
                 (type == LFL::TableItem::PasswordInput)) {
        IOSTextField *textfield = _second_col ? [[cell.contentView subviews] lastObject] : cell.accessoryView;
        val = LFL::GetNSString(textfield.text);
        if (val.empty() && !textfield.modified) {
          if (v->size() && (*v)[0] != 1) val = (*v)[0] == 2 ? v->substr(1) : *v;
        }
      } else if (type == LFL::TableItem::Label) {
        UILabel *label = _second_col ? [[cell.contentView subviews] lastObject] : cell.accessoryView;
        val = LFL::GetNSString(label.text);
      } else if (type == LFL::TableItem::Selector) {
        UISegmentedControl *segmented_control = [[cell.contentView subviews] lastObject];
        val = LFL::GetNSString([segmented_control titleForSegmentAtIndex: segmented_control.selectedSegmentIndex]);
      } else if (type == LFL::TableItem::Toggle) {
        UISwitch *onoff = (UISwitch*)cell.accessoryView;
        val = onoff.on ? "1" : "";
      } else if (type == LFL::TableItem::Picker || type == LFL::TableItem::FontPicker) {
        IOSPicker *picker_control = [[cell.contentView subviews] lastObject];
        val = [picker_control getItem]->PickedString();
      }
      ret.emplace_back(*k, val);
    }
    return ret;
  }

  - (void(^)(const std::string&)) makeChangedCB: (const LFL::TableItem&)compiled_item  {
    return Block_copy(^(const std::string &v){
      auto it = compiled_item.depends.find(v);
      if (it == compiled_item.depends.end()) return;
      for (auto &c : it->second) {
        CHECK_LT(c.section, data.size());
        CHECK_LT(c.row, data[c.section].item.size());
        auto &ci = data[c.section].item[c.row];
        bool dropdown = c.type == LFL::TableItem::DropdownKey || c.type == LFL::TableItem::DropdownValue ||
                        c.type == LFL::TableItem::FixedDropdown;

        if (1)            ci.hidden     = c.hidden;
        if (c.left_icon)  ci.left_icon  = c.left_icon  == -1 ? 0 : c.left_icon;
        if (c.right_icon) ci.right_icon = c.right_icon == -1 ? 0 : c.right_icon;
        if (c.key.size()) ci.key        = c.key;
        if (c.cb)         ci.cb         = c.cb;
        if (c.type) {     ci.type       = c.type;
          if (dropdown) [self setDropdown:c.section row:c.row index:0];
        }
        if (dropdown) [self setValue:c.section row:c.row val:c.val];
        else          ci.val = c.val;

        NSIndexPath *p = [NSIndexPath indexPathForRow:c.row inSection:c.section];
        [self.tableView reloadRowsAtIndexPaths:@[p] withRowAnimation:UITableViewRowAnimationNone];
      }
    });
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
SystemAlertView::~SystemAlertView() { if (auto alert = FromVoid<IOSAlert*>(impl)) [alert release]; }
SystemAlertView::SystemAlertView(AlertItemVec items) : impl([[IOSAlert alloc] init: move(items)]) {}
void SystemAlertView::Show(const string &arg) {
  auto alert = FromVoid<IOSAlert*>(impl);
  if (alert.add_text) [alert.alert textFieldAtIndex:0].text = MakeNSString(arg);
  [alert.alert show];
}

void SystemAlertView::ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
  auto alert = FromVoid<IOSAlert*>(impl);
  alert.confirm_cb = move(confirm_cb);
  alert.alert.title = MakeNSString(title);
  alert.alert.message = MakeNSString(msg);
  if (alert.add_text) [alert.alert textFieldAtIndex:0].text = MakeNSString(arg);
  [alert.alert show];
}

string SystemAlertView::RunModal(const string &arg) {
  auto alert = FromVoid<IOSAlert*>(impl);
  if (alert.add_text) [alert.alert textFieldAtIndex:0].text = MakeNSString(arg);
  alert.done = false;
  [alert.alert show];
  NSRunLoop *rl = [NSRunLoop currentRunLoop];
  // do { [rl runMode:NSRunLoopCommonModes beforeDate:[NSDate distantFuture]]; }
  do { [rl runMode:NSRunLoopCommonModes beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.3]]; }
  while(!alert.done);
  return alert.add_text ? GetNSString([alert.alert textFieldAtIndex:0].text) : "";
}

SystemMenuView::~SystemMenuView() { if (auto menu = FromVoid<IOSMenu*>(impl)) [menu release]; }
SystemMenuView::SystemMenuView(const string &t, MenuItemVec i) : impl([[IOSMenu alloc] init:t items:move(i)]) {}
void SystemMenuView::Show() { [FromVoid<IOSMenu*>(impl).actions showInView:[UIApplication sharedApplication].keyWindow]; }
unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(vector<MenuItem> items) { return nullptr; }

SystemToolbarView::~SystemToolbarView() { if (auto toolbar = FromVoid<IOSToolbar*>(impl)) [toolbar release]; }
SystemToolbarView::SystemToolbarView(MenuItemVec items) : impl([[IOSToolbar alloc] init: move(items)]) {}
void SystemToolbarView::Show(bool show_or_hide) { [FromVoid<IOSToolbar*>(impl) show:show_or_hide]; }
void SystemToolbarView::ToggleButton(const string &n) { [FromVoid<IOSToolbar*>(impl) toggleButtonNamed: n]; }

SystemTableView::~SystemTableView() { if (auto table = FromVoid<IOSTable*>(impl)) [table release]; }
SystemTableView::SystemTableView(const string &title, const string &style, TableItemVec items, int second_col) {
  auto table = [[IOSTable alloc] initWithStyle: UITableViewStyleGrouped];
  if (second_col) table.second_col = second_col;
  [table load:this withTitle:title withStyle:style items:Table::Convert(move(items))];
  impl = table;
}

void SystemTableView::DelNavigationButton(int align) { return [FromVoid<IOSTable*>(impl) clearNavigationButton:align]; }
void SystemTableView::AddNavigationButton(int align, const TableItem &item) { return [FromVoid<IOSTable*>(impl) loadNavigationButton:item withAlign:align]; }
void SystemTableView::AddToolbar(SystemToolbarView *t) {
  [FromVoid<IOSTable*>(impl) setToolbar: FromVoid<IOSToolbar*>(t->impl)];
  [FromVoid<IOSTable*>(impl).toolbar.toolbar setNeedsLayout];
}

void SystemTableView::Show(bool show_or_hide) {
  if (show_or_hide && show_cb) show_cb();
  [FromVoid<IOSTable*>(impl) show:show_or_hide];
}

void SystemTableView::AddRow(int section, TableItem item) { return [FromVoid<IOSTable*>(impl) addRow:section withItem:move(item)]; }
string SystemTableView::GetKey(int section, int row) { return [FromVoid<IOSTable*>(impl) getKey:section row:row]; }
int SystemTableView::GetTag(int section, int row) { return [FromVoid<IOSTable*>(impl) getTag:section row:row]; }
void SystemTableView::SetTag(int section, int row, int val) { [FromVoid<IOSTable*>(impl) setTag:section row:row val:val]; }
void SystemTableView::SetValue(int section, int row, const string &val) { [FromVoid<IOSTable*>(impl) setValue:section row:row val:val]; }
void SystemTableView::SetHidden(int section, int row, bool val) { [FromVoid<IOSTable*>(impl) setHidden:section row:row val:val]; }
void SystemTableView::SetTitle(const string &title) { FromVoid<IOSTable*>(impl).title = LFL::MakeNSString(title); }
PickerItem *SystemTableView::GetPicker(int section, int row) { return [FromVoid<IOSTable*>(impl) getPicker:section row:row]; }
StringPairVec SystemTableView::GetSectionText(int section) { return [FromVoid<IOSTable*>(impl) dumpDataForSection:section]; }
void SystemTableView::SetEditableSection(int section, int start_row, LFL::IntIntCB cb) {
  FromVoid<IOSTable*>(impl).delete_row_cb = move(cb);
  FromVoid<IOSTable*>(impl).editable_section = section;
  FromVoid<IOSTable*>(impl).editable_start_row = start_row;
}

void SystemTableView::SelectRow(int section, int row) {
  FromVoid<IOSTable*>(impl).selected_section = section;
  FromVoid<IOSTable*>(impl).selected_row = row;
} 

void SystemTableView::BeginUpdates() { [FromVoid<IOSTable*>(impl).tableView beginUpdates]; }
void SystemTableView::EndUpdates() { [FromVoid<IOSTable*>(impl).tableView endUpdates]; }
void SystemTableView::SetDropdown(int section, int row, int val) { [FromVoid<IOSTable*>(impl) setDropdown:section row:row index:val]; }
void SystemTableView::SetSectionValues(int section, const StringVec &item) { [FromVoid<IOSTable*>(impl) setSectionValues:section items:item]; }
void SystemTableView::ReplaceSection(int section, const string &h, int image, int flag, TableItemVec item, Callback add_button)
{ [FromVoid<IOSTable*>(impl) replaceSection:section items:move(item) header:h image:image flag:flag addbutton:move(add_button)]; }

SystemTextView::~SystemTextView() { if (auto view = FromVoid<IOSTextView*>(impl)) [view release]; }
SystemTextView::SystemTextView(const string &title, File *f) : SystemTextView(title, f ? f->Contents() : "") {}
SystemTextView::SystemTextView(const string &title, const string &text) :
  impl([[IOSTextView alloc] initWithTitle:MakeNSString(title) andText:[[[NSString alloc]
       initWithBytes:text.data() length:text.size() encoding:NSASCIIStringEncoding] autorelease]]) {}

SystemNavigationView::~SystemNavigationView() { if (auto nav = FromVoid<IOSNavigation*>(impl)) [nav release]; }
SystemNavigationView::SystemNavigationView() : impl([[IOSNavigation alloc] initWithNavigationBarClass:nil toolbarClass:nil]) {
  auto nav = FromVoid<IOSNavigation*>(impl);
  [nav setToolbarHidden:YES animated:NO];
}

void SystemNavigationView::Show(bool show_or_hide) {
  auto nav = FromVoid<IOSNavigation*>(impl);
  LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
  if ((shown = show_or_hide)) {
    if (root->show_cb) root->show_cb();
    INFO("LFViewController.presentViewController IOSNavigation frame=", LFL::GetCGRect(uiapp.controller.view.frame).DebugString());
    uiapp.top_controller = nav;
    [uiapp.controller presentViewController:nav animated:YES completion:nil];
  } else {
    INFO("LFViewController.dismissViewController ", GetNSString(NSStringFromClass([uiapp.controller class])), " frame=", LFL::GetCGRect(uiapp.controller.view.frame).DebugString());
    uiapp.top_controller = uiapp.root_controller;
    [uiapp.controller dismissViewControllerAnimated:YES completion:nil];
  }
}

SystemTableView *SystemNavigationView::Back() {
  for (UIViewController *c in [FromVoid<IOSNavigation*>(impl).viewControllers reverseObjectEnumerator]) {
    if ([c isKindOfClass:[IOSTable class]])
      if (auto lself = static_cast<IOSTable*>(c).lfl_self) return lself;
  } 
  return nullptr;
}

void SystemNavigationView::PushTableView(SystemTableView *t) {
  if (!root) root = t;
  if (t->show_cb) t->show_cb();
  [FromVoid<IOSNavigation*>(impl) pushViewController: FromVoid<IOSTable*>(t->impl) animated: YES];
}

void SystemNavigationView::PushTextView(SystemTextView *t) {
  if (t->show_cb) t->show_cb();
  [FromVoid<IOSNavigation*>(impl) pushViewController: FromVoid<IOSTextView*>(t->impl) animated: YES];
}

void SystemNavigationView::PopToRoot() {
  if (root) [FromVoid<IOSNavigation*>(impl) popToRootViewControllerAnimated: YES];
}

void SystemNavigationView::PopAll() {
  if (root && !(root=0)) {
    [FromVoid<IOSNavigation*>(impl) popToRootViewControllerAnimated: NO];
    [FromVoid<IOSNavigation*>(impl) setViewControllers:@[] animated:NO];
  }
}

void SystemNavigationView::PopView(int n) {
  for (int i = 0; i != n; ++i)
    [FromVoid<IOSNavigation*>(impl) popViewControllerAnimated: (i == n - 1)];
}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &cb) {
  static IOSFontPicker *font_chooser = [[IOSFontPicker alloc] init];
  [font_chooser selectFont:cur_font.name size:cur_font.size cb:cb];
  [[LFUIApplication sharedAppDelegate].glk_view addSubview: font_chooser];
}

int Application::LoadSystemImage(const string &n) {
  UIImage *image = [UIImage imageNamed:MakeNSString(n)];
  if (!image) return 0;
  [image retain];
  app_images.push_back(image);
  return app_images.size();
}

}; // namespace LFL