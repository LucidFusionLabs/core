/*
 * $Id$
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
#include "core/app/flow.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/ios_common.h"

static LFL::FreeListVector<UIImage*> app_images;
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

@interface IOSTextField : UITextField<UITextFieldDelegate>
  @property (nonatomic) bool modified;
  @property (nonatomic, assign) LFL::StringCB changed_cb;
  - (void)textFieldDidChange:(IOSTextField*)sender;
@end
@implementation IOSTextField
  - (void)textFieldDidChange:(IOSTextField*)sender {
    if (_changed_cb) _changed_cb(LFL::GetNSString(self.text));
  }
  - (BOOL)textFieldShouldReturn:(UITextField *)textField {
    [textField resignFirstResponder];
    return YES;
  }
@end

@interface IOSSwitch : UISwitch
  @property (nonatomic, assign) LFL::StringCB changed_cb;
@end
@implementation IOSSwitch
@end

@interface IOSSlider : UISlider
  @property (nonatomic, retain) UILabel *label;
  @property (nonatomic, assign) LFL::StringCB changed_cb;
@end
@implementation IOSSlider
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
    _toolbar  = [self createUIToolbar: [self getToolbarFrame] resignAfterClick:false];
    _toolbar2 = [self createUIToolbar: [self getToolbarFrame] resignAfterClick:true];
    return self;
  }

  - (void)setTheme:(const std::string&)n {
    if      (n == "Red")   { _toolbar.barTintColor = _toolbar2.barTintColor = [UIColor colorWithRed:255/255.0 green: 59/255.0 blue: 48/255.0 alpha:.3]; }
    else if (n == "Green") { _toolbar.barTintColor = _toolbar2.barTintColor = [UIColor colorWithRed: 76/255.0 green:217/255.0 blue:100/255.0 alpha:.8]; }
    else if (n == "Dark")  { [_toolbar setBarStyle: UIBarStyleBlackTranslucent]; [_toolbar2 setBarStyle: UIBarStyleBlackTranslucent]; }
    else                   { [_toolbar setBarStyle: UIBarStyleDefault];          [_toolbar2 setBarStyle: UIBarStyleDefault]; }
  }

  - (NSMutableArray*)createUIToolbarItems: (BOOL)resignAfterClick {
    NSMutableArray *items = [[NSMutableArray alloc] init];
    UIBarButtonItem *spacer = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemFlexibleSpace target:nil action:nil];
    UIFont *font = [UIFont boldSystemFontOfSize: 7];
    NSDictionary<NSAttributedStringKey, id> *fontattr = @{NSFontAttributeName: font};

    for (int i=0, l=data.size(); i<l; i++) {
      if (i || l == 1) [items addObject: spacer];
      NSString *K = [NSString stringWithUTF8String: data[i].shortcut.c_str()];
      UIBarButtonItem *item;
      if (int icon = data[i].image) {
          CHECK_LE(icon, app_images.size());
          item = [[UIBarButtonItem alloc] initWithImage: app_images[icon - 1]
            style:UIBarButtonItemStylePlain target:self action:(resignAfterClick ? @selector(onClickResign:) : @selector(onClick:))];
      } else {
        item = [[UIBarButtonItem alloc] initWithTitle:(([K length] && LFL::isascii([K characterAtIndex:0])) ? [NSString stringWithFormat:@"%@", K] : [NSString stringWithFormat:@"%@\U0000FE0E", K])
          style:UIBarButtonItemStylePlain target:self action:(resignAfterClick ? @selector(onClickResign:) : @selector(onClick:))];
      }
      if (data.size() > 5) {
        [item setTitleTextAttributes:fontattr forState:UIControlStateNormal];
        [item setTitleTextAttributes:fontattr forState:UIControlStateSelected];
        [item setTitleTextAttributes:fontattr forState:UIControlStateDisabled];
      }
      [item setTag:i];
      [items addObject:item];
      [item release];
    }
    if (data.size() == 1) [items addObject: spacer];
    [spacer release];
    return items;
  }

  - (UIToolbar*)createUIToolbar:(CGRect)rect resignAfterClick:(BOOL)resignAfterClick {
    NSMutableArray *items = [self createUIToolbarItems: resignAfterClick];
    UIToolbar *tb = [[UIToolbar alloc] initWithFrame: rect];
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

  - (void)onClickResign:(id)sender {
    [self onClick: sender];
    [[LFUIApplication sharedAppDelegate].controller resignFirstResponder];
  }

  - (void)show: (bool)show_or_hide {
    auto uiapp = [LFUIApplication sharedAppDelegate];
    uiapp.controller.input_accessory_toolbar = show_or_hide ? _toolbar : nil;
    uiapp.glk_view.inputAccessoryView = show_or_hide ? _toolbar2 : nil;
    if ([uiapp.glk_view isFirstResponder]) [uiapp.glk_view reloadInputViews];
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

  - (void)setTheme:(const std::string&)n {
    if (n == "Dark") self.navigationBar.barStyle = UIBarStyleBlack;
    else             self.navigationBar.barStyle = UIBarStyleDefault;
  }
@end

@implementation IOSTable
  {
    std::vector<LFL::TableSection<LFL::TableItem>> data;
    bool dark_theme, clear_theme;
    LFL::Localization *localization;
  }

  - (id)initWithStyle:(UITableViewStyle)style andLocalization:(LFL::Localization*)loc {
    self = [super initWithNibName:nil bundle:nil];
    _tableView = [[UITableView alloc] initWithFrame: [LFUIApplication sharedAppDelegate].controller.view.frame style:style];
    _tableView.autoresizingMask = UIViewAutoresizingFlexibleHeight | UIViewAutoresizingFlexibleWidth;
    _tableView.delegate = self;
    _tableView.dataSource = self;
    _orig_bg_color = _tableView.backgroundColor;
    _orig_separator_color = _tableView.separatorColor;
    localization = loc;
    [self.view addSubview: _tableView];
    return self;
  }

  - (void)setTheme:(const std::string&)n {
    if ((dark_theme = (n == "Dark"))) {
      _tableView.backgroundColor = [UIColor colorWithWhite:0.249 alpha:1.000];
      _tableView.separatorColor = [UIColor colorWithWhite:0.664 alpha:1.000];
    } else if ((clear_theme = (n == "Clear"))) {
      _tableView.backgroundColor = [UIColor colorWithWhite:0 alpha:0];
      dark_theme = true;
    } else {
      _tableView.backgroundColor = _orig_bg_color;
      _tableView.separatorColor = _orig_separator_color;
    }
    if (self.isViewLoaded && _tableView.window) [_tableView reloadData];
    else _needs_reload = true;
  }

  - (void)load:(LFL::TableViewInterface*)lself withTitle:(const std::string&)title withStyle:(const std::string&)sty items:(std::vector<LFL::TableSection<LFL::TableItem>>)item {
    _lfl_self = lself;
    _style = sty;
    _needs_reload = true;
    data = move(item);
    self.title = LFL::MakeNSString(title);
    if (_style != "indent") self.tableView.separatorInset = UIEdgeInsetsZero;
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
    auto &hi = data[section];
    hi.header = move(h);
    hi.flag = f;
    hi.item = move(item);
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

  - (void)setHidden:(int)section row:(int)r val:(int)v {
    [self checkExists:section row:r];
    auto &hi = data[section];
    if (v < 0) {
      hi.item[r].hidden = !hi.item[r].hidden;
      NSIndexPath *p = [NSIndexPath indexPathForRow:r inSection:section];
      [self.tableView reloadRowsAtIndexPaths:@[p] withRowAnimation:UITableViewRowAnimationNone];
    } else hi.item[r].hidden = v;

    if (hi.flag & LFL::TableSectionInterface::Flag::DeleteRowsWhenAllHidden) {
      for (auto &i : hi.item) if (!i.hidden) return;
      hi.item.clear();
      [self.tableView reloadSections:[NSIndexSet indexSetWithIndex: section]
        withRowAnimation:UITableViewRowAnimationNone];
      if (hi.flag & LFL::TableSectionInterface::Flag::ClearLeftNavWhenEmpty)
        self.navigationItem.leftBarButtonItem = nil;
    }
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

  - (void)setColor:(int)section row:(int)r val:(const LFL::Color&)v {
    [self checkExists:section row:r];
    auto &ci = data[section].item[r];
    ci.font.fg = v;
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

  - (void)setSectionColors:(int)section items:(const LFL::vector<LFL::Color>&)item {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_EQ(item.size(), data[section].item.size());
    for (int i=0, l=data[section].item.size(); i != l; ++i) [self setColor:section row:i val:item[i]];
  }

  - (void)setSectionEditable:(int)section startRow:(int)sr skipLastRows:(int)sll withCb:(LFL::IntIntCB)cb {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    data[section].SetEditable(sr, sll, move(cb));
  }

  - (void)applyChangeList:(const LFL::TableSectionInterface::ChangeList&)changes {
    UITableView *tv = _tableView;
    LFL::TableSection<LFL::TableItem>::ApplyChangeList(changes, &data, [=](const LFL::TableSectionInterface::Change &d){
      NSIndexPath *p = [NSIndexPath indexPathForRow:d.row inSection:d.section];
      [tv reloadRowsAtIndexPaths:@[p] withRowAnimation:UITableViewRowAnimationNone];
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
    CGFloat ret = 0;
    if (path.section >= data.size() || path.row >= data[path.section].item.size()) ret = tableView.rowHeight;
    else {
      auto &hi = data[path.section];
      const auto &ci = hi.item[path.row];
      if (ci.hidden) ret = 0;
      else if ((ci.flags & iOSTableItem::GUILoaded) &&
               (ci.type == LFL::TableItem::Picker || ci.type == LFL::TableItem::FontPicker)) ret = ci.height;
      else if (hi.flag & LFL::TableSectionInterface::Flag::DoubleRowHeight)
        ret = tableView.rowHeight > 0 ? tableView.rowHeight * 2 : tableView.rowHeight;
      else if (ci.flags & LFL::TableItem::Flag::SubText) ret = UITableViewAutomaticDimension;
      else ret = tableView.rowHeight;
    }
    return ret;
  }
  
  - (void)clearNavigationButton:(int)align {
    if (align == LFL::HAlign::Right) self.navigationItem.rightBarButtonItem = nil;
    else                             self.navigationItem.leftBarButtonItem  = nil;
  }

  - (void)loadNavigationButton:(const LFL::TableItem&)item withAlign:(int)align {
    if (item.key == localization->GetLocalizedString("edit")) {
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
    UITableViewCell *cell = [self.tableView dequeueReusableCellWithIdentifier: @""];
    CHECK_EQ(nullptr, cell);
    if (cell == nil) {
      int row = path.row, section = path.section;
      CHECK_LT(section, data.size());
      CHECK_LT(row, data[section].item.size());
      auto &hi = data[section];
      auto &ci = hi.item[row];
      ci.flags |= iOSTableItem::GUILoaded;
      bool subtext = ci.flags & LFL::TableItem::Flag::SubText;
      bool is_selected_row = section == _selected_section && row == _selected_row;
      UIColor *blue  = [UIColor colorWithRed: 0.0/255 green:122.0/255 blue:255.0/255 alpha:1];

      cell = [[UITableViewCell alloc] initWithStyle: (subtext ? UITableViewCellStyleSubtitle : UITableViewCellStyleDefault)
        reuseIdentifier: nil];
      cell.selectionStyle = UITableViewCellSelectionStyleNone;
      if      (!LFL::Color::IsTransparent(ci.font.bg)) { LFL::Color c(ci.font.bg);                    [cell setBackgroundColor:[UIColor colorWithRed:c.r() green:c.g() blue:c.b() alpha:c.a()]]; }
      else if (is_selected_row && (hi.flag & LFL::TableSectionInterface::Flag::HighlightSelectedRow)) [cell setBackgroundColor:[UIColor lightGrayColor]];
      else if (clear_theme)                                                                           [cell setBackgroundColor:[UIColor colorWithWhite:0 alpha:0]];
      else if (dark_theme)                                                                            [cell setBackgroundColor:[UIColor colorWithWhite:0.349 alpha:1.000]];

      if (dark_theme) {
        cell.textLabel.textColor = [UIColor whiteColor];
        if (subtext) cell.detailTextLabel.textColor = [UIColor whiteColor];
      }

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
        textfield.hidden = ci.hidden;
        textfield.changed_cb = ci.right_cb;
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
        [textfield setDelegate: textfield];
        [textfield addTarget:self action:@selector(textFieldDidChange:) 
          forControlEvents:UIControlEventEditingChanged];

        if (ci.HasPlaceholderValue()) [textfield setPlaceholder: LFL::MakeNSString(ci.GetPlaceholderValue())];
        else if (ci.val.size())       [textfield setText:        LFL::MakeNSString(ci.val)];
        
        if (dark_theme) textfield.textColor = [UIColor whiteColor];
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
        IOSPicker *picker = [[IOSPicker alloc] initWithColumns: move(item) andTheme:(dark_theme ? "Dark" : "Light")];
        picker.hidden = ci.hidden;
        picker.autoresizingMask = UIViewAutoresizingFlexibleWidth;
        if (row > 0) {
          LFL::StringVec v, joined(picker_cols, "");
          LFL::Split(data[section].item[row-1].val, LFL::isspace, &v);
          LFL::Join(&joined, v, " ", false); 
          [picker selectRows: joined];
        }
        [cell.contentView addSubview:picker];
        CGRect r = picker.frame;
        ci.height = r.size.height;
        r.size.width = cell.frame.size.width;
        picker.frame = r;

      } else if (ci.type == LFL::TableItem::Button) {
        if (int icon = ci.left_icon) {
          CHECK_LE(icon, app_images.size());
          UIImage *image = app_images[icon - 1]; 
          IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
          [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
          button.cb = [=](){ auto &item = data[section].item[row]; if (item.cb) item.cb(); };
          button.frame = cell.frame;
          button.hidden = ci.hidden;
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
          if (!LFL::Color::IsTransparent(ci.font.bg)) [cell.textLabel setFont:[UIFont boldSystemFontOfSize:[UIFont labelFontSize]]];
          cell.textLabel.text = LFL::MakeNSString(ci.key);
          cell.textLabel.textAlignment = NSTextAlignmentCenter;
          [cell.textLabel sizeToFit];
        }

      } else if (ci.type == LFL::TableItem::Toggle) {
        IOSSwitch *onoff = [[IOSSwitch alloc] init];
        onoff.changed_cb = ci.right_cb;
        onoff.hidden = ci.hidden;
        onoff.on = ci.val == "1";
        [onoff addTarget: self action: @selector(switchFlipped:) forControlEvents: UIControlEventValueChanged];
        cell.textLabel.text = LFL::MakeNSString(ci.key);
        cell.accessoryView = onoff;
        [onoff release];

      } else if (ci.type == LFL::TableItem::Slider) {
        cell.textLabel.text = LFL::MakeNSString(ci.key);
        [cell.textLabel sizeToFit];

        UILabel *label = [[UILabel alloc] initWithFrame: CGRectMake(-40, 0, 40, 44)];
        label.hidden = ci.hidden;
        label.text = LFL::MakeNSString(ci.val);
        label.textColor = cell.textLabel.textColor;
        label.hidden = YES;

        IOSSlider *slider = [[IOSSlider alloc] initWithFrame:
          [self getCellFrame: cell.textLabel.frame.size.width + label.frame.size.width]];
        slider.minimumValue = ci.minval;
        slider.maximumValue = ci.maxval;
        slider.hidden = ci.hidden;
        slider.continuous = YES;
        slider.value = LFL::atof(ci.val);
        slider.label = label;
        [slider addTarget:self action:@selector(sliderMoved:) forControlEvents:UIControlEventValueChanged];
        [slider addTarget:self action:@selector(sliderDone:)  forControlEvents:(UIControlEventTouchUpInside | UIControlEventTouchUpOutside)];
        [slider addSubview: label];
        [label release];

        cell.accessoryView = slider;
        [slider release];

      } else if (ci.type == LFL::TableItem::Label) {
        cell.textLabel.text = LFL::MakeNSString(ci.key);
        if (subtext) {
          cell.detailTextLabel.text = LFL::MakeNSString(ci.val);
          cell.detailTextLabel.lineBreakMode = NSLineBreakByWordWrapping;
          cell.detailTextLabel.numberOfLines = 0;
        } else {
          [cell.textLabel sizeToFit];
          UILabel *label = [[UILabel alloc] initWithFrame: [self getCellFrame: cell.textLabel.frame.size.width]];
          label.hidden = ci.hidden;
          label.text = LFL::MakeNSString(ci.val);
          label.adjustsFontSizeToFitWidth = TRUE;
          label.textAlignment = NSTextAlignmentRight;
          if (dark_theme) label.textColor = [UIColor whiteColor];
          cell.accessoryView = label;
          [label release];
        }

      } else {
        cell.textLabel.text = LFL::MakeNSString(ci.key);
        if (subtext) {
          cell.detailTextLabel.text = LFL::MakeNSString(ci.val);
          if (ci.type == LFL::TableItem::Command) {
            cell.detailTextLabel.adjustsFontSizeToFitWidth = TRUE;
            cell.detailTextLabel.textColor = [UIColor lightGrayColor];
          } else {
            cell.detailTextLabel.lineBreakMode = NSLineBreakByWordWrapping;
            cell.detailTextLabel.numberOfLines = 0;
          }
          if (!LFL::Color::IsTransparent(ci.font.fg) && (ci.flags & LFL::TableItem::Flag::ColoredSubText)) {
            LFL::Color c(ci.font.fg);
            cell.detailTextLabel.textColor = [UIColor colorWithRed:c.r() green:c.g() blue:c.b() alpha:c.a()];
          }
        }
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
        button.hidden = ci.hidden;
        button.frame = CGRectMake(0, 0, 40, 40);
        [button setImage:image forState:UIControlStateNormal];
        button.cb = LFL::bind(ci.right_cb, "");
        [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
        if (cell.accessoryView) [cell.accessoryView addSubview: button];
        else cell.accessoryView = button;
      } else if (ci.right_text.size()) {
        UIView *addView = 0;
        UILabel *label = [[UILabel alloc] init];
        if (!LFL::Color::IsTransparent(ci.font.fg) && (ci.flags & LFL::TableItem::Flag::ColoredRightText)) {
          LFL::Color c(ci.font.fg);
          label.textColor = [UIColor colorWithRed:c.r() green:c.g() blue:c.b() alpha:c.a()];
        } else label.textColor = self.view.tintColor;
        label.text = LFL::MakeNSString(ci.right_text);
        label.hidden = ci.hidden;
        [label sizeToFit];
        if (ci.right_cb) {
          IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
          button.hidden = ci.hidden;
          button.frame = label.frame;
          button.cb = LFL::bind(ci.right_cb, "");
          [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
          [button addSubview: label];
          [label release];
          addView = button;
        } else addView = label;
        if (cell.accessoryView) [cell.accessoryView addSubview: addView];
        else cell.accessoryView = addView;
        [addView release];
      }
    }
    return cell;
  }

  - (void)tableView:(UITableView *)tableView commitEditingStyle:(UITableViewCellEditingStyle)editingStyle forRowAtIndexPath:(NSIndexPath *)path {
    [self checkExists:path.section row:path.row];
    if (editingStyle == UITableViewCellEditingStyleDelete) {
      auto &hi = data[path.section];
      if (hi.delete_row_cb) hi.delete_row_cb(path.row, hi.item[path.row].tag);
      hi.item.erase(hi.item.begin() + path.row);
      [tableView beginUpdates];
      [tableView deleteRowsAtIndexPaths:@[path] withRowAnimation:UITableViewRowAnimationNone];
      if (!hi.item.size()) {
        [self.tableView setEditing:NO animated:NO];
        [self.tableView reloadSections:[NSIndexSet indexSetWithIndex: path.section]
          withRowAnimation:UITableViewRowAnimationNone];
      }
      [tableView endUpdates];
    }
  }

  - (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    auto &hi = data[section];
    if (!hi.item.size() && !hi.header.left_icon) return nil;
    return LFL::MakeNSString(hi.header.key);
  }

  - (CGFloat)tableView:(UITableView *)tableView heightForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    auto &hi = data[section];
    if (!hi.item.size() && !hi.header.left_icon) return 0.001;
    if (int h = hi.header_height) return h;
    if (int image = hi.header.left_icon) {
      UIImageView *image_view = [[UIImageView alloc] initWithImage: app_images[image - 1]];
      hi.header_height = 44 + image_view.frame.size.height;
      [image_view release];
      return hi.header_height;
    }
    return 35; // UITableViewAutomaticDimension;
  }

  - (CGFloat)tableView:(UITableView *)tableView heightForFooterInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    auto &hi = data[section];
    if (!hi.item.size() && !hi.header.left_icon) return 0.001;
    return 5; // UITableViewAutomaticDimension;
  }

  - (UIView *)tableView:(UITableView *)tableView viewForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    auto &hi = data[section];
    if (!hi.item.size() && !hi.header.left_icon) return nil;
    UIView *headerView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, tableView.frame.size.width, 44)];
    headerView.autoresizingMask = UIViewAutoresizingFlexibleWidth;
    if (hi.header.hidden) return headerView;

    if (int image = hi.header.left_icon) {
      UIImageView *image_view = [[UIImageView alloc] initWithImage: app_images[image - 1]];
      hi.header_height = 44 + image_view.frame.size.height;
      image_view.contentMode = UIViewContentModeCenter;
      image_view.center = CGPointMake(headerView.frame.size.width/2, image_view.frame.size.height/2);
      image_view.autoresizingMask = UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleLeftMargin;
      headerView.frame = CGRectMake(0, 0, tableView.frame.size.width, hi.header_height);
      [headerView addSubview: image_view];
      [image_view release];
    }

    int key_size = hi.header.key.size();
    if (key_size) {
      UILabel *label = [[UILabel alloc] initWithFrame:
        CGRectMake(50, headerView.frame.size.height-1-21-11, headerView.frame.size.width-100,
                   (hi.header.flags & LFL::TableItem::Flag::SubText) ? 44 : 21)];
      label.text = LFL::MakeNSString(hi.header.key);
      label.textAlignment = NSTextAlignmentCenter;
      label.autoresizingMask = UIViewAutoresizingFlexibleRightMargin | UIViewAutoresizingFlexibleLeftMargin;
      label.lineBreakMode = NSLineBreakByWordWrapping;
      label.numberOfLines = 0;
      if (dark_theme) label.textColor = [UIColor lightGrayColor];
      [headerView addSubview:label];
      [label release];
    }

    bool edit_button = hi.flag & LFL::TableSectionInterface::Flag::EditButton;
    if (edit_button || hi.header.right_text.size()) {
      IOSButton *button = [IOSButton buttonWithType:UIButtonTypeSystem];
      if (edit_button) {
        [button setTitle:LFL::MakeNSString(localization->GetLocalizedString("edit")) forState:UIControlStateNormal];
        [button addTarget:self action:@selector(toggleEditMode:) forControlEvents:UIControlEventTouchUpInside];
      } else {
        button.showsTouchWhenHighlighted = TRUE;
        button.cb = [=](){ auto &item = data[section].header; if (item.right_cb) item.right_cb(""); };
        [button setTitle:LFL::MakeNSString(hi.header.right_text) forState:UIControlStateNormal];
        [button addTarget:button action:@selector(buttonClicked:) forControlEvents:UIControlEventTouchUpInside];
      }
      [button sizeToFit];
      [button setFrame:CGRectMake(tableView.frame.size.width - button.frame.size.width - 11,
                                   headerView.frame.size.height-1-21-11, button.frame.size.width, 21)];
      button.autoresizingMask = UIViewAutoresizingFlexibleLeftMargin;
      [headerView addSubview:button];
    }
    return headerView;
  }

  - (void)toggleEditMode:(UIButton*)button {
    [self.tableView setEditing:!self.tableView.editing animated:YES];
    if (self.tableView.editing) [button setTitle:LFL::MakeNSString(localization->GetLocalizedString("done")) forState:UIControlStateNormal];
    else                        [button setTitle:LFL::MakeNSString(localization->GetLocalizedString("edit")) forState:UIControlStateNormal];
    [button sizeToFit];
    [button setFrame:CGRectMake(self.tableView.frame.size.width - button.frame.size.width - 11, 11, button.frame.size.width, 21)];
  }

  - (BOOL)tableView:(UITableView *)tableView canEditRowAtIndexPath:(NSIndexPath *)path {
    if (path.section >= data.size()) return NO;
    auto &hi = data[path.section];
    return hi.editable_startrow >= 0 && path.item >= hi.editable_startrow &&
      path.item < (hi.item.size() - hi.editable_skiplastrows);
  }

  - (UITableViewCellEditingStyle)tableView:(UITableView *)tableView editingStyleForRowAtIndexPath:(NSIndexPath *)path {
    if (![self tableView:tableView canEditRowAtIndexPath: path]) return UITableViewCellEditingStyleNone;
    return (data[path.section].flag & LFL::TableSectionInterface::Flag::EditableIfHasTag && !data[path.section].item[path.row].tag) ?
      UITableViewCellEditingStyleNone : UITableViewCellEditingStyleDelete;
  }

  - (BOOL)tableView:(UITableView *)tableView canMoveRowAtIndexPath:(NSIndexPath *)indexPath {
    if (indexPath.section >= data.size()) return NO;
    return data[indexPath.section].flag & LFL::TableSectionInterface::Flag::MovableRows;
  }

  - (void)tableView:(UITableView *)tableView moveRowAtIndexPath:(NSIndexPath *)sourceIndexPath toIndexPath:(NSIndexPath *)destinationIndexPath {
    CHECK_LT(sourceIndexPath.section, data.size());
    auto &ss = data[sourceIndexPath.section];
    CHECK_LT(sourceIndexPath.row, ss.item.size());
    auto si = ss.item.begin() + sourceIndexPath.row;
    auto v = move(*si);
    ss.item.erase(si);

    CHECK_LT(destinationIndexPath.section, data.size());
    auto &ts = data[destinationIndexPath.section];
    CHECK_LE(destinationIndexPath.row, ts.item.size());
    ts.item.insert(ts.item.begin() + destinationIndexPath.row, move(v));
    _lfl_self->changed = true;
  }

  - (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)path {
    [self checkExists:path.section row:path.row];
    [self.tableView deselectRowAtIndexPath:path animated:NO];
    _selected_row = path.row;
    _selected_section = path.section;
    auto &ci = data[path.section].item[path.row];
    if (ci.cb) ci.cb();
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

  - (IBAction) switchFlipped: (IOSSwitch*) onoff {
    if (onoff.changed_cb) onoff.changed_cb(onoff.on ? "1" : "");
    _lfl_self->changed = true;
  }

  - (void) sliderMoved: (IOSSlider*)slider {
    int progressAsInt = (int)roundf(slider.value);
    slider.label.text = LFL::MakeNSString(LFL::StrCat(progressAsInt));
    slider.label.hidden = NO;
    _lfl_self->changed = true;
  }

  - (void) sliderDone: (IOSSlider*)slider {
    slider.label.hidden = YES;
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
    } else if (ci.type == LFL::TableItem::Slider) {
      IOSSlider *slider = LFL::objc_dynamic_cast<IOSSlider>(cell.accessoryView);
      int progressAsInt = (int)roundf(slider.value);
      val = LFL::StrCat(progressAsInt);
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
const int Texture::updatesystemimage_pf = Pixel::RGB24;

struct iOSToolbarView : public ToolbarViewInterface {
  string theme;
  IOSToolbar *toolbar;
  ~iOSToolbarView() { [toolbar release]; }
  iOSToolbarView(const string &theme, MenuItemVec items) : toolbar([[IOSToolbar alloc] init: move(items)]) { [toolbar setTheme: theme]; }
  void Show(bool show_or_hide) { [toolbar show:show_or_hide]; }
  void ToggleButton(const string &n) { [toolbar toggleButtonNamed: n]; }
  void SetTheme(const string &x) { theme=x; [toolbar setTheme: theme]; }
  string GetTheme() { return theme; }
};

struct iOSTextView : public TextViewInterface {
  IOSTextView *view;
  ~iOSTextView() { [view release]; }
  iOSTextView(const string &title, File *f) : iOSTextView(title, f ? f->Contents() : "") {}
  iOSTextView(const string &title, const string &text) :
    view([[IOSTextView alloc] initWithTitle:MakeNSString(title) andText:[[[NSString alloc]
         initWithBytes:text.data() length:text.size() encoding:NSASCIIStringEncoding] autorelease]]) {}
  void Show(bool show_or_hide) {}
};

struct iOSNavigationView : public NavigationViewInterface {
  NSMutableArray *stack;
  IOSNavigation *nav=0;
  string theme;
  bool overlay;
  ~iOSNavigationView() { if (nav) [nav release]; if (stack) [stack release]; }
  iOSNavigationView(const string &style, const string &t) : stack([[NSMutableArray alloc] init]),
    theme(t), overlay(style == "overlay") {}

  TableViewInterface *Back() {
    for (UIViewController *c in [nav.viewControllers reverseObjectEnumerator]) {
      if ([c isKindOfClass:[IOSTable class]])
        if (auto lself = static_cast<IOSTable*>(c).lfl_self) return lself;
    } 
    return nullptr;
  }

  void Show(bool show_or_hide) {
    LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
    if ((shown = show_or_hide)) {
      if (!nav) {
        nav = [[IOSNavigation alloc] initWithNavigationBarClass:nil toolbarClass:nil];
        [nav setViewControllers:stack animated:NO];
        [nav setToolbarHidden:YES animated:NO];
        [nav setTheme: theme];
        [stack release];
        stack = 0;
      }
      if (auto back = Back()) if (back->show_cb) back->show_cb();
      INFO("LFViewController.presentViewController IOSNavigation frame=", LFL::GetCGRect(uiapp.controller.view.frame).DebugString());
      uiapp.overlay_top_controller = overlay;
      uiapp.top_controller = nav;
      if (uiapp.controller.ready && uiapp.controller.presentedViewController != nav)
        [uiapp.controller presentViewController:nav animated:YES completion:nil];
    } else if (nav) {
      if (auto back = Back()) if (back->hide_cb) back->hide_cb();
      INFO("LFViewController.dismissViewController ", GetNSString(NSStringFromClass([uiapp.controller.presentedViewController class])), " frame=", LFL::GetCGRect(uiapp.controller.view.frame).DebugString());
      stack = [NSMutableArray arrayWithArray: nav.viewControllers];
      [stack retain];
      [nav release];
      nav = 0;
      uiapp.top_controller = uiapp.root_controller;
      [uiapp.controller dismissViewControllerAnimated:NO completion:nil];
    }
  }

  void PushTableView(TableViewInterface *t) {
    if (t->show_cb) t->show_cb();
    if (nav) {
      if (root) {
        [nav pushViewController: dynamic_cast<iOSTableView*>(t)->table animated: YES];
      } else if ((root = t)) {
        [nav setViewControllers:@[ dynamic_cast<iOSTableView*>(t)->table ] animated: YES];
      }
    } else {
      if (!root) root = t;
      [stack addObject: dynamic_cast<iOSTableView*>(t)->table];
    }
  }

  void PushTextView(TextViewInterface *t) {
    if (nav) {
      if (t->show_cb) t->show_cb();
      [nav pushViewController: dynamic_cast<iOSTextView*>(t)->view animated: YES];
    } else {
      [stack addObject: dynamic_cast<iOSTextView*>(t)->view];
    }
  }

  void PopToRoot() {
    if (nav) {
      [nav popToRootViewControllerAnimated: NO];
    } else if (stack.count > 1) [stack removeObjectsInRange: NSMakeRange(1, stack.count-1)];
  }

  void PopAll() {
    if (nav) {
      [nav setViewControllers:@[] animated: NO];
      root = 0;
    } else [stack removeAllObjects];
  }

  void PopView(int n) {
    if (nav) {
      for (int i = 0; i != n; ++i)
        [nav popViewControllerAnimated: (i == n - 1)];
    } else if (n && stack.count >= n) [stack removeObjectsInRange: NSMakeRange(stack.count-n, stack.count-1)];
  }

  void SetTheme(const string &x) { [nav setTheme: (theme=x)]; }
};

iOSCollectionView::iOSCollectionView(const string &title, const string &style, const string &theme, vector<CollectionItem> items) {}
void iOSCollectionView::SetToolbar(ToolbarViewInterface *t) {}
void iOSCollectionView::Show(bool show_or_hide) {}
View *iOSCollectionView::AppendFlow(Flow*) { return nullptr; }

iOSTableView::~iOSTableView() { [table release]; }
iOSTableView::iOSTableView(Localization *l, const string &title, const string &style, const string &theme, TableItemVec items) :
  table([[IOSTable alloc] initWithStyle: UITableViewStyleGrouped andLocalization: l]) {
    // INFOf("iOSTableView %s IOSTable: %p", title.c_str(), table);
    [table load:this withTitle:title withStyle:style items:TableSection<TableItem>::Convert(move(items))];
    [table setTheme: theme];
  }

void iOSTableView::DelNavigationButton(int align) { return [table clearNavigationButton:align]; }
void iOSTableView::AddNavigationButton(int align, const TableItem &item) { return [table loadNavigationButton:item withAlign:align]; }

void iOSTableView::SetToolbar(ToolbarViewInterface *t) {
  int toolbar_height = 44;
  CGRect frame = table.tableView.frame;
  if (table.toolbar) {
    [table.toolbar removeFromSuperview];
    table.toolbar = nil;
    frame.size.height += toolbar_height;
  }
  if (t) {
    table.toolbar = dynamic_cast<iOSToolbarView*>(t)->toolbar.toolbar;
    table.toolbar.frame = CGRectMake(frame.origin.x, frame.origin.y+frame.size.height-toolbar_height, frame.size.width, toolbar_height);
    [table.view addSubview: table.toolbar];
    frame.size.height -= toolbar_height;
  }
  table.tableView.frame = frame;
}

void iOSTableView::Show(bool show_or_hide) {
  if (show_or_hide && show_cb) show_cb();
  [table show:show_or_hide];
  shown = show_or_hide;
}

View *iOSTableView::AppendFlow(Flow *flow) {
  auto uiapp = [LFUIApplication sharedAppDelegate];
  float s = 1.0 / [uiapp getScale];
  Box b = *flow->container;
  b.y = uiapp.screen_h - (uiapp.gl_y + b.y + b.h);
  b.scale(s, s);

  [table.tableView setFrame: MakeCGRect(b)];
  if (!shown) Show(true);
  return nullptr;
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
void iOSTableView::SetHidden(int section, int row, int val) { [table setHidden:section row:row val:val]; }
void iOSTableView::SetColor(int section, int row, const Color &val) { [table setColor:section row:row val:val]; }
void iOSTableView::SetTitle(const string &title) { table.title = LFL::MakeNSString(title); }
void iOSTableView::SetTheme(const string &theme) { [table setTheme:theme]; }
PickerItem *iOSTableView::GetPicker(int section, int row) { return [table getPicker:section row:row]; }
StringPairVec iOSTableView::GetSectionText(int section) { return [table dumpDataForSection:section]; }
void iOSTableView::SetSectionEditable(int section, int start_row, int skip_last_rows, LFL::IntIntCB cb) {
  [table setSectionEditable:section startRow:start_row skipLastRows:skip_last_rows withCb:move(cb)];
}

void iOSTableView::SelectRow(int section, int row) {
  table.selected_section = section;
  table.selected_row = row;
} 

void iOSTableView::BeginUpdates() { [table.tableView beginUpdates]; }
void iOSTableView::EndUpdates() { [table.tableView endUpdates]; }
void iOSTableView::SetSectionValues(int section, const StringVec &item) { [table setSectionValues:section items:item]; }
void iOSTableView::SetSectionColors(int section, const vector<Color> &item) { [table setSectionColors:section items:item]; }
void iOSTableView::ApplyChangeList(const TableSectionInterface::ChangeList &changes) { [table applyChangeList:changes]; }
void iOSTableView::ReplaceRow(int section, int row, TableItem h) { [table replaceRow:section row:row val:move(h)]; }
void iOSTableView::ReplaceSection(int section, TableItem h, int flag, TableItemVec item)
{ [table replaceSection:section items:move(item) header:move(h) flag:flag]; }

int SystemToolkit::LoadImage(const string &n) {
  if (n.empty()) return app_images.Insert(nullptr) + 1;
  UIImage *image = [UIImage imageNamed:MakeNSString(StrCat("drawable-xhdpi/", n, ".png"))];
  if (!image) return 0;
  [image retain];
  return app_images.Insert(move(image)) + 1;
}

void SystemToolkit::UpdateImage(int n, Texture &t) {
  CHECK_RANGE(n-1, 0, app_images.size());
  CGImageRef image = MakeCGImage(t);
  if (app_images[n-1]) [app_images[n-1] release];
  if (!(app_images[n-1] = [[UIImage alloc] initWithCGImage:image])) ERROR("UpdateImage failed");
  CGImageRelease(image);
}

void SystemToolkit::UnloadImage(int n) {
  if (auto image = app_images[n-1]) { [image release]; app_images[n-1] = nullptr; }
  app_images.Erase(n-1);
}

unique_ptr<ToolbarViewInterface> SystemToolkit::CreateToolbar(Window*, const string &theme, MenuItemVec items, int flag) { return make_unique<iOSToolbarView>(theme, move(items)); }
unique_ptr<TableViewInterface> SystemToolkit::CreateTableView(Window *w, const string &title, const string &style, const string &theme, TableItemVec items) { return make_unique<iOSTableView>(w->parent, title, style, theme, move(items)); }
unique_ptr<CollectionViewInterface> SystemToolkit::CreateCollectionView(Window*, const string &title, const string &style, const string &theme, vector<CollectionItem> items) { return make_unique<iOSCollectionView>(title, style, theme, move(items)); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(Window*, const string &title, File *file) { return make_unique<iOSTextView>(title, file); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(Window*, const string &title, const string &text) { return make_unique<iOSTextView>(title, text); }
unique_ptr<NavigationViewInterface> SystemToolkit::CreateNavigationView(Window*, const string &style, const string &theme) { return make_unique<iOSNavigationView>(style, theme); }

}; // namespace LFL
