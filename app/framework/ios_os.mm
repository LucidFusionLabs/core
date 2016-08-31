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
@end
@implementation IOSButton
@end

@interface IOSBarButtonItem : UIBarButtonItem
  @property (nonatomic, assign) LFL::Callback cb;
@end
@implementation IOSBarButtonItem
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
    int toolbar_height;
    std::unordered_map<std::string, void*> toolbar_titles;
    std::unordered_map<void*, LFL::Callback> toolbar_cb;
  }

  - (id)init: (const LFL::MenuItemVec&) kv {
    self = [super init];
    NSMutableArray *items = [[NSMutableArray alloc] init];
    UIBarButtonItem *spacer = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemFlexibleSpace target:nil action:nil];
    for (int i=0, l=kv.size(); i<l; i++) {
      if (i) [items addObject: spacer];
      NSString *K = [NSString stringWithUTF8String: kv[i].shortcut.c_str()];
      UIBarButtonItem *item =
        [[UIBarButtonItem alloc] initWithTitle:(([K length] && LFL::isascii([K characterAtIndex:0])) ? [NSString stringWithFormat:@"%@", K] : [NSString stringWithFormat:@"%@\U0000FE0E", K])
        style:UIBarButtonItemStylePlain target:self action:@selector(onClick:)];
      [item setTag:(kv[i].name == "toggle")];
      [items addObject:item];
      toolbar_titles[kv[i].shortcut] = item;
      toolbar_cb[item] = kv[i].cb;
      [item release];
    }
    toolbar_height = 44;
    _toolbar = [[UIToolbar alloc] initWithFrame: [self getToolbarFrame]];
    // [_toolbar setBarStyle:UIBarStyleBlackTranslucent];
    [_toolbar setItems:items];
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
    if (![sender isKindOfClass:[UIBarButtonItem class]]) FATALf("unknown sender: %p", sender);
    UIBarButtonItem *item = (UIBarButtonItem*)sender;
    auto it = toolbar_cb.find(item);
    if (it != toolbar_cb.end()) {
      it->second();
      if (item.tag) [self toggleButton:item];
    }
    [[LFUIApplication sharedAppDelegate].controller resignFirstResponder];
  }

  - (void)show: (bool)show_or_hide {
    if (show_or_hide) {
      show_bottom.push_back(self);
      [[LFUIApplication sharedAppDelegate].window addSubview: _toolbar];
    } else {
      LFL::VectorEraseByValue(&show_bottom, self);
      [_toolbar removeFromSuperview];
    }
  }

  + (int)getBottomHeight {
    int ret = 0;
    for (auto t : show_bottom) ret += t->toolbar_height;
    return ret;
  }

  + (void)updateFrame {
    for (auto t : show_bottom) t->_toolbar.frame = [t getToolbarFrame];
  }

  static std::vector<IOSToolbar*> show_bottom, show_top;
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
  @property (copy)              void (^completed)();
@end

@implementation IOSTable
  {
    int section_index;
    std::vector<LFL::CompiledTable> data;
    std::vector<IOSTable*> dropdowns;
  }
  
  - (void)load:(LFL::SystemTableView*)lself withTitle:(const std::string&)title withStyle:(const std::string&)sty items:(const std::vector<LFL::TableItem>&)item {
    _lfl_self = lself;
    _style = sty;
    data.emplace_back();
    for (auto &i : item) {
      if (i.type == "separator") {
        data.emplace_back(i.key);
        section_index++;
      } else {
        data[section_index].item.emplace_back(i);
      }
    }

    self.title = LFL::MakeNSString(title);
    self.tableView.separatorInset = UIEdgeInsetsZero;
    [self.tableView setSeparatorStyle:UITableViewCellSeparatorStyleSingleLine];
    [self.tableView setSeparatorColor:[UIColor blackColor]];
    if (_style == "modal" || _style == "dropdown") {
      _modal_nav = [[UINavigationController alloc] initWithRootViewController: self];
      _modal_nav.modalTransitionStyle = UIModalTransitionStyleCoverVertical;
    }
  }

  - (void)replaceSection:(int)section items:(const std::vector<LFL::TableItem>&)item header:(const std::string&)h flag:(int)f {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    data[section] = LFL::CompiledTable(h, f);
    for (auto &i : item) data[section].item.emplace_back(i);
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
    return data[section].item[r].item.key;
  }

  - (int)getTag:(int)section row:(int)r {
    [self checkExists:section row:r];
    return data[section].item[r].item.tag;
  }

  - (void)setTag:(int)section row:(int)r val:(int)v {
    [self checkExists:section row:r];
    data[section].item[r].item.tag = v;
  }

  - (void)setValue:(int)section row:(int)r val:(const std::string&)v {
    [self checkExists:section row:r];
    auto &ci = data[section].item[r];
    ci.item.val = v;
    if (ci.loaded && ci.type == LFL::CompiledTableItem::Dropdown) {
      std::vector<std::string> vv=LFL::Split(v, ',');
      CHECK_RANGE(ci.ref, 0, dropdowns.size());
      auto dropdown_table = dropdowns[ci.ref];
      CHECK_EQ(dropdown_table->data[0].item.size(), vv.size()-1);
      for (int j=1; j < vv.size(); ++j) dropdown_table->data[0].item[j-1].item.val = vv[j];
    }
  }

  - (void)setDropdown:(int)section row:(int)r index:(int)ind {
    [self checkExists:section row:r];
    NSIndexPath *path = [NSIndexPath indexPathForRow:r inSection:section];
    UITableViewCell *cell = [self.tableView cellForRowAtIndexPath: path];
    const std::string *k=0, *type=0, *v=0;
    auto &ci = data[path.section].item[path.row];
    [self loadCellItem:cell withPath:path withItem:&ci outK:&k outT:&type outV:&v];
    if (ci.type == LFL::CompiledTableItem::Dropdown) {
      CHECK_RANGE(ci.ref, 0, dropdowns.size());
      auto dropdown_table = dropdowns[ci.ref];
      CHECK_EQ(0, dropdown_table.selected_section);
      CHECK_LT(ind, dropdown_table->data[0].item.size());
      dropdown_table.selected_row = ind;
      [self.tableView reloadRowsAtIndexPaths:@[path] withRowAnimation:UITableViewRowAnimationNone];
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

  - (CGRect)getCellFrame {
    if (_second_col) return CGRectMake(_second_col, 10, self.tableView.frame.size.width - _second_col - 30, 30);
    else             return CGRectMake(0, 0, 150, 30);
  }

  - (void)loadNavigationButton:(const LFL::TableItem&)item withAlign:(int)align {
    if (item.key == "Edit") {
      self.navigationItem.rightBarButtonItem = [self editButtonItem];
    } else {
      IOSBarButtonItem *button = [[IOSBarButtonItem alloc] init];
      button.cb = item.cb;
      button.title = LFL::MakeNSString(item.key);
      [button setTarget:self];
      [button setAction:@selector(navButtonClicked:)];
      if (align == LFL::HAlign::Right) self.navigationItem.rightBarButtonItem = button;
      else                             self.navigationItem.leftBarButtonItem  = button;
      [button release];
    }
  }

  - (void)loadCellItem:(UITableViewCell*)cell withPath:(NSIndexPath*)path withItem:(LFL::CompiledTableItem*)item
                       outK:(const std::string **)ok outT:(const std::string **)ot outV:(const std::string **)ov {
    bool dropdown;
    if (!item->loaded) {
      item->loaded = true;
      const std::string &kt = item->item.key, tt = item->item.type, vt = item->item.val;
      if ((dropdown = kt.find(',') != std::string::npos)) {
        std::vector<std::string> kv=LFL::Split(kt, ','), tv=LFL::Split(tt, ','), vv=LFL::Split(vt, ',');
        CHECK_GT(kv.size(), 0);
        CHECK_EQ(kv.size(), tv.size());
        CHECK_EQ(kv.size(), vv.size());
        item->type = LFL::CompiledTableItem::Dropdown;
        item->ref = dropdowns.size();
        item->control = vv[0] != "nocontrol";
        std::vector<LFL::TableItem> dropdown_options;
        for (int i=1, l=kv.size(); i != l; ++i) dropdown_options.push_back({kv[i], tv[i], vv[i]});
        auto dropdown_table = [[IOSTable alloc] initWithStyle: UITableViewStyleGrouped];
        [dropdown_table load:nullptr withTitle:kv[0] withStyle:"dropdown" items:dropdown_options];
        dropdown_table.completed = ^{
          [self.tableView beginUpdates];
          [self.tableView reloadRowsAtIndexPaths:@[path] withRowAnimation:UITableViewRowAnimationNone];
          [self.tableView endUpdates];
        };
        dropdowns.push_back(dropdown_table);
      } else item->type = 0;
    } else dropdown = item->type == LFL::CompiledTableItem::Dropdown;

    const LFL::CompiledTableItem *ret;
    if (dropdown) {
      CHECK_RANGE(item->ref, 0, dropdowns.size());
      auto dropdown_table = dropdowns[item->ref];
      CHECK_EQ(0, dropdown_table.selected_section);
      CHECK_RANGE(dropdown_table.selected_row, 0, dropdown_table->data[0].item.size());
      ret = &dropdown_table->data[0].item[dropdown_table.selected_row];
    } else ret = item;

    bool parent_dropdown = _style == "dropdown";
    *ok = &ret->item.key;
    *ot = parent_dropdown ? LFL::Singleton<std::string>::Get() : &ret->item.type;
    *ov = parent_dropdown ? LFL::Singleton<std::string>::Get() : &ret->item.val;
  }

  - (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)path {
    static NSString *cellIdentifier = @"cellIdentifier";
    UITableViewCell *cell = [self.tableView dequeueReusableCellWithIdentifier:cellIdentifier];
    if (cell) { [cell release]; cell = nil; }
    if (cell == nil) {
      CHECK_LT(path.section, data.size());
      CHECK_LT(path.row, data[path.section].item.size());
      cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleDefault reuseIdentifier:cellIdentifier];
      const std::string *k=0, *type=0, *v=0;
      auto &compiled_item = data[path.section].item[path.row];
      [self loadCellItem:cell withPath:path withItem:&compiled_item outK:&k outT:&type outV:&v];
      compiled_item.gui_loaded = true;

      if (compiled_item.type == LFL::CompiledTableItem::Dropdown && compiled_item.control) {
        UIButton *button = [UIButton buttonWithType:UIButtonTypeCustom];
        [button setTitle:@"\U000002C5" forState:UIControlStateNormal];
        button.frame = CGRectMake(_second_col - 10, 10, 10, 30.0);
        [button setTitleColor:[UIColor blueColor] forState:UIControlStateNormal];
        [button addTarget:self action:@selector(dropDownClicked:) forControlEvents:UIControlEventTouchUpInside];
        [button setTag: compiled_item.ref];
        [cell.contentView addSubview: button];
      }

      if (int icon = compiled_item.item.left_icon) {
        CHECK_LE(icon, app_images.size());
        cell.imageView.image = app_images[icon - 1]; 
      }

      bool textinput=0, numinput=0, pwinput=0;
      if ((textinput = *type == "textinput") || (numinput = *type == "numinput") || (pwinput = *type == "pwinput")) {
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
        if (path.section == _selected_section && path.row == _selected_row) [textfield becomeFirstResponder];
        [textfield release];

      } else if (*type == "select") {
        NSArray *itemArray = LFL::MakeNSStringArray(LFL::Split(*v, ','));
        IOSSegmentedControl *segmented_control = [[IOSSegmentedControl alloc] initWithItems:itemArray];
        segmented_control.frame = cell.frame;
        segmented_control.autoresizingMask = UIViewAutoresizingFlexibleWidth;
        segmented_control.segmentedControlStyle = UISegmentedControlStylePlain;
        segmented_control.selectedSegmentIndex = 0; 
        if (compiled_item.item.depends.size()) {
          segmented_control.changed = ^(const std::string &v){
            auto it = compiled_item.item.depends.find(v);
            if (it == compiled_item.item.depends.end()) return;
            [self.tableView beginUpdates];
            for (auto c : it->second) {
              CHECK_LT(c.first, data.size());
              CHECK_LT(c.second, data[c.first].item.size());
              data[c.first].item[c.second].item.val = c.third;
              NSIndexPath *p = [NSIndexPath indexPathForRow:c.second inSection:c.first];
              [self.tableView reloadRowsAtIndexPaths:@[p] withRowAnimation:UITableViewRowAnimationNone];
            }
            [self.tableView endUpdates];
          };
        }
        [segmented_control addTarget:self action:@selector(segmentedControlClicked:)
          forControlEvents: UIControlEventValueChanged];
        [cell.contentView addSubview:segmented_control];
        [segmented_control release]; 

      } else if (*type == "button") {
        cell.textLabel.text = LFL::MakeNSString(*k);
        cell.textLabel.textAlignment = NSTextAlignmentCenter;

      } else if (*type == "toggle") {
        UISwitch *onoff = [[UISwitch alloc] init];
        onoff.on = *v == "1";
        [onoff addTarget: self action: @selector(switchFlipped:) forControlEvents: UIControlEventValueChanged];
        cell.textLabel.text = LFL::MakeNSString(*k);
        cell.accessoryView = onoff;
        [onoff release];

      } else if (*type == "label") {
        UILabel *label = [[UILabel alloc] initWithFrame: [self getCellFrame]];
        label.text = LFL::MakeNSString(*v);
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

      if (int icon = compiled_item.item.right_icon) {
        CHECK_LE(icon, app_images.size());
        UIImage *image = app_images[icon - 1]; 
        IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
        button.frame = CGRectMake(0, 0, 40, 40);
        [button setImage:image forState:UIControlStateNormal];
        button.cb = compiled_item.item.right_icon_cb;
        [button addTarget:self action:@selector(rightIconClicked:) forControlEvents:UIControlEventTouchUpInside];
        if (cell.accessoryView) [cell.accessoryView addSubview: button];
        else cell.accessoryView = button;
      } else if (compiled_item.item.right_text.size()) {
        UILabel *label = [[UILabel alloc] init];
        label.textColor = self.view.tintColor;
        label.text = LFL::MakeNSString(compiled_item.item.right_text);
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
      _delete_row_cb(path.row, data[path.section].item[path.row].item.tag);
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

#if 1
  - (UIView *)tableView:(UITableView *)tableView viewForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    UIView *headerView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, tableView.frame.size.width, 44)];
    UILabel *label = [[UILabel alloc] initWithFrame:CGRectMake(50, 11, tableView.frame.size.width-100, 21)];
    label.text = LFL::MakeNSString(data[section].header);
    label.textAlignment = NSTextAlignmentCenter;
    [headerView addSubview:label];

    if (data[section].flag & LFL::SystemTableView::EditButton) {
      UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
      [button addTarget:self action:@selector(toggleEditMode:) forControlEvents:UIControlEventTouchUpInside];
      [button setTitle:@"Edit" forState:UIControlStateNormal];
      [button sizeToFit];
      [button setFrame:CGRectMake(tableView.frame.size.width - button.frame.size.width - 11, 11, button.frame.size.width, 21)];
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
#endif

  - (BOOL)tableView:(UITableView *)tableView canEditRowAtIndexPath:(NSIndexPath *)path {
    return path.section == _editable_section;
  }

  - (UITableViewCellEditingStyle)tableView:(UITableView *)tableView editingStyleForRowAtIndexPath:(NSIndexPath *)path {
    if (path.section != _editable_section || path.row < _editable_start_row) return UITableViewCellEditingStyleNone;
    [self checkExists:path.section row:path.row];
    return (data[path.section].flag & LFL::SystemTableView::EditableIfHasTag && !data[path.section].item[path.row].item.tag) ?
      UITableViewCellEditingStyleNone : UITableViewCellEditingStyleDelete;
  }

  - (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)path {
    [self checkExists:path.section row:path.row];
    _selected_row = path.row;
    _selected_section = path.section;
    const auto &compiled_item = data[path.section].item[path.row];
    const std::string &t = compiled_item.item.type;
    if (t == "command" || t == "button") compiled_item.item.cb();
    if (_modal_nav) {
      [self show: false];
      if (_completed) _completed();
    }
  }

  - (void)show:(bool)show_or_hide {
    auto uiapp = [LFUIApplication sharedAppDelegate];
    if (show_or_hide) {
      if (_modal_nav) [uiapp.top_controller presentModalViewController:self.modal_nav animated:YES];
      else            [uiapp.view addSubview: self.tableView];
    } else {
      if (_modal_nav) [uiapp.top_controller dismissViewControllerAnimated:YES completion:nil];
      else            [self.tableView removeFromSuperview];
    }
  }

  - (void)viewWillAppear:   (BOOL)animated { if (_toolbar) [_toolbar show: true];  }
  - (void)viewWillDisappear:(BOOL)animated { if (_toolbar) [_toolbar show: false]; }
  
  - (void)textFieldDidChange:(IOSTextField*)sender {
    _lfl_self->changed = true;
    [sender textFieldDidChange: sender];
  }

  - (void)dropDownClicked:(UIButton *)sender {
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

  - (void)rightIconClicked:(IOSButton*)sender {
    if (sender.cb) sender.cb();
  }

  - (IBAction)navButtonClicked:(IOSBarButtonItem*)sender {
    if (sender.cb) sender.cb();
  }

  - (IBAction) switchFlipped: (UISwitch*) onoff {
    _lfl_self->changed = true;
  }

  - (void)willMoveToParentViewController:(UIViewController *)parent {
    if (parent == nil && _lfl_self && _lfl_self->hide_cb) _lfl_self->hide_cb();
  }

  - (LFL::StringPairVec)dumpDataForSection: (int)ind {
    LFL::StringPairVec ret;
    CHECK_LT(ind, data.size());
    for (int i=0, l=data[ind].item.size(); i != l; i++) {
      NSIndexPath *path = [NSIndexPath indexPathForRow: i inSection: ind];
      UITableViewCell *cell = [self.tableView cellForRowAtIndexPath: path];
      const std::string *k=0, *type=0, *v=0;
      auto &compiled_item = data[path.section].item[path.row];
      [self loadCellItem:cell withPath:path withItem:&compiled_item outK:&k outT:&type outV:&v];

      if (compiled_item.type == LFL::CompiledTableItem::Dropdown) {
        CHECK_RANGE(compiled_item.ref, 0, dropdowns.size());
        auto dropdown_table = dropdowns[compiled_item.ref];
        CHECK_EQ(0, dropdown_table.selected_section);
        CHECK_LT(dropdown_table.selected_row, dropdown_table->data[0].item.size());
        ret.emplace_back(LFL::GetNSString(dropdown_table.title),
                         dropdown_table->data[0].item[dropdown_table.selected_row].item.key);
      }

      std::string val;
      if (!compiled_item.gui_loaded) {
          if (v->size() && (*v)[0] != 1) val = (*v)[0] == 2 ? v->substr(1) : *v;
      } else if ((*type == "textinput") || (*type == "numinput") || (*type == "pwinput")) {
        IOSTextField *textfield = _second_col ? [[cell.contentView subviews] lastObject] : cell.accessoryView;
        val = LFL::GetNSString(textfield.text);
        if (val.empty() && !textfield.modified) {
          if (v->size() && (*v)[0] != 1) val = (*v)[0] == 2 ? v->substr(1) : *v;
        }
      } else if (*type == "label") {
        UILabel *label = _second_col ? [[cell.contentView subviews] lastObject] : cell.accessoryView;
        val = LFL::GetNSString(label.text);
      } else if (*type == "select") {
        UISegmentedControl *segmented_control = [[cell.contentView subviews] lastObject];
        val = LFL::GetNSString([segmented_control titleForSegmentAtIndex: segmented_control.selectedSegmentIndex]);
      } else if (*type == "toggle") {
        UISwitch *onoff = (UISwitch*)cell.accessoryView;
        val = onoff.on ? "1" : "";
      }
      ret.emplace_back(*k, val);
    }
    return ret;
  }
@end

@interface IOSNavigation : NSObject<UINavigationControllerDelegate>
  @property (nonatomic, retain) UINavigationController *controller;
@end

@implementation IOSNavigation
  - (id)init {
    self = [super init];
    _controller = [[UINavigationController alloc] initWithNavigationBarClass:nil toolbarClass:nil];
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
    LFL::StringVecCB font_change_cb;
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

  - (void)selectFont: (const std::string &)name size:(int)s cb:(LFL::StringVecCB)v {
    font_change_cb = move(v);
    if (picked_row.size() != columns.size()) picked_row.resize(columns.size());
    for (auto b = columns[0].begin(), e = columns[0].end(), i = b; i != e; ++i)
      if (*i == name) picked_row[0] = i - b;
    picked_row[1] = LFL::Clamp(s-1, 1, 64);
    [self.picker selectRow:picked_row[0] inComponent:0 animated:NO];
    [self.picker selectRow:picked_row[1] inComponent:1 animated:NO];
  }

  - (bool)didSelect {
    font_change_cb(LFL::StringVec{columns[0][picked_row[0]], columns[1][picked_row[1]]});
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
SystemAlertView::~SystemAlertView() { if (auto alert = FromVoid<IOSAlert*>(impl)) [alert release]; }
SystemAlertView::SystemAlertView(AlertItemVec items) : impl([[IOSAlert alloc] init: move(items)]) {}
void SystemAlertView::Show(const string &arg) {
  auto alert = FromVoid<IOSAlert*>(impl);
  if (alert.add_text) [alert.alert textFieldAtIndex:0].text = MakeNSString(arg);
  [alert.alert show];
}

void SystemAlertView::ShowCB(const string &title, const string &arg, StringCB confirm_cb) {
  auto alert = FromVoid<IOSAlert*>(impl);
  alert.confirm_cb = move(confirm_cb);
  alert.alert.title = MakeNSString(title);
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
  [table load:this withTitle:title withStyle:style items:move(items)];
  impl = table;
}

void SystemTableView::AddNavigationButton(const TableItem &item, int align) { return [FromVoid<IOSTable*>(impl) loadNavigationButton:item withAlign:align]; }
void SystemTableView::AddToolbar(SystemToolbarView *t) {
  [FromVoid<IOSTable*>(impl) setToolbar: FromVoid<IOSToolbar*>(t->impl)];
  [FromVoid<IOSTable*>(impl).toolbar.toolbar setNeedsLayout];
}

void SystemTableView::Show(bool show_or_hide) {
  if (show_or_hide && show_cb) show_cb();
  [FromVoid<IOSTable*>(impl) show:show_or_hide];
}

string SystemTableView::GetKey(int section, int row) { return [FromVoid<IOSTable*>(impl) getKey:section row:row]; }
int SystemTableView::GetTag(int section, int row) { return [FromVoid<IOSTable*>(impl) getTag:section row:row]; }
void SystemTableView::SetTag(int section, int row, int val) { [FromVoid<IOSTable*>(impl) setTag:section row:row val:val]; }
void SystemTableView::SetValue(int section, int row, const string &val) { [FromVoid<IOSTable*>(impl) setValue:section row:row val:val]; }
void SystemTableView::SetTitle(const string &title) { FromVoid<IOSTable*>(impl).title = LFL::MakeNSString(title); }
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
void SystemTableView::ReplaceSection(int section, const string &h, int flag, TableItemVec item) { [FromVoid<IOSTable*>(impl) replaceSection:section items:move(item) header:h flag:flag]; }

SystemNavigationView::~SystemNavigationView() { if (auto nav = FromVoid<IOSNavigation*>(impl)) [nav release]; }
SystemNavigationView::SystemNavigationView() : impl([[IOSNavigation alloc] init]) {}
void SystemNavigationView::Show(bool show_or_hide) {
  auto nav = FromVoid<IOSNavigation*>(impl);
  LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
  if ((shown = show_or_hide)) {
    if (root->show_cb) root->show_cb();
    uiapp.top_controller = nav.controller;
    [uiapp.controller presentViewController: nav.controller animated:YES completion:nil];
  } else {
    uiapp.top_controller = uiapp.controller;
    [uiapp.controller dismissViewControllerAnimated:YES completion:nil];
    if ([uiapp isKeyboardFirstResponder]) [uiapp showKeyboard];
    else                                  [uiapp hideKeyboard];
  }
}

SystemTableView *SystemNavigationView::Back() {
  for (UIViewController *c in
       [FromVoid<IOSNavigation*>(impl).controller.viewControllers reverseObjectEnumerator]) {
    if ([c isKindOfClass:[IOSTable class]])
      if (auto lself = static_cast<IOSTable*>(c).lfl_self) return lself;
  } 
  return nullptr;
}

void SystemNavigationView::PushTable(SystemTableView *t) {
  if (!root) root = t;
  if (t->show_cb) t->show_cb();
  [FromVoid<IOSNavigation*>(impl).controller pushViewController: FromVoid<IOSTable*>(t->impl) animated: YES];
}

void SystemNavigationView::PopAll() { [FromVoid<IOSNavigation*>(impl).controller popToRootViewControllerAnimated: YES]; }
void SystemNavigationView::PopTable(int n) {
  for (int i = 0; i != n; ++i)
    [FromVoid<IOSNavigation*>(impl).controller popViewControllerAnimated: (i == n - 1)];
}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &cb) {
  static IOSFontPicker *font_chooser = [[IOSFontPicker alloc] init];
  [font_chooser selectFont:cur_font.name size:cur_font.size cb:cb];
  [[LFUIApplication sharedAppDelegate].view addSubview: font_chooser.picker];
}

void Application::OpenSystemBrowser(const string &url_text) {
  NSString *url_string = [[NSString alloc] initWithUTF8String: url_text.c_str()];
  NSURL *url = [NSURL URLWithString: url_string];
  [[UIApplication sharedApplication] openURL:url];
  [url_string release];
}

bool Application::OpenSystemAppPreferences() {
  bool can_open_settings = &UIApplicationOpenSettingsURLString != NULL;
  if (!can_open_settings) return false;
  NSURL *url = [NSURL URLWithString:UIApplicationOpenSettingsURLString];
  [[UIApplication sharedApplication] openURL:url];
  return true;
}

void Application::SaveKeychain(const string &keyname, const string &val_in) {
  NSString *k = [[NSString stringWithFormat:@"%s://%s", name.c_str(), keyname.c_str()] retain];
  NSMutableString *val = [[NSMutableString stringWithUTF8String: val_in.c_str()] retain];
  UIAlertController *alertController = [UIAlertController alertControllerWithTitle: [NSString stringWithFormat: @"%s Keychain", name.c_str()]
    message:[NSString stringWithFormat:@"Save password for %s?", keyname.c_str()] preferredStyle:UIAlertControllerStyleAlert];
  UIAlertAction *actionNo  = [UIAlertAction actionWithTitle:@"No"  style:UIAlertActionStyleDefault handler: nil];
  UIAlertAction *actionYes = [UIAlertAction actionWithTitle:@"Yes" style:UIAlertActionStyleDefault
    handler:^(UIAlertAction *){
      [IOSKeychain save:k data: val];
      [val replaceCharactersInRange:NSMakeRange(0, [val length]) withString:[NSString stringWithFormat:@"%*s", [val length], ""]];
      [val release];
      [k release];
    }];
  [alertController addAction:actionYes];
  [alertController addAction:actionNo];
  [[LFUIApplication sharedAppDelegate].controller presentViewController:alertController animated:YES completion:nil];
}

bool Application::LoadKeychain(const string &keyname, string *val_out) {
  NSString *k = [NSString stringWithFormat:@"%s://%s", name.c_str(), keyname.c_str()];
  NSString *val = [IOSKeychain load: k];
  if (val) val_out->assign(GetNSString(val));
  else     val_out->clear();
  return   val_out->size();
}

void Application::ShowAds() {}
void Application::HideAds() {}

int Application::LoadSystemImage(const string &n) {
  UIImage *image = [UIImage imageNamed:MakeNSString(n)];
  if (!image) return 0;
  app_images.push_back(image);
  return app_images.size();
}

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
