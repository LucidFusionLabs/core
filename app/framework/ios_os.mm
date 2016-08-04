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
  @property (nonatomic) std::string cmd;
@end
@implementation IOSButton
@end

@interface IOSBarButtonItem : UIBarButtonItem
  @property (nonatomic) std::string cmd;
@end
@implementation IOSBarButtonItem
@end

@interface IOSSegmentedControl : UISegmentedControl
  @property (copy) void (^changed)(const std::string&);
@end
@implementation IOSSegmentedControl
@end

@interface IOSAlert : NSObject<UIAlertViewDelegate>
  @property (nonatomic, retain) UIAlertView *alert;
  @property (nonatomic)         bool         add_text, done;
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
    if      ((_add_text = _style == "textinput")) _alert.alertViewStyle = UIAlertViewStylePlainTextInput;
    else if ((_add_text = _style == "pwinput"))   _alert.alertViewStyle = UIAlertViewStyleSecureTextInput;
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
  - (id)init:(const std::string&)title_text items:(const std::vector<LFL::MenuItem>&)item {
    menu = item;
    NSString *title = [NSString stringWithUTF8String: title_text.c_str()];
    _actions = [[UIActionSheet alloc] initWithTitle:title delegate:self
      cancelButtonTitle:@"Cancel" destructiveButtonTitle:nil otherButtonTitles:nil];
    for (auto &i : menu) [_actions addButtonWithTitle: LFL::MakeNSString(i.name)];
    return self;
  }

  - (void)actionSheet:(UIActionSheet *)actions clickedButtonAtIndex:(NSInteger)buttonIndex {
    if (buttonIndex < 1 || buttonIndex > menu.size()) { ERRORf("invalid buttonIndex %d size=%d", buttonIndex, menu.size()); return; }
    ShellRun(menu[buttonIndex-1].cmd.c_str());
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
  @property (nonatomic, retain) UINavigationController *modal_nav;
  @property (nonatomic, assign) IOSToolbar *toolbar;
  @property (nonatomic)         std::string style;
  @property (nonatomic)         int editable_section, selected_section, selected_row, second_col;
  @property (copy)              void (^completed)();
@end

@implementation IOSTable
  {
    int section_index;
    std::vector<LFL::CompiledTable> data;
    std::vector<IOSTable*> dropdowns;
  }
  
  - (void)load:(const std::string&)title withStyle:(const std::string&)sty items:(const std::vector<LFL::TableItem>&)item {
    _style = sty;
    data.emplace_back();
    for (auto &i : item) {
      if (i.type == "separator") {
        data.push_back({i.key, std::vector<LFL::CompiledTableItem>()});
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

  - (void)replaceSection:(int)section items:(const std::vector<LFL::TableItem>&)item {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    data[section] = LFL::CompiledTable();
    for (auto &i : item) data[section].item.emplace_back(i);
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex: section]
      withRowAnimation:UITableViewRowAnimationNone];
  }

  - (void)setSectionDropdown:(int)section row:(int)r index:(int)ind {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_LT(r, data[section].item.size());
    NSIndexPath *path = [NSIndexPath indexPathForRow:r inSection:section];
    UITableViewCell *cell = [self.tableView cellForRowAtIndexPath: path];
    const std::string *k=0, *type=0, *v=0;
    auto &compiled_item = data[path.section].item[path.row];
    [self loadCellItem:cell withPath:path withItem:&compiled_item outK:&k outT:&type outV:&v];
    if (compiled_item.type == LFL::CompiledTableItem::Dropdown) {
      CHECK_RANGE(compiled_item.ref, 0, dropdowns.size());
      auto dropdown_table = dropdowns[compiled_item.ref];
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
    for (int i=0, l=data[section].item.size(); i != l; ++i) {
      auto &ci = data[section].item[i];
      ci.item.val = item[i];
      if (ci.loaded && ci.type == LFL::CompiledTableItem::Dropdown) {
        std::vector<std::string> vv=LFL::Split(item[i], ',');
        CHECK_RANGE(ci.ref, 0, dropdowns.size());
        auto dropdown_table = dropdowns[ci.ref];
        CHECK_EQ(dropdown_table->data[0].item.size(), vv.size()-1);
        for (int j=1; j != vv.size(); ++j) dropdown_table->data[0].item[j].item.val = vv[j];
      }
    }
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex: section]
      withRowAnimation:UITableViewRowAnimationNone];
  }

  - (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView { return data.size(); }
  - (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    return data[section].item.size();
  }

  - (void)loadNavigationButton:(const LFL::TableItem&)item withAlign:(int)align {
    if (item.key == "Edit") {
      self.navigationItem.rightBarButtonItem = [self editButtonItem];
    } else {
      IOSBarButtonItem *button = [[IOSBarButtonItem alloc] init];
      button.cmd = item.val;
      button.title = LFL::MakeNSString(item.key);
      [button setTarget:self];
      [button setAction:@selector(rightNavButtonClicked:)];
      self.navigationItem.rightBarButtonItem = button;
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
        std::vector<LFL::TableItem> dropdown_options;
        for (int i=1, l=kv.size(); i != l; ++i) dropdown_options.push_back({kv[i], tv[i], vv[i]});
        auto dropdown_table = [[IOSTable alloc] initWithStyle: UITableViewStyleGrouped];
        [dropdown_table load:kv[0] withStyle:"dropdown" items:dropdown_options];
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

      if (compiled_item.type == LFL::CompiledTableItem::Dropdown) {
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

      if (int icon = compiled_item.item.right_icon) {
        CHECK_LE(icon, app_images.size());
        UIImage *image = app_images[icon - 1]; 
        IOSButton *button = [IOSButton buttonWithType:UIButtonTypeCustom];
        button.frame = CGRectMake(0, 0, image.size.width, image.size.height);
        [button setImage:image forState:UIControlStateNormal];
        button.cmd = compiled_item.item.right_icon_cmd;
        [button addTarget:self action:@selector(rightIconClicked:) forControlEvents:UIControlEventTouchUpInside];
        cell.accessoryView = button;
      }

      bool textinput=0, numinput=0, pwinput=0;
      if ((textinput = *type == "textinput") || (numinput = *type == "numinput") || (pwinput = *type == "pwinput")) {
        UITextField *textfield;
        if (_second_col) textfield = [[UITextField alloc] initWithFrame: CGRectMake(_second_col, 10, tableView.frame.size.width - _second_col - 30, 30)];
        else             textfield = [[UITextField alloc] initWithFrame: CGRectMake(0, 0, 150, 30)];
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

        if (pwinput) [textfield setText:        LFL::MakeNSString(*v)];
        else         [textfield setPlaceholder: LFL::MakeNSString(*v)];
        cell.textLabel.text = LFL::MakeNSString(*k);
        if (_second_col) [cell.contentView addSubview: textfield];
        else {
          textfield.textAlignment = NSTextAlignmentRight;
          cell.accessoryView = textfield;
        }
        if (path.section == 0 && path.row == 0) [textfield becomeFirstResponder];
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
        [onoff addTarget: self action: @selector(switchFlipped:) forControlEvents: UIControlEventValueChanged];
        cell.textLabel.text = LFL::MakeNSString(*k);
        cell.accessoryView = onoff;
        [onoff release];

      } else if (*type == "label") {
        UILabel *label = [[UILabel alloc] init];
        label.text = LFL::MakeNSString(*v);
        cell.textLabel.text = LFL::MakeNSString(*k);
        cell.accessoryView = label;
        [label release];

      } else {
        cell.textLabel.text = LFL::MakeNSString(*k);
      }
    }
    return cell;
  }

  - (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    CHECK_LT(section, data.size());
    return LFL::MakeNSString(data[section].header);
  }

  - (UITableViewCellEditingStyle)tableView:(UITableView *)tableView editingStyleForRowAtIndexPath:(NSIndexPath *)path {
    return path.section == _editable_section ? UITableViewCellEditingStyleDelete : UITableViewCellEditingStyleNone;
  }

  - (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)path {
    CHECK_LT(path.section, data.size());
    CHECK_LT(path.row, data[path.section].item.size());
    _selected_row = path.row;
    _selected_section = path.section;
    const auto &compiled_item = data[path.section].item[path.row];
    const std::string &t = compiled_item.item.type;
    if (t == "command" || t == "button") ShellRun(compiled_item.item.val.c_str());
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

  - (void)dropDownClicked:(UIButton *)sender {
    int dropdown_ind = sender.tag;
    CHECK_LT(dropdown_ind, dropdowns.size());
    auto dropdown_table = dropdowns[dropdown_ind];
    [dropdown_table show:true];
  }

  - (void)segmentedControlClicked:(IOSSegmentedControl *)segmented_control {
    if (segmented_control.changed) segmented_control.changed
      (LFL::GetNSString([segmented_control titleForSegmentAtIndex: segmented_control.selectedSegmentIndex]));
  }

  - (void)rightIconClicked:(IOSButton*)sender {
    if (!sender.cmd.empty()) ShellRun(sender.cmd.c_str());
  }

  - (IBAction)rightNavButtonClicked:(IOSBarButtonItem*)sender {
    if (!sender.cmd.empty()) ShellRun(sender.cmd.c_str());
  }

  - (IBAction) switchFlipped: (UISwitch*) onoff {
    INFO("switch now ", onoff.on ? "On" : "Off");
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
      if ((*type == "textinput") || (*type == "numinput") || (*type == "pwinput")) {
        UITextField *textfield = _second_col ? [[cell.contentView subviews] lastObject] : cell.accessoryView;
        val = LFL::GetNSString(textfield.text);
        if (val.empty()) val = *v;
      } else if (*type == "select") {
        UISegmentedControl *segmented_control = [[cell.contentView subviews] lastObject];
        val = LFL::GetNSString([segmented_control titleForSegmentAtIndex: segmented_control.selectedSegmentIndex]);
      } else if (*type == "label") {
        UILabel *label = (UILabel*)cell.accessoryView;
        val = LFL::GetNSString(label.text);
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
UIAlertView *GetUIAlertView(SystemToolbarWidget *w) { return FromVoid<IOSAlert*>(w->impl).alert; }
void SystemAlertWidget::Show(const string &arg) {
  auto alert = FromVoid<IOSAlert*>(impl);
  if (alert.add_text) [alert.alert textFieldAtIndex:0].text = MakeNSString(arg);
  [alert.alert show];
}

string SystemAlertWidget::RunModal(const string &arg) {
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

SystemMenuWidget::~SystemMenuWidget() { if (auto menu = FromVoid<IOSMenu*>(impl)) [menu release]; }
SystemMenuWidget::SystemMenuWidget(const string &t, const vector<MenuItem> &i) : impl([[IOSMenu alloc] init:t items:i]) {}
void SystemMenuWidget::Show() { [FromVoid<IOSMenu*>(impl).actions showInView:[UIApplication sharedApplication].keyWindow]; }
unique_ptr<SystemMenuWidget> SystemMenuWidget::CreateEditMenu(const vector<MenuItem> &items) { return nullptr; }

SystemToolbarWidget::~SystemToolbarWidget() { if (auto toolbar = FromVoid<IOSToolbar*>(impl)) [toolbar release]; }
SystemToolbarWidget::SystemToolbarWidget(const StringPairVec &items) : impl([[IOSToolbar alloc] init: items]) {}
void SystemToolbarWidget::Show(bool show_or_hide) { [FromVoid<IOSToolbar*>(impl) show:show_or_hide]; }
void SystemToolbarWidget::ToggleButton(const string &n) { [FromVoid<IOSToolbar*>(impl) toggleButtonNamed: n]; }

SystemTableWidget::~SystemTableWidget() { if (auto table = FromVoid<IOSTable*>(impl)) [table release]; }
SystemTableWidget::SystemTableWidget(const string &title, const string &style, const vector<TableItem>&items, int second_col) {
  auto table = [[IOSTable alloc] initWithStyle: UITableViewStyleGrouped];
  if (second_col) table.second_col = second_col;
  [table load:title withStyle:style items:items];
  impl = table;
}

void SystemTableWidget::AddNavigationButton(const TableItem &item, int align) { return [FromVoid<IOSTable*>(impl) loadNavigationButton:item withAlign:align]; }
void SystemTableWidget::AddToolbar(SystemToolbarWidget *t) { [FromVoid<IOSTable*>(impl) setToolbar: FromVoid<IOSToolbar*>(t->impl)]; }
void SystemTableWidget::SetEditableSection(int section) { FromVoid<IOSTable*>(impl).editable_section = section; }
StringPairVec SystemTableWidget::GetSectionText(int section) { return [FromVoid<IOSTable*>(impl) dumpDataForSection:section]; }
void SystemTableWidget::BeginUpdates() { [FromVoid<IOSTable*>(impl).tableView beginUpdates]; }
void SystemTableWidget::EndUpdates() { [FromVoid<IOSTable*>(impl).tableView endUpdates]; }
void SystemTableWidget::ReplaceSection(const vector<TableItem> &item, int section) { [FromVoid<IOSTable*>(impl) replaceSection:section items:item]; }
void SystemTableWidget::SetSectionValues(const StringVec &item, int section) { [FromVoid<IOSTable*>(impl) setSectionValues:section items:item]; }
void SystemTableWidget::SetSectionDropdownValue(int section, int row, int val) { [FromVoid<IOSTable*>(impl) setSectionDropdown:section row:row index:val]; }

void SystemTableWidget::Show(bool show_or_hide) {
  if (show_or_hide && show_cb) show_cb(this);
  [FromVoid<IOSTable*>(impl) show:show_or_hide];
}

SystemNavigationWidget::~SystemNavigationWidget() { if (auto nav = FromVoid<IOSNavigation*>(impl)) [nav release]; }
SystemNavigationWidget::SystemNavigationWidget(SystemTableWidget *r) : impl([[IOSNavigation alloc] init: FromVoid<IOSTable*>(r->impl)]), root(r) {}
void SystemNavigationWidget::Show(bool show_or_hide) {
  auto nav = FromVoid<IOSNavigation*>(impl);
  LFUIApplication *uiapp = [LFUIApplication sharedAppDelegate];
  if (show_or_hide) {
    if (root->show_cb) root->show_cb(root);
    uiapp.top_controller = nav.controller;
    [uiapp.controller presentViewController: nav.controller animated:YES completion:nil];
  } else {
    uiapp.top_controller = uiapp.controller;
    [uiapp.controller dismissViewControllerAnimated:YES completion:nil];
  }
}

void SystemNavigationWidget::PushTable(SystemTableWidget *t) {
  if (t->show_cb) t->show_cb(t);
  [FromVoid<IOSNavigation*>(impl).controller pushViewController: FromVoid<IOSTable*>(t->impl) animated: YES];
}

void SystemNavigationWidget::PopTable(int n) {
  for (int i = 0; i != n; ++i)
    [FromVoid<IOSNavigation*>(impl).controller popViewControllerAnimated: (i == n - 1)];
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
