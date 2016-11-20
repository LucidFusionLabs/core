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
#import <QuartzCore/QuartzCore.h>
#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/osx_common.h"

static std::vector<NSImage*> app_images;

@interface OSXAlert : NSObject<NSTextFieldDelegate>
  @property (nonatomic, retain) NSAlert     *alert;
  @property (nonatomic, retain) NSTextField *input;
  @property (nonatomic)         bool         add_text;
  @property (nonatomic)         std::string  style;
  @property (nonatomic, assign) LFL::StringCB cancel_cb, confirm_cb;
@end

@implementation OSXAlert
  - (id)init:(LFL::AlertItemVec) kv {
    CHECK_EQ(4, kv.size());
    CHECK_EQ("style", kv[0].first);
    _style      = move(kv[0].second);
    _cancel_cb  = move(kv[2].cb);
    _confirm_cb = move(kv[3].cb);
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
    } else if ((_add_text = _style == "pwinput")) {
      _input = [[NSSecureTextField alloc] initWithFrame:NSMakeRect(0, 0, 200, 24)];
      _input.delegate = self;
      [_alert setAccessoryView: _input];
    }

    return self;
  }

  - (void)alertDidEnd:(NSAlert *)alert returnCode:(NSInteger)returnCode contextInfo:(void *)contextInfo {
    if (returnCode != NSAlertFirstButtonReturn) { if (_cancel_cb) _cancel_cb(""); }
    else { if (_confirm_cb) _confirm_cb(_add_text ? [[_input stringValue] UTF8String] : ""); }
  }
  
  - (void)modalAlertDidEnd:(NSAlert *)alert returnCode:(NSInteger)returnCode contextInfo:(void *)contextInfo {
    NSWindow *window = [LFL::GetTyped<GameView*>(LFL::app->focused->id) window];
    [NSApp endSheet: window];
  }

  - (void)controlTextDidChange:(NSNotification *)notification {}
  - (void)controlTextDidEndEditing:(NSNotification *)notification {}
@end

@interface OSXPanel : NSObject
  @property(readonly, assign) NSWindow *window;
  - (void) show;
  - (void) addTextField:(NSTextField*)tf withCB:(LFL::StringCB)cb;
@end

@implementation OSXPanel
  {
    std::vector<std::pair<NSButton*,    LFL::StringCB>> buttons;
    std::vector<std::pair<NSTextField*, LFL::StringCB>> textfields;
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

  - (void) addButton: (NSButton*)button withCB: (LFL::StringCB)cb {
    [button setTarget: self];
    [button setAction: @selector(buttonAction:)];
    [button setTag: buttons.size()];
    buttons.emplace_back(button, move(cb));
  }
  
  - (void) addTextField: (NSTextField*)textfield withCB: (LFL::StringCB)cb {
    [textfield setTarget: self];
    [textfield setAction: @selector(textFieldAction:)];
    [textfield setTag: textfields.size()];
    textfields.emplace_back(textfield, move(cb));
  }

  - (void) buttonAction: (id)sender {
    int tag = [sender tag];
    CHECK_RANGE(tag, 0, buttons.size());
    buttons[tag].second([[sender stringValue] UTF8String]);
  }

  - (void) textFieldAction: (id)sender {
    int tag = [sender tag];
    CHECK_RANGE(tag, 0, textfields.size());
    textfields[tag].second([[sender stringValue] UTF8String]);
  }
@end

@interface OSXToolbar : NSObject
  @property (nonatomic, retain) NSToolbar *toolbar;
  + (int)getBottomHeight;
@end

@implementation OSXToolbar
  + (int)getBottomHeight { return 0; }
@end

@interface OSXFontChooser : NSObject
@end

@implementation OSXFontChooser
  {
    NSFont *font;
    LFL::StringVecCB font_change_cb;
  }

  - (void)selectFont: (const char *)name size:(int)s cb:(LFL::StringVecCB)v {
    font_change_cb = move(v);
    font = [NSFont fontWithName:[NSString stringWithUTF8String:name] size:s];
    NSFontManager *fontManager = [NSFontManager sharedFontManager];
    [fontManager setSelectedFont:font isMultiple:NO];
    [fontManager setDelegate:self];
    [fontManager setTarget:self];
    [fontManager orderFrontFontPanel:self];
  }

  - (void)changeFont:(id)sender {
    font = [sender convertFont:font];
    float size = [[[font fontDescriptor] objectForKey:NSFontSizeAttribute] floatValue];
    font_change_cb(LFL::StringVec{ [[font fontName] UTF8String], LFL::StrCat(size) });
  }
@end

@interface OSXNavigation : NSResponder
  @property (nonatomic, copy)   NSViewController *rootViewController;
  @property (nonatomic, retain) NSMutableArray *viewControllers;
  @property (nonatomic, retain) NSView *view, *headerView;
  @property (nonatomic, retain) NSTextField *headerTitle;
  @property (nonatomic, retain) NSButton *headerBackButton;
  @property (nonatomic)         int header_height;
@end

@implementation OSXNavigation
  {
    CATransition *transition;
  }

  - (id)init { 
    self = [super init];
    _header_height = 30;
    NSView *contentView = LFL::GetTyped<GameView*>(LFL::app->focused->id).window.contentView;
    _view = [[NSView alloc] initWithFrame: contentView.frame];
    _view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _viewControllers = [[NSMutableArray alloc] init];
    CGRect frame = _view.frame;

    _headerView = [[NSView alloc] initWithFrame:
      CGRectMake(0, frame.size.height - _header_height, frame.size.width, _header_height)];
    _headerView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    _headerTitle = [[NSTextField alloc] initWithFrame: CGRectMake(0, 0, frame.size.width, _header_height)];
    _headerTitle.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _headerTitle.alignment = NSCenterTextAlignment;
    _headerTitle.editable = NO;
    _headerTitle.selectable = NO;
    [_headerView addSubview: _headerTitle];

    _headerBackButton = [[NSButton alloc] initWithFrame: CGRectMake(0, 0, _header_height*2, _header_height)];
    _headerBackButton.hidden = YES;
    _headerBackButton.target = self;
    _headerBackButton.action = @selector(backButtonClicked:);
    [_headerView addSubview: _headerBackButton];
    [_view addSubview: _headerView];

    transition = [CATransition animation];
    [transition setType:kCATransitionPush];
    [transition setSubtype:kCATransitionFromRight];
    NSDictionary *animations = [NSDictionary dictionaryWithObject:transition forKey:@"subviews"];
    [_view setAnimations:animations];
    return self;
  }

  - (void)pushViewController:(NSViewController*)vc animated:(BOOL)anim {
    [self popView];
    [_viewControllers addObject: vc];
    [self pushView];
  }

  - (NSViewController*)popViewControllerAnimated:(BOOL)anim {
    int count = [_viewControllers count];
    if (!count) return nil;
    [transition setSubtype:kCATransitionFromLeft];
    NSViewController *topViewController = [_viewControllers objectAtIndex: count-1];
    [self popView];
    [_viewControllers removeLastObject];
    [self pushView];
    return topViewController;
  }

  - (void)pushView {
    int count = [_viewControllers count];
    CHECK(count);
    NSViewController *topViewController = [_viewControllers objectAtIndex: count-1];
    [[_view animator] addSubview:topViewController.view];
    CGRect frame = _view.frame;
    frame.size.height -= _header_height;
    topViewController.view.frame = frame;

    _headerTitle.stringValue = topViewController.title;
    _headerBackButton.hidden = count == 1;
    if (count > 1) {
      NSViewController *backViewController = [_viewControllers objectAtIndex: count-2];
      _headerBackButton.title = backViewController.title;
    }
  }

  - (void)popView {
    int count = [_viewControllers count];
    if (!count) return;
    NSViewController *topViewController = [_viewControllers objectAtIndex: count-1];
    [topViewController.view removeFromSuperview];
  }

  - (void)backButtonClicked:(id)sender { [self popViewControllerAnimated: YES]; }
@end

@interface OSXTableCellView : NSTableCellView
  @property (nonatomic, retain) NSTextField *textValue;
@end

@implementation OSXTableCellView 
@end

@interface OSXTable : NSViewController<NSTableViewDataSource, NSTableViewDelegate>
  @property (nonatomic, retain) NSTableView *tableView;
  @property (nonatomic, copy)   NSScrollView *tableContainer;
  @property (nonatomic, assign) LFL::SystemTableView *lfl_self;
  @property (nonatomic, assign) LFL::IntIntCB delete_row_cb;
  @property (nonatomic)         std::string style;
  @property (nonatomic)         int editable_section, editable_start_row, selected_section, selected_row,
                                    second_col, row_height;
@end

@implementation OSXTable
  {
    int data_rows;
    std::vector<LFL::Table> data;
    std::vector<OSXTable*> dropdowns;
  }

  - (id)init: (LFL::SystemTableView*)lself withTitle:(const std::string&)title andStyle:(const std::string&)style items:(std::vector<LFL::Table>)item { 
    self = [super init];
    _lfl_self = lself;
    _style = style;
    _editable_section = _editable_start_row = -1;
    _row_height = 30;
    data = move(item);
    int count = 0;
    for (auto &i : data) {
      i.start_row = data_rows + count++;
      data_rows += i.item.size();
    }
    self.title = LFL::MakeNSString(title);

    NSView *contentView = LFL::GetTyped<GameView*>(LFL::app->focused->id).window.contentView;
    _tableContainer = [[NSScrollView alloc] initWithFrame: contentView.frame];
    self.view = [_tableContainer autorelease];
    self.view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    _tableView = [[NSTableView alloc] initWithFrame: contentView.frame];
    [_tableView setDelegate:self];
    [_tableView setDataSource:self];
    [_tableView setHeaderView:nil];
    _tableView.gridStyleMask = NSTableViewSolidHorizontalGridLineMask;
    [_tableView setSelectionHighlightStyle:NSTableViewSelectionHighlightStyleNone];
    [_tableContainer setDocumentView:_tableView];
    [_tableContainer setHasVerticalScroller:YES];

    NSTableColumn *column = [[NSTableColumn alloc] initWithIdentifier:@"Column"];
    [column setWidth: _tableView.frame.size.width];
    [_tableView addTableColumn:column];
    [column release];
    return self;
  }

  - (void)replaceSection:(int)section items:(std::vector<LFL::TableItem>)item header:(const std::string&)h image:(int)im flag:(int)f addbutton:(LFL::Callback)addb {
    bool added = section == data.size();
    if (added) data.emplace_back();
    CHECK_LT(section, data.size());
    int item_size = item.size();
    data_rows += item_size - data[section].item.size();
    data[section] = LFL::Table(h, im, f, move(addb));
    data[section].item = move(item);
    if (added) data.back().start_row = data_rows + data.size() - item_size - 1;
#if 0
    [_tableView reloadDataForRowIndexes: [NSIndexSet indexSetWithIndexesInRange:NSMakeRange(0, data[section].item.size())]
      columnIndexes: [NSIndexSet indexSetWithIndexesInRange:NSMakeRange(0, 1)]];
#else
    [_tableView reloadData];
#endif
  }

  - (void)checkExists:(int)section row:(int)r {
    if (section == data.size()) { data.emplace_back(); data.back().start_row = data_rows + data.size(); }
    CHECK_LT(section, data.size());
    CHECK_LT(r, data[section].item.size());
  }

  // - (void)loadView { self.view = _tableContainer; }
  - (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView { return data_rows + data.size(); }

  - (CGFloat)tableView:(NSTableView *)tableView heightOfRow:(NSInteger)collapsed_row {
    int section = -1, row = -1;
    [self findSectionAndRowIndex: collapsed_row outSection: &section outRow: &row];
    if (section < 0 || section >= data.size() || row < 0 || row >= data[section].item.size()) return _row_height;
    const auto &ci = data[section].item[row];
    if (ci.gui_loaded &&
             (ci.type == LFL::TableItem::Picker || ci.type == LFL::TableItem::FontPicker)) return ci.height;
    else return _row_height; 
  }

  - (void)findSectionAndRowIndex:(int)collapsed_row outSection:(int*)out_section outRow:(int*)out_row {
    auto it = LFL::lower_bound(data.begin(), data.end(), LFL::Table(collapsed_row),
                               LFL::MemberLessThanCompare<LFL::Table, int, &LFL::Table::start_row>());
    if (it != data.end() && it->start_row == collapsed_row) { *out_section = it - data.begin(); return; }
    CHECK_NE(data.begin(), it);
    *out_section = (it != data.end() ? (it - data.begin()) : data.size()) - 1;
    *out_row = collapsed_row - data[*out_section].start_row - 1;
  }

#if 0
  - (id)tableView:(NSTableView *)tableView objectValueForTableColumn:(NSTableColumn *)tableColumn row:(NSInteger)row {
    return [NSString stringWithFormat:@"%ld", row];
  }
#endif

  - (NSView *)tableView:(NSTableView *)tableView viewForTableColumn:(NSTableColumn *)tableColumn row:(NSInteger)collapsed_row {
    if (!data.size()) return nil;
    NSString *identifier = [tableColumn identifier];
    OSXTableCellView *cellView = [tableView makeViewWithIdentifier:identifier owner:self];
    if (cellView == nil) {
      int section = -1, row = -1;
      [self findSectionAndRowIndex: collapsed_row outSection: &section outRow: &row];
      if (row < 0) return (cellView = [self headerForSection: section]);

      [self checkExists:section row:row];
      auto &compiled_item = data[section].item[row];
      int type = compiled_item.type, icon = 0;
      const LFL::string *v = &compiled_item.val;

      cellView = [[[OSXTableCellView alloc] initWithFrame:CGRectMake(0, 0, 200, _row_height)] autorelease];
      cellView.identifier = identifier;

      if (type != LFL::TableItem::Button && (icon = compiled_item.left_icon)) {
        CHECK_LE(icon, app_images.size());
        cellView.imageView = [NSImageView imageViewWithImage: app_images[icon - 1]];
        cellView.imageView.frame = CGRectMake(0, 0, _row_height, _row_height);
        [cellView addSubview:cellView.imageView];
      }

      cellView.textField = [[[NSTextField alloc] initWithFrame:
        CGRectMake(icon ? _row_height : 0, 0, 100, _row_height)] autorelease];
      cellView.textField.bordered = NO;
      cellView.textField.drawsBackground = NO;
      cellView.textField.editable = NO;
      cellView.textField.selectable = NO;
      cellView.textField.stringValue = LFL::MakeNSString(data[0].item[row].key);
      [cellView addSubview:cellView.textField];

      bool textinput=0, numinput=0, pwinput=0;
      if ((textinput = type == LFL::TableItem::TextInput) || (numinput = type == LFL::TableItem::NumberInput)
          || (pwinput = type == LFL::TableItem::PasswordInput)) {
        cellView.textValue = [[[NSTextField alloc] initWithFrame:
          CGRectMake(tableView.frame.size.width - 100, 0, 100, _row_height)] autorelease];
        cellView.textValue.bordered = NO;
        cellView.textValue.drawsBackground = NO;
        cellView.textValue.editable = YES;
        cellView.textValue.selectable = YES;
        cellView.textValue.alignment = NSRightTextAlignment;
        if (v->size() && ((*v)[0] == 1 || (*v)[0] == 2)) cellView.textValue.placeholderString = LFL::MakeNSString(v->substr(1));
        else if (v->size())                              cellView.textValue.stringValue       = LFL::MakeNSString(*v);
        [cellView addSubview:cellView.textValue];
      
      } else if (type == LFL::TableItem::Selector) {
      } else if (type == LFL::TableItem::Picker || type == LFL::TableItem::FontPicker) {
      } else if (type == LFL::TableItem::Button) {
      }
    }
    return cellView;
  }

  - (OSXTableCellView *)headerForSection:(int)section {
    return nil;
  }

  - (BOOL)tableView:(NSTableView *)tableView shouldSelectRow:(NSInteger)collapsed_row {
    int section = -1, row = -1;
    [self findSectionAndRowIndex: collapsed_row outSection: &section outRow: &row];
    if (row < 0) NO;
    [self checkExists:section row:row];
    _selected_section = section;
    _selected_row = row;

    auto &compiled_item = data[section].item[row];
    if (compiled_item.type == LFL::TableItem::Command || compiled_item.type == LFL::TableItem::Button) {
      compiled_item.cb();
    } else if (compiled_item.type == LFL::TableItem::Label && row + 1 < data[section].item.size()) {
      auto &next_compiled_item = data[section].item[row+1];
      if (next_compiled_item.type == LFL::TableItem::Picker ||
          next_compiled_item.type == LFL::TableItem::FontPicker) {
        next_compiled_item.hidden = !next_compiled_item.hidden;
        // [self reloadRowAtIndexPath:path withRowAnimation:UITableViewRowAnimationNone];
      }
    }
    return NO;
  }
@end

@interface OSXTextView : NSViewController<NSTextViewDelegate>
  @property (nonatomic, retain) NSTextView *textView;
  @property (nonatomic, retain) NSString *text;
@end

@implementation OSXTextView
  - (id)initWithTitle:(NSString*)title andText:(NSString*)t {
    self = [super init];
    self.title = title;
    self.text = t;
    return self;
  }
@end

namespace LFL {
static void AddNSMenuItems(NSMenu *menu, vector<MenuItem> items) {
  NSMenuItem *item;
  for (auto &i : items) { 
    const string &k=i.shortcut, &n=i.name;
    if (n == "<separator>") { [menu addItem:[NSMenuItem separatorItem]]; continue; }

    NSString *key = nil;
    if      (k == "<left>")  { unichar fk = NSLeftArrowFunctionKey;  key = [NSString stringWithCharacters:&fk length:1]; }
    else if (k == "<right>") { unichar fk = NSRightArrowFunctionKey; key = [NSString stringWithCharacters:&fk length:1]; }
    else if (k == "<up>"   ) { unichar fk = NSUpArrowFunctionKey;    key = [NSString stringWithCharacters:&fk length:1]; }
    else if (k == "<down>")  { unichar fk = NSDownArrowFunctionKey;  key = [NSString stringWithCharacters:&fk length:1]; }
    else key = [NSString stringWithUTF8String: k.c_str()];

    item = [menu addItemWithTitle: [NSString stringWithUTF8String: n.c_str()]
                 action:           (i.cb ? @selector(callbackRun:) : nil)
                 keyEquivalent:    key];
    if (i.cb) [item setRepresentedObject: [[ObjcCallback alloc] initWithCB: move(i.cb)]];
  }
}

SystemAlertView::~SystemAlertView() { if (auto alert = FromVoid<OSXAlert*>(impl)) [alert release]; }
SystemAlertView::SystemAlertView(AlertItemVec items) { impl = [[OSXAlert alloc] init: move(items)]; }

void SystemAlertView::Show(const string &arg) {
  auto alert = FromVoid<OSXAlert*>(impl);
  if (alert.add_text) [alert.input setStringValue: MakeNSString(arg)];
  [alert.alert beginSheetModalForWindow:[GetTyped<GameView*>(app->focused->id) window] modalDelegate:alert
    didEndSelector:@selector(alertDidEnd:returnCode:contextInfo:) contextInfo:nil];
  [GetTyped<GameView*>(app->focused->id) clearKeyModifiers];
}

void SystemAlertView::ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
  auto alert = FromVoid<OSXAlert*>(impl);
  alert.confirm_cb = move(confirm_cb);
  [alert.alert setMessageText: MakeNSString(title)];
  [alert.alert setInformativeText: MakeNSString(msg)];
  [alert.alert beginSheetModalForWindow:[GetTyped<GameView*>(app->focused->id) window] modalDelegate:alert
    didEndSelector:@selector(alertDidEnd:returnCode:contextInfo:) contextInfo:nil];
  [GetTyped<GameView*>(app->focused->id) clearKeyModifiers];
}

string SystemAlertView::RunModal(const string &arg) {
  auto alert = FromVoid<OSXAlert*>(impl);
  NSWindow *window = [GetTyped<GameView*>(app->focused->id) window];
  if (alert.add_text) [alert.input setStringValue: MakeNSString(arg)];
  [alert.alert beginSheetModalForWindow:window modalDelegate:alert
    didEndSelector:@selector(modalAlertDidEnd:returnCode:contextInfo:) contextInfo:nil];
  [NSApp runModalForWindow: window];
  return alert.add_text ? GetNSString([alert.input stringValue]) : "";
}

SystemMenuView::~SystemMenuView() { if (auto menu = FromVoid<NSMenu*>(impl)) [menu release]; }
SystemMenuView::SystemMenuView(const string &title_text, MenuItemVec items) {
  NSString *title = [NSString stringWithUTF8String: title_text.c_str()];
  NSMenu *menu = [[NSMenu alloc] initWithTitle: title];
  AddNSMenuItems(menu, move(items));
  NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
}

void SystemMenuView::Show() {}
unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(MenuItemVec items) {
  NSMenuItem *item; 
  NSMenu *menu = [[NSMenu alloc] initWithTitle:@"Edit"];
  item = [menu addItemWithTitle:@"Copy"  action:@selector(copy:)  keyEquivalent:@"c"];
  item = [menu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
  AddNSMenuItems(menu, move(items));
  item = [[NSMenuItem alloc] initWithTitle:@"Edit" action:nil keyEquivalent:@""];
  [item setSubmenu: menu];
  [[NSApp mainMenu] addItem: item];
  [menu release];
  [item release];
  return make_unique<SystemMenuView>(nullptr);
}

SystemPanelView::~SystemPanelView() { if (auto panel = FromVoid<OSXPanel*>(impl)) [panel release]; }
SystemPanelView::SystemPanelView(const Box &b, const string &title, PanelItemVec items) {
  OSXPanel *panel = [[OSXPanel alloc] initWithBox: b];
  [[panel window] setTitle: [NSString stringWithUTF8String: title.c_str()]];
  for (auto &i : items) {
    const Box &b = i.box;
    const string &t = i.type;
    if (t == "textbox") {
      NSTextField *textfield = [[NSTextField alloc] initWithFrame:NSMakeRect(b.x, b.y, b.w, b.h)];
      [panel addTextField: textfield withCB: move(i.cb)];
      [[[panel window] contentView] addSubview: textfield];
    } else if (PrefixMatch(t, "button:")) {
      NSButton *button = [[[NSButton alloc] initWithFrame:NSMakeRect(b.x, b.y, b.w, b.h)] autorelease];
      [panel addButton: button withCB: move(i.cb)];
      [[[panel window] contentView] addSubview: button];
      [button setTitle: [NSString stringWithUTF8String: t.substr(7).c_str()]];
      [button setButtonType:NSMomentaryLightButton];
      [button setBezelStyle:NSRoundedBezelStyle];
    } else ERROR("unknown panel item ", t);
  }
  impl = panel;
}

void SystemPanelView::Show() { [FromVoid<OSXPanel*>(impl) show]; }
void SystemPanelView::SetTitle(const string &title) { 
  [[FromVoid<OSXPanel*>(impl) window] setTitle: [NSString stringWithUTF8String: title.c_str()]];
}

SystemToolbarView::~SystemToolbarView() {}
SystemToolbarView::SystemToolbarView(MenuItemVec items) : impl(0) {}
void SystemToolbarView::Show(bool show_or_hide) {}
void SystemToolbarView::ToggleButton(const string &n) {}

SystemTableView::~SystemTableView() { if (auto table = FromVoid<OSXTable*>(impl)) [table release]; }
SystemTableView::SystemTableView(const string &title, const string &style, TableItemVec items, int second_col) :
  impl([[OSXTable alloc] init:this withTitle:title andStyle:style items:Table::Convert(move(items))]) {}

void SystemTableView::DelNavigationButton(int align) {}
void SystemTableView::AddNavigationButton(int align, const TableItem &item) {}
void SystemTableView::AddToolbar(SystemToolbarView *t) {
}

void SystemTableView::Show(bool show_or_hide) {
  auto table = FromVoid<OSXTable*>(impl);
  NSView *contentView = LFL::GetTyped<GameView*>(LFL::app->focused->id).window.contentView;
  if (show_or_hide) {
    [[contentView animator] addSubview:table.tableContainer];
    [table.tableContainer setFrameOrigin:CGPointMake(0, 0)];
  } else 
    [table.tableContainer removeFromSuperview];
}

void SystemTableView::AddRow(int section, TableItem item) {}
string SystemTableView::GetKey(int section, int row) { return ""; }
int SystemTableView::GetTag(int section, int row) { return 0; }
void SystemTableView::SetTag(int section, int row, int val) {}
void SystemTableView::SetValue(int section, int row, const string &val) {}
void SystemTableView::SetHidden(int section, int row, bool val) {}
void SystemTableView::SetTitle(const string &title) {}
PickerItem *SystemTableView::GetPicker(int section, int row) { return 0; }
StringPairVec SystemTableView::GetSectionText(int section) { return StringPairVec(); }
void SystemTableView::SetEditableSection(int section, int start_row, LFL::IntIntCB cb) {
}

void SystemTableView::SelectRow(int section, int row) {
} 

void SystemTableView::BeginUpdates() {}
void SystemTableView::EndUpdates() {}
void SystemTableView::SetDropdown(int section, int row, int val) {}
void SystemTableView::SetSectionValues(int section, const StringVec &item) {}
void SystemTableView::ReplaceSection(int section, const string &h, int image, int flag, TableItemVec item, Callback add_button)
{ [FromVoid<OSXTable*>(impl) replaceSection:section items:move(item) header:h image:image flag:flag addbutton:move(add_button)]; }

SystemTextView::~SystemTextView() { if (auto view = FromVoid<OSXTextView*>(impl)) [view release]; }
SystemTextView::SystemTextView(const string &title, File *f) : SystemTextView(title, f ? f->Contents() : "") {}
SystemTextView::SystemTextView(const string &title, const string &text) :
  impl([[OSXTextView alloc] initWithTitle:MakeNSString(title) andText:[[[NSString alloc]
       initWithBytes:text.data() length:text.size() encoding:NSASCIIStringEncoding] autorelease]]) {}

SystemNavigationView::~SystemNavigationView() { if (auto nav = FromVoid<OSXNavigation*>(impl)) [nav release]; }
SystemNavigationView::SystemNavigationView() : impl([[OSXNavigation alloc] init]) {}
void SystemNavigationView::Show(bool show_or_hide) {
  auto nav = FromVoid<OSXNavigation*>(impl);
  GameContainerView *contentView = LFL::GetTyped<GameView*>(LFL::app->focused->id).window.contentView;
  if ((shown = show_or_hide)) {
    [contentView.gameView removeFromSuperview];
    [[contentView animator] addSubview:nav.view];
    [nav.view setFrameOrigin:CGPointMake(0, 0)];
  } else 
    [nav.view removeFromSuperview];
}

SystemTableView *SystemNavigationView::Back() {
  for (NSViewController *c in [FromVoid<OSXNavigation*>(impl).viewControllers reverseObjectEnumerator]) {
    if ([c isKindOfClass:[OSXTable class]])
      if (auto lself = static_cast<OSXTable*>(c).lfl_self) return lself;
  } 
  return nullptr;
}

void SystemNavigationView::PushTableView(SystemTableView *t) {
  if (!root) root = t;
  if (t->show_cb) t->show_cb();
  [FromVoid<OSXNavigation*>(impl) pushViewController: FromVoid<OSXTable*>(t->impl) animated: YES];
}

void SystemNavigationView::PushTextView(SystemTextView *t) {
  if (t->show_cb) t->show_cb();
  [FromVoid<OSXNavigation*>(impl) pushViewController: FromVoid<OSXTextView*>(t->impl) animated: YES];
}

void SystemNavigationView::PopToRoot() {}
void SystemNavigationView::PopAll() {}

void SystemNavigationView::PopView(int n) {
  for (int i = 0; i != n; ++i)
    [FromVoid<OSXNavigation*>(impl) popViewControllerAnimated: (i == n - 1)];
}

SystemAdvertisingView::SystemAdvertisingView() {}
void SystemAdvertisingView::Show() {}
void SystemAdvertisingView::Hide() {}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &choose_cb) {
  static OSXFontChooser *font_chooser = [OSXFontChooser alloc];
  [font_chooser selectFont:cur_font.name.c_str() size:cur_font.size cb:choose_cb];
}

void Application::ShowSystemFileChooser(bool choose_files, bool choose_dirs, bool choose_multi, const StringVecCB &choose_cb) {
  NSOpenPanel *panel = [NSOpenPanel openPanel];
  [panel setCanChooseFiles:choose_files];
  [panel setCanChooseDirectories:choose_dirs];
  [panel setAllowsMultipleSelection:choose_multi];
  NSInteger clicked = [panel runModal];
  app->LoseFocus();
  if (clicked != NSFileHandlingPanelOKButton) return;

  StringVec arg;
  for (NSURL *url in [panel URLs]) arg.emplace_back([[url absoluteString] UTF8String]);
  if (arg.size()) choose_cb(arg);
}

void Application::ShowSystemContextMenu(const MenuItemVec &items) {
  NSEvent *event = [NSEvent mouseEventWithType: NSLeftMouseDown
                            location:           NSMakePoint(app->focused->mouse.x, app->focused->mouse.y)
                            modifierFlags:      NSLeftMouseDownMask
                            timestamp:          0
                            windowNumber:       [[LFL::GetTyped<GameView*>(app->focused->id) window] windowNumber]
                            context:            [[LFL::GetTyped<GameView*>(app->focused->id) window] graphicsContext]
                            eventNumber:        0
                            clickCount:         1
                            pressure:           1];

  NSMenuItem *item;
  NSMenu *menu = [[NSMenu alloc] init];
  for (auto &i : items) {
    const char *k = i.shortcut.c_str(), *n = i.name.c_str();
    if (!strcmp(n, "<separator>")) { [menu addItem:[NSMenuItem separatorItem]]; continue; }
    item = [menu addItemWithTitle: [NSString stringWithUTF8String: n]
                 action:           (i.cb ? @selector(callbackRun:) : nil)
                 keyEquivalent:    [NSString stringWithUTF8String: k]];
    if (i.cb) [item setRepresentedObject: [[ObjcCallback alloc] initWithCB: i.cb]];
  }

  [NSMenu popUpContextMenu:menu withEvent:event forView:LFL::GetTyped<GameView*>(app->focused->id)];
  [LFL::GetTyped<GameView*>(app->focused->id) clearKeyModifiers];
}

void Application::OpenSystemBrowser(const string &url_text) {
  CFURLRef url = CFURLCreateWithBytes(0, MakeUnsigned(url_text.c_str()), url_text.size(), kCFStringEncodingASCII, 0);
  if (url) { LSOpenCFURLRef(url, 0); CFRelease(url); }
}

int Application::LoadSystemImage(const string &n) {
  NSImage *image = [[NSImage alloc] initWithContentsOfFile: MakeNSString(StrCat(app->assetdir, "../", n)) ];
  if (!image) return 0;
  app_images.push_back(image);
  return app_images.size();
}

string Application::GetSystemDeviceName() {
  string ret(1024, 0);
  if (gethostname(&ret[0], ret.size())) return "";
  ret.resize(strlen(ret.c_str()));
  return ret;
}

Connection *Application::ConnectTCP(const string &hostport, int default_port, Connection::CB *connected_cb, bool background_services) {
#if 1
  INFO("Application::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = ", background_services);
  if (background_services) return new NSURLSessionStreamConnection(hostport, default_port, connected_cb ? move(*connected_cb) : Connection::CB());
  else return app->net->tcp_client->Connect(hostport, default_port, connected_cb);
#else
  INFO("Application::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = false"); 
  return app->net->tcp_client->Connect(hostport, default_port, connected_cb);
#endif
}

}; // namespace LFL
