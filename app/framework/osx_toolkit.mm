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

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/osx_common.h"

struct OSXTableItem { enum { GUILoaded=LFL::TableItem::Flag::User1 }; };
static LFL::FreeListVector<NSImage*> app_images;

@interface OSXToolbar : NSObject
  @property (nonatomic, retain) NSToolbar *toolbar;
  + (int)getBottomHeight;
@end

@implementation OSXToolbar
  + (int)getBottomHeight { return 0; }
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
    NSView *contentView = dynamic_cast<LFL::OSXWindow*>(LFL::app->focused)->view.window.contentView;
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
  @property (nonatomic, retain) NSButton *toggleButton;
@end

@implementation OSXTableCellView 
@end

@interface OSXTableView : NSTableView
@end

@implementation OSXTableView
  - (void)drawGridInClipRect:(NSRect)clipRect {
    NSRect lastRowRect = [self rectOfRow:[self numberOfRows]-1];
    NSRect myClipRect = NSMakeRect(0, 0, lastRowRect.size.width, NSMaxY(lastRowRect));
    NSRect finalClipRect = NSIntersectionRect(clipRect, myClipRect);
    [super drawGridInClipRect:finalClipRect];
  }
@end

@interface OSXTable : NSViewController<NSTableViewDataSource, NSTableViewDelegate>
  @property (nonatomic, retain) OSXTableView *tableView;
  @property (nonatomic, copy)   NSScrollView *tableContainer;
  @property (nonatomic, assign) LFL::TableViewInterface *lfl_self;
  @property (nonatomic, assign) LFL::IntIntCB delete_row_cb;
  @property (nonatomic)         std::string style;
  @property (nonatomic)         int editable_section, editable_start_row, selected_section, selected_row,
                                    row_height;
@end

@implementation OSXTable
  {
    int data_rows;
    std::vector<LFL::TableSection<LFL::TableItem>> data;
  }

  - (id)init: (LFL::TableViewInterface*)lself withTitle:(const std::string&)title andStyle:(const std::string&)style items:(std::vector<LFL::TableSection<LFL::TableItem>>)item { 
    self = [super init];
    self.title = LFL::MakeNSString(title);
    _lfl_self = lself;
    _style = style;
    _editable_section = _editable_start_row = -1;
    _row_height = 30;
    data = move(item);

    int count = 0;
    NSMutableIndexSet *hide_indices = [[[NSMutableIndexSet alloc] init] autorelease];
    for (auto &i : data) {
      i.start_row = data_rows + count++;
      data_rows += i.item.size();
      [OSXTable addHiddenIndices:i to:hide_indices];
    }

    NSView *contentView = dynamic_cast<LFL::OSXWindow*>(LFL::app->focused)->view.window.contentView;
    _tableContainer = [[NSScrollView alloc] initWithFrame: contentView.frame];
    self.view = [_tableContainer autorelease];
    self.view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    _tableView = [[OSXTableView alloc] initWithFrame: contentView.frame];
    [_tableView setDelegate:self];
    [_tableView setDataSource:self];
    [_tableView setHeaderView:nil];
    [_tableView hideRowsAtIndexes:hide_indices withAnimation:NSTableViewAnimationEffectNone];
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

  - (void)addRow:(int)section withItem:(LFL::TableItem)item {
    if (section == data.size()) { data.emplace_back(); data.back().start_row = data_rows + data.size(); }
    CHECK_LT(section, data.size());
    data[section].item.emplace_back(move(item));
    data_rows++;
    // [self.tableView insertRowsAtIndexPaths:@[path] withRowAnimation:UITableViewRowAnimationNone];
    [_tableView reloadData];
  }

  - (void)replaceSection:(int)section items:(std::vector<LFL::TableItem>)item header:(LFL::TableItem)h flag:(int)f {
    bool added = section == data.size();
    if (added) data.emplace_back(data_rows + data.size());
    CHECK_LT(section, data.size());
    int item_size = item.size(), size_delta = item_size - data[section].item.size();
    NSMutableIndexSet *show_indices = [[[NSMutableIndexSet alloc] init] autorelease];
    [OSXTable addHiddenIndices:data[section] to:show_indices];
    [_tableView unhideRowsAtIndexes:show_indices withAnimation:NSTableViewAnimationEffectNone];

    data[section] = LFL::TableSection<LFL::TableItem>(move(h), f, data[section].start_row);
    data[section].item = move(item);
    data_rows += size_delta;
    if (size_delta) for (int i=section+1, e=data.size(); i < e; ++i) data[i].start_row += size_delta;

#if 0
    [_tableView reloadDataForRowIndexes: [NSIndexSet indexSetWithIndexesInRange:NSMakeRange(0, data[section].item.size())]
      columnIndexes: [NSIndexSet indexSetWithIndexesInRange:NSMakeRange(0, 1)]];
#else
    [_tableView reloadData];
#endif
    NSMutableIndexSet *hide_indices = [[[NSMutableIndexSet alloc] init] autorelease];
    [OSXTable addHiddenIndices:data[section] to:hide_indices];
    [_tableView hideRowsAtIndexes:hide_indices withAnimation:NSTableViewAnimationEffectNone];
  }

  - (void)checkExists:(int)section row:(int)r {
    if (section == data.size()) { data.emplace_back(); data.back().start_row = data_rows + data.size(); }
    CHECK_LT(section, data.size());
    CHECK_LT(r, data[section].item.size());
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
    NSIndexSet *index = [NSIndexSet indexSetWithIndex: data[section].start_row + r + 1];
    if ((data[section].item[r].hidden = v)) [_tableView   hideRowsAtIndexes:index withAnimation:NSTableViewAnimationEffectNone];
    else                                    [_tableView unhideRowsAtIndexes:index withAnimation:NSTableViewAnimationEffectNone];
  }

  - (void)setValue:(int)section row:(int)r val:(const std::string&)v {
    [self checkExists:section row:r];
    auto &ci = data[section].item[r];
    ci.val = v;
  }
  
  - (void)setColor:(int)section row:(int)r val:(const LFL::Color&)v {
    [self checkExists:section row:r];
    auto &ci = data[section].item[r];
    ci.SetFGColor(v);
  }

  - (void)setSectionValues:(int)section items:(const LFL::StringVec&)item {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_EQ(item.size(), data[section].item.size());
    for (int i=0, l=data[section].item.size(); i != l; ++i) [self setValue:section row:i val:item[i]];
    // [self.tableView reloadSections:[NSIndexSet indexSetWithIndex: section]
    //  withRowAnimation:UITableViewRowAnimationNone];
  }

  - (void)setSectionColors:(int)section items:(const LFL::vector<LFL::Color>&)item {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_EQ(item.size(), data[section].item.size());
    for (int i=0, l=data[section].item.size(); i != l; ++i) [self setColor:section row:i val:item[i]];
  }

  - (void)applyChangeList:(const LFL::TableSectionInterface::ChangeList&)changes {
    LFL::TableSection<LFL::TableItem>::ApplyChangeList(changes, &data, [=](const LFL::TableSectionInterface::Change &d){
    #if 0
      NSIndexPath *p = [NSIndexPath indexPathForRow:d.row inSection:d.section];
      [self.tableView reloadRowsAtIndexPaths:@[p] withRowAnimation:UITableViewRowAnimationNone];
    #endif
    });
    [_tableView reloadData];
  }

  - (LFL::PickerItem*)getPicker:(int)section row:(int)r { return nil; } 

  // - (void)loadView { self.view = _tableContainer; }
  - (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView { return data_rows + data.size(); }

  - (CGFloat)tableView:(NSTableView *)tableView heightOfRow:(NSInteger)collapsed_row {
    int section = -1, row = -1;
    LFL::TableSection<LFL::TableItem>::FindSectionOffset(data, collapsed_row, &section, &row);
    if (section < 0 || section >= data.size() || row < 0 || row >= data[section].item.size()) return _row_height;
    const auto &ci = data[section].item[row];
    if ((ci.flags & OSXTableItem::GUILoaded) &&
        (ci.type == LFL::TableItem::Picker || ci.type == LFL::TableItem::FontPicker)) return ci.height;
    else return _row_height; 
  }

#if 0
  - (id)tableView:(NSTableView *)tableView objectValueForTableColumn:(NSTableColumn *)tableColumn row:(NSInteger)row {
    return [NSString stringWithFormat:@"%ld", row];
  }
#endif

  - (NSView *)tableView:(NSTableView *)tableView viewForTableColumn:(NSTableColumn *)tableColumn row:(NSInteger)collapsed_row {
    if (!data.size()) return nil;
    NSString *identifier = [NSString stringWithFormat:@"row%ld", collapsed_row]; // [tableColumn identifier];
    OSXTableCellView *cellView = [tableView makeViewWithIdentifier:identifier owner:self];
    if (cellView == nil) {
      int section = -1, row = -1;
      LFL::TableSection<LFL::TableItem>::FindSectionOffset(data, collapsed_row, &section, &row);
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
      cellView.textField.stringValue = LFL::MakeNSString(compiled_item.key);
      [cellView addSubview:cellView.textField];

      bool textinput=0, numinput=0, pwinput=0, label=0;
      if ((textinput = type == LFL::TableItem::TextInput) || (numinput = type == LFL::TableItem::NumberInput)
          || (pwinput = type == LFL::TableItem::PasswordInput) || (label = type == LFL::TableItem::Label)) {
        cellView.textValue = [[[NSTextField alloc] initWithFrame:
          CGRectMake(tableView.frame.size.width - 5 - 100, 0, 100, _row_height)] autorelease];
        cellView.textValue.bordered = NO;
        cellView.textValue.drawsBackground = NO;
        cellView.textValue.editable = cellView.textValue.selectable = !label;
        cellView.textValue.alignment = NSRightTextAlignment;
        if (v->size() && ((*v)[0] == 1 || (*v)[0] == 2)) cellView.textValue.placeholderString = LFL::MakeNSString(v->substr(1));
        else if (v->size())                              cellView.textValue.stringValue       = LFL::MakeNSString(*v);
        [cellView addSubview:cellView.textValue];
      
      } else if (type == LFL::TableItem::Selector) {
      } else if (type == LFL::TableItem::Picker || type == LFL::TableItem::FontPicker) {
      } else if (type == LFL::TableItem::Button) {
      } else if (type == LFL::TableItem::Toggle) {
				cellView.toggleButton = [[NSButton alloc] initWithFrame:
					CGRectMake(tableView.frame.size.width - _row_height*2, 0, _row_height*2, _row_height)];
        [cellView.toggleButton setButtonType:NSToggleButton];
				cellView.toggleButton.alternateTitle = @"On";
				cellView.toggleButton.title = @"Off";
				cellView.toggleButton.target = self;
				cellView.toggleButton.action = @selector(toggleButtonClicked:);
        cellView.toggleButton.state = (*v == "1") ? NSOnState : NSOffState;
				[cellView addSubview: cellView.toggleButton];
				[cellView.toggleButton setNeedsDisplay:YES];
      }
    }
    return cellView;
  }

  - (OSXTableCellView *)headerForSection:(int)section {
    return [[[OSXTableCellView alloc] initWithFrame:CGRectMake(0, 0, 200, _row_height)] autorelease];
    return nil;
  }

  - (BOOL)tableView:(NSTableView *)tableView shouldSelectRow:(NSInteger)collapsed_row {
    int section = -1, row = -1;
    LFL::TableSection<LFL::TableItem>::FindSectionOffset(data, collapsed_row, &section, &row);
    if (row < 0) return NO;
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
        [self setHidden:section row:row val:next_compiled_item.hidden];
      }
    }
    return NO;
  }

  - (void)toggleButtonClicked:(id)sender { INFO("toggle flipped"); }

  - (LFL::StringPairVec)dumpDataForSection: (int)ind {
    LFL::StringPairVec ret;
    return ret;
  }

  + (void)addHiddenIndices:(const LFL::TableSection<LFL::TableItem>&)in to:(NSMutableIndexSet*)out {
    for (auto b = in.item.begin(), e = in.item.end(), j = b; j != e; ++j) 
      if (j->hidden) [out addIndex: in.start_row + (j - b) + 1];
  }
@end

@interface OSXText : NSViewController<NSTextViewDelegate>
  @property (nonatomic, retain) NSTextView *textView;
  @property (nonatomic, retain) NSString *text;
@end

@implementation OSXText
  - (id)initWithTitle:(NSString*)title andText:(NSString*)t {
    self = [super init];
    self.title = title;
    self.text = t;
    return self;
  }
@end

namespace LFL {
struct OSXToolbarView : public ToolbarViewInterface {
  string theme;
  ~OSXToolbarView() {}
  OSXToolbarView(MenuItemVec items) {}
  void Show(bool show_or_hide) {}
  void ToggleButton(const string &n) {}
  void SetTheme(const string &x) { theme=x; }
  string GetTheme() { return theme; }
};

struct OSXTableView : public TableViewInterface {
  OSXTable *table;
  ~OSXTableView() { [table release]; }
  OSXTableView(const string &title, const string &style, TableItemVec items) :
    table([[OSXTable alloc] init:this withTitle:title andStyle:style items:TableSection<TableItem>::Convert(move(items))]) {}

  void DelNavigationButton(int align) {}
  void AddNavigationButton(int align, const TableItem &item) {}
  void SetToolbar(ToolbarViewInterface *t) {}

  void Show(bool show_or_hide) {
    NSView *contentView = dynamic_cast<OSXWindow*>(app->focused)->view.window.contentView;
    if (show_or_hide) {
      [[contentView animator] addSubview:table.tableContainer];
      [table.tableContainer setFrameOrigin:CGPointMake(0, 0)];
    } else 
      [table.tableContainer removeFromSuperview];
  }

  string GetKey(int section, int row) { return [table getKey:section row:row]; }
  int GetTag(int section, int row) { return [table getTag:section row:row]; }
  string GetValue(int section, int row) { return ""; }
  PickerItem *GetPicker(int section, int row) { return [table getPicker:section row:row]; }
  StringPairVec GetSectionText(int section) { return [table dumpDataForSection:section]; }

  void BeginUpdates() { [table.tableView beginUpdates]; }
  void EndUpdates() { [table.tableView endUpdates]; }
  void AddRow(int section, TableItem item) { return [table addRow:section withItem:move(item)]; }
  void SelectRow(int section, int row) {} 
  void ReplaceRow(int section, int row, TableItem item) {}
  void ReplaceSection(int section, TableItem h, int flag, TableItemVec item)
  { [table replaceSection:section items:move(item) header:h flag:flag]; }

  void ApplyChangeList(const TableSectionInterface::ChangeList &changes) { [table applyChangeList:changes]; }
  void SetSectionValues(int section, const StringVec &item) { [table setSectionValues:section items:item]; }
  void SetSectionColors(int section, const vector<Color> &item) { [table setSectionColors:section items:item]; }
  void SetHeader(int section, TableItem header) {}
  void SetKey(int section, int row, const string &val) { [table setKey:section row:row val:val]; }
  void SetTag(int section, int row, int val) { [table setTag:section row:row val:val]; }
  void SetValue(int section, int row, const string &val) { [table setValue:section row:row val:val]; }
  void SetSelected(int section, int row, int selected) {}
  void SetHidden(int section, int row, int val) { [table setHidden:section row:row val:val]; }
  void SetColor(int section, int row, const Color &val) { [table setColor:section row:row val:val]; }
  void SetTitle(const string &title) { table.title = LFL::MakeNSString(title); }
  void SetTheme(const string &theme) {}
  void SetSectionEditable(int section, int start_row, int skip_last_rows, LFL::IntIntCB cb) {
    table.delete_row_cb = move(cb);
    table.editable_section = section;
    table.editable_start_row = start_row;
  }
};

struct OSXTextView : public TextViewInterface {
  OSXText *view;
  ~OSXTextView() { [view release]; }
  OSXTextView(const string &title, File *f) : OSXTextView(title, f ? f->Contents() : "") {}
  OSXTextView(const string &title, const string &text) :
    view([[OSXText alloc] initWithTitle:MakeNSString(title) andText:[[[NSString alloc]
         initWithBytes:text.data() length:text.size() encoding:NSASCIIStringEncoding] autorelease]]) {}
};

struct OSXNavigationView : public NavigationViewInterface {
  OSXNavigation *nav;
  ~OSXNavigationView() { [nav release]; }
  OSXNavigationView() : nav([[OSXNavigation alloc] init]) {}

  void Show(bool show_or_hide) {
    GameContainerView *contentView = dynamic_cast<OSXWindow*>(app->focused)->view.window.contentView;
    if ((shown = show_or_hide)) {
      [contentView.gameView removeFromSuperview];
      [[contentView animator] addSubview:nav.view];
      [nav.view setFrameOrigin:CGPointMake(0, 0)];
    } else 
      [nav.view removeFromSuperview];
  }

  TableViewInterface *Back() {
    for (NSViewController *c in [nav.viewControllers reverseObjectEnumerator]) {
      if ([c isKindOfClass:[OSXTable class]])
        if (auto lself = static_cast<OSXTable*>(c).lfl_self) return lself;
    } 
    return nullptr;
  }

  void PushTableView(TableViewInterface *t) {
    if (!root) root = t;
    if (t->show_cb) t->show_cb();
    [nav pushViewController: dynamic_cast<OSXTableView*>(t)->table animated: YES];
  }

  void PushTextView(TextViewInterface *t) {
    if (t->show_cb) t->show_cb();
    [nav pushViewController: dynamic_cast<OSXTextView*>(t)->view animated: YES];
  }

  void PopToRoot() {}
  void PopAll() {}

  void PopView(int n) {
    for (int i = 0; i != n; ++i)
      [nav popViewControllerAnimated: (i == n - 1)];
  }

  void SetTheme(const string &theme) {}
};

int Application::LoadSystemImage(const string &n) {
  NSImage *image = [[NSImage alloc] initWithContentsOfFile: MakeNSString(StrCat(app->assetdir, "../drawable-xhdpi/", n, ".png")) ];
  if (!image) return 0;
  return app_images.Insert(move(image)) + 1;
}

void Application::UpdateSystemImage(int n, Texture&) {
}

void Application::UnloadSystemImage(int n) {
  if (auto image = app_images[n-1]) { [image release]; app_images[n-1] = nullptr; }
  app_images.Erase(n-1);
}

unique_ptr<ToolbarViewInterface> SystemToolkit::CreateToolbar(const string &theme, MenuItemVec items, int flag) { return make_unique<OSXToolbarView>(move(items)); }
unique_ptr<TableViewInterface> SystemToolkit::CreateTableView(const string &title, const string &style, const string &theme, TableItemVec items) { return make_unique<OSXTableView>(title, style, move(items)); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(const string &title, File *file) { return make_unique<OSXTextView>(title, file); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(const string &title, const string &text) { return make_unique<OSXTextView>(title, text); }
unique_ptr<NavigationViewInterface> SystemToolkit::CreateNavigationView(const string &style, const string &theme) { return make_unique<OSXNavigationView>(); }

}; // namespace LFL
