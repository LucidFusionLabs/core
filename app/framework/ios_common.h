/*
 * $Id: ios_common.h 1336 2014-12-08 09:29:59Z justin $
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

@interface MyTouchView : UIView {}
@end

@interface LFViewController : UIViewController<UIActionSheetDelegate> {}
  @property (nonatomic, retain) UIToolbar *input_accessory_toolbar;
  @property BOOL ready, showing_keyboard, pinch_occurring;
  - (CGRect)getKeyboardFrame;
  - (CGRect)getKeyboardToolbarFrame;
  - (void)initNotifications;
  - (void)shutdownNotifications;
  - (void)shutdownGestureRecognizers;
@end

@interface LFUIWindow : UIWindow {}
@end

@interface LFGLKView : GLKView<UIKeyInput> {}
  @property (nonatomic, strong) UIView *inputAccessoryView;
  @property (nonatomic, assign) LFL::Callback copy_cb;
  @property BOOL resign_textfield_on_return, frame_on_keyboard_input;
@end

@interface LFGLKViewController : GLKViewController<GLKViewControllerDelegate, GLKViewDelegate> {}
  @property BOOL needs_gl_reset;
@end

@interface LFUIApplication : NSObject<UIApplicationDelegate, ObjcWindow> {}
  @property (nonatomic, retain) LFUIWindow *window;
  @property (nonatomic, retain) LFViewController *controller;
  @property (nonatomic, retain) LFGLKViewController *glk_controller;
  @property (nonatomic, retain) LFGLKView *glk_view;
  @property (nonatomic, retain) UIView *lview, *rview;
  @property (nonatomic, retain) UINavigationBar *title_bar;
  @property (nonatomic, retain) NSMutableDictionary *main_wait_fh;
  @property (nonatomic, assign) UIViewController *root_controller, *top_controller;
  @property BOOL overlay_top_controller, frame_disabled, frame_on_mouse_input, downscale, show_title;
  @property int screen_y, screen_width, screen_height;
  @property CGFloat scale;
  + (LFUIApplication *) sharedAppDelegate;
  + (CGSize)currentWindowSize;
  + (CGSize)sizeInOrientation:(UIInterfaceOrientation)orientation;
  - (CGFloat)getScale;
  - (CGRect)getFrame;
  - (bool)isKeyboardFirstResponder;
  - (void)hideKeyboard;
  - (void)showKeyboard;
  - (UIViewController*)findTopViewController;
  + (UIViewController*)findTopViewControllerWithRootViewController:(UIViewController*)rootViewController;
@end

@interface IOSToolbar : NSObject
  @property (nonatomic, retain) UIToolbar *toolbar, *toolbar2;
  + (int)getBottomHeight;
@end

@interface IOSButton : UIButton 
  @property (nonatomic, assign) LFL::Callback cb;
  - (void)buttonClicked:(IOSButton*)sender;
@end

@interface IOSBarButtonItem : UIBarButtonItem
  @property (nonatomic, assign) LFL::Callback cb;
  - (IBAction)buttonClicked:(IOSBarButtonItem*)sender;
@end

@interface IOSAlert : NSObject
  @property (nonatomic, retain) UIAlertController *alert;
  @property (nonatomic, retain) UIView *flash;
  @property (nonatomic, retain) UILabel *flash_text;
  @property (nonatomic)         std::string style;
  @property (nonatomic)         bool add_text, done;
  @property (nonatomic, assign) LFL::StringCB cancel_cb, confirm_cb;
@end

@interface IOSTable : UIViewController<UITableViewDataSource, UITableViewDelegate>
  @property (nonatomic, retain) UITableView *tableView;
  @property (nonatomic, retain) UIView *header;
  @property (nonatomic, retain) UILabel *header_label;
  @property (nonatomic, retain) UIToolbar *toolbar;
  @property (nonatomic, retain) UIColor *orig_bg_color, *orig_separator_color;
  @property (nonatomic, assign) LFL::TableViewInterface *lfl_self;
  @property (nonatomic, assign) std::string style;
  @property (nonatomic, assign) int selected_section, selected_row;
  @property (nonatomic, assign) bool needs_reload;
  - (id)initWithStyle: (UITableViewStyle)style;
@end

@interface IOSNavigation : UINavigationController<UINavigationControllerDelegate>
@end

namespace LFL {
struct iOSWindow : public Window {
  GLKView *glkview=0;
  ~iOSWindow() { ClearChildren(); }
  void SetCaption(const string &c);
  void SetResizeIncrements(float x, float y);
  void SetTransparency(float v);
  bool Reshape(int w, int h);
};

struct iOSTableView : public TableViewInterface {
  IOSTable *table;
  ~iOSTableView();
  iOSTableView(const string &title, const string &style, const string &theme, TableItemVec items);

  void DelNavigationButton(int align);
  void AddNavigationButton(int align, const TableItem &item);
  void SetToolbar(ToolbarViewInterface *t);
  void Show(bool show_or_hide);

  string GetKey(int section, int row);
  string GetValue(int section, int row);
  int GetTag(int section, int row);
  PickerItem *GetPicker(int section, int row);
  StringPairVec GetSectionText(int section);

  void BeginUpdates();
  void EndUpdates();
  void AddRow(int section, TableItem item);
  void SelectRow(int section, int row);
  void ReplaceRow(int section, int row, TableItem item);
  void ReplaceSection(int section, TableItem h, int flag, TableItemVec item);
  void ApplyChangeList(const TableSection::ChangeList &changes);
  void SetSectionValues(int section, const StringVec &item);
  void SetSectionColors(int section, const vector<Color> &item);
  void SetSectionEditable(int section, int start_row, int skip_last_rows, LFL::IntIntCB cb);
  void SetHeader(int section, TableItem h);
  void SetKey(int section, int row, const string &val);
  void SetTag(int section, int row, int val);
  void SetValue(int section, int row, const string &val);
  void SetSelected(int section, int row, int selected);
  void SetHidden(int section, int row, bool val);
  void SetColor(int section, int row, const Color&);
  void SetTitle(const string &title);
  void SetTheme(const string &title);
};

UIAlertView            *GetUIAlertView(AlertViewInterface*);
UIActionSheet          *GetUIActionSheet(MenuViewInterface*);
UIToolbar              *GetUIToolbar(ToolbarViewInterface*);
UITableViewController  *GetUITableViewController(TableViewInterface*);
UINavigationController *GetUINavigationController(NavigationViewInterface*);
}; // namespace LFL
