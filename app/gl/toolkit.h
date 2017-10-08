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

#ifndef LFL_CORE_APP_GL_TOOLKIT_H__
#define LFL_CORE_APP_GL_TOOLKIT_H__
namespace LFL {

struct ToolbarView : public View, public ToolbarViewInterface {
  struct ToolbarViewItem : public MenuItem {
    bool down=false;
    unique_ptr<Widget::Button> button;
    ToolbarViewItem(MenuItem i) : MenuItem(move(i)) {}
    ToolbarViewItem() {}
  };

  vector<ToolbarViewItem> data;
  DrawableBoxArray *out=0;
  string theme;
  Font *font=0, *selected_font=0;
  Color *selected_outline=0;
  ToolbarView(Window *w, const string &theme, MenuItemVec items, Font *F=0, Font *SF=0, Color *SO=0);

  void Layout() override;
  void Draw() override;
  View *AppendFlow(Flow*) override;

  void Show(bool show_or_hide) override;
  void ToggleButton(const string &n) override;
  void SetTheme(const string &theme) override;
  string GetTheme() override { return theme; }
};

struct CollectionView : public View, public CollectionViewInterface {
  struct CollectionViewItem : public CollectionItem {
    Box val_box;
    unique_ptr<Widget::Button> button;
    CollectionViewItem(CollectionItem i) : CollectionItem(move(i)) {}
    CollectionViewItem() {}
  };
  typedef TableSection<CollectionViewItem> CollectionViewSection;

  vector<CollectionViewSection> data;
  ToolbarViewInterface *toolbar;
  string title, style, theme;
  Widget::Slider scrollbar;
  DrawableBoxArray *out=0;
  int row_height=0, decay_box_line=-1, decay_box_left=0, selected_section=-1, selected_row=-1, scrolled=0;
  CollectionView(Window *w, const string &title, const string &style, const string &theme, vector<CollectionItem> items);

  void Layout() override;
  void Draw() override;
  void OnClick(int, point, point, int);
  void CheckExists(int section, int row);
  View *AppendFlow(Flow*) override;
  void Show(bool show_or_hide) override;
  void SetToolbar(ToolbarViewInterface*) override;
};

struct TableView : public View, public TableViewInterface {
  struct TableViewItem : public TableItem {
    Box val_box;
    unique_ptr<TextBox> textbox;
    unique_ptr<Widget::Slider> slider;
    unique_ptr<Browser> browser;
    TableViewItem(TableItem i) : TableItem(move(i)) {}
    TableViewItem() {}
  };
  typedef TableSection<TableViewItem> TableViewSection;

  vector<TableViewSection> data;
  TableViewItem nav_left, nav_right;
  ToolbarViewInterface *toolbar=0;
  string title, style, theme;
  Widget::Slider scrollbar;
  DrawableBoxArray *out=0;
  int row_height=0, decay_box_line=-1, decay_box_left=0, selected_section=-1, selected_row=-1, scrolled=0;
  TableView(Window *w, const string &title, const string &style, const string &theme, TableItemVec items);

  void Layout() override;
  void Draw() override;
  void OnClick(int, point, point, int);
  void CheckExists(int section, int row);
  View *AppendFlow(Flow*) override;

  void DelNavigationButton(int id) override;
  void AddNavigationButton(int id, const TableItem &item) override;
  void SetToolbar(ToolbarViewInterface*) override;
  void Show(bool show_or_hide) override;

  string GetKey(int section, int row) override;
  string GetValue(int section, int row) override;
  int GetTag(int section, int row) override;
  PickerItem *GetPicker(int section, int row) override;
  StringPairVec GetSectionText(int section) override;

  void BeginUpdates() override;
  void EndUpdates() override;
  void AddRow(int section, TableItem item) override;
  void SelectRow(int section, int row) override;
  void ReplaceRow(int section, int row, TableItem item) override;
  void ReplaceSection(int section, TableItem header, int flag, TableItemVec item) override;
  void ApplyChangeList(const TableSectionInterface::ChangeList&) override;
  void SetSectionValues(int section, const StringVec&) override;
  void SetSectionColors(int seciton, const vector<Color>&) override;
  void SetSectionEditable(int section, int start_row, int skip_last_rows, IntIntCB cb=IntIntCB()) override;
  void SetHeader(int section, TableItem header) override;
  void SetKey(int secton, int row, const string &key) override;
  void SetTag(int section, int row, int val) override;
  void SetValue(int section, int row, const string &val) override;
  void SetSelected(int section, int row, int selected) override;
  void SetHidden(int section, int row, int val) override;
  void SetColor(int section, int row, const Color &val) override;
  void SetTitle(const string &title) override;
  void SetTheme(const string &theme) override;
};

struct NavigationView : public View, public NavigationViewInterface {
  string style, theme;
  vector<StackViewInterface*> stack;
  NavigationView(Window *w, const string &style, const string &theme);

  View *AppendFlow(Flow*) override;
  void Layout() override;
  void Draw() override;

  TableViewInterface *Back() override;
  void Show(bool show_or_hide) override;
  void PushTableView(TableViewInterface*) override;
  void PushTextView(TextViewInterface*) override;
  void PopView(int num=1) override;
  void PopToRoot() override;
  void PopAll() override;
  void SetTheme(const string &theme) override;
};

struct Toolkit : public ToolkitInterface {
  unique_ptr<AlertViewInterface> CreateAlert(Window*, AlertItemVec items) override;
  unique_ptr<PanelViewInterface> CreatePanel(Window*, const Box&, const string &title, PanelItemVec) override;
  unique_ptr<ToolbarViewInterface> CreateToolbar(Window*, const string &theme, MenuItemVec items, int flag) override;
  unique_ptr<MenuViewInterface> CreateMenu(Window*, const string &title, MenuItemVec items) override;
  unique_ptr<MenuViewInterface> CreateEditMenu(Window*, MenuItemVec items) override;
  unique_ptr<CollectionViewInterface> CreateCollectionView
    (Window*, const string &title, const string &style, const string &theme, vector<CollectionItem> items) override;
  unique_ptr<TableViewInterface> CreateTableView
    (Window*, const string &title, const string &style, const string &theme, TableItemVec items) override;
  unique_ptr<TextViewInterface> CreateTextView(Window*, const string &title, File *file) override;
  unique_ptr<TextViewInterface> CreateTextView(Window*, const string &title, const string &text) override;
  unique_ptr<NavigationViewInterface> CreateNavigationView(Window*, const string &style, const string &theme) override;
};

}; // namespace LFL
#endif // LFL_CORE_APP_GL_TOOLKIT_H__
