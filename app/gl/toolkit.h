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
  MenuItemVec data;
  string theme;
  ToolbarView(Window *w, const string &theme, MenuItemVec items);

  void Layout();
  void Draw();

  void Show(bool show_or_hide);
  void ToggleButton(const string &n);
  void SetTheme(const string &theme);
  string GetTheme() { return theme; }
};

struct TableView : public View, public TableViewInterface {
  struct TableViewItem : public TableItem {
    Box val_box;
    unique_ptr<TextBox> textbox;
    unique_ptr<Widget::Slider> slider;
    TableViewItem(TableItem i) : TableItem(move(i)) {}
    TableViewItem() {}
  };
  typedef TableSection<TableViewItem> TableViewSection;

  vector<TableViewSection> data;
  string title, style, theme;
  Widget::Slider scrollbar;
  DrawableBoxArray *out=0;
  int row_height=0, decay_box_line=-1, decay_box_left=0;
  TableView(Window *w, const string &title, const string &style, const string &theme, TableItemVec items);

  void Layout();
  void Draw();
  void AppendFlow(Flow *flow);
  void OnClick(int, point, point, int);

  void DelNavigationButton(int id);
  void AddNavigationButton(int id, const TableItem &item);
  void SetToolbar(ToolbarViewInterface*);
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
  void ReplaceSection(int section, TableItem header, int flag, TableItemVec item);
  void ApplyChangeList(const TableSectionInterface::ChangeList&);
  void SetSectionValues(int section, const StringVec&);
  void SetSectionColors(int seciton, const vector<Color>&);
  void SetSectionEditable(int section, int start_row, int skip_last_rows, IntIntCB cb=IntIntCB());
  void SetHeader(int section, TableItem header);
  void SetKey(int secton, int row, const string &key);
  void SetTag(int section, int row, int val);
  void SetValue(int section, int row, const string &val);
  void SetSelected(int section, int row, int selected);
  void SetHidden(int section, int row, int val);
  void SetColor(int section, int row, const Color &val);
  void SetTitle(const string &title);
  void SetTheme(const string &theme);
};

struct NavigationView : public View, public NavigationViewInterface {
  string style, theme;
  NavigationView(Window *w, const string &style, const string &theme);

  void Layout();
  void Draw();

  TableViewInterface *Back();
  void Show(bool show_or_hide);
  void PushTableView(TableViewInterface*);
  void PushTextView(TextViewInterface*);
  void PopView(int num=1);
  void PopToRoot();
  void PopAll();
  void SetTheme(const string &theme);
};

struct Toolkit : public ToolkitInterface {
  unique_ptr<AlertViewInterface> CreateAlert(AlertItemVec items);
  unique_ptr<PanelViewInterface> CreatePanel(const Box&, const string &title, PanelItemVec);
  unique_ptr<ToolbarViewInterface> CreateToolbar(const string &theme, MenuItemVec items, int flag);
  unique_ptr<MenuViewInterface> CreateMenu(const string &title, MenuItemVec items);
  unique_ptr<MenuViewInterface> CreateEditMenu(MenuItemVec items);
  unique_ptr<TableViewInterface> CreateTableView
    (const string &title, const string &style, const string &theme, TableItemVec items);
  unique_ptr<TextViewInterface> CreateTextView(const string &title, File *file);
  unique_ptr<TextViewInterface> CreateTextView(const string &title, const string &text);
  unique_ptr<NavigationViewInterface> CreateNavigationView(const string &style, const string &theme);
};

}; // namespace LFL
#endif // LFL_CORE_APP_GL_TOOLKIT_H__
