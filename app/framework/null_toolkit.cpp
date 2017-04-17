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

namespace LFL {
struct NullAlertView : public AlertViewInterface {
  void Hide() {}
  void Show(const string &arg) {}
  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {}
  string RunModal(const string &arg) { return string(); }
};

struct NullMenuView : public MenuViewInterface {
  void Show() {}
};

struct NullPanelView : public PanelViewInterface {
  void Show() {}
  void SetTitle(const string &title) {}
};

struct NullToolbarView : public ToolbarViewInterface {
  string theme;
  void Show(bool show_or_hide) {}
  void ToggleButton(const string &n) {}
  void SetTheme(const string &x) { theme=x; }
  string GetTheme() { return theme; }
};

struct NullTableView : public TableViewInterface {
  vector<TableSection> data;
  NullTableView(const string &title, const string &style, TableItemVec items) :
    data(TableSection::Convert(move(items))) {}

  void DelNavigationButton(int align) {}
  void AddNavigationButton(int align, const TableItem &item) {}
  void SetToolbar(ToolbarViewInterface *t) {}
  void Show(bool show_or_hide) {}

  string GetKey(int section, int row) { return ""; }
  string GetValue(int section, int row) { return ""; }
  int GetTag(int section, int row) { return 0; }
  PickerItem *GetPicker(int section, int row) { return 0; }

  StringPairVec GetSectionText(int section) {
    StringPairVec ret;
    CHECK_RANGE(section, 0, data.size());
    for (auto &i : data[section].item) ret.emplace_back(i.key, i.val);
    return ret;
  }

  void BeginUpdates() {}
  void EndUpdates() {}
  void AddRow(int section, TableItem item) {}
  void SelectRow(int section, int row) {} 
  void ReplaceRow(int section, int row, TableItem item) {}
  void ReplaceSection(int section, TableItem h, int flag, TableItemVec item) {}
  void ApplyChangeList(const TableSection::ChangeList&) {}
  void SetSectionValues(int section, const StringVec &item) {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_EQ(item.size(), data[section].item.size());
    for (int i=0, l=data[section].item.size(); i != l; ++i) data[section].item[i].val = item[i];
  }

  void SetSectionColors(int section, const vector<Color> &item) {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    CHECK_EQ(item.size(), data[section].item.size());
    for (int i=0, l=data[section].item.size(); i != l; ++i) data[section].item[i].SetFGColor(item[i]);
  }

  void SetHeader(int section, TableItem) {}
  void SetKey(int section, int row, const string &val) {}
  void SetTag(int section, int row, int val) {}
  void SetValue(int section, int row, const string &val) {}
  void SetSelected(int section, int row, int selected) {}
  void SetHidden(int section, int row, bool val) {}
  void SetColor(int section, int row, const Color &val) {}
  void SetTitle(const string &title) {}
  void SetTheme(const string &theme) {}
  void SetSectionEditable(int section, int start_row, int skip_last_rows, LFL::IntIntCB cb) {}
};

struct NullTextView : public TextViewInterface {
};

struct NullNavigationView : public NavigationViewInterface {
  void Show(bool show_or_hide) {}
  TableViewInterface *Back() { return nullptr; }
  void PushTableView(TableViewInterface *t) {}
  void PushTextView(TextViewInterface *t) {}
  void PopToRoot() {}
  void PopAll() {}
  void PopView(int n) {}
  void SetTheme(const string &theme) {}
};

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB&) {}
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB&) {}
void Application::ShowSystemContextMenu(const vector<MenuItem>&items) {}
int Application::LoadSystemImage(const string &n) { static int ret=0; return ++ret; }
void Application::UpdateSystemImage(int n, Texture&) {}

unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(AlertItemVec items) { return make_unique<NullAlertView>(); }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<ToolbarViewInterface> SystemToolkit::CreateToolbar(const string &theme, MenuItemVec items) { return make_unique<NullToolbarView>(); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(const string &title, MenuItemVec items) { return make_unique<NullMenuView>(); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(vector<MenuItem> items) { return nullptr; }
unique_ptr<TableViewInterface> SystemToolkit::CreateTableView(const string &title, const string &style, const string &theme, TableItemVec items) { return make_unique<NullTableView>(title, style, move(items)); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(const string &title, File *file) { return make_unique<NullTextView>(); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(const string &title, const string &text) { return make_unique<NullTextView>(); }
unique_ptr<NavigationViewInterface> SystemToolkit::CreateNavigationView(const string &style, const string &theme) { return make_unique<NullNavigationView>(); }

}; // namespace LFL
