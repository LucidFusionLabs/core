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
SystemAlertView::~SystemAlertView() {}
SystemAlertView::SystemAlertView(AlertItemVec items) {}
void SystemAlertView::Show(const string &arg) {}
void SystemAlertView::ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {}
string SystemAlertView::RunModal(const string &arg) { return string(); }

SystemMenuView::~SystemMenuView() {}
SystemMenuView::SystemMenuView(const string &title_text, MenuItemVec items) {}
unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(MenuItemVec items) { return nullptr; }
void SystemMenuView::Show() {}

SystemPanelView::~SystemPanelView() {}
SystemPanelView::SystemPanelView(const Box &b, const string &title, PanelItemVec items) {}
void SystemPanelView::Show() {}
void SystemPanelView::SetTitle(const string &title) {}

SystemToolbarView::~SystemToolbarView() {}
SystemToolbarView::SystemToolbarView(MenuItemVec items) : impl(0) {}
void SystemToolbarView::Show(bool show_or_hide) {}
void SystemToolbarView::ToggleButton(const string &n) {}

SystemTableView::~SystemTableView() {}
SystemTableView::SystemTableView(const string &title, const string &style, TableItemVec items, int second_col) :
  impl(new vector<Table>(Table::Convert(move(items)))) {}

StringPairVec SystemTableView::GetSectionText(int section) {
  StringPairVec ret;
  auto data = FromVoid<vector<Table>*>(impl);
  CHECK_RANGE(section, 0, data->size());
  for (auto &i : (*data)[section].item) ret.emplace_back(i.key, i.val);
  return ret;
}

void SystemTableView::SetSectionValues(int section, const StringVec &item) {
  auto data = FromVoid<vector<Table>*>(impl);
  if (section == data->size()) data->emplace_back();
  CHECK_LT(section, data->size());
  CHECK_EQ(item.size(), (*data)[section].item.size());
  for (int i=0, l=(*data)[section].item.size(); i != l; ++i) (*data)[section].item[i].val = item[i];
}

void SystemTableView::DelNavigationButton(int align) {}
void SystemTableView::AddNavigationButton(int align, const TableItem &item) {}
void SystemTableView::AddToolbar(SystemToolbarView *t) {}
void SystemTableView::Show(bool show_or_hide) {}
void SystemTableView::AddRow(int section, TableItem item) {}
string SystemTableView::GetKey(int section, int row) { return ""; }
int SystemTableView::GetTag(int section, int row) { return 0; }
void SystemTableView::SetTag(int section, int row, int val) {}
void SystemTableView::SetKey(int section, int row, const string &val) {}
void SystemTableView::SetValue(int section, int row, const string &val) {}
void SystemTableView::SetHidden(int section, int row, bool val) {}
void SystemTableView::SetTitle(const string &title) {}
PickerItem *SystemTableView::GetPicker(int section, int row) { return 0; }
void SystemTableView::SetEditableSection(int section, int start_row, LFL::IntIntCB cb) {}
void SystemTableView::SelectRow(int section, int row) {} 
void SystemTableView::BeginUpdates() {}
void SystemTableView::EndUpdates() {}
void SystemTableView::ReplaceSection(int section, const string &h, int image, int flag, TableItemVec item, Callback add_button) {}

SystemTextView::~SystemTextView() {}
SystemTextView::SystemTextView(const string &title, File *f) : SystemTextView(title, f ? f->Contents() : "") {}
SystemTextView::SystemTextView(const string &title, const string &text) : impl(0) {}

SystemNavigationView::~SystemNavigationView() {}
SystemNavigationView::SystemNavigationView() : impl(0) {}
void SystemNavigationView::Show(bool show_or_hide) {}
SystemTableView *SystemNavigationView::Back() { return nullptr; }
void SystemNavigationView::PushTableView(SystemTableView *t) {}
void SystemNavigationView::PushTextView(SystemTextView *t) {}
void SystemNavigationView::PopToRoot() {}
void SystemNavigationView::PopAll() {}
void SystemNavigationView::PopView(int n) {}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB&) {}
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB&) {}
void Application::ShowSystemContextMenu(const vector<MenuItem>&items) {}
int Application::LoadSystemImage(const string &n) { static int ret=0; return ++ret; }
void Application::UpdateSystemImage(int n, Texture&) {}

}; // namespace LFL
