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

#include <QtOpenGL>
#include <QApplication>
#include <QInputDialog>
#include <QMessageBox>
#include <QTableView>
#include <QTableWidget>
#include "qt_common.h"

namespace LFL {
struct QtAlert {
  string style;
  bool add_text = 0;
  StringCB cancel_cb, confirm_cb;
  unique_ptr<QInputDialog> alert;
  unique_ptr<QMessageBox> msg;

  QtAlert(AlertItemVec kv) {
    CHECK_EQ(4, kv.size());
    CHECK_EQ("style", kv[0].first);
    cancel_cb  = move(kv[2].cb);
    confirm_cb = move(kv[3].cb);
    style      = move(kv[0].second);
    if ((add_text = (style == "textinput" || style == "pwinput"))) {
      alert = make_unique<QInputDialog>();
      alert->setWindowTitle(MakeQString(kv[1].first));
      alert->setLabelText(MakeQString(kv[1].second));
      alert->setCancelButtonText(MakeQString(kv[2].first));
      alert->setOkButtonText(MakeQString(kv[3].first));
      alert->setModal(false);
    } else {
      msg = make_unique<QMessageBox>();
      msg->setAttribute(Qt::WA_DeleteOnClose);
      msg->setText(MakeQString(kv[1].first));
      msg->setInformativeText(MakeQString(kv[1].second));
      msg->setModal(false);
    }
  }

  void Update(string t, string m, StringCB cb) { 
    if (add_text) {
      alert->setWindowTitle(MakeQString(t));
      alert->setLabelText(MakeQString(m));
    } else { 
    }
    confirm_cb = move(cb);
  }
};

struct QtTable {
  unique_ptr<QTableView> table;
  unique_ptr<QStandardItemModel> model;
  LFL::SystemTableView *lfl_self;
  LFL::IntIntCB delete_row_cb;
  std::string style;
  bool modal_nav=0;
  int editable_section=-1, editable_start_row=-1, selected_section=0, selected_row=0, second_col=0, row_height=30, data_rows=0;
  vector<Table> data;
  vector<unique_ptr<QtTable>> dropdowns;

  QtTable(SystemTableView *lself, const string &title, string s, vector<Table> item) :
    table(make_unique<QTableView>()), model(make_unique<QStandardItemModel>()),
    lfl_self(lself), style(move(s)), data(move(item)) {
    modal_nav = (style == "modal" || style == "dropdown");
    table->setWindowTitle(MakeQString(title));
    table->setModel(model.get());
  }
};

SystemAlertView::~SystemAlertView() { if (auto alert = FromVoid<QtAlert*>(impl)) delete alert; }
SystemAlertView::SystemAlertView(AlertItemVec items) : impl(new QtAlert(move(items))) {}
void SystemAlertView::ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb)
{ FromVoid<QtAlert*>(impl)->Update(title, msg, move(confirm_cb)); Show(arg); }
void SystemAlertView::Show(const string &arg) { RunModal(arg); }
string SystemAlertView::RunModal(const string &arg) {
  app->ReleaseMouseFocus();
  auto alert = FromVoid<QtAlert*>(impl);
  if (alert->add_text) {
    alert->alert->setTextValue(MakeQString(arg));
    alert->alert->open();
  } else {
    alert->msg->open();
  }
  return string(); 
}

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

SystemTableView::~SystemTableView() { if (auto table = FromVoid<QtTable*>(impl)) delete table; }
SystemTableView::SystemTableView(const string &title, const string &style, TableItemVec items, int second_col) :
  impl(new QtTable(this, title, style, Table::Convert(move(items)))) {}

StringPairVec SystemTableView::GetSectionText(int section) {
  StringPairVec ret;
  return ret;
}

void SystemTableView::SetSectionValues(int section, const StringVec &item) {
}

void SystemTableView::DelNavigationButton(int align) {}
void SystemTableView::AddNavigationButton(int align, const TableItem &item) {}
void SystemTableView::AddToolbar(SystemToolbarView *t) {}

void SystemTableView::Show(bool show_or_hide) {
  auto w = GetTyped<QtWindowInterface*>(app->focused->id);
  auto table = FromVoid<QtTable*>(impl);
  if (show_or_hide) {
    w->layout->addWidget(table->table.get());
    w->layout->setCurrentWidget(table->table.get());
  }
  //if (show_or_hide) table->table->show();
  //else              table->table->hide();
}

void SystemTableView::AddRow(int section, TableItem item) {}
string SystemTableView::GetKey(int section, int row) { return ""; }
int SystemTableView::GetTag(int section, int row) { return 0; }
void SystemTableView::SetTag(int section, int row, int val) {}
void SystemTableView::SetValue(int section, int row, const string &val) {}
void SystemTableView::SetHidden(int section, int row, bool val) {}
void SystemTableView::SetTitle(const string &title) {}
PickerItem *SystemTableView::GetPicker(int section, int row) { return 0; }
void SystemTableView::SetEditableSection(int section, int start_row, LFL::IntIntCB cb) {}
void SystemTableView::SelectRow(int section, int row) {} 
void SystemTableView::BeginUpdates() {}
void SystemTableView::EndUpdates() {}
void SystemTableView::SetDropdown(int section, int row, int val) {}
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

}; // namespace LFL
