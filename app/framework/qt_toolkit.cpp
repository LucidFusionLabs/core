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
static vector<unique_ptr<QIcon>> app_images;

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

class QtDrawBorderDelegate : public QStyledItemDelegate {
  public:
  QtDrawBorderDelegate(QObject *parent=0) : QStyledItemDelegate(parent) {}
  void paint(QPainter* painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override {
    const QRect rect(option.rect);
    painter->setPen(Qt::gray);
    int col = index.column(), cols = index.model()->columnCount();
    if (col == 0)         painter->drawLine(rect.topLeft(),    rect.bottomLeft());
    if (col == cols - 1)  painter->drawLine(rect.topRight(),   rect.bottomRight());
    if (index.row() == 0) painter->drawLine(rect.topLeft(),    rect.topRight());
    if (1)                painter->drawLine(rect.bottomLeft(), rect.bottomRight());
    QStyledItemDelegate::paint(painter, option, index);
  }
}; // DrawBorderDelegate

class QtTableModel : public QStandardItemModel {
  public:
  vector<Table> *v;
  QtTableModel(vector<Table> *V) : v(V) {}
  QVariant data(const QModelIndex &index, int role) const override {
    if (role == Qt::ForegroundRole && index.column() == 1) {
      int section = -1, row = -1;
      Table::FindSectionOffset(*v, index.row(), &section, &row);
      if (section >= 0 && row < (*v)[section].item.size() && (*v)[section].item[row].right_text.size())
        return QColor(0, 122, 255);
    }
    return QStandardItemModel::data(index, role);
  }
};

struct QtTable {
  vector<Table> data;
  unique_ptr<QTableView> table;
  unique_ptr<QtTableModel> model;
  SystemTableView *lfl_self;
  IntIntCB delete_row_cb;
  string style;
  bool modal_nav=0;
  int editable_section=-1, editable_start_row=-1, selected_section=0, selected_row=0, second_col=0, row_height=30, data_rows=0;
  vector<unique_ptr<QtTable>> dropdowns;
  QtDrawBorderDelegate drawborder;

  QtTable(SystemTableView *lself, const string &title, string s, vector<Table> item) :
    table(make_unique<QTableView>()), model(make_unique<QtTableModel>(&data)),
    lfl_self(lself), style(move(s)), data(move(item)) {
    modal_nav = (style == "modal" || style == "dropdown");

    vector<int> hide_indices;
    for (auto sb = data.begin(), se = data.end(), s = sb; s != se; ++s) {
      s->start_row = data_rows + (s - sb);
      data_rows += s->item.size();
      model->appendRow(MakeRow(*s));
      for (auto rb = s->item.begin(), re = s->item.end(), r = rb; r != re; ++r) {
        if (r->hidden) hide_indices.push_back(s->start_row + (r - rb));
        model->appendRow(MakeRow(&(*r)));
      }
    }

    table->setShowGrid(false);
    table->setWindowTitle(MakeQString(title));
    table->setItemDelegate(&drawborder);
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->horizontalHeader()->hide();
    table->verticalHeader()->hide();
    table->setModel(model.get());
    QObject::connect(table.get(), &QTableView::clicked, bind(&QtTable::HandleCellClicked, this, _1));
    HideIndices(hide_indices, true);
  }

  QList<QStandardItem*> MakeRow(const Table &s) {
    auto key = make_unique<QStandardItem>(MakeQString(s.header));
    auto val = make_unique<QStandardItem>();
    key->setTextAlignment(Qt::AlignRight | Qt::AlignBottom);
    key->setFlags(Qt::ItemIsEnabled);
    val->setFlags(Qt::ItemIsEnabled);
    return QList<QStandardItem*>{ key.release(), val.release() };
  }

  void LoadCellItem(TableItem *item, const string **ok, int *ot, const std::string **ov) {
    if (!item->loaded) {
      item->loaded = true;
      if (item->type == LFL::TableItem::DropdownKey || item->type == LFL::TableItem::DropdownValue ||
          item->type == LFL::TableItem::FixedDropdown) {
        item->ref = dropdowns.size();
        auto dropdown_table = make_unique<QtTable>(nullptr, item->key, "dropdown", Table::Convert(item->MoveChildren()));
        // if (item->depends.size()) dropdown_table.changed = [self makeChangedCB: *item];
        // dropdown_table.completed = ^{ [self reloadRowAtIndexPath:path withRowAnimation:UITableViewRowAnimationNone]; };
        dropdowns.emplace_back(move(dropdown_table));
      }
    }

    const LFL::TableItem *ret;
    bool dropdown_value = item->type == LFL::TableItem::DropdownValue, parent_dropdown = style == "dropdown";
    if (dropdown_value || item->type == LFL::TableItem::DropdownKey || item->type == LFL::TableItem::FixedDropdown) {
      CHECK_RANGE(item->ref, 0, dropdowns.size());
      auto dropdown_table = dropdowns[item->ref].get();
      CHECK_EQ(0, dropdown_table->selected_section);
      CHECK_RANGE(dropdown_table->selected_row, 0, dropdown_table->data[0].item.size());
      ret = &dropdown_table->data[0].item[dropdown_table->selected_row];
    } else ret = item;

    *ok = dropdown_value  ? &item->key                         : &ret->key;
    *ov = parent_dropdown ? LFL::Singleton<std::string>::Get() : &ret->val;
    *ot = parent_dropdown ? 0 : ret->type;
  }

  QList<QStandardItem*> MakeRow(TableItem *item) {
    int type;
    const string *ok, *ov;
    LoadCellItem(item, &ok, &type, &ov);
    auto key = make_unique<QStandardItem>
      (item->left_icon ? *app_images[item->left_icon-1] : QIcon(), MakeQString(*ok));
    auto val = make_unique<QStandardItem>(MakeQString(item->right_text.size() ? item->right_text : *ov));
    key->setFlags(Qt::ItemIsEnabled);
    if (!(type == TableItem::TextInput || type == TableItem::NumberInput
          || type == TableItem::PasswordInput)) val->setFlags(Qt::ItemIsEnabled); 
    val->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
    if (type == TableItem::Toggle) {
      val->setText("");
      val->setCheckable(true);
      val->setCheckState(Qt::Unchecked);
    }
    return QList<QStandardItem*>{ key.release(), val.release() };
  }

  int GetCollapsedRowId(int section, int row) const { return data[section].start_row + 1 + row; }

  void AddRow(int section, TableItem item) {
    if (section == data.size()) data.emplace_back();
    CHECK_LT(section, data.size());
    data[section].item.emplace_back(move(item));
  }

  void SetValue(int section, int row, const string &v) {
    CheckExists(section, row);
    auto &ci = data[section].item[row];
    ci.val = v;
    auto val = model->item(GetCollapsedRowId(section, row), 1);
    if ((ci.type == TableItem::DropdownKey || ci.type == TableItem::DropdownValue ||
        ci.type == TableItem::FixedDropdown) && ci.loaded) {
      vector<string> vv = Split(v, ',');
      CHECK_RANGE(ci.ref, 0, dropdowns.size());
      auto dropdown_table = dropdowns[ci.ref].get();
      CHECK_EQ(dropdown_table->data[0].item.size(), vv.size()-1) << ": " << v;
      for (int j=1; j < vv.size(); ++j) dropdown_table->SetValue(0, j-1, vv[j]);
    } else if (ci.type == TableItem::Toggle) val->setCheckState(v == "1" ? Qt::Checked : Qt::Unchecked);
    else                                     val->setText(MakeQString(v));
  }

  void SetDropdown() {
  }

  void SetHidden(int section, int row, bool v) {
    CheckExists(section, row);
    table->setRowHidden(GetCollapsedRowId(section, row), v);
  }

  void CheckExists(int section, int r) {
    if (section == data.size()) { data.emplace_back(); data.back().start_row = data_rows + data.size(); }
    CHECK_LT(section, data.size());
    CHECK_LT(r, data[section].item.size());
  }

  void HideIndices(const vector<int> &ind, bool v) { for (auto &i : ind) table->setRowHidden(i, v); }
  void HideHiddenRows(const Table &s, bool v) {
    for (auto rb = s.item.begin(), re = s.item.end(), r = rb; r != re; ++r)
      if (r->hidden) table->setRowHidden(s.start_row + (r - rb), v);
  }

  void HandleCellClicked(const QModelIndex &index) {
    int section = -1, row = -1;
    LFL::Table::FindSectionOffset(data, index.row(), &section, &row);
    if (row < 0) return;
    CheckExists(section, row);
    selected_section = section;
    selected_row = row;

    auto &compiled_item = data[section].item[row];
    if (compiled_item.type == LFL::TableItem::Command || compiled_item.type == LFL::TableItem::Button) {
      compiled_item.cb();
    } else if (compiled_item.type == LFL::TableItem::Label && row + 1 < data[section].item.size()) {
      auto &next_compiled_item = data[section].item[row+1];
      if (next_compiled_item.type == LFL::TableItem::Picker ||
          next_compiled_item.type == LFL::TableItem::FontPicker) {
        next_compiled_item.hidden = !next_compiled_item.hidden;
      }
    }
  }
};

struct QtNavigation {
  unique_ptr<QWidget> header_widget, content_widget;
  unique_ptr<QLabel> header_label;
  unique_ptr<QPushButton> header_back, header_forward;
  unique_ptr<QHBoxLayout> header_layout;
  unique_ptr<QStackedLayout> content_layout;

  QtNavigation() : header_widget(make_unique<QWidget>()), content_widget(make_unique<QWidget>()),
  header_label(make_unique<QLabel>()), header_back(make_unique<QPushButton>()), header_forward(make_unique<QPushButton>()),
  header_layout(make_unique<QHBoxLayout>()), content_layout(make_unique<QStackedLayout>()) {
    QObject::connect(header_back.get(), &QPushButton::clicked, bind(&QtNavigation::HandleBackClicked, this));
    QSizePolicy sp_retain = header_back->sizePolicy();
    sp_retain.setRetainSizeWhenHidden(true);
    header_back->setSizePolicy(sp_retain);
    header_forward->setSizePolicy(sp_retain);
    header_forward->setHidden(true);
    header_layout->addWidget(header_back.get());
    header_layout->addStretch();
    header_layout->addWidget(header_label.get());
    header_layout->addStretch();
    header_layout->addWidget(header_forward.get());
    header_widget->setLayout(header_layout.get());
    content_widget->setLayout(content_layout.get());
    content_layout->setMenuBar(header_widget.get());
  }
  
  QWidget *GetWidget(int ind) { return content_layout->itemAt(ind)->widget(); }
  QWidget *GetBackWidget() { int count = content_layout->count(); return count ? GetWidget(count-1) : nullptr; }

  void UpdateBackButton() {
    int count = content_layout->count(); 
    if (count < 2) { header_back->setHidden(true); return; }
    auto widget = GetWidget(count - 2);
    header_back->setText(widget ? widget->windowTitle() : "");
    header_forward->setText(widget ? widget->windowTitle() : "");
    header_back->setHidden(false);
  }

  void HandleBackClicked() {
    auto w = GetBackWidget();
    if (!w) return;
    content_layout->removeWidget(w);
    if ((w = GetBackWidget())) header_label->setText(w->windowTitle());
    UpdateBackButton();
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

StringPairVec SystemTableView::GetSectionText(int ind) {
  StringPairVec ret;
  auto table = FromVoid<QtTable*>(impl);
  CHECK_LT(ind, table->data.size());

  for (int start_row=table->data[ind].start_row, l=table->data[ind].item.size(), type, i=0; i != l; i++) {
    const string *k, *v;
    auto &compiled_item = table->data[ind].item[i];
    table->LoadCellItem(&compiled_item, &k, &type, &v);

    if (compiled_item.type == TableItem::DropdownKey || compiled_item.type == TableItem::DropdownValue ||
        compiled_item.type == TableItem::FixedDropdown) {
      CHECK_RANGE(compiled_item.ref, 0, table->dropdowns.size());
      auto dropdown_table = table->dropdowns[compiled_item.ref].get();
      CHECK_EQ(0, dropdown_table->selected_section);
      CHECK_LT(dropdown_table->selected_row, dropdown_table->data[0].item.size());
      ret.emplace_back(GetQString(dropdown_table->table->windowTitle()),
                       dropdown_table->data[0].item[dropdown_table->selected_row].key);
    }

    string val;
    if (type == TableItem::Toggle) {
      val = table->model->itemFromIndex(table->model->index(start_row+i+1, 1))->checkState()
        == Qt::Checked ? "1" : "";
    } else val = GetQString(table->model->index(start_row+i+1, 1).data().toString());

    ret.emplace_back(*k, val);
  }
  return ret;
}

void SystemTableView::SetSectionValues(int section, const StringVec &item) {
  auto table = FromVoid<QtTable*>(impl);
  if (section == table->data.size()) table->data.emplace_back();
  CHECK_LT(section, table->data.size());
  CHECK_EQ(item.size(), table->data[section].item.size());
  for (int i=0, l=table->data[section].item.size(); i != l; ++i) table->SetValue(section, i, item[i]);
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
  } else {
    w->layout->setCurrentWidget(w->container);
    w->layout->removeWidget(table->table.get());
  }
}

void SystemTableView::AddRow(int section, TableItem item) { FromVoid<QtTable*>(impl)->AddRow(section, move(item)); }
string SystemTableView::GetKey(int section, int row) { auto table = FromVoid<QtTable*>(impl); table->CheckExists(section, row); return table->data[section].item[row].key; }
int SystemTableView::GetTag(int section, int row) { auto table = FromVoid<QtTable*>(impl); table->CheckExists(section, row); return table->data[section].item[row].tag; }
void SystemTableView::SetTag(int section, int row, int val) { auto table = FromVoid<QtTable*>(impl); table->CheckExists(section, row); table->data[section].item[row].tag = val; }
void SystemTableView::SetValue(int section, int row, const string &val) { FromVoid<QtTable*>(impl)->SetValue(section, row, val); }
void SystemTableView::SetHidden(int section, int row, bool val) { FromVoid<QtTable*>(impl)->SetHidden(section, row, val); }
void SystemTableView::SetTitle(const string &title) { FromVoid<QtTable*>(impl)->table->setWindowTitle(MakeQString(title)); }

PickerItem *SystemTableView::GetPicker(int section, int row) { return 0; }
void SystemTableView::SetEditableSection(int section, int start_row, LFL::IntIntCB cb) {}
void SystemTableView::SelectRow(int section, int row) {} 
void SystemTableView::BeginUpdates() {}
void SystemTableView::EndUpdates() {}
void SystemTableView::SetDropdown(int section, int row, int val) {}

void SystemTableView::ReplaceSection(int section, const string &h, int image, int flag, TableItemVec item, Callback add_button) {
  auto t = FromVoid<QtTable*>(impl);
  bool added = section == t->data.size();
  if (added) t->data.emplace_back(t->data_rows + t->data.size());
  CHECK_LT(section, t->data.size());
  int old_item_size = t->data[section].item.size(), item_size = item.size();
  int size_delta = item_size - old_item_size, section_start_row = t->data[section].start_row;
  t->HideHiddenRows(t->data[section], false);

  t->data[section] = LFL::Table(h, image, flag, move(add_button), section_start_row);
  t->data[section].item = move(item);
  t->data_rows += size_delta;
  for (int i=0; i < item_size; ++i) {
    auto row = t->MakeRow(&t->data[section].item[i]);
    if (i < old_item_size) {
      t->model->setItem(section_start_row + 1 + i, 0, row[0]);
      t->model->setItem(section_start_row + 1 + i, 1, row[1]);
    } else t->model->insertRow(section_start_row + 1 + i, row);
  }
  if (size_delta < 0) t->model->removeRows(section_start_row + 1 + item_size, -size_delta);
  if (size_delta) for (int i=section+1, e=t->data.size(); i < e; ++i) t->data[i].start_row += size_delta;

  t->HideHiddenRows(t->data[section], true);
}

SystemTextView::~SystemTextView() {}
SystemTextView::SystemTextView(const string &title, File *f) : SystemTextView(title, f ? f->Contents() : "") {}
SystemTextView::SystemTextView(const string &title, const string &text) : impl(0) {}

SystemNavigationView::~SystemNavigationView() { if (auto nav = FromVoid<QtNavigation*>(impl)) delete nav; }
SystemNavigationView::SystemNavigationView() : impl(new QtNavigation()) {}

void SystemNavigationView::Show(bool show_or_hide) {
  auto w = GetTyped<QtWindowInterface*>(app->focused->id);
  auto nav = FromVoid<QtNavigation*>(impl);
  if (show_or_hide) {
    w->layout->addWidget(nav->content_widget.get());
    w->layout->setCurrentWidget(nav->content_widget.get());
  } else {
    w->layout->setCurrentWidget(w->container);
    w->layout->removeWidget(nav->content_widget.get());
  }
}

SystemTableView *SystemNavigationView::Back() { return nullptr; }

void SystemNavigationView::PushTableView(SystemTableView *t) {
  auto nav = FromVoid<QtNavigation*>(impl);
  auto table = FromVoid<QtTable*>(t->impl);
  nav->content_layout->addWidget(table->table.get());
  nav->content_layout->setCurrentWidget(table->table.get());
  nav->header_label->setText(table->table->windowTitle());
  nav->UpdateBackButton();
}

void SystemNavigationView::PushTextView(SystemTextView *t) {}
void SystemNavigationView::PopToRoot() {}
void SystemNavigationView::PopAll() {}

void SystemNavigationView::PopView(int n) {
  auto nav = FromVoid<QtNavigation*>(impl);
  for (int i=0; i<n; ++i) {
    auto w = nav->GetBackWidget();
    if (!w) return;
    nav->content_layout->removeWidget(w);
  }
}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB&) {}
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB&) {}
void Application::ShowSystemContextMenu(const vector<MenuItem>&items) {}

int Application::LoadSystemImage(const string &n) {
  app_images.emplace_back(make_unique<QIcon>(MakeQString(StrCat(app->assetdir, "../", n))));
  return app_images.size();
}

}; // namespace LFL
