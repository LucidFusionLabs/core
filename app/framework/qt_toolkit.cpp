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
#include <QToolBar>
#include <QTableView>
#include <QTableWidget>
#include <QPlainTextEdit>
#include <QPixmap>
#include "qt_common.h"

namespace LFL {
static vector<unique_ptr<QIcon>> app_images;
struct QtTable;

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
      alert->connect(alert.get(), &QInputDialog::finished, [=](int res){
        if (res == 1) { if (confirm_cb) confirm_cb(GetQString(alert->textValue())); }
        else          { if (cancel_cb) cancel_cb(""); } });
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
      msg->setText(MakeQString(t));
      msg->setInformativeText(MakeQString(m));
    }
    confirm_cb = move(cb);
  }
};

struct QtPanel {
  unique_ptr<QDialog> dialog;
  QHBoxLayout *layout=0;
  QLabel *title=0;
  QLineEdit *text=0;
  QtPanel(const string &title_text) : dialog(make_unique<QDialog>()) {
    layout = new QHBoxLayout(dialog.get());
    layout->addWidget((title = new QLabel(MakeQString(title_text), dialog.get())));
  }
};

struct QtToolbar {
  unique_ptr<QToolBar> toolbar;
  bool init=0;
  QtToolbar(MenuItemVec v) : toolbar(make_unique<QToolBar>()) {
    for (auto b = v.begin(), e = v.end(), i = b; i != e; ++i) {
      QAction *action = toolbar->addAction(MakeQString(i->shortcut));
      if (i->cb) toolbar->connect(action, &QAction::triggered, move(i->cb));
    }
  }
};

struct QtTableInterface {
  vector<TableSection> data;
  int data_rows=0;
  virtual ~QtTableInterface() {}
  QtTableInterface(vector<TableSection> d) : data(move(d)) {}

  void CheckExists(int section, int r) {
    CHECK_LT(section, data.size());
    CHECK_LT(r, data[section].item.size());
  }

  virtual void HandleCellClicked(const QModelIndex &index) = 0;
};

class QtStyledItemDelegate : public QStyledItemDelegate {
  public:
  QtTableInterface *table;
  QtStyledItemDelegate(QObject *parent=0, QtTableInterface *T=0) : QStyledItemDelegate(parent), table(T) {}
  void paint(QPainter* painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override {
    const QRect rect(option.rect);
    painter->setPen(Qt::gray);
    int col = index.column(), cols = index.model()->columnCount();
    if (col == 0)         painter->drawLine(rect.topLeft(),    rect.bottomLeft());
    if (col == cols - 1)  painter->drawLine(rect.topRight(),   rect.bottomRight());
    if (index.row() == 0) painter->drawLine(rect.topLeft(),    rect.topRight());
    if (1)                painter->drawLine(rect.bottomLeft(), rect.bottomRight());
    if (col == 1) {
      int section = -1, row = -1;
      LFL::TableSection::FindSectionOffset(table->data, index.row(), &section, &row);
      if (row >= 0) {
        table->CheckExists(section, row);
        auto &ci = table->data[section].item[row];
        bool right_aligned = (index.flags() & Qt::ItemIsUserCheckable) || ci.right_icon;
        if (right_aligned) {
          auto new_option = option;
          const int text_margin = QApplication::style()->pixelMetric(QStyle::PM_FocusFrameHMargin) + 1;
          new_option.rect = QStyle::alignedRect
            (option.direction, Qt::AlignRight, QSize(option.decorationSize.width() + 5, option.decorationSize.height()),
             QRect(option.rect.x() + text_margin, option.rect.y(), option.rect.width() - (2 * text_margin), option.rect.height()));
          return QStyledItemDelegate::paint(painter, new_option, index);
        }
      }
    }
    return QStyledItemDelegate::paint(painter, option, index);
  }

  bool editorEvent(QEvent *event, QAbstractItemModel *model, const QStyleOptionViewItem &option, const QModelIndex &index) override {
    Q_ASSERT(event);
    Q_ASSERT(model);
    if (index.flags() & Qt::ItemIsUserCheckable) {
      QVariant value = index.data(Qt::CheckStateRole);
      if (!value.isValid()) return false;

      if (event->type() == QEvent::MouseButtonRelease) {
        QRect check_rect = GetRealignedRect(option);
        if (!check_rect.contains(static_cast<QMouseEvent*>(event)->pos())) return false;
      } else if (event->type() == QEvent::KeyPress) {
        if (static_cast<QKeyEvent*>(event)->key() != Qt::Key_Space &&
            static_cast<QKeyEvent*>(event)->key() != Qt::Key_Select) return false;
      } else return false;

      Qt::CheckState state = Qt::CheckState(value.toInt()) == Qt::Checked ? Qt::Unchecked : Qt::Checked;
      return model->setData(index, state, Qt::CheckStateRole);

    } else if (event->type() == QEvent::MouseButtonRelease) {
      int section = -1, row = -1;
      if (!(index.flags() & Qt::ItemIsEnabled)) return false;
      LFL::TableSection::FindSectionOffset(table->data, index.row(), &section, &row);
      if (row < 0) return false;
      auto &ci = table->data[section].item[row];

      if (ci.right_icon_cb) {
        QRect check_rect = GetRealignedRect(option);
        if (check_rect.contains(static_cast<QMouseEvent*>(event)->pos())) { ci.right_icon_cb(); return true; }
      }

      table->HandleCellClicked(index);
      return true;
    }
    return false;
  }

  QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const override {
    auto editor = new QLineEdit(parent);
    return editor;
  }

  void setEditorData(QWidget *editor_widget, const QModelIndex &index) const override {
    auto editor = static_cast<QLineEdit*>(editor_widget);
    editor->setPlaceholderText("");
    int section = -1, row = -1;
    LFL::TableSection::FindSectionOffset(table->data, index.row(), &section, &row);
    if (row < 0) return;
    table->CheckExists(section, row);
    auto &ci = table->data[section].item[row];
    if (ci.HasPlaceholderValue())
      editor->setPlaceholderText(MakeQString(ci.GetPlaceholderValue()));
  }

  static QRect GetRealignedRect(const QStyleOptionViewItem &option) {
    const int text_margin = QApplication::style()->pixelMetric(QStyle::PM_FocusFrameHMargin) + 1;
    return QStyle::alignedRect(option.direction, Qt::AlignRight, option.decorationSize,
                               QRect(option.rect.x()     + (2 * text_margin), option.rect.y(),
                                     option.rect.width() - (2 * text_margin), option.rect.height()));
  }
};

class QtTableModel : public QStandardItemModel {
  public:
  vector<TableSection> *v;
  QtTableModel(vector<TableSection> *V) : v(V) {}
  QVariant data(const QModelIndex &index, int role) const override {
    if (role == Qt::ForegroundRole) {
      int section = -1, row = -1;
      TableSection::FindSectionOffset(*v, index.row(), &section, &row);
      if (section >= 0 && row < (*v)[section].item.size()) {
        auto &ci = (*v)[section].item[row];
        if ((index.column() == 0 && ci.dropdown_key.size()) ||
            (index.column() == 1 && ci.right_text.size())) return QColor(0, 122, 255);
        if (index.column() == 1 && ci.has_placeholder_val) return QColor(0xc0, 0xc0, 0xc0);
      }
    }
    return QStandardItemModel::data(index, role);
  }
};

class QtTableWidget : public QTableView {
  public:
  QtTable *lfl_parent;
  QtTableWidget(QtTable *P) : lfl_parent(P) {}
};

struct QtTable : public QtTableInterface {
  TableItem left_nav, right_nav;
  unique_ptr<QTableView> table;
  unique_ptr<QtTableModel> model;
  SystemTableView *lfl_self;
  IntIntCB delete_row_cb;
  string style;
  int editable_section=-1, editable_start_row=-1, selected_section=0, selected_row=0, row_height=30;
  QtStyledItemDelegate item_delegate;

  QtTable(SystemTableView *lself, const string &title, string sty, vector<TableSection> item) :
    QtTableInterface(move(item)), table(make_unique<QtTableWidget>(this)),
    model(make_unique<QtTableModel>(&data)), lfl_self(lself), style(move(sty)),
    item_delegate(nullptr, this) {

    vector<int> hide_indices;
    for (auto sb = data.begin(), se = data.end(), s = sb; s != se; ++s) {
      int section = (s - sb);
      s->start_row = data_rows + section;
      data_rows += s->item.size();
      model->appendRow(MakeRow(*s));
      for (auto rb = s->item.begin(), re = s->item.end(), r = rb; r != re; ++r) {
        if (r->hidden) hide_indices.push_back(s->start_row + 1 + (r - rb));
        model->appendRow(MakeRow(&(*r)));
      }
    }

    table->setShowGrid(false);
    table->setWindowTitle(MakeQString(title));
    table->setItemDelegate(&item_delegate);
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->horizontalHeader()->hide();
    table->verticalHeader()->hide();
    table->setModel(model.get());
    HideIndices(hide_indices, true);

    // QObject::connect(table.get(), &QTableView::clicked, bind(&QtTable::HandleCellClicked, this, _1));
    QObject::connect(model.get(), &QStandardItemModel::itemChanged, bind(&QtTable::HandleCellChanged, this, _1));
  }

  QList<QStandardItem*> MakeRow(const TableSection &s) {
    auto key = make_unique<QStandardItem>(MakeQString(s.header.key));
    auto val = make_unique<QStandardItem>();
    key->setTextAlignment(Qt::AlignRight | Qt::AlignBottom);
    key->setFlags(Qt::ItemIsEnabled);
    val->setFlags(Qt::ItemIsEnabled);
    return QList<QStandardItem*>{ key.release(), val.release() };
  }

  QList<QStandardItem*> MakeRow(TableItem *item) {
    int type = item->type;
    auto key = make_unique<QStandardItem>
      (item->left_icon ? *app_images[item->left_icon-1] : QIcon(), MakeQString(item->key));
    auto val = make_unique<QStandardItem>
      (item->right_icon ? *app_images[item->right_icon-1] : QIcon(), MakeQString
       (item->right_text.size() ? item->right_text :
        ((item->has_placeholder_val = item->HasPlaceholderValue()) ? item->GetPlaceholderValue() : item->val)));
    key->setFlags(Qt::ItemIsEnabled);
    if (!(type == TableItem::TextInput || type == TableItem::NumberInput
          || type == TableItem::PasswordInput)) val->setFlags(Qt::ItemIsEnabled); 
    if (type == TableItem::Toggle) {
      val->setText("");
      val->setCheckable(true);
      val->setCheckState(Qt::Unchecked);
    }
    val->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
    return QList<QStandardItem*>{ key.release(), val.release() };
  }

  int GetCollapsedRowId(int section, int row) const { return data[section].start_row + 1 + row; }

  void AddSection() {
    data.emplace_back(data_rows + data.size());
    model->appendRow(MakeRow(data.back()));
  }

  void AddRow(int section, TableItem item) {
    if (section == data.size()) AddSection();
    CHECK_LT(section, data.size());
    data[section].item.emplace_back(move(item));
    model->appendRow(MakeRow(&data[section].item.back()));
    for (auto i = data.begin() + section + 1, e = data.end(); i != e; ++i) i->start_row++;
    data_rows++;
  }

  void SetKey(int section, int row, const string &v) {
    CheckExists(section, row);
    auto &ci = data[section].item[row];
    ci.key = v;
    if (ci.depends.size()) ApplyItemDeps(ci, v);
  }

  void SetValue(int section, int row, const string &v) {
    CheckExists(section, row);
    auto &ci = data[section].item[row];
    ci.val = v;
    auto val = model->item(GetCollapsedRowId(section, row), 1);
    model->blockSignals(true);
    if (ci.type == TableItem::Toggle) val->setCheckState(v == "1" ? Qt::Checked : Qt::Unchecked);
    else if (!ci.right_text.size())
      val->setText(MakeQString((ci.has_placeholder_val = ci.HasPlaceholderValue()) ?
                               ci.GetPlaceholderValue() : ci.val));
    model->blockSignals(false);
  }

  void SetHidden(int section, int row, bool v) {
    CheckExists(section, row);
    data[section].item[row].hidden = v;
    table->setRowHidden(GetCollapsedRowId(section, row), v);
  }

  void HideIndices(const vector<int> &ind, bool v) { for (auto &i : ind) table->setRowHidden(i, v); }
  void HideHiddenRows(const TableSection &s, bool v) {
    for (auto rb = s.item.begin(), re = s.item.end(), r = rb; r != re; ++r)
      if (r->hidden) table->setRowHidden(s.start_row + 1 + (r - rb), v);
  }

  void HandleCellClicked(const QModelIndex &index) {
    int section = -1, row = -1;
    LFL::TableSection::FindSectionOffset(data, index.row(), &section, &row);
    if (row < 0) return;
    CheckExists(section, row);
    selected_section = section;
    selected_row = row;

    auto &ci = data[section].item[row];
    if (ci.dropdown_key.size() && index.column() == 0) {
      if (ci.cb) ci.cb();
    } else if (ci.type == LFL::TableItem::Command || ci.type == LFL::TableItem::Button) {
      if (ci.cb) ci.cb();
    } else if (ci.type == LFL::TableItem::Label && row + 1 < data[section].item.size()) {
      auto &next_ci = data[section].item[row+1];
      if (next_ci.type == LFL::TableItem::Picker ||
          next_ci.type == LFL::TableItem::FontPicker) {
        next_ci.hidden = !next_ci.hidden;
      }
    }
  }

  void HandleCellChanged(QStandardItem *item) {
    if (item->column() != 1) return;
    int section = -1, row = -1;
    LFL::TableSection::FindSectionOffset(data, item->row(), &section, &row);
    if (row < 0) return;
    CheckExists(section, row);
    auto &ci = data[section].item[row];
    if (item->text().size()) { ci.has_placeholder_val = false; return; }
    if (ci.HasPlaceholderValue()) {
      ci.has_placeholder_val = true;
      model->blockSignals(true);
      item->setText(MakeQString(ci.GetPlaceholderValue()));
      model->blockSignals(false);
    }
  }

  void ApplyItemDeps(const TableItem &ci, const string &v) {
    TableSection::ApplyItemDepends(ci, v, &data, [=](const LFL::TableItem::Dep &d){
      int section_start_row = data[d.section].start_row;
      auto row = MakeRow(&data[d.section].item[d.row]);
      model->blockSignals(true);
      model->setItem(section_start_row + 1 + d.row, 0, row[0]);
      model->setItem(section_start_row + 1 + d.row, 1, row[1]);
      model->blockSignals(false);
      table->setRowHidden(section_start_row + 1 + d.row, data[d.section].item[d.row].hidden);
    });
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
    QObject::connect(header_back   .get(), &QPushButton::clicked, bind(&QtNavigation::HandleBackClicked,    this));
    QObject::connect(header_forward.get(), &QPushButton::clicked, bind(&QtNavigation::HandleForwardClicked, this));
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
    if (auto view = dynamic_cast<QtTableWidget*>(content_layout->currentWidget())) {
      if (view->lfl_parent->left_nav.cb) {
        header_back->setText(MakeQString(view->lfl_parent->left_nav.key));
        header_back->setHidden(false);
        return;
      }
    }
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
    if (auto view = dynamic_cast<QtTableWidget*>(w)) {
      if (view->lfl_parent->lfl_self->hide_cb) view->lfl_parent->lfl_self->hide_cb();
      if (view->lfl_parent->left_nav.cb) {
        view->lfl_parent->left_nav.cb();
        return;
      }
    }
    content_layout->removeWidget(w);
    if ((w = GetBackWidget())) header_label->setText(w->windowTitle());
    UpdateBackButton();
  }

  void HandleForwardClicked() {
    auto w = GetBackWidget();
    if (!w) return;
    if (auto view = dynamic_cast<QtTableWidget*>(w)) {
      if (view->lfl_parent->right_nav.cb) {
        view->lfl_parent->right_nav.cb();
        return;
      }
    }
  }
};

struct QtAlertView : public SystemAlertView {
  QtAlert *alert;
  ~QtAlertView() { delete alert; }
  QtAlertView(AlertItemVec items) : alert(new QtAlert(move(items))) {}

  void Show(const string &arg) { RunModal(arg); }
  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb)
  { alert->Update(title, msg, move(confirm_cb)); Show(arg); }

  string RunModal(const string &arg) {
    app->ReleaseMouseFocus();
    if (alert->add_text) {
      alert->alert->setTextValue(MakeQString(arg));
      alert->alert->open();
    } else {
      alert->msg->open();
    }
    return string(); 
  }
};

struct QtMenuView : public SystemMenuView {
  QMenu *menu;
  ~QtMenuView() { delete menu; }
  QtMenuView(const string &title_text, MenuItemVec v) {
    auto mb = dynamic_cast<QtWindowInterface*>(app->focused)->window->menuBar();
    menu = new QMenu(MakeQString(title_text), mb);
    for (auto b = v.begin(), e = v.end(), i = b; i != e; ++i) {
      QAction *action = menu->addAction(MakeQString(i->name));
      if (i->shortcut.size()) action->setShortcut(QKeySequence(Qt::CTRL + i->shortcut[0]));
      if (i->cb) mb->connect(action, &QAction::triggered, move(i->cb));
    }
    mb->addMenu(menu);
  }
  void Show() {}
};

struct QtPanelView : public SystemPanelView {
  QtPanel *panel;
  ~QtPanelView() { delete panel; }
  QtPanelView(const Box &b, const string &title, PanelItemVec items) : panel(new QtPanel(title)) {
    for (auto &i : items) {
      const Box &b = i.box;
      const string &t = i.type;
      if (t == "textbox") {
        panel->layout->addWidget((panel->text = new QLineEdit(panel->dialog.get())));
      } else if (PrefixMatch(t, "button:")) {
        auto button = new QPushButton(MakeQString(t.substr(7)), panel->dialog.get());
        if (i.cb) QObject::connect(button, &QPushButton::clicked, [=](){
                                   i.cb(panel->text ? GetQString(panel->text->text()) : string());                         
                                   });
        panel->layout->addWidget(button);
      } else ERROR("unknown panel item ", t);
    }
  }

  void SetTitle(const string &title) { panel->title->setText(MakeQString(title)); }
  void Show() { panel->dialog->show(); }
};

struct QtToolbarView : public SystemToolbarView {
  QtToolbar *tb;
  ~QtToolbarView() { delete tb; }
  QtToolbarView(MenuItemVec items) : tb(new QtToolbar(move(items))) {}

  void ToggleButton(const string &n) {}
  void Show(bool show_or_hide) {  
    if (!tb->init && (tb->init=1)) 
      dynamic_cast<QtWindowInterface*>(app->focused)->layout->setMenuBar(tb->toolbar.get());
    if (show_or_hide) tb->toolbar->show();
    else              tb->toolbar->hide();
  }
};

struct QtTableView : public SystemTableView {
  QtTable *table;
  ~QtTableView() { delete table; }
  QtTableView(const string &title, const string &style, TableItemVec items) :
    table(new QtTable(this, title, style, TableSection::Convert(move(items)))) {}

  StringPairVec GetSectionText(int ind) {
    StringPairVec ret;
    CHECK_LT(ind, table->data.size());

    for (int start_row=table->data[ind].start_row, l=table->data[ind].item.size(), i=0; i != l; i++) {
      auto &ci = table->data[ind].item[i];
      string val;

      if (ci.type == TableItem::Toggle)
        val = table->model->itemFromIndex(table->model->index(start_row+i+1, 1))->checkState() == Qt::Checked ? "1" : "";
      else if (!ci.has_placeholder_val)
        val = GetQString(table->model->index(start_row+i+1, 1).data().toString());

      if (ci.dropdown_key.size()) ret.emplace_back(ci.dropdown_key, ci.key);
      ret.emplace_back(ci.key, val);
    }
    return ret;
  }

  void SetSectionValues(int section, const StringVec &item) {
    if (section == table->data.size()) table->AddSection();
    CHECK_LT(section, table->data.size());
    CHECK_EQ(item.size(), table->data[section].item.size());
    for (int i=0, l=table->data[section].item.size(); i != l; ++i) table->SetValue(section, i, item[i]);
  }

  void AddToolbar(SystemToolbarView *t) {}
  void AddNavigationButton(int align, const TableItem &item) {
    if      (align == HAlign::Left)  table->left_nav  = item;
    else if (align == HAlign::Right) table->right_nav = item;
  }

  void DelNavigationButton(int align) {
    if      (align == HAlign::Left)  table->left_nav  = TableItem();
    else if (align == HAlign::Right) table->right_nav = TableItem();
  }

  void Show(bool show_or_hide) {
    auto w = dynamic_cast<QtWindowInterface*>(app->focused);
    if (show_or_hide) {
      if (show_cb) show_cb();
      w->layout->addWidget(table->table.get());
      w->layout->setCurrentWidget(table->table.get());
    } else {
      w->layout->setCurrentWidget(w->opengl_container);
      w->layout->removeWidget(table->table.get());
    }
  }

  void AddRow(int section, TableItem item) { table->AddRow(section, move(item)); }
  string GetKey(int section, int row) { table->CheckExists(section, row); return table->data[section].item[row].key; }
  int GetTag(int section, int row) { table->CheckExists(section, row); return table->data[section].item[row].tag; }
  void SetTag(int section, int row, int val) { table->CheckExists(section, row); table->data[section].item[row].tag = val; }
  void SetKey(int section, int row, const string &val) { table->SetKey(section, row, val); }
  void SetValue(int section, int row, const string &val) { table->SetValue(section, row, val); }
  void SetHidden(int section, int row, bool val) { table->SetHidden(section, row, val); }
  void SetTitle(const string &title) { table->table->setWindowTitle(MakeQString(title)); }
  void BeginUpdates() {}
  void EndUpdates() {}

  PickerItem *GetPicker(int section, int row) { return 0; }

  void SelectRow(int section, int row) {
    table->selected_section = section;
    table->selected_row = row; 
  }

  void SetEditableSection(int section, int start_row, LFL::IntIntCB cb) {
    table->delete_row_cb = move(cb);
    table->editable_section = section;
    table->editable_start_row = start_row;
  }

  void ReplaceSection(int section, TableItem h, int flag, TableItemVec item) {
    bool added = section == table->data.size();
    if (added) table->AddSection();
    CHECK_LT(section, table->data.size());
    int old_item_size = table->data[section].item.size(), item_size = item.size();
    int size_delta = item_size - old_item_size, section_start_row = table->data[section].start_row;
    table->HideHiddenRows(table->data[section], false);

    table->data[section] = LFL::TableSection(move(h), flag, section_start_row);
    table->data[section].item = move(item);
    table->data_rows += size_delta;
    for (int i=0; i < item_size; ++i) {
      auto row = table->MakeRow(&table->data[section].item[i]);
      if (i < old_item_size) {
        table->model->blockSignals(true);
        table->model->setItem(section_start_row + 1 + i, 0, row[0]);
        table->model->setItem(section_start_row + 1 + i, 1, row[1]);
        table->model->blockSignals(false);
      } else table->model->insertRow(section_start_row + 1 + i, row);
    }
    if (size_delta < 0) table->model->removeRows(section_start_row + 1 + item_size, -size_delta);
    if (size_delta) for (int i=section+1, e=table->data.size(); i < e; ++i) table->data[i].start_row += size_delta;

    table->HideHiddenRows(table->data[section], true);
  }
};

struct QtTextView : public SystemTextView {
  QPlainTextEdit *text;
  ~QtTextView() { delete text; }
  QtTextView(const string &title, File *f) : QtTextView(title, f ? f->Contents() : "") {}
  QtTextView(const string &title, const string &data) : text(new QPlainTextEdit(MakeQString(data))) {
    text->setWindowTitle(MakeQString(title));
    text->setReadOnly(true);
  }
};

struct QtNavigationView : public SystemNavigationView {
  QtNavigation *nav;
  ~QtNavigationView() { delete nav; }
  QtNavigationView() : nav(new QtNavigation()) {}

  void Show(bool show_or_hide) {
    auto w = dynamic_cast<QtWindowInterface*>(app->focused);
    if (show_or_hide) {
      w->layout->addWidget(nav->content_widget.get());
      w->layout->setCurrentWidget(nav->content_widget.get());
    } else {
      w->layout->setCurrentWidget(w->opengl_container);
      w->layout->removeWidget(nav->content_widget.get());
    }
  }

  SystemTableView *Back() { 
    for (int i = nav->content_layout->count()-1; i >= 0; --i) {
      QWidget *qw = nav->content_layout->widget(i);
      if (auto qt = dynamic_cast<QtTableWidget*>(qw)) return qt->lfl_parent->lfl_self;
    }
    return nullptr;
  }

  void PushTableView(SystemTableView *t) {
    if (!root) root = t;
    if (t->show_cb) t->show_cb();
    auto table = dynamic_cast<QtTableView*>(t)->table;
    nav->content_layout->addWidget(table->table.get());
    nav->content_layout->setCurrentWidget(table->table.get());
    nav->header_label->setText(table->table->windowTitle());
    nav->UpdateBackButton();
  }

  void PushTextView(SystemTextView *t) {
    if (t->show_cb) t->show_cb();
    auto text = dynamic_cast<QtTextView*>(t)->text;
    nav->content_layout->addWidget(text);
    nav->content_layout->setCurrentWidget(text);
    nav->header_label->setText(text->windowTitle());
    nav->UpdateBackButton();
  }

  void PopAll()    { int n=nav->content_layout->count(); PopView(n); }
  void PopToRoot() { int n=nav->content_layout->count(); PopView(n-1); }
  void PopView(int n) {
    for (int i=0; i<n; ++i) {
      auto w = nav->GetBackWidget();
      if (!w) return;
      nav->content_layout->removeWidget(w);
    }
  }
};

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &font_change_cb) {
  QFontDialog *font_chooser = new QFontDialog(QFont(MakeQString(cur_font.name), cur_font.size));
  font_chooser->show();
  QObject::connect(font_chooser, &QFontDialog::fontSelected, [=](const QFont &f) {
    font_change_cb(StringVec{ GetQString(f.family()), StrCat(f.pointSize()) });
  });
}

void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB &choose_cb) {
  QFileDialog *file_chooser = new QFileDialog();
  if (!files && dirs) file_chooser->setOptions(QFileDialog::ShowDirsOnly);
  QObject::connect(file_chooser, &QFileDialog::filesSelected, [=](const QStringList &selected){
    if (selected.size()) choose_cb(GetQStringList(selected));
  });
  file_chooser->show();
}

void Application::ShowSystemContextMenu(const vector<MenuItem> &items) {
  auto screen = app->focused;
  QMenu *menu = new QMenu();
  for (auto &i : items) {
    QAction *action = menu->addAction(MakeQString(i.name));
    if (i.cb) menu->connect(action, &QAction::triggered, i.cb);
  }
  menu->popup(dynamic_cast<QtWindowInterface*>(screen)->opengl_window->mapToGlobal
              (MakeQPoint(Input::TransformMouseCoordinate(screen->mouse))));
}

int Application::LoadSystemImage(const string &n) {
  app_images.emplace_back(make_unique<QIcon>(MakeQString(StrCat(app->assetdir, "../drawable-xhdpi/",
                                                                n, ".png"))));
  return app_images.size();
}

void Application::UpdateSystemImage(int n, Texture &t) {
  CHECK_RANGE(n-1, 0, app_images.size());
  QPixmap pixmap;
  pixmap.convertFromImage(MakeQImage(t));
  app_images[n-1] = make_unique<QIcon>(move(pixmap));
}

unique_ptr<SystemAlertView> SystemAlertView::Create(AlertItemVec items) { return make_unique<QtAlertView>(move(items)); }
unique_ptr<SystemPanelView> SystemPanelView::Create(const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<SystemToolbarView> SystemToolbarView::Create(MenuItemVec items) { return make_unique<QtToolbarView>(move(items)); }
unique_ptr<SystemMenuView> SystemMenuView::Create(const string &title, MenuItemVec items) { return make_unique<QtMenuView>(title, move(items)); }
unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(MenuItemVec items) { return make_unique<QtMenuView>("Edit", move(items)); }
unique_ptr<SystemTableView> SystemTableView::Create(const string &title, const string &style, TableItemVec items) { return make_unique<QtTableView>(title, style, move(items)); }
unique_ptr<SystemTextView> SystemTextView::Create(const string &title, File *file) { return make_unique<QtTextView>(title, file); }
unique_ptr<SystemTextView> SystemTextView::Create(const string &title, const string &text) { return make_unique<QtTextView>(title, text); }
unique_ptr<SystemNavigationView> SystemNavigationView::Create() { return make_unique<QtNavigationView>(); }


}; // namespace LFL
