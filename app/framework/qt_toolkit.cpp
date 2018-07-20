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
static FreeListVector<unique_ptr<QIcon>> app_images;
struct QtTableView;

struct QtAlertView : public AlertViewInterface {
  QtWindowInterface *win;
  string style;
  bool add_text = 0;
  StringCB cancel_cb, confirm_cb;
  unique_ptr<QInputDialog> alert;
  unique_ptr<QMessageBox> msg;

  QtAlertView(Window *W, AlertItemVec kv) : win(dynamic_cast<QtWindowInterface*>(W)) {
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

  string RunModal(const string &arg) { return ""; }
  void Hide() {}

  void Show(const string &arg) {
    win->parent->ReleaseMouseFocus();
    if (add_text) {
      alert->setTextValue(MakeQString(arg));
      alert->open();
    } else {
      msg->open();
    }
  }

  void ShowCB(const string &t, const string &m, const string &arg, StringCB cb) {
    if (add_text) {
      alert->setWindowTitle(MakeQString(t));
      alert->setLabelText(MakeQString(m));
    } else { 
      msg->setText(MakeQString(t));
      msg->setInformativeText(MakeQString(m));
    }
    confirm_cb = move(cb);
    Show(arg);
  }
};

struct QtPanelView : public PanelViewInterface {
  unique_ptr<QDialog> dialog;
  QHBoxLayout *layout=0;
  QLabel *title=0;
  QLineEdit *text=0;

  QtPanelView(const Box &b, const string &title_text, PanelItemVec items) : dialog(make_unique<QDialog>()) {
    layout = new QHBoxLayout(dialog.get());
    layout->addWidget((title = new QLabel(MakeQString(title_text), dialog.get())));
    for (auto &i : items) {
      const Box &b = i.box;
      const string &t = i.type;
      if (t == "textbox") {
        layout->addWidget((text = new QLineEdit(dialog.get())));
      } else if (PrefixMatch(t, "button:")) {
        auto button = new QPushButton(MakeQString(t.substr(7)), dialog.get());
        if (i.cb) QObject::connect(button, &QPushButton::clicked, [=](){ i.cb(text ? GetQString(text->text()) : string()); });
        layout->addWidget(button);
      } else ERROR("unknown panel item ", t);
    }
  }

  void SetTitle(const string &title_text) { title->setText(MakeQString(title_text)); }
  void Show() { dialog->show(); }
};
 
struct QtToolbarView : public ToolbarViewInterface {
  QtWindowInterface *win;
  unique_ptr<QToolBar> toolbar;
  string theme;
  bool init=0;

  QtToolbarView(Window *W, MenuItemVec v) : win(dynamic_cast<QtWindowInterface*>(W)), toolbar(make_unique<QToolBar>()) {
    for (auto b = v.begin(), e = v.end(), i = b; i != e; ++i) {
      QAction *action = toolbar->addAction(MakeQString(i->shortcut));
      if (i->cb) toolbar->connect(action, &QAction::triggered, move(i->cb));
    }
  }

  string GetTheme() { return theme; }
  void SetTheme(const string &x) { theme=x; }
  void ToggleButton(const string &n) {}
  void Show(bool show_or_hide) {  
    if (!init && (init=1)) win->layout->setMenuBar(toolbar.get());
    if (show_or_hide) toolbar->show();
    else              toolbar->hide();
  }
};

struct QtMenuView : public MenuViewInterface {
  Window *w;
  QMenu *menu;

  ~QtMenuView() { delete menu; }
  QtMenuView(Window *W, const string &title_text, MenuItemVec v) : w(W) {
    auto mb = dynamic_cast<QtWindowInterface*>(w)->window->menuBar();
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

struct QtTableInterface {
  vector<TableSection<TableItem>> data;
  int data_rows=0;
  virtual ~QtTableInterface() {}
  QtTableInterface(vector<TableSection<TableItem>> d) : data(move(d)) {}

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
      LFL::TableSection<TableItem>::FindSectionOffset(table->data, index.row(), &section, &row);
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
      LFL::TableSection<TableItem>::FindSectionOffset(table->data, index.row(), &section, &row);
      if (row < 0) return false;
      auto &ci = table->data[section].item[row];

      if (ci.right_cb) {
        QRect check_rect = GetRealignedRect(option);
        if (check_rect.contains(static_cast<QMouseEvent*>(event)->pos())) { ci.right_cb(""); return true; }
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
    LFL::TableSection<TableItem>::FindSectionOffset(table->data, index.row(), &section, &row);
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
  vector<TableSection<TableItem>> *v;
  QtTableModel(vector<TableSection<TableItem>> *V) : v(V) {}
  QVariant data(const QModelIndex &index, int role) const override {
    if (role == Qt::ForegroundRole) {
      int section = -1, row = -1;
      TableSection<TableItem>::FindSectionOffset(*v, index.row(), &section, &row);
      if (section >= 0 && row < (*v)[section].item.size()) {
        auto &ci = (*v)[section].item[row];
        if ((index.column() == 0 && ci.dropdown_key.size()) ||
            (index.column() == 1 && ci.right_text.size())) return QColor(0, 122, 255);
        if (index.column() == 1 && (ci.flags & TableItem::Flag::PlaceHolderVal)) return QColor(0xc0, 0xc0, 0xc0);
      }
    }
    return QStandardItemModel::data(index, role);
  }
};

struct QtCollectionView : public CollectionViewInterface {
  QtCollectionView(const string &title, string sty, vector<CollectionItem> item) {}
  void SetToolbar(ToolbarViewInterface *t) {}
  void Show(bool show_or_hide) {}
};

class QtTableWidget : public QTableView {
  public:
  QtTableView *parent;
  QtTableWidget(QtTableView *P) : parent(P) {}
};

struct QtTableView : public QtTableInterface, public TableViewInterface {
  QtWindowInterface *win;
  TableItem left_nav, right_nav;
  unique_ptr<QTableView> table;
  unique_ptr<QtTableModel> model;
  IntIntCB delete_row_cb;
  string style;
  int editable_section=-1, editable_start_row=-1, selected_section=0, selected_row=0, row_height=30;
  QtStyledItemDelegate item_delegate;

  QtTableView(Window *W, const string &title, string sty, vector<TableItem> item) :
    QtTableInterface(TableSection<TableItem>::Convert(move(item))), win(dynamic_cast<QtWindowInterface*>(W)),
    table(make_unique<QtTableWidget>(this)), model(make_unique<QtTableModel>(&data)),
    style(move(sty)), item_delegate(nullptr, this) {

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
    QObject::connect(model.get(), &QStandardItemModel::itemChanged, bind(&QtTableView::HandleCellChanged, this, _1));
  }

  QList<QStandardItem*> MakeRow(const TableSection<TableItem> &s) {
    auto key = make_unique<QStandardItem>(MakeQString(s.header.key));
    auto val = make_unique<QStandardItem>();
    key->setTextAlignment(Qt::AlignRight | Qt::AlignBottom);
    key->setFlags(Qt::ItemIsEnabled);
    val->setFlags(Qt::ItemIsEnabled);
    return QList<QStandardItem*>{ key.release(), val.release() };
  }

  QList<QStandardItem*> MakeRow(TableItem *item) {
    int type = item->type;
    item->flags = item->HasPlaceholderValue() ? TableItem::Flag::PlaceHolderVal : 0;
    auto key = make_unique<QStandardItem>
      (item->left_icon ? *app_images[item->left_icon-1] : QIcon(), MakeQString(item->key));
    auto val = make_unique<QStandardItem>
      (item->right_icon ? *app_images[item->right_icon-1] : QIcon(), MakeQString
       (item->right_text.size() ? item->right_text :
        (item->HasPlaceholderValue() ? item->GetPlaceholderValue() : item->val)));
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

  void DelNavigationButton(int align) {
    if      (align == HAlign::Left)  left_nav  = TableItem();
    else if (align == HAlign::Right) right_nav = TableItem();
  }

  void AddNavigationButton(int align, const TableItem &item) {
    if      (align == HAlign::Left)  left_nav  = item;
    else if (align == HAlign::Right) right_nav = item;
  }

  void SetToolbar(ToolbarViewInterface *t) {}

  void Show(bool show_or_hide) {
    if (show_or_hide) {
      if (show_cb) show_cb();
      win->layout->addWidget(table.get());
      win->layout->setCurrentWidget(table.get());
    } else {
      win->layout->setCurrentWidget(win->opengl_container);
      win->layout->removeWidget(table.get());
    }
  }

  string GetKey  (int section, int row) { CheckExists(section, row); return data[section].item[row].key; }
  string GetValue(int section, int row) { return ""; }
  int    GetTag  (int section, int row) { CheckExists(section, row); return data[section].item[row].tag; }
  PickerItem *GetPicker(int section, int row) { return 0; }

  StringPairVec GetSectionText(int ind) {
    StringPairVec ret;
    CHECK_LT(ind, data.size());

    for (int start_row=data[ind].start_row, l=data[ind].item.size(), i=0; i != l; i++) {
      auto &ci = data[ind].item[i];
      string val;

      if (ci.type == TableItem::Toggle)
        val = model->itemFromIndex(model->index(start_row+i+1, 1))->checkState() == Qt::Checked ? "1" : "";
      else if (!(ci.flags & TableItem::Flag::PlaceHolderVal))
        val = GetQString(model->index(start_row+i+1, 1).data().toString());

      if (ci.dropdown_key.size()) ret.emplace_back(ci.dropdown_key, ci.key);
      ret.emplace_back(ci.key, val);
    }
    return ret;
  }
  
  void BeginUpdates() {}
  void EndUpdates() {}

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

  void SelectRow(int section, int row) {
    selected_section = section;
    selected_row = row; 
  }

  void ReplaceRow(int section, int row, TableItem item) {}

  void ReplaceSection(int section, TableItem h, int flag, TableItemVec item) {
    bool added = section == data.size();
    if (added) AddSection();
    CHECK_LT(section, data.size());
    int old_item_size = data[section].item.size(), item_size = item.size();
    int size_delta = item_size - old_item_size, section_start_row = data[section].start_row;
    HideHiddenRows(data[section], false);

    data[section] = LFL::TableSection<LFL::TableItem>(move(h), flag, section_start_row);
    data[section].item = move(item);
    data_rows += size_delta;
    for (int i=0; i < item_size; ++i) {
      auto row = MakeRow(&data[section].item[i]);
      if (i < old_item_size) {
        model->blockSignals(true);
        model->setItem(section_start_row + 1 + i, 0, row[0]);
        model->setItem(section_start_row + 1 + i, 1, row[1]);
        model->blockSignals(false);
      } else model->insertRow(section_start_row + 1 + i, row);
    }
    if (size_delta < 0) model->removeRows(section_start_row + 1 + item_size, -size_delta);
    if (size_delta) for (int i=section+1, e=data.size(); i < e; ++i) data[i].start_row += size_delta;

    HideHiddenRows(data[section], true);
  }

  void ApplyChangeList(const TableSectionInterface::ChangeList &ci) {
    TableSection<TableItem>::ApplyChangeList(ci, &data, [=](const LFL::TableSectionInterface::Change &d){
      int section_start_row = data[d.section].start_row;
      auto row = MakeRow(&data[d.section].item[d.row]);
      model->blockSignals(true);
      model->setItem(section_start_row + 1 + d.row, 0, row[0]);
      model->setItem(section_start_row + 1 + d.row, 1, row[1]);
      model->blockSignals(false);
      table->setRowHidden(section_start_row + 1 + d.row, data[d.section].item[d.row].hidden);
    });
  }

  void SetSectionValues(int section, const StringVec &item) {
    if (section == data.size()) AddSection();
    CHECK_LT(section, data.size());
    CHECK_EQ(item.size(), data[section].item.size());
    for (int i=0, l=data[section].item.size(); i != l; ++i) SetValue(section, i, item[i]);
  }

  void SetSectionColors(int section, const vector<Color> &item) {
    if (section == data.size()) AddSection();
    CHECK_LT(section, data.size());
    CHECK_EQ(item.size(), data[section].item.size());
    for (int i=0, l=data[section].item.size(); i != l; ++i) SetColor(section, i, item[i]);
  }

  void SetHeader(int section, TableItem header) {
  }

  void SetKey(int section, int row, const string &v) {
    CheckExists(section, row);
    auto &ci = data[section].item[row];
    ci.key = v;
  }
  
  void SetTag(int section, int row, int val) {
    CheckExists(section, row);
    data[section].item[row].tag = val;
  }

  void SetValue(int section, int row, const string &v) {
    CheckExists(section, row);
    auto &ci = data[section].item[row];
    ci.val = v;
    auto val = model->item(GetCollapsedRowId(section, row), 1);
    model->blockSignals(true);
    if (ci.type == TableItem::Toggle) val->setCheckState(v == "1" ? Qt::Checked : Qt::Unchecked);
    else if (!ci.right_text.size()) {
      ci.flags |= (ci.HasPlaceholderValue() ? TableItem::Flag::PlaceHolderVal : 0);
      val->setText(MakeQString(ci.HasPlaceholderValue() ? ci.GetPlaceholderValue() : ci.val));
    }
    model->blockSignals(false);
  }

  void SetHidden(int section, int row, int v) {
    CheckExists(section, row);
    data[section].item[row].hidden = v;
    table->setRowHidden(GetCollapsedRowId(section, row), v);
  }

  void SetSelected(int section, int row, int v) {
    CheckExists(section, row);
    data[section].item[row].selected = v;
  }

  void SetColor(int section, int row, const Color &v) {
    CheckExists(section, row);
    data[section].item[row].font.fg = v;
  }

  void SetTitle(const string &title) { table->setWindowTitle(MakeQString(title)); }
  void SetTheme(const string &theme) {}
  void SetSectionEditable(int section, int start_row, int skip_last_rows, LFL::IntIntCB cb) {
    delete_row_cb = move(cb);
    editable_section = section;
    editable_start_row = start_row;
  }
  
  void HideIndices(const vector<int> &ind, bool v) { for (auto &i : ind) table->setRowHidden(i, v); }
  void HideHiddenRows(const TableSection<TableItem> &s, bool v) {
    for (auto rb = s.item.begin(), re = s.item.end(), r = rb; r != re; ++r)
      if (r->hidden) table->setRowHidden(s.start_row + 1 + (r - rb), v);
  }

  void HandleCellClicked(const QModelIndex &index) {
    int section = -1, row = -1;
    LFL::TableSection<LFL::TableItem>::FindSectionOffset(data, index.row(), &section, &row);
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
    LFL::TableSection<LFL::TableItem>::FindSectionOffset(data, item->row(), &section, &row);
    if (row < 0) return;
    CheckExists(section, row);
    auto &ci = data[section].item[row];
    if (item->text().size()) { ci.flags &= ~TableItem::Flag::PlaceHolderVal; return; }
    if (ci.HasPlaceholderValue()) {
      ci.flags |= TableItem::Flag::PlaceHolderVal;
      model->blockSignals(true);
      item->setText(MakeQString(ci.GetPlaceholderValue()));
      model->blockSignals(false);
    }
  }
};

struct QtTextView : public TextViewInterface {
  QPlainTextEdit *text;
  ~QtTextView() { delete text; }
  QtTextView(const string &title, File *f) : QtTextView(title, f ? f->Contents() : "") {}
  QtTextView(const string &title, const string &data) : text(new QPlainTextEdit(MakeQString(data))) {
    text->setWindowTitle(MakeQString(title));
    text->setReadOnly(true);
  }
  void Show(bool show_or_hide) {}
};

struct QtNavigationView : public NavigationViewInterface {
  QtWindowInterface *win;
  unique_ptr<QWidget> header_widget, content_widget;
  unique_ptr<QLabel> header_label;
  unique_ptr<QPushButton> header_back, header_forward;
  unique_ptr<QHBoxLayout> header_layout;
  unique_ptr<QStackedLayout> content_layout;

  QtNavigationView(Window *W) : win(dynamic_cast<QtWindowInterface*>(W)),
  header_widget(make_unique<QWidget>()), content_widget(make_unique<QWidget>()),
  header_label(make_unique<QLabel>()), header_back(make_unique<QPushButton>()), header_forward(make_unique<QPushButton>()),
  header_layout(make_unique<QHBoxLayout>()), content_layout(make_unique<QStackedLayout>()) {
    QObject::connect(header_back   .get(), &QPushButton::clicked, bind(&QtNavigationView::HandleBackClicked,    this));
    QObject::connect(header_forward.get(), &QPushButton::clicked, bind(&QtNavigationView::HandleForwardClicked, this));
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
      if (view->parent->left_nav.cb) {
        header_back->setText(MakeQString(view->parent->left_nav.key));
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
      if (view->parent->hide_cb) view->parent->hide_cb();
      if (view->parent->left_nav.cb) {
        view->parent->left_nav.cb();
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
      if (view->parent->right_nav.cb) {
        view->parent->right_nav.cb();
        return;
      }
    }
  }

  void Show(bool show_or_hide) {
    if (show_or_hide) {
      win->layout->addWidget(content_widget.get());
      win->layout->setCurrentWidget(content_widget.get());
    } else {
      win->layout->setCurrentWidget(win->opengl_container);
      win->layout->removeWidget(content_widget.get());
    }
  }

  TableViewInterface *Back() { 
    for (int i = content_layout->count()-1; i >= 0; --i) {
      QWidget *qw = content_layout->widget(i);
      if (auto qt = dynamic_cast<QtTableWidget*>(qw)) return qt->parent;
    }
    return nullptr;
  }

  void PushTableView(TableViewInterface *t) {
    if (!root) root = t;
    if (t->show_cb) t->show_cb();
    auto table = dynamic_cast<QtTableView*>(t);
    content_layout->addWidget(table->table.get());
    content_layout->setCurrentWidget(table->table.get());
    header_label->setText(table->table->windowTitle());
    UpdateBackButton();
  }

  void PushTextView(TextViewInterface *t) {
    if (t->show_cb) t->show_cb();
    auto text = dynamic_cast<QtTextView*>(t)->text;
    content_layout->addWidget(text);
    content_layout->setCurrentWidget(text);
    header_label->setText(text->windowTitle());
    UpdateBackButton();
  }

  void PopAll()    { int n=content_layout->count(); PopView(n); }
  void PopToRoot() { int n=content_layout->count(); PopView(n-1); }
  void PopView(int n) {
    for (int i=0; i<n; ++i) {
      auto w = GetBackWidget();
      if (!w) return;
      content_layout->removeWidget(w);
    }
  }

  void SetTheme(const string&) {}
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
  QMenu *menu = new QMenu();
  for (auto &i : items) {
    QAction *action = menu->addAction(MakeQString(i.name));
    if (i.cb) menu->connect(action, &QAction::triggered, i.cb);
  }
  menu->popup(dynamic_cast<QtWindowInterface*>(focused)->opengl_window->mapToGlobal
              (MakeQPoint(Input::TransformMouseCoordinate(focused, focused->mouse))));
}

int Application::LoadSystemImage(const string &n) {
  return app_images.Insert
    (make_unique<QIcon>(MakeQString(StrCat(assetdir, "../drawable-xhdpi/", n, ".png")))) + 1;
}

void Application::UpdateSystemImage(int n, Texture &t) {
  CHECK_RANGE(n-1, 0, app_images.size());
  QPixmap pixmap;
  pixmap.convertFromImage(MakeQImage(t));
  app_images[n-1] = make_unique<QIcon>(move(pixmap));
}

void Application::UnloadSystemImage(int n) {
  if (app_images[n-1]) app_images[n-1].reset();
  app_images.Erase(n-1);
}

unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(Window *w, AlertItemVec items) { return make_unique<QtAlertView>(w, move(items)); }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(Window *w, const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<ToolbarViewInterface> SystemToolkit::CreateToolbar(Window *w, const string &theme, MenuItemVec items, int flag) { return make_unique<QtToolbarView>(w, move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(Window *w, const string &title, MenuItemVec items) { return make_unique<QtMenuView>(w, title, move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(Window *w, MenuItemVec items) { return make_unique<QtMenuView>(w, "Edit", move(items)); }
unique_ptr<CollectionViewInterface> SystemToolkit::CreateCollectionView(Window *w, const string &title, const string &style, const string &theme, vector<CollectionItem> items) { return make_unique<QtCollectionView>(title, style, move(items)); }
unique_ptr<TableViewInterface> SystemToolkit::CreateTableView(Window *w, const string &title, const string &style, const string &theme, TableItemVec items) { return make_unique<QtTableView>(w, title, style, move(items)); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(Window *w, const string &title, File *file) { return make_unique<QtTextView>(title, file); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(Window *w, const string &title, const string &text) { return make_unique<QtTextView>(title, text); }
unique_ptr<NavigationViewInterface> SystemToolkit::CreateNavigationView(Window *w, const string &style, const string &theme) { return make_unique<QtNavigationView>(w); }

}; // namespace LFL
