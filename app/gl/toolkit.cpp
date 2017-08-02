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

#include "core/app/gl/view.h"
#include "core/app/gl/toolkit.h"

namespace LFL {
ToolbarView::ToolbarView(Window *w, const string &t, MenuItemVec items) :
  View(w), data(move(items)), theme(t) {}

void ToolbarView::Show(bool show_or_hide) {}
void ToolbarView::ToggleButton(const string &n) {}
void ToolbarView::SetTheme(const string &t) { theme=t; }
void ToolbarView::Layout() {}
void ToolbarView::Draw() {}

TableView::TableView(Window *w, const string &t, const string &s, const string &th, TableItemVec items) :
  View(w), data(TableViewSection::Convert(move(items))), title(t), style(s), theme(th), scrollbar(this) { Activate(); }

void TableView::DelNavigationButton(int id) {}
void TableView::AddNavigationButton(int id, const TableItem &item) {}
void TableView::SetToolbar(ToolbarViewInterface*) {}
void TableView::Show(bool show_or_hide) {}
string TableView::GetKey(int section, int row) { return 0; }
string TableView::GetValue(int section, int row) { return 0; }
int TableView::GetTag(int section, int row) { return 0; }
PickerItem *TableView::GetPicker(int section, int row) { return nullptr; }
StringPairVec TableView::GetSectionText(int section) { return StringPairVec(); }
void TableView::BeginUpdates() {}
void TableView::EndUpdates() {}
void TableView::AddRow(int section, TableItem item) {}
void TableView::SelectRow(int section, int row) {}
void TableView::ReplaceRow(int section, int row, TableItem item) {}
void TableView::ReplaceSection(int section, TableItem header, int flag, TableItemVec item) {}
void TableView::ApplyChangeList(const TableSectionInterface::ChangeList&) {}
void TableView::SetSectionValues(int section, const StringVec&) {}
void TableView::SetSectionColors(int seciton, const vector<Color>&) {}
void TableView::SetSectionEditable(int section, int start_row, int skip_last_rows, IntIntCB cb) {}
void TableView::SetHeader(int section, TableItem header) {}
void TableView::SetKey(int secton, int row, const string &key) {}
void TableView::SetTag(int section, int row, int val) {}
void TableView::SetValue(int section, int row, const string &val) {}
void TableView::SetSelected(int section, int row, int selected) {}
void TableView::SetHidden(int section, int row, int val) {}
void TableView::SetColor(int section, int row, const Color &val) {}
void TableView::SetTitle(const string &title) {}
void TableView::SetTheme(const string &theme) {}

void TableView::Layout() {
  mouse.Clear();
  mouse.AddClickBox(Box(0, -box.h, box.w - scrollbar.dot_size, box.h),
                    MouseController::CoordCB(bind(&TableView::OnClick, this, _1, _2, _3, _4)));
}

void TableView::OnClick(int but, point p, point d, int down) {
  if (!down) return;
  int line_clicked = -RelativePosition(root->mouse).y / row_height, section_id = -1, row = -1;
  if (line_clicked < 0) return;
  decay_box_line = line_clicked;
  decay_box_left = 10;
  TableViewSection::FindSectionOffset(data, line_clicked, &section_id, &row);
  if (section_id < 0 || section_id >= data.size() || row < 0 || row >= data[section_id].item.size()) return;
  TableViewSection &section = data[section_id];
  TableViewItem &item = section.item[row];
  if (item.type == TableItem::TextInput) {
    app->OpenTouchKeyboard();
    if (item.textbox) item.textbox->Activate();
  }
}

void TableView::Draw() {
  View::Draw();
  if (out && decay_box_line >= 0 && decay_box_line < out->line.size() && decay_box_left > 0) {
      BoxOutline().DrawGD(root->gd, out->line[decay_box_line] + box.TopLeft());
      decay_box_left--;
  }
}

void TableView::AppendFlow(Flow *flow) {
  if (!(row_height = flow->cur_attr.font->Height())) row_height = 16;
  out = flow->out;
  flow->AppendNewline();
  for (auto &s : data)
    for (auto &i : s.item) {
      flow->AppendText(0,  i.key + ":");
      switch(i.type) {
        case TableItem::TextInput: {
          flow->AppendRow(.6, .4, &i.val_box);
        }; break;
        
        default: {
          flow->AppendText(.6, i.val);
        }; break;
      }
      flow->AppendNewlines(1);
    }
#if 0
      menuflow.AppendRow(.6, .4, &b);
      tab3_player_name.Draw(b + box.TopLeft());

      menuflow.AppendRow(.6, .35, &b);
      tab3_sensitivity.LayoutFixed(b);
      tab3_sensitivity.Update();

      menuflow.AppendRow(.6, .35, &b);
      tab3_volume.LayoutFixed(b);
      tab3_volume.Update();

      if (tab3_volume.dirty) {
        tab3_volume.dirty = false;
        app->SetVolume(int(tab3_volume.scrolled * tab3_volume.doc_height));
      }
#endif
}

NavigationView::NavigationView(Window *w, const string &s, const string &t) : View(w) {}
TableViewInterface *NavigationView::Back() { return nullptr; }
void NavigationView::Show(bool show_or_hide) {}
void NavigationView::PushTableView(TableViewInterface*) {}
void NavigationView::PushTextView(TextViewInterface*) {}
void NavigationView::PopView(int num) {}
void NavigationView::PopToRoot() {}
void NavigationView::PopAll() {}
void NavigationView::SetTheme(const string &theme) {}
void NavigationView::Layout() {}
void NavigationView::Draw() {}

unique_ptr<AlertViewInterface> Toolkit::CreateAlert(AlertItemVec items) { return Singleton<SystemToolkit>::Get()->CreateAlert(move(items)); }
unique_ptr<PanelViewInterface> Toolkit::CreatePanel(const Box &b, const string &title, PanelItemVec items) { return Singleton<SystemToolkit>::Get()->CreatePanel(b, title, move(items)); }
unique_ptr<MenuViewInterface> Toolkit::CreateMenu(const string &title, MenuItemVec items) { return Singleton<SystemToolkit>::Get()->CreateMenu(title, move(items)); }
unique_ptr<MenuViewInterface> Toolkit::CreateEditMenu(MenuItemVec items) { return Singleton<SystemToolkit>::Get()->CreateEditMenu(move(items)); }
unique_ptr<ToolbarViewInterface> Toolkit::CreateToolbar(const string &theme, MenuItemVec items, int flag) { return make_unique<ToolbarView>(app->focused, theme, move(items)); }
unique_ptr<TableViewInterface> Toolkit::CreateTableView(const string &title, const string &style, const string &theme, TableItemVec items) { return make_unique<TableView>(app->focused, title, style, theme, move(items)); }
unique_ptr<TextViewInterface> Toolkit::CreateTextView(const string &title, File *file) { return nullptr; }
unique_ptr<TextViewInterface> Toolkit::CreateTextView(const string &title, const string &text) { return nullptr; }
unique_ptr<NavigationViewInterface> Toolkit::CreateNavigationView(const string &style, const string &theme) { return make_unique<NavigationView>(app->focused, style, theme); }

}; // namespace LFL
