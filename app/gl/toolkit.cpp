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
#include "core/web/browser/browser.h"

namespace LFL {
static FreeListVector<Texture> app_images([](Texture *t) { t->Clear(); });

ToolbarView::ToolbarView(Window *w, const string &t, MenuItemVec items, Font *F, Font *SF, Color *SO, int flag) :
  View(w), theme(t), font(F), selected_font(SF), selected_outline(SO) {
  for (auto &i : items) data.emplace_back(move(i));
  if (flag) align = flag & Align::Mask;
  Activate();
}

void ToolbarView::Draw(const point &p) { View::Draw(p); }
void ToolbarView::Show(bool show_or_hide) {}
void ToolbarView::SetTheme(const string &t) { theme=t; }
void ToolbarView::ToggleButton(const string &n) {
  for (auto &i : data) if (i.shortcut == n) i.down = !i.down;
}

View *ToolbarView::Layout(Flow *flow) {
  ClearView();
  box = *flow->container;
  out = flow->out;
  for (int ind=0, l=data.size(); ind != l; ++ind) {
    auto &i = data[ind];
    if (!i.button) {
      i.button = make_unique<Widget::Button>(this, nullptr, "", MouseControllerCallback());
      i.button->outline_topleft     = &Color::grey80;
      i.button->outline_bottomright = &Color::grey40;
    }
    i.button->box = Box(box.w / data.size(), box.h);
    i.button->text = i.shortcut;
    i.button->cb = i.cb ? MouseController::CB(i.cb) : MouseControllerCallback();
    i.button->solid = theme == "Clear" ? nullptr : &Color::grey60;
    i.button->outline = (selected_outline && selected == ind) ? selected_outline : nullptr;
    i.button->Layout(flow, (selected_font && selected == ind) ? selected_font : (font ? font : flow->cur_attr.font));
  }
  return this;
}

CollectionView::CollectionView(Window *w, const string &t, const string &s, const string &th, vector<CollectionItem> items) :
  View(w), data(CollectionViewSection::Convert(move(items))), title(t), style(s), theme(th), scrollbar(this) { Activate(); }

void CollectionView::CheckExists(int section, int row) {
  CHECK_RANGE(section, 0, data.size());
  CHECK_RANGE(row, 0, data[section].item.size());
}

void CollectionView::OnClick(int but, point p, point d, int down) {}
void CollectionView::Draw(const point &p) {
  View::Draw(p);
  GraphicsContext gc(root->gd);
  if (selected_section >=0 && selected_row >= 0) {
    auto b = data[selected_section].item[selected_row].button->box + p + box.TopLeft();
    gc.gd->SetColor(Color::grey20); BoxTopLeftOutline    ().Draw(&gc, b);
    gc.gd->SetColor(Color::grey60); BoxBottomRightOutline().Draw(&gc, b);
  }
}

View *CollectionView::Layout(Flow *flow_in) {
  Flow child_flow(flow_in->container, flow_in->cur_attr.font, ResetView()), *flow = &child_flow;
  if (!(row_height = flow->cur_attr.font->Height())) row_height = 16;
  box = flow->container->RelativeCoordinatesBox();
  out = flow->out;

  auto font = flow->cur_attr.font;
  scrollbar.LayoutAttached(Box(0, -box.h, box.w, box.h - row_height));
  flow->p.y += (scrolled = int(scrollbar.scrolled * scrollbar.doc_height));
  int bpl=4, bw=box.w*2/13.0, bh=bw, sx=bw/2, px=(box.w - (bw*bpl + sx*(bpl-1)))/2;
  flow->AppendNewline();
  flow->p.x += px;

  for (auto &s : data) {
    int ind=0;
    for (auto &i : s.item) {
      if (!i.button) {
        i.button = make_unique<Widget::Button>(this, nullptr, "", MouseControllerCallback());
        i.button->outline_w           = 1;
        i.button->outline_topleft     = &Color::grey60;
        i.button->outline_bottomright = &Color::grey20;
        i.button->v_align = Align::Bottom;
      }
      i.button->box = Box(bw, bh);
      i.button->text = i.key;
      i.button->image = i.icon ? &app_images[i.icon-1] : nullptr;
      i.button->cb = i.cb ? MouseController::CB(i.cb) : MouseControllerCallback();
      i.button->Layout(flow, flow->cur_attr.font);
      flow->p.x += sx;

      if (++ind % bpl != 0) continue;
      flow->AppendNewlines(2);
      if (ind < s.item.size()) flow->p.x += px;
    }
  }

  if (toolbar) {
    Box tb_box = box; // (0, -box.h, box.w, box.h);
    if (toolbar->align & Align::Top) tb_box.y = tb_box.TopLeft().y - row_height;
    else tb_box.y = tb_box.BottomLeft().y;
    tb_box.h = row_height;
    Flow toolbarflow(&tb_box, font, dynamic_cast<ToolbarView*>(toolbar)->ResetView());
    toolbarflow.layout.align_center = 1;
    if (auto v = toolbar->Layout(&toolbarflow)) {
      v->box = tb_box;
      child_view.push_back(v);
    }
  }

  mouse.AddClickBox(Box(0, -box.h, box.w - scrollbar.dot_size, box.h),
                    MouseController::CoordCB(bind(&CollectionView::OnClick, this, _1, _2, _3, _4)));
  scrollbar.SetDocHeight(flow->Height());
  child_flow.Complete();
  return this;
}

void CollectionView::SetToolbar(ToolbarViewInterface *tb) { toolbar = tb; }
void CollectionView::Show(bool show_or_hide) {}

TableView::TableView(Window *w, const string &t, const string &s, const string &th, TableItemVec items) :
  View(w), data(TableViewSection::Convert(move(items))), title(t), style(s), theme(th), scrollbar(this) { Activate(); }

void TableView::DelNavigationButton(int id) { nav_left.type=0; }
void TableView::AddNavigationButton(int id, const TableItem &item) { nav_left = item; }
void TableView::SetToolbar(ToolbarViewInterface *tb) { toolbar = tb; }
void TableView::Show(bool show_or_hide) {}

string        TableView::GetKey   (int s, int r) { CheckExists(s, r); return data[s].item[r].key; }
string        TableView::GetValue (int s, int r) { CheckExists(s, r); return data[s].item[r].val; }
int           TableView::GetTag   (int s, int r) { CheckExists(s, r); return data[s].item[r].tag; }
PickerItem   *TableView::GetPicker(int s, int r) { CheckExists(s, r); return data[s].item[r].picker; }

StringPairVec TableView::GetSectionText(int ind) {
  StringPairVec ret;
  CHECK_RANGE(ind, 0, data.size());
  for (int i=0, l=data[ind].item.size(); i != l; i++) {
    auto &ci = data[ind].item[i];
    if (ci.dropdown_key.size()) ret.emplace_back(ci.dropdown_key, ci.key);
    ret.emplace_back(ci.key, GetValue(ind, i));
  }
  return ret;
}

void TableView::BeginUpdates() {}
void TableView::EndUpdates() {}
void TableView::AddRow(int section, TableItem item) {
  if (section == data.size()) data.emplace_back();
  CHECK_RANGE(section, 0, data.size());
  data[section].item.emplace_back(move(item));
}

void TableView::SelectRow(int section, int row) { selected_section=section; selected_row=row; }
void TableView::ReplaceRow(int s, int r, TableItem item) { CheckExists(s, r); data[s].item[r] = move(item); }

void TableView::ReplaceSection(int section, TableItem header, int flag, TableItemVec item) {
  if (section == data.size()) data.emplace_back();
  CHECK_RANGE(section, 0, data.size());
  auto &hi = data[section];
  hi.header = move(header);
  hi.flag = flag;
  hi.item.resize(item.size());
  for (int i=0, l=item.size(); i != l; ++i) hi.item[i] = move(item[i]);
}

void TableView::ApplyChangeList(const TableSectionInterface::ChangeList &changes) {
  TableViewSection::ApplyChangeList(changes, &data, [=](const LFL::TableSectionInterface::Change &d){});
}

void TableView::SetSectionValues(int section, const StringVec &v) {
  CHECK_RANGE(section, 0, data.size());
  CHECK_EQ(v.size(), data[section].item.size());
  for (int i=0, l=data[section].item.size(); i != l; ++i) SetValue(section, i, v[i]);
}

void TableView::SetSectionColors(int section, const vector<Color> &v) {
  CHECK_RANGE(section, 0, data.size());
  CHECK_EQ(v.size(), data[section].item.size());
  for (int i=0, l=data[section].item.size(); i != l; ++i) SetColor(section, i, v[i]);
}

void TableView::SetSectionEditable(int section, int start_row, int skip_last_rows, IntIntCB cb) {
  CHECK_RANGE(section, 0, data.size());
  data[section].SetEditable(start_row, skip_last_rows, move(cb));
}

void TableView::SetHeader(int s, TableItem header) { CHECK_RANGE(s, 0, data.size()); data[s].header = move(header); }
void TableView::SetKey     (int s, int r, const string &key) { CheckExists(s, r); data[s].item[r].key = key; }
void TableView::SetTag     (int s, int r, int val)           { CheckExists(s, r); data[s].item[r].tag = val; }
void TableView::SetValue   (int s, int r, const string &val) { CheckExists(s, r); data[s].item[r].val = val; }
void TableView::SetSelected(int s, int r, int selected)      { CheckExists(s, r); data[s].item[r].selected = selected; }
void TableView::SetHidden  (int s, int r, int val)           { CheckExists(s, r); data[s].item[r].hidden = val; }
void TableView::SetColor   (int s, int r, const Color &val)  { CheckExists(s, r); data[s].item[r].font.fg = val; }
void TableView::SetTitle(const string &t) { title = t; }
void TableView::SetTheme(const string &t) { theme = t; }

void TableView::CheckExists(int section, int row) {
  CHECK_RANGE(section, 0, data.size());
  CHECK_RANGE(row, 0, data[section].item.size());
}

void TableView::OnClick(int but, point p, point d, int down) {
  if (!down) return;
  int line_clicked = -p.y / row_height, section_id = -1, row = -1;
  if (line_clicked < 0) return;
  decay_box_line = line_clicked;
  decay_box_left = 10;
  TableViewSection::FindSectionOffset(data, line_clicked, &section_id, &row);
  if (section_id < 0 || section_id >= data.size() || row < 0 || row >= data[section_id].item.size()) return;
  TableViewSection &section = data[section_id];
  TableViewItem &item = section.item[row];
  if (item.type == TableItem::TextInput) {
    // app->OpenTouchKeyboard();
    if (item.textbox) item.textbox->Activate();
  }
}

void TableView::Draw(const point &p) {
  View::Draw(p);
  if (out && decay_box_line >= 0 && decay_box_line < out->line.size() && decay_box_left > 0) {
      BoxOutline().DrawGD(root->gd, out->line[decay_box_line] + p + box.TopLeft());
      decay_box_left--;
  }
  // browser.Paint(&menuflow, box.TopLeft() + point(0, 0 /*scrolled*/));
  // // browser.doc.gui.Draw();
  // browser.UpdateScrollbar();
}

View *TableView::Layout(Flow *flow_in) {
  Flow child_flow(flow_in->container, flow_in->cur_attr.font, ResetView()), *flow = &child_flow;
  if (!(row_height = flow->cur_attr.font->Height())) row_height = 16;
  box = flow->container->RelativeCoordinatesBox();
  out = flow->out;

  scrollbar.LayoutAttached(Box(0, -box.h, box.w, box.h - row_height));
  flow->p.y += (scrolled = int(scrollbar.scrolled * scrollbar.doc_height));
  flow->AppendNewline();

  for (auto &s : data)
    for (auto &i : s.item) {
      flow->AppendText(0, i.key);
      switch(i.type) {
        case TableItem::TextInput: {
          if (!i.textbox) {
            i.textbox = make_unique<TextBox>(root);
            i.textbox->style.font = flow->cur_attr.font;
            i.textbox->cursor.type = TextBox::Cursor::Underline;
            i.textbox->bg_color = Color::clear;
            i.textbox->deactivate_on_enter = true;
            i.textbox->cmd_prefix.clear();
            i.textbox->SetToggleKey(0, true);
            i.textbox->UpdateCursor();
          } else i.textbox->style.font = flow->cur_attr.font;
          i.textbox->AssignInput(i.val);
          i.textbox->runcb = i.right_cb;
          flow->AppendRow(.6, .4, &i.val_box);
          flow->out->PushBack(i.val_box, flow->cur_attr, i.textbox.get());
        }; break;

        case TableItem::Slider: {
          if (!i.slider) i.slider = make_unique<Widget::Slider>(this, Widget::Slider::Flag::Horizontal);
          flow->AppendRow(.6, .35, &i.val_box);
          i.slider->LayoutFixed(i.val_box);
          i.slider->Update();
        }; break;

        case TableItem::Selector: {
          vector<string> options;
          Split(i.val, iscomma, nullptr, &options);
          const string &v = options.size() ? options[i.selected % options.size()] : *Singleton<string>::Get();
          flow->AppendText(.6, v);
        }; break;

        case TableItem::Button: {
          if (!i.button) {
            i.button = make_unique<Widget::Button>(this, nullptr, "", MouseControllerCallback());
            i.button->outline_topleft     = &Color::grey80;
            i.button->outline_bottomright = &Color::grey40;
          }
          flow->AppendRow(.6, .35, &i.val_box);
          i.button->box = i.val_box;
          i.button->text = i.key;
          i.button->cb = i.cb ? MouseController::CB(i.cb) : MouseControllerCallback();
          i.button->solid = theme == "Clear" ? nullptr : &Color::grey60;
          i.button->Layout(flow, flow->cur_attr.font);
        }; break;

        case TableItem::WebView: {
          // if (!i.browser) i.browser = make_unique<Browser>(nullptr, nullptr, nullptr, nullptr, nullptr, this, box);
          // i.browser->Open(i.val);
        }; break;

        default: {
          flow->AppendText(.6, i.val);
        }; break;
      }
      flow->AppendNewlines(1);
    }

  scrollbar.SetDocHeight(flow->Height());
  if (toolbar) {
    Box b(box.x, box.h, box.w, row_height);
    Flow toolbarflow(&b, flow->cur_attr.font, out);
    if (auto v = toolbar->Layout(&toolbarflow)) child_view.push_back(v);
  }

#if 0
      if (tab3_volume.dirty) {
        tab3_volume.dirty = false;
        app->SetVolume(int(tab3_volume.scrolled * tab3_volume.doc_height));
      }
#endif
  mouse.AddClickBox(Box(0, -box.h, box.w - scrollbar.dot_size, box.h),
                    MouseController::CoordCB(bind(&TableView::OnClick, this, _1, _2, _3, _4)));
  child_flow.Complete();
  return this;
}

NavigationView::NavigationView(Window *w, const string &s, const string &t) : View(w) {}
TableViewInterface *NavigationView::Back() { return stack.size() ? dynamic_cast<TableViewInterface*>(stack.back()) : nullptr; }
void NavigationView::Show(bool show_or_hide) {}

void NavigationView::PushTableView(TableViewInterface *t) { if (t->show_cb) t->show_cb(); stack.push_back(t); }
void NavigationView::PushTextView (TextViewInterface  *t) { if (t->show_cb) t->show_cb(); stack.push_back(t); }

void NavigationView::PopView(int num) {
  for (int i=0; i<num && stack.size(); ++i) {
    StackViewInterface *t = stack.back();
    if (!i) t->Show(false);
    if (t->hide_cb) t->hide_cb();
    stack.pop_back();
  }
}

void NavigationView::PopToRoot() { if (stack.size() > 1) stack.resize(1);
  for (int i=0; stack.size() > 1; ++i) {
    StackViewInterface *t = stack.back();
    if (!i) t->Show(false);
    if (t->hide_cb) t->hide_cb();
    stack.pop_back();
  }
}

void NavigationView::PopAll() {
  for (int i=0; stack.size(); ++i) {
    StackViewInterface *t = stack.back();
    if (!i) t->Show(false);
    if (t->hide_cb) t->hide_cb();
    stack.pop_back();
  }
}

void NavigationView::SetTheme(const string &t) { theme = t; }
void NavigationView::Draw(const point &p) {}
View *NavigationView::Layout(Flow *flow) { return stack.size() ? stack.back()->Layout(flow) : nullptr; }

void Toolkit::UnloadImage(int n) { app_images.Erase(n-1); }
int Toolkit::LoadImage(const string &n) { return app_images.Insert(Texture(win)) + 1; }
void Toolkit::UpdateImage(int n, Texture &t) {
  CHECK_RANGE(n-1, 0, app_images.size());
  app_images[n-1] = Texture(t);
}

unique_ptr<AlertViewInterface> Toolkit::CreateAlert(Window *w, AlertItemVec items) { return system_toolkit->CreateAlert(w, move(items)); }
unique_ptr<PanelViewInterface> Toolkit::CreatePanel(Window *w, const Box &b, const string &title, PanelItemVec items) { return system_toolkit->CreatePanel(w, b, title, move(items)); }
unique_ptr<MenuViewInterface> Toolkit::CreateMenu(Window *w, const string &title, MenuItemVec items) { return system_toolkit->CreateMenu(w, title, move(items)); }
unique_ptr<MenuViewInterface> Toolkit::CreateEditMenu(Window *w, MenuItemVec items) { return system_toolkit->CreateEditMenu(w, move(items)); }
unique_ptr<ToolbarViewInterface> Toolkit::CreateToolbar(Window *w, const string &theme, MenuItemVec items, int flag) { return make_unique<ToolbarView>(w, theme, move(items), nullptr, nullptr, nullptr, flag); }
unique_ptr<CollectionViewInterface> Toolkit::CreateCollectionView(Window *w, const string &title, const string &style, const string &theme, vector<CollectionItem> items) { return make_unique<CollectionView>(w, title, style, theme, move(items)); }
unique_ptr<TableViewInterface> Toolkit::CreateTableView(Window *w, const string &title, const string &style, const string &theme, TableItemVec items) { return make_unique<TableView>(w, title, style, theme, move(items)); }
unique_ptr<TextViewInterface> Toolkit::CreateTextView(Window *w, const string &title, File *file) { return nullptr; }
unique_ptr<TextViewInterface> Toolkit::CreateTextView(Window *w, const string &title, const string &text) { return nullptr; }
unique_ptr<NavigationViewInterface> Toolkit::CreateNavigationView(Window *w, const string &style, const string &theme) { return make_unique<NavigationView>(w, style, theme); }

}; // namespace LFL
