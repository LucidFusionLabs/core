/*
 * $Id: android_toolkit.h 770 2013-09-25 00:27:33Z justin $
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

#ifndef LFL_CORE_APP_FRAMEWORK_ANDROID_TOOLKIT_H__
#define LFL_CORE_APP_FRAMEWORK_ANDROID_TOOLKIT_H__
namespace LFL {
  
struct AndroidTableView : public TableViewInterface {
  GlobalJNIObject impl;
  AndroidTableView(const string &title, const string &style, TableItemVec items);
  static jobject NewTableScreenObject(AndroidTableView *parent, const string &title, const string &style, TableItemVec items);

  void SetTheme(const string &theme);
  void AddNavigationButton(int halign_type, const TableItem &item);
  void DelNavigationButton(int halign_type);
  void SetToolbar(ToolbarViewInterface *toolbar);
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
  void ApplyChangeList(const TableSectionInterface::ChangeList &changes);
  void SetSectionValues(int section, const StringVec &in);
  void SetSectionColors(int seciton, const vector<Color>&);
  void SetTag(int section, int row, int val);
  void SetKey(int seciton, int row, const string &key);
  void SetValue(int section, int row, const string &val);
  void SetSelected(int section, int row, int val);
  void SetHidden(int section, int row, int val);
  void SetColor(int section, int row, const Color &val);
  void SetTitle(const string &title);
  void SetSectionEditable(int section, int start_row, int skip_last_rows, IntIntCB iicb);
  void SetHeader(int section, TableItem header);
};

}; // namespace LFL
#endif // LFL_CORE_APP_FRAMEWORK_ANDROID_TOOLKIT_H__
