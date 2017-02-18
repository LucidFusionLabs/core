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

#include <shellapi.h>
#include <commdlg.h>

namespace LFL {
SystemMenuView::~SystemMenuView() {}
SystemMenuView::SystemMenuView(const string &title_text, const vector<MenuItem>&items) {
  WinWindow *win = dynamic_cast<WinWindow*>(screen);
  if (!win->menu) { win->menu = CreateMenu(); win->context_menu = CreatePopupMenu(); }
  HMENU hAddMenu = CreatePopupMenu();
  for (auto &i : items) {
    if (i.name == "<separator>") AppendMenu(hAddMenu, MF_MENUBARBREAK, 0, NULL);
    else AppendMenu(hAddMenu, MF_STRING, win->start_msg_id + win->menu_cmds.size(), i.name.c_str());
    win->menu_cmds.push_back(i.cmd);
  }
  AppendMenu(win->menu, MF_STRING | MF_POPUP, (UINT)hAddMenu, title.c_str());
  AppendMenu(win->context_menu, MF_STRING | MF_POPUP, (UINT)hAddMenu, title.c_str());
  if (win->menubar) SetMenu(dynamic_cast<WinWindow*>(screen)->hwnd, win->menu);
}

unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(const vector<MenuItem>&items) { return nullptr; }

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const string &choose_cmd) {
  LOGFONT lf;
  memzero(lf);
  HDC hdc = GetDC(NULL);
  lf.lfHeight = -MulDiv(cur_font.size, GetDeviceCaps(hdc, LOGPIXELSY), 72);
  lf.lfWeight = (cur_font.flag & FontDesc::Bold) ? FW_BOLD : FW_NORMAL;
  lf.lfItalic = cur_font.flag & FontDesc::Italic;
  strncpy(lf.lfFaceName, cur_font.name.c_str(), sizeof(lf.lfFaceName)-1);
  ReleaseDC(NULL, hdc);
  CHOOSEFONT cf;
  memzero(cf);
  cf.lpLogFont = &lf;
  cf.lStructSize = sizeof(cf);
  cf.hwndOwner = dynamic_cast<WinWindow*>(screen)->hwnd;
  cf.Flags = CF_SCREENFONTS | CF_INITTOLOGFONTSTRUCT;
  if (!ChooseFont(&cf)) return;
  int flag = FontDesc::Mono | (lf.lfWeight > FW_NORMAL ? FontDesc::Bold : 0) | (lf.lfItalic ? FontDesc::Italic : 0);
  screen->shell->Run(StrCat(choose_cmd, " ", lf.lfFaceName, " ", cf.iPointSize/10, " ", flag));
}

}; // namespace LFL
