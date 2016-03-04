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

#include "core/app/app.h"

namespace LFL {
void Application::AddNativeMenu(const string &title, const vector<MenuItem>&items) {
  WinWindow *win = static_cast<WinWindow*>(screen->impl);
  if (!win->menu) { win->menu = CreateMenu(); win->context_menu = CreatePopupMenu(); }
  HMENU hAddMenu = CreatePopupMenu();
  for (auto &i : items) {
    if (tuple_get<1>(i) == "<seperator>") AppendMenu(hAddMenu, MF_MENUBARBREAK, 0, NULL);
    else AppendMenu(hAddMenu, MF_STRING, win->start_msg_id + win->menu_cmds.size(), tuple_get<1>(i).c_str());
    win->menu_cmds.push_back(tuple_get<2>(i));
  }
  AppendMenu(win->menu,         MF_STRING | MF_POPUP, (UINT)hAddMenu, title.c_str());
  AppendMenu(win->context_menu, MF_STRING | MF_POPUP, (UINT)hAddMenu, title.c_str());
  if (win->menubar) SetMenu((HWND)screen->id, win->menu);
}

void Application::LaunchNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd) {
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
  cf.hwndOwner = (HWND)screen->id;
  cf.Flags = CF_SCREENFONTS | CF_INITTOLOGFONTSTRUCT;
  if (!ChooseFont(&cf)) return;
  int flag = FontDesc::Mono | (lf.lfWeight > FW_NORMAL ? FontDesc::Bold : 0) | (lf.lfItalic ? FontDesc::Italic : 0);
  screen->shell.Run(StrCat(choose_cmd, " ", lf.lfFaceName, " ", cf.iPointSize/10, " ", flag));
}

void Application::OpenSystemBrowser(const string &url_text) {
  ShellExecute(NULL, "open", url_text.c_str(), NULL, NULL, SW_SHOWNORMAL);
}

void Application::PlaySoundEffect(SoundAsset *sa) {
  audio->QueueMix(sa, MixFlag::Reset | MixFlag::Mix | (audio->loop ? MixFlag::DontQueue : 0), -1, -1);
}

void Application::PlayBackgroundMusic(SoundAsset *music) {
  audio->QueueMix(music);
  audio->loop = music;
}

}; // namespace LFL

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpCmdLine, int nCmdShow) {
  vector<const char *> av;
  vector<string> a(1);
  a[0].resize(1024);
  GetModuleFileName(hInst, &(a[0])[0], a[0].size());
  LFL::StringWordIter word_iter(lpCmdLine);
  for (string word = IterNextString(&word_iter); !word_iter.Done(); word = IterNextString(&word_iter)) a.push_back(word);
  for (auto &i : a) av.push_back(i.c_str());
  av.push_back(0);
#ifdef LFL_WINVIDEO
  LFL::WinApp *winapp = LFL::Singleton<LFL::WinApp>::Get();
  winapp->Setup(hInst, nCmdShow);
#endif
  int ret = main(av.size()-1, &av[0]);
#ifdef LFL_WINVIDEO
  return ret ? ret : winapp->MessageLoop();
#else
  return ret;
}
