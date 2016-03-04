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

extern "C" char *iPhoneDocumentPathCopy();
extern "C" void iPhoneLog(const char *text);
extern "C" void iPhoneOpenBrowser(const char *url_text);
extern "C" void iPhoneLaunchNativeMenu(const char*);
extern "C" void iPhoneCreateNativeMenu(const char*, int, const char**, const char**);
extern "C" int  iPhonePasswordCopy(const char *, const char*, const char*,       char*, int);
extern "C" bool iPhonePasswordSave(const char *, const char*, const char*, const char*, int);
extern "C" void iPhonePlayMusic(void *handle);
extern "C" void iPhonePlayBackgroundMusic(void *handle);

namespace LFL {
void Application::AddNativeMenu(const string &title, const vector<MenuItem>&items) {
  vector<const char *> n, v;
  for (auto &i : items) { n.push_back(tuple_get<1>(i).c_str()); v.push_back(tuple_get<2>(i).c_str()); }
  iPhoneCreateNativeMenu(title.c_str(), items.size(), &n[0], &v[0]);
}

void Application::LaunchNativeMenu(const string &title) {
  iPhoneLaunchNativeMenu(title.c_str());
}

void Application::OpenSystemBrowser(const string &url_text) {
  iPhoneOpenBrowser(url_text.c_str());
}

void Application::SavePassword(const string &h, const string &u, const string &pw) {
  iPhonePasswordSave(name.c_str(), h.c_str(), u.c_str(), pw.c_str(), pw.size());
}

bool Application::LoadPassword(const string &h, const string &u, string *pw) {
  pw->resize(1024);
  pw->resize(iPhonePasswordCopy(name.c_str(), h.c_str(), u.c_str(), &(*pw)[0], pw->size()));
  return pw->size();
}

void Application::PlaySoundEffect(SoundAsset *sa) {
  iPhonePlayMusic(sa->handle);
}

void Application::PlayBackgroundMusic(SoundAsset *music) {
  iPhonePlayBackgroundMusic(music->handle);
}

}; // namespace LFL
