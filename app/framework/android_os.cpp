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
#include "core/app/bindings/jni.h"

namespace LFL {
void Application::OpenSystemBrowser(const string &url_text) {
  AndroidOpenBrowser(url_text.c_str());
}

void Application::ShowAds() {
  AndroidShowAds();
}

void Application::HideAds() {
  AndroidHideAds();
}

int Application::GetVolume() { 
  return AndroidGetVolume();
}

int Application::GetMaxVolume() { 
  return AndroidGetMaxVolume();
}

void Application::SetVolume(int v) { 
  AndroidSetVolume(v);
}

void Application::PlaySoundEffect(SoundAsset *sa) {
  AndroidPlayMusic(sa->handle);
}

void Application::PlayBackgroundMusic(SoundAsset *music) {
  AndroidPlayBackgroundMusic(music->handle);
}


}; // namespace LFL
