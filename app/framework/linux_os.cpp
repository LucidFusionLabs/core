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

namespace LFL {
void Application::ShowNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd) {}
void Application::ShowNativeFileChooser(bool files, bool dirs, bool multi, const string &choose_cmd) {}
void Application::OpenSystemBrowser(const string &url_text) {}
void Application::ShowAds() {}
void Application::HideAds() {}

}; // namespace LFL
