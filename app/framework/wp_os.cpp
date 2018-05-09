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

#include "LocalizationComponent.h"
#include "CCWinRTUtils.h"

namespace LFL {
static inline String16 GetWinRTString(Platform::String^ str) { return str->Data(); }

SystemMenuView::~SystemMenuView() {}
SystemMenuView::SystemMenuView(const string &title_text, const vector<MenuItem>&items) {}
unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(const vector<MenuItem>&items) { return nullptr; }

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const string &choose_cmd) {
void Application::OpenSystemBrowser(const string &url_text) {}

void Application::ShowAds() {}
void Application::HideAds() {}
 
string Application::GetLocalizedString(const char *key) { return string(); }
String16 Application::GetLocalizedString16(const char *key) {
  std::wstring keyString = key;
  Platform::String^ str = ref new Platform::String(keyString.data(), keyString.length());
  str = PhoneDirect3DXamlAppComponent::LocalizationComponent::Instance->GetLocalizedString(str);
  return GetWinRTString(str);
}
 
string Application::GetLocalizedInteger(int number) { return string(); }
String16 Application::GetLocalizedInteger16(int number) {
  Platform::String^ str = PhoneDirect3DXamlAppComponent::LocalizationComponent::Instance
    ->GetFormattedNumber(number);
  return GetWinRTString(str);
}

}; // namespace LFL
