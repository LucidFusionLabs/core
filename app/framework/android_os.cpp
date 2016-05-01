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
void Application::AddNativeMenu(const string &title, const vector<MenuItem>&items) {}
void Application::AddNativeEditMenu(const vector<MenuItem>&items) {}
void Application::LaunchNativeMenu(const string &title) {}
void Application::LaunchNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd) {}
void Application::LaunchNativeFileChooser(bool files, bool dirs, bool multi, const string &choose_cmd) {}
void Application::SavePassword(const string &h, const string &u, const string &pw) {}
bool Application::LoadPassword(const string &h, const string &u, string *pw) { return false; }

void Application::OpenSystemBrowser(const string &url_text) {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "openBrowser", "(Ljava/lang/String;)V"));
  jstring jurl = jni->env->NewStringUTF(url_text.c_str());
  jni->env->CallVoidMethod(jni->activity, mid, jurl);
  jni->env->DeleteLocalRef(jurl);
}

void Application::ShowAds() {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "showAds", "()V"));
  jni->env->CallVoidMethod(jni->activity, mid);
}

void Application::HideAds() {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "hideAds", "()V"));
  jni->env->CallVoidMethod(jni->activity, mid);
}

int Application::GetVolume() { 
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getVolume", "()I"));
  return jni->env->CallIntMethod(jni->activity, mid);
}

int Application::GetMaxVolume() {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "maxVolume", "()I"));
  return jni->env->CallIntMethod(jni->activity, mid);
}

void Application::SetVolume(int v) {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setVolume", "(I)V"));
  jint jv = v;
  return jni->env->CallVoidMethod(jni->activity, mid, jv);
}

}; // namespace LFL
