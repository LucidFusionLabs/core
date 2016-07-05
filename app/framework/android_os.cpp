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
static JNI *jni = Singleton<JNI>::Get();
static pair<jobjectArray, jobjectArray> ToJObjectArray(const vector<pair<string, string>>&items) {
  jobjectArray k = jni->env->NewObjectArray(items.size(), jni->string_class, NULL);
  jobjectArray v = jni->env->NewObjectArray(items.size(), jni->string_class, NULL);
  for (int i=0, l=items.size(); i != l; ++i) {
    jni->env->SetObjectArrayElement(k, i, jni->env->NewStringUTF(items[i].first .c_str()));
    jni->env->SetObjectArrayElement(v, i, jni->env->NewStringUTF(items[i].second.c_str()));
  }
  return make_pair(k, v);
}

void Application::AddNativeAlert(const string &name, const vector<pair<string, string>>&items) {
  CHECK_EQ(4, items.size());
  CHECK_EQ("style", items[0].first);
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addAlert", "(Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;)V"));
  auto kv = ToJObjectArray(items);
  jni->env->CallVoidMethod(jni->activity, mid, jni->env->NewStringUTF(name.c_str()), kv.first, kv.second);
}

void Application::LaunchNativeAlert(const string &name, const string &arg) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showAlert", "(Ljava/lang/String;)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jni->env->NewStringUTF(name.c_str()),
                           jni->env->NewStringUTF(arg.c_str()));
}

void Application::AddNativeMenu(const string &title, const vector<MenuItem>&items) {}
void Application::AddNativeEditMenu(const vector<MenuItem>&items) {}
void Application::LaunchNativeMenu(const string &title) {}
void Application::LaunchNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd) {}
void Application::LaunchNativeFileChooser(bool files, bool dirs, bool multi, const string &choose_cmd) {}

void Application::AddToolbar(const string &title, const vector<pair<string, string>>&items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addToolbar", "(Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;)V"));
  auto kv = ToJObjectArray(items);
  jni->env->CallVoidMethod(jni->activity, mid, jni->env->NewStringUTF(title.c_str()), kv.first, kv.second);
}

void Application::ToggleToolbarButton(const string&, const string &n) {}
void Application::ShowToolbar(const string &title, bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showToolbar", "(Ljava/lang/String;)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jni->env->NewStringUTF(title.c_str()));
}

void Application::AddNativeTable(const string &title, const vector<MenuItem> &items) {}
void Application::LaunchNativeTable(const string &title) {}

void Application::SavePassword(const string &h, const string &u, const string &pw) {}
bool Application::LoadPassword(const string &h, const string &u, string *pw) { return false; }

void Application::OpenSystemBrowser(const string &url_text) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "openBrowser", "(Ljava/lang/String;)V"));
  jstring jurl = jni->env->NewStringUTF(url_text.c_str());
  jni->env->CallVoidMethod(jni->activity, mid, jurl);
  jni->env->DeleteLocalRef(jurl);
}

void Application::ShowAds() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "showAds", "()V"));
  jni->env->CallVoidMethod(jni->activity, mid);
}

void Application::HideAds() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "hideAds", "()V"));
  jni->env->CallVoidMethod(jni->activity, mid);
}

}; // namespace LFL
