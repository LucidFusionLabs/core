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

NativeAlert::~NativeAlert() {}
NativeAlert::NativeAlert(const StringPairVec &items) {
  CHECK_EQ(4, items.size());
  CHECK_EQ("style", items[0].first);
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addAlert", "([Ljava/lang/String;[Ljava/lang/String;)I"));
  auto kv = ToJObjectArray(items);
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, kv.first, kv.second));
}

void NativeAlert::Show(const string &arg) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showAlert", "(ILjava/lang/String;)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jni->env->NewStringUTF(arg.c_str()));
}

NativeMenu::~NativeMenu() {}
NativeMenu::NativeMenu(const string &title, const vector<MenuItem>&items) {}
unique_ptr<NativeMenu> NativeMenu::CreateEditMenu(const vector<MenuItem> &items) { return nullptr; }
void NativeMenu::Show() {}

NativeToolbar::NativeToolbar(const StringPairVec &items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addToolbar", "([Ljava/lang/String;[Ljava/lang/String;)I"));
  auto kv = ToJObjectArray(items);
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, kv.first, kv.second));
}

NativeToolbar::~NativeToolbar() {}
void NativeToolbar::ToggleButton(const string &n) {}
void NativeToolbar::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showToolbar", "(I)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v));
}

NativeTable::~NativeTable() {}
NativeTable::NativeTable(const string &title, const vector<MenuItem> &items) {}
void NativeTable::AddToolbar(NativeToolbar*) {}
void NativeTable::Show(bool show_or_hide) {}

NativeNavigation::~NativeNavigation() {}
NativeNavigation::NativeNavigation(NativeTable *r) {}
void NativeNavigation::Show(bool show_or_hide) {}
void NativeNavigation::PushTable(NativeTable *t) {}

void Application::ShowNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd) {}
void Application::ShowNativeFileChooser(bool files, bool dirs, bool multi, const string &choose_cmd) {}

void Application::OpenSystemBrowser(const string &url_text) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "openBrowser", "(Ljava/lang/String;)V"));
  jstring jurl = jni->env->NewStringUTF(url_text.c_str());
  jni->env->CallVoidMethod(jni->activity, mid, jurl);
  jni->env->DeleteLocalRef(jurl);
}

void Application::SavePassword(const string &h, const string &u, const string &pw) {}
bool Application::LoadPassword(const string &h, const string &u, string *pw) { return false; }

void Application::ShowAds() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "showAds", "()V"));
  jni->env->CallVoidMethod(jni->activity, mid);
}

void Application::HideAds() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "hideAds", "()V"));
  jni->env->CallVoidMethod(jni->activity, mid);
}

}; // namespace LFL
