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

int GetAlertViewID(SystemAlertView *w) { return int(w->impl); }
SystemAlertView::~SystemAlertView() {}
SystemAlertView::SystemAlertView(AlertItemVec items) {
  CHECK_EQ(4, items.size());
  CHECK_EQ("style", items[0].first);
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addAlert", "([Ljava/lang/String;[Ljava/lang/String;)I"));
  // auto kv = jni->ToJObjectArray(items);
  // impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, kv.first, kv.second));
}

void SystemAlertView::Show(const string &arg) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showAlert", "(ILjava/lang/String;)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jni->ToJString(arg));
}

int GetToolbarViewID(SystemToolbarView *w) { return int(w->impl); }
SystemToolbarView::~SystemToolbarView() {}
SystemToolbarView::SystemToolbarView(MenuItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addToolbar", "([Ljava/lang/String;[Ljava/lang/String;)I"));
  auto kv = jni->ToJObjectArray(items);
  // impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, kv.first, kv.second));
}

void SystemToolbarView::ToggleButton(const string &n) {}
void SystemToolbarView::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showToolbar", "(I)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v));
}

int GetMenuViewID(SystemMenuView *w) { return int(w->impl); }
SystemMenuView::~SystemMenuView() {}
SystemMenuView::SystemMenuView(const string &title, MenuItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addMenu", "(Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;)I"));
  auto kvw = jni->ToJObjectArray(items);
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, jni->ToJString(title),
                                        tuple_get<0>(kvw), tuple_get<1>(kvw), tuple_get<2>(kvw)));
}

unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(MenuItemVec items) { return nullptr; }
void SystemMenuView::Show() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showMenu", "(I)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v));
}

int GetTableViewID(SystemTableView *w) { return int(w->impl); }
SystemTableView::~SystemTableView() {}
SystemTableView::SystemTableView(const string &title, const string &style, TableItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addTable", "(Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;)I"));
#if 0
  auto kvw = jni->ToJObjectArray(items);
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, jni->ToJString(title),
                                        tuple_get<0>(kvw), tuple_get<1>(kvw), tuple_get<2>(kvw)));
#endif
}

void SystemTableView::AddNavigationButton(const TableItem &item, int align) {}
//void SystemTableView::SetEditableSection(int section) {}

void SystemTableView::AddToolbar(SystemToolbarView *toolbar) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "addTableToolbar", "(II)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jint(toolbar->impl.v));
}

void SystemTableView::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showTable", "(IZ)V"));
    jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jboolean(show_or_hide));
}

StringPairVec SystemTableView::GetSectionText(int section) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "getTableSectionText", "(II)Ljava/util/ArrayList;"));
  jobject arraylist = jni->env->CallObjectMethod(jni->activity, mid, jint(impl.v), section);
  int size = jni->env->CallIntMethod(arraylist, jni->arraylist_size);
  StringPairVec ret;
  for (int i = 0; i != size; ++i) {
    jobject pair = jni->env->CallObjectMethod(arraylist, jni->arraylist_get, i);
    jstring ki = (jstring)jni->env->GetObjectField(pair, jni->pair_first);
    jstring vi = (jstring)jni->env->GetObjectField(pair, jni->pair_second);
    ret.emplace_back(jni->GetJString(ki), jni->GetJString(vi));
  }
  return ret;
}

void SystemTableView::SetSectionValues(const StringVec&, int section) {}
void SystemTableView::ReplaceSection(TableItemVec item, int section) {}

int GetNavigationViewID(SystemNavigationView *w) { return int(w->impl); }
SystemNavigationView::~SystemNavigationView() {}
SystemNavigationView::SystemNavigationView(SystemTableView *r) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "addNavigation", "(I)I"));
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, jint(r->impl.v)));
}

void SystemNavigationView::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showNavigation", "(IZ)V"));
    jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jboolean(show_or_hide));
}

void SystemNavigationView::PushTable(SystemTableView *t) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "pushNavigationTable", "(II)V"));
    jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jint(t->impl.v));
}

void SystemNavigationView::PopTable(int n) {}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &cb) {}
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB &cb) {}

}; // namespace LFL
