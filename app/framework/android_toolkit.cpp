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
    (jni->env->GetMethodID(jni->jalert_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/util/ArrayList;)V"));
  impl.v = jni->env->NewObject(jni->jalert_class, mid, jni->activity, jni->ToJModelItemArrayList(move(items)));
}

void SystemAlertView::Show(const string &arg) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jalert_class, "showText", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jni->ToJString(arg));
}

void SystemAlertView::ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {}
string SystemAlertView::RunModal(const string &arg) { return ""; }

int GetToolbarViewID(SystemToolbarView *w) { return int(w->impl); }
SystemToolbarView::~SystemToolbarView() {}
SystemToolbarView::SystemToolbarView(MenuItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtoolbar_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/util/ArrayList;)V"));
  impl.v = jni->env->NewObject(jni->jtoolbar_class, mid, jni->activity, jni->ToJModelItemArrayList(move(items)));
}

void SystemToolbarView::ToggleButton(const string &n) {}
void SystemToolbarView::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtoolbar_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, true);
}

int GetMenuViewID(SystemMenuView *w) { return int(w->impl); }
SystemMenuView::~SystemMenuView() {}
SystemMenuView::SystemMenuView(const string &title, MenuItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jmenu_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/util/ArrayList;)V"));
  impl.v = jni->env->NewObject(jni->jmenu_class, mid, jni->activity, jni->ToJString(title), jni->ToJModelItemArrayList(move(items)));
}

unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(MenuItemVec items) { return nullptr; }
void SystemMenuView::Show() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jmenu_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, true);
}

int GetTableViewID(SystemTableView *w) { return int(w->impl); }
SystemTableView::~SystemTableView() {}
SystemTableView::SystemTableView(const string &title, const string &style, TableItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/util/ArrayList;)V"));
  impl.v = jni->env->NewObject(jni->jtable_class, mid, jni->activity, jni->ToJString(title),
                               jni->ToJModelItemArrayList(move(items)));
}

void SystemTableView::DelNavigationButton(int) {}
void SystemTableView::AddNavigationButton(int, const TableItem &item) {}
void SystemTableView::AddToolbar(SystemToolbarView *toolbar) { ERROR("not implemented"); }

void SystemTableView::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, true);
}

string SystemTableView::GetKey(int section, int row) { return ""; }
int SystemTableView::GetTag(int section, int row) { return 0; }
void SystemTableView::SetTag(int section, int row, int val) {}
void SystemTableView::SetKey(int seciton, int row, const string &key) {}
void SystemTableView::SetValue(int section, int row, const string &val) {}
void SystemTableView::SetHidden(int section, int row, bool val) {}
void SystemTableView::SetTitle(const string &title) {}
PickerItem *SystemTableView::GetPicker(int section, int row) { return 0; }
void SystemTableView::SelectRow(int section, int row) {}

StringPairVec SystemTableView::GetSectionText(int section) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class, "getSectionText", "(Lcom/lucidfusionlabs/app/MainActivity;I)Ljava/util/ArrayList;"));
  jobject arraylist = jni->env->CallObjectMethod(jobject(impl.v), mid, jni->activity, section);
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

void SystemTableView::SetEditableSection(int section, int start_row, IntIntCB cb) {}

void SystemTableView::BeginUpdates() {}
void SystemTableView::EndUpdates() {}
void SystemTableView::AddRow(int section, TableItem item) {}
void SystemTableView::SetSectionValues(int section, const StringVec&) {}
void SystemTableView::ReplaceSection(int section, const string &h, int image, int flag, TableItemVec item, Callback add_button) {}

SystemTextView::~SystemTextView() {}
SystemTextView::SystemTextView(const string &title, File *f) : SystemTextView(title, f ? f->Contents() : "") {}
SystemTextView::SystemTextView(const string &title, const string &text) {}

int GetNavigationViewID(SystemNavigationView *w) { return int(w->impl); }
SystemNavigationView::~SystemNavigationView() {}
SystemNavigationView::SystemNavigationView() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
  impl.v = jni->env->NewObject(jni->jnavigation_class, mid, jni->activity);
}

SystemTableView *SystemNavigationView::Back() { return nullptr; }

void SystemNavigationView::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, true);
}

void SystemNavigationView::PushTableView(SystemTableView *t) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class, "pushTable", "(Lcom/lucidfusionlabs/app/MainActivity;Lcom/lucidfusionlabs/app/JTable;)V"));
    jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jobject(t->impl.v));
}

void SystemNavigationView::PushTextView(SystemTextView*) {}
void SystemNavigationView::PopView(int n) {}
void SystemNavigationView::PopToRoot() {}
void SystemNavigationView::PopAll() {}

SystemAdvertisingView::SystemAdvertisingView() {}
void SystemAdvertisingView::Show() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "showAds", "()V"));
  jni->env->CallVoidMethod(jni->activity, mid);
}

void SystemAdvertisingView::Hide() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "hideAds", "()V"));
  jni->env->CallVoidMethod(jni->activity, mid);
}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &cb) {}
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB &cb) {}
void Application::ShowSystemContextMenu(const MenuItemVec &items) {}

int Application::LoadSystemImage(const string &n) { return 1; }
void Application::UpdateSystemImage(int n, Texture &t) {}

}; // namespace LFL
