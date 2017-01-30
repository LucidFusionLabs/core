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

SystemAlertView::~SystemAlertView() { jni->env->DeleteGlobalRef(jobject(impl.v)); }
SystemAlertView::SystemAlertView(AlertItemVec items) {
  CHECK_EQ(4, items.size());
  CHECK_EQ("style", items[0].first);
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jalert_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/util/ArrayList;)V"));
  jobject v = jni->env->NewObject(jni->jalert_class, mid, jni->activity, jni->ToJModelItemArrayList(move(items)));
  impl.v = jni->env->NewGlobalRef(v);
  jni->env->DeleteLocalRef(v);
}

void SystemAlertView::Show(const string &arg) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jalert_class, "showText", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;)V"));
  jstring astr = jni->ToJString(arg);
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, astr);
  jni->env->DeleteLocalRef(astr);
}

string SystemAlertView::RunModal(const string &arg) { return ERRORv(string(), "not implemented"); }
void SystemAlertView::ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jalert_class, "showTextCB", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;J)V"));
  jstring tstr = jni->ToJString(title), mstr = jni->ToJString(msg), astr = jni->ToJString(arg);
  jlong cb = confirm_cb ? intptr_t(new StringCB(move(confirm_cb))) : 0;
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, tstr, mstr, astr, cb);
  jni->env->DeleteLocalRef(astr);
  jni->env->DeleteLocalRef(mstr);
  jni->env->DeleteLocalRef(tstr);
}

SystemToolbarView::~SystemToolbarView() { jni->env->DeleteGlobalRef(jobject(impl.v)); }
SystemToolbarView::SystemToolbarView(MenuItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtoolbar_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/util/ArrayList;)V"));
  jobject l = jni->ToJModelItemArrayList(move(items));
  jobject v = jni->env->NewObject(jni->jtoolbar_class, mid, jni->activity, l);
  impl.v = jni->env->NewGlobalRef(v);
  jni->env->DeleteLocalRef(v);
  jni->env->DeleteLocalRef(l);
}

void SystemToolbarView::ToggleButton(const string &n) {
}

void SystemToolbarView::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtoolbar_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, show_or_hide);
}

SystemMenuView::~SystemMenuView() { jni->env->DeleteGlobalRef(jobject(impl.v)); }
SystemMenuView::SystemMenuView(const string &title, MenuItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jmenu_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/util/ArrayList;)V"));
  jstring tstr = jni->ToJString(title);
  jobject l = jni->ToJModelItemArrayList(move(items));
  jobject v = jni->env->NewObject(jni->jmenu_class, mid, jni->activity, tstr, l);
  impl.v = jni->env->NewGlobalRef(v);
  jni->env->DeleteLocalRef(v);
  jni->env->DeleteLocalRef(l);
  jni->env->DeleteLocalRef(tstr);
}

unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(MenuItemVec items) { return nullptr; }
void SystemMenuView::Show() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jmenu_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, true);
}

SystemTableView::~SystemTableView() { jni->env->DeleteGlobalRef(jobject(impl.v)); }
SystemTableView::SystemTableView(const string &title, const string &style, TableItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/util/ArrayList;J)V"));
  jlong lsp = intptr_t(this);
  jstring tstr = jni->ToJString(title);
  jobject l = jni->ToJModelItemArrayList(move(items));
  jobject v = jni->env->NewObject(jni->jtable_class, mid, jni->activity, tstr, l, lsp);
  impl.v = jni->env->NewGlobalRef(v);
  jni->env->DeleteLocalRef(v);
  jni->env->DeleteLocalRef(l);
  jni->env->DeleteLocalRef(tstr);
}

void SystemTableView::AddToolbar(SystemToolbarView *toolbar) { ERROR("not implemented"); }
void SystemTableView::AddNavigationButton(int halign_type, const TableItem &item) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "addNavButton", "(Lcom/lucidfusionlabs/app/MainActivity;ILcom/lucidfusionlabs/app/JModelItem;)V"));
  jobject v = jni->ToJModelItem(item);
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(halign_type), v);
  jni->env->DeleteLocalRef(v);
}

void SystemTableView::DelNavigationButton(int halign_type) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "delNavButton", "(Lcom/lucidfusionlabs/app/MainActivity;I)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(halign_type));
}

void SystemTableView::Show(bool show_or_hide) {
  if (show_or_hide && show_cb) show_cb();
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, show_or_hide);
}

string SystemTableView::GetKey(int section, int row) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "getKey", "(Lcom/lucidfusionlabs/app/MainActivity;II)Ljava/lang/String;"));
  jstring v = jstring(jni->env->CallObjectMethod(jobject(impl.v), mid, jni->activity, jint(section), jint(row)));
  string ret = jni->GetJString(v);
  jni->env->DeleteLocalRef(v);
  return ret;
}

int SystemTableView::GetTag(int section, int row) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "getTag", "(Lcom/lucidfusionlabs/app/MainActivity;II)I"));
  jni->env->CallIntMethod(jobject(impl.v), mid, jni->activity, jint(section), jint(row));
}

void SystemTableView::SetTag(int section, int row, int val) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "setTag", "(Lcom/lucidfusionlabs/app/MainActivity;III)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(section), jint(row), jint(val));
}

void SystemTableView::SetKey(int seciton, int row, const string &key) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "setKey", "(Lcom/lucidfusionlabs/app/MainActivity;IILjava/lang/String;)V"));
  jstring kstr = jni->ToJString(key);
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(seciton), jint(row), kstr);
  jni->env->DeleteLocalRef(kstr);
}

void SystemTableView::SetValue(int section, int row, const string &val) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "setValue", "(Lcom/lucidfusionlabs/app/MainActivity;IILjava/lang/String;)V"));
  jstring vstr = jni->ToJString(val);
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(section), jint(row), vstr);
  jni->env->DeleteLocalRef(vstr);
}

void SystemTableView::SetHidden(int section, int row, bool val) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class, "setHidden", "(Lcom/lucidfusionlabs/app/MainActivity;IIZ)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(section), jint(row), jboolean(val));
}

void SystemTableView::SetTitle(const string &title) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "setTitle", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;)V"));
  jstring tstr = jni->ToJString(title);
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, tstr);
  jni->env->DeleteLocalRef(tstr);
}

void SystemTableView::SelectRow(int section, int row) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class, "selectRow", "(Lcom/lucidfusionlabs/app/MainActivity;II)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(section), jint(row));
}

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
    jni->env->DeleteLocalRef(vi);
    jni->env->DeleteLocalRef(ki);
  }
  return ret;
}

PickerItem *SystemTableView::GetPicker(int section, int row) { return 0; }
void SystemTableView::SetEditableSection(int section, int start_row, IntIntCB iicb) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class, "setEditable", "(Lcom/lucidfusionlabs/app/MainActivity;IIJ)V"));
  jlong cb = iicb ? intptr_t(new IntIntCB(move(iicb))) : 0;
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(section), jint(start_row), cb);
}

void SystemTableView::BeginUpdates() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "beginUpdates", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity);
}

void SystemTableView::EndUpdates() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "endUpdates", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity);
}

void SystemTableView::AddRow(int section, TableItem item) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "addRow", "(Lcom/lucidfusionlabs/app/MainActivity;ILcom/lucidfusionlabs/app/JModelItem;)V"));
  jobject v = jni->ToJModelItem(move(item));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(section), v);
  jni->env->DeleteLocalRef(v);
}

void SystemTableView::SetSectionValues(int section, const StringVec &in) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class,
                           "setSectionValues", "(Lcom/lucidfusionlabs/app/MainActivity;ILjava/util/ArrayList;)V"));
  jobject v = jni->ToJStringArrayList(in);
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(section), v);
  jni->env->DeleteLocalRef(v);
}

void SystemTableView::ReplaceSection(int section, const string &h, int image, int flag, TableItemVec item, Callback add_button) {
  jstring hstr = jni->ToJString(h);
  jobject l = jni->ToJModelItemArrayList(move(item));
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtable_class, "replaceSection", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;IIILjava/util/ArrayList;)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, hstr, jint(image), jint(flag), jint(section), l);
  jni->env->DeleteLocalRef(l);
  jni->env->DeleteLocalRef(hstr);
}

SystemTextView::~SystemTextView() { jni->env->DeleteGlobalRef(jobject(impl.v)); }
SystemTextView::SystemTextView(const string &title, File *f) : SystemTextView(title, f ? f->Contents() : "") {}
SystemTextView::SystemTextView(const string &title, const string &text) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jtextview_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/lang/String;)V"));
  jstring hstr = jni->ToJString(title), tstr = jni->ToJStringRaw(text);
  jobject v = jni->env->NewObject(jni->jtextview_class, mid, jni->activity, hstr, tstr);
  impl.v = jni->env->NewGlobalRef(v);
  jni->env->DeleteLocalRef(v);
  jni->env->DeleteLocalRef(tstr);
  jni->env->DeleteLocalRef(hstr);
}

SystemNavigationView::~SystemNavigationView() { jni->env->DeleteGlobalRef(jobject(impl.v)); }
SystemNavigationView::SystemNavigationView() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class,
                           "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
  jobject v = jni->env->NewObject(jni->jnavigation_class, mid, jni->activity);
  impl.v = jni->env->NewGlobalRef(v);
  jni->env->DeleteLocalRef(v);
}

SystemTableView *SystemNavigationView::Back() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class,
                           "getBackTableSelf", "(Lcom/lucidfusionlabs/app/MainActivity;)J"));
  intptr_t v = jni->env->CallLongMethod(jobject(impl.v), mid, jni->activity);
  return v ? static_cast<SystemTableView*>(Void(v)) : nullptr;
}

void SystemNavigationView::Show(bool show_or_hide) {
  shown == show_or_hide;
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, show_or_hide);
}

void SystemNavigationView::PushTableView(SystemTableView *t) {
  if (!root) root = t;
  if (t->show_cb) t->show_cb();
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class, "pushTable", "(Lcom/lucidfusionlabs/app/MainActivity;Lcom/lucidfusionlabs/app/JTable;)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jobject(t->impl.v));
}

void SystemNavigationView::PushTextView(SystemTextView *t) {
  if (t->show_cb) t->show_cb();
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class, "pushTextView", "(Lcom/lucidfusionlabs/app/MainActivity;Lcom/lucidfusionlabs/app/JTextView;)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jobject(t->impl.v));
}

void SystemNavigationView::PopView(int n) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class, "popView", "(Lcom/lucidfusionlabs/app/MainActivity;I)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity, jint(n));
}

void SystemNavigationView::PopToRoot() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class, "popToRoot", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity);
}

void SystemNavigationView::PopAll() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->jnavigation_class, "popAll", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
  jni->env->CallVoidMethod(jobject(impl.v), mid, jni->activity);
}

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

void Application::UpdateSystemImage(int n, Texture &t) {}
int Application::LoadSystemImage(const string &n) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "getDrawableResId", "(Ljava/lang/String;)I"));
  jstring nstr = jni->ToJString(n);
  jint ret = jni->env->CallIntMethod(jni->activity, mid, nstr);
  jni->env->DeleteLocalRef(nstr);
  return ret;
}

}; // namespace LFL
