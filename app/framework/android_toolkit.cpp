/*
 * $Id: android_toolkit.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "core/app/framework/android_toolkit.h"

namespace LFL {
static JNI *jni = Singleton<JNI>::Get();

struct AndroidAlertView : public AlertViewInterface {
  GlobalJNIObject impl;
  AndroidAlertView(AlertItemVec items) : impl(NewAlertScreenObject(move(items))) {}

  static jobject NewAlertScreenObject(AlertItemVec items) {
    CHECK_EQ(4, items.size());
    CHECK_EQ("style", items[0].first);
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->alertscreen_class, "<init>", "(Ljava/util/ArrayList;)V"));
    return jni->env->NewObject(jni->alertscreen_class, mid, JNI::ToModelItemArrayList(jni->env, move(items)));
  }

  void Hide() {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->alertscreen_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, jboolean(false));
  }

  void Show(const string &arg) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->alertscreen_class, "showText", "(Landroid/app/Activity;Ljava/lang/String;)V"));
    LocalJNIString astr(jni->env, JNI::ToJString(jni->env, arg));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, astr.v);
  }

  string RunModal(const string &arg) { return ERRORv(string(), "not implemented"); }
  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->alertscreen_class, "showTextCB", "(Landroid/app/Activity;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Lcom/lucidfusionlabs/core/NativeStringCB;)V"));
    LocalJNIString tstr(jni->env, JNI::ToJString(jni->env, title)), mstr(jni->env, JNI::ToJString(jni->env, msg)), astr(jni->env, JNI::ToJString(jni->env, arg));
    LocalJNIObject cb(jni->env, confirm_cb ? JNI::ToNativeStringCB(jni->env, move(confirm_cb)) : nullptr);
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, tstr.v, mstr.v, astr.v, cb.v);
  }
};

struct AndroidToolbarView : public ToolbarViewInterface {
  GlobalJNIObject impl;
  string theme;
  AndroidToolbarView(const string &t, MenuItemVec items, int f) :
    theme(t), impl(NewToolbarObject(t, move(items), f)) {}

  static jobject NewToolbarObject(const string &t, MenuItemVec items, int flag) {
    static jmethodID mid = jni->GetMethodID(jni->toolbar_class, Java::Constructor, {Java::String, Java::ArrayList, Java::I}, Java::V);
    LocalJNIObject theme(jni->env, JNI::ToJString(jni->env, t)), l(jni->env, JNI::ToModelItemArrayList(jni->env, move(items)));
    return jni->env->NewObject(jni->toolbar_class, mid, theme.v, l.v, jint(flag));
  }

  void SetTheme(const string &x) {
    static jmethodID mid = jni->GetMethodID(jni->toolbar_class, "setTheme", {Java::MainActivity, Java::String}, Java::V);
    LocalJNIObject name(jni->env, JNI::ToJString(jni->env, x));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, name.v);
  }

  string GetTheme() {
    static jfieldID fid = CheckNotNull(jni->env->GetFieldID(jni->toolbar_class, "theme", "Ljava/lang/String;"));
    LocalJNIString theme(jni->env, CheckNotNull(jstring(jni->env->GetObjectField(impl.v, fid))));
    return JNI::GetJString(jni->env, theme.v);
  }

  void ToggleButton(const string &n) {
    static jmethodID mid = jni->GetMethodID(jni->toolbar_class, "toggleButton", {Java::MainActivity, Java::String}, Java::V);
    LocalJNIObject name(jni->env, JNI::ToJString(jni->env, n));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, name.v);
  }

  void Show(bool show_or_hide) {
    static jmethodID mid = jni->GetMethodID(jni->toolbar_class, "show", {Java::MainActivity, Java::Z}, Java::V);
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, show_or_hide);
  }
};

struct AndroidMenuView : public MenuViewInterface {
  GlobalJNIObject impl;
  AndroidMenuView(const string &title, MenuItemVec items) : impl(NewMenuScreenObject(title, move(items))) {}

  static jobject NewMenuScreenObject(const string &title, MenuItemVec items) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->menuscreen_class, "<init>", "(Ljava/lang/String;Ljava/util/ArrayList;)V"));
    LocalJNIString tstr(jni->env, JNI::ToJString(jni->env, title));
    LocalJNIObject l(jni->env, JNI::ToModelItemArrayList(jni->env, move(items)));
    return jni->env->NewObject(jni->menuscreen_class, mid, tstr.v, l.v);
  }

  void Show() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->menuscreen_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, true);
  }
};

struct AndroidTextView : public TextViewInterface {
  GlobalJNIObject impl;
  AndroidTextView(const string &title, File *f) : AndroidTextView(title, f ? f->Contents() : "") {}
  AndroidTextView(const string &title, const string &text) : impl(NewTextScreenObject(title, text)) {}

  static jobject NewTextScreenObject(const string &title, const string &text) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->textscreen_class, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V"));
    LocalJNIString hstr(jni->env, JNI::ToJString(jni->env, title)), tstr(jni->env, JNI::ToJStringRaw(jni->env, text));
    return jni->env->NewObject(jni->textscreen_class, mid, hstr.v, tstr.v);
  }
};

struct AndroidNavigationView : public NavigationViewInterface {
  bool overlay;
  GlobalJNIObject impl;
  AndroidNavigationView(const string &style, const string &t) :
    overlay(style == "overlay"), impl(NewScreenNavigatorObject(style, t)) {}

  static jobject NewScreenNavigatorObject(const string &style, const string &t) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "<init>", "()V"));
    return jni->env->NewObject(jni->screennavigator_class, mid, jni->activity);
  }

  void SetTheme(const string &theme) {}
  TableViewInterface *Back() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "getBackTableNativeParent", "()J"));
    uintptr_t v = jni->env->CallLongMethod(impl.v, mid, jni->activity);
    return v ? static_cast<TableViewInterface*>(Void(v)) : nullptr;
  }

  void Show(bool show_or_hide) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, show_or_hide);
    if ((shown = show_or_hide)) {
      auto back = Back();
      if (back && back->show_cb) back->show_cb();
      if (!overlay) app->SetAppFrameEnabled(false);
    } else app->SetAppFrameEnabled(true);  
  }

  void PushTableView(TableViewInterface *t) {
    if (!root) root = t;
    if (t->show_cb) t->show_cb();
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "pushTable", "(Landroid/support/v7/app/AppCompatActivity;Lcom/lucidfusionlabs/app/TableScreen;)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, dynamic_cast<AndroidTableView*>(t)->impl.v);
  }

  void PushTextView(TextViewInterface *t) {
    if (t->show_cb) t->show_cb();
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "pushTextView", "(Landroid/support/v7/app/AppCompatActivity;Lcom/lucidfusionlabs/app/TextScreen;)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, dynamic_cast<AndroidTextView*>(t)->impl.v);
  }

  void PopView(int n) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "popView", "(Landroid/support/v7/app/AppCompatActivity;I)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(n));
  }

  void PopToRoot() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "popToRoot", "(Landroid/support/v7/app/AppCompatActivity;)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity);
  }

  void PopAll() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "popAll", "(Landroid/support/v7/app/AppCompatActivity;)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity);
  }
};

AndroidTableView::AndroidTableView(const string &title, const string &style, TableItemVec items) :
  impl(NewTableScreenObject(this, title, style, move(items))) {}

jobject AndroidTableView::NewTableScreenObject(AndroidTableView *parent, const string &title, const string &style, TableItemVec items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/util/ArrayList;J)V"));
  jlong np = uintptr_t(parent);
  LocalJNIString tstr(jni->env, JNI::ToJString(jni->env, title));
  LocalJNIObject l(jni->env, JNI::ToModelItemArrayList(jni->env, move(items)));
  return jni->env->NewObject(jni->tablescreen_class, mid, jni->activity, tstr.v, l.v, np);
}

void AndroidTableView::SetTheme(const string &theme) {}
void AndroidTableView::AddNavigationButton(int halign_type, const TableItem &item) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "addNavButton", "(Landroid/support/v7/app/AppCompatActivity;ILcom/lucidfusionlabs/core/ModelItem;)V"));
  LocalJNIObject v(jni->env, JNI::ToModelItem(jni->env, item));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(halign_type), v.v);
}

void AndroidTableView::DelNavigationButton(int halign_type) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "delNavButton", "(Landroid/support/v7/app/AppCompatActivity;I)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(halign_type));
}

void AndroidTableView::SetToolbar(ToolbarViewInterface *toolbar) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "setToolbar", "(Landroid/support/v7/app/AppCompatActivity;Lcom/lucidfusionlabs/core/ViewOwner;)V"));
  auto tb = dynamic_cast<AndroidToolbarView*>(toolbar);
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, tb ? tb->impl.v : nullptr);
}

void AndroidTableView::Show(bool show_or_hide) {
  if (show_or_hide && show_cb) show_cb();
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, show_or_hide);
}

string AndroidTableView::GetKey(int section, int row) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "getKey", "(Landroid/support/v7/app/AppCompatActivity;II)Ljava/lang/String;"));
  LocalJNIString v(jni->env, jstring(jni->env->CallObjectMethod(impl.v, mid, jni->activity, jint(section), jint(row))));
  return JNI::GetJString(jni->env, v.v);
}

string AndroidTableView::GetValue(int section, int row) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "getVal", "(Landroid/support/v7/app/AppCompatActivity;II)Ljava/lang/String;"));
  LocalJNIString v(jni->env, jstring(jni->env->CallObjectMethod(impl.v, mid, jni->activity, jint(section), jint(row))));
  return JNI::GetJString(jni->env, v.v);
}

int AndroidTableView::GetTag(int section, int row) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "getTag", "(Landroid/support/v7/app/AppCompatActivity;II)I"));
  return jni->env->CallIntMethod(impl.v, mid, jni->activity, jint(section), jint(row));
}

PickerItem *AndroidTableView::GetPicker(int section, int row) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "getPicked", "(Landroid/support/v7/app/AppCompatActivity;II)Landroid/util/Pair;"));
  LocalJNIObject a(jni->env, jni->env->CallObjectMethod(impl.v, mid, jni->activity, section, row));
  if (a.v == nullptr) return nullptr;

  LocalJNIObject al(jni->env, CheckNotNull(jni->env->GetObjectField(a.v, jni->pair_second)));
  LocalJNIObject pl(jni->env, CheckNotNull(jni->env->GetObjectField(a.v, jni->pair_first)));
  uintptr_t pp = CheckNotNull(jni->env->CallLongMethod(pl.v, jni->long_longval));
  auto picker = static_cast<PickerItem*>(Void(pp));
  if (picker->picked.size() != picker->data.size()) picker->picked.resize(picker->data.size());
  for (int i = 0, l = jni->env->CallIntMethod(al.v, jni->arraylist_size), l2 = picker->picked.size(); i < l && i < l2; i++) {
    LocalJNIObject ival(jni->env, jni->env->CallObjectMethod(al.v, jni->arraylist_get, i));
    picker->picked[i] = jni->env->CallIntMethod(ival.v, jni->int_intval);
  }
  return picker; 
}

StringPairVec AndroidTableView::GetSectionText(int section) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "getSectionText", "(Landroid/support/v7/app/AppCompatActivity;I)Ljava/util/ArrayList;"));
  jobject arraylist = jni->env->CallObjectMethod(impl.v, mid, jni->activity, section);
  int size = jni->env->CallIntMethod(arraylist, jni->arraylist_size);
  StringPairVec ret;
  for (int i = 0; i != size; ++i) {
    jobject pair = jni->env->CallObjectMethod(arraylist, jni->arraylist_get, i);
    LocalJNIString ki(jni->env, jstring(jni->env->GetObjectField(pair, jni->pair_first)));
    LocalJNIString vi(jni->env, jstring(jni->env->GetObjectField(pair, jni->pair_second)));
    ret.emplace_back(JNI::GetJString(jni->env, ki.v), JNI::GetJString(jni->env, vi.v));
  }
  return ret;
}

void AndroidTableView::BeginUpdates() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "beginUpdates", "(Landroid/support/v7/app/AppCompatActivity;)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity);
}

void AndroidTableView::EndUpdates() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "endUpdates", "(Landroid/support/v7/app/AppCompatActivity;)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity);
}

void AndroidTableView::AddRow(int section, TableItem item) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "addRow", "(Landroid/support/v7/app/AppCompatActivity;ILcom/lucidfusionlabs/core/ModelItem;)V"));
  LocalJNIObject v(jni->env, JNI::ToModelItem(jni->env, move(item)));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), v.v);
}

void AndroidTableView::SelectRow(int section, int row) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "selectRow", "(Landroid/support/v7/app/AppCompatActivity;II)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), jint(row));
}

void AndroidTableView::ReplaceRow(int section, int row, TableItem item) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "replaceRow", "(Landroid/support/v7/app/AppCompatActivity;IILcom/lucidfusionlabs/core/ModelItem;)V"));
  LocalJNIObject v(jni->env, JNI::ToModelItem(jni->env, move(item)));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), jint(row), v.v);
}

void AndroidTableView::ReplaceSection(int section, TableItem header, int flag, TableItemVec item) {
  header.type = TableItem::Separator;
  LocalJNIObject h(jni->env, JNI::ToModelItem(jni->env, move(header)));
  LocalJNIObject l(jni->env, JNI::ToModelItemArrayList(jni->env, move(item)));
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "replaceSection", "(Landroid/support/v7/app/AppCompatActivity;ILcom/lucidfusionlabs/core/ModelItem;ILjava/util/ArrayList;)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), h.v, jint(flag), l.v);
}

void AndroidTableView::ApplyChangeList(const TableSection::ChangeList &changes) {
  LocalJNIObject l(jni->env, JNI::ToModelItemChangeList(jni->env, changes));
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "applyChangeList", "(Landroid/support/v7/app/AppCompatActivity;Ljava/util/ArrayList;)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, l.v);
}

void AndroidTableView::SetSectionValues(int section, const StringVec &in) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "setSectionValues", "(Landroid/support/v7/app/AppCompatActivity;ILjava/util/ArrayList;)V"));
  LocalJNIObject v(jni->env, JNI::ToJStringArrayList(jni->env, in));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), v.v);
}

void AndroidTableView::SetSectionColors(int section, const vector<Color> &in) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "setSectionColors", "(Landroid/support/v7/app/AppCompatActivity;ILjava/util/ArrayList;)V"));
  vector<int> colors = VectorConvert<Color, int>(in, [](const Color &x){ return int(x.AsUnsigned()); });
  LocalJNIObject v(jni->env, JNI::ToIntegerArrayList(jni->env, colors));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), v.v);
}

void AndroidTableView::SetTag(int section, int row, int val) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "setTag", "(Landroid/support/v7/app/AppCompatActivity;III)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), jint(row), jint(val));
}

void AndroidTableView::SetKey(int seciton, int row, const string &key) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "setKey", "(Landroid/support/v7/app/AppCompatActivity;IILjava/lang/String;)V"));
  LocalJNIString kstr(jni->env, JNI::ToJString(jni->env, key));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(seciton), jint(row), kstr.v);
}

void AndroidTableView::SetValue(int section, int row, const string &val) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "setValue", "(Landroid/support/v7/app/AppCompatActivity;IILjava/lang/String;)V"));
  LocalJNIString vstr(jni->env, JNI::ToJString(jni->env, val));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), jint(row), vstr.v);
}

void AndroidTableView::SetSelected(int section, int row, int val) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "setSelected", "(Landroid/support/v7/app/AppCompatActivity;III)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), jint(row), jint(val));
}

void AndroidTableView::SetHidden(int section, int row, int val) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "setHidden", "(Landroid/support/v7/app/AppCompatActivity;III)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), jint(row), jint(val));
}

void AndroidTableView::SetColor(int section, int row, const Color &val) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "setColor", "(Landroid/support/v7/app/AppCompatActivity;I)V"));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(val.AsUnsigned()));
}

void AndroidTableView::SetTitle(const string &title) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "setTitle", "(Landroid/support/v7/app/AppCompatActivity;Ljava/lang/String;)V"));
  LocalJNIString tstr(jni->env, JNI::ToJString(jni->env, title));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, tstr.v);
}

void AndroidTableView::SetSectionEditable(int section, int start_row, int skip_last_rows, IntIntCB iicb) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class, "setEditable", "(Landroid/support/v7/app/AppCompatActivity;IILcom/lucidfusionlabs/core/NativeIntIntCB;)V"));
  LocalJNIObject cb(jni->env, iicb ? JNI::ToNativeIntIntCB(jni->env, move(iicb)) : nullptr);
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), jint(start_row), cb.v);
}

void AndroidTableView::SetHeader(int section, TableItem header) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->tablescreen_class,
                           "setHeader", "(Landroid/support/v7/app/AppCompatActivity;ILcom/lucidfusionlabs/core/ModelItem;)V"));
  LocalJNIObject v(jni->env, JNI::ToModelItem(jni->env, move(header)));
  jni->env->CallVoidMethod(impl.v, mid, jni->activity, jint(section), v.v);
}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &cb) { return ERROR(not_implemented); }
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB &cb) { return ERROR(not_implemented); }

void Application::ShowSystemContextMenu(const MenuItemVec &items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showContextMenu", "(Ljava/util/ArrayList;)V"));
  LocalJNIObject v(jni->env, JNI::ToModelItemArrayList(jni->env, move(items)));
  jni->env->CallVoidMethod(jni->activity, mid, v.v);
}

void Application::UpdateSystemImage(int n, Texture &t) {
  if (!t.buf) return;
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "updateBitmap", "(IIII[I)V"));
  int arr_size = t.BufferSize() / sizeof(jint);
  CHECK_EQ(t.width * t.height, arr_size) << t.DebugString();
  LocalJNIType<jintArray> arr(jni->env, jni->env->NewIntArray(arr_size));
  jni->env->SetIntArrayRegion(arr.v, 0, arr_size, reinterpret_cast<const jint*>(t.buf));
  jni->env->CallVoidMethod(jni->activity, mid, jint(n), jint(t.width), jint(t.height), jint(t.pf), arr.v);
}

int Application::LoadSystemImage(const string &n) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "getDrawableResId", "(Ljava/lang/String;)I"));
  LocalJNIString nstr(jni->env, JNI::ToJString(jni->env, n));
  jint ret = jni->env->CallIntMethod(jni->activity, mid, nstr.v);
  return ret;
}

void Application::UnloadSystemImage(int n) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "unloadBitmap", "(I)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(n));
}

extern "C" jobject Java_com_lucidfusionlabs_core_PickerItem_getFontPickerItem(JNIEnv *e, jclass c) {
  static GlobalJNIObject *picker_item = nullptr;
  if (picker_item == nullptr) {
    auto font_picker = new PickerItem();
    font_picker->picked.resize(2);
    font_picker->data.resize(2);
    font_picker->data[0].emplace_back("default");
    for (int i=0; i<64; i++) font_picker->data[1].emplace_back(StrCat(i+1));
    picker_item = new GlobalJNIObject(e, JNI::ToPickerItem(e, font_picker));
  }
  return picker_item->v;
}

unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(AlertItemVec items) { return make_unique<AndroidAlertView>(move(items)); }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<ToolbarViewInterface> SystemToolkit::CreateToolbar(const string &theme, MenuItemVec items, int flag) { return make_unique<AndroidToolbarView>(theme, move(items), flag); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(const string &title, MenuItemVec items) { return make_unique<AndroidMenuView>(title, move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(MenuItemVec items) { return nullptr; }
unique_ptr<TableViewInterface> SystemToolkit::CreateTableView(const string &title, const string &style, const string &theme, TableItemVec items) { return make_unique<AndroidTableView>(title, style, move(items)); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(const string &title, File *file) { return make_unique<AndroidTextView>(title, file); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(const string &title, const string &text) { return make_unique<AndroidTextView>(title, text); }
unique_ptr<NavigationViewInterface> SystemToolkit::CreateNavigationView(const string &style, const string &theme) { return make_unique<AndroidNavigationView>(style, theme); }

}; // namespace LFL
