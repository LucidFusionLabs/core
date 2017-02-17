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

struct AndroidAlertView : public SystemAlertView {
  jobject impl;
  ~AndroidAlertView() { jni->env->DeleteGlobalRef(impl); }
  AndroidAlertView(AlertItemVec items) {
    CHECK_EQ(4, items.size());
    CHECK_EQ("style", items[0].first);
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jalert_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/util/ArrayList;)V"));
    jobject v = jni->env->NewObject(jni->jalert_class, mid, jni->activity, jni->ToJModelItemArrayList(move(items)));
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
  }

  void Show(const string &arg) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jalert_class, "showText", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;)V"));
    jstring astr = jni->ToJString(arg);
    jni->env->CallVoidMethod(impl, mid, jni->activity, astr);
    jni->env->DeleteLocalRef(astr);
  }

  string RunModal(const string &arg) { return ERRORv(string(), "not implemented"); }
  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jalert_class, "showTextCB", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Lcom/lucidfusionlabs/app/LStringCB;)V"));
    jstring tstr = jni->ToJString(title), mstr = jni->ToJString(msg), astr = jni->ToJString(arg);
    jobject cb = confirm_cb ? jni->ToLStringCB(move(confirm_cb)) : nullptr;
    jni->env->CallVoidMethod(impl, mid, jni->activity, tstr, mstr, astr, cb);
    if (cb) jni->env->DeleteLocalRef(cb);
    jni->env->DeleteLocalRef(astr);
    jni->env->DeleteLocalRef(mstr);
    jni->env->DeleteLocalRef(tstr);
  }
};

struct AndroidToolbarView : public SystemToolbarView {
  jobject impl;
  ~AndroidToolbarView() { jni->env->DeleteGlobalRef(impl); }
  AndroidToolbarView(MenuItemVec items) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtoolbar_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/util/ArrayList;)V"));
    jobject l = jni->ToJModelItemArrayList(move(items));
    jobject v = jni->env->NewObject(jni->jtoolbar_class, mid, jni->activity, l);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
    jni->env->DeleteLocalRef(l);
  }

  void ToggleButton(const string &n) {
  }

  void Show(bool show_or_hide) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtoolbar_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, show_or_hide);
  }
};

struct AndroidMenuView : public SystemMenuView {
  jobject impl;
  ~AndroidMenuView() { jni->env->DeleteGlobalRef(impl); }
  AndroidMenuView(const string &title, MenuItemVec items) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jmenu_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/util/ArrayList;)V"));
    jstring tstr = jni->ToJString(title);
    jobject l = jni->ToJModelItemArrayList(move(items));
    jobject v = jni->env->NewObject(jni->jmenu_class, mid, jni->activity, tstr, l);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
    jni->env->DeleteLocalRef(l);
    jni->env->DeleteLocalRef(tstr);
  }

  void Show() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jmenu_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, true);
  }
};

struct AndroidTableView : public SystemTableView {
  jobject impl;
  ~AndroidTableView() { jni->env->DeleteGlobalRef(impl); }
  AndroidTableView(const string &title, const string &style, TableItemVec items) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/util/ArrayList;J)V"));
    jlong lsp = intptr_t(this);
    jstring tstr = jni->ToJString(title);
    jobject l = jni->ToJModelItemArrayList(move(items));
    jobject v = jni->env->NewObject(jni->jtable_class, mid, jni->activity, tstr, l, lsp);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
    jni->env->DeleteLocalRef(l);
    jni->env->DeleteLocalRef(tstr);
  }

  void AddToolbar(SystemToolbarView *toolbar) { ERROR("not implemented"); }
  void AddNavigationButton(int halign_type, const TableItem &item) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "addNavButton", "(Lcom/lucidfusionlabs/app/MainActivity;ILcom/lucidfusionlabs/app/JModelItem;)V"));
    jobject v = jni->ToJModelItem(item);
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(halign_type), v);
    jni->env->DeleteLocalRef(v);
  }

  void DelNavigationButton(int halign_type) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "delNavButton", "(Lcom/lucidfusionlabs/app/MainActivity;I)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(halign_type));
  }

  void Show(bool show_or_hide) {
    if (show_or_hide && show_cb) show_cb();
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, show_or_hide);
  }

  string GetKey(int section, int row) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "getKey", "(Lcom/lucidfusionlabs/app/MainActivity;II)Ljava/lang/String;"));
    jstring v = jstring(jni->env->CallObjectMethod(impl, mid, jni->activity, jint(section), jint(row)));
    string ret = jni->GetJString(v);
    jni->env->DeleteLocalRef(v);
    return ret;
  }

  int GetTag(int section, int row) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "getTag", "(Lcom/lucidfusionlabs/app/MainActivity;II)I"));
    jni->env->CallIntMethod(impl, mid, jni->activity, jint(section), jint(row));
  }

  void SetTag(int section, int row, int val) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "setTag", "(Lcom/lucidfusionlabs/app/MainActivity;III)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(row), jint(val));
  }

  void SetKey(int seciton, int row, const string &key) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "setKey", "(Lcom/lucidfusionlabs/app/MainActivity;IILjava/lang/String;)V"));
    jstring kstr = jni->ToJString(key);
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(seciton), jint(row), kstr);
    jni->env->DeleteLocalRef(kstr);
  }

  void SetValue(int section, int row, const string &val) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "setValue", "(Lcom/lucidfusionlabs/app/MainActivity;IILjava/lang/String;)V"));
    jstring vstr = jni->ToJString(val);
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(row), vstr);
    jni->env->DeleteLocalRef(vstr);
  }

  void SetHidden(int section, int row, bool val) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class, "setHidden", "(Lcom/lucidfusionlabs/app/MainActivity;IIZ)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(row), jboolean(val));
  }

  void SetTitle(const string &title) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "setTitle", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;)V"));
    jstring tstr = jni->ToJString(title);
    jni->env->CallVoidMethod(impl, mid, jni->activity, tstr);
    jni->env->DeleteLocalRef(tstr);
  }

  void SelectRow(int section, int row) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class, "selectRow", "(Lcom/lucidfusionlabs/app/MainActivity;II)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(row));
  }

  StringPairVec GetSectionText(int section) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class, "getSectionText", "(Lcom/lucidfusionlabs/app/MainActivity;I)Ljava/util/ArrayList;"));
    jobject arraylist = jni->env->CallObjectMethod(impl, mid, jni->activity, section);
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

  PickerItem *GetPicker(int section, int row) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class, "getPicked", "(Lcom/lucidfusionlabs/app/MainActivity;II)Landroid/util/Pair;"));
    jobject a = jni->env->CallObjectMethod(impl, mid, jni->activity, section, row);
    if (a == nullptr) return nullptr;

    jobject al = CheckNotNull(jni->env->GetObjectField(a, jni->pair_second));
    jobject pl = CheckNotNull(jni->env->GetObjectField(a, jni->pair_first));
    intptr_t pp = CheckNotNull(jni->env->CallLongMethod(pl, jni->long_longval));
    auto picker = static_cast<PickerItem*>(Void(pp));
    if (picker->picked.size() != picker->data.size()) picker->picked.resize(picker->data.size());
    for (int i = 0, l = jni->env->CallIntMethod(al, jni->arraylist_size), l2 = picker->picked.size();
         i < l && i < l2; i++) picker->picked[i] = jni->env->CallIntMethod(al, jni->arraylist_get, i);
    jni->env->DeleteLocalRef(pl);
    jni->env->DeleteLocalRef(al);
    jni->env->DeleteLocalRef(a);
    return picker; 
  }

  void SetEditableSection(int section, int start_row, IntIntCB iicb) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class, "setEditable", "(Lcom/lucidfusionlabs/app/MainActivity;IILcom/lucidfusionlabs/app/LIntIntCB;)V"));
    jobject cb = iicb ? jni->ToLIntIntCB(move(iicb)) : nullptr;
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(start_row), cb);
    if (cb) jni->env->DeleteLocalRef(cb);
  }

  void BeginUpdates() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "beginUpdates", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity);
  }

  void EndUpdates() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "endUpdates", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity);
  }

  void AddRow(int section, TableItem item) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "addRow", "(Lcom/lucidfusionlabs/app/MainActivity;ILcom/lucidfusionlabs/app/JModelItem;)V"));
    jobject v = jni->ToJModelItem(move(item));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), v);
    jni->env->DeleteLocalRef(v);
  }

  void SetSectionValues(int section, const StringVec &in) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class,
                             "setSectionValues", "(Lcom/lucidfusionlabs/app/MainActivity;ILjava/util/ArrayList;)V"));
    jobject v = jni->ToJStringArrayList(in);
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), v);
    jni->env->DeleteLocalRef(v);
  }

  void ReplaceSection(int section, const string &h, int image, int flag, TableItemVec item, Callback add_button) {
    jstring hstr = jni->ToJString(h);
    jobject l = jni->ToJModelItemArrayList(move(item));
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtable_class, "replaceSection", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;IIILjava/util/ArrayList;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, hstr, jint(image), jint(flag), jint(section), l);
    jni->env->DeleteLocalRef(l);
    jni->env->DeleteLocalRef(hstr);
  }
};

struct AndroidTextView : public SystemTextView {
  jobject impl;
  ~AndroidTextView() { jni->env->DeleteGlobalRef(impl); }
  AndroidTextView(const string &title, File *f) : AndroidTextView(title, f ? f->Contents() : "") {}
  AndroidTextView(const string &title, const string &text) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jtextview_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/lang/String;)V"));
    jstring hstr = jni->ToJString(title), tstr = jni->ToJStringRaw(text);
    jobject v = jni->env->NewObject(jni->jtextview_class, mid, jni->activity, hstr, tstr);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
    jni->env->DeleteLocalRef(tstr);
    jni->env->DeleteLocalRef(hstr);
  }
};

struct AndroidNavigationView : public SystemNavigationView {
  jobject impl;
  ~AndroidNavigationView() { jni->env->DeleteGlobalRef(impl); }
  AndroidNavigationView() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jnavigation_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jobject v = jni->env->NewObject(jni->jnavigation_class, mid, jni->activity);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
  }

  SystemTableView *Back() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jnavigation_class,
                             "getBackTableSelf", "(Lcom/lucidfusionlabs/app/MainActivity;)J"));
    intptr_t v = jni->env->CallLongMethod(impl, mid, jni->activity);
    return v ? static_cast<SystemTableView*>(Void(v)) : nullptr;
  }

  void Show(bool show_or_hide) {
    shown == show_or_hide;
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jnavigation_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, show_or_hide);
  }

  void PushTableView(SystemTableView *t) {
    if (!root) root = t;
    if (t->show_cb) t->show_cb();
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jnavigation_class, "pushTable", "(Lcom/lucidfusionlabs/app/MainActivity;Lcom/lucidfusionlabs/app/JTable;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, dynamic_cast<AndroidTableView*>(t)->impl);
  }

  void PushTextView(SystemTextView *t) {
    if (t->show_cb) t->show_cb();
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jnavigation_class, "pushTextView", "(Lcom/lucidfusionlabs/app/MainActivity;Lcom/lucidfusionlabs/app/JTextView;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, dynamic_cast<AndroidTextView*>(t)->impl);
  }

  void PopView(int n) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jnavigation_class, "popView", "(Lcom/lucidfusionlabs/app/MainActivity;I)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(n));
  }

  void PopToRoot() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jnavigation_class, "popToRoot", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity);
  }

  void PopAll() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->jnavigation_class, "popAll", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity);
  }
};

struct AndroidAdvertisingView : public SystemAlertView {
  void Show() {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "showAds", "()V"));
    jni->env->CallVoidMethod(jni->activity, mid);
  }

  void Hide() {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "hideAds", "()V"));
    jni->env->CallVoidMethod(jni->activity, mid);
  }
};

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &cb) {}
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB &cb) {}
void Application::ShowSystemContextMenu(const MenuItemVec &items) {}

void Application::UpdateSystemImage(int n, Texture &t) {
  if (!t.buf) return;
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "updateBitmap", "(IIII[I)V"));
  int arr_size = t.BufferSize() / sizeof(jint);
  CHECK_EQ(t.width * t.height, arr_size) << t.DebugString();
  jintArray arr = jni->env->NewIntArray(arr_size);
  jni->env->SetIntArrayRegion(arr, 0, arr_size, reinterpret_cast<const jint*>(t.buf));
  jni->env->CallVoidMethod(jni->activity, mid, jint(n), jint(t.width), jint(t.height), jint(t.pf), arr);
  jni->env->DeleteLocalRef(arr);
}

int Application::LoadSystemImage(const string &n) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "getDrawableResId", "(Ljava/lang/String;)I"));
  jstring nstr = jni->ToJString(n);
  jint ret = jni->env->CallIntMethod(jni->activity, mid, nstr);
  jni->env->DeleteLocalRef(nstr);
  return ret;
}

unique_ptr<SystemMenuView> SystemMenuView::CreateEditMenu(MenuItemVec items) { return nullptr; }

}; // namespace LFL
