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

struct AndroidAlertView : public AlertViewInterface {
  jobject impl;
  ~AndroidAlertView() { jni->env->DeleteGlobalRef(impl); }
  AndroidAlertView(AlertItemVec items) {
    CHECK_EQ(4, items.size());
    CHECK_EQ("style", items[0].first);
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->alertscreen_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/util/ArrayList;)V"));
    jobject v = jni->env->NewObject(jni->alertscreen_class, mid, jni->activity, jni->ToModelItemArrayList(move(items)));
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
  }

  void Hide() {}
  void Show(const string &arg) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->alertscreen_class, "showText", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;)V"));
    jstring astr = jni->ToJString(arg);
    jni->env->CallVoidMethod(impl, mid, jni->activity, astr);
    jni->env->DeleteLocalRef(astr);
  }

  string RunModal(const string &arg) { return ERRORv(string(), "not implemented"); }
  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->alertscreen_class, "showTextCB", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Lcom/lucidfusionlabs/app/NativeStringCB;)V"));
    jstring tstr = jni->ToJString(title), mstr = jni->ToJString(msg), astr = jni->ToJString(arg);
    jobject cb = confirm_cb ? jni->ToNativeStringCB(move(confirm_cb)) : nullptr;
    jni->env->CallVoidMethod(impl, mid, jni->activity, tstr, mstr, astr, cb);
    if (cb) jni->env->DeleteLocalRef(cb);
    jni->env->DeleteLocalRef(astr);
    jni->env->DeleteLocalRef(mstr);
    jni->env->DeleteLocalRef(tstr);
  }
};

struct AndroidToolbarView : public ToolbarViewInterface {
  jobject impl;
  string theme;
  ~AndroidToolbarView() { jni->env->DeleteGlobalRef(impl); }
  AndroidToolbarView(MenuItemVec items) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->toolbar_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/util/ArrayList;)V"));
    jobject l = jni->ToModelItemArrayList(move(items));
    jobject v = jni->env->NewObject(jni->toolbar_class, mid, jni->activity, l);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
    jni->env->DeleteLocalRef(l);
  }

  void SetTheme(const string &x) { theme=x; }
  string GetTheme() { return theme; }

  void ToggleButton(const string &n) {
  }

  void Show(bool show_or_hide) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->toolbar_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, show_or_hide);
  }
};

struct AndroidMenuView : public MenuViewInterface {
  jobject impl;
  ~AndroidMenuView() { jni->env->DeleteGlobalRef(impl); }
  AndroidMenuView(const string &title, MenuItemVec items) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->menuscreen_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/util/ArrayList;)V"));
    jstring tstr = jni->ToJString(title);
    jobject l = jni->ToModelItemArrayList(move(items));
    jobject v = jni->env->NewObject(jni->menuscreen_class, mid, jni->activity, tstr, l);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
    jni->env->DeleteLocalRef(l);
    jni->env->DeleteLocalRef(tstr);
  }

  void Show() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->menuscreen_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, true);
  }
};

struct AndroidTableView : public TableViewInterface {
  jobject impl;
  ~AndroidTableView() { jni->env->DeleteGlobalRef(impl); }
  AndroidTableView(const string &title, const string &style, TableItemVec items) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/util/ArrayList;J)V"));
    jlong lsp = uintptr_t(this);
    jstring tstr = jni->ToJString(title);
    jobject l = jni->ToModelItemArrayList(move(items));
    jobject v = jni->env->NewObject(jni->tablescreen_class, mid, jni->activity, tstr, l, lsp);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
    jni->env->DeleteLocalRef(l);
    jni->env->DeleteLocalRef(tstr);
  }

  void AddNavigationButton(int halign_type, const TableItem &item) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "addNavButton", "(Lcom/lucidfusionlabs/app/MainActivity;ILcom/lucidfusionlabs/app/ModelItem;)V"));
    jobject v = jni->ToModelItem(item);
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(halign_type), v);
    jni->env->DeleteLocalRef(v);
  }

  void DelNavigationButton(int halign_type) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "delNavButton", "(Lcom/lucidfusionlabs/app/MainActivity;I)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(halign_type));
  }
  
  void SetToolbar(ToolbarViewInterface *toolbar) {
    ERROR("not implemented");
  }

  void Show(bool show_or_hide) {
    if (show_or_hide && show_cb) show_cb();
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, show_or_hide);
  }

  string GetKey(int section, int row) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "getKey", "(Lcom/lucidfusionlabs/app/MainActivity;II)Ljava/lang/String;"));
    jstring v = jstring(jni->env->CallObjectMethod(impl, mid, jni->activity, jint(section), jint(row)));
    string ret = jni->GetJString(v);
    jni->env->DeleteLocalRef(v);
    return ret;
  }
  
  string GetValue(int section, int row) {
    return "";
  }

  int GetTag(int section, int row) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "getTag", "(Lcom/lucidfusionlabs/app/MainActivity;II)I"));
    return jni->env->CallIntMethod(impl, mid, jni->activity, jint(section), jint(row));
  }

  PickerItem *GetPicker(int section, int row) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class, "getPicked", "(Lcom/lucidfusionlabs/app/MainActivity;II)Landroid/util/Pair;"));
    jobject a = jni->env->CallObjectMethod(impl, mid, jni->activity, section, row);
    if (a == nullptr) return nullptr;

    jobject al = CheckNotNull(jni->env->GetObjectField(a, jni->pair_second));
    jobject pl = CheckNotNull(jni->env->GetObjectField(a, jni->pair_first));
    uintptr_t pp = CheckNotNull(jni->env->CallLongMethod(pl, jni->long_longval));
    auto picker = static_cast<PickerItem*>(Void(pp));
    if (picker->picked.size() != picker->data.size()) picker->picked.resize(picker->data.size());
    for (int i = 0, l = jni->env->CallIntMethod(al, jni->arraylist_size), l2 = picker->picked.size();
         i < l && i < l2; i++) picker->picked[i] = jni->env->CallIntMethod(al, jni->arraylist_get, i);
    jni->env->DeleteLocalRef(pl);
    jni->env->DeleteLocalRef(al);
    jni->env->DeleteLocalRef(a);
    return picker; 
  }

  StringPairVec GetSectionText(int section) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class, "getSectionText", "(Lcom/lucidfusionlabs/app/MainActivity;I)Ljava/util/ArrayList;"));
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

  void BeginUpdates() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "beginUpdates", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity);
  }

  void EndUpdates() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "endUpdates", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity);
  }

  void AddRow(int section, TableItem item) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "addRow", "(Lcom/lucidfusionlabs/app/MainActivity;ILcom/lucidfusionlabs/app/ModelItem;)V"));
    jobject v = jni->ToModelItem(move(item));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), v);
    jni->env->DeleteLocalRef(v);
  }

  void SelectRow(int section, int row) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class, "selectRow", "(Lcom/lucidfusionlabs/app/MainActivity;II)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(row));
  }

  void ReplaceRow(int section, int row, TableItem item) {
  }

  void ReplaceSection(int section, TableItem header, int flag, TableItemVec item) {
    header.type = TableItem::Separator;
    jobject h = jni->ToModelItem(move(header));
    jobject l = jni->ToModelItemArrayList(move(item));
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class, "replaceSection", "(Lcom/lucidfusionlabs/app/MainActivity;ILcom/lucidfusionlabs/app/ModelItem;ILjava/util/ArrayList;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), h, jint(flag), l);
    jni->env->DeleteLocalRef(l);
    jni->env->DeleteLocalRef(h);
  }

  void ApplyChangeList(const TableSection::ChangeList &changes) {
    jobject l = jni->ToModelItemChangeList(changes);
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class, "applyChangeList", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/util/ArrayList;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, l);
    jni->env->DeleteLocalRef(l);
  }

  void SetSectionValues(int section, const StringVec &in) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "setSectionValues", "(Lcom/lucidfusionlabs/app/MainActivity;ILjava/util/ArrayList;)V"));
    jobject v = jni->ToJStringArrayList(in);
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), v);
    jni->env->DeleteLocalRef(v);
  }

  void SetSectionColors(int seciton, const vector<Color>&) {
  }

  void SetTag(int section, int row, int val) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "setTag", "(Lcom/lucidfusionlabs/app/MainActivity;III)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(row), jint(val));
  }

  void SetKey(int seciton, int row, const string &key) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "setKey", "(Lcom/lucidfusionlabs/app/MainActivity;IILjava/lang/String;)V"));
    jstring kstr = jni->ToJString(key);
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(seciton), jint(row), kstr);
    jni->env->DeleteLocalRef(kstr);
  }

  void SetValue(int section, int row, const string &val) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "setValue", "(Lcom/lucidfusionlabs/app/MainActivity;IILjava/lang/String;)V"));
    jstring vstr = jni->ToJString(val);
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(row), vstr);
    jni->env->DeleteLocalRef(vstr);
  }

  void SetSelected(int section, int row, int val) {
  }

  void SetHidden(int section, int row, bool val) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class, "setHidden", "(Lcom/lucidfusionlabs/app/MainActivity;IIZ)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(row), jboolean(val));
  }

  void SetColor(int section, int row, const Color &val) {
  }

  void SetTitle(const string &title) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class,
                             "setTitle", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;)V"));
    jstring tstr = jni->ToJString(title);
    jni->env->CallVoidMethod(impl, mid, jni->activity, tstr);
    jni->env->DeleteLocalRef(tstr);
  }
  
  void SetTheme(const string &theme) {
  }

  void SetSectionEditable(int section, int start_row, int skip_last_rows, IntIntCB iicb) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->tablescreen_class, "setEditable", "(Lcom/lucidfusionlabs/app/MainActivity;IILcom/lucidfusionlabs/app/NativeIntIntCB;)V"));
    jobject cb = iicb ? jni->ToNativeIntIntCB(move(iicb)) : nullptr;
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(section), jint(start_row), cb);
    if (cb) jni->env->DeleteLocalRef(cb);
  }

  void SetHeader(int section, TableItem header) {
  }
};

struct AndroidTextView : public TextViewInterface {
  jobject impl;
  ~AndroidTextView() { jni->env->DeleteGlobalRef(impl); }
  AndroidTextView(const string &title, File *f) : AndroidTextView(title, f ? f->Contents() : "") {}
  AndroidTextView(const string &title, const string &text) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->textviewscreen_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;Ljava/lang/String;Ljava/lang/String;)V"));
    jstring hstr = jni->ToJString(title), tstr = jni->ToJStringRaw(text);
    jobject v = jni->env->NewObject(jni->textviewscreen_class, mid, jni->activity, hstr, tstr);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
    jni->env->DeleteLocalRef(tstr);
    jni->env->DeleteLocalRef(hstr);
  }
};

struct AndroidNavigationView : public NavigationViewInterface {
  jobject impl;
  bool overlay;
  ~AndroidNavigationView() { jni->env->DeleteGlobalRef(impl); }
  AndroidNavigationView(const string &style, const string &t) : overlay(style == "overlay") {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class,
                             "<init>", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jobject v = jni->env->NewObject(jni->screennavigator_class, mid, jni->activity);
    impl = jni->env->NewGlobalRef(v);
    jni->env->DeleteLocalRef(v);
  }

  TableViewInterface *Back() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class,
                             "getBackTableSelf", "(Lcom/lucidfusionlabs/app/MainActivity;)J"));
    uintptr_t v = jni->env->CallLongMethod(impl, mid, jni->activity);
    return v ? static_cast<TableViewInterface*>(Void(v)) : nullptr;
  }

  void SetTheme(const string &theme) {}
  void Show(bool show_or_hide) {
    shown == show_or_hide;
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, show_or_hide);
    if (show_or_hide) { if (!overlay) app->SetAppFrameEnabled(false); }
    else              {               app->SetAppFrameEnabled(true);  }
  }

  void PushTableView(TableViewInterface *t) {
    if (!root) root = t;
    if (t->show_cb) t->show_cb();
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "pushTable", "(Lcom/lucidfusionlabs/app/MainActivity;Lcom/lucidfusionlabs/app/TableScreen;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, dynamic_cast<AndroidTableView*>(t)->impl);
  }

  void PushTextView(TextViewInterface *t) {
    if (t->show_cb) t->show_cb();
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "pushTextView", "(Lcom/lucidfusionlabs/app/MainActivity;Lcom/lucidfusionlabs/app/TextViewScreen;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, dynamic_cast<AndroidTextView*>(t)->impl);
  }

  void PopView(int n) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "popView", "(Lcom/lucidfusionlabs/app/MainActivity;I)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity, jint(n));
  }

  void PopToRoot() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "popToRoot", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity);
  }

  void PopAll() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->screennavigator_class, "popAll", "(Lcom/lucidfusionlabs/app/MainActivity;)V"));
    jni->env->CallVoidMethod(impl, mid, jni->activity);
  }
};

struct AndroidAdvertisingView : public AdvertisingViewInterface {
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

unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(AlertItemVec items) { return make_unique<AndroidAlertView>(move(items)); }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<ToolbarViewInterface> SystemToolkit::CreateToolbar(const string &theme, MenuItemVec items) { return make_unique<AndroidToolbarView>(move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(const string &title, MenuItemVec items) { return make_unique<AndroidMenuView>(title, move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(MenuItemVec items) { return nullptr; }
unique_ptr<TableViewInterface> SystemToolkit::CreateTableView(const string &title, const string &style, const string &theme, TableItemVec items) { return make_unique<AndroidTableView>(title, style, move(items)); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(const string &title, File *file) { return make_unique<AndroidTextView>(title, file); }
unique_ptr<TextViewInterface> SystemToolkit::CreateTextView(const string &title, const string &text) { return make_unique<AndroidTextView>(title, text); }
unique_ptr<NavigationViewInterface> SystemToolkit::CreateNavigationView(const string &style, const string &theme) { return make_unique<AndroidNavigationView>(style, theme); }

}; // namespace LFL
