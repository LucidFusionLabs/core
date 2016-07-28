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

void JNI::Init(jobject a, bool first) {
  if      (1)           CHECK(activity  = env->NewGlobalRef(a));
  if      (1)           CHECK(resources = env->NewGlobalRef(env->GetObjectField(activity, activity_resources)));
  if      (1)           CHECK(view      = env->NewGlobalRef(env->GetObjectField(activity, activity_view)));
  if      (first)             gplus     = env->NewGlobalRef(env->GetObjectField(activity, activity_gplus));
  else if (gplus_class) CHECK(gplus     = env->NewGlobalRef(env->GetObjectField(activity, activity_gplus)));
}

void JNI::Free() {
  if (gplus_class) env->DeleteGlobalRef(gplus);    gplus    = 0;
  if (1)           env->DeleteGlobalRef(view);     view     = 0;
  if (1)           env->DeleteGlobalRef(activity); activity = 0;
  activity_box = Box(-1, -1);
}

int JNI::CheckForException() {
  jthrowable exception = env->ExceptionOccurred();
  if (!exception) return 0;
  env->ExceptionClear();
  LogException(exception);
  return -1;
}

void JNI::LogException(jthrowable &exception) {
  static jmethodID jni_throwable_method_get_cause =
    CheckNotNull(env->GetMethodID(throwable_class, "getCause", "()Ljava/lang/Throwable;"));
  static jmethodID jni_throwable_method_get_stack_trace =
    CheckNotNull(env->GetMethodID(throwable_class, "getStackTrace", "()[Ljava/lang/StackTraceElement;"));
  static jmethodID jni_throwable_method_tostring =
    CheckNotNull(env->GetMethodID(throwable_class, "toString", "()Ljava/lang/String;"));
  static jmethodID jni_frame_method_tostring =
    CheckNotNull(env->GetMethodID(frame_class, "toString", "()Ljava/lang/String;"));

  jobjectArray frames = (jobjectArray)env->CallObjectMethod(exception, jni_throwable_method_get_stack_trace);
  jsize frames_length = env->GetArrayLength(frames);
  string out;

  if (frames > 0) {
    jstring msg = (jstring)env->CallObjectMethod(exception, jni_throwable_method_tostring);
    out += GetJString(msg);
    env->DeleteLocalRef(msg);
  }
  for (jsize i = 0; i < frames_length; i++) { 
    jobject frame = env->GetObjectArrayElement(frames, i);
    jstring msg = (jstring)env->CallObjectMethod(frame, jni_frame_method_tostring);
    out += "\n    " + GetJString(msg);
    env->DeleteLocalRef(msg);
    env->DeleteLocalRef(frame);
  }
  if (frames > 0) {
    jthrowable cause = (jthrowable)env->CallObjectMethod(exception, jni_throwable_method_get_cause);
    if (cause) LogException(cause);
  }  

  INFOf("JNI::LogException: %s", out.c_str());
}

string JNI::GetJString(jstring x) {
  const char *buf = env->GetStringUTFChars(x, 0);
  string ret = buf;
  env->ReleaseStringUTFChars(x, buf);
  return ret;
}

pair<jobjectArray, jobjectArray> JNI::ToJObjectArray(const StringPairVec& items) {
  jobjectArray k = env->NewObjectArray(items.size(), string_class, NULL);
  jobjectArray v = env->NewObjectArray(items.size(), string_class, NULL);
  for (int i=0, l=items.size(); i != l; ++i) {
    env->SetObjectArrayElement(k, i, ToJString(items[i].first));
    env->SetObjectArrayElement(v, i, ToJString(items[i].second));
  }
  return make_pair(k, v);
}

tuple<jobjectArray, jobjectArray, jobjectArray> JNI::ToJObjectArray(const MenuItemVec &items) {
  jobjectArray k = env->NewObjectArray(items.size(), string_class, NULL);
  jobjectArray v = env->NewObjectArray(items.size(), string_class, NULL);
  jobjectArray w = env->NewObjectArray(items.size(), string_class, NULL);
  for (int i=0, l=items.size(); i != l; ++i) {
    env->SetObjectArrayElement(k, i, ToJString(items[i].shortcut));
    env->SetObjectArrayElement(v, i, ToJString(items[i].name));
    env->SetObjectArrayElement(w, i, ToJString(items[i].cmd));
  }
  return make_tuple(k, v, w);
}

BufferFile *JNI::OpenAsset(const string &fn) {
  static jmethodID get_assets_mid = CheckNotNull(env->GetMethodID(activity_class, "getAssets", "()Landroid/content/res/AssetManager;"));
  static jmethodID assetmgr_open_mid = CheckNotNull(env->GetMethodID(assetmgr_class, "open", "(Ljava/lang/String;)Ljava/io/InputStream;"));
  static jmethodID inputstream_avail_mid = CheckNotNull(env->GetMethodID(inputstream_class, "available", "()I"));
  static jmethodID channels_newchan_mid = CheckNotNull(env->GetStaticMethodID(channels_class, "newChannel", "(Ljava/io/InputStream;)Ljava/nio/channels/ReadableByteChannel;"));
  static jmethodID readbytechan_read_mid = CheckNotNull(env->GetMethodID(readbytechan_class, "read", "(Ljava/nio/ByteBuffer;)I"));

  jstring jfn = ToJString(fn);
  jobject assets = env->CallObjectMethod(activity, get_assets_mid);
  jobject input = env->CallObjectMethod(assets, assetmgr_open_mid, jfn);
  env->DeleteLocalRef(jfn);
  env->DeleteLocalRef(assets);
  if (!input || CheckForException()) return nullptr;

  int len = env->CallIntMethod(input, inputstream_avail_mid);
  if (CheckForException()) { env->DeleteLocalRef(input); return nullptr; }

  unique_ptr<BufferFile> ret = make_unique<BufferFile>(string(), fn.c_str());
  if (!len) { env->DeleteLocalRef(input); return ret.release(); }
  ret->buf.resize(len);

  jobject readable = env->CallStaticObjectMethod(channels_class, channels_newchan_mid, input);
  env->DeleteLocalRef(input);

  jobject bytes = env->NewDirectByteBuffer(&ret->buf[0], ret->buf.size());
  len = env->CallIntMethod(readable, readbytechan_read_mid, bytes);
  env->DeleteLocalRef(readable);
  env->DeleteLocalRef(bytes);

  if (len != ret->buf.size() || CheckForException()) return nullptr;
  return ret.release();
}

int GetAlertWidgetID(SystemAlertWidget *w) { return int(w->impl); }
SystemAlertWidget::~SystemAlertWidget() {}
SystemAlertWidget::SystemAlertWidget(const StringPairVec &items) {
  CHECK_EQ(4, items.size());
  CHECK_EQ("style", items[0].first);
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addAlert", "([Ljava/lang/String;[Ljava/lang/String;)I"));
  auto kv = jni->ToJObjectArray(items);
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, kv.first, kv.second));
}

void SystemAlertWidget::Show(const string &arg) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showAlert", "(ILjava/lang/String;)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jni->ToJString(arg));
}

int GetToolbarWidgetID(SystemToolbarWidget *w) { return int(w->impl); }
SystemToolbarWidget::~SystemToolbarWidget() {}
SystemToolbarWidget::SystemToolbarWidget(const StringPairVec &items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addToolbar", "([Ljava/lang/String;[Ljava/lang/String;)I"));
  auto kv = jni->ToJObjectArray(items);
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, kv.first, kv.second));
}

void SystemToolbarWidget::ToggleButton(const string &n) {}
void SystemToolbarWidget::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showToolbar", "(I)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v));
}

int GetMenuWidgetID(SystemMenuWidget *w) { return int(w->impl); }
SystemMenuWidget::~SystemMenuWidget() {}
SystemMenuWidget::SystemMenuWidget(const string &title, const vector<MenuItem>&items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addMenu", "(Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;)I"));
  auto kvw = jni->ToJObjectArray(items);
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, jni->ToJString(title),
                                        tuple_get<0>(kvw), tuple_get<1>(kvw), tuple_get<2>(kvw)));
}

unique_ptr<SystemMenuWidget> SystemMenuWidget::CreateEditMenu(const vector<MenuItem> &items) { return nullptr; }
void SystemMenuWidget::Show() {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showMenu", "(I)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v));
}

int GetTableWidgetID(SystemTableWidget *w) { return int(w->impl); }
SystemTableWidget::~SystemTableWidget() {}
SystemTableWidget::SystemTableWidget(const string &title, const string &style, const vector<TableItem> &items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class,
                           "addTable", "(Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;)I"));
#if 0
  auto kvw = jni->ToJObjectArray(items);
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, jni->ToJString(title),
                                        tuple_get<0>(kvw), tuple_get<1>(kvw), tuple_get<2>(kvw)));
#endif
}

void SystemTableWidget::AddNavigationButton(const TableItem &item, int align) {}
void SystemTableWidget::SetEditableSection(int section) {}

void SystemTableWidget::AddToolbar(SystemToolbarWidget *toolbar) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "addTableToolbar", "(II)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jint(toolbar->impl.v));
}

void SystemTableWidget::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showTable", "(IZ)V"));
    jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jboolean(show_or_hide));
}

StringPairVec SystemTableWidget::GetSectionText(int section) {
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

int GetNavigationWidgetID(SystemNavigationWidget *w) { return int(w->impl); }
SystemNavigationWidget::~SystemNavigationWidget() {}
SystemNavigationWidget::SystemNavigationWidget(SystemTableWidget *r) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "addNavigation", "(I)I"));
  impl.v = Void(jni->env->CallIntMethod(jni->activity, mid, jint(r->impl.v)));
}

void SystemNavigationWidget::Show(bool show_or_hide) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showNavigation", "(IZ)V"));
    jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jboolean(show_or_hide));
}

void SystemNavigationWidget::PushTable(SystemTableWidget *t) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "pushNavigationTable", "(II)V"));
    jni->env->CallVoidMethod(jni->activity, mid, jint(impl.v), jint(t->impl.v));
}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const string &choose_cmd) {}
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const string &choose_cmd) {}

void Application::OpenSystemBrowser(const string &url_text) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "openBrowser", "(Ljava/lang/String;)V"));
  jstring jurl = jni->ToJString(url_text);
  jni->env->CallVoidMethod(jni->activity, mid, jurl);
  jni->env->DeleteLocalRef(jurl);
}

string Application::GetSystemDeviceName() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getModelName", "()Ljava/lang/String;"));
  jstring ret = (jstring)jni->env->CallObjectMethod(jni->activity, mid);
  return jni->GetJString(ret);
}

bool Application::OpenSystemAppPreferences() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "openPreferences", "()Z"));
  return jni->env->CallBooleanMethod(jni->activity, mid);
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

String16 Application::GetLocalizedString16(const char *key) { return String16(); }
string Application::GetLocalizedString(const char *key) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->resources_class, "getString", "(I)Ljava/lang/String;"));
  jfieldID fid = CheckNotNull(jni->env->GetStaticFieldID(jni->r_string_class, key, "I"));
  int resource_id = jni->env->GetStaticIntField(jni->r_string_class, fid);
  jstring ret = (jstring)jni->env->CallObjectMethod(jni->resources, mid, resource_id);
  return jni->GetJString(ret);
}

String16 Application::GetLocalizedInteger16(int number) { return String16(); }
string Application::GetLocalizedInteger(int number) {
  return StrCat(number);
}

void GPlus::SignIn() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "signIn", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
}

void GPlus::SignOut() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "signOut", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
}

int GPlus::GetSignedIn() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "signedIn", "()Z"));
    return jni->env->CallBooleanMethod(jni->gplus, mid);
  } else { ERRORf("no gplus %p", jni->gplus); return 0; }
}

int GPlus::QuickGame() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "quickGame", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
  return 0;
}

int GPlus::Invite() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "inviteGUI", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
  return 0;
}

int GPlus::Accept() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "acceptGUI", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
  return 0;
}

}; // namespace LFL
