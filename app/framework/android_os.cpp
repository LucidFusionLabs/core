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
  // if      (first)             gplus     = env->NewGlobalRef(env->GetObjectField(activity, activity_gplus));
  // else if (gplus_class) CHECK(gplus     = env->NewGlobalRef(env->GetObjectField(activity, activity_gplus)));
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
    LocalJNIString msg(env, (jstring)env->CallObjectMethod(exception, jni_throwable_method_tostring));
    out += GetJString(msg.v);
  }
  for (jsize i = 0; i < frames_length; i++) { 
    LocalJNIObject frame(env, env->GetObjectArrayElement(frames, i));
    LocalJNIString msg(env, (jstring)env->CallObjectMethod(frame.v, jni_frame_method_tostring));
    out += "\n    " + GetJString(msg.v);
  }
  if (frames > 0) {
    jthrowable cause = (jthrowable)env->CallObjectMethod(exception, jni_throwable_method_get_cause);
    if (cause) LogException(cause);
  }  

  INFOf("JNI::LogException: %s", out.c_str());
}

string JNI::GetEnvJString(JNIEnv *e, jstring x) {
  const char *buf = e->GetStringUTFChars(x, 0);
  string ret = buf;
  e->ReleaseStringUTFChars(x, buf);
  return ret;
}

jstring JNI::ToJStringRaw(const string &x) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V"));
  LocalJNIType<jbyteArray> array(env, env->NewByteArray(x.size()));
  env->SetByteArrayRegion(array.v, 0, x.size(), reinterpret_cast<const jbyte*>(x.data()));
  LocalJNIString encoding(env, env->NewStringUTF("UTF-8"));
  return jstring(env->NewObject(string_class, mid, array.v, encoding.v));
}

jobjectArray JNI::ToJStringArray(const StringVec &items) {
  jobjectArray v = env->NewObjectArray(items.size(), string_class, NULL);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIString vi(env, ToJString(items[i]));
    env->SetObjectArrayElement(v, i, vi.v);
  }
  return v;
}

pair<jobjectArray, jobjectArray> JNI::ToJStringArrays(const StringPairVec &items) {
  jobjectArray k = env->NewObjectArray(items.size(), string_class, NULL);
  jobjectArray v = env->NewObjectArray(items.size(), string_class, NULL);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIString ki(env, ToJString(items[i].first)), vi(env, ToJString(items[i].second));
    env->SetObjectArrayElement(k, i, ki.v);
    env->SetObjectArrayElement(v, i, vi.v);
  }
  return make_pair(k, v);
}

jobject JNI::ToJStringArrayList(const StringVec &items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (auto &i : items) {
    LocalJNIObject v(env, ToJString(i));
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToJStringPairArrayList(const StringPairVec &items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (auto &i : items) {
    LocalJNIString ki(env, ToJString(i.first)), vi(env, ToJString(i.second));
    LocalJNIObject v(env, env->NewObject(pair_class, jni->pair_construct, ki.v, vi.v));
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItemArrayList(AlertItemVec items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIObject v(env, ToModelItem(items[i]));
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItemArrayList(MenuItemVec items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIObject v(env, ToModelItem(items[i]));
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItemArrayList(TableItemVec items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIObject v(env, ToModelItem(items[i]));
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItemChangeList(const TableSection::ChangeList &items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (auto &i : items) {
    LocalJNIObject v(env, ToModelItemChange(i));
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItem(AlertItem item) {
  jboolean hidden=0;
  LocalJNIString k(env, ToJString(item.first)), v(env, ToJString(item.second)), rt(env, ToJString("")), ddk(env, ToJString(""));
  LocalJNIObject rcb(env, item.cb ? ToNativeStringCB(move(item.cb)) : nullptr);
  jobject lcb = nullptr, picker = nullptr;
  jint type=0, tag=0, flags=0, left_icon=0, right_icon=0, selected=0, height=0, fg=0, bg=0;
  return env->NewObject(modelitem_class, modelitem_construct, k.v, v.v, rt.v, ddk.v, type,
                        tag, flags, left_icon, right_icon, selected, height, lcb, rcb.v, picker,
                        hidden, fg, bg);
}

jobject JNI::ToModelItem(MenuItem item) {
  LocalJNIString k(env, ToJString(item.shortcut)), v(env, ToJString(item.name)), rt(env, ToJString("")), ddk(env, ToJString(""));
  LocalJNIObject cb(env, item.cb ? ToNativeCallback(move(item.cb)) : nullptr);
  return env->NewObject(modelitem_class, modelitem_construct, k.v, v.v, rt.v, ddk.v, jint(0),
                        jint(0), jint(0), jint(0), jint(0), jint(0), jint(0), cb.v, nullptr, nullptr,
                        false, jint(0), jint(0));
}

jobject JNI::ToModelItem(TableItem item) {
  jint fg = (item.fg_a << 24) | (item.fg_r << 16) | (item.fg_g << 8) | item.fg_b;
  jint bg = (item.bg_a << 24) | (item.bg_r << 16) | (item.bg_g << 8) | item.bg_b;
  LocalJNIObject k(env, ToJString(item.key)), v(env, ToJString(item.val)), rt(env, ToJString(item.right_text)),
          ddk(env, ToJString(item.dropdown_key)), picker(env, ToPickerItem(item.picker));
  LocalJNIObject cb(env, item.cb ? ToNativeCallback(move(item.cb)) : nullptr);
  LocalJNIObject rcb(env, item.right_cb ? ToNativeStringCB(move(item.right_cb)) : nullptr);
  return env->NewObject(modelitem_class, modelitem_construct, k.v, v.v, rt.v, ddk.v, jint(item.type),
                        jint(item.tag), jint(item.flags), jint(item.left_icon), jint(item.right_icon),
                        jint(item.selected), jint(item.height), cb.v, rcb.v, picker.v, jboolean(item.hidden),
                        fg, bg);
}

jobject JNI::ToModelItemChange(const TableSection::Change &item) {
  LocalJNIObject k(env, ToJString(item.key)), v(env, ToJString(item.val));
  LocalJNIObject cb(env, item.cb ? ToNativeCallback(item.cb) : nullptr);
  return env->NewObject(modelitemchange_class, modelitemchange_construct, jint(item.section), jint(item.row),
                        jint(item.type), k.v, v.v, jint(item.left_icon), jint(item.right_icon),
                        jint(item.flags), jboolean(item.hidden), cb.v);
}

jobject JNI::ToPickerItem(PickerItem *picker_in) {
  if (!picker_in) return nullptr;
  const PickerItem *picker = picker_in;
  jlong self = uintptr_t(picker_in);
  LocalJNIObject cb(env, picker->cb ? ToNativePickerItemCB(picker->cb) : nullptr);
  LocalJNIObject l(env, env->NewObject(arraylist_class, arraylist_construct));
  for (auto &i : picker->data) {
    LocalJNIObject v(env, ToJStringArrayList(i));
    CHECK(env->CallBooleanMethod(l.v, arraylist_add, v.v));
  }
  jobject ret = env->NewObject(pickeritem_class, pickeritem_construct, l.v, cb.v, self);
  return ret;
}

jobject JNI::ToNativeCallback(Callback c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(nativecallback_class, "<init>", "(J)V"));
  jlong cbp = uintptr_t(new Callback(move(c)));
  return env->NewObject(nativecallback_class, mid, cbp);
}

jobject JNI::ToNativeStringCB(StringCB c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(nativestringcb_class, "<init>", "(J)V"));
  jlong cb = uintptr_t(new StringCB(move(c)));
  return env->NewObject(nativestringcb_class, mid, cb);
}

jobject JNI::ToNativeIntIntCB(IntIntCB c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(nativeintintcb_class, "<init>", "(J)V"));
  jlong cb = uintptr_t(new IntIntCB(move(c)));
  return env->NewObject(nativeintintcb_class, mid, cb);
}

jobject JNI::ToNativePickerItemCB(const PickerItem::CB &c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(nativepickeritemcb_class, "<init>", "(J)V"));
  jlong cb = uintptr_t(new PickerItem::CB(c));
  return env->NewObject(nativepickeritemcb_class, mid, cb);
}

BufferFile *JNI::OpenAsset(const string &fn) {
  static jmethodID get_assets_mid = CheckNotNull(env->GetMethodID(activity_class, "getAssets", "()Landroid/content/res/AssetManager;"));
  static jmethodID assetmgr_open_mid = CheckNotNull(env->GetMethodID(assetmgr_class, "open", "(Ljava/lang/String;)Ljava/io/InputStream;"));
  static jmethodID inputstream_avail_mid = CheckNotNull(env->GetMethodID(inputstream_class, "available", "()I"));
  static jmethodID channels_newchan_mid = CheckNotNull(env->GetStaticMethodID(channels_class, "newChannel", "(Ljava/io/InputStream;)Ljava/nio/channels/ReadableByteChannel;"));
  static jmethodID readbytechan_read_mid = CheckNotNull(env->GetMethodID(readbytechan_class, "read", "(Ljava/nio/ByteBuffer;)I"));

  LocalJNIString jfn(env, ToJString(fn));
  LocalJNIObject assets(env, env->CallObjectMethod(activity, get_assets_mid));
  LocalJNIObject input(env, env->CallObjectMethod(assets.v, assetmgr_open_mid, jfn.v));
  if (!input.v || CheckForException()) return nullptr;

  int len = env->CallIntMethod(input.v, inputstream_avail_mid);
  if (CheckForException()) return nullptr;

  unique_ptr<BufferFile> ret = make_unique<BufferFile>(string(), fn.c_str());
  if (!len) return ret.release();
  ret->buf.resize(len);

  LocalJNIObject readable(env, env->CallStaticObjectMethod(channels_class, channels_newchan_mid, input.v));
  LocalJNIObject bytes(env, env->NewDirectByteBuffer(&ret->buf[0], ret->buf.size()));
  len = env->CallIntMethod(readable.v, readbytechan_read_mid, bytes.v);

  if (len != ret->buf.size() || CheckForException()) return nullptr;
  return ret.release();
}

string Application::GetVersion() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getVersionName", "()Ljava/lang/String;"));
  LocalJNIString str(jni->env, (jstring)jni->env->CallObjectMethod(jni->activity, mid));
  return jni->GetJString(str.v);
}

void Application::OpenSystemBrowser(const string &url_text) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "openBrowser", "(Ljava/lang/String;)V"));
  LocalJNIString jurl(jni->env, jni->ToJString(url_text));
  jni->env->CallVoidMethod(jni->activity, mid, jurl.v);
}

string Application::GetSystemDeviceName() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getModelName", "()Ljava/lang/String;"));
  LocalJNIString str(jni->env, (jstring)jni->env->CallObjectMethod(jni->activity, mid));
  return jni->GetJString(str.v);
}

bool Application::OpenSystemAppPreferences() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "openPreferences", "()Z"));
  return jni->env->CallBooleanMethod(jni->activity, mid);
}

void Application::SaveKeychain(const string &k, const string &v) {}
bool Application::LoadKeychain(const string &k, string *v) { return false; }

String16 Application::GetLocalizedString16(const char *key) { return String16(); }
string Application::GetLocalizedString(const char *key) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->resources_class, "getString", "(I)Ljava/lang/String;"));
  jfieldID fid = CheckNotNull(jni->env->GetStaticFieldID(jni->r_string_class, key, "I"));
  int resource_id = jni->env->GetStaticIntField(jni->r_string_class, fid);
  LocalJNIString str(jni->env, (jstring)jni->env->CallObjectMethod(jni->resources, mid, resource_id));
  return jni->GetJString(str.v);
}

String16 Application::GetLocalizedInteger16(int number) { return String16(); }
string Application::GetLocalizedInteger(int number) {
  return StrCat(number);
}

void Application::LoadDefaultSettings(const StringPairVec &v) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setDefaultPreferences", "(Ljava/util/ArrayList;)V"));
  LocalJNIObject prefs(jni->env, jni->ToJStringPairArrayList(v));
  jni->env->CallVoidMethod(jni->activity, mid, prefs.v);
}

string Application::GetSetting(const string &key) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getPreference", "(Ljava/lang/String;)Ljava/lang/String;"));
  LocalJNIString jkey(jni->env, jni->ToJString(key));
  LocalJNIString jval(jni->env, (jstring)jni->env->CallObjectMethod(jni->activity, mid, jkey.v));
  return jval.v ? jni->GetJString(jval.v) : "";
}

void Application::SaveSettings(const StringPairVec &v) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "updatePreferences", "(Ljava/util/ArrayList;)V"));
  LocalJNIObject prefs(jni->env, jni->ToJStringPairArrayList(v));
  jni->env->CallVoidMethod(jni->activity, mid, prefs.v);
}

Connection *Application::ConnectTCP(const string &hostport, int default_port, Connection::CB *connected_cb, bool background_services) {
  INFO("Application::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = false"); 
  return app->net->tcp_client->Connect(hostport, default_port, connected_cb);
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
