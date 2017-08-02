/*
 * $Id$
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
const char* Java::V = "V";
const char* Java::I = "I";
const char* Java::J = "J";
const char* Java::Z = "Z";
const char* Java::Constructor = "<init>";
const char* Java::String = "Ljava/lang/String;";
const char* Java::ArrayList = "Ljava/util/ArrayList;";
const char* Java::MainActivity = "Lcom/lucidfusionlabs/app/MainActivity;";
  
static JNI *jni = Singleton<JNI>::Get();

void JNI::Init(jobject a, bool first) {
  if      (1)           CHECK(activity  = env->NewGlobalRef(a));
  if      (1)           CHECK(resources = env->NewGlobalRef(env->GetObjectField(activity, activity_resources)));
  if      (1)           CHECK(handler   = env->NewGlobalRef(env->GetObjectField(activity, activity_handler)));
  // if   (gplus_class) CHECK(gplus     = env->NewGlobalRef(env->GetObjectField(activity, activity_gplus)));
}

void JNI::Free() {
  if (gplus) env->DeleteGlobalRef(gplus);     gplus     = 0;
  if (1)     env->DeleteGlobalRef(handler);   handler   = 0;
  if (1)     env->DeleteGlobalRef(resources); resources = 0;
  if (1)     env->DeleteGlobalRef(activity);  activity  = 0;
}

jmethodID JNI::GetMethodID(jclass c, const char *name, const StringVec &args, const char *ret) {
  return CheckNotNull(env->GetMethodID(c, name, StrCat("(", Join(args, ""), ")", ret).c_str())); 
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
    out += GetJString(env, msg.v);
  }
  for (jsize i = 0; i < frames_length; i++) { 
    LocalJNIObject frame(env, env->GetObjectArrayElement(frames, i));
    LocalJNIString msg(env, (jstring)env->CallObjectMethod(frame.v, jni_frame_method_tostring));
    out += "\n    " + GetJString(env, msg.v);
  }
  if (frames > 0) {
    jthrowable cause = (jthrowable)env->CallObjectMethod(exception, jni_throwable_method_get_cause);
    if (cause) LogException(cause);
  }  

  INFOf("JNI::LogException: %s", out.c_str());
}

string JNI::GetJString(JNIEnv *env, jstring x) {
  if (!x) return "";
  const char *buf = env->GetStringUTFChars(x, 0);
  string ret = buf;
  env->ReleaseStringUTFChars(x, buf);
  return ret;
}

jstring JNI::ToJStringRaw(JNIEnv *env, const string &x) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(jni->string_class, "<init>", "([BLjava/lang/String;)V"));
  LocalJNIType<jbyteArray> array(env, env->NewByteArray(x.size()));
  env->SetByteArrayRegion(array.v, 0, x.size(), reinterpret_cast<const jbyte*>(x.data()));
  LocalJNIString encoding(env, env->NewStringUTF("UTF-8"));
  return jstring(env->NewObject(jni->string_class, mid, array.v, encoding.v));
}

jobjectArray JNI::ToJStringArray(JNIEnv *env, const StringVec &items) {
  jobjectArray v = env->NewObjectArray(items.size(), jni->string_class, NULL);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIString vi(env, ToJString(env, items[i]));
    env->SetObjectArrayElement(v, i, vi.v);
  }
  return v;
}

pair<jobjectArray, jobjectArray> JNI::ToJStringArrays(JNIEnv *env, const StringPairVec &items) {
  jobjectArray k = env->NewObjectArray(items.size(), jni->string_class, NULL);
  jobjectArray v = env->NewObjectArray(items.size(), jni->string_class, NULL);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIString ki(env, ToJString(env, items[i].first)), vi(env, ToJString(env, items[i].second));
    env->SetObjectArrayElement(k, i, ki.v);
    env->SetObjectArrayElement(v, i, vi.v);
  }
  return make_pair(k, v);
}

jobject JNI::ToJStringArrayList(JNIEnv *env, const StringVec &items) {
  jobject ret = env->NewObject(jni->arraylist_class, jni->arraylist_construct);
  for (auto &i : items) {
    LocalJNIObject v(env, ToJString(env, i));
    CHECK(env->CallBooleanMethod(ret, jni->arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToJStringPairArrayList(JNIEnv *env, const StringPairVec &items) {
  jobject ret = env->NewObject(jni->arraylist_class, jni->arraylist_construct);
  for (auto &i : items) {
    LocalJNIString ki(env, ToJString(env, i.first)), vi(env, ToJString(env, i.second));
    LocalJNIObject v(env, env->NewObject(jni->pair_class, jni->pair_construct, ki.v, vi.v));
    CHECK(env->CallBooleanMethod(ret, jni->arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToIntegerArrayList(JNIEnv *env, const vector<int> &items) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(jni->int_class, "<init>", "(I)V"));
  jobject ret = env->NewObject(jni->arraylist_class, jni->arraylist_construct);
  for (auto &i : items) {
    LocalJNIObject v(env, env->NewObject(jni->int_class, mid, i));
    CHECK(env->CallBooleanMethod(ret, jni->arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItemArrayList(JNIEnv *env, AlertItemVec items) {
  jobject ret = env->NewObject(jni->arraylist_class, jni->arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIObject v(env, ToModelItem(env, items[i]));
    CHECK(env->CallBooleanMethod(ret, jni->arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItemArrayList(JNIEnv *env, MenuItemVec items) {
  jobject ret = env->NewObject(jni->arraylist_class, jni->arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIObject v(env, ToModelItem(env, items[i]));
    CHECK(env->CallBooleanMethod(ret, jni->arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItemArrayList(JNIEnv *env, TableItemVec items) {
  jobject ret = env->NewObject(jni->arraylist_class, jni->arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    LocalJNIObject v(env, ToModelItem(env, items[i]));
    CHECK(env->CallBooleanMethod(ret, jni->arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItemChangeList(JNIEnv *env, const TableSectionInterface::ChangeList &items) {
  jobject ret = env->NewObject(jni->arraylist_class, jni->arraylist_construct);
  for (auto &i : items) {
    LocalJNIObject v(env, ToModelItemChange(env, i));
    CHECK(env->CallBooleanMethod(ret, jni->arraylist_add, v.v));
  }
  return ret;
}

jobject JNI::ToModelItem(JNIEnv *env, AlertItem item) {
  jboolean hidden=0;
  LocalJNIString k(env, ToJString(env, item.first)), v(env, ToJString(env, item.second)),
                 rt(env, ToJString(env, "")), ddk(env, ToJString(env, ""));
  LocalJNIObject rcb(env, item.cb ? ToNativeStringCB(env, move(item.cb)) : nullptr);
  jobject lcb = nullptr, picker = nullptr;
  jint type=0, tag=0, flags=0, right_icon=0, selected=0, height=0, fg=0, bg=0;
  return env->NewObject(jni->modelitem_class, jni->modelitem_construct, k.v, v.v, rt.v, ddk.v, type,
                        tag, flags, jint(item.image), right_icon, selected, height, lcb, rcb.v, picker,
                        hidden, fg, bg);
}

jobject JNI::ToModelItem(JNIEnv *env, MenuItem item) {
  LocalJNIString k(env, ToJString(env, item.shortcut)), v(env, ToJString(env, item.name)), rt(env, ToJString(env, "")), ddk(env, ToJString(env, ""));
  LocalJNIObject cb(env, item.cb ? ToNativeCallback(env, move(item.cb)) : nullptr);
  return env->NewObject(jni->modelitem_class, jni->modelitem_construct, k.v, v.v, rt.v, ddk.v, jint(0),
                        jint(0), jint(0), jint(item.image), jint(0), jint(0), jint(0), cb.v, nullptr, nullptr,
                        false, jint(0), jint(0));
}

jobject JNI::ToModelItem(JNIEnv *env, TableItem item) {
  jint fg = (item.fg_a << 24) | (item.fg_r << 16) | (item.fg_g << 8) | item.fg_b;
  jint bg = (item.bg_a << 24) | (item.bg_r << 16) | (item.bg_g << 8) | item.bg_b;
  LocalJNIObject k(env, ToJString(env, item.key)), v(env, ToJString(env, item.val)), rt(env, ToJString(env, item.right_text)),
          ddk(env, ToJString(env, item.dropdown_key)), picker(env, ToPickerItem(env, item.picker));
  LocalJNIObject cb(env, item.cb ? ToNativeCallback(env, move(item.cb)) : nullptr);
  LocalJNIObject rcb(env, item.right_cb ? ToNativeStringCB(env, move(item.right_cb)) : nullptr);
  return env->NewObject(jni->modelitem_class, jni->modelitem_construct, k.v, v.v, rt.v, ddk.v, jint(item.type),
                        jint(item.tag), jint(item.flags), jint(item.left_icon), jint(item.right_icon),
                        jint(item.selected), jint(item.height), cb.v, rcb.v, picker.v, jboolean(item.hidden),
                        fg, bg);
}

jobject JNI::ToModelItemChange(JNIEnv *env, const TableSectionInterface::Change &item) {
  LocalJNIObject k(env, ToJString(env, item.key)), v(env, ToJString(env, item.val));
  LocalJNIObject cb(env, item.cb ? ToNativeCallback(env, item.cb) : nullptr);
  return env->NewObject(jni->modelitemchange_class, jni->modelitemchange_construct, jint(item.section), jint(item.row),
                        jint(item.type), k.v, v.v, jint(item.left_icon), jint(item.right_icon),
                        jint(item.flags), jboolean(item.hidden), cb.v);
}

jobject JNI::ToPickerItem(JNIEnv *env, PickerItem *picker_in) {
  if (!picker_in) return nullptr;
  const PickerItem *picker = picker_in;
  jlong self = uintptr_t(picker_in);
  LocalJNIObject cb(env, picker->cb ? ToNativePickerItemCB(env, picker->cb) : nullptr);
  LocalJNIObject l(env, env->NewObject(jni->arraylist_class, jni->arraylist_construct));
  for (auto &i : picker->data) {
    LocalJNIObject v(env, ToJStringArrayList(env, i));
    CHECK(env->CallBooleanMethod(l.v, jni->arraylist_add, v.v));
  }
  jobject ret = env->NewObject(jni->pickeritem_class, jni->pickeritem_construct, l.v, cb.v, self);
  return ret;
}

jobject JNI::ToNativeCallback(JNIEnv *env, Callback c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(jni->nativecallback_class, "<init>", "(J)V"));
  jlong cbp = uintptr_t(new Callback(move(c)));
  return env->NewObject(jni->nativecallback_class, mid, cbp);
}

jobject JNI::ToNativeStringCB(JNIEnv *env, StringCB c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(jni->nativestringcb_class, "<init>", "(J)V"));
  jlong cb = uintptr_t(new StringCB(move(c)));
  return env->NewObject(jni->nativestringcb_class, mid, cb);
}

jobject JNI::ToNativeIntCB(JNIEnv *env, IntCB c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(jni->nativeintcb_class, "<init>", "(J)V"));
  jlong cb = uintptr_t(new IntCB(move(c)));
  return env->NewObject(jni->nativeintcb_class, mid, cb);
}

jobject JNI::ToNativeIntIntCB(JNIEnv *env, IntIntCB c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(jni->nativeintintcb_class, "<init>", "(J)V"));
  jlong cb = uintptr_t(new IntIntCB(move(c)));
  return env->NewObject(jni->nativeintintcb_class, mid, cb);
}

jobject JNI::ToNativePickerItemCB(JNIEnv *env, const PickerItem::CB &c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(jni->nativepickeritemcb_class, "<init>", "(J)V"));
  jlong cb = uintptr_t(new PickerItem::CB(c));
  return env->NewObject(jni->nativepickeritemcb_class, mid, cb);
}

BufferFile *JNI::OpenAsset(const string &fn) {
  static jmethodID get_assets_mid = CheckNotNull(env->GetMethodID(activity_class, "getAssets", "()Landroid/content/res/AssetManager;"));
  static jmethodID assetmgr_open_mid = CheckNotNull(env->GetMethodID(assetmgr_class, "open", "(Ljava/lang/String;)Ljava/io/InputStream;"));
  static jmethodID inputstream_avail_mid = CheckNotNull(env->GetMethodID(inputstream_class, "available", "()I"));
  static jmethodID channels_newchan_mid = CheckNotNull(env->GetStaticMethodID(channels_class, "newChannel", "(Ljava/io/InputStream;)Ljava/nio/channels/ReadableByteChannel;"));
  static jmethodID readbytechan_read_mid = CheckNotNull(env->GetMethodID(readbytechan_class, "read", "(Ljava/nio/ByteBuffer;)I"));

  LocalJNIString jfn(env, ToJString(env, fn));
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
  return JNI::GetJString(jni->env, str.v);
}

void Application::OpenSystemBrowser(const string &url_text) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "openBrowser", "(Ljava/lang/String;)V"));
  LocalJNIString jurl(jni->env, JNI::ToJString(jni->env, url_text));
  jni->env->CallVoidMethod(jni->activity, mid, jurl.v);
}

string Application::GetSystemDeviceName() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getModelName", "()Ljava/lang/String;"));
  LocalJNIString str(jni->env, (jstring)jni->env->CallObjectMethod(jni->activity, mid));
  return JNI::GetJString(jni->env, str.v);
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
  return JNI::GetJString(jni->env, str.v);
}

String16 Application::GetLocalizedInteger16(int number) { return String16(); }
string Application::GetLocalizedInteger(int number) {
  return StrCat(number);
}

void Application::LoadDefaultSettings(const StringPairVec &v) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setDefaultPreferences", "(Ljava/util/ArrayList;)V"));
  LocalJNIObject prefs(jni->env, JNI::ToJStringPairArrayList(jni->env, v));
  jni->env->CallVoidMethod(jni->activity, mid, prefs.v);
}

string Application::GetSetting(const string &key) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getPreference", "(Ljava/lang/String;)Ljava/lang/String;"));
  LocalJNIString jkey(jni->env, JNI::ToJString(jni->env, key));
  LocalJNIString jval(jni->env, (jstring)jni->env->CallObjectMethod(jni->activity, mid, jkey.v));
  return jval.v ? JNI::GetJString(jni->env, jval.v) : "";
}

void Application::SaveSettings(const StringPairVec &v) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "updatePreferences", "(Ljava/util/ArrayList;)V"));
  LocalJNIObject prefs(jni->env, JNI::ToJStringPairArrayList(jni->env, v));
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
