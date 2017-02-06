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

string JNI::GetEnvJString(JNIEnv *e, jstring x) {
  const char *buf = e->GetStringUTFChars(x, 0);
  string ret = buf;
  e->ReleaseStringUTFChars(x, buf);
  return ret;
}

jstring JNI::ToJStringRaw(const string &x) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V"));
  jbyteArray array = env->NewByteArray(x.size());
  env->SetByteArrayRegion(array, 0, x.size(), reinterpret_cast<const jbyte*>(x.data()));
  jstring encoding = env->NewStringUTF("UTF-8");
  jstring ret = (jstring)env->NewObject(string_class, mid, array, encoding);
  env->DeleteLocalRef(encoding);
  env->DeleteLocalRef(array);
  return ret;
}

jobjectArray JNI::ToJStringArray(StringVec items) {
  jobjectArray v = env->NewObjectArray(items.size(), string_class, NULL);
  for (int i=0, l=items.size(); i != l; ++i) {
    jstring vi = ToJString(items[i]);
    env->SetObjectArrayElement(v, i, vi);
    env->DeleteLocalRef(vi);
  }
  return v;
}

pair<jobjectArray, jobjectArray> JNI::ToJStringArrays(StringPairVec items) {
  jobjectArray k = env->NewObjectArray(items.size(), string_class, NULL);
  jobjectArray v = env->NewObjectArray(items.size(), string_class, NULL);
  for (int i=0, l=items.size(); i != l; ++i) {
    jstring ki = ToJString(items[i].first), vi = ToJString(items[i].second);
    env->SetObjectArrayElement(k, i, ki);
    env->SetObjectArrayElement(v, i, vi);
    env->DeleteLocalRef(ki);
    env->DeleteLocalRef(vi);
  }
  return make_pair(k, v);
}

jobject JNI::ToJStringArrayList(StringVec items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (auto &i : items) {
    jobject v = ToJString(i);
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v));
    env->DeleteLocalRef(v);
  }
  return ret;
}

jobject JNI::ToJStringPairArrayList(StringPairVec items) {
  static jmethodID string_pair_construct = CheckNotNull
    (env->GetMethodID(pair_class, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V"));
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    jstring ki = ToJString(items[i].first), vi = ToJString(items[i].second);
    jobject v = env->NewObject(pair_class, string_pair_construct, ki, vi);
    env->DeleteLocalRef(v);
    env->DeleteLocalRef(ki);
    env->DeleteLocalRef(vi);
  }
  return ret;
}

jobject JNI::ToJModelItemArrayList(AlertItemVec items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    jobject v = ToJModelItem(items[i]);
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v));
    env->DeleteLocalRef(v);
  }
  return ret;
}

jobject JNI::ToJModelItemArrayList(MenuItemVec items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    jobject v = ToJModelItem(items[i]);
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v));
    env->DeleteLocalRef(v);
  }
  return ret;
}

jobject JNI::ToJModelItemArrayList(TableItemVec items) {
  jobject ret = env->NewObject(arraylist_class, arraylist_construct);
  for (int i=0, l=items.size(); i != l; ++i) {
    jobject v = ToJModelItem(items[i]);
    CHECK(env->CallBooleanMethod(ret, arraylist_add, v));
    env->DeleteLocalRef(v);
  }
  return ret;
}

jobject JNI::ToJDependencyItemHashMap(TableItem::Depends items) {
  jobject ret = env->NewObject(hashmap_class, hashmap_construct);
  for (auto &di : items) {
    jstring k = ToJString(di.first);
    jobject l = env->NewObject(arraylist_class, arraylist_construct);
    for (auto &i : di.second) {
      jobject v = ToJDependencyItem(move(i));
      CHECK(env->CallBooleanMethod(l, arraylist_add, v));
      env->DeleteLocalRef(v);
    }
    env->CallObjectMethod(ret, hashmap_put, k, l);
    env->DeleteLocalRef(l);
    env->DeleteLocalRef(k);
  }
  return ret;
}

jobject JNI::ToJModelItem(AlertItem item) {
  jobject k = ToJString(item.first), v = ToJString(item.second), rt = ToJString(""), ddk = ToJString(""), p = nullptr, dep = nullptr, lcb = nullptr, rcb = nullptr;
  jobject cb = item.cb ? ToLStringCB(move(item.cb)) : nullptr;
  jobject ret = env->NewObject(jmodelitem_class, jmodelitem_construct, k, v, rt, ddk, jint(0), jint(0),
                               jint(0), jint(0), lcb, rcb, cb, false, p, dep);
  if (cb) env->DeleteLocalRef(cb);
  env->DeleteLocalRef(ddk);
  env->DeleteLocalRef(rt);
  env->DeleteLocalRef(v);
  env->DeleteLocalRef(k);
  return ret;
}

jobject JNI::ToJModelItem(MenuItem item) {
  jobject k = ToJString(item.shortcut), v = ToJString(item.name), rt = ToJString(""), ddk = ToJString(""), p = nullptr, dep = nullptr, rcb = nullptr, scb = nullptr;
  jobject cb = item.cb ? ToLCallback(move(item.cb)) : nullptr;
  jobject ret = env->NewObject(jmodelitem_class, jmodelitem_construct, k, v, rt, ddk, jint(0), jint(0),
                               jint(0), jint(0), cb, rcb, scb, false, p, dep);
  if (cb) env->DeleteLocalRef(cb);
  env->DeleteLocalRef(ddk);
  env->DeleteLocalRef(rt);
  env->DeleteLocalRef(v);
  env->DeleteLocalRef(k);
  return ret;
}

jobject JNI::ToJModelItem(TableItem item) {
  jobject k = ToJString(item.key), v = ToJString(item.val), rt = ToJString(item.right_text),
          ddk = ToJString(item.dropdown_key), picker = ToJPickerItem(item.picker),
          dep = ToJDependencyItemHashMap(move(item.depends)), scb = nullptr;
  jobject cb = item.cb ? ToLCallback(move(item.cb)) : nullptr;
  jobject rcb = item.right_icon_cb ? ToLCallback(move(item.right_icon_cb)) : nullptr;
  jobject ret = env->NewObject(jmodelitem_class, jmodelitem_construct, k, v, rt, ddk, item.type, item.flags,
                               item.left_icon, item.right_icon, cb, rcb, scb, item.hidden, picker, dep);
  if (rcb) env->DeleteLocalRef(rcb);
  if (cb) env->DeleteLocalRef(cb);
  if (dep) env->DeleteLocalRef(dep);
  if (picker) env->DeleteLocalRef(picker);
  env->DeleteLocalRef(ddk);
  env->DeleteLocalRef(rt);
  env->DeleteLocalRef(v);
  env->DeleteLocalRef(k);
  return ret;
}

jobject JNI::ToJDependencyItem(TableItem::Dep item) {
  jobject k = ToJString(item.key), v = ToJString(item.val);
  jobject cb = item.cb ? ToLCallback(move(item.cb)) : nullptr;
  jobject ret = env->NewObject(jdepitem_class, jdepitem_construct, jint(item.section), jint(item.row),
                               jint(item.type), k, v, jint(item.left_icon), jint(item.right_icon),
                               jint(item.flags), jboolean(item.hidden), cb);
  if (cb) env->DeleteLocalRef(cb);
  env->DeleteLocalRef(v);
  env->DeleteLocalRef(k);
  return ret;
}

jobject JNI::ToJPickerItem(PickerItem *picker_in) {
  if (!picker_in) return nullptr;
  const PickerItem *picker = picker_in;
  jlong self = intptr_t(picker_in);
  jobject cb = picker->cb ? ToLPickerItemCB(picker->cb) : nullptr;
  jobject l = env->NewObject(arraylist_class, arraylist_construct);
  for (auto &i : picker->data) {
    jobject v = ToJStringArrayList(i);
    CHECK(env->CallBooleanMethod(l, arraylist_add, v));
    env->DeleteLocalRef(v);
  }
  jobject ret = env->NewObject(jpickeritem_class, jpickeritem_construct, l, cb, self);
  env->DeleteLocalRef(l);
  if (cb) env->DeleteLocalRef(cb);
  return ret;
}

jobject JNI::ToLCallback(Callback c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(lcallback_class, "<init>", "(J)V"));
  jlong cb = intptr_t(new Callback(move(c)));
  return env->NewObject(lcallback_class, mid, cb);
}

jobject JNI::ToLStringCB(StringCB c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(lstringcb_class, "<init>", "(J)V"));
  jlong cb = intptr_t(new StringCB(move(c)));
  return env->NewObject(lstringcb_class, mid, cb);
}

jobject JNI::ToLIntIntCB(IntIntCB c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(lintintcb_class, "<init>", "(J)V"));
  jlong cb = intptr_t(new IntIntCB(move(c)));
  return env->NewObject(lintintcb_class, mid, cb);
}

jobject JNI::ToLPickerItemCB(const PickerItem::CB &c) {
  static jmethodID mid = CheckNotNull(env->GetMethodID(lpickeritemcb_class, "<init>", "(J)V"));
  jlong cb = intptr_t(new PickerItem::CB(c));
  return env->NewObject(lpickeritemcb_class, mid, cb);
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

string Application::GetVersion() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getVersionName", "()Ljava/lang/String;"));
  jstring str = (jstring)jni->env->CallObjectMethod(jni->activity, mid);
  string ret = jni->GetJString(str);
  jni->env->DeleteLocalRef(str);
  return ret;
}

void Application::OpenSystemBrowser(const string &url_text) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "openBrowser", "(Ljava/lang/String;)V"));
  jstring jurl = jni->ToJString(url_text);
  jni->env->CallVoidMethod(jni->activity, mid, jurl);
  jni->env->DeleteLocalRef(jurl);
}

string Application::GetSystemDeviceName() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getModelName", "()Ljava/lang/String;"));
  jstring str = (jstring)jni->env->CallObjectMethod(jni->activity, mid);
  string ret = jni->GetJString(str);
  jni->env->DeleteLocalRef(str);
  return ret;
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
  jstring str = (jstring)jni->env->CallObjectMethod(jni->resources, mid, resource_id);
  string ret = jni->GetJString(str);
  jni->env->DeleteLocalRef(str);
  return ret;
}

String16 Application::GetLocalizedInteger16(int number) { return String16(); }
string Application::GetLocalizedInteger(int number) {
  return StrCat(number);
}

void Application::LoadDefaultSettings(const StringPairVec &v) {}
string Application::GetSetting(const string &key) { return string(); }
void Application::SaveSettings(const StringPairVec &v) {}

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
