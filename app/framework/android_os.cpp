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
  for (int i=0, l=items.size(); i != l; ++i) {
    jobject v = ToJString(items[i]);
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

jobject JNI::ToJModelItem(AlertItem item) {
  jobject k = ToJString(item.first), v = ToJString(item.second), rt = ToJString("");
  jlong cb = item.cb ? intptr_t(new StringCB(move(item.cb))) : 0;
  jobject ret = env->NewObject(jmodelitem_class, jmodelitem_construct, k, v, rt, jint(0), jint(0),
                               jint(0), jlong(0), jlong(0), cb, false);
  env->DeleteLocalRef(rt);
  env->DeleteLocalRef(v);
  env->DeleteLocalRef(k);
  return ret;
}

jobject JNI::ToJModelItem(MenuItem item) {
  jobject k = ToJString(item.shortcut), v = ToJString(item.name), rt = ToJString("");
  jlong cb = item.cb ? intptr_t(new Callback(move(item.cb))) : 0;
  jobject ret = env->NewObject(jmodelitem_class, jmodelitem_construct, k, v, rt, jint(0), jint(0),
                               jint(0), cb, jlong(0), jlong(0), false);
  env->DeleteLocalRef(rt);
  env->DeleteLocalRef(v);
  env->DeleteLocalRef(k);
  return ret;
}

jobject JNI::ToJModelItem(TableItem item) {
  jobject k = ToJString(item.key), v = ToJString(item.val), rt = ToJString(item.right_text);
  jlong cb = item.cb ? intptr_t(new Callback(move(item.cb))) : 0;
  jlong rcb = item.right_icon_cb ? intptr_t(new Callback(move(item.right_icon_cb))) : 0;
  jobject ret = env->NewObject(jmodelitem_class, jmodelitem_construct, k, v, rt, item.type,
                               item.left_icon, item.right_icon, cb, rcb, jlong(0), item.hidden);
  env->DeleteLocalRef(rt);
  env->DeleteLocalRef(v);
  env->DeleteLocalRef(k);
  return ret;
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

string Application::GetVersion() { return "1.0"; }

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

void Application::SaveKeychain(const string &k, const string &v) {}
bool Application::LoadKeychain(const string &k, string *v) { return false; }

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
