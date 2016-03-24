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

#include <android/log.h>
#include <libgen.h>
#include <jni.h>

#include "core/app/bindings/jni.h"

#define ACTION_DOWN         0
#define ACTION_UP           1
#define ACTION_MOVE         2
#define ACTION_CANCEL       3
#define ACTION_OUTSIDE      4
#define ACTION_POINTER_DOWN 5
#define ACTION_POINTER_UP   6

static JNIEnv *jni_env;
static jobject jni_activity,       jni_view,       jni_gplus;
static jclass  jni_activity_class, jni_view_class, jni_gplus_class, jni_throwable_class, jni_frame_class;
static jfieldID jni_view_id, jni_gplus_id;
static jmethodID jni_view_method_swap;
static jmethodID jni_activity_method_toggle_keyboard, jni_activity_method_show_keyboard, jni_activity_method_hide_keyboard;
static jmethodID jni_activity_method_write_internal_file, jni_activity_method_play_music, jni_activity_method_play_background_music;
static jmethodID jni_gplus_method_write, jni_gplus_method_write_with_retry;
static jmethodID jni_throwable_method_get_cause, jni_throwable_method_get_stack_trace, jni_throwable_method_tostring, jni_frame_method_tostring;

static void *gplus_service;
static int jni_activity_width=0, jni_activity_height=0;

static void InitMyPointers(jobject a, bool init) {
  if      (1)               CHECK(jni_activity = jni_env->NewGlobalRef(a));
  if      (1)               CHECK(jni_view     = jni_env->NewGlobalRef(jni_env->GetObjectField(jni_activity, jni_view_id )));
  if      (init)                  jni_gplus    = jni_env->NewGlobalRef(jni_env->GetObjectField(jni_activity, jni_gplus_id));
  else if (jni_gplus_class) CHECK(jni_gplus    = jni_env->NewGlobalRef(jni_env->GetObjectField(jni_activity, jni_gplus_id)));
}

static void FreeMyPointers() {
  if (jni_gplus_class) jni_env->DeleteGlobalRef(jni_gplus);    jni_gplus    = 0;
  if (1)               jni_env->DeleteGlobalRef(jni_view);     jni_view     = 0;
  if (1)               jni_env->DeleteGlobalRef(jni_activity); jni_activity = 0;
  jni_activity_width = jni_activity_height = -1;
}

extern "C" jint JNI_OnLoad(JavaVM* vm, void* reserved) { return JNI_VERSION_1_4; }

extern "C" void Java_com_lucidfusionlabs_app_Activity_main(JNIEnv *e, jclass c, jobject a) {
  MyAppCreate();
  CHECK(jni_env = e);
  INFOf("main: env=%p", jni_env);

  CHECK(jni_activity_class = (jclass)jni_env->NewGlobalRef(jni_env->GetObjectClass(a)));
  CHECK(jni_view_id  = jni_env->GetFieldID(jni_activity_class, "view",  "Lcom/lucidfusionlabs/app/GameView;"));
  CHECK(jni_gplus_id = jni_env->GetFieldID(jni_activity_class, "gplus", "Lcom/lucidfusionlabs/app/GPlusClient;"));
  InitMyPointers(a, true);

  CHECK(jni_activity_method_toggle_keyboard       = jni_env->GetMethodID(jni_activity_class, "toggleKeyboard", "()V"));
  CHECK(jni_activity_method_show_keyboard         = jni_env->GetMethodID(jni_activity_class, "showKeyboard", "()V"));
  CHECK(jni_activity_method_hide_keyboard         = jni_env->GetMethodID(jni_activity_class, "hideKeyboard", "()V"));
  CHECK(jni_activity_method_play_music            = jni_env->GetMethodID(jni_activity_class, "playMusic", "(Landroid/media/MediaPlayer;)V"));
  CHECK(jni_activity_method_play_background_music = jni_env->GetMethodID(jni_activity_class, "playBackgroundMusic", "(Landroid/media/MediaPlayer;)V"));
  CHECK(jni_activity_method_write_internal_file   = jni_env->GetMethodID(jni_activity_class, "writeFile", "(Ljava/io/FileOutputStream;[BI)V"));

  CHECK(jni_view_class = (jclass)jni_env->NewGlobalRef(jni_env->GetObjectClass(jni_view)));
  CHECK(jni_view_method_swap = jni_env->GetMethodID(jni_view_class, "swapEGL", "()V"));

  if (jni_gplus) {
    CHECK(jni_gplus_class = (jclass)jni_env->NewGlobalRef(jni_env->GetObjectClass(jni_gplus)));
    CHECK(jni_gplus_method_write = jni_env->GetMethodID(jni_gplus_class, "write", "(Ljava/lang/String;Ljava/nio/ByteBuffer;)V"));
    CHECK(jni_gplus_method_write_with_retry = jni_env->GetMethodID(jni_gplus_class, "writeWithRetry", "(Ljava/lang/String;Ljava/nio/ByteBuffer;)V"));
  }

  CHECK(jni_throwable_class = jni_env->FindClass("java/lang/Throwable"));
  CHECK(jni_throwable_method_get_cause = jni_env->GetMethodID(jni_throwable_class, "getCause", "()Ljava/lang/Throwable;"));
  CHECK(jni_throwable_method_get_stack_trace = jni_env->GetMethodID(jni_throwable_class, "getStackTrace", "()[Ljava/lang/StackTraceElement;"));
  CHECK(jni_throwable_method_tostring = jni_env->GetMethodID(jni_throwable_class, "toString", "()Ljava/lang/String;"));

  CHECK(jni_frame_class = jni_env->FindClass("java/lang/StackTraceElement"));
  CHECK(jni_frame_method_tostring = jni_env->GetMethodID(jni_frame_class, "toString", "()Ljava/lang/String;"));

  const char *argv[2] = { "lfjni", 0 };
  int argc = 1, ret = MyAppMain(argc, argv);
  INFOf("main: env=%p ret=%d", jni_env, ret);
  FreeMyPointers();
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_mainloop(JNIEnv *e, jclass c, jobject a) {
  CHECK(jni_env = e);
  INFOf("mainloop: env=%p", jni_env);
  InitMyPointers(a, false);
  SetLFAppMainThread();
  LFAppResetGL();
  WindowUnMinimized();
  LFAppMainLoop();
  FreeMyPointers();
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_minimize(JNIEnv* env, jclass c) {
  INFOf("%s", "minimize");
  QueueWindowMinimized();
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_resize(JNIEnv *e, jclass c, jint w, jint h) { 
  bool init = !jni_activity_width && !jni_activity_height;
  jni_activity_width = w;
  jni_activity_height = h;
  if (init) return;
  QueueWindowReshaped(w, h);
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_key(JNIEnv *e, jclass c, jint down, jint keycode) {
  QueueKeyPress(keycode, down);
  LFAppWakeup((void*)1);
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_touch(JNIEnv *e, jclass c, jint action, jfloat x, jfloat y, jfloat p) {
  static float lx[2]={0,0}, ly[2]={0,0};
  int dpind = (/*FLAGS_swap_axis*/ 0) ? y < LFL::screen->width/2 : x < LFL::screen->width/2;
  if (action == ACTION_DOWN || action == ACTION_POINTER_DOWN) {
    // INFOf("%d down %f, %f", dpind, x, y);
    QueueMouseClick(1, 1, (int)x, LFL::screen->height - (int)y);
    LFAppWakeup((void*)1);
    LFL::screen->gesture_tap[dpind] = 1;
    LFL::screen->gesture_dpad_x[dpind] = x;
    LFL::screen->gesture_dpad_y[dpind] = y;
    lx[dpind] = x;
    ly[dpind] = y;
  } else if (action == ACTION_UP || action == ACTION_POINTER_UP) {
    // INFOf("%d up %f, %f", dpind, x, y);
    QueueMouseClick(1, 0, (int)x, LFL::screen->height - (int)y);
    LFAppWakeup((void*)1);
    LFL::screen->gesture_dpad_stop[dpind] = 1;
    LFL::screen->gesture_dpad_x[dpind] = 0;
    LFL::screen->gesture_dpad_y[dpind] = 0;
  } else if (action == ACTION_MOVE) {
    float vx = x - lx[dpind]; lx[dpind] = x;
    float vy = y - ly[dpind]; ly[dpind] = y;
    // INFOf("%d move %f, %f vel = %f, %f", dpind, x, y, vx, vy);
    if (vx > 1.5 || vx < -1.5 || vy > 1.5 || vy < -1.5) {
      LFL::screen->gesture_dpad_dx[dpind] = vx;
      LFL::screen->gesture_dpad_dy[dpind] = vy;
    }
    LFL::screen->gesture_dpad_x[dpind] = x;
    LFL::screen->gesture_dpad_y[dpind] = y;
  } else INFOf("unhandled action %d", action);
} 

extern "C" void Java_com_lucidfusionlabs_app_Activity_fling(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat vx, jfloat vy) {
  int dpind = y < LFL::screen->width/2;
  LFL::screen->gesture_dpad_dx[dpind] = vx;
  LFL::screen->gesture_dpad_dy[dpind] = vy;
  INFOf("fling(%f, %f) = %d of (%d, %d) and vel = (%f, %f)", x, y, dpind, LFL::screen->width, LFL::screen->height, vx, vy);
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_scroll(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat vx, jfloat vy) {
  LFL::screen->gesture_swipe_up = LFL::screen->gesture_swipe_down = 0;
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_accel(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat z) {}

extern "C" void Java_com_lucidfusionlabs_app_GPlusClient_startGame(JNIEnv *e, jclass c, jboolean server, jstring pid) {
  char buf[128];
  const char *participant_id = e->GetStringUTFChars(pid, 0);
  snprintf(buf, sizeof(buf), "%s %s", server ? "gplus_server" : "gplus_client", participant_id);
  ShellRun(buf);
  e->ReleaseStringUTFChars(pid, participant_id);
}

extern "C" void Java_com_lucidfusionlabs_app_GPlusClient_read(JNIEnv *e, jclass c, jstring pid, jobject bb, jint len) {
  const char *participant_id = e->GetStringUTFChars(pid, 0);
  if (gplus_service) EndpointRead(gplus_service, participant_id, (const char*)e->GetDirectBufferAddress(bb), len);
  e->ReleaseStringUTFChars(pid, participant_id);
}

void AndroidExceptionLog(jthrowable &exception) {
  jobjectArray frames = (jobjectArray)jni_env->CallObjectMethod(exception, jni_throwable_method_get_stack_trace);
  jsize frames_length = jni_env->GetArrayLength(frames);
  std::string out;
#if 1
  if (frames > 0) {
    jstring msg = (jstring)jni_env->CallObjectMethod(exception, jni_throwable_method_tostring);
    const char *m = jni_env->GetStringUTFChars(msg, 0);
    out += m;
    jni_env->ReleaseStringUTFChars(msg, m);
    jni_env->DeleteLocalRef(msg);
  }
  for (jsize i = 0; i < frames_length; i++) { 
    jobject frame = jni_env->GetObjectArrayElement(frames, i);
    jstring msg = (jstring)jni_env->CallObjectMethod(frame, jni_frame_method_tostring);
    const char *m = jni_env->GetStringUTFChars(msg, 0);
    out += "\n    " + std::string(m);

    jni_env->ReleaseStringUTFChars(msg, m);
    jni_env->DeleteLocalRef(msg);
    jni_env->DeleteLocalRef(frame);
  }
  if (frames > 0) {
    jthrowable cause = (jthrowable)jni_env->CallObjectMethod(exception, jni_throwable_method_get_cause);
    if (cause) AndroidExceptionLog(cause);
  }  
#endif
  INFOf("AndroidException: %s", out.c_str());
}

int AndroidException() {
  jthrowable exception = jni_env->ExceptionOccurred();
  if (!exception) return 0;
  jni_env->ExceptionClear();
  AndroidExceptionLog(exception);
  return -1;
}

extern "C" int AndroidVideoInit(int *gles_version) {
  LFL::screen->width = jni_activity_width;
  LFL::screen->height = jni_activity_height;

  jint v;
  jfieldID fid;
  CHECK(fid = jni_env->GetFieldID(jni_activity_class, "egl_version", "I"));
  CHECK(v   = jni_env->GetIntField(jni_activity, fid));
  *gles_version = v;
  INFOf("AndroidVideoInit: %d", v);
  return 0;
}

extern "C" int AndroidVideoSwap() {
  jni_env->CallVoidMethod(jni_view, jni_view_method_swap);
  return 0;
}

extern "C" int AndroidShowOrHideKeyboard(int v) {
  if      (!v)     jni_env->CallVoidMethod(jni_activity, jni_activity_method_hide_keyboard);
  else if (v == 1) jni_env->CallVoidMethod(jni_activity, jni_activity_method_show_keyboard);
  else             jni_env->CallVoidMethod(jni_activity, jni_activity_method_toggle_keyboard);
  return 0;
}

extern "C" int AndroidFileRead(const char *fn, char **out, int *size) {
  jmethodID mid; jobject bytes;
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "sizeFile", "(Ljava/lang/String;)I"));
  jstring jfn = jni_env->NewStringUTF(fn);
  if (!(*size = jni_env->CallIntMethod(jni_activity, mid, jfn))) return 0;
  *out = (char*)malloc(*size);
  CHECK(bytes = jni_env->NewDirectByteBuffer(*out, *size));
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "readFile", "(Ljava/lang/String;[BI)I"));
  int ret = jni_env->CallIntMethod(jni_activity, mid, jfn, bytes, *size);
  jni_env->DeleteLocalRef(jfn);
  return 0;
}

extern "C" void *AndroidFileOpenWriter(const char *fn) {
  jmethodID mid; jstring jfn = jni_env->NewStringUTF(fn);
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "openFileWriter", "(Ljava/lang/String;)Ljava/io/FileOutputStream;"));
  jobject handle = jni_env->CallObjectMethod(jni_activity, mid, jfn);
  jni_env->DeleteLocalRef(jfn);
  return handle;
}

extern "C" int AndroidFileWrite(void *ifw, const char *b, int l) {
  jobject handle = (jobject)ifw;
  jobject bytes = jni_env->NewDirectByteBuffer((void*)b, l);
  jni_env->CallVoidMethod(jni_activity, jni_activity_method_write_internal_file, handle, bytes, l);
  jni_env->DeleteLocalRef(bytes);
  return 0;
}

extern "C" void AndroidFileCloseWriter(void *ifw) {
  jmethodID mid; jobject handle = (jobject)ifw;
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "closeFileWriter", "(Ljava/io/FileOutputStream;)V"));
  jni_env->CallVoidMethod(jni_activity, mid, handle);
}

extern "C" int AndroidAssetRead(const char *fn, char **out, int *size) {
  jmethodID mid; jstring jfn = jni_env->NewStringUTF(fn);
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getAssets", "()Landroid/content/res/AssetManager;"));
  jobject assets = jni_env->CallObjectMethod(jni_activity, mid);
  jclass assets_class = jni_env->GetObjectClass(assets);

  CHECK(mid = jni_env->GetMethodID(assets_class, "open", "(Ljava/lang/String;)Ljava/io/InputStream;"));
  jobject input = jni_env->CallObjectMethod(assets, mid, jfn);
  jni_env->DeleteLocalRef(jfn);
  jni_env->DeleteLocalRef(assets);
  jni_env->DeleteLocalRef(assets_class);
  if (!input || AndroidException()) return -1;
  jclass input_class = jni_env->GetObjectClass(input);

  CHECK(mid = jni_env->GetMethodID(input_class, "available", "()I"));
  *size = jni_env->CallIntMethod(input, mid);
  jni_env->DeleteLocalRef(input_class);
  if (AndroidException()) { jni_env->DeleteLocalRef(input); return -1; }
  if (!*size) { jni_env->DeleteLocalRef(input); *out=(char*)""; return 0; }

  jclass channels = jni_env->FindClass("java/nio/channels/Channels");
  CHECK(mid = jni_env->GetStaticMethodID(channels, "newChannel", "(Ljava/io/InputStream;)Ljava/nio/channels/ReadableByteChannel;"));
  jobject readable = jni_env->CallStaticObjectMethod(channels, mid, input);
  jclass readable_class = jni_env->GetObjectClass(readable);
  jni_env->DeleteLocalRef(input);
  jni_env->DeleteLocalRef(channels);

  *out = (char*)malloc(*size);
  jobject bytes = jni_env->NewDirectByteBuffer(*out, *size);
  CHECK(mid = jni_env->GetMethodID(readable_class, "read", "(Ljava/nio/ByteBuffer;)I"));
  int ret = jni_env->CallIntMethod(readable, mid, bytes);
  jni_env->DeleteLocalRef(readable);
  jni_env->DeleteLocalRef(readable_class);
  jni_env->DeleteLocalRef(bytes);

  if (ret != *size || AndroidException()) return -1;
  return 0;
}

extern "C" void *AndroidLoadMusicResource(const char *fp) {
  char fn[1024];
  snprintf(fn, sizeof(fn), "%s", basename(fp));
  char *suffix = strchr(fn, '.');
  if (suffix) *suffix = 0;
  jmethodID mid; jstring jfn = jni_env->NewStringUTF(fn);
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "loadMusicResource", "(Ljava/lang/String;)Landroid/media/MediaPlayer;"));
  jobject handle = jni_env->CallObjectMethod(jni_activity, mid, jfn);
  jni_env->DeleteLocalRef(jfn);
  return handle;
}

extern "C" void AndroidPlayMusic(void *h) {
  jobject handle = (jobject)h;
  jni_env->CallVoidMethod(jni_activity, jni_activity_method_play_music, handle);
}  

extern "C" void AndroidPlayBackgroundMusic(void *h) {
  jobject handle = (jobject)h;
  jni_env->CallVoidMethod(jni_activity, jni_activity_method_play_background_music, handle);
}

extern "C" void AndroidSetVolume(int v) {
  jint jv = v;
  jmethodID mid;
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "setVolume", "(I)V"));
  return jni_env->CallVoidMethod(jni_activity, mid, jv);
}

extern "C" int AndroidGetVolume() {
  jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getVolume", "()I"));
  return jni_env->CallIntMethod(jni_activity, mid);
}

extern "C" int AndroidGetMaxVolume() {
  jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_activity_class, "maxVolume", "()I"));
  return jni_env->CallIntMethod(jni_activity, mid);
}

extern "C" int AndroidDeviceName(char *out, int size) {
  out[0] = 0;
  jmethodID mid;
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getModelName", "()Ljava/lang/String;"));
  jstring ret = (jstring)jni_env->CallObjectMethod(jni_activity, mid);
  const char *id = jni_env->GetStringUTFChars(ret, 0);
  strncpy(out, id, size-1);
  out[size-1] = 0;
  return strlen(out);
}

extern "C" int AndroidIPV4Address() {
  jmethodID mid;
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getAddress", "()I"));
  return jni_env->CallIntMethod(jni_activity, mid);
}

extern "C" int AndroidIPV4BroadcastAddress() {
  jmethodID mid;
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "getBroadcastAddress", "()I"));
  return jni_env->CallIntMethod(jni_activity, mid);
}

extern "C" void AndroidOpenBrowser(const char *url) {
  jmethodID mid; jstring jurl = jni_env->NewStringUTF(url);
  CHECK(mid = jni_env->GetMethodID(jni_activity_class, "openBrowser", "(Ljava/lang/String;)V"));
  jni_env->CallVoidMethod(jni_activity, mid, jurl);
  jni_env->DeleteLocalRef(jurl);
}

extern "C" void AndroidShowAds() {
  jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_activity_class, "showAds", "()V"));
  jni_env->CallVoidMethod(jni_activity, mid);
}

extern "C" void AndroidHideAds() {
  jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_activity_class, "hideAds", "()V"));
  jni_env->CallVoidMethod(jni_activity, mid);
}

extern "C" void AndroidGPlusSignin() {
  if (jni_gplus) {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "signIn", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
  } else ERRORf("no gplus %p", jni_gplus);
}

extern "C" void AndroidGPlusSignout() {
  if (jni_gplus) {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "signOut", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
  } else ERRORf("no gplus %p", jni_gplus);
}

extern "C" int AndroidGPlusSignedin() {
  if (jni_gplus) {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "signedIn", "()Z"));
    return jni_env->CallBooleanMethod(jni_gplus, mid);
  } else { ERRORf("no gplus %p", jni_gplus); return 0; }
}

extern "C" int AndroidGPlusQuickGame() {
  if (jni_gplus) {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "quickGame", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
  } else ERRORf("no gplus %p", jni_gplus);
  return 0;
}

extern "C" int AndroidGPlusInvite() {
  if (jni_gplus) {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "inviteGUI", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
  } else ERRORf("no gplus %p", jni_gplus);
  return 0;
}

extern "C" int AndroidGPlusAccept() {
  if (jni_gplus) {
    jmethodID mid; CHECK(mid = jni_env->GetMethodID(jni_gplus_class, "acceptGUI", "()V"));
    jni_env->CallVoidMethod(jni_gplus, mid);
  } else ERRORf("no gplus %p", jni_gplus);
  return 0;
}

extern "C" void AndroidGPlusService(void *s) { gplus_service = s; }

extern "C" int AndroidGPlusSendUnreliable(const char *participant_name, const char *buf, int len) {
  if (jni_gplus) {
    jstring pn = jni_env->NewStringUTF(participant_name);
    jobject bytes = jni_env->NewDirectByteBuffer((void*)buf, len);
    jni_env->CallVoidMethod(jni_gplus, jni_gplus_method_write, pn, bytes);
    jni_env->DeleteLocalRef(bytes);
    jni_env->DeleteLocalRef(pn);
  } else ERRORf("no gplus %p", jni_gplus);
  return 0;
}

extern "C" int AndroidGPlusSendReliable(const char *participant_name, const char *buf, int len) {
  if (jni_gplus) {
    jstring pn = jni_env->NewStringUTF(participant_name);
    jobject bytes = jni_env->NewDirectByteBuffer((void*)buf, len);
    jni_env->CallVoidMethod(jni_gplus, jni_gplus_method_write_with_retry, pn, bytes);
    jni_env->DeleteLocalRef(bytes);
    jni_env->DeleteLocalRef(pn);
  } else ERRORf("no gplus %p", jni_gplus);
  return 0;
}

namespace LFL {
const int Key::Escape     = 0xE100;
const int Key::Return     = 10;
const int Key::Up         = 0xE101;
const int Key::Down       = 0xE102;
const int Key::Left       = 0xE103;
const int Key::Right      = 0xE104;
const int Key::LeftShift  = -7;
const int Key::RightShift = -8;
const int Key::LeftCtrl   = 0xE105;
const int Key::RightCtrl  = 0xE106;
const int Key::LeftCmd    = 0xE107;
const int Key::RightCmd   = 0xE108;
const int Key::Tab        = 0xE109;
const int Key::Space      = ' ';
const int Key::Backspace  = '\b';
const int Key::Delete     = -16;
const int Key::Quote      = '\'';
const int Key::Backquote  = '`';
const int Key::PageUp     = 0xE10A;
const int Key::PageDown   = 0xE10B;
const int Key::F1         = 0xE10C;
const int Key::F2         = 0xE10D;
const int Key::F3         = 0xE10E;
const int Key::F4         = 0xE10F;
const int Key::F5         = 0xE110;
const int Key::F6         = 0xE111;
const int Key::F7         = 0xE112;
const int Key::F8         = 0xE113;
const int Key::F9         = 0xE114;
const int Key::F10        = -30;
const int Key::F11        = -31;
const int Key::F12        = -32;
const int Key::Home       = -33;
const int Key::End        = -34;

struct AndroidFrameworkModule : public Module {
  bool frame_on_keyboard_input = 0, frame_on_mouse_input = 0;
  int Init() {
    INFO("AndroidFrameworkModule::Init()");
    if (AndroidVideoInit(&app->opengles_version)) return -1;
    CHECK(!screen->id.v);
    screen->id = MakeTyped(screen);
    app->windows[screen->id.v] = screen;
    return 0;
  }
  int Frame(unsigned clicks) {
    return app->input->DispatchQueuedInput(frame_on_keyboard_input, frame_on_mouse_input);
  }
};

struct AndroidAudioAssetLoader : public AudioAssetLoader {
  virtual void *LoadAudioFile(const string &filename) { return AndroidLoadMusicResource(filename.c_str()); }
  virtual void UnloadAudioFile(void *h) {}
  virtual void *LoadAudioBuf(const char *buf, int len, const char *mimetype) { return 0; }
  virtual void UnloadAudioBuf(void *h) {}
  virtual void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) { a->handle = handle; }
  virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }
};

extern "C" void AndroidSetFrameOnKeyboardInput(int v) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_keyboard_input = v; }
extern "C" void AndroidSetFrameOnMouseInput   (int v) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_mouse_input    = v; }

string Application::GetClipboardText() { return ""; }
void Application::SetClipboardText(const string &s) {}
int  Application::SetExtraScale(bool v) {}
int  Application::SetMultisample(bool v) {}
void Application::OpenTouchKeyboard()  { AndroidShowOrHideKeyboard(1); }
void Application::CloseTouchKeyboard() { AndroidShowOrHideKeyboard(0); }
void Application::CloseTouchKeyboardAfterReturn(bool v) {} 
Box  Application::GetTouchKeyboardBox() { return Box(); }
void Application::ToggleToolbarButton(const string &n) {}
void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}
void Application::CloseWindow(Window *W) {}
void Application::MakeCurrentWindow(Window *W) {}

void Window::SetCaption(const string &v) {}
void Window::SetResizeIncrements(float x, float y) {}
void Window::SetTransparency(float v) {}
void Window::Reshape(int w, int h) {}

bool Video::CreateWindow(Window *W) { return true; }
void Video::StartWindow(Window *W) {}
int Video::Swap() {
  screen->gd->Flush();
  AndroidVideoSwap();
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void FrameScheduler::Setup() { synchronize_waits = wait_forever_thread = monolithic_frame = 0; }
bool FrameScheduler::DoWait() {
  wait_forever_sockets.Select(-1);
  for (auto &s : wait_forever_sockets.socket)
    if (wait_forever_sockets.GetReadable(s.first)) {
      if (s.first != system_event_socket) app->scheduler.Wakeup(s.second.second);
      else {
        char buf[512];
        int l = read(system_event_socket, buf, sizeof(buf));
        for (const char *p = buf, *e = p + l; p < e; p++) if (*p) return true;
      }
    }
  return false;
}

void FrameScheduler::Wakeup(void *opaque) {
  char c = opaque ? 0 : 'W';
  write(wait_forever_wakeup_socket, &c, 1);
}

bool FrameScheduler::WakeupIn(void *opaque, Time interval, bool force) { return 0; }
void FrameScheduler::ClearWakeupIn() {}
void FrameScheduler::UpdateWindowTargetFPS(Window *w) {}

void FrameScheduler::AddWaitForeverMouse() { AndroidSetFrameOnMouseInput(true); }
void FrameScheduler::DelWaitForeverMouse() { AndroidSetFrameOnMouseInput(false); }
void FrameScheduler::AddWaitForeverKeyboard() { AndroidSetFrameOnKeyboardInput(true); }
void FrameScheduler::DelWaitForeverKeyboard() { AndroidSetFrameOnKeyboardInput(false); }
void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, val);
  wait_forever_sockets.Add(fd, flag, val);
}

void FrameScheduler::DelWaitForeverSocket(Socket fd) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
  wait_forever_sockets.Del(fd);
}

unique_ptr<Module> CreateFrameworkModule() { return make_unique<AndroidFrameworkModule>(); }
unique_ptr<AssetLoaderInterface> CreateAssetLoader() { return make_unique<AndroidAudioAssetLoader>(); }

}; // namespace LFL
