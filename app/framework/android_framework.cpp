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

#include "core/app/loader.h"
#include <android/log.h>
#include <libgen.h>

namespace LFL {
static void *gplus_service;
static int jni_activity_width=0, jni_activity_height=0;
static LFL::JNI *jni = LFL::Singleton<LFL::JNI>::Get();

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
    screen->width = jni_activity_width;
    screen->height = jni_activity_height;

    jfieldID fid = CheckNotNull(jni->env->GetFieldID(jni->activity_class, "egl_version", "I"));
    jint v = CheckNotNull(jni->env->GetIntField(jni->activity, fid));
    app->opengles_version = v;
    INFOf("AndroidVideoInit: %d", v);

    CHECK(!screen->id.v);
    screen->id = MakeTyped(screen);
    app->windows[screen->id.v] = screen;
    return 0;
  }

  int Frame(unsigned clicks) {
    return app->input->DispatchQueuedInput(frame_on_keyboard_input, frame_on_mouse_input);
  }
};

struct AndroidAssetLoader : public SimpleAssetLoader {
  virtual void UnloadAudioFile(void *h) {}
  virtual void *LoadAudioFile(File*) { return 0; }
  virtual void *LoadAudioFileNamed(const string &filename) {
    char fn[1024];
    snprintf(fn, sizeof(fn), "%s", basename(filename.c_str()));
    char *suffix = strchr(fn, '.');
    if (suffix) *suffix = 0;
    jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "loadMusicResource", "(Ljava/lang/String;)Landroid/media/MediaPlayer;"));
    jstring jfn = jni->env->NewStringUTF(fn);
    jobject handle = jni->env->CallObjectMethod(jni->activity, mid, jfn);
    jni->env->DeleteLocalRef(jfn);
    return handle;
  }

  virtual void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) { a->handle = handle; }
  virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }
};

void JNI::Init(jobject a, bool init) {
  if      (1)           CHECK(activity = env->NewGlobalRef(a));
  if      (1)           CHECK(view     = env->NewGlobalRef(env->GetObjectField(activity, view_id)));
  if      (init)              gplus    = env->NewGlobalRef(env->GetObjectField(activity, gplus_id));
  else if (gplus_class) CHECK(gplus    = env->NewGlobalRef(env->GetObjectField(activity, gplus_id)));
}

void JNI::Free() {
  if (gplus_class) env->DeleteGlobalRef(gplus);    gplus    = 0;
  if (1)           env->DeleteGlobalRef(view);     view     = 0;
  if (1)           env->DeleteGlobalRef(activity); activity = 0;
  jni_activity_width = jni_activity_height = -1;
}

string JNI::GetJNIString(jstring x) {
  const char *buf = env->GetStringUTFChars(x, 0);
  string ret = buf;
  env->ReleaseStringUTFChars(x, buf);
  return ret;
}

int JNI::CheckForException() {
  jthrowable exception = jni->env->ExceptionOccurred();
  if (!exception) return 0;
  jni->env->ExceptionClear();
  LogException(exception);
  return -1;
}

void JNI::LogException(jthrowable &exception) {
  static jmethodID jni_throwable_method_get_cause =
    LFL::CheckNotNull(jni->env->GetMethodID(jni->throwable_class, "getCause", "()Ljava/lang/Throwable;"));
  static jmethodID jni_throwable_method_get_stack_trace =
    LFL::CheckNotNull(jni->env->GetMethodID(jni->throwable_class, "getStackTrace", "()[Ljava/lang/StackTraceElement;"));
  static jmethodID jni_throwable_method_tostring =
    LFL::CheckNotNull(jni->env->GetMethodID(jni->throwable_class, "toString", "()Ljava/lang/String;"));
  static jmethodID jni_frame_method_tostring =
    LFL::CheckNotNull(jni->env->GetMethodID(jni->frame_class, "toString", "()Ljava/lang/String;"));

  jobjectArray frames = (jobjectArray)jni->env->CallObjectMethod(exception, jni_throwable_method_get_stack_trace);
  jsize frames_length = jni->env->GetArrayLength(frames);
  std::string out;

#if 1
  if (frames > 0) {
    jstring msg = (jstring)jni->env->CallObjectMethod(exception, jni_throwable_method_tostring);
    out += jni->GetJNIString(msg);
    jni->env->DeleteLocalRef(msg);
  }
  for (jsize i = 0; i < frames_length; i++) { 
    jobject frame = jni->env->GetObjectArrayElement(frames, i);
    jstring msg = (jstring)jni->env->CallObjectMethod(frame, jni_frame_method_tostring);
    out += "\n    " + jni->GetJNIString(msg);
    jni->env->DeleteLocalRef(msg);
    jni->env->DeleteLocalRef(frame);
  }
  if (frames > 0) {
    jthrowable cause = (jthrowable)jni->env->CallObjectMethod(exception, jni_throwable_method_get_cause);
    if (cause) LogException(cause);
  }  
#endif

  INFOf("JNI::LogException: %s", out.c_str());
}

extern "C" int AndroidAssetRead(const char *fn, char **out, int *size) {
  jmethodID mid;
  jstring jfn = jni->env->NewStringUTF(fn);
  CHECK(mid = jni->env->GetMethodID(jni->activity_class, "getAssets", "()Landroid/content/res/AssetManager;"));
  jobject assets = jni->env->CallObjectMethod(jni->activity, mid);
  jclass assets_class = jni->env->GetObjectClass(assets);

  CHECK(mid = jni->env->GetMethodID(assets_class, "open", "(Ljava/lang/String;)Ljava/io/InputStream;"));
  jobject input = jni->env->CallObjectMethod(assets, mid, jfn);
  jni->env->DeleteLocalRef(jfn);
  jni->env->DeleteLocalRef(assets);
  jni->env->DeleteLocalRef(assets_class);
  if (!input || jni->CheckForException()) return -1;
  jclass input_class = jni->env->GetObjectClass(input);

  CHECK(mid = jni->env->GetMethodID(input_class, "available", "()I"));
  *size = jni->env->CallIntMethod(input, mid);
  jni->env->DeleteLocalRef(input_class);
  if (jni->CheckForException()) { jni->env->DeleteLocalRef(input); return -1; }
  if (!*size) { jni->env->DeleteLocalRef(input); *out=(char*)""; return 0; }

  jclass channels = jni->env->FindClass("java/nio/channels/Channels");
  CHECK(mid = jni->env->GetStaticMethodID(channels, "newChannel", "(Ljava/io/InputStream;)Ljava/nio/channels/ReadableByteChannel;"));
  jobject readable = jni->env->CallStaticObjectMethod(channels, mid, input);
  jclass readable_class = jni->env->GetObjectClass(readable);
  jni->env->DeleteLocalRef(input);
  jni->env->DeleteLocalRef(channels);

  *out = (char*)malloc(*size);
  jobject bytes = jni->env->NewDirectByteBuffer(*out, *size);
  CHECK(mid = jni->env->GetMethodID(readable_class, "read", "(Ljava/nio/ByteBuffer;)I"));
  int ret = jni->env->CallIntMethod(readable, mid, bytes);
  jni->env->DeleteLocalRef(readable);
  jni->env->DeleteLocalRef(readable_class);
  jni->env->DeleteLocalRef(bytes);

  if (ret != *size || jni->CheckForException()) return -1;
  return 0;
}

extern "C" int AndroidDeviceName(char *out, int size) {
  out[0] = 0;
  jmethodID mid;
  CHECK(mid = jni->env->GetMethodID(jni->activity_class, "getModelName", "()Ljava/lang/String;"));
  jstring ret = (jstring)jni->env->CallObjectMethod(jni->activity, mid);
  const char *id = jni->env->GetStringUTFChars(ret, 0);
  strncpy(out, id, size-1);
  out[size-1] = 0;
  return strlen(out);
}

extern "C" void AndroidGPlusService(void *s) { gplus_service = s; }
extern "C" void AndroidGPlusSignin() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "signIn", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
}

extern "C" void AndroidGPlusSignout() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "signOut", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
}

extern "C" int AndroidGPlusSignedin() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "signedIn", "()Z"));
    return jni->env->CallBooleanMethod(jni->gplus, mid);
  } else { ERRORf("no gplus %p", jni->gplus); return 0; }
}

extern "C" int AndroidGPlusQuickGame() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "quickGame", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
  return 0;
}

extern "C" int AndroidGPlusInvite() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "inviteGUI", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
  return 0;
}

extern "C" int AndroidGPlusAccept() {
  if (jni->gplus) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "acceptGUI", "()V"));
    jni->env->CallVoidMethod(jni->gplus, mid);
  } else ERRORf("no gplus %p", jni->gplus);
  return 0;
}

void Application::CloseWindow(Window *W) {}
void Application::MakeCurrentWindow(Window *W) {}

void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}

string Application::GetClipboardText() { return ""; }
void Application::SetClipboardText(const string &s) {}

void Application::AddToolbar(const vector<pair<string, string>>&items) {}
void Application::ToggleToolbarButton(const string &n) {}

void Application::OpenTouchKeyboard() {
  static jmethodID jni_activity_method_show_keyboard =
    jni->env->GetMethodID(jni->activity_class, "showKeyboard", "()V");
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_show_keyboard);
}

void Application::CloseTouchKeyboard() {
  static jmethodID jni_activity_method_hide_keyboard =
    jni->env->GetMethodID(jni->activity_class, "hideKeyboard", "()V");
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_hide_keyboard);
}

void Application::CloseTouchKeyboardAfterReturn(bool v) {} 
void Application::SetTouchKeyboardTiled(bool v) {}
bool Application::GetTouchKeyboardOpened() { return false; }
Box Application::GetTouchKeyboardBox() { return Box(); }

int  Application::SetMultisample(bool v) {}
int  Application::SetExtraScale(bool v) {}
void Application::SetDownScale(bool v) {}

void Window::SetCaption(const string &v) {}
void Window::SetResizeIncrements(float x, float y) {}
void Window::SetTransparency(float v) {}
bool Window::Reshape(int w, int h) { return false; }

bool Video::CreateWindow(Window *W) { return true; }
void Video::StartWindow(Window *W) {}
int Video::Swap() {
  static jmethodID jni_view_method_swap = CheckNotNull(jni->env->GetMethodID(jni->view_class, "swapEGL", "()V"));
  screen->gd->Flush();
  jni->env->CallVoidMethod(jni->view, jni_view_method_swap);
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void FrameScheduler::Setup() { synchronize_waits = wait_forever_thread = monolithic_frame = 0; }
bool FrameScheduler::DoWait() {
  bool wakeup = false;
  wait_forever_sockets.Select(-1);
  for (auto &s : wait_forever_sockets.socket)
    if (wait_forever_sockets.GetReadable(s.first)) {
      if (s.first != system_event_socket) wakeup = true;
      else {
        char buf[512];
        int l = read(system_event_socket, buf, sizeof(buf));
        for (const char *p = buf, *e = p + l; p < e; p++) if (*p) return true;
      }
    }
  return wakeup;
  // if (wakeup) app->scheduler.Wakeup(screen);
  // return false;
}

void FrameScheduler::Wakeup(Window *w) {
  char c = 'W';
  write(wait_forever_wakeup_socket, &c, 1);
}

bool FrameScheduler::WakeupIn(Window*, Time interval, bool force) { return 0; }
void FrameScheduler::ClearWakeupIn(Window*) {}
void FrameScheduler::UpdateWindowTargetFPS(Window *w) {}

void FrameScheduler::AddWaitForeverMouse(Window*)    { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_mouse_input    = true;  }
void FrameScheduler::DelWaitForeverMouse(Window*)    { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_mouse_input    = false; }
void FrameScheduler::AddWaitForeverKeyboard(Window*) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_keyboard_input = true;  }
void FrameScheduler::DelWaitForeverKeyboard(Window*) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_keyboard_input = false; }
void FrameScheduler::AddWaitForeverSocket(Window *w, Socket fd, int flag) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, w);
  wait_forever_sockets.Add(fd, flag, w);
}

void FrameScheduler::DelWaitForeverSocket(Window*, Socket fd) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
  wait_forever_sockets.Del(fd);
}

unique_ptr<Module> CreateFrameworkModule() { return make_unique<AndroidFrameworkModule>(); }
unique_ptr<AssetLoaderInterface> CreateAssetLoader() { return make_unique<AndroidAssetLoader>(); }

extern "C" jint JNI_OnLoad(JavaVM* vm, void* reserved) { return JNI_VERSION_1_4; }

extern "C" void Java_com_lucidfusionlabs_app_Activity_main(JNIEnv *e, jclass c, jobject a) {
  const char *argv[2] = { "lfjni", 0 };
  MyAppCreate(1, argv);
  CHECK(jni->env = e);
  auto env = jni->env;
  INFOf("main: env=%p", env);

  CHECK(jni->activity_class = (jclass)env->NewGlobalRef(env->GetObjectClass(a)));
  CHECK(jni->view_id  = env->GetFieldID(jni->activity_class, "view",  "Lcom/lucidfusionlabs/app/GameView;"));
  CHECK(jni->gplus_id = env->GetFieldID(jni->activity_class, "gplus", "Lcom/lucidfusionlabs/app/GPlusClient;"));
  jni->Init(a, true);

  CHECK(jni->view_class = (jclass)env->NewGlobalRef(env->GetObjectClass(jni->view)));
  if (jni->gplus) CHECK(jni->gplus_class = (jclass)env->NewGlobalRef(env->GetObjectClass(jni->gplus)));
  CHECK(jni->throwable_class = env->FindClass("java/lang/Throwable"));
  CHECK(jni->frame_class = env->FindClass("java/lang/StackTraceElement"));

  int ret = MyAppMain();
  INFOf("main: env=%p ret=%d", env, ret);
  jni->Free();
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_mainloop(JNIEnv *e, jclass c, jobject a) {
  CHECK(jni->env = e);
  INFOf("mainloop: env=%p", jni->env);
  jni->Init(a, false);
  SetLFAppMainThread();
  LFAppResetGL();
  WindowUnMinimized();
  LFAppMainLoop();
  jni->Free();
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
  LFAppWakeup();
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_touch(JNIEnv *e, jclass c, jint action, jfloat x, jfloat y, jfloat p) {
  static float lx[2]={0,0}, ly[2]={0,0};
  int dpind = (/*FLAGS_swap_axis*/ 0) ? y < LFL::screen->width/2 : x < LFL::screen->width/2;
  if (action == AndroidEvent::ACTION_DOWN || action == AndroidEvent::ACTION_POINTER_DOWN) {
    // INFOf("%d down %f, %f", dpind, x, y);
    QueueMouseClick(1, 1, (int)x, LFL::screen->height - (int)y);
    LFAppWakeup();
    LFL::screen->gesture_tap[dpind] = 1;
    LFL::screen->gesture_dpad_x[dpind] = x;
    LFL::screen->gesture_dpad_y[dpind] = y;
    lx[dpind] = x;
    ly[dpind] = y;
  } else if (action == AndroidEvent::ACTION_UP || action == AndroidEvent::ACTION_POINTER_UP) {
    // INFOf("%d up %f, %f", dpind, x, y);
    QueueMouseClick(1, 0, (int)x, LFL::screen->height - (int)y);
    LFAppWakeup();
    LFL::screen->gesture_dpad_stop[dpind] = 1;
    LFL::screen->gesture_dpad_x[dpind] = 0;
    LFL::screen->gesture_dpad_y[dpind] = 0;
  } else if (action == AndroidEvent::ACTION_MOVE) {
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

}; // namespace LFL
