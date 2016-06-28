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
static Box activity_box;
static JNI *jni = Singleton<JNI>::Get();

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
    screen->x      = activity_box.x;
    screen->y      = activity_box.y;
    screen->width  = activity_box.w;
    screen->height = activity_box.h;

    jfieldID fid = CheckNotNull(jni->env->GetFieldID(jni->activity_class, "egl_version", "I"));
    jint v = CheckNotNull(jni->env->GetIntField(jni->activity, fid));
    app->opengles_version = v;
    INFOf("AndroidFrameworkModule opengles_version: %d", v);

    CHECK(!screen->id.v);
    screen->id = MakeTyped(screen);
    app->windows[screen->id.v] = screen;

    Socket fd[2];
    CHECK(SystemNetwork::OpenSocketPair(fd));
    app->scheduler.AddWaitForeverSocket(screen, (app->scheduler.system_event_socket = fd[0]), SocketSet::READABLE);
    app->scheduler.wait_forever_wakeup_socket = fd[1];
    app->scheduler.Wakeup(screen);
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
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "loadMusicResource", "(Ljava/lang/String;)Landroid/media/MediaPlayer;"));
    string fn = basename(filename.c_str());
    jstring jfn = jni->env->NewStringUTF(fn.substr(0, fn.find('.')).c_str());
    jobject handle = jni->env->CallObjectMethod(jni->activity, mid, jfn);
    jni->env->DeleteLocalRef(jfn);
    return jni->env->NewGlobalRef(handle);
  }

  virtual void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) { a->handle = handle; }
  virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }
};

void JNI::Init(jobject a, bool first) {
  if      (1)           CHECK(activity = env->NewGlobalRef(a));
  if      (1)           CHECK(view     = env->NewGlobalRef(env->GetObjectField(activity, view_id)));
  if      (first)             gplus    = env->NewGlobalRef(env->GetObjectField(activity, gplus_id));
  else if (gplus_class) CHECK(gplus    = env->NewGlobalRef(env->GetObjectField(activity, gplus_id)));
}

void JNI::Free() {
  if (gplus_class) env->DeleteGlobalRef(gplus);    gplus    = 0;
  if (1)           env->DeleteGlobalRef(view);     view     = 0;
  if (1)           env->DeleteGlobalRef(activity); activity = 0;
  activity_box = Box(-1, -1);
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
    CheckNotNull(jni->env->GetMethodID(jni->throwable_class, "getCause", "()Ljava/lang/Throwable;"));
  static jmethodID jni_throwable_method_get_stack_trace =
    CheckNotNull(jni->env->GetMethodID(jni->throwable_class, "getStackTrace", "()[Ljava/lang/StackTraceElement;"));
  static jmethodID jni_throwable_method_tostring =
    CheckNotNull(jni->env->GetMethodID(jni->throwable_class, "toString", "()Ljava/lang/String;"));
  static jmethodID jni_frame_method_tostring =
    CheckNotNull(jni->env->GetMethodID(jni->frame_class, "toString", "()Ljava/lang/String;"));

  jobjectArray frames = (jobjectArray)jni->env->CallObjectMethod(exception, jni_throwable_method_get_stack_trace);
  jsize frames_length = jni->env->GetArrayLength(frames);
  string out;

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

  INFOf("JNI::LogException: %s", out.c_str());
}

BufferFile *JNI::OpenAsset(const string &fn) {
  static jmethodID get_assets_mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getAssets", "()Landroid/content/res/AssetManager;"));
  static jmethodID assetmgr_open_mid = CheckNotNull(jni->env->GetMethodID(jni->assetmgr_class, "open", "(Ljava/lang/String;)Ljava/io/InputStream;"));
  static jmethodID inputstream_avail_mid = CheckNotNull(jni->env->GetMethodID(jni->inputstream_class, "available", "()I"));
  static jmethodID channels_newchan_mid = CheckNotNull(jni->env->GetStaticMethodID(jni->channels_class, "newChannel", "(Ljava/io/InputStream;)Ljava/nio/channels/ReadableByteChannel;"));
  static jmethodID readbytechan_read_mid = CheckNotNull(jni->env->GetMethodID(jni->readbytechan_class, "read", "(Ljava/nio/ByteBuffer;)I"));

  jstring jfn = jni->env->NewStringUTF(fn.c_str());
  jobject assets = jni->env->CallObjectMethod(jni->activity, get_assets_mid);
  jobject input = jni->env->CallObjectMethod(assets, assetmgr_open_mid, jfn);
  jni->env->DeleteLocalRef(jfn);
  jni->env->DeleteLocalRef(assets);
  if (!input || jni->CheckForException()) return nullptr;

  int len = jni->env->CallIntMethod(input, inputstream_avail_mid);
  if (jni->CheckForException()) { jni->env->DeleteLocalRef(input); return nullptr; }

  unique_ptr<BufferFile> ret = make_unique<BufferFile>(string(), fn.c_str());
  if (!len) { jni->env->DeleteLocalRef(input); return ret.release(); }
  ret->buf.resize(len);

  jobject readable = jni->env->CallStaticObjectMethod(jni->channels_class, channels_newchan_mid, input);
  jni->env->DeleteLocalRef(input);

  jobject bytes = jni->env->NewDirectByteBuffer(&ret->buf[0], ret->buf.size());
  len = jni->env->CallIntMethod(readable, readbytechan_read_mid, bytes);
  jni->env->DeleteLocalRef(readable);
  jni->env->DeleteLocalRef(bytes);

  if (len != ret->buf.size() || jni->CheckForException()) return nullptr;
  return ret.release();
}

string JNI::GetDeviceName() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getModelName", "()Ljava/lang/String;"));
  jstring ret = (jstring)jni->env->CallObjectMethod(jni->activity, mid);
  return GetJNIString(ret);
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
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "showKeyboard", "()V"));
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_show_keyboard);
}

void Application::CloseTouchKeyboard() {
  static jmethodID jni_activity_method_hide_keyboard =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "hideKeyboard", "()V"));
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_hide_keyboard);
}

void Application::ToggleTouchKeyboard() {
  static jmethodID jni_activity_method_toggle_keyboard =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "toggleKeyboard", "()V"));
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_toggle_keyboard);
}

void Application::CloseTouchKeyboardAfterReturn(bool v) {
#if 0
  static jmethodID jni_activity_method_hide_keyboard_after_enter =
    jni->env->GetMethodID(jni->activity_class, "hideKeyboardAfterEnter", "()V");
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_hide_keyboard_after_enter);
#endif
} 

void Application::SetTouchKeyboardTiled(bool v) {}
Box Application::GetTouchKeyboardBox() { return Box(); }

int  Application::SetMultisample(bool v) {}
int  Application::SetExtraScale(bool v) {}
void Application::SetDownScale(bool v) {}

void Application::SetTitleBar(bool v) {
  if (!v) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "disableTitle", "()V"));
    jni->env->CallVoidMethod(jni->activity, mid);
  }
}

void Application::SetKeepScreenOn(bool v) {
  if (v) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "enableKeepScreenOn", "()V"));
    jni->env->CallVoidMethod(jni->activity, mid);
  }
}

void Window::SetCaption(const string &text) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setCaption", "(Ljava/lang/String;)V"));
  jstring jtext = jni->env->NewStringUTF(text.c_str());
  jni->env->CallVoidMethod(jni->activity, mid, jtext);
  jni->env->DeleteLocalRef(jtext);
}

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

void FrameScheduler::Setup() {
  synchronize_waits = wait_forever_thread = 0;
}

bool FrameScheduler::DoWait() {
  wait_forever_sockets.Select(-1);
  if (wait_forever_sockets.GetReadable(system_event_socket)) {
    char buf[512];
    read(system_event_socket, buf, sizeof(buf));
    return true;
  }
  return true;
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

unique_ptr<Module> CreateFrameworkModule() { return make_unique<AndroidFrameworkModule>(); }
unique_ptr<AssetLoaderInterface> CreateAssetLoader() { return make_unique<AndroidAssetLoader>(); }

extern "C" jint JNI_OnLoad(JavaVM* vm, void* reserved) { return JNI_VERSION_1_4; }

extern "C" void Java_com_lucidfusionlabs_app_Activity_Create(JNIEnv *e, jclass c, jobject a) {
  CHECK(jni->env = e);
  CHECK(jni->activity_class = (jclass)e->NewGlobalRef(e->GetObjectClass(a)));
  CHECK(jni->view_id  = e->GetFieldID(jni->activity_class, "view",  "Lcom/lucidfusionlabs/app/GameView;"));
  CHECK(jni->gplus_id = e->GetFieldID(jni->activity_class, "gplus", "Lcom/lucidfusionlabs/app/GPlusClient;"));
  jni->Init(a, true);

  CHECK(jni->view_class         = (jclass)e->NewGlobalRef(e->GetObjectClass(jni->view)));
  CHECK(jni->throwable_class    = (jclass)e->NewGlobalRef(e->FindClass("java/lang/Throwable")));
  CHECK(jni->frame_class        = (jclass)e->NewGlobalRef(e->FindClass("java/lang/StackTraceElement")));
  CHECK(jni->assetmgr_class     = (jclass)e->NewGlobalRef(e->FindClass("android/content/res/AssetManager")));
  CHECK(jni->inputstream_class  = (jclass)e->NewGlobalRef(e->FindClass("java/io/InputStream")));
  CHECK(jni->channels_class     = (jclass)e->NewGlobalRef(e->FindClass("java/nio/channels/Channels")));
  CHECK(jni->readbytechan_class = (jclass)e->NewGlobalRef(e->FindClass("java/nio/channels/ReadableByteChannel")));
  if (jni->gplus) CHECK(jni->gplus_class = (jclass)e->NewGlobalRef(e->GetObjectClass(jni->gplus)));

  static const char *argv[2] = { "lfl_jni", 0 };
  MyAppCreate(1, argv);
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_Main(JNIEnv *e, jclass c, jobject a) {
  CHECK(jni->env = e);
  INFOf("Main: env=%p", jni->env);
  int ret = MyAppMain();
  INFOf("Main: env=%p ret=%d", jni->env, ret);
  jni->Free();
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_NewMainLoop(JNIEnv *e, jclass c, jobject a, bool reset) {
  CHECK(jni->env = e);
  INFOf("NewMainLoop: env=%p reset=%d", jni->env, reset);
  jni->Init(a, false);
  SetLFAppMainThread();
  if (reset) LFAppResetGL();
  WindowUnMinimized();
  int ret = LFAppMainLoop();
  INFOf("NewMainLoop: env=%p ret=%d", jni->env, ret);
  jni->Free();
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_Minimize(JNIEnv* env, jclass c) {
  INFOf("%s", "minimize");
  QueueWindowMinimized();
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_Reshaped(JNIEnv *e, jclass c, jint x, jint y, jint w, jint h) { 
  bool init = !activity_box.w && !activity_box.h;
  if (init) { activity_box = Box(x, y, w, h); return; }
  if (activity_box.x == x && activity_box.y == y && activity_box.w == w && activity_box.h == h) return;
  activity_box = Box(x, y, w, h);
  QueueWindowReshaped(x, y, w, h);
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_KeyPress(JNIEnv *e, jclass c, jint down, jint keycode) {
  QueueKeyPress(keycode, down);
  LFAppWakeup();
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_Touch(JNIEnv *e, jclass c, jint action, jfloat x, jfloat y, jfloat p) {
  static float lx[2]={0,0}, ly[2]={0,0};
  int dpind = (/*FLAGS_swap_axis*/ 0) ? y < screen->width/2 : x < screen->width/2;
  if (action == AndroidEvent::ACTION_DOWN || action == AndroidEvent::ACTION_POINTER_DOWN) {
    // INFOf("%d down %f, %f", dpind, x, screen->height - y);
    QueueMouseClick(1, 1, screen->x + x, screen->y + screen->height - y);
    LFAppWakeup();
    screen->gesture_tap[dpind] = 1;
    screen->gesture_dpad_x[dpind] = x;
    screen->gesture_dpad_y[dpind] = y;
    lx[dpind] = x;
    ly[dpind] = y;
  } else if (action == AndroidEvent::ACTION_UP || action == AndroidEvent::ACTION_POINTER_UP) {
    // INFOf("%d up %f, %f", dpind, x, y);
    QueueMouseClick(1, 0, screen->x + x, screen->y + screen->height - y);
    LFAppWakeup();
    screen->gesture_dpad_stop[dpind] = 1;
    screen->gesture_dpad_x[dpind] = 0;
    screen->gesture_dpad_y[dpind] = 0;
  } else if (action == AndroidEvent::ACTION_MOVE) {
    float vx = x - lx[dpind];
    float vy = y - ly[dpind];
    lx[dpind] = x;
    ly[dpind] = y;
    // INFOf("%d move %f, %f vel = %f, %f", dpind, x, y, vx, vy);
    if (vx > 1.5 || vx < -1.5 || vy > 1.5 || vy < -1.5) {
      screen->gesture_dpad_dx[dpind] = vx;
      screen->gesture_dpad_dy[dpind] = vy;
    }
    screen->gesture_dpad_x[dpind] = x;
    screen->gesture_dpad_y[dpind] = y;
  } else INFOf("unhandled action %d", action);
} 

extern "C" void Java_com_lucidfusionlabs_app_Activity_Fling(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat vx, jfloat vy) {
  int dpind = y < screen->width/2;
  screen->gesture_dpad_dx[dpind] = vx;
  screen->gesture_dpad_dy[dpind] = vy;
  INFOf("fling(%f, %f) = %d of (%d, %d) and vel = (%f, %f)", x, y, dpind, screen->width, screen->height, vx, vy);
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_Scroll(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat vx, jfloat vy) {
  screen->gesture_swipe_up = screen->gesture_swipe_down = 0;
}

extern "C" void Java_com_lucidfusionlabs_app_Activity_Accel(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat z) {}

extern "C" void Java_com_lucidfusionlabs_app_GPlusClient_startGame(JNIEnv *e, jclass c, jboolean server, jstring pid) {
  char buf[128];
  const char *participant_id = e->GetStringUTFChars(pid, 0);
  snprintf(buf, sizeof(buf), "%s %s", server ? "gplus_server" : "gplus_client", participant_id);
  ShellRun(buf);
  e->ReleaseStringUTFChars(pid, participant_id);
}

extern "C" void Java_com_lucidfusionlabs_app_GPlusClient_read(JNIEnv *e, jclass c, jstring pid, jobject bb, jint len) {
  static GPlus *gplus = Singleton<GPlus>::Get();
  const char *participant_id = e->GetStringUTFChars(pid, 0);
  if (gplus->server) EndpointRead(gplus->server, participant_id, (const char*)e->GetDirectBufferAddress(bb), len);
  e->ReleaseStringUTFChars(pid, participant_id);
}

}; // namespace LFL
