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
const int Key::Insert     = -35;

struct AndroidFrameworkModule : public Module {
  bool frame_on_keyboard_input = 0, frame_on_mouse_input = 0;

  int Init() {
    INFO("AndroidFrameworkModule::Init()");
    auto screen = app->focused;
    screen->x      = jni->activity_box.x;
    screen->y      = jni->activity_box.y;
    screen->width  = jni->activity_box.w;
    screen->height = jni->activity_box.h;

    jfieldID fid = CheckNotNull(jni->env->GetFieldID(jni->activity_class, "egl_version", "I"));
    jint v = CheckNotNull(jni->env->GetIntField(jni->activity, fid));
    app->opengles_version = v;
    INFOf("AndroidFrameworkModule opengles_version: %d", v);

    CHECK(!screen->id.v);
    screen->id = MakeTyped(screen);
    app->windows[screen->id.v] = screen;

    Socket fd[2];
    CHECK(SystemNetwork::OpenSocketPair(fd));
    app->scheduler.AddMainWaitSocket(screen, (app->scheduler.system_event_socket = fd[0]), SocketSet::READABLE);
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

void Application::CloseWindow(Window *W) {}
void Application::MakeCurrentWindow(Window *W) {}

void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}

string Application::GetClipboardText() { return ""; }
void Application::SetClipboardText(const string &s) {}

void Application::OpenTouchKeyboard(bool) {
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

void Application::SetAppFrameEnabled(bool) {}
void Application::SetAutoRotateOrientation(bool) {}
void Application::SetVerticalSwipeRecognizer(int touches) {}
void Application::SetHorizontalSwipeRecognizer(int touches) {}
void Application::SetPanRecognizer(bool enabled) {}
void Application::SetPinchRecognizer(bool enabled) {}
void Application::SetTouchKeyboardTiled(bool v) {}
int  Application::SetMultisample(bool v) {}
int  Application::SetExtraScale(bool v) {}
void Application::SetDownScale(bool v) {}
void Application::ShowSystemStatusBar(bool v) {}

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
  auto gd = app->focused->gd;
  gd->Flush();
  jni->env->CallVoidMethod(jni->view, jni_view_method_swap);
  gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void FrameScheduler::Setup() {
  synchronize_waits = wait_forever_thread = 0;
}

bool FrameScheduler::DoMainWait() {
  bool ret = false;
  wait_forever_sockets.Select(-1);
  for (auto &s : wait_forever_sockets.socket) {
    if (auto f = static_cast<function<bool()>*>(s.second.second)) 
      if ((*f)()) ret = true;
  }
  if (wait_forever_sockets.GetReadable(system_event_socket)) {
    char buf[512];
    read(system_event_socket, buf, sizeof(buf));
    ret = true;
  }
  return ret;
}

void FrameScheduler::Wakeup(Window *w) {
  char c = 'W';
  write(wait_forever_wakeup_socket, &c, 1);
}

bool FrameScheduler::WakeupIn(Window*, Time interval, bool force) { return 0; }
void FrameScheduler::ClearWakeupIn(Window*) {}
void FrameScheduler::UpdateWindowTargetFPS(Window *w) {}

void FrameScheduler::AddMainWaitMouse(Window*)    { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_mouse_input    = true;  }
void FrameScheduler::DelMainWaitMouse(Window*)    { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_mouse_input    = false; }
void FrameScheduler::AddMainWaitKeyboard(Window*) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_keyboard_input = true;  }
void FrameScheduler::DelMainWaitKeyboard(Window*) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_keyboard_input = false; }
void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()> f) {
  if (fd == InvalidSocket) return;
  wait_forever_sockets.Add(fd, flag, f ? new function<bool()>(move(f)) : nullptr);
}

void FrameScheduler::DelMainWaitSocket(Window*, Socket fd) {
  if (fd == InvalidSocket) return;
  auto it = wait_forever_sockets.socket.find(fd);
  if (it == wait_forever_sockets.socket.end()) return;
  if (auto f = static_cast<function<bool()>*>(it->second.second)) delete f;
  wait_forever_sockets.Del(fd);
}

unique_ptr<Module> CreateFrameworkModule() { return make_unique<AndroidFrameworkModule>(); }
unique_ptr<AssetLoaderInterface> CreateAssetLoader() { return make_unique<AndroidAssetLoader>(); }

extern "C" jint JNI_OnLoad(JavaVM* vm, void* reserved) { return JNI_VERSION_1_4; }

extern "C" void Java_com_lucidfusionlabs_app_JModelItem_close(JNIEnv *e, jobject a) {
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppCreate(JNIEnv *e, jobject a) {
  CHECK(jni->env = e);
  CHECK(jni->activity_class = (jclass)e->NewGlobalRef(e->GetObjectClass(a)));
  CHECK(jni->activity_resources = e->GetFieldID(jni->activity_class, "resources", "Landroid/content/res/Resources;"));
  CHECK(jni->activity_view      = e->GetFieldID(jni->activity_class, "view",      "Lcom/lucidfusionlabs/app/MainView;"));
  CHECK(jni->activity_gplus     = e->GetFieldID(jni->activity_class, "gplus",     "Lcom/lucidfusionlabs/app/GPlusClient;"));
  static jmethodID activity_getpkgname_mid =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getPackageName", "()Ljava/lang/String;"));

  jni->Init(a, true);
  jni->package_name = jni->GetJString((jstring)jni->env->CallObjectMethod(jni->activity, activity_getpkgname_mid));
  std::replace(jni->package_name.begin(), jni->package_name.end(), '.', '/');

  CHECK(jni->view_class         = (jclass)e->NewGlobalRef(e->GetObjectClass(jni->view)));
  CHECK(jni->arraylist_class    = (jclass)e->NewGlobalRef(e->FindClass("java/util/ArrayList")));
  CHECK(jni->string_class       = (jclass)e->NewGlobalRef(e->FindClass("java/lang/String")));
  CHECK(jni->pair_class         = (jclass)e->NewGlobalRef(e->FindClass("android/util/Pair")));
  CHECK(jni->resources_class    = (jclass)e->NewGlobalRef(e->FindClass("android/content/res/Resources")));
  CHECK(jni->r_string_class     = (jclass)e->NewGlobalRef(e->FindClass(StrCat(jni->package_name, "/R$string").c_str())));
  CHECK(jni->throwable_class    = (jclass)e->NewGlobalRef(e->FindClass("java/lang/Throwable")));
  CHECK(jni->frame_class        = (jclass)e->NewGlobalRef(e->FindClass("java/lang/StackTraceElement")));
  CHECK(jni->assetmgr_class     = (jclass)e->NewGlobalRef(e->FindClass("android/content/res/AssetManager")));
  CHECK(jni->inputstream_class  = (jclass)e->NewGlobalRef(e->FindClass("java/io/InputStream")));
  CHECK(jni->channels_class     = (jclass)e->NewGlobalRef(e->FindClass("java/nio/channels/Channels")));
  CHECK(jni->readbytechan_class = (jclass)e->NewGlobalRef(e->FindClass("java/nio/channels/ReadableByteChannel")));
  CHECK(jni->jmodelitem_class   = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/JModelItem")));
  CHECK(jni->jalert_class       = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/JAlert")));
  CHECK(jni->jtoolbar_class     = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/JToolbar")));
  CHECK(jni->jmenu_class        = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/JMenu")));
  CHECK(jni->jtable_class       = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/JTable")));
  CHECK(jni->jtextview_class    = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/JTextView")));
  CHECK(jni->jnavigation_class  = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/JNavigation")));
  jclass jmodelitem_class=0, jalert_class=0, jtoolbar_class=0, jtable_class=0, jnavigation_class=0;
  CHECK(jni->arraylist_construct = e->GetMethodID(jni->arraylist_class, "<init>", "()V"));
  CHECK(jni->arraylist_size = e->GetMethodID(jni->arraylist_class, "size", "()I"));
  CHECK(jni->arraylist_get = e->GetMethodID(jni->arraylist_class, "get", "(I)Ljava/lang/Object;"));
  CHECK(jni->arraylist_add = e->GetMethodID(jni->arraylist_class, "add", "(Ljava/lang/Object;)Z"));
  CHECK(jni->jmodelitem_construct = e->GetMethodID(jni->jmodelitem_class, "<init>", "(Ljava/lang/String;Ljava/lang/String;IIIJJJZ)V"));
  CHECK(jni->pair_first  = e->GetFieldID(jni->pair_class, "first",  "Ljava/lang/Object;"));
  CHECK(jni->pair_second = e->GetFieldID(jni->pair_class, "second", "Ljava/lang/Object;"));
  if (jni->gplus) CHECK(jni->gplus_class = (jclass)e->NewGlobalRef(e->GetObjectClass(jni->gplus)));

  static const char *argv[2] = { "LFLApp", 0 };
  MyAppCreate(1, argv);
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppMain(JNIEnv *e, jobject a) {
  CHECK(jni->env = e);
  INFOf("Main: env=%p", jni->env);
  int ret = MyAppMain();
  INFOf("Main: env=%p ret=%d", jni->env, ret);
  jni->Free();
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppNewMainLoop(JNIEnv *e, jobject a, bool reset) {
  CHECK(jni->env = e);
  INFOf("NewMainLoop: env=%p reset=%d", jni->env, reset);
  jni->Init(a, false);
  SetLFAppMainThread();
  if (reset) LFAppResetGL();
  app->focused->UnMinimized();
  int ret = LFAppMainLoop();
  INFOf("NewMainLoop: env=%p ret=%d", jni->env, ret);
  jni->Free();
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppMinimize(JNIEnv* env, jobject a) {
  INFOf("%s", "minimize");
  app->RunInMainThread([=](){ app->focused->Minimized(); });
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppReshaped(JNIEnv *e, jobject a, jint x, jint y, jint w, jint h) { 
  bool init = !jni->activity_box.w && !jni->activity_box.h;
  if (init) { jni->activity_box = Box(x, y, w, h); return; }
  if (jni->activity_box.x == x && jni->activity_box.y == y &&
      jni->activity_box.w == w && jni->activity_box.h == h) return;
  jni->activity_box = Box(x, y, w, h);
  app->RunInMainThread([=](){ app->focused->Reshaped(Box(x, y, w, h)); });
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppKeyPress(JNIEnv *e, jobject a, jint keycode, jint mod, jint down) {
  app->input->KeyPress(keycode, mod, down);
  LFAppWakeup();
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppTouch(JNIEnv *e, jobject a, jint action, jfloat x, jfloat y, jfloat p) {
  static float lx[2]={0,0}, ly[2]={0,0};
  auto screen = app->focused;
  int dpind = (/*FLAGS_swap_axis*/ 0) ? y < screen->width/2 : x < screen->width/2;
  if (action == AndroidEvent::ACTION_DOWN || action == AndroidEvent::ACTION_POINTER_DOWN) {
    // INFOf("%d down %f, %f", dpind, x, screen->height - y);
    app->input->QueueMouseClick(1, 1, point(screen->x + x, screen->y + screen->height - y));
    LFAppWakeup();
    // screen->gesture_tap[dpind] = 1;
    // screen->gesture_dpad_x[dpind] = x;
    // screen->gesture_dpad_y[dpind] = y;
    lx[dpind] = x;
    ly[dpind] = y;
  } else if (action == AndroidEvent::ACTION_UP || action == AndroidEvent::ACTION_POINTER_UP) {
    // INFOf("%d up %f, %f", dpind, x, y);
    app->input->QueueMouseClick(1, 0, point(screen->x + x, screen->y + screen->height - y));
    LFAppWakeup();
    // screen->gesture_dpad_stop[dpind] = 1;
    // screen->gesture_dpad_x[dpind] = 0;
    // screen->gesture_dpad_y[dpind] = 0;
  } else if (action == AndroidEvent::ACTION_MOVE) {
    float vx = x - lx[dpind];
    float vy = y - ly[dpind];
    lx[dpind] = x;
    ly[dpind] = y;
    // INFOf("%d move %f, %f vel = %f, %f", dpind, x, y, vx, vy);
    if (vx > 1.5 || vx < -1.5 || vy > 1.5 || vy < -1.5) {
      // screen->gesture_dpad_dx[dpind] = vx;
      // screen->gesture_dpad_dy[dpind] = vy;
    }
    // screen->gesture_dpad_x[dpind] = x;
    // screen->gesture_dpad_y[dpind] = y;
  } else INFOf("unhandled action %d", action);
} 

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppFling(JNIEnv *e, jobject a, jfloat x, jfloat y, jfloat vx, jfloat vy) {
  auto screen = app->focused;
  int dpind = y < screen->width/2;
  // screen->gesture_dpad_dx[dpind] = vx;
  // screen->gesture_dpad_dy[dpind] = vy;
  INFOf("fling(%f, %f) = %d of (%d, %d) and vel = (%f, %f)", x, y, dpind, screen->width, screen->height, vx, vy);
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppScroll(JNIEnv *e, jobject a, jfloat x, jfloat y, jfloat vx, jfloat vy) {
  // screen->gesture_swipe_up = screen->gesture_swipe_down = 0;
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppAccel(JNIEnv *e, jobject a, jfloat x, jfloat y, jfloat z) {}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppShellRun(JNIEnv *e, jobject a, jstring text) {
  app->focused->shell->Run(e->GetStringUTFChars(text, 0));
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppRunCallbackInMainThread(JNIEnv *e, jobject a, jlong cb) {
  app->RunCallbackInMainThread(new Callback(*static_cast<Callback*>(Void(cb))));
}

extern "C" void Java_com_lucidfusionlabs_app_MainActivity_AppRunStringCBInMainThread(JNIEnv *e, jobject a, jlong cb, jstring text) {
  string t = JNI::GetEnvJString(e, text);
  app->RunCallbackInMainThread(new Callback([=](){ (*static_cast<StringCB*>(Void(cb)))(t); }));
}

extern "C" void Java_com_lucidfusionlabs_app_GPlusClient_startGame(JNIEnv *e, jobject a, jboolean server, jstring pid) {
  char buf[128];
  const char *participant_id = e->GetStringUTFChars(pid, 0);
  snprintf(buf, sizeof(buf), "%s %s", server ? "gplus_server" : "gplus_client", participant_id);
  app->focused->shell->Run(buf);
  e->ReleaseStringUTFChars(pid, participant_id);
}

extern "C" void Java_com_lucidfusionlabs_app_GPlusClient_read(JNIEnv *e, jobject a, jstring pid, jobject bb, jint len) {
  static GPlus *gplus = Singleton<GPlus>::Get();
  const char *participant_id = e->GetStringUTFChars(pid, 0);
  if (gplus->server) gplus->server->EndpointRead(participant_id, (const char*)e->GetDirectBufferAddress(bb), len);
  e->ReleaseStringUTFChars(pid, participant_id);
}

}; // namespace LFL
