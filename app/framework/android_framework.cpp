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

const int Texture::updatesystemimage_pf = Pixel::BGRA;

struct AndroidWindow : public Window {
  void SetCaption(const string &text) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setCaption", "(Ljava/lang/String;)V"));
    LocalJNIString jtext(jni->env, JNI::ToJString(jni->env, text));
    jni->env->CallVoidMethod(jni->activity, mid, jtext.v);
  }

  void SetResizeIncrements(float x, float y) {}
  void SetTransparency(float v) {}
  bool Reshape(int w, int h) { return false; }
};

struct AndroidFrameworkModule : public Module {
  bool frame_on_keyboard_input = 0, frame_on_mouse_input = 0;

  int Init() {
    jfieldID w_fid = CheckNotNull(jni->env->GetFieldID(jni->view_class, "surface_width", "I"));
    jfieldID h_fid = CheckNotNull(jni->env->GetFieldID(jni->view_class, "surface_height", "I"));
    jfieldID v_fid = CheckNotNull(jni->env->GetFieldID(jni->view_class, "egl_version", "I"));

    app->focused->width = CheckNotNull(jni->env->GetIntField(jni->view, w_fid));
    app->focused->height = CheckNotNull(jni->env->GetIntField(jni->view, h_fid));
    app->opengles_version = CheckNotNull(jni->env->GetIntField(jni->view, v_fid));
    INFO("AndroidFrameworkModule::Init(), opengles_version=", app->opengles_version);

    auto screen = app->focused;
    CHECK(!screen->id);
    screen->id = screen;
    app->windows[screen->id] = screen;

    Socket fd[2];
    CHECK(SystemNetwork::OpenSocketPair(fd));
    app->scheduler.AddMainWaitSocket(screen, (app->scheduler.system_event_socket = fd[0]), SocketSet::READABLE);
    app->scheduler.main_wait_wakeup_socket = fd[1];
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
    LocalJNIString jfn(jni->env, JNI::ToJString(jni->env, fn.substr(0, fn.find('.'))));
    LocalJNIObject handle(jni->env, jni->env->CallObjectMethod(jni->activity, mid, jfn.v));
    return jni->env->NewGlobalRef(handle.v);
  }

  virtual void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) { a->handle = handle; }
  virtual int RefillAudio(SoundAsset *a, int reset) { return 0; }
};

struct AndroidTimer : public TimerInterface {
  Callback cb;
  GlobalJNIObject fired_cb;
  Time next = Time::zero();
  AndroidTimer(Callback c) :
    cb(move(c)), fired_cb(JNI::ToNativeCallback(jni->env, bind(&AndroidTimer::FiredCB, this))) {}

  void FiredCB() {
    next = Time::zero();
    cb();
  }

  bool Clear() {
    if (next == Time::zero()) return false;
    next = Time::zero();
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->handler_class, "removeCallbacks", "(Ljava/lang/Runnable;)V"));
    jni->env->CallVoidMethod(jni->handler, mid, fired_cb.v);
    return true;
  }

  void Run(Time interval, bool force=false) {
    Time target = Now() + interval;
    if (next != Time::zero()) {
      if (force || target.count() < next.count()) { Clear(); next=target; }
    } else next = target;
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->handler_class, "postDelayed", "(Ljava/lang/Runnable;J)Z"));
    jni->env->CallBooleanMethod(jni->handler, mid, fired_cb.v, jlong(interval.count()));
  }
};

int Application::Suspended() {
  INFO("Application::Suspended");
  return 0;
}

void Application::RunCallbackInMainThread(Callback cb) {
  message_queue.Write(new Callback(move(cb)));
  if (!FLAGS_target_fps) scheduler.Wakeup(focused);
}

void Application::CloseWindow(Window *W) {}
void Application::MakeCurrentWindow(Window *W) {}
void Application::GrabMouseFocus() {}
void Application::ReleaseMouseFocus() {}

string Application::GetClipboardText() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getClipboardText", "()Ljava/lang/String;"));
  LocalJNIString text(jni->env, jstring(jni->env->CallObjectMethod(jni->activity, mid)));
  return JNI::GetJString(jni->env, text.v); 
}

void Application::SetClipboardText(const string &s) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setClipboardText", "(Ljava/lang/String;)V"));
  LocalJNIString v(jni->env, JNI::ToJString(jni->env, s));
  jni->env->CallVoidMethod(jni->activity, mid, v.v);
}

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

void Application::SetAppFrameEnabled(bool v) {
  if ((app->frame_disabled = !v)) DrawSplash(Color::black);
  INFO("Application frame_disabled = ", app->frame_disabled);
}

void Application::SetPanRecognizer(bool enabled) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "enablePanRecognizer", "(Z)V"));
  jni->env->CallVoidMethod(jni->activity, mid, enabled);
}

void Application::SetPinchRecognizer(bool enabled) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "enablePinchRecognizer", "(Z)V"));
  jni->env->CallVoidMethod(jni->activity, mid, enabled);
}

void Application::SetAutoRotateOrientation(bool) {}
void Application::SetVerticalSwipeRecognizer(int touches) {}
void Application::SetHorizontalSwipeRecognizer(int touches) {}
void Application::SetTouchKeyboardTiled(bool v) {}
int  Application::SetMultisample(bool v) { return 0; }
int  Application::SetExtraScale(bool v) { return 0; }
void Application::SetDownScale(bool v) {}
void Application::ShowSystemStatusBar(bool v) {}

string Application::PrintCallStack() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getCurrentStackTrace", "()Ljava/lang/String;"));
  LocalJNIString text(jni->env, jstring(jni->env->CallObjectMethod(jni->activity, mid)) );
  return JNI::GetJString(jni->env, text.v); 
}

void Application::SetTitleBar(bool v) {
  if (!v) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "disableTitle", "()V"));
    jni->env->CallVoidMethod(jni->activity, mid);
  }
}

void Application::SetKeepScreenOn(bool v) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "enableKeepScreenOn", "(Z)V"));
  jni->env->CallVoidMethod(jni->activity, mid, v);
}

void Application::SetTheme(const string &v) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setTheme", "(Ljava/lang/String;)V"));
  LocalJNIString vstr(jni->env, JNI::ToJString(jni->env, v));
  jni->env->CallVoidMethod(jni->activity, mid, vstr.v);
}

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

FrameScheduler::FrameScheduler() :
  maxfps(&FLAGS_target_fps), wakeup_thread(&frame_mutex, &wait_mutex), rate_limit(1), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(1), run_main_loop(1) {}

bool FrameScheduler::DoMainWait() {
  bool ret = false;
  main_wait_sockets.Select(-1);
  for (auto i = main_wait_sockets.socket.begin(); i != main_wait_sockets.socket.end(); /**/) {
    iter_socket = i->first;
    auto f = static_cast<function<bool()>*>(i++->second.second);
    if (f) if ((*f)()) ret = true;
  }
  int wakeups = 0;
  iter_socket = InvalidSocket;
  if (main_wait_sockets.GetReadable(system_event_socket)) {
    char buf[512];
    wakeups = read(system_event_socket, buf, sizeof(buf));
    if (wakeups >= 0) for (auto p = buf, e = p + wakeups; p != e; ++p) if (*p == 'W') { ret = true; break; }
  }
  return ret;
}

void FrameScheduler::UpdateWindowTargetFPS(Window *w) {}
void FrameScheduler::Wakeup(Window*, int flag) {
  char c = (flag & WakeupFlag::ContingentOnEvents) ? ' ' : 'W';
  write(main_wait_wakeup_socket, &c, 1);
}

void FrameScheduler::AddMainWaitMouse(Window*)    { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_mouse_input    = true;  }
void FrameScheduler::DelMainWaitMouse(Window*)    { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_mouse_input    = false; }
void FrameScheduler::AddMainWaitKeyboard(Window*) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_keyboard_input = true;  }
void FrameScheduler::DelMainWaitKeyboard(Window*) { dynamic_cast<AndroidFrameworkModule*>(app->framework.get())->frame_on_keyboard_input = false; }
void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()> f) {
  if (fd == InvalidSocket) return;
  main_wait_sockets.Add(fd, flag, f ? new function<bool()>(move(f)) : nullptr);
}

void FrameScheduler::DelMainWaitSocket(Window*, Socket fd) {
  if (fd == InvalidSocket) return;
  if (iter_socket != InvalidSocket)
    CHECK_EQ(iter_socket, fd) << "Can only remove current socket from wait callback";
  auto it = main_wait_sockets.socket.find(fd);
  if (it == main_wait_sockets.socket.end()) return;
  if (auto f = static_cast<function<bool()>*>(it->second.second)) delete f;
  main_wait_sockets.Del(fd);
}

Window *Window::Create() { return new AndroidWindow(); }
Application *CreateApplication(int ac, const char* const* av) { return new Application(ac, av); }
unique_ptr<Module> CreateFrameworkModule() { return make_unique<AndroidFrameworkModule>(); }
unique_ptr<AssetLoaderInterface> CreateAssetLoader() { return make_unique<AndroidAssetLoader>(); }
unique_ptr<TimerInterface> SystemToolkit::CreateTimer(Callback cb) { return make_unique<AndroidTimer>(move(cb)); };

static void NativeAPI_shutdownMainLoop() {
  app->suspended = true;
  if (app->focused->unfocused_cb) app->focused->unfocused_cb();
  while (app->message_queue.HandleMessages()) {}
  app->focused->gd->Finish();
}

extern "C" jint JNI_OnLoad(JavaVM* vm, void* reserved) { return JNI_VERSION_1_4; }

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_create(JNIEnv *e, jclass c, jobject a) {
  CHECK(jni->env = e);
  CHECK(jni->activity_class = (jclass)e->NewGlobalRef(e->GetObjectClass(a)));
  CHECK(jni->activity_resources = e->GetFieldID(jni->activity_class, "resources", "Landroid/content/res/Resources;"));
  CHECK(jni->activity_view      = e->GetFieldID(jni->activity_class, "gl_view",   "Lcom/lucidfusionlabs/app/OpenGLView;"));
  CHECK(jni->activity_handler   = e->GetFieldID(jni->activity_class, "handler",   "Landroid/os/Handler;"));
  // CHECK(jni->activity_gplus     = e->GetFieldID(jni->activity_class, "gplus",     "Lcom/lucidfusionlabs/app/GPlusClient;"));
  static jmethodID activity_getpkgname_mid =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getPackageName", "()Ljava/lang/String;"));

  jni->Init(a, true);
  jni->package_name = jni->GetJString(jni->env, (jstring)jni->env->CallObjectMethod(jni->activity, activity_getpkgname_mid));
  std::replace(jni->package_name.begin(), jni->package_name.end(), '.', '/');

  CHECK(jni->view_class         = (jclass)e->NewGlobalRef(e->GetObjectClass(jni->view)));
  CHECK(jni->arraylist_class    = (jclass)e->NewGlobalRef(e->FindClass("java/util/ArrayList")));
  CHECK(jni->hashmap_class      = (jclass)e->NewGlobalRef(e->FindClass("java/util/HashMap")));
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
  CHECK(jni->handler_class      = (jclass)e->NewGlobalRef(e->FindClass("android/os/Handler")));
  CHECK(jni->modelitem_class    = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/core/ModelItem")));
  CHECK(jni->modelitemchange_class=(jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/core/ModelItemChange")));
  CHECK(jni->pickeritem_class   = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/core/PickerItem")));
  CHECK(jni->toolbar_class      = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/Toolbar")));
  CHECK(jni->alertscreen_class  = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/AlertScreen")));
  CHECK(jni->menuscreen_class   = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/MenuScreen")));
  CHECK(jni->tablescreen_class  = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/TableScreen")));
  CHECK(jni->textscreen_class   = (jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/TextScreen")));
  CHECK(jni->screennavigator_class=(jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/app/ScreenFragmentNavigator")));
  CHECK(jni->nativecallback_class=(jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/core/NativeCallback")));
  CHECK(jni->nativestringcb_class=(jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/core/NativeStringCB")));
  CHECK(jni->nativeintcb_class   =(jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/core/NativeIntCB")));
  CHECK(jni->nativeintintcb_class=(jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/core/NativeIntIntCB")));
  CHECK(jni->nativepickeritemcb_class=(jclass)e->NewGlobalRef(e->FindClass("com/lucidfusionlabs/core/NativePickerItemCB")));
  CHECK(jni->int_class          = (jclass)e->NewGlobalRef(e->FindClass("java/lang/Integer")));
  CHECK(jni->long_class         = (jclass)e->NewGlobalRef(e->FindClass("java/lang/Long")));
  CHECK(jni->arraylist_construct = e->GetMethodID(jni->arraylist_class, "<init>", "()V"));
  CHECK(jni->arraylist_size = e->GetMethodID(jni->arraylist_class, "size", "()I"));
  CHECK(jni->arraylist_get = e->GetMethodID(jni->arraylist_class, "get", "(I)Ljava/lang/Object;"));
  CHECK(jni->arraylist_add = e->GetMethodID(jni->arraylist_class, "add", "(Ljava/lang/Object;)Z"));
  CHECK(jni->hashmap_construct = e->GetMethodID(jni->hashmap_class, "<init>", "()V"));
  CHECK(jni->hashmap_size = e->GetMethodID(jni->hashmap_class, "size", "()I"));
  CHECK(jni->hashmap_get = e->GetMethodID(jni->hashmap_class, "get", "(Ljava/lang/Object;)Ljava/lang/Object;"));
  CHECK(jni->hashmap_put = e->GetMethodID(jni->hashmap_class, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;"));
  CHECK(jni->pair_construct = e->GetMethodID(jni->pair_class, "<init>", "(Ljava/lang/Object;Ljava/lang/Object;)V"));
  CHECK(jni->pair_first  = e->GetFieldID(jni->pair_class, "first",  "Ljava/lang/Object;"));
  CHECK(jni->pair_second = e->GetFieldID(jni->pair_class, "second", "Ljava/lang/Object;"));
  CHECK(jni->modelitem_construct = e->GetMethodID(jni->modelitem_class, "<init>", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IIIIIIILcom/lucidfusionlabs/core/NativeCallback;Lcom/lucidfusionlabs/core/NativeStringCB;Lcom/lucidfusionlabs/core/PickerItem;ZII)V"));
  CHECK(jni->modelitemchange_construct = e->GetMethodID(jni->modelitemchange_class, "<init>", "(IIILjava/lang/String;Ljava/lang/String;IIIZLcom/lucidfusionlabs/core/NativeCallback;)V"));
  CHECK(jni->pickeritem_construct = e->GetMethodID(jni->pickeritem_class, "<init>", "(Ljava/util/ArrayList;Lcom/lucidfusionlabs/core/NativePickerItemCB;J)V"));
  CHECK(jni->int_intval = e->GetMethodID(jni->int_class, "intValue", "()I"));
  CHECK(jni->long_longval = e->GetMethodID(jni->long_class, "longValue", "()J"));
  if (jni->gplus) CHECK(jni->gplus_class = (jclass)e->NewGlobalRef(e->GetObjectClass(jni->gplus)));

  static const char *argv[2] = { "LFLApp", 0 };
  MyAppCreate(1, argv);
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_main(JNIEnv *e, jclass c) {
  CHECK(jni->env = e);
  INFOf("Main: env=%p", jni->env);
  int ret = MyAppMain();

  NativeAPI_shutdownMainLoop();
  INFOf("Main: env=%p ret=%d", jni->env, ret);
  jni->Free();
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_newMainLoop(JNIEnv *e, jclass c, jobject a, bool reset) {
  CHECK(jni->env = e);
  INFOf("NewMainLoop: env=%p reset=%d", jni->env, reset);
  jni->Init(a, false);
  app->suspended = false;
  SetLFAppMainThread();
  if (reset) app->ResetGL(ResetGLFlag::Reload);
  if (app->focused->focused_cb) app->focused->focused_cb();
  int ret = app->MainLoop();

  NativeAPI_shutdownMainLoop();
  INFOf("NewMainLoop: env=%p ret=%d", jni->env, ret);
  jni->Free();
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_minimize(JNIEnv* env, jclass c) {
  INFOf("%s", "minimize");
  app->RunInMainThread([=](){ app->suspended = true; });
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_reshaped(JNIEnv *e, jclass c, jint x, jint y, jint w, jint h) { 
  static jmethodID mid = CheckNotNull(e->GetMethodID(jni->view_class, "onSynchronizedReshape", "()V"));
  app->RunNowInMainThread([=](){
    jni->env->CallVoidMethod(jni->view, mid);
    app->focused->Reshaped(Box(x, y, w, h));
  });
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_keyPress(JNIEnv *e, jclass c, jint keycode, jint mod, jint down) {
  app->input->QueueKeyPress(keycode, mod, down);
  app->scheduler.Wakeup(app->focused, FrameScheduler::WakeupFlag::ContingentOnEvents);
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_touch(JNIEnv *e, jclass c, jint action, jfloat x, jfloat y, jfloat p) {
  static float lx[2]={0,0}, ly[2]={0,0};
  auto screen = app->focused;
  int dpind = (/*FLAGS_swap_axis*/ 0) ? y < screen->width/2 : x < screen->width/2;
  if (action == AndroidEvent::ACTION_DOWN || action == AndroidEvent::ACTION_POINTER_DOWN) {
    // INFOf("%d down %f, %f", dpind, x, screen->height - y);
    app->input->QueueMouseClick(1, 1, point(screen->x + x, screen->y + screen->height - y));
    app->scheduler.Wakeup(app->focused, FrameScheduler::WakeupFlag::ContingentOnEvents);
    // screen->gesture_tap[dpind] = 1;
    // screen->gesture_dpad_x[dpind] = x;
    // screen->gesture_dpad_y[dpind] = y;
    lx[dpind] = x;
    ly[dpind] = y;
  } else if (action == AndroidEvent::ACTION_UP || action == AndroidEvent::ACTION_POINTER_UP) {
    // INFOf("%d up %f, %f", dpind, x, y);
    app->input->QueueMouseClick(1, 0, point(screen->x + x, screen->y + screen->height - y));
    app->scheduler.Wakeup(app->focused, FrameScheduler::WakeupFlag::ContingentOnEvents);
    // screen->gesture_dpad_stop[dpind] = 1;
    // screen->gesture_dpad_x[dpind] = 0;
    // screen->gesture_dpad_y[dpind] = 0;
  } else if (action == AndroidEvent::ACTION_MOVE) {
    point p(screen->x + x, screen->y + screen->height - y);
    app->input->QueueMouseMovement(p, p - screen->mouse);
    app->scheduler.Wakeup(app->focused, FrameScheduler::WakeupFlag::ContingentOnEvents);
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

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_fling(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat vx, jfloat vy) {
  auto screen = app->focused;
  int dpind = y < screen->width/2;
  // screen->gesture_dpad_dx[dpind] = vx;
  // screen->gesture_dpad_dy[dpind] = vy;
  INFOf("fling(%f, %f) = %d of (%d, %d) and vel = (%f, %f)", x, y, dpind, screen->width, screen->height, vx, vy);
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_scroll(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat vx, jfloat vy) {
  // screen->gesture_swipe_up = screen->gesture_swipe_down = 0;
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_accel(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat z) {
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_scale(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat dx, jfloat dy, jboolean begin) {
  app->input->QueueMouseZoom(v2(x, y), v2(-(dx-1.0)+1.0, -(dy-1.0)+1.0), begin); 
  app->scheduler.Wakeup(app->focused, FrameScheduler::WakeupFlag::ContingentOnEvents);
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_shellRun(JNIEnv *e, jclass c, jstring text) {
  app->focused->shell->Run(e->GetStringUTFChars(text, 0));
}

extern "C" jboolean Java_com_lucidfusionlabs_app_NativeAPI_getFrameEnabled(JNIEnv *e, jclass c) {
  return !app->frame_disabled;
}

extern "C" void Java_com_lucidfusionlabs_core_NativeCallback_RunCallbackInMainThread(JNIEnv *e, jclass c, jlong cb) {
  app->RunCallbackInMainThread(*static_cast<Callback*>(Void(cb)));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeStringCB_RunStringCBInMainThread(JNIEnv *e, jclass c, jlong cb, jstring text) {
  app->RunCallbackInMainThread(bind(*static_cast<StringCB*>(Void(cb)), JNI::GetJString(e, text)));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeIntCB_RunIntCBInMainThread(JNIEnv *e, jclass c, jlong cb, jint x) {
  app->RunCallbackInMainThread(bind(*static_cast<IntCB*>(Void(cb)), x));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeIntIntCB_RunIntIntCBInMainThread(JNIEnv *e, jclass c, jlong cb, jint x, jint y) {
  app->RunCallbackInMainThread(bind(*static_cast<IntIntCB*>(Void(cb)), x, y));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeCallback_FreeCallback(JNIEnv *e, jclass c, jlong cb) {
  delete static_cast<Callback*>(Void(cb));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeStringCB_FreeStringCB(JNIEnv *e, jclass c, jlong cb) {
  delete static_cast<StringCB*>(Void(cb));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeIntCB_FreeIntCB(JNIEnv *e, jclass c, jlong cb) {
  delete static_cast<IntCB*>(Void(cb));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeIntIntCB_FreeIntIntCB(JNIEnv *e, jclass c, jlong cb) {
  delete static_cast<IntIntCB*>(Void(cb));
}

extern "C" void Java_com_lucidfusionlabs_core_NativePickerItemCB_FreePickerItemCB(JNIEnv *e, jclass c, jlong cb) {
  delete static_cast<PickerItem::CB*>(Void(cb));
}

extern "C" void Java_com_lucidfusionlabs_app_TableScreen_RunHideCB(JNIEnv *e, jobject a) {
  static jfieldID self_fid    = CheckNotNull(e->GetFieldID(jni->tablescreen_class, "nativeParent", "J"));
  static jfieldID changed_fid = CheckNotNull(e->GetFieldID(jni->tablescreen_class, "changed",  "Z"));
  uintptr_t self = CheckNotNull(e->GetLongField(a, self_fid));
  TableViewInterface *view = static_cast<TableViewInterface*>(Void(self));
  view->changed = e->GetBooleanField(a, changed_fid);
  if (view->hide_cb) app->RunCallbackInMainThread(view->hide_cb);
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
