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

#include "core/app/loader.h"
#include "core/app/shell.h"
#include <android/log.h>
#include <libgen.h>

namespace LFL {
static JNI *jni = Singleton<JNI>::Set();

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

struct AndroidWindow : public Window {
  AndroidWindow(Application *a) : Window(a) {}

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
  WindowHolder *window;
  FrameScheduler *scheduler;
  Input *input;
  AndroidFrameworkModule(WindowHolder *w, FrameScheduler *s, Input *i) : window(w), scheduler(s), input(i) {}

  int Init() {
    INFO("AndroidFrameworkModule::Init()");
    auto screen = window->focused;
    CHECK(!screen->id);
    screen->id = screen;
    window->windows[screen->id] = screen;

    Socket fd[2];
    CHECK(SystemNetwork::OpenSocketPair(fd));
    scheduler->AddMainWaitSocket(screen, (scheduler->system_event_socket = fd[0]), SocketSet::READABLE);
    scheduler->main_wait_wakeup_socket = fd[1];
    screen->Wakeup();
    return 0;
  }

  int Frame(unsigned clicks) {
    return input->DispatchQueuedInput(frame_on_keyboard_input, frame_on_mouse_input);
  }
};

struct AndroidAssetLoader : public SimpleAssetLoader {
  AndroidAssetLoader(AssetLoading *a) : SimpleAssetLoader(a) {}
  virtual void UnloadAudioFile(AudioAssetLoader::Handle&) {}
  virtual AudioAssetLoader::Handle LoadAudioFile(unique_ptr<File>) { return AudioAssetLoader::Handle(this, nullptr); }
  virtual AudioAssetLoader::Handle LoadAudioFileNamed(const string &filename) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "loadMusicResource", "(Ljava/lang/String;)Landroid/media/MediaPlayer;"));
    string fn = basename(filename.c_str());
    LocalJNIString jfn(jni->env, JNI::ToJString(jni->env, fn.substr(0, fn.find('.'))));
    LocalJNIObject handle(jni->env, jni->env->CallObjectMethod(jni->activity, mid, jfn.v));
    return AudioAssetLoader::Handle(this, jni->env->NewGlobalRef(handle.v));
  }

  virtual void LoadAudio(AudioAssetLoader::Handle &h, SoundAsset *a, int seconds, int flag) { /*a->handle = h;*/ }
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

struct AndroidAlertView : public AlertViewInterface {
  GlobalJNIObject impl;
  AndroidAlertView(AlertItemVec items) : impl(NewAlertScreenObject(move(items))) {}

  static jobject NewAlertScreenObject(AlertItemVec items) {
    CHECK_EQ(4, items.size());
    CHECK_EQ("style", items[0].first);
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->alertscreen_class, "<init>", "(Ljava/util/ArrayList;)V"));
    return jni->env->NewObject(jni->alertscreen_class, mid, JNI::ToModelItemArrayList(jni->env, move(items)));
  }

  void Hide() {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->alertscreen_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, jboolean(false));
  }

  void Show(const string &arg) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->alertscreen_class, "showText", "(Landroid/app/Activity;Ljava/lang/String;)V"));
    LocalJNIString astr(jni->env, JNI::ToJString(jni->env, arg));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, astr.v);
  }

  string RunModal(const string &arg) { return ERRORv(string(), "not implemented"); }
  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->alertscreen_class, "showTextCB", "(Landroid/app/Activity;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Lcom/lucidfusionlabs/core/NativeStringCB;)V"));
    LocalJNIString tstr(jni->env, JNI::ToJString(jni->env, title)), mstr(jni->env, JNI::ToJString(jni->env, msg)), astr(jni->env, JNI::ToJString(jni->env, arg));
    LocalJNIObject cb(jni->env, confirm_cb ? JNI::ToNativeStringCB(jni->env, move(confirm_cb)) : nullptr);
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, tstr.v, mstr.v, astr.v, cb.v);
  }
};

struct AndroidMenuView : public MenuViewInterface {
  GlobalJNIObject impl;
  AndroidMenuView(const string &title, MenuItemVec items) : impl(NewMenuScreenObject(title, move(items))) {}

  static jobject NewMenuScreenObject(const string &title, MenuItemVec items) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->menuscreen_class, "<init>", "(Ljava/lang/String;Ljava/util/ArrayList;)V"));
    LocalJNIString tstr(jni->env, JNI::ToJString(jni->env, title));
    LocalJNIObject l(jni->env, JNI::ToModelItemArrayList(jni->env, move(items)));
    return jni->env->NewObject(jni->menuscreen_class, mid, tstr.v, l.v);
  }

  void Show() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->menuscreen_class, "show", "(Lcom/lucidfusionlabs/app/MainActivity;Z)V"));
    jni->env->CallVoidMethod(impl.v, mid, jni->activity, true);
  }
};

int Application::Suspended() {
  INFO("Application::Suspended");
  return 0;
}

struct AndroidNag : public NagInterface {
  AndroidNag(const string &name, int min_days, int min_uses, int min_events, int remind_days) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "createNag", "(Ljava/lang/String;IIII)V"));
    LocalJNIString nstr(jni->env, JNI::ToJString(jni->env, name));
    jni->env->CallVoidMethod(jni->activity, mid, nstr.v, jint(min_days), jint(min_uses), jint(min_events), jint(remind_days));
  }
};

void ThreadDispatcher::RunCallbackInMainThread(Callback cb) {
  message_queue.WriteCallback(make_unique<Callback>(move(cb)));
  if (!FLAGS_target_fps) wakeup->Wakeup();
}

void Application::CloseWindow(Window *W) {}
void WindowHolder::MakeCurrentWindow(Window *W) {}
void MouseFocus::GrabMouseFocus() {}
void MouseFocus::ReleaseMouseFocus() {}

string Clipboard::GetClipboardText() {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getClipboardText", "()Ljava/lang/String;"));
  LocalJNIString text(jni->env, jstring(jni->env->CallObjectMethod(jni->activity, mid)));
  return JNI::GetJString(jni->env, text.v); 
}

void Clipboard::SetClipboardText(const string &s) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setClipboardText", "(Ljava/lang/String;)V"));
  LocalJNIString v(jni->env, JNI::ToJString(jni->env, s));
  jni->env->CallVoidMethod(jni->activity, mid, v.v);
}

void TouchKeyboard::OpenTouchKeyboard() {
  static jmethodID jni_activity_method_show_keyboard =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "showKeyboard", "()V"));
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_show_keyboard);
}

void TouchKeyboard::CloseTouchKeyboard() {
  static jmethodID jni_activity_method_hide_keyboard =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "hideKeyboard", "()V"));
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_hide_keyboard);
}

void TouchKeyboard::ToggleTouchKeyboard() {
  static jmethodID jni_activity_method_toggle_keyboard =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "toggleKeyboard", "()V"));
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_toggle_keyboard);
}

void TouchKeyboard::CloseTouchKeyboardAfterReturn(bool v) {
#if 0
  static jmethodID jni_activity_method_hide_keyboard_after_enter =
    jni->env->GetMethodID(jni->activity_class, "hideKeyboardAfterEnter", "()V");
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_hide_keyboard_after_enter);
#endif
} 

void WindowHolder::SetAppFrameEnabled(bool v) {
  if ((frame_disabled = !v)) jni->app->DrawSplash(Color::black);
  INFO("Application frame_disabled = ", frame_disabled);
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
void TouchKeyboard::SetTouchKeyboardTiled(bool v) {}
int  Application::SetMultisample(bool v) { return 0; }
int  Application::SetExtraScale(bool v) { return 1; }
void Application::SetDownScale(bool v) {}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &cb) { return ERROR("not implemented"); }
void Application::ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB &cb) { return ERROR("not implemented"); }
void Application::ShowSystemStatusBar(bool v) {
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setSystemStatusBar", "(Z)V"));
  jni->env->CallVoidMethod(jni->activity, mid, jboolean(v));
}

void Application::ShowSystemContextMenu(const MenuItemVec &items) {
  static jmethodID mid = CheckNotNull
    (jni->env->GetMethodID(jni->activity_class, "showContextMenu", "(Ljava/util/ArrayList;)V"));
  LocalJNIObject v(jni->env, JNI::ToModelItemArrayList(jni->env, move(items)));
  jni->env->CallVoidMethod(jni->activity, mid, v.v);
}


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

bool Video::CreateWindow(WindowHolder *H, Window *W) { return true; }
void Video::StartWindow(Window *W) {}
int Video::Swap(Window *W) {
  static jmethodID jni_activity_method_swap = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "viewSwapEGL", "()V"));
  auto gd = W->gd;
  gd->Flush();
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_swap);
  gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

FrameScheduler::FrameScheduler(WindowHolder *w) :
  window(w), maxfps(&FLAGS_target_fps), rate_limit(1), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(1), run_main_loop(1) {}

bool FrameScheduler::DoMainWait(bool only_poll) {
  bool ret = false;
  main_wait_sockets.Select(only_poll ? 0 : -1);
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

void Window::Wakeup(int flag) {
  char c = (flag & WakeupFlag::ContingentOnEvents) ? ' ' : 'W';
  write(parent->scheduler.main_wait_wakeup_socket, &c, 1);
}

void FrameScheduler::UpdateWindowTargetFPS(Window *w) {}
void FrameScheduler::AddMainWaitMouse(Window*)    { dynamic_cast<AndroidFrameworkModule*>(window->framework.get())->frame_on_mouse_input    = true;  }
void FrameScheduler::DelMainWaitMouse(Window*)    { dynamic_cast<AndroidFrameworkModule*>(window->framework.get())->frame_on_mouse_input    = false; }
void FrameScheduler::AddMainWaitKeyboard(Window*) { dynamic_cast<AndroidFrameworkModule*>(window->framework.get())->frame_on_keyboard_input = true;  }
void FrameScheduler::DelMainWaitKeyboard(Window*) { dynamic_cast<AndroidFrameworkModule*>(window->framework.get())->frame_on_keyboard_input = false; }
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

unique_ptr<Window> CreateWindow(Application *a) { return make_unique<AndroidWindow>(a); }
unique_ptr<Module> CreateFrameworkModule(Application *a) { return make_unique<AndroidFrameworkModule>(a, &a->scheduler, a->input.get()); }
unique_ptr<AssetLoaderInterface> CreateAssetLoader(AssetLoading *a) { return make_unique<AndroidAssetLoader>(a); }
unique_ptr<TimerInterface> SystemToolkit::CreateTimer(Callback cb) { return make_unique<AndroidTimer>(move(cb)); };
unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(Window*, AlertItemVec items) { return make_unique<AndroidAlertView>(move(items)); }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(Window*, const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(Window*, const string &title, MenuItemVec items) { return make_unique<AndroidMenuView>(title, move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(Window*, MenuItemVec items) { return nullptr; }
unique_ptr<NagInterface> SystemToolkit::CreateNag(const string &id, int min_days, int min_uses, int min_events, int remind_days) { return make_unique<AndroidNag>(id, min_days, min_uses, min_events, remind_days); }

static void NativeAPI_shutdownMainLoop() {
  auto app = jni->app;
  app->suspended = true;
  if (app->focused->unfocused_cb) app->focused->unfocused_cb();
  while (app->message_queue.HandleMessages()) {}
  // app->ResetGL(LFL::ResetGLFlag::Delete);
  app->focused->gd->Finish();
}

extern "C" jint JNI_OnLoad(JavaVM* vm, void* reserved) { return JNI_VERSION_1_4; }

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_create(JNIEnv *e, jclass c, jobject a) {
  CHECK(jni->env = e);
  CHECK(jni->activity_class = (jclass)e->NewGlobalRef(e->GetObjectClass(a)));
  CHECK(jni->activity_resources = e->GetFieldID(jni->activity_class, "resources", "Landroid/content/res/Resources;"));
  CHECK(jni->activity_handler   = e->GetFieldID(jni->activity_class, "handler",   "Landroid/os/Handler;"));
  // CHECK(jni->activity_gplus     = e->GetFieldID(jni->activity_class, "gplus",     "Lcom/lucidfusionlabs/app/GPlusClient;"));
  static jmethodID activity_getpkgname_mid =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getPackageName", "()Ljava/lang/String;"));

  jni->Init(a, true);
  jni->package_name = jni->GetJString(jni->env, (jstring)jni->env->CallObjectMethod(jni->activity, activity_getpkgname_mid));
  std::replace(jni->package_name.begin(), jni->package_name.end(), '.', '/');

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
  CHECK(jni->runnable_class     = (jclass)e->NewGlobalRef(e->FindClass("java/lang/Runnable")));
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
  jni->app = static_cast<Application*>(MyAppCreate(1, argv));
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_main(JNIEnv *e, jclass c, jint w, jint h, jint v) {
  CHECK(jni->env = e);
  INFOf("Main: env=%p w=%d h=%d opengles_version=%d", jni->env, w, h, v);
  auto app = jni->app;
  app->focused->gl_w = w;
  app->focused->gl_h = h;
  app->focused->gd->version = v;
  int ret = MyAppMain(app);

  NativeAPI_shutdownMainLoop();
  INFOf("Main: env=%p ret=%d", jni->env, ret);
  jni->Free();
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_newMainLoop(JNIEnv *e, jclass c, jobject a, bool reset) {
  CHECK(jni->env = e);
  INFOf("NewMainLoop: env=%p reset=%d", jni->env, reset);
  jni->Init(a, false);
  auto app = jni->app;
  app->suspended = false;
  app->SetMainThread();
  if (reset) app->ResetGL(/*LFL::ResetGLFlag::Delete | */ResetGLFlag::Reload);
  if (app->focused->focused_cb) app->focused->focused_cb();
  int ret = app->MainLoop();

  NativeAPI_shutdownMainLoop();
  INFOf("NewMainLoop: env=%p ret=%d", jni->env, ret);
  jni->Free();
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_minimize(JNIEnv* env, jclass c) {
  INFOf("%s", "minimize");
  auto app = jni->app;
  app->RunInMainThread([=](){ app->suspended = true; });
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_reshaped(JNIEnv *e, jclass c, jint x, jint y, jint w, jint h) { 
  static jmethodID mid = CheckNotNull(e->GetMethodID(jni->activity_class, "viewOnSynchronizedReshape", "()V"));
  auto app = jni->app;
  app->RunNowInMainThread([=](){
    jni->env->CallVoidMethod(jni->activity, mid);
    app->focused->Reshaped(point(w, h), Box(x, y, w, h));
  });
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_keyPress(JNIEnv *e, jclass c, jint keycode, jint mod, jint down) {
  auto app = jni->app;
  app->input->QueueKeyPress(keycode, mod, down);
  app->focused->Wakeup(Window::WakeupFlag::ContingentOnEvents);
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_touch(JNIEnv *e, jclass c, jint action, jfloat x, jfloat y, jfloat p) {
  static float lx[2]={0,0}, ly[2]={0,0};
  auto app = jni->app;
  auto screen = app->focused;
  int dpind = (/*FLAGS_swap_axis*/ 0) ? y < screen->gl_w/2 : x < screen->gl_w/2;
  if (action == AndroidEvent::ACTION_DOWN || action == AndroidEvent::ACTION_POINTER_DOWN) {
    // INFOf("%d down %f, %f", dpind, x, screen->height - y);
    app->input->QueueMouseClick(1, 1, point(screen->gl_x + x, screen->gl_y + screen->gl_h - y));
    app->focused->Wakeup(Window::WakeupFlag::ContingentOnEvents);
    // screen->gesture_tap[dpind] = 1;
    // screen->gesture_dpad_x[dpind] = x;
    // screen->gesture_dpad_y[dpind] = y;
    lx[dpind] = x;
    ly[dpind] = y;
  } else if (action == AndroidEvent::ACTION_UP || action == AndroidEvent::ACTION_POINTER_UP) {
    // INFOf("%d up %f, %f", dpind, x, y);
    app->input->QueueMouseClick(1, 0, point(screen->gl_x + x, screen->gl_y + screen->gl_h - y));
    app->focused->Wakeup(Window::WakeupFlag::ContingentOnEvents);
    // screen->gesture_dpad_stop[dpind] = 1;
    // screen->gesture_dpad_x[dpind] = 0;
    // screen->gesture_dpad_y[dpind] = 0;
  } else if (action == AndroidEvent::ACTION_MOVE) {
    point p(screen->gl_x + x, screen->gl_y + screen->gl_h - y);
    app->input->QueueMouseMovement(p, p - screen->mouse);
    app->focused->Wakeup(Window::WakeupFlag::ContingentOnEvents);
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
  auto screen = jni->app->focused;
  int dpind = y < screen->gl_w/2;
  // screen->gesture_dpad_dx[dpind] = vx;
  // screen->gesture_dpad_dy[dpind] = vy;
  INFOf("fling(%f, %f) = %d of (%d, %d) and vel = (%f, %f)", x, y, dpind, screen->gl_w, screen->gl_h, vx, vy);
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_scroll(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat vx, jfloat vy) {
  // screen->gesture_swipe_up = screen->gesture_swipe_down = 0;
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_accel(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat z) {
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_scale(JNIEnv *e, jclass c, jfloat x, jfloat y, jfloat dx, jfloat dy, jboolean begin) {
  auto app = jni->app;
  auto screen = app->focused;
  app->input->QueueMouseZoom(v2(screen->gl_x + x, screen->gl_y + screen->gl_h - y), v2(-(dx-1.0)+1.0, -(dy-1.0)+1.0), begin);
  app->focused->Wakeup(Window::WakeupFlag::ContingentOnEvents);
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_log(JNIEnv *e, jclass c, jint level, jstring text) {
  jni->app->Log(level, nullptr, 0, jni->GetJString(e, text).c_str());
}

extern "C" void Java_com_lucidfusionlabs_app_NativeAPI_shellRun(JNIEnv *e, jclass c, jstring text) {
  jni->app->focused->shell->Run(jni->GetJString(e, text));
}

extern "C" jboolean Java_com_lucidfusionlabs_app_NativeAPI_getSuspended(JNIEnv *e, jclass c) {
  return jni->app->suspended;
}

extern "C" jboolean Java_com_lucidfusionlabs_app_NativeAPI_getFrameEnabled(JNIEnv *e, jclass c) {
  return !jni->app->frame_disabled;
}

extern "C" void Java_com_lucidfusionlabs_core_NativeCallback_RunCallbackInMainThread(JNIEnv *e, jclass c, jlong cb, jobject done_cb) {
  jni->app->RunCallbackInMainThread(*static_cast<Callback*>(Void(cb)));
  if (done_cb) JNI::MainThreadRunRunnableOnUiThread(make_unique<GlobalJNIObject>(e, done_cb, false));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeStringCB_RunStringCBInMainThread(JNIEnv *e, jclass c, jlong cb, jstring text, jobject done_cb) {
  jni->app->RunCallbackInMainThread(bind(*static_cast<StringCB*>(Void(cb)), JNI::GetJString(e, text)));
  if (done_cb) JNI::MainThreadRunRunnableOnUiThread(make_unique<GlobalJNIObject>(e, done_cb, false));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeIntCB_RunIntCBInMainThread(JNIEnv *e, jclass c, jlong cb, jint x, jobject done_cb) {
  jni->app->RunCallbackInMainThread(bind(*static_cast<IntCB*>(Void(cb)), x));
  if (done_cb) JNI::MainThreadRunRunnableOnUiThread(make_unique<GlobalJNIObject>(e, done_cb, false));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeIntIntCB_RunIntIntCBInMainThread(JNIEnv *e, jclass c, jlong cb, jint x, jint y, jobject done_cb) {
  jni->app->RunCallbackInMainThread(bind(*static_cast<IntIntCB*>(Void(cb)), x, y));
  if (done_cb) JNI::MainThreadRunRunnableOnUiThread(make_unique<GlobalJNIObject>(e, done_cb, false));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeCallback_FreeCallback(JNIEnv *e, jclass c, jlong cb) {
  unique_ptr<Callback> v(static_cast<Callback*>(Void(cb)));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeStringCB_FreeStringCB(JNIEnv *e, jclass c, jlong cb) {
  unique_ptr<StringCB> v(static_cast<StringCB*>(Void(cb)));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeIntCB_FreeIntCB(JNIEnv *e, jclass c, jlong cb) {
  unique_ptr<IntCB> v(static_cast<IntCB*>(Void(cb)));
}

extern "C" void Java_com_lucidfusionlabs_core_NativeIntIntCB_FreeIntIntCB(JNIEnv *e, jclass c, jlong cb) {
  unique_ptr<IntIntCB> v(static_cast<IntIntCB*>(Void(cb)));
}

extern "C" void Java_com_lucidfusionlabs_core_NativePickerItemCB_FreePickerItemCB(JNIEnv *e, jclass c, jlong cb) {
  unique_ptr<PickerItem::CB> v(static_cast<PickerItem::CB*>(Void(cb)));
}

extern "C" jobject Java_com_lucidfusionlabs_core_PickerItem_getFontPickerItem(JNIEnv *e, jclass c) {
  static unique_ptr<GlobalJNIObject> picker_item;
  if (!picker_item) {
    auto font_picker = new PickerItem();
    font_picker->picked.resize(2);
    font_picker->data.resize(2);
    font_picker->data[0].emplace_back("default");
    for (int i=0; i<64; i++) font_picker->data[1].emplace_back(StrCat(i+1));
    picker_item = make_unique<GlobalJNIObject>(e, JNI::ToPickerItem(e, font_picker));
  }
  return picker_item->v;
}

extern "C" void Java_com_lucidfusionlabs_app_TableScreen_RunHideCB(JNIEnv *e, jobject a) {
  static jfieldID self_fid    = CheckNotNull(e->GetFieldID(jni->tablescreen_class, "nativeParent", "J"));
  static jfieldID changed_fid = CheckNotNull(e->GetFieldID(jni->tablescreen_class, "changed",  "Z"));
  uintptr_t self = CheckNotNull(e->GetLongField(a, self_fid));
  TableViewInterface *view = static_cast<TableViewInterface*>(Void(self));
  if (e->GetBooleanField(a, changed_fid)) view->changed = true;
  if (view->hide_cb) jni->app->RunCallbackInMainThread(view->hide_cb);
}

extern "C" void Java_com_lucidfusionlabs_app_GPlusClient_startGame(JNIEnv *e, jobject a, jboolean server, jstring pid) {
  jni->app->focused->shell->Run(StrCat((server ? "gplus_server " : "gplus_client "), jni->GetJString(e, pid)));
}

extern "C" void Java_com_lucidfusionlabs_app_GPlusClient_read(JNIEnv *e, jobject a, jstring pid, jobject bb, jint len) {
  static GPlus *gplus = Singleton<GPlus>::Set();
  if (gplus->server) gplus->server->EndpointRead(jni->GetJString(e, pid), (const char*)e->GetDirectBufferAddress(bb), len);
}

}; // namespace LFL
