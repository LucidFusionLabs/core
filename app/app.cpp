/*
 * $Id: lfapp.cpp 1335 2014-12-02 04:13:46Z justin $
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

#include "core/app/app.h"
#include "core/app/flow.h"
#include "core/app/gui.h"
#include "core/app/ipc.h"
#include "core/app/net/resolver.h"

#include <time.h>
#include <fcntl.h>

#ifdef WIN32
#define CALLBACK __stdcall
#include <Windns.h>
#define stat(x,y) _stat(x,y)
#define gmtime_r(i,o) memcpy(o, gmtime(&in), sizeof(tm))
#define localtime_r(i,o) memcpy(o, localtime(&in), sizeof(tm))
#else
#include <signal.h>
#include <pthread.h>
#include <dlfcn.h>
#include <sys/resource.h>
#endif

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#ifndef LFL_IPHONE
#include <ApplicationServices/ApplicationServices.h>
#endif
#endif

#ifdef LFL_X11INPUT
#include "X11/Xlib.h"
#undef KeyPress
#endif

#ifdef LFL_ANDROID
#include <android/log.h>
extern "C" void AndroidSetFrameOnKeyboardInput(int v);
extern "C" void AndroidSetFrameOnMouseInput   (int v);
#endif

#ifdef LFL_WXWIDGETS
#include <wx/wx.h>
#include <wx/glcanvas.h>
#endif

#if defined(LFL_GLFWVIDEO) || defined(LFL_GLFWINPUT)
#include "GLFW/glfw3.h"
#endif

#if defined(LFL_SDLAUDIO) || defined(LFL_SDLVIDEO) || defined(LFL_SDLINPUT)
#include "SDL.h"
#endif

#if defined(LFL_IPHONE)
extern "C" char *iPhoneDocumentPathCopy();
extern "C" void iPhoneLog(const char *text);
extern "C" void iPhoneOpenBrowser(const char *url_text);
extern "C" void iPhoneLaunchNativeMenu(const char*);
extern "C" void iPhoneCreateNativeMenu(const char*, int, const char**, const char**);
extern "C" void iPhoneTriggerFrame(void*);
extern "C" bool iPhoneTriggerFrameIn(void*, int ms, bool force);
extern "C" void iPhoneClearTriggerFrameIn(void *O);
extern "C" void iPhoneUpdateTargetFPS(void*);
extern "C" void iPhoneAddWaitForeverMouse(void*);
extern "C" void iPhoneDelWaitForeverMouse(void*);
extern "C" void iPhoneAddWaitForeverKeyboard(void*);
extern "C" void iPhoneDelWaitForeverKeyboard(void*);
extern "C" void iPhoneAddWaitForeverSocket(void*, int fd);
extern "C" void iPhoneDelWaitForeverSocket(void*, int fd);
extern "C" int  iPhonePasswordCopy(const char *, const char*, const char*,       char*, int);
extern "C" bool iPhonePasswordSave(const char *, const char*, const char*, const char*, int);
extern "C" void iPhonePlayMusic(void *handle);
extern "C" void iPhonePlayBackgroundMusic(void *handle);
#elif defined(__APPLE__)
extern "C" void OSXStartWindow(typed_ptr);
extern "C" void OSXCreateNativeEditMenu();
extern "C" void OSXCreateNativeMenu(const char*, int, const char**, const char**, const char**);
extern "C" void OSXLaunchNativeContextMenu(typed_ptr, int, int, int, const char**, const char**, const char**);
extern "C" void OSXLaunchNativeFontChooser(const char *, int, const char *);
extern "C" void OSXLaunchNativeFileChooser(bool, bool, bool, const char *);
extern "C" void OSXTriggerFrame(typed_ptr);
extern "C" bool OSXTriggerFrameIn(typed_ptr, int ms, bool force);
extern "C" void OSXClearTriggerFrameIn(typed_ptr);
extern "C" void OSXUpdateTargetFPS(typed_ptr);
extern "C" void OSXAddWaitForeverMouse(typed_ptr);
extern "C" void OSXDelWaitForeverMouse(typed_ptr);
extern "C" void OSXAddWaitForeverKeyboard(typed_ptr);
extern "C" void OSXDelWaitForeverKeyboard(typed_ptr);
extern "C" void OSXAddWaitForeverSocket(typed_ptr, int fd);
extern "C" void OSXDelWaitForeverSocket(typed_ptr, int fd);
#endif

extern "C" void BreakHook() {}
extern "C" void ShellRun(const char *text) { return LFL::screen->shell->Run(text); }
extern "C" NativeWindow *GetNativeWindow() { return LFL::screen; }
extern "C" LFApp        *GetLFApp()        { return LFL::app; }
extern "C" int LFAppMain()                 { return LFL::app->Main(); }
extern "C" int LFAppMainLoop()             { return LFL::app->MainLoop(); }
extern "C" int LFAppFrame(bool handle_ev)  { return LFL::app->EventDrivenFrame(handle_ev); }
extern "C" void LFAppWakeup(void *v)       { return LFL::app->scheduler.Wakeup(v); }
extern "C" void LFAppResetGL()             { return LFL::app->ResetGL(); }
extern "C" const char *LFAppDownloadDir()  { return LFL::app->dldir.c_str(); }
extern "C" void LFAppAtExit()              { delete LFL::app; }
extern "C" void LFAppShutdown()                   { LFL::app->run=0; LFAppWakeup(0); }
extern "C" void WindowReshaped(int w, int h)      { LFL::screen->Reshaped(w, h); }
extern "C" void WindowMinimized()                 { LFL::screen->Minimized(); }
extern "C" void WindowUnMinimized()               { LFL::screen->UnMinimized(); }
extern "C" void WindowClosed()                    { LFL::app->CloseWindow(LFL::screen); }
extern "C" void QueueWindowReshaped(int w, int h) { LFL::app->RunInMainThread(LFL::bind(&LFL::Window::Reshaped,    LFL::screen, w, h)); }
extern "C" void QueueWindowMinimized()            { LFL::app->RunInMainThread(LFL::bind(&LFL::Window::Minimized,   LFL::screen)); }
extern "C" void QueueWindowUnMinimized()          { LFL::app->RunInMainThread(LFL::bind(&LFL::Window::UnMinimized, LFL::screen)); }
extern "C" void QueueWindowClosed()               { LFL::app->RunInMainThread(LFL::bind([=](){ LFL::app->CloseWindow(LFL::screen); })); }
extern "C" int  KeyPress  (int b, int d)                    { return LFL::app->input->KeyPress  (b, d); }
extern "C" int  MouseClick(int b, int d, int x,  int y)     { return LFL::app->input->MouseClick(b, d, LFL::point(x, y)); }
extern "C" int  MouseMove (int x, int y, int dx, int dy)    { return LFL::app->input->MouseMove (LFL::point(x, y), LFL::point(dx, dy)); }
extern "C" void QueueKeyPress  (int b, int d)               { return LFL::app->input->QueueKeyPress  (b, d); }
extern "C" void QueueMouseClick(int b, int d, int x, int y) { return LFL::app->input->QueueMouseClick(b, d, LFL::point(x, y)); }
extern "C" void EndpointRead(void *svc, const char *name, const char *buf, int len) { LFL::app->net->EndpointRead(static_cast<LFL::Service*>(svc), name, buf, len); }

extern "C" NativeWindow *SetNativeWindowByID(void *id) { return SetNativeWindow(LFL::FindOrNull(LFL::app->windows, id)); }
extern "C" NativeWindow *SetNativeWindow(NativeWindow *W) {
  CHECK(W);
  if (W == LFL::screen) return W;
  LFL::app->MakeCurrentWindow((LFL::screen = static_cast<LFL::Window*>(W)));
  return W;
}

extern "C" void SetLFAppMainThread() {
  LFL::Thread::id_t id = LFL::Thread::GetId();
  if (LFL::app->main_thread_id != id) INFOf("LFApp->main_thread_id changed from %llx to %llx", LFL::app->main_thread_id, id);
  LFL::app->main_thread_id = id; 
}

extern "C" void LFAppFatal() {
  ERROR("LFAppFatal");
  if (bool suicide=true) *reinterpret_cast<volatile int*>(0) = 0;
  LFL::app->run = 0;
  exit(-1);
}

#ifndef WIN32
extern "C" void HandleSigInt(int sig) { INFO("interrupt"); LFAppShutdown(); }
#else
extern "C" BOOL WINAPI HandlerCtrlC(DWORD sig) { INFO("interrupt"); LFAppShutdown(); return TRUE; }
extern "C" void OpenSystemConsole(const char *title) {
  FLAGS_open_console=1;
  AllocConsole();
  SetConsoleTitle(title);
  freopen("CONOUT$", "wb", stdout);
  freopen("CONIN$", "rb", stdin);
  SetConsoleCtrlHandler(HandleCtrlC, 1);
}
extern "C" void CloseSystemConsole() {
  fclose(stdin);
  fclose(stdout);
  FreeConsole();
}
#endif

namespace LFL {
#ifdef LFL_DEBUG
DEFINE_int(loglevel, 7, "Log level: [Fatal=-1, Error=0, Info=3, Debug=7]");
#else
DEFINE_int(loglevel, 0, "Log level: [Fatal=-1, Error=0, Info=3, Debug=7]");
#endif
DEFINE_bool(lfapp_audio, false, "Enable audio in/out");
DEFINE_bool(lfapp_video, false, "Enable OpenGL");
DEFINE_bool(lfapp_input, false, "Enable keyboard/mouse input");
DEFINE_bool(lfapp_network, false, "Enable asynchronous network engine");
DEFINE_bool(lfapp_camera, false, "Enable camera capture");
DEFINE_bool(lfapp_cuda, false, "Enable CUDA acceleration");
DEFINE_bool(daemonize, false, "Daemonize server");
DEFINE_bool(max_rlimit_core, true, "Max core dump rlimit");
DEFINE_bool(max_rlimit_open_files, false, "Max number of open files rlimit");
DEFINE_int(threadpool_size, 0, "Threadpool size");
DEFINE_int(target_fps, 0, "Max frames per second");
#ifdef LFL_MOBILE
DEFINE_int(peak_fps, 30, "Peak FPS");
#else
DEFINE_int(peak_fps, 60, "Peak FPS");
#endif
DEFINE_bool(open_console, 0, "Open console on win32");
DEFINE_bool(cursor_grabbed, false, "Center cursor every frame");
DEFINE_bool(frame_debug, false, "Print each frame");
DEFINE_bool(rcon_debug, false, "Print game protocol commands");

Application *app = new Application();
Window *screen = new Window();
void Log(int level, const char *file, int line, const string &m) { app->Log(level, file, line, m); }

#ifdef LFL_IPHONE
static pthread_key_t tls_key;
void ThreadLocalStorage::Init() { pthread_key_create(&tls_key, 0); ThreadInit(); }
void ThreadLocalStorage::Free() { ThreadFree(); pthread_key_delete(tls_key); }
void ThreadLocalStorage::ThreadInit() { pthread_setspecific(tls_key, new ThreadLocalStorage()); }
void ThreadLocalStorage::ThreadFree() { delete ThreadLocalStorage::Get(); }
ThreadLocalStorage *ThreadLocalStorage::Get() { return (ThreadLocalStorage*)pthread_getspecific(tls_key); }
#else
thread_local ThreadLocalStorage *tls_instance = 0;
void ThreadLocalStorage::Init() {}
void ThreadLocalStorage::Free() {}
void ThreadLocalStorage::ThreadInit() {}
void ThreadLocalStorage::ThreadFree() { delete tls_instance; tls_instance = nullptr; }
ThreadLocalStorage *ThreadLocalStorage::Get() { return tls_instance ? tls_instance : (tls_instance = new ThreadLocalStorage()); }
#endif
Allocator *ThreadLocalStorage::GetAllocator(bool reset_allocator) {
  ThreadLocalStorage *tls = Get();
  if (!tls->alloc) tls->alloc = make_unique<FixedAllocator<1024*1024>>();
  if (reset_allocator) tls->alloc->Reset();
  return tls->alloc.get();
}

string Flag::GetString() const { string v=Get(); return StrCat(name, v.size()?" = ":"", v.size()?v:"", " : ", desc); } 
string FlagMap::Get   (const string &k) const { Flag *f = FindOrNull(flagmap, k); return f ? f->Get()    : "";    }
bool   FlagMap::IsBool(const string &k) const { Flag *f = FindOrNull(flagmap, k); return f ? f->IsBool() : false; }

bool FlagMap::Set(const string &k, const string &v) {
  Flag *f = FindOrNull(flagmap, k);
  if (!f) return false;
  f->override = true;
  INFO("set flag ", k, " = ", v);
  if (f->Get() != v) dirty = true;
  else return true;
  f->Update(v.c_str());
  return true;
}

string FlagMap::Match(const string &key, const char *source_filename) const {
  vector<int> keyv(key.size()), dbiv;
  Vec<int>::Assign(&keyv[0], key.data(), key.size());

  vector<string> db;
  for (auto &i : flagmap) {
    if (source_filename && strcmp(source_filename, i.second->file)) continue;
    db.push_back(i.first);
  }

  string mindistval;
  double dist, mindist = INFINITY;
  for (auto &t : db) {
    dbiv.resize(t.size());
    Vec<int>::Assign(&dbiv[0], t.data(), t.size());
    if ((dist = Levenshtein(keyv, dbiv)) < mindist) { mindist = dist; mindistval = t; }
  }
  return mindistval;
}

int FlagMap::getopt(int argc, const char **argv, const char *source_filename) {
  for (optind=1; optind<argc; /**/) {
    const char *arg = argv[optind], *key = arg + 1, *val = "";
    if (*arg != '-' || *(arg+1) == 0) break;

    if (++optind < argc && !(IsBool(key) && *(argv[optind]) == '-')) val = argv[optind++];
    if (!strcmp(key, "fullhelp")) { Print(); return -1; }

    if (!Set(key, val)) {
#ifdef __APPLE__
      if (PrefixMatch(key, "psn_")) continue;
#endif
      INFO("unknown flag: ", key);
      string nearest1 = Match(key), nearest2 = Match(key, source_filename);
      INFO("Did you mean -", nearest2.size() ? StrCat(nearest2, " or -") : "", nearest1, " ?");
      INFO("usage: ", argv[0], " -k v");
      Print(source_filename);
      return -1;
    }
  }
  return optind;
}

void FlagMap::Print(const char *source_filename) const {
  for (AllFlags::const_iterator i = flagmap.begin(); i != flagmap.end(); i++) {
    if (source_filename && strcmp(source_filename, i->second->file)) continue;
    INFO(i->second->GetString());
  }
  if (source_filename) INFO("fullhelp : Display full help"); 
}

/* Application */

Application::Application() :
  tex_mode(2, 1, 0), grab_mode(2, 0, 1),
  fill_mode(3, GraphicsDevice::Fill, GraphicsDevice::Line, GraphicsDevice::Point) {
  run=1; initialized=0; main_thread_id=0; frames_ran=0; memzero(log_time); 
  fonts = make_unique<Fonts>();
}

void Application::Log(int level, const char *file, int line, const string &message) {
  {
    ScopedMutex sm(log_mutex);
    char tbuf[64];
    tm last_log_time = log_time;
    logtime(tbuf, sizeof(tbuf), &log_time);
    if (DayChanged(log_time, last_log_time)) 
      WriteLogLine(tbuf, StrCat("Date changed to ", logfileday(log_time)).c_str(), __FILE__, __LINE__);
    WriteLogLine(tbuf, message.c_str(), file, line);
  }
  if (level == LFApp::Log::Fatal) LFAppFatal();
  if (run && FLAGS_lfapp_video && screen && screen->console) screen->console->Write(message);
}

void Application::WriteLogLine(const char *tbuf, const char *message, const char *file, int line) {
  fprintf(stdout, "%s %s (%s:%d)\r\n", tbuf, message, file, line);
  fflush(stdout);
  if (logfile) {
    fprintf(logfile, "%s %s (%s:%d)\r\n", tbuf, message, file, line);
    fflush(logfile);
  }
#ifdef LFL_IPHONE
  iPhoneLog(StringPrintf("%s (%s:%d)", message, file, line).c_str());
#endif
#ifdef LFL_ANDROID
  __android_log_print(ANDROID_LOG_INFO, screen->caption.c_str(), "%s (%s:%d)", message, file, line);
#endif
}

void Application::CreateNewWindow() {
  Window *new_window = new Window();
  if (window_init_cb) window_init_cb(new_window);
  new_window->gd = static_cast<GraphicsDevice*>(LFAppCreateGraphicsDevice(video->opengles_version));
  CHECK(CreateWindow(new_window));
#if 0 // ndef LFL_QT
  MakeCurrentWindow(new_window);
  StartNewWindow(new_window);
  MakeCurrentWindow(orig_window);
#endif
}

void Application::StartNewWindow(Window *new_window) {
  video->InitGraphicsDevice(new_window);
  input->Init(new_window);
  new_window->default_font.Load();
  if (window_start_cb) window_start_cb(new_window);
  video->StartWindow();
}

NetworkThread *Application::CreateNetworkThread(bool detach, bool start) {
  CHECK(net);
  if (detach) VectorEraseByValue(&modules, static_cast<Module*>(net.get()));
  network_thread = make_unique<NetworkThread>(net.get(), !detach);
  if (start) network_thread->thread->Start();
  return network_thread.get();
}

void Application::AddNativeMenu(const string &title, const vector<MenuItem>&items) {
#if defined(LFL_IPHONE)
  vector<const char *> n, v;
  for (auto &i : items) { n.push_back(tuple_get<1>(i).c_str()); v.push_back(tuple_get<2>(i).c_str()); }
  iPhoneCreateNativeMenu(title.c_str(), items.size(), &n[0], &v[0]);
#elif defined(LFL_OSXVIDEO)
  vector<const char *> k, n, v;
  for (auto &i : items) { k.push_back(tuple_get<0>(i).c_str()); n.push_back(tuple_get<1>(i).c_str()); v.push_back(tuple_get<2>(i).c_str()); }
  OSXCreateNativeMenu(title.c_str(), items.size(), &k[0], &n[0], &v[0]);
#elif defined(LFL_WINVIDEO)
  WinWindow *win = static_cast<WinWindow*>(screen->impl);
  if (!win->menu) { win->menu = CreateMenu(); win->context_menu = CreatePopupMenu(); }
  HMENU hAddMenu = CreatePopupMenu();
  for (auto &i : items) {
    if (tuple_get<1>(i) == "<seperator>") AppendMenu(hAddMenu, MF_MENUBARBREAK, 0, NULL);
    else AppendMenu(hAddMenu, MF_STRING, win->start_msg_id + win->menu_cmds.size(), tuple_get<1>(i).c_str());
    win->menu_cmds.push_back(tuple_get<2>(i));
  }
  AppendMenu(win->menu,         MF_STRING | MF_POPUP, (UINT)hAddMenu, title.c_str());
  AppendMenu(win->context_menu, MF_STRING | MF_POPUP, (UINT)hAddMenu, title.c_str());
  if (win->menubar) SetMenu((HWND)screen->id, win->menu);
#endif
}

void Application::AddNativeEditMenu() {
#if defined(LFL_OSXVIDEO)
  OSXCreateNativeEditMenu();
#endif
}

void Application::LaunchNativeMenu(const string &title) {
#if defined(LFL_IPHONE)
  iPhoneLaunchNativeMenu(title.c_str());
#endif
}

void Application::LaunchNativeContextMenu(const vector<MenuItem>&items) {
#if defined(LFL_OSXVIDEO)
  vector<const char *> k, n, v;
  for (auto &i : items) { k.push_back(tuple_get<0>(i).c_str()); n.push_back(tuple_get<1>(i).c_str()); v.push_back(tuple_get<2>(i).c_str()); }
  OSXLaunchNativeContextMenu(screen->id, screen->mouse.x, screen->mouse.y, items.size(), &k[0], &n[0], &v[0]);
#endif
}

void Application::LaunchNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd) {
#if defined(LFL_OSXVIDEO)
  OSXLaunchNativeFontChooser(cur_font.name.c_str(), cur_font.size, choose_cmd.c_str());
#elif defined(LFL_WINVIDEO)
  LOGFONT lf;
  memzero(lf);
  HDC hdc = GetDC(NULL);
  lf.lfHeight = -MulDiv(cur_font.size, GetDeviceCaps(hdc, LOGPIXELSY), 72);
  lf.lfWeight = (cur_font.flag & FontDesc::Bold) ? FW_BOLD : FW_NORMAL;
  lf.lfItalic = cur_font.flag & FontDesc::Italic;
  strncpy(lf.lfFaceName, cur_font.name.c_str(), sizeof(lf.lfFaceName)-1);
  ReleaseDC(NULL, hdc);
  CHOOSEFONT cf;
  memzero(cf);
  cf.lpLogFont = &lf;
  cf.lStructSize = sizeof(cf);
  cf.hwndOwner = (HWND)screen->id;
  cf.Flags = CF_SCREENFONTS | CF_INITTOLOGFONTSTRUCT;
  if (!ChooseFont(&cf)) return;
  int flag = FontDesc::Mono | (lf.lfWeight > FW_NORMAL ? FontDesc::Bold : 0) | (lf.lfItalic ? FontDesc::Italic : 0);
  screen->shell.Run(StrCat(choose_cmd, " ", lf.lfFaceName, " ", cf.iPointSize/10, " ", flag));
#endif
}

void Application::LaunchNativeFileChooser(bool files, bool dirs, bool multi, const string &choose_cmd) {
#if defined(LFL_OSXVIDEO)
  OSXLaunchNativeFileChooser(files, dirs, multi, choose_cmd.c_str());
#endif
}

void Application::OpenSystemBrowser(const string &url_text) {
#if defined(LFL_ANDROID)
  AndroidOpenBrowser(url_text.c_str());
#elif defined(LFL_IPHONE)
  iPhoneOpenBrowser(url_text.c_str());
#elif defined(__APPLE__)
  CFURLRef url = CFURLCreateWithBytes(0, MakeUnsigned(url_text.c_str()), url_text.size(), kCFStringEncodingASCII, 0);
  if (url) { LSOpenCFURLRef(url, 0); CFRelease(url); }
#elif defined(LFL_WINVIDEO)
  ShellExecute(NULL, "open", url_text.c_str(), NULL, NULL, SW_SHOWNORMAL);
#endif
}

void Application::SavePassword(const string &h, const string &u, const string &pw) {
#if defined(LFL_IPHONE)
  iPhonePasswordSave(name.c_str(), h.c_str(), u.c_str(), pw.c_str(), pw.size());
#endif
}

bool Application::LoadPassword(const string &h, const string &u, string *pw) {
#if defined(LFL_IPHONE)
  pw->resize(1024);
  pw->resize(iPhonePasswordCopy(name.c_str(), h.c_str(), u.c_str(), &(*pw)[0], pw->size()));
  return pw->size();
#endif
  return 0;
}

void Application::ShowAds() {
#if defined(LFL_ANDROID)
  AndroidShowAds();
#endif
}

void Application::HideAds() {
#if defined(LFL_ANDROID)
  AndroidHideAds();
#endif
}

int Application::GetVolume() { 
#if defined(LFL_ANDROID)
  return AndroidGetVolume();
#else
  return 0;
#endif
}

int Application::GetMaxVolume() { 
#if defined(LFL_ANDROID)
  return AndroidGetMaxVolume();
#else
  return 10;
#endif
}

void Application::SetVolume(int v) { 
#if defined(LFL_ANDROID)
  AndroidSetVolume(v);
#endif
}

void Application::PlaySoundEffect(SoundAsset *sa) {
#if defined(LFL_ANDROID)
  AndroidPlayMusic(sa->handle);
#elif defined(LFL_IPHONE)
  iPhonePlayMusic(sa->handle);
#else
  audio->QueueMix(sa, MixFlag::Reset | MixFlag::Mix | (audio->loop ? MixFlag::DontQueue : 0), -1, -1);
#endif
}

void Application::PlayBackgroundMusic(SoundAsset *music) {
#if defined(LFL_ANDROID)
  AndroidPlayBackgroundMusic(music->handle);
#elif defined(LFL_IPHONE)
  iPhonePlayBackgroundMusic(music->handle);
#else
  audio->QueueMix(music);
  audio->loop = music;
#endif
}

void *Application::GetSymbol(const string &n) {
#ifdef WIN32
  return GetProcAddress(GetModuleHandle(NULL), n.c_str());
#else
  return dlsym(RTLD_DEFAULT, n.c_str());
#endif
}

StringPiece Application::LoadResource(int id) {
#ifdef WIN32
  HRSRC resource = FindResource(NULL, MAKEINTRESOURCE(id), MAKEINTRESOURCE(900));
  HGLOBAL resource_data = ::LoadResource(NULL, resource);
  return StringPiece((char*)LockResource(resource_data), SizeofResource(NULL, resource));
#else
  return StringPiece();
#endif
}

void Application::Daemonize(const char *dir) {
#ifndef WIN32
  char fn1[256], fn2[256];
  snprintf(fn1, sizeof(fn1), "%s%s.stdout", dir, app->progname.c_str());
  snprintf(fn2, sizeof(fn2), "%s%s.stderr", dir, app->progname.c_str());
  FILE *fout = fopen(fn1, "a"); fprintf(stderr, "open %s %s\n", fn1, fout ? "OK" : strerror(errno));
  FILE *ferr = fopen(fn2, "a"); fprintf(stderr, "open %s %s\n", fn2, ferr ? "OK" : strerror(errno));
  Daemonize(fout, ferr);
#endif
}

void Application::Daemonize(FILE *fout, FILE *ferr) {
#ifndef WIN32
  int pid = fork();
  if (pid < 0) { fprintf(stderr, "fork: %d\n", pid); exit(-1); }
  if (pid > 0) { fprintf(stderr, "daemonized pid: %d\n", pid); exit(0); }

  int sid = setsid();
  if (sid < 0) { fprintf(stderr, "setsid: %d\n", sid); exit(-1); }

  close(STDIN_FILENO); 
  close(STDOUT_FILENO);
  close(STDERR_FILENO);

  if (fout) dup2(fileno(fout), 1);
  if (ferr) dup2(fileno(ferr), 2);
#endif
}

int Application::Create(int argc, const char **argv, const char *source_filename, void (*create_cb)()) {
  if (!done_create_cb && (done_create_cb=1) && create_cb) create_cb();
#ifdef LFL_GLOG
  google::InstallFailureSignalHandler();
#endif
  SetLFAppMainThread();
  time_started = Now();
  progname = argv[0];
  startdir = LocalFile::CurrentDirectory();

#ifdef WIN32
  bindir = progname.substr(0, DirNameLen(progname, true));
#else
  bindir = LocalFile::JoinPath(startdir, progname.substr(0, DirNameLen(progname, true)));
#endif

#if defined(LFL_ANDROID)
#elif defined(__APPLE__)
  char rpath[1024];
  CFBundleRef mainBundle = CFBundleGetMainBundle();
  CFURLRef respath = CFBundleCopyResourcesDirectoryURL(mainBundle);
  CFURLGetFileSystemRepresentation(respath, true, MakeUnsigned(rpath), sizeof(rpath));
  CFRelease(respath);
  if (PrefixMatch(rpath, startdir+"/")) assetdir = StrCat(rpath + startdir.size()+1, "/assets/");
  else assetdir = StrCat(rpath, "/assets/"); 
#else
  assetdir = StrCat(bindir, "/assets/"); 
#endif

#ifdef WIN32
  { /* winsock startup */
    WSADATA wsadata;
    WSAStartup(MAKEWORD(2,2), &wsadata);
  }
#else
  pid = getpid();

  /* handle SIGINT */
  signal(SIGINT, HandleSigInt);

  { /* ignore SIGPIPE */
    struct sigaction sa;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sa.sa_handler = SIG_IGN;

    if (sigaction(SIGPIPE, &sa, NULL) == -1) return -1;
  }
#endif

  srand(fnv32(&pid, sizeof(int), time(0)));
  if (logfilename.size()) {
    logfile = fopen(logfilename.c_str(), "a");
    if (logfile) SystemNetwork::SetSocketCloseOnExec(fileno(logfile), 1);
  }

  ThreadLocalStorage::Init();
  atexit(LFAppAtExit);

#ifdef WIN32
  string console_title = StrCat(screen->caption, " console");
  if (argc > 1) OpenSystemConsole(console_title.c_str());
#endif

  if (Singleton<FlagMap>::Get()->getopt(argc, argv, source_filename) < 0) return -1;

#ifdef WIN32
  if (argc > 1) {
    if (!FLAGS_open_console) CloseSystemConsole();
  }
  else if (FLAGS_open_console) OpenSystemConsole();
  if (argc > 1) OpenSystemConsole(console_title.c_str());
#endif

  {
#if defined(LFL_IPHONE)
    char *path = iPhoneDocumentPathCopy();
    dldir = StrCat(path, "/");
    free(path);
#elif defined(WIN32)
    char path[MAX_PATH];
    if (!SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL|CSIDL_FLAG_CREATE, NULL, 0, path))) return -1;
    dldir = StrCat(path, "/");
#endif
  }

  INFO("startdir = ", startdir);
  INFO("assetdir = ", assetdir);
  INFO("dldir = ", dldir);

#ifndef WIN32
  if (FLAGS_max_rlimit_core) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_CORE, &rl) == -1) return ERRORv(-1, "core getrlimit ", strerror(errno));

    rl.rlim_cur = rl.rlim_max;
    if (setrlimit(RLIMIT_CORE, &rl) == -1) return ERRORv(-1, "core setrlimit ", strerror(errno));
  }

#ifndef LFL_MOBILE
  if (FLAGS_max_rlimit_open_files) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == -1) return ERRORv(-1, "files getrlimit ", strerror(errno));
#ifdef __APPLE__
    rl.rlim_cur = rl.rlim_max = OPEN_MAX;
#else
    rl.rlim_cur = rl.rlim_max; // 999999
#endif
    INFO("setrlimit(RLIMIT_NOFILE, ", rl.rlim_cur, ")");
    if (setrlimit(RLIMIT_NOFILE, &rl) == -1) return ERRORv(-1, "files setrlimit ", strerror(errno));
  }
#endif // LFL_MOBILE
#endif // WIN32

  if (FLAGS_daemonize) {
    Daemonize();
    SetLFAppMainThread();
  }

  return 0;
}

int Application::Init() {
  if (FLAGS_lfapp_video) {
    shaders = make_unique<Shaders>();
    if ((video = make_unique<Video>())->Init()) return ERRORv(-1, "video init failed");
  }
  else { windows[screen->id.value] = screen; }

  thread_pool.Open(X_or_1(FLAGS_threadpool_size));
  if (FLAGS_threadpool_size) thread_pool.Start();

  if (FLAGS_lfapp_audio) {
    if (LoadModule((audio = make_unique<Audio>()).get())) return ERRORv(-1, "audio init failed");
  }
  else { FLAGS_chans_in=FLAGS_chans_out=1; }

  if (FLAGS_lfapp_audio || FLAGS_lfapp_video) {
    if ((asset_loader = make_unique<AssetLoader>())->Init()) return ERRORv(-1, "asset loader init failed");
  }

  video->InitFonts();

  if (FLAGS_lfapp_input) {
    if (LoadModule((input = make_unique<Input>()).get())) return ERRORv(-1, "input init failed");
  }

  if (FLAGS_lfapp_network) {
    if (LoadModule((net = make_unique<Network>()).get())) return ERRORv(-1, "network init failed");
  }

  if (FLAGS_lfapp_camera) {
    if (LoadModule((camera = make_unique<Camera>()).get())) return ERRORv(-1, "camera init failed");
  }

  if (FLAGS_lfapp_cuda) {
    (cuda = make_unique<CUDA>())->Init();
  }

  scheduler.Init();
  if (scheduler.monolithic_frame) frame_time.GetTime(true);
  else                    screen->frame_time.GetTime(true);
  INFO("Application::Init() succeeded");
  initialized = true;
  return 0;
}

int Application::Start() {
  if (FLAGS_lfapp_audio && audio->Start()) return ERRORv(-1, "lfapp audio start failed");
  return 0;
}

int Application::HandleEvents(unsigned clicks) {
  int events = 0, module_events;
  for (auto i = modules.begin(); i != modules.end() && run; ++i)
    if ((module_events = (*i)->Frame(clicks)) > 0) events += module_events;

  // handle messages sent to main thread
  if (run) events += message_queue.HandleMessages();

  // fake threadpool that executes in main thread
  if (run && !FLAGS_threadpool_size) events += thread_pool.worker[0].queue->HandleMessages();

  return events;
}

int Application::EventDrivenFrame(bool handle_events) {
  if (!MainThread()) ERROR("Frame() called from thread ", Thread::GetId());
  unsigned clicks = screen->frame_time.GetTime(true).count();
  if (handle_events) HandleEvents(clicks);

  int ret = screen->Frame(clicks, 0);
  if (FLAGS_frame_debug) INFO("frame_debug Application::Frame Window ", screen->id, " = ", ret);

  frames_ran++;
  return clicks;
}

int Application::TimerDrivenFrame(bool got_wakeup) {
  if (!MainThread()) ERROR("MonolithicFrame() called from thread ", Thread::GetId());
  unsigned clicks = frame_time.GetTime(true).count();
  int events = HandleEvents(clicks) + got_wakeup;

  for (auto i = windows.begin(); run && i != windows.end(); ++i) {
    auto w = i->second;
#ifdef LFL_ANDROID
    if (w->minimized || (!w->target_fps && !events)) continue;
#else
    if (w->minimized || !w->target_fps) continue;
#endif
    int ret = w->Frame(clicks, 0);
    if (FLAGS_frame_debug) INFO("frame_debug Application::Frame Window ", w->id, " = ", ret);
  }

  frames_ran++;
  return clicks;
}

int Application::Main() {
  ONCE({
    scheduler.Start();
#ifdef LFL_IPHONE
    return 0;
#endif
  });
  if (Start()) return -1;
#if 0
#if defined(LFL_WINVIDEO) || defined(LFL_WXWIDGETS)
  return 0;
#endif
#endif
  if (!scheduler.run_main_loop) return 0;
  return MainLoop();
}

int Application::MainLoop() {
  INFO("MainLoop: Begin, run=", run);
  while (run) {
    bool got_wakeup = scheduler.FrameWait();
    TimerDrivenFrame(got_wakeup);
#ifdef LFL_ANDROID
    if (screen->minimized) { INFO("MainLoop: minimized"); return 0; }
#endif
    scheduler.FrameDone();
    MSleep(1);
  }
  INFO("MainLoop: End, run=", run);
  return 0;
}

void Application::ResetGL() {
  for (auto &w : windows) w.second->ResetGL();
  fonts->ResetGL();
}

Application::~Application() {
  run = 0;
  INFO("exiting");
  if (network_thread) {
    network_thread->Write(new Callback([](){}));
    network_thread->thread->Wait();
    network_thread->net->Free();
  }
  message_queue.HandleMessages();
  if (!FLAGS_threadpool_size && thread_pool.worker.size()) thread_pool.worker[0].queue->HandleMessages();
  if (fonts) fonts.reset();
  if (shaders) shaders.reset();
  vector<Window*> close_list;
  for (auto &i : windows) close_list.push_back(i.second);
  for (auto &i : close_list) CloseWindow(i);
  if (exit_cb) exit_cb();
  if (cuda) cuda->Free();
  for (auto &m : modules) m->Free();
  if (video) video->Free();
  scheduler.Free();
  if (logfile) fclose(logfile);
#ifdef WIN32
  if (FLAGS_open_console) PromptFGets("Press [enter] to continue...");
#endif
}

/* Window */

Window::Window() : caption("lfapp"), fps(128) {
  id = gl = surface = glew_context = impl = user1 = user2 = user3 = typed_ptr{0, nullptr};
  minimized = cursor_grabbed = frame_init = animating = 0;
  target_fps = FLAGS_target_fps;
  multitouch_keyboard_x = .93; 
  cam = make_unique<Entity>(v3(5.54, 1.70, 4.39), v3(-.51, -.03, -.49), v3(-.03, 1, -.03));
  SetSize(point(640, 480));
}

Window::~Window() {
  if (console) console->WriteHistory(LFAppDownloadDir(), "console", "");
  if (gd) delete gd;
}

Box Window::Box(float xp, float yp, float xs, float ys, float xbl, float ybt, float xbr, float ybb) const {
  if (isinf(xbr)) xbr = xbl;
  if (isinf(ybb)) ybb = ybt;
  return LFL::Box(width  * (xp + xbl),
                  height * (yp + ybb),
                  width  * xs - width  * (xbl + xbr),
                  height * ys - height * (ybt + ybb), false);
}

void Window::InitConsole(const Callback &animating_cb) {
  gui.push_back((console = make_unique<Console>(gd, animating_cb)).get());
  console->ReadHistory(LFAppDownloadDir(), "console");
  console->Write(StrCat(screen->caption, " started"));
  console->Write("Try console commands 'cmds' and 'flags'");
}

size_t Window::NewGUI() { my_gui.emplace_back(unique_ptr<GUI>()); return my_gui.size()-1; }
void Window::DelGUI(GUI *g) { RemoveGUI(g); VectorRemoveUnique(&my_gui, g); }

void Window::BringDialogToFront(Dialog *d) {
  if (top_dialog == d) return;
  if (top_dialog) top_dialog->LoseFocus();
  int zsort_ind = 0;
  for (auto &d : dialogs) d->zsort = ++zsort_ind;
  d->zsort = 0;
  sort(dialogs.begin(), dialogs.end(), Dialog::LessThan);
  (top_dialog = d)->TakeFocus();
}

void Window::GiveDialogFocusAway(Dialog *d) {
  if (top_dialog == d) { top_dialog=0; d->LoseFocus(); }
}

void Window::DrawDialogs() {
  for (auto i = screen->dialogs.begin(), e = screen->dialogs.end(); i != e; ++i) (*i)->Draw();
  if (screen->console) screen->console->Draw();
  if (FLAGS_draw_grid) {
    Color c(.7, .7, .7);
    glIntersect(screen->mouse.x, screen->mouse.y, &c);
    default_font->Draw(StrCat("draw_grid ", screen->mouse.x, " , ", screen->mouse.y), point(0,0));
  }
}

void Window::SetSize(const point &d) {
  pow2_width  = NextPowerOfTwo((width  = d.x));
  pow2_height = NextPowerOfTwo((height = d.y));
}

void Window::Reshaped(int w, int h) {
  INFO("Window::Reshaped(", w, ", ", h, ")");
  SetSize(point(w, h));
  if (!gd) return;
  gd->ViewPort(LFL::Box(width, height));
  gd->DrawMode(screen->gd->default_draw_mode);
  for (auto g = screen->gui.begin(); g != screen->gui.end(); ++g) (*g)->Layout();
  if (app->reshaped_cb) app->reshaped_cb();
}

void Window::ResetGL() {
  Video::InitGraphicsDevice(this);
  for (auto &g : screen->gui    ) g->ResetGL();
  for (auto &g : screen->dialogs) g->ResetGL();
}

void Window::SwapAxis() {
  FLAGS_rotate_view = FLAGS_rotate_view ? 0 : -90;
  FLAGS_swap_axis = FLAGS_rotate_view != 0;
  Reshaped(height, width);
}

int Window::Frame(unsigned clicks, int flag) {
  if (screen != this) app->MakeCurrentWindow(this);

  if (FLAGS_lfapp_video) {
    if (!frame_init && (frame_init = true))  {
#ifdef LFL_IPHONE
      screen->GetIntegerv(GL_FRAMEBUFFER_BINDING_OES, &screen->gd->default_framebuffer);
      INFO("default_framebuffer = ", screen->gd->default_framebuffer);
#endif
    }
    gd->DrawMode(gd->default_draw_mode);
    gd->Clear();
    gd->LoadIdentity();
  }

  /* frame */
  int ret = frame_cb ? frame_cb(screen, clicks, flag) : 0;

  /* allow app to skip frame */
  if (ret < 0) return ret;
  fps.Add(clicks);

  if (FLAGS_lfapp_video) {
    app->video->Swap();
  }
  return ret;
}

void Window::RenderToFrameBuffer(FrameBuffer *fb) {
  int dm = screen->gd->draw_mode;
  fb->Attach();
  // screen->gd->ViewPort(Box(fb->tex.width, fb->tex.height));
  screen->gd->DrawMode(screen->gd->default_draw_mode);
  screen->gd->Clear();
  frame_cb(0, 0, 0);
  fb->Release();
  screen->gd->RestoreViewport(dm);
}

/* FrameScheduler */

#if 0
FrameScheduler::FrameScheduler() : maxfps(&FLAGS_target_fps), wakeup_thread(&frame_mutex, &wait_mutex) {
#if defined(LFL_IPHONEINPUT)
  rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = 0;
#elif defined(LFL_ANDROIDINPUT)
  synchronize_waits = wait_forever_thread = monolithic_frame = 0;
#elif defined(LFL_WININPUT) || defined(LFL_X11INPUT)
  synchronize_waits = wait_forever_thread = 0;
#elif defined(LFL_WXWIDGETS)
  rate_limit = synchronize_waits = monolithic_frame = 0;
#endif
}
#endif

void FrameScheduler::Init() { 
  screen->target_fps = FLAGS_target_fps;
  wait_forever = !FLAGS_target_fps;
  maxfps.timer.GetTime(true);
  if (wait_forever && synchronize_waits) frame_mutex.lock();
}

void FrameScheduler::Free() { 
  if (wait_forever && synchronize_waits) frame_mutex.unlock();
  if (wait_forever && wait_forever_thread) wakeup_thread.Wait();
}

void FrameScheduler::Start() {
  if (!wait_forever) return;
  if (wait_forever_thread) wakeup_thread.Start();
#if defined(LFL_ANDROID)
  Socket fd[2];
  CHECK(SystemNetwork::OpenSocketPair(fd));
  AddWaitForeverSocket((system_event_socket = fd[0]), SocketSet::READABLE, 0);
  wait_forever_wakeup_socket = fd[1];
#endif
}

void FrameScheduler::FrameDone() { if (rate_limit && app->run && FLAGS_target_fps) maxfps.Limit(); }

bool FrameScheduler::FrameWait() {
  bool ret = false;
  if (wait_forever && !FLAGS_target_fps) {
    if (synchronize_waits) {
      wait_mutex.lock();
      frame_mutex.unlock();
    }
    DoWait();
    if (synchronize_waits) {
      frame_mutex.lock();
      wait_mutex.unlock();
    }
  }
  return ret;
}

#if 0
void FrameScheduler::DoWait() {
#if defined(LFL_WININPUT) || defined(LFL_IPHONEINPUT) || defined(LFL_WXWIDGETS)
#elif defined(LFL_ANDROIDINPUT) || defined(LFL_X11INPUT)
  wait_forever_sockets.Select(-1);
  for (auto &s : wait_forever_sockets.socket)
    if (wait_forever_sockets.GetReadable(s.first)) {
      if (s.first != system_event_socket) app->scheduler.Wakeup(s.second.second);
#ifdef LFL_ANDROIDINPUT
      else {
        char buf[512];
        int l = read(system_event_socket, buf, sizeof(buf));
        for (const char *p = buf, *e = p + l; p < e; p++) if (*p) { ret = true; break; }
      }
#endif
    }
#elif defined(LFL_GLFWINPUT)
  glfwWaitEvents();
#elif defined(LFL_SDLINPUT)
  SDL_WaitEvent(NULL);
#else
  // FATAL("not implemented");
#endif
}
#endif

#if 0
void FrameScheduler::Wakeup(void *opaque) {
  if (wait_forever && screen) {
#if defined(LFL_WININPUT)
    InvalidateRect((HWND)screen->id, NULL, 0);
    // PostMessage((HWND)screen->id, WM_USER, 0, 0);
#elif defined(LFL_X11INPUT)
    XEvent exp;
    exp.type = Expose;
    exp.xexpose.window = (::Window)screen->id;
    XSendEvent((Display*)screen->surface, exp.xexpose.window, 0, ExposureMask, &exp);
#elif defined(LFL_ANDROIDINPUT)
    char c = opaque ? 0 : 'W';
    write(wait_forever_wakeup_socket, &c, 1);
#elif defined(LFL_IPHONEINPUT)
    iPhoneTriggerFrame(screen->id);
#elif defined(LFL_WXWIDGETS)
    if (wait_forever_thread) ((wxGLCanvas*)screen->id)->Refresh();
#elif defined(LFL_GLFWINPUT)
    if (wait_forever_thread) glfwPostEmptyEvent();
#elif defined(LFL_SDLINPUT)
    if (wait_forever_thread) {
      static int my_event_type = SDL_RegisterEvents(1);
      CHECK_GE(my_event_type, 0);
      SDL_Event event;
      SDL_zero(event);
      event.type = my_event_type;
      SDL_PushEvent(&event);
    }
#else
    // FATAL("not implemented");
#endif
  }
}
#endif

#if 0
bool FrameScheduler::WakeupIn(void *opaque, Time interval, bool force) {
  // CHECK(!screen->target_fps);
#if defined(LFL_IPHONEINPUT)
  return iPhoneTriggerFrameIn(screen->id, interval.count(), force);
#elif defined(LFL_OSXVIDEO)
  return OSXTriggerFrameIn(screen->id, interval.count(), force);
#endif
  return 0;
}
#endif

#if 0
void FrameScheduler::ClearWakeupIn() {
#if defined(LFL_IPHONEINPUT)
  iPhoneClearTriggerFrameIn(screen->id);
#endif
}
#endif

void FrameScheduler::UpdateTargetFPS(int fps) {
  screen->target_fps = fps;
  if (monolithic_frame) {
    int next_target_fps = 0;
    for (const auto &w : app->windows) Max(&next_target_fps, w.second->target_fps);
    FLAGS_target_fps = next_target_fps;
  }
  CHECK(screen->id.value);
#if defined(LFL_IPHONEINPUT)
  // iPhoneUpdateTargetFPS(screen->id);
#elif defined(LFL_OSXVIDEO)
  // OSXUpdateTargetFPS(screen->id);
#elif defined(LFL_QT)
  // QTTriggerFrame(screen->id.value);
#endif
}

void FrameScheduler::SetAnimating(bool is_animating) {
  screen->animating = is_animating;
  int target_fps = is_animating ? FLAGS_peak_fps : 0;
  if (target_fps != screen->target_fps) {
    UpdateTargetFPS(target_fps);
    Wakeup(0);
  }
}

#if 0
void FrameScheduler::AddWaitForeverMouse() {
  CHECK(screen->id.value);
#if defined(LFL_WINVIDEO)
  static_cast<WinWindow*>(screen->impl)->frame_on_mouse_input = true;
#elif defined(LFL_ANDROIDINPUT)
  AndroidSetFrameOnMouseInput(true);
#elif defined(LFL_IPHONEINPUT)
  iPhoneAddWaitForeverMouse(screen->id);
#endif
}
#endif

#if 0
void FrameScheduler::DelWaitForeverMouse() {
  CHECK(screen->id.value);
#if defined(LFL_WINVIDEO)
  static_cast<WinWindow*>(screen->impl)->frame_on_mouse_input = false;
#elif defined(LFL_ANDROIDINPUT)
  AndroidSetFrameOnMouseInput(false);
#elif defined(LFL_IPHONEINPUT)
  iPhoneDelWaitForeverMouse(screen->id);
#endif
}
#endif

#if 0
void FrameScheduler::AddWaitForeverKeyboard() {
  CHECK(screen->id.value);
#if defined(LFL_WINVIDEO)
  static_cast<WinWindow*>(screen->impl)->frame_on_keyboard_input = true;
#elif defined(LFL_ANDROIDINPUT)
  AndroidSetFrameOnKeyboardInput(true);
#elif defined(LFL_IPHONEINPUT)
  iPhoneAddWaitForeverKeyboard(screen->id);
#endif
}
#endif

#if 0
void FrameScheduler::DelWaitForeverKeyboard() {
  CHECK(screen->id.value);
#if defined(LFL_WINVIDEO)
  static_cast<WinWindow*>(screen->impl)->frame_on_keyboard_input = false;
#elif defined(LFL_ANDROIDINPUT)
  AndroidSetFrameOnKeyboardInput(false);
#elif defined(LFL_IPHONEINPUT)
  iPhoneDelWaitForeverKeyboard(screen->id);
#endif
}
#endif

#if 0
void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, val);
#if defined(LFL_WINVIDEO)
  WSAAsyncSelect(fd, (HWND)screen->id, WM_USER, FD_READ | FD_CLOSE);
#elif defined(LFL_X11INPUT) || defined(LFL_ANDROIDINPUT)
  wait_forever_sockets.Add(fd, flag, val);  
#elif defined(LFL_IPHONEINPUT)
  if (!wait_forever_thread) { CHECK_EQ(SocketSet::READABLE, flag); iPhoneAddWaitForeverSocket(screen->id, fd); }
#endif
}
#endif

#if 0
void FrameScheduler::DelWaitForeverSocket(Socket fd) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
#if defined(LFL_X11INPUT) || defined(LFL_ANDROIDINPUT)
  wait_forever_sockets.Del(fd);
#elif defined(LFL_WINVIDEO)
  WSAAsyncSelect(fd, (HWND)screen->id, WM_USER, 0);
#elif defined(LFL_IPHONEINPUT)
  CHECK(screen->id);
  iPhoneDelWaitForeverSocket(screen->id, fd);
#endif
}
#endif

}; // namespace LFL

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpCmdLine, int nCmdShow) {
  vector<const char *> av;
  vector<string> a(1);
  a[0].resize(1024);
  GetModuleFileName(hInst, &(a[0])[0], a[0].size());
  LFL::StringWordIter word_iter(lpCmdLine);
  for (string word = IterNextString(&word_iter); !word_iter.Done(); word = IterNextString(&word_iter)) a.push_back(word);
  for (auto &i : a) av.push_back(i.c_str());
  av.push_back(0);
#ifdef LFL_WINVIDEO
  LFL::WinApp *winapp = LFL::Singleton<LFL::WinApp>::Get();
  winapp->Setup(hInst, nCmdShow);
#endif
  int ret = main(av.size()-1, &av[0]);
#ifdef LFL_WINVIDEO
  return ret ? ret : winapp->MessageLoop();
#else
  return ret;
#endif
}
#endif
