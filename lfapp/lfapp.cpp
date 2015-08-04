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

extern "C" {
#ifdef LFL_FFMPEG
#include <libavformat/avformat.h>
#endif
};

#ifdef LFL_PROTOBUF
#include <google/protobuf/message.h>
#endif

#include "lfapp/lfapp.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"

#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef WIN32
#define CALLBACK __stdcall
#include <Shlobj.h>
#include <Windns.h>
#define stat(x,y) _stat(x,y)
#define gmtime_r(i,o) memcpy(o, gmtime(&in), sizeof(tm))
#define localtime_r(i,o) memcpy(o, localtime(&in), sizeof(tm))
#else
#include <signal.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/resource.h>
#endif

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#ifndef LFL_IPHONE
#include <ApplicationServices/ApplicationServices.h>
#endif
#endif

#ifdef LFL_LINUXINPUT
#include "X11/Xlib.h"
#undef KeyPress
#endif

#ifdef LFL_ANDROID
#include <android/log.h>
#endif

#ifdef LFL_QT
extern "C" void QTTriggerFrame();
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
#elif defined(__APPLE__)
extern "C" void OSXStartWindow(void*);
extern "C" void OSXCreateNativeMenu(const char*, int, const char**, const char**, const char**);
extern "C" void OSXLaunchNativeFontChooser(const char *, int, const char *);
extern "C" void OSXTriggerFrame(void*);
extern "C" bool OSXTriggerFrameIn(void*, int ms, bool force);
extern "C" void OSXClearTriggerFrameIn(void *O);
extern "C" void OSXUpdateTargetFPS(void*);
extern "C" void OSXAddWaitForeverMouse(void*);
extern "C" void OSXDelWaitForeverMouse(void*);
extern "C" void OSXAddWaitForeverKeyboard(void*);
extern "C" void OSXDelWaitForeverKeyboard(void*);
extern "C" void OSXAddWaitForeverSocket(void*, int fd);
extern "C" void OSXDelWaitForeverSocket(void*, int fd);
#endif

extern "C" void BreakHook() {}
extern "C" void ShellRun(const char *text) { return LFL::app->shell.Run(text); }
extern "C" NativeWindow *GetNativeWindow() { return LFL::screen; }
extern "C" LFApp        *GetLFApp()        { return LFL::app; }
extern "C" int LFAppMain()                 { return LFL::app->Main(); }
extern "C" int LFAppMainLoop()             { return LFL::app->MainLoop(); }
extern "C" int LFAppFrame(bool handle_ev)  { return LFL::app->EventDrivenFrame(handle_ev); }
extern "C" const char *LFAppDownloadDir()  { return LFL::app->dldir.c_str(); }
extern "C" void LFAppShutdown() { LFL::app->run=0; LFL::app->scheduler.Wakeup(0); }
extern "C" void WindowReshaped(int w, int h) { LFL::screen->Reshaped(w, h); }
extern "C" void WindowMinimized()            { LFL::screen->Minimized(); }
extern "C" void WindowUnMinimized()          { LFL::screen->UnMinimized(); }
extern "C" void WindowClosed()               { LFL::screen->Closed(); }
extern "C" int  KeyPress  (int b, int d)                 { return LFL::app->input.KeyPress  (b, d); }
extern "C" int  MouseClick(int b, int d, int x,  int y)  { return LFL::app->input.MouseClick(b, d, LFL::point(x, y)); }
extern "C" int  MouseMove (int x, int y, int dx, int dy) { return LFL::app->input.MouseMove (LFL::point(x, y), LFL::point(dx, dy)); }
extern "C" void EndpointRead(void *svc, const char *name, const char *buf, int len) { LFL::app->network.EndpointRead((LFL::Service*)svc, name, buf, len); }
extern "C" NativeWindow *SetNativeWindowByID(void *id) { return SetNativeWindow(LFL::FindOrNull(LFL::Window::active, id)); }
extern "C" NativeWindow *SetNativeWindow(NativeWindow *W) {
  CHECK(W);
  if (W == LFL::screen) return W;
  LFL::Window::MakeCurrent((LFL::screen = static_cast<LFL::Window*>(W)));
  return W;
}
extern "C" void SetLFAppMainThread() {
  LFL::Thread::Id id = LFL::Thread::GetId();
  if (LFL::app->main_thread_id != id) INFOf("LFApp->main_thread_id changed from %llx to %llx", LFL::app->main_thread_id, id);
  LFL::app->main_thread_id = id; 
}
extern "C" void LFAppFatal() {
  ERROR("LFAppFatal");
  if (bool suicide=true) *(volatile int*)0 = 0;
  LFL::app->run = 0;
  exit(-1);
}

namespace LFL {
Application *app = new Application();
Window *screen = new Window();

DEFINE_bool(lfapp_audio, false, "Enable audio in/out");
DEFINE_bool(lfapp_video, false, "Enable OpenGL");
DEFINE_bool(lfapp_input, false, "Enable keyboard/mouse input");
DEFINE_bool(lfapp_camera, false, "Enable camera capture");
DEFINE_bool(lfapp_cuda, false, "Enable CUDA acceleration");
DEFINE_bool(lfapp_network, false, "Enable asynchronous network engine");
DEFINE_bool(lfapp_debug, false, "Enable debug mode");
DEFINE_bool(cursor_grabbed, false, "Center cursor every frame");
DEFINE_bool(daemonize, false, "Daemonize server");
DEFINE_bool(rcon_debug, false, "Print rcon commands");
DEFINE_bool(frame_debug, false, "Print each frame");
DEFINE_string(nameserver, "", "Default namesver");
DEFINE_bool(max_rlimit_core, true, "Max core dump rlimit");
DEFINE_bool(max_rlimit_open_files, false, "Max number of open files rlimit");
DEFINE_int(loglevel, -1, "Log level: [Fatal=-1, Error=0, Info=3, Debug=7]");
DEFINE_int(threadpool_size, 0, "Threadpool size");
DEFINE_int(sample_rate, 16000, "Audio sample rate");
DEFINE_int(sample_secs, 3, "Seconds of RingBuf audio");
DEFINE_int(chans_in, -1, "Audio input channels");
DEFINE_int(chans_out, -1, "Audio output channels");
DEFINE_int(target_fps, 0, "Max frames per second");
DEFINE_bool(open_console, 0, "Open console on win32");

void Allocator::Reset() { FATAL(Name(), ": reset"); }
Allocator *Allocator::Default() { return Singleton<MallocAlloc>::Get(); }

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
void ThreadLocalStorage::ThreadFree() { Replace(&tls_instance, static_cast<ThreadLocalStorage*>(nullptr)); }
ThreadLocalStorage *ThreadLocalStorage::Get() { return tls_instance ? tls_instance : (tls_instance = new ThreadLocalStorage()); }
#endif
Allocator *ThreadLocalStorage::GetAllocator(bool reset_allocator) {
  ThreadLocalStorage *tls = Get();
  if (!tls->alloc) tls->alloc = new FixedAlloc<1024*1024>;
  if (reset_allocator) tls->alloc->Reset();
  return tls->alloc;
}

void Log(int level, const char *file, int line, const string &m) { app->Log(level, file, line, m); }
bool Running() { return app->run; }
bool MainThread() { return Thread::GetId() == app->main_thread_id; }
void DefaultLFAppWindowClosedCB(Window *W) { delete W; }
double FPS() { return screen->fps.FPS(); }
double CamFPS() { return app->camera.fps.FPS(); }
void RunInMainThread(Callback *cb) {
  app->message_queue.Write(cb);
  if (!FLAGS_target_fps) app->scheduler.Wakeup(0);
}
void PressAnyKey() {
  printf("Press [enter] to continue..."); fflush(stdout);
  char buf[32]; fgets(buf, sizeof(buf), stdin);
}
bool FGets(char *buf, int len) { return NBFGets(stdin, buf, len); }
bool NBFGets(FILE *f, char *buf, int len, int timeout) {
#ifndef WIN32
  int fd = fileno(f);
  SelectSocketSet ss;
  ss.Add(fd, SocketSet::READABLE, 0);
  ss.Select(timeout);
  if (!app->run || !ss.GetReadable(fd)) return 0;
  fgets(buf, len, f);
  return 1;
#else
  return 0;
#endif
}
bool NBReadable(Socket fd, int timeout) {
  SelectSocketSet ss;
  ss.Add(fd, SocketSet::READABLE, 0);
  ss.Select(timeout);
  return app->run && ss.GetReadable(fd);
}
int NBRead(Socket fd, char *buf, int len, int timeout) {
  if (!NBReadable(fd, timeout)) return 0;
  int o = 0, s = 0;
  do if ((s = read(fd, buf+o, len-o)) > 0) o += s;
  while (s > 0 && len - o > 1024);
  return o;
}
int NBRead(Socket fd, string *buf, int timeout) {
  int l = NBRead(fd, (char*)buf->data(), buf->size(), timeout);
  buf->resize(max(0,l));
  return l;
}
string NBRead(Socket fd, int len, int timeout) {
  string ret(len, 0);
  NBRead(fd, &ret, timeout);
  return ret;
}

void *MallocAlloc::Malloc(int size) { return ::malloc(size); }
void *MallocAlloc::Realloc(void *p, int size) { 
  if (!p) return ::malloc(size);
#ifdef __APPLE__
  else return ::reallocf(p, size);
#else
  else return ::realloc(p, size);
#endif
}
void  MallocAlloc::Free(void *p) { return ::free(p); }

MMapAlloc::~MMapAlloc() {
#ifdef WIN32
  UnmapViewOfFile(addr);
  CloseHandle(map);
  CloseHandle(file);
#else
  munmap(addr, size);
#endif
}

MMapAlloc *MMapAlloc::Open(const char *path, bool logerror, bool readonly, long long size) {
#ifdef LFL_ANDROID
  return 0;
#endif
#ifdef WIN32
  HANDLE file = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  if (file == INVALID_HANDLE_VALUE) { if (logerror) ERROR("CreateFile ", path, " failed ", GetLastError()); return 0; }

  DWORD hsize, lsize=GetFileSize(file, &hsize);

  HANDLE map = CreateFileMapping(file, 0, PAGE_READONLY, 0, 0, 0);
  if (!map) { ERROR("CreateFileMapping ", path, " failed"); return 0; }

  void *addr = MapViewOfFile(map, readonly ? FILE_MAP_READ : FILE_MAP_COPY, 0, 0, 0);
  if (!addr) { ERROR("MapViewOfFileEx ", path, " failed ", GetLastError()); return 0; }

  INFO("MMapAlloc::open(", path, ")");
  return new MMapAlloc(file, map, addr, lsize);
#else
  int fd = ::open(path, O_RDONLY);
  if (fd < 0) { if (logerror) ERROR("open ", path, " failed: ", strerror(errno)); return 0; }

  if (!size) {
    struct stat s;
    if (fstat(fd, &s)) { ERROR("fstat failed: ", strerror(errno)); close(fd); return 0; }
    size = s.st_size;
  }

  char *buf = (char *)mmap(0, size, PROT_READ | (readonly ? 0 : PROT_WRITE) , MAP_PRIVATE, fd, 0);
  if (buf == MAP_FAILED) { ERROR("mmap failed: ", strerror(errno)); close(fd); return 0; }

  close(fd);
  INFO("MMapAlloc::open(", path, ")");
  return new MMapAlloc(buf, size);
#endif
}

void *BlockChainAlloc::Malloc(int n) { 
  n = NextMultipleOfPowerOfTwo(n, 16);
  CHECK_LT(n, block_size);
  if (cur_block_ind == -1 || blocks[cur_block_ind].len + n > block_size) {
    cur_block_ind++;
    if (cur_block_ind >= blocks.size()) blocks.emplace_back(block_size);
    CHECK_EQ(blocks[cur_block_ind].len, 0);
    CHECK_LT(n, block_size);
  }
  Block *b = &blocks[cur_block_ind];
  char *ret = &b->buf[b->len];
  b->len += n;
  return ret;
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
  vector<int> keyv(key.size());
  for (int j=0, l=key.size(); j<l; j++) keyv[0] = key[j];

  vector<string> db;
  for (AllFlags::const_iterator i = flagmap.begin(); i != flagmap.end(); i++) {
    if (source_filename && strcmp(source_filename, i->second->file)) continue;
    db.push_back(i->first);
  }

  string mindistflag = "";
  double dist, mindist = INFINITY;
  for (int i = 0; i < db.size(); i++) {
    const string &t = db[i];
    vector<int> dbiv(t.size());
    for (int j=0, l=t.size(); j<l; j++) dbiv[j] = t[j];
    if ((dist = Levenshtein(keyv, dbiv)) < mindist) { mindist = dist; mindistflag = t; }
  }
  return mindistflag;
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

#ifdef WIN32
BOOL WINAPI CtrlHandler(DWORD sig) { INFO("interrupt"); LFAppShutdown(); return TRUE; }
void OpenConsole() {
  FLAGS_open_console=1;
  AllocConsole();
  SetConsoleTitle(StrCat(screen->caption, " console").c_str());
  freopen("CONOUT$", "wb", stdout);
  freopen("CONIN$", "rb", stdin);
  SetConsoleCtrlHandler(CtrlHandler ,1);
}
void CloseConsole() {
  fclose(stdin);
  fclose(stdout);
  FreeConsole();
}
void Application::Daemonize(const char *dir) {}
#else /* WIN32 */
void HandleSigInt(int sig) { INFO("interrupt"); LFAppShutdown(); }
void OpenConsole() {}
void CloseConsole() {}
void Application::Daemonize(const char *dir) {
  char fn1[256], fn2[256];
  snprintf(fn1, sizeof(fn1), "%s%s.stdout", dir, app->progname.c_str());
  snprintf(fn2, sizeof(fn2), "%s%s.stderr", dir, app->progname.c_str());
  FILE *fout = fopen(fn1, "a"); fprintf(stderr, "open %s %s\n", fn1, fout ? "OK" : strerror(errno));
  FILE *ferr = fopen(fn2, "a"); fprintf(stderr, "open %s %s\n", fn2, ferr ? "OK" : strerror(errno));
  Daemonize(fout, ferr);
}
void Application::Daemonize(FILE *fout, FILE *ferr) {
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
}
#endif /* WIN32 */

#ifdef LFL_IPHONE
void Vault::SavePassword(const string &h, const string &u, const string &pw) {
  iPhonePasswordSave(app->name.c_str(), h.c_str(), u.c_str(), pw.c_str(), pw.size());
}
bool Vault::LoadPassword(const string &h, const string &u, string *pw) {
  pw->resize(1024);
  pw->resize(iPhonePasswordCopy(app->name.c_str(), h.c_str(), u.c_str(), &(*pw)[0], pw->size()));
  return pw->size();
}
#endif

void SystemBrowser::Open(const char *url_text) {
#if defined(LFL_ANDROID)
  AndroidOpenBrowser(url_text);
#elif defined(LFL_IPHONE)
  iPhoneOpenBrowser(url_text);
#elif defined(__APPLE__)
  CFURLRef url = CFURLCreateWithBytes(0, (UInt8*)url_text, strlen(url_text), kCFStringEncodingASCII, 0);
  if (url) { LSOpenCFURLRef(url, 0); CFRelease(url); }
#elif defined(LFL_WINVIDEO)
  ShellExecute(NULL, "open", url_text, NULL, NULL, SW_SHOWNORMAL);
#endif
}

void Advertising::ShowAds() {
#if defined(LFL_ANDROID)
  AndroidShowAds();
#endif
}

void Advertising::HideAds() {
#if defined(LFL_ANDROID)
  AndroidHideAds();
#endif
}

void Application::Log(int level, const char *file, int line, const string &message) {
  char tbuf[64];
  logtime(tbuf, sizeof(tbuf));
  {
    ScopedMutex sm(log_mutex);

    fprintf(stdout, "%s %s (%s:%d)\r\n", tbuf, message.c_str(), file, line);
    fflush(stdout);

    if (logfile) {
      fprintf(app->logfile, "%s %s (%s:%d)\r\n", tbuf, message.c_str(), file, line);
      fflush(app->logfile);
    }
#ifdef LFL_IPHONE
    iPhoneLog(StringPrintf("%s (%s:%d)", message.c_str(), file, line).c_str());
#endif
#ifdef LFL_ANDROID
    __android_log_print(ANDROID_LOG_INFO, screen->caption.c_str(), "%s (%s:%d)", message.c_str(), file, line);
#endif
  }
  if (level == LFApp::Log::Fatal) LFAppFatal();
  if (FLAGS_lfapp_video && screen && screen->lfapp_console) screen->lfapp_console->Write(message);
}

void Application::CreateNewWindow(const function<void(Window*)> &start_cb) {
  Window *orig_window = screen;
  Window *new_window = new Window();
  if (window_init_cb) window_init_cb(new_window);
  app->video.CreateGraphicsDevice(new_window);
  CHECK(Window::Create(new_window));
  Window::MakeCurrent(new_window);
  app->video.InitGraphicsDevice(new_window);
  app->input.Init(new_window);
  start_cb(new_window);
#ifdef LFL_OSXINPUT
  OSXStartWindow(screen->id);
#endif
  Window::MakeCurrent(orig_window);
}

NetworkThread *Application::CreateNetworkThread() {
  VectorEraseByValue(&app->modules, static_cast<Module*>(&network));
  NetworkThread *ret = new NetworkThread(&app->network);
  ret->thread->Start();
  return ret;
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
  app->shell.Run(StrCat(choose_cmd, " ", lf.lfFaceName, " ", cf.iPointSize/10, " ", flag));
#endif
}

void Application::LaunchNativeMenu(const string &title) {
#if defined(LFL_IPHONE)
  iPhoneLaunchNativeMenu(title.c_str());
#endif
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

StringPiece Application::LoadResource(int id) {
#ifdef WIN32
  HRSRC resource = FindResource(NULL, MAKEINTRESOURCE(id), MAKEINTRESOURCE(900));
  HGLOBAL resource_data = ::LoadResource(NULL, resource);
  return StringPiece((char*)LockResource(resource_data), SizeofResource(NULL, resource));
#else
  return StringPiece();
#endif
}

int Application::Create(int argc, const char **argv, const char *source_filename) {
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

#ifdef __APPLE__
  char rpath[1024];
  CFBundleRef mainBundle = CFBundleGetMainBundle();
  CFURLRef respath = CFBundleCopyResourcesDirectoryURL(mainBundle);
  CFURLGetFileSystemRepresentation(respath, true, (UInt8*)rpath, sizeof(rpath));
  CFRelease(respath);
  INFO("chdir(", rpath, ")");
  chdir(rpath);
#else
  INFO("chdir(", bindir, ")");
  chdir(bindir.c_str());
#endif

#ifndef LFL_ANDROID
  assetdir = "assets/";
#endif

#ifdef _WIN32
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

#ifdef _WIN32
  if (argc > 1) OpenConsole();
#endif

  if (Singleton<FlagMap>::Get()->getopt(argc, argv, source_filename) < 0) return -1;

#ifdef _WIN32
  if (argc > 1) {
    if (!FLAGS_open_console) CloseConsole();
  }
  else if (FLAGS_open_console) OpenConsole();
#endif

  {
#if defined(LFL_IPHONE)
    char *path = iPhoneDocumentPathCopy();
    dldir = StrCat(path, "/");
    free(path);
#elif defined(_WIN32)
    char path[MAX_PATH];
    if (!SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL|CSIDL_FLAG_CREATE, NULL, 0, path))) return -1;
    dldir = StrCat(path, "/");
#endif
  }

  const char *LFLHOME=getenv("LFLHOME");
  if (LFLHOME && *LFLHOME) chdir(LFLHOME);
  INFO(screen->caption, ": lfapp init: LFLHOME=", LocalFile::CurrentDirectory(), " DLDIR=", LFAppDownloadDir());

#ifndef _WIN32
  if (FLAGS_max_rlimit_core) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_CORE, &rl) == -1) { ERROR("core getrlimit ", strerror(errno)); return -1; }

    rl.rlim_cur = rl.rlim_max;
    if (setrlimit(RLIMIT_CORE, &rl) == -1) { ERROR("core setrlimit ", strerror(errno)); return -1; }
  }
#endif

#ifdef __linux__
  if (FLAGS_max_rlimit_open_files) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == -1) { ERROR("files getrlimit ", strerror(errno)); return -1; }

    rl.rlim_cur = rl.rlim_max = 999999;
    INFO("setrlimit(RLIMIT_NOFILE, ", rl.rlim_cur, ")");
    if (setrlimit(RLIMIT_NOFILE, &rl) == -1) { ERROR("files setrlimit ", strerror(errno)); return -1; }
  }
#endif

#ifdef LFL_HEADLESS
  Window::Create(screen);
#endif

  if (FLAGS_daemonize) {
    Daemonize();
    SetLFAppMainThread();
  }

  return 0;
}

int Application::Init() {
  if (FLAGS_lfapp_video) {
#if defined(LFL_GLFWVIDEO) || defined(LFL_GLFWINPUT)
    INFO("lfapp_open: glfwInit()");
    if (!glfwInit()) { ERROR("glfwInit: ", strerror(errno)); return -1; }
#endif
  }

  if (FLAGS_lfapp_audio || FLAGS_lfapp_video) {
#if defined(LFL_SDLVIDEO) || defined(LFL_SDLAUDIO) || defined(LFL_SDLINPUT)
    int SDL_Init_Flag = 0;
#ifdef LFL_SDLVIDEO
    SDL_Init_Flag |= (FLAGS_lfapp_video ? SDL_INIT_VIDEO : 0);
#endif
#ifdef LFL_SDLAUDIO
    SDL_Init_Flag |= (FLAGS_lfapp_audio ? SDL_INIT_AUDIO : 0);
#endif
    INFO("lfapp_open: SDL_init()");
    if (SDL_Init(SDL_Init_Flag) < 0) { ERROR("SDL_Init: ", SDL_GetError()); return -1; }
#endif
  }

  if (FLAGS_lfapp_video) {
    if (video.Init()) { ERROR("video init failed"); return -1; }
  } else {
    Window::active[screen->id] = screen;
  }

  thread_pool.Open(X_or_1(FLAGS_threadpool_size));
  if (FLAGS_threadpool_size) thread_pool.Start();

#ifdef LFL_FFMPEG
  INFO("lfapp_open: ffmpeg_init()");
  //av_log_set_level(AV_LOG_DEBUG);
  av_register_all();
#endif /* LFL_FFMPEG */

  if (FLAGS_lfapp_audio) {
    if (LoadModule(&audio)) { ERROR("audio init failed"); return -1; }
  }
  else { FLAGS_chans_in=FLAGS_chans_out=1; }

  if (FLAGS_lfapp_audio || FLAGS_lfapp_video) {
    if (assets.Init()) { ERROR("assets init failed"); return -1; }
  }

  app->video.InitFonts();
  if (FLAGS_lfapp_console && !screen->lfapp_console) screen->InitLFAppConsole();

  if (FLAGS_lfapp_input) {
    if (LoadModule(&input)) { ERROR("input init failed"); return -1; }
    input.Init(screen);
  }

  if (FLAGS_lfapp_network) {
    if (LoadModule(&network)) { ERROR("network init failed"); return -1; }
  }

  if (FLAGS_lfapp_camera) {
    if (LoadModule(&camera)) { ERROR("camera init failed"); return -1; }
  }

  if (FLAGS_lfapp_cuda) {
    cuda.Init();
  }

  scheduler.Init();
  if (scheduler.monolithic_frame) frame_time.GetTime(true);
  else                    screen->frame_time.GetTime(true);
  INFO("lfapp_open: succeeded");
  initialized = true;
  return 0;
}

int Application::Start() {
  if (FLAGS_lfapp_audio && audio.Start()) { ERROR("lfapp audio start failed"); return -1; }
  return 0;
}

int Application::HandleEvents(unsigned clicks) {
  for (auto i = modules.begin(); i != modules.end() && run; ++i) (*i)->Frame(clicks);

  // handle messages sent to main thread
  if (run) message_queue.HandleMessages();

  // fake threadpool that executes in main thread
  if (run && !FLAGS_threadpool_size) thread_pool.worker[0].queue->HandleMessages();

  return 0;
}

int Application::EventDrivenFrame(bool handle_events) {
  if (!MainThread()) ERROR("Frame() called from thread ", Thread::GetId());
  unsigned clicks = screen->frame_time.GetTime(true).count(), flag = 0;
  if (handle_events) HandleEvents(clicks);

  int ret = screen->Frame(clicks, audio.mic_samples, camera.have_sample, flag);
  if (FLAGS_frame_debug) INFO("frame_debug Application::Frame Window ", screen->id, " = ", ret);

  frames_ran++;
  return clicks;
}

int Application::TimerDrivenFrame() {
  if (!MainThread()) ERROR("MonolithicFrame() called from thread ", Thread::GetId());
  unsigned clicks = frame_time.GetTime(true).count(), flag = 0;
  HandleEvents(clicks);

  for (auto i = Window::active.begin(); run && i != Window::active.end(); ++i) {
    if (!i->second->target_fps) continue;
    int ret = i->second->Frame(clicks, audio.mic_samples, camera.have_sample, flag);
    if (FLAGS_frame_debug) INFO("frame_debug Application::Frame Window ", i->second->id, " = ", ret);
  }

  frames_ran++;
  return clicks;
}

int Application::Main() {
#ifdef LFL_IPHONE
  ONCE({ return 0; });
#endif
  if (Start()) return Exiting();
#if defined(LFL_OSXVIDEO) || defined(LFL_WINVIDEO) || defined(LFL_QT) || defined(LFL_WXWIDGETS)
  return 0;
#endif
  return MainLoop();
}

int Application::MainLoop() {
  INFO("MainLoop: Begin, run=", run);
  while (run) {
    // if (!minimized)
    scheduler.FrameWait();
    TimerDrivenFrame();
    scheduler.FrameDone();
#ifdef LFL_IPHONE
    // if (minimized) run = 0;
#endif
    MSleep(1);
  }
  INFO("MainLoop: End, run=", run);
  return Free();
}

int Application::Free() {
  while (!Window::active.empty()) Window::Close(Window::active.begin()->second);

  if (FLAGS_lfapp_video)  video .Free();
  if (FLAGS_lfapp_audio)  audio .Free();
  if (FLAGS_lfapp_camera) camera.Free();

  return Exiting();
}

int Application::Exiting() {
  run = 0;
  INFO("exiting");
  scheduler.Free();
#ifdef _WIN32
  if (FLAGS_open_console) PressAnyKey();
#endif
  return 0;
}

/* FrameScheduler */

FrameScheduler::FrameScheduler() : maxfps(&FLAGS_target_fps), wakeup_thread(&frame_mutex, &wait_mutex) {
#if defined(LFL_OSXINPUT) || defined(LFL_IPHONEINPUT)
  rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = 0;
#elif defined(LFL_WININPUT) || defined(LFL_LINUXINPUT)
  synchronize_waits = wait_forever_thread = 0;
  rate_limit = FLAGS_target_fps;
#elif defined(LFL_QT) || defined(LFL_WXWIDGETS)
  rate_limit = synchronize_waits = monolithic_frame = 0;
#endif
}

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
  if (wait_forever && wait_forever_thread) wakeup_thread.Start();
}

void FrameScheduler::FrameDone() { if (rate_limit && app->run && FLAGS_target_fps) maxfps.Limit(); }

void FrameScheduler::FrameWait() {
  if (wait_forever && !FLAGS_target_fps) {
    if (synchronize_waits) {
      wait_mutex.lock();
      frame_mutex.unlock();
    }
#if defined(LFL_OSXINPUT) || defined(LFL_WININPUT) || defined(LFL_IPHONEINPUT) || defined(LFL_QT) || defined(LFL_WXWIDGETS)
#elif defined(LFL_LINUXINPUT)
    wait_forever_sockets.Select(-1);
#elif defined(LFL_GLFWINPUT)
    glfwWaitEvents();
#elif defined(LFL_SDLINPUT)
    SDL_WaitEvent(NULL);
#else
    // FATAL("not implemented");
#endif
    if (synchronize_waits) {
      frame_mutex.lock();
      wait_mutex.unlock();
    }
  }
}

void FrameScheduler::Wakeup(void *opaque) {
  if (wait_forever && screen) {
#if defined(LFL_OSXINPUT)
    OSXTriggerFrame(screen->id);
#elif defined(LFL_WININPUT)
    InvalidateRect((HWND)screen->id, NULL, 0);
    // PostMessage((HWND)screen->id, WM_USER, 0, 0);
#elif defined(LFL_LINUXINPUT)
    XEvent exp;
    exp.type = Expose;
    exp.xexpose.window = (::Window)screen->id;
    XSendEvent((Display*)screen->surface, exp.xexpose.window, 0, ExposureMask, &exp);
#elif defined(LFL_IPHONEINPUT)
    iPhoneTriggerFrame(screen->id);
#elif defined(LFL_QT)
    if (wait_forever_thread) QTTriggerFrame();
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

bool FrameScheduler::WakeupIn(void *opaque, Time interval, bool force) {
  // CHECK(!screen->target_fps);
#if defined(LFL_IPHONEINPUT)
  return iPhoneTriggerFrameIn(screen->id, interval.count(), force);
#elif defined(LFL_OSXINPUT)
  return OSXTriggerFrameIn(screen->id, interval.count(), force);
#endif
  return 0;
}

void FrameScheduler::ClearWakeupIn() {
#if defined(LFL_IPHONEINPUT)
  iPhoneClearTriggerFrameIn(screen->id);
#elif defined(LFL_OSXINPUT)
  OSXClearTriggerFrameIn(screen->id);
#endif
}

void FrameScheduler::UpdateTargetFPS(int fps) {
  screen->target_fps = fps;
  if (monolithic_frame) {
    int next_target_fps = 0;
    for (const auto &w : Window::active) Max(&next_target_fps, w.second->target_fps);
    FLAGS_target_fps = next_target_fps;
  }
  CHECK(screen->id);
#if defined(LFL_IPHONEINPUT)
  iPhoneUpdateTargetFPS(screen->id);
#elif defined(LFL_OSXINPUT)
  OSXUpdateTargetFPS(screen->id);
#endif
}

void FrameScheduler::AddWaitForeverMouse() {
  CHECK(screen->id);
#if defined(LFL_IPHONEINPUT)
  iPhoneAddWaitForeverMouse(screen->id);
#elif defined(LFL_OSXINPUT)
  OSXAddWaitForeverMouse(screen->id);
#elif defined(LFL_WINVIDEO)
  static_cast<WinWindow*>(screen->impl)->frame_on_mouse_input = true;
#endif
}

void FrameScheduler::DelWaitForeverMouse() {
  CHECK(screen->id);
#if defined(LFL_IPHONEINPUT)
  iPhoneDelWaitForeverMouse(screen->id);
#elif defined(LFL_OSXINPUT)
  OSXDelWaitForeverMouse(screen->id);
#elif defined(LFL_WINVIDEO)
  static_cast<WinWindow*>(screen->impl)->frame_on_mouse_input = false;
#endif
}

void FrameScheduler::AddWaitForeverKeyboard() {
  CHECK(screen->id);
#if defined(LFL_IPHONEINPUT)
  iPhoneAddWaitForeverKeyboard(screen->id);
#elif defined(LFL_OSXINPUT)
  OSXAddWaitForeverKeyboard(screen->id);
#elif defined(LFL_WINVIDEO)
  static_cast<WinWindow*>(screen->impl)->frame_on_keyboard_input = true;
#endif
}

void FrameScheduler::DelWaitForeverKeyboard() {
  CHECK(screen->id);
#if defined(LFL_IPHONEINPUT)
  iPhoneDelWaitForeverKeyboard(screen->id);
#elif defined(LFL_OSXINPUT)
  OSXDelWaitForeverKeyboard(screen->id);
#elif defined(LFL_WINVIDEO)
  static_cast<WinWindow*>(screen->impl)->frame_on_keyboard_input = false;
#endif
}

void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, val);
#if defined(LFL_OSXINPUT)
  if (!wait_forever_thread) { CHECK_EQ(SocketSet::READABLE, flag); OSXAddWaitForeverSocket(screen->id, fd); }
#elif defined(LFL_WINVIDEO)
  WSAAsyncSelect(fd, (HWND)screen->id, WM_USER, FD_READ | FD_CLOSE);
#elif defined(LFL_LINUXINPUT)
  wait_forever_sockets.Add(fd, flag, val);  
#elif defined(LFL_IPHONEINPUT)
  if (!wait_forever_thread) { CHECK_EQ(SocketSet::READABLE, flag); iPhoneAddWaitForeverSocket(screen->id, fd); }
#endif
}

void FrameScheduler::DelWaitForeverSocket(Socket fd) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
#if defined(LFL_OSXINPUT)
  CHECK(screen->id);
  OSXDelWaitForeverSocket(screen->id, fd);
#elif defined(LFL_LINUXINPUT)
  wait_forever_sockets.Del(fd);
#elif defined(LFL_WINVIDEO)
  WSAAsyncSelect(fd, (HWND)screen->id, WM_USER, 0);
#elif defined(LFL_IPHONEINPUT)
  CHECK(screen->id);
  iPhoneDelWaitForeverSocket(screen->id, fd);
#endif
}
}; // namespace LFL
