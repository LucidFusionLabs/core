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

#include "core/app/gui.h"
#include "core/app/ipc.h"

#include <time.h>
#include <fcntl.h>

#ifdef LFL_WINDOWS
#define CALLBACK __stdcall
#include <Windns.h>
#else
#include <signal.h>
#include <pthread.h>
#include <dlfcn.h>
#include <sys/resource.h>
#endif

#ifdef LFL_APPLE
#include <CoreFoundation/CoreFoundation.h>
#endif

#ifdef LFL_ANDROID
#include <android/log.h>
#endif

#ifdef LFL_IPHONE
extern "C" char *iPhoneDocumentPathCopy();
extern "C" void iPhoneLog(const char *text);
#endif

extern "C" void BreakHook() {}
extern "C" void ShellRun(const char *text) { return LFL::screen->shell->Run(text); }
extern "C" NativeWindow *GetNativeWindow() { return LFL::screen; }
extern "C" LFApp        *GetLFApp()        { return LFL::app; }
extern "C" int LFAppMain()                 { return LFL::app->Main(); }
extern "C" int LFAppMainLoop()             { return LFL::app->MainLoop(); }
extern "C" int LFAppFrame(bool handle_ev)  { return LFL::app->EventDrivenFrame(handle_ev); }
extern "C" void LFAppTimerDrivenFrame()    { LFL::app->TimerDrivenFrame(true); }
extern "C" void LFAppWakeup(void *v)       { return LFL::app->scheduler.Wakeup(v); }
extern "C" void LFAppResetGL()             { return LFL::app->ResetGL(); }
extern "C" const char *LFAppDownloadDir()  { return LFL::app->dldir.c_str(); }
extern "C" void LFAppAtExit()              { delete LFL::app; }
extern "C" void LFAppShutdown()                   { LFL::app->run=0; LFAppWakeup(0); }
extern "C" void WindowReshaped(int w, int h)      { LFL::screen->Reshaped(w, h); }
extern "C" void WindowMinimized()                 { LFL::screen->Minimized(); }
extern "C" void WindowUnMinimized()               { LFL::screen->UnMinimized(); }
extern "C" bool WindowClosed()                    { LFL::app->CloseWindow(LFL::screen); return LFL::app->windows.empty(); }
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

extern "C" unsigned LFAppNextRandSeed() {
  static LFL::mutex m;
  LFL::ScopedMutex sm(m);
  return (LFL::FLAGS_rand_seed = LFL::fnv32(&LFL::FLAGS_rand_seed, sizeof(unsigned)));
}

extern "C" void LFAppFatal() {
  ERROR("LFAppFatal");
  LFL::app->run = 0;
  if (bool suicide=true) *reinterpret_cast<volatile int*>(0) = 0;
  throw std::runtime_error("LFAppFatal");
}

#ifndef LFL_WINDOWS
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
DEFINE_string(logfile, "", "Log file name");
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
DEFINE_unsigned(rand_seed, 0, "Random number generator seed");
DEFINE_bool(open_console, 0, "Open console on win32");
DEFINE_bool(cursor_grabbed, false, "Center cursor every frame");
DEFINE_bool(frame_debug, false, "Print each frame");
DEFINE_bool(rcon_debug, false, "Print game protocol commands");

Application *app = nullptr;
Window *screen = nullptr;
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

int FlagMap::getopt(int argc, const char* const* argv, const char *source_filename) {
  for (optind=1; optind<argc; /**/) {
    const char *arg = argv[optind], *key = arg + 1, *val = "";
    if (*arg != '-' || *(arg+1) == 0) break;

    if (++optind < argc && !(IsBool(key) && *(argv[optind]) == '-')) val = argv[optind++];
    if (!strcmp(key, "fullhelp")) { Print(); return -1; }

    if (!Set(key, val)) {
#ifdef LFL_APPLE
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
    WriteLogLine(log_pid ? StrCat("[", pid, "] ", tbuf).c_str() : tbuf, message.c_str(), file, line);
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
  __android_log_print(ANDROID_LOG_INFO, app->name.c_str(), "%s (%s:%d)", message, file, line);
#endif
}

void Application::CreateNewWindow() {
  Window *orig_window = screen, *new_window = new Window();
  if (window_init_cb) window_init_cb(new_window);
  new_window->gd = CreateGraphicsDevice(opengles_version).release();
  CHECK(Video::CreateWindow(new_window));
  if (!new_window->started && (new_window->started = true)) {
    MakeCurrentWindow(new_window);
    StartNewWindow(new_window);
    MakeCurrentWindow(orig_window);
  }
}

void Application::StartNewWindow(Window *new_window) {
  if (new_window->gd) {
    if (!new_window->gd->done_init) new_window->gd->Init(new_window->Box());
    new_window->default_font.Load();
  }
  if (window_start_cb) window_start_cb(new_window);
  Video::StartWindow(new_window);
}

NetworkThread *Application::CreateNetworkThread(bool detach, bool start) {
  CHECK(net);
  if (detach) VectorEraseByValue(&modules, static_cast<Module*>(net.get()));
  network_thread = make_unique<NetworkThread>(net.get(), !detach);
  if (start) network_thread->thread->Start();
  return network_thread.get();
}

void *Application::GetSymbol(const string &n) {
#ifdef LFL_WINDOWS
  return GetProcAddress(GetModuleHandle(NULL), n.c_str());
#else
  return dlsym(RTLD_DEFAULT, n.c_str());
#endif
}

StringPiece Application::LoadResource(int id) {
#ifdef LFL_WINDOWS
  HRSRC resource = FindResource(NULL, MAKEINTRESOURCE(id), MAKEINTRESOURCE(900));
  HGLOBAL resource_data = ::LoadResource(NULL, resource);
  return StringPiece((char*)LockResource(resource_data), SizeofResource(NULL, resource));
#else
  return StringPiece();
#endif
}

void Application::Daemonize(const char *dir, const char *progname) {
#ifndef LFL_WINDOWS
  char fn1[256], fn2[256];
  snprintf(fn1, sizeof(fn1), "%s%s.stdout", dir, progname);
  snprintf(fn2, sizeof(fn2), "%s%s.stderr", dir, progname);
  FILE *fout = fopen(fn1, "a"); fprintf(stderr, "open %s %s\n", fn1, fout ? "OK" : strerror(errno));
  FILE *ferr = fopen(fn2, "a"); fprintf(stderr, "open %s %s\n", fn2, ferr ? "OK" : strerror(errno));
  Daemonize(fout, ferr);
#endif
}

void Application::Daemonize(FILE *fout, FILE *ferr) {
#ifndef LFL_WINDOWS
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

int Application::Create(int argc, const char* const* argv, const char *source_filename) {
#ifdef LFL_GLOG
  google::InstallFailureSignalHandler();
#endif
  SetLFAppMainThread();
  time_started = Now();
  progname = argv[0];
  startdir = LocalFile::CurrentDirectory();

#if defined(LFL_ANDROID)
#elif defined(LFL_APPLE)
  char rpath[1024];
  CFBundleRef mainBundle = CFBundleGetMainBundle();
  CFURLRef respath = CFBundleCopyResourcesDirectoryURL(mainBundle);
  CFURLGetFileSystemRepresentation(respath, true, MakeUnsigned(rpath), sizeof(rpath));
  CFRelease(respath);
  if (PrefixMatch(rpath, startdir+"/")) assetdir = StrCat(rpath + startdir.size()+1, "/assets/");
  else assetdir = StrCat(rpath, "/assets/"); 
#else
  assetdir = StrCat(bindir, "assets/"); 
#endif

#ifdef LFL_WINDOWS
  bindir = progname.substr(0, DirNameLen(progname, true));

  { /* winsock startup */
    WSADATA wsadata;
    WSAStartup(MAKEWORD(2,2), &wsadata);
  }

  string console_title = StrCat(name, " console");
  if (argc > 1) OpenSystemConsole(console_title.c_str());

#else
  pid = getpid();
  bindir = LocalFile::JoinPath(startdir, progname.substr(0, DirNameLen(progname, true)));

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

  if (Singleton<FlagMap>::Get()->getopt(argc, argv, source_filename) < 0) return -1;
  if (!FLAGS_rand_seed) FLAGS_rand_seed = fnv32(&pid, sizeof(int), time(0));
  unsigned init_rand_seed = FLAGS_rand_seed;
  srand(init_rand_seed);

  ThreadLocalStorage::Init();
  Singleton<NullAllocator>::Get();
  Singleton<MallocAllocator>::Get();

#ifdef LFL_WINDOWS
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
#elif defined(LFL_WINDOWS)
    char path[MAX_PATH];
    if (!SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL|CSIDL_FLAG_CREATE, NULL, 0, path))) return -1;
    dldir = StrCat(path, "/");
#endif
  }

#ifdef LFL_DEBUG
  if (FLAGS_logfile.empty() && !FLAGS_logfile_.override) FLAGS_logfile = StrCat(dldir, name, ".txt");
#endif
  if (!FLAGS_logfile.empty()) {
    logfile = fopen(FLAGS_logfile.c_str(), "a");
    if (logfile) SystemNetwork::SetSocketCloseOnExec(fileno(logfile), 1);
  }

  INFO("startdir = ", startdir);
  INFO("assetdir = ", assetdir);
  INFO("dldir = ", dldir);
  INFO("rand_seed = ", init_rand_seed);

#ifndef LFL_WINDOWS
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
#ifdef LFL_APPLE
    rl.rlim_cur = rl.rlim_max = OPEN_MAX;
#else
    rl.rlim_cur = rl.rlim_max; // 999999
#endif
    INFO("setrlimit(RLIMIT_NOFILE, ", rl.rlim_cur, ")");
    if (setrlimit(RLIMIT_NOFILE, &rl) == -1) return ERRORv(-1, "files setrlimit ", strerror(errno));
  }
#endif // LFL_MOBILE
#endif // LFL_WINDOWS

  if (FLAGS_daemonize) {
    Daemonize("", progname.c_str());
    SetLFAppMainThread();
  }

  if (FLAGS_lfapp_video && FLAGS_default_font.empty())
    fonts->DefaultFontEngine()->SetDefault();

  return 0;
}

int Application::Init() {
  if (LoadModule((framework = unique_ptr<Module>(CreateFrameworkModule())).get()))
    return ERRORv(-1, "platform init failed");

  thread_pool.Open(X_or_1(FLAGS_threadpool_size));
  if (FLAGS_threadpool_size) thread_pool.Start();

  if (FLAGS_lfapp_video) {
    if (!screen->gd) screen->gd = CreateGraphicsDevice(opengles_version).release();
    shaders = make_unique<Shaders>();
    screen->gd->Init(screen->Box());
  } else { windows[screen->id.v] = screen; }

#ifdef LFL_WINDOWS
  if (FLAGS_lfapp_video && splash_color) {
    screen->gd->ClearColor(*splash_color);
    screen->gd->Clear();
    screen->gd->Flush();
    Video::Swap();
    screen->gd->ClearColor(screen->gd->clear_color);
  }
#endif

  if (FLAGS_lfapp_audio) {
    if (LoadModule((audio = make_unique<Audio>()).get())) return ERRORv(-1, "audio init failed");
  }
  else { FLAGS_chans_in=FLAGS_chans_out=1; }

  if (FLAGS_lfapp_audio || FLAGS_lfapp_video) {
    if ((asset_loader = make_unique<AssetLoader>())->Init()) return ERRORv(-1, "asset loader init failed");
  }

  if (FLAGS_lfapp_video) fonts->LoadDefaultFonts();
  screen->default_font = FontRef(FontDesc::Default(), false);

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
  if (!FLAGS_threadpool_size && thread_pool.worker.size()) thread_pool.worker[0].queue->HandleMessages();
  else thread_pool.Stop();
  message_queue.HandleMessages();
  if (fonts) fonts.reset();
  if (shaders) shaders.reset();
  vector<Window*> close_list;
  for (auto &i : windows) close_list.push_back(i.second);
  for (auto &i : close_list) CloseWindow(i);
  if (exit_cb) exit_cb();
  if (cuda) cuda->Free();
  for (auto &m : modules) m->Free();
  scheduler.Free();
  if (logfile) fclose(logfile);
#ifdef LFL_WINDOWS
  if (FLAGS_open_console) PromptFGets("Press [enter] to continue...");
#endif
}

/* Window */

Window::Window() : caption(app->name), fps(128) {
  id = gl = surface = glew_context = impl = user1 = user2 = user3 = typed_ptr{0, nullptr};
  started = minimized = cursor_grabbed = frame_init = animating = 0;
  target_fps = FLAGS_target_fps;
  multitouch_keyboard_x = .93; 
  cam = make_unique<Entity>(v3(5.54, 1.70, 4.39), v3(-.51, -.03, -.49), v3(-.03, 1, -.03));
  SetSize(point(640, 480));
}

Window::~Window() {
  dialogs.clear();
  my_gui.clear();
  if (console) console->WriteHistory(LFAppDownloadDir(), "console", "");
  console.reset();
  delete gd;
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
  console->Write(StrCat(caption, " started"));
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
  for (auto i = dialogs.begin(), e = dialogs.end(); i != e; ++i) (*i)->Draw();
  if (console) console->Draw();
  if (FLAGS_draw_grid) {
    Color c(.7, .7, .7);
    glIntersect(mouse.x, mouse.y, &c);
    default_font->Draw(StrCat("draw_grid ", mouse.x, " , ", mouse.y), point(0,0));
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
  gd->DrawMode(gd->default_draw_mode);
  for (auto g = gui.begin(); g != gui.end(); ++g) (*g)->Layout();
  if (reshaped_cb) reshaped_cb();
}

void Window::ResetGL() {
  gd->Init(Box());
  for (auto &g : gui    ) g->ResetGL();
  for (auto &g : dialogs) g->ResetGL();
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
      gd->GetIntegerv(GraphicsDevice::FramebufferBinding, &gd->default_framebuffer);
      INFO("default_framebuffer = ", gd->default_framebuffer);
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
    Video::Swap();
  }
  return ret;
}

void Window::RenderToFrameBuffer(FrameBuffer *fb) {
  int dm = gd->draw_mode;
  fb->Attach();
  // gd->ViewPort(Box(fb->tex.width, fb->tex.height));
  gd->DrawMode(gd->default_draw_mode);
  gd->Clear();
  frame_cb(0, 0, 0);
  fb->Release();
  gd->RestoreViewport(dm);
}

/* FrameScheduler */

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
    ret = DoWait();
    if (synchronize_waits) {
      frame_mutex.lock();
      wait_mutex.unlock();
    }
  }
  return ret;
}

void FrameScheduler::UpdateTargetFPS(int fps) {
  screen->target_fps = fps;
  if (monolithic_frame) {
    int next_target_fps = 0;
    for (const auto &w : app->windows) Max(&next_target_fps, w.second->target_fps);
    FLAGS_target_fps = next_target_fps;
  }
  CHECK(screen->id.v);
  UpdateWindowTargetFPS(screen);
}

void FrameScheduler::SetAnimating(bool is_animating) {
  screen->animating = is_animating;
  int target_fps = is_animating ? FLAGS_peak_fps : 0;
  if (target_fps != screen->target_fps) {
    UpdateTargetFPS(target_fps);
    Wakeup(0);
  }
}

}; // namespace LFL
