/*
 * $Id: lfapp.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef LFL_LFAPP_LFAPP_H__
#define LFL_LFAPP_LFAPP_H__

#include <sstream>
#include <typeinfo>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <list>
#include <queue>
#include <deque>
#include <iterator>
#include <algorithm>
#include <memory>
#include <numeric>
#include <limits>
#include <cwctype>
#define LFL_STL_NAMESPACE std

#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>
#include <type_traits>
#define LFL_STL11_NAMESPACE std

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <windows.h>
#include <process.h>
#include <malloc.h>
#include <io.h>
#include <sstream>
#include <typeinfo>
#define _USE_MATH_DEFINES
typedef SOCKET Socket;
#else /* _WIN32 */
#include <unistd.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/socket.h>
typedef int Socket;
#endif
inline int SystemBind(Socket s, const sockaddr *sa, int l) { return bind(s, sa, l); }

using LFL_STL_NAMESPACE::min;
using LFL_STL_NAMESPACE::max;
using LFL_STL_NAMESPACE::swap;
using LFL_STL_NAMESPACE::pair;
using LFL_STL_NAMESPACE::vector;
using LFL_STL_NAMESPACE::string;
using LFL_STL_NAMESPACE::basic_string;
using LFL_STL_NAMESPACE::map;
using LFL_STL_NAMESPACE::set;
using LFL_STL_NAMESPACE::list;
using LFL_STL_NAMESPACE::deque;
using LFL_STL_NAMESPACE::inserter;
using LFL_STL_NAMESPACE::binary_search;
using LFL_STL_NAMESPACE::sort;
using LFL_STL_NAMESPACE::unique;
using LFL_STL_NAMESPACE::reverse;
using LFL_STL_NAMESPACE::equal_to;
using LFL_STL_NAMESPACE::lower_bound;
using LFL_STL_NAMESPACE::make_pair;
using LFL_STL_NAMESPACE::set_difference;
using LFL_STL_NAMESPACE::numeric_limits;
using LFL_STL11_NAMESPACE::unordered_map;
using LFL_STL11_NAMESPACE::unordered_set;
using LFL_STL11_NAMESPACE::shared_ptr;
using LFL_STL11_NAMESPACE::unique_ptr;
using LFL_STL11_NAMESPACE::tuple;
using LFL_STL11_NAMESPACE::array;
using LFL_STL11_NAMESPACE::move;
using LFL_STL11_NAMESPACE::bind;
using LFL_STL11_NAMESPACE::function;
using LFL_STL11_NAMESPACE::placeholders::_1;
using LFL_STL11_NAMESPACE::placeholders::_2;
using LFL_STL11_NAMESPACE::placeholders::_3;
using LFL_STL11_NAMESPACE::placeholders::_4;
using LFL_STL11_NAMESPACE::placeholders::_5;
using LFL_STL11_NAMESPACE::placeholders::_6;
using LFL_STL11_NAMESPACE::placeholders::_7;
using LFL_STL11_NAMESPACE::placeholders::_8;
using LFL_STL11_NAMESPACE::mutex;
using LFL_STL11_NAMESPACE::lock_guard;
using LFL_STL11_NAMESPACE::unique_lock;
using LFL_STL11_NAMESPACE::condition_variable;
using LFL_STL11_NAMESPACE::chrono::hours;
using LFL_STL11_NAMESPACE::chrono::minutes;
using LFL_STL11_NAMESPACE::chrono::seconds;
using LFL_STL11_NAMESPACE::chrono::milliseconds;
using LFL_STL11_NAMESPACE::chrono::microseconds;
using LFL_STL11_NAMESPACE::chrono::duration;
using LFL_STL11_NAMESPACE::chrono::duration_cast;
using LFL_STL11_NAMESPACE::chrono::system_clock;
using LFL_STL11_NAMESPACE::chrono::steady_clock;
using LFL_STL11_NAMESPACE::chrono::high_resolution_clock;
using LFL_STL11_NAMESPACE::enable_if;
using LFL_STL11_NAMESPACE::is_integral;
using LFL_STL11_NAMESPACE::is_floating_point;
using LFL_STL11_NAMESPACE::make_unsigned;
using LFL_STL11_NAMESPACE::make_shared;
#define tuple_get LFL_STL11_NAMESPACE::get

#include <cfloat>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#ifdef _WIN32
#include <float.h>
#include <direct.h>
#undef ERROR
#undef CALLBACK
#define getcwd _getcwd
#define chdir _chdir
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define snprintf _snprintf
#define S_IFDIR _S_IFDIR
typedef int socklen_t;
extern char *optarg;
extern int optind;
#endif

#ifdef __linux__
#include <arpa/inet.h>
#endif

#ifdef LFL_ANDROID
#include <sys/endian.h>
#include "jni/lfjni.h"
#endif

#ifdef LFL_QT
#include <QtOpenGL>
#endif

#define  INFO(...) ((::LFApp::Log::Info  <= ::LFL::FLAGS_loglevel) ? ::LFL::Log(::LFApp::Log::Info,  __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)) : void())
#define DEBUG(...) ((::LFApp::Log::Debug <= ::LFL::FLAGS_loglevel) ? ::LFL::Log(::LFApp::Log::Debug, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)) : void())
#define ERROR(...) ((::LFApp::Log::Error <= ::LFL::FLAGS_loglevel) ? ::LFL::Log(::LFApp::Log::Error, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)) : void())
#define FATAL(...) { ::LFL::Log(::LFApp::Log::Fatal, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)); throw(0); }
#define ERRORv(v, ...) LFL::Log(::LFApp::Log::Error, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)), v

#define ONCE(x) { static bool once=0; if (!once && (once=1)) { x; } }
#define EVERY_N(x, y) { static int every_N=0; if (every_N++ % (x) == 0) { y; } }
#define EX_LT(x, a)    if (!((x) <  (a)))  INFO((x), " < ",  (a), ": EX_LT(",    #x, ", ", #a, ")"); 
#define EX_GT(x, a)    if (!((x) >  (a)))  INFO((x), " > ",  (a), ": EX_GT(",    #x, ", ", #a, ")"); 
#define EX_GE(x, a)    if (!((x) >= (a)))  INFO((x), " >= ", (a), ": EX_GE(",    #x, ", ", #a, ")"); 
#define EX_LE(x, a)    if (!((x) <= (a)))  INFO((x), " <= ", (a), ": EX_LE(",    #x, ", ", #a, ")"); 
#define EX_EQ(x, a)    if (!((x) == (a)))  INFO((x), " == ", (a), ": EX_EQ(",    #x, ", ", #a, ")");
#define EX_NE(x, a)    if (!((x) != (a)))  INFO((x), " != ", (a), ": EX_NE(",    #x, ", ", #a, ")");
#define CHECK_LT(x, a) if (!((x) <  (a))) FATAL((x), " < ",  (a), ": CHECK_LT(", #x, ", ", #a, ")"); 
#define CHECK_GT(x, a) if (!((x) >  (a))) FATAL((x), " > ",  (a), ": CHECK_GT(", #x, ", ", #a, ")"); 
#define CHECK_GE(x, a) if (!((x) >= (a))) FATAL((x), " >= ", (a), ": CHECK_GE(", #x, ", ", #a, ")"); 
#define CHECK_LE(x, a) if (!((x) <= (a))) FATAL((x), " <= ", (a), ": CHECK_LE(", #x, ", ", #a, ")"); 
#define CHECK_EQ(x, a) if (!((x) == (a))) FATAL((x), " == ", (a), ": CHECK_EQ(", #x, ", ", #a, ")");
#define CHECK_NE(x, a) if (!((x) != (a))) FATAL((x), " != ", (a), ": CHECK_NE(", #x, ", ", #a, ")");
#define CHECK_RANGE(x, y, z) { CHECK_GE(x, y); CHECK_LT(x, z); }
#define CHECK(x) if (!(x)) FATAL(#x)

#define DEFINE_FLAG(name, type, initial, description) \
  type FLAGS_ ## name = initial; \
  FlagOfType<type> FLAGS_ ## name ## _(#name, description, __FILE__, __LINE__, &FLAGS_ ## name)

#define DEFINE_int(name, initial, description) DEFINE_FLAG(name, int, initial, description)
#define DEFINE_bool(name, initial, description) DEFINE_FLAG(name, bool, initial, description)
#define DEFINE_float(name, initial, description) DEFINE_FLAG(name, float, initial, description)
#define DEFINE_double(name, initial, description) DEFINE_FLAG(name, double, initial, description)
#define DEFINE_string(name, initial, description) DEFINE_FLAG(name, string, initial, description)

namespace LFL {
using LFL_STL11_NAMESPACE::isinf;
using LFL_STL11_NAMESPACE::isnan;
typedef void* Void;
typedef lock_guard<mutex> ScopedMutex;
typedef function<void()> Callback;
typedef function<void(const string&)> StringCB;
typedef function<bool(const string&, const string&,       string*)> LoadPasswordCB;
typedef function<void(const string&, const string&, const string&)> SavePasswordCB;
typedef tuple<string, string, string> MenuItem;
template <class X> struct Singleton { static X *Get() { static X instance; return &instance; } };
void Log(int level, const char *file, int line, const string &m);
}; // namespace LFL

#include "lfapp/lfexport.h"
#include "lfapp/string.h"
#include "lfapp/time.h"

namespace LFL {
struct Allocator {
  virtual ~Allocator() {}
  virtual const char *Name() = 0;
  virtual void *Malloc(int size) = 0;
  virtual void *Realloc(void *p, int size) = 0;
  virtual void Free(void *p) = 0;
  virtual void Reset();
  static Allocator *Default();
#define AllocatorNew(allocator, type, constructor_args) (new((allocator)->Malloc(sizeof type)) type constructor_args)
};

struct NullAlloc : public Allocator {
  const char *Name() { return "NullAlloc"; }
  void *Malloc(int size) { return 0; }
  void *Realloc(void *p, int size) { return 0; }
  void Free(void *p) {}
};

struct ThreadLocalStorage {
  Allocator *alloc=0;
  std::default_random_engine rand_eng;
  virtual ~ThreadLocalStorage() { delete alloc; }
  ThreadLocalStorage() : rand_eng(std::random_device{}()) {}
  static void Init();
  static void Free();
  static void ThreadInit();
  static void ThreadFree();
  static ThreadLocalStorage *Get();
  static Allocator *GetAllocator(bool reset_allocator=true);
};

struct Module {
  virtual int Init ()         { return 0; }
  virtual int Start()         { return 0; }
  virtual int Frame(unsigned) { return 0; }
  virtual int Free ()         { return 0; }
};
}; // namespace LFL

#include "lfapp/math.h"
#include "lfapp/file.h"
#include "lfapp/lftypes.h"

namespace LFL {
bool Running();
bool MainThread();
bool MainProcess();
void RunInMainThread(Callback *cb);
void RunInNetworkThread(const Callback &cb);
void DefaultLFAppWindowClosedCB(Window *);
double FPS();
double CamFPS();
void PressAnyKey();
bool FGets(char *buf, int size);
bool NBFGets(FILE*, char *buf, int size, int timeout=0);
int NBRead(Socket fd, char *buf, int size, int timeout=0);
int NBRead(Socket fd, string *buf, int timeout=0);
string NBRead(Socket fd, int size, int timeout=0);
bool NBReadable(Socket fd, int timeout=0);

struct MallocAlloc : public Allocator {
  const char *Name() { return "MallocAlloc"; }
  void *Malloc(int size);
  void *Realloc(void *p, int size);
  void Free(void *p);
};

struct NewAlloc : public Allocator {
  const char *Name() { return "NewAlloc"; }
  void *Malloc(int size) { return new char[size]; }
  void *Realloc(void *p, int size) { return !p ? Malloc(size) : 0; }
  void Free(void *p) { delete [] (char *)p; }
};

template <int S> struct FixedAlloc : public Allocator {
  const char *Name() { return "FixedAlloc"; }
  static const int size = S;
  char buf[S];
  int len=0;
  virtual void Reset() { len=0; }
  virtual void *Malloc(int n) { CHECK_LE(len + n, S); char *ret = &buf[len]; len += NextMultipleOf16(n); return ret; }
  virtual void *Realloc(void *p, int n) { CHECK_EQ(nullptr, p); return this->Malloc(n); }
  virtual void Free(void *p) {}
};

struct MMapAlloc : public Allocator {
#ifdef _WIN32
  HANDLE file, map; void *addr; long long size;
  MMapAlloc(HANDLE File, HANDLE Map, void *Addr, int Size) : file(File), map(Map), addr(Addr), size(Size) {}
#else
  void *addr; long long size;
  MMapAlloc(void *Addr, long long Size) : addr(Addr), size(Size) {}
#endif
  virtual ~MMapAlloc();
  static MMapAlloc *Open(const char *fn, const bool logerror=true, const bool readonly=true, long long size=0);

  const char *Name() { return "MMapAlloc"; }
  void *Malloc(int size) { return 0; }
  void *Realloc(void *p, int size) { return 0; }
  void Free(void *p) { delete this; }
};

struct BlockChainAlloc : public Allocator {
  const char *Name() { return "BlockChainAlloc"; }
  struct Block { string buf; int len=0; Block(int size=0) : buf(size, 0) {} };
  vector<Block> blocks;
  int block_size, cur_block_ind;
  BlockChainAlloc(int s=1024*1024) : block_size(s), cur_block_ind(-1) {}
  void Reset() { for (auto &b : blocks) b.len = 0; cur_block_ind = blocks.size() ? 0 : -1; }
  void *Realloc(void *p, int n) { CHECK_EQ(nullptr, p); return this->Malloc(n); }
  void *Malloc(int n);
  void Free(void *p) {}
};

struct StringAlloc {
  string buf;
  void Reset() { buf.clear(); }
  int Alloc(int bytes) { int ret=buf.size(); buf.resize(ret + bytes); return ret; }
};

struct Flag {
  const char *name, *desc, *file;
  int line;
  bool override;
  Flag(const char *N, const char *D, const char *F, int L) : name(N), desc(D), file(F), line(L), override(0) {}
  virtual ~Flag() {}

  string GetString() const;
  virtual string Get() const = 0;
  virtual bool IsBool() const = 0;
  virtual void Update(const char *text) = 0;
};

struct FlagMap {
  typedef map<string, Flag*> AllFlags;
  const char *optarg=0;
  int optind=0;
  AllFlags flagmap;
  bool dirty=0;
  FlagMap() {}

  int getopt(int argc, const char **argv, const char *source_filename);
  void Add(Flag *f) { flagmap[f->name] = f; }
  bool Set(const string &k, const string &v);
  bool IsBool(const string &k) const;
  string Get(const string &k) const;
  string Match(const string &k, const char *source_filename=0) const;
  void Print(const char *source_filename=0) const;
};

template <class X> struct FlagOfType : public Flag {
  X *v;
  virtual ~FlagOfType() {}
  FlagOfType(const char *N, const char *D, const char *F, int L, X *V)
    : Flag(N, D, F, L), v(V) { Singleton<FlagMap>::Get()->Add(this); } 

  string Get() const { return ToString(*v); }
  bool IsBool() const { return TypeId<X>() == TypeId<bool>(); }
  void Update(const char *text) { if (text) *v = Scannable::Scan(*v, text); }
};

struct Thread {
  typedef unsigned long long id_t;
  id_t id=0;
  Callback cb;
  mutex start_mutex;
  unique_ptr<std::thread> impl;
  Thread(const Callback &CB=Callback()) : cb(CB) {}
  void Open(const Callback &CB) { cb=CB; }
  void Wait() { if (impl) impl->join(); }
  void Start() {
    ScopedMutex sm(start_mutex);
    impl = move(unique_ptr<std::thread>(new std::thread(bind(&Thread::ThreadProc, this))));
    id = std::hash<std::thread::id>()(impl->get_id());
  }
  void ThreadProc() {
    { ScopedMutex sm(start_mutex); }
    INFOf("Started thread(%llx)", id);
    ThreadLocalStorage::ThreadInit();
    cb();
    ThreadLocalStorage::ThreadFree();
  }
  static id_t GetId() { return std::hash<std::thread::id>()(std::this_thread::get_id()); }
};

struct WorkerThread {
  unique_ptr<CallbackQueue> queue;
  unique_ptr<Thread> thread;
  WorkerThread() : queue(new CallbackQueue()) {}
  void Init(const Callback &main_cb) { thread = unique_ptr<Thread>(new Thread(main_cb)); }
};

struct ThreadPool {
  vector<WorkerThread> worker;
  int round_robin_next=0;

  void Open(int num) {
    CHECK(worker.empty());
    worker.resize(num);
    for (auto &w : worker) w.Init(bind(&CallbackQueue::HandleMessagesLoop, w.queue.get()));
  }
  void Start() { for (auto &w : worker) w.thread->Start(); }
  void Stop()  { for (auto &w : worker) w.thread->Wait(); }
  void Write(Callback *cb) {
    worker[round_robin_next].queue->Write(cb);
    round_robin_next = (round_robin_next + 1) % worker.size();
  }
};

struct Timer {
  Time begin;
  Timer() { Reset(); }
  Time Reset() { Time last_begin=begin; begin=Now(); return last_begin; }
  Time GetTime() const { return Now() - begin; }
  Time GetTime(bool do_reset) {
    if (!do_reset) return GetTime();
    Time last_begin = Reset();
    return max(Time(0), begin - last_begin);
  }
};

struct PerformanceTimers {
  struct Accumulator {
    string name;
    Time time;
    Accumulator(const string &n="") : name(n), time(0) {}
  };
  vector<Accumulator> timers;
  Timer cur_timer;
  int cur_timer_id;
  PerformanceTimers() { cur_timer_id = Create("Default"); }

  int Create(const string &n) { timers.push_back(Accumulator(n)); return timers.size()-1; }
  void AccumulateTo(int timer_id) { timers[cur_timer_id].time += cur_timer.GetTime(true); cur_timer_id = timer_id; }
  string DebugString() const { string v; for (auto &t : timers) StrAppend(&v, t.name, " ", t.time.count() / 1000.0, "\n"); return v; }
};
}; // namespace LFL

#include "lfapp/audio.h"
#include "lfapp/video.h"
#include "lfapp/font.h"
#include "lfapp/scene.h"
#include "lfapp/assets.h"
#include "lfapp/input.h"
#include "lfapp/shell.h"
#include "lfapp/network.h"
#include "lfapp/camera.h"

namespace LFL {
::std::ostream& operator<<(::std::ostream& os, const point &x);
::std::ostream& operator<<(::std::ostream& os, const Box   &x);

struct RateLimiter {
  int *target_hz;
  float avgframe;
  Timer timer;
  RollingAvg<unsigned> sleep_bias;
  RateLimiter(int *HZ) : target_hz(HZ), avgframe(0), sleep_bias(32) {}
  void Limit() {
    Time since = timer.GetTime(true), targetframe(1000 / *target_hz);
    Time sleep = max(Time(0), targetframe - since - FMilliseconds(sleep_bias.Avg()));
    if (sleep != Time(0)) { MSleep(sleep.count()); sleep_bias.Add((timer.GetTime(true) - sleep).count()); }
  }
};

struct FrameScheduler {
  RateLimiter maxfps;
  mutex frame_mutex, wait_mutex;
  SocketWakeupThread wakeup_thread;
  SelectSocketSet wait_forever_sockets;
  Socket system_event_socket = -1, wait_forever_wakeup_socket = -1;
  bool rate_limit = 1, wait_forever = 1, wait_forever_thread = 1, synchronize_waits = 1, monolithic_frame = 1;
  FrameScheduler();

  void Init();
  void Free();
  void Start();
  bool FrameWait();
  void FrameDone();
  void Wakeup(void*);
  bool WakeupIn(void*, Time interval, bool force=0);
  void ClearWakeupIn();
  void UpdateTargetFPS(int fps);
  void SetAnimating(bool);
  void AddWaitForeverMouse();
  void DelWaitForeverMouse();
  void AddWaitForeverKeyboard();
  void DelWaitForeverKeyboard();
  void AddWaitForeverSocket(Socket fd, int flag, void *val=0);
  void DelWaitForeverSocket(Socket fd);
};

struct BrowserInterface {
  virtual void Draw(const Box&) = 0;
  virtual void Open(const string &url) = 0;
  virtual void Navigate(const string &url) { Open(url); }
  virtual Asset *OpenImage(const string &url) { return 0; }
  virtual void OpenStyleImport(const string &url) {}
  virtual void MouseMoved(int x, int y) = 0;
  virtual void MouseButton(int b, bool d, int x, int y) = 0;
  virtual void MouseWheel(int xs, int ys) = 0;
  virtual void KeyEvent(int key, bool down) = 0;
  virtual void BackButton() = 0;
  virtual void ForwardButton() = 0;
  virtual void RefreshButton() = 0;
  virtual string GetURL() const = 0;
};

struct JSContext {
  virtual ~JSContext() {}
  virtual string Execute(const string &s) = 0;
  static JSContext *Create(Console *js_console=0, LFL::DOM::Node *document=0);
};

struct LuaContext {
  virtual ~LuaContext() {}
  virtual string Execute(const string &s) = 0;
  static LuaContext *Create();
};

struct CUDA : public Module { int Init(); };

struct Application : public ::LFApp, public Module {
  string name, progname, logfilename, startdir, bindir, assetdir, dldir;
  int pid=0;
  FILE *logfile=0;
  mutex log_mutex;
  Time time_started;
  Timer frame_time;
  ThreadPool thread_pool;
  CallbackQueue message_queue;
  FrameScheduler scheduler;
  NetworkThread *network_thread=0;
  ProcessAPIClient *render_process=0;
  ProcessAPIServer *main_process=0;
  Window::Map windows;
  Callback reshaped_cb, create_win_f;
  function<void(Window*)> window_init_cb, window_closed_cb;
  unordered_map<string, StringPiece> asset_cache;
  CategoricalVariable<int> tex_mode, grab_mode, fill_mode;
  const Color *splash_color = &Color::black;
  Shell shell;

  vector<Module*> modules;
  Audio *audio=0;
  Video *video=0;
  Input *input=0;
  Assets *assets=0;
  Network *network=0;
  Camera *camera=0;
  CUDA *cuda=0;

  Application() : create_win_f(bind(&Application::CreateNewWindow, this, function<void(Window*)>())),
  window_closed_cb(DefaultLFAppWindowClosedCB), tex_mode(2, 1, 0), grab_mode(2, 0, 1),
  fill_mode(3, GraphicsDevice::Fill, GraphicsDevice::Line, GraphicsDevice::Point), shell(0, 0, 0)
  { run=1; initialized=0; main_thread_id=0; frames_ran=0; }

  void Log(int level, const char *file, int line, const string &message);
  int LoadModule(Module *M) { modules.push_back(M); return M->Init(); }
  Window *GetWindow(void *id) { return FindOrNull(windows, id); }
  bool CreateWindow(Window *W);
  void CloseWindow(Window *W);
  void MakeCurrentWindow(Window *W);
  void CreateNewWindow(const Window::StartCB &start_cb = Window::StartCB());
  void StartNewWindow(Window *new_window);
  NetworkThread *CreateNetworkThread(bool detach_existing_module, bool start);

  int Create(int argc, const char **argv, const char *source_filename, void (*create_cb)()=0);
  int Init();
  int Start();
  int HandleEvents(unsigned clicks);
  int EventDrivenFrame(bool handle_events);
  int TimerDrivenFrame(bool got_wakeup);
  int Main();
  int MainLoop();
  int Free();
  int Exiting();
  void ResetGL();

  void GrabMouseFocus();
  void ReleaseMouseFocus();
  string GetClipboardText();
  void SetClipboardText(const string &s);
  void OpenSystemBrowser(const string &url);

  void AddNativeMenu(const string &title, const vector<MenuItem> &items);
  void AddNativeEditMenu();
  void LaunchNativeMenu(const string &title);
  void LaunchNativeContextMenu(const vector<MenuItem> &items);
  void LaunchNativeFontChooser(const FontDesc &cur_font, const string &choose_cmd);
  void LaunchNativeFileChooser(bool files, bool dirs, bool multi, const string &choose_cmd);

  /// AddToolbar item values with prefix "toggle" stay depressed
  void AddToolbar(const vector<pair<string, string>>&items);
  void ToggleToolbarButton(const string &n);

  void OpenTouchKeyboard();
  void CloseTouchKeyboard();
  void CloseTouchKeyboardAfterReturn(bool);
  Box GetTouchKeyboardBox();

  int SetExtraScale(bool on); /// e.g. Retina display
  int SetMultisample(bool on);

  bool LoadPassword(const string &host, const string &user,       string *pw_out);
  void SavePassword(const string &host, const string &user, const string &pw);

  void ShowAds();
  void HideAds();

  int GetVolume();
  int GetMaxVolume();
  void SetVolume(int v);
  void PlaySoundEffect(SoundAsset*);
  void PlayBackgroundMusic(SoundAsset*);

  static void Daemonize(const char *dir="");
  static void Daemonize(FILE *fout, FILE *ferr);
  static void *GetSymbol(const string &n);
  static StringPiece LoadResource(int id);
};
extern Application *app;

#ifdef LFL_WININPUT
struct WinApp {
  HINSTANCE hInst = 0;
  int nCmdShow = 0;
  void Setup(HINSTANCE hI, int nCS) { hInst = hI; nCmdShow = nCS; }
  void CreateClass();
  int MessageLoop();
  static LRESULT APIENTRY WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
};
struct WinWindow {
  bool menubar = 0, frame_on_keyboard_input = 0, frame_on_mouse_input = 0;
  point prev_mouse_pos, resize_increment;
  int start_msg_id = WM_USER + 100;
  HMENU menu = 0, context_menu = 0;
  vector<string> menu_cmds;
  bool RestrictResize(int m, RECT*);
};
#endif

}; // namespace LFL

#if defined(LFL_QT)
#define main LFLQTMain
#elif defined(LFL_WXWIDGETS)
#define main LFLWxWidgetsMain
#elif defined(LFL_IPHONE)
#define main iPhoneMain
#elif defined(LFL_OSXVIDEO)
#define main OSXMain
#endif

extern "C" int main(int argc, const char **argv);

#endif // LFL_LFAPP_LFAPP_H__
