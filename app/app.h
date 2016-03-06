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

#ifndef LFL_CORE_APP_APP_H__
#define LFL_CORE_APP_APP_H__

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

#ifdef LFL_WINDOWS
#define NOMINMAX
#define _USE_MATH_DEFINES
#include <winsock2.h>
#include <windows.h>
#include <process.h>
#include <malloc.h>
#include <io.h>
#include <sstream>
#include <typeinfo>
typedef SOCKET Socket;
#else
#include <unistd.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/socket.h>
typedef int Socket;
#endif

#include <cfloat>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#ifdef LFL_WINDOWS
#include <float.h>
#include <direct.h>
typedef int socklen_t;
extern char *optarg;
extern int optind;
#undef ERROR
#undef CALLBACK
#define S_IFDIR _S_IFDIR
#define getcwd _getcwd
#define chdir _chdir
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define snprintf _snprintf
#endif

#ifdef LFL_LINUX
#include <arpa/inet.h>
#endif

#ifdef LFL_ANDROID
#include <sys/endian.h>
#endif

#define  INFO(...) ((::LFApp::Log::Info  <= ::LFL::FLAGS_loglevel) ? ::LFL::Log(::LFApp::Log::Info,  __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)) : void())
#define ERROR(...) ((::LFApp::Log::Error <= ::LFL::FLAGS_loglevel) ? ::LFL::Log(::LFApp::Log::Error, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)) : void())
#define FATAL(...) { ::LFL::Log(::LFApp::Log::Fatal, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)); throw(0); }
#define ERRORv(v, ...) LFL::Log(::LFApp::Log::Error, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)), v
#ifdef LFL_DEBUG
#define DEBUG(...) ((::LFApp::Log::Debug <= ::LFL::FLAGS_loglevel) ? ::LFL::Log(::LFApp::Log::Debug, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)) : void())
#else
#define DEBUG(...)
#endif

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
#define DEFINE_unsigned(name, initial, description) DEFINE_FLAG(name, unsigned, initial, description)

namespace LFL {
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
using LFL_STL11_NAMESPACE::size_t;
using LFL_STL11_NAMESPACE::ptrdiff_t;
using LFL_STL11_NAMESPACE::int8_t;
using LFL_STL11_NAMESPACE::int16_t;
using LFL_STL11_NAMESPACE::int32_t;
using LFL_STL11_NAMESPACE::int64_t;
using LFL_STL11_NAMESPACE::uint8_t;
using LFL_STL11_NAMESPACE::uint16_t;
using LFL_STL11_NAMESPACE::uint32_t;
using LFL_STL11_NAMESPACE::uint64_t;
using LFL_STL11_NAMESPACE::unordered_map;
using LFL_STL11_NAMESPACE::unordered_set;
using LFL_STL11_NAMESPACE::shared_ptr;
using LFL_STL11_NAMESPACE::unique_ptr;
using LFL_STL11_NAMESPACE::weak_ptr;
using LFL_STL11_NAMESPACE::tuple;
using LFL_STL11_NAMESPACE::array;
using LFL_STL11_NAMESPACE::move;
using LFL_STL11_NAMESPACE::bind;
using LFL_STL11_NAMESPACE::forward;
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
using LFL_STL11_NAMESPACE::is_signed;
using LFL_STL11_NAMESPACE::is_pod;
using LFL_STL11_NAMESPACE::make_unsigned;
using LFL_STL11_NAMESPACE::make_signed;
using LFL_STL11_NAMESPACE::make_shared;
using LFL_STL11_NAMESPACE::make_tuple;
using LFL_STL11_NAMESPACE::isinf;
using LFL_STL11_NAMESPACE::isnan;
#define tuple_get LFL_STL11_NAMESPACE::get

typedef lock_guard<mutex> ScopedMutex;
typedef function<void()> Callback;
typedef function<void(const string&)> StringCB;
typedef tuple<string, string, string> MenuItem;
template <class X> struct Singleton { static X *Get() { static X instance; return &instance; } };
void Log(int level, const char *file, int line, const string &m);
}; // namespace LFL

#include "core/app/export.h"
#include "core/app/types/string.h"
#include "core/app/types/time.h"

namespace LFL {
extern Window *screen;
extern Application *app;

struct Allocator {
  virtual ~Allocator() {}
  virtual void *Malloc(int size) = 0;
  virtual void *Realloc(void *p, int size) = 0;
  virtual void Free(void *p) = 0;
  virtual void Reset();
  template <class X, class... Args> X* New(Args&&... args) { return new(Malloc(sizeof(X))) X(forward<Args>(args)...); }
  static Allocator *Default();
};

struct NullAllocator : public Allocator {
  const char *Name() { return "NullAllocator"; }
  void *Malloc(int size) { return 0; }
  void *Realloc(void *p, int size) { return 0; }
  void Free(void *p) {}
};

struct ThreadLocalStorage {
  unique_ptr<Allocator> alloc;
  std::default_random_engine rand_eng;
  ThreadLocalStorage() : rand_eng(std::random_device{}()) {}
  static void Init();
  static void Free();
  static void ThreadInit();
  static void ThreadFree();
  static ThreadLocalStorage *Get();
  static Allocator *GetAllocator(bool reset_allocator=true);
};

struct Module {
  virtual ~Module() {}
  virtual int Init ()         { return 0; }
  virtual int Start()         { return 0; }
  virtual int Frame(unsigned) { return 0; }
  virtual int Free ()         { return 0; }
};
}; // namespace LFL

#include "core/app/math.h"
#include "core/app/file.h"
#include "core/app/types/types.h"

namespace LFL {
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

  int getopt(int argc, const char* const* argv, const char *source_filename);
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
    impl = make_unique<std::thread>(bind(&Thread::ThreadProc, this));
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
  WorkerThread() : queue(make_unique<CallbackQueue>()) {}
  void Init(const Callback &main_cb) { thread = make_unique<Thread>(main_cb); }
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

#include "core/app/audio.h"
#include "core/app/video.h"
#include "core/app/font.h"
#include "core/app/scene.h"
#include "core/app/assets.h"
#include "core/app/input.h"
#include "core/app/shell.h"
#include "core/app/network.h"
#include "core/app/camera.h"

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
  bool rate_limit = 1, wait_forever = 1, wait_forever_thread = 1, synchronize_waits = 1;
  bool monolithic_frame = 1, run_main_loop = 1;
  FrameScheduler() : maxfps(&FLAGS_target_fps), wakeup_thread(&frame_mutex, &wait_mutex) { Setup(); }

  void Setup();
  void Init();
  void Free();
  void Start();
  void DoWait();
  bool FrameWait();
  void FrameDone();
  void Wakeup(void*);
  bool WakeupIn(void*, Time interval, bool force=0);
  void ClearWakeupIn();
  void UpdateTargetFPS(int fps);
  void UpdateWindowTargetFPS(Window*);
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
  static unique_ptr<JSContext> Create(Console *js_console=0, LFL::DOM::Node *document=0);
};

struct LuaContext {
  virtual ~LuaContext() {}
  virtual string Execute(const string &s) = 0;
  static unique_ptr<LuaContext> Create();
};

struct CUDA : public Module { int Init(); };

struct Window : public ::NativeWindow {
  typedef function<int(Window*, unsigned, int)> FrameCB;

  GraphicsDevice *gd=0;
  point mouse, mouse_wheel;
  string caption;
  FrameCB frame_cb;
  Timer frame_time;
  RollingAvg<unsigned> fps;
  unique_ptr<Entity> cam;
  unique_ptr<Console> console;
  vector<GUI*> gui;
  vector<unique_ptr<GUI>> my_gui;
  vector<unique_ptr<InputController>> input;
  vector<unique_ptr<Dialog>> dialogs;
  FontRef default_font = FontRef(FontDesc::Default(), false);
  function<TextBox*()> default_textbox = []{ return nullptr; };
  TextBox *active_textbox=0;
  Dialog *top_dialog=0;
  unique_ptr<Shell> shell;

  Window();
  virtual ~Window();

  LFL::Box Box()                   const { return LFL::Box(width, height); }
  LFL::Box Box(float xs, float ys) const { return LFL::Box(width*xs, height*ys); }
  LFL::Box Box(float xp, float yp, float xs, float ys,
               float xbl=0, float ybt=0, float xbr=-INFINITY, float ybb=-INFINITY) const;

  void SetSize(const point &d);
  void SetCaption(const string &c);
  void SetResizeIncrements(float x, float y);
  void SetTransparency(float v);
  void Reshape(int w, int h);
  void Reshaped(int w, int h);
  void Minimized()   { minimized=1; }
  void UnMinimized() { minimized=0; }
  void ResetGL();
  void SwapAxis();
  int  Frame(unsigned clicks, int flag);
  void RenderToFrameBuffer(FrameBuffer *fb);
  void InitConsole(const Callback &animating_cb);

  template <class X> X* GetGUI(size_t i) { return i < gui.size() ? dynamic_cast<X*>(gui[i]) : nullptr; }
  template <class X> X* GetOwnGUI(size_t i) { return i < my_gui.size() ? dynamic_cast<X*>(my_gui[i].get()) : nullptr; }
  template <class X> X* GetInputController(size_t i) { return i < input.size() ? dynamic_cast<X*>(input[i].get()) : nullptr; }
  template <class X> X* AddInputController(unique_ptr<X> g) { return VectorAddUnique(&input, move(g)); }
  template <class X> void DelGUIPointer(X **g) { DelGUI(*g); *g = nullptr; }
  template <class X> X* AddGUI(unique_ptr<X> g) {
    auto gp = VectorAddUnique(&my_gui, move(g));
    gui.push_back(gp);
    DEBUGf("AddGUI[%zd] %s %p", gui.size()-1, typeid(X).name(), gui.back());
    return gp;
  }
  template <class X> X* ReplaceGUI(size_t i, unique_ptr<X> g) {
    auto gp = g.get();
    if (auto p = my_gui[i].get()) RemoveGUI(p);
    gui.push_back((my_gui[i] = move(g)).get());
    return gp;
  }
  void DelInputController(InputController *g) { VectorRemoveUnique(&input, g); }
  void RemoveGUI(GUI *g) { VectorEraseByValue(&gui, g); }
  void DelGUI(GUI *g);
  size_t NewGUI();

  template <class X> X* AddDialog(unique_ptr<X> d) { auto dp = VectorAddUnique(&dialogs, move(d)); OnDialogAdded(dp); return dp; }
  virtual void OnDialogAdded(Dialog *d) { if (dialogs.size() == 1) BringDialogToFront(d); }
  void BringDialogToFront(Dialog*);
  void GiveDialogFocusAway(Dialog*);
  void DrawDialogs();
};

struct Application : public ::LFApp {
  string name, progname, logfilename, startdir, bindir, assetdir, dldir;
  int pid=0, opengles_version=2;
  FILE *logfile=0;
  tm log_time;
  mutex log_mutex;
  Time time_started;
  Timer frame_time;
  ThreadPool thread_pool;
  CallbackQueue message_queue;
  FrameScheduler scheduler;
  unique_ptr<NetworkThread> network_thread;
  unique_ptr<ProcessAPIClient> render_process;
  unique_ptr<ProcessAPIServer> main_process;
  unordered_map<void*, Window*> windows;
  Callback reshaped_cb, exit_cb;
  function<void(Window*)> window_init_cb, window_start_cb, window_closed_cb = [](Window *w){ delete w; };
  unordered_map<string, StringPiece> asset_cache;
  CategoricalVariable<int> tex_mode, grab_mode, fill_mode;
  const Color *splash_color = &Color::black;
  bool log_pid=0;

  vector<Module*> modules;
  unique_ptr<Module> framework;
  unique_ptr<Audio> audio;
  unique_ptr<Input> input;
  unique_ptr<Fonts> fonts;
  unique_ptr<Shaders> shaders;
  unique_ptr<AssetLoader> asset_loader;
  unique_ptr<Network> net;
  unique_ptr<Camera> camera;
  unique_ptr<CUDA> cuda;

  virtual ~Application();
  Application();

  bool Running() const { return run; }
  bool MainThread() const { return Thread::GetId() == main_thread_id; }
  bool MainProcess() const { return !main_process; }
  double FPS() const { return screen->fps.FPS(); }
  double CamFPS() const { return camera->fps.FPS(); }
  Window *GetWindow(void *id) const { return FindOrNull(windows, id); }
  int LoadModule(Module *m) { return m ? PushBack(modules, m)->Init() : 0; }
  void Log(int level, const char *file, int line, const string &message);
  void WriteLogLine(const char *tbuf, const char *message, const char *file, int line);
  void MakeCurrentWindow(Window*);
  void CloseWindow(Window*);
  void CreateNewWindow();
  void StartNewWindow(Window*);
  NetworkThread *CreateNetworkThread(bool detach_existing_module, bool start);

  int Create(int argc, const char* const* argv, const char *source_filename);
  int Init();
  int Start();
  int HandleEvents(unsigned clicks);
  int EventDrivenFrame(bool handle_events);
  int TimerDrivenFrame(bool got_wakeup);
  int Main();
  int MainLoop();
  void ResetGL();

  void LoseFocus();
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

  template <class... Args> void RunInMainThread(Args&&... args) {
    message_queue.Write(new Callback(forward<Args>(args)...));
    if (!FLAGS_target_fps) scheduler.Wakeup(0);
  }
  template <class... Args> void RunInNetworkThread(Args&&... args) {
    if (auto nt = network_thread.get()) nt->Write(new Callback(forward<Args>(args)...));
    else (Callback(forward<Args>(args)...))();
  }

  static void Daemonize(const char *dir="");
  static void Daemonize(FILE *fout, FILE *ferr);
  static void *GetSymbol(const string &n);
  static StringPiece LoadResource(int id);
};

unique_ptr<Module> CreateFrameworkModule();
unique_ptr<GraphicsDevice> CreateGraphicsDevice(int ver);
unique_ptr<Module> CreateAudioModule(Audio*);
unique_ptr<Module> CreateCameraModule(CameraState*);

}; // namespace LFL
#endif // LFL_CORE_APP_APP_H__
