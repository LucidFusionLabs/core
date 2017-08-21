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
#include <cctype>
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
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include <process.h>
#include <malloc.h>
#include <io.h>
#include <sstream>
#include <typeinfo>
typedef SOCKET Socket;
struct iovec { void *iov_base; size_t iov_len; };
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
#undef CreateWindow
#define S_IFDIR _S_IFDIR
#define getcwd _getcwd
#define chdir _chdir
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define snprintf _snprintf
#endif

#if defined(LFL_LINUX) || defined(LFL_EMSCRIPTEN)
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

#define LABEL(x) " " #x "=", x
#define ONCE(x) { static bool once=0; if (!once && (once=1)) { x; } }
#define ONCE_ELSE(x, y) { static bool once=0; if (!once && (once=1)) { x; } else { y; } }
#define EVERY_N(x, y) { static int every_N=0; if (every_N++ % (x) == 0) { y; } }
#define EX_LT(x, a)    if (!((x) <  (a))) INFO((x), " < ",  (a), ": EX_LT(",    #x, ", ", #a, ")")
#define EX_GT(x, a)    if (!((x) >  (a))) INFO((x), " > ",  (a), ": EX_GT(",    #x, ", ", #a, ")")
#define EX_GE(x, a)    if (!((x) >= (a))) INFO((x), " >= ", (a), ": EX_GE(",    #x, ", ", #a, ")")
#define EX_LE(x, a)    if (!((x) <= (a))) INFO((x), " <= ", (a), ": EX_LE(",    #x, ", ", #a, ")")
#define EX_EQ(x, a)    if (!((x) == (a))) INFO((x), " == ", (a), ": EX_EQ(",    #x, ", ", #a, ")")
#define EX_NE(x, a)    if (!((x) != (a))) INFO((x), " != ", (a), ": EX_NE(",    #x, ", ", #a, ")")
#define CHECK_LT(x, a) if (!((x) <  (a))) ::LFL::FatalMessage(__FILE__, __LINE__, ::LFL::StrCat((x), " < ",  (a), ": CHECK_LT(", #x, ", ", #a, ")")).GetStream()
#define CHECK_GT(x, a) if (!((x) >  (a))) ::LFL::FatalMessage(__FILE__, __LINE__, ::LFL::StrCat((x), " > ",  (a), ": CHECK_GT(", #x, ", ", #a, ")")).GetStream()
#define CHECK_GE(x, a) if (!((x) >= (a))) ::LFL::FatalMessage(__FILE__, __LINE__, ::LFL::StrCat((x), " >= ", (a), ": CHECK_GE(", #x, ", ", #a, ")")).GetStream()
#define CHECK_LE(x, a) if (!((x) <= (a))) ::LFL::FatalMessage(__FILE__, __LINE__, ::LFL::StrCat((x), " <= ", (a), ": CHECK_LE(", #x, ", ", #a, ")")).GetStream()
#define CHECK_EQ(x, a) if (!((x) == (a))) ::LFL::FatalMessage(__FILE__, __LINE__, ::LFL::StrCat((x), " == ", (a), ": CHECK_EQ(", #x, ", ", #a, ")")).GetStream()
#define CHECK_NE(x, a) if (!((x) != (a))) ::LFL::FatalMessage(__FILE__, __LINE__, ::LFL::StrCat((x), " != ", (a), ": CHECK_NE(", #x, ", ", #a, ")")).GetStream()
#define CHECK_RANGE(x, y, z) { CHECK_GE(x, y); CHECK_LT(x, z); }
#define CHECK(x) if (!(x)) FATAL(#x)

#ifdef LFL_DEBUG
#define DEBUG_CHECK_LT(x, a) CHECK_LT(x, a)
#define DEBUG_CHECK_GT(x, a) CHECK_GT(x, a)
#define DEBUG_CHECK_GE(x, a) CHECK_GE(x, a)
#define DEBUG_CHECK_LE(x, a) CHECK_LE(x, a)
#define DEBUG_CHECK_EQ(x, a) CHECK_EQ(x, a)
#define DEBUG_CHECK_NE(x, a) CHECK_NE(x, a)
#define DEBUG_CHECK_RANGE(x, y, z) CHECK_RANGE(x, y, z)
#define DEBUG_CHECK(x) CHECK(x)
#else
#define DEBUG_CHECK_LT(x, a)
#define DEBUG_CHECK_GT(x, a)
#define DEBUG_CHECK_GE(x, a)
#define DEBUG_CHECK_LE(x, a)
#define DEBUG_CHECK_EQ(x, a)
#define DEBUG_CHECK_NE(x, a)
#define DEBUG_CHECK_RANGE(x, y, z)
#define DEBUG_CHECK(x)
#endif

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
using LFL_STL11_NAMESPACE::remove_reference;
using LFL_STL11_NAMESPACE::enable_if;
using LFL_STL11_NAMESPACE::is_integral;
using LFL_STL11_NAMESPACE::is_floating_point;
using LFL_STL11_NAMESPACE::is_signed;
using LFL_STL11_NAMESPACE::is_pod;
using LFL_STL11_NAMESPACE::make_unsigned;
using LFL_STL11_NAMESPACE::make_signed;
using LFL_STL11_NAMESPACE::make_unique;
using LFL_STL11_NAMESPACE::make_shared;
using LFL_STL11_NAMESPACE::make_tuple;
using LFL_STL11_NAMESPACE::isinf;
using LFL_STL11_NAMESPACE::isnan;
using LFL_STL11_NAMESPACE::atoi;
using LFL_STL11_NAMESPACE::atof;
#define tuple_get LFL_STL11_NAMESPACE::get

template <class X> struct Singleton { static X *Get() { static X instance; return &instance; } };
template <class X> struct V2;
typedef V2<float> v2;
typedef V2<int> point;
typedef lock_guard<mutex> ScopedMutex;
typedef vector<string> StringVec;
typedef pair<string, string> StringPair;
typedef vector<StringPair> StringPairVec;
typedef function<void()> Callback;
typedef function<void(int)> IntCB;
typedef function<void(int, int)> IntIntCB;
typedef function<void(const string&)> StringCB;
typedef function<void(int, const string&)> IntStringCB;
typedef function<void(const string&, const string&)> StringStringCB;
typedef function<void(const StringVec&)> StringVecCB; 
void Log(int level, const char *file, int line, const string &m);
}; // namespace LFL

#include "core/app/export.h"
#include "core/app/types/string.h"
#include "core/app/types/time.h"

namespace LFL {
extern Application *app;
extern const char *not_implemented;
extern const bool DEBUG, MOBILE, IOS, ANDROIDOS;

struct FatalMessage {
  const char *file; int line; string msg; std::stringstream stream;
  FatalMessage(const char *F, int L, string M) : file(F), line(L), msg(move(M)) {}
  ~FatalMessage() { ::LFL::Log(::LFApp::Log::Fatal, file, line, msg.append(stream.str())); throw(0); }
  std::stringstream& GetStream() { return stream; }
};

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
  ThreadLocalStorage() : rand_eng(LFAppNextRandSeed()) {}
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

namespace LFL {
::std::ostream& operator<<(::std::ostream& os, const point &x);
::std::ostream& operator<<(::std::ostream& os, const Box   &x);
}; // namespace LFL

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

  string Get() const { return Printable(*v); }
  bool IsBool() const { return TypeId<X>() == TypeId<bool>(); }
  void Update(const char *text) { if (text) *v = Scannable::Scan(*v, text); }
};

struct Thread {
  typedef unsigned long long id_t;
  id_t id=0;
  Callback cb;
  mutex start_mutex;
  unique_ptr<std::thread> impl;
  virtual ~Thread() { Wait(); }
  Thread(const Callback &CB=Callback()) : cb(CB) {}

  void Open(const Callback &CB) { cb=CB; }
  void Wait() { if (impl) { impl->join(); impl.reset(); id=0; } }
  void Start();
  void ThreadProc();
  static id_t GetId() { return std::hash<std::thread::id>()(std::this_thread::get_id()); }
};

struct WorkerThread {
  unique_ptr<CallbackQueue> queue;
  unique_ptr<Thread> thread;
  bool run=0;
  WorkerThread() : queue(make_unique<CallbackQueue>()),
  thread(make_unique<Thread>(bind(&CallbackQueue::HandleMessagesWhile, queue.get(), &run))) {}

  void Stop() { if (!run) return; run=0; queue->Shutdown(); thread->Wait(); }
  void Start() { run=1; thread->Start(); }
};

struct ThreadPool {
  vector<WorkerThread> worker;
  int round_robin_next=0;
  virtual ~ThreadPool() { Stop(); }

  void Open(int num) { CHECK(worker.empty()); for (int i=0; i<num; i++) worker.emplace_back(); }
  void Start() { for (auto &w : worker) w.Start(); }
  void Stop()  { for (auto &w : worker) w.Stop(); }
  void Write(Callback *cb);
};

struct Timer {
  Time begin;
  Timer() { Reset(); }
  Time Reset() { Time last_begin=begin; begin=Now(); return last_begin; }
  Time GetTime() const { return Now() - begin; }
  Time GetTime(bool do_reset);
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

struct HAlign { enum { Left  =1, Center=2, Right=3 }; };
struct VAlign { enum { Bottom=1, Center=2, Top  =3 }; };
}; // namespace LFL

#include "core/app/toolkit.h"
#ifdef LFL_ANDROID
#include "core/app/framework/android_common.h"
#endif
#include "core/app/audio.h"
#include "core/app/gl.h"
#include "core/app/font.h"
#include "core/app/scene.h"
#include "core/app/loader.h"
#include "core/app/assets.h"
#include "core/app/layers.h"
#include "core/app/input.h"
#include "core/app/shell.h"
#include "core/app/network.h"
#include "core/app/camera.h"

namespace LFL {
struct RateLimiter {
  int *target_hz;
  float avgframe;
  Timer timer;
  RollingAvg<unsigned> sleep_bias;
  RateLimiter(int *HZ) : target_hz(HZ), avgframe(0), sleep_bias(32) {}
  void Limit();
};

struct FrameScheduler {
  struct WakeupFlag { enum { InMainThread=1, ContingentOnEvents=2 }; };
  RateLimiter maxfps;
  mutex frame_mutex, wait_mutex;
  SocketWakeupThread wakeup_thread;
  SelectSocketSet main_wait_sockets;
  Socket system_event_socket = InvalidSocket, main_wait_wakeup_socket = InvalidSocket, iter_socket = InvalidSocket;
  const bool rate_limit, wait_forever, wait_forever_thread, synchronize_waits, monolithic_frame, run_main_loop;
  FrameScheduler();

  void Init();
  void Free();
  void Start();
  bool MainWait();
  bool DoMainWait(bool poll=0);
  void Wakeup(Window*, int flag=0);
  void UpdateTargetFPS(Window*, int fps);
  void UpdateWindowTargetFPS(Window*);
  void SetAnimating(Window*, bool);
  void AddMainWaitMouse(Window*);
  void DelMainWaitMouse(Window*);
  void AddMainWaitKeyboard(Window*);
  void DelMainWaitKeyboard(Window*);
  void AddMainWaitSocket(Window*, Socket fd, int flag, function<bool()> = []{ return 1; });
  void DelMainWaitSocket(Window*, Socket fd);
};

struct FrameWakeupTimer {
  Window *root;
  bool needs_frame=false;
  unique_ptr<TimerInterface> timer;
  FrameWakeupTimer(Window *w);
  void ClearWakeupIn();
  bool WakeupIn(Time interval);
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

struct Window : public ::LFAppWindow {
  typedef function<int(Window*, unsigned, int)> FrameCB;

  GraphicsDevice *gd=0;
  point mouse, mouse_v, mouse2, mouse2_v, mouse_wheel;
  string caption;
  Callback reshaped_cb, focused_cb, unfocused_cb;
  FrameCB frame_cb;
  Timer frame_time;
  RollingAvg<unsigned> fps;
  unique_ptr<Console> console;
  vector<View*> view;
  vector<InputController*> input;
  vector<unique_ptr<View>> my_view;
  vector<unique_ptr<InputController>> my_input;
  vector<unique_ptr<Dialog>> dialogs;
  FontRef default_font = FontRef(FontDesc::Default(), false);
  function<MouseController*()> default_controller = []{ return nullptr; };
  function<KeyboardController*()> default_textbox = []{ return nullptr; };
  MouseController *active_controller=0;
  KeyboardController *active_textbox=0;
  Dialog *top_dialog=0;
  unique_ptr<Shell> shell;
  CategoricalVariable<int> tex_mode, grab_mode, fill_mode;

  Window();
  virtual ~Window();
  static Window *Create();

  virtual void SetCaption(const string &c)           = 0;
  virtual void SetResizeIncrements(float x, float y) = 0;
  virtual void SetTransparency(float v)              = 0;
  virtual bool Reshape(int w, int h)                 = 0;

  LFL::Box Box()                   const { return LFL::Box(gl_x, gl_y, gl_w, gl_h); }
  LFL::Box Box(float xs, float ys) const { return LFL::Box(gl_w*xs, gl_h*ys); }
  LFL::Box Box(float xp, float yp, float xs, float ys,
               float xbl=0, float ybt=0, float xbr=-INFINITY, float ybb=-INFINITY) const;
  void SetBox  (const point &win_d, const LFL::Box &gl_box);
  void Reshaped(const point &win_d, const LFL::Box &gl_box);
  void Minimized()   { minimized=1; }
  void UnMinimized() { minimized=0; }
  void ResetGL(int flag);
  void SwapAxis();
  void ClearChildren();
  int  Frame(unsigned clicks, int flag);
  void RenderToFrameBuffer(FrameBuffer *fb);
  void InitConsole(const Callback &animating_cb);

  template <class X> X* GetView(size_t i) { return i < view.size() ? dynamic_cast<X*>(view[i]) : nullptr; }
  template <class X> X* GetOwnView(size_t i) { return i < my_view.size() ? dynamic_cast<X*>(my_view[i].get()) : nullptr; }
  template <class X> X* GetInputController(size_t i) { return i < input.size() ? dynamic_cast<X*>(input[i]) : nullptr; }
  template <class X> X* GetOwnInputController(size_t i) { return i < my_input.size() ? dynamic_cast<X*>(my_input[i].get()) : nullptr; }
  template <class X> X* AddInputController(unique_ptr<X> g) { auto gp = VectorAddUnique(&my_input, move(g)); input.push_back(gp); return gp; }
  template <class X> void DelViewPointer(X **g) { DelView(*g); *g = nullptr; }
  template <class X> X* AddView(unique_ptr<X> g) {
    auto gp = VectorAddUnique(&my_view, move(g));
    view.push_back(gp);
    DEBUGf("AddView[%zd] %s %p", view.size()-1, typeid(X).name(), view.back());
    return gp;
  }
  template <class X> X* ReplaceView(size_t i, unique_ptr<X> g) {
    auto gp = g.get();
    if (auto p = my_view[i].get()) RemoveView(p);
    view.push_back((my_view[i] = move(g)).get());
    return gp;
  }
  void DelInputController(InputController *g) { RemoveInputController(g); VectorRemoveUnique(&my_input, g); }
  void RemoveInputController(InputController *g) { VectorEraseByValue(&input, g); }
  void RemoveView(View *v) { VectorEraseByValue(&view, v); }
  void DelView(View *v);
  size_t NewView();

  template <class X> X* AddDialog(unique_ptr<X> d) { auto dp = VectorAddUnique(&dialogs, move(d)); OnDialogAdded(dp); return dp; }
  virtual void OnDialogAdded(Dialog *d);
  void BringDialogToFront(Dialog*);
  void GiveDialogFocusAway(Dialog*);
  void DrawDialogs();
};

struct Video {
  static int Swap();
  static bool CreateWindow(Window *W);
  static void StartWindow(Window *W);
  static void *BeginGLContextCreate(Window *);
  static void *CompleteGLContextCreate(Window *, void *gl_context);
};

struct Application : public ::LFApp {
  string name, progname, startdir, bindir, assetdir, savedir;
  int pid=0, opengles_version=2, argc=0;
  FILE *logfile=0, *logout=stdout, *logerr=stderr;
  const char* const* argv=0;
  tm log_time;
  mutex log_mutex;
  Time time_started;
  Timer frame_time;
  ThreadPool thread_pool;
  CallbackQueue message_queue;
  FrameScheduler scheduler;
  unique_ptr<SocketServicesThread> network_thread;
  unique_ptr<ProcessAPIClient> render_process;
  unique_ptr<ProcessAPIServer> main_process;
  unordered_map<const void*, Window*> windows;
  function<void(Window*)> window_init_cb, window_start_cb, window_closed_cb = [](Window *w){ delete w; };
  Window *focused=0;
  ToolkitInterface *toolkit;

  unordered_map<string, StringPiece> asset_cache;
  const Color *splash_color = &Color::black;
  StringCB open_url_cb;
  Callback exit_cb;

  vector<Module*> modules;
  unique_ptr<Module> framework;
  unique_ptr<Audio> audio;
  unique_ptr<Input> input;
  unique_ptr<Fonts> fonts;
  unique_ptr<Shaders> shaders;
  unique_ptr<AssetLoader> asset_loader;
  unique_ptr<SocketServices> net;
  unique_ptr<Camera> camera;
  unique_ptr<CUDA> cuda;

  AssetMap      asset;
  SoundAssetMap soundasset;
  MovieAssetMap movieasset;

  virtual ~Application();
  Application(int ac, const char* const* av);

  bool Running() const { return run; }
  bool MainThread() const { return Thread::GetId() == main_thread_id; }
  bool MainProcess() const { return !main_process; }
  Window *GetWindow(void *id) const { return FindOrNull(windows, id); }
  int LoadModule(Module *m) { return m ? PushBack(modules, m)->Init() : 0; }

  string   GetLocalizedString   (const char *key);
  String16 GetLocalizedString16 (const char *key);
  string   GetLocalizedInteger  (int number);
  String16 GetLocalizedInteger16(int number);
  string   GetPackageName();
  string   GetVersion();
  string   GetSystemDeviceName();
  string   GetSystemDeviceId();
  string   PrintCallStack();

  void Log(int level, const char *file, int line, const char *message);
  int Create(const char *source_filename);
  int Init();
  int Start();
  int HandleEvents(unsigned clicks);
  int EventDrivenFrame(bool handle_events, bool draw_frame);
  int TimerDrivenFrame(bool got_wakeup);
  int Main();
  int MainLoop();
  int Suspended();
  void DrawSplash(const Color &c);
  void ResetGL(int flag);
  void MakeCurrentWindow(Window*);
  void CloseWindow(Window*);
  void CreateNewWindow();
  void StartNewWindow(Window*);
  void LoseFocus();
  void GrabMouseFocus();
  void ReleaseMouseFocus();
  string GetClipboardText();
  void SetClipboardText(const string &s);
  void OpenSystemBrowser(const string &url);
  void ShowSystemContextMenu(const MenuItemVec &items);
  void ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB&);
  void ShowSystemFileChooser(bool files, bool dirs, bool multi, const StringVecCB&);
  void ShowSystemStatusBar(bool);
  int LoadSystemImage(const string &fn);
  void UpdateSystemImage(int n, Texture&);
  void UnloadSystemImage(int n);
  bool OpenSystemAppPreferences();
  void OpenTouchKeyboard();
  void CloseTouchKeyboard();
  void CloseTouchKeyboardAfterReturn(bool);
  void SetTouchKeyboardTiled(bool);
  void ToggleTouchKeyboard();
  bool GetAppFrameEnabled();
  void SetAppFrameEnabled(bool);
  void SetAutoRotateOrientation(bool);
  void SetVerticalSwipeRecognizer(int touches);
  void SetHorizontalSwipeRecognizer(int touches);
  void SetPanRecognizer(bool enabled);
  void SetPinchRecognizer(bool enabled);
  int SetMultisample(bool on);
  int SetExtraScale(bool on); /// e.g. Retina display
  void SetDownScale(bool on);
  void SetTitleBar(bool on);
  void SetKeepScreenOn(bool on);
  void SetExtendedBackgroundTask(Callback);
  void SetTheme(const string&);

  bool LoadKeychain(const string &key, string *val);
  void SaveKeychain(const string &key, const string &val);

  SocketServicesThread *CreateNetworkThread(bool detach_existing_module, bool start);
  Connection *ConnectTCP(const string &hostport, int default_port, Connection::CB *connected_cb,
                         bool background_services = false);

  int GetVolume();
  int GetMaxVolume();
  void SetVolume(int v);
  void PlaySoundEffect(SoundAsset*, const v3 &pos=v3(), const v3 &vel=v3());
  void PlayBackgroundMusic(SoundAsset*);

  void RunCallbackInMainThread(Callback cb);
  void RunNowInMainThread(Callback cb) { if (MainThread()) cb(); else RunCallbackInMainThread(move(cb)); }
  template <class... Args> void RunInMainThread(Args&&... args) {
    RunCallbackInMainThread(Callback(forward<Args>(args)...));
  }
  template <class... Args> void RunInNetworkThread(Args&&... args) {
    if (auto nt = network_thread.get()) nt->Write(new Callback(forward<Args>(args)...));
    else (Callback(forward<Args>(args)...))();
  }
  template <class... Args> void RunInThreadPool(Args&&... args) {
    thread_pool.Write(new Callback(forward<Args>(args)...));
  } 

  static StringPiece LoadResource(int id);
  static void *GetSymbol(const string &n);
  static string GetSetting(const string &key);
  static void SaveSettings(const StringPairVec&);
  static void SaveSetting(const string &k, const string &v) { SaveSettings({ StringPair(k,v) }); }
  static void LoadDefaultSettings(const StringPairVec&);
  static void Daemonize(const char *dir, const char *progname);
  static void Daemonize(FILE *fout, FILE *ferr);
  static void WriteLogLine(const char *tbuf, const char *message, const char *file, int line);
  static void WriteDebugLine(const char *message, const char *file, int line);
};

inline string   LS  (const char *n) { return app->GetLocalizedString(n); }
inline String16 LS16(const char *n) { return app->GetLocalizedString16(n); }

Application *CreateApplication(int ac, const char* const* av);
unique_ptr<Module> CreateFrameworkModule();
unique_ptr<Module> CreateAudioModule(Audio*);
unique_ptr<Module> CreateCameraModule(CameraState*);
unique_ptr<GraphicsDevice> CreateGraphicsDevice(Window*, int ver);
VideoResamplerInterface *CreateVideoResampler();
void InitCrashReporting(const string &id, const string &name, const string &email);
void TestCrashReporting();

}; // namespace LFL
#endif // LFL_CORE_APP_APP_H__
