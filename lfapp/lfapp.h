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

#ifndef __LFL_LFAPP_LFAPP_H__
#define __LFL_LFAPP_LFAPP_H__

#ifndef _WIN32
#include <sstream>
#include <typeinfo>
#endif

#include <vector>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <deque>
#include <algorithm>
#define LFL_STL_NAMESPACE std

#include <memory>
#include <numeric>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#define LFL_STL11_NAMESPACE std

#ifdef _WIN32
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
typedef int Socket;
#endif

using LFL_STL_NAMESPACE::min;
using LFL_STL_NAMESPACE::max;
using LFL_STL_NAMESPACE::swap;
using LFL_STL_NAMESPACE::pair;
using LFL_STL_NAMESPACE::vector;
using LFL_STL_NAMESPACE::string;
using LFL_STL_NAMESPACE::basic_string;
using LFL_STL_NAMESPACE::map;
using LFL_STL_NAMESPACE::set;
using LFL_STL_NAMESPACE::deque;
using LFL_STL_NAMESPACE::inserter;
using LFL_STL_NAMESPACE::binary_search;
using LFL_STL_NAMESPACE::sort;
using LFL_STL_NAMESPACE::unique;
using LFL_STL_NAMESPACE::reverse;
using LFL_STL_NAMESPACE::equal_to;
using LFL_STL_NAMESPACE::lower_bound;
using LFL_STL_NAMESPACE::set_difference;
using LFL_STL11_NAMESPACE::unordered_map;
using LFL_STL11_NAMESPACE::unordered_set;
using LFL_STL11_NAMESPACE::shared_ptr;
using LFL_STL11_NAMESPACE::unique_ptr;
using LFL_STL11_NAMESPACE::move;
using LFL_STL11_NAMESPACE::bind;
using LFL_STL11_NAMESPACE::function;
using LFL_STL11_NAMESPACE::placeholders::_1;
using LFL_STL11_NAMESPACE::placeholders::_2;
using LFL_STL11_NAMESPACE::placeholders::_3;
using LFL_STL11_NAMESPACE::placeholders::_4;
using LFL_STL11_NAMESPACE::placeholders::_5;
using LFL_STL11_NAMESPACE::placeholders::_6;
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
using LFL_STL11_NAMESPACE::chrono::high_resolution_clock;

#include <errno.h>
#include <math.h>
#include <cfloat>

#ifdef _WIN32
#include <float.h>
#include <direct.h>
#undef CALLBACK
#define isinf(x) (x <= -INFINITY || x >= INFINITY)
#define isnan _isnan
#define isfinite _finite
#define getcwd _getcwd
#define chdir _chdir
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define snprintf _snprintf
#define S_IFDIR _S_IFDIR
#define socklen_t int
int close(int socket);
extern char *optarg;
extern int optind;
#endif

#ifdef __APPLE__
#include <cmath>
extern "C" int isnan(double);
extern "C" int isinf(double);
#define isfinite(x) (!isnan(x) && !isinf(x))
#else
#define isfinite(x) finite(x)
#endif

#ifdef LFL_ANDROID
#include <sys/endian.h>
#include "lfjni/lfjni.h"
#endif

#ifdef __linux__
#include <arpa/inet.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define  INFO(...) ::LFL::Log(::LFApp::Log::Info,  __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__))
#define DEBUG(...) ::LFL::Log(::LFApp::Log::Debug, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__))
#define ERROR(...) ::LFL::Log(::LFApp::Log::Error, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__))
#define FATAL(...) { ::LFL::Log(::LFApp::Log::Fatal, __FILE__, __LINE__, ::LFL::StrCat(__VA_ARGS__)); throw(0); }

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
typedef lock_guard<mutex> ScopedMutex;
typedef function<void()> Callback;
template <class X> struct Singleton { static X *Get() { static X instance; return &instance; } };
void Log(int level, const char *file, int line, const string &m);
}; // namespace LFL

#include "lfapp/lfexport.h"
#include "lfapp/string.h"
#include "lfapp/time.h"

namespace LFL {
Time Now();
void MSleep(int x);
inline bool Equal(float a, float b, float eps=1e-6) { return fabs(a-b) < eps; }

struct Allocator {
    virtual ~Allocator() {}
    virtual const char *Name() = 0;
    virtual void *Malloc(int size) = 0;
    virtual void *Realloc(void *p, int size) = 0;
    virtual void Free(void *p) = 0;
    virtual void Reset();
    static Allocator *Default();
#define AllocatorNew(allocator, type, constructor_args) (new((allocator)->Malloc(sizeof type )) type constructor_args)
};

struct NullAlloc : public Allocator {
    const char *Name() { return "NullAlloc"; }
    void *Malloc(int size) { return 0; }
    void *Realloc(void *p, int size) { return 0; }
    void Free(void *p) {}
};

struct Module {
    virtual int Init ()         { return 0; }
    virtual int Start()         { return 0; }
    virtual int Frame(unsigned) { return 0; }
    virtual int Free ()         { return 0; }
};

struct FrameFlag { enum { DontSkip=8 }; };
typedef function<int(Window*, unsigned, unsigned, bool, int)> FrameCB;
}; // namespace LFL

#include "lfapp/math.h"
#include "lfapp/lftypes.h"
#include "lfapp/file.h"

namespace LFL {
bool Running();
bool MainThread();
void RunInMainThread(Callback *cb);
void DefaultLFAppWindowClosedCB(Window *);
double FPS();
double CamFPS();
void PressAnyKey();
bool FGets(char *buf, int size);
bool NBFGets(FILE*, char *buf, int size, int timeout=0);
int NBRead(int fd, char *buf, int size, int timeout=0);
int NBRead(int fd, string *buf, int timeout=0);
string NBRead(int fd, int size, int timeout=0);

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

template <int S> struct FixedAlloc : public Allocator {
    const char *Name() { return "FixedAlloc"; }
    static const int size = S;
    char buf[S];
    int len=0;
    virtual void Reset() { len=0; }
    virtual void *Malloc(int n) { CHECK_LE(len + n, S); char *ret = &buf[len]; len += NextMultipleOf16(n); return ret; }
    virtual void *Realloc(void *p, int n) { CHECK_EQ(NULL, p); return this->Malloc(n); }
    virtual void Free(void *p) {}
};

struct BlockChainAlloc : public Allocator {
    const char *Name() { return "BlockChainAlloc"; }
    struct Block { string buf; int len=0; Block(int size=0) : buf(size, 0) {} };
    vector<Block> blocks;
    int block_size, cur_block_ind;
    BlockChainAlloc(int s=1024*1024) : block_size(s), cur_block_ind(-1) {}
    void Reset() { for (auto &b : blocks) b.len = 0; cur_block_ind = blocks.size() ? 0 : -1; }
    void *Realloc(void *p, int n) { CHECK_EQ(NULL, p); return this->Malloc(n); }
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

struct ThreadLocalStorage {
    Allocator *alloc=0;
    virtual ~ThreadLocalStorage() { delete alloc; }
    static thread_local ThreadLocalStorage *instance;
    static ThreadLocalStorage *Get() {
        if (!instance) instance = new ThreadLocalStorage();
        return instance;
    }
    static Allocator *GetAllocator(bool reset_allocator=true) {
        ThreadLocalStorage *tls = Get();
        if (!tls->alloc) tls->alloc = new FixedAlloc<1024*1024>;
        if (reset_allocator) tls->alloc->Reset();
        return tls->alloc;
    }
};

struct Thread {
    typedef unsigned long long Id;
    Id id=0;
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
        cb();
        delete ThreadLocalStorage::instance;
    }
    static Id GetId() { return std::hash<std::thread::id>()(std::this_thread::get_id()); }
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

struct ProcessPipe {
    int pid=0;
    FILE *in=0, *out=0;
    virtual ~ProcessPipe() { Close(); }
    int Open(const char **argv);
    int OpenPTY(const char **argv);
    int Close();
};

struct InterProcessResource {
    int id=-1;
    string url;
    char *buf=0;
    const int len=0;
    InterProcessResource(int size, const string &ipr_url=string());
    ~InterProcessResource();
};

struct NTService {
    static int Install  (const char *name, const char *path);
    static int Uninstall(const char *name);
    static int WrapMain (const char *name, MainCB main_cb, int argc, const char **argv);
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
    string DebugString() const { string v; for (auto &t : timers) StrAppend(&v, t.name, " ", t.time / 1000.0, "\n"); return v; }
};

struct Crypto {
    string MD5(const string &in);
    string Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt);
};
}; // namespace LFL

#include "lfapp/audio.h"
#include "lfapp/video.h"
#include "lfapp/font.h"
#include "lfapp/scene.h"
#include "lfapp/assets.h"
#include "lfapp/input.h"
#include "lfapp/network.h"
#include "lfapp/camera.h"

namespace LFL {
::std::ostream& operator<<(::std::ostream& os, const point &x);
::std::ostream& operator<<(::std::ostream& os, const Box   &x);

struct FrameRateLimitter {
    int *target_hz;
    float avgframe;
    Timer timer;
    RollingAvg<unsigned> sleep_bias;
    FrameRateLimitter(int *HZ) : target_hz(HZ), avgframe(0), sleep_bias(32) {}
    void Limit() {
        Time since = timer.GetTime(true), targetframe(1000 / *target_hz);
        Time sleep = max(Time(0), targetframe - since - FMilliseconds(sleep_bias.Avg()));
        if (sleep != Time(0)) { MSleep(sleep.count()); sleep_bias.Add((timer.GetTime(true) - sleep).count()); }
    }
};

struct FrameScheduler {
    FrameRateLimitter maxfps;
    mutex frame_mutex, wait_mutex;
    SocketWakeupThread wakeup_thread;
    bool rate_limit = 1, wait_forever = 1, wait_forever_thread = 1, synchronize_waits = 1, monolithic_frame = 1;
    FrameScheduler();

    void Init();
    void Free();
    void Start();
    void FrameWait();
    void FrameDone();
    void Wakeup();
    void UpdateTargetFPS(int fps);
    void AddWaitForeverMouse();
    void DelWaitForeverMouse();
    void AddWaitForeverKeyboard();
    void DelWaitForeverKeyboard();
    void AddWaitForeverSocket(Socket fd, int flag, void *val=0);
    void DelWaitForeverSocket(Socket fd);
};

struct BrowserInterface {
    virtual void Draw(Box *viewport) = 0;
    virtual void Open(const string &url) = 0;
    virtual void Navigate(const string &url) { Open(url); }
    virtual Asset *OpenImage(const string &url) { return 0; }
    virtual void OpenStyleImport(const string &url) {}
    virtual void MouseMoved(int x, int y) = 0;
    virtual void MouseButton(int b, bool d) = 0;
    virtual void MouseWheel(int xs, int ys) = 0;
    virtual void KeyEvent(int key, bool down) = 0;
    virtual void BackButton() = 0;
    virtual void ForwardButton() = 0;
    virtual void RefreshButton() = 0;
    virtual string GetURL() = 0;
};

struct JSContext {
    virtual ~JSContext() {}
    virtual string Execute(const string &s) = 0;
};
JSContext *CreateV8JSContext(Console *js_console=0, LFL::DOM::Node *document=0);

struct LuaContext {
    virtual ~LuaContext() {}
    virtual string Execute(const string &s) = 0;
};
LuaContext *CreateLuaContext();

struct SystemBrowser { static void Open(const char *url); };
struct Clipboard { static string Get(); static void Set(const string &s); };
struct TouchDevice { static void OpenKeyboard(); static void CloseKeyboard(); };
struct Advertising { static void ShowAds(); static void HideAds(); };
struct CUDA : public Module { int Init(); };

struct Application : public ::LFApp, public Module {
    string progname, logfilename, startdir, assetdir, dldir;
    FILE *logfile=0;
    mutex log_mutex;
    Time time_started;
    Timer frame_time;
    ThreadPool thread_pool;
    CallbackQueue message_queue;
    FrameScheduler scheduler;
    Callback reshaped_cb;
    function<void(Window*)> window_init_cb, window_closed_cb;
    Audio audio;
    Video video;
    Input input;
    Assets assets;
    Network network;
    Camera camera;
    CUDA cuda;
    Shell shell;
    vector<Module*> modules;
    CategoricalVariable<int> tex_mode, grab_mode, fill_mode;

    Application() : window_closed_cb(DefaultLFAppWindowClosedCB), tex_mode(2, 1, 0), grab_mode(2, 0, 1),
    fill_mode(3, GraphicsDevice::Fill, GraphicsDevice::Line, GraphicsDevice::Point)
    { run=1; initialized=0; main_thread_id=0; frames_ran=pre_frames_ran=0; }

    void Log(int level, const char *file, int line, const string &message);
    void CreateNewWindow(const function<void(Window*)> &start_cb = function<void(Window*)>());
    NetworkThread *CreateNetworkThread();
    int LoadModule(Module *M) { modules.push_back(M); return M->Init(); }
    string BinDir() const { return LocalFile::JoinPath(startdir, progname.substr(0, DirNameLen(progname, true))); }

    int Create(int argc, const char **argv, const char *source_filename);
    int Init();
    int Start();
    int PreFrame(unsigned clicks);
    int PostFrame();
    int Frame();
    int Main();
    int MainLoop();
    int Free();
    int Exiting();

    static void Daemonize(const char *dir="");
    static void Daemonize(FILE *fout, FILE *ferr);
};
extern Application *app;
}; // namespace LFL

#if defined(LFL_QT)
#define main LFLQTMain
#elif defined(LFL_IPHONE)
#define main iPhoneMain
#elif defined(LFL_OSXVIDEO)
#define main OSXMain
#endif

extern "C" int main(int argc, const char **argv);

#endif // __LFL_LFAPP_LFAPP_H__
