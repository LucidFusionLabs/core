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
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"

#include <time.h>
#include <fcntl.h>

#ifdef WIN32
#define CALLBACK __stdcall
#include <Shlobj.h>
#include <Windns.h>
#include <sys/stat.h>
#define stat(x,y) _stat(x,y)
#define gmtime_r(i,o) memcpy(o, gmtime(&in), sizeof(tm))
#define localtime_r(i,o) memcpy(o, localtime(&in), sizeof(tm))
#else
#include <dirent.h>
#include <signal.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#endif

#ifdef LFL_CUDA
#include <cuda_runtime.h>
#include "lfcuda/lfcuda.h"
#include "speech/hmm.h"
#include "speech/speech.h"
#endif

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#ifndef LFL_IPHONE
#include <ApplicationServices/ApplicationServices.h>
#endif
#endif

#if defined(LFL_GLFWVIDEO) || defined(LFL_GLFWINPUT)
#include "GLFW/glfw3.h"
#endif

#if defined(LFL_SDLAUDIO) || defined(LFL_SDLVIDEO) || defined(LFL_SDLINPUT)
#include "SDL.h"
#endif

#ifdef LFL_ANDROID
#include <android/log.h>
#endif

#ifdef LFL_LIBARCHIVE
#include "libarchive/archive.h"
#include "libarchive/archive_entry.h"
#endif

#ifdef LFL_ICONV
#include <iconv.h>
#endif

#ifdef LFL_OPENSSL
#include <openssl/evp.h>
#include <openssl/md5.h>
#include <openssl/err.h>
#endif

extern "C" {
#ifdef LFL_LUA
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
#endif

#ifdef LFL_SREGEX
#include "sregex.h"
#endif
};

#ifdef LFL_REGEX
#include "regexp.h"
#endif

#ifdef LFL_V8JS
#include <v8.h>
#endif

#if defined(LFL_IPHONE)
extern "C" void iPhoneOpenBrowser(const char *url_text);
#elif defined(__APPLE__)
extern "C" void OSXStartWindow(void*);
extern "C" void OSXTriggerFrame(void*);
extern "C" void OSXUpdateTargetFPS(void*);
extern "C" void OSXAddWaitForeverMouse(void*);
extern "C" void OSXDelWaitForeverMouse(void*);
extern "C" void OSXAddWaitForeverKeyboard(void*);
extern "C" void OSXDelWaitForeverKeyboard(void*);
extern "C" void OSXAddWaitForeverSocket(void*, int fd);
#endif

namespace LFL {
Application *app = new Application();
Window *screen = new Window();
thread_local ThreadLocalStorage *ThreadLocalStorage::instance = 0;

DEFINE_bool(lfapp_audio, false, "Enable audio in/out");
DEFINE_bool(lfapp_video, false, "Enable OpenGL");
DEFINE_bool(lfapp_input, false, "Enable keyboard/mouse input");
DEFINE_bool(lfapp_camera, false, "Enable camera capture");
DEFINE_bool(lfapp_cuda, true, "Enable CUDA acceleration");
DEFINE_bool(lfapp_network, false, "Enable asynchronous network engine");
DEFINE_bool(lfapp_debug, false, "Enable debug mode");
DEFINE_bool(cursor_grabbed, false, "Center cursor every frame");
DEFINE_bool(daemonize, false, "Daemonize server");
DEFINE_bool(rcon_debug, false, "Print rcon commands");
DEFINE_string(nameserver, "", "Default namesver");
DEFINE_bool(max_rlimit_core, false, "Max core dump rlimit");
DEFINE_bool(max_rlimit_open_files, false, "Max number of open files rlimit");
DEFINE_int(loglevel, 7, "Log level: [Fatal=-1, Error=0, Info=3, Debug=7]");
DEFINE_int(threadpool_size, 0, "Threadpool size");
DEFINE_int(sample_rate, 16000, "Audio sample rate");
DEFINE_int(sample_secs, 3, "Seconds of RingBuf audio");
DEFINE_int(chans_in, -1, "Audio input channels");
DEFINE_int(chans_out, -1, "Audio output channels");
DEFINE_int(target_fps, 0, "Max frames per second");
DEFINE_bool(open_console, 0, "Open console on win32");

::std::ostream& operator<<(::std::ostream& os, const point &x) { return os << x.DebugString(); }
::std::ostream& operator<<(::std::ostream& os, const Box   &x) { return os << x.DebugString(); }

Printable::Printable(const pair<int, int> &x) : string(StrCat("pair(", x.first, ", ", x.second, ")")) {}
Printable::Printable(const vector<string> &x) : string(StrCat("{", Vec<string>::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<double> &x) : string(StrCat("{", Vec<double>::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<float>  &x) : string(StrCat("{", Vec<float> ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const vector<int>    &x) : string(StrCat("{", Vec<int>   ::Str(&x[0], x.size()), "}")) {}
Printable::Printable(const Color          &x) : string(x.DebugString()) {}
Printable::Printable(const String16       &x) : string(String::ToUTF8(x)) {}
Printable::Printable(const void           *x) : string(StringPrintf("%p", x)) {}
string Flag::GetString() const { string v=Get(); return StrCat(name, v.size()?" = ":"", v.size()?v:"", " : ", desc); } 

bool Running() { return app->run; }
bool MainThread() { return Thread::GetId() == app->main_thread_id; }
void RunInMainThread(Callback *cb) { app->message_queue.Write(MessageQueue::CallbackMessage, cb); }
void Log(int level, const char *file, int line, const string &m) { app->Log(level, file, line, m); }
void DefaultLFAppWindowClosedCB(Window *W) { delete W; }
double FPS() { return screen->fps.FPS(); }
double CamFPS() { return app->camera.fps.FPS(); }
void PressAnyKey() {
    printf("Press [enter] to continue..."); fflush(stdout);
    char buf[32]; fgets(buf, sizeof(buf), stdin);
}
bool FGets(char *buf, int len) { return NBFGets(stdin, buf, len); }
bool NBFGets(FILE *f, char *buf, int len) {
#ifndef WIN32
    int fd = fileno(f);
    SelectSocketSet ss;
    ss.Add(fd, SocketSet::READABLE, 0);
    ss.Select(0);
    if (!app->run || !ss.GetReadable(fd)) return 0;
    fgets(buf, len, f);
    return 1;
#else
    return 0;
#endif
}
int NBRead(int fd, char *buf, int len) {
    SelectSocketSet ss;
    ss.Add(fd, SocketSet::READABLE, 0);
    ss.Select(0);
    if (!app->run || !ss.GetReadable(fd)) return 0;
    int o = 0, s = 0;
    do if ((s = ::read(fd, buf+o, len-o)) > 0) o += s;
    while (s > 0 && len - o > 1024);
    return o;
}
int NBRead(int fd, string *buf) {
    int l = NBRead(fd, (char*)buf->data(), buf->size());
    buf->resize(max(0,l));
    return l;
}
string NBRead(int fd, int len) {
    string ret(len, 0);
    NBRead(fd, &ret);
    return ret;
}

timeval Time2timeval(Time x) {
    struct timeval ret = { (long)ToSeconds(x), 0 };
    ret.tv_usec = ToMicroSeconds(x - Seconds(ret.tv_sec));
    return ret;
}

int isfileslash(int c) { return c == LocalFile::Slash; }
int isdot(int c) { return c == '.'; }
int iscomma(int c) { return c == ','; }
int isand(int c) { return c == '&'; }
int isdquote(int c) { return c == '"'; }
int issquote(int c) { return c == '\''; }
int istick(int c) { return c == '`'; }
int isdig(int c) { return (c >= '0' && c <= '9'); }
int isnum(int c) { return isdig(c) || c == '.'; }
int isquote(int c) { return isdquote(c) || issquote(c) || istick(c); }
int notspace(int c) { return !isspace(c); }
int notalpha(int c) { return !isalpha(c); }
int notalnum(int c) { return !isalnum(c); }
int notnum(int c) { return !isnum(c); }
int notcomma(int c) { return !iscomma(c); }
int notdot(int c) { return !isdot(c); }
float my_atof(const char *v) { return v ? ::atof(v) : 0; }
int atoi(const char *v) { return v ? ::atoi(v) : 0; }
int atoi(const short *v) {
    if (!v) return 0; int ret = 0; const short *p;
    if (!(p = NextChar(v, notspace))) return 0;
    bool neg = *p == '-';
    for (p += neg; *p >= '0' && *p <= '9'; p++) ret = ret*10 + *p - '0';
    return neg ? -ret : ret;
}

int DoubleSort(double a, double b) {
    bool na=isnan(a), nb=isnan(b);

    if (na && nb) return 0;
    else if (na) return 1;
    else if (nb) return -1;

    if      (a < b) return 1;
    else if (a > b) return -1;
    else return 0;
}
int DoubleSort (const void *a, const void *b) { return DoubleSort(*(double*)a, *(double*)b); }
int DoubleSortR(const void *a, const void *b) { return DoubleSort(*(double*)b, *(double*)a); }

const char *Default(const char *in, const char *default_in) { return (in && in[0]) ? in : default_in; }
string   ReplaceEmpty (const string   &in, const string   &replace_with) { return in.empty() ? replace_with : in; }
String16 ReplaceEmpty (const String16 &in, const String16 &replace_with) { return in.empty() ? replace_with : in; }
String16 ReplaceEmpty (const String16 &in, const string   &replace_with) { return in.empty() ? String::ToUTF16(replace_with) : in; }
string ReplaceNewlines(const string   &in, const string   &replace_with) {
    string ret;
    for (const char *p = in.data(); p-in.data() < in.size(); p++) {
        if (*p == '\r' && *(p+1) == '\n') { ret += replace_with; p++; }
        else if (*p == '\n') ret += replace_with;
        else ret += string(p, 1);
    }
    return ret;
}

#define StringPrintfImpl(ret, fmt, vsprintf, buftype, offset) \
    (ret)->resize(offset + 4096); \
    va_list ap, ap2; \
    va_start(ap, fmt); \
    va_copy(ap2, ap); \
    int len = -1; \
    for (int i=0; len < 0 || len >= (ret)->size()-1; ++i) { \
        if (i) { \
            va_copy(ap, ap2); \
            (ret)->resize((ret)->size() * 2); \
        } \
        len = vsprintf((buftype*)((ret)->data() + offset), (ret)->size(), fmt, ap); \
        va_end(ap); \
    } \
    va_end(ap2); \
    (ret)->resize(offset + len);

void StringAppendf(string *out, const char *fmt, ...) {
    int offset = out->size();
    StringPrintfImpl(out, fmt, vsnprintf, char, offset);
}
void StringAppendf(String16 *uc_out, const char *fmt, ...) {
    string outs, *out = &outs;
    StringPrintfImpl(out, fmt, vsnprintf, char, 0);
    String::Append(outs, uc_out);
}

string StringPrintf(const char *fmt, ...) {
    string ret;
    StringPrintfImpl(&ret, fmt, vsnprintf, char, 0);
    return ret;
}
string WStringPrintf(const wchar_t *fmt, ...) {
    string ret;
    StringPrintfImpl(&ret, fmt, vswprintf, wchar_t, 0);
    return ret;
}
String16 String16Printf(const char *fmt, ...) {
    string ret;
    StringPrintfImpl(&ret, fmt, vsnprintf, char, 0);
    return String::ToUTF16(ret);
}

#define StrCatInit(s) string out; out.resize(s); Serializable::MutableStream o((char*)out.data(), out.size());
#define StrCatAdd(x) memcpy(o.Get(x.size()), x.data(), x.size())
#define StrCatReturn() CHECK_EQ(o.error, 0); return out;
string StrCat(const Printable &x1, const Printable &x2) { StrCatInit(x1.size()+x2.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3) { StrCatInit(x1.size()+x2.size()+x3.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()+x13.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatAdd(x13); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13, const Printable &x14) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()+x13.size()+x14.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatAdd(x13); StrCatAdd(x14); StrCatReturn(); }
string StrCat(const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13, const Printable &x14, const Printable &x15) { StrCatInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()+x13.size()+x14.size()+x15.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatAdd(x13); StrCatAdd(x14); StrCatAdd(x15); StrCatReturn(); }

#define StrAppendInit(s); out->resize(out->size()+(s)); Serializable::MutableStream o((char*)out->data()+out->size()-(s), (s));
#define StrAppendReturn() CHECK_EQ(o.error, 0)
void StrAppend(string *out, const Printable &x1) { (*out) += x1; }
void StrAppend(string *out, const Printable &x1, const Printable &x2) { StrAppendInit(x1.size()+x2.size()); StrCatAdd(x1); StrCatAdd(x2); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3) { StrAppendInit(x1.size()+x2.size()+x3.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrAppendReturn(); }
void StrAppend(string *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13) { StrAppendInit(x1.size()+x2.size()+x3.size()+x4.size()+x5.size()+x6.size()+x7.size()+x8.size()+x9.size()+x10.size()+x11.size()+x12.size()+x13.size()); StrCatAdd(x1); StrCatAdd(x2); StrCatAdd(x3); StrCatAdd(x4); StrCatAdd(x5); StrCatAdd(x6); StrCatAdd(x7); StrCatAdd(x8); StrCatAdd(x9); StrCatAdd(x10); StrCatAdd(x11); StrCatAdd(x12); StrCatAdd(x13); StrAppendReturn(); }

void StrAppend(String16 *out, const Printable &x1) { String::Append(StrCat(x1), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2) { String::Append(StrCat(x1,x2), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3) { String::Append(StrCat(x1,x2,x3), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4) { String::Append(StrCat(x1,x2,x3,x4), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5) { String::Append(StrCat(x1,x2,x3,x4,x5), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6) { String::Append(StrCat(x1,x2,x3,x4,x5,x6), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12), out); }
void StrAppend(String16 *out, const Printable &x1, const Printable &x2, const Printable &x3, const Printable &x4, const Printable &x5, const Printable &x6, const Printable &x7, const Printable &x8, const Printable &x9, const Printable &x10, const Printable &x11, const Printable &x12, const Printable &x13) { String::Append(StrCat(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13), out); }

int Split(const string &in, int (*ischar)(int), string *left, string *right) { return Split(in.c_str(), ischar, left, right); }
int Split(const char   *in, int (*ischar)(int), string *left, string *right) {
    const char *p = in;
    for (/**/; *p && !ischar(*p); p++);
    if (!*p) { *left=in; *right=""; return 1; }
    left->assign(in, p-in);

    for (/**/; *p && ischar(*p); p++);
    if (!*p) { *right=""; return 1; }
    right->assign(p);
    return 2;
}

void Join(string *out, const vector<string> &in) { return StrAppend(out, in, 0, in.size()); }
void Join(string *out, const vector<string> &in, int inB, int inE) {
    int size = 0;        for (int i = inB; i < inE; i++) size += in[i].size();
    StrAppendInit(size); for (int i = inB; i < inE; i++) { StrCatAdd(in[i]); } StrAppendReturn();
}
string Join(const vector<string> &strs, const string &separator) {
    string ret;
    for (vector<string>::const_iterator i = strs.begin(); i != strs.end(); i++) StrAppend(&ret, ret.size()?separator:"", *i);
    return ret;
}
string Join(const vector<string> &strs, const string &separator, int beg_ind, int end_ind) {
    string ret;
    for (int i = beg_ind; i < strs.size() && i < end_ind; i++) StrAppend(&ret, ret.size()?separator:"", strs[i]);
    return ret;
}

template <class X, class Y>
bool PrefixMatch(const X* in, const Y* pref, int cs) {
    while (*in && *pref &&
           ((cs && *in == *pref) || 
            (!cs && ::tolower(*in) == ::tolower(*pref)))) { in++; pref++; }
    return !*pref;
}
bool PrefixMatch(const char     *in, const string   &pref, int cs) { return PrefixMatch<char,  char> (in,         pref.c_str(), cs); }
bool PrefixMatch(const string   &in, const char     *pref, int cs) { return PrefixMatch<char,  char> (in.c_str(), pref,         cs); }
bool PrefixMatch(const string   &in, const string   &pref, int cs) { return PrefixMatch<char,  char> (in.c_str(), pref.c_str(), cs); }
bool PrefixMatch(const String16 &in, const String16 &pref, int cs) { return PrefixMatch<short, short>(in.c_str(), pref.c_str(), cs); }
bool PrefixMatch(const String16 &in, const char     *pref, int cs) { return PrefixMatch<short, char> (in.c_str(), pref, cs); }
bool PrefixMatch(const char     *in, const char     *pref, int cs) { return PrefixMatch<char,  char> (in, pref, cs); }
bool PrefixMatch(const short    *in, const short    *pref, int cs) { return PrefixMatch<short, short>(in, pref, cs); }

template <class X, class Y>
bool SuffixMatch(const X *in, int inlen, const Y *pref, int preflen, int cs) {
    if (inlen < preflen) return 0;
    const X *in_suffix = in + inlen - preflen;
    for (in += inlen-1, pref += preflen-1;
         in >= in_suffix &&
         ((cs && *in == *pref) ||
          (!cs && ::tolower(*in) == ::tolower(*pref))); in--, pref--) {}
    return in < in_suffix;
}
bool SuffixMatch(const short    *in, const short    *pref, int cs) { return SuffixMatch(String16(in), String16(pref), cs); }
bool SuffixMatch(const char     *in, const char     *pref, int cs) { return SuffixMatch(string(in),   string(pref),   cs); }
bool SuffixMatch(const char     *in, const string   &pref, int cs) { return SuffixMatch(string(in),   pref,           cs); }
bool SuffixMatch(const string   &in, const char     *pref, int cs) { return SuffixMatch(in,           string(pref),   cs); }
bool SuffixMatch(const string   &in, const string   &pref, int cs) { return SuffixMatch<char,  char> (in.data(), in.size(), pref.data(), pref.size(), cs); }
bool SuffixMatch(const String16 &in, const String16 &pref, int cs) { return SuffixMatch<short, short>(in.data(), in.size(), pref.data(), pref.size(), cs); }
bool SuffixMatch(const String16 &in, const string   &pref, int cs) { return SuffixMatch<short, char> (in.data(), in.size(), pref.data(), pref.size(), cs); }

template <class X, class Y>
bool StringEquals(const X *s1, const Y *s2, int cs) {
    while (*s1 && *s2 &&
           ((cs && *s1 == *s2) ||
            (!cs && ::tolower(*s1) == ::tolower(*s2)))) { s1++; s2++; }
    return !*s1 && !*s2;
}
bool StringEquals(const String16 &s1, const String16 &s2, int cs) { return s1.size() == s2.size() && StringEquals(s1.c_str(), s2.c_str(), cs); }
bool StringEquals(const string   &s1, const string   &s2, int cs) { return s1.size() == s2.size() && StringEquals(s1.c_str(), s2.c_str(), cs); }
bool StringEquals(const string   &s1, const char     *s2, int cs) { return                           StringEquals(s1.c_str(), s2,         cs); }
bool StringEquals(const char     *s1, const string   &s2, int cs) { return                           StringEquals(s1,         s2.c_str(), cs); }
bool StringEquals(const char     *s1, const char     *s2, int cs) { return cs ? !strcmp(s1, s2) : !strcasecmp(s1, s2); }
bool StringEquals(const short    *s1, const short    *s2, int cs) { return StringEquals<short, short>(s1, s2, cs); }
bool StringEquals(const String16 &s1, const char     *s2, int cs) { return StringEquals<short, char>(s1.c_str(), s2, cs); }

bool StringEmptyOrEquals(const string   &cmp, const string   &ref, int cs) { return cmp.empty() || StringEquals(cmp, ref, cs); }
bool StringEmptyOrEquals(const String16 &cmp, const String16 &ref, int cs) { return cmp.empty() || StringEquals<short, short>(cmp.c_str(), ref.c_str(), cs); }
bool StringEmptyOrEquals(const String16 &cmp, const string   &ref, int cs) { return cmp.empty() || StringEquals<short, char >(cmp.c_str(), ref.c_str(), cs); }
bool StringEmptyOrEquals(const string   &cmp, const string   &ref1, const string   &ref2, int cs) { return cmp.empty() || StringEquals              (cmp,         ref1,         cs) || StringEquals              (cmp,         ref2,         cs); }
bool StringEmptyOrEquals(const String16 &cmp, const String16 &ref1, const String16 &ref2, int cs) { return cmp.empty() || StringEquals<short, short>(cmp.c_str(), ref1.c_str(), cs) || StringEquals<short, short>(cmp.c_str(), ref2.c_str(), cs); }
bool StringEmptyOrEquals(const String16 &cmp, const string   &ref1, const string   &ref2, int cs) { return cmp.empty() || StringEquals<short, char >(cmp.c_str(), ref1.c_str(), cs) || StringEquals<short, char >(cmp.c_str(), ref2.c_str(), cs); }

bool StringReplace(string *text, const string &needle, const string &replace) {
    int pos = text->find(needle);
    if (pos == string::npos) return false;
    text->erase(pos, needle.size());
    text->insert(pos, replace);
    return true;
}

int sprint(char *out, int len, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(out,len,fmt,ap);
    if (ret > len) ret = len;
    va_end(ap);
    return ret;
}

template <class X> int IsNewline(const X *line) {
    if (!*line) return 0;
    if (*line == '\n') return 1;
    if (*line == '\r' && *(line+1) == '\n') return 2;
    return 0;
}
template int IsNewline(const char *line);

template <class X> int ChompNewline(X *line, int len) {
    int ret = 0;
    if (line[len-1] == '\n') { line[len-1] = 0; ret++; }
    if (line[len-2] == '\r') { line[len-2] = 0; ret++; }
    return ret;
}
template int ChompNewline(char *line, int len);

template <class X> int ChompNewlineLength(const X *line, int len) {
    int ret = 0;
    if (line[len-1] == '\n') ret++;
    if (line[len-2] == '\r') ret++;
    return ret;
}

template <class X> int DirNameLen(const StringPieceT<X> &path, bool include_slash) {
    int len = path.Length();
    const X *start = path.buf + len - 1, *slash = 0;
    for (const X *p = start; p > path.buf; --p) if (isfileslash(*p)) { slash=p; break; }
    return !slash ? 0 : len - (start - slash + !include_slash);
}
int DirNameLen(const StringPiece   &text, bool include_slash) { return DirNameLen<char> (text, include_slash); }
int DirNameLen(const String16Piece &text, bool include_slash) { return DirNameLen<short>(text, include_slash); }

int BaseDir(const char *path, const char *cmp) {
    int l1=strlen(path), l2=strlen(cmp), slash=0, s1=-1, s2=0;
    if (!l1 || l2 > l1) return 0;
    if (*(path+l1-1) == '/') return 0;
    for (const char *p=path+l1-1; p>=path; p--) if (isfileslash(*p)) {
        slash++; /* count bacwards */
        if (slash == 1) s2 = p-path;
        if (slash == 2) { s1 = p-path; break; }
    }
    if (slash < 1 || s2-s1-1 != l2) return 0;
    return !strncasecmp(&path[s1]+1, cmp, l2);
}

const char *ParseProtocol(const char *url, string *protO) {
    static const int hdr_size = 3;
    static const char hdr[] = "://";
    const char *prot_end = strstr(url, hdr), *prot, *host;
    if (prot_end) { prot = url; host = prot_end + hdr_size; }
    else          { prot = 0;   host = url;                 }
    while (prot && *prot && isspace(*prot)) prot++;
    while (host && *host && isspace(*host)) host++;
    if (protO) protO->assign(prot ? prot : "", prot ? prot_end-prot : 0);
    return host;
}

const char *BaseName(const StringPiece &path, int *outlen) {
    const char *ret = path.buf;
    int len = path.Length();
    for (const char *p = path.buf+len-1; p > path.buf; --p) if (isfileslash(*p)) { ret=p+1; break;}
    if (outlen) {
        int namelen = len - (ret-path.buf), baselen;
        NextChar(ret, isdot, namelen, &baselen);
        *outlen = baselen ? baselen : namelen;
    }
    return ret;
}

template <int S> const char *NextChunk(const StringPiece &text, bool final, int *outlen) {
    int add = final ? max(text.len, 0) : S;
    if (text.len < add) return 0;
    *outlen = add;
    return text.buf + add;
}

const char *NextProto(const StringPiece &text, bool final, int *outlen) {
    if (text.len < ProtoHeader::size) return 0;
    ProtoHeader hdr(text.buf);
    if (ProtoHeader::size + hdr.len > text.len) return 0;
    *outlen = hdr.len;
    return text.buf + ProtoHeader::size + hdr.len;
}

template <class X, bool chomp> const X *NextLine(const StringPieceT<X> &text, bool final, int *outlen) {
    const X *ret=0, *p = text.buf;
    for (/**/; !text.Done(p); ++p) { 
        if (*p == '\n') { ret = p+1; break; }
    }
    if (!ret) { if (outlen) *outlen = p - text.buf; return final ? text.buf : 0; }
    if (outlen) {
        int ol = ret-text.buf-1;
        if (chomp && ret-2 >= text.buf && *(ret-2) == '\r') ol--;
        *outlen = ol;
    }
    return ret;
}
const char  *NextLine   (const StringPiece   &text, bool final, int *outlen) { return NextLine<char,  true >(text, final, outlen); }
const short *NextLine   (const String16Piece &text, bool final, int *outlen) { return NextLine<short, true >(text, final, outlen); }
const char  *NextLineRaw(const StringPiece   &text, bool final, int *outlen) { return NextLine<char,  false>(text, final, outlen); }
const short *NextLineRaw(const String16Piece &text, bool final, int *outlen) { return NextLine<short, false>(text, final, outlen); }

template <class X>       X *NextChar(      X *text, int (*ischar)(int), int len, int *outlen) { return (X*)NextChar((const X *)text, ischar, len, outlen); }
template <class X> const X *NextChar(const X *text, int (*ischar)(int), int len, int *outlen) { return NextChar(text, ischar, 0, len, outlen); }
template <class X>       X *NextChar(      X *text, int (*ischar)(int), int (*isquotec)(int), int len, int *outlen) { return (X*)NextChar((const X *)text, ischar, isquotec, len, outlen); }
template <class X> const X *NextChar(const X *text, int (*ischar)(int), int (*isquotec)(int), int len, int *outlen) {
    const X *ret=0, *p;
    bool have_len = len >= 0, in_quote = false;
    for (p=text; (have_len ? p-text<len : *p); p++) {
        if (!in_quote && ischar(*p)) { ret=p; break; }
        if (isquotec && isquotec(*p)) in_quote = !in_quote;
    }
    if (outlen) *outlen = ret ? ret-text : p-text;
    return ret;
}
template       char*  NextChar<char >(      char*,  int (*)(int), int, int*);
template const char*  NextChar<char >(const char*,  int (*)(int), int, int*);
template       short* NextChar<short>(      short*, int (*)(int), int, int*);
template const short* NextChar<short>(const short*, int (*)(int), int, int*);

template <class X> int LengthChar(const StringPieceT<X> &text, int (*ischar)(int)) {
    const X *p;
    for (p = text.buf; !text.Done(p); ++p) if (!ischar(*p)) break;
    return p - text.buf;
}
template int LengthChar(const StringPiece  &, int (*)(int));
template int LengthChar(const String16Piece&, int (*)(int));

template <class X> int RLengthChar(const StringPieceT<X> &text, int (*ischar)(int)) {
    int len = text.Length();
    if (len <= 0) return 0;
    const X *p;
    for (p = text.buf; text.buf - p < len; --p) if (!ischar(*p)) break;
    return text.buf - p;
}
template int RLengthChar(const StringPiece  &, int (*)(int));
template int RLengthChar(const String16Piece&, int (*)(int));

string strip(const char *s, int (*stripchar1)(int), int (*stripchar2)(int)) {
    string ret;
    for (/**/; *s; s++) if ((!stripchar1 || !stripchar1(*s)) && (!stripchar2 || !stripchar2(*s))) ret += *s;
    return ret;
}

string togrep(const char *s, int (*grepchar1)(int), int (*grepchar2)(int)) {
    string ret;
    for (/**/; *s; s++) if ((!grepchar1 || grepchar1(*s)) || (!grepchar2 || grepchar2(*s))) ret += *s;
    return ret;
}

template <class X>
basic_string<X> toconvert(const X *s, int (*tochar)(int), int (*ischar)(int)) {
    basic_string<X> input = s;
    for (int i=0; i<input.size(); i++)
        if (!ischar || ischar(input[i]))
            input[i] = tochar(input[i]);

    return input;
}
string toconvert  (const string   &s, int (*tochar)(int), int (*ischar)(int)) { return toconvert<char> (s.c_str(), tochar, ischar); }
string toconvert  (const char     *s, int (*tochar)(int), int (*ischar)(int)) { return toconvert<char> (s,         tochar, ischar); }
String16 toconvert(const String16 &s, int (*tochar)(int), int (*ischar)(int)) { return toconvert<short>(s.c_str(), tochar, ischar); }
String16 toconvert(const short    *s, int (*tochar)(int), int (*ischar)(int)) { return toconvert<short>(s,         tochar, ischar); }

string   toupper(const char     *s) { return toconvert(s        , ::toupper, isalpha); }
string   toupper(const string   &s) { return toconvert(s.c_str(), ::toupper, isalpha); }
String16 toupper(const short    *s) { return toconvert(s        , ::toupper, isalpha); }
String16 toupper(const String16 &s) { return toconvert(s.c_str(), ::toupper, isalpha); }

string   tolower(const char     *s) { return toconvert(s        , ::tolower, isalpha); }
string   tolower(const string   &s) { return toconvert(s.c_str(), ::tolower, isalpha); }
String16 tolower(const short    *s) { return toconvert(s        , ::tolower, isalpha); }
String16 tolower(const String16 &s) { return toconvert(s.c_str(), ::tolower, isalpha); }

string CHexEscape(const string &text) {
    string ret;
    for (int i=0; i<text.size(); ++i) StringAppendf(&ret, "\\x%02x", (unsigned char)text[i]);
    return ret;
}

unsigned fnv32(const void *buf, unsigned len, unsigned hval) {
    if (!len) len = strlen((const char *)buf);
    unsigned char *bp = (unsigned char *)buf, *be = bp + len;
    while (bp < be) {
        hval += (hval<<1) + (hval<<4) + (hval<<7) + (hval<<8) + (hval<<24);
        hval ^= (unsigned)*bp++;
    }
    return hval;
}

unsigned long long fnv64(const void *buf, unsigned len, unsigned long long hval) {
    if (!len) len = strlen((const char *)buf);
    unsigned char *bp = (unsigned char *)buf, *be = bp + len;
    while (bp < be) {
        hval += (hval<<1) + (hval<<4) + (hval<<5) + (hval<<7) + (hval<<8) + (hval<<40);
        hval ^= (unsigned long long)*bp++;
    }
    return hval;
}

#if defined (WIN32) || defined(LFL_ANDROID)
int is_leap(unsigned y) { y += 1900; return (y % 4) == 0 && ((y % 100) != 0 || (y % 400) == 0); }

time_t timegm(struct tm *tm) {
    static const unsigned ndays[2][12] = {
        {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
        {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
    };

    time_t res = 0;
    for (int i = 70; i < tm->tm_year; ++i) res += is_leap(i) ? 366 : 365;
    for (int i = 0; i < tm->tm_mon; ++i) res += ndays[is_leap(tm->tm_year)][i];

    res += tm->tm_mday - 1;
    res *= 24;
    res += tm->tm_hour;
    res *= 60;
    res += tm->tm_min;
    res *= 60;
    res += tm->tm_sec;
    return res;
}
#endif

const char *dayname(int wday) {
    static const char *dn[] = { "Sun", "Mon", "Tue", "Wed", "Thr", "Fri", "Sat" };
    if (wday < 0 || wday >= 7) return 0;
    return dn[wday];
}

int RFC822Day(const char *text) {
    static const char *dn[] = { "Sun", "Mon", "Tue", "Wed", "Thr", "Fri", "Sat" };
    for (int i=0, l=sizeofarray(dn); i<l; i++) if (!strcmp(text, dn[i])) return i;
    return 0;
}

const char *monthname(int mon) {
    static const char *mn[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
    if (mon < 0 || mon >= 12) return 0;
    return mn[mon];
}

int RFC822Month(const char *text) {
    static const char *mn[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
    for (int i=0, l=sizeofarray(mn); i<l; i++) if (!strcmp(text, mn[i])) return i;
    return 0;
}

int RFC822TimeZone(const char *text) {
    static const char *tzname[] = { "GMT", "EST", "EDT", "CST", "CDT", "MST", "MDT", "PST", "PDT", };
    static const int tzoffset[] = { 0,     -5,    -4,    -6,    -5,    -7,    -6,    -8,    -7     };
    for (int i=0, l=sizeofarray(tzname); i<l; i++) if (!strcmp(text, tzname[i])) return tzoffset[i];
    return 0;
}

void GMTtm(time_t in, struct tm *t) { gmtime_r(&in, t); }
void GMTtm(struct tm *t) { return GMTtm(time(0), t); }

void localtm(time_t in, struct tm *t) { localtime_r(&in, t);}
void localtm(struct tm *t) { return localtm(time(0), t); }

string logtime(Time t) { char buf[128] = {0}; logtime(t, buf, sizeof(buf)); return buf; }
int logtime(char *buf, int size) { return logtime(Now(), buf, size); }
int logtime(Time t, char *buf, int size) { time_t tt=t/1000; return logtime(tt, t-tt*1000, buf, size); }
int logtime(time_t secs, int ms, char *buf, int size) { struct tm tm; localtm(secs, &tm); return logtime(&tm, ms, buf, size); }
int logtime(struct tm *tm, int ms, char *buf, int size) {
    return snprintf(buf, size, "%02d:%02d:%02d.%03d", tm->tm_hour, tm->tm_min, tm->tm_sec, ms);
}

string logfileday(Time t) { char buf[128] = {0}; logfileday(Time2time_t(t), buf, sizeof(buf)); return buf; }
int logfileday(char *buf, int size) { return logfileday(time(0), buf, size); }
int logfileday(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return logfileday(&tm, buf, size); }
int logfileday(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%04d-%02d-%02d", 1900+tm->tm_year, tm->tm_mon+1, tm->tm_mday);
}

string logfiledaytime(Time t) { char buf[128] = {0}; logfiledaytime(Time2time_t(t), buf, sizeof(buf)); return buf; }
int logfiledaytime(char *buf, int size) { return logfiledaytime(time(0), buf, size); }
int logfiledaytime(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return logfiledaytime(&tm, buf, size); }
int logfiledaytime(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%04d-%02d-%02d_%02d:%02d", 1900+tm->tm_year, tm->tm_mon+1, tm->tm_mday, tm->tm_hour, tm->tm_min);
}

int httptime(char *buf, int size) { return httptime(time(0), buf, size); }
int httptime(time_t t, char *buf, int size) { struct tm tm; GMTtm(t, &tm); return httptime(&tm, buf, size); }
int httptime(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%s, %d %s %d %02d:%02d:%02d GMT",
        dayname(tm->tm_wday), tm->tm_mday, monthname(tm->tm_mon), 1900+tm->tm_year,
        tm->tm_hour, tm->tm_min, tm->tm_sec);
}

string localhttptime(Time t) { char buf[128] = {0}; localhttptime(Time2time_t(t), buf, sizeof(buf)); return buf; }
int localhttptime(char *buf, int size) { return localhttptime(time(0), buf, size); }
int localhttptime(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return localhttptime(&tm, buf, size); }
int localhttptime(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%s, %d %s %d %02d:%02d:%02d %s",
        dayname(tm->tm_wday), tm->tm_mday, monthname(tm->tm_mon), 1900+tm->tm_year,
        tm->tm_hour, tm->tm_min, tm->tm_sec,
#ifdef WIN32
        "");
#else
        tm->tm_zone);
#endif
}

string localsmtptime(Time t) { char buf[128] = {0}; localsmtptime(Time2time_t(t), buf, sizeof(buf)); return buf; }
int localsmtptime(char *buf, int size) { return localsmtptime(time(0), buf, size); }
int localsmtptime(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return localsmtptime(&tm, buf, size); }
int localsmtptime(struct tm *tm, char *buf, int size) {
    int tzo = 
#ifdef WIN32
        0;
#else
        RFC822TimeZone(tm->tm_zone)*100;
#endif
    return snprintf(buf, size, "%s, %02d %s %d %02d:%02d:%02d %s%04d",
        dayname(tm->tm_wday), tm->tm_mday, monthname(tm->tm_mon), 1900+tm->tm_year,
        tm->tm_hour, tm->tm_min, tm->tm_sec, tzo<0?"-":"", abs(tzo));
}

string localmboxtime(Time t) { char buf[128] = {0}; localmboxtime(Time2time_t(t), buf, sizeof(buf)); return buf; }
int localmboxtime(char *buf, int size) { return localmboxtime(time(0), buf, size); }
int localmboxtime(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return localmboxtime(&tm, buf, size); }
int localmboxtime(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%s %s%s%d %02d:%02d:%02d %d",
        dayname(tm->tm_wday), monthname(tm->tm_mon), tm->tm_mday < 10 ? "  " : " ",
        tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec, 1900+tm->tm_year);
}

string intervaltime(Time t) { time_t tt=t/1000; char buf[64] = {0}; intervaltime(tt, t-tt*1000, buf, sizeof(buf)); return buf; }
int intervaltime(time_t t, int ms, char *buf, int size) {
    int hours = t/3600;
    t -= hours*3600;
    int minutes = t/60;
    int seconds = t - minutes*60;
    return snprintf(buf, size, "%02d:%02d:%02d.%03d", hours, minutes, seconds, ms);
}

string intervalminutes(Time t) { time_t tt=t/1000; char buf[64] = {0}; intervalminutes(tt, t-tt*1000, buf, sizeof(buf)); return buf; }
int intervalminutes(time_t t, int ms, char *buf, int size) {
    int minutes = t/60;
    int seconds = t - minutes*60;
    return snprintf(buf, size, "%02d:%02d", minutes, seconds);
}

bool RFC822Time(const char *text, int *hour, int *min, int *sec) {
    int textlen = strlen(text);
    if (textlen < 5 || text[2] != ':') return false;
    if (hour) *hour = atoi(text);
    if (min) *min = atoi(&text[3]);
    if (textlen == 5) { 
        if (sec) *sec = 0;
        return true;
    }
    if (textlen != 8 || text[5] != ':') return false;
    if (sec) *sec = atoi(&text[6]);
    return true;
}

Time RFC822Date(const char *text) {
    struct tm tm; memset(&tm, 0, sizeof(tm));
    const char *comma = strchr(text, ','), *start = comma ? comma + 1 : text, *parsetext;
    StringWordIter words(start);
    tm.tm_mday = atoi(BlankNull(words.Next()));
    tm.tm_mon = RFC822Month(BlankNull(words.Next()));
    tm.tm_year = atoi(BlankNull(words.Next())) - 1900;
    const char *timetext = BlankNull(words.Next());
    if (!RFC822Time(timetext, &tm.tm_hour, &tm.tm_min, &tm.tm_sec))
        { ERROR("RFC822Date('", text, "') RFC822Time('", timetext, "') failed"); return 0; }
    int hours_from_gmt = RFC822TimeZone(BlankNull(words.Next()));
    return (timegm(&tm) - hours_from_gmt * 3600) * 1000;
}

bool NumericTime(const char *text, int *hour, int *min, int *sec) {
    int textlen = strlen(text);
    StringWordIter words(StringPiece(text, textlen), isint<':'>);
    *hour = atoi(BlankNull(words.Next()));
    *min = atoi(BlankNull(words.Next()));
    *sec = atoi(BlankNull(words.Next()));
    if (textlen >= 2 && !strcmp(text+textlen-2, "pm") && *hour != 12) *hour += 12;
    return true;
}

Time NumericDate(const char *datetext, const char *timetext, const char *timezone) {
    struct tm tm; memset(&tm, 0, sizeof(tm));
    StringWordIter words(datetext, isint<'/'>);
    tm.tm_mon = atoi(BlankNull(words.Next())) - 1;
    tm.tm_mday = atoi(BlankNull(words.Next()));
    tm.tm_year = atoi(BlankNull(words.Next())) - 1900;
    NumericTime(timetext, &tm.tm_hour, &tm.tm_min, &tm.tm_sec);
    int hours_from_gmt = RFC822TimeZone(BlankNull(timezone));
    return (timegm(&tm) - hours_from_gmt * 3600) * 1000;
}

Time SinceDayBegan(Time t, int gmt_offset_hrs) {
    Time ret = (t % Hours(24)) + Hours(gmt_offset_hrs);
    return ret < 0 ? ret + Hours(24) : ret;
}

bool    IsDaylightSavings(Time t) { struct tm tm; localtm(t ? Time2time_t(t) : time(0), &tm); return tm.tm_isdst; }
#ifndef WIN32
const char *LocalTimeZone(Time t) { struct tm tm; localtm(t ? Time2time_t(t) : time(0), &tm); return tm.tm_zone; }
#else
const char *LocalTimeZone(Time t) { return _tzname[_daylight]; }
#endif

void Allocator::Reset() { FATAL(Name(), ": reset"); }

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
    if (cur_block_ind == -1 || blocks[cur_block_ind].len + n > blocks[cur_block_ind].size) {
        cur_block_ind++;
        if (cur_block_ind >= blocks.size()) blocks.push_back(Block(new char[block_size], block_size));
        CHECK_EQ(blocks[cur_block_ind].len, 0);
        CHECK_LT(n, block_size);
    }
    Block *b = &blocks[cur_block_ind];
    char *ret = b->buf + b->len;
    b->len += n;
    return ret;
}

#ifdef WIN32
MainCB nt_service_main = 0;
const char *nt_service_name = 0;
SERVICE_STATUS_HANDLE nt_service_status_handle = 0;

void WIN32_Init() { timeBeginPeriod(1); }
Time Now() { return timeGetTime(); }
void Msleep(int milliseconds) { Sleep(milliseconds); }
int close(int socket) { return closesocket(socket); }
BOOL WINAPI CtrlHandler(DWORD sig) { INFO("interrupt"); app->run=0; return 1; }

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

const char LocalFile::Slash = '\\';
int LocalFile::isdirectory(const string &filename) {
    DWORD attr = ::GetFileAttributes(filename.c_str());
    if (attr == INVALID_FILE_ATTRIBUTES) { ERROR("GetFileAttributes(", filename, ") failed: ", strerror(errno)); return 0; }
    return attr & FILE_ATTRIBUTE_DIRECTORY;
}

void Process::Daemonize(const char *dir) {}
int Process::OpenPTY(const char **argv) { return Open(argv); }
int Process::Open(const char **argv) {
    SECURITY_ATTRIBUTES sa;
    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = 1;
    HANDLE pipeinR, pipeinW, pipeoutR, pipeoutW, h;
    if (!CreatePipe(&pipeinR, &pipeinW, &sa, 0)) return -1;
    if (!CreatePipe(&pipeoutR, &pipeoutW, &sa, 0)) { CloseHandle(pipeinR); CloseHandle(pipeinW); return -1; }

    STARTUPINFO si;
    memset(&si, 0, sizeof(si));
    si.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    si.wShowWindow = SW_HIDE;
    si.hStdInput = pipeoutR;
    si.hStdOutput = pipeinW;
    si.hStdError = pipeinW;

    PROCESS_INFORMATION pi;
    if (!CreateProcess(0, (LPSTR)argv[0], 0, 0, 1, CREATE_NEW_PROCESS_GROUP, 0, 0, &si, &pi)) return -1;
    CloseHandle(pi.hThread);
    CloseHandle(pipeinW);
    CloseHandle(pipeoutR);

    in = fdopen(_open_osfhandle((long)pipeinR, O_TEXT), "r");
    out = fdopen(_open_osfhandle((long)pipeoutW, O_TEXT), "w");
    return 0;
}

BOOL UpdateSCMStatus(DWORD dwCurrentState, DWORD dwWin32ExitCode,
                     DWORD dwServiceSpecificExitCode, DWORD dwCheckPoint,
                     DWORD dwWaitHint) {
    SERVICE_STATUS serviceStatus;
    serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    serviceStatus.dwCurrentState = dwCurrentState;
    serviceStatus.dwServiceSpecificExitCode = dwServiceSpecificExitCode;
    serviceStatus.dwCheckPoint = dwCheckPoint;
    serviceStatus.dwWaitHint = dwWaitHint;

    if (dwCurrentState == SERVICE_START_PENDING) serviceStatus.dwControlsAccepted = 0;
    else serviceStatus.dwControlsAccepted = SERVICE_ACCEPT_STOP |SERVICE_ACCEPT_SHUTDOWN;

    if (dwServiceSpecificExitCode == 0) serviceStatus.dwWin32ExitCode = dwWin32ExitCode;
    else serviceStatus.dwWin32ExitCode = ERROR_SERVICE_SPECIFIC_ERROR;

    return SetServiceStatus(nt_service_status_handle, &serviceStatus);
}

void HandleNTServiceControl(DWORD controlCode) {
    if (controlCode == SERVICE_CONTROL_SHUTDOWN || controlCode == SERVICE_CONTROL_STOP) {
        UpdateSCMStatus(SERVICE_STOPPED, NO_ERROR, 0, 0, 0);
        app->run = 0;
    } else {
        UpdateSCMStatus(SERVICE_RUNNING, NO_ERROR, 0, 0, 0);
    }
}

int DispatchNTServiceMain(int argc, char **argv) {
    nt_service_status_handle = RegisterServiceCtrlHandler(nt_service_name, (LPHANDLER_FUNCTION)HandleNTServiceControl);
    if (!nt_service_status_handle) { ERROR("RegisterServiceCtrlHandler: ", GetLastError()); return -1; }

    if (!UpdateSCMStatus(SERVICE_RUNNING, NO_ERROR, 0, 0, 0)) {
        ERROR("UpdateSCMStatus: ", GetLastError()); return -1;
    }
    
    return nt_service_main(argc, (const char **)argv);
}

int NTService::Install(const char *name, const char *path) {
    SC_HANDLE schSCManager = OpenSCManager(0, 0, SC_MANAGER_CREATE_SERVICE);
    if (!schSCManager) { ERROR("OpenSCManager: ", GetLastError()); return -1; }

    SC_HANDLE schService = CreateService( 
        schSCManager,    	  /* SCManager database      */ 
        name,			      /* name of service         */ 
        name,                 /* service name to display */ 
        SERVICE_ALL_ACCESS,   /* desired access          */ 
        SERVICE_WIN32_SHARE_PROCESS|SERVICE_INTERACTIVE_PROCESS, 
        SERVICE_DEMAND_START, /* start type              */ 
        SERVICE_ERROR_NORMAL, /* error control type      */ 
        path,			      /* service's binary        */ 
        0,                    /* no load ordering group  */ 
        0,                    /* no tag identifier       */ 
        0,                    /* no dependencies         */ 
        0,                    /* LocalSystem account     */ 
        0);                   /* no password             */
    if (!schService) { ERROR("CreateService: ", GetLastError()); return -1; }

    INFO("service ", name, " installed - see Control Panel > Services");
    CloseServiceHandle(schSCManager);
    return 0;
}

int NTService::Uninstall(const char *name) {
    SC_HANDLE schSCManager = OpenSCManager(0, 0, SC_MANAGER_CREATE_SERVICE);
    if (!schSCManager) { ERROR("OpenSCManager: ", GetLastError()); return -1; }

    SC_HANDLE schService = OpenService(schSCManager, name, SERVICE_ALL_ACCESS);
    if (!schService) { ERROR("OpenService: ", GetLastError()); return -1; }

    if (!DeleteService(schService)) { ERROR("DeleteService: ", GetLastError()); return -1; }

    INFO("service ", name, " uninstalled");
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);
    return 0;
}

int NTService::WrapMain(const char *name, MainCB main_cb, int argc, const char **argv) {
    nt_service_name = name;
    nt_service_main = main_cb;

    SERVICE_TABLE_ENTRY serviceTable[] = {
        { (LPSTR)name, (LPSERVICE_MAIN_FUNCTION)DispatchNTServiceMain},
        { 0, 0 }
    };

    if (!StartServiceCtrlDispatcher(serviceTable)) {
        ERROR("StartServiceCtrlDispatcher ", GetLastError());
        return -1;
    }
    return 0;
}

#else /* WIN32 */

void OpenConsole() {}
void CloseConsole() {}
void Msleep(int milliseconds) { usleep(milliseconds * 1000); }
void HandleSigInt(int sig) { app->run=0; app->scheduler.Wakeup(); }
Time Now() { struct timeval tv; gettimeofday(&tv, 0); return (Time)tv.tv_sec * 1000 + tv.tv_usec / 1000; }

const char LocalFile::Slash = '/';
int LocalFile::IsDirectory(const string &filename) {
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID) /* XXX */
    struct stat buf;
    if (stat(filename.c_str(), &buf)) { ERROR("stat(", filename, ") failed: ", strerror(errno)); return 0; }
    return buf.st_mode & S_IFDIR;
#endif
}

int Process::Open(const char **argv) {
    int pipein[2], pipeout[2], ret;
    if (pipe(pipein) < 0) return -1;
    if (pipe(pipeout) < 0) { close(pipein[0]); close(pipein[1]); return -1; }

    if ((ret = fork())) { 
        close(pipein[1]);
        close(pipeout[0]);
        if (ret < 0) { close(pipein[0]); close(pipeout[1]); return -1; }
        in = fdopen(pipein[0], "r");
        out = fdopen(pipeout[1], "w");
    } else {
        close(pipein[0]);
        close(pipeout[1]);
        close(0); close(1); close(2);
        dup2(pipein[1], 2);
        dup2(pipein[1], 1);
        dup2(pipeout[0], 0);
        execvp(argv[0], (char*const*)argv);
    }
    return 0;
}

extern "C" pid_t forkpty(int *, char *, struct termios *, struct winsize *);
int Process::OpenPTY(const char **argv) {
    // struct termios term;
    // struct winsize win;
    char name[PATH_MAX];
    int fd = -1;
    if ((pid = forkpty(&fd, name, 0, 0))) {
        if (pid < 0) { close(fd); return -1; }
        fcntl(fd, F_SETFL, O_NONBLOCK);
        in = fdopen(fd, "r");
        out = fdopen(fd, "w");
    } else {
        execvp(argv[0], (char*const*)argv);
    }
    return 0;
}

int Process::Close() {
    if (pid) { kill(pid, SIGHUP); pid = 0; }
    if (in)  { fclose(in);        in  = 0; }
    if (out) { fclose(out);       out = 0; }
    return 0;
}

void Process::Daemonize(const char *dir) {
    char fn1[256], fn2[256];
    snprintf(fn1, sizeof(fn1), "%s%s.stdout", dir, app->progname.c_str());
    snprintf(fn2, sizeof(fn2), "%s%s.stderr", dir, app->progname.c_str());
    FILE *fout = fopen(fn1, "a"); fprintf(stderr, "open %s %s\n", fn1, fout ? "OK" : strerror(errno));
    FILE *ferr = fopen(fn2, "a"); fprintf(stderr, "open %s %s\n", fn2, ferr ? "OK" : strerror(errno));
    Daemonize(fout, ferr);
}

void Process::Daemonize(FILE *fout, FILE *ferr) {
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

int NTService::Install(const char *name, const char *path) { FATAL("not implemented"); }
int NTService::Uninstall(const char *name) { FATAL("not implemented"); }
int NTService::WrapMain(const char *name, MainCB main_cb, int argc, const char **argv) { return main_cb(argc, argv); }

#endif /* WIN32 */

struct DownloadDirectory {
    string text;
    DownloadDirectory() {
#ifdef LFL_IPHONE
        char *path = NSFMDocumentPath();
        text = string(path) + "/";
        free(path);
#endif
#ifdef _WIN32
        char path[MAX_PATH];
        if (!SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL|CSIDL_FLAG_CREATE, NULL, 0, path))) return;
        text = string(path) + "/";
#endif
    }
};

string File::Contents() {
    if (!Opened()) return "";
    int l = Size();
    if (!l) return "";
    Reset();

    string ret;
    ret.resize(l);
    Read((char*)ret.data(), l);
    return ret;
}

const char *File::NextLine(int *offset, int *nextoffset) {
    const char *nl;
    if (!(nl = nr.GetNextRecord(this, offset, nextoffset, LFL::NextLine))) return 0;
    if (nl) nr.buf[nr.record_offset + nr.record_len] = 0;
    return nl;
}

const char *File::NextLineRaw(int *offset, int *nextoffset) {
    const char *nl;
    if (!(nl = nr.GetNextRecord(this, offset, nextoffset, LFL::NextLineRaw))) return 0;
    if (nl) nr.buf[nr.record_offset + nr.record_len] = 0;
    return nl;
}

const char *File::NextChunk(int *offset, int *nextoffset) {
    const char *nc;
    if (!(nc = nr.GetNextRecord(this, offset, nextoffset, LFL::NextChunk<4096>))) return 0;
    if (nc) nr.buf[nr.record_offset + nr.record_len] = 0;
    return nc;
}

const char *File::NextProto(int *offset, int *nextoffset, ProtoHeader *bhout) {
    const char *np;
    if (!(np = nr.GetNextRecord(this, offset, nextoffset, LFL::NextProto))) return 0;
    if (bhout) *bhout = ProtoHeader(np);
    return np + ProtoHeader::size;
}

const char *File::NextRecord::GetNextRecord(File *f, int *offsetOut, int *nextoffsetOut, NextRecordCB nextcb) {
    const char *next, *text; int left; bool read_short = false;
    if (buf_dirty) buf_offset = buf.size();
    for (;;) {
        left = buf.size() - buf_offset;
        text = buf.data() + buf_offset;
        if (!buf_dirty && left>0 && (next = nextcb(StringPiece(text, left), read_short, &record_len))) {

            if (offsetOut) *offsetOut = file_offset - buf.size() + buf_offset;
            if (nextoffsetOut) *nextoffsetOut = file_offset - buf.size() + (next - buf.data());

            record_offset = buf_offset;
            buf_offset = next-buf.data();
            return text;
        }
        if (read_short) return 0;

        buf.erase(0, buf_offset);
        int buf_filled = buf.size();
        buf.resize(buf.size() < 4096 ? 4096 : buf.size()*2);
        int len = f->Read((char*)buf.data()+buf_filled, buf.size()-buf_filled);
        read_short = len < buf.size()-buf_filled;
        buf.resize(max(len,0) + buf_filled);
        buf_dirty = false;
        buf_offset = 0;
    }
}

int File::WriteProto(const ProtoHeader *hdr, const Proto *msg, bool doflush) {
#ifdef LFL_PROTOBUF
    std::string v = msg->SerializeAsString();
    CHECK_EQ(hdr->len, v.size());
    v.insert(0, (const char *)hdr, ProtoHeader::size);
    int ret = (Write(v.c_str(), v.size()) == v.size()) ? v.size() : -1;
    if (doflush) Flush();
    return ret;
#else
    return -1;
#endif
}

int File::WriteProto(ProtoHeader *hdr, const Proto *msg, bool doflush) {
#ifdef LFL_PROTOBUF
    std::string v = msg->SerializeAsString();
    hdr->SetLength(v.size());
    v.insert(0, (const char *)hdr, ProtoHeader::size);
    int ret = (Write(v.c_str(), v.size()) == v.size()) ? v.size() : -1;
    if (doflush) Flush();
    return ret;
#else
    return -1;
#endif
}

int File::WriteProtoFlag(const ProtoHeader *hdr, bool doflush) {
    int ret = Write(&hdr->flag, sizeof(int)) == sizeof(int) ? sizeof(int) : -1;
    if (doflush) Flush();
    return ret;
}

long long BufferFile::Seek(long long offset, int whence) {
    if (offset < 0 || offset >= buf.size()) return -1;
    nr.buf_dirty = true;
    return rdo = wro = nr.file_offset = offset;
}

int BufferFile::Read(void *out, size_t size) {
    size_t l = min(size, buf.size() - rdo);
    memcpy(out, buf.data() + rdo, l);
    rdo += l;
    nr.file_offset += l;
    nr.buf_dirty = true;
    return l;
}

int BufferFile::Write(const void *In, size_t size) {
    const char *in = (const char *)In;
    if (size == -1) size = strlen(in);
    size_t l = min(size, buf.size() - wro);
    buf.replace(wro, l, in, l);
    if (size > l) buf.append(in + l, size - l);
    wro += size;
    nr.file_offset += size;
    nr.buf_dirty = true;
    return size;
}

#ifdef LFL_ANDROID
#if 0
bool LocalFile::open(const char *path, const char *mode, bool pre_create) {
    char *b=0; int l=0, ret;
    FILE *f = fopen(path, mode);
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    l = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = (char*)malloc(l);
    fread(b, l, 1, f);
    fclose(f);
    impl = new BufferFile(string(b, l));
    ((BufferFile*)impl)->free = true;
    return true;
}
#endif
bool LocalFile::open(const string &path, const string &mode, bool pre_create) {
    if ((writable = strchr(mode.c_str(), 'w'))) {
        impl = android_internal_open_writer(path.c_str());
        return impl;
    }

    char *b=0; int l=0, ret;
    bool internal_path = !strchr(path.c_str(), '/');
    if (internal_path) { if ((ret = android_internal_read(path.c_str(), &b, &l))) return false; }
    else               { if ((ret = android_file_read    (path.c_str(), &b, &l))) return false; }

    impl = new BufferFile(string(b, l));
    ((BufferFile*)impl)->free = true;
    return true;
}

void LocalFile::reset() { if (impl && !writable) ((BufferFile*)impl)->reset(); }
int LocalFile::size() { return (impl && !writable) ? ((BufferFile*)impl)->size() : -1; }
void LocalFile::close() { if (impl) { if (writable) android_internal_close_writer(impl); else delete ((BufferFile*)impl); impl=0; } }
long long LocalFile::seek(long long offset, int whence) { return (impl && !writable) ? ((BufferFile*)impl)->seek(offset, whence) : -1; }
int LocalFile::read(void *buf, int size) { return (impl && !writable) ? ((BufferFile*)impl)->read(buf, size) : -1; }
int LocalFile::write(const void *buf, int size) { return impl ? (writable ? android_internal_write(impl, (const char*)buf, size) : ((BufferFile*)impl)->write(buf, size)) : -1; }
bool LocalFile::flush() { return false; }

#else /* LFL_ANDROID */
bool LocalFile::mkdir(const string &dir, int mode) {
#ifdef _WIN32
    return _mkdir(dir.c_str()) == 0;
#else
    return ::mkdir(dir.c_str(), mode) == 0;
#endif
}
int LocalFile::WhenceMap(int n) {
    if      (n == Whence::SET) return SEEK_SET;
    else if (n == Whence::CUR) return SEEK_CUR;
    else if (n == Whence::END) return SEEK_END;
    else return -1;
}

bool LocalFile::Open(const string &path, const string &mode, bool pre_create) {
    fn = path;
    if (pre_create) {
        FILE *created = fopen(fn.c_str(), "a");
        if (created) fclose(created);
    }
#ifdef _WIN32
    if (!(impl = fopen(fn.c_str(), StrCat(mode, "b").c_str()))) return 0;
#else
    if (!(impl = fopen(fn.c_str(), mode.c_str()))) return 0;
#endif
    nr.Reset();

    if (!Opened()) return false;
    writable = strchr(mode.c_str(), 'w');
    return true;
}

void LocalFile::Reset() {
    fseek((FILE*)impl, 0, SEEK_SET);
    nr.Reset();
}

int LocalFile::Size() {
    if (!impl) return -1;

    int place = ftell((FILE*)impl);
    fseek((FILE*)impl, 0, SEEK_END);

    int ret = ftell((FILE*)impl);
    fseek((FILE*)impl, place, SEEK_SET);
    return ret;
}

void LocalFile::Close() {
    if (impl) fclose((FILE*)impl);
    impl = 0;
    nr.Reset();
}

long long LocalFile::Seek(long long offset, int whence) {
    long long ret = fseek((FILE*)impl, offset, WhenceMap(whence));
    if (ret < 0) return ret;
    if (whence == Whence::SET) ret = offset;
    else ret = ftell((FILE*)impl);
    nr.file_offset = ret;
    nr.buf_dirty = true;
    return ret;
}

int LocalFile::Read(void *buf, size_t size) {
    int ret = fread(buf, 1, size, (FILE*)impl);
    if (ret < 0) return ret;
    nr.file_offset += ret;
    nr.buf_dirty = true;
    return ret;
}

int LocalFile::Write(const void *buf, size_t size) {
    int ret = fwrite(buf, 1, size!=-1?size:strlen((char*)buf), (FILE*)impl);
    if (ret < 0) return ret;
    nr.file_offset += ret;
    nr.buf_dirty = true;
    return ret;
}

bool LocalFile::Flush() { fflush((FILE*)impl); return true; }
#endif /* LFL_ANDROID */

string LocalFile::CurrentDirectory(int max_size) {
    string ret(max_size, 0); 
    getcwd((char*)ret.data(), ret.size());
    ret.resize(strlen(ret.data()));
    return ret;
}

DirectoryIter::DirectoryIter(const string &path, int dirs, const char *Pref, const char *Suf) : P(Pref), S(Suf), init(0) {
    if (LocalFile::IsDirectory(path)) pathname = path;
    else {
        if (LocalFile(path, "r").Opened()) {
            pathname = string(path, DirNameLen(path)) + LocalFile::Slash;
            filemap[BaseName(path)] = 1;
        }
        return;
    }
#ifdef _WIN32
    _finddatai64_t f; int h;
    string match = StrCat(path, "*.*");
    if ((h = _findfirsti64(match.c_str(), &f)) < 0) return;
    
    do {
        if (!strcmp(f.name, ".")) continue;

        bool isdir = f.attrib & _A_SUBDIR;
        if (dirs >= 0 && isdir != dirs) continue;

        filemap[StrCat(f.name, isdir?"/":"")] = 1;
    }
    while (!_findnexti64(h, &f));

    _findclose(h);
#else /* _WIN32 */
#ifdef LFL_IPHONE

    NSFMreaddir(path, dirs, (void**)this, add);

#else /* LFL_IPHONE */
    DIR *dir; dirent *dent; string dirname=path;
    if (dirname.empty()) dirname = ".";
    if (dirname.size() > 1 && dirname[dirname.size()-1] == '/') dirname.erase(dirname.size()-1);
    if (!(dir = opendir(dirname.c_str()))) return;

    while ((dent = readdir(dir))) {
        if (!strcmp(dent->d_name, ".")) continue;
        if (dent->d_type != DT_DIR && dent->d_type != DT_REG && dent->d_type != DT_LNK) continue;

        bool isdir = dent->d_type == DT_DIR;
        if (dirs >= 0 && isdir != dirs) continue;

        filemap[StrCat(dent->d_name, isdir?"/":"")] = 1;
    }

    closedir(dir);
#endif /* __APPLE__ */
#endif /* _WIN32 */
}

const char *DirectoryIter::Next() {
    const char *fn=0;
    for (;;) {
        if (!init) { init=1; iter = filemap.begin(); }
        else iter++;
        if (iter == filemap.end()) return 0;
        const char *k = (*iter).first.c_str();

        if (!strcmp(k, "../")) continue;
        if (P && !PrefixMatch(k, P)) continue;
        if (S && !SuffixMatch(k, S)) continue;
        return k;
    }
}

template <class X> const X *StringLineIterT<X>::Next() {
    first = false;
    if (offset < 0) return 0;
    const X *line = in+offset, *next = NextLine(StringPieceT<X>(line, (len >= 0 ? len-offset : -1)), false, &linelen);
    offset = next ? next-in : -1;
    if (linelen) linelen -= ChompNewlineLength(line, linelen);
    if (linelen || ((flag & Flag::BlankLines) && next)) {
        if (flag & Flag::InPlace) return line;
        buf.assign(line, linelen);
        return buf.c_str();
    }
    return 0;
}

template struct StringLineIterT<char>;
template struct StringLineIterT<short>;

template <class X> 
StringWordIterT<X>::StringWordIterT(const StringPieceT<X> &input, int (*delim)(int), int (*quote)(int), int inflag)
    : in(input.buf), len(input.len), wordlen(0), offset(0), IsQuote(quote), flag(inflag) {
    IsSpace = delim ? delim : ::isspace;
    if (in) SkipSpace();
}

template <class X> const X *StringWordIterT<X>::Next() {
    if (offset < 0) return 0;
    const X *word = in+offset, *next = NextChar(word, IsSpace, IsQuote, (len >= 0 ? len-offset : -1), &wordlen);
    if ((offset = next ? next-in : -1) >= 0) SkipSpace();
    if (wordlen) {
        if (flag & Flag::InPlace) return word;
        buf.assign(word, wordlen);
        return buf.c_str();
    }
    return 0;
}

template <class X> const X *StringWordIterT<X>::Remaining() {
    if (flag & Flag::InPlace) return in+offset;
    if (len >= 0) buf.assign(in+offset, len-offset);

    else buf.assign(in+offset);
    return buf.c_str();
}

template <class X> void StringWordIterT<X>::SkipSpace() {
    if (len >= 0) while (offset < len && IsSpace(*(in+offset))) offset++;
    else          while (*(in+offset) && IsSpace(*(in+offset))) offset++;
}

template struct StringWordIterT<char>;
template struct StringWordIterT<short>;

const char *IterWordIter::Next() {
    if (!iter) return 0;
    const char *w = word.in ? word.Next() : 0;
    while(!w) {
        line_count++;
        const char *line = iter->Next();
        if (!line) return 0;
        word = StringWordIter(line, word.IsSpace);
        w = word.Next();
    }
    return w;
}    

#ifdef LFL_LIBARCHIVE
ArchiveIter::~ArchiveIter() { free(dat); if (impl) archive_read_finish((archive*)impl); }
ArchiveIter::ArchiveIter(const char *path) {
    entry=0; dat=0; impl=0;

    if (!(impl = archive_read_new())) return;
    if (archive_read_support_format_zip          ((archive*)impl) != 0) INFO("no zip support");
    if (archive_read_support_format_tar          ((archive*)impl) != 0) INFO("no tar support");
    if (archive_read_support_compression_gzip    ((archive*)impl) != 0) INFO("no gzip support");
    if (archive_read_support_compression_none    ((archive*)impl) != 0) INFO("no none support");
    if (archive_read_support_compression_compress((archive*)impl) != 0) INFO("no compress support");

    if (archive_read_open_filename((archive*)impl, path, 65536) != 0) {
        archive_read_finish((archive*)impl); impl=0; return;
    }
}
const char *ArchiveIter::Next() {
    int ret;
    if (!impl) return 0;
    if ((ret = archive_read_next_header((archive*)impl, (archive_entry**)&entry))) { ERROR("read_next: ", ret, " ", archive_error_string((archive*)impl)); return 0; }
    return archive_entry_pathname((archive_entry*)entry);
}
const void *ArchiveIter::Data() {
    int l=Size(); free(dat); dat=malloc(l);
    if (archive_read_data_into_buffer((archive*)impl, dat, l)) { free(dat); dat=0; }
    return dat;
}
void ArchiveIter::Skip() { archive_read_data_skip((archive*)impl); }
long long ArchiveIter::Size() { return archive_entry_size((archive_entry*)entry); }

#else /* LFL_LIBARCHIVE */

ArchiveIter::~ArchiveIter() { free(dat); }
ArchiveIter::ArchiveIter(const char *path) { entry=0; dat=0; impl=0; }
const char *ArchiveIter::Next() { return 0; }
long long ArchiveIter::Size() { return 0; }
const void *ArchiveIter::Data() { return 0; }
void ArchiveIter::Skip() {}
#endif /* LFL_LIBARCHIVE */

#ifdef LFL_REGEX
Regex::~Regex() { re_free((regexp*)impl); }
Regex::Regex(const string &patternstr) {
    regexp* compiled = 0;
    if (!re_comp(&compiled, patternstr.c_str())) impl = compiled;
}
int Regex::Match(const string &text, vector<Regex::Result> *out) {
    if (!impl) return -1;
    regexp* compiled = (regexp*)impl;
    vector<regmatch> matches(re_nsubexp(compiled));
    int retval = re_exec(compiled, text.c_str(), matches.size(), &matches[0]);
    if (retval < 1) return retval;
    if (out) for (auto i : matches) out->emplace_back(i.begin, i.end);
    return 1;
}
#else
Regex::~Regex() {}
Regex::Regex(const string &patternstr) {}
int Regex::Match(const string &text, vector<Regex::Result> *out) { ERROR("regex not implemented"); return 0; }
#endif

#ifdef LFL_SREGEX
StreamRegex::~StreamRegex() {
    if (ppool) sre_destroy_pool((sre_pool_t*)ppool);
    if (cpool) sre_destroy_pool((sre_pool_t*)cpool);
}
StreamRegex::StreamRegex(const string &patternstr) : ppool(sre_create_pool(1024)), cpool(sre_create_pool(1024)) {
    sre_uint_t ncaps;
    sre_int_t err_offset = -1;
    sre_regex_t *re = sre_regex_parse((sre_pool_t*)cpool, (sre_char *)patternstr.c_str(), &ncaps, 0, &err_offset);
    prog = sre_regex_compile((sre_pool_t*)ppool, re);
    sre_reset_pool((sre_pool_t*)cpool);
    res.resize(2*(ncaps+1));
    ctx = sre_vm_pike_create_ctx((sre_pool_t*)cpool, (sre_program_t*)prog, &res[0], res.size()*sizeof(sre_int_t));
}
int StreamRegex::Match(const string &text, vector<Regex::Result> *out, bool eof) {
    int offset = last_end + since_last_end;
    sre_int_t rc = sre_vm_pike_exec((sre_vm_pike_ctx_t*)ctx, (sre_char*)text.data(), text.size(), eof, NULL);
    if (rc >= 0) {
        since_last_end = 0;
        for (int i = 0, l = res.size(); i < l; i += 2) 
            out->emplace_back(res[i] - offset, (last_end = res[i+1]) - offset);
    } else since_last_end += text.size();
    return 1;
}
#else
StreamRegex::~StreamRegex() {}
StreamRegex::StreamRegex(const string &patternstr) {}
int StreamRegex::Match(const string &text, vector<Regex::Result> *out, bool eof) { return 0; }
#endif

#ifdef  LFL_UNICODE_DEBUG
#define UnicodeDebug(...) ERROR(__VA_ARGS__)
#else
#define UnicodeDebug(...)
#endif

#ifdef LFL_ICONV
template <class X, class Y>
int String::Convert(const StringPieceT<X> &in, basic_string<Y> *out, const char *from, const char *to) {
    iconv_t cd = iconv_open(to, from);
    if (cd < 0) { ERROR("failed convert ", from, " to ", to); out->clear(); return 0; }

    out->resize(in.len*4/sizeof(Y)+4);
    char *inp = (char*)in.buf, *top = (char*)out->data();
    size_t in_remaining = in.len*sizeof(X), to_remaining = out->size()*sizeof(Y);
    if (iconv(cd, &inp, &in_remaining, &top, &to_remaining) == -1)
    { /* ERROR("failed convert ", from, " to ", to); */ iconv_close(cd); out->clear(); return 0; }
    out->resize(out->size() - to_remaining/sizeof(Y));
    iconv_close(cd);

    return in.len - in_remaining/sizeof(X);
}
#else /* LFL_ICONV */
template <class X, class Y>
int String::Convert(const StringPieceT<X> &in, basic_string<Y> *out, const char *from, const char *to) {
    if (strcmp(from, to)) ONCE(ERROR("conversion not supported.  copying.  #define LFL_ICONV"));
    String::Copy(in, out);
    return in.len;
}
#endif /* LFL_ICONV */
template int String::Convert<char,  char >(const StringPiece  &, string  *, const char*, const char*);
template int String::Convert<char,  short>(const StringPiece  &, String16*, const char*, const char*);
template int String::Convert<short, char >(const String16Piece&, string  *, const char*, const char*);
template int String::Convert<short, short>(const String16Piece&, String16*, const char*, const char*);

String16 String::ToUTF16(const StringPiece &text, int *consumed) {
    int input = text.Length(), output = 0, c_bytes, c;
    String16 ret;
    ret.resize(input);
    const char *b = text.data(), *p = b;
    for (; !text.Done(p); p += c_bytes, output++) {
        ret[output] = UTF8::ReadGlyph(text, p, &c_bytes);
        if (!c_bytes) break;
    }
    CHECK_LE(output, input);
    if (consumed) *consumed = p - b;
    ret.resize(output);
    return ret;
}

String16 UTF16::WriteGlyph(int codepoint) { return String16(1, codepoint); }
int UTF16::ReadGlyph(const String16Piece &s, const short *p, int *len, bool eof) { *len=1; return *(const unsigned short *)p; }
int UTF8 ::ReadGlyph(const StringPiece   &s, const char  *p, int *len, bool eof) {
    *len = 1;
    unsigned char c0 = *(const unsigned char *)p;
    if ((c0 & (1<<7)) == 0) return c0; // ascii
    if ((c0 & (1<<6)) == 0) { UnicodeDebug("unexpected continuation byte"); return c0; }
    for ((*len)++; *len < 4; (*len)++) {
        if (s.Done(p + *len - 1)) { UnicodeDebug("unexpected end of string"); *len=eof; return c0; }
        if ((c0 & (1<<(7 - *len))) == 0) break;
    }

    int ret = 0;
    if      (*len == 2) ret = c0 & 0x1f;
    else if (*len == 3) ret = c0 & 0x0f;
    else if (*len == 4) ret = c0 & 0x07;
    else { UnicodeDebug("invalid len ", *len); *len=1; return c0; }

    for (int i = *len; i > 1; i--) {
        unsigned char c = *(const unsigned char *)++p;
        if ((c & 0xc0) != 0x80) { UnicodeDebug("unexpected non-continuation byte"); *len=1; return c0; }
        ret = (ret << 6) | (c & 0x3f);
    }
    return ret;
}
string UTF8::WriteGlyph(int codepoint) {
#if 1
    string out;
    short in[] = { (short)codepoint, 0 };
    String::Convert(String16Piece(in, 1), &out, "UTF-16LE", "UTF-8");
    return out;
#else
#endif
}

Base64::Base64() : encoding_table("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"), decoding_table(256, 0) {
    mod_table[0]=0; mod_table[1]=2; mod_table[2]=1;
    for (int i = 0; i < 64; i++) decoding_table[(unsigned char)encoding_table[i]] = i;
}

string Base64::Encode(const char *in, size_t input_length) {
    const unsigned char *data = (const unsigned char *) in;
    string encoded_data(4 * ((input_length + 2) / 3), 0);
    for (int i = 0, j = 0; i < input_length;) {
        unsigned octet_a = i < input_length ? data[i++] : 0;
        unsigned octet_b = i < input_length ? data[i++] : 0;
        unsigned octet_c = i < input_length ? data[i++] : 0;
        unsigned triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;
        encoded_data[j++] = encoding_table[(triple >> 3 * 6) & 0x3F];
        encoded_data[j++] = encoding_table[(triple >> 2 * 6) & 0x3F];
        encoded_data[j++] = encoding_table[(triple >> 1 * 6) & 0x3F];
        encoded_data[j++] = encoding_table[(triple >> 0 * 6) & 0x3F];
    }
    for (int i = 0; i < mod_table[input_length % 3]; i++) encoded_data[encoded_data.size() - 1 - i] = '=';
    return encoded_data;
}

string Base64::Decode(const char *data, size_t input_length) {
    CHECK_EQ(input_length % 4, 0);
    string decoded_data(input_length / 4 * 3, 0);
    if (data[input_length - 1] == '=') decoded_data.erase(decoded_data.size()-1);
    if (data[input_length - 2] == '=') decoded_data.erase(decoded_data.size()-1);
    for (int i = 0, j = 0; i < input_length;) {
        unsigned sextet_a = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
        unsigned sextet_b = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
        unsigned sextet_c = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
        unsigned sextet_d = data[i] == '=' ? 0 & i++ : decoding_table[data[i++]];
        unsigned triple = (sextet_a << 3 * 6) + (sextet_b << 2 * 6) + (sextet_c << 1 * 6) + (sextet_d << 0 * 6);
        if (j < decoded_data.size()) decoded_data[j++] = (triple >> 2 * 8) & 0xFF;
        if (j < decoded_data.size()) decoded_data[j++] = (triple >> 1 * 8) & 0xFF;
        if (j < decoded_data.size()) decoded_data[j++] = (triple >> 0 * 8) & 0xFF;
    }
    return decoded_data;
}

string GraphViz::Footer() { return "}\r\n"; }
string GraphViz::DigraphHeader(const string &name) {
    return StrCat("digraph ", name, " {\r\n"
                  "rankdir=LR;\r\n"
                  "size=\"8,5\"\r\n"
                  "node [style = solid];\r\n"
                  "node [shape = circle];\r\n");
}
string GraphViz::NodeColor(const string &s) { return StrCat("node [color = ", s, "];\r\n"); }
string GraphViz::NodeShape(const string &s) { return StrCat("node [shape = ", s, "];\r\n"); }
string GraphViz::NodeStyle(const string &s) { return StrCat("node [style = ", s, "];\r\n"); }
void GraphViz::AppendNode(string *out, const string &n1, const string &label) {
    StrAppend(out, "\"", n1, "\"",
              (label.size() ? StrCat(" [ label = \"", label, "\" ] ") : ""),
              ";\r\n");
}
void GraphViz::AppendEdge(string *out, const string &n1, const string &n2, const string &label) {
    StrAppend(out, "\"", n1, "\" -> \"", n2, "\"",
              (label.size() ? StrCat(" [ label = \"", label, "\" ] ") : ""),
              ";\r\n");
}

#ifdef LFL_OPENSSL
string Crypto::MD5(const string &in) {
    string out(MD5_DIGEST_LENGTH, 0);
    ::MD5((const unsigned char*)in.c_str(), in.size(), (unsigned char*)out.data());
    return out;
}

string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) {
    unsigned char iv[8] = {0,0,0,0,0,0,0,0};
    EVP_CIPHER_CTX ctx; 
    EVP_CIPHER_CTX_init(&ctx); 
    EVP_CipherInit_ex(&ctx, EVP_bf_cbc(), NULL, NULL, NULL, encrypt_or_decrypt);
    EVP_CIPHER_CTX_set_key_length(&ctx, passphrase.size());
    EVP_CipherInit_ex(&ctx, NULL, NULL, (const unsigned char *)passphrase.c_str(), iv, encrypt_or_decrypt); 

    int outlen = 0, tmplen = 0;
    string out(in.size()+encrypt_or_decrypt*EVP_MAX_BLOCK_LENGTH, 0);
    EVP_CipherUpdate(&ctx, (unsigned char*)out.data(), &outlen, (const unsigned char *)in.c_str(), in.size());
    EVP_CipherFinal_ex(&ctx, (unsigned char*)out.data() + outlen, &tmplen); 
    if (in.size() % 8) outlen += tmplen;

    EVP_CIPHER_CTX_cleanup(&ctx); 
    if (encrypt_or_decrypt) {
        CHECK_LE(outlen, out.size());
        out.resize(outlen);
    }
    return out;
}
#else
string Crypto::MD5(const string &in) { FATAL("not implemented"); }
string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) { FATAL("not implemented"); }
#endif

void SystemBrowser::Open(const char *url_text) {
#if defined(LFL_ANDROID)
    AndroidOpenBrowser(url_text);
#elif defined(LFL_IPHONE)
    iPhoneOpenBrowser(url_text);
#elif defined(__APPLE__)
    CFURLRef url = CFURLCreateWithBytes(0, (UInt8*)url_text, strlen(url_text), kCFStringEncodingASCII, 0);
    if (url) { LSOpenCFURLRef(url, 0); CFRelease(url); }
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

string FlagMap::Get   (const string &k) const { Flag *f = FindOrNull(flagmap, k); return f ? f->Get()    : "";    }
bool   FlagMap::IsBool(const string &k) const { Flag *f = FindOrNull(flagmap, k); return f ? f->IsBool() : false; }

bool FlagMap::Set(const string &k, const string &v) {
    Flag *f = FindOrNull(flagmap, k);
    if (!f) return false;
    INFO("set flag ", k, " = ", v);
    if (f->Get() != v) f->override = dirty = true;
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

void Application::Log(int level, const char *file, int line, const string &message) {
    if (level > FLAGS_loglevel || (level >= LFApp::Log::Debug && !FLAGS_lfapp_debug)) return;
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
#ifdef LFL_ANDROID
        __android_log_print(ANDROID_LOG_INFO, caption.c_str(), "%s (%s:%d)", message.c_str(), file, line);
#endif
    }
    if (level == LFApp::Log::Fatal) LFAppFatal();
    if (FLAGS_lfapp_video && screen && screen->console) screen->console->Write(message);
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

int Application::Create(int argc, const char **argv, const char *source_filename) {
#ifdef LFL_GLOG
    google::InstallFailureSignalHandler();
#endif
    SetLFAppMainThread();
    progname = argv[0];
    startdir = LocalFile::CurrentDirectory();
    time_started = Now();

#ifdef __APPLE__
    char rpath[1024];
    CFBundleRef mainBundle = CFBundleGetMainBundle();
    CFURLRef respath = CFBundleCopyResourcesDirectoryURL(mainBundle);
    CFURLGetFileSystemRepresentation(respath, true, (UInt8*)rpath, sizeof(rpath));
    CFRelease(respath);
    INFO("chdir(", rpath, ")");
    chdir(rpath);
#endif

#ifdef _WIN32
    { /* winsock startup */
        WSADATA wsadata;
        WSAStartup(MAKEWORD(2,2), &wsadata);
    }
#else
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

    srand(time(0));
    if (logfilename.size()) logfile = fopen(logfilename.c_str(), "a");

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
        Process::Daemonize();
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

#ifdef LFL_FFMPEG
    INFO("lfapp_open: ffmpeg_init()");
    //av_log_set_level(AV_LOG_DEBUG);
    av_register_all();
#endif /* LFL_FFMPEG */

    thread_pool.Open(X_or_1(FLAGS_threadpool_size));
    if (FLAGS_threadpool_size) thread_pool.Start();

    if (FLAGS_lfapp_audio || FLAGS_lfapp_video) {
        if (assets.Init()) { ERROR("assets init failed"); return -1; }
    }

    if (FLAGS_lfapp_audio) {
        INFO("lfapp_open: audio_init()");
        if (LoadModule(&audio)) { INFO("audio init failed"); return -1; }
    }
    else { FLAGS_chans_in=FLAGS_chans_out=1; }

    if (FLAGS_lfapp_camera) {
        INFO("lfapp_open: camera_init()");
        if (LoadModule(&camera)) { INFO("camera init failed"); return -1; }
    }

    if (FLAGS_lfapp_video) {
        INFO("lfapp_open: video_init()");
        if (video.Init()) { INFO("video init failed"); return -1; }
    } else {
        Window::active[screen->id] = screen;
    }

    if (FLAGS_lfapp_input) {
        INFO("lfapp_open: input_init()");
        if (LoadModule(&input)) { INFO("input init failed"); return -1; }
        input.Init(screen);
    }

    if (FLAGS_lfapp_network) {
        INFO("lfapp_open: network_init()");
        if (LoadModule(&network)) { INFO("service init failed"); return -1; }

        network.Enable(Singleton<UDPClient>::Get());
        vector<IPV4::Addr> nameservers;
        if (FLAGS_nameserver.empty()) {
            INFO("network_init(): service_enable(new Resolver(defaultNameserver()))");
            Resolver::DefaultNameserver(&nameservers);
        } else {
            INFO("network_init(): service_enable(new Resolver(", FLAGS_nameserver, "))");
            IPV4::ParseCSV(FLAGS_nameserver, &nameservers);
        }
        for (int i = 0; i < nameservers.size(); ++i) Singleton<Resolver>::Get()->Connect(nameservers[i]);

        INFO("network_init(): service_enable(new HTTPClient)");
        network.Enable(Singleton<HTTPClient>::Get());
    }

    if (FLAGS_lfapp_cuda) {
        INFO("lfapp_open: cuda_init()");
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
    if (FLAGS_lfapp_audio && audio.Start()) return -1;
    return 0;
}

int Application::PreFrame(unsigned clicks) {
    pre_frames_ran++;

    for (auto i = modules.begin(); i != modules.end() && run; ++i) (*i)->Frame(clicks);

    // handle messages sent to main thread
    if (run) message_queue.HandleMessages();

    // fake threadpool that executes in main thread
    if (run && !FLAGS_threadpool_size) thread_pool.queue[0]->HandleMessages();

    return 0;
}

int Application::PostFrame() {
    frames_ran++;
    scheduler.FrameDone();
    return 0;
}

int Application::Frame() {
    if (!MainThread()) ERROR("Frame() called from thread ", Thread::GetId());
    scheduler.FrameWait();
    unsigned clicks = scheduler.monolithic_frame ? frame_time.GetTime(true) : screen->frame_time.GetTime(true);

    int flag = 0;
    PreFrame(clicks);

    if (scheduler.monolithic_frame) {
        Window *previous_screen = screen;
        for (auto i = Window::active.begin(); run && i != Window::active.end(); ++i)
            i->second->Frame(clicks, audio.mic_samples, camera.have_sample, flag);
        if (previous_screen && previous_screen != screen) Window::MakeCurrent(previous_screen);
    } else {
        screen->Frame(clicks, audio.mic_samples, camera.have_sample, flag);
    }

    PostFrame();

    return clicks;
}

int Application::Main() {
    if (Start()) return Exiting();
#if defined(LFL_QT) || defined(LFL_OSXVIDEO)
    return 0;
#endif
    return MainLoop();
}
    
int Application::MainLoop() {
    while (run) {
        // if (!minimized)
        Frame();
#ifdef LFL_IPHONE
        // if (minimized) run = 0;
#endif
        Msleep(1);
    }

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
    if (FLAGS_open_console) press_any_key();
#endif
    return 0;
}

/* FrameScheduler */

FrameScheduler::FrameScheduler() : maxfps(&FLAGS_target_fps), select_thread(&frame_mutex, &wait_mutex) {
#ifdef LFL_OSXINPUT
    rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = 0;
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
    if (wait_forever && wait_forever_thread) select_thread.Wait();
}
void FrameScheduler::Start() {
    if (wait_forever && wait_forever_thread) select_thread.Start();
}
void FrameScheduler::FrameDone() { if (rate_limit && app->run && FLAGS_target_fps) maxfps.Limit(); }
void FrameScheduler::FrameWait() {
    if (wait_forever && !FLAGS_target_fps) {
        if (synchronize_waits) {
            wait_mutex.lock();
            frame_mutex.unlock();
        }
#if defined(LFL_QT)
#elif defined(LFL_GLFWINPUT)
        glfwWaitEvents();
#elif defined(LFL_SDLINPUT)
        SDL_WaitEvent(NULL);
#elif defined(LFL_OSXINPUT)
#else
        FATAL("not implemented");
#endif
        if (synchronize_waits) {
            frame_mutex.lock();
            wait_mutex.unlock();
        }
    }
}
void FrameScheduler::Wakeup() {
    if (wait_forever) {
#if defined(LFL_QT)
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
#elif defined(LFL_OSXINPUT)
        OSXTriggerFrame(screen->id);
#else
        FATAL("not implemented");
#endif
    }
}
void FrameScheduler::UpdateTargetFPS(int fps) {
    screen->target_fps = fps;
    if (monolithic_frame) {
        int next_target_fps = 0;
        for (const auto &w : Window::active) Max(&next_target_fps, w.second->target_fps);
        FLAGS_target_fps = next_target_fps;
    }
#if defined(LFL_OSXINPUT)
    CHECK(screen->id);
    OSXUpdateTargetFPS(screen->id);
#endif
}
void FrameScheduler::AddWaitForeverMouse() {
#if defined(LFL_OSXINPUT)
    CHECK(screen->id);
    OSXAddWaitForeverMouse(screen->id);
#endif
}
void FrameScheduler::DelWaitForeverMouse() {
#if defined(LFL_OSXINPUT)
    CHECK(screen->id);
    OSXDelWaitForeverMouse(screen->id);
#endif
}
void FrameScheduler::AddWaitForeverKeyboard() {
#if defined(LFL_OSXINPUT)
    CHECK(screen->id);
    OSXAddWaitForeverKeyboard(screen->id);
#endif
}
void FrameScheduler::DelWaitForeverKeyboard() {
#if defined(LFL_OSXINPUT)
    CHECK(screen->id);
    OSXDelWaitForeverKeyboard(screen->id);
#endif
}
void FrameScheduler::AddWaitForeverService(Service *svc) {
    if (wait_forever && wait_forever_thread) svc->active.mirror = &select_thread;
}
void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) {
    if (wait_forever && wait_forever_thread) select_thread.Add(fd, flag, val);
#ifdef LFL_OSXINPUT
    if (!wait_forever_thread) { CHECK_EQ(SocketSet::READABLE, flag); OSXAddWaitForeverSocket(screen->id, fd); }
#endif
}
void FrameScheduler::DelWaitForeverSocket(Socket fd) {
    if (wait_forever && wait_forever_thread) select_thread.Del(fd);
}

/* CUDA */

#ifdef LFL_CUDA
void PrintCUDAProperties(cudaDeviceProp *prop) {
    DEBUGf("Major revision number:         %d", prop->major);
    DEBUGf("Minor revision number:         %d", prop->minor);
    DEBUGf("Name:                          %s", prop->name);
    DEBUGf("Total global memory:           %u", prop->totalGlobalMem);
    DEBUGf("Total shared memory per block: %u", prop->sharedMemPerBlock);
    DEBUGf("Total registers per block:     %d", prop->regsPerBlock);
    DEBUGf("Warp size:                     %d", prop->warpSize);
    DEBUGf("Maximum memory pitch:          %u", prop->memPitch);
    DEBUGf("Maximum threads per block:     %d", prop->maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i) DEBUGf("Maximum dimension %d of block: %d", i, prop->maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i) DEBUGf("Maximum dimension %d of grid:  %d", i, prop->maxGridSize[i]);
    DEBUGf("Clock rate:                    %d", prop->clockRate);
    DEBUGf("Total constant memory:         %u", prop->totalConstMem);
    DEBUGf("Texture alignment:             %u", prop->textureAlignment);
    DEBUGf("Concurrent copy and execution: %s", (prop->deviceOverlap ? "Yes" : "No"));
    DEBUGf("Number of multiprocessors:     %d", prop->multiProcessorCount);
    DEBUGf("Kernel execution timeout:      %s", (prop->kernelExecTimeoutEnabled ? "Yes" : "No"));
}

AcousticModel::State *cuda_AM_state(AcousticModel::StateCollection *model, int id) { return model->getState(id); }
void cuda_AM_begin(AcousticModel::StateCollection *model, AcousticModel::StateCollection::Iterator *iter) { return model->beginState(iter); }
void cuda_AM_next(AcousticModel::StateCollection *model, AcousticModel::StateCollection::Iterator *iter) { return model->nextState(iter); }

int CUDA::Init() {
    FLAGS_lfapp_cuda = 0;

    int cuda_devices=0; cudaError_t err;
    if ((err = cudaGetDeviceCount(&cuda_devices)) != cudaSuccess)
    { ERROR("cudaGetDeviceCount error ", cudaGetErrorString(err)); return 0; }

    cudaDeviceProp prop;
    for (int i=0; i<cuda_devices; i++) {
        if ((err = cudaGetDeviceProperties(&prop, i)) != cudaSuccess) { ERROR("cudaGetDeviceProperties error ", err); return 0; }
        if (FLAGS_lfapp_debug) PrintCUDAProperties(&prop);
        if (strstr(prop.name, "Emulation")) continue;
        FLAGS_lfapp_cuda=1;
    }

    if (FLAGS_lfapp_cuda) {
        INFO("CUDA device detected, enabling acceleration: lfapp_cuda(", FLAGS_lfapp_cuda, ") devices ", cuda_devices);
        cudaSetDeviceFlags(cudaDeviceBlockingSync);
        cuda_init_hook();
    }
    else INFO("no CUDA devices detected ", cuda_devices);
    return 0;
}
#else
int CUDA::Init() { FLAGS_lfapp_cuda=0; INFO("CUDA not supported lfapp_cuda(", FLAGS_lfapp_cuda, ")"); return 0; }
#endif /* LFL_CUDA */

/* Script engines */

#ifdef LFL_LUA
struct MyLuaContext : public LuaContext {
    lua_State *L;
    ~MyLuaContext() { lua_close(L); }
    MyLuaContext() : L(luaL_newstate()) {
        luaopen_base(L);
        luaopen_table(L);
        luaopen_io(L);
        luaopen_string(L);
        luaopen_math(L);
    }
    string Execute(const string &s) {
        if (luaL_loadbuffer(L, s.data(), s.size(), "MyLuaExec")) { ERROR("luaL_loadstring ", lua_tostring(L, -1)); return ""; }
        if (lua_pcall(L, 0, LUA_MULTRET, 0))                     { ERROR("lua_pcall ",       lua_tostring(L, -1)); return ""; }
        return "";
    }
};
LuaContext *CreateLuaContext() { return new MyLuaContext(); }
#else /* LFL_LUA */
LuaContext *CreateLuaContext() { return 0; }
#endif /* LFL_LUA */

#ifdef LFL_V8JS
v8::Local<v8::String> NewV8String(v8::Isolate *I, const char  *s) { return v8::String::NewFromUtf8(I, s); }
v8::Local<v8::String> NewV8String(v8::Isolate *I, const short *s) { return v8::String::NewFromTwoByte(I, (const uint16_t *)s); }
template <class X> inline X CastV8InternalFieldTo(v8::Local<v8::Object> &self, int field_index) {
    return static_cast<X>(v8::Local<v8::External>::Cast(self->GetInternalField(field_index))->Value());
}
#define V8_SimpleMemberReturn(X, type, ret) \
    v8::Local<v8::Object> self = args.Holder(); \
    X *inst = CastV8InternalFieldTo<X*>(self, 1); \
    args.GetReturnValue().Set(type(args.GetIsolate(), (ret)));

#define V8_ObjectMemberReturn(X, Y, OT, ret) \
    v8::Local<v8::Object> self = args.Holder(); \
    X *impl = CastV8InternalFieldTo<X*>(self, 1); \
    Y *val = (ret); \
    if (!val) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; } \
    MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0); \
    v8::Local<v8::Object> impl_obj = (js_context->*OT)->NewInstance(); \
    impl_obj->SetInternalField(0, v8::External::New(args.GetIsolate(), js_context)); \
    impl_obj->SetInternalField(1, v8::External::New(args.GetIsolate(), val)); \
    impl_obj->SetInternalField(2, v8::Integer ::New(args.GetIsolate(), TypeId(val))); \
    args.GetReturnValue().Set(impl_obj);

template <typename X, int (X::*Y)() const> void MemberIntFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
    V8_SimpleMemberReturn(X, v8::Integer::New, (inst->*Y)());
}
template <typename X, int (X::*Y)() /***/> void MemberIntGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
    V8_SimpleMemberReturn(X, v8::Integer::New, (inst->*Y)());
}
template <typename X, int (X::*Y)() const> void MemberIntGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
    V8_SimpleMemberReturn(X, v8::Integer::New, (inst->*Y)());
}
template <typename X, DOM::DOMString (X::*Y)() const> void MemberStringFunc(const v8::FunctionCallbackInfo<v8::Value> &args) {
    V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)().c_str());
}
template <typename X, DOM::DOMString (X::*Y)() const> void MemberStringFuncGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
    V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)().c_str());
}
template <typename X, DOM::DOMString (X::*Y)(int)> void MemberStringFuncInt(const v8::FunctionCallbackInfo<v8::Value> &args) {
    if (!args.Length()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
    V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)(args[0]->Int32Value()).c_str());
}
template <typename X, DOM::DOMString (X::*Y)(int)> void IndexedMemberStringProperty(uint32_t index, const v8::PropertyCallbackInfo<v8::Value>& args) {
    V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)(index).c_str());
}
template <typename X, DOM::DOMString (X::*Y)(const DOM::DOMString &)> void NamedMemberStringProperty(v8::Local<v8::String> name, const v8::PropertyCallbackInfo<v8::Value>& args) {
    string v = BlankNull(*v8::String::Utf8Value(name));
    if (v == "toString" || v == "valueOf" || v == "length" || v == "item") return;
    V8_SimpleMemberReturn(X, NewV8String, (inst->*Y)(DOM::DOMString(v)).c_str());
}

struct MyV8JSInit { MyV8JSInit() { v8::V8::Initialize(); } };

struct MyV8JSContext : public JSContext {
    v8::Isolate*                   isolate;
    v8::Isolate::Scope             isolate_scope;
    v8::HandleScope                handle_scope;
    v8::Handle<v8::Context>        context;
    v8::Context::Scope             context_scope;
    v8::Handle<v8::ObjectTemplate> global, console, window, node, node_list, named_node_map, css_style_declaration;
    Console*                       js_console;

    virtual ~MyV8JSContext() {}
    MyV8JSContext(Console *C, DOM::Node *D) : isolate(v8::Isolate::New()), isolate_scope(isolate),
    handle_scope(isolate), context(v8::Context::New(isolate)), context_scope(context), global(v8::ObjectTemplate::New()),
    console(v8::ObjectTemplate::New()), window(v8::ObjectTemplate::New()), node(v8::ObjectTemplate::New()),
    node_list(v8::ObjectTemplate::New()), named_node_map(v8::ObjectTemplate::New()), css_style_declaration(v8::ObjectTemplate::New()),
    js_console(C) {
        console->SetInternalFieldCount(1);
        console->Set(v8::String::NewFromUtf8(isolate, "log"), v8::FunctionTemplate::New(isolate, consoleLog));
        v8::Local<v8::Object> console_obj = console->NewInstance();
        console_obj->SetInternalField(0, v8::External::New(isolate, this));
        context->Global()->Set(v8::String::NewFromUtf8(isolate, "console"), console_obj);

        window->SetInternalFieldCount(1);
        window->Set(v8::String::NewFromUtf8(isolate, "getComputedStyle"), v8::FunctionTemplate::New(isolate, windowGetComputedStyle));
        v8::Local<v8::Object> window_obj = window->NewInstance();
        window_obj->SetInternalField(0, v8::External::New(isolate, this));
        window_obj->Set(v8::String::NewFromUtf8(isolate, "console"), console_obj);
        context->Global()->Set(v8::String::NewFromUtf8(isolate, "window"), window_obj);
 
        node->SetInternalFieldCount(3);
        node->SetAccessor(v8::String::NewFromUtf8(isolate, "nodeName"),
                          MemberStringFuncGetter<DOM::Node, &DOM::Node::nodeName>, donothingSetter);
        node->SetAccessor(v8::String::NewFromUtf8(isolate, "nodeValue"),
                          MemberStringFuncGetter<DOM::Node, &DOM::Node::nodeValue>, donothingSetter);
        node->SetAccessor(v8::String::NewFromUtf8(isolate, "childNodes"), 
                          MemberObjectGetter<DOM::Node, DOM::NodeList,
                                            &DOM::Node::childNodes, &MyV8JSContext::node_list>, donothingSetter);
        node->SetAccessor(v8::String::NewFromUtf8(isolate, "attributes"), 
                          ElementObjectGetter<DOM::Node, DOM::NamedNodeMap,
                                             &DOM::Element::attributes, &MyV8JSContext::named_node_map>, donothingSetter);

        node_list->SetInternalFieldCount(3);
        node_list->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                               MemberIntGetter<DOM::NodeList, &DOM::NodeList::length>, donothingSetter);
        node_list->Set(v8::String::NewFromUtf8(isolate, "item"),
                       v8::FunctionTemplate::New(isolate, MemberObjectFuncInt<DOM::NodeList, DOM::Node, 
                                                                             &DOM::NodeList::item, &MyV8JSContext::node>));
        node_list->SetIndexedPropertyHandler(IndexedMemberObjectProperty<DOM::NodeList, DOM::Node,
                                             &DOM::NodeList::item, &MyV8JSContext::node>);

        named_node_map->SetInternalFieldCount(3);
        named_node_map->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                                    MemberIntGetter<DOM::NamedNodeMap, &DOM::NamedNodeMap::length>, donothingSetter);
        named_node_map->Set(v8::String::NewFromUtf8(isolate, "item"),
                            v8::FunctionTemplate::New(isolate, MemberObjectFuncInt<DOM::NamedNodeMap, DOM::Node, 
                                                      &DOM::NamedNodeMap::item, &MyV8JSContext::node>));
        named_node_map->SetIndexedPropertyHandler(IndexedMemberObjectProperty<DOM::NamedNodeMap, DOM::Node,
                                                  &DOM::NamedNodeMap::item, &MyV8JSContext::node>);
        named_node_map->SetNamedPropertyHandler(NamedMemberObjectProperty<DOM::NamedNodeMap, DOM::Node,
                                                &DOM::NamedNodeMap::getNamedItem, &MyV8JSContext::node>);

        css_style_declaration->SetInternalFieldCount(3);
        css_style_declaration->SetAccessor(v8::String::NewFromUtf8(isolate, "length"),
                                           MemberIntGetter<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::length>, donothingSetter);
        css_style_declaration->Set(v8::String::NewFromUtf8(isolate, "item"),
                                   v8::FunctionTemplate::New(isolate, MemberStringFuncInt<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::item>));
        css_style_declaration->SetIndexedPropertyHandler(IndexedMemberStringProperty<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::item>);
        css_style_declaration->SetNamedPropertyHandler(NamedMemberStringProperty<DOM::CSSStyleDeclaration, &DOM::CSSStyleDeclaration::getPropertyValue>);

        if (D) {
            v8::Local<v8::Object> node_obj = node->NewInstance();
            node_obj->SetInternalField(0, v8::External::New(isolate, this));
            node_obj->SetInternalField(1, v8::External::New(isolate, D));
            node_obj->SetInternalField(2, v8::Integer::New(isolate, TypeId(D)));
            context->Global()->Set(v8::String::NewFromUtf8(isolate, "document"), node_obj);
        }
    }
    string Execute(const string &s) {
        v8::Handle<v8::String> source = v8::String::NewFromUtf8(isolate, s.c_str());
        v8::Handle<v8::Script> script = v8::Script::Compile(source);
        { v8::TryCatch trycatch;
            v8::Handle<v8::Value> result = script->Run();
            if (!result.IsEmpty()) {
                if (result->IsObject() && js_console) {
                    v8::Local<v8::Object> obj = result->ToObject();
                    if (obj->InternalFieldCount() >= 3) {
                        if (obj->GetInternalField(2)->Int32Value() == TypeId<DOM::Node>()) {
                            js_console->Write(CastV8InternalFieldTo<DOM::Node*>(obj, 1)->DebugString());
                        }
                    }
                }
            } else result = trycatch.Exception();
            return BlankNull(*v8::String::Utf8Value(result));
        }
    }
    static void donothingSetter(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::PropertyCallbackInfo<void>& args) {}
    static void windowGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
        args.GetReturnValue().Set(args.Holder());
    }
    static void consoleLog(const v8::FunctionCallbackInfo<v8::Value> &args) {
        v8::Local<v8::Object> self = args.Holder(); string msg;
        MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0);
        for (int i=0; i < args.Length(); i++) StrAppend(&msg, BlankNull(*v8::String::Utf8Value(args[i]->ToString())));
        if (js_context->js_console) js_context->js_console->Write(msg);
        else INFO("VSJ8(", (void*)js_context, ") console.log: ", msg);
        args.GetReturnValue().Set(v8::Null(args.GetIsolate()));
    };
    static void windowGetComputedStyle(const v8::FunctionCallbackInfo<v8::Value> &args) {
        v8::Local<v8::Object> self = args.Holder();
        if (args.Length() < 1 || !args[0]->IsObject()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
        v8::Local<v8::Object> arg_obj = args[0]->ToObject();
        DOM::Node *impl = CastV8InternalFieldTo<DOM::Node*>(arg_obj, 1);
        DOM::CSSStyleDeclaration *val = impl->render ? &impl->render->style : 0;
        if (!val) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
        MyV8JSContext *js_context = CastV8InternalFieldTo<MyV8JSContext*>(self, 0);
        v8::Local<v8::Object> impl_obj = js_context->css_style_declaration->NewInstance();
        impl_obj->SetInternalField(0, v8::External::New(args.GetIsolate(), js_context));
        impl_obj->SetInternalField(1, v8::External::New(args.GetIsolate(), val));
        impl_obj->SetInternalField(2, v8::Integer ::New(args.GetIsolate(), TypeId(val)));
        args.GetReturnValue().Set(impl_obj);
    }
    template <typename X, typename Y, Y (X::*Z), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void MemberObjectGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
        V8_ObjectMemberReturn(X, Y, OT, &(impl->*Z));
    }
    template <typename X, typename Y, Y (DOM::Element::*Z), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void ElementObjectGetter(v8::Local<v8::String> property, const v8::PropertyCallbackInfo<v8::Value>& args) {
        V8_ObjectMemberReturn(X, Y, OT, impl->AsElement() ? &(impl->AsElement()->*Z) : 0);
    }
    template <typename X, typename Y, Y *(X::*Z)(int), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void MemberObjectFuncInt(const v8::FunctionCallbackInfo<v8::Value> &args) {
        if (!args.Length()) { args.GetReturnValue().Set(v8::Null(args.GetIsolate())); return; }
        V8_ObjectMemberReturn(X, Y, OT, (impl->*Z)(args[0]->Int32Value()));
    }
    template <typename X, typename Y, Y *(X::*Z)(int), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void IndexedMemberObjectProperty(uint32_t index, const v8::PropertyCallbackInfo<v8::Value>& args) {
        V8_ObjectMemberReturn(X, Y, OT, (impl->*Z)(index));
    }
    template <typename X, typename Y, Y *(X::*Z)(const DOM::DOMString &), v8::Handle<v8::ObjectTemplate> (MyV8JSContext::*OT)>
    static void NamedMemberObjectProperty(v8::Local<v8::String> name, const v8::PropertyCallbackInfo<v8::Value>& args) {
        string v = BlankNull(*v8::String::Utf8Value(name));
        if (v == "toString" || v == "valueOf" || v == "length" || v == "item") return;
        V8_ObjectMemberReturn(X, Y, OT, (impl->*Z)(DOM::DOMString(v)));
    }
};
JSContext *CreateV8JSContext(Console *js_console, DOM::Node *doc) { Singleton<MyV8JSInit>::Get(); return new MyV8JSContext(js_console, doc); }
#else /* LFL_V8JS */
JSContext *CreateV8JSContext(Console *js_console, DOM::Node *doc) { return 0; }
#endif /* LFL_V8JS */
}; // namespace LFL

extern "C" void BreakHook() {}
extern "C" void NotImplemented() { FATAL("not implemented"); }
extern "C" void ShellRun(const char *text) { return LFL::app->shell.Run(text); }
extern "C" NativeWindow *GetNativeWindow() { return LFL::screen; }
extern "C" LFApp        *GetLFApp()        { return LFL::app; }
extern "C" int LFAppMain()                 { return LFL::app->Main(); }
extern "C" int LFAppMainLoop()             { return LFL::app->MainLoop(); }
extern "C" int LFAppFrame()                { return LFL::app->Frame(); }
extern "C" const char *LFAppDownloadDir()  { return LFL::Singleton<LFL::DownloadDirectory>::Get()->text.c_str(); }
extern "C" void Reshaped(int w, int h)     { LFL::screen->Reshaped(w, h); }
extern "C" void Minimized()                { LFL::screen->Minimized(); }
extern "C" void UnMinimized()              { LFL::screen->UnMinimized(); }
extern "C" int  KeyPress  (int b, int d)                 { return LFL::app->input.KeyPress  (b, d); }
extern "C" int  MouseClick(int b, int d, int x,  int y)  { return LFL::app->input.MouseClick(b, d, LFL::point(x, y)); }
extern "C" int  MouseMove (int x, int y, int dx, int dy) { return LFL::app->input.MouseMove (LFL::point(x, y), LFL::point(dx, dy)); }
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
extern "C" void LFAppLog(int level, const char *file, int line, const char *fmt, ...) {
    string message;
    StringPrintfImpl(&message, fmt, vsnprintf, char, 0);
    LFL::app->Log(level, file, line, message);
}
extern "C" void LFAppFatal() {
    ERROR("LFAppFatal");
    if (bool suicide=true) *(volatile int*)0 = 0;
    LFL::app->run = 0;
    exit(-1);
}

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpCmdLine, int nCmdShow) {
    LFL::WIN32_Init();
    vector<const char *> av;
    vector<string> a(1);
    a[0].resize(1024);
	GetModuleFileName(hInst, (char*)a.data(), a.size());
	LFL::StringWordIter word_iter(lpCmdLine);
    for (auto word = word_iter.next(); word; word = word_iter.next()) a.push_back(word);
    for (auto i : a) av.push_back(i->c_str()); 
    av.push_back(0);
	return main(av.size()-1, &av[0]);
}
#endif
