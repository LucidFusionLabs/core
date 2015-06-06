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
#include "lfapp/resolver.h"

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

#ifdef LFL_ANDROID
#include <android/log.h>
#endif

#ifdef LFL_OPENSSL
#include "openssl/md5.h"
#include "openssl/err.h"
#include "openssl/dh.h"
#include "openssl/bn.h"
#endif

extern "C" {
#ifdef LFL_LUA
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
#endif
};

#ifdef LFL_V8JS
#include <v8.h>
#endif

#if defined(LFL_IPHONE)
extern "C" char *iPhoneDocumentPath();
extern "C" void iPhoneLog(const char *text);
extern "C" void iPhoneOpenBrowser(const char *url_text);
extern "C" void iPhoneTriggerFrame(void*);
#elif defined(__APPLE__)
extern "C" void OSXStartWindow(void*);
extern "C" void OSXCreateNativeMenu(const char*, int, const char**, const char**);
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
extern "C" int LFAppFrame()                { return LFL::app->Frame(); }
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
DEFINE_int(loglevel, 7, "Log level: [Fatal=-1, Error=0, Info=3, Debug=7]");
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
void RunInMainThread(Callback *cb) { app->message_queue.Write(cb); }
void DefaultLFAppWindowClosedCB(Window *W) { delete W; }
double FPS() { return screen->fps.FPS(); }
double CamFPS() { return app->camera.fps.FPS(); }
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
bool NBReadable(int fd, int timeout) {
    SelectSocketSet ss;
    ss.Add(fd, SocketSet::READABLE, 0);
    ss.Select(timeout);
    return app->run && ss.GetReadable(fd);
}
int NBRead(int fd, char *buf, int len, int timeout) {
    if (!NBReadable(fd, timeout)) return 0;
    int o = 0, s = 0;
    do if ((s = ::read(fd, buf+o, len-o)) > 0) o += s;
    while (s > 0 && len - o > 1024);
    return o;
}
int NBRead(int fd, string *buf, int timeout) {
    int l = NBRead(fd, (char*)buf->data(), buf->size(), timeout);
    buf->resize(max(0,l));
    return l;
}
string NBRead(int fd, int len, int timeout) {
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

#ifdef LFL_OPENSSL
string Crypto::MD5(const string &in) {
    string out(MD5_DIGEST_LENGTH, 0);
    ::MD5((const unsigned char*)in.c_str(), in.size(), (unsigned char*)out.data());
    return out;
}

string Crypto::SHA1(const string &in) {
    void *ctx = NewSHA1();
    UpdateSHA1(ctx, in);
    return FinishSHA1(ctx);
}

void *Crypto::NewSHA1() {
    EVP_MD_CTX *ctx = (EVP_MD_CTX*)calloc(sizeof(EVP_MD_CTX), 1);
    EVP_DigestInit(ctx, EVP_get_digestbyname("sha1"));
    return ctx;
}
void Crypto::UpdateSHA1(void *ctx, const StringPiece &in) { EVP_DigestUpdate((EVP_MD_CTX*)ctx, in.data(), in.size()); }
string Crypto::FinishSHA1(void *ctx) {
    unsigned len = 0;
    string ret(EVP_MAX_MD_SIZE, 0);
    EVP_DigestFinal((EVP_MD_CTX*)ctx, reinterpret_cast<unsigned char *>(&ret[0]), &len);
    ret.resize(len);
    free(ctx);
    return ret;
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

string Crypto::DiffieHellmanModulus(int generator, int bits) {
    DH *dh = DH_new();
    DH_generate_parameters_ex(dh, bits, generator, NULL);
    string ret(BN_num_bytes(dh->p), 0);
    BN_bn2bin(dh->p, (unsigned char*)&ret[0]);
    DH_free(dh);
    return ret;
}

Crypto::CipherAlgo Crypto::CipherAlgos::DES3() { return EVP_des_ede3_cbc(); }
Crypto::MACAlgo Crypto::MACAlgos::SHA1() { return EVP_sha1(); }
void Crypto::CipherInit(Cipher *c) { EVP_CIPHER_CTX_init(c); }
void Crypto::CipherFree(Cipher *c) { EVP_CIPHER_CTX_cleanup(c); }
int Crypto::CipherGetBlockSize(Cipher *c) { return EVP_CIPHER_CTX_block_size(c); }
int Crypto::CipherOpen(Cipher *c, CipherAlgo algo, bool dir, const StringPiece &key, const StringPiece &IV) { 
    return EVP_CipherInit(c, algo, reinterpret_cast<const unsigned char *>(key.data()),
                          reinterpret_cast<const unsigned char *>(IV.data()), dir);
}
int Crypto::CipherUpdate(Cipher *c, const StringPiece &in, char *out, int outlen) {
    return EVP_Cipher(c, reinterpret_cast<unsigned char*>(out),
                      reinterpret_cast<const unsigned char*>(in.data()), in.size());
}
void Crypto::MACOpen(MAC *m, MACAlgo algo, const StringPiece &k) { HMAC_Init(m, k.data(), k.size(), algo); }
void Crypto::MACUpdate(MAC *m, const StringPiece &in) { HMAC_Update(m, reinterpret_cast<const unsigned char *>(in.data()), in.size()); }
int Crypto::MACFinish(MAC *m, char *out, int outlen) { unsigned len=outlen; HMAC_Final(m, reinterpret_cast<unsigned char *>(out), &len); return len; }
#else
string Crypto::MD5(const string &in) { FATAL("not implemented"); }
string Crypto::SHA1(const string &in) { FATAL("not implemented"); }
void *Crypto::NewSHA1() { FATAL("not implemented"); }
void Crypto::UpdateSHA1(void *ctx, const StringPiece &in) { FATAL("not implemented"); }
string Crypto::FinishSHA1(void *ctx) { FATAL("not implemented"); }
string Crypto::Blowfish(const string &passphrase, const string &in, bool encrypt_or_decrypt) { FATAL("not implemented"); }
string Crypto::DiffieHellmanModulus(int generator, int bits) { FATAL("not implemented"); }
Crypto::CipherAlgo Crypto::CipherAlgos::DES3() { FATAL("not implemented"); }
Crypto::MACAlgo Crypto::MACAlgos::SHA1() { FATAL("not implemented"); }
void Crypto::CipherInit(Cipher *c) { FATAL("not implemented"); }
void Crypto::CipherFree(Cipher *c) { FATAL("not implemented"); }
int Crypto::CipherGetBlockSize(Cipher *c) { FATAL("not implemented"); }
int Crypto::CipherOpen(Cipher *c, CipherAlgo algo, bool dir, const StringPiece &key, const StringPiece &IV) {  FATAL("not implemented"); }
int Crypto::CipherUpdate(Cipher *c, const StringPiece &in, char *out, int outlen) { FATAL("not implemented"); }
void Crypto::MACOpen(MAC *m, MACAlgo algo, const StringPiece &k) { FATAL("not implemented"); }
void Crypto::MACUpdate(MAC *m, const StringPiece &in) { FATAL("not implemented"); }
int Crypto::MACFinish(MAC *m, char *out, int outlen) { FATAL("not implemented"); }
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
#ifdef LFL_IPHONE
        iPhoneLog(StringPrintf("%s (%s:%d)", message.c_str(), file, line).c_str());
#endif
#ifdef LFL_ANDROID
        __android_log_print(ANDROID_LOG_INFO, screen->caption.c_str(), "%s (%s:%d)", message.c_str(), file, line);
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

NetworkThread *Application::CreateNetworkThread() {
    VectorEraseByValue(&app->modules, static_cast<Module*>(&network));
    NetworkThread *ret = new NetworkThread(&app->network);
    ret->thread->Start();
    return ret;
}

void Application::AddNativeMenu(const string &title, const vector<pair<string, string>>&items) {
    vector<const char *> k, v;
    for (auto &i : items) { k.push_back(i.first.c_str()); v.push_back(i.second.c_str()); }
#ifdef LFL_OSXVIDEO
    OSXCreateNativeMenu(title.c_str(), items.size(), &k[0], &v[0]);
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
#ifndef LFL_ANDROID
    assetdir = "assets/";
#endif

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

    {
#ifdef LFL_IPHONE
        char *path = iPhoneDocumentPath();
        dldir = StrCat(path, "/");
        free(path);
#endif
#ifdef _WIN32
        char path[MAX_PATH];
        if (!SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL|CSIDL_FLAG_CREATE, NULL, 0, path))) return -1;
        dldir = StrCat(path, "/");
#endif
    }

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
        if (LoadModule(&audio)) { ERROR("audio init failed"); return -1; }
    }
    else { FLAGS_chans_in=FLAGS_chans_out=1; }

    if (FLAGS_lfapp_camera) {
        if (LoadModule(&camera)) { ERROR("camera init failed"); return -1; }
    }

    if (FLAGS_lfapp_video) {
        if (video.Init()) { ERROR("video init failed"); return -1; }
    } else {
        Window::active[screen->id] = screen;
    }

    if (FLAGS_lfapp_input) {
        if (LoadModule(&input)) { ERROR("input init failed"); return -1; }
        input.Init(screen);
    }

    if (FLAGS_lfapp_network) {
        if (LoadModule(&network)) { ERROR("network init failed"); return -1; }
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

int Application::PreFrame(unsigned clicks) {
    pre_frames_ran++;

    for (auto i = modules.begin(); i != modules.end() && run; ++i) (*i)->Frame(clicks);

    // handle messages sent to main thread
    if (run) message_queue.HandleMessages();

    // fake threadpool that executes in main thread
    if (run && !FLAGS_threadpool_size) thread_pool.worker[0].queue->HandleMessages();

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
    unsigned clicks = (scheduler.monolithic_frame ? frame_time.GetTime(true) : screen->frame_time.GetTime(true)).count();

    int flag = 0;
    PreFrame(clicks);

    if (scheduler.monolithic_frame) {
        Window *previous_screen = screen;
        for (auto i = Window::active.begin(); run && i != Window::active.end(); ++i) {
            int ret = i->second->Frame(clicks, audio.mic_samples, camera.have_sample, flag);
            if (FLAGS_frame_debug) INFO("frame_debug Application::Frame Window ", i->second->id, " = ", ret);
        }
        if (previous_screen && previous_screen != screen) Window::MakeCurrent(previous_screen);
    } else {
        int ret = screen->Frame(clicks, audio.mic_samples, camera.have_sample, flag);
        if (FLAGS_frame_debug) INFO("frame_debug Application::Frame Window ", screen->id, " = ", ret);
    }

    PostFrame();

    return clicks;
}

int Application::Main() {
#ifdef LFL_IPHONE
    ONCE({ return 0; });
#endif
    if (Start()) return Exiting();
#if defined(LFL_OSXVIDEO) || defined(LFL_QT) || defined(LFL_WXWIDGETS)
    return 0;
#endif
    return MainLoop();
}
    
int Application::MainLoop() {
    INFO("MainLoop: Begin, run=", run);
    while (run) {
        // if (!minimized)
        Frame();
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
#if defined(LFL_OSXINPUT) || defined(LFL_IPHONEINPUT) || defined(LFL_QT) || defined(LFL_WXWIDGETS)
    rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = 0;
#endif
#if defined(LFL_QT) || defined(LFL_WXWIDGETS)
    wait_forever_thread = true;
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
#if defined(LFL_OSXINPUT) || defined(LFL_IPHONEINPUT) || defined(LFL_QT) || defined(LFL_WXWIDGETS)
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
    if (wait_forever) {
#if defined(LFL_QT)
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
#elif defined(LFL_OSXINPUT)
        OSXTriggerFrame(screen->id);
#elif defined(LFL_IPHONEINPUT)
        iPhoneTriggerFrame(screen->id);
#else
        // FATAL("not implemented");
#endif
    }
}

bool FrameScheduler::WakeupIn(void *opaque, Time interval, bool force) {
    // CHECK(!screen->target_fps);
#if defined(LFL_OSXINPUT)
    return OSXTriggerFrameIn(screen->id, interval.count(), force);
#endif
    return 0;
}

void FrameScheduler::ClearWakeupIn() {
#if defined(LFL_OSXINPUT)
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
#if defined(LFL_OSXINPUT)
    OSXUpdateTargetFPS(screen->id);
#endif
}

void FrameScheduler::AddWaitForeverMouse() {
    CHECK(screen->id);
#if defined(LFL_OSXINPUT)
    OSXAddWaitForeverMouse(screen->id);
#endif
}

void FrameScheduler::DelWaitForeverMouse() {
    CHECK(screen->id);
#if defined(LFL_OSXINPUT)
    OSXDelWaitForeverMouse(screen->id);
#endif
}

void FrameScheduler::AddWaitForeverKeyboard() {
    CHECK(screen->id);
#if defined(LFL_OSXINPUT)
    OSXAddWaitForeverKeyboard(screen->id);
#endif
}

void FrameScheduler::DelWaitForeverKeyboard() {
    CHECK(screen->id);
#if defined(LFL_OSXINPUT)
    OSXDelWaitForeverKeyboard(screen->id);
#endif
}

void FrameScheduler::AddWaitForeverSocket(Socket fd, int flag, void *val) {
    if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, val);
#ifdef LFL_OSXINPUT
    if (!wait_forever_thread) { CHECK_EQ(SocketSet::READABLE, flag); OSXAddWaitForeverSocket(screen->id, fd); }
#endif
}

void FrameScheduler::DelWaitForeverSocket(Socket fd) {
    if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
#if defined(LFL_OSXINPUT)
    CHECK(screen->id);
    OSXDelWaitForeverSocket(screen->id, fd);
#endif
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

int CUDA::Init() {
    INFO("CUDA::Init()");
    FLAGS_lfapp_cuda = 0;

    int cuda_devices = 0;
    cudaError_t err;
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

#ifdef _WIN32
int close(Socket socket) { return closesocket(socket); }
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpCmdLine, int nCmdShow) {
    timeBeginPeriod(1);
    vector<const char *> av;
    vector<string> a(1);
    a[0].resize(1024);
    GetModuleFileName(hInst, (char*)a.data(), a.size());
    LFL::StringWordIter word_iter(lpCmdLine);
    for (string word = IterNextString(&word_iter); !word_iter.Done(); word = IterNextString(&word_iter)) a.push_back(word);
    for (auto &i : a) av.push_back(i.c_str());
    av.push_back(0);
    return main(av.size() - 1, &av[0]);
}
#endif
