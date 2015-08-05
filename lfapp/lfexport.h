/*
 * $Id: lfexport.h 1327 2014-11-03 23:26:43Z justin $
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

#ifndef __LFL_LFAPP_LFEXPORT_H__
#define __LFL_LFAPP_LFEXPORT_H__

#if _WIN32 || _WIN64
 #if _WIN64
  #define LFL64
 #else
  #define LFL32
 #endif
 #define LFL_IMPORT __declspec(dllimport)
 #define LFL_EXPORT __declspec(dllexport)
 #define thread_local __declspec(thread)
 #define UNION struct
#endif

#if __GNUC__
 #if __x86_64__ || __ppc64__ || __amd64__
  #define LFL64
 #else
  #define LFL32
 #endif
 #define LFL_IMPORT
 #define LFL_EXPORT
 #define thread_local __thread
 #define UNION union
#endif

#ifdef LFL_TEST
#define tvirtual virtual
#else
#define tvirtual
#endif

#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
#define LFL_MOBILE
#endif

#if defined(__linux__) && !defined(LFL_MOBILE)
#define LFL_LINUX_SERVER 
#endif

#define memzero(x) memset(&x, 0, sizeof(x))
#define memzeros(x) memset(x, 0, sizeof(x))
#define memzerop(x) memset(x, 0, sizeof(*x))
#define sizeofarray(x) (sizeof(x) / sizeof((x)[0]))

#define M_TAU (M_PI + M_PI)
#define X_or_1(x) ((x) ? (x) : 1)
#define X_or_Y(x, y) ((x) ? (x) : (y))
#define XY_or_Y(x, y) ((x) ? ((x)*(y)) : (y))
#define Xge0_or_Y(x, y) ((x) >= 0 ? (x) : (y))
#define RoundXY_or_Y(x, y) ((x) ? RoundF((x)*(y)) : (y))
#define X_or_Y_or_Z(x, y, z) ((x) ? (x) : ((y) ? (y) : (z)))
#define A_or_B(x, y) ((x.size()) ? (x) : (y))

#define  INFOf(fmt, ...) ((::LFApp::Log::Info  <= ::LFL::FLAGS_loglevel) ? LFAppLog(LFApp::Log::Info,  __FILE__, __LINE__, fmt, __VA_ARGS__) : void())
#define DEBUGf(fmt, ...) ((::LFApp::Log::Debug <= ::LFL::FLAGS_loglevel) ? LFAppLog(LFApp::Log::Debug, __FILE__, __LINE__, fmt, __VA_ARGS__) : void())
#define ERRORf(fmt, ...) ((::LFApp::Log::Error <= ::LFL::FLAGS_loglevel) ? LFAppLog(LFApp::Log::Error, __FILE__, __LINE__, fmt, __VA_ARGS__) : void())
#define FATALf(fmt, ...) { LFAppLog(LFApp::Log::Fatal, __FILE__, __LINE__, fmt, __VA_ARGS__); throw(0); }

#define DECLARE_FLAG(name, type) extern type FLAGS_ ## name
#define DECLARE_int(name) DECLARE_FLAG(name, int)
#define DECLARE_bool(name) DECLARE_FLAG(name, bool)
#define DECLARE_float(name) DECLARE_FLAG(name, float)
#define DECLARE_double(name) DECLARE_FLAG(name, double)
#define DECLARE_string(name) DECLARE_FLAG(name, string)

#ifdef __cplusplus
struct CGRect;
struct CGSize;
struct FT_FaceRec_;
typedef struct CGContext *CGContextRef;
typedef struct CGFont *CGFontRef;
typedef const struct __CTFont *CTFontRef;

namespace google {
    namespace protobuf {
#ifdef LFL_PROTOBUF
        class Message;
#else
        class Message { int fake; };
#endif
    }; // namespace protobuf
#ifdef LFL_GLOG
    LFL_IMPORT void InstallFailureSignalHandler();
#endif
}; // namespace google

namespace LFL {
DECLARE_bool(lfapp_audio);
DECLARE_bool(lfapp_video);
DECLARE_bool(lfapp_input);
DECLARE_bool(lfapp_network);
DECLARE_bool(lfapp_camera);
DECLARE_bool(lfapp_cuda);
DECLARE_bool(lfapp_debug);
DECLARE_bool(swap_axis);
DECLARE_int(loglevel);
DECLARE_int(target_fps);
DECLARE_int(threadpool_size);
DECLARE_int(invert);
DECLARE_float(ksens);
DECLARE_float(msens);
DECLARE_bool(open_console);
DECLARE_bool(max_rlimit_core);
DECLARE_bool(max_rlimit_open_files);
DECLARE_float(rotate_view);

struct Allocator;
struct Asset;
struct Atlas;
struct Bind;
struct BindMap;
struct Box;
struct BrowserInterface;
struct Color;
struct Connection;
struct Console;
struct Dialog;
struct DocumentParser;
struct DrawableBoxArray;
struct Entity;
struct File;
struct FloatContainer;
struct Flow;
struct Font;
struct Geometry;
struct Glyph;
struct GraphicsDevice;
struct GUI;
struct InputController;
struct KeyboardGUI;
struct Listener;
struct MovieAsset;
struct MultiProcessBuffer;
struct ProtoHeader;
struct Service;
struct ServiceEndpointEraseList;
struct Shader;
struct SoundAsset;
struct StyleSheet;
struct StyleContext;
struct TextGUI;
struct Texture;
struct Terminal;
struct Tiles;
struct VideoAssetLoader;
struct Window;
namespace DOM { struct Node; };

typedef google::protobuf::Message Proto;
typedef int (*MainCB)(int argc, const char **argv);
}; // namespace LFL
extern "C" {
#endif // __cplusplus

struct AVFormatContext;
struct AVFrame;
struct AVPacket;
struct AVStream;
struct SwrContext;
struct SwsContext;
struct _IplImage;
typedef struct bio_st BIO;
typedef struct ssl_st SSL;
typedef struct ssl_ctx_st SSL_CTX;
typedef struct bignum_st BIGNUM;
typedef struct bignum_ctx BN_CTX;
typedef struct hb_face_t hb_face_t;
typedef struct _CCBigNumRef *CCBigNumRef;
typedef struct ec_group_st EC_GROUP;
typedef struct ec_point_st EC_POINT;
typedef struct ec_key_st EC_KEY;

struct LFApp {
    struct Log { enum { Fatal=-1, Error=0, Info=3, Debug=7 }; int v; };
    struct Frame { enum { DontSkip=8 }; int v; };
    struct Events {
        int input, mouse_click, mouse_wheel, mouse_move, mouse_hover, key, gui, bind;
    };
    bool run, initialized;
    size_t main_thread_id;
    long long frames_ran;
};

struct NativeWindow {
    void *id, *gl, *surface, *glew_context, *impl, *user1, *user2, *user3;
    int width, height, pow2_width, pow2_height, target_fps;
    bool minimized, cursor_grabbed, frame_init;
    int opengles_version, opengles_cubemap;
    int gesture_swipe_up, gesture_swipe_down, gesture_tap[2], gesture_dpad_stop[2];
    float gesture_dpad_x[2], gesture_dpad_y[2], gesture_dpad_dx[2], gesture_dpad_dy[2], multitouch_keyboard_x;
    LFApp::Events events;
};

void NativeWindowInit();
void NativeWindowQuit();
void NativeWindowSize(int *widthOut, int *heightOut);
int NativeWindowOrientation();
NativeWindow *GetNativeWindow();
NativeWindow *SetNativeWindow(NativeWindow*);
NativeWindow *SetNativeWindowByID(void*);
LFApp *GetLFApp();

int LFAppMain();
int LFAppMainLoop();
int LFAppFrame(bool handle_events);
void LFAppLog(int level, const char *file, int line, const char *fmt, ...);
void LFAppFatal();
void LFAppShutdown();
void SetLFAppMainThread();
void WindowReshaped(int w, int h);
void WindowMinimized(); 
void WindowUnMinimized(); 
void WindowClosed();
int KeyPress(int button, int down);
int MouseClick(int button, int down, int x, int y);
int MouseMove(int x, int y, int dx, int dy);
void EndpointRead(void*, const char *name, const char *buf, int len);
void ShellRun(const char *text);
const char *LFAppDownloadDir();
void BreakHook();

#ifdef __cplusplus
};
#endif // __cplusplus
#endif // __LFL_LFAPP_LFEXPORT_H__
