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

#define  INFOf(fmt, ...) LFAppLog(LFApp::Log::Info,  __FILE__, __LINE__, fmt, __VA_ARGS__)
#define DEBUGf(fmt, ...) LFAppLog(LFApp::Log::Debug, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define ERRORf(fmt, ...) LFAppLog(LFApp::Log::Error, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define FATALf(fmt, ...) { LFAppLog(LFApp::Log::Fatal, __FILE__, __LINE__, fmt, __VA_ARGS__); throw(0); }

#define DECLARE_FLAG(name, type) extern type FLAGS_ ## name
#define DECLARE_int(name) DECLARE_FLAG(name, int)
#define DECLARE_bool(name) DECLARE_FLAG(name, bool)
#define DECLARE_float(name) DECLARE_FLAG(name, float)
#define DECLARE_double(name) DECLARE_FLAG(name, double)
#define DECLARE_string(name) DECLARE_FLAG(name, string)

#ifdef __cplusplus
namespace LFL {
DECLARE_bool(lfapp_audio);
DECLARE_bool(lfapp_video);
DECLARE_bool(lfapp_input);
DECLARE_bool(lfapp_network);
DECLARE_bool(lfapp_camera);
DECLARE_bool(lfapp_cuda);
DECLARE_bool(lfapp_wait_forever);
DECLARE_bool(lfapp_debug);
DECLARE_int(target_fps);
DECLARE_int(min_fps);
DECLARE_bool(max_rlimit_core);
DECLARE_bool(max_rlimit_open_files);
DECLARE_bool(open_console);
DECLARE_int(invert);
DECLARE_float(ksens);
DECLARE_float(msens);
}; // namespace LFL
extern "C" {
#endif

struct LFApp {
    struct Log { enum { Fatal=-1, Error=0, Info=3, Debug=7 }; int v; };
    struct Events {
        int input, mouse_click, mouse_wheel, mouse_move, mouse_hover, key, gui, bind;
    };
    bool run, initialized;
    size_t main_thread_id;
    long long frames_ran, pre_frames_ran, samples_read, samples_read_last;
};

struct NativeWindow {
    void *id, *gl, *surface, *glew_context, *user1, *user2, *user3;
    int width, height;
    bool minimized, cursor_grabbed;
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
LFApp *GetLFApp();

int LFAppMain();
int LFAppFrame();
void LFAppLog(int level, const char *file, int line, const char *fmt, ...);
void LFAppFatal();
void SetLFAppMainThread();
void Reshaped(int w, int h);
void Minimized(); 
void UnMinimized(); 
void KeyPress(int button, int down);
void MouseClick(int button, int down, int x, int y);
void MouseMove(int x, int y, int dx, int dy);
void ShellRun(const char *text);

#ifdef __cplusplus
};
#endif
#endif // __LFL_LFAPP_LFEXPORT_H__
