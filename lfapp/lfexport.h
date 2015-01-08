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
#ifdef __cplusplus
extern "C" {
#endif

struct LFApp {
    struct Events {
        int input, mouse_click, mouse_wheel, mouse_move, mouse_hover, key, gui, bind;
    };
    bool run, opened;
    long long main_thread_id, frames_ran, pre_frames_ran, samples_read, samples_read_last;
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

int LFAppMain();
int LFAppFrame();
void Reshaped(int w, int h);
void Minimized(); 
void UnMinimized(); 
void KeyPress(int button, int down, int, int);
void MouseClick(int button, int down, int x, int y);
void ShellRun(const char *text);

#ifdef __cplusplus
};
#endif
#endif // __LFL_LFAPP_LFEXPORT_H__
