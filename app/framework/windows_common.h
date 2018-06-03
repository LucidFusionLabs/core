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

#ifndef LFL_CORE_APP_FRAMEWORK_WINDOWS_COMMON_H__
#define LFL_CORE_APP_FRAMEWORK_WINDOWS_COMMON_H__
namespace LFL {

struct WindowsWindow : public Window {
  HWND hwnd = 0;
  HGLRC gl = 0;
  HDC surface = 0;
  bool menubar = 0, frame_on_keyboard_input = 0, frame_on_mouse_input = 0;
  point prev_mouse_pos, resize_increment;
  int start_msg_id = WM_USER + 100;
  HMENU menu = 0, context_menu = 0;
  vector<string> menu_cmds;

  WindowsWindow(Application *a) : Window(a) {}
  ~WindowsWindow() { ClearChildren(); }
  int Swap();
  void Wakeup(int flag=0);
  bool RestrictResize(int m, RECT*);
  void SetCaption(const string &v);
  void SetResizeIncrements(float x, float y);
  void SetTransparency(float v);
  bool Reshape(int w, int h);
  void UpdateMousePosition(const LPARAM &lParam, point *p, point *d);
  LRESULT APIENTRY WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
  static LRESULT APIENTRY WndProcDispatch(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
};


struct WindowsFrameworkModule : public Framework {
  static HINSTANCE hInst;
  static int nCmdShow;
  WindowHolder *window;
  WindowsFrameworkModule(WindowHolder *w) : window(w) {}

  int Init() override;
  void CreateClass(Application*);
  unique_ptr<Window> ConstructWindow(Application*) override;
  bool CreateWindow(WindowHolder *H, Window *W) override;
  void StartWindow(Window *W) override;
  int MessageLoop();

  static int GetKeyCode(unsigned char k);
  static string GetLastErrorText();
};

}; // namespace LFL
#endif // LFL_CORE_APP_FRAMEWORK_WINDOWS_COMMON_H__