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

#include <windowsx.h>
#include <objbase.h>
#include <commdlg.h>
#include "GL/glew.h"
#include "GL/wglew.h"
#include "core/app/shell.h"
#include "core/app/framework/windows_common.h"

namespace LFL {
const int Key::Escape     = '\x1b';
const int Key::Return     = '\r';
const int Key::Up         = 0xf00 | VK_UP;
const int Key::Down       = 0xf00 | VK_DOWN;
const int Key::Left       = 0xf00 | VK_LEFT;
const int Key::Right      = 0xf00 | VK_RIGHT;
const int Key::LeftShift  = 0xf00 | VK_SHIFT; // VK_LSHIFT;
const int Key::RightShift = 0xf00 | VK_RSHIFT;
const int Key::LeftCtrl   = 0xf00 | VK_CONTROL; // VK_LCONTROL;
const int Key::RightCtrl  = 0xf00 | VK_RCONTROL;
const int Key::LeftCmd    = 0xf00 | VK_MENU;
const int Key::RightCmd   = -12;
const int Key::Tab        = 0xf00 | VK_TAB;
const int Key::Space      = 0xf00 | VK_SPACE;
const int Key::Backspace  = '\b';
const int Key::Delete     = 0xf00 | VK_DELETE;
const int Key::Quote      = '\'';
const int Key::Backquote  = '`';
const int Key::PageUp     = 0xf00 | VK_PRIOR;
const int Key::PageDown   = 0xf00 | VK_NEXT;
const int Key::F1         = 0xf00 | VK_F1;
const int Key::F2         = 0xf00 | VK_F2;
const int Key::F3         = 0xf00 | VK_F3;
const int Key::F4         = 0xf00 | VK_F4;
const int Key::F5         = 0xf00 | VK_F5;
const int Key::F6         = 0xf00 | VK_F6;
const int Key::F7         = 0xf00 | VK_F7;
const int Key::F8         = 0xf00 | VK_F8;
const int Key::F9         = 0xf00 | VK_F9;
const int Key::F10        = 0xf00 | VK_F10;
const int Key::F11        = 0xf00 | VK_F11;
const int Key::F12        = 0xf00 | VK_F12;
const int Key::Home       = 0xf00 | VK_HOME;
const int Key::End        = 0xf00 | VK_END;
const int Key::Insert     = 0xf00 | VK_INSERT;

const int Texture::updatesystemimage_pf = Pixel::RGB24;

HINSTANCE WindowsFrameworkModule::hInst;
int       WindowsFrameworkModule::nCmdShow;

struct WindowsAlertView : public AlertViewInterface {
  void Hide() {}
  void Show(const string &arg) {}
  void ShowCB(const string &title, const string &msg, const string &arg, StringCB confirm_cb) {}
  string RunModal(const string &arg) { return string(); }
};

struct WindowsMenuView : public MenuViewInterface {
  WindowsMenuView(WindowsWindow *win, const string &title, MenuItemVec items) {
    if (!win->menu) { win->menu = CreateMenu(); win->context_menu = CreatePopupMenu(); }
    HMENU hAddMenu = CreatePopupMenu();
    for (auto &i : items) {
      if (i.name == "<separator>") AppendMenu(hAddMenu, MF_MENUBARBREAK, 0, NULL);
      else AppendMenu(hAddMenu, MF_STRING, win->start_msg_id + win->menu_cmds.size(), i.name.c_str());
      // win->menu_cmds.push_back(i.cmd);
    }
    AppendMenu(win->menu, MF_STRING | MF_POPUP, (UINT)hAddMenu, title.c_str());
    AppendMenu(win->context_menu, MF_STRING | MF_POPUP, (UINT)hAddMenu, title.c_str());
    if (win->menubar) SetMenu(win->hwnd, win->menu);
  }
  void Show() {}
};

struct WindowsPanelView : public PanelViewInterface {
  void Show() {}
  void SetTitle(const string &title) {}
};

struct WindowsNag : public NagInterface {
};

int WindowsWindow::Swap() {
  gd->Flush();
  SwapBuffers(surface);
  gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void WindowsWindow::Wakeup(int) {
  auto w = dynamic_cast<WindowsWindow*>(this);
  InvalidateRect(w->hwnd, NULL, 0);
  // PostMessage(w->hwnd, WM_USER, 0, 0);
}

void WindowsWindow::SetCaption(const string &v) { SetWindowText(hwnd, v.c_str()); }
void WindowsWindow::SetResizeIncrements(float x, float y) { resize_increment = point(x, y); }

void WindowsWindow::SetTransparency(float v) {
  if (v <= 0) SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) & (~WS_EX_LAYERED));
  else {
    SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) | (WS_EX_LAYERED));
    SetLayeredWindowAttributes(hwnd, 0, BYTE(max(1.0, (1 - v)*255.0)), LWA_ALPHA);
  }
}

bool WindowsWindow::Reshape(int w, int h) {
  long lStyle = GetWindowLong(hwnd, GWL_STYLE);
  RECT r = { 0, 0, w, h };
  AdjustWindowRect(&r, lStyle, menubar);
  SetWindowPos(hwnd, 0, 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
  return true;
}

void WindowsWindow::UpdateMousePosition(const LPARAM &lParam, point *p, point *d) {
  *p = point(GET_X_LPARAM(lParam), gl_h - GET_Y_LPARAM(lParam));
  *d = *p - prev_mouse_pos;
  prev_mouse_pos = *p;
}

bool WindowsWindow::RestrictResize(int m, RECT *r) {
  point in(r->right - r->left, r->bottom - r->top);
  RECT w = { 0, 0, in.x, in.y };
  AdjustWindowRect(&w, GetWindowLong(hwnd, GWL_STYLE), menubar);
  point extra((w.right - w.left) - in.x, (w.bottom - w.top) - in.y), content = in - extra;
  switch (m) {
    case WMSZ_TOP:         r->top    -= (NextMultipleOfN(content.y, resize_increment.y) - content.y); break;
    case WMSZ_BOTTOM:      r->bottom += (NextMultipleOfN(content.y, resize_increment.y) - content.y); break;
    case WMSZ_LEFT:        r->left   -= (NextMultipleOfN(content.x, resize_increment.x) - content.x); break;
    case WMSZ_RIGHT:       r->right  += (NextMultipleOfN(content.x, resize_increment.x) - content.x); break;
    case WMSZ_BOTTOMLEFT:  r->bottom += (NextMultipleOfN(content.y, resize_increment.y) - content.y);
                           r->left   -= (NextMultipleOfN(content.x, resize_increment.x) - content.x); break;
    case WMSZ_BOTTOMRIGHT: r->right  += (NextMultipleOfN(content.x, resize_increment.x) - content.x);
                           r->bottom += (NextMultipleOfN(content.y, resize_increment.y) - content.y); break;
    case WMSZ_TOPLEFT:     r->top    -= (NextMultipleOfN(content.y, resize_increment.y) - content.y);
                           r->left   -= (NextMultipleOfN(content.x, resize_increment.x) - content.x); break;
    case WMSZ_TOPRIGHT:    r->right  += (NextMultipleOfN(content.x, resize_increment.x) - content.x);
                           r->top    -= (NextMultipleOfN(content.y, resize_increment.y) - content.y); break;
  }
  return true;
}

LRESULT APIENTRY WindowsWindow::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
  PAINTSTRUCT ps;
  POINT cursor;
  point p, d;
  int ind, w, h;
  switch (message) {
  case WM_CREATE:                      return 0;
  case WM_DESTROY:                     parent->Shutdown(); PostQuitMessage(0); return 0;
  case WM_SIZE:                        w = LOWORD(lParam); h = HIWORD(lParam); if (w != gl_w || h != gl_h) { Reshaped(point(w, h), Box(0, 0, w, h)); Wakeup(); } return 0;
  case WM_KEYUP:   case WM_SYSKEYUP:   if (parent->input->KeyPress(WindowsFrameworkModule::GetKeyCode(wParam), 0, 0) && frame_on_keyboard_input) Wakeup(); return 0;
  case WM_KEYDOWN: case WM_SYSKEYDOWN: if (parent->input->KeyPress(WindowsFrameworkModule::GetKeyCode(wParam), 0, 1) && frame_on_keyboard_input) Wakeup(); return 0;
  case WM_LBUTTONDOWN:                 if (parent->input->MouseClick(1, 1, point(prev_mouse_pos.x, prev_mouse_pos.y)) && frame_on_mouse_input) Wakeup(); return 0;
  case WM_LBUTTONUP:                   if (parent->input->MouseClick(1, 0, point(prev_mouse_pos.x, prev_mouse_pos.y)) && frame_on_mouse_input) Wakeup(); return 0;
  case WM_RBUTTONDOWN:                 if (parent->input->MouseClick(2, 1, point(prev_mouse_pos.x, prev_mouse_pos.y)) && frame_on_mouse_input) Wakeup(); return 0;
  case WM_RBUTTONUP:                   if (parent->input->MouseClick(2, 0, point(prev_mouse_pos.x, prev_mouse_pos.y)) && frame_on_mouse_input) Wakeup(); return 0;
  case WM_MOUSEMOVE:                   UpdateMousePosition(lParam, &p, &d); if (parent->input->MouseMove(point(p.x, p.y), point(d.x, d.y)) && frame_on_mouse_input) Wakeup(); return 0;
  case WM_COMMAND:                     if ((ind = wParam - start_msg_id) >= 0) if (ind < menu_cmds.size()) shell->Run(menu_cmds[ind].c_str()); return 0;
  case WM_CONTEXTMENU:                 if (menu) { GetCursorPos(&cursor); TrackPopupMenu(context_menu, TPM_LEFTALIGN | TPM_TOPALIGN, cursor.x, cursor.y, 0, hWnd, NULL); } return 0;
  case WM_PAINT:                       BeginPaint(hwnd, &ps); if (!FLAGS_target_fps) parent->EventDrivenFrame(true, true); EndPaint(hwnd, &ps); return 0;
  case WM_SIZING:                      return resize_increment.Zero() ? 0 : RestrictResize(wParam, reinterpret_cast<LPRECT>(lParam));
  case WM_USER:                        if (!FLAGS_target_fps) parent->EventDrivenFrame(true, true); return 0;
  case WM_KILLFOCUS:                   parent->input->ClearButtonsDown(); return 0;
  default:                             break;
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

LRESULT APIENTRY WindowsWindow::WndProcDispatch(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
  WindowsWindow *win = 0;
  if (message == WM_NCCREATE) {
    auto pCreate = static_cast<CREATESTRUCT*>(Void(lParam));
    win = static_cast<WindowsWindow*>(pCreate->lpCreateParams);
#ifdef _WIN64
    SetWindowLongPtr(hWnd, 0, win);
#else
    SetWindowLongPtr(hWnd, 0, LONG(win));
#endif
  } else win = static_cast<WindowsWindow*>(Void(GetWindowLongPtr(hWnd, 0)));
  return win->WndProc(hWnd, message, wParam, lParam);
}

unique_ptr<Window> WindowsFrameworkModule::ConstructWindow(Application *A) { return make_unique<WindowsWindow>(A); }
void WindowsFrameworkModule::StartWindow(Window *W) {}
bool WindowsFrameworkModule::CreateWindow(WindowHolder *H, Window *Win) {
  auto W = dynamic_cast<WindowsWindow*>(Win);
  auto app = W->parent;
  RECT r = { 0, 0, W->gl_w, W->gl_h };
  DWORD dwStyle = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
  if (!AdjustWindowRect(&r, dwStyle, 0)) return ERRORv(false, "AdjustWindowRect");
  HWND hWnd = CreateWindowEx(WS_EX_LEFT, app->name.c_str(), W->caption.c_str(), dwStyle, 0, 0, r.right - r.left, r.bottom - r.top, NULL, NULL, WindowsFrameworkModule::hInst, W);
  if (!hWnd) return ERRORv(false, "CreateWindow: ", GetLastErrorText());
  HDC hDC = GetDC(hWnd);
  PIXELFORMATDESCRIPTOR pfd = { sizeof(PIXELFORMATDESCRIPTOR), 1, PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER,
    PFD_TYPE_RGBA, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0, };
  int pf = ChoosePixelFormat(hDC, &pfd);
  if (!pf) return ERRORv(false, "ChoosePixelFormat: ", GetLastErrorText());
  if (SetPixelFormat(hDC, pf, &pfd) != TRUE) return ERRORv(false, "SetPixelFormat: ", GetLastErrorText());
  if (!(W->gl = wglCreateContext(hDC))) return ERRORv(false, "wglCreateContext: ", GetLastErrorText());
  W->surface = hDC;
  W->hwnd = hWnd;
  app->windows[(W->id = hWnd)] = W;
  INFOf("Application::CreateWindow %p %p %p (%p)", W->id, W->surface, W->gl, W);
  app->MakeCurrentWindow(W);
  ShowWindow(hWnd, WindowsFrameworkModule::nCmdShow);
  W->Wakeup();
  return true;
}

#if 0
void *WindowsFrameworkModule::BeginGLContextCreate(Window *Win) {
  auto W = dynamic_cast<WindowsWindow*>(Win);
  if (wglewIsSupported("WGL_ARB_create_context")) return wglCreateContextAttribsARB((HDC)W->surface, (HGLRC)W->gl, 0);
  else { HGLRC ret = wglCreateContext((HDC)W->surface); wglShareLists((HGLRC)W->gl, ret); return ret; }
}

void *WindowsFrameworkModule::CompleteGLContextCreate(Window *W, void *gl_context) {
  wglMakeCurrent((HDC)dynamic_cast<WindowsWindow*>(W)->surface, (HGLRC)gl_context);
  return gl_context;
}
#endif

int WindowsFrameworkModule::Init() {
  INFO("WindowsFrameworkModule::Init()");
  CreateClass(window->focused->parent);
  CHECK(CreateWindow(window, window->focused));
  return 0;
}

void WindowsFrameworkModule::CreateClass(Application *app) {
  WNDCLASS wndClass = {};
  wndClass.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
  wndClass.lpfnWndProc = &WindowsWindow::WndProcDispatch;
  wndClass.cbWndExtra = sizeof(WindowsWindow*);
  wndClass.cbClsExtra = 0;
  wndClass.hInstance = hInst;
  wndClass.hIcon = LoadIcon(hInst, "IDI_APP_ICON");
  wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
  if (auto c = app->splash_color) wndClass.hbrBackground = CreateSolidBrush(RGB(c->R(), c->G(), c->B()));
  else                            wndClass.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
  wndClass.lpszMenuName = NULL;
  wndClass.lpszClassName = app->name.c_str();
  if (!RegisterClass(&wndClass)) ERROR("RegisterClass: ", GetLastErrorText());
}

int WindowsFrameworkModule::MessageLoop() {
  INFOf("WinApp::MessageLoop %p", window->focused);
  CoInitialize(NULL);
  MSG msg;
  auto app = window->focused->parent;
  while (app->run) {
    if (app->run && FLAGS_target_fps) app->TimerDrivenFrame(true);
    while (app->run && PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); }
    if (app->run && !FLAGS_target_fps) if (GetMessage(&msg, NULL, 0, 0)) { TranslateMessage(&msg); DispatchMessage(&msg); }
  }
  app->Exit();
  CoUninitialize();
  return msg.wParam;
}

int WindowsFrameworkModule::GetKeyCode(unsigned char k) {
  const unsigned short key_code[] = {
    0xF00, /*null*/              0xF01, /*Left mouse*/     0xF02, /*Right mouse*/     0xF03, /*Control-break*/
    0xF04, /*Middle mouse*/      0xF05, /*X1 mouse*/       0xF06, /*X2 mouse*/        0xF07, /*Undefined*/
    '\b',  /*BACKSPACE key*/     '\t',  /*TAB key*/        0xF0A, /*Reserved*/        0xF0B, /*Reserved*/
    0xF0C, /*CLEAR key*/         '\r',  /*ENTER key*/      0xF0E, /*Undefined*/       0xF0F, /*Undefined*/
    0xF10, /*SHIFT key*/         0xF11, /*CTRL key*/       0xF12, /*ALT key*/         0xF13, /*PAUSE key*/
    0xF14, /*CAPS LOCK key*/     0xF15, /*IME Kana mode*/  0xF16, /*Undefined*/       0xF17, /*IME Junja mode*/
    0xF18, /*IME final mode*/    0xF19, /*IME Hanja mode*/ 0xF1A, /*Undefined*/       '\x1b',/*ESC key*/
    0xF1C, /*IME convert*/       0xF1D, /*IME nonconvert*/ 0xF1E, /*IME accept*/      0xF1F, /*IME mode change*/
    ' ',   /*SPACEBAR*/          0xF21, /*PAGE UP key*/    0xF22, /*PAGE DOWN key*/   0xF23, /*END key*/
    0xF24, /*HOME key*/          0xF25, /*LEFT ARROW key*/ 0xF26, /*UP ARROW key*/    0xF27, /*RIGHT ARROW key*/
    0xF28, /*DOWN ARROW key*/    0xF29, /*SELECT key*/     0xF2A, /*PRINT key*/       0xF2B, /*EXECUTE key*/
    0xF2C, /*PRINT SCREEN key*/  0xF2D, /*INS key*/        0xF2E, /*DEL key*/         0xF2F, /*HELP key*/
    '0',   /*0 key*/             '1',   /*1 key*/          '2',   /*2 key*/           '3',   /*3 key*/
    '4',   /*4 key*/             '5',   /*5 key*/          '6',   /*6 key*/           '7',   /*7 key*/
    '8',   /*8 key*/             '9',   /*9 key*/          0xF3A, /*undefined*/       0xF3B, /*undefined*/
    0xF3C, /*undefined*/         0xF3D, /*undefined*/      0xF3E, /*undefined*/       0xF3F, /*undefined*/
    0xF40, /*undefined*/         'a',   /*A key*/          'b',   /*B key*/           'c',   /*C key*/
    'd',   /*D key*/             'e',   /*E key*/          'f',   /*F key*/           'g',   /*G key*/
    'h',   /*H key*/             'i',   /*I key*/          'j',   /*J key*/           'k',   /*K key*/
    'l',   /*L key*/             'm',   /*M key*/          'n',   /*N key*/           'o',   /*O key*/
    'p',   /*P key*/             'q',   /*Q key*/          'r',   /*R key*/           's',   /*S key*/
    't',   /*T key*/             'u',   /*U key*/          'v',   /*V key*/           'w',   /*W key*/
    'x',   /*X key*/             'y',   /*Y key*/          'z',   /*Z key*/           0xF5B, /*Left Windows key*/
    0xF5C, /*Right Windows key*/ 0xF5D, /*Apps-key*/       0xF5E, /*Reserved*/        0xF5F, /*Computer Sleep*/
    0xF60, /*keypad 0 key*/      0xF61, /*keypad 1 key*/   0xF62, /*keypad 2 key*/    0xF63, /*keypad 3 key*/
    0xF64, /*keypad 4 key*/      0xF65, /*keypad 5 key*/   0xF66, /*keypad 6 key*/    0xF67, /*keypad 7 key*/
    0xF68, /*keypad 8 key*/      0xF69, /*keypad 9 key*/   0xF6A, /*Multiply key*/    0xF6B, /*Add key*/
    0xF6C, /*Separator key*/     0xF6D, /*Subtract key*/   0xF6E, /*Decimal key*/     0xF6F, /*Divide key*/
    0xF70, /*F1 key*/            0xF71, /*F2 key*/         0xF72, /*F3 key*/          0xF73, /*F4 key*/
    0xF74, /*F5 key*/            0xF75, /*F6 key*/         0xF76, /*F7 key*/          0xF77, /*F8 key*/
    0xF78, /*F9 key*/            0xF79, /*F10 key*/        0xF7A, /*F11 key*/         0xF7B, /*F12 key*/
    0xF7C, /*F13 key*/           0xF7D, /*F14 key*/        0xF7E, /*F15 key*/         0xF7F, /*F16 key*/
    0xF80, /*F17 key*/           0xF81, /*F18 key*/        0xF82, /*F19 key*/         0xF83, /*F20 key*/
    0xF84, /*F21 key*/           0xF85, /*F22 key*/        0xF86, /*F23 key*/         0xF87, /*F24 key*/
    0xF88, /*Unassigned*/        0xF89, /*Unassigned*/     0xF8A, /*Unassigned*/      0xF8B, /*Unassigned*/
    0xF8C, /*Unassigned*/        0xF8D, /*Unassigned*/     0xF8E, /*Unassigned*/      0xF8F, /*Unassigned*/
    0xF90, /*NUM LOCK key*/      0xF91, /*SCROLL LOCK*/    0xF92, /*OEM specific*/    0xF93, /*OEM specific*/
    0xF94, /*OEM specific*/      0xF95, /*OEM specific*/   0xF96, /*OEM specific*/    0xF97, /*Unassigned*/
    0xF98, /*Unassigned*/        0xF99, /*Unassigned*/     0xF9A, /*Unassigned*/      0xF9B, /*Unassigned*/
    0xF9C, /*Unassigned*/        0xF9D, /*Unassigned*/     0xF9E, /*Unassigned*/      0xF9F, /*Unassigned*/
    0xFA0, /*Left SHIFT key*/    0xFA1, /*Right SHIFT*/    0xFA2, /*Left CONTROL*/    0xFA3, /*Right CONTROL*/
    0xFA4, /*Left MENU key*/     0xFA5, /*Right MENU*/     0xFA6, /*Browser Back*/    0xFA7, /*Browser Forward*/
    0xFA8, /*Browser Refresh*/   0xFA9, /*Browser Stop*/   0xFAA, /*Browser Search*/  0xFAB, /*Browser Favorites*/
    0xFAC, /*Browser Home*/      0xFAD, /*Volume Mute*/    0xFAE, /*Volume Down*/     0xFAF, /*Volume Up*/
    0xFB0, /*Next Track key*/    0xFB1, /*Previous Track*/ 0xFB2, /*Stop Media*/      0xFB3, /*Play/Pause Media*/
    0xFB4, /*Start Mail key*/    0xFB5, /*Select Media*/   0xFB6, /*Start App-1 key*/ 0xFB7, /*Start Application 2*/
    0xFB8, /*Reserved*/          0xFB9, /*Reserved*/       ';',   /*the ';:' key*/    '=',  /*the '+' key*/
    ',',   /*the ',' key*/       '-',   /*the '-' key*/    '.',   /*the '.' key*/     '/',  /*the '/?' key*/
    '\`',  /*the '`~' key*/      0xFC1, /*Reserved*/       0xFC2, /*Reserved*/        0xFC3, /*Reserved*/
    0xFC4, /*Reserved*/          0xFC5, /*Reserved*/       0xFC6, /*Reserved*/        0xFC7, /*Reserved*/
    0xFC8, /*Reserved*/          0xFC9, /*Reserved*/       0xFCA, /*Reserved*/        0xFCB, /*Reserved*/
    0xFCC, /*Reserved*/          0xFCD, /*Reserved*/       0xFCE, /*Reserved*/        0xFCF, /*Reserved*/
    0xFD0, /*Reserved*/          0xFD1, /*Reserved*/       0xFD2, /*Reserved*/        0xFD3, /*Reserved*/
    0xFD4, /*Reserved*/          0xFD5, /*Reserved*/       0xFD6, /*Reserved*/        0xFD7, /*Reserved*/
    0xFD8, /*Unassigned*/        0xFD9, /*Unassigned*/     0xFDA, /*Unassigned*/      '[',  /*the '[{' key*/
    '\\',  /*the '\|' key*/      ']',  /*the ']}' key*/    '\'',  /*'quote' key*/     0xFDF, /*misc*/
    0xFE0, /*Reserved*/          0xFE1, /*OEM specific*/   0xFE2, /*RT 102 bracket*/  0xFE3, /*OEM specific*/
    0xFE4, /*OEM specific*/      0xFE5, /*IME PROCESS*/    0xFE6, /*OEM specific*/    0xFE7, /*Used for Unicode*/
    0xFE8, /*Unassigned*/        0xFE9, /*OEM specific*/   0xFEA, /*OEM specific*/    0xFEB, /*OEM specific*/
    0xFEC, /*OEM specific*/      0xFED, /*OEM specific*/   0xFEE, /*OEM specific*/    0xFEF, /*OEM specific*/
    0xFF0, /*OEM specific*/      0xFF1, /*OEM specific*/   0xFF2, /*OEM specific*/    0xFF3, /*OEM specific*/
    0xFF4, /*OEM specific*/      0xFF5, /*OEM specific*/   0xFF6, /*Attn key*/        0xFF7, /*CrSel key*/
    0xFF8, /*ExSel key*/         0xFF9, /*Erase EOF key*/  0xFFA, /*Play key*/        0xFFB, /*Zoom key*/
    0xFFC, /*Reserved*/          0xFFD, /*PA1 key*/        0xFFE, /*Clear key*/       0xFFF, /*Unused*/
  };
  return key_code[k];
}

string WindowsFrameworkModule::GetLastErrorText() {
  DWORD error = GetLastError();
  if (!error) return string();
  LPSTR b = nullptr;
  size_t l = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                            NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&b, 0, NULL);
  string error_text(b, l);
  LocalFree(b);
  return StrCat(error, ": ", error_text);
}

void WindowHolder::MakeCurrentWindow(Window *W) { if (auto w = dynamic_cast<WindowsWindow*>(W)) wglMakeCurrent(w->surface, w->gl); }

void Application::CloseWindow(Window *W) {
  windows.erase(W->id);
  if (windows.empty()) run = false;
  if (window_closed_cb) window_closed_cb(W);
  focused = nullptr;
}

void Application::ShowSystemFontChooser(const FontDesc &cur_font, const StringVecCB &choose_cmd) {
  LOGFONT lf;
  memzero(lf);
  HDC hdc = GetDC(NULL);
  lf.lfHeight = -MulDiv(cur_font.size, GetDeviceCaps(hdc, LOGPIXELSY), 72);
  lf.lfWeight = (cur_font.flag & FontDesc::Bold) ? FW_BOLD : FW_NORMAL;
  lf.lfItalic = cur_font.flag & FontDesc::Italic;
  strncpy(lf.lfFaceName, cur_font.name.c_str(), sizeof(lf.lfFaceName) - 1);
  ReleaseDC(NULL, hdc);
  CHOOSEFONT cf;
  memzero(cf);
  cf.lpLogFont = &lf;
  cf.lStructSize = sizeof(cf);
  cf.hwndOwner = dynamic_cast<WindowsWindow*>(focused)->hwnd;
  cf.Flags = CF_SCREENFONTS | CF_INITTOLOGFONTSTRUCT;
  if (!ChooseFont(&cf)) return;
  int flag = FontDesc::Mono | (lf.lfWeight > FW_NORMAL ? FontDesc::Bold : 0) | (lf.lfItalic ? FontDesc::Italic : 0);
  focused->shell->Run(StrCat(choose_cmd, " ", lf.lfFaceName, " ", cf.iPointSize / 10, " ", flag));
}

int Application::Suspended() { return 0; }
int Application::SetExtraScale(bool v) { return false; }
void Application::SetDownScale(bool v) {}
void Application::SetTitleBar(bool v) {}
void Application::SetKeepScreenOn(bool v) {}
void Application::SetAutoRotateOrientation(bool v) {}
void Application::SetVerticalSwipeRecognizer(int touches) {}
void Application::SetHorizontalSwipeRecognizer(int touches) {}
void Application::SetPanRecognizer(bool enabled) {}
void Application::SetPinchRecognizer(bool enabled) {}
void Application::ShowSystemStatusBar(bool v) {}
void Application::SetTheme(const string &v) {}

void ThreadDispatcher::RunCallbackInMainThread(Callback cb) {
  message_queue.Write(make_unique<Callback>(move(cb)).release());
  if (!FLAGS_target_fps) wakeup->Wakeup();
}

void MouseFocus::ReleaseMouseFocus() {}
void MouseFocus::GrabMouseFocus() {}

void TouchKeyboard::ToggleTouchKeyboard() {}
void TouchKeyboard::OpenTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboardAfterReturn(bool v) {}
void TouchKeyboard::SetTouchKeyboardTiled(bool v) {}

void Clipboard::SetClipboardText(const string &in) {
  String16 s = String::ToUTF16(in);
  if (!OpenClipboard(NULL)) return;
  EmptyClipboard();
  HGLOBAL hg = GlobalAlloc(GMEM_MOVEABLE, (s.size()+1)*2);
  if (!hg) { CloseClipboard(); return; }
  memcpy(GlobalLock(hg), s.c_str(), (s.size()+1)*2);
  GlobalUnlock(hg);
  SetClipboardData(CF_UNICODETEXT, hg);
  CloseClipboard();
  GlobalFree(hg);
}

string Clipboard::GetClipboardText() {
  string ret;
  if (!OpenClipboard(NULL)) return "";
  const HANDLE hg = GetClipboardData(CF_UNICODETEXT);
  if (!hg) { CloseClipboard(); return ""; }
  ret = String::ToUTF8(reinterpret_cast<char16_t*>(GlobalLock(hg)));
  GlobalUnlock(hg);
  CloseClipboard();
  return ret;
}

FrameScheduler::FrameScheduler(WindowHolder *w) :
  window(w), maxfps(&FLAGS_target_fps), rate_limit(1), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(1), run_main_loop(0) {}

bool FrameScheduler::DoMainWait(bool only_poll) { return false;  }
void FrameScheduler::UpdateWindowTargetFPS(Window*) {}
void FrameScheduler::AddMainWaitMouse(Window *w) { dynamic_cast<WindowsWindow*>(w)->frame_on_mouse_input = true; }
void FrameScheduler::DelMainWaitMouse(Window *w) { dynamic_cast<WindowsWindow*>(w)->frame_on_mouse_input = false; }
void FrameScheduler::AddMainWaitKeyboard(Window *w) { dynamic_cast<WindowsWindow*>(w)->frame_on_keyboard_input = true; }
void FrameScheduler::DelMainWaitKeyboard(Window *w) { dynamic_cast<WindowsWindow*>(w)->frame_on_keyboard_input = false; }

void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()>) {
  if (fd != InvalidSocket) WSAAsyncSelect(fd, dynamic_cast<WindowsWindow*>(w)->hwnd, WM_USER, FD_READ | FD_CLOSE);
}

void FrameScheduler::DelMainWaitSocket(Window *w, Socket fd) {
  if (fd != InvalidSocket) WSAAsyncSelect(fd, dynamic_cast<WindowsWindow*>(w)->hwnd, WM_USER, 0);
}

unique_ptr<Framework> Framework::Create(Application *a) { return make_unique<WindowsFrameworkModule>(a); }
unique_ptr<TimerInterface> SystemToolkit::CreateTimer(Callback cb) { return nullptr; }
unique_ptr<AlertViewInterface> SystemToolkit::CreateAlert(Window *w, AlertItemVec items) { return make_unique<WindowsAlertView>(); }
unique_ptr<PanelViewInterface> SystemToolkit::CreatePanel(Window *w, const Box &b, const string &title, PanelItemVec items) { return nullptr; }
unique_ptr<MenuViewInterface> SystemToolkit::CreateMenu(Window *w, const string &title, MenuItemVec items) { return make_unique<WindowsMenuView>(dynamic_cast<WindowsWindow*>(w), title, move(items)); }
unique_ptr<MenuViewInterface> SystemToolkit::CreateEditMenu(Window *w, vector<MenuItem> items) { return nullptr; }
unique_ptr<NagInterface> SystemToolkit::CreateNag(const string &id, int min_days, int min_uses, int min_events, int remind_days) { return make_unique<WindowsNag>(); }
}; // namespace LFL

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR lpCmdLine, int nCmdShow) {
  std::vector<const char *> av;
  std::vector<std::string> a(1);
  a[0].resize(1024);
  GetModuleFileName(hInst, &(a[0])[0], a[0].size());
  LFL::StringWordIter word_iter(lpCmdLine);
  for (std::string word = word_iter.NextString(); !word_iter.Done(); word = word_iter.NextString()) a.push_back(word);
  for (auto &i : a) av.push_back(i.c_str());
  av.push_back(0);

  auto app = static_cast<LFL::Application*>(MyAppCreate(av.size() - 1, &av[0]));
  LFL::WindowsFrameworkModule::hInst = hInst;
  LFL::WindowsFrameworkModule::nCmdShow = nCmdShow;
  int ret = MyAppMain(app);
  return ret ? ret : dynamic_cast<LFL::WindowsFrameworkModule*>(app->framework.get())->MessageLoop();
}
