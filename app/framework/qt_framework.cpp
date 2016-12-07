/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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

#include <QtOpenGL>
#include <QApplication>
#include "qt_common.h"

namespace LFL {
QApplication *qapp;
static bool qapp_init = false;

const int Key::Escape     = Qt::Key_Escape;
const int Key::Return     = Qt::Key_Return;
const int Key::Up         = Qt::Key_Up;
const int Key::Down       = Qt::Key_Down;
const int Key::Left       = Qt::Key_Left;
const int Key::Right      = Qt::Key_Right;
const int Key::LeftShift  = Qt::Key_Shift;
const int Key::RightShift = -8;
const int Key::LeftCtrl   = Qt::Key_Meta;
const int Key::RightCtrl  = -10;
const int Key::LeftCmd    = Qt::Key_Control;
const int Key::RightCmd   = -12;
const int Key::Tab        = Qt::Key_Tab;
const int Key::Space      = Qt::Key_Space;
const int Key::Backspace  = Qt::Key_Backspace;
const int Key::Delete     = Qt::Key_Delete;
const int Key::Quote      = Qt::Key_Apostrophe;
const int Key::Backquote  = Qt::Key_QuoteLeft;
const int Key::PageUp     = Qt::Key_PageUp;
const int Key::PageDown   = Qt::Key_PageDown;
const int Key::F1         = Qt::Key_F1;
const int Key::F2         = Qt::Key_F2;
const int Key::F3         = Qt::Key_F3;
const int Key::F4         = Qt::Key_F4;
const int Key::F5         = Qt::Key_F5;
const int Key::F6         = Qt::Key_F6;
const int Key::F7         = Qt::Key_F7;
const int Key::F8         = Qt::Key_F8;
const int Key::F9         = Qt::Key_F9;
const int Key::F10        = Qt::Key_F10;
const int Key::F11        = Qt::Key_F11;
const int Key::F12        = Qt::Key_F12;
const int Key::Home       = Qt::Key_Home;
const int Key::End        = Qt::Key_End;
const int Key::Insert     = Qt::Key_Insert;

struct QtFrameworkModule : public Module {
  int Free() {
    delete qapp;
    qapp = nullptr;
    return 0;
  };
};

class QtWindow : public QWindow, public QtWindowInterface {
  Q_OBJECT
  public:
  unordered_map<Socket, unique_ptr<QSocketNotifier>> main_wait_socket;
  bool init=0, grabbed=0, frame_on_mouse_input=0, frame_on_keyboard_input=0;
  LFL::Window *lfl_window=0;
  point mouse_p;

  void MyWindowInit() {
    CHECK(!lfl_window->gl.v);
    QOpenGLContext *glc = new QOpenGLContext(this);
    if (app->focused->gl.v) glc->setShareContext(GetTyped<QOpenGLContext*>(app->focused->gl));
    glc->setFormat(requestedFormat());
    glc->create();
    lfl_window->gl = MakeTyped(glc);
    app->MakeCurrentWindow(lfl_window);
    dynamic_cast<QOpenGLFunctions*>(lfl_window->gd)->initializeOpenGLFunctions();
    if (qapp_init) app->StartNewWindow(lfl_window);
  }

  bool event(QEvent *event) {
    if (event->type() != QEvent::UpdateRequest) return QWindow::event(event);
    if (!init && (init = 1)) MyWindowInit();
    if (!qapp_init && (qapp_init = true)) {
      MyAppMain();
      if (!app->run) { qapp->exit(); return true; }
    }
    if (!app->focused || app->focused->impl.v != this) app->MakeCurrentWindow(lfl_window);
    app->EventDrivenFrame(true);
    if (!app->run) { qapp->exit(); return true; }
    if (app->focused->target_fps) RequestRender();
    return QWindow::event(event);
  }

  void resizeEvent(QResizeEvent *ev) {
    QWindow::resizeEvent(ev);
    if (!init) return; 
    app->MakeCurrentWindow(lfl_window);
    app->focused->Reshaped(Box(ev->size().width(), ev->size().height()));
    RequestRender();
  }

  void keyPressEvent  (QKeyEvent *ev) { keyEvent(ev, true); }
  void keyReleaseEvent(QKeyEvent *ev) { keyEvent(ev, false); }
  void keyEvent       (QKeyEvent *ev, bool down) {
    if (!init) return;
    ev->accept();
    int key = GetKeyCode(ev), fired = key ? app->input->KeyPress(key, 0, down) : 0;
    if (fired && frame_on_keyboard_input) RequestRender();
  }

  void mouseReleaseEvent(QMouseEvent *ev) { QWindow::mouseReleaseEvent(ev); mouseClickEvent(ev, false); }
  void mousePressEvent  (QMouseEvent *ev) { QWindow::mousePressEvent(ev);   mouseClickEvent(ev, true);
#if 0
    if (ev->button() == Qt::RightButton) {
      QMenu menu;
      QAction* openAct = new QAction("Open...", this);
      menu.addAction(openAct);
      menu.addSeparator();
      menu.exec(mapToGlobal(ev->pos()));
    }
#endif
  }

  void mouseClickEvent(QMouseEvent *ev, bool down) {
    if (!init) return;
    int fired = app->input->MouseClick(GetMouseButton(ev), down, GetMousePosition(ev));
    if (fired && frame_on_mouse_input) RequestRender();
  }

  void mouseMoveEvent(QMouseEvent *ev) {
    QWindow::mouseMoveEvent(ev);
    if (!init) return;
    point p = GetMousePosition(ev);
    int fired = app->input->MouseMove(p, p - mouse_p);
    if (fired && frame_on_mouse_input) RequestRender();
    if (!grabbed) mouse_p = p;
    else        { mouse_p = point(width()/2, height()/2); QCursor::setPos(mapToGlobal(QPoint(mouse_p.x, mouse_p.y))); }
  }

  void RequestRender() { QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest)); }
  void DelMainWaitSocket(Socket fd) { main_wait_socket.erase(fd); }
  void AddMainWaitSocket(Socket fd, function<bool()> readable) {
    auto notifier = make_unique<QSocketNotifier>(fd, QSocketNotifier::Read, this);
    qapp->connect(notifier.get(), &QSocketNotifier::activated, [=](Socket fd){
      if (readable && readable()) RequestRender();
    });
    main_wait_socket[fd] = move(notifier);
  }

  static unsigned GetKeyCode(QKeyEvent *ev) { int k = ev->key(); return k < 256 && isalpha(k) ? ::tolower(k) : k; }
  static point    GetMousePosition(QMouseEvent *ev) { return point(ev->x(), app->focused->height - ev->y()); }
  static unsigned GetMouseButton  (QMouseEvent *ev) {
    int b = ev->button();
    if      (b == Qt::LeftButton)  return 1;
    else if (b == Qt::RightButton) return 2;
    return 0;
  }
};

#include "app/framework/qt_framework.moc"

bool Video::CreateWindow(Window *W) {
  CHECK(!W->id.v && W->gd);
  QMainWindow *main_window = new QMainWindow();
  QtWindow *my_qwin = new QtWindow();
  QtWindowInterface *my_qwin_interface = my_qwin;
  my_qwin->window = main_window;
  my_qwin->lfl_window = W;
  my_qwin->opengl_window = my_qwin;

  QSurfaceFormat format;
  format.setSamples(16);
  my_qwin->setFormat(format);
  my_qwin->setSurfaceType(QWindow::OpenGLSurface);

  W->started = true;
  W->id = MakeTyped(my_qwin_interface);
  W->impl = MakeTyped(my_qwin);

  my_qwin->opengl_container = QWidget::createWindowContainer(my_qwin, main_window);
  my_qwin->opengl_container->setFocusPolicy(Qt::TabFocus);
  my_qwin->layout = new QStackedLayout(main_window);
  my_qwin->window->setCentralWidget(new QWidget(main_window));
  my_qwin->window->centralWidget()->setLayout(my_qwin->layout);
  my_qwin->layout->addWidget(my_qwin->opengl_container);
  my_qwin->window->resize(W->width, W->height);
  my_qwin->window->show();
  my_qwin->opengl_container->setFocus();
  my_qwin->RequestRender();

  app->windows[W->id.v] = W;
  return true;
}

void Application::CloseWindow(Window *W) {
  windows.erase(W->id.v);
  if (windows.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  focused = 0;
}

void Application::MakeCurrentWindow(Window *W) {
  focused = W; 
  GetTyped<QOpenGLContext*>(focused->gl)->makeCurrent(GetTyped<QtWindowInterface*>(focused->id)->opengl_window);
}

void Window::SetCaption(const string &v) { GetTyped<QtWindowInterface*>(id)->window->setWindowTitle(MakeQString(v)); }
void Window::SetResizeIncrements(float x, float y) { GetTyped<QtWindowInterface*>(id)->window->windowHandle()->setSizeIncrement(QSize(x, y)); }
void Window::SetTransparency(float v) { GetTyped<QtWindowInterface*>(id)->window->windowHandle()->setOpacity(1-v); }
bool Window::Reshape(int w, int h) { GetTyped<QtWindow*>(impl)->window->resize(w, h); app->MakeCurrentWindow(this); return true; }

void Video::StartWindow(Window*) {}
int Video::Swap() {
  Window *screen = app->focused;
  GetTyped<QOpenGLContext*>(screen->gl)->swapBuffers(GetTyped<QtWindowInterface*>(screen->id)->opengl_window);
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void Application::LoseFocus() {}
void Application::GrabMouseFocus()    { auto screen=app->focused; GetTyped<QtWindow*>(screen->impl)->grabbed=1; GetTyped<QtWindowInterface*>(screen->id)->window->windowHandle()->setCursor(Qt::BlankCursor); screen->grab_mode.On();  screen->cursor_grabbed=true;  }
void Application::ReleaseMouseFocus() { auto screen=app->focused; GetTyped<QtWindow*>(screen->impl)->grabbed=0; GetTyped<QtWindowInterface*>(screen->id)->window->windowHandle()->unsetCursor();              screen->grab_mode.Off(); screen->cursor_grabbed=false; }
void Application::ToggleTouchKeyboard() {}
void Application::OpenTouchKeyboard() {}
void Application::CloseTouchKeyboard() {}
void Application::CloseTouchKeyboardAfterReturn(bool v) {}
void Application::SetTouchKeyboardTiled(bool v) {}
int Application::SetExtraScale(bool v) { return false; }
void Application::SetDownScale(bool v) {}
void Application::SetTitleBar(bool v) {}
void Application::SetKeepScreenOn(bool v) {}
void Application::SetAutoRotateOrientation(bool v) {}
void Application::SetPanRecognizer(bool enabled) {}
void Application::SetPinchRecognizer(bool enabled) {}
void Application::SetVerticalSwipeRecognizer(int touches) {}
void Application::SetHorizontalSwipeRecognizer(int touches) {}
void Application::ShowSystemStatusBar(bool v) {}

string Application::GetClipboardText() { return GetQString(QApplication::clipboard()->text()); }
void Application::SetClipboardText(const string &s) { QApplication::clipboard()->setText(MakeQString(s)); }

bool FrameScheduler::DoMainWait() { return false; }
void FrameScheduler::Setup() { rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = run_main_loop = 0; }
void FrameScheduler::Wakeup(Window *w) { if (wait_forever && w) GetTyped<QtWindow*>(w->impl)->RequestRender(); }
bool FrameScheduler::WakeupIn(Window *w, Time interval, bool force) { return false; }
void FrameScheduler::ClearWakeupIn(Window*) {}
void FrameScheduler::UpdateWindowTargetFPS(Window *w) { /*QTTriggerFrame(w->id.value);*/ }
void FrameScheduler::AddMainWaitMouse(Window *w) { if (w) GetTyped<QtWindow*>(w->impl)->frame_on_mouse_input = true; }
void FrameScheduler::DelMainWaitMouse(Window *w) { if (w) GetTyped<QtWindow*>(w->impl)->frame_on_mouse_input = false; }
void FrameScheduler::AddMainWaitKeyboard(Window *w) { if (w) GetTyped<QtWindow*>(w->impl)->frame_on_keyboard_input = true; }
void FrameScheduler::DelMainWaitKeyboard(Window *w) { if (w) GetTyped<QtWindow*>(w->impl)->frame_on_keyboard_input = false; }

void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()> f) {
  if (fd == InvalidSocket) return;
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, w);
  else if (w) GetTyped<QtWindow*>(w->impl)->AddMainWaitSocket(fd, move(f));
}

void FrameScheduler::DelMainWaitSocket(Window *w, Socket fd) {
  if (fd == InvalidSocket) return;
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
  else if (w) GetTyped<QtWindow*>(w->impl)->DelMainWaitSocket(fd);
}

extern "C" int main(int argc, const char *argv[]) {
  MyAppCreate(argc, argv);
  qapp = new QApplication(argc, const_cast<char**>(argv));
  app->focused->gd = CreateGraphicsDevice(app->focused, 2).release();
  Video::CreateWindow(app->focused);
  auto w = GetTyped<QtWindow*>(app->focused->impl);
  if ((w->init = true)) { w->MyWindowInit(); app->DrawSplash(); }
  return qapp->exec();
}

unique_ptr<Module> CreateFrameworkModule() { return make_unique<QtFrameworkModule>(); }

}; // namespace LFL
