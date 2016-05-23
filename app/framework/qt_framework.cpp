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

namespace LFL {
QApplication *lfl_qapp;
static bool lfl_qt_init = false;
static std::vector<std::string> lfl_qapp_argv;  
static std::vector<const char*> lfl_qapp_av;

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

struct QtFrameworkModule : public Module {
  int Free() {
    delete lfl_qapp;
    lfl_qapp = nullptr;
    return 0;
  };
};

class QtWindow : public QWindow {
  Q_OBJECT
  public:
  QSocketNotifier *wait_forever_socket=0;
  bool init=0, grabbed=0, frame_on_mouse_input=0, frame_on_keyboard_input=0, reenable_wait_forever_socket=0;
  LFL::Window *lfl_window=0;
  point mouse_p;

  void MyInit() {
    CHECK(!lfl_window->gl.v);
    QOpenGLContext *glc = new QOpenGLContext(this);
    if (LFL::screen->gl.v) glc->setShareContext(GetTyped<QOpenGLContext*>(LFL::screen->gl));
    glc->setFormat(requestedFormat());
    glc->create();
    lfl_window->gl = MakeTyped(glc);
    LFL::app->MakeCurrentWindow(lfl_window);
    dynamic_cast<QOpenGLFunctions*>(lfl_window->gd)->initializeOpenGLFunctions();
    if (lfl_qt_init) LFL::app->StartNewWindow(lfl_window);
  }

  bool event(QEvent *event) {
    if (event->type() != QEvent::UpdateRequest) return QWindow::event(event);
    if (!init && (init = 1)) MyInit();
    if (!lfl_qt_init && (lfl_qt_init = true)) {
      MyAppMain(lfl_qapp_av.size()-1, &lfl_qapp_av[0]);
      if (!LFL::app->run) { lfl_qapp->exit(); return true; }
    }
    if (!LFL::screen || LFL::screen->impl.v != this) LFL::app->MakeCurrentWindow(lfl_window);
    app->EventDrivenFrame(true);
    if (!app->run) { lfl_qapp->exit(); return true; }
    if (reenable_wait_forever_socket && !(reenable_wait_forever_socket=0)) wait_forever_socket->setEnabled(true);
    if (LFL::screen->target_fps) RequestRender();
    return QWindow::event(event);
  }

  void resizeEvent(QResizeEvent *ev) {
    QWindow::resizeEvent(ev);
    if (!init) return; 
    LFL::app->MakeCurrentWindow(lfl_window);
    LFL::screen->Reshaped(ev->size().width(), ev->size().height());
    RequestRender();
  }

  void keyPressEvent  (QKeyEvent *ev) { keyEvent(ev, true); }
  void keyReleaseEvent(QKeyEvent *ev) { keyEvent(ev, false); }
  void keyEvent       (QKeyEvent *ev, bool down) {
    if (!init) return;
    ev->accept();
    int key = GetKeyCode(ev), fired = key ? KeyPress(key, down) : 0;
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
    int fired = LFL::app->input->MouseClick(GetMouseButton(ev), down, GetMousePosition(ev));
    if (fired && frame_on_mouse_input) RequestRender();
  }

  void mouseMoveEvent(QMouseEvent *ev) {
    QWindow::mouseMoveEvent(ev);
    if (!init) return;
    point p = GetMousePosition(ev);
    int fired = LFL::app->input->MouseMove(p, p - mouse_p);
    if (fired && frame_on_mouse_input) RequestRender();
    if (!grabbed) mouse_p = p;
    else        { mouse_p = point(width()/2, height()/2); QCursor::setPos(mapToGlobal(QPoint(mouse_p.x, mouse_p.y))); }
  }

  void RequestRender() { QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest)); }
  void DelWaitForeverSocket(Socket fd) {}
  void AddWaitForeverSocket(Socket fd) {
    CHECK(!wait_forever_socket);
    wait_forever_socket = new QSocketNotifier(fd, QSocketNotifier::Read, this);
    lfl_qapp->connect(wait_forever_socket, SIGNAL(activated(int)), this, SLOT(ReadInputChannel(int)));
  }

  static unsigned GetKeyCode(QKeyEvent *ev) { int k = ev->key(); return k < 256 && isalpha(k) ? ::tolower(k) : k; }
  static point    GetMousePosition(QMouseEvent *ev) { return point(ev->x(), LFL::screen->height - ev->y()); }
  static unsigned GetMouseButton  (QMouseEvent *ev) {
    int b = ev->button();
    if      (b == Qt::LeftButton)  return 1;
    else if (b == Qt::RightButton) return 2;
    return 0;
  }

  public slots:
  void ReadInputChannel(int fd) {
    reenable_wait_forever_socket = true; 
    wait_forever_socket->setEnabled(false);
    RequestRender(); 
  }
};

#include "app/framework/qt_framework.moc"

bool Video::CreateWindow(Window *W) {
  CHECK(!W->id.v && W->gd);
  QtWindow *my_qwin = new QtWindow();
  QWindow *qwin = my_qwin;
  my_qwin->lfl_window = W;

  QSurfaceFormat format;
  format.setSamples(16);
  qwin->setFormat(format);
  qwin->setSurfaceType(QWindow::OpenGLSurface);
  qwin->setWidth(W->width);
  qwin->setHeight(W->height);
  qwin->show();
  my_qwin->RequestRender();

  W->started = true;
  W->id = MakeTyped(qwin);
  W->impl = MakeTyped(my_qwin);
  app->windows[W->id.v] = W;
  return true;
}

void Application::CloseWindow(Window *W) {
  windows.erase(W->id.v);
  if (windows.empty()) app->run = false;
  if (app->window_closed_cb) app->window_closed_cb(W);
  screen = 0;
}

void Application::MakeCurrentWindow(Window *W) {
  screen = W; 
  GetTyped<QOpenGLContext*>(screen->gl)->makeCurrent(GetTyped<QWindow*>(screen->id));
}

void Window::SetCaption(const string &v) { GetTyped<QWindow*>(id)->setTitle(QString::fromUtf8(v.data(), v.size())); }
void Window::SetResizeIncrements(float x, float y) { GetTyped<QWindow*>(id)->setSizeIncrement(QSize(x, y)); }
void Window::SetTransparency(float v) { GetTyped<QWindow*>(id)->setOpacity(1-v); }
void Window::Reshape(int w, int h) { GetTyped<QWindow*>(id)->resize(w, h); app->MakeCurrentWindow(this); }
void Video::StartWindow(Window*) {}

int Video::Swap() {
  GetTyped<QOpenGLContext*>(screen->gl)->swapBuffers(GetTyped<QWindow*>(screen->id));
  screen->gd->CheckForError(__FILE__, __LINE__);
  return 0;
}

void Application::LoseFocus() {}
void Application::GrabMouseFocus()    { GetTyped<QtWindow*>(screen->impl)->grabbed=1; GetTyped<QWindow*>(screen->id)->setCursor(Qt::BlankCursor); app->grab_mode.On();  screen->cursor_grabbed=true;  }
void Application::ReleaseMouseFocus() { GetTyped<QtWindow*>(screen->impl)->grabbed=0; GetTyped<QWindow*>(screen->id)->unsetCursor();              app->grab_mode.Off(); screen->cursor_grabbed=false; }
void Application::OpenTouchKeyboard() {}
int Application::GetVolume() { return 0; }
int Application::GetMaxVolume() { return 0; }
void Application::SetVolume(int v) {}
void Application::ShowAds() {}
void Application::HideAds() {}

string Application::GetClipboardText() { QByteArray v = QApplication::clipboard()->text().toUtf8(); return string(v.constData(), v.size()); }
void Application::SetClipboardText(const string &s) { QApplication::clipboard()->setText(QString::fromUtf8(s.data(), s.size())); }

bool FrameScheduler::DoWait() { return false; }
void FrameScheduler::Setup() { rate_limit = synchronize_waits = wait_forever_thread = monolithic_frame = run_main_loop = 0; }
void FrameScheduler::Wakeup(Window *w) { if (wait_forever && w) GetTyped<QtWindow*>(w->impl)->RequestRender(); }
bool FrameScheduler::WakeupIn(Window *w, Time interval, bool force) { return false; }
void FrameScheduler::ClearWakeupIn(Window*) {}
void FrameScheduler::UpdateWindowTargetFPS(Window *w) { /*QTTriggerFrame(w->id.value);*/ }
void FrameScheduler::AddWaitForeverMouse(Window *w) { if (w) GetTyped<QtWindow*>(w->impl)->frame_on_mouse_input = true; }
void FrameScheduler::DelWaitForeverMouse(Window *w) { if (w) GetTyped<QtWindow*>(w->impl)->frame_on_mouse_input = false; }
void FrameScheduler::AddWaitForeverKeyboard(Window *w) { if (w) GetTyped<QtWindow*>(w->impl)->frame_on_keyboard_input = true; }
void FrameScheduler::DelWaitForeverKeyboard(Window *w) { if (w) GetTyped<QtWindow*>(w->impl)->frame_on_keyboard_input = false; }
void FrameScheduler::AddWaitForeverSocket(Window *w, Socket fd, int flag) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Add(fd, flag, w);
  if (w) GetTyped<QtWindow*>(w->impl)->AddWaitForeverSocket(fd);
}
void FrameScheduler::DelWaitForeverSocket(Window *w, Socket fd) {
  if (wait_forever && wait_forever_thread) wakeup_thread.Del(fd);
  if (w) GetTyped<QtWindow*>(w->impl)->DelWaitForeverSocket(fd);
}

#if 0
#include <QMessageBox>
void Dialog::MessageBox(const string &n) {
  app->ReleaseMouseFocus();
  QMessageBox *msg = new QMessageBox();
  msg->setAttribute(Qt::WA_DeleteOnClose);
  msg->setText("MesssageBox");
  msg->setInformativeText(n.c_str());
  msg->setModal(false);
  msg->open();
}
#endif

extern "C" int main(int argc, const char *argv[]) {
  MyAppCreate();
  for (int i=0; i<argc; i++) PushBack(lfl_qapp_argv, argv[i]);
  for (auto &a : lfl_qapp_argv) lfl_qapp_av.push_back(a.c_str());
  lfl_qapp_av.push_back(0);
  lfl_qapp = new QApplication(argc, const_cast<char**>(argv));
  LFL::screen->gd = LFL::CreateGraphicsDevice(2).release();
  Video::CreateWindow(LFL::screen);
  return lfl_qapp->exec();
}

unique_ptr<Module> CreateFrameworkModule() { return make_unique<QtFrameworkModule>(); }

}; // namespace LFL
