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

const int Texture::updatesystemimage_pf = Pixel::RGB24;

StringVec GetQStringList(const QStringList &v) {
  StringVec ret;
  for (auto i = v.constBegin(), e = v.constEnd(); i != e; ++i) ret.push_back(GetQString(*i));
  return ret;
}

void CleanUpTextureQImageData(void *info) {
  delete [] static_cast<unsigned char *>(info);
}

QImage MakeQImage(Texture &t) {
  if (!t.buf) return QImage();
  auto buf = t.ReleaseBuffer().release();
  return QImage(buf, t.width, t.height, t.LineSize(), PixelToQImageFormat(t.pf),
                CleanUpTextureQImageData, buf);
}

QImage::Format PixelToQImageFormat(int pf) {
  switch (pf) {
    case Pixel::RGB32:  return QImage::Format_RGB32;
    case Pixel::BGR32:  return QImage::Format_RGB32;
    case Pixel::RGB24:  return QImage::Format_RGB888;
    case Pixel::BGR24:  return QImage::Format_RGB888;
    case Pixel::RGBA:   return QImage::Format_ARGB32;
    case Pixel::BGRA:   return QImage::Format_ARGB32;
#if 0
    case Pixel::ALPHA8: return QImage::Format_Alpha8;
    case Pixel::GRAY8:  return QImage::Format_Grayscale8;
#endif
    default:            return QImage::Format_Invalid;
  }
}

int PixelFromQImageFormat(QImage::Format fmt) {
  switch (fmt) {
    case QImage::Format_ARGB32:     return Pixel::RGBA;
    case QImage::Format_RGB32:      return Pixel::RGB32;
    case QImage::Format_RGB888:     return Pixel::RGB24;
#if 0
    case QImage::Format_Alpha8:     return Pixel::ALPHA8;
    case QImage::Format_Grayscale8: return Pixel::GRAY8;
#endif
    default: return 0;
  }
}

class QtWindow : public QWindow, public QtWindowInterface {
  Q_OBJECT
  public:
  unordered_map<Socket, unique_ptr<QSocketNotifier>> main_wait_socket;
  bool init=0, grabbed=0, frame_on_mouse_input=0, frame_on_keyboard_input=0;
  point mouse_p;
  QtWindow(Application *a, QScreen *parent = 0) : QWindow(parent), QtWindowInterface(a) {}

  void MyWindowInit() {
    CHECK(!glc);
    glc = new QOpenGLContext(this);
    auto app = QtWindowInterface::parent;
    if (auto sglc = dynamic_cast<QtWindow*>(app->focused)->glc) glc->setShareContext(sglc);
    glc->setFormat(requestedFormat());
    glc->create();
    app->MakeCurrentWindow(this);
    dynamic_cast<QOpenGLFunctions*>(gd)->initializeOpenGLFunctions();
    if (qapp_init) app->StartNewWindow(this);
  }

  bool event(QEvent *event) {
    if (event->type() != QEvent::UpdateRequest) return QWindow::event(event);
    if (!init && (init = 1)) MyWindowInit();
    auto app = QtWindowInterface::parent;
    if (!qapp_init && (qapp_init = true)) {
      MyAppMain(app);
      if (!app->run) { qapp->exit(); return true; }
    }
    if (!app->focused || app->focused != static_cast<LFL::Window*>(this)) app->MakeCurrentWindow(this);
    app->EventDrivenFrame(true, true);
    if (!app->run) { qapp->exit(); return true; }
    if (app->focused->target_fps) RequestRender();
    return QWindow::event(event);
  }

  void resizeEvent(QResizeEvent *ev) {
    QWindow::resizeEvent(ev);
    if (!init) return; 
    auto app = QtWindowInterface::parent;
    app->MakeCurrentWindow(this);
    int w = ev->size().width(), h = ev->size().height();
    app->focused->Reshaped(point(w, h), Box(w, h));
    RequestRender();
  }

  void keyPressEvent  (QKeyEvent *ev) { keyEvent(ev, true); }
  void keyReleaseEvent(QKeyEvent *ev) { keyEvent(ev, false); }
  void keyEvent       (QKeyEvent *ev, bool down) {
    if (!init) return;
    ev->accept();
    auto app = QtWindowInterface::parent;
    int key = GetKeyCode(ev), fired = key ? app->input->KeyPress(key, 0, down) : 0;
    if (fired && frame_on_keyboard_input) RequestRender();
  }

  void mouseReleaseEvent(QMouseEvent *ev) { QWindow::mouseReleaseEvent(ev); mouseClickEvent(ev, false); }
  void mousePressEvent  (QMouseEvent *ev) { QWindow::mousePressEvent(ev);   mouseClickEvent(ev, true); }

  void mouseClickEvent(QMouseEvent *ev, bool down) {
    if (!init) return;
    auto app = QtWindowInterface::parent;
    int fired = app->input->MouseClick(GetMouseButton(ev), down, GetMousePosition(ev, app->focused->gl_h));
    if (fired && frame_on_mouse_input) RequestRender();
  }

  void mouseMoveEvent(QMouseEvent *ev) {
    QWindow::mouseMoveEvent(ev);
    if (!init) return;
    auto app = QtWindowInterface::parent;
    point p = GetMousePosition(ev, app->focused->gl_h);
    int fired = app->input->MouseMove(p, p - mouse_p);
    if (fired && frame_on_mouse_input) RequestRender();
    if (!grabbed) mouse_p = p;
    else {
      auto qw = static_cast<QWindow*>(this);
      mouse_p = point(qw->width()/2, qw->height()/2);
      QCursor::setPos(mapToGlobal(QPoint(mouse_p.x, mouse_p.y)));
    }
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

  void SetCaption(const string &v) override { window->setWindowTitle(MakeQString(v)); }
  void SetResizeIncrements(float x, float y) override { window->windowHandle()->setSizeIncrement(QSize(x, y)); }
  void SetTransparency(float v) override { window->windowHandle()->setOpacity(1-v); }
  bool Reshape(int w, int h) override { window->resize(w, h); QtWindowInterface::parent->MakeCurrentWindow(this); return true; }
  void Wakeup(int) { if (QtWindowInterface::parent->scheduler.wait_forever) RequestRender(); }
  int Swap() override {
    glc->swapBuffers(opengl_window);
    gd->CheckForError(__FILE__, __LINE__);
    return 0;
  }

  static unsigned GetKeyCode(QKeyEvent *ev) { int k = ev->key(); return k < 256 && isalpha(k) ? ::tolower(k) : k; }
  static point    GetMousePosition(QMouseEvent *ev, int h) { return point(ev->x(), h - ev->y()); }
  static unsigned GetMouseButton  (QMouseEvent *ev) {
    int b = ev->button();
    if      (b == Qt::LeftButton)  return 1;
    else if (b == Qt::RightButton) return 2;
    return 0;
  }
};

struct QtFramework : public Framework {
  unique_ptr<Window> ConstructWindow(Application *app) override { return make_unique<QtWindow>(app); }
  void StartWindow(Window *W) override {}

  bool CreateWindow(WindowHolder *H, Window *W) override {
    CHECK(!W->id && W->gd);
    W->id = W;
    W->started = true;

    QtWindow *my_qwin = dynamic_cast<QtWindow*>(W);
    QMainWindow *main_window = new QMainWindow();
    QSurfaceFormat format;
    format.setSamples(16);
    my_qwin->window = main_window;
    my_qwin->opengl_window = my_qwin;
    my_qwin->setFormat(format);
    my_qwin->setSurfaceType(QWindow::OpenGLSurface);
    my_qwin->opengl_container = QWidget::createWindowContainer(my_qwin, main_window);
    my_qwin->opengl_container->setFocusPolicy(Qt::TabFocus);
    my_qwin->layout = new QStackedLayout(main_window);
    my_qwin->window->setCentralWidget(new QWidget(main_window));
    my_qwin->window->centralWidget()->setLayout(my_qwin->layout);
    my_qwin->layout->addWidget(my_qwin->opengl_container);
    my_qwin->window->resize(W->gl_w, W->gl_h);
    my_qwin->window->show();
    my_qwin->opengl_container->setFocus();
    my_qwin->RequestRender();

    H->windows[W->id] = W;
    return true;
  }

  int Free() override {
    delete qapp;
    qapp = nullptr;
    return 0;
  };
};

#include "app/framework/qt_framework.moc"

string Clipboard::GetClipboardText() { return GetQString(QApplication::clipboard()->text()); }
void Clipboard::SetClipboardText(const string &s) { QApplication::clipboard()->setText(MakeQString(s)); }

void MouseFocus::GrabMouseFocus()    { auto w = dynamic_cast<QtWindow*>(window->focused); w->grabbed=1; w->window->windowHandle()->setCursor(Qt::BlankCursor); w->grab_mode.On();  w->cursor_grabbed=true;  }
void MouseFocus::ReleaseMouseFocus() { auto w = dynamic_cast<QtWindow*>(window->focused); w->grabbed=0; w->window->windowHandle()->unsetCursor();              w->grab_mode.Off(); w->cursor_grabbed=false; }

void TouchKeyboard::ToggleTouchKeyboard() {}
void TouchKeyboard::OpenTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboard() {}
void TouchKeyboard::CloseTouchKeyboardAfterReturn(bool v) {}
void TouchKeyboard::SetTouchKeyboardTiled(bool v) {}

void ThreadDispatcher::RunCallbackInMainThread(Callback cb) {
  message_queue.Write(new Callback(move(cb)));
  if (!FLAGS_target_fps) wakeup->Wakeup();
}

void WindowHolder::SetAppFrameEnabled(bool v) {}
void WindowHolder::MakeCurrentWindow(Window *W) {
  if (auto w = dynamic_cast<QtWindow*>((focused = W))) w->glc->makeCurrent(w->opengl_window);
}

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
void Application::SetTheme(const string &v) {}
void Application::LoseFocus() {}
int Application::Suspended() { return 0; }

void Application::CloseWindow(Window *W) {
  windows.erase(W->id);
  if (windows.empty()) run = false;
  if (window_closed_cb) window_closed_cb(W);
  focused = 0;
}

FrameScheduler::FrameScheduler(WindowHolder *w) :
  window(w), maxfps(&FLAGS_target_fps), rate_limit(0), wait_forever(!FLAGS_target_fps),
  wait_forever_thread(0), synchronize_waits(0), monolithic_frame(0), run_main_loop(0) {}

bool FrameScheduler::DoMainWait(bool only_poll) { return false; }
void FrameScheduler::UpdateWindowTargetFPS(Window *w) { /*QTTriggerFrame(w->id.value);*/ }
void FrameScheduler::AddMainWaitMouse(Window *w) { if (w) dynamic_cast<QtWindow*>(w)->frame_on_mouse_input = true; }
void FrameScheduler::DelMainWaitMouse(Window *w) { if (w) dynamic_cast<QtWindow*>(w)->frame_on_mouse_input = false; }
void FrameScheduler::AddMainWaitKeyboard(Window *w) { if (w) dynamic_cast<QtWindow*>(w)->frame_on_keyboard_input = true; }
void FrameScheduler::DelMainWaitKeyboard(Window *w) { if (w) dynamic_cast<QtWindow*>(w)->frame_on_keyboard_input = false; }

void FrameScheduler::AddMainWaitSocket(Window *w, Socket fd, int flag, function<bool()> f) {
  if (fd == InvalidSocket) return;
  else if (w) dynamic_cast<QtWindow*>(w)->AddMainWaitSocket(fd, move(f));
}

void FrameScheduler::DelMainWaitSocket(Window *w, Socket fd) {
  if (fd == InvalidSocket) return;
  else if (w) dynamic_cast<QtWindow*>(w)->DelMainWaitSocket(fd);
}

extern "C" int main(int argc, const char *argv[]) {
  qapp = new QApplication(argc, const_cast<char**>(argv));
  auto app = static_cast<Application*>(MyAppCreate(argc, argv));
  app->scheduler.wakeup_thread = make_unique<SocketWakeupThread>
    (app, app, &app->scheduler.frame_mutex, &app->scheduler.wait_mutex);
  auto fw = static_cast<QtFramework*>(app->framework.get());
  app->focused->gd = GraphicsDevice::Create(app->focused, app->shaders.get(), 2).release();
  fw->CreateWindow(app, app->focused);
  auto w = dynamic_cast<QtWindow*>(app->focused);
  if ((w->init = true)) { w->MyWindowInit(); if (app->splash_color) app->DrawSplash(*app->splash_color); }
  return qapp->exec();
}

unique_ptr<Framework> Framework::Create(Application *a) { return make_unique<QtFramework>(); }

}; // namespace LFL
