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

#include "core/app/gl/view.h"
#include "core/app/ipc.h"
#include "core/web/browser.h"
#include "core/web/document.h"

#include <QWindow>
#include <QWebView>
#include <QWebFrame>
#include <QWebElement>
#include <QMouseEvent>
#include <QApplication>

namespace LFL {
extern QApplication *lfl_qapp;

class QTWebKitBrowser : public QObject, public BrowserInterface {
  Q_OBJECT
  public:
  QWebPage page;
  Texture tex;
  point dim, mouse;
  bool mouse1_down=0;

  QTWebKitBrowser(GraphicsDeviceHolder *w, int w, int h) : tex(w) {
    Resize(w, h);
    lfl_qapp->connect(&page, SIGNAL(repaintRequested(const QRect&)),           this, SLOT(ViewUpdate(const QRect &)));
    lfl_qapp->connect(&page, SIGNAL(scrollRequested(int, int, const QRect &)), this, SLOT(Scroll(int, int, const QRect &)));
  }
  void Resize(int win, int hin) {
    page.setViewportSize(QSize((dim.x = win), (dim.y = hin)));
    if (!tex.ID) tex.CreateBacked(dim.x, dim.y);
    else         tex.Resize(dim.x, dim.y);
  }
  void Open(const string &url) { page.mainFrame()->load(QUrl(url.c_str())); }
  void MouseMoved(int x, int y) {
    mouse = point(x, screen->height - y);
    QMouseEvent me(QEvent::MouseMove, QPoint(mouse.x, mouse.y), Qt::NoButton, mouse1_down ? Qt::LeftButton : Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(&page, &me);
  }
  void MouseButton(int b, bool d, int x, int y) {
    if (b != 1) return;
    mouse1_down = d;
    QMouseEvent me(QEvent::MouseButtonPress, QPoint(mouse.x, mouse.y), Qt::LeftButton, mouse1_down ? Qt::LeftButton : Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(&page, &me);
  }
  void MouseWheel(int xs, int ys) {}
  void KeyEvent(int key, bool down) {}
  void BackButton() { page.triggerAction(QWebPage::Back); }
  void ForwardButton() { page.triggerAction(QWebPage::Forward);  }
  void RefreshButton() { page.triggerAction(QWebPage::Reload); }
  string GetURL() const { return page.mainFrame()->url().toString().toLocal8Bit().data(); }
  void Draw(const Box &w) {
    tex.DrawCrimped(w, 1, 0, 0);
    QPoint scroll = page.currentFrame()->scrollPosition();
    QWebHitTestResult hit_test = page.currentFrame()->hitTestContent(QPoint(mouse.x, mouse.y));
    QWebElement hit = hit_test.element();
    QRect rect = hit.geometry();
    Box hitwin(rect.x() - scroll.x(), w.h-rect.y()-rect.height()+scroll.y(), rect.width(), rect.height());
    screen->gd->SetColor(Color::blue);
    hitwin.Draw();
  }

  public slots:
  void Scroll(int dx, int dy, const QRect &rect) { ViewUpdate(QRect(0, 0, dim.x, dim.y)); }
  void ViewUpdate(const QRect &dirtyRect) {
    QImage image(page.viewportSize(), QImage::Format_ARGB32);
    QPainter painter(&image);
    page.currentFrame()->render(&painter, dirtyRect);
    painter.end();

    auto gd = tex.parent->gd();
    int instride = image.bytesPerLine(), bpp = Pixel::Size(tex.pf);
    int mx = dirtyRect.x(), my = dirtyRect.y(), mw = dirtyRect.width(), mh = dirtyRect.height();
    const unsigned char *in = (const unsigned char *)image.bits();
    gd->BindTexture(gd->c.Texture2D, tex.ID);

    unsigned char *buf = tex.buf;
    for (int j = 0; j < mh; j++) {
      int src_ind = (j + my) * instride + mx * bpp;
      int dst_ind = ((j + my) * tex.width + mx) * bpp;
      if (false) VideoResampler::RGB2BGRCopyPixels(buf+dst_ind, in+src_ind, mw, bpp);
      else                                  memcpy(buf+dst_ind, in+src_ind, mw*bpp);
    }
    tex.UpdateGL(Box(0, my, tex.width, dirtyRect.height()));
    if (!gd->parent->target_fps) gd->parent->Wakeup();
  }
};

#include "browser.moc"

unique_ptr<BrowserInterface> CreateQTWebKitBrowser(View*, int w, int h) { return make_unique<QTWebKitBrowser>(w, h); }

}; // namespace LFL
