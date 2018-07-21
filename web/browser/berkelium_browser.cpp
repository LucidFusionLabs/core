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

#include "berkelium/Berkelium.hpp"
#include "berkelium/Window.hpp"
#include "berkelium/WindowDelegate.hpp"
#include "berkelium/Context.hpp"

namespace LFL {
  
struct BerkeliumModule : public Module {
  ApplicationInfo *appinfo;
  BerkeliumModule(ApplicationInfo *a) : appinfo(a) {}
  int Frame(unsigned t) { Berkelium::update(); return 0; }
  int Free() { Berkelium::destroy(); return 0; }
  int Init() {
    const char *homedir = appinfo->savedir;
    INFO("berkelium init");
    Berkelium::init(
#ifdef _WIN32
                    homedir ? Berkelium::FileString::point_to(wstring(homedir).c_str(), wstring(homedir).size()) :
#else
                    homedir ? Berkelium::FileString::point_to(homedir, strlen(homedir)) :
#endif
                    Berkelium::FileString::empty());
    return 0;
  }
};

struct BerkeliumBrowser : public BrowserInterface, public Berkelium::WindowDelegate {
  int W, H;
  Texture *tex;
  Berkelium::Window *window;

  BerkeliumBrowser(Texture *t, int win, int hin) : tex(t), window(0) { 
    Singleton<BerkeliumModule>::Get();
    Berkelium::Context* context = Berkelium::Context::create();
    window = Berkelium::Window::create(context);
    delete context;
    window->setDelegate(this);
    resize(win, hin); 
  }

  void Resize(int win, int hin) {
    window->resize((W = win), (H = hin));
    if (!tex->ID) tex->CreateBacked(W, H);
    else          tex->Resize(W, H);
  }

  void Draw(const Box &w) { tex->DrawCrimped(w, 1, 0, 0); }
  void Open(const string &url) { window->navigateTo(Berkelium::URLString::point_to(url.data(), url.length())); }
  void MouseMoved(int x, int y) { window->mouseMoved((float)x/screen->width*W, (float)(screen->height-y)/screen->height*H); }
  void MouseButton(int b, bool down, int x, int y) {
    if (b == 1 && down) {
      int click_x = screen->mouse.x/screen->width*W, click_y = (screen->height - screen->mouse.y)/screen->height*H;
      basic_string<wchar_t> js = WStringPrintf(L"lfapp_browser_click(document.elementFromPoint(%d, %d).innerHTML)", click_x, click_y);
      window->executeJavascript(Berkelium::WideString::point_to(js.data(), js.size()));
    }
    window->mouseButton(b == 1 ? 0 : 1, down);
  }

  void MouseWheel(int xs, int ys) { window->mouseWheel(xs, ys); }
  void KeyEvent(int key, bool down) {
    int bk = -1;
    switch (key) {
      case Key::PageUp:    bk = 0x21; break;
      case Key::PageDown:  bk = 0x22; break;
      case Key::Left:      bk = 0x25; break;
      case Key::Up:        bk = 0x26; break;
      case Key::Right:     bk = 0x27; break;
      case Key::Down:      bk = 0x28; break;
      case Key::Delete:    bk = 0x2E; break;
      case Key::Backspace: bk = 0x2E; break;
      case Key::Return:    bk = '\r'; break;
      case Key::Tab:       bk = '\t'; break;
    }

    int mods = ctrl_key_down() ? Berkelium::CONTROL_MOD : 0 | shift_key_down() ? Berkelium::SHIFT_MOD : 0;
    window->keyEvent(down, mods, bk != -1 ? bk : key, 0);
    if (!down || bk != -1) return;

    wchar_t wkey[2] = { key, 0 };
    window->textEvent(wkey, 1);
  }

  virtual void onPaint(Berkelium::Window* wini, const unsigned char *in_buf, const Berkelium::Rect &in_rect,
                       size_t num_copy_rects, const Berkelium::Rect* copy_rects, int dx, int dy, const Berkelium::Rect& scroll_rect) {
    unsigned char *buf = tex->buf;
    int bpp = Pixel::size(tex->pf);
    screen->gd->BindTexture(GL_TEXTURE_2D, tex->ID);

    if (dy < 0) {
      const Berkelium::Rect &r = scroll_rect;
      for (int j = -dy, h = r.height(); j < h; j++) {
        int src_ind = ((j + r.top()     ) * tex->w + r.left()) * bpp;
        int dst_ind = ((j + r.top() + dy) * tex->w + r.left()) * bpp;
        memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
      }
    } else if (dy > 0) {
      const Berkelium::Rect &r = scroll_rect;
      for (int j = r.height() - dy; j >= 0; j--) {
        int src_ind = ((j + r.top()     ) * tex->w + r.left()) * bpp;
        int dst_ind = ((j + r.top() + dy) * tex->w + r.left()) * bpp;                
        memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
      }
    }
    if (dx) {
      const Berkelium::Rect &r = scroll_rect;
      for (int j = 0, h = r.height(); j < h; j++) {
        int src_ind = ((j + r.top()) * tex->w + r.left()     ) * bpp;
        int dst_ind = ((j + r.top()) * tex->w + r.left() - dx) * bpp;
        memcpy(buf+dst_ind, buf+src_ind, r.width()*bpp);
      }
    }
    for (int i = 0; i < num_copy_rects; i++) {
      const Berkelium::Rect &r = copy_rects[i];
      for(int j = 0, h = r.height(); j < h; j++) {
        int dst_ind = ((j + r.top()) * tex->w + r.left()) * bpp;
        int src_ind = ((j + (r.top() - in_rect.top())) * in_rect.width() + (r.left() - in_rect.left())) * bpp;
        memcpy(buf+dst_ind, in_buf+src_ind, r.width()*bpp);
      }
      tex->flush(0, r.top(), tex->w, r.height());        
    }

    // for (Berkelium::Window::FrontToBackIter iter = wini->frontIter(); 0 && iter != wini->frontEnd(); iter++) { Berkelium::Widget *w = *iter; }
  }

  virtual void onCreatedWindow(Berkelium::Window *win, Berkelium::Window *newWindow, const Berkelium::Rect &initialRect) { newWindow->setDelegate(this); }
  virtual void onAddressBarChanged(Berkelium::Window *win, Berkelium::URLString newURL) { INFO("onAddressBarChanged: ", newURL.data()); }
  virtual void onStartLoading(Berkelium::Window *win, Berkelium::URLString newURL) { INFO("onStartLoading: ", newURL.data()); }
  virtual void onLoad(Berkelium::Window *win) { INFO("onLoad"); }

  virtual void onJavascriptCallback(Berkelium::Window *win, void* replyMsg, Berkelium::URLString url, Berkelium::WideString funcName, Berkelium::Script::Variant *args, size_t numArgs) {
    string fname = WideStringToString(funcName);
    if (fname == "lfapp_browser_click" && numArgs == 1 && args[0].type() == args[0].JSSTRING) {
      string a1 = WideStringToString(args[0].toString());
      INFO("jscb: ", fname, " ", a1);
    } else {
      INFO("jscb: ", fname);
    }
    if (replyMsg) win->synchronousScriptReturn(replyMsg, numArgs ? args[0] : Berkelium::Script::Variant());
  }

  virtual void onRunFileChooser(Berkelium::Window *win, int mode, Berkelium::WideString title, Berkelium::FileString defaultFile) { win->filesSelected(NULL); }
  virtual void onNavigationRequested(Berkelium::Window *win, Berkelium::URLString newURL, Berkelium::URLString referrer, bool isNewWindow, bool &cancelDefaultAction) {}

  static string WideStringToString(const Berkelium::WideString& in) {
    string out; out.resize(in.size());
    for (int i = 0; i < in.size(); i++) out[i] = in.data()[i];     
    return out;
  }
};

unique_ptr<BrowserInterface> CreateBerkeliumBrowser(GUI *g, int W, int H) {
  unique_ptr<BerkeliumBrowser> browser = make_unique<BerkeliumBrowser>(a, W, H);
  Berkelium::WideString click = Berkelium::WideString::point_to(L"lfapp_browser_click");
  browser->window->bind(click, Berkelium::Script::Variant::bindFunction(click, true));
  return browser;
}

}; // namespace LFL
