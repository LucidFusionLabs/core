/*
 * $Id: browser.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"

using namespace LFL;

DEFINE_string(url, "http://news.google.com/", "Url to open");
DEFINE_int(width, 1040, "browser width");
DEFINE_int(height, 768, "browser height");

BindMap binds;

struct JavaScriptConsole : public Console {
    SimpleBrowser *simple_browser;
    JavaScriptConsole(LFL::Window *W, Font *f, SimpleBrowser *B) : Console(W, f), simple_browser(B)
    { bottom_or_top = 1; write_timestamp = blend = 0; }
    virtual void Run(string in) {
        string ret = simple_browser->js_context->Execute(in);
        if (!ret.empty()) Write(ret);
    }
};

struct MyBrowserWindow : public GUI {
    Font *menu_atlas1, *menu_atlas2;
    Box win, topbar, addressbar;
    Widget::Button back, forward, refresh;
    TextGUI address_box;
    SimpleBrowser *simple_browser;
    Browser *webkit_browser, *berkelium_browser;

    MyBrowserWindow(LFL::Window *W) : GUI(W),
    menu_atlas1(Fonts::Get("MenuAtlas1", 0, Color::black)),
    menu_atlas2(Fonts::Get("MenuAtlas2", 0, Color::black)),
    back   (this, &menu_atlas1->FindGlyph(20)->tex, 0, "", MouseController::CB([&](){ screen->browser_window->BackButton(); })),
    forward(this, &menu_atlas1->FindGlyph(22)->tex, 0, "", MouseController::CB([&](){ screen->browser_window->ForwardButton(); })),
    refresh(this, &menu_atlas2->FindGlyph(50)->tex, 0, "", MouseController::CB([&](){ screen->browser_window->RefreshButton(); })),
    address_box(W, Fonts::Get(FLAGS_default_font, 12, Color::black), 0, ToggleBool::OneShot),
    simple_browser(0), webkit_browser(0), berkelium_browser(0) {
        address_box.cmd_prefix.clear();
        address_box.deactivate_on_enter = true;
        address_box.runcb = [&](const string &t){ screen->browser_window->Open(t); };
        refresh.AddClickBox(addressbar, MouseController::CB([&](){ address_box.active = true; }));
    }

    void Layout() {
        win = topbar = screen->Box();
        topbar.h = 16;
        topbar.y = max(0, win.y + win.h - topbar.h);
        win.h    = max(0,         win.h - topbar.h);

        addressbar = topbar;
        Typed::MinusPlus(&addressbar.w, &addressbar.x, 16*3 + 20);

        Flow flow(&box, 0, Reset());
        back.Layout(&flow, point(16, 16));
        flow.p.x += 5;
        forward.Layout(&flow, point(16, 16));
        flow.p.x += 5;
        refresh.Layout(&flow, point(16, 16));
    }

    void Open() {
        Layout();
#if LFL_QT
        if (!screen->browser_window) screen->browser_window = webkit_browser = CreateQTWebKitBrowser(new Asset());
#endif
#ifdef LFL_EAWEBKIT
        if (!screen->browser_window) screen->browser_window = webkit_browser = CreateEAWebKitBrowser(new Asset());
#endif
#ifdef LFL_BERKELIUM
        if (!screen->browser_window) screen->browser_window = berkelium_browser = CreateBerkeliumBrowser(new Asset());
#endif
        if (!screen->browser_window) {
            screen->browser_window = simple_browser = new SimpleBrowser(screen, 0, win);
            simple_browser->js_console = new JavaScriptConsole(screen, Fonts::Default(), simple_browser);
            simple_browser->InitLayers();
        }
    }
    bool Dirty() { return simple_browser ? simple_browser->Dirty(&win) : true; }
    void Draw() {
        box = screen->Box();
        GUI::Draw();
        screen->gd->SetColor(Color::white);
        if (!address_box.active) {
            string url = screen->browser_window->GetURL();
            if (url != address_box.Text()) address_box.cmd_line.AssignText(url);
        }
        address_box.Draw(addressbar);
        screen->browser_window->Draw(&win);
        if (simple_browser) {
            simple_browser->DrawScrollbar();
            if (simple_browser->js_console) simple_browser->js_console->Draw();
        }
    }
};

int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    MyBrowserWindow *bw = (MyBrowserWindow*)W->user1;
    bool dont_skip = flag & FrameFlag::DontSkip;
    if (!bw->Dirty() && !W->events.bind && !W->events.input && !dont_skip) return -1;
    screen->gd->DrawMode(DrawMode::_2D);
    bw->Draw();
    screen->DrawDialogs();
    return 0;
}

void MyJavaScriptConsole() {
    MyBrowserWindow *tw = (MyBrowserWindow*)screen->user1;
    if (tw->simple_browser && tw->simple_browser->js_console) tw->simple_browser->js_console->Toggle();
}

void MyWindowDefaults(LFL::Window *W) {
    W->width = FLAGS_width;
    W->height = FLAGS_height;
    W->caption = "Browser";
    W->binds = &binds;
    if (app->opened) W->user1 = new MyBrowserWindow(W);
}

extern "C" int main(int argc, const char *argv[]) {
    
    app->frame_cb = Frame;
    app->logfilename = StrCat(dldir(), "browser.txt");
    MyWindowDefaults(screen);
    FLAGS_font_engine = "freetype";
    FLAGS_default_font = "DejaVuSans.ttf";
    FLAGS_default_font_family = "sans-serif";
    FLAGS_atlas_font_sizes = "32";

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    screen->width = FLAGS_width; screen->height = FLAGS_height;
    if (app->Init()) { app->Free(); return -1; }

    vector<string> atlas_font_size;
    Split(FLAGS_atlas_font_sizes, iscomma, &atlas_font_size);
    for (int i=0; i<atlas_font_size.size(); i++) {
        int size = ::atoi(atlas_font_size[i].c_str());
        Fonts::InsertFreetype("DejaVuSans-Bold.ttf",           "sans-serif", size, Color::white, FontDesc::Bold);
        Fonts::InsertFreetype("DejaVuSans-Oblique.ttf",        "sans-serif", size, Color::white, FontDesc::Italic);
        Fonts::InsertFreetype("DejaVuSans-BoldOblique.ttf",    "sans-serif", size, Color::white, FontDesc::Italic | FontDesc::Bold);
        Fonts::InsertFreetype("DejaVuSansMono.ttf",            "monospace",  size, Color::white, 0);
        Fonts::InsertFreetype("DejaVuSansMono-Bold.ttf",       "monospace",  size, Color::white, FontDesc::Bold);
        Fonts::InsertFreetype("DejaVuSerif.ttf",               "serif",      size, Color::white, 0);
        Fonts::InsertFreetype("DejaVuSerif-Bold.ttf",          "serif",      size, Color::white, FontDesc::Bold);
        Fonts::InsertFreetype("DejaVuSansMono-Oblique.ttf",    "cursive",    size, Color::white, 0);
        Fonts::InsertFreetype("DejaVuSerifCondensed-Bold.ttf", "fantasy",    size, Color::white, 0);
    }

    binds.push_back(Bind('6', Key::Modifier::Cmd, Bind::CB(bind([&](){ screen->console->Toggle(); }))));
    binds.push_back(Bind('7', Key::Modifier::Cmd, Bind::CB(bind(&MyJavaScriptConsole))));

    screen->user1 = new MyBrowserWindow(screen);
    MyBrowserWindow *bw = (MyBrowserWindow*)screen->user1;
    bw->Open();
    if (screen->browser_window && !FLAGS_url.empty()) screen->browser_window->Open(FLAGS_url);

    return app->Main();
}
