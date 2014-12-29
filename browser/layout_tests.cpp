/*
 * $Id: browser.cpp 1223 2014-06-11 03:35:22Z justin $
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

DEFINE_string(render_url,   "",           "Url to render");
DEFINE_string(render_png,   "render.png", "Filename to render to");
DEFINE_string(layout_tests, "",           "e.g. WebKit/LayoutTests/css2.1/");

SimpleBrowser *browser;
SimpleBrowser::RenderLog render_log;

struct LayoutTests {
    int count;
    DirectoryIter files;
    struct LayoutTest { string html_fn, png_fn, log_fn, log_expected; } current;
    LayoutTests(const string &path) : count(0), files(path.c_str(), 0, 0, ".html") {}
    string CurrentName() {
        return StringPrintf("LayoutTest-%03d %s render_log(%s) png(%s)", 
                            count, current.html_fn.c_str(), current.log_fn.c_str(), current.png_fn.c_str());
    }
    bool Next() {
        for (;;) {
            current.html_fn = BlankNull(files.Next());
            if (current.html_fn.empty()) break;
            current.html_fn = StrCat(files.pathname, current.html_fn);
            current.log_fn  = current.html_fn.substr(0, current.html_fn.size()-5) + "-expected.txt";
            current.png_fn  = current.html_fn.substr(0, current.html_fn.size()-5) + "-expected.png";
            if (!LocalFile(current.html_fn, "r").Opened()) current.html_fn.clear();
            if (!LocalFile(current.png_fn,  "r").Opened()) current.png_fn .clear();
            if (!current.log_fn.empty()) current.log_expected = LocalFile::FileContents(current.log_fn);
            else if (current.png_fn.empty()) continue;
            count++;
            return true;
        } return false;
    }
} *layout_tests;

int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    screen->gd->DrawMode(DrawMode::_2D);
    Box rootwin = screen->Box();
    bool load_done = browser->requested == browser->completed;
    if (!load_done) return -1;
    else {
        if (browser->requested) {
            browser->Draw(&rootwin);
            app->shell.screenshot(vector<string>(1, FLAGS_render_png));
            if (layout_tests) {
                INFO(layout_tests->CurrentName());
                if (!layout_tests->current.log_expected.empty())
                printf("Expected:\n%s\n", layout_tests->current.log_expected.c_str());
            }
            printf("Got:\n%s\n", render_log.data.c_str());
        }
        if (!layout_tests || !layout_tests->Next()) { app->run=0; return 0; }
        browser->Open(StrCat("file:/", layout_tests->current.html_fn));
    }
    return 0;
}

extern "C" int main(int argc, const char *argv[]) {
    
    app->frame_cb = Frame;
    app->logfilename = StrCat(dldir(), "layout_tests.txt");
    screen->caption = "layout_tests";
    screen->width = 800;
    screen->height = 600;
    FLAGS_font_engine = "freetype";
    FLAGS_default_font = "DejaVuSans.ttf";
    FLAGS_default_font_family = "sans-serif";
    FLAGS_atlas_font_sizes = "32";

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init()) { app->Free(); return -1; }

    screen->gd->ClearColor(Color::white);
    browser = new SimpleBrowser(screen, 0, screen->Box());
    browser->InitLayers();
    browser->render_log = &render_log;
    if (!FLAGS_layout_tests.empty()) layout_tests = new LayoutTests(FLAGS_layout_tests);
    else if (!FLAGS_render_url.empty()) browser->Open(FLAGS_render_url);

    return app->Main();
}
