/*
 * $Id: new_app_template.cpp 1305 2014-09-02 08:10:25Z justin $
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
#include "lfapp/gui.h"

vector<Bind> binds;
vector<Asset> assets;
vector<SoundAsset> soundassets;
Scene scene;

// engine callback
// driven by lfapp_frame()
int frame(unsigned clicks, unsigned mic_samples, bool cam_sample, unsigned events, int flag) {
    scene.get("arrow")->yawright((double)clicks/500);
    scene.draw(&assets);

    // Press tick for console
    drawmode(DrawMode::_2D);
    gui_draw();
    return 0;
}

extern "C" int main(int argc, const char *argv[]) {

    logfilename = StrCat(dldir(), "$BINNAME.txt");
    lfapp_frame_cb = frame;
    screen->width = 420;
    screen->height = 380;
    screen->caption = "$PKGNAME";

    if (lfapp_init(argc, argv)) { lfapp_close(); return -1; }
    if (lfapp_open()) { lfapp_close(); return -1; }

    // assets.push_back(Asset(name,    callback, texture,     scale, translate, rotate, geometry              0, 0, 0, 0));
    assets.push_back(Asset("axis",     glAxis,   "",          0,     0,         0,      0,                    0, 0, 0, 0));
    assets.push_back(Asset("grid",     0,        "",          0,     0,         0,      glGrid(),             0, 0, 0, 0));
    assets.push_back(Asset("room",     glRoom,   "",          0,     0,         0,      0,                    0, 0, 0, 0));
    assets.push_back(Asset("arrow",    0,        "",         .005,   1,        -90,     OBJFILE("arrow.obj"), 0, 0, 0, 0));
    load(&assets);

    // soundassets.push_back(SoundAsset(name, filename,   ringbuf, channels, sample_rate, seconds ));
    soundassets.push_back(SoundAsset("draw",  "Draw.wav", 0,       0,        0,           0       ));
    load(&soundassets);

    // binds.push_back(Bind(key,         callback,         arg,    iscmd));
    binds.push_back(Bind(Key::Backquote, Shell::console,   0,      1    ));
    binds.push_back(Bind(Key::Quote,     Shell::console,   0,      1    ));
    binds.push_back(Bind(Key::Escape,    Shell::quit,      0,      1    ));
    binds.push_back(Bind(Key::Return,    Shell::grabmode,  0,      1    ));
    binds.push_back(Bind(Key::LeftShift, Shell::rollleft,  0,      0    ));
    binds.push_back(Bind(Key::Space,     Shell::rollright, 0,      0    ));
    binds.push_back(Bind('w',            Shell::movefwd,   0,      0    ));
    binds.push_back(Bind('s',            Shell::moverev,   0,      0    ));
    binds.push_back(Bind('a',            Shell::moveleft,  0,      0    ));
    binds.push_back(Bind('d',            Shell::moveright, 0,      0    ));
    binds.push_back(Bind('q',            Shell::movedown,  0,      0    ));
    binds.push_back(Bind('e',            Shell::moveup,    0,      0    ));
    screen->binds = &binds;

    scene.add(new Entity("axis",  asset("axis")));
    scene.add(new Entity("grid",  asset("grid")));
    scene.add(new Entity("room",  asset("room")));
    scene.add(new Entity("arrow", asset("arrow"), v3(1, .24, 1)));

    // start our engine
    return lfapp_main();
}
