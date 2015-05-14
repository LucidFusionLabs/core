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
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"

namespace LFL {
AssetMap asset;
SoundAssetMap soundasset;
Scene scene;

// engine callback driven by LFL::Application
int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    screen->cam->Look();
    scene.Get("arrow")->YawRight((double)clicks);
    scene.Draw(&asset.vec);

    screen->gd->DrawMode(DrawMode::_2D);
    screen->DrawDialogs();
    Fonts::Default()->Draw(StringPrintf("Hello warld, my FPS is %.2f", FPS()), point(screen->width*.05, screen->height*.15));
    Fonts::Default()->Draw("press tick for console",                           point(screen->width*.05, screen->height*.05));
    return 0;
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    app->logfilename = StrCat(LFAppDownloadDir(), "$BINNAME.txt");
    screen->caption = "$PKGNAME";
    screen->frame_cb = Frame;
    screen->width = 420;
    screen->height = 380;
    FLAGS_target_fps = 30;
    FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = 1;

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init()) { app->Free(); return -1; }
    screen->gd->default_draw_mode = DrawMode::_3D;

    // asset.Add(Asset(name, texture,  scale, translate, rotate, geometry              0, 0, 0, callback));
    asset.Add(Asset("axis",  "",       0,     0,         0,      0,                    0, 0, 0, glAxis  ));
    asset.Add(Asset("grid",  "",       0,     0,         0,      Grid::Grid3D(),       0, 0, 0          ));
    asset.Add(Asset("room",  "",       0,     0,         0,      0,                    0, 0, 0, glRoom  ));
    asset.Add(Asset("arrow", "",      .005,   1,        -90,     "arrow.obj",          0, 0             ));
    asset.Load();
    app->shell.assets = &asset;

    // soundasset.Add(SoundAsset(name, filename,   ringbuf, channels, sample_rate, seconds ));
    app->shell.soundassets = &soundasset;

    BindMap *binds = screen->binds = new BindMap();
    // binds.push_back(Bind(key,         callback));
    binds->Add(Bind(Key::Backquote, Bind::CB(bind([&]() { screen->console->Toggle(); }))));
    binds->Add(Bind(Key::Quote,     Bind::CB(bind([&]() { screen->console->Toggle(); }))));
    binds->Add(Bind(Key::Escape,    Bind::CB(bind(&Shell::quit, &app->shell, vector<string>()))));
    binds->Add(Bind(Key::Return,    Bind::CB(bind(&Shell::grabmode, &app->shell, vector<string>()))));
    binds->Add(Bind(Key::LeftShift, Bind::TimeCB(bind(&Entity::RollLeft,   screen->cam, _1))));
    binds->Add(Bind(Key::Space,     Bind::TimeCB(bind(&Entity::RollRight,  screen->cam, _1))));
    binds->Add(Bind('w',            Bind::TimeCB(bind(&Entity::MoveFwd,    screen->cam, _1))));
    binds->Add(Bind('s',            Bind::TimeCB(bind(&Entity::MoveRev,    screen->cam, _1))));
    binds->Add(Bind('a',            Bind::TimeCB(bind(&Entity::MoveLeft,   screen->cam, _1))));
    binds->Add(Bind('d',            Bind::TimeCB(bind(&Entity::MoveRight,  screen->cam, _1))));
    binds->Add(Bind('q',            Bind::TimeCB(bind(&Entity::MoveDown,   screen->cam, _1))));
    binds->Add(Bind('e',            Bind::TimeCB(bind(&Entity::MoveUp,     screen->cam, _1))));

    scene.Add(new Entity("axis",  asset("axis")));
    scene.Add(new Entity("grid",  asset("grid")));
    scene.Add(new Entity("room",  asset("room")));
    scene.Add(new Entity("arrow", asset("arrow"), v3(1, .24, 1)));

    // start our engine
    return app->Main();
}
