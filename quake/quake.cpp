/*
 * $Id: quake.cpp 1336 2014-12-08 09:29:59Z justin $
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
#include "q3map.h"

namespace LFL {
BindMap binds;
AssetMap asset;
SoundAssetMap soundasset;

Scene scene;
MapAsset *quake_map;

// LFL::Application FrameCB
int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    screen->camMain->look();
    quake_map->Draw(*screen->camMain);
    // scene.get("arrow")->yawright((double)clicks/500);
    // scene.draw();

    // Press tick for console
    screen->gd->DrawMode(DrawMode::_2D);
    screen->DrawDialogs();
    return 0;
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

	app->logfilename = StrCat(dldir(), "quake.txt");
	app->frame_cb = Frame;
	screen->width = 640;
	screen->height = 480;
	screen->caption = "Quake";
	FLAGS_far_plane = 10000;
	FLAGS_ksens = 150;

	if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
	if (app->Init()) { app->Free(); return -1; }

	//  asset.Add(Asset(name, texture,  scale, translate, rotate, geometry, 0, 0));
	asset.Add(Asset("arrow", "", .005, 1, -90, "arrow.obj", 0, 0));
	asset.Load();
	app->shell.assets = &asset;

	//  soundasset.Add(SoundAsset(name,   filename,   ringbuf, channels, sample_rate, seconds ));
	soundasset.Add(SoundAsset("draw", "Draw.wav", 0, 0, 0, 0));
	soundasset.Load();
	app->shell.soundassets = &soundasset;

	//  binds.push_back(Bind(key,        callback));
	binds.push_back(Bind(Key::Return,    Bind::CB(bind(&Shell::grabmode, &app->shell, vector<string>()))));
	binds.push_back(Bind(Key::Escape,    Bind::CB(bind(&Shell::quit, &app->shell, vector<string>()))));
	binds.push_back(Bind(Key::Backquote, Bind::CB(bind([&]() { screen->console->Toggle(); }))));
    binds.push_back(Bind(Key::Quote,     Bind::CB(bind([&]() { screen->console->Toggle(); }))));
    binds.push_back(Bind(Key::LeftShift, Bind::TimeCB(bind(&Entity::RollLeft,   screen->camMain, _1))));
    binds.push_back(Bind(Key::Space,     Bind::TimeCB(bind(&Entity::RollRight,  screen->camMain, _1))));
    binds.push_back(Bind('w',            Bind::TimeCB(bind(&Entity::MoveFwd,    screen->camMain, _1))));
    binds.push_back(Bind('s',            Bind::TimeCB(bind(&Entity::MoveRev,    screen->camMain, _1))));
    binds.push_back(Bind('a',            Bind::TimeCB(bind(&Entity::MoveLeft,   screen->camMain, _1))));
    binds.push_back(Bind('d',            Bind::TimeCB(bind(&Entity::MoveRight,  screen->camMain, _1))));
    binds.push_back(Bind('q',            Bind::TimeCB(bind(&Entity::MoveDown,   screen->camMain, _1))));
    binds.push_back(Bind('e',            Bind::TimeCB(bind(&Entity::MoveUp,     screen->camMain, _1))));
    screen->binds = &binds;

    scene.Add(new Entity("axis",  asset("axis")));
    scene.Add(new Entity("grid",  asset("grid")));
    scene.Add(new Entity("room",  asset("room")));
    scene.Add(new Entity("arrow", asset("arrow"), v3(1, .24, 1)));

    quake_map = Q3MapAsset::Load(StrCat(ASSETS_DIR, "map-20kdm2.pk3"));
    screen->camMain->pos = v3(1910.18,443.64,410.21);
    screen->camMain->ort = v3(-0.05,0.70,0.03);
    screen->camMain->up = v3(0.00,-0.04,0.98);

    // start our engine
    return app->Main();
}
