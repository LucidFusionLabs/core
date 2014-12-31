/*
 * $Id: calculator.cpp 1334 2014-11-28 09:14:21Z justin $
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

#ifdef LFL_CLING
#define __STDC_LIMIT_MACROS
#include "cling/Interpreter/Interpreter.h"
#endif

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"
#include "ml/lp.h"

namespace LFL {
DEFINE_bool(visualize, false, "Display");
DEFINE_string(linear_program, "", "Linear program input");

Scene scene;
BindMap binds;
AssetMap asset;

int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
#ifdef LFL_CLING
    char buf[8192]={0}, result[512]={0}, *space;
    if (!FGets(buf, sizeof(buf))) return false;
    // cling::Interpreter::getSelf()->process(buf);
#else
    app->shell.FGets();
#endif
    if (!FLAGS_visualize) return 0;
    scene.Draw(&asset.vec);
    screen->gd->DrawMode(DrawMode::_2D);
    screen->DrawDialogs();
    return 0;
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    app->logfilename = StrCat(dldir(), "calculator.txt");
    app->frame_cb = Frame;
    screen->width = 420;
    screen->height = 380;
    screen->caption = "Calculator";

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }

    FLAGS_lfapp_audio = false;
    FLAGS_lfapp_video = FLAGS_visualize;

    if (app->Init()) { app->Free(); return -1; }

    app->shell.assets = &asset;
    // asset.Add(Asset(name, texture,     scale, translate, rotate, geometry        0, 0, 0));
    asset.Add(Asset("axis",  "",          0,     0,         0,      0,              0, 0, 0, Asset::DrawCB(bind(&glAxis, _1, _2))));
    asset.Add(Asset("grid",  "",          0,     0,         0,      Grid::Grid3D(), 0, 0, 0));
    asset.Load();

    // binds.push_back(Bind(key,         callback));
    binds.push_back(Bind(Key::Backquote, Bind::CB    (bind(&Shell::console,    app->shell, vector<string>()))));
    binds.push_back(Bind(Key::Quote,     Bind::CB    (bind(&Shell::console,    app->shell, vector<string>()))));
    binds.push_back(Bind(Key::Escape,    Bind::CB    (bind(&Shell::quit,       app->shell, vector<string>()))));
    binds.push_back(Bind(Key::Return,    Bind::CB    (bind(&Shell::grabmode,   app->shell, vector<string>()))));
    binds.push_back(Bind(Key::LeftShift, Bind::TimeCB(bind(&Entity::RollLeft,  screen->cam, _1))));
    binds.push_back(Bind(Key::Space,     Bind::TimeCB(bind(&Entity::RollRight, screen->cam, _1))));
    binds.push_back(Bind('w',            Bind::TimeCB(bind(&Entity::MoveFwd,   screen->cam, _1))));
    binds.push_back(Bind('s',            Bind::TimeCB(bind(&Entity::MoveRev,   screen->cam, _1))));
    binds.push_back(Bind('a',            Bind::TimeCB(bind(&Entity::MoveLeft,  screen->cam, _1))));
    binds.push_back(Bind('d',            Bind::TimeCB(bind(&Entity::MoveRight, screen->cam, _1))));
    binds.push_back(Bind('q',            Bind::TimeCB(bind(&Entity::MoveDown,  screen->cam, _1))));
    binds.push_back(Bind('e',            Bind::TimeCB(bind(&Entity::MoveUp,    screen->cam, _1))));
    screen->binds = &binds;

    scene.Add(new Entity("axis",  asset("axis")));
    scene.Add(new Entity("grid",  asset("grid")));

    // cling::Interpreter interpreter(argc, argv, "/Users/p/cling");

    if (!FLAGS_linear_program.empty()) {
        LocalFile lf(FLAGS_linear_program, "r");
        LinearProgram::Solve(&lf, 1);
        return 0;
    }

    // start our engine
    return app->Main();
}
