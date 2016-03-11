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

#include "core/app/app.h"
#include "core/web/dom.h"
#include "core/web/css.h"
#include "core/app/flow.h"
#include "core/app/gui.h"

namespace LFL {
struct MyAppState {
  AssetMap asset;
  SoundAssetMap soundasset;
} *my_app;

struct MyGUI : public GUI {
  Scene scene;
  MyGUI() {
    scene.Add(new Entity("axis",  my_app->asset("axis")));
    scene.Add(new Entity("grid",  my_app->asset("grid")));
    scene.Add(new Entity("room",  my_app->asset("room")));
    scene.Add(new Entity("arrow", my_app->asset("arrow"), v3(1, .24, 1)));
  }

  int Frame(LFL::Window *W, unsigned clicks, int flag) {
    W->cam->Look(W->gd);
    W->GetInputController<BindMap>(0)->Repeat(clicks);
    scene.Get("arrow")->YawRight(double(clicks));
    scene.Draw(&my_app->asset.vec);

    W->gd->DrawMode(DrawMode::_2D);
    W->DrawDialogs();
    W->default_font->Draw(StringPrintf("Hello warld, my FPS is %.2f", app->FPS()), point(W->width*.05, W->height*.15));
    W->default_font->Draw("press tick for console",                                point(W->width*.05, W->height*.05));
    return 0;
  }
};

void MyWindowInit(Window *W) {
  W->caption = "$PKGNAME";
  W->width = 420;
  W->height = 380;
}

void MyWindowStart(Window *W) {
  MyGUI *gui = W->AddGUI(make_unique<MyGUI>());
  W->frame_cb = bind(&MyGUI::Frame, gui, _1, _2, _3);
  W->shell = make_unique<Shell>(&my_app->asset, &my_app->soundasset, nullptr);
  if (FLAGS_console) W->InitConsole(Callback());
  BindMap *binds = W->AddInputController(make_unique<BindMap>());
  binds->Add(Key::Escape,    Bind::CB(bind(&Shell::quit,     W->shell.get(), vector<string>())));
  binds->Add(Key::Return,    Bind::CB(bind(&Shell::grabmode, W->shell.get(), vector<string>())));
  binds->Add(Key::Backquote, Bind::CB(bind(&Shell::console,  W->shell.get(), vector<string>())));
  binds->Add(Key::Quote,     Bind::CB(bind(&Shell::console,  W->shell.get(), vector<string>())));
  binds->Add(Key::LeftShift, Bind::TimeCB(bind(&Entity::RollLeft,   W->cam.get(), _1)));
  binds->Add(Key::Space,     Bind::TimeCB(bind(&Entity::RollRight,  W->cam.get(), _1)));
  binds->Add('w',            Bind::TimeCB(bind(&Entity::MoveFwd,    W->cam.get(), _1)));
  binds->Add('s',            Bind::TimeCB(bind(&Entity::MoveRev,    W->cam.get(), _1)));
  binds->Add('a',            Bind::TimeCB(bind(&Entity::MoveLeft,   W->cam.get(), _1)));
  binds->Add('d',            Bind::TimeCB(bind(&Entity::MoveRight,  W->cam.get(), _1)));
  binds->Add('q',            Bind::TimeCB(bind(&Entity::MoveDown,   W->cam.get(), _1)));
  binds->Add('e',            Bind::TimeCB(bind(&Entity::MoveUp,     W->cam.get(), _1)));
}

}; // namespace LFL
using namespace LFL;

extern "C" void MyAppCreate() {
  FLAGS_target_fps = 30;
  FLAGS_font_engine = "atlas";
  FLAGS_default_font = FLAGS_console_font = "Nobile.ttf";
  FLAGS_default_font_flag = FLAGS_console_font_flag = 0;
  FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = FLAGS_console = 1;
  app = new Application();
  screen = new Window();
  my_app = new MyAppState();
  app->exit_cb = [](){ delete my_app; };
  app->logfilename = StrCat(LFAppDownloadDir(), "$BINNAME.txt");
  app->window_start_cb = MyWindowStart;
  app->window_init_cb = MyWindowInit;
  app->window_init_cb(screen);
}

extern "C" int MyAppMain(int argc, const char* const* argv) {
  if (app->Create(argc, argv, __FILE__)) return -1;
  if (app->Init()) return -1;
  screen->gd->default_draw_mode = DrawMode::_3D;

  // my_app->asset.Add(name, texture,  scale, translate, rotate, geometry                  hull
  my_app->asset.Add("axis",  "",       0,     0,         0,      nullptr,                  nullptr, 0, 0, glAxis  );
  my_app->asset.Add("grid",  "",       0,     0,         0,      Grid::Grid3D().release(), nullptr, 0, 0          );
  my_app->asset.Add("room",  "",       0,     0,         0,      nullptr,                  nullptr, 0, 0, glRoom  );
  my_app->asset.Add("arrow", "",      .005,   1,        -90,     "arrow.obj",              nullptr, 0             );
  my_app->asset.Load();

  app->StartNewWindow(screen);
  return app->Main();
}
