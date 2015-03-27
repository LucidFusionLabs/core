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

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"

namespace LFL {
DEFINE_bool(wrap, 0, "Wrap lines");
AssetMap asset;
SoundAssetMap soundasset;
Scene scene;
EditorDialog *editor; 

void Reshaped() {
    editor->box = screen->Box();
    editor->Layout();
}
int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    screen->gd->DrawMode(DrawMode::_2D);
    screen->DrawDialogs();
    return 0;
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    app->logfilename = StrCat(LFAppDownloadDir(), "editor.txt");
    app->frame_cb = Frame;
    screen->width = 840;
    screen->height = 760;
    screen->caption = "Editor";
    FLAGS_lfapp_video = FLAGS_lfapp_input = true;

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init()) { app->Free(); return -1; }
    app->scheduler.AddWaitForeverKeyboard();
    app->scheduler.AddWaitForeverMouse();
    app->reshaped_cb = LFL::Reshaped;

    BindMap *binds = screen->binds = new BindMap();
    // binds.push_back(Bind(key,         callback));
    binds->Add(Bind('6', Key::Modifier::Cmd, Bind::CB(bind([&]() { screen->console->Toggle(); }))));

    chdir(app->startdir.c_str());
    int optind = Singleton<FlagMap>::Get()->optind;
    if (optind >= argc) { fprintf(stderr, "Usage: %s [-flags] <file>\n", argv[0]); return -1; }
    string s = LocalFile::FileContents(StrCat(argv[optind]));

    Font *font = Fonts::Get(FLAGS_default_font, "", FLAGS_default_font_size, Color::black);
    editor = new EditorDialog(screen, font, new BufferFile(s), 1, 1,
                              Dialog::Flag::Fullscreen | (FLAGS_wrap ? EditorDialog::Flag::Wrap : 0));
    editor->color = Color::white;

    // start our engine
    return app->Main();
}
