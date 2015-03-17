/*
 * $Id: chess.cpp 1336 2014-12-08 09:29:59Z justin $
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
#include "chess.h"

namespace LFL {
AssetMap asset;
SoundAssetMap soundasset;
Chess::Position position;
Box SquareCoords(int p) {
    static const int border = 5;
    int w = screen->width-2*border, h = screen->height-2*border;
    return Box(border+Chess::SquareX(p)/8.0*w, border+Chess::SquareY(p)/8.0*h, 1/8.0*w, 1/8.0*h);
}

// engine callback driven by LFL::Application
int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    Box win = screen->Box();
    screen->gd->DrawMode(DrawMode::_2D);
    screen->gd->EnableLayering();

    static Asset *board = asset("board");
    board->tex.Draw(win);

    int black_font_index[7] = { 0, 3, 2, 0, 5, 4, 1 }, bits[65];
    static Font *pieces = Fonts::Get("ChessPieces1", 0, Color::black);
    for (int i=Chess::PAWN; i <= Chess::KING; i++) {
        Bit::Indices(position.white[i], bits); for (int *b = bits; *b != -1; b++) { Box w=SquareCoords(*b); pieces->DrawGlyph(black_font_index[i]+6, w); }
        Bit::Indices(position.black[i], bits); for (int *b = bits; *b != -1; b++) { Box w=SquareCoords(*b); pieces->DrawGlyph(black_font_index[i],   w); }
    }

    screen->DrawDialogs();
    return 0;
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    app->logfilename = StrCat(LFAppDownloadDir(), "chess.txt");
    app->frame_cb = Frame;
    screen->width = 630;
    screen->height = 570;
    screen->caption = "Chess";
    FLAGS_lfapp_video = FLAGS_lfapp_input = FLAGS_lfapp_network = 1;

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init()) { app->Free(); return -1; }

    Fonts::InsertAtlas("ChessPieces1", "", 0, Color::black, 0);

    // asset.Add(Asset(name, texture,     scale, translate, rotate, geometry, 0, 0, 0));
    asset.Add(Asset("board", "board.png", 0,     0,         0,      0,        0, 0, 0));
    asset.Load();
    app->shell.assets = &asset;

    // soundasset.Add(SoundAsset(name, filename,   ringbuf, channels, sample_rate, seconds ));
    soundasset.Add(SoundAsset("draw",  "Draw.wav", 0,       0,        0,           0       ));
    soundasset.Load();
    app->shell.soundassets = &soundasset;

    BindMap *binds = screen->binds = new BindMap();
//  binds->Add(Bind(key,            callback));
    binds->Add(Bind(Key::Backquote, Bind::CB(bind([&](){ screen->console->Toggle(); }))));
    binds->Add(Bind(Key::Quote,     Bind::CB(bind([&](){ screen->console->Toggle(); }))));
    binds->Add(Bind(Key::Escape,    Bind::CB(bind(&Shell::quit,            &app->shell, vector<string>()))));

    // start our engine
    return app->Main();
}
