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

namespace LFL {
unique_ptr<ProcessAPIClient> process_api;
}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {
    app->logfilename = StrCat(LFAppDownloadDir(), "lterm-render.txt");
    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }

    int optind = Singleton<FlagMap>::Get()->optind;
    if (optind >= argc) { fprintf(stderr, "Usage: %s [-flags] <socket-name>\n", argv[0]); return -1; }
    // if (app->Init()) { app->Free(); return -1; }

    const string socket_name = StrCat(argv[optind]);
    process_api = unique_ptr<ProcessAPIClient>(new ProcessAPIClient());
    process_api->Start(StrCat(argv[optind]));
    process_api->HandleMessagesLoop();
    INFO("render: exiting");
    return 0;
}
