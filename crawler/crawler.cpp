/*
 * $Id: crawler.cpp 1306 2014-09-04 07:13:16Z justin $
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

#include "crawler.pb.h"

#include "lfapp/lfapp.h"
#include "lfapp/network.h"
#include "lfapp/dom.h"

#include "crawler.h"
#include "html.h"

namespace LFL {
DEFINE_int(q0max_completed,   1, "Max queue[0] querys");
DEFINE_int(q0max_outstanding, 5, "Max queue[0] simultaneous querys");
DEFINE_int(q1max_outstanding, 5, "Max queue[1] simultaneous querys");
DEFINE_string(forvo_init, "", "Initialize crawl queue with file of forvo querys");
DEFINE_string(forvo_dump, "", "Dump contents of forvo crawl to directory");
DEFINE_bool(forvo_crawl, false, "Crawl forvo");
DEFINE_bool(forvo_dumpmp3, false, "Dump mp3 files during forvo_dump");
DEFINE_bool(forvo_dumpwav, false, "Dump wav files during forvo_dump");
DEFINE_string(google_init, "", "Initialize crawl queue with google query");
DEFINE_bool(google_crawl, false, "Crawl google");
}; // namespace LFL

#include "forvo.h"
#include "google.h"

using namespace LFL;
Crawler *crawler = 0;

DEFINE_bool(crawl, true, "Crawl");
DEFINE_bool(scrape, true, "Scrape");

DEFINE_string(queue_dump, "", "Dump queue dump");
DEFINE_string(crawl_dump, "", "Dump crawl dump");

int frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {

    bool crawl_done = crawler->crawl_done();
    bool scrape_done = crawler->scrape_done();
    bool q0_maxed = crawler->queue[0].completed >= FLAGS_q0max_completed;

    /* done */
    if ((crawl_done && scrape_done) || (q0_maxed && crawler->crawl_done(1) && scrape_done))
    { INFO("no more input, ", crawler->queue[0].completed, " completed, ", crawler->queue[0].scraped, " scraped, exiting"); app->run=0; return 0; }

    /* crawl */
    crawler->crawl();

    /* scrape */
    crawler->scrape();

    return 0;
}

extern "C" int main(int argc, const char *argv[]) {

    app->logfilename = StrCat(dldir(), "crawler.txt");
    app->frame_cb = frame;
    screen->caption = "crawler";
    FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = FLAGS_lfapp_camera = 0;
    FLAGS_lfapp_network = 1;
#ifdef _WIN32
    FLAGS_open_console = 1;
#endif

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init()) { app->Free(); return -1; }

    if (FLAGS_queue_dump.size() || FLAGS_crawl_dump.size()) {
        ProtoFile pf(FLAGS_crawl_dump.size() ? FLAGS_crawl_dump.c_str() : FLAGS_queue_dump.c_str());
        QueueFileEntry queue_entry; CrawlFileEntry crawl_entry; Proto *entry = &queue_entry; int offset;
        if (FLAGS_crawl_dump.size()) entry = &crawl_entry;
        while (pf.next(entry, &offset)) printf("@%d %s\n", offset, entry->DebugString().c_str());
    }

    /* forvo */
    if (FLAGS_forvo_init.size() || FLAGS_forvo_crawl || FLAGS_forvo_dump.size()) {
        crawler = new ForvoApi();
        if (!crawler->add("forvo.root.queue", "forvo.root")) return -1;
        if (!crawler->add("forvo.mp3.queue", "forvo.mp3")) return -1;

        if (FLAGS_forvo_init.size()) ((ForvoApi*)crawler)->init(FLAGS_forvo_init.c_str());

        if (FLAGS_forvo_dump.size()) ((ForvoApi*)crawler)->dump(FLAGS_forvo_dump.c_str());

        if (!FLAGS_forvo_crawl) Typed::Replace<Crawler>(&crawler, 0);
    }

    /* google */
    if (FLAGS_google_init.size() || FLAGS_google_crawl) {
        crawler = new GoogleApi();
        if (!crawler->add("google.search.queue", "google.search")) return -1;
        if (!crawler->add("google.result.queue", "google.result")) return -1;

        if (FLAGS_google_init.size()) ((GoogleApi*)crawler)->init(FLAGS_google_init.c_str());

        if (!FLAGS_google_crawl) Typed::Replace<Crawler>(&crawler, 0);
    }

    /* main */
    if (!crawler) FATAL("no crawler: ", crawler);
    crawler->validate();

    int ret = app->Main();
    delete crawler;
    return ret;
}
