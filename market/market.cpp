/*
 * $Id: market.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "market.pb.h"
#include "crawler/crawler.pb.h"

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"
#include "lfapp/network.h"

#include "market.h"
#include "crawler/crawler.h"
#include "crawler/yahoo_finance.h"

namespace LFL {
DEFINE_string(MarketDir, "/Users/p/lfl/market/", "Market data directory");
DEFINE_bool(yahoo_snp500, false, "Monitor S&P 500 via Yahoo Finance");
DEFINE_int(yahoo_delay_mins, 15, "Market feeds delayed by N mins");
DEFINE_bool(visualize, false, "Trading platform GUI");
DEFINE_string(quote_dump, "", "Dump quote file");
DEFINE_string(quote_clear_response_text, "", "Clear response text from quote file");

struct Watcher;

BindMap binds;
vector<Crawler*> crawlers;
vector<Watcher*> watchers;

MarketData *marketData = NULL;

struct QuoteLogger {
    typedef map<string, Quote> QuoteMap;
    QuoteMap quotes;
    ProtoFile out;
    QuoteLogger() {}

    bool Update(const Quote &next) {
        if (next.info().symbol().empty()) return false;
        Quote *last = &quotes[next.info().symbol().c_str()];
        if (!out.opened() || next.value().time() <= last->value().time()) return false;
        if (next.value().response_time() - next.value().time() >= Minutes(60)) return false;
        out.add(&next, 0);
        *last = next;
        return true;
    }
};

struct MyYahooFinanceApi : public YahooFinanceApi {
    MyYahooFinanceApi() : out(NULL) {}
    QuoteLogger *out;
    void Results() {
        for (vector<Quote>::iterator it = results.begin(); it != results.end(); ++it) {
            // INFO("quote: ", it->DebugString());
            if (out && out->Update(*it) && marketData) marketData->AddQuote(*it);
        }
        results.clear();
    }
} *yahoo_finance = 0;

struct Watcher {
    Crawler *crawler;
    string name;
    vector<string> url;
    QuoteLogger quote_logger;
    int delay_mins, trading_period, last_trading_period;
    Time last, frequency, trading_period_remaining;
    Watcher(Crawler *c, const string &n) : crawler(c), name(n), last(0), frequency(Seconds(30)), trading_period_remaining(0), delay_mins(0), trading_period(0), last_trading_period(0) {}

    void Update() {
        Time now = Now();
        trading_period = TradingPeriod::Now(now, &trading_period_remaining, Minutes(delay_mins));
        if (!app->frames_ran || trading_period != last_trading_period) {
            INFO(name, ": ", TradingPeriod::ToString(trading_period, trading_period_remaining));
            quote_logger.out.open((trading_period == TradingPeriod::MARKET) ? MarketData::filename(name, now).c_str() : 0);
        }
        last_trading_period = trading_period;

        if (!app->frames_ran || trading_period == TradingPeriod::MARKET) {
            if (last + frequency >= now) return;
            for (int i = 0; i < url.size(); ++i) crawler->queue[0].add(url[i].c_str());
            last = now;
        }
    }
};

struct TradingPlatformGUI : public GUI {
    Widget::Scrollbar scrollbar;
    TradingPlatformGUI(LFL::Window *W, Box w) : GUI(W, w), scrollbar(this) {}
    void Layout() {
        scrollbar.LayoutAttached(box);
        Flow flow(&box, Fonts::Default(), &child_box);
        flow.p.y -= scrollbar.scrolled * scrollbar.doc_height;
        for (MarketData::SymbolMap::iterator i = marketData->symbol.begin(); i != marketData->symbol.end(); ++i) {
            if (!i->second.chart.geom) {
                Waveform WF(point(screen->width*.95, screen->height*.2), &Color::white, &i->second);
                i->second.chart = WF;
            }
            flow.AppendText(i->first);
            flow.AppendNewlines(1);
            flow.AppendBox(i->second.chart.width, i->second.chart.height, &i->second.chart);
            flow.AppendNewlines(2);
        }
        scrollbar.SetDocHeight(flow.Height());
    }
    void Draw() { scrollbar.Update(); GUI::Draw(); }
} *tradingPlatformGUI = NULL;

int frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    Time now = Now();
    if (FLAGS_lfapp_video) {
        screen->gd->DrawMode(DrawMode::_2D);
        tradingPlatformGUI->Draw();
        screen->DrawDialogs();
    }
    
    for (vector<Watcher*>::iterator it = watchers.begin(); it != watchers.end(); ++it) {
        (*it)->Update();
    }

    for (vector<Crawler*>::iterator it = crawlers.begin(); it != crawlers.end(); ++it) {
        (*it)->crawl();
        (*it)->scrape();
    }

    return 0;
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    app->frame_cb = frame;
    app->logfilename = StrCat(dldir(), "market.txt");
    screen->caption = "Market";
    screen->width = 640;
    screen->height = 480;
    FLAGS_lfapp_camera = 0;

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }

    FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = FLAGS_visualize;

    if (app->Init()) { app->Free(); return -1; }

    binds.push_back(Bind(Key::Backquote, Bind::CB(bind([&](){ screen->console->Toggle(); }))));
    binds.push_back(Bind(Key::Escape,    Bind::CB(bind(&Shell::quit, &app->shell, vector<string>()))));
    screen->binds = &binds;

    if (!FLAGS_quote_dump.empty()) {
        ProtoFile pf(FLAGS_quote_dump.c_str()); Quote entry;
        while (pf.next(&entry)) printf("%s %s %s", localhttptime(entry.value().response_time()).c_str(),
                                                   localhttptime(entry.value().time()).c_str(), entry.DebugString().c_str());
    }

    if (!FLAGS_quote_clear_response_text.empty()) {
        ProtoFile pf(FLAGS_quote_clear_response_text.c_str()), out; Quote entry;
        out.open(string(FLAGS_quote_clear_response_text + ".cleared").c_str());
        while (pf.next(&entry)) {
            entry.mutable_value()->mutable_response_text()->clear();
            out.add(&entry, 0);
        }
    }

    if (FLAGS_yahoo_snp500) {
        yahoo_finance = new MyYahooFinanceApi();
        crawlers.push_back(yahoo_finance);
        if (!yahoo_finance->add("yahoo_finance.queue", "yahoo_finance")) return -1;
        yahoo_finance->validate();

        Watcher *watcher = new Watcher(yahoo_finance, "SNP500");
        watcher->delay_mins = FLAGS_yahoo_delay_mins;
        watchers.push_back(watcher);

        yahoo_finance->out = &watcher->quote_logger;

        vector<string> symbols;
        for (int i = 0; i < 500; i++) symbols.push_back(SNP500::Symbol(i));
        symbols.push_back("^OEX");
        symbols.push_back("^VIX");
        symbols.push_back("SPY");
        symbols.push_back("XLF");
        symbols.push_back("GLD");

        string symbol_text;
        for (int i = 0; i < symbols.size(); i++) {
            symbol_text += string(symbol_text.empty() ? "" : "+") + symbols[i];
            if (i != symbols.size()-1 && (!i || (i+1) % YahooFinanceApi::MaxSymbolsPerQuery != 0)) continue;
            watcher->url.push_back(YahooFinanceApi::URL(symbol_text.c_str()));
            symbol_text.clear();
        }
    }

    // Return = price[end]/price[beg] - 1;
    // Daily_ret = price[day1]/price[day0] - 1;
    // Risk = std_metric = stddev(Daily_ret);
    // MaxDrawDown = CurrentLowPoint/TrailingHighPoint - 1;
    // 250 trading days per year
    // SharpeRatio = E[Reward-RiskFreeReward ie Libor] / stddev(num) = mean(daily_ret) / stddev(daily_ret) * sqrt(250)

    if (!watchers.size() && !FLAGS_visualize) return 0;

    if (FLAGS_visualize) {
        tradingPlatformGUI = new TradingPlatformGUI(screen, Box::FromScreen());
        marketData = new MarketData(FLAGS_MarketDir.c_str(), "SNP500");
    }

    // start our engine
    return app->Main();
}
