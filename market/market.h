/*
 * $Id: market.h 1309 2014-10-10 19:20:55Z justin $
 * Copyright (C) 2013 Lucid Fusion Labs

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

#ifndef __LFL_MARKET_MARKET_H__
#define __LFL_MARKET_MARKET_H__
namespace LFL {

const char *WallStTimeZone(Time t=0) { return IsDaylightSavings(t) ? "EDT" : "EST"; }

int WallStHoursFromLocal(int *wall_st_hours_from_gmt_out=0) {
    Time now = Now();
    int local_hours_from_gmt   = RFC822TimeZone(LocalTimeZone(now));
    int wall_st_hours_from_gmt = RFC822TimeZone(WallStTimeZone(now));
    if (wall_st_hours_from_gmt_out) *wall_st_hours_from_gmt_out = wall_st_hours_from_gmt;
    return -1 * local_hours_from_gmt + wall_st_hours_from_gmt;
}

void WallStTime(Time t, struct tm *out, int hours_to_wall_st=0) {
    if (!hours_to_wall_st) hours_to_wall_st = WallStHoursFromLocal();
    localtm(Time2time_t(t + Hours(hours_to_wall_st)), out);
}

bool TradingHoliday(int month, int day, int year) {
    if (year == 2013) {
        static int holidays[][2] = { { 1, 1 }, { 1, 21 }, { 2, 18 }, { 3, 28 }, { 3, 29 }, { 5, 24 }, { 5, 27 }, { 7, 3 }, { 7, 4 }, { 9, 2 }, { 10, 14 }, { 11, 11 }, { 11, 28 }, { 11, 29 }, { 12, 24 }, { 12, 25 }, { 12, 31 } };
        for (int i=0, l=sizeofarray(holidays); i<l; i++) if (holidays[i][0] == month && holidays[i][1] == year) return true;
    }
    return false;
}

struct TradingPeriod {
    enum { NONE=0, MARKET=1, PRE_MARKET=2, AFTER_MARKET=3 };
    static const char *ToString(int id) {
        if      (id == NONE)         return "until pre-market opens";
        else if (id == PRE_MARKET)   return "in pre-market trading";
        else if (id == MARKET)       return "in trading day";
        else if (id == AFTER_MARKET) return "in after-market trading";
        return 0;
    };
    static string ToString(int id, Time remaining) {
        return StrCat(intervaltime(remaining), " left ", ToString(id));
    }

    static int Now(Time now, Time *remaining=0, Time delayed=0) { 
        int ret = NONE, wall_st_hours_from_gmt, hours_to_wall_st = WallStHoursFromLocal(&wall_st_hours_from_gmt);
        Time days_away = 0;
        struct tm t;

        for (;; days_away += Hours(24)) {

            // Find next trading day
            for (;; days_away += Hours(24)) {
                WallStTime(now + days_away, &t, hours_to_wall_st);
                bool week_day = t.tm_wday >= 1 && t.tm_wday <= 5;
                if (week_day && !TradingHoliday(t.tm_year+1900, t.tm_mon+1, t.tm_mday)) break;
            }
            if (days_away) days_away -= SinceDayBegan(now + days_away, wall_st_hours_from_gmt);
            Time trading_day_elapsed = SinceDayBegan(now + days_away, wall_st_hours_from_gmt);

            // Find next trading period
            static int  trading_period_id  [] = { NONE,     PRE_MARKET,           MARKET,    AFTER_MARKET };
            static Time trading_period_time[] = { Hours(4), Hours(9)+Minutes(30), Hours(16), Hours(20),   };
            for (int i=0, l=sizeofarray(trading_period_id); i<l; i++) 
                if (trading_day_elapsed < (trading_period_time[i] + delayed)) {
                    if (remaining) *remaining = (trading_period_time[i] + delayed - trading_day_elapsed) + days_away;
                    return trading_period_id[i];
                }
        }
    }
};

struct SNP100 {
    static const char *Symbol(int n) {
        CHECK(n >= 0 && n < 100);
        static const char *symbols[] = {
#undef  XX
#define XX(x) x,
#undef  YY
#define YY(x)
#include "snp100.h"
        };
        return symbols[n];
    }
};

struct SNP500 {
    static const char *Symbol(int n) {
        CHECK(n >= 0 && n < 500);
        static const char *symbols[] = {
#undef  XX
#define XX(x) x,
#undef  YY
#define YY(x)
#include "snp500.h"
        };
        return symbols[n];
    }
};

struct MarketData {
    static string filename(const string &name, Time t) { return StrCat(name, ".trading_day_", logfileday(t), ".pb"); }

    struct Symbol : public Vec<float> {
        Waveform chart;
        vector<Quote> quote;
        void AddQuote(const Quote &q) { quote.push_back(q); }
        virtual int Len() const { return quote.size(); }
        virtual float Read(int n) const { return quote[n].value().price(); }
    };

    typedef map<string, Symbol> SymbolMap;
    SymbolMap symbol;

    MarketData(const char *market_dir, const char *name) {
        string prefix = name + string(".trading_day_");
        DirectoryIter d(market_dir, 0, prefix.c_str(), ".pb");
        for (const char *fn = d.Next(); Running() && fn; fn = d.Next()) {
            INFO("MarketData Open ", fn);
            ProtoFile trading_day((market_dir + string(fn)).c_str());
            if (!trading_day.Opened()) { ERROR("open ", fn, " failed"); continue; }

            Quote quote;
            while (trading_day.Next(&quote)) symbol[quote.info().symbol().c_str()].AddQuote(quote);
        }
    }

    void AddQuote(const Quote &q) { symbol[q.info().symbol().c_str()].AddQuote(q); }
};

}; // namespace LFL
#endif // __LFL_MARKET_MARKET_H__
