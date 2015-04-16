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
