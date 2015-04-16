/*
 * $Id: yahoo_finance.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef __LFL_CRAWLER_YAHOO_FINANCE_H__
#define __LFL_CRAWLER_YAHOO_FINANCE_H__
namespace LFL {

struct YahooFinanceApi : public Crawler {
    vector<Quote> results;
    virtual void Results() {}

    bool scrape(int qf, const CrawlFileEntry *entry) {
        // INFO("scrape called w content='", entry->content(), "'");
        ScrapeCSV(entry, &results);
        if (results.size()) Results();
        return true;
    }

    static const int MaxSymbolsPerQuery = 200;

    static string URL(const char *query) {
        return StringPrintf("http://download.finance.yahoo.com/d/quotes.csv?s=%s&f=snd1t1l1abva2m3m4j1mwedr", query);
    }

    static float ParseMBValue(const char *text) {
        float val = atof(text);
        const char *denomination = NextChar(text, notnum);
        if (denomination) {
            if (*denomination == 'M') val *=    1000000;
            if (*denomination == 'B') val *= 1000000000;
        }
        return val;
    }

    static void ParseRangeValue(const char *line, float *low, float *high) {
        vector<string> field;
        Split(line, notnum, &field);
        if (field.size() != 2) { *low = *high = 0; return; }
        *low = atof(field[0].c_str());
        *high = atof(field[1].c_str());
    }

    static void ScrapeCSV(const CrawlFileEntry *entry, vector<Quote> *out) {
        Time now = Now();
        StringLineIter lines(entry->content());
        for (const char *line = lines.Next(); line; line = lines.Next()) {
            vector<string> field;
            Split(line, iscomma, isdquote, &field);
            if (field.size() != 17) { ERROR(field.size(), " != 17, skipping line ", line); continue; }

            Quote result; float range_low, range_high; int ind=0;
            result.mutable_info()->set_symbol(strip(field[ind++].c_str(), isdquote).c_str()); // s
            result.mutable_info()->set_name  (strip(field[ind++].c_str(), isdquote).c_str()); // n
            string datetext = strip(field[ind++].c_str(), isdquote); // d1
            string timetext = strip(field[ind++].c_str(), isdquote); // t1
            result.mutable_value()->set_time(NumericDate(datetext.c_str(), timetext.c_str(), WallStTimeZone()).count());
            result.mutable_value()->set_price   (atof(field[ind++].c_str())); // l1
            result.mutable_value()->set_ask     (atof(field[ind++].c_str())); // a
            result.mutable_value()->set_bid     (atof(field[ind++].c_str())); // b
            result.mutable_value()->set_volume  (atof(field[ind++].c_str())); // v
            result.mutable_info()->set_avg_daily_volume(atof(field[ind++].c_str())); // a2
            result.mutable_info()->set_fifty_day_avg   (atof(field[ind++].c_str())); // m3
            result.mutable_info()->set_two_hund_day_avg(atof(field[ind++].c_str())); // m4
            result.mutable_value()->set_capitalization(ParseMBValue(field[ind++].c_str())); // j1

            ParseRangeValue(field[ind++].c_str(), &range_low, &range_high); // m
            result.mutable_value()->mutable_day_range()->set_low (range_low);
            result.mutable_value()->mutable_day_range()->set_high(range_high);

            ParseRangeValue(field[ind++].c_str(), &range_low, &range_high); // w
            result.mutable_value()->mutable_year_range()->set_low (range_low);
            result.mutable_value()->mutable_year_range()->set_high(range_high);

            result.mutable_info()->set_earnings_per_share   (atof(field[ind++].c_str())); // e
            result.mutable_info()->set_dividends_per_share  (atof(field[ind++].c_str())); // d
            result.mutable_value()->set_price_earnings_ratio(atof(field[ind++].c_str())); // r

            // result.mutable_value()->set_response_text(entry->content().data(), entry->content().size());
            result.mutable_value()->set_response_time(now.count());
            out->push_back(result);

            // INFO("line ", line, " ", localhttptime(result.value().time()));
            // INFO("proto ", result.DebugString());
        }
    }
};

}; // namespace LFL
#endif // __LFL_CRAWLER_YAHOO_FINANCE_H__
