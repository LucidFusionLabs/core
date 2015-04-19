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
#include "lfapp/time.h"

namespace LFL {
Time Now() { return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()); }
void MSleep(int ms) { std::this_thread::sleep_for(milliseconds(ms)); }

time_t Time2time_t(Time x) { return ToSeconds(x).count(); }
timeval Time2timeval(Time x) {
    microseconds us = duration_cast<microseconds>(x);
    struct timeval ret = { (int)us.count() / 1000000, (int)us.count() % 1000000 };
    return ret;
}

#if defined (WIN32) || defined(LFL_ANDROID)
int is_leap(unsigned y) { y += 1900; return (y % 4) == 0 && ((y % 100) != 0 || (y % 400) == 0); }

time_t timegm(struct tm *tm) {
    static const unsigned ndays[2][12] = {
        {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
        {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
    };

    time_t res = 0;
    for (int i = 70; i < tm->tm_year; ++i) res += is_leap(i) ? 366 : 365;
    for (int i = 0; i < tm->tm_mon; ++i) res += ndays[is_leap(tm->tm_year)][i];

    res += tm->tm_mday - 1;
    res *= 24;
    res += tm->tm_hour;
    res *= 60;
    res += tm->tm_min;
    res *= 60;
    res += tm->tm_sec;
    return res;
}
#endif

const char *dayname(int wday) {
    static const char *dn[] = { "Sun", "Mon", "Tue", "Wed", "Thr", "Fri", "Sat" };
    if (wday < 0 || wday >= 7) return 0;
    return dn[wday];
}

int RFC822Day(const char *text) {
    static const char *dn[] = { "Sun", "Mon", "Tue", "Wed", "Thr", "Fri", "Sat" };
    for (int i=0, l=sizeofarray(dn); i<l; i++) if (!strcmp(text, dn[i])) return i;
    return 0;
}

const char *monthname(int mon) {
    static const char *mn[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
    if (mon < 0 || mon >= 12) return 0;
    return mn[mon];
}

int RFC822Month(const char *text) {
    static const char *mn[] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
    for (int i=0, l=sizeofarray(mn); i<l; i++) if (!strcmp(text, mn[i])) return i;
    return 0;
}

int RFC822TimeZone(const char *text) {
    static const char *tzname[] = { "GMT", "EST", "EDT", "CST", "CDT", "MST", "MDT", "PST", "PDT", };
    static const int tzoffset[] = { 0,     -5,    -4,    -6,    -5,    -7,    -6,    -8,    -7     };
    for (int i=0, l=sizeofarray(tzname); i<l; i++) if (!strcmp(text, tzname[i])) return tzoffset[i];
    return 0;
}

void GMTtm(time_t in, struct tm *t) { gmtime_r(&in, t); }
void GMTtm(struct tm *t) { return GMTtm(time(0), t); }

void localtm(time_t in, struct tm *t) { localtime_r(&in, t);}
void localtm(struct tm *t) { return localtm(time(0), t); }

string logtime(Time t) { char buf[128] = {0}; logtime(t, buf, sizeof(buf)); return buf; }
int logtime(char *buf, int size) { return logtime(Now(), buf, size); }
int logtime(Time t, char *buf, int size) { time_t tt=Time2time_t(t); return logtime(tt, (t-Seconds(tt)).count(), buf, size); }
int logtime(time_t secs, int ms, char *buf, int size) { struct tm tm; localtm(secs, &tm); return logtime(&tm, ms, buf, size); }
int logtime(struct tm *tm, int ms, char *buf, int size) {
    return snprintf(buf, size, "%02d:%02d:%02d.%03d", tm->tm_hour, tm->tm_min, tm->tm_sec, ms);
}

string logfileday(Time t) { char buf[128] = {0}; logfileday(Time2time_t(t), buf, sizeof(buf)); return buf; }
int logfileday(char *buf, int size) { return logfileday(time(0), buf, size); }
int logfileday(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return logfileday(&tm, buf, size); }
int logfileday(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%04d-%02d-%02d", 1900+tm->tm_year, tm->tm_mon+1, tm->tm_mday);
}

string logfiledaytime(Time t) { char buf[128] = {0}; logfiledaytime(Time2time_t(t), buf, sizeof(buf)); return buf; }
int logfiledaytime(char *buf, int size) { return logfiledaytime(time(0), buf, size); }
int logfiledaytime(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return logfiledaytime(&tm, buf, size); }
int logfiledaytime(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%04d-%02d-%02d_%02d:%02d", 1900+tm->tm_year, tm->tm_mon+1, tm->tm_mday, tm->tm_hour, tm->tm_min);
}

int httptime(char *buf, int size) { return httptime(time(0), buf, size); }
int httptime(time_t t, char *buf, int size) { struct tm tm; GMTtm(t, &tm); return httptime(&tm, buf, size); }
int httptime(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%s, %d %s %d %02d:%02d:%02d GMT",
        dayname(tm->tm_wday), tm->tm_mday, monthname(tm->tm_mon), 1900+tm->tm_year,
        tm->tm_hour, tm->tm_min, tm->tm_sec);
}

string localhttptime(Time t) { char buf[128] = {0}; localhttptime(Time2time_t(t), buf, sizeof(buf)); return buf; }
int localhttptime(char *buf, int size) { return localhttptime(time(0), buf, size); }
int localhttptime(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return localhttptime(&tm, buf, size); }
int localhttptime(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%s, %d %s %d %02d:%02d:%02d %s",
        dayname(tm->tm_wday), tm->tm_mday, monthname(tm->tm_mon), 1900+tm->tm_year,
        tm->tm_hour, tm->tm_min, tm->tm_sec,
#ifdef WIN32
        "");
#else
        tm->tm_zone);
#endif
}

string localsmtptime(Time t) { char buf[128] = {0}; localsmtptime(Time2time_t(t), buf, sizeof(buf)); return buf; }
int localsmtptime(char *buf, int size) { return localsmtptime(time(0), buf, size); }
int localsmtptime(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return localsmtptime(&tm, buf, size); }
int localsmtptime(struct tm *tm, char *buf, int size) {
    int tzo = 
#ifdef WIN32
        0;
#else
        RFC822TimeZone(tm->tm_zone)*100;
#endif
    return snprintf(buf, size, "%s, %02d %s %d %02d:%02d:%02d %s%04d",
        dayname(tm->tm_wday), tm->tm_mday, monthname(tm->tm_mon), 1900+tm->tm_year,
        tm->tm_hour, tm->tm_min, tm->tm_sec, tzo<0?"-":"", abs(tzo));
}

string localmboxtime(Time t) { char buf[128] = {0}; localmboxtime(Time2time_t(t), buf, sizeof(buf)); return buf; }
int localmboxtime(char *buf, int size) { return localmboxtime(time(0), buf, size); }
int localmboxtime(time_t t, char *buf, int size) { struct tm tm; localtm(t, &tm); return localmboxtime(&tm, buf, size); }
int localmboxtime(struct tm *tm, char *buf, int size) {
    return snprintf(buf, size, "%s %s%s%d %02d:%02d:%02d %d",
        dayname(tm->tm_wday), monthname(tm->tm_mon), tm->tm_mday < 10 ? "  " : " ",
        tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec, 1900+tm->tm_year);
}

string intervaltime(Time t) { time_t tt=Time2time_t(t); char buf[64] = {0}; intervaltime(tt, (t-Seconds(tt)).count(), buf, sizeof(buf)); return buf; }
int intervaltime(time_t t, int ms, char *buf, int size) {
    int hours = t/3600;
    t -= hours*3600;
    int minutes = t/60;
    int seconds = t - minutes*60;
    return snprintf(buf, size, "%02d:%02d:%02d.%03d", hours, minutes, seconds, ms);
}

string intervalminutes(Time t) { time_t tt=Time2time_t(t); char buf[64] = {0}; intervalminutes(tt, (t-Seconds(tt)).count(), buf, sizeof(buf)); return buf; }
int intervalminutes(time_t t, int ms, char *buf, int size) {
    int minutes = t/60;
    int seconds = t - minutes*60;
    return snprintf(buf, size, "%02d:%02d", minutes, seconds);
}

bool RFC822Time(const char *text, int *hour, int *min, int *sec) {
    int textlen = strlen(text);
    if (textlen < 5 || text[2] != ':') return false;
    if (hour) *hour = atoi(text);
    if (min) *min = atoi(&text[3]);
    if (textlen == 5) { 
        if (sec) *sec = 0;
        return true;
    }
    if (textlen != 8 || text[5] != ':') return false;
    if (sec) *sec = atoi(&text[6]);
    return true;
}

Time RFC822Date(const char *text) {
    struct tm tm; memset(&tm, 0, sizeof(tm));
    const char *comma = strchr(text, ','), *start = comma ? comma + 1 : text, *parsetext;
    StringWordIter words(start);
    tm.tm_mday = atoi(IterNextString(&words));
    tm.tm_mon = RFC822Month(IterNextString(&words).c_str());
    tm.tm_year = atoi(IterNextString(&words)) - 1900;
    string timetext = IterNextString(&words);
    if (!RFC822Time(timetext.c_str(), &tm.tm_hour, &tm.tm_min, &tm.tm_sec))
        { ERROR("RFC822Date('", text, "') RFC822Time('", timetext, "') failed"); return Time(0); }
    int hours_from_gmt = RFC822TimeZone(IterNextString(&words).c_str());
    return Seconds(timegm(&tm) - hours_from_gmt * 3600);
}

bool NumericTime(const char *text, int *hour, int *min, int *sec) {
    int textlen = strlen(text);
    StringWordIter words(StringPiece(text, textlen), isint<':'>);
    *hour = atoi(IterNextString(&words));
    *min = atoi(IterNextString(&words));
    *sec = atoi(IterNextString(&words));
    if (textlen >= 2 && !strcmp(text+textlen-2, "pm") && *hour != 12) *hour += 12;
    return true;
}

Time NumericDate(const char *datetext, const char *timetext, const char *timezone) {
    struct tm tm; memset(&tm, 0, sizeof(tm));
    StringWordIter words(datetext, isint<'/'>);
    tm.tm_mon = atoi(IterNextString(&words)) - 1;
    tm.tm_mday = atoi(IterNextString(&words));
    tm.tm_year = atoi(IterNextString(&words)) - 1900;
    NumericTime(timetext, &tm.tm_hour, &tm.tm_min, &tm.tm_sec);
    int hours_from_gmt = RFC822TimeZone(BlankNull(timezone));
    return Seconds(timegm(&tm) - hours_from_gmt * 3600);
}

Time SinceDayBegan(Time t, int gmt_offset_hrs) {
    Time ret = (t % Hours(24)) + Hours(gmt_offset_hrs);
    return ret < Time(0) ? ret + Hours(24) : ret;
}

bool    IsDaylightSavings(Time t) { struct tm tm; localtm(t != Time(0) ? Time2time_t(t) : time(0), &tm); return tm.tm_isdst; }
#ifndef WIN32
const char *LocalTimeZone(Time t) { struct tm tm; localtm(t != Time(0) ? Time2time_t(t) : time(0), &tm); return tm.tm_zone; }
#else
const char *LocalTimeZone(Time t) { return _tzname[_daylight]; }
#endif
const char *WallStTimeZone(Time t) { return IsDaylightSavings(t) ? "EDT" : "EST"; }

int WallStHoursFromLocal(int *wall_st_hours_from_gmt_out) {
    Time now = Now();
    int local_hours_from_gmt   = RFC822TimeZone(LocalTimeZone(now));
    int wall_st_hours_from_gmt = RFC822TimeZone(WallStTimeZone(now));
    if (wall_st_hours_from_gmt_out) *wall_st_hours_from_gmt_out = wall_st_hours_from_gmt;
    return -1 * local_hours_from_gmt + wall_st_hours_from_gmt;
}

void WallStTime(Time t, struct tm *out, int hours_to_wall_st) {
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

const char *TradingPeriod::ToString(int id) {
    if      (id == NONE)         return "until pre-market opens";
    else if (id == PRE_MARKET)   return "in pre-market trading";
    else if (id == MARKET)       return "in trading day";
    else if (id == AFTER_MARKET) return "in after-market trading";
    return 0;
};

string TradingPeriod::ToString(int id, Time remaining) {
    return StrCat(intervaltime(remaining), " left ", ToString(id));
}

int TradingPeriod::Now(Time now, Time *remaining, Time delayed) { 
    int ret = NONE, wall_st_hours_from_gmt, hours_to_wall_st = WallStHoursFromLocal(&wall_st_hours_from_gmt);
    Time days_away = Time(0);
    struct tm t;

    for (;; days_away += Hours(24)) {

        // Find next trading day
        for (;; days_away += Hours(24)) {
            WallStTime(now + days_away, &t, hours_to_wall_st);
            bool week_day = t.tm_wday >= 1 && t.tm_wday <= 5;
            if (week_day && !TradingHoliday(t.tm_year+1900, t.tm_mon+1, t.tm_mday)) break;
        }
        if (days_away != Time(0)) days_away -= SinceDayBegan(now + days_away, wall_st_hours_from_gmt);
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

}; // namespace LFL
