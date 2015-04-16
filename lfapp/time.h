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

#ifndef __LFL_LFAPP_TIME_H__
#define __LFL_LFAPP_TIME_H__

#define ToDays(x) duration_cast<days>(x)
#define ToFDays(x) duration_cast<fdays>(x)
#define ToHours(x) duration_cast<hours>(x)
#define ToFHours(x) duration_cast<fhours>(x)
#define ToMinutes(x) duration_cast<minutes>(x)
#define ToFMinutes(x) duration_cast<fminutes>(x)
#define ToSeconds(x) duration_cast<seconds>(x)
#define ToFSeconds(x) duration_cast<fseconds>(x)
#define ToMilliseconds(x) x
#define ToFMilliseconds(x) duration_cast<FTime>(x)
#define ToMicroseconds(x) duration_cast<microseconds>(x)
#define ToFMicroseconds(x) duration_cast<fmicroseconds>(x)

namespace LFL {
typedef duration<float, std::micro>        fmicroseconds;
typedef duration<float, std::milli>        fmilliseconds;
typedef duration<float>                    fseconds;
typedef duration<float, std::ratio<60>>    fminutes;
typedef duration<float, std::ratio<3600>>  fhours;
typedef duration<float, std::ratio<86400>> fdays;
typedef duration<int,   std::ratio<86400>> days;

typedef milliseconds Time;
typedef fmilliseconds FTime;

inline Time          Days(int   x) { return duration_cast<Time>(    days(x)); }
inline Time         FDays(float x) { return duration_cast<Time>(   fdays(x)); }
inline Time         Hours(int   x) { return duration_cast<Time>(   hours(x)); }
inline Time        FHours(float x) { return duration_cast<Time>(  fhours(x)); }
inline Time       Minutes(int   x) { return duration_cast<Time>( minutes(x)); }
inline Time      FMinutes(float x) { return duration_cast<Time>(fminutes(x)); }
inline Time       Seconds(int   x) { return duration_cast<Time>( seconds(x)); }
inline Time      FSeconds(float x) { return duration_cast<Time>(fseconds(x)); }
inline Time  Milliseconds(int   x) { return Time(x); }
inline Time FMilliseconds(float x) { return duration_cast<Time>(fmilliseconds(x)); }
inline Time  Microseconds(int   x) { return duration_cast<Time>( microseconds(x)); }
inline Time FMicroseconds(float x) { return duration_cast<Time>(fmicroseconds(x)); }

time_t Time2time_t(Time x);
timeval Time2timeval(Time x);
void localtm(time_t, struct tm *t);
void GMTtm(time_t, struct tm *t);
string logtime(Time t);
int logtime(char *buf, int size);
int logtime(Time time, char *buf, int size);
int logtime(time_t secs, int ms, char *buf, int size);
int logtime(struct tm*, int ms, char *buf, int size);
string logfileday(Time t);
int logfileday(char *buf, int size);
int logfileday(time_t t, char *buf, int size);
int logfileday(struct tm *tm, char *buf, int size);
string logfiledaytime(Time t);
int logfiledaytime(char *buf, int size);
int logfiledaytime(time_t t, char *buf, int size);
int logfiledaytime(struct tm *tm, char *buf, int size);
int httptime(char *buf, int size);
int httptime(time_t time, char *buf, int size);
int httptime(struct tm*, char *buf, int size);
int localhttptime(char *buf, int size);
int localhttptime(time_t time, char *buf, int size);
int localhttptime(struct tm*, char *buf, int size);
string localhttptime(Time t);
int localsmtptime(char *buf, int size);
int localsmtptime(time_t time, char *buf, int size);
int localsmtptime(struct tm*, char *buf, int size);
string localsmtptime(Time t);
int localmboxtime(char *buf, int size);
int localmboxtime(time_t time, char *buf, int size);
int localmboxtime(struct tm*, char *buf, int size);
string localmboxtime(Time t);
int intervaltime(time_t t, int ms, char *buf, int size);
string intervaltime(Time t);
int intervalminutes(time_t t, int ms, char *buf, int size);
string intervalminutes(Time t);

int RFC822TimeZone(const char *text);
Time RFC822Date(const char *text);
Time NumericDate(const char *datetext, const char *timetext, const char *timezone);

Time SinceDayBegan(Time, int gmt_offset_hrs);
bool IsDaylightSavings(Time t=Time(0));
const char *LocalTimeZone(Time t=Time(0));
const char *WallStTimeZone(Time t=Time(0));

int WallStHoursFromLocal(int *wall_st_hours_from_gmt_out=0);
void WallStTime(Time t, struct tm *out, int hours_to_wall_st=0);
bool TradingHoliday(int month, int day, int year);

struct TradingPeriod {
    enum { NONE=0, MARKET=1, PRE_MARKET=2, AFTER_MARKET=3 };
    static const char *ToString(int id);
    static string ToString(int id, Time remaining);
    static int Now(Time now, Time *remaining=0, Time delayed=Time(0));
};

}; // namespace LFL
#endif // __LFL_LFAPP_TIME_H__
