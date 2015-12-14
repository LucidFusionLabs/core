/*
 * $Id: counter.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef LFL_ML_COUNTER_H__
#define LFL_ML_COUNTER_H__
namespace LFL {

struct Count {
  int total, good, bad;
  Count() : total(0), good(0), bad(0) {}
  bool Add(bool which) { if (which) good++; else bad++; total++; return which; }
  Count& operator+=(const Count& x) { total+=x.total; good+=x.good; bad+=x.bad; return *this; }

  static void Print(const Count *c, const char *name, const char *good="good", const char *bad="bad") {
    string out = StringPrintf("%d\t%s,\t%d %s", c->total, name, c->good, good);
    if (bad) StrAppend(&out, ", ", c->bad, " ", bad);
    StrAppend(&out, " (", (float)c->good / c->total, ")");
    if (bad) StrAppend(&out, " (weighted = ", (float)(c->good - c->bad) / c->total, ")");
    printf("%s\n", out.c_str());
  }
};

struct Counter {
  typedef map<int, int> Count;
  Count count;
  int incrs, seen;

  Counter() : incrs(0), seen(0) {}
  Counter(int n) : incrs(0), seen(0) { Incr(n); }
  Counter(const Counter& copy) : count(copy.count), incrs(copy.incrs), seen(copy.seen) {}

  void Incr(int n) {
    incrs++;

    Count::iterator i = count.find(n);
    if (i != count.end()) { (*i).second++; return; }
    else count[n] = 1;
  }

  int Get(int n) const {
    Count::const_iterator i = count.find(n);
    return i == count.end() ? 0 : (*i).second;
  }

  int Best() const {
    int max=0, ret=0;
    for (Count::const_iterator i = count.begin(); i != count.end(); i++)
      if ((*i).second > max) { max = (*i).second; ret = (*i).first; }
    return ret;
  }
};

struct CounterSquared : public Counter {
  typedef map<int, Counter> Count2;
  Count2 count2;

  CounterSquared() {}
  CounterSquared(int n) : Counter(n) {}
  CounterSquared(const CounterSquared& copy) : Counter(copy), count2(copy.count2) {}
};

struct CounterS {
  typedef map<string, int> Count;
  Count count;
  int incrs, seen;

  CounterS() : incrs(0), seen(0) {}
  CounterS(string n) : incrs(0), seen(0) { Incr(n); }
  CounterS(const CounterS& copy) : count(copy.count), incrs(copy.incrs), seen(copy.seen) {}

  void Incr(string n) {
    incrs++;

    Count::iterator i = count.find(n);
    if (i != count.end()) { (*i).second++; return; }
    else count[n] = 1;
  }

  int Get(string n) const {
    Count::const_iterator i = count.find(n);
    return i == count.end() ? 0 : (*i).second;
  }
};

}; // namespace LFL
#endif // LFL_ML_COUNTER_H__
