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

#ifndef LFL_CORE_WEB_GOOGLE_CHART_H__
#define LFL_CORE_WEB_GOOGLE_CHART_H__
namespace LFL {

struct DeltaSampler {
  typedef long long Value;
  struct Entry : public vector<Value> {
    void Assign(const vector<const Value*> &in) { resize(in.size()); for (int i=0; i<size(); i++) (*this)[i] = *in[i]; }
    void Subtract(const Entry &e) {      CHECK_EQ(size(), e.size()); for (int i=0; i<size(); i++) (*this)[i] -=  e[i]; }
    void Divide  (float v)        {                                  for (int i=0; i<size(); i++) (*this)[i] = static_cast<long long>((*this)[i] / v); }
  };

  Timer                timer;
  int                  interval;
  vector<string>       label;
  vector<const Value*> input;
  vector<Entry>        data;
  Entry                cur;

  DeltaSampler(int I, const vector<const Value*> &in, const vector<string> &l) :
    interval(I), label(l), input(in) { cur.Assign(input); }

  void Update() {
    if (timer.GetTime() < Time(interval)) return;
    float buckets = (float)timer.GetTime(true).count() / interval;
    if (fabs(buckets - 1.0) > .5) ERROR("buckets = ", buckets);
    Entry last = cur;
    cur.Assign(input);
    Entry diff = cur;
    diff.Subtract(last);
    data.push_back(diff);
  }
};

struct DeltaGrapher {
  static void JSTable(const DeltaSampler &sampler, vector<vector<string> > *out, int window_size) {
    vector<string> v; v.push_back("'Time'");
    for (int i=0; i<sampler.label.size(); i++) v.push_back(StrCat("'", sampler.label[i], "'"));
    out->push_back(v);

    v.clear();
    for (int j=0; j<sampler.label.size()+1; j++) v.push_back("0");
    out->push_back(v);

    for (int l=sampler.data.size(), i=max(0,l-window_size); i<l; i++) {
      const DeltaSampler::Entry &le = sampler.data[i];
      v.clear(); v.push_back(StrCat(i+1));
      for (int j=0; j<le.size(); j++) v.push_back(StrCat(le[j]));
      out->push_back(v);
    }
  }
};

struct GChartsHTML {
  static string JSFooter() { return "}\ngoogle.setOnLoadCallback(drawVisualization);\n</script>\n"; }
  static string JSHeader() {
    return
      "<script type=\"text/javascript\" src=\"//www.google.com/jsapi\"></script>\n"
      "<script type=\"text/javascript\">\n"
      "google.load('visualization', '1', {packages: ['corechart']});\n"
      "function drawVisualization() {\n"
      "var data; var ac;\n";
  };
  static string JSAreaChart(const string &div_id, int width, int height,
                            const string &title, const string &vaxis_label, const string &haxis_label,
                            const vector<vector<string> > &table) {
    string ret = "data = google.visualization.arrayToDataTable([\n";
    for (int i = 0; i < table.size(); i++) {
      const vector<string> &l = table[i];
      ret += "[";
      for (int j = 0; j < l.size(); j++) StrAppend(&ret, l[j], ", ");
      ret += "],\n";
    };
    StrAppend(&ret, "]);\nac = new google.visualization.AreaChart(document.getElementById('", div_id, "'));\n");
    StrAppend(&ret, "ac.draw(data, {\ntitle : '", title, "',\n");
    StrAppend(&ret, "isStacked: true,\nwidth: ", width, ",\nheight: ", height, ",\n");
    StrAppend(&ret, "vAxis: {title: \"", vaxis_label, "\"},\nhAxis: {title: \"", haxis_label, "\"}\n");
    StrAppend(&ret, "});\n");
    return ret;
  }
  static string DivElement(const string &div_id, int width, int height) {
    return StrCat("<div id=\"", div_id, "\" style=\"width: ", width, "px; height: ", height, "px;\"></div>");
  }
};

}; // namespace LFL
#endif // LFL_CORE_WEB_GOOGLE_CHART_H__
