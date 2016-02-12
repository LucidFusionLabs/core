/*
 * $Id: rss.h 1314 2014-10-16 04:43:45Z justin $
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

#ifndef LFL_WEB_RSS_H__
#define LFL_WEB_RSS_H__
namespace LFL {

struct RSSMonitor {
  struct Feed : public HTMLParser {
    struct Item {
      Feed *feed; string title, link, description, date;
      void Clear() { feed=0; title.clear(); link.clear(); description.clear(); date.clear(); }
      Time Date() const { return RFC822Date(date.c_str()); }
      bool operator<(const Item &r) const { return Date() > r.Date(); }
    };

    RSSMonitor *monitor;
    string name, url;
    vector<string> subscribers;
    Time updated, lastpubdate;
    bool updating;
    Item parse_item;
    vector<Item> items;

    Feed(RSSMonitor *m, const string &n, const string &u) : monitor(m), name(n), url(u), updated(0), lastpubdate(0), updating(0) {}

    bool Subscriber(const string &n) const {
      for (int i=0, l=subscribers.size(); i<l; i++) if (subscribers[i] == n) return true;
      return false;
    }

    void Update() {
      if (updating) return;
      updated = Now();
      updating = true;
      if (!app->net->http_client->WGet
          (url.c_str(), 0, HTTPClient::ResponseCB(bind(&HTMLParser::WGetCB, this, _1, _2, _3, _4, _5))))
      { ERROR("wget ", 0); return; }
    }

    virtual void OpenTag(const string &tag, const KV &attr, const TagStack &stack) {
      if (tag == "item") parse_item.Clear();
    }

    virtual void Text(const string &text, const TagStack &stack) {
      if (!MatchStackOffset(stack, 1, 1, "item")) return;
      if (MatchStack(stack, 1, "title")) parse_item.title += text;
      else if (MatchStack(stack, 1, "link")) parse_item.link += text;
      else if (MatchStack(stack, 1, "description")) parse_item.description += text;
      else if (MatchStack(stack, 1, "pubdate")) parse_item.date += text;
    }

    virtual void CloseTag(const string &tag, const KV &attr, const TagStack &stack) {
      if (tag == "item" && !parse_item.title.empty() && !parse_item.link.empty()) items.push_back(parse_item);
    }

    virtual void WGetContentEnd(Connection *) {
      lastpubdate=Time(0);
      sort(items.begin(), items.end());
      if (lastpubdate == Time(0) && items.size()) {
        if ((lastpubdate = items.front().Date()) == Time(0)) {
          for (int i=0; i<items.size(); i++) {
            Item &item = items[i];
            ERROR("parse error: { '", item.title, "', '", item.link, "', '", item.date, "', ", item.Date(), " } clearing");
          }
          items.clear();
        }
      }
      while (items.size() && items.back().Date() < lastpubdate) items.pop_back();
      updating = false;
    }
  };

  typedef map<string, Feed*> FeedMap;
  FeedMap feed;
  RSSMonitor() {}

  void Subscribe(const string &name, const string &url, const string &target) {
    Feed *f = FindOrNull(feed, url);
    if (!f) {
      f = new Feed(this, name, url);
      feed[url] = f;
    }
    if (!f->Subscriber(target)) f->subscribers.push_back(target);
    Update(0);
  }

  void Update(vector<Feed::Item> *out) {
    Time now = Now();
    for (FeedMap::iterator it = feed.begin(); it != feed.end(); it++) {
      Feed *f = it->second; int l;
      if (out && !f->updating && (l = f->items.size())) {
        for (int j=0; j<l; j++) {
          out->push_back(f->items[l-j-1]);
          out->back().feed = f;
        }
        f->items.clear();
        f->lastpubdate++;
      }

      if (now < f->updated + Minutes(10)) continue;
      f->Update();
    }
  }
};

}; // namespace LFL
#endif // LFL_WEB_RSS_H__
