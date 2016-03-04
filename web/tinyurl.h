/*
 * $Id: tinyurl.h 1314 2014-10-16 04:43:45Z justin $
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

#ifndef LFL_CORE_WEB_TINYURL_H__
#define LFL_CORE_WEB_TINYURL_H__
namespace LFL {

struct URLShortener {
  struct Query : public HTMLParser {
    string name, in, out, parsebuf;
    vector<string> subscribers;
    int state;
    bool done;

    Query(const string &N, const string &I, const string &O, const vector<string> &S)
      : name(N), in(I), out(O), subscribers(S), state(0), done(false)
    {
      string url = "http://tinyurl.com/create.php?url=" + HTTP::EncodeURL(in.c_str());
      if (!app->net->http_client->WGet
          (url, 0, HTTPClient::ResponseCB(bind(&HTMLParser::WGetCB, this, _1, _2, _3, _4, _5))))
      { ERROR("wget ", 0); return; }
    }
    virtual void OpenTag(const string &tag, const KV &attr, const TagStack &stack) { parsebuf.clear(); }
    virtual void Text(const string &text, const TagStack &stack) { parsebuf += text; }
    virtual void CloseTag(const string &tag, const KV &attr, const TagStack &stack) {
      if (parsebuf == "TinyURL was created!") state = 1;
      else if (state > 0 && state < 3 && parsebuf.substr(0,7) == "http://") {
        if (state == 1) out += " " + parsebuf;
        state++;
      }
    }
    virtual void WGetContentEnd(Connection*) { done = true; }
  };

  typedef set<Query*> QuerySet;
  QuerySet queries;
  URLShortener() {}

  void Shorten(const string &name, const string &url, const string &out, const vector<string> &subscribers) { 
    queries.insert(new Query(name, url, out, subscribers));
  }

  void Update(vector<Query*> *out) {
    if (!out) return;
    out->clear();
    for (auto it = queries.begin(); it != queries.end(); it++) if ((*it)->done) out->push_back(*it);
    for (auto it = out->begin(); it != out->end(); it++) queries.erase(*it);
  }
};

}; // namespace LFL
#endif // LFL_CORE_WEB_TINYURL_H__
