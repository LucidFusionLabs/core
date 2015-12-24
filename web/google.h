/*
 * $Id: google.h 1306 2014-09-04 07:13:16Z justin $
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

#ifndef LFL_WEB_GOOGLE_H__
#define LFL_WEB_GOOGLE_H__
namespace LFL {

struct GoogleApi : public Crawler, public DebugHTMLParser {
  static string URL(const char *query) {
    return StringPrintf("http://www.google.com/search?q=%s&amp;ie=utf-8&as_qdr=all&amp;aq=t&amp;rls=org:mozilla:us:official&amp;client=firefox&num=100", tolower(query).c_str());
  }

  struct Result {
    string title, url;
    Result() {}
    Result(const Result &in) : title(in.title), url(in.url) {}
    void Clear() { title.clear(); url.clear(); }
  };
  typedef vector<Result> Results;

  Results results;
  Result builder;
  GoogleApi() {}

  void Clear() { results.clear(); builder.Clear(); }

  void Init(const char *query) { queue[0].Add(URL(query)); }

  void Validate() {
    if (queue.size() != 2) FATAL("invalid queue size ", queue.size());
    queue[1].scrape = false; /* disable scraping on 2nd queue */
  }

  void Crawl() {
    if (queue[1].outstanding < FLAGS_q1max_outstanding) 
      Crawler::Crawl(1);

    bool q0_maxed = queue[0].completed + queue[0].outstanding >= FLAGS_q0max_completed;
    if (q0_maxed || queue[0].outstanding) return;
    Crawler::Crawl(0);
  }

  bool Scrape(int qf, const CrawlFileEntry *entry) {
    if (strstr(entry->content().c_str(), "computer virus or spyware application") ||
        strstr(entry->content().c_str(), "entire network is affected") ||
        strstr(entry->content().c_str(), "http://www.download.com/Antivirus"))
    { ERROR("the world's biggest scraper is upset about being scraped, shutting down, content='", entry->content().c_str(), "'"); app->run=0; return true; }

    Clear();
    HTMLParser::Parse(entry->content().data(), entry->content().size());

    for (int i=0; i<results.size(); i++) {
      queue[1].Add(results[i].url.c_str());
      INFO("RESULT='", results[i].title, "' URL='", results[i].url, "'");
    }
    return true;
  }

  static bool Filter(const TagStack &stack) {
    if (!MatchStack(stack, 2, "a", "h3") &&
        !MatchStack(stack, 3, "b", "a", "h3") &&
        !MatchStack(stack, 3, "em", "a", "h3"))
      return true;

    if (!MatchStackAttr(stack, "div", "id", 2, "ires", "res"))
      return true;

    if (!MatchStackAttr(stack, "h3", "class", 1, "r"))
      return true;

    return false;
  }

  virtual void OpenTag(const string &tag, const KV &attr, const TagStack &stack) {
    if (Filter(stack)) return;
    if (tag != "a") return;

    KV::const_iterator href = attr.find("href");
    if (href != attr.end())
      builder.url = (*href).second;
  }

  virtual void Text(const string &text, const TagStack &stack) {
    if (Filter(stack)) return;
    builder.title += text;
  }

  virtual void CloseTag(const string &tag, const KV &attr, const TagStack &stack) {
    if (Filter(stack)) return;
    if (tag != "a") return;

    if (builder.url.substr(0,5) == "http:") {
      builder.url = HTTP::EncodeURL(builder.url.c_str());
      results.push_back(builder);
    }

    builder.Clear();
  }
};

}; // namespace LFL
#endif // LFL_WEB_GOOGLE_H__
