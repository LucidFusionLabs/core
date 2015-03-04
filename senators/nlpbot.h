/*
 * $Id: nlpbot.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef __LFL_SENATORS_NLPBOT_H__
#define __LFL_SENATORS_NLPBOT_H__
namespace LFL {

struct NLPBot : public Bot {
    string name;
    // Parser parser;
    RSSMonitor rss;
    URLShortener shortener;

    // ::Parse history;
    typedef map<string, deque<string>*> SayLog;
    SayLog saylog;

    NLPBot(bool print_parse) {} // : parser(FLAGS_nlp_modeldir.c_str(), print_parse) {}
    virtual ~NLPBot() { for (auto i = saylog.begin(); i != saylog.end(); i++) delete i->second; }

    struct UserSaid { // : public NLP::SentenceHistory {
        deque<string> *wrap=0;
        int histind=-1;
#if 0
        ::Parser *parser;
        ::Parse *history;
        UserSaid(deque<string> *W, Parser *P, ::Parse *H) : wrap(W), /*parser(P), history(H),*/ histind(-1) { Load(0); }
#endif
        int Size() { return wrap->size(); }
        const string* Text(size_t ind) {
            size_t bounded_ind = wrap->size()-1 - max(size_t(0), min(ind, wrap->size()-1));
            return &(*wrap)[bounded_ind];
        }
#if 0
        void Load(int ind) {
            if (!ind) parser->Parse(this, Text(0)->c_str(), &parser->parse);
            else {
                histind = ind;
                parser->Parse(Text(histind)->c_str(), history);
            }
        }
        ::Parse* parse(int ind) { if (ind && ind != histind) Load(ind); return ind ? history : &parser->parse; }
        const NLP::Sentence *Sentence(int ind) { return &parse(ind)->sentence; }
        const NLP::PhraseTree   *Tree(int ind) { return &parse(ind)->tree; }
        NLP::PhraseTree         *TreeZero()    { return &parse(0)->tree; }
#endif
    };
    bool SayLast(const string &user, UserSaid *out) {
        deque<string> *query_say = FindOrNull(saylog, user);
        if (!query_say || !query_say->size()) return false;
        // *out = UserSaid(query_say, &parser, &history);
        return true;
    }

    virtual void Connected(BotServer *server, Connection *c, const string& nick) { name = nick; }
    virtual void Chat(BotServer *server, Connection *c, const string &source, const string &target, const string &text) {
        StringWordIter words(text.c_str());
        string cmd = BlankNull(words.Next());
        string arg1star = words.offset >= 0 ? text.substr(words.offset) : "";
        string arg1 = BlankNull(words.Next());
        string arg2star = words.offset >= 0 ? text.substr(words.offset) : "";
        string arg2 = BlankNull(words.Next());
        int a2 = atoi(arg2.c_str());
        // const ::Parse &parse = parser.parse;
        UserSaid said;

        // Handle private messages
        if (target == name) {
            if (cmd == "!join" && arg1.size()) {
                server->Join(c, arg1);
            }
            else if (cmd == "!part" && arg1.size()) {
                server->Part(c, arg1);
            }
        }

        if (!target.size() || target[0] != '#') return;

        // Handle public messages
        if (cmd == "!last") {
            if (SayLast(arg1, &said)) server->Say(c, source, target, *said.Text(a2));
        }
#if 0
        else if (cmd == "!tag") {
            if (SayLast(arg1, &said)) SayLines(server, c, source, target, said.Sentence(a2)->pos_str(Singleton<Pos::NameCB>::Get()));
        }
        else if (cmd == "!chunk") {
            if (SayLast(arg1, &said)) SayLines(server, c, source, target, said.parse(a2)->chunkstr);
        }
        else if (cmd == "!parse") {
            if (SayLast(arg1, &said)) SayLines(server, c, source, target, said.Tree(a2)->tostr());
        }
        else if (cmd == "!srl") {
            if (SayLast(arg1, &said)) SayLines(server, c, source, target, said.Tree(a2)->tostr(true));
        }
        else if (cmd == "!heads") {
            if (SayLast(arg1, &said)) SayLines(server, c, source, target, said.Tree(a2)->tostr(false, true));
        }
        else if (cmd == "!pnr") {
            if (SayLast(arg1, &said)) SayLines(server, c, source, target,
                                               NLP::PrintAntecedents(&said, said.parse(0)->antecedents, "pronoun"));
        }
#endif
        else if (cmd == "!re") {
            if (SayLast(source, &said)) {
                vector<Regex::Result> match;
                const string &text = *said.Text(0);
                Regex(arg1star).Match(text, &match);
                for (int i=0; i<match.size(); i++) {
                    server->Say(c, source, target, text.substr(match[i].begin, match[i].end-match[i].begin));
                }
            }
        }
        else if (cmd == "!rss") {
            string url;
            if      (arg1 == "sciencedaily") url = "http://feeds.sciencedaily.com/sciencedaily?format=xml";
            else if (arg1 == "bbcscienv") url = "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml";
            // if (arg1 == "newscientist") url = "http://feeds.newscientist.com/online-news";
            else if (arg1 == "nprscience") url = "http://www.npr.org/rss/rss.php?id=1007";
            else if (arg1 == "nytscience") url = "http://feeds.nytimes.com/nyt/rss/Science";
            else if (arg1 == "nytspace") url = "http://feeds.nytimes.com/nyt/rss/Space";
            if (!url.empty()) {
                rss.Subscribe(arg1, url, target);
                server->Say(c, "!rss", target, url);
            }
        }
        else {
            static const int source_say_max = 256;
            deque<string> *source_say = FindOrInsert(saylog, source)->second;
            bool source_say_full = source_say->size() == source_say_max;
            if (source_say_full) source_say->pop_front();
            source_say->push_back(text);
        }
    }
    virtual void Heartbeat(BotServer *server, Connection *c) {
        vector<RSSMonitor::Feed::Item> items;
        rss.Update(&items);
        for (int i=0, l=items.size(); i<l; i++) {
            const RSSMonitor::Feed::Item &item = items[i];
            shortener.Shorten(item.feed->name, item.link, item.title, item.feed->subscribers);
        }

        vector<URLShortener::Query*> urls;
        shortener.Update(&urls);
        for (int i=0, l=urls.size(); i<l; i++) {
            URLShortener::Query *u = urls[i];
            for (int j=0, l2=u->subscribers.size(); j<l2; j++) {
                SayLines(server, c, u->name, u->subscribers[j], u->out);
            }
            delete u;
        }
    }

    void SayLines(BotServer *server, Connection *c, const string &source, const string &target, const string &text) {
        StringLineIter parse_lines(text.c_str());
        for (const char *line = parse_lines.Next(); line; line = parse_lines.Next()) server->Say(c, source, target, line);
    }
};

}; // namespace LFL
#endif // __LFL_SENATORS_NLPBOT_H__
