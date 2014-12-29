/*
 * $Id: forvo.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef __LFL_CRAWLER_FORVO_H__
#define __LFL_CRAWLER_FORVO_H__
namespace LFL {

struct ForvoApi : public Crawler, public HTMLParser {
    static const char *LANGUAGE() { return "en"; }
    static const char *KEY() { return "f3d32ed96bfba6d8ed53600e03e96d4e"; }
    static string URL(const char *query) {
        return StringPrintf("http://apifree.forvo.com/key/%s/format/xml/action/word-pronunciations/word/%s/language/%s/order/rate-desc/",
                            KEY(), tolower(query).c_str(), LANGUAGE());
    }
    static string queryFromURL(const char *url) {
        const char *word_begin, *word_end, word_prefix[] = "/word-pronunciations/word/";
        if (!(word_begin = strstr(url, word_prefix))) return "";
        word_begin += strlen(word_prefix);
        if (!(word_end = strchr(word_begin, '/'))) return "";
        return string(word_begin, word_end-word_begin);
    }

    string text;
    void clear() { text.clear(); }

    vector<string> scraped;

    void init(const char *filename) {
        LocalFile file(filename, "r");
        if (!file.Opened()) { ERROR("Open: ", filename); return; }

        for (const char *line = file.NextLine(); line; line = file.NextLine()) 
            queue[0].add(URL(line));
    }

    void validate() {
        if (queue.size() != 2) FATAL("invalid queue size ", queue.size());
        queue[1].scrape = false; /* let crawler know when to exit */
    }

    void crawl() {
        if (queue[1].outstanding) return;
        Crawler::crawl(1);
        if (queue[1].outstanding) return;

        bool q0_maxed = queue[0].completed + queue[0].outstanding >= FLAGS_q0max_completed;
        if (q0_maxed || queue[0].outstanding) return;
        Crawler::crawl(0);
    }

    bool scrape(int qf, const CrawlFileEntry *entry) {
        clear();
        HTMLParser::Parse(entry->content().c_str(), entry->content().size());
        if (!scraped.size()) return true;

        for (int i=0; i<scraped.size(); i++) queue[1].add(scraped[i].c_str());
        scraped.clear();
        return true;
    }

    static bool filter(const TagStack &stack) { return !MatchStack(stack, 1, "pathmp3"); }

    virtual void Text(const string &content, const TagStack &stack) {
        if (filter(stack)) return;
        text += content;
    }

    virtual void CloseTag(const string &tag, const KV &attr, const TagStack &stack) {
        if (filter(stack)) return;
        string url = HTTP::encodeURL(text.c_str());
        INFO("pathmp3='", url, "'");
        scraped.push_back(url.c_str());
        clear();
    }

    void dump(const char *dir) {
        float totaltime = 0;

        typedef map<string, long long> mp3map;
        mp3map mp3;
        QueueFileEntry hdr;
        while (queue[1].in->Next(&hdr, 0, QueueFileEntry::CRAWLED)) 
            mp3[hdr.url().c_str()] = hdr.offset();

        CrawlFileEntry entry; int offset;
        while (queue[0].out->Next(&entry, &offset, QueueFileEntry::SCRAPED)) {
            HTMLParser::Parse(entry.content().data(), entry.content().size());
            if (!scraped.size()) continue;

            string word = queryFromURL(hdr.url().c_str());
            if (!word.size()) continue;

            for (int i=0; i<scraped.size(); i++) {
                mp3map::iterator iter = mp3.find(scraped[i]);
                if (iter == mp3.end()) continue;
                long long q1_offset = (*iter).second;
                if (!queue[1].out->Get(&entry, q1_offset)) continue;

                SoundAsset sa;
                int len = entry.content().size(), max_seconds = 60;
                entry.mutable_content()->insert(len, SoundAsset::FromBufPad, 0);
                sa.Load(entry.content().data(), len, ".mp3", max_seconds);
                if (!sa.wav) {
                    INFO("no wav for ", scraped[i]);
                    sa.Unload();
                    continue;
                }
                totaltime += (float)sa.wav->ring.size / sa.wav->samplesPerSec;
                
                if (FLAGS_forvo_dumpmp3) {
                    string fn = StrCat(dir, word, "_", i+1, ".mp3");
                    INFO("Writing mp3 ", fn);
                    LocalFile file(fn, "w");
                    file.Write(entry.content().data(), len);
                }

                if (FLAGS_forvo_dumpwav) {
                    string fn = StrCat(dir, word, "_", i+1, ".wav");
                    INFO("Writing wav ", fn);
                    LocalFile lf(fn, "w");
                    WavWriter wav(&lf);
                    RingBuf::Handle B(sa.wav);
                    wav.Write(&B);

                    fn = fn.substr(0, fn.size()-3) + "txt";
                    LocalFile transcript(fn, "w");
                    transcript.Write(StrCat("0 ", sa.wav->ring.size, " ", word, "\r\n").c_str());
                }

                sa.Unload();
            }
            scraped.clear();
        }
        INFO("extracted ", totaltime, " seconds of audio to ", dir);
    }
};

}; // namespace LFL
#endif // __LFL_CRAWLER_FORVO_H__
