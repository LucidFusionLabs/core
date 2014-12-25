/*
 * $Id: crawler.h 1314 2014-10-16 04:43:45Z justin $
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

#ifndef __LFL_CRAWLER_CRAWLER_H__
#define __LFL_CRAWLER_CRAWLER_H__
namespace LFL {

struct Crawler {
    virtual void validate() { CHECK(queue.size()); }
    virtual void crawl() { Crawler::crawl(0); }
    virtual void scrape() { Crawler::scrape(0); }
    virtual bool scrape(int queue, const CrawlFileEntry *entry) = 0;

    struct Queue {
        ProtoFile *in, *out;
        bool crawl, scrape;
        int outstanding, completed, scraped;
        Queue(ProtoFile *In, ProtoFile *Out) : in(In), out(Out), crawl(1), scrape(1), outstanding(0), completed(0), scraped(0) {}

        bool add(const string &url) {
            QueueFileEntry entry;
            entry.set_url(url.c_str());
            entry.set_flag(0);
            entry.set_mimetype(0);
            entry.set_offset(0);
            entry.set_created(Now());
            return in->Add(&entry, QueueFileEntry::QUEUED);
        }
    }; 
    vector<Queue> queue;

    virtual ~Crawler() {
        for (int i=0; i<queue.size(); i++) {
            delete queue[i].in;
            delete queue[i].out;
        }
    }

    bool crawl_done(int above=0) {
        int i = above;
        for (/**/; i < queue.size(); i++) {
            if (!queue[i].crawl) continue;
            if (!queue[i].in->done || queue[i].outstanding) break;
        }
        return i == queue.size();
    }

    bool scrape_done(int above=0) {
        int i = above;
        for (/**/; i < queue.size(); i++) {
            if (!queue[i].scrape) continue;
            if (!queue[i].out->done) break;
        }
        return i == queue.size();
    }

    int outstanding(int above=0) {
        int ret = 0;
        for (int i=above; i < queue.size(); i++) ret += queue[i].outstanding;
        return ret;
    }

    bool add(const char *qfn, const char *cfn) {
        ProtoFile *qf = new ProtoFile(qfn);
        if (!qf->file->opened()) { ERROR("crawler input open: ", qfn); delete qf; return false; }

        ProtoFile *cf = new ProtoFile(cfn);
        if (!cf->file->opened()) { ERROR("crawler output open: ", cfn); delete cf; return false; }

        queue.push_back(Queue(qf, cf));
        return true;
    }

    struct FetchBuffer {
        Crawler *parent;
        int queue, qf_offset, content_length;

        ProtoHeader hdr;
        QueueFileEntry request;
        string headers, content;

        FetchBuffer(Crawler *P, int Q) : parent(P), queue(Q), qf_offset(-1), content_length(0)  {}
    };

    bool scrape(int ind) {
        CrawlFileEntry entry; int offset, ret;
        if (!queue[ind].out->Next(&entry, &offset, QueueFileEntry::CRAWLED)) return false;
        ret = scrape(ind, &entry);
        INFO("scrape(", ind, ") url='", entry.request().url().c_str(), "' ", ret);
        if (!ret) return false;

        if (!queue[ind].out->Update(offset, QueueFileEntry::SCRAPED)) return false;
        queue[ind].scraped++;
        return true;
    }

    bool crawl(int ind) {
        FetchBuffer *next = new FetchBuffer(this, ind);
        if (!queue[ind].in->Next(&next->hdr, &next->request, &next->qf_offset, QueueFileEntry::QUEUED)) { delete next; return false; }
        INFO("crawl(", ind, ") oustanding=", queue[ind].outstanding+1, " url='", next->request.url().c_str(), "'");

        if (!Singleton<HTTPClient>::Get()->WGet
            (next->request.url(), 0,
             HTTPClient::ResponseCB(bind(&Crawler::WGetResponseCB, this, _1, _2, _3, _4, _5, next))))
        { delete next; return false; }

        queue[ind].outstanding++;
        return true;
    }

    void crawled(FetchBuffer *buf) {
        CrawlFileEntry entry;
        entry.mutable_request()->CopyFrom(buf->request);
        entry.set_headers(buf->headers.data(), buf->headers.size());
        entry.set_content(buf->content.data(), buf->content.size());

        int new_status = QueueFileEntry::CRAWLED;
        if (!queue[buf->queue].out->Add(&entry, new_status)) return;

        buf->hdr.SetFlag(new_status);
        buf->request.set_offset(queue[buf->queue].out->write_offset);
        queue[buf->queue].in->Update(buf->qf_offset, &buf->hdr, &buf->request);
    }

    void close(FetchBuffer *buf) {
        queue[buf->queue].outstanding--;
        queue[buf->queue].completed++;
        INFO("out(", buf->queue, ") outstanding=", queue[buf->queue].outstanding, " completed=", queue[buf->queue].completed, " url='", buf->request.url(), "'");
        delete buf;
    }

    void WGetResponseCB(Connection *c, const char *headers, const string &ct,
                        const char *content, int len, FetchBuffer *buf)
    {
        if (headers) { /* headers only = init */
            buf->headers = headers;
            buf->content_length = len;
            if (buf->content_length) buf->content.reserve(buf->content_length);
        } else if (content) { /* content only = data chunk */
            buf->content.append(content, len);
        } else { /* neither headers or content = final */
            bool success = true;
            if (buf->content_length && buf->content_length != buf->content.size()) success = false;
            if (success) crawled(buf);
            close(buf);
        }
    }
};

}; // namespace LFL
#endif // __LFL_CRAWLER_CRAWLER_H__
