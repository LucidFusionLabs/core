/*
 * $Id: resolver.cpp 1314 2014-10-16 04:43:45Z justin $
 */

#include "lfapp/lfapp.h"
#include "lfapp/network.h"
#include "lfapp/dom.h"
#include "crawler/html.h"

using namespace LFL;

DEFINE_int   (gui_port,            0,             "GUI Port");

DEFINE_string(resolve,             "",            "Retrieve MX and A for all domains in file");
DEFINE_string(ip_address,          "",            "Resolve from comma separated IP list; blank for all");

DECLARE_int  (target_fps);         //             Target_resolve_per_second = target_fps * frame_resolve_max 
DEFINE_int   (frame_resolve_max,   10,            "max resolve per frame");

struct BulkResolver {
    File *out; RecursiveResolver *rr; int min_rr_completed;
    BulkResolver() : out(0), rr(0), min_rr_completed(0) {}
    void OpenLog(const string &fn) {
        if (LocalFile(fn, "r").Opened()) FATAL(fn, " already exists");
        out = new LocalFile(fn, "w");
    }

    struct Query {
        bool Adone;
        string domain;
        DNS::Response A, MX;
        BulkResolver *parent;
        Query(const string &q, BulkResolver *p) : Adone(0), domain(q), parent(p) {}

        void Run() {
            RecursiveResolver::Request *req = new RecursiveResolver::Request
                (domain, Adone ? DNS::Type::MX : DNS::Type::A, 
                 Resolver::ResponseCB(bind(&BulkResolver::Query::ResponseCB, this, _1, _2)));
            INFO("BulkResolver Run domain=", domain, ", type=", req->type);
            parent->rr->Resolve(req);
        }
        void Output() {
            set<IPV4::Addr> Aa;
            for (int i = 0; i < A.A.size(); i++) Aa.insert(A.A[i].addr);
            string ret = StrCat("A=", domain, ":", IPV4::MakeCSV(Aa));
            DNS::AnswerMap MXe;
            DNS::MakeAnswerMap(MX.E, &MXe);
            map<int, pair<string, string> > MXa;
            for (int i = 0; i < MX.A.size(); ++i) {
                const DNS::Record &a = MX.A[i];
                if (a.type != DNS::Type::MX) continue;
                DNS::AnswerMap::const_iterator e_iter = MXe.find(a.answer);
                if (a.question.empty() || a.answer.empty() || e_iter == MXe.end()) { ERROR("missing ", a.answer); continue; }
                MXa[a.pref] = pair<string, string>(e_iter->first, IPV4::MakeCSV(e_iter->second));
            }
            for (map<int, pair<string, string> >::iterator i = MXa.begin(); i != MXa.end(); ++i) {
                string hn = i->second.first;
                if (SuffixMatch(hn, ".")) hn.erase(hn.size()-1);
                StrAppend(&ret, "; MX", i->first, "=", hn, ":", i->second.second);
            }
            ret += "\n";
            if (parent->out) parent->out->Write(ret);
        }
        void ResponseCB(IPV4::Addr addr, DNS::Response *res) {
            bool resolved = (addr != -1 && res);
            if (!resolved) ERROR("failed to resolve: ", domain, " type=", Adone ? "MX" : "A");
            if (resolved && !Adone) A  = *res;
            if (resolved &&  Adone) MX = *res;
            if (Adone) Output();
            else { Adone=1; Run(); }
        }
    };
    vector<Query*> queue, done;

    void AddQueriesFromFile(const string &fn) {
        int start_size = queue.size();
        LocalFile file(fn, "r");
        for (const char *line = file.NextLine(); line; line = file.NextLine()) {
            queue.push_back(new Query(tolower(line), this)); 
        }
        INFO("Added ", queue.size() - start_size, " from ", fn); 
    }

    void Frame() {
        Service *udp_client = Singleton<UDPClient>::Get();
        if (!queue.size() || rr->queries_completed < min_rr_completed) return;
        for (int i = 0; i < FLAGS_frame_resolve_max && queue.size() && udp_client->connect_src_pool->Available(); i++) {
            Query *q = queue.back();
            queue.pop_back();
            done.push_back(q);
            q->Run();
        }
    }

    string StatsLine() const { return rr ? StrCat(", RR=", rr->queries_completed, "/", rr->queries_requested) : ""; }

} bulk_resolver;

int frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    if (!FLAGS_resolve.empty()) bulk_resolver.Frame();
    
    char buf[256];
    if (FGets(buf, sizeof(buf))) ERROR("FPS=", FPS(), bulk_resolver.StatsLine());
    return 0;
}

struct StatusGUI : public HTTPServer::Resource {
    HTTPServer::Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
        return HTTPServer::Response("text/html; charset=UTF-8", "<html><h>Blaster</h><p>Version 1.0</p></html>\n");
    }
};

extern "C" {
int main(int argc, const char **argv) {
    screen->frame_cb = frame;
    app->logfilename = StrCat(LFAppDownloadDir(), "resolver.txt");
    FLAGS_max_rlimit_core = FLAGS_max_rlimit_open_files = 1;
    FLAGS_lfapp_network = 1;

    if (app->Create(argc, argv, __FILE__)) { ERROR("lfapp init failed: ", strerror(errno)); return app->Free(); }
    if (app->Init())                       { ERROR("lfapp open failed: ", strerror(errno)); return app->Free(); }

    HTTPServer httpd(FLAGS_gui_port, false);
    if (FLAGS_gui_port) {
        httpd.AddURL("/", new StatusGUI());
        if (app->network.Enable(&httpd)) return -1;
    }
    
    if (FLAGS_ip_address.empty()) {
        set<IPV4::Addr> ips;
        Sniffer::GetDeviceAddressSet(&ips);
        Singleton<FlagMap>::Get()->Set("ip_address", IPV4::MakeCSV(ips));
    }

    if (!FLAGS_resolve.empty()) {
        Singleton<UDPClient>::Get()->connect_src_pool = new IPV4EndpointPool(FLAGS_ip_address);
        RecursiveResolver *RR = Singleton<RecursiveResolver>::Get();
        RR->Resolve(new RecursiveResolver::Request("com"));
        RR->Resolve(new RecursiveResolver::Request("net"));
        RR->Resolve(new RecursiveResolver::Request("org"));
        bulk_resolver.min_rr_completed = 3;
        bulk_resolver.rr = Singleton<RecursiveResolver>::Get();
        bulk_resolver.OpenLog("resolve.out.txt");
        bulk_resolver.AddQueriesFromFile(FLAGS_resolve);
    }

    if (!FLAGS_gui_port && FLAGS_resolve.empty()) { INFO("nothing to do"); return 0; }

    int ret = app->Main();
    ERROR("PerformanceTimers: ", Singleton<PerformanceTimers>::Get()->DebugString());
    return ret;
}
}
