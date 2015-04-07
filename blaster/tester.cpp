/*
 * $Id: tester.cpp 1325 2014-10-29 06:08:31Z justin $
 */

#include "lfapp/lfapp.h"
#include "lfapp/network.h"
#include "lfapp/dom.h"
#include "crawler/html.h"

using namespace LFL;

DEFINE_int   (gui_port,            0,             "GUI Port");

DEFINE_string(wget,                "",            "WGet url");
DEFINE_string(nslookup,            "",            "Nslookup host");
DEFINE_string(rnslookup,           "",            "Recusively-resolve host (ie without using /etc/resolv.conf nameservers)");
DEFINE_bool  (nslookup_mx,         0,             "Lookup MX rather than A for -nslookup and -rnslookup");
DEFINE_bool  (print_iface_ips,     0,             "Print interface IP addresses");

DEFINE_string(smtp_test,           "",            "SMTP Test Role: [server,client]");
DEFINE_string(smtp_test_server_ip, "",            "Listen  on   1025-65535 of comma separated IP list");
DEFINE_string(smtp_test_client_ip, "",            "Connect from 1025-65535 of comma separated IP list");
DEFINE_int   (smtp_test_port_max,  65536,         "Max port for SMTP Test");
DEFINE_string(domain,              "",            "SMTP server domain");
                                                  
DECLARE_int  (target_fps);         //             Target_connects_per_second = target_fps * frame_connect_max 
DEFINE_int   (frame_connect_max,   10,            "max connect() per frame");

SMTPServer smtp_test_server(FLAGS_domain);
SMTPClient smtp_test_client;

struct SMTPTest {
    int test_size; Service *svc; const long long *test_connected;
    SMTPTest() : test_size(0), svc(0) { static long long zero=0; test_connected = &zero; }
    void Start() {
        ERROR("SMTPTest::Start() test_size=", test_size);
        if      (FLAGS_smtp_test == "server") { svc = &smtp_test_server; test_connected = &smtp_test_server.total_connected; }
        else if (FLAGS_smtp_test == "client") { svc = &smtp_test_client; test_connected = &smtp_test_client.total_connected; }
    }

    void Frame() {
        if (*test_connected == test_size) ERROR("SMTPTest(): ", test_size, " connected");
    }
    
    string StatsLine() const { return StrCat(", connected=", *test_connected); }

} smtp_tester;

vector<Callback> connect_queue;

int frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    if (!FLAGS_smtp_test.empty()) smtp_tester.Frame();
    if (connect_queue.size()) {
        for (int i = 0; connect_queue.size() && i < FLAGS_frame_connect_max; ++i) {
            connect_queue.back()();
            connect_queue.pop_back();
        }
    }
    
    char buf[256]; if (FGets(buf, sizeof(buf))) ERROR("FPS=", FPS(), smtp_tester.StatsLine());
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
    app->logfilename = StrCat(LFAppDownloadDir(), "tester.txt");
    FLAGS_max_rlimit_core = FLAGS_max_rlimit_open_files = 1;
    FLAGS_lfapp_network = 1;

    if (app->Create(argc, argv, __FILE__)) { ERROR("lfapp init failed: ", strerror(errno)); return app->Free(); }
    if (app->Init())                       { ERROR("lfapp open failed: ", strerror(errno)); return app->Free(); }

    HTTPServer httpd(FLAGS_gui_port, false);
    if (FLAGS_gui_port) {
        httpd.AddURL("/", new StatusGUI());
        if (app->network.Enable(&httpd)) return -1;
    }

    if (!FLAGS_wget.empty()) Singleton<HTTPClient>::Get()->WGet(FLAGS_wget);
    if (!FLAGS_nslookup .empty()) { FLAGS_dns_dump=1; Singleton<Resolver>         ::Get()->Resolve(             Resolver::Request(FLAGS_nslookup,  FLAGS_nslookup_mx ? DNS::Type::MX : DNS::Type::A)); }
    if (!FLAGS_rnslookup.empty()) { FLAGS_dns_dump=1; Singleton<RecursiveResolver>::Get()->Resolve(new RecursiveResolver::Request(FLAGS_rnslookup, FLAGS_nslookup_mx ? DNS::Type::MX : DNS::Type::A)); }
    if (FLAGS_print_iface_ips) {
        set<IPV4::Addr> ips; string text;
        Sniffer::GetDeviceAddressSet(&ips);
        INFO("Available IP Addresses = [ ", IPV4::MakeCSV(ips), " ]");
    }

    if (!FLAGS_smtp_test.empty() && (!FLAGS_smtp_test_server_ip.empty() || !FLAGS_smtp_test_client_ip.empty())) {
        vector<IPV4::Addr> listen_addrs, connect_addrs;
        IPV4::ParseCSV(FLAGS_smtp_test_server_ip, &listen_addrs);
        IPV4::ParseCSV(FLAGS_smtp_test_client_ip, &connect_addrs);
        if (FLAGS_smtp_test_client_ip.size()) CHECK_EQ(listen_addrs.size(), connect_addrs.size());

        for (int i = 0; i < listen_addrs.size(); i++) {
            for (int port = 1025; port < FLAGS_smtp_test_port_max; port++) {
                if (port >= 8080 && port < 8090) continue;
                if      (FLAGS_smtp_test == "client") smtp_test_server.QueueListen(listen_addrs[i], port);
                else if (FLAGS_smtp_test == "server") connect_queue.push_back(bind([=](){ smtp_test_client.Connect(listen_addrs[i], port, connect_addrs[i], port); }));
                else FATAL("unknown smtp_test: ", FLAGS_smtp_test);
                smtp_tester.test_size++;
            }
        }
        smtp_tester.Start();
        if (app->network.Enable(smtp_tester.svc)) return -1;
    }

    int ret = app->Main();
    ERROR("PerformanceTimers: ", Singleton<PerformanceTimers>::Get()->DebugString());
    return ret;
}
}
