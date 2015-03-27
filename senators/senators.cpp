#include "lfapp/lfapp.h"
#include "lfapp/network.h"

#ifdef LFL_OPENSSL
#include "openssl/bio.h"
#include "openssl/ssl.h"
#include "openssl/err.h"
#endif

#include "lfapp/dom.h"
#include "ml/hmm.h"
#include "speech/speech.h"
#include "crawler/html.h"
#include "ml/corpus.h"
#include "ml/counter.h"
#if 0
#include "ml/maxent.h"
#include "nlp/nlp.h"
#include "nlp/tagger.h"
#include "nlp/chunker.h"
#include "nlp/parser.h"
#include "nlp/srl.h"
#include "nlp/anaphora.h"
#include "nlp/trie.h"
#include "nlp/target.h"
#endif
#include "crawler/rss.h"
#include "crawler/tinyurl.h"

namespace LFL {
DEFINE_string(server,       "irc.freenode.net:6667",  "Server to connect to");
DEFINE_bool  (ssl,          false,                    "Connect to server using SSL");
DEFINE_bool  (print,        true,                     "Print input to stdout");
DEFINE_int   (numinit,      1,                        "Number of senators to connect initially");
DEFINE_int   (num,          1,                        "Total number of senators to connect");
DEFINE_int   (secs,         15,                       "Seconds to wait between connecting each senator");
DEFINE_string(prefix,       "privmsg #foo :",         "Prefix for 'say' cmd");
DEFINE_string(message,      "rm -rf /var",            "Default message for empty 'say' cmd");
DEFINE_float (sayfire,      0,                        "Autofire 'say' at 'sayfire' HZ"); 
DEFINE_bool  (colfire,      0,                        "Autofire 'say' with colors or not");
DEFINE_bool  (print_parse,  false,                    "Print parse");
DEFINE_int   (linelen,      110,                      "Read blocksize for SayFile");
DEFINE_string(nlp_modeldir, "../nlp/model/",          "NLP model directory");
DEFINE_string(nomcorpuspath, "../nlp/corpus/nombank/frames/", "Nombank path");

File *SayFile = 0;
vector<string> vprefix;
int vprefix_ind = 0;

vector<string> senatornames;
int senatornames_index = 0;
void load_senatornames() {
#undef XX
#define XX(x) senatornames.push_back(#x);
#include "senators.h"
}

struct Senator;
typedef map<Senator*, Connection*> Senators;
Senators senators;

struct BotServer {
    virtual void Join(Connection *c, const string &target) = 0;
    virtual void Part(Connection *c, const string &target) = 0;
    virtual void Say(Connection *c, const string &source, const string &target, const string &text) = 0;
};

struct Bot {
    virtual void Connected(BotServer *server, Connection *c, const string& nick) = 0;
    virtual void Chat(BotServer *server, Connection *c, const string& source, const string &target, const string &text) = 0;
    virtual void Heartbeat(BotServer *server, Connection *c) = 0;
};

struct IRCBotServer : public BotServer {
    void Join(Connection *c, const string &target) {
        c->write(StrCat("JOIN ", target, "\r\n"));
    }
    void Part(Connection *c, const string &target) {
        c->write(StrCat("PART ", target, "\r\n"));
    }
    void Say(Connection *c, const string &source, const string &target, const string &text) {
        c->write(StrCat("PRIVMSG ", target, " :<", source, "> ", text, "\r\n"));
    }
};

}; // namespace LFL
#include "nlpbot.h"
namespace LFL {

struct Senator : public Query {
    string nick; bool ready; Bot *bot;

    Senator() : ready(0), bot(new NLPBot(FLAGS_print_parse)) {
        if (senatornames.size()) { nick = senatornames[senatornames_index]; senatornames_index = (senatornames_index+1) % senatornames.size(); }
        else { for (int i=0; i<9; i++) nick.append(1, 'a' + ::rand() % 26); }
    }
    int Heartbeat(Connection *c) { if (bot) bot->Heartbeat(Singleton<IRCBotServer>::Get(), c); return 0; }

    int Connected(Connection *c) {
        c->write(StrCat("USER ", nick, " ", nick, " ", nick, " :Senator ", nick, "\r\nNICK ", nick, "\r\n"));
        INFO("Senator ", nick, " connected");
        ready = true;
        if (bot) bot->Connected(Singleton<IRCBotServer>::Get(), c, nick);
        return 0;
    }
    int Read(Connection *c) {
        StringLineIter iter(c->rb, c->rl);
        for (const char *line = iter.Next(); line; line = iter.Next()) {
            if (FLAGS_print) INFO("Senator ", nick, " read '", line, "'");

            if (!strncasecmp(line, "PING ", 5))
                c->write(StrCat("PONG ", line+5, "\r\n"));

            if (bot) do {
                StringWordIter words(line);
                string source = StripColon(BlankNull(words.Next()));
                string server_cmd = BlankNull(words.Next());
                if (server_cmd != "PRIVMSG") break;

                string source_nick = source.substr(0, source.find("!")); 
                string target = BlankNull(words.Next());
                string text = StripColon(words.offset >= 0 ? &line[words.offset] : "");
                bot->Chat(Singleton<IRCBotServer>::Get(), c, source_nick, target, text);
            } while(0);
        }
        c->readflush(c->rl);
        return 0;
    }
    int Closed(Connection *c) {
        INFO("Senator ", nick, " closed");
        Senators::iterator i = senators.find(this);
        if (i != senators.end()) senators.erase(i);
        return 0;
    }

    static string StripColon(const string& s) { return s.substr((s.size() && s[0] == ':')); }
};

void MyNewSenator(const vector<string>&) {
    Connection *c = 0;
#ifndef LFL_OPENSSL
    if (FLAGS_ssl) FATAL("not compiled with -DLFL_SSL see lflpub/CMakeLists.txt");
#else // LFL_OPENSSL
    if (FLAGS_ssl) c = Singleton<HTTPClient>::Get()->SSLConnect(0, FLAGS_server.c_str());
    else
#endif
    c = Singleton<HTTPClient>::Get()->Connect(FLAGS_server.c_str());
    if (!c) FATAL("connect ", FLAGS_server, " failed");
    c->query = new Senator();
    senators[(Senator*)c->query] = c;
}

void DoSendRandom(int times, const string &data, int len) {
    vector<Connection*> majority;
    for (Senators::iterator i = senators.begin(); i != senators.end(); i++)
        if ((*i).first->ready)
            majority.push_back((*i).second);

    for (int i=0; i<times; i++)
        majority[::rand() % majority.size()]->write(data.c_str(), len);

    INFO("sent * 1 * ", times, " '", data, "'");
}

void DoSend(int times, const string &data, int len) {
    for (int i=0; i<times; i++) {
        for (Senators::iterator i = senators.begin(); i != senators.end(); i++)
            if ((*i).first->ready)
                (*i).second->write(data.c_str(), len);
    }
    INFO("sent * ", times, " * ", senators.size(), " '", data, "'");
}

void MySend(const vector<string> &args) {
    float times;
    if (args.size() < 2 || (times = atof(args[0])) <= 0) { INFO("usage: send <times> <text>"); return; }
    string data = Join(args, " ", 1, args.size()) + "\r\n";
    if (times >= 1) DoSend(times, data, data.size());
    else DoSendRandom(times*senators.size(), data, data.size());
}

void MyS(const vector<string> &args) { 
    MySend(vector<string>{1, string(Join(args, " ") + "\r\n")}); 
}

const char *MyPrefix() {
    if (vprefix.size()) {
        vprefix_ind = (vprefix_ind+1) % vprefix.size();
        return vprefix[vprefix_ind].c_str();
    }
    return FLAGS_prefix.c_str();
}

void MySay(const vector<string> &args) {
    string send = StrCat(MyPrefix(), args.size() ? Join(args, " ") : FLAGS_message, "\r\n");
    DoSendRandom(1, send, send.size());
}

void MyColorSay(const vector<string> &args) {
    string in = args.size() ? Join(args, " ") : FLAGS_message, send = MyPrefix();
    for (int i=0, len=in.size(); i<len; i++) StringAppendf(&send, "%02d%c", 1+::rand()%14, in[i]);
    send += "\r\n";
    DoSendRandom(1, send, send.size());
}

void MySayFile(const vector<string> &args) {
    if (!args.size()) return;
    delete SayFile;
    SayFile = new LocalFile(args[0], "r");
    INFO("SayFile(", SayFile, ") = ", args[0], " ", SayFile->Opened());
}

void MyVPrefix(const vector<string> &args) {
    const char *space; int ind;
    if (args.size() < 2 || (ind = atoi(args[0])) < 0 || ind >= vprefix.size()) { INFO("usage: vprefix <ind> <text>"); return; }
    vprefix[ind] = Join(args, " ", 1, args.size());
    INFO("vp[", ind, "] = ", vprefix[ind]);
}

void MyVPrefixSize(const vector<string> &args) {
    if (!args.size()) return;
    vprefix.clear();
    for (int i=0, len=atoi(args[0]); i<len; i++) vprefix.push_back("");
    INFO("vp size = ", atoi(args[0]));
}

int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    for (Senators::iterator i = senators.begin(); i != senators.end(); i++) i->first->Heartbeat(i->second);
    static RollingAvg<unsigned> fps(128);
    fps.Add(clicks);

    static int sayrate_hz = 10;
    static FrameRateLimitter sayrate(&sayrate_hz);
    if (FLAGS_sayfire) {
        sayrate_hz = FLAGS_sayfire;
        sayrate.Limit();

        if (SayFile && SayFile->Opened()) {
            static string sayfilebuf;
            while (sayfilebuf.size() < FLAGS_linelen) {
                const char *line = SayFile->NextLine();
                if (!line) { SayFile->Reset(); line = SayFile->NextLine(); }
                if (!line) { delete SayFile; SayFile = 0; return 0; }
                sayfilebuf.append(line);
                sayfilebuf += " ";
            }
            FLAGS_message = sayfilebuf.substr(0, FLAGS_linelen);
            sayfilebuf = sayfilebuf.substr(FLAGS_linelen);
        }
        if (FLAGS_colfire) MyColorSay(vector<string>());
        else MySay(vector<string>());
    } 

    static int lastConnect = Now();
    if (senators.size() < FLAGS_num && lastConnect + Seconds(FLAGS_secs) < Now()) {
        MyNewSenator(vector<string>());
        lastConnect = Now();
        return 0;
    }

    app->shell.FGets();
    return 0;
}

}; // namespace LFL
using namespace LFL;

extern "C" int main(int argc, const char *argv[]) {

    screen->caption = "senators";
    app->frame_cb = Frame;
    app->logfilename = StrCat(LFAppDownloadDir(), "senators.txt");
    FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = FLAGS_lfapp_camera = 0;
    FLAGS_lfapp_network = 1;
#ifdef _WIN32
    open_console = 1;
#endif

    if (app->Create(argc, argv, __FILE__)) { app->Free(); return -1; }
    if (app->Init()) { app->Free(); return -1; }

    app->shell.command.push_back(Shell::Command("senator",   bind(&MyNewSenator,  _1)));
    app->shell.command.push_back(Shell::Command("s",         bind(&MyS,           _1)));
    app->shell.command.push_back(Shell::Command("send",      bind(&MySend,        _1)));
    app->shell.command.push_back(Shell::Command("say",       bind(&MySay,         _1)));
    app->shell.command.push_back(Shell::Command("csay",      bind(&MyColorSay,    _1)));
    app->shell.command.push_back(Shell::Command("sayfile",   bind(&MySayFile,     _1)));
    app->shell.command.push_back(Shell::Command("vprefix",   bind(&MyVPrefix,     _1)));
    app->shell.command.push_back(Shell::Command("vprefixes", bind(&MyVPrefixSize, _1)));

    load_senatornames();

    for (int i=0; i<FLAGS_numinit; i++) MyNewSenator(vector<string>());

    int ret = app->Main();
    return ret;
}
