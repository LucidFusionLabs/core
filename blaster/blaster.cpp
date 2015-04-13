/*
 * $Id: blaster.cpp 1336 2014-12-08 09:29:59Z justin $
 */

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "crawler/html.h"
#include "crawler/google_chart.h"
#include "resolver.h"

namespace LFL {
DEFINE_int   (gui_port,            0,                 "GUI Port");

DEFINE_string(domain,              "",                "[domain] template variable; used for EHLO if not -ehlo_domain");
DEFINE_bool  (ehlo_domain,         false,             "Use reverse DNS of source IP for EHLO");
DEFINE_string(template,            "",                "Template filename");
DEFINE_string(campaignid,          "",                "Campaign ID");
DEFINE_string(email_list,          "",                "Deliver template to addresses in file");
DEFINE_string(dns_cache,           "resolve.out.txt", "Deliver only to domains in DNS cache file");
DEFINE_string(configuration,       "",                "Delivery policy name");
DEFINE_string(configuration_file,  "blaster.cnf",     "Delivery policies file");
DEFINE_string(encode_uid,          "",                "Encode user id");
DEFINE_string(decode_uid,          "",                "Decode user id");

DEFINE_string(ip_address,                  "",        "Deliver from comma separated IP list; blank for all");
DEFINE_string(log_location,                "",        "Log location; blank for current directory");
DEFINE_int   (max_messages_per_connection, 0,         "Max msgs/conn; 0 for unlimited");
DEFINE_int   (max_connections,             0,         "Max total outstanding connections");
DEFINE_int   (init_connections,            0,         "Create N connections before start sending");
DEFINE_int   (max_mx_connections,          0,         "Max connections to an MX");
DEFINE_int   (max_mx_connections_per_ip,   0,         "Max connections to MX per IP");
DEFINE_int   (max_mx_connection_retrys,    2,         "Attempt N connections to each MX IP");

DECLARE_int  (target_fps);              // 100        Target_connects_per_second = target_fps * frame_connect_max
DEFINE_int   (frame_connect_max,           10,        "max connect() per frame");
}; // namespace LFL

using namespace LFL;

bool init_connections_reached = true;
vector<Callback> init_connections_queue;

struct BlasterConfig : public HTMLParser {
    string default_policy;
    map<string, string> policies;
    void Text(const string &text, const TagStack &stack) {
        CHECK(stack.size());
        const HTMLParser::KV &kv = stack.back().attr;
        HTMLParser::KV::const_iterator i = kv.find("name"), j;
        if (i == kv.end()) FATAL("Missing name attribute");
        if (default_policy.empty() && (j = kv.find("default")) != kv.end())
            if (Scannable::Scan(bool(), j->second.c_str())) default_policy = i->second;
        policies[i->second].append(text + "\n");
    }
    void Apply(const string &name) {
        string ip_addresses;
        FlagMap *fm = Singleton<FlagMap>::Get();
        if (!name.empty()) {
            INFO("Applying configuration ", name);
            map<string, string>::const_iterator i = policies.find(name);
            if (i == policies.end()) FATAL("missing policy ", name);
            StringLineIter lines(i->second);
            for (const char *line = lines.Next(); line; line = lines.Next()) {
                StringWordIter words(StringPiece(line, lines.linelen));
                string k = toconvert(tolower(BlankNull(words.Next())), tochar<'-','_'>);
                string v = BlankNull(words.Next());
                if (SuffixMatch(k, ":")) k.erase(k.size()-1);
                if (k == "ip_address") StrAppend(&ip_addresses, ip_addresses.size()?",":"", v);
                else if (!fm->Set(k, v)) FATAL("Unknown var '", k, "', Did you mean -", fm->Match(k, __FILE__), "?");
            }
        }
        if (ip_addresses.empty() && FLAGS_ip_address.empty()) {
            set<IPV4::Addr> ips; Sniffer::GetDeviceAddressSet(&ips);
            ip_addresses = IPV4::MakeCSV(ips);
        }
        if (!ip_addresses.empty()) fm->Set("ip_address", ip_addresses);
    }
} blaster_config;

struct BulkMailEncoding {
    static string Passphrase() { return "This is our crazy encryption key! Poot!"; }
    static string PerlCBCKeyFromPassphrase(const string &passphrase, int requested_key_size) {
        Crypto *crypto = Singleton<Crypto>::Get();
        string key = crypto->MD5(passphrase);
        while (key.size() < requested_key_size) StrAppend(&key, crypto->MD5(key));
        key.resize(requested_key_size);
        return key;
    }

    static string EncodeUserID(const string &in) { return EncodeUserID(Passphrase(), in); }
    static string EncodeUserID(const string &passphrase, const string &in) {
        string padded_input = in;
        if (padded_input.size() < 6) padded_input.resize(6, '|');
        string salted_input = StrCat(padded_input.substr(0, 6), StringPrintf("%c%c", rand()%256, rand()%256), padded_input.substr(6));
        while (salted_input.size() % 8) salted_input += "|";
        string key = PerlCBCKeyFromPassphrase(passphrase, 56);
        string out = Singleton<Crypto>::Get()->Blowfish(key, salted_input, true);
        string b64 = Singleton<Base64>::Get()->Encode(out.c_str(), out.size());
        return toconvert(b64, tochar2<'/','_', '+','.'>);
    }

    static string DecodeUserID(const string &in) { return DecodeUserID(Passphrase(), in); }
    static string DecodeUserID(const string &passphrase, const string &in) {
        string b64 = toconvert(in, tochar2<'_','/', '.','+'>);
        string b64d = Singleton<Base64>::Get()->Decode(b64.c_str(), b64.size());
        string key = PerlCBCKeyFromPassphrase(passphrase, 56);
        string out = Singleton<Crypto>::Get()->Blowfish(key, b64d, false);
        return strip(StrCat(out.substr(0, 6), out.substr(8)).c_str(), isint2<'|','+'>);
   }
};

struct BulkMailTemplate {
    vector<string> textblocks;
    struct Function {
        string name; int textblock_index;
        Function(const string &n, int ind) : name(n), textblock_index(ind) {}
    };
    vector<Function> variables;
    
    bool Open(const string &text) {
        if (text.empty()) return false;
        string current_textblock;
        StringLineIter lines(text, StringLineIter::Flag::BlankLines);
        for (const char *line = lines.Next(); line; line = lines.Next()) {
            if (!*line) current_textblock += "\r\n";
            for (const char *li = line; *li; /**/) {
                const char *template_var_begin = strchr(li, '['), *template_var_end = 0;
                if (!template_var_begin) {
                    current_textblock.append(li, lines.linelen-(li-line));
                    current_textblock += "\r\n";
                    break;
                }
                current_textblock.append(li, template_var_begin - li);
                textblocks.push_back(current_textblock);
                current_textblock.clear();

                CHECK((template_var_end = strchr(template_var_begin, ']')));
                if (!*(li = template_var_end+1)) current_textblock += "\r\n";
                string template_variable(template_var_begin+1, template_var_end-template_var_begin-1);
                variables.push_back(Function(tolower(template_variable), textblocks.size()));
            }
        }
        if (!current_textblock.empty()) textblocks.push_back(current_textblock);
        return true;
    }

    string Evaluate(const string &email, const string &helo_domain) {
        string out; Time now=Now(); int textblock_i=0; map<string, string> rands;
        for (int variable_i = 0; variable_i < variables.size(); variable_i++) {
            const Function &f = variables[variable_i];
            Join(&out, textblocks, textblock_i, f.textblock_index);
            textblock_i = f.textblock_index;
            string result;
            if      (f.name == "user_id")     result = BulkMailEncoding::EncodeUserID(email);
            else if (f.name == "campaign_id") result = FLAGS_campaignid;
            else if (f.name == "message-id")  result = "Encoding_method_TBD";
            else if (f.name == "domain")      result = FLAGS_domain;
            else if (f.name == "*to")         result = email;
            else if (f.name == "*date")       result = localsmtptime(now);
            else if (PrefixMatch(f.name, "random")) {
                map<string, string>::const_iterator i = rands.find(f.name);
                if (i != rands.end()) result = i->second;
                else {
                    result.resize(10);
                    for (int j=0; j<result.size(); j++) { int v=rand()%36; result[j] = (v < 10) ? ('0' + v) : ('a' + v-10); }
                    rands[f.name] = result;
                }
            }
            else FATAL("unknown template variable ", f.name);
            StrAppend(&out, result);
        }
        Join(&out, textblocks, textblock_i, textblocks.size());
        return out;
    }

} bulk_mail_template;

struct HistogramValue {
    string val; long long occurrences;
    HistogramValue() : occurrences(0) {}
    void Set(const string &v) { val=v; occurrences++; }
};

struct BulkMailer {
    long long queue_attempted, queued, completed, sent, delivered, rejected, aborted;
    File *delivery_log, *bounce_log, *retry_log;
    map<int, HistogramValue> delivery_code_histogram;
    RollingAvg<unsigned> connects_per_frame;
    string logdir, template_name;
    SMTPClient *smtp_client;
    DeltaSampler *stat_log;
    bool first_pass;
    Time started;
    BulkMailer() : queued(0), completed(0), sent(0), delivered(0), rejected(0), aborted(0), delivery_log(0), bounce_log(0), retry_log(0), connects_per_frame(100), smtp_client(0), first_pass(1), started(0) {
        vector<const long long *> table; vector<string> labels;
        table.push_back(&delivered); labels.push_back("delivered");
        table.push_back(&rejected);  labels.push_back("rejected");
        table.push_back(&aborted);   labels.push_back("aborted");
        stat_log = new DeltaSampler(Seconds(60), table, labels);
    }

    typedef vector<IPV4::Addr> MXAddrs;

    // MX is defined by unique IP-vector
    struct MX {
        MXAddrs A;
        string host;
        set<Connection*> conns;
        void Set(const string &h, const MXAddrs &a) { host=h; A=a; }
        string DebugString() const { return StrCat(host, ":", IPV4::MakeCSV(A)); }
    };
    typedef vector<MX*> MXsAddrs;

    // Domain is defined by a (non-unique) MX-vector
    struct Domain { 
        vector<string> user;
        MXsAddrs mx;
    };
    map<string, Domain*> domains;

    struct Target;
    struct TargetConnection {  // Connection to IP of MX of Target
        Target *target;
        MX *target_mx;
        Time connected;
        int address_index, address_retries;
        long long attempted, delivered, rejected, prev_attempted, prev_delivered, prev_rejected;
        bool retrying;
        TargetConnection(Target *T) : target(T), connected(0), address_index(0), address_retries(0),
            attempted(0), delivered(0), rejected(0), prev_attempted(0), prev_delivered(0), prev_rejected(0), retrying(0) {}

        bool FirstDelivery() const { return attempted == prev_attempted; }
        int NumDelivered() const { return delivered - prev_delivered; }
        int NumRejected() const { return rejected - prev_rejected; }

        int NextAddress() {
            if (retrying) retrying = 0;
            else address_retries++;
            if (address_retries >= FLAGS_max_mx_connection_retrys) { address_index++; address_retries=0; }
            return address_index;
        }

        void Connected(Connection *c) {
            connected = Now();
            target_mx->conns.insert(c);
            prev_attempted = attempted;
            prev_delivered = delivered;
            prev_rejected = rejected;
        }
    };

    struct Target {  // Target is defined by unique MX-vector
        MXsAddrs mx;
        BulkMailer *parent;
        vector<string> email;  // Emails for domains served by Target MX-vector
        int deliverable_conns;
        map<Connection*, TargetConnection*> conns;
        Target() : parent(0), deliverable_conns(0) {}
        void Set(BulkMailer *P, const MXsAddrs MX) { parent=P; mx=MX; }

        // Address(index) returns each IP of each MX in order
        IPV4::Addr Address(int index) const { int address_ind, ind=AddressIndex(index, &address_ind); return mx[ind]->A[address_ind]; }
        string     Host   (int index) const { int address_ind, ind=AddressIndex(index, &address_ind); return mx[ind]->host; }
        MX        *MXi    (int index) const { int address_ind, ind=AddressIndex(index, &address_ind); return mx[ind]; }
        int Addresses() const { int count=0; for (int i=0; i<mx.size(); i++) count += mx[i]->A.size(); return count; }
        int AddressIndex(int index, int *address_index_out) const {
            for (int i=0, count=0; i<mx.size(); i++) {
                if (count + mx[i]->A.size() > index) { *address_index_out = index - count; return i; }
                count += mx[i]->A.size();
            }
            FATAL("oob index ", index);
        }

        string DebugString() const {
            string ret = StrCat("emails=", email.size());
            for (int i=0; i<mx.size(); i++) StrAppend(&ret, " ", mx[i]->DebugString());
            return ret;
        }

        bool Connect() { return Connect(new TargetConnection(this)); }
        bool Connect(TargetConnection *tc) {
            tc->target_mx = MXi(tc->address_index);
            if (!email.size() || (FLAGS_max_mx_connections && tc->target_mx->conns.size() >= FLAGS_max_mx_connections))
            { delete tc; return false; }

            IPV4EndpointPoolFilter src_pool((IPV4EndpointPool*)parent->smtp_client->connect_src_pool);
            if (FLAGS_max_mx_connections_per_ip) {
                GetMaxedSourceIPSet(&src_pool.filter);
                if (src_pool.filter.size() >= src_pool.wrap->source_addrs.size()) { delete tc; return false; }
            }

            IPV4::Addr addr = Address(tc->address_index);
            Connection *c = parent->smtp_client->DeliverTo
                (addr, &src_pool,
                 SMTPClient::DeliverableCB(bind(&Target::DeliverableCB, tc->target, _1, tc, _2, _3)),
                 SMTPClient::DeliveredCB  (bind(&Target::DeliveredCB,   tc->target, _1, tc, _2, _3, _4)));
            if (!c) { ERROR("DeliverTo ", IPV4::Text(addr), " failed"); LostConnection(0, tc); }
            else {
                conns[c] = tc;
                tc->Connected(c);
            }
            return true;
        }

        void LostConnection(Connection *c, TargetConnection *tc) {
            if (c) {
                Time interval = Now() - tc->connected;
                long long delivered = tc->NumDelivered(), rejected = tc->NumRejected(); 
                if (!tc->FirstDelivery()) tc->target->deliverable_conns--;

                INFO("Closing ", c->Name(), " after ", intervaltime(interval), " delivered=", delivered,
                     ", rejected=", rejected, ", QPS=", (delivered + rejected) / static_cast<float>(Time2time_t(interval)),
                     ", target_deliverable_conns=", tc->target->deliverable_conns);

                if (!               conns.erase(c)) ERROR(tc->target->DebugString(),    " missing ", c->Name());
                if (!tc->target_mx->conns.erase(c)) ERROR(tc->target->DebugString(), " mx missing ", c->Name());
            }
            if (!email.size())                    { delete tc; if (c && !conns.size()) parent->Done  (this); return; }
            if (tc->NextAddress() >= Addresses()) { delete tc; if (c && !conns.size()) parent->Failed(this); return; }
            Connect(tc);
        }

        void GetMaxedSourceIPSet(set<IPV4::Addr> *out) {
            map<IPV4::Addr, int> src_addr_count;
            for (map<Connection*, TargetConnection*>::const_iterator i = conns.begin(); i != conns.end(); ++i)
                src_addr_count[i->first->src_addr]++;

            for (map<IPV4::Addr, int>::const_iterator i = src_addr_count.begin(); i != src_addr_count.end(); ++i)
                if (i->second >= FLAGS_max_mx_connections_per_ip)
                    out->insert(i->first);
        }

        bool operator<(const Target &x) const { return email.size() < x.email.size(); }
        static bool SortPointers(const Target *l, const Target *r) { return *l < *r; }

        bool DeliverableCB(Connection *c, TargetConnection *tc, const string &helo_domain, SMTP::Message *out) {
            if (!out) { ERROR("lost connection to ", c->Name()); LostConnection(c, tc); return true; }

            tc->retrying = (FLAGS_max_messages_per_connection && tc->NumDelivered() >= FLAGS_max_messages_per_connection);
            if (!email.size() || tc->retrying) return true;

            out->rcpt_to.push_back(email.back());
            out->content = bulk_mail_template.Evaluate(email.back(), helo_domain);
            out->mail_from = SMTP::EmailFrom(out->content);

            if (tc->FirstDelivery()) deliverable_conns++;
            tc->attempted++;

            email.pop_back();
            parent->sent++;

            if (!init_connections_reached)
                init_connections_queue.push_back(bind(&SMTPClient::DeliverDeferred, c));
            return init_connections_reached;
        }

        void DeliveredCB(Connection *c, TargetConnection *tc, const SMTP::Message &mail, int code, const string &msg) {
            if (!c) {
                parent->sent--;
                email.insert(email.end(), mail.rcpt_to.begin(), mail.rcpt_to.end());
                return;
            }

            bool success = SMTP::SuccessCode(code), retryable = SMTP::RetryableCode(code);
            parent->completed++;
            parent->delivery_code_histogram[code].Set(msg);
            if (success) { tc->delivered++; parent->delivered++; }
            else         { tc->rejected++;  parent->rejected++;  }

            File *out = parent->bounce_log;
            if      (success)   out = parent->delivery_log;
            else if (retryable) out = parent->retry_log;

            string delivery_mx_host = Host(tc->address_index);
            IPV4::Addr delivery_mx_ip = Address(tc->address_index);
            CHECK_EQ(mail.rcpt_to.size(), 1);
            CHECK_EQ(delivery_mx_ip, c->addr);

            string line = StrCat(logtime(Now()), " ", mail.rcpt_to[0], " (", mail.mail_from, " ", IPV4::Text(c->src_addr, c->src_port), ") ");
            StrAppend(&line, FLAGS_configuration, " ", parent->template_name,  " ", delivery_mx_host, "=", IPV4::Text(delivery_mx_ip));
            StrAppend(&line, " response: ", msg.size()?ReplaceNewlines(msg, "<EOL>"):"<Lost Connection>", "\r\n");
            out->Write(line);
            out->Flush();
        }
    };
    vector<Target*> queue, next_queue;
    set<Target*> outstanding;
    DiscreteDistribution outstanding_sampler;

    int OpenEmailList(File *f) {
        if (!f->Opened()) return 0;
        long long queue_attempted_start = queue_attempted;
        for (const char *line = f->NextLine(); line; line = f->NextLine()) {
            vector<string> email;
            Split(line, isint3<'@', ' ', '\t'>, &email);
            CHECK_EQ(email.size(), 2);
            FindOrInsert(domains, email[1])->second->user.push_back(email[0]);
            queue_attempted++;
        }
        return queue_attempted - queue_attempted_start;
    }

    int OpenDNSCache(File *f) {
        if (!f->Opened()) return 0;

        long long queued_start = queued;
        map<MXsAddrs, Target*> targets;
        map<MX*, Target*> targetMXs;
        map<IPV4::Addr, MX*> MXIPs;
        map<MXAddrs, MX*> MXs;

        // For each line of resolve.out.txt
        for (const char *line = f->NextLine(); line; line = f->NextLine()) {
            vector<ResolvedMX> resolvedA, resolvedMX, *targetMXlist = &resolvedMX;
            ParseResolverOutput(line, f->nr.record_len, &resolvedA, &resolvedMX);
            if (!resolvedA.size()) { ERROR("failed: ", line); continue; }
            if (!resolvedMX.size()) targetMXlist = &resolvedA;

            // Filter for domains added by OpenEmailList()
            if (resolvedA[0].host.empty()) { ERROR("parse: ", line); continue; }
            map<string, Domain*>::const_iterator i = domains.find(resolvedA[0].host);
            if (i == domains.end()) continue;
            Domain *domain = i->second;

            // Skip duplicate domains
            if (domain->mx.size()) { ERROR("duplicate dns_cache entry=", resolvedA[0].host); continue; }

            // For each MX responsible for domain in preference order
            for (vector<ResolvedMX>::const_iterator j = targetMXlist->begin(); j != targetMXlist->end(); ++j) {
                if (!j->A.size()) continue;

                bool inserted; // Join MXs by IP-vector
                MX *mx = FindOrInsert(MXs, j->A, &inserted)->second;
                if (inserted) mx->Set(j->host, j->A); 

                // Check that IPs are unique to MXs
                for (MXAddrs::const_iterator k = j->A.begin(); k != j->A.end(); ++k) {
                    if (j->A.empty()) continue;

                    map<IPV4::Addr, MX*>::const_iterator l = MXIPs.find(*k);
                    if (l != MXIPs.end() && l->second != mx) {
                        INFO(IPV4::Text(*k), " belongs to multiple MX-address-sets, (",
                             l->second->DebugString(), ") and (", mx->DebugString(), "), ", 
                             i->second->user.size(), " targets affected");
                    }
                    MXIPs[*k] = mx;
                }
                domain->mx.push_back(mx);
            }
            if (domain->mx.empty()) continue;

            bool inserted; // Join targets by MX-vector
            Target *targ = FindOrInsert(targets, domain->mx, &inserted)->second;
            if (inserted) {
                targ->Set(this, domain->mx);
                queue.push_back(targ);
            }

            for (int i=0; i<domain->mx.size(); i++) {
                // Check that MXs are unique to targets
                map<MX*, Target*>::const_iterator l = targetMXs.find(domain->mx[i]);
                if (l != targetMXs.end() && l->second != targ) {
                    INFO(IPV4::MakeCSV(domain->mx[i]->A), " belongs to multiple targets (",
                         l->second->DebugString(), ") and (", targ->DebugString(), ")");
                }
                targetMXs[domain->mx[i]] = targ;
            }

            // Add email addresses to target
            long long prev_targ_emails = targ->email.size();
            for (vector<string>::const_iterator j = i->second->user.begin(); j != i->second->user.end(); ++j)
                targ->email.push_back(StrCat(*j, "@", i->first));

            queued += (targ->email.size() - prev_targ_emails);
            INFO("Queued ", resolvedA[0].host, " @ ", targ->DebugString());
        }
        return queued - queued_start;
    }

    int Prepare() {
        long long prepared = 0;
        for (map<string, Domain*>::const_iterator i = domains.begin(); i != domains.end(); ++i) {
            if (!i->second->mx.size() || !i->second->user.size()) { delete i->second; continue; }
            prepared += i->second->user.size();
        }
        domains.clear();
        sort(queue.begin(), queue.end(), Target::SortPointers);
        CHECK_EQ(prepared, queued);

        Time now = Now(); int len;
        logdir = FLAGS_log_location;
        template_name = string(BaseName(FLAGS_template, &len)).substr(0, len);
        if (!logdir.empty() && logdir[logdir.size()-1] != LocalFile::Slash) logdir.append(StringPrintf("%c", LocalFile::Slash));

        string logfile = StrCat(logdir, logfiledaytime(now), "-", template_name, "-");
        delivery_log = new LocalFile(StrCat(logfile, "delivery.log"), "a"); if (!delivery_log->Opened()) FATAL("open ", delivery_log->Filename());
        bounce_log   = new LocalFile(StrCat(logfile, "bounce.log"),   "a"); if (!bounce_log  ->Opened()) FATAL("open ", bounce_log  ->Filename());
        retry_log    = new LocalFile(StrCat(logfile, "retry.log"),    "a"); if (!retry_log   ->Opened()) FATAL("open ", retry_log   ->Filename());

        smtp_client = Singleton<SMTPClient>::Get();
        started = Now();
        return prepared;
    }

    bool ConnectionsAvailable() const {
        if (FLAGS_max_connections && smtp_client->conn.size() >= FLAGS_max_connections) return false;
        return smtp_client->connect_src_pool->Available();
    }
   
    void Frame() {
        stat_log->Update();

        if (!queue.size() && first_pass) first_pass=0;
        if (!queue.size() && next_queue.size()) { queue=next_queue; next_queue.clear(); }

        if (!first_pass && (!outstanding_sampler.Size() ||
                            outstanding_sampler.samples >= (FLAGS_target_fps * FLAGS_frame_connect_max * 5))) {
            outstanding_sampler.Clear();
            for (set<Target*>::const_iterator i = outstanding.begin(); i != outstanding.end(); ++i) 
                outstanding_sampler.Add((*i)->email.size(), (*i));
            outstanding_sampler.Prepare();
        }

        int connected = 0, queue_retrys = FLAGS_frame_connect_max*2, sample_retrys = FLAGS_frame_connect_max*2;
        for (;;) {
            bool queue_avail = queue_retrys && queue.size();
            bool sample_avail = false;
            for (set<Target*>::const_iterator i = outstanding.begin(); !first_pass && i != outstanding.end(); ++i)
                if ((*i)->email.size()) { sample_avail = true; break; }

            if (!queue_avail && !sample_avail) break;
            if (connected >= FLAGS_frame_connect_max || !ConnectionsAvailable()) break;

            if (queue_avail) {
                Target *t = queue.back();
                queue.pop_back();
                if (!t->Connect()) { next_queue.push_back(t); queue_retrys--; }
                else               { outstanding.insert(t);   connected++;    }
            }
            if (connected >= FLAGS_frame_connect_max || !ConnectionsAvailable()) break;

            if (sample_avail) {
                Target *t = (Target*)outstanding_sampler.Sample();
                if (outstanding.find(t) != outstanding.end() &&
                    (!init_connections_reached || t->deliverable_conns) &&
                    t->Connect()) connected++;
                else sample_retrys--;
            }
        }
        connects_per_frame.Add(connected);

        if (!init_connections_reached && smtp_client->conn.size() >= FLAGS_init_connections) {
            init_connections_reached = true;
            for (vector<Callback>::iterator i = init_connections_queue.begin();
                 i != init_connections_queue.end(); ++i) {
                (*i)();
            }
            init_connections_queue.clear();
        }
    }

    void Done(Target *t) {
        INFO("Completed: ", t->DebugString());
        outstanding.erase(t);
        delete t;
    }

    void Failed(Target *t) {
        INFO("Failed-remaining: ", t->DebugString());
        aborted   += t->email.size();
        completed += t->email.size();
        outstanding.erase(t);
        delete t;
    }

    string StatusLine() const { 
        long long outstanding_emails = 0;
        for (set<Target*>::const_iterator i = outstanding.begin(); i != outstanding.end(); ++i) outstanding_emails += (*i)->email.size();
        string ret = StrCat("queued=", queued, ", finished=", completed, ", sent=", sent, ", delivered=", delivered);
        StrAppend(&ret, ", rejected=", rejected, ", aborted=", aborted, ", outstanding=", outstanding_emails);
        StrAppend(&ret, ", mtas=", outstanding.size(), ", connections=", Singleton<SMTPClient>::Get()->conn.size());
        return ret;
    }

} bulk_mailer;

int frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    bulk_mailer.Frame();

    char buf[256];
    if (FGets(buf, sizeof(buf))) ERROR("FPS=", FPS(), ", ", bulk_mailer.StatusLine());
    return 0;
}

struct StatusGUI : public HTTPServer::Resource {
    HTTPServer::Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
        string response = StrCat("<html><head>\n", GChartsHTML::JSHeader());

        vector<vector<string> > table;
        DeltaGrapher::JSTable(*bulk_mailer.stat_log, &table, 60);
        if (table.size()>1) StrAppend(&response, GChartsHTML::JSAreaChart("viz1", 600, 400, "Last Hour", "Sent", "Minutes", table));

        StrAppend(&response, GChartsHTML::JSFooter(), "</head><body><h>Blaster Version 1.0 Up ", intervaltime(Now() - bulk_mailer.started), "  </h>\n");
        StrAppend(&response, "<p>", bulk_mailer.StatusLine(), "</p>\n");
        StrAppend(&response, "<p>target conn/sec=", FLAGS_target_fps * FLAGS_frame_connect_max, ", conn/sec=", FPS() * bulk_mailer.connects_per_frame.Avg(), "</p>\n");
        StrAppend(&response, "<p>target_fps=", FLAGS_target_fps, ", FPS=", FPS(), "</p>\n");
        StrAppend(&response, "<p>frame_connect_max=", FLAGS_frame_connect_max, ", connects_per_frame=", bulk_mailer.connects_per_frame.Avg(), "</p>\n");

        StrAppend(&response, GChartsHTML::DivElement("viz1", 600, 400), "\n");
        StrAppend(&response, "<p>Delivery code histogram:<br/>\n");
        for (map<int, HistogramValue>::const_iterator i = bulk_mailer.delivery_code_histogram.begin(); i != bulk_mailer.delivery_code_histogram.end(); ++i)
            StrAppend(&response, i->first, ": ", i->second.val, " (count=<b>", i->second.occurrences, "</b>)<br/>\n");
        StrAppend(&response, "</p></body></html>\n");
        return HTTPServer::Response("text/html; charset=UTF-8", &response);
    }
};

extern "C" {
int main(int argc, const char **argv) {
    screen->frame_cb = frame;
    app->logfilename = StrCat(LFAppDownloadDir(), "blaster.txt");
    FLAGS_max_rlimit_core = FLAGS_max_rlimit_open_files = 1;
    FLAGS_lfapp_network = 1;

    if (app->Create(argc, argv, __FILE__)) { ERROR("lfapp init failed: ", strerror(errno)); return app->Free(); }
    if (app->Init())                       { ERROR("lfapp open failed: ", strerror(errno)); return app->Free(); }

    if (!FLAGS_encode_uid.empty()) { INFO("Encode('", FLAGS_encode_uid, "') = '", BulkMailEncoding::EncodeUserID(FLAGS_encode_uid), "'"); return 0; }
    if (!FLAGS_decode_uid.empty()) { INFO("Decode('", FLAGS_decode_uid, "') = '", BulkMailEncoding::DecodeUserID(FLAGS_decode_uid), "'"); return 0; }

    if (!FLAGS_configuration_file.empty()) {
        LocalFile lf(FLAGS_configuration_file, "r");
        if (lf.Opened()) blaster_config.Parse(&lf);
    }
    if (!FLAGS_configuration.empty()) blaster_config.Apply(FLAGS_configuration);
    else                              blaster_config.Apply(FLAGS_configuration=blaster_config.default_policy);

    HTTPServer httpd(FLAGS_gui_port, false);
    if (FLAGS_gui_port) {
        httpd.AddURL("/",    new StatusGUI());
        httpd.AddURL("/cmd", new HTTPServer::ConsoleResource);
        if (app->network.Enable(&httpd)) return -1;
    }

    CHECK(!FLAGS_domain.empty());
    if (!FLAGS_gui_port && (FLAGS_email_list.empty() || FLAGS_dns_cache.empty())) { INFO("nothing to do"); return 0; }
    CHECK(bulk_mail_template.Open(LocalFile(FLAGS_template, "r").Contents()));
    if (!FLAGS_email_list.empty()) {
        LocalFile email_list(FLAGS_email_list, "r");
        CHECK(bulk_mailer.OpenEmailList(&email_list));
    }
    if (!FLAGS_dns_cache.empty()) {
        LocalFile dns_cache(FLAGS_dns_cache, "r");
        bulk_mailer.OpenDNSCache(&dns_cache);
    }

    long long failed_to_queue = bulk_mailer.queue_attempted - bulk_mailer.queued;
    INFO("BulkMailer queued ", bulk_mailer.queued, ", failed to queue ", failed_to_queue);
    if (bulk_mailer.Prepare() <= 0 && !FLAGS_gui_port) { INFO("nothing to do"); return 0; }

    SMTPClient *smtp = Singleton<SMTPClient>::Get();
    smtp->connect_src_pool = new IPV4EndpointPool(FLAGS_ip_address);
    if (FLAGS_init_connections) init_connections_reached = false;
    if (!FLAGS_ehlo_domain) smtp->domain = FLAGS_domain;
    else {
        set<IPV4::Addr> ips; IPV4::ParseCSV(FLAGS_ip_address, &ips);
        for (set<IPV4::Addr>::const_iterator i = ips.begin(); i != ips.end(); ++i)
            smtp->domains[*i] = SystemNetwork::GetHostByAddr(*i);
        CHECK_GT(smtp->domains.size(), 0);
        smtp->domains[IPV4::Parse("0.0.0.0")  ] = "localhost";
        smtp->domains[IPV4::Parse("127.0.0.1")] = "localhost";
    }
    if (app->network.Enable(smtp)) return -1;

    int ret = app->Main();
    ERROR("PerformanceTimers: ", Singleton<PerformanceTimers>::Get()->DebugString());
    return ret;
}
}
