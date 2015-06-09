/*
 * $Id$
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

#include "lfapp/lfapp.h"
#include "lfapp/wire.h"
#include "lfapp/ipc.h"

#ifndef WIN32
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif

namespace LFL {
const char *Protocol::Name(int p) {
    switch (p) {
        case TCP:   return "TCP";
        case UDP:   return "UDP";
        case UNIX:  return "UNIX";
        case GPLUS: return "GPLUS";
        default:    return "";
    }
}

void Serializable::Header::Out(Stream *o) const { o->Htons( id); o->Htons( seq); }
void Serializable::Header::In(const Stream *i)  { i->Ntohs(&id); i->Ntohs(&seq); }

string Serializable::ToString()                   { string ret; ToString(&ret);      return ret; }
string Serializable::ToString(unsigned short seq) { string ret; ToString(&ret, seq); return ret; }

void Serializable::ToString(string *out) {
    out->resize(Size());
    return ToString((char*)out->data(), out->size());
}
void Serializable::ToString(string *out, unsigned short seq) {
    out->resize(Header::size + Size());
    return ToString((char*)out->data(), out->size(), seq);
}

void Serializable::ToString(char *buf, int len) {
    MutableStream os(buf, len);
    Out(&os);
}
void Serializable::ToString(char *buf, int len, unsigned short seq) {
    MutableStream os(buf, len);
    Header hdr = { (unsigned short)Type(), seq };
    hdr.Out(&os);
    Out(&os);
}

const IPV4::Addr IPV4::ANY = INADDR_ANY;

IPV4::Addr IPV4::Parse(const string &ip) { return inet_addr(ip.c_str()); }

void IPV4::ParseCSV(const string &text, vector<IPV4::Addr> *out) {
    vector<string> addrs; IPV4::Addr addr;
    Split(text, iscomma, &addrs);
    for (int i = 0; i < addrs.size(); i++) {
        if ((addr = Parse(addrs[i])) == INADDR_NONE) FATAL("unknown addr ", addrs[i]);
        out->push_back(addr);
    }
}

void IPV4::ParseCSV(const string &text, set<IPV4::Addr> *out) {
    vector<string> addrs; IPV4::Addr addr;
    Split(text, iscomma, &addrs);
    for (int i = 0; i < addrs.size(); i++) {
        if ((addr = Parse(addrs[i])) == INADDR_NONE) FATAL("unknown addr ", addrs[i]);
        out->insert(addr);
    }
}

string IPV4::MakeCSV(const vector<IPV4::Addr> &in) {
    string ret;
    for (vector<Addr>::const_iterator i = in.begin(); i != in.end(); ++i) StrAppend(&ret, ret.size()?",":"", IPV4::Text(*i));
    return ret;
}

string IPV4::MakeCSV(const set<IPV4::Addr> &in) {
    string ret;
    for (set<Addr>::const_iterator i = in.begin(); i != in.end(); ++i) StrAppend(&ret, ret.size()?",":"", IPV4::Text(*i));
    return ret;
}

int DNS::WriteRequest(unsigned short id, const string &querytext, unsigned short type, char *out, int len) {
    Serializable::MutableStream os(out, len);
    Header *hdr = (Header*)os.Get(Header::size);
    memset(hdr, 0, Header::size);
    hdr->rd = 1;
    hdr->id = id;
    hdr->qdcount = htons(1);

    StringWordIter words(querytext, isdot);
    for (string word = IterNextString(&words); !word.empty(); word = IterNextString(&words)) {
        CHECK_LT(word.size(), 64);
        os.Write8((unsigned char)word.size());
        os.String(word);
    }
    os.Write8((char)0);

    os.Htons(type);                      // QueryTypeClass.Type
    os.Htons((unsigned short)Class::IN); // QueryTypeClass.QClass
    return os.error ? -1 : os.offset;
}

int DNS::ReadResponse(const char *buf, int bufsize, Response *res) {
    Serializable::ConstStream is(buf, bufsize);
    const Serializable::Stream *in = &is;
    const Header *hdr = (Header*)in->Get(Header::size);

    int qdcount = ntohs(hdr->qdcount);
    int ancount = ntohs(hdr->ancount);
    int nscount = ntohs(hdr->nscount);
    int arcount = ntohs(hdr->arcount);

    for (int i = 0; i < qdcount; i++) {
        Record out; int len;
        if ((len = DNS::ReadString(in->Start(), in->Get(), in->End(), &out.question)) < 0 || !in->Advance(len + 4)) return -1;
        res->Q.push_back(out);
    }

    if (DNS::ReadResourceRecord(in, ancount, &res->A)  < 0) return -1;
    if (DNS::ReadResourceRecord(in, nscount, &res->NS) < 0) return -1;
    if (DNS::ReadResourceRecord(in, arcount, &res->E)  < 0) return -1;
    return 0;
}

int DNS::ReadResourceRecord(const Serializable::Stream *in, int num, vector<Record> *out) {
    for (int i = 0; i < num; i++) {
        Record rec; int len; unsigned short rrlen;
        if ((len = ReadString(in->Start(), in->Get(), in->End(), &rec.question)) < 0 || !in->Advance(len)) return -1;

        in->Ntohs(&rec.type);
        in->Ntohs(&rec._class);
        in->Ntohs(&rec.ttl1);
        in->Ntohs(&rec.ttl2);
        in->Ntohs(&rrlen);

        if (rec._class == Class::IN && rec.type == Type::A) {
            if (rrlen != 4) return -1;
            in->Read32(&rec.addr);
        } else if (rec._class == Class::IN && (rec.type == Type::NS || rec.type == Type::CNAME)) {
            if ((len = ReadString(in->Start(), in->Get(), in->End(), &rec.answer)) != rrlen   || !in->Advance(len)) return -1;
        } else if (rec._class == Class::IN && rec.type == Type::MX) {
            in->Ntohs(&rec.pref);
            if ((len = ReadString(in->Start(), in->Get(), in->End(), &rec.answer)) != rrlen-2 || !in->Advance(len)) return -1;
        } else {
            ERROR("unhandled type=", rec.type, ", class=", rec._class);
            in->Advance(rrlen);
            continue;
        }
        out->push_back(rec);
    }
    return in->error ? -1 : 0;
}

int DNS::ReadString(const char *start, const char *cur, const char *end, string *out) {
    if (!cur) { ERROR("DNS::ReadString null input"); return -1; }
    if (out) out->clear();
    const char *cur_start = cur, *final = 0;
    for (unsigned char len = 1; len && cur < end; cur += len+1) {
        len = *cur;
        if (len >= 64) { // Pointer to elsewhere in packet
            int offset = ntohs(*(unsigned short*)cur) & ~(3<<14);
            if (!final) final = cur + 2;
            cur = start + offset - 2;
            if (cur < start || cur >= end) { ERROR("OOB cur ", (void*)start, " ", (void*)cur, " ", (void*)end); return -1; }
            len = 1;
            continue;
        }
        if (out) StrAppend(out, out->empty() ? "" : ".", string(cur+1, len));
    }
    if (out) *out = tolower(*out);
    if (final) cur = final;
    return (cur > end) ? -1 : (cur - cur_start);
}

void DNS::MakeAnswerMap(const vector<DNS::Record> &in, AnswerMap *out) {
    for (int i = 0; i < in.size(); ++i) {
        const DNS::Record &e = in[i];
        if (e.question.empty() || !e.addr) continue;
        (*out)[e.question].push_back(e.addr);
    }
    for (int i = 0; i < in.size(); ++i) {
        const DNS::Record &e = in[i];
        if (e.question.empty() || e.answer.empty() || e.type != DNS::Type::CNAME) continue;
        AnswerMap::const_iterator a = out->find(e.answer);
        if (a == out->end()) continue;
        VectorAppend((*out)[e.question], a->second.begin(), a->second.end());
    }
}

void DNS::MakeAnswerMap(const vector<DNS::Record> &in, const AnswerMap &qmap, int type, AnswerMap *out) {
    for (int i = 0; i < in.size(); ++i) {
        const DNS::Record &e = in[i];
        if (e.type != type) continue;
        AnswerMap::const_iterator q_iter = qmap.find(e.answer);
        if (e.question.empty() || e.answer.empty() || q_iter == qmap.end())
        { ERROR("DNS::MakeAnswerMap missing ", e.answer); continue; }
        VectorAppend((*out)[e.question], q_iter->second.begin(), q_iter->second.end());
    }
}

string DNS::Response::DebugString() const {
    string ret;
    StrAppend(&ret, "Question ",   Q .size(), "\n"); for (int i = 0; i < Q .size(); ++i) StrAppend(&ret, Q [i].DebugString(), "\n");
    StrAppend(&ret, "Answer ",     A .size(), "\n"); for (int i = 0; i < A .size(); ++i) StrAppend(&ret, A [i].DebugString(), "\n");
    StrAppend(&ret, "NS ",         NS.size(), "\n"); for (int i = 0; i < NS.size(); ++i) StrAppend(&ret, NS[i].DebugString(), "\n");
    StrAppend(&ret, "Additional ", E .size(), "\n"); for (int i = 0; i < E .size(); ++i) StrAppend(&ret, E [i].DebugString(), "\n");
    return ret;
}

/* HTTP */

bool HTTP::ParseHost(const char *host, const char *host_end, string *hostO, string *portO) {
    const char *colon = strstr(host, ":"), *port = 0;
    if (!host_end) host_end = host + strlen(host);
    if (colon && colon < host_end) port = colon+1;
    if (hostO) hostO->assign(host, port ? port-host-1 : host_end-host);
    if (portO) portO->assign(port ? port : "", port ? host_end-port : 0);
    return 1;
}

bool HTTP::ResolveHost(const char *hostname, const char *host_end, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl, int default_port) {
    string h, p;
    if (!ParseHost(hostname, host_end, &h, &p)) return 0;
    return ResolveEndpoint(h, p, ipv4_addr, tcp_port, ssl, default_port);
}

bool HTTP::ResolveEndpoint(const string &host, const string &port, IPV4::Addr *ipv4_addr, int *tcp_port, bool ssl, int default_port) {
    if (ipv4_addr) {
        *ipv4_addr = SystemNetwork::GetHostByName(host);
        if (*ipv4_addr == -1) { ERROR("resolve"); return 0; }
    }
    if (tcp_port) {
        *tcp_port = !port.empty() ? atoi(port.c_str()) : (default_port ? default_port : (ssl ? 443 : 80));
        if (*tcp_port < 0 || *tcp_port >= 65536) { ERROR("oob port"); return 0; }
    }
    return 1;
}

bool HTTP::ParseURL(const char *url, string *protO, string *hostO, string *portO, string *pathO) {
    const char *host = ParseProtocol(url, protO);
    const char *host_end = strstr(host, "/");
    HTTP::ParseHost(host, host_end, hostO, portO);
    if (pathO) pathO->assign(host_end ? host_end+1 : "");
    return 1;
}

bool HTTP::ResolveURL(const char *url, bool *ssl, IPV4::Addr *ipv4_addr, int *tcp_port, string *host, string *path, int default_port, string *prot) {
    string my_prot, port, my_host, my_path; bool my_ssl;
    if (!prot) prot = &my_prot;
    if (!host) host = &my_host;
    if (!path) path = &my_path;
    if (!ssl) ssl = &my_ssl;

    ParseURL(url, prot, host, &port, path);
    *ssl = !prot->empty() && !strcasecmp(prot->c_str(), "https");
    if (!prot->empty() && strcasecmp(prot->c_str(), "http") && !*ssl) return 0;
    if (host->empty()) { ERROR("no host or path"); return 0; }
    if (!HTTP::ResolveEndpoint(*host, port, ipv4_addr, tcp_port, *ssl, default_port)) { ERROR("HTTP::ResolveURL ", host); return 0; }
    return 1;
}

string HTTP::HostURL(const char *url) {
    string my_prot, my_port, my_host, my_path;
    ParseURL(url, &my_prot, &my_host, &my_port, &my_path);
    string ret = !my_prot.empty() ? StrCat(my_prot, "://") : "http://";
    if (!my_host.empty()) ret += my_host;
    if (!my_port.empty()) ret += string(":") + my_port;
    return ret;
}

int HTTP::ParseRequest(char *buf, char **methodO, char **urlO, char **argsO, char **verO) {
    char *url, *ver, *args;
    if (!(url = (char*)FindChar(buf, isspace)))    return -1;    *url = 0;
    if (!(url = (char*)FindChar(url+1, notspace))) return -1;
    if (!(ver = (char*)FindChar(url, isspace)))    return -1;    *ver = 0;
    if (!(ver = (char*)FindChar(ver+1, notspace))) return -1;

    if ((args = strchr(url, '?'))) *args++ = 0;

    if (methodO) *methodO = buf;
    if (urlO) *urlO = url;
    if (argsO) *argsO = args;
    if (verO) *verO = ver;
    return 0;
}

char *HTTP::FindHeadersStart(char *buf) {
    char *start = strstr(buf, "\r\n");
    if (!start) return 0;
    *start = 0;
    return start + 2;
}

char *HTTP::FindHeadersEnd(char *buf) {
    char *end = strstr(buf, "\r\n\r\n");
    if (!end) return 0;
    *(end+2) = 0;
    return end + 2;
}

const char *HTTP::FindHeadersEnd(const char *buf) {
    const char *end = strstr(buf, "\r\n\r\n");
    if (!end) return 0;
    return end + 2;
}

int HTTP::GetHeaderLen(const char *beg, const char *end) { return end - beg + 2; }

int HTTP::GetHeaderNameLen(const char *beg) {
    const char *n = beg;
    while (*n && !isspace(*n) && *n != ':') n++;
    return *n == ':' ? n - beg : 0;
}

int HTTP::GetURLArgNameLen(const char *beg) {
    const char *n = beg;
    while (*n && !isspace(*n) && *n != '=' && *n != '&') n++;
    return n - beg;
}

string HTTP::GrepHeaders(const char *headers, const char *end, const string &name) {
    if (!end) end = HTTP::FindHeadersEnd(headers);
    if (!end) end = headers + strlen(headers);

    int hlen=end-headers, hnlen;
    StringLineIter lines(StringPiece(headers, hlen));
    for (const char *line = lines.Next(); line; line = lines.Next()) {
        if (!(hnlen = HTTP::GetHeaderNameLen(line))) continue;
        if (hnlen == name.size() && !strncasecmp(name.c_str(), line, hnlen)) return string(line+hnlen+2, lines.cur_len-hnlen-2);
    }
    return "";
}

#define HTTPGrepImpl(k, kl, v) \
    va_list ap; va_start(ap, num); \
    char **k = (char **)alloca(num*sizeof(char*)); \
    int *kl = (int *)alloca(num*sizeof(int)); \
    StringPiece **v = (StringPiece **)alloca(num*sizeof(char*)); \
    for (int i=0; i<num; i++) { \
        k[i] = va_arg(ap, char*); \
        kl[i] = strlen(k[i]); \
        v[i] = va_arg(ap, StringPiece*); \
    } \
    va_end(ap);


int HTTP::GrepHeaders(const char *headers, const char *end, int num, ...) {
    HTTPGrepImpl(k, kl, v);
    if (!end) end = HTTP::FindHeadersEnd(headers);
    if (!end) end = headers + strlen(headers);

    int hlen=end-headers, hnlen;
    StringLineIter lines(StringPiece(headers, hlen));
    for (const char *h = lines.Next(); h; h = lines.Next()) {
        if (!(hnlen = HTTP::GetHeaderNameLen(h))) continue;
        for (int i=0; i<num; i++) if (hnlen == kl[i] && !strncasecmp(k[i], h, hnlen)) {
            const char *hv = FindChar(h+hnlen+1, notspace, lines.cur_len-hnlen-1);
            if (!hv) v[i]->clear();
            else     v[i]->assign(hv, lines.cur_len-(hv-h));
        }
    }
    return 0;
}

int HTTP::GrepURLArgs(const char *args, const char *end, int num, ...) {
    HTTPGrepImpl(k, kl, v);
    if (!end) end = args + strlen(args);

    int alen=end-args, anlen;
    StringWordIter words(StringPiece(args, alen), isand, 0);
    for (const char *a = words.Next(); a; a = words.Next()) {
        if (!(anlen = HTTP::GetURLArgNameLen(a))) continue;
        for (int i=0; i<num; i++) if (anlen == kl[i] && !strncasecmp(k[i], a, anlen)) {
            if (*(a+anlen) && *(a+anlen) == '=') v[i]->assign(a+anlen+1, words.cur_len-anlen-1);
            else v[i]->assign(a, words.cur_len);
        }
    }
    return 0;
}

string HTTP::EncodeURL(const char *url) {
    static const char encodeURIcomponentPass[] = "~!*()'";
    static const char encodeURIPass[] = "./@#:?,;-_&";
    string ret;
    for (const unsigned char *p = (const unsigned char *)url; *p; p++) {
        if      (*p >= '0' && *p <= '9') ret += *p;
        else if (*p >= 'a' && *p <= 'z') ret += *p;
        else if (*p >= 'A' && *p <= 'Z') ret += *p;
        else if (strchr(encodeURIcomponentPass, *p)) ret += *p;
        else if (strchr(encodeURIPass, *p)) ret += *p;
        else StringAppendf(&ret, "%%%02x", *p);
    }
    return ret;
}

/* SSH */

int SSH::BinaryPacketLength(const char *b, unsigned char *padding, unsigned char *id) {
    if (padding) *padding = *(reinterpret_cast<const unsigned char *>(b + 4));
    if (id)      *id      = *(reinterpret_cast<const unsigned char *>(b + 5));
    return ntohl(*(int*)b);
}

int SSH::BigNumSize(const BigNum n) { return BigNumDataSize(n) + !(BigNumSignificantBits(n) % 8); }

BigNum SSH::ReadBigNum(BigNum n, const Serializable::Stream *i) {
    int n_len = 0;
    i->Ntohl(&n_len);
    return BigNumSetData(n, StringPiece(i->Get(n_len), n_len));
}

void SSH::WriteBigNum(const BigNum n, Serializable::Stream *o) {
    int n_len = BigNumDataSize(n);
    bool prepend_zero = !(BigNumSignificantBits(n) % 8);
    o->Htonl(n_len + prepend_zero);
    if (prepend_zero) o->Write8(char(0));
    BigNumGetData(n, o->Get(n_len));
}

string SSH::Serializable::ToString(std::mt19937 &g, unsigned *sequence_number) {
    if (sequence_number) (*sequence_number)++;
    string ret;
    ToString(&ret, g);
    return ret;
}

void SSH::Serializable::ToString(string *out, std::mt19937 &g) {
    out->resize(NextMultipleOf8(4 + SSH::BinaryPacketHeaderSize + Size()));
    return ToString(&(*out)[0], out->size(), g);
}

void SSH::Serializable::ToString(char *buf, int len, std::mt19937 &g) {
    unsigned char type = Type(), padding = len - SSH::BinaryPacketHeaderSize - Size();
    MutableStream os(buf, len);
    os.Htonl(len - 4);
    os.Write8(padding);
    os.Write8(type);
    Out(&os);
    memcpy(buf + len - padding, RandBytes(padding, g).data(), padding);
}

int SSH::MSG_KEXINIT::Size() const {
    return HeaderSize() + kex_algorithms.size() + server_host_key_algorithms.size() +
        encryption_algorithms_client_to_server.size() + encryption_algorithms_server_to_client.size() +
        mac_algorithms_client_to_server.size() + mac_algorithms_server_to_client.size() + 
        compression_algorithms_client_to_server.size() + compression_algorithms_server_to_client.size() +
        languages_client_to_server.size() + languages_server_to_client.size();
}

string SSH::MSG_KEXINIT::DebugString() const {
    string ret;
    StrAppend(&ret, "kex_algorithms: ",                          kex_algorithms.str(),                          "\n");
    StrAppend(&ret, "server_host_key_algorithms: ",              server_host_key_algorithms.str(),              "\n");
    StrAppend(&ret, "encryption_algorithms_client_to_server: ",  encryption_algorithms_client_to_server.str(),  "\n");
    StrAppend(&ret, "encryption_algorithms_server_to_client: ",  encryption_algorithms_server_to_client.str(),  "\n");
    StrAppend(&ret, "mac_algorithms_client_to_server: ",         mac_algorithms_client_to_server.str(),         "\n");
    StrAppend(&ret, "mac_algorithms_server_to_client: ",         mac_algorithms_server_to_client.str(),         "\n");
    StrAppend(&ret, "compression_algorithms_client_to_server: ", compression_algorithms_client_to_server.str(), "\n");
    StrAppend(&ret, "compression_algorithms_server_to_client: ", compression_algorithms_server_to_client.str(), "\n");
    StrAppend(&ret, "languages_client_to_server: ",              languages_client_to_server.str(),              "\n");
    StrAppend(&ret, "languages_server_to_client: ",              languages_server_to_client.str(),              "\n");
    StrAppend(&ret, "first_kex_packet_follows: ",                (int)first_kex_packet_follows,                 "\n");
    return ret;
}

void SSH::MSG_KEXINIT::Out(Serializable::Stream *o) const {
    o->String(cookie);
    o->BString(kex_algorithms);                             o->BString(server_host_key_algorithms);
    o->BString(encryption_algorithms_client_to_server);     o->BString(encryption_algorithms_server_to_client);
    o->BString(mac_algorithms_client_to_server);            o->BString(mac_algorithms_server_to_client);
    o->BString(compression_algorithms_client_to_server);    o->BString(compression_algorithms_server_to_client);
    o->BString(languages_client_to_server);                 o->BString(languages_server_to_client);
    o->Write8(first_kex_packet_follows);
    o->Write32(0);
}

int SSH::MSG_KEXINIT::In(const Serializable::Stream *i) {
    cookie = StringPiece(i->Get(16), 16);
    i->ReadString(&kex_algorithms);                             i->ReadString(&server_host_key_algorithms);
    i->ReadString(&encryption_algorithms_client_to_server);     i->ReadString(&encryption_algorithms_server_to_client);
    i->ReadString(&mac_algorithms_client_to_server);            i->ReadString(&mac_algorithms_server_to_client);
    i->ReadString(&compression_algorithms_client_to_server);    i->ReadString(&compression_algorithms_server_to_client);
    i->ReadString(&languages_client_to_server);                 i->ReadString(&languages_server_to_client);
    i->Read8(&first_kex_packet_follows);
    return i->Result();
}

int SSH::MSG_USERAUTH_REQUEST::Size() const {
    string mn = method_name.str();
    int ret = HeaderSize() + user_name.size() + service_name.size() + method_name.size();
    if      (mn == "publickey")            ret += 4*3 + 1 + algo_name.size() + secret.size() + sig.size();
    else if (mn == "password")             ret += 4*1 + 1 + secret.size();
    else if (mn == "keyboard-interactive") ret += 4*2;
    return ret;
};

void SSH::MSG_USERAUTH_REQUEST::Out(Serializable::Stream *o) const {
    o->BString(user_name);
    o->BString(service_name);
    o->BString(method_name);
    string mn = method_name.str();
    if      (mn == "publickey")            { o->Write8(static_cast<unsigned char>(1)); o->BString(algo_name); o->BString(secret); o->BString(sig); }
    else if (mn == "password")             { o->Write8(static_cast<unsigned char>(0)); o->BString(secret); }
    else if (mn == "keyboard-interactive") { o->BString(""); o->BString(""); }
}

int SSH::MSG_USERAUTH_INFO_REQUEST::Size() const {
    int ret = HeaderSize() + name.size() + instruction.size() + language.size();
    for (auto &p : prompt) ret += 4 + 1 + p.text.size();
    return ret;
}

int SSH::MSG_USERAUTH_INFO_REQUEST::In(const Serializable::Stream *i) {
    int num_prompts = 0;
    i->ReadString(&name);
    i->ReadString(&instruction);
    i->ReadString(&language);
    i->Htonl(&num_prompts);
    prompt.resize(num_prompts);
    for (auto &p : prompt) {
        i->ReadString(&p.text);
        i->Read8(&p.echo);
    }
    return i->Result();
}

int SSH::MSG_USERAUTH_INFO_RESPONSE::Size() const {
    int ret = HeaderSize();
    for (auto &r : response) ret += 4 + r.size();
    return ret;
}

void SSH::MSG_USERAUTH_INFO_RESPONSE::Out(Serializable::Stream *o) const {
    o->Htonl(static_cast<unsigned>(response.size()));
    for (auto &r : response) o->BString(r);
}

int SSH::MSG_CHANNEL_REQUEST::Size() const {
    int ret = HeaderSize() + request_type.size();
    string rt = request_type.str();
    if      (rt == "pty-req")       ret += 4*6 + term.size() + term_mode.size();
    else if (rt == "exec")          ret += 4*1 + term.size();
    else if (rt == "window-change") ret += 4*4;
    return ret;
}

void SSH::MSG_CHANNEL_REQUEST::Out(Serializable::Stream *o) const {
    o->Htonl(recipient_channel);
    o->BString(request_type);
    o->Write8(want_reply);
    string rt = request_type.str();
    if (rt == "pty-req") {
        o->BString(term);
        o->Htonl(width);
        o->Htonl(height);
        o->Htonl(pixel_width);
        o->Htonl(pixel_height);
        o->BString(term_mode);
    } else if (rt == "exec") {
        o->BString(term);
    } else if (rt == "window-change") {
        o->Htonl(width);
        o->Htonl(height);
        o->Htonl(pixel_width);
        o->Htonl(pixel_height);
    }
}

int SSH::DSSKey::In(const Serializable::Stream *i) {
    i->ReadString(&format_id);
    if (format_id.str() != "ssh-dss") { i->error = true; return -1; }
    p = ReadBigNum(p, i); 
    q = ReadBigNum(q, i); 
    g = ReadBigNum(g, i); 
    y = ReadBigNum(y, i); 
    return i->Result();
}

int SSH::DSSSignature::In(const Serializable::Stream *i) {
    StringPiece blob;
    i->ReadString(&format_id);
    i->ReadString(&blob);
    if (format_id.str() != "ssh-dss" || blob.size() != 40) { i->error = true; return -1; }
    r = BigNumSetData(r, StringPiece(blob.data(),      20));
    s = BigNumSetData(s, StringPiece(blob.data() + 20, 20));
    return i->Result();
}

/* SMTP */

void SMTP::HTMLMessage(const string& from, const string& to, const string& subject, const string& content, string *out) {
    static const char seperator[] = "XzYzZy";
    *out = StrCat("From: ", from, "\nTo: ", to, "\nSubject: ", subject,
                  "\nMIME-Version: 1.0\nContent-Type: multipart/alternative; boundary=\"", seperator, "\"\n\n");
    StrAppend(out, "--", seperator, "\nContent-type: text/html\n\n", content, "\n--", seperator, "--\n");
}

void SMTP::NativeSendmail(const string &message) {
#ifdef __linux__
    ProcessPipe smtp;
    const char *argv[] = { "/usr/bin/sendmail", "-i", "-t", 0 };
    if (smtp.Open(argv)) return;
    fwrite(message.c_str(), message.size(), 1, smtp.out);
#endif
}

string SMTP::EmailFrom(const string &message) {
    int lt, gt;
    string mail_from = HTTP::GrepHeaders(message.c_str(), 0, "From");
    if ((lt = mail_from.find("<"    )) == mail_from.npos ||
        (gt = mail_from.find(">", lt)) == mail_from.npos) FATAL("parse template from ", mail_from);
    return mail_from.substr(lt+1, gt-lt-1);
}

}; // namespace LFL
