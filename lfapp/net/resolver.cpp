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
#include "lfapp/network.h"
#include "lfapp/net/resolver.h"

#ifdef WIN32
#include <WinDNS.h>
#else
#include <netinet/in.h>
#endif

namespace LFL {
DEFINE_string(nameserver, "", "Default namesver");

int DNS::WriteRequest(unsigned short id, const string &querytext, unsigned short type, char *out, int len) {
  Serializable::MutableStream os(out, len);
  Header *hdr = reinterpret_cast<Header*>(os.Get(Header::size));
  memset(hdr, 0, Header::size);
  hdr->rd = 1;
  hdr->id = id;
  hdr->qdcount = htons(1);

  StringWordIter words(querytext, isdot);
  for (string word = words.NextString(); !word.empty(); word = words.NextString()) {
    CHECK_LT(word.size(), 64);
    os.Write8(uint8_t(word.size()));
    os.String(word);
  }
  os.Write8(char(0));

  os.Htons(type);                // QueryTypeClass.Type
  os.Htons(uint16_t(Class::IN)); // QueryTypeClass.QClass
  return os.error ? -1 : os.offset;
}

int DNS::ReadResponse(const char *buf, int bufsize, Response *res) {
  Serializable::ConstStream is(buf, bufsize);
  const Serializable::Stream *in = &is;
  const Header *hdr = reinterpret_cast<const Header*>(in->Get(Header::size));
  int qdcount = ntohs(hdr->qdcount), ancount = ntohs(hdr->ancount);
  int nscount = ntohs(hdr->nscount), arcount = ntohs(hdr->arcount), len;

  for (int i = 0; i < qdcount; i++) {
    Record out;
    if ((len = DNS::ReadString(in->Start(), in->Get(), in->End(), &out.question)) < 0 || !in->Get(len + 4)) return -1;
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
    if ((len = ReadString(in->Start(), in->Get(), in->End(), &rec.question)) < 0 || !in->Get(len)) return -1;

    in->Ntohs(&rec.type);
    in->Ntohs(&rec._class);
    in->Ntohs(&rec.ttl1);
    in->Ntohs(&rec.ttl2);
    in->Ntohs(&rrlen);

    if (rec._class == Class::IN && rec.type == Type::A) {
      if (rrlen != 4) return -1;
      in->Read32(&rec.addr);
    } else if (rec._class == Class::IN && (rec.type == Type::NS || rec.type == Type::CNAME)) {
      if ((len = ReadString(in->Start(), in->Get(), in->End(), &rec.answer)) != rrlen   || !in->Get(len)) return -1;
    } else if (rec._class == Class::IN && rec.type == Type::MX) {
      in->Ntohs(&rec.pref);
      if ((len = ReadString(in->Start(), in->Get(), in->End(), &rec.answer)) != rrlen-2 || !in->Get(len)) return -1;
    } else {
      ERROR("unhandled type=", rec.type, ", class=", rec._class);
      in->Get(rrlen);
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
      int offset = ntohs(*reinterpret_cast<const unsigned short*>(cur)) & ~(3<<14);
      if (!final) final = cur + 2;
      cur = start + offset - 2;
      if (cur < start || cur >= end) return ERRORv(-1, "OOB cur ", Void(start), " ", Void(cur), " ", Void(end));
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

/* Resolver */

bool Resolver::Nameserver::WriteResolveRequest(const Request &req) {
  INFO(c->Name(), ": resolve ", req.query);
  unsigned short id = RandKey(request_map);
  int len = DNS::WriteRequest(id, req.query, req.type, c->wb.begin(), c->wb.Capacity());
  if (c->WriteFlush(c->wb.begin(), len) != len) return false;
  request_map[id] = req;
  return true;
}

Resolver::Nameserver *Resolver::Connect(const vector<IPV4::Addr> &addrs, bool randomize) {
  Nameserver *ret = 0;
  int rand_connect_index = randomize ? Rand<int>(0, addrs.size()-1) : 0, ri = 0;
  for (auto &a : addrs) {
    if (ri++ == rand_connect_index) ret = Connect(a);
    else conn_available.push_back(a);
  }
  return ret;
}

Resolver::Nameserver *Resolver::Connect(IPV4::Addr addr) {
  unique_ptr<Nameserver> nameserver = make_unique<Nameserver>();
  Nameserver *ns = nameserver.get();
  ns->c = app->net->udp_client->PersistentConnection
    (IPV4::Text(addr, 53),
     [=](Connection *c, const char *cb, int cl) { HandleResponse(ns, reinterpret_cast<const DNS::Header*>(cb), cl); },
     [=](Connection *c)                         { Heartbeat(ns); }, 53);
  if (!ns->c) return nullptr;

  CHECK_EQ(addr, ns->c->addr);
  CHECK(!Contains(conn, addr));
  conn[addr] = move(nameserver);
  return ns;
}

void Resolver::HandleClosed(Nameserver *ns) {
  ERROR(ns->c->Name(), ": nameserver closed, timedout=", ns->timedout);
  unique_ptr<Nameserver> nameserver = Remove(&conn, ns->c->addr);
  if (ns->timedout && !remove_timedout_servers) conn_available.push_back(ns->c->addr);
  if (ns->request_map.size()) {
    bool alternatives = conn.size() || conn_available.size();
    for (auto &r : ns->request_map) if ((!alternatives || !QueueResolveRequest(r.second)) && r.second.cb) r.second.cb(-1, 0);
  }
}

void Resolver::HandleResponse(Nameserver *ns, const DNS::Header *hdr, int len) {
  if (!hdr) return HandleClosed(ns);
  auto rmiter = ns->request_map.find(hdr->id);
  if (rmiter == ns->request_map.end()) return ERROR(ns->c->Name(), ": unknown DNS reply id=", hdr->id, ", len=", len);
  Resolver::Request req = rmiter->second;
  ns->request_map.erase(rmiter);

  DNS::Response res;
  if (DNS::ReadResponse(reinterpret_cast<const char*>(hdr), len, &res)) return ERROR(ns->c->Name(), ": parse ");
  if (FLAGS_dns_dump) INFO(ns->c->Name(), ": ", res.DebugString());

  vector<IPV4::Addr> results;
  for (int i=0; i<res.A.size(); i++) if (res.A[i].type == DNS::Type::A) results.push_back(res.A[i].addr);
  IPV4::Addr ipv4_addr = results.size() ? results[Rand<int>(0, results.size()-1)] : -1;
  INFO(ns->c->Name(), ": resolved ", req.query, " to ", IPV4::Text(ipv4_addr));

  if (req.cb) req.cb(ipv4_addr, &res);
  DequeueTo(ns);
}

void Resolver::Heartbeat(Nameserver *ns) {
  Time now = Now();
  if (auto_disconnect_seconds && !ns->request_map.size() && !queue.size() &&
      (ns->c->rt + Seconds(auto_disconnect_seconds)) <= now) return ns->Timedout();

  static const Time retry_interval(1000);
  static const int retry_max = 5;
  for (auto r = ns->request_map.begin(); r != ns->request_map.end(); /**/) {
    if (r->second.stamp + retry_interval >= now) { r++; continue; }
    Resolver::Request req = r->second;
    r = ns->request_map.erase(r);
    INFO(ns->c->Name(), ": timeout resolving ", req.query, " (retrys=", req.retrys, ")");
    if ((req.retrys++ >= retry_max || !QueueResolveRequest(req)) && req.cb) req.cb(-1, 0);
  }
  DequeueTo(ns);
}

void Resolver::DequeueTo(Nameserver *ns) {
  while (queue.size() && ns->request_map.size() < max_outstanding_per_ns) {
    Resolver::Request req = PopBack(queue);
    if (!QueueResolveRequest(req) && req.cb) req.cb(-1, 0);
  }
}

void Resolver::NSLookup(const string &host, const ResponseCB &cb) {
  IPV4::Addr addr;
  if ((addr = IPV4::Parse(host)) != INADDR_NONE) cb(addr, 0);
  else if (!QueueResolveRequest(Request(host, DNS::Type::A, cb))) cb(-1, 0);
}

bool Resolver::QueueResolveRequest(const Request &req) {
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
  IPV4::Addr ipv4_addr = SystemNetwork::GetHostByName(req.query);
  INFO("resolved ", req.query, " to ", IPV4::Text(ipv4_addr));
  req.cb(ipv4_addr, NULL);
  return true;
#else
  if (!conn.size() && !conn_available.size() && !HandleNoConnections()) return ERRORv(false, "resolve called with no conns");
  for (int reread_nameservers=0; /**/; reread_nameservers++) {
    CHECK_LT(reread_nameservers, 10);
    CHECK(conn.size() || conn_available.size());
    Nameserver *ns = NextAvailableNameserver(req.retrys);
    Request outreq(req.query, req.type, req.cb, req.retrys);
    if (!ns) queue.push_back(outreq);
    else if (!ns->WriteResolveRequest(outreq)) {
      dynamic_cast<UDPClient::PersistentConnectionHandler*>(ns->c->handler.get())->responseCB = UDPClient::ResponseCB();
      ns->c->SetError();
      HandleClosed(ns);
      bool retried = conn.size() > 1 || conn_available.size() || HandleNoConnections();
      if (retried) continue;
      else return ERRORv(false, "last nameserver failed");
    }
    return true;
  }
#endif
}

Resolver::Nameserver *Resolver::NextAvailableNameserver(bool skip_first) {
  Nameserver *ns = 0;
  auto ni = conn.begin();
  if (skip_first || ni == conn.end() || ni->second->request_map.size() >= max_outstanding_per_ns) {
    auto i = ni;
    if (i != conn.end()) ++i;
    for (/**/; i != conn.end(); ++i) if (i->second->request_map.size() < max_outstanding_per_ns) break;
    if (i != conn.end()) ns = i->second.get();
    else if (conn_available.size()) ns = Connect(PopBack(conn_available));
  }
  if (!ns && ni != conn.end() && ni->second->request_map.size() < max_outstanding_per_ns) ns = ni->second.get();
  return ns;
}

/* System Resolver */

bool SystemResolver::HandleNoConnections() {
  bool first = nameservers.empty(), ret = false;
  vector<IPV4::Addr> last_nameservers;
  swap(nameservers, last_nameservers);
  if (FLAGS_nameserver.empty()) GetNameservers(&nameservers);
  else IPV4::ParseCSV(FLAGS_nameserver, &nameservers);
  if (nameservers == last_nameservers) return false;
  vector<IPV4::Addr> connect_nameservers = nameservers; 
  if (!first) FilterValues(&connect_nameservers, conn);
  for (auto &n : connect_nameservers) if (Connect(n)) ret = true;
  return ret;
}

void SystemResolver::GetNameservers(vector<IPV4::Addr> *nameservers) {
  nameservers->clear();
#ifdef LFL_ANDROID
  return;
#endif
#ifdef _WIN32
  char buf[512];
  DWORD size=sizeof(buf);
  IP4_ARRAY *IP = (IP4_ARRAY*)buf;
  if (DnsQueryConfig(DnsConfigDnsServerList, 0, 0, 0, IP, &size) || !IP->AddrCount) return ERROR("no default nameserver ", GetLastError());
  nameservers->push_back(IP->AddrArray[0]);
#else
  LocalFile file("/etc/resolv.conf", "r");
  if (!file.Opened()) return;
  NextRecordReader nr(&file);
  for (const char *line = nr.NextLine(); line; line = nr.NextLine()) {
    StringWordIter words(line);
    if (words.NextString() != "nameserver") continue;
    IPV4::Addr addr = IPV4::Parse(words.NextString());
    if (addr != -1) nameservers->push_back(addr);
  }
#endif
}

/* Recursive Resolver */

void RecursiveResolver::ConnectoToRootServers() {
  vector<IPV4::Addr> addrs;
# define XX(x)
# define YY(x) addrs.push_back(IPV4::Parse(x));
# include "lfapp/namedroot.h"
  root.resolver.Connect(addrs);
}

bool RecursiveResolver::StartResolveRequest(Request *req) {
  AuthorityTreeNode *node = GetAuthorityTreeNode(req->query, false);
  req->seen_authority.insert(node);

  DNS::Response *cached = 0;
  AuthorityTreeNode::Cache::iterator ci;
  if      (req->type == DNS::Type::A  && (ci = node->Acache .find(req->query)) != node->Acache .end()) cached = ci->second.get();
  else if (req->type == DNS::Type::MX && (ci = node->MXcache.find(req->query)) != node->MXcache.end()) cached = ci->second.get();
  if (cached) {
    IPV4::Addr addr = cached->A.size() ? cached->A[Rand<int>(0, cached->A.size()-1)].addr : -1;
    INFO("RecursiveResolver found ", req->query, " = ", IPV4::Text(addr), " in cache=", node->authority_domain);
    app->RunInMainThread(bind(&RecursiveResolver::Complete, this, req, addr, cached));
    return true;
  }

  bool ret = QueueResolveRequest(node, req);
  if (ret) queries_requested++;
  return ret;
}

bool RecursiveResolver::QueueResolveRequest(AuthorityTreeNode *node, Request *req) {
  Resolver::Request nsreq(req->query, req->type, [=](IPV4::Addr A, DNS::Response *R) { HandleResponse(req, A, R, 0); });
  return node->resolver.QueueResolveRequest(nsreq);
}

RecursiveResolver::AuthorityTreeNode *RecursiveResolver::GetAuthorityTreeNode(const string &query, bool create) {
  vector<string> q;
  Split(query, isdot, &q);
  AuthorityTreeNode *node = &root;

  for (int i = q.size()-1; i >= 0; --i) {
    auto it = node->child.find(q[i]);
    if (it != node->child.end()) { node = it->second.get(); continue; }
    if (!create) break;

    unique_ptr<AuthorityTreeNode> add = make_unique<AuthorityTreeNode>();
    add->authority_domain = Join(q, ".", i, q.size()) + ".";
    add->depth = node->depth + 1;
    node = (node->child[q[i]] = move(add)).get();
  }

  if (FLAGS_dns_dump) INFO("GetAuthorityTreeNode(", query, ", ", create, ") = ", node->authority_domain);
  return node;
}

void RecursiveResolver::HandleResponse(Request *req, IPV4::Addr addr, DNS::Response *res, vector<DNS::Response> *subres) {
  if (FLAGS_dns_dump) INFO("RecursiveResolver::Response ", int(addr), " ", Void(res), " " , Void(subres));

  if (addr != -1) {
    if (addr == 0 && !req->parent_request && res) {
      if (!req->missing_answer) {
        req->missing_answer = 1;
        req->answer.clear();
        req->answer.push_back(*res);
        DNS::AnswerMap extra;
        DNS::MakeAnswerMap(res->E, &extra);
        int new_child_requests = ResolveAnyMissingAnswers(req, res->A, &extra);
        if (new_child_requests) return;
      } else if (subres) {
        for (int i = 1; i < subres->size(); ++i)
          res->E.insert(res->E.end(), (*subres)[i].A.begin(), (*subres)[i].A.end());
      }
    }

    AuthorityTreeNode *node=0;
    if (res && (req->type == DNS::Type::A || req->type == DNS::Type::MX)) {
      node = GetAuthorityTreeNode(req->query, false);
      AuthorityTreeNode::Cache *cache = (req->type == DNS::Type::A) ? &node->Acache : &node->MXcache;
      if (Contains(*cache, req->query)) { ERROR("cache collision ", (*cache)[req->query]->DebugString(), " versus ", res->DebugString()); node=0; }
      else (*cache)[req->query] = make_unique<DNS::Response>(*res);
    }
    INFO("RecursiveResolver resolved ", req->query, " to ", IPV4::Text(addr), " (cached=", node?node->authority_domain:"<NULL>", ")");
    Complete(req, addr, res);
    queries_completed++;
    return;
  }

  DNS::AnswerMap authority_zone, extra;
  if (res) {
    DNS::MakeAnswerMap(res->E, &extra);
    for (int i = 1; subres && i < subres->size(); ++i) DNS::MakeAnswerMap((*subres)[i].A, &extra);
    DNS::MakeAnswerMap(res->NS, extra, DNS::Type::NS, &authority_zone);
    if (!authority_zone.size() && req->Ancestors() < 5 && !subres) {
      int new_child_requests = ResolveAnyMissingAnswers(req, res->NS, 0);
      if (new_child_requests) { req->answer.clear(); req->answer.push_back(*res); return; }
    }
  }

  bool ret = false;
  if (authority_zone.size() != 1) ERROR("authority_zone.size() ", authority_zone.size());
  for (auto i = authority_zone.begin(); i != authority_zone.end(); ++i) {
    AuthorityTreeNode *node = GetAuthorityTreeNode(i->first, true);
    CHECK_EQ(i->first, node->authority_domain);
    if (!node->authority.Q.size()) {
      node->authority = *res;
      node->resolver.Connect(i->second);
    } else ERROR("AuthorityTreeNode collision ", node->authority.DebugString(), " versus ", res->DebugString());

    if (Contains(req->seen_authority, node)) { ERROR("RecursiveResolver loop?"); continue; }
    ret = QueueResolveRequest(node, req); 
    req->seen_authority.insert(node);
    break;
  }

  if (!ret) {
    INFO("RecursiveResolver failed to resolve ", req->query);
    Complete(req, -1, res);
    queries_completed++;
  }
}

int RecursiveResolver::ResolveAnyMissingAnswers(Request *req, const vector<DNS::Record> &R, const DNS::AnswerMap *answer) {
  int start_requests = req->child_request.size(), start_pending_requests = req->pending_child_request.size();
  for (auto e = R.begin(); e != R.end(); ++e) {
    if (e->answer.empty() || (answer && Contains(*answer, e->answer))) continue;
    StartChildResolve(req, new Request(e->answer, DNS::Type::A, Resolver::ResponseCB(), req));
  }
  int new_requests = req->child_request.size() - start_requests, new_pending_requests = req->pending_child_request.size() - start_pending_requests;
  if (new_requests || new_pending_requests) INFO("RecursiveResolver ", req->query, " spawned ", new_requests, " subqueries, queued ", new_pending_requests);
  return new_requests + new_pending_requests;
}

void RecursiveResolver::Complete(Request *req, IPV4::Addr addr, DNS::Response *res) {
  if (req->parent_request) HandleChildResponse(req->parent_request, req, res);
  if (req->cb) req->cb(addr, res);
  delete req;
}

void RecursiveResolver::HandleChildResponse(Request *req, Request *subreq, DNS::Response *res) {
  if (res) req->answer.push_back(*res);
  req->child_request.erase(subreq);
  if (req->child_request.size()) return;
  if (!req->pending_child_request.size()) {
    INFO(req->query, ": subrequests finished, ma=", req->missing_answer, ", as=", req->answer.size());
    return HandleResponse(req, req->missing_answer ? 0 : -1, &req->answer[0], &req->answer);
  }
  subreq = *req->pending_child_request.begin();
  req->pending_child_request.erase(req->pending_child_request.begin());
  StartChildResolve(req, subreq);
}

void RecursiveResolver::StartChildResolve(Request *req, Request *subreq) {
  if (req->child_request.size()) { req->pending_child_request.insert(subreq); return; }
  req->child_request.insert(subreq);
  StartResolveRequest(subreq);
}

}; // namespace LFL
