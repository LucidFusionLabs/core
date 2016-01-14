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
#include "lfapp/resolver.h"

#ifdef WIN32
#include <WinDNS.h>
#else
#include <netinet/in.h>
#endif

namespace LFL {
unsigned short Resolver::Nameserver::GetNextID() const {
  unsigned short id, tries=0, max_tries=100;
  for (id = rand(); Contains(request_map, id) && tries<max_tries; id = rand(), tries++) { /**/ }
  CHECK_LT(tries, max_tries);
  return id;
}

bool Resolver::Nameserver::WriteResolveRequest(const Request &req) {
  INFO(c->Name(), ": resolve ", req.query);
  int len;
  unsigned short id = GetNextID();
  if ((len = DNS::WriteRequest(id, req.query, req.type, c->wb.begin(), c->wb.Capacity())) < 0) return false;
  if (c->WriteFlush(c->wb.begin(), len) != len) return false;
  request_map[id] = req;
  return true;
}

void Resolver::Nameserver::HandleResponse(Connection *cin, DNS::Header *hdr, int len) {
  CHECK_EQ(c, cin);
  if (!hdr) {
    ERROR(c->Name(), ": nameserver closed, timedout=", timedout);
    CHECK_EQ(parent->conn.erase(c->addr), 1);
    if (timedout) parent->conn_available.push_back(c->addr);
    if (request_map.size()) {
      bool alternatives = parent->conn.size() || parent->conn_available.size();
      for (auto &r : request_map)
        if ((!alternatives || !parent->QueueResolveRequest(r.second)) && r.second.cb) r.second.cb(-1, 0);
    }
    delete this;
    return;
  }

  auto rmiter = request_map.find(hdr->id);
  if (rmiter == request_map.end()) { ERROR(c->Name(), ": unknown DNS reply id=", hdr->id, ", len=", len); return; }
  Resolver::Request req = rmiter->second;
  request_map.erase(rmiter);

  DNS::Response res;
  if (DNS::ReadResponse((const char *)hdr, len, &res)) { ERROR(c->Name(), ": parse "); return; }
  if (FLAGS_dns_dump) INFO(c->Name(), ": ", res.DebugString());

  if (req.cb) {
    vector<IPV4::Addr> results;
    for (int i=0; i<res.A.size(); i++) if (res.A[i].type == DNS::Type::A) results.push_back(res.A[i].addr);
    IPV4::Addr ipv4_addr = results.size() ? results[Rand<int>(0, results.size()-1)] : -1;
    INFO(c->Name(), ": resolved ", req.query, " to ", IPV4::Text(ipv4_addr));
    req.cb(ipv4_addr, &res);
  }
  Dequeue();
}

void Resolver::Nameserver::Heartbeat() {
  Time now = Now();
  if (parent->auto_disconnect_seconds && !request_map.size() && !parent->queue.size() &&
      (c->rt + Seconds(parent->auto_disconnect_seconds)) <= now)
  { timedout=true; c->SetError(); INFO(c->Name(), ": nameserver timeout"); return; }

  static const Time retry_interval(1000);
  static const int retry_max = 5;
  for (auto r = request_map.begin(); r != request_map.end(); /**/) {
    if (r->second.stamp + retry_interval >= now) { r++; continue; }
    Resolver::Request req = r->second;
    request_map.erase(r++);

    INFO(req.ns->c->Name(), ": timeout resolving ", req.query, " (retrys=", req.retrys, ")");
    if ((req.retrys++ >= retry_max || !parent->QueueResolveRequest(req)) && req.cb) req.cb(-1, 0);
  }
  Dequeue();
}

void Resolver::Nameserver::Dequeue() {
  while (parent->queue.size() && request_map.size() < parent->max_outstanding_per_ns) {
    Resolver::Request req = PopBack(parent->queue);
    if (!parent->QueueResolveRequest(req) && req.cb) req.cb(-1, 0);
  }
}

Resolver::Nameserver *Resolver::Connect(IPV4::Addr addr) {
  CHECK(!Contains(conn, addr));
  Nameserver *ns = new Nameserver(this, addr);
  if (!ns->c) { delete ns; return 0; }
  CHECK_EQ(addr, ns->c->addr);
  conn[addr] = ns;
  return ns;
}

Resolver::Nameserver *Resolver::Connect(const vector<IPV4::Addr> &addrs) {
  Nameserver *ret = 0;
  static bool randomize = false;
  int rand_connect_index = randomize ? Rand<int>(0, addrs.size()-1) : 0, ri = 0;
  for (auto &a : addrs) {
    if (ri++ == rand_connect_index) ret = Connect(a);
    else conn_available.push_back(a);
  }
  return ret;
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
  if (!conn.size() && !conn_available.size()) { ERROR("resolve called with no conns"); return false; } 

  Nameserver *ns = 0; // Choose a nameserver
  auto ni = conn.begin();
  if (req.retrys || ni == conn.end() || ni->second->request_map.size() >= max_outstanding_per_ns) {
    auto i = ni;
    if (i != conn.end()) ++i;
    for (/**/; i != conn.end(); ++i) if (i->second->request_map.size() < max_outstanding_per_ns) break;
    if (i != conn.end()) ns = i->second;
    else if (conn_available.size()) {
      ns = Connect(conn_available.back());
      conn_available.pop_back();
    }
  }
  if (!ns && ni != conn.end() && ni->second->request_map.size() < max_outstanding_per_ns) ns = ni->second;

  // Resolve or queue
  Request outreq(ns, req.query, req.type, req.cb, req.retrys);
  if (ns) return ns->WriteResolveRequest(outreq);
  queue.push_back(outreq);
  return true;
#endif
}

void Resolver::GetDefaultNameservers(vector<IPV4::Addr> *nameservers) {
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
    if (IterNextString(&words) != "nameserver") continue;
    nameservers->push_back(IPV4::Parse(IterNextString(&words)));
  }
#endif
}

/* Recursive Resolver */

void RecursiveResolver::Request::StartChildResolve(Request *subreq) {
  if (child_request.size()) { pending_child_request.insert(subreq); return; }
  child_request.insert(subreq);
  resolver->StartResolveRequest(subreq);
}

void RecursiveResolver::Request::HandleChildResponse(Request *subreq, DNS::Response *res) {
  if (res) answer.push_back(*res);
  child_request.erase(subreq);
  if (child_request.size()) return;
  if (!pending_child_request.size()) {
    INFO(query, ": subrequests finished, ma=", missing_answer, ", as=", answer.size());
    return resolver->HandleRequestResponse(this, missing_answer ? 0 : -1, &answer[0], &answer);
  }
  subreq = *pending_child_request.begin();
  pending_child_request.erase(pending_child_request.begin());
  StartChildResolve(subreq);
}

void RecursiveResolver::Request::Complete(IPV4::Addr addr, DNS::Response *res) {
  if (parent_request) parent_request->HandleChildResponse(this, res);
  if (cb) cb(addr, res);
  delete this;
}

RecursiveResolver::AuthorityTreeNode *RecursiveResolver::GetAuthorityTreeNode(const string &query, bool create) {
  AuthorityTreeNode *node = &root;
  vector<string> q;
  Split(query, isdot, &q);

  for (int i = q.size()-1; i >= 0; --i) {
    auto it = node->child.find(q[i]);
    if (it != node->child.end()) { node = it->second; continue; }
    if (!create) break;

    AuthorityTreeNode *ret = new AuthorityTreeNode();
    ret->authority_domain = Join(q, ".", i, q.size()) + ".";
    ret->depth = node->depth + 1;
    node->child[q[i]] = ret;
    node = ret;
  }

  if (FLAGS_dns_dump) INFO("GetAuthorityTreeNode(", query, ", ", create, ") = ", node->authority_domain);
  return node;
}

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
  req->resolver = this;

  DNS::Response *cached = 0;
  AuthorityTreeNode::Cache::iterator ci;
  if      (req->type == DNS::Type::A  && (ci = node->Acache .find(req->query)) != node->Acache .end()) cached = ci->second;
  else if (req->type == DNS::Type::MX && (ci = node->MXcache.find(req->query)) != node->MXcache.end()) cached = ci->second;
  if (cached) {
    IPV4::Addr addr = cached->A.size() ? cached->A[Rand<int>(0, cached->A.size()-1)].addr : -1;
    INFO("RecursiveResolver found ", req->query, " = ", IPV4::Text(addr), " in cache=", node->authority_domain);
    RunInMainThread(new Callback(bind(&Request::Complete, req, addr, cached)));
    return true;
  }

  Resolver::Request nsreq(req->query, req->type, bind(&Request::ResponseCB, req, _1, _2));
  bool ret = node->resolver.QueueResolveRequest(nsreq);
  if (ret) queries_requested++;
  return ret;
}

int RecursiveResolver::ResolveAnyMissingAnswers(Request *req, const vector<DNS::Record> &R, const DNS::AnswerMap *answer) {
  int start_requests = req->child_request.size(), start_pending_requests = req->pending_child_request.size();
  for (auto e = R.begin(); e != R.end(); ++e) {
    if (e->answer.empty() || (answer && Contains(*answer, e->answer))) continue;
    req->StartChildResolve(new Request(e->answer, DNS::Type::A, Resolver::ResponseCB(), req));
  }
  int new_requests = req->child_request.size() - start_requests, new_pending_requests = req->pending_child_request.size() - start_pending_requests;
  if (new_requests || new_pending_requests) INFO("RecursiveResolver ", req->query, " spawned ", new_requests, " subqueries, queued ", new_pending_requests);
  return new_requests + new_pending_requests;
}

void RecursiveResolver::HandleRequestResponse(Request *req, IPV4::Addr addr, DNS::Response *res, vector<DNS::Response> *subres) {
  if (FLAGS_dns_dump) INFO("RecursiveResolver::Response ", (int)addr, " ", res, " " , subres);

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
      else (*cache)[req->query] = new DNS::Response(*res);
    }
    INFO("RecursiveResolver resolved ", req->query, " to ", IPV4::Text(addr), " (cached=", node?node->authority_domain:"<NULL>", ")");
    req->Complete(addr, res);
    queries_completed++;
    return;
  }

  bool ret = false;
  DNS::AnswerMap extra, authority_zone;
  if (res) {
    DNS::MakeAnswerMap(res->E, &extra);
    for (int i = 1; subres && i < subres->size(); ++i) DNS::MakeAnswerMap((*subres)[i].A, &extra);
    DNS::MakeAnswerMap(res->NS, extra, DNS::Type::NS, &authority_zone);
    if (!authority_zone.size() && req->Ancestors() < 5 && !subres) {
      int new_child_requests = ResolveAnyMissingAnswers(req, res->NS, 0);
      if (new_child_requests) { req->answer.clear(); req->answer.push_back(*res); return; }
    }
  }

  if (authority_zone.size() != 1) ERROR("authority_zone.size() ", authority_zone.size());
  for (auto i = authority_zone.begin(); i != authority_zone.end(); ++i) {
    AuthorityTreeNode *node = GetAuthorityTreeNode(i->first, true);
    CHECK_EQ(i->first, node->authority_domain);
    if (!node->authority.Q.size()) {
      node->authority = *res;
      node->resolver.Connect(i->second);
    } else ERROR("AuthorityTreeNode collision ", node->authority.DebugString(), " versus ", res->DebugString());

    if (Contains(req->seen_authority, node)) { ERROR("RecursiveResolver loop?"); continue; }
    ret = node->resolver.QueueResolveRequest(Resolver::Request(req->query, req->type, bind(&Request::ResponseCB, req, _1, _2)));
    req->seen_authority.insert(node);
    break;
  }

  if (!ret) {
    INFO("RecursiveResolver failed to resolve ", req->query);
    req->Complete(-1, res);
    queries_completed++;
  }
}

}; // namespace LFL
