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

namespace LFL {

void Resolver::Reset() {
    for (NameserverMap::iterator i = conn.begin(); i != conn.end(); ++i) delete i->second;
    conn.clear();
}

bool Resolver::Connected() {
    for (NameserverMap::const_iterator i = conn.begin(); i != conn.end(); ++i)
        if (i->second->c->state == Connection::Connected) return true;
    return false;
}

Resolver::Nameserver *Resolver::Connect(IPV4::Addr addr) {
    Nameserver *ns = new Nameserver(this, addr);
    if (!ns->c) { delete ns; return 0; }
    CHECK_EQ(addr, ns->c->addr);
    conn[addr] = ns;
    return ns;
}

Resolver::Nameserver *Resolver::Connect(const vector<IPV4::Addr> &addrs) {
    static bool randomize = false;
    Nameserver *ret = 0;
    int rand_connect_index = randomize ? Rand<int>(0, addrs.size()-1) : 0, ri = 0;
    for (vector<IPV4::Addr>::const_iterator i = addrs.begin(); i != addrs.end(); ++i, ++ri) {
        if (ri == rand_connect_index) ret = Connect(*i);
        else conn_available.push_back(*i);
    } return ret;
}

bool Resolver::Resolve(const Request &req) {
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
    IPV4::Addr ipv4_addr = SystemNetwork::GetHostByName(req.query);
    INFO("resolved ", req.query, " to ", IPV4::Text(ipv4_addr));
    req.cb(ipv4_addr, NULL);
    return true;
#else
    if (!conn.size() && !conn_available.size()) { ERROR("resolve called with no conns"); return false; } 

    Nameserver *ns = 0; // Choose a nameserver
    NameserverMap::iterator ni = conn.begin();
    if (req.retrys || ni == conn.end() || ni->second->requestMap.size() >= max_outstanding_per_ns) {
        NameserverMap::iterator i = ni;
        if (i != conn.end()) ++i;
        for (/**/; i != conn.end(); ++i) if (i->second->requestMap.size() < max_outstanding_per_ns) break;
        if (i != conn.end()) ns = i->second;
        else if (conn_available.size()) {
            ns = Connect(conn_available.back());
            conn_available.pop_back();
        }
    }
    if (!ns && ni != conn.end() && ni->second->requestMap.size() < max_outstanding_per_ns) ns = ni->second;

    // Resolve or queue
    Request outreq(ns, req.query, req.type, req.cb, req.retrys);
    if (ns) return ns->Resolve(outreq);
    queue.push_back(outreq);
    return true;
#endif
}

void Resolver::DefaultNameserver(vector<IPV4::Addr> *nameservers) {
    nameservers->clear();
#ifdef LFL_ANDROID
    return;
#endif
#ifdef _WIN32
    IP4_ARRAY IP; DWORD size=sizeof(IP4_ARRAY);
    if (DnsQueryConfig(DnsConfigDnsServerList, 0, 0, 0, &IP, &size) || !IP.AddrCount) return;
    nameservers->push_back(IP.AddrArray[0]);
#else
    LocalFile file("/etc/resolv.conf", "r");
    if (!file.Opened()) return;

    for (const char *line = file.NextLine(); line; line = file.NextLine()) {
        StringWordIter words(line);
        if (IterNextString(&words) != "nameserver") continue;
        nameservers->push_back(IPV4::Parse(IterNextString(&words)));
    }
#endif
}

/* Resolver::Nameserver */

bool Resolver::Nameserver::Resolve(const Request &req) {
    int len; unsigned short id = NextID();
    if ((len = DNS::WriteRequest(id, req.query, req.type, c->wb, sizeof(c->wb))) < 0) return false;
    if (c->WriteFlush(c->wb, len) != len) return false;
    requestMap[id] = req;
    return true;
}

void Resolver::Nameserver::Response(Connection *cin, DNS::Header *hdr, int len) {
    CHECK_EQ(c, cin);
    if (!hdr) {
        ERROR(c->Name(), ": nameserver closed, timedout=", timedout);
        CHECK_EQ(parent->conn.erase(c->addr), 1);
        if (timedout) parent->conn_available.push_back(c->addr);
        if (requestMap.size()) {
            bool alternatives = parent->conn.size() || parent->conn_available.size();
            for (RequestMap::iterator i = requestMap.begin(); i != requestMap.end(); ++i) {
                const Resolver::Request &req = i->second;
                if (!alternatives || !parent->Resolve(req)) { if (req.cb) req.cb(-1, 0); }
            }
        }
        delete this;
        return;
    }

    RequestMap::iterator rmiter = requestMap.find(hdr->id);
    if (rmiter == requestMap.end()) { ERROR(c->Name(), ": unknown DNS reply id=", hdr->id, ", len=", len); return; }
    Resolver::Request req = rmiter->second;
    requestMap.erase(rmiter);

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
    if (parent->auto_disconnect_seconds && !requestMap.size() && !parent->queue.size() && (c->rt + Seconds(parent->auto_disconnect_seconds)) <= now)
    { timedout=true; c->SetError(); INFO(c->Name(), ": nameserver timeout"); return; }

    static const Time retry_interval(1000);
    static const int retry_max = 5;
    for (RequestMap::iterator rmiter = requestMap.begin(); rmiter != requestMap.end(); /**/) {
        if ((*rmiter).second.stamp + retry_interval >= now) { rmiter++; continue; }
        Resolver::Request req = (*rmiter).second;
        requestMap.erase(rmiter++);

        INFO(req.ns->c->Name(), ": timeout resolving ", req.query, " (retrys=", req.retrys, ")");
        if (req.retrys++ >= retry_max || !parent->Resolve(req)) { if (req.cb) req.cb(-1, 0); }
    }
    Dequeue();
}

void Resolver::Nameserver::Dequeue() {
    while (parent->queue.size() && requestMap.size() < parent->max_outstanding_per_ns) {
        Resolver::Request req = parent->queue.back();
        parent->queue.pop_back();
        if (!parent->Resolve(req)) { if (req.cb) req.cb(-1, 0); }
    }
}

/* Recursive Resolver */

RecursiveResolver::RecursiveResolver() : queries_requested(0), queries_completed(0) {
    vector<IPV4::Addr> addrs;
#   define XX(x)
#   define YY(x) addrs.push_back(IPV4::Parse(x));
#   include "lfapp/namedroot.h"
    root.resolver.Connect(addrs);
}

bool RecursiveResolver::Resolve(Request *req) {
    AuthorityTreeNode *node = GetAuthorityTreeNode(req->query, false);
    req->seen_authority.insert((void*)node);
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
    bool ret = node->resolver.Resolve(nsreq);
    if (ret) queries_requested++;
    return ret;
}

int RecursiveResolver::ResolveMissing(Request *req, const vector<DNS::Record> &R, const DNS::AnswerMap *answer) {
    int start_requests = req->child_request.size(), start_pending_requests = req->pending_child_request.size();
    for (vector<DNS::Record>::const_iterator e = R.begin(); e != R.end(); ++e) {
        if (e->answer.empty() || (answer && Contains(*answer, e->answer))) continue;
        req->ChildResolve(new Request(e->answer, DNS::Type::A, Resolver::ResponseCB(), req));
    }
    int new_requests = req->child_request.size() - start_requests, new_pending_requests = req->pending_child_request.size() - start_pending_requests;
    if (new_requests || new_pending_requests) INFO("RecursiveResolver ", req->query, " spawned ", new_requests, " subqueries, queued ", new_pending_requests);
    return new_requests + new_pending_requests;
}

void RecursiveResolver::Response(Request *req, IPV4::Addr addr, DNS::Response *res, vector<DNS::Response> *subres) {
    if (FLAGS_dns_dump) INFO("RecursiveResolver::Response ", (int)addr, " ", res, " " , subres);
    if (addr != -1) {
        if (addr == 0 && !req->parent_request && res) {
            if (!req->missing_answer) {
                req->missing_answer = 1;
                req->answer.clear();
                req->answer.push_back(*res);
                DNS::AnswerMap extra;
                DNS::MakeAnswerMap(res->E, &extra);
                int new_child_requests = ResolveMissing(req, res->A, &extra);
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
            int new_child_requests = ResolveMissing(req, res->NS, 0);
            if (new_child_requests) { req->answer.clear(); req->answer.push_back(*res); return; }
        }
    }
    if (authority_zone.size() != 1) ERROR("authority_zone.size() ", authority_zone.size());
    for (DNS::AnswerMap::const_iterator i = authority_zone.begin(); i != authority_zone.end(); ++i) {
        AuthorityTreeNode *node = GetAuthorityTreeNode(i->first, true);
        CHECK_EQ(i->first, node->authority_domain);
        if (!node->authority.Q.size()) {
            node->authority = *res;
            node->resolver.Connect(i->second);
        } else ERROR("AuthorityTreeNode collision ", node->authority.DebugString(), " versus ", res->DebugString());

        if (Contains(req->seen_authority, (void*)node)) { ERROR("RecursiveResolver loop?"); continue; }
        ret = node->resolver.Resolve(Resolver::Request(req->query, req->type, bind(&Request::ResponseCB, req, _1, _2)));
        req->seen_authority.insert(node);
        break;
    }
    if (!ret) {
        INFO("RecursiveResolver failed to resolve ", req->query);
        req->Complete(-1, res);
        queries_completed++;
    }
}

RecursiveResolver::AuthorityTreeNode *RecursiveResolver::GetAuthorityTreeNode(const string &query, bool create) {
    AuthorityTreeNode *node = &root;
    vector<string> q;
    Split(query, isdot, &q);
    for (int i = q.size()-1; i >= 0; --i) {
        AuthorityTreeNode::Children::iterator it = node->child.find(q[i]);
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

void RecursiveResolver::Request::ChildResolve(Request *subreq) {
    if (child_request.size()) { pending_child_request.insert(subreq); return; }
    child_request.insert(subreq);
    resolver->Resolve(subreq);
}

void RecursiveResolver::Request::ChildResponse(Request *subreq, DNS::Response *res) {
    if (res) answer.push_back(*res);
    child_request.erase(subreq);
    if (child_request.size()) return;
    if (!pending_child_request.size()) {
        INFO(query, ": subrequests finished, ma=", missing_answer, ", as=", answer.size());
        return resolver->Response(this, missing_answer ? 0 : -1, &answer[0], &answer);
    }
    subreq = *pending_child_request.begin();
    pending_child_request.erase(pending_child_request.begin());
    ChildResolve(subreq);
}

void RecursiveResolver::Request::Complete(IPV4::Addr addr, DNS::Response *res) {
    if (parent_request) parent_request->ChildResponse(this, res);
    if (cb) cb(addr, res);
    delete this;
}

}; // namespace LFL
