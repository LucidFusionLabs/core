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

#ifndef __LFL_LFAPP_RESOLVER_H__
#define __LFL_LFAPP_RESOLVER_H__
namespace LFL {
    
struct Resolver {
    struct Nameserver;
    typedef map<IPV4::Addr, Nameserver*> NameserverMap;
    typedef function<void(IPV4::Addr, DNS::Response*)> ResponseCB;

    struct Request {
        Nameserver *ns;
        string query;
        unsigned short type;
        ResponseCB cb;
        Time stamp;
        unsigned retrys;

        Request() : ns(0), type(DNS::Type::A), stamp(Now()), retrys(0) {}
        Request(const string &Q, unsigned short T=DNS::Type::A, ResponseCB CB=ResponseCB(), int R=0) : ns(0), query(Q), type(T), cb(CB), stamp(Now()), retrys(R) {}
        Request(Nameserver *NS, const string &Q, unsigned short T, ResponseCB CB, int R) : ns(NS), type(T), query(Q), cb(CB), stamp(Now()), retrys(R) {}
    };

    struct Nameserver {
        Resolver *parent=0;
        Connection *c=0;
        bool timedout=0;
        typedef unordered_map<unsigned short, Request> RequestMap;
        RequestMap requestMap;

        ~Nameserver() { if (c) c->SetError(); }
        Nameserver() {}
        Nameserver(Resolver *P, IPV4::Addr addr) : parent(P),
        c(Singleton<UDPClient>::Get()->PersistentConnection
          (IPV4::Text(addr, 53),
           [&](Connection *c, const char *cb, int cl) { Response(c, (DNS::Header*)cb, cl); },
           [&](Connection *c)                         { Heartbeat(); }, 53)) {}

        unsigned short NextID() const { unsigned short id; for (id = rand(); Contains(requestMap, id); id = rand()) { /**/ } return id; }
        bool Resolve(const Request &req);
        void Response(Connection *c, DNS::Header *hdr, int len);
        void Heartbeat();
        void Dequeue();
    };

    vector<Request> queue;
    NameserverMap conn;
    vector<IPV4::Addr> conn_available;
    int max_outstanding_per_ns=10, auto_disconnect_seconds=0;

    void Reset();
    bool Connected();
    Nameserver *Connect(IPV4::Addr addr);
    Nameserver *Connect(const vector<IPV4::Addr> &addrs);
    bool Resolve(const Request &req);

    static void DefaultNameserver(vector<IPV4::Addr> *nameservers);
    static void UDPClientHeartbeatCB(Connection *c, void *a) { ((Nameserver*)a)->Heartbeat(); }
};

struct RecursiveResolver {
    struct AuthorityTreeNode {
        typedef map<string, AuthorityTreeNode*> Children;
        typedef map<string, DNS::Response*> Cache;
        int depth=0;
        Children child;
        Resolver resolver;
        Cache Acache, MXcache;
        DNS::Response authority;
        string authority_domain;
        AuthorityTreeNode() { resolver.auto_disconnect_seconds=10; }
    };
    struct Request {
        RecursiveResolver *resolver;
        string query;
        unsigned short type;
        Resolver::ResponseCB cb;
        bool missing_answer;
        vector<DNS::Response> answer;
        Request *parent_request;
        set<Request*> child_request, pending_child_request;
        set<void*> seen_authority;

        Request(const string &Q, unsigned short T=DNS::Type::A, Resolver::ResponseCB C=Resolver::ResponseCB(), Request *P=0) : resolver(0), query(Q), type(T), cb(C), missing_answer(0), parent_request(P) {}
        virtual ~Request() { CHECK_EQ(child_request.size(), 0); }

        int Ancestors() const { return parent_request ? 1 + parent_request->Ancestors() : 0; }
        void ChildResolve(Request *subreq);
        void ChildResponse(Request *subreq, DNS::Response *res);
        void Complete(IPV4::Addr addr, DNS::Response *res);
        void ResponseCB(IPV4::Addr A, DNS::Response *R) { resolver->Response(this, A, R, 0); }
    };

    AuthorityTreeNode root;
    long long queries_requested, queries_completed;
    RecursiveResolver();

    AuthorityTreeNode *GetAuthorityTreeNode(const string &query, bool create);
    bool Resolve(Request *req);
    int ResolveMissing(Request *req, const vector<DNS::Record> &R, const DNS::AnswerMap *answer);
    void Response(Request *req, IPV4::Addr addr, DNS::Response *res, vector<DNS::Response> *subres);
};

}; // namespace LFL
#endif // __LFL_LFAPP_RESOLVER_H__
