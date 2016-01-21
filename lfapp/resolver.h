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

#ifndef LFL_LFAPP_RESOLVER_H__
#define LFL_LFAPP_RESOLVER_H__
namespace LFL {
    
struct Resolver {
  struct Nameserver;
  typedef map<IPV4::Addr, Nameserver*> NameserverMap;
  typedef function<void(IPV4::Addr, DNS::Response*)> ResponseCB;

  struct Request {
    typedef unsigned short Type;
    Nameserver *ns=0;
    string query;
    Type type;
    Time stamp;
    unsigned retrys=0;
    ResponseCB cb;

    Request() : type(DNS::Type::A), stamp(Now()) {}
    Request(const string &Q, Type T=DNS::Type::A, ResponseCB CB=ResponseCB(), int R=0) : query(Q), type(T), stamp(Now()), retrys(R), cb(CB) {}
    Request(Nameserver *NS, const string &Q, Type T, ResponseCB CB, int R) : ns(NS), query(Q), type(T), stamp(Now()), retrys(R), cb(CB) {}
  };

  struct Nameserver {
    typedef unordered_map<unsigned short, Request> RequestMap;
    RequestMap request_map;
    Resolver *parent=0;
    Connection *c=0;
    bool timedout=0;

    ~Nameserver() { if (c) c->SetError(); }
    Nameserver() {}
    Nameserver(Resolver *P, IPV4::Addr addr) : parent(P),
    c(Singleton<UDPClient>::Get()->PersistentConnection
      (IPV4::Text(addr, 53),
       [&](Connection *c, const char *cb, int cl) { HandleResponse(c, reinterpret_cast<const DNS::Header*>(cb), cl); },
       [&](Connection *c)                         { Heartbeat(); }, 53)) {}

    unsigned short GetNextID() const;
    bool WriteResolveRequest(const Request &req);
    void HandleResponse(Connection *c, const DNS::Header *hdr, int len);
    void Heartbeat();
    void Dequeue();
  };

  vector<Request> queue;
  NameserverMap conn;
  vector<IPV4::Addr> conn_available;
  int max_outstanding_per_ns=10, auto_disconnect_seconds=0;

  bool Connected() const { for (auto &n : conn) if (n.second->c->state == Connection::Connected) return 1; return 0; }
  Nameserver *Connect(IPV4::Addr addr);
  Nameserver *Connect(const vector<IPV4::Addr> &addrs);

  void NSLookup(const string &host, const ResponseCB &cb);
  bool QueueResolveRequest(const Request &req);

  static void GetDefaultNameservers(vector<IPV4::Addr> *nameservers);
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
    typedef unsigned short Type;
    RecursiveResolver *resolver=0;
    string query;
    Type type;
    Resolver::ResponseCB cb;
    vector<DNS::Response> answer;
    bool missing_answer=0;
    Request *parent_request;
    unordered_set<Request*> child_request, pending_child_request;
    unordered_set<AuthorityTreeNode*> seen_authority;

    Request(const string &Q, Type T=DNS::Type::A, Resolver::ResponseCB C=Resolver::ResponseCB(), Request *P=0) :
      query(Q), type(T), cb(C), parent_request(P) {}
    virtual ~Request() { CHECK_EQ(child_request.size(), 0); }

    int Ancestors() const { return parent_request ? 1 + parent_request->Ancestors() : 0; }
    void ResponseCB(IPV4::Addr A, DNS::Response *R) { resolver->HandleRequestResponse(this, A, R, 0); }
    void StartChildResolve(Request *subreq);
    void HandleChildResponse(Request *subreq, DNS::Response *res);
    void Complete(IPV4::Addr addr, DNS::Response *res);
  };

  AuthorityTreeNode root;
  long long queries_requested=0, queries_completed=0;
  RecursiveResolver() { ConnectoToRootServers(); }

  AuthorityTreeNode *GetAuthorityTreeNode(const string &query, bool create);
  void ConnectoToRootServers();
  bool StartResolveRequest(Request *req);
  int ResolveAnyMissingAnswers(Request *req, const vector<DNS::Record> &R, const DNS::AnswerMap *answer);
  void HandleRequestResponse(Request *req, IPV4::Addr addr, DNS::Response *res, vector<DNS::Response> *subres);
};

}; // namespace LFL
#endif // LFL_LFAPP_RESOLVER_H__
