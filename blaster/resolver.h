/*
 * $Id: resolver.h 1336 2014-12-08 09:29:59Z justin $
 */

#ifndef __LFL_BLASTER_RESOLVER_H__
#define __LFL_BLASTER_RESOLVER_H__
namespace LFL {

struct ResolvedMX {
    int pref; string host; vector<IPV4::Addr> A;
    ResolvedMX(int P, const string &H, const string &AText) : pref(P), host(H) { IPV4::ParseCSV(AText, &A); }
    string DebugString() const { return StrCat("MX", pref, "=", host, ":", IPV4::MakeCSV(A)); }
};

static void ParseResolverOutput(const char *line, int linelen, vector<ResolvedMX> *A, vector<ResolvedMX> *mx) {
    int host_i = 0;
    StringWordIter hosts(line, linelen);
    for (string h = hosts.Next(); !hosts.Done(); h = hosts.Next(), host_i++) {
        StringWordIter args(h, isint4<'=', ':', ',', ';'>);
        string type = IterNextString(&args);
        string host = IterNextString(&args);
        if (!host_i) {
            CHECK_EQ(type, "A");
            A->push_back(ResolvedMX(0, host, IterNextString(&args)));
        } else {
            CHECK(PrefixMatch(type, "MX"));
            mx->push_back(ResolvedMX(atoi(type.c_str()+2), host, IterNextString(&args)));
        }
    }
}

}; // namespace LFL
#endif // __LFL_BLASTER_RESOLVER_H__
