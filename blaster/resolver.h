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
    StringWordIter hosts(line, linelen); int host_i = 0;
    for (const char *h = hosts.Next(); h; h = hosts.Next(), host_i++) {
        StringWordIter args(h, 0, isint4<'=', ':', ',', ';'>);
        string type = BlankNull(args.Next());
        string host = BlankNull(args.Next());
        if (!host_i) {
            CHECK_EQ(type, "A");
            A->push_back(ResolvedMX(0, host, BlankNull(args.Next())));
        } else {
            CHECK(PrefixMatch(type, "MX"));
            mx->push_back(ResolvedMX(atoi(type.c_str()+2), host, BlankNull(args.Next())));
        }
    }
}

}; // namespace LFL
#endif // __LFL_BLASTER_RESOLVER_H__
