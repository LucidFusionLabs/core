#include "stdregexp.hpp"
#include <cassert>
#include <cstdio>
#include <cwchar>
#include <cstdlib>

using namespace std;

typedef regular_expression::string_type str;
#ifdef REGEXP_UNICODE
#define _T(t) L##t
#define CHART wchar_t
#define CONVERT(a, b) do { mbstowcs(a, b, 255); a[255] = '\0'; } while(false)
#define STRF "%ls"
#else
#define _T(t) t
#define CHART char
#define CONVERT(a, b) strcpy(a, b)
#define STRF "%s"
#endif

bool parse_email(const str& to_match,
                 str& user_name,
				 str& host_and_domain,
				 str& domain_suffix)
{
    regular_expression email_expr(_T("^([A-Za-z0-9]+)@(.+)\\.(\\a+)$"));
    if(!email_expr.exec(to_match))
	    return false;

	// the regular expression's format should ensure that all three
	// expressions match, or the expression doesn't match at all.
	assert(email_expr.matched(1) &&
	       email_expr.matched(2) &&
		   email_expr.matched(3));

	user_name = email_expr[1];
	host_and_domain = email_expr[2];
	domain_suffix = email_expr[3];

	return true;
}

int main(int argc, char** argv)
{
	if(argc >= 2)
	{
		str user_name, host_and_domain, domain_suffix;
		CHART buffer[255];
		CONVERT(buffer, argv[1]);
		if(!parse_email(buffer, user_name, host_and_domain, domain_suffix))
		{
			printf("Not an email address\n");
			return 1;
		}
		printf("User name: " STRF "\nHost/domain: " STRF "\nDomain suffix: " STRF "\n",
			   user_name.c_str(),
			   host_and_domain.c_str(),
			   domain_suffix.c_str());
		return 0;
	}
	
	printf("Usage: %s <email address>\n", argv[0]);
	return 1;
}

