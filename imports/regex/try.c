/*
 * Simple test program for regexp(3) stuff.  Knows about debugging hooks.
 * Usage: try re [string [output [-]]]
 * The re is compiled and dumped, re_execed against the string, the result
 * is applied to output using re_sub().  The - triggers a running narrative
 * from re_exec().  Dumping and narrative don't happen unless DEBUG.
 *
 * If there are no arguments, stdin is assumed to be a stream of lines with
 * five fields:  a r.e., a string to match it against, a result code, a
 * source string for re_sub, and the proper result.  Result codes are 'c'
 * for compile failure, 'y' for match success, 'n' for match failure.
 * Field separator is tab.
 */
#include <stdio.h>
#include "regexp.h"

#ifdef ERRAVAIL
char *progname;
extern char *mkprogname();
#endif

#ifdef DEBUG
extern int regnarrate;
#endif

int errreport = 0;		/* Report errors via errseen? */
int status = 0;			/* Exit status. */

/* ARGSUSED */
main(argc, argv)
int argc;
char *argv[];
{
	regexp *r;
	int i;
    regmatch match[10];

#ifdef ERRAVAIL
	progname = mkprogname(argv[0]);
#endif

	if (argc == 1) {
		multiple();
		exit(status);
	}

	i = re_comp(&r, argv[1]);
	if (i < 0)
		error("re_comp failure", "");
	if (argc > 2) {
		i = re_exec(r, argv[2], 10, match);
		printf("%d", i);
		for (i = 1; i < 10; i++)
			if (match[i].begin != -1 && match[i].end != -1)
				printf(" \\%d", i);
		printf("\n");
	}
	if (argc > 3) {
        char* buf;
		i = re_sub(r, argv[2], argv[3], &buf);
        if(i < 0)
            error("re_sub failure", "");
		printf("%s\n", buf);
        re_free(buf);
	}
    re_free(r);
	exit(status);
}

error(s1, s2)
char *s1;
char *s2;
{
	fprintf(stderr, "regexp: ");
	fprintf(stderr, s1, s2);
	fprintf(stderr, "\n");
	exit(1);
}

int lineno;

unsigned char junk[1024];		/* Implicit init to 0. */

regexp* badregexp = (regexp*)&junk;

multiple()
{
	char rbuf[BUFSIZ];
	char *field[5];
	char *scan;
	int i;
	regexp *r;
	extern char *strchr();

	errreport = 1;
	lineno = 0;
	while (fgets(rbuf, sizeof(rbuf), stdin) != NULL) {
		rbuf[strlen(rbuf)-1] = '\0';	/* Dispense with \n. */
		lineno++;
		scan = rbuf;
		for (i = 0; i < 5; i++) {
			field[i] = scan;
			if (field[i] == NULL) {
				complain("bad testfile format", "", 0);
				exit(1);
			}
			scan = strchr(scan, '\t');
			if (scan != NULL)
				*scan++ = '\0';
		}
		try(field);
	}

	/* And finish up with some internal testing... */
	lineno = 9990;
	if ((i = re_comp(NULL, NULL)) >= 0)
		complain("re_comp(NULL, NULL) doesn't complain", "", i);
	lineno = 9991;
	if ((i = re_exec(NULL, "foo", 0, NULL)) >= 0)
		complain("re_exec(NULL, ...) doesn't complain", "", i);
	lineno = 9992;
    r = NULL;
	
	if ((i = re_comp(&r, "foo")) < 0) {
		complain("re_comp(\"foo\") fails", "", i);
		return;
	}
	lineno = 9993;
	if ((i = re_exec(r, NULL, 0, NULL)) >= 0)
		complain("re_exec(..., NULL, ...) doesn't complain", "", i);
	lineno = 9994;
	if ((i = re_sub(NULL, "foo", "foo", &scan)) >= 0)
		complain("re_sub(NULL, ..., ...) doesn't complain", "", i);
	lineno = 9995;
	if ((i = re_sub(r, "foo", NULL, &scan)) >= 0)
		complain("re_sub(..., NULL, ...) doesn't complain", "", i);
	lineno = 9996;
	if ((i = re_sub(r, "foo", "foo", NULL)) >= 0)
		complain("re_sub(..., ..., NULL) doesn't complain", "", i);
	lineno = 9997;
    if ((i = re_exec(badregexp, "foo", 0, NULL)) >= 0)
        complain("re_exec(nonsense, ...) doesn't complain", "", i);
	lineno = 9998;
    if ((i = re_sub(badregexp, "foo", "foo", &scan)) >= 0)
		complain("re_sub(nonsense, ..., ...) doesn't complain", "", i);
}

fixstr(p)
register char* p;
{
    if(p == NULL)
        return;
    for(; *p != 0; ++p)
        if(*p == 'N')
            *p = '\n';
        else if(*p == 'T')
            *p = '\t';
        else if(*p == 'S')
            *p = ' ';
        else if(*p == 'Z')
            *p = '\0';
}

try(fields)
char **fields;
{
	regexp *r;
	char* dbuf;
    int i;

    r = NULL;
	i = re_comp(&r, fields[0]);
	if (i < 0) {
		if (*fields[2] != 'c')
			complain("re_comp failure in `%s'", fields[0], i);
		return;
	}
	if (*fields[2] == 'c') {
		complain("unexpected re_comp success in `%s'", fields[0], i);
        re_free(r);
		return;
	}
    fixstr(fields[1]);
	if ((i = re_exec(r, fields[1], 0, NULL)) < 0) {
		if (*fields[2] != 'n')
			complain("re_exec failure in `%s'", fields[0], i);
		re_free(r);
		return;
	}
	if (*fields[2] == 'n' && i > 0) {
		complain("unexpected re_exec success", "", i);
        re_free(r);
		return;
	}
	i = re_sub(r, fields[1], fields[3], &dbuf);
	if (i < 0) {
		complain("re_sub complaint", "", i);
		re_free(r);
		return;
	}
    fixstr(fields[4]);
	if (dbuf && strcmp(dbuf, fields[4]) != 0)
		complain("re_sub result `%s' wrong", dbuf);
    re_free(dbuf);
	re_free(r);
}

complain(s1, s2, err)
char *s1;
char *s2;
int err;
{
    char buffer[80];
    if(err < 0)
        re_error(err, NULL, buffer, sizeof(buffer));
    
	fprintf(stderr, "try: %d: ", lineno);
	fprintf(stderr, s1, s2);
	fprintf(stderr, " (%s)\n", (err < 0) ? buffer : "");
	status = 1;
}

/* hook error */
re_report(s)
    const char* s;
{
}
