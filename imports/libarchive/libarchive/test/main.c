/*
 * Copyright (c) 2003-2007 Tim Kientzle
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Various utility routines useful for test programs.
 * Each test program is linked against this file.
 */
#include "test.h"

#include <errno.h>
#include <locale.h>
#include <stdarg.h>
#include <time.h>
#if defined(_WIN32) && !defined(__CYGWIN__)
#include <crtdbg.h>
#include <windows.h>
#include <winbase.h>
#endif

/*
 * This same file is used pretty much verbatim for all test harnesses.
 *
 * The next few lines are the only differences.
 */
#undef	PROGRAM              /* Testing a library, not a program. */
#define	LIBRARY	"libarchive"
#define	ENVBASE "LIBARCHIVE" /* Prefix for environment variables. */
#define	EXTRA_DUMP(x)	archive_error_string((struct archive *)(x))
#define	EXTRA_VERSION	archive_version()
#define KNOWNREF	"test_compat_gtar_1.tar.uu"
__FBSDID("$FreeBSD: src/lib/libarchive/test/main.c,v 1.17 2008/12/21 00:13:50 kientzle Exp $");

/*
 * "list.h" is simply created by "grep DEFINE_TEST"; it has
 * a line like
 *      DEFINE_TEST(test_function)
 * for each test.
 * Include it here with a suitable DEFINE_TEST to declare all of the
 * test functions.
 */
#undef DEFINE_TEST
#define	DEFINE_TEST(name) void name(void);
#include "list.h"

/* Interix doesn't define these in a standard header. */
#if __INTERIX__
extern char *optarg;
extern int optind;
#endif

/* Enable core dump on failure. */
static int dump_on_failure = 0;
/* Default is to remove temp dirs for successful tests. */
static int keep_temp_files = 0;
/* Default is to print some basic information about each test. */
static int quiet_flag = 0;
/* Default is to summarize repeated failures. */
static int verbose = 0;
/* Cumulative count of component failures. */
static int failures = 0;
/* Cumulative count of skipped component tests. */
static int skips = 0;
/* Cumulative count of assertions. */
static int assertions = 0;

/* Directory where uuencoded reference files can be found. */
static const char *refdir;


#if defined(_WIN32) && !defined(__CYGWIN__)

static void
invalid_parameter_handler(const wchar_t * expression,
    const wchar_t * function, const wchar_t * file,
    unsigned int line, uintptr_t pReserved)
{
	/* nop */
}

#endif

/*
 * My own implementation of the standard assert() macro emits the
 * message in the same format as GCC (file:line: message).
 * It also includes some additional useful information.
 * This makes it a lot easier to skim through test failures in
 * Emacs.  ;-)
 *
 * It also supports a few special features specifically to simplify
 * test harnesses:
 *    failure(fmt, args) -- Stores a text string that gets
 *          printed if the following assertion fails, good for
 *          explaining subtle tests.
 */
static char msg[4096];

/*
 * For each test source file, we remember how many times each
 * failure was reported.
 */
static const char *failed_filename = NULL;
static struct line {
	int line;
	int count;
	int critical;
}  failed_lines[1000];

/*
 * Called at the beginning of each assert() function.
 */
static void
count_assertion(const char *file, int line)
{
	(void)file; /* UNUSED */
	(void)line; /* UNUSED */
	++assertions;
	/* Uncomment to print file:line after every assertion.
	 * Verbose, but occasionally useful in tracking down crashes. */
	/* printf("Checked %s:%d\n", file, line); */
}

/*
 * Count this failure; return the number of previous failures.
 */
static int
previous_failures(const char *filename, int line, int critical)
{
	unsigned int i;
	int count;

	if (failed_filename == NULL || strcmp(failed_filename, filename) != 0)
		memset(failed_lines, 0, sizeof(failed_lines));
	failed_filename = filename;

	for (i = 0; i < sizeof(failed_lines)/sizeof(failed_lines[0]); i++) {
		if (failed_lines[i].line == line) {
			count = failed_lines[i].count;
			failed_lines[i].count++;
			return (count);
		}
		if (failed_lines[i].line == 0) {
			failed_lines[i].line = line;
			failed_lines[i].count = 1;
			failed_lines[i].critical = critical;
			return (0);
		}
	}
	return (0);
}

/*
 * Copy arguments into file-local variables.
 */
static const char *test_filename;
static int test_line;
static void *test_extra;
void test_setup(const char *filename, int line)
{
	test_filename = filename;
	test_line = line;
}

/*
 * Inform user that we're skipping a test.
 */
void
test_skipping(const char *fmt, ...)
{
	va_list ap;

	if (previous_failures(test_filename, test_line, 0))
		return;

	va_start(ap, fmt);
	fprintf(stderr, " *** SKIPPING: ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
	++skips;
}

/* Common handling of failed tests. */
static void
report_failure(void *extra)
{
	if (msg[0] != '\0') {
		fprintf(stderr, "   Description: %s\n", msg);
		msg[0] = '\0';
	}

#ifdef EXTRA_DUMP
	if (extra != NULL)
		fprintf(stderr, "   detail: %s\n", EXTRA_DUMP(extra));
#else
	(void)extra; /* UNUSED */
#endif

	if (dump_on_failure) {
		fprintf(stderr,
		    " *** forcing core dump so failure can be debugged ***\n");
		*(char *)(NULL) = 0;
		exit(1);
	}
}

/*
 * Summarize repeated failures in the just-completed test file.
 * The reports above suppress multiple failures from the same source
 * line; this reports on any tests that did fail multiple times.
 */
static int
summarize_comparator(const void *a0, const void *b0)
{
	const struct line *a = a0, *b = b0;
	if (a->line == 0 && b->line == 0)
		return (0);
	if (a->line == 0)
		return (1);
	if (b->line == 0)
		return (-1);
	return (a->line - b->line);
}

static void
summarize(void)
{
	unsigned int i;

	qsort(failed_lines, sizeof(failed_lines)/sizeof(failed_lines[0]),
	    sizeof(failed_lines[0]), summarize_comparator);
	for (i = 0; i < sizeof(failed_lines)/sizeof(failed_lines[0]); i++) {
		if (failed_lines[i].line == 0)
			break;
		if (failed_lines[i].count > 1 && failed_lines[i].critical)
			fprintf(stderr, "%s:%d: Failed %d times\n",
			    failed_filename, failed_lines[i].line,
			    failed_lines[i].count);
	}
	/* Clear the failure history for the next file. */
	memset(failed_lines, 0, sizeof(failed_lines));
}

/* Set up a message to display only after a test fails. */
void
failure(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vsprintf(msg, fmt, ap);
	va_end(ap);
}

/* Generic assert() just displays the failed condition. */
int
test_assert(const char *file, int line, int value, const char *condition, void *extra)
{
	count_assertion(file, line);
	if (value) {
		msg[0] = '\0';
		return (value);
	}
	failures ++;
	if (!verbose && previous_failures(file, line, 1))
		return (value);
	fprintf(stderr, "%s:%d: Assertion failed\n", file, line);
	fprintf(stderr, "   Condition: %s\n", condition);
	report_failure(extra);
	return (value);
}

/* assertEqualInt() displays the values of the two integers. */
int
test_assert_equal_int(const char *file, int line,
    int v1, const char *e1, int v2, const char *e2, void *extra)
{
	count_assertion(file, line);
	if (v1 == v2) {
		msg[0] = '\0';
		return (1);
	}
	failures ++;
	if (!verbose && previous_failures(file, line, 1))
		return (0);
	fprintf(stderr, "%s:%d: Assertion failed: Ints not equal\n",
	    file, line);
	fprintf(stderr, "      %s=%d\n", e1, v1);
	fprintf(stderr, "      %s=%d\n", e2, v2);
	report_failure(extra);
	return (0);
}

static void strdump(const char *p)
{
	if (p == NULL) {
		fprintf(stderr, "(null)");
		return;
	}
	fprintf(stderr, "\"");
	while (*p != '\0') {
		unsigned int c = 0xff & *p++;
		switch (c) {
		case '\a': fprintf(stderr, "\a"); break;
		case '\b': fprintf(stderr, "\b"); break;
		case '\n': fprintf(stderr, "\n"); break;
		case '\r': fprintf(stderr, "\r"); break;
		default:
			if (c >= 32 && c < 127)
				fprintf(stderr, "%c", c);
			else
				fprintf(stderr, "\\x%02X", c);
		}
	}
	fprintf(stderr, "\"");
}

/* assertEqualString() displays the values of the two strings. */
int
test_assert_equal_string(const char *file, int line,
    const char *v1, const char *e1,
    const char *v2, const char *e2,
    void *extra)
{
	count_assertion(file, line);
	if (v1 == NULL || v2 == NULL) {
		if (v1 == v2) {
			msg[0] = '\0';
			return (1);
		}
	} else if (strcmp(v1, v2) == 0) {
		msg[0] = '\0';
		return (1);
	}
	failures ++;
	if (!verbose && previous_failures(file, line, 1))
		return (0);
	fprintf(stderr, "%s:%d: Assertion failed: Strings not equal\n",
	    file, line);
	fprintf(stderr, "      %s = ", e1);
	strdump(v1);
	fprintf(stderr, " (length %d)\n", v1 == NULL ? 0 : (int)strlen(v1));
	fprintf(stderr, "      %s = ", e2);
	strdump(v2);
	fprintf(stderr, " (length %d)\n", v2 == NULL ? 0 : (int)strlen(v2));
	report_failure(extra);
	return (0);
}

static void wcsdump(const wchar_t *w)
{
	if (w == NULL) {
		fprintf(stderr, "(null)");
		return;
	}
	fprintf(stderr, "\"");
	while (*w != L'\0') {
		unsigned int c = *w++;
		if (c >= 32 && c < 127)
			fprintf(stderr, "%c", c);
		else if (c < 256)
			fprintf(stderr, "\\x%02X", c);
		else if (c < 0x10000)
			fprintf(stderr, "\\u%04X", c);
		else
			fprintf(stderr, "\\U%08X", c);
	}
	fprintf(stderr, "\"");
}

/* assertEqualWString() displays the values of the two strings. */
int
test_assert_equal_wstring(const char *file, int line,
    const wchar_t *v1, const char *e1,
    const wchar_t *v2, const char *e2,
    void *extra)
{
	count_assertion(file, line);
	if (v1 == NULL) {
		if (v2 == NULL) {
			msg[0] = '\0';
			return (1);
		}
	} else if (v2 == NULL) {
		if (v1 == NULL) {
			msg[0] = '\0';
			return (1);
		}
	} else if (wcscmp(v1, v2) == 0) {
		msg[0] = '\0';
		return (1);
	}
	failures ++;
	if (!verbose && previous_failures(file, line, 1))
		return (0);
	fprintf(stderr, "%s:%d: Assertion failed: Unicode strings not equal\n",
	    file, line);
	fprintf(stderr, "      %s = ", e1);
	wcsdump(v1);
	fprintf(stderr, "\n");
	fprintf(stderr, "      %s = ", e2);
	wcsdump(v2);
	fprintf(stderr, "\n");
	report_failure(extra);
	return (0);
}

/*
 * Pretty standard hexdump routine.  As a bonus, if ref != NULL, then
 * any bytes in p that differ from ref will be highlighted with '_'
 * before and after the hex value.
 */
static void
hexdump(const char *p, const char *ref, size_t l, size_t offset)
{
	size_t i, j;
	char sep;

	for(i=0; i < l; i+=16) {
		fprintf(stderr, "%04x", (unsigned)(i + offset));
		sep = ' ';
		for (j = 0; j < 16 && i + j < l; j++) {
			if (ref != NULL && p[i + j] != ref[i + j])
				sep = '_';
			fprintf(stderr, "%c%02x", sep, 0xff & (int)p[i+j]);
			if (ref != NULL && p[i + j] == ref[i + j])
				sep = ' ';
		}
		for (; j < 16; j++) {
			fprintf(stderr, "%c  ", sep);
			sep = ' ';
		}
		fprintf(stderr, "%c", sep);
		for (j=0; j < 16 && i + j < l; j++) {
			int c = p[i + j];
			if (c >= ' ' && c <= 126)
				fprintf(stderr, "%c", c);
			else
				fprintf(stderr, ".");
		}
		fprintf(stderr, "\n");
	}
}

/* assertEqualMem() displays the values of the two memory blocks. */
/* TODO: For long blocks, hexdump the first bytes that actually differ. */
int
test_assert_equal_mem(const char *file, int line,
    const char *v1, const char *e1,
    const char *v2, const char *e2,
    size_t l, const char *ld, void *extra)
{
	count_assertion(file, line);
	if (v1 == NULL || v2 == NULL) {
		if (v1 == v2) {
			msg[0] = '\0';
			return (1);
		}
	} else if (memcmp(v1, v2, l) == 0) {
		msg[0] = '\0';
		return (1);
	}
	failures ++;
	if (!verbose && previous_failures(file, line, 1))
		return (0);
	fprintf(stderr, "%s:%d: Assertion failed: memory not equal\n",
	    file, line);
	fprintf(stderr, "      size %s = %d\n", ld, (int)l);
	fprintf(stderr, "      Dump of %s\n", e1);
	hexdump(v1, v2, l < 32 ? l : 32, 0);
	fprintf(stderr, "      Dump of %s\n", e2);
	hexdump(v2, v1, l < 32 ? l : 32, 0);
	fprintf(stderr, "\n");
	report_failure(extra);
	return (0);
}

int
test_assert_empty_file(const char *f1fmt, ...)
{
	char buff[1024];
	char f1[1024];
	struct stat st;
	va_list ap;
	ssize_t s;
	int fd;


	va_start(ap, f1fmt);
	vsprintf(f1, f1fmt, ap);
	va_end(ap);

	if (stat(f1, &st) != 0) {
		fprintf(stderr, "%s:%d: Could not stat: %s\n", test_filename, test_line, f1);
		report_failure(NULL);
		return (0);
	}
	if (st.st_size == 0)
		return (1);

	failures ++;
	if (!verbose && previous_failures(test_filename, test_line, 1))
		return (0);

	fprintf(stderr, "%s:%d: File not empty: %s\n", test_filename, test_line, f1);
	fprintf(stderr, "    File size: %d\n", (int)st.st_size);
	fprintf(stderr, "    Contents:\n");
	fd = open(f1, O_RDONLY);
	if (fd < 0) {
		fprintf(stderr, "    Unable to open %s\n", f1);
	} else {
		s = sizeof(buff) < st.st_size ? sizeof(buff) : st.st_size;
		s = read(fd, buff, s);
		hexdump(buff, NULL, s, 0);
	}
	report_failure(NULL);
	return (0);
}

/* assertEqualFile() asserts that two files have the same contents. */
/* TODO: hexdump the first bytes that actually differ. */
int
test_assert_equal_file(const char *f1, const char *f2pattern, ...)
{
	char f2[1024];
	va_list ap;
	char buff1[1024];
	char buff2[1024];
	int fd1, fd2;
	int n1, n2;

	va_start(ap, f2pattern);
	vsprintf(f2, f2pattern, ap);
	va_end(ap);

	fd1 = open(f1, O_RDONLY);
	fd2 = open(f2, O_RDONLY);
	for (;;) {
		n1 = read(fd1, buff1, sizeof(buff1));
		n2 = read(fd2, buff2, sizeof(buff2));
		if (n1 != n2)
			break;
		if (n1 == 0 && n2 == 0)
			return (1);
		if (memcmp(buff1, buff2, n1) != 0)
			break;
	}
	failures ++;
	if (!verbose && previous_failures(test_filename, test_line, 1))
		return (0);
	fprintf(stderr, "%s:%d: Files are not identical\n",
	    test_filename, test_line);
	fprintf(stderr, "  file1=\"%s\"\n", f1);
	fprintf(stderr, "  file2=\"%s\"\n", f2);
	report_failure(test_extra);
	return (0);
}

int
test_assert_file_exists(const char *fpattern, ...)
{
	char f[1024];
	va_list ap;

	va_start(ap, fpattern);
	vsprintf(f, fpattern, ap);
	va_end(ap);

	if (!access(f, F_OK))
		return (1);
	if (!previous_failures(test_filename, test_line, 1)) {
		fprintf(stderr, "%s:%d: File doesn't exist\n",
		    test_filename, test_line);
		fprintf(stderr, "  file=\"%s\"\n", f);
		report_failure(test_extra);
	}
	return (0);
}

int
test_assert_file_not_exists(const char *fpattern, ...)
{
	char f[1024];
	va_list ap;

	va_start(ap, fpattern);
	vsprintf(f, fpattern, ap);
	va_end(ap);

	if (access(f, F_OK))
		return (1);
	if (!previous_failures(test_filename, test_line, 1)) {
		fprintf(stderr, "%s:%d: File exists and shouldn't\n",
		    test_filename, test_line);
		fprintf(stderr, "  file=\"%s\"\n", f);
		report_failure(test_extra);
	}
	return (0);
}

/* assertFileContents() asserts the contents of a file. */
int
test_assert_file_contents(const void *buff, int s, const char *fpattern, ...)
{
	char f[1024];
	va_list ap;
	char *contents;
	int fd;
	int n;

	va_start(ap, fpattern);
	vsprintf(f, fpattern, ap);
	va_end(ap);

	fd = open(f, O_RDONLY);
	contents = malloc(s * 2);
	n = read(fd, contents, s * 2);
	if (n == s && memcmp(buff, contents, s) == 0) {
		free(contents);
		return (1);
	}
	failures ++;
	if (!previous_failures(test_filename, test_line, 1)) {
		fprintf(stderr, "%s:%d: File contents don't match\n",
		    test_filename, test_line);
		fprintf(stderr, "  file=\"%s\"\n", f);
		if (n > 0)
			hexdump(contents, buff, n, 0);
		else {
			fprintf(stderr, "  File empty, contents should be:\n");
			hexdump(buff, NULL, s, 0);
		}
		report_failure(test_extra);
	}
	free(contents);
	return (0);
}

/*
 * Call standard system() call, but build up the command line using
 * sprintf() conventions.
 */
int
systemf(const char *fmt, ...)
{
	char buff[8192];
	va_list ap;
	int r;

	va_start(ap, fmt);
	vsprintf(buff, fmt, ap);
	r = system(buff);
	va_end(ap);
	return (r);
}

/*
 * Slurp a file into memory for ease of comparison and testing.
 * Returns size of file in 'sizep' if non-NULL, null-terminates
 * data in memory for ease of use.
 */
char *
slurpfile(size_t * sizep, const char *fmt, ...)
{
	char filename[8192];
	struct stat st;
	va_list ap;
	char *p;
	ssize_t bytes_read;
	int fd;
	int r;

	va_start(ap, fmt);
	vsprintf(filename, fmt, ap);
	va_end(ap);

	fd = open(filename, O_RDONLY);
	if (fd < 0) {
		/* Note: No error; non-existent file is okay here. */
		return (NULL);
	}
	r = fstat(fd, &st);
	if (r != 0) {
		fprintf(stderr, "Can't stat file %s\n", filename);
		close(fd);
		return (NULL);
	}
	p = malloc(st.st_size + 1);
	if (p == NULL) {
		fprintf(stderr, "Can't allocate %ld bytes of memory to read file %s\n", (long int)st.st_size, filename);
		close(fd);
		return (NULL);
	}
	bytes_read = read(fd, p, st.st_size);
	if (bytes_read < st.st_size) {
		fprintf(stderr, "Can't read file %s\n", filename);
		close(fd);
		free(p);
		return (NULL);
	}
	p[st.st_size] = '\0';
	if (sizep != NULL)
		*sizep = (size_t)st.st_size;
	close(fd);
	return (p);
}

/*
 * "list.h" is automatically generated; it just has a lot of lines like:
 * 	DEFINE_TEST(function_name)
 * It's used above to declare all of the test functions.
 * We reuse it here to define a list of all tests (functions and names).
 */
#undef DEFINE_TEST
#define	DEFINE_TEST(n) { n, #n },
struct { void (*func)(void); const char *name; } tests[] = {
	#include "list.h"
};

/*
 * This is well-intentioned, but sometimes the standard libraries
 * leave open file descriptors and expect to be able to come back to
 * them (e.g., for username lookups or logging).  Closing these
 * descriptors out from under those libraries creates havoc.
 *
 * Maybe there's some reasonably portable way to tell if a descriptor
 * is open without using close()?
 */
#if 0
static void
close_descriptors(int warn)
{
	int i;
	int left_open = 0;

	for (i = 3; i < 100; ++i) {
		if (close(i) == 0)
			++left_open;
	}
	if (warn && left_open > 0) {
		fprintf(stderr, " ** %d descriptors unclosed\n", left_open);
		failures += left_open;
		report_failure(NULL);
	}
}
#endif

/*
 * Each test is run in a private work dir.  Those work dirs
 * do have consistent and predictable names, in case a group
 * of tests need to collaborate.  However, there is no provision
 * for requiring that tests run in a certain order.
 */
static int test_run(int i, const char *tmpdir)
{
	int failures_before = failures;

	if (!quiet_flag) {
		printf("%d: %s\n", i, tests[i].name);
		fflush(stdout);
	}

	/*
	 * Always explicitly chdir() in case the last test moved us to
	 * a strange place.
	 */
	if (chdir(tmpdir)) {
		fprintf(stderr,
		    "ERROR: Couldn't chdir to temp dir %s\n",
		    tmpdir);
		exit(1);
	}
	/* Create a temp directory for this specific test. */
	if (mkdir(tests[i].name, 0755)) {
		fprintf(stderr,
		    "ERROR: Couldn't create temp dir ``%s''\n",
		    tests[i].name);
		exit(1);
	}
	/* Chdir() to that work directory. */
	if (chdir(tests[i].name)) {
		fprintf(stderr,
		    "ERROR: Couldn't chdir to temp dir ``%s''\n",
		    tests[i].name);
		exit(1);
	}
	/* Explicitly reset the locale before each test. */
	setlocale(LC_ALL, "C");
	/* Make sure there are no stray descriptors going into the test. */
	/* TODO: Find a better way to identify file descriptor leaks. */
	//close_descriptors(0);
	/* Run the actual test. */
	(*tests[i].func)();
	/* Close stray descriptors, record as errors against this test. */
	//close_descriptors(1);
	/* Summarize the results of this test. */
	summarize();
	/* If there were no failures, we can remove the work dir. */
	if (failures == failures_before) {
		if (!keep_temp_files && chdir(tmpdir) == 0) {
#if !defined(_WIN32) || defined(__CYGWIN__)
			systemf("rm -rf %s", tests[i].name);
#else
			systemf("rmdir /S /Q %s", tests[i].name);
#endif
		}
	}
	/* Return appropriate status. */
	return (failures == failures_before ? 0 : 1);
}

static void usage(const char *program)
{
	static const int limit = sizeof(tests) / sizeof(tests[0]);
	int i;

	printf("Usage: %s [options] <test> <test> ...\n", program);
	printf("Default is to run all tests.\n");
	printf("Otherwise, specify the numbers of the tests you wish to run.\n");
	printf("Options:\n");
	printf("  -d  Dump core after any failure, for debugging.\n");
	printf("  -k  Keep all temp files.\n");
	printf("      Default: temp files for successful tests deleted.\n");
#ifdef PROGRAM
	printf("  -p <path>  Path to executable to be tested.\n");
	printf("      Default: path taken from " ENVBASE " environment variable.\n");
#endif
	printf("  -q  Quiet.\n");
	printf("  -r <dir>   Path to dir containing reference files.\n");
	printf("      Default: Current directory.\n");
	printf("  -v  Verbose.\n");
	printf("Available tests:\n");
	for (i = 0; i < limit; i++)
		printf("  %d: %s\n", i, tests[i].name);
	exit(1);
}

#define	UUDECODE(c) (((c) - 0x20) & 0x3f)

void
extract_reference_file(const char *name)
{
	char buff[1024];
	FILE *in, *out;

	sprintf(buff, "%s/%s.uu", refdir, name);
	in = fopen(buff, "r");
	failure("Couldn't open reference file %s", buff);
	assert(in != NULL);
	if (in == NULL)
		return;
	/* Read up to and including the 'begin' line. */
	for (;;) {
		if (fgets(buff, sizeof(buff), in) == NULL) {
			/* TODO: This is a failure. */
			return;
		}
		if (memcmp(buff, "begin ", 6) == 0)
			break;
	}
	/* Now, decode the rest and write it. */
	/* Not a lot of error checking here; the input better be right. */
	out = fopen(name, "w");
	while (fgets(buff, sizeof(buff), in) != NULL) {
		char *p = buff;
		int bytes;

		if (memcmp(buff, "end", 3) == 0)
			break;

		bytes = UUDECODE(*p++);
		while (bytes > 0) {
			int n = 0;
			/* Write out 1-3 bytes from that. */
			if (bytes > 0) {
				n = UUDECODE(*p++) << 18;
				n |= UUDECODE(*p++) << 12;
				fputc(n >> 16, out);
				--bytes;
			}
			if (bytes > 0) {
				n |= UUDECODE(*p++) << 6;
				fputc((n >> 8) & 0xFF, out);
				--bytes;
			}
			if (bytes > 0) {
				n |= UUDECODE(*p++);
				fputc(n & 0xFF, out);
				--bytes;
			}
		}
	}
	fclose(out);
	fclose(in);
}


/* Since gzip is by far the most popular external compression program
 * available, we try to use it in the read_program and write_program
 * tests.  But if it's not available, then we can't use it.  This
 * function just tries to run gzip/gunzip to see if they're available.
 * If not, some of the external compression program tests will be
 * skipped. */
const char *
external_gzip_program(int un)
{
	static int tested = 0;
	static const char *compress_prog = NULL;
	static const char *decompress_prog = NULL;
	/* Args vary depending on the command interpreter we're using. */
#if defined(_WIN32) && !defined(__CYGWIN__)
	static const char *args = "-V >NUL 2>NUL"; /* Win32 cmd.exe */
#else
	static const char *args = "-V >/dev/null 2>/dev/null"; /* POSIX 'sh' */
#endif

	if (!tested) {
		if (systemf("gunzip %s", args) == 0)
			decompress_prog = "gunzip";
		if (systemf("gzip %s", args) == 0)
			compress_prog = "gzip";
		tested = 1;
	}
	return (un ? decompress_prog : compress_prog);
}

static char *
get_refdir(void)
{
	char tried[512] = { '\0' };
	char buff[128];
	char *pwd, *p;

	/* Get the current dir. */
	pwd = getcwd(NULL, 0);
	while (pwd[strlen(pwd) - 1] == '\n')
		pwd[strlen(pwd) - 1] = '\0';
	printf("PWD: %s\n", pwd);

	/* Look for a known file. */
	snprintf(buff, sizeof(buff), "%s", pwd);
	p = slurpfile(NULL, "%s/%s", buff, KNOWNREF);
	if (p != NULL) goto success;
	strncat(tried, buff, sizeof(tried) - strlen(tried) - 1);
	strncat(tried, "\n", sizeof(tried) - strlen(tried) - 1);

	snprintf(buff, sizeof(buff), "%s/test", pwd);
	p = slurpfile(NULL, "%s/%s", buff, KNOWNREF);
	if (p != NULL) goto success;
	strncat(tried, buff, sizeof(tried) - strlen(tried) - 1);
	strncat(tried, "\n", sizeof(tried) - strlen(tried) - 1);

	snprintf(buff, sizeof(buff), "%s/%s/test", pwd, LIBRARY);
	p = slurpfile(NULL, "%s/%s", buff, KNOWNREF);
	if (p != NULL) goto success;
	strncat(tried, buff, sizeof(tried) - strlen(tried) - 1);
	strncat(tried, "\n", sizeof(tried) - strlen(tried) - 1);

	if (memcmp(pwd, "/usr/obj", 8) == 0) {
		snprintf(buff, sizeof(buff), "%s", pwd + 8);
		p = slurpfile(NULL, "%s/%s", buff, KNOWNREF);
		if (p != NULL) goto success;
		strncat(tried, buff, sizeof(tried) - strlen(tried) - 1);
		strncat(tried, "\n", sizeof(tried) - strlen(tried) - 1);

		snprintf(buff, sizeof(buff), "%s/test", pwd + 8);
		p = slurpfile(NULL, "%s/%s", buff, KNOWNREF);
		if (p != NULL) goto success;
		strncat(tried, buff, sizeof(tried) - strlen(tried) - 1);
		strncat(tried, "\n", sizeof(tried) - strlen(tried) - 1);
	}

#if defined(_WIN32) && !defined(__CYGWIN__) && defined(_DEBUG)
	DebugBreak();
#endif
	printf("Unable to locate known reference file %s\n", KNOWNREF);
	printf("  Checked following directories:\n%s\n", tried);
	exit(1);

success:
	free(p);
	free(pwd);
	return strdup(buff);
}

int main(int argc, char **argv)
{
	static const int limit = sizeof(tests) / sizeof(tests[0]);
	int i, tests_run = 0, tests_failed = 0, option;
	time_t now;
	char *refdir_alloc = NULL;
	const char *progname = LIBRARY "_test";
	const char *tmp, *option_arg, *p;
	char tmpdir[256];
	char tmpdir_timestamp[256];

	(void)argc; /* UNUSED */

#if defined(_WIN32) && !defined(__CYGWIN__)
	/* To stop to run the default invalid parameter handler. */
	_set_invalid_parameter_handler(invalid_parameter_handler);
	/* for open() to a binary mode. */
	_set_fmode(_O_BINARY);
	/* Disable annoying assertion message box. */
	_CrtSetReportMode(_CRT_ASSERT, 0);
#endif

#ifdef PROGRAM
	/* Get the target program from environment, if available. */
	testprog = getenv(ENVBASE);
#endif

	if (getenv("TMPDIR") != NULL)
		tmp = getenv("TMPDIR");
	else if (getenv("TMP") != NULL)
		tmp = getenv("TMP");
	else if (getenv("TEMP") != NULL)
		tmp = getenv("TEMP");
	else if (getenv("TEMPDIR") != NULL)
		tmp = getenv("TEMPDIR");
	else
		tmp = "/tmp";

	/* Allow -d to be controlled through the environment. */
	if (getenv(ENVBASE "_DEBUG") != NULL)
		dump_on_failure = 1;

	/* Get the directory holding test files from environment. */
	refdir = getenv(ENVBASE "_TEST_FILES");

	/*
	 * Parse options, without using getopt(), which isn't available
	 * on all platforms.
	 */
	++argv; /* Skip program name */
	while (*argv != NULL) {
		if (**argv != '-')
			break;
		p = *argv++;
		++p; /* Skip '-' */
		while (*p != '\0') {
			option = *p++;
			option_arg = NULL;
			/* If 'opt' takes an argument, parse that. */
			if (option == 'p' || option == 'r') {
				if (*p != '\0')
					option_arg = p;
				else if (*argv == NULL) {
					fprintf(stderr,
					    "Option -%c requires argument.\n",
					    option);
					usage(progname);
				} else
					option_arg = *argv++;
				p = ""; /* End of this option word. */
			}

			/* Now, handle the option. */
			switch (option) {
			case 'd':
				dump_on_failure = 1;
				break;
			case 'k':
				keep_temp_files = 1;
				break;
			case 'p':
#ifdef PROGRAM
				testprog = option_arg;
#else
				usage(progname);
#endif
				break;
			case 'q':
				quiet_flag++;
				break;
			case 'r':
				refdir = option_arg;
				break;
			case 'v':
				verbose = 1;
				break;
			default:
				usage(progname);
			}
		}
	}

	/*
	 * Sanity-check that our options make sense.
	 */
#ifdef PROGRAM
	if (testprog == NULL)
		usage(progname);
#endif

	/*
	 * Create a temp directory for the following tests.
	 * Include the time the tests started as part of the name,
	 * to make it easier to track the results of multiple tests.
	 */
	now = time(NULL);
	for (i = 0; i < 1000; i++) {
		strftime(tmpdir_timestamp, sizeof(tmpdir_timestamp),
		    "%Y-%m-%dT%H.%M.%S",
		    localtime(&now));
		sprintf(tmpdir, "%s/%s.%s-%03d", tmp, progname,
		    tmpdir_timestamp, i);
		if (mkdir(tmpdir,0755) == 0)
			break;
		if (errno == EEXIST)
			continue;
		fprintf(stderr, "ERROR: Unable to create temp directory %s\n",
		    tmpdir);
		exit(1);
	}

	/*
	 * If the user didn't specify a directory for locating
	 * reference files, try to find the reference files in
	 * the "usual places."
	 */
	if (refdir == NULL)
		refdir = refdir_alloc = get_refdir();

	/*
	 * Banner with basic information.
	 */
	if (!quiet_flag) {
		printf("Running tests in: %s\n", tmpdir);
		printf("Reference files will be read from: %s\n", refdir);
#ifdef PROGRAM
		printf("Running tests on: %s\n", testprog);
#endif
		printf("Exercising: ");
		fflush(stdout);
		printf("%s\n", EXTRA_VERSION);
	}

	/*
	 * Run some or all of the individual tests.
	 */
	if (*argv == NULL) {
		/* Default: Run all tests. */
		for (i = 0; i < limit; i++) {
			if (test_run(i, tmpdir))
				tests_failed++;
			tests_run++;
		}
	} else {
		while (*(argv) != NULL) {
			if (**argv >= '0' && **argv <= '9') {
				i = atoi(*argv);
				if (i < 0 || i >= limit) {
					printf("*** INVALID Test %s\n", *argv);
					free(refdir_alloc);
					usage(progname);
					/* usage() never returns */
				}
			} else {
				for (i = 0; i < limit; ++i) {
					if (strcmp(*argv, tests[i].name) == 0)
						break;
				}
				if (i >= limit) {
					printf("*** INVALID Test ``%s''\n",
					       *argv);
					free(refdir_alloc);
					usage(progname);
					/* usage() never returns */
				}
			}
			if (test_run(i, tmpdir))
				tests_failed++;
			tests_run++;
			argv++;
		}
	}

	/*
	 * Report summary statistics.
	 */
	if (!quiet_flag) {
		printf("\n");
		printf("%d of %d tests reported failures\n",
		    tests_failed, tests_run);
		printf(" Total of %d assertions checked.\n", assertions);
		printf(" Total of %d assertions failed.\n", failures);
		printf(" Total of %d reported skips.\n", skips);
	}

	free(refdir_alloc);

	/* If the final tmpdir is empty, we can remove it. */
	/* This should be the usual case when all tests succeed. */
	chdir("..");
	rmdir(tmpdir);

	return (tests_failed);
}
